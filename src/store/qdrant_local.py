"""
Qdrant vector store client for local mode operation.

Provides collection management, upsert, scroll, and delete operations
for storing and retrieving 10-K filing chunks.
"""

import hashlib
import logging
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkPayload:
    """Payload data for a 10-K chunk stored in Qdrant."""

    # Required per CLAUDE.md
    cik: str
    ticker: str
    company_name: str
    filing_date: str
    section_title: str
    chunk_id: str
    source_path: str

    # Additional metadata
    accession_number: str
    fiscal_year: int
    chunk_index: int
    total_chunks: int
    text: str

    def to_dict(self) -> dict:
        """Convert to dictionary for Qdrant payload."""
        return asdict(self)


class QdrantStore:
    """Qdrant vector store client for 10-K chunks."""

    def __init__(self, client: QdrantClient | None = None):
        """
        Initialize Qdrant store.

        Args:
            client: Optional QdrantClient instance. If not provided,
                    creates a new client using settings.
        """
        self._client = client
        self._settings = get_settings()

    @property
    def client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> QdrantClient:
        """Create Qdrant client in server mode."""
        if not self._settings.qdrant_url:
            raise ValueError(
                "QDRANT_URL is required. Set it in .env or environment. "
                "Example: QDRANT_URL=http://localhost:6333"
            )
        client = QdrantClient(url=self._settings.qdrant_url)
        logger.info(f"Initialized Qdrant client at {self._settings.qdrant_url}")
        return client

    @property
    def collection_name(self) -> str:
        """Get collection name from settings."""
        return self._settings.qdrant_collection_name

    def ensure_collection(self) -> None:
        """
        Ensure collection exists with proper configuration.

        Creates the collection if it doesn't exist, with proper
        vector configuration and payload indexes.
        """
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            logger.info(f"Collection '{self.collection_name}' already exists")
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self._settings.embedding_dimension,
                distance=distance_map[self._settings.qdrant_distance_metric],
                on_disk=True,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
            ),
        )
        logger.info(
            f"Created collection '{self.collection_name}' "
            f"(dim={self._settings.embedding_dimension}, "
            f"distance={self._settings.qdrant_distance_metric})"
        )

        self._create_indexes()

    def _create_indexes(self) -> None:
        """Create payload indexes for filtering and hybrid search."""
        keyword_fields = ["ticker", "accession_number", "section_title", "cik"]
        integer_fields = ["fiscal_year"]

        for field in keyword_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.debug(f"Created keyword index on '{field}'")
            except UnexpectedResponse:
                pass

        for field in integer_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.INTEGER,
                )
                logger.debug(f"Created integer index on '{field}'")
            except UnexpectedResponse:
                pass

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="text",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                ),
            )
            logger.debug("Created full-text index on 'text'")
        except UnexpectedResponse:
            pass

    def generate_point_id(self, accession_number: str, chunk_index: int) -> str:
        """
        Generate deterministic point ID for idempotent upserts.

        Args:
            accession_number: SEC accession number.
            chunk_index: Chunk position in document.

        Returns:
            str: Deterministic UUID-like string.
        """
        key = f"{accession_number}:{chunk_index}"
        return hashlib.md5(key.encode()).hexdigest()

    def is_indexed(self, accession_number: str) -> bool:
        """
        Check if a filing is already indexed.

        Args:
            accession_number: SEC accession number to check.

        Returns:
            bool: True if any chunks exist for this accession number.
        """
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="accession_number",
                            match=MatchValue(value=accession_number),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(result[0]) > 0
        except Exception as e:
            logger.warning(f"Error checking index status: {e}")
            return False

    def delete_by_accession(self, accession_number: str) -> int:
        """
        Delete all chunks for a filing.

        Args:
            accession_number: SEC accession number.

        Returns:
            int: Number of points deleted.
        """
        try:
            points = list(self._scroll_accession(accession_number))
            if not points:
                return 0

            point_ids = [p.id for p in points]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
            logger.info(
                f"Deleted {len(point_ids)} chunks for accession {accession_number}"
            )
            return len(point_ids)
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise

    def _scroll_accession(
        self, accession_number: str
    ) -> Iterator[models.Record]:
        """
        Scroll through all points for an accession number.

        Args:
            accession_number: SEC accession number.

        Yields:
            Record: Qdrant point records.
        """
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="accession_number",
                            match=MatchValue(value=accession_number),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            yield from results
            if offset is None:
                break

    def upsert_chunks(
        self,
        chunks: list[ChunkPayload],
        embeddings: list[list[float]],
    ) -> int:
        """
        Upsert chunks with embeddings to Qdrant.

        Uses batch upsert for efficiency. Point IDs are deterministic
        based on accession_number and chunk_index for idempotency.

        Args:
            chunks: List of chunk payloads.
            embeddings: Corresponding embedding vectors.

        Returns:
            int: Number of points upserted.

        Raises:
            ValueError: If chunks and embeddings length mismatch.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "count mismatch"
            )

        if not chunks:
            return 0

        accession_number = chunks[0].accession_number
        self.delete_by_accession(accession_number)

        points = [
            PointStruct(
                id=self.generate_point_id(chunk.accession_number, chunk.chunk_index),
                vector=embedding,
                payload=chunk.to_dict(),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        batch_size = self._settings.qdrant_batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            logger.debug(
                f"Upserted batch {i // batch_size + 1} "
                f"({len(batch)} points) for {accession_number}"
            )

        logger.info(
            f"Upserted {len(points)} chunks for accession {accession_number}"
        )
        return len(points)

    def get_all_accession_numbers(self) -> set[str]:
        """
        Get all unique accession numbers in the collection.

        Used for startup recovery to identify already-indexed filings.

        Returns:
            set[str]: Set of accession numbers.
        """
        accession_numbers = set()
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=["accession_number"],
                with_vectors=False,
            )

            for record in results:
                if record.payload and "accession_number" in record.payload:
                    accession_numbers.add(record.payload["accession_number"])

            if offset is None:
                break

        logger.info(f"Found {len(accession_numbers)} indexed filings")
        return accession_numbers

    def get_collection_info(self) -> dict:
        """
        Get collection statistics.

        Returns:
            dict: Collection info including point count, vector config, etc.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
                "status": info.status.value if hasattr(info.status, "value") else str(info.status),
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}


@lru_cache
def get_qdrant_store() -> QdrantStore:
    """
    Get cached Qdrant store instance.

    Returns:
        QdrantStore: Configured Qdrant store.
    """
    store = QdrantStore()
    store.ensure_collection()
    return store
