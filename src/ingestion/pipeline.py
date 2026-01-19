"""
Ingestion pipeline for SEC 10-K filings.

Orchestrates the full processing flow:
1. Extract metadata
2. Parse HTML
3. Chunk text with section detection
4. Generate embeddings
5. Store in Qdrant
"""

import logging
import time
from pathlib import Path

import structlog

from src.config import get_settings
from src.ingestion.chunker import Chunker, TextChunk
from src.ingestion.embedder import Embedder
from src.ingestion.html_parser import HTMLParser
from src.ingestion.metadata import FilingMetadata, MetadataExtractor
from src.task_queue.tasks import FilingTask
from src.store.qdrant_local import ChunkPayload, QdrantStore, get_qdrant_store

logger = structlog.get_logger(__name__)


class IndexingPipeline:
    """
    Production-grade indexing pipeline for 10-K filings.

    Coordinates HTML parsing, chunking, embedding, and storage.
    Handles errors gracefully with detailed logging.
    """

    def __init__(
        self,
        store: QdrantStore | None = None,
        embedder: Embedder | None = None,
        parser: HTMLParser | None = None,
        chunker: Chunker | None = None,
        metadata_extractor: MetadataExtractor | None = None,
    ):
        """
        Initialize pipeline with optional dependency injection.

        For parallel processing, pass pre-created shared instances:
        - store: QdrantStore with server mode (thread-safe via HTTP API)
        - embedder: Embedder with shared OllamaClient (httpx is thread-safe)

        Args:
            store: Qdrant store instance (shared for parallel processing).
            embedder: Embedder instance (shared for parallel processing).
            parser: HTML parser instance.
            chunker: Text chunker instance.
            metadata_extractor: Metadata extractor instance.
        """
        self._store = store
        self._embedder = embedder
        self._parser = parser or HTMLParser()
        self._chunker = chunker or Chunker()
        self._metadata_extractor = metadata_extractor or MetadataExtractor()
        self._settings = get_settings()

    @property
    def store(self) -> QdrantStore:
        """Get or create Qdrant store."""
        if self._store is None:
            self._store = get_qdrant_store()
        return self._store

    @property
    def embedder(self) -> Embedder:
        """Get or create embedder."""
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def is_already_indexed(self, accession_number: str) -> bool:
        """
        Check if a filing is already indexed.

        Args:
            accession_number: SEC accession number.

        Returns:
            bool: True if already indexed.
        """
        return self.store.is_indexed(accession_number)

    def process_filing(self, task: FilingTask) -> int:
        """
        Process a single filing through the full pipeline.

        Args:
            task: Filing task with file path and metadata.

        Returns:
            int: Number of chunks indexed.

        Raises:
            Exception: If processing fails.
        """
        log = logger.bind(
            ticker=task.ticker,
            accession_number=task.accession_number,
        )
        start_time = time.time()

        log.info("pipeline_started", file_path=task.file_path)

        metadata = self._extract_metadata(task)
        log.info(
            "metadata_extracted",
            company_name=metadata.company_name,
            filing_date=metadata.filing_date,
        )

        text = self._parse_html(task.file_path)
        log.info("html_parsed", text_length=len(text))

        chunks = self._chunk_text(text)
        log.info("text_chunked", chunk_count=len(chunks))

        if not chunks:
            log.warning("no_chunks_created")
            return 0

        embeddings = self._generate_embeddings(chunks)
        log.info("embeddings_generated", embedding_count=len(embeddings))

        payloads = self._build_payloads(chunks, metadata)

        chunks_indexed = self._store_chunks(payloads, embeddings)

        elapsed = time.time() - start_time
        log.info(
            "pipeline_completed",
            chunks_indexed=chunks_indexed,
            duration_seconds=round(elapsed, 2),
        )

        return chunks_indexed

    def _extract_metadata(self, task: FilingTask) -> FilingMetadata:
        """Extract metadata from filing."""
        return self._metadata_extractor.extract(
            file_path=task.file_path,
            ticker=task.ticker,
            fallback_year=task.year,
        )

    def _parse_html(self, file_path: str) -> str:
        """Parse HTML and extract clean text."""
        return self._parser.parse_file(file_path)

    def _chunk_text(self, text: str) -> list[TextChunk]:
        """Chunk text with section detection."""
        return self._chunker.chunk_text(text)

    def _generate_embeddings(self, chunks: list[TextChunk]) -> list[list[float]]:
        """Generate embeddings for chunks."""
        texts = [chunk.text for chunk in chunks]
        return self.embedder.embed_texts(texts)

    def _build_payloads(
        self,
        chunks: list[TextChunk],
        metadata: FilingMetadata,
    ) -> list[ChunkPayload]:
        """Build Qdrant payloads from chunks and metadata."""
        total_chunks = len(chunks)

        payloads = []
        for chunk in chunks:
            chunk_id = f"{metadata.accession_number}:{chunk.chunk_index}"
            payload = ChunkPayload(
                cik=metadata.cik,
                ticker=metadata.ticker,
                company_name=metadata.company_name,
                filing_date=metadata.filing_date,
                section_title=chunk.section_title,
                chunk_id=chunk_id,
                source_path=metadata.source_path,
                accession_number=metadata.accession_number,
                fiscal_year=metadata.fiscal_year,
                chunk_index=chunk.chunk_index,
                total_chunks=total_chunks,
                text=chunk.text,
            )
            payloads.append(payload)

        return payloads

    def _store_chunks(
        self,
        payloads: list[ChunkPayload],
        embeddings: list[list[float]],
    ) -> int:
        """Store chunks in Qdrant."""
        return self.store.upsert_chunks(payloads, embeddings)

    def process_file(
        self,
        file_path: str | Path,
        ticker: str | None = None,
        year: int | None = None,
        force: bool = False,
    ) -> int:
        """
        Convenience method to process a file directly.

        Creates a FilingTask from file path and processes it.

        Args:
            file_path: Path to HTML file.
            ticker: Ticker symbol (extracted from path if not provided).
            year: Filing year (extracted from metadata if not provided).
            force: If True, reindex even if already indexed.

        Returns:
            int: Number of chunks indexed.
        """
        from src.ingestion.metadata import (
            extract_accession_from_path,
            extract_ticker_from_path,
        )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        accession = extract_accession_from_path(str(file_path)) or "unknown"
        ticker = ticker or extract_ticker_from_path(str(file_path)) or "UNKNOWN"

        if not force and self.is_already_indexed(accession):
            logger.info(
                "already_indexed_skipping",
                accession_number=accession,
            )
            return 0

        task = FilingTask(
            file_path=str(file_path),
            ticker=ticker,
            year=year or 0,
            cik="",
            accession_number=accession,
        )

        return self.process_filing(task)


def scan_raw_10k_directory(base_dir: str | None = None) -> list[dict]:
    """
    Scan raw_10k directory and return all filing info.

    Used for startup recovery and reindexing.

    Args:
        base_dir: Base directory to scan (defaults to settings).

    Returns:
        list[dict]: List of filing info dicts with file_path, ticker, accession_number.
    """
    from src.ingestion.metadata import (
        extract_accession_from_path,
        extract_ticker_from_path,
    )

    settings = get_settings()
    base_dir = Path(base_dir or settings.sec_output_dir)

    if not base_dir.exists():
        logger.warning(f"Directory not found: {base_dir}")
        return []

    filings = []
    for html_file in base_dir.glob("**/primary-document.html"):
        accession = extract_accession_from_path(str(html_file))
        ticker = extract_ticker_from_path(str(html_file))

        if accession and ticker:
            filings.append({
                "file_path": str(html_file),
                "ticker": ticker,
                "accession_number": accession,
            })

    logger.info(f"Found {len(filings)} filings in {base_dir}")
    return filings


def get_unindexed_filings(base_dir: str | None = None) -> list[dict]:
    """
    Get filings that exist on disk but are not indexed.

    Args:
        base_dir: Base directory to scan.

    Returns:
        list[dict]: List of unindexed filing info.
    """
    store = get_qdrant_store()

    all_filings = scan_raw_10k_directory(base_dir)
    indexed_accessions = store.get_all_accession_numbers()

    unindexed = [
        f for f in all_filings if f["accession_number"] not in indexed_accessions
    ]

    logger.info(
        f"Found {len(unindexed)} unindexed filings "
        f"(of {len(all_filings)} total, {len(indexed_accessions)} indexed)"
    )
    return unindexed
