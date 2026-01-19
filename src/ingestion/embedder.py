"""
Embedding generation for 10-K filing chunks.

Wraps the Ollama client with batching and progress tracking.
"""

import logging
from typing import Iterator

from src.config import get_settings
from src.llm.factory import OllamaClient, get_ollama_client, OllamaError

logger = logging.getLogger(__name__)


def batched(iterable: list, n: int) -> Iterator[list]:
    """
    Batch an iterable into chunks of size n.

    Args:
        iterable: List to batch.
        n: Batch size.

    Yields:
        Batches of size n (last batch may be smaller).
    """
    length = len(iterable)
    for i in range(0, length, n):
        yield iterable[i : i + n]


class Embedder:
    """
    Generate embeddings for text chunks using Ollama.

    Handles batching, retries, and progress tracking.
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        batch_size: int | None = None,
    ):
        """
        Initialize embedder.

        Args:
            client: Ollama client instance.
            batch_size: Number of texts to embed per API call.
        """
        settings = get_settings()
        self._client = client
        self._batch_size = batch_size or settings.embedding_batch_size

    @property
    def client(self) -> OllamaClient:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = get_ollama_client()
        return self._client

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Processes in batches for efficiency and to avoid timeouts.

        Args:
            texts: List of texts to embed.
            show_progress: Whether to log progress.

        Returns:
            list[list[float]]: List of embedding vectors.

        Raises:
            OllamaError: If embedding generation fails.
        """
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for batch_idx, batch in enumerate(batched(texts, self._batch_size)):
            if show_progress:
                logger.info(
                    f"Embedding batch {batch_idx + 1}/{total_batches} "
                    f"({len(batch)} texts)"
                )

            try:
                batch_embeddings = self.client.embed_batch(batch)
                all_embeddings.extend(batch_embeddings)

            except OllamaError as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")

                if len(batch) > 1:
                    logger.info("Retrying batch with individual requests...")
                    for text in batch:
                        try:
                            embedding = self.client.embed(text)
                            all_embeddings.append(embedding)
                        except OllamaError as e2:
                            logger.error(f"Individual embedding failed: {e2}")
                            raise
                else:
                    raise

        if len(all_embeddings) != len(texts):
            raise OllamaError(
                f"Expected {len(texts)} embeddings, got {len(all_embeddings)}"
            )

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            list[float]: Embedding vector.
        """
        return self.client.embed(text)

    def check_health(self) -> bool:
        """
        Check if Ollama embedding service is healthy.

        Returns:
            bool: True if service is available and model is loaded.
        """
        try:
            if not self.client.health_check():
                return False

            test_embedding = self.client.embed("test")
            settings = get_settings()

            if len(test_embedding) != settings.embedding_dimension:
                logger.warning(
                    f"Embedding dimension mismatch: got {len(test_embedding)}, "
                    f"expected {settings.embedding_dimension}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Embedding health check failed: {e}")
            return False


def get_embedder() -> Embedder:
    """
    Get embedder instance.

    Returns:
        Embedder: Configured embedder.
    """
    return Embedder()
