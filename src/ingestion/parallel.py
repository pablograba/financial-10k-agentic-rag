"""
Parallel processing utilities for 10-K ingestion.

Provides thread-safe wrappers and parallel execution helpers
using ThreadPoolExecutor for I/O-bound parallelism.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import structlog

from src.queue.tasks import FilingTask

logger = structlog.get_logger(__name__)


@dataclass
class ParallelProgress:
    """Thread-safe progress tracking for parallel operations."""

    total: int
    _completed: int = field(default=0, init=False, repr=False)
    _failed: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def mark_completed(self) -> None:
        """Mark one task as completed."""
        with self._lock:
            self._completed += 1

    def mark_failed(self) -> None:
        """Mark one task as failed."""
        with self._lock:
            self._failed += 1

    @property
    def completed(self) -> int:
        """Get count of completed tasks."""
        with self._lock:
            return self._completed

    @property
    def failed(self) -> int:
        """Get count of failed tasks."""
        with self._lock:
            return self._failed

    @property
    def remaining(self) -> int:
        """Get count of remaining tasks."""
        with self._lock:
            return self.total - self._completed - self._failed

    @property
    def processed(self) -> int:
        """Get count of processed tasks (completed + failed)."""
        with self._lock:
            return self._completed + self._failed


@dataclass
class ProcessingResult:
    """Result of processing a single filing."""

    accession_number: str
    ticker: str
    success: bool
    chunks_indexed: int = 0
    error_message: str | None = None
    duration_seconds: float = 0.0


def process_filings_parallel(
    filings: list[dict],
    max_workers: int = 4,
    force: bool = False,
    on_progress: Callable[[ParallelProgress, ProcessingResult], None] | None = None,
) -> tuple[int, int]:
    """
    Process multiple filings in parallel using ThreadPoolExecutor.

    Uses Qdrant server mode for thread-safe writes. The server handles
    concurrency internally, so no application-level locking is needed.

    Args:
        filings: List of filing dicts with file_path, ticker, accession_number.
        max_workers: Number of parallel workers.
        force: If True, reindex even if already indexed.
        on_progress: Optional callback for progress updates.

    Returns:
        tuple[int, int]: (succeeded_count, failed_count)
    """
    from src.ingestion.embedder import Embedder
    from src.ingestion.pipeline import IndexingPipeline
    from src.llm.factory import OllamaClient
    from src.store.qdrant_local import get_qdrant_store

    if not filings:
        return (0, 0)

    progress = ParallelProgress(total=len(filings))

    # Create shared clients (thread-safe for Qdrant server mode and httpx)
    shared_store = get_qdrant_store()
    shared_ollama = OllamaClient()

    logger.info(
        "parallel_processing_started",
        total_filings=len(filings),
        max_workers=max_workers,
        force=force,
    )

    def worker(filing: dict) -> ProcessingResult:
        """Worker function for processing a single filing."""
        start_time = time.time()
        accession = filing["accession_number"]
        ticker = filing["ticker"]

        try:
            # Create per-worker pipeline with shared clients
            embedder = Embedder(client=shared_ollama)
            pipeline = IndexingPipeline(
                store=shared_store,
                embedder=embedder,
            )

            # Check if already indexed (unless forcing)
            if not force and pipeline.is_already_indexed(accession):
                return ProcessingResult(
                    accession_number=accession,
                    ticker=ticker,
                    success=True,
                    chunks_indexed=0,
                    duration_seconds=time.time() - start_time,
                )

            # Create task and process
            task = FilingTask(
                file_path=filing["file_path"],
                ticker=ticker,
                year=0,
                cik="",
                accession_number=accession,
            )

            chunks = pipeline.process_filing(task)

            return ProcessingResult(
                accession_number=accession,
                ticker=ticker,
                success=True,
                chunks_indexed=chunks,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(
                "filing_processing_failed",
                accession_number=accession,
                ticker=ticker,
                error=str(e),
            )
            return ProcessingResult(
                accession_number=accession,
                ticker=ticker,
                success=False,
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )

    # Execute in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filing = {
            executor.submit(worker, filing): filing for filing in filings
        }

        for future in as_completed(future_to_filing):
            try:
                result = future.result()

                if result.success:
                    progress.mark_completed()
                else:
                    progress.mark_failed()

                if on_progress:
                    on_progress(progress, result)

            except Exception as e:
                progress.mark_failed()
                filing = future_to_filing[future]
                logger.exception(
                    "worker_exception",
                    filing=filing,
                    error=str(e),
                )

    # Cleanup shared Ollama client
    shared_ollama.close()

    logger.info(
        "parallel_processing_completed",
        succeeded=progress.completed,
        failed=progress.failed,
    )

    return (progress.completed, progress.failed)
