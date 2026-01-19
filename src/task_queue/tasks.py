"""
rq task definitions for 10-K filing processing.

Defines the FilingTask dataclass and the main process_filing task.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class FilingTask:
    """Task data for processing a 10-K filing."""

    file_path: str
    ticker: str
    year: int
    cik: str
    accession_number: str

    def to_dict(self) -> dict:
        """Convert to dictionary for Redis serialization."""
        return {
            "file_path": self.file_path,
            "ticker": self.ticker,
            "year": self.year,
            "cik": self.cik,
            "accession_number": self.accession_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FilingTask":
        """Create FilingTask from dictionary."""
        return cls(
            file_path=data["file_path"],
            ticker=data["ticker"],
            year=int(data["year"]),
            cik=data["cik"],
            accession_number=data["accession_number"],
        )

    def validate(self) -> None:
        """
        Validate task data.

        Raises:
            ValueError: If validation fails.
        """
        if not self.file_path:
            raise ValueError("file_path is required")
        if not Path(self.file_path).exists():
            raise ValueError(f"File not found: {self.file_path}")
        if not self.ticker:
            raise ValueError("ticker is required")
        if not self.accession_number:
            raise ValueError("accession_number is required")


@dataclass
class ProcessingResult:
    """Result of processing a filing."""

    success: bool
    accession_number: str
    chunks_indexed: int = 0
    error_message: str | None = None
    processing_time_seconds: float = 0.0


def process_filing(task_data: dict) -> ProcessingResult:
    """
    Process a 10-K filing: parse, chunk, embed, and store in Qdrant.

    This is the main rq task function. It orchestrates the full pipeline:
    1. Validate task data
    2. Check for duplicates in Qdrant
    3. Extract metadata from filing
    4. Parse HTML and extract clean text
    5. Chunk text with section detection
    6. Generate embeddings via Ollama
    7. Store in Qdrant

    Args:
        task_data: Dictionary with task fields (file_path, ticker, year, cik, accession_number).

    Returns:
        ProcessingResult: Result with success status and metrics.
    """
    import time

    start_time = time.time()
    task = FilingTask.from_dict(task_data)

    log = logger.bind(
        ticker=task.ticker,
        accession_number=task.accession_number,
        file_path=task.file_path,
    )

    try:
        task.validate()
        log.info("processing_started")

        from src.ingestion.pipeline import IndexingPipeline

        pipeline = IndexingPipeline()

        if pipeline.is_already_indexed(task.accession_number):
            log.info("already_indexed_skipping")
            return ProcessingResult(
                success=True,
                accession_number=task.accession_number,
                chunks_indexed=0,
            )

        chunks_indexed = pipeline.process_filing(task)

        elapsed = time.time() - start_time
        log.info(
            "processing_completed",
            chunks_indexed=chunks_indexed,
            duration_seconds=round(elapsed, 2),
        )

        return ProcessingResult(
            success=True,
            accession_number=task.accession_number,
            chunks_indexed=chunks_indexed,
            processing_time_seconds=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        log.error("processing_failed", error=str(e), duration_seconds=round(elapsed, 2))
        return ProcessingResult(
            success=False,
            accession_number=task.accession_number,
            error_message=str(e),
            processing_time_seconds=elapsed,
        )


def enqueue_filing(task: FilingTask) -> str:
    """
    Enqueue a filing task for processing.

    Args:
        task: FilingTask to enqueue.

    Returns:
        str: Job ID.
    """
    from src.task_queue.connection import get_queue

    queue = get_queue()
    job = queue.enqueue(
        process_filing,
        task.to_dict(),
        job_timeout=get_settings().request_timeout,
        result_ttl=86400,
        failure_ttl=86400 * 7,
    )
    logger.info(
        "task_enqueued",
        job_id=job.id,
        ticker=task.ticker,
        accession_number=task.accession_number,
    )
    return job.id
