#!/usr/bin/env python3
"""
Force re-index SEC 10-K filings.

Supports:
- Re-indexing all files
- Re-indexing specific ticker(s)
- Re-indexing only unindexed files
- Direct processing or queue-based processing
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def reindex_all(
    force: bool = False,
    use_queue: bool = False,
    workers: int = 1,
) -> tuple[int, int]:
    """
    Re-index all filings found on disk.

    Args:
        force: If True, reindex even if already indexed.
        use_queue: If True, push to Redis queue instead of direct processing.
        workers: Number of parallel workers (1=sequential).

    Returns:
        tuple[int, int]: (filings_processed, total_chunks). For queue mode, total_chunks is 0.
    """
    from src.ingestion.pipeline import (
        scan_raw_10k_directory,
        get_unindexed_filings,
    )
    from src.queue.tasks import FilingTask, enqueue_filing

    if force:
        filings = scan_raw_10k_directory()
        logger.info(f"Force reindex: found {len(filings)} filings")
    else:
        filings = get_unindexed_filings()
        logger.info(f"Found {len(filings)} unindexed filings")

    if not filings:
        logger.info("No filings to process")
        return 0, 0

    # Queue mode: push to Redis (chunks counted later by worker)
    if use_queue:
        processed = 0
        for filing in filings:
            try:
                task = FilingTask(
                    file_path=filing["file_path"],
                    ticker=filing["ticker"],
                    year=0,
                    cik="",
                    accession_number=filing["accession_number"],
                )
                enqueue_filing(task)
                processed += 1
            except Exception as e:
                logger.error(f"Failed to queue {filing['ticker']}: {e}")
        return processed, 0

    # Direct processing: sequential or parallel
    if workers <= 1:
        return _process_sequential(filings, force)
    else:
        return _process_parallel(filings, force, workers)


def _process_sequential(filings: list[dict], force: bool) -> tuple[int, int]:
    """Process filings sequentially (original behavior).

    Returns:
        tuple[int, int]: (processed_count, total_chunks)
    """
    from src.ingestion.pipeline import IndexingPipeline
    from src.queue.tasks import FilingTask

    pipeline = IndexingPipeline()
    processed = 0
    total_chunks = 0

    for filing in filings:
        try:
            task = FilingTask(
                file_path=filing["file_path"],
                ticker=filing["ticker"],
                year=0,
                cik="",
                accession_number=filing["accession_number"],
            )

            if not force and pipeline.is_already_indexed(filing["accession_number"]):
                logger.info(f"Skipping {filing['ticker']} - already indexed")
                continue

            chunks = pipeline.process_filing(task)
            total_chunks += chunks
            logger.info(
                f"Indexed {filing['ticker']} ({filing['accession_number']}): "
                f"{chunks} chunks"
            )
            processed += 1
        except Exception as e:
            logger.error(f"Failed to process {filing['ticker']}: {e}")

    logger.info(f"Sequential processing complete: {processed} filings, {total_chunks} total chunks stored")
    return processed, total_chunks


def _process_parallel(filings: list[dict], force: bool, workers: int) -> tuple[int, int]:
    """Process filings in parallel using ThreadPoolExecutor.

    Returns:
        tuple[int, int]: (succeeded_count, total_chunks)
    """
    from src.ingestion.parallel import process_filings_parallel, ParallelProgress

    logger.info(f"Starting parallel processing with {workers} workers")

    total_chunks = 0

    def progress_callback(progress: ParallelProgress, result) -> None:
        """Log progress updates."""
        nonlocal total_chunks
        pct = (progress.processed / progress.total) * 100
        logger.info(
            f"Progress: {progress.completed}/{progress.total} completed "
            f"({pct:.1f}%), {progress.failed} failed"
        )
        if result.success and result.chunks_indexed > 0:
            total_chunks += result.chunks_indexed
            logger.info(
                f"Indexed {result.ticker} ({result.accession_number}): "
                f"{result.chunks_indexed} chunks in {result.duration_seconds:.2f}s"
            )
        elif not result.success:
            logger.error(
                f"Failed {result.ticker} ({result.accession_number}): "
                f"{result.error_message}"
            )

    succeeded, failed = process_filings_parallel(
        filings=filings,
        max_workers=workers,
        force=force,
        on_progress=progress_callback,
    )

    logger.info(f"Parallel processing complete: {succeeded} succeeded, {failed} failed, {total_chunks} total chunks stored")
    return succeeded, total_chunks


def reindex_ticker(
    ticker: str,
    year: int | None = None,
    force: bool = False,
    use_queue: bool = False,
    workers: int = 1,
) -> tuple[int, int]:
    """
    Re-index filings for a specific ticker.

    Args:
        ticker: Stock ticker symbol.
        year: Specific year (optional).
        force: If True, reindex even if already indexed.
        use_queue: If True, push to Redis queue.
        workers: Number of parallel workers (1=sequential).

    Returns:
        tuple[int, int]: (filings_processed, total_chunks). For queue mode, total_chunks is 0.
    """
    from src.config import get_settings
    from src.ingestion.metadata import extract_accession_from_path
    from src.queue.tasks import FilingTask, enqueue_filing

    settings = get_settings()
    base_dir = Path(settings.sec_output_dir) / "sec-edgar-filings" / ticker.upper() / "10-K"

    if not base_dir.exists():
        logger.error(f"No filings found for {ticker} at {base_dir}")
        return 0, 0

    html_files = list(base_dir.glob("*/primary-document.html"))

    if year:
        # Filter by year if specified (check accession number year)
        filtered = []
        for f in html_files:
            accession = extract_accession_from_path(str(f))
            if accession:
                # Accession format: CIK-YY-NNNNNN, extract year
                parts = accession.split("-")
                if len(parts) >= 2:
                    acc_year = int("20" + parts[1]) if int(parts[1]) < 50 else int("19" + parts[1])
                    if acc_year == year:
                        filtered.append(f)
        html_files = filtered

    if not html_files:
        logger.warning(f"No filings found for {ticker}" + (f" year {year}" if year else ""))
        return 0, 0

    logger.info(f"Found {len(html_files)} filings for {ticker}")

    # Build filings list for processing
    filings = []
    for html_file in html_files:
        accession = extract_accession_from_path(str(html_file))
        if accession:
            filings.append({
                "file_path": str(html_file),
                "ticker": ticker.upper(),
                "accession_number": accession,
            })

    if not filings:
        return 0, 0

    # Queue mode: push to Redis (chunks counted later by worker)
    if use_queue:
        processed = 0
        for filing in filings:
            try:
                task = FilingTask(
                    file_path=filing["file_path"],
                    ticker=filing["ticker"],
                    year=year or 0,
                    cik="",
                    accession_number=filing["accession_number"],
                )
                enqueue_filing(task)
                logger.info(f"Queued {ticker} ({filing['accession_number']})")
                processed += 1
            except Exception as e:
                logger.error(f"Failed to queue {ticker} ({filing['accession_number']}): {e}")
        return processed, 0

    # Direct processing: sequential or parallel
    if workers <= 1:
        return _process_sequential(filings, force)
    else:
        return _process_parallel(filings, force, workers)


def show_status() -> None:
    """Show indexing status."""
    from src.ingestion.pipeline import scan_raw_10k_directory
    from src.store.qdrant_local import get_qdrant_store

    store = get_qdrant_store()
    info = store.get_collection_info()

    filings_on_disk = scan_raw_10k_directory()
    indexed_accessions = store.get_all_accession_numbers()

    print("\n=== Indexing Status ===")
    print(f"Collection: {info.get('name', 'N/A')}")
    print(f"Total chunks: {info.get('points_count', 0)}")
    print(f"Status: {info.get('status', 'N/A')}")
    print(f"\nFilings on disk: {len(filings_on_disk)}")
    print(f"Filings indexed: {len(indexed_accessions)}")
    print(f"Filings pending: {len(filings_on_disk) - len(indexed_accessions)}")

    # Show breakdown by ticker
    tickers = {}
    for filing in filings_on_disk:
        t = filing["ticker"]
        if t not in tickers:
            tickers[t] = {"on_disk": 0, "indexed": 0}
        tickers[t]["on_disk"] += 1
        if filing["accession_number"] in indexed_accessions:
            tickers[t]["indexed"] += 1

    if tickers:
        print("\n=== By Ticker ===")
        for ticker in sorted(tickers.keys()):
            data = tickers[ticker]
            status = "complete" if data["on_disk"] == data["indexed"] else "partial"
            print(f"{ticker}: {data['indexed']}/{data['on_disk']} ({status})")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Re-index SEC 10-K filings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show indexing status
  python reindex.py --status

  # Index all unindexed files directly
  python reindex.py --all

  # Force reindex all files (even if already indexed)
  python reindex.py --all --force

  # Parallel indexing with 4 workers
  python reindex.py --all --workers 4

  # Queue all unindexed files for worker processing
  python reindex.py --all --queue

  # Index specific ticker
  python reindex.py --ticker AAPL

  # Index specific ticker and year with parallel processing
  python reindex.py --ticker AAPL --year 2023 --force --workers 4
        """,
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show indexing status and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all filings on disk",
    )
    parser.add_argument(
        "--ticker",
        help="Process specific ticker",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Filter by year (used with --ticker)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex even if already indexed",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Push tasks to Redis queue instead of direct processing",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers (1=sequential, 2-16=parallel)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.status:
        show_status()
        return 0

    if not args.all and not args.ticker:
        parser.error("Must specify --all or --ticker")
        return 1

    try:
        if args.all:
            processed, total_chunks = reindex_all(
                force=args.force,
                use_queue=args.queue,
                workers=args.workers,
            )
        else:
            processed, total_chunks = reindex_ticker(
                ticker=args.ticker,
                year=args.year,
                force=args.force,
                use_queue=args.queue,
                workers=args.workers,
            )

        action = "queued" if args.queue else "processed"
        if args.queue:
            logger.info(f"Complete: {processed} filings {action}")
        else:
            logger.info(f"Complete: {processed} filings {action}, {total_chunks} total chunks stored")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
