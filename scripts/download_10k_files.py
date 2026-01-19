#!/usr/bin/env python3
"""
Download SEC 10-K filings from EDGAR with robust error handling and retry logic.

Supports single ticker/year downloads or bulk CSV processing.
Implements exponential backoff for transient failures and respects SEC fair access policy.

After successful downloads, pushes indexing tasks to Redis queue.
"""

import argparse
import csv
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sec_edgar_downloader import Downloader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0  # seconds
MAX_BACKOFF = 32.0  # seconds
BACKOFF_MULTIPLIER = 2.0


TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")
MIN_YEAR = 1994  # EDGAR electronic filings began
MAX_YEAR = 2030  # reasonable future limit
MAX_TICKER_LENGTH = 10


class DownloadError(Exception):
    """Base exception for download failures."""

    pass


class ValidationError(Exception):
    """Input validation failed."""

    pass


class RateLimiter:
    """Thread-safe rate limiter ensuring minimum interval between operations."""

    def __init__(self, min_interval: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between operations (default: 1.0)
        """
        self._min_interval = min_interval
        self._last_access = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until sufficient time has passed since last operation."""
        with self._lock:
            now = time.time()
            time_since_last = now - self._last_access

            if time_since_last < self._min_interval:
                time.sleep(self._min_interval - time_since_last)

            self._last_access = time.time()


class ProgressTracker:
    """Thread-safe counters for download statistics."""

    def __init__(self):
        """Initialize progress counters."""
        self._succeeded = 0
        self._failed = 0
        self._lock = threading.Lock()

    def increment_success(self) -> None:
        """Increment success counter (thread-safe)."""
        with self._lock:
            self._succeeded += 1

    def increment_failure(self) -> None:
        """Increment failure counter (thread-safe)."""
        with self._lock:
            self._failed += 1

    @property
    def succeeded(self) -> int:
        """Get current success count (thread-safe)."""
        with self._lock:
            return self._succeeded

    @property
    def failed(self) -> int:
        """Get current failure count (thread-safe)."""
        with self._lock:
            return self._failed


def validate_ticker(ticker: str) -> str:
    """
    Validate and sanitize ticker symbol.

    Args:
        ticker: Raw ticker input

    Returns:
        Uppercase sanitized ticker

    Raises:
        ValidationError: If ticker is invalid
    """
    if not ticker:
        raise ValidationError("Ticker cannot be empty")

    sanitized = ticker.strip().upper()

    if len(sanitized) > MAX_TICKER_LENGTH:
        raise ValidationError(f"Ticker too long: {sanitized} (max {MAX_TICKER_LENGTH} chars)")

    if not TICKER_PATTERN.match(sanitized):
        raise ValidationError(f"Invalid ticker format: {sanitized} (must be 1-5 uppercase letters)")

    return sanitized


def validate_year(year: int) -> int:
    """
    Validate filing year.

    Args:
        year: Filing year

    Returns:
        Validated year

    Raises:
        ValidationError: If year is out of valid range
    """
    if not isinstance(year, int):
        raise ValidationError(f"Year must be an integer, got {type(year)}")

    if year < MIN_YEAR or year > MAX_YEAR:
        raise ValidationError(f"Year {year} out of valid range ({MIN_YEAR}-{MAX_YEAR})")

    return year


def find_downloaded_file(output_dir: str, ticker: str) -> Path | None:
    """
    Find the most recently downloaded primary-document.html for a ticker.

    Args:
        output_dir: Base output directory
        ticker: Stock ticker symbol

    Returns:
        Path to primary-document.html or None if not found
    """
    base = Path(output_dir) / "sec-edgar-filings" / ticker / "10-K"
    if not base.exists():
        return None

    html_files = list(base.glob("*/primary-document.html"))
    if not html_files:
        return None

    return max(html_files, key=lambda p: p.stat().st_mtime)


def extract_accession_from_path(file_path: Path) -> str | None:
    """Extract accession number from file path."""
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part == "10-K" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def push_to_queue(
    file_path: Path,
    ticker: str,
    year: int,
    redis_enabled: bool = True,
) -> bool:
    """
    Push indexing task to Redis queue.

    Args:
        file_path: Path to downloaded HTML file
        ticker: Stock ticker symbol
        year: Filing year
        redis_enabled: Whether to actually push to Redis

    Returns:
        True if successfully pushed, False otherwise
    """
    if not redis_enabled:
        return True

    try:
        from src.task_queue.tasks import FilingTask, enqueue_filing

        accession = extract_accession_from_path(file_path)
        if not accession:
            logger.warning(f"Could not extract accession number from {file_path}")
            return False

        cik = accession.split("-")[0] if accession else ""

        task = FilingTask(
            file_path=str(file_path),
            ticker=ticker,
            year=year,
            cik=cik,
            accession_number=accession,
        )

        job_id = enqueue_filing(task)
        logger.info(f"Queued indexing task for {ticker} ({year}): job_id={job_id}")
        return True

    except ImportError:
        logger.debug("Redis queue not available, skipping task push")
        return True
    except Exception as e:
        logger.warning(f"Failed to push task to queue: {e}")
        return False


def _download_worker(
    task: tuple[str, int, int],
    company_name: str,
    email: str,
    output_dir: str,
    rate_limiter: RateLimiter,
    progress: ProgressTracker,
    redis_enabled: bool = True,
) -> bool:
    """
    Worker function for parallel downloads.

    Creates thread-local Downloader instance for thread safety.
    The sec-edgar-downloader library is not thread-safe, so each worker
    must create its own instance.

    Args:
        task: Tuple of (ticker, year, row_num)
        company_name: SEC user-agent company name
        email: SEC user-agent email
        output_dir: Download destination directory
        rate_limiter: Shared rate limiter instance
        progress: Shared progress tracker instance
        redis_enabled: Whether to push tasks to Redis queue

    Returns:
        True if download succeeded, False otherwise
    """
    ticker, year, row_num = task

    try:
        validated_ticker = validate_ticker(ticker)
        validated_year = validate_year(year)

        downloader = Downloader(
            company_name=company_name,
            email_address=email,
            download_folder=output_dir,
        )

        rate_limiter.acquire()

        success = download_single_with_retry(downloader, validated_ticker, validated_year)

        if success:
            progress.increment_success()

            # Push indexing task to Redis queue
            downloaded_file = find_downloaded_file(output_dir, validated_ticker)
            if downloaded_file:
                push_to_queue(downloaded_file, validated_ticker, validated_year, redis_enabled)
            else:
                logger.warning(f"Could not find downloaded file for {validated_ticker}")
        else:
            progress.increment_failure()

        return success

    except (ValidationError, ValueError) as e:
        logger.error(f"Row {row_num}: {e}")
        progress.increment_failure()
        return False
    except Exception as e:
        logger.exception(f"Unexpected error for {ticker} ({year}): {e}")
        progress.increment_failure()
        return False


def download_single_with_retry(
    downloader: Downloader, ticker: str, year: int, max_retries: int = MAX_RETRIES
) -> bool:
    """
    Download 10-K for a single ticker/year with exponential backoff retry.

    Args:
        downloader: SEC Edgar downloader instance
        ticker: Stock ticker symbol (validated)
        year: Filing year (validated)
        max_retries: Maximum retry attempts

    Returns:
        True if download succeeded, False otherwise
    """
    after_date = f"{year}-01-01"
    before_date = f"{year + 1}-01-01"

    backoff = INITIAL_BACKOFF
    download_start_time = time.time()

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Downloading 10-K for {ticker} (year {year}) [attempt {attempt + 1}/{max_retries}]"
            )

            downloader.get(
                "10-K",
                ticker,
                after=after_date,
                before=before_date,
                download_details=True,
            )

            elapsed = time.time() - download_start_time
            logger.info(f"✓ Successfully downloaded {ticker} ({year}) in {elapsed:.2f}s")
            return True

        except Exception as e:
            error_msg = str(e).lower()

            # Determine if error is retryable
            is_retryable = any(
                keyword in error_msg
                for keyword in [
                    "timeout",
                    "connection",
                    "network",
                    "503",
                    "502",
                    "429",
                    "temporary",
                ]
            )

            # Non-retryable errors (e.g., ticker not found, no filings)
            is_not_found = any(
                keyword in error_msg for keyword in ["no filings", "not found", "404"]
            )

            if is_not_found:
                logger.warning(f"✗ No 10-K found for {ticker} ({year}): {e}")
                return False

            if attempt < max_retries - 1 and is_retryable:
                logger.warning(
                    f"Download failed for {ticker} ({year}), retrying in {backoff:.1f}s: {e}"
                )
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            else:
                logger.error(f"✗ Failed to download {ticker} ({year}) after {attempt + 1} attempts: {e}")
                return False

    return False


def download_single(downloader: Downloader, ticker: str, year: int) -> None:
    """
    Download 10-K for a single ticker/year with validation.

    Args:
        downloader: SEC Edgar downloader instance
        ticker: Stock ticker symbol (will be validated)
        year: Filing year (will be validated)

    Raises:
        ValidationError: If inputs are invalid
    """
    try:
        validated_ticker = validate_ticker(ticker)
        validated_year = validate_year(year)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise

    success = download_single_with_retry(downloader, validated_ticker, validated_year)
    if not success:
        raise DownloadError(f"Failed to download {validated_ticker} ({validated_year})")


def download_from_csv(
    downloader: Downloader,
    csv_path: str,
    max_workers: int = 4,
    rate_limit_interval: float = 1.0,
    company_name: str | None = None,
    email: str | None = None,
    output_dir: str | None = None,
    redis_enabled: bool = True,
) -> None:
    """
    Read CSV and download 10-Ks using parallel workers.

    CSV format: ticker,year (with or without header)

    Args:
        downloader: SEC Edgar downloader instance (for metadata extraction)
        csv_path: Path to CSV file
        max_workers: Number of parallel workers (default: 4)
        rate_limit_interval: Minimum seconds between requests (default: 1.0)
        company_name: SEC user-agent company name (extracted from downloader if None)
        email: SEC user-agent email (extracted from downloader if None)
        output_dir: Download directory (extracted from downloader if None)
        redis_enabled: Whether to push tasks to Redis queue after download

    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValidationError: If CSV format is invalid
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not csv_file.is_file():
        raise ValidationError(f"Path is not a file: {csv_path}")

    logger.info(f"Processing CSV: {csv_path}")
    logger.info(f"Using {max_workers} parallel workers with {rate_limit_interval}s rate limit")

    overall_start_time = time.time()

    
    if company_name is None:
        company_name = downloader.company_name
    if email is None:
        email = downloader.email_address
    if output_dir is None:
        output_dir = str(downloader.download_folder)

    
    tasks: list[tuple[str, int, int]] = []  # List of (ticker, year, row_num)

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            # Detect if CSV has header
            sample = f.read(1024)
            f.seek(0)
            has_header = csv.Sniffer().has_header(sample)

            if has_header:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, start=2):  
                    ticker_raw = row.get("ticker", "").strip()
                    year_raw = row.get("year", "").strip()

                    if not ticker_raw or not year_raw:
                        logger.warning(f"Row {row_num}: missing ticker or year, skipping")
                        continue

                    try:
                        ticker = validate_ticker(ticker_raw)
                        year = validate_year(int(year_raw))
                        tasks.append((ticker, year, row_num))
                    except (ValidationError, ValueError) as e:
                        logger.error(f"Row {row_num}: {e}")
                        continue

            else:
                
                reader = csv.reader(f)
                for row_num, row in enumerate(reader, start=1):
                    if len(row) < 2:
                        logger.warning(f"Row {row_num}: insufficient columns, skipping")
                        continue

                    ticker_raw = row[0].strip()
                    year_raw = row[1].strip()

                    try:
                        ticker = validate_ticker(ticker_raw)
                        year = validate_year(int(year_raw))
                        tasks.append((ticker, year, row_num))
                    except (ValidationError, ValueError) as e:
                        logger.error(f"Row {row_num}: {e}")
                        continue

    except csv.Error as e:
        raise ValidationError(f"CSV parsing error: {e}")

    if not tasks:
        logger.warning("No valid tasks found in CSV")
        return

    logger.info(f"Loaded {len(tasks)} valid download tasks from CSV")

    
    rate_limiter = RateLimiter(min_interval=rate_limit_interval)
    progress = ProgressTracker()

    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _download_worker,
                task,
                company_name,
                email,
                output_dir,
                rate_limiter,
                progress,
                redis_enabled,
            ): task
            for task in tasks
        }

        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            ticker, year, row_num = task

            try:
                
                future.result()
            except Exception as e:
                logger.exception(f"Worker exception for {ticker} ({year}): {e}")

    # Log final summary with timing
    overall_elapsed = time.time() - overall_start_time
    logger.info(
        f"\nSummary: {progress.succeeded}/{len(tasks)} succeeded, "
        f"{progress.failed}/{len(tasks)} failed"
    )
    logger.info(f"Total time: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} minutes)")
    if progress.succeeded > 0:
        avg_time = overall_elapsed / progress.succeeded
        logger.info(f"Average time per successful download: {avg_time:.2f}s")


def main() -> int:
    """
    CLI entrypoint for downloading SEC 10-K filings.

    Returns:
        Exit code (0 = success, 1 = failure)
    """
    parser = argparse.ArgumentParser(
        description="Download SEC 10-K filings from EDGAR with retry logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single download
  python download_10k_files.py --ticker AAPL --year 2023 --company "MyApp" --email user@example.com

  # Bulk download from CSV
  python download_10k_files.py --csv tickers.csv --company "MyApp" --email user@example.com

  # Use environment variables (SEC_COMPANY_NAME, SEC_EMAIL)
  export SEC_COMPANY_NAME="MyApp"
  export SEC_EMAIL="user@example.com"
  python download_10k_files.py --ticker MSFT --year 2022
        """,
    )

    parser.add_argument("--ticker", help="Single stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--year", type=int, help="Filing year for single ticker (e.g., 2023)")
    parser.add_argument(
        "--csv", help="Path to CSV file with ticker,year rows (bulk mode)"
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("SEC_OUTPUT_DIR", "./data/raw_10k"),
        help="Output directory (default: ./data/raw_10k or $SEC_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--company",
        default=os.getenv("SEC_COMPANY_NAME"),
        help="Company name for SEC user-agent (required; or set $SEC_COMPANY_NAME)",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("SEC_EMAIL"),
        help="Email for SEC user-agent (required; or set $SEC_EMAIL)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for CSV mode (default: 4, range: 1-10)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Minimum seconds between requests (default: 1.0, min: 0.5)",
    )
    parser.add_argument(
        "--no-queue",
        action="store_true",
        help="Disable pushing tasks to Redis queue after download",
    )

    args = parser.parse_args()

    
    if args.workers < 1 or args.workers > 10:
        parser.error("Workers must be between 1 and 10")
    if args.rate_limit < 0.5:
        parser.error("Rate limit must be at least 0.5 seconds")

    
    if not (args.ticker and args.year) and not args.csv:
        parser.error("Must provide either (--ticker AND --year) OR --csv")

    if not args.company:
        parser.error(
            "Company name required: use --company or set SEC_COMPANY_NAME environment variable"
        )

    if not args.email:
        parser.error("Email required: use --email or set SEC_EMAIL environment variable")

    
    if "@" not in args.email or "." not in args.email:
        parser.error(f"Invalid email format: {args.email}")

    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory: {e}")
        return 1

    
    try:
        downloader = Downloader(
            company_name=args.company,
            email_address=args.email,
            download_folder=args.output_dir,
        )
        logger.info(f"Initialized downloader (company={args.company}, email={args.email})")
    except Exception as e:
        logger.error(f"Failed to initialize downloader: {e}")
        return 1

    
    redis_enabled = not args.no_queue

    try:
        if args.ticker and args.year:
            single_start_time = time.time()
            download_single(downloader, args.ticker, args.year)
            single_elapsed = time.time() - single_start_time
            logger.info(f"Download completed successfully in {single_elapsed:.2f}s")

            # Push to queue for single download
            if redis_enabled:
                downloaded_file = find_downloaded_file(args.output_dir, args.ticker.upper())
                if downloaded_file:
                    push_to_queue(downloaded_file, args.ticker.upper(), args.year, redis_enabled)

        elif args.csv:
            download_from_csv(
                downloader,
                args.csv,
                max_workers=args.workers,
                rate_limit_interval=args.rate_limit,
                company_name=args.company,
                email=args.email,
                output_dir=args.output_dir,
                redis_enabled=redis_enabled,
            )
        else:
            parser.error("Invalid combination of arguments")
            return 1

        return 0

    except (DownloadError, ValidationError, FileNotFoundError) as e:
        logger.error(f"Download failed: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())