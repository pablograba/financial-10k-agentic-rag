"""
rq worker entrypoint for 10-K filing indexer.

Provides:
- Startup recovery (re-queue unindexed files)
- Graceful shutdown handling
- Health checks
"""

import argparse
import logging
import os
import signal
import sys
import time

import structlog
from rq import Worker, Queue
from rq.job import Job

from src.config import get_settings
from src.queue.connection import get_redis_connection, get_queue, health_check
from src.queue.tasks import FilingTask, process_filing

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

shutdown_requested = False


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("shutdown_requested", signal=signal.Signals(signum).name)


def startup_recovery() -> int:
    """
    Re-queue any unindexed files found on disk.

    Called at worker startup to ensure no files are missed.

    Returns:
        int: Number of tasks re-queued.
    """
    from src.ingestion.pipeline import get_unindexed_filings

    logger.info("startup_recovery_started")

    try:
        unindexed = get_unindexed_filings()
    except Exception as e:
        logger.error("startup_recovery_failed", error=str(e))
        return 0

    if not unindexed:
        logger.info("startup_recovery_complete", requeued=0)
        return 0

    queue = get_queue()
    requeued = 0

    for filing in unindexed:
        try:
            task = FilingTask(
                file_path=filing["file_path"],
                ticker=filing["ticker"],
                year=0,
                cik="",
                accession_number=filing["accession_number"],
            )

            queue.enqueue(
                process_filing,
                task.to_dict(),
                job_timeout=get_settings().request_timeout,
            )
            requeued += 1

        except Exception as e:
            logger.warning(
                "failed_to_requeue",
                file_path=filing["file_path"],
                error=str(e),
            )

    logger.info("startup_recovery_complete", requeued=requeued)
    return requeued


def run_worker(
    queue_names: list[str] | None = None,
    burst: bool = False,
    with_recovery: bool = True,
) -> None:
    """
    Run the rq worker.

    Args:
        queue_names: List of queue names to listen on.
        burst: If True, quit after all jobs are processed.
        with_recovery: If True, run startup recovery first.
    """
    settings = get_settings()
    queue_names = queue_names or [settings.redis_queue_name]

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    logger.info(
        "worker_starting",
        queues=queue_names,
        burst=burst,
    )

    if not health_check():
        logger.error("redis_not_available")
        sys.exit(1)

    from src.llm.factory import get_ollama_client

    try:
        ollama = get_ollama_client()
        if not ollama.health_check():
            logger.warning("ollama_not_available", message="Worker will retry on task execution")
    except Exception as e:
        logger.warning("ollama_check_failed", error=str(e))

    from src.store.qdrant_local import get_qdrant_store

    try:
        store = get_qdrant_store()
        info = store.get_collection_info()
        logger.info("qdrant_ready", collection_info=info)
    except Exception as e:
        logger.error("qdrant_init_failed", error=str(e))
        sys.exit(1)

    if with_recovery:
        startup_recovery()

    conn = get_redis_connection()
    queues = [Queue(name, connection=conn) for name in queue_names]

    worker = Worker(
        queues,
        connection=conn,
        name=f"indexer-{os.getpid()}",
    )

    logger.info("worker_ready", worker_name=worker.name)

    try:
        worker.work(burst=burst, with_scheduler=False)
    except Exception as e:
        logger.error("worker_error", error=str(e))
        raise
    finally:
        logger.info("worker_stopped")


def main() -> int:
    """
    CLI entrypoint for the worker.

    Returns:
        int: Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run the 10-K indexer worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--queue",
        "-q",
        action="append",
        dest="queues",
        help="Queue names to listen on (can specify multiple)",
    )
    parser.add_argument(
        "--burst",
        "-b",
        action="store_true",
        help="Run in burst mode (quit when queue is empty)",
    )
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="Skip startup recovery scan",
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

    try:
        run_worker(
            queue_names=args.queues,
            burst=args.burst,
            with_recovery=not args.no_recovery,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("worker_interrupted")
        return 130
    except Exception as e:
        logger.exception("worker_failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
