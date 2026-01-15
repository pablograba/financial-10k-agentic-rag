"""
Redis connection factory for rq task queue.

Provides connection pooling and retry logic for Redis connections.
"""

import logging
from functools import lru_cache

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from rq import Queue
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import get_settings

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RedisConnectionError),
    before_sleep=lambda retry_state: logger.warning(
        f"Redis connection failed, retrying (attempt {retry_state.attempt_number})..."
    ),
)
def _create_redis_connection() -> Redis:
    """
    Create a Redis connection with retry logic.

    Returns:
        Redis: Connected Redis client.

    Raises:
        RedisConnectionError: If connection fails after retries.
    """
    settings = get_settings()
    client = Redis.from_url(
        settings.redis_url,
        decode_responses=False,
        socket_timeout=5.0,
        socket_connect_timeout=5.0,
        retry_on_timeout=True,
    )
    client.ping()
    logger.info(f"Connected to Redis at {settings.redis_url}")
    return client


@lru_cache
def get_redis_connection() -> Redis:
    """
    Get cached Redis connection.

    Returns:
        Redis: Connected Redis client.
    """
    return _create_redis_connection()


def get_queue(name: str | None = None) -> Queue:
    """
    Get rq Queue instance.

    Args:
        name: Queue name (defaults to settings.redis_queue_name).

    Returns:
        Queue: rq Queue instance.
    """
    settings = get_settings()
    queue_name = name or settings.redis_queue_name
    return Queue(queue_name, connection=get_redis_connection())


def health_check() -> bool:
    """
    Check Redis connection health.

    Returns:
        bool: True if Redis is healthy, False otherwise.
    """
    try:
        conn = get_redis_connection()
        conn.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False
