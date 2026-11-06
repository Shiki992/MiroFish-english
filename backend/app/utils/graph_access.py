"""Shared graph client access, request limits, and read retry policy."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Callable, TypeVar

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from .graph_client import GraphApiError, GraphClient
from .logger import get_logger

logger = get_logger("mirofish.graph_access")

T = TypeVar("T")

# Self-hosted extraction runs at about 8 seconds per episode, so long documents
# and simulations can take far longer than a hosted service's short wait.
GRAPH_INGESTION_WAIT_TIMEOUT_SECONDS = int(
    os.environ.get("GRAPH_INGESTION_WAIT_TIMEOUT_SECONDS", "3600")
)
GRAPH_HTTP_REQUEST_TIMEOUT_SECONDS = 60.0
MAX_GRAPH_SEARCH_QUERY_CHARS = 400
MAX_GRAPH_SEARCH_RESULTS = 50


def normalize_graph_search_query(query: Any) -> str:
    """Return a non-empty search query within the supported length."""

    if not isinstance(query, str):
        raise ValueError("Graph search query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("Graph search query must not be empty")
    return normalized[:MAX_GRAPH_SEARCH_QUERY_CHARS]


def normalize_graph_search_limit(limit: Any) -> int:
    """Clamp a search result limit to the supported range."""

    try:
        normalized = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Graph search limit must be an integer") from exc
    if normalized < 1:
        raise ValueError("Graph search limit must be at least 1")
    return min(normalized, MAX_GRAPH_SEARCH_RESULTS)


@lru_cache(maxsize=1)
def _cached_graph_client() -> GraphClient:
    return GraphClient()


def get_graph_client(_store_url: str | None = None) -> GraphClient:
    """Return the process-shared graph client."""

    return _cached_graph_client()


def clear_graph_client_cache() -> None:
    """Clear the cached client. Intended for tests."""

    _cached_graph_client.cache_clear()


def is_retryable_graph_error(error: BaseException) -> bool:
    """Return whether a failed *read* is safe and useful to retry."""

    if isinstance(
        error,
        (ConnectionError, TimeoutError, OSError, RedisConnectionError, RedisTimeoutError),
    ):
        return True
    if isinstance(error, GraphApiError):
        status_code = error.status_code
        return status_code in {408, 429} or (
            status_code is not None and 500 <= status_code <= 599
        )
    return False


def _retry_after_seconds(error: BaseException) -> float | None:
    if not isinstance(error, GraphApiError) or not error.headers:
        return None
    value = next(
        (
            header_value
            for header_name, header_value in error.headers.items()
            if header_name.lower() == "retry-after"
        ),
        None,
    )
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def call_graph_read_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry a safe read only for connection, 408, 429, or 5xx errors."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or not is_retryable_graph_error(error):
                raise

            retry_after = _retry_after_seconds(error)
            delay = min(
                retry_after if retry_after is not None else initial_delay * (2 ** (attempt - 1)),
                max_delay,
            )
            logger.warning(
                "Graph %s attempt %s/%s failed (%s); retrying in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                type(error).__name__,
                delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")
