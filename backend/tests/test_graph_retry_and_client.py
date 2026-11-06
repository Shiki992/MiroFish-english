from pathlib import Path

import pytest

from app.utils import graph_access
from app.utils.graph_client import GraphApiError


def test_permanent_graph_errors_fail_without_retry():
    calls = []

    def operation():
        calls.append(True)
        raise GraphApiError(status_code=400, body={"message": "bad query"})

    with pytest.raises(GraphApiError):
        graph_access.call_graph_read_with_retry(
            operation,
            operation_name="permanent failure",
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 1


def test_rate_limit_retry_respects_retry_after():
    calls = []
    sleeps = []

    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise GraphApiError(
                status_code=429,
                headers={"Retry-After": "7"},
                body={"message": "slow down"},
            )
        return "ok"

    result = graph_access.call_graph_read_with_retry(
        operation,
        operation_name="rate limited read",
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_connection_errors_are_retried():
    from redis.exceptions import ConnectionError as RedisConnectionError

    calls = []

    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise RedisConnectionError("database restarting")
        return "ok"

    assert graph_access.call_graph_read_with_retry(
        operation,
        operation_name="database read",
        sleep=lambda _seconds: None,
    ) == "ok"


def test_graph_client_is_shared():
    graph_access.clear_graph_client_cache()

    assert graph_access.get_graph_client() is graph_access.get_graph_client("ignored")


def test_env_example_no_longer_asks_for_a_hosted_graph_key():
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    contents = env_example.read_text(encoding="utf-8")

    assert "ZEP_API_KEY" not in contents
    assert "FALKORDB_HOST" in contents
