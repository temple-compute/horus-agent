#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
BrokerClient against scripted broker answers.
"""

import json
import random
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from horus_runtime.core.task.exceptions import (
    TargetLostError,
    TaskExecutionError,
)

from horus_agent.broker import BrokerClient, broker_client

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(random, "uniform", lambda *_: 0.01)


def client(handler: Handler) -> BrokerClient:
    broker = BrokerClient("http://broker", "tok")
    broker._http = httpx.AsyncClient(
        base_url="http://broker/", transport=httpx.MockTransport(handler)
    )
    return broker


def scripted(*answers: Any) -> tuple[Handler, list[httpx.Request]]:
    """Answer requests in order; an int is a bare status code."""
    seen: list[httpx.Request] = []
    queue = list(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return httpx.Response(answer, text="nope")
        return httpx.Response(200, json=answer)

    return handler, seen


async def test_waits_until_done() -> None:
    handler, seen = scripted(
        {"agent_seen_ago": 1.0},
        {"done": True, "result": {"ok": 1}},
    )
    result = await client(handler).call(
        "lab", "mkdir", {"path": "/x"}, offline_timeout=60
    )
    assert result == {"ok": 1}
    assert seen[0].method == "POST"
    body = json.loads(seen[0].content)
    assert body["agent"] == "lab"
    assert body["op"]["method"] == "mkdir"
    assert "run_id" not in body["op"]
    assert seen[1].method == "GET"
    assert seen[1].url.path == f"/agents/ops/{body['op']['op_id']}"


async def test_lost_idempotent_op_is_resent() -> None:
    handler, seen = scripted(
        {"agent_seen_ago": 1.0, "lost": True},
        {"done": True, "result": 7},
    )
    assert await client(handler).call("lab", "poll", {}, offline_timeout=60)
    first, second = (json.loads(r.content)["op"]["op_id"] for r in seen)
    assert first != second


async def test_lost_exec_is_target_lost() -> None:
    handler, _ = scripted({"agent_seen_ago": 1.0, "lost": True})
    with pytest.raises(TargetLostError, match="restarted while running"):
        await client(handler).call("lab", "exec", {}, offline_timeout=60)


async def test_stale_agent_is_target_lost() -> None:
    handler, _ = scripted({"agent_seen_ago": 120.0})
    with pytest.raises(TargetLostError, match="120s ago"):
        await client(handler).call("lab", "poll", {}, offline_timeout=60)


async def test_agent_error_raises() -> None:
    handler, _ = scripted(
        {"done": True, "error": {"type": "OSError", "message": "disk full"}}
    )
    with pytest.raises(TaskExecutionError, match="OSError: disk full"):
        await client(handler).call("lab", "mkdir", {}, offline_timeout=60)


async def test_retries_server_errors_and_transport_errors() -> None:
    handler, seen = scripted(
        503,
        httpx.ConnectError("down"),
        {"done": True, "result": None},
    )
    await client(handler).call("lab", "mkdir", {}, offline_timeout=60)
    ids = {json.loads(r.content)["op"]["op_id"] for r in seen}
    assert len(ids) == 1, "a resent submit keeps its op_id"


async def test_unreachable_broker_gives_up() -> None:
    handler, _ = scripted(httpx.ConnectError("down"))
    with pytest.raises(TaskExecutionError, match="Broker unreachable"):
        await client(handler).call("lab", "mkdir", {}, offline_timeout=0.001)


async def test_refusal_is_raised_with_detail() -> None:
    handler, _ = scripted(403)
    with pytest.raises(TaskExecutionError, match="HTTP 403: nope"):
        await client(handler).call("lab", "mkdir", {}, offline_timeout=60)


async def test_staging() -> None:
    urls = {
        "orchestrator_put_url": "a",
        "orchestrator_get_url": "b",
        "agent_put_url": "c",
        "agent_get_url": "d",
    }
    handler, _ = scripted(urls)
    assert (await client(handler).staging(60)).agent_get_url == "d"


async def test_clients_are_shared_per_loop() -> None:
    assert broker_client("http://b", "t") is broker_client("http://b", "t")
    assert broker_client("http://b", "t") is not broker_client("http://b", "u")
