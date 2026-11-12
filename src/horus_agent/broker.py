#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Orchestrator-side client for the broker that relays ops to agents.
"""

import asyncio
import random
import time
import weakref
from typing import Any
from uuid import uuid4

import httpx
from horus_runtime.core.task.exceptions import (
    TargetLostError,
    TaskExecutionError,
)
from horus_runtime.logging import horus_logger

from horus_agent.logs import describe_args
from horus_agent.protocol import (
    IDEMPOTENT_METHODS,
    POLL_WAIT_S,
    AgentInfo,
    Op,
    OpStatus,
    StagingUrls,
)

BACKOFF_MAX_S = 30.0

_CLIENTS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, str], "BrokerClient"]
] = weakref.WeakKeyDictionary()


_WORK_DIRS: dict[tuple[str, str], str] = {}


def agent_work_dir(url: str, token: str, agent: str) -> str:
    """
    The working directory *agent* reported, asked once per process.

    Blocking: target paths are computed synchronously, before any op runs.
    shortcut: a sync request inside the event loop, once per agent per run;
    move it into an async setup hook if the runtime grows one.

    Raises:
        TaskExecutionError: The broker does not know the agent, or the agent
            has not reported its working directory yet.
    """
    key = (url, agent)
    if key in _WORK_DIRS:
        return _WORK_DIRS[key]
    try:
        response = httpx.get(
            f"{url.rstrip('/')}/agents/info/{agent}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
    except httpx.TransportError as exc:
        raise TaskExecutionError(f"Broker unreachable: {exc}") from exc
    if response.status_code >= 400:
        raise TaskExecutionError(
            f"Cannot look up agent {agent!r}: HTTP {response.status_code}: "
            f"{response.text}"
        )
    info = AgentInfo.model_validate(response.json())
    if not info.work_dir:
        raise TaskExecutionError(
            f"Agent {agent!r} has not reported its working directory yet; "
            "start it with horus-agent run"
        )
    horus_logger.log.debug(f"Agent {agent!r} runs tasks in {info.work_dir}")
    _WORK_DIRS[key] = info.work_dir
    return info.work_dir


def broker_client(url: str, token: str) -> "BrokerClient":
    """
    The running loop's client for *url* and *token*, so every target in a
    run shares one connection pool.
    """
    clients = _CLIENTS.setdefault(asyncio.get_running_loop(), {})
    key = (url, token)
    if key not in clients:
        clients[key] = BrokerClient(url, token)
    return clients[key]


class BrokerClient:
    """Submits ops to the broker and waits for their results."""

    def __init__(self, url: str, token: str) -> None:
        """Bind to the broker at *url*, authenticating with *token*."""
        self._http = httpx.AsyncClient(
            base_url=url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {token}"},
            # A long-poll holds the response for up to POLL_WAIT_S.
            timeout=httpx.Timeout(30.0, read=POLL_WAIT_S + 30.0),
        )

    async def call(
        self,
        agent: str,
        method: str,
        args: dict[str, Any],
        *,
        offline_timeout: float,
    ) -> Any:
        """
        Run *method* on *agent* and return its result.

        Raises:
            TargetLostError: The agent has not polled for *offline_timeout*
                seconds, or restarted while running a non-idempotent op.
            TaskExecutionError: The op failed on the agent, or the broker
                refused or stayed unreachable.
        """
        log = horus_logger.log
        started = time.monotonic()
        while True:
            op = Op(op_id=uuid4().hex, method=method, args=args)
            tag = f"Op {op.op_id[:8]} {method} on agent {agent!r}"
            log.debug(f"{tag}: submit {describe_args(args)}")
            status = await self._request(
                "POST",
                "agents/ops/",
                offline_timeout,
                json={"agent": agent, "op": op.model_dump(exclude={"run_id"})},
            )
            while not status.done and not status.lost:
                seen = status.agent_seen_ago
                if seen is None or seen > offline_timeout:
                    raise TargetLostError(
                        f"Agent {agent!r} is offline (last seen: "
                        f"{'never' if seen is None else f'{seen:.0f}s ago'})"
                    )
                log.debug(
                    f"{tag}: waiting (agent last polled {seen:.0f}s ago)"
                )
                status = await self._request(
                    "GET",
                    f"agents/ops/{op.op_id}",
                    offline_timeout,
                    params={"wait": POLL_WAIT_S},
                )
            if status.done:
                break
            if method not in IDEMPOTENT_METHODS:
                raise TargetLostError(
                    f"Agent {agent!r} restarted while running {method}"
                )
            log.debug(f"{tag}: lost on the way; sending it again")
        log.debug(
            f"{tag}: {'failed' if status.error else 'done'} after "
            f"{time.monotonic() - started:.2f}s"
        )
        if status.error is not None:
            raise TaskExecutionError(
                f"{method} on agent {agent!r} failed: "
                f"{status.error.type}: {status.error.message}"
            )
        return status.result

    async def staging(self, offline_timeout: float) -> StagingUrls:
        """Presigned URLs for a fresh staging object."""
        response = await self._send("POST", "agents/staging/", offline_timeout)
        return StagingUrls.model_validate(response.json())

    async def _request(
        self,
        verb: str,
        path: str,
        offline_timeout: float,
        **kwargs: Any,
    ) -> OpStatus:
        response = await self._send(verb, path, offline_timeout, **kwargs)
        return OpStatus.model_validate(response.json())

    async def _send(
        self,
        verb: str,
        path: str,
        offline_timeout: float,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Send one request, retrying with backoff while the broker is down.

        A resent ``POST agents/ops/`` carries the same ``op_id``, which the
        broker queues only once.
        """
        deadline = time.monotonic() + offline_timeout
        delay = 1.0
        while True:
            try:
                response = await self._http.request(verb, path, **kwargs)
            except httpx.TransportError as exc:
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code < 500:
                    break
                failure = f"HTTP {response.status_code}"
            if time.monotonic() + delay > deadline:
                raise TaskExecutionError(f"Broker unreachable: {failure}")
            await asyncio.sleep(delay * random.uniform(0.5, 1.0))
            delay = min(delay * 2, BACKOFF_MAX_S)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise TaskExecutionError(
                f"Broker refused {verb} {path}: "
                f"HTTP {response.status_code}: {detail}"
            )
        return response
