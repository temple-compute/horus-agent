#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
The agent loop: long-poll the broker for ops, run them, post the results.

Every connection goes out from this machine; nothing listens on it.
"""

import asyncio
import contextlib
import importlib.metadata
import os
import platform
import random
import socket
import ssl
import time
from urllib.parse import urlsplit
from uuid import uuid4

import certifi
import httpx
import pydantic
from horus_runtime.logging import horus_logger

from horus_agent.config import AgentConfig
from horus_agent.logs import describe_args, describe_result, summarize
from horus_agent.ops import OpRunner
from horus_agent.protocol import POLL_WAIT_S, Op, OpResult

BACKOFF_MAX_S = 60.0
MAX_HOLDING = 1000
"""Ops taken but unanswered before the agent stops taking more."""
MAX_RUNNING = 64
"""Ops run at once; the rest wait their turn."""
MAX_BATCH_BYTES = 8 * 1024 * 1024
DRAIN_TIMEOUT_S = 30.0
"""How long a stopping agent keeps trying to deliver its last results."""


class AgentUnauthorizedError(Exception):
    """The broker no longer accepts this agent's token."""


def agent_version() -> str:
    """The installed horus-agent version."""
    try:
        return importlib.metadata.version("horus_agent")
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})


def check_broker_url(url: str) -> None:
    """
    Refuse a broker URL that is not ``https``, except on this machine's
    loopback (a local TLS proxy, tests).

    Raises:
        ValueError: *url* would send the token in clear text.
    """
    parts = urlsplit(url)
    if parts.scheme == "https" and parts.hostname:
        return
    if parts.scheme == "http" and parts.hostname in LOOPBACK:
        return
    raise ValueError(f"The broker URL must start with https://, not {url!r}")


def broker_tls(ca_bundle: str = "") -> ssl.SSLContext:
    """
    TLS settings for the broker: verify against *ca_bundle* when set, else
    certifi's CAs. ``SSL_CERT_FILE`` and ``SSL_CERT_DIR`` are ignored.
    """
    return ssl.create_default_context(cafile=ca_bundle or certifi.where())


async def _pause(stop: asyncio.Event, seconds: float) -> None:
    """Sleep *seconds*, waking early if *stop* is set."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=seconds)


class Agent:
    """One running agent process."""

    def __init__(self, config: AgentConfig) -> None:
        """Serve as the agent registered in *config*."""
        self.config = config
        self.runner = OpRunner(config.run_as, config.run_as_via)
        self._slots = asyncio.Semaphore(MAX_RUNNING)
        self.session = uuid4().hex
        """
        New on every start. The broker compares it with the session an op
        was delivered to, to tell an op lost in a restart from a slow one.
        """
        self.holding: set[str] = set()
        """
        Ops taken and not yet answered to the broker, sent with every poll:
        an op handed out but missing here never reached this agent (the
        poll response was lost), so the broker reports it lost.
        """

    async def serve(self, stop: asyncio.Event) -> None:
        """
        Poll and run ops until *stop* is set, then finish the ops already
        started and deliver their results.

        Raises:
            AgentUnauthorizedError: The broker refused the agent's token.
        """
        results: asyncio.Queue[OpResult] = asyncio.Queue()
        running: set[asyncio.Task[None]] = set()
        log = horus_logger.log
        log.debug(
            f"Agent {self.config.name!r} (id {self.config.agent_id}) "
            f"version {agent_version()}, session {self.session}, pid "
            f"{os.getpid()}, user uid {os.getuid()}, host "
            f"{socket.gethostname()}, {platform.platform()}, Python "
            f"{platform.python_version()}, cwd {os.getcwd()}, tasks run in "
            f"{self.config.work_dir}"
        )
        async with httpx.AsyncClient(
            base_url=self.config.url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {self.config.token}"},
            verify=broker_tls(self.config.ca_bundle),
            timeout=httpx.Timeout(30.0, read=POLL_WAIT_S + 30.0),
        ) as http:
            sender = asyncio.create_task(self._send_results(http, results))
            try:
                await self._poll_loop(http, stop, results, running)
            finally:
                log.debug(
                    f"Stopping: {len(running)} ops still running, "
                    f"{results.qsize()} results to deliver"
                )
                if running:
                    await asyncio.gather(*running, return_exceptions=True)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(results.join(), DRAIN_TIMEOUT_S)
                log.debug(f"Stopped; {results.qsize()} results undelivered")
                sender.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sender

    async def _poll_loop(
        self,
        http: httpx.AsyncClient,
        stop: asyncio.Event,
        results: asyncio.Queue[OpResult],
        running: set[asyncio.Task[None]],
    ) -> None:
        body: dict[str, object] = {
            "session": self.session,
            "version": agent_version(),
            "hostname": socket.gethostname(),
            "work_dir": self.config.work_dir or None,
        }
        delay = 1.0
        horus_logger.log.info(
            f"Agent {self.config.name!r} polling {self.config.url}"
        )
        log = horus_logger.log
        while not stop.is_set():
            if len(self.holding) >= MAX_HOLDING:
                # Finish some work first; the broker caps the list.
                log.debug(f"Holding {len(self.holding)} ops; not polling")
                await _pause(stop, 0.5)
                continue
            body["holding"] = sorted(self.holding)
            log.debug(
                f"Poll: waiting up to {POLL_WAIT_S}s for ops, holding "
                f"{len(self.holding)}"
            )
            started = time.monotonic()
            poll = asyncio.create_task(http.post("agents/me/poll", json=body))
            stopping = asyncio.create_task(stop.wait())
            await asyncio.wait(
                {poll, stopping}, return_when=asyncio.FIRST_COMPLETED
            )
            stopping.cancel()
            if not poll.done():
                poll.cancel()
                return
            try:
                response = poll.result()
            except httpx.TransportError as exc:
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 401:
                    raise AgentUnauthorizedError(
                        "The broker refused this agent's token; register it "
                        "again"
                    )
                if response.status_code < 400:
                    delay = 1.0
                    waited = time.monotonic() - started
                    if response.status_code != 200:
                        log.debug(
                            f"Poll: no ops (HTTP {response.status_code}) "
                            f"after {waited:.1f}s"
                        )
                    else:
                        ops = response.json()["ops"]
                        log.debug(f"Poll: {len(ops)} ops after {waited:.1f}s")
                        for raw in ops:
                            try:
                                op = Op.model_validate(raw)
                            except pydantic.ValidationError as exc:
                                # Never held: the broker reports it lost.
                                log.warning(
                                    f"Ignoring a malformed op: "
                                    f"{exc.error_count()} errors"
                                )
                                continue
                            self.holding.add(op.op_id)
                            task = asyncio.create_task(
                                self._run_one(op, results)
                            )
                            running.add(task)
                            task.add_done_callback(running.discard)
                    continue
                failure = f"HTTP {response.status_code}"
            horus_logger.log.warning(
                f"Broker poll failed ({failure}); retrying in {delay:.0f}s"
            )
            await _pause(stop, delay * random.uniform(0.5, 1.0))
            delay = min(delay * 2, BACKOFF_MAX_S)

    async def _run_one(self, op: Op, results: asyncio.Queue[OpResult]) -> None:
        log = horus_logger.log
        tag = f"Op {op.op_id[:8]} {op.method}"
        env = op.args.get("env")
        secrets = (
            [str(v) for v in env.values()] if isinstance(env, dict) else []
        )
        if summary := summarize(op.method, op.args):
            log.info(f"[run {(op.run_id or '-')[:8]}] {summary}")
        log.debug(f"{tag} start (run {op.run_id}): {describe_args(op.args)}")
        started = time.monotonic()
        async with self._slots:
            result = await self.runner.run(op)
        took = time.monotonic() - started
        if result.error is not None:
            # Only the method and error: args carry task secrets in env.
            horus_logger.log.warning(
                f"{op.method} failed: {result.error.type}: "
                f"{result.error.message}"
            )
            log.debug(f"{tag} failed after {took:.2f}s")
        else:
            answer = result.result
            if op.method == "exec" and isinstance(answer, dict):
                code = answer.get("returncode")
                log.info(f"exec exited {code} in {took:.1f}s")
            elif op.method == "launch" and isinstance(answer, dict):
                log.info(f"launched pid {answer.get('pid')}")
            log.debug(
                f"{tag} done in {took:.2f}s: "
                f"{describe_result(op.method, result.result, secrets)}"
            )
        await results.put(result)

    async def _send_results(
        self, http: httpx.AsyncClient, results: asyncio.Queue[OpResult]
    ) -> None:
        """Post results in batches of at most :data:`MAX_BATCH_BYTES`."""
        while True:
            batch = [await results.get()]
            size = len(batch[0].model_dump_json())
            while not results.empty() and size < MAX_BATCH_BYTES:
                batch.append(results.get_nowait())
                size += len(batch[-1].model_dump_json())
            horus_logger.log.debug(
                f"Results: posting {len(batch)} ({size} bytes): "
                f"{', '.join(r.op_id[:8] for r in batch)}"
            )
            await self._post_results(http, batch)
            for result in batch:
                self.holding.discard(result.op_id)
                results.task_done()

    async def _post_results(
        self, http: httpx.AsyncClient, batch: list[OpResult]
    ) -> None:
        """
        Post *batch*, retrying while the broker is unreachable. A batch the
        broker refuses (a proxy's 413, say) is retried one result at a time;
        a single refused result is dropped, and its op reported lost.
        """
        delay = 1.0
        while True:
            try:
                response = await http.post(
                    "agents/me/results",
                    json={"results": [r.model_dump() for r in batch]},
                )
            except httpx.TransportError as exc:
                failure = f"{type(exc).__name__}: {exc}"
            else:
                code = response.status_code
                if code < 400:
                    horus_logger.log.debug(
                        f"Results: delivered {len(batch)} (HTTP {code})"
                    )
                    return
                # 401 is retried: the poll loop stops the agent on it.
                if code < 500 and code != 401:
                    if len(batch) == 1:
                        horus_logger.log.warning(
                            f"Broker refused the result of an op (HTTP "
                            f"{code}); dropping it"
                        )
                        return
                    for result in batch:
                        await self._post_results(http, [result])
                    return
                failure = f"HTTP {code}"
            horus_logger.log.warning(
                f"Posting results failed ({failure}); retrying"
            )
            await asyncio.sleep(delay * random.uniform(0.5, 1.0))
            delay = min(delay * 2, BACKOFF_MAX_S)
