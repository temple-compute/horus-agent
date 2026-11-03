#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
The agent loop: long-poll the broker for ops, run them, post the results.

Every connection goes out from this machine; nothing listens on it.
"""

import asyncio
import contextlib
import importlib.metadata
import random
import socket
from uuid import uuid4

import httpx
from horus_runtime.logging import horus_logger

from horus_agent.config import AgentConfig
from horus_agent.ops import OpRunner
from horus_agent.protocol import POLL_WAIT_S, Op, OpResult

BACKOFF_MAX_S = 60.0
MAX_HOLDING = 1000
"""Ops taken but unanswered before the agent stops taking more."""
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


async def _pause(stop: asyncio.Event, seconds: float) -> None:
    """Sleep *seconds*, waking early if *stop* is set."""
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=seconds)


class Agent:
    """One running agent process."""

    def __init__(self, config: AgentConfig) -> None:
        """Serve as the agent registered in *config*."""
        self.config = config
        self.runner = OpRunner()
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
        async with httpx.AsyncClient(
            base_url=self.config.url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {self.config.token}"},
            timeout=httpx.Timeout(30.0, read=POLL_WAIT_S + 30.0),
        ) as http:
            sender = asyncio.create_task(self._send_results(http, results))
            try:
                await self._poll_loop(http, stop, results, running)
            finally:
                if running:
                    await asyncio.gather(*running, return_exceptions=True)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(results.join(), DRAIN_TIMEOUT_S)
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
        }
        delay = 1.0
        horus_logger.log.info(
            f"Agent {self.config.name!r} polling {self.config.url}"
        )
        while not stop.is_set():
            if len(self.holding) >= MAX_HOLDING:
                # Finish some work first; the broker caps the list.
                await _pause(stop, 0.5)
                continue
            body["holding"] = sorted(self.holding)
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
                    if response.status_code == 200:
                        for raw in response.json()["ops"]:
                            op = Op.model_validate(raw)
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
        result = await self.runner.run(op)
        if result.error is not None:
            # Only the method and error: args carry task secrets in env.
            horus_logger.log.warning(
                f"{op.method} failed: {result.error.type}: "
                f"{result.error.message}"
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
