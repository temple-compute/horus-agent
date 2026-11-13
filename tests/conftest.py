#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Shared fixtures: a fake broker, a running agent and a Horus context.
"""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from horus_runtime.context import HorusContext, _runtime_ctx
from horus_runtime.registry.auto_registry import AutoRegistry

from horus_agent import config
from horus_agent.config import AgentConfig
from horus_agent.daemon import Agent
from horus_agent.target import (
    BROKER_TOKEN_CONTEXT_KEY_ENV,
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    AgentTarget,
)
from tests.fake_broker import RUN_TOKEN, FakeBroker

AGENT_NAME = "lab"
AGENT_TOKEN = "tca_test"


@pytest.fixture(scope="session", autouse=True)
def init_registry() -> None:
    """Load every installed plugin, including horus_agent, once."""
    AutoRegistry.init_registry()


@pytest.fixture(autouse=True)
def process_state() -> Generator[None]:
    """Undo what ``horus-agent run`` does to the process: env and umask."""
    environ = dict(os.environ)
    umask = os.umask(0o022)
    os.umask(umask)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(environ)
        os.umask(umask)


@pytest.fixture(autouse=True)
def policy_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """
    A private stand-in for ``/etc/horus-agent``, owned by this user, which
    the policy checks trust like root.
    """
    path = tmp_path_factory.mktemp("etc") / "horus-agent"
    path.mkdir(mode=0o755)
    monkeypatch.setattr(config, "POLICY_PATH", path / "policy.toml")
    monkeypatch.setattr(config, "POLICY_OWNERS", frozenset({0, os.getuid()}))
    return path


@pytest.fixture
def horus_context() -> Generator[HorusContext]:
    """A fresh ``HorusContext`` per test."""
    ctx = HorusContext()
    ctx.bus.start()
    token = _runtime_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _runtime_ctx.reset(token)


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> Generator[FakeBroker]:
    """A running fake broker the orchestrator side is configured for."""
    fake = FakeBroker().start()
    fake.add_agent(AGENT_NAME, AGENT_TOKEN)
    monkeypatch.setenv(BROKER_URL_ENV, fake.url)
    monkeypatch.setenv(BROKER_TOKEN_ENV, RUN_TOKEN)
    monkeypatch.delenv(BROKER_TOKEN_CONTEXT_KEY_ENV, raising=False)
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
async def agent(broker: FakeBroker, tmp_path: Path) -> AsyncGenerator[Agent]:
    """An agent serving *broker* in this event loop."""
    running = Agent(
        AgentConfig(
            url=broker.url,
            agent_id="a1",
            name=AGENT_NAME,
            token=AGENT_TOKEN,
            work_dir=str(tmp_path / "agent-home"),
            job_env=("FAKE_SLURM_STATE",),  # tests/fake_slurm
        )
    )
    stop = asyncio.Event()
    task = asyncio.create_task(running.serve(stop))
    try:
        yield running
    finally:
        stop.set()
        await asyncio.wait_for(task, 10)


@pytest.fixture
def target(tmp_path: Path) -> AgentTarget:
    """An agent target whose working directory is on this machine."""
    return AgentTarget(
        agent=AGENT_NAME,
        working_directory=str(tmp_path / "agent"),
        offline_timeout=5.0,
    )
