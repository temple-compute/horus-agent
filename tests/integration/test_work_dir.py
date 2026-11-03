#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
The agent decides where tasks run; the target asks the broker for it.
"""

import asyncio
from pathlib import Path

import pytest
from horus_runtime.core.task.exceptions import TaskExecutionError
from pydantic import ValidationError

from horus_agent.daemon import Agent
from horus_agent.target import AgentTarget
from tests.conftest import AGENT_NAME
from tests.fake_broker import FakeBroker


async def _polled(broker: FakeBroker) -> None:
    for _ in range(100):
        if AGENT_NAME in broker.seen:
            return
        await asyncio.sleep(0.05)
    pytest.fail("the agent never polled")


async def test_unset_working_directory_is_the_agents(
    agent: Agent, broker: FakeBroker
) -> None:
    await _polled(broker)
    target = AgentTarget(agent=AGENT_NAME, offline_timeout=5.0)
    home = agent.config.work_dir
    assert target.resolved_working_directory == home
    await target.put_file(b"hi", f"{home}/x/f.txt")
    assert (Path(home) / "x" / "f.txt").read_bytes() == b"hi"


async def test_relative_working_directory_is_under_the_agents(
    agent: Agent, broker: FakeBroker
) -> None:
    await _polled(broker)
    target = AgentTarget(
        agent=AGENT_NAME, working_directory="flow/run1", offline_timeout=5.0
    )
    assert target.resolved_working_directory == (
        f"{agent.config.work_dir}/flow/run1"
    )


def test_absolute_working_directory_is_kept() -> None:
    target = AgentTarget(agent=AGENT_NAME, working_directory="/scratch/x/")
    assert target.resolved_working_directory == "/scratch/x"


def test_relative_working_directory_cannot_escape() -> None:
    with pytest.raises(ValidationError, match="cannot contain"):
        AgentTarget(agent=AGENT_NAME, working_directory="../etc")


@pytest.mark.usefixtures("broker")
def test_unknown_agent_has_no_working_directory() -> None:
    target = AgentTarget(agent="nobody")
    with pytest.raises(TaskExecutionError, match="HTTP 404"):
        _ = target.resolved_working_directory


def test_agent_that_never_reported(broker: FakeBroker) -> None:
    broker.add_agent("quiet", "tca_quiet")
    with pytest.raises(TaskExecutionError, match="horus-agent run"):
        _ = AgentTarget(agent="quiet").resolved_working_directory
