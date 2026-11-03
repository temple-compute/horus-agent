#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
AgentTarget behaviour that needs no agent.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from horus_builtin.artifact.file import FileArtifact
from horus_runtime.context import HorusContext
from horus_runtime.core.task.exceptions import TaskExecutionError
from pydantic import ValidationError

from horus_agent.target import (
    BROKER_TOKEN_CONTEXT_KEY_ENV,
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    AgentTarget,
)
from tests.fake_broker import RUN_TOKEN, FakeBroker


def fake_task(executor_kind: str) -> Any:
    return cast(
        Any,
        SimpleNamespace(executor=SimpleNamespace(kind=executor_kind)),
    )


def test_location_and_validation(target: AgentTarget) -> None:
    assert target.location_id == "agent://lab"
    assert target.access_cost(cast(Any, None)) is None
    with pytest.raises(ValidationError, match="absolute"):
        AgentTarget(agent="lab", working_directory="relative")


def test_bind_refuses_external_pickle_executor(target: AgentTarget) -> None:
    with pytest.raises(TaskExecutionError, match="cannot run on an agent"):
        target.bind(fake_task("python_function_external"))
    target.bind(fake_task("shell"))


def test_path_on_target(target: AgentTarget, tmp_path: Path) -> None:
    base = tmp_path / "agent"
    inside = FileArtifact(id="a", path=base / "sub" / "f.txt")
    elsewhere = FileArtifact(id="b", path=tmp_path / "other" / "g.txt")
    assert target.path_on_target(inside) == str(base / "sub" / "f.txt")
    assert target.path_on_target(elsewhere) == f"{base}/g.txt"


async def test_missing_broker_config(
    target: AgentTarget, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (BROKER_URL_ENV, BROKER_TOKEN_ENV):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(TaskExecutionError, match="No broker configured"):
        await target.mkdir("/tmp/x")


@pytest.mark.usefixtures("agent")
async def test_token_from_context(
    target: AgentTarget,
    broker: FakeBroker,
    horus_context: HorusContext,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del broker
    monkeypatch.delenv(BROKER_TOKEN_ENV)
    monkeypatch.setenv(BROKER_TOKEN_CONTEXT_KEY_ENV, "workflow_token")
    horus_context.data["workflow_token"] = RUN_TOKEN
    await target.mkdir(str(tmp_path / "made"))
    assert (tmp_path / "made").is_dir()
