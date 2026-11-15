#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
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
    with pytest.raises(ValidationError, match="cannot contain"):
        AgentTarget(agent="lab", working_directory="a/../b")


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


async def test_native_hooks_fall_back_for_old_agents(
    target: AgentTarget, monkeypatch: pytest.MonkeyPatch
) -> None:
    from horus_runtime.core.target.base import BaseTarget

    calls: list[str] = []

    async def shell(_self: Any, *args: Any) -> Any:
        calls.append(str(args[0]))
        return 7

    monkeypatch.setattr(AgentTarget, "protocol", property(lambda _: 1))
    for hook in ("path_digest", "pack_folder", "unpack_folder", "move_file"):
        monkeypatch.setattr(BaseTarget, hook, shell)
    monkeypatch.setattr(BaseTarget, "file_size", shell)
    await target.path_digest("/a")
    await target.pack_folder("/b", "/p")
    await target.unpack_folder("/p", "/c")
    await target.move_file("/d", "/e")
    assert await target.file_size("/f") == 7
    assert calls == ["/a", "/b", "/p", "/d", "/f"]


def test_bind_refuses_a_working_directory_outside_the_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from horus_agent import target as target_mod

    monkeypatch.setattr(AgentTarget, "strict", property(lambda _: True))
    monkeypatch.setattr(
        target_mod, "agent_work_dir", lambda *_: str(tmp_path / "home")
    )
    monkeypatch.setenv(BROKER_URL_ENV, "https://b")
    monkeypatch.setenv(BROKER_TOKEN_ENV, "t")
    outside = AgentTarget(agent="lab", working_directory="/etc")
    with pytest.raises(TaskExecutionError, match="outside"):
        outside.bind(fake_task("shell"))
    inside = AgentTarget(
        agent="lab", working_directory=str(tmp_path / "home" / "x")
    )
    inside.bind(fake_task("shell"))
