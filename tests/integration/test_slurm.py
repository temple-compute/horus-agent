#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Slurm through an agent: ``SlurmTarget(inner=AgentTarget(...))``.

The agent's machine is this one, with a fake Slurm CLI on ``PATH``
(``tests/fake_slurm``) that runs each batch script as a local background
job. Every Slurm command and file operation the target issues goes through
the broker to the agent, exactly as it would on a login node behind a VPN.
"""

import asyncio
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from horus_runtime.context import HorusContext
from horus_runtime.core.task.exceptions import TaskExecutionError

from horus_agent.target import AgentTarget
from tests.fake_broker import FakeBroker

FAKE_SLURM = Path(__file__).parents[1] / "fake_slurm"

pytestmark = pytest.mark.usefixtures("agent")


@pytest.fixture
def slurm_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path]:
    """Put the fake Slurm CLI on the agent's PATH; yield its state dir."""
    state = tmp_path / "slurm-state"
    monkeypatch.setenv("PATH", f"{FAKE_SLURM}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state))
    monkeypatch.delenv("FAKE_SLURM_PENDING_S", raising=False)
    yield state


def _slurm(target: AgentTarget) -> Any:
    from horus_slurm.target.slurm import SlurmTarget

    return SlurmTarget(inner=target, partition="debug")


def _workflow(work: Path, slurm: Any, command: str) -> Any:
    from horus_builtin.artifact.file import FileArtifact
    from horus_builtin.executor.shell import ShellExecutor
    from horus_builtin.runtime.command import CommandRuntime
    from horus_builtin.target.local import LocalTarget
    from horus_builtin.task.horus_task import HorusTask
    from horus_builtin.workflow.horus_workflow import HorusWorkflow
    from horus_runtime.core.workflow.edge import WorkflowEdge

    local = LocalTarget(working_directory=str(work))
    compute = HorusTask(
        id="compute",
        name="Compute",
        target=slurm,
        executor=ShellExecutor(),
        runtime=CommandRuntime(command=command),
        inputs=[FileArtifact(id="source", path=work / "in.txt")],
        outputs=[FileArtifact(id="result", path=work / "result.txt")],
    )
    collect = HorusTask(
        id="collect",
        name="Collect",
        target=local,
        executor=ShellExecutor(),
        runtime=CommandRuntime(command="cp ${result} ${final}"),
        inputs=[FileArtifact(id="result", path=work / "received.txt")],
        outputs=[FileArtifact(id="final", path=work / "final.txt")],
    )
    return HorusWorkflow(
        name="slurm via agent",
        tasks=[compute, collect],
        edges=[
            WorkflowEdge(
                source="compute",
                source_output="result",
                target="collect",
                target_input="result",
            )
        ],
        orchestrator_target=LocalTarget(working_directory=str(work)),
    )


@pytest.fixture
def work(tmp_path: Path) -> Path:
    work = tmp_path / "orchestrator"
    work.mkdir()
    (work / "in.txt").write_text("hello slurm\n")
    return work


async def test_round_trip(
    horus_context: HorusContext,
    target: AgentTarget,
    broker: FakeBroker,
    slurm_state: Path,
    work: Path,
) -> None:
    del horus_context
    from horus_runtime.core.workflow.status import WorkflowStatus

    workflow = _workflow(
        work,
        _slurm(target),
        "tr a-z A-Z < ${source} > ${result} && echo on-the-node",
    )
    await workflow.run(trigger_id="collect")

    assert workflow.status == WorkflowStatus.COMPLETED
    assert (work / "final.txt").read_text() == "HELLO SLURM\n"
    # One batch job, submitted on the agent's machine through the broker.
    scripts = list(slurm_state.glob("*.script"))
    assert len(scripts) == 1
    assert "#SBATCH --partition=debug" in scripts[0].read_text()
    methods = [op["method"] for op in broker.ops_seen]
    assert "exec" in methods
    assert any(
        "sbatch --parsable" in op["args"].get("cmd", "")
        for op in broker.ops_seen
    )
    # The job ran in the agent's working directory.
    assert any(Path(target.resolved_working_directory).rglob("result.txt"))


async def test_failed_job_fails_the_task(
    horus_context: HorusContext,
    target: AgentTarget,
    slurm_state: Path,
    work: Path,
) -> None:
    del horus_context
    from horus_runtime.core.workflow.status import WorkflowStatus

    workflow = _workflow(work, _slurm(target), "echo boom >&2; exit 3")
    with pytest.raises(TaskExecutionError, match="return code 3"):
        await workflow.run(trigger_id="collect")
    assert workflow.status == WorkflowStatus.FAILED
    # It failed as a batch job, not before submission.
    assert [p.read_text() for p in slurm_state.glob("*.code")] == ["3\n"]


async def test_pending_job_then_runs(
    horus_context: HorusContext,
    target: AgentTarget,
    slurm_state: Path,
    work: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del horus_context
    from horus_runtime.core.workflow.status import WorkflowStatus

    # Pending for longer than one poll interval, so squeue reports PENDING
    # through the agent before the job starts.
    monkeypatch.setenv("FAKE_SLURM_PENDING_S", "6")
    workflow = _workflow(work, _slurm(target), "cp ${source} ${result}")
    await workflow.run(trigger_id="collect")
    assert workflow.status == WorkflowStatus.COMPLETED
    assert (work / "final.txt").read_text() == "hello slurm\n"
    assert list(slurm_state.glob("*.started"))


async def test_cancel_scancels_the_job(
    horus_context: HorusContext,
    target: AgentTarget,
    broker: FakeBroker,
    slurm_state: Path,
    work: Path,
) -> None:
    del horus_context
    workflow = _workflow(
        work, _slurm(target), "sleep 60 && cp ${source} ${result}"
    )
    running = asyncio.create_task(workflow.run(trigger_id="collect"))

    def polling() -> bool:
        # The target knows the job id once it asks Slurm about the job.
        return any(
            "squeue" in op["args"].get("cmd", "") for op in broker.ops_seen
        )

    for _ in range(200):
        if list(slurm_state.glob("*.started")) and polling():
            break
        await asyncio.sleep(0.1)
    else:
        pytest.fail("the job never started")
    pid = int(next(slurm_state.glob("*.pid")).read_text())

    running.cancel()
    with pytest.raises((asyncio.CancelledError, Exception)):
        await asyncio.wait_for(running, 60)

    assert (slurm_state / "scancel.log").exists(), "scancel ran on the agent"
    assert list(slurm_state.glob("*.cancelled"))
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    else:
        pytest.fail("the job is still running after scancel")
