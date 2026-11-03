#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
A real workflow with a task on an agent target between two local tasks.
"""

from pathlib import Path

import pytest
from horus_runtime.context import HorusContext

from horus_agent.target import AgentTarget


@pytest.mark.usefixtures("agent")
async def test_local_agent_local(
    horus_context: HorusContext, target: AgentTarget, tmp_path: Path
) -> None:
    del horus_context
    from horus_builtin.artifact.file import FileArtifact
    from horus_builtin.artifact.folder import FolderArtifact
    from horus_builtin.executor.shell import ShellExecutor
    from horus_builtin.runtime.command import CommandRuntime
    from horus_builtin.target.local import LocalTarget
    from horus_builtin.task.horus_task import HorusTask
    from horus_builtin.workflow.horus_workflow import HorusWorkflow
    from horus_runtime.core.workflow.edge import WorkflowEdge
    from horus_runtime.core.workflow.status import WorkflowStatus

    work = tmp_path / "orchestrator"
    work.mkdir()
    (work / "in.txt").write_text("hello agent\n")
    local = LocalTarget(working_directory=str(work))

    source = FileArtifact(id="source", path=work / "in.txt")
    shouted = FolderArtifact(id="shouted", path=work / "shouted")
    final = FileArtifact(id="final", path=work / "final.txt")

    shout = HorusTask(
        id="shout",
        name="Shout",
        target=target,
        executor=ShellExecutor(),
        runtime=CommandRuntime(
            command=(
                "mkdir -p ${shouted} && "
                "tr a-z A-Z < ${source} > ${shouted}/upper.txt && "
                "hostname > ${shouted}/where.txt"
            )
        ),
        inputs=[source],
        outputs=[shouted],
    )
    collect = HorusTask(
        id="collect",
        name="Collect",
        target=local,
        executor=ShellExecutor(),
        runtime=CommandRuntime(command="cat ${shouted}/upper.txt > ${final}"),
        inputs=[FolderArtifact(id="shouted", path=work / "received")],
        outputs=[final],
    )
    workflow = HorusWorkflow(
        name="agent round trip",
        tasks=[shout, collect],
        edges=[
            WorkflowEdge(
                source="shout",
                source_output="shouted",
                target="collect",
                target_input="shouted",
            )
        ],
        orchestrator_target=LocalTarget(working_directory=str(work)),
    )

    await workflow.run(trigger_id="collect")

    assert workflow.status == WorkflowStatus.COMPLETED
    assert (work / "final.txt").read_text() == "HELLO AGENT\n"
    # The task ran in the agent's working directory, not the orchestrator's.
    assert any((tmp_path / "agent").rglob("upper.txt"))
