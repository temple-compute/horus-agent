#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
The orchestrator-side target against an agent that runs approved releases
only: checked ops end to end, through the broker.
"""

import asyncio
import contextlib
import dataclasses
import io
import tarfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from horus_builtin.artifact.file import FileArtifact
from horus_builtin.artifact.folder import FolderArtifact
from horus_builtin.artifact.string import StringArtifact
from horus_builtin.executor.shell import ShellExecutor
from horus_builtin.runtime.command import CommandRuntime
from horus_builtin.task.horus_task import HorusTask
from horus_runtime.context import running_task
from horus_runtime.core.target.exceptions import OperationRefusedError
from horus_runtime.core.task.exceptions import TaskExecutionError

from horus_agent import broker as broker_mod
from horus_agent import policy
from horus_agent import target as target_mod
from horus_agent.config import AgentConfig
from horus_agent.daemon import Agent
from horus_agent.manifest import build_manifest_from_workflow
from horus_agent.target import AgentTarget
from tests.conftest import AGENT_NAME, AGENT_TOKEN
from tests.fake_broker import RUN_ID, FakeBroker

COMMAND = 'printf "%s|%s" ${msg} "$TOKEN" > ${out}'


@contextlib.asynccontextmanager
async def _serving(
    broker: FakeBroker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release: tuple[dict[str, Any], dict[str, bytes], tuple[str, ...]],
) -> AsyncGenerator[Path]:
    """A strict agent with an approved release, given as the workflow, its
    bundled files and extra layout names; yields the run dir."""
    wf, files, names = release
    monkeypatch.setattr(broker_mod, "_INFOS", {})
    manifest = build_manifest_from_workflow(wf, files)
    manifest = dataclasses.replace(manifest, layout=(*manifest.layout, *names))
    policy.approve(manifest, "root")
    broker.release_hash = manifest.release_hash
    home = tmp_path / "agent-home"
    rules = policy.Policy(work_dir=str(home), allowed_hosts=("127.0.0.1",))
    running = Agent(
        AgentConfig(
            url=broker.url,
            agent_id="a1",
            name=AGENT_NAME,
            token=AGENT_TOKEN,
            work_dir=str(home),
        ),
        rules,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(running.serve(stop))
    try:
        for _ in range(100):
            if AGENT_NAME in broker.reported:
                break
            await asyncio.sleep(0.05)
        yield home / wf["name"] / RUN_ID
    finally:
        stop.set()
        await asyncio.wait_for(task, 10)


@pytest.fixture
async def strict_agent(
    broker: FakeBroker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[Path]:
    """A strict agent with one approved release; yields its run dir."""
    wf = {
        "name": "echo",
        "tasks": [
            {
                "id": "say",
                "name": "Say",
                "target": {"kind": "agent", "agent": AGENT_NAME},
                "executor": {"kind": "shell"},
                "runtime": {"kind": "command", "command": COMMAND},
                "inputs": [{"id": "msg", "kind": "string"}],
                "outputs": [{"id": "out", "kind": "file", "path": "o"}],
                "secrets": {"TOKEN": "${secret:t}"},
            }
        ],
    }
    names = ("src", "src.tgz", "moved", "out", "dest", "copy")
    async with _serving(broker, tmp_path, monkeypatch, (wf, {}, names)) as run:
        yield run


async def test_target_speaks_v2_to_a_strict_agent(
    strict_agent: Path, broker: FakeBroker
) -> None:
    target = AgentTarget(agent=AGENT_NAME, working_directory=str(strict_agent))
    assert target.protocol == 2 and target.strict
    with running_task("say"):
        await target.mkdir(str(strict_agent / "src" / "sub"))
    assert broker.ops_seen[-1]["v"] == 2
    assert broker.ops_seen[-1]["task_id"] == "say"
    with pytest.raises(TaskExecutionError, match="approved"):
        await (await target.run_command_sync("id")).communicate()
    with pytest.raises(TaskExecutionError, match="outside"):
        await target.mkdir("/tmp/elsewhere")
    with pytest.raises(TaskExecutionError, match="approved shell"):
        target.bind(
            HorusTask(
                id="say",
                name="Say",
                runtime=CommandRuntime(command="true"),
                executor=ShellExecutor(),
            ).model_copy(
                update={"executor": type("E", (), {"kind": "docker"})()}
            )
        )
    src = strict_agent / "src"
    await target.put_file(b"hello", str(src / "sub" / "a.txt"))
    digest = await target.path_digest(str(src))
    assert digest is not None and digest.startswith("deep:")
    assert await target.path_digest(str(strict_agent / "missing")) is None
    pkg = str(strict_agent / "src.tgz")
    await target.pack_folder(str(src), pkg)
    await target.unpack_folder(pkg, str(strict_agent / "copy"))
    assert await target.path_digest(str(strict_agent / "copy")) == digest
    await target.move_file(
        str(strict_agent / "copy"), str(strict_agent / "moved")
    )
    assert await target.file_size(str(src / "sub" / "a.txt")) == 5
    with pytest.raises(FileNotFoundError):
        await target.file_size(str(strict_agent / "missing"))
    with pytest.raises(TaskExecutionError, match="Uploads are off"):
        await target.upload_url(
            "https://127.0.0.1/x", str(src / "sub" / "a.txt")
        )
    with pytest.raises(TaskExecutionError, match="allowed"):
        await target.fetch_url("http://127.0.0.1/x", str(src / "b"))


async def test_task_launch_from_the_running_task(
    strict_agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = AgentTarget(agent=AGENT_NAME, working_directory=str(strict_agent))
    out = strict_agent / "o"
    task = HorusTask(
        id="say",
        name="Say",
        runtime=CommandRuntime(command=COMMAND),
        executor=ShellExecutor(),
        inputs=[StringArtifact(id="msg", value="hi", path=strict_agent / "m")],
        outputs=[FileArtifact(id="out", path=out)],
    )
    monkeypatch.setattr(target_mod, "_current_task", lambda: task)
    job_dir = str(strict_agent / ".horus_job" / "j")
    with running_task("say"):
        proc = await target.run_command(
            "this command is ignored",
            cwd=str(strict_agent),
            env={"TOKEN": "s3cret"},
            task=task,
        )
        assert await proc.wait() == 0
        assert out.read_text() == "hi|s3cret"
        # The secret reached the job through stdin, not a file.
        assert not list(strict_agent.glob(".horus_job/*/env"))
        # Without the task, the command is sent as is, and refused.
        with pytest.raises(OperationRefusedError, match="approved"):
            await target.run_command("id", cwd=str(strict_agent))
        with pytest.raises(TaskExecutionError, match="runs through"):
            await target.slurm_submit(
                cwd=str(strict_agent), env={}, job_dir=job_dir + "2"
            )
        with pytest.raises(TaskExecutionError, match="No job"):
            await target.slurm_status(str(strict_agent / "nojob"))
        with pytest.raises(TaskExecutionError, match="No job"):
            await target.slurm_cancel(str(strict_agent / "nojob"))
    monkeypatch.setattr(target_mod, "_current_task", lambda: None)
    with pytest.raises(TaskExecutionError, match="running task"):
        await target.slurm_submit(cwd=None, env=None, job_dir=job_dir)


RUN_SCRIPT = b'cat "$1" "$2/a.txt" "$2/sub/b.txt"\n'


def _package(members: dict[str, bytes]) -> bytes:
    """A folder package as the runtime builds it (``tar -C src .``)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return buf.getvalue()


@pytest.fixture
async def staging_agent(
    broker: FakeBroker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[Path]:
    """A strict agent whose release ships a script in ``scripts/``, a folder
    in ``data/``."""
    wf = {
        "name": "crunch",
        "tasks": [
            {
                "id": "crunch",
                "name": "Crunch",
                "target": {"kind": "agent", "agent": AGENT_NAME},
                "executor": {"kind": "shell"},
                "runtime": {
                    "kind": "command",
                    "command": "sh ${script} ${data} ${cfg} > ${out}",
                },
                "inputs": [
                    {"id": "script", "kind": "file", "path": "scripts/c.sh"},
                    {"id": "cfg", "kind": "folder", "path": "data/cfg"},
                    {"id": "data", "kind": "file", "path": "data.txt"},
                ],
                "outputs": [{"id": "out", "kind": "file", "path": "o"}],
            }
        ],
    }
    files = {
        "scripts/c.sh": RUN_SCRIPT,
        "data/cfg/a.txt": b"A\n",
        "data/cfg/sub/b.txt": b"B\n",
    }
    async with _serving(broker, tmp_path, monkeypatch, (wf, files, ())) as run:
        yield run


async def test_a_runtime_staging_sequence_passes_the_strict_agent(
    staging_agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script from a sub-folder, a folder constant, a data input and an
    output: staged as the runtime does, then run."""
    wd = staging_agent
    target = AgentTarget(agent=AGENT_NAME, working_directory=str(wd))
    task = HorusTask(
        id="crunch",
        name="Crunch",
        runtime=CommandRuntime(command="sh ${script} ${data} ${cfg} > ${out}"),
        executor=ShellExecutor(),
        inputs=[
            FileArtifact(id="script", path=wd / "c.sh"),
            FolderArtifact(id="cfg", path=wd / "cfg"),
            FileArtifact(id="data", path=wd / "data.txt"),
        ],
        outputs=[FileArtifact(id="out", path=wd / "o")],
    )
    monkeypatch.setattr(target_mod, "_current_task", lambda: task)
    folders = {"cfg": {"a.txt": b"A\n", "sub/b.txt": b"B\n"}}
    with running_task("crunch"):
        await target.mkdir(str(wd))
        await target.put_file(RUN_SCRIPT, str(wd / "c.sh"))
        for name, members in folders.items():
            pkg = str(wd / f"{name}.1a2b3c4d.horuspkg")
            await target.put_file(_package(members), pkg)
            await target.unpack_folder(pkg, str(wd / name))
            await target.remove(pkg)
        await target.put_file(b"data\n", str(wd / "data.txt"))
        proc = await target.run_command(
            "ignored", cwd=str(wd), env={}, task=task
        )
        assert await proc.wait() == 0
    assert (wd / "o").read_text() == "data\nA\nB\n"
    assert not list(wd.glob("*.horuspkg"))

    # What the strict agent still refuses on the same release.
    with running_task("crunch"):
        with pytest.raises(TaskExecutionError, match="ships"):
            await target.put_file(b"touch PWNED\n", str(wd / "c.sh"))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            body = b"touch PWNED\n"
            info = tarfile.TarInfo("real.txt")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
            link = tarfile.TarInfo("a.txt")
            link.type = tarfile.SYMTYPE
            link.linkname = "real.txt"
            tar.addfile(link)
        pkg = str(wd / "cfg.5e6f7a8b.horuspkg")
        await target.put_file(buf.getvalue(), pkg)
        with pytest.raises(TaskExecutionError, match="link"):
            await target.unpack_folder(pkg, str(wd / "cfg"))
