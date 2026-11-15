#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
An open agent runs a command wrapped the way horus-resource-monitor wraps
it, and reads the resource files back with ``uploads = false``.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from horus_runtime.core.task.exceptions import TaskExecutionError

from horus_agent import broker as broker_mod
from horus_agent.config import AgentConfig
from horus_agent.daemon import Agent
from horus_agent.policy import Policy
from horus_agent.target import AgentTarget
from tests.conftest import AGENT_NAME, AGENT_TOKEN
from tests.fake_broker import RUN_ID, FakeBroker

# What the monitor's wrapper leaves behind (see its ``wrap_command``): the
# host snapshot moved into place, the sample log in side-artifacts, GNU
# ``time -v`` lines and the GPU log in the working dir. A fixture, as the
# monitor is a separate package.
WRAPPED = (
    "{ printf 'token=t1\\nhostname=lab\\ncpus_host=4\\n'; } "
    "> .horus_host.txt.tmp"
    " && mv -f .horus_host.txt.tmp .horus_host.txt; "
    "mkdir -p side-artifacts; "
    "printf 'epoch,rss_kb,cpu_s,gpu_mem_mb,gpu_util_pct,io_read_b,io_write_b,"
    "attribution\\n1700000000,1024,0.10,,,,,exact\\n' "
    "> side-artifacts/resource-usage.csv; "
    "sh -c 'echo work'; rc=$?; "
    "printf '\\tUser time (seconds): 0.10\\n"
    "\\tMaximum resident set size (kbytes): 1024\\n' > .horus_resource.txt; "
    "printf '0, 10, 5\\n' > .horus_gpu.csv; exit $rc"
)
FILES = {
    ".horus_host.txt": b"hostname=lab",
    ".horus_resource.txt": b"Maximum resident set size (kbytes): 1024",
    ".horus_gpu.csv": b"0, 10, 5",
    "side-artifacts/resource-usage.csv": b"1700000000,1024,0.10",
}


@pytest.fixture
async def open_agent(
    broker: FakeBroker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[Path]:
    monkeypatch.setattr(broker_mod, "_INFOS", {})
    home = tmp_path / "agent-home"
    rules = Policy(
        mode="open",
        work_dir=str(home),
        allowed_hosts=("127.0.0.1",),
        uploads=False,
    )
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
        yield home / "wf" / RUN_ID / "task"
    finally:
        stop.set()
        await asyncio.wait_for(task, 10)


async def test_wrapped_command_files_are_read_back(open_agent: Path) -> None:
    cwd = open_agent
    target = AgentTarget(agent=AGENT_NAME, working_directory=str(cwd))
    await target.mkdir(str(cwd))
    proc = await target.run_command_sync(WRAPPED, cwd=str(cwd))
    assert proc.returncode == 0
    for name, start in FILES.items():
        assert (await target.get_file(str(cwd / name))).find(start) >= 0
    # Task output is not a monitor file: still refused with uploads off.
    (cwd / "result.txt").write_text("data\n")
    with pytest.raises(TaskExecutionError, match="Uploads are off"):
        await target.get_file(str(cwd / "result.txt"))
