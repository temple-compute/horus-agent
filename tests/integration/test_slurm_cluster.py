#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
Slurm through an agent on a real Slurm cluster.

``horus-agent`` runs inside the login container of horus-slurm's docker
cluster and polls a broker on this host, the way a login node behind a VPN
would. The orchestrator side (this test) never connects to the container;
jobs go through real ``sbatch`` to the compute nodes ``c1``/``c2``.

Opt-in. Start the cluster from a horus-slurm checkout, then run::

    ITEST_DIR=/tmp/horus-agent-slurm-itest-$(id -u) \\
    ITEST_PROJECT=horus-agent-slurm-itest tests/integration/cluster/up.sh
    HORUS_AGENT_SLURM_ITEST=1 uv run pytest --no-cov \\
        tests/integration/test_slurm_cluster.py
"""

import json
import os
import subprocess
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from horus_runtime.context import HorusContext

from horus_agent.target import BROKER_TOKEN_ENV, BROKER_URL_ENV
from tests.fake_broker import RUN_TOKEN, FakeBroker

LOGIN = os.environ.get(
    "HORUS_AGENT_SLURM_LOGIN", "horus-agent-slurm-itest-slurmctld-1"
)
ITEST_DIR = Path(
    os.environ.get("ITEST_DIR", f"/tmp/horus-agent-slurm-itest-{os.getuid()}")
)
INSTALL = Path(__file__).parents[1] / "slurm_cluster" / "install_agent.sh"
AGENT = "hpc"

pytestmark = pytest.mark.skipif(
    os.environ.get("HORUS_AGENT_SLURM_ITEST") != "1",
    reason="set HORUS_AGENT_SLURM_ITEST=1 with the docker Slurm cluster up",
)


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture(scope="module")
def cluster_broker() -> Generator[FakeBroker]:
    """A broker on this host, with an agent polling it from the login node."""
    broker = FakeBroker(max_wait=5.0, host="0.0.0.0").start()
    token = "tca_cluster"
    broker.add_agent(AGENT, token)
    gateway = _docker(
        "inspect",
        "-f",
        "{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}",
        LOGIN,
    ).strip()
    broker.public_url = f"http://{gateway}:{broker.port}"
    subprocess.run([str(INSTALL), LOGIN, str(ITEST_DIR)], check=True)
    config = ITEST_DIR / "agent.json"
    config.write_text(
        json.dumps(
            {
                "url": broker.public_url,
                "agent_id": "a1",
                "name": AGENT,
                "token": token,
            }
        )
    )
    config.chmod(0o600)
    uid = str(os.getuid())
    _docker(
        "exec",
        "-d",
        "-u",
        uid,
        "-e",
        f"HOME={ITEST_DIR}",
        "-e",
        f"HORUS_AGENT_CONFIG={config}",
        LOGIN,
        "bash",
        "-c",
        f"exec {ITEST_DIR}/agent-venv/bin/horus-agent run "
        f">{ITEST_DIR}/agent.log 2>&1",
    )
    deadline = time.monotonic() + 60
    while AGENT not in broker.seen:
        if time.monotonic() > deadline:
            log = (ITEST_DIR / "agent.log").read_text(errors="replace")
            pytest.fail(f"the agent never polled the broker:\n{log}")
        time.sleep(0.5)
    try:
        yield broker
    finally:
        subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                uid,
                LOGIN,
                "pkill",
                "-f",
                "horus-agent run",
            ],
            check=False,
        )
        subprocess.run(
            ["docker", "exec", "-u", uid, LOGIN, "scancel", "--me"],
            check=False,
        )
        broker.stop()


@pytest.fixture
def broker_env(
    cluster_broker: FakeBroker, monkeypatch: pytest.MonkeyPatch
) -> FakeBroker:
    monkeypatch.setenv(BROKER_URL_ENV, cluster_broker.url)
    monkeypatch.setenv(BROKER_TOKEN_ENV, RUN_TOKEN)
    return cluster_broker


def _workflow(work: Path, remote: Path, *, shared: bool) -> Any:
    """
    Two Slurm tasks on the agent's cluster, then a local task that gathers
    their outputs. ``shared`` declares the Slurm target once under
    ``targets:``, so both tasks run as steps of one allocation.
    """
    from horus_builtin.workflow.horus_workflow import HorusWorkflow

    slurm = {
        "kind": "slurm",
        "partition": "cpu",
        "cpus_per_task": 1,
        "mem": "100M",
        "time_limit": "00:05:00",
        "working_directory": str(remote),
        "inner": {
            "kind": "agent",
            "agent": AGENT,
            "working_directory": str(remote),
        },
    }
    target = {"ref": "hpc"} if shared else slurm

    def compute(name: str) -> dict[str, Any]:
        return {
            "id": name,
            "kind": "horus_task",
            "name": name,
            "target": target,
            "executor": {"kind": "shell"},
            "runtime": {
                "kind": "command",
                "command": (
                    "tr a-z A-Z < ${source} > ${out} && hostname >> ${out}"
                ),
            },
            "inputs": [
                {"id": "source", "kind": "file", "path": str(work / "in.txt")}
            ],
            "outputs": [{"id": "out", "kind": "file", "path": f"{name}.txt"}],
        }

    data: dict[str, Any] = {
        "kind": "horus_workflow",
        "name": "slurm cluster via agent",
        "orchestrator_target": {
            "kind": "local",
            "working_directory": str(work),
        },
        "tasks": [
            compute("first"),
            compute("second"),
            {
                "id": "gather",
                "kind": "horus_task",
                "name": "gather",
                "target": {"kind": "local", "working_directory": str(work)},
                "executor": {"kind": "shell"},
                "runtime": {
                    "kind": "command",
                    "command": "cat ${a} ${b} > ${all}",
                },
                "inputs": [
                    {"id": "a", "kind": "file", "path": "a.txt"},
                    {"id": "b", "kind": "file", "path": "b.txt"},
                ],
                "outputs": [
                    {
                        "id": "all",
                        "kind": "file",
                        "path": str(work / "all.txt"),
                    }
                ],
            },
        ],
        "edges": [
            {
                "source": "first",
                "source_output": "out",
                "target": "gather",
                "target_input": "a",
            },
            {
                "source": "second",
                "source_output": "out",
                "target": "gather",
                "target_input": "b",
            },
        ],
    }
    if shared:
        data["targets"] = {"hpc": slurm}
    return HorusWorkflow.model_validate(data)


@pytest.mark.parametrize("shared", [False, True], ids=["per-task", "shared"])
async def test_jobs_run_on_compute_nodes(
    horus_context: HorusContext,
    broker_env: FakeBroker,
    tmp_path: Path,
    shared: bool,
) -> None:
    del horus_context
    from horus_runtime.core.workflow.status import WorkflowStatus

    work = tmp_path / "orchestrator"
    work.mkdir()
    (work / "in.txt").write_text("hello cluster\n")
    remote = ITEST_DIR / "runs" / uuid.uuid4().hex
    workflow = _workflow(work, remote, shared=shared)

    await workflow.run(trigger_id="gather")

    assert workflow.status == WorkflowStatus.COMPLETED
    lines = (work / "all.txt").read_text().splitlines()
    assert lines[0::2] == ["HELLO CLUSTER", "HELLO CLUSTER"]
    # hostname ran inside the batch job, on a compute node.
    assert set(lines[1::2]) <= {"c1", "c2"}
    # Every Slurm command went through the agent.
    commands = [
        op["args"].get("cmd", "")
        for op in broker_env.ops_seen
        if op["method"] == "exec"
    ]
    assert any(cmd.startswith("sbatch") for cmd in commands)


def _squeue(name: str) -> str:
    return _docker(
        "exec",
        "-u",
        str(os.getuid()),
        LOGIN,
        "squeue",
        "--me",
        "-h",
        "-n",
        name,
        "-o",
        "%T",
    ).strip()


async def test_cancel_cancels_the_slurm_job(
    horus_context: HorusContext,
    broker_env: FakeBroker,
    tmp_path: Path,
) -> None:
    del horus_context, broker_env
    import asyncio

    from horus_builtin.workflow.horus_workflow import HorusWorkflow

    remote = ITEST_DIR / "runs" / uuid.uuid4().hex
    workflow = HorusWorkflow.model_validate(
        {
            "kind": "horus_workflow",
            "name": "cancel on the cluster",
            "orchestrator_target": {
                "kind": "local",
                "working_directory": str(tmp_path),
            },
            "tasks": [
                {
                    "id": "long",
                    "kind": "horus_task",
                    "name": "long",
                    "target": {
                        "kind": "slurm",
                        "partition": "cpu",
                        "cpus_per_task": 1,
                        "mem": "100M",
                        "time_limit": "00:05:00",
                        "inner": {
                            "kind": "agent",
                            "agent": AGENT,
                            "working_directory": str(remote),
                        },
                    },
                    "executor": {"kind": "shell"},
                    "runtime": {"kind": "command", "command": "sleep 300"},
                }
            ],
        }
    )
    running = asyncio.create_task(workflow.run(trigger_id="long"))
    job_name = ""
    for _ in range(120):
        names = _docker(
            "exec",
            "-u",
            str(os.getuid()),
            LOGIN,
            "squeue",
            "--me",
            "-h",
            "-t",
            "RUNNING",
            "-o",
            "%j",
        ).split()
        if names:
            job_name = names[0]
            break
        await asyncio.sleep(0.5)
    else:
        pytest.fail("the job never started on the cluster")

    running.cancel()
    with pytest.raises((asyncio.CancelledError, Exception)):
        await asyncio.wait_for(running, 120)

    for _ in range(60):
        if not _squeue(job_name):
            break
        await asyncio.sleep(0.5)
    else:
        pytest.fail(f"{job_name} is still queued after the cancel")
    state = _docker(
        "exec",
        "-u",
        str(os.getuid()),
        LOGIN,
        "sacct",
        "-X",
        "-n",
        "-P",
        "--name",
        job_name,
        "-o",
        "State",
    )
    assert state.startswith("CANCELLED")
