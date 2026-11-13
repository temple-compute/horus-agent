#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
OpRunner, the agent-side dispatcher, without a broker.
"""

import asyncio
import base64
import os
import pwd
import signal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from horus_agent import ops
from horus_agent.ops import OpRunner
from horus_agent.protocol import Op


async def run(runner: OpRunner, method: str, **args: Any) -> Any:
    result = await runner.run(
        Op(op_id=uuid4().hex, method=method, args=args, run_id="r")
    )
    if result.error is not None:
        raise RuntimeError(f"{result.error.type}: {result.error.message}")
    return result.result


async def test_unknown_op_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="Unknown op"):
        await run(OpRunner(), "teleport")


async def test_bad_env_name_is_refused() -> None:
    with pytest.raises(RuntimeError, match="Invalid environment variable"):
        await run(OpRunner(), "exec", cmd="true", env={"A;rm -rf ~": "x"})


async def test_root_is_never_removed() -> None:
    with pytest.raises(RuntimeError, match="Refusing to remove /"):
        await run(OpRunner(), "remove", path="/")


async def test_large_command_output_spills_to_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ops, "INLINE_LIMIT", 4)
    result = await run(OpRunner(), "exec", cmd="echo too-long; echo e >&2")
    path = Path(result["stdout_file"])
    try:
        assert path.read_bytes() == b"too-long\n"
        assert result["stderr"] == "ZQo="  # b"e\n", still inline
    finally:
        path.unlink()


async def test_large_file_reports_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ops, "INLINE_LIMIT", 4)
    (tmp_path / "f").write_bytes(b"12345")
    assert await run(OpRunner(), "get_file", path=str(tmp_path / "f")) == {
        "size": 5
    }


async def test_launch_is_idempotent(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "pid").write_text("4242\n")
    result = await run(OpRunner(), "launch", cmd="false", job_dir=str(job))
    assert result == {"pid": 4242}
    assert not (job / "exit_code").exists(), "nothing was started"


async def test_cancel_run_kills_its_jobs(tmp_path: Path) -> None:
    runner = OpRunner()
    job = str(tmp_path / "job")
    launched = await run(runner, "launch", cmd="sleep 30", job_dir=job)
    assert await run(runner, "cancel_run") == 1
    for _ in range(50):
        status = await run(runner, "poll", pid=launched["pid"], job_dir=job)
        if status is not None:
            break
        await asyncio.sleep(0.1)
    assert status not in (None, 0)
    assert await run(runner, "cancel_run") == 0


async def test_cancel_run_skips_finished_jobs(tmp_path: Path) -> None:
    runner = OpRunner()
    job = str(tmp_path / "job")
    launched = await run(runner, "launch", cmd="true", job_dir=job)
    for _ in range(50):
        if (tmp_path / "job" / "exit_code").exists():
            break
        await asyncio.sleep(0.05)
    assert await run(runner, "cancel_run") == 0
    del launched


async def test_answers_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "RESULT_CACHE_SIZE", 1)
    runner = OpRunner()
    for op_id in ("a", "b"):
        await runner.run(
            Op(op_id=op_id, method="path_exists", args={"path": "/"})
        )
    assert list(runner._answered) == ["b"]


async def test_poll_and_signal_only_own_jobs(tmp_path: Path) -> None:
    runner = OpRunner()
    job = str(tmp_path / "job")
    launched = await run(runner, "launch", cmd="sleep 30", job_dir=job)
    for pid in (os.getpid(), 0, -1):
        for method, extra in (("poll", {}), ("signal", {"sig": 9})):
            with pytest.raises(RuntimeError, match="No job with pid"):
                await run(runner, method, pid=pid, job_dir=job, **extra)
    with pytest.raises(RuntimeError, match="No job with pid"):
        await run(runner, "poll", pid=1, job_dir=str(tmp_path / "other"))
    await run(runner, "signal", pid=launched["pid"], job_dir=job, sig=9)
    for _ in range(50):
        status = await run(runner, "poll", pid=launched["pid"], job_dir=job)
        if status is not None:
            break
        await asyncio.sleep(0.1)
    assert status not in (None, 0)


def test_job_environment_keeps_only_safe_names() -> None:
    environ = {
        "PATH": "/bin",
        "LC_ALL": "C",
        "HTTPS_PROXY": "http://proxy:3128",
        "HORUS_AGENT_CONFIG": "/etc/agent.json",
        "AWS_SECRET_ACCESS_KEY": "s3cr3t",
        "MODULEPATH": "/opt/modules",
    }
    assert ops.job_environment(environ, ("MODULEPATH",)) == {
        "PATH": "/bin",
        "LC_ALL": "C",
        "HTTPS_PROXY": "http://proxy:3128",
        "MODULEPATH": "/opt/modules",
    }


def test_switch_user_refuses_root_and_unknown_methods(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    me = pwd.getpwuid(os.getuid()).pw_name
    with pytest.raises(ValueError, match="root"):
        ops.switch_user("root", "sudo")
    with pytest.raises(ValueError, match="sudo or setpriv"):
        ops.switch_user(me, "su")
    with pytest.raises(KeyError):
        ops.switch_user("no-such-user-horus", "sudo")
    monkeypatch.setattr(ops, "SWITCH_PATH", str(tmp_path))
    with pytest.raises(ValueError, match="not installed"):
        ops.switch_user(me, "setpriv")


def _fake_tool(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}")
    path.chmod(0o755)


@pytest.fixture
def switch_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Fake ``sudo`` and ``setpriv`` in the switcher path: they log their argv
    and environment, then run the command after ``--``.
    """
    bin_dir = tmp_path / "sbin"
    bin_dir.mkdir()
    log = tmp_path / "switch.log"
    for name in ("sudo", "setpriv"):
        _fake_tool(
            bin_dir / name,
            f'echo "{name} $*" >> {log}\n'
            f"/usr/bin/env >> {log}\n"
            'while [ "$1" != "--" ]; do shift; done\n'
            'shift\nexec "$@"\n',
        )
    monkeypatch.setattr(ops, "SWITCH_PATH", str(bin_dir))
    return log


@pytest.mark.parametrize("via", ["sudo", "setpriv"])
async def test_run_as_switches_user_without_secrets_in_argv(
    via: str, switch_log: Path, tmp_path: Path
) -> None:
    me = pwd.getpwuid(os.getuid())
    secret = f"s3cr3t-{uuid4().hex}"  # unique: found only where it leaked
    runner = OpRunner(run_as=me.pw_name, run_as_via=via)
    result = await run(
        runner,
        "exec",
        cmd='echo "$TASK_SECRET $HOME $USER"; umask',
        env={"TASK_SECRET": secret},
        cwd=str(tmp_path),
    )
    out = base64.b64decode(result["stdout"]).decode().split()
    assert out == [secret, me.pw_dir, me.pw_name, "0007"]
    logged = switch_log.read_text()
    assert logged.startswith(f"{via} ")
    # Neither the switcher's argv nor its environment holds task values.
    assert secret not in logged

    job = str(tmp_path / "job")
    launched = await run(
        runner,
        "launch",
        cmd='echo "$TASK_SECRET"; sleep 30',
        env={"TASK_SECRET": secret},
        job_dir=job,
    )
    for _ in range(50):
        if (tmp_path / "job" / "stdout.log").read_text():
            break
        await asyncio.sleep(0.1)
    assert (tmp_path / "job" / "stdout.log").read_text() == f"{secret}\n"
    assert secret not in switch_log.read_text()
    _not_in_any_cmdline(secret)
    assert await run(runner, "cancel_run") == 1
    for _ in range(50):
        status = await run(runner, "poll", pid=launched["pid"], job_dir=job)
        if status is not None:
            break
        await asyncio.sleep(0.1)
    assert status not in (None, 0)
    assert f"kill -{signal.SIGKILL}" in switch_log.read_text()


@pytest.mark.parametrize("via", ["sudo", "setpriv"])
async def test_task_env_cannot_replace_the_switcher(
    via: str, switch_log: Path, tmp_path: Path
) -> None:
    evil = tmp_path / "evil"
    evil.mkdir()
    hijacked = tmp_path / "hijacked"
    # Only the switch must ignore it: after it, PATH is the task's own.
    for name in ("sudo", "setpriv"):
        _fake_tool(evil / name, f"touch {hijacked}\n")
    runner = OpRunner(pwd.getpwuid(os.getuid()).pw_name, via)
    env = {
        "PATH": f"{evil}:/usr/bin:/bin",
        "LD_PRELOAD": str(evil / "nothing.so"),
        "BASH_ENV": str(evil / "sudo"),
        "ENV": str(evil / "sudo"),
    }
    result = await run(runner, "exec", cmd="echo ok", env=env)
    assert base64.b64decode(result["stdout"]) == b"ok\n"
    job = str(tmp_path / "job")
    await run(runner, "launch", cmd="true", env=env, job_dir=job)
    for _ in range(50):
        if (tmp_path / "job" / "exit_code").exists():
            break
        await asyncio.sleep(0.1)
    assert not hijacked.exists()
    assert switch_log.read_text().count(f"{via} ") == 2


def _not_in_any_cmdline(secret: str) -> None:
    """No process on this machine shows *secret* in its argv."""
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmdline = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        assert secret.encode() not in cmdline, cmdline


async def test_job_files_are_never_followed(tmp_path: Path) -> None:
    runner = OpRunner()
    job = tmp_path / "job"
    job.mkdir()
    secret = tmp_path / "agent.json"
    secret.write_text("tca_secret")
    (job / "stdout.log").symlink_to(secret)
    with pytest.raises(RuntimeError, match="symlink"):
        await run(
            runner,
            "read_output",
            job_dir=str(job),
            stdout_offset=0,
            stderr_offset=0,
        )
    (job / "stdout.log").unlink()
    os.mkfifo(job / "stdout.log")
    with pytest.raises(RuntimeError, match="regular file"):
        await run(
            runner,
            "read_output",
            job_dir=str(job),
            stdout_offset=0,
            stderr_offset=0,
        )
    (job / "pid").symlink_to(secret)
    with pytest.raises(RuntimeError, match="symlink"):
        await run(runner, "launch", cmd="true", job_dir=str(job))
    linked = tmp_path / "linked"
    linked.symlink_to(job)
    with pytest.raises(RuntimeError, match="symlink"):
        await run(runner, "poll", pid=None, job_dir=str(linked))


async def test_signal_never_reaches_the_agent(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    # A job dir recording the agent's own pid, as a task could write it.
    (job / "pid").write_text(f"{os.getpid()}\n")
    with pytest.raises(RuntimeError, match="agent's process group"):
        await run(
            OpRunner(), "signal", pid=os.getpid(), job_dir=str(job), sig=15
        )
