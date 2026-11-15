#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Regressions found by running a strict agent against a real backend."""

import contextlib
import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

from horus_agent import cli, config, native
from horus_agent import guard as guard_mod
from horus_agent.cli import main
from horus_agent.config import AgentConfig
from tests.unit.test_guard import RUN1, Env, env, refused

__all__ = ["env"]


async def test_a_later_task_may_make_the_dir_above_an_earlier_job(
    env: Env,
) -> None:
    await env.call("mkdir", path=str(env.run / "o"))
    job = env.run / "o" / "j1"
    env.guard.add_job(RUN1, job, started=time.time())
    await env.call("mkdir", path=str(env.run))
    await env.call("mkdir", path=str(env.run / "o"))
    await refused(env, "mkdir", "holds a job", path=str(job))
    await refused(env, "mkdir", "holds a job", path=str(job / "x"))


def _zombie() -> int:
    """A child that exited and was never reaped."""
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    while Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] != "Z":
        time.sleep(0.01)
    return pid


async def test_a_job_that_died_without_an_exit_code_is_over(
    env: Env,
) -> None:
    pid = _zombie()
    await env.call("mkdir", path=str(env.run / "o"))
    job = env.run / "j1"
    job.mkdir(parents=True)
    (job / "pid").write_text(str(pid))
    env.guard.add_job(RUN1, job, pid=pid, pgid=pid)
    try:
        assert await env.call("poll", pid=pid, job_dir=str(job)) == -1
        out = await env.call(
            "read_output", job_dir=str(job), stdout_offset=0, stderr_offset=0
        )
        assert out["exited"] is True
        assert not env.guard.running(RUN1)
    finally:
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


async def test_cancel_run_ends_the_job_and_poll_answers(env: Env) -> None:
    await env.call("mkdir", path=str(env.run / "o"))
    job = env.run / "j1"
    job.mkdir(parents=True)
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    (job / "pid").write_text(str(proc.pid))
    env.guard.add_job(
        RUN1,
        job,
        pid=proc.pid,
        pgid=proc.pid,
        start=__import__("horus_agent.ops").ops._proc_start(proc.pid),
    )
    try:
        assert await env.call("poll", pid=proc.pid, job_dir=str(job)) is None
        assert await env.call("cancel_run") == 1
        for _ in range(100):
            if await env.call("poll", pid=proc.pid, job_dir=str(job)) == -1:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("poll never saw the killed job")
        assert not env.guard.running(RUN1)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def test_a_copied_folder_keeps_the_group_of_its_parent(tmp_path: Path) -> None:
    other = [g for g in os.getgroups() if g != os.getgid()]
    parent = tmp_path / "store"
    parent.mkdir()
    if other:
        os.chown(parent, -1, other[0])
    parent.chmod(0o2750)
    src = tmp_path / "src"
    (src / "sub" / "deep").mkdir(parents=True)
    (src / "sub" / "deep" / "f.txt").write_text("x")
    native._copy_tree(src, parent / "dst")
    for path in (parent / "dst").rglob("*"):
        info = path.stat()
        assert info.st_gid == parent.stat().st_gid
        if path.is_dir():
            assert info.st_mode & stat.S_ISGID, path
        else:
            assert info.st_mode & stat.S_IRGRP, path


def test_register_takes_the_job_group_when_the_agent_belongs_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = [g for g in os.getgroups() if g != os.getgid()]
    if not other:
        pytest.skip("needs a second group")
    monkeypatch.setattr(cli, "_job_group", lambda _jobs: ("jobs", other[0]))
    work = tmp_path / "work"
    jobs = config.JobsPolicy(run_as="someone")
    cli._work_dir(str(work), jobs)
    assert work.stat().st_gid == other[0]
    assert stat.S_IMODE(work.stat().st_mode) == 0o3770
    assert cli.layout_problem(work, jobs) is None


def test_doctor_reports_one_line_per_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "cfg" / "agent.json"
    monkeypatch.setenv(config.CONFIG_ENV, str(path))
    work = tmp_path / "work"
    work.mkdir()
    work.chmod(0o770)
    config.save(
        AgentConfig(
            url="https://b",
            agent_id="i",
            name="n",
            token="t",
            work_dir=str(work),
        ),
        path,
    )
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert f"work dir {work} mode 770" in out and "FAIL" not in out
    monkeypatch.setattr(
        config, "load_jobs_policy", lambda: config.JobsPolicy(run_as="nosuch9")
    )
    assert main(["doctor"]) == 1
    assert "FAIL  run_as user" in capsys.readouterr().out
    work.rmdir()
    assert main(["doctor"]) == 1


def test_gpu_file_reads_are_routine() -> None:
    assert ".horus_gpu.csv" in guard_mod.MONITOR_READS
