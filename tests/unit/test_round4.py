#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Regression tests for the fourth review round of the policy work."""

import asyncio
import base64
import subprocess
from pathlib import Path
from typing import Any

import pytest

from horus_agent import cli, launch, policy
from horus_agent import guard as guard_mod
from horus_agent.manifest import (
    build_manifest_from_workflow,
    task_runs_python_code,
)
from horus_agent.ops import OpRunner
from horus_agent.policy import PolicyError
from horus_agent.protocol import PolicyRefusedError
from tests.unit.test_guard import RUN1, Env, env, fake_slurm, refused
from tests.unit.test_policy_hardening import launch_t, manifest, wait_exit

__all__ = ["env", "fake_slurm"]

DATA_B64 = base64.b64encode(b"x").decode()


# --- B1: no code the cloud wrote in a run -----------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "json/__init__.py",
        "tool.py",
        "a.pyc",
        "a.pyo",
        "lib.so",
        "lib.so.1.2",
        "a.pyd",
        "a.dll",
        "a.dylib",
        "a.jar",
        "Main.class",
        "a.lua",
        "a.rb",
        "a.pl",
        "a.pm",
        "a.js",
        "a.mjs",
        "a.cjs",
        "a.node",
        "a.R",
        "a.r",
        "A.PY",
        "site-packages/x",
        "__pycache__/x",
        "node_modules/x",
    ],
)
async def test_code_files_cannot_be_staged(env: Env, name: str) -> None:
    path = str(env.run / name)
    await refused(env, "put_file", "", path=path, data=DATA_B64)
    await refused(
        env, "fetch", "", path=path, url="https://files.example.org/x"
    )
    await refused(env, "mkdir", "", path=path)


async def test_the_refusal_names_the_file_and_the_setting(env: Env) -> None:
    message = await refused(
        env,
        "put_file",
        "tool.py",
        path=str(env.run / "tool.py"),
        data=DATA_B64,
    )
    assert "allow_code_suffixes" in message and '".py"' in message


async def test_packages_and_moves_cannot_bring_code_in(env: Env) -> None:
    import io
    import tarfile

    pkg = env.run / "p.tgz"
    with tarfile.open(pkg, "w:gz") as tar:
        info = tarfile.TarInfo("sub/helper.js")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"1"))
    message = await refused(
        env, "unpack", "helper.js", pkg=str(pkg), dest=str(env.run / "o")
    )
    assert "allow_code_suffixes" in message
    (env.run / "d").write_text("x")
    await refused(
        env, "move", "", src=str(env.run / "d"), dst=str(env.run / "d.rb")
    )


async def test_the_policy_can_allow_a_suffix(env: Env) -> None:
    env.with_policy(allow_code_suffixes=(".py",))
    await env.call("put_file", path=str(env.run / "data.py"), data=DATA_B64)
    await refused(
        env, "put_file", ".js", path=str(env.run / "a.js"), data=DATA_B64
    )
    await refused(env, "mkdir", "", path=str(env.run / "__pycache__"))


async def test_the_launch_scan_goes_below_the_cwd(env: Env) -> None:
    approved = manifest("python3 -m json.tool ${inp} > ${out}")
    policy.approve(approved, "root")
    deep = env.run / "a" / "b" / "json"
    deep.mkdir(parents=True)
    (deep / "__init__.py").write_text("open('PWNED', 'w').write('x')\n")
    (env.run / "in.json").write_text("{}")
    await refused(
        env,
        "task_launch",
        "__init__.py",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(env.run / "j"),
        cwd=str(env.run),
        bindings={
            "out": {"path": str(env.run / "o")},
            "inp": {"path": str(env.run / "in.json")},
        },
        env={},
    )
    assert not (env.run / "PWNED").exists()


@pytest.mark.parametrize(
    ("version", "ok"), [((3, 10), False), ((3, 12), True)]
)
async def test_old_python_with_code_in_the_cwd_does_not_start(
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int, int],
    ok: bool,
) -> None:
    env.with_policy(allow_code_suffixes=(".py",))

    async def probe() -> tuple[int, int]:
        return version

    monkeypatch.setattr(env.runner, "_python_version", probe)
    approved = manifest("python3 -m json.tool ${inp} > ${out}")
    assert task_runs_python_code(approved.tasks["t"])
    policy.approve(approved, "root")
    (env.run / "json").mkdir()
    (env.run / "json" / "__init__.py").write_text("")
    (env.run / "in.json").write_text("{}")
    call = launch_t(
        env, approved, env.run, "j", inp={"path": str(env.run / "in.json")}
    )
    if ok:
        await call
        await wait_exit(env.run / "j")
    else:
        with pytest.raises(RuntimeError, match=r"older than 3\.11"):
            await call


async def test_the_python_probe_reads_the_job_pythons(env: Env) -> None:
    version = await env.runner._python_version()
    assert version is None or version >= (3, 0)


def test_python_code_is_recorded_per_task() -> None:
    for command in (
        "python3 -m pytest",
        "python -c 'print(1)'",
        "conda run -n e python3 -c x",
        "python3 -mjson.tool",
    ):
        assert task_runs_python_code(manifest(command).tasks["t"]), command
    for command in ("python3 script.py", "bash -c 'python3 -c x'"):
        assert task_runs_python_code(manifest(command).tasks["t"]), command
    assert not task_runs_python_code(manifest("ls").tasks["t"])
    scripted = build_manifest_from_workflow(
        {
            "name": "w",
            "tasks": [
                {
                    "id": "t",
                    "target": {"kind": "agent", "agent": "lab"},
                    "executor": {"kind": "shell"},
                    "runtime": {
                        "kind": "python_script",
                        "script": "s.py",
                        "args": "-c x",
                    },
                }
            ],
        },
        {"s.py": b"1"},
    )
    assert not task_runs_python_code(scripted.tasks["t"])


def test_approve_refuses_python_dash_m_on_an_old_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved = manifest("python3 -m json.tool")
    monkeypatch.setattr(policy, "configured_code_suffixes", lambda: (".py",))

    def old(*_a: Any, **_k: Any) -> Any:
        return subprocess.CompletedProcess([], 0, stdout="3 10\n")

    monkeypatch.setattr(subprocess, "run", old)
    assert "older than 3.11" in str(cli._python_problem(approved))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, stdout="3 12\n"),
    )
    assert cli._python_problem(approved) is None
    monkeypatch.setattr(policy, "configured_code_suffixes", lambda: ())
    assert cli._python_problem(approved) is None


def test_allow_code_suffixes_is_parsed(
    policy_dir: Path, tmp_path: Path
) -> None:
    path = policy_dir / "policy.toml"
    path.write_text('allow_code_suffixes = [".py", ".R"]\n')
    path.chmod(0o644)
    rules = policy.load_policy(str(tmp_path), "https://api.example.org/x")
    assert rules.allow_code_suffixes == (".py", ".R")
    path.write_text('allow_code_suffixes = ["py"]\n')
    with pytest.raises(PolicyError, match="allow_code_suffixes"):
        policy.load_policy(str(tmp_path), "https://api.example.org/x")


def test_the_job_env_keeps_pythonsafepath() -> None:
    assert launch.job_env(Path("/h"))["PYTHONSAFEPATH"] == "1"


# --- approved scripts and constants still stage -----------------------------


def _script_release() -> Any:
    return manifest(
        "",
        runtime={
            "kind": "python_script",
            "python": "python3",
            "script": "qc.py",
            "args": "",
        },
        inputs=[],
        outputs=[],
        files={"qc.py": b"print(1)\n"},
    )


async def test_an_approved_script_stages_by_its_content(env: Env) -> None:
    approved = _script_release()
    policy.approve(approved, "root")
    where = str(env.run / "scripts" / "qc.py")
    await refused(
        env,
        "put_file",
        "allow_code_suffixes",
        release=approved.release_hash,
        path=where,
        data=base64.b64encode(b"print(2)\n").decode(),
    )
    await env.call(
        "put_file",
        release=approved.release_hash,
        path=where,
        data=base64.b64encode(b"print(1)\n").decode(),
    )
    job = env.run / "j"
    await env.call(
        "task_launch",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(job),
        cwd=str(env.run),
        script=where,
    )
    await wait_exit(job)
    assert (job / "stdout.log").read_text() == "1\n"


async def test_an_approved_constant_folder_unpacks_with_its_code(
    env: Env,
) -> None:
    import io
    import tarfile

    approved = manifest(
        "ls ${inp} > ${out}",
        inputs=[{"id": "inp", "kind": "folder", "path": "tools"}],
        files={"tools/main.py": b"print(1)\n"},
    )
    policy.approve(approved, "root")

    def package(name: str, body: bytes) -> Path:
        pkg = env.run / f"{name}.tgz"
        with tarfile.open(pkg, "w:gz") as tar:
            info = tarfile.TarInfo("main.py")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
        return pkg

    await refused(
        env,
        "unpack",
        "main.py",
        release=approved.release_hash,
        pkg=str(package("bad", b"print(2)\n")),
        dest=str(env.run / "tools"),
    )
    await env.call(
        "unpack",
        release=approved.release_hash,
        pkg=str(package("good", b"print(1)\n")),
        dest=str(env.run / "tools"),
    )
    assert (env.run / "tools" / "main.py").read_text() == "print(1)\n"
    # later writes into the approved folder are checked again
    await refused(
        env,
        "put_file",
        "allow_code_suffixes",
        release=approved.release_hash,
        path=str(env.run / "tools" / "evil.py"),
        data=DATA_B64,
    )
    job = env.run / "j"
    await env.call(
        "task_launch",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(job),
        cwd=str(env.run),
        bindings={
            "inp": {"path": str(env.run / "tools")},
            "out": {"path": str(env.run / "o")},
        },
    )
    await wait_exit(job)


# --- M2: cloud files land only at declared names ----------------------------


def _layout_release() -> Any:
    return build_manifest_from_workflow(
        {
            "name": "w",
            "artifacts": [
                {"id": "ref", "kind": "file", "path": "refs/ref.fa"}
            ],
            "tasks": [
                {
                    "id": "align",
                    "target": {"kind": "agent", "agent": "lab"},
                    "executor": {"kind": "shell"},
                    "runtime": {"kind": "command", "command": "ls ${reads}"},
                    "inputs": [
                        {"id": "reads", "kind": "file", "path": "inputs/r.fq"}
                    ],
                    "outputs": [
                        {"id": "bam", "kind": "file", "path": "results/a.bam"}
                    ],
                }
            ],
        },
        {},
    )


def test_the_layout_lists_what_the_release_declares() -> None:
    assert set(_layout_release().layout) >= {
        "align",
        "inputs",
        "r.fq",
        "results",
        "a.bam",
        "refs",
        "ref.fa",
    }


async def test_only_declared_names_can_be_staged(env: Env) -> None:
    approved = _layout_release()
    policy.approve(approved, "root")
    release = approved.release_hash
    ok = base64.b64encode(b"x").decode()
    await env.call(
        "put_file",
        release=release,
        path=str(env.run / "inputs" / "r.fq"),
        data=ok,
    )
    await env.call(
        "put_file",
        release=release,
        path=str(env.run / "align" / "results" / "a.bam"),
        data=ok,
    )
    for path in (
        env.run / "Rakefile",
        env.run / "lib" / "x.txt",
        env.run / "align" / "Taskfile.yml",
        env.run / "align" / "data" / "x",
    ):
        message = await refused(
            env,
            "put_file",
            "not an input, output",
            release=release,
            path=str(path),
            data=ok,
        )
        assert path.parts[-1] in message or path.parts[-2] in message
    await refused(
        env, "mkdir", "not an input", release=release, path=str(env.run / "z")
    )
    await refused(
        env,
        "move",
        "not an input",
        release=release,
        src=str(env.run / "inputs" / "r.fq"),
        dst=str(env.run / "Justfile"),
    )


async def test_a_launch_refuses_an_undeclared_name_in_the_cwd(
    env: Env,
) -> None:
    approved = _layout_release()
    policy.approve(approved, "root")
    cwd = env.run / "align"
    cwd.mkdir()
    (cwd / "Taskfile.yml").write_text("x")
    await refused(
        env,
        "task_launch",
        "Taskfile.yml",
        release=approved.release_hash,
        task="align",
        task_id="align",
        job_dir=str(cwd / ".horus_job" / "j"),
        cwd=str(cwd),
        bindings={
            "reads": {"path": str(env.run / "inputs" / "r.fq")},
        },
        env={},
    )


def test_the_layout_is_stored_with_the_approval(
    env: Env,  # noqa: ARG001
) -> None:
    approved = _layout_release()
    path = policy.approve(approved, "root")
    from horus_agent.manifest import Manifest

    assert Manifest.from_json(path.read_text()).layout == approved.layout


# --- minors -----------------------------------------------------------------


@pytest.mark.parametrize("name", [".horusrc", ".horus_x", ".horus-agent2"])
async def test_only_the_runtimes_own_hidden_names_pass(
    env: Env, name: str
) -> None:
    await refused(env, "put_file", "", path=str(env.run / name), data=DATA_B64)
    await env.call("put_file", path=str(env.run / "o"), data=DATA_B64)


def test_the_approve_screen_leads_with_the_warnings() -> None:
    approved = manifest("cat ${inp}", inputs=INPUTS_FOR_SHOW)
    shown = cli._show(approved)
    assert shown.index("Warning:") < shown.index("Task t")
    assert "agent tasks," in shown.splitlines()[1]


INPUTS_FOR_SHOW = [{"id": "inp", "kind": "file"}]


def test_approve_reads_a_bounded_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big = tmp_path / "r.zip"
    big.write_bytes(b"x" * 10)
    monkeypatch.setattr(cli, "MAX_DOWNLOAD_BYTES", 5)
    with pytest.raises(ValueError, match="over 5 bytes"):
        cli._archive(str(big))


async def test_a_job_dir_is_recorded_before_the_launch_awaits(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = manifest("true", inputs=[], outputs=[])
    policy.approve(approved, "root")
    seen: dict[str, Any] = {}
    job = env.run / "j"

    async def spy(*_a: Any, **_k: Any) -> dict[str, int]:
        seen["recorded"] = env.guard.job(RUN1, job) is not None
        try:
            env.guard.check_writable(
                RUN1, job / "home" / "R" / "lib" / "x", "put_file"
            )
        except PolicyRefusedError as exc:
            seen["refused"] = str(exc)
        return {"pid": 1}

    monkeypatch.setattr(env.runner, "_launch", spy)
    await env.call(
        "task_launch",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(job),
        cwd=str(env.run),
        bindings={},
    )
    assert seen["recorded"] and "holds a job" in seen["refused"]
    assert (job / "home").stat().st_mode & 0o777 == 0o700


async def test_digest_is_answered_for_what_the_run_staged(env: Env) -> None:
    (env.run / "results").mkdir()
    planted = env.run / "d"
    planted.write_text("secret")
    await refused(env, "digest", "was not staged", path=str(planted))
    with pytest.raises(RuntimeError, match="FileNotFoundError"):
        await env.call("digest", path=str(env.run / "gone"))
    await env.call("put_file", path=str(env.run / "f"), data=DATA_B64)
    assert await env.call("digest", path=str(env.run / "f"))
    env.with_policy(uploads=True)
    assert await env.call("digest", path=str(planted))


# --- cancel_run kills only a group the recorded pid leads -------------------


async def test_cancel_run_spares_a_group_the_pid_no_longer_leads(
    env: Env,
) -> None:
    import os
    import signal
    import subprocess

    mine = subprocess.Popen(["sleep", "60"], start_new_session=True)
    other = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        await env.call("mkdir", path=str(env.run / "o"))
        # a recorded pid that left the recorded group (pid reuse)
        env.guard.add_job(RUN1, env.run / "j1", pid=mine.pid, pgid=other.pid)
        await env.call("cancel_run", release="basic")
        assert other.poll() is None and mine.poll() is None
        env.guard.add_job(RUN1, env.run / "j2", pid=mine.pid, pgid=mine.pid)
        assert await env.call("cancel_run", release="basic") == 1
        mine.wait(timeout=5)
        assert other.poll() is None
    finally:
        for proc in (mine, other):
            with __import__("contextlib").suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


# --- the typed Slurm ops ----------------------------------------------------


def _slurm_args(env: Env, job: str) -> dict[str, Any]:
    return {
        "task_id": "big",
        "job_dir": str(env.run / ".horus_job" / job),
        "cwd": str(env.run),
        "bindings": {
            "data": {"path": str(env.run / "data")},
            "model": {"path": str(env.run / "m.bin")},
        },
    }


@pytest.mark.usefixtures("fake_slurm")
async def test_cancel_run_cancels_the_slurm_jobs_of_the_run(
    env: Env, fake_slurm: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_SLURM_PENDING_S", "30")
    env.runner = OpRunner(
        job_env=("FAKE_SLURM_STATE", "FAKE_SLURM_PENDING_S"), guard=env.guard
    )
    result = await env.call(
        "slurm_submit", release="slurm", task="big", **_slurm_args(env, "s1")
    )
    record = env.guard.job(RUN1, env.run / ".horus_job" / "s1")
    assert record is not None and guard_mod.job_running(
        env.run / ".horus_job" / "s1", record
    )
    await env.call("cancel_run", release="slurm")
    assert result["job_id"] in (fake_slurm / "scancel.log").read_text()
    record = env.guard.job(RUN1, env.run / ".horus_job" / "s1")
    assert record is not None
    assert not guard_mod.job_running(env.run / ".horus_job" / "s1", record)


@pytest.mark.usefixtures("fake_slurm")
async def test_a_slurm_job_that_left_the_queue_holds_no_run_open(
    env: Env,
) -> None:
    await env.call(
        "slurm_submit", release="slurm", task="big", **_slurm_args(env, "s2")
    )
    for _ in range(100):
        status = await env.call(
            "slurm_status",
            release="slurm",
            job_dir=str(env.run / ".horus_job" / "s2"),
        )
        if status["final"] is not None:
            break
        await asyncio.sleep(0.05)
    assert status["final"]["state"] in ("COMPLETED", "FAILED")
    record = env.guard.job(RUN1, env.run / ".horus_job" / "s2")
    assert record is not None
    assert not env.guard._active(RUN1, env.guard.run_record(RUN1) or {})
    assert not guard_mod.job_running(env.run / ".horus_job" / "s2", record)
    await env.call(
        "remove", release="slurm", path=str(env.run / ".horus_job" / "s2")
    )  # a finished job's dir may go
    assert not (env.run / ".horus_job" / "s2").exists()
