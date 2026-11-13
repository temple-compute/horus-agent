#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
What a compromised orchestrator or broker still cannot do to a strict
agent: inject through bindings, reach another run, swap a verified input,
read data back through metadata names, plant job bookkeeping.
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import Any, cast

import pytest

from horus_agent import cli, launch, policy
from horus_agent.manifest import (
    Manifest,
    ManifestError,
    build_manifest_from_workflow,
)
from horus_agent.native import digest
from horus_agent.ops import OpRunner
from horus_agent.protocol import PolicyRefusedError
from tests.unit.test_guard import RUN1, RUN2, Env, env, refused

__all__ = ["env"]  # the fixture


def manifest(command: str, **task: Any) -> Manifest:
    """One approved shell task running *command*."""
    spec: dict[str, Any] = {
        "id": "t",
        "target": {"kind": "agent", "agent": "lab"},
        "executor": {"kind": "shell"},
        "runtime": {"kind": "command", "command": command},
        "inputs": [{"id": "inp", "kind": "file"}],
        "outputs": [{"id": "out", "kind": "file", "path": "o"}],
    }
    spec.update(task)
    files = spec.pop("files", {})
    return build_manifest_from_workflow({"name": "w", "tasks": [spec]}, files)


async def wait_exit(job: Path) -> None:
    for _ in range(200):
        if (job / "exit_code").exists():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{job} did not finish")


# --- placeholders render as single shell words ------------------------------


@pytest.mark.parametrize(
    "command",
    [
        'cat "${inp}" > ${out}',
        "sh -c 'cat ${inp}' > ${out}",
        "cat ${inp} > ${out} <<EOF\n${inp}\nEOF",
        "eval cat ${inp} > ${out}",
        "bash -c ${inp} > ${out}",
        "cat ${HOME:-${inp}} > ${out}",
        "echo $(( ${inp} )) > ${out}",
        "echo \\${inp} > ${out}",
        "cat '${inp} > ${out}",
        'echo "${task.id}" ${inp} > ${out}',
    ],
)
def test_placeholders_outside_bare_words_are_refused(command: str) -> None:
    with pytest.raises(ManifestError, match=r"plain argument|quoting"):
        manifest(command)


@pytest.mark.parametrize(
    "command",
    [
        "cat ${inp} > ${out}",
        "cp --from=${inp}.x ${out}",
        "basename $(dirname ${inp}) > ${out}",
        'printf "%s" "$HOME" ${inp} > ${out}',
        "cat `echo ${inp}` > ${out} # ${inp} in a comment",
    ],
)
def test_bare_placeholders_are_accepted(command: str) -> None:
    manifest(command)


async def test_path_bindings_cannot_carry_shell_syntax(env: Env) -> None:
    approved = manifest("cat ${inp} > ${out}")
    policy.approve(approved, "root")
    evil = env.run / "$(touch PWNED)"
    evil.write_text("x")
    args: dict[str, Any] = {
        "task_id": "t",
        "job_dir": str(env.run / "job"),
        "cwd": str(env.run),
        "bindings": {
            "inp": {"path": str(evil)},
            "out": {"path": str(env.run / "o")},
        },
    }
    await refused(
        env,
        "task_launch",
        "characters",
        release=approved.release_hash,
        task="t",
        **args,
    )
    args["task_id"] = "t[$(id)]"
    args["bindings"]["inp"] = {"path": str(env.run / "plain")}
    await refused(
        env,
        "task_launch",
        "Invalid task id",
        release=approved.release_hash,
        task="t[$(id)]",
        **args,
    )
    assert not list(env.work.rglob("PWNED"))


# --- one run cannot reach another ------------------------------------------


async def test_run_ids_are_opaque(env: Env) -> None:
    (env.run / "secret.txt").write_text("x")
    data = base64.b64encode(b"overwritten").decode()
    for run in ("proj", "work", "..", "AAAAAAAA-1111-4111-8111-111111111111"):
        await refused(
            env,
            "put_file",
            "run id",
            run=run,
            path=str(env.run / "secret.txt"),
            data=data,
        )
    assert (env.run / "secret.txt").read_text() == "x"


async def test_runs_folders_never_overlap(env: Env) -> None:
    await env.call("mkdir", path=str(env.run / "a"))
    nested = env.run / RUN2 / "x"
    await refused(env, "mkdir", "overlaps", run=RUN2, path=str(nested))
    await refused(
        env, "put_file", "outside", run=RUN2, path=str(env.run / "a"), data=""
    )
    # A run whose folder came first keeps it from the one above.
    other = env.work / "q" / RUN2
    await env.call("mkdir", run=RUN2, path=str(other / "b"))
    run3 = "33333333-3333-4333-8333-333333333333"
    os.makedirs(other / run3)
    await refused(env, "mkdir", "overlaps", run=run3, path=str(other / run3))


# --- verified inputs are the agent's own copies -----------------------------


async def test_sibling_modules_of_the_script_do_not_load(env: Env) -> None:
    approved_src = b"import json\nprint(json.dumps({'ok': 1}))\n"
    approved = manifest(
        "",
        runtime={
            "kind": "python_script",
            "python": "python3",
            "script": "qc.py",
            "args": "",
        },
        inputs=[],
        outputs=[],
        files={"qc.py": approved_src},
    )
    policy.approve(approved, "root")
    folder = env.run / "s"
    folder.mkdir()
    (folder / "qc.py").write_bytes(approved_src)
    (folder / "json.py").write_text("open('PWNED', 'w')\ndumps = str\n")
    (env.run / "json.py").write_text("open('PWNED', 'w')\ndumps = str\n")
    job = env.run / "jq"
    await env.call(
        "task_launch",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(job),
        cwd=str(env.run),
        script=str(folder / "qc.py"),
    )
    await wait_exit(job)
    assert (job / "stdout.log").read_text() == '{"ok": 1}\n'
    assert not list(env.work.rglob("PWNED"))


async def test_constant_folders_hold_files_and_folders_only(env: Env) -> None:
    approved = manifest(
        "ls ${inp} > ${out}",
        inputs=[{"id": "inp", "kind": "folder", "path": "tools"}],
        files={"tools/main.py": b"print(1)\n"},
    )
    policy.approve(approved, "root")
    tools = env.run / "tools"
    tools.mkdir()
    (tools / "main.py").write_bytes(b"print(1)\n")
    (tools / "argparse.py").symlink_to(env.run / "payload.py")
    assert (
        digest(tools)
        == cast(Any, approved.task("t"))["inputs"]["inp"]["sha256"]
    )
    args: dict[str, Any] = {
        "task_id": "t",
        "job_dir": str(env.run / "j"),
        "cwd": str(env.run),
        "bindings": {
            "inp": {"path": str(tools)},
            "out": {"path": str(env.run / "o")},
        },
    }
    await refused(
        env,
        "task_launch",
        "not a file or folder",
        release=approved.release_hash,
        task="t",
        **args,
    )
    (tools / "argparse.py").unlink()
    await env.call(
        "task_launch", release=approved.release_hash, task="t", **args
    )
    await wait_exit(env.run / "j")
    assert (env.run / "o").read_text() == "main.py\n"


# --- metadata is decided by who wrote it ------------------------------------


async def test_data_cannot_leave_under_a_metadata_name(env: Env) -> None:
    assert env.guard.policy.uploads is False
    (env.run / "results").mkdir()
    (env.run / "results" / "patient.vcf").write_text("PATIENT-123\n")
    for dst in (".horus/x.json", "side/resource-usage.csv", "j/pid"):
        await refused(
            env,
            "move",
            "",
            src=str(env.run / "results" / "patient.vcf"),
            dst=str(env.run / dst),
        )
    # Written by anything but the orchestrator's put_file: not metadata.
    (env.run / ".horus").mkdir()
    (env.run / ".horus" / "y.json").write_text("PATIENT-123")
    await refused(
        env, "get_file", "stays here", path=str(env.run / ".horus/y.json")
    )
    other = env.work / "proj" / RUN2
    args: dict[str, Any] = {
        "task_id": "say",
        "job_dir": str(other / "job"),
        "cwd": str(other),
        "bindings": {
            "msg": {"value": "x"},
            "out": {"path": str(other / "x" / "resource-usage.csv")},
        },
        "env": {},
    }
    await refused(
        env,
        "task_launch",
        "metadata",
        run=RUN2,
        release="echo",
        task="say",
        **args,
    )


async def test_logs_off_holds_with_uploads_on(env: Env) -> None:
    env.with_policy(uploads=True)
    job = env.run / "job"
    args: dict[str, Any] = {
        "task_id": "say",
        "job_dir": str(job),
        "cwd": str(env.run),
        "bindings": {
            "msg": {"value": "x"},
            "out": {"path": str(env.run / "o")},
        },
        "env": {},
    }
    await env.call("task_launch", release="echo", task="say", **args)
    await wait_exit(job)
    for name in ("stdout.log", "stderr.log"):
        await refused(
            env,
            "get_file",
            "Logs are off",
            release="echo",
            path=str(job / name),
        )
    code = await env.call(
        "get_file", release="echo", path=str(job / "exit_code")
    )
    assert base64.b64decode(code["data"]).strip() == b"0"


# --- job bookkeeping is the agent's -----------------------------------------


async def test_planted_job_files_are_never_trusted(env: Env) -> None:
    job = env.run / "job"
    job.mkdir()
    (job / "pid").write_text(f"{os.getpid()}\n")
    args: dict[str, Any] = {
        "task_id": "say",
        "job_dir": str(job),
        "cwd": str(env.run),
        "bindings": {
            "msg": {"value": "x"},
            "out": {"path": str(env.run / "o")},
        },
        "env": {},
    }
    await refused(
        env,
        "task_launch",
        "already exists",
        release="echo",
        task="say",
        **args,
    )
    await refused(
        env, "signal", "No job", release="echo", job_dir=str(job), sig=9
    )
    # A job the agent started: its pid comes from the agent's record.
    args["job_dir"] = str(env.run / "job2")
    first = await env.call("task_launch", release="echo", task="say", **args)
    await wait_exit(env.run / "job2")
    (env.run / "job2" / "pid").write_text(f"{os.getpid()}\n")
    assert (
        cast(Any, env.guard.job(RUN1, env.run / "job2"))["pid"] == first["pid"]
    )
    await refused(
        env,
        "unpack",
        "holds a job",
        release="echo",
        pkg=str(env.run / "p.tgz"),
        dest=str(env.run / "job2"),
    )


# --- runs, revocation -------------------------------------------------------


async def test_finished_runs_do_not_count_against_the_cap(env: Env) -> None:
    env.with_policy(max_concurrent_runs=2)
    runs = [f"{n:08d}-0000-4000-8000-000000000000" for n in range(5)]
    for run in runs:
        folder = env.work / "proj" / run
        job = folder / "job"
        await env.call(
            "task_launch",
            run=run,
            release="echo",
            task="say",
            task_id="say",
            job_dir=str(job),
            cwd=str(folder),
            bindings={
                "msg": {"value": "x"},
                "out": {"path": str(folder / "o")},
            },
            env={},
        )
        await wait_exit(job)
    idle = env.work / "proj" / "99999999-0000-4000-8000-000000000000"
    await env.call(
        "mkdir", run=idle.name, release="echo", path=str(idle / "a")
    )
    busy = env.work / "proj" / "88888888-0000-4000-8000-000000000000"
    await env.call(
        "mkdir", run=busy.name, release="echo", path=str(busy / "a")
    )
    late = env.work / "proj" / "77777777-0000-4000-8000-000000000000"
    await refused(
        env,
        "mkdir",
        "runs already",
        run=late.name,
        release="echo",
        path=str(late / "a"),
    )


async def test_revoked_runs_can_still_be_stopped(env: Env) -> None:
    job = env.run / "job"
    await env.call(
        "task_launch",
        release="echo",
        task="say",
        task_id="say",
        job_dir=str(job),
        cwd=str(env.run),
        bindings={"msg": {"value": "x"}, "out": {"path": str(env.run / "o")}},
        env={},
    )
    policy.revoke(env.echo)
    await env.call("signal", release="echo", job_dir=str(job), sig=15)
    await env.call("poll", release="echo", job_dir=str(job))
    await env.call("cancel_run", release="echo")
    await refused(
        env, "mkdir", "not approved", release="echo", path=str(env.run / "new")
    )


# --- approval screen, environment, values from upstream --------------------


@pytest.mark.parametrize("bad", ["\r", "\x1b[2K", "\u202e", "\x07"])
def test_control_characters_are_refused(bad: str) -> None:
    with pytest.raises(ManifestError, match="control character"):
        manifest(f"cat ${{inp}} > ${{out}}; echo {bad}x")


def test_approve_screen_escapes_what_it_shows() -> None:
    shown = cli._show(
        Manifest(
            body={"schema": 1, "tasks": {}},
            about={"name": "a\x1b[2Kb"},
            warnings=("w\rx",),
        )
    )
    assert "\x1b" not in shown and "\r" not in shown
    assert "a\\u001b[2Kb" in shown


@pytest.mark.parametrize(
    "name",
    [
        "SBATCH_EXPORT",
        "SLURM_CONF",
        "NODE_OPTIONS",
        "https_proxy",
        "PERL5OPT",
        "GIT_SSH_COMMAND",
        "PS4",
        "R_PROFILE_USER",
    ],
)
def test_environment_names_that_change_what_runs(name: str) -> None:
    with pytest.raises(ManifestError, match="refused"):
        manifest("cat ${inp} > ${out}", secrets={name: "${secret:x}"})


def test_approve_warns_about_run_time_choices() -> None:
    found = manifest(
        "cat ${inp} ${n} > ${out}",
        inputs=[{"id": "inp", "kind": "file"}, {"id": "n", "kind": "number"}],
        secrets={"TOKEN": "${secret:t}"},
    )
    text = " ".join(found.warnings)
    assert "'inp'" in text and "'n'" in text and "'TOKEN'" in text


async def test_values_from_upstream_tasks_bind_as_values(env: Env) -> None:
    wf = {
        "name": "w",
        "tasks": [
            {
                "id": "t",
                "target": {"kind": "agent", "agent": "lab"},
                "executor": {"kind": "shell"},
                "runtime": {
                    "kind": "command",
                    "command": "echo ${s} > ${out}",
                },
                "inputs": [{"id": "s", "kind": "string"}],
                "outputs": [{"id": "out", "kind": "file", "path": "o"}],
            }
        ],
        "edges": [{"source": "up", "target": "t", "target_input": "s"}],
    }
    approved = build_manifest_from_workflow(wf, {})
    assert cast(Any, approved.task("t"))["inputs"]["s"]["role"] == "edge"
    policy.approve(approved, "root")
    job = env.run / "j"
    await env.call(
        "task_launch",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(job),
        cwd=str(env.run),
        bindings={
            "s": {"value": "a.b:1"},
            "out": {"path": str(env.run / "o")},
        },
    )
    await wait_exit(job)
    assert (env.run / "o").read_text() == "a.b:1\n"


async def test_squeue_failure_is_an_error(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing(*_: Any, **__: Any) -> Any:
        return 1, b"", b"slurm_load_jobs error: Unable to contact controller"

    monkeypatch.setattr(OpRunner, "_slurm", failing)
    with pytest.raises(RuntimeError, match="squeue failed"):
        await env.runner._slurm_status("42", env.run)


async def test_relative_outputs_land_in_the_working_directory(
    env: Env,
) -> None:
    env.with_policy(uploads=True, logs="full")
    approved = manifest(
        "echo hi > o",
        inputs=[],
        outputs=[{"id": "out", "kind": "file", "path": "o"}],
    )
    policy.approve(approved, "root")
    cwd = env.run / "t"
    args: dict[str, Any] = {
        "task_id": "t",
        "job_dir": str(cwd / ".horus_job" / "j"),
        "cwd": str(cwd),
    }
    release = approved.release_hash
    await env.call("task_launch", release=release, task="t", **args)
    await wait_exit(cwd / ".horus_job" / "j")
    got = await env.call("get_file", release=release, path=str(cwd / "o"))
    assert base64.b64decode(got["data"]) == b"hi\n"
    # A working directory reached through a link is refused.
    (env.run / "link").symlink_to(cwd)
    args.update(
        job_dir=str(cwd / ".horus_job" / "k"), cwd=str(env.run / "link")
    )
    await refused(
        env, "task_launch", "link", release=release, task="t", **args
    )


async def test_revoke_kills_running_jobs(env: Env) -> None:
    approved = manifest("sleep 30 > ${out}", inputs=[])
    policy.approve(approved, "root")
    release = approved.release_hash
    first = await env.call(
        "task_launch",
        release=release,
        task="t",
        task_id="t",
        job_dir=str(env.run / "j"),
        cwd=str(env.run),
        bindings={"out": {"path": str(env.run / "o")}},
    )
    assert await env.runner.stop_revoked() == 0
    policy.revoke(release)
    assert await env.runner.stop_revoked() == 1
    for _ in range(100):
        try:
            os.kill(first["pid"], 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("the job of a revoked release kept running")
    assert await env.runner.stop_revoked() == 0  # once per run


# --- round 2 ----------------------------------------------------------------


def string_manifest(command: str) -> Manifest:
    """One approved task with a free text parameter ``msg``."""
    return manifest(command, inputs=[{"id": "msg", "kind": "string"}])


async def launch_t(
    env: Env, approved: Manifest, run_dir: Path, job: str, **bindings: Any
) -> Any:
    return await env.call(
        "task_launch",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(run_dir / job),
        cwd=str(run_dir),
        bindings={"out": {"path": str(run_dir / "o")}, **bindings},
        env={},
    )


async def test_a_strict_agent_has_no_spill_folder(env: Env) -> None:
    (env.run / "results").mkdir()
    (env.run / "results" / "patient.vcf").write_text("PATIENT-123\n")
    dst = env.work / ".horus-spill" / RUN1 / "x.vcf"
    await refused(
        env,
        "move",
        "agent's own",
        src=str(env.run / "results" / "patient.vcf"),
        dst=str(dst),
    )
    await refused(env, "get_file", "agent's own", path=str(dst))


async def test_open_agents_hand_back_only_their_own_spill_files(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from horus_agent import ops

    env.with_policy(mode="open")
    monkeypatch.setattr(ops, "INLINE_LIMIT", 4)
    result = await env.call("exec", cmd="echo too-long", release=None)
    got = await env.call("get_file", path=result["stdout_file"], release=None)
    assert got == {"size": 9}
    planted = env.work / ".horus-spill" / "planted.vcf"
    planted.write_text("PATIENT-123\n")
    await refused(
        env, "get_file", "stays here", path=str(planted), release=None
    )


@pytest.mark.parametrize(
    "command",
    [
        "[[ ${msg} -gt 0 ]]; echo > ${out}",
        "(( ${msg} )); echo > ${out}",
        "for ((i=0;i<${msg};i++)); do :; done; echo > ${out}",
        "let ${msg}; echo > ${out}",
        "echo $[ ${msg} ] > ${out}",
        "declare -i v=${msg}; echo > ${out}",
        "a[${msg}]=1; echo > ${out}",
        "printf -v v ${msg}; echo > ${out}",
        "trap ${msg} EXIT; echo > ${out}",
        "cat > ${out} <<EOF\n\tEOF\n${msg}\nEOF",
        "cat > ${out} <<-EOF\n\t${msg}\n\tEOF",
        "echo ${msg} | sh > ${out}",
        "sh <<< ${msg} > ${out}",
        ". <(echo ${msg}); echo > ${out}",
        "env ${msg} true > ${out}",
        "python3 -c ${msg} > ${out}",
        "Rscript -e ${msg} > ${out}",
        "awk ${msg} /dev/null > ${out}",
        "${msg} > ${out}",
        "timeout 5 ${msg} > ${out}",
        "bash ${msg} > ${out}",
    ],
)
def test_code_and_arithmetic_positions_are_refused(command: str) -> None:
    with pytest.raises(ManifestError, match=r"plain argument|quoting"):
        string_manifest(command)


def test_verified_scripts_may_run_as_code() -> None:
    manifest(
        "Rscript ${s} ${inp} > ${out}",
        inputs=[
            {"id": "s", "kind": "file", "path": "s.R"},
            {"id": "inp", "kind": "file"},
        ],
        files={"s.R": b"1\n"},
    )
    with pytest.raises(ManifestError, match="plain argument"):
        manifest("Rscript ${inp} > ${out}")  # cloud data run as code


@pytest.mark.parametrize(
    ("kind", "value", "ok"),
    [
        ("number", "8", True),
        ("number", "-1.5", True),
        ("number", "a[$(id)]", False),
        ("number", "THREADS", False),
        ("boolean", "true", True),
        ("boolean", "yes", False),
        ("string", "S1.fastq,lane:2", True),
        ("string", "a b", False),
        ("string", "--to-command=id", False),
        ("string", "x;id", False),
        ("choice", "H1N1", True),
        ("choice", "H5N1", False),
    ],
)
def test_values_must_fit_their_kind(kind: str, value: str, ok: bool) -> None:
    spec = {"kind": kind, "options": ["H1N1", "H3N2"]}
    if ok:
        launch.check_value("p", spec, value)
    else:
        with pytest.raises(PolicyRefusedError, match="not a valid"):
            launch.check_value("p", spec, value)


def test_choice_options_are_approved() -> None:
    found = manifest(
        "echo ${c} > ${out}",
        inputs=[
            {
                "id": "c",
                "kind": "choice",
                "options": [{"value": "a"}, {"value": 2}],
            }
        ],
    )
    assert cast(Any, found.task("t"))["inputs"]["c"]["options"] == ["a", "2"]


@pytest.mark.parametrize(
    "name",
    [
        ".Rprofile",
        ".Renviron",
        "Makefile",
        "nextflow.config",
        "node_modules",
        "sitecustomize.py",
        "x.pth",
        ".npmrc",
        "package.json",
        "conftest.py",
    ],
)
async def test_tool_config_names_cannot_be_written(
    env: Env, name: str
) -> None:
    data = base64.b64encode(b"x").decode()
    await refused(
        env,
        "put_file",
        "settings or code",
        path=str(env.run / name),
        data=data,
    )
    await refused(
        env, "mkdir", "settings or code", path=str(env.run / name / "lib")
    )


async def test_packages_with_tool_config_names_are_refused(env: Env) -> None:
    import tarfile

    src = env.run / "pkgsrc"
    (src / "node_modules" / "lodash").mkdir(parents=True)
    (src / "node_modules" / "lodash" / "index.js").write_text("1")
    pkg = env.run / "p.tgz"
    with tarfile.open(pkg, "w:gz") as tar:
        tar.add(src, arcname=".")
    await refused(
        env, "unpack", "", pkg=str(pkg), dest=str(env.run / "unpacked")
    )
    assert not (env.run / "unpacked").exists()


async def test_working_dirs_holding_op_written_config_are_refused(
    env: Env,
) -> None:
    approved = manifest("cat .Rprofile > ${out}", inputs=[])
    policy.approve(approved, "root")
    (env.run / ".Rprofile").write_text("system('touch PWNED')\n")
    await refused(
        env,
        "task_launch",
        "settings or code",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(env.run / "j"),
        cwd=str(env.run),
        bindings={"out": {"path": str(env.run / "o")}},
        env={},
    )


async def test_jobs_get_a_clean_home_and_no_user_startup_files(
    env: Env,
) -> None:
    approved = manifest("env > ${out}", inputs=[])
    policy.approve(approved, "root")
    await launch_t(env, approved, env.run, "j")
    await wait_exit(env.run / "j")
    seen = dict(
        line.split("=", 1)
        for line in (env.run / "o").read_text().splitlines()
        if "=" in line
    )
    assert seen["HOME"] == str(env.run / "j" / "home")
    assert (env.run / "j" / "home").is_dir()
    assert seen["R_PROFILE_USER"] == seen["R_ENVIRON_USER"] == os.devnull
    assert seen["PYTHONNOUSERSITE"] == "1"
    assert not {"NODE_PATH", "BASH_ENV", "ENV"} & set(seen)


def test_the_state_dir_is_the_agents_alone(tmp_path: Path) -> None:
    from horus_agent.guard import Guard
    from horus_agent.policy import Approvals, Policy

    old = os.umask(0o007)  # the umask under run_as
    try:
        guard = Guard(
            Policy(work_dir=str(tmp_path / "w"), allowed_hosts=()),
            Approvals(),
        )
    finally:
        os.umask(old)
    assert guard.state.stat().st_mode & 0o777 == 0o711
    assert (guard.state / "runs").stat().st_mode & 0o777 == 0o700
    assert guard.const_store.stat().st_mode & 0o777 == 0o750
    assert guard.policy.audit_path.is_relative_to(guard.state)


async def test_cancelled_runs_leave_the_active_set(env: Env) -> None:
    env.with_policy(max_concurrent_runs=1)
    approved = manifest("sleep 30 > ${out}", inputs=[])
    policy.approve(approved, "root")
    release = approved.release_hash
    await launch_t(env, approved, env.run, "j")
    assert await env.call("cancel_run", release=release) == 1
    await env.call("remove", release=release, path=str(env.run / "j"))
    other = env.work / "proj" / RUN2
    await env.call("mkdir", run=RUN2, release=release, path=str(other / "x"))


async def test_a_work_dir_reached_through_a_link_works(
    env: Env, tmp_path: Path
) -> None:
    link = tmp_path / "link"
    link.symlink_to(env.work)
    env.with_policy(work_dir=str(link))
    approved = manifest("echo hi > ${out}", inputs=[])
    policy.approve(approved, "root")
    run = link / "proj" / RUN1
    await launch_t(env, approved, run, "j")
    await wait_exit(env.run / "j")
    assert (env.run / "o").read_text() == "hi\n"


async def test_same_content_under_two_names(env: Env) -> None:
    approved = manifest(
        "cat ${a} ${b} > ${out}",
        inputs=[
            {"id": "a", "kind": "file", "path": "a.txt"},
            {"id": "b", "kind": "file", "path": "b.txt"},
        ],
        files={"a.txt": b"same\n", "b.txt": b"same\n"},
    )
    policy.approve(approved, "root")
    (env.run / "a.txt").write_text("same\n")
    (env.run / "b.txt").write_text("same\n")
    await launch_t(
        env,
        approved,
        env.run,
        "j",
        a={"path": str(env.run / "a.txt")},
        b={"path": str(env.run / "b.txt")},
    )
    await wait_exit(env.run / "j")
    assert (env.run / "o").read_text() == "same\nsame\n"
    # Reused, not copied again, by a later run.
    other = env.work / "proj" / RUN2
    other.mkdir()
    (other / "a.txt").write_text("same\n")
    (other / "b.txt").write_text("same\n")
    stored = env.guard.const_store / digest(env.run / "a.txt") / "a.txt"
    before = stored.stat().st_ino
    await env.call(
        "task_launch",
        run=RUN2,
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(other / "j"),
        cwd=str(other),
        bindings={
            "out": {"path": str(other / "o")},
            "a": {"path": str(other / "a.txt")},
            "b": {"path": str(other / "b.txt")},
        },
    )
    assert stored.stat().st_ino == before


async def test_job_dirs_with_percent_are_refused(env: Env) -> None:
    approved = manifest("echo > ${out}", inputs=[])
    policy.approve(approved, "root")
    await refused(
        env,
        "task_launch",
        "characters",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(env.run / "j%j"),
        cwd=str(env.run),
        bindings={"out": {"path": str(env.run / "o")}},
    )


@pytest.mark.parametrize(
    "command",
    [
        "timeout -s KILL 60 ${msg} > ${out}",
        "timeout --signal=KILL 60 ${msg} > ${out}",
        "env -u HOME ${msg} > ${out}",
        "env -S ${msg} > ${out}",
        "flock -c ${msg} /tmp/l > ${out}",
        "sudo -s cat ${msg} > ${out}",
        "nice -n 5 ${msg} > ${out}",
        "mpirun --unknown-option x prog ${msg} > ${out}",
    ],
)
def test_wrapper_options_do_not_hide_the_program(command: str) -> None:
    with pytest.raises(ManifestError, match=r"plain argument"):
        string_manifest(command)


@pytest.mark.parametrize(
    "command",
    [
        "timeout -s KILL 60 cat ${msg} > ${out}",
        "env -u HOME cat ${msg} > ${out}",
        "srun -n 4 cat ${msg} > ${out}",
        "nice -10 cat ${msg} > ${out}",
    ],
)
def test_wrapped_programs_take_plain_arguments(command: str) -> None:
    string_manifest(command)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"max_unpack_files": 2}, "max_unpack_files"),
        ({"max_unpack_bytes": 5}, "max_unpack_bytes"),
    ],
)
async def test_unpack_is_capped(
    env: Env, change: dict[str, int], match: str
) -> None:
    import tarfile

    src = env.run / "src"
    src.mkdir()
    for name in ("a", "b", "c"):
        (src / name).write_text("0123456789")
    pkg = env.run / "p.tgz"
    with tarfile.open(pkg, "w:gz") as tar:
        tar.add(src, arcname=".")
    env.with_policy(**change)
    await refused(env, "unpack", match, pkg=str(pkg), dest=str(env.run / "d"))
    assert not (env.run / "d").exists()
    env.with_policy(max_unpack_files=10, max_unpack_bytes=100)
    await env.call("unpack", pkg=str(pkg), dest=str(env.run / "d"))
    assert (env.run / "d" / "c").read_text() == "0123456789"
