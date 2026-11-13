#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Ops checked against the machine's policy: a refusal for every op type, and
the approved paths that still work.
"""

import asyncio
import base64
import dataclasses
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from horus_agent import guard as guard_mod
from horus_agent import launch, policy
from horus_agent.guard import Guard
from horus_agent.manifest import (
    Manifest,
    build_manifest,
    build_manifest_from_workflow,
)
from horus_agent.native import digest
from horus_agent.ops import OpRunner
from horus_agent.policy import Approvals, Policy
from horus_agent.protocol import Op, PolicyRefusedError, TaskLaunchArgs

DATA = Path(__file__).parent.parent / "data" / "manifest"
FAKE_SLURM = Path(__file__).parent.parent / "fake_slurm"
HOST = "files.example.org"
REF = b">chr1\nACGT\n"
RUN1 = "11111111-1111-4111-8111-111111111111"
RUN2 = "22222222-2222-4222-8222-222222222222"


def echo_manifest() -> Manifest:
    """One approved shell task that echoes a param and a secret."""
    wf = {
        "name": "echo",
        "tasks": [
            {
                "id": "say",
                "name": "Say",
                "target": {"kind": "agent", "agent": "lab"},
                "executor": {"kind": "shell"},
                "runtime": {
                    "kind": "command",
                    "command": 'printf "%s|%s" ${msg} "$TOKEN" > ${out}',
                },
                "inputs": [{"id": "msg", "kind": "string"}],
                "outputs": [{"id": "out", "kind": "file", "path": "o"}],
                "secrets": {"TOKEN": "${secret:t}"},
            }
        ],
    }
    return build_manifest_from_workflow(wf, {})


@dataclass
class Env:
    work: Path
    run: Path
    runner: OpRunner
    guard: Guard
    basic: str
    echo: str
    slurm: str

    async def call(
        self,
        method: str,
        *,
        run: str = RUN1,
        release: str | None = "basic",
        task: str | None = None,
        v: int = 2,
        **args: Any,
    ) -> Any:
        """Run one op; a refusal comes back as RuntimeError."""
        hashes = {"basic": self.basic, "echo": self.echo, "slurm": self.slurm}
        op = Op(
            op_id=uuid4().hex,
            method=method,
            args=args,
            run_id=run,
            release_hash=hashes.get(release or "", release),
            task_id=task,
            v=v,
        )
        result = await self.runner.run(op)
        if result.error is not None:
            raise RuntimeError(f"{result.error.type}: {result.error.message}")
        return result.result

    def with_policy(self, **changes: Any) -> None:
        """Swap the policy for one with *changes*."""
        rules = dataclasses.replace(self.guard.policy, **changes)
        self.guard = Guard(rules, self.guard.approvals)
        self.runner = OpRunner(job_env=("FAKE_SLURM_STATE",), guard=self.guard)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    monkeypatch.setattr(policy, "APPROVALS_TTL_S", -1.0)
    work = tmp_path / "work"
    run = work / "proj" / RUN1
    run.mkdir(parents=True)
    hashes = []
    for manifest in (
        build_manifest((DATA / "basic.zip").read_bytes()),
        echo_manifest(),
        build_manifest((DATA / "slurm.zip").read_bytes()),
    ):
        policy.approve(manifest, "root")
        hashes.append(manifest.release_hash)
    rules = Policy(work_dir=str(work), allowed_hosts=(HOST,))
    guard = Guard(rules, Approvals())
    return Env(
        work,
        run,
        OpRunner(job_env=("FAKE_SLURM_STATE",), guard=guard),
        guard,
        *hashes,
    )


REFUSED = "PolicyRefusedError"


async def refused(env: Env, method: str, match: str = "", /, **kw: Any) -> str:
    with pytest.raises(RuntimeError, match=REFUSED) as info:
        await env.call(method, **kw)
    assert match in str(info.value)
    return str(info.value)


# --- envelope ---------------------------------------------------------------


async def test_raw_commands_are_refused(env: Env) -> None:
    await refused(env, "exec", "approved", cmd="id")
    await refused(
        env, "launch", "approved", cmd="id", job_dir=str(env.run / "j")
    )


async def test_envelope_refusals(env: Env) -> None:
    path = str(env.run / "x")
    await refused(env, "teleport", "Unknown op", path=path)
    await refused(env, "mkdir", "arguments", path=path, extra=1)
    await refused(env, "mkdir", "protocol", path=path, v=1)
    await refused(env, "mkdir", "not approved", path=path, release=None)
    await refused(env, "mkdir", "not approved", path=path, release="f" * 64)
    await refused(env, "mkdir", "not part", path=path, task="nope")
    await refused(env, "mkdir", "run id", path=path, run="../x")


async def test_a_run_keeps_its_release(env: Env) -> None:
    await env.call("mkdir", path=str(env.run / "a"))
    await refused(env, "mkdir", "match", path=str(env.run), release="echo")


async def test_revoked_release_is_refused(env: Env) -> None:
    await env.call("mkdir", path=str(env.run / "a"))
    policy.revoke(env.basic)
    await refused(env, "mkdir", "not approved", path=str(env.run / "b"))


async def test_too_many_runs(env: Env) -> None:
    env.with_policy(max_concurrent_runs=1)
    await env.call("mkdir", path=str(env.run / "a"))
    other = env.work / "proj" / RUN2
    await refused(env, "mkdir", "runs already", path=str(other), run=RUN2)


async def test_decisions_are_audited(env: Env) -> None:
    await env.call("mkdir", path=str(env.run / "a"))
    await refused(env, "exec", cmd="id")
    lines = [
        json.loads(line)
        for line in env.guard.policy.audit_path.read_text().splitlines()
    ]
    assert [(r["method"], r["verdict"]) for r in lines] == [
        ("mkdir", "allowed"),
        ("exec", "refused"),
    ]
    assert lines[0]["paths"] == [str(env.run / "a")]


# --- confinement -------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "//etc/passwd",
        "/..",
        "{work}/../etc",
        "{work}",
        "{work}/proj",
        f"{{work}}/proj/{RUN2}/x",
        f"{{work}}/.horus-agent/runs/{RUN1}.json",
        "relative/path",
        "{run}/link/passwd",
    ],
)
async def test_paths_leaving_the_run_are_refused(env: Env, raw: str) -> None:
    (env.run / "link").symlink_to("/etc")
    path = raw.format(work=env.work, run=env.run)
    await refused(env, "path_exists", path=path)


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("mkdir", {"path": "/etc/x"}),
        ("list_dir", {"path": "/etc"}),
        ("path_exists", {"path": "/etc/passwd"}),
        ("remove", {"path": "/etc/x"}),
        ("stat", {"path": "/etc/passwd"}),
        ("digest", {"path": "/etc/passwd"}),
        ("get_file", {"path": "/etc/passwd"}),
        ("read_range", {"path": "/etc/passwd", "offset": 0, "length": 1}),
        ("put_file", {"path": "/tmp/x", "data": ""}),
        ("fetch", {"path": "/tmp/x", "url": f"https://{HOST}/x"}),
        ("upload", {"path": "/etc/passwd", "url": f"https://{HOST}/x"}),
        ("pack", {"src": "/etc", "pkg": "{run}/p.tgz"}),
        ("unpack", {"pkg": "{run}/p.tgz", "dest": "/tmp/x"}),
        ("move", {"src": "{run}/a", "dst": "/tmp/a"}),
        ("poll", {"job_dir": "/tmp/j"}),
        ("signal", {"job_dir": "/tmp/j", "sig": 9}),
        (
            "read_output",
            {"job_dir": "/tmp/j", "stdout_offset": 0, "stderr_offset": 0},
        ),
        ("slurm_status", {"job_dir": "/tmp/j"}),
        ("slurm_cancel", {"job_dir": "/tmp/j"}),
        (
            "task_launch",
            {"task_id": "align", "job_dir": "/tmp/j", "cwd": "{run}"},
        ),
        (
            "slurm_submit",
            {"task_id": "big", "job_dir": "/tmp/j", "cwd": "{run}"},
        ),
    ],
)
async def test_every_op_is_confined(
    env: Env, method: str, args: dict[str, Any]
) -> None:
    filled = {
        k: v.format(run=env.run) if isinstance(v, str) else v
        for k, v in args.items()
    }
    await refused(env, method, "outside", **filled)


async def test_open_mode_confines_to_the_work_dir_only(env: Env) -> None:
    env.with_policy(mode="open")
    other = env.work / "elsewhere"
    await env.call("mkdir", path=str(other), release=None, run=RUN2)
    assert other.is_dir()
    await refused(env, "mkdir", "outside", path="/tmp/x", release=None)
    result = await env.call("exec", cmd="id -u", release=None)
    assert result["returncode"] == 0


async def test_open_mode_spills_into_the_work_dir(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from horus_agent import ops

    env.with_policy(mode="open")
    monkeypatch.setattr(ops, "INLINE_LIMIT", 4)
    result = await env.call("exec", cmd="echo too-long", release=None)
    spilled = Path(result["stdout_file"])
    assert spilled.is_relative_to(env.work / guard_mod.SPILL_DIR)
    data = await env.call("get_file", path=str(spilled), release=None)
    assert data == {"size": 9}


# --- transfers and data leaving the machine --------------------------------


async def test_transfer_refusals(env: Env) -> None:
    path = str(env.run / "f")
    await refused(env, "fetch", "allowed", path=path, url=f"http://{HOST}/x")
    await refused(env, "fetch", "allowed", path=path, url="https://evil/x")
    await refused(env, "upload", "Uploads are off", path=path, url=HOST)
    env.with_policy(downloads=False, uploads=True)
    await refused(env, "fetch", "Downloads are off", path=path, url=HOST)
    await refused(
        env, "upload", "allowed", path=path, url="https://10.0.0.1/x"
    )


async def test_uploads_off_keeps_data_but_not_metadata(env: Env) -> None:
    (env.run / "out.bam").write_bytes(b"secret")
    state = str(env.run / ".horus" / "state.json")
    await env.call(
        "put_file", path=state, data=base64.b64encode(b"{}").decode()
    )
    await refused(env, "get_file", "stays here", path=str(env.run / "out.bam"))
    await refused(
        env,
        "read_range",
        "stays here",
        path=str(env.run / "out.bam"),
        offset=0,
        length=3,
    )
    meta = await env.call("get_file", path=str(env.run / ".horus/state.json"))
    assert base64.b64decode(meta["data"]) == b"{}"


async def test_logs_off_returns_no_output(env: Env) -> None:
    job = env.run / "job"
    job.mkdir()
    (job / "stdout.log").write_text("patient 42")
    out = await env.call(
        "read_output", job_dir=str(job), stdout_offset=0, stderr_offset=0
    )
    assert out == {"stdout": "", "stderr": "", "more": False, "exited": False}


async def test_put_file_caps_and_reserved_names(env: Env) -> None:
    env.with_policy(max_put_bytes=2)
    data = base64.b64encode(b"abc").decode()
    await refused(
        env, "put_file", "max_put_bytes", path=str(env.run / "a"), data=data
    )
    small = base64.b64encode(b"1").decode()
    await refused(
        env, "put_file", "pid", path=str(env.run / "pid"), data=small
    )
    await env.call("put_file", path=str(env.run / "a"), data=small)
    assert (env.run / "a").read_bytes() == b"1"


async def test_remove_refuses_the_run_dir(env: Env) -> None:
    await refused(env, "remove", "run's directory", path=str(env.run))


async def test_jobs_not_started_here_cannot_be_polled_or_signalled(
    env: Env,
) -> None:
    job = env.run / "job"
    job.mkdir()
    (job / "pid").write_text("1\n")
    for method, extra in (
        ("poll", {}),
        ("signal", {"sig": 9}),
        ("slurm_status", {}),
        ("slurm_cancel", {}),
    ):
        await refused(env, method, "No job", job_dir=str(job), **extra)


# --- native ops --------------------------------------------------------------


async def test_native_file_ops(env: Env) -> None:
    src = env.run / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a").write_text("a")
    (src / "sub" / "b").write_text("b")
    expected = digest(src)
    assert await env.call("digest", path=str(src)) == expected
    assert (await env.call("digest", path=str(src), mode="fast")).startswith(
        "fast:"
    )
    pkg = env.run / "src.tgz"
    await env.call("pack", src=str(src), pkg=str(pkg))
    dest = env.run / "dest"
    await env.call("unpack", pkg=str(pkg), dest=str(dest))
    assert digest(dest) == expected
    await env.call("unpack", pkg=str(pkg), dest=str(dest))  # no-op
    await env.call("move", src=str(dest), dst=str(env.run / "moved" / "d"))
    assert digest(env.run / "moved" / "d") == expected
    info = await env.call("stat", path=str(src / "a"))
    assert info["type"] == "file" and info["size"] == 1
    assert await env.call("stat", path=str(env.run / "missing")) is None


@pytest.mark.parametrize("member", ["../escape", "link", "dev"])
async def test_unpack_refuses_unsafe_members(env: Env, member: str) -> None:
    pkg = env.run / "bad.tgz"
    with tarfile.open(pkg, "w:gz") as tar:
        info = tarfile.TarInfo(member)
        if member == "link":
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(info)
        elif member == "dev":
            info.type = tarfile.CHRTYPE
            tar.addfile(info)
        else:
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(RuntimeError):
        await env.call("unpack", pkg=str(pkg), dest=str(env.run / "out"))
    assert not (env.run.parent / "escape").exists()
    assert not (env.run / "out").exists()


# --- task_launch -----------------------------------------------------------


def basic_args(env: Env, **over: Any) -> dict[str, Any]:
    """A valid task_launch of the basic release's ``align`` task."""
    (env.run / "ref.fa").write_bytes(REF)
    (env.run / "reads").mkdir(exist_ok=True)
    args: dict[str, Any] = {
        "task_id": "align",
        "job_dir": str(env.run / ".horus_job" / "j1"),
        "cwd": str(env.run),
        "bindings": {
            "ref": {"path": str(env.run / "ref.fa")},
            "reads": {"path": str(env.run / "reads")},
            "bam": {"path": str(env.run / "out.bam")},
            "sample": {"value": "S1"},
            "panel": {"value": "H1N1"},
            "tags": {"value": ["a b", "3", "true"]},
        },
        "env": {"GISAID_TOKEN": "tok"},
    }
    args.update(over)
    return args


async def test_task_command_is_rendered_by_the_agent(env: Env) -> None:
    await env.call("mkdir", path=str(env.run / "x"))  # binds the run
    raw = basic_args(env)
    manifest = env.guard.approvals.get(env.basic)
    assert manifest is not None
    admitted = guard_mod.Admitted(
        TaskLaunchArgs.model_validate(raw),
        manifest,
        manifest.task("align"),
        {"job_dir": Path(raw["job_dir"]), "cwd": env.run},
    )
    prepared = await launch.prepare(env.guard, RUN1, admitted, slurm=False)
    ref = env.guard.const_store / digest(env.run / "ref.fa") / "ref.fa"
    assert prepared.command == (
        f"bwa mem -t 8 {ref} "
        f"{env.run}/reads > {env.run}/out.bam --sample S1 "
        "--panel H1N1 --tags 'a b' 3 true --id align $HOME"
    )
    # Secrets reach the job through stdin, never a file or the command.
    assert prepared.env == {
        "GISAID_TOKEN": "tok",
        **launch.job_env(Path(raw["job_dir"])),
    }
    assert not (env.run / ".horus_job" / "j1" / "env").exists()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"task_id": "upload"}, "approved release"),
        ({"bindings": {}}, "Bindings must be"),
        ({"bindings_extra": {"x": {"value": "1"}}}, "Bindings must be"),
        ({"bind": ("bam", {"path": "/etc/passwd"})}, "outside"),
        ({"bind": ("ref", {"path": "/etc/hosts"})}, "outside"),
        ({"bind": ("panel", {"value": "H5N1"})}, "approved value"),
        ({"bind": ("panel", {"path": "{run}/p"})}, "must be a value"),
        ({"bind": ("bam", {"value": "x"})}, "must be a path"),
        ({"ref_bytes": b"tampered"}, "approved file"),
        ({"env": {"LD_PRELOAD": "/x.so"}}, "not approved"),
        ({"env": {"PATH": "/x"}}, "not approved"),
        ({"env": {"HORUS_SIDE_ARTIFACTS_DIR": "/tmp"}}, "outside"),
    ],
)
async def test_task_launch_refusals(
    env: Env, change: dict[str, Any], match: str
) -> None:
    args = basic_args(env)
    if "bind" in change:
        name, binding = change.pop("bind")
        args["bindings"][name] = {
            k: v.format(run=env.run) for k, v in binding.items()
        }
    if "bindings_extra" in change:
        args["bindings"].update(change.pop("bindings_extra"))
    if "ref_bytes" in change:
        (env.run / "ref.fa").write_bytes(change.pop("ref_bytes"))
        args.pop("ref_bytes", None)
        await refused(env, "task_launch", match, task="align", **args)
        return
    args.update(change)
    await refused(env, "task_launch", match, task=args["task_id"], **args)


async def test_task_launch_runs_the_agents_copy_of_inputs(env: Env) -> None:
    (env.run / "ref.fa").write_bytes(REF)
    args = basic_args(env)
    result = await env.call("task_launch", task="align", **args)
    job = Path(args["job_dir"])
    assert result["pid"] > 1
    copy = env.guard.const_store / digest(env.run / "ref.fa") / "ref.fa"
    assert copy.read_bytes() == REF
    assert copy.stat().st_mode & 0o777 == 0o640
    # The cloud's file may change; what runs is the verified copy, which
    # no op can reach.
    data = base64.b64encode(b"x").decode()
    await env.call("put_file", path=str(env.run / "ref.fa"), data=data)
    assert copy.read_bytes() == REF
    await refused(env, "put_file", "agent's own", path=str(copy), data=data)
    await refused(env, "remove", "agent's own", path=str(copy.parent))
    for _ in range(100):
        if (job / "exit_code").exists():
            break
        await asyncio.sleep(0.05)
    assert await env.call("poll", job_dir=str(job)) is not None


async def test_python_script_hash_is_checked(env: Env) -> None:
    manifest = build_manifest((DATA / "script.zip").read_bytes())
    policy.approve(manifest, "root")
    script = env.run / "qc.py"
    script.write_text("import os  # changed\n")
    args = {
        "task_id": "make",
        "job_dir": str(env.run / "j"),
        "cwd": str(env.run),
        "script": str(script),
        "bindings": {
            "data": {"path": str(env.run / "d")},
            "report": {"path": str(env.run / "r")},
        },
    }
    await refused(
        env,
        "task_launch",
        "not the approved",
        task="make",
        release=manifest.release_hash,
        **args,
    )
    args.pop("script")
    await refused(
        env,
        "task_launch",
        "needs its script",
        task="make",
        release=manifest.release_hash,
        **args,
    )


async def test_echo_task_end_to_end(env: Env) -> None:
    job = env.run / "job"
    out = env.run / "o"
    args: dict[str, Any] = {
        "task_id": "say",
        "job_dir": str(job),
        "cwd": str(env.run),
        "bindings": {
            "msg": {"value": "hi"},
            "out": {"path": str(out)},
        },
        "env": {"TOKEN": "s3cret"},
    }
    first = await env.call("task_launch", release="echo", task="say", **args)
    again = await env.call("task_launch", release="echo", task="say", **args)
    assert first == again, "a resent launch does not start it twice"
    for _ in range(100):
        if (job / "exit_code").exists():
            break
        await asyncio.sleep(0.05)
    assert out.read_text() == "hi|s3cret"
    assert "s3cret" not in (job / "stderr.log").read_text()
    code = await env.call("poll", release="echo", job_dir=str(job))
    assert code == 0
    await env.call("signal", release="echo", job_dir=str(job), sig=15)
    args["job_dir"] = str(env.run / "job2")
    await refused(
        env,
        "slurm_submit",
        "runs through task_launch",
        release="echo",
        task="say",
        **args,
    )


# --- Slurm ------------------------------------------------------------------


@pytest.fixture
def fake_slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    state = tmp_path / "slurm-state"
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state))
    monkeypatch.setenv("PATH", f"{FAKE_SLURM}:{os.environ['PATH']}")
    return state


@pytest.mark.usefixtures("fake_slurm")
async def test_slurm_submit_status_cancel(env: Env) -> None:
    job = env.run / "job"
    args: dict[str, Any] = {
        "task_id": "big",
        "job_dir": str(job),
        "cwd": str(env.run),
        "bindings": {
            "data": {"path": str(env.run / "data")},
            "model": {"path": str(env.run / "m.bin")},
        },
    }
    await refused(
        env,
        "task_launch",
        "runs through slurm_submit",
        release="slurm",
        task="big",
        **args,
    )
    result = await env.call(
        "slurm_submit", release="slurm", task="big", **args
    )
    job_id = result["job_id"]
    again = await env.call("slurm_submit", release="slurm", task="big", **args)
    assert again == result
    script = (job / "job.sh").read_text()
    assert "#SBATCH --partition=gpu\n" in script
    assert "#SBATCH --cpus-per-task=16\n" in script
    assert "#SBATCH --mem=64G\n" in script
    assert "#SBATCH --time=02:00:00\n" in script
    assert "#SBATCH --gres=gpu:1\n" in script
    assert "#SBATCH --exclusive\n" in script
    assert f"train {env.run}/data > {env.run}/m.bin" in script
    status = await env.call("slurm_status", release="slurm", job_dir=str(job))
    assert status["job_id"] == job_id
    await env.call("slurm_cancel", release="slurm", job_dir=str(job))
    for _ in range(100):
        status = await env.call(
            "slurm_status", release="slurm", job_dir=str(job)
        )
        if status["final"] is not None:
            break
        await asyncio.sleep(0.05)
    assert status["final"]["state"] == "CANCELLED"


def test_sbatch_values_cannot_break_lines(env: Env) -> None:
    manifest = env.guard.approvals.get(env.slurm)
    assert manifest is not None
    task = json.loads(json.dumps(manifest.task("big")))
    task["target"]["slurm"]["resources"]["walltime"] = "1 #x"
    task["target"]["slurm"].pop("time_limit")
    prepared = launch.Prepared(
        "true", {}, env.run, env.run / "j", env.run, task
    )
    with pytest.raises(PolicyRefusedError, match="--time"):
        launch.sbatch_script(prepared)
    assert launch._slurm_time("90m") == "01:30:00"
    assert launch._slurm_time("2d") == "2-00:00:00"
    assert launch._slurm_time("01:00:00") == "01:00:00"
    with pytest.raises(RuntimeError):
        launch.parse_job_id(b"oops")
    assert launch.parse_job_id(b"42;cluster\n") == "42"


# --- resource samples ------------------------------------------------------


async def test_sampler_writes_monitor_files(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    for pid, pgid, ut, rss in (
        (10, 7, 150, 100),
        (11, 7, 50, 50),
        (12, 9, 999, 9),
    ):
        (proc / str(pid)).mkdir(parents=True)
        fields = ["S", "1", str(pgid)] + ["0"] * 8 + [str(ut), "0"] + ["0"] * 8
        fields += [str(rss)]
        (proc / str(pid) / "stat").write_text(
            f"{pid} (sh) {' '.join(fields)}\n"
        )
    (proc / "self").mkdir()
    rss, cpu = launch.group_usage(7, proc)
    assert rss == 150 * launch._PAGE_KB
    assert cpu == pytest.approx(200 / launch._TICKS)
    job = tmp_path / "job"
    job.mkdir()
    prepared = launch.Prepared("x", {}, tmp_path, job, tmp_path / "side", {})
    task = asyncio.create_task(launch.sample(7, prepared, 0.01, proc))
    await asyncio.sleep(0.05)
    (job / "exit_code").write_text("0\n")
    await task
    rows = (tmp_path / "side" / launch.SAMPLES_FILE).read_text().splitlines()
    assert rows[0] == launch.SAMPLES_HEADER
    assert rows[1].endswith(",,,,,exact") and len(rows) > 1
    rusage = (tmp_path / launch.RUSAGE_FILE).read_text()
    assert f"Maximum resident set size (kbytes): {rss}" in rusage


async def test_real_job_gets_samples(env: Env) -> None:
    job = env.run / "job"
    args: dict[str, Any] = {
        "task_id": "say",
        "job_dir": str(job),
        "cwd": str(env.run),
        "bindings": {
            "msg": {"value": "x"},
            "out": {"path": str(env.run / "o")},
        },
        "env": {"TOKEN": "t"},
    }
    await env.call("task_launch", release="echo", task="say", **args)
    for _ in range(100):
        if (env.run / launch.RUSAGE_FILE).exists():
            break
        await asyncio.sleep(0.05)
    assert (env.run / launch.RUSAGE_FILE).exists()
    assert (env.run / "side-artifacts" / launch.SAMPLES_FILE).exists()
