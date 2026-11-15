#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Regression tests for the sixth review round of the policy work."""

import asyncio
import base64
import io
import os
import tarfile
from pathlib import Path
from typing import Any

import pytest

from horus_agent import launch as launch_mod
from horus_agent import native, policy
from horus_agent.manifest import Manifest, file_digest
from horus_agent.template import unsafe_placeholders
from tests.unit.test_guard import Env, env, refused
from tests.unit.test_policy_hardening import manifest, wait_exit

__all__ = ["env"]

GOOD = b"touch GOOD\n"
EVIL = b"touch PWNED\n"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _release(name: str = "a.sh") -> Manifest:
    approved = manifest(
        f"bash lib/{name}",
        inputs=[{"id": "h", "kind": "file", "path": f"lib/{name}"}],
        outputs=[],
        files={f"lib/{name}": GOOD},
    )
    policy.approve(approved, "root")
    return approved


async def _launch(env: Env, approved: Manifest, job: str = "j") -> None:
    await env.call(
        "task_launch",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(env.run / job),
        cwd=str(env.run),
        bindings={},
        env={},
    )
    await wait_exit(env.run / job)


def _tar(path: Path, build: Any) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        build(tar)
    return path


def _file(tar: tarfile.TarFile, name: str, body: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(body)
    tar.addfile(info, io.BytesIO(body))


# --- B1: links in a package --------------------------------------------------


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
async def test_a_package_holds_no_links(env: Env, kind: bytes) -> None:
    approved = _release()

    def build(tar: tarfile.TarFile) -> None:
        _file(tar, "sub/real.txt", GOOD)
        link = tarfile.TarInfo("a.sh")
        link.type = kind
        link.linkname = "sub/real.txt"
        tar.addfile(link)

    pkg = _tar(env.run / "p.tgz", build)
    await refused(
        env,
        "unpack",
        "link",
        release=approved.release_hash,
        pkg=str(pkg),
        dest=str(env.run / "lib"),
    )
    assert not (env.run / "lib").exists()


@pytest.mark.parametrize("primed", [False, True])
async def test_a_link_at_a_shipped_path_never_runs(
    env: Env, primed: bool
) -> None:
    approved = _release()
    lib = env.run / "lib"
    (lib / "sub").mkdir(parents=True)
    (lib / "sub" / "real.txt").write_bytes(EVIL)
    (lib / "a.sh").symlink_to("sub/real.txt")
    if primed:
        good = env.work / "good" / "a.sh"
        good.parent.mkdir()
        good.write_bytes(GOOD)
        native.materialise(good, env.guard.const_store, file_digest(GOOD))
        await _launch(env, approved)
        assert (env.run / "GOOD").exists()
    else:
        await refused(
            env,
            "task_launch",
            "not the approved",
            release=approved.release_hash,
            task="t",
            task_id="t",
            job_dir=str(env.run / "j"),
            cwd=str(env.run),
            bindings={},
            env={},
        )
    assert not (env.run / "PWNED").exists()


# --- B2: the parent folder of a shipped file ---------------------------------


@pytest.mark.parametrize("name", ["analyze", "analyze.sh"])
async def test_a_parent_folder_swap_never_runs(env: Env, name: str) -> None:
    approved = _release(name)
    release = approved.release_hash
    run = env.run
    await env.call(
        "put_file",
        release=release,
        path=str(run / "lib" / name),
        data=b64(GOOD),
    )
    await _launch(
        env, approved, ".horus_job/j0"
    )  # the verified copy is in the store
    (run / "GOOD").unlink()
    with pytest.raises(RuntimeError):  # the staging step is refused
        await env.call(
            "put_file",
            release=release,
            path=str(run / name / name),
            data=b64(EVIL),
        )
    # Whatever got there some other way is replaced at launch.
    (run / name).mkdir()
    (run / name / name).write_bytes(EVIL)
    with pytest.raises(RuntimeError):
        await env.call(
            "move", release=release, src=str(run / name), dst=str(run / "lib")
        )
    (run / "lib" / name).chmod(0o644)
    (run / "lib" / name).write_bytes(EVIL)
    await _launch(env, approved, ".horus_job/j1")
    assert (run / "GOOD").exists()
    assert not (run / "PWNED").exists()
    sealed = run / "lib" / name
    assert sealed.read_bytes() == GOOD
    assert sealed.stat().st_mode & 0o222 == 0
    assert (run / "lib").stat().st_uid == os.geteuid()


async def test_an_ancestor_that_is_a_file_is_replaced(env: Env) -> None:
    approved = _release()
    good = env.work / "good" / "a.sh"
    good.parent.mkdir()
    good.write_bytes(GOOD)
    native.materialise(good, env.guard.const_store, file_digest(GOOD))
    (env.run / "lib").write_bytes(EVIL)
    await _launch(env, approved)
    assert (env.run / "GOOD").exists()
    assert (env.run / "lib").is_dir()


# --- a sealed path is out of reach while its task runs -----------------------


async def test_sealed_paths_are_refused_while_the_task_runs(env: Env) -> None:
    approved = manifest(
        "bash lib/a.sh",
        inputs=[{"id": "h", "kind": "file", "path": "lib/a.sh"}],
        outputs=[],
        files={"lib/a.sh": b"sleep 30\n"},
    )
    policy.approve(approved, "root")
    release = approved.release_hash
    await env.call(
        "put_file",
        release=release,
        path=str(env.run / "lib" / "a.sh"),
        data=b64(b"sleep 30\n"),
    )
    await env.call(
        "task_launch",
        release=release,
        task="t",
        task_id="t",
        job_dir=str(env.run / "j"),
        cwd=str(env.run),
        bindings={},
        env={},
    )
    sealed = str(env.run / "lib" / "a.sh")
    # The same bytes change nothing; everything else is refused.
    await env.call(
        "put_file", release=release, path=sealed, data=b64(b"sleep 30\n")
    )
    for method, args in (
        ("remove", {"path": sealed}),
        ("remove", {"path": str(env.run / "lib")}),
        ("move", {"src": str(env.run / "x"), "dst": str(env.run / "lib")}),
    ):
        await refused(env, method, "running task", release=release, **args)
    job = env.guard.job(env.run.name, env.run / "j")
    assert job is not None
    await env.call("cancel_run", release=release)


# --- M-A: nothing is staged between the last check and the start ------------


async def test_code_staged_while_a_launch_checks_never_runs(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.with_policy(allow_code_suffixes=(".py",))
    real = launch_mod._seal

    async def slow(*args: Any) -> None:
        await asyncio.sleep(0.3)  # a launch busy verifying a large input
        await real(*args)

    monkeypatch.setattr(launch_mod, "_seal", slow)
    approved = manifest("sleep 5", inputs=[], outputs=[])
    policy.approve(approved, "root")
    release = approved.release_hash
    started = asyncio.create_task(
        env.call(
            "task_launch",
            release=release,
            task="t",
            task_id="t",
            job_dir=str(env.run / ".horus_job" / "j"),
            cwd=str(env.run),
            bindings={},
            env={},
        )
    )
    await asyncio.sleep(0.05)
    with pytest.raises(RuntimeError, match="task of this run is running"):
        await env.call(
            "put_file",
            release=release,
            path=str(env.run / "json" / "__init__.py"),
            data=b64(b"open('PWNED', 'w')\n"),
        )
    await started
    assert not (env.run / "json").exists()
    await env.call("cancel_run", release=release)


# --- C1/C2: the runtime's own staging ----------------------------------------


async def test_a_large_folder_package_arrives_by_fetch(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    approved = manifest(
        "ls cfg",
        inputs=[{"id": "cfg", "kind": "folder", "path": "data/cfg"}],
        outputs=[],
        files={"data/cfg/a.txt": b"A\n"},
    )
    policy.approve(approved, "root")
    pkg = env.run / "cfg.1a2b3c4d.horuspkg"
    _tar(pkg, lambda tar: _file(tar, "a.txt", b"A\n"))
    data = pkg.read_bytes()
    pkg.unlink()

    async def download(_url: str, path: Path, _limit: int | None) -> None:
        path.write_bytes(data)

    monkeypatch.setattr("horus_agent.ops.blobs.download_to", download)
    await env.call(
        "fetch",
        release=approved.release_hash,
        path=str(pkg),
        url="https://files.example.org/x",
    )
    await env.call(
        "unpack",
        release=approved.release_hash,
        pkg=str(pkg),
        dest=str(env.run / "cfg"),
    )
    await env.call("remove", release=approved.release_hash, path=str(pkg))
    assert (env.run / "cfg" / "a.txt").read_bytes() == b"A\n"
    # Only the runtime's file name is let through, not any name.
    await refused(
        env,
        "put_file",
        "nothing may be staged",
        release=approved.release_hash,
        path=str(env.run / "cfg.horuspkg"),
        data=b64(b"x"),
    )


# --- M-B: the checker reads streams, applets and delayed runners -------------

RUNS_IT = [
    'source /dev/stdin <<< "$(cat ${x})"',
    "bash /dev/stdin < ${x}",
    "bash /dev/fd/0 < ${x}",
    "cat ${x} | bash /dev/stdin",
    ". /dev/fd/3 3< ${x}",
    "bash <(cat ${x})",
    "source <(cat ${x})",
    "python3 /proc/self/fd/0 < ${x}",
    "busybox sh ${x}",
    "busybox ash ${x}",
    'script -c "bash ${x}" /dev/null',
    "tee y < ${x}; bash y",
    "cat ${x} | tee y; bash y",
    "dd if=${x} of=y; bash y",
    'at now <<< "bash ${x}"',
    'batch <<< "bash ${x}"',
    "at -f ${x} now",
    "nohup bash ${x}",
    "setsid bash ${x}",
    "stdbuf -oL bash ${x}",
    "timeout 5 bash ${x}",
    "ionice -c3 bash ${x}",
    "timeout 5 nohup setsid ionice -c3 stdbuf -oL bash ${x}",
    "pipenv run ${x}",
    "hatch run ${x}",
    "pdm run ${x}",
    "rye run ${x}",
    "accelerate launch ${x}",
    "podman exec c ${x}",
    "docker exec c ${x}",
    "conda activate ${x}",
    "mytool run ${x}",
    "mytool --command ${x}",
]


@pytest.mark.parametrize("command", RUNS_IT)
def test_the_checker_flags_what_runs_a_placeholder(command: str) -> None:
    assert unsafe_placeholders(command, {"x"}), command


@pytest.mark.parametrize(
    "command",
    [
        "cat ${x} | tee out.txt > /dev/null",
        "tee log.txt < ${x} > /dev/null",
        "busybox cat ${x}",
        "timeout 5 samtools sort ${x}",
        "nice -n 5 gzip -c ${x} > o.gz",
        "mytool --in ${x} --out o",
    ],
)
def test_the_checker_still_passes_data_commands(command: str) -> None:
    assert not unsafe_placeholders(command, {"x"}), command
