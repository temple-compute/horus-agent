#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Regression tests for the fifth review round of the policy work."""

import base64
import io
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from horus_agent import launch, ops, policy
from horus_agent.manifest import Manifest, ManifestError
from horus_agent.template import unsafe_placeholders
from tests.unit.test_guard import RUN1, Env, env, refused
from tests.unit.test_policy_hardening import manifest, wait_exit

__all__ = ["env"]


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _helper_release() -> Manifest:
    return manifest(
        "bash helper.sh",
        inputs=[{"id": "h", "kind": "file", "path": "helper.sh"}],
        outputs=[],
        files={"helper.sh": b"touch GOOD\n"},
    )


def _package(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return path


# --- M1: a shipped name takes only its shipped content ----------------------


@pytest.mark.parametrize("where", ["helper.sh", "t/helper.sh"])
async def test_a_shipped_file_is_staged_with_its_content_only(
    env: Env, where: str
) -> None:
    approved = _helper_release()
    policy.approve(approved, "root")
    release = approved.release_hash
    path = str(env.run / where)
    await refused(
        env,
        "put_file",
        "ships",
        release=release,
        path=path,
        data=b64(b"touch PWNED\n"),
    )
    await env.call(
        "put_file", release=release, path=path, data=b64(b"touch GOOD\n")
    )


async def test_a_shipped_name_cannot_arrive_by_other_ops(env: Env) -> None:
    approved = _helper_release()
    policy.approve(approved, "root")
    release = approved.release_hash
    other = env.run / "x"
    other.write_text("touch PWNED\n")
    target = str(env.run / "helper.sh")
    await refused(
        env,
        "fetch",
        "ships",
        release=release,
        path=target,
        url="https://files.example.org/x",
    )
    await refused(
        env, "move", "ships", release=release, src=str(other), dst=target
    )
    await refused(
        env,
        "unpack",
        "ships",
        release=release,
        pkg=str(_package(env.run / "p.tgz", {"helper.sh": b"touch PWNED\n"})),
        dest=str(env.run / "t"),
    )
    assert not (env.run / "t" / "helper.sh").exists()


async def test_a_package_of_the_shipped_bytes_may_carry_the_name(
    env: Env,
) -> None:
    approved = _helper_release()
    policy.approve(approved, "root")
    await env.call(
        "unpack",
        release=approved.release_hash,
        pkg=str(_package(env.run / "p.tgz", {"helper.sh": b"touch GOOD\n"})),
        dest=str(env.run / "t"),
    )
    assert (env.run / "t" / "helper.sh").read_text() == "touch GOOD\n"


# --- M2: code is staged at its own path --------------------------------------


async def test_shipped_bytes_are_code_only_at_their_own_path(env: Env) -> None:
    code = b"open('PWNED_HIDDEN', 'w')\n"
    approved = manifest(
        "echo hi",
        inputs=[
            {"id": "lib", "kind": "folder", "path": "json"},
            {"id": "ref", "kind": "file", "path": "ref.dat"},
            {"id": "tool", "kind": "file", "path": "tool.py"},
        ],
        outputs=[],
        files={"ref.dat": code, "tool.py": code},
    )
    policy.approve(approved, "root")
    release = approved.release_hash
    await refused(
        env,
        "put_file",
        "allow_code_suffixes",
        release=release,
        path=str(env.run / "json" / "__init__.py"),
        data=b64(code),
    )
    await env.call(
        "put_file",
        release=release,
        path=str(env.run / "tool.py"),
        data=b64(code),
    )
    await env.call(
        "put_file",
        release=release,
        path=str(env.run / "ref.dat"),
        data=b64(code),
    )


async def test_the_manifest_lists_each_shipped_path(env: Env) -> None:  # noqa: ARG001
    approved: Any = _helper_release()
    assert ("helper.sh", next(iter(approved.digests))) in approved.consts
    again = Manifest.from_json(approved.to_json())
    assert again.consts == approved.consts


# --- M3: an old Python starts no task beside code ---------------------------


@pytest.mark.parametrize(
    "command",
    [
        "python3 -c 'import json'",
        "python3 -m json.tool",
        "python3 <<'EOF'\nimport json\nEOF\n",
        "python3 - <<'EOF'\nimport json\nEOF\n",
        "bash -c 'python3 -c \"import json\"'",
        "echo hi",
    ],
)
@pytest.mark.parametrize(
    ("version", "ok"), [((3, 10), False), ((3, 12), True)]
)
async def test_an_old_python_starts_no_task_beside_code(
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    version: tuple[int, int],
    ok: bool,
) -> None:
    env.with_policy(allow_code_suffixes=(".py",))

    async def probe() -> tuple[int, int]:
        return version

    monkeypatch.setattr(env.runner, "_python_version", probe)
    approved = manifest(command, inputs=[], outputs=[])
    policy.approve(approved, "root")
    (env.run / "json").mkdir()
    (env.run / "json" / "__init__.py").write_text("")
    call = env.call(
        "task_launch",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(env.run / "j"),
        cwd=str(env.run),
        bindings={},
        env={},
    )
    if ok:
        await call
        await wait_exit(env.run / "j")
    else:
        with pytest.raises(RuntimeError, match=r"older than 3\.11"):
            await call


def _real_python(version: str) -> str | None:
    found = (
        os.environ.get("HORUS_TEST_PYTHON310")
        if version == "3.10"
        else (sys.executable)
    )
    return found if found and Path(found).exists() else None


@pytest.mark.parametrize(
    ("version", "shadowed"), [("3.10", True), ("3.13", False)]
)
async def test_real_interpreters_on_a_job_env(
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    shadowed: bool,
) -> None:
    """Python 3.10 (``HORUS_TEST_PYTHON310``) imports the shadow module in
    the job's cwd despite ``PYTHONSAFEPATH``; the launch is refused first."""
    python = _real_python(version)
    if python is None:
        pytest.skip(f"no Python {version}")
    shadow = env.run / "json"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("open('PWNED', 'w')\n")
    probe_env = {"PATH": "/usr/bin:/bin", **launch.job_env(env.run)}
    import subprocess

    out = subprocess.run(
        [python, "-c", "import sys, json; print(*sys.version_info[:2])"],
        cwd=env.run,
        env=probe_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (env.run / "PWNED").exists() is shadowed
    (env.run / "PWNED").unlink(missing_ok=True)
    shutil.rmtree(shadow / "__pycache__", ignore_errors=True)
    env.with_policy(allow_code_suffixes=(".py",))
    major, minor = out.stdout.split() if out.stdout else ("3", "10")

    async def probe() -> tuple[int, int]:
        return int(major), int(minor)

    monkeypatch.setattr(env.runner, "_python_version", probe)
    approved = manifest("python3 -c 'import json'", inputs=[], outputs=[])
    policy.approve(approved, "root")
    call = env.call(
        "task_launch",
        release=approved.release_hash,
        task="t",
        task_id="t",
        job_dir=str(env.run / "j"),
        cwd=str(env.run),
        bindings={},
        env={},
    )
    if shadowed:
        with pytest.raises(RuntimeError, match=r"older than 3\.11"):
            await call
    else:
        await call
        await wait_exit(env.run / "j")


# --- M4: the checker fails closed on shapes it has no entry for -------------

REFUSED = [
    "'bash' ${x}",
    '"bash" ${x}',
    "\\bash ${x}",
    "b'a'sh ${x}",
    "rbash ${x}",
    "shopt -s expand_aliases; alias b=bash; b ${x}",
    "cp ${x} y.sh; bash y.sh",
    "ln -s ${x} y.sh; sh y.sh",
    "mv ${x} y.sh; bash y.sh",
    "cp ${x} d/; bash d/x.sh",
    "exec 3< ${x}; bash <&3",
    "fakeroot ${x}",
    "eatmydata ${x}",
    "ssh-agent ${x}",
    "nsenter bash ${x}",
    "perf stat ${x}",
    "mytool run python ${x}",
    "bun ${x}",
    "deno run ${x}",
    "ipython ${x}",
    "octave ${x}",
    "groovy ${x}",
    "pytest ${x}",
    "pip install ${x}",
    "cmake -P ${x}",
    "gdb -x ${x}",
    "expect -f ${x}",
    "node -r ${x}",
    "R CMD BATCH ${x}",
    "sqlite3 db < ${x}",
    "psql -f ${x}",
    "Rscript-4.3 ${x}",
    "rscript ${x}",
    "sed 's/^/ /e' ${x}",
    "sed -n 's/a/b/ge' ${x}",
    "perl -ne 'system($_)' ${x}",
    "unknowntool ${x}",
]
ACCEPTED = [
    "samtools sort -@ 4 ${x} -o out.bam",
    "fastqc ${x}",
    "grep python ${x}",
    "cp ${x} copy.txt; cat copy.txt",
    "sed 's/a/b/' ${x}",
    "sed -n 's/e/x/p' ${x}",
    "apptainer exec img.sif tool --in ${x}",
    "docker exec box tool --in ${x}",
    "stage() { cp $1 .; }; stage ${x}",
    "tool --in ${x}",
]


@pytest.mark.parametrize("command", REFUSED)
def test_unclassifiable_shapes_are_refused(command: str) -> None:
    with pytest.raises(ManifestError):
        manifest(command, inputs=[{"id": "x", "kind": "file"}], outputs=[])


@pytest.mark.parametrize("command", ACCEPTED)
def test_ordinary_commands_still_pass_round5(command: str) -> None:
    manifest(command, inputs=[{"id": "x", "kind": "file"}], outputs=[])


# --- minors: bounded scan ---------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "echo " + '"a" ' * 12000 + "${x}",
        "echo " + "a " * 25000 + "${x}",
        "echo " + "$(a) " * 6000 + "${x}",
        "echo " + "$(" * 150 + ")" * 150 + " ${x}",
        '"' * 30000 + "${x}",
    ],
    ids=["quotes", "words", "substitutions", "nesting", "quote-run"],
)
def test_the_scan_stays_fast_and_refuses_what_it_cannot_bound(
    command: str,
) -> None:
    import time

    started = time.monotonic()
    found = unsafe_placeholders(command, {"x"})
    assert time.monotonic() - started < 1.0
    assert found


# --- minors: a recycled pid is not the job ----------------------------------


async def test_cancel_run_spares_a_recycled_pid_that_leads_its_own_group(
    env: Env,
) -> None:
    import contextlib
    import os
    import signal
    import subprocess

    stranger = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        await env.call("mkdir", path=str(env.run / "o"))
        # recorded as another process with this pid, then reused
        env.guard.add_job(
            RUN1,
            env.run / "j1",
            pid=stranger.pid,
            pgid=stranger.pid,
            start="boot:1",
        )
        await env.call("cancel_run", release="basic")
        assert stranger.poll() is None
        env.guard.add_job(
            RUN1,
            env.run / "j2",
            pid=stranger.pid,
            pgid=stranger.pid,
            start=ops._proc_start(stranger.pid),
        )
        assert await env.call("cancel_run", release="basic") == 1
        stranger.wait(timeout=5)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(stranger.pid, signal.SIGKILL)
        stranger.wait()
