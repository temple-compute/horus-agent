#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Regression tests for the seventh review round of the policy work."""

import asyncio
import dataclasses
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from horus_agent import policy, seal
from horus_agent.manifest import Manifest, ManifestError
from horus_agent.protocol import PolicyRefusedError
from tests.unit.test_guard import Env, env, refused
from tests.unit.test_policy_hardening import manifest

__all__ = ["env"]

BAD = [
    "",
    ".",
    "..",
    "../x.sh",
    "a/../../x",
    "a/./b",
    "a//b",
    "/etc/x",
    "a\\b",
    "a\x00b",
    "é.sh",  # decomposed e + accent
    "a.",
    "a. ",
    "dir/a ",
]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize("where", ["input", "output", "script", "id", "dir"])
def test_the_builder_refuses_unplain_declared_paths(
    bad: str, where: str
) -> None:
    task: dict[str, Any] = {}
    files = {"x.sh": b"echo\n"}
    if where == "input":
        task["inputs"] = [{"id": "h", "kind": "file", "path": bad}]
    elif where == "output":
        task["outputs"] = [{"id": "o", "kind": "file", "path": bad}]
    elif where == "script":
        task["runtime"] = {"kind": "python_script", "script": bad}
    elif where == "id":
        task["id"] = bad
    else:
        task["inputs"] = [{"id": "h", "kind": "folder", "path": bad}]
    with pytest.raises(ManifestError):
        manifest("echo", files=files, **task)


def test_a_stored_approval_with_such_a_path_is_not_loaded(
    tmp_path: Path,
) -> None:
    good = manifest(
        "bash a.sh",
        inputs=[{"id": "h", "kind": "file", "path": "a.sh"}],
        files={"a.sh": b"echo\n"},
    )
    evil = dataclasses.replace(
        good,
        consts=tuple(("../../victim/" + p, d) for p, d in good.consts),
    )
    with pytest.raises(ManifestError):
        policy.approve(evil, "root", tmp_path)
    # An older version could have stored it: it is refused when read.
    stored = good.to_json().replace('"a.sh"', '"../a.sh"')
    with pytest.raises(ManifestError):
        Manifest.from_json(stored)


def test_use_time_checks_refuse_an_unplain_stored_path(env: Env) -> None:
    good = manifest(
        "bash a.sh",
        inputs=[{"id": "h", "kind": "file", "path": "a.sh"}],
        files={"a.sh": b"echo\n"},
    )
    evil = dataclasses.replace(
        good, consts=(("../../victim/profile.sh", good.consts[0][1]),)
    )
    with pytest.raises(PolicyRefusedError):
        env.guard.declared(evil)
    with pytest.raises(PolicyRefusedError):
        env.guard.layout_names(
            dataclasses.replace(good, layout=("..", "a.sh"))
        )


@pytest.mark.parametrize(
    "rel", [("..", "x"), ("a", ".."), (), ("/etc",), ("a/b",)]
)
def test_seal_never_leaves_the_run(
    tmp_path: Path, rel: tuple[str, ...]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    victim = tmp_path / "x"
    victim.write_text("original")
    src = tmp_path / "src"
    src.write_text("new")
    with pytest.raises(PolicyRefusedError):
        seal.seal(run, rel, src, frozenset({run.stat().st_uid}))
    assert victim.read_text() == "original"


# --- M2: sealing leaves an intact tree alone ---------------------------------


def _folder(root: Path, count: int = 40) -> Path:
    root.mkdir()
    for i in range(count):
        (root / f"f{i}.txt").write_text(f"data {i}\n")
    (root / "sub").mkdir()
    (root / "sub" / "x.sh").write_text("echo x\n")
    (root / "sub" / "x.sh").chmod(0o755)
    return root


def _inodes(root: Path) -> dict[str, int]:
    return {
        p.relative_to(root).as_posix(): p.stat().st_ino
        for p in root.rglob("*")
    }


def test_sealing_an_intact_tree_changes_nothing(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    stored = _folder(tmp_path / "stored")
    owners = frozenset({os.geteuid()})
    seal.seal(run, ("cfg",), stored, owners)
    before = _inodes(run / "cfg")
    seal.seal(run, ("cfg",), stored, owners)
    assert _inodes(run / "cfg") == before
    assert (run / "cfg" / "sub" / "x.sh").stat().st_mode & 0o111


def test_sealing_replaces_only_what_differs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    stored = _folder(tmp_path / "stored")
    owners = frozenset({os.geteuid()})
    seal.seal(run, ("cfg",), stored, owners)
    before = _inodes(run / "cfg")
    bad = run / "cfg" / "f3.txt"
    bad.chmod(0o644)
    bad.write_text("tampered\n")
    (run / "cfg" / "extra.txt").write_text("extra")
    (run / "cfg" / "f4.txt").unlink()
    (run / "cfg" / "f4.txt").symlink_to("/etc/passwd")
    seal.seal(run, ("cfg",), stored, owners)
    after = _inodes(run / "cfg")
    assert bad.read_text() == "data 3\n"
    assert not (run / "cfg" / "extra.txt").exists()
    assert not (run / "cfg" / "f4.txt").is_symlink()
    changed = {k for k in after if before.get(k) != after[k]}
    assert changed == {"f3.txt", "f4.txt"}


def test_a_running_sibling_never_misses_a_file(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    stored = _folder(tmp_path / "stored", 200)
    owners = frozenset({os.geteuid()})
    seal.seal(run, ("cfg",), stored, owners)
    stop = threading.Event()
    missing: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            for i in range(200):
                try:
                    (run / "cfg" / f"f{i}.txt").read_bytes()
                except OSError:
                    missing.append(f"f{i}")

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for _ in range(8):
            seal.seal(run, ("cfg",), stored, owners)
    finally:
        stop.set()
        thread.join()
    assert missing == []


# --- minors: mount points ----------------------------------------------------


def test_removal_does_not_cross_a_mount_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    (run / "cfg" / "mnt").mkdir(parents=True)
    keep = run / "cfg" / "mnt" / "patient.csv"
    keep.write_text("keep")
    stored = _folder(tmp_path / "stored")
    real = os.lstat

    def lstat(path: Any, *, dir_fd: int | None = None) -> os.stat_result:
        info = real(path, dir_fd=dir_fd)
        if path == "mnt":
            values = list(info)
            values[2] += 1  # st_dev
            return os.stat_result(values)
        return info

    monkeypatch.setattr(os, "lstat", lstat)
    with pytest.raises(PolicyRefusedError):
        seal.seal(run, ("cfg",), stored, frozenset({os.geteuid()}))
    assert keep.read_text() == "keep"


# --- minors: fetch ---


def _big_release(body: bytes) -> Manifest:
    approved = manifest(
        "bash run.sh",
        inputs=[{"id": "s", "kind": "file", "path": "run.sh"}],
        outputs=[],
        files={"run.sh": body},
    )
    policy.approve(approved, "root")
    return approved


async def test_a_large_constant_arrives_by_fetch_with_its_hash(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"# pad\n" * (1 << 20)  # over 4 MiB
    approved = _big_release(body)
    served = {"data": body}

    async def download(_url: str, path: Path, _limit: int | None) -> None:
        path.write_bytes(served["data"])

    monkeypatch.setattr("horus_agent.ops.blobs.download_to", download)
    args: dict[str, Any] = {
        "release": approved.release_hash,
        "path": str(env.run / "run.sh"),
        "url": "https://files.example.org/x",
    }
    served["data"] = body + b"touch PWNED\n"
    await refused(env, "fetch", "ships", **args)
    assert not (env.run / "run.sh").exists()
    served["data"] = body
    await env.call("fetch", **args)
    assert (env.run / "run.sh").read_bytes() == body
    assert not list(env.run.glob(".horus-tmp*"))


async def test_a_slow_fetch_does_not_hold_the_run_lock(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def download(_url: str, path: Path, _limit: int | None) -> None:
        started.set()
        await release.wait()
        path.write_bytes(b"data")

    monkeypatch.setattr("horus_agent.ops.blobs.download_to", download)
    fetching = asyncio.create_task(
        env.call(
            "fetch",
            path=str(env.run / "a"),
            url="https://files.example.org/x",
        )
    )
    await started.wait()
    try:
        await asyncio.wait_for(env.call("mkdir", path=str(env.run / "b")), 2)
    finally:
        release.set()
    await fetching
    assert (env.run / "a").read_bytes() == b"data"
