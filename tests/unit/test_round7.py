#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Regression tests for the seventh review round of the policy work."""

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from horus_agent import policy, seal
from horus_agent.manifest import Manifest, ManifestError
from horus_agent.protocol import PolicyRefusedError
from tests.unit.test_guard import Env, env
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
