#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
The release manifest: what an approved agent may run, and its hash.
"""

import copy
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from horus_agent.manifest import (
    Manifest,
    ManifestError,
    build_manifest,
    build_manifest_from_workflow,
    canonical_json,
)

DATA = Path(__file__).parent.parent / "data" / "manifest"


def archive(workflow: dict[str, Any], files: dict[str, bytes]) -> bytes:
    """A release zip of *workflow* and *files*, members in the given order."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("workflow.yaml", yaml.safe_dump(workflow, sort_keys=False))
        for path, data in files.items():
            zf.writestr(path, data)
    return buf.getvalue()


def task(**over: Any) -> dict[str, Any]:
    """One agent task; *over* replaces fields."""
    base: dict[str, Any] = {
        "id": "t",
        "name": "T",
        "kind": "horus_task",
        "target": {"kind": "agent", "agent": "lab"},
        "runtime": {"kind": "command", "command": "cat ${a} > ${b}"},
        "executor": {"kind": "shell"},
        "inputs": [{"id": "a", "kind": "file", "path": "inputs/a.txt"}],
        "outputs": [{"id": "b", "kind": "file", "path": "b.txt"}],
    }
    base.update(over)
    return base


def manifest(*tasks: dict[str, Any], **files: bytes) -> Manifest:
    """The manifest of a workflow holding *tasks*."""
    wf = {"name": "w", "kind": "horus_workflow", "tasks": list(tasks)}
    return build_manifest_from_workflow(
        wf, {"inputs/a.txt": b"A\n", **{k: v for k, v in files.items()}}
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    sorted(json.loads((DATA / "expected.json").read_text()).items()),
)
def test_golden_vectors(name: str, expected: str) -> None:
    assert build_manifest((DATA / name).read_bytes()).release_hash == expected


def test_order_of_members_and_keys_does_not_matter() -> None:
    wf = {"name": "w", "kind": "horus_workflow", "tasks": [task()]}
    one = archive(wf, {"inputs/a.txt": b"A", "x": b"1"})
    reordered = {"tasks": [dict(reversed(task().items()))], "name": "w"}
    two = archive(reordered, {"x": b"1", "inputs/a.txt": b"A"})
    assert build_manifest(one).release_hash == build_manifest(two).release_hash


def test_about_is_not_hashed() -> None:
    wf = {"name": "w", "kind": "horus_workflow", "tasks": [task()]}
    renamed = {**wf, "name": "other", "description": "new"}
    one = build_manifest(archive(wf, {"inputs/a.txt": b"A"}), {"version": 1})
    two = build_manifest(archive(renamed, {"inputs/a.txt": b"A"}))
    assert one.release_hash == two.release_hash
    assert one.about == {"name": "w", "version": 1}


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t["runtime"].update(command="cat ${a} >> ${b}"),
        lambda t: t.update(secrets={"TOKEN": "${secret:x}"}),
        lambda t: t["inputs"].append({"id": "c", "kind": "string"}),
        lambda t: t["target"].update(working_directory="sub"),
    ],
)
def test_runnable_changes_change_the_hash(change: Any) -> None:
    changed = copy.deepcopy(task())
    change(changed)
    assert manifest(task()).release_hash != manifest(changed).release_hash


def test_constant_file_byte_changes_the_hash() -> None:
    wf = {"name": "w", "kind": "horus_workflow", "tasks": [task()]}
    one = build_manifest_from_workflow(wf, {"inputs/a.txt": b"A"})
    two = build_manifest_from_workflow(wf, {"inputs/a.txt": b"B"})
    assert one.release_hash != two.release_hash


def test_roles() -> None:
    t = task(
        runtime={"kind": "command", "command": "x $a $c $d $e $f"},
        inputs=[
            {"id": "a", "kind": "file", "path": "inputs/a.txt"},
            {"id": "c", "kind": "string"},
            {"id": "d", "kind": "number", "value": 1.5},
            {"id": "e", "kind": "folder", "path": "inputs/dir"},
            {"id": "f", "kind": "file"},
        ],
    )
    files = {"inputs/dir/x": b"x"}
    inputs = manifest(t, **files).tasks["t"]["inputs"]
    assert inputs["a"]["role"] == "constant" and "sha256" in inputs["a"]
    assert inputs["c"] == {"kind": "string", "role": "param"}
    assert inputs["d"] == {
        "kind": "number",
        "role": "constant",
        "value": "1.5",
    }
    assert inputs["e"]["sha256"].startswith("deep:")
    assert inputs["f"]["role"] == "param"


def test_edge_inputs_and_cloud_tasks() -> None:
    wf = {
        "name": "w",
        "tasks": [
            task(id="a"),
            task(id="b", target={"kind": "local"}),
            task(id="c"),
        ],
        "edges": [
            {
                "source": "a",
                "source_output": "b",
                "target": "c",
                "target_input": "a",
            }
        ],
    }
    m = build_manifest_from_workflow(wf, {"inputs/a.txt": b"A"})
    assert set(m.tasks) == {"a", "c"}
    assert m.tasks["c"]["inputs"]["a"]["role"] == "edge"
    assert m.task("a[03]") is m.tasks["a"]
    assert m.task("zzz") is None


def test_round_trip_checks_the_stored_hash() -> None:
    m = manifest(task())
    assert Manifest.from_json(m.to_json()).release_hash == m.release_hash
    raw = json.loads(m.to_json())
    raw["body"]["tasks"]["t"]["runtime"]["command"] = "rm -rf ~"
    with pytest.raises(ManifestError, match="hash"):
        Manifest.from_json(json.dumps(raw))


@pytest.mark.parametrize(
    ("over", "match"),
    [
        (
            {"runtime": {"kind": "command", "command": "cat ${a.path}"}},
            "a.path",
        ),
        ({"runtime": {"kind": "command", "command": "x ${task.dir}"}}, "task"),
        ({"executor": {"kind": "python_function"}}, "executor"),
        ({"executor": {"kind": "docker", "image": "x"}}, "executor"),
        ({"runtime": {"kind": "python", "code": "x"}}, "runtime"),
        (
            {"target": {"kind": "agent", "working_directory": "/etc"}},
            "working_directory",
        ),
        (
            {"target": {"kind": "agent", "working_directory": "../x"}},
            "working_directory",
        ),
        (
            {"target": {"kind": "ssh", "inner": {"kind": "agent"}}},
            "chain",
        ),
        (
            {
                "runtime": {
                    "kind": "python_script",
                    "script": "missing.py",
                }
            },
            "not in the release",
        ),
        (
            {
                "runtime": {
                    "kind": "python_script",
                    "script": "${c}",
                },
                "inputs": [{"id": "c", "kind": "file"}],
            },
            "bundled input",
        ),
        ({"secrets": {"LD_PRELOAD": "x"}}, "LD_PRELOAD"),
        ({"secrets": {"PATH": "x"}}, "PATH"),
        ({"kind": "loop_controller"}, "kind"),
        (
            {
                "target": {
                    "kind": "slurm",
                    "partition": "a b",
                    "inner": {"kind": "agent"},
                }
            },
            "partition",
        ),
        (
            {
                "target": {
                    "kind": "slurm",
                    "extra_sbatch_args": ["--x\n#SBATCH --y"],
                    "inner": {"kind": "agent"},
                }
            },
            "extra_sbatch_args",
        ),
        (
            {
                "target": {
                    "kind": "slurm",
                    "nodes": "2",
                    "inner": {"kind": "agent"},
                }
            },
            "integer",
        ),
    ],
)
def test_builder_refusals(over: dict[str, Any], match: str) -> None:
    with pytest.raises(ManifestError, match=match):
        manifest(task(**over))


def test_shared_slurm_allocation_is_refused() -> None:
    wf = {
        "name": "w",
        "targets": {
            "pool": {"kind": "slurm", "inner": {"kind": "agent"}},
        },
        "tasks": [task(target="pool")],
    }
    with pytest.raises(ManifestError, match="shared"):
        build_manifest_from_workflow(wf, {})


def test_slurm_hop_keeps_fields_and_resources() -> None:
    m = build_manifest((DATA / "slurm.zip").read_bytes())
    slurm = m.tasks["big"]["target"]["slurm"]
    assert slurm["partition"] == "gpu"
    assert slurm["resources"] == {"cpus": 16, "memory_gb": 64, "gpus": 1}


@pytest.mark.parametrize(
    "data",
    [b"not a zip", archive({"name": "w"}, {})[:-30]],
)
def test_bad_archives(data: bytes) -> None:
    with pytest.raises(ManifestError):
        build_manifest(data)


def test_archive_without_workflow_or_yaml() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x", b"1")
    with pytest.raises(ManifestError, match=r"workflow\.yaml"):
        build_manifest(buf.getvalue())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("workflow.yaml", b"a: [")
    with pytest.raises(ManifestError, match="not YAML"):
        build_manifest(buf.getvalue())
    with pytest.raises(ManifestError, match="mapping"):
        build_manifest_from_workflow([], {})


def test_floats_are_never_hashed() -> None:
    with pytest.raises(ManifestError):
        canonical_json({"a": [1.5]})


def test_string_params_warn() -> None:
    t = task(
        runtime={"kind": "command", "command": "echo $s"},
        inputs=[{"id": "s", "kind": "string"}],
    )
    assert any("'s'" in w for w in manifest(t).warnings)
