#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
``horus-agent approve --dry-run <dir>``: which tasks of the workflows under
a folder ``approve`` would accept, and why it refuses the others.
"""

import copy
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from horus_agent.manifest import ManifestError, build_manifest_from_workflow

MAX_FILE_BYTES = 8 * 1024 * 1024
_REASON = re.compile(r"^Task '[^']*': \$\{[^}]*\} is (.*?);")


@dataclass
class DryRun:
    """The outcome of :func:`dry_run`."""

    approved: int = 0
    refused: int = 0
    skipped: int = 0
    """Tasks the agent cannot run at all (not shell, not a command)."""
    reasons: Counter[str] = field(default_factory=Counter)
    refusals: list[tuple[str, str]] = field(default_factory=list)
    """``(workflow/task, message)`` of each refused task."""

    @property
    def total(self) -> int:
        """Tasks looked at."""
        return self.approved + self.refused

    def summary(self, top: int = 12) -> str:
        """The counts and the most common refusal reasons."""
        lines = [
            f"{self.total} tasks, {self.approved} approved, "
            f"{self.refused} refused ({self.skipped} not shell tasks, "
            "left out)"
        ]
        lines += [
            f"  {n:5d}  {why}" for why, n in self.reasons.most_common(top)
        ]
        return "\n".join(lines) + "\n"


def _tasks(workflow: Any) -> list[dict[str, Any]]:
    """Every task of a workflow document and of its subworkflows."""
    out: list[dict[str, Any]] = []
    if not isinstance(workflow, dict):
        return out
    for task in workflow.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        sub = task.get("sub") or task.get("body")
        if isinstance(sub, dict):
            out += _tasks(sub)
        else:
            out.append(task)
    return out


def _bundled(folder: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.stat().st_size <= MAX_FILE_BYTES:
            files[path.relative_to(folder).as_posix()] = path.read_bytes()
    return files


def dry_run(directory: Path) -> DryRun:
    """
    Check every task of every ``workflow.yaml`` under *directory* as if it
    ran on an agent: each is moved to an agent target with the shell
    executor and built into a manifest of its own.
    """
    result = DryRun()
    for path in sorted(directory.rglob("workflow.yaml")):
        try:
            workflow = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            continue
        files = _bundled(path.parent)
        for task in _tasks(workflow):
            if (task.get("runtime") or {}).get("kind") not in (
                "command",
                "python_script",
            ):
                result.skipped += 1
                continue
            one = copy.deepcopy(task)
            one["target"] = {"kind": "agent", "agent": "dry-run"}
            one["executor"] = {"kind": "shell"}
            one.pop("sub", None)
            doc = {
                "name": path.parent.name,
                "artifacts": workflow.get("artifacts"),
                "tasks": [one],
                "edges": workflow.get("edges"),
            }
            where = f"{path.parent.name}/{task.get('id')}"
            try:
                build_manifest_from_workflow(doc, files)
            except ManifestError as exc:
                result.refused += 1
                match = _REASON.match(str(exc))
                result.reasons[match.group(1) if match else str(exc)[:80]] += 1
                result.refusals.append((where, str(exc)))
            else:
                result.approved += 1
    return result
