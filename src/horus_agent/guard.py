#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Checks every op passes before it runs, whoever sent it.

An op must name a known method with valid arguments; under a strict policy
it must also belong to an approved release and to one of its tasks. Every
path it names must sit inside its own run's directory, every URL on an
allowed https host. Each decision is appended to the audit log.
"""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import pydantic

from horus_agent.manifest import Manifest
from horus_agent.policy import Approvals, Policy
from horus_agent.protocol import ARGS, PROTOCOL, Args, Op, PolicyRefusedError

STATE_DIR = ".horus-agent"
"""Agent-only directory under the work dir: run bindings, audit log."""
SPILL_DIR = ".horus-spill"
"""Large command output an open agent hands back as a file."""
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
LIVE_RUN_S = 3600.0
"""A run with no op for this long no longer counts as running."""
RESERVED_NAMES = frozenset({"pid", "exit_code", "slurm_job_id", "env"})
"""Files the agent itself writes in a job dir; ops may not write them."""
META_NAMES = frozenset(
    {".horus_resource.txt", ".horus_gpu.csv", "resource-usage.csv"}
)
META_MAX_BYTES = 64 * 1024
WRITES = {
    "fetch": ("path",),
    "move": ("src", "dst"),
    "pack": ("pkg",),
    "remove": ("path",),
    "mkdir": ("path",),
}
"""
Path arguments each method writes to. ``put_file`` and ``unpack`` are
checked by the caller, which lets an identical rewrite of an input through.
"""
OPEN_ONLY = frozenset({"exec", "launch"})


@dataclass
class Admitted:
    """An op that passed the envelope checks."""

    args: Args
    manifest: Manifest | None
    task: dict[str, Any] | None
    paths: dict[str, Path]
    """Each path argument, resolved and confined."""


class Guard:
    """Applies one machine's :class:`Policy` to the ops it receives."""

    def __init__(self, policy: Policy, approvals: Approvals) -> None:
        """Enforce *policy*, with *approvals* as the approved releases."""
        self.policy = policy
        self.approvals = approvals
        self.work_dir = Path(policy.work_dir).resolve()
        self.state = self.work_dir / STATE_DIR
        (self.state / "runs").mkdir(mode=0o700, parents=True, exist_ok=True)

    # --- envelope ---------------------------------------------------------

    def admit(self, op: Op) -> Admitted:
        """
        Check *op*'s envelope and confine its paths.

        Raises:
            PolicyRefusedError: The op is not allowed.
        """
        model = ARGS.get(op.method)
        if model is None:
            raise PolicyRefusedError(f"Unknown op: {op.method!r}")
        strict = self.policy.strict
        if strict and op.v != PROTOCOL:
            raise PolicyRefusedError(
                f"This agent needs protocol {PROTOCOL} ops; update the "
                "orchestrator"
            )
        if strict and op.method in OPEN_ONLY:
            raise PolicyRefusedError(
                f"{op.method} runs any command; this agent runs approved "
                "workflow releases only"
            )
        try:
            args = model.model_validate(op.args)
        except pydantic.ValidationError as exc:
            raise PolicyRefusedError(
                f"Invalid arguments for {op.method}: {exc.error_count()} "
                "errors"
            ) from exc
        manifest = task = None
        if strict and self.policy.require_release:
            manifest = self.approvals.get(op.release_hash)
            if manifest is None:
                raise PolicyRefusedError(
                    "This workflow release is not approved on this agent. "
                    "On the machine run: sudo horus-agent approve <release>"
                )
        if op.method != "cancel_run" and (strict or op.run_id is not None):
            self._bind(op)
        if manifest is not None and op.task_id is not None:
            task = manifest.task(op.task_id)
            if task is None:
                raise PolicyRefusedError(
                    f"Task {op.task_id!r} is not part of the approved release"
                )
        paths = {
            name: self.confine(op.run_id, value)
            for name in model.paths
            if isinstance(value := getattr(args, name), str)
        }
        for name in WRITES.get(op.method, ()):
            self.check_writable(op.run_id, paths[name], op.method)
        return Admitted(args, manifest, task, paths)

    def _run_file(self, run_id: str) -> Path:
        return self.state / "runs" / f"{run_id}.json"

    def _bind(self, op: Op) -> None:
        """Tie the run to its release on its first op (pre-check 3)."""
        run_id = op.run_id
        if run_id is None or not RUN_ID.fullmatch(run_id):
            raise PolicyRefusedError(f"Invalid run id {run_id!r}")
        path = self._run_file(run_id)
        record = self.run_record(run_id)
        if record is None:
            live = [
                p
                for p in (self.state / "runs").glob("*.json")
                if time.time() - p.stat().st_mtime < LIVE_RUN_S
            ]
            if len(live) >= self.policy.max_concurrent_runs:
                raise PolicyRefusedError(
                    f"{len(live)} runs already active on this agent "
                    f"(max_concurrent_runs = "
                    f"{self.policy.max_concurrent_runs})"
                )
            self._save(
                run_id,
                {
                    "release_hash": op.release_hash,
                    "first_seen": time.time(),
                    "sealed": [],
                    "jobs": [],
                },
            )
        elif record.get("release_hash") != op.release_hash:
            raise PolicyRefusedError(
                "The op's release does not match its run's release"
            )
        else:
            os.utime(path)

    def run_record(self, run_id: str | None) -> dict[str, Any] | None:
        """The binding of *run_id*, or ``None`` before its first op."""
        if run_id is None:
            return None
        try:
            record: dict[str, Any] = json.loads(
                self._run_file(run_id).read_text()
            )
        except FileNotFoundError:
            return None
        return record

    def _save(self, run_id: str, record: dict[str, Any]) -> None:
        path = self._run_file(run_id)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh)
        tmp.replace(path)

    def _update(self, run_id: str | None, key: str, item: str) -> None:
        record = self.run_record(run_id)
        if run_id is None or record is None:
            return
        if item not in record[key]:
            record[key].append(item)
            self._save(run_id, record)

    # --- paths ------------------------------------------------------------

    def confine(self, run_id: str | None, raw: str) -> Path:
        """
        *raw* resolved (symlinks followed) and required to sit in the work
        dir; under a strict policy also in a directory named after the run.

        Raises:
            PolicyRefusedError: The path leaves its run's directory.
        """
        if not raw.startswith("/") or "\0" in raw:
            raise PolicyRefusedError(f"Path must be absolute: {raw!r}")
        path = Path(os.path.realpath(raw))
        if not path.is_relative_to(self.work_dir) or path == self.work_dir:
            raise PolicyRefusedError(f"Path is outside the work dir: {raw}")
        parts = path.relative_to(self.work_dir).parts
        if parts[0] == STATE_DIR:
            raise PolicyRefusedError(f"Path is the agent's own: {raw}")
        if parts[0] == SPILL_DIR and not self.policy.strict:
            return path
        if self.policy.strict and (run_id is None or run_id not in parts):
            raise PolicyRefusedError(f"Path is outside its run's dir: {raw}")
        return path

    def confine_param(self, run_id: str | None, raw: str) -> Path:
        """A run parameter path: in the run's dir or a ``read_roots`` dir."""
        path = Path(os.path.realpath(raw))
        for root in self.policy.read_roots:
            if raw.startswith("/") and path.is_relative_to(root):
                return path
        return self.confine(run_id, raw)

    def seal(self, run_id: str | None, path: Path) -> None:
        """Refuse every later write onto *path* (a verified input)."""
        self._update(run_id, "sealed", str(path))

    def add_job(self, run_id: str | None, job_dir: Path) -> None:
        """Remember a job this agent started for the run."""
        self._update(run_id, "jobs", str(job_dir))

    def is_job(self, run_id: str | None, job_dir: Path) -> bool:
        """Whether this agent started a job in *job_dir* for the run."""
        record = self.run_record(run_id)
        return record is not None and str(job_dir) in record["jobs"]

    def check_writable(self, run_id: str | None, path: Path, how: str) -> None:
        """
        Refuse writing *path* when it is (or holds) a sealed input or a job
        file the agent writes itself.
        """
        if path.name in RESERVED_NAMES:
            raise PolicyRefusedError(f"{how} may not write {path.name}")
        record = self.run_record(run_id) or {"sealed": [], "jobs": []}
        if how == "remove" and path.name == run_id:
            raise PolicyRefusedError("remove refused: the run's directory")
        if how == "remove" and all(
            (Path(job) / "exit_code").exists() for job in record["jobs"]
        ):
            return  # inputs are only held while a job of the run runs
        for sealed in map(Path, record["sealed"]):
            if (
                path == sealed
                or sealed.is_relative_to(path)
                or (path.is_relative_to(sealed))
            ):
                raise PolicyRefusedError(
                    f"{how} refused: {path} holds a verified input"
                )

    def check_readable(self, path: Path) -> None:
        """With ``uploads = false``, only run metadata may be read back."""
        spill = self.work_dir / SPILL_DIR
        if self.policy.uploads or is_meta(path) or path.is_relative_to(spill):
            return
        raise PolicyRefusedError(
            f"Uploads are off on this agent; {path.name} stays here"
        )

    def check_url(self, url: str, downloads: bool) -> None:
        """Require https and an allowed host; ``downloads``/``uploads`` on."""
        if not (self.policy.downloads if downloads else self.policy.uploads):
            what = "Downloads" if downloads else "Uploads"
            raise PolicyRefusedError(f"{what} are off on this agent")
        parts = urlsplit(url)
        if parts.scheme != "https" or (
            parts.hostname not in self.policy.allowed_hosts
        ):
            raise PolicyRefusedError(
                f"URL host {parts.hostname!r} is not allowed (https and "
                "allowed_hosts only)"
            )

    # --- audit ------------------------------------------------------------

    def audit(self, op: Op, verdict: str, reason: str = "") -> None:
        """Append one decision to the audit log (no data, no env values)."""
        paths = [
            v
            for k in ("path", "job_dir", "cwd", "src", "dst", "pkg", "dest")
            if isinstance(v := op.args.get(k), str)
        ]
        line = json.dumps(
            {
                "ts": time.time(),
                "op_id": op.op_id,
                "run_id": op.run_id,
                "release_hash": op.release_hash,
                "task_id": op.task_id,
                "method": op.method,
                "paths": paths,
                "verdict": verdict,
                "reason": reason,
            }
        )
        log = self.policy.audit_path
        log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as fh:
            fh.write(line + "\n")


def is_meta(path: Path) -> bool:
    """
    A runtime metadata file small enough to leave the machine when uploads
    are off: ``.horus/*.json``, a job's ``pid``/``exit_code``, and the
    resource usage files.
    """
    pure = PurePosixPath(path)
    meta = (
        (pure.parent.name == ".horus" and pure.suffix == ".json")
        or pure.name in META_NAMES
        or (pure.name in ("pid", "exit_code"))
    )
    try:
        return meta and path.stat().st_size <= META_MAX_BYTES
    except FileNotFoundError:
        return meta
