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

import contextlib
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
CONST_DIR = ".horus-const"
"""
Per-run copies of verified constant inputs and scripts, written by the
agent; no op may name a path inside it.
"""
RUN_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|[0-9a-f]{32}"
)
"""Run ids are opaque UUIDs, so one can never name another run's folder."""
SAFE_PATH = re.compile(r"[A-Za-z0-9._+,=@%:/\[\]-]+")
"""
Characters a path rendered into a command may hold: none of them means
anything to a shell, in any quoting context.
"""
IDLE_RUN_S = 600.0
"""A run with no job yet counts as active this long after its last op."""
RESERVED_NAMES = frozenset({"pid", "exit_code", "slurm_job_id", "env"})
"""Files the agent itself writes in a job dir; ops may not write them."""
META_NAMES = frozenset(
    {".horus_resource.txt", ".horus_gpu.csv", "resource-usage.csv"}
)
"""Names of files the agent writes for the resource monitor."""
LOG_NAMES = frozenset({"stdout.log", "stderr.log"})
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
CONTROL_OPS = frozenset(
    {
        "poll",
        "read_output",
        "signal",
        "cancel_run",
        "slurm_status",
        "slurm_cancel",
        "remove",
    }
)
"""Ops that still reach a run whose release was revoked: stop and clean."""


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
            if manifest is None and not self._bound_control(op):
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

    def _bound_control(self, op: Op) -> bool:
        """
        Whether *op* stops or cleans up a run bound before its release was
        revoked: those jobs must stay within reach.
        """
        record = self.run_record(op.run_id)
        return (
            op.method in CONTROL_OPS
            and op.release_hash is not None
            and (record is None or record["release_hash"] == op.release_hash)
            and (record is not None or op.method == "cancel_run")
        )

    def _run_file(self, run_id: str) -> Path:
        return self.state / "runs" / f"{run_id}.json"

    def _records(self) -> dict[str, dict[str, Any]]:
        """Every run binding, by run id."""
        out = {}
        for path in (self.state / "runs").glob("*.json"):
            with contextlib.suppress(OSError, ValueError):
                out[path.stem] = json.loads(path.read_text())
        return out

    def _active(self, run_id: str, record: dict[str, Any]) -> bool:
        """Whether a run still has a job running, or is about to start one."""
        jobs = record.get("jobs") or {}
        if not jobs:
            try:
                idle = time.time() - self._run_file(run_id).stat().st_mtime
            except FileNotFoundError:
                return False
            return idle < IDLE_RUN_S
        return any(not (Path(j) / "exit_code").exists() for j in jobs)

    def _bind(self, op: Op) -> None:
        """Tie the run to its release on its first op (pre-check 3)."""
        run_id = op.run_id
        if run_id is None or not RUN_ID.fullmatch(run_id):
            raise PolicyRefusedError(f"Invalid run id {run_id!r}")
        path = self._run_file(run_id)
        record = self.run_record(run_id)
        if record is None:
            live = [
                other
                for other, rec in self._records().items()
                if self._active(other, rec)
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
                    "roots": [],
                    "jobs": {},
                    "meta": [],
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
        if run_id is None or not RUN_ID.fullmatch(run_id):
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

    # --- paths ------------------------------------------------------------

    def confine(self, run_id: str | None, raw: str) -> Path:
        """
        *raw* resolved (symlinks followed) and required to sit in the work
        dir; under a strict policy also in its run's own directory: the
        folder named after the run, which no other run's folder may contain
        or sit in.

        Raises:
            PolicyRefusedError: The path leaves its run's directory.
        """
        if not raw.startswith("/") or "\0" in raw:
            raise PolicyRefusedError(f"Path must be absolute: {raw!r}")
        path = Path(os.path.realpath(raw))
        if not path.is_relative_to(self.work_dir) or path == self.work_dir:
            raise PolicyRefusedError(f"Path is outside the work dir: {raw}")
        parts = path.relative_to(self.work_dir).parts
        if parts[0] == STATE_DIR or CONST_DIR in parts:
            raise PolicyRefusedError(f"Path is the agent's own: {raw}")
        if parts[0] == SPILL_DIR and not self.policy.strict:
            return path
        if self.policy.strict:
            if run_id is None or run_id not in parts:
                raise PolicyRefusedError(
                    f"Path is outside its run's dir: {raw}"
                )
            self._root(
                run_id,
                self.work_dir.joinpath(*parts[: parts.index(run_id) + 1]),
            )
        return path

    def _root(self, run_id: str, root: Path) -> None:
        """Record *root* as a folder of the run, apart from other runs'."""
        record = self.run_record(run_id)
        if record is None:
            raise PolicyRefusedError(f"Run {run_id} is not bound")
        if str(root) in record["roots"]:
            return
        for other, rec in self._records().items():
            for theirs in map(Path, rec.get("roots") or ()):
                if other != run_id and (
                    root.is_relative_to(theirs) or theirs.is_relative_to(root)
                ):
                    raise PolicyRefusedError(
                        f"Path is outside its run's dir: {root} overlaps "
                        "another run"
                    )
        record["roots"].append(str(root))
        self._save(run_id, record)

    def roots(self, run_id: str | None) -> list[Path]:
        """The folders of the run seen so far."""
        record = self.run_record(run_id) or {}
        return [Path(r) for r in record.get("roots") or ()]

    def confine_param(self, run_id: str | None, raw: str) -> Path:
        """A run parameter path: in the run's dir or a ``read_roots`` dir."""
        path = Path(os.path.realpath(raw))
        for root in self.policy.read_roots:
            if raw.startswith("/") and path.is_relative_to(root):
                return path
        return self.confine(run_id, raw)

    def confine_bound(self, run_id: str | None, raw: str) -> Path:
        """
        A path rendered into a command or a job's directory: confined, made
        of :data:`SAFE_PATH` characters only, and not agent bookkeeping.
        """
        if not SAFE_PATH.fullmatch(raw):
            raise PolicyRefusedError(
                f"Path has characters a command cannot take safely: {raw!r}"
            )
        path = self.confine(run_id, raw)
        if _meta_name(path):
            raise PolicyRefusedError(f"Path is run metadata: {raw}")
        return path

    # --- jobs ---------------------------------------------------------------

    def add_job(self, run_id: str | None, job_dir: Path, **info: Any) -> None:
        """Remember a job this agent started for the run (pid, Slurm id)."""
        record = self.run_record(run_id)
        if run_id is None or record is None:
            return
        record["jobs"].setdefault(str(job_dir), {}).update(info)
        self._save(run_id, record)

    def job(self, run_id: str | None, job_dir: Path) -> dict[str, Any] | None:
        """What the agent recorded when it started the job in *job_dir*."""
        record = self.run_record(run_id) or {}
        job: dict[str, Any] | None = (record.get("jobs") or {}).get(
            str(job_dir)
        )
        return job

    def add_meta(self, run_id: str | None, path: Path) -> None:
        """Let *path* (run metadata) be read back when uploads are off."""
        record = self.run_record(run_id)
        if run_id is None or record is None or str(path) in record["meta"]:
            return
        record["meta"].append(str(path))
        self._save(run_id, record)

    def check_writable(self, run_id: str | None, path: Path, how: str) -> None:
        """
        Refuse writing *path* when it is run metadata, a job dir of the run
        (or holds or sits in one) or a file the agent writes itself. A
        finished job's dir may be removed.
        """
        if path.name in RESERVED_NAMES:
            raise PolicyRefusedError(f"{how} may not write {path.name}")
        if path in self.roots(run_id):
            raise PolicyRefusedError(f"{how} refused: the run's directory")
        meta = _meta_name(path)
        from_runtime = how == "put_file" and runtime_meta(path)
        if meta and not from_runtime and how not in ("remove", "mkdir"):
            raise PolicyRefusedError(f"{how} refused: {path} is run metadata")
        record = self.run_record(run_id) or {}
        for job in map(Path, record.get("jobs") or {}):
            if not (
                path == job
                or path.is_relative_to(job)
                or job.is_relative_to(path)
            ):
                continue
            if how == "remove" and (job / "exit_code").exists():
                continue
            raise PolicyRefusedError(f"{how} refused: {path} holds a job")

    def check_readable(self, run_id: str | None, path: Path) -> None:
        """
        With ``logs = "off"`` no task output leaves; with ``uploads =
        false`` only run metadata the agent or the orchestrator wrote does.
        """
        record = self.run_record(run_id) or {}
        jobs = [Path(j) for j in record.get("jobs") or {}]
        in_job = any(path.parent == job for job in jobs)
        if self.policy.logs == "off" and (
            path.name in LOG_NAMES
            or (in_job and path.name not in ("pid", "exit_code"))
        ):
            raise PolicyRefusedError(
                f"Logs are off on this agent; {path.name} stays here"
            )
        if self.policy.uploads:
            return
        if path.is_relative_to(self.work_dir / SPILL_DIR):
            return
        meta = str(path) in (record.get("meta") or ()) or (
            in_job and path.name in ("pid", "exit_code")
        )
        with contextlib.suppress(FileNotFoundError):
            meta = meta and path.lstat().st_size <= META_MAX_BYTES
        if not meta:
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
                f"URL host {parts.hostname!r} is not allowed (https only; "
                "add it to allowed_hosts in the policy)"
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


def runtime_meta(path: Path) -> bool:
    """A runtime state file the orchestrator writes: ``.horus/<x>.json``."""
    return path.parent.name == ".horus" and path.suffix == ".json"


def _meta_name(path: Path) -> bool:
    """A path named like run metadata, which data may not take."""
    pure = PurePosixPath(path)
    return (
        ".horus" in pure.parts
        or pure.name in META_NAMES
        or pure.name in RESERVED_NAMES
    )
