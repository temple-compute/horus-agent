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
import pwd
import re
import stat
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import pydantic

from horus_agent.blobs import open_file
from horus_agent.manifest import (
    Manifest,
    ManifestError,
    check_paths,
    file_digest,
)
from horus_agent.native import digest
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
HORUS_PACKAGE = re.compile(r".+\.[0-9a-f]{8}\.horuspkg")
"""
The runtime's folder package, ``<name>.<8 hex>.horuspkg`` in a working
directory: staged, unpacked (under the rules of any package) and removed.
"""
SAFE_PATH = re.compile(r"[A-Za-z0-9._+,=@:/\[\]-]+")
"""
Characters a path rendered into a command may hold: none of them means
anything to a shell, in any quoting context.
"""
TOOL_CONFIG = frozenset(
    {
        "Makefile",
        "makefile",
        "GNUmakefile",
        "nextflow.config",
        "node_modules",
        "package.json",
        "sitecustomize.py",
        "usercustomize.py",
        "conftest.py",
        "pyvenv.cfg",
        "setup.cfg",
        "pyproject.toml",
        "tox.ini",
        "pytest.ini",
        "snakefile",
        "Snakefile",
    }
)
"""
Names tools read code or settings from in their working directory or a
folder above a script. Together with every name starting with ``.`` (but
the runtime's own ``.horus*``), no op may write them into a run.
"""
CODE_SUFFIXES = (
    ".py",
    ".pyc",
    ".pyo",
    ".pth",
    ".pyd",
    ".dll",
    ".dylib",
    ".jar",
    ".class",
    ".lua",
    ".rb",
    ".pl",
    ".pm",
    ".js",
    ".mjs",
    ".cjs",
    ".node",
    ".r",
    ".rprofile",
)
"""
Suffixes of files an interpreter or the dynamic loader loads on its own
(case-insensitive, and ``.so`` with a version tail). No op may write one
into a run unless ``allow_code_suffixes`` in the policy names it.
"""
SCRIPT_SUFFIXES = (
    ".sh",
    ".bash",
    ".zsh",
    ".ksh",
    ".csh",
    ".fish",
    ".php",
    ".jl",
    ".m",
    ".tcl",
    ".ps1",
    ".ts",
    ".groovy",
    ".gradle",
    ".nf",
    ".smk",
    ".rules",
    ".mk",
    ".cmake",
    ".bat",
    ".cmd",
    ".vbs",
    ".pyz",
    ".whl",
)
SCRIPT_NAMES = frozenset(
    {
        "Rakefile",
        "justfile",
        "Justfile",
        "Taskfile.yml",
        "Taskfile.yaml",
        "Dockerfile",
        "Jenkinsfile",
        "CMakeLists.txt",
        "Vagrantfile",
        "Gemfile",
        "Pipfile",
        "Procfile",
        "gulpfile.js",
        "Gruntfile.js",
        "build.xml",
        "pom.xml",
        "meson.build",
        "Earthfile",
        "docker-compose.yml",
    }
)
"""
Names a build tool or a shell runs as code when started in their folder or
given on its command line. Below the top of a run, a file with one of them
is staged only where the release declares the name.
"""
EXECUTION_ID = re.compile(r"[0-9a-f]{32}")
"""Name of the folder the runtime makes for one run of a task: uuid4 hex."""
CODE_DIRS = frozenset({"__pycache__", "node_modules", "site-packages"})
"""Folder names interpreters search for code; never allowed."""
_SHARED_OBJECT = re.compile(r"\.so(\.[0-9]+)*")
IDLE_RUN_S = 600.0
"""A run with no job yet counts as active this long after its last op."""
RESERVED_NAMES = frozenset({"pid", "exit_code", "slurm_job_id", "env"})
"""Files the agent itself writes in a job dir; ops may not write them."""
META_NAMES = frozenset(
    {".horus_resource.txt", ".horus_gpu.csv", "resource-usage.csv"}
)
"""Names of files the agent writes for the resource monitor."""
HORUS_OWN = frozenset({".horus", ".horus_job"}) | META_NAMES
SIDE_NAMES = frozenset({"side-artifacts"})
"""Folders the agent makes in a working directory for a task's samples."""
"""The only hidden names the runtime and the agent write into a run."""
HOST_FILE = ".horus_host.txt"
"""The monitor's host snapshot: ``key=value`` lines a task's wrapper writes."""
HOST_LINE = re.compile(r"[A-Za-z0-9_]{1,64}=[ -~]{0,512}")
HOST_MAX_BYTES = 8 * 1024
MONITOR_READS = frozenset({HOST_FILE, ".horus_resource.txt", ".horus_gpu.csv"})
"""Files the resource monitor polls; their refusals are not warnings."""
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
    unpack_sha: str | None = None
    """``unpack``: the package's digest, taken once when it was checked."""
    unpacked: bool = False
    """``unpack``: the destination already holds this package."""
    same: bool = False
    """``put_file``: the file already holds exactly these bytes."""


class Guard:
    """Applies one machine's :class:`Policy` to the ops it receives."""

    def __init__(self, policy: Policy, approvals: Approvals) -> None:
        """Enforce *policy*, with *approvals* as the approved releases."""
        self.policy = policy
        self.approvals = approvals
        self.work_dir = Path(policy.work_dir).resolve()
        self.state = self.work_dir / STATE_DIR
        # Jobs share the work dir's group: they may pass through the state
        # dir to the verified copies, never list or change it.
        fresh = not self.work_dir.exists()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if fresh:  # a folder made here is sticky, whatever the umask gave it
            self.work_dir.chmod(self.work_dir.stat().st_mode | stat.S_ISVTX)
        top = self.work_dir.stat().st_mode
        if (
            policy.strict
            and top & (stat.S_IWGRP | stat.S_IWOTH)
            and not top & stat.S_ISVTX
        ):
            raise PolicyRefusedError(
                f"{self.work_dir} is writable by others without the sticky "
                "bit: a job could rename the agent's state folder "
                "(chmod 3770)"
            )
        for folder, mode in (
            (self.state, 0o711),
            (self.state / "runs", 0o700),
            (self.const_store, 0o2750),
        ):
            folder.mkdir(mode=mode, exist_ok=True)
            info = folder.lstat()
            mine = info.st_uid in (os.geteuid(), os.getuid())
            if not stat.S_ISDIR(info.st_mode) or not mine:
                raise PolicyRefusedError(
                    f"{folder} must be the agent's folder"
                )
            if folder == self.const_store:  # readable by the task user's group
                with contextlib.suppress(PermissionError):
                    os.chown(folder, -1, self.work_dir.stat().st_gid)
            folder.chmod(mode)
        self.spilled: set[str] = set()
        """Spill files this agent wrote, the only ones it hands back."""
        self._declared: dict[str, tuple[tuple[str, str, str], ...]] = {}

    @property
    def owners(self) -> frozenset[int]:
        """Uids whose folders a task may run in: the agent's, the task's."""
        uids = {os.geteuid()}
        with contextlib.suppress(KeyError):
            if self.policy.jobs.run_as:
                uids.add(pwd.getpwnam(self.policy.jobs.run_as).pw_uid)
        return frozenset(uids)

    def unlinked(self, raw: str) -> Path:
        """
        *raw* as the real path it names when no link sits below the work
        dir: the configured work dir (maybe itself reached through a link)
        swapped for the real one.
        """
        norm = Path(os.path.normpath(raw))
        for base in (
            Path(os.path.normpath(self.policy.work_dir)),
            self.work_dir,
        ):
            if norm.is_relative_to(base):
                return self.work_dir / norm.relative_to(base)
        return norm

    @property
    def const_store(self) -> Path:
        """Verified copies of constant inputs, by digest; agent-only."""
        return self.state / "const"

    # --- envelope ---------------------------------------------------------

    def admit(
        self, op: Op, *, content: str | None = None, early: bool = False
    ) -> Admitted:
        """
        Check *op*'s envelope and confine its paths. A ``fetch`` passes the
        digest of what it downloaded as *content* (the shipped file of that
        name may be fetched), or *early* when nothing was downloaded yet
        and the destination is only checked for what no download can fix.

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
            if early and manifest is not None:
                # Whatever the shipped file of that name is, the download
                # decides later; everything else is refused up front.
                content = next(iter(self._exact(paths[name], manifest)), None)
            self.check_writable(
                op.run_id,
                paths[name],
                op.method,
                manifest=manifest,
                content=content,
            )
            if not (op.method == "mkdir" and paths[name].is_dir()):
                self.check_sealed(op.run_id, paths[name], op.method)
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

    def records(self) -> dict[str, dict[str, Any]]:
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
        return any(job_running(Path(d), j) for d, j in jobs.items())

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
                for other, rec in self.records().items()
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
        if parts[0] == SPILL_DIR:
            if self.policy.strict:  # a strict agent never spills
                raise PolicyRefusedError(f"Path is the agent's own: {raw}")
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
        for other, rec in self.records().items():
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

    def confine_bound(
        self, run_id: str | None, raw: str, *, code_ok: bool = False
    ) -> Path:
        """
        A path rendered into a command or a job's directory: confined, made
        of :data:`SAFE_PATH` characters only, and not agent bookkeeping.
        *code_ok*: the file is a constant or script that gets verified.
        """
        if not SAFE_PATH.fullmatch(raw):
            raise PolicyRefusedError(
                f"Path has characters a command cannot take safely: {raw!r}"
            )
        path = self.confine(run_id, raw)
        if _meta_name(path):
            raise PolicyRefusedError(f"Path is run metadata: {raw}")
        self.check_name(path, code_ok=code_ok)
        return path

    @staticmethod
    def layout_allows(name: str, allowed: frozenset[str]) -> bool:
        """Whether *name* may sit at the top of a run or working directory."""
        return (
            name in allowed
            or name.split("[", maxsplit=1)[0] in allowed
            or HORUS_PACKAGE.fullmatch(name) is not None
            or name.endswith(".horus-sha256")  # the agent's unpack markers
            or ".horus-tmp" in name  # an unpack in progress
            or ".horus-old" in name
        )

    def layout_names(self, manifest: Manifest | None) -> frozenset[str]:
        """Names a run's folder or a task's working directory may hold."""
        if manifest is not None:
            self.plain(manifest)
        return (
            HORUS_OWN
            | SIDE_NAMES
            | frozenset(manifest.layout if manifest else ())
        )

    @staticmethod
    def plain(manifest: Manifest) -> None:
        """Refuse, at use time, a stored approval with an unsafe path."""
        try:
            check_paths(manifest)
        except ManifestError as exc:
            raise PolicyRefusedError(str(exc)) from exc

    def check_layout(
        self, run_id: str | None, path: Path, manifest: Manifest | None
    ) -> None:
        """
        Cloud-staged files land only at names the approved release declares
        (an input, an output, a script, a task) at the top of the run's
        folder or of a task's working directory.
        """
        if not self.policy.strict or manifest is None:
            return
        allowed = self.layout_names(manifest)
        for root in self.roots(run_id):
            if not path.is_relative_to(root) or path == root:
                continue
            rel = path.relative_to(root).parts
            for part in rel[1:]:
                if problem := self.script_problem(part, allowed):
                    raise PolicyRefusedError(problem)
            tops = [tuple(t.split("/")) for t in manifest.tasks]
            tops += [tuple(t.split("[")[0].split("/")) for t in manifest.tasks]
            depth = 0
            while True:
                if len(rel) <= depth:
                    return
                if (
                    rel[depth] not in allowed
                    and rel[depth].split("[")[0] not in allowed
                    and not HORUS_PACKAGE.fullmatch(rel[depth])
                ):
                    raise PolicyRefusedError(
                        f"{rel[depth]!r} is not an input, output, script or "
                        "task of the approved release, so nothing may be "
                        "staged there"
                    )
                inside = [
                    t for t in tops if rel[: len(t)] == t and len(t) > depth
                ]
                if not inside:
                    return
                depth = max(len(t) for t in inside)
                if depth < len(rel) and EXECUTION_ID.fullmatch(rel[depth]):
                    # The runtime's per-execution folder of the task: one
                    # level, whose content follows the working dir's rules.
                    self.check_execution_dir(root.joinpath(*rel[: depth + 1]))
                    depth += 1

    @staticmethod
    def check_execution_dir(path: Path) -> None:
        """Refuse an execution folder that exists as a link or a file."""
        if os.path.lexists(path) and (path.is_symlink() or not path.is_dir()):
            raise PolicyRefusedError(
                f"{path.name!r} is not a real folder, so nothing may be "
                "staged there"
            )

    @staticmethod
    def script_problem(name: str, allowed: frozenset[str]) -> str | None:
        """Why a script or build file named *name* may not be staged."""
        low = name.lower()
        if name in allowed or name.split("[", 1)[0] in allowed:
            return None
        if name in SCRIPT_NAMES or low.endswith(SCRIPT_SUFFIXES):
            return (
                f"{name!r} is a script or build file the release does not "
                "declare, so it cannot be staged there"
            )
        return None

    def check_name(self, path: Path, *, code_ok: bool = False) -> None:
        """
        Refuse a path in the work dir naming a tool's config file or folder,
        code an interpreter loads on its own, or any hidden name but the
        runtime's own. With *code_ok* the file at the end of the path is an
        approved constant (its content was checked), so its own code suffix
        passes.
        """
        if not self.policy.strict:
            return  # an open agent runs any command anyway
        allow = self.policy.allow_code_suffixes
        parts = path.relative_to(self.work_dir).parts
        for i, part in enumerate(parts):
            problem = name_problem(
                part, allow, code_ok=code_ok and i == len(parts) - 1
            )
            if problem:
                raise PolicyRefusedError(problem)

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

    def running(self, run_id: str | None) -> bool:
        """Whether a job of the run is running (or about to start)."""
        record = self.run_record(run_id) or {}
        return any(
            job_running(Path(d), j)
            for d, j in (record.get("jobs") or {}).items()
        )

    def refused_member(
        self, name: str, trusted: bool = False, *, running: bool = False
    ) -> str | None:
        """
        Why a strict agent does not unpack *name*: a tool config or code
        name, or run metadata (``.horus``, the agent's job and monitor
        files). While a job of the run *running*, not even a code name the
        policy allows (an approved constant's, with *trusted*, is fine).
        """
        if not self.policy.strict:
            return None
        if (
            name == ".horus"
            or name in META_NAMES
            or name in RESERVED_NAMES
            or name.endswith(".horus-sha256")
        ):
            return f"{name!r} is run metadata"
        if running and not trusted and code_suffix(name) is not None:
            return _running_message(name)
        return name_problem(
            name, self.policy.allow_code_suffixes, code_ok=trusted
        )

    def shipped(
        self, path: Path, content: str, manifest: Manifest | None
    ) -> bool:
        """Whether *path* is where *manifest* ships a file or folder of
        digest *content*: its declared path, with exactly that content.
        """
        return manifest is not None and any(
            d == content for d in self._exact(path, manifest)
        )

    def declared(self, manifest: Manifest) -> tuple[tuple[str, str, str], ...]:
        """
        What *manifest* ships, as ``(declared path, name, digest)``: each
        file or folder it bundles (not the files inside a folder), and the
        name the runtime stages it under (the last part of its path).
        """
        found = self._declared.get(manifest.release_hash)
        if found is None:
            self.plain(manifest)
            paths = {p for p, _d in manifest.consts}
            found = tuple(
                (p, p.rsplit("/", 1)[-1], d)
                for p, d in manifest.consts
                if not any(
                    "/".join(parts[:i]) in paths
                    for parts in [p.split("/")]
                    for i in range(1, len(parts))
                )
            )
            self._declared[manifest.release_hash] = found
        return found

    def _names(self, manifest: Manifest) -> list[tuple[str, str]]:
        """Each shipped path (members of a folder too) and its digest, and
        each declared file or folder under the name the runtime stages.
        """
        return [
            *manifest.consts,
            *((n, d) for _p, n, d in self.declared(manifest)),
        ]

    def _exact(self, path: Path, manifest: Manifest) -> list[str]:
        """Digests *manifest* ships at *path* (a declared path or the name
        the runtime stages it under ends it).
        """
        rel = "/" + path.relative_to(self.work_dir).as_posix()
        return [d for p, d in self._names(manifest) if rel.endswith("/" + p)]

    def holds_shipped(self, path: Path, manifest: Manifest | None) -> bool:
        """Whether *path* is, or is inside, a path the release ships."""
        if manifest is None:
            return False
        rel = "/" + path.relative_to(self.work_dir).as_posix() + "/"
        return any(f"/{p}/" in rel for p, _d in self._names(manifest))

    def seal_plan(
        self, manifest: Manifest, cwd: Path, root: Path
    ) -> list[tuple[Path, tuple[str, ...], str]]:
        """
        Where a command may run a shipped file or folder by its literal
        path: ``(folder, path below it, digest)``, for each place the
        release declares (its own path, or its name where the runtime
        stages it) in the working directory and each folder up to the
        run's. Members of a shipped folder are sealed with it.
        """
        bases = [cwd]
        while bases[-1] != root:
            bases.append(bases[-1].parent)
        plan: dict[Path, tuple[Path, tuple[str, ...], str]] = {}
        for path, name, content in self.declared(manifest):
            for base in bases:
                for rel in {tuple(path.split("/")), (name,)}:
                    plan[base.joinpath(*rel)] = (base, rel, content)
        return sorted(plan.values())

    def add_sealed(self, run_id: str | None, path: Path) -> None:
        """Remember that the agent sealed *path* for a launch."""
        record = self.run_record(run_id)
        if run_id is None or record is None:
            return
        if str(path) not in record.setdefault("sealed", []):
            record["sealed"].append(str(path))
            self._save(run_id, record)

    def check_sealed(self, run_id: str | None, path: Path, how: str) -> None:
        """Refuse changing a sealed path, or a folder around or inside one,
        while a job of the run is running.
        """
        record = self.run_record(run_id) or {}
        for sealed in map(Path, record.get("sealed") or ()):
            if not (
                path == sealed
                or path.is_relative_to(sealed)
                or sealed.is_relative_to(path)
            ):
                continue
            if any(
                job_running(Path(d), j)
                for d, j in (record.get("jobs") or {}).items()
            ):
                raise PolicyRefusedError(
                    f"{how} refused: {sealed.name!r} is in use by a running "
                    "task"
                )

    def trusts_tree(
        self, manifest: Manifest | None, dest: Path
    ) -> Callable[[Path], bool]:
        """Whether a folder is the constant folder *manifest* ships at
        *dest*.
        """

        def trusted(tree: Path) -> bool:
            return manifest is not None and self.shipped(
                dest, digest(tree, "deep"), manifest
            )

        return trusted

    def tree_check(
        self, manifest: Manifest | None, dest: Path
    ) -> Callable[[Path], None] | None:
        """
        For an unpack into *dest*: a check of the unpacked tree that keeps
        any name the release ships (an exact path under *dest*) to its
        shipped content, or ``None`` when nothing shipped can be touched.
        """
        if manifest is None or not self.policy.strict:
            return None
        allowed = self.layout_names(manifest)

        def check(tree: Path) -> None:
            if self.shipped(dest, digest(tree, "deep"), manifest):
                return
            for root, dirs, names in os.walk(tree):
                for name in dirs + names:
                    if problem := self.script_problem(name, allowed):
                        raise PolicyRefusedError(problem)
                    where = dest / Path(root, name).relative_to(tree)
                    if not self.holds_shipped(where, manifest):
                        continue
                    file = Path(root, name)
                    now = (
                        digest(file, "deep")
                        if file.is_dir()
                        else file_digest(file.read_bytes())
                    )
                    if not self.shipped(where, now, manifest):
                        raise PolicyRefusedError(
                            f"{name!r} is a name the release ships; only "
                            "its approved content can be staged there"
                        )

        return check

    def code_ok(self, run_id: str | None) -> set[str]:
        """Paths holding code the release approves, by content."""
        return set((self.run_record(run_id) or {}).get("code_ok") or ())

    def add_code_ok(self, run_id: str | None, path: Path) -> None:
        """Remember that *path* holds code the release approves."""
        record = self.run_record(run_id)
        if run_id is None or record is None:
            return
        if str(path) not in record.setdefault("code_ok", []):
            record["code_ok"].append(str(path))
            self._save(run_id, record)

    def end_job(self, run_id: str | None, job_dir: Path) -> None:
        """Note that a job was killed or lost: it holds no run open."""
        if self.job(run_id, job_dir) is not None:
            self.add_job(run_id, job_dir, ended=time.time())

    def mark_stopped(self, run_id: str) -> None:
        """Note that the run's jobs were killed after its release's revoke."""
        record = self.run_record(run_id)
        if record is not None:
            record["stopped"] = True
            self._save(run_id, record)

    def _top(self, run_id: str | None, path: Path) -> tuple[Path, str] | None:
        """The run root holding *path* and the first name below it."""
        for root in self.roots(run_id):
            if path.is_relative_to(root) and path != root:
                return root, path.relative_to(root).parts[0]
        return None

    def note_made(self, run_id: str | None, path: Path) -> None:
        """Remember that the agent staged or produced something at *path*."""
        record = self.run_record(run_id)
        found = self._top(run_id, path)
        if run_id is None or record is None or found is None:
            return
        entry = str(found[0] / found[1])
        if entry not in record.setdefault("made", []):
            record["made"].append(entry)
            self._save(run_id, record)

    def check_digest(self, run_id: str | None, path: Path) -> None:
        """
        With ``uploads = false`` a digest (an equality test for a guessed
        file) is answered only for what this run staged or produced.
        """
        if (
            self.policy.uploads
            or not self.policy.strict
            or not os.path.lexists(path)  # nothing there to confirm
        ):
            return
        found = self._top(run_id, path)
        made = (self.run_record(run_id) or {}).get("made") or ()
        if found is None or str(found[0] / found[1]) not in made:
            raise PolicyRefusedError(
                f"Uploads are off on this agent; {path.name} was not staged "
                "or produced by this run, so its digest stays here"
            )

    def add_meta(self, run_id: str | None, path: Path) -> None:
        """Let *path* (run metadata) be read back when uploads are off."""
        record = self.run_record(run_id)
        if run_id is None or record is None or str(path) in record["meta"]:
            return
        record["meta"].append(str(path))
        self._save(run_id, record)

    def check_writable(  # noqa: PLR0913
        self,
        run_id: str | None,
        path: Path,
        how: str,
        data: bytes | None = None,
        *,
        manifest: Manifest | None = None,
        content: str | None = None,
    ) -> None:
        """
        Refuse writing *path* when it is run metadata, a job dir of the run
        (or holds or sits in one) or a file the agent writes itself. A
        finished job's dir may be removed.
        """
        if path.name in RESERVED_NAMES or path.name.endswith(".horus-sha256"):
            raise PolicyRefusedError(f"{how} may not write {path.name}")
        if how != "remove":
            if data is not None:
                content = file_digest(data)
            approved = content is not None and self.shipped(
                path, content, manifest
            )
            self.check_name(path, code_ok=approved)
            if (
                not approved
                and self.policy.strict
                and any(
                    code_suffix(part)
                    for part in path.relative_to(self.work_dir).parts
                )
                and self.running(run_id)
            ):
                raise PolicyRefusedError(_running_message(path.name))
            if (
                not approved
                and how not in ("mkdir", "unpack")
                and self.holds_shipped(path, manifest)
            ):
                raise PolicyRefusedError(
                    f"{how} refused: {path.name!r} is a name the release "
                    "ships; only its approved content can be staged there"
                )
            self.check_layout(run_id, path, manifest)
            self.note_made(run_id, path)
            if approved and code_suffix(path.name) is not None:
                self.add_code_ok(run_id, path)
        if path in self.roots(run_id) and how != "mkdir":
            raise PolicyRefusedError(f"{how} refused: the run's directory")
        meta = _meta_name(path)
        from_runtime = how == "put_file" and runtime_meta(path)
        if meta and not from_runtime and how not in ("remove", "mkdir"):
            raise PolicyRefusedError(f"{how} refused: {path} is run metadata")
        record = self.run_record(run_id) or {}
        for job in map(Path, record.get("jobs") or {}):
            # A folder above a job may be made again: later tasks of the
            # run stage their own directories there (check_layout limits
            # the names).
            above = how == "mkdir" and job.is_relative_to(path) and path != job
            if above or not (
                path == job
                or path.is_relative_to(job)
                or job.is_relative_to(path)
            ):
                continue
            info = record["jobs"][str(job)]
            if how == "remove" and not job_running(job, info):
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
        if str(path) in self.spilled:
            return
        meta = (
            str(path) in (record.get("meta") or ())
            or (in_job and path.name in ("pid", "exit_code"))
            or self.host_snapshot(path)
        )
        with contextlib.suppress(FileNotFoundError):
            meta = meta and path.lstat().st_size <= META_MAX_BYTES
        if not meta:
            raise PolicyRefusedError(
                f"Uploads are off on this agent; {path.name} stays here"
            )

    def host_snapshot(self, path: Path) -> bool:
        """
        Whether *path* is the host snapshot of a task (``.horus_host.txt``)
        and passes as one: a plain file of ``key=value`` lines, small. With
        resource usage on, it is read back like the agent's own samples.
        """
        if path.name != HOST_FILE or not self.policy.resource_usage:
            return False
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                return False
            with open_file(path) as fh:
                data = fh.read(HOST_MAX_BYTES + 1)
        except OSError:
            return True  # nothing to leak; the read reports it missing
        if len(data) > HOST_MAX_BYTES:
            return False
        lines = data.decode(errors="replace").splitlines()
        return all(HOST_LINE.fullmatch(line) for line in lines)

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


def process_alive(pid: int) -> bool:
    """Whether process *pid* still runs (a zombie has ended)."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        if raw[raw.rfind(")") + 2 :].startswith("Z"):
            return False
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def job_running(job_dir: Path, info: dict[str, Any]) -> bool:
    """Whether a job may still run: no exit code, not ended, group alive."""
    if (job_dir / "exit_code").exists() or info.get("ended"):
        return False
    pgid = info.get("pgid")
    if not pgid:
        # Slurm, or a job whose start is still being recorded.
        started = info.get("started")
        return "slurm_job_id" in info or (
            started is not None and time.time() - started < IDLE_RUN_S
        )
    pid = info.get("pid")
    if pid == pgid and not process_alive(pid):
        return False  # the leader is gone (a zombie nobody reaped)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # alive, as the task user
    return True


def code_suffix(name: str, allow: tuple[str, ...] = ()) -> str | None:
    """
    The suffix that makes *name* code an interpreter loads implicitly (not
    among *allow*), or ``None``.
    """
    low = name.lower()
    if name in CODE_DIRS:
        return name
    if _SHARED_OBJECT.search(low):
        found: str | None = ".so"
    else:
        found = next((s for s in CODE_SUFFIXES if low.endswith(s)), None)
    if found is None or found in {a.lower() for a in allow}:
        return None
    return found


def name_problem(
    name: str, allow: tuple[str, ...] = (), *, code_ok: bool = False
) -> str | None:
    """
    Why *name* may not be written into a run, or ``None``. With *code_ok*
    (the file is an approved constant) a code suffix is no problem.
    """
    if name in HORUS_OWN or name.endswith(".horus-sha256"):
        return None
    if name in TOOL_CONFIG or name.startswith(".") or name.endswith(".pth"):
        return _tool_message(name)
    if code_ok and name not in CODE_DIRS:
        return None
    if (found := code_suffix(name, allow)) is not None:
        return (
            f"{name!r} is code that interpreters load on their own "
            f"({found}), so it cannot be staged in a run; to allow it, set "
            f'allow_code_suffixes = ["{found}"] in policy.toml'
            if found.startswith(".")
            else f"{name!r} is a folder interpreters load code from, so it "
            "cannot be staged in a run"
        )
    return None


def _running_message(name: str) -> str:
    return (
        f"{name!r} is code that interpreters load on their own, and a task "
        "of this run is running: it cannot be staged until the task ends"
    )


def _tool_message(name: str) -> str:
    return (
        f"{name!r} is a name tools read settings or code from; it cannot be "
        "written into a run"
    )


def bad_name(name: str, allow: tuple[str, ...] = ()) -> bool:
    """A name no op may write into a run (see :func:`name_problem`)."""
    return name_problem(name, allow) is not None


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
