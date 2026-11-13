#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Agent side: run each op on this machine through Horus's own LocalTarget.
"""

import asyncio
import contextlib
import os
import pwd
import re
import shlex
import shutil
import signal
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

from horus_builtin.target.local import LocalTarget
from horus_runtime.core.target.channel import build_detach_command
from horus_runtime.logging import horus_logger

from horus_agent import blobs, launch, native
from horus_agent import guard as guard_mod
from horus_agent.guard import SPILL_DIR, Admitted, Guard
from horus_agent.logs import redact
from horus_agent.protocol import (
    INLINE_LIMIT,
    Op,
    OpError,
    OpResult,
    PolicyRefusedError,
    TaskLaunchArgs,
    b64d,
    b64e,
)

RESULT_CACHE_SIZE = 1024
"""Answered ops remembered, so a redelivered op is answered, not rerun."""

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

JOB_ENV = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TZ",
        "TERM",
        "TMPDIR",
        # The machine's own outbound settings, which tasks need as well.
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        *(
            name
            for proxy in ("http_proxy", "https_proxy", "no_proxy", "all_proxy")
            for name in (proxy, proxy.upper())
        ),
    }
)
"""Agent environment variables tasks inherit, with ``LC_*``."""

JOB_UMASK = 0o007
"""Task files are shared with the agent through the work dir's group."""

SWITCH_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
"""Where ``sudo`` and ``setpriv`` are looked up: system dirs only."""
SPAWN_ENV = {"PATH": "/usr/bin:/bin"}
"""
The whole environment of the process that starts a task: the task's own
is applied after the user switch.
"""


def job_environment(
    environ: dict[str, str], extra: tuple[str, ...] = ()
) -> dict[str, str]:
    """The part of *environ* tasks may see: :data:`JOB_ENV` and *extra*."""
    keep = JOB_ENV | set(extra)
    return {
        name: value
        for name, value in environ.items()
        if (name in keep or name.startswith("LC_"))
        and _ENV_NAME.fullmatch(name)
    }


def switch_user(user: str, via: str) -> list[str]:
    """
    Command prefix that runs the rest of the command line as *user*, with
    ``sudo`` or ``setpriv`` found by absolute path in :data:`SWITCH_PATH`.

    ``sudo`` needs a sudoers rule letting the agent run anything as *user*;
    ``setpriv`` needs CAP_SETUID and CAP_SETGID (systemd
    ``AmbientCapabilities``) and drops them for the task.
    """
    entry = pwd.getpwnam(user)
    if entry.pw_uid == 0:
        raise ValueError("Tasks must not run as root")
    if via not in ("sudo", "setpriv"):
        raise ValueError(f"run_as_via must be sudo or setpriv, not {via!r}")
    tool = shutil.which(via, path=SWITCH_PATH)
    if tool is None:
        raise ValueError(f"run_as_via is {via}, but {via} is not installed")
    if via == "sudo":
        return [tool, "-n", "-u", user, "--"]
    return [
        tool,
        f"--reuid={entry.pw_uid}",
        f"--regid={entry.pw_gid}",
        "--init-groups",
        "--inh-caps=-all",
        "--",
    ]


def _path(path: str) -> Path:
    """*path* as a ``Path``, which must be absolute."""
    if not path.startswith("/"):
        raise ValueError(f"Path must be absolute: {path!r}")
    return Path(path)


def _check_env(env: dict[str, str] | None) -> None:
    """Reject variable names that would break out of ``export K=V``."""
    for name in env or {}:
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid environment variable name: {name!r}")


def _output(
    name: str, data: bytes, spill: Path | None = None
) -> dict[str, str]:
    """
    *data* inline, or past :data:`INLINE_LIMIT` spilled to a temp file (in
    *spill*, else the system's) the caller fetches and removes like any
    other large file.
    """
    if len(data) <= INLINE_LIMIT:
        return {name: b64e(data)}
    if spill is not None:
        spill.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=f"horus-agent-{name}-", dir=spill)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    horus_logger.log.debug(f"{name} is {len(data)} bytes; spilled to {path}")
    return {f"{name}_file": path}


def _job_file(
    job_dir: str, name: str, offset: int = 0, limit: int = -1
) -> bytes:
    """
    Up to *limit* bytes from *offset* of file *name* in *job_dir*, empty if
    missing. Tasks can write in job dirs: neither may be a symlink.
    """
    base = _path(job_dir)
    if base.is_symlink():
        raise ValueError(f"Refusing to follow a symlink: {base}")
    try:
        fh = blobs.open_file(base / name)
    except FileNotFoundError:
        return b""
    with fh:
        fh.seek(offset)
        data: bytes = fh.read(limit)
    return data


class OpRunner:
    """Runs ops on this machine and remembers recent answers."""

    def __init__(
        self,
        run_as: str = "",
        run_as_via: str = "sudo",
        job_env: tuple[str, ...] = (),
        guard: Guard | None = None,
    ) -> None:
        """
        Start with no answered ops and no tracked runs. Task commands run as
        *run_as* (switching with *run_as_via*), else as the agent's user,
        with the agent environment variables :func:`job_environment` keeps.
        With a *guard*, every op is checked against the machine's policy
        first.
        """
        self._local = LocalTarget()
        self._answered: OrderedDict[str, OpResult] = OrderedDict()
        self._running: dict[str, asyncio.Task[OpResult]] = {}
        self._run_jobs: dict[str, dict[str, int]] = {}
        """Per run: job dir -> process group of each job launched here."""
        self._pids: dict[str, int] = {}
        """Job dir -> pid of each job launched by this process."""
        self._switch = switch_user(run_as, run_as_via) if run_as else []
        self._umask = JOB_UMASK if run_as else 0o077
        self._job_env = job_env
        # macOS tar adds an AppleDouble "._<name>" file per file it packs,
        # which then show up as extra items in folder artifacts.
        self._fixed_env = {"COPYFILE_DISABLE": "1"}
        if run_as:
            home = pwd.getpwnam(run_as).pw_dir
            self._fixed_env |= {
                "HOME": home,
                "USER": run_as,
                "LOGNAME": run_as,
            }
        self.guard = guard
        self._spill = None if guard is None else guard.work_dir / SPILL_DIR
        self._samplers: set[asyncio.Task[None]] = set()

    async def _shell(
        self, script: str, env: dict[str, str] | None, *, capture: bool
    ) -> tuple[int, bytes, bytes]:
        """
        Run ``/bin/sh -c`` *script* as the task user, in a new session.

        The process starts with :data:`SPAWN_ENV` as its whole environment.
        The task's environment (the agent's kept variables, then *env*) is
        read from stdin once the user is switched, so no task value can
        steer the switch or appear on a command line. (The inherited pipe
        is read with ``cat``: ``/dev/stdin`` cannot be reopened as the task
        user.)
        """
        _check_env(env)
        exports = "".join(
            f"export {name}={shlex.quote(value)}\n"
            for name, value in {
                **job_environment(dict(os.environ), self._job_env),
                **self._fixed_env,
                **(env or {}),
            }.items()
        )
        out = (
            asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL
        )
        proc = await asyncio.create_subprocess_exec(
            *self._switch,
            "/bin/sh",
            "-c",
            f'umask {self._umask:03o}; eval "$(cat)"\n{script}',
            stdin=asyncio.subprocess.PIPE,
            stdout=out,
            stderr=out,
            env=SPAWN_ENV,
            start_new_session=True,
        )
        stdout, stderr = await proc.communicate(exports.encode())
        return await proc.wait(), stdout or b"", stderr or b""

    async def _signal(self, pgid: int, sig: int) -> bool:
        """
        Send *sig* to process group *pgid*, as the task user with ``run_as``
        (so only task processes can be reached). Whether it was delivered.
        """
        if self._switch:
            code, _, _ = await self._shell(
                f"kill -{int(sig)} -{int(pgid)}", None, capture=False
            )
            return code == 0
        if pgid == os.getpgrp():
            raise ValueError("Refusing to signal the agent's process group")
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def _job_pid(self, pid: int | None, job_dir: str) -> int | None:
        """*pid*, which must be the job this agent launched in *job_dir*."""
        if pid is None:
            return None
        known = self._pids.get(job_dir)
        if known is None:
            # Launched before a restart: the job dir records it.
            try:
                known = int(_job_file(job_dir, "pid").strip() or 0)
            except ValueError:
                known = 0
        if known <= 1 or known != pid:
            raise ValueError(
                f"No job with pid {pid} was launched in {job_dir}"
            )
        return known

    def _poll(self, pid: int | None, job_dir: str) -> int | None:
        """``None`` while the job runs, then its exit code (-1: lost)."""
        if code := _job_file(job_dir, "exit_code").strip():
            return int(code)
        if pid is None:
            return None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return -1
        except PermissionError:
            pass
        return None

    async def run(self, op: Op) -> OpResult:
        """
        Run *op* once and return its result.

        An op delivered again (the broker resent it, or its first answer
        was lost on the way back) gets the first run's answer.
        """
        if op.op_id in self._answered:
            horus_logger.log.debug(
                f"Op {op.op_id[:8]} delivered again; answering from cache"
            )
            return self._answered[op.op_id]
        task = self._running.get(op.op_id)
        if task is not None:
            horus_logger.log.debug(
                f"Op {op.op_id[:8]} delivered again while running; joining it"
            )
        else:
            task = asyncio.create_task(self._run(op))
            self._running[op.op_id] = task
        try:
            result = await asyncio.shield(task)
        finally:
            self._running.pop(op.op_id, None)
        self._answered[op.op_id] = result
        while len(self._answered) > RESULT_CACHE_SIZE:
            self._answered.popitem(last=False)
        return result

    async def _run(self, op: Op) -> OpResult:
        try:
            admitted = None
            if self.guard is not None:
                admitted = await self._admit(self.guard, op)
                op = op.model_copy(
                    update={
                        "args": {
                            **op.args,
                            **{k: str(v) for k, v in admitted.paths.items()},
                        }
                    }
                )
            result = await self._dispatch(op, admitted)
        except Exception as exc:  # reported to the caller
            return OpResult(
                op_id=op.op_id,
                error=OpError(type=type(exc).__name__, message=str(exc)),
            )
        return OpResult(op_id=op.op_id, result=result)

    async def _admit(self, guard: Guard, op: Op) -> Admitted:
        """Run the policy checks of *op*, auditing the decision."""
        try:
            admitted = guard.admit(op)
            await self._check(guard, op, admitted)
        except PolicyRefusedError as exc:
            guard.audit(op, "refused", str(exc))
            horus_logger.log.warning(f"Refused {op.method}: {exc}")
            raise
        guard.audit(op, "allowed")
        return admitted

    async def _check(  # noqa: PLR0912
        self, guard: Guard, op: Op, admitted: Admitted
    ) -> None:
        """The per-method policy checks (beyond the envelope)."""
        policy = guard.policy
        args, paths = op.args, admitted.paths
        match op.method:
            case "fetch":
                guard.check_url(args["url"], downloads=True)
            case "upload":
                guard.check_url(args["url"], downloads=False)
                guard.check_readable(op.run_id, paths["path"])
            case "get_file" | "read_range":
                guard.check_readable(op.run_id, paths["path"])
            case "put_file":
                data = b64d(args["data"])
                if len(data) > policy.max_put_bytes:
                    raise PolicyRefusedError("put_file is over max_put_bytes")
                path = paths["path"]
                same = (
                    path.is_file()
                    and path.stat().st_size == len(data)
                    and _read(path) == data
                )
                if not same:
                    guard.check_writable(op.run_id, path, "put_file")
                if guard_mod.runtime_meta(path):
                    guard.add_meta(op.run_id, path)
            case "unpack":
                dest = paths["dest"]
                marker = dest.parent / f".{dest.name}.horus-sha256"
                same = (
                    dest.is_dir()
                    and marker.is_file()
                    and (
                        native.read_text(marker) == native.digest(paths["pkg"])
                    )
                )
                if not same:
                    guard.check_writable(op.run_id, dest, "unpack")
            case "poll" | "signal" | "slurm_status" | "slurm_cancel":
                if policy.strict and not guard.job(
                    op.run_id, paths["job_dir"]
                ):
                    raise PolicyRefusedError(
                        f"No job of this run was started in {args['job_dir']}"
                    )
            case "task_launch" | "slurm_submit":
                if admitted.task is None or admitted.manifest is None:
                    raise PolicyRefusedError(
                        f"{op.method} needs a task of an approved release"
                    )

    async def _dispatch(  # noqa: PLR0911, PLR0912, PLR0915
        self, op: Op, admitted: Admitted | None = None
    ) -> Any:
        args = op.args
        local = self._local
        guard = self.guard
        if guard is not None and op.method in ("poll", "signal"):
            # The pid the agent recorded, never the one in the op.
            recorded = guard.job(op.run_id, Path(args["job_dir"])) or {}
            args = {**args, "pid": recorded.get("pid")}
            self._pids.setdefault(args["job_dir"], args["pid"] or 0)
        match op.method:
            case "task_launch" | "slurm_submit" if admitted is not None:
                return await self._task_launch(op, admitted)
            case "slurm_status":
                return await self._slurm_status(
                    self._slurm_id(op.run_id, args["job_dir"]),
                    Path(args["job_dir"]),
                )
            case "slurm_cancel":
                return await self._slurm_cancel(
                    self._slurm_id(op.run_id, args["job_dir"])
                )
            case "read_output" if (
                guard is not None and guard.policy.logs == "off"
            ):
                job = Path(args["job_dir"])
                return {
                    "stdout": "",
                    "stderr": "",
                    "more": False,
                    "exited": (job / "exit_code").exists(),
                }
            case "digest":
                path = Path(args["path"])
                mode = args.get("mode", "deep")
                return await asyncio.to_thread(native.digest, path, mode)
            case "pack":
                await asyncio.to_thread(
                    native.pack, Path(args["src"]), Path(args["pkg"])
                )
                return None
            case "unpack":
                await asyncio.to_thread(
                    native.unpack, Path(args["pkg"]), Path(args["dest"])
                )
                return None
            case "move":
                native.move(Path(args["src"]), Path(args["dst"]))
                return None
            case "stat":
                return native.stat(Path(args["path"]))
            case "exec":
                return await self._exec(**args)
            case "launch":
                return await self._launch(op.run_id, **args)
            case "poll":
                pid = self._job_pid(args["pid"], args["job_dir"])
                code = self._poll(pid, args["job_dir"])
                if code is not None:
                    horus_logger.log.info(
                        f"Job pid {args['pid']} in {args['job_dir']} exited "
                        f"with {code}"
                    )
                    for jobs in self._run_jobs.values():
                        jobs.pop(args["job_dir"], None)
                return code
            case "read_output":
                return self._read_output(**args)
            case "signal":
                pid = self._job_pid(args["pid"], args["job_dir"])
                if pid is not None:
                    with contextlib.suppress(ProcessLookupError):
                        await self._signal(os.getpgid(pid), args["sig"])
                return None
            case "put_file":
                path = _path(args["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with blobs.open_file(path, "wb") as fh:
                    fh.write(b64d(args["data"]))
                return None
            case "fetch":
                await blobs.download_to(
                    args["url"],
                    _path(args["path"]),
                    None if guard is None else guard.policy.max_fetch_bytes,
                )
                return None
            case "get_file":
                with blobs.open_file(_path(args["path"])) as fh:
                    size = os.fstat(fh.fileno()).st_size
                    if size > INLINE_LIMIT:
                        return {"size": size}
                    return {"data": b64e(fh.read())}
            case "upload":
                return await blobs.upload(
                    args["url"],
                    _path(args["path"]),
                    args.get("offset"),
                    args.get("length"),
                )
            case "read_range":
                data = _read(
                    _path(args["path"]), args["offset"], args["length"]
                )
                if len(data) < args["length"]:
                    raise ValueError(
                        f"{args['path']} ended before the requested range"
                    )
                return b64e(data)
            case "mkdir":
                await local.mkdir(str(_path(args["path"])))
                return None
            case "list_dir":
                entries = await local.list_dir(str(_path(args["path"])))
                return [entry._asdict() for entry in entries]
            case "path_exists":
                return await local.path_exists(str(_path(args["path"])))
            case "remove":
                path = _path(args["path"])
                if path == Path("/"):
                    raise ValueError("Refusing to remove /")
                await local.remove(str(path))
                return None
            case "cancel_run":
                return await self._cancel_run(op.run_id)
        raise ValueError(f"Unknown op: {op.method!r}")

    async def _exec(
        self,
        cmd: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        shown = redact(cmd, (env or {}).values())
        horus_logger.log.debug(f"exec in {cwd or os.getcwd()}: {shown}")
        prefix = f"cd {shlex.quote(cwd)} || exit 1\n" if cwd else ""
        returncode, out, err = await self._shell(
            prefix + cmd, env, capture=True
        )
        horus_logger.log.debug(
            f"exec exited {returncode}: {len(out)} bytes stdout, "
            f"{len(err)} bytes stderr"
        )
        return {
            "returncode": returncode,
            **_output("stdout", out, self._spill),
            **_output("stderr", err, self._spill),
        }

    async def _launch(
        self,
        run_id: str | None,
        cmd: str,
        job_dir: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        log = horus_logger.log
        if recorded := _job_file(job_dir, "pid").strip():
            # An earlier delivery of this launch already started the job.
            pid = int(recorded)
            log.debug(f"Job in {job_dir} already started as pid {pid}")
        else:
            log.debug(
                f"launch in {cwd or os.getcwd()} (job {job_dir}): "
                f"{redact(cmd, (env or {}).values())}"
            )
            inner = f"cd {shlex.quote(cwd)} && {cmd}" if cwd else cmd
            await self._shell(
                build_detach_command(inner, job_dir), env, capture=False
            )
            pid = int(_job_file(job_dir, "pid").strip() or 0)
            if pid <= 1:
                raise RuntimeError(f"The job in {job_dir} did not start")
        self._pids[job_dir] = pid
        pgid = None
        with contextlib.suppress(ProcessLookupError):
            pgid = os.getpgid(pid)
        if self.guard is not None:
            self.guard.add_job(run_id, Path(job_dir), pid=pid, pgid=pgid)
        if run_id is not None and pgid is not None:
            self._run_jobs.setdefault(run_id, {})[job_dir] = pgid
            log.debug(f"Job pid {pid} (group {pgid}) belongs to run {run_id}")
        return {"pid": pid}

    async def _task_launch(self, op: Op, admitted: Admitted) -> dict[str, Any]:
        """
        Start an approved task's command, rendered here (or submit it), in a
        job dir the agent creates. A launch delivered again gets the job it
        already started.
        """
        guard = self.guard
        assert guard is not None and admitted.task is not None
        args = admitted.args
        assert isinstance(args, TaskLaunchArgs)
        slurm = op.method == "slurm_submit"
        job_dir = admitted.paths["job_dir"]
        known = guard.job(op.run_id, job_dir) or {}
        key = "slurm_job_id" if slurm else "pid"
        if known.get(key):
            return {"job_id" if slurm else "pid": known[key]}
        if known or job_dir.exists() or job_dir.is_symlink():
            raise PolicyRefusedError(
                f"{job_dir} already exists; each job starts in a new job dir"
            )
        prepared = launch.prepare(guard, op.run_id, admitted, slurm=slurm)
        log = horus_logger.log
        log.info(f"[run {(op.run_id or '-')[:8]}] {op.method} {args.task_id}")
        job_dir.mkdir(parents=True)
        if slurm:
            return await self._sbatch(op.run_id, prepared)
        result = await self._launch(
            op.run_id,
            prepared.command,
            str(job_dir),
            cwd=str(prepared.cwd),
            env=prepared.env,
        )
        if guard.policy.resource_usage:
            guard.add_meta(op.run_id, prepared.side_dir / launch.SAMPLES_FILE)
            guard.add_meta(op.run_id, prepared.cwd / launch.RUSAGE_FILE)
            with contextlib.suppress(ProcessLookupError):
                task = launch.start_sampler(
                    os.getpgid(result["pid"]), prepared
                )
                self._samplers.add(task)
                task.add_done_callback(self._samplers.discard)
        return result

    async def stop_revoked(self) -> int:
        """
        Kill the running jobs of runs whose release is no longer approved
        (``horus-agent revoke``), once per run. Returns the jobs stopped.
        """
        guard = self.guard
        if guard is None or not (
            guard.policy.strict and guard.policy.require_release
        ):
            return 0
        approved = guard.approvals.hashes()
        stopped = 0
        for run_id, record in guard.records().items():
            if record.get("release_hash") in approved or record.get("stopped"):
                continue
            for job_dir, job in (record.get("jobs") or {}).items():
                if (Path(job_dir) / "exit_code").exists():
                    continue
                pid, pgid = job.get("pid"), job.get("pgid")
                with contextlib.suppress(ProcessLookupError):
                    # Only a group the recorded job still leads.
                    if pid and pgid and os.getpgid(pid) == pgid:
                        stopped += await self._signal(pgid, signal.SIGKILL)
                if slurm_id := job.get("slurm_job_id"):
                    with contextlib.suppress(RuntimeError):
                        await self._slurm_cancel(slurm_id)
                        stopped += 1
            guard.mark_stopped(run_id)
            horus_logger.log.warning(
                f"[run {run_id[:8]}] its release was revoked; stopped its jobs"
            )
        return stopped

    def _slurm_id(self, run_id: str | None, job_dir: str) -> str:
        """The Slurm job the agent submitted from *job_dir* for the run."""
        job = (
            None
            if self.guard is None
            else self.guard.job(run_id, Path(job_dir))
        )
        job_id = str((job or {}).get("slurm_job_id") or "")
        if not launch.JOB_ID.fullmatch(job_id):
            raise ValueError(f"No Slurm job was submitted in {job_dir}")
        return job_id

    async def _slurm(
        self, *argv: str, env: dict[str, str] | None = None
    ) -> tuple[int, bytes, bytes]:
        """
        Run one Slurm command as the task user, every argument quoted. *env*
        reaches it through stdin; ``sbatch`` passes it on to the job.
        """
        return await self._shell(shlex.join(argv), env, capture=True)

    async def _sbatch(
        self, run_id: str | None, prepared: launch.Prepared
    ) -> dict[str, Any]:
        """Submit an approved task as a batch job, once."""
        script = prepared.job_dir / "job.sh"
        with blobs.open_file(script, "w") as fh:
            fh.write(launch.sbatch_script(prepared))
            os.fchmod(fh.fileno(), 0o750)
        code, out, err = await self._slurm(
            "sbatch", "--parsable", str(script), env=prepared.env
        )
        if code != 0:
            raise RuntimeError(
                f"sbatch submission failed: {err.decode(errors='replace')}"
            )
        new_id = launch.parse_job_id(out)
        if self.guard is not None:
            self.guard.add_job(run_id, prepared.job_dir, slurm_job_id=new_id)
        return {"job_id": new_id}

    async def _slurm_status(
        self, job_id: str, job_dir: Path
    ) -> dict[str, Any]:
        """The queue state of Slurm job *job_id*, submitted from *job_dir*."""
        exit_code = _job_file(str(job_dir), "exit_code").strip()
        status: dict[str, Any] = {
            "job_id": job_id,
            "state": "",
            "reason": None,
            "nodes": None,
            "exit_code": int(exit_code) if exit_code else None,
            "final": None,
        }
        code, out, err = await self._slurm(
            "squeue", "-h", "-j", job_id, "-o", "%T|%r|%N"
        )
        # Slurm answers "Invalid job id" once the job has left the queue.
        if code != 0 and b"nvalid job id" not in err:
            raise RuntimeError(
                f"squeue failed ({code}): {err.decode(errors='replace')}"
            )
        line = out.decode(errors="replace").strip().splitlines()
        if line:
            state, reason, nodes = [*line[0].split("|"), "", ""][:3]
            status.update(state=state, reason=reason or None, nodes=nodes)
            return status
        _code, out, _err = await self._slurm(
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "-o",
            "State,ExitCode,Reason",
        )
        rows = out.decode(errors="replace").strip().splitlines()
        if rows:
            state, final_code, reason = [*rows[0].split("|"), "", ""][:3]
            status["final"] = {
                "state": state.split()[0] if state else "",
                "exit_code": final_code,
                "reason": reason or None,
            }
        return status

    async def _slurm_cancel(self, job_id: str) -> None:
        """Cancel Slurm job *job_id*."""
        code, _out, err = await self._slurm("scancel", job_id)
        if code != 0:
            raise RuntimeError(
                f"scancel failed: {err.decode(errors='replace')}"
            )

    @staticmethod
    def _read_output(
        job_dir: str, stdout_offset: int, stderr_offset: int
    ) -> dict[str, Any]:
        limit = INLINE_LIMIT // 2
        out = _job_file(job_dir, "stdout.log", stdout_offset, limit)
        err = _job_file(job_dir, "stderr.log", stderr_offset, limit)
        return {
            "stdout": b64e(out),
            "stderr": b64e(err),
            "more": len(out) == limit or len(err) == limit,
            "exited": (_path(job_dir) / "exit_code").exists(),
        }

    async def _cancel_run(self, run_id: str | None) -> int:
        """
        Kill every still-running job this agent launched for *run_id*.

        A finished job is skipped: its process group id may since belong to
        an unrelated process.
        """
        log = horus_logger.log
        killed = 0
        for job_dir, pgid in self._run_jobs.pop(run_id or "", {}).items():
            if (Path(job_dir) / "exit_code").exists():
                log.debug(f"Cancel run {run_id}: {job_dir} already finished")
                continue
            if await self._signal(pgid, signal.SIGKILL):
                killed += 1
                log.debug(
                    f"Cancel run {run_id}: killed group {pgid} ({job_dir})"
                )
        return killed


def _read(path: Path, offset: int = 0, length: int = -1) -> bytes:
    """Up to *length* bytes of *path* from *offset*, never via a symlink."""
    with blobs.open_file(path) as fh:
        fh.seek(offset)
        data: bytes = fh.read(length)
    return data
