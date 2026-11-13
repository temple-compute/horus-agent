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
import errno
import os
import pwd
import re
import shlex
import shutil
import signal
import stat
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, BinaryIO

from horus_builtin.target.local import LocalTarget
from horus_runtime.core.target.channel import build_detach_command
from horus_runtime.logging import horus_logger

from horus_agent import blobs
from horus_agent.logs import redact
from horus_agent.protocol import (
    INLINE_LIMIT,
    Op,
    OpError,
    OpResult,
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


def _output(name: str, data: bytes) -> dict[str, str]:
    """
    *data* inline, or past :data:`INLINE_LIMIT` spilled to a temp file the
    caller fetches (and removes) like any other large file.
    """
    if len(data) <= INLINE_LIMIT:
        return {name: b64e(data)}
    fd, path = tempfile.mkstemp(prefix=f"horus-agent-{name}-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    horus_logger.log.debug(f"{name} is {len(data)} bytes; spilled to {path}")
    return {f"{name}_file": path}


def _open_regular(path: Path) -> BinaryIO:
    """
    *path* opened for reading. A symlink is refused whatever it points to,
    and so is anything but a regular file (a FIFO would block the agent).
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"Refusing to follow a symlink: {path}") from exc
        raise
    fh = os.fdopen(fd, "rb")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        fh.close()
        raise ValueError(f"Not a regular file: {path}")
    return fh


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
        fh = _open_regular(base / name)
    except FileNotFoundError:
        return b""
    with fh:
        fh.seek(offset)
        return fh.read(limit)


class OpRunner:
    """Runs ops on this machine and remembers recent answers."""

    def __init__(
        self,
        run_as: str = "",
        run_as_via: str = "sudo",
        job_env: tuple[str, ...] = (),
    ) -> None:
        """
        Start with no answered ops and no tracked runs. Task commands run as
        *run_as* (switching with *run_as_via*), else as the agent's user,
        with the agent environment variables :func:`job_environment` keeps.
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
            result = await self._dispatch(op)
        except Exception as exc:  # reported to the caller
            return OpResult(
                op_id=op.op_id,
                error=OpError(type=type(exc).__name__, message=str(exc)),
            )
        return OpResult(op_id=op.op_id, result=result)

    async def _dispatch(self, op: Op) -> Any:  # noqa: PLR0911, PLR0912
        args = op.args
        local = self._local
        match op.method:
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
                await local.put_file(
                    b64d(args["data"]), str(_path(args["path"]))
                )
                return None
            case "fetch":
                await blobs.download_to(args["url"], _path(args["path"]))
                return None
            case "get_file":
                path = _path(args["path"])
                size = path.stat().st_size
                if size > INLINE_LIMIT:
                    return {"size": size}
                return {"data": b64e(path.read_bytes())}
            case "upload":
                await blobs.upload(args["url"], _path(args["path"]))
                return None
            case "read_range":
                chunks = [
                    chunk
                    async for chunk in local.read_range(
                        str(_path(args["path"])),
                        args["offset"],
                        args["length"],
                    )
                ]
                return b64e(b"".join(chunks))
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
            **_output("stdout", out),
            **_output("stderr", err),
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
        if run_id is not None:
            with contextlib.suppress(ProcessLookupError):
                pgid = os.getpgid(pid)
                self._run_jobs.setdefault(run_id, {})[job_dir] = pgid
                log.debug(
                    f"Job pid {pid} (group {pgid}) belongs to run {run_id}"
                )
        return {"pid": pid}

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
