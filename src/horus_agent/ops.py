#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
Agent side: run each op on this machine through Horus's own LocalTarget.
"""

import asyncio
import contextlib
import os
import re
import signal
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

from horus_builtin.target.local import LocalTarget
from horus_runtime.core.target.channel import JobHandle
from horus_runtime.logging import horus_logger

from horus_agent import blobs
from horus_agent.logs import safe_url
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


def _read_from(path: Path, offset: int, limit: int) -> bytes:
    """Up to *limit* bytes of *path* from *offset*; empty if missing."""
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            return fh.read(limit)
    except FileNotFoundError:
        return b""


class OpRunner:
    """Runs ops on this machine and remembers recent answers."""

    def __init__(self) -> None:
        """Start with no answered ops and no tracked runs."""
        # macOS tar adds an AppleDouble "._<name>" file per file it packs
        # (extended attributes), which then show up as extra items in folder
        # artifacts. Every command the agent runs inherits this.
        os.environ.setdefault("COPYFILE_DISABLE", "1")
        self._local = LocalTarget()
        self._answered: OrderedDict[str, OpResult] = OrderedDict()
        self._running: dict[str, asyncio.Task[OpResult]] = {}
        self._run_jobs: dict[str, dict[str, int]] = {}
        """Per run: job dir -> process group of each job launched here."""

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
                code = await local.poll(
                    JobHandle(pid=args["pid"], job_dir=args["job_dir"])
                )
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
                await local.send_signal(
                    JobHandle(pid=args["pid"], job_dir=args["job_dir"]),
                    args["sig"],
                )
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
                return self._cancel_run(op.run_id)
        raise ValueError(f"Unknown op: {op.method!r}")

    async def _exec(
        self,
        cmd: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        _check_env(env)
        horus_logger.log.debug(
            f"exec in {cwd or os.getcwd()}: {safe_url(cmd)}"
        )
        proc = await self._local.run_command_sync(cmd, cwd=cwd, env=env)
        out, err = await proc.communicate()
        returncode = await proc.wait()
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
        _check_env(env)
        pid_file = _path(job_dir) / "pid"
        log = horus_logger.log
        if pid_file.exists():
            # An earlier delivery of this launch already started the job.
            pid = int(pid_file.read_text().strip())
            log.debug(f"Job in {job_dir} already started as pid {pid}")
        else:
            log.debug(
                f"launch in {cwd or os.getcwd()} (job {job_dir}): "
                f"{safe_url(cmd)}"
            )
            handle = await self._local.launch(
                cmd, cwd=cwd, env=env, job_dir=job_dir
            )
            pid = int(handle.pid or 0)
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
        base = _path(job_dir)
        limit = INLINE_LIMIT // 2
        out = _read_from(base / "stdout.log", stdout_offset, limit)
        err = _read_from(base / "stderr.log", stderr_offset, limit)
        return {
            "stdout": b64e(out),
            "stderr": b64e(err),
            "more": len(out) == limit or len(err) == limit,
            "exited": (base / "exit_code").exists(),
        }

    def _cancel_run(self, run_id: str | None) -> int:
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
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pgid, signal.SIGKILL)
                killed += 1
                log.debug(
                    f"Cancel run {run_id}: killed group {pgid} ({job_dir})"
                )
        return killed
