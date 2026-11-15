#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Agent target: a machine the orchestrator cannot connect to.

The machine runs ``horus-agent``, which dials out to a broker and polls it
for work. Each channel primitive (``launch``, ``put_file``, ``mkdir``...)
becomes an op the broker queues for that agent; the agent runs it on its
own filesystem and posts the result back. Everything above the primitives
(``_dispatch``, ``wait``, ``cancel``, executors, transfers) is the same as
for any other remote target, such as SSH.
"""

import contextlib
import json
import os
import tempfile
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar

from horus_runtime.context import HorusContext, current_task_id
from horus_runtime.core.artifact.base import BaseArtifact
from horus_runtime.core.target.base import BaseTarget
from horus_runtime.core.target.channel import (
    ChannelProcess,
    JobHandle,
    PollingChannelProcess,
    RemoteDirEntry,
    StreamName,
    new_job_dir,
)
from horus_runtime.core.task.exceptions import TaskExecutionError
from horus_runtime.middleware.target_command import (
    TargetCommandMiddlewareContext,
)
from pydantic import PrivateAttr, field_validator

from horus_agent import blobs
from horus_agent.broker import (
    BrokerClient,
    agent_info,
    agent_work_dir,
    broker_client,
)
from horus_agent.protocol import (
    INLINE_LIMIT,
    PROTOCOL,
    AgentInfo,
    PolicyRefusedError,
    StagingUrls,
    b64d,
    b64e,
)
from horus_agent.template import placeholders

try:
    from horus_runtime.context import current_op_task_id
except ImportError:  # horus-runtime without the target file hooks
    current_op_task_id = current_task_id

if TYPE_CHECKING:
    from horus_runtime.core.artifact.store import CacheMode
    from horus_runtime.core.task.base import BaseTask

BROKER_URL_ENV = "HORUS_AGENT_BROKER_URL"
BROKER_TOKEN_ENV = "HORUS_AGENT_BROKER_TOKEN"
BROKER_TOKEN_CONTEXT_KEY_ENV = "HORUS_AGENT_BROKER_TOKEN_CONTEXT_KEY"

REFUSED_EXECUTORS = frozenset({"python_function_external"})
"""
Executors that load bytes read back from the target as code in the
orchestrator (``cloudpickle.loads``). A machine outside the orchestrator's
trust boundary must not be able to run code there.
"""


class _CompletedProcess(ChannelProcess):
    """A command the agent already ran to completion, output buffered."""

    def __init__(self, returncode: int, stdout: bytes, stderr: bytes) -> None:
        self._returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    @property
    def returncode(self) -> int:
        """Exit code of the finished command."""
        return self._returncode

    async def wait(self) -> int:
        """Return the exit code; the command has already finished."""
        return self._returncode

    async def communicate(self) -> tuple[bytes, bytes]:
        """Return the buffered ``(stdout, stderr)``."""
        return self._stdout, self._stderr

    def kill(self) -> None:
        """Nothing to kill: the command has finished."""

    def signal(self, sig: int) -> None:
        """Nothing to signal: the command has finished."""
        del sig

    async def stream(self) -> AsyncGenerator[tuple[StreamName, bytes]]:
        """Yield the buffered stdout lines, then the stderr lines."""
        for name, data in (("stdout", self._stdout), ("stderr", self._stderr)):
            for line in data.splitlines(keepends=True):
                yield name, line  # type: ignore[misc]


def _current_task() -> "BaseTask | None":
    """The runtime task running in this context, if any."""
    task_id = current_task_id()
    try:
        workflow = HorusContext.get_context().workflow
    except RuntimeError:
        return None
    if task_id is None or workflow is None:
        return None
    return next((t for t in workflow.tasks if t.id == task_id), None)


def _value(artifact: BaseArtifact) -> str | list[str] | None:
    """A value artifact's value as the runtime renders it, else ``None``."""
    value = getattr(artifact, "value", None)
    if value is None:
        return None
    if isinstance(value, list):
        return [v if isinstance(v, str) else json.dumps(v) for v in value]
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _run_directory() -> Path | None:
    """The running workflow's run directory, or ``None`` outside a run."""
    try:
        workflow = HorusContext.get_context().workflow
    except RuntimeError:
        return None
    return None if workflow is None else workflow.run_directory


class AgentTarget(BaseTarget):
    """
    Dispatch tasks to a machine running ``horus-agent``.

    The broker is configured on the orchestrator, not in the workflow:
    ``HORUS_AGENT_BROKER_URL`` and either ``HORUS_AGENT_BROKER_TOKEN`` or
    ``HORUS_AGENT_BROKER_TOKEN_CONTEXT_KEY`` (the ``HorusContext.data`` key
    holding a per-run token).
    """

    kind: str = "agent"

    kind_name: ClassVar[str] = "Agent"
    kind_description: ClassVar[str] = (
        "Machine running horus-agent, reached without inbound connections"
    )

    agent: str
    """Name the agent registered under."""

    working_directory: str | None = None
    """
    Where per-task working directories are created on the agent's machine.
    Usually left unset: the agent's own directory (``horus-agent register
    --work-dir``) is used. A relative path is taken under that directory;
    an absolute one is used as is.
    """

    offline_timeout: float = 900.0
    """
    Seconds the agent may go without polling before its tasks are treated
    as lost. Jobs keep running on the machine meanwhile, so an agent
    restart or a short network outage costs nothing.
    """

    detach_by_default: ClassVar[bool] = True

    # Each poll is a broker round trip; a running task costs two ops per
    # interval (poll + read_output).
    poll_interval: ClassVar[float] = 5.0

    _logs: dict[str, tuple[bytearray, bytearray]] = PrivateAttr(
        default_factory=dict
    )

    @field_validator("working_directory")
    @classmethod
    def _inside(cls, value: str | None) -> str | None:
        """Keep a relative working directory inside the agent's own."""
        if value and not value.startswith("/") and ".." in value.split("/"):
            raise ValueError(
                "A relative working_directory cannot contain '..'"
            )
        return value

    @property
    def resolved_working_directory(self) -> str:
        """
        ``working_directory`` if absolute, else the agent's reported
        directory (with a relative ``working_directory`` under it).
        """
        own = self.working_directory
        if own and own.startswith("/"):
            return own
        base = agent_work_dir(*self._broker_config(), self.agent)
        return f"{base}/{own}" if own and own != "." else base

    @property
    def location_id(self) -> str:
        """Return a stable URI for this agent's machine."""
        return f"{self.kind}://{self.agent}"

    def access_cost(self, artifact: BaseArtifact) -> float | None:
        """Return ``None``; artifacts always need a transfer to the agent."""
        del artifact
        return None

    def path_on_target(self, artifact: BaseArtifact) -> str:
        """
        Path where *artifact* lands on the agent's machine after transfer.

        The same mapping as the SSH target: a path already under the working
        directory stays put, an output under the run directory keeps its
        relative path (two outputs sharing a name never collide), anything
        else lands as ``working_directory/<name>``.
        """
        base = Path(self.resolved_working_directory)
        for candidate in (artifact.path, artifact.declared_path):
            if candidate is not None and candidate.is_relative_to(base):
                return str(candidate)
        run_directory = _run_directory()
        if (
            run_directory is not None
            and artifact.path != run_directory
            and artifact.path.is_relative_to(run_directory)
        ):
            return str(base / artifact.path.relative_to(run_directory))
        return f"{self.resolved_working_directory}/{artifact.path.name}"

    def bind(self, task: "BaseTask") -> None:
        """
        Refuse executors that would load agent-written bytes as code in the
        orchestrator, then bind as usual.
        """
        if task.executor.kind in REFUSED_EXECUTORS:
            raise TaskExecutionError(
                f"Executor {task.executor.kind!r} cannot run on an agent "
                f"target; use python_function or a script runtime instead"
            )
        try:
            strict = self.strict
        except TaskExecutionError:
            strict = False  # no broker yet: the agent still checks each op
        own = self.working_directory
        base = None
        if strict and own and own.startswith("/"):
            with contextlib.suppress(TaskExecutionError):  # no broker yet
                base = agent_work_dir(*self._broker_config(), self.agent)
        if base is not None and not PurePosixPath(str(own)).is_relative_to(
            base
        ):
            raise TaskExecutionError(
                f"working_directory {own} is outside agent {self.agent!r}'s "
                f"work dir {base}"
            )
        if strict and task.executor.kind != "shell":
            raise TaskExecutionError(
                f"Agent {self.agent!r} runs approved shell tasks only, not "
                f"executor {task.executor.kind!r}"
            )
        super().bind(task)

    @staticmethod
    def _broker() -> BrokerClient:
        """The broker client configured on this orchestrator."""
        return broker_client(*AgentTarget._broker_config())

    @staticmethod
    def _broker_config() -> tuple[str, str]:
        """The broker URL and token configured on this orchestrator."""
        url = os.environ.get(BROKER_URL_ENV)
        token = os.environ.get(BROKER_TOKEN_ENV)
        key = os.environ.get(BROKER_TOKEN_CONTEXT_KEY_ENV)
        if token is None and key is not None:
            value = HorusContext.get_context().data.get(key)
            token = None if value is None else str(value)
        if not url or not token:
            raise TaskExecutionError(
                f"No broker configured for agent targets: set "
                f"{BROKER_URL_ENV} and {BROKER_TOKEN_ENV} (or "
                f"{BROKER_TOKEN_CONTEXT_KEY_ENV})"
            )
        return url, token

    def _info(self) -> AgentInfo:
        """The agent's reported protocol and policy."""
        return agent_info(*self._broker_config(), self.agent)

    @property
    def protocol(self) -> int:
        """Op schema the agent speaks."""
        return min(self._info().protocol, PROTOCOL)

    @property
    def strict(self) -> bool:
        """Whether the agent runs approved releases only."""
        policy = self._info().policy or {}
        return self.protocol >= 2 and policy.get("mode", "strict") == "strict"

    async def _call(self, method: str, **args: Any) -> Any:
        """Run *method* on the agent through the broker."""
        v2 = self.protocol >= 2
        return await self._broker().call(
            self.agent,
            method,
            args,
            offline_timeout=self.offline_timeout,
            v=PROTOCOL if v2 else 1,
            task_id=current_op_task_id() if v2 else None,
        )

    def _launch_args(
        self, task: "BaseTask", *, cwd: str | None, job_dir: str, env: Any
    ) -> dict[str, Any]:
        """
        A ``task_launch`` for *task*'s own command: what each placeholder
        stands for, never the rendered command.
        """
        runtime: Any = task.runtime
        script = None
        template = str(getattr(runtime, "command", "") or "")
        if runtime.kind == "python_script":
            template = str(runtime.args)
            raw = str(runtime.script)
            refs = placeholders(raw)
            script = (
                self.path_on_target(
                    next(a for a in task.inputs if a.id in refs)
                )
                if refs
                else f"{task.working_dir}/{Path(raw).name}"
            )
        artifacts = {a.id: a for a in (*task.inputs, *task.outputs)}
        bindings: dict[str, dict[str, Any]] = {}
        for name in placeholders(template):
            artifact = artifacts.get(name)
            if artifact is None:
                continue
            value = _value(artifact)
            bindings[name] = (
                {"path": self.path_on_target(artifact)}
                if value is None
                else {"value": value}
            )
        return {
            "task_id": task.id,
            "job_dir": job_dir,
            "cwd": cwd or task.working_dir,
            "bindings": bindings,
            "env": dict(env or {}),
            "script": script,
        }

    async def _staging(self) -> StagingUrls:
        """Presigned URLs for one staging object."""
        return await self._broker().staging(self.offline_timeout)

    async def run_command_sync(
        self,
        cmd: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ChannelProcess:
        """
        Run *cmd* on the agent to completion and return its buffered output.

        Output is buffered; long-running work goes through :meth:`launch`,
        the default for this target.
        """
        result = await self._call("exec", cmd=cmd, cwd=cwd, env=env)
        return _CompletedProcess(
            result["returncode"],
            await self._output(result, "stdout"),
            await self._output(result, "stderr"),
        )

    async def _output(self, result: dict[str, Any], name: str) -> bytes:
        """A command's output stream, fetched from a file when large."""
        if name in result:
            return b64d(result[name])
        path = result[f"{name}_file"]
        try:
            return await self.get_file(path)
        finally:
            await self.remove(path)

    async def launch(
        self,
        cmd: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        job_dir: str,
    ) -> JobHandle:
        """
        Start *cmd* detached on the agent's machine and return its handle.

        The job runs in its own session with output in log files under
        *job_dir*, so it survives agent restarts.
        """
        result = await self._call(
            "launch", cmd=cmd, cwd=cwd, env=env, job_dir=job_dir
        )
        return JobHandle(pid=result["pid"], job_dir=job_dir)

    async def _run_command(
        self, ctx: TargetCommandMiddlewareContext
    ) -> ChannelProcess:
        """
        A task's own command on a strict agent is a ``task_launch``: the
        agent renders the approved command itself, so ``ctx.command`` (maybe
        rewritten by middleware) is not sent. Anything else runs as given.
        """
        task = getattr(ctx, "task", None)  # newer horus-runtime only
        if task is None or not self.strict:
            return await super()._run_command(ctx)
        job_dir = new_job_dir(ctx.cwd or self.resolved_working_directory)
        result = await self._call(
            "task_launch",
            **self._launch_args(
                task, cwd=ctx.cwd, job_dir=job_dir, env=ctx.env
            ),
        )
        return PollingChannelProcess(
            self, JobHandle(pid=result["pid"], job_dir=job_dir)
        )

    async def poll(self, handle: JobHandle) -> int | None:
        """``None`` while running; the recorded exit code once finished."""
        result: int | None = await self._call(
            "poll", pid=handle.pid, job_dir=handle.job_dir
        )
        return result

    async def read_output(self, handle: JobHandle) -> tuple[bytes, bytes]:
        """
        Return the job's ``(stdout, stderr)`` so far.

        Only the bytes written since the last call cross the broker; the
        rest is kept here, so a long log is not re-sent on every poll.
        """
        out, err = self._logs.setdefault(
            handle.job_dir, (bytearray(), bytearray())
        )
        while True:
            chunk = await self._call(
                "read_output",
                job_dir=handle.job_dir,
                stdout_offset=len(out),
                stderr_offset=len(err),
            )
            new_out, new_err = b64d(chunk["stdout"]), b64d(chunk["stderr"])
            out += new_out
            err += new_err
            if not chunk["more"]:
                if chunk["exited"]:
                    # Complete; a later call re-reads from the start.
                    del self._logs[handle.job_dir]
                return bytes(out), bytes(err)

    async def send_signal(self, handle: JobHandle, sig: int) -> None:
        """Signal the job's whole process group (best effort)."""
        try:
            await self._call(
                "signal", pid=handle.pid, job_dir=handle.job_dir, sig=sig
            )
        except TaskExecutionError:
            pass

    async def put_file(self, content: bytes | Path, remote_path: str) -> None:
        """
        Write *content* to *remote_path* on the agent's machine.

        Small payloads ride in the op; larger ones are staged in object
        storage, which the agent then downloads.
        """
        size = (
            content.stat().st_size
            if isinstance(content, Path)
            else len(content)
        )
        if size <= INLINE_LIMIT:
            data = (
                content.read_bytes() if isinstance(content, Path) else content
            )
            await self._call("put_file", path=remote_path, data=b64e(data))
            return
        urls = await self._staging()
        await blobs.upload(urls.orchestrator_put_url, content)
        await self._call("fetch", url=urls.agent_get_url, path=remote_path)

    async def _get_large(self, remote_path: str) -> str:
        """Have the agent upload *remote_path*; return a URL to read it."""
        urls = await self._staging()
        await self._call("upload", path=remote_path, url=urls.agent_put_url)
        return urls.orchestrator_get_url

    async def get_file(self, remote_path: str) -> bytes:
        """Read *remote_path* from the agent's machine."""
        result = await self._call("get_file", path=remote_path)
        if "data" in result:
            return b64d(result["data"])
        return await blobs.download(await self._get_large(remote_path))

    async def get_file_to(self, remote_path: str, local_path: Path) -> None:
        """Copy *remote_path* into *local_path*, streamed when large."""
        result = await self._call("get_file", path=remote_path)
        if "data" in result:
            local_path.write_bytes(b64d(result["data"]))
            return
        url = await self._get_large(remote_path)
        # Stream into a sibling temp file so a failed download never leaves
        # a truncated file behind.
        fd, tmp = tempfile.mkstemp(dir=local_path.parent)
        os.close(fd)
        try:
            await blobs.download_to(url, Path(tmp))
            Path(tmp).replace(local_path)
        finally:
            Path(tmp).unlink(missing_ok=True)

    async def read_range(
        self, remote_path: str, offset: int, length: int
    ) -> AsyncIterator[bytes]:
        """
        Yield *length* bytes of *remote_path* from *offset*, in op-sized
        chunks.

        Raises:
            ValueError: If *offset* is not MiB-aligned or *length* is
                negative.
        """
        if offset < 0 or offset % (1024 * 1024) or length < 0:
            raise ValueError("offset must be a non-negative multiple of 1 MiB")
        end = offset + length
        while offset < end:
            want = min(INLINE_LIMIT, end - offset)
            result = await self._call(
                "read_range", path=remote_path, offset=offset, length=want
            )
            yield b64d(result)
            offset += want

    async def mkdir(self, path: str) -> None:
        """Create *path* and missing parents on the agent's machine."""
        await self._call("mkdir", path=path)

    async def list_dir(self, path: str) -> list[RemoteDirEntry]:
        """List the immediate children of *path*, skipping symlinks."""
        entries = await self._call("list_dir", path=path)
        return [RemoteDirEntry(**entry) for entry in entries]

    async def path_exists(self, path: str) -> bool:
        """Whether *path* exists on the agent's machine."""
        exists: bool = await self._call("path_exists", path=path)
        return exists

    async def remove(self, path: str) -> None:
        """Remove *path* recursively; a missing path is not an error."""
        await self._call("remove", path=path)

    # --- native ops (protocol 2) -------------------------------------------

    async def path_digest(
        self, path: str, mode: "CacheMode" = "deep"
    ) -> str | None:
        """The runtime digest of *path*, taken by the agent."""
        if self.protocol < 2:
            return await super().path_digest(path, mode)
        try:
            result: str = await self._call("digest", path=path, mode=mode)
        except PolicyRefusedError:
            raise
        except TaskExecutionError:
            return None
        return result

    async def pack_folder(self, src: str, pkg: str) -> None:
        """The contents of folder *src* into the gzip tarball *pkg*."""
        if self.protocol < 2:
            return await super().pack_folder(src, pkg)
        await self._call("pack", src=src, pkg=pkg)

    async def unpack_folder(self, pkg: str, dest: str) -> None:
        """The tarball *pkg* into folder *dest*."""
        if self.protocol < 2:
            return await super().unpack_folder(pkg, dest)
        await self._call("unpack", pkg=pkg, dest=dest)

    async def move_file(self, src: str, dst: str) -> None:
        """Move *src* onto *dst*, creating parents."""
        if self.protocol < 2:
            return await super().move_file(src, dst)
        await self._call("move", src=src, dst=dst)

    async def file_size(self, path: str) -> int:
        """Size of the file at *path*."""
        if self.protocol < 2:
            return await super().file_size(path)
        result = await self._call("stat", path=path)
        if result is None:
            raise FileNotFoundError(path)
        return int(result["size"])

    async def fetch_url(self, url: str, path: str) -> None:
        """Have the agent download *url* into *path*."""
        await self._call("fetch", url=url, path=path)

    async def upload_url(
        self,
        url: str,
        path: str,
        offset: int | None = None,
        length: int | None = None,
    ) -> str | None:
        """Have the agent PUT *path* (or a range of it); its ETag."""
        result = await self._call(
            "upload", url=url, path=path, offset=offset, length=length
        )
        return None if result is None else result.get("etag")

    # --- Slurm through the agent (protocol 2) -------------------------------

    async def slurm_submit(
        self, *, cwd: str | None, env: dict[str, str] | None, job_dir: str
    ) -> str:
        """
        Submit the running task as a Slurm job whose script the agent renders
        from the approved release; returns the job id.

        Raises:
            TaskExecutionError: No task is running in this context.
        """
        task = _current_task()
        if task is None:
            raise TaskExecutionError("slurm_submit needs a running task")
        result = await self._call(
            "slurm_submit",
            **self._launch_args(task, cwd=cwd, job_dir=job_dir, env=env),
        )
        return str(result["job_id"])

    async def slurm_status(self, job_dir: str) -> dict[str, Any]:
        """Queue state, exit code and final accounting of the job."""
        result: dict[str, Any] = await self._call(
            "slurm_status", job_dir=job_dir
        )
        return result

    async def slurm_cancel(self, job_dir: str) -> None:
        """Cancel the job submitted from *job_dir*."""
        await self._call("slurm_cancel", job_dir=job_dir)
