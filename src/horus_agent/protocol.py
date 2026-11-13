#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Wire format shared by the orchestrator-side target and the agent.

An *op* is one target primitive (``mkdir``, ``launch``, ``get_file``...)
the orchestrator asks an agent to run. The broker queues ops per agent and
hands each result back to the caller waiting on it. Bytes travel base64
encoded; anything larger than :data:`INLINE_LIMIT` goes through presigned
staging URLs instead.

A poll sends ``{"session", "version", "hostname", "work_dir", "holding"}``:
the agent process's session id (new on each start), the directory tasks run
in on its machine, and the ids of ops it took but has not had an answer
accepted for. The broker reports an op ``lost``
when it was delivered to an older session, or delivered before a poll
that does not list it.

Broker routes, relative to its base URL:

========================================  ==============  ===================
Route                                     Caller          Answer
========================================  ==============  ===================
``POST agents/ops/``                      orchestrator    ``OpStatus``
``GET agents/ops/{op_id}?wait=<s>``       orchestrator    ``OpStatus``
``POST agents/staging/``                  orchestrator    ``StagingUrls``
``GET agents/info/{name}``                orchestrator    ``AgentInfo``
``POST agents/register``                  agent           ``Registration``
``POST agents/me/poll``                   agent           ``{"ops": [Op]}``,
                                                          or 204 when idle
``POST agents/me/results``                agent           204
========================================  ==============  ===================
"""

import base64
from typing import Any, ClassVar, Literal

from horus_runtime.core.target.exceptions import OperationRefusedError
from pydantic import BaseModel, ConfigDict, Field, model_validator

INLINE_LIMIT = 4 * 1024 * 1024
"""
Largest payload (file, command output, log chunk) carried inside an op.
A whole number of MiB, so ``read_range`` chunks stay MiB-aligned.
"""

POLL_WAIT_S = 20
"""Seconds a broker long-poll may hold the request open."""

PROTOCOL = 2
"""
Op schema this agent speaks. Version 2 ops carry the run's ``release_hash``
and the ``task_id`` they serve, and every method's args are checked.
"""


class PolicyRefusedError(OperationRefusedError):
    """The agent's policy does not allow the op."""


IDEMPOTENT_METHODS = frozenset(
    {
        "launch",
        "poll",
        "read_output",
        "signal",
        "put_file",
        "fetch",
        "get_file",
        "upload",
        "read_range",
        "mkdir",
        "list_dir",
        "path_exists",
        "remove",
        "task_launch",
        "slurm_submit",
        "slurm_status",
        "slurm_cancel",
        "digest",
        "pack",
        "unpack",
        "stat",
    }
)
"""
Methods safe to send again when the agent restarted before answering.
``exec`` is the exception: rerunning an arbitrary command (``sbatch``, say)
could start the work twice. ``launch`` is safe because the agent finds the
job already started in its job directory.
"""


class Op(BaseModel):
    """One target primitive to run on an agent."""

    op_id: str
    method: str
    args: dict[str, Any] = {}
    run_id: str | None = None
    """Set by the broker, so the agent can stop a run's jobs."""
    release_hash: str | None = None
    """Set by the broker: the release the run was launched from, if any."""
    task_id: str | None = None
    """The runtime task the op serves; ``None`` for run-level ops."""
    v: int = 1
    """Op schema version, :data:`PROTOCOL` for checked ops."""


class OpError(BaseModel):
    """Why an op failed on the agent."""

    type: str
    message: str


class OpResult(BaseModel):
    """What the agent sends back for an op."""

    op_id: str
    result: Any = None
    error: OpError | None = None


class OpStatus(BaseModel):
    """The broker's answer while the orchestrator waits on an op."""

    done: bool = False
    result: Any = None
    error: OpError | None = None
    agent_seen_ago: float | None = None
    """Seconds since the agent last polled; ``None`` if it never has."""
    lost: bool = False
    """
    The op was delivered to an agent process that has since restarted, so
    no answer will come.
    """


class StagingUrls(BaseModel):
    """
    Presigned URLs for one staging object, as each side reaches it.

    The orchestrator and the agent may see object storage under different
    hosts (in-cluster vs public), hence a pair per direction.
    """

    orchestrator_put_url: str
    orchestrator_get_url: str
    agent_put_url: str
    agent_get_url: str


class Registration(BaseModel):
    """The broker's answer to a registration."""

    agent_id: str
    name: str
    agent_token: str


class AgentInfo(BaseModel):
    """What the broker knows about an agent a run targets."""

    name: str
    work_dir: str | None = None
    """The agent's working directory, as it last reported it."""
    online: bool = False
    protocol: int = 1
    """Op schema the agent speaks; 1 for agents that do not say."""
    policy: dict[str, Any] | None = None
    """The agent's reported policy summary (informational)."""


class Args(BaseModel):
    """Arguments of one op method; unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")

    paths: ClassVar[tuple[str, ...]] = ()
    """Fields holding a path, confined to the run's directory."""


class PathArgs(Args):
    """An op on one path."""

    paths: ClassVar[tuple[str, ...]] = ("path",)
    path: str


class JobArgs(Args):
    """An op on a job the agent launched in ``job_dir``."""

    paths: ClassVar[tuple[str, ...]] = ("job_dir",)
    job_dir: str
    pid: int | None = None
    """Ignored when checked: the agent reads the pid it recorded."""


class SignalArgs(JobArgs):
    """``signal``."""

    sig: Literal[2, 9, 15]


class ReadOutputArgs(Args):
    """``read_output``."""

    paths: ClassVar[tuple[str, ...]] = ("job_dir",)
    job_dir: str
    stdout_offset: int = Field(ge=0)
    stderr_offset: int = Field(ge=0)


class PutFileArgs(PathArgs):
    """``put_file``."""

    data: str


class FetchArgs(PathArgs):
    """``fetch``."""

    url: str


class UploadArgs(PathArgs):
    """``upload``: the whole file, or ``length`` bytes from ``offset``."""

    url: str
    offset: int | None = Field(default=None, ge=0)
    length: int | None = Field(default=None, ge=0)


class ReadRangeArgs(PathArgs):
    """``read_range``."""

    offset: int = Field(ge=0)
    length: int = Field(ge=0, le=INLINE_LIMIT)


class DigestArgs(PathArgs):
    """``digest``."""

    mode: Literal["deep", "fast"] = "deep"


class PackArgs(Args):
    """``pack``: the contents of folder ``src`` into the tarball ``pkg``."""

    paths: ClassVar[tuple[str, ...]] = ("src", "pkg")
    src: str
    pkg: str


class UnpackArgs(Args):
    """``unpack``: the tarball ``pkg`` into the folder ``dest``."""

    paths: ClassVar[tuple[str, ...]] = ("pkg", "dest")
    pkg: str
    dest: str


class MoveArgs(Args):
    """``move``."""

    paths: ClassVar[tuple[str, ...]] = ("src", "dst")
    src: str
    dst: str


class Binding(Args):
    """What one placeholder of a task's command stands for."""

    path: str | None = None
    value: str | list[str] | None = None

    @model_validator(mode="after")
    def _one(self) -> "Binding":
        if (self.path is None) == (self.value is None):
            raise ValueError("A binding has exactly one of path and value")
        return self


class TaskLaunchArgs(Args):
    """
    ``task_launch`` / ``slurm_submit``: run an approved task's command, which
    the agent renders from the manifest and *bindings*.
    """

    paths: ClassVar[tuple[str, ...]] = ("job_dir", "cwd")
    task_id: str
    job_dir: str
    cwd: str
    bindings: dict[str, Binding] = {}
    env: dict[str, str] = {}
    script: str | None = None
    """Where a ``python_script`` task's script was placed."""


class ExecArgs(Args):
    """Raw ``exec``; only an open agent accepts it."""

    cmd: str
    cwd: str | None = None
    env: dict[str, str] | None = None


class LaunchArgs(ExecArgs):
    """Raw ``launch``; only an open agent accepts it."""

    job_dir: str


ARGS: dict[str, type[Args]] = {
    "exec": ExecArgs,
    "launch": LaunchArgs,
    "task_launch": TaskLaunchArgs,
    "slurm_submit": TaskLaunchArgs,
    "slurm_status": JobArgs,
    "slurm_cancel": JobArgs,
    "poll": JobArgs,
    "read_output": ReadOutputArgs,
    "signal": SignalArgs,
    "put_file": PutFileArgs,
    "fetch": FetchArgs,
    "get_file": PathArgs,
    "upload": UploadArgs,
    "read_range": ReadRangeArgs,
    "mkdir": PathArgs,
    "list_dir": PathArgs,
    "path_exists": PathArgs,
    "remove": PathArgs,
    "digest": DigestArgs,
    "pack": PackArgs,
    "unpack": UnpackArgs,
    "move": MoveArgs,
    "stat": PathArgs,
    "cancel_run": Args,
}
"""Every method a protocol-2 agent knows, with its arguments."""


def b64e(data: bytes) -> str:
    """Encode *data* for a JSON payload."""
    return base64.b64encode(data).decode("ascii")


def b64d(data: str) -> bytes:
    """Decode a payload field written by :func:`b64e`."""
    return base64.b64decode(data)
