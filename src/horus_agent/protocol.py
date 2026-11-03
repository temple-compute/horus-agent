#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
Wire format shared by the orchestrator-side target and the agent.

An *op* is one target primitive (``mkdir``, ``launch``, ``get_file``...)
the orchestrator asks an agent to run. The broker queues ops per agent and
hands each result back to the caller waiting on it. Bytes travel base64
encoded; anything larger than :data:`INLINE_LIMIT` goes through presigned
staging URLs instead.

A poll sends ``{"session", "version", "hostname", "holding"}``: the
agent process's session id (new on each start) and the ids of ops it took
but has not had an answer accepted for. The broker reports an op ``lost``
when it was delivered to an older session, or delivered before a poll
that does not list it.

Broker routes, relative to its base URL:

========================================  ==============  ===================
Route                                     Caller          Answer
========================================  ==============  ===================
``POST agents/ops/``                      orchestrator    ``OpStatus``
``GET agents/ops/{op_id}?wait=<s>``       orchestrator    ``OpStatus``
``POST agents/staging/``                  orchestrator    ``StagingUrls``
``POST agents/register``                  agent           ``Registration``
``POST agents/me/poll``                   agent           ``{"ops": [Op]}``,
                                                          or 204 when idle
``POST agents/me/results``                agent           204
========================================  ==============  ===================
"""

import base64
from typing import Any

from pydantic import BaseModel

INLINE_LIMIT = 4 * 1024 * 1024
"""
Largest payload (file, command output, log chunk) carried inside an op.
A whole number of MiB, so ``read_range`` chunks stay MiB-aligned.
"""

POLL_WAIT_S = 20
"""Seconds a broker long-poll may hold the request open."""

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


def b64e(data: bytes) -> str:
    """Encode *data* for a JSON payload."""
    return base64.b64encode(data).decode("ascii")


def b64d(data: str) -> bytes:
    """Decode a payload field written by :func:`b64e`."""
    return base64.b64decode(data)
