#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
The agent's registration, kept in one owner-only JSON file.
"""

import json
import os
import stat
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_ENV = "HORUS_AGENT_CONFIG"

POLICY_PATH = Path("/etc/horus-agent/policy.toml")
"""Settings only root may change; the agent never writes here."""
POLICY_OWNERS = frozenset({0})


@dataclass(frozen=True)
class JobsPolicy:
    """The ``[jobs]`` table of :data:`POLICY_PATH`."""

    run_as: str = ""
    """Unix user that runs task commands. Empty: the agent's own user."""
    run_as_via: str = "sudo"
    """How to switch to :attr:`run_as`: ``sudo`` or ``setpriv``."""


def load_jobs_policy(path: Path | None = None) -> JobsPolicy:
    """
    The ``[jobs]`` table of *path* (default :data:`POLICY_PATH`); defaults
    when the file is missing.

    Raises:
        ValueError: *path* or a directory above it is not owned by root or
            is writable by group or others (a sticky directory such as
            ``/tmp`` is fine), so the agent's user could change it.
    """
    path = path or POLICY_PATH
    if not path.exists():
        return JobsPolicy()
    for item in (path, *path.parents):
        info = item.stat()
        writable = info.st_mode & 0o022 and not (
            item != path and info.st_mode & stat.S_ISVTX
        )
        if info.st_uid not in POLICY_OWNERS or writable:
            raise ValueError(
                f"{path}: {item} must be owned by root and not writable by "
                "group or others"
            )
    jobs = tomllib.loads(path.read_text()).get("jobs", {})
    return JobsPolicy(
        run_as=str(jobs.get("run_as", "")),
        run_as_via=str(jobs.get("run_as_via", "sudo")),
    )


@dataclass(frozen=True)
class AgentConfig:
    """What ``horus-agent register`` stores for ``horus-agent run``."""

    url: str
    """Broker base URL."""

    agent_id: str
    name: str
    token: str
    """The agent's own credential. Never logged."""
    work_dir: str = ""
    """
    Where tasks run on this machine. Empty in configs written before it
    existed: :func:`default_work_dir` then.
    """
    ca_bundle: str = ""
    """
    CA certificates that verify the broker. Empty: ``SSL_CERT_FILE`` or
    ``SSL_CERT_DIR`` when set, else certifi's bundle.
    """
    job_env: tuple[str, ...] = ()
    """Agent environment variables passed to tasks, beyond the defaults."""


def default_work_dir() -> Path:
    """``~/horus-agent``: where tasks run unless ``--work-dir`` says else."""
    return Path.home() / "horus-agent"


def config_path() -> Path:
    """``$HORUS_AGENT_CONFIG``, else ``$XDG_CONFIG_HOME/horus-agent``."""
    if explicit := os.environ.get(CONFIG_ENV):
        return Path(explicit)
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "horus-agent" / "agent.json"


def save(config: AgentConfig, path: Path) -> None:
    """Write *config* to *path*, readable by its owner only."""
    umask = os.umask(0o077)  # also for missing parents such as ~/.config
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    finally:
        os.umask(umask)
    # O_CREAT's mode does not apply to a file that already existed.
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(asdict(config), fh)


def load(path: Path) -> AgentConfig:
    """Read the config written by :func:`save`."""
    raw = json.loads(path.read_text())
    raw["job_env"] = tuple(raw.get("job_env", ()))
    return AgentConfig(**raw)
