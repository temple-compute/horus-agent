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
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_ENV = "HORUS_AGENT_CONFIG"


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
    run_as: str = ""
    """Unix user that runs task commands. Empty: the agent's own user."""
    run_as_via: str = "sudo"
    """How to switch to :attr:`run_as`: ``sudo`` or ``setpriv``."""
    ca_bundle: str = ""
    """CA certificates that verify the broker. Empty: certifi's bundle."""
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
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    # O_CREAT's mode does not apply to a file that already existed.
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(asdict(config), fh)


def load(path: Path) -> AgentConfig:
    """Read the config written by :func:`save`."""
    raw = json.loads(path.read_text())
    raw["job_env"] = tuple(raw.get("job_env", ()))
    return AgentConfig(**raw)
