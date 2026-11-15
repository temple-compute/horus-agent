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
_MAX_LINKS = 40


def protected_real_path(path: Path) -> Path:
    """
    The real path of *path*, after checking that only root can change it.

    *path* itself must not be a symlink. A symlink in the directories above
    it (``/etc`` on macOS) is followed when its owner is in
    :data:`POLICY_OWNERS`, since only that owner can retarget it. Every
    component of the resolved path must be owned by root and not writable
    by group or others (a sticky directory such as ``/tmp`` is fine above
    *path*). Use the returned path, not *path*, to read or write.

    Raises:
        ValueError: Someone else could change *path*; the message names the
            component and why.
        OSError: *path* or a component does not exist.
    """
    cur = Path("/")
    todo = [(name, False) for name in reversed(path.parts[1:])]
    if todo:
        todo[0] = (todo[0][0], True)
    links = 0
    while todo:
        name, final = todo.pop()
        if name == "..":
            cur = cur.parent
            continue
        nxt = cur / name
        info = nxt.lstat()
        if not stat.S_ISLNK(info.st_mode):
            cur = nxt
            continue
        if final:
            raise ValueError(f"{nxt} must not be a symlink")
        if info.st_uid not in POLICY_OWNERS:
            raise ValueError(
                f"{nxt} is a symlink that root does not own: replace it with "
                "a directory or a root-owned symlink"
            )
        links += 1
        if links > _MAX_LINKS:
            raise ValueError(f"{path}: too many symlinks")
        target = Path(os.readlink(nxt))
        if target.is_absolute():
            cur = Path("/")
        todo.extend(
            (n, False) for n in reversed(target.parts[target.is_absolute() :])
        )
    for part in (cur, *cur.parents):
        info = part.lstat()
        if info.st_uid not in POLICY_OWNERS:
            raise ValueError(f"{part} must be owned by root")
        if info.st_mode & 0o022 and not (
            part != cur and info.st_mode & stat.S_ISVTX
        ):
            raise ValueError(f"{part} must not be writable by group or others")
    return cur


def read_protected(path: Path) -> str:
    """
    Text of the root-protected file *path*, read through the checked real
    path without following a symlink at the end.

    Raises:
        ValueError: As :func:`protected_real_path`.
    """
    fd = os.open(protected_real_path(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as fh:
        return fh.read()


@dataclass(frozen=True)
class JobsPolicy:
    """The ``[jobs]`` table of :data:`POLICY_PATH`."""

    run_as: str = ""
    """Unix user that runs task commands. Empty: the agent's own user."""
    run_as_via: str = "sudo"
    """How to switch to :attr:`run_as`: ``sudo`` or ``setpriv``."""
    home: str = ""
    """
    ``HOME`` of approved jobs. Empty: the :attr:`run_as` user's home, or a
    fresh one per job without :attr:`run_as`.
    """


def load_jobs_policy(path: Path | None = None) -> JobsPolicy:
    """
    The ``[jobs]`` table of *path* (default :data:`POLICY_PATH`); defaults
    when the file is missing.

    Raises:
        ValueError: *path* or a directory above it is not owned by root,
            is a symlink root does not own, or is writable by group or
            others (a sticky directory such as ``/tmp`` is fine), so the
            agent's user could change it.
    """
    path = path or POLICY_PATH
    if not path.exists():
        return JobsPolicy()
    try:
        text = read_protected(path)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    jobs = tomllib.loads(text).get("jobs", {})
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
