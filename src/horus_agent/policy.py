#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
The machine's own policy: what the agent may do, whatever it is asked.

``/etc/horus-agent/policy.toml`` and the approved release manifests in
``/etc/horus-agent/approved/`` belong to root. The agent only reads them and
refuses to start when anyone else could change them. Without a policy file
the strict defaults apply: only approved releases run.
"""

import os
import pwd
import stat
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from horus_agent import config
from horus_agent.config import JobsPolicy
from horus_agent.manifest import Manifest, ManifestError

APPROVALS_TTL_S = 10.0
"""How long a listing of approved hashes is reused; bounds revoke latency."""

OPEN_HINT = (
    'Write mode = "open" to /etc/horus-agent/policy.toml (owned by root) to '
    "accept any workflow on this machine."
)


class PolicyError(Exception):
    """The policy is unreadable, invalid or not protected."""


@dataclass(frozen=True)
class Policy:
    """The effective policy of this machine."""

    work_dir: str
    """Every run's files sit under it."""
    allowed_hosts: tuple[str, ...]
    """Hosts ``fetch`` and ``upload`` may reach, over https only."""
    mode: str = "strict"
    """``strict``: only approved releases; ``open``: any command."""
    require_release: bool = True
    downloads: bool = True
    uploads: bool = False
    """Whether files other than run metadata may leave the machine."""
    logs: str = "off"
    """``full`` or ``off``: whether task output is sent back."""
    resource_usage: bool = True
    """Whether resource samples are taken and may be read back."""
    allow_same_user: bool = False
    """Strict mode without ``[jobs] run_as``: tasks run as the agent."""
    max_put_bytes: int = 4 * 1024 * 1024
    max_fetch_bytes: int = 50 * 1024**3
    max_unpack_bytes: int = 50 * 1024**3
    """Largest total size an ``unpack`` may extract."""
    max_unpack_files: int = 1_000_000
    """Most files and folders an ``unpack`` may extract."""
    max_concurrent_runs: int = 4
    audit_log: str = ""
    """One JSON line per op decision. Empty: ``<work_dir>/.horus-agent``."""
    jobs: JobsPolicy = field(default_factory=JobsPolicy)
    """The ``[jobs]`` table: which Unix user task commands run as."""
    read_roots: tuple[str, ...] = ()
    """Extra directories a path-valued run parameter may name."""
    source: str = "defaults"
    """The file it was read from, or ``defaults``."""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def strict(self) -> bool:
        """Whether only approved releases run."""
        return self.mode == "strict"

    @property
    def audit_path(self) -> Path:
        """Where op decisions are appended."""
        if self.audit_log:
            return Path(self.audit_log)
        return Path(self.work_dir) / ".horus-agent" / "ops.jsonl"

    def summary(self) -> dict[str, Any]:
        """What the agent reports to the broker (informational only)."""
        return {
            "mode": self.mode,
            "require_release": self.require_release,
            "uploads": self.uploads,
            "downloads": self.downloads,
            "logs": self.logs,
            "resource_usage": self.resource_usage,
        }

    def describe(self) -> dict[str, Any]:
        """Every setting, for ``horus-agent policy check``."""
        out = asdict(self)
        out.pop("extra")
        out["audit_log"] = str(self.audit_path)
        return out


_BOOLS = (
    "require_release",
    "downloads",
    "uploads",
    "resource_usage",
    "allow_same_user",
)
_INTS = (
    "max_put_bytes",
    "max_fetch_bytes",
    "max_unpack_bytes",
    "max_unpack_files",
    "max_concurrent_runs",
)


def policy_path() -> Path:
    """``/etc/horus-agent/policy.toml``."""
    return config.POLICY_PATH


def approved_dir() -> Path:
    """``/etc/horus-agent/approved``."""
    return config.POLICY_PATH.parent / "approved"


def check_protected(path: Path) -> None:
    """
    Require *path* and every parent to be owned by root and not writable by
    group or others, so only root can change what the agent obeys.

    Raises:
        PolicyError: Someone else could change *path*.
    """
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise PolicyError(f"{part} must not be a symlink")
        if info.st_uid not in config.POLICY_OWNERS:
            raise PolicyError(f"{part} must be owned by root")
        if info.st_mode & 0o022 and not (
            part != path and info.st_mode & stat.S_ISVTX
        ):
            raise PolicyError(
                f"{part} must not be writable by group or others"
            )


def load_policy(work_dir: str, api_url: str) -> Policy:
    """
    The policy in force: ``policy.toml``, or the strict defaults when there
    is none. *work_dir* and *api_url* (the registration's) give the
    defaults for ``work_dir`` and ``allowed_hosts``.

    Raises:
        PolicyError: The file is invalid or not protected, or *work_dir* is
            not under the policy's.
    """
    path = policy_path()
    for protected in (path, approved_dir()):
        if protected.exists() or protected.is_symlink():
            check_protected(protected)
    host = urlsplit(api_url).hostname
    defaults: dict[str, Any] = {
        "work_dir": work_dir,
        "allowed_hosts": (host,) if host else (),
    }
    if not path.exists():
        return Policy(**defaults)
    try:
        raw = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise PolicyError(f"{path}: {exc}") from exc
    policy = _parse(raw, defaults, str(path))
    wd, own = Path(policy.work_dir), Path(work_dir)
    if not own.is_relative_to(wd):
        raise PolicyError(
            f"The agent's work dir {own} is not under the policy's {wd}"
        )
    return policy


def same_user_problem(policy: Policy) -> str | None:
    """Why *policy* may not run: strict, with tasks as the agent's user."""
    if policy.jobs.run_as:
        try:
            pwd.getpwnam(policy.jobs.run_as)
        except KeyError:
            return (
                f"[jobs] run_as names {policy.jobs.run_as!r}, which is not a "
                "user on this machine"
            )
    if policy.strict and not policy.jobs.run_as and not policy.allow_same_user:
        return (
            "Strict mode runs tasks as their own user: set [jobs] run_as in "
            f"{policy_path()} (or allow_same_user = true, or mode = "
            '"open"). horus-agent policy init writes a starting file.'
        )
    return None


def template(*, open_mode: bool, run_as: str) -> str:
    """A starting ``policy.toml`` for ``horus-agent policy init``."""
    lines = [
        "schema = 1",
        f'mode = "{"open" if open_mode else "strict"}"',
        "# Hosts transfers may reach: the broker's and the storage bucket's.",
        '# allowed_hosts = ["api.example.com", '
        '"bucket.s3.eu-north-1.amazonaws.com"]',
        "uploads = false",
        'logs = "off"',
    ]
    if run_as:
        lines += ["", "[jobs]", f'run_as = "{run_as}"', 'run_as_via = "sudo"']
    return "\n".join(lines) + "\n"


def _parse(  # noqa: PLR0912
    raw: dict[str, Any], defaults: dict[str, Any], src: str
) -> Policy:
    if raw.pop("schema", 1) != 1:
        raise PolicyError(f"{src}: unknown schema")
    jobs = raw.pop("jobs", {})
    data = raw.pop("data", {})
    raw.pop("host_info", None)  # reserved
    values: dict[str, Any] = dict(defaults)
    for key in ("mode", "logs", "work_dir", "audit_log"):
        if key in raw:
            values[key] = _typed(raw.pop(key), str, key, src)
    for key in _BOOLS:
        if key in raw:
            values[key] = _typed(raw.pop(key), bool, key, src)
    for key in _INTS:
        if key in raw:
            values[key] = _typed(raw.pop(key), int, key, src)
    if "allowed_hosts" in raw:
        values["allowed_hosts"] = tuple(
            _typed(h, str, "allowed_hosts", src)
            for h in _typed(raw.pop("allowed_hosts"), list, "hosts", src)
        )
    values["jobs"] = JobsPolicy(
        run_as=_typed(jobs.pop("run_as", ""), str, "run_as", src),
        run_as_via=_typed(jobs.pop("run_as_via", "sudo"), str, "via", src),
    )
    if "read_roots" in data:
        values["read_roots"] = tuple(
            str(Path(_typed(r, str, "read_roots", src)).resolve())
            for r in _typed(data.pop("read_roots"), list, "read_roots", src)
        )
    unknown = [
        *raw,
        *(f"jobs.{k}" for k in jobs),
        *(f"data.{k}" for k in data),
    ]
    if unknown:
        raise PolicyError(f"{src}: unknown settings {sorted(unknown)}")
    if values.get("mode", "strict") not in ("strict", "open"):
        raise PolicyError(f"{src}: mode must be strict or open")
    if values.get("logs", "off") not in ("full", "off"):
        raise PolicyError(f"{src}: logs must be full or off")
    if not str(values["work_dir"]).startswith("/"):
        raise PolicyError(f"{src}: work_dir must be absolute")
    values["work_dir"] = str(Path(values["work_dir"]).resolve())
    return Policy(**values, source=src)


def _typed[T](value: Any, kind: type[T], key: str, src: str) -> T:
    if not isinstance(value, kind) or (
        kind is int and isinstance(value, bool)
    ):
        raise PolicyError(f"{src}: {key} must be a {kind.__name__}")
    return value


class Approvals:
    """The approved release manifests, re-listed at most every few seconds."""

    def __init__(self, directory: Path | None = None) -> None:
        """Read approvals from *directory* (default: ``approved/``)."""
        self.directory = directory or approved_dir()
        self._listed: frozenset[str] = frozenset()
        self._listed_at = float("-inf")
        self._manifests: dict[str, Manifest] = {}

    def hashes(self) -> frozenset[str]:
        """The approved release hashes; a revoke shows within the TTL."""
        now = time.monotonic()
        if now - self._listed_at > APPROVALS_TTL_S:
            try:
                names = os.listdir(self.directory)
            except FileNotFoundError:
                names = []
            self._listed = frozenset(
                n.removesuffix(".json") for n in names if n.endswith(".json")
            )
            self._listed_at = now
        return self._listed

    def get(self, release_hash: str | None) -> Manifest | None:
        """The approved manifest of *release_hash*, or ``None``."""
        if not release_hash or release_hash not in self.hashes():
            return None
        if release_hash not in self._manifests:
            path = self.directory / f"{release_hash}.json"
            try:
                manifest = Manifest.from_json(path.read_text())
            except (OSError, ValueError, KeyError, ManifestError):
                return None
            if manifest.release_hash != release_hash:
                return None
            self._manifests[release_hash] = manifest
        return self._manifests[release_hash]


def approve(
    manifest: Manifest, by: str, directory: Path | None = None
) -> Path:
    """Store *manifest* as approved, world-readable, owner-writable."""
    directory = directory or approved_dir()
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    about = dict(manifest.about)
    about["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    about["approved_by"] = by
    stored = Manifest(manifest.body, about, manifest.warnings)
    path = directory / f"{manifest.release_hash}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(stored.to_json())
    tmp.chmod(0o644)
    tmp.replace(path)
    return path


def revoke(release_hash: str, directory: Path | None = None) -> bool:
    """Remove an approval; ``False`` when it did not exist."""
    directory = directory or approved_dir()
    if not release_hash.isalnum():
        return False
    try:
        (directory / f"{release_hash}.json").unlink()
    except FileNotFoundError:
        return False
    return True
