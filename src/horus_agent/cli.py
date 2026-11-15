#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
``horus-agent register`` and ``horus-agent run``.
"""

import argparse
import asyncio
import contextlib
import dataclasses
import getpass
import grp
import json
import os
import platform
import pwd
import signal
import socket
import stat
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
from horus_runtime.logging import horus_logger

from horus_agent import config, dryrun, logs, ops, policy
from horus_agent import guard as guard_mod
from horus_agent.daemon import (
    Agent,
    AgentUnauthorizedError,
    agent_version,
    broker_tls,
    check_broker_url,
)
from horus_agent.manifest import (
    Manifest,
    ManifestError,
    build_manifest,
    task_runs_python_code,
)
from horus_agent.protocol import Registration

TOKEN_ENV = "HORUS_AGENT_TOKEN"
"""Registration token, when not typed or piped in."""


def _job_group(jobs: config.JobsPolicy) -> tuple[str, int] | None:
    """Name and gid of the task user's group, None without ``run_as``."""
    if not jobs.run_as:
        return None
    try:
        gid = pwd.getpwnam(jobs.run_as).pw_gid
        return grp.getgrgid(gid).gr_name, gid
    except KeyError:
        return None


def layout_problem(path: Path, jobs: config.JobsPolicy) -> str | None:
    """
    The command that fixes the work dir's layout under ``run_as`` (group of
    the task user, sticky bit), None when it is right or there is no
    ``run_as``.
    """
    if (group := _job_group(jobs)) is None:
        return None
    info = path.stat()
    fixes = []
    if info.st_gid != group[1]:
        fixes.append(f"chgrp {group[0]} {path}")
    if not info.st_mode & stat.S_ISVTX:
        fixes.append(f"chmod 3770 {path}")
    if not fixes:
        return None
    return f"The work dir needs the run_as layout; run: {' && '.join(fixes)}"


def _work_dir(
    value: str | None, jobs: config.JobsPolicy, *, repair: bool = True
) -> Path:
    """
    *value* as an absolute path, created if missing, with no access for
    other users. With ``run_as`` a new dir gets mode 3770 and, as root, the
    task user's group; a plain 0770 one gets the sticky bit. A stricter or
    symlinked dir is left alone (``layout_problem`` names the fix).
    """
    raw = Path(value).expanduser() if value else config.default_work_dir()
    linked = raw.is_symlink()
    if linked:
        sys.stderr.write(f"{raw} is a symlink; its layout is not changed.\n")
    path = raw.resolve()
    fresh = not path.exists()
    path.mkdir(parents=True, exist_ok=True)
    info = path.stat()
    mode = info.st_mode
    if linked:
        return path
    if repair and jobs.run_as and info.st_uid == os.geteuid():
        group = _job_group(jobs)
        if group and info.st_gid != group[1]:
            # Root may; so may a member of the group.
            with contextlib.suppress(PermissionError):
                os.chown(path, -1, group[1])
        if fresh or stat.S_IMODE(mode) == 0o770:
            path.chmod(0o3770)
            return path
    if fresh:  # whatever the umask gave it
        path.chmod(0o700)
    elif mode & 0o007:
        path.chmod(mode & 0o7770)
    return path


def _token(argv_token: str | None) -> str:
    """The registration token: ``$HORUS_AGENT_TOKEN``, else stdin."""
    if argv_token:
        sys.stderr.write(
            f"--token is visible to other users; pass it on stdin or in "
            f"{TOKEN_ENV} instead.\n"
        )
        return argv_token
    if token := os.environ.get(TOKEN_ENV):
        return token
    if sys.stdin.isatty():
        return getpass.getpass("Registration token: ").strip()
    return sys.stdin.readline().strip()


def _register(args: argparse.Namespace) -> int:
    url: str = args.url
    check_broker_url(url)
    _umask(config.load_jobs_policy())
    token = _token(args.token)
    if not token:
        sys.stderr.write("No registration token given.\n")
        return 1
    ca_bundle = str(Path(args.ca_bundle).resolve()) if args.ca_bundle else ""
    jobs = config.load_jobs_policy()
    directory = _work_dir(args.work_dir, jobs)
    if problem := layout_problem(directory, jobs):
        sys.stderr.write(f"{problem}\n")  # run asks for the same fix
    name: str | None = args.name
    body = {
        "token": token,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "version": agent_version(),
        "work_dir": str(directory),
    }
    if name is not None:
        body["name"] = name
    response = httpx.post(
        f"{url.rstrip('/')}/agents/register",
        json=body,
        timeout=30.0,
        verify=broker_tls(ca_bundle),
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        sys.stderr.write(f"Registration failed: {detail}\n")
        return 1
    registration = Registration.model_validate(response.json())
    path = config.config_path()
    config.save(
        config.AgentConfig(
            url=url,
            agent_id=registration.agent_id,
            name=registration.name,
            token=registration.agent_token,
            work_dir=str(directory),
            ca_bundle=ca_bundle,
            job_env=tuple(args.job_env or ()),
        ),
        path,
    )
    sys.stdout.write(
        f"Registered as {registration.name!r} (config: {path}).\n"
        f"Tasks run in {directory}.\n"
        "Start it with: horus-agent run\n"
    )
    return 0


def _umask(jobs: config.JobsPolicy) -> None:
    """
    Keep new files from other users, sharing them with the task user's
    group under ``run_as``. Checks the ``run_as`` setup first.
    """
    if jobs.run_as:
        ops.switch_user(jobs.run_as, jobs.run_as_via)
    os.umask(ops.JOB_UMASK if jobs.run_as else 0o077)


async def _run(work_dir: str | None) -> int:
    path = config.config_path()
    try:
        registered = config.load(path)
    except FileNotFoundError:
        sys.stderr.write(
            f"No registration at {path}. Run horus-agent register first.\n"
        )
        return 1
    if path.stat().st_mode & 0o077:
        sys.stderr.write(
            f"{path} holds the agent's token and must be readable by its "
            f"owner only: chmod 600 {path}\n"
        )
        return 1
    try:
        check_broker_url(registered.url)
        jobs = config.load_jobs_policy()
        directory = _work_dir(
            work_dir or registered.work_dir or None, jobs, repair=False
        )
        if problem := layout_problem(directory, jobs):
            raise ValueError(problem)
        if not os.access(directory, os.W_OK):
            raise ValueError(
                f"The work dir {directory} is not writable (with the system "
                "unit it must be under ReadWritePaths)"
            )
        rules = policy.load_policy(str(directory), registered.url)
        if rules.source == "defaults":
            sys.stderr.write(
                "No policy file: only approved workflow releases run here. "
                f"{policy.OPEN_HINT}\n"
            )
        if problem := policy.same_user_problem(rules):
            raise ValueError(problem)
        _umask(rules.jobs)
        agent = Agent(
            dataclasses.replace(registered, work_dir=str(directory)), rules
        )
    except (ValueError, KeyError, OSError, policy.PolicyError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    kept = ops.job_environment(dict(os.environ), registered.job_env)
    if dropped := sorted(set(os.environ) - set(kept)):
        horus_logger.log.info(
            "Tasks do not get these variables of the agent's environment "
            f"(job_env in the config passes them on): {', '.join(dropped)}"
        )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _stop() -> None:
        # A second signal falls through to the default: exit now.
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)
    try:
        await agent.serve(stop)
    except AgentUnauthorizedError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


SERVICE_CONFIG = Path("/var/lib/horus-agent/.config/horus-agent/agent.json")
"""The agent's config under the system unit (README), used by sudo."""


def _service_config() -> Path:
    """The agent's config: ``HORUS_AGENT_CONFIG``, yours, or the unit's."""
    path = config.config_path()
    if (
        config.CONFIG_ENV not in os.environ
        and not path.exists()
        and SERVICE_CONFIG.exists()
    ):
        return SERVICE_CONFIG
    return path


MAX_DOWNLOAD_BYTES = 512 * 1024**2
"""Largest release archive ``approve`` reads, downloaded or from a file."""
MAX_WARNING_LINES = 40


def _archive(source: str) -> bytes:
    """A release archive: a local file, else downloaded by release id."""
    path = Path(source)
    if path.is_file():
        if path.stat().st_size > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"{source} is over {MAX_DOWNLOAD_BYTES} bytes")
        return path.read_bytes()
    where = _service_config()
    try:
        registered = config.load(where)
    except FileNotFoundError:
        raise ValueError(
            f"No agent config at {where} to download release {source} "
            f"with; set {config.CONFIG_ENV} to the agent's agent.json, or "
            "approve an exported archive file"
        ) from None
    with httpx.stream(
        "GET",
        f"{registered.url.rstrip('/')}/agents/me/releases/"
        f"{quote(source, safe='')}/archive",
        headers={"Authorization": f"Bearer {registered.token}"},
        timeout=120.0,
        verify=broker_tls(registered.ca_bundle),
    ) as response:
        if response.status_code >= 400:
            raise ValueError(
                f"Cannot download release {source}: HTTP "
                f"{response.status_code}"
            )
        body = bytearray()
        for chunk in response.iter_bytes():
            body += chunk
            if len(body) > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Release {source} is over {MAX_DOWNLOAD_BYTES} bytes"
                )
    return bytes(body)


def _safe(value: object) -> str:
    """*value* as text with every unprintable character escaped."""
    return "".join(
        ch if ch.isprintable() else f"\\u{ord(ch):04x}" for ch in str(value)
    )


def _show(manifest: Manifest) -> str:
    """
    What an approver reads before saying yes. Every field is printed with
    unprintable characters escaped, so nothing in it can hide the rest.
    """
    lines = [
        f"Release: {_safe(manifest.about.get('name'))} "
        f"(hash {manifest.release_hash})",
        f"{len(manifest.tasks)} agent tasks, "
        f"{len(manifest.warnings)} warnings",
    ]
    shown = manifest.warnings[:MAX_WARNING_LINES]
    lines += [f"Warning: {_safe(w)}" for w in shown]
    if len(manifest.warnings) > len(shown):
        lines.append(
            f"... and {len(manifest.warnings) - len(shown)} more warnings "
            "(each task below names its run-time inputs)"
        )
    for task_id, task in sorted(manifest.tasks.items()):
        runtime = task["runtime"]
        chain = " > ".join(task["target"]["chain"])
        lines.append(f"\nTask {_safe(task_id)} on {_safe(chain)}")
        if runtime["kind"] == "command":
            text = str(runtime["command"])
            lines.append("  command:")
        else:
            text = (
                f"{runtime['python']} -I <sha256 {runtime['script_sha256']}> "
                f"{runtime['args']}"
            )
            lines.append("  script:")
        lines += [f"    | {_safe(line)}" for line in text.split("\n")]
        if "slurm" in task["target"]:
            lines.append(f"  slurm: {json.dumps(task['target']['slurm'])}")
        for name, spec in sorted(task["inputs"].items()):
            detail = spec.get("value") or spec.get("sha256") or ""
            row = f"  input {_safe(name)}: {spec['role']} {_safe(detail)}"
            lines.append(row.rstrip())
        for name in sorted(task["env"]):
            lines.append(f"  env {_safe(name)}: {_safe(task['env'][name])}")
    lines.append(
        "\nNames cloud files may take at the top of a run: "
        + _safe(", ".join(manifest.layout) or "(none)")
    )
    return "\n".join(lines) + "\n"


def _python_problem(manifest: Manifest) -> str | None:
    """
    Why a release running ``python -c``/``-m`` cannot be approved here: this
    machine's Python is older than 3.11 and the policy lets code files into
    runs, so such a task would import them from its working directory.
    """
    tasks = [
        t
        for t, v in sorted(manifest.tasks.items())
        if task_runs_python_code(v)
    ]
    if not tasks or not policy.configured_code_suffixes():
        return None
    probe = subprocess.run(
        [
            "/usr/bin/env",
            "python3",
            "-I",
            "-c",
            "import sys; print(*sys.version_info[:2])",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    parts = probe.stdout.split()
    if (
        len(parts) == 2
        and all(p.isdigit() for p in parts)
        and (int(parts[0]), int(parts[1])) >= (3, 11)
    ):
        return None
    return (
        f"task {tasks[0]!r} runs python -c or -m, python3 here is older than "
        "3.11 (or missing), and allow_code_suffixes lets code files into "
        "runs, which such a Python would import from the task's working "
        "directory; use Python 3.11 or newer, or empty allow_code_suffixes"
    )


def _dry_run(directory: str) -> int:
    """Say which tasks under *directory* ``approve`` would accept."""
    folder = Path(directory)
    if not folder.is_dir():
        sys.stderr.write(f"{directory} is not a folder.\n")
        return 1
    result = dryrun.dry_run(folder)
    sys.stdout.write(result.summary())
    for where, message in result.refusals:
        sys.stdout.write(f"  {_safe(where)}: {_safe(message)}\n")
    return 0


def _approve(args: argparse.Namespace) -> int:  # noqa: PLR0911
    if args.dry_run:
        return _dry_run(args.release)
    if os.geteuid() not in config.POLICY_OWNERS:
        sys.stderr.write("Only root can approve releases (use sudo).\n")
        return 1
    try:
        manifest = build_manifest(_archive(args.release))
    except (ManifestError, ValueError, OSError, httpx.HTTPError) as exc:
        sys.stderr.write(f"Cannot approve: {exc}\n")
        return 1
    if not manifest.tasks:
        sys.stderr.write("This release runs nothing on agents.\n")
        return 1
    if problem := _python_problem(manifest):
        sys.stderr.write(f"Cannot approve: {problem}\n")
        return 1
    if args.expect_hash and args.expect_hash != manifest.release_hash:
        sys.stderr.write(
            f"Hash {manifest.release_hash} is not the expected "
            f"{args.expect_hash}.\n"
        )
        return 1
    sys.stdout.write(_show(manifest))
    if not args.yes:
        sys.stdout.write("Type yes to approve: ")
        sys.stdout.flush()
        if sys.stdin.readline().strip() != "yes":
            sys.stderr.write("Not approved.\n")
            return 1
    by = os.environ.get("SUDO_USER") or getpass.getuser()
    path = policy.approve(manifest, by)
    sys.stdout.write(f"Approved: {path}\n")
    return 0


def _approvals() -> int:
    directory = policy.approved_dir()
    rows = []
    for path in sorted(directory.glob("*.json")):
        try:
            manifest = Manifest.from_json(path.read_text())
        except (OSError, ValueError, KeyError, ManifestError):
            rows.append(f"{path.stem}  (unreadable)")
            continue
        about = manifest.about
        rows.append(
            f"{manifest.release_hash}  {about.get('name')}  "
            f"v{about.get('version', '-')}  {about.get('approved_at')}  "
            f"{about.get('approved_by')}"
        )
    sys.stdout.write("\n".join(rows or ["No approved releases."]) + "\n")
    return 0


def _revoke(release_hash: str) -> int:
    if os.geteuid() not in config.POLICY_OWNERS:
        sys.stderr.write("Only root can revoke releases (use sudo).\n")
        return 1
    if not policy.revoke(release_hash):
        sys.stderr.write(f"No approval {release_hash}.\n")
        return 1
    sys.stdout.write(
        f"Revoked {release_hash}. The agent kills its running jobs within "
        "about 20 s.\n"
    )
    return 0


def _policy_init(open_mode: bool, run_as: str) -> int:
    path = policy.policy_path()
    if os.geteuid() not in config.POLICY_OWNERS:
        sys.stderr.write(f"Only root can write {path} (use sudo).\n")
        return 1
    if path.exists():
        sys.stderr.write(f"{path} exists; edit it instead.\n")
        return 1
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fh:
        fh.write(policy.template(open_mode=open_mode, run_as=run_as))
    if group := _job_group(config.JobsPolicy(run_as=run_as)):
        sys.stdout.write(
            f"The work dir needs group {group[0]} and mode 3770 "
            "(horus-agent register sets them).\n"
        )
    sys.stdout.write(
        f"Wrote {path}. Add the storage host to allowed_hosts, then "
        "restart the agent.\n"
    )
    return 0


def _policy_check() -> int:
    try:
        registered = config.load(config.config_path())
        work_dir, url = registered.work_dir, registered.url
    except (FileNotFoundError, PermissionError):
        work_dir, url = str(config.default_work_dir()), ""
    try:
        rules = policy.load_policy(work_dir, url)
    except (policy.PolicyError, OSError) as exc:
        sys.stderr.write(f"Policy problem: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(rules.describe(), indent=2) + "\n")
    return 0


def _doctor_checks() -> list[tuple[str, str | None]]:
    """``(what was checked, problem or None)`` for this machine's layout."""
    found: list[tuple[str, str | None]] = []

    def check(what: str, problem: str | None) -> None:
        found.append((what, problem))

    try:
        registered = config.load(config.config_path())
        work_dir = Path(registered.work_dir)
    except (FileNotFoundError, PermissionError):
        work_dir = config.default_work_dir()
    path = policy.policy_path()
    try:
        for protected in (path, policy.approved_dir()):
            if protected.exists():
                policy.check_protected(protected)
        real = os.path.realpath(path)
        check(
            "policy file" + (f" (at {real})" if real != str(path) else ""),
            None,
        )
    except (policy.PolicyError, OSError) as exc:
        check("policy file", str(exc))
    try:
        jobs = config.load_jobs_policy()
    except ValueError as exc:
        check("jobs policy", str(exc))
        jobs = config.JobsPolicy()
    if jobs.run_as:
        try:
            pwd.getpwnam(jobs.run_as)
            check("run_as user", None)
        except KeyError:
            check("run_as user", f"user {jobs.run_as!r} does not exist")
    if work_dir.is_dir():
        info = work_dir.stat()
        what = (
            f"work dir {work_dir} mode {stat.S_IMODE(info.st_mode):o} "
            f"group {_group_name(info.st_gid)}"
        )
        problem = layout_problem(work_dir, jobs)
        if not problem and not os.access(work_dir, os.W_OK):
            problem = f"{work_dir} is not writable by the agent"
        check(what, problem)
    else:
        check(f"work dir {work_dir}", "missing; run horus-agent register")
    state = work_dir / guard_mod.STATE_DIR
    if state.is_dir():
        mode = stat.S_IMODE(state.stat().st_mode)
        check(
            f"state dir mode {mode:o}",
            None if not mode & 0o022 else f"chmod 711 {state}",
        )
    if jobs.run_as:
        check(*_switch_check(jobs))
    return found


def _group_name(gid: int) -> str:
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def _switch_check(jobs: config.JobsPolicy) -> tuple[str, str | None]:
    """Whether the agent may switch to ``run_as``, and the job's Python."""
    what = f"{jobs.run_as_via} rule for {jobs.run_as}"
    try:
        prefix = ops.switch_user(jobs.run_as, jobs.run_as_via)
        env = {"PATH": ops.job_environment(dict(os.environ)).get("PATH", "")}
        probe = (
            "for p in python python3; do command -v $p >/dev/null && "
            "$p -I -c 'import sys; print(*sys.version_info[:2])'; done"
        )
        done = subprocess.run(
            [*prefix, "/bin/sh", "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
            env=ops.SPAWN_ENV | env,
            check=False,
        )
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        return what, str(exc)
    if done.returncode not in (0, 1):  # 1: the last python is missing
        return what, done.stderr.strip() or f"exit {done.returncode}"
    versions = [ln.replace(" ", ".") for ln in done.stdout.split("\n") if ln]
    return (
        f"{what}; job python {' '.join(versions) or 'none on PATH'}",
        None if versions else "no python on the tasks' PATH",
    )


def _doctor() -> int:
    bad = 0
    for what, problem in _doctor_checks():
        bad += problem is not None
        sys.stdout.write(
            f"ok    {what}\n"
            if problem is None
            else f"FAIL  {what}: {problem}\n"
        )
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911
    """Entry point of the ``horus-agent`` command."""
    parser = argparse.ArgumentParser(
        prog="horus-agent",
        description="Run Horus tasks on this machine without opening "
        "inbound connections.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--debug",
        action="store_true",
        help="Log every connection, request, op and command (also: "
        f"{logs.DEBUG_ENV}=1). Secret values are never logged.",
    )
    common.add_argument(
        "--allow-root",
        action="store_true",
        help="Run as root anyway. Without run_as, tasks then run as root too.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser(
        "register",
        parents=[common],
        help="Register this machine with a one-time token.",
    )
    register.add_argument(
        "--url", required=True, help="Broker API URL (https)."
    )
    # Kept for old instructions; the token is read from stdin by default.
    register.add_argument("--token", help=argparse.SUPPRESS)
    register.add_argument(
        "--name",
        help="Agent name, used as the target's 'agent' field, when the "
        "token does not set one. Default: this machine's hostname.",
    )
    work_dir_help = (
        "Directory where tasks run on this machine, created if missing. "
        f"Default: {config.default_work_dir()}."
    )
    register.add_argument("--work-dir", help=work_dir_help)
    register.add_argument(
        "--ca-bundle",
        metavar="PEM",
        help="Trust only these CA certificates for the broker. Default: "
        "SSL_CERT_FILE or SSL_CERT_DIR when set, else the certifi bundle.",
    )
    register.add_argument(
        "--job-env",
        action="append",
        metavar="NAME",
        help="Pass this environment variable on to tasks (repeatable).",
    )
    run = commands.add_parser(
        "run", parents=[common], help="Poll for work until stopped."
    )
    run.add_argument(
        "--work-dir",
        help="Use this directory instead of the one given at registration.",
    )
    approve = commands.add_parser(
        "approve", help="Approve a workflow release on this machine (root)."
    )
    approve.add_argument(
        "release",
        help="Release id or archive file (with --dry-run: a folder).",
    )
    approve.add_argument(
        "--dry-run",
        action="store_true",
        help="Check every task of the workflows under a folder; approve none.",
    )
    approve.add_argument("--expect-hash", help="Refuse any other hash.")
    approve.add_argument(
        "--yes", action="store_true", help="Do not ask for confirmation."
    )
    commands.add_parser("approvals", help="List approved releases.")
    commands.add_parser("doctor", help="Check this machine's layout.")
    revoke = commands.add_parser("revoke", help="Withdraw an approval (root).")
    revoke.add_argument("hash", help="Release hash.")
    policy_cmd = commands.add_parser("policy", help="Show the policy.")
    policy_cmd.add_argument("action", choices=("check", "init"))
    policy_cmd.add_argument(
        "--open",
        action="store_true",
        help="init: accept any workflow (internal machines).",
    )
    policy_cmd.add_argument(
        "--run-as",
        default="horus-job",
        metavar="USER",
        help="init: the Unix user tasks run as. Default: horus-job.",
    )
    args = parser.parse_args(argv)
    logs.setup(getattr(args, "debug", False))
    match args.command:
        case "approve":
            return _approve(args)
        case "approvals":
            return _approvals()
        case "doctor":
            return _doctor()
        case "revoke":
            return _revoke(args.hash)
        case "policy" if args.action == "init":
            return _policy_init(args.open, args.run_as)
        case "policy":
            return _policy_check()
    if os.geteuid() == 0 and not args.allow_root:
        sys.stderr.write(
            "Refusing to run as root: run the agent as its own user (see "
            "Upgrading in the README), or pass --allow-root.\n"
        )
        return 1
    if args.command == "run":
        return asyncio.run(_run(args.work_dir))
    try:
        return _register(args)
    except (ValueError, KeyError, OSError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
