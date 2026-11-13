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
import dataclasses
import getpass
import json
import os
import platform
import signal
import socket
import sys
from pathlib import Path
from urllib.parse import quote

import httpx
from horus_runtime.logging import horus_logger

from horus_agent import config, logs, ops, policy
from horus_agent.daemon import (
    Agent,
    AgentUnauthorizedError,
    agent_version,
    broker_tls,
    check_broker_url,
)
from horus_agent.manifest import Manifest, ManifestError, build_manifest
from horus_agent.protocol import Registration

TOKEN_ENV = "HORUS_AGENT_TOKEN"
"""Registration token, when not typed or piped in."""


def _work_dir(value: str | None) -> Path:
    """
    *value* as an absolute path, created if missing, with no access for
    other users.
    """
    path = Path(value).expanduser() if value else config.default_work_dir()
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if (mode := path.stat().st_mode) & 0o007:
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
    directory = _work_dir(args.work_dir)
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
        directory = _work_dir(work_dir or registered.work_dir or None)
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


def _archive(source: str) -> bytes:
    """A release archive: a local file, else downloaded by release id."""
    path = Path(source)
    if path.is_file():
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
    response = httpx.get(
        f"{registered.url.rstrip('/')}/agents/me/releases/"
        f"{quote(source, safe='')}/archive",
        headers={"Authorization": f"Bearer {registered.token}"},
        timeout=120.0,
        verify=broker_tls(registered.ca_bundle),
    )
    if response.status_code >= 400:
        raise ValueError(
            f"Cannot download release {source}: HTTP {response.status_code}"
        )
    return response.content


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
    ]
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
    lines += [f"\nWarning: {_safe(w)}" for w in manifest.warnings]
    return "\n".join(lines) + "\n"


def _approve(args: argparse.Namespace) -> int:
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
    sys.stdout.write(f"Revoked {release_hash}.\n")
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
    approve.add_argument("release", help="Release id or archive file.")
    approve.add_argument("--expect-hash", help="Refuse any other hash.")
    approve.add_argument(
        "--yes", action="store_true", help="Do not ask for confirmation."
    )
    commands.add_parser("approvals", help="List approved releases.")
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
