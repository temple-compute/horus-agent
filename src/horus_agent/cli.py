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
import os
import platform
import signal
import socket
import sys
from pathlib import Path

import httpx
from horus_runtime.logging import horus_logger

from horus_agent import config, logs, ops
from horus_agent.daemon import (
    Agent,
    AgentUnauthorizedError,
    agent_version,
    broker_tls,
    check_broker_url,
)
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
        jobs = config.load_jobs_policy()
        _umask(jobs)
        directory = _work_dir(work_dir or registered.work_dir or None)
        if not os.access(directory, os.W_OK):
            raise ValueError(
                f"The work dir {directory} is not writable (with the system "
                "unit it must be under ReadWritePaths)"
            )
        agent = Agent(
            dataclasses.replace(registered, work_dir=str(directory)), jobs
        )
    except (ValueError, KeyError, OSError) as exc:
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


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)
    logs.setup(args.debug)
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
