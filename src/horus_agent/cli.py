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
import platform
import signal
import socket
import sys
from pathlib import Path

import httpx

from horus_agent import config, logs
from horus_agent.daemon import Agent, AgentUnauthorizedError, agent_version
from horus_agent.protocol import Registration


def _work_dir(value: str | None) -> Path:
    """*value* as an absolute path, created if missing."""
    path = Path(value).expanduser() if value else config.default_work_dir()
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _register(
    url: str, token: str, name: str | None, work_dir: str | None
) -> int:
    directory = _work_dir(work_dir)
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
        f"{url.rstrip('/')}/agents/register", json=body, timeout=30.0
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
        ),
        path,
    )
    sys.stdout.write(
        f"Registered as {registration.name!r} (config: {path}).\n"
        f"Tasks run in {directory}.\n"
        "Start it with: horus-agent run\n"
    )
    return 0


async def _run(work_dir: str | None) -> int:
    path = config.config_path()
    try:
        registered = config.load(path)
    except FileNotFoundError:
        sys.stderr.write(
            f"No registration at {path}. Run horus-agent register first.\n"
        )
        return 1
    directory = _work_dir(work_dir or registered.work_dir or None)
    agent = Agent(dataclasses.replace(registered, work_dir=str(directory)))
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
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser(
        "register",
        parents=[common],
        help="Register this machine with a one-time token.",
    )
    register.add_argument("--url", required=True, help="Broker API URL.")
    register.add_argument(
        "--token", required=True, help="One-time registration token."
    )
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
    run = commands.add_parser(
        "run", parents=[common], help="Poll for work until stopped."
    )
    run.add_argument(
        "--work-dir",
        help="Use this directory instead of the one given at registration.",
    )
    args = parser.parse_args(argv)
    logs.setup(args.debug)
    if args.command == "register":
        return _register(args.url, args.token, args.name, args.work_dir)
    return asyncio.run(_run(args.work_dir))
