#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Logging setup and secret-free descriptions of ops for the logs.

Logs show commands and paths, but never env values (task secrets, also
where a command embeds them), tokens, passwords in URLs or ``key=value``
pairs, or presigned URL query strings (signatures).
"""

import base64
import binascii
import logging
import os
import re
from collections.abc import Iterable
from typing import Any

from horus_runtime.logging import horus_logger

DEBUG_ENV = "HORUS_AGENT_DEBUG"
"""Set to ``1`` to log at debug level, like ``--debug``."""

PREVIEW_CHARS = 300
"""How much of a command's output a debug line shows."""

_QUERY = re.compile(r"(https?://[^\s\"'?<>]*)\?[^\s\"'<>)]*")


_USERINFO = re.compile(r"://[^/\s@:]+:[^/\s@]+@")
_SECRET_NAME = (
    r"[\w.-]*(?:token|secret|passw(?:or)?d|api[_-]?key|access[_-]?key)[\w.-]*"
)
_CREDENTIAL = re.compile(
    # Bearer x, name=x, name: x, "name": "x", --name x
    rf"(?i)((?:\bbearer|\bbasic)\s+|\b{_SECRET_NAME}[\"']?\s*[=:]\s*"
    rf"|(?<![\w-])--{_SECRET_NAME}\s+)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s;&|'\"]+)"
)
MIN_SECRET_CHARS = 6
"""Shorter env values are not hidden in text: too likely to be common."""


def safe_url(url: object) -> str:
    """*url* (or text holding URLs) with URL query strings redacted."""
    return _QUERY.sub(r"\1?<redacted>", str(url))


def redact(text: object, secrets: Iterable[str] = ()) -> str:
    """
    *text* with every value in *secrets* (an op's env values), URL query
    strings and passwords, and credential-looking assignments hidden.
    """
    out = str(text)
    for value in sorted(secrets, key=len, reverse=True):
        if len(value) >= MIN_SECRET_CHARS:
            out = out.replace(value, "<redacted>")
    out = _USERINFO.sub("://<redacted>@", out)
    out = _CREDENTIAL.sub(r"\1<redacted>", out)
    return safe_url(out)


def _env_values(args: dict[str, Any]) -> list[str]:
    env = args.get("env")
    return [str(v) for v in env.values()] if isinstance(env, dict) else []


class _ToLoguru(logging.Handler):
    """Forward httpx/httpcore records (connections, TLS, requests)."""

    def emit(self, record: logging.LogRecord) -> None:
        horus_logger.log.log(
            record.levelname, f"{record.name}: {redact(record.getMessage())}"
        )


def setup(debug: bool = False) -> bool:
    """
    Log at debug level when *debug* or :data:`DEBUG_ENV` is set, with
    httpx's connection and request records included. Returns whether debug
    logging is on.
    """
    debug = debug or os.environ.get(DEBUG_ENV, "") not in ("", "0")
    horus_logger.setup(level="DEBUG" if debug else "INFO")
    if debug:
        handler = _ToLoguru()
        for name in ("httpx", "httpcore"):
            log = logging.getLogger(name)
            log.handlers = [handler]
            log.setLevel(logging.DEBUG)
            log.propagate = False
    return debug


def _b64_size(value: str) -> int:
    return len(value) * 3 // 4 - value.count("=", -2)


def _preview(value: str, secrets: Iterable[str]) -> str:
    try:
        text = base64.b64decode(value).decode(errors="replace")
    except (binascii.Error, ValueError):
        return f"<{_b64_size(value)} bytes>"
    # Redact before cutting, or a secret across the cut would show its start.
    size = len(text)
    text = redact(text, secrets)
    if len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + f"… ({size} chars)"
    return repr(text)


QUIET_METHODS = frozenset({"poll", "read_output"})
"""Ops repeated every few seconds for each running job: debug level only."""

SUMMARY_CHARS = 200


def summarize(method: str, args: dict[str, Any]) -> str | None:
    """
    One line naming what an op does, for the info log: the command, or the
    path it touches. None for the routine job checks in :data:`QUIET_METHODS`.
    """
    if method in QUIET_METHODS:
        return None
    if "cmd" in args:
        cmd = " ".join(redact(args["cmd"], _env_values(args)).split())
        if len(cmd) > SUMMARY_CHARS:
            cmd = cmd[:SUMMARY_CHARS] + "…"
        detail = cmd
    elif "path" in args:
        detail = redact(args["path"])
        if isinstance(args.get("data"), str):
            detail += f" ({_b64_size(args['data'])} bytes)"
    elif "pid" in args:
        detail = f"pid {args['pid']} signal {args.get('sig')}"
    else:
        detail = ""
    return f"{method} {detail}".rstrip()


def describe_args(args: dict[str, Any]) -> str:
    """An op's arguments, with env values, signatures and payloads hidden."""
    parts = []
    secrets = _env_values(args)
    for key, value in args.items():
        if key == "env":
            # Names only: the values are the task's secrets.
            parts.append(f"env=[{', '.join(sorted(value or {}))}]")
        elif key == "data":
            parts.append(f"data=<{_b64_size(value)} bytes>")
        else:
            # Commands can hold presigned URLs (curl transfers).
            shown = (
                repr(redact(value, secrets))
                if isinstance(value, str)
                else redact(repr(value), secrets)
            )
            parts.append(f"{key}={shown}")
    return " ".join(parts)


def describe_result(
    method: str, value: Any, secrets: Iterable[str] = ()
) -> str:
    """
    An op's answer: command output previewed (with *secrets*, the op's env
    values, hidden), other payloads sized.
    """
    if isinstance(value, list):
        return f"{len(value)} entries"
    if not isinstance(value, dict):
        return repr(value)
    parts = []
    for key, item in value.items():
        if key in ("stdout", "stderr", "data") and isinstance(item, str):
            shown = (
                _preview(item, secrets)
                if method == "exec"
                else f"<{_b64_size(item)} bytes>"
            )
            parts.append(f"{key}={shown}")
        else:
            parts.append(f"{key}={item!r}")
    return " ".join(parts)
