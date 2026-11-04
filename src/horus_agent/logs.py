#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
Logging setup and secret-free descriptions of ops for the logs.

Debug logs show every connection, request, op and command, but never env
values (task secrets), tokens, or presigned URL query strings (signatures).
"""

import base64
import binascii
import logging
import os
import re
from typing import Any

from horus_runtime.logging import horus_logger

DEBUG_ENV = "HORUS_AGENT_DEBUG"
"""Set to ``1`` to log at debug level, like ``--debug``."""

PREVIEW_CHARS = 300
"""How much of a command's output a debug line shows."""

_QUERY = re.compile(r"(https?://[^\s\"'?<>]*)\?[^\s\"'<>)]*")


def safe_url(url: object) -> str:
    """*url* (or text holding URLs) with URL query strings redacted."""
    return _QUERY.sub(r"\1?<redacted>", str(url))


class _ToLoguru(logging.Handler):
    """Forward httpx/httpcore records (connections, TLS, requests)."""

    def emit(self, record: logging.LogRecord) -> None:
        horus_logger.log.log(
            record.levelname, f"{record.name}: {safe_url(record.getMessage())}"
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


def _preview(value: str) -> str:
    try:
        text = base64.b64decode(value).decode(errors="replace")
    except (binascii.Error, ValueError):
        return f"<{_b64_size(value)} bytes>"
    if len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + f"… ({len(text)} chars)"
    return safe_url(repr(text))


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
        cmd = " ".join(str(args["cmd"]).split())
        if len(cmd) > SUMMARY_CHARS:
            cmd = cmd[:SUMMARY_CHARS] + "…"
        detail = safe_url(cmd)
    elif "path" in args:
        detail = str(args["path"])
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
    for key, value in args.items():
        if key == "env":
            # Names only: the values are the task's secrets.
            parts.append(f"env=[{', '.join(sorted(value or {}))}]")
        elif key == "data":
            parts.append(f"data=<{_b64_size(value)} bytes>")
        else:
            # Commands can hold presigned URLs (curl transfers).
            parts.append(f"{key}={safe_url(repr(value))}")
    return " ".join(parts)


def describe_result(method: str, value: Any) -> str:
    """An op's answer: command output previewed, other payloads sized."""
    if isinstance(value, list):
        return f"{len(value)} entries"
    if not isinstance(value, dict):
        return repr(value)
    parts = []
    for key, item in value.items():
        if key in ("stdout", "stderr", "data") and isinstance(item, str):
            shown = (
                _preview(item)
                if method == "exec"
                else f"<{_b64_size(item)} bytes>"
            )
            parts.append(f"{key}={shown}")
        else:
            parts.append(f"{key}={item!r}")
    return " ".join(parts)
