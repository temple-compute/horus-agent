#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
The ``$id`` / ``${id}`` placeholder syntax of Horus command templates.

Vendored from the runtime's ``substitution`` module so the manifest builder
and the agent read a template exactly as the runtime renders it, whichever
runtime version is installed next to them.
"""

import shlex
from collections.abc import Mapping
from string import Template


class DotTemplate(Template):
    """``string.Template`` that also matches dotted names (``${a.path}``)."""

    braceidpattern = r"(?a:[_a-z][_a-z0-9]*(?:\.[_a-z0-9]+)*)"


def placeholders(text: str) -> set[str]:
    """Every placeholder name in *text*: ``{"in", "task.id"}``."""
    names: set[str] = set()
    for match in DotTemplate.pattern.finditer(text):
        name = match.group("named") or match.group("braced")
        if name:
            names.add(name)
    return names


def quote(value: str | list[str]) -> str:
    """
    *value* as shell words: one quoted word, or a list's quoted words joined
    by spaces, as the runtime renders a list value.
    """
    if isinstance(value, list):
        return " ".join(shlex.quote(item) for item in value)
    return shlex.quote(value)


def render(text: str, values: Mapping[str, str | list[str]]) -> str:
    """
    *text* with every placeholder in *values* replaced by its quoted value;
    other placeholders stay as written, ``$$`` becomes ``$``.
    """
    quoted = {name: quote(value) for name, value in values.items()}
    return DotTemplate(text).safe_substitute(quoted)
