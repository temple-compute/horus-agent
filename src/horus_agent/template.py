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

import re
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


_SENTINEL = "\0"
_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh", "ash", "busybox"})
_REPARSE = frozenset({"eval", "ssh", "su", "xargs", "parallel", "watch"})
"""Commands that parse their arguments as shell code again."""
_HEREDOC = re.compile(r"<<-?[ \t]*(['\"]?)([A-Za-z0-9_]+)\1")


def unsafe_placeholders(text: str, names: set[str]) -> dict[str, str]:
    """
    The placeholders among *names* that *text* uses somewhere a quoted
    value is not one inert shell word: inside quotes, a heredoc, a
    ``${...}`` expansion or arithmetic, or the arguments of a command that
    parses them as code again (``eval``, ``sh -c``, ``ssh``...). Name ->
    why. A template whose quoting cannot be followed marks every name.

    Bare words (``${x}``, ``--in=${x}``, ``$(cmd ${x})``) are safe: the
    agent renders each value as one quoted word.
    """
    found: list[str] = []
    skeleton = ""
    last = 0
    for match in DotTemplate.pattern.finditer(text):
        skeleton += text[last : match.start()]
        last = match.end()
        name = match.group("named") or match.group("braced")
        if match.group("escaped") is not None:
            skeleton += "$"
        elif name in names:
            found.append(name)
            skeleton += _SENTINEL
        else:
            skeleton += match.group(0)
    skeleton += text[last:]
    reasons: list[str | None] = _scan(skeleton)
    out: dict[str, str] = {}
    if len(reasons) != len(found):
        return dict.fromkeys(found, "its quoting cannot be followed")
    for name, reason in zip(found, reasons, strict=True):
        if reason is not None:
            out.setdefault(name, reason)
    return out


def _scan(s: str) -> list[str | None]:  # noqa: PLR0912, PLR0915
    """
    Walk shell text where each placeholder is a NUL and say, for each in
    order, why its spot is unsafe (``None``: a bare word). An unbalanced
    quote or substitution marks every placeholder unsafe.
    """
    out: list[str | None] = []
    # Each frame: (kind, words of the current command in it)
    stack: list[tuple[str, list[str]]] = [("top", [])]
    word = ""
    heredocs: list[str] = []
    i = 0
    n = len(s)

    def end_word() -> None:
        nonlocal word
        if word:
            stack[-1][1].append(word)
        word = ""

    def reparsed(words: list[str]) -> bool:
        names = [w.rsplit("/", 1)[-1] for w in words]
        if any(w in _REPARSE for w in names):
            return True
        return any(w in _SHELLS for w in names) and any(
            w.startswith("-") and "c" in w for w in words
        )

    while i < n:
        c = s[i]
        kind, words = stack[-1]
        if kind == "'":
            if c == "'":
                stack.pop()
            elif c == _SENTINEL:
                out.append("inside single quotes")
            i += 1
            continue
        if kind in ('"', "${", "$(("):
            close = {'"': '"', "${": "}", "$((": "))"}[kind]
            if c == _SENTINEL:
                out.append(
                    {
                        '"': "inside double quotes",
                        "${": "inside a ${...} expansion",
                        "$((": "inside arithmetic",
                    }[kind]
                )
            elif c == "\\":
                i += 2
                continue
            elif s.startswith(close, i):
                stack.pop()
                i += len(close)
                continue
            elif s.startswith("$((", i):
                stack.append(("$((", []))
                i += 3
                continue
            elif s.startswith("$(", i):
                stack.append(("$(", []))
                i += 2
                continue
            elif s.startswith("${", i):
                stack.append(("${", []))
                i += 2
                continue
            elif c == "`":
                stack.append(("`", []))
            i += 1
            continue
        # Unquoted: the top level, $(...) or `...`.
        if c == _SENTINEL:
            out.append(
                "in a command that runs it as code"
                if reparsed([*words, word])
                else None
            )
            word += "x"
            i += 1
            continue
        if c == "\\":
            word += s[i : i + 2]
            i += 2
            continue
        if c == "#" and not word:
            while i < n and s[i] != "\n":
                if s[i] == _SENTINEL:
                    out.append(None)  # a comment
                i += 1
            continue
        if c in "'\"":
            stack.append((c, []))
            word += "q"
            i += 1
            continue
        if s.startswith("$((", i):
            stack.append(("$((", []))
            i += 3
            continue
        if s.startswith("$(", i) or (c == "`" and kind != "`"):
            stack.append(("$(" if c == "$" else "`", []))
            i += 2 if c == "$" else 1
            continue
        if s.startswith("${", i):
            stack.append(("${", []))
            i += 2
            continue
        if (c == ")" and kind == "$(") or (c == "`" and kind == "`"):
            end_word()
            stack.pop()
            word = "x"
            i += 1
            continue
        if s.startswith("<<<", i):
            end_word()
            i += 3
            continue
        if heredoc := _HEREDOC.match(s, i):
            end_word()
            heredocs.append(heredoc.group(2))
            i = heredoc.end()
            continue
        if c == "\n" and heredocs:
            end_word()
            words.clear()
            i += 1
            for delim in heredocs:
                while i < n:
                    eol = s.find("\n", i)
                    line = s[i : n if eol < 0 else eol]
                    out.extend(
                        "inside a heredoc" for _ in line.split(_SENTINEL)[1:]
                    )
                    i = n if eol < 0 else eol + 1
                    if line.lstrip("\t") == delim:
                        break
                else:
                    return []
            heredocs.clear()
            continue
        if c in " \t":
            end_word()
        elif c in ";&|\n(){}":
            end_word()
            words.clear()
        else:
            word += c
        i += 1
    if len(stack) != 1 or heredocs:
        return []
    return out
