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
_SHELLS = frozenset(
    {"sh", "bash", "dash", "zsh", "ksh", "ash", "busybox", "source", "."}
)
_INTERPRETERS = frozenset(
    {
        "perl",
        "ruby",
        "node",
        "nodejs",
        "Rscript",
        "R",
        "php",
        "lua",
        "julia",
        "tclsh",
        "awk",
        "gawk",
        "mawk",
        "nawk",
        "sed",
        "jq",
        "nextflow",
        "snakemake",
        "make",
    }
)
_PYTHON = re.compile(r"python[0-9.]*|pypy[0-9.]*")
_SLURM_MPI = (
    "-n -N -c -p -t -J -np -H --np --ntasks --nodes --cpus-per-task "
    "--partition --time --job-name --host --hostfile -machinefile --map-by "
    "--bind-to"
)
_WRAPPERS: dict[str, tuple[str, str, str, int]] = {
    # name: (options taking an argument, flags, options running code,
    #        positional arguments before the command)
    "exec": ("-a", "-c -l", "", 0),
    "command": ("", "-p", "", 0),
    "builtin": ("", "", "", 0),
    "nohup": ("", "", "", 0),
    "nice": ("-n --adjustment", "", "", 0),
    "timeout": (
        "-s --signal -k --kill-after",
        "--preserve-status --foreground -v --verbose",
        "",
        1,
    ),
    "time": (
        "-o -f --output --format",
        "-p -v --verbose --portability",
        "",
        0,
    ),
    "stdbuf": ("-i -o -e", "", "", 0),
    "taskset": ("", "-c --cpu-list -a --all-tasks", "", 1),
    "numactl": (
        "-N -m -C -p -i --cpunodebind --membind --physcpubind --preferred "
        "--interleave",
        "-l --localalloc",
        "",
        0,
    ),
    "srun": (_SLURM_MPI, "-l --label --exclusive", "", 0),
    "mpirun": (_SLURM_MPI, "--oversubscribe", "", 0),
    "mpiexec": (_SLURM_MPI, "--oversubscribe", "", 0),
    "env": (
        "-u --unset -C --chdir",
        "-i --ignore-environment -0 --null -",
        "-S --split-string",
        0,
    ),
    "sudo": (
        "-u -g -C -D -h -p -r -t -U --user --group",
        "-n -E -H -P -b",
        "-s -i --shell --login",
        0,
    ),
    "doas": ("-u -C", "-n", "-s", 0),
    "chroot": ("--userspec --groups", "", "", 1),
    "flock": (
        "-w --timeout -E --conflict-exit-code",
        "-s -x -n -u -o --shared --exclusive --nonblock --unlock --close",
        "-c --command",
        1,
    ),
    "setsid": ("", "-f -w -c --fork --wait --ctty", "", 0),
}
"""Commands that run their arguments as another command."""
_REPARSE = frozenset(
    {"eval", "ssh", "su", "xargs", "parallel", "watch", "let", "trap"}
)
"""Commands that parse their arguments as shell code again."""
_ARITH_DECL = frozenset({"declare", "typeset", "local", "readonly", "export"})
_SUBSCRIPT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\[[^\]]*$")
_HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z0-9_]+)\2")
_PIPE_TO_CODE = re.compile(
    r"\|\s*(?:\S*/)?(?:sh|bash|dash|zsh|ksh|ash|busybox|source|\.|"
    r"python[0-9.]*|perl|ruby|node|Rscript|R|php|lua)(?:\s+-\S*)*\s*"
    r"(?:$|[;&|)\n<>])"
)
CODE_POSITION = "where the command runs it as code"
"""The reason given for a placeholder naming the program or script to run."""


def unsafe_placeholders(text: str, names: set[str]) -> dict[str, str]:
    """
    Find the placeholders in *text* where a value is not one inert word.

    The placeholders among *names* that *text* uses so, name -> why:

    - inside quotes, a heredoc, a ``${...}`` expansion or arithmetic
      (``$((``, ``((``, ``$[``, ``[[``, ``let``, ``declare -i``, an array
      subscript);
    - in arguments a command parses as code again (``eval``, ``ssh``,
      ``xargs``...), a here-string, or a pipeline into a shell or an
      interpreter;
    - :data:`CODE_POSITION`: as the program to run or the script or code
      handed to a shell or an interpreter.

    A template whose quoting cannot be followed, or with a ``<<-`` heredoc,
    marks every name. Bare words (``${x}``, ``--in=${x}``, ``$(cmd ${x})``)
    are safe: the agent renders each value as one quoted word.
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
    if not found:
        return {}
    if "<<-" in skeleton:
        return dict.fromkeys(found, "in a template with a <<- heredoc")
    if _PIPE_TO_CODE.search(skeleton):
        return dict.fromkeys(found, "in a template that pipes into a shell")
    reasons: list[str | None] = _scan(skeleton)
    if len(reasons) != len(found):
        return dict.fromkeys(found, "its quoting cannot be followed")
    out: dict[str, str] = {}
    for name, reason in zip(found, reasons, strict=True):
        if reason is not None:
            out.setdefault(name, reason)
    return out


def _base(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _unwrap(words: list[str]) -> list[str] | str:
    """
    *words* without the wrappers in front of the program (``timeout -s
    KILL 60``, ``env -u X``...), or why the program cannot be told: an
    option the agent does not know, or one that runs code.
    """
    rest = list(words)
    while rest and _base(rest[0]) in _WRAPPERS:
        takes, flags, code, positional = _WRAPPERS[_base(rest.pop(0))]
        while rest and rest[0].startswith("-"):
            option = rest.pop(0)
            name = option.split("=", 1)[0]
            if name in code.split():
                return "in a command that runs it as code"
            if name in takes.split() and "=" not in option:
                rest = rest[1:]
            elif not (
                name in takes.split()
                or name in flags.split()
                or re.fullmatch(r"-\d+", option)  # nice -10
            ):
                return (
                    f"after a wrapper option the agent cannot read: {option}"
                )
        rest = rest[positional:]
    return rest


def _code_reason(  # noqa: PLR0911
    words: list[str], word: str
) -> str | None:
    """
    Why a placeholder after *words* (this command's words so far) and
    *word* (the start of its own word) runs as code, if it does.
    """
    if _SUBSCRIPT.search(word):
        return "inside an array subscript"
    bases = [_base(w) for w in words]
    if any(b in _REPARSE for b in bases):
        return "in a command that runs it as code"
    if "[[" in words and "]]" not in words[words.index("[[") :]:
        return "inside [[ ]]"
    if any(b in _ARITH_DECL for b in bases) and any(
        w.startswith("-") for w in words
    ):
        return "in a typed declaration"
    if "printf" in bases and "-v" in words:
        return "in printf -v"
    # Skip assignments and wrappers to find the program.
    unwrapped = _unwrap(
        [w for w in words if not re.fullmatch(r"[A-Za-z_]\w*=.*", w)]
    )
    if isinstance(unwrapped, str):
        return unwrapped
    rest = unwrapped
    if not rest:
        return CODE_POSITION if not word else None
    program = _base(rest[0])
    if (
        program in _SHELLS
        or program in _INTERPRETERS
        or _PYTHON.fullmatch(program)
    ):
        args = rest[1:]
        if any(
            w in ("--eval", "--expr", "--command")
            or (
                w.startswith("-")
                and not w.startswith("--")
                and set(w) & set("ceEpr")
            )
            for w in args
        ):
            return CODE_POSITION
        if all(w.startswith("-") for w in args) and not word:
            return CODE_POSITION
    return None


def _scan(s: str) -> list[str | None]:  # noqa: PLR0912, PLR0915
    """
    Walk shell text where each placeholder is a NUL and say, for each in
    order, why its spot is unsafe (``None``: a bare word). An unbalanced
    quote or substitution marks every placeholder unsafe.
    """
    out: list[str | None] = []
    # Each frame: (kind, words of the current command in it, runs as code)
    stack: list[tuple[str, list[str], bool]] = [("top", [], False)]
    word = ""
    heredocs: list[str] = []
    herestring = False
    closers = {'"': '"', "${": "}", "$((": "))", "((": "))", "$[": "]"}
    inside = {
        '"': "inside double quotes",
        "${": "inside a ${...} expansion",
        "$((": "inside arithmetic",
        "((": "inside arithmetic",
        "$[": "inside arithmetic",
    }
    i = 0
    n = len(s)

    def end_word() -> None:
        nonlocal word, herestring
        if word:
            stack[-1][1].append(word)
            herestring = False
        word = ""

    def push(kind: str) -> None:
        _kind, words, code = stack[-1]
        stack.append((kind, [], code or _code_reason(words, word) is not None))

    while i < n:
        c = s[i]
        kind, words, code = stack[-1]
        if kind == "'":
            if c == "'":
                stack.pop()
            elif c == _SENTINEL:
                out.append("inside single quotes")
            i += 1
            continue
        if kind in closers:
            if c == _SENTINEL:
                out.append(inside[kind])
            elif c == "\\":
                i += 2
                continue
            elif s.startswith(closers[kind], i):
                stack.pop()
                i += len(closers[kind])
                continue
            elif s.startswith("$((", i):
                push("$((")
                i += 3
                continue
            elif s.startswith("$(", i):
                push("$(")
                i += 2
                continue
            elif s.startswith("${", i):
                push("${")
                i += 2
                continue
            elif s.startswith("$[", i):
                push("$[")
                i += 2
                continue
            elif c == "`":
                push("`")
            i += 1
            continue
        # Unquoted: the top level, $(...), `...` or <(...).
        if c == _SENTINEL:
            if herestring:
                out.append("in a here-string")
            elif code:
                out.append("in output a command runs as code")
            else:
                out.append(_code_reason(words, word))
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
            stack.append((c, [], code))
            word += "q"
            i += 1
            continue
        for opener in ("$((", "$(", "${", "$[", "<(", ">("):
            if s.startswith(opener, i):
                push("$(" if opener in ("<(", ">(") else opener)
                i += len(opener)
                break
        else:
            opener = ""
        if opener:
            continue
        if s.startswith("((", i) and not word:
            push("((")
            i += 2
            continue
        if c == "`" and kind != "`":
            push("`")
            i += 1
            continue
        if (c == ")" and kind == "$(") or (c == "`" and kind == "`"):
            end_word()
            stack.pop()
            word = "x"
            i += 1
            continue
        if s.startswith("<<<", i):
            end_word()
            herestring = True
            i += 3
            continue
        if heredoc := _HEREDOC.match(s, i):
            end_word()
            heredocs.append(heredoc.group(3))
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
                    if line == delim:
                        break
                else:
                    return []
            heredocs.clear()
            continue
        if c in " \t":
            end_word()
        elif c in ";&|\n(){}":
            end_word()
            herestring = False
            words.clear()
        else:
            word += c
        i += 1
    if len(stack) != 1 or heredocs:
        return []
    return out
