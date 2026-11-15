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
_SHELLS = "sh bash dash zsh ksh ash tcsh csh fish busybox source ."
_PROGRAMS: dict[str, tuple[str, str, str, str, str]] = {
    # program: (options taking a value, flags, options whose value is code,
    #           options making the first positional the code, positionals:
    #           first = the first is the script; data = none is code;
    #           all = every one is code; run = the one after "run")
    **dict.fromkeys(
        _SHELLS.split(),
        (
            "-o +o -O +O --rcfile --init-file",
            "-e -u -x -v -l -i -s -n -f -a -h -m -B -C -E -H -P -T -p "
            "--login --norc --noprofile --posix --restricted --verbose",
            "",
            "-c",
            "first",
        ),
    ),
    "python": (
        "-W -X --check-hash-based-pycs",
        "-u -B -O -OO -E -s -S -I -P -q -v -b -bb -d -x -R",
        "-c -m",
        "",
        "first",
    ),
    "torchrun": (
        "--nproc_per_node --nproc-per-node --nnodes --node_rank --node-rank "
        "--master_addr --master-addr --master_port --master-port "
        "--rdzv_backend --rdzv-backend --rdzv_endpoint --rdzv-endpoint "
        "--rdzv_id --rdzv-id --max_restarts --max-restarts",
        "--standalone --no_python --no-python",
        "-m --module",
        "",
        "first",
    ),
    "perl": (
        "-I -M -m -x -l -0 -C -i",
        "-w -W -X -T -t -s -n -p -a -c",
        "-e -E",
        "",
        "first",
    ),
    "ruby": (
        "-I -r -C -F -E",
        "-w -W -n -p -a -l -s -v -d",
        "-e",
        "",
        "first",
    ),
    "node": (
        "-r --require --import --loader --input-type --conditions",
        "--no-warnings --enable-source-maps --trace-warnings",
        "-e --eval -p --print",
        "",
        "first",
    ),
    "Rscript": (
        "--default-packages",
        "--vanilla --no-save --no-restore --no-environ --no-site-file "
        "--no-init-file --verbose --slave --quiet --no-echo",
        "-e",
        "",
        "first",
    ),
    "R": (
        "--args",
        "--vanilla --no-save --no-restore --save --quiet --slave --no-echo "
        "--no-environ --no-site-file --no-init-file --interactive",
        "-e -f --file",
        "",
        "data",
    ),
    "php": ("-c -d -z", "-n -q -H -C", "-r -f -B -R -E -F", "", "first"),
    "lua": ("-l", "-i -v -E -W", "-e", "", "first"),
    "julia": (
        "-p -t -O -J --project --sysimage --threads --procs --optimize",
        "-q --quiet --startup-file=no --color=yes -i",
        "-e -E -L --eval --print --load",
        "",
        "first",
    ),
    "tclsh": ("", "", "", "", "first"),
    **dict.fromkeys(
        ("awk", "gawk", "mawk", "nawk"),
        (
            "-F -v --field-separator --assign",
            "--posix --re-interval",
            "-f -e --file --source",
            "",
            "first",
        ),
    ),
    "sed": (
        "-l --line-length",
        "-n -r -E -s -u -z --quiet --silent --regexp-extended --separate",
        "-e -f --expression --file",
        "",
        "first",
    ),
    "make": (
        "-C -j -l -o -W -I --directory --jobs --load-average --old-file "
        "--what-if --include-dir",
        "-k -s -n -B -i -q -r -R --keep-going --silent --dry-run",
        "-f --file --makefile",
        "",
        "all",
    ),
    "snakemake": (
        "-c -j --cores --jobs --configfile --configfiles --config -d "
        "--directory --profile --rerun-triggers --resources",
        "-n -p -F -k --use-conda --use-singularity --use-apptainer "
        "--keep-going --dry-run --printshellcmds --forceall",
        "-s --snakefile",
        "",
        "data",
    ),
    "nextflow": (
        "-profile -params-file -w -work-dir -name -revision -r -entry "
        "-with-report -with-trace -with-timeline -with-dag -log -bucket-dir",
        "-resume -bg -q -quiet -with-docker -with-singularity -with-apptainer "
        "-with-conda",
        "-c -config -C",
        "",
        "run",
    ),
}
"""Interpreters and tools whose arguments can be code, and how to read them."""
_PYTHON = re.compile(r"python[0-9.]*|pypy[0-9.]*")
_SLURM_MPI = (
    "-n -N -c -p -t -J -A -x -w -G -np -H --np --ntasks --nodes "
    "--cpus-per-task --partition --time --job-name --host --hostfile "
    "-machinefile --map-by --bind-to --mem --gres --gpus --account --qos "
    "--export --mca --nodelist"
)
_WRAPPERS: dict[str, tuple[str, str, str, int, str]] = {
    # name: (options taking a value, flags, options running code,
    #        positionals before the command, subcommand it must use)
    "exec": ("-a", "-c -l", "", 0, ""),
    "command": ("", "-p", "", 0, ""),
    "builtin": ("", "", "", 0, ""),
    "nohup": ("", "", "", 0, ""),
    "nice": ("-n --adjustment", "", "", 0, ""),
    "ionice": ("-c -n -p --class --classdata", "-t", "", 0, ""),
    "chrt": ("", "-f -r -o -b -i -d -a -m -v -R", "", 1, ""),
    "prlimit": ("", "", "", 0, ""),
    "unshare": ("", "-m -u -i -n -p -U -r -f -C -T --fork --mount", "", 0, ""),
    "runuser": ("-u -g -G --user --group", "-l -m -p", "-c --command", 0, ""),
    "setpriv": ("", "--clear-groups --init-groups --keep-groups", "", 0, ""),
    "strace": (
        "-o -e -p -s -u -P -I -E -b -a -O -S -X",
        "-f -ff -c -C -t -tt -ttt -T -v -x -xx -y -yy -q -qq",
        "",
        0,
        "",
    ),
    "valgrind": ("", "-q -v", "", 0, ""),
    "xvfb-run": (
        "-n -s -f -e -p -w --server-num --server-args --auth-file "
        "--error-file --xauth-protocol --wait",
        "-a --auto-servernum",
        "",
        0,
        "",
    ),
    "timeout": (
        "-s --signal -k --kill-after",
        "--preserve-status --foreground -v --verbose",
        "",
        1,
        "",
    ),
    "time": (
        "-o -f --output --format",
        "-p -v --verbose --portability",
        "",
        0,
        "",
    ),
    "stdbuf": ("-i -o -e", "", "", 0, ""),
    "taskset": ("", "-c --cpu-list -a --all-tasks", "", 1, ""),
    "numactl": (
        "-N -m -C -p -i --cpunodebind --membind --physcpubind --preferred "
        "--interleave",
        "-l --localalloc",
        "",
        0,
        "",
    ),
    "srun": (_SLURM_MPI, "-l --label --exclusive", "", 0, ""),
    "mpirun": (_SLURM_MPI, "--oversubscribe --allow-run-as-root", "", 0, ""),
    "mpiexec": (_SLURM_MPI, "--oversubscribe --allow-run-as-root", "", 0, ""),
    "env": (
        "-u --unset -C --chdir",
        "-i --ignore-environment -0 --null -",
        "-S --split-string",
        0,
        "",
    ),
    "sudo": (
        "-u -g -C -D -h -p -r -t -U --user --group",
        "-n -E -H -P -b",
        "-s -i --shell --login",
        0,
        "",
    ),
    "doas": ("-u -C", "-n", "-s", 0, ""),
    "chroot": ("--userspec --groups", "", "", 1, ""),
    "flock": (
        "-w --timeout -E --conflict-exit-code",
        "-s -x -n -u -o --shared --exclusive --nonblock --unlock --close",
        "-c --command",
        1,
        "",
    ),
    "setsid": ("", "-f -w -c --fork --wait --ctty", "", 0, ""),
    **dict.fromkeys(
        ("conda", "mamba", "micromamba"),
        (
            "-n --name -p --prefix --cwd",
            "--no-capture-output --live-stream",
            "",
            0,
            "run",
        ),
    ),
    "pixi": (
        "-e --environment --manifest-path",
        "--frozen --locked",
        "",
        0,
        "run",
    ),
    "uv": (
        "--with -p --python --project --directory",
        "--no-project --isolated --frozen --locked",
        "",
        0,
        "run",
    ),
    "poetry": ("", "", "", 0, "run"),
    **dict.fromkeys(
        ("apptainer", "singularity"),
        (
            "-B --bind -H --home --pwd -W --workdir --env --env-file "
            "--overlay --security",
            "--nv --nvccli --rocm --cleanenv -c --containall --contain "
            "--writable-tmpfs --no-home -e",
            "",
            1,
            "exec",
        ),
    ),
    **dict.fromkeys(
        ("docker", "podman"),
        (
            "-v --volume -e --env -w --workdir -u --user --name --network "
            "--entrypoint --gpus --cpus -m --memory --platform --env-file",
            "--rm -i -t -it -d --init",
            "",
            1,
            "run",
        ),
    ),
}
"""Commands that run their arguments as another command."""
_KEYWORDS = frozenset(
    {"if", "then", "else", "elif", "while", "until", "do", "!", "coproc", "{"}
)
"""Shell words a command may follow."""
_REPARSE = frozenset(
    {"eval", "ssh", "su", "xargs", "parallel", "watch", "let", "trap"}
)
"""Commands that parse their arguments as shell code again."""
_ARITH_DECL = frozenset({"declare", "typeset", "local", "readonly", "export"})
_SUBSCRIPT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\[[^\]]*$")
_HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z0-9_]+)\2")
_REDIRECT = re.compile(r"\d*[<>]")
CODE_POSITION = "where the command runs it as code"
"""The reason given for a placeholder naming the program or script to run."""


def unsafe_placeholders(text: str, names: set[str]) -> dict[str, str]:
    """
    Find the placeholders in *text* that a command runs as code.

    The placeholders among *names* that *text* uses so, name -> why:

    - in arithmetic (``$((``, ``((``, ``$[``, ``[[ ]]``, ``let``,
      ``declare -i``, an array subscript) or a ``${...}`` expansion;
    - in arguments a command parses as code again (``eval``, ``ssh``,
      ``xargs``...), a here-string, or anything piped into a shell or an
      interpreter that reads its code from stdin;
    - :data:`CODE_POSITION`: as the program to run, the script or ``-c``/
      ``-e`` code of a shell or an interpreter, also inside quoted text or
      a heredoc that is such code.

    Quoted text and heredocs that are data (``--config '{"x": "${x}"}'``,
    ``cat > f <<EOF``) are fine: every value is made of characters that
    cannot end a quote or expand there. A template whose quoting cannot be
    followed, or with a ``<<-`` heredoc, marks every name.
    """
    return {n: r for n, (r, _e) in _classify(text, names).items() if r}


def embedded_placeholders(text: str, names: set[str]) -> set[str]:
    """The placeholders inserted into quoted text or a heredoc."""
    return {n for n, (_r, e) in _classify(text, names).items() if e}


def _classify(
    text: str, names: set[str]
) -> dict[str, tuple[str | None, bool]]:
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
        return dict.fromkeys(
            found, ("in a template with a <<- heredoc", False)
        )
    if _pipes_into_code(skeleton):
        return dict.fromkeys(
            found, ("in a template that pipes into a shell", False)
        )
    marks = _scan(skeleton)
    if len(marks) != len(found):
        return dict.fromkeys(found, ("its quoting cannot be followed", False))
    out: dict[str, tuple[str | None, bool]] = {}
    for name, (reason, embedded) in zip(found, marks, strict=True):
        old_reason, old_embedded = out.get(name, (None, False))
        out[name] = (old_reason or reason, old_embedded or embedded)
    return out


def uses_python_code(text: str) -> bool:
    """
    Whether a command in *text* runs Python with ``-c`` or ``-m``, which
    put the working directory on the module path before Python 3.11.
    """
    for part in re.split(r"[;&|()\n`]+", text):
        rest = _command(part.split())
        if isinstance(rest, str) or not rest:
            continue
        if _PYTHON.fullmatch(_base(rest[0])) and any(
            w in ("-c", "-m") or (w[:2] in ("-c", "-m") and w[0] == "-")
            for w in rest[1:]
            if w.startswith("-") and not w.startswith("--")
        ):
            return True
    return False


def _pipes_into_code(skeleton: str) -> bool:
    """Whether a pipe feeds a shell or an interpreter its code."""
    for target in re.split(r"\|&?", skeleton)[1:]:
        words = re.split(r"[;&|)\n]", target, maxsplit=1)[0].split()
        if words and _reads_stdin_code(words):
            return True
    return False


def _base(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _spec(program: str) -> tuple[str, str, str, str, str] | None:
    if _PYTHON.fullmatch(program):
        return _PROGRAMS["python"]
    return _PROGRAMS.get(program)


def _command(words: list[str]) -> list[str] | str:
    """
    The words from the program on, past assignments, shell keywords and
    wrappers (``timeout -s KILL 60``, ``env -u X``, ``conda run -n e``...),
    or why the program cannot be told: an option the agent does not know,
    or one that runs code.
    """
    rest = [w for w in words if not re.fullmatch(r"[A-Za-z_]\w*=.*", w)]
    while rest and (rest[0] in _KEYWORDS or _base(rest[0]) in _WRAPPERS):
        if rest[0] in _KEYWORDS:
            rest.pop(0)
            continue
        name = _base(rest.pop(0))
        takes, flags, code, positional, sub = _WRAPPERS[name]
        if sub:
            if not rest or rest[0] != sub:
                return [name, *rest]  # the tool itself, not a wrapper
            rest.pop(0)
        while rest and rest[0].startswith("-"):
            option = rest.pop(0)
            if option == "--":
                break
            key = option.split("=", 1)[0]
            if name == "command" and key in ("-v", "-V"):
                return ["command", "-v"]  # prints a path, runs nothing
            if key in code.split():
                return "in a command that runs it as code"
            if key in takes.split() and "=" not in option:
                rest = rest[1:]
            elif not (
                key in takes.split()
                or key in flags.split()
                or (key.startswith("--") and "=" in option)
                or re.fullmatch(r"-\d+", option)  # nice -10
            ):
                return (
                    f"after a wrapper option the agent cannot read: {option}"
                )
        rest = rest[positional:]
    return rest


def _read_args(  # noqa: PLR0912
    spec: tuple[str, str, str, str, str], args: list[str]
) -> tuple[str, int, bool, bool]:
    """
    Read a program's arguments as far as they go: what the next word is
    (``code``, ``data`` or ``positional``), how many positionals came, and
    whether code was given by an option or an option was not known.
    """
    takes, flags, code_opts, code_flags, _kind = spec
    takes_l, flags_l, code_l = takes.split(), flags.split(), code_opts.split()
    expect = "positional"
    positionals = 0
    code_given = unknown = False
    for word in args:
        if expect in ("code", "data"):
            expect = "positional"
            continue
        if (
            word.startswith(("-", "+"))
            and word not in ("-", "--")
            and (positionals == 0 or _kind in ("data", "all", "run"))
        ):
            key = word.split("=", 1)[0]
            if key in code_l or key in code_flags.split():
                code_given = True
                if key in code_l and "=" not in word:
                    expect = "code"
                continue
            if key in takes_l:
                if "=" not in word:
                    expect = "data"
                continue
            if key in flags_l or (key.startswith("--") and "=" in word):
                continue
            # A short cluster (-ne) or an attached value (-Ilib, -mjson).
            if not word.startswith("--") and len(word) > 2:
                head = word[:2]
                if head in code_l:
                    code_given = True
                    continue
                if head in takes_l:
                    continue
                letters = [f"-{c}" for c in word[1:]]
                if any(c in code_l for c in letters):
                    code_given = True
                    if letters[-1] in code_l:
                        expect = "code"
                    continue
                if all(c in flags_l for c in letters):
                    continue
            unknown = True
            continue
        positionals += 1
    return expect, positionals, code_given, unknown


def _program_reason(rest: list[str], word: str) -> str | None:  # noqa: PLR0911
    """Why a placeholder in the arguments *rest* of a program is code."""
    program = _base(rest[0])
    if program == "find":
        for marker in ("-exec", "-execdir", "-ok", "-okdir"):
            if marker in rest:
                after = rest[rest.index(marker) + 1 :]
                return _code_reason(after, word) if ";" not in after else None
        return None
    spec = _spec(program)
    if spec is None:
        return None
    kind = spec[4]
    args = [w for w in rest[1:] if not _REDIRECT.match(w)]
    expect, positionals, code_given, unknown = _read_args(spec, args)
    if word:  # attached: --in=${x}, -e${x}
        key = word.split("=", 1)[0] if "=" in word else word[:2]
        return CODE_POSITION if key in spec[2].split() else None
    if expect == "code":
        return CODE_POSITION
    if expect == "data":
        return None
    if kind == "all":
        return CODE_POSITION
    if kind == "run":
        return (
            CODE_POSITION
            if positionals == 1 and rest[1:2] == ["run"]
            else None
        )
    if kind == "first" and not code_given:
        if positionals == 0 or (unknown and positionals == 1):
            return CODE_POSITION
    if kind == "first" and spec[3] and code_given and positionals == 0:
        return CODE_POSITION  # sh -c ${x}
    return None


def _reads_stdin_code(words: list[str]) -> bool:
    """Whether the command in *words* reads its code from stdin."""
    rest = _command(words)
    if isinstance(rest, str) or not rest:
        return False
    spec = _spec(_base(rest[0]))
    if spec is None or spec[4] == "all":
        return False
    args = [w for w in rest[1:] if not _REDIRECT.match(w)]
    _e, positionals, code_given, _u = _read_args(spec, args)
    if code_given:
        return False
    if spec[4] == "data":
        return _base(rest[0]) == "R"
    return positionals == 0 or "-" in args[:1]


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
    if _in_test(words):
        return "inside [[ ]]"
    if any(b in _ARITH_DECL for b in bases) and any(
        w.startswith("-") for w in words
    ):
        return "in a typed declaration"
    if "printf" in bases and "-v" in words:
        return "in printf -v"
    rest = _command(words)
    if isinstance(rest, str):
        return rest
    if not rest:
        return CODE_POSITION if not word else None
    return _program_reason(rest, word)


def _in_test(words: list[str]) -> bool:
    """Whether the words so far open a ``[[`` they have not closed."""
    if "[[" not in words:
        return False
    last = len(words) - 1 - words[::-1].index("[[")
    return "]]" not in words[last:]


def _scan(s: str) -> list[tuple[str | None, bool]]:  # noqa: PLR0912, PLR0915
    """
    Walk shell text where each placeholder is a NUL and say, for each in
    order, why its spot runs it as code (``None``: it does not), and
    whether it sits in quoted text or a heredoc. An unbalanced quote or
    substitution gives an empty list.
    """
    out: list[tuple[str | None, bool]] = []
    # Each frame: (kind, words of the current command in it, runs as code)
    stack: list[tuple[str, list[str], bool]] = [("top", [], False)]
    word = ""
    heredocs: list[tuple[str, bool]] = []
    herestring = False
    closers = {"${": "}", "$((": "))", "((": "))", "$[": "]"}
    inside = {
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
        nonlocal word
        parent, words, code = stack[-1]
        if parent in ("top", "$(", "`"):
            code = code or _code_reason(words, word) is not None
        stack.append((kind, [], code))
        if kind in ("$(", "`"):
            word = ""  # a substitution starts its own command

    while i < n:
        c = s[i]
        kind, words, code = stack[-1]
        if kind == "'":
            if c == "'":
                stack.pop()
            elif c == _SENTINEL:
                out.append(
                    (
                        "inside text a command runs as code" if code else None,
                        True,
                    )
                )
            i += 1
            continue
        if kind == '"' or kind in closers:
            close = '"' if kind == '"' else closers[kind]
            if c == _SENTINEL:
                if kind == '"':
                    out.append(
                        (
                            "inside text a command runs as code"
                            if code
                            else None,
                            True,
                        )
                    )
                else:
                    out.append((inside[kind], False))
                i += 1
            elif c == "\\":
                i += 2
                continue
            elif s.startswith(close, i):
                stack.pop()
                i += len(close)
                continue
            else:
                for opener in ("$((", "$(", "${", "$["):
                    if s.startswith(opener, i):
                        push(opener)
                        i += len(opener)
                        break
                else:
                    if c == "`":
                        push("`")
                    i += 1
            continue
        # Unquoted: the top level, $(...), `...` or <(...).
        if c == _SENTINEL:
            if herestring:
                out.append(("in a here-string", False))
            elif code:
                out.append(("in output a command runs as code", False))
            else:
                out.append((_code_reason(words, word), False))
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
                    out.append((None, False))  # a comment
                i += 1
            continue
        if c in "'\"":
            push(c)
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
        if (c == ")" and kind == "$(" and not _in_test(words)) or (
            c == "`" and kind == "`"
        ):
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
            heredocs.append(
                (heredoc.group(3), code or _reads_stdin_code(list(words)))
            )
            i = heredoc.end()
            continue
        if c == "\n" and heredocs:
            end_word()
            words.clear()
            i += 1
            for delim, feeds_code in heredocs:
                while i < n:
                    eol = s.find("\n", i)
                    line = s[i : n if eol < 0 else eol]
                    out.extend(
                        (
                            "inside a heredoc a shell or interpreter runs"
                            if feeds_code
                            else None,
                            True,
                        )
                        for _ in line.split(_SENTINEL)[1:]
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
            if _in_test(words):
                words.append(c)  # [[ ... && ... ]] is still one test
            else:
                herestring = False
                words.clear()
        else:
            word += c
        i += 1
    if len(stack) != 1 or heredocs:
        return []
    return out
