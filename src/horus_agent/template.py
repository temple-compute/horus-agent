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
from contextvars import ContextVar
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


MAX_COMMAND_WORDS = 200
"""Words of one command (up to a separator) the scanner reads."""
MAX_NESTING = 100
"""Quotes and substitutions inside one another the scanner follows."""
_SENTINEL = "\0"
_QUOTED = "\x01"
"""Stands for a quoted part in a word the scanner is building."""
_SHELLS = "sh bash dash zsh ksh ash tcsh csh fish busybox rbash rzsh source ."
_PROGRAMS: dict[str, tuple[str, str, str, str, str]] = {
    # program: (options taking a value, flags, options whose value is code,
    #           options making the first positional the code, positionals:
    #           first = the first is the script; data = none is code;
    #           all = every one is code; run = the one after "run")
    **dict.fromkeys(
        _SHELLS.split(),
        (
            "-o +o -O +O",
            "-e -u -x -v -l -i -s -n -f -a -h -m -B -C -E -H -P -T -p "
            "--login --norc --noprofile --posix --restricted --verbose",
            "--rcfile --init-file",
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
    **dict.fromkeys(
        ("at", "batch"), ("-q -t", "-m -M -v -l -d -r -b", "-f", "", "data")
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
        "--input-type --conditions",
        "--no-warnings --enable-source-maps --trace-warnings",
        "-e --eval -p --print -r --require --import --loader",
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
    "java": (
        "--add-modules --add-opens --add-exports -D -X",
        "-server -client -ea -da -esa -dsa -verbose",
        "-jar -cp -classpath --class-path -p --module-path -m --module",
        "",
        "first",
    ),
    **dict.fromkeys(
        ("jupyter", "papermill"),
        (
            "--to --output --output-dir --template --kernel -k -p -y -f "
            "--parameters_file --log-level",
            "--no-input --stdout --inplace --log-output --progress-bar",
            "",
            "",
            "all",
        ),
    ),
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
    **dict.fromkeys(
        ("ipython", "groovy", "octave", "bun", "expect", "tclsh", "wish"),
        ("", "", "-c -f -e --eval", "", "first"),
    ),
    **dict.fromkeys(
        ("deno", "pytest", "py.test", "tox", "pip", "pip3", "npm", "npx"),
        ("", "", "", "", "all"),
    ),
    **dict.fromkeys(
        ("yarn", "pnpm", "bunx", "cmake", "gdb", "sqlite3", "psql", "duckdb"),
        ("", "", "-P -x --command -ex -iex -f -c", "", "all"),
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
_CODE_OPTIONS: dict[str, tuple[str, ...]] = {
    # program: options whose value is a command, a script or a library that
    # the program then runs or loads. One table for every tool.
    "git": (
        "-c",
        "--exec",
        "--exec-path",
        "--upload-pack",
        "--receive-pack",
        "-u",
        "--config-env",
    ),
    "tar": (
        "--to-command",
        "--checkpoint-action",
        "--use-compress-program",
        "-I",
        "--info-script",
        "-F",
        "--new-volume-script",
        "--rmt-command",
        "--rsh-command",
    ),
    "rsync": ("-e", "--rsh", "--rsync-path"),
    **dict.fromkeys(("scp", "sftp", "ssh-keygen"), ("-o", "-S", "-F", "-D")),
    "sort": ("--compress-program",),
    "wget": ("-e", "--execute"),
    "curl": ("-K", "--config"),
    "pandoc": ("--filter", "-F", "--lua-filter", "-L"),
    "zip": ("-TT",),
    "java": (
        "-javaagent:",
        "-agentpath:",
        "-agentlib:",
        "-Xbootclasspath",
    ),
}
_EVERYWHERE = frozenset(
    {"eval", "ssh", "su", "xargs", "parallel", "watch", "alias"}
)
"""Commands that make every placeholder of a template untrustworthy."""
_EXEC_FLAGS = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
_SCRIPT_AFTER_CODE = frozenset({"lua"})
"""Programs whose script still follows an ``-e`` of code."""
_ASSIGNERS = frozenset({"export", "declare", "typeset", "readonly", "local"})
_ASSIGNMENT = "in a variable assignment"
_ASSIGN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")
_OPERATOR = re.compile(r"(?:\d*(?:>>|>\||>&|>|<&|<>|<)|&>>|&>)")
_INTERPRETERS = "|".join(
    [
        *(re.escape(k) for k, v in _PROGRAMS.items() if v[4] == "first"),
        r"python[0-9.]*",
        r"pypy[0-9.]*",
    ]
)
_DYNAMIC_ARG = re.compile(
    rf"(?<![\w./-])(?:{_INTERPRETERS})(?:[ \t]+-[^ \t;&|]*)*[ \t]+[\"']?"
    r"\$[A-Za-z_{(@*0-9]"
)
_AWK_RUNS = re.compile(
    r"system[ \t]*\(|\|[ \t]*getline|\|&|print[^;\n]*\|[ \t]*\""
)
_DEVICE_SCRIPT = re.compile(
    rf"(?<![\w./-])(?:{_INTERPRETERS}|at|batch)(?:[ \t]+-[^ \t;&|]*)*[ \t]+"
    r"[\"']?(?:/dev/stdin|/dev/fd/|/proc/(?:self|[0-9]+)/fd/|<\(|>\()"
)
_PYTHON = re.compile(r"python[0-9.]*|pypy[0-9.]*")
_VERSIONED = re.compile(r"[-_.]?[0-9][0-9._-]*$")
_STDIN_CODE = frozenset(
    {
        "R",
        "sqlite3",
        "psql",
        "duckdb",
        "octave",
        "ipython",
        "groovy",
        "at",
        "batch",
    }
)
"""Programs that run what they read from stdin as code (a script or SQL)."""
_SLURM_MPI = (
    "-n -N -c -p -t -J -A -x -w -G -np -H --np --ntasks --nodes "
    "--cpus-per-task --partition --time --job-name --host --hostfile "
    "-machinefile --map-by --bind-to --mem --gres --gpus --account --qos "
    "--export --mca --nodelist"
)
_WRAPPERS: dict[str, tuple[str, str, str, int, str]] = {
    # name: (options taking a value, flags, options running code,
    #        positionals before the command, subcommand it must use)
    "busybox": ("", "", "", 0, ""),
    "script": (
        "-I -O -B -E -T -m --log-in --log-out --log-io --echo --log-timing "
        "--logging-format --timing",
        "-a -e -f -q -r --append --return --flush --quiet --force",
        "-c --command",
        0,
        "",
    ),
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
        (
            "fakeroot",
            "eatmydata",
            "tini",
            "dumb-init",
            "gosu",
            "su-exec",
            "ssh-agent",
            "unbuffer",
            "rlwrap",
            "ltrace",
            "firejail",
            "bwrap",
            "nsenter",
            "cgexec",
        ),
        ("-u -g -t -p -n -s -o -e", "-- -f -a --", "-c --command", 0, ""),
    ),
    "perf": ("-e -o -p -C -x -r -F -c", "-a -d -v -B", "", 1, ""),
    "faketime": ("", "-f", "", 1, ""),
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
    **dict.fromkeys(("pipenv", "hatch", "pdm", "rye"), ("", "", "", 0, "run")),
    "accelerate": ("--config_file --num_processes", "", "", 0, "launch"),
    **dict.fromkeys(
        ("apptainer", "singularity"),
        (
            "-B --bind -H --home --pwd -W --workdir --env --env-file "
            "--overlay --security",
            "--nv --nvccli --rocm --cleanenv -c --containall --contain "
            "--writable-tmpfs --no-home -e",
            "",
            1,
            "exec|run",
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
            "run|exec",
        ),
    ),
}
"""Commands that run their arguments as another command."""
_DATA_TOOLS = frozenset(
    """
    [ test train predict cat head tail wc sort uniq cut paste tr tee ls rm cp
    mv ln mkdir rmdir
    touch chmod chown stat du df file basename dirname readlink realpath
    md5sum sha1sum sha256sum sha512sum cksum gzip gunzip zcat bzip2 bunzip2
    xz unxz zstd pigz unpigz tar unzip zip cmp diff comm join split fold nl
    od hexdump seq echo printf true false test sleep date pwd grep egrep
    fgrep zgrep rev tac shuf column expand unexpand strings truncate sync
    mktemp install rename gawk rsync curl wget git du find xargs cd
    samtools bcftools bedtools bwa bwa-mem2 bowtie bowtie2 hisat2 STAR
    fastqc multiqc trimmomatic cutadapt fastp salmon kallisto picard gatk
    tabix bgzip seqtk minimap2 blastn blastp makeblastdb mafft muscle
    iqtree fasttree raxml-ng hmmer hmmsearch hmmbuild bamtools vcftools
    plink plink2 featureCounts htseq-count stringtie cufflinks bismark
    macs2 deeptools bamCoverage multiBamSummary sambamba bbduk bbmap
    bbsplit dupradar qualimap preseq rseqc spades megahit prokka
    quast busco diamond mmseqs fastx_toolkit seqkit vsearch
    gmx cpptraj pdb4amber tleap antechamber parmchk2 sander pmemd
    obabel vina autodock4 reduce
    """.split()
)
"""Programs that treat their arguments as data (files, values), never as
code to run: a placeholder as their first argument is a path or a value."""
_RUN_WORDS = frozenset(
    """
    -- run exec execute launch activate start submit repro eval source call
    shell script apply play invoke spawn serve -c --command --cmd --exec
    --execute --eval --script --run --hook --init-file --rcfile --entrypoint
    --exec-path -x --shell -e
    """.split()
)
"""
Words after which a program the agent has no entry for takes a command or a
script (a subcommand that runs, or an option whose value runs).
"""
_AT_PROGRAM = frozenset({"alias"})
"""Commands that matter only as the program of a command."""
_FUNCTION_DEF = re.compile(
    r"(?m)(?:^|[;&{(])[ \t]*(?:function[ \t]+)?([A-Za-z_][\w.-]*)[ \t]*\(\)"
)
_DEFINED: ContextVar[frozenset[str]] = ContextVar(
    "defined", default=frozenset()
)
"""Functions the template under scan defines: their bodies are scanned too."""
_COPIERS = frozenset({"cp", "mv", "ln", "install", "rsync", "scp", "dd"})
"""Programs that write a copy of their first arguments to the last one."""
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
    _DEFINED.set(frozenset(_FUNCTION_DEF.findall(skeleton)))
    if why := _whole_reason(skeleton):
        return dict.fromkeys(found, (why, False))
    marks, whole = _scan(skeleton)
    if whole:
        return dict.fromkeys(found, (whole, False))
    if len(marks) != len(found):
        return dict.fromkeys(found, ("its quoting cannot be followed", False))
    out: dict[str, tuple[str | None, bool]] = {}
    for name, (reason, embedded) in zip(found, marks, strict=True):
        old_reason, old_embedded = out.get(name, (None, False))
        out[name] = (old_reason or reason, old_embedded or embedded)
    return out


def _whole_reason(skeleton: str) -> str | None:  # noqa: PLR0911
    """Why no placeholder of a template can be trusted, from its text."""
    if "<<-" in skeleton:
        return "in a template with a <<- heredoc"
    if "$'" in skeleton:
        return "in a template with $'...' quoting"
    if _DYNAMIC_ARG.search(skeleton.replace("\\\n", " ")):
        return "in a template that runs a script named by a variable"
    if _DEVICE_SCRIPT.search(skeleton):
        return "in a template where a shell reads its script from a stream"
    if _pipes_into_code(skeleton):
        return "in a template that pipes into a shell"
    if _runs_data_lines(skeleton):
        return (
            "in a template with an editor or inline script that runs commands"
        )
    if re.search(r"\bawk\b|\bgawk\b|\bmawk\b", skeleton) and _AWK_RUNS.search(
        skeleton
    ):
        return "in a template with an awk program that runs commands"
    return None


_SED_E = re.compile(
    r"\bs([/|#,:@!%])(?:\\.|(?!\1).)*\1(?:\\.|(?!\1).)*\1[0-9a-zA-Z]*e"
)
_SED_ECMD = re.compile(r"(?<![A-Za-z_/\\-])e(?:[ \t;'\"}]|$)")
_SED_CALL = re.compile(r"\b[gs]?sed\b([^\n|]*)")
_INLINE_CODE = re.compile(
    r"\b(?:perl|ruby|node|php|lua|julia|Rscript|python[0-9.]*)\b[^\n;|]*"
    r"[ \t]-[a-zA-Z]*[ceErp]\b"
)
_INLINE_RUNS = re.compile(
    r"\b(?:system|exec|popen|spawn|spawnSync|execSync|child_process|"
    r"subprocess|qx|shell_exec|passthru|proc_open|execute)\b|`"
)


def _runs_data_lines(skeleton: str) -> bool:
    """
    Whether an editor or an inline script in *skeleton* can run the lines
    of a file it is given: ``sed`` with the ``e`` flag or command, or
    inline ``perl``/``ruby``/``python``... code that starts programs.
    """
    for call in _SED_CALL.finditer(skeleton):
        if _SED_E.search(call.group(1)) or _SED_ECMD.search(call.group(1)):
            return True
    return bool(
        _INLINE_CODE.search(skeleton) and _INLINE_RUNS.search(skeleton)
    )


def uses_python_code(text: str) -> bool:
    """
    Whether *text* may run Python: it names an interpreter anywhere (a
    ``-c``/``-m`` form, a heredoc, ``bash -c``, a script path), the one
    shape-free test that cannot miss a spelling.
    """
    return "python" in text.lower()


def _strip_redirects(words: list[str]) -> list[str]:
    """*words* without redirection operators and their targets."""
    out: list[str] = []
    skip = False
    for word in words:
        match = _OPERATOR.match(word)
        if skip:
            skip = False
        elif match:
            skip = match.end() == len(word)
        else:
            out.append(word)
    return out


def _pipes_into_code(skeleton: str) -> bool:
    """Whether a pipe feeds a shell or an interpreter its code."""
    for target in re.split(r"\|&?", skeleton)[1:]:
        words = _strip_redirects(
            re.split(r"[;&|)\n]", target, maxsplit=1)[0].split()
        )
        if words and _reads_stdin_code(words):
            return True
    return False


def _base(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _spec(program: str) -> tuple[str, str, str, str, str] | None:
    if _PYTHON.fullmatch(program):
        return _PROGRAMS["python"]
    for name in (program, program.lower(), _VERSIONED.sub("", program)):
        if len(name) > 1 and name in _PROGRAMS:
            return _PROGRAMS[name]
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
            if not rest or rest[0] not in sub.split("|"):
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
                if not rest:
                    return ["-"]  # the next word is this option's value
                rest = rest[1:]
            elif not (
                key in takes.split()
                or key in flags.split()
                or (key.startswith("--") and "=" in option)
                or re.fullmatch(r"-\d+", option)  # nice -10
                or (  # an attached value: stdbuf -oL, ionice -c3
                    not option.startswith("--")
                    and len(option) > 2
                    and option[:2] in takes.split()
                )
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
    code_given = unknown = ended = False
    for word in args:
        if expect in ("code", "data"):
            expect = "positional"
            continue
        if word == "--" and not ended:
            ended = True  # no more options, and not a positional itself
            continue
        if (
            not ended
            and word.startswith(("-", "+"))
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
        return None  # -exec anywhere refuses the whole template
    spec = _spec(program)
    if spec is None:
        return _unknown_reason(rest, word)
    if program == "R" and rest[1:2] == ["CMD"]:
        return CODE_POSITION  # R CMD BATCH/INSTALL/SHLIB run what they get
    kind = spec[4]
    args = rest[1:]
    expect, positionals, code_given, unknown = _read_args(spec, args)
    if word.startswith("-") and "--" not in args:
        # attached to an option: --in=${x}, -e${x}
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
    if kind == "first" and spec[3] and code_given:
        return CODE_POSITION  # sh -c ${x}, sh -c 'true' ${x}
    if program in _SCRIPT_AFTER_CODE and positionals == 0:
        return CODE_POSITION  # lua -e 1 ${x}
    return None


def _unknown_reason(rest: list[str], word: str) -> str | None:
    """
    Why a placeholder after a program the agent has no entry for may be
    code: it is the program's first non-option argument (``fakeroot
    ${x}``, ``eatmydata ${x}``) and the program is not a known data tool,
    or a later word is an interpreter or a shell that runs it.
    """
    program = rest[0]
    if (
        "/" in program
        or program == "-"
        or program in _DATA_TOOLS
        or program in _DEFINED.get()
    ):
        return None
    for k, other in enumerate(rest[1:], 1):
        if _spec(_base(other)) is not None and _base(other) != program:
            return _program_reason(rest[k:], word)
    if word.startswith("-"):
        return None
    if len(rest) == 1 or rest[-1] in _RUN_WORDS:
        return "after a program the agent does not know"
    return None


def _reads_stdin_code(words: list[str]) -> bool:
    """Whether the command in *words* reads its code from stdin."""
    rest = _command(words)
    if isinstance(rest, str) or not rest:
        return False
    if _base(rest[0]) in _STDIN_CODE:
        return True
    spec = _spec(_base(rest[0]))
    if spec is None or spec[4] == "all":
        return False
    args = rest[1:]
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
    if _EXEC_FLAGS.intersection(words):
        return "in a find -exec command"
    if any(b in _ARITH_DECL for b in bases) and any(
        w.startswith("-") for w in words
    ):
        return "in a typed declaration"
    if "printf" in bases and "-v" in words:
        return "in printf -v"
    rest = _command(words)
    if isinstance(rest, str):
        return rest
    if why := _assign_reason(words, word):
        return why
    if not rest:
        return CODE_POSITION
    program = _base(rest[0])
    options = _CODE_OPTIONS.get(program, ())
    if options and (
        (rest[1:] and rest[-1] in options)
        or (word.startswith("-") and word.startswith(options))
    ):
        return f"in an option of {program} that runs what it is given"
    return _program_reason(rest, word)


def _assign_reason(words: list[str], word: str) -> str | None:
    """Why a placeholder in *word* sets a shell or environment variable."""
    rest = _command(words)
    if isinstance(rest, str):
        return None
    if (
        _ASSIGN.match(word) and (not rest or _base(rest[0]) in _ASSIGNERS)
    ) or (rest and _base(rest[0]) in _ASSIGNERS):
        return _ASSIGNMENT
    return None


def _in_test(words: list[str]) -> bool:
    """Whether the words so far open a ``[[`` they have not closed."""
    if "[[" not in words:
        return False
    last = len(words) - 1 - words[::-1].index("[[")
    return "]]" not in words[last:]


def _scan(  # noqa: PLR0912, PLR0915
    s: str,
) -> tuple[list[tuple[str | None, bool]], str | None]:
    """
    Walk shell text where each placeholder is a NUL and say, for each in
    order, why its spot runs it as code (``None``: it does not), and
    whether it sits in quoted text or a heredoc; and, when the text as a
    whole cannot be trusted, why. An unbalanced quote or substitution gives
    an empty list.
    """
    out: list[tuple[str | None, bool]] = []
    # Each frame: (kind, words of the current command in it, runs as code)
    stack: list[tuple[str, list[str], bool]] = [("top", [], False)]
    word = ""
    heredocs: list[tuple[str, bool, bool, list[str], int]] = []
    here = -1  # stack depth of an open here-string word, else -1
    skip = ""  # a redirection operator whose target is the next word
    targets: list[str] = []  # files the current command writes
    started = 0  # len(out) when the current command began
    generated: list[tuple[str, int, int]] = []  # file, first mark, end
    bodies: list[tuple[list[str], int, str]] = []  # files, first mark, text
    ran: set[str] = set()  # words given to a program or an interpreter
    whole: list[str] = []
    closers = {"${": "}", "$((": "))", "((": "))", "$[": "]"}
    inside = {
        "${": "inside a ${...} expansion",
        "$((": "inside arithmetic",
        "((": "inside arithmetic",
        "$[": "inside arithmetic",
    }
    i = 0
    n = len(s)

    def end_word() -> None:  # noqa: PLR0912
        nonlocal word, here, skip
        if not word:
            return
        if len(stack) <= here:
            here = -1
        words = stack[-1][1]
        if len(words) >= MAX_COMMAND_WORDS:  # keeps the scan linear
            whole.append(
                f"in a command of over {MAX_COMMAND_WORDS} words, which "
                "cannot be checked"
            )
            word = ""
            return
        match = None if _in_test(words) else _OPERATOR.match(word)
        if (skip or match) and any(_base(w) == "exec" for w in words):
            if _command(words) == []:  # exec 3< file; bash <&3
                whole.append(
                    "in a template that holds a redirection open with exec"
                )
        if skip:
            if (
                ">" in skip
                and word[:1] not in "&-0123456789"
                and _SENTINEL not in word
            ):
                targets.append(word)
            skip = ""
        elif match:
            if match.end() == len(word):
                skip = word
            elif (
                ">" in match.group(0)
                and word[match.end() :][:1] not in "&-0123456789"
                and _SENTINEL not in word
            ):
                targets.append(word[match.end() :])
        else:
            if (
                _command(words) == []
                and not _ASSIGN.match(word)
                and (_QUOTED in word or "\\" in word)
            ):
                whole.append(
                    "in a template that runs a program named with quotes "
                    "or an escape"
                )
            at_program = _command(words) == []
            words.append(word)
            if _base(word) in _EVERYWHERE and (
                at_program or _base(word) not in _AT_PROGRAM
            ):
                whole.append(f"in a template that runs {word}")
        word = ""

    def new_command() -> None:
        nonlocal skip, started
        words = stack[-1][1]
        if (rest := _command(words)) and not isinstance(rest, str):
            if _spec(_base(rest[0])) is not None:
                ran.update(_base(w) for w in rest[1:] if w[:1] != "-")
                ran.update(w for w in rest[1:] if "/" in w)
            ran.add(_base(rest[0]))
        if (
            (rest := _command(words))
            and not isinstance(rest, str)
            and _base(rest[0]) in _COPIERS
            and len(out) > started
        ):
            pos = [w for w in rest[1:] if w[:1] != "-"]
            if len(pos) > 1:
                dest = pos[-1]
                if dest in (".", "..") or dest.endswith("/"):
                    # into a folder: the copy keeps the source's own name
                    targets.extend(
                        _base(w)
                        for w in pos[:-1]
                        if not set(w) & {_SENTINEL, "$", "*", "?"}
                    )
                    if dest.endswith("/") and dest != "/":
                        targets.append(dest)  # whatever lands in the folder
                elif _SENTINEL not in dest:
                    targets.append(dest)  # cp ${x} y.sh; bash y.sh
        if (rest := _command(words)) and not isinstance(rest, str):
            if _base(rest[0]) == "tee":  # fed by a pipe or a redirection
                generated.extend(
                    (w, 0, len(out))
                    for w in rest[1:]
                    if w[:1] != "-" and _SENTINEL not in w
                )
            if _base(rest[0]) == "dd":
                targets.extend(
                    w[3:]
                    for w in rest[1:]
                    if w.startswith("of=") and _SENTINEL not in w
                )
        if targets and len(out) > started:
            generated.extend((t, started, len(out)) for t in targets)
        targets.clear()
        words.clear()
        skip = ""
        started = len(out)

    def push(kind: str) -> None:
        nonlocal word
        if len(stack) > MAX_NESTING:
            whole.append("in a template nested too deeply to check")
        parent, words, code = stack[-1]
        if parent in ("top", "$(", "`"):
            why = _code_reason(words, word)
            code = code or (why is not None and why != _ASSIGNMENT)
        stack.append((kind, [], code))
        if kind in ("$(", "`"):
            word = ""  # a substitution starts its own command

    def quoted_reason(code: bool) -> str | None:
        if here >= 0:
            return "in a here-string"
        if code:
            return "inside text a command runs as code"
        parent = stack[-2]
        if parent[0] in ("top", "$(", "`"):
            return _assign_reason(parent[1], word)
        return None

    while i < n:
        if whole:  # nothing in the template can be trusted: stop reading
            return [], whole[0]
        c = s[i]
        kind, words, code = stack[-1]
        if kind == "'":
            if c == "'":
                stack.pop()
            elif c == _SENTINEL:
                out.append((quoted_reason(code), True))
            i += 1
            continue
        if kind == '"' or kind in closers:
            close = '"' if kind == '"' else closers[kind]
            if c == _SENTINEL:
                if kind == '"':
                    out.append((quoted_reason(code), True))
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
            if here >= 0:
                out.append(("in a here-string", False))
            elif code:
                out.append(("in output a command runs as code", False))
            elif skip or _OPERATOR.fullmatch(word):
                # a redirection target; a shell reads its code from `<`
                op = skip or word
                reads = "<" in op and ">" not in op
                why = None
                if reads and _reads_stdin_code(list(words)):
                    why = "in input a shell or interpreter reads as code"
                out.append((why, False))
            else:
                out.append((_code_reason(words, word), False))
            word += _SENTINEL
            i += 1
            continue
        if s.startswith("\\\n", i):  # a line continuation
            i += 2
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
        if (c in "$`" and not _ASSIGN.match(word)) or (
            not word and c in "'\"" and s[i + 1 : i + 2] == "$"
        ):
            if _command(words) == []:
                whole.append(
                    "in a template that runs a command named by a variable"
                )
        if c in "'\"":
            push(c)
            word += _QUOTED
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
            new_command()
            stack.pop()
            word = "x"
            i += 1
            continue
        if s.startswith("<<<", i):
            end_word()
            here = len(stack)
            i += 3
            continue
        if heredoc := _HEREDOC.match(s, i):
            end_word()
            heredocs.append(
                (
                    heredoc.group(3),
                    code or _reads_stdin_code(list(words)),
                    not heredoc.group(2),
                    list(targets),
                    started,
                )
            )
            i = heredoc.end()
            continue
        if c == "\n" and heredocs:
            end_word()
            i += 1
            for delim, feeds_code, expands, files, first in heredocs:
                body: list[str] = []
                body_start = len(out)
                while i < n:
                    eol = s.find("\n", i)
                    line = s[i : n if eol < 0 else eol]
                    if expands and ("$(" in line or "`" in line):
                        whole.append(
                            "in a template with a command substitution in a "
                            "heredoc"
                        )
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
                    body.append(line)
                else:
                    return [], None
                generated.extend((t, first, body_start) for t in files)
                bodies.append((files, body_start, "\n".join(body)))
            heredocs.clear()
            new_command()
            continue
        if c == "&" and (word.endswith((">", "<")) or s.startswith("&>", i)):
            word += c  # 2>&1, >&2, &>file: part of a redirection
        elif c in " \t":
            end_word()
        elif c in ";&|\n(){}":
            end_word()
            if _in_test(words):
                words.append(c)  # [[ ... && ... ]] is still one test
            else:
                if len(stack) <= here:
                    here = -1
                new_command()
        else:
            word += c
        i += 1
    end_word()
    new_command()
    if len(stack) != 1 or heredocs:
        return [], None
    for target, first, end in generated:
        if (
            _base(target) in ran
            or target in ran
            or (
                target.endswith("/") and any(r.startswith(target) for r in ran)
            )
        ):
            for k in range(first, end):
                if out[k][0] is None:
                    out[k] = (
                        "in a script the template writes and then runs",
                        out[k][1],
                    )
    for files, first, text in bodies:
        if not any(_base(t) in ran or t in ran for t in files):
            continue
        # The heredoc is a script the template runs: read it as shell text.
        count = text.count(_SENTINEL)
        inner, bad = (
            _scan(text + "\n") if _whole_reason(text) is None else ([], "x")
        )
        for k in range(count):
            reason = (
                "in a script the template writes and then runs"
                if bad or len(inner) != count
                else inner[k][0]
            )
            out[first + k] = (reason, True)
    return out, whole[0] if whole else None
