#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Running an approved task: the agent renders its command itself.

The op names a task of the approved manifest and binds each placeholder of
its command to a path in the run's directory or a value. The agent checks
every binding against the manifest, renders the command exactly as the
runtime would (each value shell-quoted) and runs it, directly or as a Slurm
batch job whose script it also writes itself. The sender never supplies a
command string.
"""

import asyncio
import contextlib
import os
import pwd
import re
import shlex
import stat
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horus_agent import native, seal
from horus_agent.blobs import open_file
from horus_agent.guard import SAFE_PATH, Admitted, Guard, name_problem
from horus_agent.manifest import (
    RESERVED_ENV,
    VALUE_KINDS,
    Manifest,
)
from horus_agent.protocol import PolicyRefusedError, TaskLaunchArgs
from horus_agent.template import placeholders, render

SIDE_ENV = "HORUS_SIDE_ARTIFACTS_DIR"
_SBATCH_BAD = re.compile(r"[\s#'\"\\\x00]")
JOB_ID = re.compile(r"\d+(_\d+)?")


SAFE_VALUE = re.compile(r"[A-Za-z0-9._,:=+@/-]*")
"""Characters a free text value chosen at run time may hold."""
NUMBER = re.compile(r"[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?")
LIST_KINDS = frozenset({"list", "multichoice"})
HOME_SUBDIR = "home"
"""Each approved job's own fresh home, inside its job dir."""
UNSET_ENV = (
    "NODE_PATH",
    "NODE_OPTIONS",
    "BASH_ENV",
    "ENV",
    "PERL5OPT",
    "PERL5LIB",
    "PERLLIB",
    "PERL_USE_UNSAFE_INC",
    "RUBYOPT",
    "RUBYLIB",
    "JAVA_TOOL_OPTIONS",
    "_JAVA_OPTIONS",
    "JDK_JAVA_OPTIONS",
    "PYTHONPATH",
    "PYTHONSTARTUP",
)
"""Variables an approved job never sees."""
TASK_ID = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*(\[\d+\])?")
"""A task id as the runtime writes it: ``sub/task`` or a map clone ``x[3]``."""


@dataclass
class Prepared:
    """A verified task command, ready to start."""

    command: str
    """The rendered command."""
    env: dict[str, str]
    """The task's environment; reaches it through stdin, not a file."""
    cwd: Path
    """The task's working directory, where the runtime runs it too."""
    job_dir: Path
    side_dir: Path
    task: dict[str, Any]

    @property
    def shell(self) -> str:
        """The command as the job's shell runs it."""
        return f"unset {' '.join(UNSET_ENV)}; {self.command}"


def _template(task: dict[str, Any], script: Path | None) -> str:
    runtime = task["runtime"]
    if runtime["kind"] == "command":
        return str(runtime["command"])
    if script is None:
        raise PolicyRefusedError("A python_script task needs its script")
    # The runtime does not quote the interpreter, only the script path. -I
    # keeps the script's folder, the job's cwd and PYTHON* out of sys.path.
    head = f"{runtime['python']} -I {shlex.quote(str(script))}"
    return f"{head} {runtime['args']}".rstrip()


def _working_dir(
    guard: Guard, run_id: str | None, raw: str, manifest: Manifest
) -> Path:
    """
    The task's working directory, where the runtime runs a task and puts
    its relative outputs: confined to the run, reached through no link
    below the work dir, and a folder of the agent or of the task user
    (created here when missing). Neither it nor any folder up to the run's
    may hold a tool config name an op wrote. Scripts and constants run from
    the agent's copies, and Python with ``-I`` imports nothing from it.
    """
    cwd = guard.confine_bound(run_id, raw)
    if cwd != guard.unlinked(raw):
        raise PolicyRefusedError(
            f"Working directory goes through a link: {raw}"
        )
    cwd.mkdir(parents=True, exist_ok=True)
    info = cwd.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in guard.owners:
        raise PolicyRefusedError(
            f"Working directory is not a folder the agent or its tasks "
            f"made: {raw}"
        )
    root = next(r for r in guard.roots(run_id) if cwd.is_relative_to(r))
    allow = guard.policy.allow_code_suffixes
    if found := held_code(
        cwd, root, allow, guard.code_ok(run_id), _ships(guard, manifest)
    ):
        raise PolicyRefusedError(
            f"{found[0]} holds settings or code a tool would read on its "
            f"own; it was not made by an approved task: {found[1]}"
        )
    if guard.policy.strict and not guard.policy.allow_same_user:
        allowed = guard.layout_names(manifest)
        mine = os.geteuid()
        for folder in dict.fromkeys((cwd, root)):
            with os.scandir(folder) as entries:
                for entry in entries:
                    if (
                        not guard.layout_allows(entry.name, allowed)
                        and entry.stat(follow_symlinks=False).st_uid == mine
                    ):
                        raise PolicyRefusedError(
                            f"{entry.path} is not an input, output, script or "
                            "task of the approved release, and an op wrote it"
                        )
    return cwd


SCAN_CAP = 200_000
"""Entries of the working directory looked at; beyond it the scan stops."""


def held_code(
    cwd: Path,
    root: Path,
    allow: tuple[str, ...],
    approved: set[str] = frozenset(),  # type: ignore[assignment]
    ships: Callable[[Path, str], bool] = lambda _p, _d: False,
) -> tuple[Path, str] | None:
    """
    The first entry the agent's own user wrote (an op, not a job) that a
    tool or an interpreter would load on its own, and why: anywhere below
    *cwd*, and directly in each folder from *cwd* up to *root*. Code the
    release approves passes: paths in *approved*, and files whose content
    is the content its manifest ships at that path (*ships*).
    """
    mine = os.geteuid()
    seen = 0
    folders = [cwd]
    while folders[-1] != root:
        folders.append(folders[-1].parent)
    ancestors = set(folders[1:])
    pending = list(folders)
    while pending and seen < SCAN_CAP:  # shortcut: very large trees pass
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                seen += 1
                info = entry.stat(follow_symlinks=False)
                if entry.is_dir(follow_symlinks=False):
                    if folder not in ancestors:
                        pending.append(Path(entry.path))
                if info.st_uid != mine:
                    continue
                if why := name_problem(entry.name, allow):
                    if _approved(Path(entry.path), approved, ships):
                        continue
                    return Path(entry.path), why
    return None


def _approved(
    path: Path, approved: set[str], ships: Callable[[Path, str], bool]
) -> bool:
    """Whether *path* is code the release approves, by place and content."""
    if any(str(p) in approved for p in (path, *path.parents)):
        return True
    try:
        return path.is_file() and ships(path, native.digest(path))
    except OSError:
        return False


def _ships(guard: Guard, manifest: Manifest) -> Callable[[Path, str], bool]:
    return lambda path, content: guard.shipped(path, content, manifest)


async def _verified(src: Path, store: Path, expected: str, name: str) -> Path:
    """The agent's own copy of *src*, which must be the approved one."""
    try:
        return await asyncio.to_thread(
            native.materialise, src, store, expected
        )
    except FileNotFoundError:
        raise PolicyRefusedError(f"{name!r} is missing") from None
    except (ValueError, OSError) as exc:
        raise PolicyRefusedError(
            f"{name!r} is not the approved file or folder: {exc}"
        ) from exc


async def _seal(
    guard: Guard,
    run_id: str | None,
    manifest: Manifest,
    cwd: Path,
) -> None:
    """
    Put the agent's verified copy of every shipped file or folder that sits
    where a command could run it by its literal path (see
    :mod:`horus_agent.seal`), and note each as sealed. A path that cannot
    be sealed refuses the launch.
    """
    root = next(r for r in guard.roots(run_id) if cwd.is_relative_to(r))
    found: dict[str, list[tuple[Path, tuple[str, ...]]]] = {}
    for base, rel, content in guard.seal_plan(manifest, cwd, root):
        if seal.occupied(base, rel):
            found.setdefault(content, []).append((base, rel))
    for content, places in found.items():
        stored, problem = None, None
        for base, rel in places:  # one place holding the real thing will do
            try:
                stored = await _verified(
                    base.joinpath(*rel), guard.const_store, content, rel[-1]
                )
                break
            except PolicyRefusedError as exc:
                problem = problem or exc
        if stored is None:
            raise problem or PolicyRefusedError("nothing to seal")
        for base, rel in places:
            target = base.joinpath(*rel)
            await asyncio.to_thread(seal.seal, base, rel, stored, guard.owners)
            await asyncio.to_thread(seal.verify, target, content)
            guard.add_sealed(run_id, target)


def job_home(guard: Guard, job_dir: Path) -> Path:
    """
    An approved job's ``HOME``: ``[jobs] home``, else the task user's home
    (outside any run, so no op can write there), else a fresh one in the
    job dir.
    """
    jobs = guard.policy.jobs
    if jobs.home:
        return Path(jobs.home)
    if jobs.run_as:
        with contextlib.suppress(KeyError):
            return Path(pwd.getpwnam(jobs.run_as).pw_dir)
    return job_dir / HOME_SUBDIR


def job_env(home: Path) -> dict[str, str]:
    """
    What every approved job's environment pins: its home, no user startup
    files for R, and Python never importing from the working directory
    (``python -m``/``-c`` would put it first on ``sys.path``).
    """
    return {
        "HOME": str(home),
        "R_PROFILE_USER": os.devnull,
        "R_ENVIRON_USER": os.devnull,
        "PYTHONSAFEPATH": "1",
    }


PythonVersion = Callable[[], Awaitable[tuple[int, int] | None]]
"""Asks which Python a job gets (as the task user), ``None`` if none."""
SAFE_PATH_PYTHON = (3, 11)
"""First Python that honours ``PYTHONSAFEPATH``."""


async def _check_python(
    guard: Guard,
    manifest: Manifest,
    cwd: Path,
    run_id: str | None,
    probe: PythonVersion | None,
) -> None:
    """
    With the policy letting code suffixes into runs, a working directory
    that holds such code, and a job Python older than 3.11 (which imports
    from the working directory whatever ``PYTHONSAFEPATH`` says), no task
    starts: whether a command runs Python (a heredoc, ``bash -c``, a
    script's shebang, a wrapper) is not decided by reading it.
    """
    if not guard.policy.allow_code_suffixes:
        return
    root = next(r for r in guard.roots(run_id) if cwd.is_relative_to(r))
    if (
        held_code(
            cwd, root, (), guard.code_ok(run_id), _ships(guard, manifest)
        )
        is None
        or probe is None
    ):
        return
    version = await probe()
    if version is None or version < SAFE_PATH_PYTHON:
        raise PolicyRefusedError(
            "The job's Python is older than 3.11 and the working directory "
            "holds code (see allow_code_suffixes), so no task starts "
            "here; use Python 3.11 or newer"
        )


async def prepare(
    guard: Guard,
    run_id: str | None,
    admitted: Admitted,
    *,
    slurm: bool,
    python_version: PythonVersion | None = None,
) -> Prepared:
    """
    Check an admitted ``task_launch`` / ``slurm_submit`` against its manifest
    task and render the command. Constant inputs and the script are copied
    into the agent's own folder as they are verified (off the event loop),
    and the command names those copies, so nothing an op can still change
    runs.

    Raises:
        PolicyRefusedError: Anything does not match the approved release.
    """
    args, task = admitted.args, admitted.task
    if (
        not isinstance(args, TaskLaunchArgs)
        or task is None
        or admitted.manifest is None
    ):
        raise PolicyRefusedError("Not a task launch of an approved release")
    task_id = args.task_id
    if not TASK_ID.fullmatch(task_id):
        raise PolicyRefusedError(f"Invalid task id {task_id!r}")
    chain = task["target"]["chain"]
    if (chain[0] == "slurm") != slurm:
        method = "slurm_submit" if chain[0] == "slurm" else "task_launch"
        raise PolicyRefusedError(f"Task {task_id!r} runs through {method}")
    job_dir = guard.confine_bound(run_id, args.job_dir)
    cwd = _working_dir(guard, run_id, args.cwd, admitted.manifest)
    await _seal(guard, run_id, admitted.manifest, cwd)
    await _check_python(guard, admitted.manifest, cwd, run_id, python_version)
    store = guard.const_store
    script = None
    if task["runtime"]["kind"] == "python_script":
        if args.script is None:
            raise PolicyRefusedError("A python_script task needs its script")
        script = await _verified(
            guard.confine_bound(run_id, args.script, code_ok=True),
            store,
            task["runtime"]["script_sha256"],
            "script",
        )
    template = _template(task, script)
    ids = set(task["inputs"]) | set(task["outputs"])
    wanted = {n for n in placeholders(template) if n in ids}
    if set(args.bindings) != wanted:
        raise PolicyRefusedError(
            f"Bindings must be exactly {sorted(wanted)}, got "
            f"{sorted(args.bindings)}"
        )
    values: dict[str, str | list[str]] = {
        "task.id": task_id,
        "task.name": task.get("name", task_id),
    }
    for name, binding in args.bindings.items():
        values[name] = await _bind(guard, run_id, name, task, binding, store)
    _check_env(task, args.env)
    names = {*task["env"], *args.env}
    for name, value in values.items():
        # Bash evaluates a variable named in arithmetic: a value that is the
        # name of one of the job's variables could run what that holds.
        if names & set(value if isinstance(value, list) else [value]):
            raise PolicyRefusedError(
                f"{name!r} may not be the name of an environment variable"
            )
    side = cwd / "side-artifacts"
    if SIDE_ENV in args.env:
        side = guard.confine_bound(run_id, args.env[SIDE_ENV])
    guard.note_made(run_id, cwd)
    for name, binding in args.bindings.items():
        if binding.path is not None and (task["inputs"].get(name) is None):
            guard.note_made(run_id, Path(str(values[name])))  # an output
    command = render(template, values)
    env = {**args.env, **job_env(job_home(guard, job_dir))}
    return Prepared(command, env, cwd, job_dir, side, task)


async def _bind(  # noqa: PLR0913, PLR0917
    guard: Guard,
    run_id: str | None,
    name: str,
    task: dict[str, Any],
    binding: Any,
    store: Path,
) -> str | list[str]:
    spec = task["inputs"].get(name)
    role = "output" if spec is None else spec["role"]
    if binding.path is not None:
        if role == "param":
            if not SAFE_PATH.fullmatch(binding.path):
                raise PolicyRefusedError(
                    f"{name!r}: path has characters a command cannot take "
                    "safely"
                )
            return str(guard.confine_param(run_id, binding.path))
        path = guard.confine_bound(
            run_id, binding.path, code_ok=role == "constant"
        )
        if role == "constant":
            expected = spec.get("sha256") if spec else None
            if expected is None:
                raise PolicyRefusedError(f"{name!r} must be a value")
            return str(await _verified(path, store, expected, name))
        return str(path)
    if role == "param" or (
        role == "edge"
        and spec is not None
        and (spec["kind"] in VALUE_KINDS or spec["kind"] == "item")
    ):
        # A value chosen at run time; rendered as one quoted shell word.
        value: str | list[str] = binding.value
        check_value(name, spec or {}, value)
        return value
    if role == "constant" and spec is not None:
        if binding.value != spec.get("value"):
            raise PolicyRefusedError(f"{name!r} is not the approved value")
        approved: str | list[str] = spec["value"]
        return approved
    raise PolicyRefusedError(f"{name!r} must be a path")


def check_value(
    name: str, spec: dict[str, Any], value: str | list[str]
) -> None:
    """
    A value chosen at run time must fit its input's kind: a number, a
    boolean, one of the approved options, or text of :data:`SAFE_VALUE`
    characters that does not start with ``-``.

    Raises:
        PolicyRefusedError: It does not.
    """
    kind = str(spec.get("kind", "string"))
    if isinstance(value, list) != (kind in LIST_KINDS):
        shape = "a list" if kind in LIST_KINDS else "one value"
        raise PolicyRefusedError(f"{name!r} must be {shape}")
    for item in value if isinstance(value, list) else [value]:
        if kind == "number":
            ok = bool(NUMBER.fullmatch(item))
        elif kind == "boolean":
            ok = item in (
                "true",
                "false",
                "1",
                "0",
                "on",
                "off",
                *(spec.get("options") or ()),
            )
        elif kind in ("choice", "multichoice"):
            ok = item in (spec.get("options") or ())
        else:
            ok = bool(SAFE_VALUE.fullmatch(item)) and not item.startswith("-")
        if not ok:
            raise PolicyRefusedError(
                f"{name!r} is not a valid {kind} value: {item!r}"
            )


def _check_env(task: dict[str, Any], env: Mapping[str, str]) -> None:
    for name in env:
        if name == SIDE_ENV:
            continue
        if name not in task["env"] or RESERVED_ENV.fullmatch(name):
            raise PolicyRefusedError(f"Environment {name!r} is not approved")


# --- Slurm -----------------------------------------------------------------


def _slurm_time(value: str | None) -> str | None:
    """A suffixed duration (``2h``) in Slurm's ``[D-]HH:MM:SS``."""
    if value is None:
        return None
    text = value.strip()
    unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(text[-1:].lower())
    try:
        total = round(float(text[:-1]) * unit) if unit else None
    except ValueError:
        total = None
    if total is None:
        return text
    days, rest = divmod(total, 86400)
    clock = f"{rest // 3600:02d}:{rest % 3600 // 60:02d}:{rest % 60:02d}"
    return f"{days}-{clock}" if days else clock


def sbatch_script(prepared: Prepared) -> str:
    """
    The batch script of an approved Slurm task, from the manifest's Slurm
    fields and the task's resources (explicit fields win), as horus-slurm
    writes it.

    Raises:
        PolicyRefusedError: A value would break out of its ``#SBATCH`` line.
    """
    slurm = prepared.task["target"]["slurm"]
    res = slurm.get("resources") or {}
    gpus = int(res.get("gpus") or 0)
    options: list[tuple[str, Any]] = [
        ("--job-name", f"horus-{prepared.job_dir.name}"),
        ("--output", prepared.job_dir / "stdout.log"),
        ("--error", prepared.job_dir / "stderr.log"),
        ("--nodes", slurm.get("nodes", 1)),
        ("--ntasks", slurm.get("ntasks", 1)),
        (
            "--cpus-per-task",
            slurm.get("cpus_per_task") or res.get("cpus") or 1,
        ),
        ("--partition", slurm.get("partition")),
        ("--account", slurm.get("account")),
        ("--qos", slurm.get("qos")),
        (
            "--mem",
            slurm.get("mem")
            or (f"{res['memory_gb']}G" if res.get("memory_gb") else None),
        ),
        (
            "--time",
            _slurm_time(slurm.get("time_limit") or res.get("walltime")),
        ),
        ("--gres", slurm.get("gres") or (f"gpu:{gpus}" if gpus else None)),
    ]
    lines = ["#!/bin/bash"]
    for option, value in options:
        if value is None:
            continue
        if _SBATCH_BAD.search(str(value)):
            raise PolicyRefusedError(f"Invalid value for {option}")
        lines.append(f"#SBATCH {option}={value}")
    lines += [f"#SBATCH {arg}" for arg in slurm.get("extra_sbatch_args", [])]
    exit_path = shlex.quote(str(prepared.job_dir / "exit_code"))
    lines += [
        "",
        f"cd {shlex.quote(str(prepared.cwd))} || exit 1",
        f"( {prepared.shell} ); code=$?; echo $code > {exit_path}; exit $code",
    ]
    return "\n".join(lines) + "\n"


def parse_job_id(out: bytes) -> str:
    """The job id ``sbatch --parsable`` printed."""
    raw = out.decode(errors="replace").strip().split(";", 1)[0]
    if not JOB_ID.fullmatch(raw):
        raise RuntimeError(f"sbatch returned an unexpected job id: {raw!r}")
    return raw


# --- resource samples -------------------------------------------------------

SAMPLES_FILE = "resource-usage.csv"
RUSAGE_FILE = ".horus_resource.txt"
SAMPLES_HEADER = (
    "epoch,rss_kb,cpu_s,gpu_mem_mb,gpu_util_pct,io_read_b,io_write_b,"
    "attribution"
)
SAMPLE_S = 1.0
_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PAGE_KB = (
    os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
) // 1024


def group_usage(pgid: int, proc: Path = Path("/proc")) -> tuple[int, float]:
    """``(rss_kb, cpu_s)`` summed over process group *pgid*, from /proc."""
    rss = 0
    cpu = 0.0
    for entry in proc.iterdir() if proc.is_dir() else ():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
        except OSError:
            continue
        fields = raw[raw.rfind(")") + 2 :].split()
        # fields[0] is state (stat field 3): pgrp is 5, utime 14, rss 24.
        if int(fields[2]) != pgid:
            continue
        cpu += (int(fields[11]) + int(fields[12])) / _TICKS
        rss += int(fields[21]) * _PAGE_KB
    return rss, cpu


async def sample(
    pgid: int,
    prepared: Prepared,
    interval: float = SAMPLE_S,
    proc: Path = Path("/proc"),
) -> None:
    """
    Append the job's RSS and CPU time to ``resource-usage.csv`` in its side
    artifacts dir until it exits, then write ``.horus_resource.txt`` (GNU
    ``time -v`` lines) in its working dir: the files horus-resource-monitor
    reads back, so tasks keep their CPU/RAM charts.
    """
    prepared.side_dir.mkdir(parents=True, exist_ok=True)
    samples = prepared.side_dir / SAMPLES_FILE
    if not samples.exists():
        with open_file(samples, "w") as fh:
            fh.write(SAMPLES_HEADER + "\n")
    started = time.time()
    peak, cpu = 0, 0.0
    while not (prepared.job_dir / "exit_code").exists():
        rss, used = group_usage(pgid, proc)
        if not rss:
            break  # the group is gone (killed, so no exit code comes)
        peak, cpu = max(peak, rss), max(cpu, used)
        with open_file(samples, "a") as fh:
            fh.write(f"{time.time():.3f},{rss},{used:.2f},,,,,exact\n")
        await asyncio.sleep(interval)
    wall = max(time.time() - started, 1e-6)
    minutes, seconds = divmod(wall, 60)
    with open_file(prepared.cwd / RUSAGE_FILE, "w") as fh:
        fh.write(
            f"\tUser time (seconds): {cpu:.2f}\n"
            "\tSystem time (seconds): 0.00\n"
            f"\tPercent of CPU this job got: {cpu / wall * 100:.0f}%\n"
            "\tElapsed (wall clock) time (h:mm:ss or m:ss): "
            f"{int(minutes)}:{seconds:05.2f}\n"
            f"\tMaximum resident set size (kbytes): {peak}\n"
        )


def start_sampler(pgid: int, prepared: Prepared) -> "asyncio.Task[None]":
    """Run :func:`sample` in the background; errors are dropped."""

    async def _quiet() -> None:
        with contextlib.suppress(Exception):
            await sample(pgid, prepared)

    return asyncio.create_task(_quiet())
