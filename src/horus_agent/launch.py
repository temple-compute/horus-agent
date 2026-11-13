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
import re
import shlex
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horus_agent import native
from horus_agent.blobs import open_file
from horus_agent.guard import SAFE_PATH, Admitted, Guard, bad_name
from horus_agent.manifest import RESERVED_ENV, VALUE_KINDS
from horus_agent.protocol import PolicyRefusedError, TaskLaunchArgs
from horus_agent.template import placeholders, render

SIDE_ENV = "HORUS_SIDE_ARTIFACTS_DIR"
_SBATCH_BAD = re.compile(r"[\s#'\"\\\x00]")
JOB_ID = re.compile(r"\d+(_\d+)?")


SAFE_VALUE = re.compile(r"[A-Za-z0-9._,:=+@/-]*")
"""Characters a free text value chosen at run time may hold."""
NUMBER = re.compile(r"-?[0-9]+(\.[0-9]+)?")
LIST_KINDS = frozenset({"list", "multichoice"})
HOME_SUBDIR = "home"
"""Each approved job's own fresh home, inside its job dir."""
UNSET_ENV = ("NODE_PATH", "BASH_ENV", "ENV")
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


def _working_dir(guard: Guard, run_id: str | None, raw: str) -> Path:
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
    folder = cwd
    while True:
        with os.scandir(folder) as entries:
            for entry in entries:
                if bad_name(entry.name) and (
                    entry.stat(follow_symlinks=False).st_uid == os.geteuid()
                ):
                    raise PolicyRefusedError(
                        f"{entry.path} holds settings or code a tool would "
                        "read; it was not made by an approved task"
                    )
        if folder == root:
            return cwd
        folder = folder.parent


async def _verified(src: Path, store: Path, expected: str, name: str) -> Path:
    """The agent's own copy of *src*, which must be the approved one."""
    try:
        return await asyncio.to_thread(
            native.materialise, src, store, expected
        )
    except FileNotFoundError:
        raise PolicyRefusedError(f"{name!r} is missing") from None
    except ValueError as exc:
        raise PolicyRefusedError(
            f"{name!r} is not the approved file or folder: {exc}"
        ) from exc


def job_env(job_dir: Path) -> dict[str, str]:
    """
    What every approved job's environment pins: a fresh home of its own,
    and no user startup files for R or Python.
    """
    return {
        "HOME": str(job_dir / HOME_SUBDIR),
        "R_PROFILE_USER": os.devnull,
        "R_ENVIRON_USER": os.devnull,
        "PYTHONNOUSERSITE": "1",
    }


async def prepare(
    guard: Guard, run_id: str | None, admitted: Admitted, *, slurm: bool
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
    if not isinstance(args, TaskLaunchArgs) or task is None:
        raise PolicyRefusedError("Not a task launch of an approved release")
    task_id = args.task_id
    if not TASK_ID.fullmatch(task_id):
        raise PolicyRefusedError(f"Invalid task id {task_id!r}")
    chain = task["target"]["chain"]
    if (chain[0] == "slurm") != slurm:
        method = "slurm_submit" if chain[0] == "slurm" else "task_launch"
        raise PolicyRefusedError(f"Task {task_id!r} runs through {method}")
    job_dir = guard.confine_bound(run_id, args.job_dir)
    cwd = _working_dir(guard, run_id, args.cwd)
    store = guard.const_store
    script = None
    if task["runtime"]["kind"] == "python_script":
        if args.script is None:
            raise PolicyRefusedError("A python_script task needs its script")
        script = await _verified(
            guard.confine_bound(run_id, args.script),
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
    side = cwd / "side-artifacts"
    if SIDE_ENV in args.env:
        side = guard.confine_bound(run_id, args.env[SIDE_ENV])
    command = render(template, values)
    env = {**args.env, **job_env(job_dir)}
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
        path = guard.confine_bound(run_id, binding.path)
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
            ok = item in ("true", "false")
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
