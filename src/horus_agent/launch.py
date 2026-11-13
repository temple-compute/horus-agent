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
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from horus_agent import native
from horus_agent.blobs import open_file
from horus_agent.guard import Admitted, Guard
from horus_agent.manifest import RESERVED_ENV
from horus_agent.protocol import PolicyRefusedError, TaskLaunchArgs
from horus_agent.template import placeholders, render

SIDE_ENV = "HORUS_SIDE_ARTIFACTS_DIR"
_SBATCH_BAD = re.compile(r"[\s#'\"\\\x00]")
JOB_ID = re.compile(r"\d+(_\d+)?")


@dataclass
class Prepared:
    """A verified task command, ready to start."""

    command: str
    """The rendered command."""
    env: dict[str, str]
    """The task's environment; reaches it through stdin, not a file."""
    cwd: Path
    job_dir: Path
    side_dir: Path
    task: dict[str, Any]


def _template(task: dict[str, Any], script: Path | None) -> str:
    runtime = task["runtime"]
    if runtime["kind"] == "command":
        return str(runtime["command"])
    if script is None:
        raise PolicyRefusedError("A python_script task needs its script")
    # The runtime does not quote the interpreter, only the script path.
    head = f"{runtime['python']} {shlex.quote(str(script))}"
    return f"{head} {runtime['args']}".rstrip()


def prepare(
    guard: Guard, run_id: str | None, admitted: Admitted, *, slurm: bool
) -> Prepared:
    """
    Check an admitted ``task_launch`` / ``slurm_submit`` against its manifest
    task and render the command.

    Raises:
        PolicyRefusedError: Anything does not match the approved release.
    """
    args, task, paths = admitted.args, admitted.task, admitted.paths
    if not isinstance(args, TaskLaunchArgs) or task is None:
        raise PolicyRefusedError("Not a task launch of an approved release")
    task_id = args.task_id
    chain = task["target"]["chain"]
    if (chain[0] == "slurm") != slurm:
        method = "slurm_submit" if chain[0] == "slurm" else "task_launch"
        raise PolicyRefusedError(f"Task {task_id!r} runs through {method}")
    script = None
    if task["runtime"]["kind"] == "python_script":
        if args.script is None:
            raise PolicyRefusedError("A python_script task needs its script")
        script = guard.confine(run_id, args.script)
        if native.digest(script) != task["runtime"]["script_sha256"]:
            raise PolicyRefusedError("The script is not the approved one")
        guard.seal(run_id, script)
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
        values[name] = _bind(guard, run_id, name, task, binding)
    _check_env(task, args.env)
    side = Path(args.env.get(SIDE_ENV) or paths["cwd"] / "side-artifacts")
    if SIDE_ENV in args.env:
        side = guard.confine(run_id, args.env[SIDE_ENV])
    job_dir = paths["job_dir"]
    job_dir.mkdir(parents=True, exist_ok=True)
    command = render(template, values)
    return Prepared(command, dict(args.env), paths["cwd"], job_dir, side, task)


def _bind(
    guard: Guard,
    run_id: str | None,
    name: str,
    task: dict[str, Any],
    binding: Any,
) -> str | list[str]:
    spec = task["inputs"].get(name)
    role = "output" if spec is None else spec["role"]
    if binding.path is not None:
        if role == "param":
            return str(guard.confine_param(run_id, binding.path))
        path = guard.confine(run_id, binding.path)
        if role == "constant":
            expected = spec.get("sha256") if spec else None
            if expected is None:
                raise PolicyRefusedError(f"{name!r} must be a value")
            try:
                actual = native.digest(path)
            except FileNotFoundError:
                actual = ""
            if actual != expected:
                raise PolicyRefusedError(
                    f"{name!r} is not the approved file or folder"
                )
            guard.seal(run_id, path)
        return str(path)
    if role == "param":
        value: str | list[str] = binding.value
        return value
    if role == "constant" and spec is not None:
        if binding.value != spec.get("value"):
            raise PolicyRefusedError(f"{name!r} is not the approved value")
        approved: str | list[str] = spec["value"]
        return approved
    raise PolicyRefusedError(f"{name!r} must be a path")


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
        f"( {prepared.command} ); code=$?; echo $code > {exit_path}; "
        "exit $code",
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
        if rss or used:
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
