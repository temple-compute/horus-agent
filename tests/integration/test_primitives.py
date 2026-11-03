#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
Every channel primitive, from AgentTarget through the broker to the agent.
"""

import asyncio
import signal
from pathlib import Path

import pytest
from horus_runtime.core.task.exceptions import (
    TargetLostError,
    TaskExecutionError,
)

from horus_agent import ops
from horus_agent import target as target_module
from horus_agent.daemon import Agent
from horus_agent.target import AgentTarget
from tests.fake_broker import FakeBroker

pytestmark = pytest.mark.usefixtures("agent")


@pytest.fixture
def small_inline(monkeypatch: pytest.MonkeyPatch) -> int:
    """Shrink the inline limit so staging and chunking paths run."""
    limit = 1024 * 1024
    monkeypatch.setattr(target_module, "INLINE_LIMIT", limit)
    monkeypatch.setattr(ops, "INLINE_LIMIT", limit)
    return limit


async def test_files_and_dirs(target: AgentTarget, tmp_path: Path) -> None:
    base = tmp_path / "agent"
    await target.mkdir(f"{base}/a/b")
    await target.put_file(b"hello", f"{base}/a/b/f.txt")
    assert (base / "a/b/f.txt").read_bytes() == b"hello"
    assert await target.get_file(f"{base}/a/b/f.txt") == b"hello"
    assert await target.path_exists(f"{base}/a/b/f.txt")
    assert not await target.path_exists(f"{base}/missing")

    entries = await target.list_dir(f"{base}/a")
    assert [(e.name, e.is_dir) for e in entries] == [("b", True)]

    local = tmp_path / "copy.txt"
    await target.get_file_to(f"{base}/a/b/f.txt", local)
    assert local.read_bytes() == b"hello"

    source = tmp_path / "src.txt"
    source.write_bytes(b"from path")
    await target.put_file(source, f"{base}/p.txt")
    assert (base / "p.txt").read_bytes() == b"from path"

    await target.remove(f"{base}/a")
    assert not (base / "a").exists()


async def test_large_files_use_staging(
    target: AgentTarget,
    broker: FakeBroker,
    tmp_path: Path,
    small_inline: int,
) -> None:
    base = tmp_path / "agent"
    payload = bytes(range(256)) * (small_inline // 256 * 3)
    source = tmp_path / "big.bin"
    source.write_bytes(payload)

    await target.put_file(source, f"{base}/big.bin")
    assert (base / "big.bin").read_bytes() == payload
    assert await target.get_file(f"{base}/big.bin") == payload
    local = tmp_path / "back.bin"
    await target.get_file_to(f"{base}/big.bin", local)
    assert local.read_bytes() == payload
    assert broker.blobs, "large transfers should go through staging"

    chunks = [
        chunk
        async for chunk in target.read_range(
            f"{base}/big.bin", small_inline, small_inline + 10
        )
    ]
    assert [len(c) for c in chunks] == [small_inline, 10]
    assert b"".join(chunks) == payload[small_inline : 2 * small_inline + 10]


async def test_read_range_rejects_unaligned_offset(
    target: AgentTarget,
) -> None:
    with pytest.raises(ValueError, match="multiple of 1 MiB"):
        async for _ in target.read_range("/x", 3, 1):
            pass


async def test_sync_command(target: AgentTarget, tmp_path: Path) -> None:
    proc = await target.run_command(
        "echo out; echo err >&2; exit 3",
        cwd=str(tmp_path),
        env={"GREETING": "hi"},
        detach=False,
    )
    assert await proc.wait() == 3
    assert proc.returncode == 3
    assert await proc.communicate() == (b"out\n", b"err\n")
    assert [line async for line in proc.stream()] == [
        ("stdout", b"out\n"),
        ("stderr", b"err\n"),
    ]
    proc.kill()
    proc.signal(signal.SIGTERM)


async def test_detached_command_streams_logs(
    target: AgentTarget, tmp_path: Path, small_inline: int
) -> None:
    (tmp_path / "agent").mkdir()
    # More output than one read_output op carries, so the target has to
    # come back for the rest.
    size = small_inline
    proc = await target.run_command(
        f"head -c {size} /dev/zero | tr '\\0' 'x'; echo; echo $GREETING",
        cwd=str(tmp_path / "agent"),
        env={"GREETING": "hi"},
    )
    out, err = await proc.communicate()
    assert proc.returncode == 0
    assert out == b"x" * size + b"\nhi\n"
    assert err == b""


async def test_signal_stops_detached_job(
    target: AgentTarget, tmp_path: Path
) -> None:
    (tmp_path / "agent").mkdir()
    proc = await target.run_command("sleep 30", cwd=str(tmp_path / "agent"))
    proc.signal(signal.SIGKILL)
    assert await asyncio.wait_for(proc.wait(), 20) != 0


async def test_agent_error_is_raised(target: AgentTarget) -> None:
    with pytest.raises(TaskExecutionError, match="FileNotFoundError"):
        await target.get_file("/definitely/not/here")
    with pytest.raises(TaskExecutionError, match="must be absolute"):
        await target.mkdir("relative/path")


async def test_unknown_agent_is_refused(
    target: AgentTarget,
) -> None:
    other = target.model_copy(update={"agent": "nobody"})
    with pytest.raises(TaskExecutionError, match="No agent named"):
        await other.mkdir("/tmp/x")


async def test_offline_agent_is_lost(
    broker: FakeBroker, target: AgentTarget
) -> None:
    broker.add_agent("sleeper", "tca_sleeper")
    other = target.model_copy(update={"agent": "sleeper"})
    with pytest.raises(TargetLostError, match="last seen: never"):
        await other.mkdir("/tmp/x")


async def test_redelivered_op_is_not_rerun(
    agent: Agent, tmp_path: Path
) -> None:
    from horus_agent.protocol import Op

    marker = tmp_path / "count"
    op = Op(
        op_id="once",
        method="exec",
        args={"cmd": f"echo x >> {marker}"},
    )
    first, second = await asyncio.gather(
        agent.runner.run(op), agent.runner.run(op)
    )
    third = await agent.runner.run(op)
    assert first == second == third
    assert marker.read_text() == "x\n"


async def test_large_sync_output_comes_back_whole(
    target: AgentTarget,
    tmp_path: Path,
    small_inline: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    spill = tmp_path / "spill"
    spill.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spill))
    proc = await target.run_command(
        f"head -c {small_inline + 5} /dev/zero | tr '\\0' 'y'",
        cwd=str(tmp_path),
        detach=False,
    )
    out, _ = await proc.communicate()
    assert out == b"y" * (small_inline + 5)
    assert list(spill.iterdir()) == [], "the spilled output is removed"


async def test_finished_job_logs_are_released(
    target: AgentTarget, tmp_path: Path
) -> None:
    (tmp_path / "agent").mkdir()
    proc = await target.run_command("echo done", cwd=str(tmp_path / "agent"))
    assert await proc.communicate() == (b"done\n", b"")
    assert target._logs == {}


async def test_refused_result_batches_are_split(
    broker: FakeBroker, target: AgentTarget, tmp_path: Path
) -> None:
    broker.max_results_per_post = 1
    made = [str(tmp_path / f"d{i}") for i in range(4)]
    await asyncio.gather(*(target.mkdir(path) for path in made))
    assert all(Path(path).is_dir() for path in made)
