#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
OpRunner, the agent-side dispatcher, without a broker.
"""

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from horus_agent import ops
from horus_agent.ops import OpRunner
from horus_agent.protocol import Op


async def run(runner: OpRunner, method: str, **args: Any) -> Any:
    result = await runner.run(
        Op(op_id=uuid4().hex, method=method, args=args, run_id="r")
    )
    if result.error is not None:
        raise RuntimeError(f"{result.error.type}: {result.error.message}")
    return result.result


async def test_unknown_op_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="Unknown op"):
        await run(OpRunner(), "teleport")


async def test_bad_env_name_is_refused() -> None:
    with pytest.raises(RuntimeError, match="Invalid environment variable"):
        await run(OpRunner(), "exec", cmd="true", env={"A;rm -rf ~": "x"})


async def test_root_is_never_removed() -> None:
    with pytest.raises(RuntimeError, match="Refusing to remove /"):
        await run(OpRunner(), "remove", path="/")


async def test_large_command_output_spills_to_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ops, "INLINE_LIMIT", 4)
    result = await run(OpRunner(), "exec", cmd="echo too-long; echo e >&2")
    path = Path(result["stdout_file"])
    try:
        assert path.read_bytes() == b"too-long\n"
        assert result["stderr"] == "ZQo="  # b"e\n", still inline
    finally:
        path.unlink()


async def test_large_file_reports_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ops, "INLINE_LIMIT", 4)
    (tmp_path / "f").write_bytes(b"12345")
    assert await run(OpRunner(), "get_file", path=str(tmp_path / "f")) == {
        "size": 5
    }


async def test_launch_is_idempotent(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "pid").write_text("4242\n")
    result = await run(OpRunner(), "launch", cmd="false", job_dir=str(job))
    assert result == {"pid": 4242}
    assert not (job / "exit_code").exists(), "nothing was started"


async def test_cancel_run_kills_its_jobs(tmp_path: Path) -> None:
    runner = OpRunner()
    job = str(tmp_path / "job")
    launched = await run(runner, "launch", cmd="sleep 30", job_dir=job)
    assert await run(runner, "cancel_run") == 1
    for _ in range(50):
        status = await run(runner, "poll", pid=launched["pid"], job_dir=job)
        if status is not None:
            break
        await asyncio.sleep(0.1)
    assert status not in (None, 0)
    assert await run(runner, "cancel_run") == 0


async def test_cancel_run_skips_finished_jobs(tmp_path: Path) -> None:
    runner = OpRunner()
    job = str(tmp_path / "job")
    launched = await run(runner, "launch", cmd="true", job_dir=job)
    for _ in range(50):
        if (tmp_path / "job" / "exit_code").exists():
            break
        await asyncio.sleep(0.05)
    assert await run(runner, "cancel_run") == 0
    del launched


async def test_answers_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "RESULT_CACHE_SIZE", 1)
    runner = OpRunner()
    for op_id in ("a", "b"):
        await runner.run(
            Op(op_id=op_id, method="path_exists", args={"path": "/"})
        )
    assert list(runner._answered) == ["b"]
