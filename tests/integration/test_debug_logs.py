#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
Debug logs: everything the agent does, nothing secret.
"""

import logging
from collections.abc import Generator
from pathlib import Path

import pytest
from horus_runtime.logging import horus_logger

from horus_agent import logs
from horus_agent.cli import main
from horus_agent.target import AgentTarget

SIGNED = "https://s3.example/b/k?X-Amz-Signature=abc123&X-Amz-Expires=600"


@pytest.fixture
def debug_lines() -> Generator[list[str]]:
    """Every debug-level line logged during the test."""
    lines: list[str] = []
    logs.setup(True)
    horus_logger.log.add(lambda m: lines.append(str(m)), level="DEBUG")
    try:
        yield lines
    finally:
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).handlers = []
        horus_logger.setup(level="INFO")


@pytest.mark.usefixtures("agent")
async def test_ops_and_commands_are_logged_without_secrets(
    target: AgentTarget, tmp_path: Path, debug_lines: list[str]
) -> None:
    proc = await target.run_command(
        f"echo hello; echo '{SIGNED}' >/dev/null",
        cwd=str(tmp_path),
        env={"API_KEY": "s3cr3t-value"},
        detach=False,
    )
    assert await proc.wait() == 0
    await target.put_file(b"x" * 10, str(tmp_path / "f.txt"))

    text = "\n".join(debug_lines)
    for expected in (
        "Poll: ",  # polls and what they returned
        "exec start",  # the op as the agent got it
        "env=[API_KEY]",  # env names only
        "exec in ",  # the command the agent ran
        "exec exited 0",
        "stdout='hello\\n'",  # command output preview
        "put_file start",
        "data=<10 bytes>",
        "Results: delivered",
        "httpcore.connection: connect_tcp",  # connections made
        "HTTP Request: POST",  # every request
        ": submit ",  # orchestrator side
    ):
        assert expected in text, expected
    assert "s3cr3t-value" not in text
    assert "abc123" not in text
    assert "https://s3.example/b/k?<redacted>" in text


def test_safe_url_keeps_shell_question_marks() -> None:
    assert logs.safe_url("ls a?b && curl 'http://h/p?sig=1'") == (
        "ls a?b && curl 'http://h/p?<redacted>'"
    )


def test_describe_result_sizes_payloads() -> None:
    assert logs.describe_result(
        "read_output", {"stdout": "aGk=", "more": 0}
    ) == ("stdout=<2 bytes> more=0")
    assert logs.describe_result("list_dir", [1, 2]) == "2 entries"
    assert logs.describe_result("poll", 3) == "3"


def test_debug_flag_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        monkeypatch.delenv(logs.DEBUG_ENV, raising=False)
        assert logs.setup(False) is False
        monkeypatch.setenv(logs.DEBUG_ENV, "1")
        assert logs.setup(False) is True
        monkeypatch.delenv(logs.DEBUG_ENV)
        monkeypatch.setenv("HORUS_AGENT_CONFIG", "/nonexistent/agent.json")
        assert main(["run", "--debug"]) == 1
    finally:
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).handlers = []
        horus_logger.setup(level="INFO")


@pytest.fixture
def info_lines() -> Generator[list[str]]:
    """Every info-level line logged during the test (no --debug)."""
    lines: list[str] = []
    logs.setup(False)
    sink = horus_logger.log.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        yield lines
    finally:
        horus_logger.log.remove(sink)


@pytest.mark.usefixtures("agent")
async def test_info_log_names_every_op_without_secrets(
    target: AgentTarget, tmp_path: Path, info_lines: list[str]
) -> None:
    proc = await target.run_command(
        f"echo hello; curl -s '{SIGNED}' >/dev/null || true",
        cwd=str(tmp_path),
        env={"API_KEY": "s3cr3t-value"},
        detach=False,
    )
    assert await proc.wait() == 0
    await target.put_file(b"x" * 10, str(tmp_path / "f.txt"))
    job = await target.run_command("true", cwd=str(tmp_path))
    assert await job.wait() == 0

    text = "\n".join(info_lines)
    for expected in (
        "] exec echo hello; curl -s",
        "exec exited 0 in",
        f"] put_file {tmp_path / 'f.txt'} (10 bytes)",
        "] launch ",
        "launched pid ",
        "exited with 0",
    ):
        assert expected in text, expected
    # Routine job checks stay out of the info log.
    assert "] poll" not in text
    assert "] read_output" not in text
    assert "s3cr3t-value" not in text
    assert "abc123" not in text


def test_summarize_shortens_long_commands() -> None:
    line = logs.summarize("exec", {"cmd": "x" * 500})
    assert line is not None
    assert len(line) < 220
    assert logs.summarize("poll", {"pid": 1, "job_dir": "/j"}) is None
