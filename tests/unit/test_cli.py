#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
Config file, the CLI and the agent loop's resilience.
"""

import asyncio
import random
import socket
import stat
from pathlib import Path

import pytest

from horus_agent import config
from horus_agent.cli import main
from horus_agent.config import AgentConfig
from horus_agent.daemon import Agent, AgentUnauthorizedError
from horus_agent.target import AgentTarget
from tests.fake_broker import REGISTRATION_TOKEN, FakeBroker


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cfg" / "agent.json"
    monkeypatch.setenv(config.CONFIG_ENV, str(path))
    # The default work dir is under $HOME; keep it out of the real one.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return path


def test_config_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "dir" / "agent.json"
    path.parent.mkdir()
    path.write_text("{}")
    path.chmod(0o644)
    saved = AgentConfig(url="u", agent_id="i", name="n", token="t")
    config.save(saved, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert config.load(path) == saved


def test_config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(config.CONFIG_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config.config_path() == tmp_path / "horus-agent" / "agent.json"
    monkeypatch.setenv(config.CONFIG_ENV, "/etc/a.json")
    assert config.config_path() == Path("/etc/a.json")


def test_register_writes_config(
    broker: FakeBroker,
    config_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "register",
            "--url",
            broker.url,
            "--token",
            REGISTRATION_TOKEN,
            "--name",
            "box",
            "--work-dir",
            str(config_file.parent / "work"),
        ]
    )
    assert code == 0
    saved = config.load(config_file)
    assert saved.name == "box"
    assert saved.token.startswith("tca_")
    assert saved.token not in capsys.readouterr().out
    assert saved.work_dir == str(config_file.parent / "work")
    assert Path(saved.work_dir).is_dir()
    assert broker.work_dirs["box"] == saved.work_dir


def test_register_defaults_to_hostname_and_home(
    broker: FakeBroker, config_file: Path, tmp_path: Path
) -> None:
    del config_file
    assert (
        main(["register", "--url", broker.url, "--token", REGISTRATION_TOKEN])
        == 0
    )
    name, work_dir = next(iter(broker.work_dirs.items()))
    assert name == socket.gethostname()
    assert work_dir == str(tmp_path / "home" / "horus-agent")


def test_register_with_bad_token(
    broker: FakeBroker,
    config_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["register", "--url", broker.url, "--token", "nope"])
    assert code == 1
    assert "Invalid or used token" in capsys.readouterr().err
    assert not config_file.exists()


def test_run_without_registration(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del config_file
    assert main(["run"]) == 1
    assert "horus-agent register" in capsys.readouterr().err


def test_run_with_revoked_token(
    broker: FakeBroker,
    config_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config.save(
        AgentConfig(url=broker.url, agent_id="i", name="n", token="revoked"),
        config_file,
    )
    assert main(["run"]) == 1
    assert "register it again" in capsys.readouterr().err


async def test_agent_survives_broker_errors(
    broker: FakeBroker,
    target: AgentTarget,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random, "uniform", lambda *_: 0.01)
    broker.fail_polls = 1
    broker.fail_results = 1
    agent = Agent(
        AgentConfig(url=broker.url, agent_id="a", name="lab", token="tca_x")
    )
    broker.add_agent("lab", "tca_x")
    stop = asyncio.Event()
    serving = asyncio.create_task(agent.serve(stop))
    await asyncio.sleep(0.2)
    await target.mkdir(str(tmp_path / "ok"))
    stop.set()
    await asyncio.wait_for(serving, 10)
    assert (tmp_path / "ok").is_dir()
    assert broker.fail_polls == broker.fail_results == 0


async def test_unauthorized_agent_stops(broker: FakeBroker) -> None:
    agent = Agent(
        AgentConfig(url=broker.url, agent_id="a", name="x", token="bad")
    )
    with pytest.raises(AgentUnauthorizedError):
        await asyncio.wait_for(agent.serve(asyncio.Event()), 10)


async def test_upload_refuses_files_over_the_single_put_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from horus_agent import blobs

    monkeypatch.setattr(blobs, "MAX_SINGLE_PUT", 3)
    source = tmp_path / "big"
    source.write_bytes(b"1234")
    with pytest.raises(ValueError, match="over 5 GiB"):
        await blobs.upload("http://unused", source)
