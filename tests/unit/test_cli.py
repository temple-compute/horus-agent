#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Config file, the CLI and the agent loop's resilience.
"""

import asyncio
import io
import os
import random
import socket
import ssl
import stat
from pathlib import Path

import pytest

from horus_agent import config
from horus_agent.cli import main
from horus_agent.config import AgentConfig
from horus_agent.daemon import (
    Agent,
    AgentUnauthorizedError,
    broker_tls,
    check_broker_url,
)
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


@pytest.fixture
def stdin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registration token piped in."""
    monkeypatch.delenv("HORUS_AGENT_TOKEN", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{REGISTRATION_TOKEN}\n"))


@pytest.mark.usefixtures("stdin_token")
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
    assert stat.S_IMODE(Path(saved.work_dir).stat().st_mode) == 0o700
    assert broker.work_dirs["box"] == saved.work_dir


def test_register_defaults_to_hostname_and_home(
    broker: FakeBroker,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del config_file
    monkeypatch.setenv("HORUS_AGENT_TOKEN", REGISTRATION_TOKEN)
    assert main(["register", "--url", broker.url]) == 0
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


def test_register_token_on_the_command_line_still_works(
    broker: FakeBroker,
    config_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["register", "--url", broker.url, "--token", REGISTRATION_TOKEN]
    assert main(argv) == 0
    assert "visible to other users" in capsys.readouterr().err
    assert config_file.exists()


@pytest.mark.usefixtures("stdin_token")
@pytest.mark.parametrize(
    "url", ["http://broker.example/api", "ftp://x", "https://"]
)
def test_register_refuses_plain_http(
    url: str, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["register", "--url", url]) == 1
    assert "https://" in capsys.readouterr().err
    assert not config_file.exists()


def test_run_refuses_plain_http(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(
        AgentConfig(url="http://b.example", agent_id="i", name="n", token="t"),
        config_file,
    )
    assert main(["run"]) == 1
    assert "https://" in capsys.readouterr().err


def test_refuses_root_unless_allowed(
    broker: FakeBroker,
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    assert main(["run"]) == 1
    assert "Refusing to run as root" in capsys.readouterr().err
    config.save(
        AgentConfig(url=broker.url, agent_id="i", name="n", token="revoked"),
        config_file,
    )
    assert main(["run", "--allow-root"]) == 1
    assert "register it again" in capsys.readouterr().err


def test_run_refuses_a_config_others_can_read(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(
        AgentConfig(url="https://b", agent_id="i", name="n", token="t"),
        config_file,
    )
    config_file.chmod(0o640)
    assert main(["run"]) == 1
    assert "chmod 600" in capsys.readouterr().err


def test_run_restricts_umask_and_environment(
    broker: FakeBroker,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = tmp_path / "work"
    work.mkdir(mode=0o755)
    config.save(
        AgentConfig(
            url=broker.url,
            agent_id="i",
            name="n",
            token="revoked",
            work_dir=str(work),
            job_env=("MODULEPATH",),
        ),
        config_file,
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s3cr3t")
    monkeypatch.setenv("MODULEPATH", "/opt/modules")
    assert main(["run"]) == 1
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ
    assert config.CONFIG_ENV not in os.environ
    assert os.environ["MODULEPATH"] == "/opt/modules"
    assert stat.S_IMODE(work.stat().st_mode) == 0o750
    assert os.umask(0o077) == 0o077


def test_config_without_new_fields_loads(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text('{"url": "u", "agent_id": "i", "name": "n", "token": "t"}')
    loaded = config.load(path)
    assert (loaded.run_as, loaded.run_as_via, loaded.job_env) == (
        "",
        "sudo",
        (),
    )
    saved = AgentConfig(
        url="u", agent_id="i", name="n", token="t", job_env=("A", "B")
    )
    config.save(saved, path)
    assert config.load(path) == saved


@pytest.mark.usefixtures("stdin_token")
def test_register_rejects_unknown_run_as_user(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["register", "--url", "https://b", "--run-as", "no-such-user-x"]
    assert main(argv) == 1
    assert "no-such-user-x" in capsys.readouterr().err
    assert not config_file.exists()


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


async def test_agent_skips_malformed_ops(
    broker: FakeBroker, agent: Agent, target: AgentTarget, tmp_path: Path
) -> None:
    del agent
    broker.enqueue("lab", {"op_id": "x", "method": 7})
    await target.mkdir(str(tmp_path / "still-serving"))
    assert (tmp_path / "still-serving").is_dir()


def test_broker_url_and_tls(tmp_path: Path) -> None:
    for url in (
        "https://api.example/v1",
        "http://127.0.0.1:8000",
        "http://[::1]",
    ):
        check_broker_url(url)
    for url in ("http://api.example", "https:///x", "api.example"):
        with pytest.raises(ValueError, match="https://"):
            check_broker_url(url)
    assert broker_tls().verify_mode == ssl.CERT_REQUIRED
    with pytest.raises(OSError):
        broker_tls(str(tmp_path / "missing.pem"))


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
