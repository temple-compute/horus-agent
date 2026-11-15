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

import certifi
import pytest
from horus_runtime.logging import horus_logger

from horus_agent import config, logs
from horus_agent.cli import main
from horus_agent.config import AgentConfig
from horus_agent.daemon import (
    Agent,
    AgentUnauthorizedError,
    broker_tls,
    check_broker_url,
)
from horus_agent.protocol import Op, OpResult
from horus_agent.target import AgentTarget
from tests.fake_broker import REGISTRATION_TOKEN, FakeBroker


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cfg" / "agent.json"
    monkeypatch.setenv(config.CONFIG_ENV, str(path))
    # The default work dir is under $HOME; keep it out of the real one.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return path


@pytest.fixture
def same_user() -> None:
    """A policy letting tasks run as the agent's own user (tests only)."""
    config.POLICY_PATH.write_text("allow_same_user = true\n")


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


@pytest.mark.usefixtures("same_user")
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


@pytest.mark.usefixtures("same_user")
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


@pytest.mark.usefixtures("same_user")
def test_run_restricts_umask_and_names_dropped_variables(
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
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s3cr3t-value")
    monkeypatch.setenv("MODULEPATH", "/opt/modules")
    lines: list[str] = []
    monkeypatch.setattr(logs, "setup", lambda _debug=False: False)
    sink = horus_logger.log.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        assert main(["run"]) == 1
    finally:
        horus_logger.log.remove(sink)
    dropped = next(line for line in lines if "do not get" in line)
    assert "AWS_SECRET_ACCESS_KEY" in dropped
    assert "s3cr3t-value" not in dropped
    assert "MODULEPATH" not in dropped
    assert stat.S_IMODE(work.stat().st_mode) == 0o750
    assert os.umask(0o077) == 0o077


def test_run_refuses_a_read_only_work_dir(
    config_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work = tmp_path / "work"
    work.mkdir(mode=0o500)
    config.save(
        AgentConfig(
            url="https://b",
            agent_id="i",
            name="n",
            token="t",
            work_dir=str(work),
        ),
        config_file,
    )
    assert main(["run"]) == 1
    assert "ReadWritePaths" in capsys.readouterr().err


def test_config_without_new_fields_loads(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text('{"url": "u", "agent_id": "i", "name": "n", "token": "t"}')
    loaded = config.load(path)
    assert (loaded.ca_bundle, loaded.job_env) == ("", ())
    saved = AgentConfig(
        url="u", agent_id="i", name="n", token="t", job_env=("A", "B")
    )
    config.save(saved, path)
    assert config.load(path) == saved


def test_config_parents_are_private(tmp_path: Path) -> None:
    os.umask(0o007)
    path = tmp_path / "home" / ".config" / "horus-agent" / "agent.json"
    config.save(AgentConfig(url="u", agent_id="i", name="n", token="t"), path)
    for directory in (path.parent, path.parent.parent, tmp_path / "home"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700, directory
    assert os.umask(0o007) == 0o007


@pytest.fixture
def policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "etc" / "policy.toml"
    path.parent.mkdir(mode=0o755)
    monkeypatch.setattr(config, "POLICY_PATH", path)
    monkeypatch.setattr(config, "POLICY_OWNERS", frozenset({0}))
    return path


def test_jobs_policy_must_be_root_owned(
    policy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert config.load_jobs_policy(policy) == config.JobsPolicy()
    policy.write_text('[jobs]\nrun_as = "horus-job"\nrun_as_via = "setpriv"\n')
    policy.chmod(0o644)
    with pytest.raises(ValueError, match="owned by root"):
        config.load_jobs_policy(policy)
    monkeypatch.setattr(config, "POLICY_OWNERS", {0, os.getuid()})
    assert config.load_jobs_policy(policy) == config.JobsPolicy(
        "horus-job", "setpriv"
    )
    policy.parent.chmod(0o775)
    with pytest.raises(ValueError, match=str(policy.parent)):
        config.load_jobs_policy(policy)


@pytest.mark.usefixtures("stdin_token")
def test_register_and_run_check_the_jobs_policy(
    policy: Path,
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy.write_text('[jobs]\nrun_as = "no-such-user-x"\n')
    assert main(["register", "--url", "https://b"]) == 1
    assert "owned by root" in capsys.readouterr().err
    monkeypatch.setattr(config, "POLICY_OWNERS", {0, os.getuid()})
    assert main(["register", "--url", "https://b"]) == 1
    assert "no-such-user-x" in capsys.readouterr().err
    assert not config_file.exists()
    config.save(
        AgentConfig(url="https://b", agent_id="i", name="n", token="t"),
        config_file,
    )
    assert main(["run"]) == 1
    assert "no-such-user-x" in capsys.readouterr().err


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


def test_broker_tls_keeps_ssl_cert_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "from-env.pem"))
    with pytest.raises(OSError):
        broker_tls()  # the env file is used, and is missing here
    assert broker_tls(certifi.where()).verify_mode == ssl.CERT_REQUIRED


async def test_control_ops_never_wait_for_a_slot(tmp_path: Path) -> None:
    agent = Agent(
        AgentConfig(url="https://b", agent_id="a", name="x", token="t")
    )
    agent._slots = asyncio.Semaphore(0)  # every slot taken
    results: asyncio.Queue[OpResult] = asyncio.Queue()
    control = Op(op_id="a", method="path_exists", args={"path": "/"})
    await asyncio.wait_for(agent._run_one(control, results), 5)
    work = Op(op_id="b", method="mkdir", args={"path": str(tmp_path / "d")})
    waiting = asyncio.create_task(agent._run_one(work, results))
    await asyncio.sleep(0.2)
    assert not waiting.done()
    waiting.cancel()


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


def test_strict_run_needs_a_task_user(
    config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config.save(
        AgentConfig(url="https://b", agent_id="i", name="n", token="t"),
        config_file,
    )
    assert main(["run"]) == 1
    assert "run_as" in capsys.readouterr().err


def test_policy_init(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["policy", "init", "--run-as", "nobody"]) == 0
    text = config.POLICY_PATH.read_text()
    assert 'mode = "strict"' in text and 'run_as = "nobody"' in text
    assert config.POLICY_PATH.stat().st_mode & 0o777 == 0o644
    assert main(["policy", "init", "--open"]) == 1
    assert "exists" in capsys.readouterr().err
    config.POLICY_PATH.unlink()
    assert main(["policy", "init", "--open", "--run-as", ""]) == 0
    assert 'mode = "open"' in config.POLICY_PATH.read_text()


def test_approve_by_id_names_the_missing_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(config.CONFIG_ENV, str(tmp_path / "none.json"))
    assert main(["approve", "some-release"]) == 1
    assert config.CONFIG_ENV in capsys.readouterr().err
