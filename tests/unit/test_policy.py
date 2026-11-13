#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
The machine's policy file, approvals and the CLI that manages them.
"""

import json
import os
from pathlib import Path

import pytest

from horus_agent import cli, config, policy
from horus_agent.manifest import Manifest, build_manifest

DATA = Path(__file__).parent.parent / "data" / "manifest"
URL = "https://api.example.org/api/v1"


def write_policy(policy_dir: Path, text: str) -> Path:
    path = policy_dir / "policy.toml"
    path.write_text(text)
    path.chmod(0o644)
    return path


def test_missing_file_means_strict_defaults(tmp_path: Path) -> None:
    rules = policy.load_policy(str(tmp_path), URL)
    assert rules.strict and rules.require_release
    assert not rules.uploads and rules.logs == "off"
    assert rules.allowed_hosts == ("api.example.org",)
    assert rules.source == "defaults"
    assert rules.audit_path == tmp_path / ".horus-agent" / "ops.jsonl"


def test_file_settings(policy_dir: Path, tmp_path: Path) -> None:
    write_policy(
        policy_dir,
        f'mode = "open"\nwork_dir = "{tmp_path}"\nuploads = true\n'
        'allowed_hosts = ["s3.example.org"]\nmax_concurrent_runs = 2\n'
        f'[jobs]\nrun_as = "job"\n[data]\nread_roots = ["{tmp_path}"]\n',
    )
    rules = policy.load_policy(str(tmp_path / "sub"), URL)
    assert not rules.strict and rules.uploads
    assert rules.allowed_hosts == ("s3.example.org",)
    assert rules.jobs.run_as == "job" and rules.read_roots == (str(tmp_path),)
    assert rules.summary()["mode"] == "open"
    assert rules.describe()["max_concurrent_runs"] == 2


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('mode = "lax"', "mode"),
        ('logs = "some"', "logs"),
        ("uploads = 1", "bool"),
        ("max_put_bytes = true", "int"),
        ("surprise = 1", "unknown"),
        ("schema = 2", "schema"),
        ('work_dir = "relative"', "absolute"),
        ("mode = ", "policy.toml"),
    ],
)
def test_invalid_files(
    policy_dir: Path, tmp_path: Path, text: str, match: str
) -> None:
    write_policy(policy_dir, text)
    with pytest.raises(policy.PolicyError, match=match):
        policy.load_policy(str(tmp_path), URL)


def test_work_dir_must_sit_under_the_policys(
    policy_dir: Path, tmp_path: Path
) -> None:
    write_policy(policy_dir, f'work_dir = "{tmp_path / "srv"}"\n')
    with pytest.raises(policy.PolicyError, match="not under"):
        policy.load_policy(str(tmp_path / "home"), URL)


def test_writable_policy_is_refused(policy_dir: Path, tmp_path: Path) -> None:
    path = write_policy(policy_dir, 'mode = "open"\n')
    path.chmod(0o666)
    with pytest.raises(policy.PolicyError, match="writable"):
        policy.load_policy(str(tmp_path), URL)


def test_policy_owned_by_someone_else_is_refused(
    policy_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_policy(policy_dir, 'mode = "open"\n')
    monkeypatch.setattr(config, "POLICY_OWNERS", frozenset({0}))
    with pytest.raises(policy.PolicyError, match="owned by root"):
        policy.load_policy(str(tmp_path), URL)


def test_writable_approvals_dir_is_refused(
    policy_dir: Path, tmp_path: Path
) -> None:
    approved = policy_dir / "approved"
    approved.mkdir()
    approved.chmod(0o777)
    with pytest.raises(policy.PolicyError, match="writable"):
        policy.load_policy(str(tmp_path), URL)


def test_symlinked_policy_is_refused(policy_dir: Path, tmp_path: Path) -> None:
    real = tmp_path / "p.toml"
    real.write_text('mode = "open"\n')
    (policy_dir / "policy.toml").symlink_to(real)
    with pytest.raises(policy.PolicyError, match="symlink"):
        policy.load_policy(str(tmp_path), URL)


def test_approvals_listing_and_revoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_manifest((DATA / "basic.zip").read_bytes())
    approvals = policy.Approvals()
    assert approvals.get(manifest.release_hash) is None
    monkeypatch.setattr(policy, "APPROVALS_TTL_S", -1.0)
    path = policy.approve(manifest, "alice")
    assert oct(path.stat().st_mode & 0o777) == "0o644"
    got = approvals.get(manifest.release_hash)
    assert got is not None and got.about["approved_by"] == "alice"
    assert approvals.get(None) is None
    assert policy.revoke(manifest.release_hash)
    assert not policy.revoke(manifest.release_hash)
    assert not policy.revoke("../x")
    assert approvals.get(manifest.release_hash) is None


def test_tampered_approval_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_manifest((DATA / "basic.zip").read_bytes())
    monkeypatch.setattr(policy, "APPROVALS_TTL_S", -1.0)
    path = policy.approve(manifest, "alice")
    raw = json.loads(path.read_text())
    raw["body"]["tasks"]["align"]["runtime"]["command"] = "id"
    path.write_text(json.dumps(raw))
    assert policy.Approvals().get(manifest.release_hash) is None


def test_revoke_takes_effect_after_the_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = build_manifest((DATA / "basic.zip").read_bytes())
    policy.approve(manifest, "alice")
    approvals = policy.Approvals()
    assert approvals.get(manifest.release_hash) is not None
    policy.revoke(manifest.release_hash)
    assert approvals.get(manifest.release_hash) is not None, "cached"
    monkeypatch.setattr(policy, "APPROVALS_TTL_S", -1.0)
    assert approvals.get(manifest.release_hash) is None


# --- CLI --------------------------------------------------------------------


def test_cli_approve_from_file(
    policy_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    zip_path = str(DATA / "basic.zip")
    expected = build_manifest((DATA / "basic.zip").read_bytes())
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("no\n"))
    assert cli.main(["approve", zip_path]) == 1
    assert not (policy_dir / "approved").exists()
    out = capsys.readouterr().out
    assert "bwa mem" in out and "GISAID_TOKEN" in out and "Warning" in out
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("yes\n"))
    monkeypatch.setenv("SUDO_USER", "alice")
    assert cli.main(["approve", zip_path]) == 0
    stored = policy_dir / "approved" / f"{expected.release_hash}.json"
    about = Manifest.from_json(stored.read_text()).about
    assert about["approved_by"] == "alice"
    assert cli.main(["approvals"]) == 0
    assert expected.release_hash in capsys.readouterr().out
    assert cli.main(["revoke", expected.release_hash]) == 0
    assert cli.main(["revoke", expected.release_hash]) == 1


def test_cli_approve_refusals(
    policy_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zip_path = str(DATA / "basic.zip")
    assert cli.main(["approve", zip_path, "--expect-hash", "0" * 64]) == 1
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"nope")
    assert cli.main(["approve", str(bad), "--yes"]) == 1
    monkeypatch.setattr(config, "POLICY_OWNERS", frozenset({0}))
    if os.geteuid() != 0:
        assert cli.main(["approve", zip_path, "--yes"]) == 1
        assert cli.main(["revoke", "abc"]) == 1
    assert not (policy_dir / "approved").exists()


def test_cli_approve_by_release_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from horus_agent import config

    path = tmp_path / "agent.json"
    config.save(
        config.AgentConfig(url=URL, agent_id="a", name="lab", token="tok"),
        path,
    )
    monkeypatch.setenv(config.CONFIG_ENV, str(path))
    seen: dict[str, object] = {}

    class Response:
        status_code = 200
        content = (DATA / "slurm.zip").read_bytes()

    def fake_get(url: str, **kwargs: object) -> Response:
        seen["url"] = url
        seen["headers"] = kwargs["headers"]
        return Response()

    monkeypatch.setattr("httpx.get", fake_get)
    assert cli.main(["approve", "rel-1", "--yes"]) == 0
    assert seen["url"] == f"{URL}/agents/me/releases/rel-1/archive"
    assert seen["headers"] == {"Authorization": "Bearer tok"}
    Response.status_code = 404
    assert cli.main(["approve", "rel-2", "--yes"]) == 1


def test_cli_policy_check(
    policy_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["policy", "check"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "strict"
    write_policy(policy_dir, "mode = 3\n")
    assert cli.main(["policy", "check"]) == 1
