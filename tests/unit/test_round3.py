#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""Regression tests for the third review round of the policy work."""

import base64
import io
import os
import stat
import tarfile
from pathlib import Path

import pytest

from horus_agent import launch, native, policy
from horus_agent.guard import Guard
from horus_agent.manifest import ManifestError
from horus_agent.policy import Approvals, Policy
from horus_agent.protocol import PolicyRefusedError
from tests.unit.test_guard import Env, env, refused
from tests.unit.test_policy_hardening import launch_t, manifest

__all__ = ["env"]


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def package(path: Path, members: dict[str, bytes], mode: int = 0o644) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))


# --- the unpack marker is read once -----------------------------------------


async def test_the_unpack_marker_cannot_be_written_by_an_op(env: Env) -> None:
    await refused(
        env,
        "put_file",
        "may not write",
        path=str(env.run / "..horus.horus-sha256"),
        data=b64(b"0"),
    )


async def test_unpack_decides_once_and_never_rereads_the_marker(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    (env.run / "results").mkdir()
    pkg = env.run / "r.tgz"
    package(pkg, {"x.json": b"{}"})
    sha = native.digest(pkg)
    (env.run / ".results.horus-sha256").write_text("stale\n")
    seen: list[object] = []
    real = native.unpack

    def spy(*args: object, **kw: object) -> None:
        seen.append(args[5])
        real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(native, "unpack", spy)
    await env.call("unpack", pkg=str(pkg), dest=str(env.run / "results"))
    assert seen == [sha]  # the digest taken at the check, passed along
    assert (env.run / "results" / "x.json").exists()


async def test_unpack_refuses_run_metadata_members(env: Env) -> None:
    pkg = env.run / "m.tgz"
    package(pkg, {".horus": b"x"})
    await refused(
        env, "unpack", "run metadata", pkg=str(pkg), dest=str(env.run / "o")
    )


# --- Python and other interpreters never run the working directory ----------


async def test_python_does_not_import_from_the_cwd(env: Env) -> None:
    approved = manifest("python3 -m json.tool ${inp} > ${out}")
    policy.approve(approved, "root")
    (env.run / "json").mkdir()
    (env.run / "json" / "__init__.py").write_text(
        "open('PWNED', 'w').write('x')\n"
    )
    (env.run / "in.json").write_text("{}")
    with pytest.raises(RuntimeError, match="holds settings or code"):
        await launch_t(
            env, approved, env.run, "j", inp={"path": str(env.run / "in.json")}
        )
    assert not (env.run / "PWNED").exists()


def test_job_env_pins_interpreters() -> None:
    assert launch.job_env(Path("/h"))["PYTHONSAFEPATH"] == "1"
    for name in (
        "PERL5LIB",
        "PERL5OPT",
        "NODE_OPTIONS",
        "RUBYOPT",
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "BASH_ENV",
        "ENV",
        "PYTHONPATH",
    ):
        assert name in launch.UNSET_ENV


# --- ordinary templates are approved, code positions are not ----------------


@pytest.mark.parametrize(
    "command",
    [
        "Rscript script.R ${inp} ${out}",
        "python script.py --in ${inp}",
        "jq -r .x ${inp} > ${out}",
        "snakemake --cores ${inp}",
        "python3 s.py -c ${inp} -p ${inp} -r ${out}",
        "cat > /tmp/x.sh <<NF\necho hi\nNF\nsh /tmp/x.sh ${inp} > ${out}",
        "srun --mem=4G bwa mem ${inp} > ${out}",
        "mpirun -x OMP_NUM_THREADS=4 -np 4 prog ${inp}",
        "sed -e s/a/b/ ${inp} > ${out}",
        'echo \'{"a": "${inp}"}\' > ${out}',
    ],
)
def test_ordinary_commands_are_approved(command: str) -> None:
    manifest(command)


@pytest.mark.parametrize(
    "command",
    [
        "for f in a b; do ${inp} $f; done",
        "if ${inp}; then true; fi",
        "echo ${inp} | sh 2>/dev/null",
        "echo ${inp} | env bash",
        "bash -o pipefail ${inp}",
        "perl -I lib ${inp}",
        "python3 -W ignore ${inp}",
        "awk -F , ${inp}",
        "conda run -n e ${inp}",
        "docker run img ${inp}",
    ],
)
def test_code_positions_are_refused(command: str) -> None:
    with pytest.raises(ManifestError):
        manifest(command)


def test_embedded_values_are_listed_for_the_approver() -> None:
    quiet = manifest("cat ${inp} > ${out}")
    assert not any("embedded" in w for w in quiet.warnings)
    loud = manifest("cat > ${out} <<EOF\n${inp}\nEOF")
    assert any("embedded script" in w for w in loud.warnings)
    assert any(
        "embedded script" in w for w in manifest('echo "${inp}"').warnings
    )


# --- nothing the cloud writes is executable ---------------------------------


async def test_staged_files_are_not_executable(env: Env) -> None:
    path = env.run / "tool.sh"
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    await env.call("put_file", path=str(path), data=b64(b"echo"))
    assert not path.stat().st_mode & 0o111
    pkg = env.run / "p.tgz"
    package(pkg, {"bin/run.txt": b"#!/bin/sh\n"}, mode=0o755)
    await env.call("unpack", pkg=str(pkg), dest=str(env.run / "out"))
    assert not (env.run / "out" / "bin" / "run.txt").stat().st_mode & 0o111


# --- agent-owned folders ----------------------------------------------------


def test_a_group_writable_work_dir_needs_the_sticky_bit(
    tmp_path: Path,
) -> None:
    work = tmp_path / "w"
    work.mkdir()
    work.chmod(0o770)
    with pytest.raises(PolicyRefusedError, match="sticky"):
        Guard(Policy(work_dir=str(work), allowed_hosts=()), Approvals())
    work.chmod(0o3770)
    Guard(Policy(work_dir=str(work), allowed_hosts=()), Approvals())


def test_the_store_takes_the_work_dirs_group(tmp_path: Path) -> None:
    work = tmp_path / "w"
    work.mkdir()
    work.chmod(0o3770)
    guard = Guard(Policy(work_dir=str(work), allowed_hosts=()), Approvals())
    info = guard.const_store.stat()
    assert info.st_gid == work.stat().st_gid
    assert info.st_mode & stat.S_ISGID
    assert info.st_mode & 0o750 == 0o750


# --- values -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "value"),
    [
        ({"kind": "boolean"}, "on"),
        ({"kind": "boolean"}, "0"),
        ({"kind": "boolean", "options": ["--gpu", "--cpu"]}, "--gpu"),
        ({"kind": "number"}, "1e-05"),
        ({"kind": "number"}, "+.5"),
    ],
)
def test_valid_values_are_accepted(spec: dict[str, str], value: str) -> None:
    launch.check_value("x", spec, value)


def test_a_boolean_outside_its_mapping_is_refused() -> None:
    with pytest.raises(PolicyRefusedError):
        launch.check_value("x", {"kind": "boolean"}, "--gpu")


def test_choice_options_come_from_the_workflow_input() -> None:
    approved = manifest(
        "echo ${c} > ${out}",
        inputs=[{"id": "c", "kind": "choice", "options": ["a", "b"]}],
    )
    task = approved.task("t")
    assert task is not None
    assert task["inputs"]["c"]["options"] == ["a", "b"]


# --- open mode --------------------------------------------------------------


async def test_open_mode_accepts_dotfiles(env: Env) -> None:
    env.with_policy(mode="open")
    await env.call(
        "put_file", path=str(env.run / ".gitignore"), data=b64(b"x")
    )
    pkg = env.run / "d.tgz"
    package(pkg, {".gitkeep": b"", "._a": b"", "f": b"1"})
    await env.call("unpack", pkg=str(pkg), dest=str(env.run / "o"))
    assert (env.run / "o" / ".gitkeep").exists()


async def test_strict_unpack_drops_macos_files(env: Env) -> None:
    pkg = env.run / "d.tgz"
    package(pkg, {".DS_Store": b"", "._a": b"", "f": b"1"})
    await env.call("unpack", pkg=str(pkg), dest=str(env.run / "o"))
    assert sorted(p.name for p in (env.run / "o").iterdir()) == ["f"]


_ = os


async def test_a_value_may_not_name_an_environment_variable(env: Env) -> None:
    approved = manifest(
        "echo ${n} > ${out}",
        inputs=[{"id": "n", "kind": "string"}],
        secrets={"TOKEN": "x"},
    )
    policy.approve(approved, "root")
    with pytest.raises(RuntimeError, match="environment variable"):
        await launch_t(env, approved, env.run, "j", n={"value": "TOKEN"})


@pytest.mark.parametrize(
    "command",
    [
        "perl -Ilib s.pl ${inp} > ${out}",
        "perl -ne 'print' ${inp} > ${out}",
        "Rscript --vanilla s.R ${inp} > ${out}",
        "nextflow run main.nf -profile docker --in ${inp}",
        "find . -name x -exec cat {} ; ; cat ${inp} > ${out}",
        "nice -n 5 timeout 60 cat ${inp} > ${out}",
        "conda run -n env python s.py ${inp} > ${out}",
        "command -v ${inp} > ${out}",
        "cat ${inp} | sort > ${out}",
        "awk -F , '{print $1}' ${inp} > ${out}",
        "R --vanilla -f s.R --args ${inp}",
    ],
)
def test_more_ordinary_commands_are_approved(command: str) -> None:
    manifest(command)


@pytest.mark.parametrize(
    "command",
    [
        "perl -e ${inp}",
        "perl -ne ${inp}",
        "ruby -e${inp}",
        "node -p ${inp}",
        "make -f ${inp}",
        "nextflow run ${inp}",
        "find . -exec ${inp} {} +",
        "timeout -s KILL 5 ${inp}",
        "sudo -u x ${inp}",
        "echo ${inp} | python3",
        "xargs ${inp}",
        "ssh host ${inp}",
        "python3 - <<EOF\n${inp}\nEOF",
        "cat <<-EOF\n${inp}\nEOF",
        'echo "$(${inp})"',
        "echo $(( ${inp} + 1 ))",
    ],
)
def test_more_code_positions_are_refused(command: str) -> None:
    with pytest.raises(ManifestError):
        manifest(command)
