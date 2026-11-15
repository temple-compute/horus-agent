#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
The approve-time checker: every template here once passed it (or is a
neighbour of one that did) and ran a file the cloud chose.
"""

from pathlib import Path

import pytest

from horus_agent.dryrun import dry_run
from horus_agent.manifest import ManifestError
from tests.unit.test_policy_hardening import manifest

INPUTS = [
    {"id": "inp", "kind": "file"},
    {"id": "msg", "kind": "string"},
]

REFUSED = [
    # line continuations, `--`, redirections between program and argument
    "bash \\\n ${inp} > ${out}",
    "bash \\\n\n ${inp}",
    "bash -- ${inp}",
    "bash 2>&1 ${inp}",
    "bash >${out} ${inp}",
    "bash > ${out} ${inp}",
    "python3 -- ${inp}",
    "perl -- ${inp}",
    "node -- ${inp}",
    "bash -O extglob -- ${inp}",
    "bash ./${inp}",
    "bash x/${inp}",
    "bash $PWD/${inp}",
    "bash --rcfile ${inp} -i",
    "bash --init-file ${inp}",
    "bash -c 'true' ${inp}",
    "sh -s ${inp} < in.sh",
    "bash < ${inp}",
    "java -jar ${inp}",
    "java -cp ${inp} Main",
    "java -Xmx4g -jar ${inp}",
    "jupyter nbconvert --execute ${inp}",
    "lua -e 1 ${inp}",
    # assignments
    's=${msg}; bash "$s"',
    's=${msg}; echo "$s"',
    'NAME="${msg}"; echo hi',
    "LD_PRELOAD=${msg} prog",
    "env LD_PRELOAD=${msg} prog",
    "BASH_ENV=${msg} prog",
    "PYTHONPATH=${msg} prog",
    "PYTHONSTARTUP=${msg} prog",
    "NODE_OPTIONS=${msg} prog",
    "PERL5OPT=${msg} prog",
    "RUBYOPT=${msg} prog",
    "JAVA_TOOL_OPTIONS=${msg} prog",
    "R_PROFILE_USER=${msg} prog",
    "PATH=${msg}:$PATH prog",
    "export ${msg}=1",
    "export X=${msg}",
    "declare X=${msg}",
    "readonly X=${msg}",
    # loops and programs named by variables
    "for f in ${inp}; do bash $f; done",
    "while read f; do bash $f; done < ${inp}",
    'while read f; do "$f"; done < ${inp}',
    "$CMD ${inp}",
    "for f in ${inp}; do ./$f; done",
    'for f in ${inp}; do "$f"; done',
    "for f in ${inp}; do eval $f; done",
    "for f in ${inp}; do source $f; done",
    "for f in ${inp}; do python3 $f; done",
    "cat ${inp} | while read f; do sh $f; done",
    "echo hi; ${CMD} ${inp}",
    # wrappers and re-parsing commands
    "echo ${inp} | xargs bash",
    "echo ${inp} | xargs -n1 bash",
    "echo | parallel ${msg}",
    "nohup bash ${inp}",
    "time bash ${inp}",
    "command bash ${inp}",
    "exec bash ${inp}",
    "env bash ${inp}",
    "env -i bash ${inp}",
    r"find . -exec bash ${inp} \;",
    r"find . -execdir sh ${inp} \;",
    r"find . -ok ${msg} \;",
    "eval ${msg}",
    "ssh host ${msg}",
    "watch ${msg}",
    # heredocs
    "cat <<EOF\n$(bash ${inp})\nEOF",
    "cat <<EOF\n`bash ${inp}`\nEOF",
    "cat > run.sh <<'EOF'\nbash ${inp}\nEOF\nbash run.sh",
    "cat > ./run.sh <<EOF\nbash ${inp}\nEOF\nsh ./run.sh",
    "cat > run.sh <<'EOF'\nbash ${inp}\nEOF\n. run.sh",
    "printf 'bash %s\\n' ${inp} > run.sh; bash run.sh",
    "echo bash ${inp} > run.sh; sh run.sh",
    "echo $'a' ${msg}",
    # options that run what they are given
    "git -c core.sshCommand=${msg} fetch",
    "git -c ${msg} fetch",
    "git clone --upload-pack ${msg} u",
    "git --exec-path=${msg} status",
    "tar --to-command=${msg} -xf a",
    "tar --checkpoint-action=exec=${msg} -cf a b",
    "tar --use-compress-program ${msg} -cf a b",
    "tar -I ${msg} -cf a b",
    "rsync -e ${msg} a b",
    "rsync --rsh=${msg} a b",
    "rsync --rsh ${msg} a b",
    "scp -o ProxyCommand=${msg} a b",
    "sort --compress-program=${msg} f",
    "sed -e ${msg} f",
    "sed -n ${msg}p f",
    "sed --expression=${msg} f",
    "awk -f ${msg} f",
    "awk 'BEGIN{system(\"id\")}' ${inp}",
    "awk -v c=${msg} 'BEGIN{system(c)}'",
    "make -f ${inp}",
    "make ${msg}",
    "Rscript ${inp}",
    "Rscript -e ${msg}",
    "snakemake -s ${inp}",
    "nextflow run ${inp}",
    "python ${inp}",
    "python3 -m ${msg}",
    "perl ${inp}",
    "perl -e ${msg}",
    "ruby ${inp}",
    "wget -e ${msg} u",
    "pandoc --filter ${msg} a",
    "java -javaagent:${msg} -jar a.jar",
]

ACCEPTED = [
    "for f in ${msg}; do echo $f; done",
    "for f in ${inp}; do cp $f out/; done",
    'while read -r f; do wc -l "$f"; done < ${inp}',
    "cat ${inp} > ${out}",
    "fastqc -t 4 ${inp} -o outdir",
    "bwa mem -t 4 ref.fa ${inp} 2> log > ${out}",
    "bwa mem -t 4 ref.fa ${inp} > ${out} 2>&1",
    "samtools sort -@ 4 -o ${out} ${inp}",
    "cmd \\\n  --in ${inp} \\\n  --out ${out}",
    "tar -xzf ${inp} -C d",
    "rsync -av ${inp} dest",
    "git clone ${msg} d",
    "git -c user.name=x status ${msg}",
    "python3 script.py --a ${msg} > ${out}",
    "Rscript script.R ${inp}",
    "docker run --rm -v ${inp}:/in img cmd",
    "sed -n 1,5p ${inp} > ${out}",
    "awk '{print $1}' ${inp} > ${out}",
    "grep -v ${msg} ${inp} > ${out}",
    "mkdir -p x && cd x && fastqc ${inp}",
    "if [ -s ${inp} ]; then cat ${inp} > ${out}; fi",
    "sort -k1,1 -T tmp ${inp} > ${out}",
    "find . -name x -exec cat {} ; ; cat ${inp} > ${out}",
    "echo ${msg} | tr a-z A-Z > ${out}",
    "tool --name=${msg} --in=${inp} > ${out}",
    "cat > cfg.txt <<EOF\nname ${msg}\nEOF\ntool cfg.txt ${inp}",
    "cat > run.sh <<'EOF'\nls\nEOF\nbash run.sh ${inp}",
]


def build(command: str) -> None:
    manifest(command, inputs=INPUTS)


@pytest.mark.parametrize("command", REFUSED)
def test_these_templates_are_refused(command: str) -> None:
    with pytest.raises(ManifestError, match=r"\$\{"):
        build(command)


@pytest.mark.parametrize("command", ACCEPTED)
def test_ordinary_commands_still_pass(command: str) -> None:
    build(command)


def test_a_template_over_the_cap_is_refused() -> None:
    with pytest.raises(ManifestError, match="placeholders"):
        build("echo " + "${msg} " * 2000)


def test_unparseable_templates_are_refused_not_ignored() -> None:
    for command in (
        "echo '${msg}",
        'echo "${msg}',
        "echo $(${msg}",
        "cat <<EOF\n${msg}",
    ):
        with pytest.raises(ManifestError, match="quoting"):
            build(command)


def test_the_dry_run_counts_tasks(tmp_path: Path) -> None:
    flow = tmp_path / "w1"
    flow.mkdir()
    (flow / "workflow.yaml").write_text(
        """
name: w1
artifacts: []
tasks:
  - id: ok
    kind: horus_task
    target: {kind: local}
    executor: {kind: docker}
    inputs: [{id: i, kind: file, path: inputs/i.txt}]
    outputs: [{id: o, kind: file, path: results/o.txt}]
    runtime: {kind: command, command: "cat ${i} > ${o}"}
  - id: bad
    kind: horus_task
    target: {kind: local}
    executor: {kind: docker}
    inputs: [{id: i, kind: file, path: inputs/i.txt}]
    outputs: [{id: o, kind: file, path: results/o.txt}]
    runtime: {kind: command, command: "bash -- ${i}"}
  - id: fn
    kind: horus_task
    runtime: {kind: python_function}
edges: []
"""
    )
    result = dry_run(tmp_path)
    assert (result.total, result.approved, result.refused) == (2, 1, 1)
    assert result.skipped == 1
    assert "where the command runs it as code" in result.summary()
    assert result.refusals[0][0] == "w1/bad"


def test_the_dry_run_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from horus_agent.cli import main

    flow = tmp_path / "w"
    flow.mkdir()
    (flow / "workflow.yaml").write_text(
        "name: w\ntasks:\n  - id: bad\n    runtime: "
        '{kind: command, command: "bash -- ${i}"}\n'
        "    inputs: [{id: i, kind: file}]\n"
    )
    assert main(["approve", "--dry-run", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 tasks, 0 approved, 1 refused" in out
    assert "w/bad" in out
    assert main(["approve", "--dry-run", str(tmp_path / "none")]) == 1


def test_approve_warns_about_every_run_time_path() -> None:
    from horus_agent.manifest import build_manifest_from_workflow

    approved = build_manifest_from_workflow(
        {
            "name": "w",
            "tasks": [
                {
                    "id": "t",
                    "target": {"kind": "agent", "agent": "lab"},
                    "executor": {"kind": "shell"},
                    "runtime": {"kind": "command", "command": "cat ${inp}"},
                    "inputs": [{"id": "inp", "kind": "file"}],
                }
            ],
            "edges": [{"source": "up", "target": "t", "target_input": "inp"}],
        },
        {},
    )
    assert any(
        "path or value chosen at run time" in w for w in approved.warnings
    )
