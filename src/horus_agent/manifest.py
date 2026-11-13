#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
The manifest of a workflow release: everything it can run on an agent.

Built from the release archive (``workflow.yaml`` plus the files it bundles)
by reading the YAML as plain data, so the same bytes give the same manifest
and the same :attr:`Manifest.release_hash` wherever it is built, with or
without any Horus plugin installed. An agent with a policy runs only tasks
of manifests its machine's administrator approved.

``body`` holds what runs and is hashed; ``about`` (name, version...) is shown
when approving but not hashed, so re-publishing the same runnable content
needs no new approval.
"""

import hashlib
import io
import json
import os
import posixpath
import re
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import yaml

from horus_agent.template import placeholders

SCHEMA = 1
WORKFLOW_MEMBER = "workflow.yaml"
MAX_ARCHIVE_BYTES = 2 * 1024**3
"""Largest uncompressed release the builder reads."""

VALUE_KINDS = frozenset(
    {"string", "number", "boolean", "choice", "multichoice", "list"}
)
"""Artifact kinds whose placeholder renders their value, not a path."""

RESERVED_ENV = re.compile(
    r"LD_.*|DYLD_.*|PYTHON.*|PATH|BASH_ENV|ENV|HOME|SHELL|IFS|"
    r"HORUS_SIDE_ARTIFACTS_DIR",
    re.IGNORECASE,
)
"""Environment names a task's secrets may not set on an agent."""

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SBATCH_VALUE_BAD = re.compile(r"[\s#'\"\\\x00]")
_SBATCH_LINE_BAD = re.compile(r"[\n\r\x00]")
SLURM_FIELDS = (
    "partition",
    "account",
    "qos",
    "nodes",
    "ntasks",
    "cpus_per_task",
    "mem",
    "time_limit",
    "gres",
    "extra_sbatch_args",
)
"""Slurm target fields an approved job script is rendered from."""
_SLURM_VALUES = ("partition", "account", "qos", "mem", "time_limit", "gres")
_SLURM_INTS = ("nodes", "ntasks", "cpus_per_task")
_RESOURCE_KEYS = ("cpus", "gpus", "memory_gb", "walltime")


class ManifestError(ValueError):
    """The release holds something an approved agent cannot run."""


def canonical_json(value: Any) -> bytes:
    """
    *value* as the bytes its hash is taken over: sorted keys, no spaces,
    UTF-8. Floats are refused: their text is not stable across producers.
    """
    _no_floats(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _no_floats(value: Any) -> None:
    if isinstance(value, float):
        raise ManifestError(f"Floats cannot be hashed: {value!r}")
    if isinstance(value, dict):
        for item in value.values():
            _no_floats(item)
    elif isinstance(value, list):
        for item in value:
            _no_floats(item)


def body_hash(body: Mapping[str, Any]) -> str:
    """The release hash of a manifest *body*: sha256 hex of its JSON."""
    return hashlib.sha256(canonical_json(dict(body))).hexdigest()


@dataclass(frozen=True)
class Manifest:
    """A release's runnable content (``body``) and its description."""

    body: dict[str, Any]
    about: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    """What an approver should read twice; not part of the hash."""

    @property
    def release_hash(self) -> str:
        """sha256 of :attr:`body`; the id an approval is stored under."""
        return body_hash(self.body)

    @property
    def tasks(self) -> dict[str, dict[str, Any]]:
        """Task id -> what the task runs."""
        tasks: dict[str, dict[str, Any]] = self.body["tasks"]
        return tasks

    def task(self, task_id: str) -> dict[str, Any] | None:
        """The task serving *task_id*; a map clone ``x[3]`` serves ``x``."""
        return self.tasks.get(task_id.split("[", 1)[0])

    def to_json(self) -> str:
        """The manifest as stored for an approval."""
        return json.dumps(
            {
                "about": self.about,
                "body": self.body,
                "release_hash": self.release_hash,
                "warnings": list(self.warnings),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        """
        Read :meth:`to_json` output, checking the stored hash.

        Raises:
            ManifestError: The stored hash does not match the body.
        """
        raw = json.loads(text)
        manifest = cls(
            body=raw["body"],
            about=raw.get("about") or {},
            warnings=tuple(raw.get("warnings") or ()),
        )
        if raw.get("release_hash") != manifest.release_hash:
            raise ManifestError("Stored hash does not match the manifest")
        if manifest.body.get("schema") != SCHEMA:
            raise ManifestError(
                f"Unknown manifest schema {manifest.body.get('schema')!r}"
            )
        return manifest


def file_digest(data: bytes) -> str:
    """A file's digest, as the runtime takes it: sha256 hex of its bytes."""
    return hashlib.sha256(data).hexdigest()


_TREE_TAG = b"horus-tree\n"


def tree_digest(leaves: Mapping[str, str], prefix: str = "deep:") -> str:
    """
    A folder's digest from its regular files' relative paths -> digests:
    the runtime's Merkle tree, so it equals ``ArtifactStore.digest``.
    """
    tree: dict[str, Any] = {}
    for path, digest in leaves.items():
        *dirs, name = path.split("/")
        node = tree
        for part in dirs:
            node = node.setdefault(part, {})
        node[name] = digest

    empty = prefix + hashlib.sha256(_TREE_TAG).hexdigest()

    def walk(node: Mapping[str, Any]) -> str:
        children = {
            os.fsencode(name): child if isinstance(child, str) else walk(child)
            for name, child in node.items()
        }
        body = b"".join(
            name + b"\0" + digest.encode() + b"\0"
            for name, digest in sorted(children.items())
            if digest != empty
        )
        return prefix + hashlib.sha256(_TREE_TAG + body).hexdigest()

    return walk(tree)


def build_manifest(
    archive: bytes, about: Mapping[str, Any] | None = None
) -> Manifest:
    """
    The manifest of a release archive (zip: ``workflow.yaml`` and the files
    it bundles, at their workflow-relative paths).

    Raises:
        ManifestError: The archive is unreadable or holds something an
            approved agent cannot run.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            if sum(i.file_size for i in zf.infolist()) > MAX_ARCHIVE_BYTES:
                raise ManifestError("The archive is too large")
            files = {
                info.filename: zf.read(info)
                for info in zf.infolist()
                if not info.is_dir()
            }
    except (zipfile.BadZipFile, OSError) as exc:
        raise ManifestError(f"Not a release archive: {exc}") from exc
    try:
        workflow = yaml.safe_load(files.pop(WORKFLOW_MEMBER))
    except KeyError as exc:
        raise ManifestError(f"The archive has no {WORKFLOW_MEMBER}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"{WORKFLOW_MEMBER} is not YAML: {exc}") from exc
    return build_manifest_from_workflow(workflow, files, about)


def build_manifest_from_workflow(
    workflow: Any,
    files: Mapping[str, bytes],
    about: Mapping[str, Any] | None = None,
) -> Manifest:
    """
    The manifest of a parsed ``workflow.yaml`` whose bundled files are
    *files* (archive-relative path -> bytes).

    Raises:
        ManifestError: The workflow holds something an approved agent cannot
            run.
    """
    if not isinstance(workflow, dict):
        raise ManifestError("workflow.yaml must be a mapping")
    builder = _Builder(files)
    builder.walk(workflow, prefix="")
    body = {"schema": SCHEMA, "tasks": builder.tasks, "files": builder.used}
    info = {"name": workflow.get("name")}
    info.update(about or {})
    return Manifest(body=body, about=info, warnings=tuple(builder.warnings))


def _archive_path(raw: str) -> str:
    """Where tc-os bundles the file of a declared input *raw* path."""
    if not raw.startswith("/"):
        return posixpath.normpath(raw)
    for marker in ("inputs/", "results/"):
        idx = raw.find(marker)
        if idx >= 0:
            return raw[idx:]
    return posixpath.basename(raw)


def _items(value: Any) -> list[dict[str, Any]]:
    return [v for v in value or [] if isinstance(v, dict)]


def _str(value: Any) -> str:
    """A scalar as the runtime renders it into a command."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    raise ManifestError(f"Unsupported value {value!r}")


def _option(value: Any) -> str:
    """A list item as the runtime renders it (``option_text``)."""
    return value if isinstance(value, str) else json.dumps(value)


class _Builder:
    """Collects the agent tasks of one workflow document, recursively."""

    def __init__(self, files: Mapping[str, bytes]) -> None:
        self.files = files
        self.tasks: dict[str, dict[str, Any]] = {}
        self.used: dict[str, dict[str, Any]] = {}
        self.warnings: list[str] = []

    def walk(self, workflow: dict[str, Any], prefix: str) -> None:
        roots = {a["id"]: a for a in _items(workflow.get("artifacts"))}
        named = workflow.get("targets") or {}
        if isinstance(named, list):
            named = {t.get("name") or t.get("id"): t for t in _items(named)}
        fed: dict[tuple[str, str], str] = {}
        for edge in _items(workflow.get("edges")):
            key = (str(edge.get("target")), str(edge.get("target_input")))
            fed[key] = str(edge.get("source") or "")
        for task in _items(workflow.get("tasks")):
            task_id = str(task.get("id"))
            sub = task.get("sub") or task.get("body")
            if task.get("kind") == "subworkflow" or isinstance(sub, dict):
                if not isinstance(sub, dict):
                    raise ManifestError(f"Subworkflow {task_id!r} has no body")
                self.walk(sub, f"{prefix}{task_id}/")
                continue
            target, shared = task.get("target"), False
            if isinstance(target, str):
                target, shared = named.get(target), True
            chain = list(_chain(target))
            if "agent" not in [hop.get("kind") for hop in chain]:
                continue
            self.tasks[prefix + task_id] = self._task(
                task, chain, shared, roots, fed
            )

    def _task(
        self,
        task: dict[str, Any],
        chain: list[dict[str, Any]],
        shared: bool,
        roots: dict[str, dict[str, Any]],
        fed: dict[tuple[str, str], str],
    ) -> dict[str, Any]:
        task_id = str(task.get("id"))
        where = f"Task {task_id!r}"
        kind = task.get("kind", "horus_task")
        if kind not in ("horus_task", "horus_map"):
            raise ManifestError(f"{where}: kind {kind!r} cannot run on agents")
        executor = task.get("executor") or {}
        if executor.get("kind") != "shell":
            raise ManifestError(
                f"{where}: executor {executor.get('kind')!r} cannot run on "
                "an approved agent (only shell)"
            )
        inputs = {
            a["id"]: self._input(task_id, a, roots, fed)
            for a in _items(task.get("inputs"))
        }
        if kind == "horus_map":
            for spec in [task.get("over") or {}, *_items(task.get("zip"))]:
                if spec.get("as"):
                    inputs[str(spec["as"])] = {"kind": "item", "role": "edge"}
        outputs = {
            a["id"]: {"kind": str(a.get("kind", "file"))}
            for a in _items(task.get("outputs"))
        }
        runtime = self._runtime(where, task.get("runtime") or {}, inputs)
        templates = [runtime.get("command", ""), runtime.get("args", "")]
        ids = set(inputs) | set(outputs)
        for template in templates:
            self._check_placeholders(where, template, ids)
        for name, item in inputs.items():
            if item.get("role") == "param" and item["kind"] == "string":
                self.warnings.append(
                    f"{where}: any value the launcher types for {name!r} is "
                    "passed, quoted, to its command"
                )
        return {
            "name": str(task.get("name", task_id)),
            "target": self._target(where, chain, shared, task),
            "executor": {"kind": "shell"},
            "runtime": runtime,
            "inputs": inputs,
            "outputs": outputs,
            "env": self._env(where, task.get("secrets") or {}),
        }

    def _input(
        self,
        task_id: str,
        artifact: dict[str, Any],
        roots: dict[str, dict[str, Any]],
        fed: dict[tuple[str, str], str],
    ) -> dict[str, Any]:
        kind = str(artifact.get("kind", "file"))
        source = fed.get((task_id, artifact["id"]))
        if source is not None and not source.startswith("artifact-"):
            return {"kind": kind, "role": "edge"}
        if source is not None:
            artifact = roots.get(source.removeprefix("artifact-"), artifact)
        if artifact.get("value") is not None:
            value = artifact["value"]
            rendered: str | list[str] = (
                [_option(v) for v in value]
                if isinstance(value, list)
                else _str(value)
            )
            return {"kind": kind, "role": "constant", "value": rendered}
        if kind not in VALUE_KINDS and isinstance(artifact.get("path"), str):
            digest = self._bundled(_archive_path(artifact["path"]))
            if digest is not None:
                return {"kind": kind, "role": "constant", "sha256": digest}
        return {"kind": kind, "role": "param"}

    def _bundled(self, path: str) -> str | None:
        """Digest of the file or folder bundled at *path*, if any."""
        if path in self.files:
            digest = file_digest(self.files[path])
            self.used[digest] = {"size": len(self.files[path]), "member": path}
            return digest
        prefix = path.rstrip("/") + "/"
        members = {
            name[len(prefix) :]: file_digest(data)
            for name, data in self.files.items()
            if name.startswith(prefix)
        }
        return tree_digest(members) if members else None

    def _runtime(
        self, where: str, runtime: dict[str, Any], inputs: dict[str, Any]
    ) -> dict[str, Any]:
        kind = runtime.get("kind")
        if kind == "command":
            return {"kind": "command", "command": str(runtime["command"])}
        if kind != "python_script":
            raise ManifestError(
                f"{where}: runtime {kind!r} cannot run on an approved agent"
            )
        script = str(runtime.get("script", ""))
        names = placeholders(script)
        if names:
            ref = next(iter(names))
            source = inputs.get(ref, {})
            digest = source.get("sha256")
            if (
                len(names) != 1
                or source.get("role") != "constant"
                or not digest
                or digest.startswith("deep:")
            ):
                raise ManifestError(
                    f"{where}: script {script!r} must name one bundled input"
                )
        else:
            digest = self._bundled(_archive_path(script))
            if digest is None or digest.startswith("deep:"):
                raise ManifestError(
                    f"{where}: script {script!r} is not in the release"
                )
        return {
            "kind": "python_script",
            "python": str(runtime.get("python", "python")),
            "script_sha256": digest,
            "args": str(runtime.get("args", "")),
        }

    @staticmethod
    def _check_placeholders(where: str, template: str, ids: set[str]) -> None:
        for name in placeholders(template):
            root, _dot, attr = name.partition(".")
            if root == "task":
                if attr not in ("id", "name"):
                    raise ManifestError(
                        f"{where}: ${{{name}}} cannot be checked on an agent"
                    )
            elif root in ids and attr:
                raise ManifestError(
                    f"{where}: ${{{name}}} cannot be checked on an agent; "
                    f"use ${{{root}}}"
                )

    @staticmethod
    def _env(where: str, secrets: Any) -> dict[str, Any]:
        if not isinstance(secrets, dict):
            raise ManifestError(f"{where}: secrets must be a mapping")
        for name in secrets:
            if not _ENV_NAME.fullmatch(name) or RESERVED_ENV.fullmatch(name):
                raise ManifestError(f"{where}: secret name {name!r} refused")
        return {str(k): v for k, v in secrets.items()}

    def _target(
        self,
        where: str,
        chain: list[dict[str, Any]],
        shared: bool,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        kinds = [str(hop.get("kind")) for hop in chain]
        if kinds not in (["agent"], ["slurm", "agent"]):
            raise ManifestError(
                f"{where}: target chain {kinds} cannot run on an approved "
                "agent (agent, or slurm over agent)"
            )
        agent = chain[-1]
        wd = agent.get("working_directory")
        if wd is not None and (str(wd).startswith("/") or ".." in str(wd)):
            raise ManifestError(
                f"{where}: working_directory must be relative to the "
                "agent's work dir"
            )
        out: dict[str, Any] = {"chain": kinds, "working_directory": wd}
        if kinds[0] == "slurm":
            if shared:
                raise ManifestError(
                    f"{where}: shared Slurm allocations cannot run on an "
                    "approved agent"
                )
            out["slurm"] = self._slurm(where, chain[0], task)
        return out

    @staticmethod
    def _slurm(
        where: str, hop: dict[str, Any], task: dict[str, Any]
    ) -> dict[str, Any]:
        slurm: dict[str, Any] = {
            "nodes": 1,
            "ntasks": 1,
            "extra_sbatch_args": [],
        }
        slurm.update({k: hop[k] for k in SLURM_FIELDS if k in hop})
        for key in _SLURM_VALUES:
            value = slurm.get(key)
            if value is not None:
                if not isinstance(value, str) or _SBATCH_VALUE_BAD.search(
                    value.strip()
                ):
                    raise ManifestError(f"{where}: invalid Slurm {key}")
                slurm[key] = value.strip()
        for key in _SLURM_INTS:
            value = slurm.get(key)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ManifestError(f"{where}: Slurm {key} must be an integer")
        args = slurm["extra_sbatch_args"]
        if not isinstance(args, list) or any(
            not isinstance(a, str) or _SBATCH_LINE_BAD.search(a) for a in args
        ):
            raise ManifestError(f"{where}: invalid extra_sbatch_args")
        slurm["extra_sbatch_args"] = [a.strip() for a in args]
        resources = task.get("resources") or {}
        slurm["resources"] = {
            k: resources[k] for k in _RESOURCE_KEYS if k in resources
        }
        return slurm


def _chain(target: Any) -> Iterator[dict[str, Any]]:
    """*target*, then its ``inner``, recursively."""
    while isinstance(target, dict):
        yield target
        target = target.get("inner")
