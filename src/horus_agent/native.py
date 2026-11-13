#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
File plumbing the runtime otherwise runs as shell commands on a target
(``sha256sum``, ``tar``, ``mv``, ``wc -c``), done in Python by the agent.

Results match the runtime's commands: :func:`digest` equals
``ArtifactStore.digest``, :func:`pack` writes what ``tar -C src .`` writes
and :func:`unpack` leaves what ``FolderArtifact.unpack_command`` leaves.
"""

import contextlib
import hashlib
import os
import shutil
import stat as stat_mod
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from horus_agent.blobs import open_file
from horus_agent.manifest import tree_digest
from horus_agent.protocol import PolicyRefusedError

SMALL_FILE = 1 << 20
"""``fast`` digests hash files below this size by content."""
_CHUNK = 1024 * 1024


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open_file(path) as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _leaf(path: Path, info: os.stat_result, mode: str) -> str:
    """A file's digest in *mode*, as the runtime's ``_leaf``."""
    if mode == "deep":
        return _sha256(path)
    if info.st_size < SMALL_FILE:
        return "fast:" + _sha256(path)
    tag = b"horus-stat\n%d:%d" % (info.st_size, int(info.st_mtime))
    return "fast:" + hashlib.sha256(tag).hexdigest()


def digest(path: Path, mode: str = "deep") -> str:
    """
    The runtime digest of the file or folder at *path*: a file's sha256 (or
    ``fast:``...), a folder's ``deep:``/``fast:`` Merkle hash over its regular
    files. Symlinks inside a folder are skipped, as ``find -type f`` does.

    Raises:
        FileNotFoundError: *path* is neither a file nor a folder.
    """
    info = path.stat()
    if stat_mod.S_ISREG(info.st_mode):
        return _leaf(path, info, mode)
    if not stat_mod.S_ISDIR(info.st_mode):
        raise FileNotFoundError(f"Nothing to digest at {path}")
    leaves: dict[str, str] = {}
    for root, _dirs, names in os.walk(path):
        for name in names:
            full = Path(root) / name
            entry = full.lstat()
            if stat_mod.S_ISREG(entry.st_mode):
                rel = full.relative_to(path).as_posix()
                leaves[rel] = _leaf(full, entry, mode)
    return tree_digest(leaves, f"{mode}:")


def _no_links(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """Pack regular files and folders only."""
    return info if info.isfile() or info.isdir() else None


def pack(src: Path, pkg: Path) -> None:
    """The contents of folder *src*, rooted at ``.``, as a gzip tarball."""
    if not src.is_dir():
        raise NotADirectoryError(f"Not a folder: {src}")
    pkg.parent.mkdir(parents=True, exist_ok=True)
    with (
        open_file(pkg, "wb") as fh,
        tarfile.open(fileobj=fh, mode="w:gz", compresslevel=1) as tar,
    ):
        tar.add(src, arcname=".", filter=_no_links)


def unpack(
    pkg: Path, dest: Path, refuse: Callable[[str], bool] | None = None
) -> None:
    """
    Extract the tarball *pkg* into folder *dest*, refusing absolute paths,
    ``..``, links leaving it and device files. *dest* is swapped in whole,
    and left alone when it already holds this package (sha256 marker).
    """
    marker = dest.parent / f".{dest.name}.horus-sha256"
    sha = _sha256(pkg)
    if dest.is_dir() and marker.is_file() and read_text(marker) == sha:
        return
    tmp = dest.with_name(f"{dest.name}.horus-tmp{os.getpid()}")
    old = dest.with_name(f"{dest.name}.horus-old{os.getpid()}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        with open_file(pkg) as fh, tarfile.open(fileobj=fh, mode="r:*") as tar:
            tar.extractall(tmp, filter="data")
        for _root, dirs, names in os.walk(tmp):
            if refuse and (bad := [n for n in dirs + names if refuse(n)]):
                raise PolicyRefusedError(
                    f"The package holds {bad[0]!r}, a name tools read "
                    "settings or code from"
                )
        if dest.exists() or dest.is_symlink():
            dest.rename(old)
        tmp.rename(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
    with open_file(marker, "w") as out:
        out.write(f"{sha}\n")


def read_text(path: Path) -> str:
    """A small marker file's text, never read through a symlink."""
    with open_file(path, "r") as fh:
        return str(fh.read()).strip()


def move(src: Path, dst: Path) -> None:
    """Move *src* onto *dst*, creating parents (``mkdir -p && mv -f``)."""
    if src == dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_dir() and not dst.is_symlink() and src.is_dir():
        shutil.rmtree(dst)
    os.replace(src, dst)


def stat(path: Path) -> dict[str, Any] | None:
    """``{type, size, mtime}`` of *path*, ``None`` when missing."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    kind = (
        "dir"
        if stat_mod.S_ISDIR(info.st_mode)
        else "file"
        if stat_mod.S_ISREG(info.st_mode)
        else "other"
    )
    return {"type": kind, "size": info.st_size, "mtime": int(info.st_mtime)}


def materialise(src: Path, store: Path, expected: str) -> Path:
    """
    Copy the file or folder *src* into ``store/<hash>/<name>``, hashing the
    bytes as they are copied, and return the copy. Only regular files and
    folders are copied: a link, device or FIFO anywhere is refused. The copy
    is the agent's: tasks may read it but not change it. Content verified
    once is reused, also under another name.

    Raises:
        ValueError: *src* is not what *expected* (its digest) names, or
            holds something other than files and folders.
    """
    folder = store / expected.rpartition(":")[2]
    final = folder / src.name
    if final.exists():
        return final
    if not folder.is_dir():
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=store))
        try:
            copy = tmp / src.name
            info = src.lstat()
            if stat_mod.S_ISREG(info.st_mode):
                got = _copy_file(src, copy, info)
            elif stat_mod.S_ISDIR(info.st_mode):
                got = tree_digest(_copy_tree(src, copy), "deep:")
            else:
                raise ValueError(f"Not a file or folder: {src}")
            if got != expected:
                raise ValueError(
                    f"{src.name} is not the approved file or folder"
                )
            tmp.chmod(0o750)
            with contextlib.suppress(OSError):  # another launch won the race
                tmp.rename(folder)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if final.exists():
            return final
    # The same content under another name: link to the verified copy.
    existing = next(p for p in folder.iterdir() if not p.name.startswith("."))
    with contextlib.suppress(FileExistsError):
        if existing.is_dir():
            shutil.copytree(existing, final, copy_function=os.link)
        else:
            os.link(existing, final)
    return final


def _copy_file(src: Path, dst: Path, info: os.stat_result) -> str:
    """Copy one regular file, returning the sha256 of what was copied."""
    h = hashlib.sha256()
    mode = 0o750 if info.st_mode & 0o111 else 0o640
    with (
        open_file(src) as fin,
        os.fdopen(
            os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "wb"
        ) as fout,
    ):
        while chunk := fin.read(_CHUNK):
            h.update(chunk)
            fout.write(chunk)
    dst.chmod(mode)
    return h.hexdigest()


def _copy_tree(src: Path, dst: Path) -> dict[str, str]:
    """Copy a folder of files and folders; relative path -> file digest."""
    leaves: dict[str, str] = {}
    dst.mkdir(mode=0o750)
    for root, dirs, names in os.walk(src):
        rel = Path(root).relative_to(src)
        for name in [*dirs, *names]:
            info = (Path(root) / name).lstat()
            target = dst / rel / name
            if stat_mod.S_ISDIR(info.st_mode):
                target.mkdir(mode=0o750)
                target.chmod(0o750)
            elif stat_mod.S_ISREG(info.st_mode):
                leaves[(rel / name).as_posix()] = _copy_file(
                    Path(root) / name, target, info
                )
            else:
                raise ValueError(
                    f"{src.name} holds {rel / name}, which is not a file "
                    "or folder"
                )
    dst.chmod(0o750)
    return leaves
