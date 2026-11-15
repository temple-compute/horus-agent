#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Sealing shipped files at launch.

A command may run a file the release ships by its literal path. Cloud ops
can have changed what sits there (a link, a swapped parent folder), so just
before a task starts the agent puts its own verified copy at that path:
whatever occupies the path, or an ancestor that is not a plain folder, is
removed (every step is taken relative to an open folder, never through a
link), the folders are recreated and the files made read-only.
"""

import os
import shutil
import stat
from pathlib import Path

from horus_agent import native
from horus_agent.manifest import bad_path
from horus_agent.protocol import PolicyRefusedError

DIR_MODE = 0o755
_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def occupied(base: Path, rel: tuple[str, ...]) -> bool:
    """Whether anything sits at ``base/rel``, or in its way (a link, a file
    where a folder belongs), looking at each step without following links.
    """
    here = base
    for part in rel:
        here = here / part
        try:
            info = here.lstat()
        except FileNotFoundError:
            return False
        except NotADirectoryError:
            return True
        if not stat.S_ISDIR(info.st_mode):
            return True
    return True


def seal(
    base: Path, rel: tuple[str, ...], stored: Path, owners: frozenset[int]
) -> None:
    """
    Make ``base/rel`` the agent's read-only copy of *stored*, the verified
    file or folder (copied, never linked: the store's own files stay out of
    reach of every write to the run).

    Raises:
        PolicyRefusedError: Something in the way cannot be removed.
    """
    if not rel or any(bad_path(part) or "/" in part for part in rel):
        raise PolicyRefusedError(f"{'/'.join(rel)!r} is not a plain path")
    fd = os.open(base, _DIR)
    try:
        for part in rel[:-1]:
            nxt = _enter(fd, part, owners)
            os.close(fd)
            fd = nxt
        _remove(fd, rel[-1])
        _place(fd, rel[-1], stored)
    except OSError as exc:
        raise PolicyRefusedError(
            f"{'/'.join(rel)} cannot be sealed: {exc.strerror or exc}"
        ) from exc
    finally:
        os.close(fd)


def _enter(parent: int, name: str, owners: frozenset[int]) -> int:
    """Open folder *name* of *parent*, making it first when it is missing
    or is anything but a plain folder of the agent or of the task user.
    """
    try:
        info: os.stat_result | None = os.lstat(name, dir_fd=parent)
    except FileNotFoundError:
        info = None
    if info is not None and (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid not in owners
        or info.st_dev != os.fstat(parent).st_dev
    ):
        _remove(parent, name)
        info = None
    if info is None:
        os.mkdir(name, DIR_MODE, dir_fd=parent)
    fd = os.open(name, _DIR, dir_fd=parent)
    if info is None:
        os.fchmod(fd, DIR_MODE)
    return fd


def _remove(parent: int, name: str) -> None:
    try:
        info = os.lstat(name, dir_fd=parent)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode):
        shutil.rmtree(name, dir_fd=parent)
    else:
        os.unlink(name, dir_fd=parent)


def _place(parent: int, name: str, stored: Path) -> None:
    if stored.is_dir():
        os.mkdir(name, DIR_MODE, dir_fd=parent)
        fd = os.open(name, _DIR, dir_fd=parent)
        try:
            os.fchmod(fd, DIR_MODE)
            for entry in sorted(os.scandir(stored), key=lambda e: e.name):
                _place(fd, entry.name, Path(entry.path))
        finally:
            os.close(fd)
        return
    mode = (stored.stat().st_mode & 0o111) | 0o444
    out = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent,
    )
    with os.fdopen(out, "wb") as dst, open(stored, "rb") as src:
        shutil.copyfileobj(src, dst)
    os.chmod(name, mode, dir_fd=parent)


def verify(path: Path, expected: str) -> None:
    """Refuse a sealed copy whose digest is not *expected*."""
    if native.digest(path, "deep") != expected:
        raise PolicyRefusedError(f"{path.name!r} could not be sealed")
