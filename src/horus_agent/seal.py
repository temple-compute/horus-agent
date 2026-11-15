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

import errno
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
        _sync(fd, rel[-1], stored)
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
    """Remove *name* of *parent* without following links or crossing into
    another device (a mount point is refused, never emptied).
    """
    try:
        info = os.lstat(name, dir_fd=parent)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(name, dir_fd=parent)
        return
    if info.st_dev != os.fstat(parent).st_dev:
        raise OSError(errno.EBUSY, "is a mount point")
    fd = os.open(name, _DIR, dir_fd=parent)
    try:
        if os.fstat(fd).st_dev != info.st_dev:
            raise OSError(errno.EBUSY, "is a mount point")
        for entry in os.listdir(fd):
            _remove(fd, entry)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent)


def _same_file(parent: int, name: str, stored: Path, mode: int) -> bool:
    """Whether *name* is already the agent's own copy of *stored*: a plain
    file with one link, the agent as owner, the right mode and bytes.
    """
    try:
        info = os.lstat(name, dir_fd=parent)
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != mode
        or info.st_size != stored.stat().st_size
    ):
        return False
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
    with os.fdopen(fd, "rb") as have, open(stored, "rb") as want:
        while chunk := want.read(1 << 20):
            if have.read(len(chunk)) != chunk:
                return False
        return not have.read(1)


def _sync(parent: int, name: str, stored: Path) -> None:
    """Make *name* of *parent* match *stored*, touching only what differs
    (a task of the run may be reading the rest).
    """
    if stored.is_dir():
        try:
            info: os.stat_result | None = os.lstat(name, dir_fd=parent)
        except FileNotFoundError:
            info = None
        if info is not None and (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_dev != os.fstat(parent).st_dev
        ):
            _remove(parent, name)
            info = None
        if info is None:
            os.mkdir(name, DIR_MODE, dir_fd=parent)
        fd = os.open(name, _DIR, dir_fd=parent)
        try:
            if info is None or stat.S_IMODE(info.st_mode) != DIR_MODE:
                os.fchmod(fd, DIR_MODE)
            names = {e.name for e in os.scandir(stored)}
            for extra in set(os.listdir(fd)) - names:
                _remove(fd, extra)
            for member in sorted(names):
                _sync(fd, member, stored / member)
        finally:
            os.close(fd)
        return
    mode = (stored.stat().st_mode & 0o111) | 0o444
    if _same_file(parent, name, stored, mode):
        return
    try:
        if stat.S_ISDIR(os.lstat(name, dir_fd=parent).st_mode):
            _remove(parent, name)
    except FileNotFoundError:
        pass
    tmp = f".horus-tmp-{os.getpid()}-{name}"
    _remove(parent, tmp)
    out = os.open(
        tmp,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent,
    )
    with os.fdopen(out, "wb") as dst, open(stored, "rb") as src:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fchmod(dst.fileno(), mode)
        os.fsync(dst.fileno())
    os.rename(tmp, name, src_dir_fd=parent, dst_dir_fd=parent)


def verify(path: Path, expected: str) -> None:
    """Refuse a sealed copy whose digest is not *expected*."""
    if native.digest(path, "deep") != expected:
        raise PolicyRefusedError(f"{path.name!r} could not be sealed")
