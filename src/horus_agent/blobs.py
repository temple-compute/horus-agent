#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Streamed transfers to and from presigned staging URLs.

Used on both sides: the orchestrator and the agent each move large files
through object storage instead of through the broker.
"""

import errno
import os
import stat
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO, Any

import httpx
from horus_runtime.logging import horus_logger

from horus_agent.logs import safe_url

_CHUNK = 1024 * 1024
MAX_SINGLE_PUT = 5 * 1024**3
"""S3's limit for one PUT."""
TIMEOUT = httpx.Timeout(60.0, read=300.0, write=300.0)
"""No transfer waits forever on a silent peer."""


_OPEN_FLAGS = {
    "r": os.O_RDONLY,
    "w": os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
    "a": os.O_WRONLY | os.O_CREAT | os.O_APPEND,
}


def open_file(path: Path, mode: str = "rb") -> IO[Any]:
    """
    *path* opened in *mode* (``r``, ``w`` or ``a``, text or binary). A
    symlink is refused whatever it points to, and so is anything but a
    regular file (a FIFO would block the agent). Tasks can write where the
    agent reads and writes, so the agent never follows their links.
    """
    try:
        fd = os.open(
            path, _OPEN_FLAGS[mode[0]] | os.O_NOFOLLOW | os.O_NONBLOCK, 0o666
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"Refusing to follow a symlink: {path}") from exc
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError(f"Not a regular file: {path}")
    return os.fdopen(fd, mode)


async def _read_chunks(
    path: Path, offset: int = 0, length: int | None = None
) -> AsyncIterator[bytes]:
    left = length
    with open_file(path) as fh:
        fh.seek(offset)
        while left is None or left > 0:
            chunk = fh.read(_CHUNK if left is None else min(_CHUNK, left))
            if not chunk:
                return
            if left is not None:
                left -= len(chunk)
            yield chunk


async def upload(
    url: str,
    source: Path | bytes,
    offset: int | None = None,
    length: int | None = None,
) -> dict[str, str | None]:
    """
    PUT *source* to the presigned *url*, streaming a file from disk: all of
    it, or *length* bytes from *offset* (one part of a multipart upload).
    Returns the stored object's ``etag``.

    ``Content-Length`` is always sent: S3 refuses a chunked presigned PUT.
    """
    if isinstance(source, Path):
        with open_file(source) as fh:
            size = os.fstat(fh.fileno()).st_size
        start = offset or 0
        size = max(0, min(size - start, size if length is None else length))
        if size > MAX_SINGLE_PUT:
            # shortcut: one presigned PUT, S3's limit; add multipart if
            # agents need to move larger single files.
            raise ValueError(
                f"{source} is {size} bytes; files over 5 GiB cannot be "
                "moved to or from an agent yet"
            )
        content: bytes | AsyncIterator[bytes] = _read_chunks(
            source, start, size
        )
    else:
        size = len(source)
        content = source
    log = horus_logger.log
    log.debug(f"Upload {size} bytes to {safe_url(url)}")
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.put(
            url, content=content, headers={"Content-Length": str(size)}
        )
        response.raise_for_status()
    log.debug(f"Uploaded {size} bytes in {time.monotonic() - started:.2f}s")
    return {"etag": response.headers.get("etag")}


async def download_to(
    url: str, dest: Path, max_bytes: int | None = None
) -> None:
    """
    GET the presigned *url* into *dest*, streamed, creating parents; past
    *max_bytes* the download stops and *dest* is removed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    log = horus_logger.log
    log.debug(f"Download {safe_url(url)} to {dest}")
    started = time.monotonic()
    async with (
        httpx.AsyncClient(timeout=TIMEOUT) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        written = 0
        with open_file(dest, "wb") as fh:
            async for chunk in response.aiter_bytes(_CHUNK):
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    fh.close()
                    dest.unlink(missing_ok=True)
                    raise ValueError(
                        f"Download is over the {max_bytes} byte limit"
                    )
                fh.write(chunk)
    log.debug(
        f"Downloaded {dest.stat().st_size} bytes to {dest} in "
        f"{time.monotonic() - started:.2f}s"
    )


async def download(url: str) -> bytes:
    """GET the presigned *url* into memory."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content
