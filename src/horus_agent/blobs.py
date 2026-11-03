#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
Streamed transfers to and from presigned staging URLs.

Used on both sides: the orchestrator and the agent each move large files
through object storage instead of through the broker.
"""

import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
from horus_runtime.logging import horus_logger

from horus_agent.logs import safe_url

_CHUNK = 1024 * 1024
MAX_SINGLE_PUT = 5 * 1024**3
"""S3's limit for one PUT."""


async def _read_chunks(path: Path) -> AsyncIterator[bytes]:
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            yield chunk


async def upload(url: str, source: Path | bytes) -> None:
    """
    PUT *source* to the presigned *url*, streaming a file from disk.

    ``Content-Length`` is always sent: S3 refuses a chunked presigned PUT.
    """
    if isinstance(source, Path):
        size = source.stat().st_size
        if size > MAX_SINGLE_PUT:
            # shortcut: one presigned PUT, S3's limit; add multipart if
            # agents need to move larger single files.
            raise ValueError(
                f"{source} is {size} bytes; files over 5 GiB cannot be "
                "moved to or from an agent yet"
            )
        content: bytes | AsyncIterator[bytes] = _read_chunks(source)
    else:
        size = len(source)
        content = source
    log = horus_logger.log
    log.debug(f"Upload {size} bytes to {safe_url(url)}")
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.put(
            url, content=content, headers={"Content-Length": str(size)}
        )
        response.raise_for_status()
    log.debug(f"Uploaded {size} bytes in {time.monotonic() - started:.2f}s")


async def download_to(url: str, dest: Path) -> None:
    """GET the presigned *url* into *dest*, streamed, creating parents."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log = horus_logger.log
    log.debug(f"Download {safe_url(url)} to {dest}")
    started = time.monotonic()
    async with (
        httpx.AsyncClient(timeout=None) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        with dest.open("wb") as fh:
            async for chunk in response.aiter_bytes(_CHUNK):
                fh.write(chunk)
    log.debug(
        f"Downloaded {dest.stat().st_size} bytes to {dest} in "
        f"{time.monotonic() - started:.2f}s"
    )


async def download(url: str) -> bytes:
    """GET the presigned *url* into memory."""
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content
