#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# MIT License
#
"""
Forward ``127.0.0.1:PORT`` to ``HOST:PORT``, so an agent in a container can
reach a plain-HTTP test broker on the host through a loopback URL.

Usage: python forward.py <port> <host>
"""

import asyncio
import sys


async def _pipe(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()


async def _serve(port: int, host: str) -> None:
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        up_reader, up_writer = await asyncio.open_connection(host, port)
        await asyncio.gather(
            _pipe(reader, up_writer), _pipe(up_reader, writer)
        )

    server = await asyncio.start_server(handle, "127.0.0.1", port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(_serve(int(sys.argv[1]), sys.argv[2]))
