"""Uvicorn settings shared by the background launcher and the console server.

Windows shutdown hang (reproduced 2026-10-02, CPython 3.12.6, uvicorn 0.54): on the
Proactor loop, when the server closes a socket the peer already reset,
`_ProactorBasePipeTransport._call_connection_lost` raises in `sock.shutdown()` before
`Server._detach()`. The asyncio server's active count never reaches zero, so
`Server.wait_closed()` inside uvicorn's shutdown never returns and the process keeps the
state lock. The selector loop has no such path; the app starts no asyncio subprocesses.
A bounded graceful shutdown is kept as a second guard.
"""
from __future__ import annotations

import os

GRACEFUL_SHUTDOWN_SECONDS = 10


def loop_setting() -> str:
    return 'asyncio:SelectorEventLoop' if os.name == 'nt' else 'auto'


def config(app, **kwargs):
    import uvicorn
    return uvicorn.Config(app, loop=loop_setting(), timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS, **kwargs)
