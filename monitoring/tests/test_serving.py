import asyncio
import inspect
import os
import socket
import threading
import time

import uvicorn

from monitoring.kneeops import serving


def test_windows_uses_selector_loop_and_bounded_shutdown():
    config = serving.config(lambda scope, receive, send: None)
    assert config.timeout_graceful_shutdown == serving.GRACEFUL_SHUTDOWN_SECONDS
    if os.name == 'nt':
        assert config.get_loop_factory() is asyncio.SelectorEventLoop


def test_shutdown_finishes_even_if_a_connection_never_detaches(monkeypatch):
    """Models the Proactor leak: the asyncio server keeps an active connection forever."""
    monkeypatch.setattr(serving, 'GRACEFUL_SHUTDOWN_SECONDS', 1)
    events = []

    async def app(scope, receive, send):
        if scope['type'] == 'lifespan':
            while True:
                message = await receive()
                if message['type'] == 'lifespan.startup':
                    await send({'type': 'lifespan.startup.complete'})
                elif message['type'] == 'lifespan.shutdown':
                    events.append('shutdown')
                    await send({'type': 'lifespan.shutdown.complete'})
                    return

    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(serving.config(app, host='127.0.0.1', port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(.05)
    assert server.started
    leaked = server.servers[0]
    attach = leaked._attach
    # CPython 3.12 takes no argument; 3.13+ passes the transport.
    attach() if not inspect.signature(attach).parameters else attach(object())
    started = time.monotonic()
    server.should_exit = True
    thread.join(timeout=8)
    assert not thread.is_alive(), 'shutdown waited forever on a connection that never detached'
    assert events == ['shutdown'] and time.monotonic() - started < 8
