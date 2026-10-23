import json
import os
from pathlib import Path
import socket
import sqlite3
import struct
import subprocess
import sys
import urllib.error

import pytest

from monitoring.kneeops import desktop


@pytest.fixture
def desktop_workspace(tmp_path, monkeypatch):
    for name in ('KNEE_WORKSPACE_ROOT', 'KNEEOPS_STATE_DIR', 'KNEEOPS_DOWNLOAD_DIR'):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / '개인 연구 공간'
    (root / 'config').mkdir(parents=True)
    (root / 'config/workspace.toml').write_text(
        'schema_version=1\ncompetition="rsna-knee-abnormality-detection"\n'
        '[paths]\ncache="storage/cache"\ndata="storage/competition"\n', encoding='utf-8')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    return root, root / '.local/kneeops', port


def command(action, root, state, port, **kwargs):
    return subprocess.run([sys.executable, '-X', 'utf8', '-m', 'monitoring.kneeops.desktop',
                           action, '--root', str(root), '--state', str(state), '--port', str(port), '--no-browser'],
                          cwd=Path(__file__).resolve().parents[2], capture_output=True,
                          encoding='utf-8', timeout=45, **kwargs)


def test_background_survives_launcher_exit_reuses_worker_and_stops(desktop_workspace):
    root, state, port = desktop_workspace
    try:
        launched = command('start', root, state, port)
        assert launched.returncode == 0, launched.stderr
        info = json.loads(launched.stdout)
        assert info['status'] == 'running'
        assert info['pid'] != os.getpid()
        # The launcher has exited; its detached child still answers HTTP.
        assert desktop.status(root, state, port)['run_id'] == info['run_id']
        with sqlite3.connect(state / 'app.sqlite') as con:
            con.execute("INSERT INTO jobs VALUES ('test-preserve', 'import', 'queued', 'now', NULL, '', NULL, NULL)")
        reused = command('start', root, state, port)
        assert reused.returncode == 0, reused.stderr
        assert json.loads(reused.stdout)['pid'] == info['pid']
        with sqlite3.connect(state / 'app.sqlite') as con:
            assert con.execute("SELECT status FROM jobs WHERE id='test-preserve'").fetchone()[0] == 'queued'
        stopped = command('stop', root, state, port)
        assert stopped.returncode == 0, stopped.stderr
        assert json.loads(stopped.stdout)['status'] == 'stopped'
        assert desktop.status(root, state, port)['status'] == 'stopped'
        assert not list(state.glob('desktop-*.stop'))
        assert (state / 'logs/background.log').is_file()
    finally:
        desktop.stop(root, state, port)


def test_occupied_port_never_initializes_or_changes_database(desktop_workspace):
    root, state, port = desktop_workspace
    with socket.socket() as sock:
        if os.name == 'nt':
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(('127.0.0.1', port)); sock.listen()
        result = command('run', root, state, port)
        assert result.returncode != 0
        assert not (state / 'app.sqlite').exists()


def test_other_workspace_and_foreground_server_are_not_stopped(desktop_workspace, monkeypatch):
    root, state, port = desktop_workspace
    def other(url):
        return {'app': 'kneeops'} if url.endswith('/health') else {'root': 'other', 'state': 'other'}
    monkeypatch.setattr(desktop, 'get_json', other)
    with pytest.raises(ValueError, match='Another workspace'):
        desktop.stop(root, state, port)
    def foreground(url):
        if url.endswith('/health'):
            return {'app': 'kneeops'}
        raise urllib.error.HTTPError(url, 404, 'not found', {}, None)
    monkeypatch.setattr(desktop, 'get_json', foreground)
    assert desktop.status(root, state, port)['status'] == 'foreground'
    with pytest.raises(ValueError, match=r'Ctrl\+C'):
        desktop.stop(root, state, port)
    assert not state.exists()


def close_requested_then_reset(port):
    """The server closes a socket the client has already reset.

    The client asks for `Connection: close` and resets before reading. On Windows'
    Proactor loop the server-side close then fails in sock.shutdown(), CPython skips
    Server._detach(), and uvicorn's Server.wait_closed() never returns.
    """
    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
    sock.sendall(b'GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n')
    sock.recv(4096)
    sock.sendall(b'GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n')
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('hh', 1, 0))
    sock.close()


def test_stop_completes_after_clients_reset_connections(desktop_workspace):
    root, state, port = desktop_workspace
    try:
        launched = command('start', root, state, port)
        assert launched.returncode == 0, launched.stderr
        pid = json.loads(launched.stdout)['pid']
        for _ in range(80):
            close_requested_then_reset(port)
        stopped = command('stop', root, state, port)
        assert stopped.returncode == 0, stopped.stderr
        assert json.loads(stopped.stdout)['status'] == 'stopped', stopped.stdout
        log = (state / 'logs/background.log').read_text(encoding='utf-8', errors='replace')
        assert 'Application shutdown complete' in log and 'ConnectionResetError' not in log
        # The process itself exited and released the state lock, not only the HTTP port.
        restarted = command('start', root, state, port)
        assert restarted.returncode == 0, restarted.stderr
        assert json.loads(restarted.stdout)['pid'] != pid
    finally:
        desktop.stop(root, state, port)
