"""Detached local launcher. Stop requests target a unique run, never an arbitrary PID."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser

from knee_workspace import Workspace


def paths(root=None, state=None):
    ws = Workspace.load(root)
    folder = Path(state or os.environ.get('KNEEOPS_STATE_DIR') or
                  ws.root / ws.config.get('dashboard', {}).get('state_dir', '.local/kneeops')).resolve()
    return ws.root, folder


def get_json(url):
    # A machine-wide proxy must not redirect local process control.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=2) as response:
        return json.load(response)


def status(root, state, port):
    url = f'http://127.0.0.1:{port}'
    try:
        health = get_json(url + '/api/health')
    except urllib.error.HTTPError as exc:
        raise ValueError(f'Port {port} is occupied by another service.') from exc
    except urllib.error.URLError:
        return {'status': 'stopped', 'url': url}
    if health.get('app') != 'kneeops':
        raise ValueError(f'Port {port} is occupied by another service.')
    try:
        info = get_json(url + '/api/desktop')
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        return {'status': 'foreground', 'url': url,
                'message': 'Existing console server reused. Keep its console open until it is stopped.'}
    if info.get('root') != str(root) or info.get('state') != str(state):
        raise ValueError('Another workspace is already using this port. Its process was not changed.')
    return {**info, 'status': 'running', 'url': url}


@contextmanager
def state_lock(state):
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'desktop.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('A background server already owns this state directory.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run(root, state, port):
    import uvicorn
    from . import serving
    from .api import create_app

    with state_lock(state), socket.socket() as listener:
        if os.name == 'nt':
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # Claim the port BEFORE initializing workers or changing stale DB job states.
        listener.bind(('127.0.0.1', port))
        listener.listen(2048)
        run_id = uuid.uuid4().hex
        # We hold the state lock, so leftover stop files belong to runs that no longer exist.
        for stale in state.glob('desktop-*.stop'):
            stale.unlink(missing_ok=True)
        stop_file = state / f'desktop-{run_id}.stop'
        app = create_app(root, state)
        info = {'root': str(root), 'state': str(state), 'pid': os.getpid(), 'run_id': run_id}

        @app.get('/api/desktop')
        def desktop_info():
            return info

        server = uvicorn.Server(serving.config(app, access_log=False, log_level='info'))
        done = threading.Event()

        def watch_stop():
            while not done.wait(.25):
                if stop_file.exists():
                    server.should_exit = True
                    return

        watcher = threading.Thread(target=watch_stop, daemon=True)
        watcher.start()
        try:
            server.run(sockets=[listener])
        finally:
            done.set()
            watcher.join(timeout=1)
            stop_file.unlink(missing_ok=True)


def start(root, state, port, browser=True):
    info = status(root, state, port)
    if info['status'] != 'stopped':
        if browser:
            webbrowser.open(info['url'])
        return info
    logs = state / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / 'background.log'
    command = [sys.executable, '-X', 'utf8', '-m', 'monitoring.kneeops.desktop',
               'run', '--root', str(root), '--state', str(state), '--port', str(port)]
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
    with log_path.open('ab', buffering=0) as log:
        child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True, **options)
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        info = status(root, state, port)
        if info['status'] != 'stopped':
            if browser:
                webbrowser.open(info['url'])
            return {**info, 'log': str(log_path)}
        if child.poll() is not None:
            raise ValueError(f'Background startup failed. See {log_path}')
        time.sleep(.25)
    raise ValueError(f'Startup is taking longer than expected. See {log_path}; do not start a second server.')


def stop(root, state, port):
    info = status(root, state, port)
    if info['status'] == 'stopped':
        return info
    if info['status'] != 'running':
        raise ValueError('This server was started in a console. Stop it with Ctrl+C in that console.')
    run_id = info['run_id']
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
        raise ValueError('Invalid background run identity.')
    (state / f'desktop-{run_id}.stop').touch()
    # Wait for the state lock to release, not just HTTP shutdown: workers may still be closing.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with state_lock(state):
                return {'status': 'stopped', 'url': info['url']}
        except ValueError:
            time.sleep(.25)
    return {'status': 'stopping', 'message': 'Finishing an active worker. Partial downloads are preserved.'}


def main():
    parser = argparse.ArgumentParser(description='Start/stop KneeOps without keeping a console open')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'run'])
    parser.add_argument('--root')
    parser.add_argument('--state')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        root, state = paths(args.root, args.state)
        if not 1 <= args.port <= 65535:
            raise ValueError('Port must be between 1 and 65535.')
        if args.action == 'run':
            run(root, state, args.port)
            return
        if args.action == 'start':
            result = start(root, state, args.port, not args.no_browser)
        elif args.action == 'stop':
            result = stop(root, state, args.port)
        else:
            result = status(root, state, args.port)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
