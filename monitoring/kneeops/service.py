from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from .collectors import collect_notebook, collect_topic, sync_kaggle
from .importer import import_local
from .store import Store, dumps, now


class Service:
    """One durable queue per local web process. No GPU or publishing operations."""
    def __init__(self, store: Store):
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kneeops')
        self.lock = threading.Lock()
        self.stop = threading.Event()
        with store.connect() as con:
            con.execute("UPDATE jobs SET status='interrupted',finished_at=?,error='App stopped before completion; retry explicitly' WHERE status IN ('queued','running')", (now(),))
        self.scheduler = threading.Thread(target=self._schedule, daemon=True)
        self.scheduler.start()

    def enqueue(self, kind, ref=''):
        if kind not in {'import', 'sync', 'notebook', 'topic', 'history', 'scheduled', 'data_index'}:
            raise ValueError('Unknown job kind')
        with self.lock:
            if self.store.rows("SELECT id FROM jobs WHERE status IN ('queued','running')"):
                raise ValueError('다른 수집 작업이 진행 중입니다. 완료 후 다시 실행하세요.')
            jid = str(uuid.uuid4())
            with self.store.connect() as con:
                con.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)', (jid, kind, 'queued', now(), None, ref, None, None))
            self.executor.submit(self._run, jid, kind, ref)
        return {'id': jid, 'status': 'queued'}

    def _run(self, jid, kind, ref):
        def progress(value):
            with self.store.connect() as con:
                con.execute("UPDATE jobs SET status='running',progress=? WHERE id=?", (value, jid))
        try:
            progress(kind)
            if kind == 'import': result = import_local(self.store)
            elif kind == 'data_index':
                from .data_library import build_library
                result = build_library(self.store, progress)
            elif kind in {'sync','scheduled'}:
                result = sync_kaggle(self.store, progress)
                if kind=='scheduled':
                    from .history import collect_history
                    prefs=json.loads(self.store.get_meta('preferences','{}'))
                    for comp in prefs.get('sync_history',[])[:5]:
                        history=collect_history(self.store,comp,progress)
                        result['errors'].extend(history.get('errors',[]))
            elif kind == 'notebook': result = {'revision': collect_notebook(self.store, ref)}
            elif kind == 'history':
                from .history import collect_history
                result = collect_history(self.store, ref, progress)
            else: result = {'revision': collect_topic(self.store, int(ref))}
            status = 'partial' if result.get('errors') else 'complete'
            with self.store.connect() as con:
                con.execute('UPDATE jobs SET status=?,finished_at=?,result=?,progress=? WHERE id=?', (status, now(), dumps(result), status, jid))
        except Exception as exc:
            with self.store.connect() as con:
                con.execute("UPDATE jobs SET status='failed',finished_at=?,error=? WHERE id=?", (now(), str(exc)[:1000], jid))

    def _schedule(self):
        while not self.stop.wait(10):
            self.schedule_once()

    def schedule_once(self, clock=None):
        status = schedule_status(self.store, clock)
        if status['enabled'] and status['due']:
            try:
                self.enqueue('scheduled')
            except ValueError:
                return False  # Another job is running; check again on the next tick.
            self.store.set_meta('last_scheduled_enqueue', (clock or datetime.now(timezone.utc)).isoformat())
            return True
        return False

    def close(self):
        self.stop.set()
        self.scheduler.join(timeout=2)
        self.executor.shutdown(wait=True)


def _parse(value):
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def schedule_status(store, clock=None):
    """Interval schedule anchored to the last sync attempt, so restarts do not reset it.

    The last scheduled enqueue also counts, so a run that crashed before recording
    its attempt waits a full interval instead of retrying every tick.
    """
    clock = clock or datetime.now(timezone.utc)
    prefs = json.loads(store.get_meta('preferences', '{}'))
    minutes = max(15, min(1440, int(prefs.get('sync_minutes', 60))))
    anchors = [t for t in (_parse(store.get_meta('last_sync_attempt')), _parse(store.get_meta('last_scheduled_enqueue'))) if t]
    last = max(anchors) if anchors else None
    enabled = bool(prefs.get('sync_enabled'))
    next_due = (last + timedelta(minutes=minutes)) if last else clock
    running = store.rows("SELECT kind,progress,started_at FROM jobs WHERE status IN ('queued','running') ORDER BY started_at DESC LIMIT 1")
    return {'enabled': enabled, 'minutes': minutes, 'last_sync_attempt': store.get_meta('last_sync_attempt'),
            'next_due': next_due.isoformat() if enabled else None, 'due': enabled and clock >= next_due,
            'running': running[0] if running else None, 'history': prefs.get('sync_history', [])[:5],
            'note': '프로그램이 실행 중일 때만 수집합니다. 꺼져 있던 동안의 누락분은 다시 켜진 뒤 한 번만 수집합니다.'}


def backup(store):
    directory = store.state / 'backups'
    directory.mkdir(exist_ok=True)
    path = directory / (now().replace(':', '-').replace('+', '_') + '.sqlite')
    with store.connect() as source, sqlite3.connect(path) as dest:
        source.backup(dest)
    return path
