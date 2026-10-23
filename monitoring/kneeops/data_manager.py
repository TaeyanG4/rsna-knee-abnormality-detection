"""Personal download plans and a cancellable, resumable worker independent of research sync."""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data_cache import cache_catalog
from .data_remote import safe_remote
from .store import dumps, now

SNAPSHOT = {'observed_at': '2026-10-01', 'source': 'Kaggle competition data summary API',
            'files': 819640, 'bytes': 569764475800, 'dicom_files': 819635,
            'metadata_bytes': 9151736, 'metadata_files': 5}
RESERVE = 1024 ** 3


def default_destination(store):
    return str(Path(os.environ['KNEEOPS_DOWNLOAD_DIR']).resolve()) if os.environ.get('KNEEOPS_DOWNLOAD_DIR') else str(store.ws.root / 'storage/downloads' / store.ws.config['competition'])


def destination(store, value):
    value = value.strip()
    if not value or '\x00' in value or value.startswith(('\\\\', '//')):
        raise ValueError('이 PC의 폴더 경로를 입력하세요.')
    path = store.ws.resolve(value)
    if path == Path(path.anchor) or path == store.ws.root or path == store.state:
        raise ValueError('드라이브/프로젝트/상태 폴더 자체 대신 다운로드 전용 하위 폴더를 지정하세요.')
    if path.exists() and not path.is_dir():
        raise ValueError('저장 경로가 폴더가 아닙니다.')
    disk(path)
    return path


def disk(path):
    parent = path
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError('경로의 상위 폴더를 찾을 수 없습니다.')
    try:
        usage = shutil.disk_usage(parent)
    except OSError as exc:
        raise ValueError('해당 드라이브를 사용할 수 없습니다.') from exc
    return {'free_bytes': usage.free, 'total_bytes': usage.total, 'volume_path': str(parent)}


def settings(store):
    configured = store.get_meta('data_destination', default_destination(store))
    try:
        capacity = disk(Path(configured))
    except ValueError:
        capacity = {'free_bytes': None, 'total_bytes': None, 'volume_path': configured}
    return {'destination': configured, 'default_destination': default_destination(store),
            'cache_manifest': store.get_meta('data_cache_manifest', ''), **capacity}


def studies(store, split='train', query='', limit=50):
    # Return IDs and series counts, never Report text or labels.
    if split not in {'train', 'test'}:
        raise ValueError('Invalid split')
    options = [store.ws.resolve('data') / f'{split}_series.csv', Path(settings(store)['destination']) / f'{split}_series.csv']
    counts = {}
    source = next((p for p in options if p.is_file()), None)
    if source:
        with source.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                uid = row['StudyInstanceUID']
                counts[uid] = counts.get(uid, 0) + 1
    filtered = [{'uid': uid, 'series': count} for uid, count in sorted(counts.items()) if query in uid]
    return {'items': filtered[:limit], 'total': len(filtered), 'metadata_available': source is not None}


def remote_catalog(request):
    try:
        process = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'monitoring.kneeops.data_remote', 'catalog'],
                                 input=dumps(request), capture_output=True, encoding='utf-8',
                                 timeout=180, cwd=Path(__file__).resolve().parents[2],
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired as exc:
        raise ValueError('파일 목록 조회가 3분을 초과했습니다. 선택 study 수를 줄여 다시 시도하세요.') from exc
    if process.returncode:
        raise ValueError('Kaggle 파일 목록을 조회하지 못했습니다. 이 PC의 Kaggle 인증과 네트워크를 확인하세요. 다운로드는 시작하지 않았습니다.')
    # Kaggle may emit a notice before the JSON result.
    for line in reversed(process.stdout.splitlines()):
        try:
            result = json.loads(line)
            if isinstance(result, dict) and 'files' in result:
                return result['files']
        except json.JSONDecodeError:
            pass
    raise ValueError('Kaggle 파일 목록 응답을 해석하지 못했습니다.')


def child(root, name):
    safe_remote(name)
    p = (root / name).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('파일 경로가 다운로드 폴더 밖을 가리킵니다.')
    return p


class DataManager:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kneeops-download')
        with store.connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS downloads(id TEXT PRIMARY KEY, plan TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, progress TEXT NOT NULL, error TEXT NOT NULL)')
            con.execute("UPDATE downloads SET status='interrupted',updated_at=? WHERE status IN ('queued','running','stopping')", (now(),))

    def list(self):
        items = self.store.rows('SELECT * FROM downloads ORDER BY created_at DESC LIMIT 20')
        for r in items:
            r['plan'] = json.loads(r['plan'])
            r['progress'] = json.loads(r['progress'])
            r['file_count'] = len(r['plan']['files'])
            r['plan'].pop('files')
            r['progress'].pop('completed', None)
        return items

    def get(self, id):
        records = self.store.rows('SELECT * FROM downloads WHERE id=?', (id,))
        if not records:
            raise KeyError(id)
        r = records[0]
        r['plan'], r['progress'] = json.loads(r['plan']), json.loads(r['progress'])
        return r

    def save(self, id, status, progress, error=''):
        with self.store.connect() as con:
            con.execute('UPDATE downloads SET status=?,updated_at=?,progress=?,error=? WHERE id=?',
                        (status, now(), dumps(progress), error, id))

    def preview(self, mode, selected):
        if mode not in {'metadata', 'selected', 'sample', 'all'}:
            raise ValueError('Unknown download mode')
        comp = self.store.ws.config['competition']
        dest = destination(self.store, settings(self.store)['destination'])
        if mode == 'selected' and (not selected or len(selected) > 5):
            raise ValueError('한 번에 학습 study 1~5개를 선택하세요.')
        if mode in {'sample', 'selected'}:
            if mode == 'sample':
                selected = [r['uid'] for r in studies(self.store, 'test')['items']]
                if not selected:
                    raise ValueError('먼저 CSV 메타데이터를 내려받으세요.')
            if any(not re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', u) for u in selected):
                raise ValueError('StudyInstanceUID 형식이 올바르지 않습니다.')
            selected = sorted(set(selected))
        if mode == 'all':
            files = [{'path': comp + '.zip', 'bytes': None}]
            estimate = SNAPSHOT['bytes']
        else:
            files = remote_catalog({'competition': comp, 'mode': mode, 'studies': selected,
                                    'split': 'test_series' if mode == 'sample' else 'train_series'})
            estimate = sum(f['bytes'] for f in files)
        paths = set()
        for f in files:
            safe_remote(f['path'])
            if f['path'] in paths:
                raise ValueError('중복 파일 목록입니다.')
            paths.add(f['path'])
            child(dest, f['path'])
        conflicts = [f['path'] for f in files if child(dest, f['path']).exists()]
        # Archive size is unknown; use raw bytes + 10% and keep 1 GiB spare.
        # Individual ZIP responses may temporarily require compressed + raw bytes.
        required = int(estimate * (1.1 if mode == 'all' else 2.1)) + RESERVE
        capacity = disk(dest)
        plan = {'mode': mode, 'competition': comp, 'destination': str(dest), 'files': files,
                'studies': selected if mode in {'sample', 'selected'} else [],
                'estimated_bytes': estimate, 'required_free_bytes': required, 'free_bytes': capacity['free_bytes'],
                'space_ok': capacity['free_bytes'] >= required, 'conflicts': conflicts[:20], 'conflict_count': len(conflicts),
                'estimate_note': '원본 크기 기준 보수적 추정. ZIP 실제 크기는 다를 수 있습니다. 자동 압축 해제 없음.' if mode == 'all' else '선택 study의 모든 시리즈·파일. 임시 압축 해제 공간과 1 GiB 여유 포함.',
                'source_observed_at': SNAPSHOT['observed_at'] if mode == 'all' else now()}
        id = str(uuid.uuid4())
        stamp = now()
        with self.store.connect() as con:
            con.execute('INSERT INTO downloads VALUES (?,?,?,?,?,?,?)', (id, dumps(plan), 'planned', stamp, stamp, '{}', ''))
        return {'id': id, **plan, 'file_count': len(files), 'files': files[:12]}

    def start(self, id):
        with self.lock:
            if self.store.rows("SELECT id FROM downloads WHERE status IN ('queued','running','stopping')"):
                raise ValueError('이미 다운로드가 진행 중입니다.')
            record = self.get(id)
            if record['status'] not in {'planned', 'cancelled', 'failed', 'interrupted'}:
                raise ValueError('이 계획은 다시 시작할 수 없습니다.')
            plan, progress = record['plan'], record['progress']
            dest = destination(self.store, plan['destination'])
            if str(dest) != plan['destination']:
                raise ValueError('저장 경로가 변경되었습니다. 새 미리보기를 만드세요.')
            done = progress.get('completed', {})
            for f in plan['files']:
                p = child(dest, f['path'])
                if f['path'] in done:
                    if not p.is_file() or p.stat().st_size != done[f['path']]:
                        raise ValueError('완료한 파일이 이동되거나 변경되었습니다. 새 경로에서 다시 계획하세요.')
                elif p.exists():
                    raise ValueError('같은 이름의 기존 파일을 보호하기 위해 중단했습니다. 다른 저장 폴더를 선택하세요.')
            # Partial bytes already occupy this volume. Requiring the original
            # full free-space budget would prevent a nearly complete ZIP resuming.
            staged_bytes = 0
            for i, f in enumerate(plan['files']):
                if f['path'] not in done:
                    for name in ('payload', 'unpacked'):
                        staged = child(dest, f'.kneeops/{id}/{i}/{name}')
                        if staged.is_file():
                            staged_bytes += staged.stat().st_size
            remaining = max(RESERVE, plan['required_free_bytes'] - sum(done.values()) - staged_bytes)
            if disk(dest)['free_bytes'] < remaining:
                raise ValueError('예상 필요 공간보다 드라이브 여유 공간이 부족합니다.')
            dest.mkdir(parents=True, exist_ok=True)
            # Fail now on a read-only mount, before launching any network transfer.
            probe = dest / ('.kneeops-write-' + uuid.uuid4().hex)
            try:
                with probe.open('x'):
                    pass
                probe.unlink()
            except OSError as exc:
                raise ValueError('저장 폴더에 쓸 수 없습니다. 권한 또는 Docker 쓰기 마운트를 확인하세요.') from exc
            self.cancel_event.clear()
            self.save(id, 'queued', progress)
            self.executor.submit(self._run, id)
        return {'id': id, 'status': 'queued'}

    def cancel(self, id):
        with self.lock:
            r = self.get(id)
            if r['status'] not in {'queued', 'running', 'stopping'}:
                raise ValueError('진행 중인 다운로드가 아닙니다.')
            self.cancel_event.set()
        return {'id': id, 'status': 'stopping'}

    def transfer(self, request, report):
        process = subprocess.Popen([sys.executable, '-X', 'utf8', '-m', 'monitoring.kneeops.data_remote', 'download'],
                                   stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   cwd=Path(__file__).resolve().parents[2],
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            process.stdin.write(dumps(request).encode('utf-8'))
            process.stdin.close()
            while process.poll() is None:
                if self.cancel_event.wait(.5):
                    raise InterruptedError('사용자가 다운로드를 중지했습니다.')
                report()
            if process.returncode:
                raise ValueError('전송에 실패했습니다. 인증·네트워크·공간을 확인한 뒤 재시도하세요. 미완료 파일은 보존했습니다.')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    def _run(self, id):
        r = self.get(id)
        p, progress = r['plan'], r['progress']
        dest = Path(p['destination'])
        done = progress.setdefault('completed', {})
        status, error = 'complete', ''
        try:
            for i, f in enumerate(p['files']):
                if self.cancel_event.is_set():
                    raise InterruptedError()
                if f['path'] in done:
                    continue
                target = child(dest, f['path'])
                staging = child(dest, f'.kneeops/{id}/{i}')
                staging.mkdir(parents=True, exist_ok=True)
                def report():
                    payload = staging / 'payload'
                    progress.update(current_file=f['path'], current_bytes=payload.stat().st_size if payload.exists() else 0,
                                    completed_files=len(done), completed_bytes=sum(done.values()), total_files=len(p['files']))
                    self.save(id, 'running', progress)
                    if disk(dest)['free_bytes'] < RESERVE:
                        raise ValueError('여유 공간이 1 GiB 미만이어서 전송을 중지했습니다. 미완료 파일은 보존했습니다.')
                report()
                receipt = staging / 'complete.json'
                if not receipt.exists():
                    self.transfer({'competition': p['competition'], 'mode': p['mode'], 'file': f, 'staging': str(staging)}, report)
                if self.cancel_event.is_set():
                    raise InterruptedError()
                meta = json.loads(receipt.read_text(encoding='utf-8'))
                output = child(staging, meta['file'])
                if output.stat().st_size != meta['bytes'] or (f['bytes'] is not None and meta['bytes'] != f['bytes']):
                    raise ValueError('완료 파일 크기 검증 실패')
                target.parent.mkdir(parents=True, exist_ok=True)
                # Atomic, same-volume promotion that NEVER replaces an existing file.
                os.link(output, target)
                done[f['path']] = meta['bytes']
                progress.update(completed_files=len(done), completed_bytes=sum(done.values()), current_bytes=0)
                self.save(id, 'running', progress)
                # Only our known temporary files, no recursive deletion.
                for name in ('payload', 'unpacked', 'complete.json', 'payload.kaggle-partial'):
                    (staging / name).unlink(missing_ok=True)
            progress.update(current_file='', current_bytes=0, completed_files=len(done), completed_bytes=sum(done.values()))
        except InterruptedError:
            status, error = 'cancelled', '중지됨. 미완료 파일을 보존했으며 서버가 지원하면 이어받습니다.'
        except Exception as exc:
            status, error = 'failed', str(exc)[:400]
        finally:
            self.save(id, status, progress, error)

    def close(self):
        self.cancel_event.set()
        self.executor.shutdown(wait=True)


def overview(store):
    return {'settings': settings(store), 'official': SNAPSHOT if store.ws.config['competition'] == 'rsna-knee-abnormality-detection' else None,
            'caches': cache_catalog(store)}
