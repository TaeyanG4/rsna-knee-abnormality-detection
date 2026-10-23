"""Disk-backed ZIP inventory and lossless, study-at-a-time materialization.

The central directory is indexed once. Reading one study then seeks straight to
its local headers using ZipInfo records; it never scans/loads all image pixels.
"""
from __future__ import annotations

import hashlib
import csv
import json
import os
import re
import shutil
import sqlite3
import stat
import uuid
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path

from .store import dumps, now

UID = re.compile(r'^[0-9]+(?:\.[0-9]+)+$')
CSV_NAMES = {'train.csv', 'test.csv', 'train_series.csv', 'test_series.csv', 'sample_submission.csv'}
RESERVE = 1024 ** 3


def signature(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


def uri(store, path):
    try:
        return store.ws.uri(path)
    except ValueError:
        return str(path.resolve())


def bounded(root, relative):
    root = root.resolve()
    target = root / relative
    if not target.resolve().is_relative_to(root) or target.resolve() == root:
        raise ValueError('원본 경로가 준비 폴더 밖을 가리킵니다.')
    # Do not follow junctions or links, even to another location within root.
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink() or getattr(parent, 'is_junction', lambda: False)():
            raise ValueError('원본 준비 경로에 링크가 있습니다.')
    return target


@contextmanager
def exclusive(path):
    """Cross-process lock, released by the OS after crash/shutdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('같은 원본을 준비 중입니다. 완료 후 다시 여세요.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def member_parts(info):
    name = info.filename
    parts = name.split('/')
    if (info.orig_filename != name or '\\' in name or ':' in name or '\x00' in name or
            any(p in ('', '.', '..') or p.endswith((' ', '.')) for p in parts) or
            stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1):
        raise ValueError(f'안전하지 않은 ZIP 항목: {name[:140]}')
    if name in CSV_NAMES:
        return '', '', ''
    if (len(parts) != 4 or parts[0] not in ('train_series', 'test_series') or
            not UID.fullmatch(parts[1]) or not UID.fullmatch(parts[2]) or
            not re.fullmatch(r'[0-9.]+\.(?:dcm|dicom)', parts[3], re.I)):
        raise ValueError(f'지원하지 않는 원본 ZIP 구조: {name[:140]}')
    if info.file_size <= 0 or info.file_size > 64 * 1024 ** 2:
        raise ValueError('DICOM 항목 크기가 지원 범위를 벗어납니다.')
    return parts[0].removesuffix('_series'), parts[1], parts[2]


ZIP_FIELDS = ('CRC', 'compress_type', 'compress_size', 'file_size', 'header_offset',
              'flag_bits', 'extract_version', 'create_system', '_end_offset')


class IndexedZip(zipfile.ZipFile):
    """Only use with our signature-checked index and explicit ZipInfo, never names."""
    def _RealGetContents(self):
        self.start_dir = os.fstat(self.fp.fileno()).st_size


def zip_info(row):
    info = zipfile.ZipInfo(row['path'])
    for key, value in json.loads(row['info']).items():
        setattr(info, key, value)
    return info


@contextmanager
def connect(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        with con:
            yield con
    finally:
        con.close()


def index_archive(store, path, progress=lambda text: None):
    path = Path(path).resolve()
    sig = signature(path)
    key = hashlib.sha256(dumps([uri(store, path), sig]).encode()).hexdigest()[:24]
    directory = store.state / 'archive-index'; directory.mkdir(exist_ok=True)
    target = directory / (key + '.sqlite')
    with exclusive(directory / (key + '.lock')):
        if target.is_file():
            with connect(target) as con:
                return json.loads(con.execute('SELECT payload FROM metadata').fetchone()[0])
        partial = directory / (key + '.building.sqlite')
        # This fixed staging database is owned solely by the indexer under lock.
        if partial.exists():
            partial.unlink()
        try:
            with connect(partial) as con, zipfile.ZipFile(path) as archive:
                con.executescript('''
                    CREATE TABLE files(path TEXT PRIMARY KEY COLLATE NOCASE, split TEXT, uid TEXT,
                        series TEXT, bytes INTEGER, info TEXT);
                    CREATE TABLE metadata(payload TEXT);
                    CREATE TABLE prepared(uid TEXT, split TEXT, payload TEXT, PRIMARY KEY(uid,split));
                ''')
                digest = hashlib.sha256(); count = total = 0
                for info in archive.infolist():
                    # Official archive contains files only; reject unexpected directory/link entries.
                    split, uid, series = member_parts(info)
                    record = dumps({key: getattr(info, key) for key in ZIP_FIELDS})
                    con.execute('INSERT INTO files VALUES (?,?,?,?,?,?)',
                                (info.filename, split, uid, series, info.file_size, record))
                    digest.update(dumps([info.filename, record]).encode())
                    count += 1; total += info.file_size
                    if count % 25000 == 0:
                        progress(f'전체 ZIP 목록 확인: {count:,}개 (영상 해제 전)')
                con.execute('CREATE INDEX study_files ON files(split,uid)')
                if signature(path) != sig:
                    raise ValueError('목록 생성 중 ZIP이 변경되었습니다.')
                metadata = {'id': key, 'path': uri(store, path), 'signature': sig,
                    'index': str(target), 'directory_sha256': digest.hexdigest(),
                    'indexed_at': now(), 'files': count, 'bytes': total, 'zip_bytes': sig[0],
                    'prepared_root': uri(store, path.parent / 'prepared' / key),
                    'validation': 'central directory indexed; CRC checked only for extracted studies'}
                con.execute('INSERT INTO metadata VALUES (?)', (dumps(metadata),))
            os.replace(partial, target)
            return metadata
        except (zipfile.BadZipFile, sqlite3.IntegrityError) as exc:
            raise ValueError('ZIP 목록이 손상되었거나 중복 경로가 있습니다.') from exc


def archive_studies(store, archive):
    result = []
    with connect(archive['index']) as con:
        for row in con.execute("SELECT split,uid,COUNT(*) files,SUM(bytes) bytes,COUNT(DISTINCT series) series FROM files WHERE uid<>'' GROUP BY split,uid ORDER BY split,uid"):
            key = hashlib.sha256((archive['id'] + row['split'] + row['uid']).encode()).hexdigest()[:24]
            path = store.ws.resolve(archive['prepared_root']) / (row['split'] + '_series') / row['uid']
            result.append({**dict(row), 'id': key, 'archive_id': archive['id'], 'status': 'archived',
                'path': uri(store, path), 'root': archive['prepared_root'],
                'expected_files': row['files'], 'expected_series': row['series'],
                'missing_files': 0, 'missing_series': 0, 'invalid_sizes': 0,
                'training_allowed': False, 'storage': 'zip',
                'purpose': 'train-candidate' if row['split'] == 'train' else 'public-test-inference-only'})
    return result


def available_archives(store, roots, progress):
    name = store.ws.config['competition'] + '.zip'
    archives = []
    for path in sorted({root / name for root in roots if (root / name).is_file()}):
        archives.append(index_archive(store, path, progress))
    return archives


def validate_archive(store, archive):
    path = store.ws.resolve(archive['path'])
    if not path.is_file() or signature(path) != archive['signature']:
        raise ValueError('원본 ZIP이 이동되거나 변경되었습니다. 로컬 목록을 새로고침하세요.')
    return path


def hydrate(store, result):
    prepared = {}
    for archive in result.get('archives', []):
        archive['available'] = (store.ws.resolve(archive['path']).is_file() and
                                signature(store.ws.resolve(archive['path'])) == archive['signature'])
        with connect(archive['index']) as con:
            for row in con.execute('SELECT payload FROM prepared'):
                item = json.loads(row[0])
                if store.ws.resolve(item['path']).is_dir():
                    prepared[item['id']] = item
    for row in result['studies']:
        if row['id'] in prepared:
            row.update(prepared[row['id']])
    result['prepared_studies'] = len(prepared)
    result['prepared_bytes'] = sum(row['bytes'] for row in prepared.values())
    return result


def verify_prepared_metadata(store, item):
    """Export gate: extraction receipts alone must not authorize missing training input."""
    if item.get('storage') != 'extracted':
        return item
    root = store.ws.resolve(item['root'])
    receipt = store.ws.resolve(item['receipt'])
    saved = json.loads(receipt.read_text(encoding='utf-8')) if receipt.is_file() else {}
    valid = len(saved) == item['expected_files']
    for relative, record in saved.items():
        path = bounded(root, relative)
        if not path.is_file() or signature(path) != record['signature']:
            valid = False
            break
    return {**item, 'training_allowed': valid and item['split'] == 'train',
            'status': 'complete' if valid else 'partial', 'checked_at': now(),
            'validation': 'CRC + SHA256 at extraction; size/mtime receipts checked at export; hashes not recomputed'}


def materialize(store, item, archive, progress=lambda text: None):
    path = validate_archive(store, archive)
    directory = store.state / 'archive-index'
    with exclusive(directory / (item['id'] + '.extract.lock')):
        root = store.ws.resolve(archive['prepared_root'])
        # The root itself must not have been replaced by a junction/symlink.
        bounded(path.parent, root.relative_to(path.parent))
        root.mkdir(parents=True, exist_ok=True)
        receipts = directory / (item['id'] + '.receipt.json')
        saved = json.loads(receipts.read_text(encoding='utf-8')) if receipts.exists() else {}
        with connect(archive['index']) as con:
            records = con.execute('SELECT * FROM files WHERE split=? AND uid=? ORDER BY path',
                                  (item['split'], item['uid'])).fetchall()
        if not records:
            raise ValueError('ZIP 안에 해당 study가 없습니다.')
        required = sum(r['bytes'] for r in records if not bounded(root, r['path']).is_file())
        if shutil.disk_usage(root).free < required + RESERVE:
            raise ValueError('선택 study를 준비할 공간이 부족합니다 (여유 1 GiB 유지).')
        with IndexedZip(path) as archive_file:
            for n, record in enumerate(records):
                target = bounded(root, record['path'])
                prior = saved.get(record['path'])
                if target.is_file() and prior and prior['signature'] == signature(target):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + '.kneeops-partial')
                bounded(root, partial.relative_to(root))
                if partial.exists():
                    partial.unlink()  # only our incomplete per-member staging file
                digest = hashlib.sha256(); length = 0
                try:
                    with archive_file.open(zip_info(record)) as src, partial.open('xb') as out:
                        while chunk := src.read(1024 ** 2):
                            length += len(chunk)
                            if length > record['bytes'] or shutil.disk_usage(root).free < RESERVE:
                                raise ValueError('영상 크기 또는 여유 공간 검사가 실패했습니다.')
                            digest.update(chunk); out.write(chunk)
                        out.flush(); os.fsync(out.fileno())
                    if length != record['bytes']:
                        raise ValueError('압축 해제한 파일 크기가 목록과 다릅니다.')
                    if target.exists():
                        with target.open('rb') as src:
                            match = hashlib.file_digest(src, 'sha256').hexdigest() == digest.hexdigest()
                        if not match:
                            raise ValueError('준비 폴더의 기존 파일과 원본이 다릅니다. 기존 파일을 보존했습니다.')
                    else:
                        os.link(partial, target)  # never overwrite an existing file
                    saved[record['path']] = {'bytes': length, 'sha256': digest.hexdigest(),
                                             'signature': signature(target), 'crc32': zip_info(record).CRC}
                    temp = receipts.with_suffix('.tmp')
                    temp.write_text(dumps(saved), encoding='utf-8'); os.replace(temp, receipts)
                except (zipfile.BadZipFile, zlib.error) as exc:
                    raise ValueError('원본 ZIP CRC 검사 실패: ' + record['path']) from exc
                finally:
                    if partial.exists():
                        partial.unlink()
                progress(f'원본 study 준비: {n + 1}/{len(records)} 파일')
        validate_archive(store, archive)
        result = {**item, 'status': 'complete', 'storage': 'extracted', 'prepared_at': now(),
                  'training_allowed': item['split'] == 'train', 'validation': 'all study members CRC + SHA256 on extraction',
                  'receipt': uri(store, receipts)}
        with connect(archive['index']) as con:
            con.execute('INSERT OR REPLACE INTO prepared VALUES (?,?,?)', (item['uid'], item['split'], dumps(result)))
        return store.ws.resolve(result['path']), result


def structure(store, raw, series='', offset=0):
    """Metadata only: no extraction is triggered by browsing the tree."""
    from .data_library import library
    data = library(store)
    item = next((r for r in data['studies'] if r['id'] == raw), None)
    if not item or not item.get('archive_id'):
        raise ValueError('전체 ZIP 원본 study를 선택하세요.')
    archive = next(a for a in data['archives'] if a['id'] == item['archive_id'])
    validate_archive(store, archive)
    descriptions = {}
    metadata = store.ws.resolve('data') / (item['split'] + '_series.csv')
    if metadata.is_file():
        with metadata.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['StudyInstanceUID'] == item['uid']:
                    plane = row.get('Anatomical_Plane', '')
                    fluid = {'1': '수분 민감', '0': '수분 비민감'}.get(row.get('Fluid_Sensitive'), '수분 민감도 미표기')
                    fat = {'1': '지방억제', '0': '지방억제 없음'}.get(row.get('Fat_Suppression'), '지방억제 미표기')
                    descriptions[row['SeriesInstanceUID']] = {
                        'description': row.get('SeriesDescription') or ' · '.join(filter(None, (plane, fluid, fat))),
                        'plane': plane, 'fluid_sensitive': row.get('Fluid_Sensitive'),
                        'fat_suppression': row.get('Fat_Suppression')}
    with connect(archive['index']) as con:
        rows = con.execute('SELECT series,COUNT(*) files,SUM(bytes) bytes FROM files WHERE split=? AND uid=? GROUP BY series ORDER BY series',
                           (item['split'], item['uid'])).fetchall()
        series_rows = [{**dict(r), **descriptions.get(r['series'], {'description': ''})} for r in rows]
        files = []
        total = 0
        if series:
            total = next((r['files'] for r in series_rows if r['series'] == series), 0)
            files = [dict(r) for r in con.execute('SELECT path,bytes FROM files WHERE split=? AND uid=? AND series=? ORDER BY path LIMIT 30 OFFSET ?',
                                                 (item['split'], item['uid'], series, offset))]
    return {'study': item, 'series': series_rows, 'files': files, 'total': total, 'offset': offset,
            'description_source': uri(store, metadata), 'archive': archive['path']}
