"""Local raw-data inventory and portable training inputs. Never treat test data as training."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from .data_cache import cache_catalog
from .store import dumps, now

UID = re.compile(r'^[0-9]+(?:\.[0-9]+)+$')


def portable(store, path):
    try:
        return store.ws.uri(path)
    except ValueError:
        return str(Path(path).resolve())


def library(store):
    result = json.loads(store.get_meta('data_library', '{"indexed_at":null,"studies":[],"roots":[],"files":0,"bytes":0}'))
    if result.get('archives'):
        from .data_archive import hydrate
        hydrate(store, result)
    return result


def expected_files(store, root):
    expected = {}
    with store.connect() as con:
        exists = con.execute("SELECT 1 FROM sqlite_master WHERE name='downloads'").fetchone()
        plans = con.execute('SELECT plan FROM downloads ORDER BY created_at').fetchall() if exists else []
    for row in plans:
        plan = json.loads(row['plan'])
        if Path(plan['destination']).resolve() == root:
            for item in plan['files']:
                parts = item['path'].split('/')
                if len(parts) == 4 and parts[0] in ('train_series', 'test_series'):
                    expected[item['path']] = item['bytes']
    return expected


def input_descriptor(store):
    data = library(store)
    if not data.get('descriptor'):
        raise ValueError('로컬 원본 목록을 먼저 확인하세요.')
    descriptor = json.loads(Path(data['descriptor']).read_text(encoding='utf-8'))
    if data.get('archives'):
        from .data_archive import verify_prepared_metadata
        descriptor.update(raw=[verify_prepared_metadata(store, r) for r in data['studies']], archives=data['archives'], observed_at=now(),
                          prepare_command='python -m monitoring.kneeops data prepare --raw <id> --json')
    return descriptor


def expected_series(store, split, root):
    options = [store.ws.resolve('data') / (split + '_series.csv'), root / (split + '_series.csv')]
    path = next((p for p in options if p.is_file()), None)
    result = {}
    if path:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                result.setdefault(row['StudyInstanceUID'], set()).add(row['SeriesInstanceUID'])
    return result


def build_library(store, progress=lambda text: None):
    from .data_manager import settings
    roots = {store.ws.resolve('data'), Path(settings(store)['destination']).resolve()}
    roots.update(store.ws.resolve(value) for value in json.loads(store.get_meta('data_prepared_roots', '[]')))
    items, manifests = [], []
    stamp = now()
    for root in sorted(roots):
        if not root.is_dir():
            continue
        expected = expected_files(store, root)
        for split in ('train', 'test'):
            base = root / (split + '_series')
            planned = {}
            for relative, size in expected.items():
                parts = relative.split('/')
                if parts[0] == split + '_series':
                    planned.setdefault(parts[1], {})[relative] = size
            studies = {p.name: p for p in base.iterdir() if p.is_dir() and not p.is_symlink() and UID.fullmatch(p.name)} if base.is_dir() else {}
            series_map = expected_series(store, split, root)
            for uid in sorted(set(studies) | set(planned)):
                folder = base / uid
                actual, series_found, invalid, file_records = {}, {}, 0, []
                for series in sorted(folder.iterdir()) if folder.is_dir() else []:
                    if not series.is_dir() or series.is_symlink() or not UID.fullmatch(series.name):
                        continue
                    for file in sorted(series.iterdir()):
                        if file.is_symlink() or not file.is_file() or file.suffix.lower() not in ('.dcm', '.dicom'):
                            continue
                        relative = file.relative_to(root).as_posix()
                        size = file.stat().st_size
                        wanted = planned.get(uid, {}).get(relative)
                        ok = size > 0 and (wanted is None or wanted == size)
                        invalid += int(not ok)
                        actual[relative] = size
                        series_found[series.name] = series_found.get(series.name, 0) + 1
                        file_records.append({'path': portable(store, file), 'bytes': size, 'expected_bytes': wanted, 'size_ok': ok})
                plan = planned.get(uid, {})
                known_series = series_map.get(uid, set())
                missing_files = len(set(plan) - set(actual)) if plan else None
                missing_series = len(known_series - set(series_found)) if known_series else None
                state = 'missing' if not actual else 'partial' if invalid or missing_files or missing_series else 'complete' if plan else 'unverified'
                identity = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:24]
                item = {'id': identity, 'uid': uid, 'split': split, 'path': portable(store, folder),
                        'root': portable(store, root), 'files': len(actual), 'bytes': sum(actual.values()),
                        'expected_files': len(plan) if plan else None, 'missing_files': missing_files,
                        'series': len(series_found), 'expected_series': len(known_series) if known_series else None,
                        'missing_series': missing_series, 'invalid_sizes': invalid, 'status': state,
                        'training_allowed': split == 'train' and state == 'complete',
                        'purpose': 'train-candidate' if split == 'train' else 'public-test-inference-only'}
                items.append(item)
                manifests.append({**item, 'series_files': series_found, 'file_manifest': file_records})
                progress(f'로컬 원본 목록 확인: {len(items)} study · {sum(x["files"] for x in items):,}개 파일')
    from .data_archive import available_archives, archive_studies
    archives = available_archives(store, roots, progress)
    zipped = [item for archive in archives for item in archive_studies(store, archive)]
    covered = {(row['split'], row['uid']) for row in zipped}
    loose_files = sum(row['files'] for row in items)
    # One study row for the complete archive; retain loose copies in the immutable descriptor.
    items = [row for row in items if (row['split'], row['uid']) not in covered] + zipped
    cache_inputs = [{k: c[k] for k in ('manifest', 'contract', 'studies', 'shape', 'manifest_sha256', 'uid_sha256')}
                    for c in cache_catalog(store) if c['valid']]
    for cache in cache_inputs:
        cache['manifest'] = portable(store, cache['manifest'])
    descriptor = {'schema': 1, 'created_at': stamp, 'competition': store.ws.config['competition'],
                  'metadata_root': portable(store, store.ws.resolve('data')), 'raw': manifests + zipped, 'caches': cache_inputs,
                  'archives': archives,
                  'rules': ['test_series is for inference/I-O validation only; no labels or Report for training',
                            'Use fixed grouped folds and explicit label/mask contracts; Gold58 is exposed',
                            'Local counts and expected sizes are not full pixel/CRC/SHA validation',
                            'Do not train on incomplete raw study directories',
                            'Resolve workspace:// paths through knee_workspace; do not hard-code a drive']}
    out = store.state / 'data-library'
    out.mkdir(exist_ok=True)
    snapshot = out / (stamp.replace(':', '-').replace('+', '_') + '.json')
    with snapshot.open('x', encoding='utf-8') as handle:
        handle.write(dumps(descriptor))
    result = {'indexed_at': stamp, 'studies': items, 'files': sum(x['files'] for x in items),
              'archives': archives, 'loose_files': loose_files,
              'bytes': sum(x['bytes'] for x in items), 'roots': [portable(store, p) for p in sorted(roots) if p.is_dir()],
              'descriptor': str(snapshot), 'expected_files': sum(x['expected_files'] or 0 for x in items),
              'cache_studies': max((c['studies'] for c in cache_inputs), default=0)}
    store.set_meta('data_library', dumps(result))
    return library(store)


def study_path(store, id):
    data = library(store)
    item = next((row for row in data['studies'] if row['id'] == id and row['files']), None)
    if item is None:
        raise ValueError('로컬 원본 목록을 새로고침한 뒤 내려받은 study를 선택하세요.')
    if item.get('archive_id'):
        from .data_archive import materialize
        archive = next(a for a in data['archives'] if a['id'] == item['archive_id'])
        return materialize(store, item, archive)
    folder = store.ws.resolve(item['path'])
    if not folder.is_dir():
        raise ValueError('원본 폴더가 이동되었습니다. 로컬 목록을 새로고침하세요.')
    return folder, item
