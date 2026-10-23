"""Bounded, read-only MRI access. No pickle, whole-shard loads, or arbitrary GET paths."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from .data_cache import cache_catalog, inspect_cache, within
from .data_manager import settings
from .store import TARGETS, dumps, now

MAX_PIXELS = 4_194_304
MAX_DICOM = 64 * 1024 * 1024
TAGS = ['StudyInstanceUID', 'SeriesInstanceUID', 'SeriesDescription', 'InstanceNumber',
        'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing', 'SliceThickness',
        'Rows', 'Columns', 'NumberOfFrames', 'SamplesPerPixel', 'PhotometricInterpretation',
        'BitsAllocated', 'RescaleSlope', 'RescaleIntercept', 'Modality', 'SOPInstanceUID',
        'SpacingBetweenSlices', 'RepetitionTime', 'EchoTime', 'FlipAngle', 'MagneticFieldStrength',
        'ScanningSequence', 'MRAcquisitionType']


def signature(path):
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def rows(path, limit=50_000):
    if not path.is_file():
        return []
    if path.stat().st_size > 32_000_000:
        raise ValueError('CSV 메타데이터가 32 MB 한도를 넘습니다.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        result = []
        for row in csv.DictReader(f):
            result.append(row)
            if len(result) > limit:
                raise ValueError('CSV 메타데이터 행 수가 한도를 넘습니다.')
        return result


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def stats(array):
    finite = array[np.isfinite(array)].astype(np.float64)
    result = {'pixels': int(array.size), 'finite': int(finite.size), 'nonfinite': int(array.size - finite.size)}
    if not finite.size:
        return {**result, 'min': None, 'max': None, 'mean': None, 'std': None, 'p01': None, 'p99': None,
                'zero_percent': None, 'histogram': [], 'bin_edges': []}
    counts, edges = np.histogram(finite, bins=32)
    return {**result, 'min': float(finite.min()), 'max': float(finite.max()),
            'mean': float(finite.mean()), 'std': float(finite.std()),
            'p01': float(np.percentile(finite, 1)), 'p99': float(np.percentile(finite, 99)),
            'zero_percent': float(np.count_nonzero(finite == 0) * 100 / finite.size),
            'histogram': counts.tolist(), 'bin_edges': edges.tolist()}


def region(array, box, spacing):
    if box is None:
        return None
    if len(box) != 4 or any(not math.isfinite(x) or x < 0 or x > 1 for x in box):
        raise ValueError('ROI 좌표는 0~1 범위여야 합니다.')
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError('ROI의 너비와 높이는 0보다 커야 합니다.')
    h, w = array.shape
    left, top = int(x0 * w), int(y0 * h)
    right, bottom = min(w, math.ceil(x1 * w)), min(h, math.ceil(y1 * h))
    if left >= right or top >= bottom:
        raise ValueError('ROI 안에 픽셀이 없습니다.')
    return {'box': box, 'pixel_box': [left, top, right, bottom],
            'area_mm2': (right - left) * (bottom - top) * spacing[0] * spacing[1] if spacing else None,
            **stats(array[top:bottom, left:right])}


class Viewer:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.sessions = OrderedDict()
        self.indices = OrderedDict()
        self.metadata_key = None
        self.metadata = {}
        with store.connect() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS viewer_reviews(
                id TEXT PRIMARY KEY, source_key TEXT NOT NULL, uid TEXT NOT NULL,
                updated_at TEXT NOT NULL, payload TEXT NOT NULL)''')

    def uri(self, path):
        try:
            return self.store.ws.uri(path)
        except ValueError:
            return str(path)

    def local_path(self, value):
        if not value or '\x00' in value or value.startswith(('\\\\', '//')):
            raise ValueError('이 PC에 있는 파일 또는 폴더 경로를 입력하세요.')
        path = self.store.ws.resolve(value)
        if not path.exists():
            raise ValueError('파일 또는 폴더가 없습니다. 다운로드·압축 해제 후 다시 여세요.')
        return path

    def catalog(self):
        caches = []
        for item in cache_catalog(self.store):
            # Flat training corpora (study, slice, H, W) have no slot axis; the viewer and EDA assume slot caches.
            if item['valid'] and len(item['shape']) == 5:
                path = Path(item['manifest']).resolve()
                caches.append({'id': digest(str(path))[:24], 'name': item['name'], 'path': str(path),
                               'studies': item['studies'], 'shape': item['shape'], 'contract': item['contract'],
                               'selected': str(path) == self.store.get_meta('data_cache_manifest')})
        roots = []
        for label, path in [('대회 원본', self.store.ws.resolve('data')),
                            ('다운로드 폴더', Path(settings(self.store)['destination'])),
                            ('캐시 폴더', self.store.ws.resolve('cache'))]:
            if path.is_dir():
                roots.append({'id': digest(str(path.resolve()))[:24], 'name': label, 'path': str(path.resolve())})
        for key, label in [('dicom_cache', 'DICOM 캐시'), ('test', '로컬 테스트')]:
            if key in self.store.ws.config.get('paths', {}):
                path = self.store.ws.resolve(key)
                if path.is_dir():
                    roots.append({'id': digest(str(path.resolve()))[:24], 'name': label, 'path': str(path.resolve())})
        from .data_library import library
        raw = library(self.store)
        return {'caches': caches, 'roots': roots, 'raw': raw['studies'], 'raw_indexed_at': raw['indexed_at']}

    def browse(self, root_id, relative=''):
        root = next((Path(r['path']) for r in self.catalog()['roots'] if r['id'] == root_id), None)
        if root is None:
            raise ValueError('등록된 데이터 폴더를 선택하세요.')
        path = root if not relative else within(root, relative)
        if not path.is_dir():
            raise ValueError('폴더가 아닙니다.')
        items, scanned = [], 0
        with os.scandir(path) as entries:
            for entry in entries:
                scanned += 1
                if scanned > 5000:
                    break
                p = Path(entry.path)
                if not p.resolve().is_relative_to(root.resolve()) or entry.is_symlink():
                    continue
                is_dir = entry.is_dir()
                supported = p.suffix.lower() in ('.dcm', '.dicom', '.npy') or p.name.endswith('manifest.json')
                if is_dir or supported:
                    items.append({'name': p.name, 'relative': p.relative_to(root).as_posix(),
                                  'path': str(p), 'directory': is_dir})
        return {'path': str(path), 'relative': path.relative_to(root).as_posix(),
                'items': sorted(items, key=lambda x: (not x['directory'], x['name']))[:500],
                'truncated': len(items) > 500 or scanned > 5000}

    def labels(self):
        paths = []
        for split in ('train', 'test'):
            candidates = [self.store.ws.resolve('data') / (split + '.csv'),
                          Path(settings(self.store)['destination']) / (split + '.csv')]
            paths.append(next((p for p in candidates if p.is_file()), None))
        key = [signature(p) if p else None for p in paths]
        with self.lock:
            if key != self.metadata_key:
                result = {}
                for split, path in zip(('train', 'test'), paths):
                    for row in rows(path) if path else []:
                        values = {target: number(row.get(target)) for target in TARGETS}
                        values = {k: v if v in (0, 1) else None for k, v in values.items()}
                        result[row['StudyInstanceUID']] = {'split': split, 'labels': values,
                            'report': row.get('Report', '') if split == 'train' else '',
                            'source': self.uri(path), 'source_signature': digest(signature(path)),
                            'labeled': any(v is not None for v in values.values())}
                self.metadata_key, self.metadata = key, result
            return self.metadata

    def cache_path(self, source):
        found = next((c for c in self.catalog()['caches'] if c['id'] == source), None)
        if found is None:
            raise ValueError('캐시를 다시 선택하세요.')
        return Path(found['path'])

    def cache_index(self, path):
        check = inspect_cache(path)
        if not check['valid']:
            raise ValueError('캐시 검사 실패: ' + '; '.join(check['errors']))
        manifest = json.loads(path.read_text(encoding='utf-8'))
        shape = manifest['shape_total']
        if (shape[0] > 50_000 or shape[1] > 32 or shape[2] > 4096 or math.prod(shape[:3]) > 10_000_000):
            raise ValueError('캐시의 study/시리즈/슬라이스 수가 뷰어 메타데이터 한도를 넘습니다.')
        files = [path, within(path.parent, manifest['uid_file']), within(path.parent, manifest['mask_file'])]
        files += [within(path.parent, s['file']) for s in manifest['shards']]
        sig = [signature(f) for f in files]
        key = digest(sig)
        with self.lock:
            if key in self.indices:
                self.indices.move_to_end(key)
                return self.indices[key]
            uids = [r['StudyInstanceUID'] for r in rows(files[1])]
            mask = np.load(files[2], allow_pickle=False)
            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError('마스크는 0 또는 1이어야 합니다.')
            if mask.ndim == 2:
                mask = np.repeat(mask[:, :, None], manifest['shape_total'][2], axis=2)
            if manifest['shape_total'][-1] * manifest['shape_total'][-2] > MAX_PIXELS:
                raise ValueError('영상 한 장이 4M 픽셀 한도를 넘습니다.')
            index = {'manifest': manifest, 'path': path, 'uids': uids, 'mask': mask,
                     'signature': sig, 'revision': key, 'files': files,
                     'manifest_sha256': check['manifest_sha256']}
            self.indices[key] = index
            while len(self.indices) > 8:
                self.indices.popitem(last=False)
            return index

    def studies(self, source, query='', mode='all', offset=0, target='', split='', root=''):
        if source == 'raw':
            from .data_library import library
            metadata = self.labels()
            found = []
            for i, row in enumerate(library(self.store)['studies']):
                if (split and row['split'] != split) or (root and row.get('archive_id', row['root']) != root):
                    continue
                info = metadata.get(row['uid'], {})
                if not row['files'] or query not in row['uid'] or (mode == 'missing' and row['status'] in ('complete', 'archived')) or (mode == 'labeled' and not info.get('labeled')):
                    continue
                if target and info.get('labels', {}).get(target) != 1:
                    continue
                found.append({**row, 'index': i, 'missing': row.get('missing_files'), 'labeled': info.get('labeled', False)})
            return {'items': found[offset:offset + 40], 'total': len(found), 'offset': offset, 'raw': True}
        index = self.cache_index(self.cache_path(source))
        metadata = self.labels()
        items = []
        missing = np.count_nonzero(index['mask'] == 0, axis=(1, 2))
        for i, uid in enumerate(index['uids']):
            info = metadata.get(uid, {})
            if query not in uid or (mode == 'missing' and not missing[i]) or (mode == 'labeled' and not info.get('labeled')):
                continue
            if target and info.get('labels', {}).get(target) != 1:
                continue
            items.append({'uid': uid, 'index': i, 'missing': int(missing[i]),
                          'labeled': info.get('labeled', False), 'split': info.get('split', 'unknown')})
        counts = {t: {'positive': 0, 'negative': 0, 'unknown': 0} for t in TARGETS}
        for uid in index['uids']:
            for target_name in TARGETS:
                value = metadata.get(uid, {}).get('labels', {}).get(target_name)
                counts[target_name]['positive' if value == 1 else 'negative' if value == 0 else 'unknown'] += 1
        return {'items': items[offset:offset + 40], 'total': len(items), 'offset': offset,
                'summary': {'studies': len(index['uids']), 'missing_studies': int(np.count_nonzero(missing)),
                            'missing_slices': int(missing.sum()), 'targets': counts},
                'manifest_sha256': index['manifest_sha256']}

    def open(self, source='', uid='', path=''):
        target = self.cache_path(source) if source else self.local_path(path)
        if target.suffix.lower() == '.json':
            index = self.cache_index(target)
            if uid and uid not in index['uids']:
                raise ValueError('캐시에 해당 study가 없습니다.')
            i = index['uids'].index(uid) if uid else 0
            m = index['manifest']
            slots = m.get('slots', [])
            series = []
            for s in range(m['shape_total'][1]):
                label = slots[s] if s < len(slots) else f'Slot {s + 1}'
                label = label.get('name', str(s)) if isinstance(label, dict) else label
                valid = index['mask'][i, s].astype(bool).tolist()
                series.append({'name': label, 'slices': len(valid), 'valid': valid,
                               'rows': m['shape_total'][-2], 'columns': m['shape_total'][-1],
                               'ordering': m.get('ordering', ''), 'spacing': None})
            item = {'kind': 'cache', 'index': index, 'study_index': i, 'uid': index['uids'][i],
                    'series': series, 'revision': index['revision'], 'files': index['files'],
                    'signature': index['signature'], 'path': target, 'name': target.parent.name,
                    'warnings': ['전처리된 uint8 캐시입니다. 원본 강도·물리 좌표를 복원하지 않으며 시리즈 간 슬라이스 위치가 동기화되지 않습니다.'],
                    'contract': m['contract'], 'sampling': m.get('sampling', m.get('depth', '')),
                    'normalization': m.get('normalization', ''), 'manifest_sha256': index['manifest_sha256']}
        elif target.suffix.lower() == '.npy':
            array = np.load(target, mmap_mode='r', allow_pickle=False)
            try:
                if array.dtype.kind not in 'uif' or array.ndim not in (2, 3):
                    raise ValueError('단독 NPY는 숫자형 [H,W] 또는 [Z,H,W]만 지원합니다. 5차원 shard는 manifest를 여세요.')
                shape = list(array.shape)
                if any(n < 1 for n in shape) or math.prod(shape[-2:]) > MAX_PIXELS or (len(shape) == 3 and shape[0] > 4096):
                    raise ValueError('NPY 영상 크기가 뷰어 한도를 넘습니다.')
            finally:
                array._mmap.close()
            sig = [signature(target)]
            item = {'kind': 'npy', 'uid': '', 'path': target, 'files': [target], 'signature': sig, 'revision': digest(sig),
                    'name': target.name, 'series': [{'name': 'NPY volume', 'slices': shape[0] if len(shape) == 3 else 1,
                    'rows': shape[-2], 'columns': shape[-1], 'spacing': None, 'ordering': 'array index'}],
                    'warnings': ['축 순서는 [Z,H,W]로 해석합니다. 해부학적 방향·간격·study 연결 정보가 없습니다.']}
        else:
            item = self.open_dicom(target)
        item['id'] = uuid.uuid4().hex
        item['source_key'] = digest([self.uri(item['path']), item['uid']])
        with self.lock:
            self.sessions[item['id']] = item
            while len(self.sessions) > 24:
                self.sessions.popitem(last=False)
        return self.describe(item)

    def open_dicom(self, target):
        files = []
        if target.is_file():
            if target.suffix.lower() not in ('.dcm', '.dicom', ''):
                raise ValueError('DICOM 파일/폴더, NPY 또는 캐시 manifest JSON을 여세요. ZIP은 먼저 풀어야 합니다.')
            files = [target]
        else:
            visited = 0
            for directory, dirs, names in os.walk(target, followlinks=False):
                visited += 1
                if visited > 300 or len(files) > 2500:
                    raise ValueError('한 번에 한 study 폴더를 여세요. 폴더 300개·DICOM 2,500개까지 지원합니다.')
                dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
                for name in names:
                    file = Path(directory) / name
                    if file.suffix.lower() in ('.dcm', '.dicom') and file.resolve().is_relative_to(target.resolve()):
                        files.append(file)
                        if len(files) > 2500:
                            raise ValueError('DICOM 2,500개 이하의 study 폴더를 선택하세요.')
        if not files:
            raise ValueError('DICOM 영상이 없습니다. study 폴더를 선택하거나 기존 캐시를 여세요.')
        groups, studies = {}, set()
        for file in files:
            if file.stat().st_size > MAX_DICOM:
                raise ValueError('DICOM 한 파일은 64 MiB 이하만 지원합니다.')
            try:
                d = pydicom.dcmread(file, stop_before_pixels=True, specific_tags=TAGS)
            except Exception as exc:
                raise ValueError('DICOM 헤더를 읽지 못했습니다: ' + file.name) from exc
            if int(d.get('NumberOfFrames', 1)) != 1 or int(d.get('SamplesPerPixel', 1)) != 1:
                raise ValueError('현재는 단일 프레임 흑백 DICOM을 지원합니다.')
            h, w = int(d.get('Rows', 0)), int(d.get('Columns', 0))
            if min(h, w) < 1 or h * w > MAX_PIXELS or int(d.get('BitsAllocated', 0)) not in (8, 16, 32):
                raise ValueError('DICOM 영상 크기/픽셀 형식을 지원하지 않습니다.')
            if str(d.get('PhotometricInterpretation', '')) not in ('MONOCHROME1', 'MONOCHROME2'):
                raise ValueError('흑백 MRI DICOM만 지원합니다.')
            study, series_id = str(d.get('StudyInstanceUID', '')), str(d.get('SeriesInstanceUID', ''))
            if not study or not series_id:
                raise ValueError('Study/Series UID가 없습니다.')
            studies.add(study)
            groups.setdefault(series_id, []).append((file, d))
        if len(studies) != 1:
            raise ValueError('여러 study가 포함되어 있습니다. 한 study의 하위 폴더를 여세요.')
        result, warnings = [], []
        for series_id, entries in sorted(groups.items()):
            first = entries[0][1]
            positions = []
            try:
                orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
                normal = np.cross(orientation[:3], orientation[3:])
                if len(orientation) != 6 or not np.isfinite(normal).all() or np.linalg.norm(normal) < .99:
                    raise ValueError()
                for file, d in entries:
                    if not np.allclose(np.asarray(d.ImageOrientationPatient, dtype=float), orientation, atol=.001):
                        raise ValueError()
                    position = float(np.dot(np.asarray(d.ImagePositionPatient, dtype=float), normal))
                    if not math.isfinite(position):
                        raise ValueError()
                    positions.append(position)
                order = sorted(range(len(entries)), key=lambda i: (positions[i], entries[i][0].name))
                ordered_positions = [positions[i] for i in order]
                delta = np.diff(ordered_positions)
                if len(delta) and (np.any(delta < .001) or (len(delta) > 1 and not np.allclose(delta, np.median(delta), rtol=.1, atol=.1))):
                    warnings.append('슬라이스 위치 중복 또는 간격 불균일: ' + series_id)
                method = 'IOP/IPP 물리 위치'
            except (AttributeError, ValueError, TypeError):
                order = sorted(range(len(entries)), key=lambda i: (number(entries[i][1].get('InstanceNumber')) or 0, entries[i][0].name))
                ordered_positions = []
                method = 'InstanceNumber → 파일명 (위치 정보 불완전)'
                warnings.append('물리 위치 정렬을 확인할 수 없는 시리즈입니다: ' + series_id)
            ordered = [entries[i] for i in order]
            shapes = {(int(d.Rows), int(d.Columns)) for _, d in entries}
            if len(shapes) != 1:
                raise ValueError('한 시리즈의 영상 크기가 서로 다릅니다.')
            spacing = [number(x) for x in first.get('PixelSpacing', [])]
            if len(spacing) != 2 or any(x is None or x <= 0 for x in spacing):
                spacing = None
            elif not all(len(d.get('PixelSpacing', [])) == 2 and np.allclose([number(x) or 0 for x in d.PixelSpacing], spacing) for _, d in entries):
                spacing = None
                warnings.append('시리즈의 픽셀 간격이 달라 mm 측정을 비활성화했습니다.')
            result.append({'name': str(first.get('SeriesDescription', 'MRI series'))[:120], 'series_uid': series_id,
                           'slices': len(entries), 'rows': int(first.Rows), 'columns': int(first.Columns),
                           'spacing': spacing, 'ordering': method, 'positions_mm': ordered_positions,
                           'files': [file for file, _ in ordered],
                           'header': {key: number(first.get(key)) for key in ('SliceThickness', 'SpacingBetweenSlices',
                               'RepetitionTime', 'EchoTime', 'FlipAngle', 'MagneticFieldStrength')},
                           'acquisition_type': str(first.get('MRAcquisitionType', ''))[:30],
                           'orientation': [number(x) for x in first.get('ImageOrientationPatient', [])],
                           'header_scope': 'first DICOM in series; acquisition parameters may vary'})
        sig = [signature(file) for file in files]
        return {'kind': 'dicom', 'path': target, 'uid': next(iter(studies)), 'series': result, 'files': files,
                'signature': sig, 'revision': digest(sig), 'name': target.name, 'warnings': warnings +
                ['저장 방향 그대로 표시합니다. 자동 재방향·3D 재구성·시리즈 간 위치 동기화는 수행하지 않습니다.']}

    def session(self, id):
        with self.lock:
            item = self.sessions.get(id)
            if item is None:
                raise ValueError('영상 세션이 만료되었습니다. 파일을 다시 여세요.')
            self.sessions.move_to_end(id)
            return item

    def describe(self, item):
        metadata = self.labels().get(item['uid'])
        public = {k: v for k, v in item.items() if k not in ('index', 'signature', 'files', 'path', 'study_index', 'series')}
        public['series'] = [{k: v for k, v in s.items() if k != 'files'} for s in item['series']]
        if item['kind'] == 'dicom':
            for source, target in zip(item['series'], public['series']):
                target['filenames'] = [p.name for p in source['files']]
        public['source'] = self.uri(item['path'])
        public['metadata'] = metadata
        public['reviews'] = self.reviews(item)
        return public

    def read(self, id, slot, z):
        item = self.session(id)
        if slot < 0 or slot >= len(item['series']):
            raise ValueError('시리즈 범위를 벗어났습니다.')
        series = item['series'][slot]
        if z < 0 or z >= series['slices']:
            raise ValueError('슬라이스 범위를 벗어났습니다.')
        try:
            if [signature(f) for f in item['files']] != item['signature']:
                raise ValueError('열어 둔 데이터 파일이 변경되었습니다. 다시 여세요.')
            valid = series.get('valid', [True] * series['slices'])[z]
            inverted = False
            if item['kind'] == 'cache':
                index, i = item['index'], item['study_index']
                shard = next(s for s in index['manifest']['shards'] if s['start'] <= i < s['stop'])
                file = within(index['path'].parent, shard['file'])
                array = np.load(file, mmap_mode='r', allow_pickle=False)
                try:
                    plane = np.array(array[i - shard['start'], slot, z], dtype=np.float32)
                finally:
                    array._mmap.close()
            elif item['kind'] == 'npy':
                array = np.load(item['path'], mmap_mode='r', allow_pickle=False)
                try:
                    plane = np.array(array[z] if array.ndim == 3 else array, dtype=np.float32)
                finally:
                    array._mmap.close()
            else:
                d = pydicom.dcmread(series['files'][z])
                if int(d.get('NumberOfFrames', 1)) != 1 or int(d.Rows) * int(d.Columns) > MAX_PIXELS:
                    raise ValueError('DICOM 파일이 변경되었거나 크기 제한을 넘습니다.')
                plane = d.pixel_array.astype(np.float32)
                plane = plane * float(d.get('RescaleSlope', 1)) + float(d.get('RescaleIntercept', 0))
                inverted = str(d.PhotometricInterpretation) == 'MONOCHROME1'
        except (OSError, RuntimeError, NotImplementedError) as exc:
            raise ValueError('영상을 읽지 못했습니다. 파일 또는 DICOM 디코더를 확인하세요.') from exc
        return item, series, plane, valid, inverted

    def frame(self, id, slot=0, z=0, low=None, high=None, invert=False, roi=None):
        item, series, plane, valid, mono1 = self.read(id, slot, z)
        info = stats(plane)
        if (low is None) != (high is None):
            raise ValueError('밝기 최소·최대값을 함께 입력하세요.')
        if low is None:
            low, high = (0., 255.) if item['kind'] == 'cache' else (
                info['p01'] if info['p01'] is not None else 0.,
                info['p99'] if info['p99'] is not None else 1.)
            if high <= low:
                high = low + 1
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError('밝기 최대값은 최소값보다 커야 합니다.')
        display = np.clip((np.nan_to_num(plane, nan=low, posinf=high, neginf=low) - low) / (high - low), 0, 1)
        if bool(invert) != mono1:
            display = 1 - display
        output = io.BytesIO()
        Image.fromarray((display * 255).astype(np.uint8)).save(output, format='PNG')
        return {'image': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode(),
                'slot': slot, 'slice': z, 'shape': list(plane.shape), 'valid': bool(valid),
                'window': [low, high], 'invert': invert, 'monochrome1': mono1, 'stats': info,
                'spacing': series.get('spacing'), 'roi': region(plane, roi, series.get('spacing')) if valid else None,
                'pixel_sha256': hashlib.sha256(plane.tobytes()).hexdigest(),
                'intensity_unit': 'uint8 캐시 강도' if item['kind'] == 'cache' else '픽셀 강도 (MRI, HU 아님)'}

    def contact(self, id, slot):
        item = self.session(id)
        if slot < 0 or slot >= len(item['series']):
            raise ValueError('시리즈 범위를 벗어났습니다.')
        n = item['series'][slot]['slices']
        slices = sorted(set(np.linspace(0, n - 1, min(n, 12)).round().astype(int).tolist()))
        frames = []
        for z in slices:
            f = self.frame(id, slot, z)
            image = Image.open(io.BytesIO(base64.b64decode(f['image'].split(',')[1])))
            image.thumbnail((160, 160))
            output = io.BytesIO()
            image.save(output, format='PNG')
            frames.append({'slice': z, 'valid': f['valid'], 'image': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()})
        return {'items': frames, 'total': n, 'note': '최대 12장의 균등 간격 미리보기. 자동 대비를 적용하며 모든 슬라이스 검사 결과가 아닙니다.'}

    def reviews(self, item):
        return [json.loads(r['payload']) for r in self.store.rows(
            'SELECT payload FROM viewer_reviews WHERE source_key=? ORDER BY updated_at DESC LIMIT 100', (item['source_key'],))]

    def review(self, id, slot, z, note, flags, roi=None, low=None, high=None, invert=False):
        item = self.session(id)
        frame = self.frame(id, slot, z, low, high, invert, roi)
        frame.pop('image')
        review_id = digest([item['source_key'], item['revision'], slot, z])
        payload = {'id': review_id, 'uid': item['uid'], 'source': self.uri(item['path']),
                   'revision': item['revision'], 'revision_type': 'file path/size/mtime signature',
                   'manifest_sha256': item.get('manifest_sha256'), 'slot': slot, 'series': item['series'][slot]['name'],
                   'slice': z, 'note': note, 'flags': flags, 'frame': frame, 'updated_at': now()}
        with self.store.connect() as con:
            con.execute('INSERT OR REPLACE INTO viewer_reviews VALUES(?,?,?,?,?)',
                        (review_id, item['source_key'], item['uid'], payload['updated_at'], dumps(payload)))
        return payload
