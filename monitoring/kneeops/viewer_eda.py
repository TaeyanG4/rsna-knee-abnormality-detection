"""Scope-aware MRI EDA: metadata for the cohort, bounded pixels for one series."""
from __future__ import annotations

import base64
import hashlib
import io
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .data_library import library
from .store import TARGETS, now
from .viewer import number, rows, stats


def frequencies(values):
    return [{'name': str(k), 'count': v} for k, v in sorted(Counter(values).items(), key=lambda x: str(x[0]))]


def distribution(values):
    if not values:
        return {'n': 0, 'min': None, 'median': None, 'p95': None, 'max': None, 'bins': []}
    a = np.asarray(values, dtype=float)
    hist, edges = np.histogram(a, bins=min(12, len(set(values))))
    return {'n': len(a), 'min': float(a.min()), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max()),
            'bins': [{'name': f'{edges[i]:.0f}–{edges[i+1]:.0f}', 'count': int(v)} for i, v in enumerate(hist)]}


def dataset_summary(viewer):
    store = viewer.store
    inventory = library(store)
    provenance, series, train_rows = [], [], []
    split_uids = {}
    for split in ('train', 'test'):
        for suffix in ('', '_series'):
            path = store.ws.resolve('data') / (split + suffix + '.csv')
            # Follow the same installed metadata fallback as the viewer.
            if not path.is_file():
                from .data_manager import settings
                path = Path(settings(store)['destination']) / path.name
            content = rows(path)
            if path.is_file():
                with path.open('rb') as handle:
                    sha = hashlib.file_digest(handle, 'sha256').hexdigest()
                provenance.append({'path': viewer.uri(path), 'sha256': sha, 'rows': len(content)})
            if suffix:
                series.extend({**r, 'split': split} for r in content)
            else:
                split_uids[split] = [r['StudyInstanceUID'] for r in content]
                if split == 'train':
                    train_rows = content
    train_ids = set(split_uids.get('train', []))
    counts = {t: {'positive': 0, 'negative': 0, 'unknown': 0} for t in TARGETS}
    # Keep train labels tied to train.csv even if a corrupt test CSV repeats a UID.
    unique_train = {r['StudyInstanceUID']: r for r in train_rows}
    train_info = []
    for row in unique_train.values():
        values = {t: number(row.get(t)) for t in TARGETS}
        values = {t: v if v in (0, 1) else None for t, v in values.items()}
        train_info.append({'labels': values, 'labeled': any(v is not None for v in values.values())})
    for info in train_info:
        for target in TARGETS:
            v = info.get('labels', {}).get(target)
            counts[target]['positive' if v == 1 else 'negative' if v == 0 else 'unknown'] += 1
    copositive = []
    for a in TARGETS:
        copositive.append([{
            'known': sum(i.get('labels', {}).get(a) is not None and i.get('labels', {}).get(b) is not None for i in train_info),
            'both_positive': sum(i.get('labels', {}).get(a) == 1 and i.get('labels', {}).get(b) == 1 for i in train_info)
        } for b in TARGETS])
    by_study = {}
    for row in series:
        by_study.setdefault((row['split'], row['StudyInstanceUID']), []).append(row)
    split_summary = {}
    for split in ('train', 'test'):
        ids = set(split_uids.get(split, []))
        scans = [r for r in series if r['split'] == split]
        raw = {r['uid']: r for r in inventory['studies'] if r['split'] == split}
        missing_planes = {plane: sorted(uid for uid in ids if plane not in {
            r.get('Anatomical_Plane') for r in by_study.get((split, uid), [])})
            for plane in ('Axial', 'Coronal', 'Sagittal')}
        split_summary[split] = {'studies': len(ids), 'series': len(scans), 'raw_studies': len(raw),
            'prepared_studies': sum(r.get('storage') == 'extracted' or (not r.get('archive_id') and r['files'] > 0) for r in raw.values()),
            'dicom_files': sum(r['files'] for r in raw.values()),
            'planes': frequencies(r.get('Anatomical_Plane') or '미표기' for r in scans),
            'protocols': frequencies(f"{r.get('Anatomical_Plane','?')} · Fluid={r.get('Fluid_Sensitive','?')} · FS={r.get('Fat_Suppression','?')}" for r in scans),
            'series_per_study': distribution([len(by_study.get((split, uid), [])) for uid in ids]),
            'files_per_study': distribution([r['files'] for r in raw.values()]),
            'missing_planes': {plane: {'count': len(uids), 'examples': [{'uid': uid, 'raw': raw.get(uid, {}).get('id')} for uid in uids[:30]]}
                               for plane, uids in missing_planes.items()},
            'missing_raw': len(ids - raw.keys()), 'duplicate_study_rows': len(split_uids.get(split, [])) - len(ids),
            'orphan_series_studies': len({r['StudyInstanceUID'] for r in scans} - ids)}
    caches = []
    for entry in viewer.catalog()['caches']:
        index = viewer.cache_index(Path(entry['path']))
        mask = index['mask'].astype(bool)
        uid_set = set(index['uids'])
        names = index['manifest'].get('slots', [])
        caches.append({**entry, 'path': viewer.uri(index['path']),
            'missing_train': len(train_ids - uid_set), 'extra_studies': len(uid_set - train_ids),
            'missing_studies': int(np.count_nonzero(~mask.all(axis=(1, 2)))),
            'slots': [{'name': (names[i].get('name', str(i)) if isinstance(names[i], dict) else names[i]) if i < len(names) else f'Slot {i+1}',
                       'empty_studies': int(np.count_nonzero(~mask[:, i].any(axis=1))),
                       'missing_slices': int(np.count_nonzero(~mask[:, i])),
                       'total_slices': int(mask[:, i].size)} for i in range(mask.shape[1])],
            'manifest_sha256': index['manifest_sha256']})
    return {'schema': 1, 'observed_at': now(), 'scope': 'official CSV and indexed raw listing; cache masks only; no cohort pixel scan',
        'splits': split_summary, 'train_test_uid_overlap': len(train_ids & set(split_uids.get('test', []))),
        'labels': counts, 'targets': TARGETS, 'copositive': copositive,
        'labeled_studies': sum(i.get('labeled', False) for i in train_info),
        'report_present': sum(bool((r.get('Report') or '').strip()) for r in unique_train.values()),
        'report_characters': distribution([len(r.get('Report') or '') for r in unique_train.values()]),
        'caches': caches, 'sources': provenance,
        'archive_revisions': [{'id': a['id'], 'directory_sha256': a['directory_sha256']} for a in inventory.get('archives', [])],
        'warnings': ['미표기는 음성이 아닙니다. 양성률의 분모는 양성+음성(라벨이 있는 study)입니다.',
                     'Gold58은 기존 실험에 노출된 진단셋이며 독립 holdout이 아닙니다.',
                     'Study UID는 환자 ID가 아닙니다. UID 교집합 0만으로 환자 단위 누출 부재를 증명할 수 없습니다.',
                     '테스트에는 추론용 Report가 없습니다. 결측 촬영 방향은 CSV 기준이며 곧바로 파일 손상을 뜻하지 않습니다.']}


def series_audit(viewer, session, slot, samples=16):
    item = viewer.session(session)
    if not 0 <= slot < len(item['series']) or not 3 <= samples <= 64:
        raise ValueError('시리즈와 표본 수(3~64)를 확인하세요.')
    series = item['series'][slot]
    indices = sorted(set(np.linspace(0, series['slices'] - 1, min(samples, series['slices'])).round().astype(int).tolist()))
    frames, hashes, pooled = [], {}, []
    previous = None
    for z in indices:
        _, _, plane, valid, _ = viewer.read(session, slot, z)
        if not valid:
            frames.append({'slice': z, 'valid': False, 'flags': ['결측 마스크']})
            previous = None
            continue
        info = stats(plane)
        checksum = hashlib.sha256(plane.tobytes()).hexdigest()
        flags = []
        if info['finite'] == 0: flags.append('유한 픽셀 없음')
        if info['std'] == 0: flags.append('균일한 영상')
        if (info['zero_percent'] or 0) >= 99: flags.append('0 값 99% 이상')
        if info['nonfinite']: flags.append('NaN/Inf 포함')
        if checksum in hashes: flags.append(f"표본 #{hashes[checksum] + 1}과 픽셀 동일")
        hashes[checksum] = z
        finite = plane[np.isfinite(plane)]
        if finite.size:
            pooled.append(finite[::max(1, finite.size // 4096)][:4096])
        low, high = info['p01'], info['p99']
        normalized = np.clip((np.nan_to_num(plane.astype(np.float64), nan=low or 0,
            posinf=high or 0, neginf=low or 0) - (low or 0)) / max((high or 0) - (low or 0), 1e-6), 0, 1)
        # Fixed 128px thumbnail makes edge energy less resolution-dependent; not a motion classifier.
        small = np.asarray(Image.fromarray(normalized.astype(np.float32)).resize((128, 128)), dtype=float)
        edge = float((np.abs(np.diff(small, axis=0)).mean() + np.abs(np.diff(small, axis=1)).mean()) / 2)
        change = float(np.abs(small - previous[1]).mean()) if previous and z == previous[0] + 1 else None
        previous = (z, small)
        frames.append({'slice': z, 'valid': True, 'min': info['min'], 'max': info['max'], 'mean': info['mean'],
                       'std': info['std'], 'zero_percent': info['zero_percent'], 'nonfinite': info['nonfinite'],
                       'edge_energy': edge, 'adjacent_change': change, 'pixel_sha256': checksum, 'flags': flags})
    positions = series.get('positions_mm', [])
    delta = np.diff(positions)
    geometry = {'pixel_spacing_mm': series.get('spacing'), 'positions_known': len(positions),
        'coverage_mm': float(max(positions) - min(positions)) if len(positions) > 1 else None,
        'step_mm': distribution(delta.tolist()), 'duplicate_positions': int(np.count_nonzero(np.abs(delta) < .001)),
        'irregular_steps': int(np.count_nonzero(~np.isclose(delta, np.median(delta), rtol=.1, atol=.1))) if len(delta) else 0}
    spacing = series.get('spacing')
    geometry['fov_mm'] = [series['rows'] * spacing[0], series['columns'] * spacing[1]] if spacing else None
    window = np.percentile(np.concatenate(pooled), [1, 99]).tolist() if pooled else None
    if window and window[0] == window[1]: window[1] += 1
    return {'schema': 1, 'observed_at': now(), 'source': viewer.uri(item['path']), 'uid': item['uid'],
        'revision': item['revision'], 'slot': slot, 'series': {k: v for k, v in series.items() if k != 'files'},
        'total_slices': series['slices'], 'sampled_slices': len(indices), 'all_slices': len(indices) == series['slices'],
        'geometry': geometry, 'frames': frames, 'recommended_window': window,
        'flagged_samples': sum(bool(f['flags']) for f in frames),
        'notes': ['픽셀 검사는 표시된 표본에 한합니다. 신호 없는 경계 슬라이스도 점검 후보가 될 수 있습니다.',
                  '경계 에너지는 해부 구조·대비에 영향받는 참고값이며 움직임이나 질병 판정이 아닙니다.',
                  '인접 변화량은 연속된 슬라이스 쌍만 계산합니다. 동일 픽셀은 표본 안에서만 비교합니다.',
                  '권장 대비는 표본별 최대4096픽셀의1–99백분위입니다. MRI 강도는 HU가 아닙니다.']}


def preprocessing_preview(viewer, session, slot, z, resolution, crop_percent, normalization):
    if resolution not in (128, 224, 256, 288, 336, 384, 512) or not 25 <= crop_percent <= 100 or normalization not in ('minmax', 'p1', 'p2'):
        raise ValueError('전처리 설정을 확인하세요.')
    item, series, plane, valid, mono1 = viewer.read(session, slot, z)
    if not valid or not np.isfinite(plane).any():
        raise ValueError('결측 또는 유한 픽셀이 없는 슬라이스는 전처리 비교에서 제외합니다.')
    h, w = plane.shape
    ch, cw = max(1, round(h * crop_percent / 100)), max(1, round(w * crop_percent / 100))
    top, left = (h - ch) // 2, (w - cw) // 2
    crop = plane[top:top+ch, left:left+cw]
    finite = crop[np.isfinite(crop)]
    if not finite.size:
        raise ValueError('선택한 crop에 유한 픽셀이 없습니다.')
    q = {'minmax': (0, 100), 'p1': (1, 99), 'p2': (2, 98)}[normalization]
    low, high = np.percentile(finite, q)
    if high <= low: high = low + 1
    normalized = np.clip((np.nan_to_num(crop.astype(np.float64), nan=low, posinf=high, neginf=low) - low) / (high - low), 0, 1)
    if mono1: normalized = 1 - normalized
    image = Image.fromarray((normalized * 255).astype('uint8'))
    # Preserve aspect ratio; pad instead of stretching anatomy to a square.
    scale = resolution / max(ch, cw)
    size = (max(1, round(cw * scale)), max(1, round(ch * scale)))
    image = image.resize(size, Image.Resampling.BILINEAR)
    square = Image.new('L', (resolution, resolution))
    square.paste(image, ((resolution-size[0])//2, (resolution-size[1])//2))
    buffer = io.BytesIO(); square.save(buffer, format='PNG')
    spacing = series.get('spacing')
    return {'image': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
            'schema': 1, 'observed_at': now(), 'source': viewer.uri(item['path']),
            'uid': item['uid'], 'revision': item['revision'], 'slice': z, 'slot': slot,
            'crop_percent': crop_percent, 'resolution': resolution,
            'pixel_sha256': hashlib.sha256(plane.tobytes()).hexdigest(),
            'original_shape': [h,w], 'crop_box': [left,top,left+cw,top+ch], 'output_shape': [resolution,resolution],
            'retained_area_percent': 100 * ch * cw / (h*w), 'upsampled': scale > 1,
            'crop_fov_mm': [ch*spacing[0],cw*spacing[1]] if spacing else None,
            'normalization': normalization, 'window': [float(low),float(high)], 'uint8_bytes': resolution**2,
            'float32_bytes': resolution**2 * 4, 'nonfinite_pixels_replaced': int(np.count_nonzero(~np.isfinite(crop))),
            'note': '현재 한 장의 중앙 crop·정규화·bilinear resize+padding 미리보기입니다. 학습 성능이나 병변 보존을 보장하지 않습니다. 캐시 확대는 원본 정보를 복구하지 않습니다.'}
