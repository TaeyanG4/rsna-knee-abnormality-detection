"""Local prediction files: discovery and diagnostic comparison.

Read-only. Only files found under the workspace roots below can be opened, by their
relative id. Scores here are diagnostics on whatever label file is chosen; they are not
competition scores, and the blend uses fixed equal weights so it cannot be tuned to the
label set it is evaluated on.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .store import TARGETS, now

ROOTS = ['artifacts', 'kaggle/submissions', 'kaggle/packages', 'storage/labels', 'storage/gold', 'storage/competition']
LABEL_ROOTS = ('storage/labels', 'storage/gold', 'storage/competition')
MAX_BYTES = 200_000_000
MAX_DEPTH = 5
UID = 'StudyInstanceUID'
_cache: dict = {}


def _header(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return next(csv.reader(f), [])


def discover(store, refresh=False):
    root = store.ws.root
    key = str(root)
    if not refresh and key in _cache and time.monotonic() - _cache[key][0] < 60:
        return _cache[key][1]
    items = []
    for rel in ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in base.rglob('*.csv'):
            parts = path.relative_to(base).parts
            if len(parts) > MAX_DEPTH or any(p.startswith('.') for p in parts):
                continue
            try:
                size = path.stat().st_size
                if size == 0 or size > MAX_BYTES:
                    continue
                header = _header(path)
            except (OSError, UnicodeDecodeError, csv.Error):
                continue
            targets = [t for t in TARGETS if t in header]
            if UID not in header or not targets or path.name.startswith('sample_submission'):
                continue
            rid = path.relative_to(root).as_posix()
            is_label = rid.startswith(LABEL_ROOTS) or path.name == 'train.csv'
            items.append({'id': rid, 'name': path.name, 'folder': path.parent.relative_to(root).as_posix(),
                          'bytes': size, 'modified': path.stat().st_mtime, 'targets': targets,
                          'role': 'label' if is_label else 'prediction', 'has_report': 'Report' in header})
    items.sort(key=lambda x: (x['role'], x['folder'], x['name']))
    result = {'predictions': [i for i in items if i['role'] == 'prediction'],
              'labels': [i for i in items if i['role'] == 'label'], 'observed_at': now(),
              'roots': ROOTS}
    _cache[key] = (time.monotonic(), result)
    return result


def _resolve(store, rid):
    known = {i['id']: i for group in ('predictions', 'labels') for i in discover(store)[group]}
    if rid not in known:
        known = {i['id']: i for group in ('predictions', 'labels') for i in discover(store, refresh=True)[group]}
    if rid not in known:
        raise ValueError(f'목록에 없는 파일입니다: {rid}')
    return store.ws.root / rid, known[rid]


def load(path):
    """uid -> float vector over TARGETS (NaN when the column or value is missing)."""
    rows = {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = (row.get(UID) or '').strip()
            if not uid:
                continue
            values = []
            for t in TARGETS:
                raw = (row.get(t) or '').strip()
                try:
                    values.append(float(raw) if raw else np.nan)
                except ValueError:
                    values.append(np.nan)
            rows[uid] = np.array(values, dtype=float)
    return rows


def rankdata(a):
    order = np.argsort(a, kind='mergesort')
    s = a[order]
    starts = np.r_[True, s[1:] != s[:-1]]
    group = np.cumsum(starts) - 1
    bounds = np.r_[np.nonzero(starts)[0], len(a)]
    average = (bounds[:-1] + bounds[1:] - 1) / 2 + 1
    ranks = np.empty(len(a))
    ranks[order] = average[group]
    return ranks


def auc(y, score):
    positive = y == 1
    n1 = int(positive.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def binarize(values):
    """Hard labels stay; soft labels in [0,1] are thresholded at 0.5 and reported as such."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values, 'empty'
    if np.isin(finite, (0, 1)).all():
        return values, 'binary'
    if finite.min() >= 0 and finite.max() <= 1:
        out = values.copy(); mask = np.isfinite(out)
        out[mask] = (out[mask] >= .5).astype(float)
        return out, 'soft>=0.5'
    return np.full_like(values, np.nan), 'unusable'


def _macro(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def compare(store, files, label='', bootstrap=200, seed=20261002):
    if not 1 <= len(files) <= 4 or len(set(files)) != len(files):
        raise ValueError('예측 파일을 1~4개 고르세요.')
    preds = []
    for rid in files:
        path, info = _resolve(store, rid)
        preds.append((rid, info, load(path)))
    common = set.intersection(*(set(p[2]) for p in preds))
    warnings = []
    if any(len(p[2]) != len(common) for p in preds):
        warnings.append('파일마다 study 집합이 달라 공통 study만 비교합니다.')
    label_rows, label_info, label_mode = None, None, {}
    if label:
        path, label_info = _resolve(store, label)
        label_rows = load(path)
        common &= {u for u, v in label_rows.items() if np.isfinite(v).any()}
        if label_info['id'].endswith('competition/train.csv'):
            warnings.append('공식 라벨은 58 study뿐이며 Gold58은 기존 선택에 노출된 진단셋입니다. 새 독립 holdout이나 대회 점수로 해석하지 마세요.')
        else:
            warnings.append('teacher/보고서 기반 라벨은 대리지표입니다. 같은 라벨로 학습한 모델은 누출된 점수가 나옵니다.')
    uids = sorted(common)
    if not uids:
        raise ValueError('선택한 파일 사이에 공통 study가 없습니다.')
    matrix = np.stack([np.stack([p[2][u] for u in uids]) for p in preds])  # files x studies x targets
    summary = []
    for (rid, info, rows), m in zip(preds, matrix):
        finite = m[np.isfinite(m)]
        summary.append({'id': rid, 'name': info['name'], 'folder': info['folder'], 'rows': len(rows), 'targets': info['targets'],
                        'min': float(finite.min()) if finite.size else None, 'max': float(finite.max()) if finite.size else None,
                        'outside_probability': bool(finite.size and (finite.min() < 0 or finite.max() > 1)),
                        'missing': int((~np.isfinite(m)).sum())})
    if any(s['outside_probability'] for s in summary):
        warnings.append('0~1 밖의 값이 있는 파일이 있습니다(logit 가능). AUC와 순위 상관은 영향이 없지만 확률 평균 blend는 쓰지 않고 순위 평균만 사용합니다.')
    # Fixed equal-weight rank average: per target, rank-normalise each file then average.
    blend = None
    if len(preds) > 1:
        blend = np.full(matrix.shape[1:], np.nan)
        for j in range(len(TARGETS)):
            cols = matrix[:, :, j]
            ok = np.isfinite(cols).all(axis=0)
            if ok.sum() >= 2:
                blend[ok, j] = np.mean([rankdata(c[ok]) / ok.sum() for c in cols], axis=0)
    correlations = []
    for a in range(len(preds)):
        for b in range(a + 1, len(preds)):
            per = {}
            for j, t in enumerate(TARGETS):
                x, y = matrix[a, :, j], matrix[b, :, j]
                ok = np.isfinite(x) & np.isfinite(y)
                per[t] = spearman(x[ok], y[ok]) if ok.sum() >= 3 else None
            correlations.append({'a': files[a], 'b': files[b], 'per_target': per, 'macro': _macro(per.values())})
    result = {'observed_at': now(), 'common_studies': len(uids), 'files': summary, 'correlations': correlations,
              'targets': TARGETS, 'warnings': warnings, 'label': None, 'blend': 'equal-weight rank mean' if blend is not None else None}
    if label_rows is not None:
        y = np.stack([label_rows[u] for u in uids])
        for j, t in enumerate(TARGETS):
            y[:, j], label_mode[t] = binarize(y[:, j])
        scored = [(f, matrix[i]) for i, f in enumerate(files)] + ([('blend', blend)] if blend is not None else [])
        per_target = []
        for j, t in enumerate(TARGETS):
            ok_y = np.isfinite(y[:, j])
            row = {'target': t, 'label_mode': label_mode[t], 'n_pos': int((y[ok_y, j] == 1).sum()), 'n_neg': int((y[ok_y, j] == 0).sum()), 'auc': {}}
            for name, m in scored:
                ok = ok_y & np.isfinite(m[:, j])
                row['auc'][name] = auc(y[ok, j], m[ok, j]) if ok.sum() else None
            per_target.append(row)
        computable = [r['target'] for r in per_target if all(r['auc'][n] is not None for n, _ in scored)]
        macro = {n: _macro([r['auc'][n] for r in per_target if r['target'] in computable]) for n, _ in scored}
        boot = _bootstrap(y, scored, computable, bootstrap, seed)
        if len(computable) < len(TARGETS):
            warnings.append(f'양성·음성이 모두 있는 target {len(computable)}개만 평균했습니다. 12-target 대회 점수와 다릅니다.')
        if any(v == 'soft>=0.5' for v in label_mode.values()):
            warnings.append('연속값 라벨은 0.5 기준으로 이진화했습니다.')
        result['label'] = {'id': label_info['id'], 'name': label_info['name'], 'studies': int(np.isfinite(y).any(axis=1).sum()),
                           'per_target': per_target, 'computable_targets': computable, 'macro': macro, 'bootstrap': boot}
    return result


def _bootstrap(y, scored, targets, reps, seed):
    if not targets or reps <= 0:
        return None
    idx = [TARGETS.index(t) for t in targets]
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    samples = {name: [] for name, _ in scored}
    for _ in range(reps):
        pick = rng.integers(0, n, n)
        for name, m in scored:
            vals = []
            for j in idx:
                yy, ss = y[pick, j], m[pick, j]
                ok = np.isfinite(yy) & np.isfinite(ss)
                v = auc(yy[ok], ss[ok]) if ok.any() else None
                if v is not None:
                    vals.append(v)
            samples[name].append(np.mean(vals) if len(vals) == len(idx) else np.nan)
    out = {}
    for name, vals in samples.items():
        arr = np.array(vals, dtype=float); arr = arr[np.isfinite(arr)]
        out[name] = {'low': float(np.percentile(arr, 2.5)), 'high': float(np.percentile(arr, 97.5)), 'valid': int(arr.size)} if arr.size else None
    if 'blend' in samples:
        singles = [name for name, _ in scored if name != 'blend']
        means = {s: np.nanmean(samples[s]) for s in singles}
        best = max(means, key=means.get)
        diff = np.array(samples['blend']) - np.array(samples[best]); diff = diff[np.isfinite(diff)]
        if diff.size:
            out['blend_vs_best'] = {'reference': best, 'low': float(np.percentile(diff, 2.5)), 'high': float(np.percentile(diff, 97.5)),
                                    'fraction_positive': float((diff > 0).mean()), 'unit': 'study', 'reps': reps}
    return out
