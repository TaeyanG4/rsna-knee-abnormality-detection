from __future__ import annotations
import os, json, time, hashlib, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pydicom
import cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut

ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
OUT = Path('/kaggle/working')
CACHE = OUT / 'train_cache_6x9x224.npy'
MASKS = OUT / 'train_cache_masks.npy'
UIDS = OUT / 'train_cache_uids.csv'
MANIFEST = OUT / 'train_cache_manifest.json'
FAILURES = OUT / 'train_cache_failures.csv'

IMG = 224
CROP_MM = 140.0
N_SLICE = 9
SPAN = (0.02, 0.98)
WORKERS = max(2, min(8, (os.cpu_count() or 4) - 1))
SLOTS = [
    ('SAG_FLUID', 'Sagittal', 1),
    ('SAG_STRUCT', 'Sagittal', 0),
    ('COR_FLUID', 'Coronal', 1),
    ('COR_STRUCT', 'Coronal', 0),
    ('AX_FLUID', 'Axial', 1),
    ('AX_STRUCT', 'Axial', 0),
]
HEADER_TAGS = [
    'ImagePositionPatient', 'ImageOrientationPatient', 'InstanceNumber',
    'PixelSpacing', 'Rows', 'Columns', 'PhotometricInterpretation',
    'RescaleSlope', 'RescaleIntercept'
]


def _series_file_count(folder: Path) -> int:
    try:
        return sum(1 for _ in folder.glob('*.dcm'))
    except Exception:
        return 0


def _pick_series(rows: list[dict], plane: str, fluid: int, study_dir: Path):
    cand = [r for r in rows if str(r.get('Anatomical_Plane')) == plane]
    preferred = [r for r in cand if int(r.get('Fluid_Sensitive', r.get('Fat_Suppression', 0)) or 0) == fluid]
    pool = preferred if preferred else []
    if not pool:
        return None
    scored = []
    for r in pool:
        uid = str(r['SeriesInstanceUID'])
        scored.append((_series_file_count(study_dir / uid), uid, r))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2] if scored else None


def _ordered_files(folder: Path):
    paths = list(folder.glob('*.dcm'))
    if not paths:
        return [], 'empty'
    recs = []
    ref_normal = None
    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=HEADER_TAGS)
            iop = np.asarray(getattr(ds, 'ImageOrientationPatient', []), dtype=float)
            ipp = np.asarray(getattr(ds, 'ImagePositionPatient', []), dtype=float)
            inst = float(getattr(ds, 'InstanceNumber', 0) or 0)
            ps = np.asarray(getattr(ds, 'PixelSpacing', []), dtype=float)
            ps0 = float(ps[0]) if ps.shape == (2,) and np.all(ps > 0) else 0.5
            normal = None
            if iop.shape == (6,):
                normal = np.cross(iop[:3], iop[3:])
                nn = float(np.linalg.norm(normal))
                if nn > 0:
                    normal = normal / nn
                    if ref_normal is None:
                        ref_normal = normal.copy()
            recs.append((path, ipp, normal, inst, ps0))
        except Exception:
            recs.append((path, None, None, 0.0, 0.5))
    if ref_normal is not None:
        usable = []
        for path, ipp, normal, inst, ps in recs:
            if isinstance(ipp, np.ndarray) and ipp.shape == (3,) and np.isfinite(ipp).all():
                usable.append((float(np.dot(ipp, ref_normal)), path, ps))
        if len(usable) == len(recs):
            usable.sort(key=lambda x: x[0])
            return [(p, ps) for _, p, ps in usable], 'geometry'
    fallback = sorted([(inst, path, ps) for path, ipp, normal, inst, ps in recs], key=lambda x: (x[0], str(x[1])))
    return [(p, ps) for _, p, ps in fallback], 'instance_fallback'


def _read_px(path: Path):
    ds = pydicom.dcmread(str(path))
    arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f'expected 2D frame, got {arr.shape}')
    if str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
        arr = arr.max() - arr
    ps = np.asarray(getattr(ds, 'PixelSpacing', []), dtype=float)
    spacing = float(ps[0]) if ps.shape == (2,) and np.all(ps > 0) else None
    return arr, spacing


def _crop_resize(arr: np.ndarray, spacing: float | None):
    h, w = arr.shape
    ps = float(spacing) if spacing is not None and spacing > 0 else CROP_MM / min(h, w)
    side = max(1, int(round(CROP_MM / max(ps, 1e-4))))
    side = min(side, h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = arr[y0:y0+side, x0:x0+side]
    return cv2.resize(crop, (IMG, IMG), interpolation=cv2.INTER_AREA)


def _build_study(index: int, uid: str, rows: list[dict]):
    study_dir = ROOT / 'train_series' / uid
    out = np.zeros((len(SLOTS), N_SLICE, IMG, IMG), dtype=np.uint8)
    mask = np.zeros((len(SLOTS), N_SLICE), dtype=np.uint8)
    stats = {'index': index, 'uid': uid, 'geometry_slots': 0, 'fallback_slots': 0, 'missing_slots': 0, 'decode_failures': 0}
    for sidx, (name, plane, fluid) in enumerate(SLOTS):
        rec = _pick_series(rows, plane, fluid, study_dir)
        if rec is None:
            stats['missing_slots'] += 1
            continue
        suid = str(rec['SeriesInstanceUID'])
        ordered, method = _ordered_files(study_dir / suid)
        if not ordered:
            stats['missing_slots'] += 1
            continue
        stats['geometry_slots' if method == 'geometry' else 'fallback_slots'] += 1
        n = len(ordered)
        lo = int(round(SPAN[0] * max(n-1, 0)))
        hi = int(round(SPAN[1] * max(n-1, 0)))
        hi = max(lo, min(hi, n-1))
        picks = np.linspace(lo, hi, N_SLICE).round().astype(int) if n > 1 else np.zeros(N_SLICE, dtype=int)
        raw = []
        for p in picks:
            path, header_ps = ordered[min(int(p), n-1)]
            try:
                arr, px_ps = _read_px(path)
                raw.append((_crop_resize(arr, px_ps or header_ps), True))
            except Exception:
                stats['decode_failures'] += 1
                raw.append((None, False))
        valid = [a for a, ok in raw if ok and a is not None]
        if not valid:
            continue
        sample = np.concatenate([a[::4, ::4].ravel() for a in valid]).astype(np.float32)
        qlo, qhi = np.percentile(sample, [2.0, 98.0])
        scale = max(float(qhi - qlo), 1e-6)
        for j, (a, ok) in enumerate(raw):
            if not ok or a is None:
                continue
            z = np.clip((a - qlo) / scale, 0.0, 1.0)
            out[sidx, j] = np.rint(z * 255.0).astype(np.uint8)
            mask[sidx, j] = 1
    return index, out, mask, stats


def sha256(path: Path, chunk=32 << 20):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main():
    t0 = time.time()
    if not (ROOT / 'train.csv').is_file():
        raise FileNotFoundError(ROOT / 'train.csv')
    train = pd.read_csv(ROOT / 'train.csv', dtype={'StudyInstanceUID': str})
    ser = pd.read_csv(ROOT / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    uids = train['StudyInstanceUID'].astype(str).tolist()
    grouped = {u: g.to_dict('records') for u, g in ser.groupby('StudyInstanceUID')}
    shape = (len(uids), len(SLOTS), N_SLICE, IMG, IMG)
    cache = np.lib.format.open_memmap(CACHE, mode='w+', dtype=np.uint8, shape=shape)
    masks = np.lib.format.open_memmap(MASKS, mode='w+', dtype=np.uint8, shape=(len(uids), len(SLOTS), N_SLICE))
    pd.DataFrame({'index': np.arange(len(uids)), 'StudyInstanceUID': uids}).to_csv(UIDS, index=False)
    failures = []
    stats_rows = []
    print('cache shape', shape, 'expected GiB', np.prod(shape)/(1024**3), 'workers', WORKERS, flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_build_study, i, uid, grouped.get(uid, [])): (i, uid) for i, uid in enumerate(uids)}
        done = 0
        for fut in as_completed(futs):
            i, uid = futs[fut]
            try:
                idx, arr, msk, st = fut.result()
                cache[idx] = arr
                masks[idx] = msk
                stats_rows.append(st)
            except Exception as exc:
                failures.append({'index': i, 'StudyInstanceUID': uid, 'error': f'{type(exc).__name__}: {exc}', 'trace': traceback.format_exc()[-1000:]})
            done += 1
            if done % 100 == 0 or done == len(uids):
                elapsed = time.time() - t0
                print(f'{done}/{len(uids)} | {elapsed/60:.1f} min | eta {(elapsed/max(done,1))*(len(uids)-done)/60:.1f} min | failures {len(failures)}', flush=True)
    cache.flush(); masks.flush()
    del cache, masks
    pd.DataFrame(failures).to_csv(FAILURES, index=False)
    st = pd.DataFrame(stats_rows)
    st.to_csv(OUT / 'train_cache_study_stats.csv', index=False)
    manifest = {
        'contract': 'rsna_knee_train_cache_6slot_9slice_224_v1',
        'train_only': True,
        'studies': len(uids),
        'shape': list(shape),
        'dtype': 'uint8',
        'slots': [{'name': n, 'plane': p, 'fluid_sensitive': int(f)} for n,p,f in SLOTS],
        'sampling_span': list(SPAN),
        'crop_mm': CROP_MM,
        'image_size': IMG,
        'normalization': 'per selected series, sampled slices pooled, percentile 2-98 -> uint8',
        'ordering': 'IOP/IPP reference-normal projection; InstanceNumber fallback only when geometry incomplete',
        'series_selection': 'longest matching plane + Fluid_Sensitive slot; no cross-fluid fallback',
        'failure_count': len(failures),
        'geometry_slots': int(st['geometry_slots'].sum()) if len(st) else 0,
        'fallback_slots': int(st['fallback_slots'].sum()) if len(st) else 0,
        'missing_slots': int(st['missing_slots'].sum()) if len(st) else 0,
        'decode_failures': int(st['decode_failures'].sum()) if len(st) else 0,
        'elapsed_seconds': time.time()-t0,
        'files': {
            CACHE.name: {'bytes': CACHE.stat().st_size},
            MASKS.name: {'bytes': MASKS.stat().st_size},
            UIDS.name: {'bytes': UIDS.stat().st_size},
        },
        'limitations': [
            'Center crop only; no learned/localized knee crop.',
            'One series per slot chosen by file count; later experiments may learn series selection.',
            'Cache is for rapid local experimentation and does not replace hidden-test runtime validation.'
        ]
    }
    print('hashing cache...', flush=True)
    manifest['files'][CACHE.name]['sha256'] = sha256(CACHE)
    manifest['files'][MASKS.name]['sha256'] = sha256(MASKS)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
