from __future__ import annotations
import os, json, time, hashlib, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pydicom
import cv2

ROOT = Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
OUT = Path('/kaggle/working')
PLANES = ['Sagittal', 'Coronal', 'Axial']
DEPTH = 16
SIZE = 256
NSHARD = 8
HDR_TAGS = ['ImagePositionPatient','ImageOrientationPatient','InstanceNumber','RescaleSlope','RescaleIntercept']


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dcm_files(folder: Path):
    return sorted(folder.glob('*.dcm'))


def choose_series(rows, study_dir: Path, plane: str):
    cand = [r for r in rows if str(r.get('Anatomical_Plane')) == plane]
    if not cand:
        return None, 0, 0
    fluid = [r for r in cand if int(r.get('Fluid_Sensitive', 0) or 0) == 1]
    pool = fluid if fluid else cand
    scored = []
    for r in pool:
        su = str(r['SeriesInstanceUID'])
        n = len(dcm_files(study_dir / su))
        scored.append((n, su, r))
    scored.sort(key=lambda z: (-z[0], z[1]))
    n, su, r = scored[0]
    return r, n, int(bool(fluid))


def ordered_files(folder: Path):
    fs = dcm_files(folder)
    rec = []
    for f in fs:
        try:
            d = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=HDR_TAGS)
            iop = np.asarray(getattr(d, 'ImageOrientationPatient', []), np.float64)
            ipp = np.asarray(getattr(d, 'ImagePositionPatient', []), np.float64)
            inst = float(getattr(d, 'InstanceNumber', np.nan))
            rec.append((f, iop, ipp, inst))
        except Exception:
            rec.append((f, np.array([]), np.array([]), np.nan))
    valid = [(iop, ipp) for _, iop, ipp, _ in rec if iop.shape == (6,) and ipp.shape == (3,)]
    if len(valid) == len(rec) and len(rec) > 0:
        row = valid[0][0][:3]
        col = valid[0][0][3:]
        normal = np.cross(row, col)
        nn = np.linalg.norm(normal)
        if nn > 1e-8:
            normal /= nn
            vals = [(float(np.dot(ipp, normal)), f) for f, iop, ipp, inst in rec]
            vals.sort(key=lambda z: z[0])
            return [f for _, f in vals], 'geometry'
    if all(np.isfinite(inst) for _, _, _, inst in rec) and rec:
        return [r[0] for r in sorted(rec, key=lambda z: z[3])], 'instance'
    return fs, 'filename_last_resort'


def decode_one(f: Path):
    d = pydicom.dcmread(str(f))
    x = d.pixel_array.astype(np.float32)
    slope = float(getattr(d, 'RescaleSlope', 1.0) or 1.0)
    intercept = float(getattr(d, 'RescaleIntercept', 0.0) or 0.0)
    x = x * slope + intercept
    return x


def center_indices(n: int, depth: int):
    if n <= 0:
        return []
    if n >= depth:
        start = (n - depth) // 2
        return list(range(start, start + depth))
    # Robust fallback for short series: repeat edge-nearest samples to 16.
    return np.linspace(0, n - 1, depth).round().astype(int).tolist()


def make_volume(folder: Path):
    fs, order_mode = ordered_files(folder)
    if not fs:
        return None, order_mode, 0, 0
    idx = center_indices(len(fs), DEPTH)
    ims = []
    decode_fail = 0
    last = None
    for j in idx:
        try:
            im = decode_one(fs[j])
            im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            last = im
        except Exception:
            decode_fail += 1
            im = last
        if im is None:
            im = np.zeros((SIZE, SIZE), np.float32)
        ims.append(im)
    vol = np.stack(ims).astype(np.float32)
    lo = float(np.nanmin(vol)); hi = float(np.nanmax(vol))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.zeros_like(vol, dtype=np.uint8)
    else:
        out = np.clip((vol - lo) / (hi - lo), 0, 1)
        out = np.rint(out * 255.0).astype(np.uint8)
    return out, order_mode, len(fs), decode_fail


def worker(task):
    idx, uid, rows = task
    study_dir = ROOT / 'train_series' / uid
    arr = np.zeros((3, DEPTH, SIZE, SIZE), np.uint8)
    mask = np.zeros(3, np.uint8)
    info = []
    for pi, plane in enumerate(PLANES):
        r, count_hint, preferred_fluid = choose_series(rows, study_dir, plane)
        if r is None:
            info.append({'index':idx,'StudyInstanceUID':uid,'plane':plane,'series_uid':'','preferred_fluid':0,'missing':1,'n_slices':0,'order_mode':'missing','decode_fail':0})
            continue
        su = str(r['SeriesInstanceUID'])
        try:
            vol, mode, n, fail = make_volume(study_dir / su)
            if vol is not None:
                arr[pi] = vol
                mask[pi] = 1
            info.append({'index':idx,'StudyInstanceUID':uid,'plane':plane,'series_uid':su,'preferred_fluid':preferred_fluid,'missing':0,'n_slices':n,'order_mode':mode,'decode_fail':fail})
        except Exception as e:
            info.append({'index':idx,'StudyInstanceUID':uid,'plane':plane,'series_uid':su,'preferred_fluid':preferred_fluid,'missing':0,'n_slices':count_hint,'order_mode':'error','decode_fail':DEPTH,'error':type(e).__name__})
    return idx, arr, mask, info


def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT/'train.csv', dtype={'StudyInstanceUID':str})
    se = pd.read_csv(ROOT/'train_series.csv', dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    groups = {u:g.to_dict('records') for u,g in se.groupby('StudyInstanceUID')}
    uids = tr.StudyInstanceUID.astype(str).tolist()
    n = len(uids)
    bounds = np.linspace(0, n, NSHARD + 1).round().astype(int)
    shards = []
    for s in range(NSHARD):
        a,b = int(bounds[s]), int(bounds[s+1])
        p = OUT / f'orthodiff_cache_{s:02d}_{a:04d}_{b:04d}.npy'
        mm = np.lib.format.open_memmap(p, mode='w+', dtype=np.uint8, shape=(b-a,3,DEPTH,SIZE,SIZE))
        shards.append((a,b,p,mm))
    masks = np.lib.format.open_memmap(OUT/'orthodiff_masks.npy', mode='w+', dtype=np.uint8, shape=(n,3))
    tasks = [(i,uids[i],groups.get(uids[i],[])) for i in range(n)]
    rows = []
    workers = min(6, max(2, (os.cpu_count() or 4)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k,(i,arr,mask,info) in enumerate(ex.map(worker,tasks,chunksize=1),1):
            for a,b,p,mm in shards:
                if a <= i < b:
                    mm[i-a] = arr
                    break
            masks[i] = mask
            rows.extend(info)
            if k % 100 == 0:
                print(f'{k}/{n} elapsed={(time.time()-t0)/60:.1f}m', flush=True)
    for _,_,_,mm in shards: mm.flush()
    masks.flush()
    pd.DataFrame({'StudyInstanceUID':uids}).to_csv(OUT/'orthodiff_uids.csv',index=False)
    stat = pd.DataFrame(rows)
    stat.to_csv(OUT/'orthodiff_stats.csv',index=False)
    manifest = {'contract':'orthodiffusion_3plane_center16_256_uint8_v1','studies':n,'shape':[n,3,DEPTH,SIZE,SIZE],'planes':PLANES,'selection':'prefer Fluid_Sensitive=1 within plane, then longest DICOM count; fallback longest plane series','ordering':'IOP/IPP geometry; InstanceNumber fallback; filename last resort recorded','normalization':'selected 16-slice volume resized to 256 then global minmax to uint8; downstream maps uint8 to [-1,1]','workers':workers,'shards':[]}
    for a,b,p,_ in shards:
        manifest['shards'].append({'file':p.name,'start':a,'stop':b,'bytes':p.stat().st_size,'sha256':sha256_file(p)})
    manifest['mask_sha256'] = sha256_file(OUT/'orthodiff_masks.npy')
    manifest['elapsed_seconds'] = time.time()-t0
    manifest['plane_present'] = {PLANES[j]:int(masks[:,j].sum()) for j in range(3)}
    manifest['order_modes'] = stat.order_mode.value_counts(dropna=False).to_dict()
    manifest['decode_fail_total'] = int(stat.decode_fail.fillna(0).sum())
    (OUT/'orthodiff_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__ == '__main__':
    main()
