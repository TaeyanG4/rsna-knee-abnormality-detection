"""Anchored v4 384 CPU subset cache, v1.0.0; never builds >232 studies.
Changelog: initial pinned CStudies adapter and compact_train mmap format.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.l1_rco_stage0_v1 import (ROOT, WORK, CACHE, RAW, CODE, SOURCE, ANATOMY, UID,
    VERSION, validate, read, write, sha, digest, need)

def key(uid):
    return hashlib.sha256(uid.encode()).hexdigest()[:24]

def safe_output(path):
    path = Path(path).resolve()
    need(path == CACHE.resolve() or path.is_relative_to(WORK.resolve()), 'cache output outside approved fixed locations')
    return path

def clear_owned(path):
    path = Path(path).resolve()
    need(path.is_relative_to((WORK / 'cache-work').resolve()) and path != (WORK / 'cache-work').resolve(), 'unsafe owned scratch cleanup')
    if path.exists():
        for attempt in range(40):
            try:
                shutil.rmtree(path); break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(min(.05 * (attempt + 1), 1))


def array_hash(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def geometry_checked(original, records):
    def wrapped(*args, **kwargs):
        g = original(*args, **kwargs)
        x, y, x1, y1 = g.crop_xyxy
        px, py, px1, py1 = g.valid_xyxy
        oh, ow = g.resized_shape
        h, w = args[0]; size = args[2]
        need(0 <= x < x1 <= w and 0 <= y < y1 <= h, 'invalid source crop')
        need(0 <= px < px1 <= size and 0 <= py < py1 <= size and (px1 - px, py1 - py) == (ow, oh), 'invalid rendered valid area')
        records.append(ow * oh / size**2)
        return g
    return wrapped

def build(reg, out, limit=None, threads=2):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    validate(reg)
    out = safe_output(out); out.mkdir(parents=True, exist_ok=True)
    ids = reg['train'] + reg['holdout']
    if limit is not None:
        need(limit == 3, 'CPU proof is exactly two train + one holdout')
        ids = reg['train'][:2] + reg['holdout'][:1]
    contract = {'contract_id': reg['contract_id'], 'ids': ids, 'size': 384, 'views': 'v4'}
    if (out / 'contract.json').exists():
        need(read(out / 'contract.json') == contract, 'existing cache belongs to another registration/subset')
    else:
        need(not any(out.iterdir()), 'refuse nonempty unregistered cache destination')
        write(out / 'contract.json', contract)
    if (out / 'receipt.json').exists():
        return check_cache(reg, out, ids)
    from tools.exp189_s6_port import _diagnostic_source
    stage = WORK / 'cache-work/runtime-parent'
    clear_owned(stage); stage.mkdir(parents=True)
    runtime = _diagnostic_source(ROOT / SOURCE, stage)
    sys.path.insert(0, str(ROOT / CODE)); sys.path.insert(0, str(runtime))
    import compact_infer as ci
    import px_cache
    geometry = []
    from training import input_provenance
    original_geometry = input_provenance.view_geometry
    input_provenance.view_geometry = geometry_checked(original_geometry, geometry)
    t_all = time.perf_counter()
    recs = []
    ds = None
    with zipfile.ZipFile(ROOT / RAW) as z:
        wanted = set(ids)
        members = {u: [] for u in ids}
        for info in z.infolist():
            p = info.filename.split('/')
            if len(p) == 4 and p[0] == 'train_series' and p[1] in wanted and p[3].endswith('.dcm'):
                members[p[1]].append(info)
        series = {u: [] for u in ids}
        for row in csv.DictReader(io.StringIO(z.read('train_series.csv').decode('utf-8'))):
            if row[UID] in wanted:
                series[row[UID]].append(row)
        for uid in ids:
            t0 = time.perf_counter()
            geometry.clear()
            base = out / 'shards' / key(uid); base.mkdir(parents=True, exist_ok=True)
            if (base / 'receipt.json').exists():
                rec = read(base / 'receipt.json')
                need(rec['contract_id'] == reg['contract_id'], 'shard contract')
                for name, h in rec['hashes'].items():
                    need(sha(base / name) == h, 'shard checksum')
                recs.append(rec); continue
            owned = WORK / 'cache-work/study'
            clear_owned(owned); data = owned / 'data'; data.mkdir(parents=True)
            identity = sorted([[i.filename, i.CRC, i.file_size, i.compress_size] for i in members[uid]])
            need(digest(identity) == reg['raw'][uid]['digest'], 'raw archive identity changed')
            need(shutil.disk_usage(out).free > 4 * 2**30 and shutil.disk_usage(WORK).free > 4 * 2**30, '4 GiB working margin missing')
            for i in members[uid]:
                f = data / i.filename
                need(f.resolve().is_relative_to(data.resolve()), 'unsafe archive path')
                f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(z.read(i))
            need(series[uid], 'missing series metadata')
            with (data / 'train_series.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(series[uid][0])); w.writeheader(); w.writerows(series[uid])
            (data / 'train.csv').write_text(UID + '\n' + uid + '\n', encoding='utf-8')
            native = owned / 'native'
            if ds is None:
                ds = ci.CStudies(data, 'train', ids, stage, ROOT / CODE, 384,
                                 eval_stride=1, views='v4', slices=3, phys_mm=3.5,
                                 anchor_weights=ROOT / ANATOMY, side_model=None, device='cpu')
                ds._setup(); torch.set_num_threads(threads)
            ds.data = str(data)
            ds._native_and_geometry(uid, native)
            st = read(native / 'studies' / (key(uid) + '.json'))
            need(not st['errors'] and st['series'] and all(not s['errors'] for s in st['series']), 'native decode/geometry failure')
            anchor = ds.bank.record(uid, st, geo_dir=str(native / 'geometry'))
            write(native / 'anchors' / (key(uid) + '.json'), anchor)
            ds.builder.cache = str(native); ds.builder.anchors = str(native / 'anchors')
            item = ds.builder.build(uid)
            need(not item['error'] and not item['skipped_series'] and len(item['px']), 'pixel builder incomplete')
            px, meta = item['px'].numpy(), item['meta'].numpy()
            need(px.dtype == np.uint8 and px.shape[1:] == (384, 384) and meta.shape == (len(px), 12) and np.isfinite(meta).all(), 'pixel contract')
            np.save(base / 'px.npy', px); np.save(base / 'meta.npy', meta.astype(np.float32))
            rec = {'contract_id': reg['contract_id'], 'uid': uid, 'rows': len(px), 'blocks': item['blocks'],
                   'removed': item['removed'], 'bytes': px.nbytes + meta.nbytes,
                   'seconds': time.perf_counter() - t0, 'anchor_method': anchor['method'], 'anchor_why': anchor['why'],
                   'hashes': {name: sha(base / name) for name in ['px.npy', 'meta.npy']}}
            rec['geometry'] = {'checked_calls': len(geometry), 'valid_area_min': min(geometry), 'valid_area_max': max(geometry), 'source_crop_and_padding_valid': True}
            if limit is not None:
                ds.eval_stride = 3
                served = ds[ids.index(uid)]
                need(not served['error'], 'serving CPU parity render failed')
                rec['serving'] = {name: array_hash(served[name].numpy()) for name in ('images', 'meta')}
                del served
            write(base / 'receipt.json', rec); recs.append(rec)
            print({'study': key(uid), 'rows': rec['rows'], 'seconds': round(rec['seconds'], 2)}, flush=True)
    n = sum(r['rows'] for r in recs)
    need(shutil.disk_usage(out).free >= n * (384**2 + 48) + 2 * 2**30, 'assembly needs duplicate subset bytes + 2GiB')
    px = np.lib.format.open_memmap(out / 'px.u8.npy', mode='w+', dtype=np.uint8, shape=(n, 384, 384))
    meta = np.lib.format.open_memmap(out / 'meta.f32.npy', mode='w+', dtype=np.float32, shape=(n, 12))
    index = {'order': ids, 'done': {}, 'blocks': {}, 'cursor': n, 'errors': {}, 'meta_cols': px_cache.META_COLS,
             'config': {'size': 384, 'views': 'v4', 'anchors': str(ROOT / ANATOMY)}, 'contract_id': reg['contract_id']}
    cur = 0
    for r in recs:
        uid = r['uid']; base = out / 'shards' / key(uid); end = cur + r['rows']
        px[cur:end] = np.load(base / 'px.npy', mmap_mode='r'); meta[cur:end] = np.load(base / 'meta.npy', mmap_mode='r')
        blocks = json_copy(r['blocks'])
        for b in blocks:
            b['views'] = [[v, cur + off] for v, off in b['views']]
        index['blocks'][uid] = blocks; index['done'][uid] = [cur, r['rows'], len(blocks)]; cur = end
    px.flush(); meta.flush(); del px, meta
    write(out / 'index.json', index)
    if limit is not None:
        import compact_train as ct
        evaluation = ct.Studies(str(out), ids, 64, 'eval', 9000, eval_stride=3, slices=3, phys_mm=3.5)
        for i, rec in enumerate(recs):
            img, metadata, _ = evaluation[i]
            need(rec['serving'] == {'images': array_hash(img.numpy()), 'meta': array_hash(metadata.numpy())}, 'serving/cache pixel or metadata mismatch')
        del evaluation
    bytes_ = sum((out / p).stat().st_size for p in ['px.u8.npy', 'meta.f32.npy', 'index.json'])
    receipt = {'version': VERSION, 'contract_id': reg['contract_id'], 'ids': ids, 'studies': len(ids),
               'bytes': bytes_, 'bytes_per_study': bytes_ / len(ids), 'seconds_per_study': sum(r['seconds'] for r in recs) / len(ids),
               'elapsed_this_invocation': time.perf_counter() - t_all, 'cpu_threads': threads,
               'geometry': [r['geometry'] for r in recs],
               'serving_cache_parity': {'status': 'PASS' if limit is not None else 'CPU_PROOF_SEPARATE', 'studies': len(ids) if limit is not None else 0, 'comparison': 'exact SHA256 of all eval-stride3 physical-window pixels and metadata'},
               'temporary_disk_factor': 'approximately 2x retained shard + mmap; no full cache authorization',
               'projection_status': 'UNMEASURED until K0; these are subset cache measurements only',
               'hashes': {p: sha(out / p) for p in ['px.u8.npy', 'meta.f32.npy', 'index.json']}, 'cuda_initialized': torch.cuda.is_initialized()}
    need(not receipt['cuda_initialized'], 'cache used CUDA')
    input_provenance.view_geometry = original_geometry
    write(out / 'receipt.json', receipt)
    return receipt

def json_copy(x):
    import json
    return json.loads(json.dumps(x))

def check_cache(reg, out, ids=None):
    out = safe_output(out); r = read(out / 'receipt.json')
    need(r['contract_id'] == reg['contract_id'] and r['ids'] == (ids or reg['train'] + reg['holdout']), 'cache roster/contract')
    for p, h in r['hashes'].items():
        need(sha(out / p) == h, 'cache checksum: ' + p)
    index = read(out / 'index.json')
    need(set(index['done']) == set(r['ids']) and not index['errors'], 'incomplete index')
    return r

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=CACHE); ap.add_argument('--cpu-proof', action='store_true')
    a = ap.parse_args()
    r = build(read(WORK / 'registration.json'), a.out, 3 if a.cpu_proof else None)
    print({k: r[k] for k in ['studies', 'bytes_per_study', 'seconds_per_study', 'cuda_initialized']})

if __name__ == '__main__':
    main()
