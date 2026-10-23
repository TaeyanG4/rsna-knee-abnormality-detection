"""H-B-182 CPU full-cohort anchored cache v1.1.0.

Changelog: spawned study workers, sealed full roster, resumable shard commits,
bounded disk admission, deterministic assembly, six-study serial equivalence.
Pre-release review fixes: bind proof hash, lock serial reference, phase progress.
v1.0.0 files remain untouched. Full execution belongs to B after reviewed K0.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import l1_rco_stage0_v1 as s
from tools.l1_rco_cache_v1 import key, geometry_checked, json_copy
from tools.job_runtime import file_lock

VERSION = '1.1.0'
WORK = ROOT / '.local/scratch/l1-rco/fullcache'
OUT = Path('G:/temp_data/rsna-knee/l1rco-cache-full')
REG = WORK / 'registration.json'
GIB = 2**30
FILES = ('px.npy', 'meta.npy')
LAYOUT = ('px.u8.npy', 'meta.f32.npy', 'index.json')
META_COLS = ['series_idx', 'slot', 'plane', 'view_id', 'rel_pos', 'log_spacing',
             'extent_130', 'fluid', 'fatsat', 'slice_order', 'n_slices', 'real_series_idx']
need, read, write, sha, digest = s.need, s.read, s.write, s.sha, s.digest


def pins():
    # Production register/build never open/hash labels. The untouched v1 serial
    # proof alone retains legacy validation's target-file byte hash (no parsing).
    p = {k: v for k, v in s.PINS.items() if k != s.TARGETS}
    for name in ('l1_rco_stage0_v1.py', 'l1_rco_cache_v1.py', 'l1_rco_runtime_v1.py',
                 'l1_rco_STAGE0_v1.md', 'exp189_s6_port.py', 's6plus.py',
                 'l1_rco_fullcache_v110.py'):
        p['tools/' + name] = sha(ROOT / 'tools' / name)
    return p


def verify_pins(p):
    need(s.TARGETS not in p, 'label input forbidden')
    for name, value in p.items():
        need(sha(ROOT / name) == value, 'input/code changed: ' + name)
    from tools.exp189_s6_port import verify_code
    verify_code(ROOT / s.CODE, ROOT / s.ANATOMY)


def register():
    p = pins(); verify_pins(p)
    full = read(ROOT / s.REG); roster = s.validate_roster(full)
    order = lambda ids: sorted(ids, key=lambda u: __import__('hashlib').sha256(('L1-RCO-v1:' + u).encode()).hexdigest())
    tr, va = order(full['train']), order(full['val'])
    from tools.s6plus import raw_index
    with zipfile.ZipFile(ROOT / s.RAW) as z:
        series_sha = __import__('hashlib').sha256(z.read('train_series.csv')).hexdigest()
    reg = {'version': VERSION, 'pins': p, 'roster': roster, 'train': tr, 'holdout': va,
           'raw': raw_index(ROOT / s.RAW, tr + va), 'series_sha256': series_sha,
           'recipe': {'size': 384, 'views': 'v4', 'slices': 3, 'phys_mm': 3.5,
                      'threads_per_worker': 2, 'side_model': None},
           'exposure': 'EXPOSED_EXPLORATORY', 'clean_oof': False}
    reg['contract_id'] = digest(reg)
    if REG.exists():
        need(read(REG) == reg, 'registration differs; preserve previous artifacts and version the change')
    else:
        write(REG, reg)
    return reg


def validate(reg):
    need(reg['version'] == VERSION and reg['contract_id'] == digest({k: v for k, v in reg.items() if k != 'contract_id'}), 'registration identity')
    need(reg['pins'] == pins(), 'registration pins differ')
    verify_pins(reg['pins'])
    full = read(ROOT / s.REG)
    need(reg['roster'] == s.validate_roster(full), 'roster identity')
    need(len(reg['train']) == 3190 and len(reg['holdout']) == 977 and
         set(reg['train']) == set(full['train']) and set(reg['holdout']) == set(full['val']), 'full cohort mismatch')
    need(set(reg['raw']) == set(reg['train'] + reg['holdout']), 'raw roster mismatch')
    return reg


def safe_output(out):
    out = Path(out).resolve()
    need(out == OUT.resolve() or out.is_relative_to(WORK.resolve()), 'unapproved cache location')
    return out


def clean_owned(path):
    path = Path(path).resolve(); parent = (WORK / 'workers').resolve()
    need(path.is_relative_to(parent) and path != parent, 'unsafe scratch cleanup')
    if path.exists():
        for attempt in range(40):
            try:
                shutil.rmtree(path); return
            except PermissionError:
                if attempt == 39: raise
                time.sleep(min(.05 * (attempt + 1), 1))


def disk_gate(out, workers, assembly_bytes=0):
    # Reserve all workers' concurrent extraction/output margins, not just one.
    output_need = (workers * 4 * GIB) if not assembly_bytes else assembly_bytes + 2 * GIB
    scratch_need = workers * 4 * GIB
    if Path(out).anchor.lower() == WORK.anchor.lower():
        need(shutil.disk_usage(out).free >= output_need + scratch_need, 'combined output/scratch disk margin')
    else:
        need(shutil.disk_usage(out).free >= output_need, 'output disk margin')
        need(shutil.disk_usage(WORK).free >= scratch_need, 'worker scratch disk margin')


def validate_shard(reg, out, uid):
    import numpy as np
    base = Path(out) / 'shards' / key(uid)
    if not (base / 'receipt.json').exists(): return None
    rec = read(base / 'receipt.json')
    need(rec['contract_id'] == reg['contract_id'] and rec['uid'] == uid and rec['rows'] > 0, 'shard identity')
    need(set(rec['hashes']) == set(FILES), 'shard payload list')
    for name in FILES:
        need(sha(base / name) == rec['hashes'][name], 'shard checksum: ' + key(uid))
    px = np.load(base / 'px.npy', mmap_mode='r'); meta = np.load(base / 'meta.npy', mmap_mode='r')
    need(px.dtype == np.uint8 and px.shape == (rec['rows'], 384, 384) and
         meta.dtype == np.float32 and meta.shape == (rec['rows'], 12) and np.isfinite(meta).all(), 'shard layout')
    for block in rec['blocks']:
        need(all(0 <= off < rec['rows'] for _, off in block['views']), 'shard view offset')
    return rec


class Renderer:
    """Same calls/order/precision as v1 build; only study ownership is parallel."""
    def __init__(self, reg, ids, slot):
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        self.torch = torch; self.reg = reg; self.ids = ids
        self.home = WORK / 'workers' / slot
        clean_owned(self.home); self.home.mkdir(parents=True)
        from tools.exp189_s6_port import _diagnostic_source
        runtime = _diagnostic_source(ROOT / s.SOURCE, self.home / 'runtime')
        sys.path.insert(0, str(ROOT / s.CODE)); sys.path.insert(0, str(runtime))
        import compact_infer
        self.ci = compact_infer
        from training import input_provenance
        self.geometry = []
        input_provenance.view_geometry = geometry_checked(input_provenance.view_geometry, self.geometry)
        self.z = zipfile.ZipFile(ROOT / s.RAW)
        wanted = set(ids); self.members = {u: [] for u in ids}
        for info in self.z.infolist():
            parts = info.filename.split('/')
            if len(parts) == 4 and parts[0] == 'train_series' and parts[1] in wanted and parts[3].endswith('.dcm'):
                self.members[parts[1]].append(info)
        series_bytes = self.z.read('train_series.csv')
        need(__import__('hashlib').sha256(series_bytes).hexdigest() == reg['series_sha256'], 'series metadata changed')
        self.series = {u: [] for u in ids}
        for row in csv.DictReader(io.StringIO(series_bytes.decode('utf-8'))):
            if row[s.UID] in wanted: self.series[row[s.UID]].append(row)
        self.ds = None

    def study(self, uid, out, workers):
        import numpy as np
        t0 = time.perf_counter(); reg = self.reg
        existing = validate_shard(reg, out, uid)
        if existing: return existing
        disk_gate(out, workers)
        self.geometry.clear(); base = Path(out) / 'shards' / key(uid); base.mkdir(parents=True, exist_ok=True)
        owned = self.home / 'study'; clean_owned(owned); data = owned / 'data'; data.mkdir(parents=True)
        entries = self.members[uid]
        identity = sorted([[i.filename, i.CRC, i.file_size, i.compress_size] for i in entries])
        need(digest(identity) == reg['raw'][uid]['digest'], 'raw archive identity changed')
        for info in entries:
            target = data / info.filename
            need(target.resolve().is_relative_to(data.resolve()), 'unsafe archive path')
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(self.z.read(info))
        need(self.series[uid], 'missing series metadata')
        with (data / 'train_series.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(self.series[uid][0])); w.writeheader(); w.writerows(self.series[uid])
        (data / 'train.csv').write_text(s.UID + '\n' + uid + '\n', encoding='utf-8')
        native = owned / 'native'
        if self.ds is None:
            self.ds = self.ci.CStudies(data, 'train', self.ids, self.home / 'runtime', ROOT / s.CODE, 384,
                                      eval_stride=1, views='v4', slices=3, phys_mm=3.5,
                                      anchor_weights=ROOT / s.ANATOMY, side_model=None, device='cpu')
            self.ds._setup(); self.torch.set_num_threads(2)
        ds = self.ds; ds.data = str(data); ds._native_and_geometry(uid, native)
        st = read(native / 'studies' / (key(uid) + '.json'))
        need(not st['errors'] and st['series'] and all(not x['errors'] for x in st['series']), 'native decode/geometry failure')
        anchor = ds.bank.record(uid, st, geo_dir=str(native / 'geometry'))
        write(native / 'anchors' / (key(uid) + '.json'), anchor)
        ds.builder.cache = str(native); ds.builder.anchors = str(native / 'anchors')
        item = ds.builder.build(uid)
        need(not item['error'] and not item['skipped_series'] and len(item['px']), 'pixel builder incomplete')
        px, meta = item['px'].numpy(), item['meta'].numpy()
        need(px.dtype == np.uint8 and px.shape[1:] == (384, 384) and meta.shape == (len(px), 12) and np.isfinite(meta).all(), 'pixel contract')
        for name, arr in [('px.npy', px), ('meta.npy', meta.astype(np.float32))]:
            with (base / (name + '.tmp')).open('wb') as f:
                np.save(f, arr); f.flush(); os.fsync(f.fileno())
            s.replace(base / (name + '.tmp'), base / name)
        rec = {'contract_id': reg['contract_id'], 'uid': uid, 'rows': len(px), 'blocks': item['blocks'],
               'removed': item['removed'], 'bytes': px.nbytes + meta.nbytes,
               'seconds': time.perf_counter() - t0, 'anchor_method': anchor['method'], 'anchor_why': anchor['why'],
               'hashes': {name: sha(base / name) for name in FILES},
               'geometry': {'checked_calls': len(self.geometry), 'valid_area_min': min(self.geometry),
                            'valid_area_max': max(self.geometry), 'source_crop_and_padding_valid': True},
               'cuda_initialized': self.torch.cuda.is_initialized(), 'worker_pid': os.getpid()}
        need(not rec['cuda_initialized'], 'CUDA initialized')
        write(base / 'receipt.json', rec)  # commit last; uncommitted files safely overwritten
        return rec


_renderer = None


def init_worker(reg, ids, run_slot):
    global _renderer
    # Stable pool slot within fixed scratch; no collision with serial reference.
    slot = (mp.current_process()._identity[-1] - 1) % 6 + 1
    _renderer = Renderer(reg, ids, run_slot + '-' + str(slot))


def render_task(uid, out, workers):
    return _renderer.study(uid, out, workers)


def assemble(reg, out, ids):
    import numpy as np
    started = last = time.perf_counter(); recs = []
    for i, uid in enumerate(ids):
        recs.append(validate_shard(reg, out, uid))
        if time.perf_counter() - last >= 15 or i + 1 == len(ids):
            progress(out, 'verify-shards', i + 1, len(ids), started, i + 1)
            last = time.perf_counter()
    need(all(recs), 'assembly requires all committed shards')
    n = sum(r['rows'] for r in recs)
    disk_gate(out, 0, n * (384**2 + 48) + 1024 * 1024)
    px = np.lib.format.open_memmap(out / 'px.u8.npy.tmp', mode='w+', dtype=np.uint8, shape=(n, 384, 384))
    meta = np.lib.format.open_memmap(out / 'meta.f32.npy.tmp', mode='w+', dtype=np.float32, shape=(n, 12))
    index = {'order': ids, 'done': {}, 'blocks': {}, 'cursor': n, 'errors': {}, 'meta_cols': META_COLS,
             'config': {'size': 384, 'views': 'v4', 'anchors': str(ROOT / s.ANATOMY)}, 'contract_id': reg['contract_id']}
    cur = 0; started = last = time.perf_counter()
    for i, rec in enumerate(recs):
        uid = rec['uid']; base = out / 'shards' / key(uid); end = cur + rec['rows']
        px[cur:end] = np.load(base / 'px.npy', mmap_mode='r'); meta[cur:end] = np.load(base / 'meta.npy', mmap_mode='r')
        blocks = json_copy(rec['blocks'])
        for b in blocks: b['views'] = [[v, cur + off] for v, off in b['views']]
        index['blocks'][uid] = blocks; index['done'][uid] = [cur, rec['rows'], len(blocks)]; cur = end
        if time.perf_counter() - last >= 15 or i + 1 == len(recs):
            progress(out, 'assemble-copy', i + 1, len(recs), started, i + 1)
            last = time.perf_counter()
    px.flush(); meta.flush(); del px, meta
    for name in LAYOUT[:2]: s.replace(out / (name + '.tmp'), out / name)
    write(out / 'index.json', index)
    return recs


def check_cache(reg, out, ids):
    receipt = read(out / 'receipt.json')
    need(receipt['contract_id'] == reg['contract_id'] and receipt['ids'] == ids, 'cache identity')
    need(set(receipt['hashes']) == set(LAYOUT), 'cache payload list')
    for name in LAYOUT: need(sha(out / name) == receipt['hashes'][name], 'cache checksum')
    for uid in ids: need(validate_shard(reg, out, uid) is not None, 'missing shard')
    index = read(out / 'index.json')
    need(index['order'] == ids and set(index['done']) == set(ids) and not index['errors'], 'index roster')
    return receipt


def progress(out, phase, done, total, started, fresh):
    elapsed = time.perf_counter() - started
    rate = fresh / elapsed if fresh else 0
    status = {'version': VERSION, 'phase': phase, 'done': done, 'total': total,
              'elapsed_seconds': elapsed, 'new_studies_per_second': rate,
              'eta_seconds': (total - done) / rate if rate else None, 'time': time.time()}
    print(json.dumps(status), flush=True)
    try:
        write(out / 'status.json', status)
        with (out / 'progress.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(status) + '\n')
    except OSError as exc:
        print('Nonfatal progress publication: ' + str(exc), flush=True)


def build(reg, out, ids, workers=6, stop_after=None):
    need(1 <= workers <= 6, 'workers must be 1..6 (12 torch threads maximum)')
    validate(reg); out = safe_output(out); out.mkdir(parents=True, exist_ok=True)
    need(ids and len(ids) == len(set(ids)) and set(ids) <= set(reg['train'] + reg['holdout']), 'build roster')
    with file_lock(WORK / 'fullcache.lock', blocking=False):
        contract = {'contract_id': reg['contract_id'], 'ids': ids, 'size': 384, 'views': 'v4'}
        if (out / 'contract.json').exists(): need(read(out / 'contract.json') == contract, 'existing cache contract')
        else:
            need(not any(out.iterdir()), 'nonempty unregistered output')
            write(out / 'contract.json', contract)
        if (out / 'receipt.json').exists(): return check_cache(reg, out, ids)
        started = time.perf_counter()
        pending = [u for u in ids if validate_shard(reg, out, u) is None]
        completed = len(ids) - len(pending); fresh = 0
        disk_gate(out, min(workers, len(pending)))
        progress(out, 'render', completed, len(ids), started, fresh)
        # stop_after is a proof-only controlled interruption at a commit boundary.
        todo = pending if stop_after is None else pending[:stop_after]
        if todo:
            import psutil
            need(psutil.virtual_memory().available >= (workers * 2 + 4) * GIB, 'CPU RAM admission margin')
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                     initializer=init_worker, initargs=(reg, ids, 'pool')) as pool:
                it = iter(todo); futures = {}
                for _ in range(min(workers, len(todo))):
                    uid = next(it); futures[pool.submit(render_task, uid, out, workers)] = uid
                while futures:
                    done, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                    for f in done:
                        uid = futures.pop(f); rec = f.result(); need(rec['uid'] == uid, 'worker UID mismatch')
                        fresh += 1; completed += 1
                        u = next(it, None)
                        if u is not None: futures[pool.submit(render_task, u, out, workers)] = u
                    progress(out, 'render', completed, len(ids), started, fresh)
        if completed != len(ids):
            progress(out, 'interrupted-proof', completed, len(ids), started, fresh)
            return {'status': 'INTERRUPTED', 'committed': completed}
        render_seconds = time.perf_counter() - started
        progress(out, 'assemble', completed, len(ids), started, fresh)
        recs = assemble(reg, out, ids)
        result = {'version': VERSION, 'contract_id': reg['contract_id'], 'ids': ids, 'studies': len(ids),
                  'workers': workers, 'torch_threads_per_worker': 2, 'new_studies': fresh,
                  'render_wall_seconds': render_seconds, 'elapsed_seconds': time.perf_counter() - started,
                  'bytes': sum((out / name).stat().st_size for name in LAYOUT),
                  'hashes': {name: sha(out / name) for name in LAYOUT},
                  'cuda_initialized': any(r['cuda_initialized'] for r in recs),
                  'exposure': 'EXPOSED_EXPLORATORY', 'clean_oof': False}
        need(not result['cuda_initialized'], 'CUDA used')
        result['elapsed_seconds'] = time.perf_counter() - started
        write(out / 'receipt.json', result)
        progress(out, 'complete', completed, len(ids), started, fresh)
        return result


def authorize_full(reg, k0, review):
    need(k0 and review, 'full cache requires K0 result and independent review')
    k0, review = Path(k0).resolve(), Path(review).resolve()
    need(k0.is_relative_to(ROOT) and review.is_relative_to(ROOT), 'evidence must be in project')
    result, verdict = read(k0), read(review)
    from tools.l1_rco_runtime_v1 import go_rule
    need(result.get('contract_id') == read(s.WORK / 'registration.json')['contract_id'], 'K0 source registration mismatch')
    need(result.get('K0_numeric_gate') == 'PASS' and go_rule(result), 'K0 did not pass')
    need(verdict.get('verdict') == 'PASS' and verdict.get('reviewer') and verdict.get('builder') == 'l1-rco' and
         verdict['reviewer'] != verdict['builder'], 'independent full-cache admission required')
    need(verdict.get('K0_result_sha256') == sha(k0) and verdict.get('contract_id') == reg['contract_id'], 'review evidence mismatch')
    proof_path = WORK / 'proof.json'
    need(verdict.get('proof_sha256') == sha(proof_path), 'review proof hash mismatch')
    evidence = read(proof_path)
    need(evidence.get('status') == 'PASS' and evidence.get('contract_id') == reg['contract_id'], 'matching equivalence proof required')
    report = (ROOT / verdict['report']).resolve()
    need(report.is_relative_to(ROOT) and verdict.get('report_sha256') == sha(report), 'review report mismatch')


def serial_reference(ids, out):
    """Run untouched v1 rendering in a fresh process on a six-study registration.

    Only WORK (output containment) is rebound; no validation or math is replaced.
    """
    with file_lock(WORK / 'fullcache.lock', blocking=False):
        return _serial_reference_locked(ids, out)


def _serial_reference_locked(ids, out):
    from tools import l1_rco_cache_v1 as old
    reg = read(s.WORK / 'registration.json')
    reg['train'] = ids[:4]; reg['holdout'] = ids[4:]
    reg['contract_id'] = digest({k: v for k, v in reg.items() if k != 'contract_id'})
    old.WORK = WORK / 'serial-reference'
    old.WORK.mkdir(parents=True, exist_ok=True)
    old.build(reg, out, threads=2)


def proof(reg):
    import subprocess
    ids = reg['train'][:4] + reg['holdout'][:2]
    parallel = WORK / 'parallel6'; resumed = WORK / 'resume6'
    serial = WORK / 'serial-reference/cache'
    started = time.perf_counter()
    subprocess.run([s.PYTHON, str(Path(__file__).resolve()), '_serial'], cwd=ROOT, check=True)
    serial_seconds = time.perf_counter() - started
    result = build(reg, parallel, ids, 6)
    interrupted = build(reg, resumed, ids, 6, stop_after=3)
    before = {u: sha(resumed / 'shards' / key(u) / 'receipt.json') for u in ids if (resumed / 'shards' / key(u) / 'receipt.json').exists()}
    resumed_result = build(reg, resumed, ids, 6)
    need(all(sha(resumed / 'shards' / key(u) / 'receipt.json') == h for u, h in before.items()), 'resume rewrote committed shards')
    for uid in ids:
        for name in FILES:
            need(sha(serial / 'shards' / key(uid) / name) == sha(parallel / 'shards' / key(uid) / name) ==
                 sha(resumed / 'shards' / key(uid) / name), 'serial/parallel/resume shard mismatch')
    for name in LAYOUT[:2]: need(sha(serial / name) == sha(parallel / name) == sha(resumed / name), 'assembled payload mismatch')
    indexes = [read(p / 'index.json') for p in (serial, parallel, resumed)]
    for index in indexes: index.pop('contract_id')
    need(indexes[0] == indexes[1] == indexes[2], 'index semantics differ')
    receipt = {'version': VERSION, 'status': 'PASS', 'contract_id': reg['contract_id'], 'studies': 6,
               'shard_payloads': 'byte-identical to untouched v1 serial', 'mmap_payloads': 'byte-identical',
               'index': 'identical except intentional registration contract_id',
               'interruption': interrupted, 'committed_shards_preserved': len(before),
               'serial_invocation_seconds': serial_seconds, 'parallel': result,
               'resume': resumed_result, 'cuda_initialized': False,
               'projection': 'User H-B-182 supplied Stage0 baseline 13.48s/study: 4167*13.48/6/3600 = 2.60h ideal render; IO/assembly/contention extra; full run UNMEASURED'}
    write(WORK / 'proof.json', receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['register', 'validate', 'proof', 'build', 'admit', '_serial'])
    ap.add_argument('--out', type=Path, default=OUT); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--k0-result', type=Path); ap.add_argument('--review', type=Path)
    a = ap.parse_args(); WORK.mkdir(parents=True, exist_ok=True)
    if a.command == 'register':
        r = register(); print({'version': VERSION, 'train': len(r['train']), 'holdout': len(r['holdout']), 'contract_id': r['contract_id']}); return
    reg = validate(read(REG))
    if a.command == 'validate': print('PASS: full roster, input/code pins; labels not opened'); return
    if a.command == '_serial': serial_reference(reg['train'][:4] + reg['holdout'][:2], WORK / 'serial-reference/cache'); return
    if a.command == 'proof':
        print({'proof': proof(reg)['status']}); return
    need(a.out.resolve() == OUT.resolve(), 'full build output is fixed')
    need(1 <= a.workers <= 6, 'workers must be 1..6')
    authorize_full(reg, a.k0_result, a.review)
    need(read(WORK / 'proof.json')['contract_id'] == reg['contract_id'] and read(WORK / 'proof.json')['status'] == 'PASS', 'matching equivalence proof required')
    if a.command == 'admit': print('PASS: exact K0/review/proof admission; no build'); return
    build(reg, a.out, reg['train'] + reg['holdout'], a.workers)


if __name__ == '__main__':
    main()
