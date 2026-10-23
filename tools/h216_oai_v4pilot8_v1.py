"""H216 v1.0.0: actual-anchor v4 pilot on the eight completed native studies.

CPU preparation never loads weights or initializes CUDA. `run` requires the
existing exclusive gpu_job owner and a bound B-direct check under solo authority.
Original decoder, geometry, anchors, renderer and training loader stay unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.l1_rco_stage0_v1 import need, sha, read, write, digest

VERSION = '1.0.0'
TAG = 'H216_OAI_V4_PILOT8_V1'
OUT = ROOT / '.local/scratch/slot-b/h216-oai-v4pilot8-v1'
GEOMETRY = ROOT / '.local/scratch/slot-b/h216-oai-geometry8-v1'
CODE = ROOT / '.local/exp189-cpu-v120/input/cloud_code'
SOURCE = ROOT / 'tools/exp189_s6_src'
WEIGHT = ROOT / 'storage/weights/external/heliosli_rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt'
WEIGHT_SHA = '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
VIEWS = {'Sagittal': [0, 1, 4, 5], 'Coronal': [0, 2, 3], 'Axial': [0, 1, 4, 5]}
PINS = {'px_cache.py': 'b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632',
        'knee_anchor_bank_v2.py': 'e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309',
        'compact_train.py': '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80'}


def key(uid):
    return hashlib.sha256(uid.encode()).hexdigest()[:24]


def check(bindings):
    for path, expected in bindings.items():
        need(sha(path) == expected, 'BOUND_FILE_CHANGED')


def modules():
    # Do not import native/geometry adapters: their CPU-only import disables CUDA.
    os.environ['RSNA_CODE_ROOT'] = str(CODE)
    os.environ['HF_HUB_OFFLINE'] = '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path[:0] = [str(CODE), str(SOURCE)]
    import torch, px_cache, compact_train, knee_anchor_bank_v2
    torch.set_num_threads(1)
    return torch, px_cache, compact_train, knee_anchor_bank_v2


def anchor_check(record, study):
    """Expose errors/fallback which the frozen anchor/renderer otherwise tolerates."""
    import numpy as np
    need(record.get('study') == study['StudyInstanceUID'], 'ANCHOR_IDENTITY')
    method = record.get('method')
    need(method == 'slab_prior' or str(method).startswith('skm_tea_joint'), 'ANCHOR_METHOD')
    why = record.get('why', [])
    need(not any(str(w).startswith('error:') or w in ('no_geo', 'not_aligned', 'no_geometry') for w in why), 'ANCHOR_ERROR')
    for name in ('joint_lps', 'slab_prior_lps'):
        a = np.asarray(record.get(name), float)
        need(a.shape == (3,) and np.isfinite(a).all(), 'ANCHOR_COORDINATES')
    if record.get('patella_lps') is not None:
        a = np.asarray(record['patella_lps'], float)
        need(a.shape == (3,) and np.isfinite(a).all(), 'PATELLA_COORDINATES')
    ids = [s['SeriesInstanceUID'] for s in study['series']]
    need(len(ids) == len(set(ids)) and set(record['series']) == set(ids), 'ANCHOR_SERIES_ROSTER')
    outside = []
    for sid in ids:
        e = record['series'][sid]
        joint = np.asarray(e.get('joint'), float)
        need(joint.shape == (2,) and np.isfinite(joint).all(), 'ANCHOR_PROJECTION')
        need(type(e.get('in_field')) is bool and np.isfinite(e.get('dist_mm', float('nan'))) and e['dist_mm'] >= 0, 'ANCHOR_PROJECTION')
        if e.get('patella') is not None:
            p = np.asarray(e['patella'], float)
            need(p.shape == (2,) and np.isfinite(p).all(), 'PATELLA_PROJECTION')
        if not e['in_field']:
            outside.append(sid)
    return dict(method=method, why=why, slab_prior=(method == 'slab_prior'),
                no_sagittal=not any(s['plane'] == 'Sagittal' for s in study['series']),
                field_centre_substitutions=outside, readiness_hold=bool(outside))


def item_check(item, study, size=384):
    import numpy as np
    px, meta = item['px'].numpy(), item['meta'].numpy()
    need(item['uid'] == study['StudyInstanceUID'] and not item['error'] and not item['skipped_series'], 'RENDER_INCOMPLETE')
    need(px.dtype == np.uint8 and px.shape[1:] == (size, size) and len(px) > 0 and bool(px.any()), 'PIXEL_CONTRACT')
    need(meta.shape == (len(px), 12) and np.isfinite(meta).all(), 'METADATA_CONTRACT')
    series = {s['SeriesInstanceUID']: s for s in study['series']}
    blocks = item['blocks']; ids = [b['series'] for b in blocks]
    need(len(set(ids)) == len(ids) and set(ids) <= set(series) and
         len(blocks) + item['removed'] == len(series), 'SERIES_ACCOUNTING')
    cursor = 0
    for bidx, b in enumerate(blocks):
        s = series[b['series']]; n = b['n_slices']
        need(b['plane'] == s['plane'] and n > 0 and [v for v, _ in b['views']] == VIEWS[s['plane']], 'VIEW_ROSTER')
        real = study['series'].index(s)
        for view, start in b['views']:
            need(start == cursor and start + n <= len(px), 'ROW_PARTITION')
            m = meta[start:start+n]
            for column, expected in ((0, bidx), (1, s['slot']), (3, view), (10, n), (11, real),
                                     (7, s.get('fluid_sensitive', -1)), (8, s.get('fat_suppression', -1))):
                expected = -1 if expected is None else expected
                need(np.all(m[:, column] == expected), 'ROW_SEMANTICS')
            need(np.array_equal(m[:, 9], np.arange(n)), 'SLICE_ORDER')
            cursor += n
    need(cursor == len(px), 'ROW_PARTITION')
    return px, meta


def served_hashes(px, meta, blocks, ct):
    """The unchanged serving window recipe, hashed incrementally to bound RAM."""
    import numpy as np
    bs = [dict(b, views=[(0, v, off) for v, off in b['views']]) for b in blocks]
    rows = np.asarray(ct.window_rows(bs, lambda f, r: meta[r], 'eval', 3, 3, phys_mm=3.5), np.int64)
    need(len(rows) > 0, 'NO_EVAL_WINDOWS')
    h = hashlib.sha256()
    for row in rows:
        for r in row[2:]:
            h.update(np.ascontiguousarray(px[r]).tobytes())
    m = np.asarray(meta[rows[:, 1]][:, [1, 3, 4, 5, 6, 7]], np.float32)
    return dict(images=h.hexdigest(), meta=hashlib.sha256(m.tobytes()).hexdigest(), windows=len(rows))


def assemble(out, recs, pxmod, progress=lambda: None):
    import numpy as np
    n = sum(r['rows'] for r in recs)
    need(shutil.disk_usage(out).free > n * (384**2 + 48) + 100 * 2**30, 'DISK_RESERVE')
    px = np.lib.format.open_memmap(out/'px.u8.npy', mode='w+', dtype=np.uint8, shape=(n, 384, 384))
    meta = np.lib.format.open_memmap(out/'meta.f32.npy', mode='w+', dtype=np.float32, shape=(n, 12))
    index = dict(order=[r['uid'] for r in recs], done={}, blocks={}, cursor=n, errors={},
                 meta_cols=pxmod.META_COLS, config=dict(size=384, views='v4'))
    cur = 0
    for r in recs:
        base = out/'shards'/key(r['uid']); end = cur + r['rows']
        check({str(base/name): h for name, h in r['hashes'].items()})
        px[cur:end] = np.load(base/'px.npy', mmap_mode='r'); meta[cur:end] = np.load(base/'meta.npy', mmap_mode='r')
        blocks = copy.deepcopy(r['blocks'])
        for b in blocks:
            b['views'] = [[v, off + cur] for v, off in b['views']]
        index['blocks'][r['uid']] = blocks; index['done'][r['uid']] = [cur, r['rows'], len(blocks)]
        cur = end; progress()
    px.flush(); meta.flush(); del px, meta
    write(out/'index.json', index)


def loader_parity(out, recs, ct, progress=lambda: None):
    ds = ct.Studies(str(out), [r['uid'] for r in recs], 64, 'eval', 9000, eval_stride=3, slices=3, phys_mm=3.5)
    for i, rec in enumerate(recs):
        x, m, idx = ds[i]
        actual = dict(images=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                      meta=hashlib.sha256(m.numpy().tobytes()).hexdigest(), windows=len(x))
        need(idx == i and actual == rec['serving'], 'SERVING_CACHE_PARITY')
        del x, m; progress()


def register():
    need(not OUT.exists(), 'OUTPUT_EXISTS')
    gr = read(GEOMETRY/'REGISTRATION.json'); result = read(GEOMETRY/'RESULT.json')
    need(result['status'] == 'GEOMETRY8_PREPARED' and result['studies'] == 8 and result['series'] == 31, 'GEOMETRY_NOT_COMPLETE')
    paths = [Path(__file__), ROOT/'tests/test_h216_oai_v4pilot8_v1.py',
             ROOT/'docs/research/h216-oai-v4pilot8-v1.md', ROOT/'docs/research/h216-v4-successor-contract-v1.md',
             ROOT/'tools/l1_rco_cache_v1.py', ROOT/'tools/l1_rco_stage0_v1.py',
             ROOT/'tools/l1_rco_runtime_v120.py', ROOT/'tools/l1_rco_stage0_v120.py', ROOT/'tools/job_runtime.py',
             GEOMETRY/'REGISTRATION.json', GEOMETRY/'RESULT.json']
    for name, h in PINS.items():
        need(sha(SOURCE/name) == h, 'FROZEN_SOURCE_DRIFT'); paths.append(SOURCE/name)
    paths.extend((CODE/'training').rglob('*.py'))
    sources = {str(p): sha(p) for p in paths}
    inputs = dict(gr['sources'])
    inputs.update({str(GEOMETRY/p): h for p, h in result['outputs'].items()})
    check(inputs)
    cases = copy.deepcopy(gr['cases'])
    for c in cases:
        c['geometry'] = str(GEOMETRY/c['token']/'geometry')
    need(len(cases) == 8 and len({c['uid'] for c in cases}) == 8, 'PILOT_CARDINALITY')
    reg = dict(version=VERSION, sources=sources, inputs=inputs, cases=cases,
               weight=dict(path=str(WEIGHT), sha256=WEIGHT_SHA, checked_at='exclusive execution only'),
               recipe=dict(size=384, views='v4', labels=False, checkpoint_startup_during_cpu=False))
    reg['contract_id'] = digest(reg); OUT.mkdir(parents=True); write(OUT/'REGISTRATION.json', reg)
    return dict(status='REGISTERED', studies=8, training_ready=False)


def registration():
    r = read(OUT/'REGISTRATION.json')
    need(r['contract_id'] == digest({k:v for k,v in r.items() if k != 'contract_id'}), 'REGISTRATION_DIGEST')
    check(r['sources'])
    return r


def render_one(study, anchor, out, pxmod, ct, progress=lambda: None):
    import numpy as np
    from tools.l1_rco_cache_v1 import geometry_checked
    from training import input_provenance
    uid = study['StudyInstanceUID']; base = out/'shards'/key(uid); base.mkdir(parents=True)
    write(out/'native/studies'/(key(uid)+'.json'), study)
    write(out/'anchors'/(key(uid)+'.json'), anchor)
    geometries = []; old = input_provenance.view_geometry
    guard = geometry_checked(old, geometries)
    def checked(*args, **kwargs):
        v = guard(*args, **kwargs)
        if len(geometries) % 64 == 0: progress()
        return v
    input_provenance.view_geometry = checked
    started = time.perf_counter()
    try:
        item = pxmod.StudySlices(str(out/'native'), [uid], 384, 'v4', str(out/'anchors')).build(uid)
    finally:
        input_provenance.view_geometry = old
    px, meta = item_check(item, study)
    need(len(geometries) == len(px), 'UNCHECKED_RENDER_GEOMETRY')
    np.save(base/'px.npy', px); np.save(base/'meta.npy', meta.astype(np.float32))
    rec = dict(uid=uid, rows=len(px), blocks=item['blocks'], removed=item['removed'],
               rendering_seconds=time.perf_counter()-started,
               geometry=dict(calls=len(geometries), valid_area_min=min(geometries), valid_area_max=max(geometries)),
               serving=served_hashes(px, meta, item['blocks'], ct),
               hashes={name:sha(base/name) for name in ('px.npy','meta.npy')})
    write(base/'receipt.json', rec); progress()
    return rec


def cpu_smoke():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    reg = registration(); out = OUT/'cpu-smoke'
    need(not out.exists(), 'SMOKE_EXISTS'); out.mkdir()
    torch, pxmod, ct, _ = modules()
    def forbidden(*args, **kwargs): raise ValueError('CPU_MODEL_LOAD_FORBIDDEN')
    torch.load = forbidden
    study = read(reg['cases'][0]['study'])
    # One real native series only. Synthetic centre used solely to exercise new
    # adapter/renderer/storage code, never counted as actual anatomical ROI.
    study['series'] = study['series'][:1]
    uid = study['StudyInstanceUID']; sid = study['series'][0]['SeriesInstanceUID']
    anchor = dict(study=uid, method='SYNTHETIC_TEST_ONLY', series={sid:dict(joint=[.5,.5],patella=None,in_field=True)})
    rec = render_one(study, anchor, out, pxmod, ct)
    assemble(out, [rec], pxmod); loader_parity(out, [rec], ct)
    need(not torch.cuda.is_initialized(), 'CPU_CUDA_INITIALIZED')
    result = dict(status='PASS_MECHANICAL_ONLY', contract_id=reg['contract_id'], series=1, rows=rec['rows'],
                  windows=rec['serving']['windows'], geometry=rec['geometry'], cuda=False, model_loads=0,
                  actual_roi=False, training_ready=False, source_study_sha256=sha(reg['cases'][0]['study']),
                  hashes={n:sha(out/n) for n in ('px.u8.npy','meta.f32.npy','index.json')})
    write(OUT/'CPU-SMOKE.json', result)
    return {k:v for k,v in result.items() if k != 'hashes'}


def review_bindings():
    reg = registration(); smoke = read(OUT/'CPU-SMOKE.json')
    need(smoke['contract_id'] == reg['contract_id'] and smoke['status'] == 'PASS_MECHANICAL_ONLY' and not smoke['cuda'], 'SMOKE_NOT_BOUND')
    return dict(registration_sha256=sha(OUT/'REGISTRATION.json'), cpu_smoke_sha256=sha(OUT/'CPU-SMOKE.json'),
                tests_sha256=sha(OUT/'TEST-RESULT.txt'))


def admit():
    review = read(OUT/'B-PRECHECK.json')
    need(review['verdict'] == 'PASS_B_DIRECT' and review['reviewer'] == 'B/Codex' and
         review['authority'] == 'Taeyang2026-10-19-solo-continuous-GPU' and review['bindings'] == review_bindings(), 'BOUND_B_REVIEW_REQUIRED')
    need(sha(OUT/'B-PRECHECK.md') == review['report_sha256'], 'REVIEW_CHANGED')
    return dict(status='CPU_ADMISSIBLE', physical_gpu_claim=False, training_ready=False)


def command():
    return [PYTHON, str(Path(__file__).resolve()), 'run']


def require_job(worker=False):
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json, command_digest
    with file_lock(LOGS/'.gpu-claim.lock'):
        job = read_json(LOGS/TAG/'job.json'); parent = psutil.Process().parent()
        need(parent is not None, 'MISSING_PARENT')
        owner = parent.parent() if worker else parent
        need(owner is not None and job.get('schema') == 'gpu-job/v2' and job.get('tag') == TAG and job.get('phase') == 'running', 'GPU_JOB_REQUIRED')
        need(job.get('child_identity') == (identity(parent.pid) if worker else identity()) and
             {k:job.get(k) for k in ('pid','create_time')} == identity(owner.pid), 'GPU_JOB_IDENTITY')
        need(job.get('command') == command() and job.get('command_sha256') == command_digest(command(), ROOT) and
             Path(job['cwd']).resolve() == ROOT, 'GPU_JOB_COMMAND')
        need(owner.cmdline()[-3:] == ['--_run-job', TAG, job['run_id']] and
             any(Path(x).name == 'gpu_job.py' for x in owner.cmdline()), 'GPU_JOB_PARENT')
        if worker:
            argv = parent.cmdline()
            need(len(argv) == 3 and Path(argv[0]).resolve() == Path(PYTHON).resolve() and
                 Path(argv[1]).resolve() == Path(__file__).resolve() and argv[2] == 'run', 'WORKER_PARENT_COMMAND')
        return owner.pid


def worker():
    require_job(worker=True); reg = registration()
    check(reg['inputs']); need(sha(WEIGHT) == WEIGHT_SHA, 'WEIGHT_CHANGED')
    started = time.perf_counter(); torch, pxmod, ct, anchors = modules()
    need(torch.cuda.is_available(), 'CUDA_UNAVAILABLE')
    admission = read(OUT/'ADMISSION.json'); counts = dict(model_calls=0, progress_units=0)
    def progress(phase='render'):
        counts['progress_units'] += 1
        peak = torch.cuda.max_memory_reserved()
        from tools.l1_rco_runtime_v120 import memory_ok
        need(memory_ok(peak, admission['desktop_bytes'], admission['total_bytes']), 'MEMORY_GATE')
        write(OUT/'status.json', dict(phase=phase,time=time.time(),steps=counts['progress_units'],
              windows=counts['model_calls'],peak_reserved=peak))
    torch.cuda.reset_peak_memory_stats(); model = anchors.load_model(WEIGHT, 'cuda')
    torch.cuda.synchronize(); startup = time.perf_counter()-started; progress('model-loaded')
    def after_forward(module, args, output):
        torch.cuda.synchronize(); counts['model_calls'] += 1; progress('anchor')
    hook = model.register_forward_hook(after_forward); bank = anchors.AnchorBank(model, 'cuda')
    recs = []; audits = []
    for case in reg['cases']:
        study = read(case['study']); t = time.perf_counter(); before = counts['model_calls']
        anchor = bank.record(case['uid'], study, geo_dir=case['geometry'])
        write(OUT/'anchors'/(key(case['uid'])+'.json'), anchor)
        audit = anchor_check(anchor, study)
        audit.update(token=case['token'], seconds=time.perf_counter()-t, model_calls=counts['model_calls']-before)
        if audit['no_sagittal']:
            need(audit['slab_prior'] and audit['model_calls'] == 0, 'NO_SAGITTAL_CONTRACT')
        audits.append(audit); progress('anchor-complete')
        rec = render_one(study, anchor, OUT, pxmod, ct, progress); recs.append(rec)
        print(json.dumps(dict(completed=len(recs),total=8,anchor_method=audit['method'],rows=rec['rows'])), flush=True)
    hook.remove(); del bank, model; torch.cuda.empty_cache()
    assemble(OUT, recs, pxmod, progress); loader_parity(OUT, recs, ct, progress)
    hold = any(a['readiness_hold'] for a in audits)
    result = dict(version=VERSION,contract_id=reg['contract_id'],status='HOLD' if hold else 'V4_PILOT_PREPARED',
                  studies=len(recs),series=sum(len(r['blocks']) for r in recs),removed_series=sum(r['removed'] for r in recs),
                  rows=sum(r['rows'] for r in recs),windows=sum(r['serving']['windows'] for r in recs),
                  startup_seconds=startup,all_up_seconds=time.perf_counter()-started,
                  anchor_seconds=sum(a['seconds'] for a in audits),render_seconds=sum(r['rendering_seconds'] for r in recs),
                  anchors=audits,model_calls=counts['model_calls'],peak_reserved=torch.cuda.max_memory_reserved(),
                  parity='PASS',actual_roi=True,full_cohort_ready=False,training_ready=False,
                  hashes={n:sha(OUT/n) for n in ('px.u8.npy','meta.f32.npy','index.json')})
    write(OUT/'RESULT.json', result); progress('complete' if not hold else 'hold')
    need(not hold, 'OUT_OF_FIELD_READINESS_HOLD')


def run():
    owner = require_job(); admit(); reg = registration()
    need(not (OUT/'ADMISSION.json').exists(), 'ATTEMPT_ALREADY_ADMITTED')
    from tools.l1_rco_runtime_v120 import active_ml_processes, gpu_snapshot, stale_reason
    need(not active_ml_processes(owner), 'ANOTHER_GPU_PROCESS')
    snap = gpu_snapshot()
    need(snap['desktop_bytes'] + 2*2**30 < snap['total_bytes'], 'NO_GPU_HEADROOM')
    need(shutil.disk_usage(OUT).free > 4*2**30 + 100*2**30, 'PILOT_DISK_RESERVE')
    write(OUT/'ADMISSION.json', dict(snap,contract_id=reg['contract_id'],time=time.time()))
    status = dict(phase='launch',time=time.time(),steps=0,windows=0,peak_reserved=0); write(OUT/'status.json',status)
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='0'; env['PYTHONPATH']=str(ROOT)
    child = None
    try:
        with (OUT/'worker.log').open('x',encoding='utf-8') as log:
            child = subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                time.sleep(15)
                try: status = read(OUT/'status.json')
                except (OSError,ValueError): pass
                reason = stale_reason(status,time.time())
                need(not reason,'ZERO_PROGRESS_WATCHDOG')
            need(child.returncode == 0, 'WORKER_FAILED')
        need(read(OUT/'RESULT.json')['status'] == 'V4_PILOT_PREPARED', 'INCOMPLETE_RESULT')
    finally:
        if child is not None and child.poll() is None:
            if os.name == 'nt': subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=False)
            else: child.terminate()
            child.wait(timeout=30)
    return dict(status='V4_PILOT_PREPARED',studies=8,training_ready=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['register','cpu-smoke','admit','run','worker']); a=ap.parse_args()
    if a.action not in ('run','worker'): os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        result = {'register':register,'cpu-smoke':cpu_smoke,'admit':admit,'run':run,'worker':worker}[a.action]()
        if result is not None: print(json.dumps(result,allow_nan=False))
    except Exception as exc:
        code = str(exc) if type(exc) is ValueError and str(exc).replace('_','').isalnum() else 'PRIVATE_DETAILS_SUPPRESSED'
        result = dict(status='FAILED',error_type=type(exc).__name__,code=code,partial_preserved=True)
        if a.action in ('run','worker') and (OUT/'ADMISSION.json').exists():
            dest = OUT/(a.action+'-FAILURE.json')
            if not dest.exists(): write(dest,result)
        print(json.dumps(result)); raise SystemExit(1)


if __name__ == '__main__': main()
