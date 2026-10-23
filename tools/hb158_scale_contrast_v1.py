"""H-B-158 v1.0.0: serving-only scale contrast; B owns CUDA execution.

Changelog 1.0.0: authenticated CONTROL fold-0, frozen release functions,
320/384 probability mean, fixed four-ruler screen, reviewed GPU controller.
This is a local diagnostic, NOT a release stage or a submission qualification.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hb145_output_io as io
from training.arms import coatnet_history as history

VERSION = '1.0.0'
BUILDER = 'hb158-builder'
WORK = ROOT / '.local/scratch/hb158/H-B-158'
CORPUS = ROOT / 'storage/cache/derived/EXP143_RAPTOR96_384_C140_S02'
CHECKPOINT = ROOT / 'artifacts/experiments/EXP181_COATNET_F0_BRIDGE/CONTROL/model_best.pt'
MODEL_SHA = 'f41d5c2707b4b73bc16ded2635a656c91876eb58fa0a8a12a408236160dbd1c2'
TARGETS = history.TARGETS
RULERS = ('jev', 'aquila', 'dread', 'consensus')
VIEWS = (320, 384)
GIB = 2**30
NODES = ('build_backbone', 'RaptorClassifier', '_eval_centers', '_slot_index',
         '_mirror_tri', 'eval_windows', 'gpu_windows', 'infer_probs', 'infer_pair_probs')
POLICY = dict(primary_min=.001, positive_ci_rulers=2, per_target_floor=-.003,
              bootstrap_draws=300, bootstrap_seed=20261003, combine_flip_all=False,
              views=[320, 384], k=94, chunk=16, anatomical_mirror=True,
              mean='equal probability mean BEFORE any ranks', cuda_precision='fp16')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return history.sha256(path)


def jhash(value):
    return history.json_hash(value)


def source_bindings():
    paths = [Path(__file__), ROOT/'tests/test_hb158_scale_contrast_v1.py']
    paths += [ROOT/p for p in (
        'training/arms/coatnet_history.py', 'training/src/exp143_own0949.py',
        'tools/c85r7_flipall_v1115.py', 'tools/c85r7_flipall_timing.py',
        'tools/hb149_monitor_v125.py', 'tools/hb145_output_io.py',
        'tools/gpu_job.py', 'tools/job_runtime.py', 'tools/exp179_screen.py',
        'tools/exp149_label_sources.py', 'tools/exp156_k_sweep.py',
        'tools/exp170_convnext_t_l2.py', 'tools/exp115_vlm_eval.py',
        'tools/exp162_dino_folds_kaggle.py', 'training/src/raptor_common.py')]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def release_source():
    from tools.c85r7_flipall_v1115 import parent
    _, files = parent()  # authenticates ALL embedded bytes; never runs notebook cells
    return files['public-serving.py'].decode(), {n: hashlib.sha256(b).hexdigest()
                                                for n, b in files.items()}


def serving(res):
    """Only allowlisted definitions from the authenticated release, no cell/main.

    The sole definition edit generalizes the existing shape assertion 320 to
    INPUT_RES. All math/model/window definitions, including CUDA infer_probs,
    remain byte-identical. CPU adapter is explicitly a float32 smoke, not CUDA
    numerical parity evidence. Native corpus is the authenticated release cache.
    """
    import numpy as np
    import torch
    import timm
    need(res in VIEWS, 'unregistered scale')
    source, pins = release_source()
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in NODES]
    need({n.name for n in selected} == set(NODES), 'release definition inventory changed')
    proofs = {}
    code = []
    for node in selected:
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
        raw = '\n'.join(source.splitlines()[start-1:node.end_lineno]) + '\n'
        original = raw
        if node.name == 'infer_pair_probs':
            need(raw.count('(94,3,320,320)') == 1, 'release shape assertion changed')
            raw = raw.replace('(94,3,320,320)', '(94,3,INPUT_RES,INPUT_RES)')
        proofs[node.name] = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                executed_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                                change='shape assertion only' if raw != original else 'byte-identical')
        code.append(raw)
    ns = dict(np=np, torch=torch, nn=torch.nn, F=torch.nn.functional, timm=timm,
              INPUT_RES=res, IMG=384, NORM='imagenet', SLOT_SIZES=[26,22,18,12,18],
              SAGITTAL_SLOTS=(0,1), _WLUT={},
              _MEAN=torch.tensor([.485,.456,.406]).view(3,1,1),
              _STD=torch.tensor([.229,.224,.225]).view(3,1,1))
    exec(compile('\n'.join(code), '<authenticated-release-definitions>', 'exec'), ns)
    return ns, dict(embedded_sha256=pins, definitions=proofs)


def authenticate():
    record = history.verified_arm(history.CONTROL)
    p = record['p']
    need(sha(CHECKPOINT) == MODEL_SHA and record['run']/ 'model_best.pt' == CHECKPOINT,
         'not the authenticated EXP181 CONTROL checkpoint')
    fold = p['fold']
    fold_path = ROOT/fold['ordered_ids']
    need(sha(fold_path) == fold['ordered_ids_sha256'], 'fold roster changed')
    roster = io.read_json(fold_path)
    need(roster['val'] == p['val_ids'] and set(roster['train']) == set(p['train_ids']),
         'train/heldout fold exclusion mismatch')
    need(sha(CORPUS/'manifest.json') == p['corpus_manifest_sha256'], 'native corpus changed')
    from tools.exp179_screen import load_context, TARGETS as ruler_targets
    need(list(ruler_targets) == list(TARGETS), 'ruler/model target order mismatch')
    ids, refs, draws, metadata = load_context(ROOT)
    need(len(ids) == 888 and set(ids) == set(p['val_ids']), 'ruler cohort differs from heldout fold')
    need(len(draws) == POLICY['bootstrap_draws'], 'bootstrap count changed')
    need(metadata['bootstrap_seed'] == POLICY['bootstrap_seed'], 'bootstrap seed changed')
    need(not set(ids) & set(p['train_ids']), 'fold contamination')
    return ids, refs, draws, dict(checkpoint_sha256=MODEL_SHA,
        protocol_sha256=record['protocol_sha256'], train_n=len(p['train_ids']), heldout_n=len(ids),
        train_val_overlap=0, fold=0, oai=False, ruler_metadata=metadata,
        corpus_manifest_sha256=p['corpus_manifest_sha256'],
        interpretation='held-out checkpoint; reused/exposed ruler screen, NOT untouched final validation',
        clean_oof=False, exposure='EXPOSED_SCREEN', checkpoint_fold_excluded=True)


def registration():
    ids, _, _, identity = authenticate()
    _, proof = serving(320)
    return dict(version=VERSION, builder=BUILDER, policy=POLICY, identity=identity,
                ids=ids, sources=source_bindings(), serving_proof=proof)


def registered():
    saved = io.read_json(WORK/'registration.json')
    need(saved == registration(), 'registration stale; re-register and repeat smoke/review')
    return saved


def volume(uid):
    import numpy as np
    from training.src.exp143_own0949 import Corpus
    corpus = Corpus(CORPUS)
    row = corpus.rows[uid]
    entry = corpus.manifest['files'][row]
    path = CORPUS/f'study_{row:05d}.npy'
    receipt = CORPUS/f'study_{row:05d}.json'
    need(entry['uid'] == uid and entry['row'] == row and sha(path) == entry['sha256']
         and sha(receipt) == entry['receipt_sha256'], 'native study bytes/receipt changed')
    vol, mask = corpus.get(uid)
    need(vol.shape == (96,384,384) and vol.dtype == np.uint8 and mask.shape == (96,)
         and mask.any(), 'native study contract invalid')
    return vol, mask, dict(volume_sha256=entry['sha256'], receipt_sha256=entry['receipt_sha256'])


def make_model(ns, device):
    import torch
    bb = ns['build_backbone']('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False)
    model = ns['RaptorClassifier'](bb, F_dim=bb.num_features)
    need(sha(CHECKPOINT) == MODEL_SHA, 'checkpoint changed before model load')
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    return model.eval().to(device)


def valid_probs(values, n):
    import numpy as np
    values = np.asarray(values)
    need(values.shape == (n,12) and np.isfinite(values).all()
         and ((values >= 0) & (values <= 1)).all(), 'invalid probabilities; no fallback')
    return values


def infer(ns, model, vol, mask, device):
    """Exact GPU window/pair functions; CPU-only float32 adapter for dry-run."""
    import torch
    res = ns['INPUT_RES']
    x = ns['gpu_windows'](vol, mask, 94, res, device)
    mirrored = ns['gpu_windows'](vol, mask, 94, res, device, mirror=True)
    if device == 'cpu':
        # Check both native window implementations on actual images, both views.
        for windows, mirror in ((x,False), (mirrored,True)):
            ref = ns['eval_windows'](vol, mask, 94, res, mirror=mirror)
            need(torch.equal(windows, ref), 'CPU release LUT/window parity failed')
        with torch.inference_mode():
            a, b = (model(z.unsqueeze(0)).float().sigmoid()[0].numpy() for z in (x,mirrored))
        result = .5*(a+b)
    else:
        result = ns['infer_pair_probs'](model, x, mirrored, device)
    return valid_probs([result], 1)[0]


def scientific_gate(pair):
    refs = pair['references']
    need(set(refs) == set(RULERS), 'exact four rulers required')
    finite = lambda x: isinstance(x, (int,float)) and math.isfinite(x)
    complete = all(set(r['per_target_delta']) == set(TARGETS) and
                   all(finite(x) for x in r['per_target_delta'].values()) and
                   r['ci95'] is not None and len(r['ci95']) == 2 and
                   all(finite(x) for x in r['ci95']) for r in refs.values())
    primary = finite(pair['primary_mean_delta']) and pair['primary_mean_delta'] >= .001
    positive = sum(r['ci95'] is not None and finite(r['ci95'][0]) and r['ci95'][0] > 0
                   for r in refs.values())
    floor = complete and all(x >= -.003 for r in refs.values() for x in r['per_target_delta'].values())
    return dict(status='GO_LOCAL_ONLY' if complete and primary and positive >= 2 and floor else 'HOLD',
                complete=complete, primary_gain=primary, positive_ci_rulers=positive,
                all_target_floor=floor, policy=POLICY, independent_result_review_required=True,
                submission_authorized=False)


def comparisons(arms, refs, draws):
    import numpy as np
    from tools import exp115_vlm_eval as evaluator
    from tools.exp170_convnext_t_l2 import paired_jev_consensus
    a, b = arms['320'], arms['384']
    valid_probs(a, len(a)); valid_probs(b, len(a))
    mean = .5*(np.asarray(a)+np.asarray(b))
    # The identical refs/masks/draws object is used for every arm; no arm-specific dropping.
    full = paired_jev_consensus(b, a, refs, draws, evaluator)
    pair = paired_jev_consensus(mean, a, refs, draws, evaluator)
    return dict(full384_minus_crop320=full, mean_minus_crop320=pair, gate=scientific_gate(pair))


def check_review(review, reg_sha, smoke_sha):
    need(review.get('verdict') == 'PASS' and review.get('reviewer') not in (None,'',BUILDER)
         and review.get('independent') is True and review.get('scope') == 'HB158_SCALE'
         and review.get('registration_sha256') == reg_sha
         and review.get('cpu_smoke_sha256') == smoke_sha,
         'independent source/smoke-bound PASS required')
    evidence = Path(review.get('evidence_path','')).resolve()
    need(ROOT in evidence.parents and evidence.is_file() and
         sha(evidence) == review.get('evidence_sha256'), 'independent review evidence missing/changed')


def admit(dry_run=False):
    registered()
    smoke = io.read_json(WORK/'cpu-smoke.json')
    need(smoke.get('status') == 'CPU_SMOKE_PASS' and smoke.get('studies') == 2
         and smoke.get('registration_sha256') == sha(WORK/'registration.json'), 'CPU smoke stale/missing')
    review = WORK/'review.json'
    if dry_run and not review.exists():
        return dict(status='HOLD_REVIEW_REQUIRED', cuda_started=False)
    check_review(io.read_json(review), sha(WORK/'registration.json'), sha(WORK/'cpu-smoke.json'))
    return dict(status='ADMITTED_FOR_B_GPU_CALIBRATION', cuda_started=False,
                memory_admission='pending measured full-shape two-study calibration')


def execute(device, ids, out):
    import numpy as np
    import torch
    torch.set_num_threads(8)
    if device != 'cpu':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    out.mkdir(parents=True, exist_ok=True)
    import timm
    io.receipt(out.parent/('cpu-runtime.json' if device == 'cpu' else 'runtime.json'), dict(
        python=sys.version, torch=torch.__version__, timm=timm.__version__, numpy=np.__version__,
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=device,
        gpu=None if device == 'cpu' else torch.cuda.get_device_name(0),
        capability=None if device == 'cpu' else list(torch.cuda.get_device_capability(0)),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, chunk=16, threads=8))
    reg_sha = sha(WORK/'registration.json')
    arm_values, arm_times = {}, {}
    # One view/model resident at a time; preserve all release windows and chunk16.
    for res in VIEWS:
        ns, _ = serving(res)
        model = make_model(ns, device)
        values, times = [], []
        for i, uid in enumerate(ids):
            vol, mask, hashes = volume(uid)
            if device != 'cpu':
                torch.cuda.synchronize()
            started = time.perf_counter()
            pred = infer(ns, model, vol, mask, device)
            if device != 'cpu':
                torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            row = dict(identity=reg_sha, uid=uid, view=res, probabilities=pred.tolist(),
                       seconds=seconds, input=hashes, precision='fp32' if device == 'cpu' else 'fp16',
                       peak_reserved_bytes=0 if device == 'cpu' else torch.cuda.max_memory_reserved(),
                       peak_allocated_bytes=0 if device == 'cpu' else torch.cuda.max_memory_allocated())
            if device != 'cpu':
                from tools.c85r7_flipall_timing import memory_admission
                desktop = io.read_json(out.parent/'desktop.json')
                memory_admission(desktop, [row['peak_reserved_bytes']])
            io.journal(out/f'{res}-{i:04d}.json', row, out)
            values.append(pred); times.append(seconds)
            print(json.dumps(dict(view=res, completed=i+1, total=len(ids))), flush=True)
        arm_values[str(res)] = np.asarray(values)
        arm_times[str(res)] = times
        del model, ns
        gc.collect()
        if device != 'cpu':
            torch.cuda.empty_cache()
    return arm_values, arm_times


def cpu_smoke():
    reg = registered()
    ids = reg['ids'][:2]
    arms, times = execute('cpu', ids, WORK/'cpu-journal')
    valid_probs(.5*(arms['320']+arms['384']), 2)
    # Exercise the real four-ruler/paired-bootstrap path with historical predictions,
    # not the undersized two-study smoke as scientific evidence.
    all_ids, refs, draws, _ = authenticate()
    historical = history.load_oof(CHECKPOINT.parent/history.OOF_PRIMARY, all_ids)
    replay = comparisons({'320':historical, '384':historical.copy()}, refs, draws)
    need(replay['gate']['status'] == 'HOLD' and replay['mean_minus_crop320']['primary_mean_delta'] == 0,
         'identical-arm evaluator replay failed')
    result = dict(version=VERSION, status='CPU_SMOKE_PASS', studies=2,
                  registration_sha256=sha(WORK/'registration.json'), timing_seconds=times,
                  real_study_window_parity='exact at both resolutions and mirrors',
                  rulers_cpu_replay='888 historical rows; identical arms correctly HOLD',
                  cuda_validated=False, float32_only=True)
    io.atomic_json(WORK/'cpu-smoke.json', result)
    return result


def guard_reason(now, last_progress, gpu, total_mib):
    if now-last_progress >= 900:
        return '15 minutes without committed study/view progress'
    if gpu[1] >= 95 and gpu[2] < 90 and gpu[0] >= total_mib-3072 and now-last_progress >= 600:
        return 'WDDM stall signature for 10 minutes without progress'
    return None


def no_other_ml(allowed):
    import psutil
    query = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True,
                           check=True, timeout=5)
    for row in csv.reader(query.stdout.splitlines()):
        if len(row) < 2 or not row[0].strip().isdigit():
            continue
        pid = int(row[0])
        try:
            name = psutil.Process(pid).name().lower()
        except psutil.NoSuchProcess:
            continue
        need(pid in allowed or not any(x in name for x in ('python','jupyter','train','inference')),
             'other GPU ML process active; desktop C+G apps are allowed')


def require_gpu_job():
    import psutil
    ancestors = {p.pid: p for p in psutil.Process().parents()}
    job = io.read_json(ROOT/'artifacts/logs/HB158_SCALE/job.json')
    need(job.get('tag') == 'HB158_SCALE' and job.get('pid') in ancestors
         and ancestors[job['pid']].create_time() == job.get('create_time')
         and job.get('phase') in ('spawning','running'), 'must run through gpu_job tag HB158_SCALE')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0'), 'physical GPU zero required for telemetry parity')
    no_other_ml(set(ancestors) | {os.getpid()})


def run_binding():
    return {name: sha(WORK/name) for name in
            ('registration.json', 'cpu-smoke.json', 'review.json')}


def require_controller(phase):
    """Reject direct internal-worker calls, even under the correct GPU tag."""
    import psutil
    parent = psutil.Process().parent()
    need(parent is not None, 'supervised controller parent required')
    ticket = json.loads(os.environ.get('HB158_CONTROLLER', '{}'))
    argv = parent.cmdline()
    need(ticket.get('pid') == parent.pid and ticket.get('created') == parent.create_time()
         and ticket.get('phase') == phase and ticket.get('binding') == run_binding()
         and len(argv) == 3 and Path(argv[1]).resolve() == Path(__file__).resolve()
         and argv[2] == 'run', 'internal worker requires the live reviewed run controller')
    if phase == 'full':
        completed_phase('calibration', 4)
        check_memory_receipt()


def completed_phase(phase, count):
    """A successful, immutable controller receipt is required before reading rows."""
    from tools.c85r7_flipall_timing import memory_admission
    folder = WORK/'gpu'/phase
    receipt = io.read_json(folder/'controller.json')
    need(receipt.get('exit_code') == 0 and receipt.get('cleanup_errors') == []
         and receipt.get('binding') == run_binding() and receipt.get('phase') == phase
         and len(receipt.get('journal_sha256', {})) == count,
         'completed source/review-bound controller receipt required')
    rows = []
    for name, digest in receipt['journal_sha256'].items():
        need(Path(name).name == name and name.endswith('.json'), 'invalid journal filename')
        path = folder/'journal'/name
        need(sha(path) == digest, 'completed journal changed')
        rows.append(io.read_json(path))
    memory_admission(receipt['desktop'], [r['peak_reserved_bytes'] for r in rows],
                     receipt['peak_device_mib']*2**20)
    return receipt


def check_memory_receipt():
    value = io.read_json(WORK/'gpu/memory-admission.json')
    need(value.get('status') == 'MEASURED_PASS' and value.get('binding') == run_binding()
         and value.get('calibration_controller_sha256') == sha(WORK/'gpu/calibration/controller.json'),
         'measured calibration admission missing/stale')
    return value


def supervise(command, folder, desktop, healthy_rate=None):
    """Owned child, shared retry/cleanup/telemetry primitives, no wall deadline."""
    import psutil
    from tools import hb149_monitor_v125 as mon
    from tools.c85r7_flipall_timing import _GuardedTelemetry, memory_admission
    folder.mkdir(parents=True, exist_ok=True)
    io.atomic_json(folder/'desktop.json', desktop)
    journal = folder/'journal'
    need(not journal.exists() or not list(journal.glob('*.json')), 'attempt has partial outputs; retain and review before a new version')
    started = last_progress = time.perf_counter()
    observed = {}
    rate = mon.RateGuard(healthy_rate, started)
    telemetry = _GuardedTelemetry(os.environ.copy(), .5, desktop)
    status = mon._StatusWriter(folder/'status.json')
    child = job = None
    tracked = {}
    code, error = 1, None
    peak, journal_hashes = 0, {}
    binding = run_binding()
    env = os.environ.copy()
    env['HB158_CONTROLLER'] = json.dumps(dict(pid=os.getpid(), created=psutil.Process().create_time(),
        phase=folder.name, binding=binding))
    try:
        with (folder/'run.log').open('w', encoding='utf-8') as log:
            status.start(); telemetry.start()
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=os.name != 'nt',
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            process = psutil.Process(child.pid)
            tracked[process.pid] = process
            job = mon._WindowsJob(child)
            while child.poll() is None:
                now = time.perf_counter()
                peak, _ = telemetry.guard_snapshot()
                current = mon._journal_snapshot(journal)
                if current != observed:
                    last_progress = now
                observed = current
                sample = telemetry.take()
                if sample is not None:
                    need(sample[1] is None, 'GPU telemetry error')
                    reason = guard_reason(now, last_progress, sample[0], desktop['total_mib'])
                    need(reason is None, reason)
                need(now-last_progress < 900, 'no progress for 15 minutes')
                reason = rate.update(now, len(current))
                need(reason is None, reason)
                try:
                    owned = [process] + process.children(recursive=True)
                    owned_ram = sum(getattr(p.memory_info(), 'private', p.memory_info().rss) for p in owned)
                except psutil.NoSuchProcess:
                    if child.poll() is not None:
                        break
                    raise
                for proc in owned:
                    tracked[proc.pid] = proc
                need(owned_ram < mon.RAM_LIMIT_BYTES, 'owned process RAM guard')
                status.submit(dict(state='RUNNING', phase=folder.name, completed_views=len(current),
                                   idle_seconds=now-last_progress, peak_device_mib=peak,
                                   throughput=rate.snapshot()))
                time.sleep(.5)
            code = child.wait()
            need(code == 0, 'serving worker failed; inspect retained run.log')
            peak, _ = telemetry.guard_snapshot()
            # Memory admission uses actual full-shape CUDA reserved peaks AND device peak.
            rows = [io.read_json(p) for p in sorted(journal.glob('*.json'))]
            memory_admission(desktop, [r['peak_reserved_bytes'] for r in rows], peak*2**20)
            journal_hashes = {p.name: sha(p) for p in sorted(journal.glob('*.json'))}
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        code = 1
    finally:
        cleanup = mon._cleanup(child, tracked, job, psutil)
        telemetry.close() if telemetry.thread.ident is not None else None
        if status.thread.ident is not None:
            status.submit(dict(state='COMPLETE' if code == 0 else 'FAILED', error=error,
                               phase=folder.name, completed_views=len(observed)))
            status.close()
        io.receipt(folder/'controller.json', dict(exit_code=code, error=error, cleanup_errors=cleanup,
                   healthy_rate=healthy_rate, desktop=desktop, no_wall_cap=True,
                   binding=binding, phase=folder.name, journal_sha256=journal_hashes,
                   peak_device_mib=peak, elapsed_seconds=time.perf_counter()-started))
    need(code == 0 and not cleanup, error or 'owned child cleanup failed')


def evaluate_outputs(reg):
    import numpy as np
    admit()
    completed_phase('calibration', 4)
    check_memory_receipt()
    completed_phase('full', 2*len(reg['ids']))
    ids, refs, draws, identity = authenticate()
    need(ids == reg['ids'] and identity == reg['identity'], 'evaluation inputs changed')
    arms, timing = {}, {}
    for res in VIEWS:
        rows = [io.read_json(WORK/'gpu/full/journal'/f'{res}-{i:04d}.json') for i in range(len(ids))]
        need([r['uid'] for r in rows] == ids and all(r['view'] == res and
             r['identity'] == sha(WORK/'registration.json') for r in rows), 'output cohort/identity mismatch')
        arms[str(res)] = valid_probs([r['probabilities'] for r in rows], len(ids))
        seconds = np.asarray([r['seconds'] for r in rows])
        need(np.isfinite(seconds).all() and (seconds > 0).all(), 'invalid measured times')
        timing[str(res)] = dict(total_seconds=float(seconds.sum()),
            mean_seconds_per_study=float(seconds.mean()), median_seconds_per_study=float(np.median(seconds)),
            p95_seconds_per_study=float(np.percentile(seconds,95)))
    result = comparisons(arms, refs, draws)
    result.update(version=VERSION, identity=identity, timing=timing,
        extra_seconds_per_study=timing['384']['mean_seconds_per_study'],
        two_view_seconds_per_study=sum(x['mean_seconds_per_study'] for x in timing.values()),
        timing_scope='4070 native cache -> windows -> both anatomical mirrors -> probabilities; synchronized; decode/model load excluded',
        hidden_projection=None, hidden_projection_note='No cross-hardware/family extrapolation without B measured release phase baseline',
        registration_sha256=sha(WORK/'registration.json'), independent_result_review_required=True)
    io.atomic_json(WORK/'result.json', result)
    return dict(status='RESULT_REVIEW_REQUIRED', rows=len(ids), result=str(WORK/'result.json'))


def run_gpu():
    admit()
    require_gpu_job()
    from tools.c85r7_flipall_timing import desktop_snapshot
    reg = registered()
    calibration = WORK/'gpu/calibration'
    cmd = [sys.executable, str(Path(__file__).resolve()), '_worker', '--phase']
    supervise(cmd+['calibration'], calibration, desktop_snapshot())
    receipt = completed_phase('calibration', 4)
    # Whole-phase measured commit rate includes model load/hash/I/O costs.
    healthy = 60*4/receipt['elapsed_seconds']
    io.atomic_json(WORK/'gpu/memory-admission.json', dict(status='MEASURED_PASS',
                   calibration_controller_sha256=sha(calibration/'controller.json'),
                   binding=run_binding(), healthy_views_per_minute=healthy))
    supervise(cmd+['full'], WORK/'gpu/full', desktop_snapshot(), healthy)
    return evaluate_outputs(reg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['register','validate','prepare','cpu-smoke','admit','run','evaluate','_worker'])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--phase', choices=['calibration','full'])
    args = ap.parse_args(argv)
    WORK.mkdir(parents=True, exist_ok=True)
    if args.command == 'register':
        value = registration()
        io.atomic_json(WORK/'registration.json', value)
        result = dict(status='REGISTERED_CPU_ONLY', heldout=888)
    elif args.command == 'validate':
        registered()
        result = dict(status='VALIDATED_CPU_ONLY', checkpoint_fold_excluded=True)
    elif args.command == 'prepare':
        reg = registered()
        for uid in reg['ids'][:2]:
            volume(uid)
        result = dict(status='PREPARED_CPU_ONLY', studies=2, native_cache='authenticated')
        io.atomic_json(WORK/'prepare.json', result)
    elif args.command == 'cpu-smoke':
        result = cpu_smoke()
    elif args.command == 'admit':
        result = admit(args.dry_run)
    elif args.command == '_worker':
        need(args.phase is not None, 'worker phase required')
        require_controller(args.phase)
        admit(); require_gpu_job()
        reg = registered()
        ids = reg['ids'][:2] if args.phase == 'calibration' else reg['ids']
        execute('cuda:0', ids, WORK/'gpu'/args.phase/'journal')
        result = dict(status='WORKER_COMPLETE', studies=len(ids))
    elif args.command == 'evaluate':
        result = evaluate_outputs(registered())
    else:
        result = run_gpu()
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
