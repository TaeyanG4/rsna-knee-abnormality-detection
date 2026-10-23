"""L1 RCO K0 runner/measurement v1.2.0. GPU run is B-only, review-gated.
Changelog: V2 attempt; evaluation chunk 32 only, unchanged training and gates.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.l1_rco_stage0_v120 import (ROOT, WORK, CACHE, CODE, INIT, VERSION, PYTHON,
    read, write, sha, digest, need, validate, trainer_argv, eval_pins)

GIB = 2**30

def memory_ok(peak, desktop, total):
    return all(math.isfinite(x) and x >= 0 for x in (peak, desktop, total)) and peak <= 12 * GIB and peak <= 15e9 and desktop + peak + 2 * GIB <= total

def go_rule(m):
    return (m['steps'] == 200 and m['eval_batches'] > 0 and m['eval_finite']
            and m['windows_per_s'] >= 80 and all(math.isfinite(x) for x in m['losses'])
            and len(m['losses']) == 200 and sum(m['losses'][-20:]) < sum(m['losses'][:20])
            and memory_ok(m['peak_reserved'], m['desktop_bytes'], m['total_bytes']))

def gpu_snapshot():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], text=True)
    rows = [x.strip().split(', ') for x in out.splitlines() if x.strip()]
    need(len(rows) == 1 and '4070 Ti SUPER' in rows[0][1], 'requires exclusive local RTX 4070 Ti SUPER')
    r = rows[0]
    return {'index': int(r[0]), 'name': r[1], 'total_bytes': int(r[2]) * 2**20,
            'desktop_bytes': int(r[3]) * 2**20, 'util': float(r[4]), 'power': float(r[5])}

def active_ml_processes(owner_pid=None):
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    return [line for line in out.splitlines() if 'python' in line.lower() and line.split(',')[0].strip() != str(owner_pid)]

def stale_reason(status, now, telemetry=None):
    age = now - status['time']
    if age >= 900:
        return '15-minute zero-progress watchdog'
    if telemetry and age >= 600 and telemetry['util'] >= 95 and telemetry['power'] < 90 and telemetry['desktop_bytes'] >= .90 * telemetry['total_bytes']:
        return 'WDDM paging signature for >=10 minutes without progress'
    return None

class Monitor:
    def __init__(self, out, enabled=True):
        self.enabled = enabled; self.steps = 0; self.out = Path(out)
        if not enabled:
            return
        import torch
        self.torch = torch
        self.admission = read(self.out / 'admission.json')
        self.losses = []; self.windows = 0; self.eval_batches = 0; self.eval_finite = True
        self.train_start = None; self.train_seconds = 0.; self.peak = 0
        self.phase_name = 'init'; self.phase('init')

    def emit(self):
        peak = self.torch.cuda.max_memory_reserved()
        self.peak = max(self.peak, peak)
        need(memory_ok(self.peak, self.admission['desktop_bytes'], self.admission['total_bytes']), 'measured memory gate failed; recipe must not auto-shrink')
        status = {'phase': self.phase_name, 'steps': self.steps, 'windows': self.windows, 'time': time.time(),
                  'peak_reserved': self.peak, 'eval_batches': self.eval_batches,
                  'train_seconds': self.train_seconds, 'version': VERSION}
        try:
            write(self.out / 'status.json', status)
        except PermissionError:
            pass  # parent watchdog eventually stops a persistently unwritable heartbeat

    def phase(self, name):
        self.phase_name = name
        if name == 'train':
            self.train_start = time.perf_counter()
        self.emit()

    def step(self, loss, mask):
        self.torch.cuda.synchronize()
        v = float(loss.detach())
        need(math.isfinite(v), 'nonfinite training loss')
        self.losses.append(v); self.steps += 1; self.windows += int(mask.sum())
        self.train_seconds = time.perf_counter() - self.train_start
        need(self.steps <= 200, 'smoke step cap')
        self.emit()

    def evaluation(self, logits, mask):
        self.torch.cuda.synchronize()
        self.eval_batches += 1
        self.eval_finite &= bool(self.torch.isfinite(logits).all())
        need(self.eval_finite, 'nonfinite evaluation logits')
        self.emit()

    def finish(self):
        self.phase('complete')
        result = {'version': VERSION, 'contract_id': self.admission['contract_id'], 'steps': self.steps,
                  'windows': self.windows, 'train_seconds': self.train_seconds,
                  'windows_per_s': self.windows / self.train_seconds, 'losses': self.losses,
                  'eval_batches': self.eval_batches, 'eval_finite': self.eval_finite,
                  'peak_reserved': self.peak, 'desktop_bytes': self.admission['desktop_bytes'],
                  'total_bytes': self.admission['total_bytes'], 'clean_oof': False,
                  'exposure': 'EXPOSED_EXPLORATORY', 'T4_serving_increment': 'UNMEASURED',
                  'review_binding': self.admission['review_binding']}
        result['K0_numeric_gate'] = 'PASS' if go_rule(result) else 'FAIL'
        result['decision'] = 'HOLD: independent result review and B decision required; no full fit authorization'
        write(self.out / 'result.json', result)
        cache = read(Path(read(WORK / 'prepare.json')['cache']) / 'receipt.json')
        projection = read(WORK / 'projection.json')
        projection.update({'status': 'MEASURED_LOCAL_PENDING_INDEPENDENT_REVIEW',
                           'full_cache_bytes': cache['bytes_per_study'] * 4167,
                           'full_cache_hours': cache['seconds_per_study'] * 4167 / 3600,
                           'fold0_fit_hours': 3190 * 64 * 16 / result['windows_per_s'] / 3600,
                           'fit_projection_scope': 'training lower bound, excludes evaluation/checkpoints; subset LR schedule differs from full fit',
                           'cache_projection_scope': 'same CPU serial renderer; first-study setup included, assembly/checksum extra; retained shards require ~2x disk',
                           'kaggle_T4_serving_status': 'UNMEASURED',
                           'K0_result_sha256': sha(self.out / 'result.json')})
        write(WORK / 'projection-after-K0.json', projection)

def bindings(reg, cache):
    prep = read(WORK / 'prepare.json')
    need(prep['eval_pins'] == eval_pins() and prep['version'] == VERSION and prep['eval_chunk'] == 32, 'evaluation source/recipe changed')
    need(prep['contract_id'] == reg['contract_id'], 'prepare contract mismatch')
    need(prep['targets_sha256'] == sha(WORK / 'targets.csv') and prep['trainer_sha256'] == sha(WORK / 'compact_train_l1.py'), 'prepared file changed')
    need(prep['argv'] == trainer_argv(cache) and prep['cache'] == str(Path(cache).resolve()), 'cache/argv mismatch')
    proof = read(WORK / 'eval-proof.json')
    need(proof['status'] == 'PASS' and proof['trainer_sha256'] == prep['trainer_sha256'] and proof['contract_id'] == reg['contract_id'], 'eval proof identity')
    return {'contract_id': reg['contract_id'], 'prepare_sha256': sha(WORK / 'prepare.json'),
            'eval_proof_sha256': sha(WORK / 'eval-proof.json'),
            'cpu_smoke_sha256': sha(WORK / 'cpu-smoke.json'), 'cache_receipt_sha256': sha(Path(cache) / 'receipt.json')}

def admit(cache, review, dry_run=False, owner_pid=None):
    reg = validate(read(WORK / 'registration.json'))
    need(len(reg['train']) == 200 and len(reg['holdout']) == 32, 'K0 requires registered 200+32 subset')
    from tools.l1_rco_cache_v1 import check_cache
    blockers = []
    binding = None
    try:
        check_cache(reg, cache)
        binding = bindings(reg, cache)
        smoke = read(WORK / 'cpu-smoke.json')
        need(smoke['contract_id'] == reg['contract_id'] and smoke['status'] == 'PASS' and not smoke['cuda_initialized'], 'CPU smoke receipt')
        r = read(review)
        need(r['verdict'] == 'PASS' and r['builder'] != r['reviewer'] and r['bindings'] == binding, 'independent exact-artifact review required')
        need(sha(ROOT / r['report']) == r['report_sha256'], 'review report changed')
    except (ValueError, FileNotFoundError, KeyError) as e:
        blockers.append(str(e))
    if dry_run:
        result = {'dry_run': True, 'status': 'BLOCKED' if blockers else 'CPU_ADMISSIBLE', 'blockers': blockers, 'bindings': binding}
        write(WORK / 'admit-dry-run.json', result)
        return result
    need(not blockers, '; '.join(blockers))
    need(not active_ml_processes(owner_pid), 'another Python ML GPU process exists')
    snapshot = gpu_snapshot()
    need(snapshot['desktop_bytes'] + 2 * GIB < snapshot['total_bytes'], 'no headroom')
    snapshot.update({'contract_id': reg['contract_id'], 'review_binding': binding})
    return snapshot

def verify_job_record(job, command, current, parent, parent_argv):
    from tools.job_runtime import command_digest
    need(job.get('schema') == 'gpu-job/v2' and job.get('tag') == 'L1_RCO_K0_V2' and job.get('phase') == 'running', 'not a running L1 gpu_job')
    need(job.get('child_identity') == current and {k: job.get(k) for k in ('pid', 'create_time')} == parent, 'gpu_job process identity mismatch')
    need(job.get('command') == command and job.get('command_sha256') == command_digest(command, ROOT) and Path(job['cwd']).resolve() == ROOT, 'gpu_job command mismatch')
    need(parent_argv[-3:] == ['--_run-job', 'L1_RCO_K0_V2', job['run_id']] and any(Path(x).name == 'gpu_job.py' for x in parent_argv), 'gpu_job parent argv mismatch')


def require_gpu_job(cache, review):
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json
    command = [PYTHON, str(ROOT / 'tools/l1_rco_runtime_v120.py'), 'run', '--cache', str(cache), '--review', str(review)]
    # The same lock closes the spawn -> child_identity publication race.
    with file_lock(LOGS / '.gpu-claim.lock'):
        job = read_json(LOGS / 'L1_RCO_K0_V2/job.json')
        parent = psutil.Process().parent()
        need(parent is not None, 'missing gpu_job parent')
        verify_job_record(job, command, identity(), identity(parent.pid), parent.cmdline())
        return parent.pid  # gpu_job environment snapshot may own an idle CUDA context


def run(cache, review):
    # Only B invokes this via gpu_queue -> gpu_job. No CPU preparation calls it.
    owner_pid = require_gpu_job(cache, review)
    admission = admit(cache, review, owner_pid=owner_pid)
    out = WORK / 'k0'; out.mkdir(parents=True, exist_ok=True)
    need(not (out / 'admission.json').exists(), 'attempt already admitted; retain partial attempt; B must register a new version to repeat')
    write(out / 'admission.json', admission)
    write(out / 'status.json', {'phase': 'launch', 'steps': 0, 'windows': 0, 'train_seconds': 0, 'time': time.time()})
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '0'; env['CT_PREDICT_CHUNK'] = '32'
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / CODE)])
    child = None
    try:
        with (out / 'train.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen(trainer_argv(cache), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            last_good = time.time(); history = []; best = 0.
            status = read(out / 'status.json')
            while child.poll() is None:
                time.sleep(15)
                try:
                    status = read(out / 'status.json')
                except (OSError, json.JSONDecodeError):
                    pass  # retain last readable status; missing heartbeat must still time out
                try:
                    telemetry = gpu_snapshot()
                except (OSError, subprocess.SubprocessError, ValueError):
                    telemetry = None  # telemetry failure alone never aborts inference
                reason = stale_reason(status, time.time(), telemetry)
                need(not reason, reason or '')
                if status['phase'] == 'train':
                    history.append((time.time(), status['windows']))
                    if len(history) >= 2 and history[-1][0] - history[0][0] >= 120:
                        rate = (history[-1][1] - history[0][1]) / (history[-1][0] - history[0][0])
                        best = max(best, rate)
                        if rate >= .25 * best:
                            last_good = time.time()
                        need(time.time() - last_good < 1800, '30-minute throughput collapse watchdog')
                        history = history[-9:]
            need(child.returncode == 0, 'trainer child failed; inspect owned train.log after stop')
            result = read(out / 'result.json')
            need(result['steps'] == 200, 'incomplete K0')
    except Exception as e:
        write(out / 'failure.json', {'error': str(e), 'time': time.time(), 'partial_preserved': True})
        raise
    finally:
        if child is not None and child.poll() is None:
            # This exact Popen-owned process tree only, including DataLoader children.
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True, check=False)
            else:
                child.terminate()
            child.wait(timeout=30)

def cpu_smoke(cache):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    torch.set_num_threads(2)
    reg = validate(read(WORK / 'registration.json'))
    from tools.l1_rco_cache_v1 import check_cache
    ids = reg['train'][:2] + reg['holdout'][:1]
    proof = check_cache(reg, cache, ids)
    need(proof['serving_cache_parity']['status'] == 'PASS' and proof['serving_cache_parity']['studies'] == 3 and all(g['source_crop_and_padding_valid'] and g['checked_calls'] > 0 for g in proof['geometry']), 'real serving/cache and geometry proof missing')
    # Original trainer parser + fold/target assembly, on real cached UIDs.
    env = os.environ.copy(); env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / CODE)])
    p = subprocess.run(trainer_argv(cache, check=True), cwd=ROOT, env=env, capture_output=True, text=True)
    (WORK / 'check-targets.log').write_text(p.stdout + p.stderr, encoding='utf-8')
    need(p.returncode == 0, 'check-targets failed; see log')
    spec = importlib.util.spec_from_file_location('l1_cpu_trainer', WORK / 'compact_train_l1.py')
    ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
    ds = ct.Studies(str(cache), ids, 64, 'train', 9000, train_sampling='cover', cover_stride=3, slices=3, slice_jitter=.25, phys_mm=3.5)
    batch = ct.collate([ds[0], ds[1]])
    img, meta, mask, idx = batch
    need(img.shape[-2:] == (384, 384) and bool(mask.any()) and bool(torch.isfinite(meta).all()), 'real loader contract')
    ev = ct.Studies(str(cache), [ids[-1]], 64, 'eval', 9000, eval_stride=3, slices=3, phys_mm=3.5)
    eimg, emeta, emask, _ = ct.collate([ev[0]])
    need(bool(emask.any()), 'holdout evaluation loader')
    net = ct.Net('convnext_small', str(ROOT / INIT), crop=352, in_chans=3, spatial=True, extent=False)
    # Full spatial resolution, two real physical windows. No GPU/full batch claim.
    x = img[:1, :2, :, 16:368, 16:368]; m = meta[:1, :2]; mk = mask[:1, :2]
    net.train(); z = net(x, m, mk)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z.float(), torch.full_like(z, .5))
    loss.backward()
    need(bool(torch.isfinite(z).all()) and any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and p.grad.abs().sum() > 0 for p in net.parameters()), 'CPU forward/backward')
    net.eval()
    with torch.no_grad():
        ez = net(eimg[:, :2, :, 16:368, 16:368], emeta[:, :2], emask[:, :2], chunk=32)
    need(bool(torch.isfinite(ez).all()) and not torch.cuda.is_initialized(), 'CPU eval/CUDA isolation')
    receipt = {'version': VERSION, 'contract_id': reg['contract_id'], 'status': 'PASS', 'cuda_initialized': torch.cuda.is_initialized(),
               'real_train_studies': 2, 'real_holdout_studies': 1, 'model_windows': 2, 'spatial_crop': 352,
               'train_loader_windows': int(mask.sum()), 'eval_loader_windows': int(emask.sum()),
               'finite_forward_backward_eval': True, 'check_targets_exit': p.returncode,
               'serving_cache_parity': 'PASS: all physical eval windows, pixels and metadata', 'valid_area_geometry': 'PASS',
               'NOT_CHECKED': ['CUDA/full-batch memory/throughput', 'T4 serving', 'full 232 subset cache'],
               'cache_receipt_sha256': sha(Path(cache) / 'receipt.json')}
    write(WORK / 'cpu-smoke.json', receipt)
    return receipt

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['cpu-smoke', 'admit', 'run'])
    ap.add_argument('--cache', type=Path, default=CACHE)
    ap.add_argument('--review', type=Path, default=WORK / 'K0-REVIEW.json')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.action != 'run':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if a.action == 'cpu-smoke':
        result = cpu_smoke(a.cache)
    elif a.action == 'admit':
        need(a.dry_run, 'use run through gpu_job for real admission')
        result = admit(a.cache, a.review, True)
    else:
        need(not a.dry_run, 'run does not accept dry-run; use admit')
        result = run(a.cache, a.review)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
