"""Prepare a fixed single-model recipe transfer; check deployment batch on V8.

prepare reads metadata and opaque file bytes only. local requires a separate root
authorization and performs no label access, scoring, training, or DICOM decoding.
The local result is engineering evidence, not T4/hidden-runtime or accuracy proof.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'artifacts/experiments/EXP089_RECIPE_TRANSFER_G1_PROTOCOL_V1'
ARMS = [('B224', 'EXP084_B_G1', 224, 8),
        ('R384', 'EXP088_RAPTOR384_G1_V4', 384, 12)]
LIMITS = {'max_abs': .002, 'p99_abs': .002, 'mean_abs': .0005}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def prepare():
    need(not PROTOCOL.exists(), 'Preserve existing protocol; choose a new version')
    previous = read(ROOT / 'artifacts/experiments/EXP088_RAPTOR384_G1_PROTOCOL_V7/manifest.json')
    ordered = read(ROOT / previous['ordered_ids']['path'])['val']
    engineering = read(ROOT / '.local/b-rn3f-staging/20261004-v1/assets/engineering100.json')['ids']
    selected = [uid for uid in ordered if uid in set(engineering)][:8]
    need(len(selected) == 8, 'Need eight V UIDs already covered by stored DICOM/cache evidence')
    sources = [Path(__file__), ROOT / 'tools/process_resource_receipt.py', ROOT / 'tools/gpu_job.py',
               *[ROOT / 'training/src' / n for n in
                 ['raptor_common.py', 'raptor_runtime.py', 'raptor_ensemble.py', 'raptor_dicom.py']]]
    arms, bindings = [], {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sources}
    for name, tag, res, epoch in ARMS:
        directory = ROOT / 'artifacts/experiments' / tag
        summary, provenance = read(directory / 'summary.json'), read(directory / 'provenance.json')
        need(summary['selected_epoch_1based'] == epoch and summary['skipped_updates'] == 0,
             'Only completed fixed final checkpoints are allowed')
        need(provenance['train_ids_ordered_sha256'] == previous['ordered_ids']['train_sha256']
             and provenance['eval_ids_ordered_sha256'] == previous['ordered_ids']['val_sha256'],
             'Matched G1 training/validation UID ancestry required')
        for filename in ['model_best.pt', 'oof_best.csv', 'summary.json', 'provenance.json']:
            p = directory / filename
            bindings[p.relative_to(ROOT).as_posix()] = sha(p)
        arms.append({'name': name, 'tag': tag, 'res': res, 'epoch_0based': epoch - 1,
                     'arch': summary['args']['arch'], 'checkpoint': (directory / 'model_best.pt').relative_to(ROOT).as_posix(),
                     'reference': (directory / 'oof_best.csv').relative_to(ROOT).as_posix()})
    p = {'schema': 'exp089-single-recipe-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
         'status': 'PREPARED_NOT_AUTHORIZED', 'arms': arms, 'uids': selected,
         'uid_selection': 'first eight ordered V members present in existing engineering100; not representative',
         'targets': previous['targets'], 'corpus': previous['paths']['corpus'],
         'corpus_contract': 'raptor64_336_c140_s02', 'norm': 'imagenet', 'k_eval': 24,
         'window_policy': 'within-slot-valid', 'batch': 1, 'amp': 'fp16', 'runtime_policy': 'strict',
         'tolerances': LIMITS, 'bindings': bindings, 'timeout_seconds': 240,
         'postrun_sampled_rss_acceptance_bytes': 16 * 2**30,
         'limits_note': 'Timeout enforced by existing supervisor; RSS sampled/postrun acceptance, not a memory kill cap',
         'CPU_workers': 1, 'internal_threads': 1, 'retries': 0,
         'scope': 'new deployment batch engineering only, no labels/metrics/DICOM/Gold; neither T4 nor hidden guarantee'}
    PROTOCOL.mkdir()
    write_new(PROTOCOL / 'manifest.json', p)
    digest = sha(PROTOCOL / 'manifest.json')
    work = ROOT / '.local/exp089-recipe-transfer/local-v1'
    python = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
    argv = [python, str(ROOT / 'tools/gpu_job.py'), '--tag', 'EXP089_RECIPE_TRANSFER_LOCAL_V1', '--',
            python, str(ROOT / 'tools/process_resource_receipt.py'), '--out', str(work / 'supervisor'),
            '--timeout', '240', '--cwd', str(ROOT), '--', python, str(Path(__file__)), 'local',
            '--protocol', str(PROTOCOL / 'manifest.json'), '--sha256', digest,
            '--decision', str(PROTOCOL / 'root-local-decision.json'), '--out', str(work / 'worker')]
    write_new(PROTOCOL / 'local-plan.json', {'status': 'SEPARATE_SOURCE_REVIEW_AND_ROOT_DECISION_REQUIRED',
               'protocol_sha256': digest, 'cwd': str(ROOT), 'argv': argv, 'GPU': True, 'scoring': False})
    print(json.dumps({'protocol': str(PROTOCOL), 'sha256': digest, 'selected_UIDs': len(selected),
                      'model_deserialization': 0, 'GPU': False, 'remote': False}))


def selected_reference(path, ids, targets):
    import numpy as np
    found = {}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == ['StudyInstanceUID', *targets], 'Reference target order changed')
        for row in reader:
            uid = row['StudyInstanceUID']
            if uid in ids:
                need(uid not in found, 'Duplicate reference UID')
                found[uid] = [float(row[t]) for t in targets]
    need(set(found) == set(ids), 'Selected reference UIDs missing')
    ref = np.asarray([found[u] for u in ids], dtype=np.float64)
    need(bool(np.isfinite(ref).all() and (ref >= 0).all() and (ref <= 1).all()), 'Invalid reference probabilities')
    return ref


def local(a):
    need(sha(a.protocol) == a.sha256, 'Exact protocol required')
    p, decision = read(a.protocol), read(a.decision)
    need(decision.get('status') == 'AUTHORIZE_ONE_LOCAL_ENGINEERING_CHECK'
         and decision.get('protocol_sha256') == a.sha256, 'Separate root authorization required')
    need(p['schema'] == 'exp089-single-recipe-v1', 'Wrong protocol')
    for path, digest in p['bindings'].items():
        need(sha(ROOT / path) == digest, 'Frozen engineering binding changed: ' + path)
    out = Path(a.out).resolve()
    need(out.is_relative_to(ROOT) and not out.exists(), 'New workspace output required')
    out.mkdir(parents=True)
    receipt = {'status': 'RUNNING', 'protocol_sha256': a.sha256, 'arms': [], 'cleanup_errors': [],
               'training': 0, 'scoring': 0, 'label_reads': 0, 'DICOM_decode': 0, 'fp32_fallbacks': 0}
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0')
    sys.path.insert(0, str(ROOT / 'training/src'))
    start, primary = time.perf_counter(), None
    try:
        import numpy as np
        import torch
        from raptor_common import Corpus, LAB, environment, load_checkpoint, write_predictions
        from raptor_runtime import configure_runtime, predict
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One visible GPU required')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        runtime = configure_runtime('fp16', 'cuda', 'strict')
        need(LAB == p['targets'], 'Target order changed')
        corpus = Corpus(ROOT / p['corpus'])
        need(corpus.contract == p['corpus_contract'], 'Corpus contract changed')
        for arm in p['arms']:
            model = None
            try:
                model, meta = load_checkpoint(ROOT / arm['checkpoint'])
                expected = {'arch': arm['arch'], 'res': arm['res'], 'epoch': arm['epoch_0based'],
                            'lab': p['targets'], 'contract': p['corpus_contract'], 'k_eval': 24,
                            'norm': 'imagenet', 'window_policy': 'within-slot-valid'}
                need(all(meta.get(k) == v for k, v in expected.items()), 'Saved checkpoint contract differs')
                probs, timing = predict(model, corpus, p['uids'], 24, arm['res'], 'fp16', 'cuda',
                                        norm='imagenet', batch=1, window_policy='within-slot-valid')
                reference = selected_reference(ROOT / arm['reference'], p['uids'], p['targets'])
                need(probs.shape == (8, 12) and bool(np.isfinite(probs).all()), 'Incomplete/nonfinite output')
                delta = np.abs(probs.astype(np.float64) - reference)
                errors = {'max_abs': float(delta.max()), 'p99_abs': float(np.quantile(delta, .99)),
                          'mean_abs': float(delta.mean())}
                valid = all(errors[k] <= p['tolerances'][k] for k in errors)
                receipt['arms'].append({'name': arm['name'], 'status': 'PASS' if valid else 'FAIL',
                      'errors': errors, 'limits': p['tolerances'], 'timing': timing})
                need(valid, 'Deployment batch1 differs from stored final probabilities beyond fixed tolerance')
                write_predictions(out / (arm['name'] + '-engineering.csv'), p['uids'], probs)
            finally:
                if model is not None:
                    try:
                        model.to('cpu')
                        del model
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception as exc:
                        receipt['cleanup_errors'].append(str(exc))
            need(not receipt['cleanup_errors'], 'Device cleanup failed')
        receipt.update(status='PASS', runtime=runtime, environment=environment(),
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        primary = exc
        receipt.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        receipt['wall_seconds'] = time.perf_counter() - start
        receipt['cleanup_complete'] = not receipt['cleanup_errors']
        receipt['primary_exception_preserved'] = primary is not None
        write_new(out / 'receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'wall_seconds': receipt['wall_seconds'], 'model_runs': 2}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    r = sub.add_parser('local')
    for flag in ['protocol', 'sha256', 'decision', 'out']:
        r.add_argument('--' + flag, required=True)
    a = parser.parse_args()
    prepare() if a.command == 'prepare' else local(a)


if __name__ == '__main__':
    main()
