"""EXP105 own-arm worker (H-B-15): one arm of the reviewed release v5 (5-fold mean_prob ensemble at K36) on one pinned
GPU, inputs from the anchor's shared decode (raptor_shared_input.StoreCorpus: rows without a usable store record are
decoded by the unchanged DicomCorpus), scored one study at a time with the unchanged raptor_runtime.predict (batch 1).

A study whose input cannot be prepared (store row unusable and the DICOM fallback failing) is listed in
receipt['unscored'] with its error and left out of predictions.csv; the parent keeps the anchor's rank for it.
Every other problem -- release/asset bindings, dependency versions, checkpoint metadata, a non-finite member logit or
probability, a store audit mismatch, the deadline -- ends with status FAIL and exit code 1, and the parent keeps the
anchor-only submission. Bindings, checkpoint checks and the ensemble module are the reviewed v5 functions
(exp104_release_v5.py, loaded from --src and pinned by sha256). No labels, training or scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

SCHEMA = 'exp105-arm-worker-v1'
UID = 'StudyInstanceUID'
EXP104_SOURCE_SHA256 = '2100a5cb5b035c96ecd344b0383425412f54c24c59e5e6cdacec9cc555677aeb'
RELEASE_MANIFEST_SHA256 = 'dd1a341fc4c4f4d696c75f75d88227c96b2d721b29b92790c5fcf72931306c84'
K_EVAL = 36
INPUT_FAILURE = 'input preparation failed for '   # raptor_runtime.predict's wrapper for prepare_input errors


class AuditFailed(RuntimeError):
    pass


class DeadlineExceeded(RuntimeError):
    pass


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


def load_v5(src):
    path = Path(src) / 'exp104_release_v5.py'
    need(sha(path) == EXP104_SOURCE_SHA256, 'exp104_release_v5.py differs from the reviewed v5 source')
    spec = importlib.util.spec_from_file_location('exp104_release_v5_for_exp105_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quiet(*args, **kwargs):
    return None


def score_studies(predict, model, corpus, ids, arm, amp, device, deadline=None, clock=time.time, log=print):
    """One predict call per study (batch 1, the same numerical path as one call over all studies).

    -> (probs {uid: 12 floats}, unscored {uid: reason}, input hashes [{uid, sha256}], ordered aggregate sha256)."""
    import numpy as np
    probs, unscored, hashes = {}, {}, []
    aggregate = hashlib.sha256()
    started = time.perf_counter()
    for i, uid in enumerate(ids):
        if deadline is not None and clock() > deadline:
            raise DeadlineExceeded('deadline reached after %d of %d studies' % (i, len(ids)))
        try:
            corpus.volume(uid)
            corpus.mask(uid)
            failure = None
        except Exception as exc:              # store row unusable and the DICOM fallback failed
            failure = 'input: %s: %s' % (type(exc).__name__, exc)
        if getattr(corpus, 'trusted', True) is False:
            raise AuditFailed('shared store audit mismatch at %s: %s' % (uid, getattr(corpus, 'audits', None)))
        if failure is None:
            try:
                p, timing = predict(model, corpus, [uid], arm['k_eval'], arm['res'], amp, device, norm=arm['norm'],
                                    batch=1, log=_quiet, window_policy=arm['window_policy'])
            except RuntimeError as exc:
                if not str(exc).startswith(INPUT_FAILURE):
                    raise                     # inference failure (incl. non-finite logits): the whole arm fails
                failure = 'prepare: %s' % exc
        if failure is not None:
            unscored[uid] = failure[:500]
            continue
        p = np.asarray(p)
        need(p.shape == (1, 12) and bool(np.isfinite(p).all()) and bool(((p >= 0) & (p <= 1)).all()),
             'non-finite or out-of-range probability for %s' % uid)
        need(timing.get('fp32_fallbacks') == 0 and timing.get('all_finite') is True, 'precision fallback for %s' % uid)
        record = timing['input_tensor_hashes'][0]
        need(record['uid'] == str(uid), 'input hash order differs')
        hashes.append(record)
        aggregate.update((json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n').encode())
        probs[uid] = [float(v) for v in p[0]]
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            log('  %d/%d studies, %.0fs, unscored %d' % (i + 1, len(ids), time.perf_counter() - started, len(unscored)),
                flush=True)
    return probs, unscored, hashes, aggregate.hexdigest()


def write_predictions(path, ids, probs, targets):
    import csv
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow([UID, *targets])
        for uid in ids:
            if uid in probs:
                w.writerow([uid, *[repr(float(v)) for v in probs[uid]]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ('assets', 'new-assets', 'release-manifest', 'src', 'env', 'store', 'ids-json', 'rows-csv',
                'dicom-root', 'arm', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--cpus', default='', help='comma-separated CPU ids for sched_setaffinity')
    p.add_argument('--audit', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', default='fp16')
    p.add_argument('--deadline', type=float, default=None, help='epoch seconds; FAIL when passed')
    a = p.parse_args(argv)
    assets, new, out = Path(a.assets), Path(a.new_assets), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    r = {'schema': SCHEMA, 'status': 'RUNNING', 'arm': a.arm, 'pid': os.getpid(), 'device': a.device, 'amp': a.amp,
         'training': 0, 'scoring': 0, 'label_reads': 0, 'retries': 0}
    code = 1
    try:
        # Under memory pressure the kernel must kill this worker (the run falls back to the anchor-only file),
        # never the anchor's notebook process (which would lose the whole submission). Review 2026-10-07.
        try:
            with open('/proc/self/oom_score_adj', 'w') as f:
                f.write('1000')
            r['oom_score_adj'] = 1000
        except OSError as exc:
            r['oom_score_adj'] = 'unset: ' + str(exc)
        if a.cpus and hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, {int(c) for c in a.cpus.split(',')})
            r['cpus'] = sorted(os.sched_getaffinity(0))
        threads = str(max(1, a.threads))
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                          KNEE_WORKSPACE_ROOT=str(assets), OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                          OPENBLAS_NUM_THREADS=threads)
        sys.path[:0] = [x for x in (a.env, str(assets / 'training' / 'src'), a.src) if x]
        t = time.perf_counter()
        v5 = load_v5(a.src)
        release = v5.checked_manifest(a.release_manifest, RELEASE_MANIFEST_SHA256)
        arm = v5.arm_by_name(release, a.arm)
        need(arm['k_eval'] == K_EVAL and arm['aggregator'] == 'mean_prob', 'arm is not the K36 mean_prob v5 arm')
        v5.verify_old_assets(assets)                       # serving sources and wheels by sha256
        v5.verify_new_assets(release, new, [arm])          # assets manifest + every checkpoint file by sha256
        r.update(release=release['release'], release_manifest_sha256=RELEASE_MANIFEST_SHA256,
                 new_assets_manifest_sha256=release['new_assets_manifest_sha256'], k_eval=arm['k_eval'],
                 n_models=len(arm['checkpoints']), rank_weight=arm['weight'],
                 verify_seconds=round(time.perf_counter() - t, 3))
        import numpy as np
        import pandas as pd
        import torch
        import timm, cv2, pydicom
        r['versions'] = {'torch': torch.__version__, 'timm': timm.__version__, 'cv2': cv2.__version__,
                         'pydicom': pydicom.__version__, 'numpy': np.__version__, 'python': sys.version.split()[0]}
        if a.device == 'cuda':
            import gdcm
            r['versions']['gdcm'] = gdcm.Version.GetVersion()
            need(sys.platform.startswith('linux') and sys.version_info[:2] == (3, 12), 'Linux CPython 3.12 required')
            need((torch.__version__, timm.__version__, cv2.__version__, pydicom.__version__, r['versions']['gdcm'])
                 == v5.VERSIONS, 'pinned dependency versions required: %s' % r['versions'])
            need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0),
                 'exactly one visible T4 required')
            r['gpu'] = torch.cuda.get_device_name(0)
        torch.set_num_threads(int(threads))
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        from raptor_common import LAB, load_checkpoint, environment
        from raptor_runtime import configure_runtime, predict
        from raptor_dicom import DirectoryDicomSource
        from raptor_shared_input import SHARED_INPUT_VERSION, StoreCorpus
        need(LAB == v5.TARGETS, 'target order differs')
        for module_name, rel in (('raptor_common', 'training/src/raptor_common.py'),
                                 ('raptor_runtime', 'training/src/raptor_runtime.py'),
                                 ('raptor_dicom', 'training/src/raptor_dicom.py')):
            need(Path(sys.modules[module_name].__file__).resolve() == (assets / rel).resolve(),
                 '%s resolved outside the EXP089 assets' % module_name)
        r['runtime'] = configure_runtime(a.amp, a.device, 'strict')
        r['environment'] = environment()
        t = time.perf_counter()
        members, checks = [], []
        for ck in arm['checkpoints']:
            member, meta = load_checkpoint(new / ck['path'])
            checks.append(v5.checkpoint_check(meta, arm, ck))
            members.append(member.eval().to(a.device))     # CPU memory holds at most one checkpoint
        model = v5.mean_prob_module(members)                # reviewed v5: non-finite member logit fails closed
        r['checkpoints'] = checks
        r['model_load_seconds'] = round(time.perf_counter() - t, 3)
        ids = [str(u) for u in json.loads(Path(a.ids_json).read_text(encoding='utf-8'))]
        need(ids and len(ids) == len(set(ids)), 'ids must be unique and nonempty')
        rows = pd.read_csv(a.rows_csv, dtype={UID: str, 'SeriesInstanceUID': str}).to_dict('records')
        source = DirectoryDicomSource(a.dicom_root)
        corpus = StoreCorpus(a.store, ids, rows, source, arm['contract'], a.split, audit=a.audit)
        t = time.perf_counter()
        probs, unscored, hashes, aggregate = score_studies(predict, model, corpus, ids, arm, a.amp, a.device,
                                                           deadline=a.deadline)
        seconds = time.perf_counter() - t
        need(probs, 'no study scored')
        write_predictions(out / 'predictions.csv', ids, probs, v5.TARGETS)
        stats = corpus.stats()
        r.update(status='PASS', adapter=SHARED_INPUT_VERSION, studies=len(ids), ids_sha256=ids_sha256(ids),
                 scored=len(probs), unscored=unscored, served=stats['served'], audits=stats['audits'],
                 store_error=stats['store_error'], store_trusted=stats['trusted'],
                 timing={'seconds': round(seconds, 3), 'per_study_s': round(seconds / len(ids), 4),
                         'ordered_scored_inputs_sha256': aggregate, 'input_tensor_hashes': hashes},
                 fallback_receipts={u: x for u, x in list(corpus.receipts.items()) if x.get('source') != 'store'}
                 if len(corpus.receipts) - stats['served'].get('store', 0) <= 50 else 'more than 50 (omitted)',
                 torch_threads=torch.get_num_threads(), predictions_sha256=sha(out / 'predictions.csv'),
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if a.device == 'cuda' else 0)
        code = 0
    except BaseException as exc:
        r.update(status='FAIL', error={'type': type(exc).__name__, 'message': str(exc)[:2000]})
        code = 1
    finally:
        r['worker_wall_seconds'] = round(time.perf_counter() - started, 3)
        tmp = out / 'receipt.json.tmp'
        tmp.write_text(json.dumps(r, indent=1, default=str), encoding='utf-8')
        os.replace(tmp, out / 'receipt.json')
    print(json.dumps({'status': r['status'], 'arm': a.arm, 'seconds': r['worker_wall_seconds'],
                      'scored': r.get('scored'), 'unscored': len(r.get('unscored') or {}),
                      'error': r.get('error')}), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
