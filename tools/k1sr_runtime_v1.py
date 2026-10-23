"""K1 shared-R6 release runtime v1.0.0.

Changelog: first in-pass integration; unchanged renderer and frozen K1 trainer.
Only the classifier owner loads K1. Loader state is created after spawn.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import signal
import sys
import time

from tools import c85r7_rco_overlay_v1 as util
from tools import k1_shared_render_v1 as shared
from tools import l1_rco_k2_worker_v102 as w

VERSION = '1.0.0'
GUARD_SECONDS = 8 * 3600
CALL_TIMEOUT_SECONDS = 120
PARENT_WEIGHT, K1_WEIGHT = .80, .20
MANIFEST_SHA = util.MANIFEST_SHA


def remaining():
    return float(os.environ['K1SR_STARTED']) + GUARD_SECONDS - time.time()


@contextlib.contextmanager
def deadline(seconds):
    """POSIX Python interrupt; a stuck native CUDA call is not preemptible."""
    if seconds <= 0:
        raise TimeoutError('K1 wall-clock guard')
    if not hasattr(signal, 'setitimer'):
        # Windows CPU verification only. Production is Linux with SIGALRM.
        yield
        return
    def expired(signum, frame):
        raise TimeoutError('K1 operation deadline')
    old = signal.signal(signal.SIGALRM, expired)
    prior = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *prior)
        signal.signal(signal.SIGALRM, old)


def disabled(out):
    return (Path(out).parent / 'k1-disabled.json').exists() or remaining() <= 0


def disable(out, error):
    # Optional-branch telemetry must never break the parent's loop.
    try:
        w.write(Path(out).parent / 'k1-disabled.json', dict(status='DISABLED', error=str(error)[:400]))
    except Exception:
        pass


class LoaderDataset:
    def __init__(self, dataset, out):
        self.dataset, self.out = dataset, str(out)
        self.adapter = None
        self.failed = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ds = self.dataset
        if self.failed or disabled(self.out):
            return ds[index]
        started = time.perf_counter()
        try:
            import compact_infer as ci
            if self.adapter is None:
                self.adapter = shared.SharedR6(ds, ci)
            with deadline(min(remaining(), CALL_TIMEOUT_SECONDS)):
                parent, k1, detail = self.adapter[index]
            parent['_k1sr'] = dict(item=k1, detail=detail)
            return parent
        except Exception as exc:
            self.failed = True
            disable(self.out, exc)
            # SharedR6 restores hooks/forced-side in finally. Parent keeps its
            # original fallback policy even when only K1's capture failed.
            parent = ds[index]
            parent['_k1sr'] = dict(error=type(exc).__name__ + ': ' + str(exc)[:300],
                                  failed_capture_seconds=time.perf_counter() - started)
            return parent


def load_classifier(device):
    assets = util.mount('knee-l1-rcof0-v1-assets', 'asset-manifest.json')
    w.assets_check(assets, MANIFEST_SHA)  # assets only; never DICOM headers
    name = 'k1sr_frozen_trainer'
    spec = importlib.util.spec_from_file_location(name, assets / 'source/compact_train.py')
    ct = importlib.util.module_from_spec(spec)
    sys.modules[name] = ct
    spec.loader.exec_module(ct)
    w.need(ct.LABELS == w.LABELS, 'K1 target order')
    return w.load_model(ct, assets / 'weights/E_prior_s0_f0.pt', device)


@contextlib.contextmanager
def precision():
    """K1's TF32 contract, restored before the next parent model call."""
    import torch
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.backends.cudnn.benchmark, torch.get_float32_matmul_precision())
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(old[3])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[:2]
        torch.backends.cudnn.benchmark = old[2]


class Owner:
    def __init__(self, out, device):
        self.out, self.device, self.model = Path(out), str(device), None
        self.failed = False

    def handle(self, item, parent_handle):
        captured = item.pop('_k1sr', None)
        t = time.perf_counter()
        parent_handle(item)  # exact parent first, no model in loader process
        rec = dict(uid=item['uid'], parent_classify_seconds=time.perf_counter() - t,
                   parent_render_seconds=item.get('seconds'), ok=False,
                   classifier_seconds=0., load_seconds=0., extra_render_seconds=0.)
        started = time.perf_counter()
        try:
            if self.failed or disabled(self.out):
                raise TimeoutError('K1 disabled/8h guard')
            w.need(captured is not None and 'error' not in captured, 'K1 capture unavailable')
            rec.update(captured['detail'])
            k1 = captured['item']
            x, meta = w.validate_item(k1, item['uid'])
            w.need(self.device.startswith('cuda'), 'release K1 requires CUDA FP16')
            with deadline(min(remaining(), CALL_TIMEOUT_SECONDS)), w.offline(), precision():
                if self.model is None:
                    t = time.perf_counter()
                    self.model = load_classifier(self.device)
                    rec['load_seconds'] = time.perf_counter() - t
                t = time.perf_counter()
                prediction = shared.predict(self.model, x, meta, self.device, fp16=True)
                rec['classifier_seconds'] = time.perf_counter() - t
            if remaining() <= 0:
                raise TimeoutError('K1 completed after guard')
            rec.update(ok=True, prediction=prediction.tolist(), windows=len(x),
                       finished_elapsed_seconds=time.time()-float(os.environ['K1SR_STARTED']))
        except Exception as exc:
            self.failed = True
            rec['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
            disable(self.out, rec['error'])
            self.model = None
            # Reclaim a recoverable K1 OOM without changing parent model state.
            with contextlib.suppress(Exception):
                import torch
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
        rec['k1_owner_seconds'] = time.perf_counter() - started
        if captured and 'failed_capture_seconds' in captured:
            rec['failed_capture_seconds'] = captured['failed_capture_seconds']
        try:
            with (self.out / 'k1-studies.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(rec, allow_nan=False) + '\n')
        except Exception as exc:
            self.failed = True
            disable(self.out, exc)


def patch_main(source):
    """Three exact hooks; CStudies and every parent model/math byte unchanged."""
    edits = [
        ('    pos = 0\n', '    _k1_owner = _k1sr.Owner(out, str(dev))\n    pos = 0\n'),
        ('        loader = torch.utils.data.DataLoader(ds,',
         '        ds = _k1sr.LoaderDataset(ds, out)\n        loader = torch.utils.data.DataLoader(ds,'),
        ('                handle(item)\n', '                _k1_owner.handle(item, handle)\n'),
    ]
    for before, after in edits:
        w.need(source.count(before) == 1, 'K1 parent main hook drift')
        source = source.replace(before, after)
    return source


def worker_main():
    import compact_infer as ci
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    original_main = ci.main
    try:
        w.need(w.sha(ci.__file__) == shared.R6_RUNTIME_SHA, 'R6 runtime source drift')
        patched = patch_main(inspect.getsource(ci.main))
        ci._k1sr = sys.modules[__name__]
        # Renderer identity and executed main are recorded separately.
        output.mkdir(parents=True, exist_ok=True)
        (output / 'k1sr-executed-main.py').write_text(patched, encoding='utf-8')
        w.write(output / 'k1sr-source.json', dict(renderer_sha256=w.sha(ci.__file__),
            executed_main_sha256=hashlib.sha256(patched.encode()).hexdigest(), runtime_sha256=w.sha(__file__)))
        exec(compile(patched, str(output / 'k1sr-executed-main.py'), 'exec'), ci.__dict__)
    except Exception as exc:
        disable(output, exc)
        ci.main = original_main
    ci.main()


def install(runtime_source):
    """Called by R6 port after its unchanged diagnostic_source transformation."""
    entry = Path(runtime_source) / 'k1sr_worker.py'
    try:
        entry.write_text('from tools.k1sr_runtime_v1 import worker_main\n'
                         'if __name__ == "__main__":\n    worker_main()\n', encoding='utf-8')
    except Exception as exc:
        disable(Path(runtime_source).parent / 'controller/worker0', exc)
        return 'compact_infer.py'
    return entry.name


def blend(parent_payload, records):
    import numpy as np
    ids, parent = util.table(parent_payload)
    w.need(len(records) == len(ids) and all(r.get('ok') is True for r in records), 'incomplete K1')
    lookup = {r['uid']: r for r in records}
    w.need(len(lookup) == len(ids) and set(lookup) == set(ids), 'K1 roster mismatch')
    pred = np.asarray([lookup[uid]['prediction'] for uid in ids], dtype=np.float32).astype(np.float64)
    w.need(pred.shape == parent.shape and np.isfinite(pred).all() and
           ((pred >= 0) & (pred <= 1)).all(), 'K1 probabilities')
    result = util.ranks(PARENT_WEIGHT * util.ranks(parent) + K1_WEIGHT * util.ranks(pred)).astype(np.float32)
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow([w.UID] + w.LABELS)
    writer.writerows([uid, *[format(float(x), '.9g') for x in row]] for uid, row in zip(ids, result))
    return buffer.getvalue().encode()


def finish(output, parent_run, config, parent_seconds):
    """Publish only a complete blend; every K1 error keeps exact parent bytes."""
    output, parent_run = Path(output), Path(parent_run)
    w.write(output / 'k1sr-receipt.json', dict(status='COMPARING', k1_executed=False))
    parent = (output / 'submission.csv').read_bytes()
    util.write_once(output / 'k1sr-parent.csv', parent)
    rec = dict(schema='k1-shared-r6-release/v1', version=VERSION, **config,
        status='DEGRADED_PARENT_ONLY', k1_executed=False, k1_studies=0, k1_seconds=0.,
        parent_seconds=parent_seconds, parent_seconds_scope='parent pipeline wall including in-pass K1',
        fp16=True, guard_seconds=GUARD_SECONDS, call_timeout_seconds=CALL_TIMEOUT_SECONDS,
        parent_sha256=hashlib.sha256(parent).hexdigest(), phases={}, per_study=[],
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    payload = parent
    try:
        rows = []
        for path in sorted((parent_run / 'pr7/pr7-run/s_r7/controller').glob('worker*/k1-studies.jsonl')):
            rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
        rec['per_study'] = rows
        rec['k1_studies'] = sum(r.get('ok') is True for r in rows)
        for key in ('load_seconds', 'classifier_seconds', 'extra_render_seconds', 'k1_owner_seconds',
                    'failed_capture_seconds', 'parent_classify_seconds'):
            rec['phases'][key] = sum(float(r.get(key, 0.)) for r in rows)
        rec['k1_seconds'] = sum(rec['phases'][k] for k in
            ('k1_owner_seconds', 'extra_render_seconds', 'failed_capture_seconds'))
        for path in sorted(parent_run.glob('**/*-exit.json')):
            rec['phases'][path.relative_to(parent_run).as_posix()] = json.loads(path.read_bytes())
        w.need(not (parent_run / 'pr7/pr7-run/s_r7/controller/k1-disabled.json').exists(), 'K1 disabled')
        w.need(all(0 <= r['finished_elapsed_seconds'] <= GUARD_SECONDS for r in rows), 'K1 guard completion')
        payload = blend(parent, rows)
        rec.update(status='COMPLETE_SHARED_K1', k1_executed=True)
    except Exception as exc:
        rec['error'] = type(exc).__name__ + ': ' + str(exc)[:400]
        payload = parent
    rec['submission_sha256'] = hashlib.sha256(payload).hexdigest()
    if config.get('diagnostic'):
        rec['projection'] = dict(studies=1300, method='full N32 wall/32*1300 including startup (conservative)',
            parent_pipeline_wall_seconds_per_study=parent_seconds / 32,
            full_wall_hours=parent_seconds / 32 * 1300 / 3600,
            never_submit=True)
    # Invalidate before publishing so an interrupted overlay cannot retain PASS.
    w.write(output / 'k1sr-receipt.json', dict(status='PUBLISHING', k1_executed=False))
    util.publish(output / 'submission.csv', payload)
    w.write(output / 'k1sr-receipt.json', rec)
    return rec
