"""K1 shared-R6 three-checkpoint release runtime v6.0.0.

Changelog: one study input transfer shared by three models; explicit chunk 32.
Transfer and per-model classification timing; unchanged load/classify/global guards.
Previous: rank-mean three FP16 classifiers on shared windows; all-or-parent fallback.
The v1 runtime remains untouched; rendering and parent serving are unchanged.
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

VERSION = '6.0.0'
MODEL_IDS = ('k1_v2', 'k1_v3g_seed0', 'k1_v3g_seed1')
SEED0_MODEL_SHA = '6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5'
SEED0_MANIFEST_SHA = 'b09620471faa361f55e187a5ef188a9768be6c7e238afb69c87436309c140d77'
SEED0_SOURCE_SHA = '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80'
SEED1_MODEL_SHA = '29fc2b2bbaa4e4e8ad341bba17e47c5110388db697458a8cbb9ff8d83fa9399b'
SEED1_MANIFEST_SHA = '154d5c9ae21e69e1b22ec59bed8fb31dbabe5319c36ed38ecb142498156fd6e6'
SEED1_SOURCE_SHA = '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80'
GUARD_SECONDS = 8 * 3600
CALL_TIMEOUT_SECONDS = 120
LOAD_TIMEOUT_SECONDS = 1500
PARENT_WEIGHT, K1_WEIGHT = .80, .20
MANIFEST_SHA = util.MANIFEST_SHA
ENCODER_CHUNK = 32  # Registered release constant, never selected from environment.
BENCHMARK_CHUNKS = (32, 64)  # 64 requires full N32 FP16 parity and separate registration.


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


@contextlib.contextmanager
def phase_deadline(seconds, phase):
    """Separate load/classify budgets, both bounded by the global guard."""
    budget = min(remaining(), seconds)
    started = time.perf_counter()
    with deadline(budget):
        yield
    # Also reject late native returns and enforce CPU Windows verification.
    if time.perf_counter() - started > budget or remaining() <= 0:
        raise TimeoutError('K1 ' + phase + ' deadline')


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


def load_classifier(device, model_id):
    """Strict checkpoint/source identity before deserialization; identical Net."""
    w.need(model_id in MODEL_IDS, 'unknown K1 model')
    if model_id == MODEL_IDS[0]:
        assets = util.mount('knee-l1-rcof0-v1-assets', 'asset-manifest.json')
        w.assets_check(assets, MANIFEST_SHA)
        model_sha = util.MODEL_SHA
    else:
        seed0 = model_id == 'k1_v3g_seed0'
        model_pin = SEED0_MODEL_SHA if seed0 else SEED1_MODEL_SHA
        manifest_pin = SEED0_MANIFEST_SHA if seed0 else SEED1_MANIFEST_SHA
        source_pin = SEED0_SOURCE_SHA if seed0 else SEED1_SOURCE_SHA
        assets = util.mount('knee-k1-v3gs0-v1-assets' if seed0 else 'knee-k1-v3gs1-v1-assets', 'asset-manifest.json')
        w.need(w.sha(assets / 'asset-manifest.json') == manifest_pin, 'v3G manifest pin')
        manifest = w.read(assets / 'asset-manifest.json')
        expected = {'weights/E_prior_s0_f0.pt': model_pin,
                    'source/compact_train.py': source_pin}
        w.need(manifest['files'] == expected, 'v3G inventory pin')
        files = {p.relative_to(assets).as_posix() for p in assets.rglob('*') if p.is_file()}
        w.need(files - {'asset-manifest.json', 'dataset-metadata.json'} == set(expected), 'v3G inventory')
        for name, pin in expected.items():
            path = assets / name
            w.need(path.resolve().is_relative_to(assets.resolve()) and not path.is_symlink(), 'unsafe v3G asset')
            w.need(w.sha(path) == pin, 'v3G file identity')
        model_sha = model_pin
    name = 'k1x3sr_frozen_trainer_' + model_id
    spec = importlib.util.spec_from_file_location(name, assets / 'source/compact_train.py')
    ct = importlib.util.module_from_spec(spec)
    sys.modules[name] = ct
    spec.loader.exec_module(ct)
    w.need(ct.LABELS == w.LABELS, 'K1 target order')
    import torch
    checkpoint = assets / 'weights/E_prior_s0_f0.pt'
    w.need(w.sha(checkpoint) == model_sha, 'checkpoint SHA256 before deserialization')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ct.Net('convnext_small', None, crop=352, in_chans=3,
                   spatial=True, extent=False, state_spec=None)
    expected = model.state_dict()
    w.need(set(state) == set(expected), 'deploy state key coverage')
    w.need(all(state[k].shape == expected[k].shape and state[k].dtype == expected[k].dtype for k in state),
           'deploy state shape/dtype coverage')
    w.need(all(torch.isfinite(v).all().item() for v in state.values()), 'nonfinite checkpoint')
    model.load_state_dict(state, strict=True)
    return model.float().eval().requires_grad_(False).to(device)


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


def transfer_inputs(x, meta, device):
    """Keep the v5 crop/dtypes/layout; transfer each study exactly once.

    CPU is supported only for explicit engineering replay, never Owner serving.
    Blocking transfers preserve v5 stream semantics; no pin/pool/launch changes.
    """
    import torch
    mask = torch.ones((1, len(x)), dtype=torch.bool, device=device)
    return x[None, ..., 16:368, 16:368].to(device), meta[None].to(device), mask


def predict_prepared(model, prepared, *, chunk=ENCODER_CHUNK, fp16=True):
    """Same v5 forward and sigmoid, reusing device inputs across models."""
    import torch
    w.need(type(chunk) is int and chunk in BENCHMARK_CHUNKS, 'unregistered encoder chunk')
    x, meta, mask = prepared
    device_type = x.device.type
    w.need(not fp16 or device_type == 'cuda', 'FP16 experiment requires CUDA')
    with torch.inference_mode(), torch.autocast(device_type, dtype=torch.float16, enabled=fp16):
        logits = model(x, meta, mask, chunk=chunk)
    w.need(tuple(logits.shape) == (1, 12) and torch.isfinite(logits).all().item(), 'invalid FP16 logits')
    return logits.float().sigmoid().cpu().numpy()[0]


class Owner:
    def __init__(self, out, device):
        self.out, self.device, self.models = Path(out), str(device), {}
        self.failed = False

    def handle(self, item, parent_handle):
        captured = item.pop('_k1sr', None)
        t = time.perf_counter()
        parent_handle(item)  # exact parent first, no model in loader process
        rec = dict(uid=item['uid'], parent_classify_seconds=time.perf_counter() - t,
                   parent_render_seconds=item.get('seconds'), ok=False,
                   classifier_seconds=0., transfer_seconds=0., encoder_chunk=ENCODER_CHUNK,
                   load_seconds=0., extra_render_seconds=0.,
                   load_timeout_seconds=LOAD_TIMEOUT_SECONDS, call_timeout_seconds=CALL_TIMEOUT_SECONDS,
                   k1_models=list(MODEL_IDS), per_model_seconds={m: dict(load=0., classify=0.) for m in MODEL_IDS})
        started = time.perf_counter()
        prepared = None
        try:
            if self.failed or disabled(self.out):
                raise TimeoutError('K1 disabled/8h guard')
            w.need(captured is not None and 'error' not in captured, 'K1 capture unavailable')
            rec.update(captured['detail'])
            k1 = captured['item']
            x, meta = w.validate_item(k1, item['uid'])
            w.need(self.device.startswith('cuda'), 'release K1 requires CUDA FP16')
            with w.offline(), precision():
                predictions = {}
                for model_id in MODEL_IDS:
                    if model_id not in self.models:
                        t = time.perf_counter()
                        try:
                            with phase_deadline(LOAD_TIMEOUT_SECONDS, "load " + model_id):
                                self.models[model_id] = load_classifier(self.device, model_id)
                        finally:
                            elapsed = time.perf_counter() - t
                            rec['per_model_seconds'][model_id]['load'] = elapsed
                            rec['load_seconds'] += elapsed
                    if prepared is None:
                        t = time.perf_counter()
                        try:
                            with phase_deadline(CALL_TIMEOUT_SECONDS, "transfer"):
                                prepared = transfer_inputs(x, meta, self.device)
                        finally:
                            rec['transfer_seconds'] = time.perf_counter() - t
                            rec['classifier_seconds'] += rec['transfer_seconds']
                    t = time.perf_counter()
                    try:
                        with phase_deadline(CALL_TIMEOUT_SECONDS - rec["classifier_seconds"], "classify"):
                            prediction = predict_prepared(self.models[model_id], prepared, chunk=ENCODER_CHUNK, fp16=True)
                            import numpy as np
                            w.need(prediction.shape == (12,) and np.isfinite(prediction).all() and
                                   ((prediction >= 0) & (prediction <= 1)).all(), 'invalid K1 model output')
                            predictions[model_id] = prediction.tolist()
                    finally:
                        elapsed = time.perf_counter() - t
                        rec['per_model_seconds'][model_id]['classify'] = elapsed
                        rec['classifier_seconds'] += elapsed
            if remaining() <= 0:
                raise TimeoutError('K1 completed after guard')
            rec.update(ok=True, predictions=predictions, windows=len(x),
                       finished_elapsed_seconds=time.time()-float(os.environ['K1SR_STARTED']))
        except Exception as exc:
            self.failed = True
            rec['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
            disable(self.out, rec['error'])
            prepared = None
            self.models.clear()
            # Reclaim a recoverable K1 OOM without changing parent model state.
            with contextlib.suppress(Exception):
                import torch
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
        prepared = None  # Do not retain study tensors across parent calls.
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
        entry.write_text('from tools.k1sr_runtime_v6 import worker_main\n'
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
    ranks = []
    for model_id in MODEL_IDS:
        w.need(all(r.get('k1_models') == list(MODEL_IDS) and
                   set(r.get('predictions', {})) == set(MODEL_IDS) for r in records), 'all three K1 models required')
        pred = np.asarray([lookup[uid]['predictions'][model_id] for uid in ids], dtype=np.float32).astype(np.float64)
        w.need(pred.shape == parent.shape and np.isfinite(pred).all() and
               ((pred >= 0) & (pred <= 1)).all(), 'K1 probabilities')
        ranks.append(util.ranks(pred))
    k1_rank_mean = np.mean(np.stack(ranks), axis=0)
    # Keep the v1 outer blend: rank the combined K1 before weighting.
    result = util.ranks(PARENT_WEIGHT * util.ranks(parent) +
                        K1_WEIGHT * util.ranks(k1_rank_mean)).astype(np.float32)
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow([w.UID] + w.LABELS)
    writer.writerows([uid, *[format(float(x), '.9g') for x in row]] for uid, row in zip(ids, result))
    return buffer.getvalue().encode()


def finish(output, parent_run, config, parent_seconds):
    """Publish only a complete blend; every K1 error keeps exact parent bytes."""
    output, parent_run = Path(output), Path(parent_run)
    w.write(output / 'k1sr-receipt.json', dict(status='COMPARING', k1_executed=False, diagnostic=bool(config.get('diagnostic')), never_submit=bool(config.get('diagnostic'))))
    diagnostic = bool(config.get('diagnostic'))
    if diagnostic:
        w.need(config.get('never_submit') is True, 'diagnostic must never submit')
        w.need(not (output / 'submission.csv').exists(), 'diagnostic submission path forbidden')
    parent = (output / ('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).read_bytes()
    util.write_once(output / 'k1sr-parent.csv', parent)
    rec = dict(schema='k1-shared-r6-release/v6', version=VERSION, **config,
        status='DEGRADED_PARENT_ONLY', k1_executed=False, k1_studies=0, k1_seconds=0.,
        parent_seconds=parent_seconds, parent_seconds_scope='parent pipeline wall including in-pass K1',
        fp16=True, guard_seconds=GUARD_SECONDS, call_timeout_seconds=CALL_TIMEOUT_SECONDS,
        load_timeout_seconds=LOAD_TIMEOUT_SECONDS, call_timeout_scope="transfer + all classifiers per study; excludes loading",
        parent_sha256=hashlib.sha256(parent).hexdigest(), phases={}, per_study=[],
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    rec['encoder_chunk'] = ENCODER_CHUNK
    rec['k1_models'] = list(MODEL_IDS)
    rec['per_model_seconds'] = {m: dict(load=0., classify=0.) for m in MODEL_IDS}
    payload = parent
    try:
        rows = []
        for path in sorted((parent_run / 'pr7/pr7-run/s_r7/controller').glob('worker*/k1-studies.jsonl')):
            rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
        rec['per_study'] = rows
        rec['k1_studies'] = sum(r.get('ok') is True for r in rows)
        rec['k1_models'] = list(MODEL_IDS)
        rec['per_model_seconds'] = {m: {phase: sum(float(r.get('per_model_seconds', {}).get(m, {}).get(phase, 0.))
            for r in rows) for phase in ('load', 'classify')} for m in MODEL_IDS}
        for key in ('load_seconds', 'transfer_seconds', 'classifier_seconds', 'extra_render_seconds', 'k1_owner_seconds',
                    'failed_capture_seconds', 'parent_classify_seconds'):
            rec['phases'][key] = sum(float(r.get(key, 0.)) for r in rows)
        rec['k1_seconds'] = sum(rec['phases'][k] for k in
            ('k1_owner_seconds', 'extra_render_seconds', 'failed_capture_seconds'))
        for path in sorted(parent_run.glob('**/*-exit.json')):
            rec['phases'][path.relative_to(parent_run).as_posix()] = json.loads(path.read_bytes())
        w.need(not (parent_run / 'pr7/pr7-run/s_r7/controller/k1-disabled.json').exists(), 'K1 disabled')
        w.need(all(0 <= r['finished_elapsed_seconds'] <= GUARD_SECONDS for r in rows), 'K1 guard completion')
        payload = blend(parent, rows)
        rec.update(status='COMPLETE_SHARED_K1X3', k1_executed=True)
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
    w.write(output / 'k1sr-receipt.json', dict(status='PUBLISHING', k1_executed=False, diagnostic=diagnostic, never_submit=diagnostic))
    if diagnostic:
        rec.update(diagnostic=True, never_submit=True, qualified_for_submission=False,
                   diagnostic_sha256=rec['submission_sha256'], submission_sha256=None)
    util.publish(output / ('diagnostic-k1x3.csv' if diagnostic else 'submission.csv'), payload)
    w.write(output / 'k1sr-receipt.json', rec)
    return rec
