"""K1 N-model shared-R6 release runtime v10.0.0.

Changelog v10.0.0: separate capture 900 s and transfer/classifier 300 s deadlines.

Changelog v9.0.0: remove finish-only deadline check; retain row completion guard.

Changelog v8.0.0: study-local fallback; systemic/fraction failures raise only
after final parent CSV and receipt publication. v7 remains immutable.
Changelog v7.2.0: registered per-target outer blend; scalar math unchanged.
Changelog v7.1.0: registered stride 2/3 raw-row windows, receipt metadata.
Changelog v7.0.0: registered 1..6 classifier identities, single input transfer, optional
P-only parent. Prior guards, FP16 forward, rendering and failure policy retained.
The generator bakes REGISTERED into each immutable payload; no env override.
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
from tools import k1_shared_render_v2 as shared
from tools import l1_rco_k2_worker_v102 as w
from tools import k1n_config_v1 as config_schema

VERSION = '10.0.0'
REGISTERED = dict(config_schema.current(), k1_failure_policy='study_fallback_fail_loud', k1_runtime_version=10)
REGISTERED = config_schema.validate(REGISTERED)
POLICY = REGISTERED['k1_failure_policy']
MAX_FAILED_FRACTION = REGISTERED['k1_max_failed_fraction']
MODEL_IDS = tuple(m['id'] for m in REGISTERED['models'])
MODEL_SPECS = {m['id']: m for m in REGISTERED['models']}
CONFIG_SHA = hashlib.sha256(util.encoded(REGISTERED)).hexdigest()
GUARD_SECONDS = REGISTERED.get('k1_guard_hours', 8.0) * 3600
CAPTURE_TIMEOUT_SECONDS = 900
CALL_TIMEOUT_SECONDS = 300
LOAD_TIMEOUT_SECONDS = 1500
K1_WEIGHT = (tuple(REGISTERED['alpha_vector']) if 'alpha_vector' in REGISTERED
             else REGISTERED['alpha'])
PARENT_WEIGHT = (tuple(1.0 - a for a in K1_WEIGHT) if isinstance(K1_WEIGHT, tuple)
                 else 1.0 - K1_WEIGHT)
MANIFEST_SHA = util.MANIFEST_SHA
ENCODER_CHUNK = REGISTERED['encoder_chunk']
K1_EVAL_STRIDE = REGISTERED['k1_eval_stride']
BENCHMARK_CHUNKS = (32,)  # 64 requires full N32 FP16 parity and separate registration.


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
                self.adapter = shared.SharedR6(ds, ci, eval_stride=K1_EVAL_STRIDE)
            with deadline(min(remaining(), CAPTURE_TIMEOUT_SECONDS)):
                parent, k1, detail = self.adapter[index]
            parent['_k1sr'] = dict(item=k1, detail=detail)
            return parent
        except Exception as exc:
            # Capture failure belongs to this study. A later study still uses
            # the same adapter; only a global guard disables the worker branch.
            if remaining() <= 0:
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
    identity = MODEL_SPECS[model_id]
    assets = util.mount(identity['dataset_slug'].split('/')[1], 'asset-manifest.json')
    manifest_pin = identity['manifest_sha256']
    if manifest_pin == util.MANIFEST_SHA:
        w.assets_check(assets, manifest_pin)
    else:
        w.need(w.sha(assets / 'asset-manifest.json') == manifest_pin, 'K1 manifest pin')
        manifest = w.read(assets / 'asset-manifest.json')
        expected = {identity['checkpoint']: identity['checkpoint_sha256'],
                    identity['trainer_source']: identity['trainer_sha256']}
        w.need(manifest['files'] == expected, 'K1 inventory pin')
        files = {p.relative_to(assets).as_posix() for p in assets.rglob('*') if p.is_file()}
        w.need(files - {'asset-manifest.json', 'dataset-metadata.json'} == set(expected), 'K1 inventory')
        for name, pin in expected.items():
            path = assets / name
            w.need(path.resolve().is_relative_to(assets.resolve()) and not path.is_symlink(), 'unsafe K1 asset')
            w.need(w.sha(path) == pin, 'K1 file identity')
    model_sha = identity['checkpoint_sha256']
    trainer = assets / identity['trainer_source']
    w.need(w.sha(trainer) == identity['trainer_sha256'], 'trainer source pin')
    name = 'k1nsr_frozen_trainer_' + model_id
    spec = importlib.util.spec_from_file_location(name, trainer)
    ct = importlib.util.module_from_spec(spec)
    sys.modules[name] = ct
    spec.loader.exec_module(ct)
    w.need(ct.LABELS == w.LABELS, 'K1 target order')
    import torch
    checkpoint = assets / identity['checkpoint']
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
        rec = dict(uid=item['uid'], config_sha256=CONFIG_SHA, model_identities=REGISTERED['models'], parent_classify_seconds=time.perf_counter() - t,
                   parent_render_seconds=item.get('seconds'), ok=False,
                   classifier_seconds=0., transfer_seconds=0., encoder_chunk=ENCODER_CHUNK,
                   k1_eval_stride=K1_EVAL_STRIDE,
                   load_seconds=0., extra_render_seconds=0.,
                   load_timeout_seconds=LOAD_TIMEOUT_SECONDS, call_timeout_seconds=CALL_TIMEOUT_SECONDS,
                   k1_models=list(MODEL_IDS), per_model_seconds={m: dict(load=0., classify=0.) for m in MODEL_IDS})
        started = time.perf_counter()
        prepared = None
        phase = 'capture'
        try:
            if self.failed or disabled(self.out):
                raise TimeoutError('K1 disabled/configured wall-clock guard')
            w.need(captured is not None and 'error' not in captured, 'K1 capture unavailable')
            rec.update(captured['detail'])
            k1 = captured['item']
            x, meta = w.validate_item(k1, item['uid'])
            w.need(self.device.startswith('cuda'), 'release K1 requires CUDA FP16')
            with w.offline(), precision():
                predictions = {}
                for model_id in MODEL_IDS:
                    if model_id not in self.models:
                        phase = 'load'
                        t = time.perf_counter()
                        try:
                            with phase_deadline(LOAD_TIMEOUT_SECONDS, "load " + model_id):
                                self.models[model_id] = load_classifier(self.device, model_id)
                        finally:
                            elapsed = time.perf_counter() - t
                            rec['per_model_seconds'][model_id]['load'] = elapsed
                            rec['load_seconds'] += elapsed
                    phase = 'study'
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
            rec['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
            rec['systemic_failure'] = phase == 'load' or self.failed or disabled(self.out)
            if rec['systemic_failure']:
                self.failed = True
                disable(self.out, rec['error'])
                self.models.clear()
            prepared = None
            # Reclaim a recoverable K1 OOM without changing parent model state.
            with contextlib.suppress(Exception):
                import torch
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
        prepared = None  # Do not retain study tensors across parent calls.
        rec['finished_elapsed_seconds'] = time.time()-float(os.environ['K1SR_STARTED'])
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
        entry.write_text('from tools.k1sr_runtime_v10 import worker_main\n'
                         'if __name__ == "__main__":\n    worker_main()\n', encoding='utf-8')
    except Exception as exc:
        disable(Path(runtime_source).parent / 'controller/worker0', exc)
        return 'compact_infer.py'
    return entry.name


def blend(parent_payload, records):
    import numpy as np
    ids, parent = util.table(parent_payload)
    w.need(len(records) == len(ids), 'incomplete K1 telemetry')
    lookup = {r['uid']: r for r in records}
    w.need(len(lookup) == len(ids) and set(lookup) == set(ids), 'K1 roster mismatch')
    w.need(all(type(r.get('ok')) is bool and r.get('config_sha256') == CONFIG_SHA
               and r.get('k1_models') == list(MODEL_IDS) for r in records), 'K1 record identity')
    failed = np.asarray([not lookup[uid]['ok'] for uid in ids])
    w.need((~failed).any(), 'no study executed K1')
    ranks = []
    for model_id in MODEL_IDS:
        w.need(all(set(r.get('predictions', {})) == set(MODEL_IDS)
                   for r in records if r['ok']), 'all registered K1 models required')
        # Preserve the full roster rank denominator; no invented K1 probability.
        pred = np.asarray([parent[i] if failed[i] else lookup[uid]['predictions'][model_id]
                           for i, uid in enumerate(ids)], dtype=np.float32).astype(np.float64)
        w.need(pred.shape == parent.shape and np.isfinite(pred).all() and
               ((pred >= 0) & (pred <= 1)).all(), 'K1 probabilities')
        ranks.append(util.ranks(pred))
    k1_rank_mean = np.mean(np.stack(ranks), axis=0)
    parent_rank = util.ranks(parent)
    k1_rank = util.ranks(k1_rank_mean)
    # Substitute AFTER ranking: re-ranking the substituted values here would
    # destroy the parent-only contribution for a failed study. The unchanged
    # final outer rank may still move its percentile relative to other studies.
    k1_rank[failed] = parent_rank[failed]
    weighted = np.asarray(PARENT_WEIGHT) * parent_rank + np.asarray(K1_WEIGHT) * k1_rank
    weighted[failed] = parent_rank[failed]  # exact, including floating-point ties
    result = util.ranks(weighted).astype(np.float32)
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow([w.UID] + w.LABELS)
    writer.writerows([uid, *[format(float(x), '.9g') for x in row]] for uid, row in zip(ids, result))
    return buffer.getvalue().encode()


def finish(output, parent_run, config, parent_seconds):
    """Publish once, then raise on systemic failure; never silently score parent."""
    output, parent_run = Path(output), Path(parent_run)
    w.write(output / 'k1sr-receipt.json', dict(status='COMPARING', k1_executed=False, diagnostic=bool(config.get('diagnostic')), never_submit=bool(config.get('diagnostic'))))
    diagnostic = bool(config.get('diagnostic'))
    if diagnostic:
        w.need(config.get('never_submit') is True, 'diagnostic must never submit')
        w.need(not (output / 'submission.csv').exists(), 'diagnostic submission path forbidden')
    parent = (output / ('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).read_bytes()
    util.write_once(output / 'k1sr-parent.csv', parent)
    rec = dict(schema='k1-shared-r6-release/v10', version=VERSION, **config,
        status='DEGRADED_PARENT_ONLY', k1_executed=False, k1_studies=0, k1_seconds=0.,
        parent_seconds=parent_seconds, parent_seconds_scope='parent pipeline wall including in-pass K1',
        fp16=True, guard_seconds=GUARD_SECONDS, call_timeout_seconds=CALL_TIMEOUT_SECONDS,
        capture_timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
        load_timeout_seconds=LOAD_TIMEOUT_SECONDS, call_timeout_scope="transfer + all classifiers per study; excludes loading",
        parent_sha256=hashlib.sha256(parent).hexdigest(), phases={}, per_study=[],
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    rec['encoder_chunk'] = ENCODER_CHUNK
    rec['k1_eval_stride'] = K1_EVAL_STRIDE
    rec['config_sha256'] = CONFIG_SHA
    rec['model_identities'] = REGISTERED['models']
    rec.update(k1_failure_policy=POLICY, k1_max_failed_fraction=MAX_FAILED_FRACTION,
               failed_study_count=0, failed_study_fraction=0., fail_loud_triggered=False)
    if 'alpha_vector' in REGISTERED:
        rec['alpha_vector'] = REGISTERED['alpha_vector']
        rec['alpha_targets'] = util.LABELS
    rec['parent_mode'] = 'INF3_P_R224' if REGISTERED['drop_o'] else 'INF3'
    rec['k1_models'] = list(MODEL_IDS)
    rec['per_model_seconds'] = {m: dict(load=0., classify=0.) for m in MODEL_IDS}
    payload = parent
    ids, _ = util.table(parent)
    rec.update(failed_study_count=len(ids), failed_study_fraction=1.)
    try:
        w.need(config.get('config_sha256') == CONFIG_SHA, 'finish registration mismatch')
        rows = []
        for path in sorted((parent_run / 'pr7/pr7-run/s_r7/controller').glob('worker*/k1-studies.jsonl')):
            rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
        rec['per_study'] = rows
        rec['k1_studies'] = sum(r.get('ok') is True for r in rows)
        successful = {r['uid'] for r in rows if r.get('ok') is True}
        failed_count = len(set(ids) - successful)
        rec.update(failed_study_count=failed_count, failed_study_fraction=failed_count / len(ids))
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
        w.need(not any(r.get('systemic_failure') for r in rows), 'K1 systemic failure')
        w.need(all(0 <= r['finished_elapsed_seconds'] <= GUARD_SECONDS for r in rows), 'K1 guard completion')
        w.need(rec['failed_study_fraction'] <= MAX_FAILED_FRACTION, 'K1 failed-study fraction exceeds registered limit')
        payload = blend(parent, rows)
        rec.update(status='COMPLETE_SHARED_K1N', k1_executed=True)
    except Exception as exc:
        rec['error'] = type(exc).__name__ + ': ' + str(exc)[:400]
        rec.update(status='FAILED_K1_FAIL_LOUD', fail_loud_triggered=True)
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
    util.publish(output / ('diagnostic-k1n.csv' if diagnostic else 'submission.csv'), payload)
    w.write(output / 'k1sr-receipt.json', rec)
    if rec['fail_loud_triggered']:
        raise RuntimeError('K1 fail-loud: ' + rec['error'] + '; parent CSV and receipt preserved')
    return rec
