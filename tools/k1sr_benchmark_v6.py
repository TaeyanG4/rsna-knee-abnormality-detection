"""H-B-221 v1.0.0: local N32 v5/v6 paired runtime benchmark, no labels/network.

Changelog: balanced arm order, synchronized phase timing and per-arm memory;
one frozen roster, three exact checkpoints, fixed FP16 gates. B launches GPU.
Chunk 64 is an experiment only: this script never changes a release constant.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

from tools import k1sr_runtime_v5 as v5
from tools import k1sr_runtime_v6 as v6
from tools import k1_shared_render_harness_v1 as h

w = v6.w
OUT = h.ROOT / '.local/scratch/astra-b1/h-b-221'
ARMS = ('v5', 'v6_chunk32', 'v6_chunk64')


def identity():
    return dict(sources={Path(p).name: w.sha(p) for p in
        (__file__, v5.__file__, v6.__file__, h.__file__, v6.shared.__file__, w.__file__)},
        selection_sha256=w.sha(h.OUT / 'selection.private.json'),
        model_sha256=dict(zip(v6.MODEL_IDS, (v6.util.MODEL_SHA, v6.SEED0_MODEL_SHA, v6.SEED1_MODEL_SHA))),
        chunks=[32, 64], studies=32, max_abs_limit=5e-3, min_spearman=.999)


def validate(review=None):
    """CPU-only preflight; does not claim/admit a GPU or touch CUDA."""
    rec = identity()
    selection = w.read(h.OUT / 'selection.private.json')
    w.need(len(selection['ids']) == len(set(selection['ids'])) == 32 and
           set(selection['largest8']) <= set(selection['ids']), 'exact N32 roster required')
    w.need(v6.ENCODER_CHUNK == 32, 'release must remain chunk32')
    if review is not None:
        approval = w.read(review)
        w.need(approval.get('verdict') == 'PASS' and approval.get('builder') == 'astra-b1' and
               approval.get('reviewer') and approval['reviewer'] != 'astra-b1' and
               approval.get('identity') == rec, 'independent pre-execution review missing/stale')
    return dict(status='VALIDATED_CPU_ONLY', identity=rec, independent_review_verified=review is not None,
                qualified_for_submission=False)


@contextlib.contextmanager
def local_assets():
    from tools import k1x3sr_assets_v1 as seed0
    from tools import k1x2sr_assets_v1 as seed1
    mounts = {'knee-l1-rcof0-v1-assets': h.ASSETS,
              'knee-k1-v3gs0-v1-assets': seed0.DEST, 'knee-k1-v3gs1-v1-assets': seed1.DEST}
    with patch.object(v6.util, 'mount', side_effect=lambda slug, marker: mounts[slug]):
        yield


def measure_arm(models, x, meta, arm, device, *, fp16=True, progress=lambda *a: None):
    """CPU FP32 replay uses the same v6 functions, with autocast disabled.

    Models stay resident in every arm. Reset allocator cache outside timing so
    reserved peaks are attributable to the arm, not a previous chunk64 trial.
    v5's per-model timing includes its repeated transfer, as in runtime v5.
    """
    import torch
    w.need(arm in ARMS, 'unknown benchmark arm')
    cuda = device.startswith('cuda')
    w.need(not fp16 or cuda, 'FP16 benchmark requires CUDA')
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else lambda: None
    if cuda:
        sync()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device) if cuda else None
    prepared = None
    row = dict(transfer_seconds=None if arm == 'v5' else 0., per_model_seconds={}, predictions={})
    sync()
    start = time.perf_counter()
    try:
        if arm != 'v5':
            progress('transfer-' + arm)
            t = time.perf_counter()
            prepared = v6.transfer_inputs(x, meta, device)
            sync()
            row['transfer_seconds'] = time.perf_counter() - t
        for mid in v6.MODEL_IDS:
            progress('classify-' + arm + '-' + mid)
            t = time.perf_counter()
            if arm == 'v5':
                pred = v5.shared.predict(models[mid], x, meta, device, fp16=fp16)
            else:
                pred = v6.predict_prepared(models[mid], prepared,
                    chunk=32 if arm == 'v6_chunk32' else 64, fp16=fp16)
            sync()
            row['per_model_seconds'][mid] = time.perf_counter() - t
            row['predictions'][mid] = pred.tolist()
        row.update(total_seconds=time.perf_counter()-start, baseline_allocated_bytes=baseline,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if cuda else None)
        return row
    finally:
        prepared = None


def equality(reference, candidate):
    import numpy as np
    from scipy.stats import spearmanr
    a, b = np.asarray(reference), np.asarray(candidate)
    w.need(a.shape == b.shape == (32, 12) and np.isfinite(a).all() and np.isfinite(b).all(),
           'FP16 gate requires complete finite N32 x 12 outputs')
    delta = float(np.max(np.abs(a-b)))
    correlations = [float(spearmanr(a[:, j], b[:, j]).statistic) for j in range(12)]
    passed = delta <= 5e-3 and all(np.isfinite(r) and r >= .999 for r in correlations)
    return dict(status='PASS' if passed else 'FAIL', max_abs=delta,
                per_target_spearman=[r if np.isfinite(r) else None for r in correlations],
                max_abs_limit=5e-3, min_spearman=.999)


def summarize(records, desktop_used, total_memory):
    import numpy as np
    w.need(len(records) == 32 and len({r['uid'] for r in records}) == 32, 'incomplete/duplicate N32')
    result = {}
    for arm in ARMS:
        timings = [r['arms'][arm]['total_seconds'] for r in records]
        peak = max(r['arms'][arm]['peak_reserved_bytes'] for r in records)
        comparisons = {}
        if arm != 'v5':
            for mid in v6.MODEL_IDS:
                comparisons[mid] = equality([r['arms']['v5']['predictions'][mid] for r in records],
                                           [r['arms'][arm]['predictions'][mid] for r in records])
            def rank_mean(which):
                return np.mean([v6.util.ranks(np.asarray([
                    r['arms'][which]['predictions'][m] for r in records], dtype=np.float32).astype(np.float64))
                    for m in v6.MODEL_IDS], axis=0)
            comparisons['k1_rank_mean'] = equality(rank_mean('v5'), rank_mean(arm))
        result[arm] = dict(mean_seconds=statistics.mean(timings), median_seconds=statistics.median(timings),
            p95_seconds=float(np.percentile(timings, 95)), maximum_seconds=max(timings),
            peak_reserved_bytes=peak,
            peak_allocated_bytes=max(r['arms'][arm]['peak_allocated_bytes'] for r in records),
            memory_gate=peak <= 12 * 2**30 and desktop_used + peak + 2 * 2**30 <= total_memory,
            equality=comparisons, numeric_pass=all(v['status'] == 'PASS' for v in comparisons.values()))
    for arm in ARMS[1:]:
        result[arm]['estimated_1300_study_saving_hours'] = (
            result['v5']['mean_seconds'] - result[arm]['mean_seconds']) * 1300 / 3600
        result[arm]['engineering_gate_pass'] = result[arm]['numeric_pass'] and result[arm]['memory_gate']
    return result


def run(review):
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / 'GPU-BENCHMARK.json'
    w.write(report, dict(status='STARTING', qualified_for_submission=False))
    records, wd = [], None
    try:
        w.need(review is not None, 'independent pre-execution review required')
        checked = validate(review)
        if os.name == 'nt':
            driver = r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
            w.need(w.sha(driver) == '712cde267b942a6dbbea6e860a8a53372ad71ee6aadd94cca6bd1a424ab7e14b', 'Windows DLL pin')
            os.environ['S6PLUS_PIN_DLL'] = driver
            from tools import s6plus
        torch = w.configure('cuda:0')
        free, total = torch.cuda.mem_get_info()
        desktop_used = total-free
        # Keep the latest per-study report if the watchdog must terminate us.
        wd = w.Watchdog(OUT, OUT / 'benchmark-watchdog.json', True)
        selection = w.read(h.OUT / 'selection.private.json')
        for name, crc, size, sha in selection['members']:
            w.need(w.sha(h.OUT / 'raw' / name) == sha, 'raw bytes changed')
            wd.progress('raw-verified')
        w.need(w.sha(h.OUT / 'raw/train_series.csv') == selection['series_sha256'], 'series bytes changed')
        sys.path[:0] = [str(h.runtime_source()), str(h.ASSETS / 'cloud_code')]
        import compact_infer as ci
        with w.offline(), v6.precision(), local_assets():
            w.assets_check(h.ASSETS, h.ASSETS_SHA, wd.progress)
            w.install_label_free_source(wd.progress)
            # Rendering is outside the measured classifier comparison, on CPU.
            ds = ci.CStudies(h.OUT/'raw', 'train', selection['ids'], OUT/'render', h.ASSETS/'cloud_code',
                384, eval_stride=1, views='v4', slices=3, anchor_weights=h.ASSETS/'weights/anatomy.pt',
                phys_mm=3.5, side_model=h.BUNDLE/'side_cls.pt', extent_meta=True, fluid=True, device='cpu')
            adapter = v6.shared.SharedR6(ds, ci)
            models, loads = {}, {}
            for mid in v6.MODEL_IDS:
                wd.progress('load-' + mid)
                start = time.perf_counter()
                models[mid] = v6.load_classifier('cuda:0', mid)
                torch.cuda.synchronize()
                loads[mid] = time.perf_counter()-start
            for i, uid in enumerate(selection['ids']):
                wd.progress('render')
                parent, item, detail = adapter[i]
                del parent
                x, meta = w.validate_item(item, uid)
                if i == 0:
                    for arm in ARMS:
                        measure_arm(models, x, meta, arm, 'cuda:0', progress=wd.progress)
                order = ARMS[i % 3:] + ARMS[:i % 3]
                row = dict(uid=uid, windows=len(x), input_sha256=w.tensor_sha(x),
                    meta_sha256=w.tensor_sha(meta), route=detail['route'], arm_order=list(order), arms={})
                for arm in order:
                    row['arms'][arm] = measure_arm(models, x, meta, arm, 'cuda:0', progress=wd.progress)
                records.append(row)
                wd.completed += 1
                wd.progress('study-complete', len(x))
                w.write(report, dict(status='RUNNING', identity=checked['identity'], records=records,
                                    qualified_for_submission=False))
                print(json.dumps(dict(completed=len(records), windows=len(x))), flush=True)
                del item, x, meta
        w.need(identity() == checked['identity'], 'source/roster changed during benchmark')
        summary = summarize(records, desktop_used, total)
        passed = all(summary[a]['engineering_gate_pass'] for a in ARMS[1:])
        w.write(report, dict(status='PASS' if passed else 'FAIL', identity=checked['identity'],
            review_sha256=w.sha(review), records=records, summary=summary, load_seconds=loads,
            environment=w.environment('cuda:0'), desktop_used_before_models_bytes=desktop_used,
            total_memory_bytes=total, warmup='first study, once per arm, excluded',
            timing_scope='K1 input transfer + three forwards only; no rendering, parent, or model loading',
            timing_warning='4070 engineering estimate; T4 N32 full parent still required',
            chunk64_release_registered=False, independent_result_review='REQUIRED',
            qualified_for_submission=False))
        print(json.dumps(dict(status='PASS' if passed else 'FAIL', report=str(report))), flush=True)
    except BaseException as exc:
        w.write(report, dict(status='FAILED', error=type(exc).__name__+': '+str(exc)[:400],
                            records=records, qualified_for_submission=False))
        raise
    finally:
        if wd is not None:
            wd.close()
    w.need(passed, 'benchmark gate failed; inspect per-arm results, no automatic retry')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['validate', 'run'])
    p.add_argument('--review', type=Path)
    args = p.parse_args()
    if args.command == 'validate':
        print(json.dumps(validate(args.review), indent=2))
    else:
        w.need(args.review is not None, '--review is required for GPU execution')
        run(args.review)


if __name__ == '__main__':
    main()
