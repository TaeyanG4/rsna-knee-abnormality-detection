"""v2.0.0: registered stride 2/3 K1 windows from shared R6 raw rows.

Changelog: explicit eval_stride for windows and original-side rerender;
v1 renderer contract, raw-row capture and restoration are unchanged.

No cohort scan, disk cache, launch, model selection, or prediction fallback.
This is an integration adapter, not a modified/published INF3 release.
Call in the renderer process, then transport the returned K1 item to the
classifier process. Never put a classifier inside a DataLoader worker.
"""
from __future__ import annotations

import time

from tools import l1_rco_k2_worker_v102 as w

VERSION = '2.0.0'
R6_INFER_SHA = '8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d'
R6_RUNTIME_SHA = '7cc97837dfd7b41d9604b6f43a6340bb7c34fa9708878a930334cc7e0cdf8b0e'
K1_INFER_SHA = '34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a'
FP32_MAX_ABS = 1e-6
FP16_MAX_ABS = 5e-3
FP16_MIN_SPEARMAN = .999
R6_SOURCES = {
    'compact_train': '04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba',
    'px_cache': 'b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632',
    'knee_anchor_bank_v2': 'e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309',
    'probe_infer': 'a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980',
    'side_common': 'bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566',
}


def stride_check(eval_stride):
    w.need(type(eval_stride) is int and eval_stride in (2, 3), 'K1 eval_stride must be 2 or 3')
    return eval_stride


def original_side(dataset, renderer_module, index, eval_stride=3):
    """Restore the K1 contract even after R6 installed a forced-side entry."""
    stride_check(eval_stride)
    uid = dataset.uids[index]
    forced = renderer_module.FORCED_SIDE.pop(uid, None)
    old = {k: getattr(dataset, k) for k in ('side_net', 'eval_stride', 'extent_meta', 'fluid')}
    try:
        dataset.side_net, dataset.eval_stride, dataset.extent_meta, dataset.fluid = None, eval_stride, False, False
        item = dataset[index]
        w.validate_item(item, uid)
        return item
    finally:
        for key, value in old.items():
            setattr(dataset, key, value)
        renderer_module.FORCED_SIDE.pop(uid, None)
        if forced is not None:
            renderer_module.FORCED_SIDE[uid] = forced


def from_rows(item, uid, window_rows, eval_stride=3):
    """Recompute exact K1 centres/edges from raw rows at the registered stride."""
    stride_check(eval_stride)
    import numpy as np
    import torch
    px, meta = item['px'], item['meta']
    w.need(px.dtype == torch.uint8 and px.ndim == 3 and
           tuple(px.shape[1:]) == (384, 384), 'R6 px layout')
    w.need(meta.dtype == torch.float32 and tuple(meta.shape) == (len(px), 12)
           and torch.isfinite(meta).all().item(), 'R6 metadata layout')
    p, m = px.numpy(), meta.numpy()
    blocks = []
    for b in item['blocks']:
        n = b['n_slices']
        w.need(isinstance(n, int) and n > 0, 'empty block')
        views = []
        for v, start in b['views']:
            w.need(0 <= start and start + n <= len(px), 'block outside rows')
            views.append((0, v, start))
        blocks.append(dict(n_slices=n, views=views))
    cand = window_rows(blocks, lambda f, r: m[r], 'eval', 3, eval_stride, phys_mm=3.5)
    w.need(bool(cand), 'no K1 windows')
    rows = np.asarray(cand, dtype=np.int64)
    images = np.stack([p[rows[:, 2 + d]] for d in range(3)], 1)
    centers = m[rows[:, 1]]
    meta6 = np.stack([centers[:, j] for j in (1, 3, 4, 5, 6, 7)], 1).astype(np.float32)
    return dict(uid=uid, error=None, images=torch.from_numpy(images),
                meta=torch.from_numpy(meta6), side=None, anchor=None)


class SharedR6:
    """One instance per loader process, sequential use only (same as CStudies).

    Accept only the embedded R6 source or its exact parent diagnostic patch.
    A side-corrected study is rerendered with correction disabled, using the
    identical R6/K1 rendering methods. FORCED_SIDE is removed during that
    render; merely setting side_net=None would retain the corrected geometry.
    """
    def __init__(self, dataset, renderer_module, eval_stride=3):
        from pathlib import Path
        self.eval_stride = stride_check(eval_stride)
        w.need(w.sha(renderer_module.__file__) in (R6_INFER_SHA, R6_RUNTIME_SHA), 'unpinned R6 renderer')
        w.need(type(dataset) is renderer_module.CStudies, 'wrong renderer class')
        w.need((dataset.size, dataset.views, dataset.slices, dataset.phys_mm,
                dataset.eval_stride) == (384, 'v4', 3, 3.5, 1), 'R6 render contract')
        w.need(w.sha(dataset.anchor_weights) == w.ANATOMY_SHA, 'anatomy mismatch')
        # Hash the actual modules used by _setup, not just a nearby source tree.
        import compact_train, px_cache, knee_anchor_bank_v2, probe_infer, side_common
        root = Path(renderer_module.__file__).parent
        for module in (compact_train, px_cache, knee_anchor_bank_v2, probe_infer, side_common):
            w.need(Path(module.__file__).resolve() == (root / (module.__name__ + '.py')).resolve(),
                   'renderer import collision: ' + module.__name__)
            w.need(w.sha(module.__file__) == R6_SOURCES[module.__name__], 'R6 dependency hash')
        if dataset.side_model:
            w.need(w.sha(dataset.side_model) ==
                   'ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a', 'side model hash')
        w.need((dataset.side_threshold, dataset.side_position_mm, dataset.side_max_x) ==
               (2.0, 20.0, 0.0), 'side decision contract')
        self.ds, self.module, self.active = dataset, renderer_module, False

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        w.need(not self.active, 'SharedR6 is not thread safe')
        self.active = True
        ds, ci = self.ds, self.module
        uid = ds.uids[index]
        saved_forced = ci.FORCED_SIDE.pop(uid, None)
        original_build = original_native = None
        capture, side = {}, {}
        started = time.perf_counter()
        try:
            if ds.native is None:
                ds._setup()
            original_build, original_native = ds.builder.build, ds._native_and_geometry

            def native(*args, **kwargs):
                note = original_native(*args, **kwargs)
                w.need(note is None or isinstance(note.get('flipped'), bool), 'missing side decision')
                side['note'] = note
                return note

            def build(study):
                item = original_build(study)
                w.need(study == uid and 'note' in side, 'capture outside current study')
                note = side['note']
                if note is None or not note['flipped']:
                    t = time.perf_counter()
                    capture['item'] = from_rows(item, uid, ds.window_rows, self.eval_stride)
                    capture['seconds'] = time.perf_counter() - t
                return item

            ds.builder.build, ds._native_and_geometry = build, native
            parent = ds[index]
            w.need(parent.get('uid') == uid and parent.get('error') is None,
                   'parent rendering failed; no successful prediction fallback')
            ds.builder.build, ds._native_and_geometry = original_build, original_native
            note = side['note']
            corrected = bool(note and note['flipped'])
            if corrected:
                # R6 images cannot be unflipped to recover anchors/views exactly.
                t = time.perf_counter()
                k1 = original_side(ds, ci, index, self.eval_stride)
                extra = time.perf_counter() - t
            else:
                k1 = capture['item']
                k1['anchor'] = parent.get('anchor')
                extra = capture['seconds']
            w.validate_item(k1, uid)
            return parent, k1, dict(route='original-side-rerender' if corrected else 'shared-rows',
                                   k1_eval_stride=self.eval_stride,
                                   extra_render_seconds=extra,
                                   render_wall_seconds=time.perf_counter() - started)
        finally:
            if original_build is not None:
                ds.builder.build = original_build
                ds._native_and_geometry = original_native
            ci.FORCED_SIDE.pop(uid, None)
            if saved_forced is not None:
                ci.FORCED_SIDE[uid] = saved_forced
            self.active = False


def predict(model, x, meta, device, fp16=False):
    """FP32 uses the frozen worker verbatim. FP16 is an opt-in experiment."""
    if not fp16:
        return w.predict(model, x, meta, device)
    import torch
    w.need(device.startswith('cuda'), 'FP16 experiment requires CUDA')
    mask = torch.ones((1, len(x)), dtype=torch.bool, device=device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        logits = model(x[None, ..., 16:368, 16:368].to(device),
                       meta[None].to(device), mask, chunk=32)
    w.need(tuple(logits.shape) == (1, 12) and torch.isfinite(logits).all().item(), 'invalid FP16 logits')
    return logits.float().sigmoid().cpu().numpy()[0]


def compare_predictions(original, shared, half=None):
    """Fixed engineering gates, no label access or accuracy claim."""
    import numpy as np
    from scipy.stats import spearmanr
    a, b = np.asarray(original), np.asarray(shared)
    w.need(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 12 and
           np.isfinite(a).all() and np.isfinite(b).all(), 'invalid prediction matrices')
    delta = float(np.max(np.abs(a - b)))
    out = dict(fp32_max_abs=delta, fp32_pass=delta <= FP32_MAX_ABS,
               fp32_limit=FP32_MAX_ABS, fp16_status='NOT_RUN')
    if half is not None:
        h = np.asarray(half)
        w.need(h.shape == a.shape and len(h) >= 32 and np.isfinite(h).all(), 'FP16 needs >=32 finite studies')
        correlations = [float(spearmanr(b[:, j], h[:, j]).statistic) for j in range(12)]
        maximum = float(np.max(np.abs(b - h)))
        passed = maximum <= FP16_MAX_ABS and all(np.isfinite(r) and r >= FP16_MIN_SPEARMAN for r in correlations)
        out.update(fp16_status='PASS' if passed else 'FAIL', fp16_max_abs=maximum,
                   fp16_spearman=correlations, fp16_max_abs_limit=FP16_MAX_ABS,
                   fp16_spearman_floor=FP16_MIN_SPEARMAN)
    return out
