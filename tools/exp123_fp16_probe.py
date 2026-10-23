"""EXP123 (H-B-43) fp16 serving probe: trained VLM weights, 100 fold-0 validation studies, fp16 autocast vs fp32 (and bf16).

  python tools/exp123_fp16_probe.py --checkpoint artifacts/experiments/EXP123_QWEN3VL_PILOT_F0/weights_epoch3.pt --device cuda \
      --out artifacts/experiments/EXP123_QWEN3VL_FP16_PROBE_F0

Serving recipe under test (numerical-parity rule): weights stored in fp32 (bf16 checkpoint upcast exactly, trained parameters fp32), forward
under torch.autocast(fp16) - never .half(); SDPA flash disabled to mirror T4 (no flash) unless --allow-flash. For every module of the model a
forward hook keeps the max |output| and whether any output was non-finite, per precision. Reported: non-finite outputs (studies), the first
modules that went non-finite, the largest activations per module (fp32 run = true magnitudes vs fp16 max 65504), fp16-vs-fp32 probability
parity (tolerance max |dp| 2e-3, per-target Spearman >= 0.999; diagnostic) and timing with a T4 projection for 1,300 studies.
Decision (amended H-B-43 rule, 2026-10-07 21:23 KST): any non-finite fp16 output -> serve in fp32; otherwise fp16 is allowed. The T4
projection uses ASSUMED slow-down factors (--t4-factor-*); only a T4 run measures it. GPU only after training (one GPU, one task).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    for _k, _v in {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1'}.items():
        os.environ.setdefault(_k, _v)
import numpy as np  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FP16_MAX = 65504.0
RISK = 3.0e4
PARITY_MAX_ABS = 2e-3
PARITY_SPEARMAN = 0.999
SERVE_STUDIES = 1300
STOP_HOURS = 3.0


def load_runner():
    sys.path.insert(0, str(ROOT / 'training' / 'src'))
    spec = importlib.util.spec_from_file_location('exp123_qwen3vl_pilot_runner', ROOT / 'training' / 'runners' / 'exp123_qwen3vl_pilot.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class ActivationMonitor:
    """Forward hooks on every named module: running max |output| (as tensors, no host sync per call) and a non-finite flag."""

    def __init__(self, model):
        self.handles, self.amax, self.nonfinite, self.order = [], {}, {}, []
        for name, mod in model.named_modules():
            if not name:
                continue
            self.handles.append(mod.register_forward_hook(self._hook(name)))
        self.enabled = False

    def _hook(self, name):
        import torch

        def fn(_m, _inp, out):
            if not self.enabled:
                return
            t = out
            if isinstance(t, (tuple, list)):
                t = next((x for x in t if torch.is_tensor(x)), None)
            elif hasattr(t, 'last_hidden_state'):
                t = t.last_hidden_state
            elif hasattr(t, 'pooler_output') and torch.is_tensor(getattr(t, 'pooler_output')):
                t = t.pooler_output
            if not torch.is_tensor(t) or not t.is_floating_point() or t.numel() == 0:
                return
            d = t.detach()
            bad = (~torch.isfinite(d)).any()
            m = torch.nan_to_num(d.abs(), nan=0.0, posinf=0.0, neginf=0.0).amax().float()
            if name not in self.amax:
                self.order.append(name)
                self.amax[name], self.nonfinite[name] = m, bad.int()
            else:
                self.amax[name] = torch.maximum(self.amax[name], m)
                self.nonfinite[name] = self.nonfinite[name] + bad.int()
        return fn

    def reset(self):
        self.amax, self.nonfinite, self.order = {}, {}, []

    def snapshot(self):
        return {n: dict(max_abs_finite=float(self.amax[n]), nonfinite_calls=int(self.nonfinite[n])) for n in self.order}

    def remove(self):
        for h in self.handles:
            h.remove()


def spearman(a, b):
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True, help='weights_epoch3.pt from the successful EXP123 pilot')
    ap.add_argument('--n-studies', type=int, default=100)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--precisions', default='fp16,fp32,bf16', help='first entry is the serving candidate; fp32 is the reference')
    ap.add_argument('--out', required=True)
    ap.add_argument('--weights', default=str(ROOT / 'storage/weights/qwen__Qwen3-VL-2B-Instruct'))
    ap.add_argument('--corpus', default=str(ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'))
    ap.add_argument('--k-eval', type=int, default=0, help='0 = the checkpoint k_eval (24)')
    ap.add_argument('--text-layers', type=int, default=None, help='default: as trained (checkpoint args)')
    ap.add_argument('--vision-layers', type=int, default=None)
    ap.add_argument('--allow-flash', action='store_true')
    ap.add_argument('--t4-factor-fp16', type=float, default=3.0, help='ASSUMED T4 / local time ratio for fp16 autocast')
    ap.add_argument('--t4-factor-fp32', type=float, default=6.0, help='ASSUMED T4 / local time ratio for fp32')
    ap.add_argument('--top', type=int, default=30)
    a = ap.parse_args(argv)
    precs = [p.strip() for p in a.precisions.split(',') if p.strip()]
    if not precs or any(p not in ('fp16', 'fp32', 'bf16') for p in precs) or 'fp32' not in precs:
        raise SystemExit('--precisions must list fp16/bf16/fp32 and include fp32 as the reference')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'output exists: {out}')
    r = load_runner()
    from tools import exp123_qwen3vl_pilot as g
    pilot_out = ROOT / 'artifacts' / 'experiments' / g.TAG
    expected_ckpt = pilot_out / 'weights_epoch3.pt'
    if Path(a.checkpoint).resolve() != expected_ckpt.resolve():
        raise SystemExit('fp16 probe requires the EXP123 fixed-final pilot checkpoint')
    summary_path = pilot_out / 'summary.json'
    success_marker = ROOT / 'artifacts' / 'logs' / g.TAG / 'SUCCESS'
    if (not success_marker.is_file() or not summary_path.is_file()
            or json.loads(summary_path.read_text(encoding='utf-8')).get('status') != 'COMPLETED'):
        raise SystemExit('fp16 probe requires a completed EXP123 pilot')
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')) if a.device == 'cpu' else 1)
    if a.device == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit('CUDA requested but not available')
        if not a.allow_flash:
            torch.backends.cuda.enable_flash_sdp(False)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('tag') != g.TAG or ck.get('epoch') != 3:
        raise SystemExit('wrong EXP123 pilot checkpoint')
    targs = ck.get('args', {})
    text_layers = a.text_layers if a.text_layers is not None else int(targs.get('text_layers', 0))
    vision_layers = a.vision_layers if a.vision_layers is not None else int(targs.get('vision_layers', 0))
    k = a.k_eval or int(targs.get('k_eval', 24))
    res = int(targs.get('res', 384))
    t0 = time.time()
    model, cfg, report = r.build_model(a.weights, text_layers=text_layers, vision_layers=vision_layers, base_dtype='float32',
                                       lora_r=int(targs.get('lora_r', 16)), lora_alpha=int(targs.get('lora_alpha', 32)),
                                       lora_dropout=float(targs.get('lora_dropout', 0.05)), grad_ckpt=False,
                                       head_dropout=float(targs.get('head_dropout', 0.0)), device=a.device)
    if ck is not None:
        r.load_trainable(model, ck['trainable'])
    model.to(a.device).eval()
    build_s = time.time() - t0
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.weights)
    mean, std, patch, merge, _ = r.processor_stats(a.weights)
    tpi = (res // patch // merge) ** 2
    p_ids, p_types, p_text = r.build_prompt(tok, (k + 1) // 2, tpi, r.token_ids(cfg))
    if ck is not None and ck.get('prompt_text', {}).get(k, ck.get('prompt_text', {}).get(str(k))) not in (None, p_text):
        raise SystemExit('prompt differs from the trained prompt')
    ids_t, types_t = torch.tensor([p_ids], device=a.device), torch.tensor([p_types], device=a.device)
    _, val = g.fold_ids(ROOT)
    studies = val[:a.n_studies]
    ds = r.PairStudies(Path(a.corpus), studies, {u: np.zeros(12, np.float32) for u in studies}, k, res, False, 42, mean, std, patch=patch, merge=merge)
    batches = [ds[i] for i in range(len(ds))]  # decode once; every precision sees identical inputs
    mon = ActivationMonitor(model.core)
    dtypes = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=None)
    runs = {}
    for prec in precs:
        mon.reset(); mon.enabled = True
        probs, bad_studies, times = [], [], []
        with torch.no_grad():
            for b in batches:
                if a.device == 'cuda':
                    torch.cuda.synchronize()
                ts = time.perf_counter()
                logits = model(ids_t, types_t, b['pixel_values'].to(a.device), b['grid'].to(a.device), amp_dtype=dtypes[prec])
                p = torch.sigmoid(logits.float())
                if a.device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - ts)
                pn = p.cpu().numpy()[0]
                if not np.isfinite(pn).all() or not torch.isfinite(logits).all():
                    bad_studies.append(b['uid'])
                probs.append(pn)
        mon.enabled = False
        acts = mon.snapshot()
        ranked = sorted(acts.items(), key=lambda kv: -kv[1]['max_abs_finite'])
        runs[prec] = dict(nonfinite_studies=len(bad_studies), nonfinite_uids=bad_studies[:10], seconds=float(np.sum(times)),
                          seconds_per_study=float(np.mean(times[1:] if len(times) > 1 else times)),
                          modules_nonfinite=[n for n, v in acts.items() if v['nonfinite_calls']][:20],
                          modules_over_risk=[dict(module=n, max_abs=v['max_abs_finite']) for n, v in ranked if v['max_abs_finite'] > RISK][:50],
                          top_modules=[dict(module=n, max_abs=v['max_abs_finite']) for n, v in ranked[:a.top]],
                          max_activation=ranked[0][1]['max_abs_finite'] if ranked else None, n_modules=len(acts), probs=np.stack(probs))
        if a.device == 'cuda':
            runs[prec]['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    mon.remove()
    ref = runs['fp32']['probs']
    parity = {}
    for prec in precs:
        if prec == 'fp32':
            continue
        P = runs[prec]['probs']
        ok = np.isfinite(P).all()
        d = np.abs(P - ref) if ok else None
        sp = min(spearman(P[:, t], ref[:, t]) for t in range(P.shape[1])) if ok and len(P) > 2 else None
        parity[prec] = dict(max_abs_diff=float(d.max()) if ok else None, mean_abs_diff=float(d.mean()) if ok else None, min_target_spearman=sp,
                            within_tolerance=bool(ok and d.max() <= PARITY_MAX_ABS and (sp is None or sp >= PARITY_SPEARMAN)),
                            tolerance=dict(max_abs=PARITY_MAX_ABS, spearman=PARITY_SPEARMAN))
    cand = precs[0]
    nonfinite = runs[cand]['nonfinite_studies'] > 0 if cand != 'fp32' else False
    serve = 'fp32' if (cand == 'fp32' or nonfinite) else cand
    proj = {}
    for prec, factor in (('fp16', a.t4_factor_fp16), ('fp32', a.t4_factor_fp32)):
        if prec in runs and a.device == 'cuda':  # CPU timings say nothing about T4
            local = runs[prec]['seconds_per_study']
            proj[prec] = dict(local_seconds_per_study=local, assumed_t4_factor=factor, projected_t4_hours=local * factor * SERVE_STUDIES / 3600)
    served = proj.get(serve)
    decision = dict(candidate=cand, fp16_nonfinite=bool(nonfinite), serving_precision=serve,
                    rule='amended H-B-43: non-finite fp16 output on the probed studies -> serve fp32 (no stop)',
                    parity_note='fp16-vs-fp32 parity outside tolerance needs review before fp16 serving, even when finite' if cand in parity and not parity[cand]['within_tolerance'] else None,
                    projected_t4_hours_served=served['projected_t4_hours'] if served else None,
                    t4_stop=bool(served and served['projected_t4_hours'] > STOP_HOURS),
                    t4_note='projection uses assumed T4 factors; a T4 run is needed to confirm (stop if > 3 h)')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for prec in precs:
        R = runs[prec]['probs']
        rows += [dict(StudyInstanceUID=u, precision=prec, **{t: float(R[i, j]) for j, t in enumerate(r.LAB)}) for i, u in enumerate(studies)]
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / 'probe_predictions.csv', index=False)
    body = dict(schema='exp123-fp16-probe-v1', plan_item='H-B-43', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), checkpoint=a.checkpoint,
                checkpoint_epoch=(ck or {}).get('epoch'), device=a.device, n_studies=len(studies), eval_ids_sha256=g.uid_hash(studies), k_eval=k,
                pairs=(k + 1) // 2, storage_dtype='float32', flash_sdp=bool(a.allow_flash), build_seconds=round(build_s, 1), model=report,
                runs={p: {kk: vv for kk, vv in v.items() if kk != 'probs'} for p, v in runs.items()}, parity_vs_fp32=parity, t4_projection=proj,
                decision=decision, environment=r.environment(a.device))
    (out / 'probe.json').write_text(json.dumps(body, indent=1, allow_nan=True), encoding='utf-8')
    print(json.dumps(dict(decision=decision, parity=parity, nonfinite={p: runs[p]['nonfinite_studies'] for p in precs},
                          max_activation={p: runs[p]['max_activation'] for p in precs}), indent=1))
    return body


if __name__ == '__main__':
    main()
