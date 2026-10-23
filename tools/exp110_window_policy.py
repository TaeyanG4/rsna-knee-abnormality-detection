"""EXP110 (H-B-30): test-time window policy on the five Dread CoAtNet5 fold models, pooled OOF on all 4,343 G1 studies.

Variants (inference only, fixed weights, strict FP16 like EXP096):
  ctrl_wsv36  within-slot-valid, K36  -> must reproduce each fold's stored K36 OOF (control)
  glob62      global-valid, K62       -> every valid position of the 64-slice stack, windows may cross slot borders
                                         (the public maxspan-v5 serving policy, k_eval 62; D-A-050)
  wsv62       within-slot-valid, K62  -> all within-slot positions (evenly re-sampled to 62)
Scored against rint(Dread soft labels) like D-A-049; contrasts vs ctrl_wsv36 with a 500-draw component bootstrap
(VALIDATION_GROUP_CONTRACT_V1 components, seed 20261003).

  python tools/exp110_window_policy.py                 # full run on CUDA
  python tools/exp110_window_policy.py --smoke         # CPU, fold 0, 3 studies, ctrl only: code-path check, no GPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'training' / 'src'))
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
FOLDS = {0: dict(ckpt='artifacts/experiments/EXP094L_B8_SEED42/model_best.pt', ref='artifacts/experiments/EXP096_K_SWEEP/B_K36.csv'),
         **{k: dict(ckpt=f'artifacts/experiments/EXP099_B8_DREAD_F{k}/model_best.pt', ref=f'artifacts/experiments/EXP099_B8_DREAD_F{k}/oof_k36.csv') for k in (1, 2, 3, 4)}}
VARIANTS = {'ctrl_wsv36': ('within-slot-valid', 36), 'glob62': ('global-valid', 62), 'wsv62': ('within-slot-valid', 62)}
LABELS = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
COMPONENTS = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
OUT = ROOT / 'artifacts/experiments/EXP110_WINDOW_POLICY'
RES = 224
CTRL_TOL = 2e-3  # FP16 inference parity bound; stored K36 of folds 1-4 came from the training runner (other batch shapes)


def auc(y, s):
    P = int(y.sum()); N = len(y) - P
    if P == 0 or N == 0:
        return np.nan
    r = pd.Series(s).rank(method='average').to_numpy()
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def macro(Y, P):
    return float(np.nanmean([auc(Y[:, t], P[:, t]) for t in range(12)]))


class PolicyWindows:
    def __init__(self, corpus_dir, ids, res, k, policy):
        self.corpus_dir, self.ids, self.res, self.k, self.policy, self.corpus = corpus_dir, list(ids), res, k, policy, None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch
        from raptor_common import Corpus, select_centers, windows_from
        from raptor_data import IMAGENET_MEAN, IMAGENET_STD
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid = self.ids[i]; vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        c = select_centers(mask, self.corpus.depth, self.k, self.corpus.spec, self.policy, None)
        x = (windows_from(vol, c, self.res) - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), uid


def collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def bootstrap(Y, A, B, comp_of, uids, draws=500, seed=20261003):
    """95% interval of macro(A) - macro(B), resampling whole components with replacement."""
    comps = sorted(set(comp_of[u] for u in uids))
    idx_of = {c: [] for c in comps}
    for i, u in enumerate(uids):
        idx_of[comp_of[u]].append(i)
    rng = np.random.default_rng(seed); d = []
    for _ in range(draws):
        pick = rng.integers(0, len(comps), len(comps))
        ii = np.concatenate([idx_of[comps[j]] for j in pick])
        d.append(macro(Y[ii], A[ii]) - macro(Y[ii], B[ii]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from raptor_common import build_model, write_predictions
    from raptor_runtime import configure_runtime
    device = 'cpu' if a.smoke else 'cuda'
    runtime = configure_runtime('none' if a.smoke else 'fp16', device, policy='strict')
    out = OUT / ('smoke' if a.smoke else '')
    if not a.smoke and (OUT / 'result.json').exists():
        raise SystemExit('result exists: ' + str(OUT / 'result.json'))
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(ROOT / LABELS, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    folds = {0: FOLDS[0]} if a.smoke else FOLDS
    variants = {'ctrl_wsv36': VARIANTS['ctrl_wsv36']} if a.smoke else VARIANTS
    pooled = {v: [] for v in variants}; pooled_uids = []; folds_out = {}
    for k, spec in folds.items():
        ref = pd.read_csv(ROOT / spec['ref'], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')[T]
        uids = ref.index.tolist()[:3] if a.smoke else ref.index.tolist()
        ck = torch.load(ROOT / spec['ckpt'], map_location='cpu', weights_only=False)
        model = build_model(ck['arch'], pretrained=False).to(device); model.load_state_dict(ck['model']); model.eval()
        fo = dict(ckpt=spec['ckpt'], ckpt_sha256=hashlib.sha256((ROOT / spec['ckpt']).read_bytes()).hexdigest(), n=len(uids), runs={})
        for name, (policy, kk) in variants.items():
            dl = DataLoader(PolicyWindows(ROOT / CORPUS, uids, RES, kk, policy), batch_size=4, shuffle=False, num_workers=0 if a.smoke else a.workers, collate_fn=collate)
            t0 = time.time(); P, U = [], []
            with torch.no_grad():
                for x, u in dl:
                    if a.smoke:
                        logits = model(x.to(device))
                    else:
                        with torch.autocast('cuda', dtype=torch.float16):
                            logits = model(x.cuda())
                    if not torch.isfinite(logits).all():
                        raise RuntimeError(f'nonfinite logits fold {k} {name}')
                    P.append(torch.sigmoid(logits.float()).cpu().numpy()); U += u
            P = np.concatenate(P)
            if not a.smoke:
                torch.cuda.synchronize()
            wall = time.time() - t0
            assert U == uids
            write_predictions(out / f'F{k}_{name}.csv', U, P)
            fo['runs'][name] = dict(wall_seconds=round(wall, 1), per_study_ms=round(1000 * wall / len(uids), 1))
            if name == 'ctrl_wsv36':
                dev = float(np.abs(P - ref.loc[uids].to_numpy()).max())
                fo['control_max_abs_vs_stored_k36'] = dev
                print(f'fold {k} control max|d| {dev:.2e}', flush=True)
                if not a.smoke and dev > CTRL_TOL:
                    raise SystemExit(f'control does not reproduce fold {k} K36 OOF (max |d| {dev}); stopping')
            pooled[name].append(P)
            print(f'fold {k} {name} {wall:.1f}s', flush=True)
        pooled_uids += uids; folds_out[k] = fo
        del model
        if not a.smoke:
            torch.cuda.empty_cache()
    if a.smoke:
        print('smoke OK', json.dumps(folds_out, indent=1)); return
    Y = np.rint(lab.loc[pooled_uids, T].to_numpy().astype(np.float32)).astype(int)
    comp_of = {m: c['component_id'] for c in json.loads((ROOT / COMPONENTS).read_text(encoding='utf-8')) for m in c['members']}
    P = {v: np.concatenate(pooled[v]) for v in variants}
    res = dict(schema='exp110-window-policy-v1', plan_item='H-B-30', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), runtime=runtime,
               n_studies=len(pooled_uids), labels=LABELS, folds=folds_out, variants={}, contrasts={})
    for v in variants:
        res['variants'][v] = dict(policy=variants[v][0], k=variants[v][1], macro12=macro(Y, P[v]),
                                  targets={T[t]: float(auc(Y[:, t], P[v][:, t])) for t in range(12)})
        write_predictions(OUT / f'pooled_{v}.csv', pooled_uids, P[v])
    for v in ('glob62', 'wsv62'):
        d = res['variants'][v]['macro12'] - res['variants']['ctrl_wsv36']['macro12']
        res['contrasts'][f'{v}-ctrl_wsv36'] = dict(delta=d, ci95=bootstrap(Y, P[v], P['ctrl_wsv36'], comp_of, pooled_uids),
                                                   per_fold=[macro(Y[s], P[v][s]) - macro(Y[s], P['ctrl_wsv36'][s]) for s in
                                                             np.split(np.arange(len(pooled_uids)), np.cumsum([f['n'] for f in folds_out.values()])[:-1])])
    (OUT / 'result.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({v: round(r['macro12'], 6) for v, r in res['variants'].items()} | {c: (round(x['delta'], 6), [round(z, 6) for z in x['ci95']]) for c, x in res['contrasts'].items()}, indent=1))


if __name__ == '__main__':
    main()
