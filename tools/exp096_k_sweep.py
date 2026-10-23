"""EXP096 (H-B-07): test-time window count/density curve with fixed weights (B and R384), V888, strict precision.

  python tools/exp096_k_sweep.py --model B --model R384

For each model: K = 12, 24 (control; must reproduce oof_best within tolerance), 36, 48, and a shifted-center K24 (the odd
positions of the K48 grid). Writes artifacts/experiments/EXP096_K_SWEEP/<model>_K<k>.csv and result.json with macro/12 AUCs,
wall time per K, and the TTA (K24 + shifted K24 mean probability). Inference only; no Gold; labels read only for scoring.
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
MODELS = {'B': dict(ckpt='artifacts/experiments/EXP084_B_G1/model_best.pt', ref='artifacts/experiments/EXP084_B_G1/oof_best.csv', res=224),
          'R384': dict(ckpt='artifacts/experiments/EXP088_RAPTOR384_G1_V4/model_best.pt', ref='artifacts/experiments/EXP088_RAPTOR384_G1_V4/oof_best.csv', res=384)}
LABELS = 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
META = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/group-metadata.csv'
CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
OUT = ROOT / 'artifacts/experiments/EXP096_K_SWEEP'
KS = [12, 24, 36, 48]


def avg_rank(x):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); sx = x[o]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def auc(y, s):
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return np.nan
    r = avg_rank(s); return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


class SweepWindows:
    """Eval-only dataset: K evenly spaced within-slot-valid centers, or the shifted (odd-position) half of a 2K grid."""

    def __init__(self, corpus_dir, ids, res, k, shifted=False):
        self.corpus_dir, self.ids, self.res, self.k, self.shifted, self.corpus = corpus_dir, list(ids), res, k, shifted, None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch
        from raptor_common import Corpus, select_centers, windows_from
        from raptor_data import IMAGENET_MEAN, IMAGENET_STD
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid = self.ids[i]; vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        if self.shifted:
            c = select_centers(mask, self.corpus.depth, 2 * self.k, self.corpus.spec, 'within-slot-valid', None)[1::2]
        else:
            c = select_centers(mask, self.corpus.depth, self.k, self.corpus.spec, 'within-slot-valid', None)
        x = (windows_from(vol, c, self.res) - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), uid


def collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--model', action='append', choices=list(MODELS), required=True); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from raptor_common import build_model, write_predictions
    from raptor_runtime import configure_runtime
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(ROOT / LABELS).set_index('StudyInstanceUID'); meta = pd.read_csv(ROOT / META).set_index('StudyInstanceUID')
    runtime = configure_runtime('fp16', 'cuda', policy='strict')
    result = dict(schema='exp096-k-sweep-v1', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), plan_item='H-B-07', runtime=runtime, models={})
    for name in a.model:
        spec = MODELS[name]; ck = torch.load(ROOT / spec['ckpt'], map_location='cpu', weights_only=False)
        ref = pd.read_csv(ROOT / spec['ref']).set_index('StudyInstanceUID')[T]; uids = ref.index.tolist()
        Y = np.rint(lab.loc[uids, T].to_numpy().astype(np.float32)).astype(int)
        model = build_model(ck['arch'], pretrained=False).cuda(); model.load_state_dict(ck['model']); model.eval()
        entry = dict(ckpt=spec['ckpt'], ckpt_sha256=hashlib.sha256((ROOT / spec['ckpt']).read_bytes()).hexdigest(), arch=ck['arch'], k_eval_trained=ck.get('k_eval'), runs={})
        preds = {}
        for k, shifted in [(12, False), (24, False), (36, False), (48, False), (24, True)]:
            key = f'K{k}' + ('_shift' if shifted else '')
            dl = DataLoader(SweepWindows(ROOT / CORPUS, uids, spec['res'], k, shifted), batch_size=4, shuffle=False, num_workers=a.workers, collate_fn=collate)
            t0 = time.time(); P, U = [], []
            with torch.no_grad():
                for x, u in dl:
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = model(x.cuda())
                    if not torch.isfinite(logits).all():
                        raise RuntimeError('nonfinite logits ' + key)
                    P.append(torch.sigmoid(logits.float()).cpu().numpy()); U += u
            P = np.concatenate(P); torch.cuda.synchronize(); wall = time.time() - t0
            assert U == uids
            write_predictions(OUT / f'{name}_{key}.csv', U, P); preds[key] = P
            aucs = np.array([auc(Y[:, t], P[:, t]) for t in range(12)])
            entry['runs'][key] = dict(macro12=float(aucs.mean()), targets={T[t]: float(aucs[t]) for t in range(12)}, wall_seconds=round(wall, 1), per_study_ms=round(1000 * wall / len(uids), 1))
            print(name, key, round(float(aucs.mean()), 6), round(wall, 1), 's', flush=True)
        ctrl = np.abs(preds['K24'] - ref.to_numpy()).max()
        entry['control_K24_vs_oof_best_max_abs'] = float(ctrl)
        tta = (preds['K24'] + preds['K24_shift']) / 2; aucs = np.array([auc(Y[:, t], tta[:, t]) for t in range(12)])
        write_predictions(OUT / f'{name}_TTA24.csv', uids, tta)
        entry['runs']['TTA24'] = dict(macro12=float(aucs.mean()), targets={T[t]: float(aucs[t]) for t in range(12)})
        ref_auc = np.array([auc(Y[:, t], ref.to_numpy()[:, t]) for t in range(12)]).mean(); entry['oof_best_macro12'] = float(ref_auc)
        result['models'][name] = entry
        del model; torch.cuda.empty_cache()
    (OUT / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({m: {k: (round(v['macro12'], 6), v.get('wall_seconds')) for k, v in e['runs'].items()} | {'ctrl_maxabs': e['control_K24_vs_oof_best_max_abs']} for m, e in result['models'].items()}, indent=1))


if __name__ == '__main__':
    main()
