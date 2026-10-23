"""EXP097 (H-B-06): frozen-weight window-feature diagnostics on V888 — slot ablation, attention mass, uniform pooling.

  python tools/exp097_slot_diag.py --model B --model R384

Step 1 (GPU, minutes): for each model, encode the K24 eval windows of every V888 study once; store features, window centers
and their slot index. Step 2 (CPU): re-run the stored head (LayerNorm → attention → per-finding pooling) to (a) reproduce
oof_best, (b) report attention mass per slot per finding, (c) leave-one-slot-out AUC per finding (windows of that slot removed
before the softmax), (d) uniform mean pooling instead of attention. No training, no Gold. Writes artifacts/experiments/EXP097_SLOT_DIAG/.
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
OUT = ROOT / 'artifacts/experiments/EXP097_SLOT_DIAG'
SLOT_NAMES = ['Sag-FS', 'Sag-nonFS', 'Cor-FS', 'Cor-nonFS', 'Axial']


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


class EvalWindows:
    def __init__(self, corpus_dir, ids, res, k):
        self.corpus_dir, self.ids, self.res, self.k, self.corpus = corpus_dir, list(ids), res, k, None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch
        from raptor_common import Corpus, select_centers, windows_from
        from raptor_data import IMAGENET_MEAN, IMAGENET_STD
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid = self.ids[i]; vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        c = select_centers(mask, self.corpus.depth, self.k, self.corpus.spec, 'within-slot-valid', None)
        x = (windows_from(vol, c, self.res) - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), np.array(c, dtype=np.int64), uid


def collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), np.stack([b[1] for b in batch]), [b[2] for b in batch]


def head_numpy(hp, feats, mask=None, uniform=False):
    """feats (K,F) float32; returns logits (12,) and attention (K,12). mask: boolean (K,) windows kept."""
    import torch
    with torch.no_grad():
        h = torch.nn.functional.layer_norm(torch.from_numpy(feats), (feats.shape[1],), hp['norm.weight'], hp['norm.bias'])
        z = torch.tanh(h @ hp['att.0.weight'].T + hp['att.0.bias']) @ hp['att.2.weight'].T + hp['att.2.bias']
        if mask is not None:
            z = z.masked_fill(~torch.from_numpy(mask)[:, None], float('-inf'))
        a = torch.softmax(z, dim=0) if not uniform else (torch.from_numpy(mask).float()[:, None].expand_as(z) / float(mask.sum()) if mask is not None else torch.full_like(z, 1.0 / z.shape[0]))
        pooled = torch.einsum('kn,kf->nf', a, h)
        logits = (pooled * hp['clsW']).sum(-1) + hp['clsb']
        return logits.numpy(), a.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--model', action='append', choices=list(MODELS), required=True); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from raptor_common import Corpus, build_model
    from raptor_runtime import configure_runtime
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(ROOT / LABELS).set_index('StudyInstanceUID'); meta = pd.read_csv(ROOT / META).set_index('StudyInstanceUID')
    corpus = Corpus(ROOT / CORPUS); sizes = [int(s[2]) for s in corpus.spec['slots']]; bounds = np.cumsum([0] + sizes)
    runtime = configure_runtime('fp16', 'cuda', policy='strict')
    result = dict(schema='exp097-slot-diag-v1', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), plan_item='H-B-06', slots=SLOT_NAMES, slot_sizes=sizes, runtime=runtime, models={})
    for name in a.model:
        spec = MODELS[name]; ck = torch.load(ROOT / spec['ckpt'], map_location='cpu', weights_only=False)
        ref = pd.read_csv(ROOT / spec['ref']).set_index('StudyInstanceUID')[T]; uids = ref.index.tolist()
        Y = np.rint(lab.loc[uids, T].to_numpy().astype(np.float32)).astype(int); comp = meta.loc[uids, 'component_id'].to_numpy()
        model = build_model(ck['arch'], pretrained=False).cuda(); model.load_state_dict(ck['model']); model.eval()
        feats_path = OUT / f'{name}_feats_K24.npz'
        if not feats_path.exists():
            dl = DataLoader(EvalWindows(ROOT / CORPUS, uids, spec['res'], 24), batch_size=4, shuffle=False, num_workers=a.workers, collate_fn=collate)
            F, C, U = [], [], []; t0 = time.time()
            with torch.no_grad():
                for x, c, u in dl:
                    with torch.autocast('cuda', dtype=torch.float16):
                        f = model.encode(x.cuda())
                    F.append(f.float().cpu().numpy()); C.append(c); U += u
            F, C = np.concatenate(F), np.concatenate(C); assert U == uids
            np.savez(feats_path, feats=F, centers=C, uids=np.array(uids)); print(name, 'features', F.shape, round(time.time() - t0, 1), 's', flush=True)
        z = np.load(feats_path); F, C = z['feats'], z['centers']
        hp = {k: v.float().cpu() for k, v in ck['model'].items() if not k.startswith('backbone.')}
        slot_of = np.where(C[..., None] >= bounds[None, None, :-1], 1, 0).sum(-1) - 1; slot_of[C < 0] = -1
        base_logits, att = [], []
        for i in range(len(uids)):
            lg, at = head_numpy(hp, F[i]); base_logits.append(lg); att.append(at)
        base_logits, att = np.stack(base_logits), np.stack(att)
        base_p = 1 / (1 + np.exp(-base_logits)); repro = float(np.abs(base_p - ref.to_numpy()).max())
        base_auc = np.array([auc(Y[:, t], base_logits[:, t]) for t in range(12)])
        mass = {}
        for s in range(5):
            m = (slot_of == s)
            mass[SLOT_NAMES[s]] = {T[t]: float(np.mean([att[i, m[i], t].sum() for i in range(len(uids)) if m[i].any()])) for t in range(12)}
            mass[SLOT_NAMES[s]]['window_share'] = float(m.mean())
        ablate = {}
        for s in range(5):
            lg = []
            for i in range(len(uids)):
                keep = slot_of[i] != s
                if keep.sum() == 0 or (slot_of[i] == s).sum() == 0:
                    lg.append(base_logits[i]); continue
                lg.append(head_numpy(hp, F[i], mask=keep)[0])
            lg = np.stack(lg); a_s = np.array([auc(Y[:, t], lg[:, t]) for t in range(12)])
            ablate[SLOT_NAMES[s]] = dict(macro12_delta=float(a_s.mean() - base_auc.mean()), targets={T[t]: float(a_s[t] - base_auc[t]) for t in range(12)})
        uni = np.stack([head_numpy(hp, F[i], mask=(slot_of[i] >= 0), uniform=True)[0] for i in range(len(uids))])
        a_u = np.array([auc(Y[:, t], uni[:, t]) for t in range(12)])
        rng = np.random.default_rng(20261003); cu = np.unique(comp); idx = {c: np.where(comp == c)[0] for c in cu}
        se = {}
        for t in range(12):
            d = [auc(Y[ii, t], base_logits[ii, t]) for ii in [np.concatenate([idx[c] for c in rng.choice(cu, size=len(cu), replace=True)]) for _ in range(200)]]
            se[T[t]] = float(np.nanstd(d))
        result['models'][name] = dict(ckpt=spec['ckpt'], ckpt_sha256=hashlib.sha256((ROOT / spec['ckpt']).read_bytes()).hexdigest(), repro_max_abs_vs_oof_best=repro,
                                      base_macro12=float(base_auc.mean()), base_targets={T[t]: float(base_auc[t]) for t in range(12)}, bootstrap_se_targets=se,
                                      attention_mass_by_slot=mass, leave_one_slot_out=ablate,
                                      uniform_pooling=dict(macro12=float(a_u.mean()), delta=float(a_u.mean() - base_auc.mean()), targets={T[t]: float(a_u[t] - base_auc[t]) for t in range(12)}))
        print(name, 'repro max abs', repro, 'base', round(float(base_auc.mean()), 6), 'uniform Δ', round(float(a_u.mean() - base_auc.mean()), 6), {k: round(v['macro12_delta'], 5) for k, v in ablate.items()}, flush=True)
        del model; torch.cuda.empty_cache()
    (OUT / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
