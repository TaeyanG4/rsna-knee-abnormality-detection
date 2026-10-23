"""EXP124 inference-only coronal/axial horizontal flip TTA on EXP113 and EXP117 F0-F4.

The corpus slot labels determine which K36 windows may be flipped. Sagittal windows
are untouched. Plain inference is checked against each frozen fold OOF before the
new pooled plain/TTA tables and paired component-bootstrap comparison are written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/src'))
from tools import exp113_radr50_local as e113
from tools import exp117_multihead as e117
from training.src.raptor_common import Corpus, LAB, build_model, select_centers, write_predictions
from training.src.raptor_data import collate
from training.src.raptor_radimagenet import CenterSliceWindows
from training.src.raptor_runtime import configure_runtime

BASE = ROOT / 'artifacts/experiments/EXP124_FLIP_TTA'
MODELS = ('exp113', 'exp117')
UID = e113.UID
TARGETS = e113.T


def plane_ranges(spec):
    """Read planes from the cache contract, never infer them from K36 position."""
    slots = spec.get('slots')
    if not isinstance(slots, list) or not slots:
        raise ValueError('plane identification failed: corpus spec has no slots')
    result, start = [], 0
    for item in slots:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError('plane identification failed: malformed slot')
        plane, _fluid, count = item
        if plane not in ('Sagittal', 'Coronal', 'Axial') or not isinstance(count, int) or count < 3:
            raise ValueError(f'plane identification failed: ambiguous plane/size {item!r}')
        result.append(dict(plane=plane, start=start, stop=start + count))
        start += count
    if {r['plane'] for r in result} != {'Sagittal', 'Coronal', 'Axial'} or start != spec.get('depth'):
        raise ValueError('plane identification failed: Sagittal/Coronal/Axial coverage or depth is ambiguous')
    return result


def flip_mask(centers, ranges):
    """Boolean per K36 window. A centre on a slot boundary is rejected."""
    flags = []
    for c in centers:
        if c == -1:
            flags.append(False)
            continue
        hits = [r for r in ranges if r['start'] + 1 <= c < r['stop'] - 1]
        if len(hits) != 1:
            raise ValueError(f'plane identification failed: window center {c} crosses/escapes slot')
        flags.append(hits[0]['plane'] in ('Coronal', 'Axial'))
    return flags


def mirror_windows(x, flags):
    import torch
    if x.ndim != 4 or len(flags) != len(x):
        raise ValueError('window tensor/plane mask mismatch')
    out = x.clone()
    ix = [i for i, yes in enumerate(flags) if yes]
    if ix:
        out[ix] = torch.flip(out[ix], dims=(-1,))
    return out


class PlaneAwareWindows:
    def __init__(self, base, corpus, ranges):
        self.base, self.corpus, self.ranges = base, corpus, ranges

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, uid = self.base[i]
        centers = select_centers(self.corpus.mask(uid), self.corpus.depth, self.base.k, self.corpus.spec,
                                 self.base.window_policy, None)
        return x, mirror_windows(x, flip_mask(centers, self.ranges)), y, uid


def collate_pair(batch):
    import torch
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), [b[3] for b in batch])


def model_config(name):
    if name == 'exp113':
        return e113, e113.PROTOCOL, 'EXP113_RADR50_LOCAL_F', 'EXP113_RADR50_LOCAL_OOF'
    if name == 'exp117':
        return e117, e117.PROTOCOL, 'EXP117_MULTIHEAD_F', 'EXP117_MULTIHEAD_OOF'
    raise ValueError('unknown model')


def make_model(name, ckpt, p):
    model = build_model(p['common_args']['arch'], pretrained=False)
    if name == 'exp117':
        model = e117.build_multihead(model, len(e117.HEAD_NAMES))
    model.load_state_dict(ckpt['model'], strict=True)
    return model.eval()


def served_probabilities(name, logits):
    import torch
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError('nonfinite inference logits')
    return torch.sigmoid(logits.float()).cpu().numpy()


def finish_probabilities(name, chunks):
    import numpy as np
    whole = np.concatenate(chunks)
    return e117.rank_mean(whole) if name == 'exp117' else whole


def infer_pair(model, loader, device, name):
    import torch
    plain, flipped, ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for x, xm, _y, batch_ids in loader:
            with torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else torch.no_grad():
                a = model(x.to(device))
                b = model(xm.to(device))
            plain.append(served_probabilities(name, a))
            flipped.append(served_probabilities(name, b))
            ids += list(batch_ids)
    return finish_probabilities(name, plain), finish_probabilities(name, flipped), ids


def stored_predictions(path, ids):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path, dtype={UID: str})
    if list(df.columns) != [UID] + TARGETS or df[UID].tolist() != ids:
        raise ValueError(f'stored OOF order/columns differ: {path}')
    values = df[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'stored OOF is nonfinite: {path}')
    return values


def compare_plain_tta(ids, plain, tta, draws=300):
    from tools import exp115_vlm_eval as mr
    refs = mr.references(ids, ROOT)
    idx = mr.bootstrap_index(mr.component_groups(ids, ROOT), draws, 20261003)
    return {r: mr.compare(tta, plain, refs[r], idx) for r in mr.REFERENCES}


def run(name, device='cuda', limit=None, output_root=BASE, workers=2):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    guard, protocol_dir, prefix, _oof = model_config(name)
    ppath = ROOT / protocol_dir / 'manifest.json'
    p = guard.read(ppath)
    corpus = Corpus(ROOT / p['paths']['corpus'])
    if corpus.contract != 'raptor64_336_c140_s02' or guard.sha256(corpus.dir / 'manifest.json') != p['corpus_manifest_sha256']:
        raise ValueError('corpus contract/manifest changed')
    ranges = plane_ranges(corpus.spec)
    if p['common_args']['window_policy'] != 'within-slot-valid' or p['common_args']['input_policy'] != 'center3':
        raise ValueError('plane-aware TTA only supports the registered within-slot centre3 recipe')
    out = Path(output_root) / name
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    if limit is not None and (device != 'cpu' or limit < 1):
        raise ValueError('--limit is CPU-only and positive')
    configure_runtime('none' if device == 'cpu' else 'fp16', device, policy=p['common_args']['runtime_policy'])
    all_ids, all_plain, all_tta, folds = [], [], [], {}
    for fold in e113.FOLDS:
        f = p['folds'][str(fold)]
        ids = guard.read(ROOT / f['ordered_ids']['path'])['val']
        if e113.uid_hash(ids) != f['ordered_ids']['val_sha256']:
            raise ValueError('fold validation UID order changed')
        if limit is not None:
            ids = ids[:limit]
        ckpt_path = ROOT / 'artifacts/experiments' / f'{prefix}{fold}' / 'model_best.pt'
        ckpt_sha = guard.sha256(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt.get('tag') != f'{prefix}{fold}' or ckpt.get('fold') != fold:
            raise ValueError('checkpoint fold/tag mismatch')
        prov = ckpt.get('provenance') or {}
        if prov.get('protocol_sha256') != guard.sha256(ppath) or prov.get('eval_ids_ordered_sha256') != f['ordered_ids']['val_sha256']:
            raise ValueError('checkpoint protocol/UID provenance mismatch')
        model = make_model(name, ckpt, p).to(device)
        labels = {u: np.zeros(len(LAB), np.float32) for u in ids}
        a = p['common_args']
        ds = CenterSliceWindows(corpus.dir, ids, labels, a['res'], 36, False, a['norm'], p['seed'],
                                window_policy=a['window_policy'], input_policy=a['input_policy'])
        loader = DataLoader(PlaneAwareWindows(ds, corpus, ranges), batch_size=f['eval_batch'], shuffle=False,
                            num_workers=min(workers, 2), collate_fn=collate_pair, persistent_workers=False,
                            worker_init_fn=guard.worker_init)
        plain, mirrored, got = infer_pair(model, loader, device, name)
        if got != ids:
            raise ValueError('inference UID order changed')
        tta = (plain + mirrored) / 2
        stored_path = ROOT / 'artifacts/experiments' / f'{prefix}{fold}' / 'oof_k36.csv'
        stored = stored_predictions(stored_path, ids) if limit is None else stored_predictions_subset(stored_path, ids)
        diff = float(np.max(np.abs(plain - stored)))
        folds[str(fold)] = dict(rows=len(ids), checkpoint_sha256=ckpt_sha, stored_oof_sha256=guard.sha256(stored_path),
                                plain_max_abs_diff=diff)
        if limit is None and diff > 0.002:
            raise RuntimeError(f'{name} F{fold} plain reproduction differs by {diff:.6g} (>0.002); stop before scoring')
        all_ids.extend(ids); all_plain.append(plain); all_tta.append(tta)
        del model
    plain = np.concatenate(all_plain)
    tta = np.concatenate(all_tta)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('fold validation UIDs overlap')
    metrics = compare_plain_tta(all_ids, plain, tta) if limit is None else None
    out.mkdir(parents=True)
    write_predictions(out / 'oof_plain_k36.csv', all_ids, plain)
    write_predictions(out / 'oof_tta_k36.csv', all_ids, tta)
    body = dict(schema='exp124-flip-tta-v1', model=name, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                protocol_sha256=guard.sha256(ppath), corpus_manifest_sha256=guard.sha256(corpus.dir / 'manifest.json'),
                plane_ranges=ranges, transform='horizontal pixel mirror of coronal and axial windows only; sagittal untouched',
                multihead_rule='rank mean each pass over all fold studies, then average plain and mirrored served scores' if name == 'exp117' else None,
                folds=folds, rows=len(all_ids), metrics=metrics, device=device,
                outputs={n: guard.sha256(out / n) for n in ('oof_plain_k36.csv', 'oof_tta_k36.csv')},
                risk='corpus keeps anatomical plane labels but does not record/reorient DICOM in-plane direction; horizontal is assumed left-right')
    (out / 'result.json').write_text(json.dumps(body, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(rows=len(all_ids), plane_ranges=ranges, max_plain_diff=max(x['plain_max_abs_diff'] for x in folds.values()),
                          deltas={r: v['delta'] for r, v in metrics.items()} if metrics else None), indent=2))
    return body


def stored_predictions_subset(path, ids):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path, dtype={UID: str}).set_index(UID)
    if not set(ids) <= set(df.index) or list(df.columns) != TARGETS:
        raise ValueError('stored OOF subset mismatch')
    return df.loc[ids, TARGETS].to_numpy(np.float64)


def queue_entries(after):
    result, prior = [], after
    for name in MODELS:
        item = dict(tag=f'EXP124_FLIP_TTA_{name.upper()}', command=[e113.train_python(), str(ROOT / 'tools/exp124_flip_tta.py'),
                                                                  'run', '--model', name, '--device', 'cuda'],
                    continue_on_failure=True)
        if prior:
            item['after'] = prior
        result.append(item)
        prior = item['tag']
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    s = sub.add_parser('run'); s.add_argument('--model', choices=MODELS, required=True)
    s.add_argument('--device', choices=['cuda', 'cpu'], default='cuda'); s.add_argument('--limit', type=int)
    s.add_argument('--output-root', default=str(BASE)); s.add_argument('--workers', type=int, default=2)
    s = sub.add_parser('queue-json'); s.add_argument('--after', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.action == 'run':
        run(a.model, a.device, a.limit, a.output_root, a.workers)
    else:
        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            json.dump(queue_entries(None if a.after.lower() == 'none' else a.after), f, indent=2)
            f.write('\n')
        print(path)


if __name__ == '__main__':
    main()
