"""EXP120 F0 checkpoint rescue: diagnose registered FP16 eval, then K24/K36 FP32 OOF.

This is a new evaluation protocol. The trained checkpoint and EXP120 runner stay frozen.
No training is performed. ``screen`` scores the new OOF against EXP113 F0 and a
CoAtNet5+RadR50 fold-0 rank stack.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/src'))
from tools import exp120_densenet_local as ex
from tools import exp113_radr50_local as e113
from training.src.raptor_common import Corpus, LAB, build_model, write_predictions
from training.src.raptor_data import collate
from training.src.raptor_radimagenet import CenterSliceWindows
from training.src.raptor_runtime import configure_runtime

TAG = 'EXP120_DENSENET_LOCAL_F0_EVALFP32'
OUT = ROOT / 'artifacts/experiments' / TAG
CKPT = ROOT / 'artifacts/experiments/EXP120_DENSENET_LOCAL_F0/model_best.pt'
K24 = 'oof_best.csv'
K36 = 'oof_k36.csv'
UID = e113.UID
TARGETS = e113.T


def checkpoint_contract(ckpt, p):
    if ckpt.get('tag') != ex.tag_for(0) or ckpt.get('fold') != 0:
        raise ValueError('not the EXP120 fold-0 checkpoint')
    if ckpt.get('arch') != p['common_args']['arch'] or ckpt.get('res') != p['common_args']['res']:
        raise ValueError('checkpoint architecture/resolution differ from protocol')
    for k in ('norm', 'window_policy', 'input_policy'):
        if ckpt.get(k) != p['common_args'][k]:
            raise ValueError('checkpoint input contract differs: ' + k)
    if ckpt.get('k_eval') != p['common_args']['k_eval'] or ckpt.get('epoch') != p['common_args']['epochs'] - 1:
        raise ValueError('checkpoint K/epoch differ from fixed-final protocol')
    prov = ckpt.get('provenance') or {}
    if prov.get('protocol_sha256') != ex.sha256(ex.manifest_path()) or prov.get('eval_ids_ordered_sha256') != p['folds']['0']['ordered_ids']['val_sha256']:
        raise ValueError('checkpoint provenance differs from registered fold')


def first_nonfinite_module(model, x, device='cuda'):
    """Run first batch in FP16 and name the earliest module with non-finite output."""
    import torch
    first = []
    hooks = []

    def tensors(obj):
        if isinstance(obj, torch.Tensor):
            yield obj
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                yield from tensors(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                yield from tensors(item)

    def hook(name):
        def check(_module, _inputs, output):
            if not first:
                for tensor in tensors(output):
                    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                        first.append(dict(module=name, class_name=type(_module).__name__, dtype=str(tensor.dtype),
                                          shape=list(tensor.shape), nonfinite=int((~torch.isfinite(tensor)).sum().item())))
                        break
        return check

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(hook(name)))
    result = dict(status='not_run', first_nonfinite=None, output_finite=None, error=None)
    try:
        amp = torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            logits = model(x)
        result.update(status='completed', first_nonfinite=first[0] if first else None,
                      output_finite=bool(torch.isfinite(logits).all()))
    except Exception as err:
        result.update(status='exception', first_nonfinite=first[0] if first else None,
                      error=f'{type(err).__name__}: {err}')
    finally:
        for handle in hooks:
            handle.remove()
    return result


def evaluation_loader(corpus, ids, p, device, workers=2, k=24):
    """Exact EXP120 runner validation dataset, batch size, order, collate and seed."""
    import numpy as np
    from torch.utils.data import DataLoader
    a = p['common_args']
    labels = {u: np.zeros(len(LAB), np.float32) for u in ids}
    ds = CenterSliceWindows(corpus.dir, ids, labels, a['res'], k, False, a['norm'], p['seed'],
                            window_policy=a['window_policy'], input_policy=a['input_policy'])
    return DataLoader(ds, batch_size=p['folds']['0']['eval_batch'], shuffle=False, num_workers=min(workers, 2),
                      collate_fn=collate, persistent_workers=False, worker_init_fn=ex.worker_init)


def evaluate_model(model, loader, device):
    import numpy as np
    import torch
    model.eval()
    preds, ids = [], []
    with torch.inference_mode():
        for x, _y, batch_ids in loader:
            logits = model(x.to(device))  # deliberately no autocast
            if logits.shape != (len(batch_ids), len(TARGETS)) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError('FP32 evaluation logits are invalid')
            preds.append(torch.sigmoid(logits.float()).cpu().numpy())
            ids += list(batch_ids)
    return np.concatenate(preds), ids


def evaluate(device='cuda', limit=None, out=OUT, workers=2):
    import torch
    p = ex.read(ex.manifest_path())
    if p.get('schema') != ex.SCHEMA:
        raise ValueError('EXP120 protocol schema changed')
    if ex.sha256(ROOT / ex.RUNNER) != p['source_bindings'][ex.RUNNER]:
        raise ValueError('EXP120 runner binding changed')
    fold = p['folds']['0']
    ids = ex.read(ROOT / fold['ordered_ids']['path'])['val']
    if e113.uid_hash(ids) != fold['ordered_ids']['val_sha256']:
        raise ValueError('validation UID order changed')
    corpus = Corpus(ROOT / p['paths']['corpus'])
    if ex.sha256(corpus.dir / 'manifest.json') != p['corpus_manifest_sha256']:
        raise ValueError('corpus manifest changed')
    if any(u not in corpus for u in ids):
        raise ValueError('validation UID missing from corpus')
    if limit is not None:
        if device != 'cpu' or limit < 1:
            raise ValueError('--limit is CPU-only and positive')
        ids = ids[:limit]
    out = Path(out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    digest = ex.sha256(CKPT)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    checkpoint_contract(ckpt, p)
    configure_runtime('none' if device == 'cpu' else 'fp16', device, policy=p['common_args']['runtime_policy'])
    model = build_model(p['common_args']['arch'], pretrained=False)
    model.load_state_dict(ckpt['model'], strict=True)
    model = model.to(device).eval()
    first_x = next(iter(evaluation_loader(corpus, ids, p, device, workers, p['common_args']['k_eval'])))[0]
    diagnostic = first_nonfinite_module(model, first_x.to(device)) if device == 'cuda' else dict(
        status='skipped_cpu', first_nonfinite=None, output_finite=None, error=None)
    if device == 'cuda' and diagnostic['status'] == 'completed' and diagnostic['output_finite']:
        diagnostic['note'] = 'registered FP16 failure did not reproduce on this first batch'
    t24 = time.monotonic()
    pred24, ids24 = evaluate_model(model, evaluation_loader(corpus, ids, p, device, workers, p['common_args']['k_eval']), device)
    secs24 = time.monotonic() - t24
    t36 = time.monotonic()
    pred36, ids36 = evaluate_model(model, evaluation_loader(corpus, ids, p, device, workers, p['extra_eval']['k_eval']), device)
    secs36 = time.monotonic() - t36
    if ids24 != ids or ids36 != ids:
        raise ValueError('evaluation UID order changed')
    out.mkdir(parents=True)
    write_predictions(out / K24, ids, pred24)
    write_predictions(out / K36, ids, pred36)
    body = dict(schema='exp120-evalfp32-v1', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                source_tag=ex.tag_for(0), source_checkpoint=str(CKPT), checkpoint_sha256=digest,
                protocol_sha256=ex.sha256(ex.manifest_path()), runner_sha256=p['source_bindings'][ex.RUNNER],
                deviation='FP32 evaluation with no autocast instead of registered FP16 evaluation',
                reason='EXP120 trained 20 epochs, saved finite weights, but registered FP16 validation produced non-finite logits',
                diagnostic=diagnostic, device=device, rows=len(ids), k_eval=[p['common_args']['k_eval'], p['extra_eval']['k_eval']],
                eval_batch=fold['eval_batch'], normalization=p['common_args']['norm'], input_policy=p['common_args']['input_policy'],
                window_policy=p['common_args']['window_policy'],
                outputs={K24: ex.sha256(out / K24), K36: ex.sha256(out / K36)},
                timing_seconds=dict(k24=secs24, k36=secs36),
                estimated_five_fold_1300_rtx_seconds=secs36 / len(ids) * 1300 * 5,
                serving_cost='FP16 would be invalid on T4 if this overflow reproduces there; FP32 adds time and memory. RTX FP32 extrapolation is not a T4 runtime measurement.')
    (out / 'provenance.json').write_text(json.dumps(body, indent=2), encoding='utf-8')
    print(json.dumps(body, indent=2))
    return body


def checked_predictions(path, ids):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(path, dtype={UID: str})
    if frame[UID].duplicated().any() or not set(ids) <= set(frame[UID]) or list(frame.columns) != [UID] + TARGETS:  # pooled OOF tables hold all folds
        raise ValueError(f'prediction table mismatch: {path}')
    arr = frame.set_index(UID).loc[ids, TARGETS].to_numpy(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f'nonfinite predictions: {path}')
    return arr


def screen(out=OUT, draws=300):
    import numpy as np
    from tools import exp115_vlm_eval as mr
    out = Path(out)
    prov = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
    if prov['outputs'][K36] != ex.sha256(out / K36) or prov['rows'] != 888:
        raise ValueError('FP32 F0 output incomplete or changed')
    ids = ex.read(ROOT / ex.read(ex.manifest_path())['folds']['0']['ordered_ids']['path'])['val']
    dense = checked_predictions(out / K36, ids)
    rad = checked_predictions(ROOT / 'artifacts/experiments/EXP113_RADR50_LOCAL_F0/oof_k36.csv', ids)
    coat = checked_predictions(ROOT / 'artifacts/experiments/EXP099_B8_DREAD_OOF/oof_k36.csv', ids)
    refs = mr.references(ids, ROOT)
    idx = mr.bootstrap_index(mr.component_groups(ids, ROOT), draws, 20261003)
    pair = (mr.pct_rank(coat) + mr.pct_rank(rad)) / 2
    triple = (mr.pct_rank(coat) + mr.pct_rank(rad) + mr.pct_rank(dense)) / 3
    results = {'dense_minus_rad': {r: mr.compare(dense, rad, refs[r], idx) for r in mr.REFERENCES},
               'triple_minus_coat_rad': {r: mr.compare(triple, pair, refs[r], idx) for r in mr.REFERENCES}}
    body = dict(schema='exp120-evalfp32-screen-v1', rows=len(ids), draws=draws, reference='fold-0 only, matched UIDs',
                method='per-target percentile rank then equal mean; paired component bootstrap', contrasts=results,
                serving_cost=prov['serving_cost'])
    path = out / 'screen.json'
    with path.open('x', encoding='utf-8') as f:
        json.dump(body, f, indent=2, allow_nan=False)
    print(json.dumps({k: {r: v['delta'] for r, v in d.items()} for k, d in results.items()}, indent=2))
    return body


def queue_entries(after):
    cmd = [ex.train_python(), str(ROOT / 'tools/exp120_eval_ckpt.py'), 'evaluate', '--device', 'cuda']
    item = dict(tag=TAG, command=cmd, continue_on_failure=True)
    if after:
        item['after'] = after
    return [item]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    s = sub.add_parser('evaluate'); s.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    s.add_argument('--limit', type=int); s.add_argument('--out', default=str(OUT)); s.add_argument('--workers', type=int, default=2)
    s = sub.add_parser('screen'); s.add_argument('--draws', type=int, default=300)
    s = sub.add_parser('queue-json'); s.add_argument('--after', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.action == 'evaluate':
        evaluate(a.device, a.limit, a.out, a.workers)
    elif a.action == 'screen':
        screen(draws=a.draws)
    else:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf-8') as f:
            json.dump(queue_entries(None if a.after.lower() == 'none' else a.after), f, indent=2)
            f.write('\n')
        print(out)


if __name__ == '__main__':
    main()
