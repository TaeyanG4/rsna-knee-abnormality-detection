"""Cross-environment parity probe for the G1 B8 recipe (numerical-parity-skill procedure, steps 1-1 to 1-5 and 3).

  python tools/parity_b8_probe.py --out <dir> --b-ckpt <B224.pt> [--corpus DIR] [--steps 32] [--infer-n 64]

One process, one fixed numerical policy (tools/parity_runtime.configure_parity: fp16 autocast, SDPA mem-efficient
with flash off, strict determinism, 1 CPU thread like the training DataLoader workers, seed 42). Writes:
  runtime.json      runtime snapshot (torch/CUDA/cuDNN/timm/... + policy) for `parity_runtime.py --diff`
  inputs.json       SHA256 of the first --steps training batches (CPU path; must be identical across GPUs)
  infer.npz         B224 checkpoint probabilities on the first --infer-n G1 validation studies (fp16, K24, batch 1)
  train.npz         generic-init state hash, step-1 loss/grad-norm, per-step loss/grad-norm/scale for --steps
                    updates (dropout off so CUDA RNG does not enter), selected weights after step 1 and step N
  summary.json      timings and pass-through metadata
No labels of the validation studies are read; no Gold; nothing is written outside --out.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'training' / 'src')]
ARCH = 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'
WEIGHT_KEYS = ['norm.weight', 'att.0.weight', 'att.2.weight', 'clsW', 'clsb', 'backbone.stem.conv1.weight', 'backbone.head.norm.weight']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True); ap.add_argument('--b-ckpt', required=True)
    ap.add_argument('--corpus', default=str(ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'))
    ap.add_argument('--labels', default=str(ROOT / 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'))
    ap.add_argument('--ordered-ids', default=str(ROOT / 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json'))
    ap.add_argument('--protocol', default=str(ROOT / 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'))
    ap.add_argument('--steps', type=int, default=32); ap.add_argument('--infer-n', type=int, default=64)
    ap.add_argument('--sdpa', default='efficient', choices=['efficient', 'math', 'flash', 'keep'])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    from parity_runtime import configure_parity
    info = configure_parity(amp='fp16', sdpa=a.sdpa, strict=True, threads=1, seed=42)
    import torch as _t
    # Training parity needs the hard mode: with warn_only=True (the skill helper's default) PyTorch keeps the
    # non-deterministic memory-efficient SDPA backward; with warn_only=False it switches to the deterministic one
    # (same mode as training/src/raptor_runtime.configure_runtime strict).
    _t.use_deterministic_algorithms(True, warn_only=False)
    info['policy']['deterministic_warn_only'] = False
    (out / 'runtime.json').write_text(json.dumps(info, indent=2, default=str), encoding='utf-8')
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from raptor_common import build_model, load_checkpoint, load_label_csv, LAB
    from raptor_data import StudyWindows, collate
    t0 = time.time(); summary = {'policy': info['policy'], 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    ordered = json.loads(Path(a.ordered_ids).read_text(encoding='utf-8'))
    train_ids, val_ids = ordered['train'], ordered['val']
    labels, label_hash = load_label_csv(Path(a.labels))
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(12, np.float32) for u in val_ids}}
    summary['label_hash'] = label_hash
    # --- 1-3: deterministic CPU inputs (same order a seeded DataLoader produces with any worker count)
    tds = StudyWindows(Path(a.corpus), train_ids, all_labels, 224, 12, True, 'imagenet', 42, window_policy='within-slot-valid')
    tl = DataLoader(tds, batch_size=8, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate, generator=torch.Generator().manual_seed(42))
    batches, input_hashes = [], []
    for i, (x, y, uids) in enumerate(tl):
        input_hashes.append({'step': i + 1, 'uids_sha256': hashlib.sha256('\n'.join(uids).encode()).hexdigest(),
                             'x_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest(), 'y_sha256': hashlib.sha256(y.numpy().tobytes()).hexdigest()})
        batches.append((x, y))
        if len(batches) == a.steps:
            break
    (out / 'inputs.json').write_text(json.dumps(input_hashes, indent=1), encoding='utf-8')
    summary['input_seconds'] = round(time.time() - t0, 1)
    # --- 1-4/1-5: inference parity with the frozen B224 checkpoint (fp16 autocast, K24, batch 1)
    t1 = time.time()
    model, meta = load_checkpoint(Path(a.b_ckpt)); model = model.cuda().eval()
    vds = StudyWindows(Path(a.corpus), val_ids[:a.infer_n], all_labels, 224, 24, False, 'imagenet', 42, window_policy='within-slot-valid')
    probs, logits_all = [], []
    with torch.inference_mode():
        for i in range(len(vds)):
            x, _, _ = vds[i]
            with torch.autocast('cuda', dtype=torch.float16):
                lg = model(x[None].cuda())
            logits_all.append(lg.float().cpu().numpy()[0]); probs.append(torch.sigmoid(lg.float()).cpu().numpy()[0])
    np.savez(out / 'infer.npz', probs=np.stack(probs), logits=np.stack(logits_all), ids=np.array(val_ids[:a.infer_n]))
    summary['infer_seconds'] = round(time.time() - t1, 1); summary['b_ckpt_sha256'] = hashlib.sha256(Path(a.b_ckpt).read_bytes()).hexdigest()
    del model; torch.cuda.empty_cache()
    # --- 3: training parity from the generic initial state, dropout off
    t2 = time.time()
    torch.manual_seed(42)
    model = build_model(ARCH, pretrained=True)
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    init_sha = h.hexdigest()
    model = model.cuda().train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if hasattr(m, 'drop_prob'):
            m.drop_prob = 0.0
    model.backbone.set_grad_checkpointing(True)
    protocol = json.loads(Path(a.protocol).read_text(encoding='utf-8'))
    pos_weight = torch.tensor(protocol['loss']['baseline_pos_weight'], dtype=torch.float32, device='cuda')
    head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]; bb = [p for n, p in model.named_parameters() if n.startswith('backbone.')]
    opt = torch.optim.AdamW([{'params': bb, 'lr': 3e-5}, {'params': head, 'lr': 1e-3}], weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 1e-3], total_steps=3448, pct_start=0.15)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    traj, saved = [], {}
    for i, (x, y) in enumerate(batches, start=1):
        x, y = x.cuda(), y.cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            loss = lossf(model(x).float(), y)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        before = scaler.get_scale(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if scaler.get_scale() >= before:
            sched.step()
        traj.append({'step': i, 'loss': float(loss.item()), 'grad_norm': float(gn.item()), 'scale': float(before)})
        if i in (1, len(batches)):
            sd = model.state_dict()
            for k in WEIGHT_KEYS:
                if k in sd:
                    saved[f'step{i}__{k}'] = sd[k].detach().float().cpu().numpy()
    np.savez(out / 'train.npz', loss=np.array([t['loss'] for t in traj]), grad_norm=np.array([t['grad_norm'] for t in traj]),
             scale=np.array([t['scale'] for t in traj]), **saved)
    summary.update(train_seconds=round(time.time() - t2, 1), init_state_sha256=init_sha, trajectory=traj, total_seconds=round(time.time() - t0, 1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('torch', 'gpu', 'init_state_sha256', 'total_seconds')}), flush=True)


if __name__ == '__main__':
    main()
