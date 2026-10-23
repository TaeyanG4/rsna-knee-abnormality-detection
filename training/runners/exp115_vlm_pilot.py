"""EXP115 (H-B-43): Qwen3.5-2B vision-language model + LoRA, fold-0 pilot (protocol: tools/exp115_vlm_pilot.py).

Data: the CoAtNet K24 evaluation windows (raptor_common.select_centers within-slot-valid on RAPTOR64_336_C140_S02, windows_from 336 -> 384)
packed as 12 two-frame temporal patches per study (slot-aware pairing, Qwen3-VL patchify order, grid [1,24,24], 144 LM tokens each). Training
samples k_train windows per study per epoch with the B8 sampler (study_rng(seed, epoch, uid)) plus the B8 intensity jitter, sorted before
pairing. Normalisation is the Qwen processor's (mean/std 0.5 from preprocessor_config.json).
Model: transformers Qwen3_5Model (frozen bf16 language model + peft LoRA r16 on its linear layers, fully trainable fp32 vision tower) and a
12-logit head on [masked mean of the image-token hidden states ++ last-token hidden state]. BCE with the Dread fold-0 pos_weight, bf16 autocast,
gradient checkpointing, micro-batch 1 + accumulation, AdamW (vit/lora/head groups), warmup + cosine. Weights are saved every epoch; the
fixed-final model writes fold-0 predictions oof_k24.csv (primary) and oof_k36.csv (diagnostic) in the oof_k36.csv format of EXP099.
No metric is computed here (tools/exp115_vlm_eval.py scores and applies the stop rule).

  --mode full    the registered run (protocol-bound, CUDA, artifacts/experiments/EXP115_VLM_PILOT_F0); queue only
  --mode smoke   tiny CPU/GPU check into --out (subset of studies, optional truncated layer counts, --max-steps)
  --validate-only  checks protocol, membership, labels, pos_weight, prompt and one packed study on CPU, then exits
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

REGISTERED_ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1'}
SUPPLIED_ENV = []
if __name__ == '__main__':
    # Before numpy/torch load their thread pools: a queue runner may lack these (D-B-017); explicit different values are kept and the full-mode
    # guard refuses them.
    for _k, _v in REGISTERED_ENV.items():
        if _k not in os.environ:
            os.environ[_k] = _v; SUPPLIED_ENV.append(_k)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
UID = 'StudyInstanceUID'
LAB = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
       'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
LORA_TARGET = (r'^language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)'
               r'|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|mlp\.(gate_proj|up_proj|down_proj))$')
PROMPT_TEXT = ("Knee MRI study shown as {n} image pairs; each pair holds two adjacent three-slice windows of one series, ordered "
               "sagittal, coronal, axial. Assess ACL tear, MCL tear, medial meniscus tear, lateral meniscus tear, medial, lateral and "
               "patellofemoral osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion and fracture.")
EXIT_GATE_FAILED = 3


# ====================================================================== pure helpers (no torch at import)
def slot_sizes(spec):
    sizes = [int(s[2]) for s in spec['slots']]
    if sum(sizes) != int(spec['depth']):
        raise ValueError('slot sizes do not match corpus depth')
    return sizes


def slot_of(center, sizes):
    """Slot index of a cached plane index; -1 for an explicit empty window."""
    if center < 0:
        return -1
    start = 0
    for s, size in enumerate(sizes):
        if center < start + size:
            return s
        start += size
    raise ValueError(f'center {center} outside depth {start}')


def pair_indices(centers, sizes):
    """Slot-aware pairing of window positions (centers must be ascending, -1 = empty, duplicates allowed).

    Consecutive windows of the same slot form a pair; the odd leftover of a slot is paired with the next odd leftover in slot order (a
    leftover without partner is paired with itself, as Qwen duplicates a single image). Pairs come back in the order of their first member.
    Returns (pairs, number of cross-slot pairs).
    """
    centers = [int(c) for c in centers]
    n_real = sum(c >= 0 for c in centers)
    if centers[:n_real] != sorted(c for c in centers if c >= 0) or any(c != -1 for c in centers[n_real:]):
        raise ValueError('centers must be ascending with empty windows (-1) last')
    pos = list(range(len(centers)))
    pairs, leftovers = [], []
    for _, grp in itertools.groupby(pos, key=lambda i: slot_of(centers[i], sizes)):
        grp = list(grp)
        pairs += [(grp[j], grp[j + 1]) for j in range(0, len(grp) - 1, 2)]
        if len(grp) % 2:
            leftovers.append(grp[-1])
    cross = 0
    for j in range(0, len(leftovers), 2):
        a = leftovers[j]; b = leftovers[j + 1] if j + 1 < len(leftovers) else a
        cross += int(b != a and slot_of(centers[a], sizes) != slot_of(centers[b], sizes))
        pairs.append((a, b))
    pairs.sort(key=lambda p: p[0])
    return pairs, cross


def study_centers(mask, depth, spec, k, train, seed, epoch, uid):
    """The B8 window choice: within-slot-valid; train = study_rng sample of k (sorted here for pairing), eval = evenly spaced k."""
    from raptor_common import select_centers, study_rng
    rng = study_rng(seed, epoch, uid) if train else None
    cs = select_centers(mask, depth, k, spec, 'within-slot-valid', rng)
    real = sorted(c for c in cs if c >= 0)
    return real + [-1] * (len(cs) - len(real))


def lr_lambda_factory(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return f


def accumulation_groups(n, accum):
    """Size of the accumulation group each micro-step belongs to (the last group of an epoch may be smaller)."""
    return [min(accum, n - (i // accum) * accum) for i in range(n)]


def build_prompt(tokenizer, n_images, tokens_per_image, ids):
    """ChatML user turn with n_images vision blocks; each <|image_pad|> expanded to tokens_per_image. Returns (input_ids, mm_types, text)."""
    text = ('<|im_start|>user\n' + '<|vision_start|><|image_pad|><|vision_end|>' * n_images + PROMPT_TEXT.format(n=n_images)
            + '<|im_end|>\n<|im_start|>assistant\n')
    raw = tokenizer(text, add_special_tokens=False)['input_ids']
    if raw.count(ids['image']) != n_images or raw.count(ids['vision_start']) != n_images or raw.count(ids['vision_end']) != n_images:
        raise ValueError('tokenizer did not keep the vision special tokens intact')
    out = []
    for t in raw:
        out.extend([t] * tokens_per_image if t == ids['image'] else [t])
    types = [1 if t == ids['image'] else 0 for t in out]
    return out, types, text


def write_predictions(path, ids, probs):
    """Same layout as EXP099 oof_k36.csv: StudyInstanceUID + the 12 targets, one row per study in the given order."""
    import pandas as pd
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (len(ids), len(LAB)) or not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1:
        raise ValueError('invalid prediction matrix')
    df = pd.DataFrame(probs.astype(np.float32), columns=LAB); df.insert(0, UID, list(ids))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()


# ====================================================================== torch pieces
def patchify_pairs(frames, patch=16, merge=2):
    """(P, 2, C, H, W) float -> (P*gh*gw, C*2*patch*patch) in the Qwen3-VL video patchify order, plus grid (P, 3) = [1, gh, gw]."""
    import torch
    P, T, C, H, W = frames.shape
    if T != 2 or H % (patch * merge) or W % (patch * merge):
        raise ValueError(f'bad frame block {tuple(frames.shape)}')
    gh, gw = H // patch, W // patch
    x = frames.reshape(P, 1, T, C, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flat = x.reshape(P * gh * gw, C * T * patch * patch)
    grid = torch.tensor([[1, gh, gw]] * P, dtype=torch.long)
    return flat, grid


class PairStudies:
    """Map-style dataset: one study -> packed pixel values for its frame pairs (+ label, uid, pairing stats). Picklable for workers."""

    def __init__(self, corpus_dir, ids, labels, k, res, train, seed, mean, std, jitter=0.2, patch=16, merge=2):
        import torch
        self.corpus_dir, self.ids, self.labels, self.k, self.res = str(corpus_dir), list(ids), labels, int(k), int(res)
        self.train, self.seed, self.jitter, self.patch, self.merge = bool(train), int(seed), float(jitter), patch, merge
        self.mean = np.asarray(mean, np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(std, np.float32).reshape(1, 3, 1, 1)
        self.corpus = None
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self.epoch.fill_(int(epoch))

    def __len__(self):
        return len(self.ids)

    def _corpus(self):
        if self.corpus is None:
            from raptor_common import Corpus
            self.corpus = Corpus(Path(self.corpus_dir))
        return self.corpus

    def frames(self, uid, epoch):
        from raptor_common import study_rng, windows_from
        c = self._corpus()
        centers = study_centers(c.mask(uid), c.depth, c.spec, self.k, self.train, self.seed, epoch, uid)
        x = windows_from(c.volume(uid), centers, self.res)                      # (K,3,res,res) in [0,1]
        if self.train and self.jitter:
            u = study_rng(self.seed, epoch, uid, 'intensity').random()
            x = np.clip(x * (1 + (u - .5) * self.jitter), 0, 1)
        x = (x - self.mean) / self.std
        pairs, cross = pair_indices(centers, slot_sizes(c.spec))
        f = np.stack([np.stack([x[a], x[b]]) for a, b in pairs])               # (P,2,3,res,res)
        if not np.isfinite(f).all():
            raise ValueError(f'nonfinite input for {uid}')
        return f.astype(np.float32), centers, pairs, cross

    def __getitem__(self, i):
        import torch
        uid = self.ids[i]
        f, centers, pairs, cross = self.frames(uid, int(self.epoch.item()))
        pv, grid = patchify_pairs(torch.from_numpy(f), self.patch, self.merge)
        y = torch.as_tensor(self.labels[uid], dtype=torch.float32)
        return dict(pixel_values=pv, grid=grid, y=y, uid=uid, cross=cross, empty=sum(c < 0 for c in centers))


def first_item(batch):
    return batch[0]


def worker_init(_):
    import torch
    torch.set_num_threads(1)


def make_head(hidden, n_out=12, dropout=0.0):
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(2 * hidden)
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(2 * hidden, n_out)
            nn.init.trunc_normal_(self.fc.weight, std=0.01); nn.init.zeros_(self.fc.bias)  # initial logit std ~0.64, as the B8 head (0.02 x sqrt(1024))

        def forward(self, pooled):
            return self.fc(self.drop(self.norm(pooled)))
    return Head()


def make_classifier(core, head):
    import torch
    import torch.nn as nn

    class VLMClassifier(nn.Module):
        """core: Qwen3_5Model (+LoRA). Logits (B, 12) from [mean over image-token states ++ last-token state], head in fp32."""

        def __init__(self):
            super().__init__()
            self.core, self.head = core, head

        def forward(self, input_ids, mm_token_type_ids, pixel_values, image_grid_thw, amp_dtype=None):
            dev = input_ids.device.type
            with torch.autocast(dev, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                h = self.core(input_ids=input_ids, mm_token_type_ids=mm_token_type_ids, pixel_values=pixel_values,
                              image_grid_thw=image_grid_thw, use_cache=False).last_hidden_state
            h = h.float()
            m = (mm_token_type_ids == 1).unsqueeze(-1).to(h.dtype)
            visual = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
            pooled = torch.cat([visual, h[:, -1]], dim=-1)
            with torch.autocast(dev, enabled=False):
                return self.head(pooled)
    return VLMClassifier()


def build_model(weights, *, text_layers=0, vision_layers=0, base_dtype='bfloat16', lora_r=16, lora_alpha=32, lora_dropout=0.05,
                grad_ckpt=True, head_dropout=0.0, attn='sdpa'):
    """Qwen3_5Model from the local snapshot (+ optional truncation for smoke tests), LoRA on the language model, trainable fp32 vision tower."""
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    from peft import LoraConfig, inject_adapter_in_model
    dtype = getattr(torch, base_dtype)
    cfg = AutoConfig.from_pretrained(weights)
    full_text, full_vis = cfg.text_config.num_hidden_layers, cfg.vision_config.depth
    if text_layers:
        cfg.text_config.num_hidden_layers = int(text_layers); cfg.text_config.layer_types = list(cfg.text_config.layer_types[:int(text_layers)])
    if vision_layers:
        cfg.vision_config.depth = int(vision_layers)
    from transformers.utils import logging as hf_logging
    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # the load report table is replaced by the explicit key checks below
    try:
        core, info = Qwen3_5Model.from_pretrained(weights, config=cfg, dtype=dtype, attn_implementation=attn, output_loading_info=True)
    finally:
        hf_logging.set_verbosity(verbosity)
    missing = sorted(info.get('missing_keys', []))
    if missing or info.get('mismatched_keys'):
        raise RuntimeError(f'weights do not cover the model: missing {missing[:5]}, mismatched {list(info.get("mismatched_keys", []))[:5]}')
    unexpected = sorted(info.get('unexpected_keys', []))
    bad = [k for k in unexpected if not unexpected_key_ok(k, cfg.text_config.num_hidden_layers, cfg.vision_config.depth)]
    if bad:
        raise RuntimeError(f'unexpected checkpoint keys for this model: {bad[:5]}')
    core.requires_grad_(False)
    core.visual.to(torch.float32).requires_grad_(True)
    lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=LORA_TARGET, bias='none')
    core = inject_adapter_in_model(lcfg, core)
    core.visual.requires_grad_(True)  # peft marks only adapters trainable; the vision tower is fine-tuned jointly (D-B-013)
    lora_params = 0
    for n, p in core.named_parameters():
        if 'lora_' in n:
            p.data = p.data.float(); p.requires_grad_(True); lora_params += p.numel()
    lora_modules = sorted({n.rsplit('.lora_', 1)[0] for n, _ in core.named_parameters() if 'lora_' in n})
    if not lora_modules or any(not m.startswith('language_model.') for m in lora_modules):
        raise RuntimeError('LoRA must attach to language-model linear layers only')
    wrong = [n for n, p in core.named_parameters() if p.requires_grad != (n.startswith('visual.') or 'lora_' in n)]
    if wrong:
        raise RuntimeError(f'trainable set is not exactly vision tower + LoRA: {wrong[:3]}')
    if any(p.dtype != torch.float32 for n, p in core.named_parameters() if p.requires_grad):
        raise RuntimeError('trainable parameters must be fp32')
    if grad_ckpt:
        core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    head = make_head(cfg.text_config.hidden_size, len(LAB), head_dropout)
    model = make_classifier(core, head)
    ckpt_layers = sum(1 for m in core.modules() if getattr(m, 'gradient_checkpointing', False))
    report = dict(text_layers=cfg.text_config.num_hidden_layers, text_layers_full=full_text, vision_layers=cfg.vision_config.depth,
                  vision_layers_full=full_vis, base_dtype=base_dtype, unexpected_keys=len(unexpected),
                  params_total=sum(p.numel() for p in model.parameters()),
                  params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  params_vision=sum(p.numel() for p in core.visual.parameters()), params_lora=lora_params, lora_modules=len(lora_modules),
                  params_head=sum(p.numel() for p in head.parameters()), grad_ckpt_modules=ckpt_layers,
                  lora=dict(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target=LORA_TARGET))
    return model, cfg, report


def unexpected_key_ok(key, n_text, n_vision):
    """Checkpoint keys the classifier core may skip: the MTP head, the tied lm_head, and layers beyond a smoke-test truncation."""
    import re
    k = key.removeprefix('model.')
    if key.startswith(('mtp.', 'lm_head.')) or k.startswith(('mtp.', 'lm_head.')):
        return True
    m = re.match(r'^language_model\.layers\.(\d+)\.', k)
    if m:
        return int(m.group(1)) >= n_text
    m = re.match(r'^visual\.blocks\.(\d+)\.', k)
    if m:
        return int(m.group(1)) >= n_vision
    return False


def trainable_state(model):
    return {n: p.detach().to('cpu', dtype=p.dtype).clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    """Strict: every trainable parameter must be in `state` with the same shape, and nothing else."""
    import torch
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(state):
        raise RuntimeError(f'trainable parameter names differ: missing {sorted(set(params) - set(state))[:3]}, extra {sorted(set(state) - set(params))[:3]}')
    with torch.no_grad():
        for n, p in params.items():
            if tuple(p.shape) != tuple(state[n].shape):
                raise RuntimeError('shape mismatch ' + n)
            p.copy_(state[n].to(p.device, p.dtype))


def token_ids(cfg):
    return dict(image=int(cfg.image_token_id), vision_start=int(cfg.vision_start_token_id), vision_end=int(cfg.vision_end_token_id))


def processor_stats(weights):
    pc = json.loads((Path(weights) / 'preprocessor_config.json').read_text(encoding='utf-8'))
    return pc['image_mean'], pc['image_std'], int(pc['patch_size']), int(pc['merge_size']), int(pc['temporal_patch_size'])


def environment(device):
    info = dict(python=sys.version.split()[0], platform=platform.platform(), hostname=platform.node(), time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    for mod in ('torch', 'transformers', 'peft', 'numpy', 'safetensors', 'tokenizers'):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    import torch
    info.update(cuda=torch.version.cuda)
    if device == 'cuda':
        info.update(cudnn=torch.backends.cudnn.version())
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memory_gib=round(p.total_memory / 2**30, 2), bf16=torch.cuda.is_bf16_supported())
    import importlib.util
    info['fla_available'] = importlib.util.find_spec('fla') is not None
    info['causal_conv1d_available'] = importlib.util.find_spec('causal_conv1d') is not None
    return info


def gate_decision(step_times, gate_steps, n_train, epochs, n_val, k_eval, k_eval_extra, eval_fraction, peak_gib, max_hours, max_peak_gib):
    """G0 projection from measured micro-step times (warm-up third discarded)."""
    window = step_times[gate_steps // 3:gate_steps]
    if not window:
        raise ValueError('no gate window')
    step = float(np.mean(window))
    extra = (k_eval_extra / k_eval) if k_eval_extra else 0.0
    train_s = step * n_train * epochs
    eval_s = step * eval_fraction * n_val * (1.0 + extra)
    hours = (train_s + eval_s) / 3600
    ok = hours <= max_hours and peak_gib <= max_peak_gib
    return dict(status='GATE_PASSED' if ok else 'GATE_FAILED', mean_micro_step_seconds=step, window=[gate_steps // 3, gate_steps],
                projected_train_hours=train_s / 3600, projected_eval_hours=eval_s / 3600, projected_total_hours=hours,
                projected_epoch_hours=step * n_train / 3600, peak_allocated_gib=peak_gib, max_hours=max_hours, max_peak_gib=max_peak_gib,
                eval_fraction_assumption=eval_fraction)


# ====================================================================== CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    B = argparse.BooleanOptionalAction
    ap.add_argument('--mode', choices=['full', 'smoke'], required=True)
    ap.add_argument('--protocol')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', help='smoke only: output directory')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--weights', default=str(ROOT / 'storage/weights/qwen__Qwen3.5-2B'))
    ap.add_argument('--corpus', default=str(ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'))
    ap.add_argument('--labels', default=str(ROOT / 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'))
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--k-train', type=int, default=24)
    ap.add_argument('--k-eval', type=int, default=24)
    ap.add_argument('--k-eval-extra', type=int, default=36, help='0 disables the diagnostic second evaluation')
    ap.add_argument('--res', type=int, default=384)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr-vit', type=float, default=2e-5)
    ap.add_argument('--lr-lora', type=float, default=2e-4)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--warmup-frac', type=float, default=0.03)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--amp', choices=['bf16', 'none'], default='bf16')
    ap.add_argument('--eval-amp', choices=['bf16', 'none'], default='bf16')
    ap.add_argument('--grad-ckpt', action=B, default=True)
    ap.add_argument('--jitter', type=float, default=0.2)
    ap.add_argument('--head-dropout', type=float, default=0.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--gate-steps', type=int, default=48, help='0 disables the G0 gate')
    ap.add_argument('--gate-max-hours', type=float, default=4.0)
    ap.add_argument('--gate-max-peak-gib', type=float, default=14.5)
    ap.add_argument('--gate-eval-fraction', type=float, default=0.35)
    ap.add_argument('--text-layers', type=int, default=0, help='smoke only: truncate the language model (0 = all 24)')
    ap.add_argument('--vision-layers', type=int, default=0, help='smoke only: truncate the vision tower (0 = all 24)')
    ap.add_argument('--max-train', type=int, default=0, help='smoke only: first N training studies')
    ap.add_argument('--max-val', type=int, default=0, help='smoke only: first N validation studies')
    ap.add_argument('--max-steps', type=int, default=0, help='smoke only: micro-steps per epoch cap')
    ap.add_argument('--save-weights', action=B, default=True)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'training' / 'src'))
    from tools import exp115_vlm_pilot as guard
    supplied_env = list(SUPPLIED_ENV) + (guard.supply_registered_env() if a.mode == 'full' else [])
    if a.mode == 'full':
        if not a.protocol:
            raise SystemExit('--protocol required in full mode')
        manifest, train_ids, val_ids = guard.validate_arguments(a, ROOT, hash_weights=True)
        out = ROOT / 'artifacts' / 'experiments' / a.tag
        guard_install_gold_blocker()
    else:
        if not a.out:
            raise SystemExit('smoke needs --out')
        manifest = None
        train_ids, val_ids = guard.fold_ids(ROOT)
        if a.max_train:
            train_ids = train_ids[:a.max_train]
        if a.max_val:
            val_ids = val_ids[:a.max_val]
        out = Path(a.out)
    from raptor_common import Corpus, load_label_csv
    corpus = Corpus(Path(a.corpus))
    labels, label_hash = load_label_csv(Path(a.labels))
    if any(u not in labels for u in train_ids):
        raise SystemExit('labels do not cover the training ids')
    if any(u not in corpus for u in train_ids + val_ids):
        raise SystemExit('corpus does not cover the fold ids')
    if a.mode == 'full':
        if label_hash != manifest['labels']['semantic_sha256']:
            raise SystemExit('Dread semantic hash mismatch')
        guard.validate_membership(ROOT, manifest, corpus.ids.astype(str).tolist(), train_ids, val_ids)
    from tools import exp099_kfold as k99
    pos_weight_np = k99.pos_weight(labels, train_ids)
    if a.mode == 'full' and not np.array_equal(pos_weight_np, np.asarray(manifest['labels']['pos_weight'], np.float32)):
        raise SystemExit('pos_weight differs from the registered EXP099 fold-0 vector')
    all_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(len(LAB), np.float32) for u in val_ids}}

    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')) if a.device == 'cpu' else 1)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.weights)
    from transformers import AutoConfig
    cfg_probe = AutoConfig.from_pretrained(a.weights)
    mean, std, patch, merge, tps = processor_stats(a.weights)
    if tps != 2 or a.res % (patch * merge):
        raise SystemExit('unsupported processor geometry')
    tokens_per_image = (a.res // patch // merge) ** 2
    ids = token_ids(cfg_probe)
    prompts = {}
    for k in sorted({a.k_train, a.k_eval} | ({a.k_eval_extra} if a.k_eval_extra else set())):
        n_pairs = (k + 1) // 2
        p_ids, p_types, p_text = build_prompt(tokenizer, n_pairs, tokens_per_image, ids)
        prompts[k] = (torch.tensor([p_ids]), torch.tensor([p_types]), p_text)
    ds_kw = dict(res=a.res, seed=a.seed, mean=mean, std=std, patch=patch, merge=merge)
    tds = PairStudies(corpus.dir, train_ids, all_labels, a.k_train, train=True, jitter=a.jitter, **ds_kw)

    if a.validate_only:
        s_tr, s_va = tds[0], PairStudies(corpus.dir, val_ids, all_labels, a.k_eval, train=False, **ds_kw)[0]
        if torch.cuda.is_initialized():
            raise SystemExit('--validate-only touched CUDA')
        print(json.dumps(dict(status='VALIDATED_NO_TRAINING', mode=a.mode, tag=a.tag, train=len(train_ids), val=len(val_ids),
                              pos_weight=[round(float(x), 6) for x in pos_weight_np], label_semantic_sha256=label_hash,
                              supplied_env=supplied_env, prompt_tokens={k: int(v[0].shape[1]) for k, v in prompts.items()},
                              train_sample=dict(pixel_values=list(s_tr['pixel_values'].shape), grid=s_tr['grid'][0].tolist(), pairs=int(s_tr['grid'].shape[0]), cross=s_tr['cross']),
                              val_sample=dict(pixel_values=list(s_va['pixel_values'].shape), pairs=int(s_va['grid'].shape[0]), cross=s_va['cross']),
                              output_exists=out.exists() and any(out.iterdir())), indent=1))
        return 0

    dev = a.device
    if dev == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available')
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Refusing to overwrite existing run: {out}')
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / 'train.log', 'a', encoding='utf-8')

    def log(msg):
        print(msg, flush=True); log_file.write(msg + '\n'); log_file.flush()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t_build = time.time()
    model, cfg, report = build_model(a.weights, text_layers=a.text_layers, vision_layers=a.vision_layers,
                                     base_dtype='bfloat16' if a.amp == 'bf16' else 'float32', lora_r=a.lora_r, lora_alpha=a.lora_alpha,
                                     lora_dropout=a.lora_dropout, grad_ckpt=a.grad_ckpt, head_dropout=a.head_dropout)
    model.to(dev)
    report['build_seconds'] = round(time.time() - t_build, 1)
    log(f'{a.tag} | mode {a.mode} | fold {a.fold} | train {len(train_ids)} | val {len(val_ids)} | device {dev} | model {json.dumps(report)}')
    amp_dtype = torch.bfloat16 if a.amp == 'bf16' else None
    eval_dtype = torch.bfloat16 if a.eval_amp == 'bf16' else None
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32, device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    core = model.core
    groups = dict(vit=[p for n, p in core.named_parameters() if p.requires_grad and n.startswith('visual.')],
                  lora=[p for n, p in core.named_parameters() if p.requires_grad and 'lora_' in n],
                  head=list(model.head.parameters()))
    n_tr = sum(len(v) for v in groups.values())
    if n_tr != sum(1 for p in model.parameters() if p.requires_grad):
        raise RuntimeError('optimizer groups do not cover every trainable parameter')
    opt = torch.optim.AdamW([dict(params=groups['vit'], lr=a.lr_vit), dict(params=groups['lora'], lr=a.lr_lora),
                             dict(params=groups['head'], lr=a.lr_head)], weight_decay=a.wd)
    micro_per_epoch = min(len(train_ids), a.max_steps) if a.max_steps else len(train_ids)
    group_size = accumulation_groups(micro_per_epoch, a.accum)
    updates_per_epoch = -(-micro_per_epoch // a.accum)
    total_updates = updates_per_epoch * a.epochs
    warmup = max(1, int(round(a.warmup_frac * total_updates)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(warmup, total_updates))
    gen = torch.Generator().manual_seed(a.seed)
    from torch.utils.data import DataLoader
    tl = DataLoader(tds, batch_size=1, shuffle=True, generator=gen, num_workers=a.workers, collate_fn=first_item,
                    persistent_workers=a.workers > 0, worker_init_fn=worker_init, pin_memory=dev == 'cuda')
    provenance = dict(plan_item='H-B-43', tag=a.tag, mode=a.mode, args=vars(a), environment=environment(dev), model=report,
                      runner_sha256=sha256_file(Path(__file__)), guard_sha256=sha256_file(Path(guard.__file__)),
                      protocol_sha256=sha256_file(Path(a.protocol)) if a.protocol else None, label_semantic_sha256=label_hash,
                      train_ids_ordered_sha256=uid_hash(train_ids), eval_ids_ordered_sha256=uid_hash(val_ids),
                      pos_weight=pos_weight_np.tolist(), prompt_sha256={k: hashlib.sha256(v[2].encode()).hexdigest() for k, v in prompts.items()},
                      prompt_tokens={k: int(v[0].shape[1]) for k, v in prompts.items()}, tokens_per_image=tokens_per_image,
                      processor=dict(mean=mean, std=std, patch=patch, merge=merge, temporal_patch=tps),
                      updates_per_epoch=updates_per_epoch, total_updates=total_updates, warmup_updates=warmup, supplied_env=supplied_env,
                      checkpoint_selection='fixed-final', metric_access='none (tools/exp115_vlm_eval.py)')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=1), encoding='utf-8')

    def batch_to(b, k):
        p_ids, p_types, _ = prompts[k]
        return (p_ids.to(dev), p_types.to(dev), b['pixel_values'].to(dev, non_blocking=True), b['grid'].to(dev))

    def peak_gib():
        return torch.cuda.max_memory_allocated() / 2**30 if dev == 'cuda' else 0.0

    def rss_peak_gib():
        try:
            import psutil
            mi = psutil.Process().memory_info()
            return round(getattr(mi, 'peak_wset', mi.rss) / 2**30, 2)
        except Exception:
            return None

    history, step_times, updates, gate = [], [], 0, None
    cross_pairs, empty_windows = 0, 0
    t_start = time.time()
    for ep in range(a.epochs):
        tds.set_epoch(ep); model.train(); opt.zero_grad(set_to_none=True)
        if dev == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        tot, n_seen, t0, wait_s, t_last = 0.0, 0, time.time(), 0.0, time.monotonic()
        for i, b in enumerate(tl):
            if i >= micro_per_epoch:
                break
            t_got = time.monotonic(); wait_s += t_got - t_last; t_cycle = t_last
            ids_t, types_t, pv, grid = batch_to(b, a.k_train)
            logits = model(ids_t, types_t, pv, grid, amp_dtype=amp_dtype)
            y = b['y'].to(dev).view(1, -1)
            if logits.shape != (1, len(LAB)) or not torch.isfinite(logits).all():
                raise RuntimeError(f'nonfinite or invalid training logits at epoch {ep} step {i} ({b["uid"]})')
            loss = lossf(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite training loss at epoch {ep} step {i}')
            (loss / group_size[i]).backward()
            if (i + 1) % a.accum == 0 or i + 1 == micro_per_epoch:
                gnorm = torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], a.clip, error_if_nonfinite=True)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); updates += 1
            tot += float(loss.item()); n_seen += 1; cross_pairs += b['cross']; empty_windows += b['empty']
            if dev == 'cuda':
                torch.cuda.synchronize()
            t_last = time.monotonic(); step_times.append(t_last - t_cycle)  # full cycle incl. loader wait
            if ep == 0 and a.gate_steps and dev == 'cuda' and i + 1 == a.gate_steps:
                gate = gate_decision(step_times, a.gate_steps, len(train_ids), a.epochs, len(val_ids), a.k_eval, a.k_eval_extra,
                                     a.gate_eval_fraction, peak_gib(), a.gate_max_hours, a.gate_max_peak_gib)
                gate.update(reserved_gib=torch.cuda.max_memory_reserved() / 2**30, observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                (out / 'gate-receipt.json').write_text(json.dumps(gate, indent=1), encoding='utf-8')
                log(f"G0 {gate['status']}: {gate['mean_micro_step_seconds']:.2f}s/micro-step, projected {gate['projected_total_hours']:.2f} h "
                    f"(epoch {gate['projected_epoch_hours']:.2f} h), peak {gate['peak_allocated_gib']:.2f} GiB")
                if gate['status'] != 'GATE_PASSED':
                    log('stopping at the G0 gate (registered; no retry)'); log_file.close()
                    return EXIT_GATE_FAILED
            if (i + 1) % 100 == 0:
                rate = (time.time() - t0) / (i + 1)
                log(f'  ep{ep + 1} step {i + 1}/{micro_per_epoch} loss {tot / n_seen:.4f} | {rate:.2f}s/step | epoch ETA {rate * (micro_per_epoch - i - 1) / 60:.0f} min '
                    f'| peak {peak_gib():.2f} GiB | loader wait {wait_s:.0f}s')
        rec = dict(epoch=ep + 1, loss=tot / max(1, n_seen), micro_steps=n_seen, optimizer_updates=updates, epoch_seconds=round(time.time() - t0, 1),
                   mean_micro_step_seconds=float(np.mean(step_times[-n_seen:])) if n_seen else None, loader_wait_seconds=round(wait_s, 1),
                   peak_allocated_gib=round(peak_gib(), 3), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3) if dev == 'cuda' else 0.0,
                   process_peak_rss_gib=rss_peak_gib(), lr=[g['lr'] for g in opt.param_groups])
        if a.save_weights:
            t_save = time.time()
            torch.save(dict(trainable=trainable_state(model), epoch=ep + 1, tag=a.tag, report=report, args=vars(a), lab=LAB,
                            prompt_text={k: v[2] for k, v in prompts.items()}, lora_target=LORA_TARGET), out / f'weights_epoch{ep + 1}.pt')
            rec['save_seconds'] = round(time.time() - t_save, 1)
        history.append(rec)
        (out / 'history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
        log(f"epoch {ep + 1}: loss {rec['loss']:.4f} | {rec['epoch_seconds']}s | peak {rec['peak_allocated_gib']} GiB alloc / {rec['peak_reserved_gib']} GiB reserved")
    del tl  # stops persistent loader workers before evaluation
    if updates != total_updates:
        raise RuntimeError(f'update count {updates} != {total_updates}')

    @torch.no_grad()
    def evaluate(ids_list, k):
        model.eval()
        ds = PairStudies(corpus.dir, ids_list, all_labels, k, train=False, **ds_kw)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=min(a.workers, 2), collate_fn=first_item, worker_init_fn=worker_init)
        P, U = [], []
        for b in dl:
            logits = model(*batch_to(b, k), amp_dtype=eval_dtype)
            if logits.shape != (1, len(LAB)) or not torch.isfinite(logits).all():
                raise RuntimeError(f"invalid evaluation logits for {b['uid']}")
            P.append(torch.sigmoid(logits.float()).cpu().numpy()[0]); U.append(b['uid'])
        return np.stack(P), U

    evals = {}
    for k, name in ((a.k_eval, f'oof_k{a.k_eval}.csv'), (a.k_eval_extra, f'oof_k{a.k_eval_extra}.csv')):
        if not k:
            continue
        t_eval = time.time()
        P, U = evaluate(val_ids, k)
        if U != list(val_ids):
            raise RuntimeError('prediction UID order changed')
        write_predictions(out / name, U, P)
        evals[name] = dict(k=k, pairs=(k + 1) // 2, rows=len(U), seconds=round(time.time() - t_eval, 1),
                           seconds_per_study=round((time.time() - t_eval) / max(1, len(U)), 3))
        log(f'{name}: {len(U)} rows in {evals[name]["seconds"]}s')
    summary = dict(status='COMPLETED', tag=a.tag, mode=a.mode, train=len(train_ids), val=len(val_ids), epochs=a.epochs, optimizer_updates=updates,
                   history=history, gate=gate, evaluations=evals, total_seconds=round(time.time() - t_start, 1),
                   peak_allocated_gib=max((h['peak_allocated_gib'] for h in history), default=0.0),
                   cross_slot_pairs_per_study=cross_pairs / max(1, sum(h['micro_steps'] for h in history)),
                   empty_windows_per_study=empty_windows / max(1, sum(h['micro_steps'] for h in history)),
                   selected_checkpoint=f'weights_epoch{a.epochs}.pt (fixed-final)', primary_predictions=f'oof_k{a.k_eval}.csv',
                   cuda_initialized=bool(torch.cuda.is_initialized()), model=report,
                   caveat='one fold, one seed, weak Dread labels; scores only via tools/exp115_vlm_eval.py')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.name not in ('train.log', 'output-inventory.json'))
    (out / 'output-inventory.json').write_text(json.dumps({n: dict(sha256=sha256_file(out / n), bytes=(out / n).stat().st_size) for n in names}, indent=1),
                                               encoding='utf-8')
    log(f"done in {summary['total_seconds']}s -> {out}"); log_file.close()
    return 0


def guard_install_gold_blocker():
    from tools.exp084_group_loss import install_gold_blocker
    install_gold_blocker(ROOT)


if __name__ == '__main__':
    sys.exit(main())
