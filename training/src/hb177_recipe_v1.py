"""H-B-177 recipe v1.0.0: ConvNeXt-S optimization, unchanged HB156 geometry/head.

Changelog: new recipe; no parent or serving file edits. See PROTOCOL.md.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/src'))
from hb156_model_v1 import ARCH, build_model as parent_model
from exp143_own0949 import load_imagenet

VERSION = '1.0.0'
RECIPE = dict(arch=ARCH, seed=42, epochs=20, batch=1, accumulation=16,
              bb_lr=2.5e-4, head_lr=1e-3, wd=.02, layer_decay=.8,
              warmup_epochs=2, start_factor=.01, min_factor=.01,
              drop_path=.2, ema_decay=.999, ema_power=2/3,
              clip=3., precision='bf16', grad_checkpointing=True,
              train_k=12, eval_k=94, res=384, inference_res=320,
              workers=4, checkpoint_selection='final-epoch-EMA',
              labels='registered-replacement-soft', norm='imagenet',
              window_policy='legacy', validation='none', oai=False)


def make_model(init=None):
    import timm
    model = parent_model(384, backbone=ARCH, backbone_factory=lambda: timm.create_model(
        ARCH, pretrained=False, num_classes=0, in_chans=3, global_pool='avg',
        drop_path_rate=RECIPE['drop_path']))
    if init is not None:
        load_imagenet(model, init)
    model.backbone.set_grad_checkpointing(RECIPE['grad_checkpointing'])
    return model


def layer_id(name):
    """Official ConvNeXt 3/3/27/3 -> IDs 0..13, translated to timm keys.

    optim_factory.py:get_num_layer_for_convnext groups stage2 in triples;
    timm moves downsample_layers.0 to stem and others inside each stage.
    Reject unrecognized parameters rather than silently misgrouping them.
    """
    if name.startswith('stem.'):
        return 0
    if name.startswith(('head.', 'norm_pre.')):
        return 13
    if name.startswith('stages.'):
        parts = name.split('.')
        stage = int(parts[1])
        if parts[2] == 'downsample':
            return {1: 2, 2: 3, 3: 12}[stage]
        if parts[2] == 'blocks':
            block = int(parts[3])
            return stage + 1 if stage < 2 else (3 + block // 3 if stage == 2 else 12)
    raise ValueError('unregistered backbone parameter: ' + name)


def parameter_groups(model):
    groups = {}
    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            raise ValueError('unexpected frozen parameter')
        assert id(param) not in seen
        seen.add(id(param))
        bb = name.startswith('backbone.')
        layer = layer_id(name[9:]) if bb else 13
        # All norm vectors, gamma/layer-scale and biases are excluded.
        no_decay = param.ndim <= 1 or name.endswith('.bias') or name.endswith('.gamma')
        lr = RECIPE['bb_lr'] * RECIPE['layer_decay'] ** (13-layer) if bb else RECIPE['head_lr']
        key = (bb, layer, no_decay)
        if key not in groups:
            groups[key] = dict(params=[], names=[], lr=lr, peak_lr=lr,
                               weight_decay=0. if no_decay else RECIPE['wd'])
        groups[key]['params'].append(param)
        groups[key]['names'].append(name)
    return list(groups.values())


def lr_factor(update, total, warmup):
    if not 0 <= update < total or not 0 < warmup < total-1:
        raise ValueError('schedule bounds')
    if update < warmup:
        return RECIPE['start_factor'] + (1-RECIPE['start_factor']) * update / warmup
    phase = (update-warmup) / (total-1-warmup)
    return RECIPE['min_factor'] + (1-RECIPE['min_factor']) * .5 * (1+math.cos(math.pi*phase))


def set_lr(optimizer, update, total, warmup):
    factor = lr_factor(update, total, warmup)
    for group in optimizer.param_groups:
        group['lr'] = group['peak_lr'] * factor


def optimization(model):
    import torch
    return torch.optim.AdamW(parameter_groups(model), betas=(.9,.999), eps=1e-8)


class EMA:
    """timm ModelEmaV3 warmup equation; FP32 on model device, exact resume state."""
    def __init__(self, model):
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.updates = 0

    def update(self, model):
        import torch
        self.updates += 1
        decay = min(RECIPE['ema_decay'], 1-(1+self.updates)**(-RECIPE['ema_power']))
        with torch.no_grad():
            for key, value in self.module.state_dict().items():
                source = model.state_dict()[key].detach()
                if value.is_floating_point():
                    value.lerp_(source, 1-decay)
                else:
                    value.copy_(source)

    def state_dict(self):
        return dict(model=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state):
        self.module.load_state_dict(state['model'], strict=True)
        self.updates = state['updates']


def microbatch_step(model, optimizer, ema, batches, loss_fn, device, precision):
    """Identical smoke/fit accumulation, including final partial group."""
    import torch
    from exp143_own0949 import amp_context
    optimizer.zero_grad(set_to_none=True)
    for x, y, _ in batches:
        with amp_context(device, precision):
            loss = loss_fn(model(x.to(device)).float(), y.to(device)) / len(batches)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite loss')
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['clip'], error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema.update(model)


def public_decision(score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('invalid public score')
    return 'ALIVE' if score >= .950 else ('DEAD' if score <= .946 else 'HOLD')
