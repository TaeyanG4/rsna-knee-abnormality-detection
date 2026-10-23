"""Candidate-local differentiable CPU position interpolation; passive import."""
from __future__ import annotations
from types import MethodType

POLICY = 'canonical518-trainable-cpu-fp32-bicubic-aa-v2'


def resample_cpu(posemb, new_size, old_size, num_prefix_tokens):
    import torch
    from timm.layers.pos_embed import resample_abs_pos_embed
    if torch.get_num_threads() != 1:
        raise ValueError('Position interpolation requires one internal CPU thread')
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError('Strict deterministic algorithms required, warn_only forbidden')
    if posemb.dtype != torch.float32:
        raise ValueError('Canonical position parameter must remain FP32')
    # .to retains the autograd copy nodes. Never detach or cache between steps.
    with torch.autocast('cpu', enabled=False):
        cpu_pos = posemb.to(device='cpu', dtype=torch.float32)
        result = resample_abs_pos_embed(cpu_pos, new_size=new_size, old_size=old_size,
                                       num_prefix_tokens=num_prefix_tokens,
                                       interpolation='bicubic', antialias=True)
    return result.to(device=posemb.device, dtype=posemb.dtype)


def position_embed_cpu(self, x):
    import torch
    if not self.dynamic_img_size or self.pos_embed is None:
        raise ValueError('EXP086 V2 requires canonical dynamic position embedding')
    B, H, W, C = x.shape
    if (H, W) != (16, 16) or tuple(self.patch_embed.grid_size) != (37, 37):
        raise ValueError('Frozen canonical37 to runtime16 grid required')
    pos_embed = resample_cpu(self.pos_embed, (H, W), self.patch_embed.grid_size,
                            0 if self.no_embed_class else self.num_prefix_tokens)
    x = x.view(B, -1, C)
    to_cat = []
    if self.cls_token is not None:
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
    if self.reg_token is not None:
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
    if self.no_embed_class:
        x = x + pos_embed
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed
    return self.pos_drop(x)


def install(backbone):
    if getattr(backbone, '_exp086_position_policy', None) is not None:
        raise ValueError('Position adapter is already installed')
    if (tuple(backbone.pos_embed.shape) != (1, 1370, 384)
            or not backbone.pos_embed.requires_grad or not backbone.dynamic_img_size
            or tuple(backbone.patch_embed.grid_size) != (37, 37)
            or backbone.num_prefix_tokens != 1 or backbone.no_embed_class):
        raise ValueError('Canonical trainable DINOv2-S position contract differs')
    backbone._pos_embed = MethodType(position_embed_cpu, backbone)
    backbone._exp086_position_policy = POLICY
    return backbone
