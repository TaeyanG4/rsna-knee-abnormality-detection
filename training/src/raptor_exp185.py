"""EXP185: spatial-token finding MIL on the unchanged EXP179 ConvNeXt parameters."""
from __future__ import annotations

from types import MethodType


def valid_windows(x):
    """Detect the loader's zero-image padding after ImageNet normalization."""
    import torch

    if x.ndim != 5 or x.shape[2] != 3:
        raise ValueError('expected B,K,3,H,W windows')
    mean = x.new_tensor((.485, .456, .406)).view(1, 1, 3, 1, 1)
    std = x.new_tensor((.229, .224, .225)).view(1, 1, 3, 1, 1)
    background = -mean / std
    present = ~torch.isclose(x, background, rtol=0, atol=1e-6).all(dim=(2, 3, 4))
    # Empty studies remain finite; the first padding window is the fixed fallback.
    present[:, 0] |= ~present.any(dim=1)
    return present


def spatial_tokens(model, x):
    """Return B,K*H*W,F tokens and a matching validity mask."""
    if x.ndim != 5:
        raise ValueError('expected B,K,C,H,W')
    b, k = x.shape[:2]
    fmap = model.backbone.forward_features(x.flatten(0, 1))
    features = model.backbone.num_features
    if fmap.ndim != 4 or fmap.shape[1] != features:
        raise ValueError('ConvNeXt NCHW spatial features required')
    # ConvNeXt's global head applies this norm after spatial averaging. For
    # retained tokens apply the *same existing* norm per spatial position.
    fmap = model.backbone.head.norm(fmap)
    _, _, h, w = fmap.shape
    tokens = fmap.permute(0, 2, 3, 1).reshape(b, k * h * w, features)
    mask = valid_windows(x).repeat_interleave(h * w, dim=1)
    return tokens, mask


def finding_attention(model, tokens, mask):
    """Twelve independently normalized token distributions and logits."""
    import torch

    if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('token/mask shape differs')
    if not mask.any(dim=1).all():
        raise ValueError('at least one token per study required')
    h = model.norm(tokens)
    scores = model.att(h).float().masked_fill(~mask[..., None], torch.finfo(torch.float32).min)
    weights = torch.softmax(scores, dim=1).to(h.dtype)
    pooled = torch.einsum('btn,btf->bnf', weights, h)
    logits = (pooled * model.clsW).sum(-1) + model.clsb
    return logits, weights


def collapsed_logits(model, x):
    """Reference path: exact existing global-pool ConvNeXt + per-window MIL."""
    import torch

    b, k = x.shape[:2]
    fmap = model.backbone.forward_features(x.flatten(0, 1))
    feats = model.backbone.forward_head(fmap).reshape(b, k, -1)
    h = model.norm(feats)
    a = torch.softmax(model.att(h), dim=1)
    pooled = torch.einsum('bkn,bkf->bnf', a, h)
    return (pooled * model.clsW).sum(-1) + model.clsb


def install_spatial_pooling(model):
    """Replace only the two pooling methods; keep every state_dict key."""
    if getattr(model, 'pool_policy', None) is not None:
        raise ValueError('pool policy already installed')

    def encode(self, x):
        return spatial_tokens(self, x)

    def head(self, encoded):
        return finding_attention(self, *encoded)[0]

    model.encode = MethodType(encode, model)
    model.head = MethodType(head, model)
    model.pool_policy = 'finding_spatial_tokens_v1'
    return model


def build_offline_model_mil(arch, weights_path, expected_sha256):
    from tools.exp166_convnext_t import build_offline_model
    return_model = build_offline_model(arch, weights_path, expected_sha256)
    model, report = return_model
    return install_spatial_pooling(model), report


def load_checkpoint(path):
    """Explicit serving constructor for an EXP185 checkpoint."""
    import torch
    from raptor_common import build_model

    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('model_kind') != 'finding_spatial_tokens_v1':
        raise ValueError('not an EXP185 spatial-token checkpoint')
    model = install_spatial_pooling(build_model(ck['arch'], pretrained=False))
    model.load_state_dict(ck['model'], strict=True)
    return model, ck
