"""H-C-17 isolated feature reuse for EXP113/117 centre-slice mirror TTA.

This is preparation code, not an admitted serving path. The caller binds the
registered centre3 checkpoint, window order and EXP124 plane mask. Only the
encoder work is shared; both complete bags still pass through their own heads.
No cross-checkpoint cache, window deduplication, sigmoid or ranking lives here.

GPU numerical parity and speed must be measured before integration: encoding a
smaller mirrored batch can select different floating-point kernels. The window
counts returned here do not measure time saved.
"""
from __future__ import annotations

import torch


def _heads(model, features):
    if callable(getattr(model, "head", None)):
        return model.head(features)
    heads = getattr(model, "heads", None)
    if heads is None or not len(heads):
        raise ValueError("expected a Raptor head or nonempty source-head list")
    return torch.stack([head(features) for head in heads], dim=1)


@torch.inference_mode()
def mirror_pair_logits(model, windows, flip_flags, *, input_policy):
    """Return (plain logits, mirrored logits, window counts), without ranking.

    windows: [B,K,3,H,W], already normalized.
    flip_flags: bool [B,K]; True ONLY for the fixed coronal/axial transform.
    False entries (including empty sentinels) are copied unchanged, in place.

    For EXP117, apply sigmoid then rank each head over the *entire fold/study
    cohort* for each view separately, mean the heads, then mean the two views.
    Ranking inside a minibatch or averaging features changes the estimator.
    """
    if input_policy != "center3":
        raise ValueError("only the registered centre3 input contract is supported")
    if windows.ndim != 5 or windows.shape[2] != 3 or min(windows.shape) < 1:
        raise ValueError("expected nonempty [B,K,3,H,W] windows")
    if not windows.is_floating_point():
        raise ValueError("expected normalized floating-point windows")
    flags = torch.as_tensor(flip_flags, device=windows.device)
    if flags.dtype != torch.bool or flags.shape != windows.shape[:2]:
        raise ValueError("flip mask must be bool [B,K], preserving each bag's order")
    if not callable(getattr(model, "encode", None)):
        raise ValueError("model must expose the registered independent-window encoder")
    if any(module.training for module in model.modules()):
        raise ValueError("all model modules must be in eval mode")
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and not module.track_running_stats:
            raise ValueError("batch-dependent normalization cannot reuse packed features")

    b, k = windows.shape[:2]
    features = model.encode(windows)
    if features.ndim != 3 or features.shape[:2] != (b, k):
        raise ValueError("encoder must return [B,K,F]")
    # Flatten and scatter in the original B,K order, even with different masks
    # per study, repeated centres or empty sentinels. No window is removed.
    changed = flags.flatten().nonzero(as_tuple=False).flatten()
    flipped_features = features.clone().reshape(b * k, -1)
    if changed.numel():
        selected = windows.flatten(0, 1).index_select(0, changed)
        mirrored = torch.flip(selected, dims=(-1,)).unsqueeze(0)
        replacement = model.encode(mirrored)
        if replacement.shape != (1, changed.numel(), features.shape[-1]):
            raise ValueError("packed encoder feature shape changed")
        flipped_features.index_copy_(0, changed, replacement[0])
    flipped_features = flipped_features.reshape_as(features)
    plain_logits = _heads(model, features)
    mirror_logits = _heads(model, flipped_features)
    if plain_logits.shape != mirror_logits.shape or not (
        torch.isfinite(plain_logits).all() and torch.isfinite(mirror_logits).all()
    ):
        raise ValueError("invalid TTA logits")
    return plain_logits, mirror_logits, {
        "naive_windows": 2 * b * k,
        "encoded_windows": b * k + changed.numel(),
        "reused_windows": b * k - changed.numel(),
    }
