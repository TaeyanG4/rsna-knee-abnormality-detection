"""EXP192 v1.0.0: fixed-call, alternating 140/100-mm CoAtNet windows.

Changelog: 1.0.0 adds one shared transform for train, BN replay and OOF.
No augmentation, centre selection, model or RNG-stream changes beyond scale.
"""
from __future__ import annotations

import numpy as np
from exp143_own0949 import (
    Corpus, StudyDataset, DEPTH, IMG, TRAIN_RES, EVAL_RES, K_TRAIN, K_EVAL,
    MEAN, STD, mirror_triplets, amp_context,
)
from raptor_common import train_centers, eval_centers, study_rng

VERSION = "1.0.0"
DETAIL_SIZE = round(IMG * 100 / 140)  # 274 px = nominal 99.8958 mm
DETAIL_LOW = (IMG - DETAIL_SIZE) // 2  # [55:329], exactly centred
CROP_SPEC = dict(name="alternating-context-detail", version=VERSION,
    training_only=False, nominal_mm=[140, 100], detail_slice=[55, 329],
    allocation="even=context,odd=detail; position in selected-centre list",
    interpolation="torch bilinear align_corners=False antialias=False, float32 /255 before resize",
    train_k=12, eval_k=94, train_res=384, eval_res=320,
    bn="same K12 mixed windows, deterministic centres, no intensity jitter",
    mirror="frozen anatomical mirror before resize; allocation unchanged")


def scale_windows(vol, centers, res, mirror=False):
    """Each centre encoded once; parity resets per study, never per chunk/plane."""
    import torch
    import torch.nn.functional as F
    cs = np.asarray(centers)
    if (vol.shape != (DEPTH, IMG, IMG) or vol.dtype != np.uint8
            or cs.ndim != 1 or len(cs) not in (K_TRAIN, K_EVAL)
            or cs.dtype.kind not in "iu" or ((cs < 1) | (cs > DEPTH - 2)).any()
            or res not in (TRAIN_RES, EVAL_RES)):
        raise ValueError("EXP192 requires native uint8 cache, K12/K94 valid centres and 384/320 output")
    tri = vol[np.stack([cs - 1, cs, cs + 1], 1)]
    if mirror:
        tri = mirror_triplets(tri, cs)
    out = np.empty((len(cs), 3, res, res), np.float32)
    # Resize per window, as EXP179 does. Bounded intermediate memory in workers.
    for j, window in enumerate(tri):
        if j % 2:
            window = window[:, DETAIL_LOW:DETAIL_LOW + DETAIL_SIZE,
                            DETAIL_LOW:DETAIL_LOW + DETAIL_SIZE]
        x = torch.from_numpy(window.astype(np.float32) / 255.)
        if x.shape[-1] != res:
            x = F.interpolate(x[None], size=(res, res), mode="bilinear",
                              align_corners=False, antialias=False)[0]
        out[j] = x.numpy()
    return out


def windows(vol, mask, train=False, seed=42, epoch=0, uid="x", mirror=False, jitter=True):
    if mask.shape != (DEPTH,) or not mask.any():
        raise ValueError("EXP192 requires a nonempty native mask")
    if train and mirror:
        raise ValueError("mirror is inference only")
    centers = (train_centers(mask, DEPTH, K_TRAIN, study_rng(seed, epoch, uid)) if train
               else eval_centers(mask, DEPTH, K_EVAL))
    x = scale_windows(vol, centers, TRAIN_RES if train else EVAL_RES, mirror)
    if train and jitter:
        x = np.clip(x * (1 + (study_rng(seed, epoch, uid, "intensity").random() - .5) * .20), 0, 1)
    return np.ascontiguousarray((x - MEAN) / STD)


class CropStudyDataset(StudyDataset):
    def __getitem__(self, i):
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.path)
        uid, ep = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.get(uid)
        # BN still draws K12 at the bridge's epoch16, with jitter disabled.
        x = windows(vol, mask, True, self.seed, ep, uid, jitter=self.augment)
        return torch.from_numpy(x), torch.tensor(self.labels[uid], dtype=torch.float32), uid


def infer_pair(model, vol, mask, device="cpu", precision="none", chunk=16):
    """EXP181 inference loop with only its window transform replaced."""
    import torch
    if any(m.training for m in model.modules()):
        raise ValueError("inference requires eval mode")
    if chunk < 1:
        raise ValueError("positive chunk required")
    probs = []
    with torch.inference_mode():
        for mirror in (False, True):
            x = torch.from_numpy(windows(vol, mask, mirror=mirror))
            with amp_context(device, precision):
                feats = torch.cat([model.backbone(z.to(device)) for z in x.split(chunk)], 0)
                logits = model.head(feats.unsqueeze(0)).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite prediction; no fallback")
            probs.append(logits.sigmoid()[0].cpu().numpy())
    return probs[0], probs[1]
