"""EXP181 v1.0.0 AUG1 for the EXP143 v2 CoAtNet triplet contract (training only).

D-B-066 / EXP127 AUG1 transferred to the 96-slot native corpus: per-window
affine (rotation +-10 deg, scale 0.9-1.1, translation +-5 %), gamma 0.8-1.25,
horizontal flip p=.5 on coronal/axial windows only. Applied after the unchanged
EXP143 intensity jitter and before ImageNet normalization; one transform per
triplet, shared by its three channels. Draws continue the SAME
study_rng(seed, epoch, uid) stream after the frozen centre selection, in a fixed
per-window order (angle, scale, dx, dy, gamma, flip), also for empty windows.
An all-zero (missing-input) triplet stays exactly zero. Evaluation, BN replay
(augment=False) and inference are the frozen EXP143 paths, untouched.
Frozen sources are imported, never edited.
"""
from __future__ import annotations

import numpy as np

from exp143_own0949 import (DEPTH, IMG, K_TRAIN, MEAN, SLOT_SIZES, STD, TRAIN_RES,
                            Corpus, StudyDataset)

VERSION = "1.0.0"
AUG1_SPEC = dict(name="AUG1", angle_deg=[-10.0, 10.0], scale=[0.9, 1.1], translate_frac=[-0.05, 0.05],
                 gamma=[0.8, 1.25], flip_p=0.5, flip_planes=["Coronal", "Axial"],
                 flip_rule="slot index of the centre (np.searchsorted(cumsum(SLOT_SIZES), c, 'right')) >= 2",
                 interpolation="bilinear", fill=0.0, order="jitter -> affine -> gamma -> flip -> normalize",
                 rng="study_rng(seed, epoch, uid) continued after train_centers; draws: angle, scale, dx, dy, gamma, flip",
                 training_only=True)


def plane_slots(centers):
    """Slot index of each triplet centre, the same rule as frozen mirror_triplets."""
    if sum(SLOT_SIZES) != DEPTH:
        raise ValueError("slot sizes do not sum to depth")
    return np.searchsorted(np.cumsum(SLOT_SIZES), np.asarray(centers), side="right")


def train_windows01(vol, mask, seed, epoch, uid):
    """Frozen EXP143 training windows() before normalization, plus the live RNG.

    normalize(result[0]) must equal exp143_own0949.windows(vol, mask, True, seed,
    epoch, uid) bit for bit (tested); centres come from the same first draws.
    """
    from raptor_common import study_rng, train_centers
    if vol.shape != (DEPTH, IMG, IMG) or vol.dtype != np.uint8 or mask.shape != (DEPTH,):
        raise ValueError("EXP181 requires native uint8 96x384x384")
    if not mask.any():
        raise ValueError("empty study cannot enter EXP181")
    rng = study_rng(seed, epoch, uid)
    centers = train_centers(mask, DEPTH, K_TRAIN, rng)
    cs = np.asarray(centers)
    tri = np.asarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])
    low = (IMG - TRAIN_RES) // 2
    x = tri[:, :, low:low + TRAIN_RES, low:low + TRAIN_RES].astype(np.float32) / 255.
    x = np.clip(x * (1 + (study_rng(seed, epoch, uid, "intensity").random() - .5) * .20), 0, 1)
    return x, centers, rng


def normalize(x):
    return np.ascontiguousarray((x - MEAN) / STD)


def augment_triplets(x, centers, rng):
    """AUG1 on (K, 3, H, W) float32 in [0, 1]; returns a new array, same dtype/shape."""
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import affine
    if x.ndim != 4 or x.shape[1] != 3 or x.shape[0] != len(centers) or x.dtype != np.float32:
        raise ValueError("AUG1 expects K x 3 x H x W float32 triplets")
    slots = plane_slots(centers)
    if (slots < 0).any() or (slots >= len(SLOT_SIZES)).any():
        raise ValueError("AUG1 centre outside the 96-slot contract")
    height, width = x.shape[-2:]
    out = np.empty_like(x)
    for j in range(len(centers)):
        # Fixed draw order for every window, including empty ones.
        angle = rng.uniform(-10., 10.)
        scale = rng.uniform(.9, 1.1)
        dx, dy = rng.uniform(-.05, .05) * width, rng.uniform(-.05, .05) * height
        gamma = rng.uniform(.8, 1.25)
        flip = rng.random() < .5
        if not x[j].any():
            out[j] = x[j]  # missing input preserved exactly
            continue
        t = affine(torch.from_numpy(np.ascontiguousarray(x[j])), angle=angle, translate=[dx, dy],
                   scale=scale, shear=[0., 0.], interpolation=InterpolationMode.BILINEAR, fill=0.)
        a = np.power(np.clip(t.numpy(), 0, 1), np.float32(gamma))
        if flip and slots[j] >= 2:
            a = a[..., ::-1]
        out[j] = a
    return out


def aug1_windows(vol, mask, seed, epoch, uid):
    x, centers, rng = train_windows01(vol, mask, seed, epoch, uid)
    x = normalize(augment_triplets(x, centers, rng))
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("invalid EXP181 AUG1 input")
    return x


class Aug1StudyDataset(StudyDataset):
    """Training draws use AUG1; augment=False (SWA BN replay) is the frozen path."""

    def __getitem__(self, i):
        if not self.augment:
            return super().__getitem__(i)
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.path)
        uid, ep = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.get(uid)
        x = aug1_windows(vol, mask, self.seed, ep, uid)
        return torch.from_numpy(x), torch.tensor(self.labels[uid], dtype=torch.float32), uid
