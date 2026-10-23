"""EXP179: nominal 140/100-mm center3 views, without changing RNG streams."""
from __future__ import annotations

import numpy as np
from raptor_common import Corpus, select_centers, study_rng
from raptor_radimagenet import CenterSliceWindows, center3_windows_from, normalise
from raptor_exp127_aug1 import augment_windows


def detail_indices(k):
    if k not in (12, 24, 36):
        raise ValueError('EXP179 requires registered even K12/K24/K36')
    # Alternating positions in the UNCHANGED selected-centre list; no RNG draws.
    return np.arange(1, k, 2)


def scale_windows(vol, centers, res, policy='mixed'):
    if vol.shape != (64, 336, 336) or vol.dtype != np.uint8 or res != 224:
        raise ValueError('EXP179 requires uint8 RAPTOR64 336px -> 224px')
    if policy not in ('context', 'mixed'):
        raise ValueError('unknown crop policy')
    positions = detail_indices(len(centers))
    if policy == 'context':
        return center3_windows_from(vol, centers, res)
    # Every selected centre is encoded exactly once. Do not resize context
    # windows then crop them: detail uses the original 336px cache directly.
    out = np.empty((len(centers), 3, res, res), np.float32)
    context = np.arange(0, len(centers), 2)
    out[context] = center3_windows_from(vol, [centers[j] for j in context], res)
    out[positions] = center3_windows_from(vol[:, 48:288, 48:288],
                                        [centers[j] for j in positions], res)
    return out


class ScaleSliceWindows(CenterSliceWindows):
    def __init__(self, *args, crop_policy='mixed', **kwargs):
        super().__init__(*args, **kwargs)
        if crop_policy not in ('context', 'mixed'):
            raise ValueError('unknown crop policy')
        self.crop_policy = crop_policy

    def __getitem__(self, i):
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid, epoch = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        rng = study_rng(self.seed, epoch, uid) if self.train else None
        centers = select_centers(mask, self.corpus.depth, self.k,
                                 self.corpus.spec, self.window_policy, rng)
        x = scale_windows(vol, centers, self.res, self.crop_policy)
        if self.train:
            if self.jitter:
                intensity = study_rng(self.seed, epoch, uid, 'intensity').random()
                x = np.clip(x * (1 + (intensity - .5) * self.jitter), 0, 1)
            x = augment_windows(x, centers, rng, self.corpus.spec)
        x = normalise(x, self.norm)
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('invalid EXP179 input')
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid
