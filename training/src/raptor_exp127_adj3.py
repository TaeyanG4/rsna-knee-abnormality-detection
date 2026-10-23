"""EXP127 ADJ3: adjacent *cached slots*, never native-DICOM neighbours.

Window centres, sampler, brightness and normalisation stay those of EXP113.
Channels are [previous, centre, next], confined to one corpus series slot.
At an edge or an invalid neighbour repeat the valid centre, never cross a slot.
The registered within-slot-valid selector normally selects three valid slots,
so the edge fallback is defensive. input_policy='center3' disables the change.
"""
from __future__ import annotations

import numpy as np

from raptor_common import Corpus, select_centers, study_rng
from raptor_radimagenet import CenterSliceWindows, normalise


def adjacent_indices(centers, mask, spec, depth):
    sizes = [int(s[2]) for s in spec['slots']]
    if sum(sizes) != depth or len(mask) != depth:
        raise ValueError('slot/mask depth mismatch')
    bounds, lo = [], 0
    for size in sizes:
        bounds.append((lo, lo + size))
        lo += size
    result = []
    for c in centers:
        c = int(c)
        if c == -1:
            result.append((-1, -1, -1))
            continue
        if not 0 <= c < depth or not mask[c] > 0:
            raise ValueError('invalid adjacent3 centre')
        lo, hi = next((lo, hi) for lo, hi in bounds if lo <= c < hi)
        prev, nxt = max(lo, c - 1), min(hi - 1, c + 1)
        result.append((prev if mask[prev] > 0 else c, c, nxt if mask[nxt] > 0 else c))
    return result


def adjacent3_windows_from(vol, centers, mask, spec, res):
    import torch
    import torch.nn.functional as F
    out = np.empty((len(centers), 3, res, res), np.float32)
    for j, triplet in enumerate(adjacent_indices(centers, mask, spec, len(vol))):
        if triplet[1] == -1:
            out[j] = 0
            continue
        tri = np.stack([vol[c] for c in triplet]).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)  # CPU, never .cuda() or an implicit device
        if t.shape[-2:] != (res, res):
            t = F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False)[0]
        out[j] = t.numpy()
    return out


class AdjacentSliceWindows(CenterSliceWindows):
    def __init__(self, *args, input_policy='adjacent3', **kwargs):
        if input_policy not in ('center3', 'adjacent3'):
            raise ValueError('unknown EXP127 input policy')
        super().__init__(*args, input_policy='center3', **kwargs)
        self.input_policy = input_policy

    def __getitem__(self, i):
        if self.input_policy == 'center3':
            return super().__getitem__(i)
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid, epoch = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        rng = study_rng(self.seed, epoch, uid) if self.train else None
        centers = select_centers(mask, self.corpus.depth, self.k, self.corpus.spec, self.window_policy, rng)
        x = adjacent3_windows_from(vol, centers, mask, self.corpus.spec, self.res)
        if self.train and self.jitter:
            intensity = study_rng(self.seed, epoch, uid, 'intensity').random()
            x = np.clip(x * (1 + (intensity - .5) * self.jitter), 0, 1)
        x = normalise(x, self.norm)
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('invalid adjacent3 input')
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid
