"""EXP127 AUG1: one preregistered augmentation-stack intervention.

After unchanged brightness jitter and before 2x-1: per-window affine, gamma,
then p=.5 horizontal flip on coronal/axial only. One draw stream: the existing
study_rng(seed, epoch, uid, 'windows'), continued *after* centre selection.
No global numpy/torch random calls. All channels share a window's transform.
No augmentation in evaluation. Plane names do not prove DICOM orientation.
"""
from __future__ import annotations

import numpy as np

from raptor_common import Corpus, SLOTS64, select_centers, study_rng
from raptor_radimagenet import CenterSliceWindows, center3_windows_from, normalise


def augment_windows(x, centers, rng, spec):
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import affine
    if [tuple(s) for s in spec['slots']] != SLOTS64 or x.shape[0] != len(centers):
        raise ValueError('AUG1 requires registered 64-slot plane layout')
    out = np.empty_like(x)
    height, width = x.shape[-2:]
    for j, c in enumerate(centers):
        # Fixed draw order, including sagittal/empty windows, for reproducibility.
        angle = rng.uniform(-10., 10.)
        scale = rng.uniform(.9, 1.1)
        dx, dy = rng.uniform(-.05, .05) * width, rng.uniform(-.05, .05) * height
        gamma = rng.uniform(.8, 1.25)
        flip = rng.random() < .5
        if c == -1:
            out[j] = x[j]
            continue
        if not 0 <= c < 64:
            raise ValueError('AUG1 centre out of range')
        t = affine(torch.from_numpy(x[j]), angle=angle, translate=[dx, dy], scale=scale,
                   shear=[0., 0.], interpolation=InterpolationMode.BILINEAR, fill=0.)
        a = np.power(np.clip(t.numpy(), 0, 1), np.float32(gamma))
        if flip and 32 <= c < 64:
            a = a[..., ::-1]
        out[j] = a
    return out


class AugmentedSliceWindows(CenterSliceWindows):
    def __init__(self, *args, augmentation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation = bool(augmentation)

    def __getitem__(self, i):
        if not self.train or not self.augmentation:
            return super().__getitem__(i)
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid, epoch = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        rng = study_rng(self.seed, epoch, uid)
        centers = select_centers(mask, self.corpus.depth, self.k, self.corpus.spec, self.window_policy, rng)
        x = center3_windows_from(vol, centers, self.res)
        if self.jitter:
            intensity = study_rng(self.seed, epoch, uid, 'intensity').random()
            x = np.clip(x * (1 + (intensity - .5) * self.jitter), 0, 1)
        x = normalise(augment_windows(x, centers, rng, self.corpus.spec), self.norm)
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('invalid AUG1 input')
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid
