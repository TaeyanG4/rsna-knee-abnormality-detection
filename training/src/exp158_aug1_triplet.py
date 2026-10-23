"""EXP158: import AUG1's exact transform while retaining DINO triplet windows."""
from __future__ import annotations

import numpy as np
from raptor_common import Corpus, select_centers, study_rng, windows_from
from raptor_data import StudyWindows, IMAGENET_MEAN, IMAGENET_STD
from raptor_exp127_aug1 import augment_windows


class Aug1TripletWindows(StudyWindows):
    def __getitem__(self, i):
        if not self.train:
            return super().__getitem__(i)
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid, epoch = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        rng = study_rng(self.seed, epoch, uid)
        centers = select_centers(mask, self.corpus.depth, self.k, self.corpus.spec,
                                 self.window_policy, rng)
        x = windows_from(vol, centers, self.res)
        if self.jitter:
            intensity = study_rng(self.seed, epoch, uid, 'intensity').random()
            x = np.clip(x * (1 + (intensity - .5) * self.jitter), 0, 1)
        x = augment_windows(x, centers, rng, self.corpus.spec)
        if self.norm == 'imagenet':
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        elif self.norm != 'none':
            raise ValueError('unknown norm')
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError('invalid EXP158 input')
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid
