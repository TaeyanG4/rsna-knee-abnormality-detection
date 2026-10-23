"""Training/evaluation dataset for the native-adjacent triplet cache (H-B-25), without editing raptor_data.py.

NativeTripletWindows is raptor_data.StudyWindows with exactly one line changed: windows_from(vol, centers, res) ->
raptor_native3.windows_from_native3(vol, centers, res). Centres come from select_centers on the 64-entry mask with the same
sha256(seed, epoch, uid) streams, and the intensity jitter / ImageNet normalisation are the same, so on the same seed every epoch
draws the same centres and jitter as the baseline; only the two outer channels of each window come from the native neighbours.
Module-level class: DataLoader workers on Windows are spawned and must unpickle it by reference.
"""
from __future__ import annotations

import numpy as np

from raptor_common import Corpus, select_centers, study_rng
from raptor_data import IMAGENET_MEAN, IMAGENET_STD, StudyWindows
from raptor_native3 import is_native3_spec, windows_from_native3


def open_native3(corpus_dir):
    corpus = Corpus(corpus_dir)
    if not is_native3_spec(corpus.spec):
        raise ValueError(f"{corpus_dir} is not a native3 cache (spec planes / plane_order)")
    return corpus


class NativeTripletWindows(StudyWindows):
    def __getitem__(self, i):
        import torch
        if self.corpus is None:
            self.corpus = open_native3(self.corpus_dir)
        uid, epoch = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        rng = study_rng(self.seed, epoch, uid) if self.train else None
        centers = select_centers(mask, self.corpus.depth, self.k, self.corpus.spec, self.window_policy, rng)
        x = windows_from_native3(vol, centers, self.res)
        if self.train and self.jitter:
            intensity = study_rng(self.seed, epoch, uid, "intensity").random()
            x = np.clip(x * (1 + (intensity - .5) * self.jitter), 0, 1)
        if self.norm == "imagenet":
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
        elif self.norm != "none":
            raise ValueError(f"unknown norm {self.norm}")
        if not np.isfinite(x).all():
            raise ValueError(f"nonfinite input for {uid}")
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid
