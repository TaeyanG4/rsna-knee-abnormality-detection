"""All-position (saturation) evaluation windows for the within-slot-valid policy, without editing raptor_data.py / raptor_common.py.

select_centers(mask, depth, k, spec, "within-slot-valid") with k equal to the number of valid centres returns every valid centre
exactly once (np.linspace(0, n - 1, n).round() is the identity), so "all positions" is the existing evaluation path with a
per-study K. AllValidWindows reuses StudyWindows.__getitem__ unchanged (train=False: no jitter, deterministic) and only sets
self.k per study; use it with batch_size=1 because K differs between studies (padding would add attention mass to empty windows).
A study without any valid centre gets K=1, i.e. one explicit empty (-1) window, as select_centers does for any K.
"""
from __future__ import annotations

import numpy as np

from raptor_common import Corpus
from raptor_data import StudyWindows


def valid_centers(mask, depth, spec, policy="within-slot-valid"):
    """The candidate centre list select_centers draws from (same rule, same order)."""
    if policy not in ("global-valid", "within-slot-valid"):
        raise ValueError(f"unsupported policy for all-position evaluation: {policy}")
    sizes = [int(slot[2]) for slot in spec["slots"]]
    if sum(sizes) != depth or len(mask) != depth:
        raise ValueError("slot sizes / mask do not match corpus depth")
    cs, start = [], 0
    for size in ([depth] if policy == "global-valid" else sizes):
        cs.extend(c for c in range(start + 1, start + size - 1) if np.all(mask[c - 1:c + 2] > 0))
        start += size
    return cs


def all_position_k(mask, depth, spec, policy="within-slot-valid"):
    return max(1, len(valid_centers(mask, depth, spec, policy)))


class AllValidWindows(StudyWindows):
    """Evaluation-only dataset: every valid centre of each study once (variable K; batch_size must be 1)."""

    def __init__(self, corpus_dir, ids, labels, res, norm, seed, window_policy="within-slot-valid"):
        super().__init__(corpus_dir, ids, labels, res, 1, False, norm, seed, window_policy=window_policy)

    def __getitem__(self, i):
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid = self.ids[i]
        self.k = all_position_k(self.corpus.mask(uid), self.corpus.depth, self.corpus.spec, self.window_policy)
        return super().__getitem__(i)
