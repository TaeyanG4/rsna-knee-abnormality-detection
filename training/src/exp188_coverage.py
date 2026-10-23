"""EXP188 training-only centre allocation on the frozen EXP143 CoAtNet recipe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from exp143_own0949 import Corpus, DEPTH, IMG, K_TRAIN, MEAN, SLOT_SIZES, STD, TRAIN_RES, StudyDataset
from raptor_common import study_rng

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT / "artifacts/experiments/EXP188_COATNET_COVERAGE_F0/protocol-v1/coverage-metadata.json"
def eligible_centres(mask):
    """Legacy K12 candidate population: mask-positive interior centres."""
    mask = np.asarray(mask)
    if mask.ndim != 1 or len(mask) < 3 or not np.isin(mask, (0, 1)).all():
        raise ValueError("invalid binary mask")
    return np.flatnonzero(mask[1:-1]).astype(np.int64) + 1


def independent_groups(mask, slots, eligible=None):
    """Nonfallback, complete in-slot triplets in five fixed cache slots."""
    mask = np.asarray(mask)
    eligible = eligible_centres(mask) if eligible is None else np.asarray(eligible, dtype=np.int64)
    if len(slots) != 5 or tuple(int(s["stop"]) - int(s["start"]) for s in slots) != tuple(SLOT_SIZES):
        raise ValueError("EXP143 slot geometry differs")
    groups = []
    for slot in slots:
        start, stop = int(slot["start"]), int(slot["stop"])
        if slot.get("series") is None or slot.get("fallback"):
            continue
        valid = eligible[(eligible > start) & (eligible < stop - 1)]
        valid = valid[np.asarray([bool(mask[c - 1:c + 2].all()) for c in valid], dtype=bool)]
        if len(valid):
            groups.append((slot["group"], valid))
    return groups


def coverage_centres(mask, slots, k, rng):
    """One uniform centre per available group, then uniform remaining legacy centres."""
    if k < 0:
        raise ValueError("K must be nonnegative")
    eligible = eligible_centres(mask)
    groups = independent_groups(mask, slots, eligible)
    if len(groups) > k:
        indices = sorted(rng.sample(range(len(groups)), k))
        groups = [groups[int(i)] for i in indices]
    selected = [int(rng.choice(centres)) for _, centres in groups]
    remaining = eligible[~np.isin(eligible, selected)]
    fill = min(k - len(selected), len(remaining))
    if fill:
        selected.extend(int(c) for c in rng.sample(remaining.tolist(), fill))
    if len(selected) > 1:
        rng.shuffle(selected)
    # A rare underfilled bag uses empty windows; nonnegative centres remain unique.
    return selected + [-1] * (k - len(selected))


def metadata_mask_sha(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def coverage_windows(vol, mask, slots, seed, epoch, uid):
    if vol.shape != (DEPTH, IMG, IMG) or vol.dtype != np.uint8 or mask.shape != (DEPTH,):
        raise ValueError("EXP188 requires native uint8 96x384x384")
    if not mask.any():
        raise ValueError("empty training study")
    rng = study_rng(seed, epoch, uid)
    centers = coverage_centres(mask, slots, K_TRAIN, rng)
    # Match EXP181's copy of the frozen EXP143 train-window preprocessing.
    cs = np.asarray(centers, dtype=np.int64)
    valid = cs >= 0
    safe = cs.copy()
    safe[~valid] = 1
    tri = np.asarray(vol[np.stack([safe - 1, safe, safe + 1], axis=1)])
    tri[~valid] = 0
    low = (IMG - TRAIN_RES) // 2
    x = tri[:, :, low:low + TRAIN_RES, low:low + TRAIN_RES].astype(np.float32) / 255.
    x = np.clip(x * (1 + (study_rng(seed, epoch, uid, "intensity").random() - .5) * .20), 0, 1)
    x[~valid] = 0
    return np.ascontiguousarray((x - MEAN) / STD)


class CoverageStudyDataset(StudyDataset):
    """Only augment=True training draws change; BN/evaluation use frozen StudyDataset."""

    _rows = None

    @classmethod
    def rows(cls):
        if cls._rows is None:
            cls._rows = json.loads(METADATA.read_text(encoding="utf-8"))["studies"]
        return cls._rows

    def __getitem__(self, i):
        if not self.augment:
            return super().__getitem__(i)
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.path)
        uid, ep = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.get(uid)
        row = self.rows()[uid]
        if metadata_mask_sha(mask) != row["mask_sha256"]:
            raise ValueError("EXP188 study mask differs from registered provenance")
        x = coverage_windows(vol, mask, row["slots"], self.seed, ep, uid)
        return torch.from_numpy(x), torch.tensor(self.labels[uid], dtype=torch.float32), uid
