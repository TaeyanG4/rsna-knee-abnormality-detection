"""Side-effect-free dataset adapter shared by training and CPU contract tests."""
import numpy as np
from raptor_common import Corpus, select_centers, study_rng, windows_from

IMAGENET_MEAN = np.array([.485, .456, .406], np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([.229, .224, .225], np.float32).reshape(3, 1, 1)


class StudyWindows:
    def __init__(self, corpus_dir, ids, labels, res, k, train, norm, seed, jitter=.20,
                 window_policy="legacy"):
        import torch
        self.corpus_dir, self.ids, self.labels, self.res, self.k = corpus_dir, list(ids), labels, res, k
        self.train, self.norm, self.seed, self.jitter = train, norm, seed, jitter
        self.window_policy = window_policy
        self.corpus = None
        # Shared tensor makes set_epoch visible to persistent DataLoader workers.
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid, epoch = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        rng = study_rng(self.seed, epoch, uid) if self.train else None
        centers = select_centers(mask, self.corpus.depth, self.k, self.corpus.spec, self.window_policy, rng)
        x = windows_from(vol, centers, self.res)
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


def collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]
