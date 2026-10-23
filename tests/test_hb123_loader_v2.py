"""v2 regression: execute production loader expressions across persistent epochs."""
import ast
import inspect

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler

from training.arms import mil_laterality_v2 as arm


class SyntheticSides:
    def require(self, ids):
        assert all(0 <= int(uid) < 35 for uid in ids)

    def side(self, uid):
        return ("L", "R", "U")[int(uid) % 3]


class SyntheticCorpus:
    ids = [str(i) for i in range(35)]
    spec = {"slots": [("Sagittal", "x", 1), ("Coronal", "x", 1), ("Axial", "x", 1)]}

    def volume(self, uid):
        return np.arange(12, dtype=np.float32).reshape(3, 2, 2) + int(uid)


class IndexWindows:
    """Spawn-safe index/epoch probe with the actual canonical dataset decorator."""
    def __init__(self):
        self.corpus = SyntheticCorpus()
        self.corpus_dir = "unused-synthetic"
        self.ids = self.corpus.ids
        self.k = 12
        self.epoch = torch.zeros(1, dtype=torch.int64).share_memory_()

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __getitem__(self, index):
        uid = self.ids[index]
        return (torch.from_numpy(self.corpus.volume(uid).copy()),
                self.epoch.clone(), uid)


def production_loader(module, dataset, seed, workers):
    # Extract, do not reimplement, the exact generator and DataLoader calls.
    run = ast.parse(inspect.getsource(module.run)).body[0]
    nodes = [n for n in run.body if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in ("generator", "loader")]
    assert len(nodes) == 2
    scope = dict(module.run.__globals__, dataset=dataset, doc={"seed": seed},
                 batch=8, workers=workers, device="cpu")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "production-loader", "exec"), scope)
    loader = scope["loader"]
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.generator is scope["generator"]
    assert loader.worker_init_fn is arm.shared.worker_init
    assert loader.collate_fn is arm.shared.collate
    assert loader.batch_size == 8 and loader.drop_last
    assert loader.num_workers == workers and loader.persistent_workers == (workers > 0)
    return loader


def epoch_batches(module, seed, workers, canonical):
    module.seed_all(seed)
    dataset = IndexWindows()
    if canonical:
        dataset = arm.lat.CanonicalDataset(dataset, SyntheticSides())
    loader = production_loader(module, dataset, seed, workers)
    streams = []
    try:
        for epoch in range(3):
            dataset.set_epoch(epoch)
            batches = []
            for _, observed_epoch, ids in loader:
                assert observed_epoch.eq(epoch).all()  # shared epoch survives spawn
                batches.append(ids)
            assert len(batches) == 4  # 35 indices, drop the last three
            assert len({uid for batch in batches for uid in batch}) == 32
            streams.append(batches)
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    return streams


@pytest.mark.parametrize("seed", [42, 43])
def test_three_epoch_control_arm_batch_order_and_negative_control(seed):
    torch.set_num_threads(2)
    control = epoch_batches(arm.shared, seed, 2, False)
    canonical = epoch_batches(arm, seed, 2, True)
    legacy = epoch_batches(arm.shared, seed, 0, False)
    assert canonical == control  # full batch boundaries and index streams
    assert legacy[0] == control[0]
    assert all(legacy[e] != control[e] for e in (1, 2))
    assert control[0] != control[1] != control[2]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("workers", [0, 1, 3, True])
def test_v2_worker_drift_fails_before_stage_access(workers):
    with pytest.raises(arm.life.ContractError, match="exactly two persistent workers"):
        arm.run(None, None, None, None, mode="smoke", device="cpu", workers=workers)
