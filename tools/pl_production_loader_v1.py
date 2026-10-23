"""Production loader 1.1.0: spawn workers, bounded queues, epoch-stable shuffle.

MILWindows already hashes (seed, epoch, study, stream) and shares its epoch
tensor. Separate sampler/worker generators prevent persistent-worker resets
from shifting the shuffle stream. No screen imports or sources are changed.
"""
import os
import random
from contextlib import contextmanager

VERSION = '1.1.0'


def resolve_workers(workers=None, profile='local'):
    if profile not in ('local', 'vast'):
        raise ValueError('unknown loader profile')
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    default = min(6, max(0, cpus - 2))
    value = default if workers is None else workers
    if type(value) is not int or not 0 <= value <= max(0, cpus - 1):
        raise ValueError('workers must leave one logical CPU free')
    return value


def seed_worker(_):
    import numpy as np
    import torch
    import cv2
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def memory_plan(workers, prefetch_factor=2, ram_budget_gb=40, batch_size=8):
    if type(prefetch_factor) is not int or not 1 <= prefetch_factor <= 4:
        raise ValueError('prefetch_factor must be 1..4')
    if not 0 < ram_budget_gb <= 40:
        raise ValueError('RAM budget must be positive and <=40 GB')
    # Decimal GB; 12 GiB parent/optimizer/export allowance, 1 GiB per worker,
    # duplicate pageable+pinned queues, working batch, and augmentation scratch.
    batch = batch_size * 12 * 3 * 224 * 224 * 4
    estimate = 12 * 2**30 + workers * 2**30 + (2 * workers * prefetch_factor + 4) * batch
    import psutil
    if estimate > ram_budget_gb * 10**9 or estimate > psutil.virtual_memory().available:
        raise ValueError('loader RAM admission exceeds budget or available RAM')
    return dict(estimated_bytes=estimate, ram_budget_bytes=int(ram_budget_gb*10**9),
                workers=workers, prefetch_factor=prefetch_factor, batch_bytes=batch)


def make_loader(dataset, collate_fn, seed, *, workers=None, profile='local',
                batch_size=8, prefetch_factor=2, ram_budget_gb=40, device='cpu'):
    import torch
    workers = resolve_workers(workers, profile)
    plan = memory_plan(workers, prefetch_factor, ram_budget_gb, batch_size)
    sampler_gen = torch.Generator().manual_seed(seed)
    worker_gen = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(dataset, generator=sampler_gen)
    kwargs = dict(batch_size=batch_size, sampler=sampler, drop_last=True,
                  num_workers=workers, generator=worker_gen, worker_init_fn=seed_worker,
                  collate_fn=collate_fn, pin_memory=device=='cuda', persistent_workers=workers>0)
    if workers:
        kwargs.update(prefetch_factor=prefetch_factor, multiprocessing_context='spawn')
    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader, sampler_gen, dict(plan, version=VERSION, pin_memory=device=='cuda',
           persistent_workers=workers>0, shuffle='independent-generator-seed-plus-epoch',
           augmentation='MILWindows-study-hash-shared-epoch')


@contextmanager
def managed_loader(loader):
    try:
        yield loader
    finally:
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
