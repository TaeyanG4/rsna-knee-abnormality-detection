"""AUG1 inputs with the frozen EXP129 heads/observers, never a second recipe.

The disabled variant still has AUG1 enabled. POS delegates image generation
to EXP127 and replays centre selection with a NEW private study RNG solely
to attach slot coordinates; affine/gamma/flip cannot change the slice slot.
No global RNG, augmentation draw, image or label changes are introduced.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from raptor_common import select_centers, study_rng, seed_all
from raptor_exp127_aug1 import AugmentedSliceWindows
from raptor_exp129 import (build_variant as _build_variant, slot_positions,
    collate, move_input, model_forward, base_state_hashes, adapter_state_hashes,
    initial_adapter_state, head_equivalence)

VARIANTS = ('POS', 'LATE', 'CTX')


class PositionAugmentedSliceWindows(AugmentedSliceWindows):
    def __getitem__(self, i):
        images, labels, uid = super().__getitem__(i)
        # Replay only sampling, not augmentation; never rewind a shared RNG.
        rng = study_rng(self.seed, int(self.epoch.item()), uid) if self.train else None
        centers = select_centers(self.corpus.mask(uid), self.corpus.depth, self.k,
                                 self.corpus.spec, self.window_policy, rng)
        positions = torch.from_numpy(slot_positions(centers, self.corpus.spec, self.corpus.depth))
        return (images, positions), labels, uid


def loader_factory(variant, enabled=True):
    if variant not in VARIANTS:
        raise ValueError('EXP139 supports POS, LATE, CTX only')
    return PositionAugmentedSliceWindows if enabled and variant == 'POS' else AugmentedSliceWindows


def build_variant(base_model, variant, seed, enabled=True):
    if variant not in VARIANTS:
        raise ValueError('EXP139 supports POS, LATE, CTX only')
    return _build_variant(base_model, variant, seed, enabled=enabled)


def seed_for_device(seed, device):
    if device == 'cuda':
        seed_all(seed)  # preserve the original GPU initialization trajectory
    elif device == 'cpu':
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
    else:
        raise ValueError('unregistered device')


def ordered_equivalence(variant, corpus_dir, train_ids, val_ids, labels, seed, res=224):
    """Tiny ordered CPU loader proof against the augmented control, K12/24/36."""
    from torch.utils.data import DataLoader
    checked = 0
    with torch.device('cpu'):
        for train, ids, k, epochs in ((True, list(train_ids)[:8], 12, (0, 1)),
                                     (False, list(val_ids)[:2], 24, (0,)),
                                     (False, list(val_ids)[:2], 36, (0,))):
            args = (corpus_dir, ids, labels, res, k, train, 'radimagenet', seed)
            kw = dict(window_policy='within-slot-valid', input_policy='center3')
            datasets = [AugmentedSliceWindows(*args, **kw),
                        loader_factory(variant, False)(*args, **kw),
                        loader_factory(variant)(*args, **kw)]
            opts = dict(batch_size=8 if train else 4, shuffle=train, num_workers=0,
                        drop_last=train, collate_fn=collate, pin_memory=False)
            loaders = [DataLoader(ds, generator=torch.Generator(device='cpu').manual_seed(seed), **opts)
                       for ds in datasets]
            for epoch in epochs:
                for ds in datasets:
                    ds.set_epoch(epoch)
                if len(loaders[0]) == 0 or len({len(x) for x in loaders}) != 1:
                    raise AssertionError('empty/different ordered equivalence loaders')
                for a, b, c in zip(*loaders, strict=True):
                    images = c[0][0] if isinstance(c[0], tuple) else c[0]
                    if (a[2] != b[2] or a[2] != c[2] or not torch.equal(a[0], b[0])
                            or not torch.equal(a[0], images) or not torch.equal(a[1], b[1])
                            or not torch.equal(a[1], c[1])):
                        raise AssertionError('variant changed AUG1 inputs/labels/order')
                    if isinstance(c[0], tuple):
                        positions = c[0][1]
                        if positions.shape != (*images.shape[:2], 1) or not torch.isfinite(positions).all():
                            raise AssertionError('invalid POS shape/values')
                        if bool((positions < 0).any()) or bool((positions > 1).any()):
                            raise AssertionError('POS outside slot coordinates')
                    checked += 1
    if torch.cuda.is_initialized():
        raise AssertionError('ordered equivalence initialized CUDA')
    return dict(variant=variant, seed=seed, device='cpu', batches=checked,
                control='EXP127_AUG1', exact_inputs_labels_and_order=True,
                enabled_images_equal_control=True, epochs=[0, 1], train_k=12, eval_k=[24, 36])
