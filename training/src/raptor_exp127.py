"""EXP127 loader routing and ordered CPU equivalence against EXP113's loader."""
from __future__ import annotations

from functools import partial

from raptor_radimagenet import CenterSliceWindows
from raptor_exp127_adj3 import AdjacentSliceWindows
from raptor_exp127_aug1 import AugmentedSliceWindows


def loader_factory(variant, enabled=True):
    if variant == 'ADJ3':
        return AdjacentSliceWindows if enabled else partial(AdjacentSliceWindows, input_policy='center3')
    if variant == 'EP40':
        return CenterSliceWindows  # schedule-only intervention; no loader fork
    if variant == 'AUG1':
        return partial(AugmentedSliceWindows, augmentation=enabled)
    raise ValueError('unknown EXP127 variant')


def ordered_equivalence(variant, corpus_dir, train_ids, val_ids, labels, seed, res=224):
    """Tiny actual corpus, 2 training epochs and both eval K. No model/CUDA.

Same generator, shuffle, collate, drop_last, bs and window arguments as EXP113.
Returns counts only; no study IDs, images, report or label rows are logged.
Caller chooses the real protocol's first 8 training and 2 validation studies.
"""
    import torch
    from torch.utils.data import DataLoader
    from raptor_data import collate
    checked, enabled_checked = 0, 0
    with torch.device('cpu'):
        for train, ids, k, epochs in ((True, train_ids[:8], 12, (0, 1)),
                                     (False, val_ids[:2], 24, (0,)),
                                     (False, val_ids[:2], 36, (0,))):
            args = (corpus_dir, ids, labels, res, k, train, 'radimagenet', seed)
            base = CenterSliceWindows(*args, window_policy='within-slot-valid', input_policy='center3')
            other = loader_factory(variant, enabled=False)(*args, window_policy='within-slot-valid', input_policy='center3')
            opts = dict(batch_size=8 if train else 4, shuffle=train, num_workers=0,
                        drop_last=train, collate_fn=collate, pin_memory=False)
            left = DataLoader(base, generator=torch.Generator(device='cpu').manual_seed(seed), **opts)
            right = DataLoader(other, generator=torch.Generator(device='cpu').manual_seed(seed), **opts)
            for epoch in epochs:
                base.set_epoch(epoch)
                other.set_epoch(epoch)
                if len(left) != len(right) or len(left) == 0:
                    raise AssertionError('empty/different ordered equivalence loaders')
                for a, b in zip(left, right, strict=True):
                    if a[2] != b[2] or not torch.equal(a[0], b[0]) or not torch.equal(a[1], b[1]):
                        raise AssertionError('variant-off ordered batch differs from EXP113')
                    if a[0].device.type != 'cpu' or b[0].device.type != 'cpu':
                        raise AssertionError('equivalence must stay on CPU')
                    checked += 1
            # Also exercise the enabled loader on one study per train/eval K.
            policy = 'adjacent3' if variant == 'ADJ3' else 'center3'
            active = loader_factory(variant)(*args, window_policy='within-slot-valid', input_policy=policy)
            x, y, uid = active[0]
            if (x.device.type != 'cpu' or x.dtype != torch.float32 or x.shape != (k, 3, res, res)
                    or not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1
                    or uid != ids[0] or not torch.equal(y, torch.from_numpy(labels[uid]))):
                raise AssertionError('enabled loader input contract failed')
            enabled_checked += 1
    if torch.cuda.is_initialized():
        raise AssertionError('ordered equivalence initialized CUDA')
    return dict(variant=variant, seed=seed, device='cpu', batches=checked,
                exact_inputs_labels_and_order=True, enabled_samples_checked=enabled_checked,
                epochs=[0, 1], train_k=12, eval_k=[24, 36])
