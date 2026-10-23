"""EXP129 CPU-safe loader/head interventions; original Raptor state keys survive.

POS changes only the MIL logits. CTX adds an initially zero residual to the
normalised window embeddings. Extra CTX stochastic draws are isolated so the
original attention dropout consumes the control's RNG stream at step zero.
"""
from __future__ import annotations

import copy
import hashlib
from types import MethodType

import numpy as np
import torch
from torch import nn

from raptor_common import Corpus, select_centers, study_rng
from raptor_radimagenet import CenterSliceWindows

VARIANTS = ('POS', 'LATE', 'CTX', 'LRH', 'LRL')


def slot_positions(centers, spec, depth):
    """Cached index/(slot size-1), independently within each corpus series slot.

    Explicit empty centre -1 maps to zero. This is corpus geometry, never a
    sample-dependent min/max over selected windows or over the validity mask.
    """
    sizes = [int(s[2]) for s in spec['slots']]
    if any(s <= 0 for s in sizes) or sum(sizes) != depth:
        raise ValueError('slot sizes do not match corpus depth')
    lookup = np.concatenate([np.linspace(0., 1., n, dtype=np.float32)
                             if n > 1 else np.zeros(1, np.float32) for n in sizes])
    out = np.zeros((len(centers), 1), np.float32)
    for j, c in enumerate(centers):
        if c == -1:
            continue
        if int(c) != c or not 0 <= int(c) < depth:
            raise ValueError('invalid cached centre index')
        out[j, 0] = lookup[int(c)]
    return out


class PositionSliceWindows(CenterSliceWindows):
    """Exact control images/labels plus geometry associated with sampled centres."""

    def __getitem__(self, i):
        # Delegate all image generation/jitter to the frozen control loader.
        image, label, uid = super().__getitem__(i)
        epoch = int(self.epoch.item())
        rng = study_rng(self.seed, epoch, uid) if self.train else None
        centers = select_centers(self.corpus.mask(uid), self.corpus.depth, self.k,
                                 self.corpus.spec, self.window_policy, rng)
        positions = torch.from_numpy(slot_positions(centers, self.corpus.spec, self.corpus.depth))
        return (image, positions), label, uid


def loader_factory(variant, enabled=True):
    if variant not in VARIANTS:
        raise ValueError('unknown EXP129 variant')
    return PositionSliceWindows if variant == 'POS' and enabled else CenterSliceWindows


def collate(batch):
    if isinstance(batch[0][0], tuple):
        x = tuple(torch.stack([b[0][j] for b in batch]) for j in range(len(batch[0][0])))
    else:
        x = torch.stack([b[0] for b in batch])
    return x, torch.stack([b[1] for b in batch]), [b[2] for b in batch]


def move_input(x, device, non_blocking=False):
    if isinstance(x, tuple):
        return tuple(move_input(t, device, non_blocking) for t in x)
    return x.to(device, non_blocking=non_blocking)


def model_forward(model, x):
    return model(x)


class ContextResidual(nn.Module):
    """2048->256, two 4-head transformer layers, zero 256->2048 residual.

    Running the residual inside fork_rng restores both CPU and (only when the
    actual input is CUDA) its device's RNG. No CUDA API is called on CPU input.
    The new block is deterministic relative to the current training RNG state,
    while adding no draws to the baseline attention dropout's stream.
    """

    def __init__(self, dim=2048, width=256, dropout=.2):
        super().__init__()
        self.reduce = nn.Linear(dim, width)
        # Independently initialise both layers; TransformerEncoder clones a
        # layer with identical initial values by default.
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            width, 4, dim_feedforward=1024, dropout=dropout,
            batch_first=True, norm_first=False) for _ in range(2)])
        self.expand = nn.Linear(width, dim)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, h):
        devices = [h.device.index] if h.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            z = self.reduce(h)
            for layer in self.layers:
                z = layer(z)
            return self.expand(z)


def _pos_head(self, feats, positions):
    h = self.norm(feats)
    if positions.shape != (*h.shape[:2], 1):
        raise ValueError('POS requires one position per window')
    a = torch.softmax(self.att(h) + self.exp129_pos_projection(positions.to(dtype=h.dtype)), dim=1)
    pooled = torch.einsum('bkn,bkf->bnf', a, h)
    return (pooled * self.clsW).sum(-1) + self.clsb


def _pos_forward(self, x):
    if not isinstance(x, tuple) or len(x) != 2:
        raise ValueError('POS expects (image windows, within-slot positions)')
    images, positions = x
    return self.head(self.encode(images), positions)


def _ctx_head(self, feats):
    h = self.norm(feats)
    h = h + self.exp129_context(h)
    a = torch.softmax(self.att(h), dim=1)
    pooled = torch.einsum('bkn,bkf->bnf', a, h)
    return (pooled * self.clsW).sum(-1) + self.clsb


def _new_adapter(variant, seed, feature_dim=2048, drop=.2):
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        # CPU generator only: torch.manual_seed also calls CUDA seed APIs.
        torch.random.default_generator.manual_seed(int(seed))
        if variant == 'POS':
            adapter = nn.Linear(1, 12, bias=False)
            nn.init.zeros_(adapter.weight)
        elif variant == 'CTX':
            adapter = ContextResidual(feature_dim, 256, drop)
        else:
            raise ValueError('only POS/CTX add adapters')
    return adapter


def initial_adapter_state(variant, seed, feature_dim=2048, drop=.2):
    """Preregister adapter hashes without constructing or loading a backbone."""
    model = nn.Module()
    if variant in ('POS', 'CTX'):
        key = 'exp129_pos_projection' if variant == 'POS' else 'exp129_context'
        model.add_module(key, _new_adapter(variant, seed, feature_dim, drop))
    elif variant not in VARIANTS:
        raise ValueError('unknown EXP129 variant')
    return adapter_state_hashes(model)


def build_variant(base_model, variant, seed, enabled=True):
    """Install only added modules without replacing/reinitialising original keys.

    Construct adapters on CPU with a private seed and restore caller CPU RNG.
    Call before model.to(device). Disabled and schedule-only paths return the
    identical object with no changes to methods, state, or RNG.
    """
    if variant not in VARIANTS:
        raise ValueError('unknown EXP129 variant')
    if not enabled or variant not in ('POS', 'CTX'):
        return base_model
    if any(k.startswith('exp129_') for k in base_model.state_dict()):
        raise ValueError('EXP129 adapter already installed')
    dim = int(base_model.clsW.shape[1])
    if dim != 2048 or int(base_model.n) != 12:
        raise ValueError('EXP129 POS/CTX requires the registered 2048-wide trunk')
    adapter = _new_adapter(variant, seed, dim, float(base_model.att[2].p))
    if variant == 'POS':
        base_model.add_module('exp129_pos_projection', adapter)
        base_model.head = MethodType(_pos_head, base_model)
        base_model.forward = MethodType(_pos_forward, base_model)
    else:
        base_model.add_module('exp129_context', adapter)
        base_model.head = MethodType(_ctx_head, base_model)
    return base_model


def _state_hashes(model, adapter):
    combined, keys = hashlib.sha256(), {}
    for key, value in sorted(model.state_dict().items()):
        if key.startswith('exp129_') != adapter:
            continue
        t = value.detach().cpu().contiguous()
        raw = t.numpy().tobytes()
        combined.update(key.encode())
        combined.update(raw)
        keys[key] = [hashlib.sha256(raw).hexdigest(), str(t.dtype).replace('torch.', ''), list(t.shape)]
    return combined.hexdigest(), keys


def base_state_hashes(model):
    return _state_hashes(model, False)


def adapter_state_hashes(model):
    return _state_hashes(model, True)


class _HeadReference(nn.Module):
    """Copy only the small original head, never a backbone or CUDA inference."""

    def __init__(self, model):
        super().__init__()
        self.norm = copy.deepcopy(model.norm).cpu()
        self.att = copy.deepcopy(model.att).cpu()
        self.clsW = nn.Parameter(model.clsW.detach().cpu().clone())
        self.clsb = nn.Parameter(model.clsb.detach().cpu().clone())
        self.n = int(model.n)

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def head_equivalence(model, variant, seed):
    """Serialisable CPU head proof; no backbone inference or caller RNG changes.

    Checks outputs, original-parameter gradients, input gradients, and global
    RNG after a stochastic training forward for off and zero-residual models.
    The constant-position numerical check uses a NONZERO POS projection.
    """
    if variant not in VARIANTS:
        raise ValueError('unknown EXP129 variant')
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        base = _HeadReference(model)
        off = build_variant(copy.deepcopy(base), variant, seed, enabled=False)
        active = build_variant(copy.deepcopy(base), variant, seed)
        assert base_state_hashes(base) == base_state_hashes(off) == base_state_hashes(active)
        feats = torch.randn(2, 4, base.clsW.shape[1])
        positions = torch.rand(2, 4, 1)
        checks = []
        for train in (False, True):
            for target, mode in ((off, 'off'), (active, 'step0')):
                base.train(train)
                target.train(train)
                base.zero_grad(set_to_none=True)
                target.zero_grad(set_to_none=True)
                x = feats.clone().requires_grad_(True)
                z = feats.clone().requires_grad_(True)
                before = torch.get_rng_state()
                left = base.head(x)
                after = torch.get_rng_state()
                left.square().sum().backward()
                torch.set_rng_state(before)
                right = target.head(z, positions) if variant == 'POS' and mode == 'step0' else target.head(z)
                right_after = torch.get_rng_state()
                right.square().sum().backward()
                if not torch.equal(left, right) or not torch.equal(x.grad, z.grad):
                    raise AssertionError('EXP129 head output/input gradient equivalence failed')
                tp = dict(target.named_parameters())
                for name, param in base.named_parameters():
                    if not torch.equal(param.grad, tp[name].grad):
                        raise AssertionError('EXP129 original-head gradient differs: ' + name)
                if not torch.equal(after, right_after):
                    raise AssertionError('EXP129 extra block perturbed control dropout RNG')
                checks.append(dict(training=train, mode=mode, exact_outputs=True,
                                   exact_original_gradients=True, exact_input_gradients=True,
                                   exact_rng_state=True))
        constant_max_abs = None
        if variant == 'POS':
            base.eval()
            active.eval()
            with torch.no_grad():
                active.exp129_pos_projection.weight.copy_(torch.linspace(-2., 2., 12).view(12, 1))
                left = base.head(feats)
                right = active.head(feats, torch.full((2, 4, 1), .37))
                constant_max_abs = float((left - right).abs().max())
                if not torch.allclose(left, right, rtol=1e-5, atol=1e-6):
                    raise AssertionError('constant-position control failed numerical softmax equivalence')
    return dict(device='cpu', variant=variant, seed=int(seed), checks=checks,
                constant_position_max_abs=constant_max_abs,
                constant_position_control=('analytic cancellation; no other adapter intervention'
                                           if variant == 'POS' else None),
                context_rng_policy=('fork_rng restores caller CPU/input-device state; residual dropout '
                                    'uses that forward state' if variant == 'CTX' else None))


def ordered_equivalence(variant, corpus_dir, train_ids, val_ids, labels, seed, res=224):
    """Ordered tiny corpus check: disabled exact; enabled POS images exact too."""
    from torch.utils.data import DataLoader
    checked, enabled_checked = 0, 0
    with torch.device('cpu'):
        for train, ids, k, epochs in ((True, list(train_ids)[:8], 12, (0, 1)),
                                     (False, list(val_ids)[:2], 24, (0,)),
                                     (False, list(val_ids)[:2], 36, (0,))):
            args = (corpus_dir, ids, labels, res, k, train, 'radimagenet', seed)
            kw = dict(window_policy='within-slot-valid', input_policy='center3')
            base = CenterSliceWindows(*args, **kw)
            other = loader_factory(variant, enabled=False)(*args, **kw)
            active = loader_factory(variant)(*args, **kw)
            opts = dict(batch_size=8 if train else 4, shuffle=train, num_workers=0,
                        drop_last=train, collate_fn=collate, pin_memory=False)
            loaders = [DataLoader(ds, generator=torch.Generator(device='cpu').manual_seed(seed), **opts)
                       for ds in (base, other, active)]
            for epoch in epochs:
                for ds in (base, other, active):
                    ds.set_epoch(epoch)
                if len(loaders[0]) == 0 or len({len(x) for x in loaders}) != 1:
                    raise AssertionError('empty/different ordered equivalence loaders')
                for a, b, c in zip(*loaders, strict=True):
                    image = c[0][0] if isinstance(c[0], tuple) else c[0]
                    if (a[2] != b[2] or a[2] != c[2] or not torch.equal(a[0], b[0])
                            or not torch.equal(a[0], image) or not torch.equal(a[1], b[1])
                            or not torch.equal(a[1], c[1])):
                        raise AssertionError('ordered images/labels differ from EXP113')
                    if isinstance(c[0], tuple):
                        positions = c[0][1]
                        if (positions.shape != (*image.shape[:2], 1) or not torch.isfinite(positions).all()
                                or positions.min() < 0 or positions.max() > 1):
                            raise AssertionError('invalid POS positions')
                    checked += 1
                    enabled_checked += 1
    if torch.cuda.is_initialized():
        raise AssertionError('ordered equivalence initialized CUDA')
    return dict(variant=variant, seed=seed, device='cpu', batches=checked,
                exact_inputs_labels_and_order=True, enabled_samples_checked=enabled_checked,
                enabled_images_equal_control=True, epochs=[0, 1], train_k=12, eval_k=[24, 36],
                constant_position_control=('no other change: constant additive per-target logit '
                                           'cancels analytically in window softmax' if variant == 'POS' else None))
