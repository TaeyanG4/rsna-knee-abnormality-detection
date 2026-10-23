"""CoAtNet v1.0.0: explicit data/model factories and immutable CPU evidence.

Changelog: extract the EXP181/188/192 scientific data/model seam without
patching modules or invoking legacy trainers. Production fitting is separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from training.src import exp143_own0949 as base
from training.src import exp181_aug1 as aug1
from training.src import exp188_coverage as coverage
from training.src import exp192_crop as crop
from training.src.raptor_common import study_rng, train_centers
from tools.exp112_full_swa import recompute_bn, state_dict_sha256

VERSION = '1.0.0'
CORPUS = 'storage/cache/derived/EXP143_RAPTOR96_384_C140_S02'
INIT = 'storage/weights/exp143-imagenet-init/coatnet384-imagenet.pt'
CONTROL = 'artifacts/experiments/EXP181_COATNET_F0_BRIDGE/'
METADATA = 'artifacts/experiments/EXP188_COATNET_COVERAGE_F0/protocol-v1/coverage-metadata.json'
PINS = {
    CONTROL + 'protocol-v1/manifest-CONTROL.json': 'd2009bdcae5918bb9f1d8747201f1aac2243d89607ba9ef7d589a75fa856353e',
    CONTROL + 'CONTROL/summary.json': '623a5a9fbe8a13020c657e59ba51c42a9714dd85a05c0587f78c224e44b7fbf6',
    CONTROL + 'CONTROL/provenance.json': 'f1a4f94bc764259240cde1bd44db06a2ad4cf9ea9167222b20ddf75b7ea38a79',
    METADATA: '2f6b0407dd1ef024bad1bbb8a92f109a3e057b7809680bbec015525e223f5855',
}
SOURCES = (
    'training/arms/coatnet.py', 'training/runners/coatnet_arms.py',
    'training/src/exp143_own0949.py', 'training/src/exp181_aug1.py',
    'training/src/exp188_coverage.py', 'training/src/exp192_crop.py',
    'training/src/raptor_common.py', 'training/src/exp143_v2.py',
    'tools/exp112_full_swa.py',
)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


@dataclass(frozen=True)
class Config:
    sampler: str = 'legacy'
    crop: str = 'context'
    augmentation: str = 'intensity'
    head: str = 'raptor-attention-12'

    def __post_init__(self):
        if (self.sampler, self.crop, self.augmentation, self.head) not in {
            ('legacy', 'context', 'intensity', 'raptor-attention-12'),
            ('coverage', 'context', 'intensity', 'raptor-attention-12'),
            ('legacy', 'alternating-context-detail', 'intensity', 'raptor-attention-12'),
            ('legacy', 'context', 'aug1', 'raptor-attention-12'),
        }:
            raise ValueError('unsupported scientific configuration; combinations need new registration')


PRESETS = dict(control=Config(), coverage=Config(sampler='coverage'),
               crop=Config(crop='alternating-context-detail'), aug1=Config(augmentation='aug1'))


@dataclass
class ControlContext:
    root: Path
    registration: dict = field(repr=False)
    inputs: dict

    def report(self):
        p = self.registration
        return dict(status='CONTROL_CONTRACT_VERIFIED', version=VERSION,
            scientific_scope='DATA_MODEL_PARITY_ONLY', launch_authorized=False,
            train=len(p['train_ids']), val=len(p['val_ids']), seed=p['seed'], fold=p['fold']['fold'],
            updates_per_epoch=216, total_updates=3456, bn_studies=3455,
            train_roster_sha256=p['fold']['bridge_train_sha256'],
            val_roster_sha256=p['fold']['val_sha256'], inputs=self.inputs,
            runtime_qualification='UNAVAILABLE_IN_LEGACY_CONTROL',
            next_requirement='Versioned target/runtime evidence and production lifecycle integration')


def load_control(root):
    """Reuse immutable scientific inputs; never infer missing historical runtime."""
    root = Path(root).resolve()
    for rel, expected in PINS.items():
        if sha256(root / rel) != expected:
            raise ValueError('CONTROL input changed: ' + rel)
    p = read(root / (CONTROL + 'protocol-v1/manifest-CONTROL.json'))
    inputs = dict(PINS)
    inputs.update({INIT: p['init_sha256'], CORPUS + '/manifest.json': p['corpus_manifest_sha256'],
                   p['fold']['ordered_ids']: p['fold']['ordered_ids_sha256']})
    for rel, expected in inputs.items():
        if sha256(root / rel) != expected:
            raise ValueError('scientific input changed: ' + rel)
    # These counts are scientific invariants, never elapsed-time admission gates.
    if (p['seed'] != 42 or p['fold']['fold'] != 0 or len(p['train_ids']) != 3455
            or len(p['val_ids']) != 888 or set(p['train_ids']) & set(p['val_ids'])):
        raise ValueError('CONTROL scientific identity changed')
    return ControlContext(root, p, inputs)


def training_windows(config, vol, mask, seed, epoch, uid, *, augment, slots=None):
    if config.crop == 'alternating-context-detail':
        return crop.windows(vol, mask, True, seed, epoch, uid, jitter=augment)
    if augment:
        if config.sampler == 'coverage':
            if slots is None:
                raise ValueError('coverage requires explicit study metadata')
            return coverage.coverage_windows(vol, mask, slots, seed, epoch, uid)
        if config.augmentation == 'aug1':
            return aug1.aug1_windows(vol, mask, seed, epoch, uid)
        return base.windows(vol, mask, True, seed, epoch, uid)
    if vol.shape != (base.DEPTH, base.IMG, base.IMG) or vol.dtype != np.uint8 or mask.shape != (base.DEPTH,) or not mask.any():
        raise ValueError('BN requires a nonempty native corpus study')
    cs = np.asarray(train_centers(mask, base.DEPTH, base.K_TRAIN, study_rng(seed, epoch, uid)))
    x = np.asarray(vol[np.stack([cs - 1, cs, cs + 1], 1)], np.float32) / 255.
    return np.ascontiguousarray((x - base.MEAN) / base.STD)


def evaluation_windows(config, vol, mask, mirror=False):
    transform = crop.windows if config.crop == 'alternating-context-detail' else base.windows
    return transform(vol, mask, mirror=mirror)


class StudyDataset(base.StudyDataset):
    """Explicit immutable config and per-instance coverage input; Windows spawn safe."""
    def __init__(self, corpus, ids, labels, seed, config=Config(), augment=True, metadata=None):
        super().__init__(corpus, ids, labels, seed, augment)
        self.config = config
        self.metadata = None
        if config.sampler == 'coverage':
            if metadata is None:
                raise ValueError('coverage metadata path required')
            if sha256(metadata) != PINS[METADATA]:
                raise ValueError('coverage metadata identity changed')
            self.metadata = read(metadata)['studies']

    def __getitem__(self, i):
        if self.corpus is None:
            self.corpus = base.Corpus(self.path)
        uid, epoch = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.get(uid)
        row = self.metadata[uid] if self.metadata is not None else None
        if row is not None and coverage.metadata_mask_sha(mask) != row['mask_sha256']:
            raise ValueError('coverage mask identity changed')
        x = training_windows(self.config, vol, mask, self.seed, epoch, uid,
                             augment=self.augment, slots=row['slots'] if row else None)
        return torch.from_numpy(x), torch.tensor(self.labels[uid], dtype=torch.float32), uid


class RaptorClassifier(nn.Module):
    """Exact frozen parameter construction order and state keys; no module patching."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        features = backbone.num_features
        self.norm = nn.LayerNorm(features)
        self.att = nn.Sequential(nn.Linear(features, 256), nn.Tanh(), nn.Dropout(.2), nn.Linear(256, 12))
        self.clsW = nn.Parameter(torch.zeros(12, features))
        self.clsb = nn.Parameter(torch.zeros(12))
        nn.init.trunc_normal_(self.clsW, std=.02)
        self.n = 12

    def encode(self, x):
        batch, windows = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).view(batch, windows, -1)

    def head(self, feats):
        h = self.norm(feats)
        attention = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', attention, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x):
        return self.head(self.encode(x))


def build_model(config=Config(), res=384, backbone_factory=None):
    if not isinstance(config, Config) or res not in (384, 320):
        raise ValueError('registered config and 384/320 resolution required')
    if backbone_factory is None:
        import timm
        def backbone_factory():
            return timm.create_model(base.ARCH, pretrained=False, num_classes=0,
                                     in_chans=3, global_pool='avg', img_size=res)
    return RaptorClassifier(backbone_factory())


def infer_pair(model, config, vol, mask, device='cpu', precision='none', chunk=16):
    if any(m.training for m in model.modules()):
        raise ValueError('inference requires eval mode')
    if type(chunk) is not int or chunk < 1:
        raise ValueError('positive integer chunk required')
    probs = []
    with torch.inference_mode():
        for mirror in (False, True):
            x = torch.from_numpy(evaluation_windows(config, vol, mask, mirror))
            with base.amp_context(device, precision):
                feats = torch.cat([model.backbone(z.to(device)) for z in x.split(chunk)], 0)
                logits = model.head(feats.unsqueeze(0)).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError('nonfinite prediction; no fallback')
            probs.append(logits.sigmoid()[0].cpu().numpy())
    return probs[0], probs[1]


def cpu_smoke(root, variant, out):
    """Real CoAtNet FP32 update, BN, two K94 views, strict reload. Never admission."""
    if variant not in PRESETS:
        raise ValueError('unknown variant')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU smoke requires uninitialized CUDA')
    root, out = Path(root).resolve(), Path(out).resolve()
    # CPU artifacts must stay outside experiment/release trees.
    if not out.is_relative_to(root / '.local/refactor'):
        raise ValueError('CPU output must be below .local/refactor')
    context = load_control(root)
    sources = {p: sha256(root / p) for p in SOURCES}
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config, p = PRESETS[variant], context.registration
    uid = p['train_ids'][0]
    dataset = StudyDataset(root / CORPUS, [uid], p['train_labels'], 42, config,
                           metadata=root / METADATA if config.sampler == 'coverage' else None)
    dataset.set_epoch(1)
    x, y, _ = dataset[0]
    study_row = base.Corpus(root / CORPUS).rows[uid]
    study_inputs = {f'{CORPUS}/study_{study_row:05d}.{suffix}': sha256(root / CORPUS / f'study_{study_row:05d}.{suffix}')
                    for suffix in ('npy', 'json')}
    base._cpu_seed(42)
    model = build_model(config)
    base.load_imagenet(model, root / INIT)
    initial = state_dict_sha256(model.state_dict())
    if initial != p['expected_initial_state_sha256']:
        raise ValueError('initial state differs from CONTROL')
    from training.src.exp143_v2 import make_optimization
    optimizer, scheduler, _ = make_optimization(model, p['recipe'], 3456, 'none')
    model.train()
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p['pos_weight']))(model(x[None]), y[None])
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite CPU loss')
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), p['recipe']['clip'], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    updated = state_dict_sha256(model.state_dict())
    if updated == initial:
        raise AssertionError('CPU update did not change model')
    del optimizer, scheduler, loss, x, y
    dataset.augment = False
    dataset.set_epoch(16)
    bn = recompute_bn(model, [base.collate([dataset[0]])], 'cpu')
    state_hash = state_dict_sha256(model.state_dict())
    with (out / 'cpu-model.pt').open('xb') as stream:
        torch.save(dict(model=model.state_dict(), config=asdict(config), scientific_scope='CPU_SMOKE_ONLY'), stream)
    del model
    model = build_model(config, 320).eval()
    checkpoint = torch.load(out / 'cpu-model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    if state_dict_sha256(model.state_dict()) != state_hash:
        raise AssertionError('strict checkpoint reload changed state')
    vol, mask = dataset.corpus.get(uid)
    plain, mirror = infer_pair(model, config, vol, mask, chunk=2)
    predictions = np.stack([plain, mirror, (plain + mirror) * .5])
    with (out / 'predictions.npy').open('xb') as stream:
        np.save(stream, predictions, allow_pickle=False)
    # Recheck inputs/sources before publishing success; failures leave no receipt.
    for rel, expected in {**context.inputs, **study_inputs, **sources}.items():
        if sha256(root / rel) != expected:
            raise ValueError('input/source changed during smoke: ' + rel)
    if torch.cuda.is_initialized():
        raise AssertionError('CPU smoke initialized CUDA')
    result = dict(schema='coatnet-cpu-evidence/v1', version=VERSION, status='HOLD',
        scientific_scope='CPU_SMOKE_ONLY', launch_authorized=False, variant=variant, config=asdict(config),
        initial_state_sha256=initial, updated_state_sha256=updated, final_state_sha256=state_hash,
        optimizer_updates=1, precision='fp32', training_windows=12, training_resolution=384,
        gradient_norm=float(gradient_norm), bn=bn, evaluation_windows_per_view=94,
        evaluation_resolution=320, prediction_views=['plain', 'mirror', 'mean'],
        prediction_shape=list(predictions.shape), checkpoint_reload_exact=True, cuda_initialized=False,
        inputs={**context.inputs, **study_inputs}, sources=sources,
        artifacts={name: sha256(out / name) for name in ('cpu-model.pt', 'predictions.npy')},
        seconds=time.monotonic() - started,
        limitations=['One-study FP32 diagnostic, not accumulated production fitting or full BN replay',
                     'No actual-target GPU observation or independent PASS',
                     'Legacy CONTROL lacks complete shared-lifecycle environment evidence'])
    write_new(out / 'receipt.json', result)
    return result
