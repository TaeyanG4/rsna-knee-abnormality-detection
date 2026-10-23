"""RadImageNet ResNet50 initialisation and centre-slice x3 inputs for the Raptor MIL model (EXP107, H-B-24). New module; the
shared training sources (raptor_common.py, raptor_data.py) are imported unchanged.

Weights: the public Kaggle dataset marwanmath/resnet-50-radimagenet-marwan, file ResNet50.pt (sha256 RADIMAGENET_R50_SHA256, the encoder the
0.943 anchor pins). It is a torch state_dict of nn.Sequential(*list(torchvision resnet50().children())[:-2]) saved under the attribute
`backbone`, so its keys are backbone.{0,1,4,5,6,7}.*; STAGE_MAP renames them to timm resnet50 names (conv1, bn1, layer1..layer4). timm's
resnet50 with num_classes=0 has exactly these 318 tensors; the load is strict (no missing, unexpected, shape-mismatched or non-finite tensor)
and every loaded tensor is re-checked byte for byte.

Input (input_policy 'center3'): the same window centres as StudyWindows (select_centers, so the within-slot-valid policy and the per-study
sampling streams are unchanged), but each window is the centre cache plane repeated as the three channels instead of the (c-1, c, c+1)
triplet. Normalisation 'radimagenet': x in [0, 1] -> 2x - 1, i.e. the anchor's uint8 / 127.5 - 1 (and the scaling whose conv1 statistics
match the checkpoint's bn1 running statistics best).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from raptor_common import Corpus, build_model, select_centers, study_rng
from raptor_data import IMAGENET_MEAN, IMAGENET_STD, StudyWindows

RADIMAGENET_R50_SHA256 = '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734'
RADIMAGENET_R50_KEYS = 318
RADIMAGENET_R50_PARAMS = 23508032
STAGE_MAP = {'0': 'conv1', '1': 'bn1', '4': 'layer1', '5': 'layer2', '6': 'layer3', '7': 'layer4'}
ARCH = 'resnet50'
INPUT_POLICIES = ('center3',)
NORMS = ('radimagenet', 'imagenet', 'none')


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def map_radimagenet_state(state):
    """backbone.<i>.<rest> (torchvision Sequential indices) -> timm resnet50 names; refuses anything else."""
    mapped = {}
    for key, value in state.items():
        m = re.match(r'^backbone\.(\d+)\.(.+)$', key)
        if not m or m.group(1) not in STAGE_MAP:
            raise ValueError(f'unexpected RadImageNet key: {key}')
        name = STAGE_MAP[m.group(1)] + '.' + m.group(2)
        if name in mapped:
            raise ValueError(f'duplicate mapped key: {name}')
        mapped[name] = value
    return mapped


def strict_load_backbone(backbone, mapped):
    """Strict load with an explicit report; any missing / unexpected / shape-mismatched / non-finite tensor raises."""
    import torch
    expected = backbone.state_dict()
    missing = sorted(set(expected) - set(mapped))
    unexpected = sorted(set(mapped) - set(expected))
    shapes = sorted(k for k in set(expected) & set(mapped) if tuple(expected[k].shape) != tuple(mapped[k].shape))
    dtypes = sorted(k for k in set(expected) & set(mapped) if expected[k].dtype != mapped[k].dtype)
    nonfinite = sorted(k for k, v in mapped.items() if v.is_floating_point() and not bool(torch.isfinite(v).all()))
    if missing or unexpected or shapes or dtypes or nonfinite:
        raise RuntimeError(f'RadImageNet load refused: missing={missing[:10]} unexpected={unexpected[:10]} shapes={shapes[:10]} '
                           f'dtypes={dtypes[:10]} nonfinite={nonfinite[:10]}')
    result = backbone.load_state_dict(mapped, strict=True)
    after = backbone.state_dict()
    differing = sorted(k for k, v in mapped.items() if not torch.equal(after[k].detach().cpu(), v.detach().cpu()))
    if differing:
        raise RuntimeError(f'loaded tensors differ from the checkpoint: {differing[:10]}')
    return dict(source_keys=len(mapped), target_keys=len(expected), missing=missing, unexpected=unexpected, shape_mismatch=shapes,
                dtype_mismatch=dtypes, nonfinite=nonfinite, strict_result=str(result), byte_equal_after_load=True,
                params=int(sum(p.numel() for p in backbone.parameters())), key_map={f'backbone.{i}': n for i, n in STAGE_MAP.items()})


def load_radimagenet_state(path, expected_sha256=RADIMAGENET_R50_SHA256):
    import torch
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise RuntimeError(f'RadImageNet weight file sha256 {digest} != registered {expected_sha256}: {path}')
    state = torch.load(Path(path), map_location='cpu', weights_only=True)
    if len(state) != RADIMAGENET_R50_KEYS:
        raise RuntimeError(f'RadImageNet state has {len(state)} tensors, expected {RADIMAGENET_R50_KEYS}')
    return state, digest


def build_radimagenet_model(arch, weights_path, expected_sha256=RADIMAGENET_R50_SHA256):
    """RaptorClassifier(timm resnet50, random init consumed exactly as build_model) with the RadImageNet backbone loaded strictly.

    Call after seed_all(seed): build_model draws the backbone's random init and then the head's, so the head init is the same function of
    the seed as for any timm backbone; the backbone tensors are then replaced by the checkpoint (bytes checked)."""
    if arch != ARCH:
        raise ValueError(f'RadImageNet weights are a {ARCH}; got arch {arch!r}')
    state, digest = load_radimagenet_state(weights_path, expected_sha256)
    model = build_model(arch, pretrained=False)
    report = strict_load_backbone(model.backbone, map_radimagenet_state(state))
    if report['params'] != RADIMAGENET_R50_PARAMS:
        raise RuntimeError(f"backbone parameter count {report['params']} != {RADIMAGENET_R50_PARAMS}")
    report.update(weights_sha256=digest, weights_file=Path(weights_path).name)
    return model, report


def center3_windows_from(vol, centers, res):
    """(K,3,res,res) float32 in [0,1]; the centre cached plane repeated as the three channels (c = -1 -> zeros, as windows_from)."""
    import torch
    import torch.nn.functional as F
    depth = vol.shape[0]
    out = np.empty((len(centers), 3, res, res), np.float32)
    for j, c in enumerate(centers):
        if c == -1:
            out[j] = 0
            continue
        c = max(1, min(int(c), depth - 2))  # the same clamp as windows_from, so the centre plane is windows_from's channel 1
        plane = vol[c][None].astype(np.float32) / 255.0
        t = torch.from_numpy(plane)
        if t.shape[-1] != res or t.shape[-2] != res:
            t = F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False)[0]
        out[j] = t.numpy()  # (1,res,res) broadcast to the three channels
    return out


def normalise(x, norm):
    if norm == 'radimagenet':
        return x * np.float32(2.0) - np.float32(1.0)
    if norm == 'imagenet':
        return (x - IMAGENET_MEAN) / IMAGENET_STD
    if norm == 'none':
        return x
    raise ValueError(f'unknown norm {norm}')


class CenterSliceWindows(StudyWindows):
    """StudyWindows with the centre plane x3 as channels; centres, sampling streams and intensity jitter are StudyWindows' own."""

    def __init__(self, corpus_dir, ids, labels, res, k, train, norm, seed, jitter=.20, window_policy='legacy', input_policy='center3'):
        if input_policy not in INPUT_POLICIES:
            raise ValueError(f'unknown input policy {input_policy}')
        if norm not in NORMS:
            raise ValueError(f'unknown norm {norm}')
        super().__init__(corpus_dir, ids, labels, res, k, train, norm, seed, jitter=jitter, window_policy=window_policy)
        self.input_policy = input_policy

    def __getitem__(self, i):
        import torch
        if self.corpus is None:
            self.corpus = Corpus(self.corpus_dir)
        uid, epoch = self.ids[i], int(self.epoch.item())
        vol, mask = self.corpus.volume(uid), self.corpus.mask(uid)
        rng = study_rng(self.seed, epoch, uid) if self.train else None
        centers = select_centers(mask, self.corpus.depth, self.k, self.corpus.spec, self.window_policy, rng)
        x = center3_windows_from(vol, centers, self.res)
        if self.train and self.jitter:
            intensity = study_rng(self.seed, epoch, uid, 'intensity').random()
            x = np.clip(x * (1 + (intensity - .5) * self.jitter), 0, 1)
        x = normalise(x, self.norm)
        if x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f'invalid input for {uid}')
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid
