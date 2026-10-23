"""Portable six-slot knee reader; no data, weights, or devices are opened on import.

Adapted from goodpjw2008's ``rsna-knee-2-5d-convnext-reader`` dataset v1
(2026-10-05), distributed under Apache License 2.0:
https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader
License: https://www.apache.org/licenses/LICENSE-2.0
SPDX-License-Identifier: Apache-2.0

The retained source audit is D-C-014. This reviewed adaptation separates DICOM
decoding from array preprocessing, rejects malformed/empty inputs, makes loaders
Windows-picklable, and preserves metadata order for decoded-count ties.
Initialization defaults to OFFLINE random weights; the caller must pin/load its
allowed initialization. The explicit timm tag is a new recipe choice, not a claim
about the author's historical initialization. No downloaded module is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]
N_SLOTS, SIZE, MAX_SLICES = 6, 384, 64
MM_PER_PX = 0.4
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CANON = {0: (+1, (1, +1), (2, -1)),
         1: (+1, (0, +1), (2, -1)),
         2: (-1, (0, +1), (1, +1))}


def _positive_int(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if not math.isfinite(number) or number != int(number) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def slot_of(plane, fat_sup):
    """Slots 0..2 are fat-suppressed sagittal/coronal/axial; 3..5 are not."""
    if plane not in PLANES:
        raise ValueError(f"Unknown anatomical plane: {plane!r}")
    try:
        flag = float(fat_sup)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fat_Suppression must be 0 or 1") from exc
    if flag not in (0, 1):
        raise ValueError("Fat_Suppression must be 0 or 1")
    return PLANES.index(plane) + 3 * (1 - int(flag))


def build_study_table(series_df, n_slices: Mapping[str, int]):
    """Group metadata by six slots, using SUCCESSFULLY DECODED/cache slice counts.

    Accept a pandas DataFrame or an iterable of row mappings. Counts below three
    are ineligible. Stable sorting preserves CSV/metadata row order for ties.
    A series may appear only once in metadata; repeated rows are rejected.
    """
    rows = series_df.to_dict("records") if hasattr(series_df, "to_dict") else series_df
    table, seen, counts = {}, set(), {}
    for row in rows:
        if row["StudyInstanceUID"] is None or row["SeriesInstanceUID"] is None:
            raise ValueError("Study and series UIDs must be nonempty")
        study, series = str(row["StudyInstanceUID"]), str(row["SeriesInstanceUID"])
        if not study or not series or study == "nan" or series == "nan":
            raise ValueError("Study and series UIDs must be nonempty")
        if series in seen:
            raise ValueError(f"Duplicate series metadata: {series}")
        seen.add(series)
        slot = slot_of(row["Anatomical_Plane"], row["Fat_Suppression"])
        count = _positive_int(n_slices.get(series, 0), "decoded slice count", minimum=0)
        if count < 3:
            continue
        counts[series] = count
        table.setdefault(study, [[] for _ in range(N_SLOTS)])[slot].append(series)
    for slots in table.values():
        for candidates in slots:
            candidates.sort(key=lambda uid: -counts[uid])
    return table


def window_centres(n, k, train, lo=0.04, hi=0.96, rng=None):
    """Three-slice window centres; eval linspace or stratified train jitter."""
    n, k = _positive_int(n, "n", 3), _positive_int(k, "k")
    if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi <= 1):
        raise ValueError("Require finite 0 <= lo < hi <= 1")
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    if train:
        edges = np.linspace(a, b, k + 1)
        random = np.random if rng is None else rng
        centres = edges[:-1] + random.random(k) * (edges[1:] - edges[:-1])
    else:
        centres = np.linspace(a, b, k)
    return np.clip(np.round(centres).astype(int), 1, n - 2)


@dataclass(frozen=True)
class DecodedSlice:
    """One already-decoded 2D frame; the caller owns DICOM I/O/error accounting.

    Multiframe DICOMs must be expanded by the caller with frame-specific metadata.
    ``source_id`` should include a stable frame ID; it breaks position/instance
    ties. If omitted, remaining ties preserve caller order.
    """
    pixels: np.ndarray
    pixel_spacing: Sequence[float]
    image_orientation: Sequence[float] | None = None
    image_position: Sequence[float] | None = None
    instance_number: float = 0.0
    source_id: str = ""
    photometric: str = "MONOCHROME2"
    rescale_slope: float | None = None
    rescale_intercept: float | None = None


def _vector(value, length, name):
    try:
        out = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if out.shape != (length,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return out


def _prepare_slice(item: DecodedSlice):
    image = np.asarray(item.pixels, dtype=np.float32)
    if image.ndim != 2 or min(image.shape) < 1 or not np.isfinite(image).all():
        raise ValueError("Decoded pixels must be a nonempty finite 2D array")
    spacing = _vector(item.pixel_spacing, 2, "pixel spacing")
    if (spacing <= 0).any():
        raise ValueError("Pixel spacing must be strictly positive")
    orientation = None if item.image_orientation is None else _vector(item.image_orientation, 6, "orientation")
    if orientation is not None and np.linalg.norm(np.cross(orientation[:3], orientation[3:])) < 1e-6:
        raise ValueError("Orientation axes must not be collinear")
    position = None if item.image_position is None else _vector(item.image_position, 3, "position")
    if not math.isfinite(float(item.instance_number)):
        raise ValueError("Instance number must be finite")
    if item.photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError("Only monochrome decoded slices are supported")
    if item.photometric == "MONOCHROME1":
        image = image.max() - image
    if (item.rescale_slope is None) != (item.rescale_intercept is None):
        raise ValueError("Supply both rescale slope and intercept, or neither")
    if item.rescale_slope is not None:
        slope, intercept = float(item.rescale_slope), float(item.rescale_intercept)
        if not math.isfinite(slope) or not math.isfinite(intercept):
            raise ValueError("Rescale parameters must be finite")
        image = image * slope + intercept
        if not np.isfinite(image).all():
            raise ValueError("Rescaled pixels must be finite")
    return image, orientation, position, spacing, float(item.instance_number), str(item.source_id)


def canonical_resample(image, pixel_spacing, image_orientation=None, plane=0):
    """Pure per-axis physical crop/resample/pad to 384 at 0.4 mm/pixel."""
    plane = _positive_int(plane, "plane", minimum=0)
    if plane not in CANON:
        raise ValueError("Plane index must be 0, 1 or 2")
    image, orientation, _, spacing, _, _ = _prepare_slice(
        DecodedSlice(image, pixel_spacing, image_orientation))
    if orientation is not None:
        row_cos, col_cos = orientation[:3], orientation[3:]
        _, (col_axis, col_sign), (row_axis, row_sign) = CANON[plane]
        if abs(row_cos[col_axis]) < abs(row_cos[row_axis]):
            image, row_cos, col_cos, spacing = image.T, col_cos, row_cos, spacing[::-1]
        if row_cos[col_axis] * col_sign < 0:
            image = image[:, ::-1]
        if col_cos[row_axis] * row_sign < 0:
            image = image[::-1]
    h, w = image.shape
    crop_h = min(h, int(round(SIZE * MM_PER_PX / spacing[0])))
    crop_w = min(w, int(round(SIZE * MM_PER_PX / spacing[1])))
    if min(crop_h, crop_w) < 1:
        raise ValueError("Pixel spacing produces an empty physical crop")
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    crop = np.ascontiguousarray(image[top:top + crop_h, left:left + crop_w])
    new_h = max(1, min(SIZE, int(round(crop_h * spacing[0] / MM_PER_PX))))
    new_w = max(1, min(SIZE, int(round(crop_w * spacing[1] / MM_PER_PX))))
    interpolation = cv2.INTER_AREA if new_h < crop_h else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
    if not np.isfinite(resized).all():
        raise ValueError("Resampled pixels must be finite")
    output = np.zeros((SIZE, SIZE), dtype=np.float32)
    top, left = (SIZE - new_h) // 2, (SIZE - new_w) // 2
    output[top:top + new_h, left:left + new_w] = resized
    return output


def preprocess_decoded_series(slices: Sequence[DecodedSlice], plane_name=None):
    """Canonicalize >=3 decoded frames; return uint8 volume and cache metadata.

    Missing spacing is rejected rather than guessed. No source image is modified.
    Position sorting uses the first available IOP; if any IPP is missing it falls
    back to instance number. The caller should separately audit within-series
    orientation consistency, decode failures, and metadata fallbacks.
    """
    if len(slices) < 3:
        raise ValueError("A usable series needs at least three decoded slices")
    if plane_name is not None and plane_name not in PLANES:
        raise ValueError(f"Unknown anatomical plane: {plane_name!r}")
    prepared = [_prepare_slice(item) for item in slices]
    orientation = next((item[1] for item in prepared if item[1] is not None), None)
    normal = None if orientation is None else np.cross(orientation[:3], orientation[3:])
    plane = int(np.abs(normal).argmax()) if normal is not None else (PLANES.index(plane_name) if plane_name else 0)
    info = {"n_raw": len(prepared), "plane": PLANES[plane]}
    if normal is not None and all(item[2] is not None for item in prepared):
        sign = CANON[plane][0] * np.sign(normal[plane])
        prepared.sort(key=lambda item: (float(np.dot(item[2], normal)) * sign, item[4], item[5]))
        positions = np.asarray([np.dot(item[2], normal) for item in prepared])
        info.update(order="position_then_instance_then_source_id", slice_gap=float(np.median(np.abs(np.diff(positions)))))
    else:
        prepared.sort(key=lambda item: (item[4], item[5]))
        info["order"] = "instance_then_source_id"
    if len(prepared) > MAX_SLICES:
        prepared = [prepared[i] for i in np.linspace(0, len(prepared) - 1, MAX_SLICES).round().astype(int)]
    volume = np.stack([canonical_resample(item[0], item[3], item[1], plane) for item in prepared])
    lo, hi = np.percentile(volume[:, ::4, ::4], [1, 99.5])
    volume = np.clip((volume - lo) / max(hi - lo, 1e-6), 0, 1)
    info.update(n=len(volume), intensity_low=float(lo), intensity_high=float(hi))
    return (volume * 255).round().astype(np.uint8), info


@dataclass(frozen=True)
class NpySeriesLoader:
    """Top-level callable is pickleable under Windows DataLoader spawn."""
    cache_dir: str | Path

    def __call__(self, series):
        return np.load(Path(self.cache_dir) / f"{series}.npy", mmap_mode="r", allow_pickle=False)


@dataclass(frozen=True)
class ArraySeriesLoader:
    arrays: Mapping[str, np.ndarray]

    def __call__(self, series):
        return self.arrays[series]


class StudyDataset(torch.utils.data.Dataset):
    def __init__(self, studies, table, cache_dir=None, targets=None, k=8, train=False,
                 loader: Callable[[str], np.ndarray] | None = None):
        self.studies, self.table = list(studies), table
        self.k, self.train = _positive_int(k, "k"), bool(train)
        if loader is None and cache_dir is None:
            raise ValueError("Provide cache_dir or a picklable array loader")
        self.loader = loader if loader is not None else NpySeriesLoader(cache_dir)
        self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)
        if self.targets is not None:
            if self.targets.shape != (len(self.studies), len(LABELS)) or not np.isfinite(self.targets).all():
                raise ValueError("Targets must be finite [studies, 12]")
            if ((self.targets < 0) | (self.targets > 1)).any():
                raise ValueError("Targets must be probabilities in [0, 1]")

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        study = self.studies[index]
        slots = self.table.get(study)
        if slots is None or len(slots) != N_SLOTS or not any(slots):
            raise ValueError(f"Study {study!r} has no valid six-slot series table")
        x = np.zeros((N_SLOTS, self.k, 3, SIZE, SIZE), dtype=np.uint8)
        mask, pos = np.zeros(N_SLOTS, dtype=bool), np.zeros((N_SLOTS, self.k), dtype=np.float32)
        for slot, candidates in enumerate(slots):
            if not candidates:
                continue
            series = candidates[np.random.randint(len(candidates))] if self.train else candidates[0]
            volume = np.asarray(self.loader(series))
            if volume.dtype != np.uint8 or volume.ndim != 3 or volume.shape[1:] != (SIZE, SIZE):
                raise ValueError(f"Series {series!r} must be uint8 [slices,384,384]")
            if not 3 <= len(volume) <= MAX_SLICES:
                raise ValueError(f"Series {series!r} must contain 3..64 decoded cached slices")
            centres = window_centres(len(volume), self.k, self.train)
            x[slot] = np.stack([volume[[centre - 1, centre, centre + 1]] for centre in centres])
            mask[slot], pos[slot] = True, centres / (len(volume) - 1)
        y = np.zeros(len(LABELS), dtype=np.float32) if self.targets is None else self.targets[index].copy()
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos), torch.from_numpy(y)


def make_views(x, res, train, group, fov=0.92):
    """Same series-shared affine, gain/gamma and ImageNet normalization recipe."""
    res, group = _positive_int(res, "res"), _positive_int(group, "group")
    if x.dtype != torch.uint8 or x.ndim != 4 or x.shape[1] != 3 or x.shape[0] == 0 or min(x.shape[2:]) < 1:
        raise ValueError("Views require nonempty uint8 [windows,3,H,W]")
    if x.shape[0] % group or not math.isfinite(fov) or not 0 < fov <= 1:
        raise ValueError("Windows must divide into groups and 0 < fov <= 1")
    n, ns, device = len(x), len(x) // group, x.device
    x = x.float().div_(255)
    if train:
        angle = (torch.rand(ns, device=device) - .5) * 2 * math.radians(12)
        scale = fov * (1 + (torch.rand(ns, device=device) - .5) * .24)
        tx, ty = (torch.rand(ns, device=device) - .5) * .12, (torch.rand(ns, device=device) - .5) * .12
        gamma = torch.exp((torch.rand(ns, device=device) - .5) * .5)
        gain = 1 + (torch.rand(ns, device=device) - .5) * .2
    else:
        angle, scale = torch.zeros(ns, device=device), torch.full((ns,), fov, device=device)
        tx = ty = torch.zeros(ns, device=device)
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1).repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    if x.shape[-1] > 1.5 * res:
        x = F.avg_pool2d(x, 2) if x.shape[-1] >= 2 * res else F.interpolate(x, scale_factor=.75, mode="bilinear", antialias=True)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if train:
        x = x.clamp_min(1e-4).pow(gamma.repeat_interleave(group).view(n, 1, 1, 1)) * gain.repeat_interleave(group).view(n, 1, 1, 1)
    mean, std = (torch.tensor(values, device=device).view(1, 3, 1, 1) for values in (MEAN, STD))
    return (x - mean) / std


class KneeNet(nn.Module):
    def __init__(self, backbone="convnext_tiny.in12k_ft_in1k", pretrained=False,
                 dim=384, depth=2, drop=.2, drop_path=.1):
        super().__init__()
        if dim < 8 or dim % 8 or depth < 0:
            raise ValueError("dim must be a positive multiple of 8 and depth nonnegative")
        import timm
        kwargs = {"drop_path_rate": drop_path} if drop_path else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kwargs)
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else None
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * .02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))

    @staticmethod
    def _validate_mask_pos(mask, pos, batch, slots, windows):
        if slots != N_SLOTS or batch < 1 or windows < 1 or mask.dtype != torch.bool or mask.shape != (batch, slots):
            raise ValueError("Expected nonempty six-slot input and boolean [B,6] mask")
        if not bool(mask.any(1).all()):
            raise ValueError("Each study must have at least one valid series")
        if pos.shape != (batch, slots, windows) or not bool(torch.isfinite(pos).all()) or bool(((pos < 0) | (pos > 1)).any()):
            raise ValueError("Positions must be finite [B,6,K] in [0,1]")

    def head(self, feat, mask, pos):
        if feat.ndim != 4:
            raise ValueError("Features must be [B,6,K,D]")
        batch, slots, windows, _ = feat.shape
        self._validate_mask_pos(mask, pos, batch, slots, windows)
        slot = torch.arange(slots, device=feat.device).view(1, slots, 1).expand(batch, slots, windows)
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
        h = h.reshape(batch, slots * windows, -1)
        pad = ~mask.unsqueeze(-1).expand(batch, slots, windows).reshape(batch, slots * windows)
        if self.tf is not None:
            h = self.tf(h, src_key_padding_mask=pad)
        attention = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        pooled = torch.einsum("btc,btd->bcd", attention.to(h.dtype), h)
        return (pooled * self.cls_w).sum(-1) + self.cls_b

    def forward(self, x, mask, pos, res=256, train=False):
        if x.ndim != 6 or x.dtype != torch.uint8 or x.shape[3:] != (3, SIZE, SIZE):
            raise ValueError("Input must be uint8 [B,6,K,3,384,384]")
        batch, slots, windows = x.shape[:3]
        self._validate_mask_pos(mask, pos, batch, slots, windows)
        keep = mask.reshape(-1)
        views = x.reshape(batch * slots, windows, *x.shape[3:])[keep].flatten(0, 1)
        encoded = self.enc(make_views(views, res, train, windows).contiguous(memory_format=torch.channels_last))
        feat = encoded.new_zeros(batch * slots, windows, encoded.shape[-1])
        feat[keep] = encoded.reshape(-1, windows, encoded.shape[-1])
        return self.head(feat.reshape(batch, slots, windows, -1), mask, pos)
