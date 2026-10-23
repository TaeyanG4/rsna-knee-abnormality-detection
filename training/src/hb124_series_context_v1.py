"""H-B-124 within-real-series window context, v1.0.0 (2026-10-13).

New module; frozen MIL sources stay unchanged. Positions describe sparse cache
source indices, never native adjacency. No label/report access or network I/O.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

VERSION = "1.0.0"
POLICY = "real-series-distinct-source-width3-residual-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class SeriesProvenance:
    """Map the EXISTING sampled centers to audited per-cache-plane provenance.

    The sidecar contains acquisition/cache metadata only. Missing provenance on
    any nonzero cache plane is a hard failure, not an inferred slot identity.
    """

    def __init__(self, corpus, picks_path=None):
        self.path = Path(picks_path or corpus.dir / "picks.jsonl")
        self.sha256 = sha(self.path)
        self.by_uid = {}
        self.corpus = corpus
        sizes = [int(s[2]) for s in corpus.spec["slots"]]
        require(sum(sizes) == corpus.depth, "slot depth differs")
        require(sha(corpus.dir / "ids.npy") == corpus.manifest["ids_sha256"], "corpus IDs changed")
        require(sha(corpus.dir / "masks.npy") == corpus.manifest["masks_sha256"], "corpus masks changed")
        for line in self.path.open(encoding="utf-8"):
            row = json.loads(line)
            uid = row["uid"]
            require(uid in corpus.row and uid not in self.by_uid, "unknown or duplicate metadata study")
            require(len(row["slots"]) == len(sizes), "slot count differs")
            shard = corpus.manifest["shards"][row["shard"]]
            require(shard["start"] + row["row"] == corpus.row[uid]
                    and 0 <= row["row"] < shard["stop"] - shard["start"], "metadata cache row differs")
            series = np.full(corpus.depth, -1, np.int64)
            position = np.zeros(corpus.depth, np.float32)
            names, counts, start = {}, {}, 0
            for size, entry in zip(sizes, row["slots"]):
                if entry is not None and entry["files"] > 0:
                    name, count, indices = entry["series"], entry["files"], entry["slices"]
                    require(isinstance(name, str) and bool(name), "missing real series UID")
                    require(type(count) is int and count > 0 and len(indices) == size, "source count differs")
                    require(all(type(i) is int and 0 <= i < count for i in indices)
                            and all(a <= b for a, b in zip(indices, indices[1:])), "invalid source indices")
                    require(name not in counts or counts[name] == count, "same-series source count differs")
                    counts[name] = count
                    names.setdefault(name, len(names))
                    series[start:start + size] = names[name]
                    position[start:start + size] = np.asarray(indices, np.float32) / max(count - 1, 1)
                start += size
            require(not np.any((corpus.mask(uid) > 0) & (series < 0)), "nonzero cache plane lacks provenance")
            self.by_uid[uid] = (series, position)
        require(set(self.by_uid) == set(corpus.row), "metadata coverage differs")

    def windows(self, uid, centers):
        require(uid in self.by_uid, "study provenance missing")
        require(all(type(c) in (int, np.int64, np.int32) and -1 <= c < self.corpus.depth
                    for c in centers), "invalid cache centers")
        centers = np.asarray(centers, np.int64)
        series, position = self.by_uid[uid]
        valid = centers >= 0
        safe = np.maximum(centers, 0)
        valid &= self.corpus.mask(uid)[safe] > 0
        require(not np.any(valid & (series[safe] < 0)), "sampled window lacks provenance")
        return {"series": torch.from_numpy(np.where(valid, series[safe], -1)),
                "position": torch.from_numpy(np.where(valid, position[safe], np.float32(0))),
                "valid": torch.from_numpy(valid)}


class WithinSeriesContext(nn.Module):
    """Width-3 bottleneck over previous/current/next window summaries.

    Equal source positions (including resampled/crop duplicates) are averaged
    for neighbor lookup; they are never called cross-slice neighbors. The
    current summary remains each window's own feature. Missing neighbors are
    zeroed. Single-position series and invalid windows remain exact identity.
    """

    def __init__(self, features, hidden=32):
        super().__init__()
        require(features > 0 and hidden > 0, "invalid feature dimensions")
        # Adding the arm cannot shift the control's RNG/data/dropout stream.
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed also seeds CUDA (including lazy callbacks),
            # which fork_rng(devices=[]) deliberately does not restore.
            torch.random.default_generator.manual_seed(124)
            self.norm = nn.LayerNorm(features)
            self.conv = nn.Conv1d(features + 1, hidden, kernel_size=3)
            self.out = nn.Linear(hidden, features, bias=False)
            nn.init.zeros_(self.out.weight)
        self.features = features

    @staticmethod
    def neighbor_weights(series, position, valid):
        same = (series[:, :, None] == series[:, None, :]) & valid[:, :, None] & valid[:, None, :]
        delta = position[:, None, :] - position[:, :, None]
        left = same & (delta < 0)
        right = same & (delta > 0)
        left_gap = delta.masked_fill(~left, -torch.inf).amax(-1)
        right_gap = delta.masked_fill(~right, torch.inf).amin(-1)
        lw = left & (delta == left_gap[..., None])
        rw = right & (delta == right_gap[..., None])
        # Form normalized weights in float32; the feature matmul follows AMP.
        lw = lw.float() / lw.sum(-1, keepdim=True).clamp_min(1)
        rw = rw.float() / rw.sum(-1, keepdim=True).clamp_min(1)
        has_left, has_right = left.any(-1), right.any(-1)
        gaps = torch.stack((left_gap.masked_fill(~has_left, 0), torch.zeros_like(left_gap),
                            right_gap.masked_fill(~has_right, 0)), -1)
        return lw, rw, gaps, has_left | has_right

    def forward(self, tokens, token_mask, series, position, valid):
        require(tokens.ndim == 4 and tokens.shape[-1] == self.features, "expected B,K,T,F tokens")
        b, k, t, f = tokens.shape
        require(token_mask.shape == (b, k, t) and token_mask.dtype == torch.bool, "token mask differs")
        require(series.shape == position.shape == valid.shape == (b, k), "window metadata shape differs")
        require(series.dtype == torch.int64 and valid.dtype == torch.bool and position.is_floating_point(),
                "window metadata dtype differs")
        require(all(x.device == tokens.device for x in (token_mask, series, position, valid)), "device differs")
        require(bool(torch.isfinite(position).all()) and bool(((position >= 0) & (position <= 1)).all()),
                "invalid normalized source position")
        require(bool((series[valid] >= 0).all()), "valid window lacks real-series provenance")
        present = valid & token_mask.any(-1)
        active_tokens = token_mask & present[..., None]
        clean = tokens.masked_fill(~active_tokens[..., None], 0)
        summary = clean.sum(-2) / active_tokens.sum(-1).clamp_min(1)[..., None]
        summary = self.norm(summary).masked_fill(~present[..., None], 0)
        lw, rw, gaps, has_neighbor = self.neighbor_weights(series, position, present)
        neighbors = torch.stack((torch.bmm(lw.to(summary.dtype), summary), summary,
                                 torch.bmm(rw.to(summary.dtype), summary)), -1)
        inputs = torch.cat((neighbors, gaps.to(summary.dtype).unsqueeze(-2)), -2).reshape(b*k, f+1, 3)
        residual = self.out(torch.nn.functional.gelu(self.conv(inputs).squeeze(-1))).reshape(b, k, f)
        residual = residual.masked_fill(~(present & has_neighbor)[..., None], 0)
        return tokens + residual[..., None, :].to(tokens.dtype) * active_tokens[..., None]

    def encoded(self, encoded, metadata):
        """Apply between shared model.encode(images) and model.head(encoded)."""
        tokens, mask = encoded
        require(tokens.ndim == 3 and mask.shape == tokens.shape[:2], "encoded shape differs")
        b, k = metadata["series"].shape
        require(b == tokens.shape[0] and k > 0 and tokens.shape[1] % k == 0, "encoded window count differs")
        output = self(tokens.reshape(b, k, -1, self.features), mask.reshape(b, k, -1), **metadata)
        return output.reshape_as(tokens), mask
