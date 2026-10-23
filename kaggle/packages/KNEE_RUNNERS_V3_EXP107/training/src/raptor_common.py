"""Shared pieces of the Raptor baseline (dreaddevelopment recipe) made local.

Contract names identify a corpus layout. Checkpoints only make sense on the corpus
contract they were trained on, so every artifact records the contract name.

Reference: public notebooks dreaddevelopment/knee-mri-training-the-twelve-finding-model and
pjmathematician/rsna-knee-d4-blend (view table). Nothing here is executed from those
notebooks; the logic was re-implemented from their published source.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = "StudyInstanceUID"

SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]

# Corpus contracts (public Raptor arms). Values from the public view table.
CONTRACTS = {
    # maxspan-v5 (public checkpoint raptor_ft_coatnet_v5_full_swa.pt, gold 0.9214 per its header)
    "raptor64_336_c140_s02": dict(slots=SLOTS64, img=336, crop_mm=140.0, span=(0.02, 0.98)),
    # native384dense-v10
    "raptor64_384_c140_s02": dict(slots=SLOTS64, img=384, crop_mm=140.0, span=(0.02, 0.98)),
    # native384-v8
    "raptor44_384_c140_s06": dict(slots=SLOTS44, img=384, crop_mm=140.0, span=(0.06, 0.94)),
}
PUBLIC_ARMS = {
    "maxspan-v5": dict(file="dreaddevelopment__raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt", contract="raptor64_336_c140_s02", k_eval=62),
    "native384dense-v10": dict(file="dreaddevelopment__raptor-knee-native384dense/raptor_ft_coatnet_v10_full.pt", contract="raptor64_384_c140_s02", k_eval=62),
    "native384-v8": dict(file="dreaddevelopment__raptor-knee-native384/raptor_ft_coatnet_v8_full_swa.pt", contract="raptor44_384_c140_s06", k_eval=42),
}


def workspace_root() -> Path:
    env = os.environ.get("KNEE_WORKSPACE_ROOT")
    here = Path(env).resolve() if env else Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "workspace.toml").is_file():
            return candidate
    raise RuntimeError("config/workspace.toml not found; set KNEE_WORKSPACE_ROOT")


ROOT = workspace_root()
COMP = ROOT / "storage" / "competition"
CACHE = ROOT / "storage" / "cache" / "derived"
EXPERIMENTS = ROOT / "artifacts" / "experiments"
LOGS = ROOT / "artifacts" / "logs"
WEIGHTS = ROOT / "storage" / "weights"


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        while chunk := f.read(1 << 24):
            h.update(chunk); read += len(chunk)
            if limit and read >= limit:
                break
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def environment() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "hostname": platform.node(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, gpu_memory_gib=round(p.total_memory / 2**30, 2), bf16=torch.cuda.is_bf16_supported())
    except (ImportError, RuntimeError, AssertionError):
        pass  # hidden GPU (CUDA_VISIBLE_DEVICES="") or no torch: record what we have
    for mod in ("timm", "numpy", "pydicom", "cv2"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        info["driver"] = out.stdout.strip()
    except Exception:
        pass
    return info


# ------------------------------------------------------------------ corpus access
class Corpus:
    """Memory-mapped sharded corpus written by raptor_corpus.py."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.contract = self.manifest["contract"]
        self.spec = self.manifest["spec"]
        self.ids = np.load(self.dir / "ids.npy", allow_pickle=True).astype(str)
        self.masks = np.load(self.dir / "masks.npy")
        self.row = {u: i for i, u in enumerate(self.ids)}
        self._shards = [None] * len(self.manifest["shards"])
        self._bounds = [(s["start"], s["stop"], s["file"]) for s in self.manifest["shards"]]
        self.depth = int(self.spec["depth"])

    def __len__(self):
        return len(self.ids)

    def __contains__(self, uid):
        return uid in self.row

    def _shard(self, k):
        if self._shards[k] is None:
            self._shards[k] = np.load(self.dir / self._bounds[k][2], mmap_mode="r")
        return self._shards[k]

    def volume(self, uid):
        r = self.row[uid]
        for k, (start, stop, _) in enumerate(self._bounds):
            if start <= r < stop:
                return self._shard(k)[r - start]
        raise KeyError(uid)

    def mask(self, uid):
        return self.masks[self.row[uid]]


def eval_centers(mask, depth, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, depth - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def train_centers(mask, depth, k, rng: random.Random):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, depth - 2))]
    pool = cs * (k // len(cs) + 1)
    rng.shuffle(pool)
    return pool[:k]


def study_rng(seed, epoch, uid, stream="windows"):
    """Stable across processes/OS; input sampling must not depend on wall time."""
    key = json.dumps([int(seed), int(epoch), str(uid), str(stream)], separators=(",", ":"))
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def select_centers(mask, depth, k, spec, policy="legacy", rng=None):
    """Opt-in within-slot, mask-valid triplets; -1 means an explicit empty input.

    The cache mask measures nonzero pixels, not acquisition/clinical validity.
    Legacy behavior stays available for existing checkpoints.
    """
    if policy == "legacy":
        return train_centers(mask, depth, k, rng) if rng is not None else eval_centers(mask, depth, k)
    if policy not in ("global-valid", "within-slot-valid"):
        raise ValueError(f"unknown window policy: {policy}")
    if k <= 0 or len(mask) != depth:
        raise ValueError("invalid window count or mask depth")
    sizes = [int(slot[2]) for slot in spec["slots"]]
    if sum(sizes) != depth:
        raise ValueError("slot sizes do not match corpus depth")
    cs, start = [], 0
    for size in ([depth] if policy == "global-valid" else sizes):
        cs.extend(c for c in range(start + 1, start + size - 1) if np.all(mask[c - 1:c + 2] > 0))
        start += size
    if not cs:
        return [-1] * k
    if rng is not None:
        pool = cs * (k // len(cs) + 1)
        rng.shuffle(pool)
        return pool[:k]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def windows_from(vol, centers, res):
    """(K,3,res,res) float32 in [0,1]; three consecutive cached planes per window."""
    import torch
    import torch.nn.functional as F
    depth = vol.shape[0]
    out = np.empty((len(centers), 3, res, res), np.float32)
    for j, c in enumerate(centers):
        if c == -1:
            out[j] = 0
            continue
        c = max(1, min(int(c), depth - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res or t.shape[-2] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        out[j] = t.numpy()
    return out


# ------------------------------------------------------------------ model
def build_backbone(arch: str, pretrained: bool):
    import timm
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    kw.update(global_pool="token", dynamic_img_size=True) if is_vit else kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


def build_model(arch: str, pretrained: bool, drop: float = 0.2):
    import torch
    import torch.nn as nn

    class RaptorClassifier(nn.Module):
        """Backbone + per-finding attention pooling (verbatim structure of the public arm)."""

        def __init__(self, backbone, F_dim, n=12):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(F_dim)
            self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
            self.clsW = nn.Parameter(torch.zeros(n, F_dim))
            self.clsb = nn.Parameter(torch.zeros(n))
            nn.init.trunc_normal_(self.clsW, std=0.02)
            self.n = n

        def encode(self, x):
            B, K = x.shape[:2]
            return self.backbone(x.flatten(0, 1)).view(B, K, -1)

        def head(self, feats):
            h = self.norm(feats)
            a = torch.softmax(self.att(h), dim=1)
            pooled = torch.einsum("bkn,bkf->bnf", a, h)
            return (pooled * self.clsW).sum(-1) + self.clsb

        def forward(self, x):
            return self.head(self.encode(x))

    bb = build_backbone(arch, pretrained)
    return RaptorClassifier(bb, bb.num_features)


def load_checkpoint(path: Path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["model"], strict=True)
    meta = {k: v for k, v in ck.items() if k != "model"}
    return model, meta


# ------------------------------------------------------------------ metrics / io
def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    per = {}
    for j, name in enumerate(LAB):
        col = y[:, j]
        ok = np.isfinite(col)
        if ok.sum() and len(set(col[ok].round().astype(int))) > 1:
            per[name] = float(roc_auc_score(col[ok].round().astype(int), p[ok, j]))
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def write_predictions(path: Path, ids, probs):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(probs, columns=LAB); df.insert(0, UID, list(ids))
    df.to_csv(path, index=False)
    return path


def load_label_csv(path: Path, columns=None):
    """uid -> float32[12]; accepts the public soft-label CSV layouts (plain or *_soft columns)."""
    import pandas as pd
    df = pd.read_csv(path, dtype={UID: str})
    if df[UID].isna().any() or df[UID].duplicated().any():
        raise ValueError(f"missing or duplicate study UID in {path}")
    cols = columns or [c if c in df.columns else c + "_soft" for c in LAB]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing label columns {missing}")
    df = df.dropna(subset=cols)
    arr = df[cols].to_numpy(dtype=np.float32)
    if arr.min() < 0 or arr.max() > 1:
        raise ValueError(f"{path}: label values outside [0,1]")
    return dict(zip(df[UID].astype(str), arr)), sha256_text(df[[UID] + cols].to_csv(index=False, lineterminator="\n"))


def gold_labels():
    import pandas as pd
    tr = pd.read_csv(COMP / "train.csv", dtype={UID: str})
    g = tr[tr[LAB].notna().all(axis=1)]
    return dict(zip(g[UID], g[LAB].to_numpy(dtype=np.float32)))


def load_folds(path: Path):
    import pandas as pd
    df = pd.read_csv(path, dtype={UID: str})
    return dict(zip(df[UID], df["fold"].astype(int))), sha256_text(df[[UID, "fold"]].to_csv(index=False, lineterminator="\n"))
