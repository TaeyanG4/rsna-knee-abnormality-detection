"""H-B-123 CPU-only real-data transform/serving smoke v1.0.0.

Uses the collected MIL-S model to exercise the shared MIL implementation; no
scientific score or model selection. Synthetic loss verifies backward only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.arms import laterality_v1 as lat
from training.arms import mil
from training.src import raptor_common as common
from training.src.raptor_runtime import configure_runtime
from tools.hb123_laterality_v1 import CORPUS, DEFAULT, validate, write


class ManualCorpus:
    """Independent fixed-layout oracle, used only by this smoke."""
    def __init__(self, base, side):
        self.base, self.side = base, side
        self.spec = base.spec
    def mask(self, uid): return self.base.mask(uid)
    def volume(self, uid):
        v = self.base.volume(uid).copy()
        if self.side == "L":
            v[32:64] = v[32:64, :, ::-1].copy()
        return v


def run(args):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    configure_runtime("none", "cpu")
    validated = validate(args)
    sides = lat.SideMap(args.side_map, validated["side_map_sha256"])
    corpus = common.Corpus(args.corpus)
    selected = [next(u for u in corpus.ids if sides.side(u) == side and corpus.mask(u)[32:].any())
                for side in ("L", "R", "U")]
    targets = {u: np.zeros(12, np.float32) for u in selected}
    result = {"status": "ENGINEERING_SMOKE", "real_data_studies": 3, "sides": ["L", "R", "U"],
              "side_map_sha256": sides.sha256, "threads": torch.get_num_threads(),
              "checkpoint_sha256": lat.sha(args.checkpoint), "input_comparisons": 0}
    start = time.perf_counter()
    for uid in selected:
        for train, k in ((True, 12), (False, 24), (False, 36)):
            base = mil.MILWindows(args.corpus, [uid], targets, k, train, 42)
            ds = lat.CanonicalDataset(base, sides)
            ds.set_epoch(7)
            oracle = mil.MILWindows(args.corpus, [uid], targets, k, train, 42)
            oracle.corpus = ManualCorpus(corpus, sides.side(uid))
            oracle.set_epoch(7)
            state = torch.random.get_rng_state().clone()
            actual, _, actual_uid = ds[0]
            expected, _, expected_uid = oracle[0]
            assert actual_uid == expected_uid and torch.equal(actual, expected)
            assert torch.equal(state, torch.random.get_rng_state())
            assert torch.equal(actual, ds[0][0])
            result["input_comparisons"] += 1
    # Spawn crosses the Windows process boundary and exercises lazy corpus setup.
    ds = lat.CanonicalDataset(mil.MILWindows(args.corpus, selected, targets, 36, False, 42), sides)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1,
                        worker_init_fn=mil.worker_init, collate_fn=mil.collate)
    spawned = [(x, ids) for x, _, ids in loader]
    assert len(spawned) == 3
    assert all(torch.equal(x[0], ds[i][0]) and ids == [selected[i]] for i, (x, ids) in enumerate(spawned))
    result["spawn_parity"] = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["arch"] == "convnext_small.fb_in22k_ft_in1k"
    assert checkpoint["model_kind"] == "finding_spatial_tokens_v1"
    model = mil.FindingMIL(common.build_backbone(checkpoint["arch"], pretrained=False))
    mil.strict_load(model, checkpoint["model"])
    del checkpoint
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=mil.collate)
    predictions, ids, batches = mil.evaluate(model, loader, "cpu", "none", corpus, tta=True)
    assert ids == selected and batches == 3 and np.isfinite(predictions).all()
    # This checks differentiability on real images without reading any labels.
    train_ds = lat.CanonicalDataset(mil.MILWindows(args.corpus, selected[:1], targets, 12, True, 42), sides)
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(train_ds[0][0][None])
    logits.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    result.update(finite_forward=True, finite_backward=True, synthetic_loss_only=True,
                  readout="K36 plane-flip TTA", cuda_initialized=torch.cuda.is_initialized(),
                  elapsed_seconds=time.perf_counter() - start,
                  prediction_sha256=__import__("hashlib").sha256(predictions.tobytes()).hexdigest())
    assert not result["cuda_initialized"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "predictions.npy", predictions)
    write(out / "smoke.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side-map", default=DEFAULT)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--checkpoint", default=".local/hb105-collected-v1/s42/outputs/model_best.pt")
    ap.add_argument("--out", default=".local/scratch/sol-5/hb123/cpu-smoke")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
