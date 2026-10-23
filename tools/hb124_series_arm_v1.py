"""H-B-124 v1.0.0 CPU metadata audit and real-image engineering smoke.

No labels/reports, downloads, training launch or metric/epoch selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from training.arms import mil
from training.src.hb124_series_context_v1 import SeriesProvenance, WithinSeriesContext, POLICY, sha

CORPUS = "storage/cache/derived/RAPTOR64_336_C140_S02"
SCRATCH = ROOT / ".local/scratch/sol-3/hb124"


def audit(corpus, provenance):
    counts = {"studies": len(corpus), "sampled_windows": 0, "valid_windows": 0,
              "same_series_directed_links": 0, "cross_series_links": 0, "empty_bags": 0}
    # All real cache studies, deterministic serving K36; no pixel/label load.
    for uid in corpus.ids:
        centers = mil.common.select_centers(corpus.mask(uid), 64, 36, corpus.spec, "within-slot-valid")
        meta = {k: v[None] for k, v in provenance.windows(uid, centers).items()}
        lw, rw, _, _ = WithinSeriesContext.neighbor_weights(**meta)
        links = (lw + rw) > 0
        different = meta["series"][:, :, None] != meta["series"][:, None, :]
        counts["sampled_windows"] += len(centers)
        counts["valid_windows"] += int(meta["valid"].sum())
        counts["same_series_directed_links"] += int(links.sum())
        counts["cross_series_links"] += int((links & different).sum())
        counts["empty_bags"] += int(not meta["valid"].any())
    assert counts["cross_series_links"] == 0
    return {"status": "PROVENANCE_PASS", "policy": POLICY, "counts": counts,
            "bindings": {"manifest": sha(corpus.dir / "manifest.json"), "picks": provenance.sha256,
                         "builder": sha(ROOT / "training/src/raptor_corpus.py")}}


def smoke(corpus, provenance):
    start = time.perf_counter()
    # Reuse the immutable Small factory/initial state only for engineering;
    # launch integration must bind sol-2's fresh shared control, not this smoke.
    config = json.loads((ROOT / "training/configs/hb105-small-s42-f0-v1.json").read_text(encoding="utf-8"))
    torch.manual_seed(42)
    model = mil.build_model(config["config"]["model"], ROOT).eval()
    model.series_context = WithinSeriesContext(model.backbone.num_features)
    # Exact current data code with synthetic target placeholders: never read labels.
    uid = str(corpus.ids[0])
    dataset = mil.MILWindows(corpus.dir, [uid], {uid: np.zeros(12, np.float32)}, 12, True, 42)
    dataset.set_epoch(0)
    x, _, _ = dataset[0]
    rng = mil.common.study_rng(42, 0, uid)
    centers = mil.common.select_centers(corpus.mask(uid), 64, 12, corpus.spec, "within-slot-valid", rng)
    metadata = {k: v[None] for k, v in provenance.windows(uid, centers).items()}
    encoded = model.encode(x[None])
    with torch.no_grad():
        control_logits = model.head(encoded)
    output = model.series_context.encoded(encoded, metadata)
    logits = model.head(output)
    assert torch.equal(logits, control_logits), "zero-init no longer equals control"
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    finite_gradients = len(gradients)
    assert model.series_context.out.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    # Exercise a learned residual and strict reload without copying a 350MB
    # backbone checkpoint into scratch. Shared lifecycle integration must also
    # test its own complete checkpoint when the shared runner is available.
    context_optimizer = torch.optim.SGD(model.series_context.parameters(), lr=.01)
    context_optimizer.step()
    model.zero_grad(set_to_none=True)
    del encoded, output, logits, loss, gradients
    SCRATCH.mkdir(parents=True, exist_ok=True)
    checkpoint = SCRATCH / "context-smoke.pt"
    torch.save(model.series_context.state_dict(), checkpoint)
    restored = WithinSeriesContext(model.backbone.num_features)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    # Full real encoder passes on evaluation and flip-TTA views at the exact
    # registered K values; metadata is unchanged by image-plane flips.
    paths = []
    max_effect = 0.
    with torch.no_grad():
        for k in (24, 36):
            ds = mil.MILWindows(corpus.dir, [uid], {uid: np.zeros(12, np.float32)}, k, False, 42)
            images, _, _ = ds[0]
            centers = mil.common.select_centers(corpus.mask(uid), 64, k, corpus.spec, "within-slot-valid")
            meta = {name: v[None] for name, v in provenance.windows(uid, centers).items()}
            for flip in (False, True):
                inputs = mil.flip_inputs(images[None], [uid], corpus, k) if flip else images[None]
                encoded = model.encode(inputs)
                contextual = model.series_context.encoded(encoded, meta)
                reloaded = restored.encoded(encoded, meta)
                assert torch.equal(contextual[0], reloaded[0]), "context checkpoint changed features"
                max_effect = max(max_effect, float((contextual[0] - encoded[0]).abs().max()))
                logits = model.head(contextual)
                assert logits.shape == (1, 12) and torch.isfinite(logits).all()
                paths.append(f"k{k}-{'flip-' if flip else ''}real-encoder-context-head")
        empty_count = 0
        for empty_uid in corpus.ids:
            centers = mil.common.select_centers(corpus.mask(empty_uid), 64, 12, corpus.spec, "within-slot-valid")
            if all(center == -1 for center in centers):
                ds = mil.MILWindows(corpus.dir, [empty_uid], {empty_uid: np.zeros(12, np.float32)}, 12, False, 42)
                images, _, _ = ds[0]
                meta = {name: value[None] for name, value in provenance.windows(empty_uid, centers).items()}
                encoded = model.encode(images[None])
                contextual = model.series_context.encoded(encoded, meta)
                assert torch.equal(encoded[0], contextual[0]), "empty fallback changed"
                assert torch.equal(model.head(encoded), model.head(contextual))
                assert torch.isfinite(model.head(contextual)).all()
                empty_count += 1
                break
        assert empty_count == 1, "real all-empty fallback was not exercised"
        paths.append("real-empty-study-fallback")
    assert max_effect > 0, "smoke did not exercise a learned interaction"
    return {"status": "REAL_IMAGE_CPU_SMOKE_PASS", "policy": POLICY,
            "model": config["config"]["model"]["arch"], "real_encoder_backward": "B1 K12 224px",
            "synthetic_targets": True, "label_rows_read": 0, "control_zero_init_maxabs": 0,
            "finite_gradients": finite_gradients,
            "context_parameters": sum(p.numel() for p in model.series_context.parameters()),
            "strict_context_reload": True, "learned_residual_maxabs": max_effect,
            "extra_paths": paths, "elapsed_seconds": round(time.perf_counter() - start, 3),
            "cpu_threads": torch.get_num_threads(), "cuda_initialized": torch.cuda.is_initialized()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("metadata", "smoke"))
    args = parser.parse_args(argv)
    corpus = mil.common.Corpus(ROOT / CORPUS)
    provenance = SeriesProvenance(corpus)
    result = audit(corpus, provenance) if args.command == "metadata" else smoke(corpus, provenance)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / (args.command + ".json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
