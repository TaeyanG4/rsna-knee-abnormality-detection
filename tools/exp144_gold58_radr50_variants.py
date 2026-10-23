"""EXP144 v1.0.0 — Gold58 (official labels) check of RadR50 variants that the Dread ruler cannot arbitrate.

Why (B, 2026-10-09, user: "<owner remark removed>"): EXP138's multi-head bundle loses on Dread
(-0.0015) but wins on the independent flight_v4 reference (+0.0113) vs the AUG1 single head (D-B-073);
Dread is the single head's own training source, so the Dread ruler is biased for this label-mix change.
The 4,343-study OOF cohort contains no Gold58 study, so this tool runs every fold checkpoint on the 58
official-label studies (none of them in any training/validation fold) and compares macro AUC.

Variants (two seeds x five folds each; plain and EXP124 coronal/axial flip TTA):
  base   = EXP113 (s42) / EXP118 (s43)                 pre-AUG1 single head (re-uses EXP130 local outputs)
  aug1   = EXP127_AUG1 F0 + EXP134_AUG1 F1-F4           AUG1 single head
  bundle = EXP138_BUNDLE_AUG1_MH F0-F4                 AUG1 + four label-source heads (served rank mean)
Fold scores: probabilities (single head) or per-pass head rank mean (bundle), arithmetic mean over folds,
then seed mean. Report only: 58 studies give wide intervals; no adoption gate.

  python tools/exp144_gold58_radr50_variants.py predict --device cuda --workers 2
  python tools/exp144_gold58_radr50_variants.py compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".local/exp144-gold58"
EXP130_LOCAL = ROOT / ".local/exp130-gold58/local"
VARIANTS = {
    "aug1": {42: ["EXP127_AUG1_S42_F0"] + [f"EXP134_AUG1_S42_F{f}" for f in range(1, 5)],
             43: ["EXP127_AUG1_S43_F0"] + [f"EXP134_AUG1_S43_F{f}" for f in range(1, 5)]},
    "bundle": {s: [f"EXP138_BUNDLE_AUG1_MH_S{s}_F{f}" for f in range(5)] for s in (42, 43)},
}


def predict(device="cuda", workers=2):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "training/src"))
    from tools import exp130_gold58_compare as g130
    from tools import exp124_flip_tta as flip
    from tools import exp117_multihead as e117
    from training.src.raptor_common import Corpus, build_model
    from training.src.raptor_radimagenet import CenterSliceWindows
    from training.src.raptor_runtime import configure_runtime
    g130.need(not OUT.exists(), "output exists; refusing overwrite")
    ids, _ = g130.official_gold(ROOT)
    configure_runtime("none" if device == "cpu" else "fp16", device, policy="strict")
    p = json.loads((ROOT / "artifacts/experiments/EXP113_RADR50_LOCAL_PROTOCOL_V1/manifest.json").read_text(encoding="utf-8"))
    corpus = Corpus(ROOT / p["paths"]["corpus"])
    g130.need(corpus.contract == "raptor64_336_c140_s02" and set(ids) <= set(corpus.ids), "corpus/Gold58 mismatch")
    for fold in range(5):  # same EXP099 fold contract for every variant (paired screens)
        f = json.loads((ROOT / p["folds"][str(fold)]["ordered_ids"]["path"]).read_text(encoding="utf-8"))
        g130.need(not (set(ids) & (set(f["train"]) | set(f["val"]))), "Gold58 overlaps a training/validation fold")
    args = p["common_args"]
    ranges = flip.plane_ranges(corpus.spec)
    outputs, receipt = {}, {}
    for variant, seeds in VARIANTS.items():
        for seed, tags in seeds.items():
            labels = {uid: np.zeros(len(g130.TARGETS), np.float32) for uid in ids}
            ds = CenterSliceWindows(corpus.dir, ids, labels, 224, 36, False, args["norm"], seed,
                                    window_policy=args["window_policy"], input_policy=args["input_policy"])
            loader = DataLoader(flip.PlaneAwareWindows(ds, corpus, ranges), batch_size=4, shuffle=False,
                                num_workers=workers, collate_fn=flip.collate_pair)
            plain, tta, hashes = [], [], []
            for fold, tag in enumerate(tags):
                path = ROOT / "artifacts/experiments" / tag / "model_best.pt"
                ck = torch.load(path, map_location="cpu", weights_only=False)
                g130.need(ck.get("tag") == tag and ck.get("fold") == fold, f"checkpoint tag/fold differs: {tag}")
                model = build_model(ck["arch"], pretrained=False)
                name = "exp113"
                if variant == "bundle":
                    model = e117.build_multihead(model, len(e117.HEAD_NAMES)); name = "exp117"
                model.load_state_dict(ck["model"], strict=True)
                model = model.to(device).eval()
                a, b, got = flip.infer_pair(model, loader, device, name)
                g130.need(got == ids and a.shape == (58, 12), "cohort/shape differs")
                plain.append(a); tta.append((a + b) / 2); hashes.append(g130.sha256(path))
                del model, ck
                if device == "cuda":
                    torch.cuda.empty_cache()
            outputs[f"{variant}_s{seed}_plain"] = np.mean(plain, axis=0).tolist()
            outputs[f"{variant}_s{seed}_tta"] = np.mean(tta, axis=0).tolist()
            receipt[f"{variant}_s{seed}"] = {"tags": tags, "checkpoints_sha256": hashes}
            print(json.dumps({"done": f"{variant}_s{seed}"}), flush=True)
    OUT.mkdir(parents=True)
    for k, v in outputs.items():
        g130.write_predictions(OUT / f"gold58_{k}.csv", ids, v)
    (OUT / "receipt.json").write_text(json.dumps({"schema": "exp144-v1", "studies": 58, "device": device,
                                                  "k_eval": 36, "models": receipt}, indent=2) + "\n", encoding="utf-8")
    return sorted(outputs)


def compare(draws=2000):
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from tools import exp130_gold58_compare as g130
    ids, labels = g130.official_gold(ROOT)
    y = np.asarray(labels, float)
    def load(path):
        return np.asarray(g130.prediction_csv(path, ids), float)
    cand = {}
    for v in ("base", "aug1", "bundle"):
        for mode in ("plain", "tta"):
            src = EXP130_LOCAL if v == "base" else OUT
            pre = "" if v == "base" else f"{v}_"
            s = [load(src / f"gold58_{pre}s{seed}_{mode}.csv") for seed in (42, 43)]
            cand[f"{v}_{mode}"] = (s[0] + s[1]) / 2
    from sklearn.metrics import roc_auc_score
    elig = [t for t in range(12) if 0 < y[:, t].sum() < len(y)]
    def macro(p, ix):
        out = []
        for t in elig:
            yy = y[ix, t]
            if 0 < yy.sum() < len(yy):
                out.append(roc_auc_score(yy, p[ix, t]))
        return float(np.mean(out))
    full = np.arange(len(ids))
    res = {k: macro(v, full) for k, v in cand.items()}
    rng = np.random.default_rng(20261009)
    draws_ix = [rng.integers(0, len(ids), len(ids)) for _ in range(draws)]
    pairs = [("bundle_tta", "aug1_tta"), ("bundle_plain", "aug1_plain"), ("aug1_tta", "base_tta"),
             ("bundle_tta", "base_tta")]
    report = {"macro_auc": res, "eligible_targets": len(elig), "pairs": {}}
    for a, b in pairs:
        d = np.array([macro(cand[a], ix) - macro(cand[b], ix) for ix in draws_ix])
        report["pairs"][f"{a} - {b}"] = {"delta": res[a] - res[b], "ci95": np.percentile(d, [2.5, 97.5]).tolist(),
                                         "p_positive": float((d > 0).mean())}
    report["per_target_bundle_tta_minus_aug1_tta"] = {
        g130.TARGETS[t]: roc_auc_score(y[:, t], cand["bundle_tta"][:, t]) - roc_auc_score(y[:, t], cand["aug1_tta"][:, t])
        for t in elig}
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("predict"); pr.add_argument("--device", default="cuda"); pr.add_argument("--workers", type=int, default=2)
    sub.add_parser("compare")
    a = ap.parse_args(argv)
    print(json.dumps(predict(a.device, a.workers) if a.cmd == "predict" else compare(), indent=2))


if __name__ == "__main__":
    main()
