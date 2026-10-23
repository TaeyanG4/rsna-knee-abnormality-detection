"""Freeze H-C-15 inputs for a new standalone recipe; no pixels or model execution.

Usage: system Python tools/open_recipe_prepare.py --out <fresh directory>
Only stdlib is used. Reads existing data/index read-only and creates a new manifest.
An input inventory is not CPU/GPU readiness and never dispatches a job.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import shutil
import sqlite3
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def validate_unique_split(train, val, train_n=3455, val_n=888):
    """Never elide input guards when Python optimization is enabled."""
    require(len(train) == len(set(train)) == train_n, "Training UID count/uniqueness differs")
    require(len(val) == len(set(val)) == val_n, "Validation UID count/uniqueness differs")
    require(not set(train) & set(val), "Training/validation UID overlap")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def binding(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            "bytes": path.stat().st_size, "sha256": sha(path)}

def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def uid_sha(uids):
    return hashlib.sha256("\n".join(uids).encode()).hexdigest()

def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]

def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / "artifacts" / "experiments"):
        raise ValueError("Output must be a fresh artifacts/experiments directory")
    if out.exists():
        raise FileExistsError("Refuse to overwrite frozen preparation")
    base_path = ROOT / "artifacts/experiments/EXP094_B8_VARIANT_PROTOCOL_V4/manifest.json"
    base = json.loads(base_path.read_text())
    label_path, ids_path, groups_path = [ROOT / p for p in (
        base["paths"]["labels"], base["ordered_ids"]["path"], base["groups"]["path"])]
    for p, expected in [(label_path, base["labels"]["sha256"]),
                        (ids_path, base["ordered_ids"]["sha256"]),
                        (groups_path, base["groups"]["sha256"])]:
        if sha(p) != expected:
            raise ValueError(f"Control input drift: {p}")
    ids = json.loads(ids_path.read_text())
    train, val = ids["train"], ids["val"]
    validate_unique_split(train, val)
    require(uid_sha(train) == base["ordered_ids"]["train_sha256"], 'Input contract failed: uid_sha(train) == base["ordered_ids"]["train_sha256"]')
    require(uid_sha(val) == base["ordered_ids"]["val_sha256"], 'Input contract failed: uid_sha(val) == base["ordered_ids"]["val_sha256"]')
    require(not set(train) & set(val), 'Input contract failed: not set(train) & set(val)')
    cohort = set(train) | set(val)
    gold_path = ROOT / "storage/gold/gold_labels.csv"
    gold = {r[UID] for r in read_rows(gold_path)}
    require(len(gold) == 58 and not cohort & gold, 'Input contract failed: len(gold) == 58 and not cohort & gold')
    require(not cohort & set(base["excluded_uids"]), 'Input contract failed: not cohort & set(base["excluded_uids"])')
    groups = {r[UID]: r for r in read_rows(groups_path)}
    require(all(groups[u]["split"] == "train" for u in train), 'Input contract failed: all(groups[u]["split"] == "train" for u in train)')
    require(all(groups[u]["split"] == "val" for u in val), 'Input contract failed: all(groups[u]["split"] == "val" for u in val)')
    tc, vc = [{groups[u]["component_id"] for u in a} for a in (train, val)]
    require(len(tc) == 3323 and len(vc) == 873 and not tc & vc, 'Input contract failed: len(tc) == 3323 and len(vc) == 873 and not tc & vc')
    label_rows = read_rows(label_path)
    require(list(label_rows[0]) == [UID] + TARGETS, 'Input contract failed: list(label_rows[0]) == [UID] + TARGETS')
    labels = {r[UID]: [f32(r[t]) for t in TARGETS] for r in label_rows}
    require(len(labels) == len(label_rows) and cohort <= labels.keys(), 'Input contract failed: len(labels) == len(label_rows) and cohort <= labels.keys()')
    require(all(math.isfinite(x) and 0 <= x <= 1 for u in cohort for x in labels[u]), 'Input contract failed: all(math.isfinite(x) and 0 <= x <= 1 for u in cohort for x in labels[u])')
    # Python round on exact float32 values implements ties-to-even, as numpy.rint does.
    reference = [[int(round(x)) for x in labels[u]] for u in val]
    counts = [sum(row[c] for row in reference) for c in range(12)]
    require(all(0 < n < 888 for n in counts), 'Input contract failed: all(0 < n < 888 for n in counts)')
    control_path = ROOT / "artifacts/experiments/EXP084_B_G1/oof_best.csv"
    control_rows = read_rows(control_path)
    require([r[UID] for r in control_rows] == val, 'Input contract failed: [r[UID] for r in control_rows] == val')
    require(all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1 for r in control_rows for t in TARGETS), 'Input contract failed: all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1 for r in control_rows for t in TARGETS)')
    series_path = ROOT / "storage/competition/train_series.csv"
    all_series = read_rows(series_path)
    require(len({r["SeriesInstanceUID"] for r in all_series}) == len(all_series), 'Input contract failed: len({r["SeriesInstanceUID"] for r in all_series}) == len(all_series)')
    series = [r for r in all_series if r[UID] in cohort]
    by_study = defaultdict(list)
    for r in series:
        require(r["Anatomical_Plane"] in PLANES and r["Fat_Suppression"] in ("0", "1"), 'Input contract failed: r["Anatomical_Plane"] in PLANES and r["Fat_Suppression"] in ("0", "1")')
        by_study[r[UID]].append(r)
    require(by_study.keys() == cohort, 'Input contract failed: by_study.keys() == cohort')
    app_db = ROOT / ".local/kneeops/app.sqlite"
    with sqlite3.connect(app_db.as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        raw = conn.execute("SELECT value FROM meta WHERE key='data_library'").fetchone()[0]
    archive = json.loads(raw)["archives"][0]
    index_path = Path(archive["index"]).resolve()
    archive_path = Path(archive["path"].replace("workspace://", str(ROOT) + "/")).resolve()
    require(archive_path.is_file() and index_path.is_file(), 'Input contract failed: archive_path.is_file() and index_path.is_file()')
    # Counts are file counts, NOT decoded frames. Never use this as actual serving selection.
    with sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        records = conn.execute("SELECT uid,series,COUNT(*) FROM files WHERE split='train' AND path LIKE '%.dcm' GROUP BY uid,series").fetchall()
        counts_by_key = {(u, s): n for u, s, n in records}
        selected_uids = sorted(train, key=lambda u: hashlib.sha256(("HC15_CACHE_PILOT_V1|" + u).encode()).hexdigest())[:2]
        sample = []
        for uid in selected_uids:
            # Full series in source CSV order, at most12; no images or labels inspected.
            for row in by_study[uid]:
                members = conn.execute("SELECT path,info FROM files WHERE split='train' AND uid=? AND series=? ORDER BY path",
                                       (uid, row["SeriesInstanceUID"])).fetchall()
                members = [{"path": p, "zip_info": json.loads(i)} for p, i in members if p.endswith(".dcm")]
                sample.append({"study": uid, "series": row["SeriesInstanceUID"], "csv_metadata": row, "members": members})
        sample = sample[:12]
    missing = [r["SeriesInstanceUID"] for r in series if counts_by_key.get((r[UID], r["SeriesInstanceUID"]), 0) == 0]
    require(not missing, f"Missing archive series: {len(missing)}")
    slot_counts = Counter()
    for uid in cohort:
        slots = {PLANES.index(r["Anatomical_Plane"]) + 3 * (1-int(r["Fat_Suppression"])) for r in by_study[uid]}
        slot_counts[len(slots)] += 1
    init = Path.home() / ".cache/huggingface/hub/models--timm--convnext_tiny.in12k_ft_in1k/snapshots/aa096f03029c7f0ec052013f64c819b34f8ad790/model.safetensors"
    require(init.is_file(), "Independent local ImageNet initialization absent")
    labels_contract = {"train_uids": train, "val_uids": val, "targets": TARGETS,
                       "train_soft_float32": [labels[u] for u in train], "val_soft_float32": [labels[u] for u in val],
                       "val_binary": reference, "target_mask": "all twelve present per study; no adaptive masking"}
    capacity = len(series) * 64 * 384 * 384
    free = shutil.disk_usage(ROOT).free
    versions = {}
    for name in ("torch", "timm", "numpy", "pandas", "pydicom", "safetensors", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    out.mkdir(parents=True, exist_ok=False)
    save(out / "cohort.json", labels_contract)
    save(out / "sample.json", {"selection": "first two T UIDs by SHA256(HC15_CACHE_PILOT_V1|uid), at most12 source-CSV-order series",
                              "series": sample, "decoded_frames_limit": 768,
                              "note": "No decode yet. Stop before exceeding768frames; no automatic sample replacement."})
    with (out / "series.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(series[0]))
        writer.writeheader(); writer.writerows(series)
    manifest = {
        "schema": "hc15-open-recipe-preparation-v1", "created_kst": datetime.now().astimezone().isoformat(),
        "status": "INPUTS_FROZEN_NOT_EXECUTION_READY", "source_plan": "H-C-15", "gpu_dispatch_allowed": False,
        "input_bindings": [binding(p) for p in (base_path, label_path, ids_path, groups_path, gold_path, series_path,
                              control_path, ROOT / "artifacts/experiments/EXP084_B_G1/provenance.json", Path(__file__))],
        "cohort": {"train": 3455, "validation": 888, "train_components": 3323, "validation_components": 873,
                   "train_uid_sha256": uid_sha(train), "validation_uid_sha256": uid_sha(val), "positive_counts": counts,
                   "gold_excluded": 58, "patient_independence": "not established; retain known component exclusions"},
        "initialization": {"architecture": "convnext_tiny.in12k_ft_in1k", "asset": binding(init),
                           "adaptation": "Local generic ImageNet checkpoint selected prospectively; not a claim of author historical initialization identity. Never use competition checkpoint for independent validation."},
        "environment": {"python": sys.version, "executable": sys.executable, "packages": versions},
        "archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size, "index": str(index_path),
                    "library_metadata_sha256": hashlib.sha256(raw.encode()).hexdigest(), "original_directory_sha256": archive.get("directory_sha256")},
        "cache": {"series": len(series), "all_public_train_series": len(all_series), "upper_bound_bytes": capacity,
                  "free_bytes_at_preparation": free, "capacity_gate_20percent_headroom": 1.2*capacity < free,
                  "csv_slot_count_distribution": dict(slot_counts), "decoded_coverage": "UNMEASURED",
                  "selection": "After decoding, retain at most64frames; keep retained n>=3; evaluation ranks by retained n (capped64), ties in source CSV order; training uniform random available series per slot. decoded_count is provenance only, not the evaluation ranking key.",
                  "not_reusable": "Raptor existing pixel shards use a different five-slot geometry/intensity contract; require new cache.",
                  "sample_series": len(sample), "sample_dicom_files": sum(len(s['members']) for s in sample)},
        "recipe": {"cache_size": 384, "mm_per_pixel": 0.4, "max_slices": 64, "model_res": 256,
                   "train_windows_per_slot": 8, "eval_windows_per_slot": 12, "slots": 6, "transformer_depth": 2,
                   "epochs": 14, "effective_batch": 8, "microbatch": "UNRESOLVED_BEFORE_PILOT", "successful_updates": 6034,
                   "loss": "plain soft BCE; no positive weights", "optimizer": "AdamW", "encoder_lr": 1e-4, "head_lr": 5e-4,
                   "weight_decay": 0.02, "ema": 0.998, "precision": "BF16 native local only; must be tested in later GPU pilot",
                   "scheduler": "1epoch linear warmup then cosine, optimizer-update steps", "clip_grad": 3,
                   "checkpoint_selection": "fixed final EMA; final weak validation only; no Gold selection",
                   "scope": "whole visual+optimization package and pinned initialization; adapted Dread+G1 vs public v0"},
        "evaluation": {"primary": "candidate final-EMA raw macro12 minus EXP084_B_G1 on identical V888 Dread float32->rint",
                       "candidate_k": "12 per present six-slot series", "control_k": "historical24 windows total",
                       "same_input_policy": False, "interpretation": "whole recipe contrast; not isolated architecture/label/precision effect",
                       "uncertainty": "paired component bootstrap500 seed20261003; exposed weak development set, no private/clinical guarantee",
                       "decision_rule": "UNREGISTERED_UNTIL_READINESS_AND_COST_KNOWN; do not fit"},
        "remaining": ["portable reader/model numerical contracts", "actual decoded-count coverage/cache timing",
                      "standalone training and serving runner with exact source/init/runtime binding",
                      "checkpoint/optimizer/scheduler/EMA/RNG resume proof", "independent readiness review",
                      "whole-cache allocation/time contract", "reviewed queue outer wrapper, GPU timing/VRAM gate and primary decision rule"],
        "generated": [binding(out / name) for name in ("cohort.json", "sample.json", "series.csv")],
    }
    save(out / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "series": len(series), "upper_cache_GiB": round(capacity/2**30,3),
                      "free_GiB": round(free/2**30,3), "sample_series": len(sample),
                      "sample_dicom_files": manifest["cache"]["sample_dicom_files"], "init_sha256": manifest["initialization"]["asset"]["sha256"]}))

if __name__ == "__main__":
    main()
