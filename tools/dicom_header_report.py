"""Validate fixed100 CPU header-I/O evidence and describe paired timing/order effects."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from dicom_header_benchmark import operational_receipt_equal


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing_group(rows, indices):
    values = [rows[i] for i in indices]
    a = np.asarray([r["baseline"]["seconds"] for r in values], float)
    b = np.asarray([r["candidate"]["seconds"] for r in values], float)
    return {"n": len(values), "baseline_seconds": float(a.sum()), "candidate_seconds": float(b.sum()),
            "sum_time_ratio": float(b.sum() / a.sum()), "geometric_pair_ratio": float(np.exp(np.log(b / a).mean())),
            "median_pair_ratio": float(np.median(b / a)), "candidate_faster": int((b < a).sum()),
            "median_baseline_seconds": float(np.median(a)), "median_candidate_seconds": float(np.median(b)),
            "p95_baseline_seconds": float(np.percentile(a, 95)), "p95_candidate_seconds": float(np.percentile(b, 95))}


def analyze(result_path, package_path, selection_path, order_path, prior_path):
    result, package, selection, order, prior = map(read, (result_path, package_path, selection_path, order_path, prior_path))
    ids = selection["ids"]
    proofs = {v["uid"]: v for v in order["studies"]}
    references = {v["uid"]: v for v in prior["study_receipts"]}
    rows = result.get("studies", [])
    expected = {"volume_exact", "mask_exact", "receipt_exact", "remote_volume_hash_exact", "remote_mask_hash_exact", "remote_operational_receipt_exact"}
    checks = {"status_pass": result.get("status") == "PASS", "comparison_v2": result.get("version") == "dicom-header-benchmark-v2",
              "fixed100_ids_order": result.get("ids") == ids and [r["uid"] for r in rows] == ids and len(set(ids)) == len(ids) == 100,
              "every_gate_present_true": all(set(r["checks"]) == expected and all(r["checks"].values()) for r in rows),
              "all_operational_proofs_recomputed": all(operational_receipt_equal(references[r["uid"]], r["receipt"], proofs[r["uid"]], r["ordering_proof"]) for r in rows),
              "baseline_source_sha": result["source_sha256"]["baseline"] == package["baseline_adapter_sha256"],
              "candidate_source_sha": result["source_sha256"]["candidate"] == package["candidate_adapter_sha256"],
              "benchmark_source_sha": result["source_sha256"]["tool"] == package["payloads"]["dicom_header_benchmark.py"]["sha256"],
              "selection_sha": result["source_sha256"]["selection"] == package["payloads"]["selection.json"]["sha256"],
              "series_sha": result["source_sha256"]["series"] == selection["train_series_sha256"],
              "reference_receipt_sha": result["source_sha256"]["reference_receipt"] == package["payloads"]["reference.json"]["sha256"],
              "ordering_reference_sha": result["source_sha256"]["ordering_reference"] == sha(order_path) == package["payloads"]["ordering-reference.json"]["sha256"],
              "positive_finite_timings": all(np.isfinite(r[k]["seconds"]) and r[k]["seconds"] > 0 for r in rows for k in ("baseline", "candidate"))}
    report = {"schema": 1, "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "PASS" if all(checks.values()) else "FAILED",
              "checks": checks, "input_files_sha256": {str(p): sha(p) for p in (result_path, package_path, selection_path, order_path, prior_path)},
              "scope": "fixed engineering development cohort; CPU input rendering only, not model accuracy, T4 inference or full candidate runtime"}
    if not all(checks.values()):
        return report
    report["overall"] = timing_group(rows, range(100))
    report["baseline_first"] = timing_group(rows, range(0, 100, 2))
    report["candidate_first"] = timing_group(rows, range(1, 100, 2))
    report["nonexact_derived_position_receipts"] = sum(not r["observations"]["remote_receipt_exact"] for r in rows)
    report["derived_position_hash_differences"] = sum(len(r["observations"]["derived_position_hash_differences"]) for r in rows)
    before = sum(r["baseline"]["total_logical_bytes_read"] for r in rows)
    after = sum(r["candidate"]["total_logical_bytes_read"] for r in rows)
    report["logical_io"] = {"baseline_bytes": before, "candidate_bytes": after, "reduction_fraction": 1 - after / before,
        "candidate_header_files": sum(r["candidate"]["header"]["files_completed"] for r in rows),
        "candidate_header_bytes": sum(r["candidate"]["header"]["logical_bytes_read"] for r in rows),
        "scope": "render read API bytes; excludes additional ordering-proof read, OS caching/read-ahead and physical I/O"}
    logs = np.log([r["candidate"]["seconds"] / r["baseline"]["seconds"] for r in rows])
    first, second = logs[0::2], logs[1::2]
    rng = np.random.default_rng(20261003)
    bootstrap = (rng.choice(first, (10000, len(first))).mean(1) + rng.choice(second, (10000, len(second))).mean(1)) / 2
    report["order_balanced_descriptive_ratio"] = {"candidate_over_baseline": float(np.exp((first.mean() + second.mean()) / 2)),
        "stratified_pair_bootstrap_percentile95": np.exp(np.percentile(bootstrap, [2.5, 97.5])).tolist(),
        "candidate_first_over_second_ratio_factor": float(np.exp(second.mean() - first.mean())),
        "resamples": 10000, "seed": 20261003,
        "limitation": "single timing per implementation/study; deterministic alternating order, no repeated crossover or randomized machine replication; CI describes sampled pairs and does not include hardware/cache drift"}
    report["limitations"] = ["second pass may benefit from first pass cache; fixed cohort is not a random hidden-test sample",
                             "CPU and T4 hosts can have different CPU/RAM/storage characteristics",
                             "input identity pass provides no AUC improvement evidence",
                             "T4 five-model benchmark and joint anchor budget remain necessary before shipping"]
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "package", "selection", "order", "prior", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError("choose a new report path")
    report = analyze(a.result, a.package, a.selection, a.order, a.prior)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
