"""EXP169 v5q visible-qualification comparator (local only; never embedded in a kernel).

Changelog
- v1.1.0 (astra, 2026-10-11): all timing is informational; no deadline contract,
  guard or projection rejection. v5q delegates to this single local comparator.
- v1.0.0 (Claude opus for B, 2026-10-11): FIX-Q1 of the v5q 1.1.0 verification
  (.local/claude-builds/v5q-verify-report.md). Runs every blocking check of
  ``exp169_release_v5q.compare_visible`` (registered tolerance and source bindings, A172
  receipts, member/partner/anchor parity, IDs and order, output hashes, S6 provenance with the
  source code-tree pin, sibling S6 parity receipt, exact trio CSV and final-CSV formula), but
  the S6 timing projection from the 3 visible studies is informational: it is recorded with a
  ``within_guard`` flag and never prevents or changes the parity result. Receipt-integrity
  timing fields stay blocking (the four service terms incl. ``anchor_s``, finite projections,
  deadline = start + 6 h 45 m). Also passes the S6 port explicitly, so A5 works when run as a
  script (the frozen comparator imports ``tools.exp189_s6_port``, which is not importable when
  ``tools/exp169_release_v5q.py`` itself is the script). The frozen v5q module is unchanged
  because its bytes are embedded in the staged candidate kernels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import exp169_release_v5q as v5q  # noqa: E402
from tools import exp189_s6_port as s6  # noqa: E402

VERSION = "1.1.0"
SERVICE_TERMS = {"decode_s", "render_s", "anchor_s", "gpu_s"}
require, sha, validate_ids, _read_csv = v5q.require, v5q.sha, v5q.validate_ids, v5q._read_csv


def timing_summary(v5_receipt: dict) -> dict:
    """Blocking only for receipt integrity; the projection itself is informational."""
    timing = v5_receipt["timing_projection"]
    conservative = float(timing["projected_conservative_seconds"])
    observed = float(timing["projected_from_observed_seconds"])
    require(set(timing["service_per_study"]) == SERVICE_TERMS and
            math.isfinite(conservative) and math.isfinite(observed) and
            all(math.isfinite(float(value)) and float(value) >= 0
                for value in timing["service_per_study"].values()),
            "v5q timing receipt incomplete")
    return {"informational": True,
            "basis": "3 visible studies; per-loader-worker setup (imports, anatomy and side models) "
                     "and CUDA warm-up are booked into decode_s/gpu_s, so this over-states hidden per-study "
                     "time; take timing evidence from a many-study run",
            "projected_conservative_seconds": conservative,
            "projected_from_observed_seconds": observed,
            "s6_seconds_per_study": timing["s6_seconds_per_study"],
            "service_per_study": timing["service_per_study"]}


def compare_visible(candidate: str, output: Path, sibling: Path, result: Path) -> dict:
    import numpy as np
    output, sibling, result = Path(output), Path(sibling), Path(result)
    require(candidate in v5q.CANDIDATES, "unknown candidate")
    require(not result.exists(), "fresh parity result required")
    plan = json.loads((v5q.OUT / "visible-tolerance.json").read_text())
    require(plan["schema"] == v5q.TOLERANCE_SCHEMA and
            plan["s6_probability_atol"] == v5q.S6_ATOL and plan["s6_probability_rtol"] == 0 and
            plan["s6_bad_cells"] == 0 and
            plan["a172_anchor_max_abs"] == v5q.ANCHOR_MAX_ATOL and
            plan["a172_anchor_mean_abs"] == v5q.ANCHOR_MEAN_ATOL and
            plan["a172_partner_max_abs"] == v5q.PARTNER_ATOL and
            plan["timing_policy"] == "informational; no self-imposed limits",
            "registered tolerance changed")
    require(plan["stage_notebook_sha256"] == sha(v5q.STAGE / "kernel/release.ipynb") and
            plan["stage_archive_sha256"] == sha(v5q.STAGE / "dataset/exp169-assets.zip.bin") and
            plan["port_source_sha256"] == sha(v5q.ROOT / "tools/exp189_s6_port.py") == v5q.S6_PORT_SHA,
            "source binding changed")
    ref = v5q.STAGE / "kernel-output"
    baseline_receipt = json.loads((ref / "exp169-executed-candidate.json").read_text())
    run_receipt = json.loads((output / "exp169-executed-candidate.json").read_text())
    v5_receipt = json.loads((output / "v5q-receipt.json").read_text())
    require(baseline_receipt["status"] == run_receipt["status"] == "PASS_BUNDLE" and
            baseline_receipt["candidate"] == run_receipt["candidate"] == "A172" and
            v5_receipt["status"] == v5q.PASS_STATUS and v5_receipt["candidate"] == candidate and
            v5_receipt["generator_version"] == v5q.VERSION and
            v5_receipt["source_sha256"] == v5q.S6_PORT_SHA and
            v5_receipt["a172_source_sha256"] == v5q.A172_SOURCE_SHA, "execution receipt mismatch")
    timing = timing_summary(v5_receipt)
    require(sha(output / "a172-submission.csv") == run_receipt["submission_sha256"] ==
            v5_receipt["a172_submission_sha256"], "A172 copied submission differs")
    diffs = {}
    for part, keys in (("exp169-trio/raw.npz", ("own", "public", "exp143")),
                       ("partner-raw.npz", ("l2", "dino", "convnext"))):
        with np.load(ref / part, allow_pickle=False) as old, np.load(output / part, allow_pickle=False) as new:
            require(set(old.files) == set(new.files) == {"ids", "targets", *keys} if part.startswith("exp169-trio")
                    else set(old.files) == set(new.files) == {"ids", *keys}, "member schema differs")
            validate_ids(old["ids"].tolist(), new["ids"].tolist())
            if "targets" in old:
                require(old["targets"].tolist() == new["targets"].tolist() == list(v5q.LABELS),
                        "target order differs")
            for key in keys:
                a, b = old[key].astype(np.float64), new[key].astype(np.float64)
                require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
                        "member shape/finite differs")
                delta = np.abs(a - b)
                diffs[key] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}
    require(json.loads((ref / "exp169-trio/inputs.json").read_text()) ==
            json.loads((output / "exp169-trio/inputs.json").read_text()), "A172 inputs differ")
    require(json.loads((ref / "exp169-trio/binding.json").read_text()) ==
            json.loads((output / "exp169-trio/binding.json").read_text()), "A172 binding differs")
    old_ids, old_csv = _read_csv(ref / "exp169-trio/submission.csv")
    new_ids, new_csv = _read_csv(output / "exp169-trio/submission.csv")
    validate_ids(old_ids, new_ids)
    anchor_delta = np.abs(old_csv - new_csv)
    s6_ids, s6_values = _read_csv(output / "s6-port-predictions.csv")
    public_ids, public_values = _read_csv(sibling / "ours_sub.csv")
    sibling_ids, sibling_values = _read_csv(sibling / "exp189_port_predictions.csv")
    validate_ids(old_ids, s6_ids, public_ids, sibling_ids)
    sibling_receipt = json.loads((sibling / "exp189_parity_receipt.json").read_text())
    require(sibling_receipt["bad_cells"] == 0 and sibling_receipt["rows"] == len(old_ids) and
            len(sibling_receipt["gpu_names"]) == 2 and
            all("T4" in name for name in sibling_receipt["gpu_names"]),
            "public S6 parity receipt failed")
    require(sha(output / "s6-port-predictions.csv") == v5_receipt["s6_raw_sha256"] and
            sha(output / "submission.csv") == v5_receipt["submission_sha256"] and
            sha(output / "v5q-trio.csv") == v5_receipt["trio_csv_sha256"] and
            sha(output / "s6-port/provenance.json") == v5_receipt["s6_provenance_sha256"],
            "v5q output hashes differ")
    provenance = json.loads((output / "s6-port/provenance.json").read_text())
    require(provenance["dataset_slug"] == s6.DATASET_SLUG and
            provenance["dataset_version"] == "10-equivalent" and
            provenance["bundle_id"] == s6.BUNDLE_ID and
            provenance["licence"] == "other" and
            provenance["readme_permission_line"] == s6.README_PERMISSION and
            provenance["gold_in_training"] == 0 and
            provenance["manifest_sha256"] == s6.MANIFEST_SHA256 and
            provenance.get("code_tree_pinned_in_source") is True and
            provenance.get("code_tree_sha256") == s6.CODE_TREE_SHA256,
            "S6 provenance differs")
    a172 = v5q._a172_module()
    with np.load(output / "exp169-trio/raw.npz", allow_pickle=False) as anchor, \
         np.load(output / "partner-raw.npz", allow_pickle=False) as partner:
        q, _ = a172.aggregate({key: partner[key] for key in ("l2", "dino", "convnext")})
        members = (anchor["public"], s6_values, anchor["exp143"], anchor["own"])
        expected_trio = v5q.trio_values(candidate, old_ids, *members, s6_port=s6)
        expected = v5q.blend_arrays(candidate, old_ids, *members, q, a172=a172, s6_port=s6)
    trio_ids, saved_trio = _read_csv(output / "v5q-trio.csv")
    validate_ids(old_ids, trio_ids)
    require(np.array_equal(saved_trio, expected_trio), "v5q trio CSV differs from the registered members")
    final_ids, actual = _read_csv(output / "submission.csv")
    validate_ids(old_ids, final_ids)
    rounded = np.asarray([[float(format(float(v), ".10g")) for v in row] for row in expected])
    require(np.array_equal(actual, rounded), "v5q final CSV formula differs")
    public_delta = np.abs(s6_values - public_values)
    sibling_delta = np.abs(s6_values - sibling_values)
    passed = (all(diffs[key]["max_abs"] <= v5q.ANCHOR_MAX_ATOL and
                  diffs[key]["mean_abs"] <= v5q.ANCHOR_MEAN_ATOL for key in ("own", "public", "exp143"))
              and all(diffs[key]["max_abs"] <= v5q.PARTNER_ATOL for key in ("l2", "dino", "convnext"))
              and float(anchor_delta.max()) <= v5q.ANCHOR_MAX_ATOL
              and float(anchor_delta.mean()) <= v5q.ANCHOR_MEAN_ATOL
              and float(public_delta.max()) <= v5q.S6_ATOL
              and float(sibling_delta.max()) <= v5q.S6_ATOL)
    result_data = {"status": "PASS_VISIBLE_PARITY_UNQUALIFIED" if passed else "HOLD_PARITY",
                   "comparator_version": VERSION, "candidate": candidate, "studies": len(old_ids),
                   "member_diffs": diffs,
                   "a172_anchor_csv_max_abs": float(anchor_delta.max()),
                   "a172_anchor_csv_mean_abs": float(anchor_delta.mean()),
                   "s6_public_max_abs": float(public_delta.max()),
                   "s6_sibling_port_max_abs": float(sibling_delta.max()),
                   "s6_anatomy_anchors": v5_receipt.get("s6_anatomy_anchors"),
                   "timing_informational": timing,
                   "timing_projection": v5_receipt["timing_projection"],
                   "tolerance_sha256": sha(v5q.OUT / "visible-tolerance.json")}
    result.write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")
    require(passed, "visible parity failed; inspect result")
    return result_data


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP169 v5q visible parity (timing informational)")
    parser.add_argument("--candidate", choices=v5q.CANDIDATES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sibling-output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_visible(args.candidate, args.output, args.sibling_output, args.result), indent=2))


if __name__ == "__main__":
    main()
