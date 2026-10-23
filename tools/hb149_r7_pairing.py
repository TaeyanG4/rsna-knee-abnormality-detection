"""H-B-149 v1.2.4: immutable paired cohort and aggregate-only L2 readout.

Changelog v1.2.4 (2026-10-14): the baseline may be a COMPLETED v1.1.3 run whose
ONLY failures are exactly the registered D-B-177 BadZipFile studies
(tools/hb145_data_exclusions_v1.py), provided none is in the frozen selection;
every other failure still rejects. Readout may evaluate the labelled
intersection when given a label-availability exclusion record fixed before any
flip result; GO rules, cohort and science unchanged. v1.2.3 backed up.
Changelog v1.2.3: validation/inference/readout require authenticated v113
carry-over unconditionally; historical v112 selection registration stays frozen.
Changelog v1.2.2: verified v113 carry-over bridge; frozen selection unchanged;
new receipts/readout bind the actual v113 identity.
Changelog 2026-10-13: seeded 1,000-study cohort, strict pairing/provenance,
paired study-bootstrap AUC and Spearman, explicit T4 runtime scenarios.
Changelog v1.2.1: enforce target floor; register f0+f2 flip policy and
1300-study, 7.5-hour gate. Selection and inference remain version 1.2.0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb145_r7_heldout_v1 as base
from tools import hb145_baseline_contract_v113 as contract
from tools import hb145_data_exclusions_v1 as data_exclusions

VERSION = "1.2.4"
LABEL_EXCLUSION_SCHEMA = "hb149-label-availability-exclusions/1"
LABEL_EXCLUSION_VERSION = "1.2.4"
ARTIFACT_VERSION = "1.2.0"
VARIANTS = {"full_r7_flip": (0, 1, 2, 3), "r7_flip_f0_f2": (0, 2)}
TARGET_FLOOR = -.005
HIDDEN_STUDIES = 1300
MAX_HOURS = 7.5
RELEASE_BASE = "C85-R7 v1.2.2 insurance base (deadline publication)"
SCRATCH = ROOT / ".local/scratch/astra-b5/H-B-149"
BASELINE = ROOT / ".local/scratch/astra-b5/H-B-145/full-v113-ready"
SELECTION = SCRATCH / "selection-v120.json"
SEED = 14920261013
SIZE = 1000
WEAK = ("Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def owned_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(SCRATCH.resolve()) or path == SCRATCH.resolve():
        raise ValueError("output must be inside fixed H-B-149 scratch")
    return path


def cohort(identity):
    if (identity.get("version") not in ("1.1.2", "1.1.3") or identity.get("models") != "heldout,f0"
            or identity.get("device") != "cuda" or identity.get("clean_oof") is not False
            or identity.get("exposure") != "EXPOSED"):
        raise ValueError("expected frozen EXPOSED v1.1.2/v1.1.3 CUDA dual-role baseline")
    population = identity["uids"]
    if len(set(population)) != len(population):
        raise ValueError("duplicate baseline UID")
    eligible = [u for u in population if base.fold_k(u) in (1, 2, 3)]
    if len(eligible) < SIZE:
        raise ValueError("insufficient folds 1-3 population")
    ranked = sorted(eligible, key=lambda u: (hashlib.sha256(f"HB149|{SEED}|{u}".encode()).hexdigest(), u))
    return sorted(ranked[:SIZE])


def select(baseline=BASELINE, output=SELECTION):
    identity = read_json(Path(baseline) / "identity.json")
    uids = cohort(identity)
    value = {"version": ARTIFACT_VERSION, **base.EXPOSURE, "seed": SEED, "size": SIZE,
             "algorithm": "lowest sha256(HB149|seed|UID), then lexicographic UID order",
             "baseline_identity": base.fingerprint(identity), "uids": uids,
             "fold_counts": [sum(base.fold_k(u) == k for u in uids) for k in range(4)]}
    output = owned_output(output)
    if output.exists() and read_json(output) != value:
        raise ValueError("selection changed: use a new version, never silently resample")
    base.atomic_json(output, value)
    return {k: v for k, v in value.items() if k != "uids"}


def verify_selection(selection, baseline):
    record = read_json(selection)
    identity = read_json(Path(baseline) / "identity.json")
    actual = base.fingerprint(identity)
    lineage = contract.validate(baseline, identity)
    accepted = {actual, lineage["source_identity"]}
    if (record.get("version") != ARTIFACT_VERSION or record.get("seed") != SEED
            or record.get("size") != SIZE or record.get("uids") != cohort(identity)
            or record.get("baseline_identity") not in accepted):
        raise ValueError("selection/baseline identity mismatch")
    return record, identity


def predictions(path, uids, role):
    """Read numeric predictions internally; never echo UIDs or individual values."""
    required = set(uids)
    result, seen = {}, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = [base.UID, *base.LABELS, "model_role", "model_fold", "exposure", "clean_oof", "status"]
        if not set(needed) <= set(reader.fieldnames or []):
            raise ValueError("prediction schema mismatch")
        for row in reader:
            uid = row[base.UID]
            if uid in seen:
                raise ValueError("duplicate prediction UID")
            seen.add(uid)
            if uid not in required:
                continue
            if (row["model_role"] != role or int(row["model_fold"]) != base.fold_k(uid)
                    or row["exposure"] != "EXPOSED" or row["clean_oof"].lower() != "false"
                    or row["status"] not in ("ok", "dropped series")):
                raise ValueError("prediction provenance/fold/status mismatch")
            values = [float(row[t]) for t in base.LABELS]
            if any(not base.math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError("invalid prediction")
            result[uid] = values
    if set(result) != required or (role == "heldout_flip" and seen != required):
        raise ValueError("paired prediction coverage mismatch")
    return result


def baseline_status(selection=SELECTION, baseline=BASELINE):
    """v1.2.4: completion gate plus the registered data-exclusion path.

    Returns (selection record, identity, aggregate status). A nonzero failure
    count is accepted only when tools/hb145_data_exclusions_v1.verify_run
    proves the failures are exactly the registered BadZipFile studies AND none
    of them is in the frozen selection. No other relaxation exists.
    """
    selection_record, identity = verify_selection(selection, baseline)
    receipt_path = Path(baseline) / "run_receipt.json"
    if not receipt_path.exists():
        raise ValueError("H-B-145 full must complete successfully before paired inference")
    receipt = read_json(receipt_path)
    if (receipt.get("identity") != base.fingerprint(identity)
            or receipt.get("completed") != len(identity["uids"])
            or receipt.get("selected") != len(identity["uids"])
            or receipt.get("decode_failures") != 0
            or receipt.get("exit_code") != 0):
        raise ValueError("H-B-145 full must complete successfully before paired inference")
    uids = selection_record["uids"]
    exclusion = None
    if receipt.get("failures") != 0:
        try:
            exclusion = data_exclusions.verify_run(baseline, identity, receipt)
        except (ValueError, KeyError, TypeError, OSError):
            raise ValueError("H-B-145 full must complete successfully before paired inference: "
                             "failures are not exactly the registered data exclusions") from None
        if set(exclusion["uids"]) & set(uids):
            raise ValueError("registered data exclusion intersects the frozen selection; version required")
    # Authenticate every selected CSV value against its committed baseline journal.
    rows = predictions(Path(baseline) / "heldout_raw.csv", uids, "heldout")
    records = base.load_done(baseline, uids, base.fingerprint(identity))
    if set(records) != set(uids):
        raise ValueError("missing baseline paired journals")
    for uid in uids:
        rec = records[uid]
        if rec["status"] == "failure" or rows[uid] != rec["predictions"][str(base.fold_k(uid))]["raw"]:
            raise ValueError("baseline CSV differs from committed journal")
    status = {"baseline_ready": True, "paired_studies": len(uids),
              "producer_failures": receipt.get("failures"),
              "data_exclusions": None if exclusion is None else {
                  "accepted": True, "count": exclusion["count"], "error": exclusion["error"],
                  "reason": exclusion["reason"], "sha256": exclusion["exclusions_sha256"],
                  "in_frozen_selection": 0}}
    return selection_record, identity, status


def baseline_ready(selection=SELECTION, baseline=BASELINE):
    selection_record, identity, _ = baseline_status(selection, baseline)
    return selection_record, identity


def label_exclusions(path, selection, uids):
    """Label-availability exclusion registered before any flip result.

    Only selected studies without a registered L2 row may be listed; the
    readout then requires labels for exactly the remaining studies.
    """
    record = read_json(path)
    excluded = record.get("uids")
    if (record.get("schema") != LABEL_EXCLUSION_SCHEMA or record.get("version") != LABEL_EXCLUSION_VERSION
            or record.get("decided_before_flip_result") is not True
            or record.get("gate_policy_unchanged") is not True
            or record.get("selection_sha256") != base.r7.s6.sha256(selection)
            or not isinstance(excluded, list) or len(excluded) != len(set(excluded))
            or record.get("count") != len(excluded) or record.get("selected_studies") != len(uids)
            or record.get("labelled_studies") != len(uids) - len(excluded)
            or not set(excluded) < set(uids)):
        raise ValueError("label-availability exclusion record differs")
    return set(excluded), {"sha256": base.r7.s6.sha256(path), "count": len(excluded),
                           "reason": record.get("reason"), "source_sha256": record.get("source_sha256"),
                           "decided_before_flip_result": True}


def selected_uids(selection, baseline, members, require_complete=False, cpu_smoke=False):
    record, identity = (baseline_ready(selection, baseline) if require_complete and not cpu_smoke
                        else verify_selection(selection, baseline))
    if set(members) != set(identity["uids"]):
        raise ValueError("DICOM population differs from frozen baseline")
    return record["uids"]


def runtime_estimate(studies=HIDDEN_STUDIES, variant="full_r7_flip"):
    if not isinstance(studies, int) or isinstance(studies, bool) or studies <= 0:
        raise ValueError("studies must be a positive integer")
    # N32-N3 receipt slope; scaling applies ONLY to incremental R7 work.
    slope = (680.704481005 - 344.507129447) / (32 - 3)
    fraction = len(VARIANTS[variant]) / 4
    added = studies * slope * fraction / 3600
    return {"version": VERSION, "variant": variant, "flip_folds": list(VARIANTS[variant]),
            "studies": studies, "baseline_hidden_hours_approx": 4.7,
            "r7_n32_seconds_per_study": slope, "additional_r7_fraction": fraction,
            "formula_hours": "4.7 + N * ((680.704481005 - 344.507129447) / 29) * (flip_fold_count / 4) * c / 3600",
            "calibration_factors": [.75, .84],
            "total_hours_calibrated_range": [4.7 + added * c for c in (.75, .84)],
            "extra_hours_unscaled": added, "total_hours_unscaled": 4.7 + added,
            "headroom_hours_unscaled": 9 - (4.7 + added),
            "go_projection_limit_hours": MAX_HOURS,
            "within_go_budget": 4.7 + added <= MAX_HOURS,
            "go_projection_rule": "c=1 (unscaled); conditional 0.75-0.84 speedup cannot rescue GO",
            "release_base": RELEASE_BASE,
            "qualification": "planning estimate, not measured bound; hidden size approximate; equal fold costs assumed; 0.75-0.84 applies to incremental N32 R7 projection only, transfer to flip unproven; exact candidate T4 timing required"}


def metrics(labels, plain, flip, draws=2000, seed=SEED):
    """Paired resampling of entire studies. L2 soft targets thresholded at >=0.5.

    Blank/NaN labels masked identically in both arms. Reject non-[0,1] labels.
    A macro bootstrap draw is usable only if all its 12 AUCs are defined.
    """
    import numpy as np
    from scipy.stats import rankdata
    labels, plain, flip = (np.asarray(x, dtype=float) for x in (labels, plain, flip))
    if labels.shape != plain.shape or plain.shape != flip.shape or labels.ndim != 2 or labels.shape[1] != 12:
        raise ValueError("paired matrix shape mismatch")
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws required")
    if (np.isinf(labels).any() or ((labels[np.isfinite(labels)] < 0) | (labels[np.isfinite(labels)] > 1)).any()
            or not np.isfinite(plain).all() or not np.isfinite(flip).all()
            or ((plain < 0) | (plain > 1) | (flip < 0) | (flip > 1)).any()):
        raise ValueError("invalid label/prediction range")
    def auc(y, p):
        ok = np.isfinite(y)
        y, p = y[ok] >= .5, p[ok]
        positive = int(y.sum())
        negative = len(y) - positive
        if not positive or not negative:
            return np.nan
        return (rankdata(p)[y].sum() - positive * (positive + 1) / 2) / (positive * negative)
    def rho(a, b):
        a, b = rankdata(a), rankdata(b)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    def evaluate(indices):
        return np.array([[auc(labels[indices, j], plain[indices, j]),
                          auc(labels[indices, j], flip[indices, j]),
                          rho(plain[indices, j], flip[indices, j])] for j in range(12)])
    point = evaluate(np.arange(len(labels)))
    rng = np.random.default_rng(seed)
    boots = np.stack([evaluate(rng.integers(0, len(labels), len(labels))) for _ in range(draws)])
    def value(x):
        return float(x) if np.isfinite(x) else None
    def summary(p, samples):
        samples = samples[np.isfinite(samples)]
        return {"value": value(p), "ci95": np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
                "valid_bootstraps": len(samples)}
    def group(indices):
        p, b = point[indices].mean(axis=0), boots[:, indices].mean(axis=1)
        return {"no_flip_auc": summary(p[0], b[:, 0]), "flip_auc": summary(p[1], b[:, 1]),
                "delta_auc": summary(p[1] - p[0], b[:, 1] - b[:, 0]),
                "spearman_mean": summary(p[2], b[:, 2])}
    targets = []
    for j, name in enumerate(base.LABELS):
        valid = np.isfinite(labels[:, j])
        targets.append({"target": name, "labelled": int(valid.sum()),
                        "positive": int((labels[valid, j] >= .5).sum()), **group([j])})
    result = {"version": VERSION, **base.EXPOSURE, "studies": len(labels), "draws": draws, "seed": seed,
              "label_rule": "L2 >= 0.5 positive; missing masked identically; no label imputation",
              "ci_rule": "paired study bootstrap percentile 95%; fixed target sets, undefined draws counted",
              "macro": group(list(range(12))), "weak_four": group([base.LABELS.index(n) for n in WEAK]),
              "weak_four_targets": list(WEAK), "per_target": targets}
    delta = result["macro"]["delta_auc"]
    target_deltas = [row["delta_auc"]["value"] for row in targets]
    floor_pass = all(x is not None and x >= TARGET_FLOOR for x in target_deltas)
    result["target_floor"] = {"minimum_delta_auc": TARGET_FLOOR, "passed": floor_pass,
                              "worst_delta_auc": min(target_deltas) if all(x is not None for x in target_deltas) else None}
    result["screen_gate"] = "ADVANCE_TO_BLEND_AND_T4_MEASUREMENT" if (delta["value"] is not None
        and delta["value"] >= .001 and delta["ci95"] and delta["ci95"][0] > 0
        and delta["valid_bootstraps"] >= .95 * draws and floor_pass) else "HOLD"
    return result


def variant_readout(labels, plain, flip, folds, draws=2000, studies=HIDDEN_STUDIES):
    """Apply fixed model-fold policy to paired heldout predictions, no retuning.

    This is a heldout-policy screen, NOT a four-model serving ensemble AUC.
    The registered cohort excludes f0, so f0's flip effect is not measured.
    """
    import numpy as np
    plain, flip, folds = np.asarray(plain), np.asarray(flip), np.asarray(folds)
    if plain.ndim != 2 or plain.shape != flip.shape or folds.shape != (len(plain),) or not np.isin(folds, (0, 1, 2, 3)).all():
        raise ValueError("paired fold policy shape/value mismatch")
    variants = {}
    for name, selected in VARIANTS.items():
        mask = np.isin(folds, selected)
        candidate = np.where(mask[:, None], flip, plain)
        stats = metrics(labels, plain, candidate, draws)
        timing = runtime_estimate(studies, name)
        stats.update(flip_folds=list(selected), studies_using_flip=int(mask.sum()),
                     runtime=timing, go_gate="GO" if stats["screen_gate"] != "HOLD" and timing["within_go_budget"] else "HOLD")
        variants[name] = stats
    passed = [name for name in VARIANTS if variants[name]["go_gate"] == "GO"]
    return {"variants": variants, "go_variants": passed,
            "go_gate": "GO: " + ", ".join(passed) if passed else "HOLD: no variant passes",
            "release_base": RELEASE_BASE, "release_authorized": False,
            "evaluation_scope": "EXPOSED heldout model-policy screen, not four-model ensemble or C85 blend AUC; f0 absent in registered cohort; real blend validation and exact T4 timing still required",
            "registration": "fixed f0+f2, chosen before real readout; f1+f3 plain; no fold selection or ratio tuning from results"}


def readout(selection, baseline, flip_dir, labels_path, output, draws=2000, studies=HIDDEN_STUDIES,
            label_exclusion=None):
    import numpy as np
    record, baseline_identity, baseline_state = baseline_status(selection, baseline)
    flip_dir = Path(flip_dir)
    identity = read_json(flip_dir / "identity.json")
    receipt = read_json(flip_dir / "run_receipt.json")
    uids = record["uids"]
    excluded, exclusion = set(), None
    if label_exclusion is not None:
        excluded, exclusion = label_exclusions(label_exclusion, selection, uids)
    evaluated = [u for u in uids if u not in excluded]
    if (identity.get("version") != ARTIFACT_VERSION or identity.get("models") != "heldout_flip"
            or identity.get("cpu_smoke") is not False or identity.get("device") != "cuda"
            or identity.get("clean_oof") is not False or identity.get("exposure") != "EXPOSED"
            or identity.get("uids") != uids or identity.get("flip_tta") is not True
            or identity.get("baseline_identity") != base.fingerprint(baseline_identity)
            or identity.get("selection_sha256") != base.r7.s6.sha256(selection)
            or receipt.get("identity") != base.fingerprint(identity)
            or receipt.get("completed") != SIZE or receipt.get("selected") != SIZE or receipt.get("failures") != 0
            or receipt.get("exit_code") != 0 or receipt.get("decode_failures") != 0):
        raise ValueError("flip completion/identity mismatch")
    plain = predictions(Path(baseline) / "heldout_raw.csv", uids, "heldout")
    flip = predictions(flip_dir / "heldout_flip_raw.csv", uids, "heldout_flip")
    from tools import hb149_r7_flip_v120 as runner
    journals = runner.load_done(flip_dir, uids, base.fingerprint(identity))
    for uid in uids:
        if uid not in journals or flip[uid] != journals[uid]["predictions"][str(base.fold_k(uid))]["raw"]:
            raise ValueError("flip CSV differs from committed journal")
    labels_path = Path(labels_path).resolve()
    if ".local/relabel" in labels_path.as_posix().lower():
        raise ValueError("restricted label location; use an explicitly supplied numeric L2 export")
    labels = {}
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [base.UID, *base.LABELS]:
            raise ValueError("L2 export must contain UID and exactly 12 numeric target columns, no text")
        seen = set()
        for row in reader:
            uid = row[base.UID]
            if uid in seen:
                raise ValueError("duplicate label UID")
            seen.add(uid)
            if uid in excluded:
                raise ValueError("label-availability exclusion is stale: an excluded study has an L2 row")
            if uid in plain:
                try:
                    labels[uid] = [float(row[t]) if row[t].strip() else np.nan for t in base.LABELS]
                except (ValueError, TypeError, AttributeError):
                    raise ValueError("L2 export contains a malformed numeric label") from None
    if set(labels) != set(evaluated):
        raise ValueError("missing paired L2 study labels")
    # Labelled intersection only; the excluded studies stay authenticated above.
    comparison = variant_readout([labels[u] for u in evaluated], [plain[u] for u in evaluated],
                                 [flip[u] for u in evaluated], [base.fold_k(u) for u in evaluated], draws, studies)
    result = dict(comparison["variants"]["full_r7_flip"])
    result.update(comparison)
    result.update(selected_studies=len(uids), evaluated_studies=len(evaluated),
                  label_availability_exclusion=exclusion,
                  baseline_data_exclusions=baseline_state["data_exclusions"],
                  baseline_identity=base.fingerprint(baseline_identity),
                  selection_baseline_identity=record["baseline_identity"],
                  carry_over=contract.validate(baseline, baseline_identity),
                  flip_identity=base.fingerprint(identity),
                  selection_sha256=base.r7.s6.sha256(selection), labels_sha256=base.r7.s6.sha256(labels_path))
    output = owned_output(output)
    base.atomic_json(output, result)
    table = output.with_suffix(".csv")
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "target", "labelled", "positive", "no_flip_auc", "flip_auc", "delta_auc", "delta_ci_low", "delta_ci_high", "spearman"])
        for name, stats in result["variants"].items():
          for row in stats["per_target"]:
            writer.writerow([name, row["target"], row["labelled"], row["positive"], row["no_flip_auc"]["value"],
                             row["flip_auc"]["value"], row["delta_auc"]["value"],
                             *(row["delta_auc"]["ci95"] or [None, None]), row["spearman_mean"]["value"]])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "validate", "readout", "runtime"))
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--flip-dir", type=Path, default=SCRATCH / "full-v122")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--label-exclusion", type=Path,
                        help="registered label-availability exclusion record (labelled intersection)")
    parser.add_argument("--output", type=Path, default=SCRATCH / "paired-readout-v122.json")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--studies", type=int, default=HIDDEN_STUDIES)
    a = parser.parse_args(argv)
    if a.command == "select":
        result = select(a.baseline, a.selection)
    elif a.command == "validate":
        _, _, result = baseline_status(a.selection, a.baseline)
        result = {"version": VERSION, **result}
    elif a.command == "runtime":
        if a.studies <= 0:
            parser.error("--studies must be positive")
        result = {"version": VERSION, "variants": {name: runtime_estimate(a.studies, name) for name in VARIANTS}}
        base.atomic_json(owned_output(a.output), result)
    else:
        if not a.labels:
            parser.error("readout requires an explicit numeric --labels L2 export")
        result = readout(a.selection, a.baseline, a.flip_dir, a.labels, a.output, a.draws, a.studies,
                         a.label_exclusion)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
