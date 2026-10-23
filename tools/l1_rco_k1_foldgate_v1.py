"""v1.2.0: CPU own-holdout falsifier; aggregate output only, no model fitting.

Changelog v1.2.0: explicit fullbag fold0 vs seed0 screen with HOST_BOUND_EXPORT.
Changelog v1.1.1: K1-only hash-bound host export validation; exact local path retained.
Changelog v1.1.0: explicit partial-fold K1-only one-sided health gate.
Changelog: exact full-fold rosters, authenticated production prediction receipts,
four fixed rulers. Missing/nonfinite evidence is HOLD, never cohort shrinkage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from tools import l1_rco_k1_folds_v1 as k

OUT = ROOT / ".local/scratch/astra-b2/h224-device-plan-v1/own-holdout-gate.json"
RULERS = ("jev_hybrid", "aquila_v1", "dread", "consensus")
K1_OUT = OUT.with_name("k1-only-health.json")
HEALTH_POLICY = (
    "HEALTH_PASS iff at least one k in 1..3 is present, every present k and every "
    "fixed ruler satisfy K1_fk >= K1_f0 - 0.005, and no row or target AUC is "
    "invalid/undefined. Better is fine. Unavailable folds are PENDING; "
    "HEALTH_PASS covers present folds only, not completion of all folds."
)


def health_metadata():
    return dict(mode="k1-only", policy=HEALTH_POLICY, clean_oof=False,
                own_holdout_evaluation=True, exposure="EXPOSED_EXPLORATORY",
                r7_complement_comparison=dict(
                    status="NOT_RUN",
                    reason="Explicit --k1-only: R7 fk-only predictions for folds 1..3 "
                           "are unavailable; GPU inference is not scheduled."),
                comparison="Cross-fold/host deltas are descriptive, not paired gain evidence.")


def default_k1_manifest():
    folds = {}
    for fold in range(4):
        base = (Path(".local/scratch/l1-rco/k1v3-grouping/fit") if fold == 0 else
                Path(f".local/scratch/astra-b2/h224-device-plan-v1/exports/f{fold}/fit"))
        folds[str(fold)] = dict(k1_csv=str(base / "holdout.csv"),
                                k1_result=str(base / "result.json"))
    return dict(schema="h224-k1-only-input/v1", folds=folds)


def rosters():
    parent = k.read(k.PARENT)
    ids = sorted(parent["train"] + parent["holdout"])
    k.need(len(ids) == len(set(ids)) == 4167, "common full labelled roster")
    return {f: [u for u in ids if k.s.fold(u) == f] for f in range(4)}


def predictions(path, ids, fold):
    import numpy as np
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        k.need(len(fields) == len(set(fields))
               and {k.s.UID, *k.s.LABELS, "status", "model_fold"} <= set(fields),
               "prediction columns missing/duplicated")
        rows = list(reader)
    k.need(all(None not in r and all(v is not None for v in r.values()) for r in rows),
           "malformed prediction rows")
    k.need(len(rows) == len(ids) and {r[k.s.UID] for r in rows} == set(ids),
           "exact own-holdout roster required; no dropping invalid rows")
    by = {r[k.s.UID]: r for r in rows}
    k.need(all(r["status"] == "ok" and r["model_fold"] == str(fold) for r in rows),
           "prediction status/fold mismatch")
    try:
        values = np.array([[float(by[u][c]) for c in k.s.LABELS] for u in ids])
    except (ValueError, TypeError):
        raise ValueError("nonnumeric prediction probabilities") from None
    k.need(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
           "invalid prediction probabilities")
    return values


HOST_FLOAT_FIELDS = ("constant_loss", "threshold")
HOST_PATH_FLAGS = ("--px", "--targets", "--out", "--init")
HOST_FLOAT_ATOL = 1e-12


def prepared_hashes(reg, backend=None):
    """Reproduce prepare() bytes in memory; never write training artifacts."""
    k = backend or globals()["k"]
    rows = k.training_rows(reg)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    source = (k.ROOT / k.s.SOURCE / "compact_train.py").read_text(encoding="utf-8")
    return dict(targets_sha256=hashlib.sha256(stream.getvalue().encode("utf-8")).hexdigest(),
                trainer_sha256=hashlib.sha256(k.derive_trainer(source).encode("utf-8")).hexdigest())


def validate_host_export(paths, result, expected, fold, backend=None):
    """Only explicitly enumerated host fields may differ; all else is exact."""
    k = backend or globals()["k"]
    base = Path(paths["k1_result"]).resolve().parent.parent
    export_path = base / "EXPORT.json"
    export = k.read(export_path)
    k.need(export["fold"] == fold and export["contract_id"] == result["contract_id"],
           "host export fold/contract mismatch")
    bound = {"registration.json": base / "registration.json",
             "prepare.json": base / "prepare.json",
             "fit/result.json": Path(paths["k1_result"]),
             "fit/holdout.csv": Path(paths["k1_csv"])}
    for name, path in bound.items():
        k.need(path.resolve() == (base / name).resolve()
               and export["sha256"][name] == k.sha(path), "host export hash/path mismatch")
    reg = k.read(bound["registration.json"])
    k.need(reg["contract_id"] == result["contract_id"]
           and k.digest({n: v for n, v in reg.items() if n != "contract_id"}) == reg["contract_id"],
           "host registration digest/contract mismatch")
    normalized = json.loads(json.dumps(reg))
    differences = []
    for field in HOST_FLOAT_FIELDS:
        actual, local = reg["collapse_guard"][field], expected["collapse_guard"][field]
        k.need(type(actual) in (int, float) and math.isfinite(actual)
               and abs(actual - local) <= HOST_FLOAT_ATOL, "host collapse constant mismatch")
        if actual != local:
            differences.append("registration.collapse_guard." + field)
        normalized["collapse_guard"][field] = local
    normalized["contract_id"] = expected["contract_id"]
    # Includes exact ordered UID lists, all pins, recipe and every unknown field.
    k.need(normalized == expected, "host registration scientific identity mismatch")
    prep = k.read(bound["prepare.json"])
    k.need(set(prep) == {"contract_id", "trainer_sha256", "targets_sha256",
                        "train_count", "holdout_training_rows", "argv", "cache"}
           and prep["contract_id"] == reg["contract_id"]
           and prep["train_count"] == len(expected["train"])
           and prep["holdout_training_rows"] == 0, "host prepare identity mismatch")
    hashes = prepared_hashes(expected) if backend is None else prepared_hashes(expected, backend)
    k.need(all(prep[n] == value for n, value in hashes.items()),
           "host prepared targets/trainer hash mismatch")
    local_argv = k.trainer_argv(k.CACHE, expected)
    argv = prep["argv"]
    k.need(isinstance(argv, list) and len(argv) == len(local_argv), "host argv shape mismatch")
    path_fields = {0: "python", 1: "trainer",
                   **{local_argv.index(flag) + 1: flag for flag in HOST_PATH_FLAGS}}
    guard_index = local_argv.index("--l1-guard-constant") + 1
    for index, local in enumerate(local_argv):
        actual = argv[index]
        if index in path_fields:
            k.need(isinstance(actual, str) and
                   (PurePosixPath(actual).is_absolute() or PureWindowsPath(actual).is_absolute()),
                   "host argv path must be absolute")
            if actual != local:
                differences.append("prepare.argv." + path_fields[index])
        elif index == guard_index:
            k.need(actual == repr(reg["collapse_guard"]["constant_loss"]),
                   "host argv collapse constant mismatch")
            if actual != local:
                differences.append("prepare.argv.--l1-guard-constant")
        else:
            k.need(actual == local, "host argv recipe mismatch")
    k.need(prep["cache"] == argv[local_argv.index("--px") + 1], "host cache/argv mismatch")
    if prep["cache"] != str(Path(k.CACHE).resolve()):
        differences.append("prepare.cache")
    return dict(mode="HOST_BOUND_EXPORT", contract_id=reg["contract_id"],
                local_expected_contract_id=expected["contract_id"],
                differing_fields=differences, collapse_constant_atol=HOST_FLOAT_ATOL,
                export_sha256=k.sha(export_path),
                files={n: export["sha256"][n] for n in bound})


def validate_k1(paths, ids, fold, *, allow_host_export=False):
    result = k.read(paths["k1_result"])
    k.need(result["status"] == "COMPLETE" and result["cpu_test"] is False
           and result["model_fold"] == fold and result["holdout_rows"] == len(ids)
           and result["holdout_sha256"] == k.sha(paths["k1_csv"]),
           "K1 production result/CSV mismatch")
    if fold == 0:
        contract = k.read(k.PARENT)["contract_id"]
    else:
        k.configure(fold)
        expected = k.expected_registration()
        contract = expected["contract_id"]
        if allow_host_export and result["contract_id"] != contract:
            return validate_host_export(paths, result, expected, fold)
    k.need(result["contract_id"] == contract, "K1 v3-G seed0 contract mismatch")
    return dict(mode="EXACT_LOCAL", contract_id=contract, differing_fields=[])


def macro_aucs(scores, refs):
    import numpy as np
    from tools.l1_rco_k1_eval_v1 import auc
    k.need(set(refs) == set(RULERS), "exact four rulers required")
    result = {}
    for ruler in RULERS:
        targets, mask, _ = refs[ruler]
        k.need(targets.shape == scores.shape == mask.shape and mask.dtype == bool
               and np.isfinite(targets).all() and np.isin(targets, [0, 1]).all(),
               "invalid ruler values/shape/mask")
        values = [float(auc(targets[mask[:, c], c], scores[mask[:, c], c]))
                  for c in range(12)]
        k.need(np.isfinite(values).all(), "undefined target AUC; HOLD")
        result[ruler] = float(np.mean(values))
    return result


def decide_k1(metrics):
    import math
    k.need(0 in metrics and set(metrics) <= set(range(4)), "K1 f0 baseline required")
    for values in metrics.values():
        k.need(set(values) == set(RULERS)
               and all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()),
               "all four finite macro AUCs required")
    present = sorted(set(metrics) - {0})
    detail = {}
    for ruler in RULERS:
        deltas = {str(f): metrics[f][ruler] - metrics[0][ruler] for f in present}
        detail[ruler] = dict(fold_vs_f0=deltas,
                             one_sided_pass=all(v >= -.005 - 1e-12 for v in deltas.values()))
    passed = all(d["one_sided_pass"] for d in detail.values())
    states = {"0": "BASELINE"}
    for fold in (1, 2, 3):
        states[str(fold)] = ("PENDING" if fold not in metrics else
                            "HEALTH_PASS" if all(metrics[fold][r] >= metrics[0][r] - .005 - 1e-12
                                                 for r in RULERS) else "HOLD")
    return dict(health_metadata(), status=("PENDING" if not present else
                "HEALTH_PASS" if passed else "HOLD"), detail=detail, folds=states,
                all_folds_evaluated=len(present) == 3,
                pending_folds=[f for f in (1, 2, 3) if f not in metrics])


def evaluate_k1(manifest):
    from tools import exp156_k_sweep as rulers
    k.need(manifest["schema"] == "h224-k1-only-input/v1"
           and "0" in manifest["folds"]
           and set(manifest["folds"]) <= {"0", "1", "2", "3"},
           "K1-only manifest schema/folds")
    metrics, pins, errors = {}, {}, {}
    for fold, ids in rosters().items():
        if str(fold) not in manifest["folds"]:
            continue
        try:
            spec = manifest["folds"][str(fold)]
            k.need(set(spec) == {"k1_csv", "k1_result"}, "K1-only input file fields")
            paths = {name: ROOT / path for name, path in spec.items()}
            # Both absent means not exported yet; partial exports are invalid.
            if fold != 0 and not any(p.exists() for p in paths.values()):
                continue
            contract = validate_k1(paths, ids, fold, allow_host_export=True)
            scores = predictions(paths["k1_csv"], ids, fold)
            metrics[fold] = macro_aucs(scores, rulers.references(ids, root=ROOT))
            pins[str(fold)] = dict(files={n: k.sha(p) for n, p in paths.items()},
                                   cohort_sha256=k.digest(ids), studies=len(ids),
                                   contract=contract)
        except (ValueError, KeyError, OSError, TypeError, AttributeError) as exc:
            # Do not echo exception payloads: malformed inputs may contain row values.
            errors[str(fold)] = dict(status="HOLD", error_type=type(exc).__name__,
                                    reason="Invalid/incomplete K1 production evidence or ruler AUC")
    if 0 in metrics:
        result = decide_k1(metrics)
    else:
        result = dict(health_metadata(), status="HOLD", all_folds_evaluated=False,
                      folds={str(f): "UNASSESSED" if f in metrics else "PENDING"
                             for f in range(4)},
                      pending_folds=[f for f in (1, 2, 3) if f not in metrics])
    if errors:
        result.update(status="HOLD", all_folds_evaluated=False)
        result["folds"].update({f: "HOLD" for f in errors})
        result["pending_folds"] = [f for f in result["pending_folds"] if str(f) not in errors]
    return dict(result, metrics=metrics, evidence=pins, errors=errors)


def evaluate_fullbag(manifest):
    """Paired exact fold0 roster; never use cross-fold health tolerance."""
    from tools import l1_rco_k1_fullbag_v1 as fullbag
    from tools import exp156_k_sweep as rulers
    k.need(manifest["schema"] == "k1-fullbag-gate-input/v1"
           and set(manifest["models"]) == {"k1_v3g_seed0", "k1_v3g_fullbag"},
           "fullbag input schema/models")
    expected = fullbag.expected_registration()
    ids = expected["holdout"]
    k.need(len(ids) == len(set(ids)) == 977 and set(ids) == set(rosters()[0]),
           "exact 977 fold0 roster")
    refs = rulers.references(ids, root=ROOT)
    metrics, evidence = {}, {}
    for model, spec in manifest["models"].items():
        k.need(set(spec) == {"k1_csv", "k1_result"}, "fullbag input fields")
        paths = {name: ROOT / path for name, path in spec.items()}
        if model == "k1_v3g_seed0":
            binding = validate_k1(paths, ids, 0)
        else:
            result = k.read(paths["k1_result"])
            k.need(result["status"] == "COMPLETE" and result["cpu_test"] is False
                   and result["model_fold"] == 0 and result["holdout_rows"] == 977
                   and result["holdout_sha256"] == k.sha(paths["k1_csv"]),
                   "fullbag production result/CSV mismatch")
            binding = validate_host_export(paths, result, expected, 0, fullbag.k)
        metrics[model] = macro_aucs(predictions(paths["k1_csv"], ids, 0), refs)
        evidence[model] = dict(binding=binding, files={n: k.sha(p) for n, p in paths.items()})
    result = fullbag.decide(metrics["k1_v3g_seed0"], metrics["k1_v3g_fullbag"])
    return dict(result, mode="k1-only-fullbag", metrics=metrics, evidence=evidence,
                holdout_rows=977, cohort_sha256=k.digest(ids), clean_oof=False,
                own_holdout_evaluation=True, exposure="EXPOSED_EXPLORATORY",
                interpretation="Host/window exposure confounded; independent review pending",
                r7_complement_comparison="NOT_RUN")


def decide(metrics):
    import math
    k.need(set(metrics) == set(range(4)), "all four folds required")
    for fold in range(4):
        k.need(set(metrics[fold]) == {"k1", "r7"}, "both models required")
        for model in ("k1", "r7"):
            values = metrics[fold][model]
            k.need(set(values) == set(RULERS)
                   and all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()),
                   "all four finite macro AUCs required")
    detail = {}
    for ruler in RULERS:
        base = metrics[0]["k1"][ruler]
        deltas = {str(f): metrics[f]["k1"][ruler] - base for f in (1, 2, 3)}
        wins = [f for f in range(4) if metrics[f]["k1"][ruler] >= metrics[f]["r7"][ruler]]
        # The user's "<owner remark removed>" wording is applied literally, before results.
        detail[ruler] = dict(fold_vs_f0=deltas, k1_ge_r7_folds=wins,
                             within_005=all(abs(v) <= .005 + 1e-12 for v in deltas.values()),
                             at_least_three_folds=len(wins) >= 3)
    passed = all(d["within_005"] and d["at_least_three_folds"] for d in detail.values())
    return dict(status="SCREEN_PASS_PENDING_REVIEW" if passed else "HOLD",
                detail=detail, clean_oof=False, own_holdout_evaluation=True,
                policy="absolute .005 band vs f0; >=3/4 folds vs same-fold R7; every fixed ruler",
                comparison="Cross-fold/host deltas are descriptive, not paired gain evidence.")


def evaluate(manifest):
    import numpy as np
    from tools import exp156_k_sweep as rulers
    from tools.l1_rco_k1_eval_v1 import auc
    k.need(manifest["schema"] == "h224-own-holdout-input/v1"
           and set(manifest["folds"]) == {"0", "1", "2", "3"}, "input manifest schema/folds")
    metrics, pins = {}, {}
    for fold, ids in rosters().items():
        paths = {name: ROOT / path for name, path in manifest["folds"][str(fold)].items()}
        k.need(set(paths) == {"k1_csv", "k1_result", "r7_csv", "r7_receipt"}, "input file fields")
        validate_k1(paths, ids, fold)
        receipt = k.read(paths["r7_receipt"])
        k.need(receipt["complete"] is True and receipt["mode"] == f"screen_f{fold}"
               and receipt["model_selection"] == f"f{fold}_only"
               and receipt["output_kind"] == "pre_stacker" and receipt["clean_oof"] is False
               and receipt["csv_sha256"] == k.sha(paths["r7_csv"]),
               "R7 same-fold production provenance required")
        refs = rulers.references(ids, root=ROOT)
        metrics[fold] = {}
        for model in ("k1", "r7"):
            scores = predictions(paths[model + "_csv"], ids, fold)
            metrics[fold][model] = {}
            for ruler in RULERS:
                targets, mask, _ = refs[ruler]
                values = [float(auc(targets[mask[:,c],c], scores[mask[:,c],c])) for c in range(12)]
                k.need(np.isfinite(values).all(), "undefined target AUC; HOLD")
                metrics[fold][model][ruler] = float(np.mean(values))
        pins[str(fold)] = dict(files={n: k.sha(p) for n,p in paths.items()},
                               cohort_sha256=k.digest(ids), studies=len(ids))
    return dict(decide(metrics), metrics=metrics, evidence=pins)


def dry_run():
    from tools import exp156_k_sweep as rulers
    counts = {}
    for fold, ids in rosters().items():
        refs = rulers.references(ids, root=ROOT)
        counts[str(fold)] = dict(studies=len(ids), cohort_sha256=k.digest(ids),
                                coverage={n: int(mask.sum()) for n,(_,mask,_) in refs.items()})
    return dict(status="REFERENCE_COVERAGE_ONLY", folds=counts, clean_oof=False,
                gate_evaluated=False, all_data_own_holdout_possible=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fullbag", action="store_true", help="Exact fold0 fullbag vs seed0 >=3/4 screen")
    ap.add_argument("--k1-only", action="store_true",
                    help="One-sided K1 health gate; missing fold exports stay PENDING; no R7 reads")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        k.need(not args.fullbag or (args.k1_only and args.manifest and not args.dry_run),
               "--fullbag requires --k1-only --manifest; no --dry-run")
        k.need(args.dry_run or args.k1_only or args.manifest is not None, "--manifest required")
        if args.fullbag:
            result = evaluate_fullbag(k.read(args.manifest))
        elif args.dry_run:
            result = dry_run()
            if args.k1_only:
                result.update(health_metadata())
        elif args.k1_only:
            result = evaluate_k1(k.read(args.manifest) if args.manifest else default_k1_manifest())
        else:
            result = evaluate(k.read(args.manifest))
    except (ValueError, KeyError, OSError, TypeError, AttributeError) as exc:
        result = dict(status="HOLD", reason="Invalid gate input/evidence",
                      error_type=type(exc).__name__, clean_oof=False)
        if args.k1_only and not args.fullbag:
            result.update(health_metadata())
    k.write(args.output or (K1_OUT if args.k1_only else OUT), result)
    print(json.dumps(result, allow_nan=False))
    return 2 if result["status"] == "HOLD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
