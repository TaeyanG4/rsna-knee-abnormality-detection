"""H-B-150 native320 versus matched H-B-146 PL224 readout, v1.0.0.

Changelog 2026-10-14: frozen paired 42/43 EXPOSED screen; arithmetic
seed-mean prediction AUC gate, all seed/ruler/target cells guarded.
The caller authenticates completed prediction receipts before this pure API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import multi_lens_readout_l4_v2 as lens

VERSION = "1.0.0"
SEEDS = ("42", "43")
RULE = {"macro_threshold": .001, "weak4_threshold": .001,
        "minimum_same_rulers": 3, "cell_floor": -.003,
        "seeds": list(SEEDS), "rulers": list(lens.RULERS),
        "aggregate": "AUC of arithmetic seed-mean predictions; not mean seed AUC",
        "cell_scope": "each seed and arithmetic seed-mean prediction; all four rulers and twelve targets"}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def validate_records(summary, receipt, expected, *, csv_sha256, checkpoint_sha256,
                     uid_sha256, rows, baseline=False, gpu_attestation=None):
    """Validate completed immutable prediction metadata; no file/label access.

    expected carries the review-bound registration/prepare/labels hashes,
    model/weights, GPU name, and completed update count. Older H146 receipts
    lacking a GPU name need an independently pinned carry-over attestation.
    """
    identity = receipt.get("identity", {})
    _need(identity == summary.get("identity"), "summary/receipt identity differs")
    _need(identity.get("arm") in ("global", "fill") and identity.get("seed") in (42, 43), "wrong PL arm/seed")
    for key in ("arm", "seed", "epochs", "registration_sha256", "prepare_sha256",
                "labels_sha256", "weights", "model"):
        _need(key in expected and identity.get(key) == expected[key], "identity binding differs: " + key)
    _need(identity["epochs"] == 20 and identity.get("smoke") is False
          and identity.get("engineering_only") is False and summary.get("smoke") is False
          and receipt.get("engineering_only") is False, "older/partial/engineering baseline forbidden")
    _need(identity.get("exposed_exploratory") is True
          and identity.get("status") in ("EXPOSED", "EXPOSED_EXPLORATORY"), "EXPOSED identity required")
    _need(summary.get("device") == "cuda", "scientific CUDA baseline required")
    steps = expected.get("steps")
    skips = summary.get("counters", {}).get("skips")
    _need(type(steps) is int and steps > 0 and summary.get("steps") == steps
          and summary.get("counters", {}).get("updates") == steps
          and type(skips) is int and 0 <= skips <= max(1, int(steps * .01)),
          "incomplete fit update count or skipped-update allowance exceeded")
    _need(receipt.get("k") == 36 and receipt.get("tta") is True, "K36 flip prediction required")
    _need(receipt.get("csv_sha256") == csv_sha256 and receipt.get("checkpoint_sha256")
          == summary.get("checkpoint_sha256") == checkpoint_sha256, "prediction/checkpoint hash differs")
    _need(receipt.get("rows") == rows and receipt.get("uid_sha256") == uid_sha256, "prediction roster differs")
    runtime = identity.get("runtime", {})
    _need(str(runtime.get("device", "")).startswith("cuda") and runtime.get("policy") == "strict"
          and runtime.get("precision_fallback_allowed") is False, "registered CUDA runtime differs")
    if "runtime" in expected:
        _need(runtime == expected["runtime"], "paired execution runtime differs")
    actual_gpu = identity.get("gpu_name")
    if baseline and actual_gpu is None:
        att = gpu_attestation or {}
        _need(att.get("reviewed") is True and att.get("checkpoint_sha256") == checkpoint_sha256
              and att.get("csv_sha256") == csv_sha256 and att.get("identity") == identity,
              "legacy baseline needs pinned independently reviewed GPU carry-over attestation")
        actual_gpu = att.get("gpu_name")
    _need(expected.get("gpu_name") == "NVIDIA GeForce RTX 4070 Ti SUPER"
          and actual_gpu == expected["gpu_name"], "paired 4070 GPU identity differs")
    if not baseline:
        _need(identity.get("resolution") == 320 and identity.get("clean_oof") is False
              and identity.get("exposure") == "EXPOSED"
              and identity.get("config_sha256") == expected.get("config_sha256")
              and bool(expected.get("config_sha256")), "H150 native320 config/exposure differs")
    return identity


def authenticated_prediction(directory, ids, expected, *, name="val", baseline=False,
                             gpu_attestation=None, gpu_attestation_sha256=None, metadata_only=False,
                             gpu_attestation_record=None):
    """Read only completed outputs, authenticating before returning predictions.

    The parent admission receipt must pin expected and any GPU attestation hash.
    A heartbeat is the only file opened before completion has been established.
    """
    from tools import hb146_pl_labels_v111 as pl
    directory = Path(directory)
    heartbeat = pl.read(directory / "heartbeat.json")
    _need(heartbeat.get("status") == "complete", "fit not complete; output access forbidden")
    _need(name in ("val", "gold", "oai"), "unregistered prediction panel")
    path = directory / (name + ".csv")
    receipt = pl.read(path.with_suffix(".receipt.json"))
    summary = pl.read(directory / "summary.json")
    att = gpu_attestation_record
    if gpu_attestation is not None:
        _need(bool(gpu_attestation_sha256) and pl.sha(gpu_attestation) == gpu_attestation_sha256,
              "unbound baseline GPU attestation")
        att = pl.read(gpu_attestation)
    # Exact training.arms.mil.uid_hash protocol; no training/CUDA imports.
    validate_records(summary, receipt, expected, csv_sha256=pl.sha(path),
                     checkpoint_sha256=pl.sha(directory / "final.pt"),
                     uid_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(), rows=len(ids), baseline=baseline,
                     gpu_attestation=att)
    if metadata_only:
        return receipt
    return pl.aligned(pl.table(path), ids, exact=True), receipt


def gate(seed_points, mean_points, engineering=False):
    """Fail closed unless finite exact 2x4x12 paired and 4x12 mean evidence."""
    seeds, mean = np.asarray(seed_points, dtype=float), np.asarray(mean_points, dtype=float)
    valid = bool(seeds.shape == (2, 4, 12) and mean.shape == (4, 12)
                 and np.isfinite(seeds).all() and np.isfinite(mean).all())
    macro = mean.mean(axis=1) if mean.shape == (4, 12) else np.full(4, np.nan)
    weak = mean[:, lens._base.WEAK].mean(axis=1) if mean.shape == (4, 12) else np.full(4, np.nan)
    qualifying = ((macro >= .001) & (weak >= .001)) if valid else np.zeros(4, dtype=bool)
    floor_ok = bool(valid and np.all(seeds >= -.003) and np.all(mean >= -.003))
    passed = bool(valid and qualifying.sum() >= 3 and floor_ok and not engineering)
    return lens._base.clean({
        "GO": passed, "status": "ENGINEERING_ONLY" if engineering else "GO" if passed else "NO_GO",
        "eligible": valid and not engineering, "finite_exact_evidence": valid,
        "cell_floor_pass": floor_ok, "qualifying_report_rulers": [r for r, ok in zip(lens.RULERS, qualifying) if ok],
        "report_macro_seed_mean_deltas": dict(zip(lens.RULERS, macro)),
        "report_weak4_seed_mean_deltas": dict(zip(lens.RULERS, weak)),
        "seed_point_deltas": seeds, "seed_mean_prediction_point_deltas": mean,
        "rule": RULE, "status_scope": "EXPOSED_EXPLORATORY", "clean_oof": False,
        "not_release_authorization": True})


def _paired(predictions, rows):
    if len(predictions) != 2 or {str(s) for s in predictions} != set(SEEDS):
        raise ValueError("Exactly paired seeds 42 and 43 required")
    normalized = {str(s): np.asarray(p, dtype=float) for s, p in predictions.items()}
    for p in normalized.values():
        if p.shape != (rows, 12) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Finite aligned N x 12 probabilities required")
    return {seed: normalized[seed] for seed in SEEDS}


def evaluate(ctx, arms, controls, image_reads=None, engineering=False):
    """Use authenticated L4-v2 telemetry, with this experiment's registered gate.

    Arms are native-cache 320 PL predictions; controls are the same winning
    H-B-146 PL arm at 224 (not H-B-146's no-PL control). Parent runner owns
    receipt, split, cache, label, GPU type, and hyperparameter authentication.
    Images are descriptive regardless of an upstream eligibility flag.
    """
    aa, cc = _paired(arms, len(ctx.ids)), _paired(controls, len(ctx.ids))
    images = ({name: {**entry, "eligible": False} for name, entry in image_reads.items()}
              if image_reads else None)
    result = lens.evaluate(ctx, aa, cc, label_change=True, image_reads=images)
    seeds = np.stack([lens._tensor(ctx.delta(aa[s], cc[s]))[0] for s in SEEDS])
    mean_delta = lens._tensor(ctx.delta(np.mean(list(aa.values()), axis=0),
                                       np.mean(list(cc.values()), axis=0)))
    registered = gate(seeds, mean_delta[0], engineering)
    result["report_seed_mean_macros"] = {
        r: lens._base.stat(mean_delta[:, k].mean(axis=1), .001)
        for k, r in enumerate(lens.RULERS)}
    result["lenses"]["L6"] = lens._base.unavailable("No registered compatible partner; no bundle claim")
    result["lenses"]["L7"].update(eligible=False, PASS=False, role="descriptive only; never selects H-B-150")
    for entry in result["lenses"]["L7"].get("entries", {}).values():
        entry.update(eligible=False, PASS=False, role="descriptive/exposed; no gate contribution")
    result["legacy_lens_verdict"] = {k: result.get(k) for k in
        ("verdict", "passing_lenses", "decision_label", "release_scope")}
    result.update(version=VERSION, readout_backend="multi_lens_readout_l4_v2", registered_go=registered,
                  status_scope="EXPOSED_EXPLORATORY", exposure="EXPOSED", clean_oof=False,
                  engineering_only=bool(engineering), verdict=registered["status"],
                  decision_label=registered["status"], release_scope="none",
                  not_release_authorization=True)
    result["notes"] = [
        "Matched comparators are H-B-146 winning-arm PL fits, seeds 42/43, same 4070 GPU type.",
        "Gate uses AUC of arithmetic seed-mean predictions, not mean seed AUC.",
        "Four report rulers are correlated and are not independent replications.",
        "Gold58 and OAI are descriptive only; neither can qualify or rescue GO.",
        "Role f0 pseudo-labels are EXPOSED; clean_oof=false. No confirmatory or release claim.",
        "Legacy L1-L7 and their thresholds are diagnostic telemetry; registered_go alone is the H-B-150 screen."]
    return lens._base.clean(result)


def expected_record(c, seed, baseline=False):
    from tools import hb150_build_v1 as build
    w = ROOT / c["pl_work"]
    labels = build.read(w / "screen_f0/provenance.json")
    return {"arm": c["arm"], "seed": seed, "epochs": 20,
            "registration_sha256": build.sha(ROOT / c["registration"]),
            "prepare_sha256": build.sha(w / "prepare.json") if baseline else build.sha(build.WORK / f"prepared-{c['arm']}.json"),
            "labels_sha256": labels["arm_hashes"][c["arm"]], "weights": c["weights"],
            "model": c["recipe"]["model"], "steps": c["optimizer_steps"], "gpu_name": build.GPU,
            "config_sha256": build.sha(build.config_path(c["arm"], seed))}


def validate_baselines(c, attestation_path):
    """Metadata/hash-only admission; caller independent review pins bundle hash.

    Bundle: {schema:'hb150-baseline-attestation/v1',members:{'42':att,'43':att}}.
    Each att contains reviewed:true, gpu_name, checkpoint_sha256, csv_sha256,
    and the complete H146 identity. No label/prediction CSV is parsed here.
    """
    from tools import hb150_build_v1 as build
    w = ROOT / c["pl_work"]
    for seed in (42, 43):
        _need(build.read(w / "fits" / f"{c['arm']}-s{seed}" / "heartbeat.json").get("status") == "complete",
              "fit not complete; output access forbidden")
    bundle = build.read(attestation_path)
    _need(bundle.get("schema") == "hb150-baseline-attestation/v1"
          and set(bundle.get("members", {})) == set(SEEDS), "baseline attestation bundle differs")
    decision = build.read(w / "readout.json")
    _need(decision.get("registered_go", {}).get("GO") is True
          and decision["registered_go"].get("selected_arm") == c["arm"], "H146 winning PL arm differs")
    records = {}
    for seed in (42, 43):
        records[str(seed)] = authenticated_prediction(w / "fits" / f"{c['arm']}-s{seed}", c["val"],
            expected_record(c, seed, baseline=True), baseline=True, metadata_only=True,
            gpu_attestation_record=bundle["members"][str(seed)])
        _need(decision.get("prediction_bindings", {}).get(f"val-{c['arm']}-{seed}") == records[str(seed)],
              "H146 winning readout refers to older prediction pair")
    _need(records["42"]["identity"]["runtime"] == records["43"]["identity"]["runtime"],
          "baseline seed runtimes differ")
    return records


def run(arm, review, exposed_exploratory=False):
    from tools import hb150_build_v1 as build
    from tools import hb146_pl_labels_v111 as pl
    _need(exposed_exploratory, "explicit --exposed-exploratory required")
    # Never inspect predictions until the independent review and all input pins pass.
    c = build.require_fit(build.config_path(arm, 42), review)
    review_record = build.read(review)
    bundle = build.read(ROOT / review_record["baseline_attestation"])
    w = ROOT / c["pl_work"]
    arms, controls, bindings = {}, {}, {}
    runtime = None
    for seed in (42, 43):
        for baseline, target in ((True, controls), (False, arms)):
            directory = (w if baseline else build.WORK) / "fits" / f"{arm}-s{seed}"
            values, receipt = authenticated_prediction(directory, c["val"], expected_record(c, seed, baseline),
                baseline=baseline, gpu_attestation_record=bundle["members"][str(seed)] if baseline else None)
            if runtime is None:
                runtime = receipt["identity"]["runtime"]
            _need(receipt["identity"]["runtime"] == runtime, "paired execution runtime differs")
            target[str(seed)] = values
            bindings[("baseline224" if baseline else "native320") + f"-s{seed}"] = receipt
    ctx = lens._base.make_context(c["val"], ROOT)
    # Gold is optional descriptive evidence. No directory listing; only finished fits.
    images = None
    gold_paths = [(w if baseline else build.WORK) / "fits" / f"{arm}-s{seed}" / "gold.csv"
                  for baseline in (True, False) for seed in (42, 43)]
    if all(path.is_file() for path in gold_paths):
        gold_labels = pl.aligned(pl.table(pl.GOLD, allow_missing=True), c["gold"])
        gold_predictions = {True: [], False: []}
        for baseline in (True, False):
            for seed in (42, 43):
                directory = (w if baseline else build.WORK) / "fits" / f"{arm}-s{seed}"
                # GPU attestation covers the val CSV; use the already authenticated
                # same identity's GPU for descriptive Gold, preserving all hashes.
                expect = expected_record(c, seed, baseline)
                if baseline:
                    att = dict(bundle["members"][str(seed)])
                    att["csv_sha256"] = pl.sha(directory / "gold.csv")
                else:
                    att = None
                values, receipt = authenticated_prediction(directory, c["gold"], expect, name="gold",
                    baseline=baseline, gpu_attestation_record=att)
                _need(receipt["identity"]["runtime"] == runtime, "Gold runtime differs")
                gold_predictions[baseline].append(values)
                bindings[("baseline224" if baseline else "native320") + f"-gold-s{seed}"] = receipt
        groups = pl.read(ROOT / lens._base.COMPONENTS)
        covered = {u for component in groups for u in component["members"]}
        missing = set(c["gold"]) - covered
        groups += [{"component_id": "gold-unmapped-" + u, "members": [u]} for u in sorted(missing)]
        images = {"gold58": {"ids": c["gold"], "labels": gold_labels, "mask": np.isfinite(gold_labels),
                  "components": groups, "arm": np.mean(gold_predictions[False], axis=0),
                  "control": np.mean(gold_predictions[True], axis=0), "eligible": False,
                  "component_policy": "known groups plus unmapped singleton studies; descriptive only"}}
    result = evaluate(ctx, arms, controls, image_reads=images)
    result.update(prediction_bindings=bindings, ruler_input_hashes=ctx.bindings, chosen_arm=arm,
                  independent_review_sha256=build.sha(review),
                  oai_status="unavailable: no H150 OAI panel registered; descriptive only, excluded from GO")
    path = build.WORK / f"readout-{arm}.json"
    from tools.hb150_io_v1 import atomic_json
    atomic_json(path, result)
    return {"readout": str(path), "status": result["registered_go"]["status"],
            "exposure": "EXPOSED", "clean_oof": False, "rows": len(c["val"])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("global", "fill"), required=True)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--exposed-exploratory", action="store_true")
    args = ap.parse_args(argv)
    print(json.dumps(run(args.arm, args.review, args.exposed_exploratory)))


if __name__ == "__main__":
    main()
