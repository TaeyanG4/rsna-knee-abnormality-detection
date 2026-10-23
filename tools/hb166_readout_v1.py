"""H-B-166 registered readout + seven lenses v1.0.0 (2026-10-15). Local CPU only.

Pairs the two multiview fits with the EXISTING H-B-155 control fits (same
fold 2 rows, seeds 42/43, recipe, 4090 stack). Everything decisive is fixed in
tools/hb166_build_v1.py (GATE, LENS_PLAN, L6 partner pins) before any fit:
  registered pass signal = 4-ruler mean 12-target macro delta >= +0.002 AND
  paired component-bootstrap 95 % lower bound > 0 on >= 2/4 rulers AND no
  target 4-ruler mean delta < -0.005 AND both seeds positive on the 4-ruler mean
  (aggregate: AUC of arithmetic seed-mean probabilities). PASS => fold-3
  confirmation, never a release (fold 2 informed the idea).
Lenses L1-L5 via tools/multi_lens_readout_l4_v2.evaluate, L6 = HB105 fold-2
partner bundle on the same 864 rows, L7 = Gold58 (exposed, descriptive) with
the registered OAI ineligibility reason. Never prints label rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools import hb166_build_v1 as b
from tools import multi_lens_readout_l4_v2 as lens

SEEDS = ("42", "43")
UID = "StudyInstanceUID"


def registered_gate(seed_deltas, mean_delta, engineering=False):
    """seed_deltas: {seed: [1+draws, 4, 12]}, mean_delta: same shape for seed-mean predictions."""
    g = b.GATE
    try:
        mean = np.asarray(mean_delta, dtype=float)
        seeds = {s: np.asarray(d, dtype=float) for s, d in seed_deltas.items()}
        valid = (mean.ndim == 3 and mean.shape[1:] == (4, 12) and mean.shape[0] >= 2 and np.isfinite(mean[0]).all()
                 and set(seeds) == set(SEEDS) and all(d.shape == mean.shape and np.isfinite(d[0]).all() for d in seeds.values()))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return dict(PASS=False, status="INVALID_EVIDENCE", rule=g, not_release_authorization=True)
    macro = mean.mean(axis=2)                                   # [1+draws, 4]
    four = float(macro[0].mean())
    lower = []
    for r in range(4):
        finite = macro[1:, r][np.isfinite(macro[1:, r])]
        lower.append(float(np.percentile(finite, 2.5)) if len(finite) else float("nan"))
    ci_positive = int(sum(1 for v in lower if np.isfinite(v) and v > 0))
    target = mean[0].mean(axis=0)                               # [12] 4-ruler mean per target
    worst = int(np.argmin(target))
    seed_four = {s: float(d[0].mean()) for s, d in seeds.items()}
    checks = dict(four_ruler_macro=four >= g["four_ruler_macro_min"] - 1e-12,
                  ci_lower_positive_rulers=ci_positive >= g["ci_positive_rulers_min"],
                  target_floor=bool((target >= g["target_floor"] - 1e-12).all()),
                  both_seeds_positive=all(v > 0 for v in seed_four.values()))
    passed = all(checks.values()) and not engineering
    status = "ENGINEERING_ONLY" if engineering else ("PASS_FOLD3_CONFIRMATION_REQUIRED" if passed else "NO_PASS")
    return lens._base.clean(dict(PASS=passed, status=status, checks=checks, four_ruler_macro=four,
        ruler_macro=dict(zip(lens.RULERS, macro[0])), ruler_ci95_lower=dict(zip(lens.RULERS, lower)),
        ci_positive_rulers=ci_positive, target_four_ruler_mean=dict(zip(lens.TARGETS, target)),
        worst_target=lens.TARGETS[worst], seed_four_ruler_macro=seed_four, rule=g,
        not_release_authorization=True, consequence=g["on_pass"] if passed else g["on_fail"]))


def load_csv(path, ids, pin=None):
    import pandas as pd
    path = Path(path)
    if pin is not None:
        b.need(b.sha(path) == pin, "prediction hash differs from pin: " + str(path))
    f = pd.read_csv(path, dtype={UID: str})
    b.need(f[UID].tolist() == list(ids), "prediction roster/order differs: " + str(path))
    p = f[list(lens.TARGETS)].to_numpy(dtype=float)
    b.need(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "invalid probabilities: " + str(path))
    return p


def partner_predictions(ids):
    """HB105 held-out fold-2 families; FP32 seed-probability mean (HB105 policy)."""
    out = {}
    import pandas as pd
    for family, seeds in b.L6_PARTNERS.items():
        tables = []
        for seed, (path, digest) in sorted(seeds.items()):
            f = pd.read_csv(b.ROOT / path, dtype={UID: str})
            b.need(b.sha(b.ROOT / path) == digest, "partner hash differs: " + path)
            f = f.set_index(UID)
            b.need(set(f.index) == set(ids) and len(f) == len(ids), "partner roster differs: " + path)
            tables.append(f.loc[list(ids), list(lens.TARGETS)].to_numpy(np.float32))
        out[family] = (np.mean(tables, axis=0, dtype=np.float32) if len(tables) > 1 else tables[0]).astype(float)
    return out


def l6_bundle(ctx, arm_mean, control_mean, partners):
    fixed = [lens._base.rank(p) for _, p in sorted(partners.items())]
    qa = np.mean([lens._base.rank(arm_mean), *fixed], axis=0)
    qc = np.mean([lens._base.rank(control_mean), *fixed], axis=0)
    return dict(context=ctx, arm=qa, control=qc, recipe=dict(
        family_replaced="S (own ConvNeXt-S finding-MIL slot: multiview vs H-B-155 control)",
        fixed_families=sorted(partners), Q="equal-family percentile ranks (HB105 'add' structure)",
        exposure="held-out fold-2 OOF partners; fold 2 selection-exposed", anchor_weight=None))


def gold_reads(gold_ids, arm_mean, control_mean, engineering=False):
    import pandas as pd
    path = b.ROOT / b.GOLD_LABELS
    b.need(b.sha(path) == b.GOLD_LABELS_SHA256, "Gold58 label file differs from pin")
    f = pd.read_csv(path, dtype={UID: str}).set_index(UID)
    y = f.loc[list(gold_ids), list(lens.TARGETS)].to_numpy(float)
    mask = np.isfinite(y)
    b.need(np.isin(y[mask], [0, 1]).all(), "Gold58 labels must be binary or missing")
    components = [{"component_id": "gold-" + u, "members": [u]} for u in gold_ids]
    return {"gold58": dict(ids=list(gold_ids), arm=arm_mean, control=control_mean, labels=np.nan_to_num(y), mask=mask,
                           components=components, eligible=False,
                           component_policy="Gold58 study singletons (outside the validation component contract); descriptive only")}


def lens_table(result):
    reasons = dict(
        L6="partner bundle computed" if result["lenses"]["L6"].get("eligible", True) else result["lenses"]["L6"].get("reason"),
        L7=b.LENS_PLAN["L7"])
    table = {}
    for name in ("L1", "L2", "L3", "L4", "L5", "L6", "L7"):
        entry = result["lenses"][name]
        eligible = bool(entry.get("eligible", True))
        row = dict(value=entry.get("value"), ci95=entry.get("ci95"), PASS=bool(entry.get("PASS")), eligible=eligible)
        if name == "L4":
            row["selected_targets"] = entry.get("selected_targets")
        if name == "L7":
            row["eligible"] = False
            row["reason"] = ("Gold58 entry is exposed/descriptive by the lens rule (never eligible); OAI MOAKS "
                             "ineligible: no native448/compartment-crop OAI cache exists (registered reason).")
        elif not eligible:
            row["reason"] = reasons.get(name) or entry.get("reason")
        table[name] = row
    return table


def evaluate(ctx, arms, controls, gold, partners, engineering=False):
    for table in (arms, controls):
        b.need(set(table) == set(SEEDS), "paired seeds 42/43 required")
        for p in table.values():
            p = np.asarray(p)
            b.need(p.shape == (len(ctx.ids), 12) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(),
                   "aligned finite probabilities required")
    arm_mean = np.mean([arms[s] for s in SEEDS], axis=0)
    control_mean = np.mean([controls[s] for s in SEEDS], axis=0)
    bundle = l6_bundle(ctx, arm_mean, control_mean, partners)
    reads = gold_reads(gold["ids"], np.mean([gold["arm"][s] for s in SEEDS], axis=0),
                       np.mean([gold["control"][s] for s in SEEDS], axis=0), engineering)
    result = lens.evaluate(ctx, arms, controls, label_change=False, bundle=bundle, image_reads=reads)
    seed_deltas = {s: lens._tensor(ctx.delta(arms[s], controls[s])) for s in SEEDS}
    mean_delta = lens._tensor(ctx.delta(arm_mean, control_mean))
    result["lenses"]["L7"]["reason"] = lens_table(result)["L7"]["reason"]
    result.update(exposure="EXPOSED", clean_oof=False, bias_statement=b.BIAS_STATEMENT,
                  registered_gate=registered_gate(seed_deltas, mean_delta, engineering),
                  lens_table=lens_table(result), engineering_only=engineering,
                  notes=["AUC of seed-mean probabilities for the registered gate; L1-L4 use mean seed AUC deltas (lens tool).",
                         "Existing H-B-155 controls (not refit), same fold/seeds/recipe/4090 stack; EXPOSED, clean_oof=false.",
                         "Fold 2 informed the idea: PASS => fold-3 confirmation, never a release.",
                         "300 component bootstrap draws; no multiplicity adjustment; lenses are evidence, not gates."])
    return lens._base.clean(result)


def validate_fit(c, directory, seed):
    summary, receipt = b.read(directory / "summary.json"), b.read(directory / "val.receipt.json")
    i = receipt.get("identity", {})
    b.need(i == summary.get("identity"), "summary/receipt identity differs")
    expected = dict(experiment="H-B-166 multiview paired pilot", arm="multiview", seed=seed, fold=2, epochs=20,
        smoke=False, engineering_only=False, gpu_name=b.GPU, batch=8, microbatch=2, accumulation=4, workers=2,
        config_sha256=b.sha(b.config_path(seed)), registration_sha256=b.sha(b.WORK / "registration.json"),
        train_sha256=b.uid_hash(c["train"]), val_sha256=b.uid_hash(c["val"]), gold_sha256=b.uid_hash(c["gold"]),
        window_budget=c["window_budget"], view_embedding=c["view_embedding"], labels_sha256=b.sha(b.ROOT / c["label_path"]),
        runtime_snapshot_sha256=b.sha(directory / "runtime.json"))
    for key, value in expected.items():
        b.need(i.get(key) == value, "old/foreign fit identity: " + key)
    b.need(summary.get("passed") is True and summary.get("smoke") is False and summary.get("device") == "cuda"
           and summary.get("steps") == c["optimizer_steps"] == summary["counters"]["updates"]
           and 0 <= summary["counters"]["skips"] <= c["optimizer_steps"] // 100
           and summary.get("runtime_pairing") is True
           and 0 < summary.get("peak_reserved_bytes", 0) <= b.MEMORY_GATE_BYTES, "partial/unpaired fit forbidden")
    b.need(b.read(directory / "heartbeat.json").get("status") == "complete", "fit incomplete")
    b.require_runtime(b.read(directory / "runtime.json"))
    for name, rows, ids in (("val", 864, c["val"]), ("gold", 58, c["gold"])):
        r = b.read(directory / (name + ".receipt.json"))
        b.need(r.get("k") == 36 and r.get("tta") is True and r.get("engineering_only") is False
               and r.get("rows") == rows == len(ids) and r.get("uid_sha256") == b.uid_hash(ids)
               and r.get("csv_sha256") == b.sha(directory / (name + ".csv"))
               and r.get("checkpoint_sha256") == summary.get("checkpoint_sha256") == b.sha(directory / "final.pt"),
               "prediction binding differs: " + name)
    return summary


def validate_panel(c, directory, seed):
    s = b.read(directory / "summary.json")
    r = b.read(directory / "gold.receipt.json")
    b.need(s.get("passed") is True and s.get("smoke") is False and s.get("device") == "cuda"
           and s.get("runtime_pairing") is True and s.get("checkpoint_sha256") == b.CONTROL_PINS[seed]["checkpoint"]
           and s.get("config_sha256") == b.sha(b.config_path(seed)) and s.get("rows") == 58, "control panel differs")
    b.need(r.get("csv_sha256") == b.sha(directory / "gold.csv") == s.get("csv_sha256") and r.get("k") == 36
           and r.get("tta") is True and r.get("uid_sha256") == b.uid_hash(c["gold"])
           and r.get("checkpoint_sha256") == b.CONTROL_PINS[seed]["checkpoint"], "control gold binding differs")
    b.require_runtime(b.read(directory / "runtime.json"))
    return s


def run(review, engineering=False, out=None):
    registration = b.registration_binding()
    arms, controls, gold, bindings = {}, {}, dict(arm={}, control={}), {}
    configs = {}
    for seed in (42, 43):
        cp = b.config_path(seed)
        c = b.validate(cp) if engineering else b.require_fit(cp, review)
        configs[seed] = c
        control_dir = b.CONTROL_WORK / "fits" / b.control_member(seed)
        controls[str(seed)] = load_csv(control_dir / "val.csv", c["val"], b.CONTROL_PINS[seed]["val_csv"])
        bindings[b.control_member(seed)] = dict(val_csv=b.CONTROL_PINS[seed]["val_csv"],
                                                checkpoint=b.CONTROL_PINS[seed]["checkpoint"])
        if engineering:
            rng = np.random.default_rng(seed)
            arms[str(seed)] = np.clip(controls[str(seed)] + rng.normal(0, .01, controls[str(seed)].shape), 0, 1)
            gold["control"][str(seed)] = rng.uniform(.05, .95, (len(c["gold"]), 12))
            gold["arm"][str(seed)] = gold["control"][str(seed)]
            continue
        fit_dir = b.WORK / "fits" / b.member(seed)
        summary = validate_fit(c, fit_dir, seed)
        arms[str(seed)] = load_csv(fit_dir / "val.csv", c["val"])
        gold["arm"][str(seed)] = load_csv(fit_dir / "gold.csv", c["gold"])
        panel_dir = b.WORK / "panel" / b.control_member(seed)
        validate_panel(c, panel_dir, seed)
        gold["control"][str(seed)] = load_csv(panel_dir / "gold.csv", c["gold"])
        bindings[b.member(seed)] = dict(val_csv=b.sha(fit_dir / "val.csv"), gold_csv=b.sha(fit_dir / "gold.csv"),
                                        checkpoint=summary["checkpoint_sha256"])
        bindings["panel-" + b.control_member(seed)] = dict(gold_csv=b.sha(panel_dir / "gold.csv"))
    b.need(configs[42]["val"] == configs[43]["val"] and configs[42]["gold"] == configs[43]["gold"], "seed rosters differ")
    gold["ids"] = configs[42]["gold"]
    ctx = lens._base.make_context(configs[42]["val"], b.ROOT)
    partners = partner_predictions(configs[42]["val"])
    result = evaluate(ctx, arms, controls, gold, partners, engineering)
    result.update(prediction_bindings=bindings, ruler_input_hashes=ctx.bindings, registration=registration,
                  l6_partner_pins={f: {s: d for s, (_, d) in v.items()} for f, v in b.L6_PARTNERS.items()})
    path = Path(out) if engineering and out is not None else b.WORK / ("readout-engineering.json" if engineering else "readout.json")
    if engineering:
        from tools.hb150_io_v1 import atomic_json
        atomic_json(path, result)
    else:
        b.need(not path.exists(), "readout.json is create-only")
        b.immutable(path, result)
    return dict(status=result["registered_gate"]["status"], verdict=result["verdict"],
                decision_label=result["decision_label"], readout=str(path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path)
    p.add_argument("--engineering", action="store_true",
                   help="data-path check: synthetic multiview predictions around the real control CSVs; never evidence")
    a = p.parse_args(argv)
    if not a.engineering and a.review is None:
        p.error("--review required for the scientific readout")
    print(json.dumps(run(a.review, a.engineering)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
