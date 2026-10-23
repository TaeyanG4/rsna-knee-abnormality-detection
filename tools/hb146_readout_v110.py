"""H-B-146 1.1.0 EXPOSED_EXPLORATORY six-fit seven-lens screen.

Changelog 1.1.0: explicit exposed flag, global/fill paired controls, two-branch
GO registration, frozen arm selection and raw f0 teacher AUC context. v1 kept.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb146_pl_labels_v110 as pl
from tools import multi_lens_readout_l4_v2 as lens
import numpy as np


def gate(report_delta, report_weak_delta, gold_delta, oai_delta, oai_eligible, engineering=False):
    """Frozen two-branch screen; three SAME rulers must satisfy both metrics."""
    macro, weak = np.asarray(report_delta, dtype=float), np.asarray(report_weak_delta, dtype=float)
    report_valid = macro.shape == weak.shape == (4,) and np.isfinite(macro).all() and np.isfinite(weak).all()
    qualifying = (macro >= .002) & (weak >= .002) if report_valid else np.zeros(4, dtype=bool)
    report_pass = bool(report_valid and qualifying.sum() >= 3)
    image_valid = (macro.shape == (4,) and np.isfinite(macro).all() and
                   gold_delta is not None and oai_delta is not None and
                   np.isfinite([gold_delta, oai_delta]).all() and bool(oai_eligible))
    image_pass = bool(image_valid and gold_delta >= .003 and oai_delta >= .003 and np.all(macro >= -.003))
    passed = (report_pass or image_pass) and not engineering
    return {"GO": passed, "status": "ENGINEERING_ONLY" if engineering else "GO" if passed else "NO_GO",
            "report_branch": report_pass, "image_branch": image_pass,
            "qualifying_report_rulers": [r for r, ok in zip(lens.RULERS, qualifying) if ok],
            "report_macro_seed_mean_deltas": lens._base.clean(dict(zip(lens.RULERS, macro))),
            "report_weak4_seed_mean_deltas": lens._base.clean(dict(zip(lens.RULERS, weak))),
            "selection_macro_delta": float(macro.mean()) if macro.shape == (4,) and np.isfinite(macro).all() else None,
            "gold_seed_mean_delta": lens._base.clean(gold_delta), "oai_weak4_seed_mean_delta": lens._base.clean(oai_delta),
            "oai_eligible": bool(oai_eligible), "gold_role": "descriptive/exposed image-read lens",
            "status_scope": "EXPOSED_EXPLORATORY"}


def choose_arm(gates, engineering=False):
    """Select passing arm by mean four-ruler macro delta; exact tie goes global."""
    passing = [side for side in ("global", "fill") if gates[side]["GO"]]
    chosen = max(passing, key=lambda side: (gates[side]["selection_macro_delta"], side == "global")) if passing else None
    if engineering:
        chosen = None
    return {"GO": chosen is not None, "status": "ENGINEERING_ONLY" if engineering else "GO" if chosen else "NO_GO",
            "selected_arm": chosen, "selection_rule": "highest mean four-report-ruler seed-mean macro delta among passing arms; tie global",
            "status_scope": "EXPOSED_EXPLORATORY"}


def prediction(work, side, seed, name, ids):
    out = work / "fits" / f"{side}-s{seed}"
    path = out / (name + ".csv")
    r = pl.read(path.with_suffix(".receipt.json"))
    summary = pl.read(out / "summary.json")
    pl.need(not r["engineering_only"] and not summary["smoke"] and r["identity"] == summary["identity"]
            and r["identity"]["arm"] == side and r["identity"]["seed"] == seed and r["identity"]["epochs"] == 20
            and r["identity"]["prepare_sha256"] == pl.sha(work / "prepare.json")
            and r["identity"]["registration_sha256"] == pl.sha(work / "registration.json")
            and r["csv_sha256"] == pl.sha(path) and r["checkpoint_sha256"] == summary["checkpoint_sha256"]
            == pl.sha(out / "final.pt") and r["k"] == 36 and r["tta"] is True, "prediction binding differs")
    return pl.aligned(pl.table(path), ids, exact=True), r


def run(work, engineering=False, ancestry=None, exposed_exploratory=False):
    from tools import hb146_screen_v110 as screen
    from tools import oai_moaks_panel as panel
    screen.prepared(work, engineering)
    if not engineering:
        _, _, provenance = pl.validate_labels(work)
        pl.require_exposed(provenance, exposed_exploratory)
    s = pl.registration(work)
    ctx = lens._base.make_context(s["val"], ROOT)
    gold = pl.table(pl.GOLD, allow_missing=True)
    ids_oai, admission = screen.panel_admit(work, ancestry, engineering, exposed_exploratory)
    groups = pl.read(ROOT / lens._base.COMPONENTS)
    membership = {u for c in groups for u in c["members"]}
    # The report-screen component contract omits Gold. Keep known groups and
    # explicitly use singleton study bootstrap for uncovered Gold, descriptive only.
    missing_gold = set(s["gold"]) - membership
    gold_groups = groups + [{"component_id": "gold-unmapped-" + u, "members": [u]} for u in sorted(missing_gold)]
    image_inputs = {
        "gold58": {"ids": s["gold"], "labels": pl.aligned(gold, s["gold"]), "components": gold_groups,
                   "eligible": False, "component_policy": f"known components plus {len(missing_gold)} unmapped singleton studies; descriptive CI assumes their independence"},
        "oai_moaks": {"ids": ids_oai, "labels": panel.outcomes(ids_oai),
                      "components": [{"component_id": u.split('_')[1], "members": [u]} for u in ids_oai],
                      "eligible": admission["eligible"], "component_policy": "one registered baseline knee per participant"}}
    rosters = {"val": s["val"], "gold": s["gold"], "oai": ids_oai}
    predictions, bindings, runtimes = {}, {}, []
    for name, ids in rosters.items():
        predictions[name] = {side: {} for side in ("control", "global", "fill")}
        for side in ("control", "global", "fill"):
            for seed in (42, 43):
                if engineering:
                    # Data-path test only: deterministic fake probabilities, identical pairs.
                    values = np.random.default_rng(seed).uniform(.1, .9, (len(ids), 12)).astype(np.float32)
                else:
                    values, r = prediction(work, side, seed, name, ids)
                    bindings[f"{name}-{side}-{seed}"] = r
                    runtimes.append(r["identity"]["runtime"])
                    if name == "oai":
                        a = pl.read(work / "fits" / f"{side}-s{seed}" / "oai-admission.json")
                        pl.need(a["eligible"] is True and a["binding_sha256"] == admission["binding_sha256"]
                                and a["csv_sha256"] == r["csv_sha256"] and
                                a["cache_sha256"] == pl.sha(ROOT / panel.CACHES["64"] / "manifest.json"), "panel admission differs")
                predictions[name][side][str(seed)] = values
    pl.need(not runtimes or all(r == runtimes[0] for r in runtimes), "paired execution runtime differs")
    results = {}
    for side in ("global", "fill"):
        for name, key in (("gold58", "gold"), ("oai_moaks", "oai")):
            entry = image_inputs[name]
            entry["mask"] = np.isfinite(entry["labels"])
            entry["arm"] = np.mean(list(predictions[key][side].values()), axis=0)
            entry["control"] = np.mean(list(predictions[key]["control"].values()), axis=0)
        arms, controls = predictions["val"][side], predictions["val"]["control"]
        result = lens.evaluate(ctx, arms, controls, label_change=True, image_reads=image_inputs)
        result["lenses"]["L6"] = lens._base.unavailable(
            "No registered compatible fixed partner; no bundle claim.")
        sd = ctx.delta(np.mean(list(arms.values()), axis=0), np.mean(list(controls.values()), axis=0))
        result["report_seed_mean_macros"] = {r: lens._base.stat(sd[:, k].mean(1), -.003)
                                             for k, r in enumerate(lens.RULERS)}
        images = result["lenses"]["L7"]["entries"]
        result["registered_go"] = gate(sd[0].mean(1), sd[0][:, lens._base.WEAK].mean(1),
            images["gold58"]["value"], images["oai_moaks"]["value"], admission["eligible"], engineering)
        results[side] = result
    if engineering:
        teacher = np.random.default_rng(0).uniform(.1, .9, (len(s["val"]), 12))
        teacher_binding = {"engineering_only": True}
    else:
        teacher_path = Path(provenance["teacher_csv"])
        teacher = pl.aligned(pl.table(teacher_path), s["val"])
        teacher_binding = {"path": str(teacher_path), "sha256": pl.sha(teacher_path)}
    teacher_auc = ctx.score(teacher)[0]
    selection = choose_arm({side: result["registered_go"] for side, result in results.items()}, engineering)
    result = {"version": pl.VERSION, "status": "EXPOSED_EXPLORATORY",
              "engineering_only": engineering, "arms": results, "registered_go": selection,
              "prediction_bindings": bindings, "ruler_input_hashes": ctx.bindings,
              "admission": admission, "teacher_f0_binding": teacher_binding,
              "teacher_f0_validation_auc": {r: {"macro": lens._base.clean(teacher_auc[k].mean()),
                  "weak4": lens._base.clean(teacher_auc[k, lens._base.WEAK].mean()),
                  "targets": lens._base.clean(dict(zip(lens.TARGETS, teacher_auc[k])))}
                  for k, r in enumerate(lens.RULERS)},
              "limitations": provenance.get("limitations") if not engineering else "engineering fake teacher",
              "notes": ["EXPOSED_EXPLORATORY; never a clean confirmatory claim.",
                  "Gate uses AUC of arithmetic seed-mean predictions; not mean seed AUC.",
                  "Gold58 is descriptive/exposed; OAI teacher ancestry evidence partial: ImageNet init, fastMRI+, SKM-TEA segmenter.",
                  "300 component bootstrap draws; no corrected multiplicity claim.",
                  "Public LB of a production candidate is the decisive arbiter."]}
    path = work / ("readout-engineering.json" if engineering else "readout.json")
    pl.write(path, result)
    return {"rows": len(s["val"]), "gold": len(s["gold"]), "oai": len(ids_oai), "engineering_only": engineering,
            "status": result["registered_go"]["status"], "readout": str(path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=pl.WORK)
    ap.add_argument("--engineering-replay", action="store_true")
    ap.add_argument("--ancestry", type=Path)
    ap.add_argument("--exposed-exploratory", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(run(a.work, a.engineering_replay, a.ancestry, a.exposed_exploratory)))


if __name__ == "__main__":
    main()
