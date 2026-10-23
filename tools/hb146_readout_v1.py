"""H-B-146 1.0.0 seven-lens readout and separately preregistered PL GO gate.

GO uses AUC of seed-mean predictions, not mean seed AUC. Report rulers are harm
guards; they reward the L2 control by construction. Gold58 remains descriptive.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb146_pl_labels_v1 as pl
from tools import multi_lens_readout_l4_v2 as lens
import numpy as np


def gate(report_delta, gold_delta, oai_delta, oai_eligible, engineering=False):
    report_delta = np.asarray(report_delta)
    valid = report_delta.shape == (4,) and np.isfinite(report_delta).all()
    valid = valid and gold_delta is not None and oai_delta is not None
    valid = valid and np.isfinite([gold_delta, oai_delta]).all()
    passed = bool(valid and np.all(report_delta >= -.003) and gold_delta >= 0 and oai_delta >= 0)
    return {"GO": passed and bool(oai_eligible) and not engineering,
            "status": "ENGINEERING_ONLY" if engineering else ("HOLD" if not valid or not oai_eligible else "GO" if passed else "NO_GO"),
            "report_macro_seed_mean_deltas": lens._base.clean(dict(zip(lens.RULERS, report_delta))),
            "gold_seed_mean_delta": lens._base.clean(gold_delta), "oai_weak4_seed_mean_delta": lens._base.clean(oai_delta),
            "oai_eligible": bool(oai_eligible), "gold_role": "descriptive/exposed; required guard, never sole selection"}


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


def run(work, engineering=False, ancestry=None):
    from tools import hb146_screen_v1 as screen
    from tools import oai_moaks_panel as panel
    screen.prepared(work, engineering)
    if not engineering:
        _, _, provenance = pl.validate_labels(work)
        pl.require_clean_teacher(provenance)
    s = pl.registration(work)
    ctx = lens._base.make_context(s["val"], ROOT)
    gold = pl.table(pl.GOLD, allow_missing=True)
    ids_oai, admission = screen.panel_admit(work, ancestry, engineering)
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
        predictions[name] = {side: {} for side in ("pl", "control")}
        for side in ("pl", "control"):
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
                        pl.need(a["eligible"] is True and a["ancestry_sha256"] == admission["ancestry_sha256"]
                                and a["csv_sha256"] == r["csv_sha256"] and
                                a["cache_sha256"] == pl.sha(ROOT / panel.CACHES["64"] / "manifest.json"), "panel admission differs")
                predictions[name][side][str(seed)] = values
    pl.need(not runtimes or all(r == runtimes[0] for r in runtimes), "paired execution runtime differs")
    for name, key in (("gold58", "gold"), ("oai_moaks", "oai")):
        entry = image_inputs[name]
        entry["mask"] = np.isfinite(entry["labels"])
        entry["arm"] = np.mean(list(predictions[key]["pl"].values()), axis=0)
        entry["control"] = np.mean(list(predictions[key]["control"].values()), axis=0)
    arms, controls = predictions["val"]["pl"], predictions["val"]["control"]
    result = lens.evaluate(ctx, arms, controls, label_change=True, image_reads=image_inputs)
    result["lenses"]["L6"] = lens._base.unavailable(
        "No registered label-clean fixed partner on hash-fold-0-minus-duplicates; old EXP099 OOF is incompatible. No bundle claim.")
    sd = ctx.delta(np.mean(list(arms.values()), axis=0), np.mean(list(controls.values()), axis=0))
    result["report_seed_mean_macros"] = {r: lens._base.stat(sd[:, k].mean(1), -.003)
                                         for k, r in enumerate(lens.RULERS)}
    images = result["lenses"]["L7"]["entries"]
    result["registered_go"] = gate(sd[0].mean(1), images["gold58"]["value"], images["oai_moaks"]["value"],
                                   admission["eligible"], engineering)
    result.update(engineering_only=engineering, prediction_bindings=bindings, ruler_input_hashes=ctx.bindings,
                  notes=["L1-L4 report rulers favor L2 by construction; interpret as harm guards.",
                         "L5 inherits seven-lens thresholds; registered GO separately uses arithmetic seed-mean predictions.",
                         "L7 Gold58 is exposed/descriptive; OAI admission includes teacher ancestry.",
                         "300 component bootstrap draws; no claim of corrected multiplicity."])
    path = work / ("readout-engineering.json" if engineering else "readout.json")
    pl.write(path, result)
    return {"rows": len(s["val"]), "gold": len(s["gold"]), "oai": len(ids_oai), "engineering_only": engineering,
            "status": result["registered_go"]["status"], "readout": str(path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=pl.WORK)
    ap.add_argument("--engineering-replay", action="store_true")
    ap.add_argument("--ancestry", type=Path)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.work, a.engineering_replay, a.ancestry)))


if __name__ == "__main__":
    main()
