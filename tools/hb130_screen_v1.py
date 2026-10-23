"""H-B-130 fold-0 screen v1.0.0 (2026-10-13, claude-hb130): Gold58 inference + full readout.

The arm (training.arms.hb130_label_share_arm_v1:ARM) is configured, verified, trained and
panel-inferred by the generic shared-control card (tools/mil_screen_arm_build_v1.py,
.local/run-scripts/mil-screen-arm-v1.ps1). This tool adds what that card lacks for a LABEL arm:

  gold58-infer --work W --seed S --side arm|control [--engineering-smoke]
      Gold58 (58 official studies, never trained on) on the TRAINING corpus with the same
      hooked inference path as the held-out readout (evaluate_with, MILWindows K36 eval
      centers, batch 4, flip mean, strict runtime). CUDA via tools/gpu_job.py; CPU = 2-study
      engineering smoke only.
  readout --work W [--engineering-smoke]
      Seven lenses (multi_lens_readout_l4_v2, corrected L4) with label_change=True (the aquila
      ruler is self-referential for this arm), L5 two seeds, L6 fold-0 Q7+S surrogate, L7 OAI
      MOAKS panel (descriptive) + Gold58 (exposed, descriptive), and the registered H-B-130
      weak-five rule block (Gold58 + OAI weak-five per seed, other-seven harm guard).
  waiting --work W

Every artifact is create-only. Never prints label rows; no training, admission or release.
Changelog 1.0.0: first version (PROTOCOL .local/scratch/claude-hb130/PROTOCOL-v1.md).
"""
import argparse
import csv
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np

from tools import mil_screen_arm_build_v1 as build

hooks, life, need, sha, read = build.hooks, build.life, build.need, build.sha, build.read
control_build = build.control_build
ROOT, CORPUS, WORKERS = build.ROOT, build.CORPUS, build.WORKERS
VERSION = "1.0.0"
ARM_REFERENCE = "training.arms.hb130_label_share_arm_v1:ARM"
GOLD = "storage/gold/gold_labels.csv"
GOLD_SHA = "990dd72929224e76391c93bd24e861dc75eaa69bc6f14ed71463abc9911e3b66"
GOLD_PATH = "hooked evaluate_with: MILWindows K36 eval centers on RAPTOR64_336_C140_S02, batch4, flip-mean, strict runtime"
GOLD_REASON = ("EXPOSED_DESCRIPTIVE: Gold58 (58 studies, official labels) has been read by many earlier "
               "screens; descriptive only, never eligible, never a gate on its own.")
WEAK_FIVE = ("Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis")
FIVE_IX = tuple(hooks.TARGETS.index(t) for t in WEAK_FIVE)
OTHER_IX = tuple(i for i in range(12) if i not in FIVE_IX)
RULE = dict(weak_five=list(WEAK_FIVE), other_seven=[hooks.TARGETS[i] for i in OTHER_IX],
            combined="per seed: (Gold58 weak-five mean delta + OAI weak-five mean delta) / 2",
            go="combined > 0 in both seeds AND harm guard",
            harm_guard="mean over seed deltas of other-seven macro mean(JEV, consensus) AUC delta >= -0.001",
            status="mechanical flag only; holistic decision belongs to B")


def gold_roster():
    need(sha(ROOT / GOLD) == GOLD_SHA, "pinned Gold58 file changed")
    with (ROOT / GOLD).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [hooks.UID, *hooks.TARGETS], "Gold58 columns differ")
        rows = list(reader)
    ids = sorted(r[hooks.UID] for r in rows)
    need(len(ids) == 58 == len(set(ids)), "Gold58 roster differs")
    return ids, rows


def gold_outcomes(ids):
    _, rows = gold_roster()
    by = {r[hooks.UID]: r for r in rows}
    y = np.asarray([[float(by[u][t]) for t in hooks.TARGETS] for u in ids], np.float64)
    need(np.isfinite(y).all() and np.isin(y, (0, 1)).all(), "Gold58 outcomes must be complete binary")
    return y


def gold_paths(work, seed, side, smoke):
    folder = Path(work) / ("gold58-smoke" if smoke else "gold58-hooked-v1")
    path = folder / f"{side}-s{seed}.csv"
    return path, Path(str(path) + ".json")


def check_arm(record):
    need(record["arm"] == ARM_REFERENCE, "work dir is not the H-B-130 arm")


def side_inputs(work, record, seed, side, smoke):
    from tools import hb122_screen_v1 as screen
    need(side in ("arm", "control"), "side differs")
    if side == "arm":
        return build.arm_snapshot(work, record, seed, smoke), build.spec_for(record, seed), build.import_arm(record["arm"])
    return screen.snapshot("control", seed, smoke), control_build.spec_for("control", seed), hooks.IDENTITY


def gold58_infer(work, seed, side, smoke=False):
    """Same hooked path for arm and shared-control checkpoints; Gold58 never trained on."""
    import torch
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    work, record, _ = build.state(work)
    check_arm(record)
    path, receipt_path = gold_paths(work, seed, side, smoke)
    need(not path.exists() and not receipt_path.exists(), "Gold58 output exists; authenticate or version it")
    snap, spec, use_arm = side_inputs(work, record, seed, side, smoke)
    ids, _ = gold_roster()
    roster = read(ROOT / spec.document["config"]["roots"]["ordered"])
    need(not set(ids) & set(roster["train"] + roster["val"]), "Gold58 overlaps the fold-0 roster")
    use_ids = ids[:2] if smoke else ids
    device, amp = ("cpu", "none") if smoke else ("cuda", "fp16")
    hooks.configure_runtime(amp, device, policy="strict")
    observed = hooks.observation(device)
    if not smoke:
        life.check_target_observation(spec, observed)
    c = spec.document["config"]
    need(sha(CORPUS / "manifest.json") == spec.inventory[c["roots"]["manifest"]], "mounted corpus differs")
    corpus = hooks.common.Corpus(CORPUS)
    hooks.verify_shards(corpus, use_ids)
    checkpoint = torch.load(ROOT / snap["checkpoint"], map_location="cpu", weights_only=False)
    need(checkpoint["provenance"]["spec_sha256"] == spec.sha256 and checkpoint["epoch"] == (0 if smoke else 19)
         and sha(ROOT / snap["checkpoint"]) == snap["bindings"][snap["checkpoint"]], "Gold58 checkpoint differs")
    model = use_arm.attach(hooks.build_model(c["model"], ROOT), spec, ROOT)
    hooks.strict_load(model, checkpoint["model"])
    del checkpoint
    model = model.to(device)
    zeros = {u: np.zeros(12, np.float32) for u in use_ids}
    ds = use_arm.dataset(hooks.MILWindows(CORPUS, use_ids, zeros, 36, False, spec.document["seed"]),
                         spec, ROOT, train=False)
    loader = hooks.DataLoader(ds, batch_size=4, shuffle=False, num_workers=min(WORKERS, 2),
                              collate_fn=use_arm.collate_fn, worker_init_fn=hooks.worker_init)
    start = time.monotonic()
    values, ordered, _ = hooks.evaluate_with(use_arm, model, loader, device, amp, corpus, tta=True)
    need(ordered == use_ids and values.shape == (len(use_ids), 12) and np.isfinite(values).all()
         and ((values >= 0) & (values <= 1)).all(), "invalid Gold58 forward")
    life.create_new(path, life.prediction_csv_bytes(use_ids, values, hooks.TARGETS, id_column=hooks.UID))
    receipt = dict(version=VERSION, side=side, seed=seed, arm=record["arm"] if side == "arm" else "identity",
                   snapshot=snap, path=GOLD_PATH, corpus_manifest_sha256=sha(CORPUS / "manifest.json"),
                   gold_sha256=GOLD_SHA, predictions_sha256=sha(path), rows=len(use_ids),
                   ids_sha256=hooks.uid_hash(use_ids), engineering_smoke=smoke, device=device,
                   observation=observed, seconds=time.monotonic() - start, sources=source_pins())
    build.same_or_new(receipt_path, receipt)
    need(not smoke or not torch.cuda.is_initialized(), "CPU smoke initialized CUDA")
    return dict(status="GOLD58_INFERRED", side=side, seed=seed, rows=len(use_ids), seconds=receipt["seconds"])


def gold_prediction(work, record, seed, side, smoke, snap):
    from tools import hb122_screen_v1 as screen
    path, receipt_path = gold_paths(work, seed, side, smoke)
    receipt = read(receipt_path)
    ids, _ = gold_roster()
    ids = ids[:2] if smoke else ids
    need(receipt["version"] == VERSION and receipt["side"] == side and receipt["seed"] == seed
         and receipt["snapshot"] == snap and receipt["path"] == GOLD_PATH and receipt["gold_sha256"] == GOLD_SHA
         and receipt["predictions_sha256"] == sha(path) and receipt["rows"] == len(ids)
         and receipt["ids_sha256"] == hooks.uid_hash(ids) and receipt["engineering_smoke"] is smoke
         and receipt["device"] == ("cpu" if smoke else "cuda")
         and receipt["arm"] == (record["arm"] if side == "arm" else "identity"), "Gold58 binding differs")
    return ids, screen.load_predictions(path, ids), receipt


def gold_item(ids, arms, controls, y):
    """Seed-mean predictions, study singletons (no component contract for Gold58)."""
    need(set(arms) == set(controls) == {"42", "43"} and len(ids) == len(set(ids)), "Gold58 seeds/IDs differ")
    y = np.asarray(y, np.float64)
    need(y.shape == (len(ids), 12) and np.isfinite(y).all() and np.isin(y, (0, 1)).all(), "invalid Gold58 outcomes")
    for array in (*arms.values(), *controls.values()):
        need(array.shape == y.shape and np.isfinite(array).all() and ((array >= 0) & (array <= 1)).all(),
             "invalid Gold58 probabilities")
    return dict(ids=ids, arm=np.stack(list(arms.values())).mean(0, dtype=np.float32),
                control=np.stack(list(controls.values())).mean(0, dtype=np.float32),
                labels=y, mask=np.isfinite(y), eligible=False,
                components=[dict(component_id=u, members=[u]) for u in ids],
                component_policy="Gold58 study singletons: no validation component contract; exposed, diagnostic only")


def evaluate(ctx, arms, controls, bundle, oai, gold=None, label_change=True):
    """hb122_screen_v1.evaluate with label_change and an optional Gold58 image read.

    With gold=None and label_change=False this equals hb122_screen_v1.evaluate (tested).
    """
    from tools import hb122_screen_v1 as screen
    ml, corrected = screen.ml, screen.corrected
    reads = {"oai_moaks": oai} if gold is None else {"oai_moaks": oai, "gold58": gold}
    result = corrected.evaluate(ctx, arms, controls, label_change=label_change, bundle=bundle, image_reads=reads)
    w = ml.weights(oai["ids"], oai["components"])
    a = ml.auc_draws(oai["arm"], oai["labels"], oai["mask"], w)
    c = ml.auc_draws(oai["control"], oai["labels"], oai["mask"], w)
    ix = list(screen.PANEL_TARGETS)
    overall = ml.stat((a[:, ix]-c[:, ix]).mean(1), eligible=False, reason=screen.ANCESTRY)
    overall.update(targets=[hooks.TARGETS[t] for t in ix], arm_auc=ml.clean(a[0, ix].mean()),
                   control_auc=ml.clean(c[0, ix].mean()), metric="mapped-nine macro AUC")
    entry = result["lenses"]["L7"]["entries"]["oai_moaks"]
    for statistic in entry["targets"].values():
        statistic.update(eligible=False, reason=screen.ANCESTRY)
    entry.update(overall_macro=overall, ancestry_reason=screen.ANCESTRY,
                 unavailable_targets={hooks.TARGETS[t]: "No registered MOAKS mapping" for t in (1, 10, 11)})
    result["lenses"]["L6"]["final_proxy"] = ml.unavailable(screen.FINAL_PROXY)
    result.update(scope="FOLD0_SCREEN_ONLY", release_scope="none", not_release_authorization=True,
        bootstrap=dict(draws=ml.DRAWS, seed=ml.SEED, interval="paired percentile 95%",
                       limitations="Conditional on fitted models; excludes fit/selection/private-shift uncertainty"),
        warnings=[screen.ANCESTRY, screen.FINAL_PROXY, "Holistic decision belongs to B; no automatic release.",
                  "Report-ruler supervision is exposed; L7 cannot be treated as independent.",
                  "Negative/flat two-seed evidence is retained; coefficient is never tuned."])
    if gold is not None:
        g = result["lenses"]["L7"]["entries"]["gold58"]
        for statistic in g["targets"].values():
            statistic.update(eligible=False, reason=GOLD_REASON)
        g.update(eligible=False, reason=GOLD_REASON, metric="all-12 macro AUC delta of seed-mean predictions")
    if label_change:
        result["warnings"].append("Label arm: the aquila ruler is an input of the arm labels (self-referential); "
                                  "report rulers are harm guards/telemetry for the weak-five, not evidence.")
    return ml.clean(result)


def _paired(ml, item_ids, components, arm, control, labels, mask):
    w = ml.weights(item_ids, components)
    return ml.auc_draws(arm, labels, mask, w) - ml.auc_draws(control, labels, mask, w)


def weak_five_rule(ctx, arms, controls, oai, oai_seeds, gold, gold_seeds):
    """Registered H-B-130 block: per-seed Gold58/OAI weak-five deltas and other-seven guard."""
    from tools import hb122_screen_v1 as screen
    ml = screen.ml
    five, other = list(FIVE_IX), list(OTHER_IX)
    per_seed, draws = {}, {"gold": [], "oai": [], "combined": [], "other": []}
    for seed in arms:
        g = _paired(ml, gold["ids"], gold["components"], gold_seeds["arm"][seed], gold_seeds["control"][seed],
                    gold["labels"], gold["mask"])
        o = _paired(ml, oai["ids"], oai["components"], oai_seeds["arm"][seed], oai_seeds["control"][seed],
                    oai["labels"], oai["mask"])
        r = ctx.delta(arms[seed], controls[seed])
        gv, ov = g[:, five].mean(1), o[:, five].mean(1)
        values = dict(gold=gv, oai=ov, combined=(gv + ov) / 2, other=r[:, :2][:, :, other].mean((1, 2)))
        for key, value in values.items():
            draws[key].append(value)
        per_seed[seed] = dict(
            gold58_weak_five=ml.stat(gv, 0, eligible=False, reason=GOLD_REASON),
            oai_weak_five=ml.stat(ov, 0, eligible=False, reason=screen.ANCESTRY),
            combined=ml.stat(values["combined"], 0, eligible=False, reason="registered H-B-130 rule"),
            other_seven_macro_jev_consensus=ml.stat(values["other"], -.001),
            gold58_targets={hooks.TARGETS[t]: ml.clean(g[0, t]) for t in five},
            oai_targets={hooks.TARGETS[t]: ml.clean(o[0, t]) for t in five},
            report_rulers_weak_five={ruler: {hooks.TARGETS[t]: ml.clean(r[0, i, t]) for t in five}
                                     for i, ruler in enumerate(ml.RULERS)})
    mean = {k: np.mean(v, axis=0) for k, v in draws.items()}
    combined_pos = all(per_seed[s]["combined"]["value"] is not None and per_seed[s]["combined"]["value"] > 0
                       for s in per_seed)
    components_pos = all(per_seed[s][k]["value"] is not None and per_seed[s][k]["value"] > 0
                         for s in per_seed for k in ("gold58_weak_five", "oai_weak_five"))
    guard = bool(np.isfinite(mean["other"][0]) and mean["other"][0] >= -.001 - 1e-12)
    return ml.clean(dict(
        rule=RULE, per_seed=per_seed,
        seed_mean_of_deltas=dict(gold58_weak_five=ml.stat(mean["gold"], 0, eligible=False, reason=GOLD_REASON),
                                 oai_weak_five=ml.stat(mean["oai"], 0, eligible=False, reason=screen.ANCESTRY),
                                 combined=ml.stat(mean["combined"], 0, eligible=False),
                                 other_seven_macro_jev_consensus=ml.stat(mean["other"], -.001)),
        combined_positive_both_seeds=combined_pos, both_components_positive_both_seeds=components_pos,
        harm_guard_other_seven=guard,
        registered_go_flag="GO_RULE_MET" if combined_pos and guard else "GO_RULE_NOT_MET",
        note="Mechanical registered flag; B decides holistically over all lenses."))


def source_pins():
    from tools import hb122_screen_v1 as screen
    names = ("tools/hb130_screen_v1.py", "tools/mil_screen_arm_build_v1.py", "training/arms/hb130_label_share_arm_v1.py",
             "training/arms/mil_screen_hooks_v1.py", GOLD)
    return {**screen.source_pins(), **{n: sha(ROOT / n) for n in names}}


def readout(work, smoke=False):
    from tools import hb122_screen_v1 as screen
    work, record, arm = build.state(work)
    check_arm(record)
    controls = {str(s): screen.snapshot("control", s, smoke) for s in (42, 43)}
    arms = {str(s): build.arm_snapshot(work, record, s, smoke) for s in (42, 43)}
    need(len({v["runtime_sha256"] for v in (*controls.values(), *arms.values())}) == 1, "fits ran on different runtimes")
    panels, golds, receipts = {"arm": {}, "control": {}}, {"arm": {}, "control": {}}, {}
    panel_ids = gold_ids = None
    for side, snaps in (("arm", arms), ("control", controls)):
        for seed, snap in snaps.items():
            ids, values, receipt = build.panel_prediction(work, record, int(seed), side, smoke, snap)
            need(panel_ids is None or ids == panel_ids, "panel paired roster differs")
            panel_ids, panels[side][seed] = ids, values
            gids, gvalues, greceipt = gold_prediction(work, record, int(seed), side, smoke, snap)
            need(gold_ids is None or gids == gold_ids, "Gold58 paired roster differs")
            gold_ids, golds[side][seed] = gids, gvalues
            if not smoke:
                spec = build.spec_for(record, int(seed)) if side == "arm" else control_build.spec_for("control", int(seed))
                life.check_target_observation(spec, receipt["observation"])
                life.check_target_observation(spec, greceipt["observation"])
            receipts[f"panel-{side}-s{seed}"] = sha(build.panel_paths(work, int(seed), side, smoke)[1])
            receipts[f"gold58-{side}-s{seed}"] = sha(gold_paths(work, int(seed), side, smoke)[1])
    from tools import oai_moaks_panel as p
    y = p.outcomes(panel_ids)
    from tools import hb105_aligned_oof_v1 as aligned
    raw, rosters, auth = aligned.authenticate()
    ids = rosters[0]["val"]
    arrays = {side: {seed: (raw[0]["S" + seed].copy() if smoke else
                            screen.load_predictions(ROOT / snap["prediction"], ids)) for seed, snap in snaps.items()}
              for side, snaps in (("arm", arms), ("control", controls))}
    arm_bundle, control_bundle = screen.fixed_bundle(raw, arrays["arm"], arrays["control"], aligned.variants)
    ctx = screen.ml.make_context(ids, ROOT)
    need(ctx.bindings == auth["ruler_bindings"], "ruler binding changed")
    bundle = dict(context=ctx, arm=arm_bundle, control=control_bundle, recipe=dict(scope="ALIGNED_OOF_SURROGATE_ONLY",
        change="only fold0 S42/S43 replaced by arm vs fresh control; folds1-4 fixed",
        arithmetic=aligned.POLICY["arithmetic"], final_proxy=screen.FINAL_PROXY))
    oai = screen.image_item(panel_ids, panels["arm"], panels["control"], y)
    gold = gold_item(gold_ids, golds["arm"], golds["control"], gold_outcomes(gold_ids))
    result = evaluate(ctx, arrays["arm"], arrays["control"], bundle, oai, gold, label_change=True)
    result["hb130_weak_five"] = weak_five_rule(ctx, arrays["arm"], arrays["control"], oai, panels, gold, golds)
    result.update(version=VERSION, arm=arm.name, label_change=True,
                  status="ENGINEERING_CONTROL_REPLAY" if smoke else "COMPLETE_FOLD0_SCREEN",
                  fitted_members=dict(arm=arms, control=controls), receipts=receipts, panel_path=build.PANEL_PATH,
                  gold58_path=GOLD_PATH, aligned_authentication=auth, ruler_bindings=ctx.bindings, rows=len(ids),
                  panel_rows=len(panel_ids), gold58_rows=len(gold_ids), engineering_smoke=smoke, sources=source_pins(),
                  replay_reason=("CPU fits cannot supply heldout OOF; deployed S used on both sides" if smoke else None))
    for snap in (*arms.values(), *controls.values()):
        for path, digest in snap["bindings"].items():
            need(sha(ROOT / path) == digest, "member changed during scoring")
    output = work / ("screen-hb130-cpu-smoke-v1.json" if smoke else "screen-hb130-v1.json")
    build.same_or_new(output, result)
    rule = result["hb130_weak_five"]
    return dict(status=result["status"], output=build.rel(output), rows=len(ids), panel_rows=len(panel_ids),
                gold58_rows=len(gold_ids), lenses=list(result["lenses"]), verdict=result.get("verdict"),
                registered_go_flag=rule["registered_go_flag"])


def waiting(work):
    work, record, _ = build.state(work)
    check_arm(record)
    missing = list(build.waiting(work)["missing"])
    for seed in (42, 43):
        for side in ("arm", "control"):
            missing += [build.rel(p) for p in gold_paths(work, seed, side, False) if not p.is_file()]
    return dict(status="WAITING" if missing else "READY_FOR_READOUT", missing=missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("gold58-infer", "readout", "waiting"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--seed", type=int, choices=(42, 43))
    ap.add_argument("--side", choices=("arm", "control"))
    ap.add_argument("--engineering-smoke", action="store_true")
    a = ap.parse_args()
    print(f"hb130_screen v{VERSION}", flush=True)
    if a.command == "gold58-infer":
        need(a.seed is not None and a.side is not None, "--seed and --side required")
        result = gold58_infer(a.work, a.seed, a.side, a.engineering_smoke)
    elif a.command == "readout":
        result = readout(a.work, a.engineering_smoke)
    else:
        result = waiting(a.work)
    print(life.canonical_bytes(result).decode())


if __name__ == "__main__":
    main()
