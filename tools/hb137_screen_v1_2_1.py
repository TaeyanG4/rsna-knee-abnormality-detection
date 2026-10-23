"""Changelog v1.2.1: measured memory gate and idle/WDDM watchdog via safety module.
Versioned successor preserves hash-bound hb137_screen_v1.py and prior stages.

H-B-137 v1.2.1: frozen-parent authentication, three contrasts and B launch card.

No framework replacement. Uses shared builder/hooks and corrected seven-lens
scorer. configure/cpu-card/parity/readout are CPU; GPU commands are printed only.
budget-run is called INSIDE gpu_job by B, never by configure or CPU tests.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from training.arms import hb137_spatial_adapter_v1_2_1 as spatial
from tools import mil_screen_arm_build_v1_2_1 as build
from tools import hb122_screen_v1 as screen

hooks, life, need = build.hooks, build.life, build.need
ROOT, WORK = build.ROOT, build.ROOT / spatial.WORK
VERSION = "1.2.1"
KINDS = ("head", "adapter")
CONTRASTS = (("head", "parent"), ("adapter", "parent"), ("adapter", "head"))
BUDGET_SECONDS = 10500  # reserve five minutes of the 3h cap for wrapper overhead


def panel_contract():
    p, ids, route, pins = screen.panel_context()
    return dict(version=VERSION, ids_sha256=hooks.uid_hash(ids), rows=len(ids),
                routing=route, panel_pins=pins, sources=screen.source_pins(),
                cache_manifest_sha256=hooks.sha(ROOT / build.PANEL_CACHE / "manifest.json"))


def configure():
    build.same_or_new(ROOT / spatial.MANIFEST, dict(version=VERSION, parents={
        str(s): screen.snapshot("control", s) for s in (42, 43)}))
    build.same_or_new(WORK / "panel-v1.json", panel_contract())
    return {k: build.configure("training.arms.hb137_spatial_adapter_v1_2_1:" + k.upper(),
                              WORK / k, WORK / k / "configs") for k in KINDS}


def panel_binding(receipt, contract):
    need(receipt["sources"] == contract["sources"]
         and receipt["cache_manifest_sha256"] == contract["cache_manifest_sha256"],
         "panel sources/cache drift")


def protected_equal(arrays):
    need(set(arrays) == {"parent", *KINDS}, "missing three-way member")
    for k in arrays:
        need(set(arrays[k]) == {"42", "43"}, "missing paired seed")
        for s, a in arrays[k].items():
            need(a.ndim == 2 and a.shape[1] == 12 and a.shape == arrays["parent"][s].shape
                 and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), "invalid member array")
            need(np.array_equal(a[:, spatial.PROTECTED], arrays["parent"][s][:, spatial.PROTECTED]),
                 "protected-eight outputs changed")
    return True


def decision(results):
    """Point gates specified prospectively; CIs remain visible, no significance gate."""
    hp, ap, ah = (results[a + "-" + b] for a, b in CONTRASTS)
    val = lambda r: r["lenses"]["L5"]["weak_four"]["value"]
    finite = lambda v: v is not None and np.isfinite(v)
    gates = dict(
        head_control_not_harmed=finite(val(hp)) and val(hp) >= 0,
        adapter_parent_gain=finite(val(ap)) and val(ap) >= .001,
        adapter_beats_head=finite(val(ah)) and val(ah) > 0,
        adapter_seed_signs=all(r["lenses"]["L5"]["sign_agreement"]["L1"] for r in (ap, ah)),
        surrogate_no_harm=all(finite(r["lenses"]["L6"]["value"]) and r["lenses"]["L6"]["value"] >= 0
                              for r in (hp, ap, ah)),
        existing_harm_guards=all(r["harm_guard"]["PASS"] for r in (hp, ap, ah)))
    gates = {k: bool(v) for k, v in gates.items()}
    return dict(gates=gates, diagnostic="NECESSARY_CHECKS_PASS" if all(gates.values()) else "STOP",
                admission="HOLD", release_scope="none", final_blend_harm="UNRESOLVED",
                reason="True final-blend anchors unavailable; B review and untouched-fold confirmation required.")


def three_way(ctx, arrays, panels, ids_panel, outcomes, bundle_factory):
    protected_equal(arrays)
    protected_equal(panels)
    results = {}
    for a, b in CONTRASTS:
        aa, bb = bundle_factory(arrays[a], arrays[b])
        bundle = dict(context=ctx, arm=aa, control=bb,
                      recipe=dict(scope="ALIGNED_OOF_SURROGATE_ONLY", change="fixed fold0 S42/S43 replacement",
                                  final_proxy=screen.FINAL_PROXY))
        results[a + "-" + b] = screen.evaluate(ctx, arrays[a], arrays[b], bundle,
            screen.image_item(ids_panel, panels[a], panels[b], outcomes))
    return dict(version=VERSION, contrasts=results, decision=decision(results), protected_eight_exact=True,
                replication="Two matched head seeds; not independent backbone replications",
                not_release_authorization=True)


def readout():
    # Authenticate all fits BEFORE importing the historical CPU-only reader.
    states = {k: build.state(WORK / k) for k in KINDS}
    snaps = {"parent": {str(s): screen.snapshot("control", s) for s in (42, 43)}}
    need(snaps["parent"] == spatial.parents(), "parent lifecycle drift")
    for k, (w, record, _) in states.items():
        snaps[k] = {str(s): build.arm_snapshot(w, record, s) for s in (42, 43)}
    need(len({s["runtime_sha256"] for pair in snaps.values() for s in pair.values()}) == 1,
         "matched runtime differs")
    contract = hooks.read(WORK / "panel-v1.json")
    need(contract == panel_contract(), "registered panel drift")
    panels, receipts, ids_panel = {}, {}, None
    for k, pair in snaps.items():
        w, record, _ = states["head" if k == "parent" else k]
        side = "control" if k == "parent" else "arm"
        panels[k] = {}
        for s, snap in pair.items():
            ids, a, rec = build.panel_prediction(w, record, int(s), side, False, snap)
            panel_binding(rec, contract)
            spec = hooks.control_spec(int(s)) if k == "parent" else build.spec_for(record, int(s))
            life.check_target_observation(spec, rec["observation"])
            need(ids_panel is None or ids_panel == ids, "paired panel roster differs")
            ids_panel, panels[k][s] = ids, a
            receipts[build.rel(build.panel_paths(w, int(s), side, False)[1])] = hooks.sha(
                build.panel_paths(w, int(s), side, False)[1])
            receipts[build.rel(build.panel_paths(w, int(s), side, False)[0])] = rec["predictions_sha256"]
    from tools import oai_moaks_panel as p
    outcomes = p.outcomes(ids_panel)
    from tools import hb105_aligned_oof_v1 as aligned
    raw, rosters, auth = aligned.authenticate()
    ids = rosters[0]["val"]
    ctx = screen.ml.make_context(ids, ROOT)
    need(ctx.bindings == auth["ruler_bindings"], "ruler binding drift")
    arrays = {k: {s: screen.load_predictions(ROOT / snap["prediction"], ids) for s, snap in pair.items()}
              for k, pair in snaps.items()}
    result = three_way(ctx, arrays, panels, ids_panel, outcomes,
                       lambda a, b: screen.fixed_bundle(raw, a, b, aligned.variants))
    result.update(status="COMPLETE_FOLD0_SCREEN", fitted_members=snaps, panel_receipts=receipts,
                  aligned_authentication=auth, panel_contract=contract, sources={
                      p: hooks.sha(ROOT / p) for p in spatial.ADAPTER.sources})
    for pair in snaps.values():
        for snap in pair.values():
            for path, digest in snap["bindings"].items():
                need(hooks.sha(ROOT / path) == digest, "member changed during scoring")
    for path, digest in receipts.items():
        need(hooks.sha(ROOT / path) == digest, "panel receipt changed during scoring")
    build.same_or_new(WORK / "screen-three-contrasts-v1.json", result)
    return dict(status=result["status"], output=build.rel(WORK / "screen-three-contrasts-v1.json"),
                decision=result["decision"])


def head_checks(result):
    weak = result["lenses"]["L5"]["weak_four"]["value"]
    blend = result["lenses"]["L6"]["value"]
    return bool(weak is not None and np.isfinite(weak) and weak >= 0
                and blend is not None and np.isfinite(blend) and blend >= 0
                and result["harm_guard"]["PASS"])


def head_gate():
    """Stop before adapter fits if the matched head control is already damaged."""
    w, record, _ = build.state(WORK / "head")
    snaps = {"parent": {str(s): screen.snapshot("control", s) for s in (42, 43)},
             "head": {str(s): build.arm_snapshot(w, record, s) for s in (42, 43)}}
    need(snaps["parent"] == spatial.parents(), "parent lifecycle drift")
    need(len({s["runtime_sha256"] for pair in snaps.values() for s in pair.values()}) == 1,
         "matched runtime differs")
    from tools import hb105_aligned_oof_v1 as aligned
    raw, rosters, auth = aligned.authenticate()
    ids = rosters[0]["val"]
    ctx = screen.ml.make_context(ids, ROOT)
    need(ctx.bindings == auth["ruler_bindings"], "ruler binding drift")
    arrays = {k: {s: screen.load_predictions(ROOT / snap["prediction"], ids) for s, snap in pair.items()}
              for k, pair in snaps.items()}
    # Reuse the same protected-row validation; no adapter result is claimed here.
    protected_equal({**arrays, "adapter": arrays["head"]})
    a, b = screen.fixed_bundle(raw, arrays["head"], arrays["parent"], aligned.variants)
    result = screen.corrected.evaluate(ctx, arrays["head"], arrays["parent"], bundle=dict(
        context=ctx, arm=a, control=b, recipe=dict(scope="ALIGNED_OOF_SURROGATE_ONLY")))
    result["lenses"]["L6"]["final_proxy"] = screen.ml.unavailable(screen.FINAL_PROXY)
    result["lenses"]["L7"] = screen.ml.unavailable("Early futility check before six full panel runs; final readout requires L7.")
    result.update(status="HEAD_FUTILITY_PASS" if head_checks(result) else "STOP_FAILED_HEAD_CONTROL",
                  version=VERSION, fitted_members=snaps, admission="HOLD", release_scope="none",
                  bootstrap=dict(draws=screen.ml.DRAWS, seed=screen.ml.SEED, interval="paired percentile 95%"))
    for pair in snaps.values():
        for snap in pair.values():
            for path, digest in snap["bindings"].items():
                need(hooks.sha(ROOT / path) == digest, "member changed during scoring")
    build.same_or_new(WORK / "head-futility-v1.json", result)
    need(head_checks(result), "head-only control failed; stop before adapter fits")
    return dict(status=result["status"], output=build.rel(WORK / "head-futility-v1.json"), admission="HOLD")


def parity(seed):
    """Real heldout K36+flip: full frozen parent vs both one-update CPU refits."""
    import torch
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    hooks.configure_runtime("none", "cpu", policy="strict")
    spec = hooks.control_spec(seed)
    corpus, _, ids, _, _ = hooks.validate_data_with(hooks.IDENTITY, spec, ROOT, build.CORPUS)
    ids = ids[:1]
    hooks.verify_shards(corpus, ids)
    ds = hooks.MILWindows(build.CORPUS, ids, {u: np.zeros(12, np.float32) for u in ids}, 36, False, seed)
    loader = hooks.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=hooks.collate)
    parent = hooks.build_model(spec.document["config"]["model"], ROOT)
    snap = screen.snapshot("control", seed)
    need(snap == spatial.parents()[str(seed)], "parent manifest drift")
    ck = hooks.load_checkpoint(ROOT / snap["checkpoint"], map_location="cpu", weights_only=False)
    spatial.check_parent_checkpoint(ck, spec, snap)
    hooks.strict_load(parent, ck["model"])
    del ck
    baseline = hooks.evaluate_with(hooks.IDENTITY, parent, loader, "cpu", "none", corpus, tta=True)[0]
    evidence = {}
    for k in KINDS:
        w, record, arm = build.state(WORK / k)
        # Direct wrapper of authenticated parent exercises exact initial parity.
        model = spatial.FrozenWeakMIL(deepcopy(parent), adapter=k == "adapter")
        initial = hooks.evaluate_with(arm, model, loader, "cpu", "none", corpus, tta=True)[0]
        need(np.array_equal(initial, baseline), "initial parent parity failed")
        fit = build.arm_snapshot(w, record, seed, smoke=True)
        ck = hooks.load_checkpoint(ROOT / fit["checkpoint"], map_location="cpu", weights_only=False)
        hooks.strict_load(model, ck["model"])
        need(all(torch.equal(v, model.parent.state_dict()[n]) for n, v in parent.state_dict().items()),
             "CPU optimizer changed frozen parent")
        values = hooks.evaluate_with(arm, model, loader, "cpu", "none", corpus, tta=True)[0]
        need(np.array_equal(values[:, spatial.PROTECTED], baseline[:, spatial.PROTECTED]),
             "protected CPU TTA parity failed")
        evidence[k] = dict(initial_max_abs=float(abs(initial - baseline).max()),
                           protected_max_abs=float(abs(values[:, spatial.PROTECTED] - baseline[:, spatial.PROTECTED]).max()),
                           weak_max_abs=float(abs(values[:, spatial.WEAK] - baseline[:, spatial.WEAK]).max()),
                           trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                           parent_state_unchanged=True, reload=True)
        del model, ck
    need(not torch.cuda.is_initialized(), "CPU parity initialized CUDA")
    result = dict(status="REAL_PARENT_PARITY_PASS", seed=seed, rows=1, windows=36, flip=True,
                  cuda_initialized=False, arms=evidence)
    build.same_or_new(WORK / f"parity-s{seed}-v1.json", result)
    return result


def commands():
    """Head pair first, adapter pair second; one parent panel pair reused."""
    phases = {k: [] for k in ("smoke", "review", "fit", "panel", "readout")}
    for k in KINDS:
        doc = build.commands(WORK / k)
        for seed in ("42", "43"):
            cs = doc["commands"][seed]
            for phase, indexes in (("smoke", (0,)), ("review", (1, 2)), ("fit", (3, 4)),
                                   ("panel", (5, 6) if k == "head" else (5,))):
                for i in indexes:
                    cmd = cs[i]
                    if "tools/gpu_job.py" in cmd:
                        split = cmd.index("--") + 1
                        cmd = [*cmd[:split], build.PY, "-B", "-m", "tools.hb137_screen_v1_2_1", "budget-run", "--", *cmd[split:]]
                    phases[phase].append(cmd)
        if k == "head":
            phases["fit"].append([build.PY, "-B", "-m", "tools.hb137_screen_v1_2_1", "head-gate"])
    phases["readout"] = [[build.PY, "-B", "-m", "tools.hb137_screen_v1_2_1", "readout"]]
    phases["smoke"].insert(0, [build.PY, "-B", "tools/gpu_job.py", "--tag",
        "hb137-v121-memory-smoke", "--wait", "--", build.PY, "-B", "-m",
        "tools.hb137_screen_v1_2_1", "memory-smoke"])
    return dict(status="B_COMMANDS_ONLY", phases=phases, gpu_limit_seconds=BUDGET_SECONDS,
                conditional="Only if astra-b3 path is not admitted; B independent review required")


def budget_run(command, ledger=None, limit=BUDGET_SECONDS):
    from tools import hb137_safety_v1_2_1 as safety
    if ledger is None:
        safety.admission()
    return safety.run(command, ledger, limit)


def memory_smoke():
    from tools.hb137_safety_v1_2_1 import memory_smoke as run
    return run()


def cpu_memory_smoke():
    from tools.hb137_safety_v1_2_1 import memory_smoke as run
    return run("cpu")


def main():
    # Remainder is deliberately parsed separately; budget-run wraps exact argv.
    if len(sys.argv) > 1 and sys.argv[1] == "budget-run":
        need(sys.argv[2:3] == ["--"], "budget-run requires -- command")
        result = budget_run(sys.argv[3:])
    else:
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("command", choices=("configure", "cpu-card", "parity", "commands", "readout", "head-gate", "memory-smoke", "cpu-memory-smoke"))
        ap.add_argument("--seed", type=int, choices=(42, 43))
        a = ap.parse_args()
        if a.command in ("cpu-card", "parity"):
            need(a.seed is not None, "seed required")
        if a.command == "cpu-card":
            result = {k: build.cpu_card(WORK / k, a.seed) for k in KINDS}
        elif a.command == "parity":
            result = parity(a.seed)
        else:
            result = globals()[a.command.replace("-", "_")]()
    print(life.canonical_bytes(result).decode())


if __name__ == "__main__":
    main()
