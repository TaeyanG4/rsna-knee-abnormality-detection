"""H-B-146 1.1.1 screen adapter: new split, shared hooked train/evaluate paths.

Changelog 2026-10-13 v1.1.1: final EXPOSED producer contract; bounded atomic writes and nonfatal heartbeat. v1.1.0 retained.

Changelog 1.1.0: EXPOSED_EXPLORATORY three arms, explicit flag and bound admission.
v1.0 and shared runners are unchanged. Smoke uses two batch-1 updates, K12/224,
and cannot create a scientific checkpoint. No training metrics select epochs.
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb146_pl_labels_v111 as pl
from training.arms.hb146_pl_v1 import Arm
from training.arms import mil_screen_hooks_v1 as h
from training.arms import mil

SOURCES = ["tools/hb146_pl_labels_v111.py", "tools/hb146_teacher_v111.py", "tools/hb146_screen_v111.py",
           "tools/hb146_readout_v111.py", "training/arms/hb146_pl_v1.py",
           "training/arms/mil_screen_hooks_v1.py", "training/arms/mil_screen_v1.py",
           "training/arms/mil.py", "training/arms/mil_small.py", "training/arms/lifecycle.py",
           "training/src/raptor_common.py", "training/src/raptor_runtime.py", "training/src/raptor_exp185.py",
           "tools/multi_lens_readout_l4_v2.py", "tools/multi_lens_readout.py", "tools/oai_moaks_panel.py"]


def prepare(work, engineering=False, exposed_exploratory=False):
    pl.need(exposed_exploratory, "explicit --exposed-exploratory required")
    s, _, _ = pl.validate_labels(work, engineering)
    labels_dir = work / ("engineering" if engineering else "screen_f0")
    value = {"version": "1.1.1", "engineering_only": engineering,
             "pins": pl.pins([ROOT / p for p in SOURCES] + [work / "registration.json",
                             labels_dir / "global.csv", labels_dir / "fill.csv", labels_dir / "provenance.json"]),
             "train": len(s["train"]), "val": len(s["val"]), "seeds": [42, 43],
             "status": pl.STATUS, "limitations": pl.LIMITATIONS, "exposed_exploratory": True,
             "arms": ["control", "global", "fill"], "epochs": 20, "train_k": 12, "batch": 8,
             "weights": s["weights"], "selection": "fixed_epoch20", "go": s["go"]}
    dest = work / ("prepare-engineering.json" if engineering else "prepare.json")
    if dest.exists() and not engineering:
        pl.need(pl.read(dest) == value, "prepared inputs changed; re-review/version before launch")
    else:
        pl.write(dest, value)
    return {"prepared": True, "engineering_only": engineering, "train": value["train"], "val": value["val"]}


def prepared(work, engineering=False):
    path = work / ("prepare-engineering.json" if engineering else "prepare.json")
    value = pl.read(path)
    pl.need(value["engineering_only"] is engineering and value["seeds"] == [42, 43], "wrong preparation")
    pl.require_exposed(value, True)
    pl.need(value["arms"] == ["control", "global", "fill"], "wrong arms")
    pl.check_pins(value["pins"])
    pl.validate_labels(work, engineering)
    return value


def admit(work, review=None, engineering=False, exposed_exploratory=False):
    prepared(work, engineering)
    _, _, provenance = pl.validate_labels(work, engineering)
    pl.require_exposed(provenance, exposed_exploratory)
    result = {"admitted": True, "engineering_only": engineering, "status": pl.STATUS,
              "limitations": pl.LIMITATIONS, "exposed_exploratory": True,
              "prepare_sha256": pl.sha(work / ("prepare-engineering.json" if engineering else "prepare.json")),
              "authorization": "B explicitly authorizes EXPOSED_EXPLORATORY screen; not clean confirmation"}
    dest = work / ("admission-engineering.json" if engineering else "admission.json")
    if dest.exists() and not engineering:
        pl.need(pl.read(dest) == result, "admission differs; version required")
    else:
        pl.write(dest, result)
    return result


def setup(device):
    import cv2
    h.torch.set_num_threads(2)
    h.torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    amp = "none" if device == "cpu" else "fp16"
    runtime = h.configure_runtime(amp, device, policy="strict")
    return amp, runtime


def output_dir(work, arm, seed, smoke, device):
    return work / (f"smoke-{device}" if smoke else "fits") / f"{arm}-s{seed}"


def save_predictions(path, ids, values, checkpoint, identity, *, k, tta, engineering):
    h.common.write_predictions(path, ids, values)
    pl.write(path.with_suffix(".receipt.json"), {"csv_sha256": pl.sha(path), "checkpoint_sha256": pl.sha(checkpoint),
             "identity": identity, "k": k, "tta": tta, "engineering_only": engineering,
             "uid_sha256": h.uid_hash(ids), "rows": len(ids)})


def evaluate(model, arm, cache, ids, seed, device, amp, k, tta):
    corpus = h.common.Corpus(cache)
    h.verify_shards(corpus, ids)
    zeros = {u: h.np.zeros(12, h.np.float32) for u in ids}
    ds = arm.dataset(h.MILWindows(cache, ids, zeros, k, False, seed), train=False)
    loader = h.DataLoader(ds, batch_size=1 if device == "cpu" else 4, shuffle=False,
                          num_workers=0, collate_fn=arm.collate_fn)
    values, ordered, _ = h.evaluate_with(arm, model, loader, device, amp, corpus, tta=tta)
    pl.need(ordered == ids, "prediction UID order differs")
    return values


def fit(work, arm_name, seed, device, smoke=False, engineering=False, review=None, exposed_exploratory=False):
    pl.need(exposed_exploratory, "explicit --exposed-exploratory required")
    pl.need(seed in (42, 43) and arm_name in ("control", "global", "fill"), "unregistered arm/seed")
    pl.need(not engineering or smoke, "engineering fixtures forbidden in scientific fits")
    prepared(work, engineering)
    if not smoke:
        pl.need(device == "cuda", "full screen uses registered CUDA recipe")
        admit(work, review, exposed_exploratory=exposed_exploratory)
    s, soft, provenance = pl.validate_labels(work, engineering, arm="global" if arm_name == "control" else arm_name)
    s0 = pl.table(ROOT / s["recipe"]["labels"]["path"])
    pl.need(h.np.array_equal(pl.frozen_weights(s0, s["train"]), h.np.array(s["weights"], h.np.float32)),
            "frozen L2 weight mismatch")
    amp, runtime = setup(device)
    h.seed_all(seed)
    arm = Arm(soft if arm_name != "control" else None)
    cache = ROOT / s["recipe"]["roots"]["corpus"]
    ids = s["train"][:2] if smoke else s["train"]
    h.verify_shards(h.common.Corpus(cache), ids)
    ds = arm.dataset(h.MILWindows(cache, ids, s0, 12, True, seed), train=True)
    gen = h.torch.Generator().manual_seed(seed)
    loader = h.DataLoader(ds, batch_size=1 if smoke else 8, shuffle=True, drop_last=True,
                          num_workers=0, generator=gen, collate_fn=arm.collate_fn)
    epochs = 1 if smoke else 20
    steps = len(loader) * epochs
    pl.need(not smoke or steps == 2, "smoke must attempt exactly two updates")
    model = h.build_model(s["recipe"]["model"], ROOT).to(device)
    opt, sched, scaler = h.optimizer_stack(model, steps, device, amp)
    out = output_dir(work, arm_name, seed, smoke, device)
    out.mkdir(parents=True, exist_ok=True)
    pl.need(smoke or not (out / "final.pt").exists(), "scientific checkpoint exists; no overwrite")
    identity = {"arm": arm_name, "seed": seed, "registration_sha256": pl.sha(work / "registration.json"),
                "prepare_sha256": pl.sha(work / ("prepare-engineering.json" if engineering else "prepare.json")),
                "runtime": runtime, "smoke": smoke, "engineering_only": engineering,
                "labels_sha256": pl.sha(ROOT / s["recipe"]["labels"]["path"]) if arm_name == "control" else provenance["arm_hashes"][arm_name], "epochs": epochs,
                "status": pl.STATUS, "limitations": pl.LIMITATIONS, "exposed_exploratory": True,
                "weights": s["weights"], "model": s["recipe"]["model"]}
    counters, history = {"updates": 0, "skips": 0}, []
    start = time.monotonic()
    for epoch in range(epochs):
        pl.write_status(out / "heartbeat.json", {"status": "running", "epoch": epoch, "epochs": epochs})
        ds.set_epoch(epoch)
        history.append(h.train_epoch_with(arm, model, loader, opt, sched, scaler,
                       h.torch.tensor(s["weights"]), device, amp, max(1, int(steps * .01)) if not smoke else 0, counters))
    pl.need(counters["updates"] == steps and (not smoke or counters["skips"] == 0), "incomplete fit")
    checkpoint = out / "final.pt"
    tmp = out / "final.tmp"
    h.torch.save({"model": model.state_dict(), "identity": identity, "epoch": epochs - 1}, tmp)
    pl.replace_retry(tmp, checkpoint)
    # Reload through strict state checks; same forward path for all readouts.
    state = h.torch.load(checkpoint, map_location="cpu", weights_only=True)
    h.strict_load(model, state["model"])
    del state, opt, sched, scaler
    val = s["val"][:1] if smoke else s["val"]
    for name, k, tta in (("val_k24", 24, False), ("val_k36", 36, False), ("val", 36, True)):
        values = evaluate(model, arm, cache, val, seed, device, amp, k, tta)
        save_predictions(out / (name + ".csv"), val, values, checkpoint, identity, k=k, tta=tta, engineering=smoke)
    gold = s["gold"][:1] if smoke else s["gold"]
    values = evaluate(model, arm, cache, gold, seed, device, amp, 36, True)
    save_predictions(out / "gold.csv", gold, values, checkpoint, identity, k=36, tta=True, engineering=smoke)
    result = {"arm": arm_name, "seed": seed, "device": device, "smoke": smoke, "steps": steps,
              "counters": counters, "seconds": time.monotonic() - start, "runtime": runtime,
              "checkpoint_sha256": pl.sha(checkpoint), "identity": identity, "history": history}
    pl.write(out / "summary.json", result)
    pl.write_status(out / "heartbeat.json", {"status": "complete", "epochs": epochs, "counters": counters})
    return {k: result[k] for k in ("arm", "seed", "device", "smoke", "steps", "counters", "seconds")}


def panel_admit(work, ancestry=None, engineering=False, exposed_exploratory=False):
    from tools import oai_moaks_panel as p
    pl.need(exposed_exploratory, "explicit --exposed-exploratory required")
    prepared(work, engineering)
    ids = p.validate()
    p.routing()
    binding = {"status": pl.STATUS, "limitations": pl.LIMITATIONS, "exposed_exploratory": True,
               "engineering_only": engineering, "oai_disclosure": pl.OAI_STATUS,
               "prepare_sha256": pl.sha(work / ("prepare-engineering.json" if engineering else "prepare.json")),
               "panel_registration_sha256": pl.sha(p.OUT / "registration.json"),
               "cache_sha256": pl.sha(ROOT / p.CACHES["64"] / "manifest.json")}
    digest = pl.hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return ids, {**binding, "eligible": not engineering, "binding_sha256": digest,
                 "eligibility_scope": "exploratory only; no full OAI-free teacher ancestry claim"}


def panel(work, arm_name, seed, device, smoke=False, engineering=False, ancestry=None, exposed_exploratory=False):
    from tools import oai_moaks_panel as p
    ids, admission = panel_admit(work, ancestry, engineering, exposed_exploratory)
    pl.need(not engineering or smoke, "engineering panel must be smoke")
    ids = ids[:1] if smoke else ids
    out = output_dir(work, arm_name, seed, smoke, device)
    summary = pl.read(out / "summary.json")
    checkpoint = out / "final.pt"
    pl.need(summary["checkpoint_sha256"] == pl.sha(checkpoint) and summary["smoke"] is smoke,
            "panel checkpoint mismatch")
    check_panel_identity(work, summary["identity"], arm_name, seed, smoke, engineering)
    amp, runtime = setup(device)
    state = h.torch.load(checkpoint, map_location="cpu", weights_only=True)
    pl.need(state["identity"] == summary["identity"] and state["epoch"] == (0 if smoke else 19), "panel checkpoint incomplete")
    model = h.build_model(state["identity"]["model"], ROOT)
    h.strict_load(model, state["model"])
    del state
    values = evaluate(model.to(device), Arm(), ROOT / p.CACHES["64"], ids, seed, device, amp, 36, True)
    save_predictions(out / "oai.csv", ids, values, checkpoint, summary["identity"], k=36, tta=True, engineering=smoke)
    pl.write(out / "oai-admission.json", {**admission, "runtime": runtime,
             "csv_sha256": pl.sha(out / "oai.csv"), "cache_sha256": pl.sha(ROOT / p.CACHES["64"] / "manifest.json")})
    return {"rows": len(ids), "eligible": admission["eligible"], "smoke": smoke}


def check_panel_identity(work, identity, arm, seed, smoke, engineering):
    pl.need(identity["arm"] == arm and identity["seed"] == seed and identity["smoke"] is smoke
            and identity["engineering_only"] is engineering
            and identity["registration_sha256"] == pl.sha(work / "registration.json")
            and identity["prepare_sha256"] == pl.sha(work / ("prepare-engineering.json" if engineering else "prepare.json")),
            "stale or mismatched panel preparation")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "admit", "fit", "panel-admit", "panel"])
    ap.add_argument("--work", type=Path, default=pl.WORK)
    ap.add_argument("--arm", choices=["control", "global", "fill"], default="global")
    ap.add_argument("--seed", type=int, choices=[42, 43], default=42)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--engineering", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--review", type=Path)
    ap.add_argument("--ancestry", type=Path)
    ap.add_argument("--exposed-exploratory", action="store_true")
    a = ap.parse_args(argv)
    if a.command == "prepare":
        result = prepare(a.work, a.engineering, a.exposed_exploratory)
    elif a.command == "admit":
        result = admit(a.work, a.review, a.engineering, a.exposed_exploratory)
    elif a.command == "panel-admit":
        ids, admission = panel_admit(a.work, a.ancestry, a.engineering, a.exposed_exploratory)
        result = {"rows": len(ids), **admission}
    elif a.command == "panel":
        result = panel(a.work, a.arm, a.seed, a.device, a.smoke, a.engineering, a.ancestry, a.exposed_exploratory)
    else:
        result = fit(a.work, a.arm, a.seed, a.device, a.smoke, a.engineering, a.review, a.exposed_exploratory)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
