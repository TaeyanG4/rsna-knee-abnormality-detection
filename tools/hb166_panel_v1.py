"""H-B-166 control Gold58 panel inference v1.0.0 (2026-10-15).

L7 needs Gold58 predictions of the EXISTING H-B-155 control checkpoints
(fits/control-s4x/final.pt, SHA256-pinned); the control fits exported no gold
rows. This tool runs exactly the control serving path of tools/hb155_fit_v1.py
(hb155_crops_v1.Windows on native448, NativeArm forward, hooks-v3
evaluate_with, K36, flip TTA, batch 4, fp16 strict runtime) on the same 4090
stack as the multiview fits. Gold58 is exposed/descriptive (never eligible).
CPU smoke additionally replays two fold-2 validation studies and compares them
with the stored GPU val.csv (fp32 CPU vs fp16 GPU tolerance 0.02).
"""
import argparse
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

from tools.parity_runtime import runtime_info
from tools import hb155_build_v1 as b155
from tools import hb155_fit_v1 as base
from tools import hb166_build_v1 as b
from tools.hb150_io_v1 import atomic_json
from training.arms import mil_screen_hooks_v3 as h

ROOT = Path(__file__).resolve().parents[1]
VERSION = b.VERSION
PARITY_TOLERANCE = .02


def predict(model, c155, ids, device, amp, progress, phase):
    corpus = h.common.Corpus(ROOT / c155["corpus"])
    labels = {uid: h.np.zeros(12, h.np.float32) for uid in ids}
    ds = base.crops.Windows(h.MILWindows(ROOT / c155["corpus"], ids, labels, 36, False, c155["seed"]), c155["resolution"])
    loader = h.DataLoader(ds, batch_size=1 if device == "cpu" else 4, shuffle=False, num_workers=c155["workers"],
                          worker_init_fn=h.worker_init, collate_fn=base.native.NativeArm.collate_fn)
    old = h.PROGRESS
    h.PROGRESS = progress
    try:
        began = time.monotonic()
        values, ordered, batches = h.evaluate_with(base.native.NativeArm(), model, loader, device, amp, corpus,
                                                   tta=True, phase=phase)
    finally:
        h.PROGRESS = old
    h.need(ordered == ids, "panel UID order differs")
    return values, batches, time.monotonic() - began


def panel(seed, out, *, device="cpu", smoke=True, review=None):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ENVIRONMENT: see tools/hb166_fit_v1.hide_gpus_for_cpu
    config = b.config_path(seed, engineering=device == "cpu" and smoke)
    c = b.validate(config)
    registration = b.registration_binding()
    if not smoke:
        h.need(device == "cuda", "scientific panel requires CUDA")
        b.require_fit(config, review)
    h.need(device == "cpu" or (os.environ.get("HB155_TOKEN") and os.environ.get("HB155_HEARTBEAT")),
           "CUDA execution requires watchdog supervision")
    c155 = b155.validate(ROOT / c["control_config"])
    h.need(h.sha(ROOT / c["control_config"]) == b.CONTROL_PINS[seed]["config"] and c155["seed"] == seed, "control config differs")
    member = b.control_member(seed)
    checkpoint = b.CONTROL_WORK / "fits" / member / "final.pt"
    h.need(checkpoint.is_file() and h.sha(checkpoint) == b.CONTROL_PINS[seed]["checkpoint"],
           "control checkpoint missing or differs from pin")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=smoke)
    progress = base.Progress(os.environ.get("HB155_HEARTBEAT"), os.environ.get("HB155_TOKEN"))
    progress.phase_start("startup")
    atomic_json(out / "heartbeat.json", dict(status="running", exposure="EXPOSED", clean_oof=False), critical=False)
    h.torch.set_num_threads(2)
    amp = "none" if device == "cpu" else "fp16"
    runtime = h.configure_runtime(amp, device, policy="strict")
    pairing, snapshot_hash, gpu_name = None, None, "cpu"
    if device == "cuda":
        snapshot = runtime_info()
        atomic_json(out / "runtime.json", snapshot)
        pairing = b.require_runtime(snapshot)
        snapshot_hash = h.sha(out / "runtime.json")
        gpu_name = h.torch.cuda.get_device_name(0)
        h.need(gpu_name == b.GPU, "paired panel requires the registered GPU")
        capacity = h.torch.cuda.get_device_properties(0).total_memory
        h.torch.cuda.set_per_process_memory_fraction(min(1., b.MEMORY_GATE_BYTES / capacity))
        h.torch.cuda.reset_peak_memory_stats()
    gold = c["gold"][:1 if device == "cpu" else 8] if smoke else c["gold"]
    corpus = h.common.Corpus(ROOT / c155["corpus"])
    h.verify_shards(corpus, gold + (c155["val"][:2] if device == "cpu" else []))
    model = h.build_model(c155["recipe"]["model"], ROOT)
    state = h.torch.load(checkpoint, map_location="cpu", weights_only=True)
    h.need(state["identity"]["arm"] == "control" and state["identity"]["seed"] == seed, "checkpoint identity differs")
    h.strict_load(model, state["model"])
    del state
    model = model.to(memory_format=h.torch.channels_last).to(device)
    progress.completed("startup", {"checkpoint_verified": True})
    identity = dict(version=VERSION, experiment="H-B-166 control Gold58 panel", arm="control-panel", seed=seed,
        control_member=member, checkpoint_sha256=b.CONTROL_PINS[seed]["checkpoint"], config_sha256=h.sha(config),
        control_config_sha256=b.CONTROL_PINS[seed]["config"], registration_sha256=next(iter(registration.values())),
        smoke=smoke, engineering_only=smoke, device=device, gpu_name=gpu_name, runtime=runtime,
        runtime_snapshot_sha256=snapshot_hash, k=36, tta=True, batch=1 if device == "cpu" else 4,
        serving_path="hb155_crops_v1.Windows(native448) + NativeArm + mil_screen_hooks_v3.evaluate_with",
        exposure="EXPOSED", clean_oof=False, gold_sha256=h.uid_hash(gold))
    start = time.monotonic()
    try:
        values, batches, seconds = predict(model, c155, gold, device, amp, progress, "panel_inference")
        path = out / "gold.csv"
        ok = base.export_prediction(path, gold, values, checkpoint, identity, k=36, tta=True, engineering=smoke)
        parity = None
        if device == "cpu":
            import pandas as pd
            ids = c155["val"][:2]
            replay, _, _ = predict(model, c155, ids, device, amp, progress, "panel_inference")
            stored_path = b.CONTROL_WORK / "fits" / member / "val.csv"
            h.need(h.sha(stored_path) == b.CONTROL_PINS[seed]["val_csv"], "stored control predictions differ")
            stored = pd.read_csv(stored_path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
            reference = stored.loc[ids, list(h.TARGETS)].to_numpy(float)
            parity = dict(studies=len(ids), max_abs_diff=float(abs(replay - reference).max()),
                          tolerance=PARITY_TOLERANCE, reference="stored GPU fp16 val.csv (K36 flip)")
            h.need(parity["max_abs_diff"] <= PARITY_TOLERANCE, "control serving-path replay differs from stored predictions")
        peak = h.torch.cuda.max_memory_reserved() if device == "cuda" else 0
        h.need(device == "cpu" or 0 < peak <= b.MEMORY_GATE_BYTES, "12 GiB measured memory gate failed")
        result = dict(version=VERSION, passed=bool(ok), status="PASS" if ok else "EXPORT_FAILED", arm="control-panel",
            seed=seed, device=device, smoke=smoke, engineering_only=smoke, config_sha256=h.sha(config),
            checkpoint_sha256=b.CONTROL_PINS[seed]["checkpoint"], gpu_name=gpu_name, full_shape=device == "cuda",
            runtime_pairing=pairing, peak_reserved_bytes=peak, rows=len(gold), batches=batches,
            healthy_rates=dict(panel=batches / seconds), seconds=time.monotonic() - start, parity=parity,
            cpu_cuda_initialized=h.torch.cuda.is_initialized() if device == "cpu" else None,
            identity=identity, csv_sha256=h.sha(path) if ok else None, exposure="EXPOSED", clean_oof=False)
        h.need(device != "cpu" or result["cpu_cuda_initialized"] is False, "CPU panel initialised CUDA")
        atomic_json(out / "summary.json", result, critical=False)
        atomic_json(out / "heartbeat.json", dict(status="complete" if ok else "export_failed",
                                                exposure="EXPOSED", clean_oof=False), critical=False)
        return result
    except BaseException as exc:
        atomic_json(out / "failure.json", dict(status="FAILED", error=type(exc).__name__), critical=False)
        atomic_json(out / "heartbeat.json", dict(status="failed", error=type(exc).__name__), critical=False)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("smoke", "fit"))
    ap.add_argument("--seed", type=int, choices=b.SEEDS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--review", type=Path)
    a = ap.parse_args(argv)
    value = panel(a.seed, a.out, device=a.device, smoke=a.command == "smoke", review=a.review)
    print(json.dumps({k: value[k] for k in ("passed", "arm", "seed", "device", "rows", "seconds", "parity")}))
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
