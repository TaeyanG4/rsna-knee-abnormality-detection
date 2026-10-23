"""H-B-166 CPU memory/shape/time check v1.0.0 (2026-10-15). CPU only, never initialises CUDA.

Each phase runs in a fresh subprocess with the REAL ConvNeXt-S finding-MIL
(registered initial state) in fp32:
  train-control : one training microbatch, 2 studies x 12 full-field windows (control shape)
  train-mv      : one training microbatch, 2 studies x (12 full + 12 crop) windows (H-B-166 shape)
  eval-mv       : one no-grad inference microbatch, 2 studies x (36 + 36) windows (K36 shape)
For each: saved-for-backward activation bytes (autograd saved_tensors_hooks,
unique non-parameter storages), process peak RSS above the post-build baseline,
and wall time. The batch of 8 is 4 sequential microbatches of 2 (accumulation
4), so a microbatch is the memory peak. 4090 projection = measured control 4090
peak (H-B-155 fits) + the measured fp32 activation increase (upper bound; fp16
autocast saves most activations at half size). Loader timing compares real
per-study __getitem__ cost (train mode, real caches) for control vs multiview.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # empty, never -1: see tools/hb166_fit_v1.hide_gpus_for_cpu
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"
CONTROL_4090_PEAK = 4305453056   # H-B-155 control fits s42/s43 max_memory_reserved (summary.json)
CARD_BYTES = 24 * 10**9


def peak_rss():
    import psutil
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        return int(info.peak_wset)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def phase(name, threads):
    import torch
    torch.set_num_threads(threads)
    from tools import hb166_build_v1 as b
    from training.arms import hb166_multiview_v1 as mv
    from training.arms import mil_screen_hooks_v3 as h
    c = b.read(b.config_path(42))
    model = h.build_model(c["recipe"]["model"], ROOT)
    multiview = name != "train-control"
    if multiview:
        model = mv.attach_views(model)
    model = model.to(memory_format=torch.channels_last)
    k_full, k_crop, grad = {"train-control": (12, 0, True), "train-mv": (12, 12, True), "eval-mv": (36, 36, False)}[name]
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, k_full + k_crop, 3, 224, 224, generator=gen)
    views = torch.tensor([[0] * k_full + [1] * k_crop] * 2, dtype=torch.int64)
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    saved = {}

    def pack(t):
        storage = t.untyped_storage()
        if storage.data_ptr() not in params:
            saved[storage.data_ptr()] = storage.nbytes()
        return t

    baseline = peak_rss()
    began = time.perf_counter()
    if grad:
        model.train()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            logits = mv.ARM.forward(model, x, views) if multiview else \
                model(x.flatten(0, 1).contiguous(memory_format=torch.channels_last).reshape(x.shape))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.full_like(logits, .5))
        loss.backward()
    else:
        model.eval()
        with torch.no_grad():
            logits = mv.ARM.forward(model, x, views)
    seconds = time.perf_counter() - began
    assert logits.shape == (2, 12) and bool(torch.isfinite(logits).all())
    peak = peak_rss()
    return dict(phase=name, windows_per_study=k_full + k_crop, studies=2, images=2 * (k_full + k_crop), grad=grad,
                saved_activation_bytes=int(sum(saved.values())), rss_baseline_bytes=baseline, rss_peak_bytes=peak,
                rss_increase_bytes=peak - baseline, seconds=seconds, threads=threads,
                parameters=sum(p.numel() for p in model.parameters()),
                cuda_initialized=torch.cuda.is_initialized())


def loader_timing(studies, threads):
    import torch
    torch.set_num_threads(threads)
    from tools import hb166_build_v1 as b
    from training.arms import hb155_crops_v1 as crops
    from training.arms import hb166_multiview_v1 as mv
    from training.arms import mil_screen_hooks_v3 as h
    c = b.read(b.config_path(42))
    ids = c["train"][:studies]
    labels = {u: h.np.full(12, .5, h.np.float32) for u in ids}
    control = crops.Windows(h.MILWindows(ROOT / c["corpus"], ids, labels, 12, True, 42), 224)
    multi = mv.make_dataset(ROOT / c["corpus"], ROOT / c["crop_corpus"], ids, labels, 12, True, 42, 224)
    out = {}
    for name, ds in (("control", control), ("multiview", multi)):
        ds.set_epoch(1)
        ds[0]  # open memmaps
        for i in range(len(ids)):
            ds[i]  # warm the page cache for both caches
        began = time.perf_counter()
        for i in range(len(ids)):
            ds[i]
        out[name] = (time.perf_counter() - began) / len(ids)
    return dict(studies=studies, seconds_per_study=out, ratio=out["multiview"] / out["control"],
                cuda_initialized=torch.cuda.is_initialized())


def run(out, threads, studies):
    from tools import hb166_build_v1 as b
    results = {}
    for name in ("train-control", "train-mv", "eval-mv"):
        proc = subprocess.run([sys.executable, "-B", "-m", "tools.hb166_memcheck_v1", "phase", name,
                               "--threads", str(threads)], cwd=ROOT, capture_output=True, text=True, check=True)
        results[name] = json.loads(proc.stdout.strip().splitlines()[-1])
    proc = subprocess.run([sys.executable, "-B", "-m", "tools.hb166_memcheck_v1", "loader", "--studies", str(studies),
                           "--threads", "1"], cwd=ROOT, capture_output=True, text=True, check=True)
    loader = json.loads(proc.stdout.strip().splitlines()[-1])
    ctrl, mvt, ev = results["train-control"], results["train-mv"], results["eval-mv"]
    extra = mvt["saved_activation_bytes"] - ctrl["saved_activation_bytes"]
    projected_upper = CONTROL_4090_PEAK + extra
    projected_fp16 = CONTROL_4090_PEAK + extra // 2
    fits = [b.read(b.CONTROL_WORK / "fits" / b.control_member(s) / "summary.json") for s in b.SEEDS]
    train_s = sum(sum(r["wall_seconds"] for r in f["history"]) for f in fits) / len(fits)
    val_s = sum(sum(f["validation_seconds"].values()) for f in fits) / len(fits)
    total_s = sum(f["seconds"] for f in fits) / len(fits)
    step_s = sum(train_s / f["steps"] for f in fits) / len(fits)
    compute_ratio = mvt["seconds"] / ctrl["seconds"]
    ratio = max(compute_ratio, loader["ratio"])
    gold_s = val_s / 3 * (58 / 864) * 2 * ratio  # one K36-flip pass on 58 studies
    est = dict(control_train_seconds=train_s, control_validation_seconds=val_s, control_total_seconds=total_s,
               control_step_seconds=step_s, compute_ratio_cpu=compute_ratio, loader_ratio_cpu=loader["ratio"],
               scaling_ratio_used=ratio,
               loader_bound_check=dict(control_seconds_per_study_local=loader["seconds_per_study"]["control"],
                                       studies_per_worker_per_step=4,
                                       local_loader_seconds_per_step=loader["seconds_per_study"]["control"] * 4,
                                       observed_4090_step_seconds=step_s),
               multiview_fit_seconds=(total_s - val_s) * ratio + val_s * compute_ratio + gold_s + 120,
               note="multiview fit = (control total - validation) x max(compute, loader) ratio + validation x compute "
                    "ratio + Gold58 K36-flip pass + 120 s for hashing the second (crop) cache's shards")
    est["multiview_fit_hours"] = est["multiview_fit_seconds"] / 3600
    est["two_fits_plus_panels_hours"] = (2 * est["multiview_fit_seconds"] + 2 * 120) / 3600
    gate = b.MEMORY_GATE_BYTES
    result = dict(schema="hb166-memcheck/v1", version=VERSION, phases=results, loader=loader,
        accumulation_note="batch 8 = 4 sequential microbatches of 2 studies; the microbatch is the activation peak",
        control_4090_peak_reserved_bytes=CONTROL_4090_PEAK, activation_increase_fp32_bytes=extra,
        projected_4090_peak_upper_bytes=projected_upper, projected_4090_peak_fp16_bytes=projected_fp16,
        memory_gate_bytes=gate, card_bytes=CARD_BYTES,
        gate_pass_projection=projected_upper <= gate and mvt["rss_increase_bytes"] < CARD_BYTES,
        cpu_fp32_full_shape_rss_increase_bytes=mvt["rss_increase_bytes"],
        eval_rss_increase_bytes=ev["rss_increase_bytes"], fit_time_estimate=est,
        cuda_initialized=any(r["cuda_initialized"] for r in results.values()) or loader["cuda_initialized"])
    from tools.hb150_io_v1 import atomic_json
    atomic_json(out, result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("run", "phase", "loader"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--studies", type=int, default=8)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.command == "phase":
        print(json.dumps(phase(a.name, a.threads)))
    elif a.command == "loader":
        print(json.dumps(loader_timing(a.studies, a.threads)))
    else:
        from tools import hb166_build_v1 as b
        r = run(a.out or b.WORK / "memcheck.json", a.threads, a.studies)
        print(json.dumps(dict(projected_4090_peak_upper_gib=r["projected_4090_peak_upper_bytes"] / 2**30,
                              projected_4090_peak_fp16_gib=r["projected_4090_peak_fp16_bytes"] / 2**30,
                              cpu_fp32_rss_increase_gib=r["cpu_fp32_full_shape_rss_increase_bytes"] / 2**30,
                              fit_hours=r["fit_time_estimate"]["multiview_fit_hours"],
                              cuda_initialized=r["cuda_initialized"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
