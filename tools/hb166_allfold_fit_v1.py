"""H-B-166 multi-view paired pilot fit v1.2.2 (2026-10-15).

Fork of tools/hb155_fit_v1.py v1.2.2 for ONE registered change: the bag holds
the unchanged control full-field native448 windows plus compartment-crop
windows with a zero-initialised view-type embedding
(training/arms/hb166_multiview_v1.py). Recipe, optimizer, schedule, AMP,
clip, skips, fixed final epoch, K24/K36/K36flip validation, CSV/receipt export,
progress journal and watchdog contract are those of H-B-155. Additions: two
authenticated caches, view ids, per-view flip TTA, Gold58 K36-flip export for
L7, rotating epoch checkpoints (latest verified pair kept), 12 GiB measured
memory gate (BUILDER-CHECKLIST v1.0.6) and a runtime-pairing check against the
control fits' 4090 snapshot. CPU smoke runs the REAL ConvNeXt-S model.
"""
import argparse
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from tools.parity_runtime import runtime_info
from tools import hb155_fit_v1 as base
from tools import hb166_allfold_build_v1 as b
from tools.hb150_io_v1 import atomic_json, safe_unlink, JournalWriteError
from training.arms import hb166_multiview_v1 as mv
from training.arms import mil_screen_hooks_v3 as h

ROOT = Path(__file__).resolve().parents[1]
VERSION = b.VERSION
Progress = base.Progress


def tiny_model():
    """Unit tests only: real FindingMIL code path on a random convnext_atto."""
    h.torch.manual_seed(0)
    return h.FindingMIL(h.common.build_backbone("convnext_atto", pretrained=False))


def build(c, device, cpu_model):
    if device == "cuda" or cpu_model == "real":
        model = h.build_model(c["recipe"]["model"], ROOT)
    else:
        h.need(cpu_model == "tiny", "unknown CPU model")
        model = tiny_model()
    return mv.attach_views(model)


def save_rotating(path, previous, **kwargs):
    """hb155 save_epoch, then remove the previous verified pair (disk bound)."""
    base.save_epoch(path, **kwargs)
    receipt = b.read(Path(path).with_suffix(".receipt.json"))
    h.need(receipt["sha256"] == h.sha(path) and receipt["identity"] == kwargs["identity"], "new epoch checkpoint unverified")
    if previous is not None and Path(previous).exists():
        old = Path(previous)
        h.need(old.parent == Path(path).parent and not old.is_symlink(), "rotation outside owned epochs directory")
        safe_unlink(old.with_suffix(".receipt.json"))
        safe_unlink(old)


def hide_gpus_for_cpu():
    """ENVIRONMENT (2026-10-15): on this Windows torch 2.5.1+cu121 build CUDA_VISIBLE_DEVICES=-1 (or any
    invalid value) makes CPU autograd backward / interpreter exit raise a native access violation, while
    an EMPTY value hides every GPU safely (device_count 0, CUDA never initialised). CPU paths force ''
    before any CUDA query; CUDA paths never call this."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def fit(config_path, out, *, device="cpu", smoke=True, review=None, resume=None, cpu_model="real"):
    if device == "cpu":
        hide_gpus_for_cpu()
    c = b.validate(config_path)
    registration = b.registration_binding()
    h.need(smoke or device == "cuda", "scientific fits require CUDA")
    h.need(smoke == c["engineering_only"] or (smoke and device == "cuda"),
           "CPU smoke uses engineering configs; scientific configs need CUDA")
    if device == "cuda":
        b.require_gpu()
    if not smoke:
        b.require_fit(config_path, review)
    h.need(device == "cpu" or (os.environ.get("HB155_TOKEN") and os.environ.get("HB155_HEARTBEAT")),
           "CUDA execution requires watchdog supervision")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=smoke or resume is not None)
    if not smoke and (out / "final.pt").exists():
        status = b.read(out / "heartbeat.json") if (out / "heartbeat.json").exists() else {}
        h.need(resume is not None and status.get("status") != "complete",
               "scientific final checkpoint already exists; only explicit partial resume allowed")
    progress = Progress(os.environ.get("HB155_HEARTBEAT"), os.environ.get("HB155_TOKEN"))
    progress.phase_start("startup")
    atomic_json(out / "heartbeat.json", dict(status="running", exposure="EXPOSED", clean_oof=False), critical=False)
    h.torch.set_num_threads(2)
    amp = "none" if device == "cpu" else "fp16"
    runtime = h.configure_runtime(amp, device, policy="strict")
    runtime_snapshot_hash, pairing = None, None
    if device == "cuda":
        snapshot = runtime_info()
        atomic_json(out / "runtime.json", snapshot)
        pairing = b.require_runtime(snapshot)
        runtime_snapshot_hash = h.sha(out / "runtime.json")
    h.seed_all(c["seed"])
    if device == "cuda":
        gpu_name = h.torch.cuda.get_device_name(0)
        h.need(gpu_name == c["gpu_name"], "paired comparison requires the registered GPU")
        capacity = h.torch.cuda.get_device_properties(0).total_memory
        h.torch.cuda.set_per_process_memory_fraction(min(1., c["memory_gate_bytes"] / capacity))
        h.torch.cuda.reset_peak_memory_stats()
    else:
        gpu_name = "cpu"
    full_corpus = h.common.Corpus(ROOT / c["corpus"])
    crop_corpus = h.common.Corpus(ROOT / c["crop_corpus"])
    h.need(full_corpus.contract == c["corpus_contract"] == mv.FULL_CONTRACT
           and crop_corpus.contract == c["crop_corpus_contract"] == mv.CROP_CONTRACT, "cache contracts differ")
    train = c["train"][:2 if device == "cpu" else 16] if smoke else c["train"]
    val = c["val"][:1 if device == "cpu" else 16] if smoke else c["val"]
    gold = c["gold"][:1 if device == "cpu" else 16] if smoke else c["gold"]
    every = list(dict.fromkeys(train + val + gold))
    h.verify_shards(full_corpus, every)
    progress.completed("startup", {"shards_verified": "native448"})   # heartbeat between the two ~56 GB hashes
    h.verify_shards(crop_corpus, every)
    progress.completed("startup", {"shards_verified": "crop-cache-v220"})
    labels = {uid: h.np.full(12, .5, h.np.float32) for uid in train} if smoke else base.scientific_targets(c)
    labels.update({uid: h.np.zeros(12, h.np.float32) for uid in val + gold})
    arm = mv.ARM

    def dataset(ids, k, train_mode):
        return mv.make_dataset(ROOT / c["corpus"], ROOT / c["crop_corpus"], ids, labels, k, train_mode,
                               c["seed"], c["resolution"])

    ds = dataset(train, c["train_k"], True)
    batch = 2 if device == "cpu" else 8
    generator = h.torch.Generator().manual_seed(c["seed"])
    loader = h.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=c["workers"],
                          persistent_workers=True, worker_init_fn=h.worker_init, generator=generator,
                          collate_fn=arm.collate_fn, pin_memory=device == "cuda")
    epochs = 1 if smoke else 20
    steps = len(loader) * epochs
    h.need(steps > 0 and (smoke or steps == c["optimizer_steps"]), "registered update count differs")
    model = build(c, device, cpu_model).to(memory_format=h.torch.channels_last).to(device)
    with h.torch.no_grad():
        probe = h.torch.zeros(1, 2, 3, c["resolution"], c["resolution"], device=device)
        arm.forward(model.eval(), probe, h.torch.tensor([[mv.FULL, mv.CROP]], device=device))
        model.train()
    opt, sched, scaler = h.optimizer_stack(model, steps, device, amp)
    head_ids = {id(p) for p in opt.param_groups[1]["params"]}
    h.need(id(model.view_embed) in head_ids, "view embedding must be in the head parameter group")
    identity = dict(version=VERSION, experiment="H-B-166 multiview paired pilot", arm="multiview", seed=c["seed"],
        config_sha256=h.sha(config_path), registration_sha256=next(iter(registration.values())),
        labels_sha256=None if smoke else h.sha(ROOT / c["label_path"]), model=c["recipe"]["model"],
        cpu_model=cpu_model if device == "cpu" else None, weights=c["weights"], epochs=epochs,
        resolution=c["resolution"], image_size=c["resolution"], smoke=smoke, engineering_only=smoke,
        exposure="EXPOSED", clean_oof=False, status="EXPOSED_ALLFOLD_REPLACEMENT", estimand=c["estimand"],
        runtime=runtime, gpu_name=gpu_name, batch=batch, microbatch=2, accumulation=4, fold=c["fold"],
        workers=c["workers"], runtime_snapshot_sha256=runtime_snapshot_hash, corpus_contract=c["corpus_contract"],
        crop_corpus_contract=c["crop_corpus_contract"], window_budget=c["window_budget"],
        view_embedding=c["view_embedding"], exclusions_sha256=c["exclusions_sha256"],
        side_unknown_policy=c["side_unknown_policy"], control_member=c["control_member"],
        train_sha256=h.uid_hash(c["train"]), val_sha256=h.uid_hash(c["val"]), gold_sha256=h.uid_hash(c["gold"]))
    start_epoch, history, counters = 0, [], {"updates": 0, "skips": 0}
    if resume is not None:
        start_epoch, history, counters = h.restore_epoch(resume, model=model, opt=opt, sched=sched,
            scaler=scaler, generator=generator, identity=identity, device=device, loader=loader)
        h.need(0 <= start_epoch <= epochs and counters["updates"] == start_epoch * len(loader), "resume updates differ")
    start = time.monotonic()
    try:
        previous = Path(resume) if resume is not None and Path(resume).parent == out / "epochs" else None
        for epoch in range(start_epoch, epochs):
            ds.set_epoch(epoch)
            began = time.monotonic()
            record = mv.train_epoch(arm, model, loader, opt, sched, scaler, h.torch.tensor(c["weights"]),
                                    device, amp, 0 if smoke else steps // 100, counters, progress)
            record["wall_seconds"] = time.monotonic() - began
            history.append({**record, "epoch": epoch})
            path = out / "epochs" / f"epoch-{epoch:03d}.pt"
            save_rotating(path, previous, epoch=epoch, model=model, opt=opt, sched=sched, scaler=scaler,
                          generator=generator, history=history, counters=counters, identity=identity,
                          device=device, progress=progress)
            previous = path
        h.need(counters["updates"] == steps and (not smoke or counters["skips"] == 0), "incomplete fixed-final fit")
        checkpoint, temp = out / "final.pt", out / "final.tmp"
        progress.phase_start("checkpoint_write")
        with temp.open("wb") as stream:
            h.torch.save(dict(model=model.state_dict(), identity=identity, epoch=epochs - 1,
                              model_kind="hb166_multiview_finding_mil_v1"),
                         base.ProgressWriter(stream, progress=progress))
        base.safe_replace(temp, checkpoint)
        state = h.torch.load(checkpoint, map_location="cpu", weights_only=True)
        h.strict_load(model, state["model"])
        del state, opt, sched, scaler
        progress.completed("checkpoint_reload")
        durations, batches_by_name, export_failures = {}, {}, []
        for name, ids, k, tta, phase in (("val_k24", val, 24, False, "validation"), ("val_k36", val, 36, False, "validation"),
                                         ("val", val, 36, True, "validation"), ("gold", gold, 36, True, "panel_inference")):
            eval_loader = h.DataLoader(dataset(ids, k, False), batch_size=1 if device == "cpu" else 4, shuffle=False,
                                       num_workers=c["workers"], worker_init_fn=h.worker_init, collate_fn=arm.collate_fn)
            began = time.monotonic()
            values, ordered, batches = mv.evaluate(arm, model, eval_loader, device, amp, full_corpus, crop_corpus,
                                                   progress, tta=tta, phase=phase)
            durations[name] = time.monotonic() - began
            batches_by_name[name] = batches
            h.need(ordered == ids, "prediction UID order differs")
            if not base.export_prediction(out / (name + ".csv"), ids, values, checkpoint, identity,
                                          k=k, tta=tta, engineering=smoke):
                export_failures.append(name)
        healthy_rates = dict(train_step=sum(r["batches"] for r in history) / sum(r["wall_seconds"] for r in history),
                             validation=min(batches_by_name[n] / durations[n] for n in ("val_k24", "val_k36", "val")),
                             panel=batches_by_name["gold"] / durations["gold"])
        peak = h.torch.cuda.max_memory_reserved() if device == "cuda" else 0
        h.need(device == "cpu" or 0 < peak <= c["memory_gate_bytes"], "12 GiB measured memory gate failed")
        step_seconds = [s for r in history for s in r["step_seconds"]]
        result = dict(version=VERSION, passed=not export_failures, status="PASS" if not export_failures else "EXPORT_FAILED",
            arm="multiview", seed=c["seed"], gpu_name=gpu_name, batch=batch, val_batch=1 if device == "cpu" else 4,
            optimizer_steps=counters["updates"] - counters["skips"],
            validation_shapes=["k24", "k36", "k36flip", "gold_k36flip"], config_sha256=h.sha(config_path),
            registration_sha256=identity["registration_sha256"], engineering_only=smoke, smoke=smoke, device=device,
            cpu_model=identity["cpu_model"], full_shape=device == "cuda", peak_reserved_bytes=peak,
            memory_gate_bytes=c["memory_gate_bytes"], runtime_pairing=pairing, healthy_rates=healthy_rates,
            cpu_cuda_initialized=h.torch.cuda.is_initialized() if device == "cpu" else None,
            counters=counters, steps=steps, seconds=time.monotonic() - start, runtime=runtime,
            step_seconds=step_seconds, checkpoint_sha256=h.sha(checkpoint), identity=identity, history=history,
            validation_seconds=durations, export_failures=export_failures, exposure="EXPOSED", clean_oof=False)
        h.need(device != "cpu" or result["cpu_cuda_initialized"] is False, "CPU smoke initialized CUDA")
        if not atomic_json(out / "summary.json", result, critical=False):
            export_failures.append("summary")
            result.update(passed=False, status="EXPORT_FAILED")
        atomic_json(out / "heartbeat.json", dict(status="complete" if not export_failures else "export_failed",
                                                exposure="EXPOSED", clean_oof=False), critical=False)
        return result
    except BaseException as exc:
        atomic_json(out / "failure.json", dict(status="FAILED", error=type(exc).__name__,
            reason="persistent_journal_failure" if isinstance(exc, JournalWriteError) else "fit_failure"), critical=False)
        atomic_json(out / "heartbeat.json", dict(status="failed", error=type(exc).__name__), critical=False)
        raise
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("smoke", "fit"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--review", type=Path)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--cpu-model", choices=("real", "tiny"), default="real")
    args = ap.parse_args(argv)
    if args.device == "cpu":
        hide_gpus_for_cpu()
    value = fit(args.config, args.out, device=args.device, smoke=args.command == "smoke", review=args.review,
                resume=args.resume, cpu_model=args.cpu_model)
    print(json.dumps({key: value[key] for key in ("passed", "arm", "seed", "device", "engineering_only", "steps", "seconds")}))
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
