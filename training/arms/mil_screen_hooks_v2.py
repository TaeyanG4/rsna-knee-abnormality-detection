"""Shared MIL hooks runner v2.0.0, 2026-10-13.

Changelog: explicit resolution, accumulated microbatches at fixed effective
batch, bounded two-step target smoke with measured memory/time. Native v1 is
the first consumer. Frozen v1 and its controls remain unchanged.
"""
from training.arms.mil_screen_hooks_v1 import *
from training.arms.native_resolution_v1 import validate_data

VERSION = "2.0.0"


def train_epoch_accum(arm, model, loader, opt, sched, scaler, weights, device, amp, max_skips, counters):
    model.train()
    opt.zero_grad(set_to_none=True)
    total, batches, durations = 0., 0, []
    lossf = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    for x, y, ids, *extra in loader:
        need(not extra, "accumulation currently supports image-only arms")
        if device == "cuda":
            torch.cuda.synchronize()
        then = time.perf_counter()
        value = 0.
        for start in range(0, len(ids), arm.microbatch):
            stop = min(start + arm.microbatch, len(ids))
            with autocast(device, amp):
                logits = arm.forward(model, x[start:stop].to(device, non_blocking=True), None)
                need(logits.shape == (stop-start, 12) and bool(torch.isfinite(logits).all()), "invalid logits")
                loss = lossf(logits.float(), y[start:stop].to(device)) * ((stop-start) / len(ids))
            need(bool(torch.isfinite(loss)), "nonfinite loss")
            scaler.scale(loss).backward()
            value += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=amp != "fp16")
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if amp == "fp16" and device == "cuda" and scaler.get_scale() < before:
            counters["skips"] += 1
            need(counters["skips"] <= max_skips, "scientific optimizer skip limit exceeded")
        else:
            sched.step()
        counters["updates"] += 1
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - then)
        total += value
        batches += 1
    need(batches > 0, "empty training loader")
    return dict(loss=total / batches, batches=batches, step_seconds=durations)


def run(spec, stage, corpus_dir, destination, *, arm, mode, device, workers=0, resume=None,
        note=None, evidence_root=None):
    need(mode in ("smoke", "full") and device in ("cpu", "cuda"), "execution mode differs")
    need(mode == "smoke" or device == "cuda", "full CPU training unsupported")
    need(type(workers) is int and 0 <= workers <= 2, "screen permits at most two workers")
    life.validate_stage(spec, stage)
    corpus, train_ids, val_ids, labels, weights = validate_data(spec, stage, corpus_dir, engineering=mode == "smoke")
    c, doc = spec.document["config"], spec.document
    amp = "none" if device == "cpu" else "fp16"
    runtime = configure_runtime(amp, device, policy="strict")
    observed = observation(device)
    verification_hash = None
    if device == "cuda":
        capacity = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 11 * (1 << 30) / capacity))
        torch.cuda.reset_peak_memory_stats()
        life.check_target_observation(spec, observed)
    if mode == "full":
        need(note is not None and evidence_root is not None, "full fit requires evidence")
        verification_hash = life.require_execution_evidence(spec, stage, note, evidence_root, fresh_observation=observed)
    need(workers == 2, "registered two persistent workers required")
    resolution = c["windows"]["res"]
    batch, epochs = 8, 20
    if mode == "smoke":
        batch, epochs = (2 if device == "cpu" else 8), 1
        train_ids = train_ids[:2] if device == "cpu" else train_ids[:16]
        val_ids = val_ids[:1] if device == "cpu" else val_ids[:16]
    shard_evidence = verify_shards(corpus, train_ids + val_ids if mode == "smoke" else None)
    seed_all(doc["seed"])
    selected_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(12, np.float32) for u in val_ids}}
    dataset = arm.dataset(MILWindows(corpus_dir, train_ids, selected_labels, 12, True, doc["seed"]), spec, stage, train=True)
    generator = torch.Generator().manual_seed(doc["seed"])
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, worker_init_fn=worker_init, generator=generator,
                        collate_fn=arm.collate_fn, pin_memory=device == "cuda")
    steps = len(loader) * epochs
    need(steps > 0 and (mode == "smoke" or steps == c["updates"]["steps"]), "loader steps differ")
    model = arm.attach(build_model(c["model"], stage), spec, stage).to(device)
    with torch.no_grad():
        arm.forward(model.eval(), torch.zeros(1, 1, 3, resolution, resolution, device=device), arm.probe(device))
        model.train()
    opt, sched, scaler = optimizer_stack(model, steps, device, amp)
    identity = dict(spec_sha256=spec.sha256, mode=mode, device=device, observation=observed,
                    runtime=runtime, batch=batch, steps=steps, workers=workers)
    start_epoch, history, counters = 0, [], {"updates": 0, "skips": 0}
    if resume is not None:
        start_epoch, history, counters = restore_epoch(resume, model=model, opt=opt, sched=sched, scaler=scaler,
            generator=generator, identity=identity, device=device, loader=loader)
        need(0 <= start_epoch <= epochs and counters["updates"] == start_epoch * len(loader), "resume count differs")
    starting_updates, starting_skips = counters["updates"], counters["skips"]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    out, work = destination / "outputs", destination / "epochs"
    out.mkdir()
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs):
            dataset.set_epoch(epoch)
            then = time.perf_counter()
            record = train_epoch_accum(arm, model, loader, opt, sched, scaler, torch.from_numpy(weights), device, amp,
                                      steps // 100, counters)
            record.update(epoch=epoch, epoch_seconds=time.perf_counter() - then, **counters)
            history.append(record)
            save_latest_epoch(work / f"epoch-{epoch:03d}.pt", epoch=epoch, model=model, opt=opt, sched=sched,
                scaler=scaler, generator=generator, history=history, counters=counters, identity=identity, device=device)
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    need(counters["updates"] == steps and counters["skips"] <= steps // 100, "fixed-final update count differs")
    provenance = dict(tool_version=VERSION, spec_sha256=spec.sha256, verification_sha256=verification_hash,
        run_kind=spec.run_kind, identity=identity, shard_verification=shard_evidence, checkpoint_selection="fixed-final",
        gold_access="disabled", metric_access="disabled", bn_module_count=0,
        train_ids_ordered_sha256=uid_hash(train_ids), eval_ids_ordered_sha256=uid_hash(val_ids),
        engineering_targets="constant 0.5; no label rows opened" if mode == "smoke" else None,
        screen=arm.screen, **arm.provenance(spec), objective=c["updates"].get("weak_rank_v1", "deployed_BCE"))
    checkpoint = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        arch=c["model"]["arch"], res=resolution, lab=TARGETS, contract=corpus.contract, k_eval=24,
        norm="imagenet", window_policy="within-slot-valid", epoch=epochs - 1, tag=doc["arm"],
        amp_train=amp, amp_eval=amp, fold=doc["fold"], label_set=c["labels"]["name"], provenance=provenance,
        input_policy="center3", init="timm-in22k", init_weights_sha256=c["model"]["weights_sha256"],
        flip_tta=READOUT["tta"], model_kind="finding_spatial_tokens_v1")
    with (out / "model_best.pt").open("xb") as stream:
        torch.save(checkpoint, stream)
    strict_load(model, torch.load(out / "model_best.pt", map_location="cpu", weights_only=False)["model"])
    validation_batches = 0
    validation_seconds = {}
    for k, tta, filename in ((24, False, "oof_best.csv"), (36, False, "oof_k36.csv"), (36, True, "oof_k36_flip_tta.csv")):
        ds = arm.dataset(MILWindows(corpus_dir, val_ids, selected_labels, k, False, doc["seed"]), spec, stage, train=False)
        evaluation = DataLoader(ds, batch_size=4, shuffle=False, num_workers=min(workers, 2), collate_fn=arm.collate_fn, worker_init_fn=worker_init)
        if device == "cuda":
            torch.cuda.synchronize()
        readout_start = time.perf_counter()
        probabilities, ids, batches = evaluate_with(arm, model, evaluation, device, amp, corpus, tta=tta)
        if device == "cuda":
            torch.cuda.synchronize()
        validation_seconds[filename] = time.perf_counter() - readout_start
        need(ids == val_ids, "readout order differs")
        life.create_new(out / filename, life.prediction_csv_bytes(ids, probabilities, TARGETS, id_column=UID))
        validation_batches += batches
    results = dict(real_data=True, train_batches=sum(r["batches"] for r in history[start_epoch:]),
        validation_batches=validation_batches, bn_batches=0, bn_module_count=0,
        optimizer_steps=counters["updates"] - starting_updates - counters["skips"] + starting_skips,
        checkpoint_reloaded=True, finite_outputs=True)
    summary = dict(tool_version=VERSION, status="ENGINEERING_SMOKE" if mode == "smoke" else "FIXED_FINAL_COMPLETE",
        scientific_status="UNASSESSED", run_kind=spec.run_kind, results=results, observation=observed,
        elapsed_seconds=time.perf_counter() - start_time, train=len(train_ids), val=len(val_ids), epochs=epochs,
        attempted_updates=counters["updates"], skipped_updates=counters["skips"], spec_sha256=spec.sha256,
        cpu_cuda_initialized=torch.cuda.is_initialized() if device == "cpu" else None)
    summary.update(resolution=resolution, microbatch=arm.microbatch,
        validation_seconds=validation_seconds,
        engineering_targets=mode == "smoke", step_seconds=[v for r in history for v in r["step_seconds"]],
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None)
    if device == "cuda":
        need(summary["peak_reserved_bytes"] <= 11 * (1 << 30), "11 GiB ceiling exceeded")
    need(device != "cpu" or not summary["cpu_cuda_initialized"], "CPU initialized CUDA")
    for name, value in (("history.json", history), ("provenance.json", provenance), ("summary.json", summary)):
        write_new(out / name, value)
    write_new(out / "output-inventory.json", {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                                             for name in OUTPUTS if name != "output-inventory.json"})
    return summary

