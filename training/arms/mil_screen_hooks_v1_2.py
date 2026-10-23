"""MIL hooks v1.2.0. Changelog: completed phase progress and chunked checkpoint I/O.
Scientific functions derive from v1 unchanged except progress instrumentation.
The original hooks, runner, builder and all v1.1 stages remain intact.
"""
from training.arms.mil_screen_hooks_v1 import *
from tools.hb137_safety_v1_2 import heartbeat, save_checkpoint, load_checkpoint, checkpoint_sha
VERSION = "1.2.0"

def save_epoch(path, *, epoch, model, opt, sched, scaler, generator, history, counters, identity, device):
    payload = {"schema": "mil-epoch/v1", "identity": identity, "epoch": epoch,
               "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
               "scaler": scaler.state_dict(), "generator": generator.get_state(), "history": history,
               "counters": counters, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        save_checkpoint(payload, stream)
    write_new(path.with_suffix(".receipt.json"), {"schema": "mil-epoch-receipt/v1", "sha256": checkpoint_sha(path),
                                                "identity": identity, "epoch": epoch})


def restore_epoch(path, *, model, opt, sched, scaler, generator, identity, device, loader=None):
    path = Path(path)
    receipt = read(path.with_suffix(".receipt.json"))
    need(receipt["sha256"] == checkpoint_sha(path) and receipt["identity"] == identity, "resume receipt differs")
    ck = load_checkpoint(path, map_location="cpu", weights_only=False)
    need(ck["schema"] == "mil-epoch/v1" and ck["identity"] == identity and ck["epoch"] == receipt["epoch"], "resume identity differs")
    need((ck["cuda_rng"] is not None) == (device == "cuda"), "resume device differs")
    strict_load(model, ck["model"])
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    scaler.load_state_dict(ck["scaler"])
    # A persistent iterator consumes a worker base_seed only on construction.
    # Recreate it before restoring the shared sampler generator; its prefetched
    # indices are discarded by the next iterator reset. Otherwise a restart
    # consumes an extra base_seed and silently changes the next epoch's shuffle.
    if loader is not None and loader.persistent_workers and loader._iterator is None:
        iter(loader)
    generator.set_state(ck["generator"])
    random.setstate(ck["python_rng"])
    np.random.set_state(ck["numpy_rng"])
    torch.set_rng_state(ck["torch_rng"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck["epoch"] + 1, ck["history"], ck["counters"]


def save_latest_epoch(path, **kwargs):
    """Publish and verify the new pair before removing our previous pair.

    Only receipt-bound checkpoints in this invocation's own epochs directory
    are removed. A supplied external resume checkpoint is never removed.
    """
    path = Path(path).resolve()
    save_epoch(path, **kwargs)
    receipt = read(path.with_suffix(".receipt.json"))
    need(receipt["sha256"] == checkpoint_sha(path) and receipt["identity"] == kwargs["identity"],
         "new resume checkpoint failed verification")
    previous = path.parent / f"epoch-{kwargs['epoch'] - 1:03d}.pt"
    if previous.exists():
        need(not previous.is_symlink() and previous.resolve().parent == path.parent,
             "resume cleanup outside owned directory")
        old_receipt = previous.with_suffix(".receipt.json")
        old = read(old_receipt)
        need(old["identity"] == kwargs["identity"] and old["epoch"] == kwargs["epoch"] - 1
             and old["sha256"] == checkpoint_sha(previous), "previous resume checkpoint differs")
        previous.unlink()
        old_receipt.unlink()


def train_epoch_with(arm, model, loader, opt, sched, scaler, weights, device, amp, max_skips, counters):
    model.train()
    opt.zero_grad(set_to_none=True)
    total, batches = 0., 0
    lossf = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    for x, y, ids, *extra in loader:
        with autocast(device, amp):
            logits = arm.forward(model, x.to(device, non_blocking=True), arm.move(extra, device))
            need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid training logits")
            loss = lossf(logits.float(), y.to(device, non_blocking=True))
        need(bool(torch.isfinite(loss)), "nonfinite training loss")
        heartbeat("training_forward")
        scaler.scale(loss).backward()
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
        counters["updates"] += 1  # same attempted-update convention as legacy
        total += loss.item()
        batches += 1
        if device == "cuda":
            torch.cuda.synchronize()
        heartbeat("training_step", completed=counters["updates"])
    need(batches > 0, "empty training loader")
    return {"loss": total / batches, "batches": batches}


@torch.no_grad()
def evaluate_with(arm, model, loader, device, amp, corpus, *, tta=False, phase="validation"):
    model.eval()
    predictions, ordered, batches = [], [], 0
    for x, _, ids, *extra in loader:
        extra = arm.move(extra, device)
        with autocast(device, amp):
            logits = arm.forward(model, x.to(device), extra)
        need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid evaluation logits")
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        heartbeat(phase, completed=batches, part="forward")
        if tta:
            flipped = flip_inputs(x, ids, corpus, loader.dataset.k)
            with autocast(device, amp):
                logits = arm.forward(model, flipped.to(device), extra)
            need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid TTA logits")
            probabilities = (probabilities + torch.sigmoid(logits.float()).cpu().numpy()) * np.float32(.5)
            heartbeat(phase, completed=batches, part="flip")
        predictions.append(probabilities)
        ordered += ids
        batches += 1
    need(bool(predictions), "empty evaluation")
    return np.concatenate(predictions), ordered, batches


def run(spec, stage, corpus_dir, destination, *, arm, mode, device, workers=0, resume=None,
        note=None, evidence_root=None):
    need(mode in ("smoke", "full") and device in ("cpu", "cuda"), "execution mode differs")
    need(mode == "smoke" or device == "cuda", "full CPU training unsupported")
    need(type(workers) is int and 0 <= workers <= 2, "screen permits at most two workers")
    life.validate_stage(spec, stage)
    corpus, train_ids, val_ids, labels, weights = validate_data_with(arm, spec, stage, corpus_dir)
    c, doc = spec.document["config"], spec.document
    components = component_map(spec, stage)
    amp = "none" if device == "cpu" else "fp16"
    runtime = configure_runtime(amp, device, policy="strict")
    observed = observation(device)
    verification_hash = None
    if device == "cuda":
        life.check_target_observation(spec, observed)
    if mode == "full":
        need(note is not None and evidence_root is not None, "full fit requires evidence")
        verification_hash = life.require_execution_evidence(spec, stage, note, evidence_root, fresh_observation=observed)
    batch, epochs = 8, 20
    if mode == "smoke":
        batch, epochs = (2 if device == "cpu" else 8), 1
        train_ids = smoke_pair(train_ids, labels, components) if device == "cpu" else train_ids[:8]
        val_ids = val_ids[:1]
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
        arm.forward(model.eval(), torch.zeros(1, 1, 3, 224, 224, device=device), arm.probe(device))
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
    start_time = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs):
            dataset.set_epoch(epoch)
            then = time.perf_counter()
            record = train_epoch_with(arm, model, loader, opt, sched, scaler, torch.from_numpy(weights), device, amp,
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
        screen=arm.screen, **arm.provenance(spec), objective=c["updates"].get("weak_rank_v1", "deployed_BCE"))
    checkpoint = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        arch=c["model"]["arch"], res=224, lab=TARGETS, contract=corpus.contract, k_eval=24,
        norm="imagenet", window_policy="within-slot-valid", epoch=epochs - 1, tag=doc["arm"],
        amp_train=amp, amp_eval=amp, fold=doc["fold"], label_set=c["labels"]["name"], provenance=provenance,
        input_policy="center3", init="timm-in22k", init_weights_sha256=c["model"]["weights_sha256"],
        flip_tta=READOUT["tta"], model_kind="finding_spatial_tokens_v1")
    with (out / "model_best.pt").open("xb") as stream:
        save_checkpoint(checkpoint, stream)
    strict_load(model, load_checkpoint(out / "model_best.pt", map_location="cpu", weights_only=False)["model"])
    validation_batches = 0
    for k, tta, filename in ((24, False, "oof_best.csv"), (36, False, "oof_k36.csv"), (36, True, "oof_k36_flip_tta.csv")):
        ds = arm.dataset(MILWindows(corpus_dir, val_ids, selected_labels, k, False, doc["seed"]), spec, stage, train=False)
        evaluation = DataLoader(ds, batch_size=4, shuffle=False, num_workers=min(workers, 2), collate_fn=arm.collate_fn, worker_init_fn=worker_init)
        probabilities, ids, batches = evaluate_with(arm, model, evaluation, device, amp, corpus, tta=tta)
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
    need(device != "cpu" or not summary["cpu_cuda_initialized"], "CPU initialized CUDA")
    for name, value in (("history.json", history), ("provenance.json", provenance), ("summary.json", summary)):
        write_new(out / name, value)
    write_new(out / "output-inventory.json", {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                                             for name in OUTPUTS if name != "output-inventory.json"})
    return summary
