"""Shared MIL-S screen runner v1.0.0 (2026-10-13), H122/H123/H124 control.

Explicit derivative of mil_small.run; unchanged model, windows, loader,
optimizer, scheduler, scaler, resume and readout functions imported directly.
Full training changes only the optional registered weak_rank_v1 objective.
Control uses the deployed train_epoch function itself. CPU smoke uses two
deterministically selected training studies (engineering only) so ranking is
nonempty; CUDA smoke/full retain the ordinary first-eight/full training roster.
No deployed module mutation. No scientific selection or GPU admission here.
2026-10-13 claude-hb122 (crashed sol-2 stand-in), before first registration:
bind PROTOCOL-v1.1 (registered workers=2, scratch root); run/validate bodies
unchanged (H123 AST-binds them).
"""
from training.arms.mil_small import *
from training.arms import mil_small as deployed
from training.arms import weak_rank_v1 as ranking

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = deployed.SOURCE_FILES + [
    "training/arms/weak_rank_v1.py", "training/arms/mil_screen_v1.py",
    "training/runners/mil_screen_v1.py", "tools/hb122_build_v1.py",
    ".local/scratch/sol-2/hb122/PROTOCOL-v1.md", ".local/scratch/claude-hb122/PROTOCOL-v1.1.md",
    "training/configs/hb105-small-s42-f0-v1.json",
    "training/configs/hb105-small-s43-f0-v1.json"]


def validate_recipe(spec):
    d = spec.document
    need(spec.run_kind == "member" and d["seed"] in (42, 43) and d["fold"] == 0,
         "screen requires local fold0 member")
    old = life.ArmSpec((ROOT / f'training/configs/hb105-small-s{d["seed"]}-f0-v1.json').read_bytes()).document
    c = d["config"]
    scientific = life.parse_json(life.canonical_bytes(c))
    extra = scientific["updates"].pop("weak_rank_v1", None)
    need(extra is None or extra == ranking.CONTRACT, "unregistered loss change")
    kind = "CONTROL" if extra is None else "RANK"
    need(d["arm"] == f'HB122_{kind}_S{d["seed"]}_F0', "member objective/name differs")
    need(scientific == old["config"], "deployed scientific recipe differs")
    need(d["inputs"]["data"] == old["inputs"]["data"]
         and d["inputs"]["rosters"] == old["inputs"]["rosters"], "deployed data differs")
    need(set(d["inputs"]["sources"]) == set(SOURCE_FILES) and d["outputs"] == OUTPUTS,
         "source/output inventory differs")
    need(all(d["inputs"]["sources"][p] == h for p, h in old["inputs"]["sources"].items()),
         "deployed source binding differs")
    return c


def validate_data(spec, root, corpus_dir):
    c = validate_recipe(spec)
    root, corpus_dir = Path(root), Path(corpus_dir)
    life.validate_inputs(spec, root)
    for path, digest in spec.document["inputs"]["sources"].items():
        need(sha(ROOT / path) == digest, "executing source differs from stage")
    manifest = read(corpus_dir / "manifest.json")
    need(sha(corpus_dir / "manifest.json") == spec.inventory[c["roots"]["manifest"]], "mounted corpus differs")
    for name in ("ids", "masks"):
        need(sha(corpus_dir / (name + ".npy")) == manifest[name + "_sha256"], "corpus index differs")
    corpus = common.Corpus(corpus_dir)
    need(corpus.depth == 64 and [tuple(s) for s in corpus.spec["slots"]] == common.SLOTS64, "corpus layout differs")
    r = c["roots"]
    canonical = read(root / r["canonical"])
    with (root / r["groups"]).open(newline="", encoding="utf-8") as stream:
        groups = list(csv.DictReader(stream))
    train, val = validate_membership(corpus.ids.tolist(), read(root / r["ordered"]), groups,
                                    canonical["train"] + canonical["val"], r["order_policy"])
    need((len(train), len(val)) == (r["n_train"], r["n_val"])
         and uid_hash(train) == r["train_sha256"] and uid_hash(val) == r["val_sha256"], "roster identity differs")
    labels, semantic = load_labels(root / c["labels"]["path"])
    need(semantic == c["labels"]["semantic_sha256"] and all(u in labels for u in train), "label identity differs")
    weights = positive_weights(labels, train)
    need(np.array_equal(weights, np.asarray(c["labels"]["pos_weight"], np.float32)), "positive weights differ")
    return corpus, train, val, labels, weights


def component_map(spec, stage):
    with (Path(stage) / spec.document["config"]["roots"]["groups"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    need(all(r["component_id"] for r in rows), "missing component identity")
    return {r[UID]: r["component_id"] for r in rows if r["split"] == "train"}


def smoke_pair(train, labels, components):
    """First pair with coverage of every weak target; no validation selection."""
    y = np.stack([labels[u] for u in train])
    weak = [t for t, _ in ranking.MAPPING]
    for i, uid in enumerate(train):
        candidates = np.flatnonzero((np.abs(y[:, weak] - y[i, weak]) >= .5).all(1))
        for j in candidates:
            if components[uid] != components[train[j]]:
                return [uid, train[j]]
    raise life.ContractError("no all-target smoke pair available")


def train_epoch(model, loader, opt, sched, scaler, weights, device, amp, max_skips, counters,
                *, components, enabled):
    if not enabled:
        return deployed.train_epoch(model, loader, opt, sched, scaler, weights, device, amp, max_skips, counters)
    model.train()
    opt.zero_grad(set_to_none=True)
    total, batches = 0., 0
    coverage = {str(t): dict(eligible=0, low=0, high=0, fallback_strata=0, batches_with_pairs=0) for t, _ in ranking.MAPPING}
    for x, y, ids in loader:
        with autocast(device, amp):
            logits = model(x.to(device, non_blocking=True))
            need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid training logits")
            loss, counts = ranking.objective(logits, y.to(device, non_blocking=True), weights.to(device),
                                             [components[u] for u in ids], enabled=True)
        need(bool(torch.isfinite(loss)), "nonfinite training loss")
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
        counters["updates"] += 1
        total += loss.item()
        batches += 1
        for t, counts_t in counts.items():
            for key, count in counts_t.items():
                coverage[t][key] += count
            coverage[t]["batches_with_pairs"] += int(counts_t["eligible"] > 0)
    need(batches > 0, "empty training loader")
    return dict(loss=total / batches, batches=batches, pair_coverage=coverage)


def run(spec, stage, corpus_dir, destination, *, mode, device, workers=0, resume=None,
        note=None, evidence_root=None):
    need(mode in ("smoke", "full") and device in ("cpu", "cuda"), "execution mode differs")
    need(mode == "smoke" or device == "cuda", "full CPU training unsupported")
    need(type(workers) is int and 0 <= workers <= 2, "screen permits at most two workers")
    life.validate_stage(spec, stage)
    corpus, train_ids, val_ids, labels, weights = validate_data(spec, stage, corpus_dir)
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
    dataset = MILWindows(corpus_dir, train_ids, selected_labels, 12, True, doc["seed"])
    generator = torch.Generator().manual_seed(doc["seed"])
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, worker_init_fn=worker_init, generator=generator,
                        collate_fn=collate, pin_memory=device == "cuda")
    steps = len(loader) * epochs
    need(steps > 0 and (mode == "smoke" or steps == c["updates"]["steps"]), "loader steps differ")
    model = build_model(c["model"], stage).to(device)
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 3, 224, 224, device=device))
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
            record = train_epoch(model, loader, opt, sched, scaler, torch.from_numpy(weights), device, amp,
                                 steps // 100, counters, components=components,
                                 enabled="weak_rank_v1" in c["updates"])
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
        screen="HB122_SHARED_V1", objective=c["updates"].get("weak_rank_v1", "deployed_BCE"))
    checkpoint = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        arch=c["model"]["arch"], res=224, lab=TARGETS, contract=corpus.contract, k_eval=24,
        norm="imagenet", window_policy="within-slot-valid", epoch=epochs - 1, tag=doc["arm"],
        amp_train=amp, amp_eval=amp, fold=doc["fold"], label_set=c["labels"]["name"], provenance=provenance,
        input_policy="center3", init="timm-in22k", init_weights_sha256=c["model"]["weights_sha256"],
        flip_tta=READOUT["tta"], model_kind="finding_spatial_tokens_v1")
    with (out / "model_best.pt").open("xb") as stream:
        torch.save(checkpoint, stream)
    strict_load(model, torch.load(out / "model_best.pt", map_location="cpu", weights_only=False)["model"])
    validation_batches = 0
    for k, tta, filename in ((24, False, "oof_best.csv"), (36, False, "oof_k36.csv"), (36, True, "oof_k36_flip_tta.csv")):
        ds = MILWindows(corpus_dir, val_ids, selected_labels, k, False, doc["seed"])
        evaluation = DataLoader(ds, batch_size=4, shuffle=False, num_workers=min(workers, 2), collate_fn=collate, worker_init_fn=worker_init)
        probabilities, ids, batches = evaluate(model, evaluation, device, amp, corpus, tta=tta)
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
