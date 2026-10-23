"""H-B-105 production Small MIL members v1.0.0 (2026-10-12).

Changelog: isolated member-only derivative of mil_small v1.0.0; folds 1-4,
seeds 42/43, pinned H-B-105 science, final checkpoint only, no resume files.
All model, window, optimizer, loss and readout factories are unchanged.
"""
from training.arms.mil import *
from training.arms import mil as base


VERSION = "1.0.0"
RECIPE_HASHES = {42: "a669aef426473d368fa1847b5ec6e3d331ab1258246aa110dd684153d255c847",
                 43: "d1ca12fd12806a05fef0d31a7f0b4c155cb86779c275326031323db8186bf0a1"}
ARCH = "convnext_small.fb_in22k_ft_in1k"
SOURCE_FILES = base.SOURCE_FILES + ["training/arms/mil_small_members.py",
    "training/runners/mil_small_members.py", "tools/hb105_members.py", "tools/arm_lifecycle.py",
    "training/configs/hb105-small-s42-f0-v1.json", "training/configs/hb105-small-s43-f0-v1.json"]
def validate_recipe(spec):
    doc, config = spec.document, spec.document["config"]
    need(doc["family"] == "convnext-mil" and doc["seed"] in (42, 43)
         and doc["fold"] in (1, 2, 3, 4) and spec.run_kind == "member", "unsupported MIL member")
    for key, expected in (("windows", WINDOWS), ("sampler", SAMPLER), ("readout", READOUT),
                          ("targets", TARGETS), ("bn", {"policy": "none", "module_count": 0})):
        need(life.canonical_bytes(config[key]) == life.canonical_bytes(expected), key + " recipe differs")
    need(set(config["updates"]) == set(UPDATES) | {"loader_batches", "steps", "max_skips"}, "update fields differ")
    need(all(config["updates"][k] == v for k, v in UPDATES.items()), "update recipe differs")
    need(set(config["model"]) == {"arch", "kind", "weights", "weights_sha256", "initial_state",
                                   "initial_file_sha256", "initial_semantic_sha256"}, "model fields differ")
    need(config["model"]["arch"] == ARCH and config["model"]["kind"] == "finding_spatial_tokens_v1", "model differs")
    need(set(config["labels"]) == {"path", "name", "semantic_sha256", "pos_weight"}, "label fields differ")
    need(set(config["roots"]) == {"corpus", "manifest", "ordered", "groups", "canonical",
                                   "n_train", "n_val", "train_sha256", "val_sha256", "order_policy"}, "root fields differ")
    need(config["roots"]["order_policy"] == ("base-canonical" if doc["fold"] == 0 else "sorted-uid"),
         "registered fold order policy differs")
    need(doc["outputs"] == OUTPUTS and set(doc["inputs"]["sources"]) == set(SOURCE_FILES), "source/output inventory differs")
    # Paths used by this factory must all be pinned, not merely mentioned in prose.
    for group, paths in (("data", [config["roots"]["manifest"], config["labels"]["path"],
                                   config["model"]["weights"], config["model"]["initial_state"]]),
                         ("rosters", [config["roots"][k] for k in ("ordered", "groups", "canonical")])):
        need(set(doc["inputs"][group]) == set(paths), "factory input bindings differ")
    need(doc["inputs"]["data"][config["model"]["weights"]] == config["model"]["weights_sha256"]
         and doc["inputs"]["data"][config["model"]["initial_state"]] == config["model"]["initial_file_sha256"],
         "model input identity differs")
    return config


def validate_data(spec, root, corpus_dir):
    config = validate_recipe(spec)
    root, corpus_dir = Path(root), Path(corpus_dir)
    life.validate_inputs(spec, root)
    recipe_path = f"training/configs/hb105-small-s{spec.document['seed']}-f0-v1.json"
    recipe = life.ArmSpec((root / recipe_path).read_bytes())
    need(recipe.sha256 == RECIPE_HASHES[spec.document['seed']], "H-B-105 recipe hash differs")
    original = recipe.document["config"]
    for key in ("model", "windows", "sampler", "readout", "targets", "bn", "backend_policy"):
        need(config[key] == original[key], "H-B-105 recipe differs: " + key)
    for key in ("path", "name", "semantic_sha256"):
        need(config["labels"][key] == original["labels"][key], "H-B-105 labels differ")
    # The loaded source must match the staged source, even when launched elsewhere.
    source_root = Path(__file__).resolve().parents[2]
    for path, digest in spec.document["inputs"]["sources"].items():
        need(sha(source_root / path) == digest, "executing source differs from stage")
    manifest = read(corpus_dir / "manifest.json")
    need(sha(corpus_dir / "manifest.json") == spec.inventory[config["roots"]["manifest"]], "mounted corpus differs")
    for name in ("ids", "masks"):
        need(sha(corpus_dir / (name + ".npy")) == manifest[name + "_sha256"], "corpus index differs")
    corpus = common.Corpus(corpus_dir)
    need(corpus.depth == 64 and [tuple(s) for s in corpus.spec["slots"]] == common.SLOTS64, "corpus layout differs")
    roots = config["roots"]
    base = read(root / roots["canonical"])
    canonical = base["train"] + base["val"]
    with open(root / roots["groups"], newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train, val = validate_membership(corpus.ids.tolist(), read(root / roots["ordered"]), rows, canonical, roots["order_policy"])
    need((len(train), len(val)) == (roots["n_train"], roots["n_val"])
         and uid_hash(train) == roots["train_sha256"] and uid_hash(val) == roots["val_sha256"], "roster identity differs")
    labels, semantic = load_labels(root / config["labels"]["path"])
    need(semantic == config["labels"]["semantic_sha256"] and all(u in labels for u in train), "label identity/coverage differs")
    weights = positive_weights(labels, train)
    need(np.array_equal(weights, np.asarray(config["labels"]["pos_weight"], np.float32)), "positive weights differ")
    batches = len(train) // 8
    need(config["updates"]["loader_batches"] == batches and config["updates"]["steps"] == batches * 20
         and config["updates"]["max_skips"] == batches * 20 // 100, "update counts differ")
    return corpus, train, val, labels, weights


def run(spec, stage, corpus_dir, destination, *, mode, device, workers=0,
        note=None, evidence_root=None):
    """Final-checkpoint-only production execution (no resume). CUDA full fits require exact bound PASS.

    CPU smoke uses the real model and data, one training study/update and one
    validation study for each readout; it is explicitly a FP32 engineering run.
    Target smoke uses the registered batch of eight and the target FP16 policy.
    """
    need(mode in ("smoke", "full") and device in ("cpu", "cuda"), "execution mode differs")
    need(mode == "smoke" or device == "cuda", "full CPU training is unsupported")
    need(type(workers) is int and 0 <= workers <= 12, "worker count differs")
    life.validate_stage(spec, stage)
    corpus, train_ids, val_ids, labels, weights = validate_data(spec, stage, corpus_dir)
    config, doc = spec.document["config"], spec.document
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
        batch, epochs = (1 if device == "cpu" else 8), 1
        train_ids, val_ids = train_ids[:batch], val_ids[:1]
    shard_evidence = verify_shards(corpus, train_ids + val_ids if mode == "smoke" else None)
    seed_all(doc["seed"])
    selected_labels = {**{u: labels[u] for u in train_ids}, **{u: np.zeros(12, np.float32) for u in val_ids}}
    dataset = MILWindows(corpus_dir, train_ids, selected_labels, 12, True, doc["seed"])
    generator = torch.Generator().manual_seed(doc["seed"])
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, worker_init_fn=worker_init, generator=generator,
                        collate_fn=collate, pin_memory=device == "cuda")
    steps = len(loader) * epochs
    need(steps > 0 and (mode == "smoke" or steps == config["updates"]["steps"]), "loader steps differ")
    model = build_model(config["model"], stage).to(device)
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 3, 224, 224, device=device))
        model.train()
    opt, sched, scaler = optimizer_stack(model, steps, device, amp)
    identity = {"spec_sha256": spec.sha256, "mode": mode, "device": device, "observation": observed,
                "runtime": runtime, "batch": batch, "steps": steps, "workers": workers}
    start_epoch, history, counters = 0, [], {"updates": 0, "skips": 0}
    starting_updates, starting_skips = counters["updates"], counters["skips"]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    out = destination / "outputs"
    out.mkdir()
    start_time = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs):
            dataset.set_epoch(epoch)
            then = time.perf_counter()
            record = train_epoch(model, loader, opt, sched, scaler, torch.from_numpy(weights), device, amp, steps // 100, counters)
            record.update(epoch=epoch, epoch_seconds=time.perf_counter() - then, **counters)
            history.append(record)
    finally:
        # Persistent workers must stop before readouts (Windows mmap/spawn).
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    need(counters["updates"] == steps and counters["skips"] <= steps // 100, "fixed-final update count differs")
    provenance = {"tool_version": VERSION, "spec_sha256": spec.sha256, "verification_sha256": verification_hash,
                  "run_kind": spec.run_kind,
                  "identity": identity, "shard_verification": shard_evidence, "checkpoint_selection": "fixed-final",
                  "gold_access": "disabled", "metric_access": "disabled", "bn_module_count": 0,
                  "train_ids_ordered_sha256": uid_hash(train_ids), "eval_ids_ordered_sha256": uid_hash(val_ids)}
    checkpoint = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  "arch": config["model"]["arch"], "res": 224, "lab": TARGETS, "contract": corpus.contract, "k_eval": 24,
                  "norm": "imagenet", "window_policy": "within-slot-valid", "epoch": epochs - 1,
                  "tag": doc["arm"], "amp_train": amp, "amp_eval": amp, "fold": doc["fold"],
                  "label_set": config["labels"]["name"], "provenance": provenance, "input_policy": "center3",
                  "init": "timm-in22k", "init_weights_sha256": config["model"]["weights_sha256"],
                  "flip_tta": READOUT["tta"], "model_kind": "finding_spatial_tokens_v1"}
    with (out / "model_best.pt").open("xb") as stream:
        torch.save(checkpoint, stream)
    strict_load(model, torch.load(out / "model_best.pt", map_location="cpu", weights_only=False)["model"])
    validation_batches = 0
    for k, tta, filename in ((24, False, "oof_best.csv"), (36, False, "oof_k36.csv"), (36, True, "oof_k36_flip_tta.csv")):
        ds = MILWindows(corpus_dir, val_ids, selected_labels, k, False, doc["seed"])
        evaluation = DataLoader(ds, batch_size=4, shuffle=False, num_workers=min(workers, 2),
                                collate_fn=collate, worker_init_fn=worker_init)
        probabilities, ids, batches = evaluate(model, evaluation, device, amp, corpus, tta=tta)
        need(ids == val_ids, "readout order differs")
        life.create_new(out / filename, life.prediction_csv_bytes(ids, probabilities, TARGETS, id_column=UID))
        validation_batches += batches
    # Smoke evidence describes this invocation, never work inherited on resume.
    results = {"real_data": True, "train_batches": sum(r["batches"] for r in history[start_epoch:]),
               "validation_batches": validation_batches, "bn_batches": 0, "bn_module_count": 0,
               "optimizer_steps": counters["updates"] - starting_updates - counters["skips"] + starting_skips,
               "checkpoint_reloaded": True, "finite_outputs": True}
    summary = {"tool_version": VERSION, "status": "ENGINEERING_SMOKE" if mode == "smoke" else "FIXED_FINAL_COMPLETE",
               "scientific_status": "UNASSESSED", "run_kind": spec.run_kind, "results": results, "observation": observed,
               "elapsed_seconds": time.perf_counter() - start_time, "train": len(train_ids), "val": len(val_ids),
               "epochs": epochs, "attempted_updates": counters["updates"], "skipped_updates": counters["skips"],
               "spec_sha256": spec.sha256, "cpu_cuda_initialized": torch.cuda.is_initialized() if device == "cpu" else None}
    need(device != "cpu" or not summary["cpu_cuda_initialized"], "CPU execution initialized CUDA")
    for name, value in (("history.json", history), ("provenance.json", provenance), ("summary.json", summary)):
        write_new(out / name, value)
    write_new(out / "output-inventory.json", {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                                              for name in OUTPUTS if name != "output-inventory.json"})
    return summary
