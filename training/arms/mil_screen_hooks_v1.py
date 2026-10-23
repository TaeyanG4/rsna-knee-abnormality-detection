"""Shared MIL-S arm hooks v1.0.0 (2026-10-13): H-B-123/H-B-124 binding point.

Hooked derivative of mil_screen_v1.run/validate_data and the deployed
mil.train_epoch/evaluate. An Arm may wrap the window dataset, collate extra
per-study metadata, attach modules before device placement, route EVERY
forward (training, CPU/CUDA smoke, K24/K36/flip readouts, reload) through one
function and add provenance. With identity hooks the code is AST-equal to the
shared runner and its real CPU smoke is byte-identical (tests). Arms bind the
fresh shared control spec and differ from it by exactly ONE config key.
Rank objective (weak_rank_v1) is never trained here; use mil_screen_v1.
No GPU admission, no scientific selection, no deployed-source mutation.
"""
from training.arms.mil_screen_v1 import *
from training.arms import mil_screen_v1 as shared

VERSION = "1.0.0"
HOOK_SOURCES = ["training/arms/mil_screen_hooks_v1.py", "training/runners/mil_screen_hooks_v1.py",
                "tools/mil_screen_arm_build_v1.py"]


def control_path(seed):
    need(seed in (42, 43), "unregistered seed")
    return f"training/configs/hb122-control-s{seed}-f0-v1.json"


def control_spec(seed):
    spec = life.ArmSpec((ROOT / control_path(seed)).read_bytes())
    c = shared.validate_recipe(spec)
    need(spec.document["arm"] == f"HB122_CONTROL_S{seed}_F0" and "weak_rank_v1" not in c["updates"],
         "shared control spec differs")
    return spec


class Arm:
    """Identity hooks reproduce the shared control exactly. Subclass per arm.

    Subclasses set name/section/key/sources and implement contract(seed); any
    hook they need is overridden. Objects and their dataset/collate callables
    must be importable module-level definitions (Windows spawn workers pickle
    them). Wrapped datasets must expose .k, .ids and set_epoch() unchanged.
    """
    name = None          # member arm = f"{name}_S{seed}_F0"; None = shared control
    section = None       # config section holding the single registered change
    key = None           # its key; value must equal contract(seed)
    sources = ()         # extra bound sources: arm module, sidecars, protocol
    screen = "HB122_SHARED_V1"
    collate_fn = staticmethod(collate)

    def contract(self, seed):
        raise NotImplementedError

    def validate_recipe(self, spec):
        if self.name is None:
            c = shared.validate_recipe(spec)
            need("weak_rank_v1" not in c["updates"], "hooks runner never trains the rank objective")
            return c
        return single_difference(self, spec)

    def dataset(self, dataset, spec, stage, *, train):
        return dataset

    def attach(self, model, spec, stage):
        return model

    def probe(self, device):
        return None

    def move(self, extra, device):
        need(not extra, "identity arm received batch metadata")
        return None

    def forward(self, model, x, extra):
        return model(x)

    def provenance(self, spec):
        return {}


def expected_sources(arm, control):
    return set(control.document["inputs"]["sources"]) | set(HOOK_SOURCES) | set(arm.sources) | {
        control_path(control.document["seed"])}


def single_difference(arm, spec):
    """Arm spec == shared control spec except one key, arm name and sources."""
    d = spec.document
    need(spec.run_kind == "member" and d["seed"] in (42, 43) and d["fold"] == 0, "arm requires fold0 member")
    need(type(arm.name) is str and arm.name and arm.section and arm.key, "arm identity incomplete")
    control = control_spec(d["seed"])
    old = control.document
    need(arm.section in old["config"] and type(old["config"][arm.section]) is dict, "arm section differs")
    c = d["config"]
    scientific = life.parse_json(life.canonical_bytes(c))
    need(arm.key not in old["config"][arm.section], "arm key collides with control recipe")
    need(scientific[arm.section].pop(arm.key, None) == arm.contract(d["seed"]), "arm contract differs")
    need(scientific == old["config"], "shared scientific recipe differs")
    need(d["arm"] == f'{arm.name}_S{d["seed"]}_F0' and d["attempt"] == old["attempt"], "arm member name differs")
    for key in ("family", "seed", "fold", "target_environment", "outputs"):
        need(d[key] == old[key], "shared execution identity differs: " + key)
    need(d["inputs"]["data"] == old["inputs"]["data"] and d["inputs"]["rosters"] == old["inputs"]["rosters"]
         and d["inputs"]["controls"] == {}, "shared data differs")
    need(set(d["inputs"]["sources"]) == expected_sources(arm, control), "arm source inventory differs")
    need(all(d["inputs"]["sources"][p] == h for p, h in old["inputs"]["sources"].items()),
         "shared source binding differs")
    need(d["inputs"]["sources"][control_path(d["seed"])] == control.sha256, "control spec binding differs")
    target = [p for p, h in old["inputs"]["evidence"].items() if h == old["target_environment"]["evidence"]["declaration.json"]]
    need(len(target) == 1 and d["inputs"]["evidence"].get(target[0]) == old["inputs"]["evidence"][target[0]],
         "arm must bind the shared target declaration")
    return c


def validate_data_with(arm, spec, root, corpus_dir):
    c = arm.validate_recipe(spec)
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
    need(batches > 0, "empty training loader")
    return {"loss": total / batches, "batches": batches}


@torch.no_grad()
def evaluate_with(arm, model, loader, device, amp, corpus, *, tta=False):
    model.eval()
    predictions, ordered, batches = [], [], 0
    for x, _, ids, *extra in loader:
        extra = arm.move(extra, device)
        with autocast(device, amp):
            logits = arm.forward(model, x.to(device), extra)
        need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid evaluation logits")
        probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        if tta:
            flipped = flip_inputs(x, ids, corpus, loader.dataset.k)
            with autocast(device, amp):
                logits = arm.forward(model, flipped.to(device), extra)
            need(logits.shape == (len(ids), 12) and bool(torch.isfinite(logits).all()), "invalid TTA logits")
            probabilities = (probabilities + torch.sigmoid(logits.float()).cpu().numpy()) * np.float32(.5)
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
        torch.save(checkpoint, stream)
    strict_load(model, torch.load(out / "model_best.pt", map_location="cpu", weights_only=False)["model"])
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


def sampled_centers(dataset, index):
    """Exact window centers MILWindows.__getitem__ uses for this item/epoch.

    study_rng is a fresh hash-seeded random.Random and select_centers is its
    first consumer, so this replay never touches the live augmentation RNG.
    """
    need(isinstance(dataset, MILWindows), "registered MILWindows required")
    if dataset.corpus is None:
        dataset.corpus = common.Corpus(dataset.corpus_dir)
    uid, epoch = dataset.ids[index], int(dataset.epoch.item())
    rng = common.study_rng(dataset.seed, epoch, uid) if dataset.train else None
    centers = common.select_centers(dataset.corpus.mask(uid), 64, dataset.k, dataset.corpus.spec,
                                    "within-slot-valid", rng)
    return np.asarray(centers, np.int64)


class CenterWindows(torch.utils.data.Dataset):
    """MILWindows plus the exact sampled centers as a fourth item (int64[K]).

    Images, labels, IDs and every RNG stream are untouched; set_epoch/k/ids
    pass through. Base class for arms needing per-window metadata (H-B-124).
    """

    def __init__(self, inner):
        need(isinstance(inner, MILWindows), "registered MILWindows required")
        self.inner = inner

    @property
    def k(self):
        return self.inner.k

    @property
    def ids(self):
        return self.inner.ids

    def set_epoch(self, epoch):
        self.inner.set_epoch(epoch)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        x, y, uid = self.inner[index]
        return x, y, uid, torch.from_numpy(sampled_centers(self.inner, index))


def collate_centers(rows):
    xs, ys, ids, centers = zip(*rows)
    return torch.stack(xs), torch.stack(ys), list(ids), torch.stack(centers)


def load_arm(reference, spec):
    """'package.module:ATTR' -> Arm instance whose module file is a bound source."""
    import importlib
    module_name, _, attr = reference.partition(":")
    need(module_name and attr, "arm reference must be module:attribute")
    module = importlib.import_module(module_name)
    arm = getattr(module, attr)
    need(isinstance(arm, Arm), "arm reference is not an Arm")
    relative = Path(module.__file__).resolve().relative_to(ROOT).as_posix()
    sources = spec.document["inputs"]["sources"]
    need(arm.name is None or (relative in sources and relative in arm.sources), "arm module is not a bound source")
    return arm


IDENTITY = Arm()
