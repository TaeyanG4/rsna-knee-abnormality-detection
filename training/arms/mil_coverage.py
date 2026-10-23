"""H-B-106 MIL coverage preset v1.0.0 (2026-10-11).

Changelog: isolated sampler-only EXP185 contrast; no shared MIL source changes.
Reuse the frozen model, optimizer, transforms, loss and readouts. Fixed-final
storage writes no intermediate checkpoints; there is no wall-clock limit.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.arms import lifecycle as life, mil
from training.arms.mil_registration import control_records

VERSION = "1.0.0"
METADATA = "training/configs/mil-coverage-metadata-v1.json"
PRESET = {"preset": "coverage", "metadata": METADATA,
          "allocation": "one-per-available-group-then-control-shuffle-without-replacement",
          "fallback_credit": False, "augmentation_rng": "control-post-sampling",
          "checkpoint_retention": "fixed-final-only"}
EXTRA_SOURCES = ["training/arms/mil_coverage.py", "training/runners/mil_coverage.py",
                 "tools/hb106_mil_coverage.py", "tools/arm_lifecycle.py", "tools/exp180_k12_coverage.py"]
GROUPS = ("sagittal_fluid", "sagittal_other", "coronal_fluid", "coronal_other", "axial")


def populations(mask, slots):
    """Exactly CONTROL's within-slot-valid population, with provenance groups."""
    mask = np.asarray(mask)
    mil.need(mask.shape == (64,) and np.isin(mask, (0, 1)).all(), "invalid coverage mask")
    mil.need(len(slots) == 5, "five coverage slots required")
    pool, groups, start = [], [], 0
    for group, slot, (plane, _, size) in zip(GROUPS, slots, mil.common.SLOTS64):
        mil.need(slot["start"] == start and slot["stop"] == start + size
                 and slot["plane"] == plane and slot["group"] == group and type(slot["fallback"]) is bool,
                 "coverage slot geometry differs")
        centres = [c for c in range(start + 1, start + size - 1) if mask[c-1:c+2].all()]
        pool.extend(centres)
        if slot["series"] is not None and not slot["fallback"] and centres:
            groups.append((slot["group"], centres))
        start += size
    mil.need(len({g for g, _ in groups}) == len(groups), "duplicate coverage group")
    return pool, groups


def coverage_centres(mask, slots, k, rng):
    """Reserve one/group, CONTROL shuffle the remainder, then shuffle bag.

    The caller's RNG is advanced exactly as CONTROL, keeping AUG1 parameter
    draws paired. Coverage uses a clone of the SAME initial windows stream.
    Empty padding handles a population smaller than K without duplicates.
    """
    mil.need(type(k) is int and k > 0, "invalid K")
    pool, groups = populations(mask, slots)
    mil.need(k >= len(groups), "K cannot cover available groups")
    draw = random.Random()
    draw.setstate(rng.getstate())
    mil.common.select_centers(mask, 64, k, {"slots": mil.common.SLOTS64}, "within-slot-valid", rng)
    selected = [int(draw.choice(cs)) for _, cs in groups]
    remaining = [c for c in pool if c not in selected]
    draw.shuffle(remaining)  # same uniform shuffle operation as CONTROL
    selected.extend(remaining[:k-len(selected)])
    draw.shuffle(selected)
    return selected + [-1] * (k-len(selected))


class CoverageWindows(mil.MILWindows):
    """Per-instance metadata, spawn safe; eval delegates byte-for-byte to MIL."""
    def __init__(self, *args, metadata, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = mil.read(metadata)["studies"] if self.train else None

    def __getitem__(self, index):
        if not self.train:
            return super().__getitem__(index)
        if self.corpus is None:
            self.corpus = mil.common.Corpus(self.corpus_dir)
        uid, epoch = self.ids[index], int(self.epoch.item())
        mask, row = self.corpus.mask(uid), self.rows[uid]
        mil.need(life.content_sha256(mask.astype(int).tolist()) == row["mask_sha256"], "coverage mask changed")
        rng = mil.common.study_rng(self.seed, epoch, uid)
        centers = coverage_centres(mask, row["slots"], self.k, rng)
        x = mil.scale_windows(self.corpus.volume(uid), centers)
        intensity = mil.common.study_rng(self.seed, epoch, uid, "intensity").random()
        x = np.clip(x * (1 + (intensity-.5)*.20), 0, 1)
        x = mil.augment_windows(x, centers, rng, self.corpus.spec)
        mean = np.array([.485, .456, .406], np.float32).reshape(3, 1, 1)
        std = np.array([.229, .224, .225], np.float32).reshape(3, 1, 1)
        x = (x-mean)/std
        mil.need(x.dtype == np.float32 and np.isfinite(x).all(), "invalid coverage windows")
        return torch.from_numpy(x), torch.from_numpy(self.labels[uid]), uid


def base_spec(spec):
    """Project only the explicit extension away for frozen MIL contract checks.

    This projection never grants execution evidence: admission and all output
    identities always bind the full coverage spec, including metadata/sources.
    """
    d = spec.document
    mil.need(spec.run_kind == "experiment" and d["fold"] == 0 and d["seed"] in (42, 43), "coverage pair differs")
    mil.need(d["config"]["sampler"] == {**mil.SAMPLER, **PRESET}, "coverage preset differs")
    mil.need(set(d["inputs"]["sources"]) == set(mil.SOURCE_FILES + EXTRA_SOURCES), "coverage sources differ")
    d["config"]["sampler"] = dict(mil.SAMPLER)
    for p in EXTRA_SOURCES:
        d["inputs"]["sources"].pop(p)
    mil.need(METADATA in d["inputs"]["data"], "unbound coverage metadata")
    d["inputs"]["data"].pop(METADATA)
    result = life.ArmSpec.from_document(d)
    mil.validate_recipe(result)
    return result


def validate_data(spec, root, corpus_dir):
    root = Path(root)
    life.validate_inputs(spec, root)
    projected = base_spec(spec)
    executing = Path(__file__).resolve().parents[2]
    for p, h in spec.document["inputs"]["sources"].items():
        mil.need(mil.sha(executing/p) == h, "executing coverage source differs")
    corpus, train, val, labels, weights = mil.validate_data(projected, root, corpus_dir)
    summary, provenance = control_records(spec, root)
    for record in (summary, provenance):
        mil.need(record["exp185"]["model_kind"] == "finding_spatial_tokens_v1"
                 and record["exp185"]["model_sha256"] == spec.document["inputs"]["sources"]["training/src/raptor_exp185.py"],
                 "paired control must be EXP185 MIL")
    rows = mil.read(root/METADATA)
    mil.need(rows["schema"] == "mil-coverage-metadata/v1" and set(rows["studies"]) == set(train), "coverage roster differs")
    mil.need(set(rows["source_sha256"]) == {"manifest.json", "ids.npy", "masks.npy", "picks.jsonl"}, "metadata provenance differs")
    for name, digest in rows["source_sha256"].items():
        mil.need(digest == mil.sha(Path(corpus_dir)/name), "metadata corpus differs")
    mil.need(rows["train_sha256"] == mil.uid_hash(train), "metadata train identity differs")
    for uid in train:
        mask, row = corpus.mask(uid), rows["studies"][uid]
        mil.need(life.content_sha256(mask.astype(int).tolist()) == row["mask_sha256"], "coverage mask binding differs")
        populations(mask, row["slots"])
    return corpus, train, val, labels, weights


def run(spec, stage, corpus_dir, destination, *, mode, device, workers=0, note=None, evidence_root=None):
    """Explicit isolated MIL run; unchanged training math, final checkpoint only."""
    mil.need(mode in ("smoke", "full") and device in ("cpu", "cuda"), "invalid execution")
    mil.need(mode == "smoke" or device == "cuda", "full CPU fit unsupported")
    mil.need(type(workers) is int and 0 <= workers <= 12, "invalid worker count")
    life.validate_stage(spec, stage)
    corpus, train, val, labels, weights = validate_data(spec, stage, corpus_dir)
    d, c = spec.document, spec.document["config"]
    amp = "none" if device == "cpu" else "fp16"
    runtime = mil.configure_runtime(amp, device, policy="strict")
    observed = mil.observation(device)
    verification_hash = None
    if device == "cuda":
        life.check_target_observation(spec, observed)
    if mode == "full":
        mil.need(note is not None and evidence_root is not None, "full fit requires PASS evidence")
        verification_hash = life.require_execution_evidence(spec, stage, note, evidence_root, fresh_observation=observed)
    batch, epochs = 8, 20
    if mode == "smoke":
        batch, epochs = (1 if device == "cpu" else 8), 1
        train, val = train[:batch], val[:1]
    shard_evidence = mil.verify_shards(corpus, train+val if mode == "smoke" else None)
    mil.seed_all(d["seed"])
    selected_labels = {**{u: labels[u] for u in train}, **{u: np.zeros(12, np.float32) for u in val}}
    dataset = CoverageWindows(corpus_dir, train, selected_labels, 12, True, d["seed"], metadata=Path(stage)/METADATA)
    generator = torch.Generator().manual_seed(d["seed"])
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, worker_init_fn=mil.worker_init, generator=generator,
                        collate_fn=mil.collate, pin_memory=device == "cuda")
    steps = len(loader)*epochs
    mil.need(steps > 0 and (mode == "smoke" or steps == c["updates"]["steps"]), "step count differs")
    model = mil.build_model(c["model"], stage).to(device)
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, 3, 224, 224, device=device))
        model.train()
    opt, sched, scaler = mil.optimizer_stack(model, steps, device, amp)
    identity = dict(spec_sha256=spec.sha256, mode=mode, device=device, observation=observed,
                    runtime=runtime, batch=batch, steps=steps, workers=workers)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    out = destination/"outputs"
    out.mkdir()
    history, counters = [], {"updates": 0, "skips": 0}
    start = time.perf_counter()
    try:
        for epoch in range(epochs):
            dataset.set_epoch(epoch)
            then = time.perf_counter()
            record = mil.train_epoch(model, loader, opt, sched, scaler, torch.from_numpy(weights), device, amp, steps//100, counters)
            record.update(epoch=epoch, epoch_seconds=time.perf_counter()-then, **counters)
            history.append(record)
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    mil.need(counters["updates"] == steps and counters["skips"] <= steps//100, "fixed-final updates differ")
    provenance = dict(tool_version=VERSION, spec_sha256=spec.sha256, verification_sha256=verification_hash,
        run_kind=spec.run_kind, identity=identity, shard_verification=shard_evidence,
        checkpoint_selection="fixed-final", checkpoint_retention="fixed-final-only",
        gold_access="disabled", metric_access="disabled", bn_module_count=0,
        train_ids_ordered_sha256=mil.uid_hash(train), eval_ids_ordered_sha256=mil.uid_hash(val), sampler=PRESET)
    checkpoint = dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        arch=mil.ARCH, res=224, lab=mil.TARGETS, contract=corpus.contract, k_eval=24, norm="imagenet",
        window_policy="within-slot-valid", epoch=epochs-1, tag=d["arm"], amp_train=amp, amp_eval=amp,
        fold=0, label_set=c["labels"]["name"], provenance=provenance, input_policy="center3", init="timm-in22k",
        init_weights_sha256=c["model"]["weights_sha256"], flip_tta=mil.READOUT["tta"], model_kind="finding_spatial_tokens_v1")
    with (out/"model_best.pt").open("xb") as f:
        torch.save(checkpoint, f)
    mil.strict_load(model, torch.load(out/"model_best.pt", map_location="cpu", weights_only=False)["model"])
    validation_batches = 0
    for k, tta, filename in ((24,False,"oof_best.csv"),(36,False,"oof_k36.csv"),(36,True,"oof_k36_flip_tta.csv")):
        ds = mil.MILWindows(corpus_dir, val, selected_labels, k, False, d["seed"])
        evaluation = DataLoader(ds, batch_size=4, shuffle=False, num_workers=min(workers,2), collate_fn=mil.collate, worker_init_fn=mil.worker_init)
        probabilities, ids, batches = mil.evaluate(model, evaluation, device, amp, corpus, tta=tta)
        mil.need(ids == val, "readout order differs")
        life.create_new(out/filename, life.prediction_csv_bytes(ids, probabilities, mil.TARGETS, id_column=mil.UID))
        validation_batches += batches
    results = dict(real_data=True, train_batches=sum(r["batches"] for r in history), validation_batches=validation_batches,
        bn_batches=0, bn_module_count=0, optimizer_steps=steps-counters["skips"], checkpoint_reloaded=True, finite_outputs=True)
    summary = dict(tool_version=VERSION, status="ENGINEERING_SMOKE" if mode=="smoke" else "FIXED_FINAL_COMPLETE",
        scientific_status="UNASSESSED", run_kind=spec.run_kind, results=results, observation=observed,
        elapsed_seconds=time.perf_counter()-start, train=len(train), val=len(val), epochs=epochs,
        attempted_updates=steps, skipped_updates=counters["skips"], spec_sha256=spec.sha256,
        cpu_cuda_initialized=torch.cuda.is_initialized() if device=="cpu" else None)
    mil.need(device != "cpu" or not summary["cpu_cuda_initialized"], "CPU initialized CUDA")
    for name, value in (("history.json",history),("provenance.json",provenance),("summary.json",summary)):
        mil.write_new(out/name, value)
    mil.write_new(out/"output-inventory.json", {n:dict(sha256=mil.sha(out/n),bytes=(out/n).stat().st_size) for n in mil.OUTPUTS if n!="output-inventory.json"})
    return summary
