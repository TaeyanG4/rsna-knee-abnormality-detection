"""H-B-122 registration/evidence tools v1.0.0; no GPU training launcher.

Member lifecycle is a transport/evidence mechanism, not scientific acceptance.
The shared control's scientific config is byte-equal to the deployed recipe.
Ranking recipes differ solely by the fixed weak_rank_v1 update extension.

2026-10-13 (claude-hb122 stand-in for crashed sol-2, before first registration):
scratch root moved to .local/scratch/claude-hb122; the sol-2 target declaration
is reused unchanged; every fit/smoke uses WORKERS=2 persistent loader workers
(PROTOCOL-v1.1), and the coverage audit replays that exact sampler stream.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"

from training.arms import lifecycle as life, mil_screen_v1 as mil
from tools import hb105_mil_small as old

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".local/scratch/claude-hb122"
TARGET = ROOT / ".local/scratch/sol-2/hb122/target-v1.json"
PAIRS = tuple((kind, seed) for kind in ("control", "rank") for seed in (42, 43))
CORPUS = ROOT / "storage/cache/derived/RAPTOR64_336_C140_S02"
# Registered loader worker count for every CPU/CUDA smoke and full fit (also the
# H123/H124 arms). Persistent workers draw the sampler base seed once, exactly
# like the deployed HB105 fits; workers=0 would redraw it every epoch.
WORKERS = 2


def key(kind, seed):
    mil.need((kind, seed) in PAIRS, "unknown screen member")
    return f"{kind}-s{seed}"


def spec_path(kind, seed):
    key(kind, seed)
    return ROOT / f"training/configs/hb122-{kind}-s{seed}-f0-v1.json"


def spec_for(kind, seed):
    return life.ArmSpec(spec_path(kind, seed).read_bytes())


def same_or_new(path, value):
    path = Path(path)
    raw = life.canonical_bytes(value)
    if path.exists():
        mil.need(path.read_bytes() == raw, "immutable artifact differs; new version required")
    else:
        life.create_new(path, raw)


def declare_target():
    # Small target identity probe only, never a fit or synthetic target attestation.
    path = TARGET
    mil.configure_runtime("fp16", "cuda", policy="strict")
    observed = mil.observation("cuda")
    mil.need(observed["runtime"]["gpu"] == "NVIDIA GeForce RTX 4070 Ti SUPER", "shared screen requires local 4070 Ti SUPER")
    same_or_new(path, observed)
    return dict(status="TARGET_DECLARED_NO_TRAINING", sha256=mil.sha(path))


def configure():
    target = TARGET
    profile, _ = life.environment_from_declaration(target, expected_sha256=mil.sha(target))
    results = []
    for kind, seed in PAIRS:
        recipe = ROOT / f"training/configs/hb105-small-s{seed}-f0-v1.json"
        doc = life.ArmSpec(recipe.read_bytes()).document
        if kind == "rank":
            doc["config"]["updates"]["weak_rank_v1"] = mil.ranking.CONTRACT
            doc.update(arm=f"HB122_RANK_RECIPE_S{seed}_F0", attempt="v1")
        # A separately named control recipe preserves the deployed bytes while
        # keeping lifecycle evidence disjoint from the source inventory.
        recipe = ROOT / f"training/configs/hb122-{kind}-recipe-s{seed}-f0-v1.json"
        same_or_new(recipe, life.ArmSpec.from_document(doc).document)
        doc.update(schema=life.MEMBER_SPEC_SCHEMA, run_kind="member", attempt="v1",
                   arm=f"HB122_{kind.upper()}_S{seed}_F0", target_environment=profile)
        doc["inputs"] = dict(sources={p: mil.sha(ROOT / p) for p in mil.SOURCE_FILES},
            data=doc["inputs"]["data"], rosters=doc["inputs"]["rosters"], controls={},
            evidence={p.relative_to(ROOT).as_posix(): mil.sha(p) for p in (recipe, target)})
        spec = life.ArmSpec.from_document(doc)
        mil.validate_data(spec, ROOT, CORPUS)
        same_or_new(spec_path(kind, seed), spec.document)
        results.append(dict(member=key(kind, seed), spec_sha256=spec.sha256))
    return dict(status="FOUR_SPECS_REGISTERED", specs=results)


def prepare(kind, seed):
    spec = spec_for(kind, seed)
    run = WORK / key(kind, seed)
    life.validate_inputs(spec, ROOT)
    same_or_new(run / "registered.json", spec.document)
    life.prepare(spec, ROOT, run / "stage")
    life.validate_stage(spec, run / "stage")
    mil.validate_data(spec, run / "stage", CORPUS)
    return dict(status="REGISTER_VALIDATE_PREPARE_PASS", member=key(kind, seed))


def admit(kind, seed, evidence="cpu"):
    spec = spec_for(kind, seed)
    run = WORK / key(kind, seed)
    name = "verification-pass.json" if evidence == "pass" else "verification.json"
    return life.admission_dry_run(spec, run / "stage", mil.read(run / evidence / name), run / evidence)


def finish(kind, seed, smoke=False):
    spec = spec_for(kind, seed)
    run = WORK / key(kind, seed)
    mode = "cpu" if smoke else "full"
    note = run / ("cpu/verification.json" if smoke else "pass/verification-pass.json")
    folder = run / ("cpu-export" if smoke else "export")
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / "export.json"
    if record.exists():
        result = mil.read(record)
        mil.need(mil.sha(folder / "artifacts.tar") == result["archive_sha256"], "export archive changed")
    else:
        result = old.finish(spec, run / mode / "outputs", note, folder / "artifacts.tar", folder / "receipt.json", smoke=smoke)
        same_or_new(record, result)
    dest = run / ("cpu-collected" if smoke else "collected")
    if not dest.exists():
        life.collect_artifacts(spec, folder / "artifacts.tar", dest, expected_receipt_sha256=result["receipt_sha256"])
    life.verify_artifact_receipt(spec, dest / "outputs", mil.read(dest / "receipt.json"),
                                expected_receipt_sha256=result["receipt_sha256"])
    old.check_outputs(spec, dest / "outputs", smoke=smoke)
    return dict(status="EXPORTED_COLLECTED", member=key(kind, seed), engineering_smoke=smoke)


def cpu_card(kind, seed):
    prepare(kind, seed)
    run = WORK / key(kind, seed)
    if not (run / "cpu").exists():
        command = [sys.executable, "-B", "-m", "training.runners.mil_screen_v1", "smoke",
            "--spec", str(spec_path(kind, seed)), "--stage", str(run / "stage"), "--corpus", str(CORPUS),
            "--device", "cpu", "--workers", str(WORKERS), "--out", str(run / "cpu")]
        subprocess.run(command, cwd=ROOT, check=True)
    result = admit(kind, seed)
    mil.need(result["status"] == "HOLD", "CPU evidence cannot grant target admission")
    finish(kind, seed, smoke=True)
    summary = mil.read(run / "cpu/outputs/summary.json")
    mil.need(summary["results"]["optimizer_steps"] == 1 and summary["cpu_cuda_initialized"] is False,
             "CPU smoke incomplete")
    card = dict(status="CPU_CARD_PASS", member=key(kind, seed), register=True, validate=True, prepare=True,
                admission="HOLD", export_collect=True, summary=summary)
    same_or_new(run / "cpu-card.json", card)
    return dict(status=card["status"], member=card["member"], admission="HOLD", results=summary["results"])


def sampler_batches(ids, seed, epochs, workers=WORKERS):
    """Replay the runner's training sampler stream: same DataLoader arguments
    (batch8, shuffle, drop_last, persistent iff workers>0, seeded generator).

    Dataset content never affects index draws, so a UID list reproduces the
    exact batch order of the registered fit without decoding any image.
    """
    import torch
    loader = mil.DataLoader(list(ids), batch_size=8, shuffle=True, drop_last=True, num_workers=workers,
                            persistent_workers=workers > 0, generator=torch.Generator().manual_seed(seed),
                            collate_fn=list)
    try:
        for epoch in range(epochs):
            for batch in loader:
                yield epoch, batch
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


def coverage():
    """All 20 registered batch streams, aggregate summaries only; never label rows."""
    import torch
    torch.set_num_threads(2)
    result = {}
    for seed in (42, 43):
        spec = spec_for("rank", seed)
        _, train, _, labels, _ = mil.validate_data(spec, ROOT, CORPUS)
        components = mil.component_map(spec, ROOT)
        counts = {str(t): dict(batches_with_pairs=0, total_pairs=0, within_pairs=0, low_pairs=0, high_pairs=0,
                               fallback_strata=0, max_pairs_in_batch=0) for t, _ in mil.ranking.MAPPING}
        batches, uid_draws = 0, 0
        for _, ids in sampler_batches(train, seed, 20):
            _, batch = mil.ranking.pair_sets(mil.np.stack([labels[u] for u in ids]), [components[u] for u in ids])
            batches += 1
            uid_draws += len(ids)
            for target, c in batch.items():
                counts[target]["batches_with_pairs"] += int(c["eligible"] > 0)
                counts[target]["total_pairs"] += c["eligible"]
                counts[target]["within_pairs"] += c["low"] + c["high"]
                counts[target]["low_pairs"] += c["low"]
                counts[target]["high_pairs"] += c["high"]
                counts[target]["fallback_strata"] += c["fallback_strata"]
                counts[target]["max_pairs_in_batch"] = max(counts[target]["max_pairs_in_batch"], c["eligible"])
        mil.need(batches == 8620 and uid_draws == 8620 * 8, "registered update count differs")
        mil.need(all(c["batches_with_pairs"] and c["max_pairs_in_batch"] <= 28 for c in counts.values()),
                 "no ranking exposure or pair bound exceeded")
        for c in counts.values():
            c.update(fraction_batches_with_pairs=round(c["batches_with_pairs"] / batches, 6),
                     mean_pairs_per_batch=round(c["total_pairs"] / batches, 6))
        result[str(seed)] = dict(batches=batches, workers=WORKERS, persistent_workers=WORKERS > 0,
                                 sampler="DataLoader(batch8,shuffle,drop_last,generator=seed)", targets=counts)
    same_or_new(WORK / "pair-coverage-v1.json", result)
    return dict(status="REAL_BATCH_COVERAGE_PASS", seeds=result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("declare", "configure", "prepare", "cpu-card", "coverage", "admit", "combine", "finish"))
    ap.add_argument("--kind", choices=("control", "rank"))
    ap.add_argument("--seed", type=int, choices=(42, 43))
    ap.add_argument("--evidence", choices=("cpu", "pass"), default="cpu")
    ap.add_argument("--engineering-smoke", action="store_true")
    ap.add_argument("--reviewer-name")
    ap.add_argument("--reviewer-host")
    a = ap.parse_args()
    if a.command in ("declare", "configure", "coverage"):
        result = {"declare": declare_target, "configure": configure, "coverage": coverage}[a.command]()
    else:
        run = WORK / key(a.kind, a.seed)
        if a.command == "admit":
            result = admit(a.kind, a.seed, a.evidence)
        elif a.command == "combine":
            mil.need(a.reviewer_name and a.reviewer_host, "independent reviewer identity required")
            result = old.combine(spec_for(a.kind, a.seed), run / "cpu", run / "target", run / "pass",
                                 a.reviewer_name, a.reviewer_host)
        elif a.command == "finish":
            result = finish(a.kind, a.seed, a.engineering_smoke)
        else:
            result = {"prepare": prepare, "cpu-card": cpu_card}[a.command](a.kind, a.seed)
    print(life.canonical_bytes(result).decode())


if __name__ == "__main__":
    main()
