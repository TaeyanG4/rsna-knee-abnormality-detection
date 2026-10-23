"""EXP180 v1.0.1: CPU-only replay of registered K12 training centre draws.

Reads metadata, masks, and study-ID strata; never opens volumes or labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "src"))
from raptor_common import select_centers, study_rng, train_centers  # noqa: E402

VERSION = "1.0.1"
UID = "StudyInstanceUID"
GROUPS = ("sagittal_fluid", "sagittal_other", "coronal_fluid", "coronal_other", "axial")
REGISTERED_LAYOUTS = {
    64: (("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)),
    96: (("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)),
}
R64 = ROOT / "storage/cache/derived/RAPTOR64_336_C140_S02"
E143 = ROOT / "storage/cache/derived/EXP143_RAPTOR96_384_C140_S02"
E143_PROTOCOL = ROOT / "artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json"
ROSTERS = ROOT / "artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1"
STRATA = ROOT / ".local/astra-improve/oof-error-v2-run-001/strata_study_ids.json"
SERIES = ROOT / "storage/competition/train_series.csv"
OUT = ROOT / "artifacts/experiments/EXP180_K12_COVERAGE/result.json"
RECIPES = (
    ("l2_radr50", "raptor64", (42, 43), 20, "within-slot-valid"),
    ("convnext_t_l2", "raptor64", (42,), 20, "within-slot-valid"),
    ("coatnet_exp143_v2", "exp143", (42,), 16, "legacy"),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_rosters(directory: Path):
    folds, val_count, train_count = {}, Counter(), Counter()
    common = None
    for fold in range(5):
        path = directory / f"fold{fold}" / "ordered-ids.json"
        obj = read_json(path)
        train, val = list(obj["train"]), list(obj["val"])
        require(len(train) == len(set(train)) and len(val) == len(set(val)), "duplicate roster UID")
        require(not set(train) & set(val), "train/validation overlap")
        whole = set(train) | set(val)
        require(len(whole) == 4343 and (common is None or whole == common), "roster cohort differs")
        common = whole
        train_count.update(train)
        val_count.update(val)
        folds[fold] = train
    require(set(val_count.values()) == {1} and set(train_count.values()) == {4},
            "expected four train exposures and one validation exposure per UID")
    return folds, common


def read_series_flags(path: Path, cohort: set[str]):
    flags = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require({UID, "SeriesInstanceUID", "Fluid_Sensitive"} <= set(reader.fieldnames or ()),
                "series metadata columns missing")
        for row in reader:
            uid = row[UID]
            if uid not in cohort:
                continue
            raw = row["Fluid_Sensitive"]
            require(raw in ("0", "1", "0.0", "1.0"), "unknown series contrast flag")
            key = (uid, row["SeriesInstanceUID"])
            value = int(float(raw))
            require(key not in flags or flags[key] == value, "conflicting series contrast flag")
            flags[key] = value
    return flags


def provenance(spec, selected, mask, uid, flags):
    """One cache slot is credited only to its own requested contrast and series."""
    if "depth" in spec:
        require(tuple(map(tuple, spec["slots"])) == REGISTERED_LAYOUTS.get(spec["depth"]),
                "cache slot layout mismatch")
    sizes = [int(slot[2]) for slot in spec["slots"]]
    require(len(sizes) == 5 and sum(sizes) == len(mask) and len(selected) == 5,
            "cache slot layout mismatch")
    require(all(x in (0, 1) for x in mask), "nonbinary cache mask")
    slots, start = [], 0
    for i, ((plane, fluid, size), item) in enumerate(zip(spec["slots"], selected)):
        item = item or {}
        series = item.get("series")
        actual = item.get("protocol_flag")
        if series is not None:
            if actual is None:
                require((uid, series) in flags, "selected series lacks contrast metadata")
                actual = flags[(uid, series)]
            else:
                actual = int(actual)
                if (uid, series) in flags:
                    require(flags[(uid, series)] == actual, "receipt/series contrast disagreement")
        fallback = series is not None and int(fluid) in (0, 1) and actual != int(fluid)
        slots.append(dict(group=GROUPS[i], start=start, stop=start + int(size),
                          series=series, fallback=fallback, plane=plane))
        start += int(size)
    return slots


def load_raptor(directory: Path, cohort: set[str], flags):
    manifest = read_json(directory / "manifest.json")
    require(manifest["contract"] == "raptor64_336_c140_s02" and manifest["spec"]["depth"] == 64
            and manifest.get("finished"), "RAPTOR64 manifest differs")
    ids = np.load(directory / "ids.npy", allow_pickle=False).astype(str).tolist()
    masks = np.load(directory / "masks.npy", allow_pickle=False)
    require(len(ids) == len(set(ids)) == len(masks) and set(cohort) <= set(ids),
            "RAPTOR64 UID/mask coverage differs")
    rows = {uid: i for i, uid in enumerate(ids)}
    picks = {}
    with (directory / "picks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["uid"] in cohort:
                require(item["uid"] not in picks and item["split"] == "train", "RAPTOR64 picks differ")
                picks[item["uid"]] = item["slots"]
    require(set(picks) == cohort, "RAPTOR64 picks missing roster UIDs")
    data = {}
    for uid in cohort:
        mask = np.asarray(masks[rows[uid]], dtype=np.uint8)
        data[uid] = (mask, provenance(manifest["spec"], picks[uid], mask, uid, flags))
    return manifest["spec"], data, dict(cache_ids=len(ids), cohort_ids=len(cohort),
        metadata_sha256={name: sha(directory / name) for name in ("manifest.json", "ids.npy", "masks.npy", "picks.jsonl")})


def load_exp143(directory: Path, cohort: set[str], flags, protocol_path=E143_PROTOCOL):
    manifest = read_json(directory / "manifest.json")
    require(manifest["contract"] == "exp143_raptor96_384_c140_s02" and
            manifest["spec"]["depth"] == 96 and manifest.get("finished"), "EXP143 manifest differs")
    ids = manifest["ids"]
    require(len(ids) == len(set(ids)) and set(cohort) <= set(ids), "EXP143 UID coverage differs")
    protocol = read_json(protocol_path)
    train_ids, gold_ids = set(protocol["train_ids"]), set(protocol["excluded_gold_ids"])
    require(len(train_ids) == 4347 and len(gold_ids) == 58 and not train_ids & gold_ids and
            train_ids | gold_ids == set(ids) and set(cohort) <= train_ids and
            len(train_ids - cohort) == 4, "EXP143 v2 training cohort differs")
    require(protocol["recipe"]["train_k"] == 12 and protocol["recipe"]["epochs"] == 16 and
            protocol["recipe"]["seed_local"] == 42 and protocol["recipe"]["window_policy"] == "legacy",
            "EXP143 v2 recipe differs")
    rows = {uid: i for i, uid in enumerate(ids)}
    receipt_hash = hashlib.sha256()
    data = {}
    for uid in sorted(cohort):
        path = directory / f"study_{rows[uid]:05d}.json"
        raw = path.read_bytes()
        receipt_hash.update(len(raw).to_bytes(8, "big"))
        receipt_hash.update(raw)
        receipt = json.loads(raw)
        require(receipt["uid"] == uid and len(receipt["mask"]) == 96, "EXP143 receipt differs")
        mask = np.asarray(receipt["mask"], dtype=np.uint8)
        data[uid] = (mask, provenance(manifest["spec"], receipt["slots"], mask, uid, flags))
    return manifest["spec"], data, dict(cache_ids=len(ids), cohort_ids=len(cohort),
        registered_train_ids=len(train_ids), extra_registered_train_ids=len(train_ids - cohort),
        excluded_gold_ids=len(gold_ids), extra_cache_ids=len(set(ids) - cohort),
        receipt_stream_sha256=receipt_hash.hexdigest(),
        metadata_sha256={"manifest.json": sha(directory / "manifest.json"),
                         "protocol-v2/manifest.json": sha(protocol_path)})


def slot_index(center: int, slots):
    return next((i for i, slot in enumerate(slots) if slot["start"] <= center < slot["stop"]), None)


def availability(mask, slots):
    """Reason for absent coverage, and whether an independent valid triplet exists."""
    state = {}
    for slot in slots:
        start, stop = slot["start"], slot["stop"]
        valid = any(np.all(mask[c - 1:c + 2] > 0) for c in range(start + 1, stop - 1))
        if slot["series"] is None:
            reason = "missing_series"
        elif slot["fallback"]:
            reason = "fallback_only"
        elif not valid:
            reason = "no_valid_triplet"
        else:
            reason = "available"
        state[slot["group"]] = reason
    return state


def audit_bag(centers, mask, slots, available):
    """Count only complete, valid, nonfallback windows as group coverage."""
    covered = set()
    event = Counter()
    distinct = {int(c) for c in centers if c >= 0}
    event["duplicate_centres"] = sum(c >= 0 for c in centers) - len(distinct)
    event["bags_with_duplicates"] = int(event["duplicate_centres"] > 0)
    for raw in centers:
        c = int(raw)
        if c < 0:
            event["empty_windows"] += 1
            continue
        indices = {slot_index(x, slots) for x in (c - 1, c, c + 1)}
        if len(indices) != 1 or None in indices:
            event["boundary_windows"] += 1
            real = {slots[i]["series"] for x in (c - 1, c, c + 1)
                    if 0 <= x < len(mask) and mask[x] > 0
                    for i in [slot_index(x, slots)] if i is not None and slots[i]["series"] is not None}
            if len(real) > 1:
                event["mixed_series_windows"] += 1
            continue
        slot = slots[next(iter(indices))]
        if not np.all(mask[c - 1:c + 2] > 0):
            event["invalid_windows"] += 1
        elif slot["fallback"]:
            event["fallback_windows"] += 1
        elif slot["series"] is not None:
            covered.add(slot["group"])
    omitted = {group for group, reason in available.items() if reason == "available" and group not in covered}
    event["bags_omitting_any"] = int(bool(omitted))
    return event, covered, omitted


def blank():
    return {"studies": 0, "bags": 0, "bags_omitting_any": 0, "duplicate_centres": 0,
            "bags_with_duplicates": 0, "empty_windows": 0, "boundary_windows": 0,
            "mixed_series_windows": 0, "invalid_windows": 0, "fallback_windows": 0,
            "groups": {g: {"studies_available": 0, "studies_missing_series": 0,
                           "studies_fallback_only": 0, "studies_no_valid_triplet": 0,
                           "available_bags": 0, "covered_bags": 0, "omitted_bags": 0}
                       for g in GROUPS}}


def add_cache(target, state):
    target["studies"] += 1
    for group, reason in state.items():
        target["groups"][group]["studies_" + reason] += 1


def add_bag(target, event, available, covered, omitted):
    target["bags"] += 1
    for key in ("bags_omitting_any", "duplicate_centres", "bags_with_duplicates", "empty_windows",
                "boundary_windows", "mixed_series_windows", "invalid_windows", "fallback_windows"):
        target[key] += event[key]
    for group in available:
        if available[group] == "available":
            target["groups"][group]["available_bags"] += 1
            target["groups"][group]["covered_bags"] += group in covered
            target["groups"][group]["omitted_bags"] += group in omitted


def merge(target, other):
    for key in target:
        if key != "groups":
            target[key] += other[key]
    for group in GROUPS:
        for key in target["groups"][group]:
            target["groups"][group][key] += other["groups"][group][key]


def finalize(stats):
    out = dict(stats)
    out["omitting_any_rate"] = stats["bags_omitting_any"] / stats["bags"] if stats["bags"] else None
    out["duplicate_centres_per_bag"] = stats["duplicate_centres"] / stats["bags"] if stats["bags"] else None
    out["groups"] = {}
    for group, values in stats["groups"].items():
        out["groups"][group] = dict(values, omission_rate=(values["omitted_bags"] / values["available_bags"]
            if values["available_bags"] else None))
    return out


def replay_recipe(name, spec, data, folds, seeds, epochs, policy):
    cohort = set(data)
    per_uid = {uid: blank() for uid in cohort}
    overall = blank()
    per_fold = {str(f): blank() for f in folds} if policy != "legacy" else {"full_train": blank()}
    train_folds = {uid: [] for uid in cohort}
    if policy == "legacy":
        for uid in cohort:
            train_folds[uid] = ["full_train"]
    else:
        for fold, ids in folds.items():
            for uid in ids:
                train_folds[uid].append(str(fold))
    for uid in sorted(cohort):
        mask, slots = data[uid]
        state = availability(mask, slots)
        add_cache(overall, state)
        add_cache(per_uid[uid], state)
        for fold in train_folds[uid]:
            add_cache(per_fold[fold], state)
        for seed in seeds:
            for epoch in range(epochs):
                rng = study_rng(seed, epoch, uid)
                centers = (train_centers(mask, len(mask), 12, rng) if policy == "legacy" else
                           select_centers(mask, len(mask), 12, spec, policy, rng))
                require(len(centers) == 12, "sampler did not return K12")
                event, covered, omitted = audit_bag(centers, mask, slots, state)
                for fold in train_folds[uid]:
                    add_bag(overall, event, state, covered, omitted)
                    add_bag(per_uid[uid], event, state, covered, omitted)
                    add_bag(per_fold[fold], event, state, covered, omitted)
    return dict(recipe=name, policy=policy, seeds=list(seeds), epochs=epochs, k=12,
                expected_bags=sum(len(v) for v in train_folds.values()) * len(seeds) * epochs,
                summary=finalize(overall), per_fold={f: finalize(s) for f, s in per_fold.items()}), per_uid


def join_strata(per_uid, source, strata, relation):
    require(source in strata, "strata model source missing")
    output = {}
    for target, classes in strata[source].items():
        output[target] = {}
        for cls, ids in classes.items():
            require(isinstance(ids, list), "stratum IDs must be a list")
            matched = set(ids) & set(per_uid)
            stats = blank()
            for uid in matched:
                merge(stats, per_uid[uid])
            output[target][cls] = dict(source_studies=len(set(ids)), matched_studies=len(matched),
                                       **finalize(stats))
    return {"relation": relation, "source_model": source, "targets": output}


def run(rosters=ROSTERS, r64=R64, e143=E143, series=SERIES, strata=STRATA):
    started = time.monotonic()
    folds, cohort = load_rosters(rosters)
    flags = read_series_flags(series, cohort)
    spec64, data64, meta64 = load_raptor(r64, cohort, flags)
    spec96, data96, meta96 = load_exp143(e143, cohort, flags)
    strata_body = read_json(strata)["strata"]
    outputs = {}
    for name, source, seeds, epochs, policy in RECIPES:
        spec, data = (spec64, data64) if source == "raptor64" else (spec96, data96)
        summary, per_uid = replay_recipe(name, spec, data, folds, seeds, epochs, policy)
        if name == "coatnet_exp143_v2":
            summary["strata"] = {model: join_strata(per_uid, model, strata_body, "cross_model_diagnostic")
                                 for model in ("l2_radr50", "convnext_t_l2")}
        else:
            summary["strata"] = {name: join_strata(per_uid, name, strata_body, "native_model")}
        require(summary["summary"]["bags"] == summary["expected_bags"], "replay bag count differs")
        outputs[name] = summary
    return dict(experiment="EXP180_K12_COVERAGE", version=VERSION, cohort_studies=len(cohort),
                display_loss="not measured: montage selection metadata is not an input",
                unit="eligible UID × fold × seed × epoch; centre draws use the exact training study_rng",
                training_loader_caveat="batch drop_last visitation is not replayed; these are eligible draws",
                inputs={"rosters": {str(f): sha(rosters / f"fold{f}" / "ordered-ids.json") for f in folds},
                        "raptor64": meta64, "exp143": meta96, "train_series_sha256": sha(series),
                        "strata_sha256": sha(strata)}, recipes=outputs, cpu_seconds=round(time.monotonic()-started, 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(int(os.environ.get(key, "1")) <= 4, f"{key} must be <=4")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "studies": result["cohort_studies"],
                      "cpu_seconds": result["cpu_seconds"],
                      "bags": {k: v["summary"]["bags"] for k, v in result["recipes"].items()}}))


if __name__ == "__main__":
    main()
