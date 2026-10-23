"""EXP137 per-source production-cache QA and frozen RadR50 transfer gate.

No CUDA import or model construction occurs in ``qa`` or ``compare``.
Inference is deliberately serial: one checkpoint and one small batch at a time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # B fix: run as a script, training/src modules import each other top-level
for _p in (ROOT / "training" / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
BASE = ROOT / "artifacts/experiments/EXP137_EXT_QA"
SOURCES = ("MRNET", "KNEEMRI", "FASTMRI_PLUS", "OAI", "SKM_TEA")
TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
SEEDS = {42: ("EXP113_RADR50_LOCAL_PROTOCOL_V1", "EXP113_RADR50_LOCAL_F"),
         43: ("EXP118_RADR50_LOCAL_S43_PROTOCOL_V1", "EXP118_RADR50_LOCAL_S43_F")}
CONTRACT = "raptor64_336_c140_s02"
BOOTSTRAP_SEED = 1372026
QA_SEED = 137


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_name(source):
    name = source.upper().replace("-", "_").replace("+", "_PLUS")
    need(name in SOURCES, f"unknown source: {source}")
    return name


def corpus_paths(source, root=ROOT):
    root = Path(root)
    return (root / "storage/cache/derived" / f"EXT_{source}_RAPTOR64_336_C140",
            root / "storage/cache/derived/RAPTOR64_336_C140_S02")


def open_corpus(path):
    from raptor_common import Corpus
    corpus = Corpus(path)
    need(corpus.contract == CONTRACT and corpus.depth == 64 and corpus.spec.get("img") == 336,
         f"unexpected production corpus contract: {path}")
    need(len(corpus.ids) > 0 and len(set(corpus.ids)) == len(corpus.ids), "empty or duplicate corpus ids")
    need(corpus.masks.shape == (len(corpus.ids), corpus.depth), "mask shape differs")
    if Path(path).name.startswith("EXT_"):
        need(corpus.manifest.get("finished") is True, "external production corpus is unfinished")
    return corpus


def groups(spec):
    expected = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1),
                ("Coronal", 0), ("Axial", -1)]
    slots = spec.get("slots", [])
    need(len(slots) == len(expected), "slot groups differ")
    out, start = [], 0
    for item, (plane, fluid) in zip(slots, expected):
        need(len(item) == 3 and item[0] == plane and int(item[1]) == fluid and int(item[2]) >= 3,
             "slot group differs")
        stop = start + int(item[2])
        out.append((f"{plane}_fluid{fluid}" if fluid >= 0 else plane, start, stop))
        start = stop
    need(start == spec["depth"], "slot depth differs")
    return out


def sampled_ids(ids, mask, start, stop, n, seed):
    eligible = [str(uid) for uid, row in zip(ids, mask) if np.any(row[start:stop] > 0)]
    return random.Random(seed).sample(eligible, min(n, len(eligible)))


def middle_image(corpus, uid, start, stop):
    midpoint = (start + stop - 1) / 2
    filled = [i for i in range(start, stop) if corpus.mask(uid)[i] > 0]
    need(bool(filled), "requested image from empty slot")
    index = min(filled, key=lambda i: (abs(i - midpoint), i))
    return np.asarray(corpus.volume(uid)[index], np.uint8)


def correlation(a, b):
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    x -= x.mean(); y -= y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    return None if denominator == 0 else float((x @ y) / denominator)


def corpus_statistics(corpus):
    """Stream complete corpus. Histograms cover all pixels of filled slots only."""
    result = {}
    for name, start, stop in groups(corpus.spec):
        result[name] = dict(start=start, stop=stop, filled_studies=0, filled_slots=0,
                            histogram=np.zeros(256, np.int64), foreground_sum=0.0,
                            mean_image=np.zeros((corpus.spec["img"], corpus.spec["img"]), np.float64),
                            mean_image_n=0)
    for uid in corpus.ids:
        uid = str(uid)
        mask = corpus.mask(uid)
        vol = corpus.volume(uid)
        for name, start, stop in groups(corpus.spec):
            group = result[name]
            filled = np.flatnonzero(mask[start:stop] > 0) + start
            if not len(filled):
                continue
            group["filled_studies"] += 1
            group["filled_slots"] += len(filled)
            for index in filled:
                image = np.asarray(vol[index])
                need(image.dtype == np.uint8, "corpus is not uint8")
                group["histogram"] += np.bincount(image.ravel(), minlength=256)
                group["foreground_sum"] += float(np.mean(image > 16))
            group["mean_image"] += middle_image(corpus, uid, start, stop)
            group["mean_image_n"] += 1
    summary = {}
    for name, group in result.items():
        count = int(group["histogram"].sum())
        cumulative = np.cumsum(group["histogram"])
        quantiles = {str(q): int(np.searchsorted(cumulative, q * max(count - 1, 0) / 100 + 1))
                     for q in (1, 5, 25, 50, 75, 95, 99)} if count else {}
        image = group["mean_image"] / group["mean_image_n"] if group["mean_image_n"] else None
        summary[name] = dict(slot_fill_rate=group["filled_slots"] / (len(corpus) * (group["stop"] - group["start"])),
                             study_fill_rate=group["filled_studies"] / len(corpus),
                             filled_slots=group["filled_slots"], uint8_quantiles=quantiles,
                             mean_foreground_fraction=(group["foreground_sum"] / group["filled_slots"]
                                                       if group["filled_slots"] else None),
                             mean_image_flip_correlation=({"left_right": correlation(image, np.fliplr(image)),
                                                           "up_down": correlation(image, np.flipud(image))}
                                                          if image is not None else None))
    return summary


def qa_flags(external, competition):
    flags = []
    for name, ext in external.items():
        ref = competition[name]
        if ext["filled_slots"] == 0:
            flags.append(f"{name}: empty")
            continue
        if abs(ext["slot_fill_rate"] - ref["slot_fill_rate"]) > .25:
            flags.append(f"{name}: slot-fill gap > 0.25")
        if ref["uint8_quantiles"] and abs(ext["uint8_quantiles"]["50"] - ref["uint8_quantiles"]["50"]) > 50:
            flags.append(f"{name}: median uint8 gap > 50")
        if ref["mean_foreground_fraction"] is not None and abs(ext["mean_foreground_fraction"] - ref["mean_foreground_fraction"]) > .25:
            flags.append(f"{name}: foreground gap > 0.25")
        for axis in ("left_right", "up_down"):
            a = ext["mean_image_flip_correlation"][axis]
            b = (ref["mean_image_flip_correlation"] or {}).get(axis)
            if a is None or b is None:
                flags.append(f"{name}: {axis} orientation probe undefined")
            elif abs(a - b) > .30:
                flags.append(f"{name}: {axis} mean-image symmetry gap > 0.30")
    return flags


def montage(external, competition, name, start, stop, n, path):
    import cv2
    ei = sampled_ids(external.ids, external.masks, start, stop, n, QA_SEED + start)
    ci = sampled_ids(competition.ids, competition.masks, start, stop, n, QA_SEED + 1000 + start)
    pairs = min(len(ei), len(ci))
    if not pairs:
        return 0
    thumb = 168
    tile = np.full((thumb + 22, thumb * 2 + 12, 3), 20, np.uint8)
    cells = []
    for j in range(pairs):
        cell = tile.copy()
        for k, (corpus, uid) in enumerate(((external, ei[j]), (competition, ci[j]))):
            image = cv2.resize(middle_image(corpus, uid, start, stop), (thumb, thumb), interpolation=cv2.INTER_AREA)
            x = k * (thumb + 12)
            cell[22:, x:x + thumb] = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(cell, f"{'EXT' if k == 0 else 'COMP'} {j + 1}", (x + 3, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 240, 240), 1)
        cells.append(cell)
    columns = min(4, pairs)
    rows = math.ceil(pairs / columns)
    canvas = np.full((rows * tile.shape[0], columns * tile.shape[1], 3), 20, np.uint8)
    for j, cell in enumerate(cells):
        y, x = divmod(j, columns)
        canvas[y * tile.shape[0]:(y + 1) * tile.shape[0], x * tile.shape[1]:(x + 1) * tile.shape[1]] = cell
    need(cv2.imwrite(str(path), canvas), f"could not write montage: {path}")
    return pairs


def qa(source, n=24, root=ROOT, base=BASE):
    source = source_name(source)
    need(n > 0, "--n must be positive")
    ext_path, comp_path = corpus_paths(source, root)
    external, competition = open_corpus(ext_path), open_corpus(comp_path)
    ext_hash, comp_hash = digest(ext_path / "manifest.json"), digest(comp_path / "manifest.json")
    need(groups(external.spec) == groups(competition.spec), "external/competition groups differ")
    output = Path(base) / source
    output.mkdir(parents=True, exist_ok=True)
    stats = dict(external=corpus_statistics(external), competition=corpus_statistics(competition))
    sample_counts = {}
    for name, start, stop in groups(external.spec):
        sample_counts[name] = montage(external, competition, name, start, stop, n,
                                      output / f"montage_{name}.png")
    flags = qa_flags(stats["external"], stats["competition"])
    need(ext_hash == digest(ext_path / "manifest.json") and
         comp_hash == digest(comp_path / "manifest.json"), "corpus changed during QA")
    summary = dict(schema="exp137-qa-v1", source=source, external_studies=len(external),
                   competition_studies=len(competition), source_manifest_sha256=ext_hash,
                   competition_manifest_sha256=comp_hash,
                   montage_pairs=sample_counts, montage_seed=QA_SEED, foreground_threshold_uint8=16,
                   statistics=stats, flags=flags, resolved_flags={}, manual_montage_review="PENDING")
    atomic_json(output / "qa.json", summary)
    return summary


def read_table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames is not None and "study_key" in reader.fieldnames, "label key missing")
        rows = {}
        for row in reader:
            key = row["study_key"]
            need(bool(key) and key not in rows, "empty or duplicate study keys")
            rows[key] = row
    need(bool(rows), "empty label table")
    return rows


def binary_label(value):
    if value is None or not str(value).strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"non-binary label: {value!r}") from None
    need(math.isfinite(number) and number in (0., 1.), f"non-binary label: {value!r}")
    return int(number)


def rank_average(values):
    """EXP130 mean-tie percentile ranks, independently implemented for tiny tests."""
    values = np.asarray(values, np.float64)
    need(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "invalid ranks")
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), np.float64)
    at = 0
    while at < len(order):
        end = at + 1
        while end < len(order) and values[order[end]] == values[order[at]]:
            end += 1
        result[order[at:end]] = ((at + end - 1) / 2 + 1) / len(values)
        at = end
    return result


def rank_matrix(matrix):
    return np.stack([rank_average(matrix[:, j]) for j in range(matrix.shape[1])], axis=1)


def auc(y, scores):
    y, scores = np.asarray(y, np.int8), np.asarray(scores, np.float64)
    need(len(y) == len(scores) and np.isin(y, (0, 1)).all(), "invalid AUC input")
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if not pos or not neg:
        return None
    ranks = rank_average(scores) * len(scores)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bootstrap_auc(y, scores, draws=1000, seed=BOOTSTRAP_SEED):
    y = np.asarray(y, np.int8)
    scores = np.asarray(scores, np.float64)
    point = auc(y, scores)
    if point is None:
        return None, None
    rng = np.random.default_rng(seed)
    samples = []
    attempts = 0
    while len(samples) < draws and attempts < draws * 20:
        indices = rng.integers(0, len(y), len(y))
        attempts += 1
        value = auc(y[indices], scores[indices])
        if value is not None:
            samples.append(value)
    need(len(samples) == draws, "too few valid bootstrap draws")
    return point, [float(x) for x in np.quantile(samples, [.025, .975])]


def native_comparisons(source):
    # (native label, displayed prediction, predictor column(s), primary)
    mapping = {
        "MRNET": [("abnormal", "max 12", TARGETS, False), ("acl", "ACL", ("ACL",), True),
                  ("any_meniscus", "max meniscus", ("Medial Meniscus", "Lateral Meniscus"), True)],
        "KNEEMRI": [("acl_any", "ACL", ("ACL",), True),
                    ("acl_complete", "ACL", ("ACL",), False)],
        "FASTMRI_PLUS": [("acl_any", "ACL", ("ACL",), True), ("mcl_any", "MCL", ("MCL",), True),
                         ("any_meniscus", "max meniscus", ("Medial Meniscus", "Lateral Meniscus"), True),
                         ("effusion", "Effusion", ("Effusion",), True),
                         ("acl_low", "ACL", ("ACL",), False),
                         ("acl_high", "ACL", ("ACL",), False),
                         ("mcl_low_mod", "MCL", ("MCL",), False),
                         ("mcl_high", "MCL", ("MCL",), False),
                         ("meniscus_tear", "max meniscus", ("Medial Meniscus", "Lateral Meniscus"), False),
                         ("displaced_meniscal_tissue", "max meniscus", ("Medial Meniscus", "Lateral Meniscus"), False),
                         ("any_cartilage", "max OA", ("Medial OA", "Lateral OA", "PF OA"), False),
                         ("cartilage_partial", "max OA", ("Medial OA", "Lateral OA", "PF OA"), False),
                         ("cartilage_full", "max OA", ("Medial OA", "Lateral OA", "PF OA"), False),
                         ("subchondral_edema", "Contusion", ("Contusion",), False),
                         ("fracture_contusion_dislocation", "max fracture/contusion", ("Fracture", "Contusion"), False),
                         ("periarticular_cysts", "Baker's", ("Baker's",), False)],
        "OAI": [], "SKM_TEA": [],
    }
    return mapping[source]


def evaluate_predictions(source, ids, predictions, root=ROOT, draws=1000, limited=False):
    source = source_name(source)
    ids = [str(i) for i in ids]
    predictions = np.asarray(predictions, np.float64)
    need(predictions.shape == (len(ids), len(TARGETS)) and np.isfinite(predictions).all(), "prediction shape/values differ")
    label_dir = Path(root) / "storage/labels/external"
    mapped = read_table(label_dir / f"{source}_12.csv")
    native = read_table(label_dir / f"{source}_native.csv")
    need(set(mapped) == set(native), "mapped/native study-key sets differ")
    need(set(ids) <= set(mapped), "production study lacks label row")
    result = {}

    def add(name, column, predictor, primary):
        decoded = [binary_label(column.get(uid, "")) for uid in ids]
        index = np.array([i for i, value in enumerate(decoded) if value is not None], dtype=int)
        y = np.array([decoded[i] for i in index], np.int8)
        score = predictor[index]
        positives, negatives = int(y.sum()), int(len(y) - y.sum())
        point, ci = bootstrap_auc(y, score, draws, BOOTSTRAP_SEED + len(result)) if len(y) else (None, None)
        result[name] = dict(positives=positives, negatives=negatives, missing=len(ids)-len(y),
                            auc=point, ci95=ci, low_count=min(positives, negatives) < 10,
                            gate_eligible=primary and min(positives, negatives) >= 30,
                            primary=primary)

    mapped_primary = source in ("OAI", "SKM_TEA")
    for j, target in enumerate(TARGETS):
        add("mapped:" + target, {uid: row[target] for uid, row in mapped.items()},
            predictions[:, j], mapped_primary)
    for name, predictor_name, columns, primary in native_comparisons(source):
        score = np.max(predictions[:, [TARGETS.index(c) for c in columns]], axis=1)
        add(f"native:{name} <- {predictor_name}", {uid: row.get(name, "") for uid, row in native.items()},
            score, primary)
    eligible = [v for v in result.values() if v["gate_eligible"]]
    auc_gate = (not limited and bool(eligible) and all(v["ci95"][0] > .60 for v in eligible)
                and any(v["auc"] >= .75 for v in eligible))
    return dict(schema="exp137-zeroshot-v1", source=source, studies=len(ids), limited=limited,
                label_rows=len(mapped), unmatched_label_rows=len(mapped)-len(ids),
                mapped_labels_sha256=digest(label_dir / f"{source}_12.csv"),
                native_labels_sha256=digest(label_dir / f"{source}_native.csv"),
                bootstrap=dict(draws=draws, method="study resample, valid class draws, percentile 95%", seed=BOOTSTRAP_SEED),
                targets=result, eligible_primary=[k for k, v in result.items() if v["gate_eligible"]],
                auc_gate=auc_gate)


def blend_folds(folds):
    """EXP130: probability mean F0-F4, then mean of seed-specific percentile ranks."""
    need(set(folds) == {(seed, fold) for seed in SEEDS for fold in range(5)}, "ten folds required")
    seed_mean = {seed: np.mean([folds[(seed, fold)] for fold in range(5)], axis=0) for seed in SEEDS}
    return (rank_matrix(seed_mean[42]) + rank_matrix(seed_mean[43])) / 2


def infer_fold(model, loader, device):
    """Use EXP124 plane-aware pair inference; tests pass tiny fake models."""
    from tools import exp124_flip_tta as flip
    plain, mirrored, ids = flip.infer_pair(model, loader, device, "exp113")
    return (plain + mirrored) / 2, ids


def zeroshot(source, device="cuda", batch=1, limit=None, root=ROOT, base=BASE):
    import torch
    from torch.utils.data import DataLoader
    from tools import exp124_flip_tta as flip
    from raptor_common import build_model
    from raptor_radimagenet import CenterSliceWindows

    source = source_name(source)
    need(device in ("cpu", "cuda") and batch > 0 and (limit is None or limit > 0), "invalid inference arguments")
    need(batch <= (2 if device == "cuda" else 8), "batch exceeds bounded-memory evaluation cap")
    root = Path(root)
    cache_path, _competition = corpus_paths(source, root)
    corpus = open_corpus(cache_path)
    ids = sorted(map(str, corpus.ids))
    if limit is not None:
        ids = ids[:limit]
    mapped = read_table(root / "storage/labels/external" / f"{source}_12.csv")
    need(set(ids) <= set(mapped), "production study lacks mapped labels")
    output = Path(base) / source
    output.mkdir(parents=True, exist_ok=True)
    ids_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    cache_hash = digest(cache_path / "manifest.json")
    labels = {uid: np.zeros(len(TARGETS), np.float32) for uid in ids}
    folds, provenance = {}, {}
    for seed, (protocol_name, prefix) in SEEDS.items():
        protocol_path = root / "artifacts/experiments" / protocol_name / "manifest.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        args = protocol["common_args"]
        need(protocol["seed"] == seed and args["arch"] == "resnet50" and args["res"] == 224
             and args["norm"] == "radimagenet" and args["window_policy"] == "within-slot-valid"
             and args["input_policy"] == "center3", "frozen RadR50 recipe differs")
        for fold in range(5):
            fold_ids_path = root / protocol["folds"][str(fold)]["ordered_ids"]["path"]
            fold_ids = json.loads(fold_ids_path.read_text(encoding="utf-8"))
            need(set(ids).isdisjoint(set(fold_ids["train"]) | set(fold_ids["val"])),
                 "external IDs overlap fold data")
        ds = CenterSliceWindows(cache_path, ids, labels, 224, 36, False, args["norm"], seed,
                                window_policy=args["window_policy"], input_policy=args["input_policy"])
        loader = DataLoader(flip.PlaneAwareWindows(ds, corpus, flip.plane_ranges(corpus.spec)),
                            batch_size=batch, shuffle=False, num_workers=0, collate_fn=flip.collate_pair)
        for fold in range(5):
            checkpoint_path = root / "artifacts/experiments" / f"{prefix}{fold}" / "model_best.pt"
            checkpoint_hash = digest(checkpoint_path)
            receipt = dict(seed=seed, fold=fold, ids_sha256=ids_hash, cache_manifest_sha256=cache_hash,
                           protocol_sha256=digest(protocol_path), checkpoint_sha256=checkpoint_hash,
                           k_eval=36, tta="EXP124 sagittal unchanged, coronal/axial horizontal flip")
            part = output / f"fold_s{seed}_f{fold}_{'limit'+str(limit) if limit else 'full'}.npz"
            if part.exists():
                with np.load(part, allow_pickle=False) as stored:
                    need(json.loads(str(stored["receipt"])) == receipt, f"stale resume file: {part}")
                    probabilities = stored["probabilities"]
                need(probabilities.shape == (len(ids), len(TARGETS)) and np.isfinite(probabilities).all(),
                     "invalid resume probabilities")
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                need(checkpoint.get("tag") == f"{prefix}{fold}" and checkpoint.get("fold") == fold and
                     (checkpoint.get("provenance") or {}).get("protocol_sha256") == receipt["protocol_sha256"],
                     "checkpoint provenance differs")
                model = build_model(args["arch"], pretrained=False)
                model.load_state_dict(checkpoint["model"], strict=True)
                del checkpoint
                model = model.to(device).eval()
                probabilities, got = infer_fold(model, loader, device)
                need(got == ids and probabilities.shape == (len(ids), len(TARGETS)), "inference order/shape differs")
                del model
                temporary = part.with_name(part.name + ".partial")
                with temporary.open("wb") as stream:
                    np.savez(stream, probabilities=probabilities, receipt=json.dumps(receipt, sort_keys=True))
                os.replace(temporary, part)
            folds[(seed, fold)] = probabilities
            provenance[f"s{seed}_f{fold}"] = receipt
    blended = blend_folds(folds)
    need(cache_hash == digest(cache_path / "manifest.json"), "external corpus changed during inference")
    suffix = f"limit{limit}" if limit else "full"
    prediction_path = output / f"predictions_{suffix}.csv"
    temporary = prediction_path.with_name(prediction_path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["study_key", *TARGETS])
        writer.writerows([[uid, *map(float, row)] for uid, row in zip(ids, blended)])
    os.replace(temporary, prediction_path)
    result = evaluate_predictions(source, ids, blended, root, limited=limit is not None)
    result.update(predictions_csv=str(prediction_path), predictions_sha256=digest(prediction_path),
                  cache_manifest_sha256=cache_hash, folds=provenance)
    atomic_json(output / f"zeroshot_{suffix}.json", result)
    return result


def compare(source, base=BASE, root=ROOT):
    source = source_name(source)
    output = Path(base) / source
    qa_path, result_path = output / "qa.json", output / "zeroshot_full.json"
    need(qa_path.is_file() and result_path.is_file(), "run qa and full zeroshot first")
    qa_result = json.loads(qa_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    need(qa_result["source"] == result["source"] == source and
         qa_result["source_manifest_sha256"] == result["cache_manifest_sha256"],
         "QA/zeroshot source snapshot differs")
    label_dir = Path(root) / "storage/labels/external"
    need(result["predictions_sha256"] == digest(result["predictions_csv"]),
         "prediction CSV changed since zero-shot scoring")
    need(result["mapped_labels_sha256"] == digest(label_dir / f"{source}_12.csv") and
         result["native_labels_sha256"] == digest(label_dir / f"{source}_native.csv"),
         "labels changed since zero-shot scoring")
    resolved = qa_result.get("resolved_flags", {})
    unresolved = [flag for flag in qa_result["flags"] if not str(resolved.get(flag, "")).strip()]
    gate = (result["auc_gate"] and not unresolved and
            qa_result["manual_montage_review"] == "PASS")
    lines = [f"### EXP137 {source}: {'PASS' if gate else 'FLAG / HOLD'}",
             "", f"Studies: {result['studies']}; gate eligible: {', '.join(result['eligible_primary']) or 'none'}.",
             f"QA flags: {', '.join(qa_result['flags']) or 'none'}; unresolved: {', '.join(unresolved) or 'none'}; "
             f"manual montage review: {qa_result['manual_montage_review']}.",
             "", "| Label / predictor | Pos | Neg | AUC | 95% CI | Low count | Primary gate |",
             "|---|---:|---:|---:|---:|:---:|:---:|"]
    for flag in qa_result["flags"]:
        if flag not in unresolved:
            lines.insert(4, f"Resolved QA: {flag} — {resolved[flag]}")
    for name, item in result["targets"].items():
        ci = item["ci95"]
        lines.append(f"| {name} | {item['positives']} | {item['negatives']} | "
                     f"{item['auc']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
                     f"{'yes' if item['low_count'] else 'no'} | {'yes' if item['gate_eligible'] else 'no'} |"
                     if ci else f"| {name} | {item['positives']} | {item['negatives']} | undefined | — | yes | no |")
    lines += ["", "Next: visually inspect every montage for crop, laterality, plane and contrast; "
              "set manual_montage_review to PASS only after inspection. Investigate every flag before a one-source "
              "training screen. A no-gain result needs written evidence or HOLD.",
              "", "This is a preprocessing gate, not a model-quality or leaderboard claim."]
    verdict = "\n".join(lines)
    print(verdict)
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("qa")
    p.add_argument("--source", required=True)
    p.add_argument("--n", type=int, default=24)
    p = sub.add_parser("zeroshot")
    p.add_argument("--source", required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("compare")
    p.add_argument("--source", required=True)
    args = parser.parse_args(argv)
    if args.action == "qa":
        report = qa(args.source, args.n)
        print(json.dumps({"source": report["source"], "flags": report["flags"],
                          "qa_json": str(BASE / report["source"] / "qa.json")}))
    elif args.action == "zeroshot":
        report = zeroshot(args.source, args.device, args.batch, args.limit)
        print(json.dumps({"source": report["source"], "studies": report["studies"],
                          "auc_gate": report["auc_gate"]}))
    else:
        compare(args.source)


if __name__ == "__main__":
    main()
