# Changelog v1.1.3 (2026-10-14): accept strictly sealed v113 carry-over; PL science unchanged.
"""H-B-146 PL-v1, 1.1.1: UID-only joins, hash-fold split, frozen L2 weights.

Changelog 2026-10-13 v1.1.1: final EXPOSED producer contract; bounded atomic writes and nonfatal heartbeat. v1.1.0 retained.

No scores are used in registration. Labels are never printed. Production and
screen teacher contracts are deliberately distinct; engineering fixtures cannot
be admitted for scientific fits. Changelog 2026-10-13 v1.1: exposed exploratory six-fit design, train-only scale and frozen silence mask. v1.0 retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

VERSION = "1.1.1"
UID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WORK = ROOT / ".local/scratch/astra-b4/H-B-146/v1.1.3"
BASE = ROOT / "training/configs/hb122-control-s42-f0-v1.json"
FOLDS = ROOT / "artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/folds.csv"
GOLD = ROOT / "storage/competition/train.csv"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replace_retry(source, destination):
    """Bound Windows sharing violations; durable artifacts still fail closed."""
    for attempt in range(4):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(.05 * (attempt + 1))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    replace_retry(tmp, path)


def write_status(path, value):
    """Best-effort telemetry must never abort the fit; receipts use write()."""
    try:
        write(path, value)
        return True
    except OSError:
        return False


TEACHER_EXPOSURE = {
    "exposure": "EXPOSED", "clean_oof": False, "pooled_target_priors": True,
    "pooled_state_priors": True, "fastmri_plus_auxiliary": True,
    "anchor_ancestry": "unresolved", "side_ancestry": "unresolved",
}


def require_teacher_exposure(value, *, csv=False):
    for key, expected in TEACHER_EXPOSURE.items():
        actual = value.get(key)
        if csv:
            need(actual == str(expected), "teacher EXPOSED provenance differs: " + key)
        else:
            need(type(actual) is type(expected) and actual == expected,
                 "teacher EXPOSED provenance differs: " + key)


def pins(paths):
    return {str(Path(p).resolve()): sha(p) for p in paths}


def check_pins(values):
    for path, digest in values.items():
        need(sha(path) == digest, "input hash changed: " + path)


def hash_fold(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4


def table(path, *, id_column=UID, allow_missing=False):
    # Whitelist columns at parse time: never access any report-text field.
    frame = pd.read_csv(path, usecols=lambda c: c in [id_column, *TARGETS, *[t + "_soft" for t in TARGETS]],
                        dtype={id_column: str})
    need(id_column in frame, "missing UID column")
    need(frame[id_column].notna().all() and not frame[id_column].duplicated().any(), "missing/duplicate UID")
    need(frame[id_column].str.len().gt(0).all(), "empty UID")
    cols = TARGETS if all(t in frame for t in TARGETS) else [t + "_soft" for t in TARGETS]
    need(all(t in frame for t in cols), "target schema incomplete")
    a = frame[cols].to_numpy(dtype=np.float32)
    finite = np.isfinite(a)
    need((allow_missing or finite.all()) and ((a[finite] >= 0) & (a[finite] <= 1)).all(), "invalid labels/probabilities")
    return dict(zip(frame[id_column], a))


def aligned(values, ids, *, exact=False):
    need(len(ids) == len(set(ids)), "duplicate requested UID")
    need(set(ids) <= set(values), "missing requested UID")
    if exact:
        need(set(ids) == set(values), "unexpected UID")
    return np.stack([values[u] for u in ids]).astype(np.float32)


def blend(l2, teacher, ids):
    a, b = aligned(l2, ids), aligned(teacher, ids)
    need(np.isfinite(a).all() and np.isfinite(b).all() and
         ((a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)).all(), "invalid soft targets")
    return np.float32(.5) * a + np.float32(.5) * b


def split(rows, gold):
    need(set(rows.columns) == {UID, "component_id", "fold"}, "fold schema differs")
    need(not rows[UID].duplicated().any() and rows.notna().all().all(), "invalid fold rows")
    groups = {}
    for uid, group in zip(rows[UID].astype(str), rows.component_id.astype(str)):
        groups.setdefault(group, []).append(uid)
    cross = {g for g, ids in groups.items() if len({hash_fold(u) for u in ids}) > 1}
    excluded = {u for g in cross for u in groups[g]}
    ids = sorted(set(rows[UID].astype(str)) - excluded - set(gold))
    tr, va = [u for u in ids if hash_fold(u) != 0], [u for u in ids if hash_fold(u) == 0]
    need(tr and va, "empty split")
    return {"train": tr, "val": va, "excluded_cross_hash": sorted(excluded), "gold": sorted(gold),
            "cross_hash_components": len(cross), "cohort": sorted(rows[UID].astype(str)),
            "components": {u: g for g, us in groups.items() for u in us}}


def frozen_weights(l2, train):
    prevalence = aligned(l2, train).mean(0).clip(.03, .7)
    return ((1 - prevalence) / prevalence).clip(1, 10).astype(np.float32)


def register(out, exposed_exploratory=False):
    need(exposed_exploratory, "explicit --exposed-exploratory required")
    out = Path(out)
    base = read(BASE)
    c = base["config"]
    check_pins({str(ROOT / p): h for p, h in base["inputs"]["data"].items()})
    labels = table(ROOT / c["labels"]["path"])
    gold = {u: a for u, a in table(GOLD, allow_missing=True).items() if np.isfinite(a).all()}
    rows = pd.read_csv(FOLDS, dtype=str)
    s = split(rows, gold)
    aligned(labels, s["cohort"])
    need(len(s["cohort"]) == 4343 and s["cross_hash_components"] == 42 and
         len(s["excluded_cross_hash"]) == 176 and len(gold) == 58 and len(s["train"]) == 3190, "registered cohort audit differs")
    mask, mask_pins = silence_mask(s["cohort"])
    s.update(status=STATUS, exposed_exploratory=True, limitations=LIMITATIONS,
             silence={"definition": SILENCE_RULE, "targets": TARGETS, "pins": mask_pins,
                      "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()},
             scale={"method": "logit_intercept_probability_mean", "slope": 1., "clip": 1e-6,
                    "fit_roster": "train", "reason": "match L2 mean probability; preserve teacher ranking"},
             version=VERSION, alpha=.5, targets=TARGETS, weights=frozen_weights(labels, s["train"]).tolist(),
             recipe=c, pins=pins([BASE, FOLDS, GOLD, ROOT / c["labels"]["path"],
                                 ROOT / c["roots"]["manifest"]]),
             go={"report_weak4_and_macro_min": .002, "minimum_same_rulers": 3,
                 "image_seed_mean_delta_min": .003, "image_branch_each_report_macro_min": -.003,
                 "selection": "among GO arms: highest mean four-ruler seed-mean-prediction macro delta; exact tie global",
                 "aggregation": "AUC of arithmetic seed-mean predictions",
                 "arms": ["control", "global", "fill"], "seeds": [42,43],
                 "report_selection": "K36_flip", "gold": "descriptive/exposed",
                 "oai": OAI_STATUS, "q7s_weight": 0.})
    if (out / "registration.json").exists():
        need(read(out / "registration.json") == s, "registration differs; version required")
    else:
        write(out / "registration.json", s)
    return {"train": len(s["train"]), "val": len(s["val"]), "excluded": len(s["excluded_cross_hash"]), "gold": len(gold)}


def registration(out):
    s = read(Path(out) / "registration.json")
    need(s["version"] == VERSION and s["alpha"] == .5 and s["targets"] == TARGETS, "registration differs")
    check_pins(s["pins"])
    check_pins(s["silence"]["pins"])
    need(s["status"] == STATUS and s["limitations"] == LIMITATIONS and s["exposed_exploratory"] is True, "registration exposure differs")
    return s


STATUS = "EXPOSED_EXPLORATORY"
LIMITATIONS = [
    "f0 pooled bias: 12 population-level report base rates; no UID-label assignment",
    "f0 pooled state priors: 71 population-level report state priors",
    "fastMRI+ auxiliary supervision on the shared encoder; upstream ancestry unresolved",
    "SKM-TEA V-Net anchor used for crops; complete ancestry uncertified",
    "author pool 4,344 versus own 4,343 unreconciled",
    "Gold58 exposed/descriptive; exploratory screen cannot establish a clean confirmatory gain",
]
OAI_STATUS = "teacher ancestry evidence partial: ImageNet init, fastMRI+, SKM-TEA segmenter"
SILENCE_RULE = "EXP191/D-B-124/D-B-130: numeric marker == 0.5 AND numeric derived_relabel < 0.59; missing excluded; predicate applied to all 12 targets"
MARKER = ROOT / "storage/labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_full.csv"
RELABEL = ROOT / "storage/labels/derived_relabel/relabel_v1.csv"
SILENCE_PINS = {str(MARKER): "96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537",
                str(RELABEL): "8aeaf4048b10ccb4fd48d229706b366bffdabb89019772d19f5e2bdf29d6ffc8"}


def silence_mask(ids):
    check_pins(SILENCE_PINS)
    # Numeric columns only; never load report text or .local/relabel.
    marker, guard = table(MARKER, allow_missing=True), table(RELABEL, allow_missing=True)
    mask = np.zeros((len(ids), 12), dtype=bool)
    for i, uid in enumerate(ids):
        if uid in marker and uid in guard:
            mask[i] = (marker[uid] == .5) & np.isfinite(guard[uid]) & (guard[uid] < .59)
    return mask, pins([MARKER, RELABEL, ROOT / "tools/exp191_softprior.py", ROOT / "tools/exp182_synovitis_fill.py"])


def scaled_teacher(l2, teacher, train, ids):
    """Slope=1 affine logit map; intercept fits mean PROBABILITY on train only."""
    need(train and set(train) <= set(ids), "invalid scale training roster")
    target = aligned(l2, train).astype(np.float64).mean(0).clip(1e-6, 1 - 1e-6)
    p = aligned(teacher, train).astype(np.float64).clip(1e-6, 1 - 1e-6)
    logits = np.log(p / (1-p))
    lo, hi = np.full(12, -40.), np.full(12, 40.)
    for _ in range(100):
        mid = (lo + hi) / 2
        mean = (1 / (1 + np.exp(-(logits + mid)))).mean(0)
        lo = np.where(mean < target, mid, lo)
        hi = np.where(mean >= target, mid, hi)
    shift = (lo + hi) / 2
    full = aligned(teacher, ids).astype(np.float64).clip(1e-6, 1 - 1e-6)
    result = (1 / (1 + np.exp(-(np.log(full / (1-full)) + shift)))).astype(np.float32)
    return result, {"method": "logit_intercept_probability_mean", "slope": [1.] * 12,
                    "intercept": shift.tolist(), "fit_rows": len(train),
                    "train_uid_sha256": hashlib.sha256("\n".join(train).encode()).hexdigest(),
                    "clip": 1e-6, "target_mean": target.tolist()}


def arm_targets(base, scaled, mask):
    need(base.shape == scaled.shape == mask.shape and mask.dtype == bool, "arm target shapes differ")
    return {"global": np.float32(.5) * base + np.float32(.5) * scaled,
            "fill": np.where(mask, scaled, base)}


def require_exposed(provenance, exposed_exploratory=False):
    need(exposed_exploratory, "HOLD: explicit --exposed-exploratory required; never clean")
    for entry in (provenance, provenance.get("teacher", provenance)):
        need(entry.get("status") == STATUS and entry.get("limitations") == LIMITATIONS
             and entry.get("exposed_exploratory") is True, "exposure limitations/flag differ")


def validate_teacher(csv_path, receipt_path, ids, mode="screen_f0"):
    r = read(receipt_path)
    require_exposed(r, True)
    require_teacher_exposure(r)
    need(r.get("version") == VERSION and r.get("producer_version") == "1.1.3" and r.get("seal_version") == "1.1.3" and bool(r.get("carry_over")),
         "teacher version differs")
    need(mode == r["mode"] == "screen_f0" and r["model_selection"] == "f0_only"
         and r["output_kind"] == "pre_stacker" and r["complete"] is True and r["failures"] == 0
         and r["csv_sha256"] == sha(csv_path), "teacher contract differs")
    check_pins(r["pins"])
    exposure = pd.read_csv(csv_path, usecols=list(TEACHER_EXPOSURE), dtype=str, keep_default_na=False)
    for row in exposure.to_dict("records"):
        require_teacher_exposure(row, csv=True)
    meta = pd.read_csv(csv_path, usecols=[UID, "model_role", "model_fold", "status", "output_kind"], dtype={UID: str})
    need(meta[UID].is_unique and set(ids) <= set(meta[UID]) and meta.model_role.eq("f0").all()
         and meta.model_fold.eq(0).all() and meta.output_kind.eq("pre_stacker").all()
         and meta.status.isin(["ok", "dropped series"]).all(), "wrong teacher role/coverage/status")
    return table(csv_path), r


def build(out, teacher=None, receipt=None, mode="screen_f0", engineering=False, exposed_exploratory=False):
    need(exposed_exploratory, "explicit --exposed-exploratory required")
    out = Path(out)
    s = registration(out)
    l2path = ROOT / s["recipe"]["labels"]["path"]
    l2, ids = table(l2path), s["cohort"]
    if engineering:
        # Nonconstant deterministic teacher exercises calibration and all masks.
        t = {u: np.random.default_rng(i).uniform(.02,.98,12).astype(np.float32) for i,u in enumerate(ids)}
        r, extra = {"engineering_only": True, "status": STATUS, "limitations": LIMITATIONS,
                    "exposed_exploratory": True}, []
    else:
        need(teacher is not None and receipt is not None, "teacher CSV and receipt required")
        t, r = validate_teacher(teacher, receipt, ids, mode)
        extra = [teacher, receipt]
    scaled, scaling = scaled_teacher(l2, t, s["train"], ids)
    mask, mask_pins = silence_mask(ids)
    need(hashlib.sha256(mask.tobytes()).hexdigest() == s["silence"]["mask_sha256"], "silence mask changed")
    base = aligned(l2, ids)
    dest = out / ("engineering" if engineering else mode)
    dest.mkdir(parents=True, exist_ok=True)
    arrays = arm_targets(base, scaled, mask)
    hashes = {}
    for arm, array in arrays.items():
        frame = pd.DataFrame(array, columns=TARGETS)
        frame.insert(0, UID, ids)
        path = dest / (arm + ".csv")
        # Scientific labels are immutable once built.
        if path.exists() and not engineering:
            need(np.array_equal(aligned(table(path), ids), array), "scientific labels differ; version required")
        else:
            frame.to_csv(path, index=False)
        hashes[arm] = sha(path)
    if not engineering and (dest / "scale.json").exists():
        need(read(dest / "scale.json") == scaling, "scientific scale differs; version required")
        need(np.array_equal(np.load(dest / "silence-mask.npy", allow_pickle=False), mask), "scientific silence differs")
    else:
        np.save(dest / "silence-mask.npy", mask, allow_pickle=False)
        write(dest / "scale.json", scaling)
    p = {"version": VERSION, "status": STATUS, "limitations": LIMITATIONS, "exposed_exploratory": True,
         "alpha": .5, "mode": mode, "engineering_only": engineering, "arm_hashes": hashes,
         "csv_sha256": hashes["global"], "teacher_csv": str(Path(teacher).resolve()) if teacher else None,
         "pins": {**pins([out / "registration.json", l2path, dest / "scale.json", dest / "silence-mask.npy", *extra]), **mask_pins},
         "teacher": r, "rows": len(ids), "teacher_extra_rows": len(set(t) - set(ids)),
         "weights": s["weights"], "weight_source": "L2_new_train_roster_only", "q7s_weight": 0.}
    path = dest / "provenance.json"
    if path.exists() and not engineering:
        need(read(path) == p, "scientific provenance differs; version required")
    else:
        write(path, p)
    return {"rows": len(ids), "engineering_only": engineering, "status": STATUS,
            "teacher_extra_rows": p["teacher_extra_rows"], "silent_cells": int(mask.sum()), "scale_fit_rows": len(s["train"])}


def validate_labels(out, engineering=False, mode="screen_f0", arm="global"):
    s = registration(out)
    dest = Path(out) / ("engineering" if engineering else mode)
    p = read(dest / "provenance.json")
    require_exposed(p, True)
    need(p["version"] == VERSION and p["alpha"] == .5 and p["mode"] == mode and
         p["engineering_only"] is engineering and p["weights"] == s["weights"], "PL provenance differs")
    check_pins(p["pins"])
    for name in ("global", "fill"):
        need(p["arm_hashes"][name] == sha(dest / (name + ".csv")), "arm labels changed")
    values = table(dest / (arm + ".csv"))
    aligned(values, s["cohort"], exact=True)
    return s, values, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["register", "build", "validate"])
    ap.add_argument("--out", type=Path, default=WORK)
    ap.add_argument("--teacher", type=Path)
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--mode", choices=["screen_f0"], default="screen_f0")
    ap.add_argument("--engineering", action="store_true")
    ap.add_argument("--exposed-exploratory", action="store_true")
    a = ap.parse_args(argv)
    need(a.exposed_exploratory, "explicit --exposed-exploratory required")
    if a.command == "register":
        result = register(a.out, a.exposed_exploratory)
    elif a.command == "build":
        result = build(a.out, a.teacher, a.receipt, a.mode, a.engineering, a.exposed_exploratory)
    else:
        s, _, _ = validate_labels(a.out, a.engineering, a.mode)
        result = {"valid": True, "rows": len(s["cohort"]), "engineering_only": a.engineering}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
