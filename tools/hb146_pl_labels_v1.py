"""H-B-146 PL-v1, 1.0.0: UID-only joins, hash-fold split, frozen L2 weights.

No scores are used in registration. Labels are never printed. Production and
screen teacher contracts are deliberately distinct; engineering fixtures cannot
be admitted for scientific fits. Changelog: 2026-10-13 initial implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

VERSION = "1.0.0"
UID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WORK = ROOT / ".local/scratch/astra-b4/H-B-146"
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


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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


def register(out):
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
         len(s["excluded_cross_hash"]) == 176 and len(gold) == 58, "registered cohort audit differs")
    s.update(version=VERSION, alpha=.5, targets=TARGETS, weights=frozen_weights(labels, s["train"]).tolist(),
             recipe=c, pins=pins([BASE, FOLDS, GOLD, ROOT / c["labels"]["path"],
                                 ROOT / c["roots"]["manifest"]]),
             go={"image_seed_mean_delta_min": 0., "each_report_macro_delta_min": -.003,
                 "report_selection": "K36_flip", "gold": "descriptive/exposed", "q7s_weight": 0.})
    if (out / "registration.json").exists():
        need(read(out / "registration.json") == s, "registration differs; version required")
    else:
        write(out / "registration.json", s)
    return {"train": len(s["train"]), "val": len(s["val"]), "excluded": len(s["excluded_cross_hash"]), "gold": len(gold)}


def registration(out):
    s = read(Path(out) / "registration.json")
    need(s["version"] == VERSION and s["alpha"] == .5 and s["targets"] == TARGETS, "registration differs")
    check_pins(s["pins"])
    return s


def validate_teacher(csv_path, receipt_path, ids, mode):
    r = read(receipt_path)
    need(r["csv_sha256"] == sha(csv_path) and r["mode"] == mode and r["output_kind"] == "pre_stacker",
         "teacher provenance/mode differs")
    check_pins(r["pins"])
    need(r["complete"] is True and r["failures"] == 0, "incomplete teacher pass")
    need(r["source_pool_membership"] == "manifest_contract_unverified_membership", "unrecognised exposure policy")
    need(r.get("validation_label_exposure") == "pooled_prior_not_clean" and
         r.get("auxiliary_label_exposure") == "unknown",
         "existing R7 requires explicit pooled-prior/unknown-auxiliary disclosure")
    meta = pd.read_csv(csv_path, usecols=[UID, "fold_k", "status", "output_kind"], dtype={UID: str})
    need(not meta[UID].duplicated().any() and set(ids) <= set(meta[UID]), "teacher UID coverage differs")
    need(meta.status.isin(["ok", "dropped series"]).all() and meta.output_kind.eq("pre_stacker").all(), "teacher failure/stacker")
    expected = np.zeros(len(meta), int) if mode == "screen_f0" else np.array([hash_fold(u) for u in meta[UID]])
    need(np.array_equal(meta.fold_k.to_numpy(), expected), "wrong teacher fold; validation label leakage")
    need(r["model_selection"] == ("f0_only" if mode == "screen_f0" else "sha1_first8_mod4"), "wrong model selection")
    return table(csv_path), r


def require_clean_teacher(provenance):
    """Review cannot waive exposure; the current R7 contract intentionally fails."""
    teacher = provenance["teacher"]
    need(teacher.get("validation_label_exposure") == "none" and
         teacher.get("auxiliary_label_exposure") == "none",
         "HOLD: R7 pooled-prior/auxiliary validation-label exposure; a new clean teacher design/version is required")


def build(out, teacher=None, receipt=None, mode="screen_f0", engineering=False):
    out = Path(out)
    s = registration(out)
    l2path = ROOT / s["recipe"]["labels"]["path"]
    l2 = table(l2path)
    ids = s["cohort"]
    if engineering:
        t = {u: np.full(12, .5, np.float32) for u in ids}
        r, extra = {"engineering_only": True}, []
    else:
        need(teacher is not None and receipt is not None, "teacher CSV and receipt required")
        t, r = validate_teacher(teacher, receipt, ids, mode)
        extra = [teacher, receipt]
    a = blend(l2, t, ids)
    dest = out / ("engineering" if engineering else mode)
    dest.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(a, columns=TARGETS)
    frame.insert(0, UID, ids)
    frame.to_csv(dest / "labels.csv", index=False)
    p = {"version": VERSION, "alpha": .5, "mode": mode, "engineering_only": engineering,
         "csv_sha256": sha(dest / "labels.csv"), "pins": pins([out / "registration.json", l2path, *extra]),
         "teacher": r, "rows": len(ids), "teacher_extra_rows": len(set(t) - set(ids)),
         "weights": s["weights"], "weight_source": "L2_new_train_roster_only", "q7s_weight": 0.}
    write(dest / "provenance.json", p)
    return {"rows": len(ids), "engineering_only": engineering, "teacher_extra_rows": p["teacher_extra_rows"]}


def validate_labels(out, engineering=False, mode="screen_f0"):
    s = registration(out)
    dest = Path(out) / ("engineering" if engineering else mode)
    p = read(dest / "provenance.json")
    need(p["version"] == VERSION and p["alpha"] == .5 and p["mode"] == mode and
         p["engineering_only"] is engineering and p["weights"] == s["weights"] and
         p["csv_sha256"] == sha(dest / "labels.csv"), "PL provenance differs")
    check_pins(p["pins"])
    values = table(dest / "labels.csv")
    aligned(values, s["cohort"], exact=True)
    return s, values, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["register", "build", "validate"])
    ap.add_argument("--out", type=Path, default=WORK)
    ap.add_argument("--teacher", type=Path)
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--mode", choices=["screen_f0", "production_heldout"], default="screen_f0")
    ap.add_argument("--engineering", action="store_true")
    a = ap.parse_args(argv)
    if a.command == "register":
        result = register(a.out)
    elif a.command == "build":
        result = build(a.out, a.teacher, a.receipt, a.mode, a.engineering)
    else:
        s, _, _ = validate_labels(a.out, a.engineering, a.mode)
        result = {"valid": True, "rows": len(s["cohort"]), "engineering_only": a.engineering}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
