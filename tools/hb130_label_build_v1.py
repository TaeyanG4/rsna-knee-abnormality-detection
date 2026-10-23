"""H-B-130 per-target aquila-share label builder v1.0.0 (2026-10-13, claude-hb130).

Builds storage/labels/derived_constellation/l2_dread_aquila_pertarget_v1.csv with the SAME
construction as the control labels l2_dread75_aquila25_v1 (EXP184 mix_rows arithmetic:
float64 weights, per-cell float32 cast, L2 canonical UID order, identical CSV dialect), but
aquila share 0.60 for Effusion, Synovitis, PF OA, Lateral OA, Medial OA and 0.25 elsewhere
(MCL unchanged). Before writing, the all-0.25 rebuild must reproduce the pinned L2 bytes.
Gold58 stays excluded exactly as in the control (absent from labels and fold-0 roster).

  python -B -m tools.hb130_label_build_v1 build     # create-only; prints aggregates only
  python -B -m tools.hb130_label_build_v1 check     # re-verify the written file

Never prints or serializes label rows; CPU only; no network.
Changelog 1.0.0: first version (registered by .local/scratch/claude-hb130/PROTOCOL-v1.md).
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
    os.environ.setdefault(_key, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools import exp184_aquila_share as e184

VERSION = "1.0.0"
TOOL = "tools/hb130_label_build_v1.py"
NAME = "l2_dread_aquila_pertarget_v1"
OUT = f"storage/labels/derived_constellation/{NAME}.csv"
PROVENANCE = f"storage/labels/derived_constellation/{NAME}.provenance.json"
PROTOCOL = ".local/scratch/claude-hb130/PROTOCOL-v1.md"
E184_TOOL = "tools/exp184_aquila_share.py"
TARGETS = tuple(e184.TARGETS)
UID = e184.UID
FIVE = ("Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis")
SHARES = tuple(60 if t in FIVE else 25 for t in TARGETS)  # aquila percent per target
GOLD = "storage/gold/gold_labels.csv"
GOLD_SHA = "990dd72929224e76391c93bd24e861dc75eaa69bc6f14ed71463abc9911e3b66"
ORDERED = "artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold0/ordered-ids.json"
ORDERED_SHA = "0a1b2f3d67ba9588855d671cc7119d168fbfb8d0b4499357ad68249cbcc2f473"
TRAIN_SHA = "a69580bd6a68dc802cce4f214ac7afffb9742a7f6ad7963c33245b9de72cace5"
VAL_SHA = "56ae86a4c180cc46d28e2a192485d133d58f4d05907b67fa18076f199f507dc4"
CONTROL_SEMANTIC = "e7973769a7036a8fddf902a62a58e6991fb15ac45bd1761828cbd7b5fcfe4657"
CONTROL_POS_WEIGHT = (4.449650764465332, 9.348213195800781, 1.3974807262420654, 3.895580291748047,
                      2.780895471572876, 4.685205936431885, 2.0544726848602295, 1.9205825328826904,
                      1.5157464742660522, 5.019646167755127, 3.3302481174468994, 8.444528579711914)
need, sha, read = e184.need, e184.sha, e184.read


def mix_rows_pertarget(ids, dread, aquila, shares=SHARES):
    """exp184.mix_rows with a per-target share vector; same float64 ops, float32 cast, repr."""
    need(len(shares) == 12 and all(s in (25, 60) for s in shares), "unregistered share vector")
    d_weight = np.asarray([(100 - s) / 100 for s in shares], dtype=np.float64)
    a_weight = np.asarray([s / 100 for s in shares], dtype=np.float64)
    rows = []
    for uid in ids:
        values = (d_weight * dread[uid] + a_weight * aquila[uid]).astype(np.float32)
        rows.append([uid, *[repr(float(x)) for x in values]])
    return rows


def _gold_ids(root):
    import csv
    need(sha(root / GOLD) == GOLD_SHA, "pinned Gold58 file changed")
    with (root / GOLD).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [UID, *TARGETS], "Gold58 columns differ")
        ids = [r[UID] for r in reader]
    need(len(ids) == 58 == len(set(ids)), "Gold58 roster differs")
    return set(ids)


def _roster(root):
    from training.arms import mil
    need(sha(root / ORDERED) == ORDERED_SHA, "fold-0 roster changed")
    ordered = read(root / ORDERED)
    train, val = list(ordered["train"]), list(ordered["val"])
    need(mil.uid_hash(train) == TRAIN_SHA and mil.uid_hash(val) == VAL_SHA, "fold-0 roster identity differs")
    return train, val


def _columns(rows):
    return [[r[j + 1] for r in rows] for j in range(12)]


def construct(root=ROOT):
    """Return (header, ids, text, aggregates) in memory; nothing is written or printed."""
    root = Path(root)
    need((root / PROTOCOL).is_file(), "register PROTOCOL-v1.md before building labels")
    for path, digest in ((e184.L2, e184.L2_SHA), (e184.DREAD, e184.DREAD_SHA), (e184.AQUILA, e184.AQUILA_SHA)):
        need(sha(root / path) == digest, "pinned label input changed: " + path)
    prov = read(root / e184.L2_PROVENANCE)
    need(prov["csv_sha256"] == e184.L2_SHA and prov["gold58_included"] is False and prov["rows"] == 4347,
         "L2 provenance differs")
    header, rows = e184._rows(root / e184.L2)
    need(header == [UID, *TARGETS] and len(rows) == 4347 and len({r[0] for r in rows}) == 4347,
         "L2 header/UID roster differs")
    ids = [r[0] for r in rows]
    dread, aquila = e184._source(root / e184.DREAD, set(ids)), e184._source(root / e184.AQUILA, set(ids))
    # Construction identity: all-0.25 per-target rebuild == exp184 rebuild == pinned L2 bytes.
    control_text = e184._csv_text(header, mix_rows_pertarget(ids, dread, aquila, (25,) * 12))
    need(control_text == e184._csv_text(header, e184.mix_rows(ids, dread, aquila, 25)),
         "per-target mixer differs from exp184 mix_rows")
    need(control_text.encode() == (root / e184.L2).read_bytes(), "control construction does not reproduce pinned L2")
    arm_rows = mix_rows_pertarget(ids, dread, aquila)
    text = e184._csv_text(header, arm_rows)
    gold = _gold_ids(root)
    train, val = _roster(root)
    need(not gold & set(ids) and not gold & (set(train) | set(val)), "Gold58 must stay excluded")
    need(set(train) | set(val) <= set(ids), "fold-0 roster label coverage differs")
    old_cols, new_cols = _columns(rows), _columns(arm_rows)
    untouched = [t for t in TARGETS if t not in FIVE]
    cells_changed = {t: sum(a != b for a, b in zip(old_cols[j], new_cols[j])) for j, t in enumerate(TARGETS)}
    need(all(cells_changed[t] == 0 for t in untouched), "untouched target text differs")
    old_v = np.asarray([[float(x) for x in r[1:]] for r in rows], np.float32)
    new_v = np.asarray([[float(x) for x in r[1:]] for r in arm_rows], np.float32)
    need(np.isfinite(new_v).all() and ((new_v >= 0) & (new_v <= 1)).all(), "arm label cell invalid")
    pos = {u: i for i, u in enumerate(ids)}
    tr = [pos[u] for u in train]
    aggregates = dict(
        rows=len(ids), control_rows=len(rows), header_equal=True, uid_order_equal=True,
        gold58_in_labels=len(gold & set(ids)), gold58_in_fold0_roster=len(gold & (set(train) | set(val))),
        fold0_train=len(train), fold0_val=len(val),
        aquila_share_percent=dict(zip(TARGETS, SHARES)),
        cells_changed=cells_changed,
        untouched_byte_identical=untouched,
        mean_all_control={t: float(old_v[:, j].mean(dtype=np.float64)) for j, t in enumerate(TARGETS)},
        mean_all_arm={t: float(new_v[:, j].mean(dtype=np.float64)) for j, t in enumerate(TARGETS)},
        mean_shift_all={t: float(new_v[:, j].mean(dtype=np.float64) - old_v[:, j].mean(dtype=np.float64))
                        for j, t in enumerate(TARGETS)},
        mean_shift_fold0_train={t: float(new_v[tr, j].mean(dtype=np.float64) - old_v[tr, j].mean(dtype=np.float64))
                                for j, t in enumerate(TARGETS)},
        max_abs_cell_change={t: float(np.abs(new_v[:, j] - old_v[:, j]).max()) for j, t in enumerate(TARGETS)})
    return header, ids, text, aggregates


def runner_identity(path, root=ROOT):
    """Exact runner functions: semantic hash and fold-0 positive weights of a label file."""
    from training.arms import mil
    train, _ = _roster(Path(root))
    labels, semantic = mil.load_labels(Path(root) / path)
    need(all(u in labels for u in train), "label coverage differs")
    return semantic, [float(x) for x in mil.positive_weights(labels, train)]


def build(root=ROOT, dry_run=False):
    root = Path(root)
    _, _, text, aggregates = construct(root)
    digest = hashlib.sha256(text.encode()).hexdigest()
    result = dict(status="DRY_RUN" if dry_run else "BUILT", name=NAME, path=OUT, csv_sha256=digest,
                  builder=TOOL, version=VERSION, label_rows_printed=False, aggregates=aggregates)
    if dry_run:
        return result
    need(not (root / OUT).exists() and not (root / PROVENANCE).exists(), "versioned labels already exist")
    with (root / OUT).open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)
    need(sha(root / OUT) == digest, "label write differs")
    semantic, weights = runner_identity(OUT, root)
    control_semantic, control_weights = runner_identity(e184.L2, root)
    need(control_semantic == CONTROL_SEMANTIC and tuple(control_weights) == CONTROL_POS_WEIGHT,
         "control runner identity differs from the HB122 control spec")
    result.update(semantic_sha256=semantic, pos_weight=weights, control_pos_weight=control_weights)
    e184.write_json(root / PROVENANCE, dict(
        name=NAME, rows=aggregates["rows"], csv_sha256=digest, semantic_sha256=semantic,
        fold0_pos_weight=weights, builder=TOOL, builder_sha256=sha(root / TOOL), version=VERSION,
        construction_from=E184_TOOL, construction_from_sha256=sha(root / E184_TOOL),
        formula="float32((1-s_t)*Dread + s_t*aquila_v1) per cell, s_t=0.60 for Medial OA/Lateral OA/PF OA/"
                "Effusion/Synovitis, 0.25 otherwise (= l2_dread75_aquila25_v1 for those seven)",
        aquila_share_percent=aggregates["aquila_share_percent"], canonical_order=e184.L2,
        inputs={e184.L2: e184.L2_SHA, e184.DREAD: e184.DREAD_SHA, e184.AQUILA: e184.AQUILA_SHA, GOLD: GOLD_SHA,
                ORDERED: ORDERED_SHA},
        protocol=PROTOCOL, protocol_sha256=sha(root / PROTOCOL), gold58_included=False, oai=False,
        licence="competition-derived labels; Dread public dataset licence retained; local research use only",
        created=time.strftime("%Y-%m-%d %H:%M:%S %z"), label_rows_printed=False,
        checks={k: aggregates[k] for k in ("rows", "control_rows", "gold58_in_labels", "gold58_in_fold0_roster",
                                           "cells_changed", "untouched_byte_identical", "mean_shift_all",
                                           "mean_shift_fold0_train")}))
    return result


def check(root=ROOT):
    root = Path(root)
    _, _, text, aggregates = construct(root)
    prov = read(root / PROVENANCE)
    digest = hashlib.sha256(text.encode()).hexdigest()
    need(sha(root / OUT) == digest == prov["csv_sha256"], "written labels differ from construction")
    semantic, weights = runner_identity(OUT, root)
    need(semantic == prov["semantic_sha256"] and weights == prov["fold0_pos_weight"], "runner identity differs")
    return dict(status="CHECKED", name=NAME, csv_sha256=digest, semantic_sha256=semantic, pos_weight=weights,
                provenance_sha256=sha(root / PROVENANCE), aggregates=aggregates, label_rows_printed=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("build", "dry-run", "check"))
    a = ap.parse_args(argv)
    print(f"hb130_label_build v{VERSION}", file=sys.stderr)
    result = check() if a.command == "check" else build(dry_run=a.command == "dry-run")
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
