"""Build the immutable EXP168 RSNA soft labels. CPU only; aggregate QA only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tools import exp143_v2 as base
from tools import exp155_own0949_oai_v11 as exp155

ROOT = Path(__file__).resolve().parents[1]
DREAD = ROOT / "storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv"
JEV = ROOT / "storage/labels/riadmohamed42__jev-knee-labels/HYBRID_labels_binary.csv"
DREAD_SHA = "8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1"
JEV_SHA = "aa64723de96d45408588c6fb6c234980d610e996e3f1f4241111213aeefe0316"
OUT = ROOT / "storage/labels/derived_constellation/l2_dread75_jev25_v1.csv"
PROVENANCE = OUT.with_suffix(".provenance.json")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_source(path, ids, binary=False):
    """Read target columns only; extra source rows never enter the output."""
    wanted = set(ids)
    found = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or base.UID not in reader.fieldnames:
            raise ValueError("source lacks UID")
        columns = [t if t in reader.fieldnames else t + "_soft" for t in base.TARGETS]
        if not set(columns) <= set(reader.fieldnames):
            raise ValueError("source lacks a target column")
        seen = set()
        for row in reader:
            uid = row[base.UID]
            if not uid or uid in seen:
                raise ValueError("missing or duplicate source UID")
            seen.add(uid)
            if uid not in wanted:
                continue
            values = np.asarray([float(row[column]) for column in columns], dtype=np.float32)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("incomplete or out-of-range training source")
            if binary and not np.isin(values, (0.0, 1.0)).all():
                raise ValueError("JEV HYBRID cells must be binary")
            found[uid] = values
    if set(found) != wanted:
        raise ValueError("source does not cover all registered training studies")
    return found


def mix(dread, jev):
    a, b = np.asarray(dread, np.float32), np.asarray(jev, np.float32)
    if (a.shape != (12,) or b.shape != (12,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or ((a < 0) | (a > 1)).any()
            or not np.isin(b, (0., 1.)).all()):
        raise ValueError("invalid Dread/JEV target vector")
    return (np.float32(.75) * a + np.float32(.25) * b).astype(np.float32)


def build(out=OUT, provenance=PROVENANCE):
    if Path(out).resolve() != OUT.resolve() or Path(provenance).resolve() != PROVENANCE.resolve():
        raise ValueError("EXP168 labels must use the registered versioned paths")
    if OUT.exists() or PROVENANCE.exists():
        raise FileExistsError("versioned EXP168 labels already exist")
    if sha(DREAD) != DREAD_SHA or sha(JEV) != JEV_SHA:
        raise ValueError("registered Dread/JEV source file changed")
    prior = base.load_registration(exp155.BASE_PROTOCOL)
    ids, gold = prior["train_ids"], prior["excluded_gold_ids"]
    if len(ids) != 4347 or len(set(ids)) != 4347 or set(ids) & set(gold) or len(gold) != 58:
        raise ValueError("EXP143 v2 non-Gold cohort changed")
    dread = read_source(DREAD, ids)
    jev = read_source(JEV, ids, binary=True)
    means = np.zeros(12, dtype=np.float64)
    changed = np.zeros(12, dtype=np.int64)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([base.UID, *base.TARGETS])
        for uid in ids:
            value = mix(dread[uid], jev[uid])
            if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError("invalid mixed soft label")
            writer.writerow([uid, *[format(float(v), ".9g") for v in value]])
            means += value.astype(np.float64)
            changed += (value != dread[uid])
    readback = base.read_label_table(OUT)
    if set(readback) != set(ids):
        raise ValueError("derived label serialization changed cohort")
    record = dict(name="l2_dread75_jev25_v1", rows=len(ids), targets=list(base.TARGETS),
                  formula="float32(0.75*Dread labels_llm_soft + 0.25*JEV HYBRID), .9g CSV",
                  source_sha256={str(DREAD.relative_to(ROOT)): sha(DREAD),
                                 str(JEV.relative_to(ROOT)): sha(JEV)},
                  baseline_protocol_sha256=sha(exp155.BASE_PROTOCOL),
                  baseline_train_ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                  gold58_included=False, oai=False, csv_sha256=sha(OUT),
                  aggregate_qa=dict(mean_by_target=dict(zip(base.TARGETS, (means / len(ids)).tolist())),
                                    changed_cells_by_target=dict(zip(base.TARGETS, changed.tolist()))),
                  use="EXP168 RSNA training only; OAI v2 labels unchanged")
    with PROVENANCE.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(status="LABELS_BUILT", rows=len(ids), csv_sha256=record["csv_sha256"],
                provenance_sha256=sha(PROVENANCE), aggregate_qa=record["aggregate_qa"])


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
