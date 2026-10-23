"""ledger_jev_bench v1.1.0 — add an extra reference label set to the evidence ledger for historical submissions.

Changelog
- v1.1.0 (2026-10-09): generic `--label NAME=PATH[:bin|soft]` (default: jev_hybrid, binary). `soft` scores the
  expected AUC against the label probabilities themselves (the bench metric supports soft labels), for label sets
  whose positives carry low probabilities but correct ranks (aquila_v1 MCL / PF OA, D-B-078). Keys that already
  have a public reading only receive the new source; nothing already recorded is touched (append-only ledger).
- v1.0.0 (2026-10-09): first version. Re-uses the exact point definitions of the frozen ruler bench
  (.local/ruler-bench-20261007/bench.py, executed only up to its soft-gold section, so nothing is written there)
  and its metric (expected AUC, binarised labels, macro over 12 targets). Added jev_hybrid for every non-leaky
  scored bench point and imported the two pre-registered bench candidates later submitted (rad5 public 0.925,
  D-B-049; coat5rad5 0.926, D-B-057) with their bench values (computed 2026-10-07 22:47, before the readings).

CPU only, 2 threads. Never reads report text (labels are read with explicit usecols).
Run:  python tools/ledger_jev_bench.py [--label aquila_v1_soft=storage/labels/derived_constellation/aquila_v1.csv:soft] [--dry-run]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / ".local/ruler-bench-20261007/bench.py"
POINTS_CSV = ROOT / ".local/ruler-bench-20261007/points.csv"
DEFAULT_LABEL = "jev_hybrid=storage/labels/riadmohamed42__jev-knee-labels/HYBRID_labels_binary.csv:bin"
CUT = "# ----------------------------------------------------------------------------------------------- soft gold"
# later submissions of pre-registered bench candidates (first copy of each duplicate submission)
CAND_READINGS = {"rad5": ("56924947", 0.925, "H-B-31 B RadR50x5 seed 42 (copies 56925075/56925301 same score), D-B-049"),
                 "coat5rad5": ("56925392", 0.926, "H-B-31 A CoAtNet5 + RadR50x5 rank mean (copy 56925454 same score), D-B-057")}
SKIP = {"eval_set", "point", "submission", "public", "n_studies", "models_per_family", "train_labels",
        "leaky_oof", "candidate", "oof_vs_served"}


def load_bench():
    src = BENCH.read_text(encoding="utf-8")
    assert CUT in src, "bench.py layout changed"
    ns = {"__name__": "bench_points"}
    exec(compile(src.split(CUT)[0], str(BENCH), "exec"), ns)  # loaders, POINTS, CANDS, metrics only
    return ns


def parse_label(spec):
    name, rest = spec.split("=", 1)
    path, mode = (rest.rsplit(":", 1) if rest.endswith((":bin", ":soft")) else (rest, "bin"))
    return name, ROOT / path, mode


def label_values(ns, path, mode):
    import numpy as np
    import pandas as pd
    T, U = ns["T"], ns["U"]
    lab = pd.read_csv(path, usecols=[U] + T, dtype={U: str}).drop_duplicates(U).set_index(U)[T].apply(pd.to_numeric, errors="coerce")
    brn = [p for p in ns["POINTS"] if p["name"] == "B_RN3F"][0]
    V888 = list(ns["V888"])
    sets = {"V888": V888, "C529": [u for u in V888 if u in brn["scores"].index], "G4343": list(ns["G1"])}
    full = {"V888": [p["name"] for p in ns["POINTS"] if p["name"] != "B_RN3F"],
            "C529": [p["name"] for p in ns["POINTS"]],
            "G4343": ["relabel5", "distill5", "coat5rn5", "coat5", "R1", "R2", "CNX5"]}
    out, cover = {}, {}
    for es, uids in sets.items():
        sub = lab.reindex(uids)
        Y = ns["binar"](sub) if mode == "bin" else sub.values.astype(float)
        cover[es] = float((~np.isnan(Y)).all(axis=1).mean())
        for p in ns["POINTS"] + ns["CANDS"]:
            if not p["cand"] and p["name"] not in full[es]:
                continue
            if not set(uids) <= set(p["scores"].index):
                continue
            out[(es, p["name"])] = ns["macro"](p["scores"].reindex(uids).values, Y)[0]
    return out, cover


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", action="append", help="NAME=PATH[:bin|soft]; default " + DEFAULT_LABEL)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print("ledger_jev_bench v1.1.0", flush=True)
    sys.path.insert(0, str(ROOT))
    from tools import evidence_ledger as led
    ns = load_bench()
    labels = [parse_label(s) for s in (a.label or [DEFAULT_LABEL])]
    vals, covers = {}, {}
    for name, path, mode in labels:
        vals[name], covers[name] = label_values(ns, path, mode)
    rows = list(csv.DictReader(POINTS_CSV.open(encoding="utf-8")))
    have_read = {r["key"] for r in led._read(led.DIR / "readings.csv")}
    have_meas = {(r["key"], r["eval_set"], r["source"]) for r in led._read(led.DIR / "measurements.csv")}
    plan = []
    for r in rows:
        es, pt = r["eval_set"], r["point"]
        key = f"bench-{es}-{pt}"
        if r["leaky_oof"] == "True" or any((es, pt) not in vals[n] for n, _, _ in labels):
            continue
        new = [f"{n}={vals[n][(es, pt)]}" for n, _, _ in labels if (key, es, n) not in have_meas]
        if key not in have_read and r["candidate"] == "True" and pt in CAND_READINGS:
            bench = [f"{k}={v}" for k, v in r.items() if k not in SKIP and v not in ("", None) and (key, es, k) not in have_meas]
            plan.append(dict(key=key, es=es, sources=bench + new, reading=CAND_READINGS[pt],
                             note="pre-registered bench candidate (bench values computed 2026-10-07 22:47 before its public reading); extra labels added 2026-10-09 by ledger_jev_bench"))
        elif key in have_read and new:
            plan.append(dict(key=key, es=es, sources=new, reading=None,
                             note="added 2026-10-09 by ledger_jev_bench v1.1.0 (same bench metric; computed after the public reading)"))
    print(json.dumps({"labels": [(n, str(p.relative_to(ROOT)), m) for n, p, m in labels], "row_coverage": covers,
                      "planned_keys": len(plan),
                      "values": {n: {f"{es}/{pt}": round(v, 4) for (es, pt), v in sorted(d.items())} for n, d in vals.items()}},
                     indent=1), flush=True)
    if a.dry_run:
        return
    for p in plan:
        led.record(p["key"], p["es"], p["sources"], note=p["note"])  # before the reading: bench values predate it
        if p["reading"] and p["key"] not in have_read:
            led.reading(p["key"], p["reading"][0], p["reading"][1], note=p["reading"][2])
    print(json.dumps({"recorded_keys": len(plan)}), flush=True)


if __name__ == "__main__":
    main()
