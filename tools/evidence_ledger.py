"""Evidence ledger v1.1.0 — which local evidence source tracks the public LB? (B, 2026-10-09)

User rule (Taeyang, 2026-10-09 01:3x KST): judge every candidate on SEVERAL evidence sources, and for every
submission record what each source said beforehand; after 10-20 scored submissions the source that tracks the
LB best should become visible. This ledger is append-only and CPU-light.

Files (artifacts/evidence_ledger/):
  measurements.csv  key, eval_set, source, value, recorded_at, note     (one row per source value, BEFORE scoring)
  readings.csv      key, submission_id, public, read_at, note           (the public score once COMPLETE)
Values are absolute scores of the candidate on that source and eval set (e.g. macro AUC of the five-fold OOF vs
the flight_v4 labels on the 4,343-study set, or Gold58 macro AUC from fresh inference). Sources are only compared
within one eval_set. A key is one submitted candidate (use the submission ID used with kaggle_submit_once).

  python tools/evidence_ledger.py record  --key H-B-34-exp140-aug1 --eval-set OOF4343 --source dread=0.9001 --source flight_v4=0.8585 --note "..."
  python tools/evidence_ledger.py reading --key H-B-34-exp140-aug1 --submission-id 57000000 --public 0.939
  python tools/evidence_ledger.py import-bench .local/ruler-bench-20261007/points.csv   # historical V888 points (leaky OOF excluded)
  python tools/evidence_ledger.py report                                                # per source: n, Spearman, pairwise sign agreement
  python tools/evidence_ledger.py report --retrospective   # also counts values computed after the reading -> report_retrospective.md

Changelog: v1.1.0 (2026-10-09) report --retrospective, a separate labelled view that includes values computed after
the public reading (e.g. a fixed new label set applied to old OOF); the main report stays pre-registered only.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "artifacts/evidence_ledger"
MEAS = ["key", "eval_set", "source", "value", "recorded_at", "note"]
READ = ["key", "submission_id", "public", "read_at", "note"]
MIN_PUBLIC_GAP = 0.002  # pairs closer than this on public are ties (public noise, D-B-016), not counted for sign agreement


def _append(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _read(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def record(key, eval_set, sources, note="", root=DIR):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    have = {(r["key"], r["eval_set"], r["source"]) for r in _read(root / "measurements.csv")}
    rows = []
    for item in sources:
        name, value = item.split("=", 1)
        if (key, eval_set, name) in have:
            raise ValueError(f"already recorded: {key}/{eval_set}/{name} (append-only)")
        rows.append(dict(key=key, eval_set=eval_set, source=name, value=float(value), recorded_at=now, note=note))
    if key in {r["key"] for r in _read(root / "readings.csv")}:
        rows = [dict(r, note=(r["note"] + " | recorded AFTER the public reading").strip(" |")) for r in rows]
    _append(root / "measurements.csv", MEAS, rows)
    return len(rows)


def reading(key, submission_id, public, note="", root=DIR):
    if key in {r["key"] for r in _read(root / "readings.csv")}:
        raise ValueError(f"reading already recorded for {key}")
    _append(root / "readings.csv", READ, [dict(key=key, submission_id=submission_id, public=float(public),
                                               read_at=time.strftime("%Y-%m-%d %H:%M:%S"), note=note)])


def import_bench(points_csv, root=DIR):
    rows = _read(Path(points_csv))
    skip = {"eval_set", "point", "submission", "public", "n_studies", "models_per_family", "train_labels",
            "leaky_oof", "candidate", "oof_vs_served"}
    meas, reads, n = [], [], 0
    have = {r["key"] for r in _read(root / "readings.csv")}
    for r in rows:
        if r.get("leaky_oof") == "True" or not r.get("public"):
            continue
        key = f"bench-{r['eval_set']}-{r['point']}"
        if key in have:
            continue
        for src, val in r.items():
            if src in skip or val in ("", None):
                continue
            meas.append(dict(key=key, eval_set=r["eval_set"], source=src, value=float(val),
                             recorded_at="2026-10-07 22:47:00", note="historical ruler bench (D-B-042), pre-registered 22:47"))
        reads.append(dict(key=key, submission_id=str(r["submission"]).split(".")[0], public=float(r["public"]),
                          read_at="historical", note="imported from ruler bench"))
        n += 1
    _append(root / "measurements.csv", MEAS, meas)
    _append(root / "readings.csv", READ, reads)
    return n


def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def report(root=DIR, retrospective=False):
    pub = {r["key"]: float(r["public"]) for r in _read(root / "readings.csv")}
    by = {}
    for r in _read(root / "measurements.csv"):
        if r["key"] in pub and (retrospective or "recorded AFTER" not in r["note"]):
            by.setdefault((r["eval_set"], r["source"]), {})[r["key"]] = float(r["value"])
    out = []
    for (es, src), vals in sorted(by.items()):
        keys = sorted(vals)
        if len(keys) < 2:
            out.append(dict(eval_set=es, source=src, n=len(keys)))
            continue
        x, y = [vals[k] for k in keys], [pub[k] for k in keys]
        agree = total = 0
        for a, b in itertools.combinations(keys, 2):
            dp = pub[a] - pub[b]
            if abs(dp) < MIN_PUBLIC_GAP:
                continue
            total += 1
            agree += (vals[a] - vals[b]) * dp > 0
        out.append(dict(eval_set=es, source=src, n=len(keys), spearman=round(_spearman(x, y), 3),
                        sign_agreement=f"{agree}/{total}", sign_rate=round(agree / total, 3) if total else None))
    out.sort(key=lambda d: (d["eval_set"], -(d.get("sign_rate") or -1), -(d.get("spearman") or -1)))
    name = "report_retrospective" if retrospective else "report"
    (root / f"{name}.json").parent.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    title = ("# Evidence ledger report — RETROSPECTIVE (includes values computed after the public reading)"
             if retrospective else "# Evidence ledger report")
    lines = [title, "", f"Pairs closer than {MIN_PUBLIC_GAP} on public are ties and not counted.", "",
             "| eval set | source | n | Spearman vs public | pairwise sign agreement |", "|---|---|---|---|---|"]
    for d in out:
        lines.append(f"| {d['eval_set']} | {d['source']} | {d['n']} | {d.get('spearman', '')} | {d.get('sign_agreement', '')} |")
    (root / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("--key", required=True); r.add_argument("--eval-set", required=True)
    r.add_argument("--source", action="append", required=True); r.add_argument("--note", default="")
    g = sub.add_parser("reading"); g.add_argument("--key", required=True); g.add_argument("--submission-id", required=True)
    g.add_argument("--public", type=float, required=True); g.add_argument("--note", default="")
    i = sub.add_parser("import-bench"); i.add_argument("points")
    rp = sub.add_parser("report"); rp.add_argument("--retrospective", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        print(json.dumps({"recorded": record(a.key, a.eval_set, a.source, a.note)}))
    elif a.cmd == "reading":
        reading(a.key, a.submission_id, a.public, a.note); print(json.dumps({"reading": a.key}))
    elif a.cmd == "import-bench":
        print(json.dumps({"imported_points": import_bench(a.points)}))
    else:
        print(json.dumps(report(retrospective=a.retrospective), indent=2))


if __name__ == "__main__":
    main()
