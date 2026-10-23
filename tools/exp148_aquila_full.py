"""EXP148 v1.0.1 (pooled-account limits are deferred, not fatal) — full image-checked labelling with Astra, variant F (aquila_v1). (B, 2026-10-09)

User decision (03:5x KST): "<owner remark removed>".
Recipe = EXP147 variant F exactly as validated on Gold58 (D-B-078: Astra F 0.9187): one exam per Codex call, the
exam's report TEXT (piped from train.csv by this script; B never reads or prints it), the flight_v4 report label
(p + conf), the 15 central slots @336 montages (EXP146 renderer = EXP145 recipe) and the relabel v1.1 task paragraph
and official definitions verbatim (EXP147 _prompt_f). Parsing as EXP147 v1.2.1 (one format-only re-ask).
Naming (user rule: constellations): Astra -> aquila, Sol -> lyra, Opus -> cygnus, combination -> orion.

  python tools/exp148_aquila_full.py render                 # resumable, one CPU process
  python tools/exp148_aquila_full.py run --parallel 12      # resumable; stops cleanly on a usage/quota error
  python tools/exp148_aquila_full.py status
  python tools/exp148_aquila_full.py merge                  # -> storage/labels/derived_constellation/aquila_v1.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp146_constellation_labels as e146  # noqa: E402
from tools import exp147_input_ab as e147  # noqa: E402

WORK = ROOT / ".local/exp148-aquila"
OUT = WORK / "out"
STOP = WORK / "STOPPED_USAGE_LIMIT"
MODEL = "gpt-6-astra"
NAME = "aquila_v1"
LIMIT = re.compile(r"usage limit|rate limit|quota|insufficient_quota|exceeded your|429|billing|out of credits|credits", re.I)


def render():
    return e146.render()


def run(parallel=12, limit=0):
    import pandas as pd
    key = json.loads(e146.KEY.read_text(encoding="utf-8"))
    labels = e146._labels(key)
    df = pd.read_csv(ROOT / "storage/competition/train.csv", usecols=["StudyInstanceUID", "Report"], dtype=str).set_index("StudyInstanceUID")
    OUT.mkdir(parents=True, exist_ok=True)
    if STOP.exists():
        STOP.unlink()  # a resumed run clears the previous stop
    codes = [c for c in sorted(key) if not (OUT / f"{c}.json").exists()
             and all((e146.IMG / f"{c}_{p}.png").exists() for p in ("sag", "cor", "ax"))]
    if limit:
        codes = codes[:limit]
    lock = threading.Lock()
    counts, started = {}, time.time()
    recent = []

    def call(code, suffix):
        prompt = e147._prompt_f(code, str(df.loc[key[code], "Report"]), labels) + suffix
        raw = OUT / f"{code}.raw.md"
        args = ["codex", "exec", "-m", MODEL, "-s", "read-only", "-C", str(e146.IMG), "--skip-git-repo-check"]
        for p in ("sag", "cor", "ax"):
            args += ["-i", str(e146.IMG / f"{code}_{p}.png")]
        args += ["-o", str(raw), "-"]
        r = subprocess.run(args, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1500)
        text = raw.read_text(encoding="utf-8") if raw.exists() else ""
        tok = re.findall(r"tokens used\s*\n?\s*([\d,]+)", r.stdout + r.stderr)
        return r, text, (int(tok[-1].replace(",", "")) if tok else None)

    def one(code):
        if STOP.exists():
            return code, "skipped_after_stop"
        try:
            for suffix in ("", "\nFORMAT: every one of the 12 values must be a plain number between 0 and 1 (no text, no objects).\n"):
                r, text, tok = call(code, suffix)
                parsed = e147.p145_parse(text, code)
                if parsed:
                    (OUT / f"{code}.json").write_text(json.dumps({"labels": parsed, "tokens": tok, "format_reask": bool(suffix)}) + "\n", encoding="utf-8")
                    with lock:
                        recent.append(0)
                    return code, "ok"
                tail = r.stdout[-4000:] + r.stderr[-4000:]
                if not text.strip() and LIMIT.search(tail):
                    # v1.0.1: the Codex provider pools several accounts; one account's limit is not exhaustion.
                    # Defer this exam (no output written, a later resume retries it); stop only when >= 20 of the
                    # last 50 calls hit a limit.
                    with lock:
                        recent.append(1)
                        hits = sum(recent[-50:])
                        if hits >= 20 and not STOP.exists():
                            STOP.write_text(json.dumps({"code": code, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "limit_hits_last50": hits,
                                                        "tail": tail[-1200:]}) + "\n", encoding="utf-8")
                    return code, "usage_limit_deferred"
                with lock:
                    recent.append(0)
                if not text.strip():
                    break
            (OUT / f"{code}.failed.txt").write_text((r.stdout[-3000:] + r.stderr[-3000:]), encoding="utf-8")
            return code, "failed"
        except Exception as exc:  # one exam never stops the pool
            (OUT / f"{code}.failed.txt").write_text(repr(exc), encoding="utf-8")
            return code, "failed"

    print(json.dumps({"todo": len(codes), "parallel": parallel, "model": MODEL}), flush=True)
    with ThreadPoolExecutor(parallel) as ex:
        for n, (code, st) in enumerate(ex.map(one, codes), 1):
            with lock:
                counts[st] = counts.get(st, 0) + 1
            if n % 100 == 0 or st == "failed" or (st == "usage_limit_deferred" and counts[st] % 10 == 1):
                print(json.dumps({"done": n, "of": len(codes), "last": code, "status": st, "counts": counts,
                                  "hours": round((time.time() - started) / 3600, 2)}), flush=True)
    final = {"counts": counts, "stopped_usage_limit": STOP.exists(), "hours": round((time.time() - started) / 3600, 2)}
    print(json.dumps(final), flush=True)
    return final


def status():
    key = json.loads(e146.KEY.read_text(encoding="utf-8"))
    ok = len(list(OUT.glob("T*.json"))) if OUT.exists() else 0
    failed = len(list(OUT.glob("*.failed.txt"))) if OUT.exists() else 0
    rendered = sum(1 for c in key if (e146.IMG / f"{c}_ax.png").exists())
    return {"exams": len(key), "rendered": rendered, "labelled": ok, "failed_files": failed, "stopped": STOP.exists(),
            "percent": round(100 * ok / len(key), 1)}


def merge():
    import pandas as pd
    from tools import exp130_gold58_compare as g130
    key = json.loads(e146.KEY.read_text(encoding="utf-8"))
    rows = {f.stem: json.loads(f.read_text(encoding="utf-8"))["labels"] for f in OUT.glob("T*.json")}
    gold, _ = g130.official_gold(ROOT)
    gold = set(gold)
    T = g130.TARGETS
    df = pd.DataFrame([{"StudyInstanceUID": key[c], **{t: min(.98, max(.02, v[t])) for t in T}} for c, v in sorted(rows.items())])
    train = df[~df["StudyInstanceUID"].isin(gold)]
    e146.OUTDIR.mkdir(parents=True, exist_ok=True)
    path = e146.OUTDIR / f"{NAME}.csv"
    train.to_csv(path, index=False)
    df[df["StudyInstanceUID"].isin(gold)].to_csv(WORK / f"{NAME}_gold58_eval.csv", index=False)
    prov = {"name": NAME, "reader_model": MODEL + " (Codex)", "method": "EXP148 = EXP147 variant F (report text + flight_v4 label + 15 central slots @336 + relabel v1.1 rules verbatim), one exam per call",
            "rows_training": int(len(train)), "rows_gold58_removed": int(len(df) - len(train)), "exams_labelled": int(len(df)), "exams_expected": len(key),
            "clip": [0.02, 0.98], "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evidence": "D-B-078 (Gold58: Astra F 0.9187)", "use": "local research only; derived from competition data",
            "calibration_note": "MCL / PF OA positives often receive low probabilities (rules stricter than the official labels; ranks intact) - combine by rank"}
    (e146.OUTDIR / f"{NAME}.provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    return prov


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("render")
    r = sub.add_parser("run"); r.add_argument("--parallel", type=int, default=12); r.add_argument("--limit", type=int, default=0)
    sub.add_parser("status"); sub.add_parser("merge")
    a = ap.parse_args(argv)
    res = render() if a.cmd == "render" else run(a.parallel, a.limit) if a.cmd == "run" else status() if a.cmd == "status" else merge()
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
