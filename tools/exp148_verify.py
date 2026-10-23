"""EXP148 verifier v1.0.0 — prove that no exam was skipped, failed silently or got a malformed label (B, 2026-10-09).

User: "<owner remark removed>". Checks every code of the EXP146 key:
  missing outputs (not yet labelled / deferred by a limit), *.failed.txt files, format re-asks, malformed values
  (missing target, non-finite, outside 0..1), suspicious rows (all 12 values identical; a row identical to another
  exam's row), the raw answer naming a different exam code, and per-target means vs the flight_v4 input label.
Prints aggregates only (no report text, no UIDs).

  python tools/exp148_verify.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp146_constellation_labels as e146  # noqa: E402
from tools import exp148_aquila_full as e148  # noqa: E402


def main():
    import numpy as np
    from tools import exp130_gold58_compare as g130
    T = g130.TARGETS
    key = json.loads(e146.KEY.read_text(encoding="utf-8"))
    codes = sorted(key)
    out = e148.OUT
    rep = {"exams": len(codes)}
    done, missing, malformed, reask, wrong_code, constant = [], [], [], 0, [], []
    rows = {}
    for c in codes:
        f = out / f"{c}.json"
        if not f.exists():
            missing.append(c)
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        lab = j.get("labels", {})
        bad = [t for t in T if t not in lab or not isinstance(lab[t], (int, float)) or not math.isfinite(lab[t]) or not 0 <= lab[t] <= 1]
        if bad:
            malformed.append(c)
            continue
        if j.get("format_reask"):
            reask += 1
        raw = (out / f"{c}.raw.md")
        if raw.exists():
            names = set(re.findall(r'"(T\d{4})"', raw.read_text(encoding="utf-8")))
            if names and names != {c}:
                wrong_code.append(c)
        v = tuple(round(lab[t], 4) for t in T)
        if len(set(v)) == 1:
            constant.append(c)
        rows[c] = v
        done.append(c)
    dup = [v for v, n in Counter(rows.values()).items() if n > 1]
    dup_codes = sorted(c for c, v in rows.items() if v in set(dup))
    failed = sorted(p.stem for p in out.glob("*.failed.txt"))
    rep.update({"labelled_ok": len(done), "missing_not_yet_labelled": len(missing), "failed_files": len(failed),
                "failed_still_missing": len([c for c in failed if c in set(missing)]),
                "malformed": len(malformed), "format_reask_used": reask, "answer_names_other_exam": len(wrong_code),
                "all_12_values_identical": len(constant), "rows_identical_to_another_exam": len(dup_codes),
                "stopped_flag": e148.STOP.exists()})
    if done:
        labels = e146._labels({c: key[c] for c in done})
        A = np.array([rows[c] for c in done], float)
        F = np.array([[labels[c][t]["p"] for t in T] for c in done], float)
        rep["mean_by_target"] = {t: {"aquila": round(float(A[:, i].mean()), 3), "flight_input": round(float(F[:, i].mean()), 3),
                                     "corr": round(float(np.corrcoef(A[:, i], F[:, i])[0, 1]), 3)} for i, t in enumerate(T)}
    rep["examples"] = {"missing_first": missing[:5], "malformed": malformed[:5], "wrong_code": wrong_code[:5],
                       "constant": constant[:5], "duplicates": dup_codes[:10]}
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    main()
