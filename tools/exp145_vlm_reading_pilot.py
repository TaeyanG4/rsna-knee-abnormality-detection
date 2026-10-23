"""EXP145 v1.2.0 — stage-1 pilot: can a general vision LLM (Opus / Sol) read knee MRI well enough to fill
report-silent label cells? (B, 2026-10-09; user: "<owner remark removed>", "<owner remark removed>")

Blind design: 20 Gold58 studies (fixed random draw, seed 20261009) are shown as three per-plane montages
(sagittal 7 slices, coronal 4, axial 4 = 15 slices) from the RAPTOR64_336_C140_S02 corpus, under neutral codes
S01..S20. v1.1.0 (user condition 01:5x KST: "<owner remark removed>"): readers get the images PLUS the
report-derived flight_v4 label of each study (probability + extraction confidence; no report text, no UID, no Gold
label) and the official labelling rules (docs/runbooks/relabel-prompt-v1_1.md definitions). Baselines: flight_v4
alone and our AUG1 RadR50. Scoring is done here against the official Gold58 labels and
prints only aggregate AUCs. Comparator: our AUG1 RadR50 (EXP144 aug1_tta) on the same 20 studies.
Registered stop rule (before any reading): a reader whose macro AUC (12 targets) on the 20 studies is below
ours - 0.10 is stopped (no stage 2); stage 2 (all 58 studies) only if the gap is within 0.04.

  python tools/exp145_vlm_reading_pilot.py render
  python tools/exp145_vlm_reading_pilot.py score --reader opus=.local/exp145-vlm-pilot/out_opus.json --reader sol=...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".local/exp145-vlm-pilot"
IMG = OUT / "images"
CORPUS = ROOT / "storage/cache/derived/RAPTOR64_336_C140_S02"
N, SEED = 20, 20261009
GROUPS = {"sag": [(0, 18), (18, 14)], "cor": [(32, 12), (44, 8)], "ax": [(52, 12)]}  # fluid-sensitive group first
COUNT = {"sag": 7, "cor": 4, "ax": 4}
GRID = {"sag": (4, 2), "cor": (2, 2), "ax": (2, 2)}
SPAN = {"sag": (.18, .82), "cor": (.3, .75), "ax": (.3, .8)}  # v1.0.1: v1.0.0 spread slices over the whole stack

PROMPT = """You are an experienced musculoskeletal radiologist producing final labels for knee MRI exams
(codes S01..S20 in this batch). For each exam you get:
  1) report_labels.json - what a text model extracted from the radiologist's REPORT for this exam: for each finding
     "p" = probability the finding is present per the report, "conf" = extraction confidence (low conf usually
     means the report is ambiguous or does not mention the finding);
  2) three montage images read from the exam itself:
     <code>_sag.png  sagittal slices (7 tiles across the joint, fat-suppressed / fluid-sensitive when available),
     <code>_cor.png  coronal slices (4 tiles, front to back),
     <code>_ax.png   axial slices (4 tiles, top to bottom).
     Each tile is a 140 mm field of view, 336 px, windowed per series. Left/right knee and the sagittal direction
     are not given: tell medial from lateral by anatomy (fibular head = lateral, MCL = medial).
Decide each finding the way the official labels were made (subspecialty MSK radiologists reading the images;
borderline findings graded NEGATIVE to favour specificity; one knee per exam). Use the report label where it is
clear and consistent with the images; use the images to decide what the report leaves ambiguous or unmentioned,
and to correct it when the images clearly disagree.

OFFICIAL DEFINITIONS (rules):
- ACL: high-grade tear - more than 50% of fibres torn, or complete tear. Low-grade sprain, partial < 50%, mucoid
  degeneration, intact reconstructed graft -> negative.
- MCL: high-grade acute tear. Sprain, low-grade/partial, chronic thickening, Pellegrini-Stieda -> negative.
- Medial Meniscus / Lateral Meniscus: a tear (signal reaching the articular surface on at least 2 images).
  Degeneration/intrasubstance signal, post-operative change without a new tear -> negative.
- Medial OA / Lateral OA / PF OA (medial tibiofemoral, lateral tibiofemoral, patellofemoral): an area of at least
  ~1 cm with more than 50% loss of cartilage thickness (high-grade/full-thickness loss, moderate-to-severe OA).
  Mild chondrosis, superficial fibrillation, small focal defects -> negative.
- Effusion: moderate or large joint effusion only. Trace/small/physiologic -> negative.
- Synovitis: synovitis or synovial thickening/proliferation present (including effusion-synovitis).
- Baker's: popliteal (Baker's) cyst of moderate or large size only. Small cyst -> negative.
- Contusion: bone marrow edema/contusion WITHOUT a fracture line; degenerative subchondral edema next to cartilage
  loss is not a contusion.
- Fracture: acute fracture with a cortical break (incl. osteochondral or avulsion). Old healed fracture, stress
  reaction without a fracture line -> negative.

For EVERY exam and finding give your final probability (0.00-1.00) that the official image read is positive.
Do not skip any exam or finding. Answer with ONLY one JSON object, no prose:
{"S01": {"ACL": 0.12, "MCL": ..., "Medial Meniscus": ..., "Lateral Meniscus": ..., "Medial OA": ...,
 "Lateral OA": ..., "PF OA": ..., "Effusion": ..., "Synovitis": ..., "Baker's": ..., "Contusion": ...,
 "Fracture": ...}, "S02": {...}, ...}
"""


def _gold():
    sys.path.insert(0, str(ROOT))
    from tools import exp130_gold58_compare as g130
    return g130.official_gold(ROOT)


def render(stage=1):
    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT / "training/src"))
    from raptor_common import Corpus
    ids, _ = _gold()
    if stage == 1:
        if OUT.exists():
            raise SystemExit("output exists; refusing overwrite")
        rng = np.random.default_rng(SEED)
        pick = sorted(rng.choice(len(ids), N, replace=False).tolist())
        key = {f"S{i + 1:02d}": ids[j] for i, j in enumerate(pick)}
        new = dict(key)
        used = {}
        IMG.mkdir(parents=True)
    else:  # v1.2.0 stage 2: the remaining Gold58 exams get codes S21.. in Gold58 order; stage-1 files untouched
        key = json.loads((OUT / "key_private.json").read_text(encoding="utf-8"))
        used = json.loads((OUT / "slots.json").read_text(encoding="utf-8"))
        if len(key) != N:
            raise SystemExit("stage 2 already rendered")
        rest = [u for u in ids if u not in set(key.values())]
        new = {f"S{N + i + 1:02d}": u for i, u in enumerate(rest)}
        key.update(new)
    corpus = Corpus(CORPUS)
    for code, uid in new.items():
        vol, mask = corpus.volume(uid), np.asarray(corpus.mask(uid), bool)
        used[code] = {}
        for plane, groups in GROUPS.items():
            slots = []
            for start, width in groups:
                filled = [s for s in range(start, start + width) if mask[s]]
                if len(filled) >= COUNT[plane]:
                    lo, hi = SPAN[plane]  # central part of the stack: the edge slots lie outside the joint
                    slots = [filled[int(round(k))] for k in np.linspace(lo * (len(filled) - 1), hi * (len(filled) - 1), COUNT[plane])]
                    break
            cols, rows = GRID[plane]
            canvas = np.zeros((rows * 336, cols * 336), np.uint8)
            for t, s in enumerate(slots):
                r, c = divmod(t, cols)
                tile = np.array(vol[s], np.uint8)
                cv2.putText(tile, f"{plane.upper()} {t + 1}/{len(slots)}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, 255, 1, cv2.LINE_AA)
                canvas[r * 336:(r + 1) * 336, c * 336:(c + 1) * 336] = tile
            if not slots:
                cv2.putText(canvas, f"{plane.upper()}: not available", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.imwrite(str(IMG / f"{code}_{plane}.png"), canvas)
            used[code][plane] = [int(s) for s in slots]
    (OUT / "key_private.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")  # UIDs only, never shown to readers
    (OUT / "slots.json").write_text(json.dumps(used, indent=2) + "\n", encoding="utf-8")
    (OUT / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    return {"studies": len(key), "new": len(new), "images": len(list(IMG.glob("*.png")))}


FLIGHT = ROOT / "storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv"


def inputs():
    """v1.1.0: per-exam report-derived labels (flight_v4 p + conf) under the neutral codes, and the prompt."""
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from tools import exp130_gold58_compare as g130
    key = json.loads((OUT / "key_private.json").read_text(encoding="utf-8"))
    d = pd.read_csv(FLIGHT, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    out = {c: {t: {"p": round(float(d.loc[u, t]), 3), "conf": round(float(d.loc[u, t + "__conf"]), 3)} for t in g130.TARGETS}
           for c, u in key.items()}
    (OUT / "report_labels.json").write_text(json.dumps(out, indent=1) + chr(10), encoding="utf-8")
    (OUT / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    return {"exams": len(out), "files": ["report_labels.json", "prompt.txt"]}


def score(readers, subset="all", draws=2000):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(ROOT))
    from tools import exp130_gold58_compare as g130
    ids, labels = _gold()
    y_all = np.asarray(labels, float)
    key = json.loads((OUT / "key_private.json").read_text(encoding="utf-8"))
    codes = sorted(key)
    if subset == "stage1":
        codes = [c for c in codes if int(c[1:]) <= N]
    elif subset == "stage2":
        codes = [c for c in codes if int(c[1:]) > N]
    rows = [ids.index(key[c]) for c in codes]
    y = y_all[rows]
    preds = {}
    s = [np.asarray(g130.prediction_csv(ROOT / f".local/exp144-gold58/gold58_aug1_s{sd}_tta.csv", ids), float) for sd in (42, 43)]
    preds["ours_aug1_tta"] = ((s[0] + s[1]) / 2)[rows]
    import pandas as pd
    fl = pd.read_csv(FLIGHT, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    preds["flight_v4_alone"] = fl.loc[[key[c] for c in codes], g130.TARGETS].to_numpy(float)
    for name, path in readers.items():
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        preds[name] = np.array([[float(d[c][t]) for t in g130.TARGETS] for c in codes])
    elig = [t for t in range(12) if 0 < y[:, t].sum() < len(y)]
    silent = [g130.TARGETS.index(t) for t in ("Effusion", "Synovitis", "Baker's")]
    out = {"studies": len(codes), "eligible_targets": len(elig), "readers": {}}
    for name, p in preds.items():
        per = {g130.TARGETS[t]: round(float(roc_auc_score(y[:, t], p[:, t])), 4) for t in elig}
        out["readers"][name] = {"macro_auc": round(float(np.mean(list(per.values()))), 4),
                                "silent3_auc": round(float(np.mean([per[g130.TARGETS[t]] for t in silent if t in elig])), 4),
                                "per_target": per}
    ours = out["readers"]["ours_aug1_tta"]["macro_auc"]
    for name in readers:
        gap = out["readers"][name]["macro_auc"] - ours
        out["readers"][name]["gap_vs_ours"] = round(gap, 4)
        out["readers"][name]["registered_decision"] = ("STOP (gap < -0.10)" if gap < -0.10 else
                                                       "STAGE 2 candidate (gap within 0.04)" if gap >= -0.04 else "inconclusive (-0.10..-0.04)")
    rng = np.random.default_rng(20261009)

    def macro_ix(p, ix):
        return float(np.mean([roc_auc_score(y[ix, t], p[ix, t]) for t in elig if 0 < y[ix, t].sum() < len(ix)]))
    boots = [rng.integers(0, len(codes), len(codes)) for _ in range(draws)]
    for name in readers:
        for base in ("flight_v4_alone", "ours_aug1_tta"):
            d = np.array([macro_ix(preds[name], ix) - macro_ix(preds[base], ix) for ix in boots])
            out["readers"][name][f"vs_{base}"] = {
                "delta": round(out["readers"][name]["macro_auc"] - out["readers"][base]["macro_auc"], 4),
                "ci95": [round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])],
                "p_positive": round(float((d > 0).mean()), 3)}
        v = out["readers"][name]
        v["stage2_rule"] = "EFFECTIVE" if v["vs_flight_v4_alone"]["ci95"][0] > 0 and v["gap_vs_ours"] >= -0.04 else "NOT effective"
    out["subset"] = subset
    (OUT / f"score_{subset}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rd = sub.add_parser("render"); rd.add_argument("--stage", type=int, default=1, choices=[1, 2])
    sub.add_parser("inputs")
    sc = sub.add_parser("score"); sc.add_argument("--reader", action="append", required=True)
    sc.add_argument("--subset", default="all", choices=["all", "stage1", "stage2"])
    a = ap.parse_args(argv)
    if a.cmd == "render":
        print(json.dumps(render(a.stage), indent=2))
    elif a.cmd == "inputs":
        print(json.dumps(inputs(), indent=2))
    else:
        print(json.dumps(score(dict(r.split("=", 1) for r in a.reader), a.subset), indent=2))


if __name__ == "__main__":
    main()
