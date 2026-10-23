"""EXP147 v1.2.1 (robust parsing + one format-only re-ask) (variant F: report text + report label + 15-slot images + verbatim rules) (reader comparison on variant B: --model/--tag, engine codex|claude) — which IMAGE INPUT gives the best image-checked labels? (B, 2026-10-09)

User (02:3x KST): "<owner remark removed>" — before the one-shot
full labelling run (lyra_v1), compare input designs on the 58 Gold58 exams with the SAME reader (Sol), the same
report label (flight_v4 p + conf), the same official rules and the same neutral codes/key as EXP145.

  A  15 central slots @336 (EXP145 montages), 5 exams per call         - existing EXP145 Sol result
  B  15 central slots @336 (same images), 1 exam per call              - batching effect
  C  36 slots @384 (EXP143 96-slot corpus: SAG-FS 12, SAG 4, COR-FS 8, COR 4, AX 8), 1 exam per call
  D  62 slots @384 (all fluid-sensitive: SAG-FS 26, COR-FS 18, AX 18), 1 exam per call
  E  96 slots @384 (every slot of every series), 1 exam per call
C-E montages are 2 rows x 3 tiles of 384 px (1152 x 768) so neither reader downscales them.
Registered choice rule (before any reading): highest macro AUC (12 targets, 58 exams); a cheaper variant within
0.005 of the best wins; ties broken by the silent-prone mean (Effusion, Synovitis, Baker's) and the meniscus mean.

  python tools/exp147_input_ab.py render
  python tools/exp147_input_ab.py run --variant B --parallel 6
  python tools/exp147_input_ab.py score
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp145_vlm_reading_pilot as p145  # noqa: E402

WORK = ROOT / ".local/exp147-input-ab"
C143 = ROOT / "storage/cache/derived/EXP143_RAPTOR96_384_C140_S02"
SERIES = [("SAG-FS", "sagittal fat-suppressed / fluid-sensitive", 0, 26), ("SAG", "sagittal non-fat-suppressed (T1/PD)", 26, 22),
          ("COR-FS", "coronal fat-suppressed / fluid-sensitive", 48, 18), ("COR", "coronal non-fat-suppressed (T1/PD)", 66, 12),
          ("AX", "axial", 78, 18)]
PICK = {"C": {"SAG-FS": 12, "SAG": 4, "COR-FS": 8, "COR": 4, "AX": 8},
        "D": {"SAG-FS": 26, "COR-FS": 18, "AX": 18},
        "E": {"SAG-FS": 26, "SAG": 22, "COR-FS": 18, "COR": 12, "AX": 18}}
COST_ORDER = ["A", "B", "C", "D", "E"]
RULES = p145.PROMPT[p145.PROMPT.index("Decide each finding the way"):]


def _key():
    return json.loads((p145.OUT / "key_private.json").read_text(encoding="utf-8"))


def render():
    import cv2
    import numpy as np
    key = _key()
    man = json.loads((C143 / "manifest.json").read_text(encoding="utf-8"))
    row = {f["uid"]: f["row"] for f in man["files"]}
    layout = {}
    for v, pick in PICK.items():
        out = WORK / f"reader_{v}"
        out.mkdir(parents=True, exist_ok=True)
        layout[v] = {}
        for code, uid in sorted(key.items()):
            r = row[uid]
            vol = np.load(C143 / f"study_{r:05d}.npy", mmap_mode="r")
            mask = np.asarray(json.loads((C143 / f"study_{r:05d}.json").read_text(encoding="utf-8"))["mask"], bool)
            files = []
            for name, desc, start, width in SERIES:
                if name not in pick:
                    continue
                filled = [s for s in range(start, start + width) if mask[s]]
                n = min(pick[name], len(filled))
                if n == 0:
                    continue
                if n < len(filled):  # sub-sample from the central 10-90 % of the stack
                    sel = [filled[int(round(k))] for k in np.linspace(.1 * (len(filled) - 1), .9 * (len(filled) - 1), n)]
                else:
                    sel = filled
                for m, chunk in enumerate([sel[i:i + 6] for i in range(0, len(sel), 6)]):
                    canvas = np.zeros((768, 1152), np.uint8)
                    for t, s in enumerate(chunk):
                        rr, cc = divmod(t, 3)
                        tile = np.array(vol[s], np.uint8)
                        cv2.putText(tile, f"{name} {sel.index(s) + 1}/{len(sel)}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 2, cv2.LINE_AA)
                        canvas[rr * 384:(rr + 1) * 384, cc * 384:(cc + 1) * 384] = tile
                    fn = f"{code}_{name}_{m + 1}.png"
                    cv2.imwrite(str(out / fn), canvas)
                    files.append({"file": fn, "series": name, "desc": desc, "tiles": [sel.index(s) + 1 for s in chunk], "of": len(sel)})
            layout[v][code] = files
    (WORK / "layout.json").write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
    return {v: {"exams": len(d), "images_per_exam_mean": round(sum(len(x) for x in d.values()) / len(d), 1)} for v, d in layout.items()}


def _prompt(variant, code, labels, layout):
    if variant == "B":
        txt = p145.PROMPT.replace("(codes S01..S20 in this batch)", f"(exam {code} only)")
        txt += f"\nThe attached images are, in order: {code}_sag.png, {code}_cor.png, {code}_ax.png.\n"
    else:
        lines = [f"  {f['file']}: {f['desc']} series, tiles {f['tiles'][0]}-{f['tiles'][-1]} of {f['of']} "
                 f"(ordered across the knee; tile label = series and position)" for f in layout[code]]
        txt = ("You are an experienced musculoskeletal radiologist producing final labels for ONE knee MRI exam (code "
               f"{code}). You get:\n  1) report_labels.json - what a text model extracted from the radiologist's REPORT for this "
               "exam: for each finding \"p\" = probability the finding is present per the report, \"conf\" = extraction confidence "
               "(low conf usually means the report is ambiguous or does not mention the finding);\n  2) montage images of the "
               "exam itself (each tile 384 px, 140 mm field of view, windowed per series; left/right knee and sagittal "
               "direction not given - tell medial from lateral by anatomy: fibular head = lateral, MCL = medial). "
               "Look at every tile of every series. The attached images are, in order:\n" + "\n".join(lines) + "\n")
        txt += RULES.replace("For EVERY exam and finding", "For this exam and EVERY finding")
    txt += "\nreport_labels.json for this exam:\n" + json.dumps({code: labels[code]}, indent=1)
    txt += f"\n\nDo not open any file or run any command; use only the attached images and the data in this prompt. Return ONLY the JSON object {{\"{code}\": {{...12 findings...}}}}.\n"
    return txt


RULE_DOC = ROOT / "docs/runbooks/relabel-prompt-v1_1.md"


def _rules_full():
    """v1.2.0 (variant F): the relabel v1.1 task paragraph and official definitions VERBATIM from the frozen doc."""
    doc = RULE_DOC.read_text(encoding="utf-8")
    intro = doc[doc.index("You read knee MRI radiology reports"):doc.index("\n", doc.index("You read knee MRI radiology reports"))]
    defs = doc[doc.index("Official definitions:"):doc.index("For each finding output:")]
    return intro, defs


def _prompt_f(code, report, labels):
    """Variant F (user 2026-10-09: "<owner remark removed>"): report TEXT + report-derived label + 15-slot images + full rules.
    The report text is piped by this script only; B never reads or prints it."""
    intro, defs = _rules_full()
    return ("You are an experienced musculoskeletal radiologist producing final labels for ONE knee MRI exam (code "
            f"{code}).\n\nTASK (from the competition labelling guideline):\n{intro}\nHere you ALSO have the exam's images, "
            "so decide each finding from the report AND the images: use the report where it is explicit; use the images to "
            "decide findings the report leaves ambiguous, ungraded or unmentioned, and to correct the report label when the "
            "images clearly disagree.\n\n" + defs +
            "\nIf a thresholded finding is mentioned without grade/size and the images do not settle it, give a probability "
            "between 0.4 and 0.6.\n\nINPUTS\n1) RADIOLOGY REPORT (verbatim; read it in its own language):\n<<<REPORT\n"
            f"{report}\nREPORT>>>\n\n2) report_labels.json - a text model's extraction from the same report (p = probability "
            "present, conf = extraction confidence):\n" + json.dumps({code: labels[code]}, indent=1) +
            "\n\n3) Images (three montages; 336 px tiles, 140 mm field of view, windowed per series; left/right knee and the "
            "sagittal direction are not given - tell medial from lateral by anatomy: fibular head = lateral, MCL = medial):\n"
            f"The attached images are, in order: {code}_sag.png (7 sagittal fluid-sensitive tiles across the joint), "
            f"{code}_cor.png (4 coronal tiles, front to back), {code}_ax.png (4 axial tiles, top to bottom).\n\n"
            "For this exam give your final probability (0.00-1.00) that the official image read is positive for EVERY one of: "
            "ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, "
            "Fracture.\nDo not open any file or run any command; use only the attached images and the data in this prompt. "
            f"Return ONLY the JSON object {{\"{code}\": {{...12 findings...}}}}.\n")


def run(variant, parallel=6, model="gpt-6-sol", tag="", engine="codex"):
    labels = json.loads((p145.OUT / "report_labels.json").read_text(encoding="utf-8"))
    layout = json.loads((WORK / "layout.json").read_text(encoding="utf-8")) if variant not in ("B", "F") else {}
    img_dir = (p145.OUT / "reader") if variant in ("B", "F") else (WORK / f"reader_{variant}")
    reports = {}
    if variant == "F":
        import pandas as pd
        df = pd.read_csv(ROOT / "storage/competition/train.csv", usecols=["StudyInstanceUID", "Report"], dtype=str).set_index("StudyInstanceUID")
        reports = {c: str(df.loc[u, "Report"]) for c, u in _key().items()}  # never printed or written by B
    out_dir = WORK / (variant + (f"_{tag}" if tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = sorted(_key())
    todo = [c for c in codes if not (out_dir / f"{c}.json").exists()]

    def one(code):
        try:
            return one_try(code, "")
        except Exception as e:  # v1.2.1: one exam's error never stops the pool
            (out_dir / f"{code}.failed.txt").write_text(repr(e), encoding="utf-8")
            return code, "failed"

    def one_try(code, _unused):
        st = attempt(code, "")
        if st[1] == "unparsable":  # one format-only re-ask; the content instructions are unchanged
            st = attempt(code, "\nFORMAT: every one of the 12 values must be a plain number between 0 and 1 (no text, no objects).\n")
        return st

    def attempt(code, suffix):
        files = ([f"{code}_{p}.png" for p in ("sag", "cor", "ax")] if variant in ("B", "F") else [f["file"] for f in layout[variant][code]])
        prompt = (_prompt_f(code, reports[code], labels) if variant == "F" else _prompt(variant, code, labels, layout.get(variant, {}))) + suffix
        raw = out_dir / f"{code}.raw.md"
        if engine == "claude":  # v1.1.0: Claude CLI reads the same images with its Read tool, one exam per process
            paths = ", ".join(str(img_dir / f) for f in files)
            prompt = re.sub(r"The attached images are, in order:[^\n]*\n",
                            lambda _m: f"Open these image files with the Read tool (in this order) before deciding: {paths}\n", prompt, count=1)
            prompt = prompt.replace("Do not open any file or run any command; use only the attached images",
                                    "Do not open any other file and do not run commands; use only these images")
            args = ["claude", "-p", "--model", model, "--allowedTools", "Read", "--add-dir", str(img_dir), "--output-format", "text"]
            r = subprocess.run(args, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=1200, cwd=str(img_dir))
            raw.write_text(r.stdout, encoding="utf-8")
            tok = []
        else:
            args = ["codex", "exec", "-m", model, "-s", "read-only", "-C", str(img_dir), "--skip-git-repo-check"]
            for f in files:
                args += ["-i", str(img_dir / f)]
            args += ["-o", str(raw), "-"]
            r = subprocess.run(args, input=prompt, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=1200)
            tok = re.findall(r"tokens used\s*\n?\s*([\d,]+)", r.stdout + r.stderr)
        text = raw.read_text(encoding="utf-8") if raw.exists() else ""
        parsed = p145_parse(text, code)
        if parsed:
            (out_dir / f"{code}.json").write_text(json.dumps({"labels": parsed, "tokens": int(tok[-1].replace(",", "")) if tok else None,
                                                              "format_reask": bool(suffix)}) + "\n", encoding="utf-8")
            return code, "ok"
        if text.strip() and not suffix:
            return code, "unparsable"
        (out_dir / f"{code}.failed.txt").write_text((r.stdout[-3000:] + r.stderr[-3000:]), encoding="utf-8")
        return code, "failed"

    started, counts = time.time(), {}
    with ThreadPoolExecutor(parallel) as ex:
        for code, st in ex.map(one, todo):
            counts[st] = counts.get(st, 0) + 1
    return {"variant": variant, "tag": tag, "model": model, "engine": engine, "counts": counts, "minutes": round((time.time() - started) / 60, 1)}


def p145_parse(text, code):
    from tools import exp130_gold58_compare as g130
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    d = d.get(code, d)
    if not isinstance(d, dict) or any(t not in d for t in g130.TARGETS):
        return None
    out = {}
    for t in g130.TARGETS:  # v1.2.1: accept {"s":..,"p":..} objects; any non-numeric value -> unparsable (format re-ask)
        v = d[t]
        if isinstance(v, dict):
            v = v.get("p")
        try:
            out[t] = min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            return None
    return out


def score():
    readers, tokens = {"A": str(p145.OUT / "out_sol.json")}, {}
    extra = sorted(d.name for d in WORK.iterdir() if d.is_dir() and (d.name.startswith("B_") or d.name.startswith("F")))
    for v in ("B", "C", "D", "E", *extra):
        d = WORK / v
        files = sorted(d.glob("S*.json")) if d.exists() else []
        if len(files) != 58:
            continue
        merged, tk = {}, []
        for f in files:
            j = json.loads(f.read_text(encoding="utf-8"))
            merged[f.stem] = j["labels"]
            if j.get("tokens"):
                tk.append(j["tokens"])
        (WORK / f"out_{v}.json").write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
        readers[v] = str(WORK / f"out_{v}.json")
        tokens[v] = round(sum(tk) / len(tk)) if tk else None
    res = p145.score(readers, "all")
    rows = {}
    for v in readers:
        r = res["readers"][v]
        pt = r["per_target"]
        rows[v] = {"macro": r["macro_auc"], "silent3": r["silent3_auc"],
                   "meniscus": round((pt["Medial Meniscus"] + pt["Lateral Meniscus"]) / 2, 4),
                   "vs_flight": r["vs_flight_v4_alone"], "tokens_per_exam": tokens.get(v)}
    best = max(rows, key=lambda v: rows[v]["macro"])
    input_rows = {v: x for v, x in rows.items() if v in COST_ORDER}  # the input-design rule ignores reader variants
    best_in = max(input_rows, key=lambda v: input_rows[v]["macro"])
    choice = next(v for v in COST_ORDER if v in input_rows and input_rows[v]["macro"] >= input_rows[best_in]["macro"] - 0.005)
    out = {"rows": rows, "best_macro": best, "registered_choice": choice,
           "ours": res["readers"]["ours_aug1_tta"]["macro_auc"], "flight": res["readers"]["flight_v4_alone"]["macro_auc"]}
    (WORK / "comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("render")
    r = sub.add_parser("run"); r.add_argument("--variant", required=True, choices=["B", "C", "D", "E", "F"]); r.add_argument("--parallel", type=int, default=6)
    r.add_argument("--model", default="gpt-6-sol"); r.add_argument("--tag", default=""); r.add_argument("--engine", default="codex", choices=["codex", "claude"])
    sub.add_parser("score")
    a = ap.parse_args(argv)
    print(json.dumps(render() if a.cmd == "render" else run(a.variant, a.parallel, a.model, a.tag, a.engine) if a.cmd == "run" else score(), indent=2))


if __name__ == "__main__":
    main()
