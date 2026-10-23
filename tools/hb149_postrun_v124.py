"""H-B-149 postrun v1.2.4 (2026-10-14).

Changelog v1.2.4: labelled-intersection readout. The registered L2 export covers
989 of the frozen 1,000 selected studies; the 11 unlabelled studies are a
label-availability exclusion registered before any flip result
(label-availability-exclusions-v124.json, hash-bound in
postrun-registration-v124.json). The baseline may carry exactly the registered
D-B-177 BadZipFile data exclusions (pairing v1.2.4), none in the selection.
Every GO rule, variant, draw count, seed and runtime gate is unchanged.
v1.2.3 (tools/hb149_postrun_v123.py) is preserved and now fails closed.
Changelog v1.2.3: new CPU-only, registered numeric L2 export, paired readout and
aggregate decision card. No release authorization; no cohort/label imputation.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb149_r7_pairing as pair

VERSION = "1.2.4"
WORK = ROOT / ".local/scratch/astra-b5/H-B-149"
CONFIG = ROOT / "training/configs/hb122-control-s42-f0-v1.json"
SOURCE = "storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv"
SOURCE_SHA = "79ce307e246cc54a9a90f32fd1ab21d58ebd23b4b7e7b1a80207adcea1bcc2c9"
SELECTION_SHA = "c1df38a3aafbb2a144f95cbb3942e9fbedd2a385bf2dec80a68aab09aa09f71b"
REGISTRATION = WORK / "registration-v121.json"
AMENDMENT = WORK / "postrun-registration-v124.json"
LABEL_EXCLUSION = WORK / "label-availability-exclusions-v124.json"
OUT = WORK / "postrun-v124"
FLIP_DIR = WORK / "full-v122"
PRODUCTION_IDENTITY = "f46bbdc29c71c7b5196987b2ea43a4d5c256f4b0c02b0e4fcde1ee2fb48480aa"
CODE_FILES = ("tools/hb149_postrun_v124.py", "tools/hb149_r7_pairing.py",
              "tools/hb145_baseline_contract_v113.py", "tools/hb145_r7_carryover_v113.py",
              "tools/hb145_output_io.py", "tools/hb145_data_exclusions_v1.py")
DRAW_COUNT = 2000
STUDIES = 1300
output_io = pair.base.output_io
sha = pair.base.r7.s6.sha256


def need(ok, message):
    if not ok:
        raise ValueError(message)


def registration():
    r = pair.read_json(REGISTRATION)
    expected = dict(registered_before_real_readout=True,
                    variants={k: list(v) for k, v in pair.VARIANTS.items()},
                    macro_delta_min=.001, paired_ci_lower_strictly_above=0,
                    per_target_delta_min=-.005, minimum_valid_bootstrap_fraction=.95,
                    seed=pair.SEED, draws=DRAW_COUNT, planning_hidden_studies=STUDIES,
                    maximum_projected_hours=7.5, runtime_gate_calibration=1,
                    release_base=pair.RELEASE_BASE)
    need(all(r.get(k) == v for k, v in expected.items()), "registered GO policy differs")
    need(pair.VERSION == VERSION and pair.TARGET_FLOOR == -.005 and pair.MAX_HOURS == 7.5,
         "pairing implementation version/policy differs")
    amendment = pair.read_json(AMENDMENT)
    need(amendment.get("version") == VERSION
         and amendment["historical_registration_sha256"] == sha(REGISTRATION)
         and amendment["historical_pairing_sha256"] == r["pairing_sha256"]
         and amendment["source_identity"] == r["production_identity"]
         and amendment["production_identity"] == PRODUCTION_IDENTITY
         and amendment["decision_policy_unchanged"] is True
         and amendment["flip_output_present_at_registration"] is False
         and amendment["label_exclusion_sha256"] == sha(LABEL_EXCLUSION)
         and amendment["data_exclusion_sha256"] == pair.data_exclusions.REGISTERED_SHA256
         == sha(pair.data_exclusions.PATH),
         "registration lineage differs")
    code_root = Path(__file__).resolve().parents[1]
    need(set(amendment["code_sha256"]) == set(CODE_FILES)
         and all(sha(code_root / name) == digest for name, digest in amendment["code_sha256"].items()),
         "registered postrun code differs")
    return r


def export_l2(selection, out, label_exclusion):
    """Read only the registered numeric export; record hashes, never label rows.

    The selected studies without an L2 row must equal the registered
    label-availability exclusion exactly; nothing is filled or borrowed.
    """
    out = pair.owned_output(out)
    out.mkdir(parents=True, exist_ok=True)
    config = pair.read_json(CONFIG)
    spec = config["config"]["labels"]
    need(spec["path"] == SOURCE and config["inputs"]["data"].get(SOURCE) == SOURCE_SHA,
         "registered L2 source differs")
    source = (ROOT / SOURCE).resolve()
    need(source.is_relative_to((ROOT / "storage/labels/derived_constellation").resolve()),
         "L2 source outside registered numeric directory")
    need(sha(source) == SOURCE_SHA, "registered L2 hash differs")
    selection_hash = sha(selection)
    uids = pair.read_json(selection)["uids"]
    need(len(uids) == len(set(uids)), "duplicate selection UID")
    excluded, exclusion = pair.label_exclusions(label_exclusion, selection, uids)
    need(exclusion["source_sha256"] == SOURCE_SHA, "label-availability exclusion bound to another L2 source")
    selected, seen = {}, set()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [pair.base.UID, *pair.base.LABELS], "L2 numeric-only schema differs")
        for row in reader:
            uid = row[pair.base.UID]
            need(uid and uid not in seen and None not in row, "L2 UID/row schema differs")
            seen.add(uid)
            try:
                values = [float(row[t]) if row[t].strip() else None for t in pair.base.LABELS]
            except (ValueError, TypeError, AttributeError):
                raise ValueError("malformed numeric L2 cell") from None
            need(all(v is None or math.isfinite(v) and 0 <= v <= 1 for v in values),
                 "L2 numeric range differs")
            if uid in uids:
                selected[uid] = [row[t] for t in pair.base.LABELS]
    need(sha(source) == SOURCE_SHA and sha(selection) == selection_hash, "input changed during export")
    missing = set(uids) - selected.keys()
    ready = missing == excluded
    receipt = dict(version=VERSION, **pair.base.EXPOSURE, source=SOURCE,
                   source_sha256=SOURCE_SHA, config_sha256=sha(CONFIG),
                   selection_sha256=selection_hash, source_studies=len(seen),
                   selected_studies=len(uids), covered_studies=len(selected),
                   missing_studies=len(missing), registered_label_exclusions=len(excluded),
                   label_exclusion_sha256=exclusion["sha256"],
                   status="READY_LABELLED_INTERSECTION" if ready else "COVERAGE_BLOCKED")
    if not ready:
        output_io.atomic_json(out / "l2-export-receipt.json", receipt)
        raise ValueError("registered L2 coverage differs from the registered label-availability exclusion")
    dest = out / "l2-numeric.csv"
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([pair.base.UID, *pair.base.LABELS])
    writer.writerows([uid, *selected[uid]] for uid in uids if uid not in excluded)
    temporary = dest.with_suffix(".csv.tmp")
    output_io.retry(lambda: temporary.write_text(buf.getvalue(), encoding="utf-8", newline=""))
    output_io.replace(temporary, dest)
    receipt["export_sha256"] = sha(dest)
    output_io.atomic_json(out / "l2-export-receipt.json", receipt)
    return dest, receipt


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def decision_lines(result, engineering=False):
    """Recheck every registered condition, including the floor; never trust a GO string."""
    registration()
    need(result.get("exposure") == "EXPOSED" and result.get("clean_oof") is False
         and result.get("release_authorized") is False, "readout exposure/release scope differs")
    lines, decisions = [], {}
    scope = f"evaluated_studies={result.get('evaluated_studies', 'NA')}/{result.get('selected_studies', 'NA')}"
    for name in pair.VARIANTS:
        s = result["variants"][name]
        d = s["macro"]["delta_auc"]
        ci = d["ci95"]
        targets = s["per_target"]
        need([t["target"] for t in targets] == pair.base.LABELS, "readout targets differ")
        ds = [t["delta_auc"]["value"] for t in targets]
        worst = min(ds) if all(finite(x) for x in ds) else None
        timing = pair.runtime_estimate(STUDIES, name)
        need(s["runtime"] == timing and s["draws"] == DRAW_COUNT and s["seed"] == pair.SEED,
             "readout runtime/bootstrap registration differs")
        floor = worst is not None and worst >= pair.TARGET_FLOOR
        go = (finite(d["value"]) and d["value"] >= .001 and ci is not None and len(ci) == 2
              and all(finite(x) for x in ci) and ci[0] > 0
              and d["valid_bootstraps"] >= .95 * DRAW_COUNT and floor and timing["within_go_budget"])
        need(s["go_gate"] == ("GO" if go else "HOLD"), "readout GO gate disagrees with registration")
        decisions[name] = bool(go) and not engineering
        fmt = lambda x: f"{x:+.6f}" if finite(x) else "NA"
        bounds = ",".join(fmt(x) for x in ci) if ci else "NA,NA"
        lines.append(f"{name} GO={'ENGINEERING_ONLY' if engineering else 'GO' if go else 'HOLD'} "
                     f"macro_delta={fmt(d['value'])} CI95=[{bounds}] "
                     f"weak_four_delta={fmt(s['weak_four']['delta_auc']['value'])} "
                     f"worst_target_delta={fmt(worst)} floor=-0.005000 floor_pass={int(floor)} "
                     f"runtime_h={timing['total_hours_unscaled']:.6f} limit_h=7.500000 "
                     f"valid_bootstraps={d['valid_bootstraps']}/{DRAW_COUNT} {scope} EXPOSED clean_oof=false")
    lines.append("SUBMIT_CANDIDATE=" + ("FLIP_F0F2" if decisions["r7_flip_f0_f2"] else "NONE"))
    return lines


def run(selection, baseline, flip_dir, out, label_exclusion, engineering=False):
    registration()
    labels, exported = export_l2(selection, out, label_exclusion)
    # Frozen pairing writes CSV directly. Keep that output provisional, retry
    # sharing violations, and publish only after the entire readout succeeds.
    result_path = out / "paired-readout-v124.json"
    pending = out / "paired-readout-v124.pending.json"
    result = output_io.retry(lambda: pair.readout(selection, baseline, flip_dir, labels, pending,
                                                  draws=DRAW_COUNT, studies=STUDIES,
                                                  label_exclusion=label_exclusion))
    need(result["labels_sha256"] == exported["export_sha256"], "L2 export changed during readout")
    need(result["label_availability_exclusion"]["sha256"] == exported["label_exclusion_sha256"]
         and result["evaluated_studies"] == exported["covered_studies"]
         and result["selected_studies"] == exported["selected_studies"], "labelled intersection differs")
    amendment = pair.read_json(AMENDMENT)
    if not engineering:
        need(result["baseline_identity"] == amendment["production_identity"]
             and result["selection_sha256"] == SELECTION_SHA
             and result["label_availability_exclusion"]["sha256"] == amendment["label_exclusion_sha256"],
             "production postrun lineage differs")
    result["engineering_only"] = engineering
    lines = decision_lines(result, engineering)
    output_io.replace(pending.with_suffix(".csv"), result_path.with_suffix(".csv"))
    output_io.atomic_json(result_path, result)
    receipt = dict(version=VERSION, **pair.base.EXPOSURE, engineering_only=engineering,
                   release_authorized=False, registration_sha256=sha(REGISTRATION),
                   amendment_sha256=sha(AMENDMENT),
                   code_sha256=sha(__file__), pairing_sha256=sha(pair.__file__),
                   readout_sha256=sha(result_path), labels_sha256=exported["export_sha256"],
                   aggregate_csv_sha256=sha(result_path.with_suffix(".csv")),
                   baseline_identity=result["baseline_identity"], flip_identity=result["flip_identity"],
                   selection_sha256=result["selection_sha256"],
                   selected_studies=result["selected_studies"], evaluated_studies=result["evaluated_studies"],
                   label_availability_exclusion=result["label_availability_exclusion"],
                   baseline_data_exclusions=result["baseline_data_exclusions"], decision_lines=lines,
                   scope="candidate screen only; serving blend validation and exact T4 timing still required")
    output_io.atomic_json(out / "postrun-receipt-v124.json", receipt)
    print("\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, choices=[DRAW_COUNT], default=DRAW_COUNT)
    parser.add_argument("--studies", type=int, choices=[STUDIES], default=STUDIES)
    parser.add_argument("--export-only", action="store_true")
    a = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        registration()
        need(sha(pair.SELECTION) == SELECTION_SHA, "frozen selection hash differs")
        if a.export_only:
            _, receipt = export_l2(pair.SELECTION, OUT, LABEL_EXCLUSION)
            print(f"L2_EXPORT={receipt['status']} studies={receipt['covered_studies']}/{receipt['selected_studies']} "
                  f"label_exclusions={receipt['registered_label_exclusions']} sha256={receipt['export_sha256']}")
        else:
            run(pair.SELECTION, pair.BASELINE, FLIP_DIR, OUT, LABEL_EXCLUSION)
    except Exception as error:
        # No exception values/traceback: malformed input can contain confidential text.
        print(f"POSTRUN=BLOCKED error_type={type(error).__name__}; inspect aggregate receipts", file=sys.stderr)
        print("SUBMIT_CANDIDATE=NONE")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
