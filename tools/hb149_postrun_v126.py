"""H-B-149 postrun v1.2.6 (2026-10-14).

Changelog: accept only completed full-v125, authenticate its v1.2.5 execution
and all 551 carried journals before the unchanged v1.2.4 registered readout.
v124 code, registration, exclusions, GO policy and output names are preserved.
Export-only remains an independent numeric export; it cannot produce a GO.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb149_postrun_v124 as legacy
from tools import hb149_r7_flip_v120 as flip

VERSION = "1.2.6"
WORK = legacy.WORK
FLIP_DIR = WORK / "full-v125"
RESUME_REGISTRATION = WORK / "resume-registration-v125.json"
RESUME_SHA = "fcb4439d82b1f7795e012bc047c65413705a860c699b76e09e967391ee74d643"
IDENTITY = "512c7d5d696a27f137ee29af9c5aa83b761dde0bd8141d23476add60c102ab0d"
need, sha, read = legacy.need, legacy.sha, legacy.pair.read_json


def validate_flip(directory):
    """Read only after B has stopped the producer; no alternate directory allowed."""
    directory = Path(directory)
    need(directory.resolve() == FLIP_DIR.resolve(), "only full-v125 is accepted")
    need(sha(RESUME_REGISTRATION) == RESUME_SHA, "resume registration changed")
    reg = read(RESUME_REGISTRATION)
    need(reg.get("version") == "1.2.5" and reg.get("scientific_identity") == IDENTITY,
         "resume registration identity differs")
    # Read terminal receipts first: partial output cannot reach label export/readout.
    resource = read(directory / "run_resources.json")
    need(resource.get("version") == "1.2.5" and resource.get("identity") == IDENTITY
         and resource.get("state") == "COMPLETED" and resource.get("exit_code") == 0
         and resource.get("completed_count") == 1000
         and resource.get("cleanup_complete") is True
         and resource.get("stalled") is False and resource.get("ram_guard_stopped") is False,
         "resume not COMPLETED with 1000 journals")
    receipt = read(directory / "run_receipt.json")
    need(receipt.get("version") == "1.2.0" and receipt.get("identity") == IDENTITY
         and receipt.get("models") == "heldout_flip"
         and receipt.get("selected") == receipt.get("completed") == 1000
         and receipt.get("failures") == receipt.get("decode_failures") == receipt.get("exit_code") == 0
         and receipt.get("codec_coverage_complete") is True
         and receipt.get("resources") == {k: v for k, v in resource.items() if k != "samples"}
         and receipt.get("exposure") == "EXPOSED" and receipt.get("clean_oof") is False,
         "run receipt incomplete or inconsistent")
    ident = read(directory / "identity.json")
    need(flip.fingerprint(ident) == IDENTITY and len(ident["uids"]) == len(set(ident["uids"])) == 1000,
         "scientific identity differs")
    amendment = read(directory / "execution-amendment-v125.json")
    hashes = reg["journals"]
    expected_names = {flip.record_path(directory, u).name for u in ident["uids"]}
    need(len(hashes) == 551 and set(hashes) <= expected_names
         and amendment.get("version") == "1.2.5"
         and amendment.get("registration_sha256") == RESUME_SHA
         and amendment.get("scientific_identity") == IDENTITY
         and amendment.get("carried") == 551 and amendment.get("journal_hashes") == hashes
         and amendment.get("executed_runtime_sha256") == reg["executed_runtime_sha256"]
         and amendment.get("execution") == reg["execution"], "carry-over manifest differs")
    execution = read(directory / "execution-runtime-v125.json")
    runtime = directory / "stage/r7_runtime"
    need(execution.get("version") == "1.2.5"
         and execution.get("registration_sha256") == RESUME_SHA
         and execution.get("engineering_only") is False
         and execution.get("runtime_sha256") == reg["executed_runtime_sha256"]
         == sha(runtime / "compact_infer.py")
         and execution.get("runtime_manifest_sha256") == sha(runtime / "runtime_sha256.json")
         and read(runtime / "runtime_sha256.json").get("compact_infer.py") == reg["executed_runtime_sha256"],
         "execution runtime binding differs")
    need({p.name for p in (directory / "journal").glob("*.json")} == expected_names,
         "expected exactly 1000 journal files")
    need(not any((directory / "journal").glob("*.tmp")), "uncommitted journal temporary file")
    need(all(sha(directory / "journal" / name) == digest for name, digest in hashes.items()),
         "carried journals changed")
    records = flip.load_done(directory, ident["uids"], IDENTITY)
    need(len(records) == 1000 and all(r.get("models") == "heldout_flip"
         and r.get("status") != "failure" and r.get("decode_failures") == 0
         and r.get("decoded_files") == r.get("dicom_files")
         and r.get("exposure") == "EXPOSED" and r.get("clean_oof") is False
         for r in records.values()), "invalid completed journals")
    files = ("run_receipt.json", "run_resources.json", "identity.json",
             "execution-amendment-v125.json", "execution-runtime-v125.json")
    return dict(version=VERSION, **legacy.pair.base.EXPOSURE, scientific_identity=IDENTITY,
                resume_registration_sha256=RESUME_SHA, journals=1000, carried_byte_identical=551,
                evidence_sha256={name: sha(directory / name) for name in files},
                code_sha256=sha(__file__))


def run(selection, baseline, flip_dir, out, label_exclusion):
    proof = validate_flip(flip_dir)
    legacy.registration()
    legacy.output_io.atomic_json(out / "flip-validation-v126.json", proof)
    return legacy.run(selection, baseline, flip_dir, out, label_exclusion)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, choices=[legacy.DRAW_COUNT], default=legacy.DRAW_COUNT)
    parser.add_argument("--studies", type=int, choices=[legacy.STUDIES], default=legacy.STUDIES)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.export_only:
        return legacy.main(["--export-only", "--draws", str(args.draws), "--studies", str(args.studies)])
    try:
        need(sha(legacy.pair.SELECTION) == legacy.SELECTION_SHA, "frozen selection hash differs")
        run(legacy.pair.SELECTION, legacy.pair.BASELINE, FLIP_DIR, legacy.OUT, legacy.LABEL_EXCLUSION)
    except Exception as error:
        print(f"POSTRUN=BLOCKED error_type={type(error).__name__}; inspect aggregate receipts", file=sys.stderr)
        print("SUBMIT_CANDIDATE=NONE")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
