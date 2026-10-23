"""Explicit multi-checkpoint Raptor serving; no ranking, blending, or submission.

Manifest: {"members":[{"name":"F0","path":"weights/F0.pt","sha256":"..."},...]}
Paths are relative to the manifest. Outputs are a raw mean CSV, a member NPZ,
and a receipt. Acquired DICOM errors, member failures, and NaN/Inf stop the run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--series-csv", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dicom-root", help="directory containing <study>/<series>/*.dcm")
    group.add_argument("--archive", help="explicit ZIP source; requires --archive-index")
    parser.add_argument("--archive-index")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--out", required=True, help="new CSV for the unranked uniform probability mean")
    parser.add_argument("--amp", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--runtime-policy", choices=["strict", "throughput"], default="strict")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--residency", choices=["resident", "sequential"], default="sequential",
                        help="resident keeps all weights on device; sequential transfers one member per batch")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if bool(args.archive) != bool(args.archive_index):
        parser.error("--archive and --archive-index must be supplied together")
    if args.batch < 1 or args.limit < 0:
        parser.error("--batch must be positive and --limit nonnegative")
    output = Path(args.out)
    if output.suffix.lower() != ".csv":
        parser.error("--out must end in .csv")
    receipt_path, member_path = output.with_suffix(".json"), output.with_suffix(".members.npz")
    for path in (output, receipt_path, member_path):
        if path.exists():
            raise FileExistsError(f"refusing existing ensemble output: {path}")

    import numpy as np
    from raptor_common import LAB, environment, write_predictions
    from raptor_dicom import ADAPTER_VERSION, DicomCorpus, DirectoryDicomSource, IndexedZipDicomSource
    from raptor_ensemble import ENSEMBLE_VERSION, file_sha256, load_members, predict_ensemble, validate_members
    from raptor_runtime import configure_runtime

    receipt = {"status": "RUNNING", "artifact_kind": "raw_uniform_checkpoint_ensemble",
               "ensemble_version": ENSEMBLE_VERSION, "adapter_version": ADAPTER_VERSION,
               "args": vars(args), "submission_created": False, "ranking_applied": False,
               "checkpoint_weighting": "equal_probability_mean"}
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus, members = None, []
    try:
        members, manifest = load_members(args.manifest)
        contract = validate_members(members)
        runtime = configure_runtime(args.amp, args.device, args.runtime_policy)
        receipt.update(manifest=manifest, runtime=runtime, contract=contract,
                       members=[{"name": m.name, "checkpoint_sha256": m.checkpoint_sha256,
                                 "checkpoint_path": m.checkpoint_path} for m in members],
                       code_sha256={name: file_sha256(Path(__file__).with_name(name)) for name in
                                    ("raptor_ensemble_serve.py", "raptor_ensemble.py", "raptor_dicom.py",
                                     "raptor_common.py", "raptor_runtime.py")})
        source = (DirectoryDicomSource(args.dicom_root) if args.dicom_root else
                  IndexedZipDicomSource(args.archive, args.archive_index))
        with source:
            corpus = DicomCorpus(args.test_csv, args.series_csv, source, contract["contract"], args.split)
            ids = corpus.ids[:args.limit] if args.limit else corpus.ids
            receipt.update(input_manifest=corpus.manifest, requested_studies=len(corpus.ids),
                           predicted_studies=len(ids), scope="sample" if len(ids) < len(corpus.ids) else "provided_cohort")
            mean, raw, inference = predict_ensemble(members, corpus, ids, amp=args.amp,
                                                    device=runtime["device"], batch=args.batch,
                                                    residency=args.residency, runtime_policy=args.runtime_policy)
            receipt.update(inference=inference, study_receipts=[corpus.receipts[u] for u in ids],
                           environment=environment())
            # No output is produced until every requested study and member succeeds.
            # NPZ preserves the exact float32 per-member probabilities for parity.
            with member_path.open("xb") as handle:
                np.savez_compressed(handle, study_uids=np.asarray(ids),
                                    member_names=np.asarray([m.name for m in members]),
                                    targets=np.asarray(LAB), member_probabilities=raw,
                                    mean_probabilities=mean)
            write_predictions(output, ids, mean)
            receipt.update(status="COMPLETE", output=str(output), output_sha256=file_sha256(output),
                           member_outputs={"path": str(member_path), "sha256": file_sha256(member_path),
                                           "format": "npz", "probabilities_key": "member_probabilities",
                                           "shape": list(raw.shape), "names_key": "member_names",
                                           "study_ids_key": "study_uids"})
    except Exception as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                       study_receipts=list(corpus.receipts.values()) if corpus is not None else [])
        if hasattr(exc, "receipt"):
            receipt["inference"] = exc.receipt
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        raise
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"raw ensemble: {output}\nmember probabilities: {member_path}\nreceipt: {receipt_path}")
    return receipt


if __name__ == "__main__":
    main()
