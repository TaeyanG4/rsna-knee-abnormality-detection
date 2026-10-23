"""Serve one existing Raptor checkpoint from previously unseen DICOM studies.

Writes raw probabilities and input/runtime receipts, never a blended Kaggle
submission. All data paths are explicit; no train labels or Reports are read.

Example (run only through the authorized GPU job workflow):
  python training/src/raptor_serve.py --ckpt MODEL.pt --test-csv test.csv \
    --series-csv test_series.csv --dicom-root test_series --out raw_predictions.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--series-csv", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dicom-root", help="directory holding <study>/<series>/*.dcm")
    source.add_argument("--archive", help="explicit local ZIP, used with --archive-index")
    parser.add_argument("--archive-index")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="ZIP index partition; no labels are loaded")
    parser.add_argument("--out", required=True, help="new raw prediction CSV path; existing outputs are rejected")
    parser.add_argument("--amp", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--runtime-policy", choices=["strict", "throughput"], default="strict")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="explicit sample only; receipt marks reduced scope")
    args = parser.parse_args(argv)
    if bool(args.archive) != bool(args.archive_index):
        parser.error("--archive and --archive-index must be provided together")
    if args.batch < 1 or args.limit < 0:
        parser.error("--batch must be positive and --limit nonnegative")
    output = Path(args.out)
    receipt_path = output.with_suffix(".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing existing raw output/receipt: {output}")

    # Passive helpers only. No imports from training or acquisition entry points.
    from raptor_common import LAB, environment, load_checkpoint, write_predictions
    from raptor_dicom import ADAPTER_VERSION, DicomCorpus, DirectoryDicomSource, IndexedZipDicomSource
    from raptor_runtime import checkpoint_norm, configure_runtime, predict_batches
    runtime = configure_runtime(args.amp, args.device, args.runtime_policy)
    device = runtime["device"]
    model, metadata = load_checkpoint(Path(args.ckpt))
    contract = metadata.get("contract")
    if not contract:
        raise ValueError("checkpoint must identify its corpus contract")
    if metadata.get("lab", LAB) != LAB:
        raise ValueError("checkpoint target order does not match Raptor's 12 targets")
    norm, norm_source = checkpoint_norm(metadata)
    window_policy = metadata.get("window_policy", "legacy")
    k_eval, resolution = int(metadata.get("k_eval", 24)), int(metadata.get("res", 384))
    if k_eval < 1 or resolution < 1:
        raise ValueError("checkpoint has invalid k_eval/resolution")
    input_source = (DirectoryDicomSource(args.dicom_root) if args.dicom_root else
                    IndexedZipDicomSource(args.archive, args.archive_index))
    receipt = {"status": "RUNNING", "artifact_kind": "raw_checkpoint_probabilities",
               "adapter": ADAPTER_VERSION, "checkpoint": str(Path(args.ckpt).resolve()),
               "checkpoint_sha256": _hash(args.ckpt), "contract": contract, "runtime": runtime,
               "norm": norm, "norm_source": norm_source, "window_policy": window_policy,
               "k_eval": k_eval, "resolution": resolution, "args": vars(args),
               "code_sha256": {name: _hash(Path(__file__).with_name(name)) for name in
                                  ("raptor_serve.py", "raptor_dicom.py", "raptor_common.py", "raptor_runtime.py")}}
    corpus = None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with input_source:
            corpus = DicomCorpus(args.test_csv, args.series_csv, input_source, contract, args.split)
            ids = corpus.ids[:args.limit] if args.limit else corpus.ids
            receipt.update(input_manifest=corpus.manifest, requested_studies=len(corpus.ids),
                           predicted_studies=len(ids), scope="sample" if len(ids) < len(corpus.ids) else "provided_cohort")
            probabilities, inference = predict_batches(model, corpus, ids, k_eval, resolution,
                                                        args.amp, device, norm=norm, batch=args.batch,
                                                        window_policy=window_policy)
            # Only a complete, finite inference reaches the prediction write.
            write_predictions(output, ids, probabilities)
            receipt.update(status="COMPLETE", inference=inference, output_sha256=_hash(output),
                           study_receipts=[corpus.receipts[u] for u in ids], environment=environment())
    except Exception as exc:
        receipt.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                       study_receipts=list(corpus.receipts.values()) if corpus is not None else [])
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        raise
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"raw predictions: {output}\nreceipt: {receipt_path}")
    return receipt


if __name__ == "__main__":
    main()
