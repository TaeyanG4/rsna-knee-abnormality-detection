"""Preregister exact operational DICOM ordering from the preserved local cohort."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

from dicom_header_benchmark import ROOT, digest, dump, ordering_proof, passive_adapter, read_json


def build(directory, selection_path, series_csv, prior_receipt_path, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError("choose a new immutable ordering contract path")
    selection, prior = read_json(selection_path), read_json(prior_receipt_path)
    ids = selection["ids"]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("fixed100 only")
    if digest(series_csv) != selection["train_series_sha256"]:
        raise ValueError("series source identity changed")
    reference = {r["uid"]: r for r in prior["study_receipts"]}
    with Path(series_csv).open(encoding="utf-8-sig", newline="") as f:
        rows = [{k: r[k] for k in ("StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive")}
                for r in csv.DictReader(f) if r["StudyInstanceUID"] in set(ids)]
    sys.path.insert(0, str(ROOT / "training/src"))
    adapter_path = ROOT / "training/src/raptor_dicom.py"
    adapter = passive_adapter(adapter_path, "order_reference_adapter")
    contract = "raptor64_336_c140_s02"
    spec = adapter._contract_spec(contract)
    result = {"version": "directory-order-contract-v2", "status": "PREREGISTERED",
              "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "contract": contract, "ids": ids,
              "source_sha256": {"selection": digest(selection_path), "series_csv": digest(series_csv),
                                "prior_runtime_receipt": digest(prior_receipt_path), "adapter": digest(adapter_path),
                                "tool": digest(__file__), "benchmark_helper": digest(ROOT / "tools/dicom_header_benchmark.py")},
              "scope": "new comparison contract; prior v1 failure preserved; no render/sort math or pixel gate changed",
              "gate": ["current baseline/candidate volume, mask and full receipt exact", "prior/current volume and mask exact",
                       "reference ordering digest must match original prior receipt", "current ordering digest must match current receipt",
                       "full ordered member paths/spacing hash and original DICOM geometry hash exact",
                       "all remaining receipt fields including selection, selected raw SHA, normalization exact"],
              "reason": "derived float64 dot-product serialization varies with fused arithmetic; operational input order remains separately exact",
              "studies": []}
    for uid in ids:
        proof = ordering_proof(adapter, directory, uid, [r for r in rows if r["StudyInstanceUID"] == uid], spec)
        expected = {s["slot"]: s["ordered_sources_sha256"] for s in reference[uid]["slots"] if "ordered_sources_sha256" in s}
        if {s["slot"]: s["ordered_sources_sha256"] for s in proof["slots"]} != expected:
            raise ValueError(f"local operational reference does not reproduce prior runtime digest for {uid}")
        result["studies"].append(proof)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "selection", "series-csv", "prior-receipt", "out"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    result = build(a.directory, a.selection, a.series_csv, a.prior_receipt, a.out)
    print(json.dumps({"status": result["status"], "studies": len(result["studies"])}))
