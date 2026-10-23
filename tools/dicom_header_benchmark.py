"""CPU-only directory header-reader parity; never reads labels or model weights.

Optionally stage only the fixed studies' selected series from a read-only indexed
ZIP. Inputs stay unchanged and output directories are never reused. Timings are
warm-cache CPU comparisons, not a promise about Kaggle I/O or end-to-end runtime.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def passive_adapter(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def render(adapter, directory, uid, records, spec, enabled):
    class Observed(adapter.DirectoryDicomSource):
        def read(self, member):
            raw = super().read(member)
            self.full_read_calls += 1
            self.full_read_bytes += len(raw)
            return raw
    kwargs = {"header_only": True} if enabled else {}
    source = Observed(directory, **kwargs)
    source.full_read_calls, source.full_read_bytes = 0, 0
    start = time.perf_counter()
    with source:
        volume, mask, receipt = adapter.prepare_study(uid, records, source, spec, split="train")
    elapsed = time.perf_counter() - start
    stats = {"seconds": elapsed, "full_read_calls": source.full_read_calls,
             "full_read_bytes": source.full_read_bytes, "header": getattr(source, "header_io", {})}
    stats["total_logical_bytes_read"] = source.full_read_bytes + stats["header"].get("logical_bytes_read", 0)
    return volume, mask, receipt, stats


def ordering_proof(adapter, directory, uid, records, spec):
    """Record actual adapter order and original geometry, outside timed rendering.

    No rounding/tolerance changes the render or ordering computation. This proves
    exact operational order even when BLAS contracts a derived float dot product
    differently. Raw DICOM geometry strings and spacing remain exact gates.
    """
    class Observed(adapter.DirectoryDicomSource):
        def read_header(self, member):
            header = super().read_header(member)
            def raw(tag):
                value = getattr(header, tag, None)
                if value is None:
                    return None
                if isinstance(value, (str, int, float)):
                    return str(value)
                return [str(v) for v in value]
            self.geometry[member.path] = {tag: raw(tag) for tag in
                ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing", "InstanceNumber")}
            return header
    result = {"uid": uid, "slots": []}
    with Observed(directory, header_only=True) as source:
        used = set()
        for slot_index, (plane, fluid, _) in enumerate(spec["slots"]):
            row = adapter.pick_series(records, plane, fluid, used)
            if row is None:
                continue
            series = row["SeriesInstanceUID"]; used.add(series)
            source.geometry = {}
            sr = {"instance_order_fallbacks": 0}
            ordered, median = adapter._ordered_members(source, source.members("train", uid, series), sr)
            def exact_hash(value):
                return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            result["slots"].append({"slot": slot_index, "series": series,
                "ordered_sources_sha256": sr["ordered_sources_sha256"],
                "ordered_path_spacing_sha256": exact_hash([[member.path, spacing] for member, spacing in ordered]),
                "raw_geometry_sha256": exact_hash([[name, source.geometry[name]] for name in sorted(source.geometry)]),
                "files": len(ordered),
                "median_spacing": median, "instance_order_fallbacks": sr["instance_order_fallbacks"]})
    return result


def operational_receipt_equal(reference_receipt, actual_receipt, reference_order, actual_order):
    """v2 exact contract: only derived position-float digest may differ.

    Each proof must bind its original digest to the corresponding full receipt.
    Every raw geometry value, actual file ordering, spacing and remaining receipt
    field is exact. Missing/extra fields, changed selection or pixels fail.
    """
    if reference_order.get("uid") != reference_receipt.get("uid") or actual_order.get("uid") != actual_receipt.get("uid"):
        return False
    cleaned = []
    for receipt, proof in ((reference_receipt, reference_order), (actual_receipt, actual_order)):
        receipt, proof = copy.deepcopy(receipt), copy.deepcopy(proof)
        by_slot = {s["slot"]: s for s in receipt["slots"] if "ordered_sources_sha256" in s}
        if set(by_slot) != {s["slot"] for s in proof["slots"]} or len(by_slot) != len(proof["slots"]):
            return False
        for slot in proof["slots"]:
            original = by_slot[slot["slot"]]
            if slot.get("series") != original.get("series") or slot.get("ordered_sources_sha256") != original.get("ordered_sources_sha256"):
                return False
            del slot["ordered_sources_sha256"]
            del original["ordered_sources_sha256"]
        cleaned.append((receipt, proof))
    return cleaned[0] == cleaned[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", required=True)
    p.add_argument("--series-csv", required=True)
    p.add_argument("--baseline-adapter", required=True)
    p.add_argument("--candidate-adapter", default=str(ROOT / "training/src/raptor_dicom.py"))
    p.add_argument("--reference-receipt")
    p.add_argument("--ordering-reference", help="explicit v2 order/geometry contract; default retains v1 exact receipt gate")
    p.add_argument("--archive")
    p.add_argument("--index")
    p.add_argument("--dicom-root")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--contract", default="raptor64_336_c140_s02")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists():
        raise FileExistsError("choose a new immutable output directory")
    if bool(a.dicom_root) == bool(a.archive and a.index):
        p.error("choose --dicom-root or both --archive and --index")
    selection = read_json(a.selection)
    if not 1 <= a.count <= len(selection["ids"]):
        p.error("count must fit the preregistered cohort")
    ids = selection["ids"][:a.count]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cohort UIDs")
    series_hash = digest(a.series_csv)
    if selection.get("train_series_sha256") and series_hash != selection["train_series_sha256"]:
        raise ValueError("series identity changed")
    with Path(a.series_csv).open(encoding="utf-8-sig", newline="") as f:
        records = [{k: row[k] for k in ("StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive")}
                   for row in csv.DictReader(f) if row["StudyInstanceUID"] in set(ids)]
    by_uid = {uid: [r for r in records if r["StudyInstanceUID"] == uid] for uid in ids}
    sys.path.insert(0, str(ROOT / "training/src"))
    baseline = passive_adapter(a.baseline_adapter, "header_benchmark_baseline")
    candidate = passive_adapter(a.candidate_adapter, "header_benchmark_candidate")
    spec = baseline._contract_spec(a.contract)
    if candidate._contract_spec(a.contract) != spec:
        raise ValueError("adapter render contract differs")
    reference = {}
    order_reference = {}
    if a.reference_receipt:
        reference = {r["uid"]: r for r in read_json(a.reference_receipt)["study_receipts"]}
        if not set(ids).issubset(reference):
            raise ValueError("reference receipt omits requested cohort UIDs")
    if a.ordering_reference:
        if not a.reference_receipt:
            raise ValueError("v2 ordering requires the original receipt")
        order_data = read_json(a.ordering_reference)
        if order_data.get("version") != "directory-order-contract-v2" or order_data.get("contract") != a.contract:
            raise ValueError("unknown ordering comparison contract")
        order_reference = {r["uid"]: r for r in order_data["studies"]}
        if not set(ids).issubset(order_reference):
            raise ValueError("ordering reference omits requested cohort UIDs")
    out.mkdir(parents=True)
    directory = Path(a.dicom_root).resolve() if a.dicom_root else out / "dicom"
    result = {"schema": 1, "version": "dicom-header-benchmark-v1", "status": "RUNNING",
              "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ids": ids,
              "input_contract": spec, "contract": a.contract, "studies": [],
              "source_sha256": {"baseline": digest(a.baseline_adapter), "candidate": digest(a.candidate_adapter),
                                "selection": digest(a.selection), "series": series_hash, "tool": digest(__file__)},
              "scope": "fixed-prefix cohort CPU engineering check, no labels/predictions; warm cache, no remote speed claim",
              "byte_scope": "read API bytes including rereads; excludes filesystem read-ahead/cache/block behavior",
              "staging": {"seconds": 0., "files": 0, "bytes": 0}, "pass_order": "alternate baseline-first and candidate-first"}
    if a.ordering_reference:
        result["version"] = "dicom-header-benchmark-v2"
        result["comparison_contract"] = "directory-order-contract-v2: exact raw geometry, actual ordered paths/spacing and all other receipt fields; derived float-position SHA retained as observation"
        result["source_sha256"]["ordering_reference"] = digest(a.ordering_reference)
    if a.reference_receipt:
        result["source_sha256"]["reference_receipt"] = digest(a.reference_receipt)
    dump(out / "result.json", result)
    try:
        if not a.dicom_root:
            stage_start = time.perf_counter()
            with baseline.IndexedZipDicomSource(a.archive, a.index) as source:
                for uid in ids:
                    used = set()
                    for plane, fluid, _ in spec["slots"]:
                        row = baseline.pick_series(by_uid[uid], plane, fluid, used)
                        if row is None:
                            continue
                        series = row["SeriesInstanceUID"]; used.add(series)
                        for member in source.members("train", uid, series):
                            filename = Path(member.path).name
                            target = directory / baseline._component(uid) / baseline._component(series) / filename
                            if not target.resolve().is_relative_to(directory.resolve()):
                                raise ValueError("staged DICOM escapes output directory")
                            target.parent.mkdir(parents=True, exist_ok=True)
                            raw = source.read(member)
                            with target.open("xb") as handle:
                                handle.write(raw)
                            result["staging"]["files"] += 1; result["staging"]["bytes"] += len(raw)
            result["staging"]["seconds"] = time.perf_counter() - stage_start
            dump(out / "result.json", result)
        import numpy as np
        import cv2
        cv2.setNumThreads(1)
        for index, uid in enumerate(ids):
            values = {}
            names = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            for name in names:
                values[name] = render(baseline if name == "baseline" else candidate, directory,
                                      uid, by_uid[uid], spec, name == "candidate")
            av, am, ar, astats = values["baseline"]
            bv, bm, br, bstats = values["candidate"]
            checks = {"volume_exact": bool(np.array_equal(av, bv)), "mask_exact": bool(np.array_equal(am, bm)),
                      "receipt_exact": ar == br}
            if uid in reference:
                checks["remote_volume_hash_exact"] = reference[uid]["volume_sha256"] == br["volume_sha256"]
                checks["remote_mask_hash_exact"] = reference[uid]["mask_sha256"] == br["mask_sha256"]
                if a.ordering_reference:
                    order = ordering_proof(candidate, directory, uid, by_uid[uid], spec)
                    checks["remote_operational_receipt_exact"] = operational_receipt_equal(reference[uid], br, order_reference[uid], order)
                else:
                    checks["remote_receipt_exact"] = reference[uid] == br
            entry = {"uid": uid, "checks": checks, "baseline": astats, "candidate": bstats,
                     "receipt": br}
            if a.ordering_reference:
                entry["ordering_proof"] = order
                entry["observations"] = {"remote_receipt_exact": reference[uid] == br,
                    "derived_position_hash_differences": [{"slot": s["slot"],
                        "reference": next(r["ordered_sources_sha256"] for r in order_reference[uid]["slots"] if r["slot"] == s["slot"]),
                        "actual": s["ordered_sources_sha256"]} for s in order["slots"]
                        if s["ordered_sources_sha256"] != next(r["ordered_sources_sha256"] for r in order_reference[uid]["slots"] if r["slot"] == s["slot"])],
                    "ordering_proof_scope": "additional validation outside render timing and byte counters"}
            result["studies"].append(entry)
            dump(out / "result.json", result)
            print(json.dumps({"completed": index + 1, "total": len(ids), "checks": checks}), flush=True)
            if not all(checks.values()):
                raise RuntimeError(f"parity failed for {uid}")
        result["summary"] = {}
        for name in ("baseline", "candidate"):
            values = [r[name] for r in result["studies"]]
            result["summary"][name] = {"seconds": sum(v["seconds"] for v in values),
                                      "median_seconds": statistics.median(v["seconds"] for v in values),
                                      "total_logical_bytes_read": sum(v["total_logical_bytes_read"] for v in values)}
        before, after = result["summary"]["baseline"], result["summary"]["candidate"]
        result["summary"]["logical_byte_reduction_fraction"] = 1 - after["total_logical_bytes_read"] / before["total_logical_bytes_read"]
        result["summary"]["warm_cache_time_reduction_fraction"] = 1 - after["seconds"] / before["seconds"]
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAILED"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        dump(out / "result.json", result)
    return result


if __name__ == "__main__":
    main()
