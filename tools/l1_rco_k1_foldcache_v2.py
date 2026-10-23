"""H224 parity v2.0.0: explicit META_EPS exception, maximum absolute error 1e-12.

Changelog: additive to immutable v1; export authenticated META only, preserve
exact mode, require exact pixels/blocks/rows and all 32 reference studies.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
# v1 is a CPU CLI that changes CUDA visibility on import. Preserve caller state.
_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
from tools import l1_rco_k1_foldcache_v1 as old
if _cuda is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda
f = old.f
MAX_TOL = 1e-12
META = ROOT / ".local/scratch/astra-b2/h224-meta-eps-v1/local32-meta.npz"


def identity(reference):
    ref = f.read(reference)
    reg = f.validate(f.read(f.REG))
    ids = old.selected_ids(reg)
    f.need(len(ids) == len(set(ids)) == 32 and ref["ids"] == ids
           and ref["contract_id"] == reg["contract_id"]
           and ref["renderer_pins"] == reg["pins"]
           and set(ref["studies"]) == set(ids), "parity reference identity")
    return ref, reg, ids


def meta_arrays(cache, ids):
    import numpy as np
    index = f.read(Path(cache) / "index.json")
    meta = np.load(Path(cache) / "meta.f32.npy", mmap_mode="r", allow_pickle=False)
    return {uid: np.array(meta[offset:offset+n], copy=True)
            for uid in ids for offset, n, _ in [index["done"][uid]]}


def checked_arrays(arrays, ref, ids):
    import numpy as np
    f.need(set(arrays) == set(ids), "META roster mismatch")
    for uid in ids:
        a = arrays[uid]
        f.need(a.dtype == np.dtype("float32") and a.shape == (ref["studies"][uid]["rows"], 12)
               and np.isfinite(a).all(), "META shape/dtype/nonfinite")


def export_meta(cache, reference=old.HASHES, output=META):
    import numpy as np
    ref, reg, ids = identity(reference)
    from tools.l1_rco_k1_v1 import complete_cache
    complete_cache(dict(train=reg["train"], holdout=reg["holdout"],
                        fullcache_contract_id=reg["contract_id"]), cache)
    f.need(f.sha(Path(cache)/"receipt.json") == ref["source_receipt_sha256"],
           "META export source receipt changed")
    arrays = meta_arrays(cache, ids)
    checked_arrays(arrays, ref, ids)
    f.need(all(old.array_hash(arrays[u]) == ref["studies"][u]["meta"] for u in ids),
           "META reference hash mismatch")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to replace existing evidence. Re-runs verify it instead.
    if output.exists():
        load_reference(output, ref, ids)
    else:
        with output.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    digest = f.sha(output)
    sidecar = output.with_suffix(".npz.sha256")
    text = digest + "  " + output.name + "\n"
    if sidecar.exists():
        f.need(sidecar.read_text(encoding="ascii") == text, "META sidecar changed")
    else:
        sidecar.write_text(text, encoding="ascii")
    return dict(status="PASS", studies=32, sha256=digest, bytes=output.stat().st_size)


def load_reference(path, ref, ids):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        f.need(len(archive.files) == 32 and set(archive.files) == set(ids),
               "META reference roster mismatch")
        arrays = {u: archive[u] for u in ids}
    checked_arrays(arrays, ref, ids)
    f.need(all(old.array_hash(arrays[u]) == ref["studies"][u]["meta"] for u in ids),
           "META reference hash mismatch")
    return arrays


def compare_meta(actual, reference, ref, ids, tolerance):
    import numpy as np
    f.need(math.isfinite(tolerance) and 0 < tolerance <= MAX_TOL,
           "META tolerance must be finite, positive and <= 1e-12")
    checked_arrays(actual, ref, ids)
    checked_arrays(reference, ref, ids)
    records = {}
    for uid in ids:
        # Compare in float64; do not round/subtract in float32 or use relative tolerance.
        diff = np.abs(actual[uid].astype(np.float64) - reference[uid].astype(np.float64))
        maximum = float(diff.max())
        f.need(maximum <= tolerance, "META absolute difference exceeds tolerance; HOLD")
        records[uid] = dict(changed_cells=int(np.count_nonzero(diff)),
                            changed_columns=np.flatnonzero(np.any(diff != 0, axis=0)).tolist(),
                            max_abs_diff=maximum, actual_meta_sha256=old.array_hash(actual[uid]),
                            reference_meta_sha256=old.array_hash(reference[uid]))
    return dict(per_study=records,
                changed_studies=sum(r["changed_cells"] > 0 for r in records.values()),
                changed_cells=sum(r["changed_cells"] for r in records.values()),
                max_abs_diff=max(r["max_abs_diff"] for r in records.values()))


def assess(cache, reference=old.HASHES, meta_reference=None, tolerance=None):
    ref, reg, ids = identity(reference)
    actual = old.study_hashes(cache, ids)
    result = dict(status="PASS", studies=32, reference_sha256=f.sha(reference),
                  cache_receipt_sha256=f.sha(Path(cache)/"receipt.json"),
                  contract_id=reg["contract_id"], smoke=False)
    if tolerance is None:
        f.need(actual == ref["studies"], "cache payload/meta/blocks parity mismatch; HOLD")
        return dict(result, mode="EXACT")
    f.need(meta_reference is not None, "explicit --meta-reference required")
    f.need(all(actual[u][key] == ref["studies"][u][key]
               for u in ids for key in ("px", "blocks", "rows")),
           "px/blocks/rows must be EXACT; HOLD")
    arrays = meta_arrays(cache, ids)
    checked_arrays(arrays, ref, ids)
    f.need(all(old.array_hash(arrays[u]) == actual[u]["meta"] for u in ids),
           "cache META changed during parity")
    reference_arrays = load_reference(meta_reference, ref, ids)
    stats = compare_meta(arrays, reference_arrays, ref, ids, tolerance)
    return dict(result, mode="META_EPS", tolerance=tolerance, **stats,
                exact_fields=["px", "blocks", "rows"],
                meta_reference_sha256=f.sha(meta_reference),
                actual_studies_sha256=f.digest(actual), schema="h224-meta-eps/v1")


def parity(cache, reference=old.HASHES, meta_reference=None, tolerance=None):
    result = assess(cache, reference, meta_reference, tolerance)
    output = Path(cache)/"H224-PARITY.json"
    if output.exists():
        f.need(f.read(output) == result, "existing parity receipt differs; preserve it, use a reviewed versioned cache stage")
    else:
        f.write(output, result)
    return result


def validate_receipt(cache, reference=old.HASHES, meta_reference=META):
    proof = f.read(Path(cache)/"H224-PARITY.json")
    f.need(proof["status"] == "PASS" and proof["studies"] == 32 and not proof["smoke"]
           and proof["reference_sha256"] == f.sha(reference)
           and proof["cache_receipt_sha256"] == f.sha(Path(cache)/"receipt.json"),
           "Linux cache parity evidence mismatch")
    mode = proof.get("mode", "EXACT")
    if mode == "EXACT":
        # Original receipts have no mode. Keep original exact acceptance contract.
        f.need(not any(k in proof for k in ("tolerance", "per_study",
                                           "meta_reference_sha256", "schema")),
               "META_EPS receipt cannot downgrade to EXACT")
        return proof
    f.need(mode == "META_EPS", "unknown parity mode")
    f.need(proof == assess(cache, reference, meta_reference, proof["tolerance"]),
           "META_EPS receipt evidence mismatch")
    return proof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["export-meta", "parity", "validate-receipt"])
    ap.add_argument("--cache", type=Path, default=old.CACHE)
    ap.add_argument("--reference", type=Path, default=old.HASHES)
    ap.add_argument("--meta-reference", type=Path, default=META)
    ap.add_argument("--output", type=Path, default=META)
    ap.add_argument("--meta-abs-tol", type=float)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.action == "export-meta":
        result = export_meta(a.cache, a.reference, a.output)
    elif a.action == "validate-receipt":
        result = validate_receipt(a.cache, a.reference, a.meta_reference)
    else:
        result = parity(a.cache, a.reference, a.meta_reference, a.meta_abs_tol)
    print(json.dumps({k: v for k, v in result.items() if k in
                     ("status", "studies", "sha256", "bytes", "mode", "tolerance",
                      "changed_studies", "changed_cells", "max_abs_diff")}, allow_nan=False))


if __name__ == "__main__":
    main()
