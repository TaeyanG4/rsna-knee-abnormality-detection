"""Decode only the prospectively frozen H-C-15 cache sample on one CPU thread.

No whole-corpus mode is implemented. Requires a fresh output directory and retains
all new arrays/receipts. Reads the existing ZIP by frozen member metadata; never
extracts it or modifies source/cache files. Budget: <=12 series, <=768 frames.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

class IndexedZip(zipfile.ZipFile):
    def _RealGetContents(self):
        self.start_dir = os.fstat(self.fp.fileno()).st_size

def member_info(record):
    z = zipfile.ZipInfo(record["path"])
    allowed = {"compress_type", "comment", "extra", "create_system", "create_version", "extract_version",
               "reserved", "flag_bits", "volume", "internal_attr", "external_attr", "header_offset", "CRC",
               "compress_size", "file_size", "date_time", "_raw_time", "_end_offset", "orig_filename", "filename"}
    for key, value in record["zip_info"].items():
        if key not in allowed:
            raise ValueError(f"Unrecognized frozen ZipInfo field: {key}")
        setattr(z, key, tuple(value) if key == "date_time" else value)
    if z.filename != record["path"]:
        raise ValueError("Member identity mismatch")
    return z

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-seconds", type=int, default=180)
    a = ap.parse_args()
    if not 1 <= a.max_seconds <= 600:
        raise ValueError("Sample deadline must be1..600seconds")
    out = a.out.resolve()
    if not out.is_relative_to(ROOT / "artifacts/experiments") or out.exists():
        raise ValueError("Require a fresh output under artifacts/experiments")
    manifest = json.loads(a.manifest.read_text())
    sample_binding = next(x for x in manifest["generated"] if x["path"].endswith("/sample.json"))
    sample_path = ROOT / sample_binding["path"]
    if sha(sample_path) != sample_binding["sha256"]:
        raise ValueError("Sample drift")
    sample = json.loads(sample_path.read_text())
    if len(sample["series"]) > 12 or sample["decoded_frames_limit"] > 768:
        raise ValueError("Sample exceeds registered budget")
    # Imports are local reviewed adapters, never downloaded public Python.
    import cv2
    import numpy as np
    import pydicom
    import torch
    from training.adapters.goodpjw_reader import DecodedSlice, preprocess_decoded_series
    cv2.setNumThreads(1); torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    archive = Path(manifest["archive"]["path"])
    if archive.stat().st_size != manifest["archive"]["bytes"]:
        raise ValueError("Source archive stat changed")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); decoded = 0; rows = []; errors = []
    try:
        with IndexedZip(archive) as z:
            for entry in sample["series"]:
                if time.monotonic()-start >= a.max_seconds:
                    raise TimeoutError("Sample elapsed deadline reached")
                t0 = time.monotonic(); slices = []
                for member in entry["members"]:
                    if time.monotonic()-start >= a.max_seconds:
                        raise TimeoutError("Sample elapsed deadline reached before next decode")
                    data = z.read(member_info(member))
                    ds = pydicom.dcmread(io.BytesIO(data))
                    frames = int(ds.get("NumberOfFrames", 1) or 1)
                    if frames != 1:
                        raise ValueError("Enhanced/multiframe DICOM requires a separately reviewed per-frame geometry reader")
                    if decoded + frames > sample["decoded_frames_limit"]:
                        raise RuntimeError("Frame bound reached; no automatic replacement")
                    pixels = ds.pixel_array
                    if frames == 1 and pixels.ndim == 2:
                        arrays = [pixels]
                    elif pixels.ndim == 3 and len(pixels) == frames:
                        arrays = list(pixels)
                    else:
                        raise ValueError("Unsupported pixel/frame layout")
                    decoded += len(arrays)
                    for frame, arr in enumerate(arrays):
                        slices.append(DecodedSlice(
                            pixels=arr,
                            pixel_spacing=list(map(float, ds.PixelSpacing)) if "PixelSpacing" in ds else None,
                            image_orientation=list(map(float, ds.ImageOrientationPatient)) if "ImageOrientationPatient" in ds else None,
                            image_position=list(map(float, ds.ImagePositionPatient)) if "ImagePositionPatient" in ds else None,
                            instance_number=float(ds.get("InstanceNumber", 0) or 0)+frame*1e-3,
                            source_id=member["path"]+f"#{frame}",
                            photometric=str(ds.get("PhotometricInterpretation", "MONOCHROME2")),
                            rescale_slope=float(ds.RescaleSlope) if "RescaleSlope" in ds else None,
                            rescale_intercept=float(ds.RescaleIntercept) if "RescaleIntercept" in ds else None))
                volume, info = preprocess_decoded_series(slices, plane_name=entry["csv_metadata"]["Anatomical_Plane"])
                assert volume.dtype == np.uint8 and 3 <= len(volume) <= 64 and volume.shape[1:] == (384,384)
                dest = out / (entry["series"] + ".npy")
                np.save(dest, volume, allow_pickle=False)
                rows.append({"study": entry["study"], "series": entry["series"], "path": dest.relative_to(ROOT).as_posix(),
                             "sha256": sha(dest), "shape": list(volume.shape), "decoded": len(slices), "metadata": info,
                             "seconds": time.monotonic()-t0, "bytes": dest.stat().st_size})
    except Exception as exc:
        errors.append({"type": type(exc).__name__, "message": str(exc)})
    elapsed = time.monotonic()-start
    result = {"schema": "hc15-cache-sample-v1", "status": "PASS_SAMPLE_ONLY" if not errors else "INCOMPLETE",
              "manifest_sha256": sha(a.manifest), "sample_sha256": sha(sample_path),
              "adapter_sha256": sha(ROOT / "training/adapters/goodpjw_reader.py"), "runner_sha256": sha(__file__),
              "decoded_frames": decoded, "completed_series": len(rows), "seconds": elapsed, "rows": rows, "errors": errors,
              "cuda_initialized": torch.cuda.is_initialized(),
              "full_cache_projection_seconds_by_series": elapsed / len(rows) * manifest["cache"]["series"] if rows else None,
              "projection_limitation": "Two hash-selected training studies only. This is a rough throughput projection, not full runtime/coverage evidence; source frames/compression vary.",
              "full_cache_committed": False}
    if result["cuda_initialized"]:
        raise RuntimeError("Unexpected CUDA initialization")
    with (out / "result.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","decoded_frames","completed_series","seconds","full_cache_projection_seconds_by_series","errors")}))
    if errors:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
