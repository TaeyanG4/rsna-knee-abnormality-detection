"""EXP175 v1.2.0: local CPU image-domain audit. See the registered PROTOCOL.md.

Import is passive. No model, torch, raw DICOM, report or network API is used.
Run --help for sampling defaults; every run needs a fresh output directory.

Changelog v1.2.0: remove internal and subprocess wall limits; legacy timing
options are inert, durations remain measured, and missing-data/failure gates stay.
Changelog v1.1.1: never promote legacy rendering scalars to native spacing;
record phase wall timings, including skipped/failed/partial work; repair the
versioned execution handoff and delivery checksum regression coverage.
Changelog v1.1.0: unknown fluid flags stay unassessed; contrast requires
nonclipped support; bind reused metrics to original cache provenance; require
a version/hash-bound PASS verification note; report paired valid draws.
Changelog v1.0.1: use float64 for unbiased skewness on NumPy 2.x; keep the
reference histogram maximum in its finite bin, separate from the upper tail.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

# Set before numpy/scipy import when invoked as a script; never change a caller
# environment when imported by pytest or another tool.
if __name__ == "__main__":
    for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from scipy import ndimage, stats

VERSION = "1.2.0"
ROOT = Path(__file__).resolve().parents[1]
BASE = Path("artifacts/experiments/EXP175_IMAGE_DOMAIN_AUDIT")
CONTRACT = "raptor64_336_c140_s02"
SLOTS = (("Sagittal_fluid1", 0, 18, 1), ("Sagittal_fluid0", 18, 32, 0),
         ("Coronal_fluid1", 32, 44, 1), ("Coronal_fluid0", 44, 52, 0),
         ("Axial", 52, 64, -1))
REGISTERED = ("MRNET", "KNEEMRI", "FASTMRI_SC", "FASTMRI_PLUS", "OAI", "SKM_TEA", "OAIZIB_CM")
ALIASES = {"FASTMRI_PLUS": "FASTMRI_SC"}
QUANTILES = (1, 5, 25, 50, 75, 95, 99)
IMAGE_METRICS = ("noise_hh_flat", "noise_hh_low", "air_sd", "air_raw_sd", "air_mean",
                 "air_fraction", "air_zero_fraction", "air_positive_fraction", "snr_proxy",
                 "cnr_proxy", "laplacian_var", "hf_energy_ratio", "foreground_fraction",
                 "clip0", "clip255", "iqr", "skewness", "central_peripheral_ratio",
                 "fluid_fat_ratio", "edge_contact_fraction", "slice_difference") + tuple(
                     f"p{q:02d}" for q in QUANTILES)
GEOMETRY_METRICS = ("pixel_spacing_row", "pixel_spacing_col", "thickness", "slice_spacing",
                    "TE", "TR", "native_slices", "unique_native_slices", "duplicate_fraction",
                    "index_span_fraction", "physical_span_mm", "fill_fraction")
METRICS = IMAGE_METRICS + GEOMETRY_METRICS
HEADER_ALIASES = {
    "vendor": ("vendor", "Manufacturer", "manufacturer"),
    "field_strength": ("field_strength", "MagneticFieldStrength"),
    "series_description": ("series_description", "SeriesDescription"),
    "pixel_spacing": ("pixel_spacing", "PixelSpacing"),
    "thickness": ("thickness", "SliceThickness"),
    "slice_spacing": ("slice_spacing", "SpacingBetweenSlices"),
    "TE": ("TE", "EchoTime"), "TR": ("TR", "RepetitionTime"),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


class PhaseTimings:
    """Persist wall durations at phase boundaries and extraction checkpoints.

    RUNNING/INTERRUPTED durations are lower bounds from the last checkpoint;
    skipped phases have null durations, never fabricated zero measurements.
    """
    NAMES = ("extraction", "metrics_load", "summary", "montage",
             "input_validation", "behavior", "report")

    def __init__(self, manifest, out, started):
        self.manifest, self.out, self.started = manifest, out, started
        self.active, self.phase_started = None, None
        manifest["phase_timings"] = {
            name: dict(status="NOT_STARTED", elapsed_seconds=None, duration_complete=False)
            for name in self.NAMES}

    def checkpoint(self):
        now = time.monotonic()
        if self.active is not None:
            self.manifest["phase_timings"][self.active]["elapsed_seconds"] = now - self.phase_started
        self.manifest["elapsed_seconds"] = now - self.started
        dump(self.out / "manifest.json", self.manifest)

    def start(self, name):
        if self.active is not None:
            raise RuntimeError("finish the active phase before starting another")
        self.active, self.phase_started = name, time.monotonic()
        self.manifest["phase_timings"][name].update(
            status="RUNNING", elapsed_seconds=0., duration_complete=False)
        self.checkpoint()

    def finish(self, status="COMPLETE"):
        if self.active is not None:
            self.manifest["phase_timings"][self.active].update(
                status=status, elapsed_seconds=time.monotonic() - self.phase_started,
                duration_complete=True)
            self.active = None
        self.checkpoint()

    def skip(self, name, reason):
        self.manifest["phase_timings"][name].update(
            status=reason, elapsed_seconds=None, duration_complete=False)


def contained(base, name):
    """No absolute paths, traversal or symlink escape from trusted input roots."""
    base = Path(base).resolve()
    p = (base / str(name)).resolve()
    if not p.is_relative_to(base):
        raise ValueError("path escapes input directory")
    return p


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def boolean(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def fluid_slot_status(slot, actual):
    """Only binary fluid flags are known; -1 is an unknown sentinel."""
    actual = number(actual)
    if actual not in (0, 1):
        return dict(fluid_slot_mismatch=False, fluid_slot_check="UNASSESSED")
    if slot < 0:
        return dict(fluid_slot_mismatch=False, fluid_slot_check="ANY_CONTRAST_SLOT")
    mismatch = slot != actual
    return dict(fluid_slot_mismatch=mismatch, fluid_slot_check="MISMATCH" if mismatch else "MATCH")


def source_hashes():
    return {p.name: sha256(p) for p in (Path(__file__), ROOT / "tools/exp175_behavior.py")}


def verification_provenance(note, *, version=VERSION, hashes=None):
    """A separate review must PASS these exact sources, not merely this filename."""
    if not note:
        raise ValueError("--verification-note is required")
    path = Path(note).resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = source_hashes() if hashes is None else hashes
    if (receipt.get("verdict") != "PASS" or receipt.get("verified_tool_version") != version
            or receipt.get("source_hashes") != expected):
        raise ValueError("verification note must PASS the exact tool version and source hashes")
    return dict(path=str(path), sha256=sha256(path), verified_tool_version=version,
                source_hashes=expected)


def metrics_provenance(root, path, manifest_path, manifest):
    """Receipt for the metric bytes and the inputs at the time they were measured."""
    return dict(path=str(path.relative_to(root)), sha256=sha256(path),
                manifest_path=str(manifest_path.relative_to(root)), manifest_sha256=sha256(manifest_path),
                tool_version=manifest["version"], verification=manifest["verification"],
                source_hashes=manifest["source_hashes"], protocol_sha256=manifest["protocol_sha256"],
                input_hashes=manifest["input_hashes"])


def load_previous_metrics(root, previous):
    """Fail closed on unbound/failed legacy metrics; never substitute today's cache."""
    root, previous = Path(root).resolve(), Path(previous).resolve()
    if not previous.is_relative_to((root / BASE).resolve()) or previous.name != "metrics.csv":
        raise ValueError("--behavior-only must point to a prior EXP175 metrics.csv")
    manifest_path = previous.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") not in ("COMPLETE", "COMPLETE_WITH_GAPS")
            or (previous.parent / "FAILURE.json").exists()):
        raise ValueError("original metrics manifest is not a completed valid audit")
    if manifest.get("metrics_sha256") != sha256(previous):
        raise ValueError("original metrics checksum missing or differs; rerun image audit")
    try:
        verification = manifest["verification"]
        checked = verification_provenance(verification["path"], version=manifest["version"],
                                          hashes=manifest["source_hashes"])
        if checked != verification:
            raise ValueError("original verification note changed")
        hashes = manifest["input_hashes"]
        if not isinstance(hashes, dict) or not hashes or not manifest["protocol_sha256"]:
            raise ValueError("original input provenance missing")
    except (KeyError, TypeError) as exc:
        raise ValueError("original metrics provenance missing; rerun image audit") from exc
    frame = pd.read_csv(previous, dtype={"uid": str, "source": str, "split": str, "group": str})
    if len(frame) != manifest.get("metric_rows"):
        raise ValueError("original metric row count differs")
    required = {"source", "split", "uid", "domain", "group"}
    if not required.issubset(frame.columns):
        raise ValueError("original metrics schema differs")
    normalized = {k.replace("\\", "/"): v for k, v in hashes.items()}
    for source in frame.loc[frame.domain == "production", "source"].unique():
        cache = "RAPTOR64_336_C140_S02" if source == "COMPETITION" else f"EXT_{source}_RAPTOR64_336_C140"
        bound = normalized.get(f"storage/cache/derived/{cache}/manifest.json")
        if not isinstance(bound, str) or len(bound) != 64 or any(c not in "0123456789abcdef" for c in bound):
            raise ValueError("original production cache hash missing or malformed: " + str(source))
    return frame, metrics_provenance(root, previous, manifest_path, manifest)


def stable_sample(ids, n, seed, salt):
    """Stable across processes/platforms and independent of labels and row order."""
    ids = list(map(str, ids))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate study IDs")
    return sorted(ids, key=lambda u: (hashlib.sha256(f"{seed}|{salt}|{u}".encode()).digest(), u))[:n]


def slice_indices(valid, n=3):
    valid = np.asarray(valid, dtype=int)
    if not len(valid):
        return np.empty(0, dtype=int)
    return valid[np.unique(np.linspace(0, len(valid) - 1, min(n, len(valid))).round().astype(int))]


def mad_sigma(values):
    values = np.asarray(values)
    return float(np.median(np.abs(values - np.median(values))) / .6744897501960817)


def tissue_ratio(image, fluid=None, fat=None):
    """Only validated, disjoint, nontrivial tissue masks license tissue names."""
    if fluid is None or fat is None:
        return np.nan
    fluid, fat = np.asarray(fluid), np.asarray(fat)
    if fluid.shape != image.shape or fat.shape != image.shape:
        raise ValueError("ROI/image shapes differ")
    if fluid.dtype != bool or fat.dtype != bool or np.any(fluid & fat):
        raise ValueError("ROIs must be disjoint boolean masks")
    if min(fluid.sum(), fat.sum()) < 16:
        return np.nan
    # Clipped tissue cannot license a contrast ratio even when other tissue
    # gives the image ample signal. Keep only measured, nonclipped support.
    nonclipped = (image > 1 / 255) & (image < 254 / 255)
    fluid, fat = fluid & nonclipped, fat & nonclipped
    if min(fluid.sum(), fat.sum()) < 16:
        return np.nan
    denominator = float(np.median(image[fat]))
    return float(np.median(image[fluid]) / denominator) if denominator > 0 else np.nan


def image_metrics(image, fluid=None, fat=None):
    """One 2D image already on a declared [0,1] display scale. No rewindowing."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 8 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("expected finite 2D [0,1] image with sides >=8")
    result = {key: np.nan for key in IMAGE_METRICS}
    result.update({f"p{q:02d}": float(v) for q, v in zip(QUANTILES, np.percentile(x, QUANTILES))})
    fg = x > 16 / 255
    signal_mask = fg & (x < 254 / 255)
    result.update(clip0=float(np.mean(x == 0)), clip255=float(np.mean(x == 1)),
                  foreground_fraction=float(fg.mean()), iqr=result["p75"] - result["p25"],
                  noise_support=0, low_noise_support=0, air_support=0,
                  nonclipped_signal_support=int(signal_mask.sum()),
                  valid_signal=bool(signal_mask.sum() >= 64 and np.ptp(x) > 0),
                  fs_consistency="UNASSESSED_NO_VALIDATED_TISSUE_ROIS")
    if x.std() > 1e-7:
        # SciPy's bias correction can produce float64 values for an in-place
        # float32 result; NumPy 2.x rejects that safe cast. Promote only the
        # moment input, preserving float32 image filtering and support masks.
        result["skewness"] = float(stats.skew(x.ravel().astype(np.float64), bias=False))
    edge = np.concatenate((fg[0], fg[-1], fg[1:-1, 0], fg[1:-1, -1]))
    result["edge_contact_fraction"] = float(edge.mean())
    low = x <= 16 / 255
    seed = np.zeros(x.shape, bool)
    seed[0], seed[-1], seed[:, 0], seed[:, -1] = low[0], low[-1], low[:, 0], low[:, -1]
    air = ndimage.binary_propagation(seed, mask=low)
    result["air_support"] = int(air.sum())
    result["air_fraction"] = float(air.mean())
    if air.sum() >= 64:
        av = x[air]
        result.update(air_mean=float(av.mean()), air_raw_sd=float(av.std()),
                      air_zero_fraction=float(np.mean(av == 0)), air_positive_fraction=float(np.mean(av > 0)))
        if result["air_zero_fraction"] <= .5:
            result["air_sd"] = result["air_raw_sd"]
    if not result["valid_signal"]:
        return result
    smooth = ndimage.gaussian_filter(x, 2., mode="reflect")
    gy, gx = np.gradient(smooth)
    gradient = np.hypot(gx, gy)
    interior = ndimage.binary_erosion((x > 1 / 255) & (x < 254 / 255) & fg, iterations=1)
    if interior.sum() >= 128:
        flat = interior & (gradient <= np.percentile(gradient[interior], 25))
        # Restrict all four HH pixels, not just the top-left point.
        support = flat[:-1, :-1] & flat[1:, :-1] & flat[:-1, 1:] & flat[1:, 1:]
        hh = (x[:-1, :-1] - x[1:, :-1] - x[:-1, 1:] + x[1:, 1:]) / 2
        result["noise_support"] = int(support.sum())
        if support.sum() >= 64:
            result["noise_hh_flat"] = mad_sigma(hh[support])
        ls = support & (smooth[:-1, :-1] < np.median(smooth[interior]))
        result["low_noise_support"] = int(ls.sum())
        if ls.sum() >= 64:
            result["noise_hh_low"] = mad_sigma(hh[ls])
    noise = result["noise_hh_flat"]
    signal = x[signal_mask]
    if np.isfinite(noise) and noise > 1e-7:
        result["snr_proxy"] = float(np.median(signal) / noise)
        result["cnr_proxy"] = float(np.diff(np.percentile(signal, [10, 90]))[0] / noise)
    interior_fg = ndimage.binary_erosion(fg, iterations=2)
    if interior_fg.sum() >= 64:
        result["laplacian_var"] = float(np.var(ndimage.laplace(x)[interior_fg]))
        var = np.var(x[interior_fg])
        if var > 1e-10:
            hp = x - ndimage.gaussian_filter(x, 1., mode="reflect")
            result["hf_energy_ratio"] = float(np.mean(hp[interior_fg] ** 2) / var)
    yy, xx = np.indices(x.shape)
    rr = ((yy - (x.shape[0] - 1) / 2) / (x.shape[0] / 2)) ** 2 + ((xx - (x.shape[1] - 1) / 2) / (x.shape[1] / 2)) ** 2
    central, peripheral = signal_mask & (rr < .25), signal_mask & (rr > .55) & (rr < 1.)
    if min(central.sum(), peripheral.sum()) >= 64:
        denominator = np.percentile(x[peripheral], 90)
        if denominator > 0:
            result["central_peripheral_ratio"] = float(np.percentile(x[central], 90) / denominator)
    result["fluid_fat_ratio"] = tissue_ratio(x, fluid, fat)
    if np.isfinite(result["fluid_fat_ratio"]):
        result["fs_consistency"] = "ROI_RATIO_ONLY_MANUAL_SEQUENCE_REVIEW_REQUIRED"
    return result


def volume_metrics(images, fluid=None, fat=None):
    """Median per-slice metrics; preserve valid-slice counts and support."""
    values = [image_metrics(x, None if fluid is None else fluid[i], None if fat is None else fat[i])
              for i, x in enumerate(images)]
    out = {}
    for key in IMAGE_METRICS + ("noise_support", "low_noise_support", "air_support", "nonclipped_signal_support"):
        v = np.asarray([r[key] for r in values], float)
        finite = np.isfinite(v)
        out[key] = float(np.median(v[finite])) if finite.any() else np.nan
        out[key + "_valid_slices"] = int(finite.sum())
    out["sampled_slices"] = len(values)
    out["valid_signal_slices"] = sum(v["valid_signal"] for v in values)
    out["fs_consistency"] = values[0]["fs_consistency"] if values else "UNASSESSED"
    if len(images) > 1:
        out["slice_difference"] = float(np.median(np.mean(np.abs(np.diff(images, axis=0)), axis=(1, 2))))
        out["slice_difference_valid_slices"] = len(images) - 1
    return out


def geometry(pick, meta, filled, size):
    declared_unknown = boolean(meta.get("spacing_unknown", pick.get("spacing_unknown", False)))
    spacing = meta.get("pixel_spacing", [])
    if isinstance(spacing, str):
        try:
            spacing = json.loads(spacing.replace("\\", ",") if "[" in spacing else "[" + spacing.replace("\\", ",") + "]")
        except (ValueError, TypeError):
            spacing = []
    if not isinstance(spacing, (list, tuple)):
        spacing = []
    # Legacy competition picks retain PixelSpacing[0] only and may substitute
    # 0.5 on missing/read-failed headers. They cannot establish either measured
    # axis, even with spacing_unknown=False. Preserve that rendering scalar
    # separately; only selected-series metadata supplies native axis values.
    ps = [number(spacing[i]) if i < len(spacing) else np.nan for i in range(2)]
    ps = [p if np.isfinite(p) and p > 0 and not declared_unknown else np.nan for p in ps]
    unknown = not all(np.isfinite(ps))
    n = number(pick.get("files", meta.get("n_slices")))
    selected = [int(i) for i in pick.get("slices", [])]
    unique = len(set(selected))
    span = max(selected) - min(selected) if selected else np.nan
    dz = number(meta.get("slice_spacing"))
    dz = abs(dz) if np.isfinite(dz) and dz != 0 and not declared_unknown else np.nan
    return dict(pixel_spacing_row=ps[0], pixel_spacing_col=ps[1], spacing_unknown=unknown,
                pixel_spacing_source="selected_series_metadata" if any(np.isfinite(ps)) else "unavailable",
                spacing_declared_unknown=declared_unknown,
                legacy_rendering_spacing_scalar=number(pick.get("median_spacing")),
                thickness=number(meta.get("thickness")) if not declared_unknown else np.nan,
                slice_spacing=dz, TE=number(meta.get("TE")), TR=number(meta.get("TR")),
                native_slices=n, unique_native_slices=unique if selected else np.nan,
                duplicate_fraction=1 - unique / len(selected) if selected else np.nan,
                index_span_fraction=span / (n - 1) if np.isfinite(n) and n > 1 else np.nan,
                physical_span_mm=span * dz, fill_fraction=filled / size)


def shift_statistics(reference, external):
    a, b = (np.asarray(x, float) for x in (reference, external))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = dict(n_reference=len(a), n_external=len(b), flag="INSUFFICIENT", ks=np.nan,
               outside_5_95=np.nan, iqr_shift=np.nan, histogram_overlap=np.nan)
    if not len(a) or not len(b):
        return out
    qa, qb = np.percentile(a, QUANTILES), np.percentile(b, QUANTILES)
    out.update(reference_quantiles=qa.tolist(), external_quantiles=qb.tolist(),
               outside_5_95=float(np.mean((b < qa[1]) | (b > qa[-2]))),
               ks=float(stats.ks_2samp(a, b, method="asymp").statistic))
    iqr = qa[4] - qa[2]
    if iqr > 1e-12:
        out["iqr_shift"] = float((qb[3] - qa[3]) / iqr)
    # Put exact reference mass in a finite bin even for a constant reference.
    edges = np.unique(np.percentile(a, np.linspace(0, 100, 11)))
    if len(edges) == 1:
        edges = np.array([edges[0], np.nextafter(edges[0], np.inf)], dtype=np.float64)
    else:
        # Appending +inf makes the last finite bin right-open. Include the
        # reference maximum there so a disjoint upper tail has zero overlap.
        edges[-1] = np.nextafter(edges[-1], np.inf)
    edges = np.r_[-np.inf, edges, np.inf]
    ha, hb = np.histogram(a, edges)[0] / len(a), np.histogram(b, edges)[0] / len(b)
    out["histogram_overlap"] = float(np.minimum(ha, hb).sum())
    flags = []
    if min(len(a), len(b)) < 20:
        flags.append("INSUFFICIENT")
    if out["outside_5_95"] > .25:
        flags.append("OUTSIDE_RANGE")
    if np.isfinite(out["iqr_shift"]) and abs(out["iqr_shift"]) > 1:
        flags.append("MEDIAN_SHIFT")
    if iqr <= 1e-12:
        flags.append("ZERO_REFERENCE_IQR")
    if out["ks"] > .35:
        flags.append("KS_SHIFT")
    out["flag"] = "|".join(flags) if flags else "NO_FLAG"
    return out


def read_csv_fields(path, fields):
    """Never materialize reports, labels or arbitrary free-text columns."""
    return pd.read_csv(path, usecols=lambda c: c in fields, dtype=str).fillna("")


def header_metadata(root, explicit=None):
    allowed = {"SeriesInstanceUID"} | {x for names in HEADER_ALIASES.values() for x in names}
    paths = [root / "storage/cache/dicom" / name for name in ("headers.csv", "series_headers.csv", "index.csv")]
    if explicit:
        p = Path(explicit).resolve()
        if not p.is_relative_to((root / "storage/cache").resolve()) or p.suffix.lower() != ".csv":
            raise ValueError("--header-csv must be a local CSV under storage/cache")
        paths.append(p)
    result, provenance, errors, conflicted = {}, {}, [], set()
    for path in dict.fromkeys(paths):
        if not path.is_file():
            continue
        frame = read_csv_fields(path, allowed)
        provenance[str(path.relative_to(root))] = sha256(path)
        if "SeriesInstanceUID" not in frame:
            errors.append("unsupported header CSV (no SeriesInstanceUID): " + str(path.relative_to(root)))
            continue
        for row in frame.to_dict("records"):
            sid = row.pop("SeriesInstanceUID")
            if sid in conflicted:
                continue
            meta = {key: next((row[k] for k in keys if row.get(k, "") != ""), "")
                    for key, keys in HEADER_ALIASES.items()}
            if sid in result and result[sid] != meta:
                errors.append("conflicting acquisition headers for series " + sid)
                result[sid] = {"header_conflicted": True}
                conflicted.add(sid)  # Sticky even if a later row repeats one side.
            else:
                result[sid] = meta
    return result, provenance, errors


def series_tables(root):
    fields = {"StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"}
    series, splits, hashes, counts = {}, {}, {}, {}
    for split in ("train", "test"):
        path = root / f"storage/competition/{split}_series.csv"
        if not path.is_file():
            counts[split] = dict(status="MISSING")
            continue
        frame = read_csv_fields(path, fields)
        if not fields.issubset(frame.columns):
            raise ValueError("competition series schema differs")
        counts[split] = dict(series=len(frame), studies=frame.StudyInstanceUID.nunique())
        hashes[str(path.relative_to(root))] = sha256(path)
        for row in frame.to_dict("records"):
            uid, sid = row["StudyInstanceUID"], row["SeriesInstanceUID"]
            if uid in splits and splits[uid] != split:
                raise ValueError("competition train/test study overlap")
            if sid in series and series[sid] != row:
                raise ValueError("conflicting series metadata")
            splits[uid], series[sid] = split, row
    return series, splits, hashes, counts


def study_metadata(root):
    """Supplemental study-level scanner summary; never replace chosen-series fields."""
    path = root / "storage/labels/derived_meta/study_meta.csv"
    if not path.is_file():
        return {}, {}
    frame = read_csv_fields(path, {"StudyInstanceUID", "manufacturer", "field"})
    if "StudyInstanceUID" not in frame or frame.StudyInstanceUID.duplicated().any():
        raise ValueError("study acquisition metadata has missing/duplicate UID")
    rows = {r["StudyInstanceUID"]: dict(study_vendor=str(r.get("manufacturer", "")).strip().upper(),
                                       study_field_strength=str(r.get("field", "")))
            for r in frame.to_dict("records")}
    return rows, {str(path.relative_to(root)): sha256(path)}


def open_corpus(path):
    sys.path.insert(0, str(ROOT / "training/src")) if str(ROOT / "training/src") not in sys.path else None
    from raptor_common import Corpus
    # Preflight trusted local IDs without pickle before Corpus's legacy loader.
    ids = np.load(path / "ids.npy", allow_pickle=False)
    if ids.dtype.kind not in "US" or len(set(ids.astype(str))) != len(ids):
        raise ValueError("IDs must be unique non-pickled strings")
    corpus = Corpus(path)
    if corpus.contract != CONTRACT or corpus.depth != 64 or corpus.spec.get("img") != 336:
        raise ValueError("wrong production corpus contract")
    if [list(v) for v in corpus.spec["slots"]] != [["Sagittal", 1, 18], ["Sagittal", 0, 14],
                                                       ["Coronal", 1, 12], ["Coronal", 0, 8], ["Axial", -1, 12]]:
        raise ValueError("wrong slot groups")
    if corpus.masks.shape != (len(ids), 64) or not np.isin(corpus.masks, [0, 1]).all():
        raise ValueError("wrong mask contract")
    for name, keys in (("ids.npy", ("ids_sha256",)), ("masks.npy", ("masks_sha256", "mask_sha256"))):
        for key in keys:
            if key in corpus.manifest and corpus.manifest[key] != sha256(path / name):
                raise ValueError("corpus manifest checksum differs: " + name)
    if path.name.startswith("EXT_") and corpus.manifest.get("finished") is not True:
        raise ValueError("unfinished external corpus")
    stop = 0
    for start, end, name in corpus._bounds:
        contained(path, name)
        if start != stop or end <= start:
            raise ValueError("noncontiguous corpus shards")
        stop = end
    if stop != len(ids):
        raise ValueError("shard coverage differs from IDs")
    return corpus


def load_picks(path):
    result = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["uid"] in result:
                raise ValueError("duplicate picks UID")
            result[row["uid"]] = row
    return result


def external_index(root, source):
    source = ALIASES.get(source, source)
    directory = root / f"storage/cache/external/{source}/gen"
    path = directory / "index.csv"
    fields = {"study_key", "series_key", "plane", "fluid_sensitive", "rule_hit", "spacing_unknown",
              "npz_file", "npz_sha256", "n_slices", "rows", "cols", "subject_id", "visit"} | set(HEADER_ALIASES)
    if not path.exists():
        return directory, {}, {}
    frame = read_csv_fields(path, fields)
    if "series_key" not in frame or frame.series_key.duplicated().any():
        raise ValueError("missing/duplicate GENERAL series key")
    return directory, {r["series_key"]: r for r in frame.to_dict("records")}, {str(path.relative_to(root)): sha256(path)}


def selected_metadata(source, uid, slot_index, picks, index, series, headers, root):
    slots = picks.get(uid, {}).get("slots", [])
    pick = (slots[slot_index] if len(slots) > slot_index else None) or {}
    sid = str(pick.get("series", ""))
    meta = dict(index.get(sid, {}))
    if meta and str(meta.get("study_key", "")) != uid:
        raise ValueError("selected GENERAL series belongs to another study")
    if source == "COMPETITION":
        sm = series.get(sid, {})
        if sm and sm["StudyInstanceUID"] != uid:
            raise ValueError("selected series belongs to another study")
        meta.update(headers.get(sid, {}))
        # Only this known acquisition-header filename layout is supported.
        header_root = root / "storage/cache/dicom"
        for path in (contained(header_root, Path(uid) / sid / "header.json"),
                     contained(header_root, Path(uid) / (sid + ".json"))):
            if sid and path.is_file() and not meta.get("header_conflicted"):
                header = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(header, dict):
                    for key, names in HEADER_ALIASES.items():
                        meta[key] = next((header[k] for k in names if k in header), meta.get(key, ""))
                meta["header_path"] = str(path.relative_to(root))
                meta["header_sha256"] = sha256(path)
                break
        meta.update(fluid_sensitive=sm.get("Fluid_Sensitive", ""),
                    fat_suppression=sm.get("Fat_Suppression", ""), plane=sm.get("Anatomical_Plane", ""))
    else:
        # Only explicit known-contrast rules license an FS inference. Fluid flag
        # alone does not. Unknown contrasts remain missing.
        rule = str(meta.get("rule_hit", "")).lower()
        meta["fat_suppression"] = "1" if "fat-suppressed" in rule or "known contrast: corpdfs" in rule else ""
        if rule == "known contrast: mrnet coronal t1" or rule == "known contrast: corpd":
            meta["fat_suppression"] = "0"
    meta["selected_series"] = sid
    return pick, meta


def production_task(task):
    corpus, uid, indices, row, roi_dir = task
    volume = corpus.volume(uid)
    if volume.dtype != np.uint8 or volume.shape != (64, 336, 336):
        raise ValueError("production array shape/dtype differs")
    images = np.asarray(volume[indices], np.float32) / 255
    fluid = fat = None
    if roi_dir:
        path = contained(roi_dir, Path(row["source"]) / uid / (row["group"] + ".npz"))
        if path.is_file():
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z["slice_indices"], indices):
                    raise ValueError("ROI slices differ from sampled slices")
                fluid, fat = z["fluid"], z["fat"]
            if fluid.shape != images.shape or fat.shape != images.shape:
                raise ValueError("ROI volume shape differs")
            row["roi_sha256"] = sha256(path)
    return dict(row, **volume_metrics(images, fluid, fat))


def general_task(directory, meta, row, max_bytes=256 << 20):
    path = contained(directory, meta["npz_file"])
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    with zipfile.ZipFile(path) as archive:
        if sum(x.file_size for x in archive.infolist()) > max_bytes:
            raise ValueError("GENERAL uncompressed array exceeds 256 MiB cap")
    if sha256(path) != meta["npz_sha256"]:
        raise ValueError("GENERAL content hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        vol = z["vol"]
    if before != (path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError("GENERAL input changed during read")
    if vol.ndim != 3 or vol.nbytes > max_bytes or not np.isfinite(vol).all():
        raise ValueError("unsupported GENERAL volume")
    indices = slice_indices(range(len(vol)), 3)
    images = vol[indices].astype(np.float32)
    lo, hi = np.percentile(images, [2, 98])
    scaled = np.clip((images - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(images)
    base = dict(row, domain="general", slice_indices=json.dumps(indices.tolist()),
                native_window_low=float(lo), native_window_high=float(hi), native_dtype=str(vol.dtype),
                native_scale="sample-series-p02-p98", general_npz_sha256=meta["npz_sha256"],
                geometry_scope="acquisition_and_production_selection; slice_indices_are_native_audit_sample")
    return dict(base, **volume_metrics(scaled))


def summarize(frame, out, deadline=None):
    """Summarize every registered group; legacy deadline is ignored."""
    rows, comparisons, mix = [], [], []
    complete = True
    keys = ["source", "split", "group", "domain"]
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, key))
        for metric in METRICS:
            x = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            finite = x[np.isfinite(x)]
            rows.append(dict(base, metric=metric, n=len(x), n_finite=len(finite),
                             missing_fraction=1 - len(finite) / len(x),
                             **{f"q{q:02d}": v for q, v in zip(QUANTILES, np.percentile(finite, QUANTILES)
                                                              if len(finite) else [np.nan] * len(QUANTILES))}))
        if base["domain"] != "production":
            continue
        ref = frame[(frame.source == "COMPETITION") & (frame.split == "train") &
                    (frame.group == base["group"]) & (frame.domain == "production")]
        if base["source"] != "COMPETITION" or base["split"] != "train":
            for metric in METRICS:
                comparisons.append(dict(base, metric=metric, **shift_statistics(ref[metric], group[metric])))
    comp = frame[(frame.source == "COMPETITION") & (frame.domain == "production")]
    for name in ("vendor", "field_strength", "series_description", "fat_suppression",
                 "study_vendor", "study_field_strength"):
        if name not in comp:
            continue
        for (split, group_name), g in comp.groupby(["split", "group"]):
            for category, sub in g.groupby(name, dropna=False):
                category = str(category) if str(category) not in ("", "nan") else "MISSING"
                mix.append(dict(split=split, group=group_name, attribute=name, category=category,
                                n=len(sub), denominator=len(g), fraction=len(sub) / len(g)))
                for metric in METRICS:
                    x = pd.to_numeric(sub[metric], errors="coerce").to_numpy(float)
                    x = x[np.isfinite(x)]
                    rows.append(dict(source="COMPETITION", split=split, group=group_name, domain="production",
                                     attribute=name, category=category, metric=metric, n=len(sub), n_finite=len(x),
                                     missing_fraction=1 - len(x) / len(sub),
                                     **{f"q{q:02d}": v for q, v in zip(QUANTILES, np.percentile(x, QUANTILES)
                                                                      if len(x) else [np.nan] * len(QUANTILES))}))
    pd.DataFrame(rows).to_csv(out / "distributions.csv", index=False)
    pd.DataFrame(mix).to_csv(out / "competition_metadata_mix.csv", index=False)
    dump(out / "comparisons.json", comparisons)
    dump(out / "summary_status.json", dict(complete=complete, distribution_rows=len(rows), comparisons=len(comparisons)))
    return comparisons, complete


def montages(frame, corpora, out, deadline=None):
    """Render all selected tails; legacy deadline is ignored."""
    from PIL import Image, ImageDraw
    directory = out / "montages"
    directory.mkdir()
    records, complete = [], True
    prod = frame[frame.domain == "production"]
    for group_name, group in prod.groupby("group", sort=True):
        for metric in METRICS:
            selected = []
            for (source, split), sub in group.groupby(["source", "split"], sort=True):
                sub = sub[np.isfinite(pd.to_numeric(sub[metric], errors="coerce"))].sort_values([metric, "uid"])
                if len(sub):
                    selected.extend([(source, split, "low", sub.iloc[0]), (source, split, "high", sub.iloc[-1])])
            if not selected:
                continue
            width, height, columns = 228, 268, min(6, len(selected))
            canvas = Image.new("RGB", (width * columns, height * math.ceil(len(selected) / columns)), "#171717")
            draw = ImageDraw.Draw(canvas)
            filename = group_name + "__" + metric + ".png"
            for tile, (source, split, tail, row) in enumerate(selected):
                indices = json.loads(row.slice_indices)
                idx = indices[len(indices) // 2]
                im = np.asarray(corpora[source].volume(row.uid)[idx], np.uint8)
                thumb = Image.fromarray(im).resize((224, 224), Image.Resampling.BILINEAR).convert("RGB")
                col, line = tile % columns, tile // columns
                x, y = col * width, line * height
                canvas.paste(thumb, (x, y + 44))
                short = hashlib.sha256(str(row.uid).encode()).hexdigest()[:10]
                draw.text((x + 2, y + 2), f"{source} {split} {tail}\n{short} s{idx} {metric[:18]}\n{float(row[metric]):.5g}", fill="white")
                records.append(dict(file=filename, tile=tile, source=source, split=split, group=group_name,
                                    uid=row.uid, short_id=short, slice=idx, metric=metric,
                                    value=float(row[metric]), tail=tail, review="PENDING"))
            canvas.save(directory / filename)
    pd.DataFrame(records).to_csv(out / "montage_index.csv", index=False)
    return dict(sheets=len(set(r["file"] for r in records)), tiles=len(records), complete=complete, review="PENDING")


def report(out, manifest, comparisons):
    flagged = [r for r in comparisons if any(k in r["flag"] for k in ("OUTSIDE_RANGE", "MEDIAN_SHIFT", "KS_SHIFT"))]
    flagged.sort(key=lambda r: (-(r.get("ks") or 0), r["source"], r["group"], r["metric"]))
    lines = ["# EXP175 image-domain audit", "", f"Status: **{manifest['status']}**; v{VERSION}.", "",
             f"Verification: {manifest['verification']['path']} (sha256 {manifest['verification']['sha256']}); verified tool v{manifest['verification']['verified_tool_version']}.",
             f"Completed metric rows: {manifest.get('metric_rows', 0)}; source/error availability is in manifest.json.",
             "Per-phase wall durations/status are in manifest.json phase_timings; skipped durations are null, interrupted durations are checkpoint lower bounds.",
             "INVALID_INPUTS or FAILURE.json invalidates interpretation; COMPLETE_WITH_GAPS records unavailable data or requested omissions.",
             "Noise/SNR are processed-image proxies. Fluid/fat identity and FS correctness remain unassessed without validated ROIs.",
             "Production and GENERAL use different scales; do not pool them. fastMRI SC/+ overlap. Montages await human review.", "",
             "## Largest distribution alerts (descriptive, selected extremes)", "",
             "| Source/split | Group | Metric | KS | Outside train 5–95% | Flag |", "|---|---|---|---:|---:|---|"]
    for r in flagged[:30]:
        lines.append(f"| {r['source']}/{r['split']} | {r['group']} | {r['metric']} | {r['ks']:.3f} | {r['outside_5_95']:.3f} | {r['flag']} |")
    if not flagged:
        lines.append("| No alerts or insufficient measured support | | | | | |")
    lines += ["", "## Decision guidance", "",
              "No training or serving change is admitted by this descriptive audit. Read behavior.json and behavior tables with the montage review; sparse groups and weak labels can reverse apparent differences.", "",
              "| Proposal | Expected value / evidence needed | Cost estimate | Plan link |", "|---|---|---|---|",
              "| Stratify external ruler | High information; use measured coverage, label mapping and within-source associations | CPU report + manual review | H-B-65 |",
              "| Noise-matched augmentation | Uncertain small gain; needs aligned shift and repeated within-source weakness | 1–3 GPU h paired small-member screen | Future member training |",
              "| Intensity harmonization | Low-to-medium value after existing windowing; needs clipping/contrast mechanism | 1–2 CPU h + paired fit | H-B-65 / future members |",
              "| Exclude/down-weight source | No exclusion from image shift alone; requires source-specific paired gain | 1–3 GPU h | Future members |",
              "| Inference denoising | Low prior value; can erase small lesions and shift trained inputs | One paired frozen OOF inference pass | Future serving experiment |",
              "| Image-definition vs weak-ruler question | This audit cannot resolve it; keep registered comparison intact | Read-only case-mix review | H-B-82 |",
              "| No action | Preferred if associations are weak, sparse, contradictory or solely between sources | Zero GPU time | All |", "",
              "Costs are estimates, not measured runtimes or promises of score improvement. Confounding by anatomy, severity, resolution, sequence and missing slots remains. Test means only the locally cached test population, never the hidden-test distribution.", "",
              "See PROTOCOL.md in the experiment root for the frozen sampling, metrics and interpretation; manifest.json binds its hash."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    started = time.monotonic()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if not out.is_relative_to((root / BASE).resolve()):
        raise ValueError("--out must be beneath EXP175_IMAGE_DOMAIN_AUDIT")
    verification = verification_provenance(args.verification_note)
    out.mkdir(parents=True, exist_ok=False)
    protocol = root / BASE / "PROTOCOL.md"
    manifest = dict(version=VERSION, status="RUNNING", seed=args.seed, args=vars(args),
                    protocol_sha256=sha256(protocol), input_hashes={}, inventory=[], errors=[], gaps=[],
                    scope="local sampled image audit; no test labels, no inference",
                    timing_policy="measurement_only; legacy max_seconds ignored",
                    verification=verification, source_hashes=verification["source_hashes"])
    timings = PhaseTimings(manifest, out, started)
    timings.checkpoint()
    try:
        return _run_phases(args, root, out, manifest, started, None, verification, timings)
    except BaseException:
        manifest["status"] = "FAILED"
        timings.finish("FAILED")
        raise


def _run_phases(args, root, out, manifest, started, work_deadline, verification, timings):
    if args.behavior_only:
        for name in ("extraction", "summary", "montage", "input_validation"):
            timings.skip(name, "SKIPPED_REUSED_METRICS")
        timings.start("metrics_load")
        previous = Path(args.behavior_only).resolve()
        frame, provenance = load_previous_metrics(root, previous)
        manifest["metrics_provenance"] = provenance
        manifest["input_hashes"].update({provenance["path"]: provenance["sha256"],
                                         provenance["manifest_path"]: provenance["manifest_sha256"]})
        comparisons = []
        timings.finish()
    else:
        timings.skip("metrics_load", "SKIPPED_FRESH_EXTRACTION")
        timings.start("extraction")
        series, splits, hashes, series_counts = series_tables(root)
        headers, hh, errors = header_metadata(root, args.header_csv)
        study_meta, study_hashes = study_metadata(root)
        manifest["input_hashes"].update(hashes | hh | study_hashes)
        manifest.update(competition_series=series_counts, acquisition_header_series=len(headers))
        manifest["study_metadata"] = dict(studies=len(study_meta), scope="supplemental study-level summary; not selected-series headers")
        manifest["errors"].extend(errors)
        for split, counts in series_counts.items():
            if counts.get("status") == "MISSING":
                manifest["gaps"].append(f"COMPETITION {split}_series.csv missing; split/mix cannot be fully checked")
        paths = {"COMPETITION": root / "storage/cache/derived/RAPTOR64_336_C140_S02"}
        for p in sorted((root / "storage/cache/derived").glob("EXT_*_RAPTOR64_336_C140")):
            paths[p.name[4:-len("_RAPTOR64_336_C140")]] = p
        for source in REGISTERED:
            paths.setdefault(source, root / f"storage/cache/derived/EXT_{source}_RAPTOR64_336_C140")
        corpora, rows, coverage, selections, snapshots = {}, [], [], [], {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for source, path in paths.items():
                if not (path / "manifest.json").exists():
                    manifest["inventory"].append(dict(source=source, status="MISSING_PRODUCTION_CACHE"))
                    continue
                try:
                    corpus = open_corpus(path)
                    picks = load_picks(path / "picks.jsonl")
                    directory, index, ih = external_index(root, source) if source != "COMPETITION" else (None, {}, {})
                    manifest["input_hashes"].update(ih)
                    for name in ("manifest.json", "ids.npy", "masks.npy", "picks.jsonl"):
                        if (path / name).exists():
                            manifest["input_hashes"][str((path / name).relative_to(root))] = sha256(path / name)
                    for _, _, name in corpus._bounds:
                        p = contained(path, name)
                        snapshots[str(p.relative_to(root))] = (p.stat().st_size, p.stat().st_mtime_ns)
                    corpora[source] = corpus
                except Exception as exc:
                    manifest["inventory"].append(dict(source=source, status="INVALID_CACHE", error=str(exc)))
                    continue
                inventory = dict(source=source, status="AVAILABLE", total_studies=len(corpus),
                                 general_series=len(index), acquisition_pool=ALIASES.get(source, source))
                if len(picks) < len(corpus):
                    manifest["gaps"].append(f"{source}: selected-series picks incomplete")
                if source != "COMPETITION" and not index:
                    manifest["gaps"].append(f"{source}: GENERAL index missing; native images/acquisition metadata unavailable")
                manifest["inventory"].append(inventory)
                split_groups = {}
                for uid in map(str, corpus.ids):
                    split = splits.get(uid, "unknown") if source == "COMPETITION" else "external"
                    split_groups.setdefault(split, []).append(uid)
                if source == "COMPETITION" and "test" not in split_groups:
                    manifest["gaps"].append("No locally cached test images with verified test-series membership")
                for split, ids in split_groups.items():
                    for gi, (group_name, start, stop, fluid) in enumerate(SLOTS):
                        valid = [uid for uid in ids if np.any(corpus.mask(uid)[start:stop])]
                        cap = args.sample_competition if source == "COMPETITION" else args.sample_external
                        chosen = stable_sample(valid, cap, args.seed, source + split + group_name)
                        coverage.append(dict(source=source, split=split, group=group_name, population=len(ids),
                                             filled_studies=len(valid), study_fill_rate=len(valid) / len(ids),
                                             slot_fill_rate=sum(int(corpus.mask(u)[start:stop].sum()) for u in ids) / (len(ids) * (stop-start)),
                                             selected=len(chosen)))
                        general_ids = set(stable_sample(chosen, args.sample_general, args.seed, "general" + source + group_name))
                        for batch_start in range(0, len(chosen), args.workers):
                            tasks, native = [], []
                            for uid in chosen[batch_start:batch_start + args.workers]:
                                try:
                                    indices = slice_indices(np.flatnonzero(corpus.mask(uid)[start:stop]) + start)
                                    pick, meta = selected_metadata(source, uid, gi, picks, index, series, headers, root)
                                    row = dict(source=source, split=split, uid=uid, domain="production", group=group_name,
                                               slice_indices=json.dumps(indices.tolist()), selected_series=meta["selected_series"],
                                               slot_fluid=fluid, actual_fluid=number(meta.get("fluid_sensitive")),
                                               fat_suppression=str(meta.get("fat_suppression", "")),
                                               vendor=str(meta.get("vendor", "")).strip().upper(),
                                               field_strength=str(meta.get("field_strength", "")),
                                               series_description=str(meta.get("series_description", "")))
                                    row.update(fluid_slot_status(fluid, row["actual_fluid"]))
                                    row.update(study_meta.get(uid, {}) if source == "COMPETITION" else {})
                                    row.setdefault("study_vendor", "")
                                    row.setdefault("study_field_strength", "")
                                    row.update(geometry(pick, meta, int(corpus.mask(uid)[start:stop].sum()), stop-start))
                                    tasks.append((corpus, uid, indices, row, args.roi_dir))
                                    selections.append({k: row[k] for k in ("source", "split", "uid", "group", "slice_indices", "selected_series")})
                                    if meta.get("header_path"):
                                        manifest["input_hashes"][meta["header_path"]] = meta["header_sha256"]
                                    if uid in general_ids and meta.get("npz_file"):
                                        native.append((directory, meta, row))
                                except Exception as exc:
                                    manifest["errors"].append(dict(source=source, uid=uid, group=group_name, error=str(exc)))
                            futures = [pool.submit(production_task, task) for task in tasks]
                            for task, future in zip(tasks, futures):
                                try:
                                    rows.append(future.result())
                                except Exception as exc:
                                    manifest["errors"].append(dict(source=source, uid=task[1], group=group_name, error=str(exc)))
                            for item in native:
                                try:
                                    rows.append(general_task(*item))
                                except Exception as exc:
                                    manifest["errors"].append(dict(source=source, uid=item[2]["uid"], domain="general", error=str(exc)))
                # Durable per-source checkpoint preserves measured progress on external interruption.
                pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
                manifest["metrics_sha256"] = sha256(out / "metrics.csv")
                pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
                pd.DataFrame(selections).to_csv(out / "sample_manifest.csv", index=False)
                manifest["metric_rows"] = len(rows)
                manifest["shard_snapshots"] = snapshots
                timings.checkpoint()
                print(json.dumps(dict(source=source, metric_rows=len(rows), seconds=round(time.monotonic()-started))), flush=True)
        timings.finish("COMPLETE")
        if not rows:
            manifest["status"] = "NO_VALID_IMAGES"
            for name in ("summary", "montage", "input_validation", "behavior"):
                timings.skip(name, "SKIPPED_NO_VALID_IMAGES")
            timings.start("report")
            report(out, manifest, [])
            timings.finish()
            return 2
        frame = pd.DataFrame(rows)
        comp_rows = frame[(frame.source == "COMPETITION") & (frame.domain == "production")]
        for attribute in ("vendor", "field_strength", "series_description"):
            if comp_rows.empty or comp_rows[attribute].eq("").all():
                manifest["gaps"].append(f"Competition chosen-series {attribute} unavailable")
        for metric in METRICS:
            if metric not in frame:
                frame[metric] = np.nan
        timings.start("summary")
        comparisons, summary_complete = summarize(frame, out, work_deadline)
        manifest["summary_complete"] = summary_complete
        timings.finish("COMPLETE")
        timings.start("montage")
        manifest["montages"] = montages(frame, corpora, out, work_deadline)
        timings.finish("COMPLETE")
        timings.start("input_validation")
        manifest["shard_snapshots"] = snapshots
        for name, old in snapshots.items():
            p = root / name
            if not p.exists() or (p.stat().st_size, p.stat().st_mtime_ns) != tuple(old):
                manifest["errors"].append("INPUT_SHARD_CHANGED: " + name)
        # Metadata hashes are checked again: no stale join to a cache builder.
        for name, old in list(manifest["input_hashes"].items()):
            if not (root / name).exists() or sha256(root / name) != old:
                manifest["errors"].append("INPUT_METADATA_CHANGED: " + name)
        provenance = metrics_provenance(root, out / "metrics.csv", out / "manifest.json", manifest)
        # The current run's manifest is finalized below. Its source binding is
        # the captured input hashes, not a hash of an intermediate manifest.
        provenance.pop("manifest_sha256")
        manifest["metrics_provenance"] = provenance
        timings.finish("INVALID_INPUTS" if any(isinstance(error, str) and error.startswith("INPUT_")
                                               for error in manifest["errors"]) else "COMPLETE")
    invalid = any(isinstance(error, str) and error.startswith("INPUT_") for error in manifest["errors"])
    if not args.skip_behavior and not invalid:
        timings.start("behavior")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools.exp175_behavior import run_behavior
        manifest["behavior"] = run_behavior(root, frame, out, bootstraps=args.bootstraps,
                                             seed=args.seed, deadline=work_deadline,
                                             metrics_provenance=provenance, verification=verification)
        timings.finish(manifest["behavior"]["status"])
    else:
        manifest["behavior"] = {"status": "SKIPPED_INVALID_INPUTS" if invalid else
                               "SKIPPED_REQUEST"}
        timings.skip("behavior", manifest["behavior"]["status"])
    manifest["metric_rows"] = len(frame)
    manifest["elapsed_seconds"] = time.monotonic() - started
    missing = any(x["status"] != "AVAILABLE" for x in manifest["inventory"])
    missing = missing or bool(manifest["behavior"].get("missing")) or manifest["behavior"].get("status") != "COMPLETE"
    invalid = any(isinstance(error, str) and error.startswith("INPUT_") for error in manifest["errors"])
    manifest["status"] = ("INVALID_INPUTS" if invalid else
                          "COMPLETE_WITH_GAPS" if missing or manifest["errors"] or manifest["gaps"] else "COMPLETE")
    timings.start("report")
    report(out, manifest, comparisons)
    timings.finish()
    print(json.dumps(dict(status=manifest["status"], metric_rows=len(frame), seconds=round(manifest["elapsed_seconds"]))))
    return 2 if invalid else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--out", default=str(ROOT / BASE / "run-v1"))
    p.add_argument("--verification-note", required=True,
                   help="Independent PASS JSON note binding verified_tool_version and both source_hashes")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-seconds", type=int, default=7200,
                   help="Legacy compatibility option; ignored (timing is measurement only)")
    p.add_argument("--seed", type=int, default=175)
    p.add_argument("--sample-competition", type=int, default=1200)
    p.add_argument("--sample-external", type=int, default=256)
    p.add_argument("--sample-general", type=int, default=32)
    p.add_argument("--bootstraps", type=int, default=200)
    p.add_argument("--header-csv", help="Optional selected-series acquisition metadata CSV under storage/cache")
    p.add_argument("--roi-dir", help="Local validated boolean masks: SOURCE/UID/GROUP.npz (slice_indices, fluid, fat)")
    p.add_argument("--behavior-only", help="Prior EXP175 metrics.csv; rerun saved-prediction analysis in a new output directory")
    p.add_argument("--skip-behavior", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if not 1 <= args.workers <= 6:
        raise ValueError("workers must be 1..6")
    if min(args.sample_competition, args.sample_external) < 1 or args.sample_general < 0 or args.bootstraps < 1:
        raise ValueError("sample/replicate counts must be positive (GENERAL may be zero)")
    if args.roi_dir:
        r = Path(args.roi_dir).resolve()
        if not r.is_relative_to((Path(args.root) / BASE).resolve()):
            raise ValueError("--roi-dir must be local under EXP175_IMAGE_DOMAIN_AUDIT")
    if Path(args.out).exists():
        raise FileExistsError("Choose a fresh output directory; prior outputs are immutable")
    verification = verification_provenance(args.verification_note)
    if not args.worker:
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv, "--worker"],
                               check=False)
        return child.returncode
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return run(args)
    except Exception as exc:
        out = Path(args.out).resolve()
        if not isinstance(exc, FileExistsError) and out.is_relative_to((Path(args.root).resolve() / BASE).resolve()) and out.exists():
            dump(out / "FAILURE.json", dict(status="FAILED", error=str(exc), exception=type(exc).__name__,
                                              version=VERSION, verification=verification))
            path = out / "manifest.json"
            if path.exists():
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifest["status"] = "FAILED"
                dump(path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
