"""EXP155 OAI-only data adapter. CPU passive; no network or GPU operations.

The existing EXP135 OAI GENERAL index is the sole image input. Its series
selection and normalization are replayed at EXP143's native 96 x 384 geometry.
EXP136's conservative, provenance-sealed OAI_12.csv is the sole label input.
"""
from __future__ import annotations

import csv
import json
import os
import math
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.extcache import common
from tools.exp143_v2 import ROOT, TARGETS, sha256 as file_sha256
from exp143_v2 import CONTRACT, SPEC

SOURCE = "oai"
DEFAULT_CACHE = ROOT / "storage/cache/derived/EXP155_OAI_RAPTOR96_384_C140_S02"
DEFAULT_LABELS = ROOT / "storage/labels/external/OAI_12.csv"
DEFAULT_LABEL_RECEIPT = ROOT / "storage/labels/external/OAI_receipt.json"
DEFAULT_OUTPUT_LABELS = ROOT / "storage/labels/derived_constellation/exp155_oai_masked_v1.csv"
ALLOWED = ("PF OA", "Lateral OA", "Synovitis")
SCHEMA = "exp155-oai-data-v1.0.0"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _atomic_json(path: Path, value):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def _registration(root: Path, ids: list[str], index_path: Path) -> dict:
    return dict(schema=SCHEMA, contract=CONTRACT, spec=SPEC, ids=ids,
                source="EXP135 OAI GENERAL; decoded original DICOM tar series",
                gen_index=str(index_path.resolve()), gen_index_sha256=file_sha256(index_path),
                gen_receipt_sha256=file_sha256(common.gen_dir(SOURCE, root) / "receipt.json"),
                code_sha256={p: file_sha256(ROOT / p) for p in (
                    "tools/exp155_oai_data.py", "tools/extcache/common.py",
                    "tools/extcache/sources.py", "training/src/exp143_own0949.py",
                    "training/src/exp143_v2.py", "tools/exp143_v2.py",
                    "training/src/raptor_corpus.py")})


def render_study(rows: list[dict], root: Path):
    """Render with EXP135 OAI slot selection, EXP143 depth and image size."""
    import cv2
    import raptor_corpus
    from raptor_corpus import _crop_resize, pick_series

    cv2.setNumThreads(1)
    raptor_corpus._WORKER["cv2"] = cv2
    depth, img = SPEC["depth"], SPEC["img"]
    volume = np.zeros((depth, img, img), np.uint8)
    mask = np.zeros(depth, np.uint8)
    picks = []
    pick_rows = [dict(row, Anatomical_Plane=row["plane"],
                      Fluid_Sensitive=row["fluid_sensitive"],
                      SeriesInstanceUID=row["series_key"]) for row in rows]
    known = bool(rows) and all(str(r.get("rule_hit", "")).startswith("known contrast") for r in rows)
    used, offset = set(), 0
    for plane, fluid, count in SPEC["slots"]:
        selected = (common.pick_strict(pick_rows, plane, fluid, used) if known
                    else pick_series(pick_rows, plane, fluid, used))
        if selected is None:
            picks.append(None)
            offset += count
            continue
        used.add(selected["series_key"])
        series = common.load_series(selected, SOURCE, root)  # verifies GENERAL NPZ hash
        n = len(series.vol)
        lo = int(n * SPEC["span"][0])
        hi = max(int(n * SPEC["span"][1]) - 1, lo)
        chosen = np.linspace(lo, hi, count).round().astype(int) if n > 1 else np.zeros(count, int)
        arrays = series.vol[chosen].astype(np.float32)
        spacings = series.per_slice_spacing[chosen]
        good = spacings[np.isfinite(spacings) & (spacings > 0)]
        median_spacing = float(np.median(good)) if len(good) else .5
        loq, hiq = np.percentile(np.concatenate([a.ravel() for a in arrays]), [2., 98.])
        for j, (pixels, spacing) in enumerate(zip(arrays, spacings)):
            normalized = np.clip((pixels - loq) / (hiq - loq + 1e-6), 0, 1)
            if str(selected.get("spacing_unknown", "")).lower() in ("true", "1"):
                normalized = cv2.resize(normalized, (img, img), interpolation=cv2.INTER_AREA)
            else:
                effective_spacing = float(spacing) if np.isfinite(spacing) and spacing > 0 else median_spacing
                normalized = _crop_resize(normalized, effective_spacing, SPEC["crop_mm"], img)
            volume[offset + j] = (normalized * 255).astype(np.uint8)
        # The EXP143 mask is nonzero rendered slices, not merely selected slots.
        mask[offset:offset + count] = (volume[offset:offset + count].reshape(count, -1).sum(1) > 0)
        picks.append(dict(series=selected["series_key"], files=n, slices=chosen.tolist(),
                          rendered=count, window=[float(loq), float(hiq)],
                          median_spacing=median_spacing, rule_hit=selected["rule_hit"],
                          gen_sha256=selected["npz_sha256"]))
        offset += count
    if offset != depth or not mask.any():
        raise ValueError("empty or invalid OAI study geometry")
    return volume, mask, picks


def _one_study(out: Path, row: int, uid: str, rows: list[dict], root: Path):
    npy = out / f"study_{row:05d}.npy"
    receipt = out / f"study_{row:05d}.json"
    if receipt.exists():
        old = _json(receipt)
        if old.get("uid") != uid or not npy.is_file() or file_sha256(npy) != old.get("file_sha256"):
            raise ValueError("completed OAI cache study changed")
        return dict(row=row, uid=uid, sha256=old["file_sha256"], receipt_sha256=file_sha256(receipt))
    volume, mask, picks = render_study(rows, root)
    tmp = npy.with_name(npy.name + ".partial")
    with tmp.open("wb") as stream:
        np.save(stream, volume, allow_pickle=False)
    digest = file_sha256(tmp)
    if npy.exists():
        if file_sha256(npy) != digest:
            tmp.unlink()
            raise ValueError("orphaned OAI volume differs from deterministic rebuild")
        tmp.unlink()
    else:
        os.replace(tmp, npy)
    _write_new(receipt, dict(uid=uid, row=row, mask=mask.tolist(),
                             positive_mask_slices=int(mask.sum()), slots=picks,
                             file_sha256=digest,
                             gen_sha256=[r["npz_sha256"] for r in rows]))
    return dict(row=row, uid=uid, sha256=digest, receipt_sha256=file_sha256(receipt))


def build_cache(args):
    """Resume a frozen EXP135 GENERAL snapshot. Failed studies stay logged."""
    root = Path(getattr(args, "root", ROOT))
    out = Path(getattr(args, "oai_cache", DEFAULT_CACHE))
    index_path = common.gen_dir(SOURCE, root) / "index.csv"
    rows = common.read_index(SOURCE, root)
    if not rows or not index_path.is_file():
        raise ValueError("EXP135 OAI GENERAL index missing; build it with extcache gen first")
    by_study = defaultdict(list)
    for row in rows:
        if row["source"] != SOURCE or not re.fullmatch(r"OAI_[0-9]+_[RL]_V00", row["study_key"]):
            raise ValueError("GENERAL index has non-OAI records")
        by_study[row["study_key"]].append(row)
    ids = sorted(by_study)
    # A full OAI corpus is roughly 34 GiB. Keep room for temporary files and
    # the running competition job's checkpoints before writing anything.
    reserve_gib = float(getattr(args, "min_free_gib", 20.0))
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("min_free_gib must be nonnegative")
    projected = len(ids) * SPEC["depth"] * SPEC["img"] ** 2
    if shutil.disk_usage(out.parent if out.parent.exists() else root).free < projected + reserve_gib * 2**30:
        raise RuntimeError("insufficient free disk for EXP155 OAI cache plus reserve")
    registration = _registration(root, ids, index_path)
    out.mkdir(parents=True, exist_ok=True)
    regpath = out / "build-registration.json"
    if regpath.exists():
        if _json(regpath) != registration:
            raise ValueError("OAI input index/code changed; choose a new cache directory")
        if (out / "manifest.json").exists():
            check_cache(out, full_hash=False)
            return dict(status="COMPLETE", studies=len(ids), output=str(out),
                        manifest_sha256=file_sha256(out / "manifest.json"))
        if not getattr(args, "resume", False):
            raise ValueError("existing incomplete OAI build requires --resume")
    else:
        _write_new(regpath, registration)
    completed, failed = [], 0
    started = time.monotonic()
    for row, uid in enumerate(ids):
        try:
            completed.append(_one_study(out, row, uid, by_study[uid], root))
        except Exception as exc:  # resumable failure log; no silent cohort drop
            failed += 1
            _atomic_json(out / f"study_{row:05d}.failure.json",
                         dict(uid=uid, row=row, error=f"{type(exc).__name__}: {exc}"))
        if (row + 1) % 25 == 0 or row + 1 == len(ids):
            seconds = time.monotonic() - started
            print(json.dumps(dict(processed=row + 1, total=len(ids), failed=failed,
                seconds=seconds, projected_remaining_seconds=seconds / (row + 1) * (len(ids) - row - 1))),
                flush=True)
    if failed:
        return dict(status="INCOMPLETE", completed=len(completed), failed=failed,
                    total=len(ids), output=str(out))
    manifest = dict(schema=SCHEMA, contract=CONTRACT, spec=SPEC, ids=ids,
                    files=completed, finished=True,
                    build_registration_sha256=file_sha256(regpath))
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if _json(manifest_path) != manifest:
            raise ValueError("finished OAI manifest differs from resumed build")
    else:
        _write_new(manifest_path, manifest)
    check_cache(out, full_hash=False)
    return dict(status="COMPLETE", studies=len(ids), output=str(out),
                manifest_sha256=file_sha256(manifest_path))


def check_cache(path, full_hash=False):
    """EXP143 compatibility plus frozen OAI source/receipt/series provenance."""
    from tools.exp143_v2 import check_corpus
    path = Path(path)
    manifest = check_corpus(path, full_hash=full_hash)
    registration = _json(path / "build-registration.json")
    if (manifest.get("schema") != SCHEMA or registration.get("schema") != SCHEMA
            or manifest["build_registration_sha256"] != file_sha256(path / "build-registration.json")
            or manifest["ids"] != registration["ids"]
            or any(not re.fullmatch(r"OAI_[0-9]+_[RL]_V00", uid) for uid in manifest["ids"])):
        raise ValueError("OAI cache registration mismatch")
    if full_hash:
        index = Path(registration["gen_index"])
        if (file_sha256(index) != registration["gen_index_sha256"]
                or file_sha256(index.parent / "receipt.json") != registration["gen_receipt_sha256"]):
            raise ValueError("OAI GENERAL source snapshot changed")
        for relative, digest in registration["code_sha256"].items():
            if file_sha256(ROOT / relative) != digest:
                raise ValueError("OAI cache build source code changed: " + relative)
        with index.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                source_file = index.parent / row["npz_file"]
                if file_sha256(source_file) != row["npz_sha256"]:
                    raise ValueError("OAI GENERAL series bytes changed")
    for i, uid in enumerate(manifest["ids"]):
        receipt = _json(path / f"study_{i:05d}.json")
        if (receipt.get("row") != i or receipt.get("uid") != uid
                or len(receipt.get("mask", [])) != SPEC["depth"]
                or sum(receipt["mask"]) != receipt["positive_mask_slices"]
                or len(receipt.get("slots", [])) != len(SPEC["slots"])
                or not receipt.get("gen_sha256")):
            raise ValueError("OAI cache provenance/slot receipt mismatch")
    return manifest


def _read_labels(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["study_key", *TARGETS]:
            raise ValueError("EXP136 OAI_12 target schema changed")
        result = {}
        for row in reader:
            uid = row["study_key"]
            if not uid.startswith("OAI_") or uid in result or None in row:
                raise ValueError("invalid/duplicate OAI label identity")
            result[uid] = row
    return result


def build_labels(args):
    """Write only three assessable targets; all other target masks remain zero."""
    cache = Path(getattr(args, "oai_cache", DEFAULT_CACHE))
    source = Path(getattr(args, "oai_labels_source", DEFAULT_LABELS))
    source_receipt = Path(getattr(args, "oai_labels_receipt", DEFAULT_LABEL_RECEIPT))
    out = Path(args.out)
    if out.exists() or out.with_suffix(".provenance.json").exists():
        raise FileExistsError("label output/provenance exists; choose a fresh version")
    manifest = check_cache(cache, full_hash=False)
    receipt = _json(source_receipt)
    if (receipt.get("contract") != "EXP136-v1"
            or receipt.get("outputs_sha256", {}).get(source.name) != file_sha256(source)):
        raise ValueError("EXP136 OAI labels/provenance changed")
    source_rows = _read_labels(source)
    if not set(manifest["ids"]) <= set(source_rows):
        raise ValueError("OAI cache study lacks EXP136 label row")
    counts = {target: dict(positive=0, negative=0, unknown=0) for target in ALLOWED}
    included = []
    rendered_rows = []
    for uid in manifest["ids"]:
        original = source_rows[uid]
        label = {target: "" for target in TARGETS}
        masks = {"mask_" + target: 0 for target in TARGETS}
        for target in ALLOWED:
            value = original[target].strip()
            if value not in ("", "0", "1"):
                raise ValueError(f"nonbinary EXP136 OAI label: {uid}/{target}")
            counts[target]["unknown" if value == "" else "positive" if value == "1" else "negative"] += 1
            if value:
                label[target] = value
                masks["mask_" + target] = 1
        if any(masks.values()):
            rendered_rows.append({"study_key": uid, **label, **masks})
            included.append(uid)
    if not included:
        raise ValueError("zero OAI studies have an assessable author target")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["study_key", *TARGETS, *("mask_" + target for target in TARGETS)]
    with out.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rendered_rows)
    provenance = dict(schema=SCHEMA, source="EXP136 OAI_12.csv conservative mapping",
                      targets=list(ALLOWED), other_targets_masked=True, rows=len(included),
                      cache_manifest_sha256=file_sha256(cache / "manifest.json"),
                      exp136_labels_sha256=file_sha256(source),
                      exp136_receipt_sha256=file_sha256(source_receipt),
                      output_sha256=file_sha256(out), counts=counts,
                      mapping=dict(
                          pf_oa="MOAKS cartilage full-thickness >0 in PM/PL/FMA/FLA AND PF osteophyte >=1; all cartilage 0 and all osteophytes 0 negative; otherwise unknown",
                          lateral_oa="MOAKS lateral cartilage full-thickness >0 AND lateral osteophyte >=1; full compartment zeros negative; merge qualified XRKL/JSN, discordance unknown",
                          synovitis="MOAKS MSYIC grade 2/3 positive, 0/1 negative; missing/invalid unknown; effusion never substituted"))
    provenance["csv_sha256"] = provenance["output_sha256"]
    _write_new(out.with_suffix(".provenance.json"), provenance)
    return dict(status="COMPLETE", studies=len(included), counts=counts,
                labels=str(out), sha256=provenance["output_sha256"])


def add_cli_arguments(sub):
    """Register the two CPU-only data commands in the EXP155 driver."""
    cache = sub.add_parser("build-oai-cache", help="Render local EXP135 OAI GENERAL into EXP143 geometry")
    cache.add_argument("--root", type=Path, default=ROOT, help="workspace root containing EXP135 GENERAL")
    cache.add_argument("--oai-cache", type=Path, default=DEFAULT_CACHE)
    cache.add_argument("--min-free-gib", type=float, default=20.0)
    cache.add_argument("--resume", action="store_true", help="continue an incomplete registered OAI cache")
    labels = sub.add_parser("oai-labels", help="derive conservative three-target masked OAI labels")
    labels.add_argument("--oai-cache", type=Path, default=DEFAULT_CACHE)
    labels.add_argument("--oai-labels-source", type=Path, default=DEFAULT_LABELS)
    labels.add_argument("--oai-labels-receipt", type=Path, default=DEFAULT_LABEL_RECEIPT)
    labels.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_LABELS)
