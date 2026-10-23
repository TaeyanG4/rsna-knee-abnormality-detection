"""GENERAL series storage and exact Raptor64 production rendering."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
from raptor_common import CONTRACTS, Corpus  # noqa: E402
from raptor_corpus import _crop_resize, pick_series  # noqa: E402
import raptor_corpus  # noqa: E402

CONTRACT = "raptor64_336_c140_s02"
MIN_FREE = 100 * 2**30
INDEX_FIELDS = ("source", "subject_id", "knee_side", "visit", "study_key", "series_key",
                "plane", "fluid_sensitive", "rule_hit", "series_description", "rows", "cols",
                "n_slices", "pixel_spacing", "slice_spacing", "thickness", "TE", "TR",
                "field_strength", "vendor", "dtype", "npz_sha256", "npz_file",
                "source_archive_member", "spacing_unknown")


@dataclass
class Series:
    meta: dict
    vol: np.ndarray
    ipp: np.ndarray
    iop: np.ndarray
    pixel_spacing: np.ndarray
    slice_spacing: float = float("nan")
    per_slice_spacing: np.ndarray | None = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_dir(source: str, root: Path = ROOT) -> Path:
    return Path(root) / "storage/cache/external" / source.upper() / "gen"


def prod_dir(source: str, root: Path = ROOT) -> Path:
    return Path(root) / "storage/cache/derived" / f"EXT_{source.upper()}_RAPTOR64_336_C140"


def disk(root: Path = ROOT) -> dict:
    d = shutil.disk_usage(root)
    return dict(total=d.total, used=d.used, free=d.free, min_free=MIN_FREE)


def require_space(root: Path = ROOT):
    d = disk(root)
    if d["free"] < MIN_FREE:
        raise RuntimeError(f"refusing cache write: {d['free'] / 2**30:.1f} GiB free, minimum 100 GiB")


def _atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_index(source: str, root: Path = ROOT) -> list[dict]:
    path = gen_dir(source, root) / "index.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_index(source: str, rows: list[dict], root: Path = ROOT):
    path = gen_dir(source, root) / "index.csv"
    tmp = path.with_suffix(".csv.partial")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _native_dtype(vol: np.ndarray) -> np.ndarray:
    # uint16 is exact for ordinary DICOM. Signed/large integral data cannot fit uint16.
    if not np.isfinite(vol).all():
        raise ValueError("non-finite series pixels")
    if np.all(vol == np.floor(vol)):
        if vol.min() >= 0 and vol.max() <= 65535:
            return vol.astype(np.uint16)
        return vol.astype(np.int32)
    return vol.astype(np.float16)


def add_series(source: str, series: Series, root: Path = ROOT) -> dict:
    require_space(root)
    d = gen_dir(source, root)
    d.mkdir(parents=True, exist_ok=True)
    meta = dict(series.meta)
    key = str(meta["series_key"])
    name = hashlib.sha256(key.encode()).hexdigest()[:24] + ".npz"
    target = d / name
    if target.exists():
        existing = next((r for r in read_index(source, root) if r["series_key"] == key), None)
        if existing and sha256(target) == existing["npz_sha256"]:
            return existing
    vol = _native_dtype(np.asarray(series.vol))
    if vol.ndim != 3 or vol.shape[0] < 1:
        raise ValueError("series must be slices-first 3D")
    ipp = np.asarray(series.ipp, np.float32)
    iop = np.asarray(series.iop, np.float32)
    ps = np.asarray(series.pixel_spacing, np.float32)
    pss = np.asarray(series.per_slice_spacing if series.per_slice_spacing is not None
                     else np.full(vol.shape[0], ps[0]), np.float32)
    if ipp.shape != (vol.shape[0], 3) or iop.shape != (6,) or ps.shape != (2,) or pss.shape != (vol.shape[0],):
        raise ValueError("invalid GENERAL geometry")
    tmp = d / (name + ".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, vol=vol, ipp=ipp, iop=iop, pixel_spacing=ps,
                            slice_spacing=np.float32(series.slice_spacing), per_slice_spacing=pss)
    os.replace(tmp, target)
    row = {k: meta.get(k, "") for k in INDEX_FIELDS}
    row.update(source=source, rows=vol.shape[1], cols=vol.shape[2], n_slices=vol.shape[0],
               pixel_spacing=json.dumps(ps.tolist()), slice_spacing=series.slice_spacing,
               dtype=str(vol.dtype), npz_sha256=sha256(target), npz_file=name)
    rows = [r for r in read_index(source, root) if r["series_key"] != key]
    rows.append(row)
    _write_index(source, rows, root)
    _atomic_json(d / "receipt.json", dict(schema="exp135-gen-v1", source=source,
                                          series=len(rows), index_sha256=sha256(d / "index.csv"),
                                          raw_archives_deleted=False))
    return row


def load_series(row: dict, source: str, root: Path = ROOT) -> Series:
    path = gen_dir(source, root) / row["npz_file"]
    if sha256(path) != row["npz_sha256"]:
        raise ValueError(f"GENERAL hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        return Series(dict(row), z["vol"], z["ipp"], z["iop"], z["pixel_spacing"],
                      float(z["slice_spacing"]), z["per_slice_spacing"])


def pick_strict(rows, plane, fluid, used):
    """Like raptor_corpus.pick_series but without the cross-contrast fallback (fluid -1 = any)."""
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        cands = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
    return cands[0] if cands else None


def render_study(rows: list[dict], source: str, root: Path = ROOT):
    """Replay build_study's pick/window/crop arithmetic from GENERAL."""
    spec = CONTRACTS[CONTRACT]
    img, crop_mm = spec["img"], spec["crop_mm"]
    span_lo, span_hi = spec["span"]
    depth = sum(k for _, _, k in spec["slots"])
    out = np.zeros((depth, img, img), np.uint8)
    mask = np.zeros(depth, np.uint8)
    picks = []
    pick_rows = [dict(r, Anatomical_Plane=r["plane"], Fluid_Sensitive=r["fluid_sensitive"],
                      SeriesInstanceUID=r["series_key"]) for r in rows]
    # B fix after the EXP137 MRNet QA: for sources whose contrast is KNOWN, a series fills only its own contrast group
    # (the competition fallback would put e.g. MRNet coronal T1 into the fat-suppressed coronal group).
    known = bool(rows) and all(str(r.get("rule_hit", "")).startswith("known contrast") for r in rows)
    used = set()
    offset = 0
    import cv2
    raptor_corpus._WORKER["cv2"] = cv2
    cv2.setNumThreads(1)
    for plane, fluid, k in spec["slots"]:
        selected = pick_strict(pick_rows, plane, fluid, used) if known else pick_series(pick_rows, plane, fluid, used)
        if selected is None:
            picks.append(None)
            offset += k
            continue
        used.add(selected["series_key"])
        ser = load_series(selected, source, root)
        n = len(ser.vol)
        lo, hi = int(n * span_lo), max(int(n * span_hi) - 1, int(n * span_lo))
        chosen = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, int)
        arrs = ser.vol[chosen].astype(np.float32)
        pss = ser.per_slice_spacing[chosen]
        valid_ps = pss[np.isfinite(pss) & (pss > 0)]
        med_ps = float(np.median(valid_ps)) if len(valid_ps) else .5
        loq, hiq = np.percentile(np.concatenate([a.ravel() for a in arrs]), [2., 98.])
        for j, (a, ps) in enumerate(zip(arrs, pss)):
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            if str(selected.get("spacing_unknown", "")).lower() in ("true", "1"):
                aw = cv2.resize(aw, (img, img), interpolation=cv2.INTER_AREA)
            else:
                aw = _crop_resize(aw, float(ps) if np.isfinite(ps) and ps > 0 else med_ps, crop_mm, img)
            out[offset + j] = (aw * 255).astype(np.uint8)
        mask[offset:offset + k] = 1
        picks.append(dict(series=selected["series_key"], files=n, slices=chosen.tolist(),
                          rendered=k, window=[float(loq), float(hiq)], median_spacing=med_ps,
                          rule_hit=selected["rule_hit"], spacing_unknown=selected["spacing_unknown"]))
        offset += k
    return out, mask, picks


def build_production(source: str, root: Path = ROOT, shard_size: int = 16) -> dict:
    require_space(root)
    rows = read_index(source, root)
    if not rows:
        raise ValueError("GENERAL index is empty")
    by_study = defaultdict(list)
    for row in rows:
        by_study[row["study_key"]].append(row)
    d = prod_dir(source, root)
    d.mkdir(parents=True, exist_ok=True)
    manifest_path = d / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if old and old.get("shard_size") != shard_size:
        raise ValueError("existing production shard size differs; keep the original --shard-size")
    old_ids = np.load(d / "ids.npy", allow_pickle=False).astype(str).tolist() if (d / "ids.npy").exists() else []
    ids = [u for u in old_ids if u in by_study] + sorted(set(by_study) - set(old_ids))
    if len(ids) != len(by_study):
        raise ValueError("GENERAL study disappeared; do not silently replace a production corpus")
    masks = np.zeros((len(ids), 64), np.uint8)
    old_picks = {}
    if (d / "picks.jsonl").exists():
        with (d / "picks.jsonl").open(encoding="utf-8") as f:
            old_picks = {r["uid"]: r for line in f if (r := json.loads(line))}
    picks_out = []
    shards = []
    old_shards = {s["file"]: s for s in old.get("shards", [])}
    for start in range(0, len(ids), shard_size):
        require_space(root)
        block = ids[start:start + shard_size]
        shard_no = start // shard_size
        name = f"vols_shard{shard_no:04d}.npy"
        previous = old_shards.get(name)
        unchanged = (previous and previous["start"] == start and previous["stop"] == start + len(block)
                     and (d / name).is_file() and sha256(d / name) == previous["sha256"]
                     and all(uid in old_picks and old_picks[uid].get("gen_sha256") ==
                             [r["npz_sha256"] for r in by_study[uid]] for uid in block))
        if unchanged:
            shard = previous
            for uid in block:
                p = old_picks[uid]
                masks[ids.index(uid)] = p["mask"]
                picks_out.append(p)
        else:
            tmp = d / (name + ".partial")
            arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                            shape=(len(block), 64, 336, 336))
            for j, uid in enumerate(block):
                v, m, picks = render_study(by_study[uid], source, root)
                arr[j] = v
                masks[start + j] = m
                picks_out.append(dict(uid=uid, shard=shard_no, row=j, slots=picks, mask=m.tolist(),
                                      gen_sha256=[r["npz_sha256"] for r in by_study[uid]]))
            arr.flush(); del arr
            os.replace(tmp, d / name)
            shard = dict(shard=shard_no, start=start, stop=start + len(block), file=name,
                         done=True, bytes=(d / name).stat().st_size, sha256=sha256(d / name))
        shards.append(shard)
    np.save(d / "ids.npy", np.asarray(ids, dtype=f"<U{max(map(len, ids))}"))
    np.save(d / "masks.npy", masks)
    with (d / "picks.jsonl.partial").open("w", encoding="utf-8") as f:
        for p in picks_out:
            f.write(json.dumps(p) + "\n")
    os.replace(d / "picks.jsonl.partial", d / "picks.jsonl")
    c = CONTRACTS[CONTRACT]
    spec = dict(slots=c["slots"], img=c["img"], crop_mm=c["crop_mm"], span=list(c["span"]),
                depth=64, normalization="series percentile 2-98 over picked slices, clip, uint8",
                order="IOP/IPP normal projection, InstanceNumber fallback; stable by path",
                resize="cv2.INTER_AREA on [0,1] float",
                series_selection="plane match, prefer Fluid_Sensitive flag, else first csv row; a series is used once",
                mask="filled slots", spacing_unknown="full-field resize")
    manifest = dict(contract=CONTRACT, spec=spec, builder="EXP135 extcache v1", studies=len(ids),
                    shard_size=shard_size, dtype="uint8", shape_total=[len(ids), 64, 336, 336],
                    source=source, shards=shards, ids_sha256=sha256(d / "ids.npy"),
                    masks_sha256=sha256(d / "masks.npy"), picks_sha256=sha256(d / "picks.jsonl"),
                    gen_index_sha256=sha256(gen_dir(source, root) / "index.csv"), finished=True)
    _atomic_json(manifest_path, manifest)
    if len(Corpus(d)) != len(ids):
        raise RuntimeError("Corpus could not read production output")
    return manifest


def verify(source: str, root: Path = ROOT) -> dict:
    rows = read_index(source, root)
    bad = []
    for row in rows:
        path = gen_dir(source, root) / row["npz_file"]
        if not path.is_file() or sha256(path) != row["npz_sha256"]:
            bad.append(f"GENERAL {row['series_key']}")
    d = prod_dir(source, root)
    studies = 0
    if (d / "manifest.json").exists():
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        corpus = Corpus(d)
        by_study = defaultdict(list)
        for row in rows:
            by_study[row["study_key"]].append(row)
        for shard in manifest["shards"]:
            p = d / shard["file"]
            if not p.is_file() or sha256(p) != shard["sha256"]:
                bad.append(f"PRODUCTION shard {shard['file']}")
        for uid in corpus.ids:
            if uid not in by_study:
                bad.append(f"missing GENERAL study {uid}")
                continue
            v, m, _ = render_study(by_study[uid], source, root)
            if not np.array_equal(v, corpus.volume(uid)) or not np.array_equal(m, corpus.mask(uid)):
                bad.append(f"PRODUCTION derivation {uid}")
            studies += 1
    gen_bytes = sum(p.stat().st_size for p in gen_dir(source, root).glob("*.npz"))
    prod_bytes = sum(p.stat().st_size for p in d.glob("vols_shard*.npy")) if d.exists() else 0
    # OAI baseline packages contain one series each. A monolithic fastMRI archive
    # is never advertised as removable based on a partial stream conversion.
    candidates = sorted(set(r["source_archive_member"].split("::")[0]
                            for r in rows if source == "oai" and r["source_archive_member"])) if not bad else []
    return dict(source=source, general_series=len(rows), production_studies=studies,
                general_bytes=gen_bytes, production_bytes=prod_bytes, errors=bad,
                safe_to_delete_candidates=candidates,
                deletion_requires_user_approval=True)
