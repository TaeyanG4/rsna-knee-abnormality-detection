"""Build a Raptor-contract corpus (fixed slot stacks) straight from the competition ZIP.

Reads DICOM members through the archive index KneeOps already built, so nothing is
extracted to disk. Output: sharded uint8 arrays (N, depth, img, img), per-slice masks,
ids, a manifest with the contract and SHA256 of every shard, and a JSONL provenance of the
series/slices chosen per study. Gold58 and public-test studies are placed first so parity
checks can start before the full build finishes. Re-running resumes at the first missing
shard; a finished shard is never rewritten.

Usage (GPU python, from the workspace root):
  python training/src/raptor_corpus.py --contract raptor64_336_c140_s02 --workers 8
  python training/src/raptor_corpus.py --contract raptor64_336_c140_s02 --verify
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raptor_common import CACHE, COMP, CONTRACTS, LAB, ROOT, UID, environment, sha256_file  # noqa: E402

BUILDER_VERSION = "raptor_corpus.py v1 (2026-10-02)"
_WORKER = {}


def archive_paths():
    """ZIP path and index sqlite recorded by the dashboard's data library."""
    db = ROOT / ".local" / "kneeops" / "app.sqlite"
    con = sqlite3.connect(db)
    raw = con.execute("SELECT value FROM meta WHERE key='data_library'").fetchone()
    con.close()
    if not raw:
        raise SystemExit("data_library meta missing; index the full ZIP in the app first (학습 데이터 → 로컬 원본 데이터)")
    archives = json.loads(raw[0])["archives"]
    arc = archives[0]
    zip_path = Path(arc["path"].replace("workspace://", str(ROOT) + "/"))
    index = Path(arc["index"])
    if not zip_path.is_file() or not index.is_file():
        raise SystemExit(f"archive or index missing: {zip_path} / {index}")
    return zip_path, index, arc


class IndexedZip(zipfile.ZipFile):
    """Skip the 819k-entry central directory; members come from the index with explicit ZipInfo."""

    def _RealGetContents(self):
        self.start_dir = os.fstat(self.fp.fileno()).st_size


def zip_info(path, info_json):
    info = zipfile.ZipInfo(path)
    for key, value in json.loads(info_json).items():
        setattr(info, key, value)
    return info


def series_records():
    import pandas as pd
    out = {}
    for split, name in (("train", "train_series.csv"), ("test", "test_series.csv")):
        df = pd.read_csv(COMP / name, dtype={UID: str, "SeriesInstanceUID": str})
        for uid, g in df.groupby(UID, sort=False):
            out[(split, uid)] = g.to_dict("records")
    return out


def pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _init_worker(zip_path, index_path, spec):
    import cv2
    import pydicom
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    _WORKER["zip"] = IndexedZip(zip_path)
    _WORKER["con"] = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    _WORKER["spec"] = spec
    _WORKER["cv2"] = cv2
    _WORKER["pydicom"] = pydicom
    _WORKER["lut"] = apply_modality_lut
    cv2.setNumThreads(1)


def _member_bytes(path, info_json):
    with _WORKER["zip"].open(zip_info(path, info_json)) as f:
        return f.read()


def _order_series(members):
    """[(path, info, pixel_spacing)] sorted by physical position (IOP/IPP), InstanceNumber fallback."""
    pydicom = _WORKER["pydicom"]
    recs, spacings = [], []
    for path, info in members:
        try:
            h = pydicom.dcmread(io.BytesIO(_member_bytes(path, info)), stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
        except Exception:
            pos, ps = 0.0, 0.5
        recs.append((pos, path, info, ps)); spacings.append(ps)
    recs.sort(key=lambda x: x[0])  # stable: ties keep path order
    med = float(np.median(spacings)) if spacings else 0.5
    return [(p, i, ps) for _, p, i, ps in recs], med


def _read_pixels(path, info):
    pydicom = _WORKER["pydicom"]
    d = pydicom.dcmread(io.BytesIO(_member_bytes(path, info)))
    a = _WORKER["lut"](d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def _crop_resize(a, ps, crop_mm, img):
    cv2 = _WORKER["cv2"]
    h, w = a.shape
    cpx = int(round(crop_mm / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w))
    y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (img, img), interpolation=cv2.INTER_AREA)


def build_study(task):
    split, uid, rows = task
    spec = _WORKER["spec"]
    depth, img, crop_mm = spec["depth"], spec["img"], spec["crop_mm"]
    span_lo, span_hi = spec["span"]
    con = _WORKER["con"]
    members = con.execute("SELECT path, series, info FROM files WHERE split=? AND uid=? ORDER BY path", (split, uid)).fetchall()
    by_series = {}
    for path, series, info in members:
        by_series.setdefault(series, []).append((path, info))
    vol = np.zeros((depth, img, img), np.uint8)
    idx, used, picks = 0, set(), []
    for plane, fluid, k in spec["slots"]:
        r = pick_series(rows, plane, fluid, used)
        if r is None:
            picks.append(None); idx += k; continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = _order_series(by_series.get(r["SeriesInstanceUID"], []))
        if not files:
            picks.append({"series": r["SeriesInstanceUID"], "files": 0}); idx += k; continue
        n = len(files)
        lo, hi = int(n * span_lo), int(n * span_hi) - 1
        hi = max(hi, lo)
        chosen = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, int)
        arrs, pss = [], []
        for p in chosen:
            path, info, ps = files[min(int(p), n - 1)]
            try:
                arrs.append(_read_pixels(path, info)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        rendered = 0
        for a, ps in zip(arrs, pss):
            if idx >= depth:
                break
            if a is None:
                idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = _crop_resize(aw, ps if ps > 0 else med_ps, crop_mm, img)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1; rendered += 1
        picks.append({"series": r["SeriesInstanceUID"], "files": n, "slices": [int(x) for x in chosen],
                      "rendered": rendered, "window": [float(loq), float(hiq)], "median_spacing": med_ps})
        if idx >= depth:
            break
    mask = (vol.reshape(depth, -1).sum(1) > 0).astype(np.uint8)
    return uid, split, vol, mask, picks


def study_order():
    """gold58 + public test first, then the rest of train in csv order."""
    import pandas as pd
    tr = pd.read_csv(COMP / "train.csv", dtype={UID: str})
    gold = tr[tr[LAB].notna().all(axis=1)][UID].tolist()
    te = pd.read_csv(COMP / "test.csv", dtype={UID: str})[UID].tolist()
    rest = [u for u in tr[UID] if u not in set(gold)]
    return [("train", u) for u in gold] + [("test", u) for u in te] + [("train", u) for u in rest]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contract", default="raptor64_336_c140_s02", choices=sorted(CONTRACTS))
    ap.add_argument("--out", default=None, help="output directory (default storage/cache/derived/<CONTRACT>)")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--shard-size", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="build only the first N studies (smoke)")
    ap.add_argument("--verify", action="store_true", help="re-hash finished shards against the manifest and exit")
    a = ap.parse_args()

    c = CONTRACTS[a.contract]
    spec = dict(slots=c["slots"], img=c["img"], crop_mm=c["crop_mm"], span=list(c["span"]),
                depth=sum(s[2] for s in c["slots"]), normalization="series percentile 2-98 over picked slices, clip, uint8",
                order="IOP/IPP normal projection, InstanceNumber fallback; stable by path", resize="cv2.INTER_AREA on [0,1] float",
                series_selection="plane match, prefer Fluid_Sensitive flag, else first csv row; a series is used once")
    out = Path(a.out) if a.out else CACHE / a.contract.upper()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None

    if a.verify:
        if not manifest:
            raise SystemExit("no manifest")
        bad = 0
        for s in manifest["shards"]:
            if not s.get("done"):
                print("not built:", s["file"]); continue
            got = sha256_file(out / s["file"])
            ok = got == s["sha256"]; bad += not ok
            print(("ok   " if ok else "DIFF ") + s["file"], got[:16])
        raise SystemExit(1 if bad else 0)

    zip_path, index_path, arc = archive_paths()
    order = study_order()
    if a.limit:
        order = order[:a.limit]
    n = len(order)
    shards = [{"shard": k, "start": s, "stop": min(s + a.shard_size, n), "file": f"vols_shard{k:02d}.npy"}
              for k, s in enumerate(range(0, n, a.shard_size))]
    if manifest and (manifest["contract"] != a.contract or manifest["studies"] != n or manifest["shard_size"] != a.shard_size):
        raise SystemExit("existing manifest has a different contract/size; choose another --out")
    done = {s["file"]: s for s in (manifest or {}).get("shards", []) if s.get("done")}
    recs = series_records()
    ids = [u for _, u in order]
    masks = np.load(out / "masks.npy") if (out / "masks.npy").exists() and manifest else np.zeros((n, spec["depth"]), np.uint8)
    if masks.shape != (n, spec["depth"]):
        masks = np.zeros((n, spec["depth"]), np.uint8)
    np.save(out / "ids.npy", np.array(ids))
    manifest = {"contract": a.contract, "spec": spec, "builder": BUILDER_VERSION, "studies": n, "shard_size": a.shard_size,
                "dtype": "uint8", "shape_total": [n, spec["depth"], spec["img"], spec["img"]],
                "archive": {"path": arc["path"], "zip_bytes": arc["zip_bytes"], "directory_sha256": arc["directory_sha256"]},
                "order": "gold58, public test, remaining train (train.csv order)", "environment": environment(),
                "shards": [dict(s, **done.get(s["file"], {})) for s in shards]}
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"corpus {a.contract} -> {out} | {n} studies | {len(shards)} shards | {len(done)} already done | workers {a.workers}", flush=True)

    picks_path = out / "picks.jsonl"
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(str(zip_path), str(index_path), spec)) as pool, \
            open(picks_path, "a", encoding="utf-8") as picks_file:
        for s in manifest["shards"]:
            if s.get("done"):
                continue
            t0 = time.time()
            tasks = [(split, uid, recs.get((split, uid), [])) for split, uid in order[s["start"]:s["stop"]]]
            tmp = out / (s["file"] + ".partial")
            arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(tasks), spec["depth"], spec["img"], spec["img"]))
            pos = {uid: i for i, (_, uid, _) in enumerate(tasks)}
            empty = 0
            for i, (uid, split, vol, mask, picks) in enumerate(pool.imap_unordered(build_study, tasks, chunksize=2)):
                r = pos[uid]
                arr[r] = vol; masks[s["start"] + r] = mask
                empty += int(mask.sum() == 0)
                picks_file.write(json.dumps({"uid": uid, "split": split, "shard": s["shard"], "row": r, "slots": picks}) + "\n")
                if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                    print(f"  shard {s['shard']:02d}: {i + 1}/{len(tasks)} studies, {time.time() - t0:.0f}s", flush=True)
            arr.flush(); del arr
            os.replace(tmp, out / s["file"])
            picks_file.flush()
            np.save(out / "masks.npy", masks)
            s.update(done=True, bytes=(out / s["file"]).stat().st_size, sha256=sha256_file(out / s["file"]),
                     seconds=round(time.time() - t0, 1), empty_studies=empty)
            manifest["shards"] = manifest["shards"]
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            print(f"shard {s['shard']:02d} done: {len(tasks)} studies in {s['seconds']}s, empty {empty}, sha {s['sha256'][:12]}", flush=True)
    np.save(out / "masks.npy", masks)
    with open(out / "uids.csv", "w", encoding="utf-8", newline="") as f:
        f.write("StudyInstanceUID,index" + chr(10) + "".join(f"{u},{i}" + chr(10) for i, u in enumerate(ids)))
    manifest["finished"] = all(s.get("done") for s in manifest["shards"])
    # Keys the dashboard cache inspector reads (mask_file/uid_file/mask_sha256).
    manifest.update(mask_file="masks.npy", uid_file="uids.csv", mask_sha256=sha256_file(out / "masks.npy"),
                    masks_sha256=sha256_file(out / "masks.npy"), ids_sha256=sha256_file(out / "ids.npy"))
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("finished" if manifest["finished"] else "partial", out, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
