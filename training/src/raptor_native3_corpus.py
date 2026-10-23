"""Build the native-adjacent triplet cache RAPTOR64N3_336_C140_S02 (H-B-25; contract described in raptor_native3.py).

Same source, same study order, same shards (400 studies) and the same per-study math as raptor_corpus.py v1 (its worker set-up,
pick_series, _crop_resize and study list are imported, the file is not changed); per pick it additionally renders the native
previous / next slice with the slot's window. Output: uint8 (N, 64, 3, 336, 336) shards, masks.npy (N, 64) from the centre
planes, ids.npy, picks.jsonl (base pick record + native neighbour indices and position offsets) and a manifest with shard sha256,
the code hashes and a per-study check that the centre planes and the mask are byte-identical to the 64-slot cache.

  python training/src/raptor_native3_corpus.py --estimate                  # disk estimate vs free space, no build
  python training/src/raptor_native3_corpus.py --workers 3                 # build / resume into storage/cache/derived/RAPTOR64N3_336_C140_S02
  python training/src/raptor_native3_corpus.py --out DIR --limit 6         # smoke into another directory
  python training/src/raptor_native3_corpus.py --verify                    # re-hash finished shards

At most 3 worker processes, all at below-normal priority (the local GPU training loader owns the rest of the CPU). A build refuses
to start when the free space on the output drive is below 2.5x the bytes still to be written. Any centre/mask difference to the
64-slot cache stops the build after the shard in which it occurs (the contract would no longer be "only the outer channels change").
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import raptor_corpus as rc  # noqa: E402  (unchanged builder: worker set-up, pick_series, _crop_resize, study_order)
import raptor_native3 as n3  # noqa: E402
from raptor_common import CACHE, CONTRACTS, Corpus, environment, sha256_file  # noqa: E402

BUILDER_VERSION = "raptor_native3_corpus.py v1 (2026-10-07)"
MAX_WORKERS = 3
FREE_FACTOR = 2.5
BASE_DIR = CACHE / n3.BASE_CONTRACT.upper()
OUT_DIR = CACHE / n3.CORPUS_DIRNAME
_W = {}


def below_normal_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        return True
    except Exception:
        return False


def base_spec():
    """The spec dict raptor_corpus.main() writes for raptor64_336_c140_s02 (same construction)."""
    c = CONTRACTS[n3.BASE_CONTRACT]
    return dict(slots=c["slots"], img=c["img"], crop_mm=c["crop_mm"], span=list(c["span"]),
                depth=sum(s[2] for s in c["slots"]), normalization="series percentile 2-98 over picked slices, clip, uint8",
                order="IOP/IPP normal projection, InstanceNumber fallback; stable by path", resize="cv2.INTER_AREA on [0,1] float",
                series_selection="plane match, prefer Fluid_Sensitive flag, else first csv row; a series is used once")


def study_bytes(spec):
    return int(spec["depth"]) * n3.PLANES * int(spec["img"]) ** 2


def disk_estimate(n_studies, spec, out_dir):
    """Bytes for n studies (npy headers ignored) and free bytes on the output drive."""
    need = n_studies * study_bytes(spec)
    probe = Path(out_dir)
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    return dict(studies=n_studies, bytes_per_study=study_bytes(spec), estimate_bytes=need, estimate_gb=round(need / 1e9, 2),
                free_bytes=free, free_gb=round(free / 1e9, 1), required_free_gb=round(FREE_FACTOR * need / 1e9, 1),
                enough=free >= FREE_FACTOR * need)


# ------------------------------------------------------------------ worker
def _init_worker(zip_path, index_path, spec, base_dir):
    below_normal_priority()
    rc._init_worker(zip_path, index_path, spec)
    _W["base"] = Corpus(Path(base_dir)) if base_dir else None


def _order_series_pos(members):
    """raptor_corpus._order_series (same header parse, same failure fallback, same stable sort) that also keeps positions and bytes."""
    pydicom = rc._WORKER["pydicom"]
    recs, spacings = [], []
    for path, info in members:
        raw = None
        try:
            raw = rc._member_bytes(path, info)
            h = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
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
        recs.append((pos, path, info, ps, raw)); spacings.append(ps)
    recs.sort(key=lambda x: x[0])  # stable: ties keep path order (as raptor_corpus)
    med = float(np.median(spacings)) if spacings else 0.5
    return [(p, i, ps) for _, p, i, ps, _ in recs], med, [x[0] for x in recs], [x[4] for x in recs]


def _pixels(raw, path, info):
    """raptor_corpus._read_pixels on bytes already read (re-read from the archive when the header pass failed to read them)."""
    pydicom = rc._WORKER["pydicom"]
    d = pydicom.dcmread(io.BytesIO(raw if raw is not None else rc._member_bytes(path, info)))
    a = rc._WORKER["lut"](d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def _render(a, ps, med_ps, loq, hiq, crop_mm, img):
    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
    aw = rc._crop_resize(aw, ps if ps > 0 else med_ps, crop_mm, img)
    return (aw * 255).astype(np.uint8)


def build_study(task):
    """raptor_corpus.build_study with a (depth, 3, img, img) volume: plane 1 = the base render, planes 0/2 = native neighbours."""
    split, uid, rows = task
    spec = rc._WORKER["spec"]
    depth, img, crop_mm = spec["depth"], spec["img"], spec["crop_mm"]
    span_lo, span_hi = spec["span"]
    con = rc._WORKER["con"]
    members = con.execute("SELECT path, series, info FROM files WHERE split=? AND uid=? ORDER BY path", (split, uid)).fetchall()
    by_series = {}
    for path, series, info in members:
        by_series.setdefault(series, []).append((path, info))
    vol = np.zeros((depth, n3.PLANES, img, img), np.uint8)
    idx, used, picks = 0, set(), []
    stats = dict(edge_replications=0, neighbour_decode_failures=0, neighbour_is_pick=0, neighbours_rendered=0, zero_neighbour_planes=0)
    for plane, fluid, k in spec["slots"]:
        r = rc.pick_series(rows, plane, fluid, used)
        if r is None:
            picks.append(None); idx += k; continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps, positions, raws = _order_series_pos(by_series.get(r["SeriesInstanceUID"], []))
        if not files:
            picks.append({"series": r["SeriesInstanceUID"], "files": 0}); idx += k; continue
        n = len(files)
        lo, hi = int(n * span_lo), int(n * span_hi) - 1
        hi = max(hi, lo)
        chosen = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, int)
        decoded = {}

        def pixels(q):
            if q not in decoded:
                path, info, _ = files[q]
                try:
                    decoded[q] = _pixels(raws[q], path, info)
                except Exception:
                    decoded[q] = None
            return decoded[q]

        arrs, pss = [], []
        for p in chosen:
            q = min(int(p), n - 1)
            a = pixels(q)
            arrs.append(a); pss.append(files[q][2] if a is not None else med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        chosen_set = {min(int(p), n - 1) for p in chosen}
        rendered, prev_idx, next_idx, dprev, dnext = 0, [], [], [], []
        for p, a, ps in zip(chosen, arrs, pss):
            if idx >= depth:
                break
            q = min(int(p), n - 1)
            nbs = n3.neighbour_indices(q, n)
            prev_idx.append(int(nbs[0])); next_idx.append(int(nbs[1]))
            dprev.append(round(positions[nbs[0]] - positions[q], 4)); dnext.append(round(positions[nbs[1]] - positions[q], 4))
            if a is None:
                idx += 1; continue
            centre = _render(a, ps, med_ps, loq, hiq, crop_mm, img)
            vol[idx, 1] = centre
            for channel, nb in zip((0, 2), nbs):
                if nb == q:
                    vol[idx, channel] = centre; stats["edge_replications"] += 1
                    continue
                b = pixels(nb)
                if b is None:
                    vol[idx, channel] = centre; stats["neighbour_decode_failures"] += 1
                    continue
                vol[idx, channel] = _render(b, files[nb][2], med_ps, loq, hiq, crop_mm, img)
                stats["neighbours_rendered"] += 1; stats["neighbour_is_pick"] += int(nb in chosen_set)
                stats["zero_neighbour_planes"] += int(not vol[idx, channel].any())
            idx += 1; rendered += 1
        picks.append({"series": r["SeriesInstanceUID"], "files": n, "slices": [int(x) for x in chosen], "rendered": rendered,
                      "window": [float(loq), float(hiq)], "median_spacing": med_ps, "prev": prev_idx, "next": next_idx,
                      "dpos_prev": dprev, "dpos_next": dnext})
        if idx >= depth:
            break
    mask = (vol[:, 1].reshape(depth, -1).sum(1) > 0).astype(np.uint8)
    check = None
    base = _W.get("base")
    if base is not None:
        check = dict(in_base=uid in base)
        if check["in_base"]:
            check.update(centre_equal=bool(np.array_equal(vol[:, 1], base.volume(uid))), mask_equal=bool(np.array_equal(mask, base.mask(uid))))
    return uid, split, vol, mask, picks, stats, check


# ------------------------------------------------------------------ main
def manifest_registry():
    return {"module": "training/src/raptor_native3.py", "module_sha256": sha256_file(Path(n3.__file__)),
            "builder": "training/src/raptor_native3_corpus.py", "builder_sha256": sha256_file(Path(__file__)),
            "base_builder": "training/src/raptor_corpus.py", "base_builder_sha256": sha256_file(Path(rc.__file__)),
            "raptor_common_sha256": sha256_file(_HERE / "raptor_common.py")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help=f"output directory (default storage/cache/derived/{n3.CORPUS_DIRNAME})")
    ap.add_argument("--base", default=str(BASE_DIR), help="the 64-slot cache the centre planes must equal")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--shard-size", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="build only the first N studies (smoke)")
    ap.add_argument("--verify", action="store_true", help="re-hash finished shards against the manifest and exit")
    ap.add_argument("--estimate", action="store_true", help="print the disk estimate and exit")
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS} while the local GPU trains")
    below_normal_priority()
    spec = n3.spec_for(base_spec())
    out = Path(a.out) if a.out else OUT_DIR
    base_dir = Path(a.base)
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
            print(("ok   " if ok else "DIFF ") + s["file"], got[:16], flush=True)
        raise SystemExit(1 if bad else 0)

    base = Corpus(base_dir)
    if base.contract != n3.BASE_CONTRACT or base.manifest["spec"] != json.loads(json.dumps(base_spec())):
        raise SystemExit("base cache is not the raptor64_336_c140_s02 contract written by raptor_corpus.py")
    order = rc.study_order()
    if a.limit:
        order = order[:a.limit]
    n = len(order)
    ids = [u for _, u in order]
    if ids != base.ids.astype(str).tolist()[:n]:
        raise SystemExit("study order differs from the base cache ids")
    shards = [{"shard": k, "start": s, "stop": min(s + a.shard_size, n), "file": f"vols_shard{k:02d}.npy"}
              for k, s in enumerate(range(0, n, a.shard_size))]
    if manifest and (manifest["contract"] != n3.CONTRACT or manifest["studies"] != n or manifest["shard_size"] != a.shard_size):
        raise SystemExit("existing manifest has a different contract/size; choose another --out")
    done = {s["file"]: s for s in (manifest or {}).get("shards", []) if s.get("done")}
    todo = sum(s["stop"] - s["start"] for s in shards if s["file"] not in done)
    est = disk_estimate(todo, spec, out)
    est_total = disk_estimate(n, spec, out)
    print(json.dumps(dict(total=est_total, remaining=est), indent=1), flush=True)
    if a.estimate:
        return est_total
    if not est["enough"]:
        raise SystemExit(f"free space {est['free_gb']} GB < {FREE_FACTOR} x {est['estimate_gb']} GB still to write; not building")
    out.mkdir(parents=True, exist_ok=True)
    zip_path, index_path, arc = rc.archive_paths()
    if arc.get("directory_sha256") != base.manifest["archive"]["directory_sha256"]:
        raise SystemExit("source archive differs from the one the base cache was built from")
    recs = rc.series_records()
    masks = np.load(out / "masks.npy") if (out / "masks.npy").exists() and manifest else np.zeros((n, spec["depth"]), np.uint8)
    if masks.shape != (n, spec["depth"]):
        masks = np.zeros((n, spec["depth"]), np.uint8)
    np.save(out / "ids.npy", np.array(ids))
    manifest = {"contract": n3.CONTRACT, "spec": spec, "builder": BUILDER_VERSION, "base_builder": rc.BUILDER_VERSION, "studies": n,
                "shard_size": a.shard_size, "dtype": "uint8", "shape_total": [n, spec["depth"], n3.PLANES, spec["img"], spec["img"]],
                "archive": {"path": arc["path"], "zip_bytes": arc["zip_bytes"], "directory_sha256": arc["directory_sha256"]},
                "order": "gold58, public test, remaining train (train.csv order)", "environment": environment(),
                "base_corpus": {"path": str(base_dir), "manifest_sha256": sha256_file(base_dir / "manifest.json"),
                                "ids_sha256": sha256_file(base_dir / "ids.npy"), "masks_sha256": sha256_file(base_dir / "masks.npy")},
                "code_registry": manifest_registry(), "disk_estimate": est_total, "workers": a.workers, "priority": "below normal",
                "shards": [dict(s, **done.get(s["file"], {})) for s in shards]}
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"corpus {n3.CONTRACT} -> {out} | {n} studies | {len(shards)} shards | {len(done)} already done | workers {a.workers}", flush=True)
    t_build = time.time()
    picks_path = out / "picks.jsonl"
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(str(zip_path), str(index_path), spec, str(base_dir))) as pool:
        for s in manifest["shards"]:
            if s.get("done"):
                continue
            t0 = time.time()
            tasks = [(split, uid, recs.get((split, uid), [])) for split, uid in order[s["start"]:s["stop"]]]
            tmp = out / (s["file"] + ".partial")
            arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                            shape=(len(tasks), spec["depth"], n3.PLANES, spec["img"], spec["img"]))
            pos = {uid: i for i, (_, uid, _) in enumerate(tasks)}
            lines, totals, empty, mismatches = [None] * len(tasks), {}, 0, []
            for i, (uid, split, vol, mask, picks, stats, check) in enumerate(pool.imap_unordered(build_study, tasks, chunksize=2)):
                r = pos[uid]
                arr[r] = vol; masks[s["start"] + r] = mask
                empty += int(mask.sum() == 0)
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0) + value
                if not (check and check.get("in_base") and check["centre_equal"] and check["mask_equal"]):
                    mismatches.append(dict(uid=uid, check=check))
                lines[r] = json.dumps({"uid": uid, "split": split, "shard": s["shard"], "row": r, "slots": picks, "stats": stats})
                if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                    print(f"  shard {s['shard']:02d}: {i + 1}/{len(tasks)} studies, {time.time() - t0:.0f}s", flush=True)
            arr.flush(); del arr
            os.replace(tmp, out / s["file"])
            with open(picks_path, "a", encoding="utf-8") as f:
                f.write("".join(line + "\n" for line in lines))
            np.save(out / "masks.npy", masks)
            s.update(done=True, bytes=(out / s["file"]).stat().st_size, sha256=sha256_file(out / s["file"]), seconds=round(time.time() - t0, 1),
                     empty_studies=empty, neighbour_stats=totals, centre_mask_mismatches=len(mismatches), mismatch_examples=mismatches[:5])
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
            print(f"shard {s['shard']:02d} done: {len(tasks)} studies in {s['seconds']}s, empty {empty}, centre/mask mismatches "
                  f"{len(mismatches)}, sha {s['sha256'][:12]}, {totals}", flush=True)
            if mismatches:
                raise SystemExit(f"centre planes / mask differ from the base cache in shard {s['shard']}: {mismatches[:3]}")
    np.save(out / "masks.npy", masks)
    with open(out / "uids.csv", "w", encoding="utf-8", newline="") as f:
        f.write("StudyInstanceUID,index" + chr(10) + "".join(f"{u},{i}" + chr(10) for i, u in enumerate(ids)))
    manifest["finished"] = all(s.get("done") for s in manifest["shards"])
    manifest["build_seconds_this_run"] = round(time.time() - t_build, 1)
    manifest["masks_equal_base"] = bool(np.array_equal(masks, base.masks[:n]))
    manifest["centre_mask_mismatches"] = int(sum(s.get("centre_mask_mismatches", 0) for s in manifest["shards"]))
    manifest.update(mask_file="masks.npy", uid_file="uids.csv", mask_sha256=sha256_file(out / "masks.npy"),
                    masks_sha256=sha256_file(out / "masks.npy"), ids_sha256=sha256_file(out / "ids.npy"))
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("finished" if manifest["finished"] else "partial", out, f"masks_equal_base={manifest['masks_equal_base']}", flush=True)
    return manifest


if __name__ == "__main__":
    mp.freeze_support()
    main()
