"""H-B-147 OAI native448 adapter v1.0.1 (2026-10-15).

Changelog v1.0.0: audited GENERAL -> genuine native448; exact336 replay,
frozen masks/picks, authenticated resumable Corpus output. CPU only; no labels.
Changelog v1.0.1: optional committed-file progress for supervised hash passes;
all source/output authentication and immutable manifest rules remain enforced.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import hb145_output_io as io
from tools.extcache import common, oai_cache_v2 as oai

VERSION = "1.0.1"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".local/scratch/hb147-lenses/H-B-147/oai448"
SOURCE = ROOT / "storage/cache/derived/EXT_OAI_RAPTOR64_336_C140_v2"
PANEL = ROOT / "artifacts/experiments/OAI_MOAKS_PANEL_V1"
CONTRACT = "hb147_raptor64_448_c140_s02_v1"
REGISTRATION_SHA = "60fbe188c458cb01ceb1f8734303be21e5e68db4863b3e091da356119e39007f"
AUDIT_SHA = "423c8c4f6fcc87a1cd441535a9e31c5d05719f9192da18a80000be25a4ab6320"
SOURCE_SHA = "c503168d7d4ba2edd45a6a3ce0b138afa8dcfc079e0b5d12a891a25489e43fa9"
MIN_FREE = 10 * (1 << 30)
CODE = ("tools/hb147_oai_native448_v1.py", "tools/hb145_output_io.py",
        "tools/extcache/common.py", "tools/extcache/oai_cache_v2.py",
        "training/src/raptor_common.py", "training/src/raptor_corpus.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return common.sha256(Path(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def selected_ids(ids, panel):
    """Only a nonempty, unique, canonical-order subset of the frozen panel."""
    panel = list(map(str, panel))
    selected = panel if ids is None else list(map(str, ids))
    require(selected and len(selected) == len(set(selected)), "empty or duplicate IDs")
    wanted = set(selected)
    require(selected == [u for u in panel if u in wanted], "IDs outside panel or noncanonical order")
    return selected


def panel_ids():
    require(sha(PANEL / "registration.json") == REGISTRATION_SHA, "panel registration drift")
    reg = io.read_json(PANEL / "registration.json")
    require(sha(PANEL / "panel.csv") == reg["panel_sha256"], "panel roster drift")
    with (PANEL / "panel.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["study_key"], "panel must contain IDs only")
        ids = [r["study_key"] for r in reader]
    require(len(ids) == reg["n"] == 557 and len(set(ids)) == 557, "panel count mismatch")
    return ids


def context(ids=None):
    """Authenticate metadata only. Individual NPZ and source shards are hashed below."""
    selected = selected_ids(ids, panel_ids())
    reg = io.read_json(PANEL / "registration.json")
    source_files = {}
    for name in ("manifest.json", "ids.npy", "masks.npy", "picks.jsonl"):
        rel = (SOURCE / name).relative_to(ROOT).as_posix()
        source_files[rel] = sha(SOURCE / name)
        require(source_files[rel] == reg["pins"][rel], "frozen source metadata drift")
    require(sha(SOURCE / "manifest.json") == SOURCE_SHA, "source manifest identity drift")
    source = io.read_json(SOURCE / "manifest.json")
    require(source["finished"] and source["contract"] == oai.R64, "invalid source corpus")
    src_ids = np.load(SOURCE / "ids.npy", allow_pickle=False).astype(str)
    masks = np.load(SOURCE / "masks.npy", allow_pickle=False)
    positions = {u: i for i, u in enumerate(src_ids)}
    require(len(positions) == len(src_ids) and masks.shape == (len(src_ids), 64), "source roster/mask shape")
    require(masks.dtype == np.uint8 and np.isin(masks, [0, 1]).all(), "source masks not binary uint8")
    require(set(selected) <= positions.keys(), "panel not in source")
    picks = {}
    with (SOURCE / "picks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            require(rec["uid"] not in picks, "duplicate source picks")
            picks[rec["uid"]] = rec["slots"]
    audit = oai.load_audit()
    require(audit.summary_sha256 == AUDIT_SHA, "audit identity drift")
    rows, general = oai.general_binding(audit, oai.GEN_ROOT)
    plans = {u: oai.plan_study(audit, oai.R64, u, "competition_then_copy") for u in selected}
    groups = sorted({p["group_key"] for plan in plans.values() for p in plan if p["group_key"]})
    npz = [{"file": rows[g]["npz_file"], "sha256": rows[g]["npz_sha256"]} for g in groups]
    require(len({x["file"] for x in npz}) == len(npz), "duplicate GENERAL file")
    shards = [s for s in source["shards"] if any(s["start"] <= positions[u] < s["stop"] for u in selected)]
    require(all(sum(s["start"] <= positions[u] < s["stop"] for s in shards) == 1 for u in selected),
            "source shard coverage")
    binding = dict(version=VERSION, selected_ids_sha256=digest(selected), studies=len(selected),
                   panel_registration_sha256=REGISTRATION_SHA, panel_sha256=reg["panel_sha256"],
                   source_files=source_files, audit=audit.binding(), general=general,
                   general_npz=npz, source_shards=[{k: s[k] for k in ("file", "sha256")} for s in shards],
                   code_sha256={p: sha(ROOT / p) for p in CODE})
    spec = copy.deepcopy(source["spec"])
    spec["img"] = 448
    return dict(ids=selected, masks=masks[[positions[u] for u in selected]], positions=positions,
                source=source, picks=picks, audit=audit, rows=rows, plans=plans,
                binding=binding, spec=spec)


def authenticate_inputs(ctx, *, progress=None):
    for item in ctx["binding"]["general_npz"]:
        path = oai.gen_dir(oai.GEN_ROOT) / item["file"]
        require(sha(path) == item["sha256"], "GENERAL NPZ drift")
        if progress is not None:
            progress()
    for item in ctx["binding"]["source_shards"]:
        require(sha(SOURCE / item["file"]) == item["sha256"], "source shard drift")
        if progress is not None:
            progress()


def render448(rows, root):
    """Reuse exact audited arithmetic; only the local raster width is different."""
    contracts = copy.deepcopy(common.CONTRACTS)
    contracts[common.CONTRACT]["img"] = 448
    fn = types.FunctionType(common.render_study.__code__,
                            {**common.render_study.__globals__, "CONTRACTS": contracts},
                            argdefs=common.render_study.__defaults__)
    return fn(rows, "oai", root)


def render_checked(ctx, uid):
    plan, rows, audit = ctx["plans"][uid], ctx["rows"], ctx["audit"]
    old, old_mask, old_picks = oai.render(oai.R64, plan, rows, audit.groups, oai.GEN_ROOT)
    pos = ctx["positions"][uid]
    shard = next(s for s in ctx["source"]["shards"] if s["start"] <= pos < s["stop"])
    stored = np.load(SOURCE / shard["file"], mmap_mode="r", allow_pickle=False)
    require(np.array_equal(old, stored[pos - shard["start"]]), "336 pixel replay differs")
    frozen_mask = ctx["masks"][ctx["ids"].index(uid)]
    require(np.array_equal(old_mask, frozen_mask), "336 mask replay differs")
    require(old_picks == ctx["picks"][uid], "336 picks replay differs")
    del stored, old
    planned = oai.plan_rows(plan, rows)
    native, mask, picks = render448(planned, oai.GEN_ROOT)
    oai._check_picks(plan, planned, picks, audit.groups)
    require(native.dtype == np.uint8 and native.shape == (64, 448, 448), "native448 format")
    require(np.array_equal(mask, frozen_mask) and picks == old_picks, "native mask/pick drift")
    return native, dict(replay336_exact=True, picks_identical=True, masks_identical=True,
                       picks_sha256=digest(picks), mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())


def atomic_array(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    def write():
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
    io.retry(write)
    io.replace(temporary, path)


def checked_shards(path, manifest, ctx, complete, *, progress=None):
    shards = manifest["shards"]
    require(len(shards) <= len(ctx["ids"]), "too many shards")
    if complete:
        require(manifest.get("finished") is True and len(shards) == len(ctx["ids"]), "unfinished corpus")
    for i, item in enumerate(shards):
        require(item["file"] == f"vols_shard{i:04d}.npy" and item["start"] == i and item["stop"] == i + 1,
                "shard order/bounds mismatch")
        file = path / item["file"]
        require(sha(file) == item["sha256"] and file.stat().st_size == item["bytes"], "native shard drift")
        volume = np.load(file, mmap_mode="r", allow_pickle=False)
        require(volume.shape == (1, 64, 448, 448) and volume.dtype == np.uint8, "native shard shape/dtype")
        require(all(item.get(k) is True for k in ("replay336_exact", "picks_identical", "masks_identical")),
                "missing exact replay evidence")
        require(item["picks_sha256"] == digest(ctx["picks"][ctx["ids"][i]]) and
                item["mask_sha256"] == hashlib.sha256(ctx["masks"][i].tobytes()).hexdigest(), "replay identity drift")
        if progress is not None:
            progress()


def verify(path, ids=None, *, progress=None):
    """Reauthenticate current source inputs and every committed native output."""
    path = Path(path).resolve()
    require(path in (OUT.resolve(), (OUT / "smoke").resolve()), "output outside fixed roots")
    if ids is None and path == (OUT / "smoke").resolve():
        ids = np.load(path / "ids.npy", allow_pickle=False).astype(str).tolist()
    ctx = context(ids)
    if path == OUT.resolve():
        require(ctx["ids"] == panel_ids(), "full corpus needs entire panel")
    authenticate_inputs(ctx, progress=progress)
    manifest = io.read_json(path / "manifest.json")
    require(io.read_json(path / "registration.json") == ctx["binding"], "build registration drift")
    require(manifest["binding"] == ctx["binding"], "manifest source binding drift")
    require(manifest["contract"] == CONTRACT and manifest["spec"] == ctx["spec"], "native contract drift")
    require(manifest["shape_total"] == [len(ctx["ids"]), 64, 448, 448] and manifest["dtype"] == "uint8",
            "manifest shape/dtype drift")
    for name, expected in (("ids", np.asarray(ctx["ids"])), ("masks", ctx["masks"])):
        file = path / f"{name}.npy"
        require(sha(file) == manifest[f"{name}_sha256"], "output array hash drift")
        got = np.load(file, allow_pickle=False)
        require(got.dtype == expected.dtype and np.array_equal(got, expected), "output array content drift")
    checked_shards(path, manifest, ctx, True, progress=progress)
    return dict(status="VERIFIED", studies=len(ctx["ids"]), contract=CONTRACT,
                manifest_sha256=sha(path / "manifest.json"), source_npz=len(ctx["binding"]["general_npz"]))


def prepare(ids=None, *, progress=None) -> Path:
    """Build all557, or the requested canonical subset in one fixed smoke folder."""
    ctx = context(ids)
    path = OUT if ids is None else OUT / "smoke"
    path.mkdir(parents=True, exist_ok=True)
    registration = path / "registration.json"
    if registration.exists():
        require(io.read_json(registration) == ctx["binding"], "existing registration drift")
    else:
        require(not any(p.name != "smoke" for p in path.iterdir()), "unregistered output files")
        io.atomic_json(registration, ctx["binding"])
    authenticate_inputs(ctx, progress=progress)
    manifest_file = path / "manifest.json"
    if manifest_file.exists():
        manifest = io.read_json(manifest_file)
        require(manifest["binding"] == ctx["binding"] and manifest["spec"] == ctx["spec"], "resume identity drift")
        checked_shards(path, manifest, ctx, False, progress=progress)
        if manifest.get("finished") is True:
            verify(path, ctx["ids"], progress=progress)
            return path
    else:
        manifest = dict(version=VERSION, contract=CONTRACT, spec=ctx["spec"], studies=len(ctx["ids"]),
                        dtype="uint8", shape_total=[len(ctx["ids"]), 64, 448, 448], shards=[],
                        binding=ctx["binding"], finished=False,
                        masks_policy="frozen audited source masks; exact336 replay on every study",
                        source="OAI GENERAL native pixels; no resize336; no labels")
        atomic_array(path / "ids.npy", np.asarray(ctx["ids"]))
        atomic_array(path / "masks.npy", ctx["masks"])
        manifest.update(ids_sha256=sha(path / "ids.npy"), masks_sha256=sha(path / "masks.npy"))
        io.atomic_json(manifest_file, manifest)
        if progress is not None:
            progress()
    need = (len(ctx["ids"]) - len(manifest["shards"])) * 64 * 448 * 448
    require(shutil.disk_usage(path).free >= need + MIN_FREE, "insufficient disk headroom")
    start = time.perf_counter()
    for i in range(len(manifest["shards"]), len(ctx["ids"])):
        file = path / f"vols_shard{i:04d}.npy"
        require(not file.exists(), "uncommitted native shard requires review")
        then = time.perf_counter()
        volume, evidence = render_checked(ctx, ctx["ids"][i])
        atomic_array(file, volume[None])
        manifest["shards"].append(dict(start=i, stop=i + 1, file=file.name, bytes=file.stat().st_size,
                                       sha256=sha(file), seconds=time.perf_counter() - then, **evidence))
        io.atomic_json(manifest_file, manifest)
        if progress is not None:
            progress()
        print(json.dumps(dict(state="OAI448_COMMIT", studies=i + 1, total=len(ctx["ids"]),
                              seconds=round(time.perf_counter() - start, 3))), flush=True)
    manifest.update(finished=True, elapsed_this_run_seconds=time.perf_counter() - start)
    io.atomic_json(manifest_file, manifest)
    verify(path, ctx["ids"], progress=progress)
    return path


def cpu_smoke():
    from training.src.raptor_common import Corpus, select_centers
    from training.arms.native_resolution_v2 import scale_windows
    begin = time.perf_counter()
    ids = panel_ids()[:1]
    path = prepare(ids)
    corpus = Corpus(path)
    volume = corpus.volume(ids[0])
    shapes = {}
    centers = select_centers(corpus.mask(ids[0]), 64, 36, corpus.spec, "within-slot-valid", None)
    for resolution in (224, 320):
        result = scale_windows(volume, centers, resolution)
        require(result.shape == (36, 3, resolution, resolution) and np.isfinite(result).all(),
                "serving window smoke failed")
        shapes[str(resolution)] = list(result.shape)
    return dict(status="CPU_SMOKE_CACHE_PASS", studies=1, seconds=round(time.perf_counter() - begin, 3),
                volume_shape=list(volume.shape), window_shapes=shapes, masks_equal=True,
                verification=verify(path, ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "cpu-smoke"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = {"status": "PREPARED", "path": str(prepare())}
    elif args.command == "verify":
        result = verify(OUT)
    else:
        result = cpu_smoke()
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
