"""H-B-145 v1.1.3: EXPOSED fold-excluded R7 teacher inference, never training.

Changelog: 2026-10-14 v1.1.3: shared Windows-safe output I/O, nonfatal derived
exports each 100 commits, fail-closed journals, verified v112 carry-over.

Changelog: 2026-10-13 v1.1.2: isolated heartbeat retries and failure receipts/stall.
2026-10-13 v1.1.1: explicit EXPOSED provenance; restore release CUDA
anatomy/side; bounded telemetry, owned-tree finally cleanup and durable stall status.

Changelog: 2026-10-13 v1.1.0: opt-in heldout,f0 shared preparation, bounded CPU
workers, one GPU consumer, role journals/exports and resource/throughput receipts.
Default --models heldout retains the v1.0.1 execution path with explicit exposure metadata.

Changelog: 2026-10-13 v1.0.1: verify existing HB128 wheel tree, preserve pinned
worker dependencies, accept HB145_R7_CODE, project the actual archive population.

Changelog: 2026-10-13 v1.0.0 — pinned serving-source instrumentation, archive
streaming, atomic per-study resume, separate raw/NOT-cross-fitted stacker CSVs.
Original R7 files and release stages are never modified. See LAUNCH.md/REPORT.md
in .local/scratch/astra-b5/H-B-145 for the held-out and licensing limitations.
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
import zipfile

# Set before importing any numerical package; <= 2 main + 2 loader threads.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = os.environ.get("HB145_THREADS", "2")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp169_r7_port as r7
from tools import hb145_output_io as output_io

VERSION = "1.1.3"
SCRATCH = ROOT / ".local/scratch/astra-b5/H-B-145"
MANIFEST_SHA = "c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6"
DEPS_SHA = "507e5fd2ff9619b2ecc97e8b332480c79e026795d8fd4bc27b04143ea063e2fc"
UID = r7.UID
LABELS = list(r7.LABELS)
EXPOSURE = {"exposure": "EXPOSED", "clean_oof": False,
            "pooled_target_priors": True, "pooled_state_priors": True,
            "fastmri_plus_auxiliary": True, "anchor_ancestry": "unresolved",
            "side_ancestry": "unresolved"}


def role_provenance(models):
    return {role + "_" + kind: {**EXPOSURE, "stackers_cross_fitted":
            "not_applied" if kind == "raw" else False}
            for role in models.split(",") for kind in ("raw", "post")}


META = list(EXPOSURE) + [UID, "fold_k", "status", "input_series", "rendered_series", "used_series",
        "dropped_series", "deduplicated_series", "skipped_series", "dicom_files",
        "decoded_files", "decode_failures", "author_gold", "author_review_holdout",
        "author_pool_member", "gold_excluded_by_recipe", "review_included_by_recipe",
        "pooled_prior_bias", "stackers_cross_fitted", "output_kind", "error"]


def fold_k(uid):
    # EXACT compact_train.py:2015, with manifest K=4 (tr != k at :2668).
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4


def atomic_json(path, value):
    return output_io.atomic_json(path, value)


def owned_remove(path, owner, ignore_errors=False):
    original = Path(path)
    path, owner = original.resolve(), Path(owner).resolve()
    if path == owner or not path.is_relative_to(owner) or original.is_symlink():
        raise ValueError("refusing removal outside owned child directory")
    if path.exists():
        try:
            output_io.rmtree(path)
        except OSError:
            if not ignore_errors:
                raise


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@contextlib.contextmanager
def exclusive_run(out):
    """OS lock, automatically released even on process death; never delete a lock."""
    with (Path(out) / "run.lock").open("a+b") as f:
        f.seek(0, 2)
        if not f.tell():
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class TrainSource:
    """Indexes DICOM names only; never opens competition train.csv/report fields."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.members = {}
        if self.zip:
            for info in self.zip.infolist():
                parts = info.filename.split("/")
                if len(parts) == 4 and parts[0] == "train_series" and parts[-1].endswith(".dcm"):
                    if any(p in ("", ".", "..") or "\\" in p or ":" in p for p in parts):
                        raise ValueError("unsafe archive path")
                    self.members.setdefault(parts[1], []).append(info.filename)
        else:
            if self.path.name != "train_series":
                raise ValueError("directory input must be named train_series")
            for p in sorted(self.path.glob("*/*/*.dcm")):
                self.members.setdefault(p.parent.parent.name, []).append(str(p))
        for uid in self.members:
            self.members[uid].sort()
        if not self.members:
            raise ValueError("no TRAIN DICOMs found")

    def counts(self, uid):
        names = self.members[uid]
        return {"input_series": len({Path(n).parent.name for n in names}), "dicom_files": len(names)}

    def stage(self, uid, data):
        dest = Path(data) / "test_series" / uid
        owned_remove(dest, data)
        dest.mkdir(parents=True)
        for name in self.members[uid]:
            rel = Path(name)
            target = dest / rel.parent.name / rel.name
            target.parent.mkdir(exist_ok=True)
            if self.zip:
                with self.zip.open(name) as src, target.open("wb") as out:
                    shutil.copyfileobj(src, out)
            else:
                os.link(name, target)
        return dest

    def close(self):
        if self.zip:
            self.zip.close()


def decode_audit(study):
    """Actually decode every pixel array with local pydicom handlers; no headers logged."""
    import pydicom
    result = {"decoded_files": 0, "decode_failures": 0, "transfer_syntax_counts": {}}
    for path in sorted(Path(study).glob("*/*.dcm")):
        try:
            ds = pydicom.dcmread(path)
            syntax = str(ds.file_meta.TransferSyntaxUID)
            result["transfer_syntax_counts"][syntax] = result["transfer_syntax_counts"].get(syntax, 0) + 1
            pixels = ds.pixel_array
            if pixels.size == 0:
                raise ValueError("empty pixels")
            result["decoded_files"] += 1
            del pixels, ds
        except Exception:
            result["decode_failures"] += 1
    return result


def record_path(out, uid):
    return Path(out) / "journal" / (hashlib.sha256(uid.encode()).hexdigest() + ".json")


def load_done(out, uids, identity):
    result = {}
    for uid in uids:
        path = record_path(out, uid)
        if path.exists():
            rec = output_io.read_json(path)
            if rec.get(UID) != uid or rec.get("identity") != identity or rec.get("fold_k") != fold_k(uid):
                raise ValueError("resume journal identity mismatch")
            if rec.get("status") not in ("ok", "dropped series", "failure"):
                raise ValueError("invalid terminal journal status")
            if rec.get("models") == "heldout,f0":
                from tools.hb145_r7_pipeline_v111 import validate_record
                validate_record(rec)
            result[uid] = rec
    return result


def export_csv(out, uids, identity, journal_read_fatal=True):
    try:
        records = load_done(out, uids, identity)
    except OSError as error:
        if not journal_read_fatal:
            raise
        output_io.receipt(Path(out) / "journal_failure_receipt.json", {
            "version": VERSION, "state": "JOURNAL_ERROR", "identity": identity,
            "phase": "read validation", "error": type(error).__name__,
            "winerror": getattr(error, "winerror", None), "validation_complete": False})
        raise output_io.JournalCommitError("persistent journal read failure; validation incomplete") from error
    try:
        _export_csv(out, uids, records)
    except OSError as error:
        # Validation errors stay fatal. Derived publications never kill inference.
        print(f"EXPORT_DEFERRED: {type(error).__name__}; {len(records)} committed journals; retry next checkpoint/end",
              file=sys.stderr, flush=True)
        output_io.receipt(Path(out) / "export_health.json", {
            "version": VERSION, "identity": identity, "deferred": True,
            "committed": len(records), "error": type(error).__name__})
    else:
        output_io.receipt(Path(out) / "export_health.json", {
            "version": VERSION, "identity": identity, "deferred": False,
            "committed": len(records)})
    return records


def _export_csv(out, uids, records):
    hook = Path(out) / "hook.json"
    if hook.exists() and json.loads(hook.read_text(encoding="utf-8")).get("models") == "heldout,f0":
        from tools.hb145_r7_pipeline_v111 import export_roles
        export_roles(out, uids, records)
        return records
    for kind, filename in (("raw", "raw.csv"), ("post", "post_stacker_NOT_CROSSFITTED.csv")):
        path = Path(out) / filename
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=META + LABELS)
            wr.writeheader()
            for uid in sorted(records):
                rec = records[uid]
                row = {k: rec.get(k, "") for k in META}
                row["output_kind"] = "pre_stacker" if kind == "raw" else "post_stacker_NOT_CROSSFITTED"
                row["stackers_cross_fitted"] = "not_applied" if kind == "raw" else False
                row.update(zip(LABELS, rec.get(kind) or [""] * len(LABELS)))
                wr.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        output_io.replace(tmp, path)
    return records


_HOOK = None
_SOURCE = None


def hook_config():
    global _HOOK
    if _HOOK is None:
        _HOOK = json.loads(Path(os.environ["HB145_CONFIG"]).read_text(encoding="utf-8"))
    return _HOOK


def install_dataset(cls):
    original = cls.__getitem__
    def get(self, i):
        global _SOURCE
        import torch
        torch.set_num_threads(2)
        cfg = hook_config()
        if _SOURCE is None:
            _SOURCE = TrainSource(cfg["source"])
        uid = self.uids[i]
        counts = _SOURCE.counts(uid)
        try:
            study = _SOURCE.stage(uid, self.data)
            counts.update(decode_audit(study))
            item = original(self, i)
        except Exception as exc:
            item = {"uid": uid, "error": type(exc).__name__, "images": [], "seconds": [0, 0, 0]}
        finally:
            owned_remove(Path(self.data) / "test_series" / uid, self.data)
        item["hb145_counts"] = counts
        if torch.cuda.is_initialized():
            item["hb145_aux_peak_allocated"] = torch.cuda.max_memory_allocated()
            item["hb145_aux_peak_reserved"] = torch.cuda.max_memory_reserved()
        return item
    cls.__getitem__ = get


def checkpoint(uid, item, raw=None, post=None, error=None):
    cfg = hook_config()
    counts = dict(cfg.get("counts", {}).get(uid, {}))
    counts.update(item.get("hb145_counts", {}))
    used = item.get("hb145_used_series", 0)
    raw = None if raw is None else [float(x) for x in raw]
    post = None if post is None else [float(x) for x in post]
    if raw is not None and (len(raw) != 12 or len(post) != 12 or
            any(not math.isfinite(v) or not 0 <= v <= 1 for v in raw + post)):
        raise ValueError("invalid model probabilities")
    dropped = max(0, counts.get("input_series", 0) - used)
    rec = {**EXPOSURE, UID: uid, "fold_k": fold_k(uid), "identity": cfg["identity"],
           "status": "failure" if raw is None else ("dropped series" if dropped else "ok"),
           **counts, "rendered_series": item.get("hb145_rendered_series", 0),
           "used_series": used, "dropped_series": dropped,
           "deduplicated_series": item.get("hb145_removed", 0),
           "skipped_series": item.get("hb145_skipped", 0),
           **cfg["flags"].get(uid, {}), "gold_excluded_by_recipe": True,
           "review_included_by_recipe": True, "pooled_prior_bias": True,
           "raw": raw, "post": post, "error": error or "",
           "seconds_decode_render_anchor": item.get("seconds"),
           "aux_peak_allocated": item.get("hb145_aux_peak_allocated"),
           "aux_peak_reserved": item.get("hb145_aux_peak_reserved")}
    if cfg.get("models") == "heldout,f0":
        rec.update(models="heldout,f0", predictions=item.get("hb145_predictions", {}),
                   role_folds={"heldout": fold_k(uid), "f0": 0},
                   roles_coincide=fold_k(uid) == 0,
                   completed_monotonic=time.perf_counter(),
                   preparation_calls=1, auxiliary_device=cfg.get("device", "cpu"),
                   aux_cuda_initialized=item.get("hb145_aux_cuda_initialized", False))
        # Unique fold payloads only; fold 0 is one prediction referenced by two roles.
        rec.pop("raw")
        rec.pop("post")
    for name in ("author_gold", "author_review_holdout", "author_pool_member"):
        rec.setdefault(name, "unknown")
    path = record_path(cfg["out"], uid)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("identity") != cfg["identity"]:
            raise ValueError("checkpoint identity mismatch")
        # A later logging exception must not erase an already committed prediction.
        if (previous.get("raw") is not None or previous.get("predictions")) and raw is None:
            return
    output_io.journal(path, rec, cfg["out"])


def runtime_source(stage, dual=False):
    """Only bookkeeping/scheduling patches. R7 rendering, TTA, logits, stackers reused."""
    runtime = r7.diagnostic_source(r7.SOURCE, stage)
    path = runtime / "compact_infer.py"
    code = path.read_text(encoding="utf-8")
    patches = [
        ('    dev = torch.device(a.device)', '    torch.set_num_threads(2)\n    dev = torch.device(a.device)'),
        ('                   "side": side_note}', '                   "side": side_note, "hb145_rendered_series": len(item["blocks"]),\n                   "hb145_used_series": len(set(m[:, 0].tolist())),\n                   "hb145_removed": item.get("removed", 0), "hb145_skipped": len(item.get("skipped_series", []))}'),
        ('        dt = time.perf_counter() - tg', '        hb145_raw = p.copy()\n        dt = time.perf_counter() - tg'),
        ('        wr.writerow([uid] + ["%.9g" % float(x) for x in p])',
         '        HB.checkpoint(uid, item, hb145_raw, p)\n        wr.writerow([uid] + ["%.9g" % float(x) for x in p])'),
        ('                handle(item)\n                pos += 1',
         '                handle(item)\n                if item["uid"] in fallback:\n                    HB.checkpoint(item["uid"], item, error=fallback[item["uid"]])\n                pos += 1'),
        ('            pos += 1\n        del loader',
         '                HB.checkpoint(uids[pos], {}, error=fallback[uids[pos]])\n            pos += 1\n        del loader'),
        ('    (out / "receipt.json").write_text',
         '    receipt.update(**HB.EXPOSURE, identity=HB.hook_config()["identity"], role_provenance=HB.role_provenance(HB.hook_config().get("models", "heldout")))\n    receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0\n    receipt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved() if dev.type == "cuda" else 0\n    (out / "receipt.json").write_text'),
        ('if __name__ == "__main__":\n    main()',
         'from tools import hb145_r7_heldout_v1 as HB\nHB.install_dataset(CStudies)\n\nif __name__ == "__main__":\n    main()'),
    ]
    for n, (old, new) in enumerate(patches):
        code = r7.s6._replace_once(code, old, new, f"hb145 instrumentation {n}")
    if dual:
        from tools.hb145_r7_pipeline_v111 import patch_runtime
        code = patch_runtime(code)
    # Output-only substitutions: native scoring code is unchanged.
    code = output_runtime_patch(code)
    compile(code, str(path), "exec")
    path.write_text(code, encoding="utf-8", newline="\n")
    pins_path = runtime / "runtime_sha256.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    pins["compact_infer.py"] = r7.s6.sha256(path)
    atomic_json(pins_path, pins)
    return runtime


def output_runtime_patch(code):
    for name in ("work", "tmp"):
        code = code.replace(f"shutil.rmtree({name}, ignore_errors=True)",
                            f'HB.owned_remove({name}, HB.hook_config()["out"], ignore_errors=True)')
    for expression in ('out / "predictions.csv"', 'a.fluid_dump or (out / "fluid_stacker.csv")',
                       'a.state_dump or (out / "state_stacker.csv")'):
        code = code.replace(f'open({expression}, "w", encoding="utf-8", newline="")',
                            f'HB.output_io.DerivedStream({expression}, encoding="utf-8", newline="")')
    code = code.replace('Path(a.dump_logits).write_text(json.dumps(dump), encoding="utf-8")',
                        'HB.output_io.receipt(Path(a.dump_logits), dump)')
    return r7.s6._replace_once(code,
        '(out / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")',
        'HB.output_io.receipt(out / "receipt.json", receipt)', "v113 output receipt")


def selected_bundle(source, target, manifest, k):
    target.mkdir(parents=True)
    names = [r7.MODELS[k]] + [n for n in r7.declared_files(manifest) if n not in r7.MODELS]
    for name in names:
        os.link(Path(source) / name, target / name)
    if sorted(p.name for p in target.glob(r7.ARM + "_s*_f*.pt")) != [r7.MODELS[k]]:
        raise ValueError("fold model isolation failed")
    return target


def default_paths():
    with (ROOT / "config/workspace.toml").open("rb") as f:
        cfg = tomllib.load(f)
    paths = cfg["paths"]
    archive = ROOT / "storage/downloads" / cfg["competition"] / (cfg["competition"] + ".zip")
    return archive, ROOT / paths["data"] / "train_series.csv", ROOT / paths["external"] / "heliosli-r7cnxfsss-v1", ROOT / paths["weights"] / "external/heliosli_rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt"


def worker_env(code, deps, runtime=None):
    # Same extracted wheel precedence as HB128.worker_pythonpath, no installation.
    paths = [str(deps.resolve()), str(ROOT)]
    if runtime is not None:
        paths.append(str(runtime.resolve()))
    paths.append(str(code.resolve()))
    return dict(os.environ, PYTHONPATH=os.pathsep.join(paths))


def verify_deps(deps, code):
    tree = r7.s6.code_tree_manifest(deps)
    if tree["tree_sha256"] != DEPS_SHA:
        raise ValueError("pinned extracted dependency tree drift")
    probe = ("import json,pydicom,timm; print(json.dumps({"
             "'pydicom':pydicom.__version__,'timm':timm.__version__,"
             "'pydicom_file':pydicom.__file__,'timm_file':timm.__file__}))")
    result = subprocess.run([sys.executable, "-B", "-c", probe],
                            env=worker_env(code, deps), text=True, capture_output=True, check=True)
    versions = json.loads(result.stdout.strip().splitlines()[-1])
    if any(versions[k] != v for k, v in r7.s6.WHEEL_VERSIONS.items()):
        raise ValueError("pinned dependency versions differ")
    if any(not Path(versions[k + "_file"]).resolve().is_relative_to(deps.resolve())
           for k in r7.s6.WHEEL_VERSIONS):
        raise ValueError("dependencies not imported from pinned tree")
    return {"path": str(deps.resolve()), "tree_sha256": DEPS_SHA, "versions": versions}


def main(argv=None):
    archive, series, bundle, anatomy = default_paths()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "decode", "run", "export", "preprocess"))
    ap.add_argument("--source", type=Path, default=archive)
    ap.add_argument("--series", type=Path, default=series)
    ap.add_argument("--bundle", type=Path, default=bundle)
    ap.add_argument("--anatomy", type=Path, default=anatomy)
    ap.add_argument("--code", type=Path, default=os.environ.get("HB145_R7_CODE"))
    ap.add_argument("--deps", type=Path, default=ROOT / ".local/scratch/claude-hb128/runtime/deps")
    ap.add_argument("--flags", type=Path, help="UID-only CSV: UID,author_gold,author_review_holdout,author_pool_member")
    ap.add_argument("--out", type=Path, default=SCRATCH / "full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--models", choices=("heldout", "heldout,f0"), default="heldout")
    ap.add_argument("--workers", type=int, choices=range(11), default=1)
    ap.add_argument("--chunk", type=int, choices=(32, 64), default=32)
    ap.add_argument("--carry-over", type=Path, help="verified v1.1.2 journal source; prepare only")
    a = ap.parse_args(argv)
    a.out = a.out.resolve()
    if not a.out.is_relative_to(SCRATCH.resolve()) or a.out == SCRATCH.resolve():
        ap.error("--out must be a child of fixed task scratch")
    if a.limit < 0:
        ap.error("limit must be nonnegative")
    if a.carry_over and a.command != "prepare":
        ap.error("--carry-over is only allowed with prepare")
    if a.models == "heldout" and (a.workers > 1 or a.command == "preprocess"):
        ap.error("legacy heldout mode supports workers 0/1; preprocess requires heldout,f0")
    a.out.mkdir(parents=True, exist_ok=True)
    with exclusive_run(a.out):
        return execute(a)


def execute(a):
    r7.verify_source()
    deps_id = verify_deps(a.deps, a.code) if a.code else None
    if deps_id:
        sys.path.insert(0, str(a.deps.resolve()))
    src = TrainSource(a.source)
    uids = sorted(src.members)
    if a.limit:
        uids = uids[:a.limit]
    if a.models == "heldout,f0":
        # Limit selects a reproducible UID sample before prioritizing the screen pool.
        uids.sort(key=lambda u: (fold_k(u) == 0, u))
    flags = {}
    if a.flags:
        with a.flags.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            expected = [UID, "author_gold", "author_review_holdout", "author_pool_member"]
            if reader.fieldnames != expected:
                raise ValueError("flags must contain only UID and three membership flags")
            for row in reader:
                uid = row.pop(UID)
                if uid in flags or uid not in src.members or any(v not in ("0", "1", "unknown") for v in row.values()):
                    raise ValueError("invalid/duplicate membership flag")
                flags[uid] = row
    inventory = {"population": len(src.members), "selected": len(uids),
                 "dicom_files": sum(len(v) for v in src.members.values()),
                 "fold_counts": [sum(fold_k(u) == k for u in uids) for k in range(4)]}
    atomic_json(a.out / "inventory.json", inventory)
    if a.command == "decode":
        results = []
        start = time.perf_counter()
        for uid in uids:
            try:
                study = src.stage(uid, a.out / "decode_data")
                results.append({UID: uid, **src.counts(uid), **decode_audit(study)})
            finally:
                owned_remove(a.out / "decode_data" / "test_series" / uid, a.out)
        receipt = {"version": VERSION, "studies": len(results), "seconds": time.perf_counter() - start,
                   "decoded_files": sum(r["decoded_files"] for r in results),
                   "decode_failures": sum(r["decode_failures"] for r in results), "per_study": results}
        output_io.receipt(a.out / "decode_receipt.json", receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != "per_study"}))
        src.close()
        return int(receipt["decode_failures"] > 0)
    manifest, bundle_id = r7.verify_bundle(a.bundle, MANIFEST_SHA)
    code_id = r7.code_record(a.code, a.anatomy) if a.code else None
    if a.command == "run" and code_id is None:
        raise ValueError("--code must name the existing pinned mainline_code tree; no download/install is performed")
    stage = a.out / "stage"
    owned_remove(stage, a.out)
    stage.mkdir()
    runtime = runtime_source(stage, dual=a.models == "heldout,f0")
    input_stat = a.source.stat()
    identity_record = {**EXPOSURE, "role_provenance": role_provenance(a.models),
        "auxiliary_device": a.device, "stall_seconds": 900, "version": VERSION, "runner_sha256": r7.s6.sha256(__file__),
        "bundle": bundle_id, "code": code_id, "deps": deps_id, "source": str(a.source.resolve()),
        "source_stat": [input_stat.st_size, input_stat.st_mtime_ns],
        "member_digest": fingerprint(src.members), "uids": uids, "flags": flags,
        "series_sha256": r7.s6.sha256(a.series), "device": a.device,
        "amp": "fp16" if a.device == "cuda" else "fp32", "workers": a.workers,
        "chunk": a.chunk, "models": a.models,
        "monitor_sha256": r7.s6.sha256(ROOT / "tools/hb145_r7_monitor_v111.py") if a.models == "heldout,f0" else None,
        "pipeline_sha256": r7.s6.sha256(ROOT / "tools/hb145_r7_pipeline_v111.py") if a.models == "heldout,f0" else None,
        "output_io_sha256": r7.s6.sha256(ROOT / "tools/hb145_output_io.py"),
        "carryover_sha256": r7.s6.sha256(ROOT / "tools/hb145_r7_carryover_v113.py"),
        "runtime_sha256": r7.s6.sha256(runtime / "compact_infer.py")}
    identity = fingerprint(identity_record)
    pin = a.out / "identity.json"
    if pin.exists() and json.loads(pin.read_text(encoding="utf-8")) != identity_record:
        raise ValueError("resume identity differs; use a separate fixed run directory")
    atomic_json(pin, identity_record)
    cfg = {"source": str(a.source.resolve()), "out": str(a.out), "identity": identity, "flags": flags,
           "models": a.models, "device": a.device, "uids": uids,
           "counts": {uid: src.counts(uid) for uid in uids}}
    atomic_json(a.out / "hook.json", cfg)
    if getattr(a, "carry_over", None):
        from tools.hb145_r7_carryover_v113 import carry_over
        carry_over(a.carry_over, a.out, identity_record, runtime / "compact_infer.py")
    from tools.hb145_r7_carryover_v113 import validate_ready
    validate_ready(a.out, identity)
    done = export_csv(a.out, uids, identity)
    if a.command in ("prepare", "export"):
        print(json.dumps({**inventory, "done": len(done), "code_verified": code_id is not None,
                          "runtime_compiles": True, "full_inference_ready": code_id is not None}))
        src.close()
        return 0
    data = stage / "data"
    data.mkdir()
    # Metadata only, not train.csv. Whitelist columns; reject missing study/series mapping.
    with a.series.open(encoding="utf-8-sig", newline="") as f, (data / "test_series.csv").open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        columns = [UID, "SeriesInstanceUID", "Anatomical_Plane"]
        if any(c not in reader.fieldnames for c in columns):
            raise ValueError("series metadata schema mismatch")
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        seen = set()
        for row in reader:
            if row[UID] in src.members:
                writer.writerow({c: row[c] for c in columns})
                seen.add(row[UID])
        if not set(uids) <= seen:
            raise ValueError("selected study lacks series metadata")
    with (data / "test.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([UID])
        writer.writerows([u] for u in uids)
    src.close()
    env = worker_env(a.code, a.deps, runtime)
    env["HB145_CONFIG"] = str(a.out / "hook.json")
    if a.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    if a.models == "heldout,f0":
        from tools.hb145_r7_pipeline_v111 import execute_pipeline
        return execute_pipeline(a, manifest, runtime, data, env, uids, done, identity)
    start = time.perf_counter()
    receipts = []
    try:
        for k in range(4):
            pending = [u for u in uids if fold_k(u) == k and u not in done]
            if not pending:
                continue
            fold = stage / f"f{k}"
            fold.mkdir()
            models = selected_bundle(a.bundle, fold / "bundle", manifest, k)
            atomic_json(fold / "uids.json", pending)
            args = [sys.executable, "-B", str(runtime / "compact_infer.py"),
                "--data", str(data), "--code", str(a.code.resolve()), "--models", str(models),
                "--arm", r7.ARM, "--output", str(fold / "output"), "--assignment", str(fold / "uids.json"),
                "--device", a.device, "--amp", "fp16", "--workers", str(a.workers),
                "--prefetch", "1", "--chunk", str(a.chunk), *r7.worker_args(manifest, models, a.anatomy.resolve())]
            atomic_json(a.out / f"invocation_f{k}.json", args)
            with (a.out / f"worker_f{k}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt_path = fold / "output/receipt.json"
            if result.returncode:
                raise RuntimeError(f"fold {k} worker failed; see worker_f{k}.log; no fabricated completed rows")
            receipt = output_io.read_receipt(receipt_path)
            receipt.update(**EXPOSURE, identity=identity, role_provenance=role_provenance(a.models))
            if receipt.get("models", [r7.MODELS[k]]) != [r7.MODELS[k]]:
                raise ValueError("executed model roster mismatch")
            output_io.receipt(a.out / f"receipt_f{k}.json", receipt)
            receipts.append(receipt)
            done = export_csv(a.out, uids, identity)
            if not set(pending) <= set(done):
                raise RuntimeError("worker did not journal every assigned study")
    finally:
        done = export_csv(a.out, uids, identity)
        seconds = time.perf_counter() - start
        scored = sum(r.get("scored", 0) for r in receipts)
        summary = {**EXPOSURE, "role_provenance": role_provenance(a.models), "identity": identity, "version": VERSION, "completed": len(done), "selected": len(uids),
            "failures": sum(r["status"] == "failure" for r in done.values()),
            "decoded_files": sum(r.get("decoded_files", 0) for r in done.values()),
            "decode_failures": sum(r.get("decode_failures", 0) for r in done.values()),
            "new_scored": scored, "invocation_seconds": seconds,
            "seconds_per_new_study_including_startup": seconds / scored if scored else None,
            "projected_4344_hours_including_startup": seconds / scored * 4344 / 3600 if scored else None,
            "projected_population_hours_including_startup": seconds / scored * inventory["population"] / 3600 if scored else None,
            "peak_main_allocated_bytes": max((r.get("peak_allocated_bytes", 0) for r in receipts), default=0),
            "peak_main_reserved_bytes": max((r.get("peak_reserved_bytes", 0) for r in receipts), default=0),
            "peak_aux_reserved_bytes": max((r.get("aux_peak_reserved") or 0 for r in done.values()), default=0),
            "vram_note": "main and loader allocator peaks exclude CUDA context/driver memory; sum is a conservative allocator bound for workers=1",
            "codecs_all_selected_verified": len(done) == len(uids) and all(r.get("decoded_files") == r.get("dicom_files") and r.get("decode_failures") == 0 for r in done.values())}
        output_io.receipt(a.out / "run_receipt.json", summary)
    print(json.dumps(summary))
    return int(summary["failures"] > 0 or summary["completed"] != len(uids))


if __name__ == "__main__":
    raise SystemExit(main())
