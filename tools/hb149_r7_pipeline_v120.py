"""H-B-149 execution v1.2.5: frozen v1.1.2 pipeline fork, one flip heldout fold.
Changelog 2026-10-14 v1.2.5: bounded CSV publication; nonfatal receipts.
Changelog 2026-10-13: role/scheduler/export names only; release math and
anatomy/side device unchanged; v1.1.2 monitor imported without modification.
"""
from __future__ import annotations

import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools import hb149_r7_flip_v120 as h

ROLE_META = ["model_role", "model_fold", "roles_coincide", "prediction_reference", "auxiliary_device"]


def required_folds(uid):
    k = h.fold_k(uid)
    if k == 0:
        raise ValueError("heldout_flip only accepts hash folds 1-3")
    return [k]


def validate_record(rec):
    k = h.fold_k(rec[h.UID])
    if rec.get("role_folds") != {"heldout_flip": k} or rec.get("roles_coincide") is not False or rec.get("flip_tta") is not True:
        raise ValueError("role/fold journal mismatch")
    predictions = rec.get("predictions", {})
    if rec["status"] != "failure" and set(predictions) != {str(f) for f in required_folds(rec[h.UID])}:
        raise ValueError("incomplete dual-role journal")
    for key, pred in predictions.items():
        if key not in {str(f) for f in required_folds(rec[h.UID])}:
            raise ValueError("unexpected fold prediction")
        for kind in ("raw", "post"):
            values = pred[kind]
            if len(values) != 12 or any(not h.math.isfinite(x) or not 0 <= x <= 1 for x in values):
                raise ValueError("invalid role probabilities")


def export_roles(out, uids, records):
    for role in ("heldout_flip",):
        for kind in ("raw", "post"):
            suffix = "raw" if kind == "raw" else "post_stacker_NOT_CROSSFITTED"
            path = Path(out) / f"{role}_{suffix}.csv"
            tmp = path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=h.META + ROLE_META + h.LABELS)
                writer.writeheader()
                for uid in uids:
                    if uid not in records:
                        continue
                    rec = records[uid]
                    fold = rec["role_folds"][role]
                    row = {k: rec.get(k, "") for k in h.META + ROLE_META}
                    row.update(model_role=role, model_fold=fold, roles_coincide=rec["roles_coincide"],
                               prediction_reference=f"{h.record_path(out, uid).name}#predictions/{fold}",
                               output_kind="pre_stacker" if kind == "raw" else "post_stacker_NOT_CROSSFITTED",
                               stackers_cross_fitted="not_applied" if kind == "raw" else False)
                    pred = rec.get("predictions", {}).get(str(fold), {})
                    row.update(zip(h.LABELS, pred.get(kind, [""] * 12)))
                    writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
            h.output_io.replace(tmp, path)


def capture(uid, item, raw, post):
    item.setdefault("hb145_predictions", {})[str(item["hb145_active_fold"])] = {
        "raw": [float(v) for v in raw], "post": [float(v) for v in post]}


def commit(item, error=None):
    uid = item["uid"]
    pred = item.get("hb145_predictions", {})
    if set(pred) != {str(k) for k in required_folds(uid)}:
        error = error or "one or more model roles failed"
    heldout = pred.get(str(h.fold_k(uid)), {}) if not error else {}
    h.checkpoint(uid, item, heldout.get("raw"), heldout.get("post"), error=error)


def cpu_worker_init(worker_id):
    # Executed before _setup: no auxiliary CUDA context in any loader.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


def worker_init(worker_id):
    # Preserve release self.device; CPU invocations already hide CUDA in parent env.
    import torch
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


_SOURCE = None


def install_dataset(cls):
    original = cls.__getitem__
    def get(self, i):
        global _SOURCE
        import numpy as np
        import pydicom
        import torch
        cfg = h.hook_config()
        torch.set_num_threads(1)
        uid = self.uids[i]
        if _SOURCE is None:
            _SOURCE = h.TrainSource(cfg["source"])
        counts = _SOURCE.counts(uid)
        decoded, failed, syntax = set(), set(), {}
        pixel_fn, geom = None, None
        try:
            study = _SOURCE.stage(uid, self.data)
            if self.native is None:
                self._setup()
            import training.dicom_geometry as geom
            pixel_fn = geom.pixels_from_dataset
            def observed(ds, image=None, path="archive"):
                # Identical serving conversion, observed instead of a second audit decode.
                if image is None:
                    try:
                        image = ds.pixel_array.astype(np.float32)
                    except Exception:
                        failed.add(str(path))
                        raise
                    decoded.add(str(path))
                    syntax[str(path)] = str(ds.file_meta.TransferSyntaxUID)
                return pixel_fn(ds, image=image, path=path)
            geom.pixels_from_dataset = observed
            item = original(self, i)
            # Audit only files the serving renderer did not request (e.g. dropped series).
            for path in sorted(study.glob("*/*.dcm")):
                key = path.relative_to(self.data).as_posix()
                if key in decoded or key in failed:
                    continue
                try:
                    ds = pydicom.dcmread(path)
                    if ds.pixel_array.size == 0:
                        raise ValueError("empty pixels")
                    decoded.add(key)
                    syntax[key] = str(ds.file_meta.TransferSyntaxUID)
                except Exception:
                    failed.add(key)
        except Exception as exc:
            item = {"uid": uid, "error": type(exc).__name__, "images": [], "seconds": [0, 0, 0]}
        finally:
            if geom is not None and pixel_fn is not None:
                geom.pixels_from_dataset = pixel_fn
            h.owned_remove(Path(self.data) / "test_series" / uid, self.data)
        counts.update(decoded_files=len(decoded), decode_failures=len(failed),
                      transfer_syntax_counts={v: list(syntax.values()).count(v) for v in set(syntax.values())})
        item["hb145_counts"] = counts
        item["hb145_aux_cuda_initialized"] = torch.cuda.is_initialized()
        return item
    cls.__getitem__ = get


def patch_runtime(code):
    wrap = '''    hb145_models = dict(zip([int(p.stem.rsplit("f", 1)[1]) for p in files], models))
    if set(hb145_models) != set(range(4)):
        raise ValueError("dual-role roster must have exactly f0..f3")
    hb145_one = handle
    hb145_checked = set()
    def handle(item):
        nonlocal models
        uid = item["uid"]
        for k in PIPE.required_folds(uid):
            models = [hb145_models[k]]
            item["hb145_active_fold"] = k
            # Use the original compile gate separately for each model on supported hosts.
            if EAGER:
                COMPILE["checked"] = k in hb145_checked
            hb145_one(item)
            hb145_checked.add(k)
            if uid in fallback:
                break
        PIPE.commit(item, fallback.get(uid))

    pos = 0'''
    patches = [
        ('HB.install_dataset(CStudies)', 'from tools import hb149_r7_pipeline_v120 as PIPE\nPIPE.install_dataset(CStudies)'),
        ('                torch.set_num_threads(2)', '                torch.set_num_threads(1)'),
        ('        HB.checkpoint(uid, item, hb145_raw, p)', '        PIPE.capture(uid, item, hb145_raw, p)'),
        ('    pos = 0', wrap),
        ('num_workers=a.workers,\n', 'num_workers=a.workers, worker_init_fn=PIPE.worker_init,\n'),
        ('                if item["uid"] in fallback:\n                    HB.checkpoint(item["uid"], item, error=fallback[item["uid"]])\n', ''),
    ]
    for index, (before, after) in enumerate(patches):
        code = h.r7.s6._replace_once(code, before, after, f"v111 scheduler {index}")
    return code


from tools.hb145_r7_monitor_v111 import monitored_run


def execute_pipeline(a, manifest, runtime, data, env, uids, done, identity):
    if manifest.get("flip_tta") is not True:
        raise ValueError("heldout_flip requires built-in --flip-tta")
    stage = runtime.parent
    pending = [u for u in uids if u not in done]
    if a.command == "preprocess":
        pending = list(uids)
    h.atomic_json(stage / "uids.json", pending)
    env.update(HB145_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    # All four models loaded once; only the role-selected singleton is scored.
    args = [sys.executable, "-B", str(runtime / "compact_infer.py"),
            "--data", str(data), "--code", str(a.code.resolve()), "--models", str(a.bundle.resolve()),
            "--arm", h.r7.ARM, "--output", str(stage / "output"), "--assignment", str(stage / "uids.json"),
            "--device", a.device, "--amp", "fp16", "--workers", str(a.workers),
            "--prefetch", "1", "--chunk", str(a.chunk),
            *h.r7.worker_args(manifest, a.bundle.resolve(), a.anatomy.resolve())]
    if a.device == "cpu" and "--compile" in args:
        args.remove("--compile")
    h.atomic_json(a.out / "invocation_dual.json", args)
    if a.command == "preprocess":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        args = [sys.executable, "-B", str(Path(__file__).resolve()), "--cpu-preprocess", str(a.out / "invocation_dual.json")]
    tick_count = 0
    def refresh():
        nonlocal tick_count
        tick_count += 1
        if tick_count % 15 == 0:
            h.export_csv(a.out, uids, identity)
    code, resource = (0, {})
    if pending:
        code, resource = monitored_run(args, env, a.out / f"{a.command}_dual.log",
                                       a.out / f"{a.command}_resources.json", refresh if a.command == "run" else None)
    if a.command == "preprocess":
        print(json.dumps({"cpu_preprocess_exit": code, "receipt": str(a.out / "cpu_preprocess.json")}))
        return code
    records = h.export_csv(a.out, uids, identity)
    worker_receipt = stage / "output/receipt.json"
    receipt = json.loads(worker_receipt.read_text(encoding="utf-8")) if worker_receipt.exists() else {}
    if pending and (not receipt or receipt.get("models") != h.r7.MODELS):
        code = code or 1
    if receipt and not receipt.get("flip_tta"):
        code = code or 1
    if receipt:
        receipt.update(**h.EXPOSURE, identity=identity, version=h.VERSION, role_provenance=h.role_provenance(a.models))
        h.output_io.receipt(a.out / "receipt_dual.json", receipt)
    times = sorted(r["completed_monotonic"] for uid, r in records.items() if uid in pending and r["status"] != "failure")
    # Drop the first worker wave (worker/model imports, first decode/compile); tail includes pipeline draining.
    warm = min(max(a.workers, 1), len(times) - 1) if len(times) > 1 else 0
    steady = (times[-1] - times[warm - 1]) / (len(times) - warm) if warm else None
    active = [s for s in resource.get("samples", []) if times and s[0] >= times[max(0, warm - 1)] and s[0] <= times[-1]]
    summary = {"version": h.VERSION, "role_provenance": h.role_provenance(a.models), "models": a.models, "selected": len(uids), "completed": len(records),
               "new_studies": len(times), "exit_code": code,
               "failures": sum(r["status"] == "failure" for r in records.values()),
               "fold_counts": [sum(h.fold_k(u) == k for u in records) for k in range(4)],
               "unique_model_predictions": sum(len(r["predictions"]) for r in records.values()),
               "decoded_files": sum(r.get("decoded_files", 0) for r in records.values()),
               "decode_failures": sum(r.get("decode_failures", 0) for r in records.values()),
               "codec_coverage_complete": bool(records) and all(r.get("decoded_files") == r.get("dicom_files") and r.get("decode_failures") == 0 for r in records.values()),
               "warmup_studies_excluded": warm, "steady_studies": len(times) - warm,
               "steady_seconds_per_study": steady,
               "projected_4407_hours": steady * 4407 / 3600 if steady else None,
               "steady_tree_cpu_percent_one_core_100": sum(s[3] for s in active) / len(active) if active else None,
               "steady_system_cpu_percent": sum(s[4] for s in active) / len(active) if active else None,
               "steady_gpu_util_percent": sum(s[6] for s in active) / len(active) if active else None,
               "peak_main_allocated_bytes": receipt.get("peak_allocated_bytes"),
               "peak_main_reserved_bytes": receipt.get("peak_reserved_bytes"),
               "auxiliary_device": a.device, **h.EXPOSURE, "identity": identity, "workers": a.workers, "prefetch_per_worker": 1,
               "chunk": a.chunk, "resources": {k: v for k, v in resource.items() if k != "samples"}}
    h.output_io.receipt(a.out / "run_receipt.json", summary)
    print(json.dumps(summary))
    return int(bool(code) or summary["failures"] > 0 or len(records) != len(uids))


def cpu_preprocess(invocation):
    args = json.loads(Path(invocation).read_text(encoding="utf-8"))
    def value(flag, default=None):
        return args[args.index(flag) + 1] if flag in args else default
    mod = importlib.import_module("compact_infer")
    import torch
    cpu_worker_init(0)
    uids = json.loads(Path(value("--assignment")).read_text(encoding="utf-8"))
    dataset = mod.CStudies(value("--data"), "test", uids, str(Path(value("--output")).parent / "cpu-prep"),
        value("--code"), int(value("--size", "288")), int(value("--eval-stride", "3")), value("--views", "v1"),
        int(value("--slices", "3")), value("--anchor-weights"), float(value("--phys-mm", "0")),
        value("--side-model"), float(value("--side-threshold", "2")), float(value("--side-max-x", "0")),
        float(value("--side-position-mm", "20")), extent_meta=True, fluid=True, device="cpu")
    workers = int(value("--workers"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers,
        prefetch_factor=1 if workers else None, multiprocessing_context="spawn" if workers else None,
        worker_init_fn=cpu_worker_init, timeout=900 if workers else 0)
    rows = []
    for item in loader:
        mirror_ok = False
        if not item["error"]:
            # Real-data CPU smoke of the exact built-in TTA transform, after side/crop.
            images, meta = item["images"][None], item["meta"][None]
            mirrored, mirrored_meta = mod.mirror_windows(images, meta)
            restored, restored_meta = mod.mirror_windows(mirrored, mirrored_meta)
            mirror_ok = bool(torch.equal(restored, images) and torch.equal(restored_meta, meta))
            if not mirror_ok:
                raise ValueError("real-data mirror involution failed")
        rows.append({"error": item["error"], "windows": len(item["images"]),
                     "counts": item["hb145_counts"], "aux_cuda": item["hb145_aux_cuda_initialized"],
                     "mirror_involution": mirror_ok})
    cfg = h.hook_config()
    result = {"version": h.VERSION, **h.EXPOSURE, "identity": cfg["identity"],
              "role_provenance": h.role_provenance(cfg["models"]),
              "studies": len(rows), "failures": sum(bool(r["error"]) for r in rows),
              "decoded_files": sum(r["counts"]["decoded_files"] for r in rows),
              "decode_failures": sum(r["counts"]["decode_failures"] for r in rows),
              "aux_cuda_initialized": any(r["aux_cuda"] for r in rows),
              "real_data_mirror_involution": all(r["mirror_involution"] for r in rows),
              "windows": [r["windows"] for r in rows], "error_types": [r["error"] for r in rows if r["error"]]}
    h.atomic_json(Path(invocation).parent / "cpu_preprocess.json", result)
    print(json.dumps(result))
    return int(result["failures"] > 0 or result["decode_failures"] > 0 or result["aux_cuda_initialized"])


if __name__ == "__main__":
    raise SystemExit(cpu_preprocess(sys.argv[2]))
