"""Build/run a fixed D01 engineering benchmark; build is CPU/read-only input work.

No label, report, metric, submission, remote upload or training operation exists.
The runtime uses the immutable D01 dataset's serving code and five checkpoints.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
UID = "StudyInstanceUID"
SERIES = "SeriesInstanceUID"
ASSET_SHA = "e62893ebb1e22dbb1288ea6c805097bbc96ef5e43759f42a22d7cfb3bbef036b"
SYNTAXES = {
    "1.2.840.10008.1.2": "Implicit VR Little Endian",
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian",
    "1.2.840.10008.1.2.4.57": "JPEG Lossless Process 14",
    "1.2.840.10008.1.2.4.70": "JPEG Lossless SV1",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
    "1.2.840.10008.1.2.4.91": "JPEG 2000",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def passive_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fixed_ids(base_ids, eligible, excluded, count=100):
    """Retain the original engineering sample, with deterministic UID-hash fill."""
    allowed = set(eligible) - set(excluded)
    result = list(dict.fromkeys(uid for uid in base_ids if uid in allowed))[:count]
    candidates = sorted(allowed - set(result), key=lambda uid: hashlib.sha256(("d01-runtime-20261003:" + uid).encode()).hexdigest())
    result.extend(candidates[:count - len(result)])
    if len(result) != count:
        raise ValueError("not enough eligible development UIDs")
    return result, candidates


def header_record(source, member, uid, series, slot, selected):
    """Read just a DICOM header through the existing ZIP index; no pixel decode."""
    import pydicom
    entry = zipfile.ZipInfo(member.path)
    for key, value in json.loads(member.info).items():
        setattr(entry, key, value)
    with source.archive.open(entry, "r") as f:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
    syntax = str(ds.file_meta.TransferSyntaxUID)
    return {"uid": uid, "series": series, "slot": slot, "path": member.path,
            "known_selected_slice": selected, "transfer_syntax": syntax,
            "compressed": bool(ds.file_meta.TransferSyntaxUID.is_compressed),
            "rows": int(getattr(ds, "Rows", 0)), "columns": int(getattr(ds, "Columns", 0)),
            "manufacturer": str(getattr(ds, "Manufacturer", "")),
            "field_strength": str(getattr(ds, "MagneticFieldStrength", ""))}


def scan_headers(adapter, source, ids, series_rows, prior_receipts, extra_candidates, extra_limit=400):
    """One header per chosen series, then bounded compressed-syntax discovery.

    Existing parity receipts identify actually selected slices for the base sample.
    Other series use their lexical middle member, explicitly only a representative.
    Newly observed compressed syntaxes add their study (at most four) to the cohort.
    """
    by_uid = collections.defaultdict(list)
    for row in series_rows:
        by_uid[row[UID]].append(row)
    prior = {(s["uid"], slot.get("series")): slot.get("selected_sources", [])
             for s in prior_receipts for slot in s.get("slots", [])}
    slots = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", -1)]
    records, failures, additions, seen_compressed = [], [], [], set()
    planned = list(ids) + [u for u in extra_candidates if u not in ids][:extra_limit]
    scanned = []
    for uid in planned:
        used = set()
        scanned.append(uid)
        for slot, (plane, fluid) in enumerate(slots):
            row = adapter.pick_series(by_uid[uid], plane, fluid, used)
            if row is None:
                continue
            series = row[SERIES]; used.add(series)
            try:
                members = source.members("train", uid, series)
                if not members:
                    raise ValueError("selected metadata series has no files")
                known = prior.get((uid, series), [])
                lookup = {m.path: m for m in members}
                member = lookup.get(known[0]["path"]) if known else None
                selected = member is not None
                member = member or members[len(members) // 2]
                record = header_record(source, member, uid, series, slot, selected)
                record["series_files"] = len(members)
                records.append(record)
                if record["compressed"] and record["transfer_syntax"] not in seen_compressed:
                    seen_compressed.add(record["transfer_syntax"])
                    if uid not in ids and uid not in additions:
                        additions.append(uid)
            except Exception as exc:
                failures.append({"uid": uid, "series": series, "error": f"{type(exc).__name__}: {exc}"})
        if uid not in ids and len(seen_compressed) >= 4:
            break
    return {"records": records, "failures": failures, "scanned_uids": scanned,
            "additional_compressed_uids": additions, "extra_study_limit": extra_limit,
            "scope": "bounded representative header scan, not complete transfer-syntax or ZIP-integrity audit",
            "syntax_counts": dict(collections.Counter(r["transfer_syntax"] for r in records))}


def build(root, out, asset_stage):
    root, out, asset_stage = map(Path, (root, out, asset_stage))
    if out.exists():
        raise FileExistsError("choose a new immutable benchmark directory")
    if digest(asset_stage / "assets/d01-assets-manifest.json") != ASSET_SHA:
        raise ValueError("immutable D01 v1 dataset identity changed")
    parity_path = root / "artifacts/predictions/serving-20261003/cpu-parity100.json"
    failures_path = root / "docs/handoffs/evidence/2026-10/2026-10-03-priority-execution/archive-failures.json"
    folds_path = root / "artifacts/experiments/EXP026_GROUPED_FROZEN_PROBE/folds.csv"
    series_path = root / "storage/competition/train_series.csv"
    public_path = root / "docs/handoffs/evidence/2026-10/2026-10-03-replan/kaggle-public.json"
    sample_path = root / "artifacts/predictions/serving-20261003/R2F0_test3_dicom.json"
    parity, sample = read_json(parity_path), read_json(sample_path)
    # Gold flags only define exclusion; no clinical labels or report text are read.
    eligible = [r[UID] for r in rows(folds_path) if r["is_gold"].lower() == "false"
                and r["quarantined_gold_report_duplicate"].lower() == "false" and int(r["fold"]) >= 0]
    excluded = sorted({r["uid"] for r in read_json(failures_path)})
    base_ids, candidates = fixed_ids(parity["uids"], eligible, excluded)
    sys.path.insert(0, str(asset_stage / "assets/training/src"))
    import raptor_dicom as adapter
    src = sample["input_manifest"]["source"]
    started = time.perf_counter()
    with adapter.IndexedZipDicomSource(src["archive"], src["index"]) as source:
        scan = scan_headers(adapter, source, base_ids, rows(series_path),
                            [r["receipt"] for r in parity["studies"] if r.get("receipt")], candidates)
        public_headers = []
        for study in sample["study_receipts"]:
            for slot in study["slots"]:
                selected = slot.get("selected_sources", [])
                if selected:
                    path = selected[0]["path"]
                    info = source.connection.execute("SELECT info FROM files WHERE path=?", (path,)).fetchone()[0]
                    public_headers.append(header_record(source, adapter.DicomMember(path, info), study["uid"], slot["series"], slot["slot"], True))
    ids = base_ids + scan["additional_compressed_uids"]
    if set(ids) & set(excluded) or not set(ids).issubset(eligible):
        raise AssertionError("cohort eligibility violated")
    sources = {p.relative_to(root).as_posix(): digest(p) for p in
               (parity_path, failures_path, folds_path, series_path, public_path, sample_path)}
    # Preserve the exact public page record/revision containing the official statement.
    official = read_json(public_path)
    def find_pages(value):
        if isinstance(value, dict):
            if isinstance(value.get("content"), str) and "mix of transfer syntaxes" in value["content"]:
                yield value
            else:
                for v in value.values():
                    yield from find_pages(v)
        elif isinstance(value, list):
            for v in value:
                yield from find_pages(v)
    pages = list(find_pages(official))
    if not pages:
        raise ValueError("official transfer-syntax evidence missing")
    selection = {"schema": 1, "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "scope": "engineering development cohort only; no labels, metrics, Gold selection, anchor runtime or submission",
                 "ids": ids, "base_count": len(base_ids), "studies": len(ids), "excluded_corrupt_uids": excluded,
                 "selection": "retain cpu-parity100 eligible UIDs; exclude two known ZIP corruptions; SHA256 deterministic fill to100; append newly discovered compressed-syntax studies",
                 "assets_manifest_sha256": ASSET_SHA, "train_series_sha256": digest(series_path),
                 "source_sha256": sources, "header_scan": scan, "public_test13_headers": public_headers,
                 "official_transfer_syntax_pages": pages,
                 "official_source": {"url": official["competition"]["url"] + "/data",
                                     "competition_id": official["competition"]["id"],
                                     "observed_at": official["observed_at"],
                                     "page_name": "data-description", "revision": None,
                                     "revision_note": "upstream page response has no revision ID; retained snapshot SHA256 is the content identity",
                                     "snapshot_sha256": digest(public_path)},
                 "archive": {"bytes": Path(src["archive"]).stat().st_size, "index_sha256": digest(src["index"]),
                             "identity_scope": "existing indexed archive; full265GB archive hash not recomputed"},
                 "local_header_scan_seconds": round(time.perf_counter() - started, 3),
                 "limits": {"worker_timeout_seconds": 3500, "kaggle_session_seconds": 3600}}
    out.mkdir(parents=True)
    dump(out / "selection-manifest.json", selection)
    source = Path(__file__).read_text(encoding="utf-8")
    # Keep the full header audit locally; only the execution contract belongs
    # inside the notebook (large embedded audits can exceed API code limits).
    execution_selection = dict(selection)
    execution_selection["header_scan"] = {
        "full_local_evidence_sha256": digest(out / "selection-manifest.json"),
        "summary": {key: value for key, value in selection["header_scan"].items()
                    if not isinstance(value, list)},
    }
    execution_selection.pop("public_test13_headers", None)
    execution_selection.pop("official_transfer_syntax_pages", None)
    dump(out / "execution-manifest.json", execution_selection)
    selection_text = (out / "execution-manifest.json").read_text(encoding="utf-8")
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    select_sha = hashlib.sha256(selection_text.encode()).hexdigest()
    code = f'''from pathlib import Path
import hashlib, os, subprocess, sys, json, time
boot_started = time.perf_counter()
source = {source!r}
selection = {selection_text!r}
assert hashlib.sha256(source.encode()).hexdigest() == {source_sha!r}
assert hashlib.sha256(selection.encode()).hexdigest() == {select_sha!r}
base = Path('/kaggle/working/d01-runtime-benchmark')
base.mkdir(exist_ok=False)
(base/'benchmark.py').write_text(source, encoding='utf-8')
(base/'selection-manifest.json').write_text(selection, encoding='utf-8')
hits = list(Path('/kaggle/input').glob('*/d01-assets-manifest.json')) + list(Path('/kaggle/input').glob('datasets/*/*/d01-assets-manifest.json'))
hits = sorted(set(p.resolve() for p in hits if hashlib.sha256(p.read_bytes()).hexdigest() == {ASSET_SHA!r}))
assert len(hits) == 1, 'pinned D01 v1 assets missing'
competition = next((p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p/'train_series.csv').is_file()), None)
assert competition is not None
env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
try:
    result = subprocess.run([sys.executable, str(base/'benchmark.py'), 'run', '--assets', str(hits[0].parent), '--competition', str(competition), '--selection', str(base/'selection-manifest.json'), '--work', str(base/'results')], check=True, timeout=3500, env=env)
    state = 'COMPLETE'
except Exception as exc:
    state = 'FAILED'
    (base/'launcher-failure.json').write_text(json.dumps({{'status':state, 'error':str(exc), 'elapsed_seconds':time.perf_counter()-boot_started}}, indent=2))
    raise
finally:
    (base/'launcher-timing.json').write_text(json.dumps({{'status':state, 'notebook_cell_seconds':time.perf_counter()-boot_started, 'scope':'starts at benchmark cell, excludes Kaggle provisioning'}}, indent=2))
'''
    notebook = {"nbformat": 4, "nbformat_minor": 4, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}}, "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# D01 R2 runtime and decoder benchmark\n", "Fixed development engineering cohort; no accuracy measurement or submission. Existing D01 assets are reused.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)}]}
    metadata = read_json(asset_stage / "preflight/kernel-metadata.json")
    metadata.update(id="taeyangg4/knee-d01-r2-runtime100", title="knee-d01-r2-runtime100", code_file="d01-runtime100.ipynb")
    dump(out / "kernel-metadata.json", metadata)
    dump(out / metadata["code_file"], notebook)
    dump(out / "package-review.json", {"state": "STAGED_ONLY", "studies": len(ids), "source_sha256": source_sha,
         "selection_sha256": select_sha, "assets_manifest_sha256": ASSET_SHA,
         "command_not_executed": f'.venv/Scripts/kaggle.exe kernels push -p "{out.resolve()}" --accelerator NvidiaTeslaT4 --timeout 3600',
         "budget": "at most60min session (2T4 allocation may count two GPU-hours); worker58m20s; expected duration unmeasured",
         "scope_limit": "R2 engineering benchmark only; anchor49+ branch and total candidate9h remain unverified"})
    return selection


def decoder_capabilities():
    from pydicom.pixels import get_decoder
    result = {}
    for uid, name in SYNTAXES.items():
        try:
            decoder = get_decoder(uid)
            result[uid] = {"name": name, "available": bool(decoder.is_available),
                           "available_plugins": list(decoder.available_plugins),
                           "missing_dependencies": list(decoder.missing_dependencies)}
        except Exception as exc:
            result[uid] = {"name": name, "available": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


class RSSMonitor:
    def __init__(self):
        self.peak = 0; self.stop = threading.Event(); self.samples = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        import psutil
        process = psutil.Process()
        while not self.stop.is_set():
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak, total); self.samples += 1
            self.stop.wait(0.1)

    def __enter__(self):
        import psutil  # Fail visibly if the base image lacks required instrumentation.
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set(); self.thread.join(timeout=2)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "total": sum(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": values[min(len(values)-1, int(.95*len(values)))], "max": max(values)}


def run(assets, competition, selection_path, work):
    """Remote-only explicit execution. Does not read train.csv or clinical targets."""
    started = time.perf_counter()
    assets, competition, selection_path, work = map(Path, (assets, competition, selection_path, work))
    work.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "stage": "asset_validation", "selection_sha256": digest(selection_path),
              "assets_manifest_sha256": ASSET_SHA, "timings": {}, "failures": [],
              "scope": "fixed development runtime/decoder check; not independent quality or full anchor runtime validation"}
    monitor = RSSMonitor(); hooks = []; torch = None; source = None; corpus = None
    def progress(stage):
        report["stage"] = stage
        report["elapsed_seconds"] = time.perf_counter() - started
        dump(work / "benchmark-receipt.json", report)
        print(stage, f"{report['elapsed_seconds']:.1f}s", flush=True)
    try:
        with monitor:
            selection = read_json(selection_path)
            if digest(assets / "d01-assets-manifest.json") != ASSET_SHA or selection["assets_manifest_sha256"] != ASSET_SHA:
                raise ValueError("D01 asset/selection identity mismatch")
            package = passive_module(assets / "d01_package.py", "d01_pinned_package")
            manifest = package._validate_assets(assets)
            report["timings"]["asset_validation"] = time.perf_counter() - started
            phase = time.perf_counter(); progress("offline_environment")
            package._activate_offline_environment(assets, work)
            report["timings"]["offline_environment"] = time.perf_counter() - phase
            import numpy as np
            import pydicom
            import torch as torch_module
            torch = torch_module
            from raptor_common import environment, write_predictions
            from raptor_dicom import DicomCorpus, DirectoryDicomSource
            from raptor_ensemble import load_members, predict_ensemble
            if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
                raise RuntimeError("actual Kaggle T4 required")
            torch.set_num_threads(2)
            report["environment"] = environment()
            report["decoder_capabilities"] = decoder_capabilities()
            report["all_official_syntaxes_available"] = all(v["available"] for v in report["decoder_capabilities"].values())
            if digest(competition / "train_series.csv") != selection["train_series_sha256"]:
                raise ValueError("training series metadata identity changed")
            ids = selection["ids"]
            if len(set(ids)) != len(ids) or set(ids) & set(selection["excluded_corrupt_uids"]):
                raise ValueError("invalid fixed cohort")
            selected_rows = [{key: row[key] for key in (UID, SERIES, "Fluid_Sensitive", "Anatomical_Plane")}
                             for row in rows(competition / "train_series.csv") if row[UID] in set(ids)]
            observed = {}; bytes_read = 0; read_calls = 0
            class ObservedSource(DirectoryDicomSource):
                def read(self, member):
                    nonlocal bytes_read, read_calls
                    raw = super().read(member); bytes_read += len(raw); read_calls += 1
                    if member.path not in observed:
                        ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
                        observed[member.path] = {"transfer_syntax": str(ds.file_meta.TransferSyntaxUID),
                                                 "compressed": bool(ds.file_meta.TransferSyntaxUID.is_compressed)}
                    return raw
            decode_times, study_times = {}, {}
            active = {"uid": None, "start": None}
            class TimedCorpus(DicomCorpus):
                def _ensure(self, uid):
                    fresh = self._cached_uid != str(uid)
                    if fresh:
                        active.update(uid=str(uid), start=time.perf_counter())
                    result = super()._ensure(uid)
                    if fresh:
                        decode_times[str(uid)] = time.perf_counter() - active["start"]
                    return result
            phase = time.perf_counter(); progress("model_loading")
            members, _ = load_members(assets / "d01-assets-manifest.json")
            report["timings"]["model_loading"] = time.perf_counter() - phase
            if len(members) != 5:
                raise ValueError("five R2 members required")
            def final_member_hook(module, inputs, output):
                torch.cuda.synchronize()
                study_times[active["uid"]] = time.perf_counter() - active["start"]
                if len(study_times) % 10 == 0:
                    report["completed_studies"] = len(study_times)
                    progress("inference")
            hooks.append(members[-1].model.register_forward_hook(final_member_hook))
            torch.cuda.reset_peak_memory_stats()
            phase = time.perf_counter(); progress("inference")
            with ObservedSource(competition / "train_series") as source:
                corpus = TimedCorpus.from_records(ids, selected_rows, source, manifest["contract"], split="train")
                mean, raw, inference = predict_ensemble(members, corpus, amp="fp16", device="cuda", batch=1,
                                                        residency="resident", runtime_policy="strict")
            torch.cuda.synchronize()
            report["timings"]["inference_including_model_placement_offload"] = time.perf_counter() - phase
            if raw.shape != (5, len(ids), 12) or not np.isfinite(raw).all():
                raise RuntimeError("incomplete/nonfinite five-model inference")
            report["inference"] = inference
            report["study_receipts"] = [corpus.receipts[uid] for uid in ids]
            report["dicom_io"] = {"bytes_read": bytes_read, "read_calls": read_calls,
                                   "distinct_header_paths": len(observed),
                                   "scope": "includes header and pixel read; extra syntax header parsing measured in runtime"}
            compressed_cases = []
            for receipt in report["study_receipts"]:
                for slot in receipt["slots"]:
                    for item in slot.get("selected_sources", []):
                        # Directory source paths omit the train_series prefix.
                        obs = observed.get(item["path"])
                        if obs and obs["compressed"]:
                            compressed_cases.append({"uid": receipt["uid"], "series": slot["series"], **item, **obs,
                                                     "pixel_decode": "SUCCESS", "evidence": "successful prepare_study selected_sources"})
            report["compressed_selected_decode_cases"] = compressed_cases
            report["actual_compressed_syntax_counts"] = dict(collections.Counter(r["transfer_syntax"] for r in compressed_cases))
            report["actual_decode_coverage"] = "compressed selected slices decoded" if compressed_cases else "no compressed selected slice observed; bounded scan is not proof of absence"
            report["per_study"] = [{"uid": uid, "decode_seconds": decode_times[uid],
                                    "decode_through_last_model_seconds": study_times[uid]} for uid in ids]
            report["study_time_distribution"] = distribution(list(study_times.values()))
            report["decode_time_distribution"] = distribution(list(decode_times.values()))
            report["runtime_projection"] = {"studies": 1300,
                "seconds_at_sample_mean_plus_startup": 1300 * statistics.mean(study_times.values()) + phase - started,
                "scope": "engineering scaling estimate only; sample is intentionally enriched, excludes anchor and cannot establish hidden total runtime"}
            write_predictions(work / "engineering-raw-mean.csv", ids, mean)
            np.savez_compressed(work / "engineering-members.npz", study_uids=np.asarray(ids), member_probabilities=raw)
            report.update(status="COMPLETE", completed_studies=len(ids), all_finite=True,
                          caveats=["development studies may have trained these models; no metric is computed", "anchor49+ and total candidate runtime not measured", "GPU timings include instrumentation and cold first forward"])
    except Exception as exc:
        report.update(status="FAILED", all_finite=False)
        failure = {"stage": report["stage"], "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        if hasattr(exc, "receipt"):
            failure["inference_receipt"] = exc.receipt
        if corpus is not None:
            failure["study_receipts"] = list(corpus.receipts.values())
        report["failures"].append(failure)
        raise
    finally:
        for hook in hooks:
            hook.remove()
        if torch is not None and torch.cuda.is_initialized():
            report["cuda_memory"] = {"max_allocated_bytes": torch.cuda.max_memory_allocated(),
                                      "max_reserved_bytes": torch.cuda.max_memory_reserved(), "device": 0}
        report["rss"] = {"peak_parent_plus_children_bytes": monitor.peak, "samples": monitor.samples, "interval_seconds": .1}
        report["timings"]["worker_total_seconds"] = time.perf_counter() - started
        dump(work / "benchmark-receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    command = sub.add_parser("build")
    command.add_argument("--root", default=str(ROOT)); command.add_argument("--out", required=True)
    command.add_argument("--asset-stage", default=str(ROOT / ".local/d01-staging/20261003-v1"))
    command = sub.add_parser("run")
    for name in ("assets", "competition", "selection", "work"):
        command.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.action == "build":
        result = build(args.root, args.out, args.asset_stage)
        print(json.dumps({"studies": result["studies"], "syntax_counts": result["header_scan"]["syntax_counts"]}))
    else:
        run(args.assets, args.competition, args.selection, args.work)


if __name__ == "__main__":
    main()
