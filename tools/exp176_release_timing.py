"""EXP176 v1.1.0: offline inventory/stage/push-plan/collect/report for EXP169 timing.

No CLI command starts a GPU, uploads, downloads or submits. The generated private
notebook alone calls run_benchmark on Kaggle 2xT4. Serving sources stay immutable.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePath, PurePosixPath
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile
import zlib

try:
    from tools import release_cohort_check as cohort
except ModuleNotFoundError:
    import release_cohort_check as cohort

VERSION = "1.1.0"
INVENTORY_VERSION = "dicom-filenames-v1"
SELECTION = "image-total-and-longest-series-quartiles-v1"
ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/releases/EXP169_PARTNER_BUNDLE_A1_stage/kernel/release.ipynb"
RELEASE_SHA = "e96a6c854751efd1684c25553a61ed49a1c70217a814fb0df3c25dddc8d39f75"
PARTNER_SHA = "c48aac04128f524116097077a7f1ffd1365f421e3af123c91bb228da09d51665"
KERNEL = "taeyangg4/knee-exp176-release-timing"
DATASETS = ["taeyangg4/knee-exp169-partner-bundle-assets",
            "taeyangg4/knee-exp155-own0949-oai-v1-2-assets",
            "nartaa/rsna-knee-publication-swa-weights-20261007"]
GUARD = 24300
KAGGLE_LIMIT = 32400
SERIES_FIELDS = ["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
                 "Fat_Suppression", "Anatomical_Plane"]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = cohort.UID
_BENCH = None
_ORIGINAL_PRE_ONE = None


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jwrite(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def fresh(path):
    path = Path(path).resolve()
    need(not path.exists(), "Fresh output directory required: " + str(path))
    path.mkdir(parents=True)
    return path


def read_series(path):
    path = Path(path)
    need(path.name == "train_series.csv", "Only train_series.csv may select the benchmark")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames == SERIES_FIELDS, "Unexpected train_series schema; review before proceeding")
        rows = list(reader)
    seen = set()
    for row in rows:
        cohort.identity([row[UID]])
        cohort.identity([row["SeriesInstanceUID"]])
        key = (row[UID], row["SeriesInstanceUID"])
        need(key not in seen, "Duplicate series row")
        seen.add(key)
    need(rows, "Empty train_series.csv")
    return rows


def safe_component(value):
    cohort.identity([value])
    need(":" not in value and not any(ord(c) < 32 for c in value), "Unsafe filename/UID component")
    return value


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def filename_record(uid, series, names):
    names = sorted(names)
    need(names and len(names) == len(set(names)), "Empty or duplicate DICOM filename inventory")
    for name in names:
        safe_component(name)
        need(name.endswith(".dcm"), "Expected lowercase .dcm filenames used by serving")
    return {UID: uid, "SeriesInstanceUID": series, "image_count": len(names),
            "filenames_sha256": hashlib.sha256(("\n".join(names)+"\n").encode("utf-8")).hexdigest()}


def validate_inventory(value, rows=None, series_sha=None):
    need(isinstance(value, dict) and set(value) == {"schema", "train_series_sha256", "series"},
         "Invalid image inventory schema")
    need(value["schema"] == INVENTORY_VERSION, "Unsupported image inventory version")
    def is_sha(x):
        return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)
    need(is_sha(value["train_series_sha256"]), "Invalid train series SHA256")
    need(series_sha is None or value["train_series_sha256"] == series_sha, "Inventory train series file drift")
    need(isinstance(value["series"], list) and value["series"], "Empty image inventory")
    seen, records = set(), []
    for r in value["series"]:
        need(isinstance(r, dict) and set(r) == {UID, "SeriesInstanceUID", "image_count", "filenames_sha256"},
             "Invalid series inventory schema")
        key = (safe_component(r[UID]), safe_component(r["SeriesInstanceUID"]))
        need(key not in seen, "Duplicate inventory series")
        need(type(r["image_count"]) is int and r["image_count"] > 0, "Invalid image count")
        need(is_sha(r["filenames_sha256"]), "Invalid filename SHA256")
        seen.add(key)
        records.append(dict(r))
    if rows is not None:
        expected = {(r[UID], r["SeriesInstanceUID"]) for r in rows}
        need(len(expected) == len(rows) and seen == expected, "Inventory/metadata series membership mismatch")
    return {"schema": INVENTORY_VERSION, "train_series_sha256": value["train_series_sha256"],
            "series": sorted(records, key=lambda r: (r[UID], r["SeriesInstanceUID"]))}


def scan_inventory(rows, images, series_sha):
    """Enumerate names only: directories or a ZIP central directory; never DICOM bytes.

    Directory input is the train image root. ZIPs must use train_series/ or
    train_images/ at archive root, as in the official competition archive.
    """
    images = Path(images).resolve()
    keys = {(safe_component(r[UID]), safe_component(r["SeriesInstanceUID"])) for r in rows}
    need(len(keys) == len(rows), "Duplicate metadata series")
    records = []
    if images.is_dir():
        for uid, sid in sorted(keys):
            directory = images / uid / sid
            need(directory.resolve().is_relative_to(images) and directory.is_dir(),
                 "Missing/escaping train series directory")
            names = []
            for path in directory.iterdir():
                if path.name.endswith(".dcm"):
                    need(path.resolve().is_relative_to(images) and path.is_file(),
                         "Missing/escaping DICOM filename")
                    names.append(path.name)
            records.append(filename_record(uid, sid, names))
    else:
        need(images.is_file() and images.suffix.lower() == ".zip", "Expected train image directory or ZIP")
        grouped, roots = defaultdict(list), set()
        with zipfile.ZipFile(images) as archive:
            for info in archive.infolist():
                # Ignore test data/metadata without opening any archive member.
                raw = info.filename
                if raw.split("/", 1)[0] not in ("train_series", "train_images"):
                    continue
                roots.add(raw.split("/", 1)[0])
                if info.is_dir():
                    continue
                parts = raw.split("/")
                need(len(parts) == 4, "Unexpected train archive path")
                for part in parts:
                    safe_component(part)
                if not parts[-1].endswith(".dcm"):
                    continue
                key = tuple(parts[1:3])
                need(key in keys, "Archive/metadata series membership mismatch")
                need((info.external_attr >> 16) & 0o170000 != 0o120000, "Symlink in train archive")
                grouped[key].append(parts[-1])
        need(len(roots) == 1, "Expected one unambiguous train archive root")
        need(set(grouped) == keys, "Archive/metadata series membership mismatch")
        records = [filename_record(*key, grouped[key]) for key in sorted(keys)]
    return validate_inventory({"schema": INVENTORY_VERSION, "train_series_sha256": series_sha,
                               "series": records}, rows, series_sha)


def inventory(images, out, series=ROOT / "storage/competition/train_series.csv"):
    rows = read_series(series)
    value = scan_inventory(rows, images, sha(series))
    destination = fresh(out)
    jwrite(destination / "image-inventory.json", value)
    return {"status": "REGISTERED_FILENAME_INVENTORY", "series": len(value["series"]),
            "inventory_sha256": digest_json(value), "file_sha256": sha(destination / "image-inventory.json")}


def study_sizes(value):
    sizes = defaultdict(lambda: {"series_count": 0, "total_images": 0, "max_series_images": 0})
    for r in value["series"]:
        s = sizes[r[UID]]
        s["series_count"] += 1
        s["total_images"] += r["image_count"]
        s["max_series_images"] = max(s["max_series_images"], r["image_count"])
    return dict(sizes)


def coverage_requirements(sizes, n):
    need(type(n) is int and 40 <= n <= len(sizes), "Benchmark requires >=40 distinct train studies")
    q = n // 4
    result = {}
    for metric in ("total_images", "max_series_images"):
        ordered = sorted(sizes, key=lambda u: (sizes[u][metric], u))
        result[metric] = {
            "spread_ids": [ordered[round(i*(len(ordered)-1)/(q-1))] for i in range(q)],
            "tail_ids": sorted(sizes, key=lambda u: (-sizes[u][metric], u))[:q]}
    return result


def coverage_check(ids, sizes):
    cohort.identity(ids)
    need(set(ids) <= set(sizes), "Unknown inventory study")
    required = coverage_requirements(sizes, len(ids))
    for metric, groups in required.items():
        for kind, anchors in groups.items():
            need(set(anchors) <= set(ids), "Missing image coverage: " + metric + "/" + kind)
    return {"status": "PASS_FILENAME_IMAGE_COUNTS_AND_LONG_SERIES", "requirements": required,
            "ranges": {metric: {
                "population": [min(s[metric] for s in sizes.values()), max(s[metric] for s in sizes.values())],
                "selected": [min(sizes[u][metric] for u in ids), max(sizes[u][metric] for u in ids)]}
                for metric in required}}


def choose_cohort(value, n=48):
    value = validate_inventory(value)
    sizes = study_sizes(value)
    required = coverage_requirements(sizes, n)
    selected = set()
    for groups in required.values():
        selected.update(groups["spread_ids"])
        selected.update(groups["tail_ids"])
    # Overlap between the four reservations is filled in a deterministic round
    # robin over both descending metrics; never substitute series-count proxies.
    orders = [sorted(sizes, key=lambda u: (-sizes[u][metric], u)) for metric in required]
    for pair in zip(*orders):
        for uid in pair:
            if len(selected) < n:
                selected.add(uid)
    ids = sorted(selected)
    return {**cohort.identity(ids), "ids": ids, "scope": "LABEL_FREE_TRAIN_TIMING_SAMPLE",
            "selection": SELECTION, "inventory_sha256": digest_json(value),
            "train_series_sha256": value["train_series_sha256"],
            "image_count_coverage": coverage_check(ids, sizes),
            "study_sizes": {u: sizes[u] for u in ids},
            "series_counts": {u: sizes[u]["series_count"] for u in ids},
            "population_studies": len(sizes), "population_series": len(value["series"]),
            "population_series_count_min_max": [min(s["series_count"] for s in sizes.values()),
                                                  max(s["series_count"] for s in sizes.values())],
            "selected_series_count_min_max": [min(sizes[u]["series_count"] for u in ids),
                                               max(sizes[u]["series_count"] for u in ids)]}


def check_inventory(value, intended):
    expected = choose_cohort(value, intended["studies"])
    need(intended == expected, "Inventory/cohort/coverage binding mismatch")
    return {"status": "MATCH_REGISTERED_FILENAME_INVENTORY", "inventory_sha256": expected["inventory_sha256"],
            "coverage": expected["image_count_coverage"], **cohort.identity(expected["ids"])}


def write_csv(path, fields, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_view(competition, out, intended):
    """Whole image root symlink preserves DirectoryDicomSource containment checks.

    Only selected CSV rows are visible to both consumers. Per-study symlinks
    escaping a physical test_series directory are deliberately NOT used.
    """
    competition, out = Path(competition).resolve(), Path(out)
    rows = read_series(competition / "train_series.csv")
    need(sha(competition / "train_series.csv") == intended["train_series_sha256"], "Train series file drift")
    selected = [r for r in rows if r[UID] in set(intended["ids"])]
    cohort.check(sorted({r[UID] for r in selected}), intended)
    roots = [competition / "train_series", competition / "train_images"]
    available = [p for p in roots if p.is_dir()]
    need(len(available) == 1, "Expected one unambiguous competition train image root")
    images = available[0].resolve()
    value = scan_inventory(rows, images, sha(competition / "train_series.csv"))
    checked = check_inventory(value, intended)  # All population filenames, before any inference.
    out = fresh(out)
    (out / "test_series").symlink_to(images, target_is_directory=True)
    write_csv(out / "test.csv", [UID], [{UID: u} for u in intended["ids"]])
    write_csv(out / "test_series.csv", SERIES_FIELDS, selected)
    write_csv(out / "sample_submission.csv", [UID, *TARGETS],
              [{UID: u, **{t: 0 for t in TARGETS}} for u in intended["ids"]])
    jwrite(out / "intended-cohort.json", intended)
    jwrite(out / "image-inventory.json", value)
    jwrite(out / "inventory-check.json", checked)
    return out


def _kaggle_path(value):
    """Keep Kaggle paths POSIX even when supplied as a Windows Path."""
    return PurePosixPath(value.as_posix() if isinstance(value, PurePath) else value)


def rewrite_cell(source, view):
    """Only route inputs and insert observation hooks; exact release tool is untouched."""
    view = _kaggle_path(view)
    original = "Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')"
    need(source.count(original) == 2, "Release competition routing changed")
    source = source.replace(original, "Path(" + repr(str(view)) + ")")
    original_sample = "Path('/kaggle/input/competitions/rsna-knee-abnormality-detection/sample_submission.csv')"
    need(source.count(original_sample) == 1, "Release sample routing changed")
    source = source.replace(original_sample, "Path(" + repr(str(view / "sample_submission.csv")) + ")")
    need(source.count("\nmain()\n") == 1, "Release trio entrypoint changed")
    source = source.replace("\nmain()\n", "\n_bench.install_trio(globals(), _e152)\n_bench.call('trio', main)\n")
    anchor = "    _pilot_dir = Path('/kaggle/working/exp169-pilot')"
    need(source.count(anchor) == 1, "Release partner hook changed")
    return source.replace(anchor, "    _bench.install_partner(_e169)\n" + anchor)


def code_cell(notebook):
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    need(len(cells) == 1, "Expected exact EXP169 single code cell")
    return "".join(cells[0]["source"])


def stage(out, inventory, studies=48, hidden_studies=1300, series=ROOT / "storage/competition/train_series.csv"):
    need(isinstance(hidden_studies, int) and hidden_studies >= studies, "Invalid assumed hidden size")
    need(sha(RELEASE) == RELEASE_SHA, "Exact EXP169 v1.0.2 notebook changed; do not silently rebind")
    need(sha(ROOT / "tools/exp169_partner_bundle_release.py") == PARTNER_SHA, "Hash-bound release tool changed")
    value = validate_inventory(jread(inventory), read_series(series), sha(series))
    intended = choose_cohort(value, studies)
    config = {"version": VERSION, "release_sha256": RELEASE_SHA, "partner_tool_sha256": PARTNER_SHA,
              "intended": intended, "assumed_hidden_studies": hidden_studies,
              "guard_seconds": GUARD, "kaggle_limit_seconds": KAGGLE_LIMIT}
    original = RELEASE.read_bytes()
    source = code_cell(json.loads(original))
    rewrite_cell(source, PurePosixPath("/kaggle/working/exp176-cohort"))  # fail before creating stage
    metadata = jread(RELEASE.with_name("kernel-metadata.json"))
    need(metadata["dataset_sources"] == DATASETS, "Unexpected attached release datasets")
    metadata.update(id=KERNEL, title=KERNEL.split("/")[1], code_file="benchmark.ipynb")
    need(metadata["is_private"] and not metadata["enable_internet"], "Private offline kernel required")
    out = fresh(out)
    kernel = out / "kernel"
    kernel.mkdir()
    payload = {"exp176_release_timing.py": Path(__file__).read_text(encoding="utf-8"),
               "release_cohort_check.py": Path(cohort.__file__).read_text(encoding="utf-8"),
               "exp169-release.ipynb": original.decode("utf-8"), "config.json": json.dumps(config)}
    encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    bootstrap = ("import base64, hashlib, json, sys, zlib\nfrom pathlib import Path\n"
                 "_p = Path('/kaggle/working/exp176-wrapper'); _p.mkdir(exist_ok=False)\n"
                 f"_files = json.loads(zlib.decompress(base64.b64decode({encoded!r})))\n"
                 "for _name, _value in _files.items(): (_p/_name).write_text(_value, encoding='utf-8')\n"
                 "sys.path.insert(0, str(_p))\nimport exp176_release_timing as bench\n"
                 "bench.run_benchmark(_p)\n")
    notebook = {"nbformat": 4, "nbformat_minor": 4,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                           f"# EXP176 v{VERSION} — TRAIN timing only; never submit\nExact EXP169 v1.0.2 assets, K36, fixed weights. 48+ recommended; all qualification is sample-only."},
                          {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": bootstrap}]}
    jwrite(kernel / "benchmark.ipynb", notebook)
    jwrite(kernel / "kernel-metadata.json", metadata)
    jwrite(out / "config.json", config)
    jwrite(out / "intended-cohort.json", intended)
    jwrite(out / "image-inventory.json", value)
    shutil.copyfile(RELEASE, out / "exp169-release.ipynb")
    files = ["config.json", "intended-cohort.json", "image-inventory.json", "exp169-release.ipynb", "kernel/benchmark.ipynb", "kernel/kernel-metadata.json"]
    receipt = {"status": "STAGED_NOT_PUSHED", "version": VERSION, "config": config,
               "files": {f: sha(out/f) for f in files},
               "source_sha256": {"benchmark": sha(__file__), "cohort_helper": sha(cohort.__file__)}}
    need((kernel / "benchmark.ipynb").stat().st_size < 1000000, "Notebook exceeds 1MB budget")
    jwrite(out / "build-receipt.json", receipt)
    return receipt


def checked_stage(path):
    path = Path(path).resolve()
    r = jread(path / "build-receipt.json")
    need(r["status"] == "STAGED_NOT_PUSHED" and r["version"] == VERSION, "Invalid stage receipt")
    for name, digest in r["files"].items():
        need(sha(path/name) == digest, "Stage drift: " + name)
    check_inventory(jread(path / "image-inventory.json"), r["config"]["intended"])
    need(sha(__file__) == r["source_sha256"]["benchmark"] and
         sha(cohort.__file__) == r["source_sha256"]["cohort_helper"], "Wrapper changed after staging")
    return r


def psquote(value):
    return "'" + str(value).replace("'", "''") + "'"


def push_plan(stage_dir):
    path = Path(stage_dir).resolve()
    checked_stage(path)
    py = psquote(ROOT / ".venv/Scripts/python.exe")
    return {"status": "PLAN_ONLY_NOT_EXECUTED", "commands": [
        "# B checks CPU tests, live quota and reserved Kaggle lane before executing.",
        f"& {py} -m kaggle kernels push -p {psquote(path/'kernel')} --accelerator NvidiaTeslaT4",
        "# Record the returned version. After COMPLETE, download only the bounded result archive:",
        f"& {py} -X utf8 tools/kaggle_output_file.py --kernel {KERNEL} --version <VERSION> --file exp176-results.zip --out {psquote(path/'exp176-results.zip')} --max-bytes 16777216",
        f"& {py} tools/exp176_release_timing.py collect --stage {psquote(path)} --archive {psquote(path/'exp176-results.zip')} --out {psquote(path/'collected')}",
        f"& {py} tools/exp176_release_timing.py report --collected {psquote(path/'collected')} --out {psquote(path/'report')}",
        "# Timing benchmark only. No submission command exists."]}


def _trio_pre_one(uid):
    # Top-level pickleable function, inherited by unchanged fork pool workers.
    started = time.perf_counter()
    result = _ORIGINAL_PRE_ONE(uid)
    _BENCH.event("trio_decode", started, uid=uid)
    return result


class Observer:
    """Timing-only delegation; no tensor conversions, extra CUDA sync or numerical edits."""
    def __init__(self, out, view):
        self.out, self.view = Path(out), _kaggle_path(view)
        self.events = self.out / "events"
        self.events.mkdir()
        self.local = threading.local()
        self.phase = "setup"
        self.lock = threading.Lock()
        self.stages = {}

    def event(self, kind, started, **fields):
        record = {"kind": kind, "phase": self.phase, "seconds": time.perf_counter()-started,
                  "pid": os.getpid(), "thread": threading.get_ident(), **fields}
        data = (json.dumps(record, allow_nan=False) + "\n").encode()
        # One atomic append per event; forked decoders each have a distinct file.
        fd = os.open(self.events / f"{os.getpid()}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)

    def call(self, name, fn, *args, **kwargs):
        prior, self.phase = self.phase, name
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.stages[name] = time.perf_counter()-started
            self.phase = prior

    def wrap(self, obj, name, kind, fields=None):
        original = obj[name] if isinstance(obj, dict) else getattr(obj, name)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            started = time.perf_counter()
            result = original(*args, **kwargs)
            extra = fields(args, kwargs) if fields else {}
            self.event(kind, started, **extra)
            return result
        if isinstance(obj, dict):
            obj[name] = observed
        else:
            setattr(obj, name, observed)
        return original

    def install_trio(self, ns, adapter):
        global _ORIGINAL_PRE_ONE
        ns["find_test_root"] = lambda: str(self.view)
        _ORIGINAL_PRE_ONE, ns["_pre_one"] = ns["_pre_one"], _trio_pre_one
        self.wrap(ns, "load_model", "trio_model_load", lambda a, k: {"gpu": str(a[3])})
        # input_record receives volume, not UID; UID comes from its consumer frame.
        original = ns["_E152"].input_record
        @functools.wraps(original)
        def input_record(*args, **kwargs):
            frame = inspect.currentframe().f_back
            self.local.uid = frame.f_locals["sid"]
            del frame
            return original(*args, **kwargs)
        ns["_E152"].input_record = input_record
        self.wrap(ns, "infer_pair_probs", "trio_inference", lambda a, k:
                  {"uid": self.local.uid, "gpu": str(a[3])})
        self.wrap(ns, "gpu_windows", "trio_prepare_enqueue")
        self.wrap(ns["_E152"], "output_values", "trio_aggregation")
        self.wrap(ns["_E152"], "save_raw", "trio_packaging")

    def install_partner(self, module):
        import raptor_dicom
        import raptor_runtime
        need(sha(module.__file__) == PARTNER_SHA, "Executing partner source hash mismatch")
        self.wrap(module, "_load_models", "partner_model_load", lambda a, k: {"gpu": str(a[2])})
        volume, mask = raptor_dicom.DicomCorpus.volume, raptor_dicom.DicomCorpus.mask
        def timed_volume(corpus_obj, uid):
            started = time.perf_counter()
            result = volume(corpus_obj, uid)
            self.local.uid = uid
            self.local.normalizations = 0
            self.event("partner_decode", started, uid=uid)
            return result
        def timed_mask(corpus_obj, uid):
            result = mask(corpus_obj, uid)
            self.local.prepare_start = time.perf_counter()
            return result
        raptor_dicom.DicomCorpus.volume = timed_volume
        raptor_dicom.DicomCorpus.mask = timed_mask
        normalize = raptor_runtime.normalize_input
        def timed_normalize(*args, **kwargs):
            result = normalize(*args, **kwargs)
            self.local.normalizations += 1
            if self.local.normalizations == 2:
                self.event("partner_prepare", self.local.prepare_start, uid=self.local.uid)
            return result
        raptor_runtime.normalize_input = timed_normalize
        original_infer = module._infer_one
        def timed_infer(*args, **kwargs):
            # i is the immutable global study index in the unchanged consumer.
            frame = inspect.currentframe().f_back
            index = frame.f_locals["i"]
            outer = frame.f_locals["ids"] if "ids" in frame.f_locals else None
            # consumer closes over scores, but not ids; current partner call supplies it.
            uid = self.partner_ids[index] if outer is None else outer[index]
            del frame
            started = time.perf_counter()
            result = original_infer(*args, **kwargs)
            self.event("partner_gpu", started, uid=uid, gpu=str(args[2]))
            return result
        module._infer_one = timed_infer
        original_scores = module.partner_scores
        def scores(*args, **kwargs):
            self.partner_ids = list(args[2])
            return self.call("partner_" + ("pilot_scores" if self.phase == "pilot" else "full"), original_scores, *args, **kwargs)
        module.partner_scores = scores
        original_pilot, original_serve = module.pilot, module.serve
        module.pilot = lambda *a, **k: self.call("pilot", original_pilot, *a, **k)
        # serve's default function was captured at definition time; pass wrapped original explicitly.
        module.serve = lambda *a, **k: self.call("serve", original_serve, *a, partner=scores, **k)
        self.wrap(module, "aggregate", "partner_aggregation")
        self.wrap(module, "final_values", "final_aggregation")
        self.wrap(module, "write_csv", "prediction_packaging")


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.error = None

    def start(self):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                processes = [self.process, *self.process.children(recursive=True)]
                rss = 0
                for p in processes:
                    try:
                        rss += p.memory_info().rss
                    except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                        pass
                vm = self.psutil.virtual_memory()
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=5)
                gpu = [[int(v.strip()) for v in row.split(",")] for row in raw.strip().splitlines()]
                self.samples.append({"rss_tree_bytes": rss, "system_used_bytes": vm.total-vm.available,
                                     "system_total_bytes": vm.total, "gpu_mib": gpu,
                                     "phase": _BENCH.phase if _BENCH is not None else "setup"})
                self.stop.wait(1)
        except Exception as exc:
            self.error = type(exc).__name__ + ": " + str(exc)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=7)
        need(not self.thread.is_alive() and not self.error and self.samples, "Memory telemetry incomplete: " + str(self.error))
        by_phase = {}
        for phase in sorted({x["phase"] for x in self.samples}):
            samples = [x for x in self.samples if x["phase"] == phase]
            by_phase[phase] = {"samples": len(samples),
                               "peak_process_tree_rss_bytes": max(x["rss_tree_bytes"] for x in samples),
                               "peak_system_used_bytes": max(x["system_used_bytes"] for x in samples),
                               "peak_device_used_bytes": [max(next(r[1] for r in x["gpu_mib"] if r[0] == i) for x in samples)*2**20 for i in range(2)]}
        return {"interval_seconds": 1, "samples": len(self.samples), "by_phase": by_phase,
                "peak_process_tree_rss_bytes": max(x["rss_tree_bytes"] for x in self.samples),
                "peak_system_used_bytes": max(x["system_used_bytes"] for x in self.samples),
                "system_total_bytes": min(x["system_total_bytes"] for x in self.samples),
                "peak_device_used_bytes": [max(next(r[1] for r in x["gpu_mib"] if r[0] == i) for x in self.samples)*2**20 for i in range(2)],
                "device_total_bytes": [min(next(r[2] for r in x["gpu_mib"] if r[0] == i) for x in self.samples)*2**20 for i in range(2)],
                "caveat": "1s sampled peaks; tree RSS double-counts shared fork pages; system used includes notebook environment"}


def pack_results(output, archive):
    paths = [p for p in Path(output).rglob("*") if p.is_file()]
    need(all(p.suffix in (".json", ".jsonl", ".csv") for p in paths), "Unexpected result payload")
    manifest = {str(p.relative_to(output).as_posix()): {"sha256": sha(p), "bytes": p.stat().st_size} for p in paths}
    jwrite(Path(output)/"files.json", manifest)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths + [Path(output)/"files.json"]:
            z.write(p, p.relative_to(output).as_posix())


def run_benchmark(wrapper):
    global _BENCH
    wrapper = Path(wrapper)
    config = jread(wrapper / "config.json")
    need(config["version"] == VERSION and sha(wrapper/"exp169-release.ipynb") == RELEASE_SHA, "Frozen source mismatch")
    working = Path("/kaggle/working")
    for name in ("exp155-assets", "exp169-trio", "exp169-partner", "exp169-pilot", "submission.csv", "exp176-results.zip"):
        need(not (working/name).exists(), "Stale release output: " + name)
    mounts = [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"), Path("/kaggle/input/rsna-knee-abnormality-detection")]
    present = [p for p in mounts if (p/"train_series.csv").is_file()]
    need(present, "Competition train mount missing")
    preflight_start = time.perf_counter()
    view = build_view(present[0], working/"exp176-cohort", config["intended"])
    inventory_preflight_seconds = time.perf_counter()-preflight_start
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    need(torch.cuda.device_count() == 2 and all("T4" in torch.cuda.get_device_name(i) for i in range(2)), "Exactly 2xT4 required")
    started = time.perf_counter()
    output = fresh(working/"exp176-results")
    jwrite(output/"config.json", config)
    for name in ("image-inventory.json", "inventory-check.json"):
        shutil.copyfile(view/name, output/name)
    _BENCH = Observer(output, view)
    monitor = MemoryMonitor()
    monitor.start()
    status, error = "FAILED", None
    try:
        ns = {"__name__": "__main__", "_bench": _BENCH}
        source = rewrite_cell(code_cell(jread(wrapper/"exp169-release.ipynb")), view)
        release_start = time.perf_counter()
        exec(compile(source, "EXP169-v1.0.2-benchmark-wrapper", "exec"), ns)
        _BENCH.stages["release_total"] = time.perf_counter()-release_start
        receipt = jread(working/"exp169-executed-candidate.json")
        need(receipt["status"] == "PASS_BUNDLE" and receipt["candidate"] == "TRIO_PLUS_BUNDLE", "Intended candidate did not execute")
        check = cohort.check(cohort.read_ids(working/"submission.csv"), config["intended"])
        jwrite(output/"cohort-check.json", check)
        status = "COMPLETE_SAMPLE_BENCHMARK"
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        # A benchmark never leaves a root submission.csv; retain predictions only as evidence.
        if (working/"submission.csv").is_file():
            (working/"submission.csv").rename(output/"benchmark-predictions.csv")
        try:
            memory = monitor.finish()
        except Exception as exc:
            memory = {"error": str(exc)}
            status, error = "FAILED", "Incomplete memory measurement"
        for src, dest in [("exp169-executed-candidate.json", "executed-candidate.json"),
                          ("exp169-pilot/partner-pilot.json", "partner-pilot.json"),
                          ("exp169-trio/receipt.json", "trio-receipt.json"),
                          ("exp169-trio/runtime.json", "trio-runtime.json")]:
            if (working/src).is_file():
                shutil.copyfile(working/src, output/dest)
        receipt = {"status": status, "error": error, "version": VERSION, "config": config,
                   "qualification": "TRAIN_SAMPLE_ONLY", "stages": _BENCH.stages,
                   "wall_before_result_packaging_seconds": time.perf_counter()-started,
                   "inventory_preflight_seconds": inventory_preflight_seconds,
                   "memory": memory,
                   "source_sha256": {"benchmark": sha(__file__), "cohort_helper": sha(cohort.__file__)},
                   "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)]}
        jwrite(output/"benchmark-receipt.json", receipt)
        # Record measured payload serialization; final archive cost is bounded and reported separately.
        t = time.perf_counter()
        provisional = working/"exp176-packaging-probe.zip"
        pack_results(output, provisional)
        receipt["result_packaging_seconds"] = time.perf_counter()-t
        receipt["result_packaging_bytes"] = provisional.stat().st_size
        (output/"files.json").unlink()
        jwrite(output/"benchmark-receipt.json", receipt)
        pack_results(output, working/"exp176-results.zip")
    need(status == "COMPLETE_SAMPLE_BENCHMARK", error or "Benchmark failed")
    print(json.dumps({"status": status, "archive": "exp176-results.zip", "studies": config["intended"]["studies"]}))


def collect(stage_dir, archive, out):
    build = checked_stage(stage_dir)
    out = fresh(out)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and len(names) < 256, "Invalid result archive inventory")
        need(sum(i.file_size for i in z.infolist()) < 64*2**20, "Oversized uncompressed results")
        for name in names:
            need(not name.startswith(("/", "\\")) and "\\" not in name and ":" not in name and
                 ".." not in Path(name).parts and not name.endswith("/"), "Unsafe result archive path")
        need("files.json" in names, "Missing hash inventory")
        manifest = json.loads(z.read("files.json"))
        need(set(names) == set(manifest) | {"files.json"}, "Result archive inventory mismatch")
        for name, info in manifest.items():
            data = z.read(name)
            need(len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["sha256"], "Result hash mismatch")
        z.extractall(out)
    receipt = jread(out/"benchmark-receipt.json")
    need(receipt["config"] == build["config"] and receipt["source_sha256"] == build["source_sha256"], "Result not bound to staged benchmark")
    jwrite(out/"collection.json", {"status": "COLLECTED_HASH_VERIFIED", "archive_sha256": sha(archive),
                                 "build_receipt_sha256": sha(Path(stage_dir)/"build-receipt.json"),
                                 "benchmark_status": receipt["status"]})
    return {"status": "COLLECTED_HASH_VERIFIED", "benchmark_status": receipt["status"]}


def summarize_events(events):
    groups = defaultdict(list)
    for e in events:
        need(isinstance(e["seconds"], (float, int)) and math.isfinite(e["seconds"]) and e["seconds"] >= 0, "Invalid event duration")
        groups[(e["phase"], e["kind"])].append(e)
    summary = {}
    for (phase, kind), values in groups.items():
        by_study = defaultdict(float)
        for e in values:
            if "uid" in e:
                by_study[e["uid"]] += e["seconds"]
        times = sorted(by_study.values())
        summary[phase+"/"+kind] = {"events": len(values), "sum_seconds": sum(e["seconds"] for e in values),
                                    "studies": len(times), "study_mean_seconds": statistics.mean(times) if times else None,
                                    "study_p95_seconds": times[math.ceil(.95*len(times))-1] if times else None,
                                    "study_max_seconds": max(times) if times else None,
                                    "per_study_seconds": dict(by_study)}
    return summary


def projection(receipt, executed):
    """Conservative sensitivity calculation, never a hidden measurement or CI.

    Scaling whole measured stage walls (including their loads) deliberately
    overcounts fixed work. Never sum concurrent decode/GPU accumulated times.
    """
    n = receipt["config"]["intended"]["studies"]
    hidden = receipt["config"]["assumed_hidden_studies"]
    stages = receipt["stages"]
    trio, partner, pilot = stages["trio"], stages["partner_full"], stages["pilot"]
    wall = receipt["wall_before_result_packaging_seconds"]
    need(isinstance(n, int) and n >= 40 and isinstance(hidden, int) and hidden >= n,
         "Invalid measured/projected cohort sizes")
    need(all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in
             (trio, partner, pilot, wall, receipt["result_packaging_seconds"])), "Invalid measured wall durations")
    need(wall >= trio+partner+pilot, "Overlapping/invalid outer stage wall times")
    residual = max(0., wall - trio-partner-pilot)
    # serve's non-partner residual includes raw NPZ serialization, parity,
    # global ranks and final CSV/receipt packaging. Scale it with the cohort.
    partner_output = stages["serve"] - partner
    need(math.isfinite(partner_output) and 0 <= partner_output <= residual,
         "Invalid serving/output wall durations")
    setup = residual-partner_output
    packaging = 2*receipt["result_packaging_seconds"]
    base = (trio+partner+partner_output)*hidden/n + pilot + setup + packaging
    conservative = 1.5*((trio+partner+partner_output)*hidden/n + setup) + pilot + packaging + 600
    p = executed["pilot"]
    need(p["studies"] == 40 and math.isfinite(p["elapsed_seconds"]) and p["elapsed_seconds"] > 0,
         "Invalid original pilot measurement")
    pilot_gate_partner = 1.5*p["elapsed_seconds"]*hidden/p["studies"]+600
    projected_trio = 1.5*trio*hidden/n
    gate_remaining = GUARD - projected_trio-p["elapsed_seconds"]-1.5*(setup+partner_output*hidden/n)-packaging
    return {"assumed_hidden_studies": hidden, "measured_sample_studies": n,
            "assumption": "Hidden cohort ~1300 studies; same per-study mix/throughput as deliberately tail-enriched train sample. Size is an assumption, not observed hidden membership.",
            "linear_total_seconds": base, "conservative_total_seconds": conservative,
            "trio_scaled_seconds": trio*hidden/n, "partner_scaled_seconds": partner*hidden/n,
            "pilot_fixed_seconds": pilot, "setup_aggregation_packaging_residual_seconds": residual,
            "fixed_setup_seconds": setup, "partner_parity_aggregation_packaging_scaled_seconds": partner_output*hidden/n,
            "result_packaging_allowance_seconds": packaging,
            "projected_release_pilot_gate_partner_seconds": pilot_gate_partner,
            "projected_release_pilot_gate_remaining_seconds": gate_remaining,
            "projected_release_pilot_gate_pass": pilot_gate_partner <= gate_remaining,
            "guard_seconds": GUARD, "kaggle_limit_seconds": KAGGLE_LIMIT,
            "guard_margin_seconds": GUARD-conservative, "kaggle_margin_seconds": KAGGLE_LIMIT-conservative,
            "method": "whole-stage wall scaling plus serve-minus-partner output/parity scaling; 1.5x safety factor + 600s; fixed 40-study pilot once; no double-count of nested serve/partner or concurrent worker sums",
            "not_a_confidence_bound": True}


def report(collected, out):
    source = Path(collected)
    need(jread(source/"collection.json")["status"] == "COLLECTED_HASH_VERIFIED", "Collect first")
    for name, info in jread(source/"files.json").items():
        need(sha(source/name) == info["sha256"], "Collected result drift")
    receipt = jread(source/"benchmark-receipt.json")
    need(receipt["status"] == "COMPLETE_SAMPLE_BENCHMARK", "Failed benchmark: inspect failure receipt; no timing qualification")
    intended = receipt["config"]["intended"]
    inventory_check = check_inventory(jread(source/"image-inventory.json"), intended)
    need(jread(source/"inventory-check.json") == inventory_check, "Runtime inventory verification missing/mismatched")
    cohort.check(cohort.read_ids(source/"benchmark-predictions.csv"), intended)
    need(intended["studies"] >= 40, "At least 40 benchmark studies required")
    executed = jread(source/"executed-candidate.json")
    need(executed["status"] == "PASS_BUNDLE" and executed["candidate"] == "TRIO_PLUS_BUNDLE" and
         executed["studies"] == intended["studies"] and
         executed["executed_arms"] == ["EXP155_BLEND3", "L2_RADR50_10", "DINO_BASE_5", "EXP170_L2_CONVNEXT_T_5"] and
         executed["weights"] == {"trio": .85, "bundle": .15} and
         executed["stages"]["parity"]["status"] == "PASS" and
         executed["submission_sha256"] == sha(source/"benchmark-predictions.csv"), "Release candidate/parity/output binding failed")
    with (source/"benchmark-predictions.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames == [UID, *TARGETS], "Benchmark target order differs")
        for row in reader:
            need(all(math.isfinite(float(row[t])) and 0 <= float(row[t]) <= 1 for t in TARGETS),
                 "Invalid benchmark predictions")
    events = []
    for path in sorted((source/"events").glob("*.jsonl")):
        events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    summary = summarize_events(events)
    for phase, kind in [("trio", "trio_decode"), ("trio", "trio_inference"),
                        ("partner_full", "partner_decode"), ("partner_full", "partner_prepare"), ("partner_full", "partner_gpu")]:
        entry = summary[phase+"/"+kind]
        cohort.check(sorted(entry["per_study_seconds"]), intended)
    pilot_n = executed["pilot"]["studies"]
    need(pilot_n == 40, "Expected unchanged 40-study pilot")
    for kind in ("partner_decode", "partner_prepare", "partner_gpu"):
        need(summary["partner_pilot_scores/"+kind]["studies"] == pilot_n, "Pilot event coverage missing")
    p = projection(receipt, executed)
    memory = receipt["memory"]
    memory_pass = (memory["peak_system_used_bytes"] < 27*2**30 and
                   max(memory["peak_device_used_bytes"]) < 15*2**30)
    result = {"status": "PROJECTION_WITHIN_GUARD" if p["guard_margin_seconds"] >= 0 and p["projected_release_pilot_gate_pass"] and memory_pass else "HOLD_RUNTIME_OR_MEMORY",
              "qualification": "TRAIN_SAMPLE_ONLY_NOT_FULL_COHORT", "full_rehearsal_required": True,
              "hb41_coverage_gate": inventory_check["coverage"]["status"],
              "inventory_check": inventory_check,
              "inventory_preflight_seconds": receipt["inventory_preflight_seconds"],
              "version": VERSION, "projection": p, "memory": memory, "memory_thresholds_pass": memory_pass,
              "intended": intended, "stage_wall_seconds": receipt["stages"], "events": summary,
              "release_reported_qualification_ignored": executed["qualification"],
              "limitations": ["No labels, AUC or accuracy qualification; train images may be familiar to checkpoints.",
                              "Counts are lowercase .dcm filenames per metadata series, not decoded frame counts or proof of readable images.",
                              "Full-population filename preflight is benchmark-only overhead, recorded separately from release projection.",
                              "Tail-enriched sample projection is a sensitivity estimate, not a hidden runtime bound.",
                              "Per-study GPU observations include host transfers/output synchronization; trio preparation measures enqueue time.",
                              "Memory sampled every second; short spikes may be missed; retain original CUDA allocator peaks.",
                              "Read-only wrappers add timing/logging overhead; pilot then full may warm filesystem caches."]}
    out = fresh(out)
    jwrite(out/"report.json", result)
    lines = [f"# EXP176 release timing v{VERSION}", "", result["status"] + "; TRAIN SAMPLE ONLY", "",
             f"Measured {intended['studies']} train studies; assume {p['assumed_hidden_studies']} hidden studies.",
             f"Linear projection {p['linear_total_seconds']/3600:.3f} h; conservative projection {p['conservative_total_seconds']/3600:.3f} h.",
             f"6h45 guard margin {p['guard_margin_seconds']/60:.1f} min; 9h margin {p['kaggle_margin_seconds']/60:.1f} min.",
             f"Projected original pilot gate passes: {p['projected_release_pilot_gate_pass']}; memory thresholds: {memory_pass}.", "",
             "H-B-41 filename coverage PASS: both total-image and longest-series spread/tails verified against the frozen inventory.",
             "Full-size rehearsal and intended-cohort proof remain required. No submission authorization.", "", *["- " + s for s in result["limitations"]]]
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"status": result["status"], "qualification": result["qualification"], "report_sha256": sha(out/"report.json")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inventory")
    s.add_argument("--images", type=Path, required=True, help="Train image root or official competition ZIP; filenames only")
    s.add_argument("--out", required=True)
    s.add_argument("--series", type=Path, default=ROOT/"storage/competition/train_series.csv")
    s = sub.add_parser("stage")
    s.add_argument("--out", required=True)
    s.add_argument("--inventory", type=Path, required=True)
    s.add_argument("--studies", type=int, default=48)
    s.add_argument("--hidden-studies", type=int, default=1300)
    s.add_argument("--series", type=Path, default=ROOT/"storage/competition/train_series.csv")
    s = sub.add_parser("push-plan"); s.add_argument("--stage", required=True)
    s = sub.add_parser("collect")
    for arg in ("stage", "archive", "out"):
        s.add_argument("--"+arg, required=True)
    s = sub.add_parser("report"); s.add_argument("--collected", required=True); s.add_argument("--out", required=True)
    a = vars(p.parse_args(argv)); command = a.pop("command")
    if "stage" in a:
        a["stage_dir"] = a.pop("stage")
    result = {"inventory": inventory, "stage": stage, "push-plan": push_plan, "collect": collect, "report": report}[command](**a)
    if command == "stage":
        result = {k: result[k] for k in ("status", "version", "files", "source_sha256")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
