"""EXP189: pinned serving adapter for heliosli's public S6 worker.

The inference implementation lives unchanged in exp189_s6_src, extracted from
cell 4 of the restored public notebook. This adapter verifies every input
before launching that worker and returns its six-decimal output in caller order.

Changelog
- v1.2.0 (astra, 2026-10-11): stage the public reader's series metadata; retain
  full fallback tracebacks; explicit CPU serving; remove all execution deadlines.
  Hash-pinned public sources remain intact; checked runtime adaptations are saved.
  CPU device reaches anatomy/side helpers; use -1 visibility on Windows.
- v1.0.0 (sol): initial port, bundle/source/8-module/anatomy pins, controller timeout.
- v1.1.0 (Claude opus for B, 2026-10-11): fail-closed pins for the whole executed
  code dataset tree (CODE_TREE_SHA256 covers every file the S6 path can import,
  incl. training.geometry_bank / training.anatomy / training.anatomy_runtime and
  their imports; the static import closure is recorded in provenance) and for both
  dependency wheels (PyPI digests); wheel install bounded by a deadline-aware
  process-group timeout plus installed-version check; shared process-group runner;
  service_components() adds the per-study anatomy-anchor time; anatomy_fallbacks()
  counts slab-prior anchors (logged, never rejected, like the public notebook);
  observe_code() / `pin-code` record and pin the observed code tree.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Sequence

VERSION = "1.2.0"
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "exp189_s6_src"
UID = "StudyInstanceUID"
LABELS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
BUNDLE_ID = "209a32c8b50a6b1c"
DATASET_SLUG = "heliosli/rsna-knee-compact-v1"
README_PERMISSION = "Feel free to blend my model with yours!"
MANIFEST_SHA256 = "1c1211e64b73f1f6433ade804816572db1a2aa5422ed9c57365184bc6a9517e6"
README_SHA256 = "6061da301e7d49dc5bbe00fcf81b42c04f3c78c905e0b4b3e804d5ac2ff2f613"
HASHES = {
    "E_lrprior_s0_f0.pt": "55983f47c25f78e8cd3b9f3de081d207e2f6c68c248ae4eb8dee806b0419b7ed",
    "E_lrprior_s0_f1.pt": "da5c8d51855771a7e3759db8a8353655251099535c22baba1e0522753e7612c0",
    "E_lrprior_s0_f2.pt": "ee7747abbcf265c068295ae4b857f65bfa13045e80a02ad701c5d476eaea1a90",
    "E_lrprior_s0_f3.pt": "b9223d0583e951b4352fce5bc8775e54b9e7dc7c6ba7fd72db75637ab65eb4cd",
    "state_spec.json": "e342034a17d084f3378c04b43e0b622ecab4b052d40ca90199f344c185382d78",
    "side_cls.pt": "ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a",
}
SOURCE_HASHES = {
    "px_cache.py": "b13e38a7ad3bd496f8027e02209653e224b0926db4f0853eeb0d076a2d6e2632",
    "compact_train.py": "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80",
    "compact_infer.py": "34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a",
    "probe_infer.py": "a6cc31448019c84be5868d0a2ff1d99719b23b0f86110efc03efff3ce6bd6980",
    "probe_dual.py": "9171522ada7ad94145529bb0a47ec48b6f0d8d32d651a60f3f207558649e36b8",
    "knee_anchor_bank_v2.py": "e564807a129dfcd3c984348917436ee6083066a4d685340587ebd1d3ea14f309",
    "side_common.py": "bbcc5d027834e6a6fd48c4a8cf228e25661b7522c792587c840fe4d7cd5c0566",
}
CODE_HASHES = {
    "native_cache.py": "c53ce8985406b3b918d9e821d43518479e89d9f946daebd817f378e3949718b9",
    "native_data.py": "f59afe205c8e193369fa3ccd22edcce8df3b293205fc48ddb1dd45169ee43b36",
    "input_provenance.py": "b3a22500c7f976f6ee040239ada5cacecf2326861cb49af1c7d8cfdeeb55fcda",
    "intensity_mapping.py": "19be2d8cbd6f8971c1288d7512a8cca41d143470c2ba6e9b9096fde035fa9d41",
    "coronal_views.py": "841f216d2682ae2c0f8afd82fc902b9fa74924d7145b952069dea8b0bbd03c40",
    "exact_duplicates.py": "51b38db21ce9c5fe2b6278a835983414756b61ac6af81aba380b44b41b075897",
    "dicom_geometry.py": "12aa501f9704aef15b7a3bbd560951acb6d95b89b7de0141d6cf89fc3f051ab1",
    "common.py": "8283fa1f8e7ef8cfacbba52df1715636be3c9ad688e39912814e7d5bcf57105f",
}
ANATOMY_SHA256 = "5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e"
# Digest of the complete extracted code tree (sorted "relpath\0sha256\n" lines, see
# code_tree_manifest). None = not yet observed: every caller that does not pass an
# explicit observed digest fails closed. Populate only with `pin-code`.
CODE_TREE_SHA256: str | None = "16fc225ecee3b40f9117eef5b6b41ec21d2b17dc988144d08338700122a7490d"
CODE_TREE_FILES: int | None = 820
# The two wheels the public cell installs (dependencies/wheels). Digests and sizes are
# PyPI's published values; the dataset listing reports the same byte sizes.
WHEEL_HASHES = {
    "pydicom-3.0.1-py3-none-any.whl": "db32f78b2641bd7972096b8289111ddab01fb221610de8d7afa835eb938adb41",
    "timm-1.0.25-py3-none-any.whl": "bef7f61dd717cb2dbbb7e326f143e13d660a47ecbd84116e6fe33732bed5c484",
}
WHEEL_VERSIONS = {"pydicom": "3.0.1", "timm": "1.0.25"}
MODEL_ANCHOR_PREFIX = "skm_tea_joint"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path.name}: {actual} != {expected}")


def verify_bundle(bundle_dir: str | Path) -> dict:
    bundle = Path(bundle_dir)
    _check_hash(bundle / "manifest.json", MANIFEST_SHA256)
    _check_hash(bundle / "README.md", README_SHA256)
    if README_PERMISSION not in (bundle / "README.md").read_text(encoding="utf-8"):
        raise ValueError("README permission line is absent")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    expected_models = [f"E_lrprior_s0_f{i}.pt" for i in range(4)]
    required = {
        "bundle_id": BUNDLE_ID, "arm": "E_lrprior", "models": expected_models,
        "backbone": "convnext_small", "size": 384, "crop": 352,
        "slices": 3, "phys_mm": 3.5, "views": "v4", "eval_stride": 1,
        "state_spec": "state_spec.json", "side_model": "side_cls.pt",
        "side_threshold": 2.0, "side_position_mm": 20.0, "side_max_x": 0.0,
        "gold_in_training": 0, "official_labels_read_by_trainer": False,
        "anchor_weights_sha256": ANATOMY_SHA256,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} differs from pinned S6: {manifest.get(key)!r}")
    if manifest.get("models2") is not None:
        raise ValueError("unexpected second model group")
    if manifest.get("sha256") != HASHES:
        raise ValueError("manifest SHA256 map differs from pinned S6")
    found_models = {path.name for path in bundle.glob("E_lrprior_s*_f*.pt")}
    if found_models != set(expected_models):
        raise ValueError(f"S6 model roster differs from pinned manifest: {sorted(found_models)}")
    for name, digest in HASHES.items():
        _check_hash(bundle / name, digest)
    return manifest


def verify_source(source_dir: str | Path = SOURCE) -> None:
    for name, digest in SOURCE_HASHES.items():
        _check_hash(Path(source_dir) / name, digest)


# Modules the vendored worker sources import directly from the code dataset; each
# must resolve inside the pinned tree (their own imports are followed statically).
DIRECT_CODE_MODULES = (
    "training/__init__.py", "training/anatomy.py", "training/anatomy_runtime.py",
    "training/coronal_views.py", "training/dicom_geometry.py", "training/exact_duplicates.py",
    "training/geometry_bank.py", "training/input_provenance.py", "training/intensity_mapping.py",
    "training/native_cache.py", "training/native_data.py",
)


def code_tree_manifest(code_dir: str | Path) -> dict:
    """SHA256 of every file below the extracted code dir and one digest over all of them."""
    root = Path(code_dir)
    files: dict[str, str] = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        here = Path(current)
        for name in dirs + names:
            if (here / name).is_symlink():
                raise ValueError("symlink in S6 code tree: " + (here / name).relative_to(root).as_posix())
        for name in names:
            files[(here / name).relative_to(root).as_posix()] = sha256(here / name)
    if not files:
        raise FileNotFoundError("S6 code tree is empty")
    return {"tree_sha256": tree_digest(files), "files": len(files), "sha256": dict(sorted(files.items()))}


def tree_digest(files: dict[str, str]) -> str:
    lines = "".join(f"{rel}\0{files[rel]}\n" for rel in sorted(files))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _imported_names(path: Path, module: str, is_package: bool) -> set[str]:
    tree = ast.parse(path.read_bytes(), filename=str(path))
    package = module if is_package else module.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                if node.level - 1 > len(parts):
                    continue
                base = ".".join(parts[:len(parts) - (node.level - 1)] + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                names.add(base)
                names.update(base + "." + alias.name for alias in node.names if alias.name != "*")
    return names


def _resolve_module(code: Path, dotted: str) -> list[tuple[str, Path, bool]]:
    """Files executed when `dotted` is imported from the code dir (package __init__s first)."""
    found, path, parts = [], code, dotted.split(".")
    for i, part in enumerate(parts):
        path = path / part
        name = ".".join(parts[:i + 1])
        if (path / "__init__.py").is_file():
            found.append((name, path / "__init__.py", True))
        elif (path.parent / (part + ".py")).is_file():
            found.append((name, path.parent / (part + ".py"), False))
            break
        elif not path.is_dir():
            break
    return found


def executed_closure(code_dir: str | Path, source_dir: str | Path = SOURCE) -> dict[str, str]:
    """Static import closure of the vendored S6 sources inside the code dir: relpath -> SHA256.

    Function-level and conditional imports are included (over-approximation). Dynamic
    imports are not visible here; CODE_TREE_SHA256 pins those bytes as well.
    """
    code = Path(code_dir)
    roots = {p.stem if p.suffix == ".py" else p.name for p in code.iterdir()}
    pending = []
    for source in sorted(Path(source_dir).glob("*.py")):
        pending.extend(_imported_names(source, source.stem, False))
    seen: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if name.split(".")[0] not in roots:
            continue
        for module, path, is_package in _resolve_module(code, name):
            rel = path.relative_to(code).as_posix()
            if rel not in seen:
                seen[rel] = sha256(path)
                pending.extend(_imported_names(path, module, is_package))
    return dict(sorted(seen.items()))


def verify_code(code_dir: str | Path, anatomy_weights: str | Path, *,
                tree_sha256: str | None = None) -> dict:
    """Fail closed unless the whole extracted code tree equals the pinned (or explicitly
    supplied observed) digest, the eight public module pins hold, every directly imported
    mainline module resolves inside it, and anatomy.pt matches."""
    expected = CODE_TREE_SHA256 if tree_sha256 is None else tree_sha256
    if not (isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected)):
        raise ValueError("S6 code-tree pin is not populated; fail closed (pin-code from an observed manifest)")
    for name, digest in CODE_HASHES.items():
        _check_hash(Path(code_dir) / "training" / name, digest)
    _check_hash(Path(anatomy_weights), ANATOMY_SHA256)
    manifest = code_tree_manifest(code_dir)
    if manifest["tree_sha256"] != expected:
        raise ValueError(f"S6 code tree differs from pin: {manifest['tree_sha256']} != {expected}")
    if tree_sha256 is None and CODE_TREE_FILES is not None and manifest["files"] != CODE_TREE_FILES:
        raise ValueError("S6 code tree file count differs from pin")
    closure = executed_closure(code_dir)
    missing = [rel for rel in DIRECT_CODE_MODULES if rel not in closure]
    if missing:
        raise ValueError("S6 executed code closure is incomplete: " + ", ".join(missing))
    if any(manifest["sha256"].get(rel) != digest for rel, digest in closure.items()):
        raise ValueError("S6 executed closure differs from the pinned tree")
    return {"tree_sha256": manifest["tree_sha256"], "files": manifest["files"], "closure_sha256": closure}


def _extract_code(src: Path, code: Path) -> None:
    code.mkdir(parents=True, exist_ok=False)
    if (src / "cloud_code.zip").is_file():
        with zipfile.ZipFile(src / "cloud_code.zip") as archive:
            for name in archive.namelist():
                target = (code / name).resolve()
                if not target.is_relative_to(code.resolve()):
                    raise ValueError("unsafe code archive path")
            archive.extractall(code)
    else:
        shutil.copytree(src / "cloud_code", code, dirs_exist_ok=True)


def _collect_wheels(src: Path, wheels: Path) -> list[Path]:
    found = [] if (src / "dependencies.zip").is_file() else sorted(src.rglob("*.whl"))
    wheels.mkdir(parents=True, exist_ok=False)
    if (src / "dependencies.zip").is_file():
        with zipfile.ZipFile(src / "dependencies.zip") as archive:
            for name in archive.namelist():
                if name.endswith(".whl"):
                    (wheels / Path(name).name).write_bytes(archive.read(name))
    else:
        for file in found:
            shutil.copyfile(file, wheels / file.name)
    paths = sorted(wheels.glob("*.whl"))
    if not paths:
        raise FileNotFoundError("public code dataset has no dependency wheels")
    return paths


def verify_wheels(paths: Sequence[Path]) -> dict[str, str]:
    names = sorted(Path(p).name for p in paths)
    if names != sorted(WHEEL_HASHES):
        raise ValueError(f"S6 dependency wheel roster differs from pin: {names}")
    for path in paths:
        _check_hash(Path(path), WHEEL_HASHES[Path(path).name])
    return dict(sorted(WHEEL_HASHES.items()))


def install_wheels(paths: Sequence[Path], code_dir: str | Path) -> dict:
    """Offline --no-deps install without a time limit, then check the
    versions a fresh interpreter with the worker PYTHONPATH resolves."""
    verify_wheels(paths)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    _run_group([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *map(str, paths)],
               env, label="S6 wheel install")
    probe = ("import importlib.metadata as m, json, pydicom, timm; print(json.dumps({"
             "'pydicom': m.version('pydicom'), 'timm': m.version('timm'), "
             "'pydicom_module': pydicom.__version__, 'timm_module': timm.__version__}))")
    worker_env = dict(env, PYTHONPATH=str(code_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = _run_group([sys.executable, "-c", probe], worker_env,
                     label="S6 wheel version check", capture=True)
    versions = json.loads(out.strip().splitlines()[-1])
    expected = {**WHEEL_VERSIONS, **{k + "_module": v for k, v in WHEEL_VERSIONS.items()}}
    if versions != expected:
        raise ValueError(f"S6 installed dependency versions differ: {versions}")
    return versions


def prepare_code(code_source: str | Path, destination: str | Path, install_wheels: bool = True, *,
                 code_tree_sha256: str | None = None) -> tuple[Path, Path]:
    """Unpack the pinned public mainline code dataset as public cell 4 does.

    The destination must be fresh. Tree, module, anatomy and wheel pins are checked
    before anything from the dataset is installed or executed.
    """
    src, dst = Path(code_source), Path(destination)
    code = dst / "mainline_code"
    _extract_code(src, code)
    anatomy = src / "anatomy.pt"
    verify_code(code, anatomy, tree_sha256=code_tree_sha256)
    paths = _collect_wheels(src, dst / "wheels")
    verify_wheels(paths)
    if install_wheels:
        _install_wheels(paths, code)
    return code, anatomy


_install_wheels = install_wheels


def observe_code(code_source: str | Path, output_json: str | Path, *, work_dir: str | Path | None = None) -> dict:
    """Record (never enforce) the code tree, static closure and wheels as prepare_code sees
    them. Used by measurement/parity kernels and by `pin-code`; no import, no install."""
    src = Path(code_source)
    with tempfile.TemporaryDirectory(prefix="exp189_observe_", dir=work_dir) as tmp:
        code = Path(tmp) / "mainline_code"
        _extract_code(src, code)
        manifest = code_tree_manifest(code)
        closure = executed_closure(code)
        wheels = {p.name: sha256(p) for p in _collect_wheels(src, Path(tmp) / "wheels")}
    record = {"schema": "exp189-code-observation-v1", "port_version": VERSION,
              "tree_sha256": manifest["tree_sha256"], "files": manifest["files"],
              "file_sha256": manifest["sha256"], "closure_sha256": closure,
              "anatomy_sha256": sha256(src / "anatomy.pt"), "wheel_sha256": dict(sorted(wheels.items())),
              "public_module_pins_match": all(manifest["sha256"].get("training/" + n) == d
                                              for n, d in CODE_HASHES.items()),
              "wheel_pins_match": wheels == WHEEL_HASHES,
              "direct_modules_present": all(rel in closure for rel in DIRECT_CODE_MODULES)}
    Path(output_json).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def write_provenance(path: str | Path, *, bundle_dir: str | Path, code_dir: str | Path | None = None,
                     anatomy_weights: str | Path | None = None, code_tree_sha256: str | None = None) -> dict:
    verify_bundle(bundle_dir)
    verify_source()
    code_record = None
    if code_dir is not None or anatomy_weights is not None:
        if code_dir is None or anatomy_weights is None:
            raise ValueError("code_dir and anatomy_weights must be supplied together")
        code_record = verify_code(code_dir, anatomy_weights, tree_sha256=code_tree_sha256)
    record = {
        "dataset_slug": DATASET_SLUG, "dataset_version": "10-equivalent",
        "bundle_id": BUNDLE_ID, "licence": "other", "readme_permission_line": README_PERMISSION,
        "gold_in_training": 0, "official_labels_read_by_trainer": False,
        "manifest_sha256": MANIFEST_SHA256, "readme_sha256": README_SHA256,
        "sha256": HASHES, "source_sha256": SOURCE_HASHES,
        "code_sha256": CODE_HASHES if code_dir is not None else None,
        "anchor_weights_sha256": ANATOMY_SHA256 if anatomy_weights is not None else None,
        "port_version": VERSION,
        "code_tree_sha256": None if code_record is None else code_record["tree_sha256"],
        "code_tree_files": None if code_record is None else code_record["files"],
        "code_tree_pinned_in_source": code_record is not None and code_tree_sha256 is None,
        "code_closure_sha256": None if code_record is None else code_record["closure_sha256"],
        "wheel_sha256": WHEEL_HASHES, "wheel_versions": WHEEL_VERSIONS,
    }
    Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _study_ids(study_dirs: Sequence[str | Path]) -> tuple[list[str], list[Path]]:
    paths = [Path(p).resolve() for p in study_dirs]
    ids = [p.name for p in paths]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("study list must be nonempty with unique directory names")
    if any(not p.is_dir() for p in paths):
        raise FileNotFoundError("all study paths must be directories")
    return ids, paths


def _stage_studies(paths: Sequence[Path], ids: Sequence[str], stage: Path) -> None:
    """Make a minimal competition data root for the public directory reader.

    ``native_cache.prepare`` hashes ``<data>/test_series.csv`` before it reads
    any DICOM. Keep the source metadata rows intact, including the anatomical
    plane that chooses the decoder, even when the source split is train.
    """
    roots = {path.parent.parent for path in paths}
    splits = {path.parent.name for path in paths}
    if len(roots) != 1 or len(splits) != 1 or not splits <= {"test_series", "train_series"}:
        raise ValueError("study paths must share one competition train_series or test_series root")
    source_csv = next(iter(roots)) / (next(iter(splits)) + ".csv")
    with source_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        if not header or UID not in header or "SeriesInstanceUID" not in header or "Anatomical_Plane" not in header:
            raise ValueError(f"invalid series metadata columns: {source_csv}")
        selected = [row for row in reader if row[UID] in set(ids)]
    found = {row[UID] for row in selected}
    if found != set(ids):
        raise ValueError(f"series metadata missing requested studies: {sorted(set(ids) - found)}")
    with (stage / "test_series.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(selected)
    series = stage / "test_series"
    series.mkdir(parents=True)
    for uid, path in zip(ids, paths):
        try:
            os.symlink(path, series / uid, target_is_directory=True)
        except OSError as exc:
            # Windows without Developer Mode: hardlink only the requested DICOM
            # files. No recursive copy, raw-data mutation, or shell junction.
            if os.name != "nt" or getattr(exc, "winerror", None) != 1314:
                raise
            for source in path.rglob("*.dcm"):
                dest = series / uid / source.relative_to(path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, dest)
    with (stage / "test.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([UID])
        w.writerows([[uid] for uid in ids])
    with (stage / "sample_submission.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([UID, *LABELS])
        w.writerows([[uid, *([0.5] * len(LABELS))] for uid in ids])


def _replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"S6 diagnostic patch {name} expected once, found {count}")
    return source.replace(old, new, 1)


def _diagnostic_source(source_dir: str | Path, stage: Path) -> Path:
    """Auditably adapt a runtime copy; model/preprocessing math is unchanged.

    Every replacement must match exactly once. The public originals stay pinned;
    the executable derivative adds diagnostics and CPU support and drops timing
    policy. CUDA serving keeps the public precision and scheduling behavior.
    """
    runtime = stage / "s6_runtime"
    shutil.copytree(source_dir, runtime)
    worker = runtime / "compact_infer.py"
    code = worker.read_text(encoding="utf-8")
    patches = (
        ("import argparse, contextlib, csv, hashlib, io, json, os, shutil, sys, time",
         "import argparse, contextlib, csv, hashlib, io, json, os, shutil, sys, time, traceback", "traceback import"),
        ('"error": repr(exc)[:300], "images":',
         '"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "images":', "decode exception"),
        ('fallback[uid] = item["error"] or "no usable window"',
         'fallback[uid] = (item["error"] or "no usable window") + ("\\n" + item.get("traceback", "") if item.get("traceback") else "")',
         "decode fallback"),
        ('fallback[uid] = "gpu: " + repr(exc)[:200]',
         'fallback[uid] = "gpu: " + f"{type(exc).__name__}: {exc}\\n" + traceback.format_exc()', "GPU fallback"),
        ('fallback[uids[pos]] = "loader: " + repr(exc)[:200]',
         'fallback[uids[pos]] = "loader: " + f"{type(exc).__name__}: {exc}\\n" + traceback.format_exc()',
         "loader fallback"),
        ('item["seconds"][1], round(dt, 3)])',
         'item["seconds"][1], round(dt, 3), item["seconds"][2] if len(item["seconds"]) > 2 else 0.])',
         "per-study anchor timing"),
        ('"per_study": per_study[:50]', '"per_study": per_study', "uncapped study timing"),
        ('    ap.add_argument("--item-timeout", type=float, default=900.)\n', '', 'no loader time limit'),
        ('    ap.add_argument("--deadline", type=float, default=0.)',
         '    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")', 'explicit worker device'),
        ('    dev = torch.device("cuda")',
         '    dev = torch.device(a.device)\n    if dev.type == "cpu":\n        torch.set_num_threads(2)', 'worker device'),
        ('side_position_mm=20.0, extent_meta=False):',
         'side_position_mm=20.0, extent_meta=False, device="cuda"):', 'dataset device argument'),
        ('        self.native = None', '        self.device = device\n        self.native = None', 'dataset device'),
        ('            dev = "cuda" if torch.cuda.is_available() else "cpu"',
         '            dev = self.device', 'anatomy device'),
        ('            sdev = "cuda" if torch.cuda.is_available() else "cpu"',
         '            sdev = self.device', 'side device'),
        ('extent_meta=N_EXTENT[0] > 0)', 'extent_meta=N_EXTENT[0] > 0, device=a.device)', 'forward dataset device'),
        ('    cap = torch.cuda.get_device_capability()[0]\n    amp = a.amp if a.amp != "auto" else ("bf16" if cap >= 8 else "fp16")',
         '    cap = torch.cuda.get_device_capability()[0] if dev.type == "cuda" else 0\n'
         '    amp = "fp32" if dev.type == "cpu" else (a.amp if a.amp != "auto" else ("bf16" if cap >= 8 else "fp16"))', 'CPU fp32'),
        ('    timed_out = False\n    while pos < len(uids) and not timed_out:',
         '    while pos < len(uids):', 'no worker deadline'),
        ('multiprocessing_context="spawn" if a.workers else None,\n                                             timeout=a.item_timeout if a.workers else 0)',
         'multiprocessing_context="spawn" if a.workers else None)', 'unbounded loader'),
        ('                if a.deadline and time.time() > a.deadline:\n                    timed_out = True\n                    break\n', '', 'no early termination'),
        ('    for u in uids[pos:]:\n        fallback[u] = "not reached before the deadline"\n', '', 'no deadline fallback'),
        ('"fallback": fallback, "timed_out": timed_out, "amp": amp,',
         '"fallback": fallback, "device": dev.type, "auxiliary_device": a.device, "cuda_initialized": torch.cuda.is_initialized(), "amp": amp,', 'device receipt'),
        ('    receipt = {"status": "PASS",',
         '    if dev.type == "cpu" and torch.cuda.is_initialized():\n        raise RuntimeError("CPU worker initialized CUDA")\n    receipt = {"status": "PASS",', 'CPU device postcondition'),
        ('("studies", "scored", "timed_out", "seconds", "amp")',
         '("studies", "scored", "device", "seconds", "amp")', 'device completion log'),
    )
    for old, new, name in patches:
        code = _replace_once(code, old, new, name)
    if code.count('torch.cuda.get_device_name()') != 2:
        raise ValueError('S6 expected two device-name diagnostics')
    code = code.replace('torch.cuda.get_device_name()',
                        '(torch.cuda.get_device_name() if dev.type == "cuda" else "CPU")')
    compile(code, str(worker), 'exec')
    worker.write_text(code, encoding="utf-8", newline="\n")
    controller = runtime / "probe_dual.py"
    code = controller.read_text(encoding="utf-8")
    code = _replace_once(code, 'for u in missing[:40]', 'for u in missing', "uncapped fallback reasons")
    code = _replace_once(code, '    ap.add_argument("--deadline", type=float, default=0.)',
                         '    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")', 'controller device')
    begin = code.index('    try:\n        names = subprocess.check_output')
    end = code.index('\n\n    def cost', begin)
    code = code[:begin] + '''    if a.device == "cpu":
        names, visible = ["CPU"], ["-1"]
    else:
        try:
            names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                            text=True).strip().splitlines()
        except Exception:
            names = ["unknown"]
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visibility is not None:
            visible = [v.strip() for v in visibility.split(",") if v.strip() and v.strip() != "-1"]
            if not visible:
                raise SystemExit("CUDA is disabled; select --device cpu explicitly")
            names = [names[int(v)] if v.isdigit() and int(v) < len(names) else v for v in visible]
        else:
            visible = [str(i) for i in range(max(1, len(names)))]
    n_gpu = len(visible)''' + code[end:]
    code = _replace_once(code, 'CUDA_VISIBLE_DEVICES=str(gi)', 'CUDA_VISIBLE_DEVICES=visible[gi]', 'respect CUDA visibility')
    code = _replace_once(code, '"--deadline", str(a.deadline)', '"--device", a.device', 'forward device')
    # A worker can fail before its per-study exception handler (imports/models).
    # Keep its complete log against every unscored assignment in that case.
    code = _replace_once(code, 'receipts.append({"status": "NO_RECEIPT", "worker": gi})',
        'receipts.append({"status": "NO_RECEIPT", "worker": gi, "fallback": {u: '
        '(wdir / "log.txt").read_text(encoding="utf-8", errors="replace") for u in part}})', 'startup fallback traceback')
    compile(code, str(controller), 'exec')
    controller.write_text(code, encoding="utf-8", newline="\n")
    (runtime / 'runtime_sha256.json').write_text(json.dumps(
        {p.name: sha256(p) for p in sorted(runtime.glob('*.py'))}, indent=2) + '\n', encoding='utf-8')
    return runtime


def read_predictions(path: str | Path, study_ids: Sequence[str]) -> list[dict[str, str | float]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != [UID, *LABELS]:
            raise ValueError("S6 output columns differ from pinned label order")
        rows = list(reader)
    if [r[UID] for r in rows] != list(study_ids):
        raise ValueError("S6 output study order differs from request")
    out = []
    for row in rows:
        values = {label: float(row[label]) for label in LABELS}
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("S6 output has a nonfinite or out-of-range probability")
        out.append({UID: row[UID], **values})
    return out


def _kill_group(process: subprocess.Popen) -> None:
    """Stop the child and every descendant in its own session/process group."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        process.wait()


def _run_group(args: list[str], env: dict[str, str] | None, *, label: str = "S6 controller",
               capture: bool = False) -> str | None:
    """Wait for completion without a time limit; clean up on actual interruption."""
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    if capture:
        options.update(stdout=subprocess.PIPE, text=True)
    process = subprocess.Popen(args, env=env, **options)
    try:
        out, _ = process.communicate()
    except BaseException:
        _kill_group(process)
        raise
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")
    return out


def _run_controller(args: list[str], env: dict[str, str]) -> None:
    _run_group(args, env, label="S6 controller")


def infer_studies(study_dirs: Sequence[str | Path], *, bundle_dir: str | Path, code_dir: str | Path,
                  anatomy_weights: str | Path, output_dir: str | Path, workers: int = 2,
                  amp: str = "auto", device: str = "cuda", max_fallback_fraction: float = 0.03,
                  source_dir: str | Path = SOURCE,
                  code_tree_sha256: str | None = None) -> list[dict[str, str | float]]:
    """Score study directories on CPU or visible CUDA devices; return ordered rows.

    ``code_dir`` is the extracted public mainline code dataset. The original
    probe_dual controller creates one worker per visible GPU. No GPU is touched
    until all weight, source, code-tree, mainline, and anatomy hashes pass.
    ``code_tree_sha256`` defaults to the pinned CODE_TREE_SHA256; only measurement
    and parity kernels pass an explicitly observed digest.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    ids, paths = _study_ids(study_dirs)
    bundle_dir, code_dir, anatomy_weights = (Path(p).resolve() for p in (bundle_dir, code_dir, anatomy_weights))
    verify_bundle(bundle_dir)
    verify_source(source_dir)
    verify_code(code_dir, anatomy_weights, tree_sha256=code_tree_sha256)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"S6 output directory must be empty: {out}")
    write_provenance(out / "provenance.json", bundle_dir=bundle_dir, code_dir=code_dir,
                     anatomy_weights=anatomy_weights, code_tree_sha256=code_tree_sha256)
    with tempfile.TemporaryDirectory(prefix="exp189_stage_", dir=out) as stage_name:
        stage = Path(stage_name)
        _stage_studies(paths, ids, stage)
        runtime_source = _diagnostic_source(source_dir, out)
        args = [sys.executable, "-u", str(runtime_source / "probe_dual.py"),
                "--data", str(stage), "--split", "test", "--code", str(code_dir),
                "--bundle", str(bundle_dir), "--arm", "E_lrprior", "--output", str(out / "controller"),
                "--amp", amp, "--loader-workers", str(workers), "--device", device,
                "--max-fallback-fraction", str(max_fallback_fraction), "--worker", "compact_infer.py",
                "--worker-args", json.dumps([
                    "--backbone", "convnext_small", "--size", "384", "--crop", "352",
                    "--eval-stride", "1", "--views", "v4",
                    "--state-spec", str(Path(bundle_dir) / "state_spec.json"),
                    "--side-model", str(Path(bundle_dir) / "side_cls.pt"),
                    "--side-threshold", "2.0", "--side-position-mm", "20.0", "--side-max-x", "0.0",
                    "--slices", "3", "--phys-mm", "3.5", "--anchor-weights", str(anatomy_weights),
                ])]
        env = dict(os.environ, PYTHONPATH=str(runtime_source) + os.pathsep + str(code_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        (out / 'invocation.json').write_text(json.dumps({"args": args, "device": device,
            "data_source": str(paths[0].parent.parent), "series_sha256": sha256(stage / 'test_series.csv')}, indent=2) + '\n', encoding='utf-8')
        _run_controller(args, env)
    receipt = json.loads((out / "controller" / "submission_receipt.json").read_text(encoding="utf-8"))
    if receipt["rows"] != len(ids) or receipt["fallback_studies"] > max_fallback_fraction * len(ids):
        raise RuntimeError("S6 controller output incomplete")
    rows = read_predictions(out / "controller" / "submission.csv", ids)
    shutil.copyfile(out / "controller" / "submission.csv", out / "predictions.csv")
    try:
        print("S6_ANATOMY_ANCHORS", json.dumps(anatomy_fallbacks(receipt)), flush=True)
    except Exception as exc:  # logging only; public behaviour never rejects on anchors
        try:
            print("S6_ANATOMY_ANCHORS unavailable:", repr(exc)[:200], flush=True)
        except Exception:
            pass
    return rows


def _worker_receipts(controller: dict) -> list[dict]:
    workers = controller.get("workers")
    if not isinstance(workers, list) or not workers or not all(
            isinstance(w, dict) and w.get("status") == "PASS" and isinstance(w.get("stats"), dict)
            for w in workers):
        raise ValueError("S6 worker receipts incomplete")
    return workers


def service_components(controller: dict) -> dict:
    """Per-study S6 service seconds from the worker receipts: decode + render +
    anatomy anchor + GPU (the controller's per_study_mean omits anchor_s)."""
    keys = ("decode_s", "render_s", "anchor_s", "gpu_s")
    totals = dict.fromkeys((*keys, "rows"), 0.0)
    for worker in _worker_receipts(controller):
        for key in totals:
            value = float(worker["stats"][key])
            if not (math.isfinite(value) and value >= 0):
                raise ValueError("S6 worker timing value invalid: " + key)
            totals[key] += value
    rows = totals["rows"]
    if rows <= 0 or rows != controller.get("scored"):
        raise ValueError("S6 worker timing rows differ from scored studies")
    per_study = {key: totals[key] / rows for key in keys}
    return {"totals": totals, "per_study": per_study,
            "service_seconds_per_study": sum(per_study.values())}


def anatomy_fallbacks(controller: dict) -> dict:
    """Scored studies whose knee anchor did not come from the SKM-TEA model (slab prior or
    none). Counted and logged only: the public notebook scores them unchanged."""
    methods: dict[str, int] = {}
    for worker in _worker_receipts(controller):
        for method, count in (worker.get("anchor_methods") or {}).items():
            methods[str(method)] = methods.get(str(method), 0) + int(count)
    scored = int(controller.get("scored", 0))
    model = sum(count for method, count in methods.items() if method.startswith(MODEL_ANCHOR_PREFIX))
    return {"scored": scored, "model_anchor": model, "fallback": scored - model,
            "methods": dict(sorted(methods.items()))}


def _pct(values: Sequence[float]) -> list[float]:
    """pandas Series.rank(method='average', pct=True), including tied values."""
    n = len(values)
    if not n or not all(math.isfinite(v) for v in values):
        raise ValueError("rank inputs must be finite and nonempty")
    order = sorted(range(n), key=lambda i: values[i])
    ranked = [0.0] * n
    k = 0
    while k < n:
        j = k + 1
        while j < n and values[order[j]] == values[order[k]]:
            j += 1
        rank = (k + 1 + j) / (2 * n)
        for pos in order[k:j]:
            ranked[pos] = rank
        k = j
    return ranked


def p954(raptor_pred: Sequence[dict], s6_pred: Sequence[dict]) -> list[dict[str, str | float]]:
    """Public cell 6's 0.70/0.30 per-finding percentile blend.

    Inputs must be complete, aligned raw probability rows. For failed-model
    overrides, use ``p954_with_failures`` with explicit failure masks.
    """
    return p954_with_failures(raptor_pred, s6_pred)


def p954_with_failures(raptor_pred: Sequence[dict], s6_pred: Sequence[dict],
                       raptor_failed: Sequence[bool] | None = None,
                       s6_failed: Sequence[bool] | None = None) -> list[dict[str, str | float]]:
    ids = [str(row[UID]) for row in raptor_pred]
    if ids != [str(row[UID]) for row in s6_pred] or len(ids) != len(set(ids)) or not ids:
        raise ValueError("blend inputs must have identical, unique study order")
    n = len(ids)
    rf = list(raptor_failed) if raptor_failed is not None else [False] * n
    sf = list(s6_failed) if s6_failed is not None else [False] * n
    if len(rf) != n or len(sf) != n:
        raise ValueError("failure masks have incorrect length")
    out = [{UID: uid} for uid in ids]
    for label in LABELS:
        rr = _pct([float(row[label]) for row in raptor_pred])
        sr = _pct([float(row[label]) for row in s6_pred])
        for i in range(n):
            value = 0.7 * rr[i] + 0.3 * sr[i]
            if rf[i]:
                value = sr[i]
            if sf[i]:
                value = rr[i]
            if rf[i] and sf[i]:
                value = 0.5
            out[i][label] = struct.unpack("f", struct.pack("f", value))[0]
    return out


def check_observation(record: dict) -> dict:
    """Validate an observe_code record against every independent pin before it may become
    the CODE_TREE pin: digest recomputation, 8 public module pins, anatomy, wheels, closure."""
    files = record.get("file_sha256")
    if not isinstance(files, dict) or not files or tree_digest(files) != record.get("tree_sha256") \
            or len(files) != record.get("files"):
        raise ValueError("observation tree digest is inconsistent")
    if any(files.get("training/" + name) != digest for name, digest in CODE_HASHES.items()):
        raise ValueError("observation differs from the public mainline module pins")
    if record.get("anatomy_sha256") != ANATOMY_SHA256:
        raise ValueError("observation anatomy.pt differs from the pin")
    if record.get("wheel_sha256") != WHEEL_HASHES:
        raise ValueError("observation wheels differ from the pinned PyPI digests: " + repr(record.get("wheel_sha256")))
    closure = record.get("closure_sha256") or {}
    if any(rel not in closure for rel in DIRECT_CODE_MODULES) or \
            any(files.get(rel) != digest for rel, digest in closure.items()):
        raise ValueError("observation closure is incomplete or inconsistent")
    return {"tree_sha256": record["tree_sha256"], "files": record["files"], "closure_files": len(closure)}


def pin_code(record: dict, *, write: bool = False, path: str | Path = __file__) -> dict:
    """Return (and with write=True, store in this source) the validated code-tree pin."""
    pin = check_observation(record)
    if write:
        target = Path(path)
        text = target.read_text(encoding="utf-8")
        old_digest, old_files = "CODE_TREE_SHA256: str | None = None\n", "CODE_TREE_FILES: int | None = None\n"
        if text.count(old_digest) != 1 or text.count(old_files) != 1:
            raise ValueError("unpopulated CODE_TREE pin lines not found; refusing to overwrite a pin")
        text = text.replace(old_digest, f'CODE_TREE_SHA256: str | None = "{pin["tree_sha256"]}"\n')
        text = text.replace(old_files, f"CODE_TREE_FILES: int | None = {int(pin['files'])}\n")
        target.write_text(text, encoding="utf-8", newline="\n")
        pin["port_sha256"] = sha256(target)
    return pin


def _pin_main(argv: Sequence[str]) -> None:
    ap = argparse.ArgumentParser(prog="exp189_s6_port.py pin-code",
                                 description="Validate an observed code tree and print or store its pin.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--observed", help="exp189 code observation JSON (from a parity/timing kernel output)")
    src.add_argument("--code-source", help="local copy of the code dataset mount (cloud_code/, anatomy.pt, wheels)")
    ap.add_argument("--write", action="store_true", help="store the pin in this source file")
    a = ap.parse_args(argv)
    if a.observed:
        record = json.loads(Path(a.observed).read_text(encoding="utf-8"))
    else:
        with tempfile.TemporaryDirectory(prefix="exp189_pin_") as tmp:
            record = observe_code(a.code_source, Path(tmp) / "observed.json", work_dir=tmp)
    print(json.dumps(pin_code(record, write=a.write), indent=2), flush=True)


def main() -> None:
    if sys.argv[1:2] == ["pin-code"]:
        _pin_main(sys.argv[2:])
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study_dirs", nargs="*", help="directories named by StudyInstanceUID")
    ap.add_argument("--studies-json", help="JSON list of study directory paths")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--anatomy-weights", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", choices=("auto", "fp16", "bf16", "fp32"), default="auto")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--max-fallback-fraction", type=float, default=0.03)
    a = ap.parse_args()
    dirs = list(a.study_dirs)
    if a.studies_json:
        dirs.extend(json.loads(Path(a.studies_json).read_text(encoding="utf-8")))
    rows = infer_studies(dirs, bundle_dir=a.bundle, code_dir=a.code, anatomy_weights=a.anatomy_weights,
                         output_dir=a.output, workers=a.workers, amp=a.amp, device=a.device,
                         max_fallback_fraction=a.max_fallback_fraction)
    print(json.dumps({"rows": len(rows), "output": str(Path(a.output) / "predictions.csv")}), flush=True)


if __name__ == "__main__":
    main()
