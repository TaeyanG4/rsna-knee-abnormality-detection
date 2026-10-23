"""Offline, hash-authenticated EXP143-v2 own-only anchor for EXP169 v2.

This builder does not import torch, deserialize checkpoints, run inference or
call an account API. It copies the already authenticated original average and
reuses the frozen EXP152 notebook/runtime. Runtime ``checked_model`` retains
the complete ``own_meta`` and tensor-schema checks before model inference.
The caller validates the verification verdict; this helper binds its note.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.1"
BASE_NAME = "exp152_own0949_release.py"
BASE_SHA256 = "ebb8e06734b72f40f090389b23d8767137c8eb6c92dee547367c1f18188f94e2"
AVERAGE = Path("artifacts/releases/EXP143_S42v2_average_v101")
AVERAGE_RECEIPT_SHA256 = "d6bf6ad480baa46ad80b0a17ef9739387a80753c616ac6dd9a3b888def5d4a86"
CHECKPOINT_SHA256 = "a163f3f8fa7ae57a32ea5237ce36501f74331d7fcec826bc50eb904a8e2f3295"
TRAINING_PROTOCOL = Path("artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json")
TRAINING_PROTOCOL_SHA256 = "b2aaca64289249595e532b74d361175ed78b1170109eee302ec29068e0e7af11"
PROTOCOL = Path("artifacts/experiments/EXP169_RELEASE_V2/PROTOCOL.md")
PROTOCOL_SHA256 = "a15f1c51890b67802b07aefdb683234fe6290ea9107b85322067a65cf3fe0ce0"
DEFERRED_CHECKS = [
    "EXP152 checked_model: own_meta frozen inference fields and protocol/run identity",
    "EXP152 checked_model: exact model state keys, shapes and dtypes",
    "EXP152 checked_model: strict model.load_state_dict before inference",
]


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_binding(path):
    path = Path(path)
    return {"sha256": _sha(path), "bytes": path.stat().st_size}


def _load_shared(root):
    path = root / "tools" / BASE_NAME
    _need(_sha(path) == BASE_SHA256, "Frozen EXP152 source differs")
    # A private instance avoids changing independently imported EXP152 globals.
    spec = importlib.util.spec_from_file_location(__name__ + "_exp152", path)
    _need(spec is not None and spec.loader is not None, "EXP152 import is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stage(out, dataset, verification_note, root=ROOT):
    """Create an own-only EXP152-compatible stage; never upload or run it.

    ``verification_note`` must be an existing, caller-approved note inside
    ``root``. Its bytes, both protocols, builder source and every output other
    than the build receipt itself are bound in ``build-receipt.json``. The
    returned dictionary additionally supplies that receipt's SHA256.
    """
    root = Path(root).resolve()
    out = Path(out)
    out = (root / out).resolve() if not out.is_absolute() else out.resolve()
    note = Path(verification_note)
    note = (root / note).resolve() if not note.is_absolute() else note.resolve()
    _need(out.is_relative_to(root) and out != root, "Stage must be inside the project")
    _need(not out.exists(), "Choose a fresh own-anchor stage directory")
    _need(note.is_relative_to(root) and note.is_file(), "Verification note must exist inside the project")
    _need(isinstance(dataset, str) and re.fullmatch(r"[a-z0-9_-]+/[a-z0-9_-]+", dataset),
          "Invalid dataset ID")
    _need(_sha(root / PROTOCOL) == PROTOCOL_SHA256, "Registered EXP169 v2 protocol differs")
    _need(_sha(root / TRAINING_PROTOCOL) == TRAINING_PROTOCOL_SHA256,
          "Registered EXP143 v2 protocol differs")
    receipt_path = root / AVERAGE / "average-receipt.json"
    _need(_sha(receipt_path) == AVERAGE_RECEIPT_SHA256, "Authenticated average receipt differs")
    receipt = _read_json(receipt_path)
    shared = _load_shared(root)
    _need(receipt.get("schema") == "exp152-average-v1" and
          receipt.get("version") == shared.VERSION and
          receipt.get("status") == "VERIFIED_FIXED_SWA_BN" and
          receipt.get("checkpoint") == shared.OWN_NAME and
          receipt.get("checkpoint_sha256") == CHECKPOINT_SHA256 and
          receipt.get("protocol_sha256") == TRAINING_PROTOCOL_SHA256 and
          receipt.get("run_dir") == shared.RUN and receipt.get("oai") is False and
          receipt.get("epochs_1based") == shared.EPOCHS,
          "Authenticated own average identity differs")
    _need(isinstance(receipt.get("state_schema"), dict) and receipt["state_schema"],
          "Authenticated own state schema is missing")
    checkpoint = root / AVERAGE / shared.OWN_NAME
    _need(_sha(checkpoint) == CHECKPOINT_SHA256, "Authenticated own checkpoint differs")
    verification = {"path": note.relative_to(root).as_posix(), "sha256": _sha(note),
                    "verdict_validation": "CALLER_RESPONSIBILITY"}
    builder = {"version": VERSION, "source_sha256": _sha(Path(__file__)),
               "release_protocol_sha256": PROTOCOL_SHA256,
               "verification_note": verification,
               "checkpoint_deserialized_at_stage": False,
               "deferred_runtime_metadata_checks": DEFERRED_CHECKS}
    source, credits = shared.public_source()
    serving = shared.serving_source(source)
    out.mkdir(parents=True)
    assets, ds, own = out / "assets", out / "dataset", out / "own"
    for directory in (assets, ds, own):
        directory.mkdir()
    shutil.copyfile(checkpoint, assets / shared.OWN_NAME)
    shutil.copyfile(root / "tools" / BASE_NAME, assets / BASE_NAME)
    shared.write_text(assets / "public-serving.py", serving)
    _need(_sha(assets / shared.OWN_NAME) == CHECKPOINT_SHA256,
          "Copied own checkpoint differs")
    _need(_sha(assets / BASE_NAME) == BASE_SHA256, "Copied EXP152 source differs")
    manifest = dict(
        schema="exp152-release-v1", version=shared.VERSION,
        protocol_sha256=TRAINING_PROTOCOL_SHA256, epochs_1based=shared.EPOCHS,
        average_receipt_sha256=AVERAGE_RECEIPT_SHA256,
        source_final_sha256=receipt["source_final_sha256"],
        public_source_sha256=shared.SOURCE_SHA,
        serving_sha256=_sha(assets / "public-serving.py"),
        checkpoints={shared.OWN_NAME: dict(sha256=CHECKPOINT_SHA256, oai=False)},
        own_state_schema=receipt["state_schema"],
        asset_sha256={name: _sha(assets / name) for name in
                      (shared.OWN_NAME, BASE_NAME, "public-serving.py")},
        ranks=dict(own="public numpy double argsort / (N-1)"),
        policy=shared.POLICY, tolerance=shared.TOL, dataset=dataset,
        source_bindings=receipt["source_bindings"],
        approval_required={"own": "each submission"},
        own_anchor_builder=builder,
        qualification="UNQUALIFIED_NOT_PUSHED",
        final_b_rank_source="authenticated raw.npz own probabilities; average ties; not the ordinal standalone CSV",
    )
    shared.write_json(assets / "release-manifest.json", manifest)
    shared.verify_assets(assets, manifest)
    archive = ds / "exp152-assets.zip.bin"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(assets.iterdir()):
            bundle.write(path, path.name)
    shared.write_json(ds / "dataset-metadata.json", dict(
        title=dataset.split("/")[1], id=dataset, isPrivate=True,
        licenses=[{"name": "other"}],
        description="Private own EXP143 v2 OAI-free checkpoint and frozen EXP152 serving runtime. No public/OAI checkpoint or source training data.",
    ))
    nb, title, description = shared.notebook(
        "own", dataset, _sha(archive), _sha(assets / "release-manifest.json"),
        BASE_SHA256, credits,
    )
    metadata = shared.kernel_metadata("own", dataset)
    _need(metadata["dataset_sources"] == [dataset] and metadata["is_private"] is True and
          metadata["enable_internet"] is False, "Own-only private notebook metadata differs")
    shared.write_json(own / "release.ipynb", nb)
    shared.write_json(own / "kernel-metadata.json", metadata)
    shared.write_text(own / "DESCRIPTION.md", "# " + title + "\n\n" + description + "\n")
    _need(_sha(note) == verification["sha256"], "Verification note changed during staging")
    files = {path.relative_to(out).as_posix(): _file_binding(path)
             for directory in (assets, ds, own) for path in sorted(directory.iterdir())}
    report = dict(
        schema="exp169-own-anchor-stage-v1", status="STAGED_NOT_UPLOADED",
        version=shared.VERSION, builder=builder, private_required=True,
        dataset=dataset, qualification="UNQUALIFIED_NOT_PUSHED", oai=False,
        executed_arms=[], allowed_arms=["EXP143_V2"],
        archive_sha256=files["dataset/exp152-assets.zip.bin"]["sha256"],
        manifest_sha256=files["assets/release-manifest.json"]["sha256"],
        dataset_metadata_sha256=files["dataset/dataset-metadata.json"]["sha256"],
        kernels={"own": {name: files["own/" + name]["sha256"] for name in
                         ("release.ipynb", "kernel-metadata.json", "DESCRIPTION.md")}},
        projection=shared.projection(), files=files,
        gpu_runs=0, uploads=0, submissions=0,
    )
    shared.write_json(out / "build-receipt.json", report)
    return dict(status=report["status"], out=str(out), dataset=dataset,
                manifest_sha256=report["manifest_sha256"],
                archive_sha256=report["archive_sha256"],
                build_receipt_sha256=_sha(out / "build-receipt.json"))
