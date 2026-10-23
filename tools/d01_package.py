"""Stage D01 privately; build only copies assets and never calls Kaggle.

D01 is fixed before leaderboard feedback: 90% per-target anchor rank plus
10% per-target rank of the five R2 checkpoints' raw probability mean.
The T4 preflight is a serving/numerical check, not independent model validation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

UID = "StudyInstanceUID"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MARKER = "d01-assets-manifest.json"
PRECHECK = "d01-preflight.json"
OPENCV_DATASET = "mattiaangeli/opencv-python-headless-4120088-x86"
OPENCV_SHA256 = "236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9"
SOURCES = ["raptor_common.py", "raptor_runtime.py", "raptor_dicom.py", "raptor_ensemble.py"]
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def blend_frames(anchor, r2, ids):
    """Align by UID, reject malformed frames, and apply the preregistered blend."""
    import numpy as np
    import pandas as pd
    ids = list(map(str, ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("test IDs must be nonempty and unique")
    arrays = []
    for name, frame in (("anchor", anchor), ("r2", r2)):
        if list(frame.columns) != [UID, *LABELS]:
            raise ValueError(f"{name}: wrong target schema/order")
        frame = frame.copy()
        frame[UID] = frame[UID].astype(str)
        if frame[UID].duplicated().any() or set(frame[UID]) != set(ids):
            raise ValueError(f"{name}: missing, extra, or duplicate studies")
        values = frame.set_index(UID).loc[ids, LABELS].astype(float)
        if not np.isfinite(values.to_numpy()).all() or not ((values >= 0) & (values <= 1)).all().all():
            raise ValueError(f"{name}: nonfinite or out-of-range probabilities/ranks")
        arrays.append(values.rank(axis=0, method="average", pct=True).to_numpy())
    result = pd.DataFrame(0.9 * arrays[0] + 0.1 * arrays[1], columns=LABELS)
    result.insert(0, UID, ids)
    return result


def check_preflight(expected, receipt, member_probs, ids, local_f0):
    """Evaluate a fixed device gate; return diagnostics even when it fails."""
    import numpy as np
    ids = list(map(str, ids))
    actual = receipt.get("inference", receipt)
    checks = {
        "study_identity": ids == expected["ids"],
        "input_tensor_hashes": actual.get("input_tensor_hashes") == expected["input_tensor_hashes"],
        "all_members_finite": bool(np.isfinite(member_probs).all()),
        "five_members": tuple(member_probs.shape) == (5, len(ids), 12),
        "complete_no_fallback": actual.get("status") == "COMPLETE" and actual.get("fp32_fallbacks") == 0
                                and actual.get("neutral_fills") == 0 and not actual.get("failures"),
        "volume_mask_hashes": [{key: row.get(key) for key in ("uid", "volume_sha256", "mask_sha256")}
                               for row in receipt.get("study_receipts", [])] == expected["study_hashes"],
    }
    delta = None
    if checks["study_identity"] and checks["five_members"] and checks["all_members_finite"]:
        delta = np.abs(np.asarray(member_probs[0], np.float64) - np.asarray(local_f0, np.float64))
        checks["f0_max_abs"] = bool(delta.max() <= expected["tolerances"]["f0_max_abs"])
        checks["f0_mean_abs"] = bool(delta.mean() <= expected["tolerances"]["f0_mean_abs"])
    else:
        checks.update(f0_max_abs=False, f0_mean_abs=False)
    return {"status": "PASS" if all(checks.values()) else "FAILED", "checks": checks,
            "tolerances": expected["tolerances"], "f0_max_abs": float(delta.max()) if delta is not None else None,
            "f0_mean_abs": float(delta.mean()) if delta is not None else None,
            "scope": "three public test studies; serving/numerics only; no clinical or hidden-runtime claim"}


def _cell(source, kind="code"):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def _bootstrap_code(manifest_hash, mode):
    return f'''# D01 runs in a child process: its strict settings do not mutate the anchor.
from pathlib import Path as _D01Path
import hashlib as _d01_hashlib, os as _d01_os, subprocess as _d01_sp, sys as _d01_sys
_d01_hits = list(_D01Path('/kaggle/input').glob('*/{MARKER}')) + list(_D01Path('/kaggle/input').glob('datasets/*/*/{MARKER}'))
_d01_hits = sorted(set(p.resolve() for p in _d01_hits))
_d01_hits = [p for p in _d01_hits if _d01_hashlib.sha256(p.read_bytes()).hexdigest() == {manifest_hash!r}]
assert len(_d01_hits) == 1, 'D01 pinned private assets missing or changed'
_d01_assets = _d01_hits[0].parent
_d01_comp_hits = [_D01Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), _D01Path('/kaggle/input/rsna-knee-abnormality-detection')]
_d01_comp = next((p for p in _d01_comp_hits if (p/'test.csv').is_file()), None)
assert _d01_comp is not None, 'competition inputs missing'
_d01_cmd = [_d01_sys.executable, str(_d01_assets/'d01_package.py'), {mode!r}, '--assets', str(_d01_assets), '--competition', str(_d01_comp), '--work', '/kaggle/working/d01']
_d01_env = dict(_d01_os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
_d01_sp.run(_d01_cmd, check=True, timeout=3600, env=_d01_env)
'''


def _candidate_finalize_code():
    return '''# Preserve the anchor output, then publish D01 only on complete R2 success.
_d01_anchor = _D01Path('/kaggle/working/anchor0943_submission.csv')
assert not _d01_anchor.exists(), 'refuse a stale anchor output'
_d01_os.replace('/kaggle/working/submission.csv', _d01_anchor)
_d01_sp.run([_d01_sys.executable, str(_d01_assets/'d01_package.py'), 'blend', '--assets', str(_d01_assets), '--competition', str(_d01_comp), '--work', '/kaggle/working/d01', '--anchor', str(_d01_anchor), '--out', '/kaggle/working/submission.csv'], check=True)
'''


def build(root, out, wheels, expected_receipt, expected_csv):
    root, out = Path(root), Path(out)
    if out.exists():
        raise FileExistsError(f"choose a new immutable staging directory: {out}")
    anchor_dir = root / "kaggle/packages/KNEE_ANCHOR_0943"
    anchor_meta = json.loads((anchor_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    anchor_path = anchor_dir / anchor_meta["code_file"]
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    user = anchor_meta["id"].split("/")[0]
    expected_receipt, expected_csv = Path(expected_receipt), Path(expected_csv)
    original = json.loads(expected_receipt.read_text(encoding="utf-8"))
    if original.get("status") != "COMPLETE":
        raise ValueError("expected local inference was not complete")
    if original.get("norm") != "imagenet" or original.get("window_policy") != "legacy":
        raise ValueError("expected local input contract differs from D01")
    if original.get("runtime", {}).get("effective_amp") != "fp16":
        raise ValueError("expected reference is not FP16")
    missing = [name for name in SOURCES if not (root / "training/src" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"serving sources not ready: {missing}")
    wheel_files = list(Path(wheels).glob("*.whl"))
    required = {"timm-1.0.19-", "pydicom-3.0.2-"}
    if any(not any(p.name.startswith(prefix) for p in wheel_files) for prefix in required):
        raise FileNotFoundError("offline timm1.0.19 and pydicom3.0.2 wheels are required; OpenCV comes from its pinned dataset")
    assets = out / "assets"
    assets.mkdir(parents=True)
    members = []
    for fold in range(5):
        name = f"EXP072_R2MIX_R34_F{fold}"
        run = root / "artifacts/experiments" / name
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if (summary["fold"], summary["backbone"], summary["img"], summary["contract"], summary["args"]["k_eval"], summary["args"]["norm"]) != (fold, "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", 224, "raptor64_336_c140_s02", 24, "imagenet"):
            raise ValueError(f"unexpected R2 contract: {name}")
        destination = assets / "weights" / f"r2_f{fold}_best.pt"
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(run / "model_best.pt", destination)
        members.append({"name": f"R2F{fold}", "path": destination.relative_to(assets).as_posix(),
                        "sha256": sha256(destination), "training_run": name, "checkpoint": "best"})
    if members[0]["sha256"] != original["checkpoint_sha256"]:
        raise ValueError("expected F0 predictions belong to a different checkpoint")
    for name in SOURCES:
        destination = assets / "training/src" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "training/src" / name, destination)
    shutil.copy2(Path(__file__), assets / "d01_package.py")
    (assets / "config").mkdir()
    (assets / "config/workspace.toml").write_text('schema_version = 1\nlayout_version = 2\ncompetition = "rsna-knee-abnormality-detection"\n', encoding="utf-8")
    (assets / "wheels").mkdir()
    for wheel in wheel_files:
        shutil.copy2(wheel, assets / "wheels" / wheel.name)
    if (Path(wheels).parent / "provenance.json").is_file():
        shutil.copy2(Path(wheels).parent / "provenance.json", assets / "wheels/provenance.json")
    expected = {"ids": [row["uid"] for row in original["inference"]["input_tensor_hashes"]],
                "input_tensor_hashes": original["inference"]["input_tensor_hashes"],
                "ordered_inputs_sha256": original["inference"]["ordered_inputs_sha256"],
                "checkpoint_sha256": original["checkpoint_sha256"],
                "study_hashes": [{key: row[key] for key in ("uid", "volume_sha256", "mask_sha256")}
                                 for row in original["study_receipts"]],
                "source_receipt_sha256": sha256(expected_receipt), "source_csv_sha256": sha256(expected_csv),
                "local_environment": {key: original["environment"].get(key) for key in ("torch", "cuda", "cudnn", "gpu", "numpy", "timm", "cv2", "pydicom")},
                "tolerances": {"f0_max_abs": 0.002, "f0_mean_abs": 0.0005},
                "tolerance_reason": "fixed engineering gate before fresh T4 observation; not a quality threshold"}
    dump(assets / "validation/expected_test3.json", expected)
    shutil.copy2(expected_csv, assets / "validation/local_f0_test3.csv")
    manifest = {"schema": 1, "candidate": "D01", "members": members, "targets": LABELS,
                "contract": "raptor64_336_c140_s02", "norm": "imagenet", "window_policy": "legacy",
                "res": 224, "k_eval": 24, "amp": "fp16", "batch": 1, "runtime_policy": "strict", "residency": "resident",
                "external_wheels": [{"dataset": OPENCV_DATASET, "sha256": OPENCV_SHA256,
                                     "glob": "opencv_python_headless-4.12.0.88-*.whl"}],
                "blend": {"anchor_rank": 0.9, "r2_rank_of_probability_mean": 0.1, "rank_method": "average", "final_rerank": False},
                "anchor_notebook_sha256": sha256(anchor_path), "anchor_cells": len(anchor["cells"]),
                "caveats": ["R2 weak OOF is indirectly leaky and used mixed labels for epoch selection", "Gold58 is exposed", "three-study preflight cannot establish hidden-cohort runtime"],
                "files": {p.relative_to(assets).as_posix(): sha256(p) for p in sorted(assets.rglob("*")) if p.is_file()}}
    dump(assets / MARKER, manifest)
    digest = sha256(assets / MARKER)
    dump(assets / "dataset-metadata.json", {"title": "knee-d01-r2-assets", "id": f"{user}/knee-d01-r2-assets",
         "licenses": [{"name": "other"}], "isPrivate": True,
         "description": "Private D01 serving assets. Fixed five R2 checkpoints and offline numerical preflight; no clinical quality claim."})
    preflight_id = f"{user}/knee-d01-r2-t4-preflight"
    for kind in ("preflight", "candidate"):
        directory = out / kind; directory.mkdir()
        metadata = copy.deepcopy(anchor_meta)
        metadata.update(id=preflight_id if kind == "preflight" else f"{user}/knee-d01-anchor90-r2-10",
                        title="knee-d01-r2-t4-preflight" if kind == "preflight" else "knee-d01-anchor90-r2-10",
                        code_file=f"d01-{kind}.ipynb", is_private=True, enable_internet=False)
        if kind == "preflight":
            metadata.update(dataset_sources=[f"{user}/knee-d01-r2-assets", OPENCV_DATASET], kernel_sources=[], model_sources=[])
            notebook = {"nbformat": 4, "nbformat_minor": 4, "metadata": copy.deepcopy(anchor.get("metadata", {})),
                        "cells": [_cell("# D01 T4 preflight\nServing/precision check on three public test studies. This is not a competition submission.\n", "markdown"),
                                  _cell(_bootstrap_code(digest, "preflight"))]}
        else:
            metadata["dataset_sources"].append(f"{user}/knee-d01-r2-assets")
            metadata["kernel_sources"].append(preflight_id)
            notebook = copy.deepcopy(anchor)
            notebook["cells"] = [_cell(_bootstrap_code(digest, "candidate-r2")), *notebook["cells"], _cell(_candidate_finalize_code())]
            if notebook["cells"][1:-1] != anchor["cells"]:
                raise AssertionError("anchor cell content changed")
        dump(directory / "kernel-metadata.json", metadata)
        dump(directory / metadata["code_file"], notebook)
    summary = {"assets": str(assets), "asset_manifest_sha256": digest,
               "weights_bytes": sum((assets / member["path"]).stat().st_size for member in members),
               "anchor_original_unchanged": sha256(anchor_path) == manifest["anchor_notebook_sha256"],
               "state": "STAGED_ONLY_NOT_UPLOADED_OR_EXECUTED", "preflight": str(out / "preflight"), "candidate": str(out / "candidate")}
    dump(out / "package-review.json", summary)
    return summary


def _validate_assets(assets):
    assets = Path(assets)
    manifest = json.loads((assets / MARKER).read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        path = (assets / name).resolve()
        if not path.is_relative_to(assets.resolve()) or sha256(path) != digest:
            raise ValueError(f"D01 asset changed: {name}")
    return manifest


def _activate_offline_environment(assets, work):
    destination = work / "environment"
    wheels = sorted((assets / "wheels").glob("*.whl"))
    roots = [Path("/kaggle/input/opencv-python-headless-4120088-x86"),
             Path("/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86")]
    opencv = sorted({p.resolve() for root in roots if root.is_dir()
                     for p in root.rglob("opencv_python_headless-4.12.0.88-*.whl") if sha256(p) == OPENCV_SHA256})
    if len(opencv) != 1:
        raise RuntimeError("pinned external OpenCV wheel is missing or changed")
    wheels.extend(opencv)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(destination), *map(str, wheels)], check=True)
    sys.path.insert(0, str(destination))
    sys.path.insert(0, str(assets / "training/src"))
    os.environ["KNEE_WORKSPACE_ROOT"] = str(assets)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import timm, pydicom, cv2
    if (timm.__version__, pydicom.__version__, cv2.__version__) != ("1.0.19", "3.0.2", "4.12.0"):
        raise RuntimeError("isolated D01 library pins were not respected")


def run_r2(assets, competition, work, preflight):
    assets, competition, work = Path(assets), Path(competition), Path(work)
    if work.exists():
        raise FileExistsError(f"refuse stale D01 work: {work}")
    work.mkdir(parents=True)
    manifest = _validate_assets(assets)
    digest = sha256(assets / MARKER)
    if not preflight:
        hits = list(Path("/kaggle/input").glob(f"*/{PRECHECK}")) + list(Path("/kaggle/input").glob(f"notebooks/*/*/{PRECHECK}"))
        reports = [json.loads(p.read_text()) for p in hits]
        if not any(r.get("status") == "PASS" and r.get("assets_manifest_sha256") == digest for r in reports):
            raise RuntimeError("matching successful T4 preflight output is not attached")
    _activate_offline_environment(assets, work)
    import numpy as np
    import pandas as pd
    import torch
    from raptor_common import environment, write_predictions
    from raptor_dicom import DicomCorpus, DirectoryDicomSource
    from raptor_ensemble import load_members, predict_ensemble
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("D01 requires actual Kaggle T4; local FP16 is not a T4 preflight")
    members, _ = load_members(assets / MARKER)
    start = time.perf_counter()
    with DirectoryDicomSource(competition / "test_series") as source:
        corpus = DicomCorpus(competition / "test.csv", competition / "test_series.csv", source, manifest["contract"])
        mean, probabilities, inference = predict_ensemble(members, corpus, amp="fp16", device="cuda", batch=1,
                                                         residency="resident", runtime_policy="strict")
        if not np.isfinite(probabilities).all() or probabilities.shape != (5, len(corpus.ids), 12):
            raise RuntimeError("incomplete or nonfinite R2 ensemble")
        receipt = {"status": "COMPLETE", "assets_manifest_sha256": digest, "inference": inference,
                   "study_receipts": [corpus.receipts[uid] for uid in corpus.ids], "members": manifest["members"],
                   "environment": environment(), "seconds": time.perf_counter() - start, "ids": list(corpus.ids)}
        write_predictions(work / "r2-mean.csv", corpus.ids, mean)
        np.savez_compressed(work / "r2-members.npz", ids=np.asarray(corpus.ids), member_probabilities=probabilities, mean_probabilities=mean)
        dump(work / "r2-receipt.json", receipt)
        if preflight:
            expected = json.loads((assets / "validation/expected_test3.json").read_text())
            local = pd.read_csv(assets / "validation/local_f0_test3.csv", dtype={UID: str}).set_index(UID).loc[expected["ids"], LABELS].to_numpy()
            report = check_preflight(expected, receipt, probabilities, corpus.ids, local)
            report.update(assets_manifest_sha256=digest, environment=environment(), r2_seconds=receipt["seconds"],
                          receipt_sha256=sha256(work / "r2-receipt.json"))
            dump(work.parent / PRECHECK, report)
            if report["status"] != "PASS":
                raise RuntimeError("D01 preflight failed; inspect d01-preflight.json")
    return receipt


def finalize(assets, competition, work, anchor, out):
    import pandas as pd
    assets, competition, work, anchor, out = map(Path, (assets, competition, work, anchor, out))
    if out.exists():
        raise FileExistsError(f"refuse existing D01 submission: {out}")
    receipt = json.loads((work / "r2-receipt.json").read_text())
    if receipt.get("status") != "COMPLETE" or receipt.get("assets_manifest_sha256") != sha256(assets / MARKER):
        raise RuntimeError("R2 inference incomplete or from different assets")
    ids = pd.read_csv(competition / "test.csv", dtype={UID: str})[UID].tolist()
    result = blend_frames(pd.read_csv(anchor, dtype={UID: str}), pd.read_csv(work / "r2-mean.csv", dtype={UID: str}), ids)
    temporary = out.with_suffix(".csv.tmp")
    result.to_csv(temporary, index=False)
    os.replace(temporary, out)
    dump(work / "d01-release.json", {"status": "COMPLETE", "candidate": "D01", "studies": len(ids),
         "formula": "0.9*rankpct(anchor)+0.1*rankpct(mean5(raw_R2)); average ties; no final rerank",
         "anchor_sha256": sha256(anchor), "r2_sha256": sha256(work / "r2-mean.csv"), "submission_sha256": sha256(out),
         "assets_manifest_sha256": sha256(assets / MARKER)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--root", default=str(ROOT))
    build_parser.add_argument("--out", required=True)
    build_parser.add_argument("--wheels", required=True)
    build_parser.add_argument("--expected-receipt", default=str(ROOT / "artifacts/predictions/serving-20261003/R2F0_test3_dicom.json"))
    build_parser.add_argument("--expected-csv", default=str(ROOT / "artifacts/predictions/serving-20261003/R2F0_test3_dicom.csv"))
    for action in ("preflight", "candidate-r2", "blend"):
        child = sub.add_parser(action)
        for flag in ("assets", "competition", "work"):
            child.add_argument(f"--{flag}", required=True)
        if action == "blend":
            child.add_argument("--anchor", required=True)
            child.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.action == "build":
        print(json.dumps(build(args.root, args.out, args.wheels, args.expected_receipt, args.expected_csv), indent=2))
    elif args.action == "blend":
        finalize(args.assets, args.competition, args.work, args.anchor, args.out)
    else:
        run_r2(args.assets, args.competition, args.work, args.action == "preflight")


if __name__ == "__main__":
    main()
