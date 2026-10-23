"""Versioned CPU numerical input gate; no model inference, training or submission.

Freeze the protocol and expected integer-input hashes before evaluating the100UID
numerical holdout. It is not a clinical holdout. Original D01v1 FAIL is unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ASSET_SHA = "e62893ebb1e22dbb1288ea6c805097bbc96ef5e43759f42a22d7cfb3bbef036b"
GDCM_SHA = "512d0d903e12c5ad389939a257817bf182ce288431b3d474d62cb0c0833b8b56"
UID = "StudyInstanceUID"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def array_sha(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def activate_sources(assets):
    assets = Path(assets)
    if sha(assets / "d01-assets-manifest.json") != ASSET_SHA:
        raise ValueError("D01v1 manifest identity mismatch")
    manifest = read(assets / "d01-assets-manifest.json")
    for name, expected in manifest["files"].items():
        if name.startswith(("training/", "wheels/", "config/")) or name == "d01_package.py":
            if sha(assets / name) != expected:
                raise ValueError(f"used D01v1 source changed: {name}")
    os.environ["KNEE_WORKSPACE_ROOT"] = str(assets.resolve())
    sys.path.insert(0, str(assets / "training/src"))


def constants():
    import numpy as np
    return (np.array([.485, .456, .406], np.float32).reshape(1, 3, 1, 1),
            np.array([.229, .224, .225], np.float32).reshape(1, 3, 1, 1))


def reference(volume, centers):
    """Independent float64 reference for the specific336to224 input contract."""
    import numpy as np
    if volume.dtype != np.uint8 or volume.shape[-2:] != (336, 336):
        raise ValueError("reference requires uint8,336x336 planes")
    if any(int(c) != c or not 1 <= c < len(volume)-1 for c in centers):
        raise ValueError("reference requires valid integer triplet centers")
    q = (np.arange(224, dtype=np.float64) + .5) * 1.5 - .5
    lo = np.floor(q).astype(int); hi = lo + 1; weight = q - lo
    if set(np.unique(weight)) != {.25, .75}:
        raise AssertionError("nonexact coordinate coefficients")
    x = np.stack([volume[int(c)-1:int(c)+2] for c in centers]).astype(np.float64) / 255.0
    horizontal = x[..., lo] * (1-weight) + x[..., hi] * weight
    windows = horizontal[..., lo, :] * (1-weight)[:, None] + horizontal[..., hi, :] * weight[:, None]
    mean, std = constants()
    return windows, (windows - mean.astype(np.float64)) / std.astype(np.float64)


def compare(actual, expected, tolerance):
    import numpy as np
    if actual.shape != expected.shape or actual.dtype != np.float32:
        return {"pass": False, "reason": "shape or dtype mismatch", "shape": list(actual.shape), "dtype": str(actual.dtype)}
    if not np.isfinite(actual).all():
        return {"pass": False, "reason": "nonfinite tensor"}
    delta = np.abs(actual.astype(np.float64) - expected)
    bad = int(np.count_nonzero(delta > tolerance))
    return {"pass": bad == 0, "max_abs": float(delta.max()), "rms": float(np.sqrt(np.mean(delta*delta))),
            "mean_abs": float(delta.mean()), "exceedances": bad, "atol": tolerance, "rtol": 0.0}


def expected_row(uid, volume, mask, centers):
    return {"uid": str(uid), "volume_shape": list(volume.shape), "volume_dtype": str(volume.dtype),
            "mask_shape": list(mask.shape), "mask_dtype": str(mask.dtype),
            "volume_sha256": array_sha(volume), "mask_sha256": array_sha(mask), "centers": list(map(int, centers))}


def check_case(volume, mask, centers, expected, protocol):
    import numpy as np
    from raptor_common import windows_from
    from raptor_runtime import normalize_input, IMAGENET_MEAN, IMAGENET_STD
    exact = expected_row(expected["uid"], volume, mask, centers) == expected
    coeffs = array_sha(IMAGENET_MEAN) == protocol["mean_sha256"] and array_sha(IMAGENET_STD) == protocol["std_sha256"]
    if not exact or not coeffs:
        return {"uid": expected["uid"], "pass": False, "exact_input": exact, "exact_coefficients": coeffs}
    wref, nref = reference(volume, centers)
    windows = windows_from(volume, centers, 224)
    normalized = normalize_input(windows, "imagenet")
    wcheck = compare(windows, wref, protocol["window_atol"])
    ncheck = compare(normalized, nref, protocol["normalized_atol"])
    mean, std = constants()
    low = -mean.astype(np.float64)/std.astype(np.float64)
    high = (1-mean.astype(np.float64))/std.astype(np.float64)
    ranges = bool(np.isfinite(windows).all() and np.isfinite(normalized).all()
                  and (windows >= -protocol["window_atol"]).all() and (windows <= 1+protocol["window_atol"]).all()
                  and (normalized >= low-protocol["normalized_atol"]).all()
                  and (normalized <= high+protocol["normalized_atol"]).all())
    return {"uid": expected["uid"], "pass": exact and coeffs and ranges and wcheck["pass"] and ncheck["pass"],
            "exact_input": exact, "exact_coefficients": coeffs, "finite_channel_ranges": ranges,
            "windows": wcheck, "normalized": ncheck,
            "window_tensor_sha256": array_sha(windows), "normalized_tensor_sha256": array_sha(normalized)}


def synthetic_cases():
    import numpy as np
    y, x = np.indices((336, 336)); rng = np.random.default_rng(20261003)
    cases = {"zero": np.zeros((336,336), np.uint8), "one": np.full((336,336),255,np.uint8),
             "checkerboard": ((x+y)%2*255).astype(np.uint8), "horizontal_ramp": (x%256).astype(np.uint8),
             "vertical_ramp": (y%256).astype(np.uint8), "random": rng.integers(0,256,(336,336),dtype=np.uint8)}
    impulse = np.zeros((336,336),np.uint8); impulse[167,167]=255; cases["impulse"] = impulse
    for name, image in cases.items():
        yield name, np.repeat(image[None],3,axis=0), np.ones(3,np.uint8), [1]


def freeze(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError("new immutable staging directory required")
    assets = ROOT / ".local/d01-staging/20261003-v1/assets"
    activate_sources(assets)
    from raptor_common import Corpus, select_centers
    selection = ROOT / ".local/d01-runtime-staging/20261003-v4/selection-manifest.json"
    selected = read(selection)
    corpus_dir = ROOT / "storage/cache/derived/RAPTOR64_336_C140_S02"
    corpus = Corpus(corpus_dir)
    records = []
    for uid in selected["ids"]:
        volume, mask = corpus.volume(uid), corpus.mask(uid)
        centers = select_centers(mask, corpus.depth, 24, corpus.spec, policy="legacy")
        records.append(expected_row(uid, volume, mask, centers))
    data = out / "assets"; data.mkdir(parents=True)
    expected = {"ids": selected["ids"], "rows": records,
                "synthetic": [expected_row(name, vol, mask, centers) for name,vol,mask,centers in synthetic_cases()],
                "source": "immutable local corpus integer bytes and centers; no resize/normalized holdout outputs observed during freeze"}
    dump(data / "expected-inputs.json", expected)
    shutil.copy2(__file__, data / "d01_input_gate.py")
    mean, std = constants()
    protocol = {"schema": "d01-numerical-input-gate-v2", "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "status": "FROZEN_BEFORE_100_NUMERICAL_OUTPUTS", "v1_result": "FAILED_UNCHANGED",
                "helper_sha256": sha(data / "d01_input_gate.py"), "expected_sha256": sha(data / "expected-inputs.json"),
                "selection_sha256": sha(selection), "assets_manifest_sha256": ASSET_SHA,
                "train_series_sha256": selected["train_series_sha256"],
                "corpus_manifest_sha256": sha(corpus_dir / "manifest.json"),
                "window_atol": 5e-7, "normalized_atol": 3e-6, "rtol": 0.0,
                "derived_cross_platform_atol": {"windows": 1e-6, "normalized": 6e-6},
                "mean_sha256": array_sha(mean), "std_sha256": array_sha(std),
                "contract": {"volume_shape": [64,336,336], "volume_dtype": "uint8", "centers": 24,
                             "window_shape": [24,3,224,224], "window_dtype": "float32", "mode": "bilinear",
                             "align_corners": False, "antialias": False, "window_policy": "legacy", "norm": "imagenet"},
                "math": "336/224=1.5, in-bounds quarter/three-quarter coefficients exact; fp32 envelope4.7684e-7 window and2.4435e-6 normalized, under standard equivalent bilinear arithmetic; not a compiler proof",
                "numerical_holdout": "same fixed engineering100 as runtime100; disjoint public3; no clinical independence claim",
                "pass_rule": "all100 and7synthetics pass integer identity,coefficients,shape,dtype,finite,range and elementwise absolute reference bounds; anyfailure fails wholegate; no tolerance relaxation after observation",
                "separate_gates": ["allfiveT4model inference and original probability tolerances", "decoder capabilities/fixtures", "anchor49+ and total9h runtime"],
                "gdcm_wheel_sha256": GDCM_SHA}
    dump(data / "protocol.json", protocol)
    dump(data / "dataset-metadata.json", {"title": "knee-d01-input-gate-v2", "id": "taeyangg4/knee-d01-input-gate-v2",
                                          "licenses": [{"name":"other"}], "isPrivate":True,
                                          "description":"Private frozen CPU numerical engineering protocol and input hashes; no images, labels or reports."})
    notebook_dir = out / "preflight"; notebook_dir.mkdir()
    psha = sha(data / "protocol.json")
    code = f'''from pathlib import Path
import hashlib, os, subprocess, sys
def find(name, digest):
    paths = list(Path('/kaggle/input').glob('*/'+name)) + list(Path('/kaggle/input').glob('datasets/*/*/'+name))
    paths = sorted(set(p.resolve() for p in paths if hashlib.sha256(p.read_bytes()).hexdigest()==digest))
    assert len(paths)==1, 'pinned input missing: '+name
    return paths[0].parent
gate = find('protocol.json', {psha!r})
assets = find('d01-assets-manifest.json', {ASSET_SHA!r})
codec = next(p for p in [Path('/kaggle/input/datasets/taeyangg4/knee-d01-codec-assets'),Path('/kaggle/input/knee-d01-codec-assets')] if p.is_dir())
competition = next(p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p/'train_series.csv').is_file())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run([sys.executable,str(gate/'d01_input_gate.py'),'run','--gate',str(gate),'--assets',str(assets),'--competition',str(competition),'--codec',str(codec),'--out','/kaggle/working/input-gate-v2'],check=True,timeout=1500,env=env)
'''
    nb = {"nbformat":4,"nbformat_minor":4,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"}},
          "cells":[{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(keepends=True)}]}
    meta = read(ROOT / ".local/d01-codec-staging/20261003-v1/preflight-v2/kernel-metadata.json")
    meta.update(id="taeyangg4/knee-d01-input-gate-v2",title="knee-d01-input-gate-v2",code_file="input-gate-v2.ipynb",
                competition_sources=["rsna-knee-abnormality-detection"])
    meta["dataset_sources"].append("taeyangg4/knee-d01-input-gate-v2")
    dump(notebook_dir / "input-gate-v2.ipynb", nb); dump(notebook_dir / "kernel-metadata.json",meta)
    return protocol


def run(gate, assets, out, corpus=None, competition=None, codec=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gate, assets, out = map(Path,(gate,assets,out)); out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); report={"status":"RUNNING","cases":[],"synthetic":[],"failures":[],"mode":"local_cache" if corpus else "remote_dicom"}
    source=None
    try:
        protocol=read(gate / "protocol.json")
        report.update(protocol_sha256=sha(gate / "protocol.json"),helper_sha256=sha(__file__))
        if sha(__file__) != protocol["helper_sha256"] or sha(gate / "expected-inputs.json") != protocol["expected_sha256"]:
            raise ValueError("frozen helper/expected identity changed")
        expected=read(gate / "expected-inputs.json")
        activate_sources(assets)
        if not corpus:
            wheels=[p for p in Path(codec).rglob("*.whl") if sha(p)==GDCM_SHA]
            if len(wheels)!=1:
                raise ValueError("pinned gdcm wheel missing")
            destination=out / "codec"
            subprocess.run([sys.executable,"-m","pip","install","--no-index","--no-deps","--target",str(destination),str(wheels[0])],check=True)
            sys.path.insert(0,str(destination))
            spec=importlib.util.spec_from_file_location("d01_pinned_package",assets / "d01_package.py")
            package=importlib.util.module_from_spec(spec); spec.loader.exec_module(package)
            package._activate_offline_environment(assets,out)
        import torch
        torch.set_num_threads(2)
        from raptor_common import Corpus, select_centers, environment
        from raptor_dicom import DirectoryDicomSource, DicomCorpus
        report["environment"]=environment()
        if corpus:
            if sha(Path(corpus)/"manifest.json") != protocol["corpus_manifest_sha256"]:
                raise ValueError("local corpus manifest changed")
            inputs=Corpus(corpus)
        else:
            from pydicom.pixels import get_decoder
            import gdcm
            report["gdcm_version"]=gdcm.Version.GetVersion()
            report["lossless_decoders"]={uid:{"available":bool(get_decoder(uid).is_available),"plugins":list(get_decoder(uid).available_plugins)} for uid in ("1.2.840.10008.1.2.4.57","1.2.840.10008.1.2.4.70")}
            if not all(x["available"] for x in report["lossless_decoders"].values()):
                raise RuntimeError("JPEG Lossless decoder unavailable")
            competition=Path(competition)
            if sha(competition/"train_series.csv") != protocol["train_series_sha256"]:
                raise ValueError("series metadata changed")
            with (competition/"train_series.csv").open(newline="",encoding="utf-8-sig") as f:
                metadata=list(csv.DictReader(f))
            source=DirectoryDicomSource(competition/"train_series")
            inputs=DicomCorpus.from_records(expected["ids"],metadata,source,"raptor64_336_c140_s02",split="train")
        if inputs.contract != "raptor64_336_c140_s02":
            raise ValueError("wrong corpus contract")
        for index, record in enumerate(expected["rows"]):
            uid=record["uid"]; volume,mask=inputs.volume(uid),inputs.mask(uid)
            centers=select_centers(mask,inputs.depth,24,inputs.spec,policy="legacy")
            result=check_case(volume,mask,centers,record,protocol)
            report["cases"].append(result)
            if (index+1)%10==0:
                report["elapsed_seconds"]=time.perf_counter()-started
                dump(out/"receipt.json",report); print(f'input gate {index+1}/{len(expected["rows"])}',flush=True)
        for record,case in zip(expected["synthetic"],synthetic_cases(),strict=True):
            name,volume,mask,centers=case
            if name!=record["uid"]:
                raise ValueError("synthetic case order changed")
            report["synthetic"].append(check_case(volume,mask,centers,record,protocol))
        ok=len(report["cases"])==100 and len(report["synthetic"])==7 and all(r["pass"] for r in report["cases"]+report["synthetic"])
        report["status"]="PASS" if ok else "FAILED"
        if hasattr(inputs,"receipts"):
            report["dicom_receipts"]=list(inputs.receipts.values())
        if not ok:
            raise RuntimeError("frozen numerical input gate failed; inspect per-case evidence")
    except Exception as exc:
        report["status"]="FAILED"
        failure={"error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}
        if hasattr(exc,"receipt"):
            failure["source_receipt"]=exc.receipt
        report["failures"].append(failure)
        raise
    finally:
        if source: source.close()
        report["elapsed_seconds"]=time.perf_counter()-started
        dump(out/"receipt.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="action",required=True)
    freeze_parser=sub.add_parser("freeze"); freeze_parser.add_argument("--out",required=True)
    run_parser=sub.add_parser("run")
    for name in ("gate","assets","out"):
        run_parser.add_argument("--"+name,required=True)
    for name in ("corpus","competition","codec"):
        run_parser.add_argument("--"+name)
    args=vars(parser.parse_args()); action=args.pop("action")
    result=freeze(**args) if action=="freeze" else run(**args)
    print(json.dumps({"status":result["status"]}))


if __name__=="__main__": main()
