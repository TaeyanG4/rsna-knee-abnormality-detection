"""Stage a private CPU input-I/O comparison, reusing immutable D01 model assets.

No Kaggle calls, predictions, submissions, training or account writes occur here.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
ASSET_SHA = "e62893ebb1e22dbb1288ea6c805097bbc96ef5e43759f42a22d7cfb3bbef036b"
BASELINE_SHA = "e478626baaf757260de3f8b03f143e727ebb130f575404788e2d9dad1264867e"
GDCM_SHA = "512d0d903e12c5ad389939a257817bf182ce288431b3d474d62cb0c0833b8b56"


def encode(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "zlib_base64": base64.b64encode(zlib.compress(raw, 9)).decode("ascii")}


def build(out, selection_path, reference_path, parity_path, ordering_reference=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError("choose a new immutable staging directory")
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    parity = json.loads(Path(parity_path).read_text(encoding="utf-8"))
    adapter = ROOT / "training/src/raptor_dicom.py"
    adapter_sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
    if parity["status"] != "PASS" or parity["ids"] != selection["ids"] or len(parity["ids"]) != 100:
        raise ValueError("full fixed100 local parity required before remote staging")
    if parity["source_sha256"]["candidate"] != adapter_sha or parity["source_sha256"]["baseline"] != BASELINE_SHA:
        raise ValueError("local parity code identity changed")
    reference = {"study_receipts": reference["study_receipts"]}
    selection = {"ids": selection["ids"], "train_series_sha256": selection["train_series_sha256"]}
    payload = {
        "raptor_dicom_header_v1.py": encode(adapter.read_bytes()),
        "dicom_header_benchmark.py": encode((ROOT / "tools/dicom_header_benchmark.py").read_bytes()),
        "selection.json": encode(json.dumps(selection, separators=(",", ":")).encode()),
        "reference.json": encode(json.dumps(reference, separators=(",", ":")).encode()),
    }
    extra_args = []
    if ordering_reference:
        ordering = json.loads(Path(ordering_reference).read_text(encoding="utf-8"))
        if ordering.get("version") != "directory-order-contract-v2" or ordering["ids"] != selection["ids"]:
            raise ValueError("ordering reference must bind the exact fixed100 cohort")
        payload["ordering-reference.json"] = encode(Path(ordering_reference).read_bytes())
        extra_args = ["--ordering-reference", "/kaggle/working/d01-header-io/ordering-reference.json"]
    code = f'''import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import ast, base64, hashlib, importlib.util, json, subprocess, sys, time, zlib
from pathlib import Path
base = Path('/kaggle/working/d01-header-io')
base.mkdir(exist_ok=False)
payload = {payload!r}
for name, entry in payload.items():
    raw = zlib.decompress(base64.b64decode(entry['zlib_base64']))
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
    (base/name).write_bytes(raw)
    if name.endswith('.py'): ast.parse(raw.decode('utf-8'))
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
roots = [Path('/kaggle/input/knee-d01-r2-assets'), Path('/kaggle/input/datasets/taeyangg4/knee-d01-r2-assets')]
hits = [p for root in roots if root.is_dir() for p in root.rglob('d01-assets-manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest() == {ASSET_SHA!r}]
assert len(hits) == 1, 'immutable D01 asset identity unavailable'
assets = hits[0].parent
baseline = assets/'training/src/raptor_dicom.py'
assert hashlib.sha256(baseline.read_bytes()).hexdigest() == {BASELINE_SHA!r}
competition = next((p for p in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p/'train_series.csv').is_file()), None)
assert competition is not None
package = load(assets/'d01_package.py', 'pinned_d01_package')
package._validate_assets(assets)
package._activate_offline_environment(assets, base)
codec_roots = [Path('/kaggle/input/knee-d01-codec-assets'), Path('/kaggle/input/datasets/taeyangg4/knee-d01-codec-assets')]
wheels = [p for root in codec_roots if root.is_dir() for p in root.rglob('python_gdcm-3.2.6-*.whl') if hashlib.sha256(p.read_bytes()).hexdigest() == {GDCM_SHA!r}]
assert len(wheels) == 1, 'pinned gdcm wheel unavailable'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', str(base/'environment'), str(wheels[0])], check=True)
import torch
assert not torch.cuda.is_available(), 'CPU benchmark may not allocate GPU'
torch.set_num_threads(2)
benchmark = load(base/'dicom_header_benchmark.py', 'header_benchmark')
result = benchmark.main(['--selection', str(base/'selection.json'), '--series-csv', str(competition/'train_series.csv'),
    '--baseline-adapter', str(baseline), '--candidate-adapter', str(base/'raptor_dicom_header_v1.py'),
    '--reference-receipt', str(base/'reference.json'), '--dicom-root', str(competition/'train_series'),
    '--count', '100', '--out', str(base/'results')] + {extra_args!r})
print(json.dumps({{'status': result['status'], 'summary': result['summary']}}, indent=2))
'''
    ast.parse(code)
    notebook = {"nbformat": 4, "nbformat_minor": 4, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}}, "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# D01 input I/O paired check\nFixed 100 engineering studies, CPU only. No accuracy metric, model inference or submission.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)}]}
    metadata = json.loads((ROOT / ".local/d01-runtime-staging/20261003-v4/kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.update(id="taeyangg4/knee-d01-header-io100", title="knee-d01-header-io100",
                    code_file="d01-header-io100.ipynb", enable_gpu=False, keywords=["cpu"])
    metadata.pop("machine_shape", None)
    metadata["dataset_sources"].append("taeyangg4/knee-d01-codec-assets")
    out.mkdir(parents=True)
    review = {"status": "STAGED_ONLY", "scope": "paired fixed100 CPU input I/O; not end-to-end T4 or hidden runtime proof",
              "asset_manifest_sha256": ASSET_SHA, "baseline_adapter_sha256": BASELINE_SHA,
              "candidate_adapter_sha256": adapter_sha, "payloads": {k: {a: b for a, b in v.items() if a != "zlib_base64"} for k, v in payload.items()},
              "local_parity_path": str(Path(parity_path).resolve()),
              "local_parity_sha256": hashlib.sha256(Path(parity_path).read_bytes()).hexdigest(),
              "command_not_executed": f'.venv/Scripts/kaggle.exe kernels push -p "{out.resolve()}" --accelerator None --timeout 3600',
              "max_session_seconds": 3600, "gpu_hours": 0,
              "limitations": ["two passes per study share cache; order alternates", "bytes are logical reads, not physical disk traffic",
                              "does not change serving defaults or prove full 9-hour candidate budget"]}
    review["comparison_contract"] = "directory-order-contract-v2" if ordering_reference else "original exact receipt v1"
    for name, data in [(metadata["code_file"], notebook), ("kernel-metadata.json", metadata), ("package-review.json", review)]:
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return review


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--parity", required=True)
    p.add_argument("--ordering-reference")
    a = p.parse_args()
    print(json.dumps(build(a.out, a.selection, a.reference, a.parity, a.ordering_reference), indent=2))
