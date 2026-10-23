"""One entry point for burst training on Colab or a Kaggle notebook (or any rented box).

Steps: stage the private datasets into a scratch workspace, install deps, run raptor_train.py
with the given arguments, then package results (summary, history, OOF/gold CSVs, best checkpoint)
into <results>/<TAG>. When a Kaggle token is present the run is uploaded as its own private
dataset <user>/knee-raptor-res-<tag>, so parallel runs never overwrite each other.

Colab:   python run_remote.py --tag EXP070_COLAB_F0 --train "--fold 0 --res 224 ..."
Kaggle:  python run_remote.py --tag ... --train "..." --no-upload   (inputs are detected; results land in /kaggle/working/results)
Local check: python tools/colab/run_remote.py --tag SMOKE --workspace <dir> --results <dir> --tools <dir> --corpus <dir> --train "--smoke --device cpu --res 128 --limit 6" --no-upload
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLS_DS = "knee-raptor-tools"
CORPUS_DS = "knee-raptor-corpus-336"
RESULTS_PREFIX = "knee-raptor-res-"
RESULT_FILES = ("summary.json", "history.json", "train.log", "oof_best.csv", "oof_swa.csv", "gold_best.csv", "gold_swa.csv",
                "test_best.csv", "model_best.pt", "model_swa.pt")


def results_slug(tag: str) -> str:
    """Dataset slug for one run. Kept identical to tools/remote_bundle.py (a test checks it)."""
    slug = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")
    return (RESULTS_PREFIX + slug)[:50].rstrip("-")


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def kaggle_user():
    try:
        out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "username:" in line:
                return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return os.environ.get("KAGGLE_USERNAME", "")


def setup_token():
    """Colab: read KAGGLE_API_TOKEN from the secrets panel; elsewhere rely on the usual files."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("KAGGLE_API_TOKEN")
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            d = Path.home() / ".kaggle"; d.mkdir(exist_ok=True); (d / "access_token").write_text(token)
            return True
    except Exception:
        pass
    return (Path.home() / ".kaggle" / "access_token").exists() or (Path.home() / ".kaggle" / "kaggle.json").exists()


def find_kaggle_input(name):
    for base in (Path("/kaggle/input"), Path("/kaggle/input/datasets")):
        for d in base.glob(f"*/{name}") if base.exists() else []:
            return d
        if (base / name).exists():
            return base / name
    return None


def dataset_files(ref: str):
    r = subprocess.run(["kaggle", "datasets", "files", ref, "--csv", "--page-size", "200"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"cannot list {ref}: {(r.stderr or r.stdout)[-300:]}")
    return [(row["name"], int(row["size"])) for row in csv.DictReader(io.StringIO(r.stdout)) if row.get("name")]


def ensure_dataset(ref: str, dest: Path):
    """Download a dataset file by file, skipping files already present with the right size.

    One whole-dataset zip would need twice the corpus size on disk (32 GB zip + 32 GB files),
    which a Colab T4 runtime does not have; per-file downloads peak at one shard and resume
    after a reconnect.
    """
    dest.mkdir(parents=True, exist_ok=True)
    files = dataset_files(ref)
    total = sum(size for _, size in files)
    t0, got = time.time(), 0
    for name, size in files:
        target = dest / name
        if target.exists() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        sh(["kaggle", "datasets", "download", "-d", ref, "-f", name, "-p", str(target.parent), "--unzip", "-q", "-o"])
        if not target.exists() or target.stat().st_size != size:
            raise SystemExit(f"{name}: expected {size} bytes, got {target.stat().st_size if target.exists() else 'nothing'}")
        got += size
    print(f"{ref}: {len(files)} files, {total / 1e9:.1f} GB in {dest} (downloaded {got / 1e9:.1f} GB in {time.time() - t0:.0f}s)", flush=True)
    return dest


def corpus_complete(d: Path | None) -> bool:
    if d is None or not (d / "manifest.json").exists():
        return False
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return all((d / s["file"]).exists() for s in manifest.get("shards", [])) and (d / "masks.npy").exists()


def stage(workspace: Path, tools_dir: Path | None, corpus_dir: Path | None, user: str):
    workspace.mkdir(parents=True, exist_ok=True)
    tools_dir = tools_dir or find_kaggle_input(TOOLS_DS)
    if not corpus_complete(corpus_dir):
        attached = find_kaggle_input(CORPUS_DS)
        if corpus_complete(attached):
            corpus_dir = attached
        elif user:
            corpus_dir = ensure_dataset(f"{user}/{CORPUS_DS}", corpus_dir or workspace / "_corpus")
        else:
            raise SystemExit("corpus incomplete and no Kaggle token to download it")
    if tools_dir is None:
        tools_dir = ensure_dataset(f"{user}/{TOOLS_DS}", workspace / "_tools")
    # The tools bundle is a workspace skeleton: copy it over the workspace root.
    for p in tools_dir.rglob("*"):
        if p.is_file() and p.name != "dataset-metadata.json":
            dst = workspace / p.relative_to(tools_dir); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
    corpus_target = workspace / "storage" / "cache" / "derived" / "RAPTOR64_336_C140_S02"
    if not corpus_target.exists():
        corpus_target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(corpus_dir, corpus_target, target_is_directory=True)
        except OSError:
            return corpus_dir  # no symlink rights (Windows): the trainer gets the path explicitly; never copy 32 GB
    return corpus_target


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "none"
    except Exception:
        return "none"


def upload(results: Path, user: str, tag: str):
    """Create (or version, on a rerun) the private per-run dataset."""
    if not (results / "summary.json").exists():
        raise SystemExit(f"nothing to upload: {results / 'summary.json'} missing")
    ds = f"{user}/{results_slug(tag)}"
    meta = {"title": results_slug(tag), "id": ds, "licenses": [{"name": "other"}], "isPrivate": True,
            "description": f"Private KneeOps remote run {tag}."}
    (results / "dataset-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    status = subprocess.run(["kaggle", "datasets", "files", ds], capture_output=True, text=True)
    exists = status.returncode == 0 and "name" in status.stdout
    cmd = ["kaggle", "datasets", "version", "-p", str(results), "-m", tag] if exists else \
          ["kaggle", "datasets", "create", "-p", str(results)]
    sh(cmd)
    print("uploaded to", ds, flush=True)
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="STAGE")
    ap.add_argument("--train", default="", help="arguments passed to raptor_train.py (quoted string)")
    ap.add_argument("--workspace", default=None, help="scratch; default /tmp/knee on Kaggle (not saved), /content/knee on Colab")
    ap.add_argument("--results", default=None, help="default /kaggle/working/results on Kaggle (saved output), /content/results on Colab")
    ap.add_argument("--tools", default=None, help="local tools bundle dir (skips download)")
    ap.add_argument("--corpus", default=None, help="local corpus dir (skips download)")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--upload-only", action="store_true", help="upload an existing <results>/<TAG> without training")
    ap.add_argument("--stage-only", action="store_true", help="download/check tools and corpus, then stop (Colab cell 2)")
    a = ap.parse_args()
    on_kaggle = Path("/kaggle/working").exists()
    # Absolute paths: the trainer runs with the workspace as cwd, so relative paths would double up.
    # On Kaggle the workspace lives outside /kaggle/working so the saved output holds only results.
    workspace = Path(a.workspace or ("/tmp/knee" if on_kaggle else "/content/knee")).resolve()
    results = Path(a.results or ("/kaggle/working/results" if on_kaggle else "/content/results")).resolve() / a.tag
    user = kaggle_user() if setup_token() else ""
    if not user and (a.upload_only or not a.no_upload):
        raise SystemExit("Kaggle token missing: add KAGGLE_API_TOKEN to Colab secrets, or pass --no-upload")
    if a.upload_only:
        upload(results, user, a.tag); return
    if not a.train and not a.stage_only:
        raise SystemExit("--train is required unless --upload-only or --stage-only")
    if not user and (a.tools is None or a.corpus is None) and not (find_kaggle_input(TOOLS_DS) and find_kaggle_input(CORPUS_DS)):
        raise SystemExit("Kaggle token missing and datasets not attached: add KAGGLE_API_TOKEN or attach both datasets")
    if not a.no_install and not a.stage_only:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.19", "scikit-learn"], check=False)
    corpus = stage(workspace, Path(a.tools).resolve() if a.tools else None, Path(a.corpus).resolve() if a.corpus else None, user)
    if a.stage_only:
        print("staged:", workspace, "| corpus", corpus.resolve(), flush=True)
        return
    env = {**os.environ, "KNEE_WORKSPACE_ROOT": str(workspace), "PYTHONUTF8": "1"}
    t0 = time.time()
    cmd = [sys.executable, str(workspace / "training" / "src" / "raptor_train.py"), "--tag", a.tag, "--corpus", str(corpus), *a.train.split()]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=workspace, env=env)
    exp = workspace / "artifacts" / "experiments" / a.tag
    results.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES:  # keep partial outputs (log, history) even when training failed
        if (exp / name).exists():
            shutil.copy2(exp / name, results / name)
    (results / "run_remote.json").write_text(json.dumps({
        "tag": a.tag, "train_args": a.train, "seconds": round(time.time() - t0), "exit_code": proc.returncode,
        "platform": "kaggle" if on_kaggle else ("colab" if Path("/content").exists() else "other"), "gpu": gpu_name(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1), encoding="utf-8")
    print("results in", results, sorted(p.name for p in results.iterdir()), flush=True)
    if proc.returncode:
        raise SystemExit(f"training failed with exit code {proc.returncode}; partial results kept in {results}")
    if not a.no_upload:
        upload(results, user, a.tag)


if __name__ == "__main__":
    main()
