"""Bounded, auditable acquisition of public code and competition metadata.

Does not execute downloaded notebooks, create Git, or download the DICOM corpus.
The installed Kaggle CLI handles account access; credentials are never inspected.
"""
from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path

import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = workspace_root()
COMP = "rsna-knee-abnormality-detection"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FIELDS = {
    "experiments.csv": "experiment_id,datetime,base_model,label_source,label_version,fold,backbone,pretraining,resolution,crop_mm,planes,sequences,slice_count,window_count,sampling,loss,augmentation,tta,weak_oof_macro,gold_macro_auc,gold_bootstrap_mean,gold_bootstrap_std,public_lb,runtime_sec,peak_vram,prediction_path,snapshot,notes,keep,status".split(","),
    "label_experiments.csv": "label_exp,teacher,prompt_version,language_method,negation,severity_logic,unknown_handling,ACL_auc,MCL_auc,MM_auc,LM_auc,MedOA_auc,LatOA_auc,PFOA_auc,Effusion_auc,Synovitis_auc,Baker_auc,Contusion_auc,Fracture_auc,macro_auc,notes".split(","),
    "model_manifest.csv": "model_id,source,version,path,sha256,architecture,training_labels,fold,preprocessing,inference_geometry,license,status".split(","),
    "notebook_manifest.csv": "notebook_id,source,version,path,sha256,author,last_update,copied_from,datasets,models,notebook_inputs,license,status".split(","),
    "artifact_manifest.csv": "path,bytes,sha256,source,acquired_utc,license,status".split(","),
    "submission_log.csv": "submission_id,experiment_id,datetime,kernel_ref,kernel_version,path,sha256,public_lb,rank,runtime_sec,status,notes".split(","),
}

def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: Path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"))

def run_cli(args: list[str], logname: str, timeout: int = 180) -> dict:
    started = time.monotonic()
    try:
        p = subprocess.run(["kaggle", *args], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
        out = {"args": args, "utc": stamp(), "returncode": p.returncode,
               "elapsed_sec": time.monotonic()-started, "stdout": p.stdout, "stderr": p.stderr}
    except (subprocess.TimeoutExpired, OSError) as exc:
        out = {"args": args, "utc": stamp(), "returncode": -1, "elapsed_sec": time.monotonic()-started,
               "stdout": "", "stderr": str(exc)}
    (ROOT/"logs"/f"{logname}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{logname}: rc={out['returncode']} {out['elapsed_sec']:.1f}s", flush=True)
    return out

def initialize() -> None:
    for dirname in "data metadata dicom_cache pixel_cache processed labels external kaggle_notebooks weights src experiments predictions oof gold test submissions reports logs snapshots kaggle_submission".split():
        (ROOT/dirname).mkdir(exist_ok=True)
    for name, fields in FIELDS.items():
        dest = ROOT/"experiments"/name
        if not dest.exists():
            with dest.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(fields)
    packages = {}
    for name in ["kaggle", "kagglesdk", "torch", "numpy", "pandas", "scikit-learn", "pydicom", "requests", "timm", "torchvision"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    env = {"observed_utc":stamp(), "workspace":str(ROOT), "python":sys.version,
           "git_exists":(ROOT/".git").exists(), "disk_free_bytes":shutil.disk_usage(ROOT).free,
           "packages":packages, "policy":"Do not initialize Git; no full DICOM download; no reports at inference."}
    (ROOT/"metadata"/"environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

def pull(ref: str) -> dict:
    dest = ROOT/"kaggle_notebooks"/ref.replace("/", "__")
    dest.mkdir(exist_ok=True)
    if list(dest.glob("*.ipynb")) or list(dest.glob("*.py")):
        return {"ref":ref, "status":"already_present"}
    result = run_cli(["kernels", "pull", ref, "-p", str(dest), "-m"], "pull_"+ref.replace("/", "__"))
    return {"ref":ref, "status":"downloaded" if result["returncode"] == 0 else "failed"}

def main() -> None:
    initialize()
    jobs = [
        (["competitions","leaderboard",COMP,"--show","--page-size","30","--format","json"], "leaderboard_top30"),
        (["competitions","submissions",COMP,"--page-size","5","--format","json"], "account_submissions"),
        (["kernels","list","--competition",COMP,"--sort-by","voteCount","--page-size","40","--format","json"], "kernels_votes"),
    ]
    for q in ["rsna knee labels", "knee llm", "knee max span", "knee raptor", "knee radimagenet", "knee soft labels"]:
        jobs.append((["datasets","list","-s",q,"--page-size","40","--format","json"], "datasets_"+q.replace(" ","_")))
    for name in ["train_series.csv", "test.csv", "test_series.csv", "sample_submission.csv"]:
        if not (ROOT/"data"/name).exists():
            jobs.append((["competitions","download",COMP,"-f",name,"-p",str(ROOT/"data"),"-q"], "download_"+name))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda job: run_cli(*job), jobs))
    catalog = read_json(ROOT/"metadata"/"kernels_score_desc.json")
    refs = [r["ref"] for r in catalog[:20]]
    refs += ["hdhsjdjd/rsna-knee-raptor-v5-w035", "nishantkharga/rsna-knee-full-4-arm-ensemble-v55", "nathanjacob/4-arm-ensemble-explained-rsna-knee-0-937"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(pull, list(dict.fromkeys(refs))))
    (ROOT/"metadata"/"acquisition_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps({"notebooks_requested":len(results),"downloaded":sum(x["status"]=="downloaded" for x in results)},indent=2), flush=True)

if __name__ == "__main__":
    main()
