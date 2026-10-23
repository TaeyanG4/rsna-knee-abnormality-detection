"""Local-only inspection: no network, subprocess, or execution of public code."""
from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import csv
import hashlib
import json
from pathlib import Path
import re
import pandas as pd

ROOT=workspace_root()
TARGETS=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

def main():
    train=pd.read_csv(ROOT/"data"/"train.csv",dtype={"StudyInstanceUID":str})
    present=train[TARGETS].notna().sum(axis=1)
    if not present.isin([0,12]).all():
        raise ValueError("Partial-label studies need a separate evaluation policy")
    gold=train.loc[present==12,["StudyInstanceUID"]+TARGETS].copy()
    if not gold[TARGETS].isin([0,1]).all().all():
        raise ValueError("Gold targets are not binary")
    if train.StudyInstanceUID.duplicated().any():
        raise ValueError("Duplicate train UID")
    gold.to_csv(ROOT/"gold"/"gold_labels.csv",index=False)
    stats=pd.DataFrame({"target":TARGETS,"n":len(gold),"positive_count":gold[TARGETS].sum().values})
    stats["prevalence"]=stats.positive_count/stats.n
    stats.to_csv(ROOT/"gold"/"gold_target_counts.csv",index=False)
    gold[TARGETS].corr().to_csv(ROOT/"gold"/"gold_label_correlation.csv")
    print(f"TRAIN={len(train)} GOLD={len(gold)} WEAK_ONLY={int((present==0).sum())}")
    print(stats.to_string(index=False))
    if (ROOT/"data"/"train_series.csv").exists():
        series=pd.read_csv(ROOT/"data"/"train_series.csv")
        summary={"rows":len(series),"columns":list(series.columns),"cardinality":{c:int(series[c].nunique()) for c in series}}
        summary["small_categorical_counts"]={c:series[c].fillna("MISSING").value_counts().astype(int).to_dict() for c in series if series[c].nunique()<30}
        (ROOT/"metadata"/"series_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print("SERIES",len(series),list(series.columns))
    rows=[]
    for folder in sorted((ROOT/"kaggle_notebooks").iterdir()):
        p=folder/"kernel-metadata.json"
        if not p.exists(): continue
        meta=json.loads(p.read_text(encoding="utf-8-sig"))
        nbpath=folder/meta["code_file"]
        if not nbpath.exists(): continue
        raw=nbpath.read_bytes()
        code,md,outputs=[],[],[]
        if nbpath.suffix==".ipynb":
            nb=json.loads(raw)
            for i,cell in enumerate(nb.get("cells",[])):
                text="".join(cell.get("source",[]))
                if cell.get("cell_type")=="code":
                    code.append(f"# CELL {i}\n"+text)
                else: md.append(text)
                for o in cell.get("outputs",[]):
                    val=o.get("text",o.get("data",{}).get("text/plain",[]))
                    outputs.append("".join(val) if isinstance(val,list) else str(val))
        else: code=[raw.decode("utf-8-sig")]
        text="\n\n".join(code)
        (folder/"source_review.py").write_text(text,encoding="utf-8")
        (folder/"description_review.md").write_text("\n\n".join(md),encoding="utf-8")
        (folder/"outputs_review.txt").write_text("\n\n".join(outputs),encoding="utf-8")
        evidence=[]
        for n,line in enumerate(text.splitlines(),1):
            if len(line)<500 and re.search(r"(?i)(crop_mm|img_size|n_windows|num_windows|span|n_slices|num_slices|backbone|ensemble|blend|coatnet|dinov|radimage|\.pth|\.pt\b|ImagePositionPatient|ImageOrientationPatient|InstanceNumber)",line):
                evidence.append(f"{n}: {line}")
        (folder/"config_evidence.txt").write_text("\n".join(evidence),encoding="utf-8")
        rows.append({"notebook_id":meta["id"],"source":"https://www.kaggle.com/code/"+meta["id"],"version":"unresolved","path":str(nbpath.relative_to(ROOT)),"sha256":hashlib.sha256(raw).hexdigest(),"author":meta["id"].split("/")[0],"last_update":"","copied_from":"unresolved","datasets":";".join(meta.get("dataset_sources",[])),"models":";".join(meta.get("model_sources",[])),"notebook_inputs":";".join(meta.get("kernel_sources",[])),"license":"unresolved","status":"source_inspected_not_executed"})
    with (ROOT/"experiments"/"notebook_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f"LOCAL_NOTEBOOKS_EXTRACTED={len(rows)}")

if __name__=="__main__": main()
