"""Local, non-executing inspection of CSV schemas and notebook source provenance."""
from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import ast
import hashlib
import itertools
import json
from pathlib import Path
import re
import pandas as pd

ROOT=workspace_root()

def main():
    gold=pd.read_csv(ROOT/"gold"/"gold_labels.csv",dtype={"StudyInstanceUID":str})
    train=pd.read_csv(ROOT/"data"/"train.csv",dtype={"StudyInstanceUID":str})
    schemas=[]
    for p in sorted((ROOT/"labels").glob("*/*.csv")):
        df=pd.read_csv(p,dtype={"StudyInstanceUID":str})
        missing=sorted(set(gold.StudyInstanceUID)-set(df.get("StudyInstanceUID",[])))
        meta=p.parent/"dataset-metadata.json"
        info=json.loads(meta.read_text(encoding="utf-8-sig")).get("info",{}) if meta.exists() else {}
        row={"path":str(p.relative_to(ROOT)),"rows":len(df),"columns":list(df.columns),"gold_missing_count":len(missing),"gold_missing_ids":missing,
             "licenses":info.get("licenses",[]),"dataset_id":info.get("datasetId"),"description":info.get("description","")}
        schemas.append(row)
        print(p.parent.name,p.name,"ROWS",len(df),"GOLD_MISSING",len(missing),"COLS",list(df.columns))
    (ROOT/"metadata"/"label_schemas.json").write_text(json.dumps(schemas,indent=2),encoding="utf-8")
    series=pd.read_csv(ROOT/"data"/"train_series.csv")
    slot_count=series.groupby(["StudyInstanceUID","Anatomical_Plane","Fat_Suppression"]).size().unstack([1,2],fill_value=0)
    report={"fluid_equals_fatsuppression_count":int((series.Fluid_Sensitive==series.Fat_Suppression).sum()),"series_count":len(series),
            "studies_missing_any_six_slots":int((slot_count==0).any(axis=1).sum()),
            "gold_studies_missing_any_six_slots":int((slot_count.reindex(gold.StudyInstanceUID)==0).any(axis=1).sum()),
            "slot_missing_studies":{str(c):int((slot_count[c]==0).sum()) for c in slot_count},
            "report_missing":int(train.Report.isna().sum()),"test_rows":len(pd.read_csv(ROOT/"data"/"test.csv"))}
    (ROOT/"metadata"/"data_integrity_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("DATA_INTEGRITY",json.dumps(report))
    notebooks=[]
    for folder in sorted((ROOT/"kaggle_notebooks").iterdir()):
        mp=folder/"kernel-metadata.json"
        if not mp.exists(): continue
        meta=json.loads(mp.read_text(encoding="utf-8-sig"))
        path=folder/meta["code_file"]
        if not path.exists() or path.suffix!=".ipynb":continue
        nb=json.loads(path.read_bytes())
        sources=[];embedded=[];parse_errors=[];classes=set();functions=set();cellhashes=[];declared_sha=set()
        for i,cell in enumerate(nb.get("cells",[])):
            if cell.get("cell_type")!="code": continue
            text="".join(cell.get("source",[]));sources.append(text)
            cellhashes.append(hashlib.sha256(text.encode()).hexdigest())
            declared_sha.update(re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])",text))
            try: tree=ast.parse(text)
            except SyntaxError as exc:
                parse_errors.append({"cell":i,"line":exc.lineno,"message":str(exc)})
                continue
            for node in ast.walk(tree):
                if isinstance(node,ast.ClassDef): classes.add(node.name)
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):functions.add(node.name)
                if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                    name=node.targets[0].id if isinstance(node.targets[0],ast.Name) else "unnamed"
                    value=node.value.value
                    if len(value)>1000 and (name.endswith("SRC") or name.endswith("CODE")):
                        outfile=folder/f"embedded_{i}_{name}.py"
                        outfile.write_text(value,encoding="utf-8")
                        embedded.append({"variable":name,"cell":i,"path":str(outfile.relative_to(ROOT)),"sha256":hashlib.sha256(value.encode()).hexdigest(),"lines":len(value.splitlines())})
                        sources.append(value)
        text="\n".join(sources)
        useful={line.strip() for line in text.splitlines() if len(line.strip())>25 and not line.strip().startswith("#")}
        rec={"id":meta["id"],"title":meta["title"],"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
             "cell_sha256":cellhashes,"declared_sha256_not_verified_weights":sorted(declared_sha),"classes":sorted(classes),"functions":sorted(functions),
             "embedded_sources":embedded,"ast_parse_warnings_not_runtime_failures":parse_errors,
             "source_lines_including_embedded":len(text.splitlines()),"datasets":meta.get("dataset_sources",[]),"models":meta.get("model_sources",[]),
             "notebook_inputs":meta.get("kernel_sources",[]),"metadata_keys":list(nb.get("metadata",{})),
             "line_fingerprints":useful}
        notebooks.append(rec)
    edges=[]
    for a,b in itertools.combinations(notebooks,2):
        x,y=a["line_fingerprints"],b["line_fingerprints"]
        intersection=len(x&y)
        union=len(x|y)
        edges.append({"a":a["id"],"b":b["id"],"line_jaccard":intersection/max(1,union),
                      "smaller_source_coverage":intersection/max(1,min(len(x),len(y))),
                      "identical_cells":len(set(a["cell_sha256"])&set(b["cell_sha256"])),
                      "common_datasets":";".join(sorted(set(a["datasets"])&set(b["datasets"])))})
    for r in notebooks:r.pop("line_fingerprints")
    (ROOT/"metadata"/"notebook_provenance_audit.json").write_text(json.dumps(notebooks,indent=2),encoding="utf-8")
    pd.DataFrame(edges).sort_values("line_jaccard",ascending=False).to_csv(ROOT/"metadata"/"notebook_similarity.csv",index=False)
    print("NOTEBOOKS",len(notebooks),"SIMILARITY_PAIRS",len(edges),"EMBEDDED_SOURCES",sum(len(r["embedded_sources"]) for r in notebooks))

if __name__=="__main__":main()
