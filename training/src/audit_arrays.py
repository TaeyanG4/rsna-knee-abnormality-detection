"""Hash acquired weights and inspect numeric public artifacts without pickle loading."""
from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile
import numpy as np
import pandas as pd

ROOT=workspace_root()

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()

def main():
    findings={"weights":[],"arrays":[],"corpus":{}}
    for path in sorted((ROOT/"weights").rglob("*.pt")):
        rec={"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":sha256(path),"loaded":False}
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                entries=[x for x in z.infolist() if x.filename.endswith("/data.pkl") or x.filename=="data.pkl"]
                if len(entries)==1 and entries[0].file_size<10_000_000:
                    # Opcode disassembly reads bytes; no objects are instantiated.
                    opcodes=list(pickletools.genops(z.read(entries[0])))
                    text=[]
                    for op,arg,pos in opcodes[-180:]:
                        val=repr(arg)
                        text.append(f"{pos} {op.name} {val[:400]}")
                    out=ROOT/"metadata"/(path.stem+"_pickle_tail.txt")
                    out.write_text("\n".join(text),encoding="utf-8")
                    rec["static_metadata_disassembly"]=str(out.relative_to(ROOT))
                    rec["zip_entries"]=len(z.infolist())
        findings["weights"].append(rec)
        print("WEIGHT",json.dumps(rec))
    corpus=ROOT/"processed"/"dreaddevelopment__knee-raptor-corpus"
    if (corpus/"all_ids.npy").exists():
        ids=np.load(corpus/"all_ids.npy",allow_pickle=False)
        masks=np.load(corpus/"all_masks.npy",allow_pickle=False)
        gold=pd.read_csv(ROOT/"gold"/"gold_labels.csv",dtype={"StudyInstanceUID":str})
        findings["corpus"]={"rows":len(ids),"ids_dtype":str(ids.dtype),"mask_shape":list(masks.shape),"gold_overlap":len(set(ids.astype(str))&set(gold.StudyInstanceUID)),
                            "pixels_acquired":(corpus/"all_vols.npy").exists(),"matches_maxspan_64_slice_contract":bool(masks.shape[1]==64),
                            "ids_sha256":sha256(corpus/"all_ids.npy"),"mask_sha256":sha256(corpus/"all_masks.npy")}
        print("CORPUS",json.dumps(findings["corpus"]))
    for path in sorted((ROOT/"external").rglob("*.npz")):
        rec={"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"arrays":{}}
        with np.load(path,allow_pickle=False) as arrays:
            for name in arrays.files:
                try:
                    a=arrays[name]
                    rec["arrays"][name]={"shape":list(a.shape),"dtype":str(a.dtype)}
                    if np.issubdtype(a.dtype,np.number) and a.size:
                        rec["arrays"][name].update({"finite":bool(np.isfinite(a).all()),"min":float(np.nanmin(a)),"max":float(np.nanmax(a))})
                except ValueError as exc: rec["arrays"][name]={"unreadable_without_pickle":str(exc)}
        findings["arrays"].append(rec)
        print("ARRAY",json.dumps(rec))
    for p in (ROOT/"kaggle_notebooks").glob("mattiaangeli*/*.ipynb"):
        nb=json.loads(p.read_bytes())
        findings["candidate_notebook_metadata"]=nb.get("metadata",{})
    (ROOT/"metadata"/"binary_artifact_audit.json").write_text(json.dumps(findings,indent=2),encoding="utf-8")

if __name__=="__main__":main()
