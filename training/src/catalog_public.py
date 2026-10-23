"""Inspect downloaded code without execution; acquire small label datasets only."""
from __future__ import annotations
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import re
from acquire import ROOT, read_json, run_cli, pull, stamp

def parse_stdout(text: str):
    start = text.find("[\n")
    if start < 0:
        start = text.find("[{ ")
    if start < 0:
        start = text.find("[]")
    if start < 0:
        return []
    return json.JSONDecoder().raw_decode(text[start:])[0]

def stdout_records(path: Path):
    return parse_stdout(read_json(path).get("stdout", ""))

def get_dataset(ref: str, download: bool = False) -> dict:
    name = ref.replace("/", "__")
    dest = ROOT/("labels" if download else "external")/name
    dest.mkdir(exist_ok=True)
    meta = run_cli(["datasets","metadata",ref,"-p",str(dest)], "dataset_meta_"+name)
    files = run_cli(["datasets","files",ref,"--page-size","200","--format","json"], "dataset_files_"+name)
    status = "metadata_only"
    if download:
        done = dest/".download_complete.json"
        if done.exists():
            status = "already_present"
        else:
            res = run_cli(["datasets","download","-d",ref,"-p",str(dest),"--unzip","-q"], "dataset_download_"+name)
            status = "downloaded" if res["returncode"]==0 else "failed"
            if status=="downloaded":
                done.write_text(json.dumps({"source":ref,"utc":stamp()}),encoding="utf-8")
    return {"ref":ref,"path":str(dest),"status":status,"metadata_rc":meta["returncode"],"files_rc":files["returncode"]}

def notebook_catalog() -> set[str]:
    rows, sources, review = [], set(), []
    for folder in sorted((ROOT/"kaggle_notebooks").iterdir()):
        metapath = folder/"kernel-metadata.json"
        if not metapath.exists():
            continue
        meta=read_json(metapath)
        nbpath=folder/meta["code_file"]
        if not nbpath.exists():
            continue
        raw=nbpath.read_bytes()
        source_hash=hashlib.sha256(raw).hexdigest()
        texts, markdown, output = [], [], []
        if nbpath.suffix==".ipynb":
            nb=read_json(nbpath)
            for idx,cell in enumerate(nb.get("cells",[])):
                source="".join(cell.get("source",[]))
                if cell.get("cell_type")=="code":
                    texts.append(f"# --- CELL {idx} ---\n"+source)
                    for o in cell.get("outputs",[]):
                        value=o.get("text",o.get("data",{}).get("text/plain",[]))
                        output.append("".join(value) if isinstance(value,list) else str(value))
                else:
                    markdown.append(source)
        else:
            texts=[raw.decode("utf-8-sig")]
        code="\n\n".join(texts)
        (folder/"source_review.py").write_text(code,encoding="utf-8")
        (folder/"description_review.md").write_text("\n\n".join(markdown),encoding="utf-8")
        (folder/"outputs_review.txt").write_text("\n\n".join(output),encoding="utf-8")
        dependencies=meta.get("dataset_sources",[])
        sources.update(dependencies)
        flagged=[]
        for pattern,label in [(r"requests\.|urllib\.|https?://","network/reference"), (r"subprocess|os\.system|!pip|%pip","shell/package"), (r"pickle\.load|torch\.load","deserialize"), (r"eval\(|exec\(","dynamic_execution"), (r"shutil\.rmtree|remove\(","file_deletion")]:
            if re.search(pattern,code): flagged.append(label)
        literals=[]
        for ln,line in enumerate(code.splitlines(),1):
            if re.search(r"(?i)(crop_mm|crop_size|img_size|image_size|n_windows|num_windows|window_count|span|n_slices|num_slices|slice_count|backbone|ensemble|blend|coatnet|dinov|radimage|\.pt['\"]|\.pth['\"])",line):
                if len(line)<600:
                    literals.append(f"{ln}: {line}")
        row={"notebook_id":meta["id"],"source":"https://www.kaggle.com/code/"+meta["id"],"version":"unresolved", "path":str(nbpath.relative_to(ROOT)),"sha256":source_hash,"author":meta["id"].split("/")[0],"last_update":"", "copied_from":"unresolved", "datasets":";".join(dependencies),"models":";".join(meta.get("model_sources",[])),"notebook_inputs":";".join(meta.get("kernel_sources",[])),"license":"unresolved","status":"source_acquired_not_executed"}
        rows.append(row)
        review.append({"id":meta["id"],"lines":len(code.splitlines()),"source_sha256":source_hash,"code_sha256":hashlib.sha256(code.encode()).hexdigest(),"datasets":dependencies,"flags":flagged,"config_evidence":literals[:150]})
    with (ROOT/"experiments"/"notebook_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/"metadata"/"notebook_static_audit.json").write_text(json.dumps(review,indent=2),encoding="utf-8")
    print(f"NOTEBOOKS_INDEXED={len(rows)} UNIQUE_DATASET_REFS={len(sources)}",flush=True)
    return sources

def main():
    votes=stdout_records(ROOT/"logs"/"kernels_votes.json")
    extra=[r["ref"] for r in votes if any(s in (r.get("title","")+r["ref"]).lower() for s in ["raptor","coatnet","radimage","report", "multi-plane", "preprocess"])][:15]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(pull,extra))
    refs=notebook_catalog()
    found={}
    for path in (ROOT/"logs").glob("datasets_*.json"):
        for row in stdout_records(path):
            found[row["ref"]]=row
    labels=[ref for ref,row in found.items() if "rsna" in ref and "label" in ref and int(row.get("size",10**12)) < 25_000_000]
    (ROOT/"metadata"/"dataset_search_catalog.json").write_text(json.dumps(list(found.values()),indent=2),encoding="utf-8")
    important=[r for r in refs if any(w in r for w in ["raptor-knee","knee-mri-fold-weights","radimagenet","rsna-knee-coat","rsna-knee-weights"]) ]
    jobs=[(ref,True) for ref in sorted(labels)]+[(ref,False) for ref in sorted(set(important)-set(labels))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(lambda args:get_dataset(*args),jobs))
    (ROOT/"metadata"/"dataset_acquisition.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"LABEL_DATASETS_REQUESTED={len(labels)}",flush=True)
    for x in result: print(x["ref"],x["status"],flush=True)

if __name__=="__main__": main()
