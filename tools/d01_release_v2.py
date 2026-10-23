"""Separate D01v2 release control. Old assets and v1 FAIL are immutable.

Build is fail-closed on pinned numerical, codec, T4 and measured runtime receipts.
No remote write is performed. The only supported header policy is the old reader.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ASSETS_SHA="e62893ebb1e22dbb1288ea6c805097bbc96ef5e43759f42a22d7cfb3bbef036b"
GDCM_SHA="512d0d903e12c5ad389939a257817bf182ce288431b3d474d62cb0c0833b8b56"
HEADER_POLICY="legacy-full-read"
OUTPUT=Path("/kaggle/working/submission.csv")
GATE_NAMES=("protocol","expected_inputs","numeric_local","numeric_remote","codec","t4","r2_runtime","anchor_runtime","runtime_gate")


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def need(condition,message):
    if not condition:raise ValueError(message)


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def bound_json(binding,base):
    path=Path(binding["path"])
    if not path.is_absolute():path=Path(base)/path
    need(path.is_file() and sha(path)==binding["sha256"],f"receipt missing/changed: {path.name}")
    return read(path)


def validate_numeric(report,protocol,protocol_sha,expected,mode):
    need(report.get("status")=="PASS" and report.get("mode")==mode,"numerical receipt status/mode")
    need(report.get("protocol_sha256")==protocol_sha and report.get("helper_sha256")==protocol["helper_sha256"],"numerical source binding")
    cases=report.get("cases",[]);synthetic=report.get("synthetic",[])
    need([r.get("uid") for r in cases]==expected["ids"] and len(cases)==100,"numerical100 UID coverage")
    need([r.get("uid") for r in synthetic]==[r["uid"] for r in expected["synthetic"]] and len(synthetic)==7,"synthetic7 coverage")
    for r in cases+synthetic:
        need(all(r.get(k) is True for k in ("pass","exact_input","exact_coefficients","finite_channel_ranges")),"numerical exact/finite/range failure")
        for key,limit in (("windows",5e-7),("normalized",3e-6)):
            c=r.get(key,{})
            need(c.get("pass") is True and c.get("exceedances")==0 and c.get("atol")==limit and c.get("rtol")==0,"numerical threshold changed")
            need(math.isfinite(c.get("max_abs",float("nan"))) and 0<=c["max_abs"]<=limit,"numerical recorded error exceeds bound")


def validate_gates(bindings,base,anchor_sha,helper_sha):
    need(set(bindings)==set(GATE_NAMES),"all required gates must be explicit")
    g={name:bound_json(bindings[name],base) for name in GATE_NAMES}
    p=g["protocol"];psha=bindings["protocol"]["sha256"]
    need(p.get("schema")=="d01-numerical-input-gate-v2" and p.get("v1_result")=="FAILED_UNCHANGED","v1 must remain FAILED; distinct v2 schema required")
    need((p.get("window_atol"),p.get("normalized_atol"),p.get("rtol"))==(5e-7,3e-6,0.0),"frozen numerical tolerances changed")
    need(p.get("assets_manifest_sha256")==ASSETS_SHA and p.get("gdcm_wheel_sha256")==GDCM_SHA,"protocol dependency binding")
    need(p.get("expected_sha256")==bindings["expected_inputs"]["sha256"],"expected integer-input manifest binding")
    validate_numeric(g["numeric_local"],p,psha,g["expected_inputs"],"local_cache")
    validate_numeric(g["numeric_remote"],p,psha,g["expected_inputs"],"remote_dicom")
    codec=g["codec"]
    need(codec.get("status")=="PASS" and codec.get("gdcm_version")=="3.2.6" and codec.get("pydicom_version")=="3.0.2","codec version/status")
    syntaxes={r.get("uid") for r in codec.get("decoders",[]) if r.get("available") is True}
    need(syntaxes=={"1.2.840.10008.1.2","1.2.840.10008.1.2.1",*[f"1.2.840.10008.1.2.4.{n}" for n in (57,70,90,91)]},"six codec capabilities required")
    fixtures=codec.get("fixtures",[])
    need(len(fixtures)>=3 and all(r.get("decode")=="SUCCESS" for r in fixtures) and any(r.get("matches_uncompressed_fixture") is True for r in fixtures),"codec real fixture evidence missing")
    t=g["t4"]
    need(t.get("schema")=="d01-t4-release-gate-v2" and t.get("status")=="PASS","fresh v2 T4 gate required; old v1 cannot substitute")
    for key,value in {"assets_manifest_sha256":ASSETS_SHA,"input_protocol_sha256":psha,"codec_wheel_sha256":GDCM_SHA,"header_policy":HEADER_POLICY,"release_helper_sha256":helper_sha}.items():
        need(t.get(key)==value,f"T4 binding differs: {key}")
    need(t.get("member_count")==5 and t.get("studies")==3 and t.get("all_finite") is True and t.get("fp32_fallbacks")==0 and t.get("neutral_fills")==0,"T4 coverage/fallback contract")
    need(t.get("cpu_threads")==2,"T4 CPU preparation thread contract")
    need(t.get("tolerances")=={"f0_max_abs":.002,"f0_mean_abs":.0005},"probability tolerance changed")
    need(0<=t.get("f0_max_abs",1)<=.002 and 0<=t.get("f0_mean_abs",1)<=.0005 and t.get("input_reference_pass") is True,"T4 numerical/probability failure")
    need("T4" in t.get("environment",{}).get("gpu",""),"actual T4 required")
    r=g["r2_runtime"];a=g["anchor_runtime"];plan=g["runtime_gate"]
    need(r.get("status")=="COMPLETE" and r.get("assets_manifest_sha256")==ASSETS_SHA and r.get("completed_studies",0)>=100,"R2 measured100 runtime missing")
    inference=r.get("inference",{})
    need(inference.get("member_count")==5 and inference.get("fp32_fallbacks")==0 and inference.get("neutral_fills")==0,"R2 measured inference contract")
    need(inference.get("residency")=="resident" and inference.get("effective_amp")=="fp16" and inference.get("batch_size")==1 and inference.get("runtime",{}).get("policy")=="strict","R2 measured runtime policy mismatch")
    need(a.get("status")=="COMPLETE" and a.get("benchmark_studies",0)>=49 and a.get("coat_readers_parallel_pairs") is False and a.get("coat_family_members")==4,"anchor49+ measured sequential path missing")
    need(a.get("anchor_notebook_sha256")==anchor_sha,"anchor runtime source mismatch")
    need(plan.get("schema")=="d01-runtime-budget-v2" and plan.get("status")=="PASS","external runtime gate remains required")
    for key,name in (("r2_receipt_sha256","r2_runtime"),("anchor_receipt_sha256","anchor_runtime")):
        need(plan.get(key)==bindings[name]["sha256"],"runtime budget receipt binding")
    need(plan.get("header_policy")==HEADER_POLICY,"unmeasured header policy not supported")
    n=plan.get("max_studies",0);need(isinstance(n,int) and n>=1300,"explicit runtime study ceiling>=1300 required")
    startup=r["timings"]["worker_total_seconds"]-r["timings"]["inference_including_model_placement_offload"]
    projected_r2=max(0,startup)+r["study_time_distribution"]["mean"]*n
    # Conservative anchor scaling retains startup on every scaled cohort; no unmeasured deduction.
    projected_anchor=a["total_cell_seconds"]*n/a["benchmark_studies"]
    rb,ab,safety=(plan.get(k,0) for k in ("r2_seconds","anchor_seconds","safety_seconds"))
    need(all(isinstance(x,(int,float)) and math.isfinite(x) and x>0 for x in (rb,ab,safety)),"finite explicit budgets required")
    need(plan.get("whole_seconds")==32400 and safety>=300 and rb+ab+safety<=32400,"shared9h budget with safety reserve violated")
    need(rb>=projected_r2 and ab>=projected_anchor,"measured projections do not fit proposed budgets")
    return g,{"r2_seconds":projected_r2,"anchor_seconds":projected_anchor,"scope":"engineering extrapolation, not a hidden-runtime guarantee"}


def stop_tree(process):
    """Terminate only the launched worker tree, including detached notebook kernels."""
    import psutil
    try:children=psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:children=[]
    if os.name=="posix":
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    else:
        try:process.terminate()
        except ProcessLookupError:pass
    for child in reversed(children):
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(children,timeout=2)
    for child in alive:
        try:child.kill()
        except psutil.NoSuchProcess:pass
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name=="posix":
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        process.kill();process.wait()


def run_child(command,deadline,phase_seconds,logfile,env=None):
    timeout=min(phase_seconds,deadline-time.monotonic())
    if timeout<=0:raise TimeoutError("shared deadline exhausted before launch")
    kwargs={"start_new_session":True} if os.name=="posix" else {"creationflags":subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW}
    with Path(logfile).open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,**kwargs)
        try:
            code=process.wait(timeout=timeout)
            if code:raise RuntimeError(f"worker exit{code}; inspect {logfile}")
        except BaseException:
            stop_tree(process);raise


def anchor_worker_notebook(anchor):
    """No source-cell mutation; append only the audit/output transfer."""
    notebook=copy.deepcopy(anchor)
    suffix="""import json as _d01_json, os as _d01_os
from pathlib import Path as _d01_Path
assert _RSNA_AUDIT.get('status') == 'COMPLETE'
assert _RSNA_AUDIT.get('coat_family_members') == 4
assert _RSNA_AUDIT.get('dino_members') == 20 and _RSNA_AUDIT.get('a5_folds') == 5 and _RSNA_AUDIT.get('raptor_views') == 4
_d01_target = _d01_Path(_d01_os.environ['D01_ANCHOR_OUTPUT'])
assert not _d01_target.exists()
_d01_os.replace('/kaggle/working/submission.csv', _d01_target)
_d01_target.with_suffix('.json').write_text(_d01_json.dumps(_RSNA_AUDIT, indent=2, default=str))
"""
    notebook["cells"].append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":suffix.splitlines(keepends=True)})
    notebook.setdefault("metadata",{})["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
    need(notebook["cells"][:-1]==anchor["cells"],"anchor cells mutated")
    return notebook


def build(spec_path,out):
    spec_path,out=Path(spec_path),Path(out)
    need(not out.exists(),"new immutable output directory required")
    spec=read(spec_path);base=spec_path.parent
    def path(name):
        p=Path(spec[name]);return p if p.is_absolute() else base/p
    assets,anchor_path=path("assets"),path("anchor")
    need(sha(assets/"d01-assets-manifest.json")==ASSETS_SHA,"old asset identity")
    anchor=read(anchor_path);need(len(anchor["cells"])==30,"expected original30anchor cells")
    gates,projection=validate_gates(spec["gates"],base,sha(anchor_path),sha(__file__))
    need(sha(path("input_helper"))==gates["protocol"]["helper_sha256"],"input gate helper source mismatch")
    out.mkdir(parents=True);data=out/"assets";data.mkdir()
    shutil.copy2(__file__,data/"d01_release_v2.py")
    shutil.copy2(path("input_helper"),data/"d01_input_gate.py")
    shutil.copy2(anchor_path,data/"anchor-original.ipynb")
    copied={}
    for name,binding in spec["gates"].items():
        source=Path(binding["path"]);source=source if source.is_absolute() else base/source
        dest=data/f"{name}.json";shutil.copy2(source,dest)
        copied[name]={"path":dest.name,"sha256":sha(dest)}
    manifest={"schema":"d01-release-v2","v1_result":"FAILED_UNCHANGED","assets_manifest_sha256":ASSETS_SHA,
              "anchor_sha256":sha(anchor_path),"release_helper_sha256":sha(__file__),"input_helper_sha256":sha(path("input_helper")),
              "header_policy":HEADER_POLICY,"codec_wheel_sha256":GDCM_SHA,"gates":copied,"projection":projection,
              "blend":{"anchor_rank":.9,"r2_rank_probability_mean":.1,"final_rerank":False}}
    dump(data/"release-manifest.json",manifest)
    dump(data/"dataset-metadata.json",{"title":"knee-d01-release-v2-assets","id":"taeyangg4/knee-d01-release-v2-assets","isPrivate":True,"licenses":[{"name":"other"}]})
    metadata=read(path("anchor_metadata"));metadata.update(id="taeyangg4/knee-d01-release-v2",title="knee-d01-release-v2",code_file="release-v2.ipynb",is_private=True,enable_internet=False)
    metadata["dataset_sources"]+= ["taeyangg4/knee-d01-r2-assets","taeyangg4/knee-d01-codec-assets","taeyangg4/knee-d01-release-v2-assets"]
    metadata["dataset_sources"]=list(dict.fromkeys(metadata["dataset_sources"]))
    marker_sha=sha(data/"release-manifest.json")
    code=f'''import time, pathlib, hashlib, subprocess, sys
started=time.monotonic()
def find(name,digest):
    paths=list(pathlib.Path('/kaggle/input').glob('*/'+name))+list(pathlib.Path('/kaggle/input').glob('datasets/*/*/'+name))
    paths=sorted(set(p.resolve() for p in paths if hashlib.sha256(p.read_bytes()).hexdigest()==digest))
    assert len(paths)==1
    return paths[0].parent
release=find('release-manifest.json',{marker_sha!r})
assets=find('d01-assets-manifest.json',{ASSETS_SHA!r})
codec=next(p for p in [pathlib.Path('/kaggle/input/datasets/taeyangg4/knee-d01-codec-assets'),pathlib.Path('/kaggle/input/knee-d01-codec-assets')] if p.is_dir())
competition=next(p for p in [pathlib.Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),pathlib.Path('/kaggle/input/rsna-knee-abnormality-detection')] if (p/'test.csv').is_file())
subprocess.run([sys.executable,str(release/'d01_release_v2.py'),'execute','--release',str(release),'--assets',str(assets),'--codec',str(codec),'--competition',str(competition),'--work','/kaggle/working/d01-release-v2','--started',str(started)],check=True)
'''
    candidate=out/"candidate";candidate.mkdir()
    dump(candidate/"release-v2.ipynb",{"nbformat":4,"nbformat_minor":4,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[{"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(keepends=True)}]})
    dump(candidate/"kernel-metadata.json",metadata)
    return manifest


def r2_worker(assets,codec,competition,work,protocol,input_helper,preflight=False):
    assets,codec,competition,work,protocol,input_helper=map(Path,(assets,codec,competition,work,protocol,input_helper))
    work.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    receipt={"status":"RUNNING","header_policy":HEADER_POLICY,"failures":[]}
    try:
        need(sha(assets/"d01-assets-manifest.json")==ASSETS_SHA,"R2 asset identity")
        p=read(protocol);need(p["assets_manifest_sha256"]==ASSETS_SHA and sha(input_helper)==p["helper_sha256"],"R2 input protocol/helper identity")
        package=load_module(assets/"d01_package.py","d01_original_package")
        package._validate_assets(assets)
        wheels=[f for f in codec.rglob("*.whl") if sha(f)==GDCM_SHA];need(len(wheels)==1,"gdcm wheel identity")
        destination=work/"codec"
        subprocess.run([sys.executable,"-m","pip","install","--no-index","--no-deps","--target",str(destination),str(wheels[0])],check=True)
        sys.path.insert(0,str(destination));package._activate_offline_environment(assets,work)
        import numpy as np
        import pandas as pd
        import torch
        from raptor_common import environment,write_predictions,select_centers
        from raptor_dicom import DicomCorpus,DirectoryDicomSource
        from raptor_ensemble import load_members,predict_ensemble
        from pydicom.pixels import get_decoder
        need(torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0),"actual T4 required")
        torch.set_num_threads(2)
        need(all(get_decoder(f"1.2.840.10008.1.2.4.{n}").is_available for n in (57,70,90,91)),"compressed decoder capability missing")
        members,_=load_members(assets/"d01-assets-manifest.json")
        with DirectoryDicomSource(competition/"test_series") as source:
            corpus=DicomCorpus(competition/"test.csv",competition/"test_series.csv",source,"raptor64_336_c140_s02")
            reference_results=[]
            if preflight:
                expected=read(assets/"validation/expected_test3.json")
                need(corpus.ids==expected["ids"],"preflight requires exact public3")
                helper=load_module(input_helper,"d01_input_gate_frozen")
                for uid,identity in zip(corpus.ids,expected["study_hashes"],strict=True):
                    volume,mask=corpus.volume(uid),corpus.mask(uid)
                    need(helper.array_sha(volume)==identity["volume_sha256"] and helper.array_sha(mask)==identity["mask_sha256"],"public3 integer input changed")
                    centers=select_centers(mask,corpus.depth,24,corpus.spec,policy="legacy")
                    reference_results.append(helper.check_case(volume,mask,centers,helper.expected_row(uid,volume,mask,centers),p))
                need(all(r["pass"] for r in reference_results),"public3 numerical reference failed")
            mean,raw,inference=predict_ensemble(members,corpus,amp="fp16",device="cuda",batch=1,residency="resident",runtime_policy="strict")
            need(raw.shape==(5,len(corpus.ids),12) and np.isfinite(raw).all(),"R2 incomplete/nonfinite")
            receipt.update(status="COMPLETE",assets_manifest_sha256=ASSETS_SHA,input_protocol_sha256=sha(protocol),codec_wheel_sha256=GDCM_SHA,
                           release_helper_sha256=sha(__file__),inference=inference,environment=environment(),ids=corpus.ids,
                           study_receipts=list(corpus.receipts.values()),seconds=time.perf_counter()-started,cpu_threads=2)
            write_predictions(work/"mean.csv",corpus.ids,mean)
            np.savez_compressed(work/"members.npz",ids=np.asarray(corpus.ids),member_probabilities=raw)
            if preflight:
                local=pd.read_csv(assets/"validation/local_f0_test3.csv",dtype={package.UID:str}).set_index(package.UID).loc[corpus.ids,package.LABELS].to_numpy()
                delta=np.abs(raw[0].astype(np.float64)-local)
                passed=delta.max()<=.002 and delta.mean()<=.0005 and inference["fp32_fallbacks"]==0 and inference["neutral_fills"]==0
                gate={"schema":"d01-t4-release-gate-v2","status":"PASS" if passed else "FAILED", "assets_manifest_sha256":ASSETS_SHA,
                      "input_protocol_sha256":sha(protocol),"codec_wheel_sha256":GDCM_SHA,"header_policy":HEADER_POLICY,
                      "release_helper_sha256":sha(__file__),"member_count":5,"studies":3,"all_finite":True,
                      "cpu_threads":2,
                      "fp32_fallbacks":inference["fp32_fallbacks"],"neutral_fills":inference["neutral_fills"],
                      "input_reference_pass":all(r["pass"] for r in reference_results),"input_reference_checks":reference_results,
                      "f0_max_abs":float(delta.max()),"f0_mean_abs":float(delta.mean()),"tolerances":{"f0_max_abs":.002,"f0_mean_abs":.0005},"environment":receipt["environment"]}
                dump(work/"t4-gate.json",gate);need(passed,"fresh v2 T4 probability gate failed")
    except Exception as exc:
        receipt["status"]="FAILED";failure={"error":str(exc),"traceback":traceback.format_exc()}
        if hasattr(exc,"receipt"):failure["inference_receipt"]=exc.receipt
        receipt["failures"].append(failure)
        (work/"mean.csv").unlink(missing_ok=True)
        raise
    finally:dump(work/"receipt.json",receipt)


def execute(release,assets,codec,competition,work,started):
    release,assets,codec,competition,work=map(Path,(release,assets,codec,competition,work))
    output=OUTPUT
    need(not output.exists() and not work.exists(),"refuse stale output/work")
    work.mkdir(parents=True)
    receipt={"status":"RUNNING","v1_result":"FAILED_UNCHANGED"}
    try:
        manifest=read(release/"release-manifest.json")
        need(manifest["release_helper_sha256"]==sha(__file__),"release helper changed")
        need(sha(release/"anchor-original.ipynb")==manifest["anchor_sha256"],"anchor changed")
        gates,projection=validate_gates(manifest["gates"],release,manifest["anchor_sha256"],sha(__file__))
        plan=gates["runtime_gate"]
        deadline=float(started)+plan["whole_seconds"]-plan["safety_seconds"]
        need(0<=time.monotonic()-float(started)<plan["whole_seconds"],"invalid shared start clock")
        import csv
        with (competition/"test.csv").open(newline="",encoding="utf-8") as f:ids=[r["StudyInstanceUID"] for r in csv.DictReader(f)]
        need(0<len(ids)<=plan["max_studies"] and len(set(ids))==len(ids),"cohort exceeds measured runtime plan")
        env=dict(os.environ,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",CUBLAS_WORKSPACE_CONFIG=":4096:8")
        command=[sys.executable,str(release/"d01_release_v2.py"),"r2","--assets",str(assets),"--codec",str(codec),"--competition",str(competition),"--work",str(work/"r2"),"--protocol",str(release/"protocol.json"),"--input-helper",str(release/"d01_input_gate.py")]
        run_child(command,deadline,plan["r2_seconds"],work/"r2.log",env)
        anchor=anchor_worker_notebook(read(release/"anchor-original.ipynb"));dump(work/"anchor-worker.ipynb",anchor)
        anchor_env=dict(os.environ,D01_ANCHOR_OUTPUT=str(work/"anchor.csv"),RSNA_COMP_ROOT=str(competition))
        anchor_env.pop("RSNA_PARALLEL_COAT_READERS",None)
        run_child([sys.executable,"-m","jupyter","nbconvert","--to","notebook","--execute",str(work/"anchor-worker.ipynb"),"--output","anchor-executed.ipynb","--output-dir",str(work),"--ExecutePreprocessor.timeout=-1"],deadline,plan["anchor_seconds"],work/"anchor.log",anchor_env)
        need(time.monotonic()<deadline,"shared deadline expired before blend")
        r2=read(work/"r2/receipt.json");need(r2["status"]=="COMPLETE" and r2["assets_manifest_sha256"]==ASSETS_SHA,"R2 release receipt incomplete")
        package=load_module(assets/"d01_package.py","d01_blend_original")
        import pandas as pd
        frame=package.blend_frames(pd.read_csv(work/"anchor.csv",dtype={package.UID:str}),pd.read_csv(work/"r2/mean.csv",dtype={package.UID:str}),ids)
        temporary=work/"final.csv.tmp";frame.to_csv(temporary,index=False)
        need(time.monotonic()<deadline,"shared deadline expired before publish")
        receipt.update(status="COMPLETE",studies=len(ids),submission_sha256=sha(temporary),release_manifest_sha256=sha(release/"release-manifest.json"),elapsed_seconds=time.monotonic()-float(started),projection=projection)
        dump(work/"release-receipt.json",receipt)
        os.replace(temporary,output)
    except BaseException as exc:
        output.unlink(missing_ok=True)
        receipt.update(status="FAILED",error=str(exc),traceback=traceback.format_exc())
        dump(work/"release-receipt.json",receipt)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="action",required=True)
    p=sub.add_parser("build");p.add_argument("--spec-path",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("r2")
    for name in ("assets","codec","competition","work","protocol","input-helper"):p.add_argument("--"+name,required=True)
    p.add_argument("--preflight",action="store_true")
    p=sub.add_parser("execute")
    for name in ("release","assets","codec","competition","work"):p.add_argument("--"+name,required=True)
    p.add_argument("--started",required=True,type=float)
    args=vars(parser.parse_args());action=args.pop("action")
    {"build":build,"r2":r2_worker,"execute":execute}[action](**args)


if __name__=="__main__":main()
