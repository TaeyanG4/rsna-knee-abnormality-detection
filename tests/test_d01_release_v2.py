"""Fail-closed release gates/process lifetime; no GPU or Kaggle operations."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

spec=importlib.util.spec_from_file_location("release_v2",Path(__file__).resolve().parents[1]/"tools/d01_release_v2.py")
release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)


def valid_gates(path):
    expected={"ids":[f"study{i}" for i in range(100)],"synthetic":[{"uid":f"synthetic{i}"} for i in range(7)]}
    def numeric(uid):
        return {"uid":uid,"pass":True,"exact_input":True,"exact_coefficients":True,"finite_channel_ranges":True,
                "windows":{"pass":True,"exceedances":0,"atol":5e-7,"rtol":0,"max_abs":1e-7},
                "normalized":{"pass":True,"exceedances":0,"atol":3e-6,"rtol":0,"max_abs":7e-7}}
    release.dump(path/"expected_inputs.json",expected)
    protocol={"schema":"d01-numerical-input-gate-v2","v1_result":"FAILED_UNCHANGED","window_atol":5e-7,"normalized_atol":3e-6,"rtol":0.,
              "assets_manifest_sha256":release.ASSETS_SHA,"gdcm_wheel_sha256":release.GDCM_SHA,"helper_sha256":"input-helper",
              "expected_sha256":release.sha(path/"expected_inputs.json")}
    release.dump(path/"protocol.json",protocol)
    psha=release.sha(path/"protocol.json")
    n={"status":"PASS","protocol_sha256":psha,"helper_sha256":"input-helper","cases":[numeric(u) for u in expected["ids"]],
       "synthetic":[numeric(r["uid"]) for r in expected["synthetic"]]}
    local={**copy.deepcopy(n),"mode":"local_cache"};remote={**copy.deepcopy(n),"mode":"remote_dicom"}
    syntaxes=["1.2.840.10008.1.2","1.2.840.10008.1.2.1"]+[f"1.2.840.10008.1.2.4.{s}" for s in (57,70,90,91)]
    codec={"status":"PASS","gdcm_version":"3.2.6","pydicom_version":"3.0.2","decoders":[{"uid":u,"available":True} for u in syntaxes],
           "fixtures":[{"decode":"SUCCESS","matches_uncompressed_fixture":True}]*3}
    t4={"schema":"d01-t4-release-gate-v2","status":"PASS","assets_manifest_sha256":release.ASSETS_SHA,"input_protocol_sha256":psha,
        "codec_wheel_sha256":release.GDCM_SHA,"header_policy":release.HEADER_POLICY,"release_helper_sha256":"helper","member_count":5,
        "studies":3,"all_finite":True,"cpu_threads":2,"fp32_fallbacks":0,"neutral_fills":0,"f0_max_abs":.002,"f0_mean_abs":.0005,
        "tolerances":{"f0_max_abs":.002,"f0_mean_abs":.0005},"input_reference_pass":True,"environment":{"gpu":"Tesla T4"}}
    r2={"status":"COMPLETE","assets_manifest_sha256":release.ASSETS_SHA,"completed_studies":100,
        "inference":{"member_count":5,"fp32_fallbacks":0,"neutral_fills":0,"residency":"resident","effective_amp":"fp16","batch_size":1,"runtime":{"policy":"strict"}},"timings":{"worker_total_seconds":110,"inference_including_model_placement_offload":100},
        "study_time_distribution":{"mean":1}}
    anchor={"status":"COMPLETE","benchmark_studies":50,"coat_readers_parallel_pairs":False,"coat_family_members":4,"anchor_notebook_sha256":"anchor","total_cell_seconds":100}
    for name,value in [("numeric_local",local),("numeric_remote",remote),("codec",codec),("t4",t4),("r2_runtime",r2),("anchor_runtime",anchor)]:release.dump(path/f"{name}.json",value)
    plan={"schema":"d01-runtime-budget-v2","status":"PASS","r2_receipt_sha256":release.sha(path/"r2_runtime.json"),
          "anchor_receipt_sha256":release.sha(path/"anchor_runtime.json"),"header_policy":release.HEADER_POLICY,"max_studies":1300,
          "whole_seconds":32400,"r2_seconds":1500,"anchor_seconds":3000,"safety_seconds":600}
    release.dump(path/"runtime_gate.json",plan)
    return {name:{"path":f"{name}.json","sha256":release.sha(path/f"{name}.json")} for name in release.GATE_NAMES}


def mutate(path,bindings,name,change):
    value=release.read(path/f"{name}.json");change(value);release.dump(path/f"{name}.json",value)
    bindings[name]["sha256"]=release.sha(path/f"{name}.json")


def test_valid_exact_bindings_and_threshold_boundary(tmp_path):
    bindings=valid_gates(tmp_path)
    _,projection=release.validate_gates(bindings,tmp_path,"anchor","helper")
    assert projection["r2_seconds"]==1310 and projection["anchor_seconds"]==2600


@pytest.mark.parametrize("case",["old_v1","missing_uid","relaxed_tolerance","wrong_codec","wrong_header","budget_short","wrong_hash"])
def test_rejects_incomplete_wrong_or_relaxed_evidence(tmp_path,case):
    bindings=valid_gates(tmp_path)
    changes={
        "old_v1":("t4",lambda x:x.update(schema="d01-preflight-v1",status="FAILED")),
        "missing_uid":("numeric_remote",lambda x:x["cases"].pop()),
        "relaxed_tolerance":("numeric_remote",lambda x:x["cases"][0]["windows"].update(atol=1e-3)),
        "wrong_codec":("t4",lambda x:x.update(codec_wheel_sha256="changed")),
        "wrong_header":("runtime_gate",lambda x:x.update(header_policy="unmeasured-header-only")),
        "budget_short":("runtime_gate",lambda x:x.update(r2_seconds=360)),
    }
    if case=="wrong_hash":(tmp_path/"t4.json").write_text("{}")
    else:
        name,change=changes[case];mutate(tmp_path,bindings,name,change)
    with pytest.raises(ValueError):release.validate_gates(bindings,tmp_path,"anchor","helper")


def test_anchor_original30cells_preserved():
    root=Path(__file__).resolve().parents[1]
    metadata=release.read(root/"kaggle/packages/KNEE_ANCHOR_0943/kernel-metadata.json")
    original=release.read(root/"kaggle/packages/KNEE_ANCHOR_0943"/metadata["code_file"])
    worker=release.anchor_worker_notebook(original)
    assert len(original["cells"])==30 and worker["cells"][:-1]==original["cells"]
    assert len(worker["cells"])==31


def test_shared_deadline_refuses_launch_after_expiry(tmp_path):
    with pytest.raises(TimeoutError):release.run_child([sys.executable,"-c","raise AssertionError('must not start')"],time.monotonic()-1,10,tmp_path/"log")
    assert not (tmp_path/"log").exists()


def test_timeout_terminates_descendant_process(tmp_path):
    pid_path=tmp_path/"child.pid"
    child="import time; time.sleep(30)"
    parent=f"import subprocess,sys,pathlib,time; p=subprocess.Popen([sys.executable,'-c',{child!r}]); pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(30)"
    with pytest.raises(subprocess_timeout_types()):
        release.run_child([sys.executable,"-c",parent],time.monotonic()+3,2,tmp_path/"log")
    assert pid_path.exists()
    pid=int(pid_path.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE


def subprocess_timeout_types():
    import subprocess
    return (subprocess.TimeoutExpired,TimeoutError)


def test_failed_release_removes_only_new_submission_and_preserves_prior(tmp_path,monkeypatch):
    output=tmp_path/"submission.csv";monkeypatch.setattr(release,"OUTPUT",output)
    output.write_text("prior")
    with pytest.raises(ValueError):release.execute(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path/"work",time.monotonic())
    assert output.read_text()=="prior"
    output.unlink()
    release.dump(tmp_path/"release-manifest.json",{"release_helper_sha256":"wrong"})
    with pytest.raises(ValueError):release.execute(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path/"work",time.monotonic())
    assert not output.exists()
    assert release.read(tmp_path/"work/release-receipt.json")["status"]=="FAILED"


def test_build_missing_gates_does_not_create_candidate(tmp_path):
    release.dump(tmp_path/"spec.json",{"assets":"missing","anchor":"missing","gates":{}})
    with pytest.raises((FileNotFoundError,ValueError)):release.build(tmp_path/"spec.json",tmp_path/"candidate")
    assert not (tmp_path/"candidate").exists()


def test_anchor_worker_failure_after_writing_csv_cannot_publish(tmp_path,monkeypatch):
    output=tmp_path/"submission.csv";monkeypatch.setattr(release,"OUTPUT",output)
    release.dump(tmp_path/"anchor-original.ipynb",{"cells":[],"metadata":{}})
    release.dump(tmp_path/"release-manifest.json",{"release_helper_sha256":release.sha(release.__file__),
                 "anchor_sha256":release.sha(tmp_path/"anchor-original.ipynb"),"gates":{}})
    (tmp_path/"test.csv").write_text("StudyInstanceUID\na\n")
    plan={"whole_seconds":32400,"safety_seconds":600,"r2_seconds":100,"anchor_seconds":100,"max_studies":1300}
    monkeypatch.setattr(release,"validate_gates",lambda *args:({"runtime_gate":plan},{}))
    calls=[]
    def worker(*args):
        calls.append(args)
        if len(calls)==2:
            output.write_text("partial anchor output")
            raise RuntimeError("anchor reader failed after partial output")
    monkeypatch.setattr(release,"run_child",worker)
    with pytest.raises(RuntimeError,match="anchor reader failed"):
        release.execute(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path/"work",time.monotonic())
    assert len(calls)==2 and not output.exists()
    assert release.read(tmp_path/"work/release-receipt.json")["status"]=="FAILED"
