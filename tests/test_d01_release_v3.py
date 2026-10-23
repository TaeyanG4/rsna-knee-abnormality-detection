"""CPU-only arbitrary-cohort, exact merge, unchanged blend and runtime binding tests."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import d01_release_v3 as release


@pytest.mark.parametrize("count",[1,2,3,50,100,1299,1300])
def test_arbitrary_n_disjoint_order_and_float32_mean(count):
    ids=[f"u{i}" for i in range(count)]; shards=release.split_ids(ids)
    assert len(shards[0])==(count+1)//2 and len(shards[1])==count//2
    raw=np.random.default_rng(count).random((5,count,12),dtype=np.float32)
    parts=[(shards[1],raw[:,1::2]),(shards[0],raw[:,::2])]
    merged,mean=release.merge_members(ids,parts)
    np.testing.assert_array_equal(merged,raw)
    np.testing.assert_array_equal(mean,raw.mean(0,dtype=np.float32))


def test_single_study_skips_empty_shard_without_neutral_fill():
    raw=np.full((5,1,12),.23,dtype=np.float32)
    assert release.split_ids(["single"])==[["single"],[]]
    merged,mean=release.merge_members(["single"],[(["single"],raw)])
    np.testing.assert_array_equal(merged,raw);assert mean.shape==(1,12)
    with pytest.raises(ValueError,match="nonempty"):release.split_ids([])


@pytest.mark.parametrize("case",["duplicate","missing","wrong_target","nonfinite","wrong_dtype"])
def test_fail_closed_merge(case):
    ids=["a","b","c"]; raw=np.full((5,3,12),.3,dtype=np.float32)
    parts=[(["a","c"],raw[:,::2].copy()),(["b"],raw[:,1::2].copy())]
    if case=="duplicate":parts[1][0][0]="a"
    if case=="missing":parts.pop()
    if case=="wrong_target":parts[1]=(parts[1][0],parts[1][1][:,:,:11])
    if case=="nonfinite":parts[1][1][0,0,0]=np.nan
    if case=="wrong_dtype":parts[1]=(parts[1][0],parts[1][1].astype(np.float64))
    with pytest.raises(ValueError):release.merge_members(ids,parts)


def test_unchanged_original_rank90_10_after_mean5():
    pd=pytest.importorskip("pandas")
    assets=ROOT/".local/d01-staging/20261003-v1/assets"
    spec=importlib.util.spec_from_file_location("frozen_package",assets/"d01_package.py")
    package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
    ids=["a","b","c"]
    anchor=pd.DataFrame(np.tile([.2,.2,.8],(12,1)).T,columns=package.LABELS);anchor.insert(0,package.UID,ids)
    raw=np.tile(np.array([.9,.1,.5],dtype=np.float32)[None,:,None],(5,1,12));raw[0,:,0]=[.1,.8,.4]
    _,mean=release.merge_members(ids,[(ids[::2],raw[:,::2]),(ids[1::2],raw[:,1::2])])
    r2=pd.DataFrame(mean,columns=package.LABELS);r2.insert(0,package.UID,ids)
    actual=release.blend(assets,anchor.iloc[::-1],r2.iloc[::-1],ids)
    expected=.9*anchor[package.LABELS].rank(method="average",pct=True)+.1*r2[package.LABELS].rank(method="average",pct=True)
    np.testing.assert_array_equal(actual[package.LABELS].to_numpy(),expected.to_numpy())
    assert actual[package.UID].tolist()==ids


def evidence():
    gate={"schema":"d01-t4-preflight-v3","mode":"train100","status":"PASS","release_helper_sha256":"helper",
          "policy_sha256":release.POLICY_SHA,"_source_sha256":"gate","r2_receipt_sha256":"r2","studies":100,
          "member_count":5,"all_finite":True,"fp32_fallbacks":0,"neutral_fills":0,
          "checks":dict.fromkeys(["fixed100","input_exact","normalized_tensor_exact","all5_probability_gate"],True)}
    launcher={"status":"COMPLETE","preflight_gate_sha256":"gate","notebook_cell_seconds":350.,"mode":"train100",
              "release_helper_sha256":"helper","policy_sha256":release.POLICY_SHA,"_source_sha256":"launcher"}
    anchor={"status":"COMPLETE","benchmark_studies":100,"coat_readers_parallel_pairs":False,
            "total_cell_seconds":2031.,"_source_sha256":"anchorreceipt","anchor_notebook_sha256":"anchor","coat_family_members":4}
    return gate,launcher,anchor


def test_budget_scales_whole_launcher_and_separates_minimum_from_target():
    plan=release.runtime_budget(*evidence(),"helper","anchor")
    assert plan["r2_seconds"]==350*13 and plan["anchor_seconds"]==2031*13
    assert plan["status"]=="PASS" and plan["minimum_reserve_pass"] and not plan["thirty_minute_target_pass"]


@pytest.mark.parametrize("case",["old_v2","helper_changed","schedule_changed","launcher_changed","anchor_changed","missing_case","too_little_time","reserve_relaxed"])
def test_budget_rejects_unbound_old_or_unsafe_evidence(case):
    gate,launcher,anchor=evidence();reserve=300
    if case=="old_v2":gate["schema"]="d01-t4-release-gate-v2"
    if case=="helper_changed":gate["release_helper_sha256"]="other"
    if case=="schedule_changed":gate["policy_sha256"]="other"
    if case=="launcher_changed":launcher["preflight_gate_sha256"]="other"
    if case=="anchor_changed":anchor["anchor_notebook_sha256"]="other"
    if case=="missing_case":gate["studies"]=99
    if case=="too_little_time":launcher["notebook_cell_seconds"]=1000
    if case=="reserve_relaxed":reserve=299
    if case=="too_little_time":
        assert release.runtime_budget(gate,launcher,anchor,"helper","anchor",reserve)["status"]=="FAILED"
    else:
        with pytest.raises(ValueError):release.runtime_budget(gate,launcher,anchor,"helper","anchor",reserve)
