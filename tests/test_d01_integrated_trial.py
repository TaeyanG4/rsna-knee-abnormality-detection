"""CPU checks for the bounded single-trial protocol and process lifecycle."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import d01_integrated_trial as trial
import d01_r2_shard_benchmark as supervisor


def test_frozen100_prefix_and_deterministic_nonlabel_extension():
    prefix=[f'old{i}' for i in range(100)];eligible=prefix+[f'new{i}' for i in range(80)]
    result=trial.extension_ids(prefix,eligible,['new5'])
    assert result[:100]==prefix and len(result)==len(set(result))==133 and 'new5' not in result
    assert result==trial.extension_ids(prefix,list(reversed(eligible)),['new5'])
    with pytest.raises(ValueError):trial.extension_ids(prefix,eligible,['old1'])


def test_projection_preserves_prior_floor_and_separate_reserves():
    value=trial.projection(100)
    assert value['projected1300_seconds']==30938.468180919
    assert value['minimum_reserve_pass'] and not value['thirty_minute_target_pass']
    assert not value['thirty_percent_runtime_overhead_pass']
    assert not trial.projection(3400)['minimum_reserve_pass']
    for bad in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):trial.projection(bad)


def test_original_anchor_cells_and_memmap_assertion_are_preserved():
    path=ROOT/'kaggle/packages/KNEE_ANCHOR_0943/rsna-knee-fast-2xt4-inference.ipynb'
    assert trial.sha(path)==trial.ANCHOR_SHA
    original=trial.read(path)
    protocol={'ids':[f'u{i}' for i in range(133)],'train_series_sha256':'x','expected133_dino_bytes':133*8128512}
    actual=trial.anchor_notebook(original,protocol)
    assert actual['cells'][1:-1]==original['cells'] and len(actual['cells'])==32
    assert "scratch_allocation" in ''.join(actual['cells'][-1]['source'])


def evidence():
    ids=[f'u{i}' for i in range(133)]
    raw=np.random.default_rng(2).random((5,133,12),dtype=np.float32)
    members=[{'name':f'F{i}','checkpoint_sha256':str(i)} for i in range(5)]
    rows=[{'uid':u,'volume_sha256':'v'+u,'mask_sha256':'m'+u} for u in ids]
    tensors=[{'uid':u,'sha256':'t'+u} for u in ids]
    prior={'study_receipts':rows[:100],'inference':{'member_results':members,'input_tensor_hashes':tensors[:100]}}
    current={'status':'COMPLETE','ids':ids,'studies':133,'release_helper_sha256':trial.V3_SHA,'policy_sha256':'policy',
        'member_count':5,'all_finite':True,'fp32_fallbacks':0,'neutral_fills':0,'mean_before_rank':True,
        'member_identities':[{'name':m['name'],'sha256':m['checkpoint_sha256']} for m in members],
        'study_receipts':copy.deepcopy(rows),'input_tensor_hashes':copy.deepcopy(tensors)}
    protocol={'ids':ids,'helper_policy_sha256':'policy','extension_input_reference':rows[100:]}
    return current,ids,raw,prior,ids[:100],raw[:,:100].copy(),protocol,supervisor.probability_gate


def test_prefix100_gate_keeps_existing_probability_tolerance():
    args=list(evidence());args[2][0,0,0]+=.0001
    result=trial.verify_r2(*args)
    assert result['probability100']['pass'] and len(result['extension33'])==33


@pytest.mark.parametrize('case',['missing','prefix_receipt','prefix_tensor','extra_input','helper','probability','nonfinite','order'])
def test_required_regression_rejects_changed_contract(case):
    args=list(evidence())
    if case=='missing':args[0]['study_receipts'].pop()
    if case=='prefix_receipt':args[0]['study_receipts'][0]['volume_sha256']='bad'
    if case=='prefix_tensor':args[0]['input_tensor_hashes'][0]['sha256']='bad'
    if case=='extra_input':args[0]['study_receipts'][132]['volume_sha256']='bad'
    if case=='helper':args[0]['release_helper_sha256']='bad'
    if case=='probability':args[2][0,0,0]=args[5][0,0,0]+.1
    if case=='nonfinite':args[2][0,132,0]=np.nan
    if case=='order':args[1]=list(reversed(args[1]))
    with pytest.raises(ValueError):trial.verify_r2(*args)


def resource():
    return {'cgroup':{'memory.current':trial.GIB,'memory.max':32*trial.GIB,'memory.events':{'oom':5,'oom_kill':2}},
        'gpu':[{'index':i,'name':'Tesla T4','used_mib':10.,'total_mib':15360.} for i in (0,1)],
        'rss_tree':100,'children':[],'disks':{'/kaggle/working':{'free':10*trial.GIB}}}


def test_oom_is_delta_and_driver_os_cache_not_a_leak():
    before=resource();after=copy.deepcopy(before)
    trial.resource_gate(before,after)
    after['cgroup']['memory.current']+=5*trial.GIB;after['gpu'][0]['used_mib']+=400
    trial.resource_gate(before,after)
    after['cgroup']['memory.events']['oom_kill']+=1
    with pytest.raises(ValueError,match='oom_kill'):trial.resource_gate(before,after)


@pytest.mark.parametrize('kind',['success','failure','timeout'])
def test_actual_cpu_process_lifecycle(monkeypatch,tmp_path,kind):
    monkeypatch.setattr(trial,'snapshot',resource)
    code={'success':'print("ok")','failure':'raise RuntimeError("expected")','timeout':'import time;time.sleep(20)'}[kind]
    work=tmp_path/kind
    args=(kind,[sys.executable,'-c',code],dict(os.environ),work,.3 if kind=='timeout' else 8,trial.time.monotonic()+12,supervisor)
    if kind=='success':
        value=trial.run_phase(*args);assert value['all_owned_processes_reaped'] and value['status']=='COMPLETE'
    else:
        with pytest.raises(ValueError):trial.run_phase(*args)
        assert trial.read(work/'phase-receipt.json')['status']=='FAILED'
