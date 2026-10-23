"""Delta-only synthetic tests, no torch/model/real predictions or data access."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import rn3_cost_v1 as cost
from tools import b_rn3f_parity as parity


def members():
    return [dict(name=f'{fam}_F{f}',family=fam,arch='resnet34.a1_in1k' if fam=='RN' else 'coat',epoch_1based=20 if fam=='RN' else 8) for f in range(3) for fam in ['B','RN']]


def test_exact_three_subset_refuses_extra_wrong_order_or_B():
    allmembers=members();rn=[allmembers[i] for i in [1,3,5]];cost.validate_subset(rn,allmembers)
    for bad in (rn[::-1],allmembers,[allmembers[i] for i in [0,3,5]]):
        with pytest.raises(ValueError):cost.validate_subset(bad,allmembers)


def arrays():
    ids=[f'synthetic{i:03}' for i in range(100)]
    raw=np.repeat((np.arange(100,dtype=np.float32)/100)[None,:,None],3,axis=0)
    raw=np.repeat(raw,12,axis=2);return ids,raw


def test_fixed_fullcohort_rank_and_ties_not_probability_mean_rank():
    ids,raw=arrays();raw[:,0:3,:]=.2
    result,ranks=cost.numerical_gate(raw,raw.copy(),ids,ids,cost.NAMES,cost.TARGETS,parity)
    assert result['status']=='PASS' and ranks.shape==(100,12) and len(result['comparisons'])==36
    assert np.array_equal(ranks,np.stack([parity.rank_pct(v) for v in raw]).mean(0))
    assert ranks[0,0]==ranks[1,0]==ranks[2,0]


@pytest.mark.parametrize('fault',['order','subset','member','target','dtype','nan'])
def test_contract_rejections(fault):
    ids,raw=arrays();expected=ids.copy();names=cost.NAMES.copy();targets=cost.TARGETS.copy()
    if fault=='order':ids=ids[::-1]
    if fault=='subset':ids=ids[:50];raw=raw[:,:50]
    if fault=='member':names[0]='B_F0'
    if fault=='target':targets=targets[::-1]
    if fault=='dtype':raw=raw.astype(np.float64)
    if fault=='nan':raw[0,0,0]=np.nan
    with pytest.raises(ValueError):cost.numerical_gate(raw,raw.copy(),ids,expected,names,targets,parity)


def test_original_per_target_mean_and_max_limits_remain():
    ids,reference=arrays();raw=reference.copy();raw[0,0,0]+=.0021
    r,_=cost.numerical_gate(raw,reference,ids,ids,cost.NAMES,cost.TARGETS,parity)
    assert r['status']=='FAIL' and r['limits']=={'max_abs':.002,'p99_abs':.002,'mean_abs':.0005}
    raw=reference.copy();raw[0,:,0]+=.0006
    r,_=cost.numerical_gate(raw,reference,ids,ids,cost.NAMES,cost.TARGETS,parity)
    assert r['status']=='FAIL'


def test_observational_timer_boundaries_return_identity_and_preserve_rng():
    sync=[];timer=cost.Timers(lambda:sync.append('sync'));value=np.arange(4);before=np.random.get_state()
    actual=timer.call('prepare',lambda:value)
    assert actual is value and sync==['sync','sync'] and timer.counts=={'prepare':1}
    after=np.random.get_state();assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    model=object();timer.pre(model,());timer.post(model,(),value)
    assert timer.counts['forward']==1 and not timer.hook_starts


@pytest.mark.parametrize('elapsed,rss,size',[(900,1,1),(899,8*1024**3+1,1),(1,1,64*1024**2+1)])
def test_resource_caps_refuse_boundary_or_excess(elapsed,rss,size):
    with pytest.raises(ValueError):cost.check_budget(elapsed,rss,size)


def test_resource_childpass_cannot_hide_cleanup_or_oom():
    r=dict(schema='process-resource-receipt-v1',supervisor_sha256=cost.RESOURCE_SHA,retries=0,status='COMPLETE',wrapper_exit_code=0,child_exit_code=0,child_exit_code_before_cleanup=0,failure=None,sample_count=1,failed_sample_attempts=0,access_failure_count=0,cgroup_telemetry_complete=True,cleanup=dict(actions=[],errors=[],survivors=[]),cgroup_start={'values':{'memory.events':{'oom':0}}},cgroup_end={'values':{'memory.events':{'oom':0}}})
    cost.validate_resource(r)
    for bad in ['cleanup','oom','child']:
        x=deepcopy(r)
        if bad=='cleanup':x['cleanup']['actions']=['killed']
        if bad=='oom':x['cgroup_end']['values']['memory.events']['oom']=1
        if bad=='child':x['child_exit_code']=1
        with pytest.raises(ValueError):cost.validate_resource(x)


def test_late_finalization_removes_stale_pass(monkeypatch,tmp_path):
    ticks=iter([1.,901.]);monkeypatch.setattr(cost.time,'monotonic',lambda:next(ticks))
    with pytest.raises(ValueError):cost.publish_result(tmp_path,{'status':'PASS'},0.,1)
    assert not (tmp_path/'result.json').exists() and not (tmp_path/'result.pending.json').exists()


def test_prepare_source_does_not_interpret_real_npz_or_load_models():
    tree=ast.parse((ROOT/'tools/rn3_cost_v1.py').read_text(encoding='utf-8'))
    prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    code=ast.unparse(prepare)
    assert 'np.load' not in code and 'load_members(' not in code and 'torch.load' not in code
    assert "'reference-members.npz'" in code and 'source.read_bytes()' in code
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(x in ast.unparse(n) for x in ['torch','raptor_common','raptor_dicom']) for n in tree.body)
