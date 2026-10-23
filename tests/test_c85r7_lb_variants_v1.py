"""Four independent sealed hypotheses: math, authentication and executable payload."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from tools import c85r7_lb_variants_v1 as m
from tools import c85r7_lb_variants_verify_v1 as v
from tools import c85r7_lb_r2_adapter_v1 as a

@pytest.mark.parametrize('tag',list(m.SPECS))
def test_real_visible_replay(tag):
    result=v.replay(tag)
    assert result['studies']==3 and result['parent_noop_csv_bytes']

@pytest.mark.parametrize('tag',list(m.SPECS))
def test_synthetic_formula_with_ties(tag):
    reg=a.registration(); rng=np.random.default_rng(140)
    raw={k:rng.integers(1,10,(13,12)).astype(np.float32)/11 for k in m.n.roster(reg)+['R']}
    own=tuple(rng.integers(1,10,(13,12)).astype(np.float64)/11 for _ in range(3))
    runtime=v.module(tag)
    if tag=='compo15': got=runtime.blend(reg,{k:raw[k] for k in runtime.roster(reg)})[0]
    else:
        r=raw['R'] if tag!='r2' else np.mean([raw['R2-s43'],raw['R2-s44']],axis=0,dtype=np.float32)
        got=runtime.blend([str(i) for i in range(13)],r,raw['S_R7'],*own)
    expected=v.reference(tag,reg,raw,own)
    assert m.b.p.csv_bytes([str(i) for i in range(13)],got)==m.b.p.csv_bytes([str(i) for i in range(13)],expected)

def test_fluid_isolation_and_failed_r():
    rng=np.random.default_rng(42); ids=['a','b','c','d','e']
    inputs=[rng.random((5,12)) for _ in range(5)]
    runtime=v.module('fluidR')
    got=runtime.blend(ids,*inputs); old=m.b.blend(ids,*inputs)
    keep=[j for j in range(12) if j not in (7,9)]
    np.testing.assert_array_equal(got[:,keep],old[:,keep])
    failed={'R':dict.fromkeys(ids,'failed')}
    got=runtime.blend(ids,*inputs,failures=failed)
    own=(17*v.rank(inputs[2])+17*v.rank(inputs[3])+12*v.rank(inputs[4]))/46
    np.testing.assert_array_equal(got[:,[7,9]],v.rank(own)[:,[7,9]])

def test_r2_partial_and_tampered_identity(tmp_path):
    reg=a.registration();ids=['a','b','c']
    journal=m.h.c.Journal(tmp_path/'NATIVE/worker0',ids,a.identity(reg))
    journal.record('a','R2-s43',np.full(12,.2,np.float32))
    journal.record('a','R2-s44',np.full(12,.4,np.float32))
    journal.record('b','R2-s44',np.full(12,.6,np.float32))
    values,failed,_=a.recover(tmp_path,ids,reg)
    np.testing.assert_array_equal(values[:,0],np.array([.3,.6,.5],np.float32))
    assert set(failed)=={'c'}
    identity=a.identity(reg);identity['members']['R2-s43']='0'*64
    m.h.dump(tmp_path/'NATIVE/worker0/journal-identity.json',identity)
    with pytest.raises(ValueError,match='identity'): a.recover(tmp_path,ids,reg)

def test_composition_rejects_unsealed_arm_and_stress():
    runtime=v.module('compo15')
    with pytest.raises(ValueError): runtime.config('B')
    with pytest.raises(ValueError): runtime.config('A',True)
    assert runtime.config()['rx']=='R' and runtime.config()['outer']==[.85,.15]

@pytest.mark.parametrize('tag',list(m.SPECS))
def test_notebook_bootstrap_in_isolated_process(tag,tmp_path):
    files,nb,meta,_,_=m.payload(tag)
    assert meta['is_private'] is True and meta['enable_internet'] is False
    parent=m.h.read(m.ROOT/m.SPECS[tag]['parent']/'kernel-metadata.json')
    for key in ('docker_image','docker_image_pinning_type','docker_image_version_id','machine_shape','competition_sources'):
        if key in parent: assert meta[key]==parent[key]
    assert set(parent['dataset_sources'])<=set(meta['dataset_sources'])
    assert meta['code_file']=='release.ipynb'
    for name,data in files.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        if name.endswith('.py'): compile(data,name,'exec')
    module='c85r7_lb_'+tag.lower()+'_runtime'
    statement=f'from tools import {module} as r' if tag=='compo15' else f'import {module} as r'
    code='import sys;sys.path.insert(0,sys.argv[1]);'+statement+';print(r.VERSION);'
    if tag=='r2': code+="print(r.r2_adapter.registration(sys.argv[1])['P']['Rplus2']['members'][0]['seed'])"
    if tag=='compo15': code+='assert r.config()["rx"]=="R";assert r.config()["outer"]==[.85,.15]'
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(tmp_path)],cwd=tmp_path,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1'),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    code=''.join(nb['cells'][-1]['source'])
    tree=ast.parse(code)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('serve','run_release')]
    assert len(calls)==1
    if tag=='compo15': assert 'rx' not in [x.arg for x in calls[0].keywords]

def test_r2_collector_uses_authenticated_seed_mean(tmp_path):
    from tools import exp169_c85r7ins_checks_v122 as check
    ids=['study02','study00','study01'];check.fixture(tmp_path,ids)
    runtime=v.module('r2');reg=a.registration()
    raw={'R2-s43':np.full((3,12),.2,np.float32),'R2-s44':np.full((3,12),.4,np.float32)}
    pr7=tmp_path/runtime.RUN/'pr7';v.r2_journal(pr7/'pr7-run/r2',reg,ids,raw)
    rec,payload=runtime.collect(tmp_path/runtime.RUN/'a172',pr7,tmp_path/'export',cohort=ids)
    assert rec['pr7']['r_prediction_fallbacks']==0
    assert rec['pr7']['r2_identity']['members']==a.identity(reg)['members']
    assert rec['intended_members'][:2]==['R2-s43','R2-s44']
