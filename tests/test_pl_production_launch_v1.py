import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tools import pl_production_launch_v1 as launch
from tools import pl_production_targets_v1 as t

def cohort():
    ids=[*t.exclusions.REGISTERED_UIDS,*[f'study-{i}' for i in range(48)]]
    rows=pd.DataFrame({t.pl.UID:ids,'component_id':ids,'fold':[str(i%5) for i in range(50)]})
    return t.build_cohort(rows,[],ids[2:],ids[:2]),ids

def test_go_absent_refuses_before_other_inputs(tmp_path):
    with pytest.raises(ValueError,match='GO absent'):
        launch.prepare(tmp_path/'absent',None,None)

@pytest.mark.parametrize('value',[{}, {'registered_go':{'selected_arm':'control'}}])
def test_no_winning_arm_refuses(tmp_path,value):
    path=tmp_path/'go.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='winning arm absent'):launch.selected(path)

def test_selected_uses_shared_validator(tmp_path,monkeypatch):
    path=tmp_path/'go.json';path.write_text(json.dumps({'registered_go':{'selected_arm':'global'}}))
    def reject(pin,arm):raise ValueError('authentic GO required')
    monkeypatch.setattr(launch.go_adapter,'validate',reject)
    with pytest.raises(ValueError,match='authentic GO'):launch.selected(path)

def test_review_absent_and_bypass(tmp_path):
    with pytest.raises(ValueError,match='review absent'):launch.review_gate(None)
    path=tmp_path/'review.json';path.write_text(json.dumps(dict(status='PASS',independent=True,reviewer='B',sources=[launch.asset(x) for x in launch.SOURCES])))
    with pytest.raises(ValueError,match='requires suite'):launch.review_gate(path)

def test_fivefold_complete_component_disjoint():
    c,ids=cohort();seen=[]
    for fold in range(5):
        train,val=t.a3.selected_rosters(c,'fivefold',fold)
        assert set(train)|set(val)==set(ids)
        assert not set(train)&set(val)
        seen+=val
    assert sorted(seen)==sorted(ids)

def test_production_includes_labelled_gold_as_descriptive_only():
    _,ids=cohort()
    rows=pd.DataFrame({t.pl.UID:ids,'component_id':ids,'fold':[str(i%5) for i in range(50)]})
    c=t.build_cohort(rows,ids[10:12],ids[2:],ids[:2])
    assert c['eligible_ids']==ids and c['gold_ids']==[]
    assert c['descriptive_gold_ids']==ids[10:12]
    counts={u:0 for u in ids}
    for fold in range(5):
        train,val=t.a3.selected_rosters(c,'fivefold',fold)
        for u in train:counts[u]+=1
        assert not set(train)&set(val)
    assert set(counts.values())=={4}

def test_review_cannot_record_failed_real_smoke(tmp_path):
    receipt=tmp_path/'smoke.json'
    receipt.write_text(json.dumps(dict(status='HOLD',kind='real_cpu_smoke',sources=[launch.asset(x) for x in launch.SOURCES])))
    with pytest.raises(ValueError,match='smoke source binding'):
        launch.record_review('B',[receipt],tmp_path/'review.json')
    assert not (tmp_path/'review.json').exists()

@pytest.mark.parametrize('arm',['global','fill'])
def test_missing_teachers_are_exact_l2_and_scale_only_available(tmp_path,arm):
    c,ids=cohort();rng=np.random.default_rng(7)
    l2={u:rng.uniform(.1,.9,12).astype(np.float32) for u in ids}
    teacher={u:rng.uniform(.01,.99,12).astype(np.float32) for u in ids[2:]}
    train,val=t.a3.selected_rosters(c,'fivefold',2)
    mask=np.ones((len(train),12),dtype=bool)
    s=t.materialize(tmp_path,c,l2,teacher,None,arm,2,'a'*64,mask,engineering=True)
    _,loaded=t.load_materialized(tmp_path)
    for u in ids[:2]:np.testing.assert_array_equal(loaded[u],l2[u])
    available=[u for u in train if u in teacher]
    scaled,_=t.pl.scaled_teacher(l2,teacher,available,available)
    expected=t.pl.arm_targets(t.pl.aligned(l2,available),scaled,np.ones((len(available),12),dtype=bool))[arm]
    np.testing.assert_array_equal(t.pl.aligned(loaded,available),expected)
    assert s['scale_fit_ids']==available
    np.testing.assert_array_equal(np.array(s['weights'],np.float32),t.pl.frozen_weights(l2,train))

def test_unregistered_missing_teacher_refused():
    c,ids=cohort()
    rows=pd.DataFrame({t.pl.UID:ids,'component_id':ids,'fold':[str(i%5) for i in range(50)]})
    with pytest.raises(ValueError,match='unregistered missing'):
        t.build_cohort(rows,[],ids[3:],ids[:2])

def test_powershell_desktop_python_gate_and_env_cleanup(tmp_path):
    import subprocess
    script=Path('.local/run-scripts/pl-production-v1.ps1').read_text()
    function=script[script.index('function Assert-PythonFree'):script.index('Push-Location')]
    probe=tmp_path/'probe.ps1'
    probe.write_text(function+'''\nfunction nvidia-smi { $global:LASTEXITCODE=0; '12, C:\\Windows\\explorer.exe'; '13, C:\\chrome.exe' }
Assert-PythonFree
function nvidia-smi { $global:LASTEXITCODE=0; '14, C:\\Python312\\python.exe' }
try { Assert-PythonFree; exit 2 } catch { if ($_.Exception.Message -notmatch 'python ML') { throw } }
''')
    result=subprocess.run(['powershell','-NoProfile','-File',str(probe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert "Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue" in script
