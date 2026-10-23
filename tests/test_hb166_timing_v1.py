"""CPU negative contracts for the unexecuted HB166 N32 diagnostic."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
from pathlib import Path
import numpy as np
import pytest
from tools import hb166_timing_v1 as t

IDS=[str(i) for i in range(32)]
IDENTITIES={'oldQ':{'model':'old'},'HB166':{'model':'new'}}

def measurement(arm):
    return dict(tag=t.TAG,arm=arm,done=32,state='COMPLETE',clean_oof=False,
        targets=t.s.TARGETS,arm_identity=IDENTITIES[arm],
        values=np.full((32,12),.4).tolist(),ids=IDS,seconds=64.,fallback=0,
        peak_reserved_bytes=4*2**30,desktop=dict(device_name='4070 Ti SUPER',total_mib=16376,used_mib=1500),
        telemetry=dict(complete=True),runtime_sha256='same')

def test_abba_projection_and_negative_delta():
    records=[measurement(a) for a in t.ORDER]
    records[1]['seconds']=records[2]['seconds']=32
    assert t.summarize(records,IDS,IDENTITIES)['local_inclusive_delta_seconds']==-32
    assert t.projection(-1)['runtime']=='HOLD'
    assert t.projection(-80)['qualified_for_submission'] is False
    assert 'projected_seconds' not in t.projection(-80)
    with pytest.raises(TypeError): t.projection(-80,1300,30000,1,True)
    records.reverse()  # ABBA is palindromic; swap distinct arms instead.
    records[0],records[1]=records[1],records[0]
    with pytest.raises(ValueError,match='order'): t.summarize(records,IDS,IDENTITIES)

@pytest.mark.parametrize('key,value',[
    ('fallback',1),('seconds',float('nan')),('peak_reserved_bytes',13*2**30),
    ('telemetry',dict(complete=False)),('done',31),('clean_oof',True),
    ('ids',IDS[::-1]),('values',np.full((32,12),float('nan')).tolist()),
    ('targets',t.s.TARGETS[::-1]),('arm_identity',{'model':'wrong'}),
    ('values',np.zeros((31,12)).tolist()),
    ('desktop',dict(device_name='4070 Ti SUPER',total_mib=16376,used_mib=12000))])
def test_bad_real_result_contract(key,value):
    record=measurement('oldQ'); record[key]=value
    with pytest.raises(ValueError): t.validate_measurement(record,IDS,IDENTITIES)

def test_runtime_and_stack_drift(tmp_path):
    (tmp_path/'a.py').write_bytes(b'ok')
    t.authenticate_runtime(tmp_path,{'a.py':b'ok'})
    with pytest.raises(ValueError,match='runtime drift'):
        t.authenticate_runtime(tmp_path,{'a.py':b'wrong'})
    records=[measurement(a) for a in t.ORDER]; records[2]['runtime_sha256']='changed'
    with pytest.raises(ValueError,match='stack drift'): t.summarize(records,IDS,IDENTITIES)

def test_no_admission_bypass(monkeypatch):
    monkeypatch.setattr(t,'registered',lambda:dict(manifest_sha256=t.s.sha(t.b.OUT/'assets/manifest.json')))
    with pytest.raises(ValueError,match='required fold missing'): t.gpu_admission()
    monkeypatch.delenv('HB166_CONTROLLER_PID',raising=False)
    with pytest.raises(ValueError,match='supervised'): t.worker(0)

def test_exact_parent_blocks_and_oldq_orchestration(monkeypatch,tmp_path):
    prep,infer=t.oldq_blocks()
    raw=t.b.previous.snapshots()['parent']['files']['exp169_partner_bundle_release.py'].decode()
    import textwrap
    assert textwrap.indent(prep,' '*16) in raw and textwrap.indent(infer,' '*16) in raw
    compile(prep,'prep','exec'); compile(infer,'infer','exec')
    assert 'plane_flags' in prep and '_infer_one' in infer
    assert "family == 'dino'" in infer

def test_receipt_signature_and_nonfatal_progress(monkeypatch,tmp_path):
    # Shared helper intentionally accepts two args; exercise the real function.
    assert t.io.receipt(tmp_path/'phase.json',dict(state='LOAD',done=0))
    monkeypatch.setattr(t.io,'atomic_json',lambda *a: (_ for _ in ()).throw(PermissionError()))
    assert t.io.receipt(tmp_path/'phase.json',dict(state='LOAD',done=0)) is False


def test_real_journal_envelope_can_be_republished(monkeypatch,tmp_path):
    monkeypatch.setattr(t,'WORK',tmp_path)
    t.write('worker.json',{'state':'COMPLETE'})
    value=t.io.read_json(tmp_path/'worker.json')
    t.write('measurement.json',value)
    assert t.io.read_json(tmp_path/'measurement.json')==value
    with pytest.raises(ValueError,match='identity drift'):
        t.write('wrong.json',dict(identity='wrong'))
