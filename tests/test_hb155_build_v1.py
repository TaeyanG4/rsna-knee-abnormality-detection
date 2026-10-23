import copy
import json
from pathlib import Path
import pytest
from tools import hb155_build_v1 as b
from tools import hb155_transfer_v1 as t

def test_recipe_unchanged_except_declared_roots(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'WORK',tmp_path)
    b.decision_binding(b.DECISION_ID)
    c=b.expected('crop',42)
    parent=b.read(b.PILOT/'native224-s42.json')['config']
    actual=copy.deepcopy(c['recipe'])
    actual['roots']=parent['roots']; actual['windows']['native_source']=parent['windows']['native_source']
    for k in ('loader_batches','steps','max_skips'): actual['updates'][k]=parent['updates'][k]
    assert actual==parent
    assert (c['batch'],c['microbatch'],c['accumulation'],c['epochs'])==(8,2,4,20)
    assert c['holdout_status']=='EXPOSED_PAIRED_FOLD2_AUTHORIZED' and not c['clean_oof']
    assert c['exposure']=='EXPOSED' and c['coordinator_decision']==b.COORDINATOR_DECISION
    assert not set(c['train']) & set(c['val'])

def test_exposed_fold_cannot_be_waived(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'WORK',tmp_path)
    b.decision_binding(b.DECISION_ID)
    monkeypatch.setattr(b,'validate',lambda p:dict(engineering_only=False,holdout_status='BLOCKED'))
    with pytest.raises(ValueError,match='D-B-157'): b.require_fit('x','fake-PASS.json')
    assert b.admit('fake-PASS.json')['status']=='HOLD'

def test_decision_missing_unknown_or_fabricated_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'WORK',tmp_path)
    with pytest.raises(ValueError,match='missing'): b.decision_binding()
    with pytest.raises(ValueError,match='unknown'): b.decision_binding('reviewer-PASS')
    p=tmp_path/'coordinator-decision.json'
    p.write_text(json.dumps(dict(b.COORDINATOR_DECISION,clean_oof=True)))
    with pytest.raises(ValueError,match='differs'): b.decision_binding()
    with pytest.raises(ValueError,match='immutable'): b.decision_binding(b.DECISION_ID)

def authorized_config(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'WORK',tmp_path)
    return dict(engineering_only=False,holdout_status='EXPOSED_PAIRED_FOLD2_AUTHORIZED',fold=2,
        exposure='EXPOSED',clean_oof=False,coordinator_decision=b.COORDINATOR_DECISION,
        estimand=b.COORDINATOR_DECISION['estimand'],selection_exposure='identical for both paired arms',
        inputs=b.decision_binding(b.DECISION_ID))

@pytest.mark.parametrize('field,value',[('fold',1),('clean_oof',True),('exposure','CLEAN'),
    ('selection_exposure','different across arms'),('estimand','unexposed absolute AUC'),
    ('coordinator_decision',{}),('inputs',{})])
def test_decision_semantics_cannot_be_changed(tmp_path,monkeypatch,field,value):
    c=authorized_config(tmp_path,monkeypatch); c[field]=value
    monkeypatch.setattr(b,'validate',lambda p:c)
    with pytest.raises(ValueError,match='explicit registered'): b.require_fit('x','fake-PASS.json')

def test_authorized_pair_still_needs_bound_independent_review(tmp_path,monkeypatch):
    c=authorized_config(tmp_path,monkeypatch)
    monkeypatch.setattr(b,'validate',lambda p:c)
    with pytest.raises(ValueError,match='independent review required'): b.require_fit('x')
    path=tmp_path/'review.json'
    r=dict(schema='hb155-review/v1',verdict='PASS',independent=True,reviewer='R',bindings={'proof':'stale'})
    path.write_text(json.dumps(r)); monkeypatch.setattr(b,'review_bindings',lambda:{'proof':'current'})
    with pytest.raises(ValueError,match='smoke-bound'): b.require_fit('x',path)
    r['bindings']={'proof':'current'}; path.write_text(json.dumps(r))
    assert b.require_fit('x',path)==c
    assert b.admit(path)['exposure']=='EXPOSED'

def test_authorized_pair_propagates_remaining_gate_failure(tmp_path,monkeypatch):
    c=authorized_config(tmp_path,monkeypatch)
    monkeypatch.setattr(b,'validate',lambda p:c)
    path=tmp_path/'review.json'
    path.write_text(json.dumps(dict(schema='hb155-review/v1',verdict='PASS',independent=True,reviewer='B',bindings={})))
    def blocked(): raise ValueError('full paired caches missing')
    monkeypatch.setattr(b,'review_bindings',blocked)
    assert b.admit(path)==dict(status='HOLD',reason='full paired caches missing')

def valid_crop_review(tmp_path,monkeypatch):
    from tools import hb155_exclusions_v1 as e
    monkeypatch.setattr(b,'ROOT',tmp_path);monkeypatch.setattr(b,'WORK',tmp_path/'work');monkeypatch.setattr(b,'CROP','crop');monkeypatch.setattr(b,'CONTROL','control')
    for d in ('crop','control','work'):(tmp_path/d).mkdir()
    monkeypatch.setattr(e,'OUT',b.WORK);monkeypatch.setattr(e,'RECORD',b.WORK/'exclusions.json');monkeypatch.setattr(e,'QA',b.WORK/'qa.json')
    monkeypatch.setattr(e,'load',lambda:None);monkeypatch.setattr(e,'verify_fold',lambda:None)
    for p in (e.RECORD,b.WORK/'fold-binding.json',tmp_path/'crop/manifest.json',tmp_path/'control/manifest.json'):p.write_text('{}')
    (tmp_path/'crop/qa-summary.json').write_text('{"thresholds":{"unknown_laterality":0.02}}')
    overlays={}
    for i in range(24):
        p=tmp_path/'crop'/f'overlay-{i}.png';p.write_bytes(bytes([i]));overlays[p.name]=b.sha(p)
    q=dict(studies=4408,excluded_studies=2,verdict='PASS',qa_pass=True,thresholds={'unknown_laterality':0.02},overlay_count=24,clean_oof=False,exposure='EXPOSED',
      pixel_hash_verification='all_chunks_both_arms',overlay_sha256=overlays,exclusions_sha256=b.sha(e.RECORD),fold_binding_sha256=b.sha(b.WORK/'fold-binding.json'),
      manifest_sha256=b.sha(tmp_path/'crop/manifest.json'),control_manifest_sha256=b.sha(tmp_path/'control/manifest.json'))
    e.QA.write_text(json.dumps(q))
    paths=(e.QA,e.RECORD,b.WORK/'fold-binding.json',tmp_path/'crop/manifest.json',tmp_path/'control/manifest.json')
    r=dict(schema='hb155-crop-review/v1',verdict='PASS',independent=True,reviewer='B',overlays_reviewed=24,bindings={b.rel(p):b.sha(p) for p in paths})
    review=b.WORK/'crop-review.json';review.write_text(json.dumps(r));return q,r,e.QA,review

@pytest.mark.parametrize('key,value',[('studies',3),('qa_pass',False),('thresholds',{}),
    ('overlay_count',23),('excluded_studies',8),('clean_oof',True),('exposure','CLEAN')])
def test_crop_qa_thresholds_and_full_random_review_required(tmp_path,monkeypatch,key,value):
    q,r,qa,review=valid_crop_review(tmp_path,monkeypatch)
    q[key]=value; qa.write_text(json.dumps(q))
    with pytest.raises(ValueError,match='full-cache crop QA'): b.crop_review_bindings()

@pytest.mark.parametrize('key,value',[('independent',False),('reviewer','builder'),
    ('overlays_reviewed',3),('bindings',{})])
def test_b_overlay_review_cannot_be_bypassed(tmp_path,monkeypatch,key,value):
    q,r,qa,review=valid_crop_review(tmp_path,monkeypatch)
    r[key]=value; review.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='independent B review'): b.crop_review_bindings()

def test_crop_review_binds_manifest_and_qa(tmp_path,monkeypatch):
    q,r,qa,review=valid_crop_review(tmp_path,monkeypatch)
    assert len(b.crop_review_bindings())==6
    (tmp_path/'crop/manifest.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='stale'): b.crop_review_bindings()

def test_engineering_cannot_fit(monkeypatch):
    monkeypatch.setattr(b,'validate',lambda p:dict(engineering_only=True))
    with pytest.raises(ValueError,match='engineering'): b.require_fit('x','fake-PASS.json')

def test_standalone_qa_and_overlay_bytes_required(tmp_path,monkeypatch):
    q,r,qa,review=valid_crop_review(tmp_path,monkeypatch)
    q['pixel_hash_verification']='build_or_resume_verified'; qa.write_text(json.dumps(q))
    with pytest.raises(ValueError,match='full-cache crop QA'): b.crop_review_bindings()
    q['pixel_hash_verification']='all_chunks_both_arms'; qa.write_text(json.dumps(q))
    (tmp_path/'crop/overlay-0.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='overlay absent/stale'): b.crop_review_bindings()

def test_runtime_review_cannot_be_bypassed(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'WORK',tmp_path); monkeypatch.setattr(b,'validate',lambda p:{})
    monkeypatch.setattr(b,'runtime_bindings',lambda:{'snapshot':'a'})
    (tmp_path/'runtime-review.json').write_text(json.dumps(dict(schema='hb155-runtime-review/v1',verdict='PASS',independent=True,reviewer='R',bindings={'snapshot':'b'})))
    with pytest.raises(ValueError,match='stale'): b.require_runtime('x',{})

def test_disk_and_path_traversal(monkeypatch,tmp_path):
    monkeypatch.setattr(t.shutil,'disk_usage',lambda p:type('Disk',(),{'free':100_000_000_000})())
    with pytest.raises(ValueError,match='disk'): t.disk_gate(tmp_path)
    assert t.disk_gate(tmp_path,100)['status']=='DISK_OK'
    with pytest.raises(ValueError,match='unsafe'): t.safe_path('../outside',tmp_path)

def test_transfer_verification_and_tamper(tmp_path):
    p=tmp_path/'data.bin'; p.write_bytes(b'example')
    manifest=tmp_path/'inventory.json'
    doc=dict(status='READY',files={'data.bin':dict(bytes=7,sha256=b.sha(p))})
    manifest.write_text(json.dumps(doc)); assert t.verify(manifest,tmp_path)['files']==1
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'): t.verify(manifest,tmp_path)
    doc['status']='DRAFT_NOT_READY'; manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='draft'): t.verify(manifest,tmp_path)

def test_results_refuses_running_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'WORK',tmp_path)
    d=tmp_path/'fits/control-s42'; d.mkdir(parents=True)
    (d/'heartbeat.json').write_text('{"status":"running"}')
    with pytest.raises(ValueError,match='incomplete'): t.results()

def test_closure_has_training_imports():
    paths=b.source_closure()
    for p in ('training/src/raptor_common.py','training/arms/lifecycle.py','tools/job_runtime.py'):
        assert p in paths
    assert all('..' not in Path(p).parts for p in paths)
