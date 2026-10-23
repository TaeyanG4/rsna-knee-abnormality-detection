"""Synthetic admission adversaries; never reads real labels or fit directories."""
import copy
import json
from pathlib import Path
import pytest
from tools import hb150_build_v1 as b


def test_no_review_cannot_launch(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'validate',lambda p: {'arm':'global'})
    with pytest.raises(ValueError,match='independent review'):
        b.require_fit(tmp_path/'config.json',None)


@pytest.mark.parametrize('patch',[
    {'verdict':'HOLD'},{'independent':False},{'reviewer':''},
    {'exposure':'clean'},{'clean_oof':True},{'schema':'old'}])
def test_review_bypass_rejected(monkeypatch,tmp_path,patch):
    monkeypatch.setattr(b,'validate',lambda p: {'arm':'global'})
    value=dict(schema='hb150-independent-review/v1',verdict='PASS',independent=True,
               reviewer='different-reviewer',exposure='EXPOSED',clean_oof=False)
    value.update(patch); path=tmp_path/'review.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='independent PASS'):
        b.require_fit(tmp_path/'config.json',path)


def test_hash_bound_review_not_transferable(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'validate',lambda p:{'arm':'global'})
    monkeypatch.setattr(b,'review_bindings',lambda *a:{'smoke':'new'})
    value=dict(schema='hb150-independent-review/v1',verdict='PASS',independent=True,
               reviewer='different-reviewer',exposure='EXPOSED',clean_oof=False,
               baseline_attestation='baseline.json',bindings={'smoke':'old'})
    path=tmp_path/'review.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='exact CPU/CUDA'):
        b.require_fit(tmp_path/'config.json',path)


def test_pair_fail_closed_before_dispatch(monkeypatch):
    monkeypatch.setattr(b,'admit',lambda *a:{'status':'HOLD'})
    monkeypatch.setattr(b.subprocess,'run',lambda *a,**k:pytest.fail('dispatch forbidden'))
    with pytest.raises(ValueError,match='admission HOLD'):
        b.pair('global',None,'fit-v1')


def test_immutable_registration_rejects_change(tmp_path):
    p=tmp_path/'registration.json';b.immutable(p,{'science':1})
    b.immutable(p,{'science':1})
    with pytest.raises(ValueError,match='immutable'):
        b.immutable(p,{'science':2})


def test_missing_pl_is_explicit_hold(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'ROOT',tmp_path)
    monkeypatch.setattr(b,'validate',lambda p:{'pl_work':'pl'})
    result=b.prepare('fill')
    assert result['status']=='HOLD' and len(result['missing'])==3


@pytest.mark.parametrize('arm',['global','fill'])
def test_prepare_authenticates_synthetic_pl_metadata(monkeypatch,arm):
    c={'pl_work':'synthetic-pl','weights':[1.]*12}
    w=b.ROOT/c['pl_work']; writes=[]; validated=[]; hashed=[]
    exposed={'status':'EXPOSED_EXPLORATORY','exposed_exploratory':True}
    provenance=dict(exposed,version='1.1.1',mode='screen_f0',alpha=.5,
                    engineering_only=False,weights=c['weights'],arm_hashes={arm:'hash-'+arm+'.csv'},
                    teacher=dict(exposed,model_selection='f0_only',output_kind='pre_stacker',
                                 complete=True,failures=0,clean_oof=False))
    prep=dict(exposed,engineering_only=False,epochs=20,batch=8,seeds=[42,43],weights=c['weights'],
              pins={b.rel(p):'hash-'+p.name for p in
                    (w/'registration.json',w/'screen_f0/provenance.json',w/'screen_f0'/f'{arm}.csv')})
    def validate(path):
        validated.append(Path(path).name)
        return c
    def read(path):
        return {'prepare.json':prep,'provenance.json':provenance}[Path(path).name]
    def sha(path):
        hashed.append(Path(path).name)
        return 'hash-'+Path(path).name
    monkeypatch.setattr(b,'validate',validate)
    monkeypatch.setattr(Path,'is_file',lambda path:True)
    monkeypatch.setattr(b,'read',read)
    monkeypatch.setattr(b,'sha',sha)
    monkeypatch.setattr(b,'immutable',lambda p,v:writes.append((p,v)))
    result=b.prepare(arm)
    assert result['status']=='PREPARED'
    assert validated==[f'{arm}-s42.json',f'{arm}-s43.json']
    assert f'{arm}.csv' in hashed  # hash only; read() rejects any label-table access
    assert len(writes)==1 and writes[0][0].name==f'prepared-{arm}.json'
    value=writes[0][1]
    assert value['exposure']=='EXPOSED' and value['clean_oof'] is False
    assert value['configs']=={str(s):f'hash-{arm}-s{s}.json' for s in (42,43)}


@pytest.mark.parametrize('arm',['global','fill'])
def test_require_fit_accepts_fully_bound_synthetic_review(monkeypatch,arm):
    from tools import hb150_readout_v1 as readout
    c={'arm':arm,'pl_work':'synthetic-pl'}; opened=[]; checked=[]
    bindings={'synthetic-smoke':'hash'}
    review=dict(schema='hb150-independent-review/v1',verdict='PASS',independent=True,
                reviewer='independent-reviewer',exposure='EXPOSED',clean_oof=False,
                baseline_attestation='synthetic-attestation.json',bindings=bindings)
    decision=dict(version='1.1.1',engineering_only=False,status='EXPOSED_EXPLORATORY',
                  registered_go={'GO':True,'selected_arm':arm})
    def read(path):
        path=Path(path); opened.append(path)
        if path.name=='synthetic-review.json':return review
        if path.name=='readout.json':return decision
        assert path.name=='summary.json'
        assert path.parent.name in (f'{arm}-s42',f'{arm}-s43')
        summary=dict(config_sha256='hash-'+path.parent.name+'.json',engineering_only=True,status='PASS')
        if path.parent.parent.name=='cpu':return dict(summary,cpu_cuda_initialized=False)
        assert path.parent.parent.name=='cuda'
        return dict(summary,device='cuda',gpu_name=b.GPU,full_shape=True,peak_reserved_bytes=1<<30,
                    optimizer_steps=2,batch=8,val_batch=4,validation_shapes=['k24','k36','k36flip'],
                    healthy_rates={'train_step':1.,'validation':2.})
    def authenticate(config,path):
        checked.append((config,path))
        return {'42':{},'43':{}}
    monkeypatch.setattr(b,'validate',lambda p:c)
    monkeypatch.setattr(Path,'is_file',lambda path:True)
    monkeypatch.setattr(b,'read',read)
    monkeypatch.setattr(b,'sha',lambda path:'hash-'+Path(path).name)
    monkeypatch.setattr(b,'review_bindings',lambda *args:bindings)
    monkeypatch.setattr(readout,'validate_baselines',authenticate)
    assert b.require_fit(Path('synthetic-config.json'),Path('synthetic-review.json'))==c
    assert checked==[(c,b.ROOT/'synthetic-attestation.json')]
    assert len([p for p in opened if p.name=='summary.json'])==4
