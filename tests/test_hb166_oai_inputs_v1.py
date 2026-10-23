"""CPU-only OAI adapter identity, geometry, resume and registered L7 tests."""
import copy
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools import hb166_oai_inputs_v1 as m


@pytest.fixture
def synthetic(monkeypatch):
    slots = [['Sagittal',1,18],['Sagittal',0,14],['Coronal',1,12],['Coronal',0,8],['Axial',-1,12]]
    spec = dict(slots=slots,img=448)
    ids = ['OAI_100_L_V00','OAI_200_R_V00']
    rows, plans, groups = {}, {}, {}
    orientations = {'Sagittal':[0,1,0,0,0,1], 'Coronal':[1,0,0,0,0,1], 'Axial':[1,0,0,0,1,0]}
    for uid in ids:
        plans[uid] = []
        for j,(plane,fluid,_) in enumerate(slots):
            key=f'{uid}-{j}'
            rows[key]=dict(study_key=uid,knee_side=uid.split('_')[2],plane=plane,
                fluid_sensitive=str(fluid),series_key=key,rule_hit='known contrast synthetic')
            plans[uid].append(dict(group_key=key,plane=plane,fluid=fluid,fallback='none'))
            groups[key]=dict(source_sequence='synthetic',contrast_family='synthetic')
    def load(row,*args):
        yy,xx=np.indices((64,64)); v=np.stack([xx*3+yy*5+z*7 for z in range(8)]).astype(np.uint16)
        iop=np.array(orientations[row['plane']],np.float32)
        normal=np.cross(iop[:3],iop[3:]); ipp=np.arange(8)[:,None]*normal*2
        return m.common.Series(row,v,ipp,iop,np.array([2.,2.]),2.,np.full(8,2.))
    monkeypatch.setattr(m.common,'load_series',load)
    return dict(rows=rows,plans=plans,audit=SimpleNamespace(groups=groups)),spec,ids


def test_original_geometry_and_side_order(synthetic):
    ctx,spec,ids=synthetic
    left,lm,lq=m.render(ctx,ids[0],spec)
    right,rm,rq=m.render(ctx,ids[1],spec)
    np.testing.assert_array_equal(right,left[m.side.permutation(spec,'R')])
    np.testing.assert_array_equal(rm,lm[m.side.permutation(spec,'R')])
    assert left.shape==(64,448,448) and left.dtype==np.uint8 and lm.all()
    for q in (lq,rq):
        assert not q['side_resolution']['conflict'] and q['missing_series']==0
        for slot in q['slots'][:4]:
            assert slot['crops'][0]['compartment']=='medial'
            assert slot['crops'][-1]['compartment']=='lateral'
            assert slot['source_sequence']=='synthetic'


def test_general_side_disagreement_rejected(synthetic):
    ctx,_,ids=synthetic
    rows=m.oai.plan_rows(ctx['plans'][ids[0]],ctx['rows'])
    rows[0]['knee_side']='R'
    with pytest.raises(ValueError,match='laterality/study'):
        m.GeneralAdapter(rows,ids[0])


def test_wrong_series_pick_rejected_even_when_availability_equal(synthetic,monkeypatch):
    ctx,spec,ids=synthetic; original=m.crop.build_study
    def wrong(*a,**kw):
        v,mask,q=original(*a,**kw); q['slots'][0]['series']='wrong'; return v,mask,q
    monkeypatch.setattr(m.crop,'build_study',wrong)
    with pytest.raises(RuntimeError,match='pick differs'):
        m.render(ctx,ids[0],spec)


def test_authenticated_copy_alias_reuses_pixels(synthetic):
    ctx,_,ids=synthetic
    plan=copy.deepcopy(ctx['plans'][ids[0]])
    plan[1]['group_key']=plan[0]['group_key']; plan[1]['fallback']=m.oai.COPY
    rows=m.oai.plan_rows(plan,ctx['rows']); adapter=m.GeneralAdapter(rows,ids[0])
    assert rows[1]['series_key']==rows[0]['series_key']+'#sag0_fallback_copy'
    np.testing.assert_array_equal(adapter._read_pixels(rows[0]['series_key']+'/000001',None),
                                  adapter._read_pixels(rows[1]['series_key']+'/000001',None))


def test_review_gate_missing_wrong_binding_and_report_tamper(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'WORK',tmp_path); reg={'version':'test'}
    with pytest.raises(ValueError,match='missing'): m.check_review(reg)
    report=tmp_path/'PRE-REVIEW.md'; report.write_text('Independent pass')
    receipt=dict(verdict='PASS',reviewer='independent',registration_sha256=m.digest(reg),
        scope='CPU crop prepare only; no GPU/admit',report=report.name,report_sha256=m.sha(report))
    m.io.atomic_json(tmp_path/'pre-review.json',receipt); m.check_review(reg)
    with pytest.raises(ValueError,match='binding'): m.check_review({'version':'drift'})
    report.write_text('tampered')
    with pytest.raises(ValueError,match='report drift'): m.check_review(reg)


@pytest.fixture
def prepared(synthetic,tmp_path,monkeypatch):
    ctx,spec,ids=synthetic
    monkeypatch.setattr(m,'WORK',tmp_path); monkeypatch.setattr(m,'OUT',tmp_path/'crop448')
    monkeypatch.setattr(m.native,'panel_ids',lambda:ids)
    monkeypatch.setattr(m.native,'context',lambda *a:ctx)
    monkeypatch.setattr(m,'register',lambda:dict(spec=spec))
    # Native is read-only in this test; supply an independent manifest fixture.
    native=tmp_path/'native'; native.mkdir(); m.io.atomic_json(native/'manifest.json',{})
    monkeypatch.setattr(m.native,'OUT',native)
    m.prepare(smoke=True)
    return tmp_path/'smoke-crop448'


def test_cache_roundtrip_resume_immutable(prepared,monkeypatch):
    assert m.validate(smoke=True,verify_native=False)['studies']==2
    before=m.sha(prepared/'manifest.json')
    monkeypatch.setattr(m,'render',lambda *a:pytest.fail('finished cache rerender'))
    assert m.prepare(smoke=True)['status']=='REUSED_FINISHED'
    assert m.sha(prepared/'manifest.json')==before


@pytest.mark.parametrize('kind',['pixels','mask','side','bounds','contract','qa'])
def test_corruption_fails_closed(prepared,kind):
    manifest=m.read(prepared/'manifest.json')
    if kind=='pixels':
        with (prepared/manifest['shards'][0]['file']).open('r+b') as stream:
            stream.seek(-1,2); value=stream.read(1)[0]
            stream.seek(-1,2); stream.write(bytes([value ^ 1]))
    elif kind in ('mask','side'):
        name='masks' if kind=='mask' else 'side_unknown'
        a=np.load(prepared/f'{name}.npy'); a[:]=0 if kind=='mask' else 1
        m.native.atomic_array(prepared/f'{name}.npy',a)
        manifest[name+'_sha256']=m.sha(prepared/f'{name}.npy')
    elif kind=='bounds': manifest['shards'][0]['start']=1
    elif kind=='contract': manifest['contract']='wrong'
    else:
        p=prepared/manifest['shards'][0]['qa_file']; q=m.read(p); q['laterality']='R'
        m.io.atomic_json(p,q); manifest['shards'][0]['qa_sha256']=m.sha(p)
    m.io.atomic_json(prepared/'manifest.json',manifest)
    with pytest.raises(ValueError): m.validate(smoke=True,verify_native=False)


def test_l7_seed_probability_mean_nan_mask_threshold_and_eligibility():
    from tools import lens_completion_1015_v1 as lens
    ids=[str(i) for i in range(40)]
    y=np.tile(np.arange(40)[:,None]%2,(1,12)).astype(float); y[:,[1,10,11]]=np.nan
    good=np.tile((np.arange(40)%2)[:,None],(1,12)).astype(np.float32)
    half=np.full((40,12),.5,np.float32)
    a={'43':good,'42':half}; c={'43':half,'42':half}
    item=lens.image_item(ids,a,c,y,False,'test')
    np.testing.assert_array_equal(item['arm'],(good+half)*np.float32(.5))
    assert not item['mask'][:,[1,10,11]].any()
    r=m.l7(ids,a,c,y,eligible=False)
    assert r['weak4']['value']==.5 and r['weak4']['threshold']==.002
    assert r['weak4']['PASS'] and not r['weak4']['eligible']
    r=m.l7(ids,c,c,y,eligible=True); assert not r['weak4']['PASS']
    # Strict mean must not silently drop an entirely unknown weak target.
    y[:,3]=np.nan
    assert m.l7(ids,a,c,y,eligible=True)['weak4']['value'] is None


def test_no_admit_cli():
    r=subprocess.run([sys.executable,str(m.ROOT/'tools/hb166_oai_inputs_v1.py'),'admit'],
        capture_output=True,text=True)
    assert r.returncode==2 and 'invalid choice' in r.stderr
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
