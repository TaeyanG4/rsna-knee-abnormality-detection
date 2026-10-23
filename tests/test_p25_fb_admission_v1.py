"""P4 v1.0.2 exact-stage offline admission integration; synthetic evidence only."""
import json
import subprocess
import pytest
from test_k1_admit_v1 import case, put, change, transport
from test_p25_fb_v1 import evidence
from tools import p25_fb_admit_v1 as a
from tools import p25_fb_stage_v1 as b


@pytest.fixture
def exact_case(case):
    case.update(stage=b.stage(),kernel=b.SLUG)
    for diagnostic, count in ((False,3),(True,32)):
        d = case['visible_dir'] if not diagnostic else case['visible_dir'].parent/'n32'
        _,cfg,k,p = evidence(count,diagnostic)
        if not diagnostic:
            for r, uid in zip(k['per_study'],['a','b','c']): r['uid']=uid
            payload=(d/'submission.csv').read_bytes()
            k.update(parent_sha256=a.base.sha(payload),submission_sha256=a.base.sha(payload))
            p['submission_sha256']=a.base.sha(payload)
        k.update(parent_seconds=1000.,phases=dict(
            classifier_seconds=count*8.,load_seconds=count*4.,extra_render_seconds=count*.125))
        p.update(status='DIAGNOSTIC_C85R7' if diagnostic else 'COMPLETE_C85R7',
                 version=cfg['release_version'],studies=count,findings=12,
                 diagnostic=diagnostic,never_submit=diagnostic,
                 fallbacks=dict(r_prediction=0,s_r7_prediction=0,r224_prediction=0),
                 codecs=dict(status='ACTIVE',s_r7_gate='ACTIVE'))
        put(d/'k1sr-receipt.json',k); put(d/'c85r7-receipt.json',p)
        if not diagnostic:
            transport(d,kernel=b.SLUG)
        else:
            case['n32_dir']=d
            for name in ('k1sr-receipt.json','c85r7-receipt.json'):
                put(d/(name+'.download.json'),dict(status='DOWNLOADED',kernel=cfg['kernel_id'],
                    version=1,file=name,local_name=name,sha256=a.base.sha((d/name).read_bytes())))
            # Real notebook layout: no projection/startup/selection/train inputs.
            parent_csv=('id,'+','.join('target'+str(i) for i in range(12))+'\n'
                + ''.join(str(i)+','+','.join(['0.5']*12)+'\n' for i in range(32)))
            (d/'k1sr-parent.csv').write_text(parent_csv)
    return case


def test_exact_stage_real_offline_prepare(exact_case,monkeypatch,capsys):
    calls=[]; run=subprocess.run
    def guarded(args,**kwargs):
        assert '--check-only' in args and not kwargs.get('shell',False)
        assert str(a.base.ROOT/'tools/kaggle_submit_once.py') in args
        calls.append(args)
        return run(args,**kwargs)
    monkeypatch.setattr(a.base.subprocess,'run',guarded)
    pinned=b.util.sha(b.stage()/'kernel/release.ipynb')
    out=a.prepare(**exact_case)
    assert 'ADMITTED_OFFLINE' in capsys.readouterr().out and len(calls)==1
    sidecar=json.loads((out/'p4-four-member-admission.json').read_bytes())
    assert sidecar['status']=='PASS_OFFLINE_P4' and not sidecar['independent_review']
    assert sidecar['projection'] is None
    assert not (exact_case['n32_dir']/'projection-inputs.json').exists()
    assert any("runtime is judged by B's 8-step gate" in line for line in sidecar['checks'])
    assert 'NOT PROVIDED' not in '\n'.join(sidecar['checks'])
    assert sidecar['extension_sha256']==a.base.sha(__import__('pathlib').Path(a.__file__).read_bytes())
    reg=json.loads((out/'ordinary-visible-registration.json').read_bytes())
    assert reg['projection'] is None
    assert str((exact_case['n32_dir']/'k1sr-receipt.json').resolve()) in reg['input_sha256']
    assert "runtime is judged by B's 8-step gate" in (out/'READY-REVIEW.md').read_text()
    assert reg['kernel_id']==b.SLUG and reg['version']==b.RELEASE
    assert reg['stage_registration_sha256']==b.util.sha(b.stage()/'registration.json')
    assert pinned==b.util.sha(b.stage()/'kernel/release.ipynb')


@pytest.mark.parametrize('fault',['three_members','wrong_hash','flat_weights','O_present','wrong_roster'])
def test_exact_stage_refuses_mutated_evidence(exact_case,fault):
    d=exact_case['visible_dir']
    def mutate(k):
        if fault=='three_members': k['per_study'][0]['predictions'].pop(b.FB_ID)
        elif fault=='wrong_hash': k['model_identities'][-1]['checkpoint_sha256']='0'*64
        elif fault=='flat_weights': k['p_weight_mode']='flat'
        elif fault=='wrong_roster': k['per_study'][0]['uid']='absent'
    if fault=='O_present':
        change(d/'c85r7-receipt.json',lambda p:p['executed_members'].append('EXP155'))
    else: change(d/'k1sr-receipt.json',mutate)
    transport(d,kernel=b.SLUG)
    with pytest.raises(a.base.Refuse): a.prepare(**exact_case)
    assert not (d/a.base.OUTPUT).exists()


def refresh_n32(d):
    for name in ('k1sr-receipt.json','c85r7-receipt.json'):
        change(d/(name+'.download.json'),lambda r:r.update(
            sha256=a.base.sha((d/name).read_bytes())))


@pytest.mark.parametrize('failed',[0,1,2])
def test_n32_failure_boundary(exact_case,failed):
    d=exact_case['n32_dir']
    def mutate(k):
        for row in k['per_study'][:failed]: row['ok']=False
        k.update(failed_study_count=failed,failed_study_fraction=failed/32,k1_studies=32-failed)
    change(d/'k1sr-receipt.json',mutate);refresh_n32(d)
    if failed==0:
        assert a.validate(**exact_case)[3] is None
    else:
        with pytest.raises(a.base.Refuse): a.validate(**exact_case)


@pytest.mark.parametrize('fault',[
    'three_members','wrong_hash','flat_weights','O_present','A5_present','stride',
    'status','not_executed','formula','contract','r224','model_hash','missing_row',
    'duplicate_uid','provenance','failed_fraction'])
def test_n32_refuses_mutated_evidence(exact_case,fault):
    d=exact_case['n32_dir']
    def mutate(k):
        if fault=='three_members': k['per_study'][0]['predictions'].pop(b.FB_ID)
        elif fault=='wrong_hash': k['model_identities'][-1]['checkpoint_sha256']='0'*64
        elif fault=='flat_weights': k['p_weight_mode']='flat'
        elif fault=='A5_present': k['a5_contract']={}
        elif fault=='stride': k['k1_eval_stride']=3
        elif fault=='status': k['status']='DEGRADED_PARENT_ONLY'
        elif fault=='not_executed': k['k1_executed']=False
        elif fault=='formula': k['formula']='wrong'
        elif fault=='contract': k['contract']={}
        elif fault=='r224': k['r224_weight']=.5
        elif fault=='model_hash': k['model_sha256']='0'*64
        elif fault=='missing_row': k['per_study'].pop()
        elif fault=='duplicate_uid': k['per_study'][1]['uid']=k['per_study'][0]['uid']
        elif fault=='failed_fraction': k['failed_study_fraction']=.051
    if fault=='O_present':
        change(d/'c85r7-receipt.json',lambda p:p['executed_members'].append('EXP155'))
    else: change(d/'k1sr-receipt.json',mutate)
    refresh_n32(d)
    if fault=='provenance':
        change(d/'k1sr-receipt.json.download.json',lambda r:r.update(sha256='0'*64))
    with pytest.raises(a.base.Refuse): a.prepare(**exact_case)
    assert not (exact_case['visible_dir']/a.base.OUTPUT).exists()


def test_n32_refuses_failed_row_with_emptied_predictions(exact_case):
    d=exact_case['n32_dir']
    def mutate(k):
        k['per_study'][0].update(ok=False,predictions={})
        k.update(k1_studies=31,failed_study_count=1,failed_study_fraction=1/32)
    change(d/'k1sr-receipt.json',mutate); refresh_n32(d)
    with pytest.raises(a.base.Refuse,match='32/32 successful studies'):
        a.prepare(**exact_case)
    assert not (exact_case['visible_dir']/a.base.OUTPUT).exists()
