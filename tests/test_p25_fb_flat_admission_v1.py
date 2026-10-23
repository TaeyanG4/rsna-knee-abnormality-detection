"""F4 admission v1.0.3: synthetic offline integration, never GPU qualification."""
import json
import subprocess
import sys
import pytest
from test_k1_admit_v1 import case,put,change,transport
from test_p25_fb_admission_v1 import exact_case,refresh_n32
from tools import p25_fb_admit_v1 as a
from tools import p25_fb_flat_stage_v1 as b


@pytest.fixture
def flat_case(exact_case):
    exact_case.update(stage=b.stage(),kernel=b.SLUG)
    for diagnostic in (False,True):
        d=exact_case['n32_dir' if diagnostic else 'visible_dir']
        cfg=b.config(b.configuration(),diagnostic)
        def update(k):
            k.update(cfg)
            for row in k['per_study']:row['config_sha256']=cfg['config_sha256']
        change(d/'k1sr-receipt.json',update)
        change(d/'c85r7-receipt.json',lambda p:p.update(version=cfg['release_version']))
        if diagnostic:
            for name in ('k1sr-receipt.json','c85r7-receipt.json'):
                change(d/(name+'.download.json'),lambda r:r.update(kernel=cfg['kernel_id']))
            refresh_n32(d)
        else:transport(d,kernel=b.SLUG)
    return exact_case


def test_flat_real_offline_prepare(flat_case,monkeypatch,capsys):
    calls=[];run=subprocess.run
    def guarded(args,**kwargs):
        assert '--check-only' in args and not kwargs.get('shell',False)
        calls.append(args);return run(args,**kwargs)
    monkeypatch.setattr(a.base.subprocess,'run',guarded)
    out=a.prepare(**flat_case)
    assert len(calls)==1 and 'ADMITTED_OFFLINE' in capsys.readouterr().out
    sidecar=json.loads((out/'f4-four-member-admission.json').read_bytes())
    assert sidecar['status']=='PASS_OFFLINE_F4' and sidecar['projection'] is None
    assert not sidecar['independent_review']
    assert not (flat_case['n32_dir']/'projection-inputs.json').exists()
    assert not (out/'p4-four-member-admission.json').exists()


@pytest.mark.parametrize('diagnostic',[False,True])
@pytest.mark.parametrize('fault',['public_mode','public_weights','three_members','wrong_fb_hash',
    'O_present','A5_present','nan','missing_weights'])
def test_flat_refuses_mutations(flat_case,diagnostic,fault):
    d=flat_case['n32_dir' if diagnostic else 'visible_dir']
    def mutate(k):
        if fault=='public_mode':k['p_weight_mode']='public'
        elif fault=='public_weights':k['p_raptor_weights']=list(b.flat.PUBLIC)
        elif fault=='three_members':k['per_study'][0]['predictions'].pop(b.FB_ID)
        elif fault=='wrong_fb_hash':k['model_identities'][-1]['checkpoint_sha256']='0'*64
        elif fault=='A5_present':k['a5_contract']={}
        elif fault=='nan':k['per_study'][0]['predictions'][b.FB_ID][0]=float('nan')
        elif fault=='missing_weights':k.pop('p_raptor_weights')
    if fault=='O_present':change(d/'c85r7-receipt.json',lambda p:p['executed_members'].append('EXP155'))
    elif fault=='nan':
        path=d/'k1sr-receipt.json';value=json.loads(path.read_bytes());mutate(value)
        path.write_text(json.dumps(value,allow_nan=True))
    else:change(d/'k1sr-receipt.json',mutate)
    if diagnostic:refresh_n32(d)
    else:transport(d,kernel=b.SLUG)
    with pytest.raises(a.base.Refuse):a.prepare(**flat_case)
    assert not (flat_case['visible_dir']/a.base.OUTPUT).exists()


@pytest.mark.parametrize('empty',[False,True])
def test_flat_n32_31_of_32_refused(flat_case,empty):
    d=flat_case['n32_dir']
    def mutate(k):
        k['per_study'][0]['ok']=False
        if empty:k['per_study'][0]['predictions']={}
        k.update(k1_studies=31,failed_study_count=1,failed_study_fraction=1/32)
    change(d/'k1sr-receipt.json',mutate);refresh_n32(d)
    with pytest.raises(a.base.Refuse,match='32/32 successful studies'):a.prepare(**flat_case)
    assert not (flat_case['visible_dir']/a.base.OUTPUT).exists()


def test_cli_status_flat(flat_case,monkeypatch,capsys):
    argv=['p25_fb_admit_v1','validate']
    for k,v in flat_case.items():argv.extend(['--'+k.replace('_','-'),str(v)])
    monkeypatch.setattr(sys,'argv',argv)
    assert a.main()==0
    assert json.loads(capsys.readouterr().out.strip())['status']=='PASS_OFFLINE_F4'


def test_n32_stage_cannot_be_admitted(flat_case):
    flat_case.update(stage=b.stage(True),kernel=b.SLUG+'-n32')
    with pytest.raises(a.base.Refuse,match='exact stage'):a.validate(**flat_case)
