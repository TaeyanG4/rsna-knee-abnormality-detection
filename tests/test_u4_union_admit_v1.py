"""Full U admission including real canonical offline check-only, never submit."""
import json
import subprocess
import pytest
from test_u4_union_stage_v1 import make_case,package,put
from tools import u_union_admit_v1 as a
from tools import u4_union_stage_v1 as b

@pytest.fixture
def case(tmp_path,monkeypatch,request):
    stage=tmp_path/('v'+b.RELEASE)
    for diag in (False,True):
        outputs,reg,_,_=package(diag);s=tmp_path/('v'+b.RELEASE+('-n32' if diag else ''))
        for name,data in outputs.items():
            path=s/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        reg=dict(reg,stage=str(s));put(s/'registration.json',reg)
    dirs={}
    for diag in (False,True):
        fault='one_failed' if diag and getattr(request,'param',False) else None
        d,reg,k,p=make_case(tmp_path/('n32' if diag else 'visible'),monkeypatch,diag,fault)
        dirs['n32_dir' if diag else 'visible_dir']=d
        for name in ('k1sr-receipt.json','c85r7-receipt.json','diagnostic-k1n.csv' if diag else 'submission.csv'):
            put(d/(name+'.download.json'),dict(status='DOWNLOADED',kernel=reg['kernel_id'],version=1,file=name,local_name=name,sha256=a.sha((d/name).read_bytes())))
        # Actual notebook layout: N32 publishes diagnostic-k1n.csv, never
        # submission.csv or the legacy projection-inputs.json side input.
    template=tmp_path/'template.csv';template.write_bytes((dirs['visible_dir']/'submission.csv').read_bytes())
    ledger=tmp_path/'ledger.jsonl';ledger.write_bytes(b'')
    monkeypatch.setattr(a.inherited,'TEMPLATE',template);monkeypatch.setattr(a.inherited,'LEDGER',ledger)
    return dict(stage=stage,kernel=b.SLUG,version=1,**dirs)

def test_full_admission_and_actual_offline_wrapper(case,monkeypatch,capsys):
    assert not (case['n32_dir']/'projection-inputs.json').exists()
    assert not (case['n32_dir']/'submission.csv').exists()
    assert (case['n32_dir']/'diagnostic-k1n.csv').is_file()
    real=subprocess.run;calls=[]
    def guarded(args,**kw):
        assert '--check-only' in args and not kw.get('shell');calls.append(args);return real(args,**kw)
    monkeypatch.setattr(a.subprocess,'run',guarded)
    e,reg,k,proj=a.validate(**case)
    assert proj is None and len([v for v in e.checks if 'Independent scipy' in v])==2
    assert any("runtime is judged by B's 8-step gate, not by this tool" in v for v in e.checks)
    assert not any('N32: NOT PROVIDED' in v for v in e.checks)
    out=a.prepare(**case)
    assert len(calls)==1 and 'ADMITTED_OFFLINE' in capsys.readouterr().out
    registration=json.loads((out/'ordinary-visible-registration.json').read_bytes())
    assert registration['independent_review']=='NOT PERFORMED BY THIS TOOL'
    assert registration['projection'] is None and registration['tool_version']=='1.0.2-u4'
    review=(out/'READY-REVIEW.md').read_text()
    assert 'Projection: null' in review and "runtime is judged by B's 8-step gate, not by this tool" in review

@pytest.mark.parametrize('case',[True],indirect=True)
def test_full_admission_allows_one_n32_failure(case):
    _,_,_,projection=a.validate(**case)
    assert projection is None
    assert json.loads((case['n32_dir']/'k1sr-receipt.json').read_bytes())['k1_studies']==31

@pytest.mark.parametrize('directory',['visible_dir','n32_dir'])
@pytest.mark.parametrize('fault',['partial_o','stride','weights','a5','download','intermediate',
    'incomplete','not_executed','fraction','too_many_failed','model_hash','formula','contract','r224','trio','fb_hash','missing_o'])
def test_full_admission_refuses(case,directory,fault):
    d=case[directory]
    if fault=='trio':name='k1sr-receipt.json';key='k1_models';value=a.MODELS
    elif fault=='fb_hash':name='k1sr-receipt.json';key='model_sha256';value=dict(a.U4_HASHES,k1_v3g_fullbag='0'*64)
    elif fault=='missing_o':name='c85r7-receipt.json';key='drop_o';value=True
    elif fault=='partial_o':name='c85r7-receipt.json';key='executed_members';value=['R','R224','S_R7']
    elif fault=='stride':name='k1sr-receipt.json';key='k1_eval_stride';value=3
    elif fault=='weights':name='u-inputs.json';key='contract';value={}
    elif fault=='a5':name='k1sr-receipt.json';key='a5_executed';value=True
    elif fault=='download':name='k1sr-receipt.json.download.json';key='sha256';value='0'*64
    elif fault=='incomplete':name='k1sr-receipt.json';key='status';value='INCOMPLETE'
    elif fault=='not_executed':name='k1sr-receipt.json';key='k1_executed';value=False
    elif fault=='fraction':name='k1sr-receipt.json';key='failed_study_fraction';value=.06
    elif fault=='too_many_failed':name='k1sr-receipt.json';key='k1_studies';value=30
    elif fault=='model_hash':name='k1sr-receipt.json';key='model_sha256';value='0'*64
    elif fault=='formula':name='k1sr-receipt.json';key='formula';value='wrong formula'
    elif fault=='contract':name='k1sr-receipt.json';key='contract';value={}
    elif fault=='r224':name='k1sr-receipt.json';key='r224_weight';value=.25
    else:
        path=d/'u-k.csv';path.write_bytes(path.read_bytes()+b' ')
        with pytest.raises((ValueError,a.Refuse)):a.validate(**case)
        return
    path=d/name;obj=json.loads(path.read_bytes());obj[key]=value;put(path,obj)
    if name in ('c85r7-receipt.json','k1sr-receipt.json'):
        side=path.with_name(name+'.download.json');obj=json.loads(side.read_bytes());obj['sha256']=a.sha(path.read_bytes());put(side,obj)
    with pytest.raises((ValueError,a.Refuse)):a.validate(**case)
