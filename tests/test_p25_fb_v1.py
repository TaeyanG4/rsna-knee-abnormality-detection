"""P4 build, four-member math and isolated admission refusal coverage."""
import copy
import json
from types import ModuleType
import numpy as np
import pytest
from tools import p25_fb_stage_v1 as b
from tools import p25_fb_admit_v1 as a
from tools import p25_fb_cpu_smoke_v1 as smoke
from tools import k1n_cpu_replay_v1 as replay


@pytest.fixture(scope='module')
def runtime():
    return smoke.runtime()


def evidence(n=3,diagnostic=False):
    c=b.configuration(); cfg=b.parent.legacy.config(c,diagnostic)
    models=c['models']; ids=[m['id'] for m in models]
    rows=[dict(uid=str(i),ok=True,config_sha256=cfg['config_sha256'],model_identities=models,
        k1_models=ids,predictions={mid:[.2+i/(n*2)]*12 for mid in ids},
        per_model_seconds={mid:dict(load=1.,classify=2.) for mid in ids},
        classifier_seconds=8.,load_seconds=4.,transfer_seconds=0.,extra_render_seconds=.125,
        finished_elapsed_seconds=20.) for i in range(n)]
    k=dict(cfg,status='COMPLETE_SHARED_K1N',model_identities=models,k1_models=ids,
        parent_mode='INF3_P_R224',k1_executed=True,fail_loud_triggered=False,
        k1_max_failed_fraction=.05,k1_eval_stride=2,per_study=rows,k1_studies=n,
        failed_study_count=0,failed_study_fraction=0.,
        per_model_seconds={mid:dict(load=float(n),classify=float(n*2)) for mid in ids})
    p=dict(drop_o=True,intended_members=['R','S_R7','R224'],executed_members=['R','S_R7','R224'],
           skipped_members=['EXP143v2','EXP155','Q_A5q'])
    return c,cfg,k,p


@pytest.mark.parametrize('diagnostic',[False,True])
def test_exact_parent_scope(diagnostic):
    outputs,reg=b.artifacts(diagnostic)
    scope=json.loads(outputs['payload-scope.json'])
    assert scope['config_changed_fields']==['kernel_id','models','release_version']
    assert scope['all_runtime_functions_identical_except_fb_asset_inventory']
    assert len(scope['untouched'])>20
    assert reg['never_submit'] is diagnostic
    c=b.configuration(); old=b.read(b.PARENT/'config.json')
    assert c['models'][:3]==old['models']
    assert c['models'][3]['checkpoint_sha256']==b.FB_SHA
    assert 'p_weight_mode' not in c and 'a5' not in c and c['drop_o'] is True
    assert c['alpha']==.25 and c['k1_eval_stride']==2
    assert c['k1_guard_hours']==8.8 and c['k1_max_failed_fraction']==.05
    assert reg['capture_timeout_seconds']==900 and reg['call_timeout_seconds']==300
    assert b.DATASET_ID in json.loads(outputs['kernel/kernel-metadata.json'])['dataset_sources']


@pytest.mark.parametrize('diagnostic',[False,True])
def test_valid_four_member_contract(diagnostic):
    c,reg,k,p=evidence(32 if diagnostic else 3,diagnostic)
    a.contract(a.base.Evidence(),c,reg,k,p,len(k['per_study']),diagnostic)


@pytest.mark.parametrize('fault',[
    'three_models','missing_fb_prediction','wrong_hash','row_hash','flat_config','flat_receipt',
    'O_present','A5_present','no_fb_timing','zero_fb_time','aggregate_time','nan',
    'failed_fraction','failed_count','wrong_stride','fail_loud','duplicate_uid','missing_row',
    'understated_classify','understated_load'])
def test_semantic_refusal(fault):
    c,reg,k,p=copy.deepcopy(evidence()); mid=b.FB_ID
    if fault=='three_models': k['k1_models'].pop()
    elif fault=='missing_fb_prediction': k['per_study'][0]['predictions'].pop(mid)
    elif fault=='wrong_hash': k['model_identities'][-1]['checkpoint_sha256']='f'*64
    elif fault=='row_hash': k['per_study'][0]['config_sha256']='f'*64
    elif fault=='flat_config': c['p_weight_mode']='flat'
    elif fault=='flat_receipt': k['p_weight_mode']='flat'
    elif fault=='O_present': p['executed_members'].append('EXP155')
    elif fault=='A5_present': k['a5_contract']={}
    elif fault=='no_fb_timing': k['per_study'][0]['per_model_seconds'].pop(mid)
    elif fault=='zero_fb_time': k['per_study'][0]['per_model_seconds'][mid]['classify']=0
    elif fault=='aggregate_time': k['per_model_seconds'][mid]['classify']+=1
    elif fault=='nan': k['per_study'][0]['predictions'][mid][0]=float('nan')
    elif fault=='failed_fraction': k['failed_study_fraction']=.051
    elif fault=='failed_count': k['failed_study_count']=1
    elif fault=='wrong_stride': k['k1_eval_stride']=3
    elif fault=='fail_loud': k['fail_loud_triggered']=True
    elif fault=='duplicate_uid': k['per_study'][1]['uid']='0'
    elif fault=='missing_row': k['per_study'].pop()
    elif fault=='understated_classify': k['per_study'][0]['classifier_seconds']=.1
    elif fault=='understated_load': k['per_study'][0]['load_seconds']=.1
    with pytest.raises(a.base.Refuse): a.contract(a.base.Evidence(),c,reg,k,p,3)


def test_four_rank_oracle_and_missing_fb(runtime):
    rng=np.random.default_rng(185); parent=rng.random((17,12),dtype=np.float32)
    _,_,k,_=evidence(17); rows=k['per_study']
    arrays=[]
    for mid in runtime.MODEL_IDS:
        arr=rng.random((17,12),dtype=np.float32); arrays.append(arr)
        for i,row in enumerate(rows): row['predictions'][mid]=arr[i].tolist()
    payload=replay.csv_bytes([str(i) for i in range(17)],parent)
    _,actual=runtime.util.table(runtime.blend(payload,rows))
    rank=runtime.util.ranks
    # Include CSV serialization in the oracle's parent values.
    _,parsed=runtime.util.table(payload)
    expected=rank(.75*rank(parsed)+.25*rank(np.mean([rank(x.astype(np.float64)) for x in arrays],axis=0)))
    assert np.allclose(actual,expected.astype(np.float32),atol=1e-8,rtol=0)
    rows[0]['predictions'].pop(b.FB_ID)
    with pytest.raises(ValueError,match='all registered'): runtime.blend(payload,rows)


@pytest.mark.parametrize('diagnostic',[False,True])
def test_missing_fb_finish_fail_loud(runtime,tmp_path,diagnostic):
    c,reg,k,p=evidence(3,diagnostic)
    k['per_study'][0]['predictions'].pop(b.FB_ID)
    records=tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'; records.mkdir(parents=True)
    (records/'k1-studies.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in k['per_study']))
    (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(
        replay.csv_bytes(['0','1','2'],np.full((3,12),.5)))
    with pytest.raises(RuntimeError,match='fail-loud'):runtime.finish(tmp_path,tmp_path/'run',reg,1.)
    r=json.loads((tmp_path/'k1sr-receipt.json').read_bytes())
    assert r['fail_loud_triggered'] and not r['k1_executed']
    if diagnostic: assert not (tmp_path/'submission.csv').exists()


def test_fb_checksum_rejects_tamper_before_deserialization(runtime,tmp_path,monkeypatch):
    (tmp_path/'asset-manifest.json').write_bytes(b.util.encoded(b.MANIFEST))
    for name in b.MANIFEST['files']:
        target=tmp_path/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(b'fixture')
    (tmp_path/'SHA256SUMS').write_bytes(b'wrong')
    monkeypatch.setattr(runtime.util,'mount',lambda *args:tmp_path)
    with pytest.raises(ValueError,match='FB SHA256SUMS pin'): runtime.load_classifier('cpu',b.FB_ID)


def test_assets_exact():
    assert b.assets('validate')['files']==5
    assert b.read(b.DATASET/'dataset-metadata.json')['isPrivate'] is True
    assert (b.DATASET/'SHA256SUMS').read_bytes()==b.SUMS


def test_admit_dryrun_is_hold():
    assert b.run('admit',True)['status']=='HOLD_NOT_ADMITTED'
    with pytest.raises(ValueError): b.run('admit')
