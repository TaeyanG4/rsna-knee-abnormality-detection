"""Synthetic completion/provenance/statistics; no actual outcomes, model or GPU."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exp085_report',ROOT/'tools/exp085_relative_position_report.py')
r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)

def sample():
    y=np.tile(np.array([0,1,0,1,0,1],np.int8)[:,None],(1,12))
    good=.1+.8*y
    return y,np.full_like(good,.5),good,1-good,good

def test_primary_raw_and_both_reference_losses_not_hidden():
    y,c,t,b,rn=sample(); s=r.scores_for(c,t,np.clip(t+.01,0,1),rn)
    rows=r.target_rows(y,s); a=r.aggregate(rows)
    assert a['contrasts']['primary_treatment_minus_control']['macro12']==.5
    assert a['contrasts']['mandatory_control_minus_B']['macro12']==-.5
    assert a['contrasts']['mandatory_treatment_minus_B']['macro12']==0.
    assert sum(v['delta_over_12']['primary_treatment_minus_control'] for v in rows)==pytest.approx(.5)
    np.testing.assert_array_equal(s['control'],c)
    assert s['blend_control'][0,0]==pytest.approx(.9*7/12+.1/3)

def test_weighted_ties_match_repeated_rows():
    y=np.array([0,1,1,0,1]); p=np.array([.4,.4,.8,.9,.9]); w=np.array([3,2,0,1,4])
    assert r.base.weighted_auc(y,p,w)==pytest.approx(roc_auc_score(np.repeat(y,w),np.repeat(p,w)))

def test_shared_bootstrap_keeps_identical_primary_zero_and_no_rerank(monkeypatch):
    y,c,t,b,rn=sample(); scores=r.scores_for(t,t,b,rn)
    monkeypatch.setattr(r.pd.DataFrame,'rank',lambda *a,**k:pytest.fail('Unexpected bootstrap/subgroup rank recomputation'))
    sub=r.target_rows(y[:4],{k:v[:4] for k,v in scores.items()})
    assert r.aggregate(sub)['complete12']
    a=r.paired_bootstrap(y,scores,['a','a','b','b','c','c'],reps=20)
    b=r.paired_bootstrap(y,scores,['a','a','b','b','c','c'],reps=20)
    assert a==b and a['valid']==20 and a['ranks_recomputed'] is False
    assert a['intervals']['primary_treatment_minus_control']['macro12']==[0.,0.]

def test_one_single_class_target_invalidates_all_macro_draws():
    y,c,t,b,rn=sample(); y[:,0]=0
    scores=r.scores_for(c,t,b,rn); result=r.paired_bootstrap(y,scores,['a','a','b','b','c','c'],reps=10)
    assert (result['valid'],result['invalid'],result['invalid_targets']['ACL'])==(0,10,10)
    assert all(v['macro12'] is None for v in result['intervals'].values())
    assert not r.aggregate(r.target_rows(y,scores))['complete12']

@pytest.mark.parametrize('bad',['nan','range','shape'])
def test_predictions_contract_invalid(bad):
    y,c,t,b,rn=sample()
    if bad=='nan':t[0,0]=np.nan
    if bad=='range':t[0,0]=1.1
    if bad=='shape':t=t[:,:11]
    with pytest.raises(ValueError):r.scores_for(c,t,b,rn)

def model_metadata(arm='control'):
    env={k:'same' for k in ['torch','cuda','cudnn','gpu','timm','numpy','pydicom','cv2']}
    p={'schema':'exp085-relative-position-v2-fp32-reduction','expected_ids':{'train':16,'val':6},'training':{'bs':8},
       'execution':{'attempts_per_fit':16,'per_fit_seconds':3600},'initial_checkpoint':{'sha256':'e'*64},
       'ordered_ids':{'train_sha256':'1'*64,'val_sha256':'2'*64},'expected_runtime':{'runtime':{'policy':'strict'}},
       'architecture':{'extra_parameters':262208},'environment':env,'source_bindings':{'tools/process_resource_receipt.py':'s'*64}}
    summary={'arm':arm,'phase':'full','status':'PASS','protocol_sha256':'d'*64,'original_B_forward_exact':True,
             'train':16,'val':6,'inference_rows':6,'updates':16,'skips':0,'selected_epoch_1based':8,'weak_oof_macro_auc':None,
             'gold_access':'disabled','metric_access':'deferred_until_both_final','checkpoint_saved':True,'predictions_saved':True,
             'initial_state_sha256':'b'*64,'elapsed_seconds':100.,
             'initial_forward_gate':{'probe_rng_and_buffers_restored':True,**{mode:{k:True for k in ['logits_exact','rng_consumption_exact','buffer_updates_exact']} for mode in ['eval','train']}},
             'last_attention_dtypes':{key:'torch.float32' for key in ['bias','logits','attention','context']}}
    provenance={'schema':p['schema'],'arm':arm,'protocol_sha256':'d'*64,'initial_checkpoint_sha256':'e'*64,
                'train_ids_ordered_sha256':'1'*64,'eval_ids_ordered_sha256':'2'*64,'gold_access':'disabled',
                'metric_access':'deferred_until_both_final','runtime':{'policy':'strict'},'extra_parameters':262208,
                'all_backbone_parameters_trainable':True,'epoch_offset':8,'initial_state_sha256':'b'*64,
                'base_initial_state_sha256':'c'*64,'environment':env}
    history=[dict(epoch=i,sampling_epoch=i+8,updates=(i+1)*2,skips=0,training_loss=.5,validation_metric=None,gold_metric=None) for i in range(8)]
    trace=[dict(epoch=i//2,sampling_epoch=i//2+8,batch=i%2,
                **{k:'a'*64 for k in ['uids_sha256','centers_sha256','positions_sha256','rng_before','rng_after']},
                **({'pixels_sha256':'a'*64} if i%2==0 else {})) for i in range(16)]
    return p,summary,provenance,history,trace

@pytest.mark.parametrize('bad',['skip','epoch','ancestor','uid','gold','dtype','initial_rng','history_metric','trace_order','trace_missing'])
def test_arm_completion_rejects_invalid_contract(bad):
    p,s,v,h,t=model_metadata()
    if bad=='skip':s['skips']=1
    if bad=='epoch':s['selected_epoch_1based']=7
    if bad=='ancestor':v['initial_checkpoint_sha256']='f'*64
    if bad=='uid':v['eval_ids_ordered_sha256']='f'*64
    if bad=='gold':v['gold_access']='enabled'
    if bad=='dtype':s['last_attention_dtypes']['context']='torch.float16'
    if bad=='initial_rng':s['initial_forward_gate']['train']['rng_consumption_exact']=False
    if bad=='history_metric':h[2]['validation_metric']=.9
    if bad=='trace_order':t[3]['batch']=0
    if bad=='trace_missing':t[0].pop('pixels_sha256')
    with pytest.raises(ValueError):r.validate_arm(s,v,h,t,p,'control','d'*64)

def test_valid_arm_preserves_skip_budget_and_ancestor():
    p,s,v,h,t=model_metadata(); result=r.validate_arm(s,v,h,t,p,'control','d'*64)
    assert result['updates']==16 and result['skips']==0 and result['base_initial_state_sha256']=='c'*64

def resource():
    return {'schema':'process-resource-receipt-v1','supervisor_sha256':'s'*64,'status':'COMPLETE','child_exit_code':0,
            'wrapper_exit_code':0,'child_exit_code_before_cleanup':0,'failure':None,'elapsed_seconds':100.,'timeout_seconds':600.,
            'cleanup':{'errors':[],'survivors':[],'actions':[]},'retries':0,'sample_count':3,'failed_sample_attempts':0,'access_failure_count':0}

@pytest.mark.parametrize('bad',['failure','cleanup','nan','timeout','telemetry'])
def test_resource_gate_rejects_false_completion(bad):
    x=resource()
    if bad=='failure':x['child_exit_code']=1
    if bad=='cleanup':x['cleanup']['actions']=[{'kill':1}]
    if bad=='nan':x['elapsed_seconds']=float('nan')
    if bad=='timeout':x['timeout_seconds']=800
    if bad=='telemetry':x['failed_sample_attempts']=1
    with pytest.raises(ValueError):r.resource_gate(x,600,'s'*64)

def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')

@pytest.fixture
def completed(tmp_path):
    p,_,_,_,_=model_metadata(); p['outputs']={}
    reg={'protocol':'protocol.json','protocol_sha256':'d'*64,'phase_output':'full','full_resource':'full_RESOURCE/receipt.json',
         'pilot_output':'pilot','pilot_resource':'pilot_RESOURCE/receipt.json','root_decision':'decision.json',
         'job':{'logs':'logs','tag':'FULL','command':['python','registered']}}
    phase={'status':'PASS','phase':'full','protocol_sha256':'d'*64,'paired_initialization_exact':True,'paired_trace_exact':True,
           'gold_access':'disabled','no_metrics_evaluated':True,'no_validation_target_values_converted_or_scored':True,'retries':0,'arms':{}}
    pilot={'status':'PASS','phase':'pilot','protocol_sha256':'d'*64,'paired_initialization_exact':True,'paired_trace_exact':True,
           'no_metrics_evaluated':True,'gold_access':'disabled','pilot_state_discarded':True,
           'conservative_full_seconds_by_arm':{'control':1000.,'treatment':1000.},'arms':{}}
    for arm in r.ARMS:
        _,s,v,h,t=model_metadata(arm)
        p['outputs'][arm]={k:f'full/{arm}/{name}' for k,name in [('summary','summary.json'),('provenance','provenance.json'),
            ('checkpoint','model_final.pt'),('predictions','oof_final.csv'),('history','history.json'),('trace','trace.json')]}
        spec=p['outputs'][arm]
        for name,value in [('provenance',v),('history',h),('trace',t)]:dump(tmp_path/spec[name],value)
        for name in ['checkpoint','predictions']:(tmp_path/spec[name]).write_bytes(b'synthetic artifact; never parse as model or predictions')
        for key,field in [('checkpoint','checkpoint_sha256'),('predictions','predictions_sha256'),('trace','trace_sha256')]:s[field]=r.sha(tmp_path/spec[key])
        dump(tmp_path/spec['summary'],s);phase['arms'][arm]=s
        pilot['arms'][arm]={'status':'PASS','phase':'pilot','initial_state_sha256':'b'*64,'updates':32,'skips':0,
                            'inference_rows':16,'checkpoint_saved':False,'predictions_saved':False}
    dump(tmp_path/'full/receipt.json',phase);dump(tmp_path/'pilot/receipt.json',pilot)
    for phase_name in ['pilot','full']:
        x=resource();x['argv']=['python',str(tmp_path/'training/runners/exp085_relative_position.py'),'--protocol',str(tmp_path/'protocol.json'),'--phase',phase_name,'--out',str(tmp_path/phase_name)]
        dump(tmp_path/f'{phase_name}_RESOURCE/receipt.json',x)
    dump(tmp_path/'decision.json',{'protocol_sha256':'d'*64,'allow_phase':'full','retries':0,'pilot_receipt_sha256':r.sha(tmp_path/'pilot/receipt.json')})
    dump(tmp_path/'logs/job.json',{'tag':'FULL','command':['python','registered']})
    (tmp_path/'logs/SUCCESS').write_text('exit 0 at 2026-10-04T03:00:00')
    return tmp_path,reg,p

def test_complete_pair_validates_without_prediction_parser(completed,monkeypatch):
    root,reg,p=completed
    monkeypatch.setattr(r.base,'predictions',lambda *a:pytest.fail('No prediction values during completion gate'))
    monkeypatch.setattr(r.base,'load_reference',lambda *a:pytest.fail('No labels during completion gate'))
    e,hashes=r.require_completed(root,reg,p)
    assert e['paired_rng_trace_exact'] and e['control']['updates']==e['treatment']['updates']==16
    assert p['outputs']['control']['checkpoint'] in hashes

@pytest.mark.parametrize('bad',['missing_arm','job_failed','phase_failed','checkpoint_changed','paired_rng','wrong_resource','decision'])
def test_incomplete_or_mismatched_pair_blocks_outcome_access(completed,monkeypatch,bad):
    root,reg,p=completed
    monkeypatch.setattr(r.base,'predictions',lambda *a:pytest.fail('Outcome accessed'))
    monkeypatch.setattr(r.base,'load_reference',lambda *a:pytest.fail('Labels accessed'))
    if bad=='missing_arm':(root/p['outputs']['treatment']['summary']).unlink()
    if bad=='job_failed':(root/'logs/FAILED').write_text('exit 1')
    if bad=='phase_failed':
        x=r.read(root/'full/receipt.json');x['status']='FAILED';dump(root/'full/receipt.json',x)
    if bad=='checkpoint_changed':(root/p['outputs']['control']['checkpoint']).write_text('changed')
    if bad=='paired_rng':
        path=root/p['outputs']['treatment']['trace'];x=r.read(path);x[2]['rng_before']='f'*64;dump(path,x)
        summary=r.read(root/p['outputs']['treatment']['summary']);summary['trace_sha256']=r.sha(path);dump(root/p['outputs']['treatment']['summary'],summary)
        phase=r.read(root/'full/receipt.json');phase['arms']['treatment']=summary;dump(root/'full/receipt.json',phase)
    if bad=='wrong_resource':
        path=root/reg['full_resource'];x=r.read(path);x['argv'][-1]=str(root/'unrelated');dump(path,x)
    if bad=='decision':
        x=r.read(root/'decision.json');x['pilot_receipt_sha256']='f'*64;dump(root/'decision.json',x)
    with pytest.raises(ValueError):r.require_completed(root,reg,p)

def test_new_module_import_does_not_import_torch():
    import subprocess
    code="import runpy,sys;runpy.run_path('tools/exp085_relative_position_report.py',run_name='passive');assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
