"""Frozen CPU-only EXP085 final evaluator; outcome access requires both final arms."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'exp085-relative-position-report-v1'
EVIDENCE = 'docs/handoffs/evidence/2026-10/2026-10-04-exp085-report-tool'
BASE_PATH = 'tools/exp084_group_loss_report.py'
BASE_SHA = '9eaec5593ad42c44dc345836fa22ec01431f5ab8f1bd790ad5c62b75a721f0fb'
ARMS = ('control', 'treatment')
SCORES = ('B_original','RN_clip','control','treatment','blend_B','blend_control','blend_treatment')
CONTRASTS = {'primary_treatment_minus_control': ('control','treatment'),
             'mandatory_control_minus_B': ('B_original','control'),
             'mandatory_treatment_minus_B': ('B_original','treatment'),
             'secondary_blend_treatment_minus_control': ('blend_control','blend_treatment'),
             'secondary_blend_control_minus_B': ('blend_B','blend_control'),
             'secondary_blend_treatment_minus_B': ('blend_B','blend_treatment'),
             'secondary_original_RN_complement': ('B_original','blend_B')}

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()

need(sha(ROOT/BASE_PATH)==BASE_SHA,'Frozen passive statistical adapter changed')
_spec=importlib.util.spec_from_file_location('exp085_base_report',ROOT/BASE_PATH)
base=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)
TARGETS, UID = base.TARGETS, base.UID
read, contained, relative, write_new = base.read, base.contained, base.relative, base.write_new
LIMITS = base.LIMITS + [
    'Same development V was already observed in EXP084; one seed and weights-only adaptation with optimizer restart.',
    'Mandatory original-B contrasts include architecture/adaptation/restart; only the paired primary isolates position input in this recipe.',
    'Input-A is absent; historical B-minus-A Fracture loss remains unresolved. EXP082 positive uncertain evidence and small RN complement remain relevant.',
    'Cache-relative position is not physical/anatomical position and may proxy acquisition protocol.',
]

def scores_for(control,treatment,b,rn):
    values=[np.asarray(v,dtype=np.float64) for v in (control,treatment,b,rn)]
    need(values[0].ndim==2 and values[0].shape[1]==12 and len(values[0])>0
         and all(v.shape==values[0].shape and np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in values),
         'Matching finite n by12 probabilities required')
    c,t,b,rn=values
    rc,rt,rb,rr=[pd.DataFrame(v).rank(method='average',pct=True).to_numpy() for v in values]
    return dict(B_original=b,RN_clip=rn,control=c,treatment=t,
                blend_B=.9*rb+.1*rr,blend_control=.9*rc+.1*rr,blend_treatment=.9*rt+.1*rr)

def target_rows(y,scores,weights=None):
    y=base.checked_binary(y)
    need(set(scores)==set(SCORES) and all(v.shape==y.shape and np.isfinite(v).all() for v in scores.values()),'Wrong score arms/shape')
    w=np.ones(len(y),dtype=np.int64) if weights is None else np.asarray(weights)
    rows=[]
    for j,target in enumerate(TARGETS):
        auc={name:base.weighted_auc(y[:,j],scores[name][:,j],w) for name in SCORES}
        auc={name:float(v) if np.isfinite(v) else None for name,v in auc.items()}
        delta={name:None if auc[a] is None or auc[b] is None else auc[b]-auc[a] for name,(a,b) in CONTRASTS.items()}
        rows.append(dict(target=target,P=int(y[:,j].sum()),N=int(len(y)-y[:,j].sum()),auc=auc,delta=delta,
                         delta_over_12={k:None if v is None else v/12 for k,v in delta.items()}))
    return rows

def aggregate(rows):
    need([r['target'] for r in rows]==TARGETS,'Exact12 target rows required')
    complete=all(all(v is not None for v in r['auc'].values()) for r in rows)
    return {'complete12':complete,'auc':{a:float(np.mean([r['auc'][a] for r in rows])) if complete else None for a in SCORES},
            'contrasts':{k:{'macro12':float(np.mean([r['delta'][k] for r in rows])) if complete else None,
                            **{t:next(r['delta'][k] for r in rows if r['target']==t) for t in ['Fracture','PF OA','MCL']}} for k in CONTRASTS}}

def paired_bootstrap(y,scores,components,reps=500,seed=20261004):
    y=base.checked_binary(y); components=np.asarray(components,dtype=str)
    groups,codes=np.unique(components,return_inverse=True)
    need(len(codes)==len(y) and len(groups)>0 and not (components=='').any() and reps>0,'Invalid component bootstrap')
    rng=np.random.default_rng(seed); digest=hashlib.sha256(); invalid_targets=np.zeros(12,np.int64); values=[]; invalid=0
    for _ in range(reps):
        draw=rng.integers(0,len(groups),len(groups)); digest.update(draw.astype('<i8').tobytes())
        weights=base.component_weights(codes,draw,len(groups))
        bad=((weights[:,None]*y).sum(0)==0)|((weights[:,None]*(1-y)).sum(0)==0)
        if bad.any(): invalid+=1; invalid_targets+=bad; continue
        rows=target_rows(y,scores,weights)
        values.append([[r['delta'][name] for r in rows] for name in CONTRASTS])
    array=np.asarray(values)
    ci=lambda x:np.quantile(x,[.025,.975]).tolist() if len(x) else None
    intervals={k:{'macro12':ci(array[:,i].mean(1)) if len(values) else None,
                  'targets':{t:ci(array[:,i,j]) if len(values) else None for j,t in enumerate(TARGETS)}}
               for i,k in enumerate(CONTRASTS)}
    return dict(attempted=reps,valid=len(values),invalid=invalid,invalid_targets=dict(zip(TARGETS,map(int,invalid_targets))),
                component_count=len(groups),seed=seed,draws_sha256=digest.hexdigest(),same_draw_all_arms_and_targets=True,
                ranks_recomputed=False,intervals=intervals,scope='Conditional fixed-score component bootstrap; all CIs share complete12 valid draws.')

def inference_contract(p):
    a=p['training']
    return {**{k:a[k] for k in ['res','k_eval','norm','window_policy','eval_amp','runtime_policy']},
            'corpus':p['paths']['corpus'],'source':p['training_code_sha256'],
            'batch':a['bs']//2,'workers':min(a['workers'],2),'internal_cpu_threads':1,
            'grad_mode':'no_grad','output':'float32 sigmoid then ordered CSV'}

def validate_protocol(p,parent):
    need(p['schema']=='exp085-relative-position-v2-fp32-reduction' and p['targets']==TARGETS
         and p['arm_order']==list(ARMS),'Wrong EXP085 protocol')
    for key in ['paths','ordered_ids','groups','expected_ids','split','excluded_uids','loss','reference_counts',
                'label_hash','eval_label_hash','split_hash','training_code_sha256','expected_runtime','environment']:
        need(p[key]==parent[key],'Parent/reference contract differs: '+key)
    a=p['training']; e=p['execution']; ev=p['evaluation']
    need(all(a[k]==v for k,v in {'res':224,'k':12,'k_eval':24,'epochs':8,'bs':8,'accum':1,'seed':42,
         'norm':'imagenet','window_policy':'within-slot-valid','amp':'fp16','eval_amp':'fp16','runtime_policy':'strict'}.items()),'Fixed training contract changed')
    need(e['full_epochs']==8 and e['epoch_offset']==8 and e['attempts_per_fit']==8*(p['expected_ids']['train']//8)
         and e['all_skips_require_failure'] is True and e['retries']==0 and e['total_time_limit_seconds']==7200
         and e['per_fit_seconds']==3600 and e['pilot_seconds']==600,'Changed fixed budget')
    need(ev['primary']=='macro12(treatment)-macro12(control)' and ev['weights']==[.9,.1]
         and ev['bootstrap_reps']==500 and ev['bootstrap_seed']==20261004 and ev['bootstrap_unit']=='component_id',
         'Changed fixed analysis contrasts/bootstrap')
    original=parent['arms']['B']; wanted=inference_contract(p)
    prior={'training':original['expected_args'],'paths':parent['paths'],'training_code_sha256':parent['training_code_sha256']}
    need(wanted==inference_contract(prior),'Original B inference contract differs; reference reuse blocked')
    need(p['initial_checkpoint']['path']==original['outputs']['checkpoint'],'Initial ancestor is not exact original B')
    need(p['architecture'].get('attention_precision')=='autocast disabled for FP32 logits/softmax/value reduction; qkv/positionMLP/projection retain ambient native autocast dtype',
         'Registered FP32 attention boundary differs')
    return wanted

def positive_number(v): return type(v) in (int,float) and math.isfinite(v) and v>0
def hexhash(v): return isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) is not None

def resource_gate(value,max_seconds,source_sha):
    need(value.get('schema')=='process-resource-receipt-v1' and value.get('supervisor_sha256')==source_sha
         and value.get('status')=='COMPLETE' and value.get('child_exit_code')==value.get('wrapper_exit_code')==0
         and value.get('child_exit_code_before_cleanup')==0 and value.get('failure') is None,'Resource execution failed')
    need(positive_number(value.get('elapsed_seconds')) and value['elapsed_seconds']<=max_seconds
         and positive_number(value.get('timeout_seconds')) and value['timeout_seconds']<=max_seconds,'Resource budget failed')
    cleanup=value.get('cleanup',{})
    need(not cleanup.get('errors') and not cleanup.get('survivors') and not cleanup.get('actions'),'Resource cleanup not natural/complete')
    need(value.get('retries')==0 and value.get('sample_count',0)>0 and value.get('failed_sample_attempts')==0
         and value.get('access_failure_count')==0,'Resource receipt incomplete')
    return value['elapsed_seconds']

def resource_command_gate(value,root,reg,phase):
    argv=value.get('argv',[])
    expected={'--phase':phase,'--protocol':str(contained(root,reg['protocol'])),
              '--out':str(contained(root,reg['phase_output'] if phase=='full' else reg['pilot_output']))}
    need(len(argv)>=2 and Path(argv[1]).resolve()==contained(root,'training/runners/exp085_relative_position.py').resolve(),
         'Resource does not supervise the registered runner')
    for flag,wanted in expected.items():
        need(argv.count(flag)==1 and argv.index(flag)+1<len(argv),'Missing/duplicate resource command flag')
        value=argv[argv.index(flag)+1]
        need(value==wanted if flag=='--phase' else Path(value).resolve()==Path(wanted).resolve(), 'Resource phase/protocol/output mismatch')

def validate_arm(summary,provenance,history,trace,p,arm,protocol_sha):
    n=p['expected_ids']; e=p['execution']; attempts=e['attempts_per_fit']; per_epoch=n['train']//p['training']['bs']
    expected={'arm':arm,'phase':'full','status':'PASS','protocol_sha256':protocol_sha,
              'original_B_forward_exact':True,'train':n['train'],'val':n['val'],'inference_rows':n['val'],
              'updates':attempts,'skips':0,'selected_epoch_1based':8,'weak_oof_macro_auc':None,
              'gold_access':'disabled','metric_access':'deferred_until_both_final','checkpoint_saved':True,'predictions_saved':True}
    need(all(summary.get(k)==v for k,v in expected.items()),'Arm completion/fixed-final/skip mismatch: '+arm)
    need(type(summary['updates']) is int and type(summary['skips']) is int,'Invalid integer counters')
    forward=summary.get('initial_forward_gate',{})
    need(forward.get('probe_rng_and_buffers_restored') is True and all(
         forward.get(mode,{}).get(key) is True for mode in ['eval','train']
         for key in ['logits_exact','rng_consumption_exact','buffer_updates_exact']), 'Initial forward/RNG/buffer gate incomplete')
    need(all(summary.get('last_attention_dtypes',{}).get(key)=='torch.float32'
         for key in ['bias','logits','attention','context']),'Actual FP32 attention reduction evidence missing')
    need(positive_number(summary.get('elapsed_seconds')) and summary['elapsed_seconds']<=e['per_fit_seconds'],'Arm budget exceeded')
    expected_p={'schema':p['schema'],'arm':arm,'protocol_sha256':protocol_sha,
                'initial_checkpoint_sha256':p['initial_checkpoint']['sha256'],
                'train_ids_ordered_sha256':p['ordered_ids']['train_sha256'],'eval_ids_ordered_sha256':p['ordered_ids']['val_sha256'],
                'gold_access':'disabled','metric_access':'deferred_until_both_final','runtime':p['expected_runtime']['runtime'],
                'extra_parameters':p['architecture']['extra_parameters'],'all_backbone_parameters_trainable':True,'epoch_offset':8}
    need(all(provenance.get(k)==v for k,v in expected_p.items()),'Arm ancestry/source/UID/runtime mismatch: '+arm)
    need(summary['initial_state_sha256']==provenance['initial_state_sha256'] and hexhash(provenance['initial_state_sha256'])
         and hexhash(provenance.get('base_initial_state_sha256')),'Initial state evidence invalid')
    need(all(provenance['environment'].get(k)==p['environment'][k] for k in ['torch','cuda','cudnn','gpu','timm','numpy','pydicom','cv2']), 'Arm environment differs')
    need(len(history)==8 and len(trace)==attempts,'Incomplete history/RNG trajectory')
    for ep,h in enumerate(history):
        need(h['epoch']==ep and h['sampling_epoch']==ep+8 and h['updates']==(ep+1)*per_epoch and h['skips']==0
             and math.isfinite(h['training_loss']) and h['validation_metric'] is None and h['gold_metric'] is None,'Intermediate metric/budget/history mismatch')
    for i,row in enumerate(trace):
        ep,batch=divmod(i,per_epoch)
        need(row['epoch']==ep and row['sampling_epoch']==ep+8 and row['batch']==batch,'Trace minibatch order differs')
        need(all(hexhash(row.get(k)) for k in ['uids_sha256','centers_sha256','positions_sha256','rng_before','rng_after']), 'Incomplete RNG/input trace')
        if batch==0: need(hexhash(row.get('pixels_sha256')),'Missing per-epoch pixel check')
    return {'updates':attempts,'skips':0,'epochs':8,'adaptation':'weights-only B8 + fresh optimizer8; not uninterrupted B16',
            'elapsed_seconds':summary['elapsed_seconds'],'initial_state_sha256':provenance['initial_state_sha256'],
            'base_initial_state_sha256':provenance['base_initial_state_sha256']}

def require_completed(root,reg,p):
    """All markers/artifacts checked before any prediction or label value is opened."""
    out=contained(root,reg['phase_output']); job=contained(root,reg['job']['logs'])
    paths=[out/'receipt.json',contained(root,reg['full_resource']),job/'SUCCESS',job/'job.json',
           contained(root,reg['pilot_output'])/'receipt.json',contained(root,reg['pilot_resource']),contained(root,reg['root_decision'])]
    paths += [contained(root,path) for arm in ARMS for path in p['outputs'][arm].values()]
    need(not (job/'FAILED').exists() and not (out/'failure.json').exists() and all(v.is_file() for v in paths),
         'Both final arms, phase/resource receipts and job SUCCESS required before outcomes')
    need(re.fullmatch(r'exit 0 at \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\s*',(job/'SUCCESS').read_text()) is not None,'Invalid SUCCESS marker')
    actualjob=read(job/'job.json')
    need(actualjob['tag']==reg['job']['tag'] and actualjob['command']==reg['job']['command'],'Actual full command changed')
    receipt=read(out/'receipt.json'); pilot=read(contained(root,reg['pilot_output'])/'receipt.json')
    need(receipt.get('status')=='PASS' and receipt.get('phase')=='full' and receipt.get('protocol_sha256')==reg['protocol_sha256']
         and receipt.get('paired_initialization_exact') is True and receipt.get('paired_trace_exact') is True
         and receipt.get('gold_access')=='disabled' and receipt.get('no_metrics_evaluated') is True
         and receipt.get('no_validation_target_values_converted_or_scored') is True and receipt.get('retries')==0,'Whole paired phase incomplete')
    need(pilot.get('status')=='PASS' and pilot.get('phase')=='pilot' and pilot.get('protocol_sha256')==reg['protocol_sha256']
         and pilot.get('paired_initialization_exact') is True and pilot.get('paired_trace_exact') is True,'Pilot contract differs')
    source=p['source_bindings']['tools/process_resource_receipt.py']
    pilot_resource=read(contained(root,reg['pilot_resource'])); full_resource=read(contained(root,reg['full_resource']))
    resource_command_gate(pilot_resource,root,reg,'pilot'); resource_command_gate(full_resource,root,reg,'full')
    pilot_time=resource_gate(pilot_resource,600,source)
    full_time=resource_gate(full_resource,7200-pilot_time,source)
    need(pilot.get('no_metrics_evaluated') is True and pilot.get('gold_access')=='disabled'
         and pilot.get('pilot_state_discarded') is True,'Pilot metric/exposure contract differs')
    projected=pilot.get('conservative_full_seconds_by_arm',{})
    need(set(projected)==set(ARMS) and all(positive_number(v) and v<=3600 for v in projected.values())
         and sum(projected.values())+pilot_time<=7200,'Pilot cost gate invalid')
    for arm in ARMS:
        pa=pilot['arms'][arm]
        need(pa.get('status')=='PASS' and pa.get('phase')=='pilot' and pa.get('updates')==32 and pa.get('skips')==0
             and pa.get('inference_rows')==16 and pa.get('checkpoint_saved') is False and pa.get('predictions_saved') is False,
             'Pilot arm completion/scope differs')
    decision=read(contained(root,reg['root_decision']))
    need(decision.get('protocol_sha256')==reg['protocol_sha256'] and decision.get('allow_phase')=='full'
         and decision.get('retries')==0 and decision.get('pilot_receipt_sha256')==sha(contained(root,reg['pilot_output'])/'receipt.json'), 'Full decision/pilot binding differs')
    execution={}; traces={}; states=[]
    for arm in ARMS:
        spec=p['outputs'][arm]; summary=read(contained(root,spec['summary'])); prov=read(contained(root,spec['provenance']))
        history=read(contained(root,spec['history'])); trace=read(contained(root,spec['trace']))
        need(receipt['arms'].get(arm)==summary,'Phase/arm receipt mismatch')
        for key,field in [('checkpoint','checkpoint_sha256'),('predictions','predictions_sha256'),('trace','trace_sha256')]:
            need(sha(contained(root,spec[key]))==summary[field],'Final artifact hash mismatch: '+arm+'/'+key)
        need(not list(contained(root,spec['summary']).parent.glob('gold*')),'Unexpected Gold artifact')
        execution[arm]=validate_arm(summary,prov,history,trace,p,arm,reg['protocol_sha256'])
        need(summary['initial_state_sha256']==pilot['arms'][arm]['initial_state_sha256'],'Pilot/full initialization differs')
        traces[arm]=trace; states.append((prov['initial_state_sha256'],prov['base_initial_state_sha256']))
    need(traces['control']==traces['treatment'] and states[0]==states[1],'Paired RNG/input/initialization differs')
    execution.update(paired_rng_trace_exact=True,pilot_supervised_seconds=pilot_time,full_supervised_seconds=full_time)
    return execution,{relative(root,path):sha(path) for path in paths}

def prepare(protocol,protocol_sha256,execution_plan,root=ROOT):
    root=Path(root); protocol_path=contained(root,protocol)
    need(sha(protocol_path)==protocol_sha256,'Protocol SHA mismatch')
    p=read(protocol_path); parent_path=contained(root,p['parent_protocol']['path'])
    need(sha(parent_path)==p['parent_protocol']['sha256'],'Parent protocol changed')
    parent=read(parent_path); base.validate_protocol(parent); infer=inference_contract(p); validate_protocol(p,parent)
    # Pre-outcome freeze must follow the explicit FP32 attention correction.
    helper_source=contained(root,'tools/exp085_relative_position.py').read_text(encoding='utf-8')
    need('enabled=False' in helper_source and 'autocast' in helper_source,'Explicit disabled-autocast correction required before freeze')
    ordered,_=base.cohort_metadata(root,parent)
    plan_path=contained(root,execution_plan); plan=read(plan_path)
    need(plan['protocol_sha256']==protocol_sha256,'Execution plan protocol differs')
    argv=plan['phase_commands']['full']['argv']; tag=argv[argv.index('--tag')+1]; command=argv[argv.index('--')+1:]
    pilot_dir=command[command.index('--pilot-dir')+1]; decision_path=command[command.index('--root-decision')+1]
    phase_output=relative(root,contained(root,p['outputs']['control']['summary']).parent.parent)
    need(not (contained(root,phase_output)/'receipt.json').exists()
         and not all(contained(root,p['outputs'][a]['summary']).exists() for a in ARMS),'Freeze before both final fits complete')
    original_analysis_path='docs/handoffs/evidence/2026-10/2026-10-04-exp084-report-tool/evaluation-v1/analysis-plan.json'
    original_analysis=read(contained(root,original_analysis_path))
    references={}; reference_hashes={}
    for name,arm in [('B_original','B'),('RN_clip','RN_clip')]:
        spec=parent['arms'][arm]; summary=read(contained(root,spec['outputs']['summary'])); prov=read(contained(root,spec['outputs']['provenance']))
        base.validate_execution(summary,prov,parent,arm,p['parent_protocol']['sha256'])
        references[name]=spec
        for path in spec['outputs'].values(): reference_hashes[path]=original_analysis['output_hashes'][path]
    need(reference_hashes[p['initial_checkpoint']['path']]==p['initial_checkpoint']['sha256'],'Initial B checkpoint differs from completed evaluated reference')
    bindings={relative(root,protocol_path):protocol_sha256,**p['source_bindings'],BASE_PATH:BASE_SHA,
              original_analysis_path:sha(contained(root,original_analysis_path)),relative(root,plan_path):sha(plan_path),
              **{'training/src/'+k:v for k,v in p['training_code_sha256'].items()}}
    for name in ['tools/exp085_relative_position_report.py','tests/test_exp085_relative_position_report.py']: bindings[name]=sha(contained(root,name))
    base.verify_bindings(root,bindings)
    reg=dict(schema=SCHEMA,prepared_at=datetime.now(timezone.utc).isoformat(),protocol=relative(root,protocol_path),protocol_sha256=protocol_sha256,
             source_bindings=bindings,input_bindings=p['input_bindings'],reference_hashes=reference_hashes,references=references,
             phase_output=phase_output,full_resource=phase_output+'_RESOURCE/receipt.json',pilot_output=relative(root,pilot_dir),
             pilot_resource=relative(root,pilot_dir)+'_RESOURCE/receipt.json',root_decision=relative(root,decision_path),
             job={'tag':tag,'logs':'artifacts/logs/'+tag,'command':command},targets=TARGETS,arms=list(ARMS),contrasts=CONTRASTS,
             inference_contract=infer,reference_counts=p['reference_counts'],expected_ids=p['expected_ids'],
             bootstrap={'reps':500,'seed':20261004,'unit':'component_id','shared_complete12_draws':True},
             output=EVIDENCE+'/evaluation-v1',limits=LIMITS,
             scope='Metadata-only freeze; no new predictions, labels values, Gold, models, GPU or EXP084 re-evaluation.')
    destination=contained(root,EVIDENCE); destination.mkdir(parents=True,exist_ok=True)
    write_new(destination/'registration.json',reg); return reg

def run(registration,registration_sha256,root=ROOT):
    root=Path(root); registration_path=contained(root,registration)
    need(sha(registration_path)==registration_sha256,'Registration hash mismatch')
    reg=read(registration_path)
    need(reg['schema']==SCHEMA and reg['targets']==TARGETS and reg['arms']==list(ARMS)
         and reg['contrasts']=={k:list(v) for k,v in CONTRASTS.items()} and reg['bootstrap']=={'reps':500,'seed':20261004,'unit':'component_id','shared_complete12_draws':True}
         and reg['output']==EVIDENCE+'/evaluation-v1','Analysis contract changed')
    out=contained(root,reg['output']); need(not out.exists(),'One-shot output already exists')
    base.verify_bindings(root,reg['source_bindings']); p=read(contained(root,reg['protocol']))
    parent=read(contained(root,p['parent_protocol']['path'])); validate_protocol(p,parent)
    ordered,groups=base.cohort_metadata(root,parent)
    execution,hashes=require_completed(root,reg,p)
    # Completion is established before touching reference/new outcome values.
    base.verify_bindings(root,{**reg['input_bindings'],**reg['reference_hashes']})
    for name,arm in [('B_original','B'),('RN_clip','RN_clip')]:
        spec=reg['references'][name]; job=contained(root,spec['logs'])
        need((job/'SUCCESS').is_file() and not (job/'FAILED').exists(),'Original reference job incomplete')
        base.validate_execution(read(contained(root,spec['outputs']['summary'])),read(contained(root,spec['outputs']['provenance'])),parent,arm,p['parent_protocol']['sha256'])
    out.mkdir(parents=True,exist_ok=False)
    try:
        write_new(out/'analysis-plan.json',dict(registration_sha256=registration_sha256,output_hashes=hashes,execution=execution,contrasts=CONTRASTS,limits=LIMITS))
        ids=ordered['val']; y=base.load_reference(root,parent,ids)
        predictions={a:base.predictions(contained(root,p['outputs'][a]['predictions']),ids) for a in ARMS}
        refs={a:base.predictions(contained(root,reg['references'][a]['outputs']['predictions']),ids) for a in ['B_original','RN_clip']}
        scores=scores_for(predictions['control'],predictions['treatment'],refs['B_original'],refs['RN_clip'])
        rows=target_rows(y,scores); totals=aggregate(rows)
        ci=paired_bootstrap(y,scores,groups.loc[ids,'component_id'].to_numpy())
        diagnostics=[]; selected=groups.loc[ids]
        for field in p['groups']['subgroup_columns']:
            vals=selected[field].replace('','unknown').to_numpy(dtype=str)
            for value in sorted(set(vals)):
                take=vals==value; rr=target_rows(y[take],{k:v[take] for k,v in scores.items()})
                diagnostics.append(dict(field=field,group=value,studies=int(take.sum()),targets=rr,aggregate=aggregate(rr),ci=None))
        base.verify_bindings(root,{**reg['source_bindings'],**reg['input_bindings'],**reg['reference_hashes'],**hashes})
        need(sha(registration_path)==registration_sha256,'Registration changed during evaluation')
        report=dict(schema=SCHEMA,evaluated_at=datetime.now(timezone.utc).isoformat(),registration_sha256=registration_sha256,
                    protocol_sha256=reg['protocol_sha256'],source_bindings=reg['source_bindings'],output_hashes=hashes,
                    reference_hashes=reg['reference_hashes'],reference_inference_contract=reg['inference_contract'],
                    studies=len(ids),targets=rows,aggregate=totals,conditional_intervals=ci,groups=diagnostics,execution=execution,
                    evaluation_environment=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__),
                    limits=LIMITS,decision='Root review required; no automatic adoption/ratio/target routing, follow-up training or submission.')
        write_new(out/'report.json',report); pd.json_normalize(rows).to_csv(out/'targets.csv',index=False,mode='x')
        write_new(out/'SUCCESS.json',dict(report_sha256=sha(out/'report.json'),registration_sha256=registration_sha256))
        return dict(output=reg['output'],aggregate=totals,valid_bootstrap_draws=ci['valid'],invalid=ci['invalid'])
    except Exception as exc:
        write_new(out/'failure.json',{'error':str(exc),'exception_type':type(exc).__name__,'rule':'Preserve one-shot failure; root reviews any new registration/version.'}); raise

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    for name in ['protocol','protocol-sha256','execution-plan']: p.add_argument('--'+name,required=True)
    r=sub.add_parser('run')
    for name in ['registration','registration-sha256']:r.add_argument('--'+name,required=True)
    args=parser.parse_args()
    result=prepare(args.protocol,args.protocol_sha256,args.execution_plan) if args.action=='prepare' else run(args.registration,args.registration_sha256)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
