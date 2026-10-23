"""Single frozen EXP087 coverage/B8 contract. Import is passive; no model access.

prepare is metadata-only. All real CPU/GPU executions require explicit modes,
exact protocol hashes and fresh output directories. No retries or control refit.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.exp084_group_loss import (worker_init,read_rows,validate_membership,validate_label_contract,
    loss_weights,install_gold_blocker,uid_hash,relpath,sha256,read,CODE_NAMES,ENV_KEYS,THREAD_ENV)

SCHEMA='exp087-coverage-b8-single-v1'
PROTOCOL='artifacts/experiments/EXP087_COVERAGE_B8_PROTOCOL_V2'
BASE='artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/manifest.json'
STAGE='.local/exp087-coverage/stage-v1'
SE='docs/handoffs/evidence/2026-10/2026-10-04-exp087-coverage-scale-v1'
EVIDENCE='docs/handoffs/evidence/2026-10/2026-10-04-exp087-training-v1'
TAGS={'pilot':'EXP087_COVERAGE_B8_G1_PILOT','full':'EXP087_COVERAGE_B8_G1'}
MANIFEST_SHA='3f1541aeb190ba49164e7bc3d067d2b4c1bc0142b87f44acd0ca9a4b1e562371'
METRICS='deferred_single_fixed_evaluation'


def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def stage_acceptance():
    sup=read(ROOT/STAGE/'supervisor.json');receipt=read(ROOT/STAGE/'receipt.json');m=read(ROOT/STAGE/'manifest.json')
    rr=read(ROOT/SE/'root-stage-result-review.json');ind=read(ROOT/SE/'result-independent-review.json')
    accepted=read(ROOT/SE/'root-stage-acceptance.json')
    if (sup['status']!='PASS' or sup['returncode']!=0 or sup['killed'] is not None or sup['error'] is not None
        or sup['cleanup_killed_pids'] or sup['cleanup_errors'] or sup['wall_seconds']>=900
        or sup['peak_process_tree_rss']>8*1024**3 or (ROOT/STAGE/'FAIL.json').exists()
        or receipt['status']!='CPU_STAGING_COMPLETE' or receipt['manifest_sha256']!=MANIFEST_SHA
        or sha256(ROOT/STAGE/'manifest.json')!=MANIFEST_SHA or len(receipt['entries'])!=315
        or m['status']!='STAGED_INPUTS_ONLY_NO_TRAINING' or not all(rr['checks'].values())
        or ind['status']!='PASS_INPUT_STAGE_ACCEPTANCE_ONLY' or accepted['status']!='PASS_INPUT_STAGE_ONLY'):
        raise ValueError('completed stage/root/independent acceptance mismatch')
    if (len(m['entries'])!=315 or len({r['uid'] for r in m['entries']})!=315
        or sum(r['split']=='train' for r in m['entries'])!=252 or sum(r['split']=='val' for r in m['entries'])!=63):
        raise ValueError('stage315 population mismatch')
    return m


def prepare():
    destination=ROOT/PROTOCOL;destination.mkdir(parents=True,exist_ok=False)
    baseline=read(ROOT/BASE);m=stage_acceptance();b=baseline['arms']['B']
    ordered=read(ROOT/baseline['ordered_ids']['path'])
    if ordered!=m['membership']:raise ValueError('stage G1 differs')
    accepted=[f'{STAGE}/supervisor.json',f'{STAGE}/receipt.json',f'{STAGE}/manifest.json',
        f'{SE}/root-stage-result-review.json',f'{SE}/result-independent-review.json',f'{SE}/root-stage-acceptance.json',f'{EVIDENCE}/root-cpu-decision.json']
    code=['tools/exp087_training.py','tools/exp087_windows_gate.py','training/runners/exp087_coverage_b8.py',
        'training/src/exp087_coverage_data.py','tools/exp087_series_coverage.py','tools/exp084_group_loss.py',
        'tests/test_exp087_training.py']
    p={k:baseline[k] for k in ('paths','groups','labels','label_hash','eval_label_hash','split_hash','ordered_ids','expected_ids',
        'excluded_uids','loss','corpus_manifest_sha256','expected_runtime','environment','training_code_sha256','initial_assets',
        'split','reference_counts','macro12_valid')}
    p.update(schema=SCHEMA,created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),status='REGISTERED_NO_EXECUTION',
        source_bindings={**baseline['source_bindings'],**{s:sha256(ROOT/s) for s in code+accepted+[BASE]}},
        preparation_read_scope='metadata only; historical label/model bindings reused without opening values or weights',
        acceptance_bindings={s:sha256(ROOT/s) for s in accepted},coverage_manifest=f'{STAGE}/manifest.json',coverage_manifest_sha256=MANIFEST_SHA,
        coverage_adapter='training/src/exp087_coverage_data.py',coverage_helper='tools/exp087_series_coverage.py',
        metric_access=METRICS,baseline_comparator='EXP084_B_G1',arms={},
        windows_gate=dict(output='.local/exp087-coverage/windows-gate-v1',seconds=180,rss_bytes=24*1024**3,workers=12,internal=1),
        training_budgets=dict(pilot_seconds=180,full_seconds=3600,total_GPU_seconds=3780,rss_bytes=24*1024**3,retry=0),
        projection='1.3*mean(last24wholeupdates)*3448+300<=3600; full clean generic initialization',
        evaluation=dict(primary='macro12(candidate)-macro12(EXP084_B_G1)',bootstrap_unit='component_id',bootstrap_reps=500,bootstrap_seed=20261004,no_rank_or_blend=True),
        reference_outputs=baseline['arms']['B']['outputs'],validation='same888 exposed weak development studies/873components; no private/patient/.943 claim')
    for mode,tag in TAGS.items():
        arm=dict(b);arm.update(id=mode,tag=tag,expected_attempts=32 if mode=='pilot' else 3448)
        arm['expected_args']={**b['expected_args'],'mode':mode}
        arm['outputs']={k:f'artifacts/experiments/{tag}/{name}' for k,name in dict(summary='summary.json',provenance='provenance.json',predictions='oof_best.csv',checkpoint='model_best.pt',history='history.json').items()}
        p['arms'][mode]=arm
    # Gate fixtures use stored policy opportunities only, no images/labels.
    potentials=read(ROOT/'docs/handoffs/evidence/2026-10/2026-10-04-exp087-coverage-pilot/policy-count-v1/potential-studies.json')
    present={r['uid'] for r in m['entries'] if r['split']=='train'}
    active=next(r['uid'] for r in potentials if r['split']=='train' and r['train_epoch_potential_replacements'][0]>0)
    zero=next(r['uid'] for r in potentials if r['split']=='train' and r['train_epoch_potential_replacements'][0]==0)
    missing=next(u for u in ordered['train'] if u not in present)
    chosen=[active,zero,missing]
    chosen+= [u for u in ordered['train'] if u not in chosen][:5]
    p['windows_gate']['uids']=chosen;p['windows_gate']['indices']=[ordered['train'].index(u) for u in chosen]
    p['windows_gate']['cases']=dict(eligible_active=active,eligible_zero_epoch0=zero,missing=missing)
    dump(destination/'manifest.json',p)
    protocol_sha=sha256(destination/'manifest.json');plans={}
    env={**THREAD_ENV,'HF_HUB_OFFLINE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','KNEE_WORKSPACE_ROOT':str(ROOT)}
    for mode in ('windows','pilot','full'):
        direct=[sys.executable,str(ROOT/'tools/exp087_training.py'),'run','--protocol',str(destination/'manifest.json'),'--protocol-sha256',protocol_sha,'--mode',mode,'--decision',str(ROOT/EVIDENCE/('root-'+mode+'-execution-decision.json')),'--execute']
        plans[mode]=dict(status='ROOT_EXECUTION_DECISION_REQUIRED',mode=mode,cwd=str(ROOT),environment={**env,**({'CUDA_VISIBLE_DEVICES':''} if mode=='windows' else {})},
            argv=direct if mode=='windows' else [sys.executable,str(ROOT/'tools/gpu_job.py'),'--tag',TAGS[mode],'--',*direct])
    dump(destination/'execution-plans.json',dict(protocol_sha256=protocol_sha,plans=plans))
    print(json.dumps(dict(status='REGISTERED_NO_EXECUTION',protocol_sha256=protocol_sha,output=PROTOCOL)))


def arm_for(protocol,tag):
    rows=[a for a in protocol['arms'].values() if a['tag']==tag]
    if len(rows)!=1:raise ValueError('unregistered single candidate tag')
    return rows[0]


def validate_bindings(protocol,root=ROOT,*,cpu=False):
    if protocol['schema']!=SCHEMA or set(protocol['arms'])!={'pilot','full'}:raise ValueError('single-candidate protocol differs')
    # Actual CPU gate never opens labels/model assets. Their hashes remain frozen
    # and are checked in the separately authorized GPU entrypoint.
    for path,expected in protocol['source_bindings'].items():
        if cpu and (path.startswith('storage/labels/') or path.endswith('.pt') or path.endswith('.safetensors')):continue
        if sha256(relpath(root,path))!=expected:raise ValueError('frozen binding changed: '+path)
    for name,expected in protocol['training_code_sha256'].items():
        if sha256(Path(root)/'training/src'/name)!=expected:raise ValueError('original training source changed')
    stage_acceptance()


def validate_arguments(args,root):
    p=read(args.protocol);validate_bindings(p,root);a=arm_for(p,args.tag)
    for k,v in a['expected_args'].items():
        if getattr(args,k,None)!=v:raise ValueError('frozen candidate argument changed: '+k)
    for k,v in p['paths'].items():
        if Path(getattr(args,k)).resolve()!=relpath(root,v).resolve():raise ValueError('input path differs: '+k)
    for k,v in {**THREAD_ENV,'HF_HUB_OFFLINE':'1'}.items():
        if os.environ.get(k)!=v:raise ValueError('registered environment required: '+k)
    if os.environ.get('EXP087_AUTHORIZED_PROTOCOL_SHA256')!=sha256(args.protocol):raise ValueError('runner requires exact supervisor authorization')
    phase_file=Path(root)/'artifacts/experiments'/args.tag/'phase-authorization.json'
    if os.environ.get('EXP087_PHASE')!=args.mode or os.environ.get('EXP087_PHASE_AUTHORIZATION_SHA256')!=sha256(phase_file):
        raise ValueError('runner phase authorization mismatch')
    ordered=read(Path(root)/p['ordered_ids']['path'])
    for k in ('train','val'):
        if uid_hash(ordered[k])!=p['ordered_ids'][k+'_sha256']:raise ValueError('ordered G1 changed')
    return p,ordered


def validate_pretraining(provenance,env,p,arm):
    expected=dict(code_sha256=p['training_code_sha256'],initial_state_sha256=arm['initial_state_sha256'],
        train_ids_sha256=p['ordered_ids']['train_sha256'],eval_ids_sha256=p['ordered_ids']['val_sha256'],
        train_ids_ordered_sha256=p['ordered_ids']['train_sha256'],eval_ids_ordered_sha256=p['ordered_ids']['val_sha256'],
        corpus_manifest_sha256=p['corpus_manifest_sha256'],checkpoint_selection='fixed-final',gold_access='disabled',
        sampling='sha256(seed,epoch,uid,stream)-v1',loss_policy=arm['loss_policy'],metric_access=METRICS,
        coverage_manifest_sha256=p['coverage_manifest_sha256'],scheduler_total_steps=3448)
    if any(provenance.get(k)!=v for k,v in expected.items()):raise ValueError('pretraining contract differs; optimizer forbidden')
    if any(provenance[k]!=p['expected_runtime'][k] for k in ('runtime','eval_runtime')):raise ValueError('baseline strict runtime differs')
    if any(env.get(k)!=p['environment'][k] for k in ENV_KEYS):raise ValueError('baseline runtime environment differs')


def stop_loader(loader):
    iterator=getattr(loader,'_iterator',None)
    workers=list(getattr(iterator,'_workers',[])) if iterator is not None else []
    if iterator is not None:iterator._shutdown_workers()
    for w in workers:
        w.join(timeout=5)
        if w.is_alive() or w.exitcode not in (0,None):raise RuntimeError('DataLoader worker did not exit cleanly')
    return dict(clean=True,pids=[w.pid for w in workers],exitcodes=[w.exitcode for w in workers])


def trace_batch(trace,uids,epoch,dataset,x):
    from tools.exp087_series_coverage import replacement_map,windows
    from raptor_common import select_centers,study_rng,Corpus
    b=dataset.baseline
    if not hasattr(trace_batch,'corpus'):trace_batch.corpus=Corpus(b.corpus_dir)
    corpus=trace_batch.corpus
    for j,uid in enumerate(uids):
        trace['_uid_hasher'].update((str(epoch)+':'+uid+'\n').encode())
        entry=dataset.store.entry(uid,'train')
        if entry is None or entry['reason'] is not None:continue
        trace['eligible_deliveries']+=1
        c=select_centers(corpus.mask(uid),64,12,corpus.spec,'within-slot-valid',study_rng(42,epoch,uid))
        mapping,_=replacement_map(c,entry['mask'],entry['native_indices']);trace['scheduled_replacements']+=len(mapping)
        if mapping and trace['actual_replacement_tensor_checks']==0:
            expected,r=windows(corpus.volume(uid),corpus.mask(uid),corpus.spec,uid,train=True,epoch=epoch,
                alternate=dataset.store.volume(entry),alt_mask=entry['mask'],native_indices=entry['native_indices'])
            if not __import__('numpy').array_equal(x[j].numpy(),expected):raise ValueError('delivered treatment tensor differs from frozen policy')
            trace['actual_replacement_tensor_checks']+=1


def new_trace():
    return dict(schema='exp087-delivered-input-trace-v1',updates=0,epochs=0,eligible_deliveries=0,scheduled_replacements=0,
        actual_replacement_tensor_checks=0,pilot_whole_update_seconds=[],scheduler_steps=0,_uid_hasher=hashlib.sha256())


def finish_trace(trace,loader,out):
    result={k:v for k,v in trace.items() if k!='_uid_hasher'}
    result['delivered_uid_sha256']=trace['_uid_hasher'].hexdigest();result['worker_cleanup']=stop_loader(loader)
    dump(Path(out)/'input-trace.json',result);return result


def pilot_finish(out,trace,loader,updates,skips,torch):
    if updates!=32 or skips or trace['actual_replacement_tensor_checks']<1:raise ValueError('pilot update/coverage/skip gate failed')
    result=finish_trace(trace,loader,out); times=result['pilot_whole_update_seconds'][-24:]
    if len(times)!=24:raise ValueError('missing whole update timing')
    if result.get('updates')!=32 or result.get('scheduler_steps')!=32:raise ValueError('pilot scheduler/update trace mismatch')
    projection=1.3*sum(times)/24*3448+300
    provenance=read(Path(out)/'provenance.json')
    dump(Path(out)/'pilot-receipt.json',dict(status='PILOT_COMPLETE_NO_FULL_AUTHORIZATION',updates=32,skips=0,scheduler_horizon=3448,
        protocol_sha256=provenance['protocol_sha256'],
        projected_full_seconds=projection,projection_pass=projection<=3600,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        no_checkpoint_saved=True,no_validation_inference=True,no_metric_access=True,
        input_trace_sha256=sha256(Path(out)/'input-trace.json'),provenance_sha256=sha256(Path(out)/'provenance.json')))


def validate_completion(out,p,arm):
    s=read(Path(out)/'summary.json');t=read(Path(out)/'input-trace.json')
    if (s['optimizer_updates']!=3448 or s['skipped_updates']!=0 or s['selected_epoch_1based']!=8 or s['train']!=3455 or s['val']!=888
        or s['weak_oof_macro_auc'] is not None or s['metric_access']!=METRICS or s['provenance']['initial_state_sha256']!=arm['initial_state_sha256']
        or t['updates']!=3448 or t['scheduler_steps']!=3448 or not t['worker_cleanup']['clean'] or t['actual_replacement_tensor_checks']<1):
        raise ValueError('full exact-update/comparator gate failed')
    for name in ('model_best.pt','oof_best.csv','provenance.json','history.json'):
        if not (Path(out)/name).is_file():raise ValueError('missing final artifact '+name)


def write_output_inventory(out):
    names=['summary.json','provenance.json','history.json','model_best.pt','oof_best.csv','input-trace.json']
    dump(Path(out)/'output-inventory.json',dict(schema='exp087-training-output-inventory-v1',files={n:dict(sha256=sha256(Path(out)/n),bytes=(Path(out)/n).stat().st_size) for n in names}))


def require_previous(p,mode,expected_sha):
    if mode in ('pilot','full'):
        gate=ROOT/p['windows_gate']['output'];r=read(gate/'receipt.json');s=read(gate/'resource-supervisor.json')
        if r['status']!='PASS_WINDOWS_INPUT_INTEGRATION' or s['status']!='PASS' or r.get('protocol_sha256')!=expected_sha or s.get('protocol_sha256')!=expected_sha:raise ValueError('Windows gate not accepted')
    if mode=='full':
        d=ROOT/'artifacts/experiments'/TAGS['pilot'];r=read(d/'pilot-receipt.json');s=read(d/'resource-supervisor.json')
        if r['updates']!=32 or r['skips'] or not r['projection_pass'] or s['status']!='PASS' or r.get('protocol_sha256')!=expected_sha or s.get('protocol_sha256')!=expected_sha:raise ValueError('pilot/projection gate not passed')


def runner_argv(p,mode,protocol_path):
    a=p['arms'][mode];argv=[sys.executable,str(ROOT/'training/runners/exp087_coverage_b8.py'),'--protocol',str(protocol_path),'--tag',a['tag']]
    for k,v in {**a['expected_args'],**{k:str(ROOT/v) for k,v in p['paths'].items()}}.items():
        if isinstance(v,bool):
            if v:argv.append('--'+k.replace('_','-'))
        else:argv+=['--'+k.replace('_','-'),str(v)]
    return argv


def supervise(command,output,seconds,rss_cap,env,protocol_sha):
    import psutil
    started=time.monotonic();proc=None;tracked={};failure=None;killed=None;peak=0;cleanup=[];killed_pids=[]
    try:
        with (output/'supervisor.stdout.log').open('x',encoding='utf-8') as stdout,(output/'supervisor.stderr.log').open('x',encoding='utf-8') as stderr:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            dump(output/'supervisor-job.json',dict(pid=proc.pid,argv=command,protocol_sha256=protocol_sha,phase=env['EXP087_PHASE'],
                phase_authorization_sha256=env['EXP087_PHASE_AUTHORIZATION_SHA256'],started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),seconds_cap=seconds,rss_cap=rss_cap,retry=0))
            while proc.poll() is None:
                try:
                    child=psutil.Process(proc.pid);tree=[child]+child.children(recursive=True);tracked.update({q.pid:q for q in tree})
                    peak=max(peak,psutil.Process(os.getpid()).memory_info().rss+sum(q.memory_info().rss for q in tree if q.is_running()))
                except psutil.NoSuchProcess:pass
                if time.monotonic()-started>=seconds or peak>rss_cap:killed='resource_cap';break
                time.sleep(.1)
    except BaseException as exc:failure=f'{type(exc).__name__}: {exc}'
    finally:
        if proc is not None:
            try:
                child=psutil.Process(proc.pid);tracked.update({q.pid:q for q in [child]+child.children(recursive=True)})
            except psutil.NoSuchProcess:pass
            except BaseException as exc:cleanup.append(str(exc))
            for q in reversed(list(tracked.values())):
                try:
                    if q.is_running():q.kill();killed_pids.append(q.pid)
                except psutil.NoSuchProcess:pass
                except BaseException as exc:cleanup.append(str(exc))
            try:
                if proc.poll() is None:proc.kill();killed_pids.append(proc.pid)
                proc.wait(timeout=10)
                _,alive=psutil.wait_procs(list(tracked.values()),timeout=2)
                if alive:cleanup.append('survivors:'+str([q.pid for q in alive]))
            except BaseException as exc:cleanup.append(str(exc))
    elapsed=time.monotonic()-started
    status='PASS' if proc is not None and proc.returncode==0 and not failure and not killed and not cleanup and not killed_pids and elapsed<seconds else 'FAIL'
    record=dict(status=status,returncode=None if proc is None else proc.returncode,error=failure,killed=killed,cleanup_errors=cleanup,cleanup_killed_pids=killed_pids,wall_seconds=elapsed,peak_tree_RSS=peak,protocol_sha256=protocol_sha,
        phase=env['EXP087_PHASE'],phase_authorization_sha256=env['EXP087_PHASE_AUTHORIZATION_SHA256'],finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    try:dump(output/'resource-supervisor.json',record)
    except BaseException:
        print(json.dumps({**record,'status':'FAIL','receipt_write_failed':True}),file=sys.stderr,flush=True);raise
    if status!='PASS':raise RuntimeError('supervised run failed; no retry')


def run(protocol_path,expected_sha,mode,execute,decision_path):
    if not execute or sha256(protocol_path)!=expected_sha:raise ValueError('exact protocol and explicit execution required')
    p=read(protocol_path);validate_bindings(p,cpu=mode=='windows');require_previous(p,mode,expected_sha)
    decision=read(decision_path)
    if decision.get('status')!='AUTHORIZE_EXACT_PHASE' or decision.get('phase')!=mode or decision.get('protocol_sha256')!=expected_sha:
        raise ValueError('root exact phase decision absent/mismatch')
    previous={}
    if mode in ('pilot','full'):
        gate=ROOT/p['windows_gate']['output']
        previous.update({str((gate/n).relative_to(ROOT)).replace('\\','/'):sha256(gate/n) for n in ('receipt.json','resource-supervisor.json')})
    if mode=='full':
        pilot=ROOT/'artifacts/experiments'/TAGS['pilot']
        previous.update({str((pilot/n).relative_to(ROOT)).replace('\\','/'):sha256(pilot/n) for n in ('pilot-receipt.json','resource-supervisor.json','provenance.json','input-trace.json')})
    if decision.get('prior_artifact_sha256',{})!=previous:raise ValueError('root previous-phase binding mismatch')
    output=ROOT/(p['windows_gate']['output'] if mode=='windows' else 'artifacts/experiments/'+TAGS[mode])
    output.mkdir(parents=True,exist_ok=False)
    dump(output/'phase-authorization.json',dict(root_decision_path=str(Path(decision_path).resolve()),root_decision_sha256=sha256(decision_path),decision=decision,
        protocol_sha256=expected_sha,execution_plans_sha256=sha256(ROOT/PROTOCOL/'execution-plans.json'),prior_artifact_sha256=previous))
    env={**os.environ,**THREAD_ENV,'HF_HUB_OFFLINE':'1','EXP087_AUTHORIZED_PROTOCOL_SHA256':expected_sha,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8',
        'EXP087_PHASE':mode,'EXP087_PHASE_AUTHORIZATION_SHA256':sha256(output/'phase-authorization.json')}
    if mode=='windows':
        env['CUDA_VISIBLE_DEVICES']='';argv=[sys.executable,str(ROOT/'tools/exp087_windows_gate.py'),'--protocol',str(protocol_path),'--output',str(output)]
        seconds=p['windows_gate']['seconds'];rss=p['windows_gate']['rss_bytes']
    else:
        argv=runner_argv(p,mode,protocol_path);seconds=p['training_budgets'][mode+'_seconds'];rss=p['training_budgets']['rss_bytes']
    supervise(argv,output,seconds,rss,env,expected_sha)


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');r=sub.add_parser('run');r.add_argument('--protocol',type=Path,required=True);r.add_argument('--protocol-sha256',required=True);r.add_argument('--mode',choices=['windows','pilot','full'],required=True);r.add_argument('--decision',type=Path,required=True);r.add_argument('--execute',action='store_true')
    a=ap.parse_args()
    if a.command=='prepare':prepare()
    else:run(a.protocol,a.protocol_sha256,a.mode,a.execute,a.decision)

if __name__=='__main__':main()
