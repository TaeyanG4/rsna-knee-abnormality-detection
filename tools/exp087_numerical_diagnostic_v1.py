"""Passive EXP087 numerical-prefix diagnostic guard and additive telemetry.

prepare reads metadata only. run requires a new root decision and gpu_job.
No original protocol/source is edited and no numerical recovery is offered.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import exp087_training as base
from tools.exp087_training import (worker_init,read_rows,validate_membership,validate_label_contract,
    loss_weights,install_gold_blocker,uid_hash,relpath,sha256,read,CODE_NAMES,ENV_KEYS,THREAD_ENV,
    validate_pretraining,trace_batch,new_trace,finish_trace,stop_loader)

EVIDENCE='docs/handoffs/evidence/2026-10/2026-10-04-exp087-numerical-diagnostic-v1'
ORIGINAL='artifacts/experiments/EXP087_COVERAGE_B8_PROTOCOL_V2/manifest.json'
PROTOCOL=EVIDENCE+'/protocol.json'
TAG='EXP087_NUMERICAL_DIAGNOSTIC_V1'
OUTPUT='artifacts/experiments/'+TAG
RUNNER='training/runners/exp087_numerical_diagnostic_v1.py'
SCHEMA='exp087-numerical-prefix-v1'
MAX_ATTEMPTS=200
OUTPUT_CAP=16*1024**2
RESERVE=64*1024


def output_bytes(output):
    total=0
    for path in Path(output).rglob('*'):
        try:info=path.stat()
        except FileNotFoundError:continue
        if stat.S_ISREG(info.st_mode):total+=info.st_size
    return total


def safe_number(value):
    x=float(value)
    return x if math.isfinite(x) else ('NaN' if math.isnan(x) else ('+Infinity' if x>0 else '-Infinity'))


def durable_json(path,value,*,replace=False):
    path=Path(path);data=(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8')
    if output_bytes(path.parent)+len(data)>OUTPUT_CAP-RESERVE:raise RuntimeError('diagnostic telemetry output cap')
    target=path.with_suffix(path.suffix+'.tmp') if replace else path
    with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    if replace:os.replace(target,path)


def journal(output,event):
    path=Path(output)/'attempts.jsonl';data=(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n').encode()
    if output_bytes(output)+len(data)>OUTPUT_CAP-RESERVE:raise RuntimeError('diagnostic telemetry output cap')
    with path.open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())


def current(output,state,stage,**fields):
    state.update(stage=stage,**fields)
    durable_json(Path(output)/'current-attempt.json',state,replace=True)


def begin_attempt(output,state,attempt,updates,epoch,batch,uids,input_finite,scale):
    last=state.get('last_completed')
    state.clear();state.update(schema='exp087-numerical-attempt-v1',attempt_1based=attempt,
        optimizer_updates_before=updates,epoch_0based=epoch,batch_0based=batch,
        ordered_batch_uid_sha256=hashlib.sha256('\n'.join(uids).encode()).hexdigest(),batch_size=len(uids),
        input_finite=bool(input_finite),logits_finite=None,loss_finite=None,loss=None,
        scale_before=safe_number(scale),last_completed=last)
    current(output,state,'before_forward')


def completed_attempt(output,state,updates,scheduler,scale):
    last=dict(attempt_1based=state['attempt_1based'],optimizer_updates=updates,scheduler_steps=scheduler,
        epoch_0based=state['epoch_0based'],batch_0based=state['batch_0based'],
        ordered_batch_uid_sha256=state['ordered_batch_uid_sha256'])
    current(output,state,'completed',last_completed=last,optimizer_updates_after=updates,scale_after=safe_number(scale))
    journal(output,state)


def found_inf_snapshot(scaler,opt):
    states=scaler._per_optimizer_states.get(id(opt),{})
    found=states.get('found_inf_per_device',{})
    return {str(device):safe_number(value.detach().item()) for device,value in found.items()}


def assert_no_found_inf(scaler,opt):
    found=found_inf_snapshot(scaler,opt)
    if not found or any(v!=0 for v in found.values()):
        raise RuntimeError('diagnostic found_inf guard: no optimizer/scaler update allowed')


def gradient_statistics(named_parameters):
    """Read-only failure-only aggregates. No tensor or parameter writeback."""
    import torch
    rows=[];native_norms=[];ref_norms=[];total=finite_total=0;missing=0
    for name,param in named_parameters:
        if param.grad is None:missing+=1;continue
        g=param.grad.detach();finite=torch.isfinite(g);n=g.numel();nf=int(finite.sum().item())
        values=g[finite]
        ref=float(torch.linalg.vector_norm(values.to(dtype=torch.float64),2).item()) if nf else 0.
        native=torch.linalg.vector_norm(g,2);native_norms.append(native);ref_norms.append(ref)
        rows.append(dict(name=name,dtype=str(g.dtype),device=str(g.device),count=n,finite=nf,nonfinite=n-nf,
            nan=int(torch.isnan(g).sum().item()),positive_inf=int(torch.isposinf(g).sum().item()),
            negative_inf=int(torch.isneginf(g).sum().item()),
            max_finite_abs=safe_number(values.abs().max().item()) if nf else None,
            native_l2=safe_number(native.item()),finite_elements_l2_fp64=safe_number(ref)))
        total+=n;finite_total+=nf
    native_total=torch.linalg.vector_norm(torch.stack(native_norms),2).item() if native_norms else 0.
    finite_ref=math.hypot(*ref_norms)
    return dict(schema='exp087-failure-gradient-stats-v1',writeback=0,fp64_on_failure_only=True,
        native_norm_method='torch.linalg.vector_norm per gradient then stacked vector_norm; observational, not optimizer clipping',
        parameters_with_grad=len(rows),parameters_without_grad=missing,count=total,finite=finite_total,
        nonfinite=total-finite_total,native_total_l2=safe_number(native_total),
        all_elements_l2_fp64=safe_number(finite_ref) if total==finite_total else None,
        finite_elements_l2_fp64=safe_number(finite_ref),parameters=rows)


def capture_failure(output,state,exc,model,opt,scaler):
    """Never replaces the original exception; caller bare-raises it."""
    record=dict(status='CHILD_FAILURE_OBSERVED',exception_type=type(exc).__name__,exception=str(exc),
        current=dict(state),no_recovery=True,optimizer_step_after_failure=False,
        scheduler_step_after_failure=False,scaler_update_after_failure=False)
    try:record['scale']=safe_number(scaler.get_scale());record['found_inf']=found_inf_snapshot(scaler,opt)
    except BaseException as capture:record['scaler_capture_error']=repr(capture)
    try:record['gradients']=gradient_statistics(model.named_parameters())
    except BaseException as capture:record['gradient_capture_error']=repr(capture)
    try:durable_json(Path(output)/'numerical-failure.json',record)
    except BaseException as capture:print('failure telemetry write error: '+repr(capture),file=sys.stderr,flush=True)


def finish_diagnostic(output,state,trace,loader,protocol_sha):
    if state.get('attempt_1based')!=MAX_ATTEMPTS or state.get('optimizer_updates_after')!=MAX_ATTEMPTS:
        raise ValueError('diagnostic bounded prefix mismatch')
    result=finish_trace(trace,loader,output)
    durable_json(Path(output)/'diagnostic-receipt.json',dict(status='BOUNDED_PREFIX_COMPLETE_NO_FAILURE_OBSERVED',
        attempts=MAX_ATTEMPTS,optimizer_updates=MAX_ATTEMPTS,scheduler_steps=result['scheduler_steps'],
        scheduler_horizon=3448,protocol_sha256=protocol_sha,training_complete=False,
        no_checkpoint=True,no_validation_inference=True,no_score=True,no_retry=True))


def arm_for(protocol,tag):
    if tag!=TAG or protocol['schema']!=SCHEMA:raise ValueError('diagnostic identity changed')
    return protocol['arms']['diagnostic']


def validate_protocol(p,protocol_path,expected_sha):
    if sha256(protocol_path)!=expected_sha or p['schema']!=SCHEMA:raise ValueError('diagnostic protocol differs')
    for path,digest in p['diagnostic_source_bindings'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('diagnostic binding changed: '+path)
    old=read(ROOT/ORIGINAL)
    if p['original_protocol']!=ORIGINAL or p['original_protocol_sha256']!=sha256(ROOT/ORIGINAL):
        raise ValueError('original frozen protocol differs')
    base.validate_bindings(old)
    for k in ('paths','ordered_ids','expected_runtime','environment','initial_assets','training_code_sha256','coverage_manifest_sha256'):
        if p[k]!=old[k]:raise ValueError('original recipe/input contract changed: '+k)
    expected={**old['arms']['full']['expected_args'],'mode':'diagnostic'}
    if p['arms']['diagnostic']['expected_args']!=expected:raise ValueError('diagnostic arguments changed')
    for key in ('initial_state_sha256','loss_policy'):
        if p['arms']['diagnostic'][key]!=old['arms']['full'][key]:raise ValueError('original initialization/loss changed')
    if p['budget']!=dict(seconds=240,attempts=200,retry=0,rss_bytes=24*1024**3,output_bytes=OUTPUT_CAP,workers=12,internal=1):
        raise ValueError('diagnostic budget changed')


def validate_arguments(args,root):
    p=read(args.protocol);validate_protocol(p,args.protocol,os.environ.get('EXP087_DIAGNOSTIC_PROTOCOL_SHA256'))
    arm=arm_for(p,args.tag)
    for k,v in arm['expected_args'].items():
        if getattr(args,k,None)!=v:raise ValueError('diagnostic frozen argument changed: '+k)
    for k,v in p['paths'].items():
        if Path(getattr(args,k)).resolve()!=relpath(root,v).resolve():raise ValueError('input path differs: '+k)
    for k,v in {**THREAD_ENV,'HF_HUB_OFFLINE':'1'}.items():
        if os.environ.get(k)!=v:raise ValueError('diagnostic environment differs: '+k)
    phase=ROOT/OUTPUT/'phase-authorization.json'
    if os.environ.get('EXP087_DIAGNOSTIC_PHASE_SHA256')!=sha256(phase):raise ValueError('diagnostic phase authorization absent')
    ordered=read(ROOT/p['ordered_ids']['path'])
    return p,ordered


def runner_argv(p,protocol_path):
    argv=[sys.executable,str(ROOT/RUNNER),'--protocol',str(protocol_path),'--tag',TAG]
    for k,v in {**p['arms']['diagnostic']['expected_args'],**{k:str(ROOT/v) for k,v in p['paths'].items()}}.items():
        if isinstance(v,bool):
            if v:argv.append('--'+k.replace('_','-'))
        else:argv+=['--'+k.replace('_','-'),str(v)]
    return argv


def prepare():
    import copy
    old=read(ROOT/ORIGINAL);p=copy.deepcopy(old)
    p.update(schema=SCHEMA,status='CPU_PREPARED_EXECUTION_NOT_AUTHORIZED',created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        original_protocol=ORIGINAL,original_protocol_sha256=sha256(ROOT/ORIGINAL),
        budget=dict(seconds=240,attempts=200,retry=0,rss_bytes=24*1024**3,output_bytes=OUTPUT_CAP,workers=12,internal=1),
        purpose='unchanged recipe numerical prefix; no performance evidence and no recovery',output=OUTPUT)
    arm=copy.deepcopy(old['arms']['full']);arm.update(id='diagnostic',tag=TAG,expected_attempts=MAX_ATTEMPTS,outputs={})
    arm['expected_args']['mode']='diagnostic';p['arms']={'diagnostic':arm}
    paths=[ORIGINAL,RUNNER,'tools/exp087_numerical_diagnostic_v1.py','tests/test_exp087_numerical_diagnostic_v1.py',
        EVIDENCE+'/root-cpu-decision.json','docs/handoffs/evidence/2026-10/2026-10-04-exp087-training-v1/root-static-failure-triage.json',
        '.local/exp087-coverage/windows-gate-v1/receipt.json','.local/exp087-coverage/windows-gate-v1/resource-supervisor.json',
        'artifacts/experiments/EXP087_COVERAGE_B8_G1_PILOT/pilot-receipt.json','artifacts/experiments/EXP087_COVERAGE_B8_G1_PILOT/resource-supervisor.json',
        'artifacts/experiments/EXP087_COVERAGE_B8_G1/resource-supervisor.json','artifacts/experiments/EXP087_COVERAGE_B8_G1/provenance.json']
    gate=ROOT/'.local/exp087-coverage/windows-gate-v1'
    pilot=ROOT/'artifacts/experiments/EXP087_COVERAGE_B8_G1_PILOT'
    if (read(gate/'receipt.json')['status']!='PASS_WINDOWS_INPUT_INTEGRATION' or read(gate/'resource-supervisor.json')['status']!='PASS'
        or read(pilot/'resource-supervisor.json')['status']!='PASS' or read(pilot/'pilot-receipt.json')['updates']!=32
        or read(ROOT/'artifacts/experiments/EXP087_COVERAGE_B8_G1/resource-supervisor.json')['status']!='FAIL'):
        raise ValueError('prior Windows/pilot PASS and original full FAIL required')
    p['diagnostic_source_bindings']={x:sha256(ROOT/x) for x in paths}
    base.dump(ROOT/PROTOCOL,p);digest=sha256(ROOT/PROTOCOL)
    command=[sys.executable,str(ROOT/'tools/gpu_job.py'),'--tag',TAG,'--',sys.executable,str(ROOT/'tools/exp087_numerical_diagnostic_v1.py'),
        'run','--protocol',str(ROOT/PROTOCOL),'--protocol-sha256',digest,'--decision',str(ROOT/EVIDENCE/'root-execution-decision.json'),'--execute']
    base.dump(ROOT/EVIDENCE/'execution-plan.json',dict(status='ROOT_EXACT_DECISION_REQUIRED',protocol_sha256=digest,cwd=str(ROOT),
        argv=command,child_argv=runner_argv(p,ROOT/PROTOCOL),environment={**THREAD_ENV,'HF_HUB_OFFLINE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'},budget=p['budget']))
    print(json.dumps(dict(status=p['status'],protocol_sha256=digest)))


def supervise(command,output,p,env,digest):
    import psutil
    started=time.monotonic();proc=None;tracked={};failure=None;killed=None;peak=0;peak_output=0;cleanup=[];killed_pids=[]
    budget=p['budget']
    try:
        with (output/'supervisor.stdout.log').open('x',encoding='utf-8') as stdout,(output/'supervisor.stderr.log').open('x',encoding='utf-8') as stderr:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            base.dump(output/'supervisor-job.json',dict(pid=proc.pid,argv=command,protocol_sha256=digest,budget=budget,
                phase_authorization_sha256=env['EXP087_DIAGNOSTIC_PHASE_SHA256'],started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            while proc.poll() is None:
                try:
                    child=psutil.Process(proc.pid);tree=[child]+child.children(recursive=True);tracked.update({q.pid:q for q in tree})
                    peak=max(peak,psutil.Process(os.getpid()).memory_info().rss+sum(q.memory_info().rss for q in tree if q.is_running()))
                except psutil.NoSuchProcess:pass
                peak_output=max(peak_output,output_bytes(output))
                if time.monotonic()-started>=budget['seconds']:killed='deadline';break
                if peak>budget['rss_bytes']:killed='tree_rss';break
                if peak_output>budget['output_bytes']-RESERVE:killed='output_cap';break
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
    elapsed=time.monotonic()-started;peak_output=max(peak_output,output_bytes(output))
    status='PASS' if proc is not None and proc.returncode==0 and not failure and not killed and not cleanup and not killed_pids and elapsed<budget['seconds'] and peak_output<=budget['output_bytes']-RESERVE else 'FAIL'
    record=dict(status=status,returncode=None if proc is None else proc.returncode,error=failure,killed=killed,cleanup_errors=cleanup,
        cleanup_killed_pids=killed_pids,wall_seconds=elapsed,peak_tree_RSS=peak,peak_output_bytes=peak_output,protocol_sha256=digest,
        numerical_failure_file_present=(output/'numerical-failure.json').is_file(),training_complete=False,
        phase_authorization_sha256=env['EXP087_DIAGNOSTIC_PHASE_SHA256'],finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    base.dump(output/'resource-supervisor.json',record)
    if status!='PASS':raise RuntimeError('diagnostic child/resource failure preserved; no retry')


def run(protocol_path,expected_sha,decision_path,execute):
    if not execute:raise ValueError('explicit execution required')
    p=read(protocol_path);validate_protocol(p,protocol_path,expected_sha);decision=read(decision_path)
    if (decision.get('status')!='AUTHORIZE_EXACT_NUMERICAL_DIAGNOSTIC' or decision.get('protocol_sha256')!=expected_sha
        or decision.get('execution_plan_sha256')!=sha256(ROOT/EVIDENCE/'execution-plan.json') or decision.get('budget')!=p['budget']):
        raise ValueError('root exact diagnostic decision absent/mismatch')
    output=ROOT/OUTPUT;output.mkdir(parents=True,exist_ok=False)
    base.dump(output/'phase-authorization.json',dict(decision=decision,root_decision_path=str(decision_path),
        root_decision_sha256=sha256(decision_path),protocol_sha256=expected_sha))
    env={**os.environ,**THREAD_ENV,'HF_HUB_OFFLINE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8',
        'EXP087_DIAGNOSTIC_PROTOCOL_SHA256':expected_sha,'EXP087_DIAGNOSTIC_PHASE_SHA256':sha256(output/'phase-authorization.json')}
    supervise(runner_argv(p,protocol_path),output,p,env,expected_sha)


def main():
    ap=argparse.ArgumentParser(description=__doc__);s=ap.add_subparsers(dest='command',required=True);s.add_parser('prepare')
    r=s.add_parser('run');r.add_argument('--protocol',type=Path,required=True);r.add_argument('--protocol-sha256',required=True)
    r.add_argument('--decision',type=Path,required=True);r.add_argument('--execute',action='store_true');a=ap.parse_args()
    if a.command=='prepare':prepare()
    else:run(a.protocol,a.protocol_sha256,a.decision,a.execute)

if __name__=='__main__':main()
