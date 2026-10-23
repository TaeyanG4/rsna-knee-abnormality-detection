"""v1.0.0: one-shot alternatives; exact fullbag v1.1.0 file-only waiter contract.

Never queries predecessor processes. Both alternatives share one durable selection.
Only B runs this on Vast, after writing a reviewed objective-gate decision.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.l1_rco_k1_next_v1 import backend, TASK, MODES
from tools import job_runtime as runtime

TERMINAL=ROOT/'.local/scratch/astra-b2/fullbag-post-v1/fullbag_terminal_v1.py'
LEDGER=TASK/'selected-once.json'


def terminal():
    spec=importlib.util.spec_from_file_location('_fullbag_terminal',TERMINAL)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision(k,path):
    d=k.read(path)
    k.need(d['schema']=='vast-next-decision/v1' and d['selected']==k.next_mode
           and d['owner']=='B' and d['review_verdict']=='PASS'
           and d['builder']!=d['reviewer'], 'B reviewed selection required')
    for name in ('gate','review_report'):
        p=ROOT/d[name]
        k.need(p.is_file() and k.sha(p)==d[name+'_sha256'], 'decision evidence changed')
    g=k.read(ROOT/d['gate'])
    k.need(g['holdout_rows']==977 and g['status'] in ('SCREEN_PASS_PENDING_REVIEW','HOLD'), 'exact completed 977-row fullbag objective gate required')
    # A non-passing gate cannot authorize the fullbag all-data refit.
    if k.next_mode=='fullbag_all':
        k.need(g['status']=='SCREEN_PASS_PENDING_REVIEW', 'fullbag objective did not pass')
    return d


def require_selection(k):
    k.need(Path(os.environ.get('KNEE_WORKSPACE_ROOT','')).resolve()==k.owner_scope().resolve(), 'canonical GPU0 owner required')
    state=k.read(LEDGER)
    k.need(state['state']=='STARTED' and state['selected']==k.next_mode
           and state['command']==k.run_command() and state['contract_id']==k.read(k.WORK/'registration.json')['contract_id'], 'durable exact selection required')
    t=terminal()
    k.need(t.probe(ROOT,state['predecessor'])=='SUCCESS','predecessor SUCCESS changed')
    k.need(decision(k,Path(state['decision_path']))==state['decision'],'decision changed')


def execute(mode,decision_path,dry_run=False):
    k=backend(mode)
    t=terminal()
    def inspect():
        k.need(not LEDGER.exists(),'alternative already selected; no automatic retry')
        k.need(not (k.OUT/'admission.json').exists(),'consumed admission')
        k.need(not (k.owner_scope()/'artifacts/logs'/k.TAG).exists(),'job already attempted')
        return t.binding(ROOT)
    if dry_run:
        try:
            bound=inspect()
            state=t.probe(ROOT,bound)
            k.need(state=='SUCCESS','predecessor not terminal SUCCESS: '+state)
            decision(k,decision_path)
            reg=k.validate(k.read(k.WORK/'registration.json'))
            admission=k.admit(reg,k.CACHE,k.REVIEW)
            k.need(admission['status']=='CPU_ADMISSIBLE','review/admission pending')
            return dict(status='PLAN_ONLY',selected=mode,command=k.launch_command())
        except (OSError,ValueError,KeyError,TypeError,AssertionError) as exc:
            return dict(status='BLOCKED',selected=mode,reason=str(exc),command=k.launch_command())
    k.need(os.name!='nt','launch is Vast Linux only')
    TASK.mkdir(parents=True,exist_ok=True)
    with runtime.file_lock(LEDGER.with_suffix('.lock'),blocking=False):
        bound=inspect()
        while True:
            state=t.probe(ROOT,bound)
            if state=='SUCCESS': break
            k.need(state=='WAIT','predecessor terminal/stall failure: '+state)
            time.sleep(30)
        chosen=decision(k,decision_path)
        reg=k.validate(k.read(k.WORK/'registration.json'))
        admission=k.admit(reg,k.CACHE,k.REVIEW)
        k.need(admission['status']=='CPU_ADMISSIBLE','admission blocked')
        k.need(t.probe(ROOT,bound)=='SUCCESS','predecessor changed')
        state=dict(version='1.0.0',state='STARTED',selected=mode,predecessor=bound,
                   command=k.run_command(),contract_id=reg['contract_id'],
                   decision_path=str(decision_path.resolve()),decision=chosen,jobs=[])
        runtime.atomic_json(LEDGER,state)
        try:
            from tools import l1_rco_k1_foldseq_v1 as seq
            started=time.time()
            env={**os.environ,'KNEE_WORKSPACE_ROOT':str(k.owner_scope()),'CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1'}
            subprocess.run(k.launch_command(),cwd=ROOT,env=env,check=True)
            logs=k.owner_scope()/'artifacts/logs'
            job=k.read(logs/k.TAG/'job.json')
            k.need(job['schema']=='gpu-job/v2' and job['tag']==k.TAG
                   and job['command']==k.run_command() and job['started_epoch']>=started
                   and Path(job['cwd']).resolve()==k.owner_scope().resolve()
                   and job['command_sha256']==runtime.command_digest(k.run_command(),k.owner_scope()),'fresh child binding')
            receipt=seq.wait_success(logs,dict(tag=k.TAG,argv=k.launch_command()),job)
            state.update(state='SUCCESS',jobs=[dict(run_id=job['run_id'],receipt=receipt)])
            runtime.atomic_json(LEDGER,state)
            return state
        except BaseException as exc:
            state.update(state='STOPPED',error_type=type(exc).__name__)
            runtime.atomic_json(LEDGER,state)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=list(MODES),required=True)
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    print(json.dumps(execute(a.mode,a.decision,a.dry_run)))


if __name__=='__main__': main()
