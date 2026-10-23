"""Execute one reviewed HC05 fit under the aggregate hard timeout. Use via gpu_job/queue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--seed',choices=[42,43],type=int,required=True);a=ap.parse_args()
    p=json.loads(a.protocol.read_text());seed=str(a.seed)
    if p.get('status')!='READY_REVIEWED' or p.get('review_required'):
        raise SystemExit('Protocol is not reviewed and ready')
    for path,key in [(a.protocol.with_name('runner.py'),'runner_sha256'),(a.protocol.with_name('runner_commands.json'),'commands_sha256'),(Path(__file__),'outer_wrapper_sha256')]:
        if sha(path)!=p[key]:raise SystemExit('Frozen code/command hash mismatch: '+str(path))
    for key in ['parity','review']:
        if sha(ROOT/p[key+'_path'])!=p[key+'_sha256']:raise SystemExit(key+' receipt changed')
    tag=p['runs'][seed]['tag'];out=ROOT/'artifacts/experiments'/tag
    if out.exists():raise SystemExit('Refuse to overwrite '+str(out))
    budget_path=a.protocol.with_name('execution_budget.json')
    if a.seed==42:
        with budget_path.open('x',encoding='utf-8') as f:json.dump({'seconds_used':0,'completed_seeds':[],'failed':False},f)
    budget=json.loads(budget_path.read_text())
    if budget['failed'] or budget['completed_seeds']!=([] if a.seed==42 else [42]):raise SystemExit('Budget/preceding fit is not valid')
    remaining=7200-budget['seconds_used']
    if remaining<300:raise SystemExit('Aggregate budget exhausted')
    command=json.loads(a.protocol.with_name('runner_commands.json').read_text())[seed]
    start=time.monotonic();flags=(subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW) if os.name=='nt' else 0
    process=subprocess.Popen(command,cwd=ROOT,creationflags=flags)
    failure=None
    try:
        rc=process.wait(timeout=remaining)
        if rc:failure='runner exit '+str(rc)
    except subprocess.TimeoutExpired:
        if os.name=='nt':subprocess.run(['taskkill.exe','/PID',str(process.pid),'/T','/F'],check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        else:process.kill()
        process.wait();failure='aggregate hard timeout'
    elapsed=time.monotonic()-start;budget['seconds_used']+=elapsed
    if budget['seconds_used']>7200:failure=failure or 'aggregate hard timeout'
    expected=p['runs'][seed]['expected_successful_updates']
    try:
        summary=json.loads((out/'summary.json').read_text());complete=json.loads((out/'runner-complete.json').read_text())
        if (summary['optimizer_updates']!=p['runs'][seed]['expected_attempts']
                or summary['optimizer_updates']-summary['skipped_updates']!=expected
                or complete['successful_updates']!=expected or complete['expected_successful_updates']!=expected):
            failure=failure or 'successful-update contract'
        inventory=json.loads((out/'output-inventory.json').read_text())
        required={'summary.json','provenance.json','history.json','model_best.pt','oof_best.csv','oof_k36.csv'}
        if set(inventory)!=required:failure=failure or 'incomplete output inventory'
        for name,entry in inventory.items():
            if name not in required:failure=failure or 'unexpected output inventory entry';continue
            if (out/name).stat().st_size!=entry['bytes'] or sha(out/name)!=entry['sha256']:failure=failure or 'output hash/size mismatch'
    except (OSError,ValueError,KeyError):failure=failure or 'incomplete output contract'
    # Validation/hashing also consumes the aggregate budget.
    budget['seconds_used']+=time.monotonic()-start-elapsed
    if budget['seconds_used']>7200:failure=failure or 'aggregate hard timeout during validation'
    record={'tag':tag,'elapsed_seconds':time.monotonic()-start,'aggregate_seconds':budget['seconds_used'],'failure':failure,'valid_for_scoring':failure is None,'protocol_sha256':sha(a.protocol)}
    if failure:budget['failed']=True
    else:budget['completed_seeds'].append(a.seed)
    budget_path.write_text(json.dumps(budget,indent=2),encoding='utf-8')
    out.mkdir(parents=True,exist_ok=True)
    (out/('INVALID.json' if failure else 'VALIDATED_SUCCESS.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record));return 1 if failure else 0
if __name__=='__main__':sys.exit(main())
