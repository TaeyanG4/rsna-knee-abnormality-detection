"""H-B-113 v1.1.0 real-data CPU card, fixed scratch, no CUDA or cleanup.

Changelog: one small real model smoke, authenticated serial packaging, failures.
Re-run reuses authenticated command results; never creates an extra smoke copy.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from training.arms import rplus_public_v110 as c, lifecycle as life

OUT=c.ROOT/'.local/scratch/sol-2/hb113-seed44'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    commands=[]
    def run(args,status,code=0):
        argv=[sys.executable,'-B','-m','training.runners.rplus_public_v110',*map(str,args)]
        log=OUT/f'command-{len(commands):02d}.log'
        record=log.with_suffix('.json')
        if record.exists():
            entry=c.read(record)
            c.need(entry['argv']==argv and life.file_sha256(log)==entry['sha256'],'existing command differs')
            proc_code=entry['exit_code'];text=log.read_text(encoding='utf-8')
        else:
            proc=subprocess.run(argv,cwd=c.ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
            text=proc.stdout+proc.stderr;proc_code=proc.returncode
            log.write_text(text,encoding='utf-8')
            entry=dict(argv=argv,exit_code=proc_code,log=log.name,sha256=life.file_sha256(log))
            c.write(record,entry)
        commands.append(entry)
        c.need(proc_code==code,'CPU command failed: '+str(log))
        # Warnings from numerical libraries may follow stdout. Parse last JSON status.
        results=[]
        for line in text.splitlines():
            try: results.append(json.loads(line))
            except json.JSONDecodeError: pass
        c.need(results and results[-1].get('status')==status,'CPU status differs: '+str(log))
        print(json.dumps(dict(command=args[0],exit_code=code,status=status)),flush=True)
        return results[-1]
    base=c.ROOT/c.BASE;specpath=base/'member-spec.json';stage=OUT/'stage';smoke=OUT/'cpu-smoke'
    run(['observe','--device','cpu','--out',OUT/'observed-cpu.json'],'OBSERVATION_RECORDED')
    registration=['register','--root',c.ROOT,'--out',specpath]
    run(registration,'REGISTERED');run(registration,'REJECTED',2)
    s=['--spec',specpath]
    run(['validate',*s,'--root',c.ROOT],'VALIDATED')
    for _ in range(2): run(['prepare',*s,'--root',c.ROOT,'--out',stage],'STAGED_NO_EXECUTION')
    run(['validate',*s,'--root',stage,'--staged'],'VALIDATED')
    common=[*s,'--stage',stage,'--corpus',c.ROOT/c.c.CORPUS,'--parent',c.ROOT/c.PARENT,'--workers','0']
    run(['train',*common,'--device','cpu','--out',OUT/'forbidden-full-cpu'],'REJECTED',2)
    run(['smoke',*common,'--device','cuda','--out',OUT/'forbidden-unreviewed-cuda'],'REJECTED',2)
    run(['smoke',*common,'--device','cpu','--out',smoke,'--serial'],'ENGINEERING_SMOKE_COMPLETE')
    spec=c.Spec(specpath.read_bytes());c.validate_stage(spec,stage)
    summary=c.read(smoke/'outputs/summary.json')
    c.need(summary['updates']==1 and summary['rsna_studies']==2 and summary['bn']['studies']==2
           and summary['cpu_cuda_initialized'] is False and summary['scientific_scope']=='ENGINEERING_SMOKE_ONLY','smoke assertions differ')
    note=c.read(smoke/'verification.json');life.verify_evidence(spec,note,smoke)
    c.need(c.read(smoke/'serial-receipt.json')['status']=='SERIAL_COMPLETE','serial chain missing')
    run(['admit',*s,'--stage',stage,'--note',smoke/'verification.json','--evidence-root',smoke],'HOLD')
    run(['combine',*s,'--cpu',smoke,'--target',smoke,'--out',OUT/'forbidden-combine'],'REJECTED',2)
    bad_sha='0'*64
    run(['export',*s,'--outputs',smoke/'outputs','--receipt',smoke/'artifact-receipt.json',
         '--archive',OUT/'forbidden-export.tar','--receipt-sha256',bad_sha],'REJECTED',2)
    final=dict(status='CPU_CARD_COMPLETE',commands=commands,smoke_seconds=summary['seconds'],
               spec_sha256=spec.sha256,cuda_initialized=False,scope=c.SCOPE)
    if (OUT/'command-results.json').exists(): c.need(c.read(OUT/'command-results.json')==final,'CPU card result differs')
    else: c.write(OUT/'command-results.json',final)
    print(json.dumps(dict(status=final['status'],commands=len(commands),smoke_seconds=summary['seconds'],cuda_initialized=False)),flush=True)


if __name__=='__main__': main()
