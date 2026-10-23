"""R+ CPU card v1.1.0: durable real-data qualification; never CUDA.

Changelog: execute registration/staging/smoke/resume/packaging/self-screen CLI.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from training.arms import rplus_base as c, lifecycle as life


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True);p.add_argument('--arm',choices=['C-BASE'],required=True)
    a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    commands=[]
    def run(args,status,code=0,module='training.runners.rplus_base'):
        argv=[sys.executable,'-B','-m',module,*map(str,args)]
        proc=subprocess.run(argv,cwd=c.ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        log=out/f'command-{len(commands):02d}.log';log.write_text(proc.stdout+proc.stderr,encoding='utf-8')
        entry=dict(argv=argv,exit_code=proc.returncode,log=log.name,sha256=life.file_sha256(log))
        commands.append(entry);c.write(log.with_suffix('.json'),entry)
        c.need(proc.returncode==code, 'CPU command failed; inspect '+str(log))
        result=json.loads(proc.stdout)
        c.need(result.get('status',result.get('verdict'))==status, 'CPU status differs; inspect '+str(log))
        print(json.dumps(dict(command=args[0],exit_code=code,status=status)),flush=True)
        return result
    specpath=out/'spec.json';stage=out/'stage';smoke=out/'smoke'
    run(['observe','--device','cpu','--out',out/'observed-cpu.json'],'OBSERVATION_RECORDED')
    registration=['register','--root',c.ROOT,'--controls',c.ROOT/c.BASE/'controls.json','--arm',a.arm,
                  '--attempt','local4070-s43-v1','--out',specpath]
    run(registration,'REGISTERED');run(registration,'REJECTED',2)
    s=['--spec',specpath]
    run(['validate',*s,'--root',c.ROOT],'VALIDATED')
    for _ in range(2): run(['prepare',*s,'--root',c.ROOT,'--out',stage],'STAGED_NO_EXECUTION')
    run(['validate',*s,'--root',stage,'--staged'],'VALIDATED')
    common=[*s,'--stage',stage,'--corpus',c.ROOT/c.control.c.CORPUS,'--oai-corpus',c.ROOT/c.OAI,'--device','cpu','--workers','0']
    run(['smoke',*common,'--out',smoke],'ENGINEERING_SMOKE_COMPLETE')
    summary=c.read(smoke/'outputs/summary.json')
    c.need(summary['updates']==1 and (summary['rsna_studies'],summary['oai_studies'])==((1,1) if a.arm=='C-OAI' else (2,0))
           and summary['bn']['studies']==2 and summary['cpu_cuda_initialized'] is False,'smoke assertions differ')
    resumed=out/'resumed'
    run(['smoke',*common,'--out',resumed,'--resume',smoke/'epochs/epoch-01.pt'],'ENGINEERING_SMOKE_COMPLETE')
    c.need(c.read(resumed/'outputs/summary.json')['final_state_sha256']==summary['final_state_sha256'],'resume model differs')
    for name in ('plain','mirror','mean'):
        c.need((smoke/f'outputs/predictions-{name}.csv').read_bytes()==(resumed/f'outputs/predictions-{name}.csv').read_bytes(),'resume readout differs')
    run(['admit',*s,'--stage',stage,'--note',smoke/'verification.json','--evidence-root',smoke],'HOLD')
    run(['combine',*s,'--cpu',smoke,'--target',smoke,'--out',out/'forbidden-combine'],'REJECTED',2)
    run(['train',*common,'--out',out/'forbidden-full-cpu'],'REJECTED',2)
    receipt=out/'artifact-receipt.json';archive=out/'smoke-artifacts.tar'
    run(['receipt',*s,'--outputs',smoke/'outputs','--note',smoke/'verification.json','--evidence-root',smoke,'--out',receipt],'ARTIFACTS_VERIFIED')
    digest=life.content_sha256(c.read(receipt))
    run(['export',*s,'--outputs',smoke/'outputs','--receipt',receipt,'--archive',archive,'--receipt-sha256',digest],'EXPORTED')
    run(['collect',*s,'--archive',archive,'--receipt-sha256',digest,'--out',out/'collected'],'COLLECTED_ARTIFACTS_VERIFIED')
    run([*s,'--root',c.ROOT,'--out',out/'self-screen','--control-self-test'],'READOUT_SPEC_PREPARED',module='tools.rplus_base_readout')
    for mode in ('mean','plain','mirror','base-minus-parent/mean','base-minus-parent/plain','base-minus-parent/mirror'):
        # This invokes the requested multi-lens tool on actual control predictions.
        argv=[sys.executable,'-B','-m','tools.multi_lens_readout','--root',str(c.ROOT),'--spec',str(out/'self-screen'/Path(mode).parent/f'spec-{Path(mode).name}.json'),'--out',str(out/'self-screen'/Path(mode).parent/f'result-{Path(mode).name}.json')]
        proc=subprocess.run(argv,cwd=c.ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        log=out/f'command-{len(commands):02d}.log';log.write_text(proc.stdout+proc.stderr,encoding='utf-8')
        entry=dict(argv=argv,exit_code=proc.returncode,log=log.name,sha256=life.file_sha256(log));commands.append(entry);c.write(log.with_suffix('.json'),entry)
        c.need(proc.returncode==0,'self-screen failed: '+str(log))
        print(json.dumps(dict(command='multi-lens-'+mode,exit_code=0,scope='CONTROL_SELF_TEST_ONLY')),flush=True)
    c.write(out/'command-results.json',commands)
    print(json.dumps(dict(status='CPU_CARD_COMPLETE',commands=len(commands),smoke_seconds=summary['seconds'],cuda_initialized=False)),flush=True)


if __name__=='__main__': main()
