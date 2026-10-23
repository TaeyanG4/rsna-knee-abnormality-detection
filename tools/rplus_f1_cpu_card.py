"""v1.2.1 real-data CPU qualification. Fixed scratch, no CUDA or GPU commands."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from training.arms import rplus_f1 as c, lifecycle as life

SCRATCH=Path('C:/kaggle-offload/sol-1/rplus-f1-v121')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--arm',choices=['parent','C-L2-f1','C-BASE-f1'],required=True)
    a=parser.parse_args()
    parent=a.arm=='parent'
    root=c.ROOT if parent else SCRATCH/'engineering-root'
    out=SCRATCH/a.arm;out.mkdir(parents=True,exist_ok=True)
    module='training.runners.coatnet_fold1' if parent else 'training.runners.rplus_f1'
    env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
    commands=[]
    def run(args,status,code=0):
        argv=[sys.executable,'-B','-m',module,*map(str,args)]
        proc=subprocess.run(argv,cwd=c.ROOT,env=env,capture_output=True,text=True,encoding='utf8')
        log=out/f'command-{len(commands):02d}.log';log.write_text(proc.stdout+proc.stderr,encoding='utf8')
        commands.append(dict(argv=argv,exit_code=proc.returncode,log=log.as_posix(),sha256=life.file_sha256(log)))
        (out/'command-results.json').write_text(json.dumps(commands,indent=2),encoding='utf8')
        c.need(proc.returncode==code,'command failed: '+str(log))
        result=json.loads(proc.stdout)
        c.need(result.get('status')==status,'status differs: '+str(log))
        print(json.dumps(dict(command=args[0],status=status,exit_code=code)),flush=True)
        return result
    specpath=c.ROOT/c.BASE/'parent-spec.json' if parent else out/'spec.json'
    if parent:
        registration=['register','--root',root,'--out',out/'spec-copy.json','--variant','control',
            '--population','fold1','--seed','43','--arm','CONTROL43-f1','--attempt','rplus-v1.2.1',
            '--target','.local/refactor/qual-4070-control-fold0-s43-8594f460d5d24d09a9f0f3f6796ab082/target.json',
            '--target-sha256','e6abf87e32d4ea79f1f0d0c872dfa2c442e33e6f14e1b3f2a6f910211aeb18b7']
    else:
        registration=['register','--root',root,'--out',specpath,'--arm',a.arm,'--attempt','engineering-v1.2.1']
    run(['observe','--device','cpu','--out',out/'observed-cpu.json'],'OBSERVATION_RECORDED')
    run(registration,'REGISTERED');run(registration,'REJECTED',2)
    if parent: c.need((out/'spec-copy.json').read_bytes()==specpath.read_bytes(),'parent registration differs')
    s=['--spec',specpath];stage=out/'stage';smoke=out/'smoke'
    run(['validate',*s,'--root',root],'VALIDATED')
    for _ in range(2): run(['prepare',*s,'--root',root,'--out',stage],'STAGED_NO_EXECUTION')
    run(['validate',*s,'--root',stage,'--staged'],'VALIDATED')
    common=[*s,'--stage',stage,'--corpus',c.ROOT/c.parent.c.CORPUS,'--device','cpu','--workers','0']
    if not smoke.exists(): run(['smoke',*common,'--out',smoke],'ENGINEERING_SMOKE_COMPLETE')
    else:
        # The parent smoke was executed before this card to unblock continuations.
        c.need(parent,'unexpected existing continuation smoke')
        life.verify_evidence(c.parent.Spec(specpath.read_bytes()),c.read(smoke/'verification.json'),smoke)
    summary=c.read(smoke/'outputs/summary.json')
    c.need(summary['updates']==1 and summary['bn']['studies']==2 and summary['bn']['bn_modules']==25
        and summary['cpu_cuda_initialized'] is False and summary['results']['real_data'],'real CPU card differs')
    resumed=out/'resumed'
    run(['smoke',*common,'--out',resumed,'--resume',smoke/'epochs/epoch-01.pt'],'ENGINEERING_SMOKE_COMPLETE')
    c.need(c.read(resumed/'outputs/summary.json')['final_state_sha256']==summary['final_state_sha256'],'resume model differs')
    for view in ('plain','mirror','mean'):
        c.need((smoke/f'outputs/predictions-{view}.csv').read_bytes()==(resumed/f'outputs/predictions-{view}.csv').read_bytes(),'resume predictions differ')
    run(['admit',*s,'--stage',stage,'--note',smoke/'verification.json','--evidence-root',smoke],'HOLD')
    run(['combine',*s,'--cpu',smoke,'--target',smoke,'--out',out/'forbidden-combine'],'REJECTED',2)
    run(['train',*common,'--out',out/'forbidden-full-cpu'],'REJECTED',2)
    receipt=out/'artifact-receipt.json';archive=out/'smoke-artifacts.tar'
    run(['receipt',*s,'--outputs',smoke/'outputs','--note',smoke/'verification.json','--evidence-root',smoke,'--out',receipt],'ARTIFACTS_VERIFIED')
    digest=life.content_sha256(c.read(receipt))
    run(['export',*s,'--outputs',smoke/'outputs','--receipt',receipt,'--archive',archive,'--receipt-sha256',digest],'EXPORTED')
    run(['collect',*s,'--archive',archive,'--receipt-sha256',digest,'--out',out/'collected'],'COLLECTED_ARTIFACTS_VERIFIED')
    report=dict(status='CPU_CARD_COMPLETE',arm=a.arm,commands=len(commands),smoke_seconds=summary['seconds'],
        updates=1,bn_studies=2,readout_studies=1,resume_model_and_csv_exact=True,cuda_initialized=False)
    c.write(out/'card-result.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
