"""Offline v1.1 CPU CLI checks; no protected screen inputs or scientific approval."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.local/scratch/pl-loader-v11/build'


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    stage=str(WORK/'child-stage')
    absent=str(WORK/'intentionally-absent.json')
    assert not Path(absent).exists()
    commands=[]
    for cmd in ('register','validate'):
        commands.append((f'release-{cmd}', ['tools.pl_production_release_v1',cmd,'--stage',stage],False))
    for arm in ('global','fill'):
        commands.append((f'preflight-{arm}',['tools.pl_production_release_v1','preflight','--stage',stage,'--arm',arm],False))
    for mode,variant in [('visible','a'),('n32','a'),('visible','b')]:
        commands.append((f'{mode}-{variant}',['tools.pl_production_qualify_v1',mode,'--variant',variant,'--stage',stage,'--dry-run'],False))
    commands.append(('prepare-missing-go',['tools.pl_production_launch_v1','prepare','--go',absent,'--memory',absent,'--review',absent],True))
    for cmd in ('validate','admit'):
        commands.append((f'train-{cmd}-missing-spec',['tools.pl_production_train_v1',cmd,'--spec',absent,'--go',absent],True))
    commands.append(('vast-prepare-missing-plan',['tools.pl_vast_split_v110','prepare','--plan',absent,'--review',absent,'--output',str(WORK/'absent-bundle')],True))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    results=[]
    for name,args,blocked in commands:
        r=subprocess.run([sys.executable,'-X','faulthandler','-B','-m',*args],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
        (WORK/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
        expected='H-B-146 GO absent' if name=='prepare-missing-go' else 'intentionally-absent.json'
        if name=='vast-prepare-missing-plan':expected='independent split review absent'
        ok=(r.returncode!=0 and expected in r.stderr) if blocked else r.returncode==0
        assert ok,(name,r.returncode)
        results.append(dict(command=name,exit_code=r.returncode,result='EXPECTED_HOLD' if blocked or name.startswith(('visible','n32')) else 'PASS'))
    (WORK/'metadata-smoke.json').write_text(json.dumps(dict(status='PASS',commands=results,real_target_prepare_admit=False),indent=2),encoding='utf-8')
    print(json.dumps(results))


if __name__=='__main__':main()
