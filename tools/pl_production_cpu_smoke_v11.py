"""v1.1.0 CPU-only real corpus parity/RAM and two-update optimizer smoke.

Synthetic targets only; no report/label files; fixed builder scratch folder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.local/scratch/pl-loader-v11/build'


def digest_batch(batch):
    x,y,ids=batch
    return hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()+json.dumps(ids).encode()).hexdigest()


def child(mode,workers,attempt):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled=False
    from training.arms import mil_screen_hooks_v1 as h
    from training.arms.hb146_pl_v1 import Arm
    from tools import pl_production_loader_v1 as loading
    from tools import pl_production_train_v1 as train
    from tools import pl_production_release_v1 as release
    corpus=ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02'
    ids=h.common.Corpus(corpus).ids[:16].tolist()
    labels={u:np.full(12,(i+1)/20,dtype=np.float32) for i,u in enumerate(ids)}
    if mode=='parity':
        arm=Arm(labels)
        def collect(n,epochs):
            ds=arm.dataset(h.MILWindows(corpus,ids,labels,12,True,42),train=True)
            loader,gen,cfg=loading.make_loader(ds,arm.collate_fn,42,workers=n,batch_size=8)
            rows=[]
            with loading.managed_loader(loader):
                for epoch in epochs:
                    ds.set_epoch(epoch);gen.manual_seed(42+epoch)
                    rows.append([digest_batch(batch) for batch in loader])
            return rows,cfg
        zero,_=collect(0,[0,1]);parallel,cfg=collect(workers,[0,1]);resume,_=collect(workers,[1])
        assert zero==parallel and resume==zero[1:] and zero[0]!=zero[1]
        result=dict(status='PASS',workers=workers,epochs=2,steps_per_epoch=2,
                    batches_targets_ids_bitwise_equal=True,persistent_epoch_changes=True,
                    epoch_boundary_resume_equal=True,loader=cfg)
    else:
        target=dict(train_ids=ids[:2],weights=[1.]*12,arm='global',engineering_only=True,
          teacher_sha256='0'*64,target_sha256='0'*64,train_ids_sha256='0'*64,
          l2_pos_weight_sha256='0'*64,registration_sha256='0'*64)
        with train.adapters():
            spec=dict(member=release.MIL[3],recipe=train.base.mil_recipe(40),corpus=str(corpus),
                source_pins=[train.base.asset(p) for p in train.base.SOURCES],target_bundle={'engineering_only':True})
        spec['spec_sha256']=train.base.p.digest(spec)
        output=WORK/f'fit-w{workers}-attempt{attempt}'
        result=train.mil_fit(spec,target,labels,output,device='cpu',smoke=True,go=None,workers=workers)
        assert result['status']=='ENGINEERING_SMOKE'
        record=torch.load(output/'resume.pt',map_location='cpu',weights_only=False)
        assert record['counters']['updates']==2 and record['next_epoch']==1
        result=dict(status='PASS',workers=workers,updates=2,checkpoint_export=True,journal_reload=True)
    assert not torch.cuda.is_initialized()
    result.update(cuda_initialized=False,real_images=16 if mode=='parity' else 2,synthetic_targets=True)
    print(json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['parity','fit'],default='parity')
    ap.add_argument('--workers',type=int,default=6);ap.add_argument('--child',action='store_true')
    ap.add_argument('--attempt',type=int,default=1);a=ap.parse_args()
    if a.child:return child(a.mode,a.workers,a.attempt)
    import psutil
    from tools.hb150_io_v1 import atomic_json
    from tools import pl_production_launch_v1 as launch
    from tools import pl_vast_split_v110 as split
    WORK.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
      OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',ATEN_CPU_CAPABILITY='default')
    attempts=[];result={};peak=0;status='HOLD'
    for attempt in range(1,4):
        log=WORK/f'{a.mode}-w{a.workers}-attempt{attempt}.log'
        start=time.monotonic()
        with log.open('w',encoding='utf-8') as stream:
            proc=subprocess.Popen([sys.executable,'-X','faulthandler','-B','-m',
              'tools.pl_production_cpu_smoke_v11','--child','--mode',a.mode,'--workers',str(a.workers),
              '--attempt',str(attempt)],env=env,stdout=stream,stderr=subprocess.STDOUT)
            owner=psutil.Process(proc.pid)
            owned={}
            try:
                while proc.poll() is None:
                    family=[owner,*owner.children(recursive=True)];rss=0
                    for p in family:
                        if p.pid!=owner.pid:owned[p.pid]=p
                        try:rss+=p.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,rss)
                    if rss>40e9 or time.monotonic()-start>240:
                        raise RuntimeError('RAM budget or 240-second smoke deadline exceeded')
                    time.sleep(.1)
            finally:
                # Retain Process identities so a crashed parent cannot orphan workers.
                for p in owned.values():
                    try:
                        if p.is_running():p.kill()
                    except psutil.NoSuchProcess:pass
                if proc.poll() is None:
                    for p in owner.children(recursive=True):
                        try:p.kill()
                        except psutil.NoSuchProcess:pass
                    proc.kill();proc.wait()
        attempts.append(dict(attempt=attempt,exit_code=proc.returncode,seconds=round(time.monotonic()-start,2)))
        if proc.returncode==0:
            result=json.loads(log.read_text().splitlines()[-1]);status='PASS';break
        # Only retry known native environment access violations, never code errors.
        if proc.returncode & 0xffffffff != 0xc0000005:break
    receipt=dict(status=status,kind='real_cpu_smoke' if a.mode=='fit' else 'real_loader_parity',
      result=result,attempts=attempts,peak_process_tree_rss_bytes=peak,ram_budget_bytes=int(40e9),
      gpu_memory_measured=False,sources=[launch.asset(p) for p in launch.SOURCES])
    atomic_json(WORK/f'{a.mode}-w{a.workers}.json',receipt,critical=True)
    if a.mode=='fit':
        atomic_json(WORK/f'vast-smoke-w{a.workers}.json',dict(receipt,sources={p:split.sha(ROOT/p) for p in split.REVIEW_SOURCES}),critical=True)
    print(json.dumps(receipt));return 0 if status=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
