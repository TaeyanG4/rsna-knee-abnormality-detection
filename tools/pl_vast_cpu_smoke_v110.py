"""1.1.1 real-image engineering smoke; synthetic targets, CPU only, no scientific GO.

Changelog: pin the production loader and expose Vast worker configuration.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import pl_vast_split_v110 as s
WORK=ROOT/'.local/scratch/pl-loader-v11/build/vast-smoke'


def child(io_only=False,workers=None):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    # CPU smoke only: avoid the host's crashing oneDNN backward kernel.
    torch.backends.mkldnn.enabled=False
    from tools import pl_production_train_v1 as train
    from tools import pl_production_release_v1 as release
    from training.src.raptor_common import Corpus
    corpus=ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02'
    ids=Corpus(corpus).ids[:2].tolist()
    labels={u:np.full(12,.5,dtype=np.float32) for u in ids}
    target=dict(train_ids=ids,weights=[1.]*12,arm='global',engineering_only=True,
        teacher_sha256='0'*64,target_sha256='0'*64,train_ids_sha256='0'*64,
        l2_pos_weight_sha256='0'*64,registration_sha256='0'*64)
    spec=dict(member=release.MIL[3],recipe=train.base.mil_recipe(40),corpus=str(corpus),
              source_pins=[train.base.asset(p) for p in train.base.SOURCES],target_bundle={'engineering_only':True})
    spec['spec_sha256']=train.base.p.digest(spec)
    # Exercise copied source pin mapping and the exact portable runtime recipe path projection.
    s.source_projection(spec,ROOT)
    for key in ('weights','initial_state'):spec['recipe']['model'][key]=str(ROOT/spec['recipe']['model'][key])
    if io_only:
        from training.arms.hb146_pl_v1 import Arm
        from training.arms import mil_screen_hooks_v1 as h
        ds=Arm(labels).dataset(h.MILWindows(corpus,ids,labels,12,True,42),train=True)
        x,y,uids=Arm(labels).collate_fn([ds[i] for i in range(2)])
        assert bool(torch.isfinite(x).all()) and tuple(y.shape)==(2,12)
        payload=WORK/'cpu-io-targets.npz'
        np.savez(payload,ids=np.asarray(ids),targets=y.numpy(),mask=np.ones((2,12),bool),weights=np.ones(12,np.float32))
        expected=s.pin(payload);s.checked(WORK,payload.name,expected)
        with np.load(payload,allow_pickle=False) as d:assert np.array_equal(d['targets'],y.numpy())
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS',real_images=2,synthetic_targets=True,transport_roundtrip=True)))
        return
    result=train.mil_fit(spec,target,labels,WORK/'real-cpu-fit',device='cpu',smoke=True,go=None,
                         workers=workers,loader_profile='vast')
    assert result['status']=='ENGINEERING_SMOKE' and not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS',updates=2,cuda_initialized=False,real_images=2,synthetic_targets=True)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--child',action='store_true');ap.add_argument('--io-only',action='store_true')
    ap.add_argument('--workers',type=int);a=ap.parse_args()
    if a.child:return child(a.io_only,a.workers)
    WORK.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',ATEN_CPU_CAPABILITY='default')
    start=time.monotonic()
    try:
        r=subprocess.run([sys.executable,'-X','faulthandler','-B',__file__,'--child']+(['--io-only'] if a.io_only else [])+
                         ([] if a.workers is None else ['--workers',str(a.workers)]),env=env,capture_output=True,text=True,timeout=360)
        status='PASS' if r.returncode==0 else 'HOLD'
        (WORK/('real-cpu-io.log' if a.io_only else 'real-cpu-smoke.log')).write_text(r.stdout+r.stderr,encoding='utf-8')
        code=r.returncode
    except subprocess.TimeoutExpired:
        status='HOLD';code='TIMEOUT'
    receipt=dict(status=status,kind='real_cpu_io' if a.io_only else 'real_cpu_smoke',exit_code=code,seconds=round(time.monotonic()-start,2),
        real_images=2,synthetic_targets=True,production_admission_tested=False,cpu_mkldnn=False,
        sources={p:s.sha(ROOT/p) for p in (*s.REVIEW_SOURCES,'tools/pl_vast_cpu_smoke_v110.py')})
    (WORK/('real-cpu-io.json' if a.io_only else 'real-cpu-smoke.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt));return 0 if status=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
