"""H216 pair-core CPU mechanics preparation; no weight load, GPU or fitting."""
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.l1_rco_stage0_v1 import read,write,sha,need
from tools import h216_oai_pair_core_v1 as core

OUT=ROOT/'.local/scratch/slot-b/h216-oai-pair-core-v1'


def prepare():
    need(not OUT.exists(),'OUTPUT_EXISTS'); OUT.mkdir()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    def forbidden(*a,**kw): raise ValueError('CHECKPOINT_LOAD_FORBIDDEN')
    torch.load=forbidden
    sys.path.insert(0,str(ROOT/'tools/exp189_s6_src'))
    import compact_train as ct
    reg=read(ROOT/'.local/scratch/l1-rco/k1v3-grouping/registration.json')
    uid=reg['train'][0]
    cache='G:/temp_data/rsna-knee/l1rco-cache-full'
    ds=ct.Studies(cache,[uid],64,'train',22602,train_sampling='cover',cover_stride=3,slices=3,slice_jitter=.25,phys_mm=3.5)
    img,meta,i=ds[0]
    # Actual RSNA cache; bounded two-window CPU graph only, no efficacy estimate.
    img,meta,mask,_=ct.collate([(img[:2],meta[:2],i)])
    img=core.crop(img,torch.Generator().manual_seed(21673))
    torch.manual_seed(21600)
    base=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False)
    need(not torch.cuda.is_initialized(),'CUDA_INITIALIZED')
    original=core.original_head(base); manifests={}; proofs={}
    for name,outputs in [('rsna_control',12),('oai_native',120)]:
        core.reset_rng(21600); model=core.TemporaryAdapter(base,outputs)
        core.configure_trainable(model,True); core.train_mode(model)
        _,manifest=core.parameter_groups(model); manifests[name]=manifest
        z=model(img,meta,mask)
        target=torch.zeros(1,30,dtype=torch.long) if outputs==120 else torch.full((1,12),.5)
        loss=core.native_loss(z,target) if outputs==120 else core.rsna_loss(z,target)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        need(grads and all(torch.isfinite(g).all() for g in grads),'GRADIENTS')
        need(all(p.grad is None for n,p in model.enc.named_parameters() if not n.startswith('stages.3.')),'FROZEN_GRADIENT')
        proofs[name]=dict(output_shape=list(z.shape),finite=True,gradient_tensors=len(grads),optimizer_steps=0)
        model.zero_grad(set_to_none=True); del model,z,loss
    core.restore_head(base,original); core.configure_trainable(base,False); core.train_mode(base)
    _,manifests['recovery']=core.parameter_groups(base)
    z=base(img,meta,mask); loss=core.rsna_loss(z,torch.full((1,12),.5)); loss.backward()
    need(bool(torch.isfinite(z).all()),'RECOVERY_FINITE')
    need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in base.parameters()),'RECOVERY_GRADIENT')
    proofs['recovery']=dict(output_shape=list(z.shape),finite=True,optimizer_steps=0,original_head_restored=True)
    write(OUT/'PARAMETER-MANIFESTS.json',manifests)
    source_paths=[Path(__file__),ROOT/'tools/h216_oai_pair_core_v1.py',ROOT/'tests/test_h216_oai_pair_core_v1.py',
                  ROOT/'docs/research/h216-oai-pair-core-v1.md',ROOT/'tools/l1_rco_k1_optim_v3.py',
                  ROOT/'tools/exp189_s6_src/compact_train.py']
    result=dict(status='PASS_CPU_MECHANICS_ONLY',recipe=core.RECIPE,proofs=proofs,
                sources={str(p):sha(p) for p in source_paths},
                manifest_sha256=sha(OUT/'PARAMETER-MANIFESTS.json'),
                actual_cache_input_sha256=hashlib.sha256(img.numpy().tobytes()+meta.numpy().tobytes()).hexdigest(),
                real_rsna_windows=2,random_weights=True,checkpoint_loads=0,cuda=False,optimizer_steps=0,
                actual_oai_roi=False,training_ready=False,outer_runtime_ready=False,efficacy_evaluated=False)
    write(OUT/'CPU-MECHANICS.json',result)
    print(json.dumps({k:result[k] for k in ('status','real_rsna_windows','random_weights','checkpoint_loads','cuda','optimizer_steps','training_ready','outer_runtime_ready')}))


if __name__=='__main__':
    try: prepare()
    except Exception as exc:
        code=str(exc) if type(exc) is ValueError and str(exc).replace('_','').isalnum() else 'PRIVATE_DETAILS_SUPPRESSED'
        fail=dict(status='FAILED',error_type=type(exc).__name__,code=code)
        if OUT.exists() and not (OUT/'FAILURE.json').exists(): write(OUT/'FAILURE.json',fail)
        print(json.dumps(fail)); raise SystemExit(1)
