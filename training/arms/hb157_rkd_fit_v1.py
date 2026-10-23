"""H-B-157 v1.0.0: R+ arithmetic with soft labels, six epochs and resumable IO.

Changelog: new isolated fitter; no frozen R+ source changes. CPU evidence uses
two real studies; CUDA smoke uses a full 16-microbatch update and K94 mirror.
"""
from __future__ import annotations
import gc
import os
from pathlib import Path
import time
import torch
import numpy as np
from torch.utils.data import DataLoader,DistributedSampler
from training.arms import coatnet as c
from training.arms.coatnet_fit import configure,observation,worker_init,verify_studies
from training.arms.rplus_data import L2Dataset,mixed
from training.arms.rplus_fit import make_optimization
from training.arms.rplus_public_fit_v110 import load_parent
from training.src.exp143_v2 import accumulation_group
from tools.exp112_full_swa import recompute_bn,swa_average,state_dict_sha256
from tools.hb150_io_v1 import atomic_json,safe_replace,safe_unlink,_retry,_failure,JournalWriteError
from tools.pl_production_watchdog_v1 import PROGRESS,ProgressWriter
from tools import hb157_rkd_v1 as b
base=c.base
need=b.need

def save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    PROGRESS.phase_start('checkpoint_write')
    try:
        with _retry(lambda:temporary.open('wb')) as f:
            torch.save(payload,ProgressWriter(f,PROGRESS))
            f.flush();os.fsync(f.fileno())
        safe_replace(temporary,path)
        PROGRESS.completed('checkpoint_write')
    except JournalWriteError: raise
    except OSError as exc: _failure(path,exc,True)
    finally: safe_unlink(temporary)

def journal(directory,model,optimizer,scheduler,scaler,identity,epoch,history,swa):
    path=directory/f'epoch-{epoch:02d}.pt'
    save(path,dict(identity=identity,epoch=epoch,history=history,model=model.state_dict(),
                   optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict()))
    receipt=dict(identity=identity,epoch=epoch,path=path.name,sha256=b.sha(path),swa_members=swa)
    atomic_json(directory/'journal.json',receipt)
    return receipt

def restore(directory,model,optimizer,scheduler,scaler,identity):
    r=b.read(directory/'journal.json')
    need(r['identity']==identity,'resume identity mismatch')
    epoch=r['epoch'];need(type(epoch) is int and 1<=epoch<=identity['epochs'],'resume epoch mismatch')
    need(r['path']==f'epoch-{epoch:02d}.pt','resume path mismatch')
    expected={f'swa-{e:02d}.pt' for e in identity['swa_epochs'] if e<=epoch}
    need(set(r['swa_members'])==expected,'resume SWA membership mismatch')
    for name,pin in {r['path']:r['sha256'],**r['swa_members']}.items():
        need(b.sha(directory/name)==pin,'resume bytes mismatch')
    ck=torch.load(directory/r['path'],map_location='cpu',weights_only=True)
    need(ck['identity']==identity and ck['epoch']==epoch,'resume checkpoint mismatch')
    need([h['epoch'] for h in ck['history']]==list(range(1,epoch+1)) and
         all(h['updates']==identity['updates_per_epoch'] and h['skips']==0 for h in ck['history']), 'resume history mismatch')
    model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
    return epoch,ck['history'],r['swa_members']

def train_epoch(model,loader,optimizer,scheduler,scaler,weights,recipe,device,precision,offset=0):
    model.train();optimizer.zero_grad(set_to_none=True)
    total=0.;updates=0
    PROGRESS.phase_start('training_step')
    for i,batch in enumerate(loader):
        x,y,mask,_=mixed.unpack_batch(batch)
        final,count=accumulation_group(i,len(loader),16)
        with base.amp_context(device,precision):
            logits=model(x.to(device)).float()
            loss=mixed.masked_study_bce(logits,y.to(device),mask.to(device),weights)
            need(bool(torch.isfinite(loss)),'nonfinite soft BCE')
        scaler.scale(loss/count).backward();total+=float(loss.detach())
        if final:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['clip'],error_if_nonfinite=True)
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
            need(scaler.get_scale()>=previous,'optimizer skipped update')
            scheduler.step();optimizer.zero_grad(set_to_none=True);updates+=1
            PROGRESS.completed('training_step',{'update':offset+updates})
    need(updates==(len(loader)+15)//16,'update count differs')
    return dict(loss=total/len(loader),updates=updates,batches=len(loader),skips=0)

def run(device,mode,resume=False):
    need(device in ('cpu','cuda') and mode in ('smoke','fit'),'execution mode')
    need(mode!='fit' or device=='cuda','CPU full fit forbidden')
    if device=='cuda':
        b.require_gpu_job(mode,True);b.require_pre_review()
        if mode=='fit':b.require_fit()
    b.validate();base.install_portable_gold_blocker();torch.set_num_threads(4);configure()
    cfg=b.scientific();factory=c.Config(**cfg['factory']);ids,labels=b.population()
    full_ids=ids
    if mode=='smoke': ids=ids[:2 if device=='cpu' else 16]
    epochs=6 if mode=='fit' else 1;swa_epochs=[4,5,6] if mode=='fit' else [1]
    folder=b.RUN if mode=='fit' else b.WORK/('cpu-run' if device=='cpu' else 'cuda-run')
    work=folder/'epochs';work.mkdir(parents=True,exist_ok=True)
    if mode=='fit': need(not (folder/'summary.json').exists(),'completed fit is immutable')
    seed=44;precision='bf16' if device=='cuda' else 'none';measurements={}
    started=time.monotonic();desktop=None
    if device=='cuda':
        from tools.hb151_build_v3 import desktop_memory_snapshot
        desktop=desktop_memory_snapshot();torch.cuda.set_device(0)
        need('4070 Ti SUPER' in torch.cuda.get_device_name(0),'registered 4070 only')
        need(torch.cuda.is_bf16_supported(),'native bf16 required')
    else: need(not torch.cuda.is_initialized(),'CPU initialized CUDA')
    corpus=base.Corpus(b.ROOT/b.CORPUS)
    # Authenticate actual reads (CPU-smoke subset or full scientific population).
    PROGRESS.phase_start('input_verification')
    print('HB157 input verification',flush=True)
    study_hashes=verify_studies(corpus,ids)
    PROGRESS.completed('input_verification')
    base._cpu_seed(seed)
    if device=='cuda': torch.cuda.manual_seed_all(seed)
    model=c.build_model(factory);initial=load_parent(model,b.ROOT/b.PARENT)
    print('HB157 authenticated public R loaded',flush=True)
    model.backbone.set_grad_checkpointing(False);model.to(device)
    optimizer,scheduler,scaler=make_optimization(model,cfg['recipe'],cfg['total_updates'],precision)
    dataset=L2Dataset(b.ROOT/b.CORPUS,ids,labels,seed,factory)
    sampler=DistributedSampler(dataset,1,0,shuffle=True,seed=seed,drop_last=True)
    workers=4 if mode=='fit' else 0
    loader=DataLoader(dataset,batch_size=1,sampler=sampler,drop_last=True,num_workers=workers,
        persistent_workers=workers>0,worker_init_fn=worker_init,collate_fn=mixed.collate,pin_memory=device=='cuda')
    identity=dict(protocol_sha256=b.sha(b.PROTOCOL),run_dir=folder.relative_to(b.ROOT).as_posix(),
        mode=mode,device=device,seed=seed,epochs=epochs,swa_epochs=swa_epochs,
        updates_per_epoch=(len(ids)+15)//16,workers=workers,train_roster_sha256=b.digest(ids),
        exposure='EXPOSED',clean_oof=False)
    start=0;history=[];swa={};resume_roundtrip=False
    if resume and (work/'journal.json').exists():
        start,history,swa=restore(work,model,optimizer,scheduler,scaler,identity)
        if 0<start<epochs and workers: iter(loader)
    elif resume: need(not list(work.glob('epoch-*.pt')),'orphan epoch exists without committed journal')
    def begin():
        if device=='cuda': torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        return time.monotonic()
    def end(phase,tick,units):
        if device=='cuda': torch.cuda.synchronize()
        measurements[phase]=dict(peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0,
                                 seconds_per_unit=(time.monotonic()-tick)/units)
    try:
        for epoch in range(start,epochs):
            base._cpu_seed(seed*1000+epoch)
            if device=='cuda': torch.cuda.manual_seed_all(seed*1000+epoch)
            dataset.set_epoch(16+epoch);sampler.set_epoch(16+epoch)
            tick=begin()
            print(f'HB157 epoch {epoch+1} begin ({device}; {len(ids)} studies)',flush=True)
            record=train_epoch(model,loader,optimizer,scheduler,scaler,torch.tensor(cfg['pos_weight']),
                               cfg['recipe'],device,precision,sum(h['updates'] for h in history))
            end('train',tick,record['updates']);history.append(dict(epoch=epoch+1,data_epoch=16+epoch,**record))
            print(f'HB157 epoch {epoch+1} committed {record["updates"]} updates',flush=True)
            if epoch+1 in swa_epochs:
                name=f'swa-{epoch+1:02d}.pt';save(work/name,dict(model=model.state_dict(),epoch=epoch+1));swa[name]=b.sha(work/name)
            # GPU smoke avoids optimizer IO; CPU exercises the real resume journal.
            if mode=='fit' or device=='cpu':
                journal(work,model,optimizer,scheduler,scaler,identity,epoch+1,history,swa)
                if device=='cpu':
                    before=state_dict_sha256(model.state_dict())
                    epoch_out,hist_out,swa_out=restore(work,model,optimizer,scheduler,scaler,identity)
                    need(epoch_out==epoch+1 and hist_out==history and swa_out==swa and
                         before==state_dict_sha256(model.state_dict()),'CPU resume roundtrip failed')
                    resume_roundtrip=True
    finally:
        if getattr(loader,'_iterator',None) is not None: loader._iterator._shutdown_workers()
    need(sum(h['updates'] for h in history)==epochs*identity['updates_per_epoch'],'total update count')
    del optimizer,scheduler,scaler,loader;gc.collect()
    states=[]
    for epoch in swa_epochs:
        name=f'swa-{epoch:02d}.pt';need(b.sha(work/name)==swa[name],'SWA bytes changed')
        states.append(torch.load(work/name,map_location='cpu',weights_only=True)['model'])
    model.load_state_dict(swa_average(states),strict=True);del states
    bn_ids=ids if mode=='fit' else ids[:2]
    bn_ds=L2Dataset(b.ROOT/b.CORPUS,bn_ids,labels,seed,factory,augment=False);bn_ds.set_epoch(22)
    bn_loader=DataLoader(bn_ds,batch_size=2,shuffle=False,num_workers=workers,worker_init_fn=worker_init,
                         collate_fn=mixed.collate,pin_memory=device=='cuda')
    def bn_batches():
        PROGRESS.phase_start('bn_replay')
        for i,batch in enumerate(mixed.bn_batches(bn_loader)):
            yield batch
            PROGRESS.completed('bn_replay',{'progress_kind':'bn_replay','cumulative_units':i+1})
    tick=begin();bn=recompute_bn(model,bn_batches(),device,base.amp_context(device,precision))
    print('HB157 SWA and BN complete',flush=True)
    end('bn',tick,bn['batches']);need(bn['studies']==bn['unique_studies']==len(bn_ids),'BN population differs')
    final_hash=state_dict_sha256(model.state_dict())
    payload=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        arch=base.ARCH,res=384,inference_res=320,lab=cfg['targets'],contract=base.CONTRACT,k_eval=94,
        norm='imagenet',window_policy='legacy',oai=False,checkpoint_kind='fixed-swa-bn',
        swa_epochs_1based=swa_epochs,seed=44,exposure='EXPOSED',clean_oof=False,
        provenance=dict(identity,public_parent_sha256=b.PARENT_SHA,initial_state_sha256=initial,
            soft_label_sha256=b.LABEL_SHA,pos_weight=cfg['pos_weight'],bn=bn,bn_epoch=22,
            swa_members=swa,study_hashes=study_hashes,full_train_roster_sha256=b.digest(full_ids)))
    save(folder/'model_best.pt',payload);del model,payload;gc.collect()
    if device=='cuda': torch.cuda.empty_cache()
    tick=begin();model=c.build_model(factory,320).to(device).eval()
    checkpoint=torch.load(folder/'model_best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    need(state_dict_sha256(model.state_dict())==final_hash,'strict serving reload differs')
    PROGRESS.phase_start('serving_mirror')
    print('HB157 strict reload; K94 mirror readout',flush=True)
    plain,mirror=c.infer_pair(model,factory,*corpus.get(ids[0]),device=device,
                             precision='fp16' if device=='cuda' else 'none',chunk=16 if device=='cuda' else 2)
    need(np.isfinite(plain).all() and np.isfinite(mirror).all(),'nonfinite serving pair')
    PROGRESS.completed('serving_mirror');end('serving',tick,1)
    b.validate();need(verify_studies(corpus,ids)==study_hashes,'study bytes changed')
    summary=dict(status='PASS' if mode=='smoke' else 'COMPLETE_TRAINING_AND_READOUT',protocol_sha256=b.sha(b.PROTOCOL),
        model_sha256=b.sha(folder/'model_best.pt'),identity=identity,history=history,bn_receipt=bn,
        real_data=True,backward=True,resume_roundtrip=resume_roundtrip,swa=True,bn=True,
        strict_reload=True,serving_pair=True,exposure='EXPOSED',clean_oof=False,
        elapsed_seconds=time.monotonic()-started,measurements=measurements,
        observation=observation(device),updates=sum(h['updates'] for h in history),bn_batches=bn['batches'],
        shape=dict(batch=1,k=12,res=384,accumulation=16,bn_batch=2,k_eval=94,crop=320,mirror=True,precision='bf16',checkpointing=False))
    if device=='cpu':
        summary['shape'].update(precision='none')
        summary['engineering_smoke_only']=True
        summary['actual_accumulation_tail']=2
        summary['cuda_initialized']=torch.cuda.is_initialized();need(not summary['cuda_initialized'],'CPU CUDA side effect')
        atomic_json(b.WORK/'cpu-smoke.json',summary)
    elif mode=='smoke':
        prop=torch.cuda.get_device_properties(0)
        summary.update(device_name=prop.name,total_memory_mib=prop.total_memory/1024**2,desktop_before_cuda=desktop)
        b.check_smoke(summary);atomic_json(b.WORK/'cuda-smoke.json',summary)
    atomic_json(folder/'summary.json',summary)
    return {k:v for k,v in summary.items() if k not in ['history','identity','observation','measurements']}
