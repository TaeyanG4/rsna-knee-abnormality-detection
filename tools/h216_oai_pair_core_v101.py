"""H216 pair core v1.0.1: explicit committed-step resume, unchanged recipe.

No launch/checkpoint read on import. The caller authenticates exclusive resource
ownership, immutable base/input/cache and the exact checkpoint file before use.
Original v1 and its CPU evidence remain intact. No automatic retry or resume.
"""
from __future__ import annotations
import copy
import random
import re
import numpy as np
import torch
from tools import h216_oai_pair_core_v1 as v1
from tools.l1_rco_stage0_v1 import digest

VERSION='1.0.1'
RECIPE=v1.RECIPE
BASE_SHA=v1.BASE_SHA
need=v1.need


class PausedAtCommit(RuntimeError):
    """Explicit caller pause saved successfully; not a completed arm."""


def identity(arm,phase,input_contract,manifest,schedule):
    need(arm in ('rsna_control','oai_native'),'ARM')
    need(re.fullmatch('[0-9a-f]{64}',input_contract) is not None,'INPUT_CONTRACT')
    return digest(dict(version=VERSION,recipe=RECIPE,base_sha256=BASE_SHA,arm=arm,
                       phase=phase,input_contract=input_contract,manifest=manifest,schedule=schedule))


def phase(model,ds,targets,ct,*,adaptation,native,device,progress,checkpoint,
          arm,input_contract,resume=None,pause_requested=None,adaptation_result=None):
    """Resume only after a committed update, by indexed schedule (no replay).

    Checkpoint callback must synchronously persist before returning. Pause checks
    occur after optimizer/scheduler/EMA commit. A saved final-step checkpoint is
    pre-EMA-application, so crossing a phase boundary applies EMA exactly once.
    """
    need(callable(progress) and callable(checkpoint),'CALLBACKS_REQUIRED')
    need(not native or adaptation,'NATIVE_RECOVERY_FORBIDDEN')
    need(native==(adaptation and arm=='oai_native'),'ARM_TARGET_MISMATCH')
    steps=400 if adaptation else 800; seed=21601 if adaptation else 21602
    phase_name='adapt' if adaptation else 'recovery'
    gen=v1.reset_rng(seed); v1.configure_trainable(model,adaptation); v1.train_mode(model)
    need(ds.mode=='train' and ds.n_win==64 and ds.train_sampling=='cover' and ds.cover_stride==3 and
         ds.slice_jitter==.25 and ds.phys_mm==3.5 and ds.slices==3,'LOADER_RECIPE')
    ds.seed=seed+1000; need(len(targets)==len(ds),'TARGET_ROSTER_CARDINALITY')
    order=v1.batch_schedule(len(ds),steps,seed)
    opt,sched,manifest=v1.optimizer(model,steps); ema=v1.ema_start(model); losses=[]; start=0
    ident=identity(arm,phase_name,input_contract,manifest,order)
    if resume is not None:
        need(resume['version']==VERSION and resume['identity']==ident and resume['arm']==arm and
             resume['phase']==phase_name and resume['input_contract']==input_contract,'RESUME_IDENTITY')
        need(resume['parameter_manifest']==manifest and resume['schedule']==order and
             resume['loader_seed']==ds.seed,'RESUME_SCHEDULE_OR_PARAMETERS')
        start=resume['step']; losses=list(resume['losses'])
        need(type(start) is int and 0<start<=steps and len(losses)==start and
             np.isfinite(losses).all() and resume['loader_epoch']==order[start-1][0],'RESUME_COMMIT')
        need(resume['scheduler']['last_epoch']==start,'RESUME_SCHEDULER_COMMIT')
        need(set(resume['ema'])==set(ema),'RESUME_EMA_KEYS')
        model.load_state_dict(resume['model'],strict=True)
        opt.load_state_dict(resume['optimizer']); sched.load_state_dict(resume['scheduler'])
        for n in ema:
            need(ema[n].shape==resume['ema'][n].shape,'RESUME_EMA_SHAPE')
            ema[n].copy_(resume['ema'][n].to(ema[n].device))
        gen.set_state(resume['crop_rng'].cpu()); torch.set_rng_state(resume['cpu_rng'].cpu())
        if str(device).startswith('cuda'):
            need(len(resume['cuda_rng'])==torch.cuda.device_count(),'RESUME_CUDA_RNG')
            torch.cuda.set_rng_state_all([r.cpu() for r in resume['cuda_rng']])
        else: need(not resume['cuda_rng'],'RESUME_DEVICE_CHANGED')
        np.random.set_state(resume['numpy_rng']); random.setstate(resume['python_rng'])
    for offset in range(start,steps):
        step=offset+1; epoch,indices=order[offset]; ds.epoch=epoch
        img,meta,mask,idx=ct.collate([ds[i] for i in indices])
        need(idx.tolist()==indices,'BATCH_TARGET_ALIGNMENT')
        img=v1.crop(img,gen).to(device); meta=meta.to(device); mask=mask.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
            logits=model(img,meta,mask); target=targets[indices].to(device)
            loss=v1.native_loss(logits,target) if native else v1.rsna_loss(logits,target)
        need(bool(torch.isfinite(loss)),'NONFINITE_LOSS'); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
        opt.step(); sched.step(); v1.ema_update(model,ema,step); losses.append(float(loss.detach()))
        progress(dict(phase=phase_name,steps=step,total=steps,loss=losses[-1],
                      grad_norm=float(norm),windows=int(mask.sum())))
        pause=bool(pause_requested and pause_requested())
        if step%100==0 or step==steps or pause:
            checkpoint(dict(version=VERSION,identity=ident,input_contract=input_contract,arm=arm,
                phase=phase_name,step=step,model=model.state_dict(),optimizer=opt.state_dict(),
                scheduler=sched.state_dict(),ema=ema,cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else [],
                crop_rng=gen.get_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                schedule=order,loader_seed=ds.seed,loader_epoch=ds.epoch,losses=list(losses),
                parameter_manifest=manifest,adaptation_result=copy.deepcopy(adaptation_result)))
        if pause: raise PausedAtCommit('EXPLICIT_PAUSE_SAVED')
    v1.ema_apply(model,ema)
    return dict(steps=steps,loss_first=losses[0],loss_last=losses[-1],manifest=manifest,
                manifest_sha256=v1.manifest_digest(manifest),phase_ema_applied=True)


def pair_arm(base,rsna_ds,oai_ds,rsna_targets,oai_targets,ct,*,arm,device,progress,
             checkpoint,input_contract,resume=None,pause_requested=None):
    """Both normal and resumed invocations require the immutable original base."""
    need(arm in ('rsna_control','oai_native'),'ARM')
    need(resume is None or resume['arm']==arm and resume['phase'] in ('adapt','recovery'),'RESUME_ARM_PHASE')
    if resume is not None and resume['phase']=='recovery':
        adapt=resume['adaptation_result']
        need(isinstance(adapt,dict) and adapt.get('steps')==400 and adapt.get('phase_ema_applied') is True and
             adapt.get('manifest_sha256')==v1.manifest_digest(adapt['manifest']),'RESUME_ADAPTATION_RECEIPT')
    else:
        head=v1.original_head(base); v1.reset_rng(21600)
        adapter=v1.TemporaryAdapter(base,120 if arm=='oai_native' else 12).to(device)
        adapt=phase(adapter,oai_ds if arm=='oai_native' else rsna_ds,
                    oai_targets if arm=='oai_native' else rsna_targets,ct,
                    adaptation=True,native=arm=='oai_native',device=device,progress=progress,
                    checkpoint=checkpoint,arm=arm,input_contract=input_contract,resume=resume,
                    pause_requested=pause_requested)
        del adapter; v1.restore_head(base,head); del head
    recovery=phase(base,rsna_ds,rsna_targets,ct,adaptation=False,native=False,device=device,
                   progress=progress,checkpoint=checkpoint,arm=arm,input_contract=input_contract,
                   resume=resume if resume is not None and resume['phase']=='recovery' else None,
                   pause_requested=pause_requested,adaptation_result=adapt)
    return dict(version=VERSION,arm=arm,adaptation=adapt,recovery=recovery,
                ready_for_efficacy_evaluation=True)
