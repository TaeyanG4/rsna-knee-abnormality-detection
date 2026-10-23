"""H216 v1.0.0 fixed staged-transfer training core; no launch entrypoint.

Both arms:400 temporary-head adaptation updates, then800 RSNA recovery updates.
Only encoder stage3 transfers across phases; recovery restores the complete
original disease head. Input/admission and final976 evaluation remain separate.
No model/checkpoint/data is loaded by importing this module.
"""
from __future__ import annotations
import copy
import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tools.l1_rco_k1_optim_v3 import parameter_groups, manifest_digest

VERSION = '1.0.0'
BASE_SHA = '6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5'
RECIPE = dict(version=VERSION, base_sha256=BASE_SHA, adaptation_steps=400, recovery_steps=800,
              batch=2, n_windows=64, crop=352, views='v4', train_sampling='cover', cover_stride=3,
              slice_jitter=.25, phys_mm=3.5, augmentation='original random crop only; no flips',
              encoder_lr=1e-4, head_lr=3e-4, weight_decay=.02, clip=5., ema=.999,
              warm_fraction=.1, lr_tail='cosine to zero', adapt_seed=21601, recovery_seed=21602,
              trainable_encoder_prefix='enc.stages.3.', control_targets=12, oai_tasks=30, oai_categories=4,
              transition='apply adaptation EMA; retain only encoder stage3; restore original full RSNA head',
              recovery='reset optimizer/scheduler/EMA/RNG; same RSNA order,windows,crop/dropout seeds in both arms')


def need(ok, code):
    if not ok: raise ValueError(code)


def reset_rng(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # CPU preparation must not initialize CUDA just to seed it.
    if torch.cuda.is_initialized(): torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed + 71)


def batch_schedule(n, steps, seed):
    """Exactly paired recovery order, independent of adaptation consumption."""
    need(n >= 2 and steps > 0, 'SCHEDULE_CARDINALITY')
    rng = np.random.default_rng(seed); out=[]; epoch=0
    while len(out) < steps:
        order = rng.permutation(n)
        for i in range(0,n-1,2):
            out.append((epoch, [int(order[i]),int(order[i+1])]))
            if len(out) == steps: break
        epoch += 1
    return out


class TemporaryAdapter(nn.Module):
    """Identical pooled-feature head family in both arms; outputs12 or30x4.

    Shares only the base encoder. The disease attention/head is never repurposed
    as OAI labels; its original parameters remain untouched during adaptation.
    """
    def __init__(self, base, outputs, hidden=384):
        super().__init__()
        need(outputs in (12,120), 'TEMP_OUTPUTS')
        self.enc = base.enc
        d = self.enc.num_features
        self.temp_norm = copy.deepcopy(base.sp_head_norm) if base.sp_head_norm is not None else nn.Identity()
        self.proj = nn.Sequential(nn.Linear(d,hidden),nn.LayerNorm(hidden))
        self.slot = nn.Embedding(10,hidden); self.view = nn.Embedding(6,hidden)
        self.geom = nn.Linear(4,hidden,bias=False)
        self.att = nn.Linear(hidden,1,bias=False); self.drop=nn.Dropout(.1)
        self.cls = nn.Linear(hidden,outputs)
        for layer in (self.slot,self.view,self.geom,self.att): nn.init.zeros_(layer.weight)
        self.register_buffer('mean',base.mean.detach().clone()); self.register_buffer('std',base.std.detach().clone())

    def forward(self,img,meta,mask):
        need(mask.ndim==2 and bool(mask.any(1).all()),'EMPTY_STUDY')
        x=img[mask].float()/255.
        f=self.enc(((x-self.mean)/self.std).contiguous(memory_format=torch.channels_last))
        need(f.ndim==4,'ENCODER_MAP_EXPECTED')
        f=f.mean((-2,-1)).float()
        if not isinstance(self.temp_norm,nn.Identity):
            n=self.temp_norm; f=F.layer_norm(f,(f.shape[-1],),n.weight,n.bias,n.eps)
        f=self.proj(f)
        tokens=f.new_zeros((*mask.shape,f.shape[-1])); tokens[mask]=f
        tokens=self.drop(tokens+self.slot(meta[...,0].long())+self.view(meta[...,1].long())+self.geom(meta[...,2:6]))
        weights=self.att(torch.tanh(tokens)).squeeze(-1).float().masked_fill(~mask,float('-inf')).softmax(1)
        return self.cls(torch.einsum('bw,bwh->bh',weights.to(tokens.dtype),tokens)).float()


def configure_trainable(model, adaptation):
    need(not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()),'UNEXPECTED_BATCHNORM')
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith('enc.stages.3.') or not name.startswith('enc.'))
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    need(any(n.startswith('enc.stages.3.') for n in names),'LAST_STAGE_MISSING')
    if adaptation: need(isinstance(model,TemporaryAdapter),'ADAPTATION_HEAD_REQUIRED')
    return names


def train_mode(model):
    model.train()
    model.enc.eval()
    model.enc.stages[3].train()


def original_head(base):
    return {n:v.detach().clone() for n,v in base.state_dict().items() if not n.startswith('enc.')}


def restore_head(base, state):
    current=base.state_dict()
    need(set(state)=={n for n in current if not n.startswith('enc.')},'HEAD_KEYSET')
    with torch.no_grad():
        for name,value in state.items():
            need(current[name].shape==value.shape,'HEAD_SHAPE')
            current[name].copy_(value)
    need(all(torch.equal(base.state_dict()[n],v) for n,v in state.items()),'HEAD_RESTORE')


def native_loss(logits, grades):
    """Mean of per-task masked categorical CE; -1 is unknown, never class0."""
    need(logits.ndim==2 and logits.shape[1]==120 and grades.shape==(len(logits),30),'NATIVE_SHAPE')
    need(grades.dtype==torch.long and bool(((grades>=-1)&(grades<=3)).all()),'NATIVE_GRADES')
    z=logits.reshape(-1,30,4); known=grades>=0
    need(bool(known.any()),'NO_OBSERVED_NATIVE_CELLS')
    losses=F.cross_entropy(z.transpose(1,2),grades.clamp_min(0),reduction='none')
    per=(losses*known).sum(0)/known.sum(0).clamp_min(1)
    return per[known.any(0)].mean()


def rsna_loss(logits, targets):
    need(logits.shape==targets.shape and logits.shape[1]==12 and bool(torch.isfinite(targets).all()) and
         bool(((targets>=0)&(targets<=1)).all()),'RSNA_TARGET_CONTRACT')
    # Equivalent to the unchanged E_prior bce_soft with all-one weights.
    return F.binary_cross_entropy_with_logits(logits,targets,reduction='none').mean(0).mean()


def crop(img, gen):
    s=img.shape[-1]; need(img.shape[-2]==s and s>=352,'CROP_SHAPE')
    y=int(torch.randint(0,s-352+1,(1,),generator=gen)); x=int(torch.randint(0,s-352+1,(1,),generator=gen))
    return img[...,y:y+352,x:x+352]


def optimizer(model,steps):
    groups,manifest=parameter_groups(model)
    opt=torch.optim.AdamW(groups,weight_decay=.02)
    warm=max(1,int(RECIPE['warm_fraction']*steps))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda i: (i+1)/warm if i<warm else
                  .5*(1+math.cos(math.pi*(i-warm)/max(1,steps-warm))))
    return opt,sched,manifest


def ema_start(model):
    return {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def ema_update(model,ema,step):
    d=min(.999,(1+step)/(10+step))
    for n,p in model.named_parameters():
        if n in ema: ema[n].mul_(d).add_(p.detach(),alpha=1-d)


@torch.no_grad()
def ema_apply(model,ema):
    params=dict(model.named_parameters())
    for n,p in ema.items(): params[n].copy_(p)


def phase(model,ds,targets,ct,*,adaptation,native,device,progress,checkpoint):
    """Runs the fixed phase; caller must supply reviewed exclusive admission.

    No orchestration, inference or implicit resume occurs here. The caller owns
    storage/provenance and must supply meaningful progress/checkpoint callbacks.
    """
    need(callable(progress) and callable(checkpoint),'CALLBACKS_REQUIRED')
    need(not native or adaptation,'NATIVE_RECOVERY_FORBIDDEN')
    steps=400 if adaptation else 800; seed=21601 if adaptation else 21602
    gen=reset_rng(seed); configure_trainable(model,adaptation); train_mode(model)
    need(ds.mode=='train' and ds.n_win==64 and ds.train_sampling=='cover' and ds.cover_stride==3 and
         ds.slice_jitter==.25 and ds.phys_mm==3.5 and ds.slices==3,'LOADER_RECIPE')
    ds.seed=seed+1000
    need(len(targets)==len(ds),'TARGET_ROSTER_CARDINALITY')
    order=batch_schedule(len(ds),steps,seed)
    opt,sched,manifest=optimizer(model,steps); ema=ema_start(model); losses=[]
    for step,(epoch,indices) in enumerate(order,1):
        ds.epoch=epoch
        img,meta,mask,idx=ct.collate([ds[i] for i in indices])
        need(idx.tolist()==indices,'BATCH_TARGET_ALIGNMENT')
        img=crop(img,gen).to(device); meta=meta.to(device); mask=mask.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
            logits=model(img,meta,mask)
            target=targets[indices].to(device)
            loss=native_loss(logits,target) if native else rsna_loss(logits,target)
        need(bool(torch.isfinite(loss)),'NONFINITE_LOSS'); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
        opt.step(); sched.step(); ema_update(model,ema,step); losses.append(float(loss.detach()))
        progress(dict(phase='adapt' if adaptation else 'recovery',steps=step,total=steps,
                      loss=losses[-1],grad_norm=float(norm),windows=int(mask.sum())))
        if step%100==0 or step==steps:
            checkpoint(dict(version=VERSION,phase='adapt' if adaptation else 'recovery',step=step,
                model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),ema=ema,
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                crop_rng=gen.get_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                schedule=order,loader_seed=ds.seed,loader_epoch=ds.epoch,losses=losses,
                parameter_manifest=manifest))
    ema_apply(model,ema)
    return dict(steps=steps,loss_first=losses[0],loss_last=losses[-1],manifest=manifest,
                manifest_sha256=manifest_digest(manifest),phase_ema_applied=True)


def pair_arm(base,rsna_ds,oai_ds,rsna_targets,oai_targets,ct,*,arm,device,progress,checkpoint):
    """Base must be freshly loaded from the same immutable completed seed0.

    Only one arm is held on GPU at a time. The outer runner must prove starting
    state identity and input hashes; this core deliberately does not invent them.
    """
    need(arm in ('rsna_control','oai_native'),'ARM')
    head=original_head(base); reset_rng(21600)
    adapter=TemporaryAdapter(base,120 if arm=='oai_native' else 12).to(device)
    ds=oai_ds if arm=='oai_native' else rsna_ds
    targets=oai_targets if arm=='oai_native' else rsna_targets
    adapt=phase(adapter,ds,targets,ct,adaptation=True,native=arm=='oai_native',device=device,
                progress=progress,checkpoint=checkpoint)
    del adapter
    restore_head(base,head); del head
    recovery=phase(base,rsna_ds,rsna_targets,ct,adaptation=False,native=False,device=device,
                   progress=progress,checkpoint=checkpoint)
    return dict(version=VERSION,arm=arm,adaptation=adapt,recovery=recovery,ready_for_efficacy_evaluation=True)
