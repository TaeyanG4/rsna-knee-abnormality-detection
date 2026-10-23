"""K1 v1.0.0 CPU-only real-cache qualification; fixed project scratch.

Changelog: generated trainer mid-epoch/epoch resume, full Net backward and serving.
The tiny resume fixture exercises actual loader/augment/optimizer/scheduler/EMA;
the separately tested full ConvNeXt is never presented as a trained candidate.
"""
from __future__ import annotations
import contextlib
import csv
import gc
import io
import os
from pathlib import Path
import sys
import time

from tools import l1_rco_k1_v1 as k

VERSION = '1.0.0'


def equal_state(a, b):
    import torch
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[x], b[x]) for x in a)
    if isinstance(a, (list, tuple)): return type(a) == type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def remove_smoke_checkpoint(out):
    # Only our generated CPU checkpoint; no recursive deletion or source data.
    path = Path(out) / 'ckpt/E_prior_s0_f0.pt'
    if path.exists():
        for attempt in range(40):
            try: path.unlink(); break
            except PermissionError:
                if attempt == 39: raise
                time.sleep(min(.05 * (attempt + 1), 1))


def cpu_smoke(reg, _production_cache=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from tools.l1_rco_k1_state_v1 import atomic_torch_save
    k.validate(reg); torch.set_num_threads(2)
    k.need(not torch.cuda.is_initialized(), 'CPU smoke must not initialize CUDA')
    work = k.WORK / 'cpu'; work.mkdir(parents=True, exist_ok=True)
    cache = k.ROOT / '.local/scratch/l1-rco/stage0/cpu-cache-reviewed'
    receipt = k.read(cache / 'receipt.json')
    k.need(receipt['studies'] == 3 and not receipt['cuda_initialized'], 'reviewed real CPU cache')
    for name, expected in receipt['hashes'].items(): k.need(k.sha(cache / name) == expected, 'real cache checksum')
    train = sorted(set(receipt['ids']) & set(reg['train']))
    holdout = sorted(set(receipt['ids']) & set(reg['holdout']))
    k.need(len(train) == 2 and len(holdout) == 1, 'real smoke roster')
    with (k.WORK / 'targets.csv').open(newline='', encoding='utf-8') as stream:
        all_rows = list(csv.DictReader(stream))
    k.need({r[k.s.UID] for r in all_rows} == set(reg['train']) and not {r[k.s.UID] for r in all_rows} & set(reg['holdout']), 'production target identity')
    rows = [r for r in all_rows if r[k.s.UID] in train]
    target = work / 'targets.csv'
    with target.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    ct = k.import_trainer(); real_net = ct.Net

    class TinyNet(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.enc = torch.nn.Sequential(torch.nn.Linear(3,8), torch.nn.Dropout(.2))
            self.head = torch.nn.Linear(8,12); self.bias = torch.nn.Parameter(torch.zeros(12))
            self.has_bn = False
        def forward(self, img, meta, mask, return_aux=False, **kwargs):
            x = img.float().mean((-1,-2)) / 255.
            x = (self.enc(x) * mask[...,None]).sum(1) / mask.sum(1)[:,None]
            z = self.head(x) + self.bias
            return (z, None, None) if return_aux else z
        def deploy_state_dict(self): return {n:p.detach().cpu() for n,p in self.state_dict().items()}

    ct.Net = TinyNet
    def execute(out, stop_steps=0, stop_epoch=-1):
        argv = k.trainer_argv(cache)[1:]
        for flag, value in {'--out':str(out),'--targets':str(target),'--workers':'0',
                            '--epochs':'3','--batch':'1','--train-windows':'2'}.items():
            argv[argv.index(flag)+1] = value
        argv.remove('--grad-ckpt')
        argv += ['--l1-cpu-test','--l1-stop-steps',str(stop_steps),'--stop-after-epoch',str(stop_epoch)]
        old = sys.argv; sys.argv = argv
        log = io.StringIO(); code = 0
        try:
            with contextlib.redirect_stdout(log): ct.main()
        except SystemExit as exc: code = exc.code
        finally: sys.argv = old
        out.mkdir(parents=True, exist_ok=True)
        with (out/'smoke.log').open('a', encoding='utf-8') as stream: stream.write(log.getvalue())
        return code

    base, mid, boundary = [work / n for n in ('uninterrupted','mid-resume','epoch-resume')]
    for out in (base,mid,boundary): remove_smoke_checkpoint(out)
    k.need(execute(base)==0, 'uninterrupted CPU trainer failed')
    k.need(execute(mid,stop_steps=1)==3, 'mid-epoch stop missing')
    ck = torch.load(mid/'ckpt/E_prior_s0_f0.pt',map_location='cpu',weights_only=False)
    k.need(ck['epoch']==0 and ck['batch']==1 and ck['step_total']==1, 'not a mid-epoch checkpoint')
    del ck
    k.need(execute(mid)==0, 'mid-epoch resume failed')
    k.need(execute(boundary,stop_epoch=0)==3, 'epoch-boundary stop missing')
    ck = torch.load(boundary/'ckpt/E_prior_s0_f0.pt',map_location='cpu',weights_only=False)
    k.need(ck['epoch']==1 and ck['batch']==0, 'not an epoch-boundary checkpoint')
    del ck
    k.need(execute(boundary)==0, 'epoch-boundary resume failed')
    expected = torch.load(base/'ckpt/E_prior_s0_f0.pt',map_location='cpu',weights_only=False)
    for out in (mid,boundary):
        actual = torch.load(out/'ckpt/E_prior_s0_f0.pt',map_location='cpu',weights_only=False)
        k.need(equal_state(expected,actual), 'resume state differs from uninterrupted')
        pool = k.read(out/'pool_uids.json')
        k.need(pool==train and not set(pool)&set(holdout), 'holdout in training artifact')
    del expected,actual
    # Check the production trainer's real 3190 target parser and fold assertions
    # without requiring the still-running cache payload. Only metadata is mocked.
    meta_cache = work/'metadata-only'
    k.write(meta_cache/'index.json',dict(done={u:[0,1,1] for u in reg['train']}))
    old = sys.argv; sys.argv = k.trainer_argv(meta_cache)[1:] + ['--check-targets']
    try:
        with (work/'check-targets.log').open('w',encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
            try: ct.main()
            except SystemExit as exc: k.need(exc.code==0, 'production target parser failed')
    finally: sys.argv=old
    # Exact production Net and spatial352 path on real cached image windows.
    ct.Net = real_net
    torch.manual_seed(1000)
    model = real_net('convnext_small',str(k.ROOT/k.s.INIT),crop=352,in_chans=3,spatial=True,extent=False)
    model.enc.set_grad_checkpointing(True); model.train()
    ds = ct.Studies(str(cache),train,2,'train',9000,train_sampling='cover',slices=3,phys_mm=3.5,slice_jitter=.25)
    ds.epoch=0
    img,meta,mask,_ = ct.collate([ds[0]])
    logits = model(img[...,16:368,16:368],meta,mask)
    target_tensor = torch.tensor([[float(rows[0][label+'|prior']) for label in k.s.LABELS]])
    loss = ct.bce_soft(logits,target_tensor,torch.ones_like(target_tensor)); loss.backward()
    mass = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    k.need(torch.isfinite(loss).item() and mass>0, 'full Net CPU backward not finite')
    fullout=work/'full-net'; atomic_torch_save(model.deploy_state_dict(),fullout/'models/E_prior_s0_f0.pt')
    del model,logits,loss; gc.collect()
    served=k.inference(reg,cache,out=fullout,cpu=True,ids=holdout,limit_windows=2)
    k.need(served['holdout_rows']==1 and served['cpu_test'] and not served['cuda_initialized'], 'real CPU serving failed')
    # Exercise launcher refusal before reading any unfinished payload/receipt.
    incomplete=work/'incomplete-cache'; k.write(incomplete/'status.json',dict(phase='render',done=4000,total=4167))
    refused=False
    try: k.complete_cache(reg,incomplete)
    except ValueError as exc: refused='incomplete' in str(exc)
    k.need(refused, 'launcher accepted incomplete cache')
    result=dict(version=VERSION,status='PASS',contract_id=reg['contract_id'],cuda_initialized=torch.cuda.is_initialized(),
                real_train_studies=2,real_holdout_studies=1,epochs=3,optimizer_steps=6,
                mid_epoch_exact_resume=True,epoch_boundary_exact_resume=True,
                resume_compared='model/optimizer/scheduler/EMA/loader/augmentation/CPU RNG/loss log/counters',
                train3190_target_parser=True,holdout_training_artifact_rows=0,
                full_net_backward_finite=True,full_net_real_serving_rows=1,incomplete_cache_refused=True,
                full_net_test_windows=2,real_cache_receipt_sha256=k.sha(cache/'receipt.json'),
                prepared_trainer_sha256=k.sha(k.WORK/'compact_train_l1.py'),prepared_targets_sha256=k.sha(k.WORK/'targets.csv'))
    k.write(k.WORK/'cpu-smoke.json',result)
    return result
