"""K1 v3-G 3.0.0 CPU-only real-cache qualification; fixed project scratch.

Changelog v3.0.0: four optimizer groups, full Net manifest, workers=1 real-loader smoke.
Reused v2.0.0: v1.0.0 smoke against the K1 v2 derivative (registered LR/epoch/
warm-frac argv) plus: best-epoch checkpoint copy identical across uninterrupted,
mid-epoch and epoch-boundary resume; real trainer collapse-guard stop (exit 4 after
the committed epoch checkpoint, collapse.json, no model) and re-stop on resume with
no further optimizer step; warm-up length and optimizer base LRs read back.
v1.0.0: generated trainer mid-epoch/epoch resume, full Net backward and serving.
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
import re
import sys
import time

from tools import l1_rco_k1_v3s1 as k

VERSION = '3.0.0'
GENERATED = ('ckpt/E_prior_s0_f0.pt', 'ckpt/best_E_prior_s0_f0.pt', 'ckpt/best.json', 'ckpt/epochs.json',
             'collapse.json', 'fit-complete.json', 'models/E_prior_s0_f0.pt')
SMOKE_GUARD = 1e-6  # forces the registered rule to fire at its first eligible epoch on the tiny fixture


def equal_state(a, b):
    import torch
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[x], b[x]) for x in a)
    if isinstance(a, (list, tuple)): return type(a) == type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def remove_smoke_outputs(out):
    # Only our generated CPU files, fixed names; no recursive deletion or source data.
    for name in GENERATED:
        path = Path(out) / name
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
    def execute(out, stop_steps=0, stop_epoch=-1, guard=None):
        argv = k.trainer_argv(cache, reg)[1:]
        for flag, value in {'--out':str(out),'--targets':str(target),'--workers':'0',
                            '--epochs':'3','--batch':'1','--train-windows':'2'}.items():
            argv[argv.index(flag)+1] = value
        argv.remove('--grad-ckpt')
        i = argv.index('--l1-guard-constant'); del argv[i:i+2]
        if guard is not None: argv += ['--l1-guard-constant', repr(guard)]
        argv += ['--l1-cpu-test','--l1-stop-steps',str(stop_steps),'--stop-after-epoch',str(stop_epoch)]
        old = sys.argv; sys.argv = argv
        log = io.StringIO(); code = 0
        try:
            with contextlib.redirect_stdout(log): ct.main()
        except SystemExit as exc: code = exc.code
        finally: sys.argv = old
        out.mkdir(parents=True, exist_ok=True)
        with (out/'smoke.log').open('a', encoding='utf-8') as stream: stream.write(log.getvalue())
        return code, log.getvalue()

    load = lambda path: torch.load(path, map_location='cpu', weights_only=False)
    base, mid, boundary, guarded = [work / n for n in ('uninterrupted','mid-resume','epoch-resume','guard-stop')]
    for out in (base,mid,boundary,guarded): remove_smoke_outputs(out)
    code, text = execute(base)
    k.need(code==0, 'uninterrupted CPU trainer failed')
    warm = re.findall(r'warm-up steps (\d+) of (\d+) \(([\d.]+) epochs, warm-frac ([\d.e-]+)\)', text)
    k.need(warm == [('1','6','0.50',repr(reg['recipe']['warm_frac']))], 'warm-frac did not reach the trainer scheduler')
    k.need(execute(mid,stop_steps=1)[0]==3, 'mid-epoch stop missing')
    ck = load(mid/'ckpt/E_prior_s0_f0.pt')
    k.need(ck['epoch']==0 and ck['batch']==1 and ck['step_total']==1, 'not a mid-epoch checkpoint')
    del ck
    k.need(execute(mid)[0]==0, 'mid-epoch resume failed')
    k.need(execute(boundary,stop_epoch=0)[0]==3, 'epoch-boundary stop missing')
    ck = load(boundary/'ckpt/E_prior_s0_f0.pt')
    k.need(ck['epoch']==1 and ck['batch']==0, 'not an epoch-boundary checkpoint')
    del ck
    k.need(execute(boundary)[0]==0, 'epoch-boundary resume failed')
    expected = load(base/'ckpt/E_prior_s0_f0.pt')
    k.need([g['lr'] for g in expected['opt']['param_groups']] != [] and expected['sched']['base_lrs'] == [reg['recipe']['lr'], reg['recipe']['lr'], reg['recipe']['head_lr'], reg['recipe']['head_lr']],
           'registered encoder/head LR did not reach the optimizer')
    base_log = expected['log']; base_best = k.read(base/'ckpt/best.json')
    best_epoch = min(range(len(base_log)), key=lambda e: (base_log[e], e))
    best_ck = load(base/'ckpt/best_E_prior_s0_f0.pt')
    k.need(base_best['epoch']==best_epoch and base_best['sha256']==k.sha(base/'ckpt/best_E_prior_s0_f0.pt') and
           best_ck['epoch']==best_epoch+1 and best_ck['batch']==0 and best_ck['log']==base_log[:best_epoch+1], 'best-epoch checkpoint copy')
    sidecar = k.read(base/'ckpt/epochs.json')
    k.need(sidecar['log']==base_log and sidecar['committed_epochs']==3 and sidecar['guard_constant'] is None, 'epoch log sidecar')
    for out in (mid,boundary):
        actual = load(out/'ckpt/E_prior_s0_f0.pt')
        k.need(equal_state(expected,actual), 'resume state differs from uninterrupted')
        best = k.read(out/'ckpt/best.json')
        k.need({x:best[x] for x in ('epoch','loss','log','step_total')}=={x:base_best[x] for x in ('epoch','loss','log','step_total')} and
               equal_state(best_ck, load(out/'ckpt/best_E_prior_s0_f0.pt')), 'best-epoch copy differs after resume')
        pool = k.read(out/'pool_uids.json')
        k.need(pool==train and not set(pool)&set(holdout), 'holdout in training artifact')
    del expected,actual,best_ck
    # Real trainer collapse guard: the registered rule fires at its first eligible epoch.
    want = k.collapse_reason(base_log, SMOKE_GUARD)
    k.need(want is not None, 'smoke guard fixture must fire')
    code, text = execute(guarded, guard=SMOKE_GUARD)
    k.need(code==k.GUARD['exit_code'] and 'COLLAPSE GUARD' in text, 'trainer collapse guard did not stop with exit 4')
    stop = k.read(guarded/'collapse.json'); ck = load(guarded/'ckpt/E_prior_s0_f0.pt')
    k.need(stop['epoch']==want['epoch'] and stop['checkpoint_kept'] and ck['epoch']==want['epoch']+1 and ck['batch']==0 and
           ck['log']==base_log[:want['epoch']+1] and not (guarded/'models/E_prior_s0_f0.pt').exists() and
           not (guarded/'fit-complete.json').exists() and (guarded/'ckpt/best_E_prior_s0_f0.pt').exists(), 'collapse stop state')
    before = k.sha(guarded/'ckpt/E_prior_s0_f0.pt'); steps = ck['step_total']; del ck
    code, text = execute(guarded, guard=SMOKE_GUARD)
    ck = load(guarded/'ckpt/E_prior_s0_f0.pt')
    k.need(code==k.GUARD['exit_code'] and 'COLLAPSE GUARD' in text and k.sha(guarded/'ckpt/E_prior_s0_f0.pt')==before and
           ck['step_total']==steps, 'collapsed run resumed or stepped')
    del ck
    # Check the production trainer's real 3190 target parser, fold assertions and the
    # registered v2 argument check without the cache payload. Only metadata is mocked.
    meta_cache = work/'metadata-only'
    k.write(meta_cache/'index.json',dict(done={u:[0,1,1] for u in reg['train']}))
    old = sys.argv; sys.argv = k.trainer_argv(meta_cache, reg)[1:] + ['--check-targets']
    try:
        with (work/'check-targets.log').open('w',encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
            try: ct.main()
            except SystemExit as exc: k.need(exc.code==0, 'production target parser failed')
    finally: sys.argv=old
    # Exact production Net and spatial352 path on real cached image windows.
    ct.Net = real_net
    torch.manual_seed(1017)
    model = real_net('convnext_small',str(k.ROOT/k.s.INIT),crop=352,in_chans=3,spatial=True,extent=False)
    from tools.l1_rco_k1_optim_v3 import parameter_groups, manifest_digest
    full_groups, full_manifest = parameter_groups(model)
    k.need(full_manifest['trainable_tensors'] == 360 and all(g['params'] for g in full_groups), 'full Net group coverage')
    k.write(work/'full-net-parameter-groups.json', full_manifest)
    model.enc.set_grad_checkpointing(True); model.train()
    ds = ct.Studies(str(cache),train,2,'train',9017,train_sampling='cover',slices=3,phys_mm=3.5,slice_jitter=.25)
    ds.epoch=0
    # Same retained v2 shim mechanism, exercised on real data with Windows spawn.
    dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=1, collate_fn=ct.collate)
    worker_batches = list(dl)
    k.need(len(worker_batches) == 2, 'workers>0 real-study loader smoke')
    del dl, worker_batches
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
                resume_compared='model/optimizer/scheduler/EMA/loader/augmentation/CPU RNG/loss log/counters + best-epoch copy',
                best_epoch_copy_exact_after_resume=True,best_epoch=best_epoch,
                warm_up_logged=dict(steps=1,total=6,warm_frac=reg['recipe']['warm_frac']),
                optimizer_base_lrs=[reg['recipe']['lr'],reg['recipe']['lr'],reg['recipe']['head_lr'],reg['recipe']['head_lr']],
                guard_stop=dict(exit_code=k.GUARD['exit_code'],epoch=want['epoch'],criteria=want['criteria'],
                                checkpoint_epoch_committed=want['epoch']+1,no_model_export=True,resume_restops_without_step=True,
                                fixture_constant=SMOKE_GUARD),
                train3190_target_parser=True,registered_v3_trainer_args_checked=True,holdout_training_artifact_rows=0,
                full_net_parameter_tensors=full_manifest['trainable_tensors'],
                full_net_parameter_manifest_sha256=manifest_digest(full_manifest),real_loader_workers=1,
                full_net_backward_finite=True,full_net_real_serving_rows=1,incomplete_cache_refused=True,
                full_net_test_windows=2,real_cache_receipt_sha256=k.sha(cache/'receipt.json'),
                prepared_trainer_sha256=k.sha(k.WORK/'compact_train_l1.py'),prepared_targets_sha256=k.sha(k.WORK/'targets.csv'))
    k.write(k.WORK/'cpu-smoke.json',result)
    return result
