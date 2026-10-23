"""H-B-162 v1.0.1: paired SAM continuation, CPU builder / B-only CUDA.

Changelog 1.0.1: CPU-only opt-in evaluation of the authenticated, complete
v1.0.0 run whose sole failure was its aggregate device-memory admission gate.
Frozen run receipts remain unchanged; breached checkpoints stay DO_NOT_SUBMIT.
Changelog 1.0.0: authenticated parent, effective-batch SAM replay, fixed paired
pilot, release K94 screen, measured memory admission and idle-only watchdog.
This experiment is EXPOSED, clean_oof=false; it is not a release candidate.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hb158_scale_contrast_v1 as serving
from tools import hb145_output_io as io
from training.arms import coatnet_history as history

VERSION = '1.0.1'
BUILDER = 'hb162-builder'
TAG = 'HB162_SAM_F0'
WORK = ROOT / '.local/scratch/hb162/H-B-162'
GATE = 'seed-mean macro >= +0.002 on >= 3/4 rulers against BOTH the AdamW continuation and the untouched parent, no ruler-target delta < −0.003, positive four-ruler mean in both continuation seeds; otherwise DEAD.'
POLICY = dict(seeds=[42, 43], arms=['ADAMW', 'SAM'], epochs=3,
    bb_lr=4.01e-6, head_lr=1.95e-4, wd=.0588, clip=3., rho=.05,
    batch=1, accumulation=16, tail='15 studies, as parent; no padding or dropping',
    train_res=384, train_k=12, train_precision='bf16', grad_checkpointing=False,
    swa_epochs=[2,3], bn_batch=2, bn_epoch=3, augmentation='unchanged parent CONTROL',
    epoch_indices=[0,1,2], seed_mean='arithmetic probability mean before AUC',
    gate=GATE, floor_scope='all four rulers/all 12 targets, seed means AND each seed, against BOTH controls',
    positive_seed_scope='SAM minus matched ADAMW AND untouched parent',
    gpu='NVIDIA GeForce RTX 4070 Ti SUPER', watchdog='900s no progress; 600s WDDM idle signature',
    max_wall_hours=None, exposure='EXPOSED', clean_oof=False)
need, sha, jhash = serving.need, serving.sha, serving.jhash
MEMORY_ERROR = 'ValueError: DO_NOT_SUBMIT: aggregate measured peak exceeds 12 GiB'
# Evaluation-only compatibility for the ONE retained v1.0.0 run. Its controller
# failed before assigning output_sha256; pin the retained manifest explicitly.
LEGACY_REGISTRATION_SHA = 'f097846319f801657b4f6fd15aa04ce7a7d88168ca3d149017b281239f2b02b7'
LEGACY_FULL_OUTPUT_SHA = '2c15807b9454d21b247133e597327be307a2d3f57d23f13ebc948a8bedc89494'
LEGACY_SOURCES = {
    'tools/hb162_sam_f0_v1.py': 'a68270db00e0961d372b5e43deae78b30aa614b80c392f794f1b70997b917c89',
    'tests/test_hb162_sam_f0_v1.py': 'e0ec22299f84f0cdbd51e6cccce5b764f5fc2a266872d9920499e6350e87e53d',
}
LEGACY_BACKUP = ROOT/'.local/backups/hb162-memory-v1.0.0'


def authenticate():
    record = history.verified_arm(history.CONTROL)
    p = record['p']
    for name in ('training/src/exp143_own0949.py', 'training/src/raptor_common.py',
                 'tools/exp112_full_swa.py'):
        need(sha(ROOT/name) == p['source_bindings'][name], 'parent scientific implementation changed: '+name)
    need(p['recipe']['wd'] == POLICY['wd'] and p['recipe']['batch'] == 1
         and p['recipe']['effective_batch'] == 16 and p['recipe']['bn_batch'] == 2,
         'parent recipe mismatch')
    ids, refs, draws, identity = serving.authenticate()
    identity.update(train_ids_sha256=history.uid_hash(p['train_ids']),
                    labels_sha256=jhash(p['train_labels']), weights_sha256=jhash(p['pos_weight']))
    return p, ids, refs, draws, identity


def registration():
    p, ids, _, _, identity = authenticate()
    _, proof = serving.serving(320)
    sources = serving.source_bindings()
    for name in ('tools/hb158_scale_contrast_v1.py', 'tools/hb162_sam_f0_v1.py',
                 'tests/test_hb162_sam_f0_v1.py', 'tools/exp112_full_swa.py'):
        sources[name] = sha(ROOT/name)
    summary = history.read_json(serving.CHECKPOINT.parent/'summary.json')
    # Historical same-card measured work, not a 4090 extrapolation or a cap.
    fit_seconds = sum(h['seconds'] for h in summary['history'])
    bn_seconds = summary['bn']['seconds']
    oof_seconds = history.read_json(serving.CHECKPOINT.parent/history.OOF_RECEIPT)['seconds']
    hours = ((fit_seconds/16)*18 + 4*bn_seconds + 5*oof_seconds)/3600
    return dict(version=VERSION, builder=BUILDER, policy=POLICY, identity=identity,
        ids=ids, train_ids=p['train_ids'], sources=sources, serving_proof=proof,
        timing_projection=dict(device=POLICY['gpu'], hours=hours, planning_range_hours=[hours*.8,hours*1.4],
            formula='18 parent-equivalent training epochs (SAM=2x) + 4 BN passes + 5 K94 panels',
            measured_parent_fit_seconds=fit_seconds, measured_parent_bn_seconds=bn_seconds,
            measured_parent_oof_seconds=oof_seconds, measured_pilot=False,
            limitation='SAM assumed 2x fit; current desktop and replay/I/O overhead unknown; calibration adds cost'),
        novelty='SAM vs matched AdamW continuation; prior SWA/EMA, AUG1, PL and scale tests do not settle SAM')


def registered():
    value = io.read_json(WORK/'registration.json')
    need(value == registration(), 'stale registration; repeat CPU smoke and independent review')
    return value


def seed_all(seed, device):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            None if device == 'cpu' else torch.cuda.get_rng_state())


def restore_rng(state, device):
    import numpy as np
    import torch
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if device != 'cpu':
        torch.cuda.set_rng_state(state[3])


def batchnorm_state(model):
    import torch
    return [(m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
            for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            and m.track_running_stats]


def restore_bn(states):
    import torch
    with torch.no_grad():
        for m, mean, var, count in states:
            m.running_mean.copy_(mean); m.running_var.copy_(var); m.num_batches_tracked.copy_(count)


def effective_step(model, optimizer, batches, loss_fn, arm, device='cpu', progress=lambda *a: None):
    """One AdamW update; replay saved tensors and all RNG, preserving first-pass BN.

    Perturb raw accumulated gradients before clipping. Only ndim>=2 tensors are
    perturbed; every trainable parameter receives the second-pass gradient.
    Exact backup/copy restores weights even on exceptions (no subtract roundoff).
    """
    import torch
    from training.src.exp143_own0949 import amp_context
    need(arm in POLICY['arms'] and 0 < len(batches) <= 16, 'invalid effective batch/arm')
    precision = 'none' if device == 'cpu' else 'bf16'
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_rng = rng_state(device)

    def backward(pass_number):
        total = 0.
        for index, (x,y) in enumerate(batches):
            with amp_context(device, precision):
                loss = loss_fn(model(x.to(device)).float(), y.to(device))/len(batches)
            need(bool(torch.isfinite(loss)), 'nonfinite loss')
            loss.backward(); total += float(loss.detach())
            progress('gradient', dict(pass_number=pass_number, microbatch=index+1))
        return total

    loss = backward(1)
    after_rng = rng_state(device)
    bn = batchnorm_state(model)
    backups = []
    if arm == 'SAM':
        params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2 and p.grad is not None]
        need(bool(params), 'no SAM weight gradients')
        norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm(2) for p in params]), 2)
        need(bool(torch.isfinite(norm)), 'nonfinite SAM gradient norm')
        try:
            with torch.no_grad():
                for p in params:
                    backups.append((p, p.detach().clone()))
                    p.add_(p.grad, alpha=POLICY['rho']/(float(norm)+1e-12))
            optimizer.zero_grad(set_to_none=True)
            restore_rng(start_rng, device)
            backward(2)
        finally:
            with torch.no_grad():
                for p, original in backups:
                    p.copy_(original)
            restore_bn(bn)
            restore_rng(after_rng, device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), POLICY['clip'], error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    progress('optimizer', dict(arm=arm, studies=len(batches)))
    return loss


def save_model(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    io.retry(torch.save, value, temporary)
    io.replace(temporary, path)


def model_and_optimizer(p, device):
    import torch
    from training.src.exp143_own0949 import build_model
    need(sha(serving.CHECKPOINT) == serving.MODEL_SHA, 'parent checkpoint changed')
    model = build_model(384)
    model.load_state_dict(torch.load(serving.CHECKPOINT, map_location='cpu', weights_only=False)['model'], strict=True)
    model.backbone.set_grad_checkpointing(False)
    model.to(device)
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=POLICY['bb_lr']),
        dict(params=[v for k,v in model.named_parameters() if not k.startswith('backbone.')], lr=POLICY['head_lr'])
    ], weight_decay=POLICY['wd'])
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(p['pos_weight'], device=device))
    return model, optimizer, loss


def train_arm(p, arm, seed, device, folder, small=False, progress=lambda *a: None):
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from training.src.exp143_own0949 import StudyDataset, collate, amp_context
    from tools.exp112_full_swa import swa_average, recompute_bn
    ids = p['train_ids'][:2] if small else p['train_ids']
    for uid in ids:
        serving.volume(uid)  # input bytes, not merely corpus manifest
        progress('input_verified', {})
    seed_all(seed, device)
    model, optimizer, loss = model_and_optimizer(p, device)
    dataset = StudyDataset(serving.CORPUS, ids, p['train_labels'], seed)
    sampler = DistributedSampler(dataset, 1, 0, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, drop_last=True,
                        num_workers=0, collate_fn=collate)
    states, timings, updates = [], [], 0
    # Smoke: two steps, each full effective16 with the same two real studies
    # repeated eight times; full-shape training, never scientific pilot evidence.
    epochs = [1,2] if small else [0,1,2]
    for epoch in epochs:
        seed_all(seed*1000+epoch, device)
        dataset.set_epoch(epoch); sampler.set_epoch(epoch)
        start = time.perf_counter()
        batches = [(x,y) for x,y,_ in loader] if small else None
        if small:
            effective_step(model, optimizer, batches*8, loss, arm, device, progress)
            updates += 1
        else:
            pending = []
            for x,y,_ in loader:
                pending.append((x,y))
                if len(pending) == 16:
                    effective_step(model, optimizer, pending, loss, arm, device, progress)
                    updates += 1; pending = []
            if pending:
                effective_step(model, optimizer, pending, loss, arm, device, progress); updates += 1
        timings.append(time.perf_counter()-start)
        if epoch >= 1:
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            states.append(state)
            save_model(folder/f'epoch_{epoch+1}.pt', dict(model=state, epoch=epoch+1))
        progress('epoch_saved', dict(epoch=epoch+1, updates=updates))
    need(updates == (2 if small else 648) and len(states) == 2, 'training accounting mismatch')
    del optimizer
    model.load_state_dict(swa_average(states), strict=True)
    del states
    dataset = StudyDataset(serving.CORPUS, ids, p['train_labels'], seed, augment=False)
    dataset.set_epoch(3)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False, num_workers=0, collate_fn=collate)
    def bn_loader():
        for batch in loader:
            yield batch
            progress('bn', {})
    seed_all(seed*1000+3, device)
    bn = recompute_bn(model, bn_loader(), device, amp_context(device, 'none' if device == 'cpu' else 'bf16'))
    need(bn['studies'] == len(ids) and bn['unique_studies'] == len(ids), 'BN cohort mismatch')
    save_model(folder/'model_best.pt', dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        parent_sha256=serving.MODEL_SHA, registration_sha256=sha(WORK/'registration.json'),
        seed=seed, arm=arm, swa_epochs_1based=[2,3], fold=0, clean_oof=False, exposure='EXPOSED'))
    progress('final_saved', {})
    del model
    gc.collect()
    if device != 'cpu':
        torch.cuda.empty_cache()
    return dict(arm=arm, seed=seed, studies=len(ids), optimizer_steps=updates,
                epoch_seconds=timings, bn=bn, model_sha256=sha(folder/'model_best.pt'))


def panel(checkpoint, ids, device, progress=lambda *a: None):
    """Release serving uses TF32 on; restore parent training's setting afterward."""
    import torch
    prior = torch.backends.cuda.matmul.allow_tf32
    try:
        if device != 'cpu':
            torch.backends.cuda.matmul.allow_tf32 = True
        return _panel(checkpoint, ids, device, progress)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior


def _panel(checkpoint, ids, device, progress=lambda *a: None):
    import numpy as np
    import torch
    ns, _ = serving.serving(320)
    model = serving.make_model(ns, device)
    if checkpoint != serving.CHECKPOINT:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        need(payload['registration_sha256'] == sha(WORK/'registration.json')
             and payload['parent_sha256'] == serving.MODEL_SHA, 'continuation provenance mismatch')
        model.load_state_dict(payload['model'], strict=True)
    values, times = [], []
    for index, uid in enumerate(ids):
        volume, mask, _ = serving.volume(uid)
        if device != 'cpu':
            torch.cuda.synchronize()
        start = time.perf_counter()
        values.append(serving.infer(ns, model, volume, mask, device))
        if device != 'cpu':
            torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        progress('panel', dict(completed=index+1))
    del model, ns
    gc.collect()
    if device != 'cpu':
        torch.cuda.empty_cache()
    return serving.valid_probs(np.asarray(values),len(ids)), times


def scientific_gate(mean_pairs, seed_pairs):
    pairs = list(mean_pairs.values()) + [p for pair in seed_pairs.values() for p in pair.values()]
    def complete(pair):
        refs = pair.get('references', {})
        return set(refs) == set(serving.RULERS) and all(
            isinstance(r.get('delta'), (int,float)) and math.isfinite(r['delta'])
            and set(r.get('per_target_delta', {})) == set(history.TARGETS)
            and all(isinstance(v,(int,float)) and math.isfinite(v) and v >= -.003
                    for v in r['per_target_delta'].values()) for r in refs.values())
    floors = all(complete(p) for p in pairs)
    roster = set(mean_pairs) == {'ADAMW','PARENT'} and set(seed_pairs) == {'42','43'} and all(
        set(p) == {'ADAMW','PARENT'} for p in seed_pairs.values())
    wins = {name: sum(r['delta'] is not None and r['delta'] >= .002 for r in p['references'].values())
            for name,p in mean_pairs.items()}
    positive = floors and all(sum(r['delta'] for r in p['references'].values())/4 > 0
                              for ps in seed_pairs.values() for p in ps.values())
    joint_wins = sum(all(p.get('references', {}).get(r, {}).get('delta', float('-inf')) is not None
                        and p['references'][r]['delta'] >= .002 for p in mean_pairs.values())
                     for r in serving.RULERS) if roster and floors else 0
    return dict(status='GO' if roster and floors and joint_wins >= 3 and positive else 'DEAD',
                wins=wins, joint_wins=joint_wins, all_target_floors=floors, both_seeds_positive=positive, registration_gate=GATE,
                independent_result_review_required=True)


def comparisons(arms, parent, refs, draws):
    from tools.exp170_convnext_t_l2 import paired_jev_consensus
    from tools import exp115_vlm_eval as evaluator
    pair = lambda a,b: paired_jev_consensus(a,b,refs,draws,evaluator)
    for value in [parent,*arms.values()]:
        serving.valid_probs(value,len(parent))
    sam = (arms['SAM42']+arms['SAM43'])/2
    adam = (arms['ADAMW42']+arms['ADAMW43'])/2
    means = dict(ADAMW=pair(sam,adam), PARENT=pair(sam,parent))
    seeds = {str(s):dict(ADAMW=pair(arms[f'SAM{s}'],arms[f'ADAMW{s}']),
                        PARENT=pair(arms[f'SAM{s}'],parent)) for s in (42,43)}
    return dict(seed_mean=means, seeds=seeds, gate=scientific_gate(means,seeds))


def check_review(review):
    need(review.get('verdict') == 'PASS' and review.get('independent') is True
         and review.get('reviewer') not in (None,'',BUILDER) and review.get('scope') == TAG
         and review.get('registration_sha256') == sha(WORK/'registration.json')
         and review.get('cpu_smoke_sha256') == sha(WORK/'cpu-smoke.json'), 'independent source/smoke-bound PASS required')
    evidence = Path(review.get('evidence_path','')).resolve()
    need(ROOT in evidence.parents and evidence.is_file() and sha(evidence) == review.get('evidence_sha256'),
         'review evidence missing or changed')


def admit(dry_run=False):
    registered()
    smoke = io.read_json(WORK/'cpu-smoke.json')
    need(smoke.get('status') == 'CPU_SMOKE_PASS' and smoke.get('registration_sha256') == sha(WORK/'registration.json')
         and smoke.get('studies') == 2 and smoke.get('steps_per_arm') == 2, 'CPU smoke missing/stale')
    if dry_run and not (WORK/'review.json').exists():
        return dict(status='HOLD_REVIEW_REQUIRED', cuda_started=False)
    check_review(io.read_json(WORK/'review.json'))
    return dict(status='ADMITTED_FOR_B_CALIBRATION', cuda_started=False)


def cpu_smoke():
    import torch
    start = time.perf_counter()
    registered()
    p, ids, refs, draws, _ = authenticate()
    torch.set_num_threads(8)
    records = []
    for arm in POLICY['arms']:
        folder = WORK/'cpu'/arm
        def progress(phase, detail):
            io.receipt(WORK/'cpu-status.json',dict(arm=arm,phase=phase,detail=detail,seconds=time.perf_counter()-start))
        record = train_arm(p, arm, 42, 'cpu', folder, True, progress)
        _, times = panel(folder/'model_best.pt', ids[:2], 'cpu', progress)
        record['serving_seconds'] = times
        records.append(record)
    _, parent_times = panel(serving.CHECKPOINT, ids[:2], 'cpu')
    parent = history.load_oof(serving.CHECKPOINT.parent/history.OOF_PRIMARY,ids)
    replay = comparisons({a+str(s):parent.copy() for a in POLICY['arms'] for s in (42,43)},parent,refs,draws)
    need(replay['gate']['status'] == 'DEAD', 'identical-arm ruler replay failed')
    value = dict(status='CPU_SMOKE_PASS', version=VERSION, registration_sha256=sha(WORK/'registration.json'),
        studies=2, steps_per_arm=2, effective_batch=16, records=records, parent_serving_seconds=parent_times,
        seconds=time.perf_counter()-start, cuda_validated=False, training_precision='fp32 CPU only',
        smoke_samples='two parent training studies repeated eight times per step; two heldout serving studies',
        ruler_replay='888 historical predictions; fixed four rulers/common masks; identical arms DEAD')
    io.atomic_json(WORK/'cpu-smoke.json',value)
    return value


def binding():
    return {name:sha(WORK/name) for name in ('registration.json','cpu-smoke.json','review.json')}


def require_gpu_job():
    import psutil
    ancestors = {p.pid:p for p in psutil.Process().parents()}
    job = io.read_json(ROOT/f'artifacts/logs/{TAG}/job.json')
    need(job.get('tag') == TAG and job.get('pid') in ancestors
         and ancestors[job['pid']].create_time() == job.get('create_time')
         and job.get('phase') in ('spawning','running'), 'required gpu_job tag '+TAG)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'0'), 'GPU zero required')
    serving.no_other_ml(set(ancestors)|{os.getpid()})


def require_controller(phase):
    import psutil
    parent = psutil.Process().parent()
    ticket = json.loads(os.environ.get('HB162_CONTROLLER','{}'))
    need(parent is not None and ticket.get('pid') == parent.pid and ticket.get('created') == parent.create_time()
         and ticket.get('phase') == phase and ticket.get('binding') == binding()
         and parent.cmdline()[1:] == [str(Path(__file__).resolve()),'run'], 'live reviewed controller required')
    if phase == 'full':
        completed('calibration')


def completed(phase, accept_memory_breach=False):
    from tools.c85r7_flipall_timing import memory_admission
    folder = WORK/'gpu'/phase
    receipt = io.read_json(folder/'controller.json')
    status = io.read_json(folder/'status.json')
    breach = (accept_memory_breach and phase == 'full' and receipt.get('exit_code') == 1
              and receipt.get('error') == MEMORY_ERROR
              and status == dict(state='FAILED', error=MEMORY_ERROR))
    normal = (receipt.get('exit_code') == 0 and receipt.get('error') is None
              and status == dict(state='COMPLETE', error=None))
    need(receipt.get('binding') == binding() and (normal or breach)
         and receipt.get('cleanup_errors') == [] and receipt.get('phase') == phase,
         'completed controller receipt missing/invalid')
    if breach:
        completed('calibration')
        need(sha(WORK/'registration.json') == LEGACY_REGISTRATION_SHA
             and receipt.get('output_sha256') is None
             and sha(folder/'output.json') == LEGACY_FULL_OUTPUT_SHA,
             'memory exception requires the pinned historical worker output')
        lines = (folder/'run.log').read_text(encoding='utf-8').splitlines()
        need(bool(lines) and json.loads(lines[-1]) == dict(status='WORKER_COMPLETE'),
             'memory exception requires successful worker termination')
    else:
        need(sha(folder/'output.json') == receipt['output_sha256'], 'worker output changed')
    output = io.read_json(folder/'output.json')
    need(output['binding'] == binding() and output['phase'] == phase, 'output identity mismatch')
    need(phase in ('calibration','full'), 'unknown completed phase')
    seeds = [42] if phase == 'calibration' else POLICY['seeds']
    roster = [(a,s) for s in seeds for a in POLICY['arms']]
    records = output.get('records', [])
    expected = {'parent.json','runtime.json'} | {
        f'{a}{s}/{name}' for a,s in roster for name in ('model_best.pt','predictions.json')}
    need([(r.get('arm'),r.get('seed')) for r in records] == roster
         and len(output['peaks']) == 2*len(roster)+1 and set(output['artifacts']) == expected,
         'completed run roster/peaks/artifacts incomplete')
    studies, steps, epochs = (2,2,2) if phase == 'calibration' else (3455,648,3)
    for r in records:
        need(r.get('studies') == studies and r.get('optimizer_steps') == steps
             and r.get('bn',{}).get('studies') == studies
             and r.get('bn',{}).get('unique_studies') == studies
             and len(r.get('epoch_seconds',[])) == epochs
             and all(math.isfinite(t) and t > 0 for t in r['epoch_seconds'])
             and r.get('model_sha256') == output['artifacts'][f"{r['arm']}{r['seed']}/model_best.pt"],
             'completed training/BN accounting invalid')
    if breach:
        # Re-run all memory checks: accept ONLY the exact aggregate-cap error.
        # In particular a worker-reserved breach or missing measurements fails.
        try:
            memory_admission(receipt['desktop'],output['peaks'],receipt['peak_device_mib']*2**20)
        except ValueError as exc:
            need(f'{type(exc).__name__}: {exc}' == MEMORY_ERROR,
                 'memory exception is not solely an aggregate peak breach')
        else:
            need(False, 'recorded aggregate memory breach is not reproducible')
        # The aggregate check precedes headroom in memory_admission. Independently
        # prove the latter as well; a second failed gate is not this exception.
        from tools.c85r7_flipall_timing import desktop_values
        used, total = desktop_values(receipt['desktop'])
        maximum = max(max(output['peaks']), receipt['peak_device_mib']*2**20-used)
        need(used + maximum + 2*2**30 <= total,
             'memory exception also violates measured desktop headroom')
    else:
        memory_admission(receipt['desktop'],output['peaks'],receipt['peak_device_mib']*2**20)
    for relative,digest in output['artifacts'].items():
        path = (folder/relative).resolve()
        need(folder.resolve() in path.parents and sha(path) == digest, 'completed artifact changed')
    return output


def evaluation_admission(accept_memory_breach=False):
    """Historical compatibility is local to CPU readouts, never GPU admission."""
    if accept_memory_breach:
        from tools.job_runtime import job_status
        state = job_status(ROOT/'artifacts/logs', TAG)['state']
        need(state in ('SUCCESS', 'FAILED'),
             'evaluation requires a stopped terminal job; live/unknown work is refused')
    saved = io.read_json(WORK/'registration.json')
    if accept_memory_breach and saved.get('version') == '1.0.0':
        need(sha(WORK/'registration.json') == LEGACY_REGISTRATION_SHA,
             'historical registration is not the pinned v1.0.0 run')
        current = registration()
        for name, digest in LEGACY_SOURCES.items():
            need(sha(LEGACY_BACKUP/name) == digest and saved['sources'].get(name) == digest,
                 'historical source backup changed: '+name)
            current['sources'][name] = digest
        current['version'] = '1.0.0'
        need(current == saved, 'historical registration dependencies/policy/identity changed')
        smoke = io.read_json(WORK/'cpu-smoke.json')
        need(smoke.get('status') == 'CPU_SMOKE_PASS'
             and smoke.get('registration_sha256') == LEGACY_REGISTRATION_SHA
             and smoke.get('studies') == 2 and smoke.get('steps_per_arm') == 2,
             'historical CPU smoke missing/stale')
        check_review(io.read_json(WORK/'review.json'))
    else:
        admit()


def evaluation_inputs(accept_memory_breach=False):
    """Authenticate every probability before either scientific or lens readout."""
    import numpy as np
    evaluation_admission(accept_memory_breach)
    completed('calibration')
    output = completed('full', accept_memory_breach)
    _, ids, refs, draws, identity = authenticate()
    need(len(ids) == 888 and len(set(ids)) == len(ids), 'evaluation roster incomplete/duplicated')
    folder = WORK/'gpu/full'
    def load(name):
        row = io.read_json(folder/name)
        need(row['ids'] == ids and sha(folder/name) == output['artifacts'][name],
             'prediction UID order/hash mismatch')
        return serving.valid_probs(np.asarray(row['probabilities']), len(ids))
    arms = {a+str(s):load(f'{a}{s}/predictions.json') for a in POLICY['arms'] for s in POLICY['seeds']}
    parent = load('parent.json')
    receipt = io.read_json(folder/'controller.json')
    breach = receipt['exit_code'] != 0
    evidence = dict(memory_gate='BREACHED_NO_OOM' if breach else 'PASS',
        checkpoint_disposition='DO_NOT_SUBMIT' if breach else 'RESULT_REVIEW_REQUIRED',
        submission_authorized=False, accepted_memory_breach=breach,
        full_output_sha256=sha(folder/'output.json'),
        evaluator_sources={name:sha(ROOT/name) for name in LEGACY_SOURCES},
        memory_evidence=dict(error=receipt.get('error'), peak_device_mib=receipt['peak_device_mib'],
            desktop=receipt['desktop'], peak_reserved_bytes=output['peaks'],
            aggregate_peak_mib=max(max(output['peaks'])/2**20,
                                  receipt['peak_device_mib']-receipt['desktop']['used_mib']),
            limit_mib=12288, calibration_controller_sha256=sha(WORK/'gpu/calibration/controller.json'),
            full_controller_sha256=sha(folder/'controller.json'), cleanup_errors=receipt['cleanup_errors']))
    return arms, parent, ids, refs, draws, identity, evidence


def gpu_worker(phase):
    require_controller(phase); admit(); require_gpu_job()
    import torch
    import numpy as np
    import timm
    from tools.c85r7_flipall_timing import memory_admission
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    need(torch.cuda.get_device_name(0) == POLICY['gpu'] and torch.cuda.is_bf16_supported()
         and torch.__version__ == '2.5.1+cu121', 'registered parent GPU/runtime required')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    folder = WORK/'gpu'/phase
    desktop = io.read_json(folder/'desktop.json')
    p, ids, _, _, _ = authenticate()
    small = phase == 'calibration'
    if small:
        ids = ids[:2]
    sequence = 0
    peaks, records, artifacts = [], [], {}
    def progress(kind, detail):
        nonlocal sequence
        sequence += 1
        peak = torch.cuda.max_memory_reserved()
        if peak:
            memory_admission(desktop,[peak])
        io.journal(folder/'progress.json',dict(identity=binding(),sequence=sequence,phase=kind,detail=detail,
                   peak_reserved_bytes=peak),folder)
    io.atomic_json(folder/'runtime.json',dict(torch=torch.__version__,numpy=np.__version__,timm=timm.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
        training_matmul_tf32=False,serving_matmul_tf32=True,cudnn_benchmark=True,
        training_precision='bf16',serving_precision='fp16'))
    for seed in ([42] if small else POLICY['seeds']):
        for arm in POLICY['arms']:
            torch.cuda.reset_peak_memory_stats()
            sub = folder/f'{arm}{seed}'
            record = train_arm(p,arm,seed,'cuda:0',sub,small,progress)
            peaks.append(torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
            values,times = panel(sub/'model_best.pt',ids,'cuda:0',progress)
            peaks.append(torch.cuda.max_memory_reserved())
            io.atomic_json(sub/'predictions.json',dict(ids=ids,probabilities=values.tolist(),seconds=times))
            records.append(record)
            for name in ('model_best.pt','predictions.json'):
                artifacts[f'{arm}{seed}/{name}'] = sha(sub/name)
    torch.cuda.reset_peak_memory_stats()
    values,times = panel(serving.CHECKPOINT,ids,'cuda:0',progress)
    peaks.append(torch.cuda.max_memory_reserved())
    io.atomic_json(folder/'parent.json',dict(ids=ids,probabilities=values.tolist(),seconds=times))
    artifacts['parent.json'] = sha(folder/'parent.json')
    artifacts['runtime.json'] = sha(folder/'runtime.json')
    io.atomic_json(folder/'output.json',dict(binding=binding(),phase=phase,peaks=peaks,records=records,artifacts=artifacts))


def supervise(phase):
    import psutil
    from tools import hb149_monitor_v125 as mon
    from tools.c85r7_flipall_timing import _GuardedTelemetry, desktop_snapshot, memory_admission
    folder = WORK/'gpu'/phase
    need(not folder.exists(), 'retained GPU attempt exists; review it before a versioned restart')
    folder.mkdir(parents=True)
    desktop = desktop_snapshot()
    io.atomic_json(folder/'desktop.json',desktop)
    telemetry = _GuardedTelemetry(os.environ.copy(),.5,desktop)
    status = mon._StatusWriter(folder/'status.json')
    child = job = None
    tracked = {}
    start = last = time.perf_counter()
    sequence = -1
    code, error, peak = 1,None,0
    output_sha = None
    env = os.environ.copy()
    env['HB162_CONTROLLER'] = json.dumps(dict(pid=os.getpid(),created=psutil.Process().create_time(),phase=phase,binding=binding()))
    try:
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            status.start(); telemetry.start()
            child = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_worker','--phase',phase],
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=os.name!='nt',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            process = psutil.Process(child.pid)
            tracked[process.pid] = process
            job = mon._WindowsJob(child)
            while child.poll() is None:
                now = time.perf_counter()
                peak,_ = telemetry.guard_snapshot()
                # Only designated progress/status is read while the worker runs.
                try:
                    progress = io.read_json(folder/'progress.json')
                    if progress['sequence'] != sequence:
                        last = now; sequence = progress['sequence']
                except FileNotFoundError:
                    pass
                sample = telemetry.take()
                if sample is not None:
                    need(sample[1] is None,'GPU telemetry failed')
                    reason = serving.guard_reason(now,last,sample[0],desktop['total_mib'])
                    need(reason is None,reason)
                need(now-last < 900,'900 seconds without progress')
                try:
                    owned = [process]+process.children(recursive=True)
                    for proc in owned:
                        tracked[proc.pid] = proc
                    need(sum(getattr(p.memory_info(),'private',p.memory_info().rss) for p in owned)
                         < mon.RAM_LIMIT_BYTES,'owned RAM limit')
                except psutil.NoSuchProcess:
                    if child.poll() is not None:
                        break
                    raise
                status.submit(dict(state='RUNNING',phase=phase,sequence=sequence,idle_seconds=now-last,peak_device_mib=peak))
                time.sleep(.5)
            code = child.wait()
            need(code == 0,'worker failed; retained log/partial checkpoints')
            peak,_ = telemetry.guard_snapshot()
            output = io.read_json(folder/'output.json')
            memory_admission(desktop,output['peaks'],peak*2**20)
            output_sha = sha(folder/'output.json')
    except BaseException as exc:
        code,error = 1,f'{type(exc).__name__}: {exc}'
    finally:
        cleanup = mon._cleanup(child,tracked,job,psutil)
        if telemetry.thread.ident is not None:
            telemetry.close()
        if status.thread.ident is not None:
            status.submit(dict(state='COMPLETE' if code == 0 else 'FAILED',error=error)); status.close()
        io.receipt(folder/'controller.json',dict(phase=phase,binding=binding(),exit_code=code,error=error,
            cleanup_errors=cleanup,desktop=desktop,peak_device_mib=peak,output_sha256=output_sha,
            elapsed_seconds=time.perf_counter()-start,no_wall_cap=True,idle_only=True))
    need(code == 0 and not cleanup,error or 'cleanup failed')
    completed(phase)


def evaluate(accept_memory_breach=False):
    arms,parent,_,refs,draws,identity,evidence = evaluation_inputs(accept_memory_breach)
    result = comparisons(arms,parent,refs,draws)
    result.update(identity=identity,version=VERSION,binding=binding(),independent_result_review_required=True,
                  **evidence)
    io.atomic_json(WORK/'result.json',result)
    return dict(status='RESULT_REVIEW_REQUIRED',scientific_gate=result['gate']['status'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register','validate','prepare','cpu-smoke','admit','run','evaluate','_worker'])
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--phase',choices=['calibration','full'])
    parser.add_argument('--accept-memory-breach',action='store_true',
                        help='CPU evaluation only: accept the pinned full-run aggregate memory breach')
    args = parser.parse_args(argv)
    if args.accept_memory_breach and args.command != 'evaluate':
        parser.error('--accept-memory-breach is restricted to evaluate')
    WORK.mkdir(parents=True,exist_ok=True)
    if args.command == 'register':
        value = registration(); io.atomic_json(WORK/'registration.json',value)
        result = dict(status='REGISTERED_CPU_ONLY',projected_4070_hours=value['timing_projection']['hours'])
    elif args.command == 'validate':
        registered(); result = dict(status='VALIDATED_CPU_ONLY',train=3455,heldout=888,overlap=0)
    elif args.command == 'prepare':
        reg = registered()
        for uid in reg['train_ids'][:2]+reg['ids'][:2]:
            serving.volume(uid)
        result = dict(status='PREPARED_CPU_ONLY',training_studies=2,heldout_studies=2)
        io.atomic_json(WORK/'prepare.json',result)
    elif args.command == 'cpu-smoke':
        result = cpu_smoke()
    elif args.command == 'admit':
        result = admit(args.dry_run)
    elif args.command == '_worker':
        need(args.phase is not None,'worker phase required'); gpu_worker(args.phase)
        result = dict(status='WORKER_COMPLETE')
    elif args.command == 'evaluate':
        result = evaluate(args.accept_memory_breach)
    else:
        admit(); require_gpu_job(); supervise('calibration'); supervise('full'); result = evaluate()
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
