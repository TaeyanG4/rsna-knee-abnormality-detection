"""R+ fit v1.0.0: matched constant-LR continuations from CONTROL43 final SWA/BN.

Changelog: isolated continuation fitter; fresh AdamW, three epochs, ordered SWA/BN.
"""
from pathlib import Path
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from training.arms import coatnet as c, rplus as contract, lifecycle as life
from training.arms.rplus_data import Dataset, mixed
from training.arms.coatnet_fit import (configure, observation, worker_init, verify_studies,
    save_torch, save_epoch, restore_epoch, record_smoke)
from training.src.exp143_v2 import accumulation_group
from tools.exp112_full_swa import recompute_bn, state_dict_sha256, swa_average
VERSION = '1.0.0'
need, base = life._need, c.base


def make_optimization(model, recipe, total_steps, precision):
    """Same AdamW groups/defaults/decay as P; fresh state, fixed low rates."""
    optimizer = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=recipe['bb_lr']),
        dict(params=[v for k,v in model.named_parameters() if not k.startswith('backbone.')], lr=recipe['head_lr'])
    ], weight_decay=recipe['wd'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _:1.)
    scaler = torch.amp.GradScaler('cuda', enabled=precision=='fp16', init_scale=1024.)
    return optimizer,scheduler,scaler


def load_parent(model, spec, stage):
    config = spec.document['config']
    path = Path(stage)/config['continuation']['checkpoint']
    need(c.sha256(path)==config['continuation']['checkpoint_sha256'], 'parent checkpoint bytes differ')
    ck = torch.load(path,map_location='cpu',weights_only=True)
    need(ck['fold']==0 and ck['seed']==43 and ck['checkpoint_kind']=='fixed-swa-bn', 'parent metadata differs')
    need(ck['lab']==config['targets'] and ck['family_config']==config['factory'], 'parent model metadata differs')
    model.load_state_dict(ck['model'],strict=True)
    need(state_dict_sha256(model.state_dict())==config['initial_state_sha256'], 'parent state differs')


def train_epoch(model, loader, optimizer, scheduler, scaler, weights, recipe, device, precision, accumulation=16):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total, updates, batches = 0., 0, 0
    for index, batch in enumerate(loader):
        x, y, mask, _ = mixed.unpack_batch(batch)
        final, count = accumulation_group(index, len(loader), accumulation)
        with base.amp_context(device, precision):
            logits = model(x.to(device)).float()
            loss = mixed.masked_study_bce(logits, y.to(device), mask.to(device), weights)
            need(bool(torch.isfinite(loss)), 'nonfinite loss')
        scaler.scale(loss / count).backward()
        total += float(loss.detach())
        batches += 1
        if final:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), recipe['clip'], error_if_nonfinite=True)
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            need(scaler.get_scale() >= previous, 'optimizer step skipped')
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
    need(batches == len(loader) and batches > 0, 'incomplete training epoch')
    return dict(loss=total/batches, updates=updates, batches=batches, skips=0)



def run(spec, stage, corpus_dir, destination, *, oai_corpus, mode, device, workers=0, resume=None, note=None, evidence_root=None):
    need(mode in ('smoke','full') and device in ('cpu','cuda'), 'invalid execution mode')
    need(mode == 'smoke' or device == 'cuda', 'full CPU training is unsupported')
    need(type(workers) is int and 0 <= workers <= 12, 'workers must be 0..12')
    if device == 'cpu':
        need(not torch.cuda.is_initialized(), 'CPU smoke must not initialize CUDA')
    base.install_portable_gold_blocker()
    contract.validate_stage(spec, stage)
    p = contract.population(spec, stage)
    source_root = Path(__file__).resolve().parents[2]
    life._verify_files(source_root, spec.document['inputs']['sources'])
    need(c.sha256(Path(corpus_dir)/'manifest.json') == spec.inventory[c.CORPUS+'/manifest.json'], 'mounted corpus differs')
    need(c.sha256(Path(oai_corpus)/'manifest.json') == spec.inventory[contract.OAI+'/manifest.json'], 'mounted OAI corpus differs')
    corpus, oai = base.Corpus(corpus_dir), base.Corpus(oai_corpus)
    doc, config = spec.document, spec.document['config']
    factory, recipe = c.Config(**config['factory']), config['recipe']
    configure()
    observed = observation(device)
    verification_hash = None
    if device == 'cuda':
        need(torch.cuda.is_bf16_supported(), 'native bf16 required')
        life.check_target_observation(spec, observed)
    if mode == 'full':
        need(note is not None and evidence_root is not None, 'full fit requires independently reviewed evidence')
        verification_hash = life.require_verification_pass(note, **spec.verification_args)
        life.verify_evidence(spec, note, evidence_root)
    train_ids = p['train_ids']
    readout_ids = p['val_ids'] if doc['fold'] == 0 else train_ids[:1]
    epochs, accumulation, swa_epochs = config['epochs'], 16, recipe['swa_1based']
    if mode == 'smoke':
        # CPU is an explicitly reduced FP32 engineering test. GPU uses one full
        # 16-microbatch group plus a tail to exercise registered accumulation.
        r, o = ((1,1) if device=='cpu' else (12,5)) if p['oai_train_ids'] else ((2,0) if device=='cpu' else (17,0))
        train_ids = p['rsna_train_ids'][:r] + p['oai_train_ids'][:o]
        readout_ids = readout_ids[:1]
        epochs, swa_epochs = 1, [1]
    selected = copy.deepcopy(p)
    selected.update(train_ids=train_ids,
        rsna_train_ids=[u for u in train_ids if u in set(p['rsna_train_ids'])],
        oai_train_ids=[u for u in train_ids if u in set(p['oai_train_ids'])],
        train_labels={u:p['train_labels'][u] for u in train_ids},
        train_masks={u:p['train_masks'][u] for u in train_ids})
    def study_inventory():
        return dict(rsna=verify_studies(corpus, selected['rsna_train_ids']+readout_ids),
                    oai=verify_studies(oai, selected['oai_train_ids']) if selected['oai_train_ids'] else {})
    study_hashes = study_inventory()
    seed = doc['seed']
    base._cpu_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = c.build_model(factory)
    load_parent(model, spec, stage)
    need(state_dict_sha256(model.state_dict()) == config['initial_state_sha256'], 'initial state differs')
    model.backbone.set_grad_checkpointing(config['checkpointing'])
    model.to(device)
    precision = 'none' if device == 'cpu' else config['precision']
    optimizer, scheduler, scaler = make_optimization(model, recipe, config['total_updates'], precision)
    dataset = Dataset(corpus_dir, oai_corpus, selected, seed, smoke=mode=='smoke')
    sampler = DistributedSampler(dataset, 1, 0, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, drop_last=True, num_workers=workers,
        persistent_workers=workers > 0, worker_init_fn=worker_init, collate_fn=mixed.collate, pin_memory=device=='cuda')
    identity = dict(spec_sha256=spec.sha256, observation=observed, mode=mode, device=device,
        workers=workers, epochs=epochs, accumulation=accumulation, swa_epochs=swa_epochs,
        train_sha256=contract.uid_hash(train_ids), readout_sha256=contract.uid_hash(readout_ids))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    out, work = destination/'outputs', destination/'epochs'
    out.mkdir(); work.mkdir()
    start, history, swa_members = 0, [], {}
    if resume:
        start, history, swa_members = restore_epoch(resume, model, optimizer, scheduler, scaler, identity, work)
        need(0 <= start <= epochs, 'resume epoch out of range')
    inherited_updates = sum(h['updates'] for h in history)
    expected_per_epoch = (len(loader)+15)//16
    need(all(h['updates'] == expected_per_epoch and h['skips'] == 0 for h in history), 'resume updates differ')
    started = time.monotonic()
    weights = torch.tensor(p['pos_weight'])
    try:
        # Persistent worker startup consumes a main-process base_seed only once.
        # Recreate it before the epoch reseed on an epoch-boundary resume.
        if 0 < start < epochs and workers:
            iter(loader)
        for epoch in range(start, epochs):
            base._cpu_seed(seed*1000+epoch)
            if device == 'cuda':
                torch.cuda.manual_seed_all(seed*1000+epoch)
            dataset.set_epoch(16+epoch); sampler.set_epoch(16+epoch)
            record = train_epoch(model, loader, optimizer, scheduler, scaler, weights, recipe, device, precision, accumulation)
            need(record['updates'] == expected_per_epoch, 'epoch updates differ')
            history.append(dict(epoch=epoch+1, **record))
            if epoch+1 in swa_epochs:
                name = f'swa-{epoch+1:02d}.pt'
                save_torch(work/name, dict(model=model.state_dict(), epoch=epoch+1))
                swa_members[name] = c.sha256(work/name)
            save_epoch(work/f'epoch-{epoch+1:02d}.pt', model, optimizer, scheduler, scaler, identity, epoch+1, history, swa_members)
    finally:
        if getattr(loader, '_iterator', None) is not None:
            loader._iterator._shutdown_workers()
    updates = sum(h['updates'] for h in history)
    need(updates == epochs*expected_per_epoch and (mode=='smoke' or updates==config['total_updates']), 'total updates differ')
    del optimizer, scheduler, scaler
    states = [torch.load(work/f'swa-{e:02d}.pt', map_location='cpu', weights_only=True)['model'] for e in swa_epochs]
    model.load_state_dict(swa_average(states), strict=True)
    del states
    model.backbone.set_grad_checkpointing(False)
    bn_ds = Dataset(corpus_dir, oai_corpus, selected, seed, augment=False, smoke=mode=='smoke')
    bn_ds.set_epoch(config['bn']['epoch'])
    bn_loader = DataLoader(bn_ds, batch_size=2, shuffle=False, drop_last=False, num_workers=workers,
        worker_init_fn=worker_init, collate_fn=mixed.collate, pin_memory=device=='cuda')
    bn = recompute_bn(model, mixed.bn_batches(bn_loader), device, base.amp_context(device, precision))
    need(bn['studies']==len(train_ids) and bn['unique_studies']==len(train_ids), 'BN population differs')
    provenance = dict(tool_version=VERSION, spec_sha256=spec.sha256, verification_sha256=verification_hash,
        scientific=config, population_sha256=doc['inputs']['rosters'][doc['population_source']],
        environment={k:v for k,v in observed['runtime'].items() if k not in ('python','driver')},
        identity=identity, study_hashes=study_hashes, swa_members=swa_members, readout=config['readout'])
    final_hash = state_dict_sha256(model.state_dict())
    payload = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()}, arch=base.ARCH,
        res=384, inference_res=320, lab=config['targets'], contract=base.CONTRACT, k_eval=94, norm='imagenet',
        window_policy='legacy', epoch=epochs-1, tag=doc['arm'], amp_train=precision, amp_eval='fp16' if device=='cuda' else 'none',
        fold=doc['fold'], label_set=recipe['labels'], provenance=provenance, seed=seed,
        checkpoint_kind='fixed-swa-bn', swa_epochs_1based=swa_epochs, oai=bool(p['oai_train_ids']), oai_exclusion=p['oai_exclusion'],
        family_config=config['factory'], scientific_scope='FOLD_CLEAN_CONTINUATION' if mode=='full' else 'ENGINEERING_SMOKE_ONLY')
    save_torch(out/'model_best.pt', payload)
    del model, payload
    model = c.build_model(factory, 320).to(device).eval()
    checkpoint = torch.load(out/'model_best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    need(state_dict_sha256(model.state_dict()) == final_hash, 'checkpoint reload differs')
    predictions = [c.infer_pair(model, factory, *corpus.get(uid), device=device,
        precision='none' if device=='cpu' else config['eval_precision'], chunk=2 if device=='cpu' else 16) for uid in readout_ids]
    plain, mirror = np.stack([p[0] for p in predictions]), np.stack([p[1] for p in predictions])
    for name, values in (('plain',plain), ('mirror',mirror), ('mean',(plain+mirror)*np.float32(.5))):
        path = out/f'predictions-{name}.csv'
        life.create_new(path, life.prediction_csv_bytes(readout_ids, values, config['targets']))
        life.load_prediction_csv(path, readout_ids, config['targets'])
    life._verify_files(source_root, doc['inputs']['sources'])
    contract.validate_stage(spec, stage)
    need(study_inventory()==study_hashes, 'study bytes changed during execution')
    if device=='cpu':
        need(not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    summary = dict(status='COMPLETE_TRAINING_AND_READOUT' if mode=='full' else 'ENGINEERING_SMOKE_COMPLETE',
        version=VERSION, mode=mode, rsna_studies=len(selected['rsna_train_ids']),
        oai_studies=len(selected['oai_train_ids']), panel_overlap=0, updates=updates, skips=0, observation=observed,
        environment=dict(observed['runtime'], hostname=observed['hostname']), bn=bn,
        readout=config['readout'], seconds=time.monotonic()-started, final_state_sha256=final_hash,
        results=dict(train_batches=sum(h['batches'] for h in history[start:]), validation_batches=len(readout_ids),
            bn_batches=bn['batches'], optimizer_steps=updates-inherited_updates, checkpoint_reloaded=True,
            finite_outputs=True, real_data=True), cpu_cuda_initialized=False if device=='cpu' else None)
    contract.write(out/'history.json', history)
    contract.write(out/'provenance.json', provenance)
    contract.write(out/'summary.json', summary)
    return summary

