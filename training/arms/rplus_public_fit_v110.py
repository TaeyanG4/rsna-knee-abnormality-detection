"""Public-R member v1.1.0: reuses tested R+ optimization, SWA and BN primitives.

Changelog: isolated seed44 contract import; training arithmetic unchanged from v1.0.0.
CPU smoke is reduced FP32 engineering evidence, never a production member.
"""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from training.arms import coatnet as c, rplus_public_v110 as contract, lifecycle as life
from training.arms.rplus_data import L2Dataset, mixed
from training.arms.rplus_fit import make_optimization, train_epoch
from training.arms.coatnet_fit import (configure, observation, worker_init, verify_studies,
    save_torch, record_smoke)
from tools.exp112_full_swa import recompute_bn, state_dict_sha256, swa_average
VERSION = '1.1.0'
need, base = life._need, c.base


def load_parent(model, path):
    need(c.sha256(path)==contract.PUBLIC_SHA, 'public parent bytes differ')
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ck=torch.load(path,map_location='cpu',weights_only=True)
    need(ck.get('arch')==base.ARCH and ck.get('res')==384,'public parent metadata differs')
    model.load_state_dict(ck['model'],strict=True)
    return state_dict_sha256(model.state_dict())


def run(spec, stage, corpus_dir, destination, *, parent, mode, device, workers=0, note=None, evidence_root=None, review=None, review_root=None):
    need(mode in ('smoke','full') and device in ('cpu','cuda'), 'invalid execution mode')
    need(mode == 'smoke' or device == 'cuda', 'full CPU training is unsupported')
    need(type(workers) is int and 0 <= workers <= 12, 'workers must be 0..12')
    if device == 'cpu':
        need(not torch.cuda.is_initialized(), 'CPU smoke must not initialize CUDA')
    pre_cuda_hash = None
    verification_hash = None
    if device == 'cuda':
        pre_cuda_hash = contract.require_pre_cuda_review(spec,review,review_root)
    if mode == 'full':
        need(note is not None and evidence_root is not None,'full fit requires independently reviewed evidence')
        verification_hash = life.require_verification_pass(note, **spec.verification_args)
        life.verify_evidence(spec,note,evidence_root)
    base.install_portable_gold_blocker()
    contract.validate_stage(spec, stage)
    p = contract.population(spec, stage)
    source_root = Path(__file__).resolve().parents[2]
    life._verify_files(source_root, spec.document['inputs']['sources'])
    need(c.sha256(Path(corpus_dir)/'manifest.json') == spec.inventory[c.CORPUS+'/manifest.json'], 'mounted corpus differs')
    corpus = base.Corpus(corpus_dir)
    doc, config = spec.document, spec.document['config']
    factory, recipe = c.Config(**config['factory']), config['recipe']
    configure()
    observed = observation(device)
    if device == 'cuda':
        need(torch.cuda.is_bf16_supported(), 'native bf16 required')
        life.check_target_observation(spec, observed)
    train_ids = p['train_ids']
    readout_ids = train_ids[:1]
    epochs, accumulation, swa_epochs = config['epochs'], 16, recipe['swa_1based']
    if mode == 'smoke':
        # CPU is an explicitly reduced FP32 engineering test. GPU uses one full
        # 16-microbatch group plus a tail to exercise registered accumulation.
        train_ids = train_ids[:2 if device=='cpu' else 17]
        epochs, swa_epochs = 1, [1]
    def study_inventory():
        return dict(rsna=verify_studies(corpus, train_ids+readout_ids))
    study_hashes = study_inventory()
    seed = doc['seed']
    base._cpu_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = c.build_model(factory)
    initial_hash = load_parent(model, parent)
    model.backbone.set_grad_checkpointing(config['checkpointing'])
    model.to(device)
    precision = 'none' if device == 'cpu' else config['precision']
    optimizer, scheduler, scaler = make_optimization(model, recipe, config['total_updates'], precision)
    dataset = L2Dataset(corpus_dir,train_ids,p['train_labels'],seed,factory)
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
    inherited_updates = 0
    expected_per_epoch = (len(loader)+15)//16
    started = time.monotonic()
    weights = torch.tensor(p['pos_weight'])
    try:
        for epoch in range(start, epochs):
            base._cpu_seed(seed*1000+epoch)
            if device == 'cuda':
                torch.cuda.manual_seed_all(seed*1000+epoch)
            dataset.set_epoch(16+epoch); sampler.set_epoch(16+epoch)
            record = train_epoch(model, loader, optimizer, scheduler, scaler, weights, recipe, device, precision, accumulation)
            need(record['updates'] == expected_per_epoch, 'epoch updates differ')
            history.append(dict(epoch=epoch+1,data_epoch=16+epoch,sampler_epoch=16+epoch, **record))
            if epoch+1 in swa_epochs:
                name = f'swa-{epoch+1:02d}.pt'
                save_torch(work/name, dict(model=model.state_dict(), epoch=epoch+1))
                swa_members[name] = c.sha256(work/name)

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
    bn_ds = L2Dataset(corpus_dir,train_ids,p['train_labels'],seed,factory,augment=False)
    bn_ds.set_epoch(config['bn']['epoch'])
    bn_loader = DataLoader(bn_ds, batch_size=2, shuffle=False, drop_last=False, num_workers=workers,
        worker_init_fn=worker_init, collate_fn=mixed.collate, pin_memory=device=='cuda')
    bn = recompute_bn(model, mixed.bn_batches(bn_loader), device, base.amp_context(device, precision))
    need(bn['studies']==len(train_ids) and bn['unique_studies']==len(train_ids), 'BN population differs')
    provenance = dict(tool_version=VERSION, spec_sha256=spec.sha256, verification_sha256=verification_hash,
        scientific=config, scientific_scope=contract.SCOPE, pre_cuda_review_sha256=pre_cuda_hash,
        parent_sha256=contract.PUBLIC_SHA, initial_state_sha256=initial_hash,
        full_roster_class_weights=True, added_oai=False, honest_oof=False, oai_panel_eligible=False,
        population_sha256=doc['inputs']['rosters'][doc['population_source']],
        source_bindings=doc['inputs']['sources'], decision_bindings=doc['inputs']['evidence'],
        environment={k:v for k,v in observed['runtime'].items() if k not in ('python','driver')},
        identity=identity, study_hashes=study_hashes, swa_members=swa_members,
        ordered_swa_members=[f'swa-{e:02d}.pt' for e in swa_epochs],bn_receipt=bn,
        bn_epoch=19,bn_train_sha256=contract.uid_hash(train_ids),readout=config['readout'])
    final_hash = state_dict_sha256(model.state_dict())
    payload = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()}, arch=base.ARCH,
        res=384, inference_res=320, lab=config['targets'], contract=base.CONTRACT, k_eval=94, norm='imagenet',
        window_policy='legacy', epoch=epochs-1, tag=doc['arm'], amp_train=precision, amp_eval='fp16' if device=='cuda' else 'none',
        fold=doc['fold'], label_set=recipe['labels'], provenance=provenance, seed=seed,
        checkpoint_kind='fixed-swa-bn', swa_epochs_1based=swa_epochs, oai=False, public_parent_sha256=contract.PUBLIC_SHA,
        ancestry_scope=contract.SCOPE,
        family_config=config['factory'], scientific_scope=contract.SCOPE if mode=='full' else 'ENGINEERING_SMOKE_ONLY')
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
        version=VERSION, mode=mode, rsna_studies=len(train_ids), scientific_scope=contract.SCOPE if mode=='full' else 'ENGINEERING_SMOKE_ONLY',
        ancestry_scope=contract.SCOPE, honest_oof=False, oai_panel_eligible=False, oai_studies_added=0, updates=updates, skips=0, observation=observed,
        environment=dict(observed['runtime'], hostname=observed['hostname']), bn=bn,
        readout=config['readout'], seconds=time.monotonic()-started, final_state_sha256=final_hash,
        results=dict(train_batches=sum(h['batches'] for h in history[start:]), validation_batches=len(readout_ids),
            bn_batches=bn['batches'], optimizer_steps=updates-inherited_updates, checkpoint_reloaded=True,
            finite_outputs=True, real_data=True), cpu_cuda_initialized=False if device=='cpu' else None)
    contract.write(out/'history.json', history)
    contract.write(out/'provenance.json', provenance)
    summary['provenance_sha256'] = life.file_sha256(out/'provenance.json')
    contract.write(out/'summary.json', summary)
    return summary

