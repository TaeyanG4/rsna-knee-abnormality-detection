"""CoAtNet fit v1.0.0: explicit accumulation, immutable epochs, ordered SWA/BN.

Changelog: single-device production/member lifecycle alongside frozen originals.
Elapsed time is measurement only. Smoke outputs are never scientific results.
"""
from pathlib import Path
import platform
import socket
import subprocess
import time
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from training.arms import coatnet as c
from training.arms import coatnet_lifecycle as contract
from training.arms import lifecycle as life
from training.src.exp143_v2 import make_optimization, accumulation_group
from tools.exp112_full_swa import recompute_bn, state_dict_sha256, swa_average

VERSION = '1.0.0'
need = life._need
base = c.base


def configure():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision('highest')


def observation(device):
    import cv2
    import pydicom
    import timm
    need(device in ('cpu', 'cuda'), 'unknown device')
    runtime = dict(python=platform.python_version(), driver='none', torch=str(torch.__version__),
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu='cpu', timm=timm.__version__,
        numpy=np.__version__, pydicom=pydicom.__version__, cv2=cv2.__version__)
    if device == 'cuda':
        runtime['gpu'] = torch.cuda.get_device_name(0)
        runtime['driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
            '--format=csv,noheader'], text=True).strip().splitlines()[0]
    return dict(hostname=socket.gethostname(), runtime=runtime, backend_policy=dict(
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision()))


def worker_init(_):
    base.install_portable_gold_blocker()
    torch.set_num_threads(1)


def verify_studies(corpus, ids):
    records = {r['uid']: r for r in corpus.manifest['files']}
    checked = {}
    for uid in dict.fromkeys(ids):
        row = records[uid]
        need(row['row'] == corpus.rows[uid], 'corpus row differs')
        for ext, key in (('npy', 'sha256'), ('json', 'receipt_sha256')):
            name = f"study_{row['row']:05d}.{ext}"
            path = corpus.directory / name
            need(not path.is_symlink() and c.sha256(path) == row[key], 'study bytes differ')
            checked[name] = row[key]
    need(bool(checked), 'empty study verification')
    return checked


def train_epoch(model, loader, optimizer, scheduler, scaler, weights, recipe, device, precision, accumulation=16):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    total, updates, batches = 0., 0, 0
    for index, batch in enumerate(loader):
        x, y, _ = base.unpack_batch(batch)
        final, count = accumulation_group(index, len(loader), accumulation)
        with base.amp_context(device, precision):
            logits = model(x.to(device)).float()
            loss = loss_fn(logits, y.to(device))
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


def save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)


def save_epoch(path, model, optimizer, scheduler, scaler, identity, epoch, history, swa_members):
    """Receipt binds the epoch plus retained, ordered SWA source checkpoints."""
    path = Path(path)
    save_torch(path, dict(schema='coatnet-epoch/v1', identity=identity, epoch=epoch,
        model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(), history=history, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if identity['device']=='cuda' else None))
    contract.write(path.with_suffix('.receipt.json'), dict(schema='coatnet-epoch-receipt/v1',
        sha256=c.sha256(path), identity=identity, epoch=epoch, swa_members=swa_members))


def restore_epoch(path, model, optimizer, scheduler, scaler, identity, destination):
    path, destination = Path(path), Path(destination)
    receipt = contract.read(path.with_suffix('.receipt.json'))
    need(receipt['schema'] == 'coatnet-epoch-receipt/v1' and receipt['identity'] == identity
         and receipt['sha256'] == c.sha256(path), 'resume receipt differs')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    need(ck['schema'] == 'coatnet-epoch/v1' and ck['identity'] == identity
         and ck['epoch'] == receipt['epoch'], 'resume identity differs')
    expected = {f'swa-{e:02d}.pt' for e in identity['swa_epochs'] if e <= ck['epoch']}
    need(set(receipt['swa_members']) == expected, 'resume SWA membership differs')
    life._verify_files(path.parent, receipt['swa_members'])
    for name in receipt['swa_members']:
        with (destination / name).open('xb') as stream:
            with (path.parent / name).open('rb') as source:
                shutil.copyfileobj(source, stream)
    model.load_state_dict(ck['model'], strict=True)
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    scaler.load_state_dict(ck['scaler'])
    torch.set_rng_state(ck['torch_rng'])
    if identity['device']=='cuda':
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
    need(len(ck['history']) == ck['epoch'], 'resume history differs')
    return ck['epoch'], ck['history'], receipt['swa_members']


def run(spec, stage, corpus_dir, destination, *, mode, device, workers=0, resume=None, note=None, evidence_root=None):
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
    corpus = base.Corpus(corpus_dir)
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
    epochs, accumulation, swa_epochs = 16, 16, recipe['swa_1based']
    if mode == 'smoke':
        # CPU is an explicitly reduced FP32 engineering test. GPU uses one full
        # 16-microbatch group plus a tail to exercise registered accumulation.
        train_ids, readout_ids = train_ids[:2 if device == 'cpu' else 17], readout_ids[:1]
        epochs, swa_epochs = 1, [1]
    study_hashes = verify_studies(corpus, train_ids + readout_ids)
    seed = doc['seed']
    base._cpu_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = c.build_model(factory)
    base.load_imagenet(model, Path(stage)/c.INIT)
    need(state_dict_sha256(model.state_dict()) == config['initial_state_sha256'], 'initial state differs')
    model.backbone.set_grad_checkpointing(config['checkpointing'])
    model.to(device)
    precision = 'none' if device == 'cpu' else config['precision']
    optimizer, scheduler, scaler = make_optimization(model, recipe, config['total_updates'], precision)
    dataset = c.StudyDataset(corpus_dir, train_ids, p['train_labels'], seed, factory,
        metadata=Path(stage)/c.METADATA if factory.sampler == 'coverage' else None)
    sampler = DistributedSampler(dataset, 1, 0, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, drop_last=True, num_workers=workers,
        persistent_workers=workers > 0, worker_init_fn=worker_init, collate_fn=base.collate, pin_memory=device=='cuda')
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
            dataset.set_epoch(epoch); sampler.set_epoch(epoch)
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
    bn_ds = c.StudyDataset(corpus_dir, train_ids, p['train_labels'], seed, factory, augment=False,
        metadata=Path(stage)/c.METADATA if factory.sampler == 'coverage' else None)
    bn_ds.set_epoch(16)
    bn_loader = DataLoader(bn_ds, batch_size=2, shuffle=False, drop_last=False, num_workers=workers,
        worker_init_fn=worker_init, collate_fn=base.collate, pin_memory=device=='cuda')
    bn = recompute_bn(model, bn_loader, device, base.amp_context(device, precision))
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
        checkpoint_kind='fixed-swa-bn', swa_epochs_1based=swa_epochs, oai=False,
        family_config=config['factory'], scientific_scope='PRODUCTION' if mode=='full' else 'ENGINEERING_SMOKE_ONLY')
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
    need(verify_studies(corpus, train_ids+readout_ids)==study_hashes, 'study bytes changed during execution')
    if device=='cpu':
        need(not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    summary = dict(status='COMPLETE_TRAINING_AND_READOUT' if mode=='full' else 'ENGINEERING_SMOKE_COMPLETE',
        version=VERSION, mode=mode, updates=updates, skips=0, observation=observed,
        environment=dict(observed['runtime'], hostname=observed['hostname']), bn=bn,
        readout=config['readout'], seconds=time.monotonic()-started, final_state_sha256=final_hash,
        results=dict(train_batches=sum(h['batches'] for h in history[start:]), validation_batches=len(readout_ids),
            bn_batches=bn['batches'], optimizer_steps=updates-inherited_updates, checkpoint_reloaded=True,
            finite_outputs=True, real_data=True), cpu_cuda_initialized=False if device=='cpu' else None)
    contract.write(out/'history.json', history)
    contract.write(out/'provenance.json', provenance)
    contract.write(out/'summary.json', summary)
    return summary


def record_smoke(spec, destination, summary, command):
    """Only an independent reviewer can convert assembled evidence to PASS."""
    destination = Path(destination)
    observed = summary['observation']
    kind = 'cpu' if summary['cpu_cuda_initialized'] is not None else 'target'
    smoke = dict(hostname=observed['hostname'], device='cpu' if kind=='cpu' else 'cuda',
        environment_sha256=life.content_sha256(observed) if kind=='cpu' else spec.bindings['environment'],
        bindings_sha256=life.content_sha256(spec.bindings), results=summary['results'])
    roles = {'checkpoint':'model_best.pt', 'readout':'predictions-mean.csv', 'summary':'summary.json', 'provenance':'provenance.json'}
    artifacts = {role:dict(path='outputs/'+name, sha256=c.sha256(destination/'outputs'/name)) for role,name in roles.items()}
    check = dict(schema=life.EVIDENCE_SCHEMA, command=command, exit_code=0, artifacts=artifacts, smoke=None)
    check_hash = life.content_sha256(check)
    contract.write(destination/'evidence'/f'{check_hash}.json', check)
    log = dict(check, smoke=dict(smoke, observation=observed))
    log_hash = life.content_sha256(log)
    contract.write(destination/'evidence'/f'{log_hash}.json', log)
    note = dict(schema=life.VERIFICATION_SCHEMA, scope=spec.scope,
        reviewer=dict(name='executed-coatnet-smoke-producer', host='Codex'), verdict='HOLD',
        blockers=['Independent review and CPU plus actual-target evidence required'],
        notes=['Engineering smoke; CPU uses FP32 and a reduced population, target uses registered AMP and accumulation'],
        bindings=spec.bindings, checks=[dict(name='executed-family-smoke', command=command, exit_code=0, log_sha256=check_hash)],
        smokes=dict(cpu=None,target=None))
    note['smokes'][kind] = dict(smoke, command=command, exit_code=0, log_sha256=log_hash)
    life.verify_evidence(spec, note, destination)
    contract.write(destination/'verification.json', note)
    return note
