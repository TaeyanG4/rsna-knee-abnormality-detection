"""Direct EXP155/OAI qualification v1.1.0; engineering evidence, never a fit.

Changelog: bind/recheck panel exclusions in train, BN and smoke receipts.
Uses the production dataset factory, masked epoch and BN path on real data.
CPU uses two studies and FP32; CUDA uses 17 studies and registered AMP.
Elapsed time is measurement only. Outputs and receipts are create-only.
"""
from pathlib import Path
import math
import time


def run(args):
    from tools import exp190_coatnet_oai_f0 as arm
    import exp155_own0949_oai_v11 as fit
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from training.arms import coatnet_fit as helpers
    from training.arms import lifecycle as life
    from tools.exp112_full_swa import recompute_bn, state_dict_sha256, swa_average

    need = life._need
    out = Path(args.out).resolve()
    need(arm.ROOT.joinpath('.local').resolve() in out.parents, 'smoke output must be below .local')
    need(not out.exists(), 'smoke output exists')
    need(args.device in ('cpu', 'cuda') and 0 <= args.workers <= 12, 'invalid device/workers')
    if args.device == 'cpu':
        need(not torch.cuda.is_initialized(), 'CPU smoke must not initialize CUDA')
    p = arm.load_registration(arm.MANIFEST, args.corpus, args.init_weights, args.oai_cache)
    arm.require_validation()
    arm.trainer_contract(fit)
    bindings = arm.source_bindings()
    helpers.configure()
    observed = helpers.observation(args.device)
    if args.device == 'cuda':
        arm.bridge.require_local_device()
        need(torch.cuda.is_bf16_supported(), 'native bf16 required')
    fit.install_portable_gold_blocker()
    args.oai_corpus = args.oai_cache
    train = arm.dataset_factory(args, p, 42)
    bn_data = arm.dataset_factory(args, p, 42, augment=False)
    train.set_epoch(0)
    bn_data.set_epoch(16)
    selection = train.selection(0)
    # Include both domains, then distinct samples in their registered order.
    r = next(i for i, j in enumerate(selection) if j in train.rsna)
    o = next(i for i, j in enumerate(selection) if j in train.oai)
    count = 2 if args.device == 'cpu' else 17
    indices = [r, o] + [i for i in range(len(train)) if i not in (r, o)][:count-2]
    bn_indices = [selection[i] for i in indices]
    train_ids = [train.ids[i] for i in bn_indices]
    val_ids = p['val_ids'][:1]
    rsna = fit.Corpus(args.corpus)
    oai = fit.Corpus(args.oai_cache)
    checked = dict(rsna=helpers.verify_studies(rsna, [u for u in train_ids if u in train.rsna_ids]+val_ids),
                   oai=helpers.verify_studies(oai, [u for u in train_ids if u in train.oai_ids]))
    started = time.monotonic()
    out.mkdir(parents=True)
    fit._cpu_seed(42)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(42)
    model = fit.build_model()
    fit.load_imagenet(model, args.init_weights)
    need(state_dict_sha256(model.state_dict()) == arm.bridge.INITIAL_STATE_SHA, 'initial state differs')
    model.backbone.set_grad_checkpointing(False)
    model.to(args.device)
    precision = 'none' if args.device == 'cpu' else 'bf16'
    optimizer, scheduler, scaler = fit.make_optimization(model, p['recipe'], 3456, precision)
    loader = DataLoader(train, batch_size=1, sampler=indices, num_workers=args.workers,
                        collate_fn=fit.collate, worker_init_fn=fit.worker_init)
    fit._cpu_seed(42000)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(42000)
    result = fit.train_epoch(model, loader, optimizer, scheduler, scaler,
        torch.tensor(p['pos_weight'], device=args.device), p['recipe'], args.device, precision, 16,
        total_steps=3456)
    need(result['updates'] == math.ceil(count/16) and result['skips'] == 0, 'optimizer schedule differs')
    # Same epoch state shape and optimizer/scheduler/scaler restoration as production.
    epoch = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()}, epoch=1,
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
    helpers.save_torch(out/'epoch.pt', epoch)
    restored = torch.load(out/'epoch.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(restored['model'], strict=True)
    optimizer.load_state_dict(restored['optimizer'])
    scheduler.load_state_dict(restored['scheduler'])
    scaler.load_state_dict(restored['scaler'])
    need(scheduler.last_epoch == result['updates'], 'scheduler restore differs')
    model.load_state_dict(swa_average([restored['model']]), strict=True)
    del epoch, restored, optimizer, scheduler, scaler
    bn_loader = DataLoader(bn_data, batch_size=2, sampler=bn_indices, num_workers=args.workers,
                          collate_fn=fit.collate, worker_init_fn=fit.worker_init)
    bn = recompute_bn(model, fit.bn_batches(bn_loader), args.device, fit.amp_context(args.device, precision))
    need(bn['studies'] == count and bn['unique_studies'] == count, 'BN population differs')
    final = state_dict_sha256(model.state_dict())
    payload = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        arch=fit.ARCH, contract=fit.CONTRACT, oai=True, lab=arm.bridge.TARGETS,
        checkpoint_kind='engineering-smoke-only', res=384, inference_res=320, k_eval=94,
        scientific_scope='ENGINEERING_ONLY', protocol_sha256=arm.HASH(arm.MANIFEST))
    helpers.save_torch(out/'model.pt', payload)
    del model, payload
    model = fit.build_model(320).eval().to(args.device)
    ck = torch.load(out/'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(ck['model'], strict=True)
    del ck
    need(state_dict_sha256(model.state_dict()) == final, 'strict reload differs')
    aa, bb = arm.infer_readout(model, rsna, val_ids, args.device,
        'none' if args.device=='cpu' else 'fp16', 2 if args.device=='cpu' else 16)
    a, b = aa[0], bb[0]
    for name, values in (('plain', a), ('mirror', b), ('mean', (a+b)*np.float32(.5))):
        path = out/f'{name}.csv'
        arm.bridge.base.write_predictions(path, val_ids, values[None])
        loaded = arm.bridge.load_oof(path, val_ids)
        need(np.array_equal(loaded.astype(np.float32), values[None]), 'CSV boundary differs')
    arm.oai_exclusions.validate_training(p)
    need(bindings == arm.source_bindings(), 'sources changed during smoke')
    need(checked == dict(rsna=helpers.verify_studies(rsna, [u for u in train_ids if u in train.rsna_ids]+val_ids),
                        oai=helpers.verify_studies(oai, [u for u in train_ids if u in train.oai_ids])), 'data changed')
    if args.device == 'cpu':
        need(not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    summary = dict(status='ENGINEERING_SMOKE_COMPLETE', version='1.1.0', launch_authorized=False,
        verdict='HOLD', blockers=['Independent review and authentic corresponding-control runtime evidence required',
        'CPU and actual 4070 evidence required together'], device=args.device, precision=precision,
        observation=observed, protocol_sha256=arm.HASH(arm.MANIFEST), sources=bindings,
        oai_exclusion={k:v for k,v in p['oai_exclusion'].items() if k != 'participants'},
        sampling=p['sampling'], full_bn_count=len(bn_data), bn=bn, optimizer=result,
        results=dict(real_data=True, train_batches=count, validation_batches=1, bn_batches=bn['batches'],
                     optimizer_steps=result['updates'], checkpoint_reloaded=True, finite_outputs=True),
        state_sha256=final, study_hashes=checked, seconds=time.monotonic()-started,
        cuda_initialized=torch.cuda.is_initialized(),
        artifacts={name:arm.HASH(out/name) for name in ('epoch.pt','model.pt','plain.csv','mirror.csv','mean.csv')})
    life.create_new(out/'receipt.json', life.canonical_bytes(summary))
    return summary
