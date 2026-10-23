"""vNext-R7-PL 1.1.1 production adapters. No implicit CUDA or network access.

Changelog 1.1.1: shared authentic H-B-146 readout admission/export adapter.

Training accepts only materialized training targets, a pinned production member
and an explicit H-B-146 GO. Smoke receipts are never production receipts.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import vnext_r7_pl_v1 as p
from tools import vnext_r7_pl_go_v111 as scientific_go

VERSION = '1.1.1'
CONTROL = 'training/configs/hb122-control-s42-f0-v1.json'
COAT_INIT = 'storage/weights/exp143-imagenet-init/coatnet384-imagenet.pt'
SOURCES = [__file__, ROOT/'training/arms/vnext_r7_pl_coat_v110.py',
           ROOT/'tools/hb146_screen_v111.py', ROOT/'training/arms/hb146_pl_v1.py',
           ROOT/'training/arms/mil_screen_hooks_v1.py', ROOT/'training/arms/mil_screen_v1.py',
           ROOT/'training/arms/mil.py', ROOT/'training/src/raptor_common.py',
           ROOT/'training/src/raptor_runtime.py', ROOT/'training/src/exp143_v2.py',
           ROOT/'training/src/exp143_own0949.py', ROOT/'tools/exp143_v2.py',
           ROOT/'tools/exp112_full_swa.py', ROOT/'tools/exp113_radr50_local.py',
           ROOT/'tools/exp101_kfold_arch.py', ROOT/'tools/exp099_kfold.py',
           ROOT/'tools/exp084_group_loss.py', ROOT/'tools/exp107_kfold_radr50.py',
           ROOT/'training/src/raptor_dicom.py', ROOT/'training/src/raptor_corpus.py',
           ROOT/'training/src/raptor_exp185.py', ROOT/'training/arms/mil_small.py',
           ROOT/'training/arms/lifecycle.py', ROOT/CONTROL,
           *[ROOT/name for name in scientific_go.SOURCES]]


def asset(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=p.sha(path), bytes=path.stat().st_size)


def target_loader(bundle):
    from tools.vnext_r7_pl_targets_v110 import load_materialized
    return load_materialized(bundle)


def mil_recipe(n):
    """Copy the complete frozen screen config; only roster-dependent steps change."""
    c = copy.deepcopy(p.read(ROOT/CONTROL)['config'])
    steps = (n // 8) * 20
    c['updates'].update(loader_batches=n // 8, steps=steps, max_skips=max(1, int(steps*.01)))
    # Labels and membership are exclusively the separately authenticated bundle.
    c.pop('labels')
    c.pop('roots')
    return c


def prepare(stage, target_bundle, output, member_id):
    reg = p.registration(stage)
    target, _ = target_loader(target_bundle)
    members = [m for m in reg['members'] if m['id'] == member_id and m['kind'] in ('MIL', 'CoAt')]
    p.need(len(members) == 1, 'unknown production member')
    member = members[0]
    for k in ('seed', 'fold'):
        p.need(member[k] == target[k], 'target/member '+k)
    p.need(target['arm'] == reg['arm'] and target['registration_sha256'] == p.digest(reg), 'target registration')
    if member['kind'] == 'CoAt':
        from tools.exp143_v2 import RECIPE
        p.need(target['mode'] == 'all-data' and not target['val_ids'], 'CoAt requires all-data')
        recipe = copy.deepcopy(RECIPE)
        corpus = ROOT/'storage/cache/derived/EXP143_RAPTOR96_384_C140_S02'
        initial = [ROOT/COAT_INIT]
    else:
        recipe = mil_recipe(len(target['train_ids']))
        corpus = ROOT/'storage/cache/derived/RAPTOR64_336_C140_S02'
        initial = [ROOT/recipe['model']['weights'], ROOT/recipe['model']['initial_state']]
    bundle = Path(target_bundle)
    bundle = bundle/'target-spec.json' if bundle.is_dir() else bundle
    spec = dict(schema='pl-production-trainer-v1.1.1', version=VERSION, member=member,
        stage=str(Path(stage).resolve()), registration_sha256=p.digest(reg),
        target_bundle=asset(bundle), recipe=recipe, corpus=str(corpus),
        inputs=[asset(x) for x in [corpus/'manifest.json', *initial]],
        source_pins=[asset(x) for x in SOURCES], workers=0, clean_oof=False,
        exposure='EXPOSED_EXPLORATORY', metric_access='disabled', gold_access='disabled',
        validation_selection='none', coat_transfer='unvalidated MIL-to-CoAt efficacy transfer')
    spec['spec_sha256'] = p.digest(spec)
    path = Path(output)/'training-spec.json'
    p.dump(path, spec, immutable=True)
    return dict(status='PREPARED', spec=str(path), member=member_id, qualified=False)


def validate(path):
    spec = p.read(path)
    p.need(spec['spec_sha256'] == p.digest({k:v for k,v in spec.items() if k != 'spec_sha256'}), 'training spec seal')
    p.need(spec['schema'] == 'pl-production-trainer-v1.1.1' and spec['workers'] == 0,
           'production trainer schema/workers')
    reg = p.registration(spec['stage'])
    p.need(spec['registration_sha256'] == p.digest(reg) and spec['member'] in reg['members'], 'member registration')
    for x in [spec['target_bundle'], *spec['inputs'], *spec['source_pins']]:
        p.pin(x)
    target, labels = target_loader(spec['target_bundle']['path'])
    p.need(target['registration_sha256'] == p.digest(reg) and target['arm'] == reg['arm'], 'target registration')
    p.need(all(target[k] == spec['member'][k] for k in ('seed','fold')), 'member targets')
    p.need(set(labels) == set(target['train_ids']) and not set(labels)&set(target['val_ids']), 'training-only targets')
    if spec['member']['kind'] == 'MIL':
        expected = mil_recipe(len(target['train_ids']))
        p.need(expected['updates']['steps'] > 0, 'empty production loader')
    else:
        from tools.exp143_v2 import RECIPE
        expected = RECIPE
        p.need(target['mode'] == 'all-data' and not target['val_ids'], 'CoAt all-data only')
    p.need(spec['recipe'] == expected, 'frozen scientific recipe')
    return spec, target, labels


def admit(path, go=None):
    spec, target, _ = validate(path)
    if go is None:
        return dict(status='HOLD', reason='H-B-146 GO absent', production_run=False)
    p.need(target.get('engineering_only') is False, 'engineering targets forbidden in production')
    go_pin = asset(go)
    decision = scientific_go.validate(go_pin, target['arm'])
    return dict(status='ADMITTED', go=go_pin, decision=decision,
                spec_sha256=spec['spec_sha256'], production_run=False)


def provenance(spec, target):
    keys = ['teacher_sha256','target_sha256','train_ids_sha256','l2_pos_weight_sha256','registration_sha256']
    result = {k:target[k] for k in keys}
    result.update(arm=target['arm'], spec_sha256=spec['spec_sha256'], clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    if target['arm'] == 'fill':
        result['silence_mask_sha256'] = target['silence_mask_sha256']
    return result


def export(spec, target, model, out, *, smoke, counters, runtime, go=None, coat_checkpoint=None):
    """New files only; strict state roundtrip and serving metadata before receipt."""
    if not smoke:
        p.need(target.get('engineering_only') is False, 'engineering targets forbidden in production export')
        scientific_go.admission(go, target['arm'], spec['spec_sha256'])
    import torch
    from training.arms.mil import strict_load
    from tools.vnext_r7_pl_worker_v1 import checkpoint_metadata
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    member = spec['member']
    prov = provenance(spec, target)
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    strict_load(model, state)
    if member['kind'] == 'MIL':
        ck = dict(model=state, arch=member['arch'], res=224, lab=p.TARGETS,
            contract='raptor64_336_c140_s02', k_eval=24, norm='imagenet', window_policy='within-slot-valid',
            epoch=0 if smoke else 19, seed=member['seed'], fold=member['fold'],
            input_policy='center3', model_kind='finding_spatial_tokens_v1',
            flip_tta='coronal_axial_horizontal_mean_probability',
            provenance=dict(spec_sha256=spec['spec_sha256'], checkpoint_selection='fixed-final', runtime=runtime))
    else:
        p.need(coat_checkpoint is not None, 'CoAt needs actual fixed-SWA-BN checkpoint')
        ck = dict(coat_checkpoint, model=state)
    ck['pl_provenance'] = prov
    ck['label_set'] = 'heldout-PL-'+target['arm']+'-EXPOSED_EXPLORATORY'
    ck['engineering_only'] = smoke
    receipt = dict(prov, version=VERSION, status='ENGINEERING_SMOKE' if smoke else 'COMPLETE',
        seed=member['seed'], fold=member['fold'], counters=counters, runtime=runtime, go=go,
        checkpoint_selection='fixed-final' if member['kind']=='MIL' else 'fixed-swa-bn',
        source_pins=spec['source_pins'], target_bundle=spec['target_bundle'], engineering_only=smoke)
    checkpoint_metadata(ck, member, receipt)
    path = out/('smoke.pt' if smoke else 'model_best.pt')
    with path.open('xb') as f:
        torch.save(ck, f)
    strict_load(model, torch.load(path, map_location='cpu', weights_only=False)['model'])
    receipt['checkpoint_sha256'] = p.sha(path)
    p.dump(out/'training-receipt.json', receipt, immutable=True)
    return dict(status=receipt['status'], checkpoint=str(path), checkpoint_sha256=receipt['checkpoint_sha256'])


def mil_fit(spec, target, labels, output, *, device, smoke, go):
    from training.arms.hb146_pl_v1 import Arm
    from training.arms import mil_screen_hooks_v1 as h
    amp = 'none' if device == 'cpu' else 'fp16'
    runtime = h.configure_runtime(amp, device, policy='strict')
    seed = spec['member']['seed']
    h.seed_all(seed)
    ids = target['train_ids'][:2] if smoke else target['train_ids']
    corpus = h.common.Corpus(spec['corpus'])
    h.verify_shards(corpus, ids)
    arm = Arm(labels)
    # The base labels are never read: the exact H-B-146 SoftSource supplies them.
    zeros = {u:h.np.zeros(12,h.np.float32) for u in ids}
    ds = arm.dataset(h.MILWindows(spec['corpus'], ids, zeros, 12, True, seed), train=True)
    gen = h.torch.Generator().manual_seed(seed)
    loader = h.DataLoader(ds, batch_size=1 if smoke else 8, shuffle=True, drop_last=True,
                         num_workers=0, generator=gen, collate_fn=arm.collate_fn)
    epochs = 1 if smoke else 20
    steps = len(loader)*epochs
    p.need(steps == (2 if smoke else spec['recipe']['updates']['steps']), 'registered update count')
    model = h.build_model(spec['recipe']['model'], ROOT).to(device)
    opt, sched, scaler = h.optimizer_stack(model, steps, device, amp)
    counters = dict(updates=0, skips=0)
    for epoch in range(epochs):
        ds.set_epoch(epoch)
        h.train_epoch_with(arm, model, loader, opt, sched, scaler, h.torch.tensor(target['weights']),
            device, amp, 0 if smoke else max(1,int(steps*.01)), counters)
    p.need(counters['updates'] == steps, 'incomplete production fit')
    result = export(spec,target,model,output,smoke=smoke,counters=counters,runtime=runtime,go=go)
    p.need(device != 'cpu' or not h.torch.cuda.is_initialized(), 'CPU initialized CUDA')
    return result


def fit(path, output, *, device='cpu', smoke=True, go=None):
    p.need(device in ('cpu','cuda') and (smoke or device == 'cuda'), 'full fitting requires CUDA')
    spec,target,labels = validate(path)
    admission = admit(path,go) if not smoke else None
    p.need(smoke or admission['status']=='ADMITTED','fit not admitted')
    if spec['member']['kind']=='MIL':
        return mil_fit(spec,target,labels,output,device=device,smoke=smoke,go=admission)
    from training.arms.vnext_r7_pl_coat_v110 import fit as coat_fit
    return coat_fit(spec,target,labels,output,device=device,smoke=smoke,go=admission)


def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('command',choices=['prepare','validate','admit','fit'])
    for name in ('stage','targets','output','spec','go'):
        a.add_argument('--'+name,type=Path)
    a.add_argument('--member')
    a.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a.add_argument('--full',action='store_true')
    args=a.parse_args(argv)
    if args.command=='prepare':
        result=prepare(args.stage,args.targets,args.output,args.member)
    elif args.command=='validate':
        spec,target,_=validate(args.spec)
        result=dict(status='PASS',spec_sha256=spec['spec_sha256'],train=len(target['train_ids']),val=len(target['val_ids']))
    elif args.command=='admit':
        result=admit(args.spec,args.go)
    else:
        result=fit(args.spec,args.output,device=args.device,smoke=not args.full,go=args.go)
    import json
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
