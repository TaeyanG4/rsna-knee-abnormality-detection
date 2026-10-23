"""Explicit MIL v1.2.0 parity and CPU launch card (2026-10-11).

Changelog: add S43 F1-4 production-member cards without paired controls.

Real cache/weights/rosters are required, never skipped. No GPU work or labels
printed. Synthetic target notes below test contracts only, never grant PASS.
"""
from __future__ import annotations

import copy
import gc
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.arms import lifecycle as life, mil

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
PROTOCOLS = ['.local/exp185-overlay/protocols/S42.json', '.local/exp186-overlay/protocols/fold1.json']


MEMBERS = [(42, fold) for fold in range(5)] + [(43, 0)]


def spec_for(fold, seed=42):
    return life.ArmSpec((ROOT / f'training/configs/mil-s{seed}-f{fold}-v3.json').read_bytes())


def test_real_cpu_observation_has_strict_json_types():
    observed = mil.observation('cpu')
    assert type(observed['runtime']['torch']) is str
    assert life.parse_json(life.canonical_bytes(observed)) == observed
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('invalid', [None, 'train_batches', 'validation_batches',
                                   'optimizer_steps', 'bn_batches', 'bn_module_count'])
def test_bn_free_pass_still_requires_executed_work(invalid):
    """Synthetic schema fixture only; never emit a launch note."""
    spec = spec_for(0)
    smokes = {}
    for kind in ('cpu', 'target'):
        smokes[kind] = dict(
            hostname='fixture-cpu' if kind == 'cpu' else spec.document['target_environment']['hostname'],
            device='cpu' if kind == 'cpu' else 'cuda',
            environment_sha256=spec.bindings['environment'],
            bindings_sha256=life.content_sha256(spec.bindings),
            command=['synthetic-schema-only', kind], exit_code=0, log_sha256='0' * 64,
            results=dict(real_data=True, train_batches=1, validation_batches=1, bn_batches=0,
                         bn_module_count=0, optimizer_steps=1, checkpoint_reloaded=True, finite_outputs=True))
    note = dict(schema=life.VERIFICATION_SCHEMA, scope=spec.scope,
                reviewer=dict(name='synthetic-fixture', host='test'), verdict='PASS', blockers=[],
                notes=['Schema only; not execution evidence.'], bindings=spec.bindings,
                checks=[dict(name='schema', command=['synthetic-schema-only'], exit_code=0,
                             log_sha256='0' * 64)], smokes=smokes)
    if invalid is None:
        assert life.require_verification_pass(note, **spec.verification_args)
    else:
        note['smokes']['cpu']['results'][invalid] = 1 if invalid.startswith('bn_') else 0
        with pytest.raises(life.ContractError):
            life.require_verification_pass(note, **spec.verification_args)


@pytest.fixture(scope='module')
def real():
    spec = spec_for(0)
    corpus, train, val, labels, weights = mil.validate_data(spec, ROOT, CORPUS)
    return spec, corpus, train, val, labels, weights


@pytest.mark.parametrize('seed,fold', MEMBERS)
def test_registered_fold_control_and_scientific_parity(seed, fold):
    spec = spec_for(fold, seed)
    corpus, train, val, labels, weights = mil.validate_data(spec, ROOT, CORPUS)
    protocol = next(p for p in spec.document['inputs']['controls'] if '/protocols/' in p)
    legacy = mil.read(ROOT / protocol)
    f = legacy['folds'][str(fold)]
    assert (len(train), len(val)) == (f['n_train'], f['n_val'])
    assert mil.uid_hash(train) == f['ordered_ids']['train_sha256']
    assert mil.uid_hash(val) == f['ordered_ids']['val_sha256']
    assert np.array_equal(weights, np.asarray(f['pos_weight'], np.float32))
    assert spec.document['config']['updates']['steps'] == f['steps'] == len(train) // 8 * 20
    assert spec.document['config']['labels']['semantic_sha256'] == legacy['label_set']['semantic_sha256']
    assert spec.document['target_environment']['runtime']['gpu'] == 'NVIDIA GeForce RTX 3090'
    assert not any('\\' in p for p in spec.inventory)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('field,value', [('seed', 44), ('fold', 5), ('family', 'coatnet-bridge')])
def test_unsupported_member_rejected(field, value):
    doc = spec_for(0).document
    doc[field] = value
    with pytest.raises(life.ContractError):
        mil.validate_recipe(life.ArmSpec.from_document(doc))


@pytest.mark.parametrize('section,key,value', [
    ('windows', 'train_k', 24), ('sampler', 'drop_last', False),
    ('updates', 'epochs', 21), ('readout', 'selection', 'best'),
    ('bn', 'module_count', 1), ('model', 'arch', 'wrong'),
    ('updates', 'wall_budget', 1)])
def test_changed_recipe_rejected(section, key, value):
    doc = spec_for(0).document
    doc['config'][section][key] = value
    with pytest.raises(life.ContractError):
        mil.validate_recipe(life.ArmSpec.from_document(doc))


@pytest.mark.parametrize('change', ['source', 'roster', 'labels', 'weights', 'updates', 'order', 'groups'])
def test_real_inputs_or_scientific_identity_tamper_rejected(change, real):
    spec, corpus, train, val, labels, weights = real
    doc = spec.document
    if change in ('source', 'roster'):
        group = 'sources' if change == 'source' else 'rosters'
        doc['inputs'][group][next(iter(doc['inputs'][group]))] = '0' * 64
    elif change == 'labels':
        doc['config']['labels']['semantic_sha256'] = '0' * 64
    elif change == 'weights':
        doc['config']['labels']['pos_weight'][0] += .01
    elif change == 'updates':
        doc['config']['updates']['steps'] += 1
    else:
        import csv
        config = doc['config']['roots']
        rows = list(csv.DictReader((ROOT / config['groups']).open(encoding='utf-8', newline='')))
        canonical = mil.read(ROOT / config['canonical'])
        if change == 'order':
            train = train[::-1]
        else:
            a = next(r for r in rows if r['split'] == 'train')
            b = next(r for r in rows if r['split'] == 'val')
            b['component_id'] = a['component_id']
        with pytest.raises(life.ContractError):
            mil.validate_membership(corpus.ids.tolist(), {'train': train, 'val': val}, rows, canonical['train']+canonical['val'])
        return
    with pytest.raises(life.ContractError):
        mil.validate_data(life.ArmSpec.from_document(doc), ROOT, CORPUS)


@pytest.mark.parametrize('seed', [42, 43])
@pytest.mark.parametrize('k,train,epoch', [(12, True, 0), (12, True, 3), (24, False, 0), (36, False, 0)])
def test_real_windows_augmentation_and_rng_match_legacy(seed, k, train, epoch, real, monkeypatch):
    _, corpus, ids, val, labels, _ = real
    monkeypatch.syspath_prepend(str(ROOT / 'training/src'))
    old = importlib.import_module('raptor_exp179')
    chosen = ids[:2] if train else val[:2]
    new = mil.MILWindows(CORPUS, chosen, labels, k, train, seed)
    legacy = old.ScaleSliceWindows(CORPUS, chosen, labels, 224, k, train, 'imagenet', seed,
                                  window_policy='within-slot-valid', input_policy='center3', crop_policy='mixed')
    new.set_epoch(epoch)
    legacy.set_epoch(epoch)
    for i in range(2):
        x, y, uid = new[i]
        reference, target, old_uid = legacy[i]
        assert uid == old_uid and torch.equal(x, reference) and torch.equal(y, target)
    # Padding and plane boundaries use the same six draws even for empty windows.
    centers = [-1, 1, 17, 19, 31, 33, 43, 45, 51, 53, 62, -1]
    x = mil.scale_windows(corpus.volume(chosen[0]), centers)
    expected = old.scale_windows(corpus.volume(chosen[0]), centers, 224)
    assert np.array_equal(x, expected)
    rng, reference_rng = random.Random(17), random.Random(17)
    assert np.array_equal(mil.augment_windows(x, centers, rng, corpus.spec),
                          old.augment_windows(expected, centers, reference_rng, corpus.spec))
    assert rng.getstate() == reference_rng.getstate()


def test_actual_model_initial_state_rng_and_logits_match_legacy(real):
    from safetensors.torch import load_file
    from training.src.raptor_exp185 import install_spatial_pooling
    spec, _, ids, _, labels, _ = real
    config = spec.document['config']['model']
    mil.seed_all(42)
    current = mil.build_model(config, ROOT)
    current_rng = torch.get_rng_state().clone()
    mil.seed_all(42)
    legacy = install_spatial_pooling(mil.common.build_model(mil.ARCH, pretrained=False))
    state = load_file(str(ROOT / config['weights']))
    state.pop('head.fc.weight', None)
    state.pop('head.fc.bias', None)
    legacy.backbone.load_state_dict(state, strict=True)
    initial = torch.load(ROOT / config['initial_state'], map_location='cpu', weights_only=True)
    legacy.load_state_dict(initial, strict=True)
    assert torch.equal(current_rng, torch.get_rng_state())
    assert len(initial) == 188 and len(legacy.backbone.state_dict()) == 180
    assert mil.state_hash(current.state_dict()) == config['initial_semantic_sha256']
    assert all(torch.equal(v, legacy.state_dict()[k]) for k, v in current.state_dict().items())
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in current.modules())
    ds = mil.MILWindows(CORPUS, ids[:1], labels, 12, True, 42)
    x = ds[0][0][:2].unsqueeze(0)
    with torch.no_grad():
        for training in (False, True):
            mil.seed_all(99)
            actual = current.train(training)(x)
            state_after = torch.get_rng_state().clone()
            mil.seed_all(99)
            expected = legacy.train(training)(x)
            assert torch.equal(actual, expected) and torch.equal(state_after, torch.get_rng_state())
    del current, legacy, initial, state
    gc.collect()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(4, 12))

    def forward(self, x):
        return self.head(self.backbone(x.mean((1, 3, 4))))


def small_stack(real, workers=0):
    _, _, ids, _, labels, _ = real
    mil.seed_all(42)
    ds = mil.MILWindows(CORPUS, ids[:8], labels, 12, True, 42)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, generator=generator, collate_fn=mil.collate,
                        worker_init_fn=mil.worker_init, multiprocessing_context='spawn' if workers else None)
    model = TinyModel()
    opt, sched, scaler = mil.optimizer_stack(model, 12, 'cpu', 'none')
    return ds, loader, model, opt, sched, scaler, generator


def shutdown(loader):
    if loader._iterator is not None:
        loader._iterator._shutdown_workers()


def test_train_epoch_matches_legacy_equations(real):
    weights = torch.from_numpy(real[-1])
    ds, loader, model, opt, sched, scaler, generator = small_stack(real)
    result = mil.train_epoch(model, loader, opt, sched, scaler, weights, 'cpu', 'none', 0, {'updates': 0, 'skips': 0})
    saved = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    ds, loader, model, opt, sched, scaler, generator = small_stack(real)
    model.train()
    lossf = nn.BCEWithLogitsLoss(pos_weight=weights)
    total = 0.
    opt.zero_grad(set_to_none=True)
    for x, y, _ in loader:
        loss = lossf(model(x).float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        total += loss.item()
    assert result == {'loss': total / len(loader), 'batches': len(loader)}
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in saved.items())


@pytest.mark.parametrize('workers', [0, 2])
def test_real_data_epoch_resume_exact_trajectory(tmp_path, real, workers):
    ds, loader, model, opt, sched, scaler, generator = small_stack(real, workers)
    identity = {'fixture': 'real-data-tiny-model', 'workers': workers}
    weights, history, counters = torch.from_numpy(real[-1]), [], {'updates': 0, 'skips': 0}
    try:
        for epoch in range(3):
            ds.set_epoch(epoch)
            history.append(mil.train_epoch(model, loader, opt, sched, scaler, weights, 'cpu', 'none', 0, counters))
            if epoch == 0:
                mil.save_epoch(tmp_path/'epoch.pt', epoch=0, model=model, opt=opt, sched=sched, scaler=scaler,
                               generator=generator, history=history, counters=counters, identity=identity, device='cpu')
        expected = copy.deepcopy(model.state_dict())
        expected_generator, expected_rng = generator.get_state().clone(), torch.get_rng_state().clone()
        expected_history, expected_counters = copy.deepcopy(history), counters.copy()
    finally:
        shutdown(loader)
    ds, loader, model, opt, sched, scaler, generator = small_stack(real, workers)
    try:
        start, history, counters = mil.restore_epoch(tmp_path/'epoch.pt', model=model, opt=opt, sched=sched,
            scaler=scaler, generator=generator, identity=identity, device='cpu', loader=loader)
        for epoch in range(start, 3):
            ds.set_epoch(epoch)
            history.append(mil.train_epoch(model, loader, opt, sched, scaler, weights, 'cpu', 'none', 0, counters))
        assert history == expected_history and counters == expected_counters
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected.items())
        assert torch.equal(expected_generator, generator.get_state()) and torch.equal(expected_rng, torch.get_rng_state())
        with pytest.raises(life.ContractError, match='receipt differs'):
            mil.restore_epoch(tmp_path/'epoch.pt', model=model, opt=opt, sched=sched, scaler=scaler,
                              generator=generator, identity={'wrong': True}, device='cpu')
        with (tmp_path/'epoch.pt').open('ab') as f:
            f.write(b'tamper')
        with pytest.raises(life.ContractError, match='receipt differs'):
            mil.restore_epoch(tmp_path/'epoch.pt', model=model, opt=opt, sched=sched, scaler=scaler,
                              generator=generator, identity=identity, device='cpu')
    finally:
        shutdown(loader)


def test_tta_readout_order_and_csv_parity(real, monkeypatch, tmp_path):
    _, corpus, _, ids, labels, _ = real
    monkeypatch.syspath_prepend(str(ROOT / 'training/src'))
    from tools import exp124_flip_tta as legacy
    model = TinyModel().eval()
    for k in (24, 36):
        ds = mil.MILWindows(CORPUS, ids[:2], labels, k, False, 42)
        loader = DataLoader(ds, batch_size=2, collate_fn=mil.collate)
        x, _, uids = next(iter(loader))
        flipped = mil.flip_inputs(x, uids, corpus, k)
        for i, uid in enumerate(uids):
            centers = mil.common.select_centers(corpus.mask(uid), 64, k, corpus.spec, 'within-slot-valid')
            mask = legacy.flip_mask(centers, legacy.plane_ranges(corpus.spec))
            assert torch.equal(flipped[i], legacy.mirror_windows(x[i], mask))
        with torch.no_grad():
            p = (model(x).sigmoid().numpy() + model(flipped).sigmoid().numpy()) * np.float32(.5)
        actual, ordered, batches = mil.evaluate(model, loader, 'cpu', 'none', corpus, tta=True)
        assert ordered == ids[:2] and batches == 1 and np.array_equal(actual, p)
        path = tmp_path/f'k{k}.csv'
        life.create_new(path, life.prediction_csv_bytes(ordered, actual, mil.TARGETS, id_column=mil.UID))
        import pandas as pd
        frame = pd.DataFrame(p.astype(np.float32), columns=mil.TARGETS)
        frame.insert(0, mil.UID, ordered)
        assert path.read_bytes() == frame.to_csv(index=False, lineterminator='\n').encode()


@pytest.mark.parametrize('seed,fold,kind', [(s, f, 'experiment') for s, f in MEMBERS]
                         + [(43, f, 'member') for f in range(1, 5)])
def test_real_model_cpu_cli_card(seed, fold, kind, tmp_path):
    """Every public CPU command with actual ConvNeXt update + 3 readouts."""
    base = Path(os.environ.get('MIL_ARM_SMOKE_OUT', str(tmp_path/'card')))
    work = base / f'{kind}-s{seed}-f{fold}'
    work.mkdir(parents=True, exist_ok=False)
    commands = []
    def command(module, *args, expected=0):
        argv = [sys.executable, '-B', '-m', module, *map(str, args)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
        result = subprocess.run(argv, cwd=ROOT, env=env, text=True, encoding='utf-8', capture_output=True)
        log = work/f'command-{len(commands):02d}.log'
        log.write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        commands.append({'argv': argv, 'returncode': result.returncode, 'log': log.name, 'sha256': mil.sha(log)})
        (work/'command-results.json').write_bytes(life.canonical_bytes(commands))
        assert result.returncode == expected, f'{module} {args[0]} exit {result.returncode}; see {log}'
        return json.loads(result.stdout)
    spec_path, stage, run = work/'spec.json', work/'stage', work/'run'
    infra = 'tools.arm_lifecycle'
    name = f'mil-member-s{seed}-f{fold}-v1' if kind == 'member' else f'mil-s{seed}-f{fold}-v3'
    draft = ROOT/f'training/configs/{name}.json'
    spec = life.ArmSpec(draft.read_bytes())
    migrated = work/'migrated-spec.json'
    if kind == 'member':
        evidence = spec.document['inputs']['evidence']
        target = next(p for p, h in evidence.items()
                      if h == spec.document['target_environment']['evidence']['declaration.json'])
        recipe = next(p for p in evidence if p != target)
        command('tools.mil_register', '--kind', 'member', '--root', ROOT,
            '--recipe', recipe.replace('/', '\\'), '--recipe-sha256', evidence[recipe],
            '--target', target.replace('/', '\\'), '--target-sha256', evidence[target],
            '--arm', spec.document['arm'], '--attempt', spec.document['attempt'], '--out', migrated)
        assert not spec.document['inputs']['controls']
    else:
        controls = spec.document['inputs']['controls']
        protocol = next(p for p in controls if '/protocols/' in p)
        control = next(p.removesuffix('/summary.json') for p in controls if p.endswith('/summary.json'))
        # Real legacy Windows paths are normalized before immutable registration.
        command('tools.mil_register', '--root', ROOT, '--protocol', protocol.replace('/', '\\'),
            '--protocol-sha256', controls[protocol], '--control', control.replace('/', '\\'),
            '--summary-sha256', controls[control+'/summary.json'],
            '--provenance-sha256', controls[control+'/provenance.json'],
            '--arm', spec.document['arm'], '--attempt', spec.document['attempt'], '--out', migrated)
    assert migrated.read_bytes() == draft.read_bytes()
    command(infra, 'register', '--draft', draft, '--root', ROOT, '--out', spec_path)
    command(infra, 'validate', '--spec', spec_path, '--root', ROOT)
    command(infra, 'prepare', '--spec', spec_path, '--root', ROOT, '--out', stage)
    command(infra, 'prepare', '--spec', spec_path, '--root', ROOT, '--out', stage)
    command(infra, 'validate', '--spec', spec_path, '--root', stage, '--stage')
    runner = 'training.runners.mil_arms'
    shared = ['--spec', spec_path, '--stage', stage, '--corpus', CORPUS]
    command(runner, 'validate', *shared)
    result = command(runner, 'smoke', *shared, '--device', 'cpu', '--workers', '0', '--out', run)
    assert result['status'] == 'ENGINEERING_SMOKE'
    assert result['results'] == dict(real_data=True, train_batches=1, validation_batches=3, bn_batches=0,
                                   bn_module_count=0, optimizer_steps=1, checkpoint_reloaded=True, finite_outputs=True)
    note_path = run/'verification.json'
    admit = command(infra, 'admit', '--spec', spec_path, '--stage', stage, '--note', note_path,
                    '--evidence-root', run, '--dry-run')
    assert admit['status'] == 'HOLD' and admit['launch_authorized'] is False
    # Re-evaluate a completed CPU checkpoint without taking a second update.
    resumed = work/'resumed'
    resume_result = command(runner, 'smoke', *shared, '--device', 'cpu', '--workers', '0', '--out', resumed,
                            '--resume', run/'epochs/epoch-000.pt')
    assert resume_result['results']['train_batches'] == resume_result['results']['optimizer_steps'] == 0
    assert mil.read(resumed/'verification.json')['verdict'] == 'HOLD'
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        assert (run/'outputs'/name).read_bytes() == (resumed/'outputs'/name).read_bytes()
    command(runner, 'train', *shared, '--device', 'cpu', '--out', work/'forbidden', expected=2)
    metrics = work/'metrics.json'
    mil.write_new(metrics, {k: v for k, v in result['results'].items() if type(v) is int})
    receipt_path = work/'receipt.json'
    receipt = command(infra, 'receipt', '--spec', spec_path, '--outputs', run/'outputs', '--out', receipt_path,
                      '--verification-sha256', mil.sha(note_path), '--exit-code', '0',
                      '--elapsed-seconds', str(result['elapsed_seconds']), '--metrics', metrics)
    archive = work/'smoke.tar'
    command(infra, 'export', '--spec', spec_path, '--outputs', run/'outputs', '--receipt', receipt_path,
            '--receipt-sha256', receipt['receipt_sha256'], '--out', archive)
    command(infra, 'collect', '--spec', spec_path, '--archive', archive,
            '--receipt-sha256', receipt['receipt_sha256'], '--out', work/'collected')
    note = mil.read(note_path)
    assert life.verify_evidence(spec, note, run) is not None
    summary = mil.read(run/'outputs/summary.json')
    assert summary['cpu_cuda_initialized'] is False
    assert summary['scientific_status'] == 'UNASSESSED'
    assert summary['run_kind'] == kind
    assert mil.read(run/'outputs/provenance.json')['run_kind'] == kind
    # BN-free schema accepts zero BN only when architecture config declares it.
    with pytest.raises(life.ContractError):
        life.validate_verification_note(note, expected_scope=spec.scope, expected_bindings=spec.bindings,
                                        target_environment=spec.document['target_environment'])
    changed = copy.deepcopy(note)
    changed['smokes']['cpu']['results']['bn_module_count'] = 1
    with pytest.raises(life.ContractError):
        life.validate_verification_note(changed, **spec.verification_args)
    changed_evidence = tmp_path/'tampered-evidence'
    shutil.copytree(run, changed_evidence)
    (changed_evidence/'outputs/summary.json').write_bytes(b'{}')
    with pytest.raises(life.ContractError, match='hash differs'):
        life.verify_evidence(spec, note, changed_evidence)
