"""CoAtNet seam v1.0.1: private scratch root for unchanged frozen CPU runner.

Changelog: copy exact scientific inputs/sources into the fixed scratch card;
retain the legacy output restriction without writing large files outside it.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from training.arms import coatnet as c
from training.src import exp143_own0949 as old
from training.src import exp181_aug1, exp188_coverage, exp192_crop
from tools.exp112_full_swa import state_dict_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def context():
    return c.load_control(ROOT)


class TinyBackbone(nn.Module):
    num_features = 4

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=32, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x)).mean((-1, -2))


def dataset(context, variant, augment=True, ids=None):
    p = context.registration
    return c.StudyDataset(ROOT / c.CORPUS, ids or p['train_ids'][:1], p['train_labels'], 42,
        c.PRESETS[variant], augment=augment,
        metadata=ROOT / c.METADATA if variant == 'coverage' else None)


@pytest.mark.parametrize('variant', c.PRESETS)
@pytest.mark.parametrize('epoch,augment', [(1, True), (16, True), (16, False)])
def test_real_training_and_bn_windows(context, variant, epoch, augment):
    current = dataset(context, variant, augment)
    p = context.registration
    legacy_cls = dict(control=old.StudyDataset, coverage=exp188_coverage.CoverageStudyDataset,
                      crop=exp192_crop.CropStudyDataset, aug1=exp181_aug1.Aug1StudyDataset)[variant]
    legacy = legacy_cls(ROOT / c.CORPUS, p['train_ids'][:1], p['train_labels'], 42, augment=augment)
    current.set_epoch(epoch)
    legacy.set_epoch(epoch)
    actual, expected = current[0], legacy[0]
    assert actual[2] == expected[2]
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[0].shape == (12, 3, 384, 384)


@pytest.mark.parametrize('variant', c.PRESETS)
@pytest.mark.parametrize('mirror', [False, True])
def test_real_eval_windows(context, variant, mirror):
    vol, mask = old.Corpus(ROOT / c.CORPUS).get(context.registration['val_ids'][0])
    expected = (exp192_crop.windows if variant == 'crop' else old.windows)(vol, mask, mirror=mirror)
    actual = c.evaluation_windows(c.PRESETS[variant], vol, mask, mirror)
    assert actual.shape == (94, 3, 320, 320)
    assert np.array_equal(actual, expected)


def test_factory_has_exact_initialization_state_gradients_and_rng():
    import raptor_common
    original_factory = raptor_common.build_backbone
    torch.random.default_generator.manual_seed(42)
    legacy = old.build_model(backbone_factory=TinyBackbone)
    legacy_rng = torch.get_rng_state().clone()
    torch.random.default_generator.manual_seed(42)
    model = c.build_model(backbone_factory=TinyBackbone)
    assert raptor_common.build_backbone is original_factory
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    assert list(model.state_dict()) == list(legacy.state_dict())
    assert state_dict_sha256(model.state_dict()) == state_dict_sha256(legacy.state_dict())
    x = torch.linspace(-1, 1, 2*12*3*32*32).reshape(2, 12, 3, 32, 32)
    for net in (legacy, model):
        torch.random.default_generator.manual_seed(42001)
        net(x).sum().backward()
    for (name, param), (other_name, other) in zip(model.named_parameters(), legacy.named_parameters()):
        assert name == other_name
        assert torch.equal(param.grad, other.grad)
    model.eval()
    legacy.eval()
    assert torch.equal(model(x), legacy(x))


@pytest.mark.parametrize('variant', c.PRESETS)
def test_real_inference_parity_and_one_head_per_view(context, variant):
    torch.random.default_generator.manual_seed(42)
    model = c.build_model(backbone_factory=TinyBackbone).eval()
    vol, mask = old.Corpus(ROOT / c.CORPUS).get(context.registration['val_ids'][0])
    calls = []
    hook = model.att.register_forward_hook(lambda _m, args, _out: calls.append(args[0].shape[1]))
    plain, mirror = c.infer_pair(model, c.PRESETS[variant], vol, mask, chunk=16)
    hook.remove()
    assert calls == [94, 94]
    if variant == 'crop':
        expected = exp192_crop.infer_pair(model, vol, mask, chunk=16)
        assert np.array_equal(plain, expected[0])
        assert np.array_equal(mirror, expected[1])
    else:
        assert np.array_equal((plain + mirror)*.5, old.infer_volume(model, vol, mask, chunk=16))


def test_sampler_all_epochs_and_tail_accumulation(context):
    from torch.utils.data import DistributedSampler
    from training.src.exp143_v2 import accumulation_group
    p = context.registration
    ds = dataset(context, 'coverage', ids=p['train_ids'])
    legacy = old.StudyDataset(ROOT / c.CORPUS, p['train_ids'], p['train_labels'], 42)
    current_sampler = DistributedSampler(ds, 1, 0, shuffle=True, seed=42, drop_last=True)
    old_sampler = DistributedSampler(legacy, 1, 0, shuffle=True, seed=42, drop_last=True)
    for epoch in range(1, 17):
        current_sampler.set_epoch(epoch)
        old_sampler.set_epoch(epoch)
        assert list(current_sampler) == list(old_sampler)
        groups = [accumulation_group(i, len(ds), 16) for i in range(len(ds))]
        assert sum(final for final, _ in groups) == 216
        assert groups[-1] == (True, 15)


def test_coverage_metadata_is_instance_local_and_mask_checked(context):
    first, second = dataset(context, 'coverage'), dataset(context, 'coverage')
    uid = first.ids[0]
    assert first.metadata is not second.metadata
    first.metadata[uid]['mask_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='mask identity'):
        first[0]
    second[0]


def test_spawn_worker_shares_epoch_without_global_metadata(context):
    from torch.utils.data import DataLoader
    ds = dataset(context, 'coverage')
    loader = DataLoader(ds, batch_size=1, num_workers=1, multiprocessing_context='spawn',
                        persistent_workers=True, collate_fn=old.collate)
    for epoch in (1, 16):
        ds.set_epoch(epoch)
        result = next(iter(loader))
        assert torch.equal(result[0][0], ds[0][0])
    del loader


@pytest.mark.parametrize('kwargs', [dict(sampler='other'), dict(head='arbitrary'),
    dict(crop='other'), dict(augmentation='other'),
    dict(sampler='coverage', crop='alternating-context-detail'), dict(sampler='coverage', augmentation='aug1')])
def test_unregistered_config_rejected(kwargs):
    with pytest.raises(ValueError, match='unsupported'):
        c.Config(**kwargs)


def test_control_contract_is_not_runtime_or_launch_qualification(context):
    report = context.report()
    assert report['train'] == 3455 and report['val'] == 888
    assert report['total_updates'] == 3456
    assert report['launch_authorized'] is False
    assert report['runtime_qualification'] == 'UNAVAILABLE_IN_LEGACY_CONTROL'
    assert 'train_labels' not in json.dumps(report)
    assert 'registration=' not in repr(context)


def test_changed_control_rejected(tmp_path):
    rel = next(iter(c.PINS))
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True)
    dest.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='CONTROL input changed'):
        c.load_control(tmp_path)


def test_fail_closed_inference():
    model = c.build_model(backbone_factory=TinyBackbone)
    with pytest.raises(ValueError, match='eval mode'):
        c.infer_pair(model, c.Config(), None, None)
    model.eval()
    with pytest.raises(ValueError, match='chunk'):
        c.infer_pair(model, c.Config(), None, None, chunk=0)


@pytest.mark.parametrize('variant', c.PRESETS)
def test_real_cpu_cli_card(context, variant):
    """Full actual architecture, native K12 update, BN and two K94 views, CPU only."""
    location = os.environ.get('COATNET_ARM_SMOKE_OUT')
    if not location:
        pytest.skip('Use run-step6a-tests.py for durable real CPU cards')
    card = Path(location) / variant
    card.resolve().relative_to((ROOT / '.local/scratch/astra-2').resolve())
    card.mkdir(parents=True, exist_ok=False)
    # The frozen runner requires <root>/.local/refactor. A sparse private root
    # keeps that contract and all bytes below the caller's fixed scratch task.
    private = card / 'mirror'
    study = old.Corpus(ROOT / c.CORPUS).rows[context.registration['train_ids'][0]]
    copied = set(context.inputs) | set(c.SOURCES) | {
        f'{c.CORPUS}/study_{study:05d}.{suffix}' for suffix in ('npy', 'json')}
    original_hashes = {rel: c.sha256(ROOT / rel) for rel in copied}
    for rel in sorted(copied):
        target = private / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
        assert c.sha256(target) == original_hashes[rel]
    assert c.load_control(private).report() == context.report()
    smoke = private / '.local/refactor/smoke'
    commands = []

    def run(args, code, status):
        argv = [sys.executable, '-B', '-m', 'training.runners.coatnet_arms', *args]
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        log = card / f'command-{len(commands):02d}.log'
        with log.open('x', encoding='utf-8') as stream:
            stream.write(result.stdout + result.stderr)
        commands.append(dict(argv=argv, returncode=result.returncode, log=log.name, sha256=c.sha256(log)))
        # Retain failed command evidence too.
        with (card / f'command-{len(commands)-1:02d}.json').open('x', encoding='utf-8') as stream:
            json.dump(commands[-1], stream, indent=2)
        assert result.returncode == code, result.stderr[-2000:] + result.stdout[-2000:]
        output = json.loads(result.stdout)
        assert output['status'] == status and output['launch_authorized'] is False
        return output

    run(['inspect-control', '--root', str(private)], 0, 'CONTROL_CONTRACT_VERIFIED')
    args = ['cpu-smoke', '--root', str(private), '--variant', variant, '--out', str(smoke)]
    receipt = run(args, 0, 'HOLD')
    before = c.sha256(smoke / 'receipt.json')
    run(args, 2, 'REJECTED')
    assert c.sha256(smoke / 'receipt.json') == before
    run(['cpu-smoke', '--root', str(private), '--variant', variant,
         '--out', str(private / 'artifacts/forbidden-smoke')], 2, 'REJECTED')
    assert not (private / 'artifacts/forbidden-smoke').exists()
    assert receipt['initial_state_sha256'] == context.registration['expected_initial_state_sha256']
    assert receipt['bn']['bn_modules'] == 25
    assert receipt['bn']['studies'] == 1 and receipt['bn']['num_batches_tracked'] == [1]
    assert receipt['checkpoint_reload_exact'] and not receipt['cuda_initialized']
    assert receipt['prediction_shape'] == [3, 12]
    assert receipt['optimizer_updates'] == 1 and receipt['training_windows'] == 12
    assert receipt['initial_state_sha256'] != receipt['updated_state_sha256']
    for rel, digest in {**receipt['inputs'], **receipt['sources']}.items():
        assert c.sha256(ROOT / rel) == digest
        assert c.sha256(private / rel) == digest
    for name, digest in receipt['artifacts'].items():
        assert c.sha256(smoke / name) == digest
    assert {rel: c.sha256(ROOT / rel) for rel in copied} == original_hashes
    predictions = np.load(smoke / 'predictions.npy', allow_pickle=False)
    assert np.isfinite(predictions).all() and ((0 <= predictions) & (predictions <= 1)).all()
    assert np.array_equal(predictions[2], .5 * (predictions[0] + predictions[1]))
    c.write_new(card / 'command-results.json', commands)
