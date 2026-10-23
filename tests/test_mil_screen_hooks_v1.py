"""Shared-arm hooks v1: structural identity, real CPU byte identity, sampler
stream, metadata path, single-difference binding and recorded HB122 cards.
Synthetic or aggregate-only; never prints label rows or report text."""
import ast
import copy
import inspect
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from training.arms import mil as base
from training.arms import mil_screen_v1 as shared
from training.arms import mil_screen_hooks_v1 as hooks
from training.arms import mil_screen_demo_arm_v1 as demo
from tools import hb122_build_v1 as build
from tools import mil_screen_arm_build_v1 as armbuild

torch.set_num_threads(2)
ROOT = build.ROOT
WORK = build.WORK
DEMO = WORK / 'demo'


def dump(source):
    return ast.dump(ast.parse(source))


def substituted(function, pairs):
    source = inspect.getsource(function)
    for new, old in pairs:
        assert new in source, new
        source = source.replace(new, old)
    return source


def test_run_is_shared_run_with_only_hook_points():
    pairs = [
        ('def run(spec, stage, corpus_dir, destination, *, arm, mode,', 'def run(spec, stage, corpus_dir, destination, *, mode,'),
        ('validate_data_with(arm, spec, stage, corpus_dir)', 'validate_data(spec, stage, corpus_dir)'),
        ('dataset = arm.dataset(MILWindows(corpus_dir, train_ids, selected_labels, 12, True, doc["seed"]), spec, stage, train=True)',
         'dataset = MILWindows(corpus_dir, train_ids, selected_labels, 12, True, doc["seed"])'),
        ('collate_fn=arm.collate_fn', 'collate_fn=collate'),
        ('model = arm.attach(build_model(c["model"], stage), spec, stage).to(device)', 'model = build_model(c["model"], stage).to(device)'),
        ('arm.forward(model.eval(), torch.zeros(1, 1, 3, 224, 224, device=device), arm.probe(device))',
         'model.eval()(torch.zeros(1, 1, 3, 224, 224, device=device))'),
        ('record = train_epoch_with(arm, model, loader, opt, sched, scaler, torch.from_numpy(weights), device, amp,\n'
         '                                      steps // 100, counters)',
         'record = train_epoch(model, loader, opt, sched, scaler, torch.from_numpy(weights), device, amp,\n'
         '                                 steps // 100, counters, components=components,\n'
         '                                 enabled="weak_rank_v1" in c["updates"])'),
        ('screen=arm.screen, **arm.provenance(spec), objective=', 'screen="HB122_SHARED_V1", objective='),
        ('ds = arm.dataset(MILWindows(corpus_dir, val_ids, selected_labels, k, False, doc["seed"]), spec, stage, train=False)',
         'ds = MILWindows(corpus_dir, val_ids, selected_labels, k, False, doc["seed"])'),
        ('evaluate_with(arm, model, evaluation, device, amp, corpus, tta=tta)', 'evaluate(model, evaluation, device, amp, corpus, tta=tta)'),
    ]
    assert dump(substituted(hooks.run, pairs)) == dump(inspect.getsource(shared.run))
    assert dump(substituted(hooks.validate_data_with, [
        ('def validate_data_with(arm, spec, root, corpus_dir):', 'def validate_data(spec, root, corpus_dir):'),
        ('c = arm.validate_recipe(spec)', 'c = validate_recipe(spec)')])) == dump(inspect.getsource(shared.validate_data))


def test_train_and_evaluate_are_deployed_functions_with_forward_hook():
    train = substituted(hooks.train_epoch_with, [
        ('def train_epoch_with(arm, model, loader,', 'def train_epoch(model, loader,'),
        ('for x, y, ids, *extra in loader:', 'for x, y, ids in loader:'),
        ('logits = arm.forward(model, x.to(device, non_blocking=True), arm.move(extra, device))',
         'logits = model(x.to(device, non_blocking=True))')])
    assert dump(train) == dump(inspect.getsource(base.train_epoch))
    assert shared.deployed.train_epoch is base.train_epoch
    evaluate = substituted(hooks.evaluate_with, [
        ('def evaluate_with(arm, model, loader,', 'def evaluate(model, loader,'),
        ('for x, _, ids, *extra in loader:', 'for x, _, ids in loader:'),
        ('        extra = arm.move(extra, device)\n', ''),
        ('logits = arm.forward(model, x.to(device), extra)', 'logits = model(x.to(device))'),
        ('logits = arm.forward(model, flipped.to(device), extra)', 'logits = model(flipped.to(device))')])
    assert dump(evaluate) == dump(inspect.getsource(base.evaluate))
    for name in ('MILWindows', 'build_model', 'optimizer_stack', 'restore_epoch', 'save_latest_epoch', 'smoke_pair',
                 'seed_all', 'worker_init', 'record_smoke', 'collate', 'component_map', 'verify_shards', 'flip_inputs'):
        assert getattr(hooks, name) is getattr(shared, name)


def toy(which, n=8):
    shared.seed_all(42)
    model = torch.nn.Sequential(torch.nn.Linear(4, 24), torch.nn.Dropout(.3), torch.nn.Linear(24, 12))
    opt, sched, scaler = shared.optimizer_stack(model, 3, 'cpu', 'none')
    y = (torch.arange(n * 12).reshape(n, 12) % 3 == 0).float()
    ids = [str(i) for i in range(n)]
    x = torch.arange(n * 4).reshape(n, 4).float() / 10
    centers = torch.arange(n, dtype=torch.int64)[:, None]
    loader = [(x, y, ids, centers) if which == 'meta' else (x, y, ids)] * 3
    counters = dict(updates=0, skips=0)
    args = (model, loader, opt, sched, scaler, torch.ones(12), 'cpu', 'none', 0, counters)
    if which == 'deployed':
        record = base.train_epoch(*args)
    else:
        arm = MetaArm() if which == 'meta' else hooks.IDENTITY
        record = hooks.train_epoch_with(arm, *args)
    return (model.state_dict(), opt.state_dict(), sched.state_dict(), counters, record,
            torch.get_rng_state(), np.random.get_state()[1], random.getstate())


class MetaArm(hooks.Arm):
    def move(self, extra, device):
        assert len(extra) == 1
        return extra[0].to(device)

    def forward(self, model, x, extra):
        assert extra.shape == (x.shape[0], 1)
        return model(x)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize('which', ['identity', 'meta'])
def test_hooked_epoch_equals_deployed_epoch_and_rng(which):
    equal_tree(toy('deployed'), toy(which))


def test_identity_arm_rejects_metadata_and_rank_spec():
    with pytest.raises(shared.life.ContractError):
        hooks.IDENTITY.move([torch.zeros(1)], 'cpu')
    with pytest.raises(shared.life.ContractError, match='rank objective'):
        hooks.IDENTITY.validate_recipe(build.spec_for('rank', 42))
    assert hooks.IDENTITY.validate_recipe(build.spec_for('control', 42))['updates'] == \
        shared.life.ArmSpec((ROOT / 'training/configs/hb105-small-s42-f0-v1.json').read_bytes()).document['config']['updates']


@pytest.mark.parametrize('train,k', [(True, 12), (False, 36)])
def test_sampled_centers_match_the_windows_actually_used(monkeypatch, train, k):
    corpus = shared.common.Corpus(build.CORPUS)
    ids = [str(u) for u in corpus.ids.tolist()[200:203]]
    ds = shared.MILWindows(build.CORPUS, ids, {u: np.zeros(12, np.float32) for u in ids}, k, train, 43)
    used, original = [], shared.common.select_centers
    monkeypatch.setattr(shared.common, 'select_centers', lambda *a, **kw: used.append(original(*a, **kw)) or used[-1])
    wrapped = hooks.CenterWindows(ds)
    for epoch in (0, 3):
        wrapped.set_epoch(epoch)
        for i in range(len(ids)):
            x, y, uid, centers = wrapped[i]
            assert np.array_equal(centers.numpy(), np.asarray(used[-2], np.int64))  # [-2]: dataset draw, [-1]: replay
            assert centers.dtype == torch.int64 and centers.shape == (k,) and uid == ids[i]
            x0, _, _ = ds[i]
            assert torch.equal(x, x0)
    assert wrapped.k == k and wrapped.ids == ids


def test_sampler_replay_equals_runner_loader_and_pins_workers():
    items = [(torch.zeros(1), torch.zeros(12), f'u{i:02d}') for i in range(37)]
    ids = [u for *_, u in items]
    def runner_stream(workers):
        loader = shared.DataLoader(items, batch_size=8, shuffle=True, drop_last=True, num_workers=workers,
                                   persistent_workers=workers > 0, worker_init_fn=shared.worker_init,
                                   generator=torch.Generator().manual_seed(42), collate_fn=shared.collate)
        out = []
        try:
            for epoch in range(3):
                out += [(epoch, batch[2]) for batch in loader]
        finally:
            if getattr(loader, '_iterator', None) is not None:
                loader._iterator._shutdown_workers()
        return out
    replay = list(build.sampler_batches(ids, 42, 3, workers=2))
    assert replay == runner_stream(2)
    zero = list(build.sampler_batches(ids, 42, 3, workers=0))
    assert zero == runner_stream(0)
    assert [b for e, b in zero if e == 0] == [b for e, b in replay if e == 0]
    assert [b for e, b in zero if e > 0] != [b for e, b in replay if e > 0]
    assert build.WORKERS == 2


def demo_spec(seed=42):
    return shared.life.ArmSpec((DEMO / f'configs/hbdemo-centers-s{seed}-f0-v1.json').read_bytes())


@pytest.mark.parametrize('seed', [42, 43])
def test_demo_arm_single_difference_binding(seed):
    spec = demo_spec(seed)
    c = demo.ARM.validate_recipe(spec)
    shared.life.validate_inputs(spec, ROOT)
    control = hooks.control_spec(seed)
    changed = copy.deepcopy(c)
    assert changed['windows'].pop('hooks_demo_v1') == demo.ARM.contract(seed)
    assert changed == control.document['config']
    d = spec.document
    assert d['target_environment'] == control.document['target_environment']
    assert d['inputs']['sources'][hooks.control_path(seed)] == control.sha256
    assert set(d['inputs']['sources']) == hooks.expected_sources(demo.ARM, control)


@pytest.mark.parametrize('change', ['contract', 'second', 'target', 'source', 'control', 'name', 'rank', 'seed', 'missing'])
def test_arm_drift_rejected(change):
    doc = demo_spec().document
    c = doc['config']
    if change == 'contract': c['windows']['hooks_demo_v1']['forward'] = 'uses metadata'
    if change == 'second': c['updates']['epochs'] = 21
    if change == 'target': doc['target_environment']['hostname'] += '-other'
    if change == 'source': doc['inputs']['sources']['training/arms/mil_screen_v1.py'] = '0' * 64
    if change == 'control': doc['inputs']['sources'][hooks.control_path(42)] = '0' * 64
    if change == 'name': doc['arm'] = 'HBDEMO_OTHER_S42_F0'
    if change == 'rank': c['updates']['weak_rank_v1'] = shared.ranking.CONTRACT
    if change == 'seed': doc['seed'] = 43
    if change == 'missing': doc['inputs']['sources'].pop('training/arms/mil_screen_demo_arm_v1.py')
    with pytest.raises(shared.life.ContractError):
        demo.ARM.validate_recipe(shared.life.ArmSpec.from_document(doc))


def test_load_arm_requires_bound_module():
    assert hooks.load_arm('training.arms.mil_screen_demo_arm_v1:ARM', demo_spec()) is demo.ARM
    with pytest.raises(shared.life.ContractError, match='bound source'):
        hooks.load_arm('training.arms.mil_screen_demo_arm_v1:ARM', build.spec_for('control', 42))
    with pytest.raises(shared.life.ContractError):
        hooks.load_arm('training.arms.mil_screen_hooks_v1:VERSION', demo_spec())


def outputs_equal(a, b):
    for name in ('oof_best.csv', 'oof_k36.csv', 'oof_k36_flip_tta.csv'):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    ma = torch.load(a / 'model_best.pt', map_location='cpu', weights_only=False)['model']
    mb = torch.load(b / 'model_best.pt', map_location='cpu', weights_only=False)['model']
    assert ma.keys() == mb.keys() and all(torch.equal(ma[k], mb[k]) for k in ma)


def test_identity_arm_real_cpu_smoke_is_byte_identical_to_shared_runner(tmp_path):
    run = WORK / 'control-s42'
    out = tmp_path / 'identity'
    command = [sys.executable, '-B', '-m', 'training.runners.mil_screen_hooks_v1', 'smoke',
               '--arm', 'training.arms.mil_screen_hooks_v1:IDENTITY', '--spec', str(build.spec_path('control', 42)),
               '--stage', str(run / 'stage'), '--corpus', str(build.CORPUS), '--device', 'cpu',
               '--workers', str(build.WORKERS), '--out', str(out)]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
    outputs_equal(run / 'cpu/outputs', out / 'outputs')
    assert json.loads((out / 'outputs/provenance.json').read_text())['screen'] == 'HB122_SHARED_V1'


@pytest.mark.parametrize('seed', [42, 43])
def test_recorded_demo_metadata_arm_cpu_smoke_byte_identical_to_control(seed):
    outputs_equal(WORK / f'control-s{seed}/cpu/outputs', DEMO / f's{seed}/cpu/outputs')
    card = json.loads((DEMO / f's{seed}/cpu-card.json').read_text())
    assert card['status'] == 'CPU_CARD_PASS' and card['admission'] == 'HOLD'
    assert armbuild.admit(DEMO, seed)['status'] == 'HOLD'
    assert (DEMO / f'panel-smoke/arm-s{seed}.csv').read_bytes() == (DEMO / f'panel-smoke/control-s{seed}.csv').read_bytes()


@pytest.mark.parametrize('member', ['control-s42', 'control-s43', 'rank-s42', 'rank-s43'])
def test_recorded_hb122_cpu_cards(member):
    card = json.loads((WORK / member / 'cpu-card.json').read_text())
    s = card['summary']
    assert card['status'] == 'CPU_CARD_PASS' and card['admission'] == 'HOLD'
    assert s['results']['optimizer_steps'] == 1 and s['cpu_cuda_initialized'] is False
    prov = json.loads((WORK / member / 'cpu/outputs/provenance.json').read_text())
    assert prov['identity']['workers'] == build.WORKERS and prov['screen'] == 'HB122_SHARED_V1'
    kind, seed = member.split('-s')
    assert build.admit(kind, int(seed))['status'] == 'HOLD'
    history = json.loads((WORK / member / 'cpu/outputs/history.json').read_text())
    assert ('pair_coverage' in history[0]) == (kind == 'rank')
    if kind == 'rank':
        assert all(v['batches_with_pairs'] == 1 for v in history[0]['pair_coverage'].values())


def test_recorded_coverage_and_engineering_screens():
    cov = json.loads((WORK / 'pair-coverage-v1.json').read_text())
    for seed in ('42', '43'):
        assert cov[seed]['batches'] == 8620 and cov[seed]['workers'] == 2 and cov[seed]['persistent_workers']
        for t in ('3', '5', '6', '8'):
            v = cov[seed]['targets'][t]
            assert v['batches_with_pairs'] > 0 and v['max_pairs_in_batch'] <= 28
    for path in (WORK / 'screen-cpu-smoke-v1.json', DEMO / 'screen-cpu-smoke-v1.json'):
        r = json.loads(path.read_text())
        assert r['status'] == 'ENGINEERING_CONTROL_REPLAY' and set(r['lenses']) == {f'L{i}' for i in range(1, 8)}
        assert all(r['lenses'][f'L{i}']['value'] == 0 for i in range(1, 7))
        assert r['release_scope'] == 'none' and not r['lenses']['L7']['eligible']


def test_arm_commands_route_every_gpu_step_through_gpu_job():
    cmds = armbuild.commands(DEMO)['commands']
    for seed in ('42', '43'):
        gpu = [c for c in cmds[seed] if 'tools/gpu_job.py' in c]
        assert len(gpu) == 4 and all(c[c.index('--') + 1] == armbuild.PY for c in gpu)
        assert any('train' in c and '--note' in c for c in gpu)
        assert all('--workers' not in c or c[c.index('--workers') + 1] == '2' for c in cmds[seed])
