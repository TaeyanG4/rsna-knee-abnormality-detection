"""CPU-only checks for EXP112 (H-B-33): the B8 CoAtNet recipe on all 4,343 G1 studies, 15 epochs, fixed SWA of epochs 12/14/15
(tools/exp112_full_swa.py, training/runners/exp112_full_swa.py, artifacts/experiments/EXP112_FULL_SWA_PROTOCOL_V1).

No GPU and no CUDA context: the end-to-end smoke runs the real runner in a subprocess on CPU against a scratch registration (outside
artifacts/) whose model is the tiny BatchNorm stand-in, so the full training -> per-epoch checkpoint -> SWA -> BatchNorm pass -> release
checkpoint path is exercised in seconds. Tests marked `prepared` check the registered protocol and its queue entry.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'src'))
from tools import exp112_full_swa as e  # noqa: E402
from tools import exp101_kfold_arch as b  # noqa: E402
from tools import exp106_b8_e16 as e106  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402

REAL = ROOT / e.PROTOCOL / 'manifest.json'
QUEUE = ROOT / e.QUEUE_FILE
EXP099 = ROOT / e.FOLD_PROTOCOL / 'manifest.json'
CHAIN = ROOT / 'artifacts' / 'experiments' / 'LOCAL_GPU_CHAIN_V13.json'
F1_CHECKPOINT = ROOT / 'artifacts' / 'experiments' / 'EXP099_B8_DREAD_F1' / 'model_best.pt'
RUNNER = ROOT / e.RUNNER
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP112 protocol not prepared')
B = 'e4ea3ae625eacde8e3ab563b6c663d715600a0cea039e1c9ee9d201367d5b38e'


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def runner():
    return load_path('exp112_runner', RUNNER)


@pytest.fixture()
def guard_env(monkeypatch):
    for k, v in {**e.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    return monkeypatch


def parsed(runner, argv):
    return runner.build_parser().parse_args(argv[2:])


def _mutate(argv, flag, value=None):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag] + ([str(value)] if value is not None else [])
    return argv


def fold_lists():
    fp = e.read(EXP099)
    trains, vals = {}, {}
    for k, f in fp['folds'].items():
        o = e.read(ROOT / f['ordered_ids']['path'])
        trains[k], vals[k] = o['train'], o['val']
    return fp, trains, vals


# ------------------------------------------------------------------ names, recipe, contract
def test_names_and_constants():
    assert e.TAG == 'EXP112_B8_FULL_SWA' and e.tag_for() == e.TAG and e.tag_for('smoke') == e.TAG + '_SMOKE'
    assert e.PROTOCOL == 'artifacts/experiments/EXP112_FULL_SWA_PROTOCOL_V1' and e.QUEUE_FILE == e.PROTOCOL + '/queue-entry.json'
    assert e.QUEUE_AFTER == 'EXP099_B8_FLIGHT_V4_F4' and e.SWA_EPOCHS == (12, 14, 15) and e.EPOCHS == 15 and e.SEED == 42
    assert e.SCHEMA not in (b.SCHEMA, b.FOLD_SCHEMA, e106.SCHEMA) and e.RUNNER not in (b.RUNNER, e106.RUNNER)
    assert (e.FOLD_MANIFEST_SHA256, e.V4_MANIFEST_SHA256, e.B_INITIAL_STATE, e.BASE_RECIPE) == \
           (e106.FOLD_MANIFEST_SHA256, e106.V4_MANIFEST_SHA256, e106.B_INITIAL_STATE, e106.BASE_RECIPE)
    assert e.epoch_file(12) == 'epoch_12.pt' and e.epoch_file(1) == 'epoch_01.pt'
    with pytest.raises(ValueError):
        e.epoch_file(0)
    with pytest.raises(ValueError):
        e.tag_for('pilot')


def test_recipe_is_b8_with_only_the_epoch_count_changed():
    common8 = e.read(EXP099)['common_args']
    assert b.recipe_args(common8) == e.BASE_RECIPE and common8['workers'] == e.FULL_WORKERS == 12
    full = e.full_recipe(common8)
    assert full == e.EXPECTED_RECIPE and {k for k in full if full[k] != e.BASE_RECIPE[k]} == {'epochs', 'fixed_epoch'}
    assert (full['arch'], full['res'], full['k'], full['bs'], full['grad_ckpt'], full['amp'], full['amp_init_scale'], full['window_policy']) == \
           ('coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', 224, 12, 8, True, 'fp16', 1024.0, 'within-slot-valid')
    assert (full['bb_lr'], full['head_lr'], full['wd'], full['runtime_policy'], full['epochs']) == (3e-5, 1e-3, 0.02, 'strict', 15)


@pytest.mark.parametrize('key, value', [('bb_lr', 5e-5), ('arch', 'resnet34.a1_in1k'), ('k', 16), ('grad_ckpt', False), ('amp', 'bf16')])
def test_full_recipe_refuses_a_drifted_b8_source(key, value):
    with pytest.raises(ValueError):
        e.full_recipe({**e.read(EXP099)['common_args'], key: value})


def test_contract_steps_and_skip_gate():
    c = e.full_contract(4343, e.EXPECTED_RECIPE)
    assert c == dict(len_train_loader=542, steps=8130, scheduler_total_steps=8130, max_fp16_skips=81)
    assert 'eval_batch' not in c  # nothing is evaluated


def test_swa_epoch_rules():
    assert e.check_swa_epochs((12, 14, 15), 15) == (12, 14, 15) and e.parse_epochs('12,14,15') == (12, 14, 15)
    assert e.SWA['epochs_1based'] == [12, 14, 15] and e.SWA['loop_indices_0based'] == [11, 13, 14] and '1-based' in e.SWA['convention']
    for bad in [(), (12, 12, 15), (14, 12, 15), (0, 1), (12, 14, 16)]:
        with pytest.raises(ValueError):
            e.check_swa_epochs(bad, 15)
    assert e.BN_DRAW_EPOCH == 15  # the 0-based window draw after the last training epoch (loop index 14)


# ------------------------------------------------------------------ training set: union of the EXP099 fold train lists, no Gold
def test_training_set_is_the_union_of_the_fold_train_lists_without_gold():
    fp, trains, vals = fold_lists()
    canonical = k99.canonical_order(e.read(ROOT / fp['paths']['base_ordered_ids']))
    union = set().union(*map(set, trains.values()))
    assert len(trains) == 5 and len(union) == 4343 == len(canonical)
    full = e.full_training_set(canonical, trains, vals)
    assert full == canonical and len(set(full)) == 4343
    gold = k99.gold_uids(ROOT)
    assert len(gold) == 58 and not set(full) & gold and e.check_no_gold(full, gold)
    assert not set(full) & set(fp['excluded_uids'])
    with pytest.raises(ValueError, match='Gold'):
        e.check_no_gold(full, {full[5]})


def test_training_set_rejects_a_broken_partition():
    _, trains, vals = fold_lists()
    canonical = list(dict.fromkeys(u for ids in vals.values() for u in ids))
    short = {k: v[:-1] if k == '1' else v for k, v in trains.items()}
    with pytest.raises(ValueError):
        e.full_training_set(canonical + ['extra'], trains, vals)
    with pytest.raises(ValueError):
        e.full_training_set(canonical, short, vals)
    with pytest.raises(ValueError):
        e.full_training_set(canonical + canonical[:1], trains, vals)


def test_full_set_pos_weight_rule():
    fp, trains, vals = fold_lists()
    canonical = k99.canonical_order(e.read(ROOT / fp['paths']['base_ordered_ids']))
    labels, _ = k99._load_labels(ROOT / e.DREAD)
    pw = e.pos_weight(labels, canonical)
    prev = np.clip(np.stack([labels[u] for u in canonical]).mean(0), 0.03, 0.7)
    assert pw.dtype == np.float32 and np.array_equal(pw, np.clip((1 - prev) / prev, 1, 10).astype(np.float32))
    assert np.array_equal(e.pos_weight(labels, trains['0']), np.asarray(fp['folds']['0']['pos_weight'], np.float32))  # same rule = fold-0 vector
    assert not np.array_equal(pw, np.asarray(fp['folds']['0']['pos_weight'], np.float32))
    if REAL.exists():
        assert np.array_equal(pw, np.asarray(e.read(REAL)['pos_weight'], np.float32))


# ------------------------------------------------------------------ SWA averaging and BatchNorm recompute on toy models
def _toy(seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3))
    with torch.no_grad():
        m[1].running_mean.uniform_(-1, 1); m[1].running_var.uniform_(0.5, 2); m[1].num_batches_tracked.fill_(seed)
    return m


def test_swa_average_math_on_a_toy_model():
    import torch
    states = [{k: v.clone() for k, v in _toy(s).state_dict().items()} for s in (1, 2, 3)]
    avg = e.swa_average(states)
    assert list(avg) == list(states[-1])
    for k, v in avg.items():
        if v.is_floating_point():
            expect = (states[0][k].double() + states[1][k].double() + states[2][k].double()) / 3
            assert v.dtype == torch.float32 and torch.equal(v, expect.float()), k
            assert torch.allclose(v, (states[0][k] + states[1][k] + states[2][k]) / 3, atol=1e-6)
    assert int(avg['1.num_batches_tracked']) == 3  # the last member's (BatchNorm recompute resets it)
    same = e.swa_average([states[0]] * 3)
    assert all(torch.equal(same[k], states[0][k]) for k in same)  # mean of identical members is exact (float64 sum)
    # torch's own running average (AveragedModel) agrees up to float32 rounding
    from torch.optim.swa_utils import AveragedModel
    models = [_toy(s) for s in (1, 2, 3)]
    swa = AveragedModel(models[0])
    for m in models:  # the first update copies, later ones average
        swa.update_parameters(m)
    for name, p in swa.module.named_parameters():
        assert torch.allclose(p, avg[name], atol=1e-6), name


def test_swa_average_refuses_inconsistent_members():
    import torch
    a, c = _toy(1).state_dict(), _toy(2).state_dict()
    with pytest.raises(ValueError, match='keys'):
        e.swa_average([a, {k: v for k, v in c.items() if k != '0.bias'}])
    with pytest.raises(ValueError, match='shape'):
        e.swa_average([a, {**c, '0.bias': torch.zeros(5)}])
    with pytest.raises(ValueError, match='integer'):
        e.swa_average([{**a, 'idx': torch.arange(3)}, {**c, 'idx': torch.arange(3) + 1}])
    with pytest.raises(ValueError):
        e.swa_average([])


def _bn_loader(n_batches=3, bs=4):
    import torch
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(bs, 1, 3, 8, 8, generator=g) * (i + 1) + i, torch.zeros(bs, 12), [f'u{i}_{j}' for j in range(bs)]) for i in range(n_batches)]


def test_recompute_bn_calls_update_bn_and_gives_the_cumulative_batch_mean():
    import copy
    import torch
    torch.manual_seed(0)
    model = e.tiny_smoke_model()
    with torch.no_grad():
        model.backbone.bn1.running_mean.fill_(5.0); model.backbone.bn1.num_batches_tracked.fill_(999)
    model.eval()
    reference = copy.deepcopy(model)
    loader = _bn_loader()
    calls = []

    def spy(inputs, net, device=None):
        calls.append(net)
        return torch.optim.swa_utils.update_bn(inputs, net, device)

    rec = e.recompute_bn(model, loader, 'cpu', update_bn=spy)
    assert len(calls) == 1 and calls[0] is model
    assert rec['batches'] == 3 and rec['studies'] == 12 and rec['unique_studies'] == 12 and rec['bn_modules'] == 2 and rec['num_batches_tracked'] == [3]
    assert rec['batch_sizes'] == [4] and rec['mode'] == 'train' and not model.training  # update_bn restores the mode
    # the same as torch's update_bn applied directly
    torch.optim.swa_utils.update_bn((x for x, _, _ in loader), reference)
    for k, v in model.state_dict().items():
        assert torch.equal(v, reference.state_dict()[k]), k
    # momentum None = arithmetic mean of the per-batch means (first BatchNorm sees the stem output)
    with torch.no_grad():
        means = [model.backbone.stem(x.flatten(0, 1)).mean((0, 2, 3)) for x, _, _ in loader]
    assert torch.allclose(model.backbone.bn1.running_mean, torch.stack(means).mean(0), atol=1e-5)
    assert int(model.backbone.bn1.num_batches_tracked) == 3 and model.backbone.bn1.momentum == 0.1


def test_recompute_bn_default_is_torch_update_bn_and_requires_batchnorm():
    import torch
    rec = e.recompute_bn(e.tiny_smoke_model(), _bn_loader(2), 'cpu')
    assert rec['function'] == 'torch.optim.swa_utils.update_bn' and rec['batches'] == 2
    with pytest.raises(ValueError, match='no BatchNorm'):
        e.recompute_bn(torch.nn.Linear(2, 2), _bn_loader(1), 'cpu')


def test_tiny_model_has_the_runner_interface():
    import torch
    m = e.tiny_smoke_model()
    assert m(torch.zeros(2, 3, 3, 224, 224)).shape == (2, 12) and m.backbone.num_features == 16
    m.backbone.set_grad_checkpointing(True)
    assert m.backbone.grad_checkpointing and any(n.startswith('backbone.') for n, _ in m.named_parameters())
    assert [type(x).__name__ for _, x in e.batchnorm_modules(m)] == ['BatchNorm2d', 'BatchNormAct2d']


# ------------------------------------------------------------------ release checkpoint format
def _payload():
    from raptor_common import LAB
    prov = dict(code_sha256={n: 'x' * 64 for n in e.CODE_NAMES})
    meta = dict(arch=e.EXPECTED_RECIPE['arch'], res=224, lab=LAB, contract='raptor64_336_c140_s02', k_eval=24, norm='imagenet',
                window_policy='within-slot-valid', tag=e.TAG, amp_train='fp16', amp_eval='fp16', fold=None, label_set='dread', provenance=prov)
    return e.release_payload(e.tiny_smoke_model().state_dict(), meta, 14, 'swa', swa_over=[12, 14, 15]), LAB


def test_release_payload_has_the_exp099_checkpoint_keys():
    ck, lab = _payload()
    assert tuple(ck)[:len(e.RELEASE_KEYS)] == e.RELEASE_KEYS and ck['epoch'] == 14 and ck['checkpoint_kind'] == 'swa'
    assert e.release_check(ck, lab)
    for broken in ({**ck, 'lab': lab[::-1]}, {k: v for k, v in ck.items() if k != 'norm'}, {**ck, 'k_eval': 24.0},
                   {**ck, 'provenance': {'code_sha256': {}}}):
        with pytest.raises(ValueError):
            e.release_check(broken, lab)


@pytest.mark.skipif(not F1_CHECKPOINT.exists(), reason='EXP099_B8_DREAD_F1 checkpoint absent')
def test_release_keys_equal_the_exp099_f1_checkpoint():
    import torch
    ck = torch.load(F1_CHECKPOINT, map_location='cpu', weights_only=False)
    assert tuple(ck) == e.RELEASE_KEYS
    assert set(e.RELEASE_REQUIRED) <= set(ck) and set(e.RELEASE_CODE) <= set(ck['provenance']['code_sha256'])
    assert (ck['arch'], ck['res'], ck['k_eval'], ck['norm'], ck['window_policy'], ck['amp_train']) == \
           (e.EXPECTED_RECIPE['arch'], 224, 24, 'imagenet', 'within-slot-valid', 'fp16')


def test_release_tools_read_the_payload(tmp_path):
    import torch
    ck, lab = _payload()
    path = tmp_path / 'model_best.pt'
    torch.save(ck, path)
    v6 = load_path('exp111_release_v6_for_exp112', ROOT / 'tools' / 'exp111_release_v6.py')
    v5 = load_path('exp104_release_v5_for_exp112', ROOT / 'tools' / 'exp104_release_v5.py')
    assert v6.REQUIRED_META == list(e.RELEASE_REQUIRED) == v5.REQUIRED_META and v6.TRAINING_CODE_BOUND == list(e.RELEASE_CODE)
    meta, code, _ = v6.checkpoint_meta(path)
    assert meta['arch'] == e.EXPECTED_RECIPE['arch'] and meta['epoch'] == 14 and meta['lab'] == lab and set(e.RELEASE_CODE) <= set(code)
    meta5, code5 = v5.checkpoint_meta(path)
    assert meta5['k_eval'] == 24 and meta5['window_policy'] == 'within-slot-valid' and set(e.RELEASE_CODE) <= set(code5)


# ------------------------------------------------------------------ no validation / no selection code path
def test_runner_has_no_validation_evaluation_or_selection_path(runner):
    src = RUNNER.read_text(encoding='utf-8')
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} \
        | {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} \
        | {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    forbidden = {'evaluate', 'write_predictions', 'macro_auc', 'roc_auc_score', 'gold_labels', 'load_folds', 'eval_labels', 'oof_ids', 'val_ids',
                 'validate_fold_membership', 'macro12', 'auc', 'best_score', 'best_auc', 'gold_auc', 'select', 'argmax', 'eval_runtime', 'eval_dtype'}
    assert not names & forbidden, names & forbidden
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not [s for s in strings if s.endswith('.csv') or 'oof' in s.lower() or 'gold_auc' in s]
    dests = {a.dest for a in runner.build_parser()._actions}
    assert not dests & {'fold', 'eval_labels', 'k_extra', 'select', 'gold'} and 'swa_epochs' in dests
    # the Gold blocker is installed right after argument validation, before the training imports
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n.func.attr for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert calls.index('validate_arguments') < calls.index('install_gold_blocker')
    assert 'recompute_bn' in calls and 'swa_average' in calls and 'release_check' in calls


def test_tool_has_no_full_data_gold_fallback():
    src = (ROOT / e.TOOL).read_text(encoding='utf-8')
    assert 'gold_labels' not in src and 'fold -1' not in src and 'roc_auc' not in src
    assert e.SWA['selection'].startswith('fixed before training') and e.recipe_record(e.EXPECTED_RECIPE, [1.0] * 12, 4343)[0]['validation'] == 'none'


# ------------------------------------------------------------------ GPU time estimate
def test_estimate_from_the_reference_runs():
    est = e.estimate_from([0.75, 0.76, 1.0], 8130, 542, overhead=150)
    assert est['seconds'] == round(0.76 * 8130 + 0.76 * 542 + 150) and est['low_seconds'] < est['seconds'] < est['high_seconds']
    real = e.estimate(ROOT)
    assert set(real['reference_runs']) == {Path(r).name for r in e.REFERENCE_RUNS} and real['steps'] == 8130 and real['len_train_loader'] == 542
    assert all(0.7 < r['seconds_per_update'] < 1.1 for r in real['reference_runs'].values())
    assert 1.6 < real['hours'] < 2.2 and real['low_hours'] < real['hours'] < real['high_hours'] < 2.7


# ------------------------------------------------------------------ scratch registration + end-to-end CPU smoke (tiny model)
@pytest.fixture(scope='module')
def scratch(tmp_path_factory):
    import os
    saved = {k: os.environ.get(k) for k in [*e.THREAD_ENV, 'HF_HUB_OFFLINE']}
    os.environ.update({**e.THREAD_ENV, 'HF_HUB_OFFLINE': '1'})
    try:
        dest = tmp_path_factory.mktemp('exp112') / 'registration'
        manifest = e.prepare(destination=dest, tiny=True)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return dest / 'manifest.json', manifest


def test_prepare_refuses_scratch_inside_artifacts_and_tiny_for_the_registered_protocol():
    with pytest.raises(SystemExit, match='outside artifacts'):
        e.prepare(destination=ROOT / 'artifacts' / 'experiments' / 'EXP112_SCRATCH_SHOULD_NOT_EXIST')
    assert not (ROOT / 'artifacts' / 'experiments' / 'EXP112_SCRATCH_SHOULD_NOT_EXIST').exists()
    with pytest.raises(ValueError, match='real CoAtNet'):
        e.prepare(tiny=True)


def test_scratch_registration_contents(scratch):
    path, p = scratch
    assert json.loads(json.dumps(p)) == e.read(path) and p['registration'] == 'scratch' and 'queue' not in p and p['status'] == 'REGISTERED_NO_EXECUTION'
    assert p['initial_state']['model'] == 'tiny-smoke' and p['initial_state']['bn_modules'] == 2 and not p['initial_state']['equals_B']
    assert p['training_set']['n'] == 4343 and p['training_set']['gold'] == dict(source=p['training_set']['gold']['source'], n=58, overlap=0,
                                                                                    sorted_uid_sha256=p['training_set']['gold']['sorted_uid_sha256'])
    ids = e.read(path.parent / e.TRAIN_IDS_FILE)['train']
    fp, trains, _ = fold_lists()
    assert ids == k99.canonical_order(e.read(ROOT / fp['paths']['base_ordered_ids'])) and set(ids) == set().union(*map(set, trains.values()))
    assert e.uid_hash(ids) == p['training_set']['uid_sha256'] and p['training_set']['each_study_in_fold_train_lists'] == 4
    assert p['contract'] == dict(len_train_loader=542, steps=8130, scheduler_total_steps=8130, max_fp16_skips=81)
    assert p['swa']['epochs_1based'] == [12, 14, 15] and p['bn_recompute']['expected_batches'] == 542 and p['bn_recompute']['expected_studies'] == 4336
    assert p['recipe_sha256'] == e.canonical_json_sha256(p['recipe']) and p['recipe']['checkpoint_selection'].startswith('none')
    assert p['common_args'] == e.EXPECTED_RECIPE and p['epochs'] == 15 and len(p['pos_weight']) == 12


def test_scratch_registration_is_smoke_only(scratch, runner, guard_env, tmp_path):
    path, p = scratch
    smoke = e.runner_argv(p, 'smoke', output_root=tmp_path, protocol_path=path, workers=0)
    got, ids, out_root, plan = e.validate_arguments(parsed(runner, smoke), ROOT)
    assert out_root == tmp_path.resolve() and plan == dict(epochs=3, swa_epochs=(1, 2, 3), smoke=True) and len(ids) == 4343
    full = e.runner_argv(p, 'full', protocol_path=path)
    with pytest.raises(ValueError, match='wrong protocol'):
        e.validate_arguments(parsed(runner, full), ROOT)
    for flag, value, message in [('--smoke-swa-epochs', '1,4', 'SWA epochs'), ('--smoke-train', '4', 'one batch'),
                                 ('--output-root', str(ROOT / 'artifacts' / 'x'), 'outside artifacts'), ('--swa-epochs', '13,14,15', 'SWA epochs'),
                                 ('--epochs', '16', 'frozen argument changed: epochs'), ('--seed', '43', 'seed changed'), ('--k', '16', 'frozen argument')]:
        with pytest.raises(ValueError, match=message):
            e.validate_arguments(parsed(runner, _mutate(smoke, flag, value)), ROOT)
    guard_env.setenv('KNEE_REMOTE_RELAX_ENV', '1')
    with pytest.raises(ValueError, match='KNEE_REMOTE_RELAX_ENV'):
        e.validate_arguments(parsed(runner, smoke), ROOT)


def test_cpu_smoke_trains_checkpoints_averages_and_recomputes_bn(scratch, tmp_path):
    import torch
    path, p = scratch
    out = tmp_path / 'smoke'
    argv = e.runner_argv(p, 'smoke', python=sys.executable, output_root=out, protocol_path=path, workers=0)
    r = subprocess.run(argv, cwd=ROOT, env=e.local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    run = out / e.tag_for('smoke')
    assert sorted(x.name for x in run.iterdir()) == sorted(['epochs', 'history.json', 'model_best.pt', 'model_final.pt', 'output-inventory.json',
                                                           'provenance.json', 'summary.json', 'swa.json', 'train.log'])
    assert not list(run.rglob('*.csv'))
    s, swa, prov = e.read(run / 'summary.json'), e.read(run / 'swa.json'), e.read(run / 'provenance.json')
    assert s['optimizer_updates'] == 3 and s['skipped_updates'] == 0 and s['train'] == 8 and s['val'] == 0 and len(s['history']) == 3
    assert s['validation'] == 'none' and s['metric_access'] == 'none' and s['gold_access'] == 'disabled' and 'weak_oof_macro_auc' not in s
    assert prov['cuda_initialized'] is False and prov['initial_state_matches_registered'] is True and prov['train_ids_ordered_sha256'] == \
        e.uid_hash(e.read(path.parent / e.TRAIN_IDS_FILE)['train'][:8])
    # per-epoch checkpoints with hashes
    inv = e.read(run / 'epochs' / e.INVENTORY)
    assert list(inv) == ['epoch_01.pt', 'epoch_02.pt', 'epoch_03.pt']
    epochs = {}
    for name, rec in inv.items():
        assert e.sha256(run / 'epochs' / name) == rec['sha256'] and rec['optimizer_updates'] == rec['epoch_1based']
        ck = torch.load(run / 'epochs' / name, map_location='cpu', weights_only=False)
        assert ck['epoch_1based'] == rec['epoch_1based'] and ck['epoch'] == rec['epoch_1based'] - 1 and ck['checkpoint_kind'] == 'epoch'
        assert e.state_dict_sha256(ck['model']) == rec['state_sha256'] == ck['state_sha256']
        epochs[rec['epoch_1based']] = ck['model']
    assert len({e.state_dict_sha256(v) for v in epochs.values()}) == 3  # every epoch updated the weights
    # model_final = plain last epoch; model_best = SWA mean of the registered smoke epochs + recomputed BatchNorm statistics
    final = torch.load(run / 'model_final.pt', map_location='cpu', weights_only=False)
    best = torch.load(run / 'model_best.pt', map_location='cpu', weights_only=False)
    assert final['checkpoint_kind'] == 'final' and all(torch.equal(final['model'][k], epochs[3][k]) for k in epochs[3])
    from raptor_common import LAB
    assert best['checkpoint_kind'] == 'swa' and tuple(best)[:len(e.RELEASE_KEYS)] == e.RELEASE_KEYS and e.release_check(best, LAB)
    assert best['swa_over'] == [1, 2, 3] and best['epoch'] == 2 and best['tag'] == e.tag_for('smoke') and best['fold'] is None
    mean = e.swa_average([epochs[1], epochs[2], epochs[3]])
    bn = {n + '.' + x for n, _ in e.batchnorm_modules(e.tiny_smoke_model()) for x in ('running_mean', 'running_var', 'num_batches_tracked')}
    for k, v in best['model'].items():
        if k in bn:
            continue
        assert torch.equal(v, mean[k]), k  # parameters are exactly the SWA mean
    assert not torch.equal(best['model']['backbone.bn1.running_mean'], mean['backbone.bn1.running_mean'])  # BatchNorm statistics recomputed
    assert int(best['model']['backbone.bn1.num_batches_tracked']) == 1 and int(epochs[3]['backbone.bn1.num_batches_tracked']) == 3
    assert swa['epochs_1based'] == [1, 2, 3] and [m['epoch_1based'] for m in swa['members']] == [1, 2, 3]
    assert swa['mean_state_sha256'] == e.state_dict_sha256(mean) and swa['final_state_sha256'] == e.state_dict_sha256(best['model'])
    bnr = swa['bn_recompute']
    assert bnr['function'] == 'torch.optim.swa_utils.update_bn' and bnr['batches'] == 1 and bnr['studies'] == 8 and bnr['num_batches_tracked'] == [1]
    assert bnr['bn_modules'] == 2 and bnr['window_draw_epoch_0based'] == 3 and bnr['mode'] == 'train' and bnr['gradients'] == 'none'
    assert e.read(run / e.INVENTORY)['model_best.pt'] == dict(sha256=e.sha256(run / 'model_best.pt'), bytes=(run / 'model_best.pt').stat().st_size)
    v6 = load_path('exp111_release_v6_smoke', ROOT / 'tools' / 'exp111_release_v6.py')
    meta, code, _ = v6.checkpoint_meta(run / 'model_best.pt')
    assert meta['lab'] == LAB and code == prov['code_sha256'] and v6.inventory_check(run / 'model_best.pt', e.sha256(run / 'model_best.pt'))


# ------------------------------------------------------------------ the registered protocol and its queue entry
@prepared
def test_registered_manifest(guard_env):
    p, fp = e.read(REAL), e.read(EXP099)
    assert p['schema'] == e.SCHEMA and p['registration'] == 'registered' and p['plan_item'] == 'H-B-33' and p['status'] == 'REGISTERED_NO_EXECUTION'
    assert p['common_args'] == e.EXPECTED_RECIPE and p['seed'] == 42 and p['epochs'] == 15 and p['swa']['epochs_1based'] == [12, 14, 15]
    for key in ('training_code_sha256', 'environment', 'corpus_manifest_sha256', 'excluded_uids'):
        assert p[key] == fp[key], key
    assert p['initial_state_sha256'] == B and p['initial_state']['equals_B'] and p['initial_state']['model'] == 'arch' and p['initial_state']['bn_modules'] == 25
    keys = e.read(REAL.parent / p['initial_state']['keys_file'])
    assert keys['combined_sha256'] == B and len(keys['keys']) == p['initial_state']['keys']
    ids = e.read(REAL.parent / e.TRAIN_IDS_FILE)['train']
    assert ids == k99.canonical_order(e.read(ROOT / fp['paths']['base_ordered_ids'])) and e.uid_hash(ids) == p['training_set']['uid_sha256']
    assert not set(ids) & k99.gold_uids(ROOT) and p['training_set']['gold']['overlap'] == 0 and p['training_set']['gold']['n'] == 58
    labels, semantic = k99._load_labels(ROOT / e.DREAD)
    assert semantic == p['label_set']['semantic_sha256'] and np.array_equal(e.pos_weight(labels, ids), np.asarray(p['pos_weight'], np.float32))
    assert p['contract'] == dict(len_train_loader=542, steps=8130, scheduler_total_steps=8130, max_fp16_skips=81)
    assert p['recipe_sha256'] == e.canonical_json_sha256(p['recipe']) and p['recipe']['common_args'] == p['common_args']
    for path, expected in p['source_bindings'].items():
        assert e.sha256(ROOT / path) == expected, path
    for name, expected in p['protocol_files'].items():
        assert e.sha256(REAL.parent / name) == expected, name
    assert {e.RUNNER, e.TOOL, e.FOLD_PROTOCOL + '/manifest.json', e.DREAD, e.V4_PROTOCOL} <= set(p['source_bindings'])


@prepared
def test_queue_entry_is_the_registered_full_run(guard_env, runner):
    p = e.read(REAL)
    plan = json.loads(QUEUE.read_text(encoding='utf-8'))
    assert isinstance(plan, list) and len(plan) == 1
    job = plan[0]
    assert job == p['queue']['entry'] == e.queue_entry(p) and p['queue']['started'] is False
    assert job['tag'] == e.TAG and job['after'] == 'EXP099_B8_FLIGHT_V4_F4' and job['continue_on_failure'] is True
    chain = json.loads(CHAIN.read_text(encoding='utf-8'))
    assert set(job) == set(chain[-1]) and job['command'][0] == chain[-1]['command'][0] and chain[-1]['tag'] == 'EXP099_B8_FLIGHT_V4_F4'
    assert e.TAG not in [j['tag'] for j in chain]
    cmd = job['command']
    assert cmd[1] == str(RUNNER) and '--validate-only' not in cmd and '--output-root' not in cmd
    args = parsed(runner, cmd)
    assert (args.mode, args.device, args.amp, args.epochs, args.workers, args.swa_epochs, args.seed) == ('full', 'cuda', 'fp16', 15, 12, '12,14,15', 42)
    got, ids, out_root, plan = e.validate_arguments(args, ROOT)
    assert len(ids) == 4343 and out_root == (ROOT / 'artifacts' / 'experiments').resolve() and plan == dict(epochs=15, swa_epochs=(12, 14, 15), smoke=False)


@prepared
@pytest.mark.parametrize('flag, value, message', [
    ('--epochs', '8', 'frozen argument changed: epochs'), ('--fixed-epoch', '8', 'frozen argument changed: fixed_epoch'),
    ('--swa-epochs', '13,14,15', 'SWA epochs'), ('--seed', '43', 'seed changed'), ('--workers', '4', 'B8 workers 12'),
    ('--tag', 'EXP106_B8_E16_F0', 'tag'), ('--tag', 'EXP112_B8_FULL_SWA_SMOKE', 'tag'), ('--bb-lr', '0.0003', 'frozen argument changed: bb_lr'),
    ('--protocol', str(ROOT / e106.PROTOCOL / 'manifest.json'), 'wrong protocol'), ('--device', 'cpu', 'frozen argument changed: device'),
    ('--amp', 'bf16', 'frozen argument changed: amp'), ('--labels', 'storage/labels/other.csv', 'input path differs: labels'),
    ('--output-root', 'C:/elsewhere', 'no --output-root')])
def test_validate_arguments_rejects(guard_env, runner, flag, value, message):
    argv = _mutate(e.runner_argv(e.read(REAL)), flag, value)
    with pytest.raises(ValueError, match=message):
        e.validate_arguments(parsed(runner, argv), ROOT)


@prepared
def test_other_guards_refuse_the_exp112_protocol(guard_env):
    import argparse
    args = argparse.Namespace(protocol=str(REAL), mode='full', fold=0)
    for guard in (b, e106):
        with pytest.raises((ValueError, KeyError)):
            guard.validate_arguments(args, ROOT)


def _fake_run(root, p, **overrides):
    run = root / e.TAG
    (run / 'epochs').mkdir(parents=True)
    epochs = {}
    for k in range(1, p['epochs'] + 1):
        f = run / 'epochs' / e.epoch_file(k); f.write_bytes(b'epoch%d' % k)
        epochs[f.name] = dict(sha256=e.sha256(f), bytes=f.stat().st_size, epoch_1based=k)
    (run / 'epochs' / e.INVENTORY).write_text(json.dumps(epochs), encoding='utf-8')
    prov = dict(seed=42, arch=p['common_args']['arch'], initial_state_sha256=B, initial_state_matches_registered=True, environment_matches_baseline=True,
                requested_workers=12, scheduler_total_steps=8130, max_fp16_skips=81, label_set='dread', training_mode='full', validation='none',
                checkpoint_selection=p['recipe']['checkpoint_selection'], gold_access='disabled', metric_access='none',
                train_ids_ordered_sha256=p['training_set']['uid_sha256'], label_semantic_sha256=p['label_set']['semantic_sha256'],
                protocol_sha256=e.sha256(REAL), recipe_sha256=p['recipe_sha256'], cuda_initialized=True)
    prov.update(overrides.get('prov', {}))
    bn = dict(batches=542, studies=4336, num_batches_tracked=[542], bn_modules=25)
    bn.update(overrides.get('bn', {}))
    swa = dict(epochs_1based=[12, 14, 15], members=[dict(epoch_1based=x) for x in overrides.get('members', [12, 14, 15])], bn_recompute=bn)
    summary = dict(optimizer_updates=8130, skipped_updates=0, train=4343, val=0, history=[{}] * overrides.get('epochs', 15))
    for name, value in (('provenance.json', prov), ('summary.json', summary), ('swa.json', swa), ('history.json', [])):
        (run / name).write_text(json.dumps(value), encoding='utf-8')
    for name in ('model_best.pt', 'model_final.pt'):
        (run / name).write_bytes(name.encode())
    inv = {n: dict(sha256=e.sha256(run / n), bytes=(run / n).stat().st_size)
           for n in ('model_best.pt', 'model_final.pt', 'summary.json', 'provenance.json', 'history.json', 'swa.json', 'epochs/' + e.INVENTORY)}
    (run / e.INVENTORY).write_text(json.dumps(inv), encoding='utf-8')
    if overrides.get('csv'):
        (run / 'oof_best.csv').write_text('x', encoding='utf-8')
    return run


@prepared
@pytest.mark.parametrize('overrides, message', [
    ({}, None), ({'prov': dict(seed=43)}, 'provenance seed'), ({'prov': dict(validation='G1 V888')}, 'provenance validation'),
    ({'members': [11, 14, 15]}, 'SWA members'), ({'bn': dict(batches=10)}, 'BatchNorm pass'), ({'epochs': 8}, 'history'),
    ({'csv': True}, 'prediction files')])
def test_verify_run(tmp_path, overrides, message):
    p = e.read(REAL)
    problems = e.verify_run(_fake_run(tmp_path, p, **overrides), p)
    if message is None:
        assert problems == []
    else:
        assert problems and any(message in x for x in problems), problems
