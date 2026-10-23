"""CPU-only checks for EXP112 V2 (H-B-33): 16 epochs, fixed SWA of 1-based epochs 13/15/16 = public v5 0-based swa_over [14, 12, 15]
(tools/exp112_full_swa_v2.py, training/runners/exp112_full_swa_v2.py, artifacts/experiments/EXP112_FULL_SWA_PROTOCOL_V2).

V1 (tests/test_exp112_full_swa.py) must keep passing and its files must stay byte-identical to the V1 manifest bindings. No GPU, no CUDA:
the end-to-end smoke runs the V2 runner in a subprocess on CPU against a scratch registration with the tiny BatchNorm stand-in model.
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
from tools import exp112_full_swa_v2 as e  # noqa: E402
from tools import exp112_full_swa as v1  # noqa: E402
from tools import exp106_b8_e16 as e106  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402

REAL = ROOT / e.PROTOCOL / 'manifest.json'
V1_REAL = ROOT / v1.PROTOCOL / 'manifest.json'
QUEUE = ROOT / e.QUEUE_FILE
EXP099 = ROOT / e.FOLD_PROTOCOL / 'manifest.json'
CHAIN_V14 = ROOT / 'artifacts' / 'experiments' / 'LOCAL_GPU_CHAIN_V14.json'
RUNNER, RUNNER_V1 = ROOT / e.RUNNER, ROOT / v1.RUNNER
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP112 V2 protocol not prepared')
B = 'e4ea3ae625eacde8e3ab563b6c663d715600a0cea039e1c9ee9d201367d5b38e'


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def runner():
    return load_path('exp112_v2_runner', RUNNER)


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


# ------------------------------------------------------------------ versions and the public epoch mapping
def test_v2_constants_follow_the_public_zero_based_swa_over():
    assert e.PUBLIC_SWA_OVER == (14, 12, 15) and e.public_to_1based(e.PUBLIC_SWA_OVER) == e.SWA_EPOCHS == (13, 15, 16)
    assert e.EPOCHS == 16 and e.BN_DRAW_EPOCH == 16 and e.check_swa_epochs(e.SWA_EPOCHS, e.EPOCHS) == (13, 15, 16)
    assert e.SWA['epochs_1based'] == [13, 15, 16] and e.SWA['loop_indices_0based'] == [12, 14, 15] == sorted(e.PUBLIC_SWA_OVER)
    assert e.EPOCH_MAPPING['epochs_1based'] == [13, 15, 16] and 'review' in e.EPOCH_MAPPING['source'] and 'D-A-050' in e.EPOCH_MAPPING['source']
    assert e.TAG == 'EXP112_B8_FULL_SWA_V2' and e.tag_for('smoke') == e.TAG + '_SMOKE' and e.QUEUE_FILE == e.PROTOCOL + '/queue-entry.json'
    assert e.PROTOCOL == 'artifacts/experiments/EXP112_FULL_SWA_PROTOCOL_V2' and e.QUEUE_AFTER == 'EXP099_B8_FLIGHT_V4_F4'
    assert e.SCHEMA != v1.SCHEMA and e.RUNNER != v1.RUNNER and e.TOOL != v1.TOOL and e.TAG != v1.TAG and e.PROTOCOL != v1.PROTOCOL


def test_versions_table_keeps_v1_selectable():
    assert set(e.VERSIONS) == {'v1', 'v2'}
    a, c = e.version('v1'), e.version('v2')
    assert (a['protocol'], a['tool'], a['runner'], a['tag'], a['epochs'], a['swa_epochs_1based'], a['steps']) == \
           (v1.PROTOCOL, v1.TOOL, v1.RUNNER, v1.TAG, 15, (12, 14, 15), 8130)
    assert (c['protocol'], c['tool'], c['runner'], c['tag'], c['epochs'], c['swa_epochs_1based'], c['steps']) == \
           (e.PROTOCOL, e.TOOL, e.RUNNER, e.TAG, 16, (13, 15, 16), 8672)
    with pytest.raises(ValueError):
        e.version('v3')


@pytest.mark.skipif(not V1_REAL.exists(), reason='EXP112 V1 protocol absent')
def test_v1_files_stay_byte_identical_to_the_v1_bindings():
    m = v1.read(V1_REAL)
    assert v1.sha256(V1_REAL) == e.V1_MANIFEST_SHA256
    for path, expected in m['source_bindings'].items():
        assert v1.sha256(ROOT / path) == expected, path  # includes tools/exp112_full_swa.py and training/runners/exp112_full_swa.py
    assert m['epochs'] == 15 and m['swa']['epochs_1based'] == [12, 14, 15]


def test_recipe_and_contract():
    full = e.full_recipe(e.read(EXP099)['common_args'])
    assert full == e.EXPECTED_RECIPE and {k for k in full if full[k] != e.BASE_RECIPE[k]} == {'epochs', 'fixed_epoch'} and full['epochs'] == 16
    assert {k for k in full if full[k] != v1.EXPECTED_RECIPE[k]} == {'epochs', 'fixed_epoch'}
    assert e.full_contract(4343, e.EXPECTED_RECIPE) == dict(len_train_loader=542, steps=8672, scheduler_total_steps=8672, max_fp16_skips=86)
    record, sha = e.recipe_record(e.EXPECTED_RECIPE, [1.0] * 12, 4343)
    assert record['epoch_mapping'] == e.EPOCH_MAPPING and record['validation'] == 'none' and sha == e.canonical_json_sha256(record)
    with pytest.raises(ValueError):
        e.full_recipe({**e.read(EXP099)['common_args'], 'bb_lr': 5e-5})


def test_v2_reuses_the_v1_helpers():
    for name in ('swa_average', 'recompute_bn', 'release_payload', 'release_check', 'full_training_set', 'check_no_gold', 'validate_training_set',
                 'state_dict_sha256', 'batchnorm_modules', 'tiny_smoke_model', 'build_run_model', 'full_contract', 'estimate_from', 'pos_weight',
                 'worker_init', 'install_gold_blocker', 'state_hashes'):
        assert getattr(e, name) is getattr(v1, name), name
    for name in ('RELEASE_KEYS', 'LOOP', 'BASE_RECIPE', 'B_INITIAL_STATE', 'FOLD_MANIFEST_SHA256', 'RUN_FILES', 'REFERENCE_RUNS'):
        assert getattr(e, name) == getattr(v1, name), name


def test_v2_runner_is_the_v1_runner_except_guard_and_epoch_numbering():
    """Statement-level AST comparison of main(): only the guard import, the swa.json record and the model_best payload differ."""
    def main_body(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        return main.body, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_parser')

    b1, p1 = main_body(RUNNER_V1)
    b2, p2 = main_body(RUNNER)
    assert ast.dump(p1) == ast.dump(p2) and len(b1) == len(b2)
    differ = [(ast.unparse(x).split('\n')[0], ast.unparse(y).split('\n')[0]) for x, y in zip(b1, b2) if ast.dump(x) != ast.dump(y)]
    assert [d[1] for d in differ] == ['from tools import exp112_full_swa_v2 as guard',
                                      differ[1][1], differ[2][1]] and differ[0][0] == 'from tools import exp112_full_swa as guard'
    assert differ[1][1].startswith('swa_record = dict(') and differ[2][1].startswith('best = guard.release_payload(')
    src = RUNNER.read_text(encoding='utf-8')
    assert 'swa_over=[e - 1 for e in swa_epochs]' in src and 'swa_epochs_1based=list(swa_epochs)' in src and 'epoch_1based=epochs' in src


def test_v2_runner_has_no_validation_or_selection_path(runner):
    tree = ast.parse(RUNNER.read_text(encoding='utf-8'))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} \
        | {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not names & {'evaluate', 'write_predictions', 'macro_auc', 'roc_auc_score', 'gold_labels', 'load_folds', 'eval_labels', 'oof_ids',
                        'validate_fold_membership', 'macro12', 'auc', 'best_score', 'gold_auc', 'argmax', 'eval_dtype'}
    dests = {a.dest for a in runner.build_parser()._actions}
    assert not dests & {'fold', 'eval_labels'} and 'swa_epochs' in dests


def test_estimate():
    est = e.estimate(ROOT)
    assert est['steps'] == 8672 and est['len_train_loader'] == 542
    assert est['seconds'] > v1.estimate(ROOT)['seconds'] and 1.8 < est['hours'] < 2.2 and est['high_hours'] < 2.8


# ------------------------------------------------------------------ scratch registration + end-to-end CPU smoke (tiny model)
@pytest.fixture(scope='module')
def scratch(tmp_path_factory):
    import os
    saved = {k: os.environ.get(k) for k in [*e.THREAD_ENV, 'HF_HUB_OFFLINE']}
    os.environ.update({**e.THREAD_ENV, 'HF_HUB_OFFLINE': '1'})
    try:
        dest = tmp_path_factory.mktemp('exp112v2') / 'registration'
        manifest = e.prepare(destination=dest, tiny=True)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return dest / 'manifest.json', manifest


def test_prepare_refusals():
    with pytest.raises(SystemExit, match='outside artifacts'):
        e.prepare(destination=ROOT / 'artifacts' / 'experiments' / 'EXP112_V2_SCRATCH_SHOULD_NOT_EXIST')
    assert not (ROOT / 'artifacts' / 'experiments' / 'EXP112_V2_SCRATCH_SHOULD_NOT_EXIST').exists()
    with pytest.raises(ValueError, match='real CoAtNet'):
        e.prepare(tiny=True)


def test_scratch_registration_contents(scratch):
    path, p = scratch
    assert json.loads(json.dumps(p)) == e.read(path) and p['registration'] == 'scratch' and p['version'] == 'v2' and 'queue' not in p
    assert p['epochs'] == 16 and p['swa']['epochs_1based'] == [13, 15, 16] and p['epoch_mapping']['public_swa_over'] == [14, 12, 15]
    assert p['contract'] == dict(len_train_loader=542, steps=8672, scheduler_total_steps=8672, max_fp16_skips=86)
    assert p['supersedes']['protocol'] == v1.PROTOCOL and p['supersedes']['manifest_sha256'] == e.V1_MANIFEST_SHA256
    assert v1.PROTOCOL + '/manifest.json' in p['source_bindings'] and v1.TOOL in p['source_bindings'] and v1.RUNNER not in p['source_bindings']
    assert p['training_set']['n'] == 4343 and p['training_set']['gold']['overlap'] == 0 and p['bn_recompute']['window_draw_epoch_0based'] == 16
    assert p['recipe_sha256'] == e.canonical_json_sha256(p['recipe']) and p['recipe_sha256'] != v1.read(V1_REAL)['recipe_sha256'] if V1_REAL.exists() else True


def test_guards_refuse_each_others_protocols(scratch, runner, guard_env, tmp_path):
    path, p = scratch
    smoke = e.runner_argv(p, 'smoke', output_root=tmp_path, protocol_path=path, workers=0)
    got, ids, out_root, plan = e.validate_arguments(parsed(runner, smoke), ROOT)
    assert plan == dict(epochs=3, swa_epochs=(1, 2, 3), smoke=True) and len(ids) == 4343
    with pytest.raises(ValueError, match='wrong protocol'):
        e.validate_arguments(parsed(runner, e.runner_argv(p, 'full', protocol_path=path)), ROOT)
    with pytest.raises(ValueError):  # the V1 guard refuses a V2 registration
        v1.validate_arguments(parsed(runner, smoke), ROOT)
    for flag, value, message in [('--swa-epochs', '12,14,15', 'SWA epochs'), ('--epochs', '15', 'frozen argument changed: epochs'),
                                 ('--fixed-epoch', '15', 'frozen argument changed: fixed_epoch'), ('--smoke-swa-epochs', '0,1', 'SWA epochs')]:
        with pytest.raises(ValueError, match=message):
            e.validate_arguments(parsed(runner, _mutate(smoke, flag, value)), ROOT)


def test_cpu_smoke_records_one_epoch_numbering(scratch, tmp_path):
    import torch
    path, p = scratch
    out = tmp_path / 'smoke'
    argv = e.runner_argv(p, 'smoke', python=sys.executable, output_root=out, protocol_path=path, workers=0)
    r = subprocess.run(argv, cwd=ROOT, env=e.local_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    run = out / e.tag_for('smoke')
    assert not list(run.rglob('*.csv')) and list(e.read(run / 'epochs' / e.INVENTORY)) == ['epoch_01.pt', 'epoch_02.pt', 'epoch_03.pt']
    s, swa, prov = e.read(run / 'summary.json'), e.read(run / 'swa.json'), e.read(run / 'provenance.json')
    assert s['optimizer_updates'] == 3 and s['validation'] == 'none' and prov['cuda_initialized'] is False and prov['initial_state_matches_registered']
    best = torch.load(run / 'model_best.pt', map_location='cpu', weights_only=False)
    from raptor_common import LAB
    assert tuple(best)[:len(e.RELEASE_KEYS)] == e.RELEASE_KEYS and e.release_check(best, LAB)
    # review m1: one stated convention -- epoch / swa_over 0-based loop indices, explicit 1-based fields next to them
    assert (best['epoch'], best['epoch_1based'], best['swa_over'], best['swa_epochs_1based']) == (2, 3, [0, 1, 2], [1, 2, 3])
    assert best['epoch_numbering'] == e.EPOCH_NUMBERING and '0-based' in best['epoch_numbering'] and '1-based' in best['epoch_numbering']
    assert (swa['swa_over_0based'], swa['epochs_1based'], swa['epoch_numbering']) == ([0, 1, 2], [1, 2, 3], e.EPOCH_NUMBERING)
    epochs = {}
    for name, rec in e.read(run / 'epochs' / e.INVENTORY).items():
        ck = torch.load(run / 'epochs' / name, map_location='cpu', weights_only=False)
        assert ck['epoch'] == ck['epoch_1based'] - 1 == rec['loop_index']
        epochs[ck['epoch_1based']] = ck['model']
    mean = e.swa_average([epochs[1], epochs[2], epochs[3]])
    bn = {n + '.' + x for n, _ in e.batchnorm_modules(e.tiny_smoke_model()) for x in ('running_mean', 'running_var', 'num_batches_tracked')}
    assert all(torch.equal(best['model'][k], mean[k]) for k in mean if k not in bn)
    assert int(best['model']['backbone.bn1.num_batches_tracked']) == 1 and swa['bn_recompute']['function'] == 'torch.optim.swa_utils.update_bn'
    v6 = load_path('exp111_release_v6_v2smoke', ROOT / 'tools' / 'exp111_release_v6.py')
    meta, code, _ = v6.checkpoint_meta(run / 'model_best.pt')
    assert meta['epoch'] == 2 and meta['lab'] == LAB and v6.inventory_check(run / 'model_best.pt', e.sha256(run / 'model_best.pt'))


# ------------------------------------------------------------------ the registered V2 protocol and its queue entry
@prepared
def test_registered_v2_manifest():
    p, fp = e.read(REAL), e.read(EXP099)
    assert p['schema'] == e.SCHEMA and p['version'] == 'v2' and p['registration'] == 'registered' and p['status'] == 'REGISTERED_NO_EXECUTION'
    assert p['common_args'] == e.EXPECTED_RECIPE and p['epochs'] == 16 and p['swa']['epochs_1based'] == [13, 15, 16]
    assert p['epoch_mapping'] == json.loads(json.dumps(e.EPOCH_MAPPING)) and p['epoch_mapping']['public_swa_over'] == [14, 12, 15]
    assert p['initial_state_sha256'] == B and p['initial_state']['model'] == 'arch' and p['initial_state']['bn_modules'] == 25
    assert p['contract'] == dict(len_train_loader=542, steps=8672, scheduler_total_steps=8672, max_fp16_skips=86)
    ids = e.read(REAL.parent / e.TRAIN_IDS_FILE)['train']
    assert ids == k99.canonical_order(e.read(ROOT / fp['paths']['base_ordered_ids'])) and e.uid_hash(ids) == p['training_set']['uid_sha256']
    assert not set(ids) & k99.gold_uids(ROOT)
    labels, _ = k99._load_labels(ROOT / e.DREAD)
    assert np.array_equal(e.pos_weight(labels, ids), np.asarray(p['pos_weight'], np.float32))
    if V1_REAL.exists():
        p1 = v1.read(V1_REAL)
        assert p['pos_weight'] == p1['pos_weight'] and p['training_set']['uid_sha256'] == p1['training_set']['uid_sha256']
        assert e.read(REAL.parent / e.KEYS_FILE)['keys'] == v1.read(V1_REAL.parent / v1.KEYS_FILE)['keys']
    for path, expected in p['source_bindings'].items():
        assert e.sha256(ROOT / path) == expected, path
    for name, expected in p['protocol_files'].items():
        assert e.sha256(REAL.parent / name) == expected, name


@prepared
def test_v2_queue_entry(guard_env, runner):
    p = e.read(REAL)
    plan = json.loads(QUEUE.read_text(encoding='utf-8'))
    assert isinstance(plan, list) and len(plan) == 1 and plan[0] == p['queue']['entry'] == e.queue_entry(p) and p['queue']['started'] is False
    job = plan[0]
    assert job['tag'] == 'EXP112_B8_FULL_SWA_V2' and job['after'] == 'EXP099_B8_FLIGHT_V4_F4' and job['continue_on_failure'] is True
    args = parsed(runner, job['command'])
    assert (args.mode, args.device, args.amp, args.epochs, args.fixed_epoch, args.workers, args.swa_epochs) == ('full', 'cuda', 'fp16', 16, 16, 12, '13,15,16')
    got, ids, out_root, plan_ = e.validate_arguments(args, ROOT)
    assert plan_ == dict(epochs=16, swa_epochs=(13, 15, 16), smoke=False) and out_root == (ROOT / 'artifacts' / 'experiments').resolve()
    if CHAIN_V14.exists():  # same entry format as the queued V1 job; only version-dependent arguments differ
        chain = json.loads(CHAIN_V14.read_text(encoding='utf-8'))
        assert e.TAG not in [j['tag'] for j in chain]
        old = next((j for j in chain if j['tag'] == v1.TAG), None)
        if old is not None:
            assert set(old) == set(job)
            a, b = old['command'], job['command']
            changed = {a[i - 1] for i in range(1, len(a)) if a[i] != b[i]}
            assert len(a) == len(b) and a[0] == b[0] and changed == {'--protocol', '--tag', '--swa-epochs', '--epochs', '--fixed-epoch'} | ({a[0]} if a[1] != b[1] else set())


@prepared
@pytest.mark.parametrize('flag, value, message', [
    ('--epochs', '15', 'frozen argument changed: epochs'), ('--swa-epochs', '12,14,15', 'SWA epochs'), ('--seed', '43', 'seed changed'),
    ('--tag', 'EXP112_B8_FULL_SWA', 'tag'), ('--protocol', str(V1_REAL), 'wrong protocol'), ('--workers', '4', 'B8 workers 12'),
    ('--output-root', 'C:/elsewhere', 'no --output-root'), ('--device', 'cpu', 'frozen argument changed: device')])
def test_v2_validate_arguments_rejects(guard_env, runner, flag, value, message):
    with pytest.raises(ValueError, match=message):
        e.validate_arguments(parsed(runner, _mutate(e.runner_argv(e.read(REAL)), flag, value)), ROOT)


@prepared
def test_v1_and_other_guards_refuse_the_v2_protocol(guard_env, runner):
    args = parsed(runner, e.runner_argv(e.read(REAL)))
    for guard in (v1, e106):
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
                requested_workers=12, scheduler_total_steps=8672, max_fp16_skips=86, label_set='dread', training_mode='full', validation='none',
                checkpoint_selection=p['recipe']['checkpoint_selection'], gold_access='disabled', metric_access='none',
                train_ids_ordered_sha256=p['training_set']['uid_sha256'], label_semantic_sha256=p['label_set']['semantic_sha256'],
                protocol_sha256=e.sha256(REAL), recipe_sha256=p['recipe_sha256'], cuda_initialized=True)
    members = overrides.get('members', [13, 15, 16])
    swa = dict(epochs_1based=[13, 15, 16], swa_over_0based=overrides.get('swa_over', [12, 14, 15]), members=[dict(epoch_1based=x) for x in members],
               bn_recompute=dict(batches=542, studies=4336, num_batches_tracked=[542], bn_modules=25, window_draw_epoch_0based=16))
    summary = dict(optimizer_updates=overrides.get('updates', 8672), skipped_updates=0, train=4343, val=0, history=[{}] * 16)
    for name, value in (('provenance.json', prov), ('summary.json', summary), ('swa.json', swa), ('history.json', [])):
        (run / name).write_text(json.dumps(value), encoding='utf-8')
    for name in ('model_best.pt', 'model_final.pt'):
        (run / name).write_bytes(name.encode())
    inv = {n: dict(sha256=e.sha256(run / n), bytes=(run / n).stat().st_size)
           for n in ('model_best.pt', 'model_final.pt', 'summary.json', 'provenance.json', 'history.json', 'swa.json', 'epochs/' + e.INVENTORY)}
    (run / e.INVENTORY).write_text(json.dumps(inv), encoding='utf-8')
    return run


@prepared
@pytest.mark.parametrize('overrides, message', [({}, None), ({'members': [12, 14, 15]}, 'SWA members'), ({'swa_over': [13, 15, 16]}, 'SWA members'),
                                                ({'updates': 8130}, 'updates')])
def test_verify_run(tmp_path, overrides, message):
    p = e.read(REAL)
    problems = e.verify_run(_fake_run(tmp_path, p, **overrides), p)
    if message is None:
        assert problems == []
    else:
        assert problems and any(message in x for x in problems), problems
