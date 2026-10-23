"""CPU-only checks for EXP106 (H-B-23): the B8 CoAtNet recipe on EXP099 fold 0 for 16 epochs (tools/exp106_b8_e16.py,
training/runners/exp106_b8_e16.py, artifacts/experiments/EXP106_B8_E16_PROTOCOL_V1).

The protocol must equal the EXP099 fold-0 / EXP094L_B8_SEED42 recipe in every argument, fold file, pos_weight and initial state; only the
epoch count (and therefore the scheduler length and the 1% FP16 skip gate) may differ. No GPU, no training.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp106_b8_e16 as e  # noqa: E402
from tools import exp101_kfold_arch as b  # noqa: E402
from tools import exp101_kfold_rn34_s43 as s43  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402

REAL = ROOT / e.PROTOCOL / 'manifest.json'
QUEUE = ROOT / e.QUEUE_FILE
EXP099 = ROOT / e.FOLD_PROTOCOL / 'manifest.json'
CHAIN_V8 = ROOT / 'artifacts' / 'experiments' / 'LOCAL_GPU_CHAIN_V8.json'
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP106 protocol not prepared')
B = 'e4ea3ae625eacde8e3ab563b6c663d715600a0cea039e1c9ee9d201367d5b38e'


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def restore_guard_name():
    """The EXP106 runner entry re-points `tools.exp101_kfold_arch`; put the real module back after the test."""
    import tools
    saved_module, saved_attr = sys.modules.get('tools.exp101_kfold_arch'), getattr(tools, 'exp101_kfold_arch', None)
    yield
    sys.modules['tools.exp101_kfold_arch'] = saved_module
    tools.exp101_kfold_arch = saved_attr


@pytest.fixture()
def wrapper(restore_guard_name):
    return load_path('exp106_runner_entry', ROOT / e.RUNNER)


@pytest.fixture()
def guard_env(monkeypatch):
    for k, v in {**e.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    return monkeypatch


def parsed(wrapper, argv):
    return wrapper.build_parser().parse_args(argv[2:])


def _mutate(argv, flag, value=None):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag] + ([str(value)] if value is not None else [])
    return argv


# ------------------------------------------------------------------ pure recipe / contract logic
def test_names_are_distinct_and_fold0_only():
    assert e.TAG == 'EXP106_B8_E16_F0' and e.tag_for(0) == e.TAG and e.tag_for(0, 'smoke') == 'EXP106_B8_E16_F0_SMOKE'
    assert e.PROTOCOL == 'artifacts/experiments/EXP106_B8_E16_PROTOCOL_V1' and e.QUEUE_FILE == e.PROTOCOL + '/queue.json'
    assert e.PROTOCOL not in (b.PROTOCOL, s43.PROTOCOL) and e.RUNNER not in (b.RUNNER, s43.RUNNER) and e.SHARED_RUNNER == b.RUNNER
    assert e.SCHEMA not in (b.SCHEMA, b.FOLD_SCHEMA) and e.FOLDS == [0]
    with pytest.raises(ValueError, match='fold 0 only'):
        e.tag_for(1)


def test_recipe_is_b8_with_only_the_epoch_count_changed():
    common8 = b.read(EXP099)['common_args']
    assert b.recipe_args(common8) == e.BASE_RECIPE and common8['workers'] == e.FULL_WORKERS == 12
    e16 = e.e16_recipe(common8)
    assert e16 == e.EXPECTED_RECIPE
    assert {k for k in e16 if e16[k] != e.BASE_RECIPE[k]} == {'epochs', 'fixed_epoch'} and e16['epochs'] == e16['fixed_epoch'] == 16
    assert (e16['arch'], e16['res'], e16['k'], e16['k_eval'], e16['bs'], e16['grad_ckpt'], e16['amp'], e16['amp_init_scale'], e16['eval_amp']) == \
           ('coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', 224, 12, 24, 8, True, 'fp16', 1024.0, 'fp16')
    assert (e16['bb_lr'], e16['head_lr'], e16['wd'], e16['runtime_policy']) == (3e-5, 1e-3, 0.02, 'strict')


@pytest.mark.parametrize('key, value', [('bb_lr', 5e-5), ('arch', 'resnet34.a1_in1k'), ('k', 16), ('grad_ckpt', False), ('amp_init_scale', 2048.0)])
def test_e16_recipe_refuses_a_drifted_b8_source(key, value):
    with pytest.raises(ValueError, match='B8 common_args differ'):
        e.e16_recipe({**b.read(EXP099)['common_args'], key: value})


def test_contract_steps_and_skip_gate():
    c = e.fold_contract(3455, e.EXPECTED_RECIPE)
    assert c == dict(len_train_loader=431, steps=6896, scheduler_total_steps=6896, max_fp16_skips=68, eval_batch=4)
    assert c['steps'] == (3455 // 8) * 16 == 2 * e.fold_contract(3455, e.BASE_RECIPE)['steps']
    assert e.fold_contract(3455, e.BASE_RECIPE)['max_fp16_skips'] == 34


def test_runtime_estimate_from_the_reference_run():
    history = b.read(ROOT / e.BASE_RUN / 'history.json')
    assert len(history) == 8 and sum(h['epoch_seconds'] for h in history) == 2633
    est = e.estimate_seconds(history)
    assert 2 * 2633 < est < 2 * e.REFERENCE_SECONDS + 200 and est == 5392


# ------------------------------------------------------------------ the real registration
@prepared
def test_manifest_equals_exp099_fold0_except_the_epoch_dependent_fields():
    p, fp = e.read(REAL), e.read(EXP099)
    assert p['schema'] == e.SCHEMA and p['plan_item'] == 'H-B-23' and p['seed'] == 42 and p['status'] == 'REGISTERED_NO_EXECUTION'
    assert p['common_args'] == e.EXPECTED_RECIPE and p['recipe']['base_recipe'] == e.BASE_RECIPE and p['recipe']['source_common_args'] == fp['common_args']
    for key in ('training_code_sha256', 'environment', 'corpus_manifest_sha256', 'excluded_uids', 'pos_weight_rule'):
        assert p[key] == fp[key], key
    assert p['label_set'] == dict(name='dread', **fp['label_sets']['dread']) and p['extra_eval'] == fp['extra_eval'] == e.EXTRA_EVAL
    assert {k: p['paths'][k] for k in ('corpus', 'base_ordered_ids', 'base_groups', 'folds')} == {k: fp['paths'][k] for k in ('corpus', 'base_ordered_ids', 'base_groups', 'folds')}
    assert list(p['folds']) == ['0'] and p['total_steps'] == 6896
    new, old = p['folds']['0'], fp['folds']['0']
    shared = set(new) & set(old)
    assert {k for k in shared if new[k] != old[k]} == {'steps', 'scheduler_total_steps', 'max_fp16_skips'}
    assert (new['steps'], new['scheduler_total_steps'], new['max_fp16_skips'], new['len_train_loader'], new['eval_batch']) == (6896, 6896, 68, 431, 4)
    assert (new['n_train'], new['n_val'], new['tag']) == (3455, 888, e.TAG)
    assert new['contract_8_epochs'] == {k: old[k] for k in ('len_train_loader', 'steps', 'scheduler_total_steps', 'max_fp16_skips', 'eval_batch')}
    assert p['fold_protocol']['sha256'] == e.FOLD_MANIFEST_SHA256 == e.sha256(EXP099)
    assert p['queue'] == dict(file=e.QUEUE_FILE, after='EXP094L_B8_SEED43', continue_on_failure=True, python=str(e.GPU_PYTHON),
                              environment=p['queue']['environment'])


@prepared
def test_initial_state_is_b():
    p = e.read(REAL)
    keys = e.read(ROOT / p['initial_state']['keys_file'])
    prov = e.read(ROOT / e.BASE_RUN / 'provenance.json')
    assert p['initial_state_sha256'] == keys['combined_sha256'] == e.B_INITIAL_STATE == B
    assert B == e.read(EXP099)['initial_state_sha256'] == prov['initial_state_sha256'] and prov['initial_state_matches_B'] is True
    assert keys['seed'] == 42 and keys['arch'] == e.EXPECTED_RECIPE['arch'] and len(keys['keys']) == p['initial_state']['keys'] and p['initial_state']['equals_B']


@prepared
def test_pos_weight_is_the_b8_vector():
    p = e.read(REAL)
    labels, _ = k99._load_labels(ROOT / e.DREAD)
    train = e.read(ROOT / p['folds']['0']['ordered_ids']['path'])['train']
    pw = e.pos_weight(labels, train)
    assert pw.dtype == np.float32 and np.array_equal(pw, np.asarray(p['folds']['0']['pos_weight'], np.float32))
    assert np.array_equal(pw, np.asarray(e.read(ROOT / e.BASE_RUN / 'provenance.json')['actual_pos_weight'], np.float32))
    assert np.array_equal(pw, np.asarray(e.read(EXP099)['folds']['0']['pos_weight'], np.float32))


@prepared
def test_bindings_hold_and_cover_the_shared_runner_and_reference_run():
    p = e.read(REAL)
    for path, expected in p['source_bindings'].items():
        assert e.sha256(ROOT / path) == expected, path
    assert {e.SHARED_RUNNER, e.BASE_TOOL, e.RUNNER, e.TOOL, e.FOLD_PROTOCOL + '/manifest.json', e.V4_PROTOCOL, e.FOLD0_REUSE,
            e.BASE_RUN + '/provenance.json', e.BASE_RUN + '/summary.json', e.PROTOCOL + '/' + e.KEYS_FILE} <= set(p['source_bindings'])
    p101 = e.read(ROOT / b.PROTOCOL / 'manifest.json')  # the shared runner/tool are the files the EXP101 protocol registered
    assert p['source_bindings'][b.RUNNER] == p101['source_bindings'][b.RUNNER] and p['source_bindings'][b.TOOL] == p101['source_bindings'][b.TOOL]


# ------------------------------------------------------------------ argument validation through the EXP106 runner entry
@prepared
def test_runner_entry_runs_the_shared_runner_with_the_exp106_guard(wrapper):
    runner = wrapper.load_runner()
    import tools
    assert Path(runner.__file__).resolve() == (ROOT / b.RUNNER).resolve()
    assert sys.modules['tools.exp101_kfold_arch'] is e and tools.exp101_kfold_arch is e
    shared = load_path('exp101_shared_parser', ROOT / b.RUNNER).build_parser()
    assert [a.dest for a in runner.build_parser()._actions] == [a.dest for a in shared._actions]


@prepared
def test_runner_argv_round_trip_validates(guard_env, wrapper):
    p = e.read(REAL)
    argv = e.runner_argv(p)
    assert argv[1] == str(ROOT / e.RUNNER) and argv[argv.index('--protocol') + 1] == str(REAL)
    args = parsed(wrapper, argv)
    for k, v in p['common_args'].items():
        assert getattr(args, k) == v and type(getattr(args, k)) is type(v), k
    assert (args.workers, args.seed, args.tag, args.epochs, args.fold, args.output_root) == (12, 42, e.TAG, 16, 0, None)
    got, spec, ordered, out_root = e.validate_arguments(args, ROOT)
    assert spec['tag'] == args.tag and len(ordered['train']) == spec['n_train'] == 3455 and out_root == (ROOT / 'artifacts' / 'experiments').resolve()


@prepared
@pytest.mark.parametrize('flag, value, message', [
    ('--epochs', '8', 'frozen argument changed: epochs'), ('--fixed-epoch', '8', 'frozen argument changed: fixed_epoch'),
    ('--seed', '43', 'seed changed'), ('--workers', '4', 'B8 workers 12'), ('--workers', '99', 'workers outside'),
    ('--tag', 'EXP094L_B8_SEED42', 'tag'), ('--tag', 'EXP106_B8_E16_F0_SMOKE', 'tag'), ('--fold', '1', 'unregistered fold'),
    ('--protocol', str(ROOT / b.PROTOCOL / 'manifest.json'), 'wrong protocol'), ('--bb-lr', '0.0003', 'frozen argument changed: bb_lr'),
    ('--k-eval', '36', 'frozen argument changed: k_eval'), ('--device', 'cpu', 'frozen argument changed: device'),
    ('--amp', 'bf16', 'frozen argument changed: amp'), ('--labels', 'storage/labels/other.csv', 'input path differs: labels'),
    ('--output-root', 'C:/elsewhere', 'no --output-root')])
def test_validate_arguments_rejects(guard_env, wrapper, flag, value, message):
    argv = _mutate(e.runner_argv(e.read(REAL)), flag, value)
    with pytest.raises(ValueError, match=message):
        e.validate_arguments(parsed(wrapper, argv), ROOT)


@prepared
def test_validate_arguments_requires_the_local_environment(guard_env, wrapper):
    args = parsed(wrapper, e.runner_argv(e.read(REAL)))
    guard_env.setenv('KNEE_REMOTE_RELAX_ENV', '1')
    with pytest.raises(ValueError, match='KNEE_REMOTE_RELAX_ENV'):
        e.validate_arguments(args, ROOT)
    guard_env.delenv('KNEE_REMOTE_RELAX_ENV')
    guard_env.setenv('OMP_NUM_THREADS', '8')
    with pytest.raises(ValueError, match='OMP_NUM_THREADS'):
        e.validate_arguments(args, ROOT)


@prepared
def test_other_guards_refuse_the_exp106_protocol(guard_env, wrapper):
    args = parsed(wrapper, e.runner_argv(e.read(REAL)))
    for guard in (b, s43):
        with pytest.raises(ValueError, match='wrong protocol'):
            guard.validate_arguments(args, ROOT)


@prepared
def test_smoke_rules(guard_env, wrapper, tmp_path):
    p = e.read(REAL)
    smoke = parsed(wrapper, e.runner_argv(p, 0, 'smoke', workers=2, output_root=tmp_path))
    assert (smoke.device, smoke.amp, smoke.eval_amp, smoke.seed, smoke.tag, smoke.epochs, smoke.smoke_epochs) == ('cpu', 'none', 'none', 42, e.TAG + '_SMOKE', 16, 1)
    assert e.validate_arguments(smoke, ROOT)[3] == tmp_path.resolve()
    inside = parsed(wrapper, e.runner_argv(p, 0, 'smoke', output_root=ROOT / 'artifacts' / 'experiments'))
    with pytest.raises(ValueError, match='outside artifacts'):
        e.validate_arguments(inside, ROOT)


# ------------------------------------------------------------------ the queue entry
@prepared
def test_queue_entry_is_the_registered_full_run(guard_env, wrapper):
    plan = json.loads(QUEUE.read_text(encoding='utf-8'))
    assert isinstance(plan, list) and len(plan) == 1
    job = plan[0]
    assert job == e.queue_entry(e.read(REAL))
    assert job['tag'] == e.TAG and job['after'] == 'EXP094L_B8_SEED43' and job['continue_on_failure'] is True
    chain = json.loads(CHAIN_V8.read_text(encoding='utf-8'))
    assert set(job) == set(chain[-1]) and job['command'][0] == chain[-1]['command'][0]  # same entry format and GPU Python as the V8 chain
    assert 'EXP094L_B8_SEED43' in [j['tag'] for j in chain] and e.TAG not in [j['tag'] for j in chain]
    cmd = job['command']
    assert cmd[1] == str(ROOT / e.RUNNER) and '--validate-only' not in cmd and '--output-root' not in cmd
    args = parsed(wrapper, cmd)
    assert (args.mode, args.device, args.amp, args.epochs, args.workers) == ('full', 'cuda', 'fp16', 16, 12)
    e.validate_arguments(args, ROOT)


# ------------------------------------------------------------------ verification of a finished run (synthetic files)
def _fake_run(root, p, **overrides):
    run = root / e.TAG; run.mkdir(parents=True)
    fold = p['folds']['0']
    (run / 'oof_best.csv').write_text('StudyInstanceUID\n', encoding='utf-8'); (run / 'oof_k36.csv').write_text('StudyInstanceUID\n', encoding='utf-8')
    prov = dict(fold=0, seed=42, arch=p['common_args']['arch'], label_set='dread', label_semantic_sha256=p['label_set']['semantic_sha256'],
                protocol_sha256=e.sha256(REAL), eval_ids_ordered_sha256=fold['ordered_ids']['val_sha256'], train_ids_ordered_sha256=fold['ordered_ids']['train_sha256'],
                training_mode='full', checkpoint_selection='fixed-final', initial_state_sha256=B, initial_state_matches_registered=True,
                environment_matches_baseline=True, remote_relaxed=False, requested_workers=12, scheduler_total_steps=6896, max_fp16_skips=68)
    prov.update(overrides.get('prov', {}))
    (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    summary = dict(provenance=prov, optimizer_updates=6896, skipped_updates=0, selected_epoch_1based=16, train=3455, val=888, args=dict(tag=e.TAG, seed=42),
                   history=[dict(epoch=i) for i in range(overrides.get('epochs', 16))])
    (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    (run / 'history.json').write_text('[]', encoding='utf-8'); (run / 'model_best.pt').write_bytes(b'weights')
    names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv']
    (run / 'output-inventory.json').write_text(json.dumps({n: dict(sha256=e.sha256(run / n)) for n in names}), encoding='utf-8')
    return run


@prepared
@pytest.mark.parametrize('overrides, message', [
    ({}, None), ({'prov': dict(seed=43)}, 'provenance seed'), ({'prov': dict(initial_state_matches_registered=False)}, 'initial_state_matches_registered'),
    ({'prov': dict(remote_relaxed=True)}, 'remote_relaxed'), ({'epochs': 8}, 'history has 8 epochs'),
    ({'prov': dict(protocol_sha256='x')}, 'protocol_sha256')])
def test_verify_run(tmp_path, overrides, message):
    p = e.read(REAL)
    run = _fake_run(tmp_path, p, **overrides)
    problems = e.verify_run(run, p)
    if message is None:
        assert problems == []
    else:
        assert problems and any(message in x for x in problems), problems
