"""CPU-only checks for EXP113 (H-B-34: the EXP107 RadImageNet ResNet50 centre-slice x3 5-fold on the local GPU; tools/exp113_radr50_local.py,
training/runners/exp113_radr50_local.py).

The protocol must equal EXP107_KFOLD_RADR50_PROTOCOL_V1 in folds, labels, pos_weight, seed, initial state, weights, recipe, K and windows; only
the device enforcement (local environment and initial state, no relax), local paths and names may differ. Queue entries must be valid
tools/gpu_queue.py jobs that the guard accepts. A registered label set (relabel_v1, flight_v4) must carry the EXP099 pos_weight recomputed from
its own labels. One CPU smoke (2 optimizer updates on 16 training studies, --device cpu, subprocess) runs the shared runner end to end.
No GPU, no CUDA context.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp113_radr50_local as g  # noqa: E402
from tools import exp107_kfold_radr50 as e107  # noqa: E402
from tools import exp101_kfold_arch as b  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402

UID = g.UID
REAL = ROOT / g.PROTOCOL / 'manifest.json'
BASE = ROOT / g.BASE_PROTOCOL / 'manifest.json'
WEIGHTS = ROOT / g.WEIGHTS_LOCAL
CORPUS = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
QUEUE = ROOT / g.QUEUE_FILE
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP113 protocol not prepared')
weights = pytest.mark.skipif(not WEIGHTS.exists(), reason='RadImageNet weights not present locally')
corpus = pytest.mark.skipif(not (CORPUS / 'manifest.json').exists(), reason='64-slot cache not present')


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def registered(name):
    return (ROOT / g.LABEL_SETS_DIR / f'{name}.json').is_file()


@pytest.fixture()
def restore_guard_name():
    """The EXP113 runner entry re-points `tools.exp107_kfold_radr50` and binds a label set; restore both after the test."""
    import tools
    saved_module, saved_attr, saved_label = sys.modules.get('tools.exp107_kfold_radr50'), getattr(tools, 'exp107_kfold_radr50', None), g.LABEL_SET
    yield
    sys.modules['tools.exp107_kfold_radr50'] = saved_module
    tools.exp107_kfold_radr50 = saved_attr
    g.LABEL_SET = saved_label


@pytest.fixture()
def entry(restore_guard_name):
    return load_path('exp113_runner_entry', ROOT / g.RUNNER)


@pytest.fixture()
def guard_env(monkeypatch):
    for k, v in {**g.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    monkeypatch.delenv(e107.WEIGHTS_ENV, raising=False)
    return monkeypatch


def parsed(entry, argv, label_set='dread'):
    """argv as a queue job runs it: the entry consumes --label-set and binds it, the shared runner's parser reads the rest."""
    name, rest = entry.split_label_set(argv[2:])
    assert name == label_set
    return entry.load_runner(name).build_parser().parse_args(rest)


def _mutate(argv, flag, value=None):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag] + ([str(value)] if value is not None else [])
    return argv


def without_thread_env():
    """local_env() minus the thread variables: the runner entry must supply them (a queue runner may lack them, D-B-017)."""
    return {k: v for k, v in g.local_env().items() if k not in g.THREAD_ENV}


# ------------------------------------------------------------------ names and the runner entry
def test_names_never_collide_with_exp107_or_exp101():
    assert g.PROTOCOL not in (e107.PROTOCOL, b.PROTOCOL) and g.OOF_DIR not in (e107.OOF_DIR, b.OOF_DIR) and g.COMPARE_DIR != g.OOF_DIR
    for k in g.FOLDS:
        for mode in ('full', 'smoke'):
            for name in ('dread', 'relabel_v1', 'flight_v4'):
                new = g.tag_for(k, mode, name)
                for old in (e107.tag_for(k, mode), b.tag_for(k, mode)):
                    assert new != old and old not in new and new not in old
            assert g.tag_for(k, mode) not in g.tag_for(k, mode, 'relabel_v1')
    assert g.tag_for(3) == 'EXP113_RADR50_LOCAL_F3' and g.tag_for(0, 'smoke') == 'EXP113_RADR50_LOCAL_F0_SMOKE'
    assert g.tag_for(2, 'full', 'relabel_v1') == 'EXP113_RADR50_LOCAL_RELABEL_V1_F2'
    assert g.oof_dir_for() == 'artifacts/experiments/EXP113_RADR50_LOCAL_OOF' and g.oof_dir_for('flight_v4') == 'artifacts/experiments/EXP113_RADR50_LOCAL_FLIGHT_V4_OOF'
    assert g.QUEUE_FILE == 'artifacts/experiments/EXP113_RADR50_LOCAL_PROTOCOL_V1/queue-entries.json'
    with pytest.raises(ValueError):
        g.tag_for(0, 'full', 'Bad-Name')


def test_entry_splits_label_set_and_supplies_only_missing_thread_env():
    entry = load_path('exp113_entry_plain', ROOT / g.RUNNER)
    assert entry.split_label_set(['--fold', '1']) == ('dread', ['--fold', '1'])
    assert entry.split_label_set(['--protocol', 'p', '--label-set', 'relabel_v1', '--fold', '1']) == ('relabel_v1', ['--protocol', 'p', '--fold', '1'])
    assert entry.split_label_set(['--label-set=flight_v4', '--mode', 'full']) == ('flight_v4', ['--mode', 'full'])
    for bad in (['--label-set'], ['--label-set', '--fold'], ['--label-set', 'a', '--label-set', 'b']):
        with pytest.raises(SystemExit):
            entry.split_label_set(bad)
    env = {'OMP_NUM_THREADS': '4'}
    supplied = entry.supply_registered_env(env)
    assert env['OMP_NUM_THREADS'] == '4' and set(supplied) == set(entry.REGISTERED_ENV) - {'OMP_NUM_THREADS'} and all(env[k] == '1' for k in supplied)
    assert {k: v for k, v in entry.REGISTERED_ENV.items() if k != 'HF_HUB_OFFLINE'} == g.THREAD_ENV and entry.REGISTERED_ENV['HF_HUB_OFFLINE'] == '1'


def test_entry_runs_the_unchanged_exp107_runner_with_the_exp113_guard(entry):
    import tools
    runner = entry.load_runner('relabel_v1')
    assert Path(runner.__file__).resolve() == (ROOT / e107.RUNNER).resolve()
    assert sys.modules['tools.exp107_kfold_radr50'] is g and tools.exp107_kfold_radr50 is g and g.LABEL_SET == 'relabel_v1'
    shared = load_path('exp107_shared_parser', ROOT / e107.RUNNER).build_parser()
    assert [a.dest for a in runner.build_parser()._actions] == [a.dest for a in shared._actions]
    names = set(re.findall(r'guard\.([A-Za-z_]+)', (ROOT / e107.RUNNER).read_text(encoding='utf-8')))
    assert len(names) > 10 and all(hasattr(g, n) for n in names), sorted(n for n in names if not hasattr(g, n))
    for name in ('worker_init', 'read_rows', 'install_gold_blocker', 'uid_hash', 'sha256', 'read', 'canonical_order', 'validate_fold_membership',
                 'pos_weight', 'fold_contract', 'state_hashes', 'mismatched_keys'):
        assert getattr(g, name) is getattr(e107, name), name  # sampling, loss weighting and hashing are EXP107's own functions
    with pytest.raises(ValueError):
        entry.load_runner('Bad Name')


# ------------------------------------------------------------------ the real registration
@prepared
def test_manifest_equals_exp107_except_device_enforcement_paths_and_names():
    p, p107 = g.read(REAL), g.read(BASE)
    assert g.sha256(BASE) == g.BASE_MANIFEST_SHA256
    differing = {k for k in set(p) | set(p107) if p.get(k) != p107.get(k)}
    assert differing <= set(g.CHANGED_FIELDS), differing - set(g.CHANGED_FIELDS)
    for key in ('common_args', 'seed', 'recipe', 'resource_args', 'smoke_allowed', 'training_code_sha256', 'environment', 'corpus_manifest_sha256',
                'excluded_uids', 'fold_protocol', 'paths', 'label_set', 'total_steps', 'pos_weight_rule', 'contract_rule', 'extra_eval', 'init_weights',
                'input', 'initial_state_sha256', 'status'):
        assert p[key] == p107[key], key
    for k in g.FOLDS:
        new, old = p['folds'][str(k)], p107['folds'][str(k)]
        assert {x for x in set(new) | set(old) if new.get(x) != old.get(x)} == {'tag'} and new['tag'] == g.tag_for(k)
    ist, ist107 = p['initial_state'], p107['initial_state']
    assert {x for x in set(ist) | set(ist107) if ist.get(x) != ist107.get(x)} == set(g.INITIAL_STATE_CHANGED)
    assert ist['keys_file'] == g.PROTOCOL + '/' + g.KEYS_FILE and 'KNEE_REMOTE_RELAX_ENV is refused' in ist['enforcement']
    assert p['common_args']['device'] == 'cuda' and p['environment']['gpu'] == g.LOCAL_GPU == p['local']['gpu']
    assert p['remote'] is None and p['local']['workers'] == g.LOCAL_WORKERS == 12 and p['local']['runner'] == g.RUNNER and p['local']['shared_runner'] == e107.RUNNER
    assert p['local']['prepare_preflight']['initial_state_equal_to_exp107'] and p['local']['estimate']['seconds_per_fold'] > 0
    assert p['schema'] == g.SCHEMA and p['plan_item'] == 'H-B-34' and p['derived_from']['sha256'] == g.BASE_MANIFEST_SHA256
    assert p['evaluation']['oof_dir'] == g.OOF_DIR and p['tags']['prefix'] == g.TAG_PREFIX and p['label_sets']['default'] == 'dread'
    assert not any(c.startswith('remote T4 trajectories') for c in p['caveats']) and any('D-B-008' in c for c in p['caveats'])


@prepared
def test_keys_file_is_exp107s_and_bindings_extend_exp107():
    p, p107 = g.read(REAL), g.read(BASE)
    assert g.sha256(ROOT / p['initial_state']['keys_file']) == g.sha256(ROOT / p107['initial_state']['keys_file']) == g.BASE_KEYS_SHA256
    assert g.read(ROOT / p['initial_state']['keys_file'])['combined_sha256'] == p['initial_state_sha256']
    assert {k: p['source_bindings'][k] for k in p107['source_bindings']} == p107['source_bindings']
    assert set(p['source_bindings']) - set(p107['source_bindings']) == {g.BASE_PROTOCOL + '/manifest.json', g.RUNNER, g.TOOL, g.PROTOCOL + '/' + g.KEYS_FILE}
    assert {e107.RUNNER, e107.TOOL, e107.RAD_MODULE} <= set(p['source_bindings'])  # the shared runner and its modules are frozen
    for path, expected in p['source_bindings'].items():
        assert g.sha256(ROOT / path) == expected, path


@prepared
def test_pos_weight_recomputed_from_dread():
    p = g.read(REAL)
    labels, _ = k99._load_labels(ROOT / g.DREAD)
    for k in g.FOLDS:
        train = g.read(ROOT / p['folds'][str(k)]['ordered_ids']['path'])['train']
        assert np.array_equal(g.pos_weight(labels, train), np.asarray(p['folds'][str(k)]['pos_weight'], np.float32))


# ------------------------------------------------------------------ argument validation through the runner entry
@prepared
@weights
@pytest.mark.parametrize('fold', [0, 4])
def test_runner_argv_round_trip_validates(guard_env, entry, fold):
    view = g.view_for('dread')
    argv = g.runner_argv(view, fold)
    assert argv[0] == g.train_python() and argv[1] == str(ROOT / g.RUNNER) and argv[argv.index('--label-set') + 1] == 'dread'
    assert argv[argv.index('--init-weights') + 1] == str(WEIGHTS) and '--output-root' not in argv
    args = parsed(entry, argv)
    for k, v in view['common_args'].items():
        assert getattr(args, k) == v and type(getattr(args, k)) is type(v), k
    assert (args.workers, args.seed, args.tag, args.device, args.amp, args.mode) == (12, 42, g.tag_for(fold), 'cuda', 'fp16', 'full')
    got, spec, ordered, out_root = g.validate_arguments(args, ROOT)
    assert spec['tag'] == args.tag and len(ordered['train']) == spec['n_train'] and out_root == (ROOT / 'artifacts' / 'experiments').resolve()
    assert got == g.read(REAL)


@prepared
@weights
@pytest.mark.parametrize('flag, value, message', [
    ('--seed', '43', 'seed changed'), ('--tag', 'EXP107_RADR50_DREAD_F0', 'tag'), ('--tag', 'EXP113_RADR50_LOCAL_F1', 'tag'),
    ('--protocol', str(BASE), 'wrong protocol'), ('--bb-lr', '0.0003', 'frozen argument changed: bb_lr'), ('--arch', 'resnet34.a1_in1k', 'frozen argument changed: arch'),
    ('--device', 'cpu', 'frozen argument changed: device'), ('--workers', '99', 'workers'), ('--labels', 'storage/labels/other.csv', 'input path differs: labels'),
    ('--eval-labels', 'storage/labels/derived_relabel/relabel_v1.csv', 'input path differs: eval_labels'),
    ('--init-weights', str(ROOT / g.RUNNER), 'init weights'), ('--output-root', str(ROOT / 'artifacts' / 'other'), 'artifacts/experiments only')])
def test_validate_arguments_rejects(guard_env, entry, flag, value, message):
    argv = _mutate(g.runner_argv(g.view_for(), 0), flag, value)
    with pytest.raises(ValueError, match=message):
        g.validate_arguments(parsed(entry, argv), ROOT)


@prepared
@weights
def test_relax_env_and_other_thread_values_are_refused(guard_env, entry):
    args = parsed(entry, g.runner_argv(g.view_for(), 0))
    guard_env.setenv('KNEE_REMOTE_RELAX_ENV', '1')
    with pytest.raises(ValueError, match='KNEE_REMOTE_RELAX_ENV'):
        g.validate_arguments(args, ROOT)
    guard_env.delenv('KNEE_REMOTE_RELAX_ENV')
    guard_env.setenv('OMP_NUM_THREADS', '4')
    with pytest.raises(ValueError, match='OMP_NUM_THREADS'):
        g.validate_arguments(args, ROOT)


@prepared
@weights
def test_smoke_rules(guard_env, entry, tmp_path):
    view = g.view_for()
    smoke = parsed(entry, g.runner_argv(view, 0, 'smoke', workers=0, output_root=tmp_path))
    assert (smoke.device, smoke.amp, smoke.eval_amp, smoke.smoke_train, smoke.smoke_val, smoke.tag) == ('cpu', 'none', 'none', 16, 2, 'EXP113_RADR50_LOCAL_F0_SMOKE')
    assert g.validate_arguments(smoke, ROOT)[3] == tmp_path.resolve()
    inside = parsed(entry, g.runner_argv(view, 0, 'smoke', output_root=ROOT / 'artifacts' / 'experiments'))
    with pytest.raises(ValueError, match='outside artifacts'):
        g.validate_arguments(inside, ROOT)


@prepared
@weights
def test_exp107_guard_refuses_this_protocol(guard_env, entry):
    args = parsed(entry, g.runner_argv(g.view_for(), 0))
    with pytest.raises(ValueError, match='wrong protocol'):
        e107.validate_arguments(args, ROOT)  # why the local run needs its own guard; the Kaggle job cannot pick it up either


# ------------------------------------------------------------------ queue entries
@prepared
def test_queue_entries_are_well_formed_gpu_queue_jobs():
    entries = g.queue_entries([0, 1, 2, 3, 4], 'PREV')
    assert [e['tag'] for e in entries] == [f'EXP113_RADR50_LOCAL_F{k}' for k in range(5)]
    assert [e['after'] for e in entries] == ['PREV'] + [f'EXP113_RADR50_LOCAL_F{k}' for k in range(4)]
    for k, e in enumerate(entries):
        assert set(e) == {'tag', 'command', 'after', 'continue_on_failure'} and e['continue_on_failure'] is True
        cmd = e['command']
        assert all(isinstance(c, str) for c in cmd) and cmd[0] == g.train_python() and Path(cmd[1]) == ROOT / g.RUNNER
        assert cmd[cmd.index('--fold') + 1] == str(k) and cmd[cmd.index('--tag') + 1] == e['tag'] and cmd[cmd.index('--label-set') + 1] == 'dread'
        assert cmd[cmd.index('--workers') + 1] == '12' and cmd[cmd.index('--device') + 1] == 'cuda' and cmd[cmd.index('--amp') + 1] == 'fp16'
        assert cmd[cmd.index('--mode') + 1] == 'full' and cmd[cmd.index('--arch') + 1] == 'resnet50' and cmd[cmd.index('--input-policy') + 1] == 'center3'
        assert not {'--output-root', '--validate-only', '--grad-ckpt', '--smoke-train'} & set(cmd)
    assert set(g.queue_entries([2], None)[0]) == {'tag', 'command', 'continue_on_failure'}
    assert json.loads(json.dumps(entries)) == entries
    with pytest.raises(SystemExit):
        g.queue_entries([7], 'PREV')


@prepared
@weights
def test_queue_commands_pass_the_guard(guard_env, entry):
    for e in g.queue_entries([0, 1, 2, 3, 4], 'PREV'):
        assert g.validate_arguments(parsed(entry, e['command']), ROOT)[1]['tag'] == e['tag']


@prepared
def test_saved_queue_entries_match_the_protocol():
    if not QUEUE.exists():
        pytest.skip('queue-entries.json not written')
    saved = json.loads(QUEUE.read_text(encoding='utf-8'))
    assert saved == g.queue_entries([0, 1, 2, 3, 4], saved[0].get('after'))


# ------------------------------------------------------------------ label sets (EXP099 registry semantics)
@prepared
@pytest.mark.parametrize('name', ['relabel_v1', 'flight_v4'])
def test_registered_label_set_carries_its_recorded_pos_weight(name):
    if not registered(name):
        pytest.skip(f'{name} not registered for EXP113')
    p = g.read(REAL); view = g.view_for(name)
    src = g.read(ROOT / g.FOLD_PROTOCOL / 'label-sets' / f'{name}.json')
    assert src['pos_weight_source'] == 'labels' and view['label_set']['sha256'] == src['sha256'] and view['label_set']['semantic_sha256'] == src['semantic_sha256']
    assert view['paths']['labels'] == view['paths']['eval_labels'] == src['path'] and view['tags']['oof_dir'] == g.oof_dir_for(name)
    labels, _ = k99._load_labels(ROOT / src['path'])
    for k in g.FOLDS:
        f, f0 = view['folds'][str(k)], p['folds'][str(k)]
        assert f['pos_weight'] == src['pos_weight'][str(k)] and f['tag'] == g.tag_for(k, 'full', name)
        assert {x for x in set(f) | set(f0) if f.get(x) != f0.get(x)} <= {'pos_weight', 'tag', 'pos_weight_source'}
        train = g.read(ROOT / f['ordered_ids']['path'])['train']
        assert np.array_equal(g.pos_weight(labels, train), np.asarray(f['pos_weight'], np.float32))
    assert {x for x in set(view) | set(p) if view.get(x) != p.get(x)} == {'label_set', 'paths', 'folds', 'tags'}


@prepared
@weights
def test_label_set_argv_validates_and_cross_label_runs_are_refused(guard_env, entry):
    name = 'relabel_v1'
    if not registered(name):
        pytest.skip(f'{name} not registered for EXP113')
    view = g.view_for(name)
    argv = g.runner_argv(view, 2)
    assert argv[argv.index('--label-set') + 1] == name and argv[argv.index('--tag') + 1] == 'EXP113_RADR50_LOCAL_RELABEL_V1_F2'
    assert Path(argv[argv.index('--labels') + 1]) == Path(argv[argv.index('--eval-labels') + 1]) == ROOT / view['label_set']['path']
    args = parsed(entry, argv, name)
    got, spec, _, _ = g.validate_arguments(args, ROOT)
    assert got['label_set']['name'] == name and spec['pos_weight'] == view['folds']['2']['pos_weight']
    g.bind_label_set('dread')
    with pytest.raises(ValueError, match='tag/fold/label-set/mode mismatch'):
        g.validate_arguments(args, ROOT)
    g.bind_label_set('distill_v1')
    with pytest.raises(ValueError, match='not registered'):
        g.validate_arguments(args, ROOT)
    entries = g.queue_entries([3, 4], 'X', label_set=name)
    assert [e['tag'] for e in entries] == ['EXP113_RADR50_LOCAL_RELABEL_V1_F3', 'EXP113_RADR50_LOCAL_RELABEL_V1_F4']
    assert g.queue_file_for(name).endswith('queue-entries-relabel_v1.json')


@prepared
def test_register_labels_refusals():
    with pytest.raises(SystemExit, match="own label set"):
        g.register_labels('dread')
    with pytest.raises(ValueError):
        g.register_labels('Bad-Name')
    if not registered('distill_v1'):
        with pytest.raises(SystemExit, match='pos_weight_source'):
            g.register_labels('distill_v1')  # Dread pos_weight with distilled targets: the shared runner would refuse it
        assert not registered('distill_v1')
    for name in ('relabel_v1', 'flight_v4'):
        if registered(name):
            with pytest.raises(SystemExit, match='already registered'):
                g.register_labels(name)


# ------------------------------------------------------------------ verification, assembly and comparison (synthetic workspace, 3 folds)
@pytest.fixture()
def mini(tmp_path):
    uids = [f'u{i:02d}' for i in range(12)]
    y = np.array([i % 2 for i in range(12)])
    (tmp_path / Path(g.DREAD).parent).mkdir(parents=True)
    pd.DataFrame({UID: uids, **{t: np.where(y == 1, 0.95, 0.05) for t in g.T}}).to_csv(tmp_path / g.DREAD, index=False)
    _, semantic = k99._load_labels(tmp_path / g.DREAD)
    folds = {k: dict(train=[u for u in uids if u not in uids[4 * k:4 * k + 4]], val=uids[4 * k:4 * k + 4]) for k in range(3)}
    proto = tmp_path / g.PROTOCOL; spec = {}
    for k, v in folds.items():
        (proto / f'fold{k}').mkdir(parents=True)
        (proto / f'fold{k}' / 'ordered-ids.json').write_text(json.dumps(v), encoding='utf-8')
        spec[str(k)] = dict(ordered_ids=dict(path=f'{g.PROTOCOL}/fold{k}/ordered-ids.json', train_sha256=g.uid_hash(v['train']), val_sha256=g.uid_hash(v['val'])),
                            n_train=8, n_val=4, steps=20, max_fp16_skips=0, tag=g.tag_for(k), pos_weight=[1.0] * 12)
    manifest = dict(schema=g.SCHEMA, seed=42, common_args=e107.EXPECTED_RECIPE, folds=spec, initial_state_sha256='init',
                    source_bindings={g.SHARED_RUNNER: 'runner-sha', g.TOOL: 'guard-sha'}, tags=dict(prefix=g.TAG_PREFIX),
                    paths=dict(corpus='storage/cache/x', labels=g.DREAD, eval_labels=g.DREAD),
                    label_set=dict(name='dread', path=g.DREAD, sha256=g.sha256(tmp_path / g.DREAD), semantic_sha256=semantic))
    (proto / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return tmp_path, folds, semantic


def _register_fake(root, name):
    """A label set registered the way register-labels writes it (source registration and label file are synthetic)."""
    lab = root / 'storage/labels/other' / f'{name}.csv'; lab.parent.mkdir(parents=True, exist_ok=True)
    lab.write_bytes((root / g.DREAD).read_bytes())
    _, semantic = k99._load_labels(lab)
    src = root / 'src' / f'{name}.json'; src.parent.mkdir(exist_ok=True); src.write_text('{}', encoding='utf-8')
    record = dict(schema=g.LABEL_SCHEMA, name=name, path=lab.relative_to(root).as_posix(), sha256=g.sha256(lab), semantic_sha256=semantic,
                  protocol_sha256=g.sha256(root / g.PROTOCOL / 'manifest.json'), pos_weight={str(k): [2.0] * 12 for k in range(3)},
                  source_registration=dict(path=src.relative_to(root).as_posix(), sha256=g.sha256(src)))
    (root / g.LABEL_SETS_DIR).mkdir(parents=True, exist_ok=True)
    (root / g.LABEL_SETS_DIR / f'{name}.json').write_text(json.dumps(record), encoding='utf-8')
    return record


def _fake_run(root, runs, k, folds, label_set='dread', tag=None, score=None, **overrides):
    view = g.protocol_view(root, g.read(root / g.PROTOCOL / 'manifest.json'), label_set, root / g.PROTOCOL / 'manifest.json')
    tag = tag or g.tag_for(k, 'full', label_set)
    run = runs / tag; run.mkdir(parents=True)
    val = folds[k]['val']; score = score or {u: (0.9 if int(u[1:]) % 2 else 0.1) for u in val}
    frame = pd.DataFrame({UID: val, **{t: [score[u] for u in val] for t in g.T}})
    frame.to_csv(run / 'oof_best.csv', index=False); frame.to_csv(run / 'oof_k36.csv', index=False)
    prov = dict(fold=k, seed=42, label_set=label_set, label_semantic_sha256=view['label_set']['semantic_sha256'], label_file_sha256=view['label_set']['sha256'],
                protocol_sha256=g.sha256(root / g.PROTOCOL / 'manifest.json'), eval_ids_ordered_sha256=g.uid_hash(val),
                train_ids_ordered_sha256=g.uid_hash(folds[k]['train']), training_mode='full', checkpoint_selection='fixed-final', arch='resnet50',
                input_policy='center3', norm='radimagenet', remote_relaxed=False, initial_state_sha256='init', initial_state_matches_registered=True,
                environment_matches_baseline=True, runner_sha256='runner-sha', guard_sha256='guard-sha',
                init_weights=dict(weights_sha256=g.RADIMAGENET_SHA256), actual_pos_weight=view['folds'][str(k)]['pos_weight'])
    prov.update(overrides)
    (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    (run / 'summary.json').write_text(json.dumps(dict(provenance=prov, optimizer_updates=20, skipped_updates=0, selected_epoch_1based=20, train=8, val=4,
                                                      args=dict(tag=tag, seed=42), total_seconds=1200)), encoding='utf-8')
    (run / 'history.json').write_text('[]', encoding='utf-8'); (run / 'model_best.pt').write_bytes(b'weights')
    names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv']
    (run / 'output-inventory.json').write_text(json.dumps({n: dict(sha256=g.sha256(run / n)) for n in names}), encoding='utf-8')
    return run


def test_assemble_local_runs_into_the_exp113_oof_dir(mini):
    root, folds, _ = mini
    runs = root / 'artifacts' / 'experiments'
    for k in folds:
        _fake_run(root, runs, k, folds)
    body = g.assemble(root=root)  # default runs dir = artifacts/experiments (local outputs)
    out = root / g.OOF_DIR
    assert body['schema'] == 'exp113-oof-v1' and body['rows'] == 12 and body['metrics']['k24']['pooled']['macro12'] == 1.0 and body['k36_missing_folds'] == []
    assert body['runs_dir'] == 'artifacts/experiments' and not (root / e107.OOF_DIR).exists()
    assert (out / 'result.md').read_text(encoding='utf-8').startswith('# EXP113 RadImageNet ResNet50 5-fold OOF (dread, local RTX 4070 Ti SUPER)')
    with pytest.raises(SystemExit):
        g.assemble(root=root)  # refuses to overwrite


@pytest.mark.parametrize('damage, message', [
    (dict(remote_relaxed=True), 'remote_relaxed'), (dict(initial_state_matches_registered=False), 'initial_state_matches_registered'),
    (dict(environment_matches_baseline=False), 'environment_matches_baseline'), (dict(protocol_sha256=g.BASE_MANIFEST_SHA256), 'protocol_sha256'),
    (dict(guard_sha256='exp107-guard'), 'guard_sha256'), (dict(actual_pos_weight=[2.0] * 12), 'pos_weight differs'), ('truncate', 'output-inventory')])
def test_verify_run_refuses_t4_style_or_damaged_runs(mini, damage, message):
    root, folds, _ = mini
    runs = root / 'runs'
    for k in folds:
        _fake_run(root, runs, k, folds, **(damage if (isinstance(damage, dict) and k == 1) else {}))
    run = runs / g.tag_for(1)
    if damage == 'truncate':
        (run / 'model_best.pt').write_bytes(b'wei')
    p = g.read(root / g.PROTOCOL / 'manifest.json')
    problems = g.verify_run(run, p, 1, g.sha256(root / g.PROTOCOL / 'manifest.json'))
    assert problems and any(message in x for x in problems), problems
    with pytest.raises(SystemExit):
        g.assemble(runs_dir=runs, root=root)
    assert not (root / g.OOF_DIR).exists()


def test_exp107_named_runs_are_not_picked_up(mini):
    root, folds, _ = mini
    runs = root / 'runs'
    for k in folds:
        _fake_run(root, runs, k, folds, tag=e107.tag_for(k))
    with pytest.raises(SystemExit):
        g.assemble(runs_dir=runs, root=root)


def test_label_set_runs_assemble_into_their_own_dir_and_need_their_registration(mini):
    root, folds, _ = mini
    runs = root / 'artifacts' / 'experiments'
    with pytest.raises(ValueError, match='not registered'):
        g.assemble('relabel_x', root=root)
    reg = _register_fake(root, 'relabel_x')
    view = g.protocol_view(root, g.read(root / g.PROTOCOL / 'manifest.json'), 'relabel_x', root / g.PROTOCOL / 'manifest.json')
    assert view['folds']['1']['pos_weight'] == [2.0] * 12 and view['paths']['labels'] == view['paths']['eval_labels'] == reg['path']
    for k in folds:
        _fake_run(root, runs, k, folds, tag=g.tag_for(k))  # dread runs only
    with pytest.raises(SystemExit):
        g.assemble('relabel_x', root=root)
    for k in folds:
        _fake_run(root, runs, k, folds, label_set='relabel_x')
    body = g.assemble('relabel_x', root=root)
    assert body['label_set'] == 'relabel_x' and (root / g.oof_dir_for('relabel_x') / 'oof_k24.csv').is_file() and not (root / g.OOF_DIR).exists()
    (root / reg['path']).write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='label file changed'):
        g.protocol_view(root, g.read(root / g.PROTOCOL / 'manifest.json'), 'relabel_x', root / g.PROTOCOL / 'manifest.json')


def _write_t4_oof(root, folds, score):
    """An exp107-oof-v1 directory with the given per-study scores (all targets equal)."""
    frames = {k: pd.DataFrame({UID: v['val'], **{t: [score[u] for u in v['val']] for t in g.T}}) for k, v in folds.items()}
    result = k99.assemble_and_score(frames, frames, {k: v['val'] for k, v in folds.items()}, root / g.DREAD)
    out = root / e107.OOF_DIR; out.mkdir(parents=True)
    result['oof_k24'].to_csv(out / 'oof_k24.csv', index=False); result['oof_k36'].to_csv(out / 'oof_k36.csv', index=False)
    body = dict(schema='exp107-oof-v1', label_set='dread', protocol_sha256='t4', metrics=result['metrics'],
                outputs={f: g.sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv')})
    (out / 'result.json').write_text(json.dumps(body), encoding='utf-8')
    return body


def test_compare_reports_local_minus_t4_per_fold_pooled_and_target(mini):
    root, folds, _ = mini
    runs = root / 'artifacts' / 'experiments'
    for k in folds:
        _fake_run(root, runs, k, folds)
    g.assemble(root=root)
    uids = [u for v in folds.values() for u in v['val']]
    t4 = {u: (0.9 if int(u[1:]) % 2 else 0.1) for u in uids}
    t4['u05'] = 0.1  # one positive of fold 1 tied with both negatives: AUC 0.75 on fold 1 for every target
    _write_t4_oof(root, folds, t4)
    body = g.compare(root=root)
    m = body['metrics']['k24']['scopes']
    assert m['fold0']['delta'] == 0.0 and m['fold2']['delta'] == 0.0 and m['fold1']['t4'] == 0.75 and m['fold1']['delta'] == pytest.approx(0.25)
    assert m['pooled']['local'] == 1.0 and m['pooled']['delta'] > 0 and set(m['pooled']['targets']) == set(g.T)
    assert m['fold0']['mean_spearman'] == pytest.approx(1.0) and m['pooled']['mean_spearman'] < 1.0
    assert body['metrics']['k24']['delta_by_fold'] == [0.0, pytest.approx(0.25), 0.0]
    assert set(body['metrics']) == {'k24', 'k36'} and 'not a pass/fail' in body['interpretation']
    assert (root / g.COMPARE_DIR / 'result.md').read_text(encoding='utf-8').startswith('# EXP113 local RadR50 5-fold vs EXP107 Kaggle T4')
    with pytest.raises(SystemExit):
        g.compare(root=root)  # refuses to overwrite


def test_compare_refuses_a_changed_t4_table(mini):
    root, folds, _ = mini
    for k in folds:
        _fake_run(root, root / 'artifacts' / 'experiments', k, folds)
    g.assemble(root=root)
    _write_t4_oof(root, folds, {u: 0.5 for v in folds.values() for u in v['val']})
    (root / e107.OOF_DIR / 'oof_k36.csv').write_text('x', encoding='utf-8')
    with pytest.raises(SystemExit, match='differs from its result.json hash'):
        g.compare(root=root)


# ------------------------------------------------------------------ the shared runner on CPU (subprocess; no CUDA context)
@prepared
@weights
@corpus
@pytest.mark.parametrize('label_set, fold', [('dread', 0), ('relabel_v1', 3)])
def test_validate_only_on_cpu(label_set, fold):
    if label_set != 'dread' and not registered(label_set):
        pytest.skip(f'{label_set} not registered for EXP113')
    view = g.view_for(label_set)
    argv = g.runner_argv(view, fold, 'full', label_set=label_set) + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=without_thread_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    out = json.loads(r.stdout)
    spec = view['folds'][str(fold)]
    assert out['status'] == 'VALIDATED_NO_TRAINING' and out['tag'] == g.tag_for(fold, 'full', label_set) and out['output_exists'] is False
    assert out['initial_state_matches_registered'] is True and out['initial_state_sha256'] == view['initial_state_sha256'] and out['remote_relaxed'] is False
    assert (out['steps'], out['len_train_loader'], out['max_fp16_skips'], out['train'], out['val']) == (spec['steps'], spec['len_train_loader'],
                                                                                                          spec['max_fp16_skips'], spec['n_train'], spec['n_val'])
    assert out['pos_weight'] == [round(float(x), 6) for x in spec['pos_weight']] and out['label_semantic_sha256'] == view['label_set']['semantic_sha256']


@prepared
@weights
@corpus
def test_cpu_smoke_two_optimizer_updates(tmp_path):
    view = g.view_for()
    fold = 1
    argv = g.runner_argv(view, fold, 'smoke', workers=2, output_root=tmp_path, smoke=dict(train=16, val=2, epochs=1))
    assert argv[argv.index('--device') + 1] == 'cpu' and argv[argv.index('--amp') + 1] == 'none'
    r = subprocess.run(argv, cwd=ROOT, env=without_thread_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    run = tmp_path / g.tag_for(fold, 'smoke')
    s = g.read(run / 'summary.json'); prov = s['provenance']
    assert s['optimizer_updates'] == 2 and s['skipped_updates'] == 0 and (s['train'], s['val'], s['mode']) == (16, 2, 'smoke')
    assert len(s['history']) == 1 and np.isfinite(s['history'][0]['loss']) and prov['scheduler_total_steps'] == 2
    assert prov['initial_state_matches_registered'] is True and prov['initial_state_sha256'] == view['initial_state_sha256']
    assert prov['cuda_initialized'] is False and prov['remote_relaxed'] is False and (prov['label_set'], prov['seed']) == ('dread', 42)
    assert prov['init_weights']['weights_sha256'] == g.RADIMAGENET_SHA256 and (prov['input_policy'], prov['norm']) == ('center3', 'radimagenet')
    assert prov['guard_sha256'] == g.sha256(ROOT / g.TOOL) and prov['runner_sha256'] == g.sha256(ROOT / e107.RUNNER)
    assert prov['actual_pos_weight'] == view['folds'][str(fold)]['pos_weight'] and prov['protocol_sha256'] == g.sha256(REAL)
    val = g.read(ROOT / view['folds'][str(fold)]['ordered_ids']['path'])['val'][:2]
    for name in ('oof_best.csv', 'oof_k36.csv'):
        df = pd.read_csv(run / name, dtype={UID: str})
        assert df[UID].tolist() == val and list(df.columns) == [UID] + g.T and np.isfinite(df[g.T].to_numpy()).all()
    inv = g.read(run / 'output-inventory.json')
    assert all(g.sha256(run / n) == v['sha256'] for n, v in inv.items())
