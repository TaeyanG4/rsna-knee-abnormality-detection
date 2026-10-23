"""CPU-only checks for EXP118 (H-B-34: the EXP113 local RadR50 5-fold with seed 43; tools/exp118_radr50_local_s43.py,
training/runners/exp118_radr50_local_s43.py).

The protocol must equal EXP113_RADR50_LOCAL_PROTOCOL_V1 in folds, labels, pos_weight, weights, recipe, K, windows and the local environment;
only the seed (43), the seed-43 initial state (backbone bytes equal to EXP113's, head re-drawn), names and paths may differ. The guard must
refuse seed 42, EXP113 tags and the EXP113 protocol; queue entries must be valid tools/gpu_queue.py jobs that the guard accepts. A fresh
process (--validate-only) must reproduce the registered seed-43 state, and one CPU smoke (2 optimizer updates on 16 training studies,
--device cpu, subprocess) runs the shared runner end to end. `compare` is checked on a synthetic workspace. No GPU, no CUDA context.
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
from tools import exp118_radr50_local_s43 as g  # noqa: E402
from tools import exp113_radr50_local as e113  # noqa: E402
from tools import exp107_kfold_radr50 as e107  # noqa: E402
from tools import exp101_kfold_arch as b  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
from tools import exp115_vlm_eval as mr  # noqa: E402

UID = g.UID
REAL = ROOT / g.PROTOCOL / 'manifest.json'
BASE = ROOT / g.BASE_PROTOCOL / 'manifest.json'
WEIGHTS = ROOT / g.WEIGHTS_LOCAL
CORPUS = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
QUEUE = ROOT / g.QUEUE_FILE
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP118 protocol not prepared')
weights = pytest.mark.skipif(not WEIGHTS.exists(), reason='RadImageNet weights not present locally')
corpus = pytest.mark.skipif(not (CORPUS / 'manifest.json').exists(), reason='64-slot cache not present')


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def restore_guard_name():
    """The EXP118 runner entry re-points `tools.exp107_kfold_radr50`; restore it after the test."""
    import tools
    saved_module, saved_attr = sys.modules.get('tools.exp107_kfold_radr50'), getattr(tools, 'exp107_kfold_radr50', None)
    yield
    sys.modules['tools.exp107_kfold_radr50'] = saved_module
    tools.exp107_kfold_radr50 = saved_attr


@pytest.fixture()
def entry(restore_guard_name):
    return load_path('exp118_runner_entry', ROOT / g.RUNNER)


@pytest.fixture()
def guard_env(monkeypatch):
    for k, v in {**g.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    monkeypatch.delenv(e107.WEIGHTS_ENV, raising=False)
    return monkeypatch


def parsed(entry, argv):
    """argv as a queue job runs it: the entry strips an optional --label-set dread, the shared runner's parser reads the rest."""
    return entry.load_runner().build_parser().parse_args(entry.strip_label_set(argv[2:]))


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


# ------------------------------------------------------------------ names, constants and the runner entry
def test_names_never_collide_with_exp113_exp107_exp101_exp117():
    from tools import exp117_multihead as e117
    assert len({g.PROTOCOL, e113.PROTOCOL, e107.PROTOCOL, b.PROTOCOL, e117.PROTOCOL}) == 5
    assert len({g.OOF_DIR, e113.OOF_DIR, e107.OOF_DIR, b.OOF_DIR, e117.OOF_DIR, g.COMPARE_DIR, e113.COMPARE_DIR}) == 7
    for k in g.FOLDS:
        for mode in ('full', 'smoke'):
            new = g.tag_for(k, mode)
            olds = [e107.tag_for(k, mode), b.tag_for(k, mode), e117.tag_for(k, mode)] + [e113.tag_for(k, mode, n) for n in ('dread', 'relabel_v1', 'flight_v4')]
            for old in olds:
                assert new != old and old not in new and new not in old
    assert g.tag_for(3) == 'EXP118_RADR50_LOCAL_S43_F3' and g.tag_for(0, 'smoke') == 'EXP118_RADR50_LOCAL_S43_F0_SMOKE'
    assert g.OOF_DIR == 'artifacts/experiments/EXP118_RADR50_LOCAL_S43_OOF'
    assert g.QUEUE_FILE == 'artifacts/experiments/EXP118_RADR50_LOCAL_S43_PROTOCOL_V1/queue-entries.json'
    assert (g.SEED, g.BASE_SEED, g.DEFAULT_AFTER, g.LABEL_SET) == (43, 42, 'EXP117_MULTIHEAD_F0', 'dread')
    assert g.BASE_MANIFEST_SHA256 == e117.BASE_MANIFEST_SHA256 and g.BASE_KEYS_SHA256 == e113.BASE_KEYS_SHA256 and g.BASE_OOF_DIR == e113.OOF_DIR


def test_exp113_files_are_reused_unchanged():
    """EXP118 binds the EXP113 manifest, tool and entry by sha256 (EXP113 is running; nothing of it may change)."""
    assert g.sha256(BASE) == g.BASE_MANIFEST_SHA256
    p113 = g.read(BASE)
    for path in (e113.TOOL, e113.RUNNER, e113.SHARED_RUNNER, e107.TOOL, p113['initial_state']['keys_file']):
        assert g.sha256(ROOT / path) == p113['source_bindings'][path], path
    for name in ('worker_init', 'read_rows', 'install_gold_blocker', 'uid_hash', 'sha256', 'read', 'canonical_order', 'validate_fold_membership',
                 'pos_weight', 'fold_contract', 'state_hashes', 'mismatched_keys', 'weights_path', 'preflight_versions', 'local_env'):
        assert getattr(g, name) is getattr(e113, name), name


def test_entry_strips_only_dread_and_reuses_the_exp113_env_supply():
    entry = load_path('exp118_entry_plain', ROOT / g.RUNNER)
    assert entry.strip_label_set(['--fold', '1']) == ['--fold', '1']
    assert entry.strip_label_set(['--protocol', 'p', '--label-set', 'dread', '--fold', '1']) == ['--protocol', 'p', '--fold', '1']
    for bad in (['--label-set', 'relabel_v1'], ['--label-set=flight_v4'], ['--label-set'], ['--label-set', 'dread', '--label-set', 'dread']):
        with pytest.raises(SystemExit):
            entry.strip_label_set(bad)
    assert Path(entry._E113.__file__).resolve() == (ROOT / e113.RUNNER).resolve()
    assert entry.supply_registered_env is entry._E113.supply_registered_env and entry.REGISTERED_ENV == entry._E113.REGISTERED_ENV
    env = {'OMP_NUM_THREADS': '4'}
    supplied = entry.supply_registered_env(env)
    assert env['OMP_NUM_THREADS'] == '4' and set(supplied) == set(entry.REGISTERED_ENV) - {'OMP_NUM_THREADS'} and all(env[k] == '1' for k in supplied)
    assert {k: v for k, v in entry.REGISTERED_ENV.items() if k != 'HF_HUB_OFFLINE'} == g.THREAD_ENV


def test_entry_runs_the_unchanged_exp107_runner_with_the_exp118_guard(entry):
    import tools
    runner = entry.load_runner()
    assert Path(runner.__file__).resolve() == (ROOT / e107.RUNNER).resolve()
    assert sys.modules['tools.exp107_kfold_radr50'] is g and tools.exp107_kfold_radr50 is g
    shared = load_path('exp107_shared_parser_118', ROOT / e107.RUNNER).build_parser()
    assert [a.dest for a in runner.build_parser()._actions] == [a.dest for a in shared._actions]
    names = set(re.findall(r'guard\.([A-Za-z_]+)', (ROOT / e107.RUNNER).read_text(encoding='utf-8')))
    assert len(names) > 10 and all(hasattr(g, n) for n in names), sorted(n for n in names if not hasattr(g, n))
    for name in names - {'validate_arguments', 'LABEL_SET', '__file__'}:
        assert getattr(g, name) is getattr(e107, name), name  # sampling, loss weighting and hashing are EXP107's own
    assert g.LABEL_SET == 'dread'


# ------------------------------------------------------------------ pure helpers
def test_seed_state_report_rules():
    k42 = {'backbone.a': ['h1', 'float32', [2]], 'backbone.b': ['h2', 'float32', [2]], 'head.w': ['h3', 'float32', [2]], 'head.ln': ['h4', 'float32', [2]]}
    k43 = {**k42, 'head.w': ['x3', 'float32', [2]]}
    r = g.seed_state_report(k43, k42)
    assert (r['keys'], r['backbone_keys'], r['head_keys'], r['head_keys_differing_from_seed42'], r['head_keys_equal_to_seed42']) == (4, 2, 2, 1, 1)
    assert r['differing'] == ['head.w'] and r['backbone_equal_to_seed42'] is True
    with pytest.raises(ValueError, match='backbone differs'):
        g.seed_state_report({**k43, 'backbone.a': ['zz', 'float32', [2]]}, k42)
    with pytest.raises(ValueError, match='did not reach'):
        g.seed_state_report(dict(k42), k42)
    with pytest.raises(ValueError, match='different keys'):
        g.seed_state_report({**k43, 'extra': ['e', 'float32', [1]]}, k42)


def test_adoption_rule_and_spread():
    up = dict(delta=0.005, ci95=[0.001, 0.009]); cons = dict(delta=0.001)
    assert g.adoption_rule(up, cons) == dict(threshold=0.004, adopts_seed43_over_seed42=True, adopts_seed42_over_seed43=False, false_adoption=True)
    down = dict(delta=-0.0045, ci95=[-0.008, -0.001])
    assert g.adoption_rule(down, dict(delta=-0.002))['adopts_seed42_over_seed43'] is True
    for d, c in ((dict(delta=0.005, ci95=[-0.001, 0.01]), cons), (dict(delta=0.003, ci95=[0.001, 0.006]), cons), (up, dict(delta=-0.0001)),
                 (down, dict(delta=0.0001))):
        assert g.adoption_rule(d, c)['false_adoption'] is False
    s = g._spread([0.001, -0.002, 0.003])
    assert s['mean'] == pytest.approx(0.000666666, abs=1e-8) and s['max_abs'] == 0.003 and s['sd'] == pytest.approx(np.std([0.001, -0.002, 0.003], ddof=1))


@weights
@corpus
def test_prepare_refuses_an_existing_destination_and_a_wrong_seed42_control(tmp_path):
    with pytest.raises(SystemExit, match='Refusing to overwrite'):
        g.prepare(dest=tmp_path)
    import torch

    def fake(arch, seed):  # any state that is not EXP113's seed-42 bytes must stop prepare before anything is written
        torch.manual_seed(seed)
        return torch.nn.Linear(2, 2), {}
    dest = tmp_path / 'registration'
    with pytest.raises(ValueError, match='seed-42 control'):
        g.prepare(model_factory=fake, dest=dest)
    assert not dest.exists() and not torch.cuda.is_initialized()


# ------------------------------------------------------------------ the real registration
@prepared
def test_manifest_equals_exp113_except_seed_initial_state_and_names():
    p, p113 = g.read(REAL), g.read(BASE)
    differing = {k for k in set(p) | set(p113) if p.get(k) != p113.get(k)}
    assert differing <= set(g.CHANGED_FIELDS), differing - set(g.CHANGED_FIELDS)
    for key in ('common_args', 'recipe', 'resource_args', 'smoke_allowed', 'training_code_sha256', 'environment', 'corpus_manifest_sha256',
                'excluded_uids', 'fold_protocol', 'paths', 'label_set', 'total_steps', 'pos_weight_rule', 'contract_rule', 'extra_eval', 'init_weights',
                'input', 'status', 'plan_item', 'remote'):
        assert p[key] == p113[key], key
    assert (p['seed'], p113['seed']) == (43, 42) and p['initial_state_sha256'] != p113['initial_state_sha256']
    for k in g.FOLDS:
        new, old = p['folds'][str(k)], p113['folds'][str(k)]
        assert {x for x in set(new) | set(old) if new.get(x) != old.get(x)} == {'tag'} and new['tag'] == g.tag_for(k)
    ist, ist113 = p['initial_state'], p113['initial_state']
    assert {x for x in set(ist) | set(ist113) if ist.get(x) != ist113.get(x)} == set(g.INITIAL_STATE_CHANGED)
    assert ist['seed'] == 43 and 'seed_all(43)' in ist['rule'] and 'seed-43 state must equal' in ist['enforcement'] and 'KNEE_REMOTE_RELAX_ENV is refused' in ist['enforcement']
    c = ist['seed42_control']
    assert c['initial_state_sha256'] == p113['initial_state_sha256'] and c['equal_to_exp113'] and c['backbone_equal_to_seed42']
    assert c['head_keys_differing_from_seed42'] == len(c['head_keys_differing']) > 0 and c['backbone_keys'] == ist['backbone_keys']
    assert p['local']['runner'] == g.RUNNER and p['local']['guard'] == g.TOOL and p['local']['shared_runner'] == e107.RUNNER
    assert p['local']['workers'] == 12 and p['local']['gpu'] == g.LOCAL_GPU and p['local']['prepare_preflight']['seed42_control_equal_to_exp113']
    assert {k for k in set(p['local']) | set(p113['local']) if p['local'].get(k) != p113['local'].get(k)} <= {
        'initial_state', 'runner', 'guard', 'runner_rule', 'queue', 'comparison', 'seed', 'prepare_preflight', 'estimate'}
    assert p['schema'] == g.SCHEMA and p['derived_from']['sha256'] == g.BASE_MANIFEST_SHA256 and p['derived_from']['changed_seed'] == {'from': 42, 'to': 43}
    assert p['evaluation']['oof_dir'] == g.OOF_DIR and p['tags']['prefix'] == g.TAG_PREFIX and p['label_sets']['allowed'] == ['dread']
    assert not any(x.startswith(('single seed', 'local RTX 4070 Ti SUPER trajectories (Windows')) for x in p['caveats']) and any('seed 43' in x for x in p['caveats'])


@prepared
def test_seed43_keys_file_and_bindings():
    p, p113 = g.read(REAL), g.read(BASE)
    keys_path = ROOT / p['initial_state']['keys_file']
    k43, k42 = g.read(keys_path), g.read(ROOT / p113['initial_state']['keys_file'])
    assert keys_path.parent == ROOT / g.PROTOCOL and k43['seed'] == 43 and k43['combined_sha256'] == p['initial_state_sha256']
    assert {k: v for k, v in k43.items() if k not in ('seed', 'combined_sha256', 'keys')} == {k: v for k, v in k42.items() if k not in ('seed', 'combined_sha256', 'keys')}
    report = g.seed_state_report(k43['keys'], k42['keys'])  # backbone equal, head re-drawn
    assert report['keys'] == p['initial_state']['keys'] and report['differing'] == p['initial_state']['seed42_control']['head_keys_differing']
    assert {k: p['source_bindings'][k] for k in p113['source_bindings']} == p113['source_bindings']
    assert set(p['source_bindings']) - set(p113['source_bindings']) == {g.BASE_PROTOCOL + '/manifest.json', g.RUNNER, g.TOOL, g.PROTOCOL + '/' + g.KEYS_FILE}
    assert p['source_bindings'][g.BASE_PROTOCOL + '/manifest.json'] == g.BASE_MANIFEST_SHA256
    assert {e107.RUNNER, e107.TOOL, e107.RAD_MODULE, e113.TOOL, e113.RUNNER} <= set(p['source_bindings'])
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
    p = g.read(REAL)
    argv = g.runner_argv(p, fold)
    assert argv[0] == g.train_python() and argv[1] == str(ROOT / g.RUNNER) and '--label-set' not in argv and '--output-root' not in argv
    assert argv[argv.index('--protocol') + 1] == str(REAL) and argv[argv.index('--init-weights') + 1] == str(WEIGHTS)
    old = e113.runner_argv(e113.view_for('dread'), fold)  # identical recipe flags and paths; only entry, protocol, tag, seed, label-set differ
    i = old.index('--label-set'); del old[i:i + 2]
    assert [j for j, (x, y) in enumerate(zip(argv, old)) if x != y] == [1, argv.index('--protocol') + 1, argv.index('--tag') + 1, argv.index('--seed') + 1]
    assert len(argv) == len(old)
    args = parsed(entry, argv)
    for k, v in p['common_args'].items():
        assert getattr(args, k) == v and type(getattr(args, k)) is type(v), k
    assert (args.workers, args.seed, args.tag, args.device, args.amp, args.mode) == (12, 43, g.tag_for(fold), 'cuda', 'fp16', 'full')
    got, spec, ordered, out_root = g.validate_arguments(args, ROOT)
    assert spec['tag'] == args.tag and len(ordered['train']) == spec['n_train'] and out_root == (ROOT / 'artifacts' / 'experiments').resolve()
    assert got == p


@prepared
@weights
@pytest.mark.parametrize('flag, value, message', [
    ('--seed', '42', 'seed changed'), ('--seed', '44', 'seed changed'), ('--tag', 'EXP113_RADR50_LOCAL_F0', 'tag'),
    ('--tag', 'EXP118_RADR50_LOCAL_S43_F1', 'tag'), ('--protocol', str(BASE), 'wrong protocol'),
    ('--bb-lr', '0.0003', 'frozen argument changed: bb_lr'), ('--arch', 'resnet34.a1_in1k', 'frozen argument changed: arch'),
    ('--device', 'cpu', 'frozen argument changed: device'), ('--workers', '99', 'workers'), ('--labels', 'storage/labels/other.csv', 'input path differs: labels'),
    ('--eval-labels', 'storage/labels/derived_relabel/relabel_v1.csv', 'input path differs: eval_labels'),
    ('--init-weights', str(ROOT / g.RUNNER), 'init weights'), ('--output-root', str(ROOT / 'artifacts' / 'other'), 'artifacts/experiments only')])
def test_validate_arguments_rejects(guard_env, entry, flag, value, message):
    argv = _mutate(g.runner_argv(g.read(REAL), 0), flag, value)
    with pytest.raises(ValueError, match=message):
        g.validate_arguments(parsed(entry, argv), ROOT)


@prepared
@weights
def test_relax_env_and_other_thread_values_are_refused(guard_env, entry):
    args = parsed(entry, g.runner_argv(g.read(REAL), 0))
    guard_env.setenv('KNEE_REMOTE_RELAX_ENV', '1')
    with pytest.raises(ValueError, match='KNEE_REMOTE_RELAX_ENV'):
        g.validate_arguments(args, ROOT)
    guard_env.delenv('KNEE_REMOTE_RELAX_ENV')
    guard_env.setenv('OMP_NUM_THREADS', '2')
    with pytest.raises(ValueError, match='OMP_NUM_THREADS'):
        g.validate_arguments(args, ROOT)


@prepared
@weights
def test_smoke_rules(guard_env, entry, tmp_path):
    p = g.read(REAL)
    smoke = parsed(entry, g.runner_argv(p, 0, 'smoke', workers=0, output_root=tmp_path))
    assert (smoke.device, smoke.amp, smoke.eval_amp, smoke.smoke_train, smoke.smoke_val, smoke.seed, smoke.tag) == (
        'cpu', 'none', 'none', 16, 2, 43, 'EXP118_RADR50_LOCAL_S43_F0_SMOKE')
    assert g.validate_arguments(smoke, ROOT)[3] == tmp_path.resolve()
    inside = parsed(entry, g.runner_argv(p, 0, 'smoke', output_root=ROOT / 'artifacts' / 'experiments'))
    with pytest.raises(ValueError, match='outside artifacts'):
        g.validate_arguments(inside, ROOT)


@prepared
@weights
def test_exp113_and_exp107_guards_refuse_this_protocol_and_vice_versa(guard_env, entry):
    args = parsed(entry, g.runner_argv(g.read(REAL), 0))
    for other in (e113, e107):
        with pytest.raises(ValueError, match='wrong protocol'):
            other.validate_arguments(args, ROOT)  # an EXP113 / EXP107 job can never pick up the seed-43 registration
    old = parsed(entry, e113.runner_argv(e113.view_for('dread'), 0))
    with pytest.raises(ValueError, match='wrong protocol'):
        g.validate_arguments(old, ROOT)
    with pytest.raises(ValueError):
        g.runner_argv(g.read(BASE), 0)  # the EXP113 manifest cannot be rendered as an EXP118 job


# ------------------------------------------------------------------ queue entries
@prepared
def test_queue_entries_are_well_formed_gpu_queue_jobs():
    entries = g.queue_entries([0, 1, 2, 3, 4])
    assert [e['tag'] for e in entries] == [f'EXP118_RADR50_LOCAL_S43_F{k}' for k in range(5)]
    assert [e['after'] for e in entries] == ['EXP117_MULTIHEAD_F0'] + [f'EXP118_RADR50_LOCAL_S43_F{k}' for k in range(4)]
    for k, e in enumerate(entries):
        assert set(e) == {'tag', 'command', 'after', 'continue_on_failure'} and e['continue_on_failure'] is True
        cmd = e['command']
        assert all(isinstance(c, str) for c in cmd) and cmd[0] == g.train_python() and Path(cmd[1]) == ROOT / g.RUNNER
        assert cmd[cmd.index('--fold') + 1] == str(k) and cmd[cmd.index('--tag') + 1] == e['tag'] and cmd[cmd.index('--seed') + 1] == '43'
        assert cmd[cmd.index('--workers') + 1] == '12' and cmd[cmd.index('--device') + 1] == 'cuda' and cmd[cmd.index('--amp') + 1] == 'fp16'
        assert cmd[cmd.index('--mode') + 1] == 'full' and cmd[cmd.index('--arch') + 1] == 'resnet50' and cmd[cmd.index('--input-policy') + 1] == 'center3'
        assert Path(cmd[cmd.index('--protocol') + 1]) == REAL
        assert not {'--output-root', '--validate-only', '--grad-ckpt', '--smoke-train', '--label-set'} & set(cmd)
    assert set(g.queue_entries([2], None)[0]) == {'tag', 'command', 'continue_on_failure'}
    assert g.queue_entries([0], 'PREV')[0]['after'] == 'PREV'
    assert json.loads(json.dumps(entries)) == entries
    with pytest.raises(SystemExit):
        g.queue_entries([7])


@prepared
@weights
def test_queue_commands_pass_the_guard(guard_env, entry):
    for e in g.queue_entries([0, 1, 2, 3, 4]):
        assert g.validate_arguments(parsed(entry, e['command']), ROOT)[1]['tag'] == e['tag']


@prepared
def test_saved_queue_entries_match_the_protocol():
    if not QUEUE.exists():
        pytest.skip('queue-entries.json not written')
    saved = json.loads(QUEUE.read_text(encoding='utf-8'))
    assert saved[0]['after'] == g.DEFAULT_AFTER and saved == g.queue_entries([0, 1, 2, 3, 4], saved[0]['after'])


# ------------------------------------------------------------------ verification, assembly and comparison (synthetic workspace, 3 folds)
N_STUDIES = 30


def _write_labels(root, path, uids, y):
    f = root / path; f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({UID: uids, **{t: np.where(y == 1, 0.95, 0.05) for t in g.T}}).to_csv(f, index=False)


@pytest.fixture()
def mini(tmp_path):
    uids = [f'u{i:02d}' for i in range(N_STUDIES)]
    y = np.array([i % 2 for i in range(N_STUDIES)])
    _write_labels(tmp_path, g.DREAD, uids, y)
    for path in {*(v[0] for v in mr.LABELS.values()), *mr.CONSENSUS_SETS.values()} - {g.DREAD}:
        _write_labels(tmp_path, path, uids, y)  # every reference agrees with Dread here
    comp = tmp_path / mr.COMPONENTS; comp.parent.mkdir(parents=True, exist_ok=True)
    comp.write_text(json.dumps([dict(component_id=f'c{i // 2}', members=[u for u in uids[i:i + 2]]) for i in range(0, N_STUDIES, 2)]), encoding='utf-8')
    _, semantic = k99._load_labels(tmp_path / g.DREAD)
    n = N_STUDIES // 3
    folds = {k: dict(train=[u for u in uids if u not in uids[n * k:n * k + n]], val=uids[n * k:n * k + n]) for k in range(3)}
    proto = tmp_path / g.PROTOCOL; spec = {}
    for k, v in folds.items():
        (proto / f'fold{k}').mkdir(parents=True)
        (proto / f'fold{k}' / 'ordered-ids.json').write_text(json.dumps(v), encoding='utf-8')
        spec[str(k)] = dict(ordered_ids=dict(path=f'{g.PROTOCOL}/fold{k}/ordered-ids.json', train_sha256=g.uid_hash(v['train']), val_sha256=g.uid_hash(v['val'])),
                            n_train=len(v['train']), n_val=n, steps=20, max_fp16_skips=0, tag=g.tag_for(k), pos_weight=[1.0] * 12)
    manifest = dict(schema=g.SCHEMA, seed=43, common_args=e107.EXPECTED_RECIPE, folds=spec, initial_state_sha256='init43',
                    source_bindings={g.SHARED_RUNNER: 'runner-sha', g.TOOL: 'guard-sha'}, tags=dict(prefix=g.TAG_PREFIX),
                    paths=dict(corpus='storage/cache/x', labels=g.DREAD, eval_labels=g.DREAD),
                    label_set=dict(name='dread', path=g.DREAD, sha256=g.sha256(tmp_path / g.DREAD), semantic_sha256=semantic))
    (proto / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return tmp_path, folds


def _perfect(uids):
    return {u: (0.9 if int(u[1:]) % 2 else 0.1) for u in uids}


def _fake_run(root, runs, k, folds, tag=None, score=None, args_seed=43, **overrides):
    p = g.read(root / g.PROTOCOL / 'manifest.json')
    tag = tag or g.tag_for(k)
    run = runs / tag; run.mkdir(parents=True)
    val = folds[k]['val']; score = score or _perfect(val)
    frame = pd.DataFrame({UID: val, **{t: [score[u] for u in val] for t in g.T}})
    frame.to_csv(run / 'oof_best.csv', index=False); frame.to_csv(run / 'oof_k36.csv', index=False)
    prov = dict(fold=k, seed=43, label_set='dread', label_semantic_sha256=p['label_set']['semantic_sha256'], label_file_sha256=p['label_set']['sha256'],
                protocol_sha256=g.sha256(root / g.PROTOCOL / 'manifest.json'), eval_ids_ordered_sha256=g.uid_hash(val),
                train_ids_ordered_sha256=g.uid_hash(folds[k]['train']), training_mode='full', checkpoint_selection='fixed-final', arch='resnet50',
                input_policy='center3', norm='radimagenet', remote_relaxed=False, initial_state_sha256='init43', initial_state_matches_registered=True,
                environment_matches_baseline=True, runner_sha256='runner-sha', guard_sha256='guard-sha',
                init_weights=dict(weights_sha256=g.RADIMAGENET_SHA256), actual_pos_weight=p['folds'][str(k)]['pos_weight'])
    prov.update(overrides)
    (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    (run / 'summary.json').write_text(json.dumps(dict(provenance=prov, optimizer_updates=20, skipped_updates=0, selected_epoch_1based=20, train=len(folds[k]['train']),
                                                      val=len(val), args=dict(tag=tag, seed=args_seed), total_seconds=1400)), encoding='utf-8')
    (run / 'history.json').write_text('[]', encoding='utf-8'); (run / 'model_best.pt').write_bytes(b'weights')
    names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv']
    (run / 'output-inventory.json').write_text(json.dumps({n: dict(sha256=g.sha256(run / n)) for n in names}), encoding='utf-8')
    return run


def _write_seed42_oof(root, folds, score, protocol_sha256=g.BASE_MANIFEST_SHA256, seed=42):
    """An exp113-oof-v1 directory (the seed-42 local OOF) with the given per-study scores (all targets equal)."""
    frames = {k: pd.DataFrame({UID: v['val'], **{t: [score[u] for u in v['val']] for t in g.T}}) for k, v in folds.items()}
    result = k99.assemble_and_score(frames, frames, {k: v['val'] for k, v in folds.items()}, root / g.DREAD)
    out = root / e113.OOF_DIR; out.mkdir(parents=True)
    result['oof_k24'].to_csv(out / 'oof_k24.csv', index=False); result['oof_k36'].to_csv(out / 'oof_k36.csv', index=False)
    body = dict(schema='exp113-oof-v1', label_set='dread', protocol_sha256=protocol_sha256, seed=seed, metrics=result['metrics'],
                outputs={f: g.sha256(out / f) for f in ('oof_k24.csv', 'oof_k36.csv')})
    (out / 'result.json').write_text(json.dumps(body), encoding='utf-8')
    return body


def test_assemble_local_runs_into_the_exp118_oof_dir(mini):
    root, folds = mini
    runs = root / 'artifacts' / 'experiments'
    for k in folds:
        _fake_run(root, runs, k, folds)
    body = g.assemble(root=root)
    out = root / g.OOF_DIR
    assert body['schema'] == 'exp118-oof-v1' and body['seed'] == 43 and body['rows'] == N_STUDIES and body['metrics']['k36']['pooled']['macro12'] == 1.0
    assert body['runs_dir'] == 'artifacts/experiments' and body['k36_missing_folds'] == [] and not (root / e113.OOF_DIR).exists()
    assert (out / 'result.md').read_text(encoding='utf-8').startswith('# EXP118 RadImageNet ResNet50 5-fold OOF, seed 43')
    with pytest.raises(SystemExit):
        g.assemble(root=root)  # refuses to overwrite


@pytest.mark.parametrize('damage, message', [
    (dict(seed=42), 'provenance seed'), (dict(initial_state_sha256='init42'), 'initial_state_sha256'),
    (dict(remote_relaxed=True), 'remote_relaxed'), (dict(initial_state_matches_registered=False), 'initial_state_matches_registered'),
    (dict(environment_matches_baseline=False), 'environment_matches_baseline'), (dict(protocol_sha256=g.BASE_MANIFEST_SHA256), 'protocol_sha256'),
    (dict(guard_sha256='exp113-guard'), 'guard_sha256'), (dict(actual_pos_weight=[2.0] * 12), 'pos_weight differs'), ('args_seed', 'seed is not'),
    ('truncate', 'output-inventory')])
def test_verify_run_refuses_seed42_style_or_damaged_runs(mini, damage, message):
    root, folds = mini
    runs = root / 'runs'
    for k in folds:
        extra = damage if (isinstance(damage, dict) and k == 1) else {}
        _fake_run(root, runs, k, folds, args_seed=42 if (damage == 'args_seed' and k == 1) else 43, **extra)
    run = runs / g.tag_for(1)
    if damage == 'truncate':
        (run / 'model_best.pt').write_bytes(b'wei')
    p = g.read(root / g.PROTOCOL / 'manifest.json')
    problems = g.verify_run(run, p, 1, g.sha256(root / g.PROTOCOL / 'manifest.json'))
    assert problems and any(message in x for x in problems), problems
    with pytest.raises(SystemExit):
        g.assemble(runs_dir=runs, root=root)
    assert not (root / g.OOF_DIR).exists()


def test_exp113_named_runs_are_not_picked_up(mini):
    root, folds = mini
    runs = root / 'runs'
    for k in folds:
        _fake_run(root, runs, k, folds, tag=e113.tag_for(k))
    with pytest.raises(SystemExit):
        g.assemble(runs_dir=runs, root=root)


def test_compare_reports_seed43_minus_seed42_pooled_per_fold_and_calibration(mini):
    root, folds = mini
    for k in folds:
        _fake_run(root, root / 'artifacts' / 'experiments', k, folds)
    g.assemble(root=root)
    uids = [u for v in folds.values() for u in v['val']]
    s42 = _perfect(uids); s42['u11'] = 0.1  # fold 1: one positive tied with all five negatives -> AUC 0.9 for every target
    _write_seed42_oof(root, folds, s42)
    body = g.compare(root=root, draws=40)
    m = body['metrics']['k36']
    assert set(body['metrics']) == {'k24', 'k36'} and set(m['pooled']) == set(mr.REFERENCES)
    for r in mr.REFERENCES:
        f = m['per_fold']
        assert f['fold0'][r]['delta'] == 0.0 and f['fold2'][r]['delta'] == 0.0 and f['fold1'][r]['seed42'] == pytest.approx(0.9)
        assert f['fold1'][r]['delta'] == pytest.approx(0.1) and m['pooled'][r]['candidate'] == 1.0 and m['pooled'][r]['delta'] > 0
        assert m['fold_delta_spread'][r]['max_abs'] == pytest.approx(0.1) and m['fold_delta_spread'][r]['mean'] == pytest.approx(0.1 / 3)
        lo, hi = m['pooled'][r]['ci95']
        assert 0 <= lo <= m['pooled'][r]['delta'] and hi > 0
    c = body['calibration']
    assert c['primary'] == 'k36' and c['seed_delta_dread'] == m['pooled']['dread']['delta'] and c['two_x_abs_seed_delta_dread'] == pytest.approx(2 * c['seed_delta_dread'])
    assert c['rule']['adopts_seed42_over_seed43'] is False and c['prior_seed_pair_dread'] == 0.0021
    assert body['reference_scorer']['module'] == g.REFERENCE_SCORER and body['bootstrap']['draws'] == 40 and body['rows'] == N_STUDIES
    assert body['reference_scorer']['coverage']['consensus']['fraction'] == 1.0
    assert set(m['two_seed_rank_mean_vs_seed42_exploratory']) == set(mr.REFERENCES) and set(m['spearman_by_target']) == set(g.T)
    assert (root / g.COMPARE_DIR / 'result.md').read_text(encoding='utf-8').startswith('# EXP118 seed 43 vs EXP113 seed 42')
    json.loads((root / g.COMPARE_DIR / 'result.json').read_text(encoding='utf-8'))
    with pytest.raises(SystemExit):
        g.compare(root=root, draws=40)  # refuses to overwrite


@pytest.mark.parametrize('kind', ['protocol', 'seed', 'hash'])
def test_compare_refuses_a_foreign_or_changed_seed42_oof(mini, kind):
    root, folds = mini
    for k in folds:
        _fake_run(root, root / 'artifacts' / 'experiments', k, folds)
    g.assemble(root=root)
    uids = [u for v in folds.values() for u in v['val']]
    if kind == 'protocol':
        _write_seed42_oof(root, folds, _perfect(uids), protocol_sha256='other')
    elif kind == 'seed':
        _write_seed42_oof(root, folds, _perfect(uids), seed=43)
    else:
        _write_seed42_oof(root, folds, _perfect(uids))
        (root / e113.OOF_DIR / 'oof_k36.csv').write_text('x', encoding='utf-8')
    with pytest.raises(SystemExit):
        g.compare(root=root, draws=10)
    assert not (root / g.COMPARE_DIR).exists()


# ------------------------------------------------------------------ the shared runner on CPU (subprocess; no CUDA context)
@prepared
@weights
@corpus
@pytest.mark.parametrize('fold', [0, 3])
def test_validate_only_reproduces_the_seed43_state_in_a_fresh_process(fold):
    p = g.read(REAL)
    argv = g.runner_argv(p, fold, 'full') + ['--validate-only']
    r = subprocess.run(argv, cwd=ROOT, env=without_thread_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    out = json.loads(r.stdout)
    spec = p['folds'][str(fold)]
    assert out['status'] == 'VALIDATED_NO_TRAINING' and out['tag'] == g.tag_for(fold) and out['output_exists'] is False
    assert out['initial_state_matches_registered'] is True and out['initial_state_sha256'] == p['initial_state_sha256'] and out['remote_relaxed'] is False
    assert out['initial_state_mismatched_keys'] == [] and out['initial_state_sha256'] != g.read(BASE)['initial_state_sha256']
    assert (out['steps'], out['len_train_loader'], out['max_fp16_skips'], out['train'], out['val']) == (spec['steps'], spec['len_train_loader'],
                                                                                                          spec['max_fp16_skips'], spec['n_train'], spec['n_val'])
    assert out['pos_weight'] == [round(float(x), 6) for x in spec['pos_weight']] and out['label_semantic_sha256'] == p['label_set']['semantic_sha256']


@prepared
@weights
@corpus
def test_cpu_smoke_two_optimizer_updates(tmp_path):
    p = g.read(REAL)
    fold = 1
    argv = g.runner_argv(p, fold, 'smoke', workers=2, output_root=tmp_path, smoke=dict(train=16, val=2, epochs=1))
    assert argv[argv.index('--device') + 1] == 'cpu' and argv[argv.index('--amp') + 1] == 'none' and argv[argv.index('--seed') + 1] == '43'
    r = subprocess.run(argv, cwd=ROOT, env=without_thread_env(), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    run = tmp_path / g.tag_for(fold, 'smoke')
    s = g.read(run / 'summary.json'); prov = s['provenance']
    assert s['optimizer_updates'] == 2 and s['skipped_updates'] == 0 and (s['train'], s['val'], s['mode']) == (16, 2, 'smoke')
    assert len(s['history']) == 1 and np.isfinite(s['history'][0]['loss']) and prov['scheduler_total_steps'] == 2
    assert prov['initial_state_matches_registered'] is True and prov['initial_state_sha256'] == p['initial_state_sha256']
    assert prov['cuda_initialized'] is False and prov['remote_relaxed'] is False and (prov['label_set'], prov['seed'], s['args']['seed']) == ('dread', 43, 43)
    assert prov['init_weights']['weights_sha256'] == g.RADIMAGENET_SHA256 and (prov['input_policy'], prov['norm']) == ('center3', 'radimagenet')
    assert prov['guard_sha256'] == g.sha256(ROOT / g.TOOL) and prov['runner_sha256'] == g.sha256(ROOT / e107.RUNNER)
    assert prov['actual_pos_weight'] == p['folds'][str(fold)]['pos_weight'] and prov['protocol_sha256'] == g.sha256(REAL)
    val = g.read(ROOT / p['folds'][str(fold)]['ordered_ids']['path'])['val'][:2]
    for name in ('oof_best.csv', 'oof_k36.csv'):
        df = pd.read_csv(run / name, dtype={UID: str})
        assert df[UID].tolist() == val and list(df.columns) == [UID] + g.T and np.isfinite(df[g.T].to_numpy()).all()
    inv = g.read(run / 'output-inventory.json')
    assert all(g.sha256(run / n) == v['sha256'] for n, v in inv.items())
