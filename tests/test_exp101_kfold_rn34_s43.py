"""CPU-only checks for the EXP101 ResNet34 seed-43 registration (tools/exp101_kfold_rn34_s43.py, training/runners/exp101_kfold_rn34_s43.py,
tools/remote_run_v3b.py, tools/remote_bundle_v3b.py).

The seed-43 protocol must equal the seed-42 one (EXP101_KFOLD_RN34_PROTOCOL_V1) in folds, pos_weight, recipe and contract; only the seed, the
initial-state hash and the output names may differ, and the output names must never collide with the seed-42 ones. No GPU, no training.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp101_kfold_rn34_s43 as s  # noqa: E402
from tools import exp101_kfold_arch as b  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
from tools import remote_bundle_v3b as rb3b  # noqa: E402
from tools import remote_run_v3 as rr3  # noqa: E402

UID = s.UID
REAL = ROOT / s.PROTOCOL / 'manifest.json'
BASE = ROOT / s.BASE_PROTOCOL / 'manifest.json'
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP101 seed-43 protocol not prepared')


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def restore_guard_name():
    """The seed-43 runner entry re-points `tools.exp101_kfold_arch`; put the real module back after the test."""
    import tools
    saved_module, saved_attr = sys.modules.get('tools.exp101_kfold_arch'), getattr(tools, 'exp101_kfold_arch', None)
    yield
    sys.modules['tools.exp101_kfold_arch'] = saved_module
    tools.exp101_kfold_arch = saved_attr


@pytest.fixture()
def wrapper(restore_guard_name):
    return load_path('exp101_s43_runner_entry', ROOT / s.RUNNER)


@pytest.fixture()
def guard_env(monkeypatch):
    for k, v in {**s.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    return monkeypatch


# ------------------------------------------------------------------ names: seed 43 never collides with seed 42
def test_seed43_names_are_distinct_from_seed42():
    assert (s.SEED, s.BASE_SEED) == (43, 42) and s.PROTOCOL != b.PROTOCOL and s.OOF_DIR != b.OOF_DIR and s.KERNEL != b.KERNEL
    assert s.RUNNER != b.RUNNER and s.TOOL != b.TOOL and s.BASE_RUNNER == b.RUNNER
    for k in s.FOLDS:
        for mode in ('full', 'smoke'):
            new, old = s.tag_for(k, mode), b.tag_for(k, mode)
            assert new != old and old not in new and new not in old  # `kernels output --file-pattern <tag>` can never pick the other seed
    assert s.tag_for(2) == 'EXP101_RN34_S43_DREAD_F2' and s.tag_for(0, 'smoke') == 'EXP101_RN34_S43_DREAD_F0_SMOKE'
    assert s.OOF_DIR == 'artifacts/experiments/EXP101_RN34_S43_DREAD_OOF' and s.KERNEL.endswith('/knee-run-v3-exp101-rn34-s43-5fold')


def test_seed42_registration_is_the_bound_one():
    assert b.sha256(BASE) == s.BASE_MANIFEST_SHA256
    p42 = b.read(BASE)
    assert p42['seed'] == 42 and p42['initial_state_sha256'].startswith('093be508')


# ------------------------------------------------------------------ the real seed-43 registration
@prepared
def test_seed43_manifest_differs_from_seed42_only_in_seed_init_and_names():
    p, p42 = s.read(REAL), s.read(BASE)
    differing = {k for k in set(p) | set(p42) if p.get(k) != p42.get(k)}
    assert differing <= set(s.CHANGED_FIELDS) and {'seed', 'initial_state_sha256', 'folds', 'tags'} <= differing
    for key in ('schema', 'common_args', 'resource_args', 'smoke_allowed', 'training_code_sha256', 'environment', 'corpus_manifest_sha256', 'excluded_uids',
                'fold_protocol', 'paths', 'label_set', 'total_steps', 'pos_weight_rule', 'contract_rule', 'extra_eval'):
        assert p[key] == p42[key], key
    assert p['seed'] == 43 and p42['seed'] == 42 and p['initial_state_sha256'] != p42['initial_state_sha256']
    for k in s.FOLDS:
        new, old = p['folds'][str(k)], p42['folds'][str(k)]
        assert {x for x in set(new) | set(old) if new.get(x) != old.get(x)} == {'tag'}
        assert new['tag'] == s.tag_for(k) and old['tag'] == b.tag_for(k)
        assert new['pos_weight'] == old['pos_weight'] and new['ordered_ids'] == old['ordered_ids'] and new['groups'] == old['groups']
        assert new['steps'] == old['steps'] == (new['n_train'] // 8) * 20
    assert p['recipe'] == {**p42['recipe'], 'changes': [*p42['recipe']['changes'], s.SEED_CHANGE]}
    assert p['evaluation'] == {**p42['evaluation'], 'oof_dir': s.OOF_DIR}
    assert {k: v for k, v in p['remote'].items() if k not in ('kernel', 'relax_env', 'bundle_dataset', 'remote_entry', 'runner')} == \
           {k: v for k, v in p42['remote'].items() if k not in ('kernel', 'relax_env')}
    assert p['remote']['kernel'] == s.KERNEL and p['remote']['docker_image'] == s.DOCKER_IMAGE
    assert p['tags'] == dict(prefix=s.TAG_PREFIX, pattern=s.TAG_PREFIX + '_F<K>[_SMOKE]', oof_dir=s.OOF_DIR, seed42_prefix=b.TAG_PREFIX)
    assert p['caveats'][:len(p42['caveats'])] == p42['caveats']
    v = p['seed_variant_of']
    assert v['sha256'] == s.BASE_MANIFEST_SHA256 and v['seed'] == 42 and v['initial_state_sha256'] == p42['initial_state_sha256'] and v['oof_dir'] == b.OOF_DIR


@prepared
def test_seed43_bindings_hold_and_extend_the_seed42_bindings():
    p, p42 = s.read(REAL), s.read(BASE)
    assert {k: p['source_bindings'][k] for k in p42['source_bindings']} == p42['source_bindings']
    assert set(p['source_bindings']) - set(p42['source_bindings']) == {s.BASE_PROTOCOL + '/manifest.json', s.RUNNER, s.TOOL, s.PROTOCOL + '/' + s.KEYS_FILE}
    for path, expected in p['source_bindings'].items():
        assert s.sha256(ROOT / path) == expected, path
    assert p['source_bindings'][s.BASE_PROTOCOL + '/manifest.json'] == s.BASE_MANIFEST_SHA256
    assert p['source_bindings'][b.RUNNER] == p42['source_bindings'][b.RUNNER]  # the training loop file is the seed-42 one


@prepared
def test_pos_weight_recomputed_from_labels_equals_seed42():
    p, p42 = s.read(REAL), s.read(BASE)
    labels, _ = k99._load_labels(ROOT / s.DREAD)
    for k in s.FOLDS:
        train = s.read(ROOT / p['folds'][str(k)]['ordered_ids']['path'])['train']
        pw = s.pos_weight(labels, train)
        assert pw.dtype == np.float32 and np.array_equal(pw, np.asarray(p['folds'][str(k)]['pos_weight'], np.float32))
        assert np.array_equal(pw, np.asarray(p42['folds'][str(k)]['pos_weight'], np.float32))


@prepared
def test_seed43_initial_state_registration():
    p, p42 = s.read(REAL), s.read(BASE)
    keys = s.read(ROOT / p['initial_state']['keys_file'])
    keys42 = s.read(ROOT / p42['initial_state']['keys_file'])
    assert keys['seed'] == 43 and keys['combined_sha256'] == p['initial_state_sha256'] and len(keys['keys']) == p['initial_state']['keys'] == 224
    assert p['initial_state_sha256'] != p42['initial_state_sha256'] == keys42['combined_sha256']
    changed = s.mismatched_keys(keys['keys'], keys42['keys'])
    assert changed == p['initial_state']['keys_differing_from_seed42'] and changed and set(keys['keys']) == set(keys42['keys'])
    assert not any(k.startswith('backbone.') for k in changed)  # same pretrained backbone; only the random head differs
    assert p['initial_state']['seed42_initial_state_sha256'] == p42['initial_state_sha256'] and p['initial_state']['computed_on'] == 'cpu'


# ------------------------------------------------------------------ argument validation through the seed-43 runner entry
def parsed(wrapper, argv):
    return wrapper.build_parser().parse_args(argv[2:])


def _mutate(argv, flag, value=None):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag] + ([str(value)] if value is not None else [])
    return argv


@prepared
def test_runner_entry_runs_the_shared_runner_with_the_seed43_guard(wrapper):
    runner = wrapper.load_runner()
    import tools
    assert Path(runner.__file__).resolve() == (ROOT / b.RUNNER).resolve()
    assert sys.modules['tools.exp101_kfold_arch'] is s and tools.exp101_kfold_arch is s
    shared = load_path('exp101_shared_parser', ROOT / b.RUNNER).build_parser()
    assert [a.dest for a in runner.build_parser()._actions] == [a.dest for a in shared._actions]


@prepared
@pytest.mark.parametrize('fold', [0, 4])
def test_runner_argv_round_trip_validates(guard_env, wrapper, fold):
    p = s.read(REAL)
    argv = s.runner_argv(p, fold)
    assert argv[1] == str(ROOT / s.RUNNER) and argv[argv.index('--protocol') + 1] == str(REAL)
    args = parsed(wrapper, argv)
    for k, v in p['common_args'].items():
        assert getattr(args, k) == v and type(getattr(args, k)) is type(v), k
    assert args.workers == 4 and args.seed == 43 and args.tag == s.tag_for(fold) and args.output_root is None
    got, spec, ordered, out_root = s.validate_arguments(args, ROOT)
    assert spec['tag'] == args.tag and len(ordered['train']) == spec['n_train'] and out_root == (ROOT / 'artifacts' / 'experiments').resolve()


@prepared
@pytest.mark.parametrize('flag, value, message', [
    ('--seed', '42', 'seed changed'), ('--tag', 'EXP101_RN34_DREAD_F0', 'tag'), ('--tag', 'EXP101_RN34_S43_DREAD_F1', 'tag'),
    ('--protocol', str(BASE), 'wrong protocol'), ('--bb-lr', '0.0003', 'frozen argument changed: bb_lr'), ('--epochs', '8', 'frozen argument changed: epochs'),
    ('--device', 'cpu', 'frozen argument changed: device'), ('--workers', '99', 'workers'), ('--labels', 'storage/labels/other.csv', 'input path differs: labels'),
    ('--output-root', '/tmp/elsewhere', 'remote runs')])
def test_validate_arguments_rejects(guard_env, wrapper, flag, value, message):
    argv = _mutate(s.runner_argv(s.read(REAL), 0), flag, value)
    with pytest.raises(ValueError, match=message):
        s.validate_arguments(parsed(wrapper, argv), ROOT)


@prepared
def test_seed42_guard_refuses_the_seed43_protocol(guard_env, wrapper):
    args = parsed(wrapper, s.runner_argv(s.read(REAL), 0))
    with pytest.raises(ValueError, match='wrong protocol'):
        b.validate_arguments(args, ROOT)  # why the seed-43 run needs its own entry; the seed-42 job cannot pick it up either


@prepared
def test_remote_and_smoke_rules(guard_env, wrapper, tmp_path):
    p = s.read(REAL)
    remote = parsed(wrapper, s.runner_argv(p, 2, output_root=tmp_path, workers=4))
    guard_env.setenv('KNEE_REMOTE_RELAX_ENV', '1')
    assert s.validate_arguments(remote, ROOT)[3] == tmp_path.resolve()
    guard_env.delenv('KNEE_REMOTE_RELAX_ENV')
    smoke = parsed(wrapper, s.runner_argv(p, 0, 'smoke', workers=0, output_root=tmp_path))
    assert (smoke.device, smoke.amp, smoke.seed, smoke.tag) == ('cpu', 'none', 43, 'EXP101_RN34_S43_DREAD_F0_SMOKE')
    assert s.validate_arguments(smoke, ROOT)[3] == tmp_path.resolve()
    inside = parsed(wrapper, s.runner_argv(p, 0, 'smoke', output_root=ROOT / 'artifacts' / 'experiments'))
    with pytest.raises(ValueError, match='outside artifacts'):
        s.validate_arguments(inside, ROOT)


@prepared
def test_remote_argv_targets_workspace_and_working_dir(tmp_path):
    argv = s.runner_argv(s.read(REAL), 3, 'full', root=tmp_path, python='python3', workers=4, output_root='/kaggle/working')
    assert argv[:2] == ['python3', str(tmp_path / s.RUNNER)] and argv[argv.index('--output-root') + 1] == '/kaggle/working'
    assert argv[argv.index('--tag') + 1] == 'EXP101_RN34_S43_DREAD_F3' and argv[argv.index('--seed') + 1] == '43'
    assert argv[argv.index('--protocol') + 1] == str(tmp_path / s.PROTOCOL / 'manifest.json')


# ------------------------------------------------------------------ run verification and assembly (synthetic workspace, 3 folds)
@pytest.fixture()
def mini(tmp_path):
    uids = [f'u{i:02d}' for i in range(12)]
    y = np.array([i % 2 for i in range(12)])
    (tmp_path / Path(s.DREAD).parent).mkdir(parents=True)
    pd.DataFrame({UID: uids, **{t: np.where(y == 1, 0.95, 0.05) for t in s.T}}).to_csv(tmp_path / s.DREAD, index=False)
    _, semantic = k99._load_labels(tmp_path / s.DREAD)
    folds = {k: dict(train=[u for u in uids if u not in uids[4 * k:4 * k + 4]], val=uids[4 * k:4 * k + 4]) for k in range(3)}
    proto = tmp_path / s.PROTOCOL; spec = {}
    for k, v in folds.items():
        (proto / f'fold{k}').mkdir(parents=True)
        (proto / f'fold{k}' / 'ordered-ids.json').write_text(json.dumps(v), encoding='utf-8')
        spec[str(k)] = dict(ordered_ids=dict(path=f'{s.PROTOCOL}/fold{k}/ordered-ids.json', train_sha256=s.uid_hash(v['train']), val_sha256=s.uid_hash(v['val'])),
                            n_train=8, n_val=4, steps=20, max_fp16_skips=0, tag=s.tag_for(k))
    manifest = dict(schema=s.SCHEMA, seed=43, common_args=s.EXPECTED_RECIPE, folds=spec,
                    label_set=dict(name='dread', path=s.DREAD, sha256=s.sha256(tmp_path / s.DREAD), semantic_sha256=semantic))
    (proto / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return tmp_path, folds, semantic


def _fake_run(root, runs, k, folds, semantic, seed=43, tag=None):
    tag = tag or s.tag_for(k)
    run = runs / tag; run.mkdir(parents=True)
    val = folds[k]['val']; score = [0.9 if int(u[1:]) % 2 else 0.1 for u in val]
    frame = lambda x: pd.DataFrame({UID: val, **{t: x for t in s.T}})  # noqa: E731
    frame(score).to_csv(run / 'oof_best.csv', index=False)
    frame(score).to_csv(run / 'oof_k36.csv', index=False)
    prov = dict(fold=k, seed=seed, label_set='dread', label_semantic_sha256=semantic, protocol_sha256=s.sha256(root / s.PROTOCOL / 'manifest.json'),
                eval_ids_ordered_sha256=s.uid_hash(val), train_ids_ordered_sha256=s.uid_hash(folds[k]['train']), training_mode='full',
                checkpoint_selection='fixed-final', initial_state_sha256='x', initial_state_matches_registered=False, remote_relaxed=True)
    (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    (run / 'summary.json').write_text(json.dumps(dict(provenance=prov, optimizer_updates=20, skipped_updates=0, selected_epoch_1based=20, train=8, val=4,
                                                      args=dict(tag=tag, seed=seed))), encoding='utf-8')
    (run / 'history.json').write_text('[]', encoding='utf-8'); (run / 'model_best.pt').write_bytes(b'weights')
    names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv']
    (run / 'output-inventory.json').write_text(json.dumps({n: dict(sha256=s.sha256(run / n)) for n in names}), encoding='utf-8')
    return run


def test_assemble_seed43_runs_into_its_own_oof_dir(mini):
    root, folds, semantic = mini
    runs = root / '.local' / 'kaggle-out' / s.KERNEL.split('/')[-1]
    for k in folds:
        _fake_run(root, runs, k, folds, semantic)
    body = s.assemble(root=root)  # default runs dir = the fetch destination of the seed-43 kernel
    out = root / s.OOF_DIR
    assert body['rows'] == 12 and body['seed'] == 43 and body['metrics']['k24']['pooled']['macro12'] == 1.0
    assert body['runs_dir'] == '.local/kaggle-out/knee-run-v3-exp101-rn34-s43-5fold' and not (root / b.OOF_DIR).exists()
    assert (out / 'result.md').read_text(encoding='utf-8').startswith('# EXP101 ResNet34 seed 43 5-fold OOF')
    with pytest.raises(SystemExit):
        s.assemble(root=root)  # refuses to overwrite


@pytest.mark.parametrize('damage, message', [('seed42', 'provenance seed'), ('truncate', 'differs from output-inventory'), ('protocol', 'protocol_sha256')])
def test_verify_run_refuses_seed42_or_damaged_runs(mini, damage, message):
    root, folds, semantic = mini
    runs = root / 'runs'
    for k in folds:
        _fake_run(root, runs, k, folds, semantic, seed=42 if (damage == 'seed42' and k == 1) else 43)
    run = runs / s.tag_for(1)
    if damage == 'truncate':
        (run / 'model_best.pt').write_bytes(b'wei')
    elif damage == 'protocol':
        prov = s.read(run / 'provenance.json'); prov['protocol_sha256'] = s.BASE_MANIFEST_SHA256
        (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    problems = s.verify_run(run, s.read(root / s.PROTOCOL / 'manifest.json'), 1, s.sha256(root / s.PROTOCOL / 'manifest.json'))
    assert problems and any(message in x for x in problems), problems
    with pytest.raises(SystemExit):
        s.assemble(runs_dir=runs, root=root)
    assert not (root / s.OOF_DIR).exists()


def test_seed42_named_runs_are_not_picked_up(mini):
    root, folds, semantic = mini
    runs = root / 'runs'
    for k in folds:
        _fake_run(root, runs, k, folds, semantic, tag=b.tag_for(k))  # seed-42 directory names only
    with pytest.raises(SystemExit):
        s.assemble(runs_dir=runs, root=root)


# ------------------------------------------------------------------ bundle, kernel and remote entry
def test_kernel_metadata_script_and_push_without_timeout():
    job = 'exp101:rn34-s43-5fold'
    meta = rb3b.kernel_metadata('someone', job)
    assert meta['docker_image'] == s.DOCKER_IMAGE and meta['docker_image'].endswith('@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461')
    assert meta['machine_shape'] == 'NvidiaTeslaT4' and meta['is_private'] and meta['enable_gpu'] and meta['enable_internet'] and not meta['enable_tpu']
    assert meta['dataset_sources'] == ['someone/knee-runners-v3b', 'someone/knee-raptor-corpus-336'] and meta['id'] == 'someone/knee-run-v3-exp101-rn34-s43-5fold'
    assert meta['id'].split('/')[1] == s.KERNEL.split('/')[1] != b.KERNEL.split('/')[1]
    script = rb3b.kernel_script(job, 'someone')
    compile(script, 'run.py', 'exec')
    assert f"'{job}'" in script and "'0,1,2,3,4'" in script and "'11.0'" in script and 'remote_run_v3b.py' in script and '/kaggle/working' in script
    assert max(len(x) for x in script.splitlines()) < 30000
    argv = rb3b.push_argv('kaggle/packages/X')
    assert '--timeout' not in argv and argv[argv.index('--accelerator') + 1] == 'NvidiaTeslaT4' and Path(argv[argv.index('-p') + 1]).is_absolute()


@prepared
def test_bundle_ships_every_seed43_binding_and_no_gold(tmp_path):
    files = rb3b.bundle_files(ROOT)
    p = s.read(REAL)
    assert set(rb3b.CODE_FILES) <= set(files) and set(p['source_bindings']) <= set(files)
    assert {s.RUNNER, s.TOOL, b.RUNNER, b.TOOL, 'tools/remote_run_v3b.py', 'tools/remote_run_v3.py', s.BASE_PROTOCOL + '/manifest.json',
            s.PROTOCOL + '/manifest.json', s.PROTOCOL + '/' + s.KEYS_FILE, s.BASE_PROTOCOL + '/' + s.KEYS_FILE} <= set(files)
    for v in p['folds'].values():
        assert files[v['groups']['path']] == v['groups']['sha256'] and files[v['ordered_ids']['path']] == v['ordered_ids']['sha256']
    assert not any(f.startswith(('storage/competition', 'storage/gold')) or f.endswith('train.csv') for f in files)
    report = rb3b.build(root=ROOT, package=tmp_path / 'pkg', user='tester')
    bm = s.read(tmp_path / 'pkg' / 'bundle_manifest.json')
    assert bm['protocols'] == {'exp101:rn34-s43-5fold': dict(path=s.PROTOCOL, sha256=s.sha256(REAL))} and report['files'] == len(bm['files']) + 1
    for f, h in bm['sha256'].items():
        assert s.sha256(tmp_path / 'pkg' / f) == h
    meta = s.read(tmp_path / 'pkg' / 'dataset-metadata.json')
    assert meta['isPrivate'] is True and meta['id'] == 'tester/knee-runners-v3b'


def test_remote_entry_adds_the_seed43_job_without_touching_v3():
    entry = load_path('remote_run_v3b_test', ROOT / 'tools' / 'remote_run_v3b.py')
    base = entry.load_base()
    assert base.JOBS['exp101:rn34-s43-5fold'] == dict(tool='exp101_kfold_rn34_s43', folds=s.FOLDS)
    assert base.JOBS['exp101:rn34-5fold'] == rr3.JOBS['exp101:rn34-5fold'] and 'exp101:rn34-s43-5fold' not in rr3.JOBS
    assert base.TOOLS_DATASET == 'knee-runners-v3b' and base.STATUS == 'exp101-status.json' and Path(base.__file__).name == 'remote_run_v3.py'
    tool = s  # the remote entry imports tools.<tool> from the staged workspace; it needs exactly these names
    for name in ('ROOT', 'PROTOCOL', 'read', 'sha256', 'THREAD_ENV', 'tag_for', 'should_start_fold', 'runner_argv', 'verify_run'):
        assert hasattr(tool, name), name
