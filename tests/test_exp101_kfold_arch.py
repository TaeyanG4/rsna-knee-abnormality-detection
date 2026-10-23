"""CPU-only checks for EXP101 (tools/exp101_kfold_arch.py, its runner's argument guard, tools/remote_bundle_v3.py, tools/remote_run_v3.py).

Recipe derivation, EXP099 fold reuse by hash, per-fold step counts, pos_weight, argument validation, run verification and OOF assembly.
No GPU, no training (the runner's --validate-only / smoke paths are exercised separately through the tool CLI).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from tools import exp101_kfold_arch as m  # noqa: E402
from tools import exp099_kfold as k99  # noqa: E402
from tools import remote_bundle_v3 as rb3  # noqa: E402
from tools import remote_run_v3 as rr3  # noqa: E402

UID = m.UID
F99 = ROOT / m.FOLD_PROTOCOL / 'manifest.json'
REAL = ROOT / m.PROTOCOL / 'manifest.json'
prepared = pytest.mark.skipif(not REAL.exists(), reason='EXP101 protocol not prepared')


def runner_module():
    spec = importlib.util.spec_from_file_location('exp101_runner', ROOT / m.RUNNER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def f99():
    return m.read(F99)


@pytest.fixture(scope='module')
def dread():
    labels, semantic = k99._load_labels(ROOT / m.DREAD)
    return labels, semantic


# ------------------------------------------------------------------ recipe
def test_recipe_from_rn_clip_arm_is_the_requested_recipe():
    arm = m.read(ROOT / m.RECIPE_SOURCE)['arms'][m.RECIPE_ARM]
    args = m.recipe_args(arm['expected_args'])
    assert args == m.EXPECTED_RECIPE
    assert args['arch'] == 'resnet34.a1_in1k' and args['epochs'] == args['fixed_epoch'] == 20 and args['grad_ckpt'] is False
    assert args['bb_lr'] == 5e-4 and args['head_lr'] == 1e-3 and args['wd'] == 0.02 and args['amp_init_scale'] == 1024.0
    assert 'workers' not in args and 'select' not in args  # resource arg / replaced by fixed-final
    assert arm['loss_policy'] == 'clipped_positive_term' and arm['expected_args']['seed'] == m.SEED


@pytest.mark.parametrize('change, message', [(dict(fixed_epoch=19), 'fixed-final'), (dict(amp='bf16'), 'FP16'), (dict(eval_amp='none'), 'FP16'),
                                             (dict(accum=2), 'accum'), (dict(device='cpu'), 'CUDA'), (dict(pretrained='none'), 'CUDA')])
def test_recipe_args_rejects(change, message):
    expected = {**m.read(ROOT / m.RECIPE_SOURCE)['arms'][m.RECIPE_ARM]['expected_args'], **change}
    with pytest.raises(ValueError, match=message):
        m.recipe_args(expected)
    missing = dict(m.read(ROOT / m.RECIPE_SOURCE)['arms'][m.RECIPE_ARM]['expected_args']); missing.pop('head_lr')
    with pytest.raises(ValueError, match='head_lr'):
        m.recipe_args(missing)


# ------------------------------------------------------------------ contracts on the EXP099 folds
def test_per_fold_steps_are_floor_ntrain_over_8_times_20(f99):
    n_train = [f99['folds'][str(k)]['n_train'] for k in m.FOLDS]
    assert n_train == [3455, 3479, 3479, 3479, 3480]
    c = [m.fold_contract(n, m.EXPECTED_RECIPE) for n in n_train]
    assert [x['len_train_loader'] for x in c] == [431, 434, 434, 434, 435]
    assert [x['steps'] for x in c] == [8620, 8680, 8680, 8680, 8700] == [(n // 8) * 20 for n in n_train]
    assert [x['max_fp16_skips'] for x in c] == [86, 86, 86, 86, 87]
    assert all(x['scheduler_total_steps'] == x['steps'] and x['eval_batch'] == 4 for x in c)
    assert sum(x['steps'] for x in c) == 43360
    assert c[0]['steps'] == m.read(ROOT / m.RECIPE_SOURCE)['arms'][m.RECIPE_ARM]['expected_attempts']  # fold 0 = G1 train = RN_clip's 8620


def test_pos_weight_per_fold_equals_exp099(f99, dread):
    labels, _ = dread
    vectors = []
    for k in m.FOLDS:
        train = m.read(ROOT / f99['folds'][str(k)]['ordered_ids']['path'])['train']
        pw = m.pos_weight(labels, train)
        assert pw.dtype == np.float32 and np.array_equal(pw, np.asarray(f99['folds'][str(k)]['pos_weight'], np.float32))
        vectors.append(pw)
    assert np.array_equal(vectors[0], np.asarray(m.read(ROOT / m.RECIPE_SOURCE)['arms'][m.RECIPE_ARM]['actual_pos_weight'], np.float32))
    assert not np.array_equal(vectors[1], vectors[2])  # computed per fold, not copied


def test_should_start_fold_budget():
    assert m.should_start_fold(0, [], 11 * 3600) == (True, 1.3 * 3600)
    assert m.should_start_fold(10 * 3600, [], 11 * 3600)[0] is False
    ok, need = m.should_start_fold(9 * 3600, [3000, 4000], 11 * 3600)
    assert ok and need == pytest.approx(5200)
    assert m.should_start_fold(9.6 * 3600, [4000], 11 * 3600)[0] is False


def test_state_hashes_match_runner_rule_and_locate_changes():
    import torch
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    combined, keys = m.state_hashes(net)
    h = hashlib.sha256()
    for key, value in sorted(net.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    assert combined == h.hexdigest() and set(keys) == set(net.state_dict())
    assert m.mismatched_keys(keys, json.loads(json.dumps(keys))) == []  # JSON round trip of the keys file
    with torch.no_grad():
        net[0].bias.add_(1.0)
    _, changed = m.state_hashes(net)
    assert m.mismatched_keys(changed, keys) == ['0.bias']
    assert m.mismatched_keys({k: v for k, v in keys.items() if k != '1.weight'}, keys) == ['1.weight']


def test_tags():
    assert m.tag_for(3) == 'EXP101_RN34_DREAD_F3' and m.tag_for(0, 'smoke') == 'EXP101_RN34_DREAD_F0_SMOKE'


# ------------------------------------------------------------------ the real registration
@prepared
def test_real_manifest_reuses_exp099_folds_by_hash(f99, dread):
    p = m.read(REAL)
    _, semantic = dread
    assert p['schema'] == m.SCHEMA and p['seed'] == 42 and p['common_args'] == m.EXPECTED_RECIPE
    assert p['fold_protocol']['sha256'] == m.sha256(F99) and p['fold_protocol']['folds_csv_sha256'] == f99['folds_csv_sha256'] == m.sha256(ROOT / f99['paths']['folds'])
    assert p['label_set'] == dict(name='dread', path=m.DREAD, sha256=f99['label_sets']['dread']['sha256'], semantic_sha256=semantic)
    for k in m.FOLDS:
        spec, old = p['folds'][str(k)], f99['folds'][str(k)]
        for key in ('ordered_ids', 'groups'):
            assert spec[key] == old[key] and spec[key]['path'].startswith(m.FOLD_PROTOCOL + '/')
            assert m.sha256(ROOT / spec[key]['path']) == spec[key]['sha256']
        assert (spec['n_train'], spec['n_val']) == (old['n_train'], old['n_val']) and spec['pos_weight'] == old['pos_weight']
        assert spec['steps'] == (spec['n_train'] // 8) * 20 and spec['max_fp16_skips'] == spec['steps'] // 100 and spec['tag'] == m.tag_for(k)
    assert p['total_steps'] == 43360
    for path, expected in p['source_bindings'].items():
        assert m.sha256(ROOT / path) == expected, path
    assert {m.RUNNER, m.TOOL, m.FOLD_PROTOCOL + '/manifest.json', m.DREAD, m.PROTOCOL + '/' + m.KEYS_FILE} <= set(p['source_bindings'])


@prepared
def test_real_initial_state_registration():
    p = m.read(REAL)
    keys = m.read(ROOT / p['initial_state']['keys_file'])
    assert keys['combined_sha256'] == p['initial_state_sha256'] and len(keys['keys']) == p['initial_state']['keys'] > 100
    rn = m.read(ROOT / m.RECIPE_SOURCE)['arms'][m.RECIPE_ARM]['initial_state_sha256']
    assert p['initial_state']['recipe_arm_initial_state_sha256'] == rn
    assert p['initial_state']['equals_recipe_arm_initial_state'] is (p['initial_state_sha256'] == rn)
    assert p['initial_state_sha256'] != m.read(F99)['initial_state_sha256']  # not the CoAtNet B8 state


# ------------------------------------------------------------------ argument validation through the runner's own parser
@pytest.fixture()
def guard_env(monkeypatch):
    for k, v in {**m.THREAD_ENV, 'HF_HUB_OFFLINE': '1'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv('KNEE_REMOTE_RELAX_ENV', raising=False)
    return monkeypatch


def parsed(argv):
    return runner_module().build_parser().parse_args(argv[2:])


@prepared
@pytest.mark.parametrize('fold', [0, 4])
def test_runner_argv_round_trip_validates(guard_env, fold):
    p = m.read(REAL)
    args = parsed(m.runner_argv(p, fold))
    for k, v in p['common_args'].items():
        assert getattr(args, k) == v and type(getattr(args, k)) is type(v), k
    assert args.workers == 4 and args.seed == 42 and args.tag == m.tag_for(fold) and args.output_root is None
    got, spec, ordered, out_root = m.validate_arguments(args, ROOT)
    assert spec['steps'] == (spec['n_train'] // 8) * 20 and len(ordered['train']) == spec['n_train']
    assert out_root == (ROOT / 'artifacts' / 'experiments').resolve()


def _mutate(argv, flag, value=None, drop=False):
    argv = list(argv)
    if drop:
        argv.remove(flag); return argv
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv += [flag] + ([str(value)] if value is not None else [])
    return argv


@prepared
@pytest.mark.parametrize('flag, value, message', [
    ('--bb-lr', '0.0003', 'frozen argument changed: bb_lr'), ('--epochs', '8', 'frozen argument changed: epochs'),
    ('--arch', 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', 'frozen argument changed: arch'), ('--grad-ckpt', None, 'frozen argument changed: grad_ckpt'),
    ('--k-eval', '36', 'frozen argument changed: k_eval'), ('--amp', 'none', 'frozen argument changed: amp'),
    ('--device', 'cpu', 'frozen argument changed: device'), ('--seed', '43', 'seed changed'), ('--workers', '99', 'workers'),
    ('--tag', 'EXP101_RN34_DREAD_F1', 'tag'), ('--labels', 'storage/labels/other.csv', 'input path differs: labels'),
    ('--output-root', '/tmp/elsewhere', 'remote runs')])
def test_validate_arguments_rejects(guard_env, flag, value, message):
    argv = _mutate(m.runner_argv(m.read(REAL), 0), flag, value)
    with pytest.raises(ValueError, match=message):
        m.validate_arguments(parsed(argv), ROOT)


@prepared
def test_validate_arguments_environment_and_fold(guard_env):
    p = m.read(REAL)
    args = parsed(m.runner_argv(p, 0))
    guard_env.setenv('HF_HUB_OFFLINE', '0')
    with pytest.raises(ValueError, match='HF_HUB_OFFLINE'):
        m.validate_arguments(args, ROOT)
    guard_env.setenv('HF_HUB_OFFLINE', '1'); guard_env.delenv('OMP_NUM_THREADS')
    with pytest.raises(ValueError, match='OMP_NUM_THREADS'):
        m.validate_arguments(args, ROOT)
    guard_env.setenv('OMP_NUM_THREADS', '1')
    bad = _mutate(_mutate(m.runner_argv(p, 0), '--fold', '7'), '--tag', 'EXP101_RN34_DREAD_F7')
    with pytest.raises(ValueError, match='unregistered fold'):
        m.validate_arguments(parsed(bad), ROOT)


@prepared
def test_remote_output_root_and_smoke_rules(guard_env, tmp_path):
    p = m.read(REAL)
    remote = parsed(m.runner_argv(p, 2, output_root=tmp_path, workers=4))
    guard_env.setenv('KNEE_REMOTE_RELAX_ENV', '1')
    assert m.validate_arguments(remote, ROOT)[3] == tmp_path.resolve()
    guard_env.delenv('KNEE_REMOTE_RELAX_ENV')
    smoke = parsed(m.runner_argv(p, 0, 'smoke', workers=0, output_root=tmp_path))
    assert (smoke.device, smoke.amp, smoke.eval_amp, smoke.smoke_train, smoke.tag) == ('cpu', 'none', 'none', 24, 'EXP101_RN34_DREAD_F0_SMOKE')
    assert m.validate_arguments(smoke, ROOT)[3] == tmp_path.resolve()
    inside = parsed(m.runner_argv(p, 0, 'smoke', output_root=ROOT / 'artifacts' / 'experiments'))
    with pytest.raises(ValueError, match='outside artifacts'):
        m.validate_arguments(inside, ROOT)
    no_root = parsed(m.runner_argv(p, 0, 'smoke'))
    with pytest.raises(ValueError, match='outside artifacts'):
        m.validate_arguments(no_root, ROOT)
    tiny = parsed(m.runner_argv(p, 0, 'smoke', output_root=tmp_path, smoke=dict(train=4)))
    with pytest.raises(ValueError, match='one batch'):
        m.validate_arguments(tiny, ROOT)
    still_frozen = parsed(_mutate(m.runner_argv(p, 0, 'smoke', output_root=tmp_path), '--bb-lr', '0.001'))
    with pytest.raises(ValueError, match='bb_lr'):
        m.validate_arguments(still_frozen, ROOT)


# ------------------------------------------------------------------ run verification and assembly (synthetic workspace, 3 folds)
@pytest.fixture()
def mini(tmp_path):
    uids = [f'u{i:02d}' for i in range(12)]
    y = np.array([i % 2 for i in range(12)])
    (tmp_path / Path(m.DREAD).parent).mkdir(parents=True)
    pd.DataFrame({UID: uids, **{t: np.where(y == 1, 0.95, 0.05) for t in m.T}}).to_csv(tmp_path / m.DREAD, index=False)
    _, semantic = k99._load_labels(tmp_path / m.DREAD)
    folds = {k: dict(train=[u for u in uids if u not in uids[4 * k:4 * k + 4]], val=uids[4 * k:4 * k + 4]) for k in range(3)}
    proto = tmp_path / m.PROTOCOL; spec = {}
    for k, v in folds.items():
        (proto / f'fold{k}').mkdir(parents=True)
        (proto / f'fold{k}' / 'ordered-ids.json').write_text(json.dumps(v), encoding='utf-8')
        spec[str(k)] = dict(ordered_ids=dict(path=f'{m.PROTOCOL}/fold{k}/ordered-ids.json', train_sha256=m.uid_hash(v['train']), val_sha256=m.uid_hash(v['val'])),
                            n_train=8, n_val=4, steps=20, max_fp16_skips=0)
    manifest = dict(schema=m.SCHEMA, common_args=m.EXPECTED_RECIPE, label_set=dict(name='dread', path=m.DREAD, sha256=m.sha256(tmp_path / m.DREAD), semantic_sha256=semantic),
                    folds=spec)
    (proto / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return tmp_path, folds, semantic


def _fake_run(root, runs, k, folds, semantic, flip_k36=True):
    run = runs / m.tag_for(k); run.mkdir(parents=True)
    val = folds[k]['val']; score = [0.9 if int(u[1:]) % 2 else 0.1 for u in val]
    frame = lambda s: pd.DataFrame({UID: val, **{t: s for t in m.T}})  # noqa: E731
    frame(score).to_csv(run / 'oof_best.csv', index=False)
    frame([1 - s for s in score] if flip_k36 else score).to_csv(run / 'oof_k36.csv', index=False)
    prov = dict(fold=k, label_set='dread', label_semantic_sha256=semantic, protocol_sha256=m.sha256(root / m.PROTOCOL / 'manifest.json'),
                eval_ids_ordered_sha256=m.uid_hash(val), train_ids_ordered_sha256=m.uid_hash(folds[k]['train']), training_mode='full',
                checkpoint_selection='fixed-final', initial_state_matches_registered=False, remote_relaxed=True, environment={'gpu': 'Tesla T4'})
    (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    (run / 'summary.json').write_text(json.dumps(dict(provenance=prov, optimizer_updates=20, skipped_updates=0, selected_epoch_1based=20, train=8, val=4)), encoding='utf-8')
    (run / 'history.json').write_text('[]', encoding='utf-8'); (run / 'model_best.pt').write_bytes(b'weights')
    names = ['summary.json', 'provenance.json', 'history.json', 'model_best.pt', 'oof_best.csv', 'oof_k36.csv']
    (run / 'output-inventory.json').write_text(json.dumps({n: dict(sha256=m.sha256(run / n)) for n in names}), encoding='utf-8')
    return run


def test_assemble_downloaded_runs(mini):
    root, folds, semantic = mini
    runs = root / '.local' / 'kaggle-out' / 'knee-run-v3-exp101-rn34-5fold'
    for k in folds:
        _fake_run(root, runs, k, folds, semantic)
    body = m.assemble(runs_dir=runs, root=root)
    out = root / m.OOF_DIR
    k24 = pd.read_csv(out / 'oof_k24.csv', dtype={UID: str})
    assert k24[UID].tolist() == sorted(u for v in folds.values() for u in v['val']) and list(k24.columns) == [UID] + m.T
    assert body['metrics']['k24']['pooled']['macro12'] == 1.0 and body['metrics']['k36']['pooled']['macro12'] == 0.0
    assert body['runs_dir'] == '.local/kaggle-out/knee-run-v3-exp101-rn34-5fold' and body['sources']['1']['checkpoint_sha256'] == m.sha256(runs / m.tag_for(1) / 'model_best.pt')
    assert (out / 'result.md').read_text(encoding='utf-8').startswith('# EXP101 ResNet34 5-fold OOF')
    with pytest.raises(SystemExit):
        m.assemble(runs_dir=runs, root=root)  # refuses to overwrite


@pytest.mark.parametrize('damage, message', [
    ('truncate', 'differs from output-inventory'), ('missing', 'missing oof_k36.csv'), ('updates', 'updates'), ('protocol', 'protocol_sha256'),
    ('fold', 'provenance fold')])
def test_verify_run_and_assemble_refuse_damaged_runs(mini, damage, message):
    root, folds, semantic = mini
    runs = root / 'runs'
    for k in folds:
        _fake_run(root, runs, k, folds, semantic)
    run = runs / m.tag_for(1)
    if damage == 'truncate':
        (run / 'model_best.pt').write_bytes(b'wei')  # e.g. an SSL-dropped download (D-B-015)
    elif damage == 'missing':
        (run / 'oof_k36.csv').unlink()
    else:
        s = m.read(run / 'summary.json'); prov = m.read(run / 'provenance.json')
        if damage == 'updates':
            s['optimizer_updates'] = 19
        elif damage == 'protocol':
            prov['protocol_sha256'] = 'other'
        else:
            prov['fold'] = 2
        (run / 'summary.json').write_text(json.dumps(s), encoding='utf-8'); (run / 'provenance.json').write_text(json.dumps(prov), encoding='utf-8')
    p = m.read(root / m.PROTOCOL / 'manifest.json')
    problems = m.verify_run(run, p, 1, m.sha256(root / m.PROTOCOL / 'manifest.json'))
    assert problems and any(message in x for x in problems), problems
    with pytest.raises(SystemExit):
        m.assemble(runs_dir=runs, root=root)
    assert not (root / m.OOF_DIR).exists()


# ------------------------------------------------------------------ bundle and kernel
def test_kernel_metadata_and_script():
    meta = rb3.kernel_metadata('someone', 'exp101:rn34-5fold')
    assert meta['docker_image'].endswith('@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461') and meta['docker_image'] == m.DOCKER_IMAGE
    assert meta['machine_shape'] == 'NvidiaTeslaT4' and meta['is_private'] and meta['enable_gpu'] and meta['enable_internet'] and not meta['enable_tpu']
    assert meta['dataset_sources'] == ['someone/knee-runners-v3', 'someone/knee-raptor-corpus-336'] and meta['id'] == 'someone/knee-run-v3-exp101-rn34-5fold'
    assert m.KERNEL.split('/')[1] == meta['id'].split('/')[1]
    script = rb3.kernel_script('exp101:rn34-5fold', 'someone')
    compile(script, 'run.py', 'exec')
    assert "'exp101:rn34-5fold'" in script and "'0,1,2,3,4'" in script and '/kaggle/working' in script and 'remote_run_v3.py' in script


@prepared
def test_bundle_files_cover_every_binding_and_no_gold(tmp_path):
    files = rb3.bundle_files(ROOT)
    p = m.read(REAL)
    assert set(rb3.CODE_FILES) <= set(files) and set(p['source_bindings']) <= set(files)
    for v in p['folds'].values():
        assert v['ordered_ids']['path'] in files and v['groups']['path'] in files and files[v['groups']['path']] == v['groups']['sha256']
    assert not any(f.startswith('storage/competition') or f.endswith('train.csv') for f in files)
    report = rb3.build(root=ROOT, package=tmp_path / 'pkg', user='tester')
    bm = m.read(tmp_path / 'pkg' / 'bundle_manifest.json')
    assert bm['protocols']['exp101:rn34-5fold']['sha256'] == m.sha256(REAL) and report['files'] == len(bm['files']) + 1
    for f, h in bm['sha256'].items():
        assert m.sha256(tmp_path / 'pkg' / f) == h
    assert m.read(tmp_path / 'pkg' / 'dataset-metadata.json')['isPrivate'] is True
    assert (tmp_path / 'pkg' / 'config' / 'workspace.toml').is_file()


@prepared
def test_remote_argv_targets_workspace_and_working_dir(tmp_path):
    p = m.read(REAL)
    argv = m.runner_argv(p, 3, 'full', root=tmp_path, python='python3', workers=4, output_root='/kaggle/working')
    assert argv[:2] == ['python3', str(tmp_path / m.RUNNER)] and argv[argv.index('--output-root') + 1] == '/kaggle/working'
    assert argv[argv.index('--corpus') + 1] == str(tmp_path / p['paths']['corpus']) and argv[argv.index('--tag') + 1] == 'EXP101_RN34_DREAD_F3'
    assert '--grad-ckpt' not in argv and argv[argv.index('--epochs') + 1] == '20' and argv[argv.index('--amp') + 1] == 'fp16'
    assert rr3.JOBS['exp101:rn34-5fold']['folds'] == m.FOLDS and rr3.STATUS == 'exp101-status.json'


def test_remote_status_file_is_atomic(tmp_path):
    rr3.write_status(tmp_path, dict(state='TRAINING', folds={'0': dict(state='OK')}))
    s = json.loads((tmp_path / rr3.STATUS).read_text(encoding='utf-8'))
    assert s['state'] == 'TRAINING' and 'updated_at' in s and not (tmp_path / (rr3.STATUS + '.tmp')).exists()
