"""CPU-only packaging/orchestration/protocol tests; no model or CUDA imports."""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('exp141_test_tool', ROOT / 'tools/exp141_kaggle_ctx.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


@pytest.fixture(autouse=True)
def prohibit_cuda(monkeypatch):
    # Any accidental import of torch during these tiny tests fails immediately.
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError('CPU packaging tests must not import/use torch: ' + name)
    monkeypatch.setitem(sys.modules, 'torch', Forbidden())


def dump(path, obj):
    g.write(path, obj)
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_current_registered_manifests_and_f0_invariants():
    protocols, bindings = g.source_contract(ROOT)
    assert set(protocols) == {42, 43}
    for seed, p in protocols.items():
        f = p['folds']['0']
        assert (f['n_train'], f['n_val'], f['steps'], f['len_train_loader']) == (3455, 888, 8620, 431)
        assert p['common_args']['epochs'] == p['common_args']['fixed_epoch'] == 20
        assert (p['common_args']['k'], p['common_args']['k_eval'], p['extra_eval']['k_eval']) == (12, 24, 36)
        assert p['common_args']['bs'] == 8 and p['common_args']['grad_ckpt'] is False
        assert p['folds']['0']['tag'] == g.tag(seed)
        assert g.sha(ROOT / g.MANIFESTS[seed]) == g.REGISTERED_SHA[seed]
    assert g.RUNNER in bindings and g.GUARD in bindings


def test_manifest_tamper_fails_closed(tmp_path):
    for name in [g.REGISTRY, *g.MANIFESTS.values()]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    path = tmp_path / g.MANIFESTS[43]
    body = g.read(path)
    body['common_args']['epochs'] = 19
    dump(path, body)
    with pytest.raises(ValueError, match='manifest hash'):
        g.source_contract(tmp_path)


def test_exact_argv_and_separate_visible_devices():
    p = g.read(ROOT / g.MANIFESTS[42])
    argv = g.runner_args(p, ROOT, Path('/kaggle/working'), Path('/weights/ResNet50.pt'))
    flags = {argv[i]: argv[i + 1] for i in range(0, len(argv), 2)}
    for key, value in p['common_args'].items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            assert flag not in flags  # registered grad_ckpt=False
        else:
            assert flags[flag] == str(value)
    assert flags['--workers'] == '2' and flags['--fold'] == '0'
    assert flags['--output-root'] == str(Path('/kaggle/working'))
    first, second = [g.child_environment(s, ROOT, {'CUDA_VISIBLE_DEVICES': '9', 'PYTHONPATH': 'bad'}) for s in g.SEEDS]
    assert first['CUDA_VISIBLE_DEVICES'] == '0' and second['CUDA_VISIBLE_DEVICES'] == '1'
    assert first['KNEE_REMOTE_RELAX_ENV'] == second['KNEE_REMOTE_RELAX_ENV'] == '1'
    assert 'PYTHONPATH' not in first and all(first[k] == '1' for k in g.THREADS)
    assert g.WORKERS * 2 == 4


@pytest.mark.parametrize('name', ['../oops', '/outside', 'C:/bad', 'bad\\name', 'x/../bad', './bad', ''])
def test_unsafe_paths_rejected(name):
    with pytest.raises(ValueError, match='Unsafe'):
        g.safe_relative(name)


def test_bundle_hash_and_content_tamper(tmp_path, monkeypatch):
    path = tmp_path / 'training/src/tiny.py'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'x = 1\n')
    receipt = dict(schema=g.SCHEMA, provenance_flag='kaggle_t4', files={
        'training/src/tiny.py': dict(sha256=g.sha(path), bytes=path.stat().st_size)})
    dump(tmp_path / g.RECEIPT, receipt)
    expected = g.sha(tmp_path / g.RECEIPT)
    monkeypatch.setattr(g, 'source_contract', lambda root: ({}, {}))
    assert g.verify_bundle(tmp_path, expected) == receipt
    path.write_bytes(b'x = 2\n')
    with pytest.raises(ValueError, match='Bundle file hash mismatch'):
        g.verify_bundle(tmp_path, expected)
    with pytest.raises(ValueError, match='receipt hash mismatch'):
        g.verify_bundle(tmp_path, '0' * 64)


def test_corpus_hashes_small_fakes(tmp_path):
    chunks = {'ids.npy': b'ids', 'masks.npy': b'mask', 'vols_shard00.npy': b'pixels'}
    for n, b in chunks.items():
        (tmp_path / n).write_bytes(b)
    manifest = dict(ids_sha256=digest(b'ids'), masks_sha256=digest(b'mask'),
        shards=[dict(file='vols_shard00.npy', bytes=6, done=True, sha256=digest(b'pixels'))])
    dump(tmp_path / 'manifest.json', manifest)
    receipt = dict(external_assets=dict(corpus_manifest=manifest, corpus_manifest_sha256=g.sha(tmp_path / 'manifest.json')))
    assert g.verify_corpus(tmp_path, receipt)['full_shards_verified']
    (tmp_path / 'vols_shard00.npy').write_bytes(b'pixelt')
    with pytest.raises(ValueError, match='Corpus hash mismatch'):
        g.verify_corpus(tmp_path, receipt)
    (tmp_path / 'vols_shard00.npy').write_bytes(b'pixels')
    (tmp_path / 'ids.npy').write_bytes(b'bad')
    with pytest.raises(ValueError, match='ids.npy'):
        g.verify_corpus(tmp_path, receipt, full=False)


class FakeProcess:
    def __init__(self, responses):
        self.responses = list(responses)
        self.returncode = None
        self.pid = 123

    def poll(self):
        if self.returncode is None and self.responses:
            self.returncode = self.responses.pop(0)
        return self.returncode

    def wait(self, timeout):
        assert self.returncode is not None
        return self.returncode


def test_two_children_launched_before_polling():
    launched = []
    class Child(FakeProcess):
        def poll(self):
            assert len(launched) == 2
            return super().poll()
    def popen(cmd, **kwargs):
        launched.append((cmd, kwargs))
        return Child([0])
    reason, codes = g.supervise([(['seed42'], {'env': {'CUDA_VISIBLE_DEVICES': '0'}}),
                                (['seed43'], {'env': {'CUDA_VISIBLE_DEVICES': '1'}})],
                               100, popen=popen, clock=lambda: 0, sleep=lambda s: None)
    assert reason is None and codes == [0, 0]
    assert [x[1]['env']['CUDA_VISIBLE_DEVICES'] for x in launched] == ['0', '1']


@pytest.mark.parametrize('failure', [False, True])
def test_deadline_and_failed_sibling_stop_process_groups(failure):
    children, stopped = [], []
    now = [0.]
    def popen(cmd, **kwargs):
        p = FakeProcess([7] if failure and not children else [None] * 10)
        children.append(p)
        return p
    def stop(p, hard):
        if p.poll() is None:
            stopped.append((children.index(p), hard))
            p.returncode = -15
    def sleep(seconds):
        now[0] += seconds
    reason, codes = g.supervise([(['a'], {}), (['b'], {})], 5,
        popen=popen, clock=lambda: now[0], sleep=sleep, stopper=stop)
    assert reason == ('child_failure' if failure else 'wall_clock_guard')
    assert len(children) == 2
    assert (1, False) in stopped
    if not failure:
        assert (0, False) in stopped


def test_partial_launch_exception_cleans_started_child():
    children = []
    def popen(cmd, **kwargs):
        if children:
            raise OSError('fake launch error')
        p = FakeProcess([None] * 5)
        children.append(p)
        return p
    def stop(p, hard):
        p.returncode = -15
    with pytest.raises(OSError, match='fake launch'):
        g.supervise([(['a'], {}), (['b'], {})], 100, popen=popen,
                    clock=lambda: 0, sleep=lambda _: None, stopper=stop)
    assert children[0].returncode == -15


def test_runtime_and_no_local_execution():
    good = dict(python='3.13.12', torch='2.11.0+cu128', timm='1.0.29', gpu='Tesla T4')
    g.verify_runtime(good, 1)
    for bad, count in [({**good, 'gpu': '4070 Ti'}, 1), (good, 2), ({**good, 'python': '3.12.6'}, 1),
                       ({**good, 'torch': '2.5.1'}, 1)]:
        with pytest.raises(ValueError):
            g.verify_runtime(bad, count)
    with pytest.raises(ValueError, match='Kaggle-only'):
        g.launch('fake', 0)
    with pytest.raises(ValueError, match='Kaggle-only'):
        g.worker(42, 'fake', Path('fake'), 0)


@pytest.fixture
def tiny_run(tmp_path):
    p = copy.deepcopy(g.read(ROOT / g.MANIFESTS[42]))
    p['common_args']['epochs'] = p['common_args']['fixed_epoch'] = 2
    f = p['folds']['0']
    f.update(n_train=16, n_val=2, steps=4, max_fp16_skips=0)
    ids = ['tiny0', 'tiny1']
    f['ordered_ids']['path'] = 'ids.json'
    dump(tmp_path / 'ids.json', {'val': ids})
    run = tmp_path / g.tag(42)
    run.mkdir()
    environment = dict(python='3.13.12', torch='2.11.0+cu128', timm='1.0.29', gpu='Tesla T4',
        cuda='12.8', cudnn=91002, numpy='2.3.3', pydicom='3.0.1', cv2='4.12.0', torchvision='0.26.0')
    prov = dict(fold=0, seed=42, training_mode='full', checkpoint_selection='fixed-final',
        protocol_sha256='protocol', label_set='dread', label_file_sha256=p['label_set']['sha256'],
        label_semantic_sha256=p['label_set']['semantic_sha256'], arch='resnet50',
        train_ids_ordered_sha256=f['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=f['ordered_ids']['val_sha256'],
        initial_state_sha256=p['initial_state_sha256'], initial_state_matches_registered=True,
        environment_matches_baseline=False, remote_relaxed=True, input_policy='center3', norm='radimagenet',
        runner_sha256=p['source_bindings'][g.RUNNER], guard_sha256=p['source_bindings'][g.GUARD],
        k_eval=24, k_eval_extra=36, exp139_variant='CTX', exp139_variant_config=p['variant_config'],
        exp139_adapter_initial_state=p['adapter_initial_state'], exp139_augmentation_config=p['augmentation_config'],
        provenance_flag='kaggle_t4', exp141_bundle_sha256='receipt', exp141_adapter_sha256='adapter',
        bit_parity_claimed=False, exp141_docker_image=g.DOCKER, exp141_cuda_visible_devices='0',
        exp139_code_sha256={n: p['source_bindings'][n] for n in (g.GUARD, g.RUNNER,
            'training/src/raptor_exp139.py', 'training/src/raptor_exp127_aug1.py',
            'training/src/raptor_exp129.py', 'training/src/raptor_exp129_late.py')},
        actual_pos_weight=f['pos_weight'], init_weights={'weights_sha256': g.WEIGHTS_SHA},
        environment={k: environment[k] for k in p['environment']}, full_environment=environment)
    history = [dict(epoch=0), dict(epoch=1)]
    summary = dict(provenance=prov, provenance_flag='kaggle_t4', args={**p['common_args'],
        'seed': 42, 'variant': 'CTX', 'tag': g.tag(42), 'workers': g.WORKERS}, mode='full',
        selected_epoch_1based=2, train=16, val=2, optimizer_updates=4, skipped_updates=0, history=history)
    receipt = {k: prov[k] for k in ('provenance_flag', 'exp141_bundle_sha256', 'exp141_adapter_sha256',
        'bit_parity_claimed', 'exp141_docker_image', 'exp141_cuda_visible_devices', 'protocol_sha256')}
    receipt.update(schema=g.SCHEMA, state='COMPLETE')
    dump(run / 'provenance.json', prov)
    dump(run / 'summary.json', summary)
    dump(run / 'history.json', history)
    dump(run / 'exp141-run.json', receipt)
    (run / 'model_best.pt').write_bytes(b'tiny fake model, never deserialized')
    for name in ('oof_best.csv', 'oof_k36.csv'):
        with (run / name).open('w', encoding='utf-8', newline='') as stream:
            w = csv.writer(stream)
            w.writerow([g.UID, *g.TARGETS])
            for uid in ids:
                w.writerow([uid, *([.5] * 12)])
    def inventory():
        dump(run / 'output-inventory.json', {n: dict(sha256=g.sha(run / n), bytes=(run / n).stat().st_size) for n in g.REQUIRED_OUTPUTS})
    inventory()
    return run, p, tmp_path, inventory


def test_complete_remote_validation_and_reject_local_spoof(tiny_run):
    run, p, root, inventory = tiny_run
    assert len(g.verify_remote_run(run, p, 'protocol', 'receipt', 'adapter', root)) == 8
    prov = g.read(run / 'provenance.json')
    prov.update(environment_matches_baseline=True, remote_relaxed=False)
    dump(run / 'provenance.json', prov)
    inventory()
    with pytest.raises(ValueError, match='environment_matches_baseline'):
        g.verify_remote_run(run, p, 'protocol', 'receipt', 'adapter', root)


@pytest.mark.parametrize('tamper', ['checkpoint', 'steps', 'epochs', 'pos_weight', 'protocol', 'initial_state', 'seed_gpu', 'uid', 'nan'])
def test_remote_fail_closed(tiny_run, tamper):
    run, p, root, inventory = tiny_run
    if tamper == 'checkpoint':
        (run / 'model_best.pt').write_bytes(b'changed')
    elif tamper in ('steps', 'epochs'):
        s = g.read(run / 'summary.json')
        s['optimizer_updates' if tamper == 'steps' else 'selected_epoch_1based'] = 1
        dump(run / 'summary.json', s)
        inventory()
    elif tamper in ('pos_weight', 'protocol', 'initial_state', 'seed_gpu'):
        prov = g.read(run / 'provenance.json')
        key = {'pos_weight': 'actual_pos_weight', 'protocol': 'protocol_sha256',
               'initial_state': 'initial_state_sha256', 'seed_gpu': 'exp141_cuda_visible_devices'}[tamper]
        prov[key] = [1.] * 12 if tamper == 'pos_weight' else 'bad'
        dump(run / 'provenance.json', prov)
        s = g.read(run / 'summary.json'); s['provenance'] = prov
        dump(run / 'summary.json', s)
        inventory()
    else:
        file = run / 'oof_k36.csv'
        text = file.read_text(encoding='utf-8')
        text = text.replace('tiny1', 'tiny0') if tamper == 'uid' else text.replace('0.5', 'nan', 1)
        file.write_text(text, encoding='utf-8')
        inventory()
    with pytest.raises(ValueError):
        g.verify_remote_run(run, p, 'protocol', 'receipt', 'adapter', root)


def test_notebook_compilation_budget_and_offline_metadata():
    code = g.notebook_source('archive-hash', 'receipt-hash')
    compile(code, '<notebook>', 'exec')
    assert 'CUDA_VISIBLE_DEVICES' not in code  # visibility assigned separately by supervisor
    assert 'pip install' not in code.replace('No pip installs', '')
    assert 'timeout=max(1,' in code and 'STARTED = time.time()' in code
    assert g.MAX_WALL_SECONDS + 60 < 11 * 3600
    # B: the user granted 3–4 extra Kaggle GPU hours at 21:37/30; today's quota moved at 1x session wall time for 2xT4 runs
    assert (g.MAX_WALL_SECONDS + 60) / 3600 <= 4.0
    # Original runner is invoked, not rendered/replaced: optimizer/training source immutable.
    tree = ast.parse((ROOT / g.TOOL).read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    code_worker = ast.get_source_segment((ROOT / g.TOOL).read_text(encoding='utf-8'), worker)
    assert 'entry.main(runner_args(' in code_worker and 'strict_initial_hashes' in code_worker
    assert 'opt.step' not in code_worker and '.replace(' not in code_worker


def test_screen_adapter_preserves_calculations_marks_confounded_and_restores(tmp_path, monkeypatch):
    import types
    guard = types.ModuleType('tools.exp139_aug1_variants')
    tiers = {'tier_a': {'screen_verdict': 'THRESHOLDS_MET', 'adoption_authorized': False}}
    guard.metric_contrast = lambda *args, **kwargs: dict(tiers=copy.deepcopy(tiers),
        per_seed_tiers={'42': copy.deepcopy(tiers), '43': copy.deepcopy(tiers)}, delta=.123)
    guard.verify_run = lambda *args, **kwargs: {'local_control': 'checked'}
    guard.screen_dir = lambda variant: 'local-screen'
    guard.screen_markdown = lambda body: 'matched local F0 screen'
    guard.forbid_cuda_init = lambda: None
    original = (guard.verify_run, guard.screen_dir, guard.metric_contrast, guard.screen_markdown)
    captured = []
    def frozen_screen(variant, root):
        assert variant == 'CTX'
        assert guard.verify_run(None, None, None, candidate=False) == {'local_control': 'checked'}
        metric = guard.metric_contrast()
        captured.append(metric)
        assert 'remote F0 descriptive screen' in guard.screen_markdown({})
        return {'metrics': metric}
    guard.screen = frozen_screen
    package = types.ModuleType('tools')
    package.exp139_aug1_variants = guard
    monkeypatch.setitem(sys.modules, 'tools', package)
    monkeypatch.setitem(sys.modules, 'tools.exp139_aug1_variants', guard)
    monkeypatch.setattr(g, 'verify_stage', lambda *a: {'receipt_sha256': 'receipt'})
    body = g.screen(tmp_path, tmp_path)
    assert body['provenance_flag'] == 'kaggle_t4'
    assert captured[0]['delta'] == .123
    assert captured[0]['tiers']['tier_a']['screen_verdict'] == 'DESCRIPTIVE_ONLY_REMOTE'
    assert captured[0]['per_seed_tiers']['43']['tier_a']['screen_verdict'] == 'DESCRIPTIVE_ONLY_REMOTE'
    assert (guard.verify_run, guard.screen_dir, guard.metric_contrast, guard.screen_markdown) == original


def test_stage_artifacts_if_present():
    if not g.DEFAULT_STAGE.exists():
        pytest.skip('stage is tested after creation')
    g.verify_stage()
    assert g.read(g.DEFAULT_STAGE / 'code-dataset/dataset-metadata.json')['isPrivate'] is True
    meta = g.read(g.DEFAULT_STAGE / 'kernel-metadata.json')
    assert meta['is_private'] and not meta['enable_internet'] and meta['enable_gpu']
    assert meta['machine_shape'] == 'NvidiaTeslaT4' and meta['docker_image'] == g.DOCKER
    assert meta['competition_sources'] == ['rsna-knee-abnormality-detection']
    assert meta['dataset_sources'] == [g.CORPUS_DATASET, g.WEIGHTS_DATASET, g.CODE_DATASET]
    with zipfile.ZipFile(g.DEFAULT_STAGE / 'code-dataset/exp141-code.zip') as z:
        names = z.namelist()
        assert 'tools/__init__.py' in names
        assert not any(n.startswith('.local/') or n.startswith('storage/competition/') for n in names)
        assert all(n.endswith('.csv') is False or 'report' not in Path(n).name.lower() for n in names)
        for seed in g.SEEDS:
            assert digest(z.read(g.MANIFESTS[seed])) == g.REGISTERED_SHA[seed]
        assert z.read(g.RUNNER) == (ROOT / g.RUNNER).read_bytes()
    for cell in g.read(g.DEFAULT_STAGE / 'release.ipynb')['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'release.ipynb', 'exec')
