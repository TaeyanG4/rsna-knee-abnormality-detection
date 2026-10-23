"""EXP179 CPU synthetic contracts; never reads reports, labels or GPU inputs."""
from __future__ import annotations
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/src'))
from tools import exp179_multiscale as e
from raptor_common import SLOTS64, select_centers, study_rng
from raptor_radimagenet import center3_windows_from
from raptor_exp127_aug1 import AugmentedSliceWindows
from raptor_exp179 import ScaleSliceWindows, scale_windows, detail_indices


@pytest.fixture
def volume():
    # Coordinate-coded central detail and distinct slice IDs; no real images.
    y, x = np.indices((336, 336))
    return np.stack([((x * 3 + y * 5 + c * 7) % 256).astype(np.uint8) for c in range(64)])


def test_exact_crop_geometry_and_unchanged_context(volume):
    cs = list(range(1, 13))
    actual = scale_windows(volume, cs, 224)
    base = center3_windows_from(volume, cs, 224)
    detail = center3_windows_from(volume[:, 48:288, 48:288], cs, 224)
    assert 336 * 100 / 140 == 240
    assert (48 + 287) / 2 == (0 + 335) / 2
    np.testing.assert_array_equal(actual[::2], base[::2])
    np.testing.assert_array_equal(actual[1::2], detail[1::2])
    assert not np.array_equal(actual[1::2], base[1::2])
    np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
    np.testing.assert_array_equal(actual[:, 1], actual[:, 2])
    np.testing.assert_array_equal(scale_windows(volume, cs, 224, 'context'), base)


@pytest.mark.parametrize('k', [12, 24, 36])
def test_exact_allocation_missing_windows_and_call_count(volume, k):
    assert len(detail_indices(k)) == k // 2
    cs = [-1] * k
    x = scale_windows(volume, cs, 224)
    assert x.shape == (k, 3, 224, 224) and x.dtype == np.float32
    assert np.count_nonzero(x) == 0
    import torch
    calls = []
    class Encoder(torch.nn.Module):
        def forward(self, images):
            calls.append(tuple(images.shape))
            return images.mean((2, 3))
    Encoder()(torch.from_numpy(x)[None].flatten(0, 1))
    assert calls == [(k, 3, 224, 224)]


@pytest.mark.parametrize('seed', [42, 43])
@pytest.mark.parametrize('train,k', [(True, 12), (False, 24), (False, 36)])
def test_loader_control_bit_exact_and_rng_unchanged(volume, seed, train, k, monkeypatch):
    import torch
    import raptor_exp179 as module
    spec = {'slots': SLOTS64}
    mask = np.ones(64, np.uint8)
    corpus = SimpleNamespace(depth=64, spec=spec, volume=lambda _: volume, mask=lambda _: mask)
    labels = {'synthetic': np.arange(12, dtype=np.float32) / 12}
    args = ('unused', ['synthetic'], labels, 224, k, train, 'imagenet', seed)
    kwargs = dict(window_policy='within-slot-valid', input_policy='center3')
    control = AugmentedSliceWindows(*args, **kwargs)
    wrapper = ScaleSliceWindows(*args, crop_policy='context', **kwargs)
    mixed = ScaleSliceWindows(*args, crop_policy='mixed', **kwargs)
    for ds in (control, wrapper, mixed):
        ds.corpus = corpus
        ds.set_epoch(3)
    a, b = control[0], wrapper[0]
    assert a[2] == b[2] and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    traces = []
    original = module.augment_windows
    def trace(x, centers, rng, spec):
        before = repr(rng.getstate())
        result = original(x, centers, rng, spec)
        traces.append((list(centers), before, repr(rng.getstate())))
        return result
    monkeypatch.setattr(module, 'augment_windows', trace)
    wrapper[0]; candidate = mixed[0]
    if train:
        assert len(traces) == 2 and traces[0] == traces[1]
    assert candidate[0].shape == a[0].shape
    assert not torch.equal(candidate[0], a[0])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('k,res,shape', [(11,224,(64,336,336)), (12,256,(64,336,336)),
                                       (12,224,(96,336,336)), (12,224,(64,224,224))])
def test_wrong_geometry_is_rejected(k, res, shape):
    with pytest.raises(ValueError):
        scale_windows(np.zeros(shape, np.uint8), [1]*k, res)


def test_runner_derivation_and_guard_contract():
    source = e.runner_source()
    compile(source, e.RUNNER, 'exec')
    assert 'CenterSliceWindows = guard.dataset_factory(a.tag)' in source
    assert 'guard.evaluate_flip_tta' in source and 'guard.stamp_outputs(out, protocol)' in source
    assert 'if not init_matches:' in source
    assert 'from tools import exp179_multiscale as guard' in source
    import ast
    names = {n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == 'guard'}
    assert all(hasattr(e, name) for name in names)


def test_queue_seed_protocol_and_budget():
    assert list(e.ARMS) == ['C43', 'M42', 'M43']
    assert e.ARM_CAP * 3 <= e.TOTAL_CAP == 14400
    for job in e.queue():
        assert not job['continue_on_failure']
    for arm, (seed, _) in e.ARMS.items():
        args = e.runner_args(arm)
        assert args[args.index('--seed')+1] == str(seed)
        assert args[args.index('--tag')+1] == e.tag(arm)
        assert args[args.index('--fold')+1] == '0'


def test_environment_hostname_lives_in_summary_and_all_versions_match():
    versions = {'torch':'a', 'cuda':'b', 'cudnn':90, 'timm':'c', 'numpy':'d',
                'cv2':'e', 'pydicom':'f'}
    control = {'provenance': {'environment': dict(versions, gpu='RTX 3090')},
               'summary': {'environment': {'hostname': 'registered-host'}}}
    e.check_environment(control, 'registered-host', 'RTX 3090', versions)
    for key in versions:
        with pytest.raises(ValueError):
            e.check_environment(control, 'registered-host', 'RTX 3090', dict(versions, **{key:'drift'}))
    with pytest.raises(ValueError):
        e.check_environment(control, 'different-host', 'RTX 3090', versions)


def test_idle_guard_allows_only_identified_logger_and_requires_headroom():
    assert e.check_idle_gpu('RTX 3090, 23000, 0', '123\n', {123}) == 'RTX 3090'
    assert e.check_idle_gpu('RTX 3090, 24000, 0', '', set()) == 'RTX 3090'
    with pytest.raises(ValueError): e.check_idle_gpu('RTX 3090, 23000, 0', '124\n', {123})
    with pytest.raises(ValueError): e.check_idle_gpu('RTX 3090, 19000, 0', '123\n', {123})
    with pytest.raises(ValueError): e.check_idle_gpu('RTX 3090, 23000, 70', '123\n', {123})


def test_direct_arm_entry_refuses_prior_unvalidated_fit(tmp_path):
    e.write(tmp_path / 'started-C43.json', {'reserved_seconds':4200})
    with pytest.raises(ValueError, match='unvalidated'):
        e.check_prior_arms(tmp_path, 'M42')
    e.check_prior_arms(tmp_path, 'C43')


def test_exact_existing_s42_control_is_complete_and_bound():
    # Read-only receipt/inventory validation, no labels or prediction rows printed.
    receipt = e.control_receipt(ROOT)
    assert receipt['hashes']['oof_k36_flip_tta.csv'] == e.C42_PRED_SHA
    assert receipt['summary']['optimizer_updates'] == 8620
    assert receipt['summary']['environment']['hostname'] == '44a123d2ba2a'


def test_tta_reuses_wrapper_dataset_and_restores(monkeypatch):
    from tools import exp166_convnext_t as old
    from training.src import raptor_radimagenet as module
    original = module.CenterSliceWindows
    def fake(*args):
        assert module.CenterSliceWindows.keywords == {'crop_policy': 'mixed'}
        raise RuntimeError('probe')
    monkeypatch.setattr(old, 'evaluate_flip_tta', fake)
    with pytest.raises(RuntimeError, match='probe'):
        e.evaluate_flip_tta(None, None, [], {}, SimpleNamespace(tag=e.tag('M42')), None, None, None)
    assert module.CenterSliceWindows is original


def test_stage_plan_has_detachment_exact_paths_and_per_arm_exports():
    text = e.vast_plan()
    assert 'setsid nohup' in text and 'MSYS_NO_PATHCONV=1' in text
    assert '/workspace/exp179' in text and 'max_gpu_seconds=14400' in text
    with pytest.raises(ValueError):
        e.vast_plan('/workspace/../outside')
    with pytest.raises(ValueError):
        e.vast_plan('C:/workspace/exp179')


def test_verification_rejects_missing_pass(tmp_path):
    p = tmp_path / 'review.md'; p.write_text('not approved', encoding='utf-8')
    with pytest.raises(ValueError, match='PASS'):
        e.check_verification(p)


def test_collect_rejects_traversal_before_writing(tmp_path, monkeypatch):
    import io
    import tarfile
    p = tmp_path / 'bad.tar'
    with tarfile.open(p, 'w') as tar:
        info = tarfile.TarInfo('../escape'); info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    monkeypatch.setattr(e, 'validate', lambda _: {})
    with pytest.raises(ValueError, match='unsafe'):
        e.collect(p, tmp_path, tmp_path / 'out', 'M42')
    assert not (tmp_path / 'out').exists()


@pytest.fixture
def package(tmp_path, monkeypatch):
    from tools import exp170_convnext_t_l2 as base
    import shutil
    root = tmp_path / 'project'; root.mkdir()
    out = tmp_path / 'overlay'
    ids = {'train': [f't{i}' for i in range(3455)], 'val': [f'v{i}' for i in range(888)]}
    e.write(root / base.FOLD0_ROSTER, ids)
    args = {'arch': 'convnext_tiny.fb_in22k_ft_in1k', 'res': 224, 'k': 12,
            'epochs': 20, 'fixed_epoch': 20, 'input_policy': 'center3',
            'amp': 'fp16', 'eval_amp': 'fp16'}
    fold = {'tag': 'old', 'ordered_ids': {'path': base.FOLD0_ROSTER,
             'sha256': e.sha(root / base.FOLD0_ROSTER),
             'train_sha256': base.uid_hash(ids['train']), 'val_sha256': base.uid_hash(ids['val'])},
             'steps': 8620, 'n_train': 3455, 'n_val': 888, 'max_fp16_skips': 0,
             'pos_weight': [2.0]*12}
    q = {'seed': 42, 'folds': {'0': fold}, 'tags': {'prefix': 'old'},
         'common_args': args, 'extra_eval': {'k_eval': 36},
         'paths': {'labels': base.LABEL, 'eval_labels': base.LABEL},
         'initial_state_sha256': 'fixed-state', 'label_set': {'sha256': 'fake'}}
    for name in (e.TOOL, e.LOADER, e.SCREEN, e.PROTOCOL, 'tools/exp170_convnext_t_l2.py'):
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, p)
    for name, data in [('config/workspace.toml', 'workspace=true'), (base.LABEL, 'synthetic'),
                       (base.LABEL_PROVENANCE, '{}')]:
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(data)
    monkeypatch.setattr(base, 'LABEL_SHA', e.sha(root / base.LABEL))
    q['label_set']['sha256'] = base.LABEL_SHA
    e.write(root / e.BASE_OVERLAY / 'protocols/fold0.json', q)
    control = {'path': e.C42, 'hashes': {'oof_k36_flip_tta.csv': e.C42_PRED_SHA},
               'summary': {'args': args}, 'provenance': {'initial_state_sha256': 'fixed-state'}}
    monkeypatch.setattr(e, 'control_receipt', lambda _: control)
    monkeypatch.setattr(base, 'validate', lambda *a, **k: {})
    names = base.REQUIRED_MODULES | {base.SOURCE_WEIGHT, base.SOURCE_MANIFEST}
    files = {n: {'sha256': 'a'*64, 'bytes': 1} for n in names}
    files[base.SOURCE_WEIGHT]['sha256'] = base.WEIGHT_SHA
    monkeypatch.setattr(base, '_source_files', lambda _: (files, 'b'*64))
    monkeypatch.setattr(e, 'verify_derivation_sources', lambda *args: None)
    monkeypatch.setattr(e, 'runner_source', lambda _: '# synthetic runner\n')
    note = root / 'review.md'
    note.write_text('Verdict: PASS\n' + '\n'.join(e.sha(root / n) for n in
                    (e.TOOL, e.LOADER, e.SCREEN, e.PROTOCOL)), encoding='utf-8')
    e.prepare(out, note, root)
    return out


def test_prepare_binds_three_arms_and_workspace_without_c42_retraining(package):
    m = e.validate(package)
    assert set(m['arms']) == {'C43','M42','M43'}
    assert (package / 'config/workspace.toml').exists()
    assert (package / e.VERIFICATION).exists()
    assert all(e.read(package / p['protocol'])['initial_state_sha256'] == 'fixed-state'
               for p in m['arms'].values())


@pytest.mark.parametrize('mutation', ['seed', 'policy', 'roster', 'labels', 'steps', 'runner', 'args'])
def test_semantic_contract_rejects_even_rehashed_mutations(package, mutation):
    m = e.read(package / 'manifest.json')
    p = package / m['arms']['M42']['protocol']; q = e.read(p)
    if mutation == 'seed': q['seed'] = 43
    elif mutation == 'policy': q['exp179']['crop_policy'] = 'context'
    elif mutation == 'roster': q['folds']['0']['ordered_ids']['val_sha256'] = 'wrong'
    elif mutation == 'labels': q['paths']['labels'] = 'wrong.csv'
    elif mutation == 'steps': q['folds']['0']['steps'] = 8619
    elif mutation == 'runner': q['runner_sha256'] = 'wrong'
    else: q['common_args']['epochs'] = 19
    e.write(p, q)
    digest = e.sha(p); m['arms']['M42']['sha256'] = digest
    entry = m['files'][m['arms']['M42']['protocol']]
    entry.update(sha256=digest, bytes=p.stat().st_size)
    e.write(package / 'manifest.json', m)
    with pytest.raises(ValueError): e.validate(package)


def test_missing_workspace_or_stale_review_rejected(package):
    (package / 'config/workspace.toml').unlink()
    with pytest.raises(ValueError, match='package drift'): e.validate(package)
