"""CPU tests for the EXP111 v6 release (v5 + a RadImageNet ResNet50 centre-plane arm kind; H-B-31).

No GPU, no network. Real-data tests (training cache, staged package, generated kernel) skip when the files are absent.
The opt-in smoke (EXP111_REAL_SMOKE=1) loads one real staged RadR50 and one CoAtNet checkpoint on CPU.
"""
import copy
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load('exp111_release_v6_under_test', 'tools/exp111_release_v6.py')
v5 = load('exp104_release_v5_for_v6', 'tools/exp104_release_v5.py')
exp098 = load('exp098_e1_release_for_v6', 'tools/exp098_e1_release.py')
SUPERVISOR = ROOT / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/t4-results-v2/B224-supervisor.json'
ARCH = 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'
RN_ARCH = 'resnet34.a1_in1k'
RAD_ARCH = 'resnet50'
CONTRACT = 'raptor64_336_c140_s02'
KERNELS = ROOT / '.local/exp111-release-v6'
CACHE = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
FOLD0 = ROOT / 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold0/ordered-ids.json'
D = 6.52503319258
TRAINING_SRC = ROOT / 'training/src'
if str(TRAINING_SRC) not in sys.path:   # the training modules (raptor_common/raptor_data/raptor_radimagenet)
    sys.path.append(str(TRAINING_SRC))


# ------------------------------------------------------------------ fixtures
def meta(**over):
    m = {'arch': ARCH, 'res': 224, 'lab': list(release.TARGETS), 'contract': CONTRACT, 'k_eval': 24,
         'norm': 'imagenet', 'window_policy': 'within-slot-valid', 'epoch': 7, 'tag': 'T', 'amp_train': 'fp16',
         'amp_eval': 'fp16'}
    m.update(over)
    return m


def rad_meta(**over):
    return meta(**{'arch': RAD_ARCH, 'norm': 'radimagenet', 'epoch': 19, 'input_policy': 'center3', 'init': 'radimagenet',
                   'init_weights_sha256': release.RADIMAGENET_SHA256, **over})


def write_code(root):
    (Path(root) / release.CODE_DIR).mkdir(parents=True, exist_ok=True)
    for n in release.CODE_FILES:
        shutil.copyfile(TRAINING_SRC / n, Path(root) / release.CODE_DIR / n)
    return {release.CODE_DIR + '/' + n: s for n, s in release.CODE_FILES.items()}


def fake_assets(root, members, dataset=None):
    """members: {name: (bytes, meta)} -> new-assets dir with a v6 assets manifest (checkpoints + input code)."""
    root = Path(root)
    (root / 'checkpoints').mkdir(parents=True)
    checkpoints = {}
    for name, (blob, m) in members.items():
        path = root / 'checkpoints' / (name + '.pt')
        path.write_bytes(blob)
        checkpoints[name] = {'path': 'checkpoints/' + name + '.pt', 'sha256': release.sha(path), 'bytes': len(blob),
                             'meta': m, 'training_code_sha256': {}, 'input_code_sha256': {}, 'source': 'synthetic',
                             'source_sidecars_sha256': {}, 'source_inventory_sha256': None, 'link_mode': 'copy'}
    code_files = write_code(root)
    am = {'schema': release.ASSETS_SCHEMA, 'dataset': dataset or release.NEW_DATASET, 'checkpoints': checkpoints,
          'code': release.code_record(), 'files': {**{c['path']: c['sha256'] for c in checkpoints.values()}, **code_files},
          'dataset_metadata_sha256': '0' * 64}
    (root / release.NEW_MANIFEST).write_text(json.dumps(am), encoding='utf-8')
    return am


def two_family_assets(tmp_path, dataset=None):
    return fake_assets(tmp_path / 'new', {'c0': (b'coat-0' * 100, meta(tag='C0')), 'c1': (b'coat-1' * 100, meta(tag='C1')),
                                          'r0': (b'rad-0' * 100, rad_meta(tag='R0')), 'r1': (b'rad-1' * 100, rad_meta(tag='R1'))},
                       dataset)


def two_arms(am, weights=(1.0, 1.0)):
    return [release.build_arm('coat5', ['c0', 'c1'], 36, am, 0.35, weights[0]),
            release.build_arm('rad5', ['r0', 'r1'], 36, am, 0.25, weights[1])]


def release_manifest(path, arm_list, assets_manifest_sha, **over):
    p = {'schema': release.SCHEMA, 'release': 'synthetic', 'release_source_sha256': release.sha(release.__file__),
         'old_asset_protocol_sha256': release.ASSETS_PROTOCOL_SHA, 'new_assets_manifest_sha256': assets_manifest_sha,
         'new_asset_dataset': release.NEW_DATASET, 'time_policy': release.TIME, 'RSS_bytes': release.RSS,
         'output_bytes': release.OUTPUT, 'prob_clamp': release.PROB_EPS, 'retries': 0, 'arms': arm_list,
         'rank_weights': release.rank_weights(arm_list) if arm_list else {}}
    p.update(over)
    Path(path).write_text(json.dumps(p, indent=1), encoding='utf-8')
    return release.sha(path)


def write_csv(path, header, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def fixture_comp(tmp_path, ids=('1.1', '1.2')):
    comp = Path(tmp_path)
    (comp / 'test_series').mkdir(parents=True)
    write_csv(comp / 'test.csv', [release.UID], [[u] for u in sorted(ids)])
    write_csv(comp / 'sample_submission.csv', [release.UID, *release.TARGETS], [[u, *([0] * 12)] for u in ids])
    write_csv(comp / 'test_series.csv', [release.UID, 'SeriesInstanceUID'], [[u, '2.' + u] for u in ids])
    return comp


def affine(torch, a, b):
    class Affine(torch.nn.Module):
        def forward(self, x):
            return x * a + b
    return Affine()


def tiny(torch, seed, res=8):
    class Tiny(torch.nn.Module):
        """(B, K, 3, r, r) -> (B, 12): window-mean of a linear map, like the attention head's output shape."""
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.w = torch.nn.Parameter(torch.randn(3 * res * res, 12, generator=g) * 0.05)
            self.b = torch.nn.Parameter(torch.randn(12, generator=g))

        def forward(self, x):
            return (x.flatten(2) @ self.w).mean(1) + self.b
    return Tiny()


def frozen_runtime(monkeypatch):
    """The serving copy of raptor_runtime from the frozen EXP089 assets (its raptor_common is hash-equal to training's)."""
    src = release.OLD_LOCAL / 'training/src'
    if not (src / 'raptor_runtime.py').is_file():
        pytest.skip('frozen EXP089 assets not present')
    protocol = json.loads((release.OLD_LOCAL / release.OLD_PROTOCOL).read_text(encoding='utf-8'))
    assert release.sha(src / 'raptor_runtime.py') == protocol['files']['training/src/raptor_runtime.py']
    common = __import__('raptor_common')
    assert release.sha(common.__file__) == protocol['files']['training/src/raptor_common.py']
    return load('frozen_raptor_runtime_for_v6', str((src / 'raptor_runtime.py').relative_to(ROOT)))


def input_modules():
    """raptor_radimagenet/raptor_data as imported by center3_prepare_input must be the shipped (hash-pinned) bytes."""
    rad = __import__('raptor_radimagenet')
    data = __import__('raptor_data')
    assert release.sha(rad.__file__) == release.CODE_FILES['raptor_radimagenet.py']
    assert release.sha(data.__file__) == release.CODE_FILES['raptor_data.py']
    return rad


class FakeCorpus:
    """DicomCorpus interface (depth/spec/mask/volume) keeping one study, counting decodes like the real corpus."""

    def __init__(self, ids, res=8, masks=None):
        from raptor_common import SLOTS64
        self.spec, self.depth = {'slots': SLOTS64}, 64
        rng = np.random.default_rng(0)
        self.data = {u: rng.integers(0, 256, size=(64, res, res), dtype=np.uint8) for u in ids}
        self.masks = masks or {}
        self.decodes, self.cached = 0, None

    def _ensure(self, uid):
        if self.cached != uid:
            self.decodes += 1
            self.cached = uid
        return self.data[uid]

    def volume(self, uid):
        return self._ensure(uid)

    def mask(self, uid):
        self._ensure(uid)
        return self.masks.get(uid, np.ones(self.depth))


def disk_corpus(root, ids, res=16, masks=None, seed=0):
    """A tiny raptor_corpus-format directory (manifest, ids, masks, one shard) readable by raptor_common.Corpus."""
    from raptor_common import SLOTS64
    root = Path(root)
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    np.save(root / 'vols_shard00.npy', rng.integers(0, 256, size=(len(ids), 64, res, res), dtype=np.uint8))
    np.save(root / 'ids.npy', np.array(ids, dtype=object), allow_pickle=True)
    np.save(root / 'masks.npy', np.ones((len(ids), 64), np.uint8) if masks is None else masks)
    manifest = {'contract': CONTRACT, 'spec': {'slots': [list(s) for s in SLOTS64], 'depth': 64, 'img': res},
                'shards': [{'start': 0, 'stop': len(ids), 'file': 'vols_shard00.npy'}]}
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root


# ------------------------------------------------------------------ 1. ensemble wrapper (v5 semantics, fail-closed)
def test_mean_prob_wrapper_is_v5_verbatim():
    torch = pytest.importorskip('torch')
    x = torch.linspace(-6, 6, 24).reshape(2, 12)
    models = [affine(torch, 1.0, 0.0), affine(torch, -0.5, 1.0), affine(torch, 2.0, -3.0)]
    with torch.inference_mode():
        got = release.mean_prob_module(models).eval()(x)
        assert torch.equal(got, v5.mean_prob_module(models).eval()(x))
        want = torch.stack([torch.sigmoid(m(x)) for m in models]).mean(0)
    assert (torch.sigmoid(got) - want).abs().max().item() <= 1e-6


@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
def test_nonfinite_member_logit_fails_closed(bad):
    torch = pytest.importorskip('torch')

    class Bad(torch.nn.Module):
        def forward(self, x):
            out = x.clone()
            out[0, 3] = bad
            return out
    ens = release.mean_prob_module([affine(torch, 1.0, 0.0), Bad()]).eval()
    with torch.inference_mode(), pytest.raises(RuntimeError, match='nonfinite member logit .member 1.'):
        ens(torch.zeros(2, 12))


def test_nonfinite_logit_fails_closed_through_frozen_predict_on_the_center3_path(monkeypatch):
    torch = pytest.importorskip('torch')
    rt = frozen_runtime(monkeypatch)
    input_modules()
    ids = ['1.1', '1.2']

    class Inf(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 12), float('inf'))
    with release.serving_input(rt, 'center3'):
        with pytest.raises(RuntimeError, match='nonfinite member logit'):
            rt.predict(release.mean_prob_module([tiny(torch, 1), Inf()]), FakeCorpus(ids), ids, 4, 8, 'none', 'cpu',
                       norm='radimagenet', batch=1, log=lambda *a: None, window_policy='within-slot-valid')
        with pytest.raises(RuntimeError, match='nonfinite logits'):     # predict's own check, single member
            rt.predict(Inf(), FakeCorpus(ids), ids, 4, 8, 'none', 'cpu', norm='radimagenet', batch=1,
                       log=lambda *a: None, window_policy='within-slot-valid')
    assert rt.prepare_input.__module__ == rt.__name__


# ------------------------------------------------------------------ 2. center3 input = the EXP107 training dataset input
def test_center3_input_equals_training_dataset_on_a_synthetic_corpus(tmp_path):
    pytest.importorskip('torch')
    input_modules()
    from raptor_common import Corpus, select_centers
    from raptor_radimagenet import CenterSliceWindows, center3_windows_from
    ids = ['1.1', '1.2', '1.3']
    masks = np.ones((3, 64), np.uint8)
    masks[1, 20:40] = 0          # gaps inside slots
    masks[2, :] = 0              # nothing valid -> every window is the explicit empty input
    root = disk_corpus(tmp_path / 'corpus', ids, res=16, masks=masks)
    corpus = Corpus(root)
    labels = {u: np.zeros(12, np.float32) for u in ids}
    for res in (16, 8):          # native size and the bilinear resize path
        ds = CenterSliceWindows(root, ids, labels, res, 36, False, 'radimagenet', 42, window_policy='within-slot-valid',
                                input_policy='center3')
        for i, uid in enumerate(ids):
            want = ds[i][0].numpy()
            got = release.center3_prepare_input(corpus, uid, 36, res, 'radimagenet', 'within-slot-valid')
            assert got.dtype == np.float32 and got.flags['C_CONTIGUOUS'] and got.shape == (36, 3, res, res)
            assert np.array_equal(got, want)
            centers = select_centers(corpus.mask(uid), corpus.depth, 36, corpus.spec, 'within-slot-valid', None)
            assert np.array_equal(got, center3_windows_from(corpus.volume(uid), centers, res) * np.float32(2) - np.float32(1))
            assert np.array_equal(got[:, 0], got[:, 1]) and np.array_equal(got[:, 1], got[:, 2])
            assert got.min() >= -1 and got.max() <= 1
    assert np.all(release.center3_prepare_input(corpus, '1.3', 12, 16, 'radimagenet', 'within-slot-valid') == -1.0)
    for bad in ('imagenet', 'none'):
        with pytest.raises(ValueError, match='requires norm radimagenet'):
            release.center3_prepare_input(corpus, '1.1', 12, 16, bad, 'within-slot-valid')


def test_center3_is_the_centre_channel_of_the_triplet_input(monkeypatch):
    pytest.importorskip('torch')
    rt = frozen_runtime(monkeypatch)
    input_modules()
    corpus = FakeCorpus(['1.1'], res=8)
    triplet = rt.prepare_input(corpus, '1.1', 12, 8, 'none', 'within-slot-valid')
    center = release.center3_prepare_input(corpus, '1.1', 12, 8, 'radimagenet', 'within-slot-valid')
    assert np.array_equal(center[:, 1], triplet[:, 1] * np.float32(2) - np.float32(1))
    assert not np.array_equal(center[:, 0], triplet[:, 0] * 2 - 1)


def test_center3_input_equals_training_dataset_on_real_cache_studies(monkeypatch):
    """The same fold-0 studies and centres through (a) the EXP107 training dataset and (b) the frozen predict with the
    v6 input swap: identical tensors (input hashes) at K36 / 224 px."""
    torch = pytest.importorskip('torch')
    if not (CACHE / 'manifest.json').is_file() or not FOLD0.is_file():
        pytest.skip('training cache not present')
    rt = frozen_runtime(monkeypatch)
    input_modules()
    from raptor_common import Corpus
    from raptor_radimagenet import CenterSliceWindows
    val = json.loads(FOLD0.read_text(encoding='utf-8'))['val']
    ids = [val[0], val[444]]
    labels = {u: np.zeros(12, np.float32) for u in ids}
    ds = CenterSliceWindows(CACHE, ids, labels, 224, 36, False, 'radimagenet', 42, window_policy='within-slot-valid',
                            input_policy='center3')
    want = [ds[i][0].numpy() for i in range(len(ids))]
    corpus = Corpus(CACHE)
    for uid, w in zip(ids, want):
        assert np.array_equal(release.center3_prepare_input(corpus, uid, 36, 224, 'radimagenet', 'within-slot-valid'), w)

    class Zero(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros(x.shape[0], 12)
    with release.serving_input(rt, 'center3') as how:
        _, timing = rt.predict(Zero(), corpus, ids, 36, 224, 'none', 'cpu', norm='radimagenet', batch=1,
                               log=lambda *a: None, window_policy='within-slot-valid')
    assert 'center3' in how and timing['norm'] == 'radimagenet'
    assert [x['sha256'] for x in timing['input_tensor_hashes']] == [rt.tensor_sha256(w) for w in want]


def test_serving_input_swaps_only_for_center3_and_always_restores(monkeypatch):
    torch = pytest.importorskip('torch')
    rt = frozen_runtime(monkeypatch)
    input_modules()
    original = rt.prepare_input
    with release.serving_input(rt, 'triplet') as how:
        assert rt.prepare_input is original and 'frozen' in how
    with release.serving_input(rt, 'center3'):
        assert rt.prepare_input is release.center3_prepare_input
    assert rt.prepare_input is original
    with pytest.raises(KeyError):
        with release.serving_input(rt, 'center3'):
            raise KeyError('boom')
    assert rt.prepare_input is original
    with pytest.raises(ValueError, match='Unknown input policy'):
        with release.serving_input(rt, 'center5'):
            pass
    # the triplet path keeps the v5 behaviour: frozen prepare_input, which refuses the radimagenet normalisation
    ids = ['1.1']
    with release.serving_input(rt, 'triplet'):
        with pytest.raises(RuntimeError, match='unsupported normalization'):
            rt.predict(tiny(torch, 1), FakeCorpus(ids), ids, 4, 8, 'none', 'cpu', norm='radimagenet', batch=1,
                       log=lambda *a: None, window_policy='within-slot-valid')


def test_center3_arm_through_frozen_predict_decodes_once_and_averages_probabilities(monkeypatch):
    torch = pytest.importorskip('torch')
    rt = frozen_runtime(monkeypatch)
    input_modules()
    ids = ['1.1', '1.2', '1.3']
    members = [tiny(torch, s) for s in (1, 2, 3)]

    def go(model):
        corpus = FakeCorpus(ids)
        with release.serving_input(rt, 'center3'):
            probs, timing = rt.predict(model, corpus, ids, 4, 8, 'none', 'cpu', norm='radimagenet', batch=1,
                                       log=lambda *a: None, window_policy='within-slot-valid')
        return probs, timing, corpus.decodes
    singles = [go(m) for m in members]
    pe, te, decodes = go(release.mean_prob_module(members))
    assert decodes == len(ids) and pe.shape == (3, 12) and pe.dtype == np.float32
    assert np.abs(pe - np.mean([s[0] for s in singles], axis=0)).max() <= 1e-6
    assert te['ordered_inputs_sha256'] == singles[0][1]['ordered_inputs_sha256']


def test_shipped_code_is_the_exp107_training_code():
    for n, s in release.CODE_FILES.items():
        assert release.sha(TRAINING_SRC / n) == s
    protocol = json.loads((ROOT / 'artifacts/experiments/EXP107_KFOLD_RADR50_PROTOCOL_V1/manifest.json').read_text(encoding='utf-8'))
    assert protocol['source_bindings']['training/src/raptor_radimagenet.py'] == release.CODE_FILES['raptor_radimagenet.py']
    assert protocol['init_weights']['sha256'] == release.RADIMAGENET_SHA256
    assert protocol['common_args']['input_policy'] == 'center3' and protocol['common_args']['norm'] == 'radimagenet'
    assert protocol['common_args']['arch'] == release.CENTER3_ARCH and protocol['extra_eval']['k_eval'] == 36
    assert protocol['common_args']['window_policy'] == 'within-slot-valid' and protocol['common_args']['res'] == 224


# ------------------------------------------------------------------ 3. arms
def test_parse_arm_spec():
    assert release.parse_arm_spec('coat5=coat_f0+coat_f1@36:1') == ('coat5', ['coat_f0', 'coat_f1'], 36, 1.0)
    assert release.parse_arm_spec('rad5=rad_f0+rad_f1+rad_f2+rad_f3+rad_f4@36:1') == \
        ('rad5', ['rad_f%d' % k for k in range(5)], 36, 1.0)
    assert release.parse_arm_spec('rad5=rad_f0@36') == ('rad5', ['rad_f0'], 36, None)
    assert release.parse_arm_spec('a') == ('a', ['a'], None, None)
    for bad in ['', 'a=b+b', 'a=', '../x', 'a@0', 'a:0', 'a:-1', 'a:nan', 'a:1e3', 'a@36:', 'a:0.7@36']:
        with pytest.raises(ValueError):
            release.parse_arm_spec(bad)
    for text in ('coat5=coat_f0+coat_f1@36:0.7', 'a', 'x=y@24', 'r=a+b:.3'):
        assert release.parse_arm_spec(text) == v5.parse_arm_spec(text)


def test_build_arm_input_policies(tmp_path):
    am = two_family_assets(tmp_path)
    coat, rad = two_arms(am)
    assert coat['input_policy'] == 'triplet' and coat['norm'] == 'imagenet' and coat['arch'] == ARCH
    assert rad['input_policy'] == 'center3' and rad['norm'] == 'radimagenet' and rad['arch'] == RAD_ARCH
    assert rad['k_eval'] == 36 and [c['meta']['k_eval'] for c in rad['checkpoints']] == [24, 24]
    assert set(coat) == release.ARM_KEYS and 'input_policy' not in coat['checkpoints'][0]['meta']
    with pytest.raises(ValueError, match='differ in arch'):
        release.build_arm('mixed', ['c0', 'r0'], 36, am, 0.35)
    bad = fake_assets(tmp_path / 'bad', {
        'n': (b'n', rad_meta(norm='imagenet')), 'a': (b'a', rad_meta(arch=RN_ARCH)), 'i': (b'i', rad_meta(init='imagenet')),
        'w': (b'w', rad_meta(init_weights_sha256='0' * 64)), 't': (b't', meta(norm='radimagenet')),
        'p': (b'p', rad_meta(input_policy='center5')), 'r': (b'r', rad_meta(input_policy='triplet'))})
    for name, message in [('n', 'norm/window policy'), ('a', 'only served for arch'), ('i', 'RadImageNet'),
                          ('w', 'RadImageNet'), ('t', 'norm/window policy'), ('p', 'Unknown input policy'),
                          ('r', 'norm/window policy')]:
        with pytest.raises(ValueError, match=message):
            release.build_arm('x', [name], 36, bad, 0.25)


def test_release_manifest_validation(tmp_path):
    am = two_family_assets(tmp_path)
    arms = two_arms(am)
    path = tmp_path / 'release-manifest.json'
    digest = release_manifest(path, arms, '1' * 64)
    p = release.checked_manifest(path, digest)
    assert [(a['name'], a['input_policy']) for a in p['arms']] == [('coat5', 'triplet'), ('rad5', 'center3')]
    assert p['rank_weights'] == {'coat5': 0.5, 'rad5': 0.5}
    with pytest.raises(ValueError, match='Release manifest changed'):
        release.checked_manifest(path, '0' * 64)
    rad = arms[1]
    no_policy = {k: v for k, v in rad.items() if k != 'input_policy'}
    bad_cases = [
        ({'schema': v5.SCHEMA}, 'policy'),
        ({'release_source_sha256': release.sha(v5.__file__)}, 'source changed'),
        ({'arms': [arms[0], no_policy]}, 'Arm fields differ'),
        ({'arms': [arms[0], {**rad, 'input_policy': 'triplet'}]}, 'norm/window policy'),
        ({'arms': [arms[0], {**rad, 'input_policy': 'center5'}]}, 'Unknown input policy'),
        ({'arms': [arms[0], {**rad, 'norm': 'imagenet'}]}, 'norm/window policy'),
        ({'arms': [{**arms[0], 'input_policy': 'center3'}, rad]}, 'norm/window policy'),
        ({'arms': [arms[0], {**rad, 'checkpoints': [{**rad['checkpoints'][0], 'meta': {
            **rad['checkpoints'][0]['meta'], 'init': 'imagenet'}}]}]}, 'RadImageNet'),
        ({'arms': [arms[0], {**rad, 'checkpoints': arms[0]['checkpoints']}]}, 'differs from arm'),
        ({'arms': [arms[0], {**rad, 'weight': 0.0}]}, 'Rank weight'),
        ({'arms': [arms[0], {**rad, 'aggregator': 'mean_logit'}]}, 'mean_prob'),
        ({'rank_weights': {'coat5': 0.7, 'rad5': 0.3}}, 'Recorded rank weights'),
    ]
    bad_cases += [({'new_asset_dataset': d}, 'Unsafe new-assets dataset') for d in release.REFUSED_DATASETS]
    for i, (over, message) in enumerate(bad_cases):
        bad = tmp_path / ('bad-%d.json' % i)
        d = release_manifest(bad, arms, '1' * 64, **over)
        with pytest.raises(ValueError, match=message):
            release.checked_manifest(bad, d)


def test_checkpoint_check_center3(tmp_path):
    am = two_family_assets(tmp_path)
    coat, rad = two_arms(am)
    ck = rad['checkpoints'][0]
    r = release.checkpoint_check({**rad_meta(tag='R0'), 'provenance': {'ignored': True}, 'model': None}, rad, ck)
    assert r['input_policy'] == 'center3' and r['k_eval_checkpoint'] == 24 and r['k_eval_release'] == 36
    assert release.checkpoint_check(meta(tag='C0'), coat, coat['checkpoints'][0])['input_policy'] == 'triplet'
    no_policy = {k: v for k, v in rad_meta(tag='R0').items() if k != 'input_policy'}
    for loaded, message in [(no_policy, 'differs from arm'), (rad_meta(tag='R0', norm='imagenet'), 'differs from arm'),
                            (rad_meta(tag='R0', init='imagenet'), 'RadImageNet'), (rad_meta(tag='R9'), 'staged record'),
                            ({k: v for k, v in rad_meta(tag='R0').items() if k != 'epoch'}, 'staged record')]:
        with pytest.raises(ValueError, match=message):
            release.checkpoint_check(loaded, rad, ck)
    with pytest.raises(ValueError, match='differs from arm'):
        release.checkpoint_check(meta(tag='C0'), rad, ck)      # a CoAtNet file in the RadR50 arm


# ------------------------------------------------------------------ 4. manifest hashes and assets
def test_assets_binding_checks_checkpoints_code_and_dataset(tmp_path):
    am = two_family_assets(tmp_path)
    root = tmp_path / 'new'
    arms = two_arms(am)
    rel = {'new_assets_manifest_sha256': release.sha(root / release.NEW_MANIFEST), 'new_asset_dataset': release.NEW_DATASET}
    assert release.verify_new_assets(rel, root, arms)['code'] == release.code_record()
    for d in (*release.REFUSED_DATASETS, release.OLD_DATASET, 'Bad/Slug', None):
        with pytest.raises(ValueError, match='Unsafe new-assets dataset'):
            release.verify_new_assets({**rel, 'new_asset_dataset': d}, root, arms)
    with pytest.raises(ValueError, match='assets manifest changed'):
        release.verify_new_assets({**rel, 'new_assets_manifest_sha256': '0' * 64}, root, arms)
    code = root / release.CODE_DIR / 'raptor_radimagenet.py'
    original = code.read_bytes()
    code.write_bytes(original + b'\n# tampered\n')
    with pytest.raises(ValueError, match='Input code changed'):
        release.verify_new_assets(rel, root, arms)
    code.write_bytes(original)
    release.verify_new_assets(rel, root, arms)
    (root / 'checkpoints/r1.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='sha256 differs'):
        release.verify_new_assets(rel, root, arms)
    # a manifest without the code record, or with another code hash, fails closed even when its digest is current
    for change in ({'code': {}}, {'code': {**release.code_record(), 'raptor_data.py': {'path': 'code/raptor_data.py',
                                                                                      'sha256': '0' * 64}}}):
        am2 = {**am, **change}
        (root / release.NEW_MANIFEST).write_text(json.dumps(am2), encoding='utf-8')
        with pytest.raises(ValueError, match='Input code record differs'):
            release.verify_new_assets({**rel, 'new_assets_manifest_sha256': release.sha(root / release.NEW_MANIFEST)},
                                      root, arms)


def ck_payload(torch, rad=False, **over):
    protocol = json.loads((release.OLD_LOCAL / release.OLD_PROTOCOL).read_text(encoding='utf-8'))
    code = {f: protocol['files']['training/src/' + f] for f in release.TRAINING_CODE_BOUND}
    provenance = {'code_sha256': code}
    if rad:
        code['raptor_data.py'] = release.CODE_FILES['raptor_data.py']
        provenance['radimagenet_module_sha256'] = release.CODE_FILES['raptor_radimagenet.py']
    m = rad_meta(**over) if rad else meta(**over)
    return {'model': {'w': torch.randn(3)}, **m, 'provenance': provenance}


def save_ck(torch, path, rad=False, inventory=True, **over):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck_payload(torch, rad, **over), path)
    if inventory:
        (path.parent / 'output-inventory.json').write_text(json.dumps(
            {path.name: {'sha256': release.sha(path), 'bytes': path.stat().st_size}}), encoding='utf-8')
    return path


def frozen_assets_or_skip():
    for p in (release.OLD_LOCAL / release.OLD_PROTOCOL, release.ACCEPTANCE, release.DOCKER_PULL):
        if not Path(p).is_file():
            pytest.skip('frozen assets / acceptance / docker pin not present')


def test_stage_assets_ships_code_checks_inventory_and_refuses_old_releases(tmp_path):
    torch = pytest.importorskip('torch')
    frozen_assets_or_skip()
    c0 = save_ck(torch, tmp_path / 'C0' / 'model_best.pt', tag='C0')
    r0 = save_ck(torch, tmp_path / 'R0' / 'model_best.pt', rad=True, tag='R0')
    pkg = tmp_path / 'pkg'
    a = SimpleNamespace(checkpoint=['coat_f0=' + str(c0), 'rad_f0=' + str(r0)], package=str(pkg), copy=True,
                        dataset='taeyangg4/knee-release-assets-v9')
    release.stage_assets(a)
    am = json.loads((pkg / release.NEW_MANIFEST).read_text(encoding='utf-8'))
    assert am['schema'] == release.ASSETS_SCHEMA and am['code'] == release.code_record()
    assert set(am['files']) == {'checkpoints/coat_f0.pt', 'checkpoints/rad_f0.pt', 'code/raptor_data.py',
                                'code/raptor_radimagenet.py'}
    assert am['checkpoints']['rad_f0']['input_code_sha256'] == {'raptor_radimagenet.py': release.CODE_FILES['raptor_radimagenet.py']}
    assert am['checkpoints']['rad_f0']['meta']['input_policy'] == 'center3' and am['checkpoints']['coat_f0']['input_code_sha256'] == {}
    assert am['checkpoints']['coat_f0']['source_inventory_sha256'] == release.sha(c0.parent / 'output-inventory.json')
    assert (pkg / 'code/raptor_radimagenet.py').read_bytes() == (TRAINING_SRC / 'raptor_radimagenet.py').read_bytes()
    md = json.loads((pkg / 'dataset-metadata.json').read_text(encoding='utf-8'))
    assert md['id'] == 'taeyangg4/knee-release-assets-v9' and md['isPrivate']
    digest = release.sha(pkg / release.NEW_MANIFEST)
    release.stage_assets(a)
    assert release.sha(pkg / release.NEW_MANIFEST) == digest
    # an inventory that records other bytes fails closed and leaves no staged file
    bad = save_ck(torch, tmp_path / 'R1' / 'model_best.pt', rad=True, tag='R1')
    (bad.parent / 'output-inventory.json').write_text(json.dumps({'model_best.pt': {'sha256': '0' * 64, 'bytes': 1}}))
    with pytest.raises(ValueError, match='run inventory'):
        release.stage_assets(SimpleNamespace(**{**vars(a), 'checkpoint': ['rad_f1=' + str(bad)]}))
    assert not (pkg / 'checkpoints/rad_f1.pt').exists()
    # a centre-plane member that is not RadImageNet-initialised is refused before copying
    wrong = save_ck(torch, tmp_path / 'R2' / 'model_best.pt', rad=True, tag='R2', init='imagenet')
    with pytest.raises(ValueError, match='RadImageNet'):
        release.stage_assets(SimpleNamespace(**{**vars(a), 'checkpoint': ['rad_f2=' + str(wrong)]}))
    assert not (pkg / 'checkpoints/rad_f2.pt').exists()
    for d in (*release.REFUSED_DATASETS, release.OLD_DATASET, release.OPENCV_DATASET, 'A/B'):
        with pytest.raises(ValueError, match='Unsafe new-assets dataset'):
            release.stage_assets(SimpleNamespace(**{**vars(a), 'dataset': d, 'package': str(tmp_path / 'x')}))
    for p in release.REFUSED_PACKAGES:
        with pytest.raises(ValueError, match='v4-v7 release-assets package'):
            release.stage_assets(SimpleNamespace(**{**vars(a), 'package': str(p)}))
    assert release.NEW_DATASET == 'taeyangg4/knee-release-assets-v8' and release.NEW_DATASET not in release.REFUSED_DATASETS
    assert release.PACKAGE.name == 'KNEE_RELEASE_ASSETS_V8' and release.PACKAGE not in release.REFUSED_PACKAGES
    assert v5.NEW_DATASET in release.REFUSED_DATASETS and v5.PACKAGE in release.REFUSED_PACKAGES


# ------------------------------------------------------------------ 5. runtime screen
def arms_of(*counts, m=0.35):
    ms = m if isinstance(m, (list, tuple)) else [m] * len(counts)
    return [{'name': 'a%d' % i, 'checkpoints': [None] * c, 'model_seconds_per_study': s}
            for i, (c, s) in enumerate(zip(counts, ms))]


def test_time_screen_for_coat5_rad5():
    assert release.TIME == v5.TIME and release.MODEL_SECONDS[RAD_ARCH] == 0.25
    assert release.MODEL_SECONDS[ARCH] == 0.35 and release.MODEL_SECONDS[RN_ARCH] == 0.15
    arms = arms_of(5, 5, m=[0.35, 0.25])
    plan = release.operational_plan(1300, arms)
    assert plan == v5.operational_plan(1300, arms)                      # formula unchanged
    assert plan['arm_projected_seconds']['a0'] == pytest.approx(1.3 * (D + 1.75) * 1300, abs=1e-9)   # 13,984.8 s
    assert plan['arm_projected_seconds']['a1'] == pytest.approx(1.3 * (D + 1.25) * 1300, abs=1e-9)   # 13,139.8 s
    assert plan['projected_seconds'] == pytest.approx(1.3 * (2 * D + 3.0) * 1300 + 600, abs=1e-6)
    assert plan['projected_seconds'] == pytest.approx(27724.61219, abs=1e-3) and plan['projected_seconds'] <= 30600
    assert release.max_studies(arms) == 1437
    release.operational_plan(1437, arms)
    with pytest.raises(ValueError, match='time screen'):
        release.operational_plan(1438, arms)
    # the RadR50 arm alone could take 0.59 s per model per study at N = 1300 before the screen fails
    release.operational_plan(1300, arms_of(5, 5, m=[0.35, 0.59]))
    with pytest.raises(ValueError, match='time screen'):
        release.operational_plan(1300, arms_of(5, 5, m=[0.35, 0.60]))
    last = release.launch_projection(plan, arms, 'a1')
    assert last == pytest.approx(plan['arm_projected_seconds']['a1'] + 300)
    release.launch_budget(1000.0, 1000.0 + 1300 * (D + 1.75) * 1.5, 1300, last, 30000.0)


def test_model_seconds_basis_recorded():
    seconds, basis = release.arm_model_seconds('rad5', RAD_ARCH, {})
    assert seconds == 0.25 and 'EXP107' in basis
    assert release.arm_model_seconds('rad5', RAD_ARCH, {'rad5': 0.3}) == (0.3, 'override --model-seconds rad5=0.3')
    with pytest.raises(ValueError, match='No model_seconds_per_study default'):
        release.arm_model_seconds('x', 'convnext_small', {})


# ------------------------------------------------------------------ 6. rank combination with two arms
def exp098_rank():
    code = exp098.CODE
    ns = {}
    exec(code[code.index('def rank(col):'):code.index('cols=[]')], ns)
    return ns['rank']


def synthetic_tables(n=257, seed=7):
    rng = random.Random(seed)
    pC = [[rng.choice([0.1, 0.25, 0.5]) if t % 3 == 0 else rng.random() for t in range(12)] for _ in range(n)]
    pR = [[round(rng.random(), 2) for _ in range(12)] for _ in range(n)]
    return pC, pR


def test_equal_rank_mean_of_two_arms_is_exp098_and_v5_bit_for_bit():
    rank = exp098_rank()
    pC, pR = synthetic_tables()
    cols98 = []
    for t in range(12):
        rc = rank([r[t] for r in pC]); rr = rank([r[t] for r in pR]); cols98.append([(a + b) / 2 for a, b in zip(rc, rr)])
    for weights in (None, [1.0, 1.0], [0.5, 0.5]):
        assert release.rank_mean([pC, pR], weights) == cols98 == v5.rank_mean([pC, pR], weights)


def stage_arm_results(base, arms, digest, ids, tables):
    for arm, probs in zip(arms, tables):
        d = base / arm['name']
        (d / 'work').mkdir(parents=True)
        write_csv(d / 'arm.csv', [release.UID, *release.TARGETS], [[u, *map(repr, p)] for u, p in zip(ids, probs)])
        r = {'status': 'PASS_TEST_COHORT_ONLY', 'arm': arm['name'], 'release_manifest_sha256': digest,
             'studies': len(ids), 'submission_sha256': release.sha(d / 'arm.csv'), 'elapsed_seconds': 1.0,
             'sampled_tree_rss_bytes': 1, 'n_models': len(arm['checkpoints']), 'k_eval': 36, 'rank_weight': arm['weight'],
             'input_policy': arm['input_policy']}
        (d / 'work/result.json').write_text(json.dumps(r), encoding='utf-8')


def test_combine_two_arms_end_to_end_and_fail_closed(tmp_path):
    ids = ['1.%d' % i for i in range(23)]
    comp = fixture_comp(tmp_path / 'comp', ids)
    am = two_family_assets(tmp_path)
    arms = two_arms(am)
    digest = release_manifest(tmp_path / 'm.json', arms, '1' * 64)
    pC, pR = synthetic_tables(n=len(ids), seed=3)
    base = tmp_path / 'base'
    stage_arm_results(base, arms, digest, ids, [pC, pR])
    a = SimpleNamespace(manifest=str(tmp_path / 'm.json'), sha256=digest, base=str(base), competition=str(comp),
                        out=str(tmp_path / 'submission.csv'), started=time.monotonic())
    r = release.combine(a)
    assert r['status'] == 'PASS_RANK_MEAN_TEST_COHORT_ONLY' and r['arm_order'] == ['coat5', 'rad5']
    assert r['rank_weights'] == {'coat5': 0.5, 'rad5': 0.5} and 'EXP098' in r['formula']
    release.write_columns(tmp_path / 'expected.csv', ids, release.rank_mean([pC, pR]))
    assert (tmp_path / 'submission.csv').read_bytes() == (tmp_path / 'expected.csv').read_bytes()
    release.csv_check(tmp_path / 'submission.csv', ids)
    assert not list(tmp_path.glob('*.exp111-tmp'))
    (tmp_path / 'submission.csv').unlink()
    shutil.rmtree(base / 'combine')
    write_csv(base / 'rad5/arm.csv', [release.UID, *release.TARGETS], [[u, *([0.5] * 12)] for u in ids])
    with pytest.raises(ValueError, match='Arm output changed'):
        release.combine(a)
    assert not (tmp_path / 'submission.csv').exists() and (base / 'combine/failure.json').is_file()


@pytest.mark.parametrize('rows', [
    [['1.1', *([.5] * 12)]],
    [['1.2', *([.5] * 12)], ['1.1', *([.5] * 12)]],
    [['1.1', *(['nan'] * 12)], ['1.2', *([.5] * 12)]],
    [['1.1', *(['inf'] * 12)], ['1.2', *([.5] * 12)]],
])
def test_reject_partial_reordered_nonfinite_output(tmp_path, rows):
    path = tmp_path / 'raw.csv'
    write_csv(path, [release.UID, *release.TARGETS], rows)
    with pytest.raises(ValueError):
        release.csv_check(path, ['1.1', '1.2'])


def test_resource_validation_has_v5_semantics():
    if not SUPERVISOR.is_file():
        pytest.skip('completed supervisor evidence not present')
    base = json.loads(SUPERVISOR.read_text())
    assert release.validate_resource(base, base['argv'], base['child_identity']['pid'], 900) == \
        v5.validate_resource(base, base['argv'], base['child_identity']['pid'], 900)
    bad = copy.deepcopy(base); bad['cgroup_end']['values']['memory.events']['oom_kill'] = 1
    with pytest.raises(ValueError, match='OOM'):
        release.validate_resource(bad, bad['argv'], bad['child_identity']['pid'], 900)


# ------------------------------------------------------------------ 7. notebook generation and line length
def test_render_code_line_length_and_round_trip():
    big = os.urandom(200_000)
    text = json.dumps({'k%d' % i: 'v"\\é\n' * 40 for i in range(120)}, indent=2, ensure_ascii=False) + '\n'
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    code = release.render_code('coat5rad5', big, text, digest)
    assert max(len(line) for line in code.split('\n')) <= release.MAX_NOTEBOOK_LINE
    assert max(len(line) for line in code.split('\n')) <= 2 * release.SOURCE_CHUNK
    src_expr = code.split('source=base64.b85decode(', 1)[1].split('\nassert hashlib.sha256(source)', 1)[0].rstrip()[:-1]
    assert __import__('base64').b85decode(eval(src_expr, {})) == big
    assert eval(code.split('\nMANIFEST=', 1)[1].split('\nDIGEST=', 1)[0], {}) == text
    assert "Path('/kaggle/working/exp111-coat5rad5')" in code and "'exp111_release_v6.py'" in code
    assert 'v6 time screen failed' in code and 'exp104' not in code


def test_kernel_metadata():
    km = release.kernel_metadata('coat5rad5', release.DOCKER_IMAGE)
    assert km['id'] == 'taeyangg4/knee-exp111-coat5rad5-v6' and km['title'] == 'knee-exp111-coat5rad5-v6'
    assert km['is_private'] and km['enable_gpu'] and not km['enable_internet'] and km['machine_shape'] == 'NvidiaTeslaT4'
    assert km['dataset_sources'] == ['taeyangg4/knee-exp089-recipe-transfer-v1', 'taeyangg4/knee-release-assets-v8',
                                     'mattiaangeli/opencv-python-headless-4120088-x86']
    assert km['docker_image'] == v5.DOCKER_IMAGE
    for d in release.REFUSED_DATASETS:
        with pytest.raises(ValueError):
            release.kernel_metadata('coat5rad5', release.DOCKER_IMAGE, d)
    with pytest.raises(ValueError):
        release.kernel_metadata('coat5rad5', 'gcr.io/kaggle-gpu-images/python:latest')


def notebook_args(tmp_path, pkg, arm, **over):
    a = {'release': 'syn', 'arm': arm, 'out': str(tmp_path / ('k-%d' % len(list(tmp_path.glob('k-*'))))),
         'package': str(pkg), 'dataset': 'taeyangg4/knee-release-assets-v9', 'model_seconds': [],
         'expected_studies': 1300, 'docker_from': str(release.DOCKER_PULL)}
    a.update(over)
    return SimpleNamespace(**a)


def test_notebook_generation_for_two_families(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    frozen_assets_or_skip()
    old = release.verify_old_assets(release.OLD_LOCAL)
    monkeypatch.setattr(release, 'verify_old_assets', lambda root, opencv=None: old)
    specs = ['c%d=%s' % (k, save_ck(torch, tmp_path / 'src' / ('C%d' % k) / 'model_best.pt', tag='C%d' % k)) for k in range(2)]
    specs += ['r%d=%s' % (k, save_ck(torch, tmp_path / 'src' / ('R%d' % k) / 'model_best.pt', rad=True, tag='R%d' % k))
              for k in range(2)]
    pkg = tmp_path / 'pkg'
    release.stage_assets(SimpleNamespace(checkpoint=specs, package=str(pkg), copy=True, dataset='taeyangg4/knee-release-assets-v9'))
    a = notebook_args(tmp_path, pkg, ['coat5=c0+c1@36:1', 'rad5=r0+r1@36:1'])
    release.notebook(a)
    out = Path(a.out)
    p = release.checked_manifest(out / 'release-manifest.json', release.sha(out / 'release-manifest.json'))
    assert [(x['name'], x['input_policy'], x['norm'], x['model_seconds_per_study'], x['weight']) for x in p['arms']] == \
        [('coat5', 'triplet', 'imagenet', 0.35, 1.0), ('rad5', 'center3', 'radimagenet', 0.25, 1.0)]
    assert p['rank_weights'] == {'coat5': 0.5, 'rad5': 0.5} and 'EXP098' in p['combination']
    assert 'EXP107' in p['model_seconds_basis']['rad5']
    release.verify_new_assets(p, pkg, p['arms'])
    plan = json.loads((out / 'execution-plan.json').read_text(encoding='utf-8'))
    assert '--timeout' not in plan['push_argv'] and plan['push_argv'][-2:] == ['--accelerator', 'NvidiaTeslaT4']
    assert plan['publish_assets_env'] == {'PYTHONIOENCODING': 'utf-8'} and plan['publish_assets_argv'][-3] == str(pkg.resolve())
    assert [x['input_policy'] for x in plan['arms']] == ['triplet', 'center3']
    code = ''.join(json.loads((out / 'release.ipynb').read_text(encoding='utf-8'))['cells'][0]['source'])
    assert max(len(line) for line in code.split('\n')) <= release.MAX_NOTEBOOK_LINE
    # a RadR50 member whose provenance names other input code is refused
    am = json.loads((pkg / release.NEW_MANIFEST).read_text(encoding='utf-8'))
    am['checkpoints']['r1']['input_code_sha256'] = {'raptor_radimagenet.py': '0' * 64}
    (pkg / release.NEW_MANIFEST).write_text(json.dumps(am), encoding='utf-8')
    a = notebook_args(tmp_path, pkg, ['coat5=c0+c1@36:1', 'rad5=r0+r1@36:1'])
    with pytest.raises(ValueError, match='input code differing'):
        release.notebook(a)
    assert not Path(a.out).exists()
    for arm, over, message in [(['m=c0+r0@36'], {}, 'differ in arch'),
                               (['coat5=c0+c1@36:1', 'rad5=r0@36'], {}, 'every arm or for none'),
                               (['coat5=c0+c1@36', 'rad5=r0@36'], {'expected_studies': 2000}, 'time screen'),
                               (['coat5=c0@36', 'rad5=r0@36'], {'dataset': 'taeyangg4/knee-release-assets-v7'},
                                'Unsafe new-assets dataset')]:
        a = notebook_args(tmp_path, pkg, arm, **over)
        with pytest.raises(ValueError, match=message):
            release.notebook(a)
        assert not Path(a.out).exists()


EXPECTED = {'coat5rad5': [('coat5', ['coat_f%d' % k for k in range(5)], 36, 1.0, 0.35, 'triplet', 'imagenet'),
                          ('rad5', ['rad_f%d' % k for k in range(5)], 36, 1.0, 0.25, 'center3', 'radimagenet')],
            'rad5': [('rad5', ['rad_f%d' % k for k in range(5)], 36, 1.0, 0.25, 'center3', 'radimagenet')]}
WEIGHTS = {'coat5rad5': {'coat5': 0.5, 'rad5': 0.5}, 'rad5': {'rad5': 1.0}}
MAX_STUDIES = {'coat5rad5': 1437, 'rad5': 2968}
PROJECTED_1300 = {'coat5rad5': 1.3 * (2 * D + 3.0) * 1300 + 600, 'rad5': 1.3 * (D + 1.25) * 1300 + 600}   # 27,724.6 / 13,739.8 s


def kernel_dir(name):
    out = KERNELS / name
    if not (out / 'kernel-metadata.json').is_file() or not (release.PACKAGE / release.NEW_MANIFEST).is_file():
        pytest.skip('kernel dir or assets package not generated')
    return out


@pytest.mark.parametrize('name', ['coat5rad5', 'rad5'])
def test_generated_kernel_dir_is_current(name):
    """The staged kernels embed this exact tool, bind the staged v8 package and pin the Python 3.12 image."""
    out = kernel_dir(name)
    km = json.loads((out / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert km == release.kernel_metadata(name, release.DOCKER_IMAGE, release.NEW_DATASET)
    assert km['id'] == 'taeyangg4/knee-exp111-%s-v6' % name
    pulled = json.loads(release.DOCKER_PULL.read_text(encoding='utf-8'))
    assert pulled['id'] == release.DOCKER_SOURCE_KERNEL and pulled['docker_image'] == km['docker_image']
    digest = release.sha(out / 'release-manifest.json')
    p = release.checked_manifest(out / 'release-manifest.json', digest)
    assert [(a['name'], [c['name'] for c in a['checkpoints']], a['k_eval'], a['weight'], a['model_seconds_per_study'],
             a['input_policy'], a['norm']) for a in p['arms']] == EXPECTED[name]
    assert all(c['meta']['k_eval'] == 24 for a in p['arms'] for c in a['checkpoints'])
    assert p['new_asset_dataset'] == 'taeyangg4/knee-release-assets-v8' and p['rank_weights'] == WEIGHTS[name]
    assert p['combination'].startswith('single arm') == (name == 'rad5')
    am = release.verify_new_assets(p, release.PACKAGE, p['arms'])
    v5_am = v5.PACKAGE / v5.NEW_MANIFEST
    if v5_am.is_file():   # the CoAtNet members are the five served by assets v5 (same bytes and metadata)
        old = json.loads(v5_am.read_text(encoding='utf-8'))['checkpoints']
        for k in range(5):
            assert {x: am['checkpoints']['coat_f%d' % k][x] for x in ('sha256', 'meta', 'source')} == \
                {x: old['coat_f%d' % k][x] for x in ('sha256', 'meta', 'source')}
    exp107 = ROOT / 'artifacts/experiments/EXP107_RADR50_DREAD_OOF/result.json'
    if exp107.is_file():   # the RadR50 members are the five whose K36 OOF was assembled
        sources = json.loads(exp107.read_text(encoding='utf-8'))['sources']
        assert [am['checkpoints']['rad_f%d' % k]['sha256'] for k in range(5)] == [sources[str(k)]['checkpoint_sha256'] for k in range(5)]
    code = ''.join(json.loads((out / 'release.ipynb').read_text(encoding='utf-8'))['cells'][0]['source'])
    assert code == release.render_code(name, Path(release.__file__).read_bytes(),
                                       (out / 'release-manifest.json').read_bytes().decode('utf-8'), digest)
    assert max(len(line) for line in code.split('\n')) <= release.MAX_NOTEBOOK_LINE
    plan = json.loads((out / 'execution-plan.json').read_text(encoding='utf-8'))
    assert plan['manifest_sha256'] == digest and plan['notebook_sha256'] == release.sha(out / 'release.ipynb')
    assert '--timeout' not in plan['push_argv'] and plan['max_studies_passing_screen'] == MAX_STUDIES[name]
    assert plan['screen_at_expected_hidden_N']['projected_seconds'] == pytest.approx(PROJECTED_1300[name], abs=1e-6)
    assert plan['push_argv'][-4:] == ['-p', str(out.resolve()), '--accelerator', 'NvidiaTeslaT4']
    assert plan['first_output'] == 'exp111-%s/%s' % (name, 'combine/result.json' if name == 'coat5rad5' else 'rad5/work/result.json')


# ------------------------------------------------------------------ 8. simulated notebook cell on a fake /kaggle tree
def fake_kaggle(root, n, legacy_mount):
    k = Path(root) / 'kaggle'
    (k / 'working').mkdir(parents=True)
    inp = k / 'input'
    old = inp / 'knee-exp089-recipe-transfer-v1' if legacy_mount else inp / 'datasets/taeyangg4/knee-exp089-recipe-transfer-v1'
    new = inp / 'knee-release-assets-v8' if legacy_mount else inp / 'datasets/taeyangg4/knee-release-assets-v8'
    cv = inp / 'datasets/mattiaangeli/opencv-python-headless-4120088-x86'
    comp = inp / 'rsna-knee-abnormality-detection' if legacy_mount else inp / 'competitions/rsna-knee-abnormality-detection'
    for d, f in ((old, 'exp089-t4-protocol.json'), (new, 'assets-manifest.json'),
                 (cv, 'opencv_python_headless-4.12.0.88-cp37-abi3-manylinux_2_17_x86_64.whl')):
        d.mkdir(parents=True)
        (d / f).write_text('{}', encoding='utf-8')
    ids = ['1.2.840.%d' % (9000 + 7 * i) for i in range(n)]
    fixture_comp(comp, ids)
    return k, ids


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate_cell(monkeypatch, tmp_path, name, n, legacy_mount=False):
    out = kernel_dir(name)
    code = ''.join(json.loads((out / 'release.ipynb').read_text(encoding='utf-8'))['cells'][0]['source'])
    k, ids = fake_kaggle(tmp_path / 'fake', n, legacy_mount)
    prefix = (tmp_path / 'fake').as_posix()
    assert code.count("'/kaggle/") >= 8
    code = code.replace("'/kaggle/", "'" + prefix + '/kaggle/')
    real_run = subprocess.run
    calls, tables, tools = [], {}, {}

    def stub(argv, check=False, **kw):
        mode = argv[2]
        opts = dict(zip(argv[3::2], argv[4::2]))
        if mode == 'combine':
            calls.append(('combine', None))
            return real_run(argv, check=check, **kw)          # the real embedded tool, real CLI
        assert mode == 'run' and check
        if 'm' not in tools:
            tools['m'] = load_path('embedded_v6_' + name + '_%d' % n, argv[1])
        tool = tools['m']
        p = tool.checked_manifest(opts['--manifest'], opts['--sha256'])
        arm = tool.arm_by_name(p, opts['--arm'])
        contract = tool.test_contract(opts['--competition'])
        plan = tool.operational_plan(contract['studies'], p['arms'])
        projection = tool.launch_projection(plan, p['arms'], arm['name'])
        tool.launch_budget(float(opts['--started']), time.monotonic(), contract['studies'], projection,
                           float(opts['--bootstrap-max-seconds']))
        calls.append(('run', arm['name'], opts['--bootstrap-max-seconds']))
        rng = random.Random(arm['name'])
        probs = [[rng.random() for _ in range(12)] for _ in contract['ids']]
        tables[arm['name']] = probs
        work, dest = Path(opts['--work']), Path(opts['--out'])
        work.mkdir(parents=True)
        tool.write_columns(dest, contract['ids'], [[row[t] for row in probs] for t in range(12)])
        result = {'status': 'PASS_TEST_COHORT_ONLY', 'arm': arm['name'], 'release_manifest_sha256': opts['--sha256'],
                  'studies': contract['studies'], 'submission_sha256': tool.sha(dest), 'elapsed_seconds': 1.0,
                  'sampled_tree_rss_bytes': 1, 'n_models': len(arm['checkpoints']), 'k_eval': arm['k_eval'],
                  'rank_weight': arm['weight'], 'model_seconds': 0.0}
        (work / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        return subprocess.CompletedProcess(argv, 0)

    saved = dict(os.environ)
    monkeypatch.setattr(subprocess, 'run', stub)
    try:
        exec(compile(code, 'release-cell-' + name, 'exec'), {'__name__': '__main__'})
    finally:
        monkeypatch.setattr(subprocess, 'run', real_run)
        os.environ.clear()
        os.environ.update(saved)
    return k, ids, calls, tables


@pytest.mark.parametrize('name', ['coat5rad5', 'rad5'])
@pytest.mark.parametrize('n,legacy', [(3, False), (4, True), (1300, False)])
def test_simulated_notebook_cell(monkeypatch, tmp_path, capsys, name, n, legacy):
    k, ids, calls, tables = simulate_cell(monkeypatch, tmp_path, name, n, legacy)
    final = k / 'working/submission.csv'
    base = k / 'working' / ('exp111-' + name)
    assert (base / 'exp111_release_v6.py').read_bytes() == Path(release.__file__).read_bytes()
    release.csv_check(final, ids)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['studies'] == n and printed['submission_sha256'] == release.sha(final)
    assert printed['rank_weights'] == WEIGHTS[name]
    if name == 'rad5':     # single arm: the arm's raw mean_prob probabilities are the submission, no combine
        assert calls == [('run', 'rad5', '600')] and printed['status'] == 'PASS_TEST_COHORT_ONLY'
        release.write_columns(base / 'expected.csv', ids, [[row[t] for row in tables['rad5']] for t in range(12)])
    else:
        boots = ['600', '600'] if n <= 3 else ['600', '30000']
        assert calls == [('run', 'coat5', boots[0]), ('run', 'rad5', boots[1]), ('combine', None)]
        assert printed['status'] == 'PASS_RANK_MEAN_TEST_COHORT_ONLY'
        release.write_columns(base / 'expected.csv', ids, release.rank_mean([tables['coat5'], tables['rad5']]))
    assert final.read_bytes() == (base / 'expected.csv').read_bytes()
    assert not list((k / 'working').glob('*.exp111-tmp'))


@pytest.mark.parametrize('name', ['coat5rad5', 'rad5'])
def test_simulated_notebook_cell_fails_closed_above_budget(monkeypatch, tmp_path, name):
    with pytest.raises(AssertionError, match='v6 time screen failed'):
        simulate_cell(monkeypatch, tmp_path, name, MAX_STUDIES[name] + 1)
    k = tmp_path / 'fake/kaggle'
    assert not (k / 'working/submission.csv').exists()
    assert not any((k / 'working' / ('exp111-' + name)).glob('*/work'))


# ------------------------------------------------------------------ 9. opt-in real checkpoint smoke (CPU)
@pytest.mark.skipif(os.environ.get('EXP111_REAL_SMOKE') != '1',
                    reason='opt-in: loads one real RadR50 (~97 MB) and one real CoAtNet-2 (~290 MB) checkpoint on CPU')
def test_real_staged_checkpoints_cpu_smoke(monkeypatch):
    """serve_arm on CPU for one-member sub-arms of both generated arms (frozen load_checkpoint -> checkpoint_check ->
    wrapper -> frozen predict with the arm's input policy)."""
    torch = pytest.importorskip('torch')
    rt = frozen_runtime(monkeypatch)
    input_modules()
    monkeypatch.setitem(sys.modules, 'raptor_runtime', rt)
    torch.set_num_threads(8)
    out = kernel_dir('coat5rad5')
    p = release.checked_manifest(out / 'release-manifest.json', release.sha(out / 'release-manifest.json'))
    ids = ['9.1']
    for arm, member, tag in ((p['arms'][0], 'coat_f0', 'EXP094L_B8_SEED42'), (p['arms'][1], 'rad_f0', 'EXP107_RADR50_DREAD_F0')):
        sub = {**arm, 'checkpoints': [c for c in arm['checkpoints'] if c['name'] == member]}
        corpus = FakeCorpus(ids, res=224)
        probs, timing, checks, how = release.serve_arm(sub, release.PACKAGE, corpus, ids, 'none', 'cpu', log=lambda *a: None)
        assert checks[0]['tag'] == tag and checks[0]['input_policy'] == arm['input_policy'] and corpus.decodes == 1
        assert np.isfinite(probs).all() and timing['norm'] == arm['norm']
        assert ('center3' in how) == (arm['input_policy'] == 'center3') and ('frozen' in how) == (arm['input_policy'] == 'triplet')
    assert not torch.cuda.is_initialized()
