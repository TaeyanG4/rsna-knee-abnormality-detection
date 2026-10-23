"""CPU tests for the EXP119 v7 release (one shared decode, many arms; H-B-32 stage 1).

No GPU: CUDA initialisation is refused in-process (CUDA_VISIBLE_DEVICES='' does not hide the GPU on this host), every
inference runs with device 'cpu' and amp 'none'. No network. Real-data tests (frozen EXP089 assets, staged packages,
generated kernels, the real-checkpoint CPU parity receipt and the runtime projection file) skip when the files are absent.
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
import zipfile
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


v7 = load('exp119_release_v7_under_test', 'tools/exp119_release_v7.py')
v6 = v7.v6
v5 = load('exp104_release_v5_for_v7', 'tools/exp104_release_v5.py')
exp098 = load('exp098_e1_release_for_v7', 'tools/exp098_e1_release.py')
TRAINING_SRC = ROOT / 'training/src'
if str(TRAINING_SRC) not in sys.path:
    sys.path.append(str(TRAINING_SRC))
ARCH = v7.COAT
RAD_ARCH = 'resnet50'
CONTRACT = 'raptor64_336_c140_s02'
D = 6.52503319258
DS_A = 'taeyangg4/knee-test-assets-a'     # v6 schema: coat c0/c1 + rad r0/r1 + centre-plane code
DS_B = 'taeyangg4/knee-test-assets-b'     # v5 schema: flight f0/f1, no code
KERNELS = v7.OUT_DIR
PARITY = v7.OUT_DIR / 'cpu-parity-v1.json'
PROJECTION = v7.OUT_DIR / 'runtime-projection-v1.json'


def quiet(*a, **k):
    return None


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    """Refuse any CUDA initialisation (the local GPU is busy training)."""
    try:
        import torch
    except ImportError:
        yield
        return

    def refuse(*a, **k):
        raise RuntimeError('CUDA is forbidden in these CPU tests')
    monkeypatch.setattr(torch.cuda, '_lazy_init', refuse)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    yield
    assert not torch.cuda.is_initialized()


# ------------------------------------------------------------------ fixtures
def meta(**over):
    m = {'arch': ARCH, 'res': 8, 'lab': list(v7.TARGETS), 'contract': CONTRACT, 'k_eval': 24, 'norm': 'imagenet',
         'window_policy': 'within-slot-valid', 'epoch': 7, 'tag': 'T', 'amp_train': 'fp16', 'amp_eval': 'fp16'}
    m.update(over)
    return m


def rad_meta(**over):
    return meta(**{'arch': RAD_ARCH, 'norm': 'radimagenet', 'epoch': 19, 'input_policy': 'center3',
                   'init': 'radimagenet', 'init_weights_sha256': v6.RADIMAGENET_SHA256, **over})


def fake_assets(root, members, dataset, schema=None):
    """members: {name: (bytes, meta)} -> assets dir; v6 schema ships the centre-plane code, v5 schema does not."""
    schema = schema or v6.ASSETS_SCHEMA
    root = Path(root)
    (root / 'checkpoints').mkdir(parents=True)
    checkpoints = {}
    for name, (blob, m) in members.items():
        path = root / 'checkpoints' / (name + '.pt')
        path.write_bytes(blob)
        checkpoints[name] = {'path': 'checkpoints/' + name + '.pt', 'sha256': v6.sha(path), 'bytes': len(blob),
                             'meta': m, 'training_code_sha256': {}, 'input_code_sha256': {}, 'source': 'synthetic',
                             'source_sidecars_sha256': {}, 'source_inventory_sha256': None, 'link_mode': 'copy'}
    files = {c['path']: c['sha256'] for c in checkpoints.values()}
    am = {'schema': schema, 'dataset': dataset, 'checkpoints': checkpoints, 'files': files,
          'dataset_metadata_sha256': '0' * 64}
    if schema == v6.ASSETS_SCHEMA:
        (root / v6.CODE_DIR).mkdir()
        for n in v6.CODE_FILES:
            shutil.copyfile(TRAINING_SRC / n, root / v6.CODE_DIR / n)
        files.update({v6.CODE_DIR + '/' + n: s for n, s in v6.CODE_FILES.items()})
        am['code'] = v6.code_record()
    (root / v6.NEW_MANIFEST).write_text(json.dumps(am), encoding='utf-8')
    return am


def two_datasets(tmp_path):
    am_a = fake_assets(tmp_path / 'a', {'c0': (b'coat-0' * 50, meta(tag='C0')), 'c1': (b'coat-1' * 50, meta(tag='C1')),
                                        'r0': (b'rad-0' * 50, rad_meta(tag='R0')),
                                        'r1': (b'rad-1' * 50, rad_meta(tag='R1'))}, DS_A)
    am_b = fake_assets(tmp_path / 'b', {'f0': (b'fl-0' * 50, meta(tag='F0')), 'f1': (b'fl-1' * 50, meta(tag='F1'))},
                       DS_B, 'knee-release-assets-v5')
    return am_a, am_b


def three_arms(am_a, am_b, weights=(1.0, 1.0, 1.0)):
    return [v6.build_arm('coat5', ['c0', 'c1'], 36, am_a, 0.35, weights[0]),
            v6.build_arm('rad5', ['r0', 'r1'], 36, am_a, 0.25, weights[1]),
            v6.build_arm('fl5', ['f0', 'f1'], 36, am_b, 0.35, weights[2])]


def set_record(root, dataset):
    am = json.loads((Path(root) / v6.NEW_MANIFEST).read_text(encoding='utf-8'))
    return {'dataset': dataset, 'schema': am['schema'], 'manifest_sha256': v6.sha(Path(root) / v6.NEW_MANIFEST),
            'input_code': am['schema'] == v6.ASSETS_SCHEMA}


def release_doc(tmp_path, arms, **over):
    sets = [set_record(tmp_path / 'a', DS_A), set_record(tmp_path / 'b', DS_B)]
    bound = {a['name']: (DS_B if a['name'].startswith('fl') else DS_A) for a in arms}
    used = set(bound.values())
    p = {'schema': v7.SCHEMA, 'release': 'synthetic', 'release_source_sha256': v6.sha(v7.__file__),
         'v6_source_sha256': v7.V6_SOURCE_SHA256, 'old_asset_protocol_sha256': v6.ASSETS_PROTOCOL_SHA,
         'asset_sets': [s for s in sets if s['dataset'] in used], 'arm_assets': bound, 'time_policy': v7.TIME,
         'decode': v7.DECODE_POLICY, 'RSS_bytes': v6.RSS, 'output_bytes': v6.OUTPUT, 'prob_clamp': v6.PROB_EPS,
         'retries': 0, 'arms': arms, 'rank_weights': v6.rank_weights(arms) if arms else {}}
    p.update(over)
    return p


def write_doc(path, p):
    Path(path).write_text(json.dumps(p, indent=1), encoding='utf-8')
    return v6.sha(path)


def write_csv(path, header, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fixture_comp(root, ids):
    comp = Path(root)
    (comp / 'test_series').mkdir(parents=True)
    write_csv(comp / 'test.csv', [v7.UID], [[u] for u in sorted(ids)])
    write_csv(comp / 'sample_submission.csv', [v7.UID, *v7.TARGETS], [[u, *([0] * 12)] for u in ids])
    write_csv(comp / 'test_series.csv', [v7.UID, 'SeriesInstanceUID'], [[u, '2.' + u] for u in ids])
    return comp


def tiny(torch, seed, res=8):
    class Tiny(torch.nn.Module):
        """(B, K, 3, r, r) -> (B, 12): window-mean of a linear map."""
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.w = torch.nn.Parameter(torch.randn(3 * res * res, 12, generator=g) * 0.05)
            self.b = torch.nn.Parameter(torch.randn(12, generator=g))

        def forward(self, x):
            return (x.flatten(2) @ self.w).mean(1) + self.b
    return Tiny()


def make_loader(torch, bad=()):
    """A stand-in for the frozen load_checkpoint: a deterministic tiny model per checkpoint name + its staged meta."""
    def loader(path):
        path = Path(path)
        am = json.loads((path.parent.parent / v6.NEW_MANIFEST).read_text(encoding='utf-8'))
        seed = int(hashlib.sha256(path.stem.encode()).hexdigest()[:6], 16)
        if path.stem in bad:
            class Inf(torch.nn.Module):
                def forward(self, x):
                    return torch.full((x.shape[0], 12), float('inf'))
            return Inf(), dict(am['checkpoints'][path.stem]['meta'])
        return tiny(torch, seed), dict(am['checkpoints'][path.stem]['meta'])
    return loader


def frozen_runtime():
    src = v6.OLD_LOCAL / 'training/src'
    if not (src / 'raptor_runtime.py').is_file():
        pytest.skip('frozen EXP089 assets not present')
    protocol = json.loads((v6.OLD_LOCAL / v6.OLD_PROTOCOL).read_text(encoding='utf-8'))
    assert v6.sha(src / 'raptor_runtime.py') == protocol['files']['training/src/raptor_runtime.py']
    common = __import__('raptor_common')
    assert v6.sha(common.__file__) == protocol['files']['training/src/raptor_common.py']
    return load('frozen_raptor_runtime_for_v7', str((src / 'raptor_runtime.py').relative_to(ROOT)))


def input_modules():
    rad, data = __import__('raptor_radimagenet'), __import__('raptor_data')
    assert v6.sha(rad.__file__) == v6.CODE_FILES['raptor_radimagenet.py']
    assert v6.sha(data.__file__) == v6.CODE_FILES['raptor_data.py']


class FakeCorpus:
    """DicomCorpus interface keeping one study, counting decodes, with DicomCorpus-style receipts."""

    def __init__(self, ids, res=8, fail=(), contract=CONTRACT):
        from raptor_common import SLOTS64
        self.spec, self.depth, self.contract = {'slots': SLOTS64}, 64, contract
        rng = np.random.default_rng(0)
        self.data = {u: rng.integers(0, 256, size=(64, res, res), dtype=np.uint8) for u in ids}
        for v in self.data.values():
            v.setflags(write=False)
        self.fail, self.decodes, self.cached, self.receipts = set(fail), 0, None, {}

    def _ensure(self, uid):
        if uid in self.fail:
            raise RuntimeError('DICOM pixels failed: ' + uid)
        if self.cached != uid:
            self.decodes += 1
            self.cached = uid
            self.receipts[uid] = {'status': 'rendered', 'volume_sha256': v7.volume_sha256(self.data[uid]),
                                  'mask_sha256': v7.mask_sha256(self._mask())}
        return self.data[uid]

    def _mask(self):
        m = np.ones(self.depth, np.uint8)
        m.setflags(write=False)
        return m

    def volume(self, uid):
        return self._ensure(uid)

    def mask(self, uid):
        self._ensure(uid)
        return self._mask()


def roots_of(tmp_path, arms):
    return {a['name']: tmp_path / ('b' if a['name'].startswith('fl') else 'a') for a in arms}


# ------------------------------------------------------------------ 1. reuse of the reviewed v6 by import
def test_v6_is_imported_by_pinned_sha_and_reused(tmp_path):
    assert v6.sha(v6.__file__) == v7.V6_SOURCE_SHA256 == v6.sha(ROOT / 'tools/exp111_release_v6.py')
    assert Path(v6.__file__).resolve() == (ROOT / 'tools/exp111_release_v6.py').resolve()
    for name in ('parse_arm_spec', 'build_arm', 'validate_arms', 'checkpoint_check', 'mean_prob_module',
                 'serving_input', 'rank_mean', 'write_columns', 'publish', 'launch_budget', 'validate_resource'):
        assert callable(getattr(v6, name))
    assert v7.TIME == v6.TIME == v5.TIME and v7.TARGETS == v6.TARGETS
    bad = tmp_path / 'bad'
    bad.mkdir()
    (bad / v7.V6_NAME).write_bytes((ROOT / 'tools/exp111_release_v6.py').read_bytes() + b'\n# drift\n')
    with pytest.raises(ValueError, match='differs from the reviewed v6'):
        v7._load_v6(bad)


# ------------------------------------------------------------------ 2. arm config parsing, datasets, weights
def test_parse_pairs():
    assert v7.parse_pairs(['taeyangg4/knee-release-assets-v8=x/y', 'a/b-c-d-e=z'], 'x') == \
        {'taeyangg4/knee-release-assets-v8': 'x/y', 'a/b-c-d-e': 'z'}
    assert v7.parse_pairs(['o/slugs=C:\\a=b'], 'x') == {'o/slugs': 'C:\\a=b'}   # first '=' splits
    for bad in [[], ['noequals'], ['o/slugs='], ['Bad/Slug=x'], ['o/sl=x'], [v6.OLD_DATASET + '=x'],
                [v6.OPENCV_DATASET + '=x'], ['o/slugs=x', 'o/slugs=y']]:
        with pytest.raises(ValueError):
            v7.parse_pairs(bad, 'x')


def test_arm_spec_and_build_arms_over_several_datasets(tmp_path, monkeypatch):
    am_a, am_b = two_datasets(tmp_path)
    old = {'files': {'training/src/' + f: 'h-' + f for f in v6.TRAINING_CODE_BOUND}}
    for am in (am_a, am_b):
        for c in am['checkpoints'].values():
            c['training_code_sha256'] = {f: 'h-' + f for f in v6.TRAINING_CODE_BOUND}
    for name in ('r0', 'r1'):
        am_a['checkpoints'][name]['training_code_sha256']['raptor_data.py'] = v6.CODE_FILES['raptor_data.py']
        am_a['checkpoints'][name]['input_code_sha256'] = {'raptor_radimagenet.py': v6.CODE_FILES['raptor_radimagenet.py']}
    manifests = {DS_A: am_a, DS_B: am_b}
    arms, specs, basis, bound = v7.build_arms(['coat5=c0+c1@36:1', 'rad5=r0+r1@36:1', 'fl5=f0+f1@36:2'], manifests,
                                              old, {})
    assert [(a['name'], a['input_policy'], a['norm'], a['k_eval'], a['weight']) for a in arms] == \
        [('coat5', 'triplet', 'imagenet', 36, 1.0), ('rad5', 'center3', 'radimagenet', 36, 1.0),
         ('fl5', 'triplet', 'imagenet', 36, 2.0)]
    assert bound == {'coat5': DS_A, 'rad5': DS_A, 'fl5': DS_B}
    assert specs == ['coat5=c0+c1@36:1', 'rad5=r0+r1@36:1', 'fl5=f0+f1@36:2'] and 'EXP107' in basis['rad5']
    assert v6.rank_weights(arms) == {'coat5': 0.25, 'rad5': 0.25, 'fl5': 0.5}
    arms, _, _, _ = v7.build_arms(['coat5=c0+c1', 'rad5=r0+r1'], manifests, old, {'rad5': 0.3})
    assert [a['weight'] for a in arms] == [1.0, 1.0] and [a['k_eval'] for a in arms] == [24, 24]
    assert arms[1]['model_seconds_per_study'] == 0.3
    cases = [(['coat5=c0+c1@36:1', 'rad5=r0+r1@36'], 'every arm or for none'),
             (['coat5=c0+c1@36:1'], 'single-arm release has no rank weight'),
             (['mix=c0+f0@36'], 'one assets dataset'),
             (['x=nope@36'], 'exactly one --assets dataset'),
             (['m=c0+r0@36'], 'differ in arch'),
             (['coat5=c0@36', 'rad5=r0@36'], None)]
    for texts, message in cases:
        if message is None:
            v7.build_arms(texts, manifests, old, {})
            continue
        with pytest.raises(ValueError, match=message):
            v7.build_arms(texts, manifests, old, {})
    with pytest.raises(ValueError, match='unknown arm'):
        v7.build_arms(['coat5=c0@36'], manifests, old, {'rad5': 0.3})
    # the same checkpoint name in two mounted datasets is ambiguous
    dup = copy.deepcopy(am_b)
    dup['checkpoints']['c0'] = dup['checkpoints']['f0']
    with pytest.raises(ValueError, match='exactly one --assets dataset'):
        v7.build_arms(['coat5=c0+c1@36'], {DS_A: am_a, DS_B: dup}, old, {})
    # training code bound to the frozen serving copies; centre-plane input code bound to the shipped code
    drift = copy.deepcopy(am_a)
    drift['checkpoints']['c1']['training_code_sha256']['raptor_runtime.py'] = 'other'
    with pytest.raises(ValueError, match='differing from the frozen serving copy'):
        v7.build_arms(['coat5=c0+c1@36'], {DS_A: drift}, old, {})
    drift = copy.deepcopy(am_a)
    drift['checkpoints']['r1']['input_code_sha256'] = {'raptor_radimagenet.py': '0' * 64}
    with pytest.raises(ValueError, match='input code differing'):
        v7.build_arms(['rad5=r0+r1@36'], {DS_A: drift}, old, {})
    # a centre-plane member staged in a v5-schema dataset (no shipped input code) is refused
    v5_only = copy.deepcopy(am_a)
    v5_only['schema'] = 'knee-release-assets-v5'
    with pytest.raises(ValueError, match='ships the input code'):
        v7.build_arms(['rad5=r0+r1@36'], {DS_A: v5_only}, old, {})


def test_equal_weights_are_exp098_bit_for_bit_and_fixed_weights_are_v5():
    code = exp098.CODE
    ns = {}
    exec(code[code.index('def rank(col):'):code.index('cols=[]')], ns)
    rng = random.Random(5)
    tables = [[[rng.choice([0.1, 0.5]) if t % 4 == 0 else rng.random() for t in range(12)] for _ in range(101)]
              for _ in range(3)]
    want = []
    for t in range(12):
        ranks = [ns['rank']([r[t] for r in tab]) for tab in tables]
        want.append([(a + b + c) / 3 for a, b, c in zip(*ranks)])
    got = v6.rank_mean(tables, [1.0, 1.0, 1.0])
    assert got == v5.rank_mean(tables, [1.0, 1.0, 1.0])
    assert max(abs(x - y) for c1, c2 in zip(got, want) for x, y in zip(c1, c2)) <= 1e-15
    two = tables[:2]
    want2 = [[(a + b) / 2 for a, b in zip(ns['rank']([r[t] for r in two[0]]), ns['rank']([r[t] for r in two[1]]))]
             for t in range(12)]
    assert v6.rank_mean(two, [1.0, 1.0]) == want2 == v6.rank_mean(two)
    weighted = v6.rank_mean(tables, [0.5, 0.3, 0.2])
    assert weighted == v5.rank_mean(tables, [0.5, 0.3, 0.2]) != got
    assert all(0 <= v <= 1 for col in weighted for v in col)


# ------------------------------------------------------------------ 3. release manifest validation
def test_release_manifest_validation(tmp_path):
    am_a, am_b = two_datasets(tmp_path)
    arms = three_arms(am_a, am_b)
    path = tmp_path / 'release-manifest.json'
    p = v7.checked_manifest(path, write_doc(path, release_doc(tmp_path, arms)))
    assert p['arm_assets'] == {'coat5': DS_A, 'rad5': DS_A, 'fl5': DS_B}
    assert p['rank_weights'] == {'coat5': 1 / 3, 'rad5': 1 / 3, 'fl5': 1 / 3}
    with pytest.raises(ValueError, match='Release manifest changed'):
        v7.checked_manifest(path, '0' * 64)
    coat, rad, fl = arms
    big = copy.deepcopy(coat)
    big['checkpoints'][0]['bytes'] = v7.MAX_RESIDENT_CHECKPOINT_BYTES
    other_contract = {**fl, 'contract': 'raptor80_336_c140_s02',
                      'checkpoints': [{**c, 'meta': {**c['meta'], 'contract': 'raptor80_336_c140_s02'}}
                                      for c in fl['checkpoints']]}
    no_code_sets = [{**set_record(tmp_path / 'a', DS_A), 'schema': 'knee-release-assets-v5', 'input_code': False},
                    set_record(tmp_path / 'b', DS_B)]
    bad = [({'schema': v6.SCHEMA}, 'policy'),
           ({'decode': {**v7.DECODE_POLICY, 'decodes_per_study': 2}}, 'policy'),
           ({'time_policy': {**v7.TIME, 'inflation': 1.0}}, 'policy'),
           ({'release_source_sha256': v6.sha(v6.__file__)}, 'source changed'),
           ({'v6_source_sha256': '0' * 64}, 'Embedded v6 source changed'),
           ({'arms': [coat, rad, other_contract]}, 'shared raptor64_336_c140_s02 volume'),
           ({'arms': [coat, {**rad, 'name': 'combine'}, fl], 'arm_assets': {'coat5': DS_A, 'combine': DS_A,
                                                                             'fl5': DS_B}}, 'Unsafe arm name|Reserved'),
           ({'arms': [coat, {**rad, 'name': 'shared'}, fl], 'arm_assets': {'coat5': DS_A, 'shared': DS_A,
                                                                            'fl5': DS_B}}, 'Reserved'),
           ({'arms': [coat, rad, {**fl, 'checkpoints': fl['checkpoints'][:1] + coat['checkpoints'][:1]}]},
            'differs from arm|only one arm|Duplicate'),
           ({'arms': [coat, {**coat, 'name': 'coat5b'}, fl], 'arm_assets': {'coat5': DS_A, 'coat5b': DS_A,
                                                                             'fl5': DS_B}}, 'only one arm'),
           ({'arms': [big, rad, fl]}, 'Resident checkpoint bytes'),
           ({'asset_sets': no_code_sets}, 'ships the input code'),
           ({'arm_assets': {'coat5': DS_A, 'rad5': DS_A, 'fl5': DS_A}}, 'binding differs'),
           ({'arm_assets': {'rad5': DS_A, 'coat5': DS_A, 'fl5': DS_B}}, 'binding differs'),
           ({'asset_sets': [set_record(tmp_path / 'a', DS_A)]}, 'binding differs'),
           ({'asset_sets': [set_record(tmp_path / 'a', DS_A)] * 2}, 'Duplicate assets dataset'),
           ({'asset_sets': [{**set_record(tmp_path / 'a', DS_A), 'input_code': False}, set_record(tmp_path / 'b', DS_B)]},
            'Asset set record differs'),
           ({'asset_sets': [{**set_record(tmp_path / 'a', DS_A), 'package': 'x'}, set_record(tmp_path / 'b', DS_B)]},
            'Asset sets malformed'),
           ({'asset_sets': [{**set_record(tmp_path / 'a', DS_A), 'dataset': v6.OLD_DATASET},
                            set_record(tmp_path / 'b', DS_B)]}, 'Unsafe assets dataset'),
           ({'rank_weights': {'coat5': 0.5, 'rad5': 0.25, 'fl5': 0.25}}, 'Recorded rank weights'),
           ({'arms': [coat, {**rad, 'weight': 0.0}, fl]}, 'Rank weight'),
           ({'arms': [coat, {**rad, 'input_policy': 'triplet'}, fl]}, 'norm/window policy'),
           ({'retries': 1}, 'policy')]
    for i, (over, message) in enumerate(bad):
        doc = release_doc(tmp_path, arms)
        doc.update(over)
        if 'arms' in over and 'rank_weights' not in over:
            try:
                doc['rank_weights'] = v6.rank_weights(doc['arms'])
            except ValueError:
                pass
        d = write_doc(tmp_path / ('bad-%d.json' % i), doc)
        with pytest.raises(ValueError, match=message):
            v7.checked_manifest(tmp_path / ('bad-%d.json' % i), d)


def test_asset_sets_bind_every_mounted_dataset(tmp_path):
    am_a, am_b = two_datasets(tmp_path)
    arms = three_arms(am_a, am_b)
    p = release_doc(tmp_path, arms)
    roots = {DS_A: tmp_path / 'a', DS_B: tmp_path / 'b'}
    assert set(v7.verify_asset_sets(p, roots, arms)) == {DS_A, DS_B}
    assert v7.code_root(p, roots) == tmp_path / 'a'
    assert v7.code_root({**p, 'arms': [arms[0], arms[2]]}, roots) is None
    with pytest.raises(ValueError, match='Mounted assets datasets differ'):
        v7.verify_asset_sets(p, {DS_A: tmp_path / 'a'}, arms)
    (tmp_path / 'b/checkpoints/f1.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='sha256 differs'):
        v7.verify_asset_sets(p, roots, arms)
    (tmp_path / 'b/checkpoints/f1.pt').unlink()
    with pytest.raises(ValueError, match='Missing checkpoint'):
        v7.verify_asset_sets(p, roots, arms)
    shutil.rmtree(tmp_path / 'b')
    am_b = fake_assets(tmp_path / 'b', {'f0': (b'fl-0' * 50, meta(tag='F0')), 'f1': (b'fl-1' * 50, meta(tag='F1'))},
                       DS_B, 'knee-release-assets-v5')
    v7.verify_asset_sets(p, roots, arms)
    code = tmp_path / 'a' / v6.CODE_DIR / 'raptor_radimagenet.py'
    original = code.read_bytes()
    code.write_bytes(original + b'\n# tampered\n')
    with pytest.raises(ValueError, match='Input code changed'):
        v7.verify_asset_sets(p, roots, arms)
    code.write_bytes(original)
    am_b['checkpoints']['f0']['meta'] = meta(tag='other')
    (tmp_path / 'b' / v6.NEW_MANIFEST).write_text(json.dumps(am_b), encoding='utf-8')
    with pytest.raises(ValueError, match='Assets manifest changed'):
        v7.verify_asset_sets(p, roots, arms)
    p2 = {**p, 'asset_sets': [set_record(tmp_path / 'a', DS_A), set_record(tmp_path / 'b', DS_B)]}
    with pytest.raises(ValueError, match='not bound by the assets manifest'):
        v7.verify_asset_sets(p2, roots, arms)


# ------------------------------------------------------------------ 4. runtime screen with ONE decode
def arms_of(*spec):
    return [{'name': 'a%d' % i, 'checkpoints': [None] * k, 'model_seconds_per_study': m} for i, (k, m) in enumerate(spec)]


def test_time_screen_has_one_decode():
    coat_rad = arms_of((5, 0.35), (5, 0.25))
    plan = v7.operational_plan(1300, coat_rad)
    assert plan['projected_seconds'] == pytest.approx(1.3 * (D + 1.75 + 1.25) * 1300 + 600, abs=1e-6)   # 16,697.3
    assert plan['separate_workers_projected_seconds'] == pytest.approx(v6.operational_plan(1300, coat_rad)['projected_seconds'])
    assert plan['separate_workers_projected_seconds'] == pytest.approx(27724.61219, abs=1e-3)
    assert plan['decode_projected_seconds'] == pytest.approx(1.3 * D * 1300)
    assert plan['arm_model_projected_seconds'] == pytest.approx({'a0': 1.3 * 1.75 * 1300, 'a1': 1.3 * 1.25 * 1300})
    assert plan['internal_seconds'] == 32100 and v7.operational_plan(3, coat_rad)['internal_seconds'] == 600
    assert v7.max_studies(coat_rad) == 2422
    v7.operational_plan(2422, coat_rad)
    with pytest.raises(ValueError, match='time screen'):
        v7.operational_plan(2423, coat_rad)
    assert v7.launch_projection(plan) == pytest.approx(plan['projected_seconds'] - 600 + 300)
    v6.launch_budget(1000.0, 1000.5, 1300, v7.launch_projection(plan), 600.0)
    # 3 and 4 arms fit with one decode; the v5/v6 per-arm screen fails them (H-B-29 note: 41,955 s + 600)
    three = arms_of((5, 0.35), (5, 0.25), (5, 0.25))
    four = arms_of((5, 0.35), (5, 0.25), (5, 0.25), (5, 0.35))
    assert v7.operational_plan(1300, three)['projected_seconds'] == pytest.approx(18809.8, abs=0.1)
    assert v7.operational_plan(1300, four)['projected_seconds'] == pytest.approx(21767.3, abs=0.1)
    labels = arms_of((5, 0.35), (5, 0.35), (5, 0.35))
    assert v7.separate_seconds(1300, labels) == pytest.approx(41954.4 + 600, abs=0.1)
    with pytest.raises(ValueError, match='time screen'):
        v6.operational_plan(1300, labels)
    assert v7.operational_plan(1300, labels)['projected_seconds'] == pytest.approx(20499.8, abs=0.1)
    with pytest.raises(ValueError):
        v7.operational_plan(0, coat_rad)
    with pytest.raises(ValueError):
        v7.operational_plan(10, arms_of((5, float('nan'))))


def test_projection_report_and_file():
    r = v7.projection_report(1300)
    by = {s['label']: s for s in r['scenarios']}
    assert len(r['scenarios']) == 6 and all(s['v7_passes_screen'] for s in r['scenarios'])
    assert [s['v5_v6_passes_screen'] for s in r['scenarios']] == [True, False, False, False, False, False]
    two = by['2 arms: coat5 + rad5']
    assert two['v7_shared_screen_seconds'] == pytest.approx(16697.3, abs=0.1) and two['v7_max_studies'] == 2422
    assert two['v5_v6_separate_screen_seconds'] == pytest.approx(27724.6, abs=0.1) and two['v5_v6_max_studies'] == 1437
    assert by['3 arms: coat5 + rad5 + rad5s43']['v7_shared_screen_seconds'] == pytest.approx(18809.8, abs=0.1)
    assert by['4 arms: coat5 + rad5 + rad5s43 + fl5']['v7_shared_screen_seconds'] == pytest.approx(21767.3, abs=0.1)
    assert by['3 arms H-B-29 label sets: coat5 + fl5 + rl5']['v5_v6_separate_screen_seconds'] == \
        pytest.approx(42554.4, abs=0.1)
    for s in r['scenarios']:
        assert s['saved_screen_seconds'] == pytest.approx(s['v5_v6_separate_screen_seconds'] - s['v7_shared_screen_seconds'],
                                                          abs=0.11)
        assert s['saved_screen_seconds'] == pytest.approx(1.3 * D * 1300 * (len(s['arms']) - 1), abs=0.2)
    if PROJECTION.is_file():
        disk = json.loads(PROJECTION.read_text(encoding='utf-8'))
        assert disk['tool_sha256'] == v6.sha(v7.__file__), 'projection file is stale: regenerate with `project`'
        assert {k: v for k, v in disk.items() if k not in ('created_at', 'tool_sha256')} == json.loads(json.dumps(r))


# ------------------------------------------------------------------ 5. shared serving = separate v5/v6 workers
def test_shared_study_view_is_read_only_and_single_study():
    vol = np.arange(64 * 4, dtype=np.uint8).reshape(64, 2, 2)
    view = v7.SharedStudy('1.1', vol, np.ones(64, np.uint8), 64, {'slots': []})
    assert view.volume('1.1') is view.volume('1.1') and not view.volume('1.1').flags.writeable
    assert vol.flags.writeable    # the caller's array is untouched; only the view is read-only
    with pytest.raises(ValueError):
        view.volume('1.1')[0, 0, 0] = 1
    with pytest.raises(KeyError):
        view.mask('1.2')
    assert view.reads == 4


def test_shared_serving_equals_separate_v6_and_v5_workers_bit_for_bit(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    rt = frozen_runtime()
    input_modules()
    monkeypatch.setitem(sys.modules, 'raptor_runtime', rt)
    common = __import__('raptor_common')
    loader = make_loader(torch)
    monkeypatch.setattr(common, 'load_checkpoint', loader)
    am_a, am_b = two_datasets(tmp_path)
    arms = three_arms(am_a, am_b)
    roots = roots_of(tmp_path, arms)
    ids = ['1.%d' % i for i in range(6)]
    corpus = FakeCorpus(ids)
    state = {}
    served = v7.serve_shared(arms, roots, corpus, ids, 'none', 'cpu', state, log=quiet, runtime_module=rt, loader=loader)
    assert corpus.decodes == len(ids)                                   # ONE decode per study for three arms
    assert served['decode']['decodes'] == len(ids) and served['scored_ids'] == ids and not served['unscored']
    assert [x['uid'] for x in served['decode']['studies']] == ids and served['decode']['shared_by'] == ['coat5', 'rad5', 'fl5']
    assert [x['volume_sha256'] for x in served['decode']['studies']] == [v7.volume_sha256(corpus.data[u]) for u in ids]
    assert set(state['models']) == {'coat5', 'rad5', 'fl5'} and state['loading'] == []
    separate_decodes = 0
    for arm in arms:
        name = arm['name']
        own = FakeCorpus(ids)
        probs, timing, checks, how = v6.serve_arm(arm, roots[name], own, ids, 'none', 'cpu', {}, log=quiet)
        separate_decodes += own.decodes
        assert served['probs'][name].dtype == np.float32 and np.array_equal(served['probs'][name], probs)
        assert served['timing'][name]['input_tensor_hashes'] == timing['input_tensor_hashes']
        assert served['timing'][name]['ordered_inputs_sha256'] == timing['ordered_inputs_sha256']
        assert v7.ordered_inputs_sha256(timing['input_tensor_hashes']) == timing['ordered_inputs_sha256']
        assert served['checks'][name] == checks and served['input_functions'][name] == how
        assert ('center3' in how) == (arm['input_policy'] == 'center3')
        if arm['input_policy'] == 'triplet':                            # the v5 worker path (its own wrapper)
            members = [loader(Path(roots[name]) / ck['path'])[0].eval() for ck in arm['checkpoints']]
            p5, t5 = rt.predict(v5.mean_prob_module(members), FakeCorpus(ids), ids, arm['k_eval'], arm['res'], 'none',
                                'cpu', norm=arm['norm'], batch=1, log=quiet, window_policy=arm['window_policy'])
            assert np.array_equal(p5, served['probs'][name])
            assert t5['ordered_inputs_sha256'] == served['timing'][name]['ordered_inputs_sha256']
    assert separate_decodes == len(arms) * len(ids)
    # arm CSVs written like the worker are byte-identical to the separate workers' CSVs
    common.write_predictions(tmp_path / 'shared.csv', ids, served['probs']['rad5'])
    common.write_predictions(tmp_path / 'separate.csv', ids,
                             v6.serve_arm(arms[1], roots['rad5'], FakeCorpus(ids), ids, 'none', 'cpu', {}, log=quiet)[0])
    assert (tmp_path / 'shared.csv').read_bytes() == (tmp_path / 'separate.csv').read_bytes()


def test_shared_serving_fails_closed(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    rt = frozen_runtime()
    input_modules()
    am_a, am_b = two_datasets(tmp_path)
    arms = three_arms(am_a, am_b)
    roots = roots_of(tmp_path, arms)
    ids = ['1.%d' % i for i in range(4)]
    loader = make_loader(torch)
    # a failed decode stops the release (v5/v6 semantics) ...
    with pytest.raises(RuntimeError, match='input preparation failed for 1.2'):
        v7.serve_shared(arms, roots, FakeCorpus(ids, fail={'1.2'}), ids, 'none', 'cpu', {}, log=quiet,
                        runtime_module=rt, loader=loader)
    # ... unless a stage-2 caller asks to skip it (the anchor keeps its rank)
    out = v7.serve_shared(arms, roots, FakeCorpus(ids, fail={'1.2'}), ids, 'none', 'cpu', {}, log=quiet,
                          runtime_module=rt, loader=loader, skip_failed_inputs=True)
    assert out['scored_ids'] == ['1.0', '1.1', '1.3'] and list(out['unscored']) == ['1.2']
    full = v7.serve_shared(arms, roots, FakeCorpus(ids), ids, 'none', 'cpu', {}, log=quiet, runtime_module=rt,
                           loader=loader)
    assert np.array_equal(out['probs']['rad5'], full['probs']['rad5'][[0, 1, 3]])
    # a non-finite member logit fails closed; loaded modules stay in state for the caller's cleanup
    state = {}
    with pytest.raises(RuntimeError, match='nonfinite member logit'):
        v7.serve_shared(arms, roots, FakeCorpus(ids), ids, 'none', 'cpu', state, log=quiet, runtime_module=rt,
                        loader=make_loader(torch, bad={'r1'}))
    assert set(state['models']) == {'coat5', 'rad5', 'fl5'}
    # a decoded volume that differs from its DICOM receipt is refused
    lying = FakeCorpus(ids)
    real = lying._ensure

    def ensure(uid):
        v = real(uid)
        lying.receipts[uid] = {**lying.receipts[uid], 'volume_sha256': '0' * 64}
        return v
    lying._ensure = ensure
    with pytest.raises(ValueError, match='differs from its DICOM receipt'):
        v7.serve_shared(arms, roots, lying, ids, 'none', 'cpu', {}, log=quiet, runtime_module=rt, loader=loader)
    # checkpoint metadata that differs from the staged record is refused before any study is served
    wrong = make_loader(torch)

    def bad_meta(path):
        model, m = wrong(path)
        return model, {**m, 'tag': 'other'}
    with pytest.raises(ValueError, match='staged record'):
        v7.serve_shared(arms, roots, FakeCorpus(ids), ids, 'none', 'cpu', {}, log=quiet, runtime_module=rt,
                        loader=bad_meta)
    with pytest.raises(ValueError, match='unique'):
        v7.serve_shared(arms, roots, FakeCorpus(ids), ids + ids[:1], 'none', 'cpu', {}, log=quiet, runtime_module=rt,
                        loader=loader)


# ------------------------------------------------------------------ 6. receipts, run (stubbed supervisor) and combine
def supervisor_receipt(argv, pid, timeout):
    events = {'values': {'memory.events': {'oom': 0, 'oom_kill': 0, 'max': 3}, 'memory.peak': 1, 'memory.max': 'max'},
              'errors': []}
    return {'schema': 'process-resource-receipt-v1', 'status': 'COMPLETE', 'failure': None, 'child_exit_code': 0,
            'child_exit_code_before_cleanup': 0, 'wrapper_exit_code': 0, 'argv': argv, 'child_identity': {'pid': pid},
            'cleanup': {'actions': [], 'errors': [], 'survivors': []}, 'retries': 0, 'timeout_seconds': timeout,
            'elapsed_seconds': 1.0, 'sample_count': 3, 'sampled_tree_rss_max_bytes': 2**30, 'failed_sample_attempts': 0,
            'access_failure_count': 0, 'access_failures': [], 'cgroup_telemetry_complete': True,
            'cgroup_start': events, 'cgroup_end': events}


def fake_worker(rt, loader, member_roots, mutate=None):
    """Stands in for process_resource_receipt.py + the Linux/T4 worker: serves the arms on CPU with serve_shared and
    writes the receipts exactly as worker() assembles them (arm_receipt), plus a valid supervisor receipt."""
    common = __import__('raptor_common')

    def run_stub(cmd, env=None, **kw):
        sup, timeout = Path(cmd[cmd.index('--out') + 1]), float(cmd[cmd.index('--timeout') + 1])
        argv = cmd[cmd.index('--') + 1:]
        assert argv[2] == 'worker' and '--timeout' not in argv
        opts = dict(zip(argv[3::2], argv[4::2]))
        work = Path(opts['--work'])
        launch = json.loads((work / 'launch.json').read_text(encoding='utf-8'))
        assert launch['worker_argv'] == argv and launch['worker_timeout_seconds'] == timeout
        p = v7.checked_manifest(opts['--manifest'], opts['--sha256'])
        contract = v6.test_contract(opts['--competition'])
        corpus = FakeCorpus(contract['ids'])
        served = v7.serve_shared(p['arms'], member_roots, corpus, contract['ids'], 'none', 'cpu', {}, log=quiet,
                                 runtime_module=rt, loader=loader)
        out = work / 'worker'
        out.mkdir()
        r = {'schema': v7.SCHEMA, 'status': 'PASS', 'pid': 4242, 'release': p['release'],
             'arm_order': [a['name'] for a in p['arms']], 'release_manifest_sha256': opts['--sha256'], 'arms': {},
             'cleanup_errors': [], 'studies': contract['studies'], 'targets': v7.TARGETS,
             'runtime': {'effective_amp': 'fp16', 'policy': 'strict', 'precision_fallback_allowed': False,
                         'backends': {'cudnn_allow_tf32': False, 'matmul_allow_tf32': False}},
             'decode': served['decode'], 'load_seconds': served['load_seconds'], 'serve_seconds': served['seconds'],
             'test_contract': contract, 'DICOM_receipts': corpus.receipts}
        for arm in p['arms']:
            path = out / (arm['name'] + '.csv')
            common.write_predictions(path, contract['ids'], served['probs'][arm['name']])
            code = v6.code_record() if arm['input_policy'] == 'center3' else {}
            r['arms'][arm['name']] = v7.arm_receipt(arm, p, served, code, v6.sha(path))
        if mutate:
            mutate(r, out)
        (out / 'receipt.json').write_text(json.dumps(r), encoding='utf-8')
        sup.mkdir(parents=True)
        (sup / 'receipt.json').write_text(json.dumps(supervisor_receipt(argv, 4242, timeout)), encoding='utf-8')
        for n in ('stdout.log', 'stderr.log'):
            (sup / n).write_text('', encoding='utf-8')
        return SimpleNamespace(returncode=0)
    return run_stub


def run_setup(tmp_path, monkeypatch, n=5, weights=(1.0, 1.0, 1.0), mutate=None):
    torch = pytest.importorskip('torch')
    rt = frozen_runtime()
    input_modules()
    if not (v6.OLD_LOCAL / 'process_resource_receipt.py').is_file():
        pytest.skip('frozen EXP089 assets not present')
    am_a, am_b = two_datasets(tmp_path)
    arms = three_arms(am_a, am_b, weights)
    digest = write_doc(tmp_path / 'release-manifest.json', release_doc(tmp_path, arms))
    ids = ['1.2.840.%d' % (500 + 3 * i) for i in range(n)]
    comp = fixture_comp(tmp_path / 'comp', ids)
    monkeypatch.setattr(v7, 'subprocess', SimpleNamespace(run=fake_worker(rt, make_loader(torch),
                                                                           roots_of(tmp_path, arms), mutate)))
    a = SimpleNamespace(manifest=str(tmp_path / 'release-manifest.json'), sha256=digest, assets=str(v6.OLD_LOCAL),
                        asset_root=[DS_A + '=' + str(tmp_path / 'a'), DS_B + '=' + str(tmp_path / 'b')],
                        base=str(tmp_path / 'base'), final=str(tmp_path / 'submission.csv'), competition=str(comp),
                        opencv='opencv.whl', started=time.monotonic(), bootstrap_max_seconds=600.0)
    return a, arms, ids


def test_run_publishes_every_arm_from_one_worker_and_combine_equals_v6(tmp_path, monkeypatch):
    a, arms, ids = run_setup(tmp_path, monkeypatch)
    r = v7.run(a)
    base = Path(a.base)
    assert r['status'] == 'PASS_TEST_COHORT_ONLY' and r['arm_order'] == ['coat5', 'rad5', 'fl5']
    assert r['decode']['decodes'] == len(ids) and 'submission_sha256' not in r
    assert json.loads((base / 'shared/work/result.json').read_text(encoding='utf-8')) == json.loads(json.dumps(r))
    for arm in arms:
        rec = r['arms'][arm['name']]
        assert rec['path'] == str(base / arm['name'] / 'arm.csv') and v6.sha(rec['path']) == rec['submission_sha256']
        assert rec['assets_dataset'] == (DS_B if arm['name'] == 'fl5' else DS_A)
    with zipfile.ZipFile(base / 'shared/work/evidence.zip') as z:
        assert set(z.namelist()) == {'launch.json', 'worker/receipt.json', 'worker/coat5.csv', 'worker/rad5.csv',
                                     'worker/fl5.csv', 'supervisor/receipt.json', 'supervisor/stdout.log',
                                     'supervisor/stderr.log'}
    launch = json.loads((base / 'shared/work/launch.json').read_text(encoding='utf-8'))
    assert launch['launch_projection_seconds'] == pytest.approx(launch['operational_plan']['projected_seconds'] - 300)
    assert not Path(a.final).exists() and not list(base.rglob('*.exp111-tmp'))
    c = SimpleNamespace(manifest=a.manifest, sha256=a.sha256, base=a.base, competition=a.competition, out=a.final,
                        started=a.started)
    s = v7.combine(c)
    assert s['status'] == 'PASS_RANK_MEAN_TEST_COHORT_ONLY' and s['rank_weights'] == {'coat5': 1 / 3, 'rad5': 1 / 3,
                                                                                     'fl5': 1 / 3}
    assert v6.sha(a.final) == s['submission_sha256'] and s['shared_decode']['decodes'] == len(ids)
    # v6 combine over the same arm CSVs (a v6 manifest and v6 per-arm results) writes the identical bytes
    v6doc = {'schema': v6.SCHEMA, 'release': 'synthetic', 'release_source_sha256': v6.sha(v6.__file__),
             'old_asset_protocol_sha256': v6.ASSETS_PROTOCOL_SHA, 'new_assets_manifest_sha256': '1' * 64,
             'new_asset_dataset': 'taeyangg4/knee-release-assets-v8', 'time_policy': v6.TIME, 'RSS_bytes': v6.RSS,
             'output_bytes': v6.OUTPUT, 'prob_clamp': v6.PROB_EPS, 'retries': 0, 'arms': arms,
             'rank_weights': v6.rank_weights(arms)}
    d6 = write_doc(tmp_path / 'v6-manifest.json', v6doc)
    b6 = tmp_path / 'base6'
    for arm in arms:
        (b6 / arm['name'] / 'work').mkdir(parents=True)
        shutil.copyfile(base / arm['name'] / 'arm.csv', b6 / arm['name'] / 'arm.csv')
        (b6 / arm['name'] / 'work/result.json').write_text(json.dumps(
            {'status': 'PASS_TEST_COHORT_ONLY', 'arm': arm['name'], 'release_manifest_sha256': d6, 'studies': len(ids),
             'submission_sha256': v6.sha(b6 / arm['name'] / 'arm.csv'), 'elapsed_seconds': 1.0,
             'sampled_tree_rss_bytes': 1, 'n_models': 2, 'k_eval': 36, 'rank_weight': arm['weight']}), encoding='utf-8')
    v6.combine(SimpleNamespace(manifest=str(tmp_path / 'v6-manifest.json'), sha256=d6, base=str(b6),
                               competition=a.competition, out=str(tmp_path / 'v6-submission.csv'),
                               started=time.monotonic()))
    assert Path(a.final).read_bytes() == (tmp_path / 'v6-submission.csv').read_bytes()
    # a second run into the same base is refused (outputs preserved)
    with pytest.raises(ValueError, match='Preserve previous outputs'):
        v7.run(a)


def test_run_with_weights_and_tampered_arm_fails_combine(tmp_path, monkeypatch):
    a, arms, ids = run_setup(tmp_path, monkeypatch, n=4, weights=(0.5, 0.3, 0.2))
    v7.run(a)
    base = Path(a.base)
    tables = [v6.load_probabilities(base / x['name'] / 'arm.csv', ids) for x in arms]
    c = SimpleNamespace(manifest=a.manifest, sha256=a.sha256, base=a.base, competition=a.competition, out=a.final,
                        started=a.started)
    s = v7.combine(c)
    v6.write_columns(tmp_path / 'expected.csv', ids, v6.rank_mean(tables, [0.5, 0.3, 0.2]))
    assert Path(a.final).read_bytes() == (tmp_path / 'expected.csv').read_bytes() and 'weighted' in s['formula']
    Path(a.final).unlink()
    shutil.rmtree(base / 'combine')
    write_csv(base / 'rad5/arm.csv', [v7.UID, *v7.TARGETS], [[u, *([0.5] * 12)] for u in ids])
    with pytest.raises(ValueError, match='Arm output changed'):
        v7.combine(c)
    assert not Path(a.final).exists() and (base / 'combine/failure.json').is_file()


def change_first_value(path):
    """Replace the last probability of the first data row (a valid CSV with other bytes than the receipt)."""
    lines = Path(path).read_text(encoding='utf-8').split('\n')
    lines[1] = lines[1].rsplit(',', 1)[0] + ',0.123'
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


@pytest.mark.parametrize('mutation,message', [
    (lambda r, out: r['decode'].update(decodes=r['decode']['decodes'] + 1), 'Shared decode trace failed'),
    (lambda r, out: r['decode']['studies'].reverse(), 'Shared decode trace failed'),
    (lambda r, out: r['DICOM_receipts'][r['decode']['studies'][0]['uid']].update(volume_sha256='0' * 64),
     'differs from the DICOM receipts'),
    (lambda r, out: r['arms']['rad5']['checkpoints'].pop(), 'Worker ensemble differs'),
    (lambda r, out: r['arms']['fl5'].update(assets_dataset=DS_A), 'Worker ensemble differs'),
    (lambda r, out: r['arms']['rad5'].update(input_code={}), 'Worker input differs'),
    (lambda r, out: r['arms']['coat5']['timing']['input_tensor_hashes'][0].update(sha256='0' * 64),
     'Input/finite trace failed'),
    (lambda r, out: r['runtime'].update(effective_amp='fp32'), 'Runtime differs'),
    (lambda r, out: r['cleanup_errors'].append('x'), 'Worker contract failed'),
    (lambda r, out: change_first_value(out / 'fl5.csv'), 'Worker output changed'),
])
def test_run_fails_closed_on_bad_worker_evidence(tmp_path, monkeypatch, mutation, message):
    a, arms, ids = run_setup(tmp_path, monkeypatch, n=4, mutate=mutation)
    with pytest.raises(ValueError, match=message):
        v7.run(a)
    base = Path(a.base)
    assert not any((base / x['name'] / 'arm.csv').exists() for x in arms) and not Path(a.final).exists()
    assert json.loads((base / 'shared/work/failure.json').read_text(encoding='utf-8'))['status'] == 'FAIL'


def test_single_arm_run_publishes_the_final_submission(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    rt = frozen_runtime()
    input_modules()
    if not (v6.OLD_LOCAL / 'process_resource_receipt.py').is_file():
        pytest.skip('frozen EXP089 assets not present')
    am_a, am_b = two_datasets(tmp_path)
    arm = v6.build_arm('rad5', ['r0', 'r1'], 36, am_a, 0.25, 1.0)
    digest = write_doc(tmp_path / 'm.json', release_doc(tmp_path, [arm]))
    ids = ['1.1', '1.2', '1.3']
    comp = fixture_comp(tmp_path / 'comp', ids)
    monkeypatch.setattr(v7, 'subprocess', SimpleNamespace(run=fake_worker(rt, make_loader(torch),
                                                                           {'rad5': tmp_path / 'a'})))
    a = SimpleNamespace(manifest=str(tmp_path / 'm.json'), sha256=digest, assets=str(v6.OLD_LOCAL),
                        asset_root=[DS_A + '=' + str(tmp_path / 'a')], base=str(tmp_path / 'base'),
                        final=str(tmp_path / 'submission.csv'), competition=str(comp), opencv='o.whl',
                        started=time.monotonic(), bootstrap_max_seconds=600.0)
    r = v7.run(a)
    assert r['submission_sha256'] == v6.sha(a.final) == r['arms']['rad5']['submission_sha256']
    assert r['operational_plan']['internal_seconds'] == 600 and not (tmp_path / 'base/rad5').exists()
    with pytest.raises(ValueError, match='two or more arms'):
        v7.combine(SimpleNamespace(manifest=a.manifest, sha256=digest, base=a.base, competition=str(comp),
                                   out=str(tmp_path / 'x.csv'), started=a.started))


# ------------------------------------------------------------------ 7. notebook generation, line length, docker pin
def test_render_code_line_length_round_trip_and_no_timeout():
    v6_source = Path(v6.__file__).read_bytes()
    big = os.urandom(150_000)
    text = json.dumps({'k%d' % i: 'v"\\é\n' * 40 for i in range(120)}, indent=2, ensure_ascii=False) + '\n'
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    code = v7.render_code('coat5rad5', v6_source, big, text, digest)
    lines = code.split('\n')
    assert max(len(line) for line in lines) <= v6.MAX_NOTEBOOK_LINE
    assert max(len(line) for line in lines) <= 2 * v6.SOURCE_CHUNK
    assert max(len(line) for line in v7.CODE.split('\n')) <= 1000
    b85 = __import__('base64').b85decode
    src = code.split('\nsource=base64.b85decode(', 1)[1].split('\nassert hashlib.sha256(source)', 1)[0].rstrip()[:-1]
    assert b85(eval(src, {})) == big
    v6src = code.split('\nv6=base64.b85decode(', 1)[1].split('\nassert hashlib.sha256(v6)', 1)[0].rstrip()[:-1]
    assert b85(eval(v6src, {})) == v6_source
    assert eval(code.split('\nMANIFEST=', 1)[1].split('\nDIGEST=', 1)[0], {}) == text
    assert "Path('/kaggle/working/exp119-coat5rad5')" in code and "'exp111_release_v6.py'" in code
    assert "'exp119_release_v7.py'" in code and 'v7 time screen failed' in code and '--timeout' not in code
    with pytest.raises(ValueError, match='reviewed v6'):
        v7.render_code('coat5rad5', v6_source + b'#', big, text, digest)
    with pytest.raises(ValueError, match='Manifest text differs'):
        v7.render_code('coat5rad5', v6_source, big, text + ' ', digest)


def test_kernel_metadata():
    km = v7.kernel_metadata('coat5rad5', v6.DOCKER_IMAGE, ['taeyangg4/knee-release-assets-v8',
                                                           'taeyangg4/knee-release-assets-v9'])
    assert km['id'] == 'taeyangg4/knee-exp119-coat5rad5-v7' and km['title'] == 'knee-exp119-coat5rad5-v7'
    assert km['dataset_sources'] == [v6.OLD_DATASET, 'taeyangg4/knee-release-assets-v8',
                                     'taeyangg4/knee-release-assets-v9', v6.OPENCV_DATASET]
    assert km['docker_image'] == v5.DOCKER_IMAGE == v6.DOCKER_IMAGE and km['is_private'] and not km['enable_internet']
    for bad in [('coat5rad5', 'gcr.io/kaggle-gpu-images/python:latest', ['o/slugs']),
                ('coat5rad5', v6.DOCKER_IMAGE, []), ('coat5rad5', v6.DOCKER_IMAGE, ['o/slugs', 'o/slugs']),
                ('x' * 36, v6.DOCKER_IMAGE, ['o/slugs']), ('Bad', v6.DOCKER_IMAGE, ['o/slugs'])]:
        with pytest.raises(ValueError):
            v7.kernel_metadata(*bad)


def ck_payload(torch, rad=False, **over):
    protocol = json.loads((v6.OLD_LOCAL / v6.OLD_PROTOCOL).read_text(encoding='utf-8'))
    code = {f: protocol['files']['training/src/' + f] for f in v6.TRAINING_CODE_BOUND}
    provenance = {'code_sha256': code}
    if rad:
        code['raptor_data.py'] = v6.CODE_FILES['raptor_data.py']
        provenance['radimagenet_module_sha256'] = v6.CODE_FILES['raptor_radimagenet.py']
    m = rad_meta(**over) if rad else meta(**over)
    return {'model': {'w': torch.randn(3)}, **m, 'provenance': provenance}


def save_ck(torch, path, rad=False, **over):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck_payload(torch, rad, **over), path)
    return path


def frozen_assets_or_skip():
    for p in (v6.OLD_LOCAL / v6.OLD_PROTOCOL, v6.ACCEPTANCE, v6.DOCKER_PULL):
        if not Path(p).is_file():
            pytest.skip('frozen assets / acceptance / docker pin not present')


def test_stage_assets_refuses_published_and_notebook_over_two_datasets(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    frozen_assets_or_skip()
    old = v6.verify_old_assets(v6.OLD_LOCAL)
    monkeypatch.setattr(v6, 'verify_old_assets', lambda root, opencv=None: old)
    for d in v7.PUBLISHED_ASSET_DATASETS:
        with pytest.raises(ValueError, match='published release-assets dataset'):
            v7.stage_assets(SimpleNamespace(checkpoint=['x=y'], package=str(tmp_path / 'p'), dataset=d, copy=True))
    for pk in v7.PUBLISHED_ASSET_PACKAGES:
        with pytest.raises(ValueError, match='published release-assets package'):
            v7.stage_assets(SimpleNamespace(checkpoint=['x=y'], package=str(pk), dataset='taeyangg4/knee-assets-new',
                                            copy=True))
    assert v7.PUBLISHED_ASSET_DATASETS[-1] == 'taeyangg4/knee-release-assets-v10'
    ds_a, ds_b = 'taeyangg4/knee-assets-test-a', 'taeyangg4/knee-assets-test-b'
    specs_a = ['c%d=%s' % (k, save_ck(torch, tmp_path / 'src' / ('C%d' % k) / 'model_best.pt', tag='C%d' % k))
               for k in range(2)]
    specs_a += ['r%d=%s' % (k, save_ck(torch, tmp_path / 'src' / ('R%d' % k) / 'model_best.pt', rad=True,
                                       tag='R%d' % k)) for k in range(2)]
    v7.stage_assets(SimpleNamespace(checkpoint=specs_a, package=str(tmp_path / 'pa'), dataset=ds_a, copy=True))
    specs_b = ['f%d=%s' % (k, save_ck(torch, tmp_path / 'src' / ('F%d' % k) / 'model_best.pt', tag='F%d' % k))
               for k in range(2)]
    v5.stage_assets(SimpleNamespace(checkpoint=specs_b, package=str(tmp_path / 'pb'), dataset=ds_b, copy=True))
    am_b = json.loads((tmp_path / 'pb' / v6.NEW_MANIFEST).read_text(encoding='utf-8'))
    assert am_b['schema'] == 'knee-release-assets-v5' and 'code' not in am_b

    def args(arm, out, **over):
        x = {'release': 'syn', 'arm': arm, 'assets': [ds_a + '=' + str(tmp_path / 'pa'), ds_b + '=' + str(tmp_path / 'pb')],
             'out': str(tmp_path / out), 'model_seconds': [], 'expected_studies': 1300, 'docker_from': str(v6.DOCKER_PULL)}
        x.update(over)
        return SimpleNamespace(**x)
    a = args(['coat5=c0+c1@36:1', 'rad5=r0+r1@36:1', 'fl5=f0+f1@36:1'], 'k1')
    plan = v7.notebook(a)
    out = Path(a.out)
    p = v7.checked_manifest(out / 'release-manifest.json', v6.sha(out / 'release-manifest.json'))
    assert p['arm_assets'] == {'coat5': ds_a, 'rad5': ds_a, 'fl5': ds_b}
    assert [s['dataset'] for s in p['asset_sets']] == [ds_a, ds_b] and [s['input_code'] for s in p['asset_sets']] == [True, False]
    v7.verify_asset_sets(p, {ds_a: tmp_path / 'pa', ds_b: tmp_path / 'pb'}, p['arms'])
    km = json.loads((out / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert km['dataset_sources'] == [v6.OLD_DATASET, ds_a, ds_b, v6.OPENCV_DATASET]
    assert plan['push_argv'][-2:] == ['--accelerator', 'NvidiaTeslaT4'] and plan['push_argv'][1].endswith('kaggle_push.py')
    assert not any('--timeout' in x for x in plan['push_argv'] + plan['push_argv_plain_cli'])
    assert plan['screen_at_expected_hidden_N']['projected_seconds'] == pytest.approx(
        1.3 * (D + 2 * 0.35 + 2 * 0.25 + 2 * 0.35) * 1300 + 600, abs=1e-6)
    assert plan['first_output'] == 'exp119-syn/combine/result.json'
    assert [x['published_before_v7'] for x in plan['asset_datasets']] == [False, False]
    code = ''.join(json.loads((out / 'release.ipynb').read_text(encoding='utf-8'))['cells'][0]['source'])
    assert max(len(line) for line in code.split('\n')) <= v6.MAX_NOTEBOOK_LINE
    for arm, over, message in [(['coat5=c0+c1@36'], {'assets': [ds_a + '=' + str(tmp_path / 'pa'),
                                                                 ds_b + '=' + str(tmp_path / 'pb')]}, 'binding differs'),
                               (['coat5=c0+f0@36'], {}, 'one assets dataset'),
                               (['coat5=c0+c1@36', 'rad5=r0+r1@36'],
                                {'assets': [ds_a + '=' + str(tmp_path / 'pa')], 'expected_studies': 3100}, 'time screen'),
                               (['coat5=c0+c1@36'], {'assets': [ds_a + '=' + str(tmp_path / 'pb')]},
                                'schema/dataset differs')]:
        b = args(arm, 'k-%d' % len(list(tmp_path.glob('k-*'))), **over)
        with pytest.raises(ValueError, match=message):
            v7.notebook(b)
        assert not Path(b.out).exists()


# ------------------------------------------------------------------ 8. generated kernels and the simulated notebook cell
EXPECTED = {
    'coat5rad5': [('coat5', ['coat_f%d' % k for k in range(5)], 'taeyangg4/knee-release-assets-v8', 'triplet'),
                  ('rad5', ['rad_f%d' % k for k in range(5)], 'taeyangg4/knee-release-assets-v8', 'center3')],
    'coat5fl5rl5': [('coat5', ['coat_f%d' % k for k in range(5)], 'taeyangg4/knee-release-assets-v8', 'triplet'),
                    ('fl5', ['fl_f%d' % k for k in range(5)], 'taeyangg4/knee-release-assets-v9', 'triplet'),
                    ('rl5', ['rl_f%d' % k for k in range(5)], 'taeyangg4/knee-release-assets-v7', 'triplet')],
    'coat5rad5fl5rl5': [('coat5', ['coat_f%d' % k for k in range(5)], 'taeyangg4/knee-release-assets-v8', 'triplet'),
                        ('rad5', ['rad_f%d' % k for k in range(5)], 'taeyangg4/knee-release-assets-v8', 'center3'),
                        ('fl5', ['fl_f%d' % k for k in range(5)], 'taeyangg4/knee-release-assets-v9', 'triplet'),
                        ('rl5', ['rl_f%d' % k for k in range(5)], 'taeyangg4/knee-release-assets-v7', 'triplet')]}


def kernel_dir(name):
    out = KERNELS / name
    if not (out / 'kernel-metadata.json').is_file():
        pytest.skip('kernel dir not generated')
    return out


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_generated_kernel_dir_is_current(name):
    out = kernel_dir(name)
    digest = v6.sha(out / 'release-manifest.json')
    p = v7.checked_manifest(out / 'release-manifest.json', digest)
    assert [(a['name'], [c['name'] for c in a['checkpoints']], p['arm_assets'][a['name']], a['input_policy'])
            for a in p['arms']] == EXPECTED[name]
    assert all(a['k_eval'] == 36 and a['weight'] == 1.0 for a in p['arms'])
    roots = {s['dataset']: Path(p['asset_packages'][s['dataset']]) for s in p['asset_sets']}
    if all((r / v6.NEW_MANIFEST).is_file() for r in roots.values()):
        assert all(v6.sha(r / v6.NEW_MANIFEST) == s['manifest_sha256'] for s, r in zip(p['asset_sets'], roots.values()))
    km = json.loads((out / 'kernel-metadata.json').read_text(encoding='utf-8'))
    km.pop('machine_shape', None)     # tools/kaggle_push.py strips it before a push (D-B-039)
    want = v7.kernel_metadata(name, v6.DOCKER_IMAGE, [s['dataset'] for s in p['asset_sets']])
    want.pop('machine_shape')
    assert km == want
    pulled = json.loads(v6.DOCKER_PULL.read_text(encoding='utf-8'))
    assert pulled['docker_image'] == p['docker_image'] == v6.DOCKER_IMAGE
    code = ''.join(json.loads((out / 'release.ipynb').read_text(encoding='utf-8'))['cells'][0]['source'])
    assert code == v7.render_code(name, Path(v6.__file__).read_bytes(), Path(v7.__file__).read_bytes(),
                                  (out / 'release-manifest.json').read_bytes().decode('utf-8'), digest)
    assert max(len(line) for line in code.split('\n')) <= v6.MAX_NOTEBOOK_LINE and '--timeout' not in code
    plan = json.loads((out / 'execution-plan.json').read_text(encoding='utf-8'))
    assert plan['manifest_sha256'] == digest and plan['notebook_sha256'] == v6.sha(out / 'release.ipynb')
    assert plan['dataset_upload'] is False and plan['submission'] is False and plan['status'] == 'CPU_PREPARED_NOT_PUSHED'
    assert all(x['published_before_v7'] for x in plan['asset_datasets'])
    assert plan['screen_at_expected_hidden_N']['projected_seconds'] == pytest.approx(
        v7.shared_seconds(1300, p['arms']) + 600, abs=1e-6)
    assert plan['max_studies_passing_screen'] == v7.max_studies(p['arms'])


def fake_kaggle(root, n, legacy_mount, datasets):
    k = Path(root) / 'kaggle'
    (k / 'working').mkdir(parents=True)
    inp = k / 'input'
    old = inp / 'knee-exp089-recipe-transfer-v1' if legacy_mount else inp / 'datasets/taeyangg4/knee-exp089-recipe-transfer-v1'
    cv = inp / 'datasets/mattiaangeli/opencv-python-headless-4120088-x86'
    comp = inp / 'rsna-knee-abnormality-detection' if legacy_mount else inp / 'competitions/rsna-knee-abnormality-detection'
    dirs = [(old, 'exp089-t4-protocol.json'), (cv, 'opencv_python_headless-4.12.0.88-cp37-abi3-manylinux_2_17_x86_64.whl')]
    for d in datasets:
        o, s = d.split('/')
        dirs.append((inp / s if legacy_mount else inp / 'datasets' / o / s, 'assets-manifest.json'))
    for d, f in dirs:
        d.mkdir(parents=True)
        (d / f).write_text('{}', encoding='utf-8')
    ids = ['1.2.840.%d' % (9000 + 7 * i) for i in range(n)]
    fixture_comp(comp, ids)
    return k, ids


def simulate_cell(monkeypatch, tmp_path, name, n, legacy_mount=False):
    out = kernel_dir(name)
    code = ''.join(json.loads((out / 'release.ipynb').read_text(encoding='utf-8'))['cells'][0]['source'])
    p0 = json.loads((out / 'release-manifest.json').read_text(encoding='utf-8'))
    k, ids = fake_kaggle(tmp_path / 'fake', n, legacy_mount, [s['dataset'] for s in p0['asset_sets']])
    prefix = (tmp_path / 'fake').as_posix()
    code = code.replace("'/kaggle/", "'" + prefix + '/kaggle/')
    real_run = subprocess.run
    calls, tables, tools = [], {}, {}

    def stub(argv, check=False, **kw):
        mode = argv[2]
        if mode == 'combine':
            calls.append(('combine',))
            return real_run(argv, check=check, **kw)          # the real embedded tool (imports the embedded v6), real CLI
        assert mode == 'run' and check and '--timeout' not in argv
        roots = [argv[i + 1] for i, x in enumerate(argv) if x == '--asset-root']
        opts = dict(zip(argv[3::2], argv[4::2]))
        if 'm' not in tools:
            tools['m'] = load_path('embedded_v7_%s_%d' % (name, n), argv[1])
        tool = tools['m']
        assert Path(tool.v6.__file__).parent == Path(argv[1]).parent
        p = tool.checked_manifest(opts['--manifest'], opts['--sha256'])
        assert [r.split('=', 1)[0] for r in roots] == [s['dataset'] for s in p['asset_sets']]
        contract = tool.v6.test_contract(opts['--competition'])
        plan = tool.operational_plan(contract['studies'], p['arms'])
        tool.v6.launch_budget(float(opts['--started']), time.monotonic(), contract['studies'],
                              tool.launch_projection(plan), float(opts['--bootstrap-max-seconds']))
        calls.append(('run', opts['--bootstrap-max-seconds'], len(roots)))
        base = Path(opts['--base'])
        work = base / 'shared/work'
        work.mkdir(parents=True)
        dest = tool.arm_destinations(p, base, opts['--final'])
        arms = {}
        for arm in p['arms']:
            rng = random.Random(arm['name'])
            probs = [[rng.random() for _ in range(12)] for _ in contract['ids']]
            tables[arm['name']] = probs
            dest[arm['name']].parent.mkdir(parents=True, exist_ok=True)
            tool.v6.write_columns(dest[arm['name']], contract['ids'], [[row[t] for row in probs] for t in range(12)])
            arms[arm['name']] = {'submission_sha256': tool.sha(dest[arm['name']]), 'path': str(dest[arm['name']]),
                                 'n_models': len(arm['checkpoints']), 'k_eval': arm['k_eval'],
                                 'rank_weight': arm['weight'], 'model_seconds': 0.0}
        result = {'status': 'PASS_TEST_COHORT_ONLY', 'schema': tool.SCHEMA, 'release_manifest_sha256': opts['--sha256'],
                  'studies': contract['studies'], 'arm_order': [a['name'] for a in p['arms']], 'arms': arms,
                  'decode': {'decodes': contract['studies']}}
        if len(p['arms']) == 1:
            result['submission_sha256'] = arms[p['arms'][0]['name']]['submission_sha256']
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
    return k, ids, calls, tables, p0


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', sorted(EXPECTED))
@pytest.mark.parametrize('n,legacy', [(3, False), (4, True), (1300, False)])
def test_simulated_notebook_cell(monkeypatch, tmp_path, capsys, name, n, legacy):
    k, ids, calls, tables, p0 = simulate_cell(monkeypatch, tmp_path, name, n, legacy)
    final = k / 'working/submission.csv'
    base = k / 'working' / ('exp119-' + name)
    assert (base / 'exp111_release_v6.py').read_bytes() == Path(v6.__file__).read_bytes()
    assert (base / 'exp119_release_v7.py').read_bytes() == Path(v7.__file__).read_bytes()
    v6.csv_check(final, ids)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['studies'] == n and printed['submission_sha256'] == v6.sha(final)
    names = [a['name'] for a in p0['arms']]
    assert calls == [('run', '600', len(p0['asset_sets'])), ('combine',)]     # ONE worker launch for every arm
    assert printed['status'] == 'PASS_RANK_MEAN_TEST_COHORT_ONLY'
    v6.write_columns(base / 'expected.csv', ids, v6.rank_mean([tables[x] for x in names]))
    assert final.read_bytes() == (base / 'expected.csv').read_bytes()
    assert not list((k / 'working').rglob('*.exp111-tmp'))


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_simulated_notebook_cell_fails_closed_above_budget(monkeypatch, tmp_path, name):
    p0 = json.loads((kernel_dir(name) / 'release-manifest.json').read_text(encoding='utf-8'))
    with pytest.raises(AssertionError, match='v7 time screen failed'):
        simulate_cell(monkeypatch, tmp_path, name, v7.max_studies(p0['arms']) + 1)
    k = tmp_path / 'fake/kaggle'
    assert not (k / 'working/submission.csv').exists()
    assert not (k / 'working' / ('exp119-' + name) / 'shared').exists()


# ------------------------------------------------------------------ 9. real-checkpoint CPU parity receipt (if produced)
def test_cpu_parity_receipt():
    if not PARITY.is_file():
        pytest.skip('run .local/exp119-release-v7/cpu_parity.py first')
    r = json.loads(PARITY.read_text(encoding='utf-8'))
    assert r['tool_sha256'] == v6.sha(v7.__file__), 'parity receipt is stale: rerun cpu_parity.py'
    assert r['v6_sha256'] == v7.V6_SOURCE_SHA256 and r['device'] == 'cpu' and r['cuda_initialized'] is False
    t = r['train_studies']
    assert len(t['ids']) >= 20
    assert t['decodes'] == {'v5_coat5_separate': len(t['ids']), 'v6_rad5_separate': len(t['ids']),
                            'v7_shared_coat5_rad5': len(t['ids'])}
    for arm in ('coat5', 'rad5'):
        a = t['arms'][arm]
        assert a['max_abs_diff'] == 0.0 and a['mean_abs_diff'] == 0.0 and a['bit_identical'] is True
        assert a['ordered_inputs_sha256_equal'] is True and a['arm_csv_bytes_equal'] is True
        assert a['max_abs_diff'] <= r['tolerance']
    v = t['volumes']
    assert v['shared_equals_v5_separate'] == v['shared_equals_v6_separate'] == len(t['ids'])
    assert v['masks_equal'] == len(t['ids']) and v['receipt_hashes_equal'] == len(t['ids'])
    assert v['shared_equals_training_cache'] == len(t['ids'])
    assert t['rank_mean']['v7_combine_equals_v6_combine'] is True
    pub = r.get('public_test')
    if pub:
        assert pub['studies'] == 3 and pub['decodes'] == 3
        for arm in ('coat5', 'rad5'):
            assert pub['arms'][arm]['max_abs_diff_vs_kaggle_t4_fp16'] <= r['tolerance']
