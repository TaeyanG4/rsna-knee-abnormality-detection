"""CPU tests for the EXP104 v5 release (v4 + configurable assets dataset + per-arm rank weights).

No GPU, no network, no real study data. The opt-in smoke (EXP104_REAL_SMOKE=1) loads two real staged checkpoints on CPU.
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


release = load('exp104_release_v5_under_test', 'tools/exp104_release_v5.py')
v4 = load('exp100_release_v4_for_v5', 'tools/exp100_release_v4.py')
v31 = load('recipe_release_v31_for_v5', 'tools/exp089_recipe_release_v31.py')
exp098 = load('exp098_e1_release_for_v5', 'tools/exp098_e1_release.py')
SUPERVISOR = ROOT / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/t4-results-v2/B224-supervisor.json'
ARCH = 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'
RN_ARCH = 'resnet34.a1_in1k'
CONTRACT = 'raptor64_336_c140_s02'
KERNELS = ROOT / '.local/exp104-release-v5'
D = 6.52503319258


# ------------------------------------------------------------------ fixtures
def meta(**over):
    m = {'arch': ARCH, 'res': 224, 'lab': list(release.TARGETS), 'contract': CONTRACT, 'k_eval': 24,
         'norm': 'imagenet', 'window_policy': 'within-slot-valid', 'epoch': 7, 'tag': 'T', 'amp_train': 'fp16',
         'amp_eval': 'fp16'}
    m.update(over)
    return m


def fake_assets(root, members, dataset=None):
    """members: {name: (bytes, meta)} -> new-assets dir with a v5 assets manifest."""
    root = Path(root)
    (root / 'checkpoints').mkdir(parents=True)
    checkpoints = {}
    for name, (blob, m) in members.items():
        path = root / 'checkpoints' / (name + '.pt')
        path.write_bytes(blob)
        checkpoints[name] = {'path': 'checkpoints/' + name + '.pt', 'sha256': release.sha(path), 'bytes': len(blob),
                             'meta': m, 'training_code_sha256': {}, 'source': 'synthetic', 'source_sidecars_sha256': {},
                             'link_mode': 'copy'}
    am = {'schema': release.ASSETS_SCHEMA, 'dataset': dataset or release.NEW_DATASET, 'checkpoints': checkpoints,
          'files': {c['path']: c['sha256'] for c in checkpoints.values()}, 'dataset_metadata_sha256': '0' * 64}
    (root / release.NEW_MANIFEST).write_text(json.dumps(am), encoding='utf-8')
    return am


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


# ------------------------------------------------------------------ 1. ensemble wrapper (v4 semantics + fail-closed)
def test_mean_prob_wrapper_is_mean_of_probabilities_not_logits():
    torch = pytest.importorskip('torch')
    x = torch.linspace(-6, 6, 24).reshape(2, 12)
    models = [affine(torch, 1.0, 0.0), affine(torch, -0.5, 1.0), affine(torch, 2.0, -3.0)]
    ens = release.mean_prob_module(models).eval()
    with torch.inference_mode():
        got = torch.sigmoid(ens(x))
        want = torch.stack([torch.sigmoid(m(x)) for m in models]).mean(0)
        logit_mean = torch.sigmoid(torch.stack([m(x) for m in models]).mean(0))
        assert torch.equal(ens(x), v4.mean_prob_module(models).eval()(x))     # finite inputs: v4 output unchanged
    assert (got - want).abs().max().item() <= 1e-6
    assert (got - logit_mean).abs().max().item() > 1e-2


def test_single_model_identity_and_clamp_of_finite_extremes():
    torch = pytest.importorskip('torch')
    ens = release.mean_prob_module([affine(torch, 1.0, 0.0)]).eval()
    z = torch.linspace(-10, 10, 240).reshape(20, 12)
    with torch.inference_mode():
        assert (torch.sigmoid(ens(z)) - torch.sigmoid(z)).abs().max().item() <= 1e-6
        out = ens(torch.tensor([[-80.0] * 6 + [80.0] * 6]))
        p = torch.sigmoid(out)
    assert torch.isfinite(out).all()
    assert torch.allclose(p[0, :6], torch.full((6,), 1e-6), rtol=0, atol=1e-9)
    assert torch.allclose(p[0, 6:], torch.full((6,), 1 - 1e-6), rtol=0, atol=1e-7)


@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
def test_nonfinite_member_logit_fails_closed(bad):
    torch = pytest.importorskip('torch')

    class Bad(torch.nn.Module):
        def forward(self, x):
            out = x.clone()
            out[0, 3] = bad
            return out
    x = torch.zeros(2, 12)
    ens = release.mean_prob_module([affine(torch, 1.0, 0.0), Bad()]).eval()
    with torch.inference_mode(), pytest.raises(RuntimeError, match='nonfinite member logit .member 1.'):
        ens(x)
    if bad != bad:   # nan: v4 already failed later through predict; inf: v4 clamped it silently
        return
    with torch.inference_mode():
        assert torch.isfinite(v4.mean_prob_module([affine(torch, 1.0, 0.0), Bad()]).eval()(x)).all()


def test_wrapper_aggregates_in_fp32_under_autocast():
    torch = pytest.importorskip('torch')
    torch.manual_seed(0)
    members = [torch.nn.Linear(12, 12), torch.nn.Linear(12, 12)]
    ens = release.mean_prob_module(members).eval()
    x = torch.linspace(-4, 4, 24).reshape(2, 12)
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
        low = [m(x) for m in members]
        out = ens(x)
    assert low[0].dtype == torch.bfloat16
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    want = torch.stack([torch.sigmoid(z.float()) for z in low]).mean(0)
    assert (torch.sigmoid(out) - want).abs().max().item() <= 1e-6


def frozen_runtime(monkeypatch):
    """The serving copy of raptor_runtime from the frozen EXP089 assets."""
    src = release.OLD_LOCAL / 'training/src'
    if not (src / 'raptor_runtime.py').is_file():
        pytest.skip('frozen EXP089 assets not present')
    protocol = json.loads((release.OLD_LOCAL / release.OLD_PROTOCOL).read_text(encoding='utf-8'))
    assert release.sha(src / 'raptor_runtime.py') == protocol['files']['training/src/raptor_runtime.py']
    if 'raptor_common' not in sys.modules:
        monkeypatch.setenv('KNEE_WORKSPACE_ROOT', str(release.OLD_LOCAL))
        monkeypatch.syspath_prepend(str(src))
    common = __import__('raptor_common')
    assert release.sha(common.__file__) == protocol['files']['training/src/raptor_common.py']
    return load('frozen_raptor_runtime_for_v5', str((src / 'raptor_runtime.py').relative_to(ROOT)))


class FakeCorpus:
    """DicomCorpus interface (depth/spec/mask/volume) keeping one study, counting decodes like the real corpus."""

    def __init__(self, ids, res=8):
        from raptor_common import SLOTS64
        self.spec, self.depth = {'slots': SLOTS64}, 64
        rng = np.random.default_rng(0)
        self.data = {u: rng.integers(0, 256, size=(64, res, res), dtype=np.uint8) for u in ids}
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
        return np.ones(self.depth)


def test_wrapper_through_unchanged_predict_decodes_each_study_once_and_fails_closed(monkeypatch):
    torch = pytest.importorskip('torch')
    rt = frozen_runtime(monkeypatch)
    ids = ['1.1', '1.2', '1.3']
    members = [tiny(torch, s) for s in (1, 2, 3)]

    def go(model):
        corpus = FakeCorpus(ids)
        probs, timing = rt.predict(model, corpus, ids, 4, 8, 'none', 'cpu', norm='imagenet', batch=1,
                                   log=lambda *a: None, window_policy='within-slot-valid')
        return probs, timing, corpus.decodes

    singles = [go(m) for m in members]
    pe, te, decodes = go(release.mean_prob_module(members))
    assert decodes == len(ids)
    assert pe.shape == (3, 12) and pe.dtype == np.float32
    assert np.abs(pe - np.mean([s[0] for s in singles], axis=0)).max() <= 1e-6
    assert te['ordered_inputs_sha256'] == singles[0][1]['ordered_inputs_sha256']

    class Inf(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 12), float('inf'))
    with pytest.raises(RuntimeError, match='nonfinite member logit'):
        go(release.mean_prob_module([members[0], Inf()]))


# ------------------------------------------------------------------ 2. manifests, assets, dataset parameter
def two_member_assets(tmp_path, b_meta=None, dataset=None):
    return fake_assets(tmp_path / 'new', {'a': (b'checkpoint-a' * 100, meta()),
                                          'b': (b'checkpoint-b' * 100, b_meta or meta(tag='U'))}, dataset)


def test_assets_binding_checks_dataset_sha_and_presence(tmp_path):
    am = two_member_assets(tmp_path)
    root = tmp_path / 'new'
    arm = release.build_arm('ens', ['a', 'b'], None, am, 0.35)
    rel = {'new_assets_manifest_sha256': release.sha(root / release.NEW_MANIFEST), 'new_asset_dataset': release.NEW_DATASET}
    release.verify_new_assets(rel, root, [arm])
    with pytest.raises(ValueError, match='schema differs'):
        release.verify_new_assets({**rel, 'new_asset_dataset': 'taeyangg4/knee-release-assets-v6'}, root, [arm])
    for bad in (release.V4_DATASET, release.OLD_DATASET, 'Bad/Slug', None):
        with pytest.raises(ValueError, match='Unsafe new-assets dataset'):
            release.verify_new_assets({**rel, 'new_asset_dataset': bad}, root, [arm])
    (root / 'checkpoints/b.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='sha256 differs'):
        release.verify_new_assets(rel, root, [arm])
    (root / 'checkpoints/b.pt').unlink()
    with pytest.raises(ValueError, match='Missing checkpoint'):
        release.verify_new_assets(rel, root, [arm])
    with pytest.raises(ValueError, match='assets manifest changed'):
        release.verify_new_assets({**rel, 'new_assets_manifest_sha256': '0' * 64}, root, [arm])


def test_custom_dataset_slug_round_trip(tmp_path):
    am = two_member_assets(tmp_path, dataset='taeyangg4/knee-release-assets-v6')
    root = tmp_path / 'new'
    arm = release.build_arm('ens', ['a', 'b'], None, am, 0.35)
    rel = {'new_assets_manifest_sha256': release.sha(root / release.NEW_MANIFEST),
           'new_asset_dataset': 'taeyangg4/knee-release-assets-v6'}
    release.verify_new_assets(rel, root, [arm])
    with pytest.raises(ValueError, match='schema differs'):
        release.verify_new_assets({**rel, 'new_asset_dataset': release.NEW_DATASET}, root, [arm])


def test_release_manifest_validation(tmp_path):
    am = two_member_assets(tmp_path)
    arms = [release.build_arm('ens', ['a', 'b'], None, am, 0.35, 0.7), release.build_arm('a', ['a'], 36, am, 0.15, 0.3)]
    path = tmp_path / 'release-manifest.json'
    digest = release_manifest(path, arms, '1' * 64)
    p = release.checked_manifest(path, digest)
    assert [a['name'] for a in p['arms']] == ['ens', 'a'] and p['arms'][1]['k_eval'] == 36
    assert p['rank_weights'] == {'ens': 0.7, 'a': 0.3}
    with pytest.raises(ValueError, match='Release manifest changed'):
        release.checked_manifest(path, '0' * 64)
    bad_cases = [
        ({'time_policy': {**release.TIME, 'inflation': 1.0}}, 'policy'),
        ({'release_source_sha256': '0' * 64}, 'source changed'),
        ({'arms': []}, 'At least one arm'),
        ({'arms': [{**arms[0], 'aggregator': 'mean_logit'}]}, 'mean_prob'),
        ({'arms': [{**arms[0], 'checkpoints': []}]}, 'checkpoints malformed'),
        ({'arms': [arms[0], arms[0]]}, 'Duplicate arm'),
        ({'arms': [{**arms[0], 'checkpoints': [arms[0]['checkpoints'][0]] * 2}]}, 'Duplicate checkpoint'),
        ({'arms': [{**arms[0], 'k_eval': 24.0}]}, 'positive integers'),
        ({'arms': [{**arms[0], 'model_seconds_per_study': 0}]}, 'model_seconds'),
        ({'arms': [{**arms[0], 'res': 384}]}, 'differs from arm'),
        ({'arms': [{**arms[0], 'checkpoints': [{**arms[0]['checkpoints'][0], 'path': '../x.pt'}]}]}, 'Unsafe'),
        ({'arms': [{**arms[0], 'weight': 0.0}, arms[1]]}, 'Rank weight'),
        ({'arms': [{**arms[0], 'weight': -1.0}, arms[1]]}, 'Rank weight'),
        ({'arms': [{**arms[0], 'weight': float('nan')}, arms[1]]}, 'Rank weight'),
        ({'arms': [{**arms[0], 'weight': True}, arms[1]]}, 'Rank weight'),
        ({'arms': [{k: v for k, v in arms[0].items() if k != 'weight'}, arms[1]]}, 'Arm fields differ'),
        ({'rank_weights': {'ens': 0.5, 'a': 0.5}}, 'Recorded rank weights'),
        ({'rank_weights': None}, 'Recorded rank weights'),
        ({'new_asset_dataset': release.V4_DATASET}, 'Unsafe new-assets dataset'),
        ({'new_asset_dataset': None}, 'Unsafe new-assets dataset'),
    ]
    for i, (over, message) in enumerate(bad_cases):
        bad = tmp_path / ('bad-%d.json' % i)
        d = release_manifest(bad, arms, '1' * 64, **over)
        with pytest.raises(ValueError, match=message):
            release.checked_manifest(bad, d)


def test_build_arm_rules(tmp_path):
    am = two_member_assets(tmp_path)
    assert release.build_arm('x', ['a', 'b'], None, am, 0.35)['weight'] == 1.0
    with pytest.raises(ValueError, match='Unknown staged checkpoint'):
        release.build_arm('x', ['a', 'missing'], None, am, 0.35)
    mixed = fake_assets(tmp_path / 'mixed', {'a': (b'a', meta()), 'r': (b'r', meta(arch=RN_ARCH, epoch=19))})
    with pytest.raises(ValueError, match='differ in arch'):
        release.build_arm('x', ['a', 'r'], None, mixed, 0.35)
    keval = fake_assets(tmp_path / 'keval', {'a': (b'a', meta()), 'b': (b'b', meta(k_eval=36))})
    with pytest.raises(ValueError, match='different k_eval'):
        release.build_arm('x', ['a', 'b'], None, keval, 0.35)
    assert release.build_arm('x', ['a', 'b'], 24, keval, 0.35)['k_eval'] == 24
    with pytest.raises(ValueError, match='Rank weight'):
        release.build_arm('x', ['a'], None, am, 0.35, 1)        # int, not float: refused


def test_parse_arm_spec_with_weights():
    assert release.parse_arm_spec('student') == ('student', ['student'], None, None)
    assert release.parse_arm_spec('student@36') == ('student', ['student'], 36, None)
    assert release.parse_arm_spec('ens=student+control@24') == ('ens', ['student', 'control'], 24, None)
    assert release.parse_arm_spec('coat5=coat_f0+coat_f1@36:0.7') == ('coat5', ['coat_f0', 'coat_f1'], 36, 0.7)
    assert release.parse_arm_spec('rn5=rn_f0+rn_f1:.3') == ('rn5', ['rn_f0', 'rn_f1'], None, 0.3)
    assert release.parse_arm_spec('a:2') == ('a', ['a'], None, 2.0)
    for bad in ['', 'a=b+b', 'a=', '../x', 'a@0', 'a=b+', 'a b', 'a:0', 'a:0.0', 'a:-1', 'a:1e3', 'a:nan',
                'a:inf', 'a:', 'a@36:', 'a:0.7@36', 'a:1.2.3']:
        with pytest.raises(ValueError):
            release.parse_arm_spec(bad)


def test_checkpoint_metadata_check_records_both_k_eval(tmp_path):
    am = two_member_assets(tmp_path)
    arm = release.build_arm('a36', ['a'], 36, am, 0.35)
    ck = arm['checkpoints'][0]
    r = release.checkpoint_check({**meta(), 'provenance': {'ignored': True}, 'lab': tuple(release.TARGETS)}, arm, ck)
    assert r['k_eval_checkpoint'] == 24 and r['k_eval_release'] == 36 and r['sha256'] == ck['sha256']
    for key, value, message in [('arch', 'other', 'differs from arm'), ('window_policy', 'legacy', 'differs from arm'),
                                ('norm', 'none', 'differs from arm'), ('lab', release.TARGETS[::-1], 'differs from arm'),
                                ('epoch', 3, 'staged record'), ('k_eval', 62, 'staged record')]:
        with pytest.raises(ValueError, match=message):
            release.checkpoint_check({**meta(), key: value}, arm, ck)


# ------------------------------------------------------------------ 3. time screen per arm
def arms_of(*counts, m=0.35):
    ms = m if isinstance(m, (list, tuple)) else [m] * len(counts)
    return [{'name': 'a%d' % i, 'checkpoints': [None] * c, 'model_seconds_per_study': s}
            for i, (c, s) in enumerate(zip(counts, ms))]


def test_time_screen_arithmetic_unchanged_from_v4():
    for arms in (arms_of(1), arms_of(1, 2), arms_of(1, 1, m=0.0), arms_of(4, m=0.1), arms_of(3, 2)):
        for n in (3, 1300):
            assert release.operational_plan(n, arms) == v4.operational_plan(n, arms)
        assert release.max_studies(arms) == v4.max_studies(arms)
    assert release.TIME == v4.TIME


def test_time_screen_for_the_two_releases():
    coat5 = arms_of(5, m=0.35)
    coat5rn5 = arms_of(5, 5, m=[0.35, 0.15])
    a = release.operational_plan(1300, coat5)
    b = release.operational_plan(1300, coat5rn5)
    assert a['projected_seconds'] == pytest.approx(1.3 * (D + 5 * 0.35) * 1300 + 600, abs=1e-9)   # 14,584.8 s
    assert b['arm_projected_seconds']['a0'] == pytest.approx(1.3 * (D + 1.75) * 1300, abs=1e-9)    # 13,984.8 s
    assert b['arm_projected_seconds']['a1'] == pytest.approx(1.3 * (D + 0.75) * 1300, abs=1e-9)    # 12,294.8 s
    assert b['projected_seconds'] == pytest.approx(26879.6121909204, abs=1e-6) and b['projected_seconds'] <= 30600
    assert release.max_studies(coat5) == 2788 and release.max_studies(coat5rn5) == 1484
    for arms, n in ((coat5, 2788), (coat5rn5, 1484)):
        release.operational_plan(n, arms)
        with pytest.raises(ValueError, match='time screen'):
            release.operational_plan(n + 1, arms)
    for n in range(1, 1401, 7):
        release.operational_plan(n, coat5rn5)
    # two arms decoding at the measured 6.525 s/study plus all ten models, without inflation: 5.6 h at 1,300
    raw = (2 * D + 5 * 0.35 + 5 * 0.15) * 1300
    assert raw == pytest.approx(20215.09, abs=0.01) and raw < 32400


def test_model_seconds_by_architecture():
    assert release.arm_model_seconds('coat5', ARCH, {})[0] == 0.35
    assert release.arm_model_seconds('rn5', RN_ARCH, {})[0] == 0.15
    assert release.arm_model_seconds('rn5', RN_ARCH, {'rn5': 0.2}) == (0.2, 'override --model-seconds rn5=0.2')
    with pytest.raises(ValueError, match='No model_seconds_per_study default'):
        release.arm_model_seconds('x', 'convnext_small', {})
    assert release.parse_model_seconds(['rn5=0.2']) == {'rn5': 0.2}
    for bad in (['rn5'], ['rn5=0'], ['rn5=-1'], ['rn5=nan'], ['rn5=0.1', 'rn5=0.2'], ['../x=0.1']):
        with pytest.raises(ValueError):
            release.parse_model_seconds(bad)


def test_launch_projection_covers_remaining_arms():
    arms = arms_of(5, 5, m=[0.35, 0.15])
    plan = release.operational_plan(1300, arms)
    first = release.launch_projection(plan, arms, 'a0')
    last = release.launch_projection(plan, arms, 'a1')
    assert first == pytest.approx(plan['projected_seconds'] - 600 + 300)
    assert last == pytest.approx(plan['arm_projected_seconds']['a1'] + 300)
    # the second arm must launch by first-cell clock + 32100 - 30 - last (about 5.4 h): the first arm's unscaled
    # projection is 1300 * (6.525 + 1.75) s = 3.0 h
    release.launch_budget(1000.0, 1000.0 + 1300 * (D + 1.75) * 1.5, 1300, last, 30000.0)
    with pytest.raises(ValueError, match='projection'):
        release.launch_budget(1000.0, 1000.0 + 32100 - 30 - last + 1, 1300, last, 30000.0)


@pytest.mark.parametrize('started,now,n,boot', [
    (1000.0, 10000.0, 1300, 30000.0), (1000.0, 10000.0, 1300, 600.0), (1000.0, 26000.0, 1300, 30000.0),
    (5.0, 65.0, 3, 600.0), (5.0, 705.0, 3, 30000.0), (1000.0, 1005.0, 3, 30000.0), (1000.0, 1060.0, 1300, 30000.0),
    (1000.0, 13000.0, 1760, 30000.0), (1000.0, 999.0, 1300, 600.0), (float('nan'), 1000.0, 1300, 600.0),
    (1000.0, 1300.0, 1300, float('inf')), (1000.0, 1000.0, 1300, 0.0)])
def test_launch_budget_matches_v31(started, now, n, boot):
    projected = v31.operational_plan(n)['projected_seconds']
    try:
        _, deadline, timeout = v31.launch_budget(started, now, n, boot)
    except ValueError as exc:
        with pytest.raises(ValueError, match=str(exc).split()[0]):
            release.launch_budget(started, now, n, projected, boot)
        return
    assert release.launch_budget(started, now, n, projected, boot) == (deadline, timeout)


# ------------------------------------------------------------------ 4. rank combination
def exp098_rank():
    code = exp098.CODE
    ns = {}
    exec(code[code.index('def rank(col):'):code.index('cols=[]')], ns)
    return ns['rank']


def synthetic_arms(n=257, seed=7):
    rng = random.Random(seed)
    pB = [[rng.choice([0.1, 0.25, 0.5]) if t % 3 == 0 else rng.random() for t in range(12)] for _ in range(n)]
    pR = [[round(rng.random(), 2) for _ in range(12)] for _ in range(n)]
    pC = [[rng.random() ** 3 for _ in range(12)] for _ in range(n)]
    return pB, pR, pC


def test_equal_weights_reproduce_v4_and_exp098_bit_for_bit(tmp_path):
    rank = exp098_rank()
    pB, pR, pC = synthetic_arms()
    cols98 = []
    for t in range(12):
        rb = rank([r[t] for r in pB]); rr = rank([r[t] for r in pR]); cols98.append([(a + b) / 2 for a, b in zip(rb, rr)])
    for weights in (None, [1.0, 1.0], [0.5, 0.5], [0.3, 0.3], [7.0, 7.0]):
        assert release.rank_mean([pB, pR], weights) == cols98 == v4.rank_mean([pB, pR])
    for weights in (None, [1.0] * 3, [0.2] * 3, [1 / 3] * 3):
        assert release.rank_mean([pB, pR, pC], weights) == v4.rank_mean([pB, pR, pC])
    ids = ['1.%d' % i for i in range(len(pB))]
    release.write_columns(tmp_path / 'v5.csv', ids, release.rank_mean([pB, pR], [0.5, 0.5]))
    v4.write_columns(tmp_path / 'v4.csv', ids, v4.rank_mean([pB, pR]))
    assert (tmp_path / 'v5.csv').read_bytes() == (tmp_path / 'v4.csv').read_bytes()


def test_weighted_rank_mean_formula_and_range():
    pB, pR, pC = synthetic_arms()
    got = release.rank_mean([pB, pR], [0.7, 0.3])
    for t in range(12):
        rb = release.average_rank([r[t] for r in pB]); rr = release.average_rank([r[t] for r in pR])
        assert got[t] == [(0.7 * a + 0.3 * b) / (0.7 + 0.3) for a, b in zip(rb, rr)]
        assert max(abs(g - (0.7 * a + 0.3 * b)) for g, a, b in zip(got[t], rb, rr)) <= 1e-15
    assert got != v4.rank_mean([pB, pR])
    three = release.rank_mean([pB, pR, pC], [2.0, 1.0, 1.0])
    rb, rr, rc = (release.average_rank([r[5] for r in x]) for x in (pB, pR, pC))
    assert three[5] == [(2.0 * a + 1.0 * b + 1.0 * c) / 4.0 for a, b, c in zip(rb, rr, rc)]
    # ranks of 1.0 stay exactly 1.0 for awkward weights (monotone rounding), so the range check never trips
    top = [[1.0 * (i + 1) for _ in range(12)] for i in range(50)]
    for weights in ([0.7, 0.3], [0.1, 0.2], [0.3, 0.3000001], [1e-8, 3.0], [9999.0, 0.00000001]):
        cols = release.rank_mean([top, top], weights)
        assert max(max(c) for c in cols) == 1.0 and min(min(c) for c in cols) > 0
    for bad in ([0.7], [0.7, 0.0], [0.7, -0.3], [0.7, float('nan')], [0.7, float('inf')], [1, 1], [True, True]):
        with pytest.raises(ValueError, match='Invalid rank weights'):
            release.rank_mean([pB, pR], bad)
    with pytest.raises(ValueError):
        release.rank_mean([pB])


def test_rank_weights_normalised():
    assert release.rank_weights([{'name': 'a', 'weight': 0.7}, {'name': 'b', 'weight': 0.3}]) == {'a': 0.7, 'b': 0.3}
    assert release.rank_weights([{'name': 'a', 'weight': 1.0}, {'name': 'b', 'weight': 1.0}]) == {'a': 0.5, 'b': 0.5}
    assert release.rank_weights([{'name': 'a', 'weight': 2.0}, {'name': 'b', 'weight': 6.0}]) == {'a': 0.25, 'b': 0.75}
    with pytest.raises(ValueError):
        release.rank_weights([{'name': 'a', 'weight': 0.0}])


def stage_arm_results(base, arms, digest, ids, tables):
    for arm, probs in zip(arms, tables):
        d = base / arm['name']
        (d / 'work').mkdir(parents=True)
        write_csv(d / 'arm.csv', [release.UID, *release.TARGETS], [[u, *map(repr, p)] for u, p in zip(ids, probs)])
        r = {'status': 'PASS_TEST_COHORT_ONLY', 'arm': arm['name'], 'release_manifest_sha256': digest,
             'studies': len(ids), 'submission_sha256': release.sha(d / 'arm.csv'), 'elapsed_seconds': 1.0,
             'sampled_tree_rss_bytes': 1, 'n_models': len(arm['checkpoints']), 'k_eval': 24}
        if 'weight' in arm:
            r['rank_weight'] = arm['weight']
        (d / 'work/result.json').write_text(json.dumps(r), encoding='utf-8')


def test_combine_weighted_end_to_end_records_weights_and_fails_closed(tmp_path):
    ids = ['1.%d' % i for i in range(9)]
    comp = fixture_comp(tmp_path / 'comp', ids)
    am = two_member_assets(tmp_path)
    arms = [release.build_arm('ens', ['a', 'b'], None, am, 0.35, 0.7), release.build_arm('a', ['a'], None, am, 0.15, 0.3)]
    digest = release_manifest(tmp_path / 'm.json', arms, '1' * 64)
    pB, pR, _ = synthetic_arms(n=len(ids), seed=3)
    base = tmp_path / 'base'
    stage_arm_results(base, arms, digest, ids, [pB, pR])
    a = SimpleNamespace(manifest=str(tmp_path / 'm.json'), sha256=digest, base=str(base), competition=str(comp),
                        out=str(tmp_path / 'submission.csv'), started=time.monotonic())
    r = release.combine(a)
    assert r['status'] == 'PASS_RANK_MEAN_TEST_COHORT_ONLY' and r['arm_order'] == ['ens', 'a']
    assert r['arm_weights'] == {'ens': 0.7, 'a': 0.3} and r['rank_weights'] == {'ens': 0.7, 'a': 0.3}
    assert r['arms']['ens']['rank_weight'] == 0.7 and 'weighted' in r['formula']
    stored = json.loads((base / 'combine/result.json').read_text(encoding='utf-8'))
    assert stored['rank_weights'] == {'ens': 0.7, 'a': 0.3}
    release.write_columns(tmp_path / 'expected.csv', ids, release.rank_mean([pB, pR], [0.7, 0.3]))
    assert (tmp_path / 'submission.csv').read_bytes() == (tmp_path / 'expected.csv').read_bytes()
    assert not list(tmp_path.glob('*.exp104-tmp'))
    # an arm result whose recorded weight differs fails closed
    (tmp_path / 'submission.csv').unlink()
    shutil.rmtree(base / 'combine')
    rj = json.loads((base / 'a/work/result.json').read_text(encoding='utf-8'))
    (base / 'a/work/result.json').write_text(json.dumps({**rj, 'rank_weight': 0.5}), encoding='utf-8')
    with pytest.raises(ValueError, match='weight differs'):
        release.combine(a)
    assert not (tmp_path / 'submission.csv').exists() and (base / 'combine/failure.json').is_file()
    # a changed arm CSV fails closed
    shutil.rmtree(base / 'combine')
    (base / 'a/work/result.json').write_text(json.dumps(rj), encoding='utf-8')
    write_csv(base / 'a/arm.csv', [release.UID, *release.TARGETS], [[u, *([0.5] * 12)] for u in ids])
    with pytest.raises(ValueError, match='Arm output changed'):
        release.combine(a)
    assert not (tmp_path / 'submission.csv').exists()


def test_combine_equal_weights_matches_v4_combine_bytes(tmp_path):
    ids = ['1.%d' % i for i in range(31)]
    comp = fixture_comp(tmp_path / 'comp', ids)
    am = two_member_assets(tmp_path)
    pB, pR, _ = synthetic_arms(n=len(ids), seed=11)
    arms5 = [release.build_arm('ens', ['a', 'b'], None, am, 0.35), release.build_arm('a', ['a'], None, am, 0.35)]
    d5 = release_manifest(tmp_path / 'm5.json', arms5, '1' * 64)
    stage_arm_results(tmp_path / 'b5', arms5, d5, ids, [pB, pR])
    release.combine(SimpleNamespace(manifest=str(tmp_path / 'm5.json'), sha256=d5, base=str(tmp_path / 'b5'),
                                    competition=str(comp), out=str(tmp_path / 's5.csv'), started=time.monotonic()))
    am4 = fake_assets(tmp_path / 'new4', {'a': (b'checkpoint-a' * 100, meta()), 'b': (b'checkpoint-b' * 100, meta(tag='U'))},
                      dataset=v4.NEW_DATASET)
    arms4 = [v4.build_arm('ens', ['a', 'b'], None, {**am4, 'schema': v4.ASSETS_SCHEMA}, 0.35),
             v4.build_arm('a', ['a'], None, {**am4, 'schema': v4.ASSETS_SCHEMA}, 0.35)]
    p4 = {'schema': v4.SCHEMA, 'release': 'synthetic', 'release_source_sha256': v4.sha(v4.__file__),
          'old_asset_protocol_sha256': v4.ASSETS_PROTOCOL_SHA, 'new_assets_manifest_sha256': '1' * 64,
          'time_policy': v4.TIME, 'RSS_bytes': v4.RSS, 'output_bytes': v4.OUTPUT, 'prob_clamp': v4.PROB_EPS,
          'retries': 0, 'arms': arms4}
    (tmp_path / 'm4.json').write_text(json.dumps(p4, indent=1), encoding='utf-8')
    d4 = v4.sha(tmp_path / 'm4.json')
    stage_arm_results(tmp_path / 'b4', arms4, d4, ids, [pB, pR])
    v4.combine(SimpleNamespace(manifest=str(tmp_path / 'm4.json'), sha256=d4, base=str(tmp_path / 'b4'),
                               competition=str(comp), out=str(tmp_path / 's4.csv'), started=time.monotonic()))
    assert (tmp_path / 's5.csv').read_bytes() == (tmp_path / 's4.csv').read_bytes()


# ------------------------------------------------------------------ 5. v4 boundaries kept
def test_sample_order_and_dynamic_membership(tmp_path):
    comp = fixture_comp(tmp_path, ('1.2', '1.1'))
    assert release.test_contract(comp)['ids'] == ['1.2', '1.1']
    write_csv(comp / 'test.csv', [release.UID], [['1.1']])
    with pytest.raises(ValueError, match='membership'):
        release.test_contract(comp)


@pytest.mark.parametrize('rows', [
    [['1.1', *([.5] * 12)]],
    [['1.2', *([.5] * 12)], ['1.1', *([.5] * 12)]],
    [['1.1', *(['nan'] * 12)], ['1.2', *([.5] * 12)]],
    [['1.1', *([1.1] * 12)], ['1.2', *([.5] * 12)]],
])
def test_reject_partial_reordered_nonfinite_outofrange(tmp_path, rows):
    path = tmp_path / 'raw.csv'
    write_csv(path, [release.UID, *release.TARGETS], rows)
    with pytest.raises(ValueError):
        release.csv_check(path, ['1.1', '1.2'])


@pytest.mark.parametrize('failure_call', [1, 2])
def test_publish_rolls_back_before_and_after_replace(tmp_path, failure_call):
    raw, final = tmp_path / 'raw.csv', tmp_path / 'submission.csv'
    raw.write_bytes(b'opaque synthetic probabilities')
    calls = []

    def check():
        calls.append(1)
        if len(calls) == failure_call:
            raise ValueError('test metadata changed')
    with pytest.raises(ValueError):
        release.publish(raw, final, release.sha(raw), time.monotonic() + 10, check)
    assert not final.exists() and not final.with_name(final.name + '.exp104-tmp').exists() and raw.exists()


def test_publish_deadline_and_existing_final(tmp_path):
    raw, final = tmp_path / 'raw.csv', tmp_path / 'submission.csv'
    raw.write_bytes(b'new')
    with pytest.raises(ValueError):
        release.publish(raw, final, release.sha(raw), time.monotonic() - 1, lambda: None)
    assert not final.exists()
    final.write_bytes(b'old')
    with pytest.raises(ValueError):
        release.publish(raw, final, release.sha(raw), time.monotonic() + 10, lambda: None)
    assert final.read_bytes() == b'old'
    final.unlink()
    release.publish(raw, final, release.sha(raw), time.monotonic() + 10, lambda: None)
    assert final.read_bytes() == b'new'


def test_resource_validation_has_v4_semantics():
    if not SUPERVISOR.is_file():
        pytest.skip('completed supervisor evidence not present')
    base = json.loads(SUPERVISOR.read_text())
    for kind in ('oom', 'oom_kill'):
        bad = copy.deepcopy(base); bad['cgroup_end']['values']['memory.events'][kind] = 1
        with pytest.raises(ValueError, match='OOM'):
            release.validate_resource(bad, bad['argv'], bad['child_identity']['pid'], 900)
    rss = copy.deepcopy(base); rss['sampled_tree_rss_max_bytes'] = release.RSS + 1
    with pytest.raises(ValueError, match='resource observation'):
        release.validate_resource(rss, rss['argv'], rss['child_identity']['pid'], 900)
    assert release.validate_resource(base, base['argv'], base['child_identity']['pid'], 900) == \
        v4.validate_resource(base, base['argv'], base['child_identity']['pid'], 900)


# ------------------------------------------------------------------ 6. staging, notebook generation
def save_ck(torch, path, **over):
    protocol = json.loads((release.OLD_LOCAL / release.OLD_PROTOCOL).read_text(encoding='utf-8'))
    code = {f: protocol['files']['training/src/' + f] for f in release.TRAINING_CODE_BOUND}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': {'w': torch.randn(3)}, **meta(**over), 'provenance': {'code_sha256': code}}, path)
    return path


def test_stage_assets_dataset_parameter_copy_and_v4_protection(tmp_path):
    torch = pytest.importorskip('torch')
    if not (release.OLD_LOCAL / release.OLD_PROTOCOL).is_file():
        pytest.skip('frozen EXP089 assets not present')
    src = save_ck(torch, tmp_path / 'EXP' / 'model_best.pt')
    pkg = tmp_path / 'pkg'
    a = SimpleNamespace(checkpoint=['coat_f0=' + str(src)], package=str(pkg), copy=True,
                        dataset='taeyangg4/knee-release-assets-v9')
    release.stage_assets(a)
    am = json.loads((pkg / release.NEW_MANIFEST).read_text(encoding='utf-8'))
    assert am['dataset'] == 'taeyangg4/knee-release-assets-v9' and am['schema'] == 'knee-release-assets-v5'
    assert am['checkpoints']['coat_f0']['link_mode'] == 'copy' and os.stat(pkg / 'checkpoints/coat_f0.pt').st_nlink == 1
    md = json.loads((pkg / 'dataset-metadata.json').read_text(encoding='utf-8'))
    assert md['id'] == 'taeyangg4/knee-release-assets-v9' and md['title'] == 'knee-release-assets-v9' and md['isPrivate']
    digest = release.sha(pkg / release.NEW_MANIFEST)
    release.stage_assets(a)
    assert release.sha(pkg / release.NEW_MANIFEST) == digest
    with pytest.raises(ValueError, match='differs from'):
        release.stage_assets(SimpleNamespace(**{**vars(a), 'dataset': 'taeyangg4/knee-release-assets-v8'}))
    for bad in (release.V4_DATASET, release.OLD_DATASET, release.OPENCV_DATASET, 'no-owner', 'A/B'):
        with pytest.raises(ValueError, match='Unsafe new-assets dataset'):
            release.stage_assets(SimpleNamespace(**{**vars(a), 'dataset': bad, 'package': str(tmp_path / 'x')}))
    with pytest.raises(ValueError, match='v4 assets package'):
        release.stage_assets(SimpleNamespace(**{**vars(a), 'package': str(release.V4_PACKAGE)}))
    assert release.PACKAGE != release.V4_PACKAGE and release.NEW_DATASET != release.V4_DATASET


def notebook_args(tmp_path, pkg, arm, **over):
    a = {'release': 'syn', 'arm': arm, 'out': str(tmp_path / ('k-%d' % len(list(tmp_path.glob('k-*'))))),
         'package': str(pkg), 'dataset': 'taeyangg4/knee-release-assets-v9', 'model_seconds': [],
         'expected_studies': 1300, 'docker_from': str(release.DOCKER_PULL)}
    a.update(over)
    return SimpleNamespace(**a)


def test_notebook_generation_weights_seconds_and_screen(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    for p in (release.OLD_LOCAL / release.OLD_PROTOCOL, release.ACCEPTANCE, release.DOCKER_PULL):
        if not Path(p).is_file():
            pytest.skip('frozen assets / acceptance / docker pin not present')
    old = release.verify_old_assets(release.OLD_LOCAL)      # real check once, then reused (577 MB of hashing)
    monkeypatch.setattr(release, 'verify_old_assets', lambda root, opencv=None: old)
    specs = ['c%d=%s' % (k, save_ck(torch, tmp_path / 'src' / ('c%d.pt' % k), tag='C%d' % k)) for k in range(5)]
    specs += ['r%d=%s' % (k, save_ck(torch, tmp_path / 'src' / ('r%d.pt' % k), arch=RN_ARCH, epoch=19, tag='R%d' % k))
              for k in range(5)]
    specs += ['x0=%s' % save_ck(torch, tmp_path / 'src' / 'x0.pt', arch='convnext_small', tag='X')]
    pkg = tmp_path / 'pkg'
    release.stage_assets(SimpleNamespace(checkpoint=specs, package=str(pkg), copy=True,
                                         dataset='taeyangg4/knee-release-assets-v9'))
    coat = 'coat5=c0+c1+c2+c3+c4@36'
    rn = 'rn5=r0+r1+r2+r3+r4@36'
    a = notebook_args(tmp_path, pkg, [coat + ':0.7', rn + ':0.3'])
    release.notebook(a)
    out = Path(a.out)
    p = release.checked_manifest(out / 'release-manifest.json', release.sha(out / 'release-manifest.json'))
    assert p['new_asset_dataset'] == 'taeyangg4/knee-release-assets-v9' and p['rank_weights'] == {'coat5': 0.7, 'rn5': 0.3}
    assert [(x['name'], x['weight'], x['model_seconds_per_study'], x['k_eval']) for x in p['arms']] == \
        [('coat5', 0.7, 0.35, 36), ('rn5', 0.3, 0.15, 36)]
    assert p['new_assets_package'] == str(pkg.resolve()) and 'weighted' in p['combination']
    km = json.loads((out / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert km['id'] == 'taeyangg4/knee-exp104-syn-v5' and km['dataset_sources'][1] == 'taeyangg4/knee-release-assets-v9'
    plan = json.loads((out / 'execution-plan.json').read_text(encoding='utf-8'))
    assert '--timeout' not in plan['push_argv'] and plan['push_argv'][-2:] == ['--accelerator', 'NvidiaTeslaT4']
    assert plan['max_studies_passing_screen'] == 1484 and plan['first_output'] == 'exp104-syn/combine/result.json'
    assert plan['screen_at_expected_hidden_N']['projected_seconds'] == pytest.approx(26879.6121909204, abs=1e-6)
    # equal weights by default
    a = notebook_args(tmp_path, pkg, [coat, rn])
    release.notebook(a)
    p = json.loads((Path(a.out) / 'release-manifest.json').read_text(encoding='utf-8'))
    assert p['rank_weights'] == {'coat5': 0.5, 'rn5': 0.5} and 'EXP098' in p['combination']
    # rules that fail closed before writing anything
    cases = [
        ([coat + ':0.7', rn], {}, 'every arm or for none'),
        ([coat + ':1.0'], {}, 'single-arm'),
        (['x=x0'], {}, 'No model_seconds_per_study default'),
        ([coat, rn], {'model_seconds': ['zz=0.2']}, 'unknown arm'),
        ([coat, rn], {'expected_studies': 1485}, 'time screen'),
        ([coat, rn], {'dataset': 'taeyangg4/knee-release-assets-v8'}, 'schema differs'),
        ([coat, rn], {'dataset': release.V4_DATASET}, 'Unsafe new-assets dataset'),
        (['m=c0+r0@36'], {}, 'differ in arch'),
    ]
    for arm, over, message in cases:
        a = notebook_args(tmp_path, pkg, arm, **over)
        with pytest.raises(ValueError, match=message):
            release.notebook(a)
        assert not Path(a.out).exists()
    a = notebook_args(tmp_path, pkg, ['x=x0'], model_seconds=['x=0.2'])
    release.notebook(a)
    p = json.loads((Path(a.out) / 'release-manifest.json').read_text(encoding='utf-8'))
    assert p['arms'][0]['model_seconds_per_study'] == 0.2 and p['model_seconds_basis']['x'].startswith('override')


def test_render_code_and_kernel_metadata():
    text = '{"x": "@@DIGEST@@ stays literal", "new_asset_dataset": "taeyangg4/knee-release-assets-v5"}\n'
    digest = hashlib.sha256(text.encode()).hexdigest()
    code = release.render_code('coat5', b'print(1)  # @@SOURCE@@ is not a placeholder\n', text, digest)
    assert "Path('/kaggle/working/exp104-coat5')" in code and repr(text) in code and repr(digest) in code
    assert "Path('/kaggle/input/'+new_slug)" in code and "'/kaggle/input/datasets/'+new_owner+'/'+new_slug" in code
    assert 'knee-release-assets-v4' not in code
    km = release.kernel_metadata('coat5rn5', release.DOCKER_IMAGE)
    assert km['id'] == 'taeyangg4/knee-exp104-coat5rn5-v5' and km['is_private'] and km['enable_gpu']
    assert not km['enable_internet'] and km['machine_shape'] == 'NvidiaTeslaT4'
    assert km['competition_sources'] == ['rsna-knee-abnormality-detection']
    assert km['dataset_sources'] == ['taeyangg4/knee-exp089-recipe-transfer-v1', 'taeyangg4/knee-release-assets-v5',
                                     'mattiaangeli/opencv-python-headless-4120088-x86']
    assert release.kernel_metadata('a', release.DOCKER_IMAGE, 'taeyangg4/other-assets')['dataset_sources'][1] == \
        'taeyangg4/other-assets'
    with pytest.raises(ValueError):
        release.kernel_metadata('coat5', 'gcr.io/kaggle-gpu-images/python:latest')
    with pytest.raises(ValueError):
        release.kernel_metadata('Bad_Name', release.DOCKER_IMAGE)
    with pytest.raises(ValueError):
        release.kernel_metadata('coat5', release.DOCKER_IMAGE, release.V4_DATASET)


def test_rendered_notebook_lines_fit_kaggle_limit():
    big = os.urandom(200_000)
    manifest_text = '{"x": 1}'
    digest = hashlib.sha256(manifest_text.encode('utf-8')).hexdigest()
    code = release.render_code('coat5', big, manifest_text, digest)
    assert max(len(line) for line in code.split('\n')) <= release.MAX_NOTEBOOK_LINE
    src_expr = code.split('source=base64.b85decode(', 1)[1].split('\nassert hashlib.sha256(source)', 1)[0].rstrip()[:-1]
    assert __import__('base64').b85decode(eval(src_expr, {})) == big
    # a long multi-line manifest with quotes, backslashes and non-ASCII is split into short lines and round-trips
    text = json.dumps({'k%d' % i: 'v"\\é\n' * 40 for i in range(120)}, indent=2, ensure_ascii=False) + '\n'
    assert len(text) > 3 * release.SOURCE_CHUNK
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    code = release.render_code('coat5', b'x', text, digest)
    assert max(len(line) for line in code.split('\n')) <= 2 * release.SOURCE_CHUNK
    expr = code.split('\nMANIFEST=', 1)[1].split('\nDIGEST=', 1)[0]
    assert eval(expr, {}) == text


EXPECTED = {'coat5': [('coat5', ['coat_f%d' % k for k in range(5)], 36, 1.0, 0.35)],
            'coat5rn5': [('coat5', ['coat_f%d' % k for k in range(5)], 36, 0.7, 0.35),
                         ('rn5', ['rn_f%d' % k for k in range(5)], 36, 0.3, 0.15)]}


def kernel_dir(name):
    out = KERNELS / name
    if not (out / 'kernel-metadata.json').is_file() or not (release.PACKAGE / release.NEW_MANIFEST).is_file():
        pytest.skip('kernel dir or assets package not generated')
    return out


@pytest.mark.parametrize('name', ['coat5', 'coat5rn5'])
def test_generated_kernel_dirs_are_current(name):
    """The staged kernels embed this exact tool, bind the staged v5 package and pin the Python 3.12 image."""
    out = kernel_dir(name)
    km = json.loads((out / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert km == release.kernel_metadata(name, release.DOCKER_IMAGE, release.NEW_DATASET)
    pulled = json.loads(release.DOCKER_PULL.read_text(encoding='utf-8'))
    assert pulled['id'] == release.DOCKER_SOURCE_KERNEL and pulled['docker_image'] == km['docker_image']
    digest = release.sha(out / 'release-manifest.json')
    p = release.checked_manifest(out / 'release-manifest.json', digest)
    assert [(a['name'], [c['name'] for c in a['checkpoints']], a['k_eval'], a['weight'], a['model_seconds_per_study'])
            for a in p['arms']] == EXPECTED[name]
    assert all(c['meta']['k_eval'] == 24 for a in p['arms'] for c in a['checkpoints'])
    assert p['new_asset_dataset'] == 'taeyangg4/knee-release-assets-v5'
    release.verify_new_assets(p, release.PACKAGE, p['arms'])
    code = ''.join(json.loads((out / 'release.ipynb').read_text(encoding='utf-8'))['cells'][0]['source'])
    assert code == release.render_code(name, Path(release.__file__).read_bytes(),
                                       (out / 'release-manifest.json').read_bytes().decode('utf-8'), digest)
    assert max(len(line) for line in code.split('\n')) <= release.MAX_NOTEBOOK_LINE
    plan = json.loads((out / 'execution-plan.json').read_text(encoding='utf-8'))
    assert plan['manifest_sha256'] == digest and plan['notebook_sha256'] == release.sha(out / 'release.ipynb')
    assert '--timeout' not in plan['push_argv']
    assert plan['max_studies_passing_screen'] == {'coat5': 2788, 'coat5rn5': 1484}[name]


# ------------------------------------------------------------------ 7. simulated notebook cell on a fake /kaggle tree
def fake_kaggle(root, n, legacy_mount):
    k = Path(root) / 'kaggle'
    (k / 'working').mkdir(parents=True)
    inp = k / 'input'
    old = inp / 'knee-exp089-recipe-transfer-v1' if legacy_mount else inp / 'datasets/taeyangg4/knee-exp089-recipe-transfer-v1'
    new = inp / 'knee-release-assets-v5' if legacy_mount else inp / 'datasets/taeyangg4/knee-release-assets-v5'
    cv = inp / 'datasets/mattiaangeli/opencv-python-headless-4120088-x86'
    comp = inp / 'rsna-knee-abnormality-detection' if legacy_mount else inp / 'competitions/rsna-knee-abnormality-detection'
    for d, f in ((old, 'exp089-t4-protocol.json'), (new, 'assets-manifest.json'),
                 (cv, 'opencv_python_headless-4.12.0.88-cp37-abi3-manylinux_2_17_x86_64.whl')):
        d.mkdir(parents=True)
        (d / f).write_text('{}', encoding='utf-8')
    ids = ['1.2.840.%d' % (9000 + 7 * i) for i in range(n)]
    fixture_comp(comp, ids)
    return k, ids


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
            tools['m'] = load_path('embedded_v5_' + name + '_%d' % n, argv[1])
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


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['coat5', 'coat5rn5'])
@pytest.mark.parametrize('n,legacy', [(3, False), (4, True), (1300, False)])
def test_simulated_notebook_cell(monkeypatch, tmp_path, capsys, name, n, legacy):
    k, ids, calls, tables = simulate_cell(monkeypatch, tmp_path, name, n, legacy)
    final = k / 'working/submission.csv'
    base = k / 'working' / ('exp104-' + name)
    assert (base / 'exp104_release_v5.py').read_bytes() == Path(release.__file__).read_bytes()
    release.csv_check(final, ids)
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['studies'] == n and printed['submission_sha256'] == release.sha(final)
    boots = ['600'] * len(EXPECTED[name]) if n <= 3 else ['600', '30000'][:len(EXPECTED[name])]
    assert [c for c in calls if c[0] == 'run'] == [('run', arm[0], b) for arm, b in zip(EXPECTED[name], boots)]
    if name == 'coat5':
        assert calls == [('run', 'coat5', '600')] and printed['status'] == 'PASS_TEST_COHORT_ONLY'
        assert printed['rank_weights'] == {'coat5': 1.0}
    else:
        assert calls[-1] == ('combine', None) and printed['status'] == 'PASS_RANK_MEAN_TEST_COHORT_ONLY'
        assert printed['rank_weights'] == {'coat5': 0.7, 'rn5': 0.3}
        assert {a: v['rank_weight'] for a, v in printed['arms'].items()} == {'coat5': 0.7, 'rn5': 0.3}
        r = json.loads((base / 'combine/result.json').read_text(encoding='utf-8'))
        assert r['arm_weights'] == {'coat5': 0.7, 'rn5': 0.3} and r['rank_weights'] == {'coat5': 0.7, 'rn5': 0.3}
        release.write_columns(base / 'expected.csv', ids, release.rank_mean([tables['coat5'], tables['rn5']], [0.7, 0.3]))
        assert final.read_bytes() == (base / 'expected.csv').read_bytes()
    assert not list((k / 'working').glob('*.exp104-tmp'))


@pytest.mark.parametrize('name,n', [('coat5rn5', 1485), ('coat5', 2789)])
def test_simulated_notebook_cell_fails_closed_above_budget(monkeypatch, tmp_path, name, n):
    with pytest.raises(AssertionError, match='v5 time screen failed'):
        simulate_cell(monkeypatch, tmp_path, name, n)
    k = tmp_path / 'fake/kaggle'
    assert not (k / 'working/submission.csv').exists()
    assert not any((k / 'working' / ('exp104-' + name)).glob('*/work'))


# ------------------------------------------------------------------ 8. opt-in real checkpoint smoke (CPU)
@pytest.mark.skipif(os.environ.get('EXP104_REAL_SMOKE') != '1',
                    reason='opt-in: loads one real CoAtNet-2 (~290 MB) and one real ResNet34 (~86 MB) checkpoint on CPU')
def test_real_staged_checkpoints_cpu_smoke(monkeypatch):
    """Worker path after the platform gate, on CPU: frozen load_checkpoint -> checkpoint_check -> wrapper -> predict."""
    torch = pytest.importorskip('torch')
    rt = frozen_runtime(monkeypatch)
    from raptor_common import load_checkpoint
    torch.set_num_threads(8)
    out = kernel_dir('coat5rn5')
    p = release.checked_manifest(out / 'release-manifest.json', release.sha(out / 'release-manifest.json'))
    release.verify_new_assets(p, release.PACKAGE, p['arms'])
    members = []
    for arm, ck_name, tag in ((p['arms'][0], 'coat_f0', 'EXP094L_B8_SEED42'), (p['arms'][1], 'rn_f0', 'EXP101_RN34_DREAD_F0')):
        ck = [c for c in arm['checkpoints'] if c['name'] == ck_name][0]
        model, m = load_checkpoint(release.PACKAGE / ck['path'])
        r = release.checkpoint_check(m, arm, ck)
        assert r['k_eval_checkpoint'] == 24 and r['k_eval_release'] == 36 and r['tag'] == tag
        members.append(model.eval())
    ids = ['9.1']

    def go(model):
        corpus = FakeCorpus(ids, res=224)
        probs, timing = rt.predict(model, corpus, ids, 36, 224, 'none', 'cpu', norm='imagenet', batch=1,
                                   log=lambda *a: None, window_policy='within-slot-valid')
        return probs, corpus.decodes
    singles = [go(m)[0] for m in members]
    pe, decodes = go(release.mean_prob_module(members))
    assert decodes == 1 and np.isfinite(pe).all()
    assert np.abs(pe - (singles[0] + singles[1]) / 2).max() <= 1e-6
    assert np.abs(singles[0] - singles[1]).max() > 1e-4
    p1, _ = go(release.mean_prob_module(members[1:]))
    assert np.abs(p1 - singles[1]).max() <= 1e-6
