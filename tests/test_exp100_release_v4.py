"""CPU tests for the EXP100 v4 ensemble release: no GPU, no network, no real study data or real model loads."""
import copy
import csv
import importlib.util
import json
import os
import random
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


release = load('exp100_release_v4_under_test', 'tools/exp100_release_v4.py')
v31 = load('recipe_release_v31_for_v4', 'tools/exp089_recipe_release_v31.py')
exp098 = load('exp098_e1_release_for_v4', 'tools/exp098_e1_release.py')
SUPERVISOR = ROOT / 'docs/handoffs/evidence/2026-10/2026-10-05-exp089-recipe-transfer-v1/t4-results-v2/B224-supervisor.json'
ARCH = 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'
CONTRACT = 'raptor64_336_c140_s02'


# ------------------------------------------------------------------ fixtures
def meta(**over):
    m = {'arch': ARCH, 'res': 224, 'lab': list(release.TARGETS), 'contract': CONTRACT, 'k_eval': 24,
         'norm': 'imagenet', 'window_policy': 'within-slot-valid', 'epoch': 7, 'tag': 'T', 'amp_train': 'bf16',
         'amp_eval': 'fp16'}
    m.update(over)
    return m


def fake_assets(root, members):
    """members: {name: (bytes, meta)} -> new-assets dir with a v4 assets manifest."""
    root = Path(root)
    (root / 'checkpoints').mkdir(parents=True)
    checkpoints = {}
    for name, (blob, m) in members.items():
        path = root / 'checkpoints' / (name + '.pt')
        path.write_bytes(blob)
        checkpoints[name] = {'path': 'checkpoints/' + name + '.pt', 'sha256': release.sha(path), 'bytes': len(blob),
                             'meta': m, 'training_code_sha256': {}, 'source': 'synthetic', 'source_sidecars_sha256': {},
                             'link_mode': 'copy'}
    am = {'schema': release.ASSETS_SCHEMA, 'dataset': release.NEW_DATASET, 'checkpoints': checkpoints,
          'files': {c['path']: c['sha256'] for c in checkpoints.values()}, 'dataset_metadata_sha256': '0' * 64}
    (root / release.NEW_MANIFEST).write_text(json.dumps(am), encoding='utf-8')
    return am


def release_manifest(path, arm_list, assets_manifest_sha, **over):
    p = {'schema': release.SCHEMA, 'release': 'synthetic', 'release_source_sha256': release.sha(release.__file__),
         'old_asset_protocol_sha256': release.ASSETS_PROTOCOL_SHA, 'new_assets_manifest_sha256': assets_manifest_sha,
         'time_policy': release.TIME, 'RSS_bytes': release.RSS, 'output_bytes': release.OUTPUT,
         'prob_clamp': release.PROB_EPS, 'retries': 0, 'arms': arm_list}
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


# ------------------------------------------------------------------ 1. ensemble wrapper
def test_mean_prob_wrapper_is_mean_of_probabilities_not_logits():
    torch = pytest.importorskip('torch')
    x = torch.linspace(-6, 6, 24).reshape(2, 12)
    models = [affine(torch, 1.0, 0.0), affine(torch, -0.5, 1.0), affine(torch, 2.0, -3.0)]
    ens = release.mean_prob_module(models).eval()
    with torch.inference_mode():
        got = torch.sigmoid(ens(x))
        want = torch.stack([torch.sigmoid(m(x)) for m in models]).mean(0)
        logit_mean = torch.sigmoid(torch.stack([m(x) for m in models]).mean(0))
    assert (got - want).abs().max().item() <= 1e-6
    assert (got - logit_mean).abs().max().item() > 1e-2
    assert len(list(ens.members)) == 3


def test_single_model_identity_and_clamp():
    torch = pytest.importorskip('torch')
    ens = release.mean_prob_module([affine(torch, 1.0, 0.0)]).eval()
    z = torch.linspace(-10, 10, 240).reshape(20, 12)
    with torch.inference_mode():
        assert (torch.sigmoid(ens(z)) - torch.sigmoid(z)).abs().max().item() <= 1e-6
        extreme = torch.tensor([[-80.0] * 6 + [80.0] * 6])
        out = ens(extreme)
        p = torch.sigmoid(out)
    assert torch.isfinite(out).all()
    assert torch.allclose(p[0, :6], torch.full((6,), 1e-6), rtol=0, atol=1e-9)
    assert torch.allclose(p[0, 6:], torch.full((6,), 1 - 1e-6), rtol=0, atol=1e-7)


def test_wrapper_aggregates_in_fp32_under_autocast():
    torch = pytest.importorskip('torch')
    torch.manual_seed(0)
    members = [torch.nn.Linear(12, 12), torch.nn.Linear(12, 12)]
    ens = release.mean_prob_module(members).eval()
    x = torch.linspace(-4, 4, 24).reshape(2, 12)
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
        low = [m(x) for m in members]
        out = ens(x)
    assert low[0].dtype == torch.bfloat16                      # members really run in reduced precision
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    want = torch.stack([torch.sigmoid(z.float()) for z in low]).mean(0)
    assert (torch.sigmoid(out) - want).abs().max().item() <= 1e-6


def frozen_runtime(monkeypatch):
    """The serving copy of raptor_runtime from the frozen EXP089 assets (byte-identical to training/src today)."""
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
    return load('frozen_raptor_runtime_for_v4', str((src / 'raptor_runtime.py').relative_to(ROOT)))


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


def test_wrapper_through_unchanged_predict_decodes_each_study_once(monkeypatch):
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
    assert decodes == len(ids)                      # one decode per study for three models
    assert pe.shape == (3, 12) and pe.dtype == np.float32
    assert np.abs(pe - np.mean([s[0] for s in singles], axis=0)).max() <= 1e-6
    assert te['ordered_inputs_sha256'] == singles[0][1]['ordered_inputs_sha256']
    assert np.abs(singles[0][0] - singles[1][0]).max() > 1e-3   # members genuinely differ
    p1, _, _ = go(release.mean_prob_module(members[:1]))
    assert np.abs(p1 - singles[0][0]).max() <= 1e-6


# ------------------------------------------------------------------ 2. manifests and assets
def two_member_assets(tmp_path, b_meta=None):
    return fake_assets(tmp_path / 'new', {'a': (b'checkpoint-a' * 100, meta()),
                                          'b': (b'checkpoint-b' * 100, b_meta or meta(tag='U'))})


def test_assets_binding_accepts_then_rejects_sha_mismatch_and_missing(tmp_path):
    am = two_member_assets(tmp_path)
    root = tmp_path / 'new'
    arm = release.build_arm('ens', ['a', 'b'], None, am, 0.35)
    rel = {'new_assets_manifest_sha256': release.sha(root / release.NEW_MANIFEST)}
    release.verify_new_assets(rel, root, [arm])
    (root / 'checkpoints/b.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='sha256 differs'):
        release.verify_new_assets(rel, root, [arm])
    (root / 'checkpoints/b.pt').unlink()
    with pytest.raises(ValueError, match='Missing checkpoint'):
        release.verify_new_assets(rel, root, [arm])
    with pytest.raises(ValueError, match='assets manifest changed'):
        release.verify_new_assets({'new_assets_manifest_sha256': '0' * 64}, root, [arm])
    forged = copy.deepcopy(arm)
    forged['checkpoints'][0]['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='not bound'):
        release.verify_new_assets(rel, root, [forged])


def test_release_manifest_validation(tmp_path):
    am = two_member_assets(tmp_path)
    arms = [release.build_arm('ens', ['a', 'b'], None, am, 0.35), release.build_arm('a', ['a'], 36, am, 0.35)]
    path = tmp_path / 'release-manifest.json'
    digest = release_manifest(path, arms, '1' * 64)
    p = release.checked_manifest(path, digest)
    assert [a['name'] for a in p['arms']] == ['ens', 'a'] and p['arms'][1]['k_eval'] == 36
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
    ]
    for i, (over, message) in enumerate(bad_cases):
        bad = tmp_path / ('bad-%d.json' % i)
        d = release_manifest(bad, arms, '1' * 64, **over)
        with pytest.raises(ValueError, match=message):
            release.checked_manifest(bad, d)


def test_build_arm_rules(tmp_path):
    am = two_member_assets(tmp_path)
    with pytest.raises(ValueError, match='Unknown staged checkpoint'):
        release.build_arm('x', ['a', 'missing'], None, am, 0.35)
    mixed = fake_assets(tmp_path / 'mixed', {'a': (b'a', meta()), 'r': (b'r', meta(res=384, arch='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'))})
    with pytest.raises(ValueError, match='differ in arch'):
        release.build_arm('x', ['a', 'r'], None, mixed, 0.35)
    keval = fake_assets(tmp_path / 'keval', {'a': (b'a', meta()), 'b': (b'b', meta(k_eval=36))})
    with pytest.raises(ValueError, match='different k_eval'):
        release.build_arm('x', ['a', 'b'], None, keval, 0.35)
    assert release.build_arm('x', ['a', 'b'], 24, keval, 0.35)['k_eval'] == 24


def test_parse_arm_spec():
    assert release.parse_arm_spec('student') == ('student', ['student'], None)
    assert release.parse_arm_spec('student@36') == ('student', ['student'], 36)
    assert release.parse_arm_spec('ens=student+control@24') == ('ens', ['student', 'control'], 24)
    assert release.parse_arm_spec('ens=student+control') == ('ens', ['student', 'control'], None)
    for bad in ['', 'a=b+b', 'a=', '../x', 'a@0', 'a=b+', 'a b']:
        with pytest.raises(ValueError):
            release.parse_arm_spec(bad)


def test_checkpoint_metadata_check_records_both_k_eval(tmp_path):
    am = two_member_assets(tmp_path)
    arm = release.build_arm('a36', ['a'], 36, am, 0.35)
    ck = arm['checkpoints'][0]
    loaded = {**meta(), 'provenance': {'ignored': True}, 'lab': tuple(release.TARGETS)}
    r = release.checkpoint_check(loaded, arm, ck)
    assert r['k_eval_checkpoint'] == 24 and r['k_eval_release'] == 36 and r['sha256'] == ck['sha256']
    for key, value, message in [('arch', 'other', 'differs from arm'), ('window_policy', 'legacy', 'differs from arm'),
                                ('norm', 'none', 'differs from arm'), ('lab', release.TARGETS[::-1], 'differs from arm'),
                                ('epoch', 3, 'staged record'), ('k_eval', 62, 'staged record')]:
        with pytest.raises(ValueError, match=message):
            release.checkpoint_check({**meta(), key: value}, arm, ck)


# ------------------------------------------------------------------ 3. time screen
def arms_of(*counts, m=0.35):
    return [{'name': 'a%d' % i, 'checkpoints': [None] * c, 'model_seconds_per_study': m} for i, c in enumerate(counts)]


def test_time_screen_arithmetic():
    d = 6.52503319258
    assert release.operational_plan(1300, arms_of(1))['projected_seconds'] == pytest.approx(1.3 * (d + 0.35) * 1300 + 600, abs=1e-9)
    two = release.operational_plan(1300, arms_of(1, 2))
    assert two['projected_seconds'] == pytest.approx(1.3 * (d + 0.35) * 1300 + 1.3 * (d + 0.70) * 1300 + 600, abs=1e-9)
    assert two['arm_projected_seconds']['a1'] == pytest.approx(1.3 * (d + 0.70) * 1300, abs=1e-9)
    # EXP098's two-arm screen 2*1.3*6.525*N+600 is the zero-model-cost special case
    assert release.operational_plan(1300, arms_of(1, 1, m=0.0))['projected_seconds'] == pytest.approx(2 * 1.3 * d * 1300 + 600, abs=1e-9)
    # measured per-model constant replaces the default through the manifest field
    assert release.operational_plan(1300, arms_of(4, m=0.1))['projected_seconds'] == pytest.approx(1.3 * (d + 0.4) * 1300 + 600, abs=1e-9)
    assert release.operational_plan(3, arms_of(1))['internal_seconds'] == 600
    assert release.operational_plan(1300, arms_of(1))['internal_seconds'] == 32100
    for arms in (arms_of(1), arms_of(1, 1), arms_of(3, 2)):
        n = release.max_studies(arms)
        release.operational_plan(n, arms)
        with pytest.raises(ValueError, match='time screen'):
            release.operational_plan(n + 1, arms)
    assert release.max_studies(arms_of(1)) == 3356
    with pytest.raises(ValueError, match='time screen'):
        release.operational_plan(1300, arms_of(4, 4, 4))


def test_launch_projection_covers_remaining_arms():
    arms = arms_of(1, 2)
    plan = release.operational_plan(1300, arms)
    first = release.launch_projection(plan, arms, 'a0')
    last = release.launch_projection(plan, arms, 'a1')
    assert first == pytest.approx(plan['projected_seconds'] - 600 + 300)
    assert last == pytest.approx(plan['arm_projected_seconds']['a1'] + 300)
    with pytest.raises(ValueError, match='Unknown arm'):
        release.launch_projection(plan, arms, 'zz')


# ------------------------------------------------------------------ 4. launch_budget reuse (v3.1 semantics)
@pytest.mark.parametrize('started,now,n,boot', [
    (1000.0, 10000.0, 1300, 30000.0), (1000.0, 10000.0, 1300, 600.0), (1000.0, 26000.0, 1300, 30000.0),
    (5.0, 65.0, 3, 600.0), (5.0, 705.0, 3, 30000.0), (1000.0, 1005.0, 3, 30000.0), (1000.0, 1060.0, 1300, 30000.0),
    (1000.0, 13000.0, 1760, 30000.0), (1000.0, 999.0, 1300, 600.0), (float('nan'), 1000.0, 1300, 600.0),
    (1000.0, 1300.0, 1300, float('inf')), (1000.0, 1000.0, 1300, 0.0)])
def test_launch_budget_matches_v31(started, now, n, boot):
    projected = v31.operational_plan(n)['projected_seconds']
    assert release.arm_projection(n, 1, 0.0) + release.TIME['arm_fixed_seconds'] == pytest.approx(projected, abs=1e-9)
    try:
        _, deadline, timeout = v31.launch_budget(started, now, n, boot)
    except ValueError as exc:
        with pytest.raises(ValueError, match=str(exc).split()[0]):
            release.launch_budget(started, now, n, projected, boot)
        return
    assert release.launch_budget(started, now, n, projected, boot) == (deadline, timeout)


def test_launch_budget_late_ensemble_arm():
    arms = arms_of(1, 2)
    plan = release.operational_plan(1300, arms)
    last = release.launch_projection(plan, arms, 'a1')
    release.launch_budget(1000.0, 1000.0 + 12000.0, 1300, last, 30000.0)
    with pytest.raises(ValueError, match='projection'):
        release.launch_budget(1000.0, 1000.0 + 32100 - 30 - last + 1, 1300, last, 30000.0)


# ------------------------------------------------------------------ 5. rank combination identical to EXP098
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


def test_rank_mean_identical_to_exp098(tmp_path):
    rank = exp098_rank()
    pB, pR, pC = synthetic_arms()
    cols98 = []
    for t in range(12):
        rb = rank([r[t] for r in pB]); rr = rank([r[t] for r in pR]); cols98.append([(a + b) / 2 for a, b in zip(rb, rr)])
    assert release.rank_mean([pB, pR]) == cols98                     # exact, not approximate
    for t in range(12):
        assert release.average_rank([r[t] for r in pR]) == rank([r[t] for r in pR])
    three = release.rank_mean([pB, pR, pC])
    assert three[5][:7] == [(x + y + z) / 3 for x, y, z in zip(rank([r[5] for r in pB]), rank([r[5] for r in pR]),
                                                                     rank([r[5] for r in pC]))][:7]
    ids = ['1.%d' % i for i in range(len(pB))]
    release.write_columns(tmp_path / 'v4.csv', ids, cols98)
    with open(tmp_path / 'e1.csv', 'w', encoding='utf-8', newline='') as f:   # EXP098 writer, verbatim
        w = csv.writer(f); w.writerow(['StudyInstanceUID', *release.TARGETS])
        for i, u in enumerate(ids):
            w.writerow([u, *[repr(cols98[t][i]) for t in range(12)]])
    assert (tmp_path / 'v4.csv').read_bytes() == (tmp_path / 'e1.csv').read_bytes()
    with pytest.raises(ValueError):
        release.rank_mean([pB])


def test_combine_end_to_end_and_tamper(tmp_path):
    ids = ['1.%d' % i for i in range(9)]
    comp = fixture_comp(tmp_path / 'comp', ids)
    am = two_member_assets(tmp_path)
    arms = [release.build_arm('ens', ['a', 'b'], None, am, 0.35), release.build_arm('a', ['a'], None, am, 0.35)]
    digest = release_manifest(tmp_path / 'm.json', arms, '1' * 64)
    pB, pR, _ = synthetic_arms(n=len(ids), seed=3)
    base = tmp_path / 'base'
    for arm, probs in zip(arms, [pB, pR]):
        d = base / arm['name']
        (d / 'work').mkdir(parents=True)
        write_csv(d / 'arm.csv', [release.UID, *release.TARGETS], [[u, *map(repr, p)] for u, p in zip(ids, probs)])
        (d / 'work/result.json').write_text(json.dumps({
            'status': 'PASS_TEST_COHORT_ONLY', 'arm': arm['name'], 'release_manifest_sha256': digest,
            'studies': len(ids), 'submission_sha256': release.sha(d / 'arm.csv'), 'elapsed_seconds': 1.0,
            'sampled_tree_rss_bytes': 1, 'n_models': len(arm['checkpoints']), 'k_eval': 24}), encoding='utf-8')
    a = SimpleNamespace(manifest=str(tmp_path / 'm.json'), sha256=digest, base=str(base), competition=str(comp),
                        out=str(tmp_path / 'submission.csv'), started=time.monotonic())
    r = release.combine(a)
    assert r['status'] == 'PASS_RANK_MEAN_TEST_COHORT_ONLY' and r['arm_order'] == ['ens', 'a']
    release.write_columns(tmp_path / 'expected.csv', ids, release.rank_mean([pB, pR]))
    assert (tmp_path / 'submission.csv').read_bytes() == (tmp_path / 'expected.csv').read_bytes()
    assert release.sha(tmp_path / 'submission.csv') == r['submission_sha256']
    assert not list(tmp_path.glob('*.exp100-tmp'))
    # a changed arm CSV fails closed: no submission, failure receipt
    (tmp_path / 'submission.csv').unlink()
    import shutil
    shutil.rmtree(base / 'combine')
    write_csv(base / 'a/arm.csv', [release.UID, *release.TARGETS], [[u, *([0.5] * 12)] for u in ids])
    with pytest.raises(ValueError, match='Arm output changed'):
        release.combine(a)
    assert not (tmp_path / 'submission.csv').exists() and (base / 'combine/failure.json').is_file()


# ------------------------------------------------------------------ 6. v3 boundaries kept
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
    assert not final.exists() and not final.with_name(final.name + '.exp100-tmp').exists() and raw.exists()


def test_publish_deadline_and_existing_final(tmp_path):
    raw, final = tmp_path / 'raw.csv', tmp_path / 'submission.csv'
    raw.write_bytes(b'new')
    with pytest.raises(ValueError):
        release.publish(raw, final, release.sha(raw), time.monotonic() - 1, lambda: None)
    assert not final.exists() and not final.with_name(final.name + '.exp100-tmp').exists()
    final.write_bytes(b'old')
    with pytest.raises(ValueError):
        release.publish(raw, final, release.sha(raw), time.monotonic() + 10, lambda: None)
    assert final.read_bytes() == b'old'
    final.unlink()
    release.publish(raw, final, release.sha(raw), time.monotonic() + 10, lambda: None)
    assert final.read_bytes() == b'new'


def test_resource_validation_has_v3_semantics():
    if not SUPERVISOR.is_file():
        pytest.skip('completed supervisor evidence not present')
    base = json.loads(SUPERVISOR.read_text())
    s = copy.deepcopy(base)
    for key in ('cgroup_start', 'cgroup_end'):
        s[key]['values']['memory.events']['max'] = 12345
        s[key]['values']['memory.events']['high'] = 7
    s['cgroup_telemetry_complete'] = False
    t = release.validate_resource(s, s['argv'], s['child_identity']['pid'], 900)
    assert t['snapshots']['cgroup_end']['memory.events']['max'] == 12345 and t['cgroup_telemetry_complete'] is False
    for kind in ('oom', 'oom_kill'):
        bad = copy.deepcopy(base); bad['cgroup_end']['values']['memory.events'][kind] = 1
        with pytest.raises(ValueError, match='OOM'):
            release.validate_resource(bad, bad['argv'], bad['child_identity']['pid'], 900)
    rss = copy.deepcopy(base); rss['sampled_tree_rss_max_bytes'] = release.RSS + 1
    with pytest.raises(ValueError, match='resource observation'):
        release.validate_resource(rss, rss['argv'], rss['child_identity']['pid'], 900)
    kill = copy.deepcopy(base); kill['cleanup']['actions'] = [{'pid': 45, 'action': 'kill'}]
    with pytest.raises(ValueError, match='cleanup'):
        release.validate_resource(kill, kill['argv'], kill['child_identity']['pid'], 900)


# ------------------------------------------------------------------ 7. staging and notebook generation
def test_stage_assets_hashes_records_metadata_and_is_idempotent(tmp_path):
    torch = pytest.importorskip('torch')
    src = tmp_path / 'EXP' / 'model_best.pt'
    src.parent.mkdir()
    fields = {**meta(), 'provenance': {'code_sha256': {'raptor_common.py': 'x'}}}
    torch.save({'model': {'w': torch.zeros(3)}, **fields}, src)
    (src.parent / 'summary.json').write_text('{}')
    pkg = tmp_path / 'pkg'
    a = SimpleNamespace(checkpoint=['student=' + str(src)], package=str(pkg), copy=False)
    release.stage_assets(a)
    am = json.loads((pkg / release.NEW_MANIFEST).read_text(encoding='utf-8'))
    assert am['files'] == {'checkpoints/student.pt': release.sha(src)}
    c = am['checkpoints']['student']
    assert c['meta'] == meta() and c['training_code_sha256'] == {'raptor_common.py': 'x'}
    assert c['link_mode'] in ('hardlink', 'copy') and 'summary.json' in c['source_sidecars_sha256']
    assert json.loads((pkg / 'dataset-metadata.json').read_text())['id'] == 'taeyangg4/knee-release-assets-v4'
    digest = release.sha(pkg / release.NEW_MANIFEST)
    release.stage_assets(a)
    assert release.sha(pkg / release.NEW_MANIFEST) == digest
    other = tmp_path / 'other.pt'
    torch.save({'model': {'w': torch.ones(3)}, **fields}, other)
    with pytest.raises(ValueError, match='Refusing'):
        release.stage_assets(SimpleNamespace(checkpoint=['student=' + str(other)], package=str(pkg), copy=True))
    release.stage_assets(SimpleNamespace(checkpoint=['control=' + str(other)], package=str(pkg), copy=True))
    am2 = json.loads((pkg / release.NEW_MANIFEST).read_text(encoding='utf-8'))
    assert sorted(am2['checkpoints']) == ['control', 'student'] and am2['checkpoints']['control']['link_mode'] == 'copy'
    (pkg / 'checkpoints' / 'stray.bin').write_bytes(b'x')
    with pytest.raises(ValueError, match='Unexpected'):
        release.stage_assets(a)


def test_render_code_and_kernel_metadata():
    text = '{"x": "@@DIGEST@@ stays literal"}\n'
    import hashlib
    digest = hashlib.sha256(text.encode()).hexdigest()
    source = b'print(1)  # @@SOURCE@@ inside the payload is not a placeholder\n'
    code = release.render_code('student', source, text, digest)
    assert "Path('/kaggle/working/exp100-student')" in code and repr(text) in code and repr(digest) in code
    meta_k = release.kernel_metadata('student', release.DOCKER_IMAGE)
    assert meta_k['id'] == 'taeyangg4/knee-exp100-student-v4' and meta_k['is_private'] and meta_k['enable_gpu']
    assert not meta_k['enable_internet'] and meta_k['machine_shape'] == 'NvidiaTeslaT4'
    assert meta_k['competition_sources'] == ['rsna-knee-abnormality-detection']
    assert meta_k['dataset_sources'][:2] == ['taeyangg4/knee-exp089-recipe-transfer-v1', 'taeyangg4/knee-release-assets-v4']
    assert '@sha256:' in meta_k['docker_image']
    with pytest.raises(ValueError):
        release.kernel_metadata('student', 'gcr.io/kaggle-gpu-images/python:latest')
    with pytest.raises(ValueError):
        release.kernel_metadata('Bad_Name', release.DOCKER_IMAGE)


@pytest.mark.parametrize('name', ['student', 'control'])
def test_generated_kernel_dirs_are_current(name):
    """The staged kernels must embed this exact tool, bind the staged package and pin the Python 3.12 image."""
    out = ROOT / '.local/exp100-release-v4' / name
    if not (out / 'kernel-metadata.json').is_file() or not (release.PACKAGE / release.NEW_MANIFEST).is_file():
        pytest.skip('kernel dir or assets package not generated')
    km = json.loads((out / 'kernel-metadata.json').read_text(encoding='utf-8'))
    assert km == release.kernel_metadata(name, release.DOCKER_IMAGE)
    pulled = json.loads(release.DOCKER_PULL.read_text(encoding='utf-8'))
    assert pulled['id'] == release.DOCKER_SOURCE_KERNEL and pulled['docker_image'] == km['docker_image']
    digest = release.sha(out / 'release-manifest.json')
    p = release.checked_manifest(out / 'release-manifest.json', digest)
    assert [a['name'] for a in p['arms']] == [name] and p['arms'][0]['k_eval'] == 24
    assert [c['name'] for c in p['arms'][0]['checkpoints']] == [name]
    release.verify_new_assets(p, release.PACKAGE, p['arms'])
    code = ''.join(json.loads((out / 'release.ipynb').read_text(encoding='utf-8'))['cells'][0]['source'])
    assert code == release.render_code(name, Path(release.__file__).read_bytes(),
                                       (out / 'release-manifest.json').read_bytes().decode('utf-8'), digest)
    plan = json.loads((out / 'execution-plan.json').read_text(encoding='utf-8'))
    assert plan['manifest_sha256'] == digest and plan['notebook_sha256'] == release.sha(out / 'release.ipynb')


@pytest.mark.skipif(os.environ.get('EXP100_REAL_SMOKE') != '1', reason='opt-in: loads the two real ~300 MB checkpoints on CPU')
def test_real_staged_checkpoints_cpu_smoke(monkeypatch):
    """Worker path after the platform gate, on CPU: frozen load_checkpoint -> checkpoint_check -> wrapper -> predict."""
    torch = pytest.importorskip('torch')
    rt = frozen_runtime(monkeypatch)
    from raptor_common import load_checkpoint
    torch.set_num_threads(4)
    arms = {}
    for name in ('student', 'control'):
        path = ROOT / '.local/exp100-release-v4' / name / 'release-manifest.json'
        if not path.is_file():
            pytest.skip('kernel dirs not generated')
        p = release.checked_manifest(path, release.sha(path))
        arms[name] = p['arms'][0]
    members, singles = [], []
    ids = ['9.1']
    for name, arm in arms.items():
        ck = arm['checkpoints'][0]
        model, meta = load_checkpoint(release.PACKAGE / ck['path'])
        r = release.checkpoint_check(meta, arm, ck)
        assert r['k_eval_checkpoint'] == r['k_eval_release'] == 24 and r['tag'] in ('EXP095_STUDENT_B8', 'EXP090_COV_BF16_CTRL')
        members.append(model.eval())
    def go(model):
        corpus = FakeCorpus(ids, res=224)
        probs, timing = rt.predict(model, corpus, ids, 24, 224, 'none', 'cpu', norm='imagenet', batch=1,
                                   log=lambda *a: None, window_policy='within-slot-valid')
        return probs, corpus.decodes
    singles = [go(m)[0] for m in members]
    pe, decodes = go(release.mean_prob_module(members))
    assert decodes == 1 and np.isfinite(pe).all()
    assert np.abs(pe - (singles[0] + singles[1]) / 2).max() <= 1e-6
    assert np.abs(singles[0] - singles[1]).max() > 1e-4
    p1, _ = go(release.mean_prob_module(members[:1]))
    assert np.abs(p1 - singles[0]).max() <= 1e-6


def test_rendered_notebook_lines_fit_kaggle_limit():
    """Kaggle never started a notebook with a 66,897-character source line (2026-10-07); keep every line short."""
    import os as _os
    big = _os.urandom(200_000)
    manifest_text = '{"x": 1}'
    digest = __import__('hashlib').sha256(manifest_text.encode('utf-8')).hexdigest()
    code = release.render_code('student', big, manifest_text, digest)
    assert max(len(line) for line in code.split('\n')) <= release.MAX_NOTEBOOK_LINE
    ns = {}
    src_expr = code.split('source=base64.b85decode(', 1)[1].split('\nassert hashlib.sha256(source)', 1)[0].rstrip()[:-1]
    assert __import__('base64').b85decode(eval(src_expr, ns)) == big
