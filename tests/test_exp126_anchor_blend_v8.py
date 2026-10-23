"""Bounded CPU-only Stage 2 gates, math, shared serving and EXP124 mirror tests."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

e = load('exp126_test', ROOT / 'tools/exp126_anchor_blend_v8.py')
h = load('exp126_v7_fixtures', ROOT / 'tests/test_exp119_release_v7.py')

@pytest.fixture(autouse=True)
def refuse_cuda(monkeypatch):
    torch = pytest.importorskip('torch')
    def refuse(*a, **k): raise RuntimeError('CUDA forbidden')
    monkeypatch.setattr(torch.cuda, '_lazy_init', refuse)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.set_num_threads(2)
    yield
    assert not torch.cuda.is_initialized()

def test_anchor_hard_gate():
    good = dict(ok=True, members=e.CHILDREN, audit_members=4, audit_status='COMPLETE',
                family_reduction='rank_of_member_probability_mean', failure_events=[])
    e.anchor_gate(good)
    for over in (dict(ok=False), dict(members=e.CHILDREN[:3]), dict(audit_members=3),
                 dict(failure_events=[dict(kind='coat_child_failed')])):
        with pytest.raises(ValueError): e.anchor_gate(dict(good, **over))

def test_runtime_corrected_costs_and_margin():
    r = e.projection()
    assert r['own_model_seconds'] == {'rad5': 2112.5}
    assert r['margin_seconds'] == 3283.5
    assert not r['fits_one_hour_margin']
    assert e.projection(anchor_seconds=25500)['fits_one_hour_margin']
    assert e.projection(own='coat5rad5')['own_model_seconds']['coat5'] == 2957.5
    assert e.projection(tta=True)['own_model_seconds']['rad5'] == 4225
    faster = e.projection(rad_k=24)
    assert faster['own_model_seconds']['rad5'] == pytest.approx(1408.3333333333333)
    assert faster['total_seconds'] == pytest.approx(28412.333333333332)
    assert faster['fits_one_hour_margin']

def test_registered_rank_formula():
    old, v7 = e.modules()
    anchor = [[float(i%3)]*12 for i in range(24)]
    own = [[float((23-i)%4)]*12 for i in range(24)]
    got = old.final_blend(v7.v6, anchor, dict(enumerate(own)), e.FINAL_WEIGHTS)
    assert got == v7.v6.rank_mean([anchor, own], [0.7, 0.3])

def test_24_study_shared_proxy_plain_equals_separate(tmp_path, monkeypatch):
    import torch
    rt = h.frozen_runtime(); h.input_modules()
    monkeypatch.setitem(sys.modules, 'raptor_runtime', rt)
    loader = h.make_loader(torch)
    monkeypatch.setattr(__import__('raptor_common'), 'load_checkpoint', loader)
    am_a, am_b = h.two_datasets(tmp_path); arms = h.three_arms(am_a, am_b)[:2]
    roots = h.roots_of(tmp_path, arms); ids = ['1.%d'%i for i in range(24)]
    corpus = h.FakeCorpus(ids)
    shared = h.v7.serve_shared(arms, roots, corpus, ids, 'none', 'cpu', {}, log=h.quiet,
                               runtime_module=e.TTARuntime(rt, False), loader=loader)
    assert corpus.decodes == 24
    for arm in arms:
        separate = h.v6.serve_arm(arm, roots[arm['name']], h.FakeCorpus(ids), ids, 'none', 'cpu', {}, log=h.quiet)[0]
        assert np.array_equal(shared['probs'][arm['name']], separate)

def test_plane_mirror_matches_exact_exp124_functions():
    import torch
    tree = ast.parse((ROOT / 'tools/exp124_flip_tta.py').read_text(encoding='utf-8'))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('plane_ranges', 'flip_mask', 'mirror_windows')]
    namespace = {}; exec(compile(ast.Module(body=funcs, type_ignores=[]), 'exp124-functions', 'exec'), namespace)
    spec = dict(depth=64, slots=[['Sagittal', True, 32], ['Coronal', True, 20], ['Axial', True, 12]])
    centers = [1, 15, 30, 33, 40, 50, 53, 60, 62, -1]
    flags = e.plane_flags(centers, spec)
    assert flags == namespace['flip_mask'](centers, namespace['plane_ranges'](spec))
    x = torch.arange(10*3*8*8).reshape(10,3,8,8)
    expected = namespace['mirror_windows'](x, flags).numpy()
    actual = x.numpy().copy()
    for i, yes in enumerate(flags):
        if yes: actual[i] = actual[i,...,::-1]
    assert np.array_equal(actual, expected)
    for c in (0, 31, 32, 51, 52, 63):
        with pytest.raises(ValueError): e.plane_flags([c], spec)

def test_24_study_tta_proxy_matches_independent_two_pass_reference(tmp_path, monkeypatch):
    import torch
    from raptor_common import select_centers
    rt = h.frozen_runtime(); h.input_modules()
    am_a, am_b = h.two_datasets(tmp_path); arm = h.three_arms(am_a, am_b)[1]
    roots = h.roots_of(tmp_path, [arm]); loader = h.make_loader(torch)
    ids = ['1.%d'%i for i in range(24)]
    model = h.v6.mean_prob_module([loader(Path(roots['rad5'])/ck['path'])[0] for ck in arm['checkpoints']])
    corpus = h.FakeCorpus(ids)
    kw = dict(norm=arm['norm'], batch=1, log=h.quiet, window_policy=arm['window_policy'])
    proxy = e.TTARuntime(rt, True)
    with h.v6.serving_input(proxy, 'center3'):
        actual, receipt = proxy.predict(model, corpus, ids, arm['k_eval'], arm['res'], 'none', 'cpu', **kw)
    with h.v6.serving_input(rt, 'center3'):
        plain, _ = rt.predict(model, h.FakeCorpus(ids), ids, arm['k_eval'], arm['res'], 'none', 'cpu', **kw)
        original = rt.prepare_input
        def mirror(c, uid, k, res, norm, window_policy='legacy'):
            x = original(c, uid, k, res, norm, window_policy)
            flags = e.plane_flags(select_centers(c.mask(uid), c.depth, k, c.spec, policy=window_policy), dict(c.spec, depth=c.depth))
            return np.stack([row[...,::-1] if flag else row for row,flag in zip(x, flags)])
        monkeypatch.setattr(rt, 'prepare_input', mirror)
        flipped, _ = rt.predict(model, h.FakeCorpus(ids), ids, arm['k_eval'], arm['res'], 'none', 'cpu', **kw)
        monkeypatch.setattr(rt, 'prepare_input', original)
    assert np.array_equal(actual, (plain+flipped)/np.float32(2))
    assert receipt['plane_flip_tta'] is True

def test_build_anchor_cells_identical(tmp_path):
    out = tmp_path/'kernel'
    e.build(SimpleNamespace(out=out, own='rad5', plane_flip_tta=False, anchor_seconds=26404))
    notebook = json.loads((out/'release.ipynb').read_text(encoding='utf-8'))
    assert e.sha(e.STAGE1_NB) == e.STAGE1_SHA
    stage1 = json.loads(e.STAGE1_NB.read_text(encoding='utf-8'))
    assert len(notebook['cells']) == len(stage1['cells']) == 34
    assert notebook['cells'][1:-1] == stage1['cells'][1:-1]
    assert notebook['cells'][0] != stage1['cells'][0]
    assert notebook['cells'][-1] != stage1['cells'][-1]
    metadata = json.loads((out/'kernel-metadata.json').read_text(encoding='utf-8'))
    assert metadata['is_private'] is True and metadata['enable_internet'] is False


def test_build_rad_k24_keeps_stage1_path_and_registered_weights(tmp_path):
    out = tmp_path/'kernel-k24'
    e.build(SimpleNamespace(out=out, own='rad5', rad_k=24, plane_flip_tta=False,
                            anchor_seconds=26404))
    conf = json.loads((out/e.CONFIG).read_text(encoding='utf-8'))
    manifest = json.loads((out/e.MANIFEST).read_text(encoding='utf-8'))
    stage1 = json.loads(e.STAGE1_NB.read_text(encoding='utf-8'))
    notebook = json.loads((out/'release.ipynb').read_text(encoding='utf-8'))
    assert conf['final_weights'] == [0.7, 0.3]
    assert conf['rad_k'] == manifest['arms'][0]['k_eval'] == 24
    assert manifest['arms'][0]['model_seconds_per_study'] == pytest.approx(0.25 * 24 / 36)
    assert notebook['cells'][1:-1] == stage1['cells'][1:-1]
    plan = json.loads((out/'execution-plan.json').read_text(encoding='utf-8'))
    assert plan['runtime']['fits_one_hour_margin']

def test_gate_failure_preserves_anchor_bytes(tmp_path):
    (tmp_path/'submission.csv').write_bytes(b'anchor bytes\n')
    r = e.post_anchor({}, {'HB15_WORKING': str(tmp_path)})
    assert r['path'] == 'anchor_only' and r['final_is_anchor']
    assert (tmp_path/'submission.csv').read_bytes() == b'anchor bytes\n'

def test_missing_checkpoint_binding_fails_closed(tmp_path):
    old, v7 = e.modules()
    ids = ['1', '2', '3']
    arm = dict(name='rad5', k_eval=36, checkpoints=[dict(name='r0', sha256='a'*64)])
    release = dict(arms=[arm])
    old.write_csv(v7.v6, tmp_path/'rad5.csv', ids, [[.1,.2,.3]]*12)
    receipt = dict(status='PASS', release_manifest_sha256='b'*64, ids_sha256=old.ids_sha256(ids),
                   studies=3, scored_ids=ids, unscored={}, store=dict(trusted=True), plane_flip_tta=False,
                   arms=dict(rad5=dict(arm=arm, csv='rad5.csv', csv_sha256=e.sha(tmp_path/'rad5.csv'),
                                       checkpoints=[dict(name='r0', sha256='a'*64, k_eval_release=36)])))
    e.write(tmp_path/'receipt.json', receipt)
    own, _ = e.checked_own(release, 'b'*64, tmp_path, ids, 0)
    assert len(own) == 3
    receipt['arms']['rad5']['checkpoints'][0].pop('sha256')
    e.write(tmp_path/'receipt.json', receipt)
    with pytest.raises(ValueError, match='Checkpoint binding'): e.checked_own(release, 'b'*64, tmp_path, ids, 0)
