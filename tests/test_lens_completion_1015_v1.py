"""Tests for tools/lens_completion_1015_v1.py v1.0.1 (synthetic; no label rows, no CUDA).
Changelog 1.0.1: HB162-only CPU memory-breach admission and shared authentication.
"""
import sys
import types

import numpy as np
import pytest

from tools import lens_completion_1015_v1 as L
from tools import multi_lens_readout as ml


def synthetic_context(n=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f'u{i:03d}' for i in range(n)]
    y = (rng.random((n, 12)) < .4).astype(float)
    y[0] = 1; y[1] = 0
    refs = {r: (y, np.ones_like(y, bool), 1) for r in ml.RULERS}
    ctx = ml.Context(ids, refs, [dict(component_id=u, members=[u]) for u in ids])
    return ctx, y


def test_plan_counts_and_unique_names():
    for hb162 in (False, True):
        for phase in ('mil-calibration', 'mil-full', 'release-calibration', 'release-full'):
            tasks = L.plan(phase, hb162)
            names = [L.task_name(t) for t in tasks]
            assert len(names) == len(set(names))
    assert len(L.plan('mil-calibration', True)) == 8          # 4 members x (gold2 + fold2val2)
    assert len(L.plan('mil-full', False)) == 4 and all(t['n'] == 58 for t in L.plan('mil-full', False))
    assert len(L.plan('release-full', False)) == 4             # EXP181 x {320,384} x {gold58,oai557}
    assert len(L.plan('release-full', True)) == 12             # + 4 continuations x 2 panels at 320
    assert len(L.plan('release-calibration', True)) == 6       # + SAM42 only
    assert sum(t['n'] for t in L.plan('release-full', True)) == 2*(58+557) + 4*(58+557)
    assert all(t['n'] == 2 for t in L.plan('release-cpu', False))
    with pytest.raises(ValueError): L.plan('release', False)


def test_image_item_validation_and_seed_mean():
    ids = ['a', 'b', 'c']
    y = np.array([[1, 0, np.nan] + [0]*9, [0, 1, 1] + [1]*9, [1, 1, 0] + [0]*9], float)
    arms = {'42': np.full((3, 12), .2), '43': np.full((3, 12), .4)}
    item = L.image_item(ids, arms, {'42': np.full((3, 12), .5)}, y, True, 'policy')
    assert np.allclose(item['arm'], .3) and item['mask'][0, 2] == False and item['labels'][0, 2] == 0
    assert item['eligible'] is True and item['components'][1] == dict(component_id='b', members=['b'])
    bad = y.copy(); bad[0, 0] = 2
    with pytest.raises(ValueError): L.image_item(ids, arms, arms, bad, False, 'p')
    with pytest.raises(ValueError): L.image_item(ids, {'42': np.full((3, 12), 1.5)}, arms, y, False, 'p')
    with pytest.raises(ValueError): L.image_item(['a', 'a', 'c'], arms, arms, y, False, 'p')


def test_evaluate_seven_lenses_reasons_bundle_and_panels():
    ctx, y = synthetic_context()
    rng = np.random.default_rng(1)
    base = rng.random((len(ctx.ids), 12))
    better = np.clip(base + .3*(y - .5), 0, 1)
    arms, controls = {'42': better}, {'42': base}
    bundle = dict(context=ctx, arm=better, control=base, recipe=dict(fixed_partner='synthetic'))
    panel_ids = [f'p{i}' for i in range(30)]
    py = (rng.random((30, 12)) < .5).astype(float); py[:, 1] = np.nan
    pa = rng.random((30, 12))
    images = {'gold58': L.image_item(panel_ids, {'42': pa}, {'42': pa}, py, False, 'gold'),
              'oai_moaks': L.image_item(panel_ids, {'42': np.clip(pa + .4*(np.nan_to_num(py) - .5), 0, 1)},
                                        {'42': pa}, py, True, 'oai')}
    images['gold58']['reason'] = 'exposed'
    result = L.evaluate(ctx, arms, controls, bundle, images, {'L5': 'single seed'})
    ls = result['lenses']
    assert set(ls) == {f'L{i}' for i in range(1, 8)}
    assert ls['L5']['eligible'] is False and ls['L5']['reason'] == 'single seed'
    assert ls['L6']['eligible'] is True and ls['L6']['value'] > 0
    assert ls['L7']['eligible'] is True and ls['L7']['entries']['gold58']['ineligible_reason'] == 'exposed'
    assert 'mapped_nine_macro' in ls['L7']['entries']['oai_moaks']
    table = L.lens_table(result)
    assert all(f'| L{i} |' in table for i in range(1, 8))
    # Without images the L7 reason is the supplied written reason.
    plain = L.evaluate(ctx, arms, controls, None, None, {'L7': 'pending GPU'})
    assert plain['lenses']['L7']['reason'] == 'pending GPU' and plain['lenses']['L6']['eligible'] is False


def test_coatnet_bundle_is_equal_own_partner_rank(monkeypatch):
    ctx, _ = synthetic_context(20)
    q = np.random.default_rng(3).random((20, 12))
    monkeypatch.setattr(L, 'q7s_slice', lambda fold, ids: (q, dict(bindings={'x': 1})))
    a = np.random.default_rng(4).random((20, 12)); c = np.random.default_rng(5).random((20, 12))
    bundle, auth = L.coatnet_bundle(ctx, ctx.ids, a, c)
    assert np.allclose(bundle['arm'], (ml.rank(q) + ml.rank(a))/2)
    assert np.allclose(bundle['control'], (ml.rank(q) + ml.rank(c))/2)
    assert bundle['recipe']['fold'] == 0 and auth['bindings'] == {'x': 1}


def fake_aligned(sizes=(3, 2, 4, 2, 2)):
    raw = {f: {k: np.full((n, 12), .1*(f+1), np.float32) for k in ('R42', 'R43', 'C4', 'M42', 'M43', 'S42', 'S43')}
           for f, n in enumerate(sizes)}
    rosters = [dict(val=[f'f{f}-{i}' for i in range(n)]) for f, n in enumerate(sizes)]
    module = types.ModuleType('tools.hb105_aligned_oof_v1')
    module.authenticate = lambda: ({f: dict(v) for f, v in raw.items()}, rosters, dict(bindings={}))
    def variants(value):
        s = np.concatenate([(value[f]['S42'] + value[f]['S43'])/2 for f in range(5)])
        other = np.concatenate([value[f]['R42'] for f in range(5)])
        return {'replace': s + other, 'add': s - other, 'Q7': other}
    module.variants = variants
    return module, raw, rosters


def test_aligned_bundle_replaces_only_the_fold_leaves(monkeypatch):
    module, raw, rosters = fake_aligned()
    monkeypatch.setitem(sys.modules, 'tools.hb105_aligned_oof_v1', module)
    import tools
    monkeypatch.setattr(tools, 'hb105_aligned_oof_v1', module, raising=False)
    ids = rosters[2]['val']
    arms = {'42': np.full((4, 12), .9), '43': np.full((4, 12), .7)}
    controls = {'42': np.full((4, 12), .3), '43': np.full((4, 12), .5)}
    (qa, qc), _ = L.aligned_bundle(2, ids, arms, controls, 'replace')
    assert qa.shape == (4, 12) and np.allclose(qa, .8 + .3) and np.allclose(qc, .4 + .3)
    with pytest.raises(ValueError, match='roster'):
        L.aligned_bundle(2, list(reversed(ids)), arms, controls, 'replace')
    with pytest.raises(ValueError, match='shape'):
        L.aligned_bundle(2, ids, {'42': np.zeros((3, 12)), '43': np.zeros((3, 12))}, controls, 'replace')


def test_worker_rejects_plan_drift(tmp_path):
    L.io.atomic_json(tmp_path/'plan.json', dict(phase='mil-full', tasks=L.plan('mil-full', False)[:1]))
    with pytest.raises(ValueError, match='plan differs'):
        L.worker('mil-full', 'cpu', tmp_path)


def test_status_ticks_are_monotonic_and_nonfatal(tmp_path):
    status = L.Status(tmp_path, 3)
    status.completed('validation', {'batch': 1}); status.tick('x')
    assert L.io.read_json(tmp_path/'status.json')['completed'] == 2
    missing = L.Status(tmp_path/'missing'/'deeper', 1)
    missing.tick()  # parent created by atomic_json or failure swallowed; never raises


def test_hb162_outputs_never_read_while_not_successful(monkeypatch):
    calls = []
    monkeypatch.setattr(L, 'hb162_terminal_state', lambda: 'running')
    import tools.hb162_sam_f0_v1 as sam
    monkeypatch.setattr(sam, 'completed', lambda phase: calls.append(phase))
    output, reason = L.hb162_outputs()
    assert output is None and 'running' in reason and calls == []
    with pytest.raises(ValueError, match='not finished'):
        L.readout_hb162()


def test_continuation_provenance_is_enforced(monkeypatch, tmp_path):
    import torch
    import tools.hb158_scale_contrast_v1 as s
    import tools.hb162_sam_f0_v1 as sam
    import tools.oai_moaks_panel as p
    ck = tmp_path/'model_best.pt'
    torch.save(dict(model={}, registration_sha256='wrong', parent_sha256=s.MODEL_SHA), ck)
    monkeypatch.setattr(L, 'ROOT', tmp_path)
    monkeypatch.setattr(L, 'gold_panel', lambda: (['g1', 'g2'], np.zeros((2, 12))))
    monkeypatch.setattr(L, 'oai_panel', lambda: (['o1', 'o2'], np.zeros((2, 12))))
    monkeypatch.setattr(p, 'Cache', lambda kind: None)
    pin = L.sha(ck)
    monkeypatch.setattr(L, 'release_models', lambda output=None: {'SAM42': dict(
        key='SAM42', checkpoint='model_best.pt', checkpoint_sha256=pin, views=[320], kind='continuation')})
    monkeypatch.setattr(s, 'serving', lambda res: ({}, {}))
    monkeypatch.setattr(s, 'make_model', lambda ns, device: torch.nn.Linear(1, 1))
    monkeypatch.setattr(sam, 'WORK', tmp_path)
    (tmp_path/'registration.json').write_text('{}')
    task = dict(family='release', key='SAM42', view=320, panel='gold58', n=2)
    with pytest.raises(ValueError, match='provenance'):
        L.run_release([task], 'cpu', tmp_path, L.Status(tmp_path, 2))
    ck.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        L.run_release([task], 'cpu', tmp_path, L.Status(tmp_path, 2))


def test_policy_and_sources_declare_exposure_and_hb155_oai_reason():
    assert L.POLICY['clean_oof'] is False and L.POLICY['exposure'] == 'EXPOSED'
    assert 'INELIGIBLE' in L.POLICY['hb155_oai'] and L.POLICY['submission_authorized'] is False
    assert 'tools/lens_completion_1015_v1.py' in L.SOURCES and 'tests/test_lens_completion_1015_v1.py' in L.SOURCES
    assert L.PANEL_TARGETS == (0, 2, 3, 4, 5, 6, 7, 8, 9)


@pytest.mark.parametrize('state',['running','unknown','orphaned','stale','absent'])
def test_breach_flag_never_reads_live_hb162(monkeypatch,state):
    import tools.hb162_sam_f0_v1 as sam
    monkeypatch.setattr(L,'hb162_terminal_state',lambda:state)
    monkeypatch.setattr(sam,'evaluation_inputs',lambda *a:pytest.fail('read active run'))
    with pytest.raises(ValueError,match='not finished'):
        L.readout_hb162(accept_memory_breach=True)


def test_breach_readout_uses_same_authentication_gate(monkeypatch):
    import tools.hb162_sam_f0_v1 as sam
    monkeypatch.setattr(L,'hb162_terminal_state',lambda:'FAILED')
    calls=[]
    def authenticate(flag):
        calls.append(flag)
        raise ValueError('sentinel: invalid full phase')
    monkeypatch.setattr(sam,'evaluation_inputs',authenticate)
    with pytest.raises(ValueError,match='not finished'):
        L.readout_hb162()
    with pytest.raises(ValueError,match='sentinel'):
        L.readout_hb162(accept_memory_breach=True)
    assert calls == [True]


@pytest.mark.parametrize('args',[
    ['register'],['validate'],['cpu-smoke'],['readout','--item','HB155'],['readout','--item','HB158']])
def test_memory_flag_is_hb162_readout_only(args):
    with pytest.raises(SystemExit) as exc:
        L.main(args+['--accept-memory-breach'])
    assert exc.value.code == 2
