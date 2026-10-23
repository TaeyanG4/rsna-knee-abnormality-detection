"""CPU contracts for EXP169 v2 pre-reading check A (DINO marginal screen), v1.1.0.

Claude opus for B, 2026-10-10. Synthetic data only; no label rows are read.
v1.1.0: flip mode uses the serving aggregation (probability mean, then rank), its own
registration, and the check-B receipt contract; plain mode is unchanged.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import exp115_vlm_eval as ev
from tools import exp169_release_v2 as v2
from tools import exp169_v2_dino_marginal as m

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ formulas
def _toy():
    rng = np.random.default_rng(0)
    R, C, D = (rng.random((5, 12)) for _ in range(3))
    R[:, 0] = [0.1, 0.4, 0.2, 0.9, 0.5]
    C[:, 0] = [0.3, 0.1, 0.2, 0.5, 0.4]
    D[:, 0] = [0.5, 0.4, 0.3, 0.2, 0.1]
    return R, C, D


def test_q_rc_and_q_rcd_formulas_on_hand_worked_toy():
    R, C, D = _toy()
    q_rcd, q_rc, ranked = m.arms(R, C, D)
    # r(R)=[0,.5,.25,1,.75]; r(C)=[.5,0,.25,1,.75]; r(D)=[1,.75,.5,.25,0]
    np.testing.assert_allclose(ranked['l2'][:, 0], [0, .5, .25, 1, .75])
    np.testing.assert_allclose(ranked['convnext'][:, 0], [.5, 0, .25, 1, .75])
    np.testing.assert_allclose(ranked['dino'][:, 0], [1, .75, .5, .25, 0])
    # Q_RC raw=[.25,.25,.25,1,.75] -> average-tie ranks 2,2,2,5,4
    np.testing.assert_allclose(q_rc[:, 0], [.25, .25, .25, 1, .75])
    # Q_RCD raw=[.5,.4167,.3333,.75,.5] -> ranks 3.5,2,1,5,3.5
    np.testing.assert_allclose(q_rcd[:, 0], [.625, .25, 0, 1, .625])


def test_arms_match_generic_formula_and_reject_shape_mismatch():
    from scipy.stats import rankdata
    R, C, D = _toy()

    def r(x):
        return (rankdata(x, axis=0, method='average') - 1) / (len(x) - 1)

    q_rcd, q_rc, _ = m.arms(R, C, D)
    np.testing.assert_allclose(q_rcd, r((r(R) + r(C) + r(D)) / 3))
    np.testing.assert_allclose(q_rc, r((r(R) + r(C)) / 2))
    with pytest.raises(ValueError):
        m.arms(R, C, D[:4])


def test_within_fold_ranks_average_ties_and_seed_mean():
    s42 = np.tile(np.array([[.2], [.2], [.5]]), (1, 12))
    s43 = np.tile(np.array([[.9], [.1], [.5]]), (1, 12))
    out = m.fold_family_ranks(s42, s43, s42, s43)
    np.testing.assert_allclose(out['C'][:, 0], [1.5 / 3, 1.5 / 3, 1.0])
    np.testing.assert_allclose(out['R'][:, 0], [(.5 + 1) / 2, (.5 + 1 / 3) / 2, (1 + 2 / 3) / 2])


# ------------------------------------------------------------------ flip: serving aggregation
def _seeds():
    s42 = np.tile(np.array([[.9], [.2], [.5]]), (1, 12))
    s43 = np.tile(np.array([[.0], [.45], [.3]]), (1, 12))
    s42[:, 1], s43[:, 1] = [.2, .2, .5], [.9, .1, .5]      # a column with tied seed-42 values
    return s42, s43


def test_flip_r_is_probability_mean_then_rank_on_toy():
    s42, s43 = _seeds()
    c, d = s43.copy(), s42.copy()
    out = m.serving_fold_ranks(s42, s43, c, d)
    # col 0: mean probabilities [.45, .325, .4] -> ranks 3, 1, 2 over n=3
    np.testing.assert_array_equal(out['R'][:, 0], np.array([3, 1, 2]) / 3)
    # col 1: mean probabilities [.55, .15, .5] -> ranks 3, 1, 2
    np.testing.assert_array_equal(out['R'][:, 1], np.array([3, 1, 2]) / 3)
    np.testing.assert_array_equal(out['R'], ev.pct_rank((s42 + s43) / 2))
    # C/D keep the plain-mode within-fold rule bit for bit
    plain = m.fold_family_ranks(s42, s43, c, d)
    np.testing.assert_array_equal(out['C'], plain['C'])
    np.testing.assert_array_equal(out['D'], plain['D'])


def test_flip_r_differs_from_seed_rank_mean_where_it_should():
    s42, s43 = _seeds()
    served = m.serving_fold_ranks(s42, s43, s42, s43)['R']
    rank_mean = m.fold_family_ranks(s42, s43, s42, s43)['R']
    # col 0: seed ranks [3,1,2] and [1,3,2] average to a three-way tie
    np.testing.assert_allclose(rank_mean[:, 0], [2 / 3] * 3)
    assert len(set(served[:, 0])) == 3
    # col 1: rank mean orders row 2 first, the probability mean orders row 0 first
    assert np.argmax(rank_mean[:, 1]) == 2 and np.argmax(served[:, 1]) == 0
    # identical seeds: both rules coincide exactly
    same = m.serving_fold_ranks(s42, s42.copy(), s42, s43)['R']
    np.testing.assert_array_equal(same, m.fold_family_ranks(s42, s42.copy(), s42, s43)['R'])


def test_flip_fold_ranks_equal_serving_family_rank_order_exactly():
    from scipy.stats import rankdata
    rng = np.random.default_rng(7)
    s42, s43, c, d = (np.round(rng.random((41, 12)), 2) for _ in range(4))   # ties present
    out = m.serving_fold_ranks(s42, s43, c, d)
    for name, served in (('R', v2.serving_family_rank([s42, s43], 'l2')),
                         ('C', v2.serving_family_rank([c], 'convnext')),
                         ('D', v2.serving_family_rank([d], 'dino'))):
        np.testing.assert_array_equal(rankdata(out[name], axis=0), rankdata(served, axis=0))
        # same ranks, scale rank / n versus (rank - 1) / (n - 1)
        np.testing.assert_allclose(out[name] * 41, served * 40 + 1, rtol=0, atol=1e-9)


def test_flip_fold_ranks_refuse_a_serving_order_mismatch(monkeypatch):
    s42, s43 = _seeds()
    real = v2.serving_family_rank
    monkeypatch.setattr(v2, 'serving_family_rank',
                        lambda members, family: 1 - real(members, family))
    with pytest.raises(ValueError, match='serving_family_rank'):
        m.serving_fold_ranks(s42, s43, s42, s43)


def test_flip_pooled_arms_follow_serving_then_global_rerank():
    from scipy.stats import rankdata

    def r(x):
        return (rankdata(x, axis=0, method='average') - 1) / (len(x) - 1)

    rng = np.random.default_rng(11)
    folds = [tuple(rng.random((n, 12)) for _ in range(4)) for n in (9, 8)]
    parts = [m.serving_fold_ranks(*f) for f in folds]
    R = np.concatenate([p['R'] for p in parts])
    C = np.concatenate([p['C'] for p in parts])
    D = np.concatenate([p['D'] for p in parts])
    q_rcd, q_rc, ranked = m.arms(R, C, D)
    expect_r = np.concatenate([ev.pct_rank((a + b) / 2) for a, b, _, _ in folds])
    np.testing.assert_array_equal(R, expect_r)
    np.testing.assert_allclose(q_rcd, r((r(R) + r(C) + r(D)) / 3))
    np.testing.assert_allclose(q_rc, r((r(R) + r(C)) / 2))


# ------------------------------------------------------------------ draws
def _refs(n=60, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 12)) < .4).astype(float)
    y[:2], y[2:4] = 1, 0
    ones = np.ones((n, 12), bool)
    return {'jev': (y, ones, 1), 'aquila': (y, ones, 1), 'dread': (y, ones, 1),
            'consensus': (y, ones, 2)}


class _Spy:
    def __init__(self):
        self.calls = []

    def auc_targets(self, y, s, mask, minimum):
        self.calls.append(hashlib.sha256(np.ascontiguousarray(y).tobytes() +
                                         np.ascontiguousarray(mask).tobytes()).hexdigest())
        return ev.auc_targets(y, s, mask, minimum)

    macro = staticmethod(ev.macro)


def test_both_arms_use_identical_draw_indices():
    refs = _refs()
    rng = np.random.default_rng(2)
    a, b = rng.random((60, 12)), rng.random((60, 12))
    groups = [np.asarray([i, i + 1]) for i in range(0, 60, 2)]
    draws = m.make_draws(groups)
    assert len(draws) == m.DRAWS == 300 and m.SEED == 20261003
    assert m.draw_digest(draws) == m.draw_digest(m.make_draws(groups))
    assert m.draw_digest(draws) != m.draw_digest(ev.bootstrap_index(groups, 300, 1))
    spy = _Spy()
    m.compare(a, b, refs, draws, spy)
    per_ref = 2 + 2 * len(draws)
    assert len(spy.calls) == 4 * per_ref
    for k in range(4):
        calls = spy.calls[k * per_ref:(k + 1) * per_ref]
        # candidate/baseline evaluated on the same rows for every draw
        assert all(calls[i] == calls[i + 1] for i in range(0, per_ref, 2))


def test_identical_arms_give_zero_paired_delta_and_ci():
    refs = _refs()
    a = np.random.default_rng(3).random((60, 12))
    draws = m.make_draws([np.asarray([i]) for i in range(60)])
    pair, status = m.compare(a, a.copy(), refs, draws, ev)
    assert pair['primary_mean_delta'] == 0 and pair['primary_mean_ci95'] == [0.0, 0.0]
    assert status['status'] == 'HOLD'


# ------------------------------------------------------------------ gate
def _pair(mean, lower, dread):
    return {'primary_mean_delta': mean, 'primary_mean_ci95': [lower, lower + .01],
            'references': {'dread': {'delta': dread}}}


@pytest.mark.parametrize('mean,lower,dread,expected', [
    (.001, .0001, 0.0, 'ADMIT'),
    (.001, .0001, -.003, 'ADMIT'),       # floor is inclusive
    (.001, .0001, -.0031, 'HOLD'),
    (.001, .0001, None, 'HOLD'),
    (0.0, .0001, 0.0, 'HOLD'),           # strict mean > 0
    (.001, 0.0, 0.0, 'HOLD'),            # strict CI lower > 0
    (.001, -.0001, .01, 'HOLD'),
    (-.001, .0001, .01, 'HOLD'),
])
def test_gate_edges(mean, lower, dread, expected):
    from tools import exp172_dino_base_l2 as e172
    assert e172.admission(_pair(mean, lower, dread))['status'] == expected


def _ruler_root(tmp_path, tiers):
    path = tmp_path / m.RULER_MAP
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'recommendations': {
        t: {'validity': tiers.get(t, 'MEDIUM'), 'references': ['jev_hybrid']}
        for t in m.TARGETS}}), encoding='utf-8')
    return tmp_path


def _ruler_pair(negative):
    per = {t: (-.01 if t in negative else .01) for t in m.TARGETS}
    return {'references': {r: {'per_target_delta': dict(per)} for r in ('jev', 'consensus', 'dread')}}


def test_ruler_review_only_for_losses_confined_to_low_targets(tmp_path):
    root = _ruler_root(tmp_path, {'Synovitis': 'LOW', 'Contusion': 'LOW'})
    low = m.ruler_diagnostics(_ruler_pair({'Synovitis', 'Contusion'}), 'HOLD', root)
    assert low['status'] == 'CHECKED' and low['flag'] == 'REVIEW'
    assert low['non_low_mean_target_delta']['jev'] == pytest.approx(.01)
    mixed = m.ruler_diagnostics(_ruler_pair({'Synovitis', 'ACL'}), 'HOLD', root)
    assert mixed['flag'] == 'NONE'
    admitted = m.ruler_diagnostics(_ruler_pair({'Synovitis'}), 'ADMIT', root)
    assert admitted['flag'] == 'NONE'
    none_negative = m.ruler_diagnostics(_ruler_pair(set()), 'HOLD', root)
    assert none_negative['flag'] == 'NONE'


def test_ruler_missing_map_is_not_checked(tmp_path):
    assert m.ruler_diagnostics(_ruler_pair(set()), 'HOLD', tmp_path)['status'] == 'NOT-CHECKED'


# ------------------------------------------------------------------ flip refusal
def _receipt(folder, tag, seed, fold, inventory, override=None):
    """Check-B (tools/exp169_v2_l2_flip_oof.py produce) receipt shape, synthetic values."""
    receipt = {**m.FLIP_RECEIPT_CONTRACT, 'tag': tag, 'seed': seed, 'fold': fold,
               'plain_gate': {'max_abs_diff': .0001, 'tolerance': m.PLAIN_CEILING, 'passed': True},
               'outputs': {'files': inventory, 'rows': 3, 'ids_sha256': 'roster',
                           'inventory_sha256': m.sha(folder / m.FLIP_INVENTORY)}}
    receipt.update(override or {})
    (folder / m.FLIP_RECEIPT).write_text(json.dumps(receipt), encoding='utf-8')


def _write_flip(root, tag, seed, fold, inventory=True, extra=False, override=None):
    folder = root / m.FLIP_ROOT / tag
    folder.mkdir(parents=True)
    entries = {}
    for name in (m.FLIP_FILE, m.FLIP_PLAIN):
        (folder / name).write_text('x\n', encoding='utf-8')
        entries[name] = {'sha256': m.sha(folder / name), 'bytes': (folder / name).stat().st_size}
    if extra:
        entries['other.csv'] = dict(entries[m.FLIP_FILE])
    if inventory:
        (folder / m.FLIP_INVENTORY).write_text(json.dumps(entries), encoding='utf-8')
        _receipt(folder, tag, seed, fold, entries, override)
    return folder


def test_flip_tags_are_the_ten_registered_l2_checkpoints(tmp_path):
    tags = m.l2_tags(tmp_path)
    assert sorted(tags.values()) == sorted(
        ['EXP149_L2_S42_F0', 'EXP149_L2_S42_F1', 'EXP157_L2_S42_F2', 'EXP157_L2_S42_F3',
         'EXP157_L2_S42_F4', *[f'EXP157_L2_S43_F{f}' for f in range(5)]])


def test_flip_mode_refuses_missing_or_unverified_inputs(tmp_path):
    import shutil
    with pytest.raises(ValueError, match='10 missing'):
        m.flip_r_inputs(tmp_path)
    tags = sorted(m.l2_tags(tmp_path).items())
    for (seed, fold), tag in tags[:9]:
        _write_flip(tmp_path, tag, seed, fold)
    with pytest.raises(ValueError, match='1 missing'):
        m.flip_r_inputs(tmp_path)
    (seed, fold), tag = tags[9]
    bad_receipts = [{'status': 'FAIL'}, {'status': 'PASS'},           # check B writes ADMITTED
                    {'schema': 'other'}, {'job_id': 'other'}, {'seed': 99 - seed},
                    {'fold': fold + 1}, {'tag': 'EXP000'},
                    {'plain_gate': {'max_abs_diff': .003, 'tolerance': .002, 'passed': True}},
                    {'plain_gate': {'max_abs_diff': .0001, 'tolerance': .002, 'passed': False}},
                    {'plain_gate': {'max_abs_diff': .0001, 'tolerance': .01, 'passed': True}}]
    for override in bad_receipts:
        bad = _write_flip(tmp_path, tag, seed, fold, override=override)
        with pytest.raises(ValueError, match='1 invalid'):
            m.flip_r_inputs(tmp_path)
        shutil.rmtree(bad)
    bad = _write_flip(tmp_path, tag, seed, fold, extra=True)          # inventory not exactly two
    with pytest.raises(ValueError, match='1 invalid'):
        m.flip_r_inputs(tmp_path)
    shutil.rmtree(bad)
    bad = _write_flip(tmp_path, tag, seed, fold)
    (bad / m.FLIP_FILE).write_text('tampered\n', encoding='utf-8')
    with pytest.raises(ValueError, match='1 invalid'):
        m.flip_r_inputs(tmp_path)
    (bad / m.FLIP_FILE).write_text('x\n', encoding='utf-8')
    (bad / m.FLIP_INVENTORY).write_text(
        (bad / m.FLIP_INVENTORY).read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='1 invalid'):                 # inventory sha != receipt
        m.flip_r_inputs(tmp_path)
    shutil.rmtree(bad)
    _write_flip(tmp_path, tag, seed, fold)
    found = m.flip_r_inputs(tmp_path)
    assert len(found) == 10 and found[(seed, fold)]['tag'] == tag
    assert found[(seed, fold)]['ids_sha256'] == 'roster' and found[(seed, fold)]['rows'] == 3


def test_flip_run_fails_before_writing_anything(tmp_path):
    reg, out = tmp_path / 'reg.json', tmp_path / 'out'
    with pytest.raises(ValueError, match='flip R inputs not ready'):
        m.run('flip', tmp_path, reg, out)
    assert not reg.exists() and not out.exists()


def test_flip_run_requires_the_prewritten_flip_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'flip_r_inputs', lambda root=None: {})
    reg, out = tmp_path / 'reg.json', tmp_path / 'out'
    with pytest.raises(ValueError, match='pre-written registration'):
        m.run('flip', tmp_path, reg, out)
    assert not reg.exists() and not out.exists()
    assert m.REGISTRATIONS['flip'].endswith('marginal-registration-flip-v1.1.json')
    assert m.REGISTRATIONS['plain'].endswith('marginal-registration.json')


def test_register_only_is_flip_only_and_write_once(tmp_path):
    with pytest.raises(ValueError, match='flip registration only'):
        m.register_only('plain', tmp_path, tmp_path / 'reg.json')
    (tmp_path / 'reg.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='write-once'):
        m.register_only('flip', tmp_path, tmp_path / 'reg.json')
    with pytest.raises(ValueError, match='EXP172'):
        m.register_only('flip', tmp_path, tmp_path / m.e172.EXP / 'reg.json')


def test_run_refuses_writes_inside_exp172(tmp_path):
    with pytest.raises(ValueError, match='EXP172'):
        m.run('plain', tmp_path, tmp_path / 'reg.json', tmp_path / m.EXP172_RESULT)


# ------------------------------------------------------------------ registration
def test_registration_is_write_once_and_frozen(tmp_path):
    payload = {key: {'v': key} for key in m.FROZEN}
    payload.update(tool_sha256='a', created_at='t0')
    first = m.register(tmp_path / 'reg.json', payload)
    assert first['created'] and first['tool_sha256_matches']
    again = m.register(tmp_path / 'reg.json', {**payload, 'created_at': 't1', 'tool_sha256': 'b'})
    assert not again['created'] and not again['tool_sha256_matches']
    assert again['sha256'] == first['sha256']
    changed = {**payload, 'reference_sha256': {'jev': 'different'}}
    with pytest.raises(ValueError, match='reference_sha256'):
        m.register(tmp_path / 'reg.json', changed)


def test_register_without_create_requires_existing_file(tmp_path):
    payload = {key: {'v': key} for key in m.FROZEN_FLIP}
    with pytest.raises(ValueError, match='registration missing'):
        m.register(tmp_path / 'reg.json', payload, m.FROZEN_FLIP, create=False)
    assert not (tmp_path / 'reg.json').exists()


def test_plain_registration_contract_unchanged_since_v1_0_0():
    path = ROOT / m.REGISTRATION
    if not path.is_file():
        pytest.skip('registered plain marginal-registration.json absent')
    frozen = json.loads(path.read_text(encoding='utf-8'))
    contract = m.mode_contract('plain', ROOT)
    assert set(contract) == {'schema', 'formulas', 'flip_contract', 'r_policy'}
    for key, value in contract.items():
        assert frozen[key] == value, key
    assert frozen['gate'] == m.GATE and m.FROZEN_BY_MODE['plain'] == m.FROZEN


def test_flip_registration_is_distinct_and_cannot_reuse_plain_file(tmp_path):
    plain, flip = m.mode_contract('plain', ROOT), m.mode_contract('flip', ROOT)
    assert flip['schema'] != plain['schema'] and flip['r_mode'] == 'flip'
    assert flip['formulas'] != plain['formulas'] and flip['r_policy'] != plain['r_policy']
    assert 'probability mean' in flip['formulas']['family_inputs']
    assert flip['formulas']['candidate'] == plain['formulas']['candidate']
    assert flip['formulas']['baseline'] == plain['formulas']['baseline']
    assert set(flip['serving_code_sha256']) == {f'{p}:{n}' for p, n in m.SERVING_CODE}
    common = {key: {'v': key} for key in m.FROZEN}
    m.register(tmp_path / 'reg.json', {**common, **plain})
    with pytest.raises(ValueError, match='registration mismatch on schema'):
        m.register(tmp_path / 'reg.json', {**common, **flip}, m.FROZEN_FLIP, create=False)


# ------------------------------------------------------------------ check-B layout contract
def _check_b_source():
    path = ROOT / 'tools/exp169_v2_l2_flip_oof.py'
    if not path.is_file():
        pytest.skip('check-B tool absent')
    return ast.parse(path.read_text(encoding='utf-8'))


def _constants(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t for t in node.targets if isinstance(t, (ast.Name, ast.Tuple))]
            for target in names:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError):
                        pass
                elif isinstance(node.value, ast.Tuple):
                    for t, v in zip(target.elts, node.value.elts):
                        if isinstance(t, ast.Name):
                            try:
                                out[t.id] = ast.literal_eval(v)
                            except (ValueError, TypeError, SyntaxError):
                                pass
    return out


def test_check_b_writer_layout_matches_flip_reader():
    tree = _check_b_source()
    c = _constants(tree)
    assert c['OUT_REL'] == m.FLIP_ROOT
    assert (c['PLAIN'], c['TTA']) == (m.FLIP_PLAIN, m.FLIP_FILE)
    assert (c['INVENTORY'], c['RECEIPT']) == (m.FLIP_INVENTORY, m.FLIP_RECEIPT)
    assert c['SCHEMA'] == m.FLIP_RECEIPT_CONTRACT['schema']
    assert c['JOB_ID'] == m.FLIP_RECEIPT_CONTRACT['job_id']
    assert c['PLAIN_TOL'] == m.PLAIN_CEILING
    produce = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'produce')
    calls = {}
    for node in ast.walk(produce):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and \
                getattr(node.value.func, 'id', None) == 'dict' and \
                isinstance(node.targets[0], ast.Name):
            calls[node.targets[0].id] = node.value
    receipt = {k.arg: k.value for k in calls['receipt'].keywords}
    assert {'schema', 'job_id', 'status', 'tag', 'seed', 'fold', 'plain_gate', 'outputs'} <= set(receipt)
    assert ast.literal_eval(receipt['status']) == m.FLIP_RECEIPT_CONTRACT['status']
    outputs = {k.arg for k in receipt['outputs'].keywords}
    assert {'files', 'inventory_sha256', 'rows', 'ids_sha256'} <= outputs
    gate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plain_gate')
    record = next(node.value for node in ast.walk(gate) if isinstance(node, ast.Assign)
                  and getattr(node.targets[0], 'id', None) == 'record')
    assert {'max_abs_diff', 'tolerance', 'passed'} <= {k.arg for k in record.keywords}
