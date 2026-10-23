"""H119 synthetic paired API tests: no real labels, models, metrics or draws."""
import copy
import hashlib
import json

import numpy as np
import pytest

from tools import multi_lens_readout as old
from tools import multi_lens_readout_l4_v2 as new


def tensor(points=(0., 0., 0., 0.), target='PF OA'):
    d = np.zeros((7, 4, 12))
    d[:, :, new.TARGETS.index(target)] = points
    # Fixed synthetic draws, not a metric/bootstrap computation.
    d[1:4] -= .012
    d[4:] += .008
    return d


class PairedContext:
    """Precomputed synthetic scores keyed by synthetic paired prediction arrays."""
    def __init__(self, deltas):
        self.ids = ['synthetic-a', 'synthetic-b']
        y = np.tile([[0.], [1.]], (1, 12))
        self.refs = {r: (y, np.ones((2, 12), bool), 1) for r in new.RULERS}
        self.scores, self.arms, self.controls, self.calls = {}, {}, {}, []
        for i, d in enumerate(deltas):
            seed = str(43 + i)
            a, c = np.full((2, 12), .6 + .1*i), np.full((2, 12), .1 + .1*i)
            self.arms[seed], self.controls[seed] = a, c
            self.scores[a.tobytes()] = d.copy()
            self.scores[c.tobytes()] = np.zeros_like(d)
        if len(deltas) > 1:
            # Deliberately different L5 delta (AUC of seed-mean predictions).
            a = np.mean(list(self.arms.values()), axis=0)
            c = np.mean(list(self.controls.values()), axis=0)
            self.scores[a.tobytes()] = np.full_like(deltas[0], .009)
            self.scores[c.tobytes()] = np.zeros_like(deltas[0])

    def score(self, p):
        return self.scores[np.asarray(p).tobytes()]

    def delta(self, a, c):
        self.calls.append((a.tobytes(), c.tobytes()))
        return self.score(a) - self.score(c)


def evaluate(deltas, **kwargs):
    ctx = PairedContext(deltas)
    return new.evaluate(ctx, ctx.arms, ctx.controls, **kwargs), ctx


CASES = [
    ((.005, .001, .0002, -.0001), 'PF OA', False, 1),
    ((.0055485, .0013487, .0040126, -.0021809), 'PF OA', False, 2),
    ((.004, .004, .004, .004), 'ACL', True, 4),
    ((.003, .003, .003, -.01), 'PF OA', True, 3),
    ((np.nextafter(.003, 0), .003, .003, -.01), 'PF OA', False, 2),
    ((np.nextafter(.003, 1), .003, .003, -.01), 'PF OA', True, 3),
    ((.001, .003, .003, .003), 'PF OA', True, 3),
]


@pytest.mark.parametrize('points,target,passed,count', CASES)
def test_actual_evaluate_and_lenses_registered_rule(points, target, passed, count):
    d = tensor(points, target)
    original = d.copy()
    out, ctx = evaluate([d])
    direct = new.lenses(d)[0]['L4']
    for l4 in (out['lenses']['L4'], out['per_seed']['43']['L4'], direct):
        assert l4['PASS'] is passed
        assert l4['selected_targets'] == ([target] if passed else [])
        assert l4['targets'][target]['rulers_ge_0_003'] == count
        assert l4['targets'][target]['ruler_point_deltas'] == dict(zip(new.RULERS, points))
    np.testing.assert_array_equal(d, original)
    assert len(ctx.calls) == 1  # no extra metric call for the correction
    assert out['not_release_authorization'] is True


def test_old_false_pass_removed_and_mean_gate_removed():
    false = tensor(CASES[0][0])
    valid = tensor((.001, .003, .003, .003))
    assert old.lenses(false)[0]['L4']['PASS']
    assert not new.lenses(false)[0]['L4']['PASS']
    assert not old.lenses(valid)[0]['L4']['PASS']
    out, _ = evaluate([valid])
    assert out['lenses']['L4']['PASS']
    assert out['lenses']['L4']['targets']['PF OA']['value'] < .003
    assert out['verdict'] == 'ADVANCE'
    assert out['release_scope'] == ['PF OA']  # diagnostic only


def test_seed_deltas_not_substituted_by_aggregate_or_l5():
    first = tensor((.006, .006, .006, 0))
    second = tensor((0, 0, 0, .006))
    out, ctx = evaluate([first, second])
    assert out['lenses']['L4']['targets']['PF OA']['rulers_ge_0_003'] == 4
    assert out['per_seed']['43']['L4']['targets']['PF OA']['rulers_ge_0_003'] == 3
    assert out['per_seed']['44']['L4']['targets']['PF OA']['rulers_ge_0_003'] == 1
    assert out['per_seed']['43']['L4']['PASS']
    assert not out['per_seed']['44']['L4']['PASS']
    assert out['lenses']['L4']['selected_targets'] == ['PF OA']
    assert out['lenses']['L5']['value'] == pytest.approx(.009)
    assert len(ctx.calls) == 3  # two seeds plus inherited seed-mean evaluation


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_unrelated_target_blocks_whole_readout_even_other_lenses(bad):
    d = tensor((.004,) * 4, 'ACL')
    d[0, 3, 11] = bad  # unrelated target/ruler, ACL remains individually valid
    out, _ = evaluate([d])
    l4 = out['lenses']['L4']
    assert l4['targets']['ACL']['PASS']
    assert not l4['targets']['Fracture']['eligible']
    assert l4['correction_status'] == 'NOT_EVALUABLE'
    assert not l4['PASS'] and not l4['eligible'] and l4['selected_targets'] == []
    assert out['verdict'] == 'NOT_EVALUABLE' and out['blocked']
    ls, guard, _ = new.lenses(d)
    ls['L7'] = {'PASS': True, 'eligible': True}
    assert new.verdict(ls, {'PASS': True})['verdict'] == 'NOT_EVALUABLE'
    assert out['per_seed']['43']['L4']['correction_status'] == 'NOT_EVALUABLE'


@pytest.mark.parametrize('bad', [None, [], np.zeros((7, 3, 12)), np.zeros((7, 5, 12)),
    np.zeros((7, 4, 11)), np.zeros((7, 4, 13)), np.zeros((0, 4, 12)),
    np.full((7, 4, 12), '0.004'), np.full((7, 4, 12), True),
    np.full((7, 4, 12), None, dtype=object)])
def test_missing_malformed_schema_rejects_entire_api(bad):
    with pytest.raises(ValueError, match='Expected numeric delta tensor'):
        new.lenses(bad)
    ctx = PairedContext([tensor()])
    ctx.delta = lambda a, c: bad
    with pytest.raises(ValueError, match='Expected numeric delta tensor'):
        new.evaluate(ctx, ctx.arms, ctx.controls)


def assert_telemetry(old_l4, new_l4):
    for before, after in [(old_l4, new_l4)] + [
            (old_l4['targets'][t], new_l4['targets'][t]) for t in new.TARGETS]:
        for key in ('value', 'ci95', 'valid_draws', 'threshold', 'positive_rulers'):
            if key in before:
                assert before[key] == after[key]
        assert 'NOT a point estimate or CI' in after['legacy_telemetry_note']


def test_all_other_lenses_points_cis_masks_ranks_and_inputs_unchanged(monkeypatch):
    ctx = PairedContext([tensor(CASES[1][0]), tensor((.001, .003, .003, .003))])
    # Exercise inherited L6 and L7 using fixed synthetic score telemetry only.
    image = dict(arm=np.ones((2,12)), control=np.zeros((2,12)), labels=np.zeros((2,12)),
                 mask=np.ones((2,12), bool), ids=ctx.ids, components=[], eligible=True)
    image_scores = lambda p, *args: np.tile(p[0], (7, 1)) * .004
    for module in (old, new._base):
        monkeypatch.setattr(module, 'auc_draws', image_scores)
        monkeypatch.setattr(module, 'weights', lambda *args: np.ones((7,2)))
    bundle = dict(context=ctx, arm=ctx.arms['43'], control=ctx.controls['43'],
                  anchor_arm=ctx.arms['44'], anchor_control=ctx.controls['44'], recipe={'fixed': True})
    before_arrays = [a.copy() for a in [*ctx.arms.values(), *ctx.controls.values()]]
    before_refs = copy.deepcopy(ctx.refs)
    args = dict(label_change=True, bundle=bundle, image_reads={'oai_moaks': image, 'gold58': image})
    before = old.evaluate(ctx, ctx.arms, ctx.controls, **args)
    after = new.evaluate(ctx, ctx.arms, ctx.controls, **args)
    for k in ('rows', 'seeds', 'harm_guard', 'rulers', 'warnings'):
        assert after[k] == before[k]
    for scope_before, scope_after in [(before['lenses'], after['lenses'])] + [
            (before['per_seed'][s], after['per_seed'][s]) for s in ctx.arms]:
        for lens in ('L1', 'L2', 'L3', 'L5', 'L6', 'L7'):
            assert scope_before[lens] == scope_after[lens]
        assert_telemetry(scope_before['L4'], scope_after['L4'])
    for original, actual in zip(before_arrays, [*ctx.arms.values(), *ctx.controls.values()]):
        np.testing.assert_array_equal(original, actual)
    for r in new.RULERS:
        for original, actual in zip(before_refs[r][:2], ctx.refs[r][:2]):
            np.testing.assert_array_equal(original, actual)
    np.testing.assert_array_equal(new._base.rank(ctx.arms['43']), old.rank(ctx.arms['43']))
    assert not after['lenses']['L7']['entries']['gold58']['eligible']
    assert 'release' not in after and 'launch_authorized' not in after
    json.dumps(after, allow_nan=False)


def test_missing_bootstrap_draw_is_not_missing_point_evidence():
    d = tensor((.003, .003, .003, 0))
    d[1, 0, 6] = np.nan
    out, _ = evaluate([d])
    assert out['lenses']['L4']['PASS']
    assert_telemetry(old.lenses(d)[0]['L4'], out['lenses']['L4'])


def test_malformed_seed_blocks_aggregate_but_other_seed_remains_individually_valid():
    a, b = tensor((.004,) * 4), tensor((.004,) * 4)
    b[0, 2, 0] = np.nan
    out, _ = evaluate([a, b])
    assert out['per_seed']['43']['L4']['PASS']
    assert not out['per_seed']['44']['L4']['eligible']
    assert out['verdict'] == 'NOT_EVALUABLE'


def test_base_pin_and_canonical_functions_never_rebound(tmp_path, monkeypatch):
    raw = new.BASE_PATH.read_bytes()
    identities = {name: getattr(old, name) for name in ('evaluate', 'lenses', 'verdict', 'auc_draws', 'rank')}
    assert hashlib.sha256(raw).hexdigest() == new.BASE_SHA256
    evaluate([tensor()])
    assert raw == new.BASE_PATH.read_bytes()
    assert all(getattr(old, name) is func for name, func in identities.items())
    tampered = tmp_path / 'base.py'
    tampered.write_bytes(raw + b'\n# source drift\n')
    monkeypatch.setattr(new, 'BASE_PATH', tampered)
    for call in (new.authenticate_base, lambda: new.lenses(tensor()),
                 lambda: evaluate([tensor()]), lambda: new.verdict({}, {})):
        with pytest.raises(RuntimeError, match='Authenticated base source mismatch'):
            call()


def test_paired_seed_mismatch_and_unversioned_verdict_fail_closed():
    ctx = PairedContext([tensor()])
    with pytest.raises(ValueError, match='Paired seed names differ'):
        new.evaluate(ctx, ctx.arms, {'different': next(iter(ctx.controls.values()))})
    ls, guard, _ = old.lenses(tensor((.004,)*4))
    assert new.verdict(ls, guard)['verdict'] == 'NOT_EVALUABLE'
