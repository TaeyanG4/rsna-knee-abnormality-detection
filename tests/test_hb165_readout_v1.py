"""CPU tests v1.0.0 for the registered H-B-165 OA-only routing readout (synthetic data; no label rows)."""
import json
import numpy as np
import pytest

from tools import hb165_readout_v1 as r
from tools import hb165_build_v1 as b

D = b.GATE['draws']


def tensor(medial=.004, lateral=.004, draws=None):
    d = np.zeros((D+1, 4, 12))
    d[:, :, r.OA[0]] = medial; d[:, :, r.OA[1]] = lateral
    if draws is not None:
        d[1:] = draws
    return d


def seeds(value=.004):
    out = {}
    for s in r.SEEDS:
        p = np.zeros((4, 12)); p[:, list(r.OA)] = value; out[s] = p
    return out


def test_route_replaces_exactly_the_two_oa_columns():
    rng = np.random.default_rng(0); c = rng.random((7, 12)); x = rng.random((7, 12))
    routed = r.route(c, x)
    assert np.array_equal(routed[:, list(r.OA)], x[:, list(r.OA)])
    assert np.array_equal(routed[:, list(r.NON_OA)], c[:, list(r.NON_OA)])
    assert [r.TARGETS[i] for i in r.OA] == ['Medial OA', 'Lateral OA']


def test_registered_pass():
    g = r.gate(tensor(), seeds(), True)
    assert g['status'] == 'PASS' and g['PASS'] is True and all(g['conditions'].values())
    assert abs(g['routed_macro_mean'] - .008/12) < 1e-15 and g['valid_draws'] == D


def test_exact_boundaries_pass_with_registered_tolerance():
    g = r.gate(tensor(.003, .003), seeds(), True)  # macro exactly .0005; each OA exactly .003 on 4/4
    assert g['status'] == 'PASS'


@pytest.mark.parametrize('case', ['macro', 'ci', 'target', 'seed'])
def test_each_registered_condition_blocks(case):
    d, s = tensor(), seeds()
    if case == 'macro':
        d = tensor(.003, .003); d[:, 3, list(r.OA)] = -.01   # both OA >= .003 on 3/4 rulers, 4-ruler macro < 0
    elif case == 'ci':
        d[1:D//2, :, list(r.OA)] = -.01   # half the draws negative -> 2.5th percentile below zero
    elif case == 'target':
        d[:, 2:, r.OA[1]] = .0029; d[:, :2, r.OA[1]] = .02  # Lateral OA >= .003 on only 2/4 rulers
    else:
        s['45'][:, list(r.OA)] = -.0001
    g = r.gate(d, s, True)
    assert g['status'] == 'FAIL' and g['PASS'] is False
    if case == 'macro':
        assert g['conditions']['each_oa_target_ge_min_on_3_of_4'] is True
    key = dict(macro='routed_macro_mean_ge_min', ci='ci_lower_gt_zero', target='each_oa_target_ge_min_on_3_of_4',
               seed='each_seed_mean_oa_gt_zero')[case]
    assert g['conditions'][key] is False


@pytest.mark.parametrize('case,reason', [('identity', 'non-OA predictions'), ('nonzero', 'not exactly zero'),
    ('draw', 'nonfinite bootstrap'), ('point', 'nonfinite point'), ('shape', 'malformed'), ('seedset', 'malformed')])
def test_invalid_evidence_holds(case, reason):
    d, s, ident = tensor(), seeds(), True
    if case == 'identity':
        ident = False
    elif case == 'nonzero':
        d[0, 1, 0] = 1e-9
    elif case == 'draw':
        d[5, :, r.OA[0]] = np.nan
    elif case == 'point':
        d[0, 0, r.OA[0]] = np.nan
    elif case == 'shape':
        d = d[:300]
    else:
        s = {'44': s['44']}
    g = r.gate(d, s, ident)
    assert g['status'] == 'HOLD' and g['PASS'] is False and reason in g['reason']


def test_engineering_never_passes():
    g = r.gate(tensor(), seeds(), True, engineering=True)
    assert g['status'] == 'ENGINEERING_ONLY' and g['PASS'] is False


def synthetic(n=80, seed=1):
    rng = np.random.default_rng(seed)
    ids = [f'u{i:03d}' for i in range(n)]
    y = (rng.random((n, 12)) < .35).astype(float); y[0] = 1; y[1] = 0
    ones = np.ones((n, 12), bool)
    refs = {name: (y, ones, 1) for name in r.RULERS}
    components = [{'component_id': f'c{i//2}', 'members': ids[i:i+2]} for i in range(0, n, 2)]
    return ids, y, refs, components


def test_scorer_point_auc_matches_sklearn_and_draws_are_reproducible():
    from sklearn.metrics import roc_auc_score
    ids, y, refs, comps = synthetic()
    p = np.random.default_rng(2).random((len(ids), 12))
    s1 = r.Scorer(ids, refs, comps, 50, 7); s2 = r.Scorer(ids, refs, comps, 50, 7)
    point = s1.score(p, point_only=True)[0]
    for j in range(12):
        assert abs(point[0, j] - roc_auc_score(y[:, j], p[:, j])) < 1e-12
    assert np.array_equal(s1.score(p), s2.score(p)) and s1.score(p).shape == (51, 4, 12)


def test_end_to_end_synthetic_lenses_and_identity():
    ids, y, refs, comps = synthetic()
    ctx = r.lens._base.Context(ids, refs, comps); ctx.bindings = {}
    scorer = r.Scorer(ids, refs, comps, D, b.GATE['bootstrap_seed'])
    rng = np.random.default_rng(3)
    controls = {s: np.clip(y*.3 + rng.random(y.shape)*.7, 0, 1) for s in r.SEEDS}
    crops = {s: np.clip(y*.5 + rng.random(y.shape)*.5, 0, 1) for s in r.SEEDS}
    partners = {f: rng.random(y.shape) for f in r.PARTNER_SEEDS}
    out = r.evaluate_all(ctx, scorer, controls, crops, partners=partners)
    assert out['non_oa_identical'] is True and out['registered_gate']['status'] in ('PASS', 'FAIL')
    for name in ('routed', 'plain'):
        ls = out['lens_summary'][name]['lenses']
        assert set(ls) == {'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'}
        assert ls['L5']['eligible'] is True and ls['L6']['eligible'] is True
        assert ls['L7']['eligible'] is False and 'OAI MOAKS' in ls['L7']['reason'] and 'missing' in ls['L7']['reason']
    rd = out['routed_descriptive']['targets']
    assert all(rd[t]['mean_delta_over_rulers'] == 0 for t in r.TARGETS if t not in ('Medial OA', 'Lateral OA'))
    json.dumps(out, allow_nan=False)
    no_partner = r.evaluate_all(ctx, scorer, controls, crops, partners=None, partners_reason='missing partner file')
    assert no_partner['lens_summary']['routed']['lenses']['L6']['reason'] == 'missing partner file'


def records(ids):
    c = dict(fold=3, arm='crop', seed=44, train=['train'], val=ids, weights=[1]*12, recipe={'model': {'name': 'm'}},
             corpus_contract='crop', label_path='labels', inputs={'labels': 'L'}, optimizer_steps=20, exclusions_sha256='E',
             side_unknown_policy='metadata-only', estimand=b.COORDINATOR_DECISION['estimand'])
    i = dict(experiment=r.FIT_EXPERIMENT, version='1.0.0', status='EXPOSED_PAIRED_FOLD3', fold=3, arm='crop', seed=44,
             resolution=224, config_sha256='C', registration_sha256='C', train_sha256=b.uid_hash(c['train']),
             val_sha256=b.uid_hash(c['val']), workers=2, epochs=20, smoke=False, engineering_only=False, gpu_name=b.GPU,
             weights=c['weights'], model=c['recipe']['model'], batch=8, microbatch=2, accumulation=4, corpus_contract='crop',
             exclusions_sha256='E', side_unknown_policy='metadata-only', prepare_sha256='P', runtime_snapshot_sha256='R',
             labels_sha256='L', exposure='EXPOSED', clean_oof=False, estimand=c['estimand'],
             runtime=dict(policy='strict', precision_fallback_allowed=False, device='cuda'))
    s = dict(identity=i, passed=True, smoke=False, device='cuda', steps=20, counters=dict(updates=20, skips=0), checkpoint_sha256='K')
    p = dict(identity=i, k=36, tta=True, engineering_only=False, rows=len(ids), uid_sha256=b.uid_hash(ids), csv_sha256='V', checkpoint_sha256='K')
    kw = dict(config_hash='C', csv_hash='V', checkpoint_hash='K', prepare_hash='P', runtime_hash='R')
    return c, s, p, kw


def test_valid_records_and_h155_or_partial_records_rejected():
    ids = ['v1', 'v2']
    c, s, p, kw = records(ids); r.validate_records(c, s, p, ids, **kw)
    for key, value in (('experiment', 'H-B-155 compartment crop paired pilot'), ('fold', 2), ('status', 'EXPOSED_PAIRED_FOLD2')):
        c, s, p, kw = records(ids); s['identity'][key] = value
        with pytest.raises(ValueError, match='foreign'):
            r.validate_records(c, s, p, ids, **kw)
    c, s, p, kw = records(ids); s['steps'] = 19
    with pytest.raises(ValueError, match='partial'):
        r.validate_records(c, s, p, ids, **kw)
    c, s, p, kw = records(ids); p['tta'] = False
    with pytest.raises(ValueError, match='K36'):
        r.validate_records(c, s, p, ids, **kw)
    c, s, p, kw = records(ids)
    with pytest.raises(ValueError, match='roster'):
        r.validate_records(c, s, p, ['v1'], **kw)


def test_partner_manifest_pin(tmp_path, monkeypatch):
    fake = tmp_path/'m.json'; fake.write_text('{"members": []}')
    monkeypatch.setattr(r, 'HB105', fake.name)
    with pytest.raises(ValueError, match='partner manifest changed'):
        r.load_partners(['a'], tmp_path)


@pytest.mark.skipif(not (b.ROOT/r.HB105).is_file() or not (b.WORK/'fold-binding.json').is_file(), reason='HB105/fold3 not local')
def test_real_fold3_partners_load_aligned():
    val = b.roster()[1]
    probs, prov = r.load_partners(val, b.ROOT)
    assert set(probs) == {'R50', 'C4', 'T'} and all(v.shape == (864, 12) for v in probs.values()) and len(prov) == 5


@pytest.mark.skipif(not (b.WORK/'registration.json').is_file(), reason='stage not registered yet')
def test_registration_binds_this_readout():
    reg = r.registration_check()
    assert reg['gate']['draws'] == 2000 and reg['readout']['module'] == 'tools/hb165_readout_v1.py'


def test_structural_zero_for_identical_non_oa_with_undefined_resamples():
    """A rare non-OA target whose AUC is undefined on some resamples in BOTH models must not HOLD the gate."""
    ids, y, refs, comps = synthetic()
    y = y.copy(); y[:, 11] = 0; y[5, 11] = 1          # Fracture: one positive -> undefined on many resamples
    refs = {name: (y, np.ones_like(y, bool), 1) for name in r.RULERS}
    ctx = r.lens._base.Context(ids, refs, comps); ctx.bindings = {}
    scorer = r.Scorer(ids, refs, comps, D, b.GATE['bootstrap_seed'])
    rng = np.random.default_rng(4)
    controls = {s: rng.random(y.shape) for s in r.SEEDS}
    crops = {s: np.clip(controls[s] + .3*y, 0, 1) for s in r.SEEDS}
    out = r.evaluate_all(ctx, scorer, controls, crops)
    g = out['registered_gate']
    assert g['undefined_non_oa_cells']['seed_mean_draws'] > 0
    assert g['status'] in ('PASS', 'FAIL') and g['valid_draws'] == D and out['non_oa_identical'] is True


def test_structural_delta_rejects_differing_non_oa_scores():
    a = np.zeros((3, 4, 12)); c = a.copy(); c[1, 0, 11] = np.nan
    d, same, undefined = r.structural_delta(a, c)
    assert same is False and undefined == 1 and np.all(d[..., list(r.NON_OA)] == 0)
    a[1, 0, 11] = np.nan
    assert r.structural_delta(a, c)[1] is True
