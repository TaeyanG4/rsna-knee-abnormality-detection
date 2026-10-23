"""Synthetic only: no real label rows, reports, predictions or GPU resources."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import evidence_ledger_rulers as R


@pytest.fixture(autouse=True)
def scipy_free_rank_for_synthetic_tests(monkeypatch):
    # scipy is absent from this CPU test environment; pandas average ranks
    # implement the same tie handling used by exp115_vlm_eval._rank.
    monkeypatch.setattr(R.metrics, '_rank', lambda x: pd.Series(x).rank(method='average').to_numpy())


def table(path, a, ids=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(a, columns=R.T)
    frame.insert(0, R.UID, ids or [f's{i}' for i in range(len(a))])
    frame.to_csv(path, index=False)


def fixture_data(n=58):
    gold = np.tile((np.arange(n) % 2)[:, None], (1, 12)).astype(float)
    sources = {k: gold.copy() for k in R.source_specs()}
    # One false negative and positive avoid full-panel COPY_SUSPECT.
    for a in sources.values():
        a[0, :] = 1
        a[1, :] = 0
    return gold, sources


def make_map():
    gold, sources = fixture_data()
    return dict(schema=R.SCHEMA, **R.analyse(gold, sources, R.source_specs()))


def ancestry(**kw):
    return dict(declared=True, training_references=[], teacher_references=[],
                gold_exposed=False, **kw)


def screen_deltas(m, delta=.002):
    return [dict(target=t, reference=r, delta=delta, ci95=[delta-.001, delta+.001],
                 baseline_auc=.8, candidate_auc=.8+delta)
            for t in R.T for r in m['recommendations'][t]['references']]


def test_source_lineage_matches_recorded_direction_and_transitive_inputs():
    # Independent literals from D-B-035 and corrected D-B-083, not a fixture
    # derived from the implementation: Steven/Pilkwang -> flight -> aquila.
    specs = R.source_specs()
    assert set(specs['flight_v4']['ancestors']) == {
        'flight_v4', 'steven_v4', 'pilkwang_v2'}
    assert set(specs['aquila_v1']['ancestors']) == {
        'aquila_v1', 'flight_v4', 'steven_v4', 'pilkwang_v2'}
    assert set(specs['steven_v4']['ancestors']) == {'steven_v4'}
    assert set(specs['pilkwang_v2']['ancestors']) == {'pilkwang_v2'}


@pytest.mark.parametrize('field', ['training_references', 'teacher_references'])
@pytest.mark.parametrize('exposed,ruler', [
    ('steven_v4', 'aquila_v1'),
    ('pilkwang_v2', 'aquila_v1'),
    ('flight_v4', 'aquila_v1'),
    ('aquila_v1', 'aquila_v1'),
    ('aquila_v1', 'flight_v4'),
    ('aquila_v1', 'steven_v4'),
    ('aquila_v1', 'pilkwang_v2'),
])
def test_lineage_exposure_cannot_produce_false_pass(field, exposed, ruler):
    m = make_map()
    # Reproduce the verifier's decisive case: only the exposed ruler qualifies,
    # with positive deltas and CIs on all twelve targets and an original PASS.
    for recommendation in m['recommendations'].values():
        recommendation['references'] = [ruler]
    rows = screen_deltas(m)
    unexposed = R.composite_decision(m, rows, 'PASS', ancestry=ancestry())
    assert unexposed['verdict'] == 'PASS'
    assert unexposed['macro12_delta'] == pytest.approx(.002)
    declaration = ancestry()
    declaration[field] = [exposed]  # no caller-supplied transitive expansion
    out = R.composite_decision(m, rows, 'PASS', ancestry=declaration)
    assert out['verdict'] == 'REVIEW' and out['original_verdict'] == 'PASS'
    assert out['macro12_delta'] is None
    assert set(out['per_target']) == set(R.T)
    for target in out['per_target'].values():
        assert target['recommended'] == [ruler]
        assert target['used'] == [] and target['delta'] is None
        assert not target['complete']
        assert target['omitted'] == [ruler + ': ancestry exposed/undeclared']


@pytest.mark.parametrize('field', ['training_references', 'teacher_references'])
def test_aquila_exposure_preserves_unrelated_ruler_eligibility(field):
    m = make_map()
    unrelated = ['jev_hybrid', 'dread', 'relabel_v1']
    for recommendation in m['recommendations'].values():
        recommendation['references'] = unrelated.copy()
    declaration = ancestry()
    declaration[field] = ['aquila_v1']
    out = R.composite_decision(m, screen_deltas(m), 'PASS', ancestry=declaration)
    assert out['verdict'] == 'PASS'
    assert out['macro12_delta'] == pytest.approx(.002)
    assert all(t['complete'] and not t['omitted'] for t in out['per_target'].values())


def test_loader_aligns_ids_and_never_fills_missing(tmp_path):
    gold, _ = fixture_data(4)
    p = tmp_path/'labels.csv'
    table(p, gold, ['d', 'b', 'a', 'c'])
    ids, a = R.load_table(p, ['a', 'absent', 'd'])
    assert ids == ['a', 'absent', 'd']
    np.testing.assert_array_equal(a[0], gold[2])
    assert np.isnan(a[1]).all()
    np.testing.assert_array_equal(a[2], gold[0])


@pytest.mark.parametrize('kind', ['duplicate', 'bad_number', 'infinite', 'range', 'binary', 'ambiguous'])
def test_loader_rejects_corruption_without_echoing_cells(tmp_path, kind):
    gold, _ = fixture_data(4)
    p = tmp_path/'labels.csv'
    table(p, gold)
    d = pd.read_csv(p, dtype=str)
    if kind == 'duplicate':
        d.loc[1, R.UID] = d.loc[0, R.UID]
    elif kind == 'ambiguous':
        d['ACL_soft'] = d['ACL']
    else:
        d.loc[0, 'ACL'] = {'bad_number': 'SENSITIVE_BAD_CELL', 'infinite': 'inf',
                           'range': '1.1', 'binary': '.3'}[kind]
    d.to_csv(p, index=False)
    with pytest.raises(ValueError) as error:
        R.load_table(p, binary=True)
    assert 'SENSITIVE_BAD_CELL' not in str(error.value)


def test_loader_soft_suffix_and_selective_columns(tmp_path, monkeypatch):
    gold, _ = fixture_data(4)
    p = tmp_path/'labels.csv'
    table(p, gold)
    d = pd.read_csv(p).rename(columns={t: t+'_soft' for t in R.T})
    d['unused_metadata'] = 'not read'
    d.to_csv(p, index=False)
    original = pd.read_csv
    def guarded(*a, **kw):
        assert callable(kw['usecols']) and not kw['usecols']('unused_metadata')
        return original(*a, **kw)
    monkeypatch.setattr(pd, 'read_csv', guarded)
    _, a = R.load_table(p)
    np.testing.assert_equal(a, gold)


def test_duplicate_header_rejected_before_pandas_mangling(tmp_path):
    p = tmp_path/'bad.csv'
    p.write_text(','.join([R.UID, *R.T, 'ACL'])+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate target/UID column'):
        R.load_table(p)


def test_thresholds_match_registered_float32_rounding():
    x = np.tile(np.array([.49, .5, .500000001, .5001, np.nan])[:, None], (1, 12))
    d = R.hard_labels(x, 'rint')[:, 0]
    a = R.hard_labels(x, 'ge0.5')[:, 0]
    np.testing.assert_equal(d, [0, 0, 0, 1, np.nan])
    np.testing.assert_equal(a, [0, 1, 1, 1, np.nan])


def test_exact_consensus_missing_dread_does_not_become_gold3():
    gold, sources = fixture_data()
    sources['dread'][:] = np.nan
    result = R.analyse(gold, sources, R.source_specs())
    for row in result['rows']:
        if row['reference'] in ('dread', 'consensus'):
            assert row['n'] == 0 and row['soft_auc'] is None and row['validity'] == 'LOW'
        if row['reference'] == 'consensus_gold3':
            assert row['n'] == 58


def test_consensus_masks_missing_and_disagreement_before_threshold():
    a = np.tile(np.array([0, 1, 0, 0])[:, None], (1, 12)).astype(float)
    b = a.copy(); b[1] = 0; b[2] = np.nan
    soft, hard, valid, agree = R.consensus([a, b])
    np.testing.assert_equal(valid[:, 0], [True, True, False, True])
    np.testing.assert_equal(agree[:, 0], [True, False, False, True])
    assert np.isnan(soft[1:3]).all() and np.isnan(hard[1:3]).all()


def test_existing_exp156_reference_parity_on_complete_synthetic_data(tmp_path):
    _, sources = fixture_data()
    ids = [f's{i}' for i in range(58)]
    sources['dread'][0, 0] = .5
    sources['aquila_v1'][1, 1] = .5
    for name, s in R.source_specs().items():
        table(tmp_path/s['path'], sources[name], ids)
    original = R.e156.references(ids, tmp_path)
    for name in ('jev_hybrid', 'aquila_v1', 'dread'):
        np.testing.assert_equal(original[name][0], R.hard_labels(sources[name], R.source_specs()[name]['rule']))
    _, hard, _, agree = R.consensus([sources[m] for m in R.e150.SETS])
    np.testing.assert_equal(original['consensus'][1], agree)
    np.testing.assert_equal(original['consensus'][0][agree], hard[agree])


def test_confusion_and_soft_auc_against_pair_enumeration():
    g = np.array([1, 1, 0, 0, 1, 0], float)
    soft = np.array([.5, .8, .5, .1, .2, .9])
    hard = (soft >= .5).astype(float)
    m = R.cell_metrics(g, soft, hard, np.ones(6, bool))
    brute = np.mean([float(p > n)+.5*float(p == n) for p in soft[g == 1] for n in soft[g == 0]])
    assert (m['tp'], m['fn'], m['tn'], m['fp']) == (2, 1, 1, 2)
    assert m['sensitivity'] == pytest.approx(2/3)
    assert m['specificity'] == pytest.approx(1/3)
    assert m['soft_auc'] == pytest.approx(brute)
    assert m['accuracy'] == .5 and m['coverage'] == 1


def test_sparse_consensus_class_support_cannot_get_high_accuracy_tier():
    g = np.array([1]*9 + [0]*49, float)
    mask = np.array([True]*2 + [False]*7 + [True]*49)
    m = R.cell_metrics(g, g, g, mask)
    assert m['accuracy'] == 1 and m['positive_coverage'] == pytest.approx(2/9)
    assert R.validity(m)[0] == 'LOW'
    m = R.cell_metrics(g, g, g, g == 0)
    assert m['soft_auc'] is None and m['sensitivity'] is None
    assert R.validity(m)[0] == 'LOW'


@pytest.mark.parametrize('errors_per_class,expected', [(1, 'HIGH'), (3, 'MEDIUM'), (10, 'LOW')])
def test_validity_tier_changes_with_confusion(errors_per_class, expected):
    g = np.array([0]*29+[1]*29, float)
    h = g.copy()
    h[:errors_per_class] = 1
    h[29:29+errors_per_class] = 0
    m = R.cell_metrics(g, h, h, np.ones(58, bool))
    assert R.validity(m)[0] == expected


def test_coverage_thresholds_use_gold_denominator():
    g = np.tile([0., 1.], 29)
    for retained, expected in [(28, 'LOW'), (29, 'MEDIUM'), (46, 'MEDIUM'), (47, 'HIGH')]:
        mask = np.arange(58) < retained
        m = R.cell_metrics(g, g, g, mask)
        assert R.validity(m)[0] == expected


def test_full_gold_copy_caps_all_targets_but_single_perfect_target_does_not():
    gold, sources = fixture_data()
    sources['jev_hybrid'] = gold.copy()
    sources['aquila_v1'][:, 0] = gold[:, 0]
    m = R.analyse(gold, sources, R.source_specs())
    assert all(r['validity'] == 'LOW' and r['copy_suspect'] for r in m['rows'] if r['reference'] == 'jev_hybrid')
    assert all(not r['copy_suspect'] for r in m['rows'] if r['reference'] == 'aquila_v1')


def test_complete_positive_screen_pass_and_original_verdict_preserved():
    m = make_map()
    result = R.composite_decision(m, screen_deltas(m), 'ADMIT', ancestry=ancestry())
    assert result['verdict'] == 'PASS' and result['original_verdict'] == 'ADMIT'
    assert result['macro12_delta'] == pytest.approx(.002)
    result = R.composite_decision(m, screen_deltas(m), 'FAIL', ancestry=ancestry())
    assert result['verdict'] == 'REVIEW' and result['original_verdict'] == 'FAIL'


@pytest.mark.parametrize('loss', [-.05, -.001])
def test_loss_only_low_cells_is_report_only(loss):
    m = make_map()
    for r in m['rows']:
        if r['target'] == 'Synovitis' and r['reference'] == 'jev_hybrid':
            r['validity'] = 'LOW'
    m['recommendations']['Synovitis'] = dict(validity='LOW', references=[], status='NEEDS_IMAGE_READ_PANEL')
    deltas = screen_deltas(m) + [dict(target='Synovitis', reference='jev_hybrid', delta=loss,
                                     ci95=[loss-.001, loss+.001])]
    out = R.composite_decision(m, deltas, 'PASS', ancestry=ancestry())
    assert out['verdict'] == 'PASS' and len(out['low_validity_losses']) == 1
    assert out['macro12_delta'] is None
    assert out['decision_macro_delta'] == pytest.approx(.002)
    assert out['report_only_targets'] == ['Synovitis']
    assert out['per_target']['Synovitis']['report_only']


def test_registered_two_low_targets_do_not_block_ten_target_decision():
    m = make_map()
    for target in ('Synovitis', 'Contusion'):
        m['recommendations'][target] = dict(validity='LOW', references=[], status='NEEDS_IMAGE_READ_PANEL')
    result = R.composite_decision(m, screen_deltas(m), 'PASS', ancestry=ancestry())
    assert result['verdict'] == 'PASS'
    assert result['decision_macro_delta'] == pytest.approx(.002)
    assert result['macro12_delta'] is None
    assert len(result['decision_targets']) == 10
    assert result['report_only_targets'] == ['Synovitis', 'Contusion']


@pytest.mark.parametrize('kind', ['missing', 'no_ci', 'trained', 'undeclared', 'retrospective', 'gold'])
def test_incomplete_or_exposed_screen_cannot_pass(kind):
    m = make_map(); rows = screen_deltas(m); anc = ancestry(); kwargs = {}
    if kind == 'missing':
        rows.pop()
    elif kind == 'no_ci':
        rows[0].pop('ci95')
    elif kind == 'trained':
        anc['training_references'] = ['steven_v4']
    elif kind == 'undeclared':
        anc['declared'] = False
    elif kind == 'retrospective':
        kwargs['retrospective'] = True
    elif kind == 'gold':
        kwargs['gold_evaluation'] = True
    out = R.composite_decision(m, rows, 'PASS', ancestry=anc, **kwargs)
    assert out['verdict'] == 'REVIEW'
    if kind in ('missing', 'trained', 'undeclared'):
        assert out['macro12_delta'] is None


def test_correlated_loss_is_review_independent_corroboration_can_fail():
    m = make_map(); rows = screen_deltas(m, -.02)
    out = R.composite_decision(m, rows, 'FAIL', ancestry=ancestry())
    assert out['verdict'] == 'REVIEW'  # all current references share report origin
    # Synthetic future independent source; no real panel is admitted by this test.
    m['sources']['aquila_v1']['evidence_group'] = 'synthetic_independent_panel'
    out = R.composite_decision(m, rows, 'FAIL', ancestry=ancestry())
    assert out['verdict'] == 'FAIL'
    rows.pop()
    out = R.composite_decision(m, rows, 'FAIL', ancestry=ancestry())
    assert out['verdict'] == 'REVIEW'  # any missing recommendation prevents a final judgment


def test_independent_complete_harm_precedes_low_loss_elsewhere():
    m = make_map()
    m['sources']['aquila_v1']['evidence_group'] = 'synthetic_independent_panel'
    for r in m['rows']:
        if r['target'] == 'Synovitis' and r['reference'] == 'jev_hybrid':
            r['validity'] = 'LOW'
    m['recommendations']['Synovitis']['references'].remove('jev_hybrid')
    rows = screen_deltas(m, -.02)
    rows.append(dict(target='Synovitis', reference='jev_hybrid', delta=-.001, ci95=[-.002, 0]))
    out = R.composite_decision(m, rows, 'FAIL', ancestry=ancestry())
    assert out['verdict'] == 'FAIL'  # strong independent evidence means losses are not LOW-only


@pytest.mark.parametrize('kind', ['duplicate', 'nan', 'arithmetic', 'interval', 'target'])
def test_bad_screen_rejected(kind):
    m = make_map(); rows = screen_deltas(m)
    if kind == 'duplicate': rows.append(rows[0].copy())
    if kind == 'nan': rows[0]['delta'] = float('nan')
    if kind == 'arithmetic': rows[0]['candidate_auc'] = .95
    if kind == 'interval': rows[0]['ci95'] = [.01, .02]
    if kind == 'target': rows[0]['target'] = 'invented'
    with pytest.raises(ValueError):
        R.composite_decision(m, rows, 'PASS', ancestry=ancestry())


def test_noise_correction_uses_prevalence_not_just_youden():
    m = dict(positives=20, negatives=38, sensitivity=.8, specificity=.9)
    pi = .2; q = .8*pi+.1*(1-pi)
    expected_k = pi*(1-pi)*(.8+.9-1)/(q*(1-q))
    out = R.noise_corrected_delta(.01, m, true_prevalence=pi, assumptions=R.NOISE_ASSUMPTIONS)
    assert out['corrected_delta'] == pytest.approx(.01/expected_k)
    assert out['corrected_delta'] != pytest.approx(.01/.7)
    assert out['status'] == 'DIAGNOSTIC_ONLY'
    with pytest.raises(ValueError):
        R.noise_corrected_delta(.01, m, true_prevalence=pi, assumptions=[])
    with pytest.raises(ValueError):
        R.noise_corrected_delta(.01, m, true_prevalence=pi, assumptions=R.NOISE_ASSUMPTIONS, reference_kind='consensus')
    m['positives'] = 2
    assert R.noise_corrected_delta(.01, m, true_prevalence=pi,
                                  assumptions=R.NOISE_ASSUMPTIONS)['status'] == 'WITHHELD'


@pytest.mark.parametrize('se,sp', [(.5, .5), (.1, .8), (1., 0.), (.55, .55)])
def test_noise_uninformative_or_unstable_channel_withheld(se, sp):
    m = dict(positives=29, negatives=29, sensitivity=se, specificity=sp)
    out = R.noise_corrected_delta(.01, m, true_prevalence=.5, assumptions=R.NOISE_ASSUMPTIONS)
    assert out['status'] == 'WITHHELD' and out['corrected_delta'] is None


def fake_verification(root):
    # A unit-test-only reviewer fixture, never valid against production root.
    for path in R.BINDINGS:
        p = root/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('SYNTHETIC', encoding='utf-8')
    note = root/'fake-verification.json'
    note.write_text(json.dumps(dict(verdict='PASS', reviewer='SYNTHETIC_TEST_ONLY',
        checklist=['synthetic fixture'], not_checked=[], bindings={p: R.sha(root/p) for p in R.BINDINGS})), encoding='utf-8')
    return note


def test_build_map_binding_freshness_aggregate_outputs_and_old_ledger_unchanged(tmp_path):
    note = fake_verification(tmp_path)
    gold, sources = fixture_data()
    table(tmp_path/R.GOLD, gold)
    for name, spec in R.source_specs().items():
        table(tmp_path/spec['path'], sources[name])
    old = tmp_path/'artifacts/evidence_ledger/measurements.csv'
    old.write_bytes(b'old ledger bytes\r\n')
    out = tmp_path/'out'
    result = R.build_map(out, note, tmp_path)
    assert len(result['rows']) == 9*12
    assert old.read_bytes() == b'old ledger bytes\r\n'
    text = (out/'map.json').read_text(encoding='utf-8')
    assert 'StudyInstanceUID' not in text and 'NaN' not in text
    assert result['verification']['reviewer'] == 'SYNTHETIC_TEST_ONLY'
    assert all(p['admitted_targets'] == [] for p in result['external_panels'].values())
    with pytest.raises(ValueError, match='fresh'):
        R.build_map(out, note, tmp_path)
    (tmp_path/R.PROTOCOL).write_text('CHANGED', encoding='utf-8')
    with pytest.raises(ValueError, match='binding'):
        R.build_map(tmp_path/'other', note, tmp_path)


def test_screen_receipt_detects_postreading_and_preserves_ledger(tmp_path, monkeypatch):
    m = make_map(); m['verification'] = {'bindings': {'test': 'synthetic'}}
    mp = tmp_path/'map.json'; mp.write_text(json.dumps(m), encoding='utf-8')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files={'map.json': R.sha(mp)})), encoding='utf-8')
    monkeypatch.setattr(R, 'verification', lambda _: dict(bindings={'test': 'synthetic'}))
    R.ledger.reading('candidate', '1', .9, root=tmp_path)
    before = (tmp_path/'readings.csv').read_bytes()
    screen = dict(key='candidate', eval_set='SYNTHETIC', cohort_sha256='a'*64,
        mask_sha256='b'*64, aggregation='registered', candidate='new', control='old',
        forecast=.002, no_change_forecast=0, retrospective=False, gold_evaluation=False,
        original_verdict='PASS', ancestry=ancestry(), deltas=screen_deltas(m))
    sp = tmp_path/'input.json'; sp.write_text(json.dumps(screen), encoding='utf-8')
    result = R.record_screen(sp, mp, tmp_path/'receipt.json', 'synthetic', tmp_path)
    assert result['retrospective'] and result['decision']['verdict'] == 'REVIEW'
    assert (tmp_path/'readings.csv').read_bytes() == before
    with pytest.raises(ValueError, match='append-only'):
        R.record_screen(sp, mp, tmp_path/'receipt.json', 'synthetic', tmp_path)
    original = R.composite_decision
    def concurrent_edit(*args, **kwargs):
        answer = original(*args, **kwargs)
        sp.write_text('{}', encoding='utf-8')
        return answer
    monkeypatch.setattr(R, 'composite_decision', concurrent_edit)
    with pytest.raises(ValueError, match='changed during recording'):
        R.record_screen(sp, mp, tmp_path/'concurrent.json', 'synthetic', tmp_path)
    assert not (tmp_path/'concurrent.json').exists()


def test_screen_receipt_rejects_source_bound_map_from_before_lineage_fix(tmp_path, monkeypatch):
    m = make_map()
    m['tool_version'] = '1.0.0'
    m['sources']['aquila_v1']['ancestors'] = ['aquila_v1']
    m['sources']['flight_v4']['ancestors'] = [
        'flight_v4', 'steven_v4', 'pilkwang_v2', 'aquila_v1']
    source = 'tools/evidence_ledger_rulers.py'
    # Exact source hash from rulers-verify-out.md (the defective version).
    m['verification'] = dict(bindings={source:
        '23bf5f38749aae8b64b430d50996fab7c842e97330217b1fc698d63325f9fb65'})
    mp = tmp_path/'map.json'
    mp.write_text(json.dumps(m), encoding='utf-8')
    (tmp_path/'manifest.json').write_text(json.dumps(
        dict(files={'map.json': R.sha(mp)})), encoding='utf-8')
    monkeypatch.setattr(R, 'verification', lambda _: dict(
        bindings={source: R.sha(Path(R.__file__))}))
    destination = tmp_path/'receipt.json'
    # A valid map hash cannot rescue obsolete source bindings. Rejection must
    # happen before even reading screen data or writing a decision receipt.
    with pytest.raises(ValueError, match='Map/tool version differs'):
        R.record_screen(tmp_path/'unread-screen.json', mp, destination, 'synthetic', tmp_path)
    assert not destination.exists()
