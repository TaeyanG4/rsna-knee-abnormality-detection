"""Synthetic-only checks for the one-shot saved-OOF rank screen.

No real OOF, labels, model, Gold data, or prepare/run CLI is read or executed.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import pfoa_fixed_rank_screen as screen


def columns(values):
    return np.tile(np.asarray(values, dtype=float)[:, None], (1, 12))


def test_average_ties_exact_fixed_weights_and_identical_rank_control():
    a = columns([.1, .1, .8, .9])
    b = columns([.9, .8, .1, .1])
    ra, rb, blend = screen.fixed_rank_blend(a, b)
    np.testing.assert_array_equal(ra[:, 0], [.375, .375, .75, 1])
    np.testing.assert_array_equal(rb[:, 0], [1, .75, .375, .375])
    np.testing.assert_allclose(blend[:, 0], [.4375, .4125, .7125, .9375], rtol=0, atol=1e-15)
    _, _, identical = screen.fixed_rank_blend(a, a)
    np.testing.assert_allclose(identical, ra, rtol=0, atol=1e-15)
    # An identical candidate must not create an AUC gain despite tied scores.
    y = columns([0, 0, 1, 1])
    result = screen.score_fold(y, a, a)
    assert result['macro_delta'] == 0


def test_columns_and_folds_are_ranked_independently():
    first = columns([.01, .02, .03, .04])
    first[:, 1] = first[::-1, 0]
    second = first + .7
    a0, _, _ = screen.fixed_rank_blend(first, first)
    a1, _, _ = screen.fixed_rank_blend(second, second)
    np.testing.assert_array_equal(a0, a1)
    np.testing.assert_array_equal(a0[:, 0], [.25, .5, .75, 1])
    np.testing.assert_array_equal(a0[:, 1], [1, .75, .5, .25])
    pooled, _, _ = screen.fixed_rank_blend(np.concatenate([first, second]), np.concatenate([first, second]))
    assert not np.array_equal(pooled[:4], a0)


def frame(values=None):
    values = columns([.1, .3, .2, .9]) if values is None else values
    result = pd.DataFrame(values, columns=screen.LAB)
    result.insert(0, screen.UID, ['study_a', 'study_b', 'study_c', 'study_d'])
    return result


def test_uid_alignment_is_invariant_to_csv_row_order():
    value = frame()
    ids = list(value[screen.UID])
    expected = screen.validate_predictions(value, ids)
    reordered = screen.validate_predictions(value.iloc[[2, 0, 3, 1]], ids)
    np.testing.assert_array_equal(reordered, expected)
    baseline = columns([.2, .2, .5, .1])
    np.testing.assert_array_equal(screen.fixed_rank_blend(baseline, reordered)[2],
                                  screen.fixed_rank_blend(baseline, expected)[2])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.001, 1.001])
def test_invalid_probabilities_fail_before_rank(bad):
    value = frame()
    value.loc[0, screen.LAB[0]] = bad
    with pytest.raises(ValueError, match='probabilities'):
        screen.validate_predictions(value, list(value[screen.UID]))
    a = columns([.1, .2, .3, .4]); a[0, 0] = bad
    with pytest.raises(ValueError, match='probabilities'):
        screen.fixed_rank_blend(a, columns([.1, .2, .3, .4]))


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'null_uid', 'unknown', 'target_order', 'target_missing'])
def test_exact_uid_and_target_contract(fault):
    value = frame(); ids = list(value[screen.UID])
    if fault == 'duplicate': value.loc[0, screen.UID] = value.loc[1, screen.UID]
    elif fault == 'missing': value = value.iloc[:-1]
    elif fault == 'null_uid': value.loc[0, screen.UID] = None
    elif fault == 'unknown': value.loc[0, screen.UID] = 'other'
    elif fault == 'target_order': value = value[[screen.UID, *reversed(screen.LAB)]]
    else: value = value.drop(columns=screen.LAB[-1])
    with pytest.raises(ValueError, match='contract'):
        screen.validate_predictions(value, ids)


def primary_fixture():
    gate = dict(both_fold_delta_positive=True, mean_fold_delta_at_least=.002,
                target_drop_review_below=-.01, max_skipped_update_fraction=.01)
    manifest = dict(folds=[0, 1], gate=gate,
                    runs=[dict(fold=f, arm=a, tag=f'run_{a}_{f}') for f in (0, 1) for a in ('A', 'B')])
    folds = [dict(fold=f, delta=.003, per_target_delta={t: .003 for t in screen.LAB},
                  skipped_update_fraction={'A': 0, 'B': 0}) for f in (0, 1)]
    report = dict(manifest_sha256='manifest-sha', folds=folds, applied_gate=gate,
                  **screen.primary.apply_gate(folds, gate))
    diagnostics = dict(manifest_sha256='manifest-sha', primary_report_sha256='primary-sha',
                       primary_gate_unchanged=gate, folds=[dict(fold=0), dict(fold=1)])
    return manifest, report, diagnostics


def validate_fixture(values, arm='A', reviewed=True):
    return screen.validate_primary(*values, 'manifest-sha', 'primary-sha', arm, reviewed)


def test_exact_primary_binding_accepts_valid_four_run_protocol():
    validate_fixture(primary_fixture(), arm='B')


@pytest.mark.parametrize('fault', ['missing_run', 'duplicate_run', 'missing_primary_fold',
    'missing_diagnostic_fold', 'manifest_binding', 'primary_binding', 'gate', 'numeric', 'review'])
def test_incomplete_or_stale_primary_refused(fault):
    values = primary_fixture(); manifest, report, diagnostics = values
    if fault == 'missing_run': manifest['runs'].pop()
    elif fault == 'duplicate_run': manifest['runs'][-1] = deepcopy(manifest['runs'][0])
    elif fault == 'missing_primary_fold': report['folds'].pop()
    elif fault == 'missing_diagnostic_fold': diagnostics['folds'].pop()
    elif fault == 'manifest_binding': report['manifest_sha256'] = 'stale'
    elif fault == 'primary_binding': diagnostics['primary_report_sha256'] = 'stale'
    elif fault == 'gate': diagnostics['primary_gate_unchanged'] = {}
    elif fault == 'numeric': report['folds'][0]['skipped_update_fraction']['A'] = .5
    with pytest.raises(ValueError):
        validate_fixture(values, reviewed=fault != 'review')


def test_b_cannot_be_chosen_when_original_gate_failed():
    values = primary_fixture(); _, report, _ = values
    for fold in report['folds']: fold['delta'] = .001
    report.update(screen.primary.apply_gate(report['folds'], values[0]['gate']))
    validate_fixture(values, arm='A')
    with pytest.raises(ValueError, match='original primary screen'):
        validate_fixture(values, arm='B')


def frozen_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, 'ROOT', tmp_path)
    source = tmp_path/'source.json'; source.write_text('{}', encoding='utf-8')
    data = []
    for fold in (0, 1):
        y = columns([0, 0, 1, 1])
        a = columns([.1, .2, .3, .4])
        data.append(dict(fold=fold, labels=y, arms={'A': a, 'B': a.copy()}, resnet=a.copy(),
                         eval_ids_sha256=f'ids-{fold}'))
    hashes = {screen.uri(source): screen.primary.sha256_file(source)}
    plan = dict(schema=screen.SCHEMA, status='FROZEN_BEFORE_BLEND_SCORING', fixed=deepcopy(screen.FIXED),
                created_at=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat(),
                source_hashes=hashes, inputs={}, selected_arm='A', target_review_complete=True,
                eval_ids_sha256={'0': 'ids-0', '1': 'ids-1'}, scope='synthetic-only')
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(plan), encoding='utf-8')
    report = {'folds': [dict(fold=f, macro_a=1., macro_b=1.) for f in (0, 1)]}
    monkeypatch.setattr(screen, 'collect', lambda *args: (data, hashes, report))
    return plan_path, source, plan, data, hashes, report


@pytest.mark.parametrize('fault', ['plan_hash', 'source_hash', 'future', 'naive_timestamp', 'fixed_ratio',
                                 'output_exists', 'collected_hash', 'collected_ids'])
def test_stale_plan_source_or_output_is_rejected_before_any_metric(tmp_path, monkeypatch, fault):
    plan_path, source, plan, data, hashes, report = frozen_plan(tmp_path, monkeypatch)
    out = tmp_path/'result'
    if fault == 'source_hash': source.write_text('{"changed":true}', encoding='utf-8')
    elif fault == 'future': plan['created_at'] = (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()
    elif fault == 'naive_timestamp': plan['created_at'] = '2026-01-01T00:00:00'
    elif fault == 'fixed_ratio': plan['fixed']['resnet'] = .2
    elif fault == 'output_exists': out.mkdir()
    elif fault == 'collected_hash': hashes['workspace://source.json'] = 'changed-after-precheck'
    elif fault == 'collected_ids': data[1]['eval_ids_sha256'] = 'wrong-fold-ids'
    # Keep the frozen JSON hashes intact when mutating the mocked collected return value.
    if fault not in ('collected_hash', 'collected_ids'):
        plan_path.write_text(json.dumps(plan), encoding='utf-8')
    digest = screen.primary.sha256_file(plan_path)
    if fault == 'plan_hash': digest = 'wrong-plan-hash'
    calls = []
    def no_metric(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Metrics must not run for a stale plan or source')
    monkeypatch.setattr(screen.primary, 'macro_auc', no_metric)
    monkeypatch.setattr(screen, 'score_fold', no_metric)
    with pytest.raises((ValueError, FileExistsError)):
        screen.run(plan_path, digest, out)
    assert not calls
    assert not (out/'report.json').exists()


def test_all_four_primary_macros_checked_before_any_blend(tmp_path, monkeypatch):
    plan_path, _, _, _, _, report = frozen_plan(tmp_path, monkeypatch)
    report['folds'][1]['macro_b'] = .25
    calls = []
    monkeypatch.setattr(screen.primary, 'macro_auc', lambda *args: (1., dict.fromkeys(screen.LAB, 1.)))
    def no_blend(*args):
        calls.append(True)
        raise AssertionError('Fourth primary mismatch must fail before the first blend')
    monkeypatch.setattr(screen, 'score_fold', no_blend)
    with pytest.raises(ValueError, match='primary scores'):
        screen.run(plan_path, screen.primary.sha256_file(plan_path), tmp_path/'result')
    assert not calls


def test_write_new_never_overwrites_frozen_artifact(tmp_path):
    path = tmp_path/'frozen.json'
    screen.write_new(path, {'original': True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError): screen.write_new(path, {'original': False})
    assert path.read_bytes() == before


@pytest.mark.parametrize('fault', ['missing', 'tf32', 'nondeterministic', 'capability'])
def test_four_fit_runtime_receipts_reject_inconsistent_or_nonstrict_backends(fault):
    receipt = {'backends': deepcopy(screen.BACKENDS), 'compute_capability': [8, 9]}
    receipts = [deepcopy(receipt) for _ in range(8)]
    screen.validate_runtime_receipts(receipts)
    if fault == 'missing': receipts.pop()
    elif fault == 'tf32': receipts[-1]['backends']['matmul_allow_tf32'] = True
    elif fault == 'nondeterministic': receipts[-1]['backends']['deterministic_algorithms'] = False
    else: receipts[-1]['compute_capability'] = [7, 5]
    with pytest.raises(ValueError): screen.validate_runtime_receipts(receipts)


def test_one_class_target_cannot_be_reported_as_macro12():
    y = columns([0, 0, 1, 1]); y[:, 3] = 0
    a = columns([.1, .2, .3, .4])
    with pytest.raises(ValueError, match='One-class'):
        screen.score_fold(y, a, a)
