"""Synthetic tests: EXP175 subgroup support, weighted clusters and safe input contracts."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from tools import exp175_behavior as b


def test_tied_weighted_auc_matches_repeated_studies():
    y = np.array([0, 1, 1, 0, 1.])
    p = np.array([.1, .5, .5, .7, .9])
    w = np.array([2, 3, 1, 4, 2])
    repeated = np.repeat(np.arange(len(y)), w)
    assert b.auc(y, p, w) == pytest.approx(b.auc(y[repeated], p[repeated]))
    assert b.auc([0, 1], [.5, .5]) == .5
    assert np.isnan(b.auc([1, 1], [.2, .9]))


def test_terciles_are_train_defined_and_do_not_split_ties():
    cuts, bins = b.train_terciles([0, 1, 2, 3, 4, 5, 100, np.nan],
                                 ['train'] * 6 + ['test', 'test'])
    assert cuts == pytest.approx([5/3, 10/3])
    assert bins[-2] == 'high' and bins[-1] == 'missing'
    cuts, bins = b.train_terciles([4, 4, 4, 5], ['train'] * 3 + ['test'])
    assert cuts == [4, 4]
    assert list(bins) == ['low', 'low', 'low', 'high']


def test_fixed_target_support_does_not_silently_change_macro():
    Y = np.tile([0, 1, 0, 1, 0, 1], (12, 1)).T.astype(float)
    Y[3:, 1] = 1
    P = Y * .8 + .1
    selection = {'low': np.array([1, 1, 1, 0, 0, 0], bool),
                 'high': np.array([0, 0, 0, 1, 1, 1], bool)}
    result = b._score_axis(Y, P, selection, np.arange(6), 6, 25, 175, None)
    assert 'MCL' not in result['fixed_support_targets']
    assert len(result['fixed_support_targets']) == 11
    assert result['strata']['high']['targets']['MCL']['auc'] is None or np.isnan(result['strata']['high']['targets']['MCL']['auc'])
    assert result['strata']['high']['macro_all12'] is None
    assert result['strata']['high']['macro_fixed_support'] == 1
    assert result['contrasts']['high_minus_low']['macro_delta'] == 0


def test_sparse_bootstrap_is_honestly_undefined():
    assert b._ci([.8, .9], 2) is None
    assert b._ci([.8] * 10 + [np.nan] * 90, 100) is None
    assert b._ci([.8] * 100, 100) == pytest.approx([.8, .8])


@pytest.mark.parametrize('both, single, has_interval', [(15, 5, False), (25, 1, True)])
def test_paired_contrast_counts_only_jointly_finite_draws(monkeypatch, both, single, has_interval):
    # Independent strata each have both classes in their point estimates. Some
    # component draws preserve classes in only one stratum, so neither stratum's
    # count (nor their minimum) is the paired contrast's effective sample count.
    Y = np.tile([0, 1, 0, 1], (12, 1)).T.astype(float)
    P = np.tile([.1, .9, .9, .1], (12, 1)).T
    selections = {'low': np.array([1, 1, 0, 0], bool),
                  'high': np.array([0, 0, 1, 1], bool)}
    samples = iter([[0, 1, 2, 3]] * both + [[0, 1, 0, 1]] * single
                   + [[2, 3, 2, 3]] * single)

    class ComponentDraws:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 4, 4)
            return np.asarray(next(samples))

    monkeypatch.setattr(b.np.random, 'default_rng', lambda seed: ComponentDraws())
    requested = both + 2 * single
    result = b._score_axis(Y, P, selections, np.arange(4), 4, requested, 175, None)
    for stratum in ('low', 'high'):
        assert result['strata'][stratum]['valid_macro_draws'] == both + single
        assert result['strata'][stratum]['macro_ci95'] is not None
    contrast = result['contrasts']['high_minus_low']
    assert contrast['macro_delta'] == -1
    assert contrast['valid_draws'] == both
    assert isinstance(contrast['valid_draws'], int)
    if has_interval:
        assert contrast['ci95'] == pytest.approx([-1, -1])
    else:
        assert contrast['ci95'] is None


@pytest.mark.parametrize('bootstraps, supported', [(0, True), (25, False)])
def test_paired_contrast_zero_valid_draws_is_explicit(bootstraps, supported):
    labels = [0, 1, 0, 1] if supported else [0, 1, 1, 1]
    Y = np.tile(labels, (12, 1)).T.astype(float)
    selections = {'low': np.array([1, 1, 0, 0], bool),
                  'high': np.array([0, 0, 1, 1], bool)}
    result = b._score_axis(Y, Y*.8+.1, selections, np.arange(4), 4, bootstraps, 175, None)
    contrast = result['contrasts']['high_minus_low']
    assert contrast['valid_draws'] == 0
    assert contrast['ci95'] is None
    assert contrast['macro_delta'] == (0 if supported else None)
    assert len(result['fixed_support_targets']) == (12 if supported else 0)


def test_test_only_category_does_not_erase_training_macro_support():
    Y = np.tile([0, 1, 0, 1], (12, 1)).T.astype(float)
    result = b._score_axis(Y, Y*.8+.1,
                           {'scanner-train': np.ones(4, bool), 'scanner-test-only': np.zeros(4, bool)},
                           np.arange(4), 4, 0, 175, None)
    assert len(result['fixed_support_targets']) == 12
    assert result['strata']['scanner-train']['macro_fixed_support'] == 1
    assert result['strata']['scanner-test-only']['macro_fixed_support'] is None


def test_allowlist_never_parses_report_column(tmp_path):
    path = tmp_path / 'numerical.csv'
    path.write_text('study_key,ACL,report\n001,1,"not a number"\n', encoding='utf-8')
    frame = b._table(path, 'study_key', ['ACL'], binary=True)
    assert frame.index.tolist() == ['001']
    assert list(frame.columns) == ['ACL']
    path.write_text('study_key,ACL\na,1\na,0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        b._table(path, 'study_key', ['ACL'], binary=True)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _behavior_provenance(root, metrics, input_hashes=None):
    """Synthetic, byte-bound receipts; independent review validation is tested by the CLI tests."""
    if input_hashes is None:
        input_hashes = {}
        for source in metrics.loc[metrics.domain == 'production', 'source'].unique():
            cache = 'RAPTOR64_336_C140_S02' if source == 'COMPETITION' else f'EXT_{source}_RAPTOR64_336_C140'
            input_hashes[f'storage/cache/derived/{cache}/manifest.json'] = hashlib.sha256(
                f'synthetic original {source} cache'.encode()).hexdigest()
    source_hashes = {name: hashlib.sha256(('synthetic reviewed ' + name).encode()).hexdigest()
                     for name in ('exp175_image_domain_audit.py', 'exp175_behavior.py')}
    note = root / 'fixture-verification.json'
    _write(note, {'verdict': 'PASS', 'verified_tool_version': b.VERSION, 'source_hashes': source_hashes})
    verification = {'path': str(note), 'sha256': _file_sha(note),
                    'verified_tool_version': b.VERSION, 'source_hashes': source_hashes}
    metric_path = root / 'original-audit/metrics.csv'
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metric_path, index=False)
    manifest_path = metric_path.parent / 'manifest.json'
    protocol_hash = hashlib.sha256(b'synthetic frozen protocol').hexdigest()
    _write(manifest_path, {'version': b.VERSION, 'verification': verification,
                          'source_hashes': source_hashes, 'protocol_sha256': protocol_hash,
                          'input_hashes': input_hashes, 'metrics_sha256': _file_sha(metric_path),
                          'metric_rows': len(metrics), 'status': 'COMPLETE'})
    provenance = {'path': str(metric_path.relative_to(root)), 'sha256': _file_sha(metric_path),
                  'manifest_path': str(manifest_path.relative_to(root)),
                  'manifest_sha256': _file_sha(manifest_path), 'tool_version': b.VERSION,
                  'verification': verification, 'source_hashes': source_hashes,
                  'protocol_sha256': protocol_hash, 'input_hashes': input_hashes}
    return {'metrics_provenance': provenance, 'verification': verification}


def _roster_fixture(root, fold_ids=None):
    fold_ids = fold_ids or [[f'f{f}a', f'f{f}b'] for f in range(5)]
    union = [x for ids in fold_ids for x in ids]
    for f, ids in enumerate(fold_ids):
        _write(root / f'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold{f}/ordered-ids.json',
               {'train': [x for x in union if x not in ids], 'val': ids})
    return fold_ids


def test_fold_overlap_is_rejected(tmp_path):
    _roster_fixture(tmp_path, [['a', 'b'], ['b', 'c'], ['d', 'e'], ['f', 'g'], ['h', 'i']])
    with pytest.raises(ValueError, match='overlap'):
        b._rosters(tmp_path, {})


def test_prediction_set_must_equal_held_out_fold(tmp_path):
    fold_ids = _roster_fixture(tmp_path)
    rosters = b._rosters(tmp_path, {})
    path = b._model_paths(tmp_path, 'L2_ConvNeXt_T', 0)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[fold_ids[1][0], *([.5] * 12)], [fold_ids[0][1], *([.5] * 12)]],
                 columns=[b.UID, *b.TARGETS]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='held-out fold'):
        b._load_model(tmp_path, 'L2_ConvNeXt_T', rosters, {})


def _external_fixture(tmp_path):
    source = 'KNEEMRI'
    ids = [f'k{i}' for i in range(12)]
    pred_path = tmp_path / f'artifacts/experiments/EXP137_EXT_QA/{source}/predictions_full.csv'
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(np.tile(np.linspace(.05, .95, 12)[:, None], (1, 12)), columns=b.TARGETS)
    pred.insert(0, 'study_key', ids)
    pred.to_csv(pred_path, index=False)
    label_dir = tmp_path / 'storage/labels/external'
    label_dir.mkdir(parents=True)
    labels = pd.DataFrame(np.nan, index=range(12), columns=b.TARGETS)
    labels['ACL'] = [0, 1] * 6
    labels.insert(0, 'study_key', ids)
    labels.to_csv(label_dir / f'{source}_12.csv', index=False)
    native = pd.DataFrame({'study_key': ids, 'acl_any': [0, 1] * 6, 'acl_complete': [0] * 9 + [1] * 3,
                           'volume_filename': ['never-use-this'] * 12})
    native.to_csv(label_dir / f'{source}_native.csv', index=False)
    metrics = pd.DataFrame({'source': [source] * 12, 'split': ['unknown'] * 12,
                            'domain': ['production'] * 12, 'group': ['Sagittal_fluid1'] * 12,
                            'uid': ids, 'noise_hh_flat': np.arange(12.), 'laplacian_var': np.arange(12.)[::-1]})
    cache_path = tmp_path / f'storage/cache/derived/EXT_{source}_RAPTOR64_336_C140/manifest.json'
    _write(cache_path, {'synthetic_cache': 'A'})
    receipt_path = pred_path.parent / 'zeroshot_full.json'
    _write(receipt_path, {'predictions_sha256': _file_sha(pred_path),
                          'mapped_labels_sha256': _file_sha(label_dir / f'{source}_12.csv'),
                          'native_labels_sha256': _file_sha(label_dir / f'{source}_native.csv'),
                          'cache_manifest_sha256': _file_sha(cache_path)})
    provenance = _behavior_provenance(tmp_path, metrics,
                                     {cache_path.relative_to(tmp_path).as_posix(): _file_sha(cache_path)})
    return metrics, provenance, cache_path, receipt_path


def test_external_errors_and_within_class_association(tmp_path):
    metrics, provenance, _, _ = _external_fixture(tmp_path)
    result = b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=0, **provenance)
    assert result['status'] == 'COMPLETE_WITH_GAPS'
    assert {r['comparison'] for r in result['external']} == {'mapped:ACL', 'native:acl_any', 'native:acl_complete'}
    assert all('within_class' in r for r in result['external'])
    assert len(result['input_sha256']) == 4
    errors = pd.read_csv(tmp_path / 'out/zeroshot_study_errors.csv')
    assert len(errors) == 36
    assert 'label' not in errors.columns
    assert 'rank blending' in result['external_interpretation']
    assert any(j.get('finite_target_labels') == 12 for j in result['joins'])


def test_external_predictions_from_rebuilt_cache_do_not_join_original_metrics(tmp_path):
    metrics, provenance, cache_path, receipt_path = _external_fixture(tmp_path)
    original_hash = provenance['metrics_provenance']['input_hashes'][cache_path.relative_to(tmp_path).as_posix()]
    _write(cache_path, {'synthetic_cache': 'B'})
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt['cache_manifest_sha256'] = _file_sha(cache_path)
    _write(receipt_path, receipt)
    assert receipt['cache_manifest_sha256'] != original_hash
    result = b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=0, **provenance)
    assert result['external'] == []
    assert any('production cache differs from original image metrics' in reason for reason in result['missing'])
    assert not (tmp_path / 'out/zeroshot_study_errors.csv').exists()


@pytest.mark.parametrize('current_cache', ['rebuilt', 'absent'])
def test_matching_original_cache_receipt_survives_current_cache_changes(tmp_path, current_cache):
    metrics, provenance, cache_path, receipt_path = _external_fixture(tmp_path)
    original_hash = _file_sha(cache_path)
    if current_cache == 'rebuilt':
        _write(cache_path, {'synthetic_cache': 'B'})
        assert _file_sha(cache_path) != original_hash
    else:
        cache_path.unlink()
    assert json.loads(receipt_path.read_text(encoding='utf-8'))['cache_manifest_sha256'] == original_hash
    result = b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=0, **provenance)
    assert {row['comparison'] for row in result['external']} == {'mapped:ACL', 'native:acl_any', 'native:acl_complete'}
    assert not any('zero-shot linkage held' in reason for reason in result['missing'])
    assert result['input_sha256'][receipt_path.relative_to(tmp_path).as_posix()] == _file_sha(receipt_path)


@pytest.mark.parametrize('missing', ['receipt', 'metrics_cache', 'prediction_cache'])
def test_external_linkage_requires_original_cache_binding_and_receipt(tmp_path, missing):
    metrics, provenance, _, receipt_path = _external_fixture(tmp_path)
    if missing == 'receipt':
        receipt_path.unlink()
    elif missing == 'metrics_cache':
        provenance['metrics_provenance']['input_hashes'] = {}
    else:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        del receipt['cache_manifest_sha256']
        _write(receipt_path, receipt)
    result = b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=0, **provenance)
    assert result['external'] == []
    assert any('KNEEMRI: zero-shot linkage held' in reason for reason in result['missing'])
    assert not (tmp_path / 'out/zeroshot_study_errors.csv').exists()


@pytest.mark.parametrize('key', ['predictions_sha256', 'mapped_labels_sha256', 'native_labels_sha256'])
def test_external_receipt_hashes_bind_exact_numerical_inputs(tmp_path, key):
    metrics, provenance, _, receipt_path = _external_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt[key] = '0' * 64
    _write(receipt_path, receipt)
    result = b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=0, **provenance)
    assert result['external'] == []
    assert any('hash differs' in reason for reason in result['missing'])
    assert not (tmp_path / 'out/zeroshot_study_errors.csv').exists()


def test_behavior_json_retains_verification_and_original_metric_provenance(tmp_path):
    metrics, provenance, _, _ = _external_fixture(tmp_path)
    result = b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=0, **provenance)
    saved = json.loads((tmp_path / 'out/behavior.json').read_text(encoding='utf-8'))
    assert saved == result
    assert saved['verification'] == provenance['verification']
    assert saved['verification']['path'] == str(tmp_path / 'fixture-verification.json')
    assert saved['verification']['sha256'] == _file_sha(tmp_path / 'fixture-verification.json')
    assert saved['verification']['verified_tool_version'] == b.VERSION
    assert saved['metrics_provenance'] == provenance['metrics_provenance']
    assert saved['metrics_provenance']['sha256'] == _file_sha(tmp_path / saved['metrics_provenance']['path'])
    assert saved['metrics_provenance']['manifest_sha256'] == _file_sha(
        tmp_path / saved['metrics_provenance']['manifest_path'])


@pytest.mark.parametrize('missing', ['metrics_provenance', 'verification'])
def test_behavior_api_requires_both_provenance_receipts(tmp_path, missing):
    metrics, provenance, _, _ = _external_fixture(tmp_path)
    del provenance[missing]
    with pytest.raises(TypeError, match=missing):
        b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=0, **provenance)
    assert not (tmp_path / 'out/behavior.json').exists()


@pytest.mark.parametrize('missing', ['path', 'sha256', 'verified_tool_version', 'source_hashes'])
def test_behavior_api_rejects_incomplete_verification_provenance(tmp_path, missing):
    metrics, provenance, _, _ = _external_fixture(tmp_path)
    del provenance['verification'][missing]
    with pytest.raises(ValueError, match='Verification provenance is required'):
        b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=0, **provenance)
    assert not (tmp_path / 'out/behavior.json').exists()


def test_behavior_api_rejects_missing_original_metric_inputs(tmp_path):
    metrics, provenance, _, _ = _external_fixture(tmp_path)
    del provenance['metrics_provenance']['input_hashes']
    with pytest.raises(ValueError, match='Original image metrics provenance is required'):
        b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=0, **provenance)
    assert not (tmp_path / 'out/behavior.json').exists()


def test_missing_oof_stays_explicit_and_legacy_deadline_is_ignored(tmp_path):
    metrics = pd.DataFrame({'source': ['COMPETITION'], 'split': ['train'], 'uid': ['x'],
                            'domain': ['production'], 'group': ['Axial']})
    provenance = _behavior_provenance(tmp_path, metrics)
    result = b.run_behavior(tmp_path, metrics, tmp_path / 'missing', bootstraps=0, **provenance)
    assert any('held' in s for s in result['missing'])
    result = b.run_behavior(tmp_path, metrics, tmp_path / 'deadline', bootstraps=0,
                            deadline=time.monotonic()-1, **provenance)
    assert result['status'] == 'COMPLETE_WITH_GAPS'
    assert any('held' in s for s in result['missing'])
    assert not any('deadline' in s.lower() for s in result['missing'])
    assert (tmp_path / 'deadline/behavior.json').is_file()


def test_association_constant_or_too_small_is_null():
    assert b._association([1, 1, 1], [0, 1, 2])['spearman'] is None
    assert b._association([1, 2], [0, 1])['spearman'] is None
    assert b._association([1, 2, 3], [3, 2, 1])['spearman'] == -1


def test_full_synthetic_competition_preserves_test_isolation_and_joint_contrast(tmp_path):
    fold_ids = _roster_fixture(tmp_path, [[f'f{f}-{i}' for i in range(30)] for f in range(5)])
    ids = [uid for part in fold_ids for uid in part]
    y = np.tile(np.arange(150) % 2, (12, 1)).T.astype(float)
    pred = y * .8 + .1
    pred[100:] = 1-pred[100:]
    for model in ('L2_ConvNeXt_T', 'L2_RadR50'):
        for f in range(5):
            for path in b._model_paths(tmp_path, model, f):
                path.parent.mkdir(parents=True, exist_ok=True)
                frame = pd.DataFrame(pred[f*30:(f+1)*30], columns=b.TARGETS)
                frame.insert(0, b.UID, fold_ids[f])
                frame.to_csv(path, index=False)
    for relative in b.REFERENCES.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(y, columns=b.TARGETS)
        frame.insert(0, b.UID, ids)
        frame.to_csv(path, index=False)
    _write(tmp_path / b.COMPONENTS, [{'component_id': f'c{i}', 'members': ids[i*2:i*2+2]}
                                    for i in range(75)])
    metrics = pd.DataFrame({'source': ['COMPETITION'] * 151, 'domain': ['production'] * 151,
                            'uid': [*ids, 'TEST-ID-NO-LABEL'], 'group': ['Axial'] * 151,
                            'split': ['train'] * 150 + ['test'], 'noise_hh_flat': [*range(150), 999],
                            'laplacian_var': [*range(149, -1, -1), 1],
                            'vendor': ['scanner'] * 151, 'field_strength': ['3.0'] * 151,
                            'study_vendor': ['supplemental-scanner'] * 151, 'study_field_strength': ['1.5'] * 151})
    result = b.run_behavior(tmp_path, metrics, tmp_path / 'out', bootstraps=25, deadline=-1,
                            **_behavior_provenance(tmp_path, metrics))
    assert result['missing'] == []
    assert result['oof_validation']['studies'] == 150
    assert len(result['input_sha256']) == 23
    supplemental = [r for r in result['competition'] if r['metric'].startswith('study_')]
    assert len(supplemental) == 8
    assert all('supplemental study' in r['metadata_scope'] for r in supplemental)
    assert all(j.get('label_row_present') == 150 for j in result['joins'])
    noise = [r for r in result['competition'] if r['metric'] == 'noise_hh_flat']
    assert len(noise) == 4
    for row in noise:
        assert row['mix']['test']['high'] == 1
        assert len(row['fixed_support_targets']) == 12
        assert row['contrasts']['high_minus_low']['macro_delta'] == pytest.approx(-1)
        assert row['contrasts']['high_minus_low']['valid_draws'] == 25
        assert row['strata']['high']['studies'] == 50
        assert row['strata']['high']['targets']['ACL']['n'] == 50
        assert row['strata']['high']['macro_ci95'] == pytest.approx([0, 0])
