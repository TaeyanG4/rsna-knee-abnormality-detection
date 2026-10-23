"""Synthetic CPU tests only: never read project predictions, labels or reports."""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from tools import exp177_bundle_aggregation_audit as audit
from tools import exp169_bundle_screen as old
from tools import exp169_partner_bundle_release as release
from tools import exp115_vlm_eval as ev


@pytest.fixture
def folds():
    rng = np.random.default_rng(47)
    result = {}
    for f, n in enumerate((17, 21, 23, 19, 25)):
        plain = {k: np.round(rng.random((n, 12)), 2) for k in audit.INPUTS}
        result[f] = {'ids': [f'f{f}-{i}' for i in range(n)], 'plain': plain,
                     'flip': {'convnext': np.round(rng.random((n, 12)), 2)}}
    return result


def references(n):
    y = np.tile((np.arange(n) % 2)[:, None], (1, 12)).astype(float)
    mask = np.ones_like(y, bool)
    mask[::7, 1] = False
    return {name: (y.copy(), mask.copy(), 2 if name == 'consensus' else 1) for name in audit.REFS}


def test_paths_never_use_dread_convnext():
    for f in range(5):
        plain, flip = audit.paths(Path('root'), f)
        assert 'EXP170_CONVNEXT_T_L2/outputs/results/EXP170_CONVNEXT_T_L2_S42_F' + str(f) in plain['convnext'].as_posix()
        assert plain['convnext'].name == 'oof_k36.csv'
        assert flip['convnext'].name == 'oof_k36_flip_tta.csv'
        assert plain['dino'] == old.paths(Path('root'), f)['dino']
        h1_exp = 'EXP149' if f < 2 else 'EXP153'
        assert plain['h1'] == Path('root/artifacts/experiments') / f'{h1_exp}_H1_S42_F{f}/oof_k36.csv'
        assert 'h1' not in flip  # fixed plain baseline, not another TTA ablation
        assert all('EXP166' not in str(p) and 'EXP167' not in str(p) for p in [*plain.values(), *flip.values()])


def test_admission_exactly_reproduces_screen_helpers(folds):
    _, variants, _, _ = audit.build_variants(folds)
    pooled = {
        'l2': np.concatenate([(ev.pct_rank(b['plain']['l2_s42']) + ev.pct_rank(b['plain']['l2_s43'])) / 2
                              for b in folds.values()]),
        'dino': np.concatenate([ev.pct_rank(b['plain']['dino']) for b in folds.values()]),
        'convnext': np.concatenate([ev.pct_rank(b['flip']['convnext']) for b in folds.values()])}
    bundle, families = old.bundle_of(pooled)
    np.testing.assert_array_equal(variants['admission.bundle'], bundle)
    for n, values in families.items():
        np.testing.assert_array_equal(variants['admission.' + n], values)


def test_seed_probability_is_distinct_not_average_of_ranks(folds):
    _, variants, pairs, _ = audit.build_variants(folds)
    expected = audit.rank(np.concatenate([ev.pct_rank((b['plain']['l2_s42'] + b['plain']['l2_s43']) / 2)
                                          for b in folds.values()]))
    np.testing.assert_array_equal(variants['fold_probability.l2'], expected)
    assert not np.array_equal(expected, variants['admission.l2'])
    assert pairs['seed_probability_minus_rank.l2']['control'] == 'admission.l2'
    assert not np.array_equal(variants['pooled64.bundle'], variants['fold_probability.bundle'])


def test_h1_is_fixed_fold_rank_baseline_not_a_bundle_member(folds):
    _, variants, pairs, means = audit.build_variants(folds)
    expected = old.global_rank(np.concatenate([ev.pct_rank(b['plain']['h1']) for b in folds.values()]))
    np.testing.assert_array_equal(variants['baseline.h1'], expected)
    assert not np.array_equal(expected, old.global_rank(np.concatenate([b['plain']['h1'] for b in folds.values()])))
    assert pairs['bundle_minus_h1'] == {'candidate': 'admission.bundle', 'control': 'baseline.h1'}
    assert pairs['serving_oof_minus_h1'] == {'candidate': 'serving_oof.bundle', 'control': 'baseline.h1'}
    assert set(means) == set(audit.FAMILIES) == {'l2', 'dino', 'convnext'}
    for b in folds.values():
        b['plain']['h1'] = 1 - b['plain']['h1']
    _, changed, _, _ = audit.build_variants(folds)
    assert not np.array_equal(changed['baseline.h1'], expected)
    for name in variants.keys() - {'baseline.h1'}:
        np.testing.assert_array_equal(changed[name], variants[name])


def test_h1_paired_comparisons_match_direct_resampling(folds):
    ids, variants, pairs, _ = audit.build_variants(folds)
    pairs = {n: p for n, p in pairs.items() if p['control'] == 'baseline.h1'}
    variants = {n: variants[n] for n in {'baseline.h1', *(p['candidate'] for p in pairs.values())}}
    n = len(ids)
    rng = np.random.default_rng(31)
    groups = [np.arange(i, min(i + 3, n)) for i in range(0, n, 3)]
    draws = ev.bootstrap_index(groups, 9, 20261003)
    weights = np.stack([np.bincount(ii, minlength=n) for ii in draws])
    refs = references(n)
    fold_labels = np.concatenate([np.full(len(b['ids']), f) for f, b in folds.items()])
    result = audit.evaluate(variants, pairs, refs, weights, fold_labels, audit.Budget())
    for name, pair in pairs.items():
        a, b = variants[pair['candidate']], variants[pair['control']]
        for refname, (y, mask, minimum) in refs.items():
            actual = result['comparisons'][name]['references'][refname]
            targets = ev.auc_targets(y, a, mask, minimum) - ev.auc_targets(y, b, mask, minimum)
            deltas = np.stack([ev.auc_targets(y[ii], a[ii], mask[ii], minimum) -
                               ev.auc_targets(y[ii], b[ii], mask[ii], minimum) for ii in draws])
            assert actual['delta'] == pytest.approx(targets.mean(), abs=1e-14)
            np.testing.assert_allclose(actual['ci95'], np.percentile(deltas.mean(1), [2.5, 97.5]), atol=1e-14)
            for j, target in enumerate(audit.TARGETS):
                assert actual['targets'][target]['delta'] == pytest.approx(targets[j], abs=1e-14)
            for f in folds:
                ix = np.flatnonzero(fold_labels == f)
                expected = np.mean(ev.auc_targets(y[ix], a[ix], mask[ix], minimum) -
                                   ev.auc_targets(y[ix], b[ix], mask[ix], minimum))
                assert result['per_fold_deltas'][str(f)][name][refname] == pytest.approx(expected, abs=1e-14)


def test_raw_family_probability_mean_and_ablation_weights(folds):
    _, v, pairs, means = audit.build_variants(folds)
    p = {n: np.concatenate([b['plain'][n] for b in folds.values()]) for n in audit.INPUTS}
    c = np.concatenate([b['flip']['convnext'] for b in folds.values()])
    expected = audit.rank(((p['l2_s42'] + p['l2_s43']) / 2 + p['dino'] + c) / 3)
    np.testing.assert_array_equal(v['family_probability.bundle'], expected)
    for family in audit.FAMILIES:
        expected = audit.rank_bundle({n: a for n, a in means.items() if n != family})
        np.testing.assert_array_equal(v['serving_oof.without_' + family], expected)
    for seed in (42, 43):
        assert pairs[f'admission.seed{seed}']['control'] == 'admission.l2'
        assert pairs[f'admission.seed{seed}.bundle']['control'] == 'admission.bundle'


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_full_serving_exact_parity_global_order_and_shards(dtype):
    rng = np.random.default_rng(71)
    values = {n: np.round(rng.random((10 if n == 'l2' else 5, 43, 12)), 2).astype(dtype)
              for n in audit.FAMILIES}
    values['l2'][:, :2] = .5
    a, families = audit.serving_aggregate(values)
    b, expected = release.aggregate(values)
    np.testing.assert_array_equal(a, b)
    for n in audit.FAMILIES:
        np.testing.assert_array_equal(families[n], expected[n])
    permutation = rng.permutation(43)
    reordered, _ = audit.serving_aggregate({n: v[:, permutation] for n, v in values.items()})
    np.testing.assert_array_equal(a, reordered[np.argsort(permutation)])
    joined = {n: np.concatenate(np.array_split(v, [3, 19], axis=1), axis=1) for n, v in values.items()}
    np.testing.assert_array_equal(a, audit.serving_aggregate(joined)[0])
    trio = rng.random((43, 12))
    np.testing.assert_array_equal(release.final_values(trio, a), .85 * audit.rank(trio) + .15 * audit.rank(a))


@pytest.mark.parametrize('bad', ['missing', 'nan', 'range', 'count', 'cohort', 'empty'])
def test_invalid_serving_inputs_rejected(bad):
    v = {n: np.full((10 if n == 'l2' else 5, 8, 12), .5) for n in audit.FAMILIES}
    if bad == 'missing':
        del v['dino']
    elif bad in ('nan', 'range'):
        v['l2'][0, 0, 0] = np.nan if bad == 'nan' else 1.1
    elif bad == 'count':
        v['l2'] = v['l2'][:2]
    elif bad == 'cohort':
        v['convnext'] = v['convnext'][:, :3]
    else:
        v = {n: a[:, :0] for n, a in v.items()}
    with pytest.raises(ValueError):
        audit.serving_aggregate(v)


def test_ties_singleton_and_constant_columns():
    x = np.tile(np.array([.1, .1, .8, .8])[:, None], (1, 12))
    np.testing.assert_allclose(audit.rank(x)[:, 0], [1/6, 1/6, 5/6, 5/6])
    np.testing.assert_array_equal(audit.rank(np.full((5, 12), .1)), .5)
    np.testing.assert_array_equal(audit.rank(np.full((1, 12), .1)), 0)


def test_order_invariance_checks_and_negative_control(folds):
    _, _, _, means = audit.build_variants(folds)
    checks = audit.invariance(folds, means)
    assert checks['fold_row_order'] == 'PASS'
    assert checks['shard_local_rank_negative_control'] == 'DETECTED'
    assert checks['serving_aggregate_float32'] == 'PASS'


def test_duplicate_missing_extra_shard_ids_fail():
    a = np.zeros((2, 12))
    with pytest.raises(ValueError, match='Duplicate shard'):
        audit.reconstruct_shards(['a', 'b'], [(['a', 'a'], a)])
    with pytest.raises(ValueError, match='Missing/extra'):
        audit.reconstruct_shards(['a', 'b'], [(['a', 'c'], a)])
    with pytest.raises(ValueError, match='Duplicate canonical'):
        audit.reconstruct_shards(['a', 'a'], [(['a', 'b'], a)])


def test_missing_optional_tta_is_not_substituted_and_complete_replacement(folds):
    _, v, _, _ = audit.build_variants(folds)
    assert not any('flip_l2' in k for k in v)
    folds[0]['flip']['l2_s42'] = folds[0]['plain']['l2_s42']
    assert 'serving_oof.flip_l2_s42' not in audit.build_variants(folds)[1]
    for b in folds.values():
        for n in ('l2_s42', 'l2_s43'):
            b['flip'][n] = b['plain'][n].copy()
    _, v, _, _ = audit.build_variants(folds)
    np.testing.assert_array_equal(v['serving_oof.flip_l2_both'], v['serving_oof.bundle'])


def test_weighted_auc_matches_repeated_rows_with_ties_masks_components():
    rng = np.random.default_rng(38)
    n = 53
    values = np.round(rng.random((n, 12)), 1)
    ref = references(n)['consensus']
    groups = [np.array([i, i + 1]) for i in range(0, n - 1, 2)] + [np.array([n - 1])]
    draws = ev.bootstrap_index(groups, 40, 20261003)
    weights = np.stack([np.bincount(ii, minlength=n) for ii in draws])
    expected = np.stack([ev.auc_targets(ref[0][ii], values[ii], ref[1][ii], ref[2]) for ii in draws])
    np.testing.assert_allclose(audit.weighted_auc(values, ref, weights), expected, atol=1e-14, rtol=0)
    # Tied positives/negatives get exactly half credit under repeated weights.
    np.testing.assert_array_equal(audit.weighted_auc(np.full_like(values, .5), ref, weights), .5)


def test_macro_keeps_twelve_targets_and_reports_invalid_draws():
    scores = np.ones((4, 12)) * .5
    scores[1, 3] = np.nan
    assert np.isnan(audit.fixed_macro(scores)[1])
    assert audit.interval(audit.fixed_macro(scores)) == {'ci95': [.5, .5], 'valid_draws': 3, 'invalid_draws': 1}
    n = 12
    ref = references(n)['jev']
    y = ref[0].copy()
    y[:, -1] = 0
    values = np.tile(np.arange(n)[:, None], (1, 12))
    res = audit.evaluate({'a': values, 'b': values[::-1]}, {'pair': {'candidate': 'a', 'control': 'b'}},
        {r: (y, ref[1], 1) for r in audit.REFS}, np.ones((3, n)), np.zeros(n), audit.Budget())
    v = res['comparisons']['pair']['references']['jev']
    assert v['delta'] is None and v['ci95'] is None and v['invalid_draws'] == 3
    assert res['scores']['a']['jev']['scorable_targets'] == 11
    json.dumps(res, allow_nan=False)


def test_paired_intervals_use_same_draws_and_cache_targets(folds):
    ids, variants, _, _ = audit.build_variants(folds)
    n = len(ids)
    a, b = variants['admission.bundle'], variants['fold_probability.bundle']
    rng = np.random.default_rng(23)
    draws = [rng.integers(0, n, n) for _ in range(9)]
    weights = np.stack([np.bincount(ii, minlength=n) for ii in draws])
    refs = references(n)
    r = audit.evaluate({'a': a, 'b': b}, {'b-a': {'candidate': 'b', 'control': 'a'},
        'a-a': {'candidate': 'a', 'control': 'a'}}, refs, weights, np.zeros(n), audit.Budget())
    ref = refs['jev']
    deltas = [np.mean(ev.auc_targets(ref[0][ii], b[ii], ref[1][ii], ref[2]) -
                      ev.auc_targets(ref[0][ii], a[ii], ref[1][ii], ref[2])) for ii in draws]
    np.testing.assert_allclose(r['comparisons']['b-a']['references']['jev']['ci95'], np.percentile(deltas, [2.5, 97.5]), atol=1e-14)
    assert r['comparisons']['a-a']['references']['jev']['ci95'] == [0., 0.]


def write_oof(path, ids=('b', 'a'), probability=.25):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['StudyInstanceUID', *audit.TARGETS])
        for uid in ids:
            writer.writerow([uid, *([probability] * 12)])
    inventory = {'oof_k36.csv': {'bytes': path.stat().st_size, 'sha256': audit.sha(path)}}
    (path.parent / 'output-inventory.json').write_text(json.dumps(inventory), encoding='utf-8')


def test_prediction_hash_schema_and_roster_guards(tmp_path):
    path = tmp_path / 'oof_k36.csv'
    write_oof(path)
    records = {}
    assert audit.prediction(path, ['a', 'b'], records, tmp_path).shape == (2, 12)
    assert 'oof_k36.csv' in records
    with pytest.raises(ValueError, match='held-out IDs'):
        audit.prediction(path, ['a', 'z'], {}, tmp_path)
    path.write_text(path.read_text() + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='inventory differs'):
        audit.prediction(path, ['a', 'b'], {}, tmp_path)


@pytest.mark.parametrize('ids,value', [(('a', 'a'), .1), (('a', 'b'), float('nan')), (('a', 'b'), -1)])
def test_prediction_malformed_values_and_duplicate_ids(tmp_path, ids, value):
    path = tmp_path / 'oof_k36.csv'
    write_oof(path, ids, value)
    with pytest.raises(ValueError):
        audit.prediction(path, ['a', 'b'], {}, tmp_path)


def test_component_bootstrap_rejects_missing_duplicate_and_cross_fold(tmp_path, monkeypatch):
    comp = tmp_path / 'components.json'
    monkeypatch.setattr(ev, 'COMPONENTS', 'components.json')
    comp.write_text(json.dumps([{'component_id': 'x', 'members': ['a', 'b']},
                                {'component_id': 'y', 'members': ['c']}]))
    weights, info = audit.component_draws(['a', 'b', 'c'], {'a': 0, 'b': 0, 'c': 1}, tmp_path)
    assert weights.shape == (300, 3) and info['groups'] == 2
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])
    with pytest.raises(ValueError, match='crosses'):
        audit.component_draws(['a', 'b', 'c'], {'a': 0, 'b': 1, 'c': 1}, tmp_path)
    with pytest.raises(ValueError, match='Missing validation'):
        audit.component_draws(['a', 'z'], {'a': 0, 'z': 0}, tmp_path)
    comp.write_text(json.dumps([{'component_id': 'x', 'members': ['a', 'a']}]))
    with pytest.raises(ValueError, match='Duplicate component'):
        audit.component_draws(['a'], {'a': 0}, tmp_path)


def test_cpu_limits_and_budget(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '19')
    with audit.cpu_limits(1):
        assert os.environ['OMP_NUM_THREADS'] == '1'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert os.environ['OMP_NUM_THREADS'] == '19'
    with pytest.raises(ValueError):
        with audit.cpu_limits(5):
            pass
    for legacy_seconds in (.1, -1, 0, 1e9, None):
        budget = audit.Budget(legacy_seconds)
        budget.cpu = time.process_time() - 100000
        budget.wall = time.monotonic() - 100000
        usage = budget.check()
        assert usage['cpu_seconds'] >= 100000
        assert usage['wall_seconds'] >= 100000


def test_protocol_fail_closed_if_unregistered_or_changed(tmp_path, monkeypatch):
    protocol = tmp_path / audit.PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_text('<!-- EXP177_CONTRACT\n{}\n-->')
    with pytest.raises(ValueError, match='unregistered'):
        audit.verify_protocol(tmp_path)
    contract = {'version': audit.VERSION, 'draws': 300, 'seed': 20261003, 'max_seconds': 7200,
                'source_bindings': {'test': 'frozen'}}
    protocol.write_text('<!-- EXP177_CONTRACT\n' + json.dumps(contract) + '\n-->')
    monkeypatch.setattr(audit, 'source_bindings', lambda root: {'test': 'changed'})
    with pytest.raises(ValueError, match='source changed'):
        audit.verify_protocol(tmp_path)
    monkeypatch.setattr(audit, 'source_bindings', lambda root: {'test': 'frozen'})
    for legacy_seconds in (-1, 0, 1e9, None):
        contract['max_seconds'] = legacy_seconds
        protocol.write_text('<!-- EXP177_CONTRACT\n' + json.dumps(contract) + '\n-->')
        assert audit.verify_protocol(tmp_path) == contract
    contract['draws'] = 1
    protocol.write_text('<!-- EXP177_CONTRACT\n' + json.dumps(contract) + '\n-->')
    with pytest.raises(ValueError, match='unregistered'):
        audit.verify_protocol(tmp_path)


def test_end_to_end_synthetic_receipts_and_no_overwrite(tmp_path, monkeypatch, folds):
    from tools import exp156_k_sweep as e156
    protocol = tmp_path / audit.PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_text('synthetic fixture only')
    monkeypatch.setattr(audit, 'verify_protocol', lambda root: {'version': audit.VERSION})
    monkeypatch.setattr(audit, 'load_inputs', lambda root, budget: (folds, {}))
    # All file reads in this integration test are synthetic metadata or references.
    for p in {e156.JEV, e156.AQUILA, e156.DREAD, *e156.CONSENSUS, ev.COMPONENTS}:
        path = tmp_path / p
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic')
    monkeypatch.setattr(e156, 'references', lambda ids, root: dict(zip(
        ('jev_hybrid', 'aquila_v1', 'dread', 'consensus'), references(len(ids)).values())))
    monkeypatch.setattr(audit, 'component_draws', lambda ids, fold_of, root:
                        (np.ones((3, len(ids))), {'draws': 3, 'fixture': True}))
    output = tmp_path / audit.EXP / 'synthetic'
    note = tmp_path / 'verification.md'
    note.write_text('Synthetic verification fixture, not a project review.', encoding='utf-8')
    provenance = {'version': audit.VERSION, 'verification_note': {'path': str(note.resolve()), 'sha256': audit.sha(note)}}
    r = audit.audit(tmp_path, output, verification_note=Path('verification.md'))
    assert r['status'] == 'DIAGNOSTIC_COMPLETE'
    result = json.loads((output / 'result.json').read_text())
    assert result['resource']['timing_policy'] == 'measurement_only'
    assert 'max_wall_seconds' not in result['resource']
    assert 'max_cpu_seconds' not in result['resource']
    assert result['tta_availability']['l2_s42']['status'] == 'UNMEASURED'
    assert result['tta_availability']['convnext']['status'] == 'FULL'
    assert set(result['cohorts']) == {'full_oof', 'tta_convnext'}
    assert 'baseline.h1' in result['cohorts']['full_oof']['scores']
    assert 'bundle_minus_h1' in result['cohorts']['full_oof']['comparisons']
    assert 'serving_oof_minus_h1' in result['cohorts']['full_oof']['comparisons']
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['outputs_sha256']['result.json'] == audit.sha(output / 'result.json')
    assert receipt['outputs_sha256']['result.md'] == audit.sha(output / 'result.md')
    for output_data in (r, result, receipt):
        for key, value in provenance.items():
            assert output_data[key] == value
    report = (output / 'result.md').read_text(encoding='utf-8')
    assert 'v' + audit.VERSION in report
    assert provenance['verification_note']['path'] in report
    assert provenance['verification_note']['sha256'] in report
    assert 'bundle_minus_h1' in report and 'serving_oof_minus_h1' in report
    assert not (output / 'failure.json').exists()
    with pytest.raises(FileExistsError):
        audit.audit(tmp_path, output, verification_note=note)
    outside = tmp_path / 'unapproved'
    with pytest.raises(ValueError, match='Output must'):
        audit.audit(tmp_path, outside, verification_note=note)
    assert not outside.exists()
    # Bind the original note throughout a run, including in a failure receipt.
    def mutate_note(by_fold, means):
        note.write_text('changed after verification was bound', encoding='utf-8')
        return {}
    monkeypatch.setattr(audit, 'invariance', mutate_note)
    changed = tmp_path / audit.EXP / 'changed-note'
    with pytest.raises(ValueError, match='Verification note changed'):
        audit.audit(tmp_path, changed, verification_note=note)
    failure = json.loads((changed / 'failure.json').read_text())
    assert failure['version'] == provenance['version']
    assert failure['verification_note'] == provenance['verification_note']
    assert not (changed / 'receipt.json').exists()
    assert not (changed / 'result.json').exists()


def test_failure_receipt_never_claims_completion(tmp_path, monkeypatch):
    from tools import exp156_k_sweep as e156
    monkeypatch.setattr(audit, 'verify_protocol', lambda root: {})
    output = tmp_path / audit.EXP / 'failed'
    note = tmp_path / 'verification.md'
    note.write_text('Synthetic verification fixture', encoding='utf-8')
    # Deliberately absent synthetic reference input.
    with pytest.raises(FileNotFoundError):
        audit.audit(tmp_path, output, verification_note=note)
    assert (output / 'failure.json').exists()
    failure = json.loads((output / 'failure.json').read_text())
    assert failure['status'] == 'FAILED_INCOMPLETE'
    assert failure['version'] == audit.VERSION
    assert failure['verification_note'] == {'path': str(note.resolve()), 'sha256': audit.sha(note)}
    assert not (output / 'receipt.json').exists()


@pytest.mark.parametrize('note_name', [None, 'missing.md'])
def test_missing_verification_note_fails_before_creating_outputs(tmp_path, note_name):
    output = tmp_path / audit.EXP / 'missing-note'
    with pytest.raises(ValueError, match='[Vv]erification.note'):
        audit.audit(tmp_path, output, verification_note=note_name)
    assert not output.exists()


def test_cli_requires_and_forwards_verification_note(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        audit.main([])
    assert exc.value.code == 2
    assert '--verification-note' in capsys.readouterr().err
    calls = []
    def fake_audit(*args):
        calls.append(args)
        return {'status': 'synthetic'}
    monkeypatch.setattr(audit, 'audit', fake_audit)
    audit.main(['--root', str(tmp_path), '--threads', '2', '--verification-note', 'review.md'])
    assert calls == [(tmp_path, None, 2, Path('review.md'))]


@pytest.mark.parametrize('h1_error', [None, 'missing', 'roster', 'receipt', 'hash'])
def test_load_inputs_routes_every_row_to_held_out_exp170_and_checks_identity(tmp_path, monkeypatch, h1_error):
    from tools import exp149_label_sources as e149
    ids = [f'study-{i}' for i in range(4343)]
    fold_of = {u: i % 5 for i, u in enumerate(ids)}
    monkeypatch.setattr(e149, 'fold_rosters', lambda manifest, root: (ids, fold_of))
    manifest = {'paths': {'base_ordered_ids': 'base.json'}, 'folds': {}}
    (tmp_path / 'base.json').write_text('{}')
    for f in range(5):
        name = f'fold{f}.json'
        (tmp_path / name).write_text('{}')
        manifest['folds'][str(f)] = {'ordered_ids': {'path': name, 'sha256': audit.sha(tmp_path / name)}}
    path = tmp_path / e149.STUDENT_MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    out = tmp_path / audit.EXP170
    out.mkdir(parents=True)
    (out / 'manifest.json').write_text('{}')
    (out / 'collection.json').write_text(json.dumps({'schema': 'exp170-convnext-t-l2-v1',
        'status': 'VERIFIED', 'completed_folds': list(range(5)), 'manifest_sha256': audit.sha(out / 'manifest.json')}))
    for f in range(5):
        plain, flip = audit.paths(tmp_path, f)
        run = plain['convnext'].parent
        run.mkdir(parents=True)
        summary = {'fold': f, 'rows': sum(v == f for v in fold_of.values()),
                   'backbone': 'convnext_tiny.fb_in22k_ft_in1k',
                   'label_set': 'l2_dread75_aquila25_v1', 'gold_access': 'disabled'}
        (run / 'summary.json').write_text(json.dumps(summary))
        (run / 'output-inventory.json').write_text(json.dumps({
            'summary.json': {'sha256': audit.sha(run / 'summary.json')}}))
        flip['convnext'].write_text('synthetic filename marker')
        plain['h1'].parent.mkdir(parents=True)
        write_oof(plain['h1'], [u for u in ids if fold_of[u] == f])
    observed = []
    real_prediction = audit.prediction
    h1_paths = [audit.paths(tmp_path, f)[0]['h1'] for f in range(5)]
    def fake_prediction(path, fid, records, root):
        assert len({fold_of[u] for u in fid}) == 1
        assert 'EXP166' not in str(path) and 'EXP167' not in str(path)
        observed.append(path)
        if path in h1_paths:
            return real_prediction(path, fid, records, root)
        return np.full((len(fid), 12), .5)
    monkeypatch.setattr(audit, 'prediction', fake_prediction)
    loaded, records = audit.load_inputs(tmp_path, audit.Budget())
    assert len(observed) == 30  # five plain (including H1), one flip, for five folds
    assert sum(len(b['ids']) for b in loaded.values()) == 4343
    for f, path in enumerate(h1_paths):
        assert str(path.relative_to(tmp_path)) in records
        assert str((path.parent / 'output-inventory.json').relative_to(tmp_path)) in records
        np.testing.assert_array_equal(loaded[f]['plain']['h1'], .25)
    if h1_error:
        path = h1_paths[0]
        if h1_error == 'missing':
            path.unlink()
        elif h1_error == 'receipt':
            (path.parent / 'output-inventory.json').unlink()
        elif h1_error == 'hash':
            path.write_text(path.read_text() + '\n', encoding='utf-8')
        else:
            write_oof(path, [u for u in ids if fold_of[u] == 1])
        message = {'missing': 'Missing registered prediction', 'receipt': 'no hash-bound receipt',
                   'hash': 'inventory differs', 'roster': 'held-out IDs'}[h1_error]
        with pytest.raises(ValueError, match=message):
            audit.load_inputs(tmp_path, audit.Budget())
        return
    # Re-bind a wrong-family summary: content hash alone must not admit it.
    run = audit.paths(tmp_path, 0)[0]['convnext'].parent
    summary = json.loads((run / 'summary.json').read_text())
    summary['label_set'] = 'dread'
    (run / 'summary.json').write_text(json.dumps(summary))
    (run / 'output-inventory.json').write_text(json.dumps({
        'summary.json': {'sha256': audit.sha(run / 'summary.json')}}))
    with pytest.raises(ValueError, match='Wrong EXP170'):
        audit.load_inputs(tmp_path, audit.Budget())
