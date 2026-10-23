"""Synthetic-only EXP079 contract, rank-comparison, and stopping checks.

No saved candidate summary, prediction, Gold label, or model is opened here.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import exp079_complement_report as report


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def protocols():
    parent = dict(
        epochs=8, folds=[0, 1], labels='labels.csv', label_hash=digest('labels'),
        fold_file='folds.csv', split_hash=digest('split'), corpus='corpus',
        corpus_manifest_sha256=digest('corpus'), excluded_uids=['bad-a', 'bad-b'],
        initial_state_sha256=digest('coat-init'),
        source_file_sha256={'labels.csv': digest('label-file'), 'folds.csv': digest('fold-file')},
        training_code_sha256={f'{name}.py': digest(name) for name in
                              ['raptor_train', 'raptor_common', 'raptor_data', 'raptor_runtime']},
        expected_ids={str(f): dict(train=3474, val=869, train_ids_sha256=digest(f'train-{f}'),
                                  eval_ids_sha256=digest(f'eval-{f}')) for f in (0, 1)},
        runs=[dict(tag=f'EXP078_INPUT_{a}_F{f}', arm=a, fold=f,
                   policy='global-valid' if a == 'A' else 'within-slot-valid')
              for f in (0, 1) for a in ('A', 'B')],
        expected_args=dict(arch='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k', res=224,
            k=12, k_eval=24, epochs=8, fixed_epoch=8, select='oof', bs=8, accum=1,
            grad_ckpt=True, workers=6, amp='fp16', eval_amp='fp16', runtime_policy='strict',
            amp_init_scale=1024., seed=42, norm='imagenet', bb_lr=3e-5, head_lr=.001,
            wd=.02, limit=0, smoke=False, pretrained='timm', oof_subsample=0, folds=5))
    recipe = dict(arch='resnet34.a1_in1k', res=224, window_policy='within-slot-valid',
        k=12, k_eval=24, epochs=20, fixed_epoch=20, checkpoint='fixed-final', seed=42,
        amp='fp16', eval_amp='fp16', runtime_policy='strict', amp_init_scale=1024,
        bb_lr=.0005, head_lr=.001, wd=.02, bs=8, accum=1, workers=4, grad_ckpt=False,
        device='cuda')
    manifest = {key: deepcopy(parent[key]) for key in ('labels', 'label_hash', 'fold_file',
        'split_hash', 'source_file_sha256', 'training_code_sha256', 'corpus',
        'corpus_manifest_sha256', 'excluded_uids', 'expected_ids')}
    manifest.update(schema='exp079-r34-complement-v1', parent_manifest_sha256=digest('parent'),
        eval_labels=parent['labels'], recipe=recipe,
        runs=[dict(tag=f'EXP079_R34_B_F{f}', fold=f, initial_state_sha256=digest('rn-init')) for f in (0, 1)],
        baselines=[dict(tag=f'EXP078_INPUT_B_F{f}', fold=f, initial_state_sha256=digest('coat-init'),
                        summary_sha256=digest(f'summary-{f}'), oof_sha256=digest(f'oof-{f}')) for f in (0, 1)],
        pretrained=dict(initial_full_model_sha256=digest('rn-init'), source='synthetic',
                        revision=digest('revision')[:40], file='model.safetensors',
                        sha256=digest('pretrained'), offline_cached_only=True),
        evaluation=dict(fixed_global_rank_weights={'baseline': .9, 'candidate': .1},
                        ties='average rank/n within fold and target, no final re-rank',
                        major_drop_review=-.01),
        stop_review=dict(optimizer_skip_fraction=.01, consecutive_skips_review=10,
                         epoch_metric_early_stopping=False),
        source_decision=dict(path='workspace://decision.json', sha256=digest('decision')),
        cpu_screen=dict(path='workspace://screen.json', sha256=digest('screen'), ratio_search=False),
        queue=dict(path='workspace://queue.json', sha256=digest('queue')))
    return manifest, parent


def make_summary(protocol, run):
    args = {**protocol['expected_args'], 'tag': run['tag'], 'fold': run['fold'],
            'window_policy': run['policy']}
    runtime = dict(requested_amp='fp16', effective_amp='fp16', device='cuda', policy='strict',
        precision_fallback_allowed=False,
        backends=dict(cudnn_benchmark=False, cudnn_deterministic=True, cudnn_allow_tf32=False,
                      matmul_allow_tf32=False, float32_matmul_precision='highest',
                      deterministic_algorithms=True, cublas_workspace_config=':4096:8'))
    p = dict(checkpoint_selection='fixed-final', code_sha256=protocol['training_code_sha256'],
        initial_state_sha256=protocol['initial_state_sha256'],
        corpus_manifest_sha256=protocol['corpus_manifest_sha256'],
        eval_label_hash=protocol['label_hash'], excluded_ids=sorted(protocol['excluded_uids']),
        sampling='sha256(seed,epoch,uid,stream)-v1',
        runtime=deepcopy(runtime), eval_runtime=deepcopy(runtime))
    p.update({k: v for k, v in protocol['expected_ids'][str(run['fold'])].items() if k.endswith('_sha256')})
    epochs = protocol['epochs']
    return dict(no_oof_epoch_selection=True, no_gold_epoch_selection=True,
        selected_epoch_1based=epochs, selected_checkpoint='best', best_epoch=epochs-1,
        results={'best': {}}, provenance=p, args=args, label_hash=protocol['label_hash'],
        eval_label_hash=protocol['label_hash'], split_hash=protocol['split_hash'], max_epochs=epochs,
        window_policy=run['policy'], fold=run['fold'], backbone=args['arch'], img=args['res'],
        train=3474, val=869, rows=869, optimizer_updates=8680, skipped_updates=0,
        history=[dict(epoch=i, loss=.2, oof_auc=None, gold_auc=None,
                      optimizer_updates=(i+1)*434, skipped_updates=0) for i in range(epochs)])


def candidate():
    manifest, parent = protocols()
    protocol, run = report.candidate_protocol(manifest, parent, manifest['runs'][0])
    return manifest, parent, protocol, run, make_summary(protocol, run)


def test_valid_protocol_and_candidate_adaptation_preserve_frozen_parent():
    manifest, parent = protocols(); before = deepcopy(parent)
    report.validate_protocol(manifest, parent, digest('parent'))
    protocol, run = report.candidate_protocol(manifest, parent, manifest['runs'][0])
    assert parent == before
    assert protocol['epochs'] == 20
    assert protocol['initial_state_sha256'] == digest('rn-init')
    for key, value in dict(arch='resnet34.a1_in1k', epochs=20, fixed_epoch=20,
                           bb_lr=.0005, grad_ckpt=False, workers=4, oof_subsample=0,
                           select='oof', norm='imagenet', amp='fp16', eval_amp='fp16').items():
        assert protocol['expected_args'][key] == value
    assert run['fold'] == 0 and run['policy'] == 'within-slot-valid'
    assert protocol['expected_ids'] == parent['expected_ids']


@pytest.mark.parametrize('fault', ['parent_sha', 'split', 'labels', 'eval_labels', 'code',
    'excluded', 'missing_candidate', 'duplicate_candidate', 'missing_baseline',
    'baseline_a', 'candidate_initial', 'baseline_initial', 'weights', 'epochs', 'precision'])
def test_changed_frozen_protocol_is_rejected(fault):
    manifest, parent = protocols()
    if fault == 'parent_sha': manifest['parent_manifest_sha256'] = digest('changed')
    elif fault == 'split': manifest['split_hash'] = digest('changed')
    elif fault == 'labels': manifest['label_hash'] = digest('changed')
    elif fault == 'eval_labels': manifest['eval_labels'] = 'different.csv'
    elif fault == 'code': manifest['training_code_sha256']['raptor_train.py'] = digest('changed')
    elif fault == 'excluded': manifest['excluded_uids'].pop()
    elif fault == 'missing_candidate': manifest['runs'].pop()
    elif fault == 'duplicate_candidate': manifest['runs'][1] = deepcopy(manifest['runs'][0])
    elif fault == 'missing_baseline': manifest['baselines'].pop()
    elif fault == 'baseline_a': manifest['baselines'][1]['tag'] = 'EXP078_INPUT_A_F1'
    elif fault == 'candidate_initial': manifest['runs'][0]['initial_state_sha256'] = digest('changed')
    elif fault == 'baseline_initial': manifest['baselines'][0]['initial_state_sha256'] = digest('changed')
    elif fault == 'weights': manifest['evaluation']['fixed_global_rank_weights']['candidate'] = .2
    elif fault == 'epochs': manifest['recipe']['fixed_epoch'] = 8
    else: manifest['recipe']['eval_amp'] = 'bf16'
    with pytest.raises(ValueError): report.validate_protocol(manifest, parent, digest('parent'))


def test_valid_final_candidate_passes():
    manifest, _, protocol, run, summary = candidate()
    assert report.validate_run(summary, protocol, run, manifest['stop_review']['optimizer_skip_fraction']) == 0


@pytest.mark.parametrize('path,value', [
    (('selected_epoch_1based',), 8), (('best_epoch',), 8), (('selected_checkpoint',), 'swa'),
    (('no_oof_epoch_selection',), False), (('no_gold_epoch_selection',), False),
    (('args', 'oof_subsample'), 200), (('args', 'arch'), 'historical-resnet'),
    (('provenance', 'initial_state_sha256'), 'coatnet-init'),
    (('provenance', 'sampling'), 'stateful-rng'),
    (('provenance', 'eval_runtime', 'effective_amp'), 'bf16'),
    (('provenance', 'eval_runtime', 'precision_fallback_allowed'), True),
    (('provenance', 'runtime', 'backends', 'matmul_allow_tf32'), True),
    (('provenance', 'runtime', 'backends', 'deterministic_algorithms'), False),
    (('rows',), 868), (('optimizer_updates',), 0), (('skipped_updates',), -1),
])
def test_invalid_candidate_receipt_is_rejected(path, value):
    manifest, _, protocol, run, summary = candidate()
    cursor = summary
    for key in path[:-1]: cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError): report.validate_run(summary, protocol, run, manifest['stop_review']['optimizer_skip_fraction'])


def test_epoch_score_peeking_is_rejected():
    manifest, _, protocol, run, summary = candidate()
    summary['history'][3]['oof_auc'] = .8
    with pytest.raises(ValueError): report.validate_run(summary, protocol, run, manifest['stop_review']['optimizer_skip_fraction'])


@pytest.mark.parametrize('bad', [float('inf'), float('nan'), 12.5, True])
def test_optimizer_counters_must_be_finite_integers(bad):
    manifest, _, protocol, run, summary = candidate()
    summary['optimizer_updates'] = bad
    for h in summary['history']: h['optimizer_updates'] = bad
    with pytest.raises(ValueError):
        report.validate_run(summary, protocol, run, manifest['stop_review']['optimizer_skip_fraction'])


@pytest.mark.parametrize('limit', [float('inf'), float('nan'), -.01, 1.01])
def test_invalid_skip_limit_is_rejected(limit):
    _, _, protocol, run, summary = candidate()
    with pytest.raises(ValueError, match='counter/skip limit'):
        report.validate_run(summary, protocol, run, limit)


@pytest.mark.parametrize('fault', ['loss', 'decreasing', 'final_mismatch'])
def test_numeric_history_is_validated(fault):
    manifest, _, protocol, run, summary = candidate()
    if fault == 'loss': summary['history'][5]['loss'] = float('nan')
    elif fault == 'decreasing': summary['history'][5]['optimizer_updates'] = 0
    else: summary['history'][-1]['optimizer_updates'] -= 1
    with pytest.raises(ValueError):
        report.validate_run(summary, protocol, run, manifest['stop_review']['optimizer_skip_fraction'])


@pytest.mark.parametrize('skipped,allowed', [(100, True), (101, False)])
def test_skip_fraction_uses_all_attempts_and_exact_one_percent_boundary(skipped, allowed):
    _, _, protocol, run, summary = candidate()
    summary.update(optimizer_updates=10000-skipped, skipped_updates=skipped)
    for i, h in enumerate(summary['history']):
        h.update(optimizer_updates=((i+1)*(10000-skipped))//20,
                 skipped_updates=((i+1)*skipped)//20)
    if allowed:
        assert report.validate_run(summary, protocol, run, .01) == .01
    else:
        with pytest.raises(ValueError, match='skip fraction'):
            report.validate_run(summary, protocol, run, .01)


def columns(values):
    return np.tile(np.asarray(values, dtype=np.float32)[:, None], (1, 12))


def test_same_fixed_blend_is_compared_to_b_and_original_a_with_average_ties():
    y = columns([0, 0, 1, 1]); b = np.full(y.shape, .5)
    rn = columns([0, .5, .5, 1]); a = y.copy()
    rows, rank_b, blend = report.paired_rows(y, b, rn, a)
    np.testing.assert_array_equal(rank_b[:, 0], [.625]*4)
    np.testing.assert_allclose(blend[:, 0], [.5875, .625, .625, .6625], rtol=0, atol=1e-15)
    for row in rows:
        assert row['P'] == row['N'] == 2
        assert row['baseline_B_auc'] == .5
        assert row['candidate_auc'] == row['blend_auc'] == .875
        assert row['original_A_auc'] == 1
        assert row['delta_vs_B'] == .375
        assert row['delta_vs_A'] == -.125
        assert row['macro_contribution_vs_B'] == .375/12
    whole = report.aggregate(rows)
    assert whole['macro12_delta_vs_B'] == .375
    assert whole['macro12_delta_vs_A'] == -.125
    # The actual score vector is retained; final re-ranking would change these values.
    assert not np.array_equal(blend[:, 0], [.25, .625, .625, 1])


def test_global_weights_apply_to_pfoa_even_when_candidate_hurts_it():
    y = columns([0, 0, 1, 1]); baseline = np.full(y.shape, .5)
    rn = columns([0, .5, .5, 1]); rn[:, report.LAB.index('PF OA')] = 1-rn[:, 0]
    rows, _, _ = report.paired_rows(y, baseline, rn, baseline)
    result = report.aggregate(rows)
    assert result['pfoa_delta_vs_B'] == -.375
    assert result['other11_delta_vs_B'] == .375
    assert result['macro12_delta_vs_B'] == .3125


def test_identical_predictions_no_gain_and_one_class_subgroup_is_not_partial_macro():
    y = columns([0, .5, 1, 1]); p = columns([0, .5, .5, 1])
    rows, _, _ = report.paired_rows(y, p, p, p)
    assert report.aggregate(rows)['macro12_delta_vs_B'] == 0
    assert rows[0]['P'] == rows[0]['N'] == 2
    y[:, 0] = 0
    rows, _, _ = report.paired_rows(y, p, p, p)
    result = report.aggregate(rows)
    assert result['macro12_delta_vs_B'] is None
    assert result['macro12_delta_vs_A'] is None
    assert rows[0]['P'] == 0 and rows[0]['N'] == 4


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.01, 1.01])
def test_invalid_candidate_probabilities_fail(bad):
    b = columns([.1, .2, .3, .4]); rn = b.copy(); rn[0, 0] = bad
    with pytest.raises(ValueError, match='probabilities'): report.fixed_blend(b, rn)


def test_conditional_bootstrap_uses_frozen_scores_and_mean_of_folds():
    y = columns([0, 1, 0, 1]); b = np.full(y.shape, .5)
    rank_b, up = report.fixed_blend(b, y)
    _, down = report.fixed_blend(b, 1-y)
    groups = np.array(['a', 'a', 'b', 'b'])
    folds = [(y, rank_b, up, groups), (y, rank_b, down, groups)]
    interval = report.primary.mean_fold_bootstrap(folds, reps=20)
    assert interval['low'] == interval['high'] == 0
    per_target = report.target_intervals(folds, reps=20)
    assert all(v['low'] == v['high'] == 0 and v['valid_reps'] == 20 for v in per_target.values())


def test_missing_fold_or_stale_last_primary_score_prevents_any_blend(tmp_path, monkeypatch):
    import pandas as pd
    manifest, _ = protocols(); y = columns([0, 0, 1, 1]); p = y.copy()
    data = [dict(fold=f, y=y, values={'A': p, 'B': p, 'candidate': p},
                 report_groups=np.array(['a', 'b', 'c', 'd']), groups=pd.DataFrame()) for f in (0, 1)]
    parent_report = {'folds': [dict(fold=f, macro_a=1., macro_b=1.) for f in (0, 1)]}
    monkeypatch.setattr(report, 'load_inputs', lambda *args: (manifest, parent_report, data[:1], {}, []))
    calls = []
    def no_score(*args):
        calls.append(True)
        raise AssertionError('Both folds and parent scores must validate before scoring')
    monkeypatch.setattr(report, 'paired_rows', no_score)
    monkeypatch.setattr(report.primary, 'macro_auc', no_score)
    with pytest.raises(ValueError, match='Both completed folds'):
        report.build('unused', 'unused', 'unused', 'unused', 'unused', tmp_path/'missing')
    assert not calls
    assert not (tmp_path/'missing').exists()
    monkeypatch.setattr(report, 'load_inputs', lambda *args: (manifest, parent_report, data, {}, []))
    monkeypatch.setattr(report.primary, 'macro_auc', lambda *args: (1., dict.fromkeys(report.LAB, 1.)))
    parent_report['folds'][1]['macro_b'] = .9
    with pytest.raises(ValueError, match='Parent primary score mismatch'):
        report.build('unused', 'unused', 'unused', 'unused', 'unused', tmp_path/'stale')
    assert not calls
    assert not (tmp_path/'stale'/'report.json').exists()


def test_existing_output_fails_before_loading_any_input(tmp_path, monkeypatch):
    out = tmp_path/'existing'; out.mkdir()
    def no_load(*args): raise AssertionError('Existing output must fail before input loading')
    monkeypatch.setattr(report, 'load_inputs', no_load)
    with pytest.raises(FileExistsError):
        report.build('unused', 'unused', 'unused', 'unused', 'unused', out)


def test_manifest_hash_mismatch_fails_before_any_artifact_read(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    manifest = tmp_path/'manifest.json'; manifest.write_text('{}', encoding='utf-8')
    def no_read(*args): raise AssertionError('Mismatched manifest hash must fail before JSON read')
    monkeypatch.setattr(report, 'read', no_read)
    with pytest.raises(ValueError, match='Frozen source changed'):
        report.load_inputs(manifest, digest('wrong-manifest'), 'unused', 'unused', 'unused')


def test_complete_synthetic_report_preserves_same_blend_and_fixed_bootstrap_plan(tmp_path, monkeypatch):
    import json
    import pandas as pd
    manifest, _ = protocols()
    y = columns([0, 0, 1, 1]); b = np.full(y.shape, .5); candidate_p = columns([0, .5, .5, 1])
    data = [dict(fold=f, y=y, values={'A': y, 'B': b, 'candidate': candidate_p},
                 report_groups=np.array(['a', 'a', 'b', 'b']),
                 groups=pd.DataFrame({'scanner_group': ['one', 'one', 'two', 'two']})) for f in (0, 1)]
    parent_report = {'folds': [dict(fold=f, macro_a=1., macro_b=.5) for f in (0, 1)]}
    monkeypatch.setattr(report, 'load_inputs', lambda *args: (manifest, parent_report, data, {}, []))
    calls = []
    def target_ci(folds, reps, seed):
        calls.append(('targets', reps, seed))
        return {t: {'low': 0., 'high': 1., 'valid_reps': reps} for t in report.LAB}
    def macro_ci(folds, reps, seed):
        calls.append(('macro', reps, seed))
        return {'low': 0., 'high': 1., 'valid_reps': reps}
    monkeypatch.setattr(report, 'target_intervals', target_ci)
    monkeypatch.setattr(report.primary, 'mean_fold_bootstrap', macro_ci)
    out = tmp_path/'complete'
    result = report.build('unused', digest('synthetic-manifest'), 'unused', 'unused', 'unused', out)
    assert result['mean_fold_delta'] == result['mean_fold_pfoa_delta'] == .375
    saved = json.loads((out/'report.json').read_text(encoding='utf-8'))
    assert saved['original_A_descriptive']['mean_fold_delta'] == -.125
    assert saved['fracture']['folds'][0]['delta_vs_A'] == -.125
    assert all(g['macro12_delta_vs_B'] is None for g in saved['groups'])
    assert calls == [('targets',500,20261003),('targets',500,20261003),('macro',500,20261003),('macro',500,20261003)]
    assert len(pd.read_csv(out/'targets.csv')) == 24
