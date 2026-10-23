"""Small synthetic EXP084 tests; no actual labels, predictions, Gold or models."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import exp084_group_loss_report as r


def sample():
    y = np.tile(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)[:, None], (1, 12))
    good = .1 + .8 * y
    return y, np.full_like(good, .5), 1 - good, good


def test_average_ties_divided_by_full_n_and_no_final_rank():
    y, b, clip, ones = sample()
    scores = r.scores_for(b, clip, ones)
    assert np.unique(scores['B']).tolist() == [7 / 12]
    assert scores['blend_ones'][0, 0] == pytest.approx(.9 * 7 / 12 + .1 / 3)
    assert scores['blend_ones'][1, 0] == pytest.approx(.9 * 7 / 12 + .1 * 5 / 6)
    rows = r.target_rows(y, scores)
    assert r.aggregate(rows)['contrasts']['primary_blend_ones_minus_clip']['macro12'] == 1
    assert sum(x['delta_over_12']['primary_blend_ones_minus_clip'] for x in rows) == pytest.approx(1)


def test_weighted_tied_auc_matches_actual_repeated_rows():
    y = np.array([0, 1, 1, 0, 1])
    pred = np.array([.4, .4, .8, .9, .9])
    weights = np.array([3, 2, 0, 1, 4])
    expected = roc_auc_score(np.repeat(y, weights), np.repeat(pred, weights))
    assert r.weighted_auc(y, pred, weights) == pytest.approx(expected)


def test_components_apply_same_multiplicity_to_every_member():
    codes = np.array([0, 0, 1, 2, 2, 2])
    assert r.component_weights(codes, np.array([2, 0, 2]), 3).tolist() == [1, 1, 0, 2, 2, 2]


def test_identical_candidates_are_zero_for_every_valid_paired_draw():
    y, b, clip, ones = sample()
    result = r.paired_bootstrap(y, r.scores_for(b, ones, ones), ['a', 'a', 'b', 'b', 'c', 'c'], reps=25)
    assert (result['attempted'], result['valid'], result['invalid']) == (25, 25, 0)
    for name in ('primary_blend_ones_minus_clip', 'secondary_rn_ones_minus_clip'):
        assert result['intervals'][name]['macro12'] == [0., 0.]
        assert all(v == [0., 0.] for v in result['intervals'][name]['targets'].values())


def test_any_single_class_target_invalidates_whole_macro_draw():
    y, b, clip, ones = sample()
    y[:, 0] = 0
    result = r.paired_bootstrap(y, r.scores_for(b, clip, ones), ['a', 'a', 'b', 'b', 'c', 'c'], reps=20)
    assert (result['valid'], result['invalid'], result['invalid_targets']['ACL']) == (0, 20, 20)
    assert all(x['macro12'] is None for x in result['intervals'].values())
    assert not r.aggregate(r.target_rows(y, r.scores_for(b, clip, ones)))['complete12']


def test_subgroup_and_bootstrap_never_call_rank_again(monkeypatch):
    y, b, clip, ones = sample()
    scores = r.scores_for(b, clip, ones)
    monkeypatch.setattr(r.pd.DataFrame, 'rank', lambda *a, **k: pytest.fail('Unexpected rerank'))
    take = np.array([True, True, False, False, True, True])
    assert r.aggregate(r.target_rows(y[take], {k: v[take] for k, v in scores.items()}))['complete12']
    assert r.paired_bootstrap(y, scores, ['a', 'a', 'b', 'b', 'c', 'c'], reps=5)['valid'] == 5


def test_bootstrap_is_deterministic_and_all_contrasts_share_draws():
    y, b, clip, ones = sample()
    scores = r.scores_for(b, clip, ones)
    a = r.paired_bootstrap(y, scores, list('abcdef'), reps=30)
    b = r.paired_bootstrap(y, scores, list('abcdef'), reps=30)
    assert a == b and a['same_draw_all_arms_and_targets']
    assert a['valid'] + a['invalid'] == 30
    assert a['invalid'] > 0


def test_binary_rule_float32_then_bankers_rounding():
    x = np.tile(np.array([.5, .500000001, .50001, .49], dtype=np.float64)[:, None], (1, 12))
    assert r.binary_labels(x)[:, 0].tolist() == [0, 0, 1, 0]


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.1, 1.1])
def test_invalid_probabilities_fail(bad):
    _, b, clip, ones = sample()
    ones[0, 0] = bad
    with pytest.raises(ValueError, match='probabilities'):
        r.scores_for(b, clip, ones)


def test_write_new_never_overwrites(tmp_path):
    p = tmp_path / 'report.json'
    r.write_new(p, {'original': True})
    with pytest.raises(FileExistsError):
        r.write_new(p, {'replacement': True})
    assert r.read(p) == {'original': True}


@pytest.mark.parametrize('path', ['../outside', '/absolute'])
def test_paths_cannot_escape_workspace(tmp_path, path):
    with pytest.raises(ValueError):
        r.contained(tmp_path, path)


def synthetic_contract(root):
    """Self-contained tiny corpus, never a reference to real workspace data."""
    def save(path, value):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        r.write_new(p, value)
        return p
    for path in ['tools/exp084_group_loss_report.py', 'tests/test_exp084_group_loss_report.py',
                 'tools/exp084_group_loss.py', 'training/runners/exp084_group_loss.py',
                 'training/src/raptor_train.py']:
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('# synthetic binding only\n')
    components = {'train': [], 'val': []}
    i = 0
    while len(components['train']) < 16 or len(components['val']) < 6:
        component = f'{i:064x}'
        key = 'val' if int.from_bytes(hashlib.sha256(('RN_GROUP_SCREEN_V1:' + component).encode()).digest(), 'big') % 5 == 2 else 'train'
        if len(components[key]) < (16 if key == 'train' else 6):
            components[key].append(component)
        i += 1
    ordered = {k: [f'{k}{j:02}' for j in range(len(v))] for k, v in components.items()}
    save('p/ordered.json', ordered)
    group_rows = []
    for key in components:
        for uid, component in zip(ordered[key], components[key]):
            group_rows.append({r.UID: uid, 'component_id': component, 'report_group_sha256': component,
                               'split': key, 'scanner_group': 'synthetic', 'field': '1.5',
                               'manufacturer': 'synthetic', 'cached_mask': 'all_nonzero'})
    r.pd.DataFrame(group_rows).to_csv(root / 'p/groups.csv', index=False)
    frame = r.pd.DataFrame([{r.UID: uid, **{t: .1 if i % 2 == 0 else .9 for t in r.TARGETS}}
                           for i, uid in enumerate(ordered['train'] + ordered['val'])])
    frame.to_csv(root / 'p/labels.csv', index=False)
    semantic = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    args = dict(epochs=20, fixed_epoch=20, bs=8, accum=1, seed=42, amp='fp16', eval_amp='fp16',
                runtime_policy='strict', window_policy='within-slot-valid', arch='synthetic')
    weights = [2.] * 12
    arms = {}
    for arm in r.ARMS:
        a = {**args, 'epochs': 8, 'fixed_epoch': 8} if arm == 'B' else args.copy()
        tag = arm
        arms[arm] = dict(id=arm, tag=tag, expected_args=a, expected_attempts=a['epochs'] * 2,
            loss_policy='ones_positive_term' if arm == 'RN_ones' else 'clipped_positive_term',
            actual_pos_weight=[1.] * 12 if arm == 'RN_ones' else weights, initial_state_sha256='init-B' if arm == 'B' else 'init-RN',
            outputs={k: 'o/' + tag + '/' + fn for k, fn in dict(summary='s.json', provenance='p.json', history='h.json', predictions='p.csv', checkpoint='m.pt').items()},
            logs='j/' + tag)
    paths = ['p/labels.csv', 'p/ordered.json', 'p/groups.csv', 'tools/exp084_group_loss_report.py',
             'tools/exp084_group_loss.py', 'training/runners/exp084_group_loss.py']
    bindings = {p: r.sha(root / p) for p in paths}
    env = {k: 'synthetic' for k in ('torch', 'cuda', 'cudnn', 'gpu', 'timm', 'numpy', 'pydicom', 'cv2')}
    m = dict(schema='exp084-group-r34-loss-v1', targets=r.TARGETS, arm_order=list(r.ARMS), arms=arms,
        execution=dict(new_fits=3, all_three_before_metrics=True),
        evaluation=dict(primary='macro12(blend_ones)-macro12(blend_clip)', weights=[.9, .1],
            rank='average tie rank/N on full V; no final/subgroup/bootstrap rerank', bootstrap_reps=500,
            bootstrap_seed=20261003, bootstrap_unit='component_id', no_ratio_epoch_target_search=True),
        split=dict(salt='RN_GROUP_SCREEN_V1', no_redraw=True),
        paths=dict(labels='p/labels.csv', eval_labels='p/labels.csv', fold_file='p/groups.csv'),
        label_hash=semantic, eval_label_hash=semantic, split_hash=bindings['p/groups.csv'],
        labels=dict(path='p/labels.csv', sha256=bindings['p/labels.csv'], semantic_sha256=semantic),
        groups=dict(path='p/groups.csv', sha256=bindings['p/groups.csv'], component_column='component_id',
                    subgroup_columns=['scanner_group', 'field', 'manufacturer', 'cached_mask']),
        ordered_ids=dict(path='p/ordered.json', sha256=bindings['p/ordered.json'],
                         train_sha256=r.uid_hash(ordered['train']), val_sha256=r.uid_hash(ordered['val'])),
        expected_ids=dict(train=16, val=6, train_components=16, val_components=6), excluded_uids=[],
        reference_counts=[dict(target=t, P=3, N=3) for t in r.TARGETS], macro12_valid=True,
        loss=dict(baseline_pos_weight=weights), source_bindings=bindings,
        training_code_sha256={'raptor_train.py': r.sha(root / 'training/src/raptor_train.py')},
        corpus_manifest_sha256='synthetic-corpus', expected_runtime={'runtime': {}, 'eval_runtime': {}}, environment=env)
    save('p/manifest.json', m)
    queue = [dict(tag=a, command=['python', 'synthetic', a]) for a in r.ARMS]
    save('p/queue.json', queue)
    return m, ordered, queue


def summary_for(m, arm, manifest_sha):
    spec = m['arms'][arm]
    ep = spec['expected_args']['epochs']
    provenance = dict(code_sha256=m['training_code_sha256'], initial_state_sha256=spec['initial_state_sha256'],
        train_ids_sha256=m['ordered_ids']['train_sha256'], eval_ids_sha256=m['ordered_ids']['val_sha256'],
        train_ids_ordered_sha256=m['ordered_ids']['train_sha256'], eval_ids_ordered_sha256=m['ordered_ids']['val_sha256'],
        corpus_manifest_sha256=m['corpus_manifest_sha256'], checkpoint_selection='fixed-final',
        sampling='sha256(seed,epoch,uid,stream)-v1', gold_access='disabled', loss_policy=spec['loss_policy'],
        actual_pos_weight=spec['actual_pos_weight'], baseline_pos_weight=m['loss']['baseline_pos_weight'],
        metric_access='deferred_until_all_three_complete', protocol_sha256=manifest_sha,
        adapter_sha256=m['source_bindings']['training/runners/exp084_group_loss.py'],
        contract_helper_sha256=m['source_bindings']['tools/exp084_group_loss.py'], runtime={}, eval_runtime={})
    history = [dict(epoch=i, loss=.5, optimizer_updates=(i + 1) * 2, skipped_updates=0,
                    selection=float(i), oof_auc=None, gold_auc=None) for i in range(ep)]
    return dict(provenance=provenance, environment=m['environment'], args=spec['expected_args'],
        gold=0, gold_access='disabled', metric_access='deferred_until_all_three_complete',
        loss_policy=spec['loss_policy'], arm=arm, selected_checkpoint='best', best_epoch=ep - 1,
        selected_epoch_1based=ep, max_epochs=ep, train=16, val=6, rows=6, no_oof_epoch_selection=True,
        weak_oof_macro_auc=None, oof_per_target=None, label_hash=m['label_hash'], eval_label_hash=m['eval_label_hash'],
        split_hash=m['split_hash'], expected_attempts=spec['expected_attempts'], optimizer_updates=spec['expected_attempts'],
        skipped_updates=0, optimizer_skip_fraction=0., history=history,
        results={'best': {'oof_rows': 6, 'metric_access': 'deferred_until_all_three_complete'}}, total_seconds=1.)


def test_prepare_freezes_paths_counts_and_hashes_without_label_values(tmp_path, monkeypatch):
    m, _, _ = synthetic_contract(tmp_path)
    real = r.pd.read_csv
    def protected(path, *a, **k):
        if Path(path).name == 'labels.csv':
            pytest.fail('Prepare read actual label values')
        return real(path, *a, **k)
    monkeypatch.setattr(r.pd, 'read_csv', protected)
    result = r.prepare('p/manifest.json', r.sha(tmp_path / 'p/manifest.json'), tmp_path)
    assert result['paths']['RN_clip']['outputs'] == m['arms']['RN_clip']['outputs']
    assert result['reference_counts'] == m['reference_counts']
    assert 'p/queue.json' in result['source_bindings']
    with pytest.raises(FileExistsError):
        r.prepare('p/manifest.json', r.sha(tmp_path / 'p/manifest.json'), tmp_path)


def test_changed_frozen_source_rejected_before_outcomes(tmp_path):
    synthetic_contract(tmp_path)
    (tmp_path / 'tools/exp084_group_loss.py').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        r.prepare('p/manifest.json', r.sha(tmp_path / 'p/manifest.json'), tmp_path)


def test_missing_third_completion_rejected_before_summary_reads(tmp_path, monkeypatch):
    m, _, _ = synthetic_contract(tmp_path)
    reg = {'paths': {a: dict(outputs=m['arms'][a]['outputs'], logs=m['arms'][a]['logs']) for a in r.ARMS}}
    monkeypatch.setattr(r, 'read', lambda *a: pytest.fail('Read an artifact before all complete'))
    with pytest.raises(ValueError, match='All three completed'):
        r.require_completed(tmp_path, reg, m)


def test_completion_rejects_failed_marker_even_with_success(tmp_path, monkeypatch):
    m, _, _ = synthetic_contract(tmp_path)
    reg = {'paths': {a: dict(outputs=m['arms'][a]['outputs'], logs=m['arms'][a]['logs']) for a in r.ARMS}}
    job = tmp_path / m['arms']['RN_clip']['logs']
    job.mkdir(parents=True)
    (job / 'SUCCESS').write_text('exit 0 at 2026-10-04T01:00:00')
    (job / 'FAILED').write_text('failed')
    monkeypatch.setattr(r, 'read', lambda *a: pytest.fail('Read failed run'))
    with pytest.raises(ValueError, match='All three completed'):
        r.require_completed(tmp_path, reg, m)


def test_execution_exact_recipe_and_realized_budget(tmp_path):
    m, _, _ = synthetic_contract(tmp_path)
    s = summary_for(m, 'RN_clip', 'manifest-sha')
    assert r.validate_execution(s, s['provenance'], m, 'RN_clip', 'manifest-sha')['optimizer_updates'] == 40
    changed = deepcopy(s)
    changed['provenance']['initial_state_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='provenance'):
        r.validate_execution(changed, changed['provenance'], m, 'RN_clip', 'manifest-sha')
    changed = deepcopy(s)
    changed['history'][0]['oof_auc'] = .7
    with pytest.raises(ValueError, match='epoch history'):
        r.validate_execution(changed, changed['provenance'], m, 'RN_clip', 'manifest-sha')


def test_skip_failure_blocks_before_scoring(tmp_path):
    m, _, _ = synthetic_contract(tmp_path)
    s = summary_for(m, 'RN_clip', 'manifest-sha')
    s.update(optimizer_updates=39, skipped_updates=1, optimizer_skip_fraction=.025)
    with pytest.raises(ValueError, match='skip fraction'):
        r.validate_execution(s, s['provenance'], m, 'RN_clip', 'manifest-sha')


def test_protocol_binds_labels_ratio_and_fixed_epochs(tmp_path):
    m, _, _ = synthetic_contract(tmp_path)
    r.validate_protocol(m)
    for mutate in (lambda x: x['evaluation'].update(weights=[.8, .2]),
                   lambda x: x['arms']['RN_ones']['expected_args'].update(fixed_epoch=19),
                   lambda x: x.update(eval_label_hash='changed')):
        changed = deepcopy(m)
        mutate(changed)
        with pytest.raises(ValueError):
            r.validate_protocol(changed)


def test_full_synthetic_prepare_complete_evaluate_once(tmp_path):
    m, ordered, queue = synthetic_contract(tmp_path)
    msha = r.sha(tmp_path / 'p/manifest.json')
    r.prepare('p/manifest.json', msha, tmp_path)
    regpath = r.EVIDENCE + '/registration.json'
    regsha = r.sha(tmp_path / regpath)
    y, b, clip, ones = sample()
    values = dict(B=b, RN_clip=clip, RN_ones=ones)
    for arm, item in zip(r.ARMS, queue):
        spec = m['arms'][arm]
        summary = summary_for(m, arm, msha)
        for key, value in [('summary', summary), ('provenance', summary['provenance']), ('history', summary['history'])]:
            target = tmp_path / spec['outputs'][key]
            target.parent.mkdir(parents=True, exist_ok=True)
            r.write_new(target, value)
        (tmp_path / spec['outputs']['checkpoint']).write_text('synthetic checkpoint; never loaded')
        df = r.pd.DataFrame(values[arm], columns=r.TARGETS)
        df.insert(0, r.UID, ordered['val'])
        df.to_csv(tmp_path / spec['outputs']['predictions'], index=False)
        job = tmp_path / spec['logs']
        job.mkdir(parents=True)
        (job / 'SUCCESS').write_text('exit 0 at 2026-10-04T01:00:00\n')
        r.write_new(job / 'job.json', item)
    result = r.run(regpath, regsha, tmp_path)
    assert result['complete12']
    assert result['aggregate']['contrasts']['primary_blend_ones_minus_clip']['macro12'] == pytest.approx(1.)
    assert result['attempted'] == result['valid'] + result['invalid'] == 500
    assert (tmp_path / r.EVIDENCE / 'evaluation-v1/SUCCESS.json').is_file()
    with pytest.raises(FileExistsError, match='One-shot'):
        r.run(regpath, regsha, tmp_path)
    with pytest.raises(ValueError, match='registration hash'):
        r.run(regpath, '0' * 64, tmp_path)
