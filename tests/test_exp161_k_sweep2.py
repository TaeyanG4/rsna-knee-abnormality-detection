"""CPU-only contract tests for EXP161. No CUDA or network calls."""
import json

import numpy as np
import pytest
from scipy.stats import rankdata

from tools import exp161_k_sweep2 as sweep


def test_served_h1_reduction_has_five_heads_two_views_and_outer_rank():
    # Distinct heads/views make accidental probability mean or single-head use visible.
    raw = np.array([
        [[[.1, .9], [.2, .8], [.8, .2], [.3, .7], [.4, .6]],
         [[.2, .8], [.3, .7], [.7, .3], [.4, .6], [.5, .5]]],
        [[[.9, .1], [.8, .2], [.2, .8], [.7, .3], [.6, .4]],
         [[.8, .2], [.7, .3], [.3, .7], [.6, .4], [.5, .5]]],
        [[[.5, .5], [.6, .4], [.4, .6], [.5, .5], [.9, .1]],
         [[.6, .4], [.5, .5], [.5, .5], [.8, .2], [.1, .9]]],
    ], dtype=np.float32)
    # The production shape always has 12 targets; repeat synthetic targets.
    raw = np.tile(raw, (1, 1, 1, 6))
    per_view = np.stack([np.mean([rankdata(raw[:, v, h], axis=0, method='average') / 3
                                  for h in range(5)], axis=0) for v in range(2)])
    expected = rankdata(per_view.mean(axis=0), axis=0, method='average') / 3
    assert np.array_equal(sweep.h1_served_oof(raw), expected)
    with pytest.raises(ValueError, match='H1 raw'):
        sweep.h1_served_oof(raw[:, :, :4])


def test_registered_accuracy_gates():
    pairs = {'jev_hybrid': {'delta': .0014}, 'aquila_v1': {'delta': .0012},
             'dread': {'delta': -.0019}}
    assert sweep.accuracy_gate(pairs, [-.0001, .002], require_ci=False)['accuracy_pass']
    assert not sweep.accuracy_gate(pairs, [-.0001, .002], require_ci=True)['accuracy_pass']
    assert sweep.accuracy_gate(pairs, [.00001, .002], require_ci=True)['accuracy_pass']
    pairs['dread']['delta'] = -.00201
    assert not sweep.accuracy_gate(pairs, [.00001, .002], require_ci=True)['accuracy_pass']


def test_timing_summary_keeps_direct_model_samples_separate():
    receipts = []
    for worker, folds in enumerate(((0, 2, 4), (1, 3))):
        fs = {}
        for family in sweep.FAMILIES:
            for fold in folds:
                rows = [888, 870, 870, 860, 855][fold]
                fs[f'{family}_f{fold}'] = {
                    'rows': rows, 'wall_seconds': 200.0,
                    'shared_encode_seconds': 20.0,
                    'head_seconds': {str(k): 1.0 for k in sweep.K_VALUES},
                    'direct': {str(k): {'studies': sweep.DIRECT_N,
                                        'gpu_seconds': 8.0, 'wall_seconds': 10.0}
                               for k in sweep.K_VALUES}}
        receipts.append({'folds': fs})
    assert sum(receipts[w]['folds'][f'aug1_f{f}']['rows']
               for w, folds in enumerate(((0, 2, 4), (1, 3))) for f in folds) == 4343
    t = sweep.timing_summary(receipts)
    assert t['h1']['direct']['94']['studies'] == 40
    assert t['h1']['direct']['94']['gpu_seconds_per_model_study'] == 1.0
    assert t['h1']['direct']['94']['wall_seconds_per_model_study'] == 1.25
    assert t['h1']['direct_probe_extra_wall_seconds'] == 150.0


def test_incomplete_collection_fails_before_copy(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    (source / 'exp161-status.json').write_text(json.dumps({'state': 'INCOMPLETE'}), encoding='utf-8')
    with pytest.raises(ValueError, match='incomplete'):
        sweep.collect(source, destination=target, stage_dir=tmp_path / 'stage', root=tmp_path)
    assert not target.exists()


def test_notebook_and_push_plan_are_offline_and_bind_reused_h1(tmp_path):
    nb = sweep.notebook('a' * 64)
    cell = nb['cells'][1]['source']
    assert "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'" in cell
    assert cell.index("os.environ['CUBLAS_WORKSPACE_CONFIG']") < cell.index('import hashlib')
    assert 'knee-exp154-b3-h1-assets' in cell
    assert "selected = [e['path'] for e in m['h1_entries']]" in cell
    stage = tmp_path / 'stage'
    (stage / 'dataset').mkdir(parents=True)
    (stage / 'kernel').mkdir()
    (stage / 'dataset/manifest.json').write_text('{}', encoding='utf-8')
    (stage / 'kernel/run.ipynb').write_text('{}', encoding='utf-8')
    plan = sweep.push_plan(stage)
    assert plan['status'] == 'PLAN_ONLY_NO_NETWORK'
    assert any('datasets create -p .' in x for x in plan['commands'])
    assert any('-m tools.exp161_k_sweep2 screen' in x for x in plan['commands'])


def test_serving_time_gate_requires_full_cohort_and_reserve(tmp_path):
    receipt = tmp_path / 'timing.json'
    receipt.write_text(json.dumps({
        'schema': 'exp161-full-serving-time-v1', 'units': 'seconds',
        'cohort': 'full competition test', 'releases': {
            'B3_H1': {'94': 32000, 'reserve': 300, 'source': 'full-run.json'},
            'H1': {'94': 31000, 'reserve': 300, 'source': 'full-run.json'}}}), encoding='utf-8')
    assert sweep.serving_time_gate(receipt, 94, 'h1')['pass_']
    data = json.loads(receipt.read_text(encoding='utf-8'))
    data['releases']['B3_H1']['reserve'] = 500
    receipt.write_text(json.dumps(data), encoding='utf-8')
    assert not sweep.serving_time_gate(receipt, 94, 'h1')['pass_']
