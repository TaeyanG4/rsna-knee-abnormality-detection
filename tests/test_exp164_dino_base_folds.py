"""CPU-only EXP164 package, fold and screen wiring checks."""
import ast
import json
from pathlib import Path

import pytest

from tools import exp164_dino_base_folds as e


def test_runner_guard_exports_every_referenced_attribute():
    source = e.runner_source()
    tree = ast.parse(source)
    names = {node.attr for node in ast.walk(tree)
             if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
             and node.value.id == 'guard'}
    missing = sorted(name for name in names if not hasattr(e, name))
    assert missing == []
    assert 'from tools import exp164_dino_base_folds as guard' in source
    assert 'from tools import exp160_dino_partner as guard' not in source


def test_sorted_rosters_preserve_registered_membership():
    rosters = e.roster()
    for fold in range(1, 5):
        source = rosters[fold][1]
        packaged = json.loads((e.ROOT / e.roster_path(fold)).read_text())
        assert packaged['train'] == sorted(source['train'])
        assert packaged['val'] == sorted(source['val'])
        assert set(packaged['train']) == set(source['train'])
        assert set(packaged['val']) == set(source['val'])


def test_one_serial_queue_and_fold_specific_steps():
    q = e.queue()
    assert [job['tag'] for job in q] == [e.tag(f) for f in range(1, 5)]
    assert [job['command'][-1] for job in q] == ['1', '2', '3', '4']
    assert [((4343-e.COUNTS[f]) // 8)*20 for f in range(1, 5)] == [8680, 8680, 8680, 8700]


def test_fold_local_weights_do_not_require_fold_zero_vector():
    import numpy as np
    vector = np.full(12, 2, dtype=np.float32)
    assert np.array_equal(e.loss_weights(vector, {}, {'loss_policy': 'clipped_positive_term',
                                                       'actual_pos_weight': None}), vector)
    with pytest.raises(ValueError):
        e.loss_weights(vector.astype(np.float64), {}, {'loss_policy': 'clipped_positive_term',
                                                       'actual_pos_weight': None})


def test_package_preparation_and_sorted_packaged_rosters(tmp_path):
    overlay = tmp_path / 'exp164-overlay'
    result = e.prepare(overlay)
    assert result['status'] == 'PREPARED_NOT_LAUNCHED'
    p = e.validate_manifest(overlay / 'manifest.json', overlay)
    assert p['serial_folds'] == [1, 2, 3, 4]
    for fold in range(1, 5):
        q = json.loads((overlay / f'protocols/fold{fold}.json').read_text())
        roster = json.loads((overlay / q['ordered_ids']['path']).read_text())
        assert roster['train'] == sorted(roster['train'])
        assert roster['val'] == sorted(roster['val'])
        assert q['arm']['actual_pos_weight'] is None
        assert q['expected_updates'] == ((4343-e.COUNTS[fold]) // 8) * 20
    assert (overlay / 'training/src/raptor_radimagenet.py').exists() is False
    assert 'training/src/raptor_radimagenet.py' in p['files']
    assert 'tools/process_resource_receipt.py' in p['files']
    assert (overlay / 'tools/exp082_r34_loss.py').exists()


def test_incomplete_collection_cannot_pass(tmp_path):
    output = tmp_path / 'outputs'
    output.mkdir()
    (output / 'collection.json').write_text(json.dumps({'schema': e.SCHEMA,
        'status': 'INCOMPLETE', 'completed_folds': [1, 2, 3],
        'incomplete_folds': {'4': 'NO_VALIDATED_SUCCESS'}}))
    result = e.screen(output, tmp_path / 'screen')
    assert result['verdict'] == 'INCOMPLETE'


def test_registered_gate_uses_h1_weight_point_one_and_ci_lower_bound():
    pairs = {'radr50': {'0.1': {'primary_mean_delta': .1, 'primary_mean_ci95': [.05, .15]}},
             'h1': {'0.1': {'primary_mean_delta': .001, 'primary_mean_ci95': [-.001, .003]},
                    '0.2': {'primary_mean_delta': .02, 'primary_mean_ci95': [.01, .03]}}}
    assert not e.screen_gate(pairs)
    pairs['h1']['0.1']['primary_mean_ci95'][0] = .0001
    assert e.screen_gate(pairs)


def test_ranks_are_normalised_inside_each_fold():
    import numpy as np

    class Evaluator:
        @staticmethod
        def pct_rank(values):
            order = np.argsort(np.argsort(values[:, 0], kind='stable'), kind='stable') + 1
            return (order / len(values))[:, None]

    parts = {f: np.array([[f*100 + 2], [f*100 + 1]], dtype=float) for f in range(5)}
    ranked = e.rank_per_fold(parts, Evaluator)
    assert ranked[:, 0].tolist() == [1.0, .5] * 5


def test_plan_is_single_detached_queue_and_forward_slash_path():
    plan = e.vast_plan('/workspace/rsna-exp164-dino-base-folds-v1')
    assert 'setsid nohup python tools/gpu_queue.py queue.json --foreground' in plan
    assert 'max_gpu_seconds_per_fold=10800' in plan
    assert '\\' not in plan
    with pytest.raises(ValueError):
        e.vast_plan(r'H:\bad')
