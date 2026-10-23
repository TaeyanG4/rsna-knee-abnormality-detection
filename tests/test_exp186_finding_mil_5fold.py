# Compatibility tests v2.0.0: retired launch guards, archived scientific parity, collected readouts.
import ast
import importlib
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from tools import exp169_bundle_screen as bundle
from tools import exp186_finding_mil_5fold as exp


def test_registered_queue_and_cap():
    assert exp.VERSION == '2.0.0' and exp.HISTORICAL_VERSION == '1.0.1'
    assert exp.REMOTE_ROOT == '/workspace/exp186b'
    assert exp.REMOTE_RESULTS == '/workspace/exp186b-results'
    assert exp.FOLDS == (1, 2, 3, 4)
    assert exp.ARM_CAP * len(exp.FOLDS) == exp.TOTAL_CAP == 14400
    assert [row['tag'] for row in exp.queue()] == [exp.tag(f) for f in exp.FOLDS]
    assert all(not row['continue_on_failure'] for row in exp.queue())
    assert all(row['tag'].startswith('EXP186B_') for row in exp.queue())
    with pytest.raises(ValueError):
        exp.tag(0)


def test_historical_package_recipe_remains_authenticated():
    overlay = exp.ROOT / '.local/exp186b-overlay'
    m = exp.validate(overlay)
    assert m['version'] == exp.HISTORICAL_VERSION
    assert m['files'][exp.TOOL]['sha256'] == exp.sha(overlay / exp.TOOL)
    assert m['files'][exp.TOOL]['sha256'] != exp.sha(exp.ROOT / exp.TOOL)


@pytest.mark.parametrize('entry,args', [('prepare', (None, None)), ('runner_source', ()),
    ('runner_args', (1,)), ('admit', ()), ('run', (1,)), ('vast_stage', (None,))])
def test_fresh_execution_retired_without_writes(tmp_path, entry, args):
    with pytest.raises(ValueError, match='retired'):
        getattr(exp, entry)(*args)
    assert not list(tmp_path.iterdir())
    assert '__getattr__' not in vars(exp)


def test_overlay_tools_namespace_resolves_inherited_vast_module(tmp_path):
    overlay = tmp_path / 'overlay'
    inherited = tmp_path / 'exp179'
    (overlay / 'tools').mkdir(parents=True)
    (inherited / 'tools').mkdir(parents=True)
    (overlay / exp.TOOLS_INIT).write_bytes(exp.TOOLS_INIT_BYTES)
    (inherited / 'tools/__init__.py').write_bytes(b'')
    (inherited / 'tools/exp179_multiscale.py').write_text('SENTINEL = 186\n')
    result = subprocess.run(
        [sys.executable, '-c',
         'from tools import exp179_multiscale; assert exp179_multiscale.SENTINEL == 186'],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': os.pathsep.join(
            (str(overlay), str(inherited)))}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_runtime_uses_recorded_vast_environment_not_local_registration():
    vast = {'gpu': 'NVIDIA GeForce RTX 3090', 'cv2': '4.10.0', 'torch': '2.5.1'}
    manifest = {'fold0': {'hostname': 'vast-node', 'environment': vast},
                'controls': {'1': {'hostname': 'vast-node', 'environment': vast}}}
    actual = {'cv2': '4.10.0', 'torch': '2.5.1'}
    exp.check_vast_environment(manifest, 1, 'vast-node', vast['gpu'], actual)
    with pytest.raises(ValueError):
        exp.check_vast_environment(manifest, 1, 'vast-node', vast['gpu'],
                                   {'cv2': '5.0.0', 'torch': '2.5.1'})
    with pytest.raises(ValueError):
        exp.check_vast_environment(manifest, 1, 'vast-node', vast['gpu'],
                                   {'cv2': '4.10.0'})


def test_pooled_fold0_uses_mixed_control():
    manifest = {'fold0': {'control_path': exp.mil.control_path(42)},
                'controls': {'1': {'path': 'exp183/fold1'}}}
    assert exp.paired_control_path(0, manifest).as_posix().endswith(
        'EXP179_MULTISCALE/outputs/M42/results/EXP179_M42_F0')
    assert exp.paired_control_path(1, manifest).as_posix().endswith('exp183/fold1')


def test_archived_windows_control_paths_are_portable_but_cannot_escape():
    manifest = exp.validate(exp.ROOT / '.local/exp186b-overlay')
    assert all(exp.paired_control_path(fold, manifest).is_dir() for fold in range(5))
    for bad in ('..\\outside', 'C:\\outside', '\\\\server\\share', '/outside'):
        manifest['controls']['1']['path'] = bad
        with pytest.raises(ValueError, match='unsafe relative path'):
            exp.paired_control_path(1, manifest)


def test_score_arrays_uses_all_folds_and_fixed_bundle(monkeypatch):
    class Evaluator:
        @staticmethod
        def pct_rank(value):
            return np.asarray(value) / 10

        @staticmethod
        def auc_targets(y, value, mask, minimum):
            return np.mean(value, axis=0)

        @staticmethod
        def macro(value):
            return np.mean(value)

    values = {}
    for fold in range(5):
        values[fold] = {name: np.full((2, 12), fold * 10 + offset, dtype=float)
                        for name, offset in (('candidate', 1), ('control', 2),
                                             ('rad42', 3), ('rad43', 5), ('dino', 7))}
    refs = {name: (np.zeros((10, 12)), np.ones((10, 12), dtype=bool), 1)
            for name in ('jev', 'consensus', 'dread', 'aquila')}
    calls = []

    def fake_bundle(parts):
        calls.append(('bundle', {k: v.copy() for k, v in parts.items()}))
        return sum(parts.values()) / 3, {}

    def fake_pair(new, old, refs, draws, evaluator):
        calls.append(('pair', new.copy(), old.copy()))
        return {'delta': float(np.mean(new-old))}

    def fake_weak(new, old, refs, draws, evaluator):
        calls.append(('weak', new.copy(), old.copy()))
        return {'status': 'REVIEW_IMAGE_READ'}

    monkeypatch.setattr(bundle, 'bundle_of', fake_bundle)
    monkeypatch.setattr(exp.base, 'paired_jev_consensus', fake_pair)
    monkeypatch.setattr(exp.mil, 'weak_target_readout', fake_weak)
    result = exp.score_arrays(values, refs, [np.arange(10)], Evaluator())
    assert result['status'] == 'REVIEW_IMAGE_READ'
    assert len(result['per_fold_direct']) == 5
    assert [x[0] for x in calls] == ['bundle', 'bundle', 'pair', 'weak', 'pair']
    assert np.array_equal(calls[2][1][:2], values[0]['candidate'])
    assert np.array_equal(calls[2][1][-2:], values[4]['candidate'])
    assert np.array_equal(calls[3][2][-2:], values[4]['control'])
    assert np.allclose(calls[0][1]['l2'][:2], 0.4)
    assert np.allclose(calls[1][1]['dino'][-2:], 4.7)
