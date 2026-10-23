# Compatibility tests v2.0.0: retired launch guards, archived scientific parity, collected readouts.
"""EXP191B CPU contracts. Never read report text or restricted relabel data."""
import copy
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from tools import exp191b_pfoa_softprior as exp


@pytest.fixture(scope='module')
def package():
    out = exp.ROOT / '.local/exp191b-overlay-v2'
    exp.validate(out)
    return out


def test_real_projection_exact_bytes_without_relabel():
    old = (exp.ROOT / exp.parent.L2).read_bytes()
    pa = (exp.ROOT / exp.parent.LABELS['PA']).read_bytes()
    data, qa = exp.project_labels(old, pa)
    assert qa['rows'] == 4347 and qa['changed_cells'] == 717
    assert exp.parent.digest_bytes(data) == 'f258c7a9c584a9e4582562ac650dc62d7eb5cc7b4492e037339ec91ef8667d66'
    rows = lambda b: list(csv.reader(io.StringIO(b.decode(), newline='')))
    control, prior, candidate = map(rows, (old, pa, data))
    pf = control[0].index('PF OA')
    for a, b, c in zip(control[1:], prior[1:], candidate[1:]):
        assert c[pf] == b[pf]
        assert c[:pf] + c[pf+1:] == a[:pf] + a[pf+1:]


@pytest.mark.parametrize('fault', ['header', 'roster', 'order', 'duplicate', 'dialect'])
def test_projection_rejects_invalid_input(fault):
    header = [exp.parent.fill.UID, *exp.TARGETS]
    rows = [['A', *['0.1']*12], ['B', *['0.2']*12]]
    old = exp.parent._text(header, rows)
    modified = copy.deepcopy(rows)
    h = header[:]
    if fault == 'header': h[-1] = 'wrong'
    if fault == 'roster': modified.pop()
    if fault == 'order': modified.reverse()
    if fault == 'duplicate':
        rows[1][0] = 'A'
        old = exp.parent._text(header, rows)
    if fault == 'dialect': old = old.replace(b'\n', b'\r\n')
    with pytest.raises((ValueError, SystemExit)):
        exp.project_labels(old, exp.parent._text(h, modified))


def test_explicit_history_does_not_modify_parent_namespace():
    from dataclasses import FrozenInstanceError
    assert exp.parent.ARMS == ('CONTROL', 'P', 'PA')
    assert exp.parent.tag('PA', 42) == 'EXP191_SOFTPRIOR_PA_S42_F0'
    config = exp.historical_config()
    assert config.tag(exp.ARM, 42) == 'EXP191B_PFOA_SOFTPRIOR_PA_PFOA_S42_F0'
    assert config.labels == {exp.ARM: exp.LABEL}
    assert exp.parent.historical_config().labels == exp.parent.LABELS
    assert 'core' not in vars(exp) and '__getattr__' not in vars(exp)
    with pytest.raises(FrozenInstanceError): config.version = 'wrong'


@pytest.mark.parametrize('arm,seed', [('CONTROL', 42), ('PA', 42), ('PA_PFOA', 44)])
def test_no_control_or_extra_seed_can_be_queued(arm, seed):
    with pytest.raises((ValueError, SystemExit)):
        exp.queue([(arm, seed)])


def test_control_reuse_is_mandatory(monkeypatch):
    monkeypatch.setattr(exp.parent, 'control_reuse', lambda *args: {'retrain': True, 'path': 'wrong'})
    with pytest.raises((ValueError, SystemExit), match='reuse failed'):
        exp.controls()


def test_exact_recipe_and_runner(package, monkeypatch):
    from tools import exp166_convnext_t as source
    real_read = source.read
    monkeypatch.setattr(source, 'read', lambda p: real_read(p if Path(p).exists() else
        exp.parent.mil.inherited_path(package, str(Path(p).relative_to(package)))))
    m = exp.validate(package)
    assert m['tasks'] == [{'arm': exp.ARM, 'seed': s} for s in exp.SEEDS]
    assert 'caps_seconds' not in m
    runner = (package / exp.RUNNER).read_text(encoding='utf-8')
    original = (exp.ROOT / exp.SOURCE_OVERLAY / exp.parent.RUNNER).read_text(encoding='utf-8')
    assert runner.replace('from tools import exp191b_pfoa_softprior as guard',
                          'from tools import exp191_softprior as guard') == original
    for s in exp.SEEDS:
        q = exp.read(package / exp.protocol_path(exp.ARM, s))
        control = exp.read(package / f'protocols/S{s}.json')
        for k in ('common_args', 'initial_state', 'initial_state_sha256', 'init_weights', 'exp179', 'exp185'):
            assert q[k] == control[k]
        fold = dict(q['folds']['0']); fold.pop('tag')
        base_fold = dict(control['folds']['0']); base_fold.pop('tag')
        assert fold == base_fold
        assert q['exp191']['frozen_pos_weight']
        assert q['paths']['labels'] == exp.LABEL
        assert q['paths']['eval_labels'] == exp.parent.L2



@pytest.mark.parametrize('fault', ['recipe', 'control', 'task', 'runner', 'label'])
def test_manifest_tamper_fails(package, tmp_path, fault):
    out = tmp_path / 'tamper'
    shutil.copytree(package, out)
    m = exp.read(out / 'manifest.json')
    if fault == 'control': m['controls']['42']['prediction_sha256'] = 'bad'
    elif fault == 'task': m['tasks'][0]['arm'] = 'CONTROL'
    else:
        name = {'recipe': exp.protocol_path(exp.ARM, 42), 'runner': exp.RUNNER, 'label': exp.LABEL}[fault]
        if fault == 'recipe':
            q = exp.read(out / name); q['common_args']['epochs'] = 1
            exp.write(out / name, q)
            m['runs'][exp.tag(exp.ARM, 42)]['sha256'] = exp.sha(out / name)
        else:
            with (out / name).open('ab') as f: f.write(b'\n')
        m['files'][name] = exp.parent._entry(out / name)
    exp.write(out / 'manifest.json', m)
    with pytest.raises((ValueError, SystemExit)):
        exp.validate(out)


@pytest.mark.parametrize('entry,args', [('prepare', (None,)), ('admit', (None, True)),
    ('run', (42,)), ('smoke', (None, None)), ('vast_stage', (None,)),
    ('validate_arguments', (None, None))])
def test_retired_execution_is_fail_closed(tmp_path, entry, args):
    with pytest.raises(ValueError, match='retired'):
        getattr(exp, entry)(*args)
    assert not list(tmp_path.iterdir())


def fake_case(deltas):
    control = {s: np.full((2, 12), .5) for s in exp.SEEDS}
    candidates = {s: control[s]+np.array(deltas[s]) for s in exp.SEEDS}
    y = np.tile(np.array([0., 1.])[:, None], (1, 12))
    refs = {r: (y, np.ones_like(y, dtype=bool), 1) for r in ('jev', 'consensus', 'dread', 'aquila')}
    evaluator = SimpleNamespace(auc_targets=lambda y, p, m, n: p[0])
    return exp.paired_readout(candidates, control, refs, [np.arange(2)]*300, evaluator)


@pytest.mark.parametrize('case', ['success', 'threshold', 'one_seed_negative', 'macro', 'synovitis', 'dread', 'nan'])
def test_preregistered_gates(case):
    pf, syn = exp.TARGETS.index('PF OA'), exp.TARGETS.index('Synovitis')
    deltas = {s: np.zeros(12) for s in exp.SEEDS}
    for d in deltas.values(): d[pf] = .004
    if case == 'threshold':
        for d in deltas.values(): d[pf] = .0019
    elif case == 'one_seed_negative': deltas[43][pf] = -.0001; deltas[42][pf] = .01
    elif case == 'macro':
        for d in deltas.values(): d[:] = -.002; d[pf] = .004
    elif case == 'synovitis':
        for d in deltas.values(): d[syn] = -.011; d[pf] = .05
    elif case == 'dread':
        for d in deltas.values(): d[:] = -.005; d[pf] = .004
    elif case == 'nan':
        # Undefined AUCs, despite valid prediction arrays.
        from tools import exp115_vlm_eval as ev
        y = np.zeros((2, 12)); refs = {r: (y, np.ones_like(y, dtype=bool), 1) for r in ('jev', 'consensus', 'dread', 'aquila')}
        p = {s: np.full((2, 12), .5) for s in exp.SEEDS}
        assert exp.paired_readout(p, p, refs, [np.arange(2)]*300, ev)['status'] == 'HOLD'
        return
    result = fake_case(deltas)
    assert result['status'] == ('SUCCESS' if case == 'success' else 'HOLD')
    if case == 'synovitis':
        assert result['harm_guard']['macro_ge_minus_0_001']
        assert not result['harm_guard']['other_targets_ge_minus_0_010']
    if case == 'dread': assert not result['harm_guard']['dread_macro_ge_minus_0_003']
    if case == 'success':
        assert result['pf_oa_component_bootstrap']['jev']['valid_draws'] == 300
        assert result['per_seed_pf_oa_component_bootstrap']['43']['jev']['ci95'] == pytest.approx([.004, .004])


def test_real_auc_seed_average_is_not_prediction_ensemble():
    from tools import exp115_vlm_eval as ev
    rng = np.random.default_rng(20261003)
    y = np.tile(np.array([0., 1.]*10)[:, None], (1, 12))
    mask = np.ones_like(y, dtype=bool)
    refs = {r: (y, mask, 1) for r in ('jev', 'consensus', 'dread', 'aquila')}
    p = {s: rng.random((20, 12)) for s in exp.SEEDS}
    c = {s: rng.random((20, 12)) for s in exp.SEEDS}
    result = exp.paired_readout(p, c, refs, [np.arange(20)]*300, ev)
    expected = np.mean([ev.auc_targets(y, p[s], mask, 1) for s in exp.SEEDS], axis=0)
    observed = np.array(list(result['seed_average']['jev']['candidate'].values()))
    np.testing.assert_allclose(observed, expected)
    assert not np.allclose(observed, ev.auc_targets(y, (p[42]+p[43])/2, mask, 1))
    assert exp.paired_readout(p, c, refs, [np.arange(20)]*300, ev) == result








def test_export_collect_roundtrip_and_receipt_guards(tmp_path, monkeypatch):
    root, results = tmp_path / 'root', tmp_path / 'results'
    root.mkdir(); results.mkdir()
    tag = exp.tag(exp.ARM, 42)
    run = results / tag; run.mkdir()
    (run / 'output.bin').write_bytes(b'fixture outputs, not a training run')
    exp.write(run / 'output-inventory.json', {})
    exp.write(root / 'manifest.json', {'fixture': True})
    hashes = {'output.bin': exp.sha(run / 'output.bin')}
    exp.write(root / f'validated-{tag}.json', dict(status='VALIDATED_SUCCESS', arm=exp.ARM,
        seed=42, elapsed_seconds=8000, output_sha256=hashes, manifest_sha256=exp.sha(root / 'manifest.json')))
    monkeypatch.setattr(exp, 'REMOTE_ROOT', str(root.resolve()))
    monkeypatch.setattr(exp, 'REMOTE_RESULTS', str(results))
    monkeypatch.setattr(exp.parent.base, 'REQUIRED_OUTPUTS', {'output.bin'})
    monkeypatch.setattr(exp, 'validate', lambda *a, **k: {})
    monkeypatch.setattr(exp.parent, 'verify_run', lambda *a, **k: hashes)
    archive = tmp_path / 'result.tar'
    assert exp.export(42, archive, root)['status'] == 'EXPORTED_VALIDATED'
    assert exp.collect(archive, root, tmp_path / 'collected', 42)['status'] == 'VERIFIED'
    with pytest.raises((KeyError, ValueError, SystemExit)):
        exp.collect(archive, root, tmp_path / 'wrong-seed', 43)
    bad = tmp_path / 'unsafe.tar'
    with tarfile.open(bad, 'w') as f:
        f.add(root / 'manifest.json', arcname='../escape.json')
    with pytest.raises((KeyError, ValueError, SystemExit)):
        exp.collect(bad, root, tmp_path / 'bad', 42)
    assert not (tmp_path / 'escape.json').exists()


def test_real_screen_replay_output_exists_guard(package, tmp_path):
    with pytest.raises((ValueError, SystemExit), match='fresh screen'):
        exp.screen(package, tmp_path, tmp_path, replay=True)





def test_collected_screen_path_real_control_predictions(package, tmp_path, monkeypatch):
    # Exercise collection->screen orchestration with explicit engineering
    # fixtures. Scientific output verification is separately exercised by the
    # actual authenticated control-reuse tests and cannot be bypassed by CLI.
    outputs = tmp_path / 'outputs'
    for s, record in exp.controls().items():
        seed = int(s)
        collected = outputs / f'{exp.ARM}-S{seed}'
        run = collected / 'results' / exp.tag(exp.ARM, seed)
        run.mkdir(parents=True)
        shutil.copyfile(exp.ROOT / record['path'] / 'oof_k36_flip_tta.csv', run / 'oof_k36_flip_tta.csv')
        exp.write(collected / 'collection.json', {'status': 'VERIFIED', 'arm': exp.ARM, 'seed': seed,
            'manifest_sha256': exp.sha(package / 'manifest.json')})
    verified = []
    monkeypatch.setattr(exp, 'verify_run', lambda p, arm, seed, overlay: verified.append(seed))
    from training.arms import collected_screen
    observed = []
    monkeypatch.setattr(collected_screen, 'readout', lambda spec, **kw: observed.append(spec) or {'holistic_decision': 'B_DECISION_PENDING'})
    result = exp.screen(package, outputs, tmp_path / 'screen')
    assert result['status'] == 'B_DECISION_PENDING' and verified == [42, 43]
    assert set(observed[0]['seeds']) == {'42', '43'} and observed[0]['label_change'] is True
    data = exp.read(Path(result['path']))
    assert not data['engineering_replay']
    assert data['registered_readout']['status'] == 'HOLD' and data['launch_authorized'] is False
    assert data['seed_average']['mean_jev_consensus']['delta']['PF OA'] == 0
    assert all(data['harm_guard'].values())



