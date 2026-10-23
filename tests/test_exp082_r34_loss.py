"""Bounded CPU/synthetic checks; no baseline trainer import, model, images or Gold."""
from copy import deepcopy
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import exp082_r34_loss as guard

ROOT = guard.ROOT


def main_body(source):
    return next(n.body for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def assignment(body, name):
    return next(n for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def sample():
    ids = ['b', 'a', 'gold-quarantined', 'bad', 'd', 'c']
    folds = {'b': 0, 'a': 2, 'gold-quarantined': -1, 'bad': 3, 'd': 1, 'c': 2}
    train, evaluation = ['b', 'd'], ['a', 'c']
    expected = dict(train=2, val=2, train_ids_sha256=guard.uid_hash(train, True), eval_ids_sha256=guard.uid_hash(evaluation, True))
    return ids, folds, {'bad'}, expected


def test_nonlabel_cohort_exclusions_and_order():
    ids, folds, excluded, expected = sample()
    train, evaluation = guard.reconstruct_ids(ids, folds, excluded, expected)
    assert train == ['b', 'd'] and evaluation == ['a', 'c']
    with pytest.raises(ValueError, match='membership'):
        guard.reconstruct_ids(ids, folds, set(), expected)
    with pytest.raises(ValueError, match='Duplicate'):
        guard.reconstruct_ids(ids + [ids[0]], folds, excluded, expected)


def test_same_membership_reordered_sequence_is_rejected():
    train, evaluation = guard.reconstruct_ids(*sample())
    ordered = {'train': train, 'eval': evaluation}
    assert guard.uid_hash(train, True) == guard.uid_hash(train[::-1], True)
    with pytest.raises(ValueError, match='ordered'):
        guard.validate_ordered_ids(train[::-1], evaluation, ordered)


def test_soft_label_weights_train_only_and_clip():
    labels = {'a': np.tile(np.array([0., .5, 1.], dtype=np.float32), 4),
              'b': np.tile(np.array([0., .5, 1.], dtype=np.float32), 4),
              'eval': np.ones(12, dtype=np.float32)}
    baseline, actual = guard.class_weights(labels, ['a', 'b'])
    np.testing.assert_array_equal(baseline, np.tile([10., 1., 1.], 4))
    np.testing.assert_array_equal(actual, np.ones(12))
    assert baseline.dtype == actual.dtype == np.float32
    protocol = {'loss': {'baseline_pos_weight': baseline.tolist(), 'actual_pos_weight': actual.tolist()}}
    guard.validate_loss_vectors(baseline, protocol)
    protocol['loss']['baseline_pos_weight'][0] = 9.
    with pytest.raises(ValueError):
        guard.validate_loss_vectors(baseline, protocol)


def test_ones_positive_term_matches_unweighted_bce_and_gradient():
    torch = pytest.importorskip('torch')
    logits = torch.linspace(-2, 2, 24, dtype=torch.float32, device='cpu').reshape(2, 12).requires_grad_()
    labels = torch.linspace(0, 1, 24, dtype=torch.float32, device='cpu').reshape(2, 12)
    weighted = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=torch.ones(12))
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    torch.testing.assert_close(weighted, plain, rtol=0, atol=0)
    g1 = torch.autograd.grad(weighted, logits, retain_graph=True)[0]
    g2 = torch.autograd.grad(plain, logits)[0]
    torch.testing.assert_close(g1, g2, rtol=0, atol=1e-8)
    assert not torch.cuda.is_initialized()


def test_rendered_static_trainer_and_unchanged_training_epoch_ast():
    original = (ROOT / 'training/src/raptor_train.py').read_text(encoding='utf-8')
    candidate = (ROOT / guard.ENTRYPOINT).read_text(encoding='utf-8')
    assert candidate == guard.render_training_source(original)
    ob, cb = main_body(original), main_body(candidate)
    old_loop = next(n for n in ob if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'ep')
    new_loop = next(n for n in cb if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'ep')
    assert ast.dump(old_loop) == ast.dump(new_loop)
    for name in ('tds', 'shuffle_generator', 'tl', 'eval_bs', 'model', 'head', 'bb', 'opt', 'steps', 'sched', 'lossf', 'scaler'):
        assert ast.dump(assignment(ob, name)) == ast.dump(assignment(cb, name)), name
    for name in ('loader', 'evaluate', 'state'):
        a = next(n for n in ob if isinstance(n, ast.FunctionDef) and n.name == name)
        b = next(n for n in cb if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(a) == ast.dump(b), name


def test_no_gold_loader_or_gold_prediction_route():
    tree = ast.parse((ROOT / guard.ENTRYPOINT).read_text(encoding='utf-8'))
    assert not any(isinstance(n, ast.Name) and n.id in {'gold_labels', 'gold_ids'} for n in ast.walk(tree))
    assert not any(isinstance(n, ast.ImportFrom) and n.module == 'raptor_train' for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {'exec', 'eval'} for n in ast.walk(tree))


def test_gold_blocker_denies_before_open_in_fresh_process(tmp_path):
    script = "from tools.exp082_r34_loss import install_gold_blocker;from pathlib import Path;import sys;root=Path(sys.argv[1]);install_gold_blocker(root);open(root/'storage/competition/train.csv')"
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode != 0 and 'Gold access forbidden' in result.stderr
    assert 'FileNotFoundError' not in result.stderr


def test_entrypoint_import_does_not_load_training_dependencies_or_run():
    script = "import importlib.util,sys;from pathlib import Path;p=Path(sys.argv[1]);s=importlib.util.spec_from_file_location('exp082_import_check',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch' not in sys.modules;assert 'raptor_train' not in sys.modules;assert 'raptor_common' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', script, str(ROOT / guard.ENTRYPOINT)], cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr


def test_pretraining_gate_rejects_initial_state_and_runtime_drift():
    protocol = {'training_code_sha256': {}, 'initial_state_sha256': 'init',
        'expected_ids': {'train_ids_sha256': 'train', 'eval_ids_sha256': 'eval'},
        'ordered_ids': {'train_sha256': 'ordered-train', 'eval_sha256': 'ordered-eval'},
        'corpus_manifest_sha256': 'corpus', 'expected_provenance': {'runtime': {'strict': True}, 'eval_runtime': {'strict': True}},
        'environment': {k: k for k in guard.ENV_KEYS}}
    p = dict(code_sha256={}, initial_state_sha256='init', train_ids_sha256='train', eval_ids_sha256='eval',
        train_ids_ordered_sha256='ordered-train', eval_ids_ordered_sha256='ordered-eval', corpus_manifest_sha256='corpus',
        checkpoint_selection='fixed-final', sampling='sha256(seed,epoch,uid,stream)-v1', gold_access='disabled',
        loss_policy='ones_positive_term', runtime={'strict': True}, eval_runtime={'strict': True})
    guard.validate_pretraining(p, protocol['environment'], protocol)
    q = deepcopy(p);q['initial_state_sha256'] = 'different'
    with pytest.raises(ValueError, match='provenance'):
        guard.validate_pretraining(q, protocol['environment'], protocol)
    q = deepcopy(p);q['eval_runtime']['strict'] = False
    with pytest.raises(ValueError, match='runtime'):
        guard.validate_pretraining(q, protocol['environment'], protocol)


def test_canonical_label_contract_mismatch_stops():
    protocol = {'label_hash': 'l', 'eval_label_hash': 'l', 'split_hash': 's'}
    guard.validate_label_contract('l', 'l', 's', protocol)
    with pytest.raises(ValueError):
        guard.validate_label_contract('l', 'changed', 's', protocol)
