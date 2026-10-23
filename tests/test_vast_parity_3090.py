"""CPU-only fake fixtures; no CUDA initialization, network, corpus or real weights."""
from __future__ import annotations

import hashlib
import json
import socket
import sys
import types
from pathlib import Path

import numpy as np
import numpy.testing  # noqa: F401  B 2026-10-09: import before the subprocess guard (numpy.testing probes SVE via subprocess on lazy import)
import pytest
import scipy.stats  # noqa: F401  same reason: scipy.stats pulls numpy.testing lazily

from tools import vast_parity_3090 as v
from tools import vast_parity_3090_training as train


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU test attempted network, process or GPU work')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(v.subprocess, 'run', forbidden)
    monkeypatch.setattr(v.subprocess, 'Popen', forbidden)
    # Real torch need not be imported at all. Fake tensor tests replace this module.
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(init=forbidden, _lazy_init=forbidden))
    monkeypatch.setitem(sys.modules, 'torch', fake)


@pytest.mark.parametrize('value', ['../x', '/tmp/x', 'C:/secret', 'x/../../y', '.local/relabel/a',
                                  'storage/competition/train.csv', '.ssh/id_ed25519', 'api-keys/key'])
def test_package_rejects_escape_and_sensitive_paths(value):
    with pytest.raises(ValueError):
        v.relative(value)


def test_path_remap_retains_recipe_and_rejects_other_drive(tmp_path):
    p = {'paths': {'corpus': r'H:\kaggle\competitions\rsna-knee-abnormality-detection\storage\cache'},
         'common_args': {'k': 12, 'k_eval': 24, 'epochs': 20}}
    got = v.remap_tree(p, 'H:/kaggle/competitions/rsna-knee-abnormality-detection')
    assert got['paths']['corpus'] == 'storage/cache'
    assert got['common_args'] == p['common_args']
    with pytest.raises(ValueError):
        v.remap_tree({'path': 'C:/not-project/weights'}, 'H:/project')
    assert train._path(tmp_path, r'H:\kaggle\competitions\rsna-knee-abnormality-detection\a') == tmp_path / 'a'
    with pytest.raises(ValueError):
        train._path(tmp_path, '../escape')


def test_static_closure_includes_relative_and_legacy_imports(tmp_path):
    for name, text in {'tools/start.py': 'from tools import helper\nimport raptor_x\n',
                       'tools/helper.py': 'from . import leaf\n', 'tools/leaf.py': '',
                       'training/src/raptor_x.py': 'import numpy\n'}.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    assert v.code_closure(tmp_path, ['tools/start.py']) == {
        'tools/start.py', 'tools/helper.py', 'tools/leaf.py', 'training/src/raptor_x.py'}


def test_stage_inventory_detects_modified_input(tmp_path):
    (tmp_path / 'x').write_bytes(b'original')
    v.write(tmp_path / 'inventory.json', {'files': {'x': v.record(tmp_path / 'x')}})
    v.verify_inventory(tmp_path)
    (tmp_path / 'x').write_bytes(b'changed')
    with pytest.raises(ValueError, match='stage hash'):
        v.verify_inventory(tmp_path)


def test_corpus_stages_training_and_full_evaluation_shards(tmp_path):
    c = tmp_path / v.CORPUS
    c.mkdir(parents=True)
    ids = [f'u{i}' for i in range(4344)]
    np.save(c / 'ids.npy', np.asarray(ids))
    v.write(c / 'manifest.json', {'shards': [
        {'start':0, 'stop':3400, 'file':'training-only.npy'},
        {'start':3400, 'stop':3755, 'file':'mixed.npy'},
        {'start':3755, 'stop':4343, 'file':'late-eval.npy'},
        {'start':4343, 'stop':4344, 'file':'unused.npy'}]})
    files = v.corpus_files(tmp_path, {'train':ids[:3455], 'val':ids[3455:4343]})
    assert v.CORPUS + '/training-only.npy' in files
    assert v.CORPUS + '/late-eval.npy' in files
    assert v.CORPUS + '/unused.npy' not in files


def test_training_equivalence_requires_both_references_and_inclusive_gap():
    base = {'metrics': {'42': {'dread': {'macro':.5}, 'consensus':{'macro':.5}}},
            'limits': {'dread':.125, 'consensus':.125}}
    scores = {'dread':{'macro':.625}, 'consensus':{'macro':.375}}
    assert v.training_verdict(scores, base)[0] == 'EQUIVALENT'
    scores['consensus']['macro'] = .374
    assert v.training_verdict(scores, base)[0] == 'NOT_EQUIVALENT'


def test_receipt_covers_nested_receipts_and_extra_files(tmp_path):
    nested = tmp_path / 'nested'
    nested.mkdir()
    (nested / 'x').write_bytes(b'result')
    v.receipt(nested)
    v.receipt(tmp_path)
    assert 'nested/receipt.json' in v.read(tmp_path / 'receipt.json')['files']
    v.verify_receipt(tmp_path)
    (tmp_path / 'unexpected').write_bytes(b'bad')
    with pytest.raises(ValueError, match='roster'):
        v.verify_receipt(tmp_path)


def test_missing_initial_blocks_before_torch_load(tmp_path):
    v.write(tmp_path / 'vast-config.json', {'initial_state': None})
    with pytest.raises(FileNotFoundError, match='no CUDA'):
        train.run_training(tmp_path, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


class FakeTensor:
    def __init__(self, values):
        self.a = np.asarray(values, dtype=np.float32)
        self.dtype, self.shape = 'torch.float32', self.a.shape

    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.a


def test_saved_initial_verifies_every_tensor_not_rng(tmp_path, monkeypatch):
    state = {'a': FakeTensor([1, 2]), 'b': FakeTensor([3])}
    h = hashlib.sha256()
    keys = {}
    for name, tensor in state.items():
        raw = tensor.numpy().tobytes()
        keys[name] = [hashlib.sha256(raw).hexdigest(), 'float32', list(tensor.shape)]
        h.update(name.encode()); h.update(raw)
    v.write(tmp_path / 'keys.json', {'keys': keys, 'combined_sha256': h.hexdigest()})
    (tmp_path / 'state.pt').write_bytes(b'fake file: torch.load is injected')
    calls = []
    def load(path, **kwargs):
        calls.append(kwargs)
        return state
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(Tensor=FakeTensor, load=load))
    assert train.verify_initial(tmp_path / 'state.pt', tmp_path / 'keys.json') is state
    assert calls == [{'map_location': 'cpu', 'weights_only': True}]
    state['b'] = FakeTensor([4])
    with pytest.raises(ValueError, match='per-key mismatch'):
        train.verify_initial(tmp_path / 'state.pt', tmp_path / 'keys.json')


def test_target_only_reader_and_prediction_order(tmp_path, monkeypatch):
    import pandas as pd
    actual_read = pd.read_csv
    p = tmp_path / 'labels.csv'
    d = pd.DataFrame({v.UID: ['b', 'a'], **{t: [.25, .75] for t in v.TARGETS},
                      'Report': ['synthetic forbidden column', 'must not materialize']})
    d.to_csv(p, index=False)
    def checked(*args, **kwargs):
        assert callable(kwargs['usecols']) and not kwargs['usecols']('Report')
        return actual_read(*args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', checked)
    assert 'Report' not in v.safe_table(p).columns
    assert v.safe_table(p, ['a', 'b'])[0, 0] == .75
    with pytest.raises(ValueError, match='order'):
        v.safe_table(p, ['a', 'b'], prediction=True)


def test_fixed_reference_rounding_and_consensus_masks(tmp_path, monkeypatch):
    x = np.asarray([[.5, .1], [.6, .9], [.2, .8]])
    data = {name: x.copy() for name in v.LABELS}
    data['steven_v4'][0, 0] = .49
    data['pilkwang_v2'][2, 1] = np.nan
    def table(path, ids):
        return data[next(k for k, rel in v.LABELS.items() if str(path).replace('\\','/').endswith(rel))]
    monkeypatch.setattr(v, 'safe_table', table)
    refs = v.references(tmp_path, ['a', 'b', 'c'])
    assert refs['dread'][0][0, 0] == 0  # rint ties-to-even, not >=.5
    assert not refs['consensus'][1][0, 0]
    assert not refs['consensus'][1][2, 1]
    assert refs['consensus'][1][1, 0]


def test_auc_uses_ties_and_fixed_mask():
    y = np.asarray([[0], [1], [0], [1.]])
    p = np.asarray([[.2], [.5], [.5], [.9]])
    r = v.auc(p, (y, np.ones(y.shape, bool)))
    assert r['macro'] == .875
    with pytest.raises(ValueError, match='no evaluable'):
        v.auc(p, (y, np.zeros(y.shape, bool)))


def test_integer_mismatch_cannot_be_hidden_by_identical_probabilities():
    left = {'rad/0': {'integer': {'volume': 'abc'}}}
    v.check_integer(left, left)
    with pytest.raises(ValueError, match='integer-stage'):
        v.check_integer(left, {'rad/0': {'integer': {'volume': 'changed'}}})
    with pytest.raises(ValueError, match='roster'):
        v.check_integer(left, {})


def test_float_report_measures_actual_inputs_and_checks_hash(tmp_path):
    left, right = tmp_path / 'l', tmp_path / 'r'
    left.mkdir(); right.mkdir()
    xx = np.ones((1, 1, 2, 2), np.float32)
    yy = xx + np.float32(1e-6)
    np.savez(left / 'input.npz', x=xx)
    np.savez(right / 'input.npz', x=yy)
    def rec(z):
        full = np.repeat(z, 3, axis=1)
        return {'rad/0': {'uid': 'fake', 'integer': {'volume': 'same'}, 'float_file': 'input.npz',
                          'tensor_sha256': v.tensor_hash(full), 'shape': list(full.shape), 'compact_channels': True}}
    a, b = rec(xx), rec(yy)
    report = v.float_report(left, right, a, b)
    assert 0 < report['float_max_abs'] < 2e-6
    assert report['float_hash_mismatches'] == 1
    b['rad/0']['tensor_sha256'] = 'tampered'
    with pytest.raises(ValueError, match='tensor/hash'):
        v.float_report(left, right, a, b)


def test_upload_plan_never_executes_and_rejects_shell_injection(tmp_path, capsys):
    (tmp_path / 'inventory.json').write_text('{}')
    commands = v.upload_plan(tmp_path, 'ssh.example.org', 12345)
    assert all('StrictHostKeyChecking=yes' in c for c in commands)
    assert all('UserKnownHostsFile=' in c for c in commands)
    assert 'id_ed25519' in commands[0]
    assert not any('StrictHostKeyChecking=no' in c for c in commands)
    assert 'NEW reviewed trust' in capsys.readouterr().out
    with pytest.raises(ValueError, match='endpoint'):
        v.upload_plan(tmp_path, 'host;touch BAD', 22)


def test_remote_script_pins_and_queue_wrapper_without_running(tmp_path):
    p = v.remote_script(tmp_path)
    text = p.read_text()
    assert 'torch==2.5.1+cu121' in text and 'timm==1.0.19 numpy==2.0.2' in text
    assert '_admit --admission' in text and 'gpu_queue.py remote-queue.json --foreground' in text
    assert 'set -euo pipefail' in text
    assert v.read(tmp_path / 'remote-queue.json')[0]['command'][-1] == '_remote-run'
    assert v.remote_script(tmp_path) == p  # idempotent exact file only
    p.write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        v.remote_script(tmp_path)


def test_pin_drift_fails_without_importing_cuda(monkeypatch):
    monkeypatch.setattr(v.importlib.metadata, 'version', lambda n: v.PINS[n])
    assert v.pins_ok() == v.PINS
    monkeypatch.setattr(v.importlib.metadata, 'version', lambda n: '0.0')
    with pytest.raises(ValueError, match='pins differ'):
        v.pins_ok()


def fake_runtime():
    return {'torch': v.PINS['torch'], 'cuda':'12.1', 'cudnn':90100,
            'cudnn_benchmark':False, 'cudnn_deterministic':True,
            'matmul_tf32':False, 'cudnn_tf32':False, 'flash_sdp':False,
            'mem_efficient_sdp':True, 'math_sdp':True, 'torch_threads':1,
            'deterministic_algorithms':True, 'deterministic_warn_only':False,
            'libs': {k:v.PINS[k] for k in ('numpy','timm','torchvision')}, 'gpu':'fake',
            'policy':dict(amp='fp16',sdpa='efficient',strict=True,threads=1,seed=42,
                          deterministic_warn_only=False,batch=1)}


def test_identically_missing_or_wrong_runtime_is_not_parity():
    v.validate_runtime(fake_runtime(), inference=True)
    with pytest.raises(ValueError, match='runtime receipt'):
        v.validate_runtime({}, inference=True)
    wrong = fake_runtime()
    wrong['matmul_tf32'] = True
    with pytest.raises(ValueError, match='runtime receipt'):
        v.validate_runtime(wrong, inference=True)


def test_frozen_protocol_values():
    tol = v.read(v.ROOT / v.REG / 'tol.json')
    source = v.read(v.ROOT / 'artifacts/experiments/PARITY_T4_LOCAL_V1/tol.json')
    assert tol['probs'] == source['keys']['probs']
    assert tol['macro_auc_abs'] == 1e-4
    baseline = v.read(v.ROOT / v.REG / 'baseline.json')
    for ref in ('dread', 'consensus'):
        assert abs(abs(baseline['42'][ref] - baseline['43'][ref]) - baseline['absolute_seed_gaps'][ref]) < 1e-15


def test_inference_compare_pass_and_score_delta_failure(tmp_path, monkeypatch):
    """End-to-end saved-array gate using tiny 300x12 arrays and a scalar input."""
    stage, left, right = [tmp_path / x for x in ('stage','local','remote')]
    for p in (stage, left, right): p.mkdir()
    ids = [f'fake{i:03}' for i in range(300)]
    v.write(stage / 'infer-ids.json', ids)
    v.write(stage / v.REG / 'tol.json', {'probs': {'max_abs': .002, 'mean_abs': .0005, 'min_spearman': .999},
                                       'macro_auc_abs': 1e-4, 'float_input_max_abs': 2e-6})
    files = {n: v.record(stage / n) for n in ('infer-ids.json', v.REG + '/tol.json')}
    v.write(stage / 'inventory.json', {'files': files})
    y = np.tile((np.arange(300) % 2)[:, None], (1, 12))
    mask = np.ones(y.shape, bool)
    monkeypatch.setattr(v, 'references', lambda *_: {'dread': (y, mask), 'consensus': (y, mask)})
    pp = np.tile(np.linspace(.2, .8, 300)[:, None], (1, 12)).astype(np.float32)
    x = np.zeros((1, 3, 1, 1), np.float32)
    inp = {f'{family}/{i:04d}': {'uid': uid, 'integer': {'volume': 'same'}, 'shape': list(x.shape),
                      'float_file': 'x.npz', 'compact_channels': False, 'tensor_sha256': v.tensor_hash(x)}
           for family in ('radr50','coatnet') for i,uid in enumerate(ids)}
    runtime = fake_runtime()
    for out in (left, right):
        v.write(out / 'runtime.json', runtime)
        v.write(out / 'inputs.json', inp)
        v.write(out / 'run.json', {'inventory_sha256':v.sha(stage/'inventory.json'), 'tol_sha256':v.sha(stage/v.REG/'tol.json')})
        np.savez(out / 'x.npz', x=x)
        for arm in v.ARMS:
            np.savez(out / (arm+'.npz'), probs=pp, ids=ids, targets=v.TARGETS)
        v.receipt(out)
    assert v.compare_runs(stage,left,right,tmp_path/'pass',True) == 0
    assert v.read(tmp_path/'pass/parity-report.json')['part1'] == 'PASS'
    # Small probability changes alone do not grant PASS when the metric gate fails.
    original_auc = v.auc
    count = 0
    def changed_metric(p, ref):
        nonlocal count
        ans = original_auc(p, ref)
        count += 1
        if count % 2 == 0: ans['macro'] += .0002
        return ans
    monkeypatch.setattr(v, 'auc', changed_metric)
    assert v.compare_runs(stage,left,right,tmp_path/'fail',True) == 1
