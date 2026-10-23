"""Tests for tools/oai_reader_readout_v1.py (CPU only; run with CUDA_VISIBLE_DEVICES='')."""
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from tools import oai_reader_readout_v1 as t


def test_cuda_hidden():
    assert not torch.cuda.is_available()


# --- checkpoints: hash refusal and weights_only ----------------------------------------------

def test_real_checkpoints_weights_only_load():
    models = t.load_models('cpu')
    assert len(models) == 3 and all(res == 256 for _, res in models)
    assert not torch.cuda.is_initialized()


def test_hash_mismatch_refused():
    bad = ('0' * 64,) + t.CKPT_SHA[1:]
    with pytest.raises(t.Stop, match='hash mismatch'):
        t.load_models('cpu', expected=bad)


class Evil:  # not allowed by weights_only
    pass


def test_weights_only_failure_is_stop(tmp_path):
    p = tmp_path / 'evil.pt'
    torch.save({'model': {}, 'args': {}, 'x': Evil()}, p)
    with pytest.raises(t.Stop, match='STOP weights_only load failed'):
        t.load_models('cpu', paths=[p], expected=(t.sha(p),))


def test_manifest_constants():
    assert t.manifest_hashes() == t.CKPT_SHA


# --- reader semantics ------------------------------------------------------------------------

def test_slot_rule_and_table():
    assert [t.slot_of('Sagittal', 1), t.slot_of('Coronal', 1), t.slot_of('Axial', t.FS_RULE['-1']),
            t.slot_of('Coronal', 0)] == [0, 1, 2, 4]
    with pytest.raises(t.Stop):
        t.slot_of('Axial', -1)
    rows = [dict(StudyInstanceUID='s', SeriesInstanceUID=n, Anatomical_Plane='Coronal', Fat_Suppression=0)
            for n in ('a', 'b', 'c', 'd')]
    table = t.build_table(rows, {'a': 10, 'b': 20, 'c': 2, 'd': 20})
    assert table['s'][4] == ['b', 'd', 'a']        # count desc, CSV order on ties, < 3 dropped


def test_window_centres_source_formula():
    for n in (3, 4, 37, 64):
        c = t.window_centres(n, 12)
        a, b = max(1.0, .04 * (n - 1)), min(n - 2.0, .96 * (n - 1))
        if b < a: a = b = (n - 1) / 2
        assert (c == np.clip(np.round(np.linspace(a, b, 12)).astype(int), 1, max(1, n - 2))).all()


def test_forward_runs_on_synthetic_input():
    m = t.make_model().eval()
    x = torch.randint(0, 255, (1, 6, 12, 3, 384, 384), dtype=torch.uint8)
    mask = torch.tensor([[True, False, False, False, True, False]]); pos = torch.rand(1, 6, 12)
    with torch.inference_mode():
        out = m(x, mask, pos)
    assert out.shape == (1, 12) and torch.isfinite(out).all()


# --- refusal paths ---------------------------------------------------------------------------

def test_loader_refuses_byte_drift_and_file_set(tmp_path):
    d = tmp_path / 'test_series' / 'U' / 'S'; d.mkdir(parents=True)
    (d / 'a.dcm').write_bytes(b'abc')
    good = {'test_series/U/S/a.dcm': t.sha_bytes(b'abc')}
    assert t.KneeLoader('U', tmp_path, good)('S') == [b'abc']
    with pytest.raises(t.Stop, match='byte drift'):
        t.KneeLoader('U', tmp_path, {'test_series/U/S/a.dcm': '0' * 64})('S')
    with pytest.raises(t.Stop, match='file set'):
        t.KneeLoader('U', tmp_path, dict(good, **{'test_series/U/S/b.dcm': '0' * 64}))('S')


def test_worker_requires_supervisor(monkeypatch):
    monkeypatch.delenv('OAI_READER_SUPERVISOR', raising=False)
    with pytest.raises(t.Stop, match='supervised'):
        t.worker()


def test_review_required(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'WORK', tmp_path)
    with pytest.raises(t.Stop, match='gpu-review'):
        t.check_review({'binding_sha256': 'x'})
    (tmp_path / 'validate.json').write_text('{}')
    for review in (dict(verdict='PASS', reviewer=t.BUILDER, tag=t.TAG, binding_sha256='x'),
                   dict(verdict='FAIL', reviewer='r', tag=t.TAG, binding_sha256='x'),
                   dict(verdict='PASS', reviewer='r', tag=t.TAG, binding_sha256='y')):
        (tmp_path / 'gpu-review.json').write_text(json.dumps(dict(review, validation_sha256=t.sha(tmp_path / 'validate.json'))))
        with pytest.raises(t.Stop, match='independent'):
            t.check_review({'binding_sha256': 'x'})
    (tmp_path / 'gpu-review.json').write_text(json.dumps(dict(verdict='PASS', reviewer='r', tag=t.TAG, binding_sha256='x',
                                                              validation_sha256=t.sha(tmp_path / 'validate.json'))))
    t.check_review({'binding_sha256': 'x'})


def test_finished_attempt_replay_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'WORK', tmp_path); (tmp_path / 'measurement.json').write_text('{}')
    monkeypatch.setattr(t, 'nvidia', lambda q: pytest.fail('GPU queried before the replay refusal'))
    with pytest.raises(t.Stop, match='replay refused'):
        t.run()
    with pytest.raises(t.Stop, match='replay refused'):
        t.seal({'binding': {}})
    monkeypatch.setenv('OAI_READER_SUPERVISOR', str(os.getppid()))
    with pytest.raises(t.Stop, match='replay refused'):
        t.worker()


def test_register_immutable_after_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(t, 'WORK', tmp_path); (tmp_path / 'gpu').mkdir()
    with pytest.raises(t.Stop, match='immutable'):
        t.register()


def test_vram_gate_and_gpu_process_refusal(monkeypatch):
    d = dict(used_bytes=2 * 2**30, total_bytes=16 * 2**30)
    assert t.vram_gate(d, 12 * 2**30) and not t.vram_gate(d, 12 * 2**30 + 1)
    assert not t.vram_gate(dict(d, used_bytes=3 * 2**30), 12 * 2**30)
    monkeypatch.setattr(t, 'nvidia', lambda q: [['123', 'C:\\x\\python.exe', '900'], ['7', 'dwm.exe', '1']])
    assert t.other_gpu_python(own=(999,)) == [['123', 'C:\\x\\python.exe', '900']]
    assert t.other_gpu_python(own=(123,)) == []


def test_cli_synthetic_flag_only_for_evaluate():
    with pytest.raises(t.Stop):
        t.main(['parity', '--synthetic'])
    with pytest.raises(t.Stop):
        t.main(['run', '--replay'])
    with pytest.raises(t.Stop):
        t.main(['evaluate', '--replay', '--synthetic'])


# --- evaluator -------------------------------------------------------------------------------

def _pre():
    from tools import r6_oai_readout_v110 as pre
    return pre, dict(bootstrap=pre.bootstrap, metric=pre.g.metric, macro7=pre.MACRO7, weak_four=pre.RULES['weak_four'])


def _synthetic(n=150, seed=1):
    pre, _ = _pre(); rng = np.random.default_rng(seed)
    y = np.full((n, 12), np.nan)
    for j in pre.MAPPED: y[:, j] = (rng.random(n) < .3).astype(float)
    noise = rng.random((n, 12)); yy = np.nan_to_num(y, nan=0.)
    return y, np.clip(.2 * yy + .8 * noise, 0, 1), np.clip(.9 * yy + .1 * noise, 0, 1), noise


def test_evaluator_identity_and_gain():
    pre, kw = _pre(); y, weak, strong, noise = _synthetic(); c = pre.label_contract(y)
    r = t.metrics(y, dict(Reader=weak, R6flip=weak, R7flip=noise), c, draws=200, **kw)
    m = r['comparisons']['Reader-minus-R6flip']['macro7']
    assert m['delta'] == 0 and m['ci95'] == [0., 0.] and 'ci97_5_family_two' in m
    assert m['statistical_basis'].startswith('unadjusted 95% CI')
    assert 'ci97_5_family_two' not in r['comparisons']['R6flip-minus-R7flip']['macro7']
    r = t.metrics(y, dict(Reader=strong, R6flip=weak, R7flip=noise), c, draws=200, **kw)
    m = r['comparisons']['Reader-minus-R6flip']['macro7']
    assert m['delta'] > 0 and m['statistical'] == 'GAIN' and m['p_normal_gt0'] > .99
    assert r['reader_standalone']['macro7']['delta'] > .9
    assert set(r['per_target_auc']) == set(t.ARMS) and r['valid_draws'] == 200
    with pytest.raises(t.Stop):
        t.metrics(y, dict(Reader=strong, R6flip=weak), c, draws=10, **kw)
    with pytest.raises(t.Stop):
        t.metrics(y, dict(Reader=strong + 1, R6flip=weak, R7flip=noise), c, draws=10, **kw)


def test_evaluator_reproduces_predecessor_bootstrap():
    """R6flip-minus-R7flip in our metrics == predecessor metrics on identical inputs."""
    pre, kw = _pre(); y, weak, strong, noise = _synthetic(seed=3); c = pre.label_contract(y)
    ours = t.metrics(y, dict(Reader=noise, R6flip=strong, R7flip=weak), c, draws=150, **kw)
    theirs = pre.metrics(y, dict(R7flip=weak, R6flip=strong), c, draws=150)
    a = ours['comparisons']['R6flip-minus-R7flip']['macro7']; b = theirs['comparisons']['R6flip-minus-R7flip']['macro_available']
    assert all(a[k] == b[k] for k in b)


def test_p_normal():
    assert abs(t.p_normal(0.0, 1.0) - .5) < 1e-12 and t.p_normal(1.0, 0.0) == 1.0 and t.p_normal(.01, .01) > .84


# --- real CPU receipts (produced by the parity / dry-run subcommands) ------------------------

def test_real_parity_and_dry_run_receipts():
    reg = t.current()
    p = json.loads((t.WORK / 'parity.json').read_text())
    assert p['state'] == 'PASS' and p['binding_sha256'] == reg['binding_sha256'] and len(p['studies']) == 2
    assert p['max_abs_delta'] <= t.PARITY_TOL
    d = json.loads((t.WORK / 'dry-run.json').read_text())
    assert d['state'] == 'PASS' and d['binding_sha256'] == reg['binding_sha256'] and len(d['knees']) == 2
    for k in d['knees']:
        assert k['mask'] == [True, True, True, False, True, False] and len(k['prediction']) == 12
    r = json.loads((t.WORK / 'replay.json').read_text())
    assert r['state'] == 'PASS_REAL_LABEL_IDENTITY_CONTROL' and r['binding_sha256'] == reg['binding_sha256']
    assert r['reader_output_read'] is False and all(c['r6_r7_reproduced'] for c in r['cohorts'].values())
    assert [r['cohorts'][c]['studies'] for c in ('pilot', 'confirmation', 'pooled')] == [96, 461, 557]
