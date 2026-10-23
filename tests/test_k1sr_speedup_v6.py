"""H-B-221 CPU engineering gates, reuse, resource and timeout regressions."""
import ast
import contextlib
import json
import os
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pytest
import torch
from tools import k1sr_runtime_v5 as old
from tools import k1sr_runtime_v6 as r
from tools import k1sr_benchmark_v6 as bench
from tools import c85r7_r6w958k1x3sr_v1164 as builder


def test_math_render_load_and_process_functions_are_ast_identical():
    def nodes(module):
        tree = ast.parse(open(module.__file__, encoding='utf-8').read())
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    a, b = nodes(old), nodes(r)
    for name in ('blend', 'LoaderDataset', 'load_classifier', 'precision', 'patch_main', 'worker_main',
                 'deadline', 'phase_deadline', 'disable', 'disabled', 'remaining'):
        assert a[name] == b[name], name


@pytest.mark.parametrize('chunk', [32, 64])
def test_real_prepared_api_no_retransfer_and_same_layout(chunk):
    x = torch.randint(0, 256, (3, 3, 384, 384), dtype=torch.uint8)
    m = torch.rand(3, 6)
    prepared = r.transfer_inputs(x, m, 'cpu')
    seen = []
    def model(a, b, mask, *, chunk):
        seen.append((id(a), id(b), id(mask), chunk))
        assert torch.equal(a, x[None, ..., 16:368, 16:368])
        assert torch.equal(b, m[None]) and mask.all() and mask.shape == (1, 3)
        return torch.full((1, 12), .3)
    for _ in range(3):
        p = r.predict_prepared(model, prepared, chunk=chunk, fp16=False)
        np.testing.assert_array_equal(p, torch.full((12,), .3).sigmoid().numpy())
    assert len(set(seen)) == 1
    with pytest.raises(Exception, match='CUDA'):
        r.predict_prepared(model, prepared)
    with pytest.raises(Exception, match='chunk'):
        r.predict_prepared(model, prepared, chunk=16, fp16=False)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mode', ['ok','transfer_fail','transfer_timeout','slow_load','load_timeout',
                                 'classify_timeout','group_timeout','global_guard'])
def test_owner_transfers_once_and_preserves_budgets(monkeypatch, tmp_path, mode):
    clock = [1000.]
    monkeypatch.setenv('K1SR_STARTED', '1000')
    monkeypatch.setattr(r, 'time', SimpleNamespace(time=lambda:clock[0], perf_counter=lambda:clock[0]))
    if mode == 'global_guard': clock[0] += r.GUARD_SECONDS-100
    @contextlib.contextmanager
    def deadline(seconds):
        if seconds <= 0: raise TimeoutError('budget')
        yield
    monkeypatch.setattr(r, 'deadline', deadline)
    monkeypatch.setattr(r, 'precision', contextlib.nullcontext)
    monkeypatch.setattr(r.w, 'validate_item', lambda *args:(np.zeros(1),np.zeros(1)))
    calls, prepared = [], object()
    def transfer(*args):
        calls.append('transfer')
        if mode == 'transfer_fail': raise RuntimeError('transfer')
        clock[0] += 121 if mode == 'transfer_timeout' else 2
        return prepared
    def load(device, mid):
        calls.append('load')
        clock[0] += 1501 if mode == 'load_timeout' else 200
        return mid
    def predict(model, tensors, *, chunk, fp16):
        assert tensors is prepared and chunk == 32 and fp16
        calls.append('classify')
        clock[0] += 121 if mode == 'classify_timeout' else 50 if mode == 'group_timeout' else 1
        return np.full(12,.3)
    monkeypatch.setattr(r, 'transfer_inputs', transfer)
    monkeypatch.setattr(r, 'load_classifier', load)
    monkeypatch.setattr(r, 'predict_prepared', predict)
    worker = tmp_path/'worker0'
    worker.mkdir()
    owner = r.Owner(worker, 'cuda')
    owner.handle(dict(uid='0', _k1sr=dict(item={},detail={})), lambda item:calls.append('parent'))
    row = json.loads((worker/'k1-studies.jsonl').read_text())
    passed = mode in ('ok','slow_load')
    assert row['ok'] == passed and calls[0] == 'parent'
    assert calls.count('transfer') <= 1
    if passed:
        assert calls.count('classify') == 3 and calls.count('transfer') == 1
        assert row['classifier_seconds'] == 5 and row['transfer_seconds'] == 2
        assert row['load_seconds'] == 600 and row['encoder_chunk'] == 32
        assert all(row['per_model_seconds'][m] == dict(load=200.,classify=1.) for m in r.MODEL_IDS)
    else:
        assert owner.failed and not owner.models and 'predictions' not in row
        assert (worker.parent/'k1-disabled.json').exists()


def test_fp16_gate_rejects_delta_rank_and_constant_outputs():
    a = np.tile(np.arange(32)[:,None]/32, (1,12))
    assert bench.equality(a,a)['status'] == 'PASS'
    assert bench.equality(a,a+.006)['status'] == 'FAIL'
    b = a.copy(); b[:,0] = b[::-1,0]
    assert bench.equality(a,b)['status'] == 'FAIL'
    assert bench.equality(np.ones((32,12)), np.ones((32,12)))['status'] == 'FAIL'
    with pytest.raises(Exception): bench.equality(a[:1],a[:1])
    b[0,0] = np.nan
    with pytest.raises(Exception): bench.equality(a,b)


def test_stale_self_or_absent_review_fails_before_gpu(tmp_path):
    p = tmp_path/'review.json'
    valid = dict(verdict='PASS', builder='astra-b1', reviewer='independent-verifier', identity=bench.identity())
    for updates in (dict(verdict='HOLD'),dict(reviewer='astra-b1'),dict(identity={}),dict(reviewer='')):
        p.write_text(json.dumps(dict(valid, **updates)))
        with pytest.raises(Exception): bench.validate(p)
    p.write_text(json.dumps(valid))
    assert bench.validate(p)['independent_review_verified']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('diagnostic', [False, True])
def test_v5_delta_proof_rejects_unrelated_changes(diagnostic):
    nb,meta,old_files,files = builder.package('k1x3sr20',diagnostic)
    _,scope = builder.v5_delta(nb,meta,files,diagnostic)
    assert json.loads(scope)['status'] == 'PASS'
    broken = dict(files, **{'exp169_r224.py':files['exp169_r224.py']+b'\n'})
    with pytest.raises(Exception, match='scope'): builder.v5_delta(nb,meta,broken,diagnostic)
    nb['cells'][0]['source'].append('# unrelated\n')
    with pytest.raises(Exception, match='scope'): builder.v5_delta(nb,meta,files,diagnostic)


def test_chunk64_cannot_be_selected_for_release():
    assert r.ENCODER_CHUNK == 32 and list(builder.VARIANTS) == ['k1x3sr20']
    assert builder.config('k1x3sr20')['encoder_chunk'] == 32
    with pytest.raises(KeyError): builder.config('k1x3sr20_chunk64')


def test_benchmark_direct_entry_cannot_bypass_review(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, 'OUT', tmp_path)
    with pytest.raises(Exception, match='review required'):
        bench.run(None)
    assert json.loads((tmp_path/'GPU-BENCHMARK.json').read_text())['status'] == 'FAILED'
    assert not torch.cuda.is_initialized()


def test_benchmark_summary_memory_parity_and_saving():
    import copy
    records = [dict(uid=str(i), arms={arm:dict(total_seconds=3. if arm == 'v5' else 2.,
        peak_reserved_bytes=4*2**30, peak_allocated_bytes=3*2**30,
        predictions={mid:[i/32]*12 for mid in r.MODEL_IDS}) for arm in bench.ARMS})
        for i in range(32)]
    result = bench.summarize(records, 2*2**30, 16*2**30)
    assert result['v6_chunk32']['engineering_gate_pass']
    assert result['v6_chunk64']['estimated_1300_study_saving_hours'] == pytest.approx(1300/3600)
    excessive = copy.deepcopy(records)
    excessive[0]['arms']['v6_chunk64']['peak_reserved_bytes'] = 13*2**30
    assert not bench.summarize(excessive, 2*2**30, 16*2**30)['v6_chunk64']['engineering_gate_pass']
    assert not bench.summarize(records, 11*2**30, 16*2**30)['v6_chunk32']['memory_gate']
    changed = copy.deepcopy(records)
    changed[0]['arms']['v6_chunk32']['predictions'][r.MODEL_IDS[0]][0] += .006
    assert not bench.summarize(changed, 0, 16*2**30)['v6_chunk32']['numeric_pass']
