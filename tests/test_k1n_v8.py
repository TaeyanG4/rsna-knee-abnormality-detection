"""CPU-only v8 policy, final publication, continuation and immutable lineage."""
import ast
import base64
import contextlib
import copy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from tools import k1n_config_v1 as schema
from tools import k1n_stage_v1 as b
from tools import k1sr_runtime_v8 as source
from test_k1n_v7 import records, replay, rank

NAMES = ['x3-stride2-drop-o-alpha-vector-v2', 'x3-stride2-alpha-vector-v2-r6s2']


def configuration(name=NAMES[0]):
    return schema.validate(json.loads((b.ROOT/'configs/k1n'/f'{name}-v8.json').read_bytes()))


def runtime(c=None):
    r = ModuleType('test_v8')
    text = Path(source.__file__).read_text(encoding='utf-8').replace(
        "REGISTERED = dict(config_schema.current(), k1_failure_policy='study_fallback_fail_loud')",
        'REGISTERED = '+repr(c or configuration()))
    exec(compile(text, '<v8 runtime>', 'exec'), r.__dict__)
    return r


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_absent_policy_every_artifact_byte_identical(name, diagnostic):
    c = json.loads((b.ROOT/'configs/k1n'/f'{name}.json').read_bytes())
    assert 'k1_failure_policy' not in schema.validate(c)
    out, _ = b.artifacts(c, diagnostic)
    dest = b.stage(c, diagnostic)
    assert {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()} == set(out)
    assert all((dest/n).read_bytes() == data for n, data in out.items())


@pytest.mark.parametrize('limit', [None, True, '0.05', -1, 1, float('nan'), float('inf')])
def test_bad_fraction(limit):
    c = configuration()
    c['k1_max_failed_fraction'] = limit
    with pytest.raises(ValueError, match='k1_max_failed_fraction'):
        schema.validate(c)


def test_policy_validation():
    c = configuration()
    assert c['k1_max_failed_fraction'] == .05
    c['k1_failure_policy'] = None
    with pytest.raises(ValueError, match='policy'):
        schema.validate(c)
    c.pop('k1_failure_policy')
    with pytest.raises(ValueError, match='requires'):
        schema.validate(c)


def finish_case(tmp_path, monkeypatch, n=100, failures=0, diagnostic=False, fault=None, c=None):
    c = c or configuration()
    r = runtime(c)
    monkeypatch.setenv('K1SR_STARTED', str(r.time.time()))
    rng = np.random.default_rng(143)
    values = [rng.integers(0, 11, (n, 12)).astype(np.float32)/10 for _ in range(4)]
    parent = replay.csv_bytes(list(map(str, range(n))), values[0])
    (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(parent)
    rows = records(r, values[1:])
    for row in rows[:failures]:
        row.update(ok=False, error='synthetic capture/classify failure')
        row.pop('predictions')
    if fault == 'missing':
        rows.pop()
    if fault == 'duplicate':
        rows[-1] = rows[0]
    if fault == 'identity':
        rows[0]['config_sha256'] = 'wrong'
    if fault == 'invalid':
        rows[0]['predictions'][r.MODEL_IDS[0]][0] = 2
    if fault == 'load':
        rows[0].update(ok=False, systemic_failure=True, error='model load failed')
    if fault == 'late':
        rows[0]['finished_elapsed_seconds'] = r.GUARD_SECONDS+1
    if fault == 'guard':
        monkeypatch.setenv('K1SR_STARTED', str(r.time.time()-r.GUARD_SECONDS-1))
    worker = tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text(
        '{' if fault == 'corrupt' else '\n'.join(map(json.dumps, rows)), encoding='utf-8')
    if fault == 'disabled':
        (worker.parent/'k1-disabled.json').write_text('{}')
    published = []
    original = r.util.publish
    def publish(path, payload):
        published.append(path)
        original(path, payload)
    monkeypatch.setattr(r.util, 'publish', publish)
    raises = failures/n > c['k1_max_failed_fraction'] or fault is not None
    if raises:
        with pytest.raises(RuntimeError, match='K1 fail-loud'):
            r.finish(tmp_path, tmp_path/'run', b.config(c, diagnostic), 10)
    else:
        r.finish(tmp_path, tmp_path/'run', b.config(c, diagnostic), 10)
    rec = json.loads((tmp_path/'k1sr-receipt.json').read_bytes())
    output = tmp_path/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')
    assert published == [output]
    assert rec['fail_loud_triggered'] is raises
    assert rec['k1_failure_policy'] == 'study_fallback_fail_loud'
    assert (tmp_path/'k1sr-parent.csv').read_bytes() == parent
    assert rec['submission_sha256' if not diagnostic else 'diagnostic_sha256'] == r.w.sha(output)
    if diagnostic:
        assert rec['never_submit'] and rec['submission_sha256'] is None
        assert not (tmp_path/'submission.csv').exists()
    if raises:
        assert output.read_bytes() == parent
        assert rec['status'] == 'FAILED_K1_FAIL_LOUD' and not rec['k1_executed']
    else:
        assert rec['status'] == 'COMPLETE_SHARED_K1N' and rec['k1_executed']
        assert rec['failed_study_count'] == failures and rec['failed_study_fraction'] == failures/n
        # Independent rank formula, with ties and per-target weights.
        pred = [a.copy() for a in values[1:]]
        for a in pred:
            a[:failures] = values[0][:failures]
        parent_rank = rank(values[0])
        k1 = rank(np.mean([rank(a) for a in pred], axis=0))
        k1[:failures] = parent_rank[:failures]
        alpha = np.asarray(c.get('alpha_vector', c['alpha']))
        weighted = (1-alpha)*parent_rank+alpha*k1
        weighted[:failures] = parent_rank[:failures]
        np.testing.assert_array_equal(r.util.table(output.read_bytes())[1], rank(weighted).astype(np.float32))
    return rec


@pytest.mark.parametrize('n,failures', [(32, 1), (100, 0), (100, 4), (100, 5), (100, 6), (1, 1)])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_failure_fraction_and_rank(tmp_path, monkeypatch, n, failures, diagnostic):
    finish_case(tmp_path, monkeypatch, n, failures, diagnostic)


@pytest.mark.parametrize('fault', ['load', 'guard', 'disabled', 'late', 'missing', 'duplicate',
                                 'identity', 'invalid', 'corrupt'])
def test_systemic_and_integrity_raise_after_publication(tmp_path, monkeypatch, fault):
    finish_case(tmp_path, monkeypatch, fault=fault)


def test_registered_fraction_override(tmp_path, monkeypatch):
    c = configuration()
    c['k1_max_failed_fraction'] = .03
    finish_case(tmp_path, monkeypatch, failures=4, c=c)


@pytest.mark.parametrize('fault', ['capture', 'timeout', 'invalid', 'load', 'load_timeout', 'guard'])
def test_owner_continues_later_study_except_systemic(tmp_path, monkeypatch, fault):
    r = runtime()
    clock = [1000.]
    monkeypatch.setenv('K1SR_STARTED', '1000')
    monkeypatch.setattr(r, 'time', SimpleNamespace(time=lambda:clock[0], perf_counter=lambda:clock[0]))
    monkeypatch.setattr(r, 'deadline', lambda s: contextlib.nullcontext())
    monkeypatch.setattr(r, 'precision', contextlib.nullcontext)
    monkeypatch.setattr(r.w, 'validate_item', lambda *a: (np.zeros(1), np.zeros(1)))
    calls, current = [], [0]
    def load(device, mid):
        calls.append(('load', current[0], mid))
        if fault == 'load':
            raise RuntimeError('load failed')
        if fault == 'load_timeout':
            clock[0] += r.LOAD_TIMEOUT_SECONDS+1
        return mid
    def predict(*a, **kw):
        calls.append(('predict', current[0]))
        if current[0] == 0:
            if fault == 'timeout':
                clock[0] += r.CALL_TIMEOUT_SECONDS+1
            if fault == 'invalid':
                return np.full(12, np.nan)
        return np.full(12, .4)
    monkeypatch.setattr(r, 'load_classifier', load)
    monkeypatch.setattr(r, 'transfer_inputs', lambda *a: object())
    monkeypatch.setattr(r, 'predict_prepared', predict)
    worker = tmp_path/'worker0'
    worker.mkdir()
    owner = r.Owner(worker, 'cuda')
    if fault == 'guard':
        clock[0] += r.GUARD_SECONDS+1
    for i in range(2):
        current[0] = i
        item = dict(uid=str(i), _k1sr=dict(item={}, detail={}))
        if i == 0 and fault == 'capture':
            item['_k1sr'] = dict(error='capture unavailable')
        owner.handle(item, lambda item: calls.append(('parent', current[0])))
    rows = [json.loads(line) for line in (worker/'k1-studies.jsonl').read_text().splitlines()]
    systemic = fault in ('load', 'load_timeout', 'guard')
    assert not rows[0]['ok'] and rows[1]['ok'] is not systemic
    assert owner.failed is systemic and (tmp_path/'k1-disabled.json').exists() is systemic
    assert [x for x in calls if x[0] == 'parent'] == [('parent', 0), ('parent', 1)]
    assert bool(owner.models) is not systemic
    assert not torch.cuda.is_initialized()
    # Owner's real telemetry must reach the same final fail-loud gate.
    if systemic:
        out = tmp_path/'output'
        out.mkdir()
        parent = replay.csv_bytes(['0', '1'], np.full((2, 12), .5))
        (out/'submission.csv').write_bytes(parent)
        dest = tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
        dest.mkdir(parents=True)
        (dest/'k1-studies.jsonl').write_bytes((worker/'k1-studies.jsonl').read_bytes())
        with pytest.raises(RuntimeError, match='fail-loud'):
            r.finish(out, tmp_path/'run', b.config(configuration()), 10)
        assert (out/'submission.csv').read_bytes() == parent
        assert json.loads((out/'k1sr-receipt.json').read_bytes())['fail_loud_triggered']


def test_loader_capture_error_does_not_disable_next_study(tmp_path, monkeypatch):
    import sys
    r = runtime()
    monkeypatch.setenv('K1SR_STARTED', str(r.time.time()))
    monkeypatch.setattr(r, 'deadline', lambda s: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, 'compact_infer', SimpleNamespace())
    class Adapter:
        def __getitem__(self, i):
            if i == 0:
                raise ValueError('missing raw capture')
            return {'uid': str(i)}, {}, {}
    ds = r.LoaderDataset([{'uid': '0'}, {'uid': '1'}], tmp_path/'worker0')
    ds.adapter = Adapter()
    assert 'error' in ds[0]['_k1sr']
    assert 'error' not in ds[1]['_k1sr']
    assert not ds.failed and not (tmp_path/'k1-disabled.json').exists()


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_full_parent_diff_and_notebook_tail(name, diagnostic):
    c = configuration(name)
    nb, meta, old, new, old_nb, old_meta = b.package(c, diagnostic)
    p = b.fail_loud.parent(c)
    assert old_nb == json.loads((b.stage(p, diagnostic)/'kernel/release.ipynb').read_bytes())
    assert new.keys()-old.keys() == {'tools/k1sr_runtime_v8.py'}
    assert old.keys()-new.keys() == {'tools/k1sr_runtime_v7.py'}
    assert {n for n in old.keys() & new.keys() if old[n] != new[n]} == {
        'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py',
        'tools/k1n_config_v1.py'}
    # Inverse all permitted notebook transforms; any other byte fails.
    normalized = copy.deepcopy(nb)
    code = ''.join(nb['cells'][2]['source'])
    blob, pins, _ = b.before.parent.archive(code)
    prior, prior_pins, _ = b.before.parent.archive(''.join(old_nb['cells'][2]['source']))
    code = b.exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(prior).decode()))
    code = b.exact(code, 'PINS = '+repr(pins), 'PINS = '+repr(prior_pins))
    code = b.exact(code, repr(b.digest(blob)), repr(b.digest(prior)))
    code = b.exact(code, 'from tools import k1sr_runtime_v8 as _k1sr',
                   'from tools import k1sr_runtime_v7 as _k1sr')
    code = b.exact(code, repr(b.config(c, diagnostic)), repr(b.config(p, diagnostic)))
    normalized['cells'][2]['source'] = code.splitlines(True)
    normalized['cells'][1] = old_nb['cells'][1]
    assert normalized == old_nb
    assert meta == dict(old_meta, id=b.config(c, diagnostic)['kernel_id'],
                       title=b.config(c, diagnostic)['kernel_id'].split('/')[1])
    # finish is the penultimate statement in the main guard, with no catch.
    tree = ast.parse(''.join(nb['cells'][2]['source']))
    main = tree.body[-1]
    assert isinstance(main, ast.If)
    assert isinstance(main.body[-2], ast.Assign)
    assert ast.unparse(main.body[-2].value.func) == '_k1sr.finish'
    assert ast.unparse(main.body[-1].value.func) == 'print'
    out, reg = b.artifacts(c, diagnostic)
    assert len(json.loads(out['payload-scope.json'])['untouched']) == 34
    assert reg['never_submit'] is diagnostic and not reg['qualified_for_submission']


def test_parent_identity_and_recipe_binding():
    c = configuration()
    c['alpha_vector'][0] = .99
    with pytest.raises(ValueError, match='recipe drift'):
        b.package(c)
