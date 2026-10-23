"""v1.1.0: runtime-v9 completion regression using baked payloads; CPU only.

Changelog v1.1.0: runtime 10 is supported; use 11 as the unsupported selector.
"""
import base64
import contextlib
import copy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from tools import k1n_stage_v1 as b
from test_k1n_v7 import records, replay, rank


def configuration(vector=False):
    return b.runtime9.registered(vector)


def runtime(c=None):
    r = ModuleType('test_v9')
    files = b.package(c or configuration())[3]
    exec(compile(files['tools/k1sr_runtime_v9.py'], '<baked v9>', 'exec'), r.__dict__)
    return r


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
    raises = failures/n > c['k1_max_failed_fraction'] or fault not in (None, 'guard')
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


@pytest.mark.parametrize('fault', ['load', 'disabled', 'late', 'missing', 'duplicate',
                                 'identity', 'invalid', 'corrupt'])
def test_systemic_and_integrity_raise_after_publication(tmp_path, monkeypatch, fault):
    finish_case(tmp_path, monkeypatch, fault=fault)


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


@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_completed_before_guard_late_finish_uses_blend(tmp_path, monkeypatch, vector, diagnostic):
    rec = finish_case(tmp_path, monkeypatch, fault='guard', c=configuration(vector),
                      diagnostic=diagnostic)
    output = tmp_path/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')
    assert output.read_bytes() != (tmp_path/'k1sr-parent.csv').read_bytes()
    assert rec['schema'] == 'k1-shared-r6-release/v9'
    assert rec['k1_guard_hours'] == 8.6 and rec['guard_seconds'] == 30960


@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_row_completed_after_guard_raises(tmp_path, monkeypatch, vector, diagnostic):
    rec = finish_case(tmp_path, monkeypatch, fault='late', c=configuration(vector),
                      diagnostic=diagnostic)
    assert 'K1 guard completion' in rec['error']


def test_v9_source_is_exact_authorized_v8_delta():
    old = (b.ROOT/'tools/k1sr_runtime_v8.py').read_text(encoding='utf-8')
    expected = old.replace('runtime v8.0.0.',
        'runtime v9.0.0.\n\nChangelog v9.0.0: remove finish-only deadline check; retain row completion guard.')
    for a,z in [
        ("VERSION = '8.0.0'", "VERSION = '9.0.0'"),
        ('tools.k1sr_runtime_v8 import', 'tools.k1sr_runtime_v9 import'),
        ("schema='k1-shared-r6-release/v8'", "schema='k1-shared-r6-release/v9'"),
        ('K1 disabled/8h guard', 'K1 disabled/configured wall-clock guard'),
        ("        w.need(remaining() > 0, 'K1 8h guard reached')\n", ''),
        ('GUARD_SECONDS = 8 * 3600', "GUARD_SECONDS = REGISTERED.get('k1_guard_hours', 8.0) * 3600"),
    ]:
        expected = expected.replace(a,z)
    assert (b.ROOT/'tools/k1sr_runtime_v9.py').read_text(encoding='utf-8') == expected
    assert b.util.sha(b.ROOT/'tools/k1sr_runtime_v8.py') == (
        '5a0c680512ca160db8bfdfff4c6de550d0b7649d8cde8c6d9e19261dbc81f104')


@pytest.mark.parametrize('selector', [None, True, 8, 9.0, '9', 11])
def test_selector_rejected(selector):
    c = configuration()
    c['k1_runtime_version'] = selector
    with pytest.raises(ValueError, match='k1_runtime_version'):
        b.schema.validate(c)


@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_full_payload_notebook_inverse_and_recipe(vector, diagnostic):
    c = configuration(vector)
    nb, meta, old, new, old_nb, old_meta = b.package(c, diagnostic)
    p = configuration() if vector else b.schema.validate(json.loads(
        (b.ROOT/'configs/k1n/current-x3.json').read_bytes()))
    fields = {k for k in p.keys() | c.keys() if p.get(k) != c.get(k)}
    assert fields == ({'alpha_vector', 'release_version', 'kernel_id'} if vector else {
        'release_version', 'kernel_id', 'r6_eval_stride', 'k1_runtime_version',
        'k1_failure_policy', 'k1_max_failed_fraction', 'k1_guard_hours'})
    assert c['k1_eval_stride'] == 3 and not c['drop_o'] and c['alpha'] == .2
    assert c['models'] == p['models']
    if vector:
        assert c['alpha_vector'] == [.1,.2,.2,.5,.35,.5,.35,.35,.5,.2,.15,.35]
        expected = dict(old)
        expected['tools/k1sr_runtime_v9.py'] = b.exact(
            old['tools/k1sr_runtime_v9.py'].decode(), 'REGISTERED = '+repr(p),
            'REGISTERED = '+repr(c)).encode()
    else:
        expected = dict(old)
        del expected['tools/k1sr_runtime_v7.py']
        expected['tools/k1sr_runtime_v9.py'] = b.exact(
            (b.ROOT/'tools/k1sr_runtime_v9.py').read_text(encoding='utf-8'),
            "REGISTERED = dict(config_schema.current(), k1_failure_policy='study_fallback_fail_loud')",
            'REGISTERED = '+repr(c)).encode()
        # v1.7.0 pins the reviewed v1.6.0 schema for historical stage replay.
        expected['tools/k1n_config_v1.py'] = (b.ROOT/'tools/k1n_frozen_v160/k1n_config_v1.py').read_bytes()
        expected['tools/k1_shared_render_v2.py'] = b.adapter(
            (b.ROOT/'tools/k1_shared_render_v2.py').read_bytes())
        port = 'exp169_r7_port.py'
        expected[port] = b.exact(b.exact(b.exact(old[port].decode(),
            'k1sr_runtime_v7', 'k1sr_runtime_v9'), b.STRIDE_ANCHOR, '"--eval-stride", "2"'),
            'VERSION = "1.0.1-r6"', 'VERSION = "1.0.2-hb228-r6s2"').encode()
        release = 'exp169_release_c85r7.py'
        expected[release] = b.exact(old[release].decode(), b.digest(old[port]),
                                   b.digest(expected[port])).encode()
    oc, nc = b.runtime9.notebook_config(old_nb), b.config(c, diagnostic)
    codec = 'exp169_release_c85r7codec.py'
    text = old[codec].decode()
    if not vector:
        text = b.exact(text, b.digest(old[release]), b.digest(expected[release]))
    text = text.replace(oc['release_version'], nc['release_version'])
    expected[codec] = b.exact(text, oc['kernel_id'], nc['kernel_id']).encode()
    assert new == expected
    normalized = copy.deepcopy(nb)
    code = ''.join(nb['cells'][2]['source'])
    blob, pins, _ = b.before.parent.archive(code)
    prior, prior_pins, _ = b.before.parent.archive(''.join(old_nb['cells'][2]['source']))
    code = b.exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(prior).decode()))
    code = b.exact(code, 'PINS = '+repr(pins), 'PINS = '+repr(prior_pins))
    code = b.exact(code, repr(b.digest(blob)), repr(b.digest(prior)))
    if not vector:
        code = b.exact(code, 'from tools import k1sr_runtime_v9 as _k1sr',
                       'from tools import k1sr_runtime_v7 as _k1sr')
    code = b.exact(code, repr(nc), repr(oc))
    normalized['cells'][2]['source'] = code.splitlines(True)
    normalized['cells'][1] = old_nb['cells'][1]
    assert normalized == old_nb
    assert meta == dict(old_meta, id=nc['kernel_id'], title=nc['kernel_id'].split('/')[1])
    out, reg = b.artifacts(c, diagnostic)
    assert reg['never_submit'] is diagnostic and not reg['qualified_for_submission']
    assert reg['k1_runtime_version'] == 9 and reg['guard_seconds'] == 30960
    assert b.artifacts(dict(reversed(list(c.items()))), diagnostic)[0] == out


@pytest.mark.parametrize('field,value', [('k1_eval_stride', 2), ('r6_eval_stride', 1),
    ('drop_o', True), ('k1_guard_hours', 8.5), ('k1_max_failed_fraction', .1)])
def test_recipe_mutations_rejected(field, value):
    c = configuration()
    c[field] = value
    with pytest.raises(ValueError, match='recipe drift'):
        b.package(c)


@pytest.mark.parametrize('name', ['x3-r6s2-v9', 'x3-r6s2-av2-v9'])
def test_delivered_cli_config(name):
    c = json.loads((b.ROOT/'configs/k1n'/f'{name}.json').read_bytes())
    assert b.run('validate', c)['status'] == 'PASS_CPU_VALIDATE'
