"""Regression for cold dataset mounts: loading is not classification time."""
import contextlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import k1sr_runtime_v4, k1sr_runtime_v5


@pytest.mark.parametrize('r', [k1sr_runtime_v4, k1sr_runtime_v5])
@pytest.mark.parametrize('mode', ['slow_load', 'slow_call', 'load_timeout', 'global_guard', 'group_timeout'])
def test_separate_load_and_classifier_budgets(monkeypatch, tmp_path, r, mode):
    clock = [1000.]
    monkeypatch.setenv('K1SR_STARTED', '1000')
    monkeypatch.setattr(r, 'time', SimpleNamespace(time=lambda: clock[0], perf_counter=lambda: clock[0]))
    if mode == 'global_guard': clock[0] += r.GUARD_SECONDS - 100
    budgets = []
    @contextlib.contextmanager
    def timer(seconds):
        assert seconds > 0
        budgets.append(seconds)
        yield
    monkeypatch.setattr(r, 'deadline', timer)
    monkeypatch.setattr(r, 'precision', contextlib.nullcontext)
    monkeypatch.setattr(r.w, 'offline', contextlib.nullcontext)
    monkeypatch.setattr(r.w, 'validate_item', lambda *args: (np.zeros(1), np.zeros(1)))
    loaded, classified, parents = [], [], []
    def load(device, mid):
        loaded.append(mid)
        clock[0] += 1501 if mode == 'load_timeout' else 200
        return mid
    def predict(model, *args, **kwargs):
        classified.append(model)
        clock[0] += 121 if mode == 'slow_call' else 70 if mode == 'group_timeout' else 1
        return np.full(12, .5)
    monkeypatch.setattr(r, 'load_classifier', load)
    monkeypatch.setattr(r.shared, 'predict', predict)
    worker = tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    owner = r.Owner(worker, 'cuda')
    owner.handle(dict(uid='0', _k1sr=dict(item={}, detail={})), lambda item: parents.append(item['uid']))
    row = json.loads((worker/'k1-studies.jsonl').read_text())
    assert parents == ['0']
    assert row['load_timeout_seconds'] == 1500 and row['call_timeout_seconds'] == 120
    assert row['ok'] == (mode == 'slow_load')
    assert owner.failed == (mode != 'slow_load')
    if mode == 'slow_load':
        assert loaded == classified == list(r.MODEL_IDS)
        assert row['load_seconds'] == 200 * len(r.MODEL_IDS)
        assert row['classifier_seconds'] == len(r.MODEL_IDS)
        assert budgets[::2] == [1500] * len(r.MODEL_IDS)
        assert budgets[1::2] == list(range(120, 120-len(r.MODEL_IDS), -1))
    else:
        assert not owner.models and 'predictions' not in row
        assert (worker.parent/'k1-disabled.json').exists()
    if mode in ('load_timeout', 'global_guard'): assert not classified
    if mode == 'slow_call': assert row['classifier_seconds'] == 121
    parent = (','.join([r.w.UID] + r.w.LABELS)+'\n0,'+','.join(['0.3']*12)+'\n').encode()
    (tmp_path/'submission.csv').write_bytes(parent)
    receipt = r.finish(tmp_path, tmp_path/'run', {}, 1.)
    assert receipt['k1_executed'] == (mode == 'slow_load')
    assert receipt['load_timeout_seconds'] == 1500
    assert receipt['call_timeout_seconds'] == 120 and receipt['guard_seconds'] == 28800
    assert receipt['per_model_seconds'] == row['per_model_seconds']
    if mode != 'slow_load': assert (tmp_path/'submission.csv').read_bytes() == parent


def test_new_registration_timeout_fields():
    from tools import c85r7_r6w958k1x3sr_v1162 as b
    for name, diagnostic in [('k1x3sr20', False), ('n224k1x3sr20', False), ('k1x3sr20', True)]:
        _, reg = b.artifacts(name, diagnostic)
        assert reg['load_timeout_seconds'] == 1500
        assert reg['call_timeout_seconds'] == 120 and reg['guard_seconds'] == 28800
        assert reg['diagnostic'] == reg['never_submit'] == diagnostic


@pytest.mark.parametrize('name,diagnostic', [('k1x3sr20',False), ('n224k1x3sr20',False), ('k1x3sr20',True)])
def test_repair_diff_has_no_unrelated_payload_changes(name, diagnostic):
    from tools import c85r7_r6w958k1x3sr_v1160 as old
    from tools import c85r7_r6w958k1x3sr_v1162 as new
    _, om, _, before = old.package(name, diagnostic)
    _, nm, _, after = new.package(name, diagnostic)
    assert {k:v for k,v in om.items() if k not in ('id','title')} == {k:v for k,v in nm.items() if k not in ('id','title')}
    expected = dict(before)
    expected.pop('tools/k1sr_runtime_v3.py')
    expected['tools/k1sr_runtime_v5.py'] = after['tools/k1sr_runtime_v5.py']
    port = 'exp169_r7_port.py'
    release = 'exp169_release_c85r7.py'
    codec = 'exp169_release_c85r7codec.py'
    expected[port] = before[port].replace(b'k1sr_runtime_v3', b'k1sr_runtime_v5')
    expected[release] = before[release].replace(old.digest(before[port]).encode(), new.digest(expected[port]).encode())
    expected[codec] = before[codec].replace(old.digest(before[release]).encode(), new.digest(expected[release]).encode())
    expected[codec] = expected[codec].replace(old.config(name,diagnostic)['release_version'].encode(), new.config(name,diagnostic)['release_version'].encode())
    expected[codec] = expected[codec].replace(old.config(name,diagnostic)['kernel_id'].encode(), new.config(name,diagnostic)['kernel_id'].encode())
    assert expected == after
