"""v1.0.0: timeout-only v10 family, injected clocks; CPU, no real sleeps."""
import base64
import contextlib
import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from tools import k1n_stage_v1 as b
from tools import k1n_stage_v10 as v
from test_k1n_members_v1 import filled, runtime
from test_k1n_v7 import replay
import test_k1n_v9 as prior


def configuration():
    c = b.runtime9.registered(False)
    return dict(c, k1_runtime_version=10, release_version='1.1.78-cpu-timeout10',
                kernel_id='test-owner/cpu-timeout10-v1')


@pytest.mark.parametrize('seconds,failed', [(121, False), (151, False), (899, False), (901, True)])
def test_loader_budget_and_study_local_failure(tmp_path, monkeypatch, seconds, failed):
    r = runtime(configuration())
    clock, budgets = [1000.], []
    monkeypatch.setenv('K1SR_STARTED', '1000')
    monkeypatch.setattr(r, 'time', SimpleNamespace(time=lambda:clock[0], perf_counter=lambda:clock[0]))
    @contextlib.contextmanager
    def deadline(budget):
        budgets.append(budget)
        start = clock[0]
        yield
        if clock[0]-start > budget:
            raise TimeoutError('injected capture deadline')
    monkeypatch.setattr(r, 'deadline', deadline)
    monkeypatch.setitem(sys.modules, 'compact_infer', SimpleNamespace())
    class Adapter:
        def __getitem__(self, i):
            clock[0] += seconds if i == 0 else 1
            return {'uid':str(i)}, {}, {}
    ds = r.LoaderDataset([{'uid':'0'}, {'uid':'1'}], tmp_path/'worker0')
    ds.adapter = Adapter()
    items = [ds[i] for i in range(2)]
    assert ('error' in items[0]['_k1sr']) is failed
    assert 'error' not in items[1]['_k1sr']
    assert budgets == [900, 900] and not ds.failed
    assert not (tmp_path/'k1-disabled.json').exists()
    # Real Owner telemetry connects the capture result to the unchanged 5% gate.
    monkeypatch.setattr(r, 'precision', contextlib.nullcontext)
    monkeypatch.setattr(r.w, 'validate_item', lambda *a: (np.zeros(1), np.zeros(1)))
    monkeypatch.setattr(r, 'load_classifier', lambda *a: object())
    monkeypatch.setattr(r, 'transfer_inputs', lambda *a: object())
    monkeypatch.setattr(r, 'predict_prepared', lambda *a, **kw: np.full(12, .4))
    worker = tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    owner = r.Owner(worker, 'cuda')  # All device operations above are CPU mocks.
    for item in items:
        owner.handle(item, lambda item: None)
    parent = replay.csv_bytes(['0','1'], np.full((2,12), .5))
    (tmp_path/'submission.csv').write_bytes(parent)
    if failed:
        with pytest.raises(RuntimeError, match='fail-loud'):
            r.finish(tmp_path, tmp_path/'run', b.config(configuration()), 10.)
    else:
        r.finish(tmp_path, tmp_path/'run', b.config(configuration()), 10.)
    receipt = json.loads((tmp_path/'k1sr-receipt.json').read_bytes())
    assert receipt['failed_study_count'] == int(failed)
    assert receipt['fail_loud_triggered'] is failed
    assert receipt['capture_timeout_seconds'] == 900
    assert receipt['call_timeout_seconds'] == 300
    assert receipt['k1_runtime_version'] == 10
    assert receipt['k1_max_failed_fraction'] == .05
    assert not torch.cuda.is_initialized()


def test_capture_deadline_clipped_by_guard(tmp_path, monkeypatch):
    r = runtime(configuration())
    monkeypatch.setattr(r, 'remaining', lambda: 45.)
    monkeypatch.setitem(sys.modules, 'compact_infer', SimpleNamespace())
    budgets = []
    def deadline(n):
        budgets.append(n)
        return contextlib.nullcontext()
    monkeypatch.setattr(r, 'deadline', deadline)
    ds = r.LoaderDataset([{'uid':'0'}], tmp_path/'worker0')
    ds.adapter = [({'uid':'0'}, {}, {})]
    assert 'item' in ds[0]['_k1sr'] and budgets == [45.]


@pytest.mark.parametrize('fault', ['capture','timeout','invalid','load','load_timeout','guard'])
def test_owner_v9_failure_contract_with_new_deadlines(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(prior, 'runtime', lambda c=None: runtime(c or configuration()))
    monkeypatch.setattr(prior, 'configuration', configuration)
    prior.test_owner_continues_later_study_except_systemic(tmp_path, monkeypatch, fault)


@pytest.mark.parametrize('failures', [0, 5, 6])
def test_unchanged_five_percent_boundary(tmp_path, monkeypatch, failures):
    c = configuration()
    monkeypatch.setattr(prior, 'runtime', lambda c=None: runtime(c))
    receipt = prior.finish_case(tmp_path, monkeypatch, failures=failures, c=c)
    assert receipt['call_timeout_seconds'] == 300 and receipt['load_timeout_seconds'] == 1500


def test_standalone_exact_timeout_delta():
    old = (b.ROOT/v.RUNTIME9).read_text(encoding='utf-8')
    expected = v.runtime_delta(old.encode()).decode().replace(
        "REGISTERED = dict(config_schema.current(), k1_failure_policy='study_fallback_fail_loud')",
        "REGISTERED = dict(config_schema.current(), k1_failure_policy='study_fallback_fail_loud', k1_runtime_version=10)")
    assert (b.ROOT/v.RUNTIME10).read_text(encoding='utf-8') == expected


@pytest.mark.parametrize('name', v.FAMILY)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_family_exact_payload_and_notebook_inverse(name, diagnostic):
    c = json.loads((b.ROOT/'configs/k1n'/f'{name}.json').read_bytes())
    pc = json.loads((b.ROOT/'configs/k1n'/f'{v.FAMILY[name]}.json').read_bytes())
    assert {k for k in c.keys() | pc.keys() if c.get(k) != pc.get(k)} == {
        'k1_runtime_version', 'release_version', 'kernel_id'}
    nb, meta, old, new, old_nb, old_meta = b.package(c, diagnostic)
    parent = v.parent_config(c)
    expected = b.exact(old[v.RUNTIME9].decode(), 'REGISTERED = '+repr(parent),
                       'REGISTERED = '+repr(b.canonical(c)))
    assert new[v.RUNTIME10] == v.runtime_delta(expected.encode())
    normalized = copy.deepcopy(nb)
    code = ''.join(nb['cells'][2]['source'])
    blob, pins, _ = b.before.parent.archive(code)
    prior_blob, prior_pins, _ = b.before.parent.archive(''.join(old_nb['cells'][2]['source']))
    for a,z in [(repr(base64.b64encode(blob).decode()),repr(base64.b64encode(prior_blob).decode())),
                ('PINS = '+repr(pins),'PINS = '+repr(prior_pins)),
                (repr(b.digest(blob)),repr(b.digest(prior_blob))),
                ('from tools import k1sr_runtime_v10 as _k1sr','from tools import k1sr_runtime_v9 as _k1sr'),
                (repr(b.config(c,diagnostic)),repr(v.replay.load().config(parent,diagnostic)))]:
        code = b.exact(code,a,z)
    normalized['cells'][2]['source'] = code.splitlines(True)
    normalized['cells'][1] = old_nb['cells'][1]
    assert normalized == old_nb
    assert meta == dict(old_meta,id=c['kernel_id']+('-n32' if diagnostic else ''),
                        title=c['kernel_id'].split('/')[1]+('-n32' if diagnostic else ''))
    out, reg = b.artifacts(c,diagnostic)
    assert reg['capture_timeout_seconds'] == 900 and reg['call_timeout_seconds'] == 300
    assert reg['never_submit'] is diagnostic and not reg['qualified_for_submission']
    assert reg['parent_is_existing_stage']
    schema = ModuleType('baked_schema10')
    exec(compile(new[v.SCHEMA], '<schema>', 'exec'), schema.__dict__)
    assert schema.validate(c) == b.schema.validate(c)
    # Verify the exact complete v9 parent, including registration source hashes.
    parent_outputs, _ = b.artifacts(pc,diagnostic)
    dest = b.stage(pc,diagnostic)
    assert set(parent_outputs) == {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
    assert all((dest/n).read_bytes() == data for n,data in parent_outputs.items())


@pytest.mark.parametrize('template', ['b5-flat-r6s2-v9','folds5-r6s2-av2-v9',
    'c5-dropo-p25-v9','c7-bal-dropo-p25s3-v9','c7-bal-dropo-p25-v9'])
@pytest.mark.parametrize('flat', [False, True])
def test_recipe_composition(template, flat):
    c = filled(b.ROOT/'configs/k1n'/f'{template}.template.json')
    c.update(k1_runtime_version=10)
    if flat:
        c['p_weight_mode'] = 'flat'
    files = b.package(c)[3]
    r = runtime(c)
    assert r.CAPTURE_TIMEOUT_SECONDS == 900 and r.CALL_TIMEOUT_SECONDS == 300
    assert r.REGISTERED == b.canonical(c)
    assert b.config(c)['formula'].endswith(v.TIMEOUT_FORMULA)


@pytest.mark.parametrize('old,new', [
    ('c6-fb-dropo-p25-v9','c6-fb-dropo-p25-v10'),
    ('c6-fb-dropo-p25-flatp','c6-fb-dropo-p25-flatp-v10'),
    ('b6-fb-flat-r6s2-v9','b6-fb-flat-r6s2-v10')])
def test_fullbag_placeholders_preserved(old,new):
    a,z = [json.loads((b.ROOT/'configs/k1n'/f'{n}.template.json').read_bytes()) for n in (old,new)]
    assert z == dict(a,k1_runtime_version=10)
    with pytest.raises(ValueError):
        b.run('prepare',z)
