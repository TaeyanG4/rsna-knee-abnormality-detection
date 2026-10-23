"""H-B-227 CPU regression contracts: target axes, pins, scalar parity, fallbacks."""
import ast
import copy
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from test_k1n_v7 import configuration, runtime, records, rank, replay
from tools import k1n_stage_v1 as b
from tools import k1n_config_v1 as schema
from tools import k1sr_runtime_v7 as v7
from tools.hb227_alpha_confirm_v1 import VECTOR


def candidate():
    return json.loads((b.ROOT / 'configs/k1n/x3-stride2-alpha-vector-v2.json').read_bytes())


@pytest.mark.parametrize('value', [
    None, .2, '0.2', {}, [], [.2]*11, [.2]*13, [[.2]]*12,
    [.2]*11+[True], [.2]*11+['.2'], [.2]*11+[float('nan')],
    [.2]*11+[float('inf')], [.2]*11+[-.01], [.2]*11+[1.01],
])
def test_vector_rejects_malformed(value):
    c = candidate()
    c['alpha_vector'] = value
    with pytest.raises(Exception, match='alpha_vector'):
        schema.validate(c)


@pytest.mark.parametrize('n', range(1, 7))
@pytest.mark.parametrize('drop', [False, True])
def test_vector_target_axis_ties_and_endpoints(n, drop):
    c = configuration(n, drop)
    c['alpha_vector'] = [0, 1, *VECTOR[2:]]
    r = runtime(c)
    rng = np.random.default_rng(227)
    parent, *arrays = [rng.integers(0, 9, (19, 12)).astype(np.float32)/8 for _ in range(n+1)]
    payload = replay.csv_bytes(list(map(str, range(19))), parent)
    actual = r.util.table(r.blend(payload, list(reversed(records(r, arrays)))))[1]
    a = np.array(c['alpha_vector'])
    expected = rank((1-a)*rank(parent) + a*rank(np.mean([rank(x) for x in arrays], axis=0)))
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    np.testing.assert_array_equal(actual[:, 0], rank(parent)[:, 0].astype(np.float32))
    np.testing.assert_array_equal(actual[:, 1], rank(np.mean([rank(x) for x in arrays], axis=0))[:, 1].astype(np.float32))


def test_flat_vector_is_byte_identical_to_scalar_and_vector_is_bound():
    c = configuration()
    r = runtime(c)
    rng = np.random.default_rng(227)
    parent, *arrays = [rng.random((32, 12)).astype(np.float32) for _ in range(4)]
    payload = replay.csv_bytes(list(map(str, range(32))), parent)
    c['alpha_vector'] = [.2]*12
    z = runtime(c)
    assert z.CONFIG_SHA != r.CONFIG_SHA
    assert r.blend(payload, records(r, arrays)) == z.blend(payload, records(z, arrays))
    with pytest.raises(Exception, match='registered K1'):
        z.blend(payload, records(r, arrays))


@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('fault', ['ok', 'missing_model', 'vector_identity', 'finish_identity'])
def test_baked_vector_finish_and_n32(tmp_path, diagnostic, fault):
    c = candidate()
    _, _, _, files, _, _ = b.package(c, diagnostic)
    r = ModuleType('alpha_baked')
    exec(compile(files['tools/k1sr_runtime_v7.py'], '<baked>', 'exec'), r.__dict__)
    cfg = b.config(c, diagnostic)
    assert cfg['alpha_vector'] == VECTOR and cfg['alpha_targets'] == replay.util.LABELS
    assert cfg['k1_weight'] == VECTOR and cfg['drop_o'] is False
    assert cfg['k1_eval_stride'] == 2
    parent = replay.csv_bytes(['0', '1', '2'], np.arange(36).reshape(3, 12)/36)
    (tmp_path / ('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(parent)
    arrays = [np.linspace(.05, .95, 36).reshape(3, 12)]*3
    rows = records(r, arrays)
    if fault == 'missing_model':
        rows[0]['predictions'].pop(r.MODEL_IDS[-1])
    if fault == 'vector_identity':
        changed = copy.deepcopy(c)
        changed['alpha_vector'][0] += .01
        rows[0]['config_sha256'] = runtime(changed).CONFIG_SHA
    if fault == 'finish_identity':
        changed = copy.deepcopy(c)
        changed['alpha_vector'].reverse()
        cfg['config_sha256'] = b.config(changed)['config_sha256']
    worker = tmp_path / 'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    (worker / 'k1-studies.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    rec = r.finish(tmp_path, tmp_path / 'run', cfg, 12)
    assert rec['status'] == ('COMPLETE_SHARED_K1N' if fault == 'ok' else 'DEGRADED_PARENT_ONLY')
    assert rec['alpha_vector'] == VECTOR and rec['alpha_targets'] == replay.util.LABELS
    assert rec['parent_mode'] == 'INF3' and rec['k1_eval_stride'] == 2
    actual = (tmp_path / ('diagnostic-k1n.csv' if diagnostic else 'submission.csv')).read_bytes()
    assert actual == (r.blend(parent, rows) if fault == 'ok' else parent)
    if diagnostic:
        assert rec['never_submit'] and rec['submission_sha256'] is None
        assert not (tmp_path / 'submission.csv').exists()


def test_only_blend_and_receipt_functions_changed_from_v710():
    backup = b.ROOT / '.local/backups/astra-b5/hb227-alpha-vector-v2/k1sr_runtime_v7.py'
    def functions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old, new = functions(backup), functions(Path(v7.__file__))
    assert old.keys() == new.keys()
    assert {k for k in old if old[k] != new[k]} == {'blend', 'finish'}


@pytest.mark.parametrize('diagnostic', [False, True])
def test_inf3_payload_is_unchanged(diagnostic):
    c = candidate()
    scalar = copy.deepcopy(c)
    del scalar['alpha_vector']
    _, meta, _, files, _, _ = b.package(c, diagnostic)
    _, scalar_meta, _, scalar_files, _, _ = b.package(scalar, diagnostic)
    assert meta == scalar_meta
    assert {k for k in files if files[k] != scalar_files[k]} == {'tools/k1sr_runtime_v7.py'}
    historical = b.before.parent.INF3.parent / ('v1.1.65-r6w958k1n3sr20' + ('-n32' if diagnostic else ''))
    nb = json.loads((historical / 'kernel/release.ipynb').read_bytes())
    before = b.before.parent.archive(''.join(nb['cells'][2]['source']))[2]
    assert files['exp169_release_c85r7.py'] == before['exp169_release_c85r7.py']
    assert files['exp169_r7_port.py'] == before['exp169_r7_port.py']
    for name in ('exp169_release_c85.py', 'exp169_r224.py'):
        assert files[name] == before[name]
    assert c['models'] == schema.current()['models']


def test_prepare_validate_vector_registration(monkeypatch, tmp_path):
    monkeypatch.setattr(b, 'stage', lambda c, diagnostic=False: tmp_path / ('n32' if diagnostic else 'release'))
    c = candidate()
    assert b.run('prepare', c)['status'] == 'PREPARED_LOCAL_ONLY'
    assert b.run('validate', c)['status'] == 'PASS_CPU_VALIDATE'
    reg = json.loads((tmp_path / 'release/registration.json').read_bytes())
    assert reg['alpha_vector'] == VECTOR and reg['alpha_targets'] == replay.util.LABELS
    assert not reg['qualified_for_submission']
    c['alpha_vector'][0] += .01
    with pytest.raises(Exception, match='artifact drift'):
        b.run('validate', c)
