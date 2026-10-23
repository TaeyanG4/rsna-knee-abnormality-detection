"""R6-only registration: complete frozen artifacts, recipes and CPU serving."""
import ast
import copy
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tools import k1n_stage_v1 as b
from tools import k1n_config_v1 as schema
from tools import k1n_compat_v1 as compat
from test_k1n_v7 import configuration, records, replay, rank
from test_hb228_v1 import worker_args

EXAMPLES = ['x3-stride2-drop-o-r6s2.json', 'x3-stride2-alpha-vector-v2-r6s2.json']


def read(name):
    return json.loads((b.ROOT/'configs/k1n'/name).read_bytes())


@pytest.mark.parametrize('value', [None, True, False, 0, 3, -1, 2., '2', [], {}])
def test_invalid_r6_stride(value):
    c = schema.current()
    c['r6_eval_stride'] = value
    with pytest.raises(ValueError, match='r6_eval_stride'):
        schema.validate(c)


@pytest.mark.parametrize('filename', [p[0].split('/')[-1] for p in b.PARENTS])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_absent_and_explicit_one_regenerate_every_frozen_artifact(filename, diagnostic):
    c = read(filename)
    assert 'r6_eval_stride' not in schema.validate(c)
    outputs, _ = b.artifacts(c, diagnostic)
    dest = b.stage(c, diagnostic)
    assert {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()} == set(outputs)
    for name, data in outputs.items():
        assert (dest/name).read_bytes() == data, name
    assert b.artifacts(dict(c, r6_eval_stride=1), diagnostic)[0] == outputs


@pytest.mark.parametrize('filename', EXAMPLES)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_exact_parent_stride_only_and_real_manifest(filename, diagnostic):
    c = read(filename)
    parent_name = filename.replace('-r6s2.json', '.json')
    original = read(parent_name)
    assert {k for k in c.keys() | original.keys() if c.get(k) != original.get(k)} == {
        'r6_eval_stride', 'release_version', 'kernel_id'}
    nb, meta, old, new, old_nb, old_meta = b.package(c, diagnostic)
    actual_parent = b.stage(original, diagnostic)
    assert old_nb == json.loads((actual_parent/'kernel/release.ipynb').read_bytes())
    assert old_meta == json.loads((actual_parent/'kernel/kernel-metadata.json').read_bytes())
    assert old.keys() == new.keys() and len(old) == 39
    changed = {n for n in old if old[n] != new[n]}
    assert changed == {'exp169_r7_port.py', 'exp169_release_c85r7.py',
        'exp169_release_c85r7codec.py', 'tools/k1_shared_render_v2.py', 'tools/k1sr_runtime_v7.py'}
    # Entire runtime byte equality after ONLY version/slug identity substitution.
    assert new['tools/k1sr_runtime_v7.py'] == b.exact(old['tools/k1sr_runtime_v7.py'].decode(),
        'REGISTERED = '+repr(schema.validate(original)), 'REGISTERED = '+repr(b.legacy(c))).encode()
    assert old['tools/k1n_config_v1.py'] == new['tools/k1n_config_v1.py']
    assert meta['dataset_sources'] == old_meta['dataset_sources']
    manifest = json.loads((b.ROOT/'storage/external/heliosli-r6-fe055406-v1/manifest.json').read_bytes())
    a = worker_args(old['exp169_r7_port.py'].decode())(manifest, 'bundle', 'anatomy')
    z = worker_args(new['exp169_r7_port.py'].decode())(manifest, 'bundle', 'anatomy')
    i = a.index('--eval-stride')+1
    assert a[i] == '1' and z[i] == '2' and '--flip-tta' in a
    a[i] = '2'
    assert a == z
    outputs, reg = b.artifacts(c, diagnostic)
    assert reg['r6_eval_stride'] == 2 and reg['k1_eval_stride'] == 2
    assert reg['never_submit'] == diagnostic and not reg['qualified_for_submission']
    assert reg['registered_config_sha256'] == b.digest(b.util.encoded(schema.validate(c)))
    assert len(json.loads(outputs['payload-scope.json'])['untouched']) == 34


@pytest.mark.parametrize('n', range(1, 7))
@pytest.mark.parametrize('drop', [False, True])
def test_any_registered_recipe_and_default_bytes(n, drop):
    c = configuration(n, drop)
    c.update(release_version='1.9.99-test-r6', kernel_id='test-owner/knee-test-r6-v1',
             k1_eval_stride=2 if n % 2 else 3)
    if n % 2:
        c['alpha_vector'] = [0, 1]+[.35]*10
    engine = compat.load('v120')
    assert b.artifacts(c)[0] == engine.artifacts(c)[0]
    scalar = copy.deepcopy(c)
    c['r6_eval_stride'] = 2
    _, _, old, new, _, _ = b.package(c)
    assert new['tools/k1sr_runtime_v7.py'] == old['tools/k1sr_runtime_v7.py']
    assert new['tools/k1n_config_v1.py'] == old['tools/k1n_config_v1.py']
    assert b.config(c)['config_sha256'] == b.config(scalar)['config_sha256']
    assert b.config(c)['formula'] == b.config(scalar)['formula']


@pytest.mark.parametrize('filename', EXAMPLES)
@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('fault', [False, True])
def test_actual_baked_runtime_blend_fallback_and_n32(tmp_path, filename, diagnostic, fault):
    c = read(filename)
    files = b.package(c, diagnostic)[3]
    r = ModuleType('r6_baked')
    exec(compile(files['tools/k1sr_runtime_v7.py'], '<actual baked runtime>', 'exec'), r.__dict__)
    rng = np.random.default_rng(228)
    values = [rng.integers(0, 9, (5, 12)).astype(np.float32)/8 for _ in range(4)]
    parent = replay.csv_bytes(list(map(str, range(5))), values[0])
    (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(parent)
    rows = records(r, values[1:])
    if fault:
        rows[0]['predictions'].pop(r.MODEL_IDS[-1])
    worker = tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    receipt = r.finish(tmp_path, tmp_path/'run', b.config(c, diagnostic), 12)
    assert receipt['status'] == ('DEGRADED_PARENT_ONLY' if fault else 'COMPLETE_SHARED_K1N')
    actual = (tmp_path/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')).read_bytes()
    if fault:
        assert actual == parent
    else:
        a = np.asarray(c.get('alpha_vector', c['alpha']))
        expected = rank((1-a)*rank(values[0])+a*rank(np.mean([rank(x) for x in values[1:]], axis=0)))
        np.testing.assert_array_equal(replay.util.table(actual)[1], expected.astype(np.float32))
    if diagnostic:
        assert receipt['never_submit'] and receipt['submission_sha256'] is None
        assert not (tmp_path/'submission.csv').exists()


def test_compat_pin_failure_and_adapter_anchor_failure(monkeypatch):
    monkeypatch.setitem(compat.PINS['v120'], 'k1n_config_v1.py', '0'*64)
    with pytest.raises(ValueError, match='compat source drift'):
        compat.load('v120')
    with pytest.raises(ValueError):
        b.adapter(b'unknown source')


def test_unrelated_identity_cannot_select_old_runtime_by_recipe():
    c = read('x3-stride2-drop-o.json')
    c.update(release_version='1.9.99-cloned-recipe',
             kernel_id='test-owner/knee-cloned-recipe-v1')
    plain = b.package(c)[3]
    changed = b.package(dict(c, r6_eval_stride=2))[3]
    assert plain['tools/k1sr_runtime_v7.py'] == changed['tools/k1sr_runtime_v7.py']
    assert plain['tools/k1n_config_v1.py'] == changed['tools/k1n_config_v1.py']
    assert b.baseline(c)[0].VERSION == '1.2.0'
    assert b.baseline(dict(c, r6_eval_stride=2))[0].VERSION == '1.2.0'


def test_registered_successor_rejects_recipe_drift():
    c = read(EXAMPLES[0])
    c['alpha'] = .20
    with pytest.raises(ValueError, match='registered successor recipe drift'):
        b.package(c)


def test_parent_drift_blocks_prepare_before_destination_creation(monkeypatch, tmp_path):
    original = b.compat.load
    def broken(profile):
        engine = original(profile)
        engine.stage = lambda *a: tmp_path/'corrupt-parent'
        return engine
    (tmp_path/'corrupt-parent').mkdir()
    monkeypatch.setattr(b.compat, 'load', broken)
    monkeypatch.setattr(b, 'stage', lambda *a: tmp_path/'must-not-exist')
    with pytest.raises(ValueError, match='parent artifact inventory drift'):
        b.run('prepare', read(EXAMPLES[0]))
    assert not (tmp_path/'must-not-exist').exists()
