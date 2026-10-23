"""Guard registration boundaries, immutable defaults and full inverse diff."""
import base64
import copy
import json
from types import ModuleType

import pytest

from tools import k1n_stage_v1 as b
from test_k1n_v8 import NAMES, configuration


def candidate(name=NAMES[0], hours=8.6):
    c = configuration(name)
    c['release_version'] = c['release_version'].replace('1.1.71', '1.1.72')
    c['kernel_id'] = c['kernel_id'][:-1]+'2'
    c['k1_guard_hours'] = hours
    return b.schema.validate(c)


@pytest.mark.parametrize('hours', [True, False, None, '8.6', 6.99, 8.8001,
                                  float('inf'), float('-inf'), float('nan'), [], {}])
def test_reject_guard(hours):
    c = configuration()
    c['k1_guard_hours'] = hours
    with pytest.raises(ValueError, match='k1_guard_hours'):
        b.schema.validate(c)


@pytest.mark.parametrize('hours', [7, 7.0, 8, 8.0, 8.6, 8.8])
def test_guard_range(hours):
    c = candidate(hours=hours)
    assert c.get('k1_guard_hours', 8.0) == hours
    assert 'k1_guard_hours' not in c or type(c['k1_guard_hours']) is float


def test_guard_requires_v8():
    c = configuration()
    c.pop('k1_failure_policy')
    c.pop('k1_max_failed_fraction')
    c['k1_guard_hours'] = 8.6
    with pytest.raises(ValueError, match='requires runtime v8'):
        b.schema.validate(c)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_default_regenerates_every_171_artifact(name, diagnostic, explicit):
    c = configuration(name)
    if explicit:
        c['k1_guard_hours'] = 8.0
    outputs, reg = b.artifacts(c, diagnostic)
    dest = b.stage(c, diagnostic)
    assert set(outputs) == {p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
    assert all((dest/n).read_bytes() == data for n, data in outputs.items())
    assert reg['guard_seconds'] == 28800


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_guard_full_inverse_diff(name, diagnostic):
    c, p = candidate(name), configuration(name)
    nb, meta, old, new, old_nb, old_meta = b.package(c, diagnostic)
    assert new.keys() == old.keys()
    assert {n for n in old if old[n] != new[n]} == {
        'tools/k1sr_runtime_v8.py', 'exp169_release_c85r7codec.py'}
    text = new['tools/k1sr_runtime_v8.py'].decode()
    text = b.exact(text, 'REGISTERED = '+repr(b.guard.without_guard(c)), 'REGISTERED = '+repr(p))
    text = b.exact(text, 'GUARD_SECONDS = 30960.0', 'GUARD_SECONDS = 8 * 3600')
    text = b.exact(text, 'CONFIG_SHA = '+repr(b.digest(b.util.encoded(c))),
                   'CONFIG_SHA = hashlib.sha256(util.encoded(REGISTERED)).hexdigest()')
    assert text.encode() == old['tools/k1sr_runtime_v8.py']
    oc, nc = b.config(p, diagnostic), b.config(c, diagnostic)
    text = new['exp169_release_c85r7codec.py'].decode()
    text = text.replace(nc['release_version'], oc['release_version'])
    text = b.exact(text, nc['kernel_id'], oc['kernel_id'])
    assert text.encode() == old['exp169_release_c85r7codec.py']
    normalized = copy.deepcopy(nb)
    code = ''.join(nb['cells'][2]['source'])
    blob, pins, _ = b.before.parent.archive(code)
    prior, prior_pins, _ = b.before.parent.archive(''.join(old_nb['cells'][2]['source']))
    code = b.exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(prior).decode()))
    code = b.exact(code, 'PINS = '+repr(pins), 'PINS = '+repr(prior_pins))
    code = b.exact(code, repr(b.digest(blob)), repr(b.digest(prior)))
    code = b.exact(code, repr(nc), repr(oc))
    normalized['cells'][2]['source'] = code.splitlines(True)
    normalized['cells'][1]['source'] = [
        x.replace(nc['release_version'], oc['release_version']) for x in normalized['cells'][1]['source']]
    assert normalized == old_nb
    assert meta == dict(old_meta, id=nc['kernel_id'], title=nc['kernel_id'].split('/')[1])
    outputs, reg = b.artifacts(c, diagnostic)
    assert reg['k1_guard_hours'] == 8.6 and reg['guard_seconds'] == 30960
    assert reg['config_sha256'] == b.digest(b.util.encoded(c))
    assert reg['never_submit'] is diagnostic and not reg['qualified_for_submission']
    assert len(json.loads(outputs['payload-scope.json'])['untouched']) == 37


@pytest.mark.parametrize('hours', [7.0, 8.6, 8.8])
def test_baked_guard_and_hash(hours, monkeypatch):
    c = candidate(hours=hours)
    r = ModuleType('guard_boundary')
    exec(compile(b.package(c)[3]['tools/k1sr_runtime_v8.py'], '<guard>', 'exec'), r.__dict__)
    assert r.GUARD_SECONDS == hours*3600
    assert r.CONFIG_SHA == b.digest(b.util.encoded(c))
    monkeypatch.setenv('K1SR_STARTED', '1000')
    monkeypatch.setattr(r.time, 'time', lambda: 1000+hours*3600-.01)
    assert r.remaining() > 0
    monkeypatch.setattr(r.time, 'time', lambda: 1000+hours*3600)
    assert r.remaining() == 0


def test_recipe_and_identity_drift_rejected():
    c = candidate()
    c['alpha_vector'][0] = .11
    with pytest.raises(ValueError, match='recipe drift'):
        b.package(c)
    c = candidate()
    c['kernel_id'] += '-unknown'
    with pytest.raises(ValueError, match='parent identity'):
        b.package(c)


@pytest.mark.parametrize('name', NAMES)
def test_written_config_validates_both_immutable_stages(name):
    """Exercise the on-disk input, including inherited repr key ordering."""
    path = b.ROOT/'configs/k1n'/f'{name}-v8-guard86.json'
    result = b.run('validate', json.loads(path.read_bytes()))
    assert result['status'] == 'PASS_CPU_VALIDATE'
    assert [r['never_submit'] for r in result['stages']] == [False, True]
