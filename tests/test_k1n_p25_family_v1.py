"""v1.0.0: scored P25 lineage, runtime-only upgrade and flat-P hedge; CPU only."""
import base64
import copy
import json
from types import ModuleType

import numpy as np
import pytest

from tools import k1n_stage_v1 as b
from test_k1n_members_v1 import filled, runtime
from test_k1n_v7 import records, replay

PARENT = b.ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.67-r6w958k1n3s2p25'
CONFIGS = b.ROOT/'configs/k1n'


def read(path):
    return json.loads(path.read_bytes())


def config(flat=False):
    return read(CONFIGS/('p25-dropo'+('-flatp' if flat else '')+'-v10.json'))


@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_exact_scored_lineage_and_notebook(flat, diagnostic):
    c = config(flat)
    pc = read(PARENT/'config.json')
    assert pc == read(CONFIGS/'x3-stride2-drop-o.json')
    fields = {k for k in c.keys() | pc.keys() if c.get(k) != pc.get(k)}
    assert fields == {'release_version','kernel_id','k1_runtime_version',
        'k1_failure_policy','k1_max_failed_fraction','k1_guard_hours'} | ({'p_weight_mode'} if flat else set())
    assert c['models'] == pc['models'] and c['alpha'] == .25 and c['k1_eval_stride'] == 2
    assert 'r6_eval_stride' not in c and 'alpha_vector' not in c
    assert all(m['checkpoint'].endswith('_f0.pt') for m in c['models'])
    comparison = PARENT.with_name(PARENT.name+'-n32') if diagnostic else PARENT
    old_nb = read(comparison/'kernel/release.ipynb')
    old_meta = read(comparison/'kernel/kernel-metadata.json')
    old_blob, old_pins, old = b.before.parent.archive(''.join(old_nb['cells'][2]['source']))
    nb, meta, _, new, _, _ = b.package(c, diagnostic)
    assert meta == dict(old_meta, id=c['kernel_id']+('-n32' if diagnostic else ''),
        title=c['kernel_id'].split('/')[1]+('-n32' if diagnostic else ''))
    changed = {n for n in old.keys() & new.keys() if old[n] != new[n]}
    assert changed == {'exp169_r7_port.py','exp169_release_c85r7.py',
        'exp169_release_c85r7codec.py','tools/k1n_config_v1.py'} | ({'exp169_r224.py'} if flat else set())
    assert old.keys()-new.keys() == {'tools/k1sr_runtime_v7.py'}
    assert new.keys()-old.keys() == {'tools/k1sr_runtime_v10.py'}
    assert new['exp169_r7_port.py'] == old['exp169_r7_port.py'].replace(b'k1sr_runtime_v7',b'k1sr_runtime_v10')
    assert new['exp169_release_c85r7.py'] == old['exp169_release_c85r7.py'].replace(
        b.digest(old['exp169_r7_port.py']).encode(), b.digest(new['exp169_r7_port.py']).encode())
    code = ''.join(nb['cells'][2]['source'])
    blob, pins, _ = b.before.parent.archive(code)
    for a,z in [(repr(base64.b64encode(blob).decode()),repr(base64.b64encode(old_blob).decode())),
                ('PINS = '+repr(pins),'PINS = '+repr(old_pins)),
                (repr(b.digest(blob)),repr(b.digest(old_blob))),
                ('from tools import k1sr_runtime_v10 as _k1sr','from tools import k1sr_runtime_v7 as _k1sr'),
                (repr(b.config(c,diagnostic)),repr(b.config(pc,diagnostic)))]:
        code = b.exact(code,a,z)
    normalized = copy.deepcopy(nb)
    normalized['cells'][2]['source'] = code.splitlines(True)
    normalized['cells'][1] = old_nb['cells'][1]
    assert normalized == old_nb
    outputs, reg = b.artifacts(c,diagnostic)
    assert reg['never_submit'] is diagnostic and not reg['qualified_for_submission']
    assert reg['capture_timeout_seconds'] == 900 and reg['call_timeout_seconds'] == 300
    assert reg['guard_seconds'] == 8.8*3600 and reg['load_timeout_seconds'] == 1500
    assert not reg['parent_is_existing_stage']  # Generic v10 factory's hypothetical v9 parent.


def test_flat_is_one_change_against_public():
    public, flat = config(), config(True)
    assert {k for k in public.keys() | flat.keys() if public.get(k) != flat.get(k)} == {
        'p_weight_mode','release_version','kernel_id'}
    a,z = (b.package(c)[3] for c in (public,flat))
    assert a.keys() == z.keys()
    assert {n for n in a if a[n] != z[n]} == {
        'exp169_r224.py','exp169_release_c85r7codec.py','tools/k1sr_runtime_v10.py',
        'tools/k1n_config_v1.py'}
    assert z['tools/k1sr_runtime_v10.py'] == b.exact(a['tools/k1sr_runtime_v10.py'].decode(),
        'REGISTERED = '+repr(b.canonical(public)), 'REGISTERED = '+repr(b.canonical(flat))).encode()


@pytest.mark.parametrize('flat', [False, True])
def test_successful_blend_bit_exact_to_scored_runtime(flat):
    pc = read(PARENT/'config.json')
    nb = read(PARENT/'kernel/release.ipynb')
    old = b.before.parent.archive(''.join(nb['cells'][2]['source']))[2]
    r7 = ModuleType('scored_p25_v7')
    exec(compile(old['tools/k1sr_runtime_v7.py'], '<scored runtime>', 'exec'), r7.__dict__)
    r10 = runtime(config(flat))
    rng = np.random.default_rng(67079)
    parent, *arrays = [rng.integers(0,9,(32,12)).astype(np.float32)/10 for _ in range(4)]
    payload = replay.csv_bytes(list(map(str,range(32))),parent)
    assert r7.blend(payload,records(r7,arrays)) == r10.blend(payload,records(r10,arrays))


@pytest.mark.parametrize('flat', [False, True])
def test_fullbag_template_only_adds_equal_fold0_member(flat):
    path = CONFIGS/('p25-fb-dropo'+('-flatp' if flat else '')+'-v10.template.json')
    raw, parent = read(path), config(flat)
    assert raw['models'][:3] == parent['models']
    assert len(raw['models']) == 4 and raw['member_weights'] == [.25]*4
    assert all(m['checkpoint'].endswith('_f0.pt') for m in raw['models'])
    assert raw['models'][3]['id'] == 'k1_v3g_fullbag'
    assert raw['models'][3]['dataset_slug'] == 'taeyangg4/knee-k1-v3gfb-v1-assets'
    assert {k for k in raw.keys() | parent.keys() if raw.get(k) != parent.get(k)} == {
        'models','member_weights','release_version','kernel_id'}
    with pytest.raises(ValueError):
        b.run('prepare',raw)
    c = filled(path)
    _, meta, _, _, _, _ = b.package(c)
    parent_meta = b.package(parent)[1]
    assert set(meta['dataset_sources']) == set(parent_meta['dataset_sources']) | {
        'taeyangg4/knee-k1-v3gfb-v1-assets'}
    r = runtime(c)
    assert r.MODEL_IDS == tuple(m['id'] for m in raw['models'])
    assert 'member_weights' not in b.canonical(c)  # Equal mean canonicalization.
