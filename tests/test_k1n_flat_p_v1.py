"""Flat P arithmetic, registration, receipt and immutable public replay tests."""
import json

import numpy as np
import pytest

from tools import k1n_stage_v1 as b
from tools import k1n_flat_p_v1 as flat
from tools import k1n_flat_p_check_v1 as check
from tools import k1n_cpu_replay_v1 as cpu
from test_k1n_v9 import finish_case


@pytest.mark.parametrize('mode',[None, True, 0, 'FLAT', '', [], {}])
def test_invalid_modes(mode):
    with pytest.raises(ValueError,match='p_weight_mode'):
        b.schema.validate(dict(check.configuration(),p_weight_mode=mode))


def test_flat_requires_v9_and_new_identity():
    c = check.configuration()
    c.pop('k1_runtime_version')
    with pytest.raises(ValueError,match='runtime v9'):
        b.config(c)
    with pytest.raises(ValueError,match='new'):
        b.config(dict(b.runtime9.registered(),p_weight_mode='flat'))


@pytest.mark.parametrize('diagnostic',[False,True])
def test_public_replay(diagnostic):
    c = b.runtime9.registered()
    assert b.artifacts(c,diagnostic) == b.artifacts(dict(c,p_weight_mode='public'),diagnostic)
    output,_ = b.artifacts(c,diagnostic)
    assert all((b.stage(c,diagnostic)/n).read_bytes() == v for n,v in output.items())


@pytest.mark.parametrize('drop',[False,True])
@pytest.mark.parametrize('diagnostic',[False,True])
def test_scope_and_binding(drop, diagnostic):
    c = check.fixture_c5() if drop else check.configuration()
    nb,meta,old,new,old_nb,old_meta = b.package(c,diagnostic)
    assert {n for n in new if new[n] != old[n]} == {flat.HELPER,flat.CODEC,flat.SCHEMA,flat.RUNTIME}
    # Runtime change is exactly its registration, not imports/timing/ranks/failure logic.
    pc, cc = flat.parent_config(flat.canonical(c)), flat.canonical(c)
    assert new[flat.RUNTIME] == old[flat.RUNTIME].replace(
        ('REGISTERED = '+repr(pc)).encode(), ('REGISTERED = '+repr(cc)).encode())
    outputs,reg = b.artifacts(c,diagnostic)
    assert reg['p_weight_mode'] == 'flat' and reg['p_raptor_weights'] == [.5]*12
    assert reg['config_sha256'] == b.digest(b.util.encoded(cc))
    assert reg['never_submit'] is diagnostic and not reg['qualified_for_submission']
    assert flat.FORMULA in reg['formula']
    assert new['exp169_release_c85.py'] == old['exp169_release_c85.py']
    assert b.config(c,diagnostic).get('runtime_projection') == b.config(pc,diagnostic).get('runtime_projection')
    assert json.loads(outputs['runtime-projection.json']) == b.members.projection(
        {k:v for k,v in cc.items() if k != 'p_weight_mode'})
    if drop:
        assert b'weight = np.asarray('+repr(flat.FLAT).encode()+b')' in new[flat.CODEC]


@pytest.mark.parametrize('failures',[False,True])
def test_synthetic_math_and_failure_survivors(failures):
    rng = np.random.default_rng(177)
    arrays = [rng.integers(0,17,(100,12))/17 for _ in range(3)]
    masks = [np.zeros(100,dtype=bool) for _ in range(3)]
    if failures:
        masks[0][[0,1,3,4]] = True
        masks[1][[0,1,2]] = True
        masks[2][[0,2,4,5]] = True
    _,_,old,new,_,_ = b.package(check.configuration())
    _,_,result = check.compare(arrays,masks,old,new)
    assert result['changed_columns'] == list(flat.CHANGED)
    masks[1][:4] = True
    for files in (old,new):
        with pytest.raises(ValueError,match='more than 3%'):
            check.math_scope(files)['p_branch'](*arrays,*masks)


@pytest.mark.parametrize('diagnostic',[False,True])
def test_drop_o_actual_boundaries(tmp_path,diagnostic):
    c = check.fixture_c5()
    files = b.package(c,diagnostic)[3]
    rng = np.random.default_rng(23)
    arrays = [rng.random((100,12)) for _ in range(3)]
    result,payload = cpu.replay(files,tmp_path/'boundary',list(map(str,range(100))),arrays,
        failures={'R':{'1':'fixture'},'R224':{'2':'fixture'},'S_R7':{'3':'fixture'}},diagnostic=diagnostic)
    assert result['p_branch_exact'] and not result['o_executed']


@pytest.mark.parametrize('diagnostic',[False,True])
@pytest.mark.parametrize('fault',[None,'identity','guard'])
def test_receipt_mode_and_failure_binding(tmp_path,monkeypatch,diagnostic,fault):
    c = check.configuration()
    rec = finish_case(tmp_path,monkeypatch,c=c,diagnostic=diagnostic,fault=fault,failures=4)
    assert rec['p_weight_mode'] == 'flat' and rec['p_raptor_weights'] == [.5]*12
    assert rec['config_sha256'] == b.config(c)['config_sha256']
    assert flat.FORMULA in rec['formula']


def test_c5_templates_preserve_placeholders():
    parent = json.loads((b.ROOT/'configs/k1n/c5-dropo-p25-v9.template.json').read_bytes())
    c = json.loads((b.ROOT/'configs/k1n/c5-dropo-p25-flatp.template.json').read_bytes())
    assert {k for k in parent.keys() | c.keys() if parent.get(k) != c.get(k)} == {
        'release_version','kernel_id','p_weight_mode'}
    with pytest.raises(ValueError):
        b.run('prepare',c)
    n32 = json.loads((b.ROOT/'configs/k1n/c5-dropo-p25-flatp-n32.template.json').read_bytes())
    assert n32['config'] == c and n32['never_submit'] and n32['diagnostic']
    with pytest.raises(ValueError):
        b.run('prepare',n32)
