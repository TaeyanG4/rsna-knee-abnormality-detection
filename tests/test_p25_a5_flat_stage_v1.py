"""Flat-P/A5 composition: exact payload scope, arithmetic and offline admission."""
from functools import lru_cache
import ast
import json
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from tools import p25_a5_flat_stage_v1 as b
from tools import k1n_cpu_replay_v1 as cpu
from tools import k1n_flat_p_check_v1 as mathcheck
from tools import k1_admit_v1 as admit
from test_k1_admit_v1 import case, change, put, transport
from test_k1_admit_a5_v1 import a5_case, refresh


@lru_cache(None)
def package(diagnostic=False): return b.package(diagnostic)


def runtime(files):
    m = ModuleType('baked_flat_a5')
    exec(compile(files[b.RUNTIME], '<baked flat A5 runtime>', 'exec'), m.__dict__)
    return m


@pytest.mark.parametrize('diagnostic', [False, True])
def test_exact_parent_scope(diagnostic):
    po, pr, oc, c, cfg, onb, nb, om, meta, old, new = package(diagnostic)
    outputs, reg = b.artifacts(diagnostic)
    assert {f['field'] for f in json.loads(outputs['field-level-diff.json'])} == {
        'release_version','kernel_id','p_weight_mode'}
    assert {n for n in old if old[n] != new[n]} == {b.RUNTIME,b.CODEC,b.flat.HELPER,b.flat.SCHEMA}
    assert set(new) == set(old)
    assert all(new[n] == old[n] for n in b.parent.MODULES)
    assert {k for k in om if om[k] != meta[k]} == {'id','title'}
    assert nb['cells'][:1] == onb['cells'][:1] and nb['cells'][3:] == onb['cells'][3:]
    assert c['models'] == oc['models'] and len(c['models']) == 3
    assert c['alpha'] == .25 and c['drop_o'] and c['k1_eval_stride'] == 2
    assert cfg['p_raptor_weights'] == [.5]*12 and cfg['k1_runtime_version'] == 10
    assert reg['a5_contract'] == pr['a5_contract'] and reg['a5_contract']['weight'] == .15
    assert reg['config_sha256'] == b.digest(b.util.encoded(c))
    assert reg['never_submit'] is diagnostic and not reg['qualified_for_submission']
    assert runtime(new).CONFIG_SHA == cfg['config_sha256']
    assert runtime(new).VERSION == '10.0.1-a5'
    scope = json.loads(outputs['payload-scope.json'])
    assert scope['k1_functions_identical'] and len(scope['untouched']) == len(old)-4


@pytest.mark.parametrize('diagnostic',[False, True])
@pytest.mark.parametrize('missing',[False, True])
def test_full_flat_p_a5_k1_math_and_finish(tmp_path, diagnostic, missing):
    *_, cfg, onb, nb, om, meta, old, new = package(diagnostic)
    r = runtime(new)
    rng = np.random.default_rng(182)
    n = 32
    ids = list(map(str,range(n)))
    arrays = [rng.integers(1,31,(n,12))/32 for _ in range(3)]
    masks = [np.zeros(n,bool) for _ in range(3)]
    before, after, comparison = mathcheck.compare(arrays,masks,old,new)
    assert comparison['changed_columns'] == [5,6,8,9]
    result, payload = cpu.replay(new,tmp_path/'parent',ids,arrays,diagnostic=diagnostic)
    assert result['p_branch_exact'] and not result['o_executed']
    raw = rng.integers(0,9,(5,n,12))/8
    mask = np.ones(n,bool)
    if missing: mask[:3] = False; raw[:,~mask] = np.nan
    pp = b.parent.a5.blend(payload,raw,mask)
    rank = lambda a: pd.DataFrame(a).rank(pct=True).to_numpy()
    ranked = [(pd.DataFrame(f[mask]).rank().to_numpy()-1)/(mask.sum()-1) for f in raw]
    mean = np.mean(ranked,axis=0).astype(np.float32)
    mixed = rank(after.astype(np.float32))
    mixed[mask] = .85*mixed[mask]+.15*rank(mean)
    expected_pp = rank(mixed).astype(np.float32)
    np.testing.assert_array_equal(b.util.table(pp)[1],expected_pp)
    ka = [rng.integers(0,8,(n,12))/7 for _ in r.MODEL_IDS]
    rows = [dict(uid=uid,ok=True,config_sha256=r.CONFIG_SHA,k1_models=list(r.MODEL_IDS),
        finished_elapsed_seconds=1., predictions={m:ka[j][i].tolist() for j,m in enumerate(r.MODEL_IDS)})
        for i,uid in enumerate(ids)]
    dest = tmp_path/'finish'; dest.mkdir()
    (dest/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(payload)
    worker = dest/'run/pr7/pr7-run/s_r7/controller/worker0'; worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    put(dest/'a5-receipt.json',dict(status='COMPLETE_A5',original_p_sha256=b.digest(payload),
                                   pprime_sha256=b.digest(pp)))
    rec = r.finish(dest,dest/'run',cfg,12.,parent_override=pp)
    actual = b.util.table((dest/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')).read_bytes())[1]
    kr = rank(np.mean([rank(a.astype(np.float32)) for a in ka],axis=0))
    expected = rank(.75*rank(expected_pp)+.25*kr).astype(np.float32)
    np.testing.assert_array_equal(actual,expected)
    assert rec['p_raptor_weights'] == [.5]*12 and rec['parent_mode'] == 'P25_A5_PPRIME'
    assert rec['a5_receipt_sha256'] == b.util.sha(dest/'a5-receipt.json')
    assert (dest/'k1sr-parent.csv').read_bytes() == pp
    if diagnostic: assert rec['never_submit'] and not (dest/'submission.csv').exists()


@pytest.fixture
def flat_case(a5_case):
    c = dict(package()[3], release_version='1.2.3-k1',kernel_id='owner/k1flat')
    c = b.parent.legacy.canonical(c)
    cfg = b.config(c)
    stage = a5_case['stage']
    a5_case['kernel'] = c['kernel_id']
    change(stage/'kernel/kernel-metadata.json',lambda m:m.update(id=c['kernel_id']))
    put(stage/'config.json',dict(c,a5=b.parent.contract()))
    change(stage/'registration.json',lambda r:r.update(cfg,a5_contract=b.parent.contract(),
        metadata_sha256=b.util.sha(stage/'kernel/kernel-metadata.json')))
    for key in ('visible_dir','n32_dir'):
        d = a5_case[key]
        diagnostic = key == 'n32_dir'
        change(d/'k1sr-receipt.json',lambda r:r.update(b.config(c,diagnostic)))
        if diagnostic:
            change(d/'k1sr-receipt.json',lambda r:r.update(per_study=[
                dict(uid=str(i),ok=True,config_sha256=cfg['config_sha256']) for i in range(32)]))
        change(d/'a5-receipt.json',lambda a:a.update(formula=b.parent.a5.FORMULA,
             checkpoint_hashes=b.parent.contract()['checkpoint_sha256']))
        refresh(d)
        if not diagnostic: transport(d,kernel=c['kernel_id'])
    return a5_case


def test_flat_a5_real_offline_wrapper(flat_case,capsys):
    out = admit.prepare(**flat_case)
    first = {p.name:p.read_bytes() for p in out.iterdir()}
    assert 'ADMITTED_OFFLINE' in capsys.readouterr().out
    assert json.loads(first['ordinary-visible-registration.json'])['tool_version'] == '1.0.6'
    admit.prepare(**flat_case)
    assert first == {p.name:p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize('key,value',[('p_weight_mode','public'),('p_raptor_weights',[.2]*12),
    ('config_sha256','0'*64),('registered_config_sha256','0'*64),('formula','wrong')])
@pytest.mark.parametrize('location',['stage','visible_dir','n32_dir'])
def test_flat_a5_tamper_refused(flat_case,location,key,value):
    path = flat_case[location]/('registration.json' if location == 'stage' else 'k1sr-receipt.json')
    change(path,lambda r:r.update({key:value}))
    if location != 'stage': refresh(flat_case[location])
    with pytest.raises(ValueError): admit.validate(**flat_case)


def test_ordinary_a5_bytes_match_v105(a5_case,monkeypatch):
    old = ModuleType('admission105')
    old.__file__ = str(b.ROOT/'tools/k1_admit_v1.py')
    backup = b.ROOT/'.local/backups/astra-b1/a5-flat-v1/k1_admit_v1.py'
    exec(compile(backup.read_bytes(),str(backup),'exec'),old.__dict__)
    monkeypatch.setattr(old,'TEMPLATE',admit.TEMPLATE)
    monkeypatch.setattr(old,'LEDGER',admit.LEDGER)
    out = old.prepare(**a5_case)
    before = {p.name:p.read_bytes() for p in out.iterdir()}
    admit.prepare(**a5_case)
    assert before == {p.name:p.read_bytes() for p in out.iterdir()}
    assert old.validate(**a5_case)[0].checks == admit.validate(**a5_case)[0].checks


def test_static_admission_hold():
    assert b.run('admit',True)['status'] == 'HOLD_NOT_ADMITTED'


@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('kind', ['repair', 'a5', 'flat', 'failed'])
def test_isolated_embedded_imports_and_historical_regression(diagnostic, kind):
    if kind == 'repair':
        notebook = package(diagnostic)[6]
        label = b.stage(diagnostic).name
    else:
        base = {'a5': b.PARENT, 'flat': b.FLAT_PARENT,
                'failed': b.PARENT.parent/'v1.1.82-r6w958k1n3s2p25a5flatpv10'}[kind]
        path = base.with_name(base.name+('-n32' if diagnostic else ''))
        notebook = json.loads((path/'kernel/release.ipynb').read_bytes())
        label = path.name
    if kind == 'failed':
        with pytest.raises(ValueError, match='unknown config fields'):
            b.selfcheck.check(notebook, label)
    else:
        result = b.selfcheck.check(notebook, label)
        assert result['isolated'] and not result['project_tools_on_sys_path']
        assert result['diagnostic'] is diagnostic
        assert result['a5_imported'] is (kind != 'flat')


def test_selfcheck_is_mandatory_before_stage_creation(monkeypatch):
    def refuse(*args):
        raise ValueError('sentinel embedded failure')
    monkeypatch.setattr(b.selfcheck, 'check', refuse)
    with pytest.raises(ValueError, match='sentinel embedded failure'):
        b.run('prepare')
