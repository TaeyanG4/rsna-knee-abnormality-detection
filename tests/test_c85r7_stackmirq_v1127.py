"""H-B-172 composed-source and CPU contract regressions; CUDA hidden."""
import copy
import importlib.util
import inspect
import os
from pathlib import Path
import pytest
from tools import c85r7_stackmirq_v1127 as b

# Reuse the independent fluid oracle and its entire numerical/failure suite.
spec=importlib.util.spec_from_file_location('hb172_fluid_tests',Path(__file__).with_name('test_c85r7_stackq_v1125.py'))
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
for name in ('test_oracle_and_unchanged_ten','test_fluid_independent_of_r7_and_uses_reranked_combined_r',
             'test_three_percent_and_sentinel','test_invalid_input'):
    globals()[name]=getattr(legacy,name)


@pytest.fixture(scope='module')
def candidate():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    return b.fluid.math_functions(b.payload()[2])


def test_byte_composition_and_bootstrap():
    s=b.snapshots(); nb,meta,files,blob,proof=b.payload(s)
    p=s['parent']; f=s['fluid']; m=s['mirror']
    assert files[b.HELPER]==f['files'][b.HELPER]
    assert files[b.SOURCE]==m['files'][b.SOURCE]
    assert b.old.digest(files[b.WRAPPER])=='17c8f6139a26c2d637034e88a848953af4b1f0e7797925ce61ef117e32fcf439'
    assert proof['both_merge_orders_equal']
    assert set(files)==set(p['files'])
    assert sum(files[k]==p['files'][k] for k in files)==28
    assert sum(files[k]==f['files'][k] for k in files)==29
    assert sum(files[k]==m['files'][k] for k in files)==29
    assert nb['cells'][:-1]==p['nb']['cells'][:-1]
    # Restore ONLY the three allowed bootstrap substitutions, then exact compare.
    raw=''.join(nb['cells'][-1]['source']).encode()
    assignments=b.old.unpack(b.flipq.restored(nb))[2]
    for key in ('BLOB','PINS'):
        raw=b.old.replace(raw,key+' = '+repr(assignments[key]),key+' = '+repr(p['assignments'][key]))
    raw=b.old.replace(raw,repr(b.old.digest(blob)),repr(b.old.digest(p['blob'])))
    assert raw==''.join(p['nb']['cells'][-1]['source']).encode()
    assert {k:v for k,v in meta.items() if k not in ('id','title')}=={k:v for k,v in p['meta'].items() if k not in ('id','title')}
    assert meta['id']==b.SLUG and meta['title']=='knee-lb-stackmirq-v1'


@pytest.mark.parametrize('source,member',[('fluid',b.SOURCE),('mirror',b.HELPER),('fluid',b.HELPER),('mirror',b.SOURCE)])
def test_reject_source_drift(source,member):
    s=b.snapshots(); s[source]['files'][member]+=b'\n# drift\n'
    with pytest.raises(ValueError): b.merge(s)


@pytest.mark.parametrize('values,expected',[
    ((.957,.958,.957,30000,100),True),((.957,.958,.95699,30000,100),False),
    ((.958,.957,.957,30000,100),True),((.958,.957,.959,32300,99.95),True),
    ((.958,.957,.959,32300,100),False)])
def test_decision(values,expected):
    assert b.decision(*values)==expected


@pytest.mark.parametrize('index,value',[(0,None),(1,None),(2,float('nan')),(3,0),(4,None),(4,-1),(0,2)])
def test_missing_or_invalid_measurements_hold(index,value):
    values=[.95,.96,.96,30000,100]; values[index]=value
    with pytest.raises(ValueError): b.decision(*values)


def test_registration_and_validation():
    reg=b.old.read(b.OUT/'registration.json')
    assert reg==b.registration() and reg['purpose']=='2026-10-17 candidate'
    assert reg['fluid_seconds_allowance']==.05 and not reg['qualified_for_submission']
    assert b.validate()['identical']==28


def test_isolated_actual_collect(tmp_path):
    # The admitted smoke procedure changes only its candidate identity and CPU
    # environment. It imports actual extracted files, outside project cwd.
    source=inspect.getsource(legacy.test_isolated_embedded_import_and_real_collect)
    assert source.count("CUDA_VISIBLE_DEVICES='-1'")==1
    source=source.replace("CUDA_VISIBLE_DEVICES='-1'","CUDA_VISIBLE_DEVICES=''")
    source=source.replace('1.1.25-stackq',b.VERSION)
    namespace=dict(legacy.__dict__,b=b)
    exec(compile(source,'<hb172-isolated-collect>','exec'),namespace)
    namespace['test_isolated_embedded_import_and_real_collect'](tmp_path)


def test_admit_rejects_stale_replay(monkeypatch):
    original=b.old.read
    def read(path):
        if path==b.SCRATCH/'cpu-replay.json': return {'status':'PASS_CPU_REPLAY','kernel_sha256':'stale'}
        return original(path)
    monkeypatch.setattr(b.old,'read',read)
    with pytest.raises(ValueError,match='fresh replay'): b.admit()
