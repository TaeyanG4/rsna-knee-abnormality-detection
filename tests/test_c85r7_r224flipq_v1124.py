"""H-B-167 strict FULL flip scope, fail-closed decisions and candidate replay."""
import copy
import importlib.util
import types

import pytest

from tools import c85r7_r224flipq_v1124 as b


def parent_tests():
    spec = importlib.util.spec_from_file_location('hb167_parent_tests', b.ROOT/'tests/test_c85r7_r224q_v1123.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def candidate():
    mod = types.ModuleType('hb167_candidate')
    mod.__file__ = str(b.SCRATCH/'candidate-runtime'/b.WRAPPER)
    exec(compile(b.payload()[2][b.WRAPPER], b.WRAPPER, 'exec'), mod.__dict__)
    return mod


def test_complete_exact_scope():
    nb, meta, files, blob, manifest, full = b.artifacts()
    parent_nb, original, _, _ = b.authenticated(b.BASE,b.PARENT_RECEIPT_SHA)
    reference = b.authenticated(b.REFERENCE,b.FLIP_RECEIPT_SHA)[1]
    assert len(files)==31 and len(manifest['unchanged'])==28
    assert {x['name'] for x in manifest['changed']}=={b.PORT,b.CHAIN,b.WRAPPER}
    assert manifest['added']==manifest['removed']==[]
    assert files[b.PORT]==reference[b.PORT] and files[b.CHAIN]==reference[b.CHAIN]
    assert b.old.FLAG_OLD.encode() not in files[b.PORT]
    for item in manifest['unchanged']:
        assert files[item['name']]==original[item['name']]
    assert files['exp169_r224.py']==original['exp169_r224.py']
    assert files['raptor224_run.py']==original['raptor224_run.py']
    assert nb['cells'][:2]==parent_nb['cells'][:2]
    assert meta['id']==b.SLUG and meta['title']=='knee-lb-r224flipq-v1'
    assert '/' not in meta['title']
    assert all(name.encode() in full for name in files)
    assert b.scope(nb,meta)==(files,blob)


@pytest.mark.parametrize('kind',['fast','bootstrap','guard','metadata','archive'])
def test_scope_rejects_mutation(kind):
    nb,meta,_,_=b.payload()
    nb,meta=copy.deepcopy(nb),copy.deepcopy(meta)
    if kind=='metadata': meta['enable_gpu']=False
    else:
        index={'fast':0,'bootstrap':1,'guard':2,'archive':2}[kind]
        nb['cells'][index]['source'].append('# unauthorized change\n')
    with pytest.raises(Exception,match='forbidden notebook/metadata diff'): b.scope(nb,meta)


@pytest.mark.parametrize('values,expected',[
    ((.957,.956,-.001,-.002,8.5),True),
    ((.956,.956,0,0,7.1),False),
    ((.957,.955999,0,0,7.1),False),
    ((.957,.957,-.001001,0,7.1),False),
    ((.957,.957,0,-.002001,7.1),False),
    ((.957,.957,0,0,8.50001),False),
])
def test_registered_thresholds(values,expected):
    assert b.decision(*values) is expected


@pytest.mark.parametrize('index',range(5))
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),None,True])
def test_nonfinite_decisions_reject(index,bad):
    values=[.957,.957,0,0,7.1]; values[index]=bad
    with pytest.raises(Exception,match='finite measured'): b.decision(*values)


@pytest.mark.parametrize('fails',[False,True])
def test_candidate_collect_real3_with_synthetic_r224(tmp_path,candidate,fails):
    # Real completed visible branch inputs; R224 is explicitly a synthetic
    # surrogate. This checks serving arithmetic/authentication, not new scores.
    parent_tests().test_collect_real_visible_predictions_with_explicit_synthetic_r224(tmp_path,candidate,fails)


@pytest.mark.parametrize('fail224',[False,True])
def test_candidate_launch_commands_unchanged(tmp_path,monkeypatch,candidate,fail224):
    parent_tests().test_new_subprocess_preserves_existing_commands(tmp_path,monkeypatch,candidate,fail224)


def test_candidate_fast_77_case_matrix(tmp_path,monkeypatch):
    from tools import c85r7_own25q3_cpu_fixtures as fixtures
    monkeypatch.setattr(fixtures,'b',types.SimpleNamespace(ROOT=b.ROOT,OUT=b.OUT,old=b.old,
        VISIBLE_IDS=b.flipq.fast.VISIBLE_IDS,reference=b.flipq.fast.reference))
    results=[fixtures.run_case(tmp_path/name,**kw) for name,kw in fixtures.case_specs()]
    assert len(results)==77 and all(r['status']=='PASS' for r in results)


def test_admission_requires_current_smoke(monkeypatch):
    original=b.old.read
    monkeypatch.setattr(b.old,'read',lambda path: {'status':'PASS','kernel_sha256':'stale'}
                        if path==b.SCRATCH/'smoke.json' else original(path))
    with pytest.raises(Exception,match='CPU smoke missing/stale'): b.admit()


def test_no_gate_copy_or_qualification():
    assert not b.registration()['qualified_for_submission']
    assert b.registration()['gate_binding'].startswith('DEFERRED')
    assert b.registration()['empirical_hidden']['planning_upper_hours']==7.1
    assert b.registration()['timing']['max_projected_hours']==8.5
