"""HB166 engineering-serving CPU contracts; no label reads or GPU execution."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from tools import hb166_release_v1 as b, hb166_serving_v1 as s


@pytest.fixture(scope='module')
def recipe():
    return s.recipe(b.WORK/'hb166-sources.json', b.old.digest(b.source_payload()))


@pytest.fixture
def c85(monkeypatch):
    path = b.WORK/'candidate-runtime/exp169_release_c85.py'
    spec = importlib.util.spec_from_file_location('hb166_test_c85', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_verbatim_spans_and_shared_class_identity(recipe):
    payload = json.loads(b.source_payload())
    for entry in payload['modules'].values():
        raw = (b.ROOT/entry['path']).read_bytes()
        assert b.old.digest(raw) == entry['sha256']
        for span in entry['spans']:
            assert b''.join(raw.splitlines(True)[span['start']-1:span['end']]) == base64.b64decode(span['bytes'])
    assert recipe.mv.h is recipe.mil and recipe.crops.MILWindows is recipe.mil.MILWindows
    assert recipe.mv.crops is recipe.crops and recipe.mv.native is recipe.native


def test_corrupt_payload_rejected(tmp_path):
    path = tmp_path/'source.json'
    path.write_bytes(b.source_payload()+b' ')
    with pytest.raises(ValueError, match='payload drift'):
        s.recipe(path,b.old.digest(b.source_payload()))


def test_manifest_missing_fold_and_wrong_targets(tmp_path):
    m = b.manifest()
    # The actual draft must reject the missing fold, not duplicate fold2.
    with pytest.raises(ValueError, match='required fold missing'):
        s.roster(b.OUT/'assets',s.sha(b.OUT/'assets/manifest.json'))
    m['targets'] = list(reversed(m['targets']))
    (tmp_path/'manifest.json').write_bytes(b.encode(m))
    with pytest.raises(ValueError, match='target order'):
        s.roster(tmp_path,s.sha(tmp_path/'manifest.json'))


def test_wrong_checkpoint_identity(recipe):
    entry = b.manifest()['roster'][0]
    entry['identity'] = {'wrong': True}
    with pytest.raises(ValueError,match='identity mismatch'):
        s.model(recipe,b.FIT/'final.pt',entry,'cpu')


def test_directory_serving_and_archive_layout(tmp_path):
    query = 'SELECT path,series,info FROM files WHERE split=? AND uid=? ORDER BY path'
    for split,folder in [('test','test_series'),('train','train_images')]:
        path=tmp_path/folder/'1'/'2'/'3.dcm'
        path.parent.mkdir(parents=True); path.write_bytes(b'dicom-transport-test')
        rows=s.DirectoryIndex(tmp_path).execute(query,(split,'1')).fetchall()
        assert len(rows)==1 and Path(rows[0][0])==path and rows[0][1]=='2'


@pytest.mark.parametrize('laterality,failures,empty,raises',[
    ('unknown',[],False,False),('left',[],False,False),
    ('unknown',['laterality_conflict'],False,True),('unknown',[],True,True)])
def test_partial_planes_unknown_conflicting_side(recipe,monkeypatch,laterality,failures,empty,raises):
    full=np.zeros((64,448,448),np.uint8); mask=np.zeros(64,np.uint8)
    if not empty: mask[:20]=1
    monkeypatch.setattr(recipe.raw,'build_study',lambda task:(None,None,full,mask,{}))
    monkeypatch.setattr(recipe.geometry,'build_study',lambda *args:(full,mask,dict(failures=failures,laterality=laterality)))
    if raises:
        with pytest.raises(ValueError): s.preprocess(recipe,'1',[])
    else:
        f,c,_=s.preprocess(recipe,'1',[])
        assert np.array_equal(f.mask('1'),mask) and np.array_equal(c.mask('1'),mask)
        assert np.all(c._side_flags==int(laterality=='unknown'))
        x,*_=s.dataset(recipe,['1'],f,c)[0]
        assert x.shape==(72,3,224,224)


def test_missing_assets_propagate_to_actual_parent_survivor(c85,tmp_path):
    ids=['1','2','3']; (tmp_path/'ids.json').write_text(json.dumps(ids))
    s.run(c85,tmp_path,tmp_path,asset_dir=tmp_path/'absent',device='cpu')
    q,fail=s.authenticated(c85,tmp_path,ids)
    assert set(fail)==set(ids) and np.all(q==.5)
    a=np.tile(np.array([.2,.7,.9])[:,None],(1,12)); z=a[::-1].copy()
    actual=c85.blend(ids,a,a,a,z,q,{'Q_A5q':fail})
    expected=.85*c85.rank(c85.rank(a))+.15*c85.rank((c85.rank(a)+c85.rank(z))/2)
    np.testing.assert_array_equal(actual,expected)
    with pytest.raises(ValueError,match='journal order'):
        s.authenticated(c85,tmp_path,ids[::-1])
    c85.p.LABELS=list(reversed(s.TARGETS))
    try:
        with pytest.raises(ValueError,match='parent target order'): s.authenticated(c85,tmp_path,ids)
    finally: c85.p.LABELS=list(s.TARGETS)


@pytest.mark.parametrize('broken',[False,True])
def test_probability_mean_and_no_partial_roster(c85,tmp_path,monkeypatch,broken):
    ids=['1']; (tmp_path/'ids.json').write_text(json.dumps(ids))
    import pandas as pd
    pd.DataFrame({'StudyInstanceUID':ids}).to_csv(tmp_path/'test_series.csv',index=False)
    monkeypatch.setattr(s,'roster',lambda _: {'roster':[{'file':'a'},{'file':'b'}]})
    monkeypatch.setattr(s,'recipe',lambda _:None)
    monkeypatch.setattr(s,'configure_transport',lambda *args:None)
    monkeypatch.setattr(s,'model',lambda r,path,*args: path.name)
    monkeypatch.setattr(s,'preprocess',lambda *args:(None,None,None))
    def predict(r,m,*args):
        if broken and m=='b': raise ValueError('second member failed')
        return np.full((1,12),.2 if m=='a' else .8,np.float32)
    monkeypatch.setattr(s,'predict',predict)
    s.run(c85,tmp_path,tmp_path,asset_dir=tmp_path,device='cpu')
    values,failures=s.authenticated(c85,tmp_path,ids)
    assert bool(failures)==broken
    np.testing.assert_allclose(values,.5,atol=0,rtol=0)


def test_embedded_entrypoint_and_exact_diff(c85,monkeypatch,tmp_path):
    _,files,nb,meta,_,_=b.payload()
    assert b.validate()['qualified_for_submission'] is False
    p=b.previous.snapshots()['parent']
    assert nb['cells'][:-1]==p['nb']['cells'][:-1]
    assert meta['title']==b.SLUG.split('/')[1]
    assert files['raptor224_run.py']==p['files']['raptor224_run.py']
    assert files['r7_localize.py']==p['files']['r7_localize.py'] if 'r7_localize.py' in files else True
    for name in ('blend','a172_source','phase','run_public'):
        import ast
        def node(raw):
            return ast.dump(next(n for n in ast.parse(raw).body if getattr(n,'name',None)==name))
        assert node(files['exp169_release_c85.py'])==node(p['files']['exp169_release_c85.py'])
    # Exercise __import__(__name__) in the actual embedded patched body.
    monkeypatch.setitem(sys.modules,'hb166_serving_v1',NS(run=lambda module,*args: module is c85))
    assert c85.run_q(NS(reset_backend=lambda:None),None,tmp_path,tmp_path) is True
