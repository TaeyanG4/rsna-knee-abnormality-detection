"""v6 numerical, failure, scheduling and isolated packaging contracts (CPU)."""
import ast
import copy
import inspect
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from scipy.stats import rankdata

from tools import exp169_release_v6 as v


@pytest.fixture(autouse=True)
def restore_modules():
    # v6 explicitly binds the frozen modules; avoid leaking that binding to
    # other release suites in the same pytest process.
    import sys
    import tools
    names = ('exp169_partner_bundle_release','exp169_release_v2','exp152_own0949_release')
    modules = {n:sys.modules.get('tools.'+n) for n in names}
    attrs = {n:getattr(tools,n,None) for n in names}
    yield
    for name in names:
        if modules[name] is not None:
            sys.modules['tools.'+name] = modules[name]
        else:
            sys.modules.pop('tools.'+name, None)
        if attrs[name] is not None:
            setattr(tools,name,attrs[name])
        elif hasattr(tools,name):
            delattr(tools,name)


def rank(a):
    return (rankdata(a, axis=0, method='average')-1)/max(1,len(a)-1)


@pytest.mark.parametrize('n',[1,2,3,97])
@pytest.mark.parametrize('ties',[False,True])
def test_q7_ranks_probability_families_not_folds(n,ties):
    rng=np.random.default_rng(91)
    arrays=[rng.random((m,n,12)).astype(np.float32) for m in (10,5,10)]
    if ties:
        arrays=[np.round(x*4)/4 for x in arrays]
    r,d,m=arrays
    actual,families=v.q_values(dict(l2=r,convnext=d,mil=m))
    expected=rank(sum(rank(x) for x in (rank(((r[:5]+r[5:])/2).mean(0)),rank(d.mean(0)),rank(m.mean(0))))/3)
    np.testing.assert_array_equal(actual,expected)
    assert set(families)=={'l2','convnext','mil'}


@pytest.mark.parametrize('n',[1,3,97])
@pytest.mark.parametrize('candidate',v.CANDIDATES)
def test_formulas_preserve_parent_arithmetic(n,candidate):
    _,v5=v.maths()
    rng=np.random.default_rng(42)
    a,b,q,r,e,s=[np.round(rng.random((n,12))*6).astype(np.float32)/6 for _ in range(6)]
    ids=[str(i) for i in range(n)]
    own=(a.astype(np.float64)+b.astype(np.float64))/2
    spec=v.candidate_specs()[candidate]
    family,weight=spec['family'],spec['partner_weight']
    if family=='B6':
        actual=v.blend(candidate,ids,a,b,q)
        expected=(1-weight)*rank(own)+weight*rank(q)
    else:
        actual=v.blend(candidate,ids,a,b,q,public=r,exp155=e,s6=s)
        parent='A5-172' if family=='A6' else 'A5q-172'
        if weight==.15:
            expected=v5.blend_arrays(parent,ids,r,s,own,e,q)
        else:
            expected=(1-weight)*rank(v5.trio_values(parent,ids,r,s,own,e))+weight*rank(q)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-.01,1.01])
def test_member_error_never_turns_into_prior(bad):
    a=np.full((10,3,12),.3,np.float32); a[2,1,4]=bad
    with pytest.raises(ValueError,match='probabilities'):
        v.q_values(dict(l2=a,convnext=np.zeros((5,3,12)),mil=np.zeros((10,3,12))))


@pytest.mark.parametrize('candidate',['B6','B6-w50'])
def test_b6_rejects_public_members(candidate):
    a=np.ones((3,12))*.4
    with pytest.raises(ValueError,match='B6 forbids'):
        v.blend(candidate,['a','b','c'],a,a,a,public=a)


def test_mil_requires_all_ten_folds():
    with pytest.raises(ValueError,match='probabilities'):
        v.q_values(dict(l2=np.zeros((10,3,12)),convnext=np.zeros((5,3,12)),mil=np.zeros((6,3,12))))


def test_infer_real_torch_forward_and_nan_rejection():
    import torch
    class Model(torch.nn.Module):
        def forward(self,x):
            return x.mean((1,2,3,4))[:,None].repeat(1,12)
    x=np.ones((3,3,8,8),np.float32)
    np.testing.assert_array_equal(v.infer(Model(),x,'cpu'),torch.sigmoid(torch.ones(12)).numpy())
    np.testing.assert_array_equal(v.infer(Model(),x,'cpu',anchor=True),v.infer(Model(),x,'cpu'))
    x[:]=np.nan
    with pytest.raises(ValueError,match='logits'):
        v.infer(Model(),x,'cpu')


def test_anchor_decode_and_members_once(monkeypatch):
    calls=[]
    ns={'gpu_windows':lambda *a,**kw:np.zeros((94,3,2,2),np.float32)}
    models={k:k for k in ('s42','s43','exp155','public')}
    monkeypatch.setattr(v,'anchor_models',lambda *a:(ns,models))
    def reader(uid,*_):
        calls.append(('decode',uid)); return np.ones((96,2,2),np.uint8),np.ones(96,bool)
    monkeypatch.setattr(v,'anchor_reader',lambda _:reader)
    def infer(model,*a,**kw):
        calls.append(('infer',model));return np.zeros(12,np.float32)
    monkeypatch.setattr(v,'infer',infer)
    result=v.anchor_worker(dict(paths={},candidate='A6',rows=[{v.UID:'a'},{v.UID:'b'}],ids=['a','b'],images='x'),'cpu')
    assert len([x for x in calls if x[0]=='decode'])==2
    assert len([x for x in calls if x[0]=='infer'])==2*2*4 # required plain/flip TTA only
    assert set(result)==set(models)


def test_worker_error_does_not_publish_npz(tmp_path,monkeypatch):
    job=tmp_path/'job.json';v.write_new(job,{'phase':'anchor','ids':['a']})
    monkeypatch.setattr(v,'backend',lambda:None)
    def fail(*_):raise RuntimeError('injected member failure')
    monkeypatch.setattr(v,'anchor_worker',fail)
    out=tmp_path/'out.npz'
    with pytest.raises(RuntimeError):v.worker(job,out,'cpu')
    assert not out.exists() and not Path(str(out)+'.json').exists()


@pytest.mark.parametrize('phase',['anchor','partner'])
def test_two_gpu_shards_preserve_global_order(tmp_path,monkeypatch,phase):
    seen=[]
    class Process:
        def __init__(self,args,env):
            spec=v.read(args[args.index('--job')+1]);out=Path(args[args.index('--out')+1])
            seen.append((env['CUDA_VISIBLE_DEVICES'],spec['ids']))
            a=np.asarray([float(x) for x in spec['ids']])[:,None].repeat(12,axis=1)
            if phase=='partner':a=a[None].repeat(10,axis=0)
            np.savez(out,ids=spec['ids'],member=a)
            v.write_new(str(out)+'.json',{'sha256':v.sha(out)})
            self.returncode=0
        def poll(self):return 0
    monkeypatch.setattr(v.subprocess,'Popen',Process)
    job=dict(ids=['4','1','7','2','9'])
    out=v.run_shards(job,tmp_path,phase)['member']
    actual=out[:,0] if phase=='anchor' else out[0,:,0]
    np.testing.assert_array_equal(actual,[4,1,7,2,9])
    assert seen==[('0',['4','7','9']),('1',['1','2'])]


def test_failed_shard_terminates_sibling(tmp_path,monkeypatch):
    instances=[]
    class Process:
        def __init__(self,*a,**kw):
            self.returncode=4 if not instances else None;self.terminated=False;instances.append(self)
        def poll(self):return self.returncode
        def terminate(self):self.terminated=True;self.returncode=-15
        def wait(self):return self.returncode
    monkeypatch.setattr(v.subprocess,'Popen',Process)
    with pytest.raises(ValueError,match='member worker failed'):
        v.run_shards({'ids':['a','b']},tmp_path,'anchor')
    assert instances[1].terminated


@pytest.mark.parametrize('name', ['../escape','/absolute','bad\\path'])
def test_unsafe_zip_rejected(tmp_path,name):
    b=io.BytesIO()
    entry=zipfile.ZipInfo('placeholder')
    entry.filename=name  # Avoid Windows ZipInfo constructor normalising backslashes.
    with zipfile.ZipFile(b,'w') as z:z.writestr(entry,b'unsafe')
    with pytest.raises(ValueError,match='unsafe ZIP'):v.safe_extract(b.getvalue(),tmp_path/'out')


def registration(tmp_path):
    src=tmp_path/'source';src.write_bytes(b'pinned')
    pin=v.file_pin(src)
    return dict(schema='exp169-v6-registration/v1',version=v.VERSION,sources={'source':pin},assets={},
                mil=[dict(seed=s,fold=f,name=f'm{s}{f}',sha256=None,bytes=None) for s in (42,43) for f in range(5)],
                f0_member_binding=None,member_screen=None,candidates=v.candidate_specs())


def test_pending_registration_and_serve_fail_before_cuda(tmp_path,monkeypatch):
    m=registration(tmp_path);path=tmp_path/'reg.json';v.write_new(path,m)
    assert v.validate(path)['status']=='BLOCKED'
    import torch
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('must block before CUDA'))
    with pytest.raises(ValueError,match='unfilled weight pins'):
        v.serve(path,'A6',tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_source_hash_drift_rejected(tmp_path):
    m=registration(tmp_path)
    Path(m['sources']['source']['path']).write_bytes(b'tamper')
    with pytest.raises(ValueError,match='registered bytes changed'):v.validate(m)


def test_create_only_receipts_and_stages(tmp_path):
    p=tmp_path/'rc.json';v.write_new(p,{'original':True})
    with pytest.raises(FileExistsError):v.write_new(p,{'original':False})
    assert v.read(p)=={'original':True}
    with pytest.raises(ValueError,match='stage must be new'):v.prepare('unused',tmp_path)


def test_every_generated_release_notebook_declares_python_kernel(tmp_path, monkeypatch):
    m = registration(tmp_path)
    m['new_dataset'] = 'owner/mil'
    m['assets'] = {key: dict(path='unused', dataset='owner/'+key)
                   for key in ('partner', 's42', 's43', 'c4', 'exp155')}
    for entry in m['mil']:
        entry['path'] = 'unused'
    path = tmp_path/'registration.json'
    v.write_new(path, m)
    monkeypatch.setattr(v, 'validate', lambda _: dict(blockers=['unpinned test weights'], pinned_members=0))
    stage = tmp_path/'stage'
    v.prepare(path, stage)
    reference = v.read(v.ROOT/'artifacts/experiments/EXP169_RELEASE_V5Q/v1.2.0/kernels/a5_172/release.ipynb')['metadata']
    notebooks = list(stage.glob('*/release.ipynb'))
    assert len(notebooks) == len(v.CANDIDATES) == 6
    for notebook in notebooks:
        metadata = v.read(notebook)['metadata']
        assert metadata['kernelspec']['name'] == 'python3'
        assert metadata['kernelspec'] == reference['kernelspec']
        assert metadata.get('language_info') == reference.get('language_info')
        kernel = v.read(notebook.parent/'kernel-metadata.json')
        assert kernel['title'] == kernel['id'].split('/')[-1]
        assert kernel['id'] == 'taeyangg4/knee-exp169-'+notebook.parent.name.lower()+'-v1'


def test_serving_has_no_clock_gates_or_frozen_barrier_calls():
    tree=ast.parse(inspect.getsource(v))
    serving={'serve','worker','anchor_worker','partner_worker','run_shards','infer','anchor_models'}
    for fn in tree.body:
        if not isinstance(fn,ast.FunctionDef) or fn.name not in serving:continue
        for call in [n for n in ast.walk(fn) if isinstance(n,ast.Call)]:
            assert not any(kw.arg=='timeout' for kw in call.keywords)
            assert not (isinstance(call.func,ast.Attribute) and call.func.attr=='partner_scores')
        for compare in [n for n in ast.walk(fn) if isinstance(n,ast.Compare)]:
            assert not any(isinstance(n,ast.Attribute) and n.attr in {'monotonic','time'} for n in ast.walk(compare))


def test_mil_metadata_rejects_wrong_fold_and_spatial_kind():
    ck=dict(arch='convnext_tiny.fb_in22k_ft_in1k',res=224,lab=list(v.LABELS),contract='raptor64_336_c140_s02',
            norm='imagenet',window_policy='within-slot-valid',fold=0,epoch=19,model_kind='finding_spatial_tokens_v1',
            input_policy='center3',k_eval=24,flip_tta='coronal_axial_horizontal_mean_probability',seed=42)
    v.mil_meta(ck,dict(seed=42,fold=0))
    for key,value in [('fold',1),('model_kind','global'),('epoch',18),('seed',43)]:
        bad={**ck,key:value}
        with pytest.raises(ValueError):v.mil_meta(bad,dict(seed=42,fold=0))


@pytest.mark.parametrize('candidate,weight',[('A6',.15),('A6q',.15),('B6',.30)])
def test_explicit_parent_weight_is_bit_exact(candidate,weight):
    rng=np.random.default_rng(903)
    a,b,q,r,e,s=[rng.random((97,12)).astype(np.float32) for _ in range(6)]
    ids=[str(i) for i in range(97)]
    extra={} if candidate=='B6' else dict(public=r,exp155=e,s6=s)
    actual=v.blend(candidate,ids,a,b,q,partner_weight=weight,**extra)
    v2,v5=v.maths()
    own=(a.astype(np.float64)+b.astype(np.float64))/2
    expected=(v2.final_values(own,q,'B172') if candidate=='B6' else
              v5.blend_arrays('A5-172' if candidate=='A6' else 'A5q-172',ids,r,s,own,e,q))
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('weight',[-.1,1.1,float('nan'),float('inf'),True,'0.3'])
def test_invalid_candidate_weight_rejected(weight):
    with pytest.raises(ValueError,match='partner weight'):v.candidate_specs({'A6':weight})


def test_weighted_candidates_share_member_inputs_and_exclude_b_public(monkeypatch):
    specs=v.candidate_specs({'A6-w30':.60})
    a=np.ones((3,12))*.4
    seen=[]
    def fake(name,ids,s42,s43,q,**kwargs):
        assert s42 is a and s43 is a and q is a
        assert kwargs['partner_weight']==specs[name]['partner_weight']
        assert ('public' in kwargs)==(specs[name]['family']!='B6')
        seen.append(name)
        return a
    monkeypatch.setattr(v,'blend',fake)
    assert len(v.candidate_values(specs,'A6-w30',['a','b','c'],a,a,a,public=a,exp155=a,s6=a))==6
    assert set(seen)==set(v.CANDIDATES)
    seen.clear()
    assert set(v.candidate_values(specs,'B6-w50',['a','b','c'],a,a,a))=={'B6','B6-w50'}


def test_candidate_registration_and_upload_drift_rejected(tmp_path):
    m=registration(tmp_path)
    m['candidates']['A6']['family']='B6'
    with pytest.raises(ValueError,match='candidate registration'):v.validate(m)
    weight=tmp_path/'mil.pt';weight.write_bytes(b'checkpoint')
    m.update(new_dataset='owner/mil',f0_member_binding={'sha256':'a'*64},mil=[dict(
        name='mil.pt',seed=42,fold=0,spec_sha256=None,sha256=v.sha(weight),bytes=weight.stat().st_size)])
    v.write_new(tmp_path/'mil-manifest.json',v.mil_manifest(m))
    assert v.check_mil_upload(m,tmp_path)==tmp_path
    weight.write_bytes(b'bad checkpoint')
    with pytest.raises(ValueError,match='MIL upload weight'):v.check_mil_upload(m,tmp_path)


def test_amendment_two_menu_and_equal_b_weight():
    assert v.DEFAULT_WEIGHTS == {'A6':.15,'A6q':.15,'A6-w30':.30,'A6q-w30':.30,'B6':.30,'B6-w50':.50}
    rng=np.random.default_rng(50)
    a,b,q=[rng.random((97,12)).astype(np.float32) for _ in range(3)]
    own=(a.astype(np.float64)+b.astype(np.float64))/2
    actual=v.blend('B6-w50',[str(i) for i in range(97)],a,b,q)
    np.testing.assert_array_equal(actual,.5*rank(own)+.5*rank(q))
    with pytest.raises(ValueError,match='unknown candidate'):
        v.candidate_specs({'B6-w45':.45})


def test_q7_rejects_dino_and_plain_convnext():
    z = np.zeros((10, 3, 12), np.float32)
    with pytest.raises(ValueError, match='Q7 family set'):
        v.q_values(dict(l2=z, dino=z[:5], mil=z))
    manifest = v.read(v.ROOT / '.local/kout-a4-v2/exp169-partner/release-manifest.json')
    entries = v.q7_entries(manifest)
    assert len(entries) == 15 and all(e['family'] != 'dino' for e in entries)
    plain = copy.deepcopy(manifest)
    next(e for e in plain['entries'] if e['family'] == 'convnext')['crop_policy'] = 'full'
    with pytest.raises(ValueError, match='mixed-scale'):
        v.q7_entries(plain)


def test_imported_a4_windows_equal_training_policy():
    import sys
    sys.path.insert(0, str(v.ROOT / v.B172 / 'assets/code'))
    sys.path.insert(0, str(v.ROOT / 'training/src'))
    from raptor_exp179 import scale_windows
    from raptor_radimagenet import normalise
    loader, windows = v.a4_policy()
    assert loader.__globals__['model_meta'].__module__ == 'tools.exp169_v4_frozen'
    volume = np.random.default_rng(12).integers(0, 256, (64,336,336), dtype=np.uint8)
    centers = list(range(35)) + [-1]
    actual = windows(volume, centers)
    np.testing.assert_array_equal(actual['convnext'], normalise(scale_windows(volume, centers, 224, 'mixed'), 'imagenet'))
    assert actual['convnext'].dtype == np.float32
    assert v.source_files(v.ROOT)['tools/exp169_v4_frozen.py'] == v.ROOT / v.V4 / 'code/assets/exp169_release_v4.py'


def test_partner_worker_loads_only_q7_and_preserves_tta(monkeypatch):
    import sys
    sys.path.insert(0, str(v.ROOT / v.B172 / 'assets/code'))
    import raptor_dicom, raptor_common
    manifest = v.read(v.ROOT / '.local/kout-a4-v2/exp169-partner/release-manifest.json')
    monkeypatch.setattr(v, 'read', lambda _: manifest)
    class Corpus:
        spec = {}
        def volume(self, uid): return np.ones((64,2,2), np.uint8)
        def mask(self, uid): return np.ones(64, np.uint8)
    monkeypatch.setattr(raptor_dicom.DicomCorpus, 'from_records', lambda *a: Corpus())
    monkeypatch.setattr(raptor_dicom, 'DirectoryDicomSource', lambda path: path)
    monkeypatch.setattr(raptor_common, 'select_centers', lambda *a: list(range(36)))
    v2, _ = v.maths()
    monkeypatch.setattr(v2.base, 'plane_flags', lambda *a: [True]*18 + [False]*18)
    loaded, inferred = [], []
    def loader(work, entries, device):
        loaded.extend(entries)
        return entries
    def windows(*a): return {k:np.ones((36,3,2,2),np.float32) for k in ('l2','convnext')}
    monkeypatch.setattr(v, 'a4_policy', lambda: (loader, windows))
    monkeypatch.setattr(v, 'load_mil', lambda path, entry, device: entry)
    monkeypatch.setattr(v, 'mil_windows', lambda *a: np.ones((36,3,2,2),np.float32))
    monkeypatch.setattr(v, 'mil_prediction', lambda *a: np.ones(12,np.float32)*.4)
    monkeypatch.setattr(v, 'backend', lambda **kw: None)
    def infer(model,*a):
        inferred.append(model['family'])
        return np.ones(12,np.float32)*.2
    monkeypatch.setattr(v, 'infer', infer)
    job = dict(paths=dict(partner='unused',mil='unused'), ids=['x','y'],rows=[],images='unused',
               mil=[dict(name=str(i)) for i in range(10)])
    scores = v.partner_worker(job, 'cpu')
    assert len(loaded) == 15 and len(inferred) == 2*2*15
    assert 'dino' not in inferred and set(scores) == {'l2','convnext','mil'}
    assert scores['l2'].shape == scores['mil'].shape == (10,2,12)
    assert scores['convnext'].shape == (5,2,12)
