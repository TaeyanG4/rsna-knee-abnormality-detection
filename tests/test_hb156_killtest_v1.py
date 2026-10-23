"""Offline H-B-156 scope, pin and loader rejection tests; no release authority."""
import copy
import json
import socket
import types
from unittest.mock import patch
import pytest
from tools import hb156_killtest_v1 as b
from tools import hb151_killtest_v102_fixtures as fast


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    b.cpu_only()
    import torch
    def fail(*args,**kwargs): pytest.fail('network/CUDA forbidden')
    monkeypatch.setattr(socket,'create_connection',fail)
    monkeypatch.setattr(torch.cuda,'_lazy_init',fail)


@pytest.fixture(scope='module')
def parent():
    return b.authenticate()[1]


@pytest.mark.parametrize('name,kwargs',fast.case_specs(),ids=[n for n,_ in fast.case_specs()])
def test_fast_path(tmp_path,name,kwargs):
    shim=types.SimpleNamespace(ROOT=b.ROOT,OUT=b.OUT,SCRATCH=b.SCRATCH,read=b.read)
    with patch.object(fast,'b',shim): assert fast.run_case(tmp_path,**kwargs)['status']=='PASS'


def test_full_authentication_scope(parent):
    assert b.validate()['placeholder'] is True
    nb=b.read(b.OUT/'kernel/release.ipynb')
    before=b.read(b.PARENT/'kernel/release.ipynb')
    assert nb['cells'][:5]==before['cells'][:5]
    assert b.DECISION==dict(alive_at_least=.950,dead_at_most=.946,otherwise_seed=43,
                          baseline_public=.946,baseline_ref='57005140',standalone=True)
    code=b.code_bytes(parent)
    assert b.embedded(parent['public-serving.py'])==b.embedded(code['public-serving.py'])
    assert b.embedded(code['public-serving.py'])


@pytest.mark.parametrize('mutation',['fast','guard','markdown','extra-cell','metadata','runtime','constructor','manifest','checkpoint','base64'])
def test_forbidden_diff(parent,mutation):
    nb,meta=b.read(b.OUT/'kernel/release.ipynb'),b.read(b.OUT/'kernel/kernel-metadata.json')
    hashes,small=b.archive_files(b.OUT/'dataset'/b.ARCHIVE)
    binding=b.read(b.OUT/'export-binding.json')
    if mutation=='fast': nb['cells'][0]['source']=['FAST=True\n']
    if mutation=='guard': nb['cells'][5]['source']=nb['cells'][5]['source'].replace('if not FAST:','if True:')
    if mutation=='markdown': nb['cells'][1]['source']='changed'
    if mutation=='extra-cell': nb['cells'].append(copy.deepcopy(nb['cells'][0]))
    if mutation=='metadata': meta['enable_internet']=True
    if mutation=='runtime': small[b.RUNTIME]+=b'\n# launch drift\n'
    if mutation=='constructor': small['public-serving.py']=small['public-serving.py'].replace(b'global_pool="avg"',b'global_pool="max"')
    if mutation=='manifest':
        m=json.loads(small['release-manifest.json']);m['policy']['chunk_windows']=32
        small['release-manifest.json']=b.encoded(m)
    if mutation=='checkpoint': hashes[b.NAME]='0'*64
    if mutation=='base64': small['public-serving.py']=small['public-serving.py'].replace(b'_FASTREAD_B64',b'_CHANGED_B64')
    with pytest.raises(ValueError): b.scope(nb,meta,hashes,small,parent,binding,b.sha(b.OUT/'dataset'/b.ARCHIVE))


@pytest.mark.parametrize('kind',['missing','unpinned','wrong-sha','invalid-pin','no-json','wrong-protocol','placeholder-json','placeholder-checkpoint'])
def test_bind_rejects_before_stage_mutation(parent,tmp_path,monkeypatch,kind):
    monkeypatch.setattr(b,'authenticate',lambda:(None,parent))
    target=tmp_path/b.NAME
    if kind=='missing':
        with pytest.raises(ValueError,match='missing'): b.bind(target)
        return
    # Hardlink same-shape CPU fixture; never copy or modify the running fit.
    target.hardlink_to(b.SCRATCH/'placeholder'/b.NAME)
    pin=target.with_suffix('.sha256')
    meta=b.identity(target,False)
    if kind!='unpinned': b.write(pin,(meta['sha256']+'  '+b.NAME+'\n').encode())
    if kind=='wrong-sha': b.write(pin,('0'*64+'  '+b.NAME+'\n').encode())
    if kind=='invalid-pin': b.write(pin,b'placeholder\n')
    if kind=='wrong-protocol': meta['protocol_sha256']='0'*64
    if kind=='placeholder-json': meta['placeholder']=True
    if kind!='no-json': b.write(target.with_suffix('.json'),b.encoded(meta))
    before=b.sha(b.OUT/'build-receipt.json')
    with pytest.raises(ValueError): b.bind(target)
    assert b.sha(b.OUT/'build-receipt.json')==before


def test_bind_dispatch_pinned_triplet(parent,tmp_path,monkeypatch):
    """Positive sidecar contract; build/strict-load covered independently, never publish test weights."""
    monkeypatch.setattr(b,'authenticate',lambda:(None,parent))
    target=tmp_path/b.NAME
    target.hardlink_to(b.SCRATCH/'placeholder'/b.NAME)
    meta=b.identity(target,False)
    b.write(target.with_suffix('.sha256'),(meta['sha256']+'  '+b.NAME+'\n').encode())
    b.write(target.with_suffix('.json'),b.encoded(meta))
    captured={}
    def capture(path,binding,original):
        captured.update(binding)
        assert path==target and original==parent
        return {'status':'TEST_ONLY_NO_BUILD'}
    monkeypatch.setattr(b,'build',capture)
    assert b.bind(target)['status']=='TEST_ONLY_NO_BUILD'
    assert captured['placeholder'] is False and captured['sha256']==b.sha(target)


@pytest.mark.parametrize('field',['arch','res','lab','k_eval','swa_epochs_1based','seed','label_set','exposure','provenance','model'])
def test_checkpoint_contract_rejection(parent,field):
    rt=b.runtime(b.code_bytes(parent)[b.RUNTIME])
    path=b.SCRATCH/'placeholder'/b.NAME
    ck=rt.safe_load(path)
    ck[field]={} if field in ('provenance','model') else None
    with patch.object(rt,'safe_load',lambda _:ck):
        with pytest.raises(ValueError): b.compatibility(path,b.identity(path,True),rt)


@pytest.mark.parametrize('label_set',['L2','WRONG_LABEL_SET','__MISSING__'])
def test_exact_registered_label_set_rejected(parent,label_set):
    rt=b.runtime(b.code_bytes(parent)[b.RUNTIME])
    path=b.SCRATCH/'placeholder'/b.NAME
    ck=rt.safe_load(path)
    if label_set=='__MISSING__': ck.pop('label_set',None)
    else: ck['label_set']=label_set
    with patch.object(rt,'safe_load',lambda _:ck):
        with pytest.raises(ValueError,match='registered L2 label-set'):
            b.compatibility(path,b.identity(path,True),rt)


def test_exact_cpu_strict_loader():
    from tools.hb156_killtest_v1_smoke import extract,loader
    model,report=loader(extract())
    assert report['status']=='PASS' and report['strict_load']
    assert report['wrong']['sha']=='REJECT' and model.backbone.num_features==768


def test_partial_parent_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'PARENT',tmp_path)
    with pytest.raises((ValueError,FileNotFoundError)): b.authenticate()


def test_bound_stage_not_overwritten(parent,tmp_path,monkeypatch):
    monkeypatch.setattr(b,'OUT',tmp_path)
    b.write(tmp_path/'RELEASED.json',b'{}')
    with pytest.raises(ValueError,match='frozen'): b.build(None,None,parent)


def test_patch_anchor_fail_closed():
    with pytest.raises(ValueError): b.replace_once(b'aa',b'a',b'b')
