"""Codec-base rebases: real replay, isolated packages, callback identity and math."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from tools import c85r7_codec_rebase_v2 as m
from tools import c85r7_codec_rebase_verify_v2 as v
from tools import c85r7_lb_r2_adapter_v1 as a
from tools import exp169_r7_selective_flip_v130 as flip


@pytest.mark.parametrize('tag', list(m.SPECS))
def test_exact_parent_own_source_and_no_wrapper(tag):
    source = m.runtime_source(tag)[1]
    assert m.own_proof(tag, source)['status'] == 'PASS_EXACT_OWN_PATH'
    assert m.own_path(source) == m.own_path(m.frozen_files()['exp169_release_c85r7codec.py'].decode())
    with pytest.raises(ValueError, match='own child differs'):
        m.own_proof(tag, source.replace('write_once(script, a172_script(', 'write_once(script, broken_script('))


@pytest.mark.parametrize('tag', list(m.SPECS))
def test_exact_generated_own_program_and_child_argv_env(tag, tmp_path, monkeypatch):
    import types
    parent = types.ModuleType('frozen_codec_parent')
    parent.__file__ = str(m.BASE/'frozen-parent.py')
    exec(compile(m.frozen_files()['exp169_release_c85r7codec.py'], parent.__file__, 'exec'), parent.__dict__)
    candidate = v.module(tag)
    root = tmp_path/'runtime'
    root.mkdir()
    files = m.frozen_files()
    for name in ('a172-cell.py.txt','q-a5q-manifest.json'):
        (root/name).write_bytes(files[name])
    out = tmp_path/'output'
    captures = []
    # The production phase wrapper records Exception as a failed child and
    # continues. This test must stop before any later production phase runs.
    class OwnBoundaryReached(BaseException): pass
    for runtime in (parent, candidate):
        # Capture at the own child boundary; no executable/model/GPU is started.
        monkeypatch.setattr(runtime, 'runtime_pins', lambda *args: None)
        monkeypatch.setattr(runtime, 'preflight', lambda *args: ['a','b','c'])
        monkeypatch.setattr(runtime, 'install_codecs', lambda *args, **kw: {'status':'INACTIVE_NO_WHEEL'})
        if tag == 'r2' and runtime is candidate:
            monkeypatch.setattr(runtime.r2_adapter,'preflight',lambda *args: None)
        def boundary(args, *, label, env):
            captures.append((args,label,env,(out/runtime.RUN/'a172.py').read_bytes()))
            raise OwnBoundaryReached()
        monkeypatch.setattr(runtime.v5q,'run_process_group',boundary)
        with pytest.raises(OwnBoundaryReached): runtime.run_release(root,out)
        # Remove only this pytest-owned temporary output between identical-path runs.
        import shutil
        assert out.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(out)
    assert captures[0] == captures[1]
    assert captures[0][0] == [sys.executable,'-u',str(out/parent.RUN/'a172.py')]


def test_new_slugs_and_frozen_stages_preserved():
    expected = {'flipall':'flipall115','own25':'own25c','fluidR':'fluidRc','r2':'r2c'}
    for tag, suffix in expected.items():
        metadata = m.h.read(m.stage(tag)/'kernel/kernel-metadata.json')
        assert metadata['id'] == 'taeyangg4/knee-lb-'+suffix+'-v1'
        assert metadata['title'] == metadata['id'].split('/')[1]
        assert (m.stage(tag)/'own-path.diff').read_bytes() == b''
    for name,sha in m.h.read(m.BACKUP/'frozen-stage-sha256.json').items():
        assert m.h.sha(m.ROOT/name) == sha


@pytest.mark.parametrize('tag',list(m.SPECS))
def test_real_visible_replay(tag):
    result=v.replay(tag)
    assert result['studies']==3 and result['parent_noop_csv_bytes']


@pytest.mark.parametrize('tag',list(m.SPECS))
def test_synthetic_formula_with_ties(tag):
    reg=a.registration(); rng=np.random.default_rng(140)
    raw={k:rng.integers(1,10,(13,12)).astype(np.float32)/11 for k in ['R','S_R7','R2-s43','R2-s44']}
    own=tuple(rng.integers(1,10,(13,12)).astype(np.float64)/11 for _ in range(3))
    runtime=v.module(tag)
    r=raw['R'] if tag!='r2' else np.mean([raw['R2-s43'],raw['R2-s44']],axis=0,dtype=np.float32)
    ids=[str(i) for i in range(13)]
    got=runtime.blend(ids,r,raw['S_R7'],*own)
    assert m.b.p.csv_bytes(ids,got)==m.b.p.csv_bytes(ids,v.reference(tag,reg,raw,own))


def test_fluid_isolation_and_failed_r():
    rng=np.random.default_rng(42); ids=['a','b','c','d','e']
    inputs=[rng.random((5,12)) for _ in range(5)]
    runtime=v.module('fluidR')
    got=runtime.blend(ids,*inputs); old=m.b.blend(ids,*inputs)
    keep=[j for j in range(12) if j not in (7,9)]
    np.testing.assert_array_equal(got[:,keep],old[:,keep])
    got=runtime.blend(ids,*inputs,failures={'R':dict.fromkeys(ids,'failed')})
    own=(17*v.rank(inputs[2])+17*v.rank(inputs[3])+12*v.rank(inputs[4]))/46
    np.testing.assert_array_equal(got[:,[7,9]],v.rank(own)[:,[7,9]])


def test_r2_partial_and_tampered_identity(tmp_path):
    reg=a.registration();ids=['a','b','c']
    journal=m.h.c.Journal(tmp_path/'NATIVE/worker0',ids,a.identity(reg))
    journal.record('a','R2-s43',np.full(12,.2,np.float32))
    journal.record('a','R2-s44',np.full(12,.4,np.float32))
    journal.record('b','R2-s44',np.full(12,.6,np.float32))
    values,failed,_=a.recover(tmp_path,ids,reg)
    np.testing.assert_array_equal(values[:,0],np.array([.3,.6,.5],np.float32))
    assert set(failed)=={'c'}
    identity=a.identity(reg);identity['members']['R2-s43']='0'*64
    m.h.dump(tmp_path/'NATIVE/worker0/journal-identity.json',identity)
    with pytest.raises(ValueError,match='identity'): a.recover(tmp_path,ids,reg)


@pytest.mark.parametrize('tag',list(m.SPECS))
def test_notebook_bootstrap_isolated_import_and_parent_bytes(tag,tmp_path):
    files,nb,meta,_=m.payload(tag)
    parent=m.h.read(m.BASE/'kernel/kernel-metadata.json')
    assert meta['is_private'] is True and meta['enable_internet'] is False
    assert {k:val for k,val in meta.items() if k not in ('id','title','dataset_sources')}=={k:val for k,val in parent.items() if k not in ('id','title','dataset_sources')}
    assert meta['dataset_sources']==parent['dataset_sources']+([a.DATASET] if tag=='r2' else [])
    assert all(files[k]==val for k,val in m.frozen_files().items())
    assert not any('deadline' in k or 'insurance' in k or 'metadata_v122' in k for k in files)
    module,source=m.runtime_source(tag)
    assert 'import c85_own_serving' not in source
    code=''.join(nb['cells'][-1]['source'])
    preamble=code.split(f'import {module} as release')[0]
    target=tmp_path/'payload'
    preamble=preamble.replace('/kaggle/working/lb-'+m.SPECS[tag]['slug']+'-runtime',target.as_posix())
    exec(compile(preamble,'<notebook>','exec'),{})
    assert {name:(target/name).read_bytes() for name in files}==files
    for name,data in files.items():
        if name.endswith('.py'): compile(data,name,'exec')
    code=f'import sys;sys.path.insert(0,sys.argv[1]);import {module} as r;print(r.VERSION);'
    if tag=='r2': code+="print(r.r2_adapter.registration(sys.argv[1])['P']['Rplus2']['members'][0]['seed'])"
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(target)],cwd=target,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1'),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr



def test_flip_codec_integration_restore_arguments_and_provenance(tmp_path):
    r7=m.b.r7
    originals=r7.diagnostic_source,r7.worker_args,r7.write_provenance
    with flip.installed(r7,'FLIP_ALL',m.metadata_io):
        rt=r7.diagnostic_source(r7.SOURCE,tmp_path)
        assert '_c85r7_flip' in (rt/'compact_infer.py').read_text()
        assert all(m.digest((rt/k).read_bytes())==val for k,val in m.h.read(rt/'runtime_sha256.json').items())
        args=r7.worker_args(dict(backbone='convnext_small',size=384,crop=352,flip_tta=False),tmp_path,tmp_path/'anatomy.pt')
        assert args.count('--flip-tta')==1
        rec=r7.write_provenance(tmp_path/'provenance.json',identity=dict(manifest_sha256='test',manifest_pinned=True,file_sha256={},declared={},features={}),code=dict(tree_sha256='test',files={},closure_sha256={},r7_closure_sha256={}),args=args)
        assert rec['flip_policy']==flip.policy('FLIP_ALL')
    assert originals==(r7.diagnostic_source,r7.worker_args,r7.write_provenance)


def test_runtime_projection_is_honest():
    assert m.SPECS['flipall']['hours']+1/3>9
    assert 'exceeds 9 h' in m.launch('flipall')
    assert all(m.SPECS[t]['hours']+1/3<9 for t in ('own25','fluidR','r2'))


@pytest.mark.parametrize('tag',list(m.SPECS))
def test_sealed_release_process_wiring(tag,tmp_path,monkeypatch):
    runtime=v.module(tag); files,_,_,_=m.payload(tag)
    root=tmp_path/'runtime'; root.mkdir()
    for name,data in files.items():
        target=root/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
    out=tmp_path/'output'; seen=[]; admissions=[]
    monkeypatch.setattr(runtime,'preflight',lambda *args: ['a','b','c'])
    monkeypatch.setattr(runtime,'install_codecs',lambda *args,**kw: {'status':'INACTIVE_NO_WHEEL'})
    monkeypatch.setattr(runtime,'collect',lambda *args: ({'codecs':{}},b''))
    if tag=='r2': monkeypatch.setattr(runtime.r2_adapter,'preflight',lambda rt: admissions.append(rt))
    def execute(args,*,label,env):
        seen.append(args)
        assert env['PYTHONPATH']==str(root)
    monkeypatch.setattr(runtime.v5q,'run_process_group',execute)
    runtime.run_release(root,out)
    assert len(seen)==2
    work=out/runtime.RUN
    assert seen[0][2]==str(work/'a172.py')
    assert not (work/'c85_own_serving.py').exists()
    program=(work/'a172.py').read_text(encoding='utf-8')
    assert program==runtime.a172_script(files['a172-cell.py.txt'].decode(),work/'a172',{'status':'INACTIVE_NO_WHEEL'})
    assert "__name__ = '__main__'" not in program and '__name__ = "__main__"' not in program
    assert seen[1][2:4]==[str(root/(m.runtime_source(tag)[0]+'.py')),'public-worker']
    assert admissions==([root] if tag=='r2' else [])


@pytest.mark.parametrize('tag',list(m.SPECS))
def test_public_process_wiring(tag,tmp_path,monkeypatch):
    from contextlib import contextmanager
    runtime=v.module(tag); out=tmp_path/'run/pr7'; work=out/'pr7-run';work.mkdir(parents=True)
    m.dump(out/'ids.json',['a','b','c'])
    m.dump(work/'r7-identity.json',{'manifest_sha256':'f'*64})
    m.dump(work/'code-paths.json',{'code':str(work/'code'),'anatomy':str(work/'anatomy.pt')})
    root=tmp_path/'runtime'; seen=[]
    monkeypatch.setattr(runtime,'runtime_pins',lambda *args: None)
    monkeypatch.setattr(runtime,'competition_root',lambda: tmp_path/'competition')
    monkeypatch.setattr(runtime.v5q,'_mount',lambda *args: tmp_path/'mount')
    monkeypatch.setattr(runtime.p,'check_pin',lambda *args: None)
    monkeypatch.setattr(runtime.s6,'install_wheels',lambda *args: None)
    monkeypatch.setattr(runtime,'gate_s_r7',lambda *args: None)
    monkeypatch.setattr(runtime.v5q,'run_process_group',lambda *args,**kw: seen.append('R'))
    monkeypatch.setattr(runtime.r7,'infer_studies',lambda *args,**kw: seen.append('R7'))
    if tag=='r2':
        def native(rt,comp,dest):
            assert rt==root and dest==work/'r2'
            assert os.environ['PYTHONPATH']==str(root)
            seen.append('R2')
        monkeypatch.setattr(runtime.r2_adapter,'run',native)
    if tag=='flipall':
        @contextmanager
        def installed(r7,variant,writer):
            assert variant=='FLIP_ALL' and r7 is runtime.r7
            seen.append('FLIP_ALL_ENTER');yield;seen.append('FLIP_ALL_EXIT')
        monkeypatch.setattr(runtime.selective_flip,'installed',installed)
    previous=dict(os.environ)
    runtime.run_public(root,out)
    assert dict(os.environ)==previous
    assert seen==(['R','FLIP_ALL_ENTER','R7','FLIP_ALL_EXIT'] if tag=='flipall' else ['R2' if tag=='r2' else 'R','R7'])
