"""H-B-228 CPU-only regressions; no network, CUDA or real label output."""
import ast
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from tools import hb228_stage_v1 as b
from tools import hb228_compare_v1 as c
from tools import hb228_pair_v1 as pair


def csv_file(path, ids, x):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow([c.UID,*c.TARGETS])
        w.writerows([[u,*r] for u,r in zip(ids,x)])


@pytest.mark.parametrize('kind',['candidate','n32','pair'])
def test_exact_stage_contract(kind):
    nb,meta,old,new,*_=b.package(kind)
    port = new['exp169_r7_port.py'].decode()
    assert '"--flip-tta"' in port
    assert b.digest(new['exp169_r7_port.py']) in new['exp169_release_c85r7.py'].decode()
    assert b.digest(new['exp169_release_c85r7.py']) in new['exp169_release_c85r7codec.py'].decode()
    assert new['tools/k1sr_runtime_v7.py'] == old['tools/k1sr_runtime_v7.py']
    assert "'k1_eval_stride': 2" in new['tools/k1sr_runtime_v7.py'].decode()
    assert meta['is_private'] and not meta['enable_internet']
    assert meta['title'] == meta['id'].split('/')[1]
    if kind == 'pair':
        assert '_k1_worker = "compact_infer.py"' in port
        assert 'install(runtime_source)' not in port
        tree=ast.parse(''.join(nb['cells'][2]['source']))
        assert len(tree.body) == 1 and isinstance(tree.body[0],ast.If)
        assert 'run_release(' not in ''.join(nb['cells'][2]['source'])
    else:
        assert new['tools/k1_shared_render_v2.py'] == b.adapter(old['tools/k1_shared_render_v2.py'])
        assert '"--eval-stride", "2"' in port
        assert 'install(runtime_source)' in port
        assert 'r6_eval_stride' in ''.join(nb['cells'][2]['source'])


def worker_args(source,stride=1):
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker_args')
    ns=dict(Path=Path,require=c.need,HB228_EVAL_STRIDE=stride)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<port args>','exec'),ns)
    return ns['worker_args']


def test_real_manifest_only_stride_changes():
    m=json.loads((b.ROOT/'storage/external/heliosli-r6-fe055406-v1/manifest.json').read_bytes())
    old,new=b.package('candidate')[2:4]
    a=worker_args(old['exp169_r7_port.py'].decode())(m,'bundle','anatomy')
    z=worker_args(new['exp169_r7_port.py'].decode())(m,'bundle','anatomy')
    i=a.index('--eval-stride')+1
    assert a[i]=='1' and z[i]=='2' and '--flip-tta' in a
    a[i]='2'; assert a==z and m['eval_stride']==1
    src=b.package('pair')[3]['exp169_r7_port.py'].decode()
    assert worker_args(src,2)(m,'bundle','anatomy')==z


def test_adapter_patch_fail_closed():
    with pytest.raises(ValueError):
        b.adapter(b'unknown adapter')


def test_comparator_alignment_constants_and_invalid(tmp_path):
    ids=['a','b','c','d']; x=np.tile(np.arange(4)[:,None]/4,(1,12)); x[:,0]=.4
    a,z=tmp_path/'a.csv',tmp_path/'b.csv'
    csv_file(a,ids,x); csv_file(z,ids[::-1],x[::-1])
    r=c.compare(a,z,expected_ids=ids)
    assert r['max_abs_delta']==0 and r['per_target'][c.TARGETS[0]]['spearman'] is None
    assert r['per_target'][c.TARGETS[1]]['spearman']==1
    csv_file(z,['a','a','c','d'],x)
    with pytest.raises(ValueError,match='duplicate'): c.compare(a,z)
    csv_file(z,['a','b','c','e'],x)
    with pytest.raises(ValueError,match='roster'): c.compare(a,z)
    for value in (np.nan,np.inf,-.1,1.1):
        y=x.copy(); y[0,0]=value
        csv_file(z,ids,y)
        with pytest.raises(ValueError,match='finite'): c.compare(a,z)
    csv_file(z,ids,x)
    with pytest.raises(ValueError,match='selection'): c.compare(a,z,expected_ids=['a','b'])


def test_ruler_auc_aggregate(tmp_path,monkeypatch):
    from tools import evidence_ledger_rulers as rulers
    ids=['a','b','c','d']
    x=np.tile(np.arange(4)[:,None]/4,(1,12))
    a,z=tmp_path/'a.csv',tmp_path/'b.csv'
    csv_file(a,ids,x); csv_file(z,ids,x[::-1])
    labels=np.tile(np.array([0,0,1,1])[:,None],(1,12)); labels[:,0]=0
    csv_file(tmp_path/'reference.csv',ids,labels)
    monkeypatch.setattr(rulers,'source_specs',lambda:dict(test=dict(
        path='reference.csv',rule='binary',ancestors=['test'])))
    monkeypatch.setattr(rulers,'GOLD','absent.csv')
    r=c.compare(a,z,rulers_root=tmp_path)
    values=r['auc']['test']['per_target']
    assert values[c.TARGETS[0]]['status']=='SINGLE_CLASS'
    assert values[c.TARGETS[1]]['delta']==-1
    assert 'labels' not in r['auc']['test'] and r['auc']['gold']['status']=='MISSING'


@pytest.mark.parametrize('bad_fallback',[False,True])
def test_worker_execution_boundary(tmp_path,monkeypatch,bad_fallback):
    work=tmp_path/'hb228-pair'; roster=work/'n32'; roster.mkdir(parents=True)
    ids=[str(i) for i in range(32)]
    pair.write(roster/'selection.json',dict(ids=ids))
    pair.write(work/'paths.json',dict(code=str(work/'code'),anatomy=str(work/'anatomy'),roster=str(roster)))
    src=b.package('pair')[3]['exp169_r7_port.py'].decode()
    port=SimpleNamespace(VERSION='1.0.2-hb228-pair',MANIFEST_SHA256='pin',
                         resolve_bundle=lambda:Path('bundle'))
    manifest=json.loads((b.ROOT/'storage/external/heliosli-r6-fe055406-v1/manifest.json').read_bytes())
    calls=[]
    def infer(studies,**kwargs):
        calls.append(kwargs)
        assert len(studies)==32 and kwargs['max_fallback_fraction']==0
        out=kwargs['output_dir']
        csv_file(out/'predictions.csv',ids,np.full((32,12),.4))
        args=worker_args(src,port.HB228_EVAL_STRIDE)(manifest,'bundle','anatomy')
        pair.write(out/'provenance.json',dict(worker_args=args))
        pair.write(out/'controller/submission_receipt.json',
                   dict(rows=32,fallback_studies=int(bad_fallback)))
    port.infer_studies=infer
    release=SimpleNamespace(RUN='codec-run',gate_s_r7=lambda *a:dict(status='ACTIVE'),
                            codec_receipt=lambda *a:{})
    monkeypatch.setitem(sys.modules,'exp169_release_c85r7codec',release)
    monkeypatch.setitem(sys.modules,'exp169_r7_port',port)
    if bad_fallback:
        with pytest.raises(ValueError,match='fallback'): pair.worker(tmp_path,tmp_path,2)
    else:
        pair.worker(tmp_path,tmp_path,1); pair.worker(tmp_path,tmp_path,2)
        assert len(calls)==2
        assert json.loads((work/'stride2/timing.json').read_bytes())['stride']==2
        assert not (tmp_path/'submission.csv').exists()


def test_pair_orchestration_same_roster_and_exports(tmp_path,monkeypatch):
    ids=[str(i) for i in range(32)]
    selection_calls=[]
    def prepare(comp,out):
        selection_calls.append(out); pair.write(out/'selection.json',dict(ids=ids))
    monkeypatch.setattr(pair.n32,'prepare',prepare)
    def child(args,**kwargs):
        stride=int(args[-1])
        work=tmp_path/'hb228-pair'
        csv_file(work/f'stride{stride}/s_r7/predictions.csv',ids,np.full((32,12),.4))
        pair.write(work/f'stride{stride}/timing.json',dict(inference_seconds=10/stride))
    release=SimpleNamespace(RUN='codec-run',competition_root=lambda:tmp_path/'comp',
        s6=SimpleNamespace(prepare_code=lambda *a,**k:(tmp_path/'code',tmp_path/'anatomy'),
                           install_wheels=lambda *a:None),
        v5q=SimpleNamespace(CODE_SLUG='code',_mount=lambda *a:tmp_path,run_process_group=child),
        install_codecs=lambda *a:{},codec_dirs=lambda *a:(tmp_path,tmp_path/'tele'),
        child_base_env=lambda *a:{},codecs=SimpleNamespace(child_env=lambda env,*a,**k:env))
    monkeypatch.setitem(sys.modules,'exp169_release_c85r7codec',release)
    pair.run(tmp_path/'runtime',tmp_path)
    assert len(selection_calls)==1
    rec=json.loads((tmp_path/'hb228-pair/receipt.json').read_bytes())
    assert rec['never_submit'] and rec['inference_seconds_saved']==5
    assert not rec['qualified_for_submission'] and not (tmp_path/'submission.csv').exists()
    with pytest.raises(ValueError,match='fresh'): pair.run(tmp_path/'runtime',tmp_path)
