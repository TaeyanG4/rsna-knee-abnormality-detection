"""Offline split transport/binder tests; all target values are synthetic."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tools import pl_vast_split_v110 as s


def fixture(root):
    members=[dict(member=dict(fold=f,seed=42,kind='MIL',id=f'mil-s42-f{f}'),spec_sha256=str(f)*64) for f in range(5)]
    b=dict(schema=s.SCHEMA,members=members,assignment=s.split(dict(members=members)),
           exposure='EXPOSED_EXPLORATORY',clean_oof=False,files={},retained={},teacher={'sha256':'a'*64})
    path=root/'bundle.json';s.write(path,b)
    return path,s.sha(path),b


def remote(root,b,expected,fold):
    d=root/f'fits/fold{fold}';d.mkdir(parents=True)
    (d/'model_best.pt').write_bytes(b'engineering fixture bytes')
    r=dict(status='COMPLETE',engineering_only=False,fold=fold,seed=42,spec_sha256=b['members'][fold]['spec_sha256'],
           teacher_sha256=b['teacher']['sha256'],checkpoint_sha256=s.sha(d/'model_best.pt'))
    s.write(d/'training-receipt.json',r)
    s.write(d/'device.json',dict(schema=s.SCHEMA,fold=fold,bundle_sha256=expected,engineering_only=False,
            device_name='NVIDIA GeForce RTX 3090',spec_sha256=r['spec_sha256'],checkpoint_sha256=r['checkpoint_sha256'],
            receipt_sha256=s.sha(d/'training-receipt.json')))


def test_split_exact_roster(tmp_path):
    _,_,b=fixture(tmp_path)
    assert s.split(b)['local_folds']==[0,1,2]
    assert s.split(b)['remote_folds']==[3,4]
    for change in ('duplicate','seed','kind','missing'):
        altered=copy.deepcopy(b)
        if change=='duplicate':altered['members'][4]['member']['fold']=3
        if change=='seed':altered['members'][4]['member']['seed']=43
        if change=='kind':altered['members'][4]['member']['kind']='CoAt'
        if change=='missing':altered['members'].pop()
        with pytest.raises(ValueError):s.split(altered)


@pytest.mark.parametrize('path',['../escape','/tmp/escape','C:/absolute','a\\b','.ssh/key','.local/relabel/x'])
def test_unsafe_paths(path):
    with pytest.raises(ValueError):s.safe(path)


def test_external_anchor_and_retained_hash(tmp_path):
    path,expected,b=fixture(tmp_path)
    with pytest.raises(ValueError,match='anchor'):s.verify(path,'0'*64)
    (tmp_path/'retained.bin').write_bytes(b'corpus')
    b['retained']={'retained.bin':s.pin(tmp_path/'retained.bin')}
    path.write_text(json.dumps(b));expected=s.sha(path)
    s.verify(path,expected,tmp_path)
    (tmp_path/'retained.bin').write_bytes(b'drift')
    with pytest.raises(ValueError,match='hash'):s.verify(path,expected,tmp_path)


def test_source_projection_authenticates_windows_pins(tmp_path):
    p=tmp_path/'tools/a.py';p.parent.mkdir();p.write_text('pass')
    spec={'source_pins':[dict(path='H:\\project\\tools\\a.py',**s.pin(p))]}
    s.source_projection(spec,tmp_path)
    p.write_text('changed')
    with pytest.raises(ValueError,match='hash'):s.source_projection(spec,tmp_path)


@pytest.mark.parametrize('folds',[(3,),(4,),(3,4)])
def test_pack_verify_partial_and_complete_results(tmp_path,folds):
    path,expected,b=fixture(tmp_path)
    for f in folds:remote(tmp_path,b,expected,f)
    s.pack_results(path,expected,tmp_path/'results.zip')
    assert s.verify_results(path,expected,tmp_path)['folds']==list(folds)
    (tmp_path/f'fits/fold{folds[0]}/model_best.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):s.verify_results(path,expected,tmp_path)


def test_pack_rejects_engineering_receipt(tmp_path):
    path,expected,b=fixture(tmp_path);remote(tmp_path,b,expected,3)
    rec=tmp_path/'fits/fold3/training-receipt.json';r=s.read(rec);r['status']='ENGINEERING_SMOKE';rec.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='incomplete'):s.pack_results(path,expected,tmp_path/'results.zip')


def test_memory_gate_full_shape_finite_rates(tmp_path):
    p=tmp_path/'memory.json'
    m=dict(schema=s.SCHEMA,bundle_sha256='anchor',status='PASS',device_name='RTX 3090',
           shape=dict(batch=8,k_train=12,k_eval=36,resolution=224,tta=True),
           peak_reserved_bytes=dict(fit=1000,tta=2000),healthy_rates={'train_step':.5})
    p.write_text(json.dumps(m));s.memory_gate(p,'anchor')
    for broken in (dict(m,shape={}),dict(m,healthy_rates={'train_step':float('nan')}),dict(m,peak_reserved_bytes=dict(fit=11*1024**3,tta=2))):
        p.write_text(json.dumps(broken))
        with pytest.raises(ValueError):s.memory_gate(p,'anchor')


def test_cpu_measure_refuses_before_cuda():
    r=subprocess.run([sys.executable,'-B','-m','tools.pl_vast_split_v110','measure','--device','cpu'],capture_output=True,text=True)
    assert r.returncode!=0 and 'explicit CUDA' in r.stderr


def test_review_rejects_hash_pinned_failure(tmp_path):
    path,expected,b=fixture(tmp_path)
    sources={}
    for name in s.REVIEW_SOURCES:
        code=tmp_path/name;code.parent.mkdir(exist_ok=True);code.write_text('fixture')
        b['files'][name]=s.pin(code);sources[name]=s.sha(code)
    evidence=tmp_path/'metadata/evidence.json';s.write(evidence,dict(status='HOLD'))
    r=dict(status='PASS',independent=True,reviewer='fixture',sources=sources,
           cpu_suite=dict(path='metadata/evidence.json',**s.pin(evidence)),real_cpu_smoke=dict(path='metadata/evidence.json',**s.pin(evidence)))
    s.write(tmp_path/'metadata/split-review.json',r)
    path.write_text(json.dumps(b));expected=s.sha(path)
    with pytest.raises(ValueError,match='passing CPU'):s.review_gate(None,path,expected)


def test_review_cannot_bypass_current_loader_evidence(tmp_path):
    path,expected,b=fixture(tmp_path)
    sources={}
    for name in s.REVIEW_SOURCES:
        code=tmp_path/name;code.parent.mkdir(exist_ok=True);code.write_text('fixture')
        b['files'][name]=s.pin(code);sources[name]=s.sha(code)
    old_sources=dict(sources);old_sources['tools/pl_production_loader_v1.py']='0'*64
    evidence=tmp_path/'metadata/evidence.json';s.write(evidence,dict(status='PASS',sources=old_sources))
    r=dict(status='PASS',independent=True,reviewer='fixture',sources=sources,
           cpu_suite=dict(path='metadata/evidence.json',**s.pin(evidence)),real_cpu_smoke=dict(path='metadata/evidence.json',**s.pin(evidence)))
    s.write(tmp_path/'metadata/split-review.json',r)
    path.write_text(json.dumps(b));expected=s.sha(path)
    with pytest.raises(ValueError,match='current-source'):s.review_gate(None,path,expected)


@pytest.mark.parametrize('remote_folds',[(3,4),(3,),()])
def test_binder_mixed_and_whole_fold_fallback(tmp_path,monkeypatch,remote_folds):
    from tools import pl_production_launch_v1 as launch
    path,expected,b=fixture(tmp_path)
    memory=tmp_path/'memory.json';s.write(memory,dict(device_name='NVIDIA GeForce RTX 4070 Ti SUPER'))
    plan=dict(members=copy.deepcopy(b['members']),arm='global',stage=str(tmp_path/'stage'),memory=launch.asset(memory),
              go={},review={'path':'fixture'},teacher=b['teacher'])
    for e in plan['members']:
        f=e['member']['fold'];e['output']=str(tmp_path/f'local{f}');e['spec']='fixture'
        d=Path(e['output']);d.mkdir();(d/'model_best.pt').write_bytes(b'fixture');s.write(d/'training-receipt.json',{})
    planpath=tmp_path/'plan.json';s.write(planpath,plan);b['original_plan']=launch.asset(planpath)
    path.write_text(json.dumps(b));expected=s.sha(path)
    for f in remote_folds:remote(tmp_path,b,expected,f)
    if remote_folds:s.pack_results(path,expected,tmp_path/'results.zip')
    monkeypatch.setattr(launch.p,'pin',lambda a:None)
    monkeypatch.setattr(launch,'review_gate',lambda a:None)
    monkeypatch.setattr(launch,'selected',lambda a:None)
    plan['go']={'path':'fixture'};planpath.write_text(json.dumps(plan));b['original_plan']=launch.asset(planpath)
    # Bundle anchor changed, so rebind engineering device receipts to the updated fixture.
    path.write_text(json.dumps(b));new=s.sha(path)
    if remote_folds:
        for f in remote_folds:
            dev=tmp_path/f'fits/fold{f}/device.json';v=s.read(dev);v['bundle_sha256']=new;dev.write_text(json.dumps(v))
        result=s.read(tmp_path/'results.json');result['bundle_sha256']=new
        for n in result['files']:result['files'][n]=s.pin(tmp_path/n)
        (tmp_path/'results.json').write_text(json.dumps(result))
    calls=[]
    monkeypatch.setattr(launch,'completed',lambda e,p:calls.append(e) or True)
    monkeypatch.setattr(launch.p,'registration',lambda _: {})
    (tmp_path/'stage/student/global').mkdir(parents=True)
    s.write(tmp_path/'stage/student/global/bindings-template.json',{})
    def delegate(stage,inputs,out):s.write(out,s.read(inputs));return {'status':'BOUND'}
    monkeypatch.setattr(launch.release,'bind_completed',delegate)
    out=tmp_path/'binding.json'
    s.bind(planpath,path,new,tmp_path if remote_folds else None,out)
    assert len(calls)==5
    assert ['3090' in e['training_gpu'] for e in s.read(out)['members']]==[f in remote_folds for f in range(5)]


def test_prepare_missing_review_is_closed(tmp_path):
    with pytest.raises(ValueError,match='review absent'):s.prepare(tmp_path/'missing',tmp_path/'output',None)

def test_record_review_cannot_promote_failed_smoke(tmp_path):
    paths=[]
    for kind in ('suite','real_cpu_smoke'):
        p=tmp_path/(kind+'.json');s.write(p,dict(status='PASS' if kind=='suite' else 'HOLD',kind=kind,
                         sources={'tools/pl_vast_split_v110.py':s.sha(s.__file__)}));paths.append(p)
    with pytest.raises(ValueError,match='passing current-source'):
        s.record_review('reviewer',*paths,tmp_path/'review.json')
    assert not (tmp_path/'review.json').exists()


@pytest.mark.parametrize('omitted',['tools/pl_production_loader_v1.py','tools/pl_production_train_v1.py'])
def test_record_review_requires_current_loader_and_train(tmp_path,omitted):
    sources={name:s.sha(s.ROOT/name) for name in s.REVIEW_SOURCES}
    sources.pop(omitted)
    paths=[]
    for kind in ('suite','real_cpu_smoke'):
        path=tmp_path/(kind+'.json');s.write(path,dict(status='PASS',kind=kind,sources=sources));paths.append(path)
    with pytest.raises(ValueError,match='current-source'):
        s.record_review('reviewer',*paths,tmp_path/'review.json')


def test_fit_forwards_vast_worker_configuration(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from tools import pl_production_train_v1 as train
    from tools import vnext_r7_pl_go_v111 as go
    spec=dict(spec_sha256='a'*64,recipe={'model':{'weights':'weights.pt','initial_state':'initial.pt'}})
    monkeypatch.setattr(s,'load_fold',lambda *a:({'arm':'global'},{'admission':{}},spec,{},{}))
    monkeypatch.setattr(s,'go_mapping',lambda *a:nullcontext())
    monkeypatch.setattr(go,'admission',lambda *a:None)
    captured={}
    def fake_fit(*args,**kwargs):
        captured.update(kwargs);s.write(tmp_path/'training-receipt.json',{'status':'ENGINEERING_SMOKE'})
        return {'checkpoint_sha256':'b'*64}
    monkeypatch.setattr(train,'mil_fit',fake_fit)
    s.fit('bundle','anchor',tmp_path,3,tmp_path,workers=3)
    assert captured['workers']==3 and captured['loader_profile']=='vast'


def test_supervisor_propagates_workers(tmp_path,monkeypatch):
    from argparse import Namespace
    from contextlib import nullcontext
    from tools import hb150_watchdog_v1 as watchdog
    from tools import job_runtime
    captured=[]
    monkeypatch.setattr(s,'verify',lambda *a:None)
    monkeypatch.setattr(s,'review_gate',lambda *a:None)
    monkeypatch.setattr(job_runtime,'file_lock',lambda *a,**kw:nullcontext())
    monkeypatch.setattr(watchdog,'supervise',lambda argv,*a,**kw:captured.extend(argv) or 0)
    args=Namespace(device='cuda',deadline=s.time.time()+500,bundle=tmp_path/'bundle.json',expected='anchor',
        review=None,smoke=True,command='fit',retained_root=tmp_path,output=tmp_path/'fit',fold=3,workers=4)
    s.supervised(args)
    assert captured[captured.index('--workers')+1]=='4'


def test_cli_forwards_workers(monkeypatch,capsys):
    captured={}
    monkeypatch.setattr(sys,'argv',['split','fit','--workers','2','--device','cpu','--smoke'])
    monkeypatch.setattr(s,'fit',lambda *a,**kw:captured.update(kw) or {'status':'PASS'})
    s.main()
    assert captured['workers']==2


def test_local_launcher_workers_are_passed():
    launcher=(s.ROOT/'.local/run-scripts/pl-local-split-v110.ps1').read_text()
    assert '[int]$Workers = 6' in launcher and "'--workers',[string]$Workers" in launcher


@pytest.mark.parametrize('arm',['global','fill'])
def test_portable_materialized_targets_and_exclusions(tmp_path,arm):
    import numpy as np
    import pandas as pd
    from tools import pl_production_targets_v1 as t
    from tools import vnext_r7_pl_train_v110 as base
    from tools import vnext_r7_pl_v1 as p
    ids=[*t.exclusions.REGISTERED_UIDS,*[f'synthetic-{i}' for i in range(48)]]
    rows=pd.DataFrame({t.pl.UID:ids,'component_id':ids,'fold':[str(i%5) for i in range(50)]})
    c=t.build_cohort(rows,[],ids[2:],ids[:2]);rng=np.random.default_rng(17)
    l2={u:rng.uniform(.1,.9,12).astype(np.float32) for u in ids}
    teacher={u:rng.uniform(.01,.99,12).astype(np.float32) for u in ids[2:]}
    train,_=t.a3.selected_rosters(c,'fivefold',3)
    target_dir=tmp_path/'metadata/fold3'
    target=t.materialize(target_dir,c,l2,teacher,None,arm,3,'a'*64,np.ones((len(train),12),bool),engineering=True)
    # Entirely synthetic fixture; flag projection exists only inside this isolated unit test.
    target['engineering_only']=False
    target['teacher_sha256']='a'*64
    target['spec_sha256']=p.digest({k:v for k,v in target.items() if k!='spec_sha256'})
    (target_dir/'target-spec.json').write_text(json.dumps(target))
    path,expected,b=fixture(tmp_path);b['arm']=arm
    spec=dict(member=b['members'][3]['member'],source_pins=[],registration_sha256=target['registration_sha256'],
              target_bundle=s.pin(target_dir/'target-spec.json'),recipe=base.mil_recipe(len(train)))
    spec['spec_sha256']=p.digest(spec);s.write(target_dir/'training-spec.json',spec)
    b['members'][3].update(spec='metadata/fold3/training-spec.json',target='metadata/fold3/target-spec.json',
                          payload='metadata/fold3/targets.npz',spec_sha256=spec['spec_sha256'])
    for f in target_dir.iterdir():b['files'][f.relative_to(tmp_path).as_posix()]=s.pin(f)
    path.write_text(json.dumps(b));expected=s.sha(path)
    _,_,_,loaded,labels=s.load_fold(path,expected,tmp_path,3)
    assert loaded['l2_only_ids'] and len(labels)==40
    for uid in loaded['l2_only_ids']:np.testing.assert_array_equal(labels[uid],l2[uid])
    with pytest.raises(ValueError,match='remote folds'):s.load_fold(path,expected,tmp_path,2)
    (target_dir/'targets.npz').write_bytes(b'corruption')
    with pytest.raises(ValueError,match='hash'):s.load_fold(path,expected,tmp_path,3)
