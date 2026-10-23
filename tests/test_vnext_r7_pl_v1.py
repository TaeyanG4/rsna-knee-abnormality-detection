"""Placeholder release contracts; no model fitting, label tables or CUDA calls."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tools import vnext_r7_pl_v1 as p
from tools import vnext_r7_pl_worker_v1 as w

IDS = ['study-z', 'study-a', 'study-b']


@pytest.mark.parametrize('fault', ['none', 'public', 'manifest', 'legacy', 'bundle', 'code'])
def test_parent_runtime_authentication(tmp_path, monkeypatch, fault):
    from tools import hb116_vnext_release as h, hb127_vnextr7_release as base
    runtime = {key: str(tmp_path/key) for key in ('public_source','own_manifest','legacy','bundle','source','code','anatomy')}
    for key in ('public_source', 'own_manifest'):
        Path(runtime[key]).write_text('pinned')
    legacy = Path(runtime['legacy'])/'module.py'; legacy.parent.mkdir(); legacy.write_text('pinned')
    expected = p.sha(legacy)
    monkeypatch.setitem(h.c.p.PINS, 'public-serving.py', expected)
    monkeypatch.setattr(h, 'OWN_MANIFEST_SHA', expected)
    checked = []
    def verify(label):
        def run(*args):
            checked.append(label)
            if fault == label: raise ValueError(label+' authentication')
        return run
    monkeypatch.setattr(base.r7, 'verify_bundle', verify('bundle'))
    monkeypatch.setattr(base.r7, 'code_record', verify('code'))
    if fault in ('public', 'manifest', 'legacy'):
        target = legacy if fault=='legacy' else Path(runtime['public_source' if fault=='public' else 'own_manifest'])
        target.write_text('wrong')
    r = dict(payload_pins={'legacy/module.py': expected})
    if fault=='none':
        p.runtime_preflight(r, runtime)
        assert checked==['bundle','code']
    else:
        with pytest.raises(ValueError): p.runtime_preflight(r, runtime)


def reg():
    return dict(members=p.member_roster('all-data', [42]),
        weights=dict(primary=.30, sensitivity=.40, own=[17,17,12],
                     own_order=['PL-CoAt','EXP155','PL-MIL'], public=[1,1]))


def fixture(root, failed=None):
    r = reg()
    rng = np.random.default_rng(146)
    raw = {e['id']: rng.integers(0, 8, (3,12)).astype(np.float32)/8 for e in r['members']}
    specs = []
    for branch in ('own','R','R7'):
        entries = [e for e in r['members'] if e['branch'] == branch]
        ident = p.identity(r, branch, entries, 'test-run')
        spec = dict(path=root/branch, ids=IDS, identity=ident)
        j = p.Journal(spec['path'], IDS, ident)
        for e in entries:
            for i, uid in enumerate(IDS):
                j.record(uid, e['id'], raw[e['id']][i], error='failed' if branch == failed else None)
                if branch == failed:
                    raw[e['id']][i] = np.nan
        j.finish()
        specs.append(spec)
    return r, specs, raw


def reference(r, raw, weight):
    # Independent tie ranks and per-study survivor renormalization.
    def rank(a):
        out = np.full(a.shape, np.nan)
        ok = np.isfinite(a).all(1)
        for j in range(12):
            v = a[ok,j]
            out[ok,j] = [(sum(v < x) + .5*(sum(v == x)+1))/len(v) for x in v]
        return out
    def avg(arrays, weights=None, dtype=np.float64):
        a = np.asarray(arrays, dtype=dtype)
        weights = np.asarray(weights if weights is not None else [1]*len(a), dtype=dtype)
        ok = np.isfinite(a).all(2)
        total = np.where(ok, weights[:,None], 0).sum(0)
        out = np.full(a.shape[1:], np.nan, dtype=dtype)
        np.divide((np.where(ok[:,:,None], a, 0)*weights[:,None,None]).sum(0), total[:,None], out=out, where=total[:,None]>0)
        return out
    mil = avg([raw[e['id']] for e in r['members'] if e['kind']=='MIL'], dtype=np.float32)
    own = avg([rank(raw['PL-CoAt']),rank(raw['EXP155']),rank(mil)], [17,17,12])
    public = avg([rank(avg([raw['R2-s43'],raw['R2-s44']], dtype=np.float32)),rank(raw['S_R7'])])
    weight = r['weights'][weight]
    result = avg([rank(public),rank(own)],[1-weight,weight])
    result[~np.isfinite(result).all(1)] = .5
    return result.astype(np.float32)


@pytest.mark.parametrize('weight',['primary','sensitivity'])
@pytest.mark.parametrize('failed',[None,'own','R','R7'])
def test_independent_arithmetic(tmp_path,weight,failed):
    r,specs,raw = fixture(tmp_path,failed)
    np.testing.assert_array_equal(p.blend(r,raw,weight),reference(r,raw,weight))
    diagnostic = weight == 'sensitivity'
    filename = 'diagnostic.csv' if diagnostic else 'submission.csv'
    result = p.publish(r,IDS,specs,tmp_path/'out',weight,filename=filename,diagnostic=diagnostic)
    assert result['fallbacks'] == sum(np.isnan(a).all(1).sum() for a in raw.values())
    with (tmp_path/'out'/filename).open() as stream:
        rows = list(p.csv.reader(stream))
    assert [row[0] for row in rows[1:]] == IDS


def test_sensitivity_serve_rejected_before_any_work(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('production sensitivity must fail before any work')
    monkeypatch.setattr(p, 'registration', forbidden)
    monkeypatch.setattr(p.Deadline, 'from_env', forbidden)
    with pytest.raises(ValueError, match='diagnostic only'):
        p.serve('absent-stage', 'absent-bindings', tmp_path/'out', weight='sensitivity')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('filename,diagnostic', [
    ('submission.csv', False), ('diagnostic.csv', False),
    ('submission.csv', True), ('SUBMISSION.CSV', True),
    ('../submission.csv', True), ('alternate.csv', False),
])
def test_sensitivity_cannot_publish_submission(tmp_path, filename, diagnostic):
    with pytest.raises(ValueError):
        p.publish(reg(), IDS, [], tmp_path/'out', 'sensitivity',
                  filename=filename, diagnostic=diagnostic)
    assert not (tmp_path/'out').exists()


def test_sensitivity_explicit_diagnostic_only(tmp_path):
    result = p.publish(reg(), IDS, [], tmp_path, 'sensitivity',
                       filename='diagnostic.csv', diagnostic=True)
    assert result['status'] == 'DIAGNOSTIC_ONLY'
    assert result['qualified_for_submission'] is False
    assert (tmp_path/'diagnostic.csv').exists()
    assert not (tmp_path/'submission.csv').exists()
    assert p.read(tmp_path/'publication.json')['status'] == 'DIAGNOSTIC_ONLY'


@pytest.mark.parametrize('branch',['own','R','R7'])
@pytest.mark.parametrize('state',['missing','truncated','corrupt','incomplete','failures-shape','stale'])
def test_receipt_never_upgrades_branch(tmp_path,branch,state):
    r,specs,_ = fixture(tmp_path/'actual')
    _,failed_specs,_ = fixture(tmp_path/'failed',branch)
    expected = p.publish(r,IDS,failed_specs,tmp_path/'expected')
    receipt = tmp_path/'actual'/branch/'receipt.json'
    value = p.read(receipt)
    if state == 'missing': receipt.unlink()
    elif state == 'truncated': receipt.write_bytes(b'{"status":')
    elif state == 'corrupt': receipt.write_bytes(b'[]')
    elif state == 'incomplete': p.dump(receipt, {'identity':value['identity']})
    elif state == 'failures-shape': p.dump(receipt, dict(value,failures=[{}]))
    else:
        value['identity']['run_id'] = 'stale'
        p.dump(receipt,value)
        with pytest.raises(ValueError,match='authentication'):
            p.publish(r,IDS,specs,tmp_path/'out')
        assert not (tmp_path/'out/submission.csv').exists()
        return
    actual = p.publish(r,IDS,specs,tmp_path/'out')
    assert actual['fallbacks'] == expected['fallbacks']
    assert (tmp_path/'out/submission.csv').read_bytes() == (tmp_path/'expected/submission.csv').read_bytes()


@pytest.mark.parametrize('site',['identity.json','assignment.json','receipt.json','row'])
@pytest.mark.parametrize('state',['missing','truncated','shape'])
@pytest.mark.parametrize('branch',['own','R','R7'])
def test_metadata_faults_publish(tmp_path,site,state,branch):
    r,specs,_ = fixture(tmp_path)
    entry = next(e for e in r['members'] if e['branch']==branch)
    path = tmp_path/branch/(f"rows/{p.digest([IDS[0],entry['id']])}.json" if site=='row' else site)
    if state=='missing': path.unlink()
    else: path.write_bytes(b'{' if state=='truncated' else b'{"unexpected":1}')
    result = p.publish(r,IDS,specs,tmp_path/'out')
    assert result['rows']==3 and result['fallbacks']>0


@pytest.mark.parametrize('site',['identity.json','assignment.json','receipt.json','row'])
@pytest.mark.parametrize('boundary',['fsync','replace'])
def test_interrupted_actual_write_sites(tmp_path,monkeypatch,site,boundary):
    ident = dict(members={'member':'pin'},branch='own')
    journal = None if site in ('identity.json','assignment.json') else p.Journal(tmp_path,IDS,ident)
    original = getattr(p.metadata.os,boundary)
    calls = 0
    def fail(*args,**kwargs):
        nonlocal calls
        calls += 1
        if (boundary=='fsync' and calls == (2 if site=='assignment.json' else 1)) or (boundary=='replace' and (Path(args[1]).name==site or site=='row')):
            raise OSError('interrupted')
        return original(*args,**kwargs)
    monkeypatch.setattr(p.metadata.os,boundary,fail)
    with pytest.raises(OSError):
        if journal is None: p.Journal(tmp_path,IDS,ident)
        elif site=='receipt.json': journal.finish()
        else: journal.record(IDS[0],'member',[.2]*12)
    monkeypatch.setattr(p.metadata.os,boundary,original)
    if site == 'assignment.json':
        assert (tmp_path/'identity.json').is_file() and not (tmp_path/site).exists()
    raw,count,_ = p.collect(tmp_path,IDS,ident)
    assert count==3 and np.isnan(raw['member']).all()


@pytest.mark.parametrize('bad',[['not-numeric']*12,None,[.1]*11,[float('nan')]*12])
def test_bad_prediction_and_failure_sticky(tmp_path,bad):
    ident = dict(members={'member':'pin'},branch='own')
    j = p.Journal(tmp_path,IDS,ident)
    j.record(IDS[0],'member',bad)
    j.record(IDS[0],'member',[.8]*12)
    j.finish()
    assert p.collect(tmp_path,IDS,ident)[1]==3


def test_corrupt_existing_row_and_bad_values_stay_failed(tmp_path):
    ident = dict(members={'member':'pin'},branch='own')
    j = p.Journal(tmp_path,IDS,ident)
    j.record(IDS[0],'member',[.8]*12)
    path = tmp_path/'rows'/(p.digest([IDS[0],'member'])+'.json')
    path.write_bytes(b'{')
    j.record(IDS[0],'member',[.8]*12)
    j.finish()
    assert p.collect(tmp_path,IDS,ident)[1]==3
    row=p.read(path);row.update(status='OK',values=['bad']*12)
    p.dump(path,row)
    assert p.collect(tmp_path,IDS,ident)[1]==3


def test_all_missing_and_optional_receipt_failure(tmp_path,monkeypatch):
    r = reg()
    monkeypatch.setattr(p,'dump',lambda *a,**k: (_ for _ in ()).throw(OSError('disk receipt')))
    result = p.publish(r,IDS,[],tmp_path)
    assert result['fallbacks']==len(IDS)*len(r['members'])
    a=np.loadtxt(tmp_path/'submission.csv',delimiter=',',skiprows=1,usecols=range(1,13))
    np.testing.assert_array_equal(a,np.full((3,12),.5))


def test_plans_and_sealed_weights():
    for arm in ('global','fill'):
        plans=p.training_configs(arm)
        assert len(plans)==13 and sum(x['family']=='MIL' and x['mode']=='fivefold' for x in plans)==10
        assert all(x['target_contract']['role']=='heldout' and x['target_contract']['clean_oof'] is False for x in plans)
        assert plans[-1]['recipe']['swa_1based']==[14,11,12]
    assert len(p.member_roster())==15
    with pytest.raises(ValueError): p.member_roster(seeds=[43])


@pytest.mark.parametrize('fail',[False,True])
def test_native_routes_and_restores(tmp_path,fail):
    from tools import hb116_vnext_release as h
    r,specs,_=fixture(tmp_path/'initial')
    specs=[dict(s,path=tmp_path/'native'/s['identity']['branch']) for s in specs if s['identity']['branch']!='R7']
    originals=(h.c.Journal,h.dump,h.q_model,h.anchor_setup)
    with pytest.raises(RuntimeError) if fail else p.context_logs():
        with w.native_adapters(dict(branches=specs),h) as journals:
            dispatch=h.c.Journal(None,None,None)
            for spec in specs:
                for member in spec['identity']['members']:
                    for uid in IDS: dispatch.record(uid,member,[.2]*12)
            if fail: raise RuntimeError('worker failed')
            for journal in journals: journal.finish()
    assert originals==(h.c.Journal,h.dump,h.q_model,h.anchor_setup)
    for spec in specs:
        assert p.collect(spec['path'],IDS,spec['identity'])[1] == (3*len(spec['identity']['members']) if fail else 0)


def test_checkpoint_provenance_rejects_stale_teacher():
    entry=dict(arch=p.COAT_ARCH,kind='CoAt',seed=42)
    rec=dict(arm='global',teacher_sha256='teacher',target_sha256='target',train_ids_sha256='ids',
        l2_pos_weight_sha256='weights',spec_sha256='spec',registration_sha256='reg',clean_oof=False,exposure='EXPOSED_EXPLORATORY')
    ck=dict(arch=p.COAT_ARCH,lab=p.TARGETS,seed=42,res=384,inference_res=320,contract='exp143_raptor96_384_c140_s02',
        k_eval=94,norm='imagenet',window_policy='legacy',oai=False,checkpoint_kind='fixed-swa-bn',swa_epochs_1based=[14,11,12],pl_provenance=rec.copy())
    w.checkpoint_metadata(ck,entry,rec)
    ck['pl_provenance']['teacher_sha256']='f0'
    with pytest.raises(ValueError,match='provenance'): w.checkpoint_metadata(ck,entry,rec)


@pytest.mark.parametrize('fault',['codec','job','all-child-fail'])
def test_supervised_serving_publishes_after_cleanup(tmp_path,monkeypatch,fault):
    r=reg();r['deadline_seconds']=31200
    comp=tmp_path/'comp';comp.mkdir()
    (comp/'sample_submission.csv').write_text(','.join([p.UID,*p.TARGETS])+'\n'+''.join(u+','+','.join(['0']*12)+'\n' for u in IDS))
    (comp/'test_series.csv').write_text(p.UID+'\n'+'\n'.join(IDS)+'\n')
    bound=dict(members=r['members'],runtime=dict(competition=str(comp)))
    monkeypatch.setattr(p,'registration',lambda stage:r)
    monkeypatch.setattr(p,'validate_bindings',lambda *a:r['members'])
    monkeypatch.setattr(p,'admit',lambda *a:None)
    monkeypatch.setattr(p,'runtime_preflight',lambda *a:None)
    monkeypatch.setattr(p,'read',lambda path:bound)
    realdump=p.dump
    if fault=='job':
        monkeypatch.setattr(p,'dump',lambda path,value,**k: (_ for _ in ()).throw(OSError('interrupted job')) if Path(path).name=='job.json' else realdump(path,value,**k))
    cleanup=[]; masks=[]
    class Supervisor:
        def __init__(self,deadline): self.deadline=deadline
        def __enter__(self): return self
        def __exit__(self,*a): cleanup.append(True)
        def run(self,*a,**k):
            if fault=='codec': realdump(tmp_path/'out/codec.json',dict(status='ACTIVE'))
            return SimpleNamespace(returncode=7)
        def start(self,*a,**k):
            masks.append(k['env']['CUDA_VISIBLE_DEVICES'])
            return SimpleNamespace(label=k['label'],returncode=7,cleanup_error=None)
        def wait(self,child): pass
    monkeypatch.setattr(p,'Supervisor',Supervisor)
    realpublish=p.publish
    def publish(*args):
        assert cleanup
        return realpublish(*args)
    monkeypatch.setattr(p,'publish',publish)
    result=p.serve('stage','bindings',tmp_path/'out',deadline=p.Deadline.start(env={}))
    assert result['rows']==3 and result['fallbacks']==18
    assert masks==([] if fault=='job' else ['0','1','0,1'])


def binding_fixture(root):
    from pl_readout_fixture_v111 import make_readout
    from tools import vnext_r7_pl_train_v110 as train
    root.mkdir(parents=True)
    r = dict(reg(), arm='global')
    def asset(name, value):
        path=root/name
        p.dump(path,value)
        return dict(path=str(path),sha256=p.sha(path),bytes=path.stat().st_size)
    # Substitute immutable retained pins only in this isolated placeholder registration.
    for e in r['members']:
        if e['kind']!='R7':
            e.update(asset(e['id']+'.pt',{'placeholder':e['id']}))
    bound=dict(registration_sha256=p.digest(r),arm='global',members=copy.deepcopy(r['members']))
    bound['go']=train.asset(make_readout(root/'readout'))
    admission=dict(status='ADMITTED',spec_sha256='spec',go=bound['go'],
                   decision=train.scientific_go.validate(bound['go'],'global'))
    bound['teacher']=asset('teacher.json',dict(producer='H-B-145',producer_version='1.1.2',
        producer_exposure='EXPOSED',status='PASS',role='heldout',
        model_fold='int(sha1(uid)[:8],16)%4',output_kind='pre_stacker',complete=True,failures=0,
        clean_oof=False,exposure='EXPOSED_EXPLORATORY',producer_identity='producer',csv_sha256='targets'))
    for e in bound['members']:
        if e['kind'] in ('MIL','CoAt'):
            e['training_receipt']=asset(e['id']+'-receipt.json',dict(status='COMPLETE',checkpoint_sha256=e['sha256'],
                arm='global',seed=e['seed'],fold=e['fold'],teacher_sha256=bound['teacher']['sha256'],
                registration_sha256=p.digest(r),clean_oof=False,exposure='EXPOSED_EXPLORATORY',
                target_sha256='targets',train_ids_sha256='train',l2_pos_weight_sha256='l2',spec_sha256='spec',go=admission))
    bound['qualification']=asset('qualification.json',dict(registration_sha256=p.digest(r),
        bindings_members_sha256=p.digest(bound['members']),device='2xT4',upper_seconds=29510,
        publication_reserve_seconds=20,total_seconds=29530,complete_ordered_csv=True,finite=True,
        zero_healthy_fallbacks=True,descendants_exited=True,gpu_memory_released=True,
        fault_n32_pass=True,native_parity=True,codecs_best_effort=True))
    return r,bound,asset


@pytest.mark.parametrize('fault',['none','go','f0','incomplete-teacher','stale-receipt','checkpoint','runtime','cleanup','weight'])
def test_binding_admission_guards(tmp_path,monkeypatch,fault):
    r,b,asset=binding_fixture(tmp_path/'assets')
    if fault=='go': b['go']=asset('bad-go.json',dict(status='HOLD'))
    if fault in ('f0','incomplete-teacher'):
        teacher=p.read(b['teacher']['path'])
        teacher.update({'role':'f0'} if fault=='f0' else {'complete':False})
        b['teacher']=asset('bad-teacher.json',teacher)
    if fault=='stale-receipt':
        e=b['members'][0]; receipt=p.read(e['training_receipt']['path'])
        e['training_receipt']=asset('bad-receipt.json',dict(receipt,teacher_sha256='wrong'))
    if fault=='checkpoint': Path(b['members'][0]['path']).write_bytes(b'changed')
    if fault in ('runtime','cleanup'):
        q=p.read(b['qualification']['path'])
        q.update({'upper_seconds':31201} if fault=='runtime' else {'descendants_exited':False})
        b['qualification']=asset('bad-qualification.json',q)
    if fault=='weight': r['weights']['primary']=.4
    monkeypatch.setattr(p,'registration',lambda stage:r)
    inputs=tmp_path/'inputs.json'; p.dump(inputs,b)
    if fault=='none':
        assert p.bind('placeholder',inputs,tmp_path/'bound.json')['status']=='BOUND'
        assert p.admit('placeholder',tmp_path/'bound.json')['status']=='ADMITTED'
    else:
        with pytest.raises(ValueError): p.admit('placeholder',inputs)


def test_package_registration_notebook_and_isolated_runtime(tmp_path):
    import subprocess,sys,zipfile
    stage=tmp_path/'stage'
    p.register(stage,mode='all-data',seeds=[42])
    p.prepare(stage)
    assert p.validate(stage)['status']=='PASS_PREPARED'
    assert p.admit(stage)['status']=='HOLD'
    cells=p.read(stage/'release-template.ipynb')['cells']
    compile(''.join(''.join(c['source']) for c in cells),'<notebook>','exec')
    runtime=tmp_path/'runtime'
    with zipfile.ZipFile(stage/'runtime.zip') as z: z.extractall(runtime)
    result=subprocess.run([sys.executable,'-B',str(runtime/'tools/vnext_r7_pl_v1.py'),
        'admit','--stage',str(stage)],cwd=runtime,env=dict(os.environ,PYTHONPATH=str(runtime)+os.pathsep+str(runtime/'tools')),
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='HOLD'
    from pl_readout_fixture_v111 import make_readout
    from tools.vnext_r7_pl_train_v110 import asset
    source=make_readout(tmp_path/'screen')
    code=("import sys;sys.path[:0]="+repr([str(runtime),str(runtime/'tools')])+";"
          "from tools import vnext_r7_pl_go_v111 as g;"
          "assert g.validate("+repr(asset(source))+",'global')['registered_go']['GO']")
    isolated=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=runtime,
                            capture_output=True,text=True,timeout=60)
    assert isolated.returncode==0,isolated.stderr
    with pytest.raises(ValueError): p.register(stage,arm='fill')


@pytest.mark.parametrize('fault',['healthy','missing','corrupt','incomplete','sticky','stale'])
def test_r7_adapter_uses_v122_guarded_collector(tmp_path,monkeypatch,fault):
    import shutil
    from tools import exp169_c85r7ins_checks_v122 as checks
    from tools import exp169_release_c85r7codec_deadline_v122 as insurance
    checks.fixture(tmp_path/'recorded',IDS)
    source=tmp_path/'recorded'/insurance.RUN/'pr7'
    actual_identity=p.read(source/'pr7-run/r7-identity.json')
    monkeypatch.setattr(insurance.r7,'verify_bundle',lambda *a:(None,actual_identity))
    monkeypatch.setattr(insurance,'install_adapters',lambda *a:None)
    r=reg();entries=[e for e in r['members'] if e['branch']=='R7']
    spec=dict(path=str(tmp_path/'out/journal'),ids=IDS,identity=p.identity(r,'R7',entries,'fixture'))
    job=dict(output=str(tmp_path/'out'),ids=IDS,bundle='unused',source='unused',device='cpu',branches=[spec])
    def infer(path):
        dest=tmp_path/'out/public/pr7-run/s_r7'
        shutil.copytree(source/'pr7-run/s_r7',dest)
        receipt=dest/'controller/worker0/receipt.json'
        if fault=='missing': receipt.unlink()
        elif fault=='corrupt': receipt.write_bytes(b'{')
        elif fault=='incomplete': p.dump(receipt,{'status':'PASS'})
        elif fault in ('sticky','stale'):
            rec=p.read(receipt)
            if fault=='sticky': rec.update(scored=2,fallback={IDS[0]:'decode failed'})
            else: rec['publication_binding']['predictions']='stale'
            p.dump(receipt,rec)
    if fault=='stale':
        with pytest.raises(ValueError): w.r7(job,infer)
        assert p.collect(spec['path'],IDS,spec['identity'])[1]==3
    else:
        w.r7(job,infer)
        _,failures,_=p.collect(spec['path'],IDS,spec['identity'])
        assert failures == (0 if fault=='healthy' else 1 if fault=='sticky' else 3)
