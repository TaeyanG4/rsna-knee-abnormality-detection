"""H174 exact single-stage scope, mount fail-closed and ABBA/review boundaries."""
import ast
import copy
import json
from pathlib import Path
import pytest
from tools import c85r7_r224r6fq_v1130 as b
from tools import c85r7_r6_timing as t
from tools import c85r7_r6_cpu_parity as parity


@pytest.fixture(scope='module')
def payload(): return b.payload()


def test_entire_archive_and_launch_identity(payload):
    nb,meta,files,blob=payload
    oldnb,oldfiles,oldblob,assign=b.parent_builder.parent()
    assert len(files)==31 and set(files)==set(oldfiles)
    assert {k for k in files if files[k]!=oldfiles[k]}=={b.PORT,b.CHAIN,b.WRAPPER}
    assert nb['cells'][:-1]==oldnb['cells'][:-1]
    # Only BLOB/PINS/archive digest change in the final cell.
    last=''.join(nb['cells'][-1]['source'])
    original=''.join(oldnb['cells'][-1]['source'])
    def normalize(source):
        tree=ast.parse(source)
        for n in ast.walk(tree):
            if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('BLOB','PINS') for x in n.targets):
                n.value=ast.Constant('ARCHIVE')
            if isinstance(n,ast.Constant) and n.value in (b.sha(blob),b.sha(oldblob)): n.value='SHA'
        return ast.dump(tree)
    assert normalize(last)==normalize(original)
    for k in files:
        if k not in {b.PORT,b.CHAIN,b.WRAPPER}: assert files[k]==oldfiles[k]
    assert files['exp169_r224.py']==oldfiles['exp169_r224.py']
    for name in (b.PORT,b.CHAIN,b.WRAPPER):
        a,z=ast.parse(oldfiles[name]),ast.parse(files[name])
        imports=lambda tr:[ast.dump(n) for n in tr.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        assert imports(a)==imports(z)
        funcs=lambda tr:{n.name:ast.dump(n) for n in tr.body if isinstance(n,ast.FunctionDef)}
        before,after=funcs(a),funcs(z)
        changed={k for k in before if before[k]!=after[k]}
        assert changed == ({'run_public'} if name==b.WRAPPER else {'preflight','run_public'} if name==b.CHAIN else set())
        # Changed functions differ only at the three bundle-mount expressions.
        oldtext=oldfiles[name].decode().replace('v5q._mount(r7.DATASET_SLUG, "manifest.json")','r7.resolve_bundle()')
        normalized=funcs(ast.parse(oldtext))
        assert all(normalized[k]==after[k] for k in before)
    oldmeta=b.old.read(b.BASE/'kernel/kernel-metadata.json')
    assert {k:v for k,v in meta.items() if k not in ('id','title','dataset_sources')}=={k:v for k,v in oldmeta.items() if k not in ('id','title','dataset_sources')}
    assert meta['id']==b.SLUG and meta['title']=='knee-lb-r224r6fq-v1'
    assert [b.PRIVATE if x=='heliosli/rsna-knee-compact-r7cnxfsss' else x for x in oldmeta['dataset_sources']]==meta['dataset_sources']


def resolver():
    ns=dict(Path=Path,DATASET_SLUG=b.PRIVATE,PUBLIC_DATASET_SLUG=b.PUBLIC,require=b.need)
    exec(b.RESOLVER,ns)
    return ns['resolve_bundle']


@pytest.mark.parametrize('slug',[b.PRIVATE,b.PUBLIC])
@pytest.mark.parametrize('layout',['flat','datasets','versioned'])
def test_mount_sources(tmp_path,slug,layout):
    p=tmp_path/slug.split('/')[-1] if layout=='flat' else tmp_path/'datasets'/slug
    if layout=='versioned': p=p/'1'
    p.mkdir(parents=True);(p/'manifest.json').write_text('{}')
    assert resolver()(tmp_path)==p


def test_mount_precedence_and_no_invalid_preferred_fallback(tmp_path):
    p=tmp_path/b.PRIVATE.split('/')[-1];q=tmp_path/b.PUBLIC.split('/')[-1]
    for path in (p,q): path.mkdir();(path/'manifest.json').write_text('{}')
    assert resolver()(tmp_path)==p
    with pytest.raises(ValueError): b.port_module().verify_bundle(resolver()(tmp_path))


def test_ambiguous_and_absent(tmp_path):
    with pytest.raises(ValueError): resolver()(tmp_path)
    for n in ('1','2'):
        p=tmp_path/'datasets'/b.PUBLIC/n;p.mkdir(parents=True);(p/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='ambiguous'): resolver()(tmp_path)


def test_public_sources_args_and_bundle():
    source=b.public_sources();files=b.parent_builder.parent()[1]
    assert len(source)==7 and all(v==files['exp169_r7_src/'+k] for k,v in source.items())
    port=b.port_module();m,_=port.verify_bundle(b.BUNDLE,source_dir=b.WORK/'runtime/exp169_r7_src')
    assert parity.public_args(m,b.BUNDLE,t.d.ANATOMY)==port.worker_args(m,b.BUNDLE,t.d.ANATOMY)
    assert '--flip-tta' in port.worker_args(m,b.BUNDLE,t.d.ANATOMY)
    assert len(m['models'])==4 and b.sha((b.BUNDLE/'manifest.json').read_bytes())==b.MANIFEST_SHA
    with pytest.raises(ValueError): port.verify_bundle(b.BUNDLE,expected_manifest_sha256='0'*64)


def arms(seconds=100):
    return {a:dict(seconds=seconds,rows=[dict(study=str(i),windows=60,batch_windows=192,gpu_seconds=1.) for i in range(32)]) for a in t.ORDER}


def test_projection_identity_signed_and_gate():
    a=arms();c=dict(r7_hidden_seconds=3600,conservative_plain_seconds=100)
    r=t.projection(a,c);assert r['central_hours']==6.7 and r['gate_hours']==6.73 and r['runtime_pass']
    for k in ('r6-1','r6-2'): a[k]['seconds']=50
    r=t.projection(a,c);assert r['central_hours']==6.2 and r['gate_hours']==6.73
    a['r6-2']['seconds']=400
    assert not t.projection(a,c)['runtime_pass']


@pytest.mark.parametrize('bad',['missing','duplicate','nan','zero','windows'])
def test_projection_reject(bad):
    a=arms()
    if bad=='missing': del a['r6-2']
    if bad=='duplicate': a['r7-1']['rows'][0]['study']='1'
    if bad=='nan': a['r6-1']['seconds']=float('nan')
    if bad=='zero': a['r6-1']['seconds']=0
    if bad=='windows': a['r6-1']['rows'][0]['windows']=59
    with pytest.raises(ValueError): t.projection(a,dict(r7_hidden_seconds=3600,conservative_plain_seconds=100))


def test_review_bypass_and_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'WORK',tmp_path)
    for name in ('validate','cpu-replay'): (tmp_path/(name+'.json')).write_text('{}')
    reg=dict(binding_sha256='binding',binding=dict(kernel_sha256='kernel'))
    good=dict(verdict='PASS',builder='astra-b1',reviewer='independent',execution_authority='B',binding_sha256='binding',
              kernel_sha256='kernel',validation_sha256=t.sha(tmp_path/'validate.json'),replay_sha256=t.sha(tmp_path/'cpu-replay.json'),
              scope='H-B-174 build and N32 execution')
    t.review_check(good,reg)
    for key,value in [('reviewer','astra-b1'),('verdict','HOLD'),('kernel_sha256','bad'),('validation_sha256','bad'),('scope','other')]:
        r=dict(good,**{key:value})
        with pytest.raises(ValueError): t.review_check(r,reg)


def test_cpu_dry_run_no_review_never_admits(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'WORK',tmp_path)
    monkeypatch.setattr(t,'current',lambda:dict(binding_sha256='x'))
    for n in ('validate','cpu-replay'): (tmp_path/(n+'.json')).write_text(json.dumps(dict(binding_sha256='x')))
    with pytest.raises(ValueError,match='independent review missing'):t.admit(None,True)
    assert not (tmp_path/'admission.json').exists()


def test_private_worker_no_controller(monkeypatch):
    monkeypatch.delenv('HB164_CONTROLLER_PID',raising=False)
    with pytest.raises(ValueError,match='supervised'):t.worker()


def prepared_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(t,'ROOT',tmp_path)
    monkeypatch.setattr(t,'WORK',tmp_path/'timing')
    monkeypatch.setattr(b,'WORK',tmp_path)
    (tmp_path/'timing').mkdir();(tmp_path/'cpu-parity').mkdir()
    result=tmp_path/'cpu-parity/result.json';result.write_text('{}')
    prediction=tmp_path/'cpu-parity/predictions.csv';prediction.write_text('original')
    identities={}
    for kind in ('r7','r6'):
        path=t.WORK/'cpu-runtime'/kind/'instrumented.py'
        path.parent.mkdir(parents=True);path.write_text('original')
        identities[kind]=dict(source_sha256=b.sha(b'original'))
    t.io.atomic_json(t.WORK/'prepare.json',dict(binding_sha256='binding',identities=identities))
    t.io.atomic_json(t.WORK/'validate.json',dict(binding_sha256='binding',prepare_sha256=t.sha(t.WORK/'prepare.json'),
        evidence={'cpu-parity/predictions.csv':t.sha(prediction)},parity_sha256=t.sha(result)))
    return prediction


def test_real_cpu_evidence_drift(tmp_path,monkeypatch):
    prediction=prepared_evidence(tmp_path,monkeypatch)
    t.cpu_evidence_check()
    prediction.write_text('modified after review')
    with pytest.raises(ValueError,match='evidence drift'):t.cpu_evidence_check()


@pytest.mark.parametrize('kind',['prepare','instrumented'])
def test_prepared_source_identity_drift(tmp_path,monkeypatch,kind):
    prepared_evidence(tmp_path,monkeypatch)
    t.cpu_evidence_check()
    path=t.WORK/'prepare.json' if kind=='prepare' else t.WORK/'cpu-runtime/r6/instrumented.py'
    path.write_text('changed after review')
    with pytest.raises(ValueError,match='prepared'):t.cpu_evidence_check()


@pytest.mark.parametrize('drift',['pixel','added','missing','metadata','junction'])
def test_worker_data_identity_drift(tmp_path,monkeypatch,drift):
    monkeypatch.setattr(t,'WORK',tmp_path)
    monkeypatch.setattr(t.d,'R224_DATA',tmp_path)
    ids=['1'];rows=[dict(StudyInstanceUID='1',SeriesInstanceUID='2')]
    monkeypatch.setattr(t.d.r7,'cohort',lambda:(ids,rows))
    folder=tmp_path/'data/train_series/1/2';folder.mkdir(parents=True)
    path=folder/'1.dcm';path.write_bytes(b'original pixels')
    metadata=tmp_path/'data/train_series.csv';metadata.write_bytes(t.d.series_bytes(rows))
    t.io.atomic_json(tmp_path/'prepare.json',dict(state='CPU_PREPARED',studies=ids,
        real_study_cpu_replay=dict(exact_window_parity=True),files={'train_series/1/2/1.dcm':t.sha(path)}))
    assert t.data_identity_check()['files']==1
    if drift=='pixel':path.write_bytes(b'modified pixels')
    elif drift=='added':(folder/'2.dcm').write_bytes(b'extra')
    elif drift=='missing':path.unlink()
    elif drift=='metadata':metadata.write_bytes(b'drift')
    elif drift=='junction':
        other=tmp_path/'other';(other/'data/train_series/1/2').mkdir(parents=True)
        monkeypatch.setattr(t.d,'R224_DATA',other)
    with pytest.raises(ValueError,match='N32'):t.data_identity_check()


def test_execution_and_replay_recheck_data_and_source():
    import inspect
    for func in (t.admit,t.admitted,t.evaluate):
        tree=ast.parse(inspect.getsource(func))
        calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        assert {'data_identity_check','cpu_evidence_check'}<=calls
    for func in (t.run,t.worker):
        tree=ast.parse(inspect.getsource(func))
        assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='admitted'
                   for n in ast.walk(tree))


def test_registered_score_floor():
    assert b.registration()['keep_score_min']==.956
    assert t.RULES['limit_hours']==9 and t.RULES['order']==list(t.ORDER)
    assert t.keep_decision(.956,dict(gate_hours=9),True)['decision']=='KEEP'
    assert t.keep_decision(.95599,dict(gate_hours=9),True)['decision']=='HOLD'
    assert t.keep_decision(.957,dict(gate_hours=9.001),True)['decision']=='HOLD'
    assert t.keep_decision(.957,dict(gate_hours=6.7),False)['decision']=='HOLD'
    with pytest.raises(ValueError):t.keep_decision(float('nan'),dict(gate_hours=6.7),True)


def test_real_historical_output_validator_rejects_corruption():
    reg=t.io.read_json(t.HISTORY/'registration.json');ids=reg['binding']['ids']
    value=t.io.read_json(t.HISTORY/'flip-1/worker-status.json');sup=t.io.read_json(t.HISTORY/'flip-1/supervision.json')
    t.check_arm(value,sup,'r7',ids)
    for kind in ('nan','duplicate','partial','model','telemetry','cleanup'):
        a,s=copy.deepcopy(value),copy.deepcopy(sup)
        if kind=='nan': a['predictions'][0][0]=float('nan')
        if kind=='duplicate': a['rows'][0]['study']=a['rows'][1]['study']
        if kind=='partial': a['done']=31
        if kind=='model': a['runtime']['models'][0]='wrong.pt'
        if kind=='telemetry': s['telemetry']['complete']=False
        if kind=='cleanup': s['cleanup']=['still alive 123']
        with pytest.raises(ValueError):t.check_arm(a,s,'r7',ids)
