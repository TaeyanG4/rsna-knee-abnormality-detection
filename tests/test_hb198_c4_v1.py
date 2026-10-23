"""H198 exact build, math, CPU contract and phased gate tests; no forwards."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import pytest
from tools import c85r7_r6w958c4_v1147 as b
from tools import hb198_c4_bridge_v1 as c
from tools import hb198_c4_gold58_v1 as worker
from tools import strict_variant_gate_c4_v1 as gate


def fixture():
    terms=dict(decode=.5,render_crop=.5,all_model_branches=12,rank_serialize=.1)
    return dict(candidate=b.KEY,slug=b.SLUG,version=b.VERSION,parent_kernel_sha256=b.PARENT_SHA,
        kernel_sha256='a'*64,fast=False,exact_diff=True,retained15=True,mixed_renderer=True,formula='H189_F_Q4',builder=b.BUILDER,reviewer='fixture-independent-reviewer',
        source_pins_current=True,review_current=True,renderer_real_pass=True,strict5_pass=True,cpu_regressions_pass=True,oof_q32_pass=True,inherited_numeric_pass=True,
        build_admitted=True,launch_review_current=True,workload=290,model_count=5,studies=58,corpus_authenticated=True,resource_claim='fixture-claimed',executor='B',logical_device='cuda:0',
        memory=dict(desktop_bytes=2*2**30,peak_reserved_bytes=3*2**30,total_bytes=16*2**30),guarded_worker=True,
        gold_state='INFERENCE_COMPLETE',gold_shape=[5,58,12],gold_dtype='float32',gold_finite=True,gold_parent_noop=True,gold_same_roster=True,gold_failures=0,
        gold_review_pass=True,joint_harm_reviewed=True,submission_intent=True,remote_assets_verified=True,live_quota_verified=True,
        remote_version=1,minimum_review_pass=True,full_visible=True,n32=True,cuda_all_entrypoints=True,child_logs_pass=True,codecs_pass=True,fallbacks_pass=True,
        all_ids_12_finite=True,duplicate_clear=True,description_accurate=True,canonical_submit_integration=True,secret_scan_v102_pass=True,
        runtime=dict(per_study_p95=terms,cohort_studies=1300,cohort_source_verified=True,full_path_measured=True,startup_fixed_proven=True,
                     fixed_startup_seconds=100,projected_seconds=100+1300*sum(terms.values())))


@pytest.mark.parametrize('phase',gate.PHASES)
def test_actual_phase_gate_positive_fixture_not_production(phase):
    rec=gate.core(phase,fixture(),fixture_only=True)
    assert rec['status']==phase and rec['fixture_only'] is True and rec['qualified_for_submission'] is False


@pytest.mark.parametrize('field,value',[
    ('candidate','other'),('slug','taeyangg4/other'),('version','1.1.39-r6w958q'),('parent_kernel_sha256','0'*64),('fast',True),
    ('exact_diff',False),('retained15',False),('mixed_renderer',False),('formula','Q7'),('builder','same'),('reviewer',b.BUILDER),
    ('source_pins_current',False),('review_current',False),('renderer_real_pass',False),('strict5_pass',False),('cpu_regressions_pass',False),
    ('oof_q32_pass',False),('inherited_numeric_pass',False)])
def test_actual_build_gate_defects(field,value):
    e=fixture();e[field]=value
    with pytest.raises(ValueError):gate.core('BUILD_PASS',e,fixture_only=True)


@pytest.mark.parametrize('field,value',[
    ('build_admitted',False),('launch_review_current',False),('workload',289),('model_count',4),('studies',57),('corpus_authenticated',False),
    ('resource_claim',None),('executor','builder'),('logical_device','cuda:1'),('guarded_worker',False)])
def test_actual_cache_gate_defects(field,value):
    e=fixture();e[field]=value
    with pytest.raises(ValueError):gate.core('CACHE_LAUNCH_PASS',e,fixture_only=True)


@pytest.mark.parametrize('field,value',[
    ('gold_state','PARTIAL'),('gold_shape',[4,58,12]),('gold_dtype','float64'),('gold_finite',False),('gold_parent_noop',False),('gold_same_roster',False),
    ('gold_failures',1),('gold_review_pass',False),('joint_harm_reviewed',False),('submission_intent',False),('remote_assets_verified',False),('live_quota_verified',False)])
def test_actual_qualification_gate_defects(field,value):
    e=fixture();e[field]=value
    with pytest.raises(ValueError):gate.core('QUALIFICATION_LAUNCH_PASS',e,fixture_only=True)


@pytest.mark.parametrize('field,value',[
    ('remote_version',0),('minimum_review_pass',False),('full_visible',False),('n32',False),('cuda_all_entrypoints',False),('child_logs_pass',False),
    ('codecs_pass',False),('fallbacks_pass',False),('all_ids_12_finite',False),('duplicate_clear',False),('description_accurate',False),
    ('canonical_submit_integration',False),('secret_scan_v102_pass',False)])
def test_actual_submission_gate_defects(field,value):
    e=fixture();e[field]=value
    with pytest.raises(ValueError):gate.core('SUBMISSION_PASS',e,fixture_only=True)


@pytest.mark.parametrize('damage',('missing','nan','startup','cohort','projection','ninehours'))
def test_runtime_full_cost_fail_closed(damage):
    e=fixture();r=e['runtime']
    if damage=='missing':del r['per_study_p95']['render_crop']
    if damage=='nan':r['per_study_p95']['all_model_branches']=float('nan')
    if damage=='startup':r['startup_fixed_proven']=False
    if damage=='cohort':r['cohort_studies']=3
    if damage=='projection':r['projected_seconds']-=1
    if damage=='ninehours':
        r['per_study_p95']['all_model_branches']=25;r['projected_seconds']=100+1300*sum(r['per_study_p95'].values())
    with pytest.raises(ValueError):gate.core('SUBMISSION_PASS',e,fixture_only=True)


@pytest.mark.parametrize('damage',('missing','nan','peak','headroom','negative'))
def test_measured_memory_gate(damage):
    e=fixture();m=e['memory']
    if damage=='missing':del m['peak_reserved_bytes']
    if damage=='nan':m['peak_reserved_bytes']=float('nan')
    if damage=='peak':m['peak_reserved_bytes']=12*2**30+1
    if damage=='headroom':m['desktop_bytes']=12*2**30
    if damage=='negative':m['desktop_bytes']=-1
    with pytest.raises(ValueError):gate.core('CACHE_LAUNCH_PASS',e,fixture_only=True)


def test_actual_production_requires_real_independent_receipt(tmp_path):
    data=dict(schema='h198-independent-review/v1',verdict='PASS',builder=b.BUILDER,reviewer='fixture-reviewer',kernel_sha256='a'*64,candidate=b.KEY)
    p=tmp_path/'review.json';p.write_text(json.dumps(data));ref=dict(path=p.name,sha256=b.sha(p))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):gate.review(ref,'a'*64,tmp_path)
    p.write_text('{}')
    with pytest.raises(ValueError,match='stale'):gate.digest_file(ref,tmp_path)


def test_actual_build_exact_every_member_and_notebook():
    nb,meta,files,blob=b.payload();assert b.read(b.OUT/'kernel/release.ipynb')==nb
    assert b.read(b.OUT/'kernel/kernel-metadata.json')==meta
    parent,original,_,_=b.parent()
    assert set(files)==set(original) and len(files)==31
    changed={n for n in files if files[n]!=original[n]}
    assert changed=={'a172-cell.py.txt','exp169_partner_bundle_release.py','exp169_release_c85.py','exp169_release_c85r7.py','exp169_release_c85r7codec.py','exp169_release_p956.py','exp169_release_v2.py','exp169_release_v5q.py','q-a5q-manifest.json'}
    assert nb['cells'][0]['source']==['FAST = False\n'] and nb['cells'][1]==parent['cells'][1]
    assert meta['dataset_sources']==b.read(b.BASE/'kernel/kernel-metadata.json')['dataset_sources']+[b.DATASET]
    a,z=(json.loads(x['q-a5q-manifest.json']) for x in (original,files))
    assert a['entries'][:15]==z['entries'][:15]
    assert all(x['source_sha256']==x['sha256'] and x['crop_policy']=='mixed' for x in z['entries'][15:])
    for n in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
        assert z['files']['code/'+n]==dict(bytes=len(files[n]),sha256=b.digest(files[n]))
        assert z['source_code'][n]==b.digest(files[n])


def test_actual_asset_hook_and_timing_adapter_compiles_unchanged():
    files=b.unpack(b.read(b.OUT/'kernel/release.ipynb'))[0]
    scope=dict(require=b.need,hashlib=hashlib,A172_CELL_SHA=b.digest(files['a172-cell.py.txt']))
    v5=c.functions(files['exp169_release_v5q.py'],['_replace_once','a172_runtime_source','a172_function_source'],scope)
    c85=c.functions(files['exp169_release_c85.py'],['a172_source'],dict(v5q=types.SimpleNamespace(**v5),need=b.need))
    generated=c85['a172_source'](files['a172-cell.py.txt'].decode(),Path('/kaggle/working/c85-run/a172'))
    compile(generated,'actual-c4-a172','exec')
    assert generated.count('_c85.overlay_c4(_partner)')==1
    assert generated.index('_c85.overlay_c4(_partner)')<generated.index('import exp169_release_v2 as _e169')
    assert b.Q_SHA in generated
    assert 'install_a172_runtime' not in generated
    partner=ast.parse(files['exp169_partner_bundle_release.py'])
    source=ast.get_source_segment(files['exp169_partner_bundle_release.py'].decode(),next(x for x in partner.body if isinstance(x,ast.FunctionDef) and x.name=='partner_scores'))
    q=c.functions(files['exp169_release_c85.py'],['q_runtime_source'])['q_runtime_source'](v5['a172_function_source']('partner_scores',source))
    compile(q,'actual-c4-q','exec')
    assert 'i, rad, img, triplet, flags = job' in q and 'c4[odd]' in q


def raw_oof(seed=101):
    rng=np.random.default_rng(seed)
    return {f:{k:rng.random((11,12),dtype=np.float32) for k in ('R42','R43','DINO','CNX','C4')} for f in range(5)}


def test_noop_c4_equals_c_and_fp32_pooled_order():
    from tools.exp169_release_v2 import rank
    raw=raw_oof()
    for row in raw.values():row['C4']=row['CNX'].copy()
    q0,q4=c.oof_values(raw);assert np.array_equal(q0,q4)
    rr=np.concatenate([(raw[f]['R42']+raw[f]['R43'])/np.float32(2) for f in range(5)])
    expected=rank((rank(rr)+rank(np.concatenate([raw[f]['DINO'] for f in range(5)]))+rank(np.concatenate([raw[f]['CNX'] for f in range(5)])))/3)
    assert np.array_equal(q0,expected) and rr.dtype==np.float32


@pytest.mark.parametrize('damage',('missingfold','extrafold','missingmember','nan','range','dtype','shape'))
def test_oof_validation(damage):
    raw=raw_oof()
    if damage=='missingfold':del raw[4]
    if damage=='extrafold':raw[5]=raw[4]
    if damage=='missingmember':del raw[0]['C4']
    if damage=='nan':raw[0]['C4'][0,0]=np.nan
    if damage=='range':raw[0]['C4'][0,0]=1.1
    if damage=='dtype':raw[0]['C4']=raw[0]['C4'].astype(float)
    if damage=='shape':raw[0]['C4']=raw[0]['C4'][:10]
    with pytest.raises(ValueError):c.oof_values(raw)


def test_full_roster_tie_permutation_and_parent_fp32_means():
    from tools import exp169_release_v2 as q
    rng=np.random.default_rng(19);leaves={k:rng.integers(0,7,(n,58,12)).astype(np.float32)/7 for k,n in [('l2',10),('dino',5),('convnext',5)]}
    actual,family=q.aggregate(leaves)
    rr=(leaves['l2'][:5]+leaves['l2'][5:])/np.float32(2)
    assert np.array_equal(family['l2'],q.rank(rr.mean(0,dtype=np.float32)))
    order=rng.permutation(58);permuted=q.aggregate({k:v[:,order] for k,v in leaves.items()})[0]
    assert np.array_equal(permuted,actual[order])
    assert np.array_equal(q.rank(np.full((58,12),.5)),np.full((58,12),.5))


@pytest.mark.parametrize('ids',(['a','a'],['a'],['a','c'],['a','b','c']))
def test_id_failure_contract(ids):
    with pytest.raises(ValueError):worker.align(ids,np.ones((len(ids),12))*.5,['a','b'])


@pytest.mark.parametrize('kind',('all','Q','R','own'))
def test_frozen_parent_failure_masks(kind):
    from tools import c85r7_r6w958q_v1139 as p
    files=b.parent()[1];ns=p.math_functions(files);rng=np.random.default_rng(27)
    ids=[str(i) for i in range(8)];args=[rng.random((8,12)) for _ in range(6)]
    names=('R','R224','S_R7','EXP143v2','EXP155','Q_A5q')
    failed={n:dict.fromkeys(ids,'test') for n in names} if kind=='all' else {'Q_A5q':{ids[0]:'test'}} if kind=='Q' else {'R':{ids[0]:'test'},'R224':{ids[0]:'test'}} if kind=='R' else {n:{ids[0]:'test'} for n in names[3:]}
    if kind in ('all','R'):
        with pytest.raises(ValueError,match='Raptor-224 failed on more than 3%'):ns['blend'](ids,*args,failures=failed)
        return
    a=ns['blend'](ids,*args,failures=failed)
    assert a.shape==(8,12) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
    if kind=='all':assert np.array_equal(a,np.full((8,12),.5))


def test_q32_registered_thresholds():
    d=np.full((4,12),.001);assert c.q32(d)
    d[:2]=0;assert not c.q32(d)
    d[:]=.001;d[2,11]=-.010001;assert not c.q32(d)
    d[:]=.001;d[:2,[3,5,6,8]]=-.0001;assert not c.q32(d)
    d[:]=.001;d[3,3]=np.nan;assert not c.q32(d)


def test_recorded_real_cpu_evidence():
    p=b.read(c.OUT/'RENDERER-PARITY.json');s=b.read(c.OUT/'CHECKPOINT-STRICT.json');o=b.read(c.OUT/'OOF-PROXY.json')
    assert p['status']=='PASS_CPU_REAL_RENDERER' and len(p['cases'])==5 and all(x['exact'] for x in p['cases'])
    assert s['status']=='PASS_CPU_STRICT_NO_FORWARD' and len(s['models'])==5
    assert p['cuda_initialized'] is s['cuda_initialized'] is False
    assert o['primary']['gate_pass'] and o['independent_auc_checks']==96 and o['independent_auc_max_error']<1e-12
    assert o['historical']['status']=='REVIEW'
