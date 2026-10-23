"""H197 pure cached arithmetic, exact production reader and gate evidence parsers.

Fixtures are explicit local synthetic evidence, never production admissions.
No model is constructed, loaded or forwarded; no notebook or worker is executed.
"""
import ast
import copy
from fractions import Fraction
import io
import json
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import pytest
from tools import c85r7_r6w958inf3_v1146 as b
from tools import strict_variant_gate_inf3_v1 as g


@pytest.fixture(scope='module')
def runtime():return b.runtime(b.payload()[2])


def leaves(n=71):
    rng=np.random.default_rng(197)
    return {f:rng.integers(1,49,(10 if f=='l2' else 5,n,12)).astype(np.float32)/50 for f in b.FAMILIES}


def independent(raw,cost):
    family=[]
    for f in b.FAMILIES:
        a=np.stack([np.array([[float(Fraction(float(q))/(Fraction(float(w))-(Fraction(float(w))-1)*Fraction(float(q))))
                              for q,w in zip(row,ww)] for row in member]) for member,ww in zip(raw[f],cost[f])])
        if f=='l2':a=(a[:5]+a[5:])/2
        family.append(a.mean(0))
    return pd.DataFrame(sum(family)/3).rank(method='average',pct=True).to_numpy()


def test_fraction_oracle_all20_checkpoint_costs_and_endpoints(runtime):
    raw=leaves(9)
    for a in raw.values():a[:,0]=0;a[:,1]=1
    actual=runtime['inf3_aggregate'](raw,np)
    np.testing.assert_array_equal(actual,independent(raw,b.costs()))
    assert np.all(actual[0]==1/9) and np.all(actual[1]==1)


@pytest.mark.parametrize('n',[1,2,58,1300])
def test_constant_average_rank_n(runtime,n):
    raw={f:np.full((10 if f=='l2' else 5,n,12),.25,dtype=np.float32) for f in b.FAMILIES}
    np.testing.assert_array_equal(runtime['inf3_aggregate'](raw,np),np.full((n,12),(n+1)/(2*n)))


def test_identity_cost_is_probability_pool_not_parent_q0(runtime):
    ns=dict(runtime);exec(compile(ast.parse(b.POOL_SOURCE),'exact-pool','exec'),ns)
    ns['INF3_COSTS']={f:np.ones((10 if f=='l2' else 5,12)) for f in b.FAMILIES}
    raw=leaves();out=ns['inf3_aggregate'](raw,np)
    r=raw['l2'].astype(float);r=((r[:5]+r[5:])/2).mean(0)
    expected=pd.DataFrame((r+raw['dino'].astype(float).mean(0)+raw['convnext'].astype(float).mean(0))/3).rank(pct=True).to_numpy()
    np.testing.assert_array_equal(out,expected)
    q0=runtime['a172'].aggregate(raw)[0];assert np.any(out!=q0)


def test_full_roster_permutation_ties_and_not_chunk_rank(runtime):
    raw=leaves(97);out=runtime['inf3_aggregate'](raw,np);ix=np.random.default_rng(198).permutation(97)
    np.testing.assert_array_equal(runtime['inf3_aggregate']({f:a[:,ix] for f,a in raw.items()},np),out[ix])
    chunks=np.concatenate([runtime['inf3_aggregate']({f:a[:,s] for f,a in raw.items()},np) for s in (slice(0,48),slice(48,None))])
    assert np.any(out!=chunks)


def test_transform_before_mean_and_no_intermediate_rank(runtime):
    raw=leaves();cost=b.costs();out=runtime['inf3_aggregate'](raw,np);means=[];wrong=[]
    for f in b.FAMILIES:
        a=raw[f].astype(float);w=np.asarray(cost[f]);trans=a/(w[:,None,:]-(w[:,None,:]-1)*a)
        if f=='l2':trans=(trans[:5]+trans[5:])/2
        means.append(trans.mean(0));mean=a.mean(0);ww=w.mean(0)
        wrong.append(mean/(ww-(ww-1)*mean))
    assert np.any(out!=pd.DataFrame(sum(wrong)/3).rank(pct=True).to_numpy())
    familyrank=[pd.DataFrame(v).rank(pct=True).to_numpy() for v in means]
    assert np.any(out!=pd.DataFrame(sum(familyrank)/3).rank(pct=True).to_numpy())
    np.testing.assert_array_equal(out,pd.DataFrame(sum(means)/3).rank(pct=True).to_numpy())


@pytest.mark.parametrize('fault',['family','extra','count','shape','dtype','empty','nan','inf','range','cohort','costshape','costnan','costlow'])
def test_invalid_leaf_or_cost_fails(runtime,fault):
    raw=leaves();ns=dict(runtime);exec(compile(ast.parse(b.POOL_SOURCE),'exact-pool','exec'),ns);ns['INF3_COSTS']=copy.deepcopy(b.costs())
    if fault=='family':raw.pop('dino')
    if fault=='extra':raw['small']=raw['dino']
    if fault=='count':raw['l2']=raw['l2'][:5]
    if fault=='shape':raw['dino']=raw['dino'][:,:,:11]
    if fault=='dtype':raw['dino']=raw['dino'].astype(float)
    if fault=='empty':raw['dino']=raw['dino'][:,:0]
    if fault=='cohort':raw['dino']=raw['dino'][:,:1]
    if fault in ('nan','inf','range'):raw['dino'][0,0,0]={'nan':np.nan,'inf':np.inf,'range':1.2}[fault]
    if fault=='costshape':ns['INF3_COSTS']['dino']=ns['INF3_COSTS']['dino'][:4]
    if fault=='costnan':ns['INF3_COSTS']['dino'][0][0]=np.nan
    if fault=='costlow':ns['INF3_COSTS']['dino'][0][0]=.5
    with pytest.raises(ValueError):ns['inf3_aggregate'](raw,np)


def test_exact_targets_cost_member_roster_and_scope(runtime):
    proof=b.authority();manifest=json.loads(b.parent()[1]['q-a5q-manifest.json'])
    assert [m['identity'] for m in proof['members']]==manifest['entries']
    expected=[(f,s,k) for f in b.FAMILIES for s in ((42,43) if f=='l2' else (42,)) for k in range(5)]
    assert [(e['family'],e['seed'],e['fold']) for e in manifest['entries']]==expected
    assert proof['targets']==list(runtime['p'].LABELS)
    assert runtime['INF3_COSTS']==b.costs()
    nb,meta,files,_=b.payload();b.scope(nb,meta)
    original=b.parent()[1];assert sum(files[n]==v for n,v in original.items())==28
    assert files['q-a5q-manifest.json']==original['q-a5q-manifest.json']
    assert files['exp169_r224.py']==original['exp169_r224.py']
    assert files['exp169_partner_bundle_release.py']==original['exp169_partner_bundle_release.py']
    assert meta['dataset_sources']==b.read(b.BASE/'kernel/kernel-metadata.json')['dataset_sources']
    assert nb['cells'][0]['source']==['FAST = False\n'] and nb['cells'][1]==b.parent()[0]['cells'][1]


@pytest.mark.parametrize('kind',['fast','loader','metadata','payload'])
def test_scope_rejects_extra_mutations(kind):
    nb,meta,_,_=b.payload()
    if kind=='fast':nb['cells'][0]['source']=['FAST = True\n']
    if kind=='loader':nb['cells'][2]['source'].append('\nprint("extra")\n')
    if kind=='metadata':meta['enable_internet']=True
    if kind=='payload':nb['cells'][1]['source'].append('\n# extra\n')
    with pytest.raises(ValueError):b.scope(nb,meta)


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b.encoded(obj));return path


def journal_fixture(tmp_path,runtime,missing=None,invalid=False,raw_mode=False,permuted=False):
    ids=[f's{i}' for i in range(7)];raw=leaves(7);m=json.loads(b.parent()[1]['q-a5q-manifest.json'])
    put(tmp_path/'q-a5q-manifest.json',m)
    # Exact serialized parent manifest is pinned; do not reformat its bytes.
    (tmp_path/'q-a5q-manifest.json').write_bytes(b.parent()[1]['q-a5q-manifest.json'])
    trio=tmp_path/'exp169-trio';put(trio/'ids.json',ids);put(trio/'journal-identity.json',m['anchor_binding'])
    lines=[dict(uid=u,member=member,values=[.3]*12) for u in ids for member in ('EXP143v2','EXP155')]
    (trio/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in lines),encoding='utf-8')
    if raw_mode:
        values={f:a.copy() for f,a in raw.items()};order=list(reversed(range(7))) if permuted else list(range(7))
        if invalid:values['dino'][0,2,1]=np.nan
        np.savez(tmp_path/'partner-raw.npz',ids=np.asarray(ids)[order],**{f:a[:,order] for f,a in values.items()})
    elif missing!='all':
        qdir=tmp_path/'q-journal';put(qdir/'ids.json',ids);put(qdir/'journal-identity.json',{'manifest_sha256':b.Q_SHA})
        rows=[]
        for j,(f,offset) in enumerate([(f,k) for f in b.FAMILIES for k in range(10 if f=='l2' else 5)]):
            for i,uid in enumerate(ids):
                if missing=='one' and j==0 and i==2:continue
                if invalid and j==10 and i==2:rows.append(dict(uid=uid,member=str(j),error='fixture invalid leaf'));continue
                rows.append(dict(uid=uid,member=str(j),values=raw[f][offset,i].tolist()))
        (qdir/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
    return ids,raw


@pytest.mark.parametrize('case',['healthy','missing','invalid','all_missing','raw','raw_invalid','raw_permuted'])
def test_actual_extracted_authenticated_reader_failure_and_roster(runtime,tmp_path,case):
    ids,raw=journal_fixture(tmp_path,runtime,missing='all' if case=='all_missing' else 'one' if case=='missing' else None,
                            invalid=case in ('invalid','raw_invalid'),raw_mode=case.startswith('raw'),permuted=case=='raw_permuted')
    got=runtime['authenticated_a172'](tmp_path);assert got[0]==ids;fail=got[-1]['failures']['Q_A5q']
    if case=='all_missing':assert len(fail)==7;assert np.all(got[3]==.5);return
    good=np.ones(7,bool)
    if case in ('missing','invalid','raw_invalid'):good[2]=False;assert set(fail)=={ids[2]};assert np.all(got[3][2]==.5)
    else:assert fail=={}
    expected=runtime['inf3_aggregate']({f:a[:,good] for f,a in raw.items()},np)
    np.testing.assert_array_equal(got[3][good],expected)


def test_legacy_telemetry_preserves_q0_and_saved_csv(runtime,tmp_path):
    ids,raw=journal_fixture(tmp_path,runtime);q0,fam=runtime['a172'].aggregate(raw)
    put(tmp_path/'exp169-executed-candidate.json',dict(bundle_rank_sha256=runtime['array_sha'](q0),family_rank_sha256={k:runtime['array_sha'](v) for k,v in fam.items()},status='PASS_BUNDLE',studies=7))
    own=np.full((7,12),.3);(tmp_path/'exp169-trio/submission.csv').write_bytes(runtime['p'].csv_bytes(ids,own))
    saved=runtime['a172'].final_values(own,q0,'A172');(tmp_path/'a172-submission.csv').write_bytes(runtime['p'].csv_bytes(ids,saved))
    got=runtime['authenticated_a172'](tmp_path);checks=got[-1]['telemetry']
    assert checks['bundle_rank_digest'] is True and checks['family_rank_digests'] is True and checks['saved_replay'] is True
    assert np.any(got[3]!=q0)


def test_actual_candidate_outer_formula_survivors_and_r224_abort():
    from tools import c85r7_r6w958q_v1139 as parent
    ns=parent.math_functions(b.payload()[2]);rng=np.random.default_rng(197);n=1200
    ids=[f'x{i}' for i in range(n)];arrays=[rng.uniform(.01,.49,(n,12)) for _ in range(6)];masks=[np.zeros(n,bool) for _ in range(6)]
    for row,bits in enumerate(itertools.product((False,True),repeat=6)):
        for mask,bit in zip(masks,bits):mask[row]=bit
    names=('R','R224','S_R7','EXP143v2','EXP155','Q_A5q')
    failures={name:dict.fromkeys([u for u,f in zip(ids,m) if f],'fixture') for name,m in zip(names,masks)}
    actual=ns['blend'](ids,*arrays,failures)
    assert actual.tobytes()==parent.oracle(arrays,masks).tobytes()
    np.testing.assert_array_equal(actual,parent.math_functions(b.parent()[1])['blend'](ids,*arrays,failures))
    failures['R224']=dict.fromkeys(ids[:37],'fixture')
    with pytest.raises(ValueError,match='more than 3%'):ns['blend'](ids,*arrays,failures)


def test_actual_frozen_cache_and_stage():
    assert b.validate()['status']=='PASS_CPU_VALIDATE'
    r=b.replay();assert r['exact_q3_bytes'] and r['parent_noop_float64_and_csv_exact'] and not r['new_metrics_computed']


def review_fixture(tmp_path,mutate=None,report_verdict='PASS',report_body=True):
    source=tmp_path/'gate-source.py';source.write_text('# synthetic gate parser fixture\n',encoding='utf-8')
    claim=dict(schema='h197-independent-review/v1',candidate=b.KEY,slug=b.SLUG,version=b.VERSION,kernel_sha256='a'*64,
               phase='BUILD_PASS',verdict='PASS',builder=b.BUILDER,reviewer='/fixture/different-reviewer',evidence_sha256='b'*64,
               checked_source_sha256={'gate-source.py':b.sha(source)})
    attested=copy.deepcopy(claim)
    if mutate:mutate(claim)
    report=tmp_path/'REPORT.md';report.write_text('Synthetic parser fixture.\n'+('H197-REVIEW-ATTESTATION: '+json.dumps(attested)+'\n' if report_body else '')+'VERDICT: '+report_verdict+'\n',encoding='utf-8')
    completion=put(tmp_path/'completion.json',dict(state='completed',fixture_only=True))
    receipt=put(tmp_path/'native.json',dict(schema='dispatch-terminal/v1',state='SUCCESS',exit_code=0,agent_id='/fixture/different-reviewer',
        host='Codex app native subagent',report_sha256=b.sha(report),out=str(report.resolve()),completion_source=str(completion.resolve()),completion_evidence_sha256=b.sha(completion)))
    claim.update(report=g.ref(report,tmp_path),native_receipt=g.ref(receipt,tmp_path))
    return g.ref(put(tmp_path/'review.json',claim),tmp_path)


def test_actual_report_parser_positive_production_schema(tmp_path):
    r=review_fixture(tmp_path);out=g.review_payload(r,'a'*64,'BUILD_PASS','b'*64,tmp_path)
    assert out['reviewer']=='/fixture/different-reviewer'  # parser only; no admission


@pytest.mark.parametrize('fault',['FAIL','missing_body','candidate','phase','kernel','evidence','assertion','verdict','reviewer','source'])
def test_report_rejects_fail_or_rebound_sidecar(tmp_path,fault):
    mutations={'candidate':lambda v:v.update(candidate='R6W958-K2-V1'),'phase':lambda v:v.update(phase='SUBMISSION_PASS'),
        'kernel':lambda v:v.update(kernel_sha256='c'*64),'evidence':lambda v:v.update(evidence_sha256='c'*64),
        'assertion':lambda v:v.update(minimum_review_pass=True),'verdict':lambda v:v.update(verdict='FAIL'),
        'reviewer':lambda v:v.update(reviewer=b.BUILDER),'source':lambda v:v['checked_source_sha256'].update({'other.py':'0'*64})}
    r=review_fixture(tmp_path,mutations.get(fault),report_verdict='FAIL' if fault=='FAIL' else 'PASS',report_body=fault!='missing_body')
    with pytest.raises(ValueError):g.review_payload(r,'a'*64,'BUILD_PASS','b'*64,tmp_path)


@pytest.mark.parametrize('fault',['candidate','phase','kernel','evidence','builder','source'])
def test_authentic_report_and_matching_sidecar_still_require_exact_context(tmp_path,fault):
    ref=review_fixture(tmp_path);value=b.read(g.file(ref,tmp_path))
    key={'kernel':'kernel_sha256','evidence':'evidence_sha256'}.get(fault,fault)
    value[key]='wrong-context'
    if fault=='source':
        value.pop('source');value['checked_source_sha256']['gate-source.py']='f'*64
    report=g.file(value['report'],tmp_path)
    attested={k:v for k,v in value.items() if k not in ('report','native_receipt')}
    report.write_text('H197-REVIEW-ATTESTATION: '+json.dumps(attested)+'\nVERDICT: PASS\n',encoding='utf-8')
    receipt_path=g.file(value['native_receipt'],tmp_path);receipt=b.read(receipt_path)
    receipt['report_sha256']=b.sha(report);put(receipt_path,receipt)
    value['report']=g.ref(report,tmp_path);value['native_receipt']=g.ref(receipt_path,tmp_path)
    ref=g.ref(put(tmp_path/'review.json',value),tmp_path)
    with pytest.raises(ValueError):g.review_payload(ref,'a'*64,'BUILD_PASS','b'*64,tmp_path)


def test_production_review_rejects_unarchived_synthetic_completion(tmp_path):
    r=review_fixture(tmp_path)
    with pytest.raises((ValueError,KeyError,OSError)):g.review(r,'a'*64,'BUILD_PASS','b'*64,tmp_path)


def test_actual_archived_prereview_is_authentic_but_not_release_admission():
    receipt=Path(str(b.BRIDGE/'RESULT-REVIEW.md')+'.receipt.json')
    actual=g.native.verify(receipt,b.ROOT)
    assert actual['agent_id']!=b.BUILDER and actual['report_sha256']==b.REVIEW_SHA


def execution_fixture(tmp_path,kind='N32'):
    n=32 if kind=='N32' else 3;ids=[f'fixture-{i}' for i in range(n)];targets=[f'T{i}' for i in range(12)]
    ip=put(tmp_path/'ids.json',ids);csvp=tmp_path/'submission.csv';pd.DataFrame({'StudyInstanceUID':ids,**{t:[.25]*n for t in targets}}).to_csv(csvp,index=False)
    log=tmp_path/'child.log';log.write_text('synthetic completed child log\n')
    e=dict(schema='h197-qualified-execution/v1',state='PASS',kind=kind,candidate=b.KEY,slug=b.SLUG,version=b.VERSION,
        kernel_sha256='a'*64,metadata_sha256='d'*64,fast=False,formula='H189_F_Q3',studies=n,remote_version=1,
        all_model_entrypoints=['own','q','raptor','r224','r6'],cuda_all_entrypoints=True,linux_actual=True,codecs_pass=True,fallbacks_pass=True,all_20_cost_identities=True,
        child_logs={k:g.ref(log,tmp_path) for k in ['own','q','raptor','r224','r6']},child_exit_codes={k:0 for k in ['own','q','raptor','r224','r6']},ids=g.ref(ip,tmp_path),output_csv=g.ref(csvp,tmp_path))
    return e,targets


@pytest.mark.parametrize('kind',['FULL_VISIBLE','N32'])
def test_execution_actual_parser_positive_schema(tmp_path,kind):
    e,targets=execution_fixture(tmp_path,kind);r=g.ref(put(tmp_path/'execution.json',e),tmp_path)
    assert g.execution(r,kind,'a'*64,'d'*64,targets,tmp_path)['studies']==(3 if kind=='FULL_VISIBLE' else 32)


@pytest.mark.parametrize('fault',['fast','candidate','version','kernel','formula','child','missing_log','cuda','codec','targets','nan','missing_id'])
def test_execution_negative_production_schema(tmp_path,fault):
    e,targets=execution_fixture(tmp_path)
    if fault in ('fast','candidate','version','kernel','formula','cuda','codec'):
        key={'kernel':'kernel_sha256','cuda':'cuda_all_entrypoints','codec':'codecs_pass'}.get(fault,fault)
        e[key]=True if fault=='fast' else False if fault in ('cuda','codec') else 'incorrect'
    if fault=='child':e['child_exit_codes']['q']=1
    if fault=='missing_log':e['child_logs'].pop('q')
    if fault=='targets':targets=targets[::-1]
    if fault in ('nan','missing_id'):
        csvp=tmp_path/'submission.csv';d=pd.read_csv(csvp)
        if fault=='nan':d.loc[0,'T0']=np.nan
        else:d=d.iloc[1:]
        d.to_csv(csvp,index=False);e['output_csv']=g.ref(csvp,tmp_path)
    with pytest.raises(ValueError):g.execution(g.ref(put(tmp_path/'execution.json',e),tmp_path),'N32','a'*64,'d'*64,targets,tmp_path)


def runtime_fixture(tmp_path):
    source=put(tmp_path/'cohort.json',dict(hidden_studies=1300,verified=True));log=tmp_path/'timing.log';log.write_text('synthetic raw timings\n')
    t=dict(schema='h197-runtime/v1',kernel_sha256='a'*64,n32_sha256='e'*64,hidden_studies=1300,cohort_source=g.ref(source,tmp_path),
        per_study_seconds={k:[1.]*32 for k in g.LINEAR_TERMS},global_seconds={k:10. for k in g.GLOBAL_TERMS},global_measured_studies=1300,
        all_sequential_phase_costs_included=True,all_inference_retries_tta_included=True,fixed_startup_seconds=30.,startup_fixed_proven=True,
        raw_receipts=[g.ref(log,tmp_path)],projected_seconds=5250.)
    return t


def test_runtime_actual_positive_fullsize_rank_cost(tmp_path):
    t=runtime_fixture(tmp_path);out=g.measured_runtime(g.ref(put(tmp_path/'runtime.json',t),tmp_path),{'sha256':'e'*64},'a'*64,tmp_path)
    assert out['projected_seconds']==5250.


@pytest.mark.parametrize('fault',['missing_decode','missing_pool','missing_ranks','n32_ranks','nonfinite','zero','over9h','incorrect_sum','startup','receipt'])
def test_runtime_refuses_missing_scalable_costs(tmp_path,fault):
    t=runtime_fixture(tmp_path)
    if fault=='missing_decode':t['per_study_seconds'].pop('decode_all')
    if fault=='missing_pool':t['per_study_seconds'].pop('inf3_inverse_pool')
    if fault=='missing_ranks':t['global_seconds'].pop('all_full_cohort_ranks')
    if fault=='n32_ranks':t['global_measured_studies']=32
    if fault=='nonfinite':t['per_study_seconds']['decode_all'][0]=1e309
    if fault=='zero':t['per_study_seconds']['inf3_inverse_pool'][0]=0
    if fault=='over9h':t['fixed_startup_seconds']=33000;t['projected_seconds']=38220
    if fault=='incorrect_sum':t['projected_seconds']=1
    if fault=='startup':t['startup_fixed_proven']=False
    if fault=='receipt':t['raw_receipts']=[]
    # Nonfinite JSON deliberately exercises decoded production data, without b.encoded.
    path=tmp_path/'runtime.json';path.write_text(json.dumps(t),encoding='utf-8')
    with pytest.raises(ValueError):g.measured_runtime(g.ref(path,tmp_path),{'sha256':'e'*64},'a'*64,tmp_path)


@pytest.mark.parametrize('fault',[None,'desktop','peak','headroom'])
def test_real_memory_schema_and_gate(tmp_path,fault):
    log=tmp_path/'memory.log';log.write_text('synthetic memory fixture\n')
    row=dict(desktop_bytes=2*2**30,peak_reserved_bytes=10*2**30,total_bytes=16*2**30,raw_receipt=g.ref(log,tmp_path))
    if fault=='desktop':row['desktop_bytes']=-1
    if fault=='peak':row['peak_reserved_bytes']=13*2**30
    if fault=='headroom':row['desktop_bytes']=5*2**30
    m=dict(schema='h197-memory/v1',kernel_sha256='a'*64,actual_full_shape_tta=True,device_convention_verified=True,measurements=[row])
    r=g.ref(put(tmp_path/'memory.json',m),tmp_path)
    if fault:
        with pytest.raises(ValueError):g.memory(r,'a'*64,tmp_path)
    else:assert g.memory(r,'a'*64,tmp_path)['measurements']==[row]


def test_real_unqualified_candidate_cannot_admit(tmp_path):
    r=put(tmp_path/'forged.json',dict(schema='h197-gate-evidence/v1',candidate=b.KEY,phase='SUBMISSION_PASS',fixture_only=False))
    with pytest.raises((ValueError,KeyError)):g.evaluate('SUBMISSION_PASS',g.ref(r),g.ref(r))
