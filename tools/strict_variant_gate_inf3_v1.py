"""H197 exact INF3 gate v1.0.2. No sender, launch or account operation.

Build, measured new-path qualification and transport admission are distinct.
Canonical integration is a separately reviewed dependency, bound only when real.
"""
from __future__ import annotations
import argparse
import csv
from contextvars import ContextVar
import json
import math
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import c85r7_r6w958inf3_v1146 as b
from tools import native_codex_dispatch_record_v1 as native
need,read,sha=b.need,b.read,b.sha
VERSION='1.0.2'
PHASES=('BUILD_PASS','QUALIFICATION_PASS','SUBMISSION_PASS')
LINEAR_TERMS=('decode_all','render_tta_all','all_model_branches','inf3_inverse_pool')
GLOBAL_TERMS=('all_full_cohort_ranks','serialize_and_export')
ENTRYPOINTS={'own','q','raptor','r224','r6'}
CHECKED=ContextVar('inf3_checked_evidence',default=None)
NATIVE_CHECKED=ContextVar('inf3_native_evidence',default=None)


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def ref(path,root=ROOT):return dict(path=Path(path).resolve().relative_to(Path(root).resolve()).as_posix(),sha256=sha(path))


def file(r,root=ROOT):
    need(isinstance(r,dict) and set(r)=={'path','sha256'},'exact path/hash reference')
    path=(Path(root)/r['path']).resolve()
    need(path.is_relative_to(Path(root).resolve()),'evidence outside project')
    need(sha(path)==r['sha256'],'stale evidence: '+r['path'])
    if CHECKED.get() is not None:CHECKED.get().append((path,r['sha256']))
    return path


def review_payload(r,kernel,phase,evidence_sha,root=ROOT):
    """Exact report/schema parser only; production review also checks archives."""
    value=read(file(r,root));note=file(value['report'],root);receipt=read(file(value['native_receipt'],root))
    text=note.read_text(encoding='utf-8');lines=[line.strip() for line in text.splitlines() if line.strip()]
    attest=[line.split(': ',1)[1] for line in lines if line.startswith('H197-REVIEW-ATTESTATION: ')]
    need(len(attest)==1 and lines[-1]=='VERDICT: PASS','actual report must end in explicit PASS with one bound attestation')
    bound=json.loads(attest[0]);claimed={k:v for k,v in value.items() if k not in ('report','native_receipt')}
    need(bound==claimed,'report/sidecar assertion mismatch')
    need(value.get('schema')=='h197-independent-review/v1' and value.get('candidate')==b.KEY and
         value.get('slug')==b.SLUG and value.get('version')==b.VERSION and
         value.get('kernel_sha256')==kernel and value.get('phase')==phase and value.get('verdict')=='PASS' and
         value.get('builder')==b.BUILDER and value.get('reviewer') not in (None,'',b.BUILDER) and
         value.get('evidence_sha256')==evidence_sha,'exact independent phase review')
    need(receipt.get('schema')=='dispatch-terminal/v1' and receipt.get('state')=='SUCCESS' and
         receipt.get('agent_id')==value['reviewer'] and receipt.get('host')=='Codex app native subagent' and
         receipt.get('exit_code')==0 and receipt.get('report_sha256')==sha(note),'actual reviewer completion')
    need(Path(receipt['out']).resolve()==note,'receipt report path differs')
    completion=Path(receipt['completion_source']);completion=completion if completion.is_absolute() else Path(root)/completion
    need(completion.resolve().is_relative_to(Path(root).resolve()) and sha(completion)==receipt['completion_evidence_sha256'],'completion evidence')
    file(dict(path=completion.resolve().relative_to(Path(root).resolve()).as_posix(),sha256=receipt['completion_evidence_sha256']),root)
    pins=value.get('checked_source_sha256',{})
    need(pins and all(sha(Path(root)/p)==h for p,h in pins.items()),'review sources changed')
    return value


def review(r,kernel,phase,evidence_sha,root=ROOT):
    value=review_payload(r,kernel,phase,evidence_sha,root)
    receipt=file(value['native_receipt'],root)
    native.verify(receipt,root)  # real completion/prompt/report/log/archive replay
    if NATIVE_CHECKED.get() is not None:NATIVE_CHECKED.get().append((receipt,root))
    return value


def execution(r,kind,kernel,metadata,targets,root=ROOT):
    """Production schema parser; positive fixtures exercise this exact function."""
    import numpy as np
    x=read(file(r,root));n={'FULL_VISIBLE':3,'N32':32}[kind]
    need(x.get('schema')=='h197-qualified-execution/v1' and x.get('state')=='PASS' and x.get('kind')==kind and
         x.get('candidate')==b.KEY and x.get('slug')==b.SLUG and x.get('version')==b.VERSION and
         x.get('kernel_sha256')==kernel and x.get('metadata_sha256')==metadata and x.get('fast') is False and
         x.get('formula')=='H189_F_Q3' and x.get('studies')==n,'exact completed non-FAST execution')
    need(type(x.get('remote_version')) is int and x['remote_version']>0,'actual remote version')
    need(set(x.get('all_model_entrypoints',[]))==ENTRYPOINTS and x.get('cuda_all_entrypoints') is True and x.get('linux_actual') is True and
         x.get('codecs_pass') is True and x.get('fallbacks_pass') is True and x.get('all_20_cost_identities') is True,'new path coverage')
    logs=x.get('child_logs',{});codes=x.get('child_exit_codes',{})
    need(set(logs)==set(codes)==set(x['all_model_entrypoints']) and bool(logs) and all(type(v) is int and v==0 for v in codes.values()),'every child log/exit required')
    for item in logs.values():file(item,root)
    ids=read(file(x['ids'],root));need(isinstance(ids,list) and len(ids)==len(set(ids))==n and all(isinstance(u,str) and u for u in ids),'execution ID roster')
    with file(x['output_csv'],root).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
    need(reader.fieldnames==['StudyInstanceUID',*targets] and [r['StudyInstanceUID'] for r in rows]==ids,'execution target/ID order')
    values=np.asarray([[float(row[t]) for t in targets] for row in rows])
    need(values.shape==(n,12) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),'all IDs x12 finite probabilities')
    return x


def measured_runtime(r,n32_ref,kernel,root=ROOT):
    """All linear costs plus separately measured full-size global operations.

    No 32-study rank time is mislabeled fixed startup or extrapolated linearly.
    Global timing uses full-shape synthetic probabilities only after qualification.
    """
    import numpy as np
    t=read(file(r,root));need(t.get('schema')=='h197-runtime/v1' and t.get('kernel_sha256')==kernel and
        t.get('n32_sha256')==n32_ref['sha256'],'runtime candidate/N32 identity')
    n=t.get('hidden_studies');need(type(n) is int and n>=32,'authenticated hidden workload')
    source=read(file(t['cohort_source'],root));need(source.get('hidden_studies')==n and source.get('verified') is True,'hidden cohort source')
    raw=t.get('per_study_seconds',{})
    need(set(raw)==set(LINEAR_TERMS) and all(isinstance(v,list) and len(v)==32 and all(finite(a) and a>0 for a in v) for v in raw.values()),'all scalable per-study costs')
    glob=t.get('global_seconds',{})
    need(set(glob)==set(GLOBAL_TERMS) and all(finite(v) and v>0 for v in glob.values()) and
         t.get('global_measured_studies')==n and t.get('all_sequential_phase_costs_included') is True and
         t.get('all_inference_retries_tta_included') is True,'full-size ranks/exports and full-path cost')
    startup=t.get('fixed_startup_seconds');need(finite(startup) and startup>=0 and t.get('startup_fixed_proven') is True,'real fixed startup')
    receipts=t.get('raw_receipts',[]);need(bool(receipts),'raw timing receipts')
    for item in receipts:file(item,root)
    rates={k:float(np.percentile(v,95)) for k,v in raw.items()}
    projected=startup+n*sum(rates.values())+sum(glob.values())
    need(finite(t.get('projected_seconds')) and abs(t['projected_seconds']-projected)<1e-6 and projected<=32400,'measured hidden projection <=9h')
    return dict(projected_seconds=projected,hidden_studies=n,per_study_p95=rates,global_seconds=glob,fixed_startup_seconds=startup)


def memory(r,kernel,root=ROOT):
    m=read(file(r,root));need(m.get('schema')=='h197-memory/v1' and m.get('kernel_sha256')==kernel and
        m.get('actual_full_shape_tta') is True and m.get('device_convention_verified') is True,'actual full-shape memory evidence')
    measurements=m.get('measurements',[]);need(bool(measurements),'memory measurements absent')
    for row in measurements:
        need(all(finite(row.get(k)) for k in ('desktop_bytes','peak_reserved_bytes','total_bytes')) and
             row['desktop_bytes']>=0 and 0<row['peak_reserved_bytes']<=12*2**30 and
             row['desktop_bytes']+row['peak_reserved_bytes']+2*2**30<=row['total_bytes'],'measured 12GiB/2GiB memory gate')
        file(row['raw_receipt'],root)
    return m


def _evaluate(phase,evidence_ref,review_ref):
    need(phase in PHASES,'unknown phase')
    b.parent(replay=True);valid=b.validate();build=read(b.OUT/'build-receipt.json')
    kernel=valid['kernel_sha256'];e=read(file(evidence_ref));rv=review(review_ref,kernel,phase,evidence_ref['sha256'])
    need(e.get('schema')=='h197-gate-evidence/v1' and e.get('candidate')==b.KEY and e.get('slug')==b.SLUG and
         e.get('version')==b.VERSION and e.get('kernel_sha256')==kernel and e.get('phase')==phase and e.get('fixture_only') is False,'exact nonfixture evidence')
    need(all(rv['checked_source_sha256'].get(p)==h for p,h in b.sources().items()),'complete build/gate/test review source closure')
    cpu=read(file(e['cpu_replay']));need(file(e['cpu_replay'])==(b.SCRATCH/'CPU-REPLAY.json').resolve(),'exact CPU receipt')
    need(cpu.get('status')=='PASS_EXTRACTED_REAL58_REPLAY' and cpu.get('kernel_sha256')==kernel and
         cpu.get('source_sha256')==b.sources() and cpu.get('registration_sha256')==sha(b.SCRATCH/'registration.json') and
         cpu.get('exact_q3_bytes') is True and cpu.get('exact_final_float64') is True and cpu.get('parent_noop_float64_and_csv_exact') is True,'actual extracted/cache/no-op replay')
    tests=read(file(e['tests']));log=file(tests['log'])
    need(tests.get('schema')=='h197-tests/v1' and tests.get('actual_run') is True and tests.get('exit_code')==0 and
         tests.get('failed')==0 and type(tests.get('passed')) is int and tests['passed']>0 and
         tests.get('kernel_sha256')==kernel and tests.get('source_sha256')==b.sources(),'actual current CPU tests')
    need(re.search(r'\b'+str(tests['passed'])+r' passed\b',log.read_text(encoding='utf-8')) is not None,'test summary/log mismatch')
    original=read(b.BRIDGE/'results.json');need(original['proxy_gate']=='PASS','unchanged original proxy gate')
    result=dict(schema='h197-inf3-gate/v1',gate_version=VERSION,status=phase,phase=phase,candidate=b.KEY,slug=b.SLUG,version=b.VERSION,
        kernel_sha256=kernel,metadata_sha256=build['metadata_sha256'],gate_sha256=sha(__file__),fixture_only=False,qualified_for_submission=False,
        evidence=evidence_ref,review=review_ref,source_sha256=b.sources(),fast=False)
    if phase in ('QUALIFICATION_PASS','SUBMISSION_PASS'):
        proof=read(b.BRIDGE/'registration.json')
        visible=execution(e['full_visible'],'FULL_VISIBLE',kernel,build['metadata_sha256'],proof['targets'])
        execution(e['n32'],'N32',kernel,build['metadata_sha256'],proof['targets'])
        result['runtime']=measured_runtime(e['runtime'],e['n32'],kernel);memory(e['memory'],kernel)
        result['remote_version']=visible['remote_version']
        need(rv.get('joint_adverse_gold_reviewed') is True and rv.get('all_new_path_gates_checked') is True,'independent result/new-path review')
    if phase=='SUBMISSION_PASS':
        minimum=review(e['minimum_review'],kernel,'SUBMISSION_MINIMUM',evidence_ref['sha256'])
        need(minimum.get('remote_version')==result['remote_version'] and minimum.get('slug')==b.SLUG and
             minimum.get('duplicate_clear') is True and minimum.get('description_accurate') is True,'minimum exact version/duplicates/description review')
        integration_ref=e['canonical_integration'];i=read(file(integration_ref));sender=ROOT/'tools/kaggle_submit_once.py'
        need(i.get('schema')=='h197-canonical-integration/v1' and i.get('status')=='PASS' and i.get('candidate')==b.KEY and
             i.get('canonical_sha256')==sha(sender) and i.get('gate_sha256')==sha(__file__) and i.get('phase')=='SUBMISSION_PASS' and
             i.get('mandatory_nonfast_gate') is True and i.get('late_recheck') is True and i.get('shared_sender') is True,'actual mandatory canonical integration')
        ir=review(e['integration_review'],kernel,'CANONICAL_INTEGRATION',integration_ref['sha256'])
        required={'tools/kaggle_submit_once.py':sha(sender),'tools/strict_variant_gate_inf3_v1.py':sha(__file__)}
        need(all(ir['checked_source_sha256'].get(p)==h for p,h in required.items()),'integration sources/review stale')
        it=read(file(i['tests']));file(it['log'])
        need(it.get('exit_code')==0 and it.get('failed')==0 and it.get('actual_run') is True and
             all(it.get(k) is True for k in ('nonfast_bypass_rejected','wrong_candidate_rejected','wrong_phase_rejected','stale_source_rejected','late_recheck_verified','shared_sender_verified')) and
             all(it.get('source_sha256',{}).get(p)==h for p,h in required.items()),'actual integration tests incomplete/stale')
        scan=read(file(e['secret_scan']));need(scan.get('version')=='1.0.2' and scan.get('status')=='PASS' and scan.get('kernel_sha256')==kernel,'broad current secret scan')
        result['canonical_sha256']=sha(sender);result['qualified_for_submission']=True
    # Source/evidence late recheck within this evaluator. The external canonical
    # sender must independently repeat its gate immediately before shared submit.
    b.validate();file(evidence_ref);file(review_ref)
    need(sha(__file__)==result['gate_sha256'],'gate changed during validation')
    if phase=='SUBMISSION_PASS':need(sha(ROOT/'tools/kaggle_submit_once.py')==result['canonical_sha256'],'sender changed during validation')
    return result


def evaluate(phase,evidence_ref,review_ref):
    checked=[];native_checked=[];token=CHECKED.set(checked);ntoken=NATIVE_CHECKED.set(native_checked)
    try:
        result=_evaluate(phase,evidence_ref,review_ref)
        for path,expected in checked:need(sha(path)==expected,'evidence changed during validation: '+str(path))
        for receipt,root in native_checked:native.verify(receipt,root)
        return result
    finally:CHECKED.reset(token);NATIVE_CHECKED.reset(ntoken)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=PHASES)
    p.add_argument('--evidence',required=True);p.add_argument('--review',required=True);p.add_argument('--out');a=p.parse_args()
    try:
        result=evaluate(a.phase,ref(a.evidence),ref(a.review))
        if a.out:b.write(Path(a.out),b.encoded(result))
        print(json.dumps(result,sort_keys=True));return 0
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(json.dumps(dict(status='HOLD',reason=str(error),qualified_for_submission=False),sort_keys=True));return 2


if __name__=='__main__':raise SystemExit(main())
