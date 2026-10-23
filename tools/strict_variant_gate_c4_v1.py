"""Exact H198 C4 admission v1.0.0. Additive, fail closed, no account actions.

Four distinct phases; no fixture receipt admits production. The canonical
submit-once integration is deliberately a separate independently reviewed change.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import c85r7_r6w958c4_v1147 as b
need,read,sha=b.need,b.read,b.sha
VERSION='1.0.0'
PHASES=('BUILD_PASS','CACHE_LAUNCH_PASS','QUALIFICATION_LAUNCH_PASS','SUBMISSION_PASS')
REQUIRED_TERMS=('decode','render_crop','all_model_branches','rank_serialize')


def digest_file(ref,root=ROOT):
    need(set(ref)=={'path','sha256'},'exact file reference required')
    path=(Path(root)/ref['path']).resolve()
    need(path.is_relative_to(Path(root).resolve()),'evidence outside project')
    need(sha(path)==ref['sha256'],'stale evidence: '+ref['path'])
    return path


def review(ref,expected,root=ROOT):
    path=digest_file(ref,root);r=read(path)
    need(r.get('schema')=='h198-independent-review/v1' and r.get('verdict')=='PASS' and
         r.get('builder')==b.BUILDER and r.get('reviewer') not in (None,'',b.BUILDER) and
         r.get('kernel_sha256')==expected and r.get('candidate')==b.KEY,'independent current review required')
    att=digest_file(r['native_receipt'],root);receipt=read(att)
    note=digest_file(r['report'],root)
    need(receipt.get('state')=='SUCCESS' and receipt.get('report_sha256')==sha(note) and
         receipt.get('agent_id')==r['reviewer'] and receipt.get('host')=='Codex app native subagent',
         'actual native reviewer completion evidence')
    need(r.get('checked_source_sha256') and all(sha(Path(root)/p)==h for p,h in r['checked_source_sha256'].items()),'review source stale')
    return r


def finite(value):return type(value) in (float,int) and math.isfinite(value)


def core(phase,e,*,fixture_only=False):
    """Common gate evaluator exercised directly by positive/negative fixtures."""
    need(phase in PHASES,'unknown admission phase')
    need(e.get('candidate')==b.KEY and e.get('slug')==b.SLUG and e.get('version')==b.VERSION,'exact candidate only')
    need(e.get('parent_kernel_sha256')==b.PARENT_SHA and e.get('fast') is False,'parent/FAST')
    need(e.get('exact_diff') is True and e.get('retained15') is True and e.get('mixed_renderer') is True,'exact renderer/member diff')
    need(e.get('formula')=='H189_F_Q4' and e.get('builder')==b.BUILDER and e.get('reviewer') not in (None,'',b.BUILDER),'formula/independent identity')
    need(e.get('source_pins_current') is True and e.get('review_current') is True,'stale source/review')
    need(e.get('renderer_real_pass') is True and e.get('strict5_pass') is True and e.get('cpu_regressions_pass') is True,'CPU evidence incomplete')
    need(e.get('oof_q32_pass') is True and e.get('inherited_numeric_pass') is True,'OOF/inherited numeric gate')
    if phase!='BUILD_PASS':
        need(e.get('build_admitted') is True and e.get('launch_review_current') is True,'BUILD and launch review required')
        need(e.get('workload')==290 and e.get('model_count')==5 and e.get('studies')==58 and e.get('corpus_authenticated') is True,'fixed cache workload/corpus')
        need(e.get('resource_claim') and e.get('executor')=='B' and e.get('logical_device')=='cuda:0','B real resource claim')
        mem=e.get('memory',{})
        need(all(finite(mem.get(k)) for k in ('desktop_bytes','peak_reserved_bytes','total_bytes')) and
             0<mem['peak_reserved_bytes']<=12*2**30 and mem['desktop_bytes']>=0 and
             mem['desktop_bytes']+mem['peak_reserved_bytes']+2*2**30<=mem['total_bytes'],'measured real full-shape memory gate')
        need(e.get('guarded_worker') is True,'progress/memory/WDDM/throughput watchdog')
    if phase in ('QUALIFICATION_LAUNCH_PASS','SUBMISSION_PASS'):
        need(e.get('gold_state')=='INFERENCE_COMPLETE' and e.get('gold_shape')==[5,58,12] and
             e.get('gold_dtype')=='float32' and e.get('gold_finite') is True and e.get('gold_parent_noop') is True and
             e.get('gold_same_roster') is True and e.get('gold_failures')==0,'exact complete Gold58 bridge')
        need(e.get('gold_review_pass') is True and e.get('joint_harm_reviewed') is True,'Gold independent result/joint-harm review')
        need(e.get('submission_intent') is True and e.get('remote_assets_verified') is True and e.get('live_quota_verified') is True,'B current candidate intent/assets/quota')
    if phase=='SUBMISSION_PASS':
        need(type(e.get('remote_version')) is int and e['remote_version']>0 and e.get('minimum_review_pass') is True,'exact remote version/minimum review')
        need(e.get('full_visible') is True and e.get('n32') is True and e.get('cuda_all_entrypoints') is True and
             e.get('child_logs_pass') is True and e.get('codecs_pass') is True and e.get('fallbacks_pass') is True,'new CUDA full visible/N32 path')
        need(e.get('all_ids_12_finite') is True and e.get('duplicate_clear') is True and e.get('description_accurate') is True,'output/duplicate/description')
        t=e.get('runtime',{});terms=t.get('per_study_p95',{})
        need(set(terms)==set(REQUIRED_TERMS) and all(finite(x) and x>0 for x in terms.values()),'all scalable runtime terms measured')
        need(t.get('cohort_studies')==1300 and t.get('cohort_source_verified') is True and t.get('full_path_measured') is True and
             t.get('startup_fixed_proven') is True and finite(t.get('fixed_startup_seconds')) and t['fixed_startup_seconds']>=0,'measured runtime/source/startup')
        projection=t['fixed_startup_seconds']+1300*sum(terms.values())
        need(finite(t.get('projected_seconds')) and abs(t['projected_seconds']-projection)<1e-6 and projection<=32400,'nine-hour runtime gate')
        need(e.get('canonical_submit_integration') is True and e.get('secret_scan_v102_pass') is True,'mandatory canonical dispatch/secret scan')
    return dict(schema='h198-c4-gate/v1',gate_version=VERSION,phase=phase,status=phase,fixture_only=fixture_only,candidate=b.KEY,
                kernel_sha256=e['kernel_sha256'],qualified_for_submission=phase=='SUBMISSION_PASS' and not fixture_only)


def evaluate(phase,evidence,review_ref):
    """Production path reconstructs construction facts; never trusts their booleans."""
    b.parent(replay=True);b.validate();cm=b.archive_check();e=read(digest_file(evidence));rec=read(b.OUT/'build-receipt.json')
    rv=review(review_ref,rec['kernel_sha256'])
    need(rv['phase']==phase,'phase-specific review')
    # Every substantive external fact must have an independently reviewed exact
    # artifact. Additional phases require exact evidence-JSON hash in the review.
    need(rv.get('evidence_sha256')==evidence['sha256'],'review/evidence binding')
    required_sources={str(Path(__file__).resolve().relative_to(ROOT)).replace('\\','/'):sha(__file__),
        'tools/c85r7_r6w958c4_v1147.py':sha(b.__file__),
        'tools/hb198_c4_bridge_v1.py':sha(ROOT/'tools/hb198_c4_bridge_v1.py'),
        'tools/hb198_c4_gold58_v1.py':sha(ROOT/'tools/hb198_c4_gold58_v1.py')}
    need(all(rv['checked_source_sha256'].get(p)==h for p,h in required_sources.items()),'complete current review source closure')
    parity=read(b.SCRATCH/'RENDERER-PARITY.json');strict=read(b.SCRATCH/'CHECKPOINT-STRICT.json');oof=read(b.SCRATCH/'OOF-PROXY.json')
    assets=read(b.SCRATCH/'ASSET-AUTH.json')
    need(assets.get('status')=='TRAINING_POLICY_BOUND' and assets.get('kernel_sha256')==rec['kernel_sha256'] and
         assets.get('c4_archive_sha256')==b.ARCHIVE_SHA and assets.get('source_model_hashes')==[x['sha256'] for x in cm['entries']],
         'C4 original archive and training-policy authentication')
    need(parity['kernel_sha256']==strict['kernel_sha256']==rec['kernel_sha256'],'CPU receipts kernel binding')
    need(oof['input_pins_sha256']==sha(b.SCRATCH/'input-pins.json'),'OOF pin identity')
    from tools.hb198_c4_bridge_v1 import validate_pins
    validate_pins()
    tests=read(digest_file(rv['tests']))
    need(tests.get('exit_code')==0 and tests.get('failed')==0 and tests.get('kernel_sha256')==rec['kernel_sha256'] and tests.get('source_sha256')==required_sources,'current real test receipt')
    hist=oof['historical']['registered_gate']
    # Historical schema is independently inspected; missing exact boolean fails.
    e.update(candidate=b.KEY,slug=b.SLUG,version=b.VERSION,parent_kernel_sha256=b.PARENT_SHA,kernel_sha256=rec['kernel_sha256'],
        fast=False,exact_diff=True,retained15=True,mixed_renderer=True,formula='H189_F_Q4',builder=b.BUILDER,reviewer=rv['reviewer'],
        source_pins_current=True,review_current=True,renderer_real_pass=parity['status']=='PASS_CPU_REAL_RENDERER',strict5_pass=strict['status']=='PASS_CPU_STRICT_NO_FORWARD' and len(strict['models'])==5,
        cpu_regressions_pass=True,oof_q32_pass=oof['primary']['gate_pass'],inherited_numeric_pass=hist.get('status')=='ADVANCE')
    if phase!='BUILD_PASS':
        prior=read(digest_file(rv['build_receipt']))
        need(prior['phase']=='BUILD_PASS' and prior['fixture_only'] is False and prior['kernel_sha256']==rec['kernel_sha256'] and prior['gate_sha256']==sha(__file__),'authentic BUILD gate receipt')
        # Re-evaluate the original admission, not only its declared status.
        need(evaluate('BUILD_PASS',prior['evidence'],prior['review'])==prior,'stale BUILD admission')
        e['build_admitted']=True
        from tools import hb198_c4_gold58_v1 as cache
        card=cache.validate()
        need(sha(digest_file(rv['launch_card']))==sha(b.SCRATCH/'gold58-launch.json'),'reviewed exact launch card')
        claim=read(digest_file(rv['resource_claim']))
        need(claim.get('state')=='CLAIMED' and claim.get('owner')=='B' and claim.get('tag')==card['tag'] and
             claim.get('physical_gpu')==0 and claim.get('logical_device')=='cuda:0','real B resource claim artifact')
        mem=read(digest_file(rv['memory_measurement']))
        need(mem.get('status')=='PASS_MEASURED' and mem.get('kernel_sha256')==rec['kernel_sha256'] and
             mem.get('renderer_sha256')==card['renderer_sha256'] and mem.get('K')==36 and mem.get('tta')=='plane-aware-coronal-axial-flip' and
             mem.get('model_sha256')==[x['sha256'] for x in card['models']] and mem.get('physical_gpu')==0 and
             '4070 Ti SUPER' in mem.get('device_name',''),'actual full-shape cache memory evidence')
        need(len(mem.get('peak_reserved_bytes',[]))==5 and all(finite(v) and v>0 for v in mem['peak_reserved_bytes']),'five measured model peaks')
        for ref in mem.get('raw_receipts',[]):digest_file(ref)
        need(bool(mem.get('raw_receipts')),'raw memory receipts required')
        e.update(launch_review_current=True,workload=card['workload'],model_count=len(card['models']),studies=len(card['corpus']['ids']),
            corpus_authenticated=True,resource_claim=claim,executor='B',logical_device='cuda:0',physical_gpu=0,guarded_worker=True,
            memory=dict(desktop_bytes=mem['desktop_bytes'],peak_reserved_bytes=max(mem['peak_reserved_bytes']),total_bytes=mem['total_bytes']))
    if phase in ('QUALIFICATION_LAUNCH_PASS','SUBMISSION_PASS'):
        import numpy as np
        card=read(b.SCRATCH/'gold58-launch.json')
        gold=read(digest_file(rv['gold_bridge']));gr=review(rv['gold_result_review'],rec['kernel_sha256'])
        need(gr.get('phase')=='GOLD_RESULT' and gr.get('evidence_sha256')==sha(digest_file(rv['gold_bridge'])),'independent actual Gold result review')
        worker=read(b.SCRATCH/'gold58/worker.json')
        need(gold.get('status')=='COMPLETE_EXPOSED_DESCRIPTIVE_BRIDGE' and gold.get('worker_sha256')==sha(b.SCRATCH/'gold58/worker.json') and
             gold.get('card_sha256')==sha(b.SCRATCH/'gold58-launch.json') and worker.get('npz_sha256')==sha(b.SCRATCH/'gold58/c4-members.npz') and
             worker.get('card_sha256')==gold['card_sha256'] and worker.get('source_sha256')==sha(ROOT/'tools/hb198_c4_gold58_v1.py'),'complete authenticated Gold outputs')
        with np.load(b.SCRATCH/'gold58/c4-members.npz',allow_pickle=False) as z:
            need(set(z.files)=={'ids','targets','folds','values'} and z['ids'].tolist()==card['corpus']['ids'] and
                 z['targets'].tolist()==card['corpus']['targets'] and z['folds'].tolist()==list(range(5)),'exact Gold roster/folds/targets')
            values=z['values'];valid=bool(values.shape==(5,58,12) and values.dtype==np.float32 and np.isfinite(values).all() and ((values>=0)&(values<=1)).all())
        intent=read(digest_file(rv['candidate_intent']))
        need(intent.get('candidate')==b.KEY and intent.get('decision')=='QUALIFY_TO_SUBMIT' and intent.get('owner')=='B','actual B candidate intent')
        assets=read(digest_file(rv['remote_assets']))
        need(assets.get('dataset')==b.DATASET and assets.get('archive_sha256')==b.ARCHIVE_SHA and assets.get('accessible') is True,'current exact remote assets')
        quota=read(digest_file(rv['quota']))
        need(quota.get('source')=='https://www.kaggle.com/settings' and quota.get('read_immediately_before_decision') is True and
             quota.get('observed_at')==intent.get('quota_observed_at') and quota.get('remaining_seconds',0)>0,'live quota evidence; new path cannot FAST')
        e.update(gold_state=worker.get('state'),gold_shape=worker.get('shape'),gold_dtype=worker.get('dtype'),gold_finite=valid,
            gold_parent_noop=gold.get('parent_noop_exact'),gold_same_roster=True,gold_failures=worker.get('failures'),gold_review_pass=True,
            joint_harm_reviewed=gr.get('joint_harm_reviewed') is True,submission_intent=True,remote_assets_verified=True,live_quota_verified=True)
    if phase=='SUBMISSION_PASS':
        import csv
        import numpy as np
        executions=[]
        for key,kind,n in (('full_visible','FULL_VISIBLE',3),('n32','N32',32)):
            execution=read(digest_file(rv[key]));executions.append(execution)
            need(execution.get('schema')=='h198-qualified-execution/v1' and execution.get('state')=='PASS' and
                 execution.get('kind')==kind and execution.get('studies')==n and execution.get('slug')==b.SLUG and
                 execution.get('kernel_sha256')==rec['kernel_sha256'] and execution.get('metadata_sha256')==rec['metadata_sha256'] and
                 execution.get('formula')=='H189_F_Q4' and execution.get('fast') is False and
                 execution.get('archive_sha256')==b.ARCHIVE_SHA,'actual complete candidate visible/N32 binding')
            need(execution.get('cuda_all_entrypoints') is True and execution.get('codecs_pass') is True and execution.get('fallbacks_pass') is True,'measured CUDA/codecs/fallback')
            logs=execution.get('child_logs',[]);need(bool(logs),'all child logs missing')
            for log in logs:digest_file(log)
            need(execution.get('all_child_exit_codes') and all(v==0 for v in execution['all_child_exit_codes']),'failed child entrypoint')
            csvpath=digest_file(execution['output_csv']);idpath=digest_file(execution['ids']);ids=read(idpath)
            with csvpath.open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream);rows=list(reader)
            need(reader.fieldnames==['StudyInstanceUID',*card['corpus']['targets']] and len(ids)==len(set(ids))==n and
                 [r['StudyInstanceUID'] for r in rows]==ids,'all execution IDs x12')
            a=np.asarray([[float(r[t]) for t in card['corpus']['targets']] for r in rows])
            need(a.shape==(n,12) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'finite complete output')
        need(executions[0]['remote_version']==e.get('remote_version'),'confirmed remote version')
        timing=read(digest_file(rv['runtime_measurement']));raw=timing.get('per_study_seconds',{})
        need(set(raw)==set(REQUIRED_TERMS) and all(isinstance(x,list) and len(x)==32 and all(finite(v) and v>0 for v in x) for x in raw.values()),'actual N32 per-study timing arrays')
        need(timing.get('execution_sha256')==rv['n32']['sha256'] and timing.get('full_path_measured') is True and
             timing.get('startup_fixed_proven') is True and timing.get('all_sequential_phase_costs_included') is True and
             timing.get('cohort_studies')==1300 and timing.get('cohort_source_verified') is True,'full-path timing and cohort source')
        for ref in timing.get('raw_timing_receipts',[]):digest_file(ref)
        need(bool(timing.get('raw_timing_receipts')),'raw timing evidence missing')
        rates={k:float(np.percentile(v,95)) for k,v in raw.items()};startup=timing['fixed_startup_seconds']
        e['runtime']=dict(per_study_p95=rates,fixed_startup_seconds=startup,cohort_studies=1300,cohort_source_verified=True,
            full_path_measured=True,startup_fixed_proven=True,projected_seconds=startup+1300*sum(rates.values()))
        minimum=review(rv['minimum_review'],rec['kernel_sha256'])
        need(minimum.get('phase')=='SUBMISSION_MINIMUM' and minimum.get('remote_version')==e['remote_version'] and
             minimum.get('slug')==b.SLUG and minimum.get('duplicate_clear') is True and minimum.get('description_accurate') is True,'exact minimum review/duplicates/description')
        integration=read(digest_file(rv['canonical_integration']))
        sender=ROOT/'tools/kaggle_submit_once.py'
        need(integration.get('status')=='PASS' and integration.get('entrypoint_sha256')==sha(sender) and integration.get('gate_sha256')==sha(__file__) and
             integration.get('mandatory_for_nonfast') is True and integration.get('candidate')==b.KEY,'current mandatory canonical sender integration')
        sender_source=sender.read_text(encoding='utf-8')
        need('strict_variant_gate_c4_v1' in sender_source and 'SUBMISSION_PASS' in sender_source and b.KEY in sender_source,'C4 absent from canonical dispatch')
        ir=review(integration['review'],rec['kernel_sha256']);need(ir.get('phase')=='CANONICAL_INTEGRATION','separate integration review')
        scan=read(digest_file(rv['secret_scan']))
        need(scan.get('status')=='PASS' and scan.get('version')=='1.0.2' and scan.get('kernel_sha256')==rec['kernel_sha256'],'current broad secret scan')
        e.update(full_visible=True,n32=True,cuda_all_entrypoints=True,child_logs_pass=True,codecs_pass=True,fallbacks_pass=True,
            all_ids_12_finite=True,minimum_review_pass=True,duplicate_clear=True,description_accurate=True,canonical_submit_integration=True,secret_scan_v102_pass=True)
    result=core(phase,e)
    result.update(gate_sha256=sha(__file__),evidence=evidence,review=review_ref,evidence_sha256=evidence['sha256'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=PHASES);p.add_argument('--evidence',required=True);p.add_argument('--review',required=True);p.add_argument('--out');a=p.parse_args()
    try:
        ref=lambda x:dict(path=b.rel(Path(x)),sha256=sha(x))
        r=evaluate(a.phase,ref(a.evidence),ref(a.review))
        if a.out:b.write(Path(a.out),b.encoded(r))
        print(json.dumps(r,sort_keys=True));return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='HOLD',phase=a.phase,reason=str(exc))));return 2


if __name__=='__main__':raise SystemExit(main())
