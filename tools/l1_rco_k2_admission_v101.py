"""K2 authentic visible/N32/parity/runtime collection, v1.0.1.

Transport is coordinator-attested evidence plus archived independent review,
not cryptographic proof of Kaggle. No fixture can create a production gate.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import strict_variant_gate as base
from tools import strict_variant_gate_k2_v101 as gate
from tools import l1_rco_k2_v101 as b
from tools import l1_rco_k2_worker_v101 as w

VERSION='1.0.1'


def positive(value):
    base.need(type(value) in (float,int) and math.isfinite(value) and value>0,'positive measured scalar required')
    return value


def nonnegative(value):
    base.need(type(value) in (float,int) and math.isfinite(value) and value>=0,'nonnegative measured scalar required')
    return value


def file_ref(spec,inputs):
    base.need(set(spec)=={'path','sha256'},'exact file reference')
    path=base.local_path(spec['path'],inputs.root)
    base.need(inputs.digest(path)==spec['sha256'],'evidence hash mismatch')
    return path


def transport(spec,inputs,registration,kind):
    path=file_ref(spec,inputs);r=inputs.json(path)
    base.need(r['schema']=='k2-transport/v1' and r['kind']==kind and r['recorded_by']=='B' and
        r['kernel']==w.SLUG and type(r['kernel_version']) is int and r['kernel_version']>0 and
        r['fixture_only'] is False and r['status']=='COMPLETE' and r['prepush'] is False,'authentic completed transport identity')
    start=datetime.fromisoformat(r['started_at']);end=datetime.fromisoformat(r['completed_at'])
    base.need(start.tzinfo is not None and end.tzinfo is not None and end>start,'remote run chronology')
    base.need(isinstance(r['remote_run_id'],str) and len(r['remote_run_id'])>=8,'remote run identifier')
    metadata=inputs.json(file_ref(r['remote_metadata'],inputs))
    base.need(metadata['id']==w.SLUG and int(metadata['versionNumber'])==r['kernel_version'],'remote metadata slug/version')
    worker=inputs.json(file_ref(r['worker_receipt'],inputs))
    output=file_ref(r['submission'],inputs)
    bootstrap=inputs.json(file_ref(r['bootstrap_timing'],inputs))
    log=inputs.read(file_ref(r['worker_log'],inputs)).decode('utf-8-sig')
    base.need('K2 real inference complete' in log and 'FAST_PATH_VISIBLE' not in log,'real completed worker log')
    # Downloaded source is pinned and independently compared with the release.
    remote_source=file_ref(r['remote_notebook'],inputs)
    expected=inputs.root/registration['stage']/'kernel/release.ipynb'
    if kind=='visible':base.need(inputs.read(remote_source)==inputs.read(expected),'remote visible code differs')
    else:
        harness=inputs.json(file_ref(r['harness_binding'],inputs))
        base.need(harness['schema']=='k2-n32-harness/v1' and harness['worker_sha256']==
            b.s6.sha256(inputs.root/'tools/l1_rco_k2_worker_v101.py') and
            harness['release_notebook_sha256']==base.sha(inputs.read(expected)) and
            harness['remote_notebook_sha256']==inputs.digest(remote_source) and
            harness['allowed_changes']==['train split','registered32 template','separate output directory'] and
            harness['roster_sha256']==registration['n32_roster_sha256'],'N32 harness binding')
        diff=file_ref(harness['complete_diff'],inputs)
        base.need(base.parse(inputs.read(remote_source))==b.n32_notebook(registration) and
            inputs.read(diff).decode('utf-8')==b.harness_diff(registration),'N32 source exceeds exact harness delta')
    ids=registration['visible_ids'] if kind=='visible' else registration['n32_ids']
    values=b.prediction_table(output,ids)
    base.need(worker['schema']=='k2-worker/v1' and worker['status']=='COMPLETE' and
        worker['kernel_id']==w.SLUG and worker['version']==w.VERSION and worker['contract']==w.CONTRACT and
        worker['manifest_sha256']==registration['asset_manifest_sha256'] and
        worker['roster_sha256']==w.digest(ids) and worker['study_count']==len(ids) and
        worker['prediction_count']==len(ids) and worker['output_sha256']==inputs.digest(output) and
        worker['fast_path'] is False and worker['placeholder'] is False and worker['fallbacks']['prediction']==0,
        'real complete formula/output receipt')
    env=worker['environment']
    base.need(env['device']=='cuda:0' and env['cuda_initialized'] is True and env['tf32'] is False and
        env['autocast'] is False and 'Linux' in env['platform'] and env['gpu'] is not None,'production CUDA Linux route')
    base.need({k:env['versions'][k] for k in b.s6.WHEEL_VERSIONS}==b.s6.WHEEL_VERSIONS,'deployment library version parity')
    codec=worker['codecs']
    base.need(codec['status']=='ACTIVE' and codec['python_abi']==env['python_abi'] and codec['syntaxes'] and
        all(v['decoded'] is True for v in codec['syntaxes'].values()),'ACTIVE ABI-compatible codec decoder evidence')
    codec_receipt=inputs.json(file_ref(r['codec_asset_receipt'],inputs))
    base.need(codec_receipt['schema']=='k2-codec-qualification/v1' and codec_receipt['status']=='PASS' and
        codec_receipt['dataset']==b.CODEC_DATASET and codec_receipt['python_abi']==env['python_abi'] and
        codec_receipt['compatible'] is True and codec_receipt['asset_version']>0 and codec_receipt['wheel_sha256'],
        'private codec asset ABI/hash not qualified')
    for name,pin in codec_receipt['wheel_sha256'].items():
        base.need(isinstance(pin,str) and len(pin)==64,'codec wheel pin')
    studies=worker['studies']
    base.need(len(studies)==len(ids) and [x['index'] for x in studies]==list(range(len(ids))) and
        [x['uid_sha256'] for x in studies]==[__import__('hashlib').sha256(uid.encode()).hexdigest() for uid in ids] and
        all(x['windows']>0 and x['error'] is None and x['prediction_fallback']==0 for x in studies),'complete ordered study executions')
    timing=timing_components(worker,bootstrap,(end-start).total_seconds())
    base.need(bootstrap['manifest_sha256']==registration['asset_manifest_sha256'],'bootstrap manifest')
    peak=positive(worker['peak_reserved_bytes']); memory=r['memory']
    base.need(memory==worker['memory'],'transport memory differs from measured worker snapshot')
    base.need(peak<=12*2**30 and positive(memory['used_before_bytes'])+peak+2*2**30<=positive(memory['total_bytes']),
        'measured memory reserve/headroom')
    base.need(memory['source']=='measured before inference' and memory['device']==env['gpu'],'memory provenance')
    return dict(transport=r,worker=worker,values=values,**timing)


def timing_components(worker,bootstrap,remote_wall):
    """All unclassified elapsed work scales; prefix journals scale quadratically."""
    t=worker['timing'];n=worker['study_count']
    keys={'fixed_seconds','codec_header_seconds','cohort_setup_seconds','study_body_seconds',
          'cumulative_journal_seconds','export_seconds','residual_seconds'}
    base.need(t['schema']=='k2-timing/v1.0.1' and set(t)==keys|{'schema','studies','journal_rows'} and
        type(n) is int and n>0 and t['studies']==n and t['journal_rows']==n*(n+1)//2,'complete timing components')
    for k in keys:nonnegative(t[k])
    measured=sum(positive(x['end_to_end_seconds']) for x in worker['studies'])
    base.need(math.isclose(t['study_body_seconds'],measured,rel_tol=1e-9,abs_tol=1e-6),'study timing reconciliation')
    worker_total=positive(worker['total_seconds']);boot_total=positive(bootstrap['total_seconds'])
    base.need(math.isclose(sum(t[k] for k in keys),worker_total,rel_tol=1e-9,abs_tol=1e-6), 'worker wall reconciliation')
    fixed_boot=nonnegative(bootstrap['fixed_startup_seconds'])
    base.need(bootstrap['schema']=='k2-bootstrap/v1' and bootstrap['timing_policy']==b.TIMING_POLICY['schema'] and
        boot_total+1e-6>=fixed_boot+worker_total,'bootstrap wall excludes worker or startup')
    total=max(boot_total,positive(remote_wall))
    fixed=fixed_boot+t['fixed_seconds'];journal=t['cumulative_journal_seconds']
    linear=total-fixed-journal
    base.need(linear>=measured and math.isclose(fixed+linear+journal,total,abs_tol=1e-6),'all-in wall reconciliation')
    return dict(timing_schema=b.TIMING_POLICY['schema'],studies=n,fixed_seconds=fixed,
        linear_seconds=linear,journal_seconds=journal,total_seconds=total,
        worker_components=t,bootstrap_total_seconds=boot_total,remote_wall_seconds=remote_wall)


def runtime_projection(visible,n32,population=1300):
    base.need(type(population) is int and population==1300,'registered projection population')
    for r,n in ((visible,3),(n32,32)):
        base.need(r['timing_schema']==b.TIMING_POLICY['schema'] and r['studies']==n,'measured visible3/N32 timing schema')
        nonnegative(r['fixed_seconds']);positive(r['linear_seconds']);nonnegative(r['journal_seconds'])
        base.need(math.isclose(r['fixed_seconds']+r['linear_seconds']+r['journal_seconds'],
            positive(r['total_seconds']),rel_tol=1e-9,abs_tol=1e-6),'projection input reconciliation')
    fixed=max(visible['fixed_seconds'],n32['fixed_seconds'])
    per=max(r['linear_seconds']/r['studies'] for r in (visible,n32))
    coefficient=max(r['journal_seconds']/(r['studies']*(r['studies']+1)) for r in (visible,n32))
    journal=population*(population+1)*coefficient
    seconds=fixed+population*per+journal
    return dict(status='PASS' if seconds<=32400 else 'FAIL',fixed_seconds=fixed,linear_per_study_seconds=per,
                journal_coefficient=coefficient,projected_journal_seconds=journal,policy=b.TIMING_POLICY,
                population=population,projected_seconds=seconds,gate_seconds=32400,
                uncertainty='Observed visible/N32 workload; GPU node variance and hidden workload remain uncertain')


def check_bundle(bundle_path,inputs):
    spec=inputs.json(bundle_path)
    base.need(spec['schema']=='k2-final-evidence/v1' and spec['fixture_only'] is False and spec['candidate_id']==b.CANDIDATE,
        'final evidence schema/identity')
    build_path=file_ref(spec['build_gate'],inputs)
    build=gate.verify_receipt(build_path,inputs,'BUILD_READY')
    reg=inputs.json(base.local_path(build['stage'],inputs.root)/'registration.json');reg=dict(reg,stage=build['stage'])
    base.need(spec['population_affirmation']['population']==reg['runtime']['population'] and
        spec['population_affirmation']['recorded_by']=='B' and spec['population_affirmation']['applicable'] is True,
        'projection population requires B applicability evidence')
    file_ref(spec['population_affirmation']['basis'],inputs)
    v=transport(spec['visible'],inputs,reg,'visible');n=transport(spec['n32'],inputs,reg,'n32')
    base.need(v['transport']['remote_run_id']!=n['transport']['remote_run_id'] and
        v['transport']['kernel_version']!=n['transport']['kernel_version'],'visible and N32 cannot substitute one run')
    # A later harness version cannot overwrite the visible release at submission.
    base.need(v['transport']['kernel_version']>n['transport']['kernel_version'],'final visible version must follow N32 harness')
    reference_path=file_ref(spec['fp32_reference'],inputs)
    reference=b.prediction_table(reference_path,reg['n32_ids'])
    provenance=inputs.json(file_ref(spec['fp32_reference_receipt'],inputs))
    base.need(provenance['schema']=='k2-fp32-reference/v1' and provenance['status']=='COMPLETE' and
        provenance['model_sha256']==w.MODEL_SHA and provenance['roster_sha256']==reg['n32_roster_sha256'] and
        provenance['output_sha256']==inputs.digest(reference_path) and provenance['contract']==w.CONTRACT,
        'FP32 reference model/roster/contract')
    base.need(provenance['source_sha256']==inputs.digest(inputs.root/'tools/l1_rco_k2_v101.py') and
        provenance['registration_sha256']==base.sha(inputs.read(base.local_path(build['stage'],inputs.root)/'registration.json')),
        'FP32 reference source/registration')
    base.need([{k:x[k] for k in ('images_sha256','metadata_sha256','windows')} for x in n['worker']['studies']]==provenance['tensors'],
        'N32 production raw images/windows/metadata differ from authenticated cache')
    cross=b.parity(n['values'],reference,'cross_runtime')
    old_path=inputs.root/'.local/scratch/l1-rco/k1v2/fit/holdout.csv'
    base.need(inputs.digest(old_path)==b.PINS['.local/scratch/l1-rco/k1v2/fit/holdout.csv'],'saved bf16 identity')
    drift=b.parity(n['values'],b.prediction_table(old_path,reg['n32_ids'],k1=True),'k1_bf16')
    base.need(cross['status']==drift['status']=='PASS','registered FP32/bf16 parity limits')
    projection=runtime_projection(v,n,reg['runtime']['population'])
    base.need(projection['status']=='PASS','hidden runtime exceeds 9 hours')
    geometry=inputs.json(file_ref(spec['geometry_parity'],inputs))
    base.need(geometry['schema']=='k2-geometry-parity/v1' and geometry['status']=='PASS' and
        geometry['roster_sha256']==reg['n32_roster_sha256'] and geometry['images_exact'] is True and
        geometry['metadata_exact'] is True and geometry['window_order_exact'] is True and
        geometry['production_route']=='cuda:0' and geometry['model_sha256']==w.MODEL_SHA,'production anchor/render parity')
    file_ref(geometry['comparison'],inputs)
    return dict(build=build,kernel_version=v['transport']['kernel_version'],visible_receipt=v['transport']['worker_receipt'],
        visible_transport=spec['visible'],parity=dict(cross_runtime=cross,k1_bf16=drift),runtime=projection)


def pin(bundle,output,root=ROOT):
    inputs=base.Inputs(Path(root));checked=check_bundle(bundle,inputs)
    result=dict(schema='k2-final-pin/v1',status='PINNED_NOT_ADMITTED',kernel_id=w.SLUG,
        kernel_version=checked['kernel_version'],bundle=base.local_path(bundle,inputs.root).relative_to(inputs.root).as_posix(),
        build_gate=inputs.json(bundle)['build_gate'],parity=checked['parity'],runtime=checked['runtime'],
        input_sha256=inputs.hashes(),qualified_for_submission=False)
    inputs.recheck();w.need(not Path(output).exists(),'final pin exists');w.write(output,result);return result


def replay_pin(path,inputs):
    result=inputs.json(path)
    base.need(result['schema']=='k2-final-pin/v1' and result['status']=='PINNED_NOT_ADMITTED' and
        result['kernel_id']==w.SLUG,'final pin schema')
    for name,pin in result['input_sha256'].items():base.need(inputs.digest(name)==pin,'final input drift')
    checked=check_bundle(result['bundle'],inputs)
    base.need(result['kernel_version']==checked['kernel_version'] and result['parity']==checked['parity'] and
        result['runtime']==checked['runtime'],'final computations changed')
    return result,checked


def final_review(review,final_pin,builder,inputs,version):
    lane=gate.independent_review(review,final_pin,builder,inputs,'K2-Final-Pin-SHA256')
    text=base.verdict(inputs.read(review))
    for required in (f'Kernel-Version: {version}',f'Candidate-ID: {b.CANDIDATE}',
                     'Duplicate-Check: PASS', 'Description-Check: PASS', 'Minimum-Submission-Review: PASS'):
        base.need(required in text,'minimum review missing '+required)
    return lane


def bind(final_pin,review,output,root=ROOT):
    inputs=base.Inputs(Path(root));fp,checked=replay_pin(final_pin,inputs)
    build=checked['build'];bp=gate.checked_pin(build['pin'],inputs)
    lane=final_review(review,final_pin,bp['builder_lane'],inputs,fp['kernel_version'])
    result=dict(build,phase='FINAL_READY',qualified_for_submission=True,
        kernel_version=fp['kernel_version'],final_pin=base.local_path(final_pin,inputs.root).relative_to(inputs.root).as_posix(),
        final_review=base.local_path(review,inputs.root).relative_to(inputs.root).as_posix(),final_reviewer_lane=lane,
        input_sha256=inputs.hashes())
    inputs.recheck();w.need(not Path(output).exists(),'final gate exists');w.write(output,result);return result


def replay_final(record,inputs,build_pin):
    fp,checked=replay_pin(record['final_pin'],inputs)
    lane=final_review(record['final_review'],record['final_pin'],build_pin['builder_lane'],inputs,fp['kernel_version'])
    base.need(record['kernel_version']==fp['kernel_version'] and lane==record['final_reviewer_lane'] and
        checked['build']['pin']==record['pin'],'final review/version replay')
    return checked


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('pin');q.add_argument('--bundle',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
    q=s.add_parser('bind')
    for n in ('final-pin','review','output'):q.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args(argv)
    try:
        r=pin(a.bundle,a.output) if a.command=='pin' else bind(a.final_pin,a.review,a.output)
        print(w.encoded({k:r[k] for k in ('status','kernel_id','qualified_for_submission')}).decode());return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(w.encoded(dict(status='FAIL',error=str(exc),qualified_for_submission=False)).decode());return 1


if __name__=='__main__':sys.exit(main())
