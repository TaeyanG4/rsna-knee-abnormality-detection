"""C85-R7-PL15 + FLIP_F0F2 1.5.0: sealed local preparation, no automatic GPU/network work.

Changelog 1.5.0: retain exact v1.4.0 PL15 a/b formulas; add only v1.3.0
FLIP_F0F2 serving; one shared qualification path, fail-closed preparation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp169_release_c85r7codec_deadline_v122 as base
from tools import vnext_r7_pl_v1 as pl
from tools import exp169_release_c85r7_pl15_v140 as predecessor
from tools import exp169_release_c85r7flip_v130 as flip_build
from tools import exp169_r7_selective_flip_v130 as flip
from tools import c85r7_v150_safety as safety
import math
from tools.hb116_vnext_release import package, mean_survivors

VERSION = '1.5.0'
INTENT_SHA = 'b1e99c495433ce94e7ed41f230392ed65082f30f0f7e87d8128c685872844608'
OUT = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.5.0'
SCRATCH = ROOT/'.local/scratch/astra-b1/c85r7-v150'
VARIANTS = {
    'a': dict(EXP143v2=17, EXP155=17, Q_A5q=12, PL_STUDENT=12),
    'b': dict(EXP143v2=17, EXP155=17, PL_STUDENT=12),
}
MIL = [e for e in pl.member_roster() if e['kind'] == 'MIL']
need, read, dump, digest = pl.need, pl.read, pl.dump, pl.digest


def payloads():
    files = pl.source_payloads()
    for module in (sys.modules[__name__], predecessor, flip_build, flip, safety):
        files['tools/'+Path(module.__file__).name] = Path(module.__file__).read_bytes()
    files['tools/c85r7_v150_runner.py'] = (ROOT/'tools/c85r7_v150_runner.py').read_bytes()
    return files


def register(stage=OUT):
    stage = Path(stage)
    need(stage.resolve() != base.OUT.resolve(), 'frozen base stage')
    intent = (OUT/'intent.json').read_bytes()
    need(hashlib.sha256(intent).hexdigest() == INTENT_SHA, 'pre-result intent drift')
    base.write_once(stage/'intent.json', intent)
    base.validate(base.OUT)
    predecessor.validate(predecessor.OUT)
    parents = {}
    for arm in ('global', 'fill'):
        # Original registrations remain intact. The explicit MIL projection is
        # independently sealed and consumed by the existing target/train/export
        # and validate_bindings functions, which support an explicit roster.
        parent = stage/'upstream'/arm
        pl.register(parent, arm=arm)
        reg = pl.registration(parent)
        student = dict(reg, members=MIL, parent_registration_sha256=digest(reg),
                       projection='MIL-only C85-R7-PL15 v1.5.0')
        folder = stage/'student'/arm
        dump(folder/'registration.json', student, immutable=True)
        base.write_once(folder/'registration.sha256', (digest(student)+'\n').encode())
        parents[arm] = dict(parent=digest(reg), student=digest(student))
    _, pins = package(payloads())
    reg = dict(name='C85-R7-PL15', version=VERSION, status='SEALED_PLACEHOLDERS', intent_sha256=INTENT_SHA,
        base_version='1.2.2', base_registration_sha256=pl.sha(base.OUT/'registration.json'),
        outer_weights=[.85, .15], variants=VARIANTS, denominators={'a':58, 'b':46},
        public='P_R7=.5*r(R)+.5*r(S_R7)', own='weighted mean of component ranks; final rank(O)',
        pl_aggregation='native float32 mean_survivors before component rank',
        members=MIL, arm_policy='actual H-B-146 v1.1.1 registered_go.selected_arm; global or fill only',
        exposure='EXPOSED_EXPLORATORY', clean_oof=False, students=parents,
        payload_pins=pins, diagnostic_zero='both variants delegate complete v1.2.2 blend including Q',
        flip_policy=flip.policy('FLIP_F0F2'), predecessor_registration_sha256=pl.sha(predecessor.OUT/'registration.json'),
        shared_code_path='one base+PL run; both compositors differ only by sealed weights/member set',
        qualification_policy='one N32 bound to code+binding+image; visible evidence for both a/b',
        decision_rule=dict(macro_min=.001, ci_lower_exclusive=0, defined_fraction_min=.95, target_floor=-.005),
        compute_seconds=27000, publication_reserve_seconds=1200, limit_seconds=32400,
        qualification='HOLD until authentic GO, exported checkpoints and target runtime/fault qualification',
        motivation='D-B-179: vNext-R7-A own .30 dilutes R7; keep public .85 sealed',
        results_access='none; both variants sealed together, no outcome-dependent coefficients')
    dump(stage/'registration.json', reg, immutable=True)
    base.write_once(stage/'registration.sha256', (digest(reg)+'\n').encode())
    return dict(status='REGISTERED_LOCAL_ONLY', seal=digest(reg), variants=['a','b'])


def registration(stage=OUT):
    stage = Path(stage)
    reg = read(stage/'registration.json')
    need(digest(reg) == (stage/'registration.sha256').read_text().strip(), 'registration seal')
    need(reg.get('intent_sha256') == INTENT_SHA and pl.sha(stage/'intent.json') == INTENT_SHA, 'pre-result intent identity')
    need(reg['version'] == VERSION and reg['variants'] == VARIANTS and reg['outer_weights'] == [.85,.15], 'formula drift')
    need(pl.sha(base.OUT/'registration.json') == reg['base_registration_sha256'], 'base identity')
    need(pl.sha(predecessor.OUT/'registration.json') == reg['predecessor_registration_sha256'], 'predecessor identity')
    need(reg['flip_policy'] == flip.policy('FLIP_F0F2'), 'flip policy drift')
    need(package(payloads())[1] == reg['payload_pins'], 'source drift')
    for arm in ('global','fill'):
        parent = pl.registration(stage/'upstream'/arm)
        student = pl.registration(stage/'student'/arm)
        need(student == dict(parent, members=MIL, parent_registration_sha256=digest(parent),
                             projection='MIL-only C85-R7-PL15 v1.5.0'), 'student projection drift')
        need(reg['students'][arm] == dict(parent=digest(parent),student=digest(student)), 'student seal')
    return reg


def prepare(stage=OUT):
    stage = Path(stage)
    reg = registration(stage)
    archive, _ = package(payloads())
    base.write_once(stage/'runtime.zip', archive)
    _, frozen_flip = flip_build.payloads('FLIP_F0F2')
    flip_archive, flip_pins = package(frozen_flip)
    base.write_once(stage/'flip-v130-runtime.zip', flip_archive)
    dump(stage/'flip-v130-pins.json', flip_pins, immutable=True)
    # Retain the exact codec/insurance runnable base separately; never edit it.
    archive, pins = base.package()
    base.write_once(stage/'base-v122-runtime.zip', archive)
    dump(stage/'base-v122-pins.json', pins, immutable=True)
    for arm in ('global','fill'):
        s = pl.registration(stage/'student'/arm)
        template = dict(registration_sha256=digest(s), arm=arm, go=None, teacher=None,
            qualification=None, members=[dict(e,path=None,sha256=None,bytes=None,training_receipt=None) for e in MIL],
            runtime=dict(legacy=None,public_source=None,own_manifest=None,competition=None,
                         bundle=None,code=None,anatomy=None,source=None,wheels=None))
        dump(stage/'student'/arm/'bindings-template.json', template, immutable=True)
        plans = [x for x in pl.training_configs(arm) if x['family']=='MIL' and x['mode']=='fivefold']
        dump(stage/'student'/arm/'training-plan.json', plans, immutable=True)
    for variant in VARIANTS:
        dump(stage/f'variant-{variant}.json', dict(registration_sha256=digest(reg),variant=variant,
            weights=VARIANTS[variant], denominator=sum(VARIANTS[variant].values()),
            arm=None, bindings=None, qualified_for_submission=False), immutable=True)
    dump(stage/'runtime-projection.json', runtime_projection(), immutable=True)
    return dict(status='PREPARED_LOCAL_ONLY', variants=2, placeholder_members=len(MIL), production_run=False)


def validate(stage=OUT):
    stage = Path(stage)
    reg = registration(stage)
    with zipfile.ZipFile(stage/'runtime.zip') as z:
        need(len(z.namelist()) == len(reg['payload_pins']), 'duplicate archive members')
        need({n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()} == reg['payload_pins'], 'archive drift')
    _, frozen_flip = flip_build.payloads('FLIP_F0F2')
    flip_archive, flip_pins = package(frozen_flip)
    need((stage/'flip-v130-runtime.zip').read_bytes()==flip_archive and read(stage/'flip-v130-pins.json')==flip_pins, 'flip archive drift')
    archive, pins = base.package()
    need((stage/'base-v122-runtime.zip').read_bytes()==archive and read(stage/'base-v122-pins.json')==pins, 'base archive drift')
    for v in VARIANTS:
        need(read(stage/f'variant-{v}.json') == dict(registration_sha256=digest(reg),variant=v,
            weights=VARIANTS[v],denominator=sum(VARIANTS[v].values()),arm=None,bindings=None,
            qualified_for_submission=False), 'variant seal')
    for arm in ('global','fill'):
        s = pl.registration(stage/'student'/arm)
        need(read(stage/'student'/arm/'bindings-template.json') == dict(
            registration_sha256=digest(s),arm=arm,go=None,teacher=None,qualification=None,
            members=[dict(e,path=None,sha256=None,bytes=None,training_receipt=None) for e in MIL],
            runtime=dict(legacy=None,public_source=None,own_manifest=None,competition=None,
                         bundle=None,code=None,anatomy=None,source=None,wheels=None)), 'placeholder drift')
        need(read(stage/'student'/arm/'training-plan.json') == [x for x in pl.training_configs(arm)
            if x['family']=='MIL' and x['mode']=='fivefold'], 'training plan drift')
    need(read(stage/'runtime-projection.json') == runtime_projection(), 'runtime drift')
    return dict(status='PASS_PREPARED', variants=2, qualified_for_submission=False)


def bind(stage, inputs, output):
    reg = registration(stage)
    bound = read(inputs)
    need(bound.get('arm') in reg['students'], 'winner arm absent')
    # The existing binder and shared real readout adapter authenticate every
    # exporter receipt. No GO schema, training recipe or teacher rules duplicated.
    return pl.bind(Path(stage)/'student'/bound['arm'], inputs, output)


def runtime_projection(student_seconds_per_study=None):
    increment = flip_build.runtime_projection('FLIP_F0F2')['incremental_hours']*3600
    base_seconds = 4.7*3600 + increment
    need(student_seconds_per_study is None or
         (math.isfinite(student_seconds_per_study) and student_seconds_per_study >= 0), 'invalid serving estimate')
    seconds = None if student_seconds_per_study is None else base_seconds+1300*student_seconds_per_study
    return dict(status='UNMEASURED_PL_HOLD' if seconds is None else ('WITHIN_CAP' if seconds <= 27000 else 'OVER_CAP_HOLD'),
        base_hours=4.7, flip_increment_hours=increment/3600, before_pl_hours=base_seconds/3600,
        N=1300, compute_seconds=27000, reserve_seconds=1200, limit_seconds=32400,
        student_seconds_per_study=student_seconds_per_study,
        projected_compute_hours=None if seconds is None else seconds/3600,
        projected_with_reserve_hours=None if seconds is None else (seconds+1200)/3600,
        max_student_seconds_per_study=(27000-base_seconds)/1300,
        assumptions='unscaled c=1; whole 10-checkpoint PL ensemble; setup/codecs/cleanup included; no Q removal credit')


def decision_gate(result):
    """Already registered FLIP_F0F2 gate; aggregate receipt only, never labels."""
    need(result.get('variant') == 'FLIP_F0F2', 'registered flip variant')
    numeric=[result.get(k) for k in ('macro_delta','ci_lower','defined_fraction')]
    targets=result.get('target_deltas',[])
    need(len(targets)==12 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in [*numeric,*targets]), 'decision metrics')
    need(numeric[0]>=.001 and numeric[1]>0 and numeric[2]>=.95 and min(targets)>=-.005, 'registered flip GO thresholds')
    return True


def review_gate(stage, review):
    need(review is not None, 'independent review absent')
    r=read(pl.pin(review))
    need(r.get('status')=='PASS' and r.get('independent') is True and bool(r.get('reviewer')), 'independent review required')
    need(r.get('registration_sha256')==digest(registration(stage)), 'review registration identity')
    smoke=r.get('smoke_receipts',[])
    need(len(smoke)>=2, 'review requires suite and real CPU smoke receipts')
    kinds=set()
    for asset in smoke:
        receipt=read(pl.pin(asset)); kinds.add(receipt.get('kind'))
        need(receipt.get('registration_sha256')==r['registration_sha256'] and receipt.get('status')=='PASS', 'review smoke identity/status')
        need(bool(receipt.get('sources')), 'smoke evidence sources absent')
        for source in receipt['sources']: pl.pin(source)
    need({'suite','real_cpu_smoke'}<=kinds, 'review smoke coverage')
    return r


def health_gate(stage, binding, image_sha256, asset):
    need(asset is not None,'full-shape memory and healthy-throughput measurement absent')
    h=read(pl.pin(asset))
    need(h.get('status')=='PASS' and h.get('registration_sha256')==digest(registration(stage)) and
         h.get('binding_sha256')==pl.sha(binding) and h.get('image_sha256')==image_sha256,'health measurement identity')
    need(h.get('device')=='2xT4' and h.get('full_shape') is True and h.get('flip_policy')==flip.policy('FLIP_F0F2'), 'full-shape flip measurement required')
    peaks=h.get('peak_reserved_mib',[])
    need(len(peaks)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<v<=10240 for v in peaks),'full-shape reserved memory exceeds 10 GiB')
    rates=h.get('healthy_commits_per_second',{})
    need(set(rates)=={'own','Q_A5q','R','S_R7','PL-0','PL-1'} and
         all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in rates.values()),'measured healthy journal rates required')
    need(bool(h.get('sources')),'health raw evidence absent')
    for source in h['sources']:pl.pin(source)
    return h


def admit(stage=OUT, binding=None, qualification=None, variant='a', review=None, flip_go=None):
    validate(stage)
    need(variant in VARIANTS, 'unregistered variant')
    reasons=[]
    if binding is None: reasons.append('genuine H-B-146 GO and v1.1.1 MIL exports absent')
    else:
        b=read(binding)
        need(b.get('arm') in ('global','fill'), 'winning arm')
        pl.validate_bindings(pl.registration(Path(stage)/'student'/b['arm']),b)
    if review is None: reasons.append('independent review bound to CPU smoke receipts absent')
    else: review_gate(stage,review)
    if flip_go is None: reasons.append('registered H-B-149 FLIP_F0F2 GO absent')
    else: decision_gate(read(pl.pin(flip_go)))
    if qualification is None: reasons.append('one shared N32 and visible a+b evidence absent')
    else:
        need(binding is not None, 'qualification requires checkpoint bindings')
        q=read(pl.pin(qualification))
        common=dict(registration_sha256=digest(registration(stage)),binding_sha256=pl.sha(binding),
                    flip_policy=flip.policy('FLIP_F0F2'))
        need(all(q.get(k)==v for k,v in common.items()) and q.get('variants')==['a','b'], 'qualification identity')
        need(q.get('device')=='2xT4' and isinstance(q.get('image_sha256'),str) and len(q['image_sha256'])==64, 'target image')
        need(q.get('n32_count')==1, 'one shared N32 required')
        health_gate(stage,binding,q['image_sha256'],q.get('health'))
        for kind,asset in [('n32',q.get('n32')),('visible',q.get('visible'))]:
            e=read(pl.pin(asset))
            need(all(e.get(k)==v for k,v in common.items()) and e.get('image_sha256')==q['image_sha256'], 'evidence identity')
            need(e.get('mode')==kind and e.get('status')=='PASS' and e.get('variants')==['a','b'], 'qualification evidence')
            need(all(e.get(k) is True for k in ('complete_ordered_csv','finite','zero_healthy_fallbacks','descendants_exited',
                'gpu_memory_released','native_parity','codecs_best_effort')), 'target qualification')
            if kind=='n32': need(e.get('studies')==32 and e.get('fault_n32_pass') is True, 'fault N32 qualification')
        projection=runtime_projection(q.get('student_seconds_per_study'))
        need(projection['status']=='WITHIN_CAP', 'runtime projection exceeds 7.5h or unmeasured')
        upper=q.get('upper_seconds',1e9)
        need(isinstance(upper,(int,float)) and math.isfinite(upper) and
             projection['projected_compute_hours']*3600<=upper<=27000 and
             q.get('publication_reserve_seconds')==1200 and upper+1200<=q.get('total_seconds',0)<=32400, 'runtime qualification')
    return dict(status='HOLD' if reasons else 'ADMITTED', reasons=reasons, qualified_for_submission=False)


# EXACT v1.4.0 arithmetic, including its diagnostic zero-PL full-Q fallback.
# The serving flip belongs upstream, never inside this compositor.
blend = predecessor.blend


def student_predictions(directory, ids, identity):
    import numpy as np
    need(list(identity['members']) == [e['id'] for e in MIL], 'student journal roster')
    raw, failures, reason=pl.collect(directory,ids,identity)
    values=mean_survivors([raw[e['id']] for e in MIL],dtype='float32')
    bad=~np.isfinite(values).all(1)
    values[bad]=.5
    return values,{u:'PL members unavailable' for u,f in zip(ids,bad) if f},failures,reason


def native_job(stage, binding, ids, rows, runtime, output, run_id, device='cuda:0'):
    """MIL-only job consumed unchanged by vnext_r7_pl_worker_v1.native.

    Construction is CPU-only. B runs workers through the shared deadline
    supervisor after combined target qualification; this module never launches.
    """
    registration(stage)
    b=read(binding)
    need(b.get('arm') in ('global','fill'), 'winning arm')
    reg=pl.registration(Path(stage)/'student'/b['arm'])
    entries=pl.validate_bindings(reg,b)
    ids=base.v5q.validate_ids(ids)
    output=Path(output)
    branch=dict(path=str(output/'own'),ids=ids,identity=pl.identity(reg,'own',entries,run_id))
    return dict(runtime,ids=ids,entries=entries,branches=[branch],codec=None,
        rows=[r for r in rows if r[pl.UID] in set(ids)],images=str(Path(runtime['competition'])/'test_series'),
        family='NATIVE',output=str(output),device=device,work=str(output),registration=reg,runtime_root=str(ROOT))


def compose(stage, binding, variant, a172_dir, pr7_dir, shards, run_id, output, *,
            base_receipt, diagnostic=True, qualification=None, review=None, flip_go=None):
    from tools.exp169_c85r7ins_checks_v122 import snapshot
    import numpy as np
    registration(stage)
    if not diagnostic:
        need(admit(stage,binding,qualification,variant,review,flip_go)['status']=='ADMITTED', 'production admission HOLD')
    need(variant in VARIANTS, 'unregistered variant')
    b=read(binding); reg=pl.registration(Path(stage)/'student'/b['arm'])
    entries=pl.validate_bindings(reg,b)
    authenticated=read(pl.pin(base_receipt))
    need(authenticated.get('flip_policy')==flip.policy('FLIP_F0F2'), 'authenticated FLIP_F0F2 baseline required')
    ids,raw,failures=snapshot(Path(a172_dir),Path(pr7_dir))
    # Bind the actual authenticated prediction snapshot as well as its serving receipt.
    need(authenticated.get('snapshot_sha256')==snapshot_digest(ids,raw,failures), 'baseline snapshot identity')
    seen=set(); student=np.full((len(ids),12),.5); missing={}; expected=pl.identity(reg,'own',entries,run_id)
    for shard in shards:
        subset=base.v5q.validate_ids(shard['ids'])
        need(set(subset)<=set(ids) and not seen.intersection(subset), 'student shard cohort')
        need(shard['identity']==expected, 'student checkpoint/run identity')
        values,bad,_,_=student_predictions(shard['path'],subset,expected)
        student[[ids.index(u) for u in subset]]=values
        seen.update(subset); missing.update(bad)
    need(seen==set(ids), 'incomplete student shard assignment')
    need(len(missing)<len(ids), 'PL student unavailable for entire cohort')
    failures=dict(failures,PL_STUDENT=missing)
    payload=base.p.csv_bytes(ids,blend(ids,raw,student,variant,failures))
    result=dict(status='DIAGNOSTIC_ONLY' if diagnostic else 'COMPLETE_LOCAL_CANDIDATE', version=VERSION,
        registration_sha256=digest(registration(stage)),binding_sha256=pl.sha(binding),variant=variant,
        csv_sha256=hashlib.sha256(payload).hexdigest(),studies=len(ids),flip_policy=flip.policy('FLIP_F0F2'),
        baseline_sha256=base_receipt['sha256'],pl_failed_studies=len(missing),
        qualified_for_submission=False,exposure='EXPOSED_EXPLORATORY',clean_oof=False)
    filename='diagnostic.csv' if diagnostic else 'submission.csv'
    output=Path(output)
    need(not (output/filename).exists(), 'fresh publication required')
    try: safety.atomic_write(output/filename,payload,immutable=True)
    except OSError as exc: result.update(status='EXPORT_HOLD',export_error=type(exc).__name__)
    safety.json_write(output/'receipt.json',result)
    return result


def snapshot_digest(ids,raw,failures):
    import numpy as np
    return digest(dict(ids=ids,arrays={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in raw.items()},failures=failures))


def replay(output=SCRATCH/'replay'):
    import numpy as np
    from tools.exp169_c85r7ins_checks_v122 import snapshot
    output=Path(output)
    baseline=flip_build.cpu_replay(output/'base')
    ids,raw,failures=snapshot(base.c85.A5Q,output/'base/replay-input/pr7')
    expected=(output/'base'/'FLIP_F0F2'/base.DIAGNOSTIC).read_bytes()
    ident=pl.identity({'fixture':True},'own',MIL,'recorded-prediction-cpu-replay')
    journal=pl.Journal(output/'student-journal',ids,ident)
    # Authentic recorded numeric predictions as stand-ins, never teacher labels.
    for i,u in enumerate(ids):
        for e in MIL:
            journal.record(u,e['id'],raw['EXP155'][i])
    journal.finish()
    student,missing,count,reason=student_predictions(output/'student-journal',ids,ident)
    need(not missing and count==0 and reason is None, 'recorded journal replay')
    parity={}
    for v in VARIANTS:
        payload=base.p.csv_bytes(ids,blend(ids,raw,None,v,failures,pl_weight=0,diagnostic=True))
        need(payload==expected,'zero PL byte parity '+v)
        parity[v]=hashlib.sha256(payload).hexdigest()
        combined=base.p.csv_bytes(ids,blend(ids,raw,student,v,failures))
        need(combined==base.p.csv_bytes(ids,predecessor.blend(ids,raw,student,v,failures)), 'flip off v140 byte parity')
        base.write_once(output/f'diagnostic-{v}.csv',combined)
        failed={k:dict.fromkeys(ids,'injected CPU failure') for k in ['R','S_R7',*VARIANTS[v]]}
        need(np.all(blend(ids,raw,student,v,failed)==.5),'all-failure insurance')
    result=dict(status='PASS_CPU_REPLAY',studies=len(ids),zero_pl_csv_sha256=parity,
        base_sha256=baseline['diagnostic_sha256'],both_variants=True,journal_adapter=True,flip_off_v140_bytes=True,zero_pl_v130_bytes=True,
        scope='authentic recorded R/own/Q; S6 stand-in for R7; recorded EXP155 stand-in for PL; no inference',
        qualified_for_submission=False)
    dump(output/'replay.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','prepare','validate','admit','replay','runtime','bind'])
    p.add_argument('--stage',type=Path,default=OUT)
    p.add_argument('--output',type=Path)
    p.add_argument('--inputs',type=Path)
    p.add_argument('--bindings',type=Path)
    p.add_argument('--qualification',type=Path)
    p.add_argument('--review',type=Path)
    p.add_argument('--flip-go',type=Path)
    p.add_argument('--variant',choices=list(VARIANTS),default='a')
    p.add_argument('--student-seconds-per-study',type=float)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.command=='replay': result=replay(a.output or SCRATCH/'replay')
    elif a.command=='runtime': result=runtime_projection(a.student_seconds_per_study)
    elif a.command=='bind':
        need(a.inputs is not None and a.output is not None,'bind requires inputs/output')
        result=bind(a.stage,a.inputs,a.output)
    elif a.command=='admit':
        pin=lambda path: None if path is None else dict(path=str(path),sha256=pl.sha(path))
        result=admit(a.stage,a.bindings,pin(a.qualification),a.variant,pin(a.review),pin(a.flip_go))
    else: result=globals()[a.command](a.stage)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
