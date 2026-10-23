"""C85-R7-PL15 1.4.0: sealed local preparation, no automatic GPU/network work.

Changelog 1.4.0: add/replace PL-MIL inside the 15% own branch; preserve
v1.2.2 codecs/insurance and reuse vNext-R7-PL 1.1.1 training/binding adapters.
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
from tools.hb116_vnext_release import package, mean_survivors

VERSION = '1.4.0'
OUT = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.4.0'
SCRATCH = ROOT/'.local/scratch/astra-b1/c85r7-pl15'
VARIANTS = {
    'a': dict(EXP143v2=17, EXP155=17, Q_A5q=12, PL_STUDENT=12),
    'b': dict(EXP143v2=17, EXP155=17, PL_STUDENT=12),
}
MIL = [e for e in pl.member_roster() if e['kind'] == 'MIL']
need, read, dump, digest = pl.need, pl.read, pl.dump, pl.digest


def payloads():
    files = pl.source_payloads()
    files['tools/'+Path(__file__).name] = Path(__file__).read_bytes()
    return files


def register(stage=OUT):
    stage = Path(stage)
    need(stage.resolve() != base.OUT.resolve(), 'frozen base stage')
    base.validate(base.OUT)
    parents = {}
    for arm in ('global', 'fill'):
        # Original registrations remain intact. The explicit MIL projection is
        # independently sealed and consumed by the existing target/train/export
        # and validate_bindings functions, which support an explicit roster.
        parent = stage/'upstream'/arm
        pl.register(parent, arm=arm)
        reg = pl.registration(parent)
        student = dict(reg, members=MIL, parent_registration_sha256=digest(reg),
                       projection='MIL-only C85-R7-PL15 v1.4.0')
        folder = stage/'student'/arm
        dump(folder/'registration.json', student, immutable=True)
        base.write_once(folder/'registration.sha256', (digest(student)+'\n').encode())
        parents[arm] = dict(parent=digest(reg), student=digest(student))
    _, pins = package(payloads())
    reg = dict(name='C85-R7-PL15', version=VERSION, status='SEALED_PLACEHOLDERS',
        base_version='1.2.2', base_registration_sha256=pl.sha(base.OUT/'registration.json'),
        outer_weights=[.85, .15], variants=VARIANTS, denominators={'a':58, 'b':46},
        public='P_R7=.5*r(R)+.5*r(S_R7)', own='weighted mean of component ranks; final rank(O)',
        pl_aggregation='native float32 mean_survivors before component rank',
        members=MIL, arm_policy='actual H-B-146 v1.1.1 registered_go.selected_arm; global or fill only',
        exposure='EXPOSED_EXPLORATORY', clean_oof=False, students=parents,
        payload_pins=pins, diagnostic_zero='both variants delegate complete v1.2.2 blend including Q',
        compute_seconds=31200, publication_reserve_seconds=1200, limit_seconds=32400,
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
    need(reg['version'] == VERSION and reg['variants'] == VARIANTS and reg['outer_weights'] == [.85,.15], 'formula drift')
    need(pl.sha(base.OUT/'registration.json') == reg['base_registration_sha256'], 'base identity')
    need(package(payloads())[1] == reg['payload_pins'], 'source drift')
    for arm in ('global','fill'):
        parent = pl.registration(stage/'upstream'/arm)
        student = pl.registration(stage/'student'/arm)
        need(student == dict(parent, members=MIL, parent_registration_sha256=digest(parent),
                             projection='MIL-only C85-R7-PL15 v1.4.0'), 'student projection drift')
        need(reg['students'][arm] == dict(parent=digest(parent),student=digest(student)), 'student seal')
    return reg


def prepare(stage=OUT):
    stage = Path(stage)
    reg = registration(stage)
    archive, _ = package(payloads())
    base.write_once(stage/'runtime.zip', archive)
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


def admit(stage=OUT, binding=None, qualification=None, variant='a'):
    validate(stage)
    need(variant in VARIANTS, 'unregistered variant')
    reasons = []
    if binding is None:
        reasons.append('real H-B-146 winning-arm GO and MIL exporter checkpoint bindings absent')
    else:
        b = read(binding)
        need(b.get('arm') in ('global','fill'), 'winning arm')
        s = pl.registration(Path(stage)/'student'/b['arm'])
        pl.validate_bindings(s,b)
    if qualification is None:
        reasons.append('C85-R7-PL15 target runtime/parity/fault qualification absent')
    else:
        need(binding is not None, 'qualification requires checkpoint bindings')
        q = read(pl.pin(qualification))
        need(q.get('registration_sha256') == digest(registration(stage)) and
             q.get('binding_sha256') == pl.sha(binding) and q.get('variant') == variant, 'qualification identity')
        need(q.get('device') == '2xT4' and all(q.get(k) is True for k in
            ('complete_ordered_csv','finite','zero_healthy_fallbacks','descendants_exited',
             'gpu_memory_released','fault_n32_pass','native_parity','codecs_best_effort')), 'target qualification')
        compute, reserve, total = (q.get(k, 1e9) for k in ('upper_seconds','publication_reserve_seconds','total_seconds'))
        need(0 < compute <= 31200 and 0 <= reserve <= 1200 and compute+reserve <= total <= 32400, 'runtime qualification')
    return dict(status='HOLD' if reasons else 'ADMITTED', reasons=reasons, qualified_for_submission=False)


def blend(ids, raw, student, variant='a', failures=None, *, pl_weight=12, diagnostic=False):
    import numpy as np
    need(variant in VARIANTS, 'unregistered variant')
    need(pl_weight == 12 or (diagnostic and pl_weight == 0), 'PL weight sealed; zero diagnostic only')
    failures = failures or {}
    if pl_weight == 0:
        return base.blend(ids, *[raw[k] for k in ('R','S_R7','EXP143v2','EXP155','Q_A5q')], failures=failures)
    ids = base.v5q.validate_ids(ids)
    need(student is not None, 'PL student absent')
    values = dict(raw, PL_STUDENT=student)
    names = ['R','S_R7',*VARIANTS[variant]]
    arrays = {k:base.v5q._probabilities(k,values[k],len(ids)) for k in names}
    masks = {k:np.asarray([u in failures.get(k,{}) for u in ids]) for k in names}
    masks['R'] |= np.all(arrays['R']==.5,axis=1)
    ranks = {k:base.c85.scored_rank(arrays[k],masks[k]) for k in names}
    rf,sf=masks['R'],masks['S_R7']
    public=.5*ranks['R']+.5*ranks['S_R7']
    public[rf],public[sf],public[rf&sf]=ranks['S_R7'][rf],ranks['R'][sf],.5
    weights=VARIANTS[variant]
    den=sum(w*(~masks[k]).astype(int) for k,w in weights.items())
    own=sum(w*ranks[k] for k,w in weights.items())/np.maximum(den,1)[:,None]
    pf,of=rf&sf,den==0
    pr,oo=base.c85.scored_rank(public,pf),base.c85.scored_rank(own,of)
    result=.85*pr+.15*oo
    result[of],result[pf],result[pf&of]=pr[of],oo[pf],.5
    return result


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


def publish(stage, binding, qualification, variant, a172_dir, pr7_dir, shards, run_id, output):
    """Authenticated final compositor; absent inputs block before any CSV write.

    B supplies shards constructed by native_job and executed through the existing
    PL worker under the SAME base Deadline/Supervisor. Only the final filename
    here is a candidate submission; base inference outputs stay in a subfolder.
    """
    from tools.exp169_c85r7ins_checks_v122 import snapshot
    import numpy as np
    need(admit(stage,binding,qualification,variant)['status']=='ADMITTED','production admission HOLD')
    output=Path(output)
    need(not output.exists(),'fresh publication directory required')
    b=read(binding)
    reg=pl.registration(Path(stage)/'student'/b['arm'])
    entries=pl.validate_bindings(reg,b)
    ids,raw,failures=snapshot(Path(a172_dir),Path(pr7_dir))
    need(len(ids)==len(set(ids)),'duplicate base studies')
    seen=set();student=np.full((len(ids),12),.5);missing={}
    expected=pl.identity(reg,'own',entries,run_id)
    for shard in shards:
        subset=base.v5q.validate_ids(shard['ids'])
        need(set(subset)<=set(ids) and not seen.intersection(subset),'student shard cohort')
        need(shard['identity']==expected,'student checkpoint/run identity')
        values,bad,_,_=student_predictions(shard['path'],subset,expected)
        student[[ids.index(u) for u in subset]]=values
        seen.update(subset);missing.update(bad)
    need(seen==set(ids),'incomplete student shard assignment')
    need(len(missing)<len(ids),'PL student unavailable for entire cohort')
    failures=dict(failures,PL_STUDENT=missing)
    values=blend(ids,raw,student,variant,failures)
    payload=base.p.csv_bytes(ids,values)
    result=dict(status='COMPLETE_LOCAL_CANDIDATE',version=VERSION,variant=variant,
        registration_sha256=digest(registration(stage)),binding_sha256=pl.sha(binding),
        qualification_sha256=qualification['sha256'],studies=len(ids),
        csv_sha256=hashlib.sha256(payload).hexdigest(),pl_failed_studies=len(missing),
        qualified_for_submission=False,exposure='EXPOSED_EXPLORATORY')
    # Immutable atomic write supplied by the v1.2.2 metadata/insurance layer.
    base.write_once(output/'submission.csv',payload)
    dump(output/'receipt.json',result,immutable=True)
    return result


def runtime_projection(student_seconds_per_study=None):
    need(student_seconds_per_study is None or student_seconds_per_study >= 0, 'negative serving estimate')
    seconds=4.7*3600
    return dict(status='UNMEASURED_PL_HOLD',base_hours=4.7,N=1300,limit_hours=9,
        compute_seconds=31200,reserve_seconds=1200,base_seconds=seconds,
        max_student_seconds_per_study_with_reserve=(31200-seconds)/1300,
        max_student_seconds_per_study_hard_limit=(32400-seconds)/1300,
        student_seconds_per_study=student_seconds_per_study,
        projected_compute_hours=None if student_seconds_per_study is None else (seconds+1300*student_seconds_per_study)/3600,
        assumptions='whole 10-checkpoint student ensemble; setup/codecs/cleanup included in increment; no Q-removal speed credit',
        scenarios=[dict(student_s_per_study=s,total_compute_hours=(seconds+1300*s)/3600,
            fits_compute_budget=seconds+1300*s<=31200) for s in (2,4,8,10,12)])


def replay(output=SCRATCH/'replay'):
    import numpy as np
    from tools.exp169_c85r7ins_checks_v122 import snapshot
    output=Path(output)
    baseline=base.cpu_replay(output/'base')
    ids,raw,failures=snapshot(base.c85.A5Q,output/'base/replay-input/pr7')
    expected=(output/'base'/base.DIAGNOSTIC).read_bytes()
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
        base.write_once(output/f'diagnostic-{v}.csv',base.p.csv_bytes(ids,blend(ids,raw,student,v,failures)))
        failed={k:dict.fromkeys(ids,'injected CPU failure') for k in ['R','S_R7',*VARIANTS[v]]}
        need(np.all(blend(ids,raw,student,v,failed)==.5),'all-failure insurance')
    result=dict(status='PASS_CPU_REPLAY',studies=len(ids),zero_pl_csv_sha256=parity,
        base_sha256=baseline['diagnostic_sha256'],both_variants=True,journal_adapter=True,
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
    p.add_argument('--variant',choices=list(VARIANTS),default='a')
    a=p.parse_args()
    if a.command=='replay': result=replay(a.output or SCRATCH/'replay')
    elif a.command=='runtime': result=runtime_projection()
    elif a.command=='bind':
        need(a.inputs is not None and a.output is not None,'bind requires inputs/output')
        result=bind(a.stage,a.inputs,a.output)
    elif a.command=='admit': result=admit(a.stage,a.bindings,variant=a.variant)
    else: result=globals()[a.command](a.stage)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
