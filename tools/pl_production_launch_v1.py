"""Production launch metadata v1.1.0. Changelog: seeded production loader; new work root. Previously: GO/review/memory gates and fivefold orchestration."""
from pathlib import Path
import argparse
import json
from tools import vnext_r7_pl_v1 as p
from tools import vnext_r7_pl_go_v111 as go_adapter
from tools import pl_production_release_v1 as release
from tools import pl_production_teacher_v1 as teacher
from tools import pl_production_targets_v1 as targets
from tools import pl_production_train_v1 as train
from tools import pl_production_watchdog_v1 as watchdog

WORK=p.ROOT/'.local/scratch/astra-b4/PL-PRODUCTION-v11'
GO=p.ROOT/'.local/scratch/astra-b4/H-B-146/launch-v115/readout.json'
SOURCE=p.ROOT/'.local/scratch/astra-b5/H-B-145/full-v113-ready'
SOURCES=[*sorted((p.ROOT/'tools').glob('pl_production*_v1.py')),
         p.ROOT/'.local/run-scripts/pl-production-v1.ps1']

def asset(path):
    return train.base.asset(path)

def selected(go):
    p.need(Path(go).is_file(),'H-B-146 GO absent')
    arm=p.read(go).get('registered_go',{}).get('selected_arm')
    p.need(arm in ('global','fill'),'registered GO winning arm absent')
    go_adapter.validate(asset(go),arm)
    return arm

def review_gate(path):
    p.need(path is not None and Path(path).is_file(),'independent production review absent')
    r=p.read(path)
    p.need(r.get('status')=='PASS' and r.get('independent') is True and r.get('reviewer'), 'independent production review required')
    p.need(r.get('sources')==[asset(x) for x in SOURCES],'review source binding changed')
    kinds=set()
    for pin in r.get('smoke_receipts',[]):
        s=p.read(p.pin(pin)); kinds.add(s.get('kind'))
        p.need(s.get('status')=='PASS' and s.get('sources')==r['sources'],'review smoke source binding')
    p.need({'suite','real_cpu_smoke'}<=kinds,'review requires suite and real CPU smoke')
    return r

def record_review(reviewer,smokes,output):
    p.need(reviewer and smokes,'reviewer and reviewed receipts required')
    r=dict(version='1.1.0',status='PASS',independent=True,reviewer=reviewer,
           sources=[asset(x) for x in SOURCES],smoke_receipts=[asset(x) for x in smokes])
    # Validate before writing an immutable human assertion.
    kinds=set()
    for pin in r['smoke_receipts']:
        s=p.read(p.pin(pin)); kinds.add(s.get('kind'))
        p.need(s.get('status')=='PASS' and s.get('sources')==r['sources'],'review smoke source binding')
    p.need({'suite','real_cpu_smoke'}<=kinds,'review requires suite and real CPU smoke')
    p.dump(output,r,immutable=True)
    return dict(status='REVIEW_RECORDED')

def prepare(go,memory,review):
    arm=selected(go)
    review_gate(review)
    watchdog.validate_memory(memory,go=go,arm=arm)
    release.register()
    release.validate()
    seal=WORK/'teacher-heldout.json'
    teacher.seal(SOURCE,seal,exposed_exploratory=True)
    teacher.validate(seal)
    members=[]
    for member in release.MIL:
        folder=WORK/arm/member['id']
        bundle=folder/'targets'
        if not (bundle/'target-spec.json').exists():
            targets.prepare(release.OUT/'student'/arm,member['id'],seal,bundle)
        with train.adapters():
            spec=folder/'training-spec.json'
            if not spec.exists():
                train.base.prepare(release.OUT/'student'/arm,bundle,folder,member['id'])
            train.base.validate(spec)
            admission=train.base.admit(spec,go)
            p.need(admission['status']=='ADMITTED','production admission failed')
        members.append(dict(member=member,spec=str(spec),output=str(folder/'fit')))
    plan=dict(version='1.1.0',arm=arm,go=asset(go),memory=asset(memory),review=asset(review),
              teacher=asset(seal),stage=str(release.OUT),members=members,
              exposure='EXPOSED_EXPLORATORY',clean_oof=False)
    p.dump(WORK/'launch-plan.json',plan,immutable=True)
    return dict(status='PREPARED',arm=arm,folds=5)

def completed(entry,plan):
    output=Path(entry['output']);path=output/'training-receipt.json'
    if not path.exists(): return False
    receipt=p.read(path)
    p.need(receipt.get('status')=='COMPLETE' and receipt.get('checkpoint_sha256')==p.sha(output/'model_best.pt')
           and receipt.get('teacher_sha256')==plan['teacher']['sha256'],'completed fold binding')
    with train.adapters():
        spec,_,_=train.base.validate(entry['spec'])
    p.need(receipt.get('spec_sha256')==spec['spec_sha256'],'completed fold spec changed')
    go_adapter.admission(receipt.get('go'),plan['arm'],spec['spec_sha256'])
    return True

def bind():
    plan=p.read(WORK/'launch-plan.json')
    for key in ('go','memory','review','teacher'): p.pin(plan[key])
    review_gate(plan['review']['path']);selected(plan['go']['path'])
    reg=p.registration(release.OUT/'student'/plan['arm'])
    entries=[]
    for entry in plan['members']:
        p.need(completed(entry,plan),'incomplete production fold')
        entries.append(dict(entry['member'],**asset(Path(entry['output'])/'model_best.pt'),
                            training_receipt=asset(Path(entry['output'])/'training-receipt.json')))
    inputs=WORK/'binding-inputs.json'
    binding=p.read(release.OUT/'student'/plan['arm']/'bindings-template.json')
    binding.update(registration_sha256=p.digest(reg),arm=plan['arm'],go=plan['go'],
                   teacher=plan['teacher'],members=entries)
    p.dump(inputs,binding,immutable=True)
    result=release.bind_completed(release.OUT,inputs,release.OUT/'bindings.json')
    for variant,weights in release.release.VARIANTS.items():
        p.dump(release.OUT/('production-variant-'+variant+'.json'),dict(variant=variant,weights=weights,
               bindings=asset(release.OUT/'bindings.json'),exposure='EXPOSED_EXPLORATORY',clean_oof=False),immutable=True)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','review','bind','completed'])
    ap.add_argument('--go',type=Path,default=GO);ap.add_argument('--memory',type=Path)
    ap.add_argument('--review',type=Path);ap.add_argument('--reviewer');ap.add_argument('--output',type=Path)
    ap.add_argument('--smoke',type=Path,action='append',default=[]);ap.add_argument('--fold',type=int)
    a=ap.parse_args()
    if a.command=='prepare': result=prepare(a.go,a.memory,a.review)
    elif a.command=='review': result=record_review(a.reviewer,a.smoke,a.output)
    elif a.command=='bind':result=bind()
    else:
        plan=p.read(WORK/'launch-plan.json');result=dict(complete=completed(plan['members'][a.fold],plan))
    print(json.dumps(result))

if __name__=='__main__':main()
