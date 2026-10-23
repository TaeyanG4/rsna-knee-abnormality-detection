"""Production child serving 1.0.0: use original PL adapters and v140 arithmetic.

New child only; no submitted registration or receipt is altered. Diagnostic
outputs need B's independent qualification and never imply submission approval.
"""
import importlib.util
import hashlib
import math
from pathlib import Path
from tools import pl_production_release_v1 as child
from tools import c85r7_v150_safety as safety
from tools import exp169_release_c85r7_pl15_v150 as inherited

pl=child.production_a7.adapter
base=child.release.base
need,read,dump,digest=pl.need,pl.read,pl.dump,pl.digest
OUT=child.OUT
VERSION=child.VERSION
VARIANTS=child.release.VARIANTS
registration,validate=child.registration,child.validate
snapshot_digest=inherited.snapshot_digest

# Isolated module globals allow the existing serving functions to consume the
# separately sealed seed42 projection, without mutating any imported live stack.
spec=importlib.util.spec_from_file_location('_pl_v140_seed42',child.release.__file__)
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
adapter.MIL=child.MIL
adapter.pl=pl
adapter.registration=registration
adapter.validate=validate
native_job=adapter.native_job
student_predictions=adapter.student_predictions
blend=adapter.blend


def review_gate(stage,review):
    need(review is not None,'independent review absent')
    r=read(pl.pin(review))
    need(r.get('status')=='PASS' and r.get('independent') is True and r.get('reviewer'), 'independent review required')
    need(r.get('registration_sha256')==digest(registration(stage)),'review child identity')
    kinds=set()
    for asset in r.get('smoke_receipts',[]):
        receipt=read(pl.pin(asset));kinds.add(receipt.get('kind'))
        need(receipt.get('status')=='PASS' and receipt.get('registration_sha256')==r['registration_sha256'],'review smoke identity')
        need(receipt.get('sources'),'smoke sources absent')
        for source in receipt['sources']:pl.pin(source)
    need({'suite','real_cpu_smoke'}<=kinds,'review requires CPU suite and real smoke')
    return r


def health_gate(stage,binding,image_sha256,asset):
    need(asset is not None,'full-shape memory/rate measurement absent')
    h=read(pl.pin(asset))
    need(h.get('status')=='PASS' and h.get('registration_sha256')==digest(registration(stage))
         and h.get('binding_sha256')==pl.sha(binding) and h.get('image_sha256')==image_sha256,'health identity')
    need(h.get('device')=='2xT4' and h.get('full_shape') is True and h.get('tta') is True,'full-shape TTA measurement required')
    peaks=h.get('peak_reserved_mib',[])
    need(len(peaks)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<v<=10240 for v in peaks),'reserved memory exceeds 10 GiB')
    rates=h.get('healthy_commits_per_second',{})
    need(set(rates)=={'own','Q_A5q','R','S_R7','PL-0','PL-1'} and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in rates.values()),'healthy rates absent')
    need(h.get('sources'),'health evidence absent')
    for source in h['sources']:pl.pin(source)
    return h


def compose(stage,binding,variant,a172_dir,pr7_dir,shards,run_id,output,*,base_receipt):
    """Frozen compositor with authenticated v122 baseline; diagnostic only."""
    from tools.exp169_c85r7ins_checks_v122 import snapshot
    import numpy as np
    registration(stage)
    b=read(binding);reg=pl.registration(Path(stage)/'student'/b['arm'])
    entries=pl.validate_bindings(reg,b)
    receipt=read(pl.pin(base_receipt))
    need(receipt.get('version')=='1.2.2' and receipt.get('status') in ('COMPLETE_C85R7','DIAGNOSTIC_C85R7')
         and receipt.get('release')=='C85R7+codecs+insurance','authenticated v122 baseline required')
    ids,raw,failures=snapshot(Path(a172_dir),Path(pr7_dir))
    need(receipt.get('snapshot_sha256')==snapshot_digest(ids,raw,failures),'baseline snapshot identity')
    seen=set();student=np.full((len(ids),12),.5);missing={}
    expected=pl.identity(reg,'own',entries,run_id)
    for shard in shards:
        subset=base.v5q.validate_ids(shard['ids'])
        need(set(subset)<=set(ids) and not seen.intersection(subset),'student shard cohort')
        need(shard['identity']==expected,'student checkpoint identity')
        values,bad,_,_=student_predictions(shard['path'],subset,expected)
        student[[ids.index(u) for u in subset]]=values
        seen.update(subset);missing.update(bad)
    need(seen==set(ids) and not missing,'incomplete or failed healthy PL predictions')
    payload=base.p.csv_bytes(ids,blend(ids,raw,student,variant,dict(failures,PL_STUDENT=missing)))
    result=dict(status='DIAGNOSTIC_ONLY',version=VERSION,variant=variant,studies=len(ids),
                registration_sha256=digest(registration(stage)),binding_sha256=pl.sha(binding),
                baseline_sha256=base_receipt['sha256'],csv_sha256=child.release.hashlib.sha256(payload).hexdigest(),
                qualified_for_submission=False,exposure='EXPOSED_EXPLORATORY',clean_oof=False)
    output=Path(output)
    safety.atomic_write(output/'diagnostic.csv',payload,immutable=True)
    safety.json_write(output/'receipt.json',result)
    return result
