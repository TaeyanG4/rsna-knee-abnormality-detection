"""H-B-113 public-R CLI v1.0.0: reviewed member and serial receipt/export/collect."""
import argparse
import json
from pathlib import Path
import sys
import shutil
from training.arms import rplus_public as c
from training.arms import rplus_public_fit as fit
from training.arms import lifecycle as life


from training.runners.coatnet_lifecycle import combine


def pack(spec,destination,note,evidence_root):
    """Serial CPU packaging only after all training/readout assertions succeed.

    Failure propagates; no success receipt is emitted until collection verifies.
    No process launch, timer, retry, release comparison, or submission.
    """
    out=Path(destination)
    digest=life.verify_evidence(spec,note,evidence_root)
    summary=c.read(out/'outputs/summary.json')
    c.need(summary['status'] in ('ENGINEERING_SMOKE_COMPLETE','COMPLETE_TRAINING_AND_READOUT'),'fit not complete')
    c.need(summary['provenance_sha256']==life.file_sha256(out/'outputs/provenance.json'),'summary provenance differs')
    provenance=c.read(out/'outputs/provenance.json')
    c.need(provenance['spec_sha256']==spec.sha256 and provenance['scientific']==spec.document['config'],'fit binding differs')
    receipt=life.artifact_receipt(spec,out/'outputs',verification_sha256=digest,exit_code=0,
        elapsed_seconds=summary['seconds'],metrics=dict(updates=summary['updates'],skips=summary['skips']))
    c.write(out/'artifact-receipt.json',receipt)
    receipt_sha=life.content_sha256(receipt)
    archive_sha=life.export_artifacts(spec,out/'outputs',receipt,out/'artifacts.tar',expected_receipt_sha256=receipt_sha)
    collected=life.collect_artifacts(spec,out/'artifacts.tar',out/'collected',expected_receipt_sha256=receipt_sha)
    serial=dict(schema='rplus-public-serial/v1',status='SERIAL_COMPLETE',spec_sha256=spec.sha256,
        scientific_scope=summary['scientific_scope'],ancestry_scope=c.SCOPE,
        verification_sha256=digest,artifact_receipt_sha256=receipt_sha,archive_sha256=archive_sha,
        collection_sha256=life.content_sha256(collected),steps=['fit','receipt','export','collect'],launch_authorized=False)
    c.write(out/'serial-receipt.json',serial)
    return serial


def parser():
    p=argparse.ArgumentParser(__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    observe=sub.add_parser('observe'); observe.add_argument('--device',choices=['cpu'],required=True); observe.add_argument('--out',required=True)
    reg=sub.add_parser('register')
    for name in ('root','out'): reg.add_argument('--'+name,required=True)
    for cmd in ('validate','prepare','smoke','train','admit','receipt','export','collect','combine'):
        q=sub.add_parser(cmd); q.add_argument('--spec',required=True)
        if cmd=='validate':
            q.add_argument('--root',required=True); q.add_argument('--staged',action='store_true')
        elif cmd=='prepare':
            q.add_argument('--root',required=True); q.add_argument('--out',required=True)
        elif cmd in ('smoke','train'):
            for name in ('stage','corpus','parent','out'): q.add_argument('--'+name,required=True)
            q.add_argument('--device',choices=['cpu','cuda'],required=True)
            q.add_argument('--workers',type=int,default=0); q.add_argument('--serial',action='store_true')
            q.add_argument('--review'); q.add_argument('--review-root')
            q.add_argument('--note'); q.add_argument('--evidence-root')
        elif cmd=='admit':
            for name in ('stage','note','evidence-root'): q.add_argument('--'+name,required=True)
            q.add_argument('--review'); q.add_argument('--review-root')
        elif cmd=='receipt':
            for name in ('outputs','note','evidence-root','out'): q.add_argument('--'+name,required=True)
        elif cmd in ('export','collect'):
            for name in ('archive','receipt-sha256'): q.add_argument('--'+name,required=True)
            if cmd=='export':
                q.add_argument('--outputs',required=True); q.add_argument('--receipt',required=True)
            else: q.add_argument('--out',required=True)
        elif cmd=='combine':
            for name in ('cpu','target','out'): q.add_argument('--'+name,required=True)
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    try:
        if a.command=='observe':
            fit.configure(); observed=fit.observation(a.device); c.write(a.out,observed)
            result=dict(status='OBSERVATION_RECORDED',sha256=life.file_sha256(a.out))
        elif a.command=='register':
            spec=c.register(a.root)
            life.create_new(a.out,spec.encoded)
            result=dict(status='REGISTERED',spec_sha256=spec.sha256)
        else:
            spec=c.Spec(Path(a.spec).read_bytes())
            if a.command=='validate':
                (c.validate_stage if a.staged else c.validate_inputs)(spec,a.root)
                result=dict(status='VALIDATED',spec_sha256=spec.sha256)
            elif a.command=='prepare': result=c.prepare(spec,a.root,a.out)
            elif a.command in ('smoke','train'):
                summary=fit.run(spec,a.stage,a.corpus,a.out,mode='smoke' if a.command=='smoke' else 'full',
                    parent=a.parent,device=a.device,workers=a.workers,note=c.read(a.note) if a.note else None,evidence_root=a.evidence_root,
                    review=c.read(a.review) if a.review else None,review_root=a.review_root)
                if a.command=='smoke':
                    fit.record_smoke(spec,a.out,summary,[sys.executable,'-B','-m','training.runners.rplus_public',*(argv or sys.argv[1:])])
                result=dict(status=summary['status'],updates=summary['updates'],results=summary['results'])
                if a.serial:
                    packet=Path(a.out) if a.command=='smoke' else Path(a.evidence_root)
                    note_path=Path(a.out)/'verification.json' if a.command=='smoke' else Path(a.note)
                    result['serial']=pack(spec,a.out,c.read(note_path),packet)
            elif a.command=='admit':
                result=c.admit(spec,a.stage,c.read(a.note),a.evidence_root,
                    c.read(a.review) if a.review else None,a.review_root)
            elif a.command=='receipt':
                note=c.read(a.note)
                digest=life.verify_evidence(spec,note,a.evidence_root)
                summary=c.read(Path(a.outputs)/'summary.json')
                result=life.artifact_receipt(spec,a.outputs,verification_sha256=digest,exit_code=0,
                    elapsed_seconds=summary['seconds'],metrics=dict(updates=summary['updates'],skips=summary['skips']))
                c.write(a.out,result)
            elif a.command=='export':
                digest=life.export_artifacts(spec,a.outputs,c.read(a.receipt),a.archive,expected_receipt_sha256=a.receipt_sha256)
                result=dict(status='EXPORTED',archive_sha256=digest)
            elif a.command=='collect':
                result=life.collect_artifacts(spec,a.archive,a.out,expected_receipt_sha256=a.receipt_sha256)
            elif a.command=='combine': result=combine(spec,a.cpu,a.target,a.out)
        print(json.dumps(dict(result,launch_authorized=False),allow_nan=False))
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,AssertionError) as exc:
        print(json.dumps(dict(status='REJECTED',launch_authorized=False,error=str(exc))))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
