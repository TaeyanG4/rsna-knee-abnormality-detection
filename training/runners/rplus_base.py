"""R+ CLI v1.1.0: immutable register/stage/fit/evidence/collect operations."""
import argparse
import json
from pathlib import Path
import sys
import shutil
from training.arms import rplus_base as c
from training.arms import rplus_base_fit as fit
from training.arms import lifecycle as life


from training.runners.coatnet_lifecycle import combine


def parser():
    p=argparse.ArgumentParser(__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    observe=sub.add_parser('observe'); observe.add_argument('--device',choices=['cpu','cuda'],required=True); observe.add_argument('--out',required=True)
    reg=sub.add_parser('register')
    for name in ('root','out','arm','attempt'):
        reg.add_argument('--'+name,required=True)
    reg.add_argument('--controls',required=True,help='JSON path -> SHA mapping for genuine completed corresponding control')
    for cmd in ('validate','prepare','smoke','train','admit','receipt','export','collect','combine'):
        q=sub.add_parser(cmd); q.add_argument('--spec',required=True)
        if cmd=='validate':
            q.add_argument('--root',required=True); q.add_argument('--staged',action='store_true')
        elif cmd=='prepare':
            q.add_argument('--root',required=True); q.add_argument('--out',required=True)
        elif cmd in ('smoke','train'):
            for name in ('stage','corpus','oai-corpus','out'): q.add_argument('--'+name,required=True)
            q.add_argument('--device',choices=['cpu','cuda'],required=True)
            q.add_argument('--workers',type=int,default=0); q.add_argument('--resume')
            q.add_argument('--note'); q.add_argument('--evidence-root')
        elif cmd=='admit':
            for name in ('stage','note','evidence-root'): q.add_argument('--'+name,required=True)
            q.add_argument('--live',action='store_true',help='Collect fresh CUDA observation on target; B only')
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
            spec=c.register(a.root,arm=a.arm,attempt=a.attempt,controls=c.read(a.controls))
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
                    oai_corpus=a.oai_corpus,device=a.device,workers=a.workers,resume=a.resume,note=c.read(a.note) if a.note else None,evidence_root=a.evidence_root)
                if a.command=='smoke':
                    fit.record_smoke(spec,a.out,summary,[sys.executable,'-B','-m','training.runners.rplus_base',*(argv or sys.argv[1:])])
                result=dict(status=summary['status'],updates=summary['updates'],results=summary['results'])
            elif a.command=='admit':
                fit.configure()
                result=c.admit(spec,a.stage,c.read(a.note),a.evidence_root,fit.observation('cuda') if a.live else None)
                if a.live: result['status']='LIVE_TARGET_EVIDENCE_READY'
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
