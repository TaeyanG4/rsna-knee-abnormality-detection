"""H-B-106 coverage build/replay/evidence/export utility v1.0.0.

Changelog: isolated EXP185 sampler contrast; create-only artifacts, no launcher.
"""
from __future__ import annotations
import argparse
from collections import Counter
import os
from pathlib import Path
import shutil

from training.arms import lifecycle as life, mil, mil_coverage as cov


def save(path, value):
    life.create_new(path, life.canonical_bytes(value))


def build(root):
    from tools import exp180_k12_coverage as audit
    root = Path(root)
    base = mil.read(root/'training/configs/mil-s42-f0-v3.json')
    ids = mil.read(root/base['config']['roots']['ordered'])['train']
    flags = audit.read_series_flags(root/audit.SERIES.relative_to(audit.ROOT), set(ids))
    _, data, info = audit.load_raptor(root/base['config']['roots']['corpus'], set(ids), flags)
    save(root/cov.METADATA, dict(schema='mil-coverage-metadata/v1',
        source_sha256=info['metadata_sha256'], series_sha256=mil.sha(audit.SERIES),
        train_sha256=mil.uid_hash(ids), studies={u:dict(mask_sha256=life.content_sha256(data[u][0].astype(int).tolist()),
        slots=data[u][1]) for u in ids}))
    result = {}
    for seed in (42,43):
        d = mil.read(root/f'training/configs/mil-s{seed}-f0-v3.json')
        control = f'artifacts/experiments/EXP185_FINDING_MIL/outputs/S{seed}/results/EXP185_FINDING_MIL_F0_S{seed}'
        hashes = {n:mil.sha(root/control/n) for n in ('summary.json','provenance.json')}
        d.update(arm=f'MIL_COVERAGE_S{seed}_F0', attempt='v1', target_environment=life.environment_from_control(
            root/control/'summary.json', root/control/'provenance.json', expected_hashes=hashes))
        d['config']['sampler'] = {**mil.SAMPLER, **cov.PRESET}
        protocol = f'.local/exp185-overlay/protocols/S{seed}.json'
        d['inputs']['controls'] = {f'{control}/{n}':h for n,h in hashes.items()}
        d['inputs']['controls'][protocol] = mil.sha(root/protocol)
        d['inputs']['data'][cov.METADATA] = mil.sha(root/cov.METADATA)
        d['inputs']['sources'] = {p:mil.sha(root/p) for p in mil.SOURCE_FILES+cov.EXTRA_SOURCES}
        spec = life.ArmSpec.from_document(d)
        cov.validate_data(spec, root, root/d['config']['roots']['corpus'])
        path = root/f'training/configs/mil-coverage-s{seed}-f0-v1.json'
        life.create_new(path, spec.encoded)
        result[str(seed)] = dict(spec_sha256=spec.sha256, control_seconds=mil.read(root/control/'summary.json')['total_seconds'])
    return dict(status='BUILT_VALIDATED', seeds=result)


def replay(root, out):
    """Same per-study/epoch windows RNG for both arms, all eligible f0 draws.

    This deliberately does not claim to replay drop_last's consumed visitation.
    """
    import numpy as np
    from tools import exp180_k12_coverage as audit
    root = Path(root)
    d = mil.read(root/'training/configs/mil-coverage-s42-f0-v1.json')
    corpus = mil.common.Corpus(root/d['config']['roots']['corpus'])
    rows = mil.read(root/cov.METADATA)['studies']
    results = {}
    for seed in (42,43):
        counts = {'control':Counter(), 'coverage':Counter()}
        rng_paired = True
        for uid, row in rows.items():
            mask, slots = corpus.mask(uid), row['slots']
            available = audit.availability(mask, slots)
            for epoch in range(20):
                a = mil.common.study_rng(seed,epoch,uid)
                b = mil.common.study_rng(seed,epoch,uid)
                bags = {'control':mil.common.select_centers(mask,64,12,corpus.spec,'within-slot-valid',a),
                        'coverage':cov.coverage_centres(mask,slots,12,b)}
                rng_paired &= a.getstate() == b.getstate()
                for name,bag in bags.items():
                    event,_,_ = audit.audit_bag(bag,mask,slots,available)
                    counts[name].update(event)
                    counts[name]['bags'] += 1
        assert rng_paired and counts['coverage']['bags_omitting_any'] == 0
        assert counts['coverage']['duplicate_centres'] == 0
        results[str(seed)] = {name:{**dict(c), 'omission_rate':c['bags_omitting_any']/c['bags']} for name,c in counts.items()}
        results[str(seed)]['augmentation_rng_matches_control'] = rng_paired
    result = dict(schema='hb106-replay/v1', scope='all fold0 eligible study draws; not drop_last consumed bags',
                  epochs=20, metadata_sha256=mil.sha(root/cov.METADATA), seeds=results)
    save(out,result)
    return result


def combine(spec, cpu, target, dest, reviewer_name, reviewer_host):
    """B calls only after independently reviewing both immutable HOLD records."""
    original = {}
    for kind,root in (('cpu',cpu),('target',target)):
        note = mil.read(root/'verification.json')
        life.verify_evidence(spec,note,root)
        assert note['verdict']=='HOLD' and note['smokes'][kind] is not None
        assert note['smokes']['target' if kind=='cpu' else 'cpu'] is None
        original[kind]=note
    dest.mkdir(parents=True,exist_ok=False)
    checks,smokes=[],{}
    for kind,root in (('cpu',cpu),('target',target)):
        note=original[kind]
        for record,role in ((note['checks'][0],'check'),(note['smokes'][kind],'smoke')):
            log=mil.read(root/'evidence'/(record['log_sha256']+'.json'))
            for artifact in log['artifacts'].values():
                member=kind+'/'+artifact['path']
                source,destination=root/artifact['path'],dest/member
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    try: os.link(source,destination)
                    except OSError: shutil.copyfile(source,destination)
                assert mil.sha(destination)==artifact['sha256']
                artifact['path']=member
            digest=life.content_sha256(log)
            save(dest/'evidence'/(digest+'.json'),log)
            updated=dict(record,log_sha256=digest)
            if role=='check':
                updated['name']=kind+'-executed-family-smoke'; checks.append(updated)
            else: smokes[kind]=updated
    merged=dict(schema=life.VERIFICATION_SCHEMA,scope=spec.scope,
        reviewer=dict(name=reviewer_name,host=reviewer_host),verdict='PASS',blockers=[],
        notes=['Independent B review of CPU and CUDA evidence'],bindings=spec.bindings,checks=checks,smokes=smokes)
    life.verify_evidence(spec,merged,dest)
    save(dest/'verification-pass.json',merged)
    return dict(status='PASS_EVIDENCE_MERGED',verification_sha256=life.content_sha256(merged))


def finish(spec, output, pass_note, archive, receipt):
    note=mil.read(pass_note)
    life.verify_evidence(spec,note,pass_note.parent)
    assert note['verdict']=='PASS'
    summary,prov,history=[mil.read(output/n) for n in ('summary.json','provenance.json','history.json')]
    c=spec.document['config']
    assert summary['status']=='FIXED_FINAL_COMPLETE' and summary['spec_sha256']==spec.sha256
    assert summary['epochs']==20 and summary['attempted_updates']==c['updates']['steps']
    assert summary['train']==c['roots']['n_train'] and summary['val']==c['roots']['n_val']
    assert summary['skipped_updates']<=c['updates']['max_skips']
    assert [h['epoch'] for h in history]==list(range(20))
    assert prov['verification_sha256']==life.content_sha256(note)
    assert prov['checkpoint_retention']=='fixed-final-only'
    metrics={k:summary[k] for k in ('attempted_updates','skipped_updates','val')}
    obj=life.artifact_receipt(spec,output,verification_sha256=life.content_sha256(note),exit_code=0,
                              elapsed_seconds=summary['elapsed_seconds'],metrics=metrics)
    save(receipt,obj)
    digest=life.content_sha256(obj)
    archive_sha=life.export_artifacts(spec,output,obj,archive,expected_receipt_sha256=digest)
    return dict(status='EXPORTED_ARTIFACTS',receipt_sha256=digest,archive_sha256=archive_sha)


def screen_specs(root, collected, out, preflight=False):
    root, collected, out=map(Path,(root,collected,out))
    result={}
    for name in ('oof_best','oof_k36','oof_k36_flip_tta'):
        seeds={}
        for seed in (42,43):
            control=root/f'artifacts/experiments/EXP185_FINDING_MIL/outputs/S{seed}/results/EXP185_FINDING_MIL_F0_S{seed}/{name}.csv'
            arm=control if preflight else collected/f's{seed}/outputs/{name}.csv'
            seeds[str(seed)]={side:[dict(path=str(p.resolve()),sha256=mil.sha(p))] for side,p in (('arm',arm),('control',control))}
        path=out/(name+'.json')
        save(path,dict(name='H-B-106 '+name+(' CONTROL_SELF_PREFLIGHT' if preflight else ''),family='convnext',seeds=seeds))
        result[name]=str(path)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('build','replay','combine','finish','screen-specs'))
    ap.add_argument('--root',type=Path,default=Path.cwd())
    for n in ('out','spec','cpu','target','outputs','pass-note','archive','receipt','collected'):
        ap.add_argument('--'+n,type=Path)
    ap.add_argument('--reviewer-name'); ap.add_argument('--reviewer-host')
    ap.add_argument('--preflight',action='store_true')
    a=ap.parse_args()
    if a.command=='build': result=build(a.root)
    elif a.command=='replay': result=replay(a.root,a.out)
    elif a.command=='screen-specs': result=screen_specs(a.root,a.collected,a.out,a.preflight)
    else:
        spec=life.ArmSpec(a.spec.read_bytes())
        if a.command=='combine': result=combine(spec,a.cpu,a.target,a.out,a.reviewer_name,a.reviewer_host)
        else: result=finish(spec,a.outputs,a.pass_note,a.archive,a.receipt)
    print(life.canonical_bytes(result).decode())


if __name__=='__main__': main()
