"""Private K2 failure qualification; labels/model selection are never read.

Full available train-cohort header/codec scan, then deterministic engineering
sample: 8 largest DICOM counts + 24 hash-selected remaining studies. No AUC,
ratios, threshold selection or successful prediction fallback. Never submit
this qualification-only notebook. B owns launch and result review.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time


def select_studies(counts):
    if len(counts)<32 or any(n<=0 for n in counts.values()):
        raise ValueError('at least32 nonempty raw studies required')
    largest=sorted(counts,key=lambda u:(-counts[u],u))[:8]
    remainder=sorted(set(counts)-set(largest),key=lambda u:hashlib.sha256(
        ('K2-failure2-engineering-v1\0'+u).encode()).hexdigest())[:24]
    return largest+remainder


def run(worker, assets, manifest_sha, data, out):
    data,out=Path(data),Path(out)
    if out.exists():raise ValueError('qualification output already exists')
    out.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('k2_repaired_worker',worker)
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    wd=w.Watchdog(out/'header-status',out/'header-failure.json',True)
    started=time.perf_counter()
    try:
        counts={}
        # No train.csv/report/label access: directory names and raw DICOMs only.
        for study in sorted((data/'train_series').iterdir()):
            if study.is_dir():
                counts[study.name]=sum(1 for _ in study.glob('*/*.dcm'))
                wd.progress('roster-directory-complete')
        ids=sorted(counts)
        w.need(ids and all(counts.values()),'empty available train cohort')
        evidence=w.codec_evidence(data,'train',ids,wd.progress)
        evidence.update(qualification_only=True,studies=len(ids),
            total_seconds=time.perf_counter()-started,roster_sha256=w.digest(ids),
            worker_sha256=w.sha(worker),qualified_for_submission=False)
        w.write(out/'whole-cohort-codecs.json',evidence)
    finally:wd.close()
    selected=select_studies(counts)
    template=out/'sample-template.csv'
    with template.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow([w.UID]+w.LABELS)
        writer.writerows([[uid]+[0]*12 for uid in selected])
    w.write(out/'selection.json',dict(qualification_only=True,studies=32,
        rule='8-largest-dicom-count then24-sha256-K2-failure2-engineering-v1',
        available=len(counts),roster_sha256=w.digest(selected),
        selected_count_range=[min(counts[u] for u in selected),max(counts[u] for u in selected)],
        counts=[counts[u] for u in selected],qualified_for_submission=False))
    dest=out/'inference32';dest.mkdir()
    cmd=[sys.executable,'-B','-u',str(worker),'--assets',str(assets),'--manifest-sha',manifest_sha,
        '--data',str(data),'--template',str(template),'--output',str(dest/'submission.csv'),
        '--scratch',str(dest/'scratch'),'--device','cuda:0','--split','train']
    result=subprocess.run(cmd,check=False)
    if result.returncode:raise RuntimeError('qualification32 worker failed; inspect exact receipt')
    rec=w.read(dest/'worker-receipt.json')
    w.need(rec['status']=='COMPLETE' and rec['study_count']==32 and rec['prediction_count']==32,
        'incomplete qualification32')
    w.need(rec['fallbacks']['prediction']==0 and all(s['error'] is None for s in rec['studies']),
        'qualification32 prediction failure')
    w.write(out/'qualification-result.json',dict(status='COMPLETE',qualification_only=True,
        worker_sha256=w.sha(worker),worker_receipt_sha256=w.sha(dest/'worker-receipt.json'),
        preflight_sha256=w.sha(out/'whole-cohort-codecs.json'),studies=32,
        total_seconds=time.perf_counter()-started,qualified_for_submission=False))
    print(json.dumps({'qualification_only':True,'status':'COMPLETE','studies':32}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('worker','assets','manifest-sha','data','out'):p.add_argument('--'+key,required=True)
    a=p.parse_args();run(a.worker,a.assets,a.manifest_sha,a.data,a.out)
