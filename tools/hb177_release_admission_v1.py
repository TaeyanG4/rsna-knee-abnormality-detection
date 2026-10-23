"""HB177 release admission v1.0.0: checkpoint-bound fit review and measured runtime.

Changelog: fail closed on absent fit approval or actual HB156 hidden duration.
This is offline evidence binding, never a launch, submission or score reader.
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import math
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import hb177_postfit_v1 as b
VERSION='1.0.0'


def safe(path,root=ROOT):
    p=(root/Path(path)).resolve()
    b.need(p.is_relative_to(root.resolve()) and not any(x.lower() in
        {'.ssh','.kaggle','.codex','api-keys','relabel'} for x in p.parts),'unsafe evidence path')
    return p


def fit_bind(report,reviewer):
    report=safe(report)
    b.need(isinstance(reviewer,str) and reviewer.strip() and reviewer!='hb177-postfit','independent reviewer required')
    text=report.read_text(encoding='utf-8')
    b.need(re.search(r'(?m)^Verdict: PASS\s*$',text) and b.CHECKPOINT_SHA in text,
           'checkpoint-bound Verdict: PASS required')
    value=dict(verdict='PASS',reviewer=reviewer,report=report.relative_to(ROOT).as_posix(),
        sha256=b.sha(report),checkpoint_sha256=b.CHECKPOINT_SHA)
    save(b.OUT/'fit-result-review-binding.json',value)
    return dict(status='FIT_REVIEW_BOUND')


def save(path,value):
    if path.exists(): b.need(b.read(path)==value,'immutable evidence binding differs')
    else: b.write(path,b.encoded(value))


def runtime_bind(evidence):
    evidence=safe(evidence); rec=b.read(evidence)
    runtime_fields(rec)
    smoke=b.read(b.OUT/'cpu-smoke.json')
    ratio=smoke['serving_ratio']
    value=dict(version=VERSION,evidence=evidence.relative_to(ROOT).as_posix(),evidence_sha256=b.sha(evidence),
        cpu_smoke_sha256=b.sha(b.OUT/'cpu-smoke.json'),hidden_seconds=rec['hidden_seconds'],
        serving_ratio=ratio,projected_seconds=rec['hidden_seconds']*ratio,
        basis='HB156 actual hidden duration multiplied by paired CPU serving cost ratio',
        limitation='CPU model-only ratio; GPU N32 and independent presubmit review additionally required')
    b.need(math.isfinite(value['projected_seconds']) and 0<value['projected_seconds']<=32400,'hidden projection exceeds 9h')
    save(b.OUT/'runtime-binding.json',value)
    return dict(status='RUNTIME_BOUND',projected_seconds=value['projected_seconds'])


def runtime_fields(rec):
    b.need(rec['kind']=='KAGGLE_HIDDEN_RUNTIME' and rec['kernel']=='taeyangg4/knee-hb156-cnxs-l2-v1'
        and type(rec['kernel_version']) is int and rec['kernel_version']>0
        and str(rec['submission_ref'])=='57198732'
        and isinstance(rec['source'],str) and bool(rec['source'].strip()),'actual HB156 hidden evidence required')
    seconds=rec['hidden_seconds']
    b.need(type(seconds) in (int,float) and math.isfinite(seconds) and seconds>0,'positive finite hidden seconds required')


def gold_bind(report,reviewer):
    from tools import hb177_gold58_v1 as gold
    report=safe(report)
    b.need(isinstance(reviewer,str) and reviewer.strip() and reviewer!=gold.BUILDER,'independent reviewer required')
    b.need(re.search(r'(?m)^Verdict: PASS\s*$',report.read_text(encoding='utf-8')),'Gold review must state Verdict: PASS')
    gold.validate()
    value=dict(verdict='PASS',builder=gold.BUILDER,reviewer=reviewer,binding=gold.review_binding(),
        report_path=report.relative_to(ROOT).as_posix(),report_sha256=b.sha(report))
    save(b.OUT/'gold-independent-review.json',value)
    return dict(status='GOLD_REVIEW_BOUND')


def visible_bind(receipt,bound,version):
    """B supplies the confirmed remote version; no network or score access."""
    receipt=safe(receipt); bound=safe(bound)
    b.need(type(version) is int and version>0,'confirmed positive remote version required')
    gate_path=ROOT/'.local/state/gates'/ (b.SLUG.split('/')[1]+'.json')
    from tools import kaggle_submit_once as submit
    # Revalidate all release gate evidence before recording transport metadata.
    submit._strict_gate(b.SLUG,dict(gate_sha256=b.sha(gate_path)),{str(gate_path):b.sha(gate_path)})
    gate=b.read(gate_path); reg=bound/'registration.json'
    b.need(Path(gate['stage']).resolve()==bound and b.read(reg)['kernel_id']==b.SLUG,'visible registration differs from gate')
    value=dict(schema_version='1.0.0',receipt_sha256=b.sha(receipt),kernel=b.SLUG,kernel_version=version,
        registration=reg.relative_to(ROOT).as_posix(),registration_sha256=b.sha(reg),gate_sha256=b.sha(gate_path))
    save(receipt.parent/'submission-binding.json',value)
    return dict(status='VISIBLE_TRANSPORT_BOUND',remote_version=version)


def visible_receipt(receipt,log,submission,wait_log,version):
    """Validate downloaded FAST output; this never claims model/CUDA qualification."""
    b.need(type(version) is int and version>0,'confirmed positive remote version required')
    receipt,log,submission,wait_log=map(safe,(receipt,log,submission,wait_log))
    b.validate()
    source=''.join(b.read(b.OUT/'kernel/release.ipynb')['cells'][0]['source'])
    tree=ast.parse(source)
    ids=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='_own25q_visible_ids' for t in n.targets))
    header=next(ast.literal_eval(n.args[0]) for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr=='writerow')
    entries=b.read(log)
    b.need(isinstance(entries,list),'downloaded Kaggle JSON log required')
    output=''.join(entry.get('data','') for entry in entries)
    b.need('FAST_PATH_VISIBLE' in output and 'FAST_PATH_DISABLED' not in output
           and 'Traceback (most recent call last)' not in output,'FAST visible log failed')
    status=wait_log.read_text(encoding='utf-8-sig')
    b.need('real ref = '+b.SLUG+' ' in status and re.search(
        re.escape(b.SLUG)+r' after [^\n]*:\s*[^\n]*\bcomplete\b',status,re.I),
        'wrapper must confirm exact pushed slug and remote COMPLETE')
    with submission.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.reader(stream,strict=True))
    b.need(rows and rows[0]==header and len(rows)==4,'FAST output header/row count differs')
    b.need(all(len(row)==13 for row in rows[1:]) and sorted(row[0] for row in rows[1:])==ids,
           'FAST output ID roster/width differs')
    b.need(all(float(x)==.5 for row in rows[1:] for x in row[1:]),'FAST output must be finite 0.5')
    value=dict(status='COMPLETE_FAST_PATH_VISIBLE',fast_path=True,version='1.0.0',
        kernel=b.SLUG,kernel_version=version,fallbacks=dict(placeholder_errors=0),
        inference_performed=False,qualification='FORMAT_ONLY; N32 actual serving review separately required',
        evidence={p.relative_to(ROOT).as_posix():b.sha(p) for p in (log,submission,wait_log)})
    save(receipt,value)
    return dict(status='FAST_FORMAT_RECEIPT_CREATED',rows=3,targets=12,inference_performed=False)


def check(source,inputs,g):
    """Gate integration records all evidence bytes in its input hash inventory."""
    fit=inputs.json(source/'fit-result-review-binding.json')
    report=safe(fit['report'],inputs.root)
    text=inputs.read(report)
    g.need(fit['verdict']=='PASS' and isinstance(fit['reviewer'],str) and fit['reviewer'].strip() and fit['reviewer']!='hb177-postfit' and
        fit['checkpoint_sha256']==b.CHECKPOINT_SHA and g.sha(text)==fit['sha256'] and
        re.search(r'(?m)^Verdict: PASS\s*$',text.decode('utf-8')) and b.CHECKPOINT_SHA.encode() in text,
        'independent checkpoint-bound fit result PASS required')
    reg=inputs.json(source/'kill-registration.json')
    g.need(reg['decision']==b.DECISION and reg['exposure']=='EXPOSED' and reg['clean_oof'] is False
        and reg['checkpoint_sha256']==b.CHECKPOINT_SHA,'kill registration differs')
    binding=inputs.json(source/'runtime-binding.json')
    evidence=safe(binding['evidence'],inputs.root); rec=inputs.json(evidence); runtime_fields(rec)
    smoke=inputs.json(source/'cpu-smoke.json'); ratio=smoke['serving_ratio']
    g.need(smoke['status']=='PASS' and smoke['studies']==2 and smoke['cuda_visible_devices']=='' and
        smoke['cuda_initialized'] is False and smoke['smoke_source_sha256']==g.sha(inputs.read(inputs.root/'tools/hb177_postfit_smoke_v1.py')),
        'current real CPU2 smoke required')
    g.need(binding['version']==VERSION and binding['evidence_sha256']==g.sha(inputs.read(evidence)) and
        binding['cpu_smoke_sha256']==g.sha(inputs.read(source/'cpu-smoke.json')) and
        binding['hidden_seconds']==rec['hidden_seconds'] and binding['serving_ratio']==ratio and
        math.isfinite(ratio) and ratio>0 and binding['projected_seconds']==rec['hidden_seconds']*ratio and
        0<binding['projected_seconds']<=32400,'runtime evidence stale or projection exceeds 9h')
    inputs.read(inputs.root/'tools/hb177_release_admission_v1.py')
    return dict(status='ADMITTED_DRY_RUN',projected_seconds=binding['projected_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['fit-bind','runtime-bind','gold-bind','visible-receipt','visible-bind','admit'])
    p.add_argument('--report',type=Path); p.add_argument('--reviewer'); p.add_argument('--evidence',type=Path)
    p.add_argument('--receipt',type=Path); p.add_argument('--bound',type=Path); p.add_argument('--version',type=int)
    p.add_argument('--log',type=Path); p.add_argument('--submission',type=Path); p.add_argument('--wait-log',type=Path)
    p.add_argument('--dry-run',action='store_true'); a=p.parse_args(); b.cpu_only()
    try:
        if a.command=='fit-bind': result=fit_bind(a.report,a.reviewer)
        elif a.command=='runtime-bind': result=runtime_bind(a.evidence)
        elif a.command=='gold-bind': result=gold_bind(a.report,a.reviewer)
        elif a.command=='visible-receipt': result=visible_receipt(a.receipt,a.log,a.submission,a.wait_log,a.version)
        elif a.command=='visible-bind': result=visible_bind(a.receipt,a.bound,a.version)
        else:
            from tools import strict_variant_gate as g
            inputs=g.Inputs(ROOT); result=check(b.OUT,inputs,g); inputs.recheck()
        print(json.dumps(result,indent=2)); return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='HOLD',error=str(exc)))); return 1


if __name__=='__main__': sys.exit(main())
