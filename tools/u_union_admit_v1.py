"""U union offline admission v1.0.2-u4.

Changelog 1.0.2-u4: exact U4 identity, four pinned members including P4 FB;
retain v1.0.1-u output for original U. Backup: .local/backups/astra-b5/u4-v1/.

Changelog 1.0.1-u: functional N32 qualification without synthetic projection inputs;
projection is null and runtime is judged by B's 8-step gate. Visible checks unchanged.
Backup: .local/backups/astra-b5/u-admit-v1.0.0-u/.
Changelog 1.0.0-u: isolated U replay, full retained O, stride2 trio/no A5, mandatory N32.
Existing k1_admit_v1 and X3/A5 admission are unchanged. Never pushes/submits.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from tools import x3_a5_admit_v1 as inherited
from tools.x3_a5_admit_v1 import Evidence,need,local,sha,encoded,write_once,command_line,ROOT,Refuse,ledger_check
from tools import c85r7_rco_overlay_v1 as util
from tools import u_union_integration_v1 as contract_source
VERSION='1.0.2-u4'
OUTPUT='u-admission-v1'
EXPECTED_CONTRACT=dict(inf3=[.85,.15],x=[.8,.2],y=[.75,.25],union=[.5,.5],k1_stride=2,r6_stride=1,r224_weight=.35,p_weight_mode='public',a5=False,drop_o=False)
MODELS=['k1_v2','k1_v3g_seed0','k1_v3g_seed1']
MEMBERS={'R','R224','S_R7','EXP143v2','EXP155','Q_A5q'}
U4_RELEASE='1.1.86-r6w958k1u4fbv10'
U4_SLUG='taeyangg4/knee-lb-r6w958k1u4fbv10-v1'
U4_MODELS=MODELS+['k1_v3g_fullbag']
U4_HASHES=dict(zip(U4_MODELS,[
    '2c3c033e25605b5fb8bcc08e0030bed081c726e10fc56d8da62524ac08d9638c',
    '6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5',
    '29fc2b2bbaa4e4e8ad341bba17e47c5110388db697458a8cbb9ff8d83fa9399b',
    'ef3f42d76f6d2d57e6bda2c225b051e56652bcc8c07281470792907d3ec6ea98']))
U4_FB=dict(id='k1_v3g_fullbag',dataset_slug='taeyangg4/knee-k1-fb-v1-assets',
    checkpoint='weights/E_prior_s0_f0.pt',checkpoint_sha256=U4_HASHES['k1_v3g_fullbag'],
    trainer_source='source/compact_train.py',
    trainer_sha256='14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80',
    manifest_sha256='c3d8651a2d87763efbf7525269f4ed9bc6a7b5077ad89851e1d8f3923f940b63')
U4_FORMULA=contract_source.FORMULA.replace('K1 trio stride2','K1 four (v2, v3-G seed0, v3-G seed1, FB fold0) stride2')

def is_u4(reg):
    suffix='-n32' if reg.get('diagnostic') else ''
    release,slug=reg.get('release_version'),reg.get('kernel_id')
    if release==U4_RELEASE+suffix or slug==U4_SLUG+suffix or reg.get('formula')==U4_FORMULA:
        need(release==U4_RELEASE+suffix and slug==U4_SLUG+suffix,'U4 exact release/slug')
        return True
    return False

def registered_models(reg):
    if not is_u4(reg):return MODELS
    models=reg.get('models')
    need(isinstance(models,list) and [m.get('id') for m in models]==U4_MODELS,'U4 exactly four registered members')
    need({m['id']:m.get('checkpoint_sha256') for m in models}==U4_HASHES
         and reg.get('model_sha256')==U4_HASHES,'U4 registered checkpoint pins')
    need(models[-1]==U4_FB and reg.get('asset_manifest_sha256',{}).get(U4_FB['id'])==U4_FB['manifest_sha256'],'U4 exact P4 FB identity')
    return U4_MODELS
def ranks(v): return rankdata(np.asarray(v,dtype=np.float64),axis=0,method='average')/len(v)
def check_contract(reg,k,parent):
    models=registered_models(reg)
    need(reg.get('u_contract')==EXPECTED_CONTRACT,'U exact weights/contract')
    need(k.get('k1_eval_stride')==2 and k.get('contract',{}).get('eval_stride')==2,'U K1 stride2')
    need(k.get('k1_models')==models,'U K1 complete registered member roster')
    if is_u4(reg):
        need(k.get('model_sha256')==U4_HASHES and k.get('model_identities')==reg['models'],'U4 receipt checkpoint pins')
    for key in ('intended_members','executed_members'):
        values=parent.get(key)
        need(isinstance(values,list) and len(values)==6 and set(values)==MEMBERS,'U complete retained O roster: '+key)
    need(reg.get('drop_o') is False and k.get('drop_o') is False and not parent.get('drop_o',False),'U retained O')
    need(k.get('parent_mode')=='U_UNION_INF3_PUBLIC','U parent mode')
    need(not any('a5' in str(key).lower() for obj in (reg,k,parent) for key in obj),'A5 registration/receipt present')
    need(k.get('k1_max_failed_fraction')==.05 and k.get('k1_failure_policy')=='study_fallback_fail_loud'
         and k.get('fail_loud_triggered') is False and 0<=k.get('failed_study_fraction',1)<=.05,'U failed fraction <= .05')
    need(k.get('guard_seconds')==8.8*3600 and k.get('capture_timeout_seconds')==900
         and k.get('call_timeout_seconds')==300 and k.get('load_timeout_seconds')==1500,'U v10 guards')

def chain(e,directory,reg,k,parent,diagnostic):
    directory=local(directory);check_contract(reg,k,parent)
    models=registered_models(reg)
    formula=U4_FORMULA if is_u4(reg) else contract_source.FORMULA
    need(not any(directory.glob('a5-*')) and not any(directory.glob('x3-a5-*')),'A5 artifacts present')
    data=e.obj(directory/'u-inputs.json'); rec=e.obj(directory/'u-receipt.json')
    need(rec.get('schema')=='u-union/v1' and rec.get('status')=='COMPLETE_U' and rec.get('contract')==EXPECTED_CONTRACT
         and rec.get('a5_executed') is False and rec.get('formula')==reg['formula']==formula,'U composition receipt')
    need(rec.get('diagnostic') is diagnostic and rec.get('never_submit') is diagnostic,'U diagnostic identity')
    need(k.get('u_receipt_sha256')==sha(e.read(directory/'u-receipt.json')),'U K1 receipt binding')
    need(rec.get('inputs_sha256')==parent.get('u_inputs_sha256')==sha(e.read(directory/'u-inputs.json')),'U input binding')
    need(data.get('schema')=='u-union-inputs/v1' and data.get('contract')==parent.get('u_contract')==EXPECTED_CONTRACT,'U input contract')
    original=e.read(directory/'u-original-inf3.csv');ids,inf3=util.table(original);n=len(ids)
    need(len(set(ids))==n and data.get('ids')==ids and rec.get('studies')==n,'U full roster')
    need((n==32 if diagnostic else n==3),'U expected visible/N32 count')
    need(sha(original)==rec.get('original_inf3_sha256')==data.get('original_inf3_sha256')==k.get('parent_sha256')
         ==sha(e.read(directory/'k1sr-parent.csv')),'U original INF3 binding')
    need(diagnostic or parent.get('submission_sha256')==sha(original),'U parent output binding')
    public=np.asarray(data.get('public'),dtype=np.float64);own=np.asarray(data.get('own'),dtype=np.float64)
    need(public.shape==own.shape==(n,12) and np.isfinite(public).all() and np.isfinite(own).all(),'U finite P/O')
    pf,of=data.get('p_failed'),data.get('o_failed')
    need(isinstance(pf,list) and isinstance(of,list) and len(pf)==len(of)==n and all(type(v) is bool for v in pf+of),'U masks')
    pf,of=np.array(pf),np.array(of)
    def scored(values,mask):
        result=np.zeros_like(values)
        if (~mask).any():result[~mask]=ranks(values[~mask])
        return result
    pr,orr=scored(public,pf),scored(own,of);original_expected=.85*pr+.15*orr
    original_expected[of]=pr[of];original_expected[pf]=orr[pf];original_expected[pf&of]=.5
    need(np.array_equal(original_expected.astype(np.float32).astype(np.float64),inf3),'U independent INF3 recomputation')
    for name,value in [('u-p.csv',public),('u-o.csv',own)]:
        roster,csvvalue=util.table(e.read(directory/name));need(roster==ids and np.array_equal(csvvalue,value.astype(np.float32).astype(np.float64)),'U '+name+' snapshot')
    rows=k.get('per_study');need(isinstance(rows,list) and len(rows)==n,'U K1 telemetry')
    need(sha(encoded(rows))==rec.get('k1_records_sha256'),'U telemetry hash')
    lookup={r['uid']:r for r in rows};need(len(lookup)==n and set(lookup)==set(ids),'U K1 roster')
    need(all(type(r.get('ok')) is bool and r.get('config_sha256')==reg['config_sha256'] and r.get('k1_models')==models for r in rows),'U per-study registration')
    need(all(not r.get('systemic_failure') and 0<=r.get('finished_elapsed_seconds',-1)<=8.8*3600 for r in rows),'U systemic/guard')
    fail=np.array([not lookup[uid]['ok'] for uid in ids]);need(int(fail.sum())==k.get('failed_study_count') and float(fail.mean())==k.get('failed_study_fraction') and fail.mean()<=.05,'U measured failure fraction')
    need(k.get('k1_studies')==int((~fail).sum()),'U successful count')
    need(all(set(r.get('predictions',{}))==set(models) for r in rows if r['ok']),'U member predictions')
    member=[]
    for mid in models:
        values=np.asarray([inf3[i] if fail[i] else lookup[uid]['predictions'][mid] for i,uid in enumerate(ids)],dtype=np.float32).astype(np.float64)
        need(values.shape==(n,12) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),'U K1 probabilities')
        member.append(ranks(values))
    kr=ranks(np.mean(np.stack(member),axis=0));ir=ranks(inf3);pr=ranks(public.astype(np.float32).astype(np.float64))
    def branch(parent,weight):
        kk=kr.copy();kk[fail]=parent[fail];v=(1-weight)*parent+weight*kk;v[fail]=parent[fail]
        return ranks(v).astype(np.float32)
    x=branch(ir,.2);y=branch(pr,.25);u=ranks(.5*ranks(x)+.5*ranks(y)).astype(np.float32)
    expected={'u-inf3.csv':ir,'u-k.csv':kr,'u-x.csv':x,'u-y.csv':y,'u-final.csv':u}
    need(set(rec.get('csv_sha256',{}))==set(expected),'U intermediate inventory')
    for name,value in expected.items():
        raw=e.read(directory/name);roster,actual=util.table(raw)
        need(sha(raw)==rec['csv_sha256'][name] and roster==ids and np.array_equal(actual,value.astype(np.float32).astype(np.float64)),'U independent replay: '+name)
    final=e.read(directory/('diagnostic-k1n.csv' if diagnostic else 'submission.csv'))
    need(final==e.read(directory/'u-final.csv') and sha(final)==k.get('diagnostic_sha256' if diagnostic else 'submission_sha256'),'U final publication')
    e.checks.append('Independent scipy full-roster INF3/K/X/Y/U replay: PASS')
    return ids

def functional_n32(e, directory, visible):
    """Qualify execution and identity; runtime admission belongs to B."""
    n = e.obj(directory / "k1sr-receipt.json")
    e.check(isinstance(n.get("status"), str) and n["status"].startswith("COMPLETE_SHARED_K1")
            and n.get("k1_executed") is True and type(n.get("k1_studies")) is int
            and 31 <= n["k1_studies"] <= 32, "N32 COMPLETE K1 executed >=31/32 (<=5% failed)")
    rows = n.get("per_study")
    need(isinstance(rows, list) and len(rows) == 32 and
         all(isinstance(r, dict) and type(r.get("ok")) is bool and isinstance(r.get("uid"), str) for r in rows)
         and len({r["uid"] for r in rows}) == 32
         and sum(r['ok'] for r in rows)==n['k1_studies'], "N32 complete unique per-study rows required")
    for key in ("model_sha256", "formula", "contract", "r224_weight"):
        e.check(key in visible and n.get(key) == visible[key], "N32 matches visible " + key)
    e.check(type(n.get("failed_study_fraction")) in (int, float)
            and 0 <= n["failed_study_fraction"] <= .05
            and n["failed_study_fraction"] == (32 - n["k1_studies"]) / 32,
            "N32 measured failed fraction <= .05")
    e.checks.append("U N32 functional qualification and full-roster replay: PASS; "
                    "runtime is judged by B's 8-step gate, not by this tool; "
                    "no hidden runtime projection claimed")
    return None



def validate(stage,kernel,version,visible_dir,n32_dir=None):
    need(n32_dir is not None,'U requires visible and N32 evidence')
    e,reg,k,_=inherited.validate(stage,kernel,version,visible_dir,None)
    stage=local(stage); visible=local(visible_dir); n32=local(n32_dir)
    config=e.obj(stage/'config.json');meta=e.obj(stage/'kernel/kernel-metadata.json')
    need(not any('knee-mri-fold-weights' in v for v in meta['dataset_sources']),'A5 dataset present')
    need(config.get('k1_eval_stride')==2 and config.get('k1_runtime_version')==10 and config.get('drop_o') is False
         and config.get('alpha')==.2 and config.get('p_weight_mode','public')=='public','U config recipe')
    need(sha(encoded(config))==reg['config_sha256'],'U config hash')
    need(k.get('model_identities')==config['models'] and [m['id'] for m in config['models']]==registered_models(reg),'U exact model identities')
    if is_u4(reg):
        need(config['models']==reg['models'] and meta['dataset_sources'].count(U4_FB['dataset_slug'])==1,'U4 config/assets identity')
    chain(e,visible,reg,k,e.obj(visible/'c85r7-receipt.json'),False)
    nr=e.obj(Path(str(stage)+'-n32')/'registration.json');n=e.obj(n32/'k1sr-receipt.json');p=e.obj(n32/'c85r7-receipt.json')
    need(nr['kernel_id']==kernel+'-n32' and nr['release_version']==reg['release_version']+'-n32','U N32 sibling stage')
    ns=Path(str(stage)+'-n32')
    need(local(nr.get('stage',''))==ns,'U N32 registered directory')
    for filename,key in (('release.ipynb','kernel_sha256'),('kernel-metadata.json','metadata_sha256')):
        need(sha(e.read(ns/'kernel'/filename))==nr.get(key),'U N32 stage hash: '+filename)
    nm=e.obj(ns/'kernel/kernel-metadata.json')
    need(nm.get('id')==nr['kernel_id'] and nm.get('dataset_sources')==meta['dataset_sources'],'U N32 metadata identity/assets')
    need(e.obj(ns/'config.json')==config,'U N32 registered config identity')
    for key in ('kernel_id','release_version','formula','config_sha256','registered_config_sha256','model_sha256','contract','k1_weight','parent_weight','r224_weight'):
        need(n.get(key)==nr.get(key),'U N32 bound '+key)
    need(n.get('config_sha256')==reg['config_sha256'] and n.get('model_identities')==config['models'],'U N32 same recipe')
    need(n.get('status')=='COMPLETE_SHARED_K1N' and n.get('k1_executed') is True and n.get('diagnostic') is True and n.get('never_submit') is True,'U N32 complete diagnostic')
    need(p.get('status')=='DIAGNOSTIC_C85R7' and p.get('version')==n['release_version'] and p.get('studies')==32 and p.get('findings')==12,'U N32 parent complete')
    inherited.verified_drop_o(e,stage,reg,n,p);inherited.prediction_fallbacks(p);inherited.codecs_active(p.get('codecs'))
    chain(e,n32,nr,n,p,True)
    versions=set()
    for name in ('k1sr-receipt.json','c85r7-receipt.json','diagnostic-k1n.csv'):
        d=e.obj(n32/(name+'.download.json'))
        need(d.get('status')=='DOWNLOADED' and d.get('kernel')==nr['kernel_id'] and type(d.get('version')) is int and d['version']>0
             and d.get('file')==d.get('local_name')==name and d.get('sha256')==sha(e.read(n32/name)),'U N32 download provenance')
        versions.add(d['version'])
    need(len(versions)==1,'U N32 version provenance')
    projected=functional_n32(e,n32,k)
    e.checks = [check for check in e.checks
                if check != "N32: NOT PROVIDED (optional; no runtime qualification claimed)"]
    e.recheck()
    return e,reg,k,projected

def prepare(stage, kernel, version, visible_dir, n32_dir=None):
    e, reg, k, projected = validate(stage, kernel, version, visible_dir, n32_dir)
    visible, stage = local(visible_dir), local(stage)
    out = local(visible / OUTPUT)
    out.mkdir(exist_ok=True)
    output_version = VERSION if is_u4(reg) else '1.0.1-u'
    registration = dict(
        schema="ordinary-visible-submission-registration/v1", tool_version=output_version,
        kernel_id=kernel, version=k["release_version"], stage=str(stage),
        kernel_sha256=reg["kernel_sha256"], metadata_sha256=reg["metadata_sha256"],
        stage_registration_sha256=e.hashes[str(stage / "registration.json")],
        formula=k["formula"], execution_verified=True, input_sha256=dict(e.hashes),
        projection=projected, independent_review="NOT PERFORMED BY THIS TOOL",
    )
    rp = out / "ordinary-visible-registration.json"
    registration_bytes = encoded(registration)
    binding = dict(schema_version="1.0.0", kernel=kernel, kernel_version=version,
                   receipt_sha256=e.hashes[str(visible / "c85r7-receipt.json")],
                   registration=str(rp), registration_sha256=sha(registration_bytes),
                   k1_receipt_sha256=e.hashes[str(visible / "k1sr-receipt.json")],
                   submission_sha256=k["submission_sha256"])
    review = (f"# K1 automated admission v{output_version}\n\n"
              f"Kernel: {kernel}\nKaggle version: {version}\nStage: {stage}\n\n"
              f"Formula (verbatim receipt): {k['formula']}\n\n"
              "Machine checks only; this document is not an independent reviewer verdict.\n"
              "B must retain the independent review and refresh live allowance/duplicates before submission.\n\n"
              + "\n".join("- " + check for check in e.checks)
              + "\n\nProjection: " + json.dumps(projected, sort_keys=True)
              + "\n\nVERDICT: PASS\n").encode()
    write_once(rp, registration_bytes)
    write_once(out / "submission-binding.json", encoded(binding))
    # Canonical wrapper binds its sibling sidecar. Original receipts stay untouched.
    write_once(out / "c85r7-receipt.json", e.read(visible / "c85r7-receipt.json"))
    pending = out / "machine-check.pending.md"
    write_once(pending, review)
    for path in (rp, out / "submission-binding.json", out / "c85r7-receipt.json", pending):
        e.read(path)
    identity = "K1-" + sha(f"{kernel}:{version}".encode())[:20]
    base = [sys.executable, str(ROOT / "tools/kaggle_submit_once.py"),
            "--id", identity, "--kernel", kernel, "--version", str(version),
            "--message", k["formula"], "--receipt", str(out / "c85r7-receipt.json")]
    result = subprocess.run(base + ["--review", str(pending), "--check-only"],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    print("CHECK-ONLY " + result.stdout.strip())
    need(result.returncode == 0, "kaggle_submit_once --check-only REFUSED")
    try:
        outcome = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise Refuse("check-only returned malformed outcome") from None
    need(isinstance(outcome, dict) and outcome.get("outcome") == "ADMITTED_OFFLINE",
         "check-only did not admit")
    e.recheck()
    ledger_check(e, kernel, version)
    final_review = out / "READY-REVIEW.md"
    write_once(final_review, review.replace(
        b"\n\nVERDICT: PASS", b"\n\nCanonical --check-only: ADMITTED_OFFLINE\n\nVERDICT: PASS"))
    print("PASS: " + str(out))
    command = command_line(base + ["--review", str(final_review)])
    print("SUBMIT COMMAND (NOT EXECUTED): " + command)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True);p.add_argument('--kernel',required=True);p.add_argument('--version',type=int,required=True);p.add_argument('--visible-dir',required=True);p.add_argument('--n32-dir',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    try:
        if a.dry_run:
            e,reg,k,projection_result=validate(a.stage,a.kernel,a.version,a.visible_dir,a.n32_dir)
            print(json.dumps(dict(status='PASS_CPU_ADMISSION',checks=e.checks,projection=projection_result)))
        else: prepare(a.stage,a.kernel,a.version,a.visible_dir,a.n32_dir)
    except (ValueError,FileNotFoundError,KeyError) as exc:
        print(json.dumps(dict(status='HOLD_NOT_ADMITTED',reason=str(exc))));raise SystemExit(2)
if __name__=='__main__':main()
