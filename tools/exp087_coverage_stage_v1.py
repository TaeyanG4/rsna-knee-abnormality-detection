"""EXP087 same-policy staging: freeze metadata, then explicit bounded CPU run.

The successful8 pilot inputs are immutable references. Only the remaining307
alternate series can be decoded. No training/model/label/prediction operation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import exp087_series_coverage as policy

sha=policy.sha; read_json=policy.read_json; dump=policy.dump; local=policy.local; uri=policy.uri; now=policy.now
PILOT=".local/exp087-coverage/pilot-v1"
PE="docs/handoffs/evidence/2026-10/2026-10-04-exp087-coverage-pilot"
EVIDENCE="docs/handoffs/evidence/2026-10/2026-10-04-exp087-coverage-scale-v1"
DETAIL=f"{PE}/policy-count-v1/potential-studies.json"
DETAIL_RAW_SHA="31ded41014328b929b325ce3387387cffa80c96dbbcce3881c4b8b5d2b169017"
HELPER_SHA="b91d32a8d9a7dca06f905428c83511b35a9d79119dbde32ba51c9a922c87a3a8"
CAPS=dict(pairs_min=315,pairs_max=315,headers=65536,header_bytes=512*1024**2,
    pixel_reads=3684,pixel_bytes=8*1024**3,rendered_bytes=415908864,rss_bytes=8*1024**3,wall_seconds=900)


def freeze(directory):
    directory=local(directory); directory.mkdir(parents=True,exist_ok=False)
    if sha(local("tools/exp087_series_coverage.py"))!=HELPER_SHA:raise ValueError("frozen v3 helper changed")
    if sha(local(DETAIL))!=DETAIL_RAW_SHA:raise ValueError("saved315 raw-byte detail changed; do not rerun census")
    details=read_json(local(DETAIL)); membership=read_json(local(f"{policy.GROUP}/ordered-ids.json"))
    if (len(details)!=315 or len({d['uid'] for d in details})!=315 or
        sum(d['split']=='train' for d in details)!=252 or sum(d['split']=='val' for d in details)!=63 or
        policy.uid_hash(membership['train'])!=policy.EXPECTED_T or policy.uid_hash(membership['val'])!=policy.EXPECTED_V):
        raise ValueError("saved population/G1 mismatch")
    pilot=read_json(local(f"{PILOT}/receipt.json")); supervisor=read_json(local(f"{PILOT}/supervisor.json"))
    independent=read_json(local(f"{PE}/result-independent-review.json"))
    if (pilot['status']!='CPU_ENGINEERING_COMPLETE' or supervisor['status']!='PASS' or
        independent['status']!='PASS_CPU_ENGINEERING_RESULT_ONLY' or len(pilot['pairs'])!=8):
        raise ValueError("completed pilot and independent result review required")
    pf=read_json(local(f"{PE}/freeze-v3/fixture.json"))
    previous={p['uid']:p for p in pilot['pairs']}; pfrows={p['uid']:p for p in pf['pairs']}
    wanted={(d['uid'],d[k]) for d in details for k in ('original','alternate')}
    metadata={}
    with local('storage/competition/train_series.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(r[policy.UID],r[policy.SERIES])
            if key in wanted:
                if key in metadata:raise ValueError('duplicate selected series metadata')
                metadata[key]=r
    if set(metadata)!=wanted:raise ValueError('selected metadata absent')
    picks=policy.pick_subset({d['uid'] for d in details})
    index=sqlite3.connect(local(policy.INDEX).as_uri()+'?mode=ro',uri=True)
    index.execute('PRAGMA query_only=ON')
    rows=[]; count_headers=0; payload_bound=0; reusable_bindings={}
    try:
        for d in details:
            uid=d['uid']
            if uid not in membership[d['split']]:raise ValueError('G1 ownership mismatch')
            original=metadata[(uid,d['original'])]; alt=metadata[(uid,d['alternate'])]
            original_series=[r.get('series') if r else None for r in picks[uid]['slots']]
            if (original_series[2]!=d['original'] or d['alternate'] in original_series or
                original['Anatomical_Plane']!='Coronal' or alt['Anatomical_Plane']!='Coronal' or
                policy.flag(original['Fluid_Sensitive'])!=1 or policy.flag(alt['Fluid_Sensitive'])!=1 or
                policy.flag(original['Fat_Suppression']) not in (0,1) or policy.flag(original['Fat_Suppression'])!=policy.flag(alt['Fat_Suppression'])):
                raise ValueError('saved selector source contract differs; no reselection')
            row=dict(uid=uid,split=d['split'],original=d['original'],alternate=d['alternate'],metadata=[original,alt],original_pick=picks[uid]['slots'][2])
            if uid in previous:
                pr=previous[uid]; fp=pfrows[uid]
                if fp['series'][0]['series']!=d['original'] or fp['series'][1]['series']!=d['alternate'] or pr['reason'] is not None:
                    raise ValueError('completed pilot selection differs')
                path=local(f'{PILOT}/{uid}-alternate.npy')
                a=np.load(path,allow_pickle=False)
                if a.shape!=(12,336,336) or a.dtype!=np.uint8 or policy.array_hash(a)!=pr['alternate']['volume_hash']:
                    raise ValueError('saved pilot alternate hash/shape differs')
                mask=(a.reshape(12,-1).sum(1)>0).astype(np.uint8)
                if policy.array_hash(mask)!=pr['alternate']['mask_hash']:raise ValueError('saved pilot mask differs')
                reusable_bindings[uri(path)]=sha(path)
                row.update(reused=True,pilot_sops=[sop for r in pr['header'] for sop in r['sops']],entry=dict(uid=uid,split=d['split'],reason=None,volume=uri(path),file_sha256=sha(path),
                    array_hash=policy.array_hash(a),mask=mask.tolist(),native_indices=fp['series'][1]['selected_indices'],
                    provenance='completed pilot8; no decode repeated'))
            else:
                sources=[]
                for series in (d['original'],d['alternate']):
                    members=[dict(path=p,info=json.loads(info)) for p,info in index.execute('SELECT path,info FROM files WHERE split=? AND uid=? AND series=? ORDER BY path',('train',uid,series))]
                    if not members:raise ValueError('selected acquired series files absent')
                    count_headers+=len(members); sources.append(members)
                if len(sources[0])!=row['original_pick']['files']:raise ValueError('original source file count changed')
                payload_bound+=sum(sorted((int(m['info']['file_size']) for m in sources[1]),reverse=True)[:12])
                row.update(reused=False,sources=sources)
            rows.append(row)
    finally:index.close()
    if sum(not r['reused'] for r in rows)!=307 or count_headers>CAPS['headers'] or payload_bound>CAPS['pixel_bytes']:
        raise ValueError('new source scope exceeds proposed caps')
    bind=[DETAIL,f'{policy.GROUP}/ordered-ids.json',f'{PE}/result-independent-review.json',f'{PILOT}/receipt.json',f'{PILOT}/supervisor.json',
        f'{EVIDENCE}/root-cpu-decision.json','tools/exp087_series_coverage.py','tools/exp087_coverage_stage_v1.py',
        'training/src/exp087_coverage_data.py','tests/test_exp087_coverage_stage_v1.py','training/src/raptor_common.py','training/src/raptor_data.py',
        'training/src/raptor_dicom.py',f'{policy.CORPUS}/manifest.json','storage/competition/train_series.csv']
    fixture=dict(schema='exp087-coverage-stage-fixture-v1',created_at=now(),policy=policy.POLICY,pairs=rows,membership=membership,
        caps=CAPS,source_sha256={**{uri(p):sha(local(p)) for p in bind},**reusable_bindings},baseline_metadata_sha256=policy.BASELINE_METADATA_SHA,
        source_stats=[policy.source_stat(policy.ARCHIVE),policy.source_stat(policy.INDEX),policy.source_stat('storage/competition/train_series.csv'),policy.source_stat(f'{policy.CORPUS}/picks.jsonl')],
        cache_shard_stats=pf['original_cache_stat'],pick_subset_sha256=policy.digest(picks),selected_metadata_sha256=policy.digest(sorted(metadata.items())),
        source_scope=dict(reused8=8,new307=307,headers_planned=count_headers,alternate_selected_payload_upper_bound=payload_bound,
            upper_bound_reason='sum12largest declared member sizes per alternate; selected native paths need future headers',raw_detail_sha256=DETAIL_RAW_SHA),
        workers=dict(requested=12,effective=1,internal=1,reason='successful serial pilot11.281s; one indexed ZIP stream, bounded memory; no pool or changed renderer'),
        no_census_rerun=True,no_new_pixels_or_headers=True)
    dump(directory/'fixture.json',fixture)
    plan=dict(schema='exp087-coverage-stage-plan-v1',created_at=now(),fixture=uri(directory/'fixture.json'),fixture_sha256=sha(directory/'fixture.json'),
        source_sha256=fixture['source_sha256'],caps=CAPS,output=uri('.local/exp087-coverage/stage-v1'),interpreter=str(Path(sys.executable).resolve()),
        environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),
        decision='ROOT_EXECUTION_REVIEW_REQUIRED',retry=0,goal='input staging only; no training or efficacy conclusion')
    dump(directory/'execution-plan.json',plan)
    launch=dict(plan_sha256=sha(directory/'execution-plan.json'),cwd=str(ROOT),environment=plan['environment'],output=plan['output'],
        argv=[plan['interpreter'],str(Path(__file__).resolve()),'run','--plan',str(directory/'execution-plan.json'),'--plan-sha256',sha(directory/'execution-plan.json'),'--execute-pixels'])
    dump(directory/'launch.json',launch)
    print(json.dumps(dict(status='FROZEN_NO_NEW_HEADERS_OR_PIXELS',scope=fixture['source_scope'],plan_sha256=launch['plan_sha256'])))


def verify_bindings(plan,fixture):
    if plan['caps']!=CAPS or fixture['caps']!=CAPS or fixture['policy']!=policy.POLICY:raise ValueError('policy/caps changed')
    if fixture['source_sha256']!=plan['source_sha256']:raise ValueError('plan source binding differs')
    for path,expected in {**fixture['source_sha256'],**fixture['baseline_metadata_sha256']}.items():
        if sha(local(path))!=expected:raise ValueError(f'source changed: {path}')
    for record in fixture['source_stats']+fixture['cache_shard_stats']:
        if policy.source_stat(record['path'])!=record:raise ValueError('original source stat changed')
    if policy.digest(policy.pick_subset({p['uid'] for p in fixture['pairs']}))!=fixture['pick_subset_sha256']:
        raise ValueError('current315 picks changed')


def headers(source,row,role,budget):
    import pydicom
    series=row['original' if role==0 else 'alternate']; expected=row['sources'][role]
    members=source.members('train',row['uid'],series)
    if [dict(path=m.path,info=json.loads(m.info)) for m in members]!=expected:raise ValueError('frozen member map differs')
    records=[]; sops=[]; syntaxes=[]
    for m in members:
        budget.add('headers',1)
        with source.archive.open(policy.zip_entry(m),'r') as f:
            ds=pydicom.dcmread(policy.HeaderReader(f,budget),stop_before_pixels=True)
        if 'PixelData' in ds:raise ValueError('pixels parsed by header stage')
        h=policy.canonical_header(ds)
        if h[policy.UID]!=row['uid'] or h[policy.SERIES]!=series:raise ValueError('header ownership mismatch')
        sop=str(getattr(ds,'SOPInstanceUID','') or '')
        if not sop:raise ValueError('missing SOPInstanceUID')
        sops.append(sop); syntaxes.append(str(ds.file_meta.TransferSyntaxUID)); records.append(dict(path=m.path,header=h))
    if len(set(sops))!=len(sops):raise ValueError('duplicate SOP within series')
    return dict(geometry=policy.geometry([r['header'] for r in records],row['uid'],series),sops=sops,
        transfer_syntaxes=sorted(set(syntaxes)),records=records),{m.path:m for m in members}


def schedule_counts(rc,mask,spec,uid,entry):
    counts=[]
    for train,epoch in [(False,0)]+[(True,e) for e in range(8)]:
        centers=rc.select_centers(mask,64,12 if train else 24,spec,'within-slot-valid',rc.study_rng(42,epoch,uid) if train else None)
        mapping={} if entry['reason'] is not None else policy.replacement_map(centers,entry['mask'],entry['native_indices'])[0]
        counts.append(dict(train=train,epoch=epoch,replacements=len(mapping)))
    return counts


def register_sops(owners,uid,split,sops):
    """Newly observed cross-G1 evidence must not be silently ignored."""
    for sop in sops:
        old=owners.get(sop)
        if old is not None and old['split']!=split:
            raise ValueError(f'cross-G1 SOP identity: {sop}: {old["uid"]} / {uid}; ancestor review required')
        owners.setdefault(sop,dict(uid=uid,split=split))


def child_run(plan_path,output,expected_sha,execute_pixels):
    if not execute_pixels or sha(plan_path)!=expected_sha or os.environ.get('EXP087_AUTHORIZED_PLAN_SHA256')!=expected_sha:
        raise ValueError('exact authorized supervisor plan required')
    import cv2
    cv2.setNumThreads(1)
    plan=read_json(plan_path)
    if sha(local(plan['fixture']))!=plan['fixture_sha256'] or local(plan['output'])!=local(output):raise ValueError('fixture/output differs')
    fixture=read_json(local(plan['fixture'])); verify_bindings(plan,fixture)
    rc,rd=policy.helpers(); budget=policy.Budget(CAPS); entries=[]; summary=[]
    corpus=rc.Corpus(local(policy.CORPUS)); sop_owners={}
    for row in fixture['pairs']:
        if row['reused']:register_sops(sop_owners,row['uid'],row['split'],row['pilot_sops'])
    try:
        with rd.IndexedZipDicomSource(local(policy.ARCHIVE),local(policy.INDEX)) as source:
            for row in fixture['pairs']:
                budget.check(); uid=row['uid']; details={}
                if row['reused']:
                    entry=row['entry']
                else:
                    first,fm=headers(source,row,0,budget); second,sm=headers(source,row,1,budget)
                    register_sops(sop_owners,uid,row['split'],first['sops']+second['sops'])
                    reason=None; volume=None; record=None
                    if not first['geometry']['eligible'] or not second['geometry']['eligible']:reason='geometry_ineligible'
                    elif set(first['sops']) & set(second['sops']):reason='shared_SOP'
                    entry=dict(uid=uid,split=row['split'],reason=reason,volume=None,file_sha256=None,mask=None,native_indices=None,
                        provenance='new alternate only; original DICOM pixels not decoded')
                    if reason is None:
                        original_order,original_indices=policy.native_selection(first['records'])
                        if original_indices!=row['original_pick']['slices']:raise ValueError('original native selection changed')
                        ordered,indices=policy.native_selection(second['records'])
                        record=dict(series=row['alternate'],metadata=row['metadata'][1],audited_headers=second['records'],
                            selected_indices=indices,ordered_paths=[h['path'] for h in ordered])
                        volume,mask,render=policy.render(source,record,sm,budget)
                        entry.update(mask=mask.tolist(),native_indices=indices,array_hash=policy.array_hash(volume))
                        if np.array_equal(volume,corpus.volume(uid)[32:44]):entry['reason']='sampled_render_duplicate'
                        elif policy.replacement_map([33,42],mask,indices)[1]=='no_valid_alt_triplet':entry['reason']='no_valid_alt_triplet'
                        path=output/(uid+'-alternate.npy'); np.save(path,volume)
                        entry.update(volume=uri(path),file_sha256=sha(path)); details['render']=render
                    details['headers']=[first,second]
                entries.append(entry)
                row_result=dict(uid=uid,split=row['split'],reused=row['reused'],reason=entry['reason'],
                    schedules=schedule_counts(rc,corpus.mask(uid),corpus.spec,uid,entry),details=details)
                dump(output/(uid+'-receipt.json'),row_result); summary.append({k:v for k,v in row_result.items() if k!='details'})
                if len(entries)%25==0:print(json.dumps(dict(completed=len(entries),budget=budget.report())),flush=True)
        manifest=dict(status='STAGED_INPUTS_ONLY_NO_TRAINING',policy=policy.POLICY,entries=entries,membership=fixture['membership'],
            fixture_sha256=plan['fixture_sha256'],adapter=uri('training/src/exp087_coverage_data.py'),adapter_sha256=sha(local('training/src/exp087_coverage_data.py')),policy_helper_sha256=HELPER_SHA,
            baseline=dict(init='same generic ImageNet CoAt as EXP084_B_G1',initial_state_sha256='e4ea3ae625eacde8e3ab563b6c663d715600a0cea039e1c9ee9d201367d5b38e',
                corpus=uri(policy.CORPUS),corpus_manifest_sha256=fixture['source_sha256'][uri(f'{policy.CORPUS}/manifest.json')],
                G1_T=3455,G1_V=888,epochs=8,updates=3448,seed=42,trainK=12,evalK=24,eval_batch=4),
            limits=['G1 exposed weak development set','scheduled counts are not all actual resized-window differences','no original307 DICOM pixel replay','input staging does not authorize training'])
        dump(output/'manifest.json',manifest)
        dump(output/'receipt.json',dict(status='CPU_STAGING_COMPLETE',policy=policy.POLICY,created_at=now(),budget=budget.report(),caps=CAPS,
            manifest_sha256=sha(output/'manifest.json'),entries=summary,workers=fixture['workers'],reused8=8,new_planned307=307,
            labels_predictions_models_GPU=0,no_training_authorization=True,
            byte_scope='declared successful member read lengths/charge, not physical compressed disk IO'))
    except BaseException as exc:
        dump(output/'FAIL.json',dict(status='FAIL',error=f'{type(exc).__name__}: {exc}',completed=len(entries),budget=budget.report()))
        raise


# Supervisor copied unchanged from frozen successful v3; this module binds scale CAPS.
def run(plan_path, expected_sha, execute_pixels):
    if not execute_pixels or sha(plan_path)!=expected_sha:
        raise ValueError("explicit pixel decision and exact plan hash required")
    plan=read_json(plan_path); fixture_path=local(plan["fixture"])
    if sha(fixture_path)!=plan["fixture_sha256"]:
        raise ValueError("fixture hash mismatch")
    fixture=read_json(fixture_path); verify_bindings(plan,fixture)
    if not CAPS["pairs_min"]<=len(fixture["pairs"])<=CAPS["pairs_max"]:
        raise ValueError("pair count out of frozen range")
    output=local(plan["output"]); output.mkdir(parents=True,exist_ok=False)
    import psutil
    env=os.environ.copy(); env.update(plan["environment"])
    env["EXP087_AUTHORIZED_PLAN_SHA256"]=expected_sha
    command=[plan["interpreter"],str(Path(__file__).resolve()),"_child","--plan",str(plan_path),"--output",str(output),"--plan-sha256",expected_sha,"--execute-pixels"]
    peak=0; started=time.monotonic(); killed=None; process=None; failure=None
    tracked={}; cleanup_errors=[]; cleanup_killed=[]
    try:
        with (output/"stdout.log").open("x",encoding="utf-8") as stdout, (output/"stderr.log").open("x",encoding="utf-8") as stderr:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            # Track immediately: even writing job.json can fail after a child starts.
            try:
                child=psutil.Process(process.pid)
                tracked.update({q.pid:q for q in [child]+child.children(recursive=True)})
            except psutil.NoSuchProcess:
                pass
            dump(output/"job.json",dict(pid=process.pid,started_at=now(),argv=command,plan_sha256=expected_sha,
                fixture_sha256=plan["fixture_sha256"],caps=CAPS,automatic_retry=0))
            while process.poll() is None:
                try:
                    child=psutil.Process(process.pid); tree=[child]+child.children(recursive=True)
                    tracked.update({q.pid:q for q in tree})
                    rss=psutil.Process(os.getpid()).memory_info().rss+sum(q.memory_info().rss for q in tree if q.is_running())
                    peak=max(peak,rss)
                except psutil.NoSuchProcess:
                    rss=0
                if time.monotonic()-started>=CAPS["wall_seconds"] or rss>CAPS["rss_bytes"]:
                    killed="wall" if time.monotonic()-started>=CAPS["wall_seconds"] else "process_tree_RSS"
                    break
                time.sleep(.1)
    except BaseException as exc:
        failure=f"{type(exc).__name__}: {exc}"
    finally:
        # Always discover/stop descendants as well as the direct child. Keep
        # previously seen Process objects if their parent has already exited.
        if process is not None:
            try:
                child=psutil.Process(process.pid)
                tracked.update({q.pid:q for q in [child]+child.children(recursive=True)})
            except psutil.NoSuchProcess:
                pass
            except BaseException as exc:
                cleanup_errors.append(f"discover: {type(exc).__name__}: {exc}")
            for q in reversed(list(tracked.values())):
                try:
                    if q.is_running():
                        q.kill(); cleanup_killed.append(q.pid)
                except psutil.NoSuchProcess:
                    pass
                except BaseException as exc:
                    cleanup_errors.append(f"kill {q.pid}: {type(exc).__name__}: {exc}")
            try:
                if process.poll() is None:
                    process.kill()
                    if process.pid not in cleanup_killed:
                        cleanup_killed.append(process.pid)
                process.wait(timeout=10)
            except BaseException as exc:
                cleanup_errors.append(f"direct child wait: {type(exc).__name__}: {exc}")
            try:
                _,alive=psutil.wait_procs(list(tracked.values()),timeout=2)
                if alive:
                    cleanup_errors.append("surviving descendants: "+",".join(str(q.pid) for q in alive))
            except BaseException as exc:
                cleanup_errors.append(f"tree wait: {type(exc).__name__}: {exc}")
    elapsed=time.monotonic()-started
    if elapsed>=CAPS["wall_seconds"]:
        killed=killed or "wall_final"
    returncode=None if process is None else process.returncode
    status="PASS" if (returncode==0 and killed is None and failure is None and not cleanup_errors
        and not cleanup_killed and elapsed<CAPS["wall_seconds"] and (output/"receipt.json").is_file()) else "FAIL"
    supervisor=dict(status=status,returncode=returncode,killed=killed,error=failure,
        cleanup_killed_pids=cleanup_killed,cleanup_errors=cleanup_errors,
        peak_process_tree_rss=peak,wall_seconds=elapsed,poll_seconds=.1,
        note="Supervisor+child+descendant RSS sampled every100ms; final wall checked after cleanup",finished_at=now())
    try:
        dump(output/"supervisor.json",supervisor)
    except BaseException as exc:
        # Cleanup has already run. If the filesystem itself rejects the failure
        # receipt, preserve a machine-readable failure on the caller's stderr.
        supervisor.update(status="FAIL",receipt_write_error=f"{type(exc).__name__}: {exc}")
        print(json.dumps(supervisor),file=sys.stderr,flush=True)
        raise RuntimeError("supervisor receipt write failed after cleanup") from exc
    if status!="PASS":
        raise RuntimeError("CPU pilot failed; preserve outputs and do not retry")
    print(json.dumps(dict(status=status,output=uri(output))))



def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze"); f.add_argument("--directory",required=True)
    r=sub.add_parser("run"); r.add_argument("--plan",required=True,type=Path); r.add_argument("--plan-sha256",required=True); r.add_argument("--execute-pixels",action="store_true")
    c=sub.add_parser("_child"); c.add_argument("--plan",required=True,type=Path); c.add_argument("--output",required=True,type=Path); c.add_argument("--plan-sha256",required=True); c.add_argument("--execute-pixels",action="store_true")
    a=p.parse_args()
    if a.command=="freeze":freeze(a.directory)
    elif a.command=="run":run(a.plan,a.plan_sha256,a.execute_pixels)
    else:child_run(a.plan,a.output,a.plan_sha256,a.execute_pixels)

if __name__=="__main__":main()
