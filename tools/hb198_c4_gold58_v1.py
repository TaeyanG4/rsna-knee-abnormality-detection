"""H198 B-only five-C4/same58 cache v1.0.0. No training, retry or submission.

CPU validate/admit-dry-run never import torch. `run` requires a real exact
CACHE_LAUNCH_PASS, gpu_job ancestry, current source pins and one claimed GPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import hb198_c4_bridge_v1 as bridge
from tools import hb145_output_io as output_io
from tools import strict_variant_gate_c4_v1 as gate
need,sha,read,rel=bridge.need,bridge.sha,bridge.read,bridge.rel
OUT=bridge.OUT
WORK=OUT/'gold58'
ADMISSION=OUT/'CACHE-ADMISSION.json'


def validate():
    card=read(OUT/'gold58-launch.json')
    need(card['candidate']==bridge.build.KEY and card['tag']==bridge.TAG and card['workload']==290 and
         card['expected_shape']==[5,58,12] and card['dtype']=='float32','exact cache card')
    need(card['source_sha256']==sha(bridge.__file__) and card['cache_worker_sha256']==sha(__file__) and
         card['gate_sha256']==sha(gate.__file__) and card['input_pins_sha256']==sha(OUT/'input-pins.json') and
         card['renderer_sha256']==sha(OUT/'candidate-runtime/exp169_partner_bundle_release.py'),'source pins')
    bridge.validate_pins();bridge.build.validate()
    for p,h in {**card['sources'],**card['parent_files']}.items():need(sha(ROOT/p)==h,'card input drift: '+p)
    need(sha(bridge.COMMON_REG)==read(bridge.COMMON/'worker.json')['registration_sha256'],'guard source authority')
    inherited=read(bridge.COMMON_REG)['pins']
    for p in ('tools/c85r7_r224_timing.py','tools/c85r7_flipall_timing.py'):
        need(sha(ROOT/p)==inherited[p],'registered guard source drift: '+p)
    need(card['models']==read(bridge.build.C4_DIR/'release-manifest.json')['entries'],'ordered five C4')
    for e in card['models']:need(sha(ROOT/e['source'].replace('\\','/'))==e['sha256']==e['source_sha256'],'checkpoint source/deploy')
    for n,h in card['corpus']['metadata'].items():need(sha(bridge.CORPUS/n)==h,'corpus metadata')
    queue=read(OUT/'gpu-queue.json')
    from tools.gpu_queue import validate_plan
    validate_plan(queue)
    need(queue==[dict(tag=bridge.TAG,command=command(),continue_on_failure=False)],'exact queue command')
    return card


def command():return [bridge.PYTHON,'-B',str(Path(__file__).resolve()),'run']


def admission():
    validate();saved=read(ADMISSION)
    need(saved.get('phase')=='CACHE_LAUNCH_PASS' and saved.get('fixture_only') is False,'real cache admission required')
    actual=gate.evaluate('CACHE_LAUNCH_PASS',saved['evidence'],saved['review'])
    need(actual==saved,'stale exact cache admission')
    evidence=read(gate.digest_file(saved['evidence']))
    need(evidence['physical_gpu']==0,'this local card binds physical GPU0 only')
    return evidence


def dry_run():
    validate()
    try:admission()
    except (FileNotFoundError,ValueError,KeyError) as exc:
        return dict(status='HOLD_EXPECTED_MISSING_REAL_ADMISSION',workload=290,reason=str(exc),launched=False)
    return dict(status='CACHE_ADMISSION_VALID_NO_LAUNCH',launched=False)


def atomic_npz(path,**arrays):
    import numpy as np
    tmp=Path(path).with_suffix('.npz.tmp')
    with tmp.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    output_io.replace(tmp,path)


def worker():
    # Child does not admit itself. Verify exact parent identity/nonce written only
    # after gpu_job admission, then verify gate again before initializing CUDA.
    import psutil
    launch=read(WORK/'launch.json')
    need(psutil.Process().parent().pid==launch['parent_pid'] and
         psutil.Process(launch['parent_pid']).create_time()==launch['parent_start'],'owned child parent identity')
    admission();card=validate()
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','logical cuda:0/physical0 binding')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.set_device(0);need(torch.cuda.device_count()==1,'single visible GPU')
    common,rad,runtime,_=bridge.modules();corpus=common.Corpus(bridge.CORPUS)
    files=bridge.build.unpack(read(bridge.build.OUT/'kernel/release.ipynb'))[0]
    from tools import exp169_partner_bundle_release as original
    funcs=bridge.functions(files['exp169_partner_bundle_release.py'],['model_meta','plane_flags','_infer_one'],dict(need=need,TARGETS=original.TARGETS))
    scope=dict(center3_windows_from=rad.center3_windows_from,windows_from=common.windows_from,normalise=rad.normalise,normalize_input=runtime.normalize_input)
    ids=card['corpus']['ids'];values=np.empty((5,58,12),np.float32);durations=[];peaks=[];started=time.monotonic()
    desktop=read(WORK/'desktop.json')
    from tools.c85r7_flipall_timing import memory_admission
    for f,e in enumerate(card['models']):
        output_io.atomic_json(WORK/'phase-main.json',dict(phase='model_load',fold=f,completed=len(durations)))
        p=ROOT/e['source'].replace('\\','/');ck=torch.load(p,map_location='cpu',weights_only=False);funcs['model_meta'](ck,e)
        model=common.build_model(ck['arch'],pretrained=False);model.load_state_dict(ck['model'],strict=True);model=model.eval().to('cuda:0');del ck
        torch.cuda.reset_peak_memory_stats(0)
        for i,u in enumerate(ids):
            tick=time.monotonic();volume=corpus.volume(u);bound=card['corpus']['volumes'][i]
            need(bound['uid']==u and hashlib.sha256(volume.tobytes()).hexdigest()==bound['sha256'],'exact cached study bytes')
            centers=common.select_centers(corpus.mask(u),64,36,corpus.spec,'within-slot-valid')
            need(list(centers)==bound['centers'],'exact centres')
            x=bridge.candidate_windows(volume,centers,scope)['payload'][2]
            flags=np.asarray(funcs['plane_flags'](centers,corpus.spec),bool)
            mirrored=x.copy();mirrored[flags]=mirrored[flags,:,:,::-1]
            value=(funcs['_infer_one'](model,x,'cuda:0')+funcs['_infer_one'](model,mirrored,'cuda:0'))/2
            bridge.probabilities(value[None],1);need(value.dtype==np.float32,'FP32 post-TTA')
            values[f,i]=value;torch.cuda.synchronize(0)
            peak=torch.cuda.max_memory_reserved(0);memory_admission(desktop,[peak])
            duration=time.monotonic()-tick;durations.append(duration)
            output_io.journal(WORK/'journal'/f'{f:01d}-{i:03d}.json',dict(identity=bridge.build.KEY,fold=f,index=i,seconds=duration,peak_reserved_bytes=peak),WORK)
            try:output_io.atomic_json(WORK/'phase-main.json',dict(phase='infer',fold=f,index=i,completed=len(durations)))
            except OSError:pass
        peaks.append(torch.cuda.max_memory_reserved(0));del model;torch.cuda.empty_cache()
    need(len(durations)==290 and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(),'complete finite workload')
    atomic_npz(WORK/'c4-members.npz',ids=np.asarray(ids),targets=np.asarray(card['corpus']['targets']),folds=np.arange(5),values=values)
    environment=dict(os=sys.platform,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),physical_gpu=0,logical_device='cuda:0',cudnn_benchmark=False,cudnn_deterministic=True)
    import timm
    environment['timm']=timm.__version__
    output_io.atomic_json(WORK/'worker.json',dict(schema='h198-c4-worker/v1',candidate=bridge.build.KEY,state='INFERENCE_COMPLETE',studies=58,models=5,workload=290,
        shape=list(values.shape),dtype=str(values.dtype),failures=0,npz_sha256=sha(WORK/'c4-members.npz'),card_sha256=sha(OUT/'gold58-launch.json'),
        admission_sha256=sha(ADMISSION),source_sha256=sha(__file__),renderer_sha256=card['renderer_sha256'],corpus_manifest_sha256=card['corpus']['metadata']['manifest.json'],
        model_sha256=[e['sha256'] for e in card['models']],seconds=time.monotonic()-started,study_seconds=durations,peak_reserved_bytes=peaks,environment=environment,
        exposure='EXPOSED',clean_oof=False,full_two_t4_parity=False))
    return dict(status='INFERENCE_COMPLETE',workload=290)


def run():
    evidence=admission()
    import psutil
    from tools.job_runtime import job_status,LOGS
    parents=psutil.Process().parents();job=job_status(LOGS,bridge.TAG,expected_command=command(),tail=False)
    need(job['state']=='running' and job['pid'] in {p.pid for p in parents},'exclusive gpu_job process ancestry')
    need(not WORK.exists(),'immutable fresh cache; no automatic retry')
    from tools import hb149_monitor_v125 as monitor
    from tools import c85r7_flipall_timing as memory
    from tools import c85r7_r224_timing as guards
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=5)
    need(not memory.gpu_python_pids(q.stdout,{os.getpid(),*[p.pid for p in parents]}),'another Python GPU owner')
    desktop=memory.desktop_snapshot();m=evidence['memory'];memory.memory_admission(desktop,[m['peak_reserved_bytes']])
    WORK.mkdir(parents=True);(WORK/'journal').mkdir()
    output_io.atomic_json(WORK/'desktop.json',desktop)
    output_io.atomic_json(WORK/'launch.json',dict(parent_pid=os.getpid(),parent_start=psutil.Process().create_time(),card_sha256=sha(OUT/'gold58-launch.json')))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    telemetry=guards._TimingTelemetry(env,.1,desktop);child=None;owner=None;tracked={};start=last=time.monotonic();done=0;healthy=None;rate=monitor.RateGuard(None,start)
    try:
        telemetry.start()
        with (WORK/'worker.log').open('x',encoding='utf-8') as stream:
            child=subprocess.Popen([bridge.PYTHON,'-B',str(Path(__file__).resolve()),'_worker'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
            owner=monitor._WindowsJob(child)
            while child.poll() is None:
                count=len(monitor._journal_snapshot(WORK/'journal'));now=time.monotonic()
                if count!=done:done=count;last=now
                if healthy is None and count>=6:
                    healthy=count*60/(now-start);rate.healthy=healthy
                collapse=rate.update(now,count);need(not collapse,collapse or '')
                snapshot=telemetry.snapshot();need(not snapshot['memory_failure'],snapshot['memory_failure'] or '')
                guards.watchdog(now,last,snapshot['latest_gpu'],desktop)
                time.sleep(1)
            need(child.returncode==0,'cache worker failed; inspect finished worker log')
    except BaseException as exc:
        output_io.atomic_json(WORK/'failure.json',dict(state='FAILED',error=type(exc).__name__+': '+str(exc),completed=done,automatic_retry=False));raise
    finally:
        errors=monitor._cleanup(child,tracked,owner,psutil) if child is not None else []
        telemetry.close()
        output_io.atomic_json(WORK/'monitor.json',dict(cleanup_errors=errors,telemetry=telemetry.snapshot(final=True),throughput=rate.snapshot(),seconds=time.monotonic()-start))
        need(not errors,'owned child cleanup failed')
    r=read(WORK/'worker.json');need(r['state']=='INFERENCE_COMPLETE' and r['npz_sha256']==sha(WORK/'c4-members.npz'),'complete cache receipt')
    return dict(status='INFERENCE_COMPLETE',workload=290)


def align(ids,values,wanted):
    import numpy as np
    ids=list(map(str,ids));wanted=list(map(str,wanted))
    need(len(ids)==len(set(ids)) and len(wanted)==len(set(wanted)) and set(ids)==set(wanted),'exact ID set; no duplicate/missing/extra')
    a=bridge.probabilities(values,len(ids));pos={u:i for i,u in enumerate(ids)}
    return a[[pos[u] for u in wanted]]


def parent_replay():
    import numpy as np
    import pandas as pd
    from tools import exp169_release_v2 as qparent
    from tools import c85r7_r6w958q_v1139 as parent
    card=validate();ids=card['corpus']['ids'];targets=card['corpus']['targets']
    common={};leaves={}
    with np.load(bridge.COMMON/'predictions.npz',allow_pickle=False) as z:common={k:z[k].copy() for k in z.files}
    with np.load(bridge.COMMON/'q-leaves.npz',allow_pickle=False) as z:leaves={k:z[k].copy() for k in z.files}
    need(common['ids'].tolist()==leaves.pop('ids').tolist()==ids and common['targets'].tolist()==leaves.pop('targets').tolist()==targets,'same58 leaf roster/targets')
    need(set(leaves)=={'l2','dino','convnext'},'exact leaf family set')
    for name,a in leaves.items():
        need(a.dtype==np.float32 and a.shape==(10 if name=='l2' else 5,58,12),'raw FP32 leaf shape')
        for v in a:bridge.probabilities(v,58)
    for name in ('R','R224','EXP143v2','EXP155','Q_A5q'):
        bridge.probabilities(common[name],58);mask=common['failed_'+name]
        need(mask.shape==(58,) and mask.dtype==bool and not mask.any(),'authenticated success masks')
    r6=read(bridge.R6/'controller/submission_receipt.json')
    need(r6['status']=='PASS' and r6['rows']==r6['scored']==58 and r6['fallback_studies']==0 and
         r6['fallback_reasons']=={} and r6['worker_exit_codes'] and all(x==0 for x in r6['worker_exit_codes'].values()),'R6 success/codecs receipt')
    df=pd.read_csv(bridge.R6/'predictions.csv',dtype={'StudyInstanceUID':str},float_precision='round_trip')
    need(df.columns.tolist()==['StudyInstanceUID',*targets],'R6 target order')
    common['S_R7']=align(df['StudyInstanceUID'].tolist(),df[targets].to_numpy(float),ids)
    files=bridge.build.parent()[1]
    for name in ('exp169_release_v2.py','exp169_partner_bundle_release.py'):
        need((ROOT/'tools'/name).read_bytes()==files[name],'actual aggregate source identity')
    q0,_=qparent.aggregate(leaves);need(np.array_equal(q0,common['Q_A5q']),'actual FP32 parent Q exact cache replay')
    ns=parent.math_functions(files);args=[common[n] for n in ('R','R224','S_R7','EXP143v2','EXP155','Q_A5q')]
    f0=ns['blend'](ids,*args);need(np.array_equal(f0,ns['reference'](*args)),'actual parent healthy F arithmetic')
    replacement=args.copy();replacement[-1]=q0;noop=ns['blend'](ids,*replacement)
    need(f0.tobytes()==noop.tobytes() and ns['p'].csv_bytes(ids,f0)==ns['p'].csv_bytes(ids,noop),'C4=C parent no-op float64/CSV')
    return card,ids,targets,leaves,args,ns,f0


def replay_receipt():
    card,ids,targets,leaves,args,ns,f0=parent_replay()
    bridge.save('PARENT-REPLAY.json',dict(status='PASS_PARENT_EXACT_NOOP',kernel_sha256=card['kernel_sha256'],studies=58,
        members=20,FP32_probability_means=True,full_roster_rank=True,zero_failures=True,exact_Q_cache=True,
        exact_F_reference=True,noop_float64_and_csv_exact=True,card_sha256=sha(OUT/'gold58-launch.json'),exposure='EXPOSED',clean_oof=False))
    return dict(status='PASS_PARENT_EXACT_NOOP',studies=58,members=20)


def gold_readout():
    import numpy as np
    import pandas as pd
    import io
    from tools import exp169_release_v2 as qparent
    from tools import hb128_gold58_pbranch_v1 as gold
    card,ids,targets,leaves,args,ns,f0=parent_replay()
    rec=read(WORK/'worker.json');monitor=read(WORK/'monitor.json')
    need(rec['state']=='INFERENCE_COMPLETE' and rec['workload']==290 and rec['models']==5 and rec['studies']==58 and rec['failures']==0 and
         rec['shape']==[5,58,12] and rec['dtype']=='float32' and rec['card_sha256']==sha(OUT/'gold58-launch.json') and
         rec['npz_sha256']==sha(WORK/'c4-members.npz') and rec['source_sha256']==sha(__file__) and rec['renderer_sha256']==card['renderer_sha256'], 'complete exact C4 receipt')
    need(monitor['cleanup_errors']==[] and rec['admission_sha256']==sha(ADMISSION),'successful owned cleanup/admission')
    admission()
    need(rec['model_sha256']==[e['sha256'] for e in card['models']] and
         rec['corpus_manifest_sha256']==card['corpus']['metadata']['manifest.json'] and len(rec['study_seconds'])==290 and
         all(math.isfinite(x) and x>0 for x in rec['study_seconds']),'measured model/corpus/work binding')
    with np.load(WORK/'c4-members.npz',allow_pickle=False) as z:
        need(set(z.files)=={'ids','targets','folds','values'} and z['ids'].tolist()==ids and z['targets'].tolist()==targets and z['folds'].tolist()==list(range(5)),'C4 fold/ID/target schema')
        c4=z['values'].copy()
    need(c4.shape==(5,58,12) and c4.dtype==np.float32,'C4 shape/dtype')
    for a in c4:bridge.probabilities(a,58)
    q4,_=qparent.aggregate({**leaves,'convnext':c4});replacement=args.copy();replacement[-1]=q4
    f4=ns['blend'](ids,*replacement);need(np.array_equal(f4,ns['reference'](*replacement)),'unchanged exact F oracle')
    values={};csv_sha={}
    for name,a in (('F0',f0),('F4',f4)):
        data=ns['p'].csv_bytes(ids,a);df=pd.read_csv(io.BytesIO(data),dtype={'StudyInstanceUID':str},float_precision='round_trip')
        values[name]=align(df['StudyInstanceUID'].tolist(),df[targets].to_numpy(float),ids)
        need(np.array_equal(values[name].astype(np.float32),a.astype(np.float32)),'FP32 CSV roundtrip')
        csv_sha[name]=hashlib.sha256(data).hexdigest()
    routing=read(bridge.ROUTING);need(sha(ROOT/'storage/competition/train.csv')==routing['storage/competition/train.csv'],'Gold official labels pin')
    labelids,y=gold.gold();y=align(labelids,y,ids);scores=gold.auc_table({**values,'Q0':args[-1],'Q4':q4},y,gold.bootstrap_weights(ids))
    result=dict(status='COMPLETE_EXPOSED_DESCRIPTIVE_BRIDGE',candidate=bridge.build.KEY,kernel_sha256=card['kernel_sha256'],worker_sha256=sha(WORK/'worker.json'),
        card_sha256=sha(OUT/'gold58-launch.json'),parent=gold.block(scores['F0']),candidate_effect=gold.block(scores['F4']),delta=gold.block(scores['F4']-scores['F0']),
        q_delta=gold.block(scores['Q4']-scores['Q0']),parent_noop_exact=True,all20_parent_leaves=True,all5_C4=True,failures=0,
        csv_sha256=csv_sha,exposure='EXPOSED',clean_oof=False,qualified_for_submission=False,
        decision='HOLD_INDEPENDENT_RESULT_REVIEW_AND_FULL_VISIBLE_N32_RUNTIME',bootstrap='300 paired study singleton draws seed20261003; fixed full58 ranks; conditional')
    bridge.save('GOLD58-BRIDGE.json',result)
    bridge.write(OUT/'GOLD58-REPORT.md','# H198 exposed exact-F Gold58 bridge\n\nFixed F(Q4)-F(Q0), all20 parent leaves, same58 full-roster ranks and serialized parent arithmetic.\n\n'+json.dumps(gold.clean(result['delta']),indent=2)+'\n\nEXPOSED/descriptive. No clean efficacy or automatic release claim. Independent review required.\n')
    return dict(status=result['status'],release='HOLD')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('validate','admit-dry-run','parent-replay','gold-readout','run','_worker'));a=p.parse_args()
    try:
        r=({'validate':lambda:dict(status='CARD_VALID',workload=validate()['workload']),'admit-dry-run':dry_run,'parent-replay':replay_receipt,'gold-readout':gold_readout,'run':run,'_worker':worker}[a.command])()
        print(json.dumps(r,sort_keys=True));return 0
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps(dict(status='HOLD',reason=str(exc))));return 2


if __name__=='__main__':raise SystemExit(main())
