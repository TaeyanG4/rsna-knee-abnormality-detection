"""PL15 portable runtime 1.1.1: publish frozen insurance before optional PL.

Changelog: portable fivefold native adapter; common absolute deadline; complete
cohort fallback; native kill/timeout/corruption diagnostics; measured admission.
No GPU, data access, network or process launch occurs at import time.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import pl15_transport_v110 as t
from tools import c85r7_v150_safety as safety
from tools import release_deadline_v1 as d


def unpack(dataset, manifest, destination):
    """Verify archive AND every member before making code importable."""
    path = t.check(dataset, next(x for x in manifest['files'] if x['path']=='runtime.blob'))
    with zipfile.ZipFile(path) as z:
        t.need(set(z.namelist())==set(manifest['sources']) and len(z.namelist())==len(manifest['sources']), 'source inventory')
        for name, expected in manifest['sources'].items():
            data = z.read(name)
            t.need(hashlib.sha256(data).hexdigest()==expected, 'source pin: '+name)
            t.write(t.safe_path(destination, name), data)
    return Path(destination).resolve()


def load_base(dataset, output):
    runtime = Path(output)/'base-runtime'
    pins = t.read(Path(dataset)/'base-v122-pins.json')
    with zipfile.ZipFile(Path(dataset)/'base-v122-runtime.blob') as z:
        t.need(set(z.namelist())==set(pins), 'frozen base inventory')
        for name, expected in pins.items():
            data=z.read(name)
            t.need(hashlib.sha256(data).hexdigest()==expected, 'frozen base source pin')
            t.write(t.safe_path(runtime,name),data,True)
    name='exp169_release_c85r7codec_deadline_v122'
    sys.path.insert(0,str(runtime))
    spec=importlib.util.spec_from_file_location(name,runtime/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.MODULE='pl15_base_child_v110'
    shim=(f'import sys\nsys.path.insert(0,{str(ROOT)!r})\n'
          'from exp169_release_c85r7codec_deadline_v122 import *\n'
          "if __name__ == '__main__':\n"
          '    from tools.pl15_runtime_v110 import base_child\n    base_child()\n')
    t.write(runtime/(module.MODULE+'.py'),shim.encode(),True)
    bootstrap=Path(output)/'bootstrap'
    startup=(f'import sys\nsys.path.insert(0,{str(ROOT)!r})\n'
        'from tools.c85r7_v150_safety import install_io, metadata_io\ninstall_io()\n'
        'from tools import release_metadata_v122 as _meta\n'
        '_io=metadata_io(_meta)\n_io.__enter__()\nsys.modules["release_metadata_v122"]=_meta\n')
    t.write(bootstrap/'sitecustomize.py',startup.encode(),True)
    return module,runtime,bootstrap


def base_child():
    p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--runtime',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--branch');p.add_argument('--input-root',default='/kaggle/input')
    a=p.parse_args();sys.path.insert(0,str(a.runtime))
    name='exp169_release_c85r7codec_deadline_v122'
    spec=importlib.util.spec_from_file_location(name,a.runtime/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module.MODULE='pl15_base_child_v110'
    with safety.metadata_io(module.metadata_io),r7_progress(module.metadata_io):
        safety.progress(a.branch+':start')
        module.child_entry(a.runtime,a.output,a.branch,input_root=a.input_root)


def compact_progress(source):
    """Instrument generated R7 CSV commits, leaving frozen input files intact."""
    if 'wr.writerow(' not in source or 'fh.flush()' not in source:return source
    lines=source.splitlines(True)
    t.need(sum(line.strip()=='fh.flush()' for line in lines)==1,'R7 progress anchor drift')
    for i,line in enumerate(lines):
        if line.strip()=='fh.flush()':
            indent=line[:len(line)-len(line.lstrip())]
            lines[i]=line+indent+'from tools.c85r7_v150_safety import progress as _pl15_progress\n'+indent+"_pl15_progress('S_R7:csv-row', committed=True)\n"
            break
    result=''.join(lines);compile(result,'<R7-progress>','exec');return result


@contextlib.contextmanager
def r7_progress(metadata):
    original=metadata.atomic_source
    metadata.atomic_source=lambda source:compact_progress(original(source))
    try:yield
    finally:metadata.atomic_source=original


def job(dataset, manifest, code, competition, ids, rows, output, run_id, device='cuda:0', fault='off'):
    from tools import vnext_r7_pl_v1 as pl
    dataset, code, competition, output = (Path(p).resolve() for p in (dataset, code, competition, output))
    entries=[]
    for entry in manifest['members']:
        e=dict(entry,path=str(t.safe_path(dataset,entry['path'])))
        if 'training_receipt' in entry:
            e['training_receipt']=dict(entry['training_receipt'],path=str(t.safe_path(dataset,entry['training_receipt']['path'])))
        entries.append(e)
    identity=dict(version=pl.VERSION,registration=manifest['science']['registration_sha256'],
                  run_id=run_id,branch='own',members={e['id']:e['sha256'] for e in entries})
    branch=dict(path=str(Path(output)/'own'),ids=ids,identity=identity)
    return dict(ids=ids,entries=entries,branches=[branch],rows=[r for r in rows if r[pl.UID] in set(ids)],
        images=str(Path(competition)/'test_series'),legacy=str(Path(code)/'legacy'),
        public_source=str(Path(code)/'tools/public-serving.py'),own_manifest=str(Path(dataset)/'own-manifest.json'),
        family='NATIVE',output=str(output),work=str(output),runtime_root=str(code),device=device,
        diagnostic_only=manifest['diagnostic_only'],fault=fault,parity=False)


def native_child(path):
    """Only smoke packages bypass provenance, and those can never enter release."""
    from tools import vnext_r7_pl_v1 as pl, vnext_r7_pl_worker_v1 as worker
    from tools import hb116_vnext_release as h, hb116_native_runtime as native
    j=t.read(path);out=Path(j['output']);original_record=pl.Journal.record
    original_q=h.q_model; original_native=worker.native_adapters
    original_inputs=native.shared_q_inputs
    parity=dict(input_studies=0,inference_checks=0,frozen_tta_checks=0,max_abs_error=0.)
    reference=[];native_pair=[];frozen_inputs=[]
    counter=[0]
    def record(self,uid,member,values=None,error=None,seconds=None):
        value=original_record(self,uid,member,values,error,seconds)
        safety.progress('PL:journal',committed=True)
        if error is None and values is not None and j['fault'] in ('kill','timeout'):
            t.dump(out/'fault-marker.json',dict(fault=j['fault'],after_real_commit=True))
            if j['fault']=='kill':os._exit(86)
            # Parent enforces a two-second diagnostic timeout after the commit.
            threading.Event().wait()
        return value
    @contextlib.contextmanager
    def smoke_adapters(task,host):
        t.need(j['device']=='cpu','stand-in package is CPU-only')
        # This fixed diagnostic directory is reusable without recursive deletion.
        # Invalidate the previous receipt first, then replace its binding. Real
        # release journals retain the frozen immutable/sticky-failure semantics.
        for s in task['branches']:
            t.dump(Path(s['path'])/'receipt.json', {'status':'INCOMPLETE'})
            pl.dump(Path(s['path'])/'identity.json',s['identity'])
            pl.dump(Path(s['path'])/'assignment.json',dict(ids=s['ids'],members=list(s['identity']['members'])))
        journals=[pl.Journal(s['path'],s['ids'],s['identity']) for s in task['branches']]
        routes={m:journal for journal in journals for m in journal.members}
        old_journal,old_dump=host.c.Journal,host.dump
        class Dispatch:
            def __init__(self,*args):pass
            def record(self,uid,member,values=None,error=None,seconds=None):
                # Discard only the previous diagnostic attempt's sticky status.
                # The actual result is immediately validated by Journal.record.
                key=pl.digest([uid,member])
                prior=Path(routes[member].directory)/'rows'/(key+'.json')
                t.dump(prior,dict(status='OK'))
                return routes[member].record(uid,member,values,error,seconds)
        models={}
        def q_model(entry,device):
            from tools import hb105_serving_v101 as hb
            _,torch,mil,_=hb.imports()
            pl.pin(entry)
            # Control SMOKE weights only; never manufacture production metadata.
            if entry['sha256'] not in models:
                ck=torch.load(entry['path'],map_location='cpu',weights_only=False)
                t.need(ck.get('identity',{}).get('smoke') is True,'authorized control smoke identity')
                model=mil.FindingMIL(mil.common.build_backbone(entry['arch'],pretrained=False))
                mil.strict_load(model,ck['model']);models[entry['sha256']]=model.eval().to(device)
            return models[entry['sha256']]
        host.c.Journal,host.dump,host.q_model=Dispatch,pl.dump,q_model
        try:yield journals
        finally:host.c.Journal,host.dump,host.q_model=old_journal,old_dump,original_q
    from tools import exp169_release_v6 as v6
    original_infer=v6.infer
    if j.get('parity'):
        # Decode before native.worker starts its prefetch thread and installs
        # its process-global SharedPixels decoder hook for the next study.
        h.configure()
        from tools import hb105_serving_v101 as hb
        _,torch,mil,_=hb.imports()
        sys.path.insert(0,j['legacy'])
        from raptor_dicom import DicomCorpus, DirectoryDicomSource
        from raptor_common import select_centers
        uid=j['ids'][0]
        corpus=DicomCorpus.from_records([uid],j['rows'],DirectoryDicomSource(j['images']))
        centers=select_centers(corpus.mask(uid),64,36,corpus.spec,'within-slot-valid')
        baseline=v6.mil_windows(corpus.volume(uid),centers)
        mirror=mil.flip_inputs(torch.from_numpy(baseline)[None],[uid],corpus,36)[0]
        frozen_inputs.extend((baseline,mirror))
        del corpus,baseline,mirror
    def inputs(vol,centers,errors=None):
        import numpy as np
        result=original_inputs(vol,centers,errors)
        exact=v6.mil_windows(vol,centers)
        t.need(np.array_equal(result['S'],exact),'native/frozen input parity')
        # Compare every native window tensor against the frozen MIL transform.
        parity['input_studies']+=1
        return result
    def infer(model,x,device,**kwargs):
        import numpy as np
        value=original_infer(model,x,device,**kwargs)
        if j.get('parity') and counter[0]<2:
            if counter[0]==0:
                # Independent frozen decoder/window/flip path, same fitted model.
                reference.append((original_infer(model,frozen_inputs[0],device)+original_infer(model,frozen_inputs[1],device))*.5)
                frozen_inputs.clear()
            native_pair.append(value);parity['inference_checks']+=1
            if counter[0]==1:
                delta=float(np.max(np.abs((native_pair[0]+native_pair[1])*.5-reference[0])))
                t.need(delta<=1e-6,'native versus frozen K36 flip inference parity')
                parity['max_abs_error']=delta;parity['frozen_tta_checks']=1
        counter[0]+=1
        return value
    try:
        pl.Journal.record=record
        if j.get('parity'):native.shared_q_inputs=inputs;v6.infer=infer
        with safety.metadata_io(pl.metadata):
            safety.progress('PL:model-load')
            if j['diagnostic_only']:
                task=dict(j,identity={'dispatch':True},entries=[dict(e,resolved=e['path']) for e in j['entries']])
                taskpath=out/'native-job.json';t.dump(taskpath,task)
                with smoke_adapters(task,h) as journals:
                    native.worker(taskpath,j['device'])
                    for journal in journals:journal.finish()
            else:worker.native(j)
        t.dump(out/'parity.json',parity)
    except BaseException as exc:
        try:t.dump(out/'failure.json',dict(status='HOLD',error=type(exc).__name__))
        except OSError:pass
        raise
    finally:
        pl.Journal.record=original_record;h.q_model=original_q;worker.native_adapters=original_native
        native.shared_q_inputs=original_inputs;v6.infer=original_infer


def collect(ids,shards):
    import numpy as np
    from tools import vnext_r7_pl_v1 as pl
    result=np.full((len(ids),12),np.nan,np.float32);seen=set()
    for shard in shards:
        subset=shard['ids'];t.need(set(subset)<=set(ids) and not seen.intersection(subset),'PL shard assignment')
        raw,bad,reason=pl.collect(shard['path'],subset,shard['identity'])
        t.need(len(raw)==5 and not bad and reason is None,'incomplete fivefold PL journal')
        mean=np.mean(np.stack(list(raw.values())),axis=0,dtype=np.float32)
        t.need(np.isfinite(mean).all(),'PL nonfinite')
        result[[ids.index(u) for u in subset]]=mean;seen.update(subset)
    t.need(seen==set(ids) and np.isfinite(result).all(),'incomplete PL cohort')
    return result


def publish(base_payload,ids,raw,failures,shards,destination,variant='a',exits=()):
    """Base bytes are durable before PL collection; no empty/partial CSV ever wins."""
    from tools import exp169_release_c85r7_pl15_v140 as compositor
    t.need(variant in ('a','b') and ids,'publication recipe')
    t.write(destination,base_payload)
    try:
        t.need(all(x['code']==0 and x['cleanup_error'] is None for x in exits),'PL process failure')
        student=collect(ids,shards)
        values=compositor.blend(ids,raw,student,variant,failures)
        payload=compositor.base.p.csv_bytes(ids,values)
        t.write(destination,payload)
        return dict(status='COMPLETE_PL',csv_sha256=t.sha(destination),fallback=False)
    except Exception as exc:
        # Includes malformed journals and persistent optional PL export failure.
        t.need(Path(destination).read_bytes()==base_payload,'fallback publication changed')
        return dict(status='COMPLETE_BASE_FALLBACK',csv_sha256=t.sha(destination),fallback=True,reason=type(exc).__name__+': '+str(exc))


class CalibrationGuard:
    """Diagnostic first-minute observed rate; production uses sealed target rates."""
    def __init__(self):self.at=time.monotonic();self.guard=None;self.rate=None
    def update(self,now,count):
        if self.guard is None:
            if now-self.at>=60 and count>0:
                self.rate=count/(now-self.at);self.guard=safety.ThroughputGuard(self.rate,now);self.guard.count=count
            return None
        return self.guard.update(now,count)


@contextlib.contextmanager
def guarded(clock_module,output,bootstrap,rates=None):
    measurements={}
    with safety.watchdog(clock_module,output,bootstrap,rates):
        previous=clock_module.Supervisor.start
        previous_finish=clock_module.Supervisor._finish
        def start(self,args,**kwargs):
            child=previous(self,args,**kwargs)
            if rates is None and child.label not in ('setup','codecs'):
                child.throughput_guard=CalibrationGuard();measurements.setdefault(child.label,child.throughput_guard)
            return child
        def finish(self,child,reason):
            previous_finish(self,child,reason)
            guard=getattr(child,'throughput_guard',None)
            if isinstance(guard,CalibrationGuard) and guard.rate is None and child.returncode==0:
                count=safety.committed_records(child.progress_marker)
                if count>0:guard.rate=count/max(.001,time.monotonic()-guard.at)
        clock_module.Supervisor.start=start
        clock_module.Supervisor._finish=finish
        try:yield measurements
        finally:
            clock_module.Supervisor.start=previous
            clock_module.Supervisor._finish=previous_finish


def launch(jobs,clock,output,codec=None,module=None,cuda=False):
    """The fault timer shortens child life; it never creates a fresh compute budget."""
    from tools import vnext_r7_pl_v1 as pl
    shards=[];exits=[];stop=threading.Event();threads=[]
    with d.Supervisor(clock) as supervisor,pl.context_logs() as logs:
        children=[]
        for index,j in enumerate(jobs):
            work=Path(j['output']);path=work/'job.json';t.dump(path,j)
            if j['fault'] in ('kill','timeout'):t.dump(work/'fault-marker.json',dict(after_real_commit=False))
            env=clock.environment(dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            code=Path(j['runtime_root']).resolve()
            env['PYTHONPATH']=os.pathsep.join(str(code/p) for p in ('.','tools','legacy'))
            if cuda:env['CUDA_VISIBLE_DEVICES']=str(index)
            if module is not None:env=module.codecs.child_env(env,codec,branch='PL',telemetry_dir=Path(output)/'codec-telemetry')
            log=logs.open(work/'child.log')
            child=supervisor.start([sys.executable,'-B','-m','tools.pl15_runtime_v110','native-child','--job',str(path)],
                label='PL-'+str(index),env=env,cwd=str(code),stdout=log,stderr=log)
            if j['fault']=='timeout':
                def timeout(c=child,w=work):
                    while not stop.wait(.1):
                        if t.read(w/'fault-marker.json').get('after_real_commit'):
                            if not stop.wait(2):
                                with supervisor._lock:supervisor._finish(c,'injected_timeout_after_commit')
                            return
                thread=threading.Thread(target=timeout,daemon=True);thread.start();threads.append(thread)
            children.append(child);shards.extend(j['branches'])
        try:
            for child in children:
                supervisor.wait(child)
                exits.append(dict(code=child.returncode,reason=child.reason,cleanup_error=child.cleanup_error))
        finally:
            stop.set()
            for thread in threads:thread.join(3)
    return shards,exits


def projection(timings):
    """Include measured setup + slower-shard worst study with 50% margin."""
    t.need(timings and all(x.get('study_seconds') for x in timings),'runtime measurements absent')
    setup=max(x['setup_seconds'] for x in timings)
    worst=max(v for x in timings for v in x['study_seconds'].values())
    t.need(math.isfinite(setup) and math.isfinite(worst) and setup>=0 and worst>0,'invalid runtime evidence')
    student=1.5*(setup+math.ceil(1300/2)*worst)
    return dict(base_seconds=16920,student_seconds=student,total_seconds=16920+student,
                limit_seconds=27000,reserve_seconds=1200,method='1.5*(max setup + 650*max observed shard study)',
                status='PASS' if 16920+student<=27000 else 'HOLD')


def evidence(dataset,expected,mode,variant,ids,jobs,exits,output,publication,cleanup,**extra):
    timings=[t.read(Path(j['output'])/'timing.json') for j in jobs]
    parities=[t.read(Path(j['output'])/'parity.json') for j in jobs]
    result=dict(status='PASS' if publication['status']=='COMPLETE_PL' else 'HOLD',version=t.VERSION,
        manifest_sha256=expected,mode=mode,variant=variant,studies=len(ids),k=36,flip=True,
        full_shape=True,folds=5,exits=exits,timings=timings,parities=parities,publication=publication,
        cleanup=cleanup,projection=projection(timings),exposure='EXPOSED_EXPLORATORY',clean_oof=False,**extra)
    t.dump(Path(output)/'execution.json',result);return result


def run(dataset,expected,output,mode='visible',variant='a',qualification=None):
    t.need(mode in ('visible','n32','submit') and (variant=='a' or mode=='visible'),'b is visible-only')
    m=t.verify(dataset,expected,checkpoints=False)
    if mode=='submit':admit(m,expected,qualification)
    clock=d.Deadline.start(27000)
    t.need(clock.seconds==27000,'shared 7.5h deadline')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    destination=output/('submission.csv' if mode=='submit' else 'diagnostic.csv')
    t.need(not destination.exists(),'fresh output required')
    module,rt,bootstrap=load_base(dataset,output)
    from tools import exp169_c85r7ins_checks_v122 as checks
    from tools.release_cleanup_checks_v121 import CleanupAudit,publication_anchor
    old=checks.m;checks.m=module
    os.environ[module.STRESS_ENV]='1' if mode!='submit' else '0';os.environ[d.FAULT_ENV]='off'
    base_dir=output/'base';fault_base=None
    rates=qualification['healthy_rates'] if qualification else None
    try:
        with safety.metadata_io(module.metadata_io),guarded(d,output/'watchdog',bootstrap,rates) as calibration:
            if mode=='n32':
                os.environ[d.FAULT_ENV]='all';fault_base=checks.target_stress(rt,output)
                base_dir=output/'insurance-stress/clean';os.environ[d.FAULT_ENV]='off'
            else:
                with CleanupAudit() as audit:
                    module.run_release(rt,base_dir,stress=mode!='submit')
                    base_cleanup=audit.verify(publication_anchor(clock))
            base_payload=(base_dir/('submission.csv' if mode=='submit' else module.DIAGNOSTIC)).read_bytes()
            # This happens BEFORE PL hashing, allocation, launch or collection.
            t.write(destination,base_payload)
            try:
                ids,raw,failures=checks.snapshot(base_dir/module.RUN/'a172',base_dir/module.RUN/'pr7')
                t.need(ids and (mode!='n32' or len(ids)==32),'cohort shape')
                competition=module.competition_root()
                with (competition/'test_series.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
                codec=module.old.codec_receipt(base_dir/module.RUN)
                jobs=[job(dataset,m,ROOT,competition,ids[i::2],rows,output/'PL'/str(i),expected) for i in range(2) if ids[i::2]]
                for j in jobs:j['parity']=mode!='submit'
                with CleanupAudit() as audit:
                    shards,exits=launch(jobs,clock,output,codec,module,True)
                    cleanup=audit.verify(publication_anchor(clock))
                publication=publish(base_payload,ids,raw,failures,shards,destination,variant,exits)
                if mode=='submit':
                    result=dict(publication,mode=mode,manifest_sha256=expected)
                    try:t.dump(output/'execution.json',result)
                    except OSError:pass
                    return result
                t.need(not any(failures.values()),'healthy qualification base')
                faults=[]
                if mode=='n32':
                    for fault in ('kill','timeout','corrupt'):
                        work=output/'pl-faults'/fault
                        faultjob=job(dataset,m,ROOT,competition,ids,rows,work/'worker',expected,fault=fault)
                        if fault=='corrupt':
                            bad=work/'corrupt.pt';t.write(bad,b'INTENTIONALLY_CORRUPT_DIAGNOSTIC')
                            faultjob['entries'][0]['path']=str(bad)
                        with CleanupAudit() as audit:
                            fs,fe=launch([faultjob],clock,work,codec,module,True)
                            fc=audit.verify(publication_anchor(clock))
                        fp=publish(base_payload,ids,raw,failures,fs,work/'diagnostic.csv',variant,fe)
                        t.need(fp['fallback'] and (work/'diagnostic.csv').read_bytes()==base_payload,'N32 fallback bytes')
                        if fault in ('kill','timeout'):
                            t.need(t.read(work/'worker/fault-marker.json')['after_real_commit'],'native fault was not reached')
                            t.need(fe[0]['code']!=0,'fault child survived')
                        faults.append(dict(fault=fault,studies=len(ids),complete_csv=True,base_byte_identical=True,exits=fe,cleanup=fc))
                measured={k:g.rate for k,g in calibration.items() if g.rate}
                # Short visible phases may finish before a calibration minute.
                for i,j in enumerate(jobs):
                    tm=t.read(Path(j['output'])/'timing.json')
                    measured['PL-'+str(i)]=5*len(j['ids'])/tm['wall_seconds']
                return evidence(dataset,expected,mode,variant,ids,jobs,exits,output,publication,cleanup,
                    base_fault_suite=fault_base,pl_faults=faults,healthy_rates=measured,image_sha256=t.IMAGE)
            except Exception as exc:
                # Qualification failures cannot remove already published insurance.
                result=dict(status='HOLD',reason=type(exc).__name__+': '+str(exc),manifest_sha256=expected,
                            mode=mode,csv_sha256=t.sha(destination))
                try:t.dump(output/'execution.json',result)
                except OSError:pass
                return result
    finally:checks.m=old


def smoke(dataset,stage,output,device='cpu'):
    from tools import exp169_c85r7ins_checks_v122 as checks
    from tools import exp169_release_c85r7_pl15_v140 as compositor
    import numpy as np
    expected=t.sha(Path(dataset)/'manifest.json');m=t.verify(dataset,expected,allow_smoke=device=='cpu')
    t.need(m['diagnostic_only'] if device=='cpu' else not m['diagnostic_only'],'smoke package mode')
    output=Path(output).resolve();code=unpack(dataset,m,output/'portable')
    # Authenticated source closure / base bootstrap is exercised locally too.
    load_base(dataset,output)
    competition=checks.VISIBLE
    with (competition/'test_series.csv').open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    uidkey=compositor.pl.UID;ids=list(dict.fromkeys(r[uidkey] for r in rows))[:3]
    t.need(ids,'real visible images absent')
    j=job(dataset,m,code,competition,ids,rows,output/'PL','smoke-'+expected,device=device);j['parity']=True
    clock=d.Deadline.start(27000,env={})
    if device=='cpu':
        shards,exits=launch([j],clock,output)
    else:
        _,_,bootstrap=load_base(dataset,output)
        with guarded(d,output/'watchdog',bootstrap):shards,exits=launch([j],clock,output,cuda=True)
    t.need(all(x['code']==0 and x['cleanup_error'] is None for x in exits),'native smoke failed; see retained child.log')
    student=collect(ids,shards)
    # Engineering compositor fixture, explicitly not an authenticated base score.
    raw={key:student.copy() for key in ('R','S_R7','EXP143v2','EXP155','Q_A5q')}
    baseline=compositor.base.p.csv_bytes(ids,compositor.base.blend(ids,*[raw[k] for k in raw],failures={}))
    publications={v:publish(baseline,ids,raw,{},shards,output/(v+'.csv'),v,exits) for v in ('a','b')}
    t.need(all(x['status']=='COMPLETE_PL' for x in publications.values()),'smoke compositor')
    faults=[]
    if device=='cpu':
        for fault in ('kill','timeout','corrupt'):
            work=output/'faults'/fault
            fj=job(dataset,m,code,competition,ids,rows,work/'PL','smoke-'+expected,device=device,fault=fault)
            if fault=='corrupt':
                bad=work/'corrupt.pt';t.write(bad,b'INTENTIONALLY_CORRUPT_DIAGNOSTIC')
                fj['entries'][0]['path']=str(bad)
            fs,fe=launch([fj],clock,work)
            fp=publish(baseline,ids,raw,{},fs,work/'fallback.csv','a',fe)
            t.need(fp['fallback'] and (work/'fallback.csv').read_bytes()==baseline,'CPU fault fallback')
            if fault in ('kill','timeout'):
                t.need(t.read(work/'PL/fault-marker.json')['after_real_commit'],'CPU native fault not reached')
            faults.append(dict(fault=fault,base_byte_identical=True,exits=fe))
    tm=t.read(output/'PL/timing.json');parity=t.read(output/'PL/parity.json')
    t.need(parity['input_studies']>=len(ids) and parity['inference_checks']==2 and parity['frozen_tta_checks']==1,'real input/inference parity')
    if device!='cpu':t.need(0<tm['peak_reserved_bytes']<=10240*1024**2,'full K36 flip memory exceeds 10GiB')
    r=dict(status='PASS',kind='real_cpu_smoke' if device=='cpu' else 'local_cuda',
        registration_sha256=t.sha(Path(stage)/'registration.json'),sources=m['sources'],manifest_sha256=expected,
        studies=len(ids),folds=5,k=36,flip=True,full_shape=True,device=device,timings=[tm],parities=[parity],exits=exits,
        diagnostic_only=m['diagnostic_only'],checkpoint_aliases=device=='cpu',production_provenance=device!='cpu',
        publications=publications,faults=faults,exposure='EXPOSED_EXPLORATORY',clean_oof=False)
    t.dump(output/'receipt.json',r);return dict(status='PASS',kind=r['kind'],studies=len(ids),folds=5,receipt=str(output/'receipt.json'))


def qualify(dataset,cuda_path,visible_path,n32_path,output):
    expected=t.sha(Path(dataset)/'manifest.json');m=t.verify(dataset,expected)
    cuda,visible,n32=map(t.read,(cuda_path,visible_path,n32_path))
    q=dict(status='PASS',manifest_sha256=expected,local_cuda=cuda,visible=visible,n32=n32,
           evidence_sha256={k:t.sha(p) for k,p in [('cuda',cuda_path),('visible',visible_path),('n32',n32_path)]},
           healthy_rates=dict(visible.get('healthy_rates',{}),**n32.get('healthy_rates',{})))
    admit(m,expected,q);t.dump(output,q);return dict(status='ADMITTED',manifest_sha256=expected)


def admit(m,expected,q):
    t.need(not m['diagnostic_only'] and q and q.get('status')=='PASS' and q.get('manifest_sha256')==expected,'qualification HOLD')
    c,v,n=q['local_cuda'],q['visible'],q['n32']
    for r,shard_count in ((c,1),(v,2),(n,2)):
        t.need(r['status']=='PASS' and r['manifest_sha256']==expected and r['k']==36 and r['flip'] is True
               and r['full_shape'] is True and r['folds']==5 and r['clean_oof'] is False,'full-shape identity HOLD')
        t.need(r['studies']>=3 and len(r['timings'])==shard_count and len(r['parities'])==shard_count
               and len(r['exits'])==shard_count,'partial evidence HOLD')
        t.need(all(x['code']==0 and x['cleanup_error'] is None for x in r['exits']),'child cleanup HOLD')
        cohort=set()
        for tm,parity in zip(r['timings'],r['parities']):
            t.need(not tm['load_failures'] and not tm.get('telemetry_errors') and 0<tm['peak_reserved_bytes']<=10240*1024**2,'memory/telemetry HOLD')
            ids=set(tm['study_seconds'])
            t.need(ids and not cohort.intersection(ids) and str(tm['device']).startswith('cuda'),'CUDA cohort HOLD')
            t.need(parity['input_studies']>=len(ids),'parity coverage HOLD')
            cohort.update(ids)
        t.need(len(cohort)==r['studies'],'timing cohort coverage HOLD')
        t.need(all(p['inference_checks']==2 and p['frozen_tta_checks']==1 and p['input_studies']>0 and p['max_abs_error']<=1e-6 for p in r['parities']),'actual inference parity HOLD')
    t.need(c['kind']=='local_cuda' and str(c['device']).startswith('cuda') and c['production_provenance'] and not c['diagnostic_only'],'genuine local CUDA required')
    for r,mode in ((v,'visible'),(n,'n32')):
        t.need(r['mode']==mode and r['variant']=='a' and r['image_sha256']==t.IMAGE and len(r['timings'])==2,'target identity')
        t.need(all('T4' in x['gpu_name'] for x in r['timings']),'2xT4 evidence required')
        t.need(r['publication']['status']=='COMPLETE_PL' and r['cleanup']['gpu_memory_released']
               and r['cleanup']['descendants_remaining']==0,'target cleanup/fallback HOLD')
        t.need(projection(r['timings'])['status']=='PASS','projected runtime exceeds 7.5h')
    t.need(n['studies']==32 and n['base_fault_suite']['status']=='PASS_DIAGNOSTIC_ONLY','N32 base suite HOLD')
    faults=n['pl_faults'];t.need({x['fault'] for x in faults}=={'kill','timeout','corrupt'} and len(faults)==3,'native N32 missing')
    t.need(all(x['studies']==32 and x['complete_csv'] and x['base_byte_identical'] and x['cleanup']['gpu_memory_released']
               and x['cleanup']['descendants_remaining']==0 for x in faults),'N32 fallback/cleanup HOLD')
    rates=q['healthy_rates']
    t.need(set(rates)>={'own','Q_A5q','R','S_R7','PL-0','PL-1'} and all(math.isfinite(x) and x>0 for x in rates.values()),'healthy rates absent')
    return dict(status='ADMITTED',variant='a',projection=projection(n['timings']),qualified_for_submission=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['native-child','cpu-smoke','cuda-smoke','qualify'])
    p.add_argument('--job',type=Path);p.add_argument('--dataset',type=Path,default=t.WORK/'dataset')
    p.add_argument('--stage',type=Path,default=t.STAGE);p.add_argument('--output',type=Path)
    p.add_argument('--cuda',type=Path);p.add_argument('--visible',type=Path);p.add_argument('--n32',type=Path)
    a=p.parse_args()
    if a.command=='native-child':native_child(a.job);return
    if a.command=='qualify':result=qualify(a.dataset,a.cuda,a.visible,a.n32,a.output)
    else:result=smoke(a.dataset,a.stage,a.output,'cpu' if a.command=='cpu-smoke' else 'cuda:0')
    import json
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
