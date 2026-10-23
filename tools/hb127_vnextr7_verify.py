"""HB127 offline CPU evidence; no labels, network, or CUDA. Scratch is fixed."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hb127_vnextr7_release as m
h = m.h
S = m.SCRATCH
NATIVE = ROOT/'.local/scratch/sol-1/hb116-runtime/cpu-NATIVE'
OLD = ROOT/'artifacts/experiments/HB116_VNEXT/v1.0.1/release/validation'
INPUT = ROOT/'.local/exp189-cpu-v120/input'


def cpu():
    h.need(os.environ.get('CUDA_VISIBLE_DEVICES') in (None, ''), 'CUDA must be empty')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Windows PowerShell removes empty environment values.
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    torch.set_num_threads(1)
    h.need(not torch.cuda.is_initialized(), 'CPU only')


def reference(reg, raw, candidate):
    """Independent pandas ranks and per-study survivor sums; never calls release math."""
    import numpy as np
    import pandas as pd
    n = len(next(iter(raw.values())))
    def get(k): return raw.get(k, np.full((n,12), np.nan))
    def mean(arr, weights=None, fp32=False):
        out = np.full((n,12), np.nan, dtype=np.float32 if fp32 else np.float64)
        for i in range(n):
            keep = [j for j,a in enumerate(arr) if np.isfinite(a[i]).all()]
            if keep:
                w = np.asarray([weights[j] if weights else 1 for j in keep], dtype=out.dtype)
                out[i] = sum(np.asarray(arr[j][i], dtype=out.dtype)*w[k] for k,j in enumerate(keep))/sum(w)
        return out
    def rank(a):
        good = np.isfinite(a).all(1)
        out = np.full((n,12), np.nan)
        out[good] = pd.DataFrame(a[good]).rank(method='average', pct=True).to_numpy()
        return out
    families = []
    for f in ('R50','C4','S'):
        es = [e for e in reg['Q']['members'] if e['family'] == f]
        a = mean([mean([get(e['id']) for e in es if e['fold']==fold], fp32=True)
                  for fold in range(5)], fp32=True) if f=='R50' else mean([get(e['id']) for e in es], fp32=True)
        families.append(rank(a))
    q = mean(families)
    own = mean([rank(mean([get('EXP143v2'),get('S43')])),rank(get('EXP155')),rank(q)],
               [17,17,12] if candidate=='A' else [11,11,18])
    p = mean([rank(mean([get('R2-s43'),get('R2-s44')], fp32=True)),rank(get('S_R7'))])
    out = mean([rank(p),rank(own)],[.7,.3])
    out[~np.isfinite(out).all(1)] = .5
    return out.astype(np.float32), p


def recorded():
    import numpy as np
    rec = h.read(NATIVE/'receipt.json')
    h.need(rec['status']=='PASS' and rec['phases']['NATIVE']['exit_codes']==[0], 'native receipt')
    h.pin(NATIVE/'raw.npz', {'sha256':rec['raw_sha256']})
    job = h.read(NATIVE/'NATIVE/worker0/job.json')
    with np.load(NATIVE/'raw.npz', allow_pickle=False) as z:
        ids = z['ids'].tolist()
        h.need(z['targets'].tolist()==h.LABELS and ids==job['ids'], 'native schema')
        raw = {k:z[k].copy() for k in z.files if k not in ('ids','targets')}
    journal, failures = h.c.read_journal(NATIVE/'NATIVE/worker0', ids, list(raw), job['identity'])
    h.need(all(not failures[k] and np.array_equal(raw[k], journal[k].astype(np.float32)) for k in raw), 'native journals')
    old = OLD/'cpu-member-raw.npz'
    h.pin(old, {'sha256':'1e6379963659e2b2e7ee7ac7d539e34f3fd31c746ba9dd98d27ea6add35708dd'})
    with np.load(old, allow_pickle=False) as z:
        h.need(z['ids'].tolist()==ids and z['targets'].tolist()==h.LABELS, 'stand-in cohort')
        raw['S_R7'] = z['S6'].copy()
    reg = h.read(h.REG/'consults/astra3-registration-draft.json')
    return reg, ids, {k:raw[k] for k in m.roster(reg)}


def replay(out=m.OUT):
    import numpy as np
    reg, ids, raw = recorded()
    results = {}
    for candidate in ('A','B'):
        actual, _ = m.blend(reg,raw,candidate)
        ref, _ = reference(reg,raw,candidate)
        old, _ = h.blend(reg,{('S6' if k=='S_R7' else k):a for k,a in raw.items()},candidate,'R2',True)
        np.testing.assert_array_equal(actual, ref)
        np.testing.assert_array_equal(actual, old)
        h.need(h.c.p.csv_bytes(ids,actual)==h.c.p.csv_bytes(ids,old), 'stand-in CSV bytes')
        results[candidate] = dict(independent_max_abs=0., v112_max_abs=0., csv_sha256=hashlib.sha256(h.c.p.csv_bytes(ids,actual)).hexdigest())
    result = dict(status='PASS', independent_exact=True, standin_exact=True, codecs='off', studies=3,
        scope='Authenticated recorded native outputs plus S6 substituted into R7; arithmetic only, not R7 inference.',
        native_receipt_sha256=h.sha(NATIVE/'receipt.json'), raw_sha256=h.sha(NATIVE/'raw.npz'),
        standin_sha256=h.sha(OLD/'cpu-member-raw.npz'), results=results)
    h.dump(out/'validation/replay.json',result)
    return result


def reset(folder):
    p = folder.resolve()
    h.need(p.is_relative_to(S.resolve()) and p != S.resolve(), 'unsafe scratch reset')
    if p.exists(): shutil.rmtree(p)
    p.mkdir(parents=True)


def stub_codec(work, packages):
    # Reuse test fixture only. This is NOT a real GDCM decoder on Windows.
    fixture = h.load_module('hb127_codec_fixture', ROOT/'tests/test_hb126_codecs.py')
    wheel, spec = fixture.fake_wheel(work/'fixture-wheel')
    rec = m.codecs.install(wheel, work/'hook-root', wheel=spec,
        telemetry_dir=work/'c85r7-run/codecs/telemetry', base_env=dict(os.environ,PYTHONPATH=str(packages)))
    h.need(rec['status']=='ACTIVE','test stub hook not active')
    h.dump(work/'c85r7-run/codecs/install.json',rec)
    return rec


def cpu_r7_child(job):
    cpu()
    # Windows CPU diagnostic deviations only; production port/source stays frozen.
    orig_args = m.r7.worker_args
    m.r7.worker_args = lambda *a,**k:[x for x in orig_args(*a,**k) if x!='--compile']
    orig_source = m.r7.diagnostic_source
    def source(*args,**kwargs):
        folder = orig_source(*args,**kwargs)
        path = folder/'probe_dual.py'
        text = path.read_text(encoding='utf-8')
        h.need(text.count('names, visible = ["CPU"], ["-1"]')==1, 'CPU controller patch')
        path.write_text(text.replace('names, visible = ["CPU"], ["-1"]','names, visible = ["CPU"], [""]'),encoding='utf-8')
        return folder
    m.r7.diagnostic_source = source
    orig_controller = m.r7.s6._run_controller
    def controller(args,env):
        env['CUDA_VISIBLE_DEVICES']=''
        return orig_controller(args,env)
    m.r7.s6._run_controller = controller
    return m.r7_worker(job)


def r7_smoke(out=m.OUT):
    import numpy as np
    work = S/'r7-smoke'
    reset(work)
    packages = work/'packages'
    for name,digest in h.c.s6.WHEEL_HASHES.items():
        path = ROOT/'.local/exp189-cpu-v120/prepare_smoke/dataset_mount'/name
        h.pin(path,{'sha256':digest})
        h.safe_extract(path,packages)
    codec = stub_codec(work,packages)
    reg,ids,raw = recorded()
    _, identity = m.r7.verify_bundle(m.BUNDLE,m.MANIFEST_SHA)
    h.dump(work/'public/pr7-run/r7-identity.json',identity)
    job = dict(ids=ids, competition=str(INPUT/'visible_data'), bundle=str(m.BUNDLE),code=str(INPUT/'cloud_code'),
        anatomy=str(INPUT/'anatomy-fast.pt'), output=str(work/'public/pr7-run/s_r7'),
        source=str(m.r7.SOURCE), wheels=str(packages),codec=codec,work=str(work),device='cpu')
    start=time.monotonic()
    phases={}
    with m.child_environment(codec,'S_R7',work):
        os.environ['PYTHONPATH']=str(packages)+os.pathsep+os.environ.get('PYTHONPATH','')
        raw['S_R7'],info=m.run_r7(job,work,phases,command=[sys.executable,'-B',str(Path(__file__)),
            'cpu-r7-child','--job',str(work/'r7-job.json')])
    h.need(phases['S_R7']['exit_codes']==[0] and not info['failures']['S_R7'], 'real R7 CPU smoke failed')
    h.need(np.isfinite(raw['S_R7']).all(), 'R7 nonfinite')
    rec = h.read(work/'public/pr7-run/s_r7/controller/submission_receipt.json')
    h.need(all(not w['cuda_initialized'] and w['device']=='cpu' for w in h.c.s6._worker_receipts(rec)), 'CUDA initialized')
    h.save_raw(work/'combined-raw.npz',ids,raw)
    results={}
    for candidate in ('A','B'):
        got,scores=m.blend(reg,raw,candidate)
        ref,p=reference(reg,raw,candidate)
        np.testing.assert_array_equal(got,ref)
        np.testing.assert_array_equal(scores['P'],p)
        runtime=work/candidate/'runtime'
        h.dump(runtime/'build-config.json',m.config(candidate))
        result=m.finish(runtime,work/candidate/'export',reg,ids,raw,phases,info,0.,start,candidate=candidate,device='CPU diagnostic')
        results[candidate]=dict(independent_max_abs=0.,prediction_sha256=result['prediction_sha256'])
    compare=m.prior.compare_p(work/'A/export/p-branch-r7.csv',m.PUBLIC)
    summary=dict(status='PASS',studies=len(ids),seconds=time.monotonic()-start,results=results,
        manifest_sha256=m.MANIFEST_SHA,combined_raw_sha256=h.sha(work/'combined-raw.npz'),public_comparison=compare,
        codec=m.hook_integration.codec_summary(work),
        diagnostic_deviations=['Windows CPU fp32; no compile; zero loader workers; CUDA mask empty',
            'Fake gdcm version stub, genuine shared hook; uncompressed data only; no native GDCM claim'],
        target_parity='UNMEASURED: CPU R2+R7 versus heliosli original-R GPU output is not matched target evidence')
    h.dump(out/'validation/r7-smoke.json',summary)
    return dict(status='PASS',studies=3,seconds=summary['seconds'],public_comparison=compare)


def cpu_native_child(job_path):
    """Real frozen DICOM/preprocessing/journal path with cheap tensor probes."""
    import numpy as np
    import torch
    from unittest.mock import patch
    from tools import hb116_native_runtime as native, exp169_release_v6 as v6
    job=h.read(job_path)
    ns,_,_=h.anchor_setup(job,[],'cpu')
    def model(*args): return torch.nn.Linear(1,1)
    def infer(model,tensor,device,**kwargs):
        return np.full(12,torch.sigmoid(tensor.double().mean()).item(),np.float32)
    def forbidden(*args,**kwargs): raise AssertionError('CUDA forbidden in CPU smoke')
    with patch.object(torch,'set_num_threads',lambda n:None), \
         patch.object(torch.cuda,'_lazy_init',forbidden), patch.object(h,'q_model',model), \
         patch.object(h,'anchor_setup',lambda job,entries,device,failed:(ns,{e['id']:model() for e in entries},None)), \
         patch.object(v6,'infer',infer):
        native.worker(job_path,'cpu')
    h.need(not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']=='','CUDA initialized')


def native_smoke(out=m.OUT):
    import numpy as np
    work=S/'native-hook-smoke'
    reset(work)
    packages=work/'packages'
    for name,digest in h.c.s6.WHEEL_HASHES.items():
        path=ROOT/'.local/exp189-cpu-v120/prepare_smoke/dataset_mount'/name
        h.pin(path,{'sha256':digest});h.safe_extract(path,packages)
    job=h.read(NATIVE/'NATIVE/worker0/job.json')
    entries=[next(e for e in job['entries'] if e.get('family')==f) for f in ('R50','C4','S')]
    entries += [e for e in job['entries'] if e['id'] in ('EXP143v2','EXP155','S43','R2-s43','R2-s44')]
    identity=dict(version=h.VERSION,members={e['id']:e['sha256'] for e in entries})
    results={};baseline=None
    for mode in ('inactive','active'):
        folder=work/mode
        folder.mkdir()
        codec=(stub_codec(folder,packages) if mode=='active' else
               m.install_codecs(ROOT,folder,input_root=folder/'missing-wheel',root=folder/'hook-root'))
        h.need((codec['status']=='ACTIVE')==(mode=='active'),'codec test setup')
        spec=dict(job,entries=entries,identity=identity,output=str(folder/'worker'))
        h.dump(folder/'job.json',spec)
        env=dict(os.environ,PYTHONPATH=str(packages)+os.pathsep+str(ROOT),CUDA_VISIBLE_DEVICES='')
        env=m.codecs.child_env(env,codec,branch='NATIVE_O_R2',telemetry_dir=folder/'c85r7-run/codecs/telemetry')
        with (folder/'child.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'cpu-native-child',
                '--job',str(folder/'job.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
        h.need(result.returncode==0,'native smoke child failed: '+mode)
        raw,errors=h.c.read_journal(folder/'worker',job['ids'],list(identity['members']),identity)
        h.need(not any(errors.values()) and all(np.isfinite(a).all() for a in raw.values()),'native smoke failures')
        if baseline is None: baseline=raw
        else:
            for member,a in raw.items(): np.testing.assert_array_equal(a,baseline[member])
        h.save_raw(folder/'raw.npz',job['ids'],raw)
        metrics=h.read(folder/'worker/timing.json')
        h.need(all(d.get('shared_pixel_hits',0)>0 for d in metrics['details'].values()),'shared decoder unused')
        summary=m.hook_integration.codec_summary(folder)
        if mode=='active':
            h.need(any(p['branch']=='NATIVE_O_R2' for p in summary['processes']),'native hook not observed')
        results[mode]=dict(status='PASS',install_status=codec['status'],studies=len(job['ids']),
            members=len(raw),raw_sha256=h.sha(folder/'raw.npz'),codec=summary,
            shared_pixel_hits=[d['shared_pixel_hits'] for d in metrics['details'].values()])
    result=dict(status='PASS',exact_active_inactive=True,results=results,
        scope='Real 3-study DICOM, frozen shared readers, Q/own/R2 input preparation and journals; deterministic tensor probes, not trained-model inference.',
        limitation='Windows pure-Python gdcm version stub; genuine shared hook; compressed GDCM decoding not validated.')
    h.dump(out/'validation/native-smoke.json',result)
    return dict(status='PASS',studies=3,members=len(entries),exact_active_inactive=True)


TEST_FILES = [
    'test_hb116_vnext_release.py','test_exp169_release_c85.py','test_exp169_release_p956.py',
    'test_exp169_release_v5q.py','test_exp169_release_v2.py','test_exp169_partner_bundle_release.py',
    'test_exp155_release_v12.py','test_exp189_s6_port.py','test_exp152_own0949_release.py',
    'test_exp169_release_p956rplus.py','test_exp169_release_p956rplus2.py','test_hb105_serving_v101.py',
    'test_exp169_release_v6.py','test_exp169_release_c85r7.py','test_hb126_codecs.py',
    'test_hb126_decoder_probe_v11.py','test_exp169_release_c85r7codec.py','test_hb126_decoder_probe.py',
    'test_hb127_vnextr7_release.py']


def timing_smoke(out=m.OUT):
    """Exercise the new R7 timing adapter with completed real CPU receipts."""
    work=S/'timing-smoke'
    reset(work)
    native=h.read(NATIVE/'receipt.json')
    r7work=S/'r7-smoke'
    phases=dict(native['phases'], **h.read(r7work/'phases.json'))
    for source, dest in (
        (NATIVE/'NATIVE/worker0/timing.json', work/'NATIVE/worker0/timing.json'),
        (r7work/'public/pr7-run/s_r7/controller/submission_receipt.json',
         work/'public/pr7-run/s_r7/controller/submission_receipt.json')):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    timing=m.timing(work, phases, len(native['ids']))
    projection=h.projection(dict(timing=timing, studies=len(native['ids']), fixed_seconds=0.))
    h.need(timing['branches']['P']==phases['S_R7']['seconds']/len(native['ids']), 'R7 timing attribution')
    h.need(projection['limit_seconds']==28800, 'original gate changed')
    result=dict(status='PASS', scope='Timing adapter execution on real CPU receipts, not GPU runtime qualification',
        timing=timing, projection=projection,
        sources={str(p.relative_to(ROOT)):h.sha(p) for p in (
            NATIVE/'receipt.json', NATIVE/'NATIVE/worker0/timing.json',
            r7work/'phases.json', r7work/'public/pr7-run/s_r7/controller/submission_receipt.json')})
    h.dump(out/'validation/timing-smoke.json', result)
    return dict(status='PASS', scope=result['scope'], retained_limit_seconds=projection['limit_seconds'])


def tests(out=m.OUT):
    import pytest
    import torch
    from unittest.mock import patch
    reports=[]
    class Results:
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed or report.skipped: reports.append(report.outcome)
    original=subprocess.Popen
    def cpu_process(*args,**kwargs):
        # Legacy tests ask for '-1'; real children always receive the requested empty mask.
        kwargs['env']=dict(kwargs.get('env') or os.environ,CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        return original(*args,**kwargs)
    def forbidden(*args,**kwargs): raise AssertionError('CUDA forbidden in CPU tests')
    started=time.monotonic()
    with patch.object(torch,'set_num_threads',lambda n:None),patch.object(torch.cuda,'_lazy_init',forbidden), \
         patch.object(subprocess,'Popen',cpu_process):
        code=pytest.main(['-q','-p','no:cacheprovider','--basetemp=.local/pytest-tmp/sol-1',
            *['tests/'+f for f in TEST_FILES]],plugins=[Results()])
    result=dict(status='PASS' if code==0 else 'FAIL',exit_code=int(code),
        counts={k:reports.count(k) for k in ('passed','failed','skipped')},seconds=time.monotonic()-started,
        files=TEST_FILES,cpu_threads=1,real_child_cuda_mask='',
        note='Legacy mask-index tests use mocked CUDA in-process only; hardware initialization forbidden.')
    h.dump(out/'validation/tests.json',result)
    h.need(code==0,'test suite failed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['replay','r7-smoke','cpu-r7-child','native-smoke','cpu-native-child','timing-smoke','tests'])
    p.add_argument('--out',type=Path,default=m.OUT)
    p.add_argument('--job',type=Path)
    a=p.parse_args()
    cpu()
    result=globals()[a.command.replace('-','_')](a.job if a.command.startswith('cpu-') else a.out)
    if not a.command.startswith('cpu-'): print(json.dumps(result),flush=True)


if __name__=='__main__': main()
