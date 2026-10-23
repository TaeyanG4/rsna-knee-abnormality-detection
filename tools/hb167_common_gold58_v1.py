"""HB167 parent common cache v1.0.3. Local serial serving adapter; no model code.

Changelog 1.0.3: validate the candidate before atomic COMPLETE publication;
finalization/worker/cleanup failures publish FAILED, and cleanup precedes commit.

Changelog 1.0.2: fresh immutable registration after shared diagnostics v1.0.3
changed during v1.0.1 CPU smoke; inference and safety logic are unchanged.

Changelog 1.0.1: Windows CPU smoke checks actual device and CUDA initialization,
not availability enumeration, while retaining present-empty CUDA_VISIBLE_DEVICES.

Changelog 1.0.0: authenticated parent extraction, five common members, CPU smoke,
review-bound single GPU job, idle-only supervision and measured memory gate.
Gold58 is EXPOSED / clean_oof=false. No score or label row is printed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hb145_output_io as io

VERSION = '1.0.3'
BUILDER = 'Codex/hb167-common'
TAG = 'HB167_COMMON_GOLD58_V103'
WORK = ROOT/'.local/scratch/hb167-common/hb167'
STAGE = ROOT/'artifacts/experiments/HB167_PARENT_COMMON/v1.0.3'
PARENT = ROOT/'.local/scratch/astra-b5/r224-gate/gate-copy-v142'
RUNTIME = WORK/'runtime'
VISIBLE = ROOT/'artifacts/experiments/EXP169_RELEASE_V5Q/v1.2.0/visible'
OWN = VISIBLE/'a5_172/exp155-assets'
QWORK = VISIBLE/'a5q_172/exp169-partner'
DICOM = ROOT/'.local/scratch/claude-hb128/dicom/train_series'
SERIES = ROOT/'storage/competition/train_series.csv'
GPU_PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
MEMBERS = ('R', 'R224', 'EXP143v2', 'EXP155', 'Q_A5q')
SCHEMA = 'HB167-parent-common/v1'
OWN_MANIFEST_SHA = 'f3ab31e13e489fb8a6abf439f13799b8a0ce39e25c933b3fc0320fdee05e5619'
Q_MANIFEST_SHA = '39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc'
OWN_HELPER_SHA = 'f6fc591e29fd3090e796a48ac65d09565eb5de9c029a00e3969b1ced5a17ca52'
ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return io.read_json(path)


def stamp(**kw):
    return dict(version=VERSION, exposure='EXPOSED', clean_oof=False, **kw)


def pin(path, expected):
    need(sha(path) == expected, 'identity mismatch: '+rel(path))


def panel():
    from tools.hb113_gold58 import gold, LABELS
    ids, _ = gold()
    with SERIES.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['StudyInstanceUID'] in ids]
    need({r['StudyInstanceUID'] for r in rows} == set(ids), 'series coverage')
    return ids, rows, list(LABELS)


def sources():
    from tools import c85r7_r224flip_diagnostics as consumer
    files = consumer.stage_files(PARENT)
    need(files == consumer.stage_files(consumer.PARENT), 'parent embedded files differ')
    need(sha(PARENT/'kernel/release.ipynb') == sha(consumer.PARENT/'kernel/release.ipynb'), 'parent notebook differs')
    return files


def assets():
    from tools import hb113_gold58 as r
    from tools import c85r7_r224_timing as r224
    files = sources()
    manifest = json.loads(files['q-a5q-manifest.json'])
    need(hashlib.sha256(files['q-a5q-manifest.json']).hexdigest() == Q_MANIFEST_SHA, 'Q manifest')
    pin(OWN/'release-manifest.json', OWN_MANIFEST_SHA)
    pin(OWN/'exp155_own0949_oai_release_v12.py', OWN_HELPER_SHA)
    own = read(OWN/'release-manifest.json')
    ck = {'R': r.PARENT, 'R224': r224.CHECKPOINT,
          'EXP143v2': OWN/'exp143-v2-e14-e11-e12-bn.pt',
          'EXP155': OWN/'exp155-v1_1-e14-e11-e12-oai-bn.pt'}
    expected = {'R': r.p956.R_SHA, 'R224': r224.CHECKPOINT_SHA,
                **{m: own['checkpoints'][ck[m].name]['sha256'] for m in ('EXP143v2','EXP155')}}
    for m, path in ck.items():
        pin(path, expected[m])
    ordered = [(e['family'], e['seed'], e['fold']) for e in manifest['entries']]
    need(ordered == [('l2',s,f) for s in (42,43) for f in range(5)] +
         [(fam,42,f) for fam in ('dino','convnext') for f in range(5)], 'Q roster order')
    for entry in manifest['entries']:
        pin(QWORK/entry['path'], entry['sha256'])
    for name, expected_sha in manifest['source_code'].items():
        pin(QWORK/'code'/name, expected_sha)
    for name in ('exp169_partner_bundle_release.py', 'exp169_release_v2.py'):
        need(files[name] == (QWORK/'code'/name).read_bytes(), 'Q embedded/source mismatch: '+name)
    return files, manifest, ck


def prepare():
    files, manifest, ck = assets()
    for name, raw in files.items():
        path = RUNTIME/name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix+'.tmp')
        io.retry(tmp.write_bytes, raw)
        io.replace(tmp, path)
    ids, rows, _ = panel()
    need(all((DICOM/u).is_dir() for u in ids), 'Gold DICOM directory coverage')
    return stamp(state='CPU_PREPARED', embedded_files=len(files), checkpoints=len(ck)+len(manifest['entries']), studies=len(ids))


def registration():
    from tools import hb149_monitor_v125, gpu_queue, c85r7_flipall_timing
    files, manifest, ck = assets()
    ids, rows, labels = panel()
    paths = [Path(__file__), SERIES, ROOT/'storage/competition/train.csv',
             ROOT/'tools/exp169_r224.py', PARENT/'kernel/release.ipynb',
             OWN/'release-manifest.json', OWN/'exp155_own0949_oai_release_v12.py',
             OWN/'exp152_own0949_release.py', *ck.values()]
    paths += [QWORK/e['path'] for e in manifest['entries']]
    paths += [QWORK/'code'/n for n in manifest['source_code']]
    # Bind actual project dependencies imported during registration. Runtime
    # inference model/decoder sources additionally come from authenticated assets.
    paths += [Path(m.__file__) for m in list(sys.modules.values())
              if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(ROOT/'tools')]
    pins = {rel(p): sha(p) for p in paths if p.suffix in ('.py','.csv','.ipynb','.json','.pt','.bin')}
    # Input bytes, not labels, bind the complete panel; no copies of DICOM data.
    dicom_pins = {rel(p): sha(p) for u in ids for p in sorted((DICOM/u).rglob('*.dcm'))}
    need(len(dicom_pins) > 58, 'empty panel inputs')
    return stamp(schema='HB167-common-registration/v1', builder=BUILDER, tag=TAG,
        ids=ids, targets=labels, rows=rows, pins=pins, dicom_pins=dicom_pins,
        embedded={n: hashlib.sha256(v).hexdigest() for n,v in files.items()},
        checkpoints={m: dict(path=rel(p), sha256=sha(p)) for m,p in ck.items()},
        q_entries=manifest['entries'], parent_kernel_sha256=sha(PARENT/'kernel/release.ipynb'),
        formula_sha256=sha(ROOT/'tools/exp169_r224.py'),
        policy=dict(serial=True, fallback='fail closed on any member/study failure',
          cpu='two studies; float32 engineering only; empty CUDA_VISIBLE_DEVICES',
          gpu='one 4070; native fp16/retry; each full-shape peak <=12 GiB and desktop+peak+2 GiB<=total',
          watchdog='idle only: 900 seconds; WDDM stall at 600 seconds; no elapsed/throughput limit',
          ranks='parent v2.aggregate once after all58; no per-batch ranks',
          scheduling='one checkpoint at a time; same per-study windows/TTA/model math; local Windows reader adapter',
          decisions='cache completeness only; independent result review before scientific use'))


def register():
    value = registration()
    path = STAGE/'registration.json'
    if path.exists():
        need(read(path) == value, 'immutable registration changed; bump version')
    else:
        io.atomic_json(path, value)
    plan = [dict(tag=TAG, command=launch_command(), continue_on_failure=False)]
    from tools.gpu_queue import validate_plan
    validate_plan(plan)
    io.atomic_json(STAGE/'gpu-plan.json', plan)
    return stamp(state='REGISTERED', sha256=sha(path), checkpoints=24, studies=58)


def validate():
    reg = read(STAGE/'registration.json')
    need(reg['version'] == VERSION and reg['builder'] == BUILDER, 'registration version/builder')
    for path, expected in {**reg['pins'], **reg['dicom_pins']}.items():
        pin(ROOT/path, expected)
    for name, expected in reg['embedded'].items():
        pin(RUNTIME/name, expected)
    ids, rows, labels = panel()
    need((ids, rows, labels) == (reg['ids'],reg['rows'],reg['targets']), 'panel/order drift')
    from tools.gpu_queue import validate_plan
    need(validate_plan(read(STAGE/'gpu-plan.json')) == [dict(tag=TAG,command=launch_command(),continue_on_failure=False)], 'queue drift')
    return reg


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def q_runtime():
    # Override both package attribute and sys.modules to prevent mutable tools
    # winning the frozen source's `from tools import ...` fallback.
    import tools
    base = load('tools.exp169_partner_bundle_release', RUNTIME/'exp169_partner_bundle_release.py')
    tools.exp169_partner_bundle_release = base
    helper = load('tools.exp152_own0949_release', RUNTIME/'exp152_own0949_release.py')
    tools.exp152_own0949_release = helper
    module = load('hb167_frozen_v2', RUNTIME/'exp169_release_v2.py')
    manifest = read(RUNTIME/'q-a5q-manifest.json')
    sys.path.insert(0, str(QWORK/'code'))
    for name in manifest['source_code']:
        if name.startswith(('raptor_', 'exp160_')):
            key = Path(name).stem
            if key in sys.modules:
                need(Path(sys.modules[key].__file__).resolve() == (QWORK/'code'/name).resolve(), 'foreign Q import: '+key)
    return module


def r224_namespace():
    import glob
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    import timm
    from tools import c85r7_r224_timing as old
    text = (RUNTIME/'raptor224_run.py').read_text(encoding='utf-8')
    tree = ast.parse(text)
    ns = dict(np=np, torch=torch, nn=torch.nn, F=torch.nn.functional, timm=timm,
              glob=glob, gc=gc, ThreadPoolExecutor=ThreadPoolExecutor, _WLUT={},
              _MEAN=torch.tensor([.485,.456,.406]).view(3,1,1),
              _STD=torch.tensor([.229,.224,.225]).view(3,1,1))
    picked = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in old.CONSTANTS:
            ns[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in old.FUNCTIONS:
            picked.append(node)
    need({n.name for n in picked} == set(old.FUNCTIONS), 'R224 function set')
    ns['MAXS'] = sum(s[2] for s in ns['SLOTS'])
    exec(compile(ast.Module(body=picked,type_ignores=[]),str(RUNTIME/'raptor224_run.py'),'exec'),ns)
    return ns


def atomic_npz(path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    def write():
        with tmp.open('wb') as f:
            np.savez(f, **arrays)
    io.retry(write)
    io.replace(tmp,path)


def check_values(values, n):
    import numpy as np
    a = np.asarray(values)
    need(a.shape==(n,12) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(), 'invalid predictions')


def check_archive(path, ids, targets):
    import numpy as np
    need(len(ids) == len(set(ids)) and ids == sorted(ids), 'unique sorted IDs required')
    with np.load(path, allow_pickle=False) as data:
        need(set(data.files) == {'ids','targets',*MEMBERS,*('failed_'+m for m in MEMBERS)}, 'archive inventory')
        need(data['ids'].tolist()==ids and data['targets'].tolist()==targets, 'archive order')
        for member in MEMBERS:
            check_values(data[member],len(ids))
            mask=data['failed_'+member]
            need(mask.shape==(len(ids),) and mask.dtype==np.bool_ and not mask.any(), 'failed member')
        need(not np.any(np.all(data['R224']==.5,axis=1)), 'R224 neutral row')


def review_binding():
    return dict(registration_sha256=sha(STAGE/'registration.json'),
                smoke_sha256=sha(WORK/'smoke/receipt.json'), tool_sha256=sha(__file__),
                plan_sha256=sha(STAGE/'gpu-plan.json'))


def check_review(value, binding):
    need(value.get('schema')=='HB167-common-review/v1' and value.get('verdict')=='PASS', 'independent PASS required')
    need(value.get('reviewer') and value['reviewer'] != BUILDER and value.get('builder')==BUILDER, 'builder cannot review self')
    need(value.get('binding')==binding and value.get('member_identity_checked') is True, 'review binding/identity missing')
    need(value.get('cpu_smoke_checked') is True and value.get('serial_memory_gate_checked') is True, 'review checks missing')
    report = (ROOT/value['report_path']).resolve()
    need(report.is_relative_to(ROOT) and sha(report)==value['report_sha256'], 'review report missing/drift')


def gpu_review():
    smoke = read(WORK/'smoke/receipt.json')
    need(smoke.get('state')=='CPU_SMOKE_ONLY' and smoke.get('studies')==2 and
         smoke.get('registration_sha256')==sha(STAGE/'registration.json'), 'current CPU smoke required')
    check_review(read(STAGE/'independent-review.json'), review_binding())


def launch_command():
    return [GPU_PYTHON,'-B',str(Path(__file__).resolve()),'run']


def worker(smoke=False):
    need(not smoke or ('CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES']==''), 'CPU must have present empty CUDA_VISIBLE_DEVICES')
    reg = validate()
    if not smoke:
        gpu_review()
        import psutil
        parents = psutil.Process().parents()
        need(any(str(Path(__file__).resolve()).lower() in ' '.join(p.cmdline()).lower() and 'run' in p.cmdline() for p in parents), 'supervised private worker required')
        need(any('gpu_job.py' in ' '.join(p.cmdline()) for p in parents), 'gpu_job ancestor required')
    import numpy as np
    import torch
    from tools import hb113_gold58 as r
    from tools import c85r7_flipall_timing as memory
    need(not smoke or not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    device = 'cpu' if smoke else 'cuda:0'
    out = WORK/('smoke' if smoke else 'gpu')
    ids = reg['ids'][:2] if smoke else reg['ids']
    rows = [x for x in reg['rows'] if x['StudyInstanceUID'] in ids]
    grouped = {u:[x for x in rows if x['StudyInstanceUID']==u] for u in ids}
    desktop = None if smoke else read(out/'desktop.json')
    started = time.monotonic()
    sequence = 0
    measurements = []
    def progress(phase):
        nonlocal sequence
        sequence += 1
        # Nonfatal heartbeat; predictions have independent durable commits.
        io.receipt(out/'status.json', stamp(state='RUNNING',sequence=sequence,phase=phase,elapsed=time.monotonic()-started))
    def measured(member,index):
        if smoke:
            return
        torch.cuda.synchronize()
        reserved = torch.cuda.max_memory_reserved()
        memory.memory_admission(desktop,[reserved])
        measurements.append(dict(member=member,study_index=index,peak_reserved_bytes=reserved,
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        io.atomic_json(out/'memory.json', measurements)
    arrays = {}
    member_seconds = {}
    # Original public serving source is executed as a module, not its main().
    nsmod = load('hb167_public_serving',RUNTIME/'public-serving.py')
    own = load('hb167_own_adapter',OWN/'exp155_own0949_oai_release_v12.py')
    public = load('hb167_public_adapter',RUNTIME/'exp152_own0949_release.py')
    decoder = r.fastread(nsmod)
    reader = decoder.make_builder(nsmod.FASTREAD_VARIANT,nsmod.IMG,nsmod.SLOTS,
                                 hdr_threads=nsmod.FR_HDR,px_threads=nsmod.FR_PX)
    if smoke:
        nsmod.infer_probs = r.cpu_probs
    for member in MEMBERS[:4]:
        tick = time.monotonic()
        progress(member+':load')
        ck = reg['checkpoints'][member]
        if not smoke:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        if member=='R224':
            ns = r224_namespace()
            torch.backends.cudnn.benchmark=True
            torch.backends.cuda.matmul.allow_tf32=True
            model = ns['load_model'](str(ROOT/ck['path']),ns['ARMS'][0]['arch'],224,device)[0]
            rr = ns['_make_reader']()
        else:
            ns = nsmod.__dict__
            helper = public if member=='R' else own
            ns.update(_E152_MANIFEST=({'checkpoints':{Path(ck['path']).name:{'sha256':ck['sha256']}}}
                      if member=='R' else read(OWN/'release-manifest.json')),
                      _E152_FILES={Path(ck['path']).name: ROOT/ck['path']})
            model = helper.checked_model(ns,ROOT/ck['path'],ARCH,384,device)[0]
        values=[]
        need(not smoke or all(p.device.type=='cpu' for p in model.parameters()), 'CPU model device')
        for i, uid in enumerate(ids):
            progress(member+':decode')
            vol, mask = (ns['build_study'](uid,grouped,str(DICOM),rr) if member=='R224' else reader(uid,grouped,str(DICOM)))
            need(vol.shape==(96,384,384) and mask.any() and np.any(vol), 'empty native decode')
            plain=ns['gpu_windows'](vol,mask,94,224 if member=='R224' else 320,device)
            progress(member+':infer')
            if member=='R224':
                value=ns['infer_probs'](model,plain,device)
            else:
                mirror=ns['gpu_windows'](vol,mask,94,320,device,mirror=True)
                value=ns['infer_pair_probs'](model,plain,mirror,device)
                del mirror
            check_values(np.asarray([value]),1)
            values.append(value)
            measured(member,i)
            atomic_npz(out/(member+'-partial.npz'),ids=np.asarray(ids[:i+1]),values=np.asarray(values))
            progress(member+':committed')
            del plain,vol,mask
        arrays[member]=np.asarray(values)
        member_seconds[member]=time.monotonic()-tick
        del model
        gc.collect()
    q = q_runtime()
    q.reset_backend()
    torch.use_deterministic_algorithms(True)
    from raptor_dicom import DicomCorpus,DirectoryDicomSource
    from raptor_common import select_centers,windows_from
    from raptor_radimagenet import center3_windows_from,normalise
    from raptor_runtime import normalize_input
    corpus=DicomCorpus.from_records(ids,rows,DirectoryDicomSource(DICOM))
    # Persist a bounded panel input cache, then load one study at a time. No
    # modified model code, view count or crop recipe is used to fit memory.
    qstart=time.monotonic()
    for i, uid in enumerate(ids):
        progress('Q:decode')
        vol,mask=corpus.volume(uid),corpus.mask(uid)
        need(mask.any() and np.any(vol), 'empty Q decode')
        centers=select_centers(mask,corpus.depth,36,corpus.spec,'within-slot-valid')
        center3=center3_windows_from(vol,centers,224)
        atomic_npz(out/'q-inputs'/f'{i}.npz',rad=normalise(center3,'radimagenet'),
                   img=normalize_input(center3,'imagenet'),
                   triplet=normalize_input(windows_from(vol,centers,224),'imagenet'),
                   flags=np.asarray(q.base.plane_flags(centers,corpus.spec),bool))
        progress('Q:prepared')
    scores={f:np.full((10 if f=='l2' else 5,len(ids),12),np.nan,np.float32) for f in ('l2','dino','convnext')}
    for j,entry in enumerate(reg['q_entries']):
        progress(f'Q:{j}:load')
        if not smoke:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model=q._load_models(QWORK,[entry],device)[0]
        need(not smoke or all(p.device.type=='cpu' for p in model.parameters()), 'CPU Q model device')
        family=entry['family']
        slot=entry['fold']+(5 if family=='l2' and entry['seed']==43 else 0)
        for i in range(len(ids)):
            with np.load(out/'q-inputs'/f'{i}.npz',allow_pickle=False) as payload:
                x=payload['triplet' if family=='dino' else 'rad' if family=='l2' else 'img']
                flags=payload['flags']
            progress(f'Q:{j}:infer')
            plain=q.base._infer_one(model,x,device,family=='dino')
            if family=='dino':
                value=plain
            else:
                mirror=x.copy(); mirror[flags]=mirror[flags,:,:,::-1]
                value=(plain+q.base._infer_one(model,mirror,device))/2
                del mirror
            check_values(np.asarray([value]),1)
            scores[family][slot,i]=value
            measured('Q:'+str(j),i)
            atomic_npz(out/'q-partial.npz',ids=np.asarray(ids),**scores)
            progress(f'Q:{j}:committed')
            del x
        del model
        gc.collect()
    arrays['Q_A5q'],_ = q.aggregate(scores)
    member_seconds['Q_A5q']=time.monotonic()-qstart
    for m,a in arrays.items():
        check_values(a,len(ids))
    need(not np.any(np.all(arrays['R224']==.5,axis=1)), 'R224 neutral row')
    atomic_npz(out/'predictions.npz',ids=np.asarray(ids),targets=np.asarray(reg['targets']),
        **arrays,**{'failed_'+m:np.zeros(len(ids),bool) for m in MEMBERS})
    atomic_npz(out/'q-leaves.npz',ids=np.asarray(ids),targets=np.asarray(reg['targets']),**scores)
    need(not smoke or not torch.cuda.is_initialized(), 'CPU smoke initialized CUDA')
    io.atomic_json(out/'worker.json',stamp(state='INFERENCE_COMPLETE',studies=len(ids),
        registration_sha256=sha(STAGE/'registration.json'),
        seconds=time.monotonic()-started,member_seconds=member_seconds,device=device,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cuda_initialized=torch.cuda.is_initialized(),
        torch=str(torch.__version__),precision='CPU fp32 engineering only' if smoke else 'parent CUDA fp16 with native R224 fp32 retry',
        predictions_sha256=sha(out/'predictions.npz'),q_leaves_sha256=sha(out/'q-leaves.npz'),
        memory_measurements=measurements))
    progress('INFERENCE_COMPLETE')


def finalize(smoke,telemetry=None):
    out=WORK/('smoke' if smoke else 'gpu')
    try:
        io.atomic_json(out/'receipt.json',stamp(schema=SCHEMA,state='FINALIZING'))
        return _finalize(smoke,telemetry)
    except BaseException as exc:
        failed_receipt(out,exc)
        raise


def failed_receipt(out,exc):
    value=stamp(schema=SCHEMA,state='FAILED',error=f'{type(exc).__name__}: {exc}')
    # Authoritative receipt must not use the best-effort/deferred writer.
    io.atomic_json(out/'receipt.json',value)
    io.receipt(out/'failure.json',value)


def _finalize(smoke,telemetry=None):
    out=WORK/('smoke' if smoke else 'gpu')
    reg=read(STAGE/'registration.json')
    worker=read(out/'worker.json')
    need(worker['state']=='INFERENCE_COMPLETE' and worker['studies']==(2 if smoke else 58), 'incomplete worker')
    need(worker['registration_sha256']==sha(STAGE/'registration.json'), 'stale worker')
    pin(out/'predictions.npz',worker['predictions_sha256'])
    pin(out/'q-leaves.npz',worker['q_leaves_sha256'])
    check_archive(out/'predictions.npz',reg['ids'][:2] if smoke else reg['ids'],reg['targets'])
    if not smoke:
        from tools.c85r7_flipall_timing import memory_admission
        need(telemetry and telemetry['complete'], 'complete real GPU telemetry required')
        memory_admission(read(out/'desktop.json'),[x['peak_reserved_bytes'] for x in worker['memory_measurements']],telemetry['peak_gpu_mib']*2**20)
    members={}
    for m in MEMBERS:
        source='raptor224_run.py' if m=='R224' else 'exp169_release_v2.py' if m=='Q_A5q' else 'public-serving.py'
        checkpoint = digest(reg['q_entries']) if m=='Q_A5q' else reg['checkpoints'][m]['sha256']
        evidence=out/(m+'-receipt.json')
        io.atomic_json(evidence,stamp(state='CPU_SMOKE_ONLY' if smoke else 'COMPLETE',member=m,
            serving_source=source,serving_source_sha256=reg['embedded'][source],
            checkpoint_sha256=checkpoint,checkpoint_manifest=reg['q_entries'] if m=='Q_A5q' else reg['checkpoints'][m],
            checkpoint_digest_kind='canonical-json ordered 20 checkpoint entries' if m=='Q_A5q' else 'file-sha256',
            registration_sha256=sha(STAGE/'registration.json'),worker_sha256=sha(out/'worker.json'),
            predictions_sha256=sha(out/'predictions.npz'),explanation=reg['policy']))
        members[m]=dict(serving_source_sha256=reg['embedded'][source],checkpoint_sha256=checkpoint,
                        receipt_path=rel(evidence),receipt_sha256=sha(evidence))
    value=stamp(schema=SCHEMA if not smoke else 'HB167-parent-common-cpu-smoke/v1',
        state='CPU_SMOKE_ONLY' if smoke else 'COMPLETE',studies=worker['studies'],members=members,
        parent_kernel_sha256=reg['parent_kernel_sha256'],formula_sha256=reg['formula_sha256'],
        predictions_path=rel(out/'predictions.npz'),predictions_sha256=sha(out/'predictions.npz'),
        registration_sha256=sha(STAGE/'registration.json'),worker_sha256=sha(out/'worker.json'),
        seconds=worker['seconds'],telemetry=telemetry)
    if not smoke:
        value['review_sha256']=sha(STAGE/'independent-review.json')
    if not smoke:
        from tools.c85r7_r224flip_diagnostics import validate_common
        validate_common(value,reg['ids'])
    result=stamp(state=value['state'],studies=value['studies'],seconds=value['seconds'])
    io.atomic_json(out/'receipt.json',value)
    return result


def run(smoke=False):
    validate()
    from tools import hb149_monitor_v125 as monitor
    from tools import c85r7_r224_timing as guards
    from tools import c85r7_flipall_timing as memory
    import psutil
    out=WORK/('smoke' if smoke else 'gpu')
    if not smoke:
        gpu_review()
        need(not any((out/name).exists() for name in ('receipt.json','worker.json','failure.json','desktop.json')), 'GPU outputs immutable; no implicit retry')
        from tools.job_runtime import job_status,LOGS
        parents=psutil.Process().parents()
        job=job_status(LOGS,TAG,expected_command=launch_command(),tail=False)
        need(job['state']=='running' and job['pid'] in {p.pid for p in parents}, 'exclusive gpu_job lock required')
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=5)
        need(not memory.gpu_python_pids(query.stdout,{os.getpid(),*[p.pid for p in parents]}), 'other Python GPU owner')
        desktop=memory.desktop_snapshot()
        io.atomic_json(out/'desktop.json',desktop)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if smoke else '0',PYTHONDONTWRITEBYTECODE='1',
             HF_HUB_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    out.mkdir(parents=True,exist_ok=True)
    telemetry=None if smoke else guards._TimingTelemetry(env,.1,desktop)
    child=None; owner=None; tracked={}
    last=time.monotonic(); sequence=None
    try:
        if telemetry: telemetry.start()
        with (out/'worker.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([GPU_PYTHON,'-B',str(Path(__file__).resolve()),'_worker']+(['--smoke'] if smoke else []),
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            owner=monitor._WindowsJob(child)
            while child.poll() is None:
                status=io.read_receipt(out/'status.json')
                if status.get('sequence') != sequence:
                    last=time.monotonic(); sequence=status.get('sequence')
                if telemetry:
                    snapshot=telemetry.snapshot()
                    need(not snapshot['memory_failure'],snapshot['memory_failure'])
                    guards.watchdog(time.monotonic(),last,snapshot['latest_gpu'],desktop)
                time.sleep(1)
            need(child.returncode==0,'worker failed; inspect completed worker.log')
    except BaseException as exc:
        failed_receipt(out,exc)
        raise
    finally:
        try:
            try:
                if child is not None:
                    errors=monitor._cleanup(child,tracked,owner,psutil)
                    need(not errors,'worker cleanup failed: '+str(errors))
            finally:
                if telemetry: telemetry.close()
        except BaseException as exc:
            failed_receipt(out,exc)
            raise
    # Nothing that can fail during worker/telemetry cleanup follows publication.
    try:
        snapshot=None if smoke else telemetry.snapshot(final=True)
        return finalize(smoke,snapshot)
    except BaseException as exc:
        failed_receipt(out,exc)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','register','validate','smoke','run','_worker','review-binding'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare': result=prepare()
    elif args.command=='register': result=register()
    elif args.command=='validate':
        validate(); result=stamp(state='CPU_VALIDATED')
    elif args.command=='smoke': result=run(True)
    elif args.command=='run': result=run(False)
    elif args.command=='_worker':
        worker(args.smoke); return
    else: result=review_binding()
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
