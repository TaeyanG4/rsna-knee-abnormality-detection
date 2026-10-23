"""HB116 vNext v1.1.2: sealed A/B constants, survivor serving, offline builder.

Frozen helpers are imported, never modified. Qualification is stricter than
serving recovery. No CLI operation uploads, submits, trains, or reads labels.
"""
from __future__ import annotations
import argparse
import base64
import csv
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp169_release_c85 as c

VERSION = '1.1.2'
REG = ROOT/'artifacts/experiments/HB116_VNEXT/v1.0.0'
OUT = ROOT/'artifacts/experiments/HB116_VNEXT'/('v'+VERSION)/'release'
CHILD_TIMEOUT_SECONDS = 28800
SCRATCH = ROOT/'.local/scratch/sol-1/hb116-runtime'
CONSTANTS = {'A': [17, 17, 12], 'B': [11, 11, 18]}
NEW_DATASET = 'taeyangg4/knee-hb116-vnext-weights-v1'
OWN_DIR = ROOT/'artifacts/releases/EXP155_S42v1_1_stage_v12/assets'
V4 = ROOT/'artifacts/releases/EXP169_V4_A4_B4_s43_stage_v413'
LEGACY = ROOT/'artifacts/releases/EXP169_B172_v2_0_2_stage/assets/code'
S43_NAME = 'exp182-s43-e14-e11-e12-bn.pt'
S43_SHA = '30d9544171896a3ea755ca9774fc5e8a77d59506eac9068bb67a979fde01fc26'
REG_SHA = '096c334e9517fa923118338103a0830275cf75be8c788eb7ceb5f4ad6e2dd4dc'
DRAFT_SHA = 'b41ccb38240597e13a3afe2a235fd24d56e89cb450c02edbd526329b7c2ed47a'
Q_AUTH = 'artifacts/experiments/HB105_SERVING_V1/input-manifest.json'
Q_AUTH_SHA = '47598e384892f11809a25ca05d5e119fd080ca0041ccf8e39f606a3e9561c84b'
OWN_MANIFEST_SHA = 'f3ab31e13e489fb8a6abf439f13799b8a0ce39e25c933b3fc0320fdee05e5619'
ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
UID, LABELS = c.p.UID, list(c.p.LABELS)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name+'.tmp')
    staged.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(staged, path)


def save_raw(path, ids, raw):
    import numpy as np
    path=Path(path)
    staged=path.with_name(path.name+'.tmp')
    with staged.open('wb') as f:
        np.savez(f,ids=np.asarray(ids),targets=np.asarray(LABELS),**raw)
    os.replace(staged,path)


def pin(path, entry):
    need(Path(path).is_file(), 'missing asset: '+str(path))
    need(('bytes' not in entry or Path(path).stat().st_size == entry['bytes']) and
         sha(path) == entry['sha256'], 'asset identity: '+str(path))


def roster(reg, rx='R', s43=True):
    need(rx in ('R', 'R2'), 'Rx must be R or R2')
    return ([e['id'] for e in reg['Q']['members']] + ['EXP143v2', 'EXP155'] +
            (['S43'] if s43 else []) + (['R'] if rx == 'R' else ['R2-s43', 'R2-s44']) + ['S6'])


def verify_roster(reg):
    entries = reg['Q']['members']
    expected = {(fam, s, f) for fam, seeds in [('C4', [42]), ('R50', [42,43]), ('S', [42,43])]
                for s in seeds for f in range(5)}
    need(len(entries) == 25 and {(e['family'],e['seed'],e['fold']) for e in entries} == expected,
         'sealed Q member/family roster differs')
    need(len({e['id'] for e in entries}) == 25 and len({e['sha256'] for e in entries}) == 25,
         'duplicate member')


def bind_q(reg, authority):
    """Recover only loader metadata from the authenticated original roster."""
    pin(authority, {'sha256': Q_AUTH_SHA})
    original = {e['id']:e for e in read(authority)['members']}
    result = []
    for e in reg['Q']['members']:
        other = original[e['id']]
        need(all(e[k] == other[k] for k in ('id','family','seed','fold','arch','bytes','sha256')),
             'HB105 authority differs: '+e['id'])
        result.append(dict(e, **({'spec_sha256':other['spec_sha256']} if e['family']=='S' else {})))
    return result


def mean_survivors(arrays, weights=None, dtype='float64'):
    import numpy as np
    a = np.asarray(arrays, dtype=dtype)
    need(a.ndim == 3 and a.shape[2] == 12, 'member shape')
    good = np.isfinite(a).all(axis=2)
    w = np.ones(len(a), dtype=dtype) if weights is None else np.asarray(weights, dtype=dtype)
    den = (good*w[:,None]).sum(axis=0, dtype=dtype)
    num = (np.where(good[:,:,None], a, 0)*w[:,None,None]).sum(axis=0, dtype=dtype)
    return np.divide(num, den[:,None], out=np.full(num.shape, np.nan, dtype=dtype), where=den[:,None]>0)


def ranks(values):
    import numpy as np
    a = np.asarray(values, np.float64)
    good = np.isfinite(a).all(axis=1)
    out = np.full_like(a, np.nan)
    if good.any():
        out[good] = c.rank(a[good])
    return out


def blend(reg, raw, candidate='A', rx='R', s43=True):
    """No sentinel/neutral fill before any nested rank; FP32 native Q means."""
    import numpy as np
    need(candidate in CONSTANTS, 'candidate')
    names = roster(reg, rx, s43)
    n = len(next(iter(raw.values())))
    missing = np.full((n,12), np.nan)
    get = lambda key: np.asarray(raw.get(key, missing))
    scores = {}
    for family in ('R50','C4','S'):
        entries = [e for e in reg['Q']['members'] if e['family'] == family]
        if family == 'R50':
            folds = [mean_survivors([get(e['id']) for e in entries if e['fold']==f], dtype='float32') for f in range(5)]
            scores[family] = mean_survivors(folds, dtype='float32')
        else:
            scores[family] = mean_survivors([get(e['id']) for e in entries], dtype='float32')
    scores['Q7S'] = mean_survivors([ranks(scores[k]) for k in ('R50','C4','S')])
    scores['A'] = mean_survivors([get(k) for k in (['EXP143v2','S43'] if s43 else ['EXP143v2'])])
    scores['O'] = mean_survivors([ranks(scores['A']), ranks(get('EXP155')), ranks(scores['Q7S'])], CONSTANTS[candidate])
    scores['Rx'] = get('R') if rx == 'R' else mean_survivors([get('R2-s43'),get('R2-s44')], dtype='float32')
    scores['P'] = mean_survivors([ranks(scores['Rx']), ranks(get('S6'))])
    final = mean_survivors([ranks(scores['P']), ranks(scores['O'])], [.7,.3])
    final[~np.isfinite(final).all(axis=1)] = .5
    return final.astype(np.float32), scores


def qualify(reg, raw, rx='R', s43=True):
    import numpy as np
    expected = set(roster(reg, rx, s43))
    need(set(raw) == expected, 'missing/unexpected member, family or branch')
    shapes = {np.asarray(a).shape for a in raw.values()}
    need(len(shapes)==1 and len(next(iter(shapes)))==2 and next(iter(shapes))[1]==12 and next(iter(shapes))[0]>0,
         'prediction cohort shape')
    need(all(np.isfinite(a).all() and (a>=0).all() and (a<=1).all() for a in raw.values()),
         'member/family/branch failed; qualification closed')
    return True


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def q_model(entry, device):
    import torch
    from tools import hb105_serving_v101 as hb
    pin(entry['resolved'], entry)
    if entry['family']=='S':
        model = hb.load_model(Path(entry['resolved']), entry, device, hb.imports())
    else:
        from raptor_common import build_model
        ck = torch.load(entry['resolved'], map_location='cpu', weights_only=False)
        expected = dict(arch=entry['arch'], res=224, lab=LABELS, contract='raptor64_336_c140_s02',
                        window_policy='within-slot-valid', norm='radimagenet' if entry['family']=='R50' else 'imagenet')
        need(all(ck.get(k)==v for k,v in expected.items()), 'Q metadata differs: '+entry['id'])
        need(ck.get('input_policy') in ('center3','center3-mixed'), 'Q input policy')
        if entry['family']=='C4':
            # These SHA-pinned originals predate the explicit crop_policy key.
            need(ck.get('crop_policy','mixed')=='mixed', 'C4 crop policy')
        model = build_model(entry['arch'], pretrained=False)
        model.load_state_dict(ck['model'], strict=True)
        model.eval().to(device)
    return model


def q_inputs(vol, centers, family):
    from raptor_radimagenet import center3_windows_from, normalise
    from raptor_runtime import normalize_input
    from tools.exp169_release_v6 import mil_windows
    if family=='S':
        return mil_windows(vol, centers)
    x = center3_windows_from(vol, centers, 224)
    if family=='C4':
        x[1::2] = center3_windows_from(vol[:,48:288,48:288], centers[1::2], 224)
        return normalize_input(x,'imagenet')
    return normalise(x,'radimagenet')


def anchor_setup(job, entries, device, failed=None):
    from tools import exp155_own0949_oai_release_v12 as own
    from tools import exp152_own0949_release as public
    from tools import exp169_release_v6 as v6
    pin(job['public_source'], {'sha256':c.p.PINS['public-serving.py']})
    pin(job['own_manifest'], {'sha256':OWN_MANIFEST_SHA})
    ns = load_module('hb116_anchor_serving', Path(job['public_source'])).__dict__
    ns.update(FR_HDR=2, FR_PX=2)
    models = {}
    for e in entries:
        try:
            path = Path(e['resolved'])
            pin(path,e)
            manifest = read(job['own_manifest']) if e['id'] in ('EXP143v2','EXP155') else {'checkpoints':{path.name:{'sha256':e['sha256']}}}
            helper = own if e['id'] in ('EXP143v2','EXP155') else public
            if e['id']=='S43':
                helper = load_module('hb116_s43_adapter', Path(public.__file__))
                m = read(job['s43_manifest'])
                need(m['checkpoint_sha256']==S43_SHA and m['serving_sha256']==c.p.PINS['public-serving.py'], 'S43 identity')
                helper.OWN_NAME = S43_NAME
                def s43_meta(ck, protocol_sha):
                    expected = dict(arch=ARCH,res=384,inference_res=320,lab=LABELS,
                        contract=helper.CONTRACT,k_eval=94,norm='imagenet',window_policy='legacy',oai=False,
                        checkpoint_kind='fixed-swa-bn',swa_epochs_1based=[14,11,12],seed=43)
                    need(all(ck.get(k)==v for k,v in expected.items()) and
                         ck.get('provenance',{}).get('protocol_sha256')==protocol_sha and
                         ck['provenance'].get('run_dir')==m['run_dir'], 'S43 metadata differs')
                helper.own_meta = s43_meta
                manifest = dict(protocol_sha256=m['protocol_sha256'],own_state_schema=m['own_state_schema'],
                                checkpoints={S43_NAME:{'sha256':S43_SHA}})
            ns.update(_E152_MANIFEST=manifest,_E152_FILES={path.name:path})
            need(callable(own._shared.input_record) and callable(helper.checked_model), 'real helper ABI')
            models[e['id']] = helper.checked_model(ns,path,ARCH,384,device)[0]
        except Exception as exc:
            if failed is None: raise
            failed[e['id']] = f'{type(exc).__name__}: {exc}'
    return ns, models, v6.anchor_reader(ns)


def worker(job_path, device):
    """Same real path for local CPU evidence and target two-T4 shards."""
    if read(job_path)['family']=='NATIVE':
        from tools.hb116_native_runtime import worker as native_worker
        return native_worker(job_path, device)
    configure()
    import numpy as np
    import torch
    from tools import exp169_release_v6 as v6
    job = read(job_path)
    sys.path.insert(0,job['legacy'])
    ids, entries = job['ids'], job['entries']
    out = Path(job['output'])
    out.mkdir(parents=True,exist_ok=True)
    journal = c.Journal(out, ids, job['identity'])
    began=time.monotonic()
    family=job['family']
    failed={}
    if family in ('R50','C4','S'):
        from raptor_dicom import DicomCorpus, DirectoryDicomSource
        from raptor_common import select_centers
        plane_flags = c.a172.base.plane_flags
        corpus=DicomCorpus.from_records(ids,job['rows'],DirectoryDicomSource(job['images']))
        models={}
        for e in entries:
            try: models[e['id']]=q_model(e,device)
            except Exception as exc: failed[e['id']]=f'{type(exc).__name__}: {exc}'
    else:
        ns,models,reader=anchor_setup(job,entries,device,failed)
        rows={}
        for row in job['rows']:
            rows.setdefault(row[UID],[]).append(row)
    setup=time.monotonic()-began
    study_times={}
    for uid in ids:
        start=time.monotonic()
        for member,error in failed.items():
            journal.record(uid,member,error='model load: '+error,seconds=0.)
        try:
            if family in ('R50','C4','S'):
                vol,mask=corpus.volume(uid),corpus.mask(uid)
                need(np.count_nonzero(mask)>=3 and np.any(vol),'empty Q study')
                centers=select_centers(mask,64,36,corpus.spec,'within-slot-valid')
                plain=q_inputs(vol,centers,family)
                flags=np.asarray(plane_flags(centers,corpus.spec),bool)
                mirror=plain.copy()
                mirror[flags]=mirror[flags,:,:,::-1]
            else:
                vol,mask=reader(uid,rows,job['images'])
                need(np.count_nonzero(mask)>=3 and np.any(vol),'empty own/R study')
                plain=ns['gpu_windows'](vol,mask,94,320,device)
                mirror=ns['gpu_windows'](vol,mask,94,320,device,mirror=True)
        except Exception as exc:
            for member in models:
                journal.record(uid,member,error=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-start)
            study_times[uid]=time.monotonic()-start
            continue
        decode=time.monotonic()-start
        for member,model in models.items():
            t=time.monotonic()
            try:
                if family=='S':
                    torch.backends.cuda.matmul.allow_tf32=False
                    torch.backends.cudnn.allow_tf32=False
                    torch.set_float32_matmul_precision('highest')
                value=(v6.infer(model,plain,device,anchor=family not in ('R50','C4','S'))+
                       v6.infer(model,mirror,device,anchor=family not in ('R50','C4','S')))*np.float32(.5)
                journal.record(uid,member,value,seconds=time.monotonic()-t)
                print('MEMBER_DONE',member,uid,round(time.monotonic()-t,3),flush=True)
            except Exception as exc:
                journal.record(uid,member,error=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-t)
        study_times[uid]=time.monotonic()-start
        print('STUDY_DONE',family,uid,round(study_times[uid],3),flush=True)
    dump(out/'timing.json',dict(family=family,device=device,setup_seconds=setup,load_failures=failed,
        gpu_name=torch.cuda.get_device_name() if device.startswith('cuda') else None,
        wall_seconds=time.monotonic()-began,study_seconds=study_times,threads=2))
    # Normal interpreter exit is required; do not hide native shutdown failures.


def safe_extract(archive, destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            need(not name.startswith(('/','\\')) and '\\' not in name and ':' not in name and '..' not in Path(name).parts,
                 'unsafe archive path')
        z.extractall(destination)


def local_assets(reg):
    """Only weights/code and test DICOM metadata, never label/report files."""
    assets=SCRATCH/'assets'
    assets.mkdir(parents=True,exist_ok=True)
    s43=V4/'s43/dataset/exp169-s43-v4.zip.bin'
    pin(s43,{'sha256':'887e294832943b8a0b5afa937459167d4425d761c05c1a154623784d60c81ef5'})
    with zipfile.ZipFile(s43) as z:
        names=[n for n in z.namelist() if Path(n).name==S43_NAME]
        need(len(names)==1,'S43 archive member')
        dest=assets/S43_NAME
        if not dest.exists():
            with z.open(names[0]) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src,dst)
        (assets/'s43-manifest.json').write_bytes(z.read('release-manifest.json'))
    members=[dict(e,resolved=str(ROOT/e['path'])) for e in bind_q(reg, ROOT/Q_AUTH)]
    members += [dict(e,id='EXP143v2',resolved=str(ROOT/e['path'])) for e in reg['OWN']['A_members'] if e.get('seed')==42]
    own=OWN_DIR/'exp155-v1_1-e14-e11-e12-oai-bn.pt'
    members += [dict(id='EXP155',resolved=str(own),sha256='1b925ea8d1469e55e95796159241cd5c34f2cff3e1e6dc71e1dadc61a2355937'),
                dict(id='S43',resolved=str(dest),sha256=S43_SHA)]
    members += [dict(e,id=f"R2-s{e['seed']}",resolved=str(ROOT/e['path'])) for e in reg['P']['Rplus2']['members']]
    # R location is directly referenced by the existing P956 stage registration.
    return members,dict(legacy=str(LEGACY),public_source=str(OWN_DIR/'public-serving.py'),own_manifest=str(OWN_DIR/'release-manifest.json'),
                        s43_manifest=str(assets/'s43-manifest.json'))


def mount(slug, filename):
    return c.v5q._mount(slug,filename)/filename


def remote_assets(reg, runtime, work, rx, s43):
    assets=work/'assets'
    own_arc=mount('taeyangg4/knee-exp155-own0949-oai-v1-2-assets','exp155-assets.zip.bin')
    pin(own_arc,{'sha256':'40b8d02847bf36938d82a79e88cb1dad4296db0cc95691b8cd0d2a2ce1a7e5fa'})
    safe_extract(own_arc,assets/'own')
    own=next((assets/'own').rglob('exp155-v1_1-e14-e11-e12-oai-bn.pt')).parent
    c4_arc=mount('taeyangg4/knee-exp169-c4-v4-assets','exp169-c4-v4.zip.bin')
    pin(c4_arc,{'sha256':'df11aa92535c9487ce6852b3ac159a162a6fb8cea989ac5a625d35faff4dd634'})
    safe_extract(c4_arc,assets/'c4')
    entries=[]
    for e in bind_q(reg, runtime/'q-authority.json'):
        path=next((assets/'c4').rglob(f"convnext_s42_f{e['fold']}.pt")) if e['family']=='C4' else mount(NEW_DATASET,e['id']+'.pt')
        entries.append(dict(e,resolved=str(path)))
    entries += [dict(id='EXP143v2',resolved=str(own/'exp143-v2-e14-e11-e12-bn.pt'),sha256=reg['OWN']['A_members'][0]['sha256']),
                dict(id='EXP155',resolved=str(own/'exp155-v1_1-e14-e11-e12-oai-bn.pt'),sha256='1b925ea8d1469e55e95796159241cd5c34f2cff3e1e6dc71e1dadc61a2355937')]
    if s43:
        archive=mount('taeyangg4/knee-exp169-s43-v4-assets','exp169-s43-v4.zip.bin')
        pin(archive,{'sha256':'887e294832943b8a0b5afa937459167d4425d761c05c1a154623784d60c81ef5'})
        safe_extract(archive,assets/'s43')
        entries.append(dict(id='S43',resolved=str(next((assets/'s43').rglob(S43_NAME))),sha256=S43_SHA))
    if rx=='R':
        entries.append(dict(id='R',resolved=str(mount(c.p.DATASETS[0],c.p.R_NAME)),sha256=c.p.R_SHA))
    else:
        manifest=reg['P']['Rplus2']['manifest']
        pin(mount('taeyangg4/knee-hb113-rplus-weights-v2','manifest.json'),manifest)
        for e in reg['P']['Rplus2']['members']:
            entries.append(dict(e,id=f"R2-s{e['seed']}",resolved=str(mount('taeyangg4/knee-hb113-rplus-weights-v2',Path(e['path']).name))))
    return entries,dict(legacy=str(runtime/'legacy'),public_source=str(own/'public-serving.py'),own_manifest=str(own/'release-manifest.json'),
                        s43_manifest=str(assets/'s43/release-manifest.json'))


def run_phases(reg, entries, paths, comp, work, device='cpu', selected=None):
    import numpy as np
    ids,_=c.v5q._read_csv(comp/'sample_submission.csv')
    with (comp/'test_series.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    # Identity failures are fatal before any operational child recovery.
    for entry in entries:
        pin(entry['resolved'],entry)
    phases={}
    raw={e['id']:np.full((len(ids),12),np.nan,np.float32) for e in entries}
    for family in selected or ('NATIVE',):
        part=entries if family=='NATIVE' else [e for e in entries if (e.get('family')==family if family in ('R50','C4','S') else
              e['id'] in (('EXP143v2','EXP155','S43') if family=='OWN' else ('R','R2-s43','R2-s44')))]
        if not part:
            continue
        t=time.monotonic()
        count=1 if device=='cpu' else 2
        children=[]
        for shard in range(count):
            shard_ids=ids[shard::count]
            if not shard_ids:
                continue
            out=work/family/f'worker{shard}'
            need(not (out/'journal-identity.json').exists(), 'stale worker journal')
            identity={'version':VERSION,'members':{e['id']:e['sha256'] for e in part}}
            job=dict(paths,ids=shard_ids,rows=[r for r in rows if r[UID] in shard_ids],images=str(comp/'test_series'),
                     entries=part,family=family,output=str(out),identity=identity)
            job_path=out/'job.json'
            dump(job_path,job)
            log=(out/'child.log').open('w',encoding='utf-8')
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            if device!='cpu':
                env['CUDA_VISIBLE_DEVICES']=str(shard)
            failure={}
            try:
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--job',str(job_path),
                                       '--device','cpu' if device=='cpu' else 'cuda:0'],env=env,stdout=log,stderr=subprocess.STDOUT)
            except (OSError,subprocess.SubprocessError) as exc:
                proc=None
                failure=dict(exit_code=1,error=f'{type(exc).__name__}: {exc}')
            children.append((proc,log,out,shard_ids,identity,failure))
        exits=[]
        failures=[]
        for proc,log,out,part_ids,identity,failure in children:
            try:
                if proc is not None:
                    try:
                        failure['exit_code']=proc.wait(timeout=CHILD_TIMEOUT_SECONDS)
                    except (OSError,subprocess.SubprocessError) as exc:
                        failure.update(exit_code=1,error=f'{type(exc).__name__}: {exc}')
                        try:
                            proc.kill()
                            proc.wait(timeout=10)
                        except (OSError,subprocess.SubprocessError) as cleanup:
                            failure['cleanup_error']=f'{type(cleanup).__name__}: {cleanup}'
                if failure['exit_code'] and 'error' not in failure:
                    failure['error']=f"ChildExit: {failure['exit_code']}"
            finally:
                log.close()
            exits.append(failure['exit_code'])
            dump(out/'exit.json',failure)
            if failure['exit_code']:
                failures.append(dict(worker=out.name,**failure))
            # Never downgrade journal identity violations to operational failures.
            if (out/'journal-identity.json').exists():
                vals,fails=c.read_journal(out,part_ids,[e['id'] for e in part],identity)
                for name,a in vals.items():
                    for j,uid in enumerate(part_ids):
                        if uid not in fails[name]:
                            raw[name][ids.index(uid)]=a[j]
        phases[family]=dict(seconds=time.monotonic()-t,exit_codes=exits,studies=len(ids),failures=failures)
        dump(work/'phases.json',phases)
        print('PHASE_DONE',family,json.dumps(phases[family]),flush=True)
    return ids,raw,phases


def run_s6(job, work, phases):
    """Recover authenticated partial S6 output after operational child failures."""
    import numpy as np
    pdir=Path(job['output']).parents[1]
    dump(pdir/'ids.json',job['ids'])
    dump(work/'s6-job.json',job)
    started=time.monotonic()
    record={}
    with (work/'s6-child.log').open('w',encoding='utf-8') as log:
        try:
            result=subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'s6-worker',
                                   '--job',str(work/'s6-job.json')],stdout=log,stderr=subprocess.STDOUT,
                                  timeout=CHILD_TIMEOUT_SECONDS)
            record['exit_code']=result.returncode
            if result.returncode:
                record['error']=f'ChildExit: {result.returncode}'
        except (OSError,subprocess.SubprocessError) as exc:
            record=dict(exit_code=1,error=f'{type(exc).__name__}: {exc}')
    dump(work/'s6-exit.json',record)
    phases['S6']=dict(seconds=time.monotonic()-started,exit_codes=[record['exit_code']],
                      studies=len(job['ids']),failures=[record] if record['exit_code'] else [])
    dump(work/'phases.json',phases)
    # Identity validation deliberately sits outside the operational catch.
    _,_,values,info=c.authenticated_p956(pdir)
    values=np.asarray(values,np.float32)
    for i,uid in enumerate(job['ids']):
        if uid in info['failures']['S6']:
            values[i]=np.nan
    return values,info


def s6_worker(job_path):
    job=read(job_path)
    if job.get('device','cuda') != 'cpu':
        c.s6.install_wheels(sorted(Path(job['wheels']).glob('*.whl')),Path(job['code']))
    c.s6.infer_studies([Path(job['competition'])/'test_series'/uid for uid in job['ids']],
        bundle_dir=Path(job['bundle']),code_dir=Path(job['code']),anatomy_weights=Path(job['anatomy']),
        output_dir=Path(job['output']),workers=0,amp='auto',device=job.get('device','cuda'),max_fallback_fraction=1.,
        source_dir=Path(job['source']))


def timings(work, phases, n):
    import numpy as np
    from scipy.stats import t as student_t
    members={}; families={}; errors=[]
    for family,phase in phases.items():
        samples=[]; worker_service=[]; setups=[]; hardware=[]; memory=[]
        for folder in (work/family).glob('worker*'):
            if (folder/'telemetry-errors.json').exists():
                errors.extend(read(folder/'telemetry-errors.json'))
            if (folder/'timing.json').exists():
                info=read(folder/'timing.json')
                errors.extend(info.get('telemetry_errors', []))
                vals=list(info['study_seconds'].values())
                samples += vals; worker_service.append(sum(vals)); setups.append(info['setup_seconds'])
                hardware.append(info.get('gpu_name'))
                if family=='NATIVE':
                    memory.append({k:info.get(k,0) for k in ('peak_allocated_bytes','peak_reserved_bytes','total_device_bytes','resident_model_bytes')})
            if (folder/'predictions.jsonl').exists():
                for line in (folder/'predictions.jsonl').read_text().splitlines(keepends=True):
                    if not line.endswith('\n'): continue  # aborted final journal write
                    try:
                        row=json.loads(line)
                        members.setdefault(row['member'],[]).append(row.get('seconds',0.))
                    except (ValueError,KeyError) as exc: errors.append(str(exc))
        if family=='S6':
            controller=read(work/'public/p956-run/s6/controller/submission_receipt.json')
            for worker in c.s6._worker_receipts(controller):
                vals=[sum(float(x) for x in row[2:6]) for row in worker['per_study']]
                samples+=vals; worker_service.append(sum(vals))
            members['S6']=samples
        upper=float(np.mean(samples)+student_t.ppf(.95,len(samples)-1)*np.std(samples,ddof=1)/np.sqrt(len(samples))) if len(samples)>1 else None
        families[family]=dict(wall_seconds=phase['seconds'],seconds_per_study=phase['seconds']/n,
            setup_upper_seconds=max(0.,phase['seconds']-max(worker_service,default=0.)),
            model_setup_seconds=setups,gpu_names=hardware,study_seconds=samples,memory=memory,
            service_seconds_per_study=float(np.mean(samples)) if samples else None,
            upper95_study_seconds=upper)
    return dict(members={k:dict(seconds=sum(v),seconds_per_study=sum(v)/n,count=len(v)) for k,v in members.items()},families=families,
                branches={'NATIVE_O_PLUS_RX':phases.get('NATIVE',{}).get('seconds',0)/n,
                          'O':sum(phases.get(k,{}).get('seconds',0) for k in ('R50','C4','S','OWN'))/n,
                          'P':sum(phases.get(k,{}).get('seconds',0) for k in ('Rx','S6'))/n},errors=errors,
                units='seconds/study; NATIVE jointly serves O and Rx; old O/Rx phase attribution unavailable when fused; members=service')


def projection(receipt):
    """Conservative two-lane service estimate, including startup and tail imbalance."""
    import numpy as np
    from scipy.stats import t as student_t
    timing=receipt['timing']
    need(not timing.get('errors'), 'timing telemetry incomplete')
    fam=timing['families']
    need(set(fam) in ({'NATIVE','S6'}, {'R50','C4','S','OWN','Rx','S6'}), 'timing family missing')
    upper=[]
    need(np.isfinite(receipt['fixed_seconds']) and receipt['fixed_seconds']>=0,'invalid fixed timing')
    for f in fam.values():
        samples=np.asarray(f['study_seconds'],dtype=np.float64)
        need(samples.ndim==1 and len(samples)==receipt['studies'] and len(samples)>1 and
             np.isfinite(samples).all() and (samples>=0).all(),'timing samples incomplete')
        need(np.isfinite(f['setup_upper_seconds']) and f['setup_upper_seconds']>=0,'invalid setup timing')
        mean=float(np.mean(samples))
        computed=float(mean+student_t.ppf(.95,len(samples)-1)*np.std(samples,ddof=1)/np.sqrt(len(samples)))
        for key,value in [('upper95_study_seconds',computed),('service_seconds_per_study',mean)]:
            if key=='upper95_study_seconds' or key in f:
                need(f.get(key) is not None and np.isfinite(f[key]) and
                     np.isclose(f[key],value,rtol=1e-10,atol=1e-10),'inconsistent timing summary: '+key)
        upper.append(computed)
    fixed=receipt['fixed_seconds']+sum(f['setup_upper_seconds']+max(f['study_seconds']) for f in fam.values())
    marginal=sum(x/2 for x in upper)
    need(np.isfinite(fixed+1300*marginal),'nonfinite runtime projection')
    return dict(fixed_seconds=fixed,marginal_seconds=marginal,studies=1300,
                projected_seconds=fixed+1300*marginal,limit_seconds=28800,
                method='sum phase one-sided Student-t 95% service means / 2; startup + one study tail per phase')


def serve(runtime, output, candidate='A', rx='R', s43=True, stress=False):
    configure()
    import numpy as np
    start=time.monotonic()
    pin(runtime/'sealed-registration.json',{'sha256':DRAFT_SHA})
    config=read(runtime/'build-config.json')
    need(config==dict(version=VERSION,candidate=candidate,constants=CONSTANTS[candidate],rx=rx,s43=s43,stress=stress),'build config differs')
    import torch
    gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    need(len(gpu_names)==2 and all('T4' in x for x in gpu_names),'exact two T4 target required')
    reg=read(runtime/'sealed-registration.json')
    verify_roster(reg)
    need(not (output/'hb116-run').exists() and not (output/'submission.csv').exists(),'stale output')
    work=output/'hb116-run'; work.mkdir(parents=True)
    comp=c.v5q._mount(c.p.COMPETITION,'test.csv')
    if stress:
        from tools import exp169_c85_stress
        comp=output/'hb116-stress-data'
        exp169_c85_stress.prepare(c.v5q._mount(c.p.COMPETITION,'train_series.csv'),comp,n=32)
    entries,paths=remote_assets(reg,runtime,work,rx,s43)
    bundle=c.v5q._mount(c.s6.DATASET_SLUG,'manifest.json')
    c.s6.verify_bundle(bundle)
    c.s6.verify_source(runtime/'tools/exp189_s6_src')
    code,anatomy=c.s6.prepare_code(c.v5q._mount(c.v5q.CODE_SLUG,'anatomy.pt'),work/'s6-code',install_wheels=False)
    setup=time.monotonic()-start
    ids,raw,phases=run_phases(reg,entries,paths,comp,work,device='cuda')
    pdir=work/'public'; dump(pdir/'ids.json',ids)
    job=dict(ids=ids,competition=str(comp),bundle=str(bundle),code=str(code),anatomy=str(anatomy),
             output=str(pdir/'p956-run/s6'),source=str(runtime/'tools/exp189_s6_src'),wheels=str(work/'s6-code/wheels'))
    raw['S6'],info=run_s6(job,work,phases)
    return finish_run(runtime,output,reg,ids,raw,phases,info,setup,start,candidate,rx,s43,stress,gpu_names)


def finish_run(runtime, output, reg, ids, raw, phases, info, setup, start,
               candidate='A', rx='R', s43=True, stress=False, gpu_names=(), device='2xT4'):
    """Common survivor export for target serving and the real CPU recovery smoke."""
    import numpy as np
    work=output/'hb116-run'
    config=read(runtime/'build-config.json')
    values,scores=blend(reg,raw,candidate,rx,s43)
    output.mkdir(parents=True,exist_ok=True)
    save_raw(output/'hb116-member-raw.npz',ids,raw)
    # Diagnostic kernel never writes submission.csv.
    name='diagnostic-hb116.csv' if stress else 'submission.csv'
    staged=output/(name+'.tmp'); staged.write_bytes(c.p.csv_bytes(ids,values)); os.replace(staged,output/name)
    missing={k:int((~np.isfinite(a).all(axis=1)).sum()) for k,a in raw.items()}
    receipt=dict(version=VERSION,status='COMPLETE_HB116',candidate=candidate,rx=rx,s43=s43,diagnostic=stress,
        studies=len(ids),ids=ids,device=device,gpu_names=gpu_names,fixed_seconds=setup,total_seconds=time.monotonic()-start,
        image=c.p.IMAGE,registration_sha256=REG_SHA,draft_sha256=DRAFT_SHA,config=config,
        config_sha256=sha(runtime/'build-config.json'),runtime_sha256=sha(Path(__file__)),
        native_runtime_sha256=sha(ROOT/'tools/hb116_native_runtime.py'),
        missing=missing,normal_visible_qualified=not any(missing.values()) and all(all(x==0 for x in p['exit_codes']) for p in phases.values())
            and all(x==0 for x in info.get('worker_exit_codes',{}).values()),
        phases=phases,s6_telemetry=info,
        raw_sha256=sha(output/'hb116-member-raw.npz'),prediction_sha256=sha(output/name),
        runtime_gate='PENDING_MATCHED_VISIBLE_AND_N32',parity_tolerance_fp16=.002)
    try:
        receipt['timing']=timings(work,phases,len(ids))
        receipt['projection']=projection(receipt)
    except Exception as exc:
        receipt['timing_error']=f'{type(exc).__name__}: {exc}'
    dump(output/'hb116-receipt.json',receipt)
    print('COMPLETE_HB116',json.dumps({k:receipt[k] for k in ('candidate','rx','studies','missing')}),flush=True)
    return receipt


def register(out=OUT):
    pin(REG/'REGISTRATION.md', {'sha256': REG_SHA})
    pin(REG/'consults/astra3-registration-draft.json', {'sha256': DRAFT_SHA})
    reg=read(REG/'consults/astra3-registration-draft.json')
    verify_roster(reg)
    entries,_=local_assets(reg)
    for e in entries:
        pin(e['resolved'],e)
    for e in reg['Q']['authority_manifests']:
        pin(ROOT/e['path'],e)
    pin(ROOT/reg['P']['Rplus2']['manifest']['path'],reg['P']['Rplus2']['manifest'])
    result=dict(version=VERSION,status='REGISTERED_CPU_ONLY',registration_sha256=sha(REG/'REGISTRATION.md'),
        draft_sha256=sha(REG/'consults/astra3-registration-draft.json'),constants=CONSTANTS,
        selected_q_members=reg['Q']['members'],files=[{k:v for k,v in e.items() if k in ('id','resolved','bytes','sha256')} for e in entries],
        runtime_cut_order=['S43','R2','S_TO_T_REQUIRES_NEW_PINNED_BUILD','NO_BUILD'])
    dump(out/'registration.json',result)
    return result


def asset_manifest(out):
    """Inventory reused bytes as well as the one new private upload."""
    reg=read(REG/'consults/astra3-registration-draft.json')
    reused=[]
    def add(slug,name,path,digest):
        pin(path,{'sha256':digest})
        reused.append(dict(dataset=slug,file=name,bytes=Path(path).stat().st_size,sha256=digest,
                           verified_local=str(Path(path).relative_to(ROOT))))
    c4=V4/'c4/dataset/exp169-c4-v4.zip.bin'
    add('taeyangg4/knee-exp169-c4-v4-assets',c4.name,c4,'df11aa92535c9487ce6852b3ac159a162a6fb8cea989ac5a625d35faff4dd634')
    with zipfile.ZipFile(c4) as z:
        for e in reg['Q']['members']:
            if e['family']!='C4': continue
            name=next(n for n in z.namelist() if Path(n).name==f"convnext_s42_f{e['fold']}.pt")
            with z.open(name) as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            need(digest==e['sha256'] and z.getinfo(name).file_size==e['bytes'],'C4 archive differs from sealed original')
            reused.append(dict(dataset='taeyangg4/knee-exp169-c4-v4-assets',file=c4.name+'!/'+name,
                               bytes=e['bytes'],sha256=digest,member=e['id']))
    entries,paths=local_assets(reg)
    for e in entries:
        if e['id'] in ('EXP143v2','EXP155'):
            add('taeyangg4/knee-exp155-own0949-oai-v1-2-assets','exp155-assets.zip.bin!/'+Path(e['resolved']).name,e['resolved'],e['sha256'])
    for name,digest in [('public-serving.py',c.p.PINS['public-serving.py']),('release-manifest.json',OWN_MANIFEST_SHA)]:
        add('taeyangg4/knee-exp155-own0949-oai-v1-2-assets','exp155-assets.zip.bin!/'+name,OWN_DIR/name,digest)
    arc=V4/'s43/dataset/exp169-s43-v4.zip.bin'
    add('taeyangg4/knee-exp169-s43-v4-assets',arc.name,arc,'887e294832943b8a0b5afa937459167d4425d761c05c1a154623784d60c81ef5')
    add('taeyangg4/knee-exp169-s43-v4-assets',arc.name+'!/'+S43_NAME,SCRATCH/'assets'/S43_NAME,S43_SHA)
    add(c.p.DATASETS[0],c.p.R_NAME,ROOT/'.local/rplus-v1/public-R-download.bin',c.p.R_SHA)
    bundle=ROOT/'storage/weights/external/heliosli_rsna-knee-compact-v1_v10-equivalent'
    c.s6.verify_bundle(bundle)
    for name,digest in dict(c.s6.HASHES,**{'manifest.json':c.s6.MANIFEST_SHA256,'README.md':c.s6.README_SHA256}).items():
        add(c.s6.DATASET_SLUG,name,bundle/name,digest)
    anatomy=ROOT/'storage/weights/external/heliosli_rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt'
    code=ROOT/'.local/exp189-cpu-v120/input/cloud_code'
    c.s6.verify_code(code,anatomy)
    add(c.v5q.CODE_SLUG,'anatomy.pt',anatomy,c.s6.ANATOMY_SHA256)
    tree=c.s6.code_tree_manifest(code)
    for name,digest in tree['sha256'].items():
        reused.append(dict(dataset=c.v5q.CODE_SLUG,file='mainline_code/'+name,sha256=digest,bytes=(code/name).stat().st_size))
    for name,digest in c.s6.WHEEL_HASHES.items():
        reused.append(dict(dataset=c.v5q.CODE_SLUG,file='dependencies/wheels/'+name,sha256=digest,
                           verification='frozen public dataset pin; verified again before target install'))
    for e in reg['P']['Rplus2']['members']+[reg['P']['Rplus2']['manifest']]:
        add('taeyangg4/knee-hb113-rplus-weights-v2',Path(e['path']).name,ROOT/e['path'],e['sha256'])
    result=dict(new_private_dataset=NEW_DATASET,new_files=read(out/'datasets/knee-hb116-vnext-weights-v1/manifest.json')['files'],
        reused_files=reused,own_archive_sha256='40b8d02847bf36938d82a79e88cb1dad4296db0cc95691b8cd0d2a2ce1a7e5fa',
        policy='Reuse exact existing bytes; original R50 and MIL-S need one new private dataset; no upload performed.')
    dump(out/'asset-manifest.json',result)
    return result


def source_payloads():
    from tools import exp169_release_v6 as v6
    files={}
    # Preserve the C85 flat import ABI under tools/. No helper byte is changed.
    for name,data in c.source_payloads().items():
        files['tools/'+name]=data
    for name,path in v6.source_files(ROOT).items():
        if name not in files:
            files[name]=path.read_bytes()
    for name in ('hb105_serving_v101.py', 'exp155_own0949_oai_release_v12.py', 'hb116_native_runtime.py'):
        files['tools/'+name]=(ROOT/'tools'/name).read_bytes()
    files['tools/hb116_vnext_release.py']=Path(__file__).read_bytes()
    for name in ('tools/__init__.py','training/__init__.py','training/arms/__init__.py','training/src/__init__.py'):
        files[name]=b''
    files['sealed-registration.json']=(REG/'consults/astra3-registration-draft.json').read_bytes()
    pin(ROOT/Q_AUTH, {'sha256':Q_AUTH_SHA})
    files['q-authority.json']=(ROOT/Q_AUTH).read_bytes()
    return files


def package(files):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,10,13,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    return buf.getvalue(),{k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())}


def notebook(blob,pins,config):
    src='\n'.join([
        'import os, sys, io, base64, hashlib, zipfile', 'from pathlib import Path',
        "os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')",
        f'blob=base64.b64decode({base64.b64encode(blob).decode()!r})',
        f'assert hashlib.sha256(blob).hexdigest()=={hashlib.sha256(blob).hexdigest()!r}',f'pins={pins!r}',
        "runtime=Path('/kaggle/working/hb116-runtime')",'assert not runtime.exists()',
        'with zipfile.ZipFile(io.BytesIO(blob)) as z:',
        '    assert len(z.namelist())==len(pins) and set(z.namelist())==set(pins)',
        '    for name in z.namelist():',
        "        assert not name.startswith(('/', '\\\\')) and '..' not in Path(name).parts and ':' not in name and '\\\\' not in name",
        '        assert hashlib.sha256(z.read(name)).hexdigest()==pins[name]',
        '    z.extractall(runtime)', 'sys.path.insert(0,str(runtime))',
        'from tools import hb116_vnext_release as release',
        f"release.serve(runtime,Path('/kaggle/working'),candidate={config['candidate']!r},rx={config['rx']!r},s43={config['s43']!r},stress={config['stress']!r})",''])
    compile(src,'<HB116>','exec')
    return dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':dict(display_name='Python 3',language='python',name='python3')},cells=[
        dict(cell_type='markdown',metadata={},source=[f'# HB116 vNext v{VERSION}\nSealed A/B; private, offline. CPU prepared; target parity/runtime and independent review required.\nCredits: nartaa public R, heliosli S6. Original source and provenance retained.\n'+('DIAGNOSTIC ONLY; NEVER SUBMIT.' if config['stress'] else '')]),
        dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=src.splitlines(True))])


def prepare(out=OUT):
    reg=read(REG/'consults/astra3-registration-draft.json')
    registration=register(out)
    dataset=out/'datasets/knee-hb116-vnext-weights-v1'
    dataset.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for e in reg['Q']['members']:
        if e['family']=='C4':
            continue
        dst=dataset/(e['id']+'.pt')
        if not dst.exists():
            shutil.copyfile(ROOT/e['path'],dst)
        pin(dst,e)
        manifest.append(dict(file=dst.name,bytes=dst.stat().st_size,sha256=sha(dst),source=e['path']))
    dump(dataset/'manifest.json',dict(version=VERSION,files=manifest))
    dump(dataset/'dataset-metadata.json',dict(id=NEW_DATASET,title=NEW_DATASET.split('/')[1],isPrivate=True,licenses=[{'name':'other'}]))
    asset_manifest(out)
    files=source_payloads()
    kernels=[]
    for rx in ('R','R2'):
        for s43 in (True,False):
            for candidate,stress in [('A',False),('B',False),('A',True)]:
                config=dict(version=VERSION,candidate=candidate,constants=CONSTANTS[candidate],rx=rx,s43=s43,stress=stress)
                folder=out/'kernels'/f"{rx.lower()}{'' if s43 else '-no-s43'}"/('stress-n32' if stress else candidate.lower())
                folder.mkdir(parents=True,exist_ok=True)
                payload=dict(files,**{'build-config.json':(json.dumps(config,sort_keys=True)+'\n').encode()})
                blob,pins=package(payload)
                dump(folder/'release.ipynb',notebook(blob,pins,config))
                slug='taeyangg4/knee-hb116-vnext-'+('stress-n32' if stress else candidate.lower())
                datasets=[NEW_DATASET,'taeyangg4/knee-exp155-own0949-oai-v1-2-assets','taeyangg4/knee-exp169-c4-v4-assets',c.s6.DATASET_SLUG,c.v5q.CODE_SLUG]
                datasets += ['taeyangg4/knee-exp169-s43-v4-assets'] if s43 else []
                datasets += [c.p.DATASETS[0] if rx=='R' else 'taeyangg4/knee-hb113-rplus-weights-v2']
                meta=dict(c.metadata(),id=slug,title=slug.split('/')[1],dataset_sources=datasets)
                dump(folder/'kernel-metadata.json',meta); dump(folder/'build-config.json',config); dump(folder/'source-pins.json',pins)
                kernels.append(str(folder.relative_to(out)))
    dump(out/'build.json',dict(version=VERSION,kernels=kernels,dataset=NEW_DATASET,status='CPU_PREPARED_TARGET_GATES_PENDING'))
    return dict(status='PREPARED',kernels=len(kernels),upload_checkpoints=len(manifest),upload_bytes=sum(e['bytes'] for e in manifest))


def validate(out=OUT):
    need(read(out/'registration.json')['draft_sha256']==sha(REG/'consults/astra3-registration-draft.json'),'registration drift')
    for e in read(out/'datasets/knee-hb116-vnext-weights-v1/manifest.json')['files']:
        pin(out/'datasets/knee-hb116-vnext-weights-v1'/e['file'],e)
    for name in read(out/'build.json')['kernels']:
        folder=out/name; meta=read(folder/'kernel-metadata.json'); config=read(folder/'build-config.json')
        need(meta['is_private'] and not meta['enable_internet'] and meta['docker_image']==c.p.IMAGE and meta['machine_shape']==c.p.MACHINE,'target metadata')
        need(config['constants']==CONSTANTS[config['candidate']],'O constants')
        src=''.join(read(folder/'release.ipynb')['cells'][1]['source'])
        compile(src,'<release>','exec')
        import ast
        tree=ast.parse(src)
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='blob' for t in n.targets))
        blob=base64.b64decode(ast.literal_eval(node.value.args[0]))
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            pins=read(folder/'source-pins.json')
            need(set(z.namelist())==set(pins),'package inventory')
            need(all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in pins.items()),'package pin')
            need(z.read('tools/hb116_vnext_release.py')==Path(__file__).read_bytes(),'stale runtime package')
            need(z.read('tools/hb116_native_runtime.py')==(ROOT/'tools/hb116_native_runtime.py').read_bytes(),'stale native package')
            need(json.loads(z.read('build-config.json'))==config,'embedded config differs')
            call=tree.body[-1].value
            need({k.arg:ast.literal_eval(k.value) for k in call.keywords}==
                 {k:config[k] for k in ('candidate','rx','s43','stress')},'notebook invocation differs')
    return dict(status='VALIDATED',kernels=len(read(out/'build.json')['kernels']))


def extract_visible():
    comp=SCRATCH/'competition'; comp.mkdir(parents=True,exist_ok=True)
    for name in ('test.csv','test_series.csv','sample_submission.csv'):
        shutil.copyfile(ROOT/'storage/competition'/name,comp/name)
    with zipfile.ZipFile(ROOT/'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip') as z:
        for name in z.namelist():
            if name.startswith('test_series/') and name.endswith('.dcm'):
                need('..' not in Path(name).parts,'unsafe test member')
                dest=comp/name
                if not dest.exists():
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(name) as src,dest.open('wb') as dst:
                        shutil.copyfileobj(src,dst)
    return comp


def reset_smoke(work):
    work=Path(work).resolve()
    need(work.parent==SCRATCH.resolve() and work.name in
         {'cpu-all','cpu-NATIVE','cpu-R50','cpu-C4','cpu-S','cpu-OWN','cpu-Rx','cpu-R','cpu-S6'},'unsafe smoke output')
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)


def smoke(family=None):
    import numpy as np
    reg=read(REG/'consults/astra3-registration-draft.json')
    entries,paths=local_assets(reg)
    if family=='S6':
        return smoke_s6()
    if family=='R':
        entries=[dict(id='R',resolved=str(ROOT/'.local/rplus-v1/public-R-download.bin'),sha256=c.p.R_SHA)]
    selected=[family] if family else ['NATIVE']
    if selected==['NATIVE']:
        entries.append(dict(id='R',resolved=str(ROOT/'.local/rplus-v1/public-R-download.bin'),sha256=c.p.R_SHA))
    if family=='R': selected=['Rx']
    entries=[e for e in entries if 'NATIVE' in selected or e.get('family') in selected or (e['id'] in ('EXP143v2','EXP155','S43') and 'OWN' in selected) or ((e['id']=='R' or e['id'].startswith('R2-')) and 'Rx' in selected)]
    for e in entries: pin(e['resolved'],e)
    work=SCRATCH/('cpu-'+(family or 'all'))
    reset_smoke(work)
    source_hashes={name:sha(ROOT/'tools'/name) for name in ('hb116_vnext_release.py','hb116_native_runtime.py')}
    ids,raw,phases=run_phases(reg,entries,paths,extract_visible(),work,selected=selected)
    need(all(sha(ROOT/'tools'/name)==digest for name,digest in source_hashes.items()),'source changed during CPU smoke')
    save_raw(work/'raw.npz',ids,raw)
    passed=all(np.isfinite(a).all() for a in raw.values()) and all(all(v==0 for v in e['exit_codes']) for e in phases.values())
    receipt=dict(status='PASS' if passed else 'FAIL',device='cpu',precision='fp32',ids=ids,members=list(raw),
                 phases=phases,source_hashes=source_hashes,raw_sha256=sha(work/'raw.npz'),timing=timings(work,phases,len(ids)))
    dump(work/'receipt.json',receipt)
    need(passed,'real CPU smoke failed; inspect child log')
    return receipt


def smoke_s6():
    import numpy as np
    work=SCRATCH/'cpu-S6'
    reset_smoke(work)
    comp=extract_visible()
    ids,_=c.v5q._read_csv(comp/'sample_submission.csv')
    pdir=work/'public'; dump(pdir/'ids.json',ids)
    job=dict(ids=ids,competition=str(comp),
        bundle=str(ROOT/'storage/weights/external/heliosli_rsna-knee-compact-v1_v10-equivalent'),
        code=str(ROOT/'.local/exp189-cpu-v120/input/cloud_code'),
        anatomy=str(ROOT/'storage/weights/external/heliosli_rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt'),
        output=str(pdir/'p956-run/s6'),source=str(c.s6.SOURCE),device='cpu')
    phases={}
    values,info=run_s6(job,work,phases)
    save_raw(work/'raw.npz',ids,{'S6':values})
    passed=phases['S6']['exit_codes']==[0] and np.isfinite(values).all()
    receipt=dict(status='PASS' if passed else 'FAIL',device='cpu',ids=ids,members=['S6'],phases=phases,
                 raw_sha256=sha(work/'raw.npz'),s6_telemetry=info)
    if passed: receipt['timing']=timings(work,phases,len(ids))
    dump(work/'receipt.json',receipt)
    need(passed,'real S6 CPU smoke failed; inspect child log')
    return receipt


def replay():
    """Independent pandas formula, including tie ranks; no inference claim."""
    import numpy as np
    import pandas as pd
    reg=read(REG/'consults/astra3-registration-draft.json')
    rng=np.random.default_rng(116)
    results={}
    r=lambda a:pd.DataFrame(a).rank(method='average',pct=True).to_numpy()
    for rx in ('R','R2'):
        raw={k:rng.integers(1,20,(37,12)).astype(np.float32)/20 for k in roster(reg,rx)}
        fam=[]
        for f in ('R50','C4','S'):
            es=[e for e in reg['Q']['members'] if e['family']==f]
            if f=='R50':
                a=np.mean([np.mean([raw[e['id']] for e in es if e['fold']==fold],axis=0,dtype=np.float32) for fold in range(5)],axis=0,dtype=np.float32)
            else:
                a=np.mean([raw[e['id']] for e in es],axis=0,dtype=np.float32)
            fam.append(r(a))
        q=np.mean(fam,axis=0)
        anchor=(raw['EXP143v2'].astype('float64')+raw['S43'])/2
        public=raw['R'] if rx=='R' else np.mean([raw['R2-s43'],raw['R2-s44']],axis=0,dtype=np.float32)
        p=(r(public)+r(raw['S6']))/2
        for candidate,weights in CONSTANTS.items():
            a,b,d=weights
            own=(a*r(anchor)+b*r(raw['EXP155'])+d*r(q))/(a+b+d)
            ref=(.7*r(p)+.3*r(own)).astype(np.float32)
            got,_=blend(reg,raw,candidate,rx)
            need(np.array_equal(got,ref),'independent arithmetic replay')
            results[candidate+'-'+rx]=float(np.max(np.abs(got-ref)))
    dump(SCRATCH/'arithmetic-replay.json',dict(status='PASS',max_abs=results,reference='independent pandas average-tie ranks'))
    return dict(status='PASS',max_abs=results)


def legacy_raw_journals(folder, ids, data, selected, registry, version=VERSION):
    """Authenticate early CPU NPZs without target metadata; never alter them."""
    import numpy as np
    for family in selected:
        need(family in ('R50','C4','S','OWN','Rx'), 'unsupported legacy schema')
        names=[k for k in data.files if (k.startswith(family+'-') if family in ('R50','C4','S') else
               k in ('EXP143v2','EXP155','S43') if family=='OWN' else k=='R' or k.startswith('R2-'))]
        need(bool(names) and all(k in registry for k in names), 'legacy members not registered')
        need(version in ('1.0.0',VERSION),'unsupported CPU evidence version')
        identity=dict(version=version,members={k:registry[k] for k in names})
        values,failures=c.read_journal(folder/family/'worker0',ids,names,identity)
        need(all(not failures[k] and np.array_equal(data[k],values[k].astype(np.float32)) for k in names),
             'legacy raw differs from authenticated journal')


def evidence(out=OUT):
    """Authenticate native journals; compare with frozen v1.0.1 CPU and T4 raw."""
    import numpy as np
    reg=read(REG/'consults/astra3-registration-draft.json')
    registered=read(out/'registration.json')
    need(registered['registration_sha256']==REG_SHA and registered['draft_sha256']==DRAFT_SHA,'evidence registration')
    registry={e['id']:e['sha256'] for e in registered['files']}
    registry['R']=c.p.R_SHA
    folder=SCRATCH/'cpu-NATIVE'
    rec=read(folder/'receipt.json')
    need(rec['status']=='PASS' and rec['phases']['NATIVE']['exit_codes']==[0],'native CPU smoke failed')
    need(all(sha(ROOT/'tools'/name)==digest for name,digest in rec['source_hashes'].items()),'CPU smoke source drift')
    need(set(rec['source_hashes'])=={'hb116_vnext_release.py','hb116_native_runtime.py'},'CPU source binding missing')
    pin(folder/'raw.npz',{'sha256':rec['raw_sha256']})
    with np.load(folder/'raw.npz',allow_pickle=False) as data:
        ids=data['ids'].tolist()
        need(len(ids)==3 and data['targets'].tolist()==LABELS,'CPU visible cohort/schema')
        raw={k:data[k].copy() for k in data.files if k not in ('ids','targets')}
    identity=dict(version=VERSION,members={k:registry[k] for k in raw})
    journal,failures=c.read_journal(folder/'NATIVE/worker0',ids,list(raw),identity)
    need(all(not failures[k] and np.array_equal(raw[k],journal[k].astype(np.float32)) for k in raw),'native CPU journal mismatch')
    baseline=ROOT/'artifacts/experiments/HB116_VNEXT/v1.0.1/release'
    old_path=baseline/'validation/cpu-member-raw.npz'
    pin(old_path,{'sha256':'1e6379963659e2b2e7ee7ac7d539e34f3fd31c746ba9dd98d27ea6add35708dd'})
    old_rec=read(baseline/'validation/cpu-evidence.json')
    need(old_rec['status']=='PASS' and old_rec['raw_sha256']==sha(old_path),'baseline CPU evidence')
    # Authenticate byte-identical unchanged S6 dependencies before reusing its
    # previous three-study CPU inference. Never describe that as a fresh rerun.
    pins=read(baseline/'kernels/r2/a/source-pins.json')
    shared=c.source_payloads()
    need(all(pins['tools/'+name]==hashlib.sha256(value).hexdigest() for name,value in shared.items()),'S6 frozen dependency drift')
    with np.load(old_path,allow_pickle=False) as data:
        need(data['ids'].tolist()==ids and data['targets'].tolist()==LABELS,'baseline cohort')
        old={k:data[k].copy() for k in data.files if k not in ('ids','targets')}
    raw['S6']=old['S6'].copy()
    need(set(raw)==set(old),'CPU full member roster')
    cpu_delta={k:float(np.max(np.abs(raw[k]-old[k]))) for k in raw}
    need(all(np.isfinite(v) and v<=.002 for v in cpu_delta.values()),'native/frozen CPU parity exceeds 2e-3')
    blends={}
    for rx in ('R','R2'):
        for s43 in (True,False):
            subset={k:raw[k] for k in roster(reg,rx,s43)}
            qualify(reg,subset,rx,s43)
            for candidate in CONSTANTS:
                got,_=blend(reg,subset,candidate,rx,s43)
                expected,_=blend(reg,old,candidate,rx,s43)
                delta=float(np.max(np.abs(got-expected)))
                need(delta<=.002,'final rank parity exceeds 2e-3')
                blends[f'{candidate}-{rx}-s43-{s43}']=delta
    target_delta={}
    for name in ('a','b'):
        prior=ROOT/f'.local/scratch/slot-b/vnext-{name}-visible'
        target_rec=read(prior/'hb116-receipt.json')
        target_path=prior/'hb116-member-raw.npz'
        pin(target_path,{'sha256':target_rec['raw_sha256']})
        with np.load(target_path,allow_pickle=False) as data:
            need(data['ids'].tolist()==ids,'prior T4 cohort')
            delta={k:float(np.max(np.abs(raw[k]-data[k]))) for k in roster(reg,'R2',True)}
        target_delta[name]=dict(max_abs=max(delta.values()),per_member=delta,
            within_2e_3=all(np.isfinite(v) and v<=.002 for v in delta.values()),
            receipt_sha256=sha(prior/'hb116-receipt.json'))
    prep=read(out/'validation/input-equivalence.json')
    need(prep['status']=='PASS' and prep['runtime_sha256']==sha(ROOT/'tools/hb116_native_runtime.py'),'input proof stale')
    target=out/'validation'; target.mkdir(parents=True,exist_ok=True)
    save_raw(target/'cpu-member-raw.npz',ids,raw)
    result=dict(status='PASS',device='cpu',studies=3,members=len(raw),fresh_native_members=len(raw)-1,
        finite_values=sum(a.size for a in raw.values()),inference_version=VERSION,
        raw_sha256=sha(target/'cpu-member-raw.npz'),source_hashes=rec['source_hashes'],
        native_receipt_sha256=sha(folder/'receipt.json'),baseline_sha256=sha(old_path),
        s6='Reused authenticated v1.0.1 CPU outputs; unchanged source dependencies, no S6 rerun.',
        cpu_member_max_abs=cpu_delta,final_eight_variant_max_abs=blends,prior_t4_comparison=target_delta,
        input_equivalence_sha256=sha(out/'validation/input-equivalence.json'),
        timing=rec['timing'],target_runtime='UNMEASURED; fresh 2xT4 visible/N32 memory and runtime gates required')
    dump(target/'cpu-evidence.json',result)
    return dict(status='PASS',members=len(raw),studies=3,fresh_native_members=31,
                max_abs_cpu=max(cpu_delta.values()),max_abs_final=max(blends.values()),
                prior_t4={k:v['max_abs'] for k,v in target_delta.items()})


def target_receipt(folder, diagnostic, reg):
    import numpy as np
    receipt=read(folder/'hb116-receipt.json')
    need(receipt['status']=='COMPLETE_HB116' and receipt['version']==VERSION and receipt['diagnostic']==diagnostic,'receipt identity')
    need(receipt['registration_sha256']==REG_SHA and receipt['draft_sha256']==DRAFT_SHA and
         receipt['runtime_sha256']==sha(Path(__file__)) and
         receipt.get('native_runtime_sha256')==sha(ROOT/'tools/hb116_native_runtime.py'),'receipt source drift')
    need(receipt['device']=='2xT4' and len(receipt['gpu_names'])==2 and all('T4' in x for x in receipt['gpu_names'])
         and receipt['image']==c.p.IMAGE,'target profile')
    need(receipt['studies']==(32 if diagnostic else 3),'full visible/N32 cohort required')
    need(receipt['normal_visible_qualified'] and not any(receipt['missing'].values()),'missing member/family/branch: qualification closed')
    phases=receipt['phases']
    exits=receipt['s6_telemetry']['worker_exit_codes']
    need(set(phases)=={'NATIVE','S6'} and
         all(p['exit_codes']==([0] if name=='S6' else [0,0]) for name,p in phases.items()) and
         len(exits)==2 and all(x==0 for x in exits.values()),'native child failure or incomplete roster')
    cfg=receipt['config']
    need(cfg==dict(version=VERSION,candidate=receipt['candidate'],constants=CONSTANTS[receipt['candidate']],
                  rx=receipt['rx'],s43=receipt['s43'],stress=diagnostic),'receipt config')
    need(receipt['config_sha256']==hashlib.sha256((json.dumps(cfg,sort_keys=True)+'\n').encode()).hexdigest(),'config digest')
    pin(folder/'hb116-member-raw.npz',{'sha256':receipt['raw_sha256']})
    prediction=folder/('diagnostic-hb116.csv' if diagnostic else 'submission.csv')
    pin(prediction,{'sha256':receipt['prediction_sha256']})
    if diagnostic: need(not (folder/'submission.csv').exists(),'stress submission forbidden')
    with np.load(folder/'hb116-member-raw.npz',allow_pickle=False) as data:
        ids=data['ids'].tolist()
        need(ids==receipt['ids'] and len(ids)==len(set(ids))==receipt['studies'] and data['targets'].tolist()==LABELS,'raw cohort/schema')
        raw={k:data[k].copy() for k in data.files if k not in ('ids','targets')}
    qualify(reg,raw,receipt['rx'],receipt['s43'])
    value,_=blend(reg,raw,receipt['candidate'],receipt['rx'],receipt['s43'])
    need(prediction.read_bytes()==c.p.csv_bytes(ids,value),'saved arithmetic replay mismatch')
    proj=projection(receipt)
    memory=receipt['timing']['families']['NATIVE'].get('memory',[])
    need(len(memory)==2 and all(0<m['peak_allocated_bytes']<=m['peak_reserved_bytes']<.95*m['total_device_bytes']
                               for m in memory),'native T4 memory headroom not demonstrated')
    need(receipt['projection']==proj,'projection replay mismatch')
    return receipt,raw


def admission_gates(visible, stress, local_raw):
    import numpy as np
    reg=read(REG/'consults/astra3-registration-draft.json')
    v,raw=target_receipt(Path(visible),False,reg)
    s,_=target_receipt(Path(stress),True,reg)
    need(all(v[k]==s[k] for k in ('rx','s43','runtime_sha256','native_runtime_sha256','image','gpu_names')),'unmatched visible/N32 build')
    # A/B share inference exactly; stress A qualifies B's identical member path.
    need(max(v['projection']['projected_seconds'],s['projection']['projected_seconds'])<=28800,'runtime limit exceeded')
    with np.load(local_raw,allow_pickle=False) as local:
        need(local['ids'].tolist()==v['ids'] and local['targets'].tolist()==LABELS,'local parity cohort/schema')
        required=roster(reg,v['rx'],v['s43'])
        need(set(required)<=set(local.files),'CPU parity members missing')
        delta={k:float(np.max(np.abs(local[k]-raw[k]))) for k in required}
        need(all(np.isfinite(x) and x<=.002 for x in delta.values()),'CPU/T4 raw parity exceeds fp16 2e-3')
    return dict(status='TECHNICAL_GATES_PASS',parity_max_abs=delta,
                visible_projection=v['projection'],stress_projection=s['projection'])


def admit(out=OUT, visible=None, stress=None, local_raw=None):
    validate(out)
    need(all(x is None for x in (visible,stress,local_raw)) or all(x is not None for x in (visible,stress,local_raw)),
         'supply visible, stress and local-raw together')
    gates=admission_gates(visible,stress,local_raw) if visible else {'status':'TARGET_GATES_PENDING'}
    return dict(status='HOLD',cpu_build='VALID',technical=gates,required=['matched full 2xT4 visible and N32 receipts',
        'per-member raw CPU/T4 max_abs <= 0.002','rule b score and qualified R2 visible decision',
        'C85 veto checks by B','independent pre-push review'],runtime_limit_seconds=28800,studies=1300,
        runtime_cuts=['drop S43','replace R2 with R','S-to-T on demand','NO_BUILD'],
        s_to_t=dict(prebuilt=False,status='ON_DEMAND_ONLY',
            trigger='Measured runtime remains above 28800 seconds after cutting S43 and R2; requires new pinned build and matched visible/N32 runs'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register','prepare','validate','replay','admit','evidence','smoke','worker','s6-worker'])
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--job',type=Path)
    parser.add_argument('--family',choices=['NATIVE','R50','C4','S','OWN','Rx','R','S6'])
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda:0'])
    parser.add_argument('--visible',type=Path)
    parser.add_argument('--stress',type=Path)
    parser.add_argument('--local-raw',type=Path)
    args=parser.parse_args()
    configure()
    if args.command=='worker': return worker(args.job,args.device)
    if args.command=='s6-worker': return s6_worker(args.job)
    if args.command=='smoke': result=smoke(args.family)
    elif args.command=='replay': result=replay()
    elif args.command=='admit': result=admit(args.out,args.visible,args.stress,args.local_raw)
    else: result=globals()[args.command](args.out)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
