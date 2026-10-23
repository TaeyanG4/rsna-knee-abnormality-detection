"""H-B-150 prebuild lifecycle v2.0.0, 2026-10-14.

Changelog v2: separate launch-v115-bound stage; scientific behavior unchanged.

Changelog: four immutable native320 PL configurations; metadata-only preparation,
smoke-bound independent review, fail-closed paired dispatch. Originals unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.hb150_io_v1 import atomic_json

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.local/scratch/astra-b3/H-B-147/H-B-150-v2'
PL_WORK = ROOT / '.local/scratch/astra-b4/H-B-146/launch-v115'
CORPUS = ROOT / '.local/scratch/astra-b3/H-B-147/native448'
GPU = 'NVIDIA GeForce RTX 4070 Ti SUPER'
VERSION = '2.0.0'
SOURCE_NAMES = [
    'tools/hb150_build_v2.py', 'tools/hb150_fit_v2.py', 'tools/hb150_readout_v2.py',
    'tools/hb150_io_v1.py', 'tools/hb150_watchdog_v2.py', 'tools/job_runtime.py', 'tools/gpu_job.py',
    'training/arms/native_resolution_v2.py', 'training/arms/mil_screen_hooks_v3.py',
    'training/arms/mil_screen_hooks_v1.py', 'training/arms/mil_screen_v1.py',
    'training/arms/mil.py', 'training/arms/mil_small.py', 'training/arms/lifecycle.py',
    'training/src/raptor_common.py', 'training/src/raptor_runtime.py',
    'training/src/raptor_exp185.py', 'training/src/raptor_corpus.py',
    'tools/hb147_native_cache_v1.py', 'tools/hb147_progress_v1.py',
    'tools/hb146_pl_labels_v111.py', 'tools/hb146_readout_v111.py',
    'tools/multi_lens_readout_l4_v2.py', 'tools/multi_lens_readout.py',
]


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def immutable(path, value):
    if Path(path).exists():
        need(read(path) == value, 'immutable artifact differs; version/review required: ' + str(path))
    else:
        atomic_json(path, value)


def config_path(arm, seed):
    need(arm in ('global', 'fill') and seed in (42, 43), 'unregistered member')
    return WORK / 'configs' / f'{arm}-s{seed}.json'


def register(pl_work=PL_WORK):
    pl_work = Path(pl_work).resolve()
    s = read(pl_work / 'registration.json')  # metadata only; never traverse silence pins
    need(s['version'].startswith('1.1.') and s['status'] == 'EXPOSED_EXPLORATORY'
         and s['exposed_exploratory'] is True, 'H146 v1.1.x EXPOSED registration required')
    need(len(s['train']) == 3190 and len(s['val']) == 977 and len(s['gold']) == 58,
         'H146 split changed')
    need(not (set(s['train']) & set(s['val'])), 'split overlap')
    manifest = read(CORPUS / 'manifest.json')
    need(manifest['finished'] is True and manifest['studies'] == 4410
         and manifest['shape_total'] == [4410,64,448,448]
         and manifest['contract'] == 'hb147_raptor64_448_c140_s02_v1', 'native cache incomplete')
    need(sha(CORPUS/'ids.npy') == manifest['ids_sha256']
         and sha(CORPUS/'masks.npy') == manifest['masks_sha256'], 'native roster/masks changed')
    need(manifest['source_manifest_sha256']=='f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e'
         and manifest['builder_sha256']==sha(ROOT/'tools/hb147_native_cache_v1.py')
         and manifest['raw_builder_sha256']==sha(ROOT/'training/src/raptor_corpus.py'), 'native builder lineage changed')
    inputs = {p: sha(ROOT/p) for p in SOURCE_NAMES}
    inputs.update({rel(p):sha(p) for p in (pl_work/'registration.json', CORPUS/'manifest.json',
                                         CORPUS/'ids.npy', CORPUS/'masks.npy')})
    for key,digest in (('initial_state','initial_file_sha256'),('weights','weights_sha256')):
        p=s['recipe']['model'][key]; d=s['recipe']['model'][digest]
        need(sha(ROOT/p)==d,'model input changed')
        inputs[p]=d
    result=[]
    for arm in ('global','fill'):
        for seed in (42,43):
            c = dict(schema='hb150-config/v1', version=VERSION, arm=arm, seed=seed,
                     registration=rel(pl_work/'registration.json'), pl_work=rel(pl_work),
                     corpus=rel(CORPUS), recipe=copy.deepcopy(s['recipe']), train=s['train'],
                     val=s['val'], gold=s['gold'], weights=s['weights'], inputs=inputs,
                     label_path=rel(pl_work/'screen_f0'/f'{arm}.csv'), exposure='EXPOSED',
                     clean_oof=False, epochs=20, batch=8, microbatch=2, resolution=320,
                     train_k=12, validation_k=[24,36], validation_flip=True, workers=0,
                     optimizer_steps=7960, selection='fixed_epoch20', gpu_name=GPU,
                     role='f0', matched_224='H-B-146 fits themselves; no rerun',
                     scientific_scope='EXPOSED exploratory combined native-cache detail plus resolution recipe',
                     gate=dict(macro=.001, weak4=.001, same_rulers=3, all_cell_floor=-.003),
                     review_required=True)
            immutable(config_path(arm,seed),c)
            result.append(f'{arm}-s{seed}')
    return dict(status='REGISTERED',members=result,train=3190,val=977,exposure='EXPOSED')


def validate(path):
    c=read(path)
    need(c['schema']=='hb150-config/v1' and c['arm'] in ('global','fill') and c['seed'] in (42,43), 'config identity')
    for p,d in c['inputs'].items():
        need(sha(ROOT/p)==d, 'bound input changed: '+p)
    s=read(ROOT/c['registration'])
    for key in ('train','val','gold','weights','recipe'):
        need(c[key]==s[key], 'H146 scientific recipe changed: '+key)
    need((c['epochs'],c['batch'],c['microbatch'],c['resolution'],c['train_k'],c['optimizer_steps'])
         ==(20,8,2,320,12,7960) and c['workers']==0 and c['selection']=='fixed_epoch20', 'recipe differs')
    need(c['exposure']=='EXPOSED' and c['clean_oof'] is False and c['role']=='f0'
         and c['review_required'] is True and c['gpu_name']==GPU, 'exposure/admission differs')
    return c


def prepare(arm):
    c=validate(config_path(arm,42)); validate(config_path(arm,43))
    w=ROOT/c['pl_work']
    files=[w/'prepare.json',w/'screen_f0/provenance.json',w/'screen_f0'/f'{arm}.csv']
    missing=[rel(p) for p in files if not p.is_file()]
    if missing:
        return dict(status='HOLD',reason='H146 scientific PL preparation not yet complete',missing=missing)
    p,r=read(files[1]),read(files[0]); t=p['teacher']
    need(p['version'].startswith('1.1.') and p['mode']=='screen_f0' and p['alpha']==.5
         and p['engineering_only'] is False and p['weights']==c['weights'], 'PL provenance differs')
    for v in (p,t,r):
        need(v.get('status')=='EXPOSED_EXPLORATORY' and v.get('exposed_exploratory') is True,
             'EXPOSED provenance required')
    need(t['model_selection']=='f0_only' and t['output_kind']=='pre_stacker'
         and t['complete'] is True and t['failures']==0 and t.get('clean_oof') is False,
         'complete EXPOSED role f0 teacher required')
    need(r['engineering_only'] is False and r['epochs']==20 and r['batch']==8
         and r['seeds']==[42,43] and r['weights']==c['weights'], 'H146 prepare recipe differs')
    need(p['arm_hashes'][arm]==sha(files[2]), 'PL label hash differs')
    # Check only allowed required pins, never follow protected relabel/report paths.
    for path in (w/'registration.json', files[1], files[2]):
        pinned = {str((ROOT/Path(k)).resolve()).lower():v for k,v in r['pins'].items()}
        need(pinned.get(str(path.resolve()).lower())==sha(path), 'H146 prepare does not authenticate '+rel(path))
    value=dict(schema='hb150-prepared/v1',arm=arm,exposure='EXPOSED',clean_oof=False,
               configs={str(s):sha(config_path(arm,s)) for s in (42,43)},
               pins={rel(p):sha(p) for p in files}, registration_sha256=sha(w/'registration.json'))
    immutable(WORK/f'prepared-{arm}.json',value)
    return dict(status='PREPARED',arm=arm,sha256=sha(WORK/f'prepared-{arm}.json'))


def review_bindings(arm,baseline_attestation):
    prep=prepare(arm)
    need(prep['status']=='PREPARED','scientific preparation HOLD')
    c=read(config_path(arm,42)); plwork=ROOT/c['pl_work']
    # Fail before touching any upstream result when future review inputs are absent.
    need(baseline_attestation is not None and Path(baseline_attestation).is_file(),
         'baseline attestation HOLD')
    for seed in (42,43):
        need((WORK/'cuda'/f'{arm}-s{seed}'/'summary.json').is_file(), 'CUDA smoke HOLD')
    paths=[WORK/f'prepared-{arm}.json',plwork/'readout.json',Path(baseline_attestation)]
    for seed in (42,43):
        paths += [config_path(arm,seed),WORK/'cpu'/f'{arm}-s{seed}'/'summary.json',
                  WORK/'cuda'/f'{arm}-s{seed}'/'summary.json']
    return {rel(p):sha(p) for p in paths}


def require_fit(path,review_path):
    c=validate(path)
    need(review_path is not None and Path(review_path).is_file(), 'independent review artifact required')
    review=read(review_path)
    need(review.get('schema')=='hb150-independent-review/v1' and review.get('verdict')=='PASS'
         and review.get('independent') is True and bool(review.get('reviewer'))
         and review.get('exposure')=='EXPOSED' and review.get('clean_oof') is False,
         'independent PASS review required')
    bindings=review_bindings(c['arm'],ROOT/review['baseline_attestation'])
    need(review.get('bindings')==bindings, 'review must bind exact CPU/CUDA receipts and prepared inputs')
    w=ROOT/c['pl_work']; decision=read(w/'readout.json')
    need(decision['version'].startswith('1.1.') and decision['engineering_only'] is False
         and decision['status']=='EXPOSED_EXPLORATORY'
         and decision['registered_go']['GO'] is True
         and decision['registered_go']['selected_arm']==c['arm'], 'H146 GO selected winner required')
    # Authenticates H146 final checkpoints/predictions and attested actual GPU, not old controls.
    from tools.hb150_readout_v2 import validate_baselines
    validate_baselines(c,ROOT/review['baseline_attestation'])
    for seed in (42,43):
        cfg=config_path(c['arm'],seed)
        cpu=read(WORK/'cpu'/f"{c['arm']}-s{seed}"/'summary.json')
        cuda=read(WORK/'cuda'/f"{c['arm']}-s{seed}"/'summary.json')
        for summary in (cpu,cuda):
            need(summary['config_sha256']==sha(cfg) and summary['engineering_only'] is True
                 and summary['status']=='PASS', 'smoke receipt mismatch')
        need(cpu['cpu_cuda_initialized'] is False, 'CPU smoke initialized CUDA')
        need(cuda['device']=='cuda' and cuda['gpu_name']==GPU and cuda['full_shape'] is True
             and 0<cuda['peak_reserved_bytes']<=10*(1<<30), 'full-shape 4070 <=10GiB smoke required')
        need(cuda['optimizer_steps']==2 and cuda['batch']==8 and cuda['val_batch']==4
             and cuda['validation_shapes']==['k24','k36','k36flip'], 'full train/validation/flip shape required')
        need(all(cuda['healthy_rates'].get(k,0)>0 for k in ('train_step','validation')), 'healthy rates absent')
    return c


def admit(arm,review=None):
    try:
        for seed in (42,43): require_fit(config_path(arm,seed),review)
        return dict(status='ADMITTED',arm=arm)
    except (OSError,KeyError,ValueError) as exc:
        return dict(status='HOLD',arm=arm,reason=str(exc))


def pair(arm,review,attempt):
    need(attempt.replace('-','').replace('_','').isalnum(),'named attempt must be alphanumeric')
    need(admit(arm,review)['status']=='ADMITTED','admission HOLD')
    for seed in (42,43):
        name=f'{arm}-s{seed}'; summary=read(WORK/'cuda'/name/'summary.json')
        rates=WORK/'cuda'/name/'healthy-rates.json'; immutable(rates,summary['healthy_rates'])
        cmd=[sys.executable,'-B','-m','tools.hb150_watchdog_v2','--attempt',str(WORK/'attempts'/f'{name}-{attempt}'),
             '--budget',str(WORK/'gpu-budget.json'),'--healthy-rates',str(rates),'--',sys.executable,'-B','-m',
             'tools.hb150_fit_v2','fit','--config',str(config_path(arm,seed)),
             '--out',str(WORK/'fits'/name),'--device','cuda','--review',str(review)]
        subprocess.run(cmd,cwd=ROOT,check=True)
    return dict(status='TWO_FITS_COMPLETE',arm=arm)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('register','validate','prepare','admit','review-bindings','pair'))
    ap.add_argument('--arm',choices=('global','fill'),default='global')
    ap.add_argument('--pl-work',type=Path,default=PL_WORK)
    ap.add_argument('--review',type=Path)
    ap.add_argument('--baseline-attestation',type=Path)
    ap.add_argument('--attempt',default='fit-v1')
    a=ap.parse_args(argv)
    if a.command=='register': result=register(a.pl_work)
    elif a.command=='validate':
        for seed in (42,43): validate(config_path(a.arm,seed))
        result=dict(status='VALIDATED',arm=a.arm)
    elif a.command=='prepare': result=prepare(a.arm)
    elif a.command=='admit': result=admit(a.arm,a.review)
    elif a.command=='review-bindings': result=review_bindings(a.arm,a.baseline_attestation)
    else: result=pair(a.arm,a.review,a.attempt)
    print(json.dumps(result))


if __name__=='__main__': main()
