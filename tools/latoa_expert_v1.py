"""v1.0.3 (2026-10-22): preregistered seed-46-only deadline revision.

Changelog v1.0.3: Taeyang/B chose s46 before scientific results after F1;
preserve v1 registration/queue/configs, use v2 route and fresh review bindings.
The successful engineering smoke is hash-bound carry-over, not a new run.

Changelog v1.0.2: local utilization <60 instead of <20; record admission
snapshot. Active-ML refusal, measured VRAM gates and watchdogs unchanged.

Changelog v1.0.1: isolated kill-on-close watchdog; explicit FAILED receipts;
reject skipped optimizer steps. Scientific v1.0.0 registration is unchanged.

CPU preparation and aggregate evaluation; B alone executes reviewed GPU queues.
Changelog: preserve HB155 crop training and fixed-final paired-seed readout;
replace population, namespace and admission, never modify historical artifacts.
"""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from tools import hb165_build_v1 as old
from tools.hb150_io_v1 import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'.local/scratch/astra-b5/latoa-expert-v1'
MODULE = 'tools.latoa_expert_v1'
FIT_MODULE = 'tools.latoa_expert_fit_v1'
PYTHON = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
GPU = 'NVIDIA GeForce RTX 4070 Ti SUPER'
SEEDS = (46,)
REGISTERED_AT = '2026-10-22T00:17:46+09:00'
BACKUP = ROOT/'.local/backups/astra-b5/latoa-expert-v1-v1.0.2'
PREREG = 'PREREGISTRATION-v2.json'
SMOKE_PREPARED_SHA = '2c5dc148073726ff873f0b570d2673e023d08eea60434988394bcb629f8b9558'
SMOKE_RECEIPT_SHA = 'f9e4102b4f2fa0424c9042544062f6a1881785f93ea90b6184bf8394200070b0'
RULERS = ('jev_hybrid', 'aquila_v1', 'dread', 'consensus')
BASELINE = ROOT/'.local/scratch/astra-b2/k1s1stride1-v1'
PARENT = ROOT/'.local/scratch/slot-b/hb155/pilot-v120/configs/crop-s42.json'
PARENT_SHA = 'e4c060ec021124a6047a392a983f74cb61f4eb9f85d80b7ab424572128f429b8'
GROUPS = ROOT/'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
ROSTER = ROOT/'.local/scratch/l1-rco/k1v3-grouping/registration.json'
RULE = dict(target='Lateral OA', crop_weight=.5, trained_weight=.5,
    baseline='rank(mean(rank(v2_stride2), rank(v3G_seed0_stride2), rank(v3G_seed1_stride2)))',
    crop='rank(seed46_epoch20_K36flip)',
    routed='copy baseline; Lateral OA = .5*rank(baseline LatOA) + .5*crop LatOA',
    rank='average ties / n; same exact 977 studies',
    rulers=list(RULERS), nonnegative_rulers_min=3, mean_delta_min=.006,
    other_11='bit identical to baseline', missing='HOLD; no rows/rulers dropped',
    clean_oof=False, exposure='EXPOSED_EXPLORATORY',
    selection='No mixture, seed or epoch search; fixed seed46 before fit results; no Gold58 rescue')

need, read, sha, uid_hash = old.need, old.read, old.sha, old.uid_hash


def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()


def exact_optimizer_steps(counters, expected):
    """Inherited `updates` means attempts; only attempts minus AMP skips are real."""
    attempts, skips = counters.get('updates'), counters.get('skips')
    need(type(attempts) is int and type(skips) is int and
         0 <= skips <= attempts, 'missing/invalid optimizer counts')
    actual = attempts - skips
    need(attempts == expected and actual == expected and skips == 0,
         'incomplete real optimizer updates; skipped attempts cannot qualify')
    return actual


def immutable(p, value):
    if p.exists(): need(read(p) == value, 'immutable candidate contract drift')
    else: atomic_json(p, value)


def population():
    from tools.l1_rco_k1_foldgate_v1 import rosters
    from tools.hb155_exclusions_v1 import UIDS
    folds = rosters(); train = sorted(sum([folds[f] for f in (1, 2, 3)], [])); val = folds[0]
    need(len(train) == 3190 and len(val) == 977 and len(set(train+val)) == 4167, 'exact977 split')
    need(not set(train+val) & set(UIDS), 'excluded study in fixed split; do not shrink')
    groups = {u: row['component_id'] for row in read(GROUPS) for u in row['members']}
    need(not {groups[u] for u in train} & {groups[u] for u in val}, 'component leakage')
    return train, val, groups


def registration():
    from tools import exp156_k_sweep as rulers
    from tools.latoa_expert_eval_v1 import CHECKPOINT_PINS
    train, val, _ = population()
    need(sha(PARENT) == PARENT_SHA, 'sealed crop recipe changed')
    return dict(schema='latoa-expert/v2', version='1.0.3', builder='astra-b5',
        registered_at=REGISTERED_AT,
        supersedes={rel(OUT/'PREREGISTRATION.json'):sha(OUT/'PREREGISTRATION.json')},
        decision='Taeyang/B before any fit result: F1 two-seed 10.47h projection; '
                 'one fixed s46, 7960 updates, approximately 5.24h training only; GO unchanged.',
        engineering_smoke_carryover={rel(BACKUP/'prepared.json'):SMOKE_PREPARED_SHA,
                                    rel(OUT/'gpu-smoke.json'):SMOKE_RECEIPT_SHA},
        registered_before_fresh_results=True, rule=RULE, fold=0, seeds=list(SEEDS), epochs=20,
        n_train=len(train), n_val=len(val), train_sha256=uid_hash(train), val_sha256=uid_hash(val),
        optimizer_steps_per_seed=len(train)//8*20, parent={rel(PARENT): PARENT_SHA},
        population_sources={rel(ROSTER): sha(ROSTER), rel(GROUPS): sha(GROUPS)},
        ruler_sources={p:sha(ROOT/p) for p in sorted(set((rulers.JEV,rulers.AQUILA,rulers.DREAD,*rulers.CONSENSUS)))},
        baseline_checkpoint_pins=CHECKPOINT_PINS,
        target_gpu=GPU, source_selection_folds=[2, 3],
        ancestry='ImageNet initialization; report-derived L2 labels and R7 geometry prior. '
        'Fold0 new for this intervention, but exposed for K1 selection and proxy rulers; NOT clean OOF.',
        candidate='P25-V10 + LatOA-only K1 seed46 crop route; conditional new stage after GO',
        scientific_readout='Exact fold0 977, four fixed correlated report-derived rulers; aggregates only',
        deadline='2026-10-22 12:00 KST', qualified_for_submission=False)


def config(seed):
    need(seed in SEEDS, 'unregistered seed')
    train, val, _ = population(); c = copy.deepcopy(read(PARENT))
    c['recipe']['roots'].update(ordered=rel(OUT/'fold0/ordered-ids.json'),
        groups=rel(OUT/'fold0/group-metadata.csv'), canonical=rel(ROSTER), order_policy='sorted-uid',
        n_train=len(train), n_val=len(val), train_sha256=uid_hash(train), val_sha256=uid_hash(val))
    steps = len(train)//8*20
    c['recipe']['updates'].update(loader_batches=len(train)//8, steps=steps, max_skips=steps//100)
    c.update(schema='latoa-expert-config/v2', version='1.0.3', experiment='LATOA_EXPERT_V1',
        seed=seed, train=train, val=val, gold=[], fold=0, optimizer_steps=steps, gpu_name=GPU,
        inputs={}, gate=RULE, holdout_status='EXPOSED_FOLD0_CONFIRMATION',
        estimand='K1 trio fixed lateral-only 50/50 rank route minus unchanged K1 trio',
        selection_exposure='Crop/OA selected on folds2/3; K1 and proxy rulers already exposed',
        coordinator_decision='Taeyang/B chose s46 only before fit results on 2026-10-22; independent review required',
        preregistration_sha256=sha(OUT/PREREG))
    return c


def config_path(seed): return OUT/'configs-v2'/f'crop-s{seed}.json'


def queue_path(mode): return OUT/('fit-s46-v103-queue.json' if mode=='fit' else 'smoke-queue.json')


def register():
    need(not (OUT/'fits').exists() and not (OUT/'RESULT.json').exists(), 'fresh results already exist')
    immutable(OUT/PREREG, registration())
    train, val, groups = population()
    immutable(OUT/'fold0/ordered-ids.json', dict(train=train, val=val))
    p=OUT/'fold0/group-metadata.csv'
    import io
    buf=io.StringIO(newline=''); writer=csv.writer(buf)
    writer.writerow(['StudyInstanceUID', 'component_id', 'split'])
    for split, ids in [('train',train), ('val',val)]:
        writer.writerows((u,groups[u],split) for u in ids)
    data=buf.getvalue().encode()
    if p.exists(): need(p.read_bytes()==data, 'group roster changed')
    else: p.write_bytes(data)
    for seed in SEEDS: immutable(config_path(seed), config(seed))
    from tools.gpu_queue import validate_plan
    for mode in ('smoke','fit'):
        plan=[dict(tag=tag(mode), command=command(mode))]
        validate_plan(plan); immutable(queue_path(mode), plan)
    return dict(status='REGISTERED', train=3190, val=977, steps_per_seed=7960, gpu_started=False)


def validate(path=None):
    need(read(OUT/PREREG) == registration(), 'preregistration drift')
    if path is None:
        for s in SEEDS: validate(config_path(s))
        return dict(status='PASS', train=3190, val=977)
    p=Path(path).resolve(); c=read(p)
    need(p==config_path(c['seed']).resolve() and c==config(c['seed']), 'exact scientific config required')
    return c


def sources():
    # Authenticate transitive local executable imports, including the unchanged fit/IO/cache stack.
    paths=old.closure(['tools/latoa_expert_v1.py', 'tools/latoa_expert_fit_v1.py',
        'tools/latoa_expert_eval_v1.py', 'tests/test_latoa_expert_v1.py',
        'tools/latoa_expert_fault_probe_v1.py', 'tests/test_latoa_expert_watchdog_v1.py'])
    return {p:sha(ROOT/p) for p in paths}


def prepare():
    validate()
    check_smoke_carryover()
    from training.arms import mil_screen_hooks_v3 as h
    c=read(config_path(46)); model=c['recipe']['model']
    need(sha(ROOT/model['initial_state'])==model['initial_file_sha256'], 'initial state pin')
    need(sha(ROOT/model['weights'])==model['weights_sha256'], 'weights pin')
    labels, semantic=h.load_labels(ROOT/c['label_path'])
    need(semantic==c['label_semantic_sha256'] and all(u in labels for u in c['train']), 'label coverage/semantics')
    cache=old.parent.cache_binding(c)
    need(shutil.disk_usage(OUT).free > 45*(1<<30), '45GiB free disk required for full resume journals')
    # Historical source dependencies actually used by the crop recipe must still match.
    parent=read(PARENT); closure=sources()
    checked={p:d for p,d in parent['inputs'].items() if p in closure}
    need(all(sha(ROOT/p)==d for p,d in checked.items()), 'historical training source drift')
    value=dict(schema='latoa-prepared/v1', sources=closure, historical_source_pins=checked,
        bindings={rel(OUT/PREREG):sha(OUT/PREREG),
                  **registration()['engineering_smoke_carryover'],
                  **{rel(config_path(s)):sha(config_path(s)) for s in SEEDS}, **cache,
                  rel(ROOT/model['initial_state']):model['initial_file_sha256'],
                  rel(ROOT/model['weights']):model['weights_sha256'],
                  rel(ROOT/c['label_path']):sha(ROOT/c['label_path']),
                  rel(OUT/'fold0/ordered-ids.json'):sha(OUT/'fold0/ordered-ids.json'),
                  rel(OUT/'fold0/group-metadata.csv'):sha(OUT/'fold0/group-metadata.csv')},
        cache='all metadata hashes and 276 shard sizes; accessed shard bytes checked by fit',
        train=3190,val=977,component_overlap=0,excluded_overlap=0,clean_oof=False)
    atomic_json(OUT/'prepared.json', value)
    return dict(status='PREPARED', source_count=len(closure), historical_sources_checked=len(checked),gpu_started=False)


def prepared():
    validate(); p=read(OUT/'prepared.json')
    need(p['sources']==sources(), 'prepared source closure changed')
    for path,d in p['bindings'].items(): need(sha(ROOT/path)==d, 'prepared input drift: '+path)
    old.parent.cache_binding(read(config_path(46)))
    return p


def review_bindings(fit=False):
    paths=[OUT/PREREG, OUT/'prepared.json', OUT/'cpu-smoke.json']
    if fit: paths += [OUT/'gpu-smoke.json', OUT/'PRE-REVIEW-v2.json']
    return {rel(p):sha(p) for p in paths}


def require_review(fit=False):
    prepared()
    smoke=read(OUT/'cpu-smoke.json')
    need(smoke['status']=='PASS' and smoke['prepared_sha256']==sha(OUT/'prepared.json')
         and smoke['cuda_initialized'] is False, 'CPU smoke binding')
    r=read(OUT/('FIT-REVIEW-v2.json' if fit else 'PRE-REVIEW-v2.json'))
    need(r.get('verdict')=='PASS' and r.get('independent') is True and
         r.get('reviewer') not in (None,'','astra-b5') and r.get('builder')=='astra-b5', 'independent review required')
    need(r.get('bindings')==review_bindings(fit), 'review artifact binding differs')
    report=(ROOT/r['report']).resolve()
    need(report.is_relative_to(ROOT) and sha(report)==r['report_sha256'], 'review report pin')
    return r


def admit():
    prepared()
    out=dict(status='BLOCKED', gpu_started=False, required_review_bindings=review_bindings())
    try: require_review(); out['status']='CPU_ADMIT_PASS'
    except (OSError,ValueError,KeyError) as exc: out['reason']='Independent review missing/invalid: '+type(exc).__name__
    atomic_json(OUT/'admit-dry-run.json',out); return out


def require_fit(path, review=None):
    validate(path); require_review(); require_review(True)
    check_smoke_carryover()
    g=read(OUT/'gpu-smoke.json')
    need(g['status']=='PASS' and g['prepared_sha256']==SMOKE_PREPARED_SHA, 'fullshape GPU smoke required')
    s=read(OUT/'gpu-smoke/summary.json')
    need(sha(OUT/'gpu-smoke/summary.json')==g['summary_sha256'] and s['full_shape']
         and not s['tiny_model'] and s['passed'], 'GPU smoke receipt identity')
    return g


def check_smoke_carryover():
    """Keep original receipts; authenticate unchanged execution/recipe for v2 review."""
    for path, digest in registration()['engineering_smoke_carryover'].items():
        need(sha(ROOT/path)==digest, 'engineering smoke carry-over drift')
    prior=read(BACKUP/'prepared.json')
    changed={'tools/latoa_expert_v1.py', 'tools/latoa_expert_eval_v1.py',
             'tests/test_latoa_expert_v1.py'}
    current=sources()
    need(set(current)==set(prior['sources']), 'smoke source closure changed')
    need(all(current[p]==d for p,d in prior['sources'].items() if p not in changed),
         'unreviewed smoke execution source changed')
    for path,d in prior['bindings'].items():
        need(sha(ROOT/path)==d, 'original smoke input drift')
    before=read(OUT/'configs/crop-s46.json'); after=read(config_path(46))
    metadata={'schema','version','gate','coordinator_decision','preregistration_sha256'}
    need({k:v for k,v in before.items() if k not in metadata}==
         {k:v for k,v in after.items() if k not in metadata}, 'smoke scientific recipe changed')


def require_runtime(path, snapshot, review=None):
    validate(path); require_review()
    need(snapshot.get('gpu')==GPU and snapshot.get('torch')=='2.5.1+cu121'
         and snapshot.get('cuda')=='12.1', 'registered local runtime required')
    for name,value in old.PINS.items(): need(snapshot.get('libs',{}).get(name)==value,'runtime library mismatch: '+name)
    for name,value in dict(matmul_tf32=False,cudnn_tf32=False,cudnn_benchmark=False,
        cudnn_deterministic=True,deterministic_algorithms=True,
        flash_sdp=True,mem_efficient_sdp=True,math_sdp=True).items():
        need(snapshot.get(name)==value,'runtime policy mismatch: '+name)
    # Child must be the exact supervised process under a live gpu_job owner.
    import psutil
    owner=int(os.environ.get('LATOA_OWNER_PID','0')); need(owner>0, 'B-owned supervisor absent')
    need(owner in [p.pid for p in psutil.Process().parents()], 'supervisor ancestry')
    if (OUT/'gpu-smoke.json').exists():
        from tools.parity_runtime import diff_runtime
        need(not diff_runtime(read(OUT/'gpu-smoke/runtime.json'), snapshot), 'runtime changed since fullshape smoke')


def cpu_smoke():
    prepared(); os.environ['CUDA_VISIBLE_DEVICES']='-1'
    from tools import latoa_expert_fit_v1 as fit
    h=fit.h; h.torch.set_num_threads(2)
    tiny=fit.fit(config_path(46),OUT/'cpu-fit',device='cpu',smoke=True)
    resumed=fit.fit(config_path(46),OUT/'cpu-fit',device='cpu',smoke=True,
                    resume=OUT/'cpu-fit/epochs/epoch-000.pt')
    need(resumed['passed'] and resumed['counters']==tiny['counters'], 'real-data full-state CPU resume')
    c=read(config_path(46)); ids=c['val'][:1]
    labels={u:h.np.zeros(12,h.np.float32) for u in ids}
    ds=fit.crops.Windows(h.MILWindows(ROOT/c['corpus'],ids,labels,36,False,46),224)
    x,_,_=ds[0]
    model=h.build_model(c['recipe']['model'],ROOT).eval().to(memory_format=h.torch.channels_last)
    with h.torch.no_grad():
        value=fit.native.NativeArm().forward(model,x[:2].unsqueeze(0),None)
    need(h.torch.isfinite(value).all().item() and tuple(value.shape)==(1,12),'real model CPU forward')
    from tools.latoa_expert_eval_v1 import route, metrics
    from tools import exp156_k_sweep as rulers
    # Exercise real structured ruler loading/shape/alignment, never output label rows or fresh scores.
    _,val,_=population(); refs=rulers.references(val)
    a=h.np.tile(h.np.arange(977,dtype=float)[:,None]/977,(1,12))
    routed=route(a,a); m=metrics(a,routed,refs)
    need(all(v['delta']==0 for v in m['rulers'].values()) and m['other_11_identical'],'identity evaluator replay')
    need(not h.torch.cuda.is_initialized() and tiny['passed'],'CPU-only engineering smoke')
    result=dict(status='PASS', prepared_sha256=sha(OUT/'prepared.json'),cuda_initialized=False,
        tiny_fit=dict(steps=tiny['steps'],passed=tiny['passed'],validation_shapes=tiny['validation_shapes'],
                      full_state_resume_passed=resumed['passed']),
        real_convnext_forward=dict(windows=2,available_k36_windows=len(x),shape=list(value.shape),finite=True,
                                  initialization_sha256=c['recipe']['model']['initial_file_sha256']),
        ruler_loader_identity_replay=dict(n=977,rulers=4,other_11_identical=True),
        limitations='CPU tiny path tests train/resume/export; real architecture forward only. GPU fullshape pending.')
    atomic_json(OUT/'cpu-smoke.json',result); return result


def tag(mode): return 'LATOA_EXPERT_V103_FIT_S46' if mode=='fit' else 'LATOA_EXPERT_V1_'+mode.upper()
def command(mode): return [PYTHON,'-B','-m',MODULE,'run-fit-s46' if mode=='fit' else 'run-'+mode]


def owner_check(mode):
    import psutil
    from tools import job_runtime as jobs
    job=read(jobs.LOGS/tag(mode)/'job.json')
    need(job['command']==command(mode) and job['command_sha256']==jobs.command_digest(command(mode),ROOT)
         and Path(job['cwd']).resolve()==ROOT, 'gpu_job exact command binding')
    wrapper=psutil.Process(job['pid'])
    need(jobs.liveness(job)=='alive' and wrapper.cmdline()[-3:]==['--_run-job',tag(mode),job['run_id']]
         and os.getppid()==job['pid'],'gpu_job live parent')
    for _ in range(50):
        job=read(jobs.LOGS/tag(mode)/'job.json')
        if job.get('child_identity')==jobs.identity(): break
        time.sleep(.1)
    need(job.get('child_identity')==jobs.identity() and job['phase']=='running','gpu_job owned child')
    return job


def run(mode):
    require_review()
    if mode=='fit': require_fit(config_path(46))
    job=owner_check(mode)
    from tools import l1_rco_runtime_v120 as safety
    from tools.latoa_expert_watchdog_v1 import supervise
    snap=safety.gpu_snapshot()
    active_ml=bool(safety.active_ml_processes(job['pid']))
    allowed=not active_ml and snap['util']<60
    atomic_json(OUT/f'local-admission-v102-{mode}.json',dict(version='1.0.2',
        status='PASS' if allowed else 'FAILED',util=snap['util'],util_limit_exclusive=60,
        active_ml=active_ml,desktop_bytes=snap['desktop_bytes'],total_bytes=snap['total_bytes']))
    need(allowed,'local GPU busy; B must reschedule')
    need(shutil.disk_usage(OUT).free>45*(1<<30),'full journal disk reserve')
    os.environ['LATOA_OWNER_PID']=str(os.getpid())
    os.environ['LATOA_DESKTOP_BYTES']=str(snap['desktop_bytes'])
    os.environ['LATOA_TOTAL_BYTES']=str(snap['total_bytes'])
    for seed in ((46,) if mode=='smoke' else SEEDS):
        dest=OUT/'gpu-smoke' if mode=='smoke' else OUT/'fits'/f'crop-s{seed}'
        need(not dest.exists(),'preserve existing attempt; no automatic retry')
        rates=None if mode=='smoke' else read(OUT/'gpu-smoke/summary.json')['healthy_rates']
        cmd=[PYTHON,'-B','-m',FIT_MODULE,mode,'--config',str(config_path(seed)),
             '--out',str(dest),'--device','cuda']
        code=supervise(cmd,OUT/f'watch-{mode}-s{seed}',healthy_rates=rates)
        if code != 0:
            prior=read(dest/'failure.json') if (dest/'failure.json').exists() else None
            atomic_json(dest/'failure.json',dict(status='FAILED',reason='watchdog_failed',
                exit_code=code,fit_failure=prior))
        need(code==0,'supervised crop job failed; preserve partial state')
        summary=read(dest/'summary.json')
        need(summary['passed'] and summary['full_shape'] and not summary['tiny_model'],'fullshape execution required')
        need(safety.memory_ok(summary['peak_reserved_bytes'],snap['desktop_bytes'],snap['total_bytes']),'measured memory gate')
        if mode=='smoke':
            atomic_json(OUT/'gpu-smoke.json',dict(status='PASS',prepared_sha256=sha(OUT/'prepared.json'),
                summary_sha256=sha(dest/'summary.json'),runtime_sha256=sha(dest/'runtime.json'),
                desktop_bytes=snap['desktop_bytes'],total_bytes=snap['total_bytes'],
                run_id=job['run_id'],independent_fit_review_required=True))
    return dict(status='SMOKE_PASS_AWAIT_FIT_REVIEW' if mode=='smoke' else 'FITS_COMPLETE_AWAIT_RESULT_REVIEW',gpu_started=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('register','validate','prepare','cpu-smoke','admit-dry-run','run-smoke','run-fit-s46','evaluate'))
    action=p.parse_args().action
    if action=='evaluate':
        from tools.latoa_expert_eval_v1 import evaluate
        value=evaluate()
    else:
        funcs={'register':register,'validate':validate,'prepare':prepare,'cpu-smoke':cpu_smoke,'admit-dry-run':admit,
               'run-smoke':lambda:run('smoke'),'run-fit-s46':lambda:run('fit')}
        value=funcs[action]()
    print(json.dumps(value,indent=2,allow_nan=False))


if __name__=='__main__': main()
