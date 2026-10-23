"""H-B-177 v1.0.3. Local seed42 and Vast seed43 paths, unchanged recipe/export ABI.
Changelog: restore gated local 4070 launch; seal both host seeds and truthful provenance.
Inherited: isolated remote paths, sealed minimal transfer, environment binding,
Device-agnostic measured-memory gate; v1.0.0 and scored serving artifacts remain unchanged.

New training recipe only. No download, Kaggle action, serving patch, or wall cap.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['KNEE_WORKSPACE_ROOT'] = str(ROOT)
for directory in (ROOT, ROOT/'training/src'):
    sys.path.insert(0, str(directory))
from tools.hb150_io_v1 import atomic_json, safe_replace, safe_unlink, _retry, _failure
from tools.pl_production_watchdog_v1 import PROGRESS, ProgressWriter
from tools.hb151_build_v3 import python_blockers, admission_snapshot, desktop_memory_snapshot, startup_publication
from hb177_recipe_v1 import RECIPE, make_model, optimization, EMA, set_lr, microbatch_step

VERSION = '1.0.3'
BUILDER = 'hb177-v103-builder'
REMOTE_ROOT = Path('/workspace/rsna-knee-abnormality-detection/.local/hb177-v103')
EXECUTION_HOSTS = {
    'local': dict(seed=42, tag='HB177_FIT_LOCAL', platform='Windows', device='RTX 4070 Ti SUPER'),
    'vast': dict(seed=43, tag='HB177_FIT', platform='Linux', device='RTX 4090 24 GB', contract='56132574'),
}
HOST = 'local' if os.name == 'nt' else 'vast'
SEED = EXECUTION_HOSTS[HOST]['seed']
TAG = EXECUTION_HOSTS[HOST]['tag']
WORK = ROOT/'.local/scratch/hb177/H-B-177'
STAGE = WORK/'v1.0.3'
BASE = STAGE
PROTOCOL = BASE/'registration.json'
STAGE = BASE/'local' if HOST == 'local' else BASE
WORK = ROOT/'.local/scratch/hb177-v103-builder/H-B-177'/HOST
RUN = STAGE/'fit'
PARENT = ROOT/'.local/scratch/hb156-l2/H-B-156-L2/package'
PARENT_MANIFEST = PARENT/'stage-convnext-l2-s42-v1/manifest.json'
INIT = PARENT/'weights/convnext-small384-imagenet.pt'
CORPUS = (ROOT if os.name == 'nt' else Path('/workspace/rsna-knee-abnormality-detection'))/'storage/cache/derived/EXP143_RAPTOR96_384_C140_S02'
SERVING = ROOT/'artifacts/experiments/HB156_KILLTEST/v1.0.0'
SOURCES = [
    'tools/hb177_build_v103.py', 'tools/hb177_vast_v103.py', 'training/src/hb177_recipe_v1.py',
    'training/src/hb156_model_v1.py', 'training/src/exp143_own0949.py',
    'training/src/raptor_common.py', 'training/src/raptor_corpus.py', 'training/src/raptor_dicom.py',
    'tools/exp143_own0949.py', 'tools/exp143_v2.py', 'tools/exp112_full_swa.py',
    'tools/exp113_radr50_local.py', 'tools/hb150_io_v1.py', 'tools/hb150_watchdog_v1.py',
    'tools/hb147_progress_v1.py', 'tools/pl_production_watchdog_v1.py',
    'tools/hb151_build_v3.py', 'tools/hb151_ancestry_v2.py',
    'training/src/exp143_v2.py', 'tools/exp101_kfold_arch.py', 'tools/exp099_kfold.py',
    'tools/exp084_group_loss.py', 'tools/exp107_kfold_radr50.py',
    'tools/job_runtime.py', 'tools/gpu_queue.py', 'tools/gpu_job.py',
    'tools/parity_runtime.py', 'tools/hb177_export_v103.py',
    *['.local/scratch/hb177/H-B-177/v1.0.3/'+n for n in ('kernel-loader.py','kernel-model.py','kernel-contract.json')],
]


def select_host(host):
    """CPU replay may select either sealed seed; GPU admission still checks the real host."""
    global HOST, SEED, TAG, STAGE, WORK, RUN
    HOST = host
    SEED, TAG = EXECUTION_HOSTS[host]['seed'], EXECUTION_HOSTS[host]['tag']
    STAGE = BASE/'local' if host == 'local' else BASE
    WORK = ROOT/'.local/scratch/hb177-v103-builder/H-B-177'/host
    RUN = STAGE/'fit'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parent():
    need(sha(PARENT_MANIFEST) == '788d98ed4b22a970bc4e09e9fcd17a864918c62e4c635fe88380e6d556a2f39c', 'parent manifest drift')
    p = read(PARENT_MANIFEST)
    need(sha(INIT) == p['init_sha256'] == 'ef97c36874559914414b9b4b7cc98f13d3b3b01c287c96d8967c4ac4b293c943', 'init drift')
    need(sha(CORPUS/'manifest.json') == p['corpus_manifest_sha256'], 'corpus manifest drift')
    need(sha(CORPUS/'build-registration.json') == p['corpus_build_sha256'], 'corpus build drift')
    need(sha(PARENT_MANIFEST.parent/p['label_file_name']) == p['label_sha256'], 'L2 drift')
    ids = p['train_ids']
    need(len(ids) == len(set(ids)) == 4347 and not set(ids) & set(p['excluded_gold_ids']), 'roster/Gold exclusion')
    need(set(ids) == set(p['train_labels']), 'label roster')
    need(all(len(v) == 12 and all(math.isfinite(x) and 0 <= x <= 1 for x in v)
             for v in p['train_labels'].values()), 'label shape/range')
    need(len(p['pos_weight']) == 12 and all(math.isfinite(x) and x > 0 for x in p['pos_weight']), 'pos_weight')
    return p


def cache_inventory():
    files=[p for p in CORPUS.iterdir() if p.is_file()]
    return dict(files=len(files), bytes=sum(p.stat().st_size for p in files))


def corpus_check(base):
    from tools.exp143_v2 import check_corpus
    if PROTOCOL.exists():
        need(cache_inventory() == read(PROTOCOL)['cache_inventory'], 'cache transfer count/bytes incomplete')
    corpus = check_corpus(CORPUS, full_hash=False)
    need(set(base['train_ids']) <= set(corpus['ids']), 'train studies absent from corpus')
    return dict(studies=len(corpus['ids']), receipts_and_shapes='PASS', full_volume_hash=False)


def register():
    need(not any((directory/n).exists() for directory in (BASE, BASE/'local')
                 for n in ('pre-review.json', 'smoke-attempt', 'fit', 'fit-review.json')), 'execution/review started; immutable')
    need(os.name == 'nt', 'registration is local-only; transfer sealed registration')
    from tools import hb177_build_v1 as original
    original.registration()
    from tools.hb177_export_v103 import seal_loader
    seal_loader()
    from tools.hb156_package_v3 import parent_inventory
    parent_inventory()
    from tools.hb177_vast_v103 import snapshot
    atomic_json(BASE/'local-environment.json', snapshot())
    p = parent()
    corpus_evidence = corpus_check(p)
    # Explicitly authenticate unchanged imported model/data implementation.
    inherited = ['training/src/hb156_model_v1.py', 'training/src/exp143_own0949.py',
                 'training/src/raptor_common.py', 'training/src/raptor_corpus.py', 'training/src/raptor_dicom.py']
    for rel in inherited:
        need(sha(ROOT/rel) == sha(PARENT/rel), 'parent implementation changed: '+rel)
    value = dict(schema='hb177-registration/v1', version=VERSION, builder=BUILDER,
        recipe=RECIPE, execution_hosts=EXECUTION_HOSTS,
        seed_policy='host seed overrides legacy recipe.seed only; both registered before either fit',
        parent_manifest_sha256=sha(PARENT_MANIFEST), init_sha256=sha(INIT),
        corpus_manifest_sha256=p['corpus_manifest_sha256'], label_sha256=p['label_sha256'],
        train_n=4347, train_ids_sha256=digest(p['train_ids']), labels_sha256=digest(p['train_labels']),
        pos_weight_sha256=digest(p['pos_weight']), exposure=p['exposure'], clean_oof=False,
        total_updates=5440, updates_per_epoch=272, source_pins={r:sha(ROOT/r) for r in SOURCES},
        serving_slug='taeyangg4/knee-hb156-cnxs-l2-v1', serving_files=original.serving_inventory(),
        original_registration_sha256=sha(original.PROTOCOL),
        local_environment_sha256=sha(BASE/'local-environment.json'),
        local_workspace=str(ROOT.resolve()),
        remote_contract='56132574', remote_root=str(REMOTE_ROOT),
        cache_inventory=cache_inventory(),
        corpus_validation=corpus_evidence,
        serving_change='no serving edits; EMA state in explicit HB156 legacy ABI envelope; training_provenance is authoritative',
        watchdog_policy=dict(total=None, idle=900, stall=600, healthy_rates=None),
        decision=dict(alive_at_least=.950, dead_at_most=.946, otherwise='HOLD', baseline=.946,
                      selection='local seed42 and Vast seed43 all-data fits; final epoch EMA only; no public tuning'),
        fold0='DEFERRED: optional separate preregistered held-out fit needs B time/resource decision; not executed by this card',
        init_source='https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k_384',
        max_wall_seconds=None)
    atomic_json(PROTOCOL, value)
    atomic_json(BASE/'lineage.json', dict(parent_inventory='verified complete',
        unchanged_imports=inherited, serving_files=value['serving_files'], embedded_files=original.embedded_inventory(),
        serving_diff=[], existing_files_modified=[], new_files=SOURCES[:2]))
    return dict(status='REGISTERED_CPU', version=VERSION, train_n=4347, total_updates=5440, execution_hosts=EXECUTION_HOSTS)


def registration():
    p = read(PROTOCOL)
    need(p['schema'] == 'hb177-registration/v1' and p['recipe'] == RECIPE, 'registration recipe drift')
    need(p.get('execution_hosts') == EXECUTION_HOSTS, 'registered host/seed drift')
    need(p['source_pins'] == {r:sha(ROOT/r) for r in SOURCES}, 'source drift; rebuild/review required')
    need(p['version'] == VERSION and p['remote_contract'] == '56132574', 'remote registration identity')
    need(p['local_environment_sha256'] == sha(BASE/'local-environment.json'), 'environment reference drift')
    base = parent()
    need(p['parent_manifest_sha256'] == sha(PARENT_MANIFEST), 'registered parent drift')
    need(p['train_ids_sha256'] == digest(base['train_ids']) and p['labels_sha256'] == digest(base['train_labels'])
         and p['pos_weight_sha256'] == digest(base['pos_weight']), 'registered data drift')
    return p, base


def save_torch(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.partial')
    PROGRESS.phase_start('checkpoint_write')
    try:
        with _retry(lambda: tmp.open('wb')) as stream:
            torch.save(value, ProgressWriter(stream, PROGRESS))
            stream.flush(); os.fsync(stream.fileno())
        safe_replace(tmp, path)
    except OSError as exc:
        _failure(path, exc, True)
    finally:
        safe_unlink(tmp)


def cpu_guard():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # PowerShell empty env assignment deletes it.
    import torch
    need(not torch.cuda.is_initialized(), 'CPU validation started after CUDA init')
    torch.set_num_threads(4)


def setup(device, base):
    import torch
    from exp143_own0949 import _cpu_seed, install_portable_gold_blocker
    install_portable_gold_blocker()
    _cpu_seed(SEED)
    model = make_model(INIT).to(device)
    ema = EMA(model)
    optimizer = optimization(model)
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(base['pos_weight'], device=device))
    return model, ema, optimizer, loss


def real_batches(base, count):
    from exp143_own0949 import StudyDataset, collate
    ds = StudyDataset(CORPUS, base['train_ids'], base['train_labels'], SEED)
    return [collate([ds[i % len(ds)]]) for i in range(count)]


def make_loader(base, *, pin_memory):
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from exp143_own0949 import StudyDataset, collate, worker_init
    ds=StudyDataset(CORPUS,base['train_ids'],base['train_labels'],SEED)
    sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=SEED,drop_last=False)
    # Worker startup must not consume the model RNG on epoch-boundary resume.
    generator=torch.Generator().manual_seed(SEED)
    loader=DataLoader(ds,batch_size=1,sampler=sampler,drop_last=False,num_workers=4,
        pin_memory=pin_memory,persistent_workers=True,collate_fn=collate,worker_init_fn=worker_init,
        generator=generator)
    need(len(loader)==4347, 'all-data roster length')
    return ds,sampler,loader


def checkpoint(ema, epoch):
    from tools.hb177_export_v103 import export
    return export(ema, epoch, PROTOCOL, RUN.relative_to(ROOT).as_posix(), RECIPE, VERSION, seed=SEED, host=HOST)


def cpu_validate():
    cpu_guard()
    import torch
    import numpy as np
    from exp143_own0949 import Corpus, infer_volume
    from hb156_model_v1 import build_model
    start = time.monotonic()
    _, base = registration()
    corpus_check(base)
    model, ema, opt, loss = setup('cpu', base)
    ds,sampler,loader = make_loader(base, pin_memory=False)
    ds.set_epoch(0); sampler.set_epoch(0)
    batch = [next(iter(loader))]
    del loader,ds,sampler
    need(tuple(batch[0][0].shape) == (1,12,3,384,384), 'real train shape')
    set_lr(opt, 0, 5440, 544)
    model.train()
    microbatch_step(model, opt, ema, batch, loss, 'cpu', 'none')
    # Full resume, optimizer moments and EMA checked on actual real-data state.
    save_torch(WORK/'cpu-resume.pt', dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict()))
    saved = torch.load(WORK/'cpu-resume.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'], strict=True); opt.load_state_dict(saved['optimizer'])
    ema.load_state_dict(saved['ema'])
    del saved, opt, model, batch
    save_torch(WORK/'cpu-export.pt', checkpoint(ema, 0))
    from tools.hb177_export_v103 import load_with_kernel
    serving = load_with_kernel(WORK/'cpu-export.pt')
    volume, mask = Corpus(CORPUS).get(base['train_ids'][0])
    # Full K94, chunk16, two-view 320-serving replay on actual study for both constructors.
    expected = infer_volume(ema.module.eval(), volume, mask, 'cpu', 'none')
    actual = infer_volume(serving, volume, mask, 'cpu', 'none')
    error = float(np.max(np.abs(expected-actual)))
    need(error == 0. and np.isfinite(actual).all() and actual.shape == (12,), 'CPU serving parity')
    need(not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES'] == '', 'CUDA touched')
    value = dict(status='PASS_CPU', protocol_sha256=sha(PROTOCOL), version=VERSION, host=HOST, seed=SEED,
        seconds=time.monotonic()-start, real_studies=1, train_shape=[1,12,3,384,384],
        backward=True, optimizer_step=True, ema_update=True, resume_roundtrip=True,
        frozen_kernel_loader=True, strict_state_dict=True, legacy_metadata_accepted=True,
        real_loader_workers=4, model_rng_independent_of_loader=True,
        serving_shape=[94,3,320,320], serving_views=2, max_abs_error=error,
        cuda_executed=False, CUDA_VISIBLE_DEVICES='', exposure='EXPOSED', clean_oof=False)
    atomic_json(STAGE/'cpu-validation.json', value)
    return value


def require_cpu():
    v = read(STAGE/'cpu-validation.json')
    need(v.get('status') == 'PASS_CPU' and v.get('protocol_sha256') == sha(PROTOCOL)
         and v.get('host') == HOST and v.get('seed') == SEED, 'stale/missing CPU validation')


def review_bindings(fit=False):
    paths = {'protocol_sha256':PROTOCOL, 'cpu_sha256':STAGE/'cpu-validation.json',
             'environment_sha256':BASE/'local-environment.json' if HOST == 'local' else STAGE/'remote-environment.json'}
    if fit:
        paths.update(runtime_sha256=STAGE/'cuda-runtime.json', smoke_sha256=STAGE/'cuda-smoke.json', smoke_receipt_sha256=STAGE/'smoke-attempt/receipt.json')
    return {k:sha(v) for k,v in paths.items()}


def check_review(review, bindings, scope):
    need(review.get('verdict') == 'PASS' and review.get('independent') is True
         and review.get('reviewer') not in (None, '', BUILDER) and review.get('scope') == scope,
         'independent '+scope+' review required')
    need(all(review.get(k) == v for k,v in bindings.items()), 'stale review')


def check_memory(s):
    total, desktop = s.get('total_mib',0), s.get('desktop_used_mib',-1)
    need(math.isfinite(total) and math.isfinite(desktop) and total > 0 and 0 <= desktop <= total, 'unmeasured GPU memory')
    need(s.get('train_updates',0) >= 3 and s.get('train_shape') == [1,12,3,384,384]
         and s.get('accumulation') == 16 and s.get('serving_views') == 2
         and s.get('serving_k') == 94 and s.get('precision') == 'bf16'
         and s.get('grad_checkpointing') is True, 'incomplete full-shape smoke')
    for key in ('train_peak_reserved_bytes','serve_peak_reserved_bytes'):
        peak = s.get(key,0)
        need(math.isfinite(peak) and 0 < peak <= 12*1024**3 and desktop+peak/1024**2+2048 <= total, 'memory gate')
    need(math.isfinite(s.get('seconds_per_update',0)) and s['seconds_per_update'] > 0, 'unmeasured throughput')


def require_admission(fit=False):
    protocol, _ = registration(); require_cpu()
    from tools.hb177_vast_v103 import require_environment, require_local_environment
    if HOST == 'local':
        need(os.name == 'nt', 'local GPU path requires Windows')
        need(str(ROOT.resolve()) == protocol['local_workspace'], 'local workspace drift')
        require_local_environment()
    else:
        need(os.name == 'posix' and ROOT.resolve() == REMOTE_ROOT, 'Vast-only GPU path')
        require_environment()
    check_review(read(STAGE/'pre-review.json'), review_bindings(), 'pre-execution')
    approval = read(STAGE/'b-authorization.json')
    need(approval.get('authorized_by') == 'B' and approval.get('protocol_sha256') == sha(PROTOCOL)
         and approval.get('pre_review_sha256') == sha(STAGE/'pre-review.json'), 'B authorization missing/stale')
    need(approval.get('allow_smoke') is True, 'B smoke authorization required')
    if fit:
        smoke = read(STAGE/'cuda-smoke.json')
        check_memory(smoke)
        need(smoke.get('protocol_sha256') == sha(PROTOCOL)
             and smoke.get('host') == HOST and smoke.get('seed') == SEED, 'stale smoke')
        receipt = read(STAGE/'smoke-attempt/receipt.json')
        need(receipt.get('exit_code') == 0 and receipt.get('reason') == 'completed' and receipt.get('survivors') == [], 'smoke supervision failed')
        check_review(read(STAGE/'fit-review.json'), review_bindings(True), 'smoke-result-and-fit')
        need(approval.get('allow_fit') is True and approval.get('fit_review_sha256') == sha(STAGE/'fit-review.json'), 'B fit authorization required')


def require_gpu_job(mode, child=False, resume=False):
    import psutil
    from tools.job_runtime import file_lock, command_digest
    def check():
        tag = TAG if mode == 'fit' else TAG+'_SMOKE'
        with file_lock(ROOT/'artifacts/logs/.gpu-claim.lock', blocking=False):
            job = read(ROOT/'artifacts/logs'/tag/'job.json')
            ancestors = [psutil.Process()] + psutil.Process().parents()
            identities = {(p.pid,p.create_time()) for p in ancestors}
            direct = job.get('child_identity') or {}
            need((job['pid'],job['create_time']) in identities and
                 (direct.get('pid'),direct.get('create_time')) in identities, 'gpu_job live ancestry required')
            expected = [sys.executable,'-B','-m','tools.hb177_build_v103',mode] + (['--resume'] if resume else [])
            need(job['tag'] == tag and job['phase'] == 'running' and job['command'] == expected
                 and job['command_sha256'] == command_digest(expected, ROOT), 'gpu_job identity/command')
            need(not python_blockers(admission_snapshot(), [p.pid for p in ancestors]), 'other Python ML compute active')
            if child:
                attempt = Path(os.environ.get('HB147_HEARTBEAT','')).parent
                need(attempt.parent.resolve() == STAGE.resolve() and
                     (attempt.name == 'smoke-attempt' if mode == 'smoke' else attempt.name.startswith('fit-attempt-')), 'watchdog attempt path')
                identity = read(attempt/'identity.json')
                need(identity['pid'] == os.getpid() and identity['create_time'] == psutil.Process().create_time(), 'watchdog child identity')
                need(read(attempt/'command.json')['token'] == os.environ.get('HB147_TOKEN') and
                     Path(os.environ['HB147_HEARTBEAT']).resolve() == (attempt/'heartbeat.json').resolve(), 'watchdog token binding')
    startup_publication(check)


def prepare():
    registration(); require_cpu()
    from tools.gpu_queue import validate_plan
    for mode in ('smoke','fit'):
        plan = [dict(tag=TAG if mode == 'fit' else TAG+'_SMOKE',
            command=[sys.executable,'-B','-m','tools.hb177_build_v103',mode], continue_on_failure=False)]
        validate_plan(plan)
        atomic_json(STAGE/(mode+'-queue.json'), plan)
    return dict(status='PREPARED_CPU', admitted=False, host=HOST, seed=SEED, tag=TAG)


def admit_dry_run():
    registration(); require_cpu()
    blockers = []
    for fit in (False,True):
        try:
            require_admission(fit)
        except (ValueError,FileNotFoundError) as exc:
            blockers.append(dict(mode='fit' if fit else 'smoke', reason=str(exc)))
    # nvidia-smi is read-only telemetry, not CUDA context creation/admission.
    snapshot = admission_snapshot()
    value = dict(status='DRY_RUN_ONLY', admitted=False, gates=blockers,
                 python_ml_blockers=python_blockers(snapshot), desktop=desktop_memory_snapshot())
    atomic_json(STAGE/'admit-dry-run.json', value)
    return value


def cuda_smoke():
    require_admission(); require_gpu_job('smoke', child=True)
    desktop = desktop_memory_snapshot()
    import torch
    from exp143_own0949 import Corpus, infer_volume
    need(math.isfinite(desktop['total_mib']) and desktop['total_mib'] > 0
         and 0 <= desktop['used_mib'] <= desktop['total_mib'], 'unmeasured GPU memory')
    torch.set_num_threads(4); torch.cuda.set_device(0)
    need(torch.cuda.is_bf16_supported(), 'native bf16 required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    torch.cuda.manual_seed_all(SEED)
    from tools.hb177_vast_v103 import snapshot
    atomic_json(STAGE/'cuda-runtime.json', snapshot(cuda=True))
    _, base = registration()
    model, ema, opt, loss = setup('cuda', base)
    torch.cuda.reset_peak_memory_stats()
    times=[]
    for update in range(3):
        PROGRESS.phase_start('training_step')
        start=time.monotonic()
        batches=real_batches(base,16)
        set_lr(opt,update,5440,544); model.train()
        microbatch_step(model,opt,ema,batches,loss,'cuda','bf16')
        torch.cuda.synchronize(); times.append(time.monotonic()-start)
        PROGRESS.completed('training_step',{'update':update+1})
        del batches
    train_peak=torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    PROGRESS.phase_start('validation')
    volume,mask=Corpus(CORPUS).get(base['train_ids'][0])
    infer_volume(ema.module.eval(),volume,mask,'cuda','bf16')
    torch.cuda.synchronize()
    PROGRESS.completed('validation',{'studies':1})
    seconds=sum(times[1:])/2
    value=dict(protocol_sha256=sha(PROTOCOL),host=HOST,seed=SEED,device_name=desktop['device_name'],total_mib=desktop['total_mib'],
        desktop_used_mib=desktop['used_mib'],train_updates=3,train_shape=[1,12,3,384,384],
        accumulation=16,serving_views=2,serving_k=94,precision='bf16',grad_checkpointing=True,
        train_peak_reserved_bytes=train_peak,serve_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        seconds_per_update=seconds,projected_fit_hours=seconds*5440/3600,
        healthy_rates={'train_step':1/seconds}, projection_note='warm timing includes data load; checkpoint IO excluded; no duration rejection')
    check_memory(value)
    atomic_json(STAGE/'cuda-smoke.json', value)


def fit(resume=False):
    require_admission(True); require_gpu_job('fit',child=True,resume=resume)
    import torch
    from exp143_own0949 import _cpu_seed
    need(not (RUN/'complete.json').exists(), 'completed fit immutable')
    need(resume or not RUN.exists(), 'existing partial fit; explicit resume required')
    RUN.mkdir(parents=True,exist_ok=resume)
    torch.set_num_threads(4); torch.cuda.set_device(0)
    need(torch.cuda.is_bf16_supported(), 'native bf16 required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=True
    torch.cuda.manual_seed_all(SEED)
    _, base=registration()
    model,ema,opt,loss=setup('cuda',base)
    identity=dict(protocol_sha256=sha(PROTOCOL),host=HOST,seed=SEED,torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(),precision='bf16')
    corpus_check(base)
    ds,sampler,loader=make_loader(base, pin_memory=True)
    start_epoch,updates=0,0
    if resume:
        state=torch.load(RUN/'resume.pt',map_location='cpu',weights_only=True)
        need(state['identity']==identity,'resume identity mismatch')
        model.load_state_dict(state['model'],strict=True); ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer']); start_epoch=state['epoch']; updates=state['updates']
        need(updates==272*start_epoch and ema.updates==updates,'resume update mismatch')
        del state
    def save(epoch):
        save_torch(RUN/'resume.pt',dict(model=model.state_dict(),ema=ema.state_dict(),
            optimizer=opt.state_dict(),epoch=epoch,updates=updates,identity=identity))
    if not resume:
        save(0)
    for epoch in range(start_epoch,20):
        _cpu_seed(SEED*1000+epoch); torch.cuda.manual_seed_all(SEED*1000+epoch)
        ds.set_epoch(epoch); sampler.set_epoch(epoch); model.train()
        group=[]; PROGRESS.phase_start('training_step')
        for i,batch in enumerate(loader):
            group.append(batch)
            if len(group)==16 or i+1==len(loader):
                set_lr(opt,updates,5440,544)
                microbatch_step(model,opt,ema,group,loss,'cuda','bf16')
                updates+=1; group=[]
                PROGRESS.completed('training_step',{'update':updates})
        save(epoch+1)
    need(updates==5440 and ema.updates==5440,'incomplete fit')
    save_torch(RUN/'hb177-final-ema.pt',checkpoint(ema,20))
    atomic_json(RUN/'complete.json',dict(status='COMPLETE_AWAITING_INDEPENDENT_RESULT_REVIEW',
        protocol_sha256=sha(PROTOCOL),host=HOST,seed=SEED,checkpoint_sha256=sha(RUN/'hb177-final-ema.pt'),updates=updates,
        exposure='EXPOSED',clean_oof=False,selection='final-epoch-EMA'))


def supervise(mode,resume=False):
    require_admission(mode=='fit'); require_gpu_job(mode,resume=resume)
    from tools.hb150_watchdog_v1 import supervise as watch
    if mode=='smoke':
        need(not resume,'smoke cannot resume')
        attempt=STAGE/'smoke-attempt'; rates=None
    else:
        # Preserve partial attempts; numbers are execution attempts, never candidates.
        index=1
        while (STAGE/f'fit-attempt-{index}').exists(): index+=1
        attempt=STAGE/f'fit-attempt-{index}'
        rates=None  # Explicit user override: idle/stall only, no throughput abort.
        live=desktop_memory_snapshot(); smoke=read(STAGE/'cuda-smoke.json')
        check_memory(dict(smoke,desktop_used_mib=live['used_mib'],total_mib=live['total_mib'],device_name=live['device_name']))
    command=[sys.executable,'-B','-m','tools.hb177_build_v103','_'+mode]+(['--resume'] if resume else [])
    return watch(command,attempt,total=None,idle=900,stall=600,healthy_rates=None,
                 gpu_job_log_dir=str(ROOT/'artifacts/logs'/(TAG if mode=='fit' else TAG+'_SMOKE')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['register','validate','prepare','admit-dry-run','smoke','fit','_smoke','_fit'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--host', choices=tuple(EXECUTION_HOSTS), help='CPU preparation only; GPU host is automatic')
    args=parser.parse_args()
    if args.host:
        need(args.action in ('validate','prepare','admit-dry-run'), '--host is CPU-only')
        select_host(args.host)
    need(not args.resume or args.action in ('fit','_fit'),'resume only for fit')
    if args.action in ('smoke','fit'):
        return supervise(args.action,args.resume)
    if args.action=='_smoke': cuda_smoke(); return 0
    if args.action=='_fit': fit(args.resume); return 0
    cpu_guard()
    result={'register':register,'validate':cpu_validate,'prepare':prepare,'admit-dry-run':admit_dry_run}[args.action]()
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
