"""H-B-182 K1 v1.0.2: CPU preparation and B-only single fold-0 RCO fit.

Changelog v1.0.2: approved exact976 primary registration, all977 output retained;
paired-study uncertainty registered before outputs and exact registration review binding.
Changelog v1.0.1: enforce both registered K0 evidence hashes before admission.
v1.0.0: full-cohort cold fit, training-only bias, resume, holdout-only export.
The pinned original/stage-0 code and all prior receipts remain unchanged.
"""
from __future__ import annotations
import argparse
import csv
import difflib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import l1_rco_stage0_v1 as s

VERSION = '1.0.2'
WORK = ROOT / '.local/scratch/l1-rco/k1'
OUT = WORK / 'fit'
CACHE = Path('G:/temp_data/rsna-knee/l1rco-cache-full')
FULL_REG = ROOT / '.local/scratch/l1-rco/fullcache/registration.json'
QUEUE = ROOT / '.local/scratch/slot-b/l1rco/k1-gpu-queue.json'
K0_RESULT = ROOT / '.local/scratch/l1-rco/k0-v2/k0/result.json'
K0_REVIEW = ROOT / '.local/scratch/slot-b/l1rco/k0v2-fullcache-review-out.md'
PYTHON = s.PYTHON
TAG = 'L1_RCO_K1_F0_V1'
need, read, write, sha, digest = s.need, s.read, s.write, s.sha, s.digest


def code_pins():
    names = ['l1_rco_k1_v1.py', 'l1_rco_k1_state_v1.py', 'l1_rco_k1_eval_v1.py', 'l1_rco_k1_smoke_v1.py',
             'l1_rco_stage0_v1.py', 'l1_rco_stage0_v120.py', 'l1_rco_runtime_v120.py']
    return {'tools/' + n: sha(ROOT / 'tools' / n) for n in names}


def register():
    from tools.l1_rco_k1_eval_v1 import evaluator_registration
    s.verify_pins()
    full = read(ROOT / s.REG); roster = s.validate_roster(full)
    cache_reg = read(FULL_REG)
    need(cache_reg['roster'] == roster and set(cache_reg['train']) == set(full['train']) and
         set(cache_reg['holdout']) == set(full['val']), 'fullcache roster mismatch')
    k0 = read(K0_RESULT)
    need(k0['K0_numeric_gate'] == 'PASS', 'K0 numeric gate must pass')
    from tools.l1_rco_runtime_v120 import go_rule
    need(go_rule(k0), 'K0 measured numeric gate reproduction failed')
    reg = dict(version=VERSION, train=sorted(full['train']), holdout=sorted(full['val']),
               roster=roster, fullcache_contract_id=cache_reg['contract_id'],
               fullcache_registration_sha256=sha(FULL_REG), pins=s.PINS, code_pins=code_pins(),
               evaluator=evaluator_registration(sorted(full['val'])), exposure='EXPOSED_EXPLORATORY', clean_oof=False,
               recipe=dict(model_fold=0, folds=4, only_folds='0', holdout_fold=None, seeds=1,
                           epochs=16, batch=2, windows=64, crop=352, eval_stride=3, eval_chunk=32,
                           bias='logit(mean(train3190_only)).clamp(.001,.999)', checkpoint_steps=100),
               K0_result_sha256=sha(K0_RESULT), K0_review_sha256=sha(K0_REVIEW),
               projection=dict(fit_hours_training_lower_bound=3190*64*16/k0['windows_per_s']/3600,
                               measured_windows_per_s=k0['windows_per_s'],
                               scope='K0 loader-inclusive rate extrapolation; full-cache I/O, checkpoints and final holdout inference extra; not a deadline or timeout'))
    reg['contract_id'] = digest(reg)
    if (WORK / 'registration.json').exists():
        old = read(WORK / 'registration.json')
        need(not (OUT / 'admission.json').exists() or old == reg, 'admitted registration is immutable')
        if old != reg: require_no_candidate_evidence()
    else:
        require_no_candidate_evidence()
    write(WORK / 'registration.json', reg)
    return reg


def require_no_candidate_evidence():
    names = ('admission.json', 'result.json', 'fit-complete.json', 'holdout.csv',
             'checkpoint.pt', 'ckpt/E_prior_s0_f0.pt', 'status.json', 'models/E_prior_s0_f0.pt')
    need(not any((OUT / name).exists() for name in names), 'registration cannot change after candidate admission or output')


def validate(reg):
    need(reg['version'] == VERSION and reg['contract_id'] == digest({k:v for k,v in reg.items() if k != 'contract_id'}), 'K1 registration digest')
    need(reg['code_pins'] == code_pins() and reg['pins'] == s.PINS, 'K1 source changed; re-register before admission')
    s.verify_pins()
    full = read(ROOT / s.REG)
    need(reg['roster'] == s.validate_roster(full) and reg['train'] == sorted(full['train']) and reg['holdout'] == sorted(full['val']), 'K1 sealed full roster')
    need(sha(FULL_REG) == reg['fullcache_registration_sha256'], 'fullcache registration changed')
    need(sha(K0_RESULT) == reg['K0_result_sha256'], 'registered K0 result changed')
    need(sha(K0_REVIEW) == reg['K0_review_sha256'], 'registered K0 review changed')
    from tools.l1_rco_k1_eval_v1 import evaluator_registration
    need(reg['evaluator'] == evaluator_registration(reg['holdout']), 'evaluator registration changed')
    return reg


def training_rows(reg, source_rows):
    # Stage-0 ignores all holdout values before numeric parsing. Filter its neutral
    # rows out too: no holdout UID/value reaches this trainer, its bias or pool.
    rows = s.target_rows(dict(train=reg['train'], holdout=[]), source_rows)
    need({r[s.UID] for r in rows} == set(reg['train']) and all(s.fold(r[s.UID]) != 0 for r in rows), 'training target roster/fold')
    return rows


def derive_trainer(source):
    def patch(before, after):
        nonlocal source
        need(source.count(before) == 1, 'K1 exact trainer patch occurrence: ' + before[:90])
        source = source.replace(before, after)
    patch('    a = ap.parse_args()', '''    ap.add_argument("--l1-cpu-test", action="store_true")
    ap.add_argument("--l1-stop-steps", type=int, default=0)
    a = ap.parse_args()
    assert a.folds == 4 and a.only_folds == "0" and a.holdout_fold is None
    assert a.seeds == 1 and a.seed_base == 0 and a.arms == "E_prior"
    assert a.resume and a.ckpt_every == 1
    assert not (a.extra or a.state_targets or a.state_spec or a.px_extra or a.px_override or a.extent or a.rare_boost or a.pair_rank or a.reeval_run)
    from tools.l1_rco_k1_state_v1 import Monitor, TrainerState
    if a.l1_cpu_test:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized()
    else:
        assert not a.l1_stop_steps
    l1 = Monitor(a.out, cpu=a.l1_cpu_test) if not a.check_targets else None''')
    patch('    dev = torch.device("cpu") if a.check_targets else torch.device("cuda")', '    dev = torch.device("cpu") if a.check_targets or a.l1_cpu_test else torch.device("cuda")')
    patch('    bias0 = torch.logit(col(pool, "prior", float("nan")).mean(0).clamp(1e-3, 1 - 1e-3))', '''    assert not gold and all(pool_fold != 0)
    if not a.l1_cpu_test:
        from tools.l1_rco_k1_v1 import check_training_ids
        check_training_ids(pool)
    bias0 = torch.logit(col(pool, "prior", float("nan")).mean(0).clamp(1e-3, 1 - 1e-3))  # only3190 train pool''')
    source = source.replace('with torch.autocast("cuda", dtype=torch.bfloat16):', 'with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):')
    patch('pin_memory=train, prefetch_factor=2 if train else 1,', 'pin_memory=train and dev.type == "cuda", prefetch_factor=(2 if train else 1) if a.workers else None,')
    start = source.index('            # resumable state, written atomically')
    end = source.index('            for ep in range(start_ep, a.epochs):', start)
    source = source[:start] + '''            l1_state = TrainerState(a.out, model, opt, sched, ema, gen, dl,
                                    [pool[i] for i in tr], a.epochs, l1,
                                    resume=a.resume, every=100, stop_steps=a.l1_stop_steps)
            start_ep = l1_state.start_ep
            log = l1_state.log
''' + source[end:]
    patch('                ds.epoch = ep\n                model.train()\n                tot = n = 0', '                model.train()\n                tot, n = l1_state.begin_epoch(ep, ds)')
    patch('                for img, meta, mask, idx in dl:', '                for img, meta, mask, idx in l1_state.batches():')
    patch('                    n += len(idx)', '                    n += len(idx)\n                    l1_state.step(loss, mask, tot, n)')
    start = source.index('                if a.ckpt_every > 0 and (ep + 1)')
    end = source.index('                if a.stop_after_epoch', start)
    source = source[:start] + '                l1_state.epoch_end(ep, log)\n' + source[end:]
    start = source.index('            os.makedirs(os.path.join(a.out, "models"), exist_ok=True)', source.index('    def run(arm, seed):'))
    end = source.index('\n    cfg = ', start)
    source = source[:start] + '''            l1_state.finish(model)
            return  # exactly one fold-0 model; holdout inference belongs to the outer job
''' + source[end:]
    compile(source, 'compact_train_k1.py', 'exec')
    return source


def check_training_ids(ids):
    reg = read(WORK / 'registration.json')
    need(sorted(ids) == reg['train'] and not set(ids) & set(reg['holdout']), 'trainer must see only3190 sealed training studies')


def trainer_argv(cache):
    cmd = s.trainer_argv(cache)
    cmd[1] = str(WORK / 'compact_train_l1.py')
    cmd[cmd.index('--targets')+1] = str(WORK / 'targets.csv')
    cmd[cmd.index('--out')+1] = str(OUT)
    cmd[cmd.index('--ckpt-every')+1] = '1'
    return cmd + ['--resume']


def run_command(cache=CACHE, review=WORK / 'K1-REVIEW.json'):
    return [PYTHON, str(ROOT / 'tools/l1_rco_k1_v1.py'), 'run', '--cache', str(Path(cache).resolve()), '--review', str(Path(review).resolve())]


def prepare(reg, cache):
    validate(reg); WORK.mkdir(parents=True, exist_ok=True)
    need(not (OUT / 'admission.json').exists(), 'cannot re-prepare an admitted fit')
    with (ROOT / s.TARGETS).open(newline='', encoding='utf-8') as stream:
        rows = training_rows(reg, csv.DictReader(stream))
    path = WORK / 'targets.csv'
    with path.with_suffix('.tmp').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    s.replace(path.with_suffix('.tmp'), path)
    original = (ROOT / s.SOURCE / 'compact_train.py').read_text(encoding='utf-8')
    derivative = derive_trainer(original)
    (WORK / 'compact_train_l1.py').write_text(derivative, encoding='utf-8')
    (WORK / 'trainer.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), derivative.splitlines(True), fromfile='14767-parent', tofile='K1-v1.0.0')), encoding='utf-8')
    receipt = dict(version=VERSION, contract_id=reg['contract_id'], targets_sha256=sha(path),
                   trainer_sha256=sha(WORK / 'compact_train_l1.py'), train_count=len(rows),
                   holdout_rows_in_training_artifacts=0, bias='training-only3190',
                   argv=trainer_argv(cache), cache=str(Path(cache).resolve()))
    write(WORK / 'prepare.json', receipt)
    plan = [dict(tag=TAG, command=run_command(cache))]
    from tools.gpu_queue import validate_plan
    validate_plan(plan); write(QUEUE, plan)
    return receipt


def complete_cache(reg, cache):
    """Read the commit receipt only after completion; no running output reads.

    The builder writes receipt last after whole-payload SHA256. Admission checks
    its identity, index hash, shapes and sizes without re-reading 400 GB while a
    GPU waits. This trusts the reviewed immutable cache commit, not shard count.
    """
    import numpy as np
    cache = Path(cache)
    status = read(cache / 'status.json')
    need(status.get('phase') == 'complete' and status.get('done') == status.get('total') == 4167, 'full cache is incomplete; B launches only after complete4167')
    receipt = read(cache / 'receipt.json'); index = read(cache / 'index.json'); contract = read(cache / 'contract.json')
    ids = reg['train'] + reg['holdout']; wanted = set(ids)
    need(receipt['version'] == '1.1.0' and receipt['contract_id'] == reg['fullcache_contract_id'] == index['contract_id'] == contract['contract_id'], 'cache contract identity')
    need(receipt['studies'] == 4167 and len(receipt['ids']) == len(set(receipt['ids'])) == 4167 and set(receipt['ids']) == wanted, 'cache4167 receipt roster')
    need(index['order'] == receipt['ids'] == contract['ids'] and set(index['done']) == set(index['blocks']) == wanted and not index['errors'], 'cache index roster/errors')
    need(set(receipt['hashes']) == {'px.u8.npy','meta.f32.npy','index.json'} and sha(cache / 'index.json') == receipt['hashes']['index.json'], 'cache manifest hash')
    need(not receipt['cuda_initialized'] and contract['size'] == 384 and contract['views'] == 'v4', 'cache recipe')
    px = np.load(cache / 'px.u8.npy', mmap_mode='r'); meta = np.load(cache / 'meta.f32.npy', mmap_mode='r')
    need(px.dtype == np.uint8 and px.shape == (index['cursor'],384,384) and meta.dtype == np.float32 and meta.shape == (index['cursor'],12), 'cache layout')
    cursor = 0
    for uid in receipt['ids']:
        offset, count, blocks = index['done'][uid]
        need(offset == cursor and count > 0 and blocks == len(index['blocks'][uid]) and blocks > 0, 'cache contiguous nonempty manifest')
        need(all(offset <= row < offset+count for block in index['blocks'][uid] for _,row in block['views']), 'cache view bounds')
        cursor += count
    need(cursor == index['cursor'] and sum((cache / n).stat().st_size for n in receipt['hashes']) == receipt['bytes'], 'cache committed size')
    return dict(receipt_sha256=sha(cache / 'receipt.json'), index_sha256=receipt['hashes']['index.json'], studies=4167,
                payload_sha256=receipt['hashes'], verification='reviewed fullcache commit SHA256; admission index+layout+size')


def review_binding(reg):
    prep = read(WORK / 'prepare.json'); smoke = read(WORK / 'cpu-smoke.json')
    need(prep['contract_id'] == reg['contract_id'] and smoke['contract_id'] == reg['contract_id'], 'CPU receipt contract')
    need(prep['trainer_sha256'] == sha(WORK / 'compact_train_l1.py') and prep['targets_sha256'] == sha(WORK / 'targets.csv'), 'prepared artifact changed')
    need(smoke['status'] == 'PASS' and not smoke['cuda_initialized'], 'real CPU smoke required')
    return dict(contract_id=reg['contract_id'], registration_sha256=sha(WORK / 'registration.json'),
                primary_cohort_id_list_sha256=reg['evaluator']['primary']['id_list_sha256'], prepare_sha256=sha(WORK / 'prepare.json'),
                cpu_smoke_sha256=sha(WORK / 'cpu-smoke.json'), fullcache_registration_sha256=sha(FULL_REG))


def admit(reg, cache, review, dry_run=True, owner_pid=None):
    blockers = []; binding = cache_proof = None
    try:
        validate(reg); prep = read(WORK / 'prepare.json')
        need(prep['argv'] == trainer_argv(cache) and prep['cache'] == str(Path(cache).resolve()), 'prepared cache/argv changed')
        binding = review_binding(reg)
        r = read(review)
        need(r['verdict'] == 'PASS' and r['builder'] != r['reviewer'] and r['bindings'] == binding, 'independent exact-artifact review required')
        need(sha(ROOT / r['report']) == r['report_sha256'], 'review report changed')
    except (ValueError, KeyError, OSError) as exc: blockers.append(str(exc))
    try:
        cache_proof = complete_cache(reg, cache)
    except (ValueError, KeyError, OSError) as exc: blockers.append(str(exc))
    result = dict(status='BLOCKED' if blockers else 'CPU_ADMISSIBLE', blockers=blockers, bindings=binding, cache=cache_proof, dry_run=dry_run)
    if dry_run:
        write(WORK / 'admit-dry-run.json', result); return result
    need(not blockers, '; '.join(blockers))
    from tools.l1_rco_runtime_v120 import active_ml_processes, gpu_snapshot, memory_ok
    need(not active_ml_processes(owner_pid), 'another Python ML GPU process exists')
    snapshot = gpu_snapshot(); k0 = read(K0_RESULT)
    need(memory_ok(k0['peak_reserved'], snapshot['desktop_bytes'], snapshot['total_bytes']), 'K0 measured peak + current desktop +2GiB exceeds memory gate')
    return dict(snapshot, contract_id=reg['contract_id'], review_binding=binding, cache=cache_proof)


def require_gpu_job(cache, review):
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json, command_digest
    command = run_command(cache, review)
    with file_lock(LOGS / '.gpu-claim.lock'):
        job = read_json(LOGS / TAG / 'job.json'); parent = psutil.Process().parent()
        need(parent is not None and job.get('schema') == 'gpu-job/v2' and job.get('tag') == TAG and job.get('phase') == 'running', 'B must use gpu_queue/gpu_job')
        need(job.get('child_identity') == identity() and {k:job.get(k) for k in ('pid','create_time')} == identity(parent.pid), 'gpu_job process identity')
        need(job.get('command') == command and job.get('command_sha256') == command_digest(command, ROOT) and Path(job['cwd']).resolve() == ROOT, 'gpu_job command identity')
        need(parent.cmdline()[-3:] == ['--_run-job', TAG, job['run_id']] and any(Path(x).name == 'gpu_job.py' for x in parent.cmdline()), 'gpu_job parent identity')
        return parent.pid


def import_trainer():
    spec = importlib.util.spec_from_file_location('l1_k1_trainer', WORK / 'compact_train_l1.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def require_infer_job(cache, review):
    import psutil
    from tools.job_runtime import LOGS, file_lock, identity, read_json, command_digest
    command = run_command(cache, review)
    with file_lock(LOGS / '.gpu-claim.lock'):
        job = read_json(LOGS / TAG / 'job.json'); parent = psutil.Process().parent()
        need(parent is not None and job.get('schema') == 'gpu-job/v2' and job.get('tag') == TAG and job.get('phase') == 'running', 'inference requires live B gpu_job')
        need(job.get('child_identity') == identity(parent.pid), 'inference must be an owned run child')
        need(job.get('command') == command and job.get('command_sha256') == command_digest(command, ROOT), 'inference parent command identity')
        need(Path(parent.cmdline()[1]).resolve() == ROOT / 'tools/l1_rco_k1_v1.py' and parent.cmdline()[2] == 'run', 'inference parent must be K1 run')


def completed_fit(reg, out=OUT):
    receipt = read(Path(out) / 'fit-complete.json')
    need(receipt['status'] == 'COMPLETE' and receipt['epochs'] == 16 and receipt['model_fold'] == 0 and
         receipt['train_studies'] == 3190 and receipt['contract_id'] == reg['contract_id'] and
         receipt['model_sha256'] == sha(Path(out) / 'models/E_prior_s0_f0.pt'), 'completed fit identity')
    return receipt


def inference(reg, cache, out=OUT, cpu=False, ids=None, limit_windows=None):
    import numpy as np
    import torch
    from tools.l1_rco_k1_state_v1 import Monitor
    ct = import_trainer(); out = Path(out)
    need(cpu or ids is None, 'production inference roster is fixed977')
    ids = list(reg['holdout'] if ids is None else ids)
    need(ids and len(ids) == len(set(ids)) and set(ids) <= set(reg['holdout']), 'holdout inference roster')
    dev = torch.device('cpu' if cpu else 'cuda')
    if cpu: need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU isolation')
    else: completed_fit(reg, out)
    model = ct.Net('convnext_small', str(ROOT / s.INIT), crop=352, in_chans=3, spatial=True, extent=False).to(dev)
    model.load_deploy_state_dict(torch.load(out / 'models/E_prior_s0_f0.pt', map_location='cpu', weights_only=False))
    model.eval(); monitor = Monitor(out, cpu=cpu); monitor.phase('eval')
    ds = ct.Studies(str(cache), ids, 64, 'eval', 0, eval_stride=3, slices=3, phys_mm=3.5)
    # Same stage-0 serving windows, center crop, bf16 and encoder chunk32. A batch
    # contains exactly one study; no labels, fitter state or target file loaded.
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0 if cpu else 3, collate_fn=ct.collate)
    values = np.empty((len(ids),12), np.float32)
    with torch.no_grad():
        for img, meta, mask, index in dl:
            if limit_windows is not None: img, meta, mask = img[:,:limit_windows], meta[:,:limit_windows], mask[:,:limit_windows]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not cpu):
                logits = model(img[...,16:368,16:368].to(dev), meta.to(dev), mask.to(dev), chunk=32)
            monitor.evaluation(logits); values[index.numpy()] = logits.float().sigmoid().cpu().numpy()
    need(values.shape == (len(ids),12) and np.isfinite(values).all(), 'holdout prediction shape/finite')
    path = out / 'holdout.csv'
    with path.with_suffix('.tmp').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow([s.UID]+s.LABELS+['status','model_fold'])
        for uid,row in zip(ids,values): writer.writerow([uid]+row.tolist()+['ok',0])
    s.replace(path.with_suffix('.tmp'), path); monitor.phase('complete')
    result = dict(version=VERSION,status='COMPLETE',contract_id=reg['contract_id'],holdout_rows=len(ids),
                  holdout_sha256=sha(path),model_sha256=sha(out / 'models/E_prior_s0_f0.pt'),
                  model_fold=0, eval_chunk=32, clean_oof=False, exposure='EXPOSED_EXPLORATORY',
                  cuda_initialized=torch.cuda.is_initialized(), cpu_test=cpu,
                  decision='PENDING independent result review; CPU evaluate applies registered K1 rule')
    write(out / 'result.json', result)
    return result


def run(cache, review):
    owner = require_gpu_job(cache, review); reg = read(WORK / 'registration.json')
    admission = admit(reg, cache, review, False, owner)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'admission.json').exists():
        prior = read(OUT / 'admission.json')
        need(prior['contract_id'] == admission['contract_id'] and prior['review_binding'] == admission['review_binding'] and prior['cache'] == admission['cache'], 'resume admission identity')
    write(OUT / 'admission.json', admission)
    print(json.dumps(dict(estimated_fit_hours=reg['projection']['fit_hours_training_lower_bound'], scope=reg['projection']['scope'])), flush=True)
    commands = [trainer_argv(cache), [PYTHON,str(ROOT / 'tools/l1_rco_k1_v1.py'),'infer','--cache',str(cache),'--review',str(review)]]
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = '0'; env['CT_PREDICT_CHUNK'] = '32'
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(ROOT / s.CODE)])
    from tools.l1_rco_runtime_v120 import stale_reason, gpu_snapshot
    from tools.l1_rco_k1_state_v1 import ThroughputGuard
    for name,command in zip(('train','infer'),commands):
        if name == 'train' and (OUT / 'fit-complete.json').exists():
            completed_fit(reg)
            continue
        child = None; write(OUT / 'status.json', dict(phase='launch-'+name,time=time.time(),windows=0))
        try:
            with (OUT / (name+'.log')).open('a',encoding='utf-8') as log:
                child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                status = read(OUT / 'status.json'); throughput = ThroughputGuard(time.time())
                while child.poll() is None:
                    time.sleep(15)
                    try: status=read(OUT / 'status.json')
                    except (OSError,json.JSONDecodeError): pass
                    try: telemetry=gpu_snapshot()
                    except (OSError,subprocess.SubprocessError,ValueError): telemetry=None
                    reason=stale_reason(status,time.time(),telemetry); need(not reason,reason or '')
                    throughput.observe(status,time.time())
                need(child.returncode==0,name+' child failed; partial checkpoint retained')
        except Exception as exc:
            write(OUT/'failure.json',dict(error=str(exc),time=time.time(),partial_preserved=True)); raise
        finally:
            if child is not None and child.poll() is None:
                if os.name=='nt': subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=False)
                else: child.terminate()
                child.wait(timeout=30)
    return read(OUT/'result.json')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['register','validate','prepare','admit','cpu-smoke','run','infer','evaluate'])
    ap.add_argument('--cache',type=Path,default=CACHE); ap.add_argument('--review',type=Path,default=WORK/'K1-REVIEW.json')
    ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
    if a.action not in ('run','infer'): os.environ['CUDA_VISIBLE_DEVICES']=''
    if a.action=='register':
        reg=register(); result=dict(status='PASS',action=a.action,contract_id=reg['contract_id'],projection=reg['projection'])
    elif a.action=='run': result=run(a.cache,a.review)
    else:
        reg=validate(read(WORK/'registration.json'))
        if a.action=='validate': result=dict(status='PASS',action=a.action,train=3190,holdout=977)
        elif a.action=='prepare': result=prepare(reg,a.cache)
        elif a.action=='admit':
            need(a.dry_run,'real admission only through B gpu_queue'); result=admit(reg,a.cache,a.review)
        elif a.action=='infer':
            require_infer_job(a.cache,a.review); result=inference(reg,a.cache)
        elif a.action=='evaluate':
            from tools.l1_rco_k1_eval_v1 import evaluate
            result=evaluate(reg,dry_run=a.dry_run)
        else:
            from tools.l1_rco_k1_smoke_v1 import cpu_smoke
            result=cpu_smoke(reg,a.cache)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
