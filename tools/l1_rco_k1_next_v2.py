"""v2.0.0: additive local seed1 and Vast seed2; frozen v1 alternatives unchanged.

Changelog: local reference cache on F: (no META_EPS), explicit F1 memory gate;
Vast seed2 retains META_EPS and the predecessor SUCCESS contract.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
TASK = ROOT / '.local/scratch/astra-b2/vast-next-v1'
LOCAL_CACHE = Path('F:/rsna-knee-cache/l1rco-cache-full')
MODES = {'local_all_seed1': ('k1_v3g_all_s1', 1, 'H235_K1_V3G_LOCAL_ALL_S1'),
         'all_seed2': ('k1_v3g_all_s2', 2, 'H235_K1_V3G_ALL_S2')}


def local_memory_gate(k, result):
    f1 = k.read(k.BASE/'f1/fit/fit-complete.json')
    k.need(f1['contract_id'] == '9bc848122830b8c903f4656a1ea555535584e2e23652a91063da72a55e2d08dc'
           and f1['status'] == 'COMPLETE' and f1['optimizer_steps'] == 15360, 'F1 memory evidence identity')
    k.need(f1['peak_reserved'] <= 12*2**30 and result['desktop_bytes'] + f1['peak_reserved'] + 2*2**30 <= result['total_bytes'], 'F1 peak + desktop + 2GiB exceeds VRAM')
    result['f1_peak_reserved'] = f1['peak_reserved']


def backend(mode):
    if mode not in MODES:
        raise ValueError('unknown next v2 mode')
    spec = importlib.util.spec_from_file_location('tools._next_v2_' + mode, ROOT/'tools/l1_rco_k1_all_v1.py')
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    mid, seed, tag = MODES[mode]
    local = mode == 'local_all_seed1'
    k.WORK = TASK/mode
    k.OUT, k.TAG = k.WORK/'fit', tag
    k.REVIEW = k.WORK/('REVIEW.json' if local else 'REVIEW-META-EPS.json')
    if local:
        k.CACHE = LOCAL_CACHE
    script = ROOT/('tools/l1_rco_k1_' + mode + '_v1.py')
    base_reg, base_source, base_argv = k.expected_registration, k.derive_trainer, k.trainer_argv
    base_pins, base_admit = k.code_pins, k.admit

    def pins():
        names = ['l1_rco_k1_next_v2.py', 'l1_rco_k1_nextsmoke_v1.py',
                 'l1_rco_k1_nextlaunch_v2.py', 'l1_rco_k1_nexthost_v1.py',
                 'l1_rco_k1_' + mode + '_v1.py', 'l1_k1' + mode + '_trainer_v1.py',
                 'l1_rco_k1_meta_admit_v1.py', 'l1_rco_k1_foldcache_v2.py']
        terminal = '.local/scratch/astra-b2/fullbag-post-v1/fullbag_terminal_v1.py'
        return dict(base_pins(), **{'tools/'+n:k.sha(ROOT/'tools'/n) for n in names},
                    **{terminal:k.sha(ROOT/terminal)})

    def registration():
        reg = base_reg()
        reg.update(model_id=mid, model_seed=seed, variant='standard',
                   producer_mode=mode,
                   storage_note='Inherited s0 filenames are storage names; model_seed/seed_base bind actual RNG.',
                   registered_parent=dict(tag='H224_K1_V3G_ALL',
                       model_sha256='47f47cf14615731de36c53735cafcd0ade9b07a0d48889c7d98ca3088fde66e0',
                       contract_id='92cc6c854ce6b691be8aa1eb886068af9d8cfd1479f207ac270918e8a623621b'),
                   registered_change=f'Model seed_base 0 -> {seed}; all-data recipe/roster unchanged.')
        reg['recipe'].update(seed_base=seed, unchanged=f'Standard all-data v3-G recipe except model RNG seed {seed}.')
        reg['changes_vs_seed0'] = [reg['registered_change']]
        if local:
            reg['execution'] = dict(host='Windows reference host', cache=str(LOCAL_CACHE), meta_eps=False)
            reg['projection'].update(fit_hours_training_lower_bound=None,
                fit_hours_planning_low=20830*3.1/3600,
                fit_hours_calendar_estimate=20830*(48514.917935699996/15360)/3600,
                scope='NVMe s/step UNMEASURED. F1 G: train_seconds/steps=3.1585; provisional 3.1-3.16 s/step, not a bound. Same compute; NVMe may reduce cache page faults only.')
        else:
            reg['projection'].update(fit_hours_training_lower_bound=11597.55070862273/3600,
                fit_hours_calendar_estimate=11597.55070862273/3600,
                scope='4090 seed0 all-data 20830 steps / 11597.55070862273 train seconds; seed2 unmeasured; excludes checkpoints and guard inference.')
        reg.pop('contract_id')
        reg['contract_id'] = k.digest(reg)
        return reg

    def derive(source):
        source = base_source(source).replace('tools.l1_rco_k1_all_v1', 'tools.l1_rco_k1_'+mode+'_v1')
        old = 'assert a.seeds == 1 and a.seed_base == 0 and a.arms == "E_prior"'
        k.need(source.count(old) == 1, 'seed assertion anchor')
        source = source.replace(old, old.replace('seed_base == 0', f'seed_base == {seed}'))
        compile(source, 'next-v2-trainer', 'exec')
        return source

    def argv(cache, reg):
        cmd = base_argv(cache, reg)
        if '--seed-base' in cmd:
            cmd[cmd.index('--seed-base')+1] = str(seed)
        else:
            cmd += ['--seed-base', str(seed)]
        return cmd

    def command(fold='all', cache=None, review=None, gpu=0, action='run'):
        k.need(fold == 'all' and gpu == 0, 'all-data GPU0 only')
        return [sys.executable, str(script), action, '--cache', str(Path(cache or k.CACHE).resolve()),
                '--review', str(Path(review or k.REVIEW).resolve())]

    def evidence(cache):
        if local:
            k.need(os.name == 'nt' and Path(cache).resolve() == LOCAL_CACHE.resolve(), 'local reference F: cache required')
            # Payload SHA256 verification on the copied arrays is B's prelaunch step.
            # Preserve the reference receipt; do not create a fabricated parity receipt.
            reference = k.parent.CACHE/'receipt.json'
            k.need(k.sha(Path(cache)/'receipt.json') == k.sha(reference), 'reference cache receipt differs')
            k.complete_cache(k.expected_registration(), cache)
            return dict(local_reference_bindings=dict(receipt_sha256=k.sha(reference),
                payload_sha256=k.read(reference)['hashes'], cache=str(LOCAL_CACHE.resolve()),
                mode='EXACT_REFERENCE_COPY', meta_eps_required=False))
        from tools import l1_rco_k1_meta_admit_v1 as meta
        proof = meta.parity.validate_receipt(cache)
        k.need(proof['mode'] == 'META_EPS', 'explicit META_EPS required')
        return dict(meta_eps_bindings=meta.evidence_binding(cache))

    def admit(reg, cache, review, dry_run=True, owner_pid=None):
        try:
            binding = evidence(cache)
            r = k.read(review)
            k.need(all(r.get(n) == v for n,v in binding.items()), 'independent host/cache review required')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if not dry_run:
                raise ValueError('next-job admission blocked: '+str(exc)) from exc
            result = dict(status='BLOCKED', blockers=[str(exc)], dry_run=True)
            k.write(k.WORK/'admit-dry-run.json', result)
            return result
        result = base_admit(reg, cache, review, dry_run, owner_pid)
        if local and not dry_run:
            local_memory_gate(k, result)
        result.update(binding)
        return result

    k.code_pins, k.expected_registration, k.derive_trainer = pins, registration, derive
    k.trainer_argv, k.command, k.admit = argv, command, admit
    k.launch_command = lambda **kw: [sys.executable,str(ROOT/'tools/gpu_job.py'),'--tag',k.TAG,'--']+command(**kw)
    k.import_trainer = lambda: __import__('tools.l1_k1'+mode+'_trainer_v1',fromlist=['Net'])
    from tools import l1_rco_k1_meta_admit_v1 as meta
    k.require_infer_job = lambda cache,review: meta.require_infer_job(k,cache,review)
    k.next_evidence, k.next_mode = evidence, mode
    return k


def main(k):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['register','prepare','validate','cpu-smoke','admit','review-binding','run','infer','export'])
    ap.add_argument('--cache', type=Path, default=k.CACHE)
    ap.add_argument('--review', type=Path, default=k.REVIEW)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.action not in ('run','infer'):
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if k.next_mode == 'local_all_seed1':
        k.need(os.name == 'nt' and a.cache.is_absolute() and a.cache.resolve() == LOCAL_CACHE.resolve(), 'local job requires absolute F: cache')
    if a.action == 'register':
        reg=k.register()
        result=dict(status='PASS',contract_id=reg['contract_id'],train=4167,holdout=0,steps=20830)
    else:
        reg=k.validate(k.read(k.WORK/'registration.json'))
        if a.action == 'prepare':
            p=k.prepare(reg,a.cache)
            result=dict(status='PASS',train=p['train_count'],trainer_sha256=p['trainer_sha256'])
        elif a.action == 'validate':
            from tools.l1_rco_k1_nexthost_v1 import prepared_hashes
            p=k.read(k.WORK/'prepare.json')
            k.need(all(p[n]==v for n,v in prepared_hashes(reg,k).items()), 'prepared hashes')
            k.need(p['argv']==k.trainer_argv(a.cache,reg), 'prepared argv')
            result=dict(status='PASS',contract_id=reg['contract_id'])
        elif a.action == 'cpu-smoke':
            from tools.l1_rco_k1_nextsmoke_v1 import cpu_smoke
            result=cpu_smoke(k,reg,a.cache)
        elif a.action == 'admit':
            k.need(a.dry_run,'admission only inside gpu_job')
            result=k.admit(reg,a.cache,a.review)
        elif a.action == 'review-binding':
            result={}
            try: result.update(bindings=k.review_binding(reg), **k.next_evidence(a.cache),status='PASS')
            except (OSError,ValueError,KeyError,TypeError) as exc:
                result.update(status='BLOCKED',blockers=[str(exc)])
        elif a.action == 'run':
            if k.next_mode == 'all_seed2':
                from tools.l1_rco_k1_nextlaunch_v2 import require_selection
                k.need(os.name!='nt','Vast only')
                require_selection(k)
            result=k.run(a.cache,a.review)
        elif a.action == 'infer':
            k.require_infer_job(a.cache,a.review)
            result=k.inference(reg,a.cache)
        else: result=k.export(reg)
    print(json.dumps(result,allow_nan=False))
