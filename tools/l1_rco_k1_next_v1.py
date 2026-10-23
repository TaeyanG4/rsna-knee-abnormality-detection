"""v1.0.0: isolated all-data alternatives; no changes to running producers.

FULLBAG_ALL changes the fullbag roster only (derived final-EMA/no-holdout
bookkeeping). ALL_SEED1 changes the standard all-data RNG seed only.
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
VERSION = '1.0.0'
TASK = ROOT / '.local/scratch/astra-b2/vast-next-v1'
MODES = {'fullbag_all': ('k1_v3g_fullbag_all', 0, 'H235_K1_V3G_FULLBAG_ALL'),
         'all_seed1': ('k1_v3g_all_s1', 1, 'H235_K1_V3G_ALL_S1')}


def backend(mode):
    if mode not in MODES:
        raise ValueError('unknown alternative')
    spec = importlib.util.spec_from_file_location('tools._next_'+mode, ROOT/'tools/l1_rco_k1_all_v1.py')
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    model_id, seed, tag = MODES[mode]
    k.WORK = TASK/mode
    k.OUT, k.TAG, k.REVIEW = k.WORK/'fit', tag, k.WORK/'REVIEW-META-EPS.json'
    script = ROOT/('tools/l1_rco_k1_'+mode+'_v1.py')
    original_reg, original_source, original_argv = k.expected_registration, k.derive_trainer, k.trainer_argv
    original_pins, original_admit = k.code_pins, k.admit

    def pins():
        files = ['l1_rco_k1_next_v1.py', 'l1_rco_k1_nextsmoke_v1.py',
                 'l1_rco_k1_nextlaunch_v1.py', 'l1_rco_k1_nexthost_v1.py',
                 'l1_rco_k1_'+mode+'_v1.py', 'l1_k1'+mode+'_trainer_v1.py',
                 'l1_rco_k1_meta_admit_v1.py', 'l1_rco_k1_foldcache_v2.py']
        if mode == 'fullbag_all':
            files += ['l1_rco_k1_fullbag_v1.py', 'l1_rco_k1_fullbag_replay_v1.py',
                      'l1_rco_k1_fullbag_check_v1.py']
        terminal = '.local/scratch/astra-b2/fullbag-post-v1/fullbag_terminal_v1.py'
        return dict(original_pins(), **{'tools/'+n:k.sha(ROOT/'tools'/n) for n in files},
                    **{terminal:k.sha(ROOT/terminal)})

    def registration():
        reg = original_reg()
        reg.update(model_id=model_id, model_seed=seed, variant='fullbag' if seed == 0 else 'standard',
                   storage_note='Inherited s0 filenames are storage names; model_seed/seed_base bind actual RNG.')
        if mode == 'fullbag_all':
            from tools import l1_rco_k1_fullbag_v1 as f
            parent = f.expected_registration()
            for field in ('windows', 'train_cap_policy', 'encoder_microbatch', 'replay_boundary', 'optimizer_steps_per_batch'):
                reg['recipe'][field] = parent['recipe'][field]
            reg.update(registered_parent=dict(tag=parent['tag'], contract_id=parent['contract_id']),
                       registered_change='Training roster: all 4167 labelled studies instead of fold0 training 3190; no holdout.')
            reg['recipe']['unchanged'] = 'Fullbag seed0 objective/replay/cap640/hyperparameters; steps follow all-data roster.'
            reg['projection'].update(fit_hours_training_lower_bound=20830*2.3/3600,
                fit_hours_calendar_estimate=20830*(18875.14643812389/7723)/3600,
                fit_hours_training_range=[20830*2.3/3600,20830*2.6/3600],
                scope='Fold0 production: 7723 steps / 18875.1464 train seconds, 172.045 windows/s; 2.3-2.6 s/step planning range. Excludes checkpoints and guard inference.')
        else:
            reg.update(registered_parent=dict(tag='H224_K1_V3G_ALL',
                model_sha256='47f47cf14615731de36c53735cafcd0ade9b07a0d48889c7d98ca3088fde66e0',
                contract_id='92cc6c854ce6b691be8aa1eb886068af9d8cfd1479f207ac270918e8a623621b'),
                registered_change='Model seed_base 0 -> 1; all-data recipe/roster unchanged.')
            reg['recipe'].update(seed_base=1, unchanged='Standard all-data v3-G recipe except model RNG seed 1.')
            reg['projection'].update(fit_hours_training_lower_bound=11597.55070862273/3600,
                fit_hours_calendar_estimate=11597.55070862273/3600,
                scope='Same 20830-step all-data production train_seconds=11597.55070862273; excludes checkpoints and guard inference; seed1 not measured.')
        reg['changes_vs_seed0'] = [reg['registered_change']]
        reg.pop('contract_id')
        reg['contract_id'] = k.digest(reg)
        return reg

    def derive(source):
        source = original_source(source)
        source = source.replace('tools.l1_rco_k1_all_v1', 'tools.l1_rco_k1_'+mode+'_v1')
        if mode == 'fullbag_all':
            from tools import l1_rco_k1_fullbag_v1 as f
            # Extract the exact reviewed replay block, rather than reimplement it.
            reference = f.derive_trainer((ROOT/k.s.SOURCE/'compact_train.py').read_text(encoding='utf-8'))
            start = reference.index('                    assert kind == "bce"')
            block = reference[start:reference.index('                    if a.forward_only:', start)]
            start = source.index('                    with torch.autocast("cuda"', source.index('for img, meta, mask, idx in l1_state.batches():'))
            end = source.index('                    if a.forward_only:', start)
            source = source[:start]+block+source[end:]
            old = '            cand = [cand[j] for j in np.sort(rng.choice(len(cand), self.n_win, replace=False))]'
            k.need(source.count(old)==1, 'fullbag cap source anchor')
            source = source.replace(old, '            raise ValueError("fullbag candidate cap exceeded; no subsampling")')
        else:
            old = 'assert a.seeds == 1 and a.seed_base == 0 and a.arms == "E_prior"'
            k.need(source.count(old)==1, 'seed assertion anchor')
            source = source.replace(old, old.replace('seed_base == 0', 'seed_base == 1'))
        compile(source, 'next-trainer', 'exec')
        return source

    def argv(cache, reg):
        cmd = original_argv(cache, reg)
        if mode == 'fullbag_all':
            cmd[cmd.index('--train-windows')+1] = '640'
        elif '--seed-base' in cmd:
            cmd[cmd.index('--seed-base')+1] = '1'
        else:
            cmd += ['--seed-base', '1']
        return cmd

    def command(fold='all', cache=None, review=None, gpu=0, action='run'):
        k.need(fold == 'all' and gpu == 0, 'all-data GPU0 only')
        return [sys.executable, str(script), action, '--cache', str(Path(cache or k.CACHE).resolve()),
                '--review', str(Path(review or k.REVIEW).resolve())]

    def evidence(cache):
        from tools import l1_rco_k1_meta_admit_v1 as meta
        proof = meta.parity.validate_receipt(cache)
        k.need(proof['mode'] == 'META_EPS', 'explicit META_EPS required')
        result = dict(meta_eps_bindings=meta.evidence_binding(cache))
        if mode == 'fullbag_all':
            from tools import l1_rco_k1_fullbag_v1 as f
            # Reuse only identical shape/replay evidence, with an explicit new review.
            memory = f.fullshape_evidence(f.expected_registration(), k.read(f.k.REVIEW))
            result['fullbag_reused_evidence'] = dict(parent_contract=memory['contract_id'],
                fullshape_sha256=k.sha(f.WORK/'fullshape-cuda.json'),
                numerical_cuda_sha256=f.numerical_binding('cuda'),
                replay_sha256=k.sha(ROOT/'tools/l1_rco_k1_fullbag_replay_v1.py'),
                reason='Same batch2, cap640, crop352, replay32 and architecture; roster does not increase tensor shape.',
                peak_reserved=memory['peak_reserved'])
        return result

    def admit(reg, cache, review, dry_run=True, owner_pid=None):
        try:
            binding = evidence(cache)
            r = k.read(review)
            k.need(all(r.get(n)==v for n,v in binding.items()), 'independent META_EPS/reused-shape review required')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if not dry_run:
                raise ValueError('next-job admission blocked: '+str(exc)) from exc
            result = dict(status='BLOCKED',blockers=[str(exc)],dry_run=True)
            k.write(k.WORK/'admit-dry-run.json',result)
            return result
        result = original_admit(reg,cache,review,dry_run,owner_pid)
        if not dry_run and mode == 'fullbag_all':
            k.need(result['desktop_bytes']+binding['fullbag_reused_evidence']['peak_reserved']+2*2**30 <= result['total_bytes'], 'current fullbag memory gate')
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
    if a.action == 'register':
        reg = k.register()
        result = dict(status='PASS',contract_id=reg['contract_id'],train=4167,holdout=0,steps=20830)
    else:
        reg = k.validate(k.read(k.WORK/'registration.json'))
        if a.action == 'prepare':
            p = k.prepare(reg,a.cache)
            result = dict(status='PASS',train=p['train_count'],trainer_sha256=p['trainer_sha256'])
        elif a.action == 'validate':
            from tools.l1_rco_k1_foldgate_v1 import prepared_hashes
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
            result=dict(bindings=k.review_binding(reg))
            try: result.update(k.next_evidence(a.cache),status='PASS')
            except (OSError,ValueError,KeyError,TypeError) as exc:
                result.update(status='BLOCKED',blockers=[str(exc)])
        elif a.action == 'run':
            from tools.l1_rco_k1_nextlaunch_v1 import require_selection
            k.need(os.name!='nt','Vast only')
            require_selection(k)
            result=k.run(a.cache,a.review)
        elif a.action == 'infer':
            k.require_infer_job(a.cache,a.review)
            result=k.inference(reg,a.cache)
        else: result=k.export(reg)
    print(json.dumps(result,allow_nan=False))
