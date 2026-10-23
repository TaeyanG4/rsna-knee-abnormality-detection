"""CPU-only honest OOF ensemble ablations, v1.0.0.

Changelog 2026-10-11: create-only analysis; frozen roster/hash verification,
equal-member ranks, family/seed deletions, paired multi-lens readouts.
No training, model loading, release, registration or submission commands.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from tools import hb95_partner_weight as hb
from tools import multi_lens_readout as ml

VERSION = '1.0.0'
OUT = Path('artifacts/experiments/ENSEMBLE_ABLATION_V1')
SCREENS = {
    'c4': 'artifacts/experiments/EXP183_MULTISCALE_5FOLD/screen/result.json',
    'mil': 'artifacts/experiments/EXP186_FINDING_MIL_5FOLD/screen-b/result.json',
    'seeds': 'artifacts/experiments/EXP185_FINDING_MIL/screen/result.json',
}
ROSTER = 'artifacts/experiments/EXP149_LABEL_SOURCES/students/manifest.json'
MEMBERS = ('R50', 'DINO', 'C4', 'MIL')
WEIGHTS = (.15, .30, .50)
need = hb.need
rank = hb.rank


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def output_path(root, out):
    allowed = (root / OUT).resolve()
    out = Path(out).resolve()
    need(out == allowed or out.is_relative_to(allowed), 'Output outside analysis directory')
    need(not out.exists(), 'Create-only output already exists')
    return out


def relative(root, path):
    path = Path(str(path).replace('\\', '/'))
    path = path if path.is_absolute() else root / path
    need(path.resolve().is_relative_to(root.resolve()), 'Input outside project')
    return path, path.relative_to(root).as_posix()


def validate_summary(summary, roster, n, fold):
    """Roster digests, not occasionally stale display fold numbers, define OOF."""
    prov = summary['provenance']
    need(summary['train'] == 4343-n and summary['val'] == n,
         'Training/validation size differs; all-data models forbidden')
    train_hashes = {roster['train_sha256'], roster.get('sorted_train_sha256', roster['train_sha256'])}
    need(prov['train_ids_ordered_sha256'] in train_hashes and
         prov['eval_ids_ordered_sha256'] == roster['val_sha256'],
         'Train/validation roster digest differs')
    return dict(train=summary['train'], val=n, registered_fold=fold,
                display_fold=summary.get('fold'),
                display_fold_disagrees=summary.get('fold', fold) != fold,
                training_order='frozen' if prov['train_ids_ordered_sha256'] == roster['train_sha256'] else 'sorted_same_roster',
                train_sha256=prov['train_ids_ordered_sha256'],
                val_sha256=prov['eval_ids_ordered_sha256'])


def load_inputs(root):
    bindings = {}
    def bound(rel):
        p, rel = relative(root, rel)
        bindings[rel] = hb.sha(p)
        return read(p)
    screens = {k: bound(p) for k, p in SCREENS.items()}
    manifest = bound(ROSTER)
    by_fold, rosters = {}, {}
    for f in range(5):
        info = manifest['folds'][str(f)]['ordered_ids']
        r = bound(info['path'])
        need(bindings[info['path']] == info['sha256'], 'Frozen roster changed')
        need(not set(r['train']) & set(r['val']), 'Train/validation overlap')
        need(hashlib.sha256('\n'.join(r['train']).encode()).hexdigest() == info['train_sha256'] and
             hashlib.sha256('\n'.join(r['val']).encode()).hexdigest() == info['val_sha256'],
             'Frozen roster content digest differs')
        by_fold[f] = r['val']; rosters[f] = dict(info,
            sorted_train_sha256=hashlib.sha256('\n'.join(sorted(r['train'])).encode()).hexdigest())
    all_ids = sum(by_fold.values(), [])
    need(len(all_ids) == len(set(all_ids)) == 4343 and len(by_fold[0]) == 888,
         'Five-fold roster differs')
    for f in range(5):
        r = read(root / rosters[f]['path'])
        need(set(r['train']) == set(all_ids)-set(by_fold[f]), 'Non-complement training roster')
    raw = {f: {} for f in range(5)}
    inputs = {}
    def load(key, entry, f):
        p, rel = relative(root, entry['path'])
        digest = hb.sha(p)
        need(digest == entry['sha256'], 'Prediction receipt mismatch: '+key)
        summary = bound(p.with_name('summary.json'))
        audit = validate_summary(summary, rosters[f], len(by_fold[f]), f)
        raw[f][key] = hb.load_oof(p, by_fold[f])
        need(hb.sha(p) == digest, 'Prediction changed while loading')
        bindings[rel] = digest
        # Only explicitly named inference/evaluation durations; never training totals.
        timing = {k: v for k, v in summary.items() if
                  ('infer' in k.lower() or 'eval' in k.lower()) and
                  ('second' in k.lower() or 'timing' in k.lower())}
        inputs[f'{key}_f{f}'] = dict(path=rel, sha256=digest, **audit,
                                    recorded_inference_timing=timing or None)
    for f in range(5):
        c = screens['c4']['input_files']; m = screens['mil']['input_files']
        for key, prefix in [('R42','rad42'), ('R43','rad43'), ('DINO','dino'), ('C4','mixed')]:
            entry = c[f'{prefix}_f{f}']
            other = m[f'{"control" if key == "C4" else prefix}_f{f}']
            need(entry['sha256'] == other['sha256'], 'Independent screen bindings disagree')
            load(key, entry, f)
        load('M42', m[f'candidate_f{f}'], f)
    for key, arm, tag, prefix in [('M43','EXP185_FINDING_MIL','S43','candidate'),
                                  ('C43','EXP179_MULTISCALE','M43','control')]:
        name = 'EXP185_FINDING_MIL_F0_S43' if key == 'M43' else 'EXP179_M43_F0'
        load(key, dict(path=f'artifacts/experiments/{arm}/outputs/{tag}/results/{name}/oof_k36_flip_tta.csv',
                       sha256=screens['seeds']['input_sha256'][prefix+'_S43']), 0)
    anchor_path = root / hb.FILES['anchor']
    receipt = bound(anchor_path.with_name('oof-receipt.json'))
    need(receipt['arm'] == 'CONTROL' and receipt['fold'] == 0 and receipt['rows'] == 888 and
         receipt['cohort_sha256'] == rosters[0]['val_sha256'], 'Anchor receipt differs')
    digest = hb.sha(anchor_path)
    need(digest == receipt['outputs'][anchor_path.name], 'Anchor OOF hash differs')
    anchor_summary = bound(anchor_path.with_name('summary.json'))
    need(hb.sha(anchor_path.with_name('summary.json')) == receipt['summary_sha256'],
         'Anchor training summary changed')
    need(anchor_summary['identity']['train_n'] == 4343-888 and
         anchor_summary['identity']['protocol_sha256'] == receipt['protocol_sha256'],
         'Anchor training size/protocol differs')
    anchor = hb.load_oof(anchor_path, by_fold[0])
    bindings[hb.FILES['anchor']] = digest
    inputs['anchor'] = dict(path=hb.FILES['anchor'], sha256=digest, receipt=receipt)
    # Inventory only this explicitly requested S43 experiment; no model/checkpoint reads.
    s43_dir = root / 'artifacts/experiments/EXP187_MIL_MEMBERS_S43'
    available = sorted(p.relative_to(root).as_posix() for p in s43_dir.glob('**/oof*.csv'))
    need(not available, 'New EXP187 OOF exists: bind validated fold receipts before analysis')
    return raw, by_fold, anchor, dict(predictions=inputs, bindings=bindings,
        mil_s43=dict(available_folds=[0], missing_folds=[1,2,3,4],
                     searched_directory=str(s43_dir.relative_to(root)), collected_predictions=available))


def members(raw, folds, mil_seeds=('M42',), rad_seeds=('R42','R43'), c4_seeds=('C4',)):
    """Rank raw predictions within held-out folds; mean seeds; pooled member rerank."""
    specs = {'R50': rad_seeds, 'DINO': ('DINO',), 'C4': c4_seeds, 'MIL': mil_seeds}
    return {name: rank(np.concatenate([np.mean([rank(raw[f][s]) for s in seeds], axis=0)
                                     for f in folds])) for name, seeds in specs.items()}


def bundle(values, names=MEMBERS):
    need(bool(names), 'Empty bundle')
    return rank(np.mean([values[k] for k in names], axis=0))


def variants(raw, folds, mil_seeds=('M42',)):
    m = members(raw, folds, mil_seeds)
    result = {'full_Q': bundle(m)}
    for k in MEMBERS:
        result['minus_'+k] = bundle(m, tuple(v for v in MEMBERS if v != k))
        result['only_'+k] = m[k]
    result['minus_ConvNeXt_family'] = bundle(m, ('R50','DINO'))
    result['Q_RC'] = bundle(m, ('R50','C4'))
    result['Q_RCD'] = bundle(m, ('R50','C4','DINO'))
    for seeds in [('R42',), ('R43',)]:
        sm = members(raw, folds, mil_seeds, seeds)
        result['Q_'+seeds[0]+'_only'] = bundle(sm)
        result['only_'+seeds[0]] = sm['R50']
    if tuple(folds) == (0,):
        for seeds, tag in [(('M42',),'M42'), (('M43',),'M43'), (('M42','M43'),'M4243')]:
            sm = members(raw, folds, seeds)
            result['Q_'+tag] = bundle(sm); result['only_'+tag] = sm['MIL']
        for seeds, tag in [(('C43',),'C43'), (('C4','C43'),'C4243')]:
            sm = members(raw, folds, mil_seeds, c4_seeds=seeds)
            result['Q_'+tag] = bundle(sm); result['only_'+tag] = sm['C4']
    return result


def comparison(ctx, arm, control):
    result = ml.evaluate(ctx, {'composed': arm}, {'composed': control})
    # Label sources are fixed here, but Aquila remains training-label-aligned.
    result['aquila_interpretation'] = 'Training-label-aligned ruler; diagnostic, not independent validation'
    scores = ctx.score(arm)
    result['absolute'] = dict(weak_four=float(scores[0,:2][:,ml.WEAK].mean()),
        macro12=float(scores[0,:2].mean()), dread=float(scores[0,2].mean()))
    return result


def compact(scope, name, control, result):
    return dict(scope=scope, variant=name, control=control,
        weak_four=result['lenses']['L1']['value'], macro12=result['lenses']['L2']['value'],
        dread=result['harm_guard']['dread_macro']['value'], guard=result['harm_guard']['PASS'],
        weak_ci95=result['lenses']['L1']['ci95'], macro_ci95=result['lenses']['L2']['ci95'],
        absolute=result['absolute'])


def analyze(root, smoke=False):
    start = time.monotonic()
    raw, by_fold, anchor, provenance = load_inputs(root)
    scopes = [('pooled_5fold_M42',tuple(range(5)),('M42',))]
    scopes += [(f'fold{f}_M42',(f,),('M42',)) for f in range(5)]
    scopes += [('fold0_M4243',(0,),('M42','M43'))]
    if smoke:
        scopes = [scopes[0], scopes[1], scopes[-1]]
    results, table, context_cache, checks = {}, [], {}, []
    for scope, folds, seeds in scopes:
        ids = sum((by_fold[f] for f in folds), [])
        if folds not in context_cache:
            context_cache[folds] = ml.make_context(ids, root)
        ctx = context_cache[folds]
        vv = variants(raw, folds, seeds)
        # Real-data loader/rank parity with the original HB95 three-member formula.
        if folds == (0,) and seeds == ('M42',):
            prior = dict(anchor=anchor, R42=raw[0]['R42'], R43=raw[0]['R43'],
                         D172=raw[0]['DINO'], C4=raw[0]['C4'], MIL=raw[0]['M42'])
            np.testing.assert_array_equal(.85*rank(anchor)+.15*vv['Q_RCD'],
                                          hb.predictions(prior,np.arange(888))['C4'][0])
            checks.append('PASS real-data HB95 Q_RCD/proxy rank parity')
        rr = {}
        for name, p in vv.items():
            r = comparison(ctx, p, vv['full_Q'])
            rr[name] = r; table.append(compact(scope,name,'full_Q',r))
        contrasts = [('DINO_added_without_MIL','Q_RCD','Q_RC'),
                     ('DINO_added_with_MIL','full_Q','minus_DINO'),
                     ('R50_alone_drop_S43','only_R42','only_R50'),
                     ('R50_alone_drop_S42','only_R43','only_R50')]
        if folds == (0,):
            contrasts += [('MIL_add_S43','Q_M4243','Q_M42'),
                          ('MIL_alone_add_S43','only_M4243','only_M42'),
                          ('MIL_S43_vs_S42','Q_M43','Q_M42'),
                          ('MIL_alone_S43_vs_S42','only_M43','only_M42'),
                          ('C4_add_S43','Q_C4243','full_Q')]
        for name, a, c in contrasts:
            r = comparison(ctx,vv[a],vv[c]); rr[name] = r
            table.append(compact(scope,name,c,r))
        results[scope] = dict(rows=len(ids), folds=folds, mil_seeds=seeds,
            ordered_roster_sha256=hb.digest_array(np.asarray(ids,dtype='S')),
            draw_weights_sha256=hb.digest_array(ctx.w.astype('<i8')),
            context_bindings=ctx.bindings, comparisons=rr)
        if folds == (0,):
            for w in WEIGHTS:
                proxy_scope = scope+f'_proxy_w{w:.2f}'
                pp = {k:(1-w)*rank(anchor)+w*rank(p) for k,p in vv.items()}
                pr = {}
                for name,p in pp.items():
                    r = comparison(ctx,p,pp['full_Q']); pr[name]=r
                    table.append(compact(proxy_scope,name,'full_Q',r))
                for name,a,c in contrasts:
                    r = comparison(ctx,pp[a],pp[c]); pr[name]=r
                    table.append(compact(proxy_scope,name,c,r))
                # Absolute utility of the partner bundle beside the anchor.
                r = comparison(ctx,pp['full_Q'],rank(anchor)); pr['full_Q_vs_anchor']=r
                table.append(compact(proxy_scope,'full_Q_vs_anchor','anchor',r))
                results[proxy_scope] = dict(rows=888, weight=w, comparisons=pr)
        print(f'PASS {scope}: {len(rr)} comparisons; 300 shared component draws',flush=True)
    for rel,digest in provenance['bindings'].items():
        need(hb.sha(root/rel)==digest, 'Input changed during analysis: '+rel)
    for ctx in context_cache.values():
        for rel,digest in ctx.bindings.items():
            need(hb.sha(root/rel)==digest, 'Evaluation input changed during analysis')
    return ml.clean(dict(version=VERSION, bootstrap=dict(draws=300,seed=20261003,
        unit='validation connected component', ranks='fixed before paired bootstrap'),
        policy='Within-fold raw ranks; equal seed mean; pooled member rerank; equal member mean; Q rerank. No fitted weights.',
        limitations=['Exploratory post-hoc comparisons; no multiplicity adjustment or equivalence claim.',
            'Weak-label held-out OOF; report-based rulers, not independent image-read truth.',
            'MIL S43 folds 1-4 unavailable: no pooled two-seed estimate or cross-fold imputation.',
            'Single CoAtNet fold-0 proxy is not a validated final multi-anchor release.',
            'C4 and MIL are distinct members of the same ConvNeXt architecture family.'],
        provenance=provenance, results=results, table=table, checks=checks,
        elapsed_seconds=round(time.monotonic()-start,3)))


def report(result):
    lines = ['# Ensemble ablation v'+VERSION, '',
        'Deltas versus full Q in the same scope (weak four and macro = mean JEV/consensus).',
        'Full Q = equal R50, DINO, C4, MIL weights; R50 uses two seeds; five-fold MIL uses S42.',
        '300 paired component draws, seed 20261003. Detailed per-target/ruler CIs: result.json.', '',
        '| Scope | Variant | Weak four | Macro-12 | Dread | Guard |',
        '|---|---|---:|---:|---:|---|']
    for r in result['table']:
        if r['scope']=='pooled_5fold_M42' or (r['scope'].endswith('_proxy_w0.30') and
                r['variant'] in ('full_Q','minus_DINO','minus_C4','minus_MIL','minus_R50','minus_ConvNeXt_family','MIL_add_S43')):
            def fmt(k): return 'undefined' if r[k] is None else f'{r[k]:+.6f}'
            lines.append('| '+' | '.join([r['scope'],r['variant'],fmt('weak_four'),fmt('macro12'),fmt('dread'),str(r['guard'])])+' |')
    lines += ['', 'All variants, all five individual folds, and proxy weights 0.15/0.30/0.50: table.csv.',
        'Rows named DINO_added_without_MIL and *_add_S43 use the explicit control in table.csv, not full Q.', '',
        'MIL S43 has fold-0 OOF only (888 rows); five-fold S42+S43 is unavailable. No substituted/in-sample predictions.',
        'Inference seconds per partner are not recorded in the inspected bound summaries/screens. Training total_seconds is not inference cost.',
        'Dropping R50 removes 10 fold/seed checkpoints (or 5 for one seed); DINO 5; C4 5; MIL S42 5.',
        'These checkpoint counts describe serving work avoided, not measured wall-time speedups.', '',
        *result['limitations'], '',
        'OOF honesty: prediction SHA256 and exact held-out roster verified; training/validation roster hashes match frozen splits.',
        'Remote training may sort the same training complement; both exact ordered hashes are verified, never arbitrary hash exceptions.',
        'DINO summaries contain stale display fold numbers; the training/evaluation roster hashes are authoritative and recorded.',
        'No original project file changed. Source/input hashes are in result.json; artifact hashes in SHA256SUMS.json.', '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--smoke',action='store_true',help='Real-data pooled/fold-0 branches; full 300 draws')
    ap.add_argument('--version',action='version',version=VERSION)
    args=ap.parse_args(argv); root=args.root.resolve()
    out=output_path(root,args.out or root/OUT/('smoke' if args.smoke else 'run'))
    source_files = ['tools/ensemble_ablation_v1.py','tools/hb95_partner_weight.py',
                    'tools/multi_lens_readout.py','tools/exp169_bundle_screen.py',
                    'tools/exp156_k_sweep.py','tools/exp125_radr50_k16_local.py']
    source_hashes={p:hb.sha(root/p) for p in source_files}
    result=analyze(root,args.smoke)
    need(source_hashes == {p:hb.sha(root/p) for p in source_files}, 'Source changed during analysis')
    result['source_sha256']=source_hashes
    out.mkdir(parents=True)
    ml.write(out/'result.json',result)
    with (out/'table.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['table'][0]))
        writer.writeheader();writer.writerows(result['table'])
    (out/'REPORT.md').write_text(report(result),encoding='utf-8')
    ml.write(out/'SHA256SUMS.json',{p.name:hb.sha(p) for p in sorted(out.iterdir())})
    print(json.dumps(dict(status='PASS',output=str(out),comparisons=len(result['table']),
                          seconds=result['elapsed_seconds'],sha256=hb.sha(out/'result.json'))))


if __name__=='__main__':
    main()
