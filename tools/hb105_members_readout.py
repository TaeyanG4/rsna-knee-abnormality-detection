"""HB105 member OOF/readout v1.0.0; create-only, CPU, no release changes.

Q7 is the D-B-131 equal-member R50/C4/MIL-T baseline. Uses the existing
ensemble_ablation_v1 rank/comparison and multi_lens_readout evaluation APIs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from tools import ensemble_ablation_v1 as ea, hb105_members as ops
from training.arms import lifecycle as life

PRIMARY = 'oof_k36_flip_tta.csv'
F0 = '.local/hb105-collected-v1'
T43 = '.local/refactor/mil-s43-members-collected-v1'
PIN = ops.EXP + '/readout-baseline-v1.json'


def collected(root, directory, spec, ids, bindings):
    directory = Path(directory)
    receipt, collection = [ops.read(directory / p) for p in ('receipt.json', 'collection.json')]
    ops.need(collection['status'] == 'COLLECTED_ARTIFACTS_VERIFIED'
             and collection['spec_sha256'] == spec.sha256, 'collection differs')
    life.verify_artifact_receipt(spec, directory/'outputs', receipt,
        expected_receipt_sha256=collection['receipt_sha256'])
    ops.need(receipt['metrics'].get('engineering_smoke', 0) == 0 and receipt['exit_code'] == 0,
             'scientific readout cannot use smoke')
    output = directory/'outputs'
    summary, prov = [ops.read(output/p) for p in ('summary.json','provenance.json')]
    c = spec.document['config']
    ops.need(summary['status'] == 'FIXED_FINAL_COMPLETE' and summary['epochs'] == 20
        and summary['attempted_updates'] == c['updates']['steps']
        and summary['skipped_updates'] <= c['updates']['max_skips'], 'unfinished fit')
    ops.need(summary['spec_sha256'] == prov['spec_sha256'] == spec.sha256
        and prov['verification_sha256'] == receipt['verification_sha256']
        and prov['checkpoint_selection'] == 'fixed-final', 'fit provenance differs')
    ops.need(prov['train_ids_ordered_sha256'] == c['roots']['train_sha256']
        and prov['eval_ids_ordered_sha256'] == c['roots']['val_sha256']
        and (summary['train'],summary['val']) == (4343-len(ids),len(ids)), 'OOF roster differs')
    roster = ops.read(root/c['roots']['ordered'])
    ops.need(roster['val'] == ids and not set(roster['train']) & set(ids), 'held-out roster differs')
    for p in (directory/'receipt.json', directory/'collection.json', output/PRIMARY,
              output/'summary.json', output/'provenance.json'):
        bindings[p.relative_to(root).as_posix()] = ops.sha(p)
    return ea.hb.load_oof(output/PRIMARY, ids)


def baseline(root):
    raw, by_fold, _, provenance = ea.load_inputs(root)
    bindings = provenance['bindings']
    for f in range(1,5):
        path = root/T43/f'f{f}-spec.json'
        spec = life.ArmSpec(path.read_bytes())
        ops.need(spec.sha256 == life.ArmSpec((root/f'training/configs/mil-member-s43-f{f}-v1.json').read_bytes()).sha256,
                 'MIL-T S43 spec differs')
        bindings[path.relative_to(root).as_posix()] = ops.sha(path)
        raw[f]['M43'] = collected(root, root/T43/f'fold{f}', spec, by_fold[f], bindings)
    for s in (42,43):
        path = root/f'training/configs/hb105-small-s{s}-f0-v1.json'
        spec = life.ArmSpec(path.read_bytes())
        ops.need(spec.sha256 == ops.mil.RECIPE_HASHES[s], 'F0 ancestor differs')
        bindings[path.relative_to(root).as_posix()] = ops.sha(path)
        raw[0][f'S{s}'] = collected(root,root/F0/f's{s}',spec,by_fold[0],bindings)
    return raw, by_fold, bindings


def family(raw, folds, keys):
    return ea.rank(np.concatenate([np.mean([ea.rank(raw[f][k]) for k in keys],axis=0) for f in folds]))


def variants(raw, folds):
    m = ea.members(raw, folds, mil_seeds=('M42','M43'))
    m['SMALL'] = family(raw,folds,('S42','S43'))
    return dict(Q7=ea.bundle(m,('R50','C4','MIL')),
        Q7_add_S=ea.bundle(m,('R50','C4','MIL','SMALL')),
        Q7_replace_T_with_S=ea.bundle(m,('R50','C4','SMALL')),
        MIL_T=m['MIL'], MIL_S=m['SMALL'])


def analyze(root, collected_root, *, preflight=False):
    root = Path(root).resolve()
    raw, by_fold, bindings = baseline(root)
    pin = ops.read(root/PIN)
    ops.need(pin['baseline'] == 'D-B-131: R50(two seeds)+C4(S42)+MIL-T(two seeds)', 'Q7 policy differs')
    ops.need(bindings == pin['bindings'], 'frozen baseline changed')
    if not preflight:
        for s,f in ops.FITS:
            raw[f][f'S{s}'] = collected(root,Path(collected_root).resolve()/f's{s}-f{f}',
                                        ops.spec_for(root,s,f),by_fold[f],bindings)
    # Replay tests the pooled branch with real held-out Tiny predictions only.
    # It is clearly marked engineering and never exported as Small OOF.
    if preflight:
        for f in range(1,5):
            for s in (42,43): raw[f][f'S{s}'] = raw[f][f'M{s}']
    results = {}
    for name, folds in [('pooled_5fold',tuple(range(5))), *[(f'fold{f}',(f,)) for f in range(5)]]:
        ids = sum((by_fold[f] for f in folds), [])
        ctx = ea.ml.make_context(ids,root)
        vv = variants(raw,folds)
        r = {}
        for k in ('Q7_add_S','Q7_replace_T_with_S'):
            comparison = ea.comparison(ctx,vv[k],vv['Q7'])
            bundled = ea.ml.evaluate(ctx,{'composed':vv[k]},{'composed':vv['Q7']},
                bundle=dict(context=ctx,arm=vv[k],control=vv['Q7'],recipe=k+' versus frozen Q7'))
            comparison.update(bundled)
            r[k] = comparison
        r['MIL_S_vs_T'] = ea.ml.evaluate(ctx,
            {str(s):family(raw,folds,(f'S{s}',)) for s in (42,43)},
            {str(s):family(raw,folds,(f'M{s}',)) for s in (42,43)})
        results[name] = dict(rows=len(ids),folds=folds,comparisons=r,context_bindings=ctx.bindings)
        for p,h in ctx.bindings.items(): ops.need(ops.sha(root/p)==h,'ruler changed')
        print(f'PASS {name}: {len(ids)} held-out rows, 3 comparisons',flush=True)
    for p,h in bindings.items(): ops.need(ops.sha(root/p)==h,'input changed')
    sources = ('tools/hb105_members_readout.py','tools/ensemble_ablation_v1.py',
               'tools/hb95_partner_weight.py','tools/multi_lens_readout.py')
    return ea.ml.clean(dict(version='1.0.0',
        source_sha256={p:ops.sha(root/p) for p in sources},
        bootstrap=dict(draws=300,seed=20261003,unit='held-out connected component'),
        status='ENGINEERING_REPLAY_NOT_SMALL_5FOLD' if preflight else 'COMPLETE_OOF_READOUT',
        policy=pin['baseline'],primary=PRIMARY,results=results,bindings=bindings,
        rank_policy='Within-fold rank; equal seed mean; pooled family rerank; equal family mean; bundle rerank',
        limitations=['F0 reused as requested; exploratory held-out report-label OOF, not independent image truth.',
                      'L7 image-read evidence unavailable; no release or public-score claim.',
                      'Preflight substitutes Tiny for missing Small F1-F4 only when explicitly requested.']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['pin','preflight','analyze'])
    ap.add_argument('--root',type=Path,default=ops.ROOT)
    ap.add_argument('--collected',type=Path,default=Path('.local/hb105m-collected-v1'))
    ap.add_argument('--out',type=Path)
    a = ap.parse_args(); root=a.root.resolve()
    if a.command == 'pin':
        _,_,bindings=baseline(root)
        ops.save(root/PIN,dict(baseline='D-B-131: R50(two seeds)+C4(S42)+MIL-T(two seeds)',bindings=bindings))
        print('BASELINE_PINNED')
    else:
        ops.need(a.out is not None and not a.out.exists(),'new output required')
        result=analyze(root,a.collected,preflight=a.command=='preflight')
        ops.save(a.out,result)
        print(result['status'])


if __name__ == '__main__': main()
