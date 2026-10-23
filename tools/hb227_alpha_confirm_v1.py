"""H-B-227 v1.0.0: one frozen CPU comparison; aggregates only, no tuning.

976: stride-2 x3, optimistic R7-f0 parent-slot proxy, exact four rulers.
Gold58: existing authenticated INF3/K1-v2 helper (not x3/stride-2 evidence).
Changelog: register vector v2 versus flat .20 before a single evaluation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from tools import h216_pair_eval_v1 as context
from tools import l1_rco_k1_eval_v1 as metric
from tools import c85r7_rco_overlay_v1 as util

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.local/scratch/astra-b5/hb227-alpha-vector-v2'
WEIGHTS = {'ACL': .1, 'MCL': .2, 'Medial Meniscus': .2, 'Lateral Meniscus': .5,
           'Medial OA': .35, 'Lateral OA': .5, 'PF OA': .35, 'Effusion': .35,
           'Synovitis': .5, "Baker's": .2, 'Contusion': .15, 'Fracture': .35}
VECTOR = [WEIGHTS[t] for t in util.LABELS]
PREDS = ROOT / '.local/scratch/slot-b/k1-stride-screen/preds'
SCOUT = ROOT / '.local/scratch/slot-b/idea-scout-1020/fable/scout_gold_v1.py'


def mix(parent, k1, alpha):
    return util.ranks((1 - np.asarray(alpha)) * util.ranks(parent) +
                      np.asarray(alpha) * util.ranks(k1)).astype(np.float32)


def load976():
    ids, primary, refs, r7, take = context.context()
    arrays = [context.load(PREDS / (m + '_s2.csv'), ids)[take].astype(np.float32)
              for m in ('v2', 's0', 's1')]
    assert len(primary) == 976 and tuple(metric.TARGETS) == tuple(util.LABELS)
    k1 = np.mean([util.ranks(a) for a in arrays], axis=0)
    return primary, refs, r7.astype(np.float32), arrays, k1


def load_gold():
    # Execute only the existing authenticated input/formula helper, before its
    # historical candidate sweep or any print. No new shares are searched.
    source = SCOUT.read_text(encoding='utf-8').split('K20 = mix(INF3, a[\'RCO\'], .20)')[0]
    assert "print(" not in source
    scope = {'__file__': str(SCOUT)}
    exec(compile(source, str(SCOUT), 'exec'), scope)
    return scope


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pins = {str(p.relative_to(ROOT)): util.sha(p) for p in
            [Path(__file__), SCOUT, Path(metric.__file__), Path(context.__file__)] +
            [PREDS / (m + '_s2.csv') for m in ('v2', 's0', 's1')]}
    registration = dict(version='1.0.0', targets=util.LABELS, alpha_vector=VECTOR,
                        baseline_alpha=.20, pins=pins, tuning=False,
                        comparison='vector v2 minus flat .20; outer re-rank retained',
                        inference='cached CPU replay only')
    # An existing registration prevents silently repeating a settled experiment.
    with (OUT / 'confirmation-registration.json').open('x', encoding='utf-8') as f:
        json.dump(registration, f, indent=2)
    primary, refs, parent, arrays, k1 = load976()
    scored = metric.metrics(mix(parent, k1, VECTOR), mix(parent, k1, .20), refs)
    rulers = {k: dict(vector_auc=v['rco'], flat_auc=v['r7'], delta=v['delta'],
                     jackknife_se=v['jackknife_se'], ci95=v['ci95'],
                     p_positive_normal=v['p_delta_positive_normal'])
              for k, v in scored['rulers'].items()}
    g = load_gold()
    gold = g['gold']
    draws = gold.auc_table({'vector': mix(g['INF3'], g['a']['RCO'], VECTOR),
                            'flat': mix(g['INF3'], g['a']['RCO'], .20)}, g['yv'], g['w'])
    gold_result = {k: gold.block(v)['macro12'] for k, v in draws.items()}
    gold_result['delta'] = gold.block(draws['vector'] - draws['flat'])
    gold_result['bootstrap_draws'] = len(g['w']) - 1
    gold_result['n'] = len(g['ids'])
    gold_result['stride_screen_overlap'] = len(set(primary) & set(g['ids']))
    gold_result['k1_proxy'] = 'existing K1 v2 run1 only; not x3 or stride 2'
    result = dict(status='CPU_CONFIRMATION_COMPLETE_NOT_QUALIFIED',
                  exposure='EXPOSED_EXPLORATORY; not clean OOF',
                  n976=976, rulers=rulers, per_target976=scored['per_target'],
                  gold58=gold_result, registration=registration,
                  caveats=[
                      'R7-f0 is an optimistic proxy for the stronger full INF3 P slot.',
                      'Four report rulers are correlated, not four independent cohorts.',
                      'Gold58 was exposed to model/weight selection and is tiny.',
                      'Gold58 uses the existing single-v2 proxy, not x3 stride-2 inference.',
                      'Jackknife/bootstrap condition on the full-roster ranks; ranks are not refit per draw.',
                      'The vector was chosen using prior sweeps; this is confirmation, not untouched validation.'])
    result = metric.clean(result)
    (OUT / 'confirmation.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], rulers=rulers,
                         gold58_macro={k: gold_result[k] for k in ('vector', 'flat')},
                         gold58_delta=gold_result['delta']['macro12'],
                         gold58_overlap=gold_result['stride_screen_overlap']), indent=2))


if __name__ == '__main__':
    main()
