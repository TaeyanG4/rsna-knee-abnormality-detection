"""H216 paired OAI-vs-control evaluator v1.0.0 (B on Claude Code, 2026-10-19). Never prints rows.

Reuses tools/l1_rco_k1_eval_v1.py unchanged: exact Q12b 976 primary cohort, the four
report rulers (jev_hybrid, aquila_v1, dread, consensus), macro-12 AUC and paired
delete-one-study jackknife. Decision rule is fixed in RULE before any arm output exists.

Actions:
  register   write REGISTRATION.json (rule + evaluator pins); refuses if arm outputs exist
  selfcheck  recompute the saved K1 v2 RCO-vs-R7 result through this wiring (must match exactly)
  evaluate   primary oai_native - rsna_control, plus descriptive secondary contrasts

Changelog: v1.0.0 first version.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import l1_rco_k1_eval_v1 as base
from tools.l1_rco_stage0_v1 import write

VERSION = '1.0.0'
OUT = ROOT / '.local/scratch/slot-b/h216-pair-eval-v1'
PAIR = ROOT / '.local/scratch/slot-b/h216-oai-pair-runtime-v101/a001'
SEED0 = ROOT / '.local/scratch/l1-rco/k1v3-grouping/fit/holdout.csv'
K1V2 = ROOT / '.local/scratch/l1-rco/k1v2'
RULE = dict(
    primary='oai_native minus rsna_control on exact Q12b976; macro-12 AUC per ruler; four rulers',
    useful='mean of the four ruler deltas > 0 AND >= 3 of 4 rulers positive AND no ruler delta < -0.002',
    otherwise='OAI_NOT_USEFUL: no OAI candidate; result kept as a finding',
    secondary='descriptive only: each arm minus original seed0 K1v3-G holdout; oai_native minus R7-f0 with the K1 KEEP rule',
    caveat='EXPOSED_EXPLORATORY report rulers; single fold/seed; not a private-LB forecast')


def ids977():
    import pandas as pd
    return sorted(pd.read_csv(SEED0, dtype={'StudyInstanceUID': str})['StudyInstanceUID'])


def load(path, ids):
    present, values = base.predictions(path, ids)
    if present != ids:
        raise ValueError('Exact 977 valid predictions required: ' + str(path))
    return values


def summary(m):
    return dict(decision=m['decision'], spearman_mean=m['spearman_mean'], rulers_passed=m['rulers_passed'],
                rulers={k: dict(a=v['rco'], b=v['r7'], delta=v['delta'], p_pos=v['p_delta_positive_normal'],
                                ci95=v['ci95']) for k, v in m['rulers'].items()})


def register():
    if (PAIR / 'rsna_control' / 'holdout.csv').exists() or (PAIR / 'oai_native' / 'holdout.csv').exists():
        raise ValueError('Arm outputs already exist; registration must precede results')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'REGISTRATION.json').exists():
        raise ValueError('Already registered')
    ids = ids977()
    spec = base.evaluator_registration(ids)
    pins = dict(spec['pins'], **{str(Path(__file__)): base.sha(Path(__file__)), str(SEED0): base.sha(SEED0)})
    reg = dict(version=VERSION, rule=RULE, primary_n=976, primary_id_list_sha256=spec['primary']['id_list_sha256'],
               pins=pins, base_rule=base.RULE, base_uncertainty=base.UNCERTAINTY)
    write(OUT / 'REGISTRATION.json', reg)
    return dict(status='REGISTERED', primary_n=976, pins=len(pins))


def context():
    ids = ids977()
    spec = base.evaluator_registration(ids)
    primary = spec['primary']['ids']
    refs = base.rulers.references(primary)
    available, r7 = base.predictions(spec['teacher'], ids, teacher=True)
    if available != primary:
        raise ValueError('R7 primary intersection changed')
    take = [ids.index(u) for u in primary]
    return ids, primary, refs, r7, take


def selfcheck():
    ids, primary, refs, r7, take = context()
    k1 = load(K1V2 / 'fit' / 'holdout.csv', ids)
    got = base.metrics(k1[take], r7, refs)
    saved = json.loads((K1V2 / 'evaluate' / 'result.json').read_text(encoding='utf-8'))
    worst = max(abs(got['rulers'][k]['delta'] - saved['rulers'][k]['delta']) for k in got['rulers'])
    ok = worst == 0.0 and got['decision'] == saved['decision'] and got['spearman_mean'] == saved['spearman_mean']
    result = dict(status='SELFCHECK_PASS' if ok else 'SELFCHECK_FAIL', max_abs_ruler_delta_diff=worst,
                  decision=got['decision'], saved_decision=saved['decision'])
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'SELFCHECK.json', result)
    return result


def evaluate():
    import numpy as np
    reg = json.loads((OUT / 'REGISTRATION.json').read_text(encoding='utf-8'))
    if reg['rule'] != RULE or reg['version'] != VERSION:
        raise ValueError('Registered rule differs')
    for p, h in reg['pins'].items():
        if base.sha(p) != h:
            raise ValueError('Registered evaluator input changed: ' + p)
    ids, primary, refs, r7, take = context()
    arms = {a: load(PAIR / a / 'holdout.csv', ids) for a in ('rsna_control', 'oai_native')}
    seed0 = load(SEED0, ids)
    prim = base.metrics(arms['oai_native'][take], arms['rsna_control'][take], refs)
    deltas = [v['delta'] for v in prim['rulers'].values()]
    useful = (float(np.mean(deltas)) > 0 and sum(d > 0 for d in deltas) >= 3 and min(deltas) >= -0.002)
    result = dict(version=VERSION, rule=RULE, n=976, exposure='EXPOSED_EXPLORATORY',
                  verdict='OAI_USEFUL' if useful else 'OAI_NOT_USEFUL', mean_ruler_delta=float(np.mean(deltas)),
                  primary=summary(prim),
                  secondary=dict(
                      control_minus_seed0=summary(base.metrics(arms['rsna_control'][take], seed0[take], refs)),
                      oai_minus_seed0=summary(base.metrics(arms['oai_native'][take], seed0[take], refs)),
                      oai_vs_r7_keep_rule=summary(base.metrics(arms['oai_native'][take], r7, refs))),
                  inputs={a: base.sha(PAIR / a / 'holdout.csv') for a in arms})
    result = base.clean(result)
    write(OUT / 'RESULT.json', result)
    import csv
    with (OUT / 'per-target-primary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['ruler', 'target', 'n', 'rco_auc', 'r7_auc', 'delta', 'spearman'])
        w.writeheader(); w.writerows(prim['per_target'])
    return dict(verdict=result['verdict'], mean_ruler_delta=result['mean_ruler_delta'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['register', 'selfcheck', 'evaluate'])
    a = ap.parse_args()
    print(json.dumps({'register': register, 'selfcheck': selfcheck, 'evaluate': evaluate}[a.action]()))


if __name__ == '__main__':
    main()
