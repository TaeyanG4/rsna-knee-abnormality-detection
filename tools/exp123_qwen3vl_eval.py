"""EXP123 (H-B-43) evaluation: score the VLM fold-0 predictions next to CoAtNet fold 0 and RadR50 fold 0, apply the pre-registered stop rule.

  python tools/exp123_qwen3vl_eval.py                               # artifacts/experiments/EXP123_QWEN3VL_PILOT_F0/oof_k24.csv
  python tools/exp123_qwen3vl_eval.py --pred PATH [--name NAME] [--out DIR] [--draws 300]

Rows: the 888 EXP099 fold-0 validation studies (G1 V888). References (all report-derived, sharing the report-text ceiling):
  dread       rint(float32 Dread soft label)                          (EXP099 contract)
  flight_v4   flight hybrid v4 soft label >= 0.5                        (tools/submission_screen.py rule)
  relabel_v1  relabel_v1 soft label >= 0.5
  consensus   D-B-035: Dread, Steven v4 blend, Pilkwang v2 and relabel_v1 binarised at >= 0.5; only (study, target) pairs where all four
              agree (and none is missing) are scored, with the agreed label; per-target AUC needs >= 2 positives and >= 2 negatives
              (.local/consensus-gold-20261007/consensus_oof.py)
AUC = average-tie rank Mann-Whitney; macro = nanmean over the 12 targets. Deltas get a component bootstrap (VALIDATION_GROUP_CONTRACT_V1
components, seed 20261003, 300 draws as D-B-035). Ensembles are per-target means of average-tie percentile ranks over the 888 rows.
Stop rule (plan H-B-43, pre-registered): VLM consensus macro more than 0.01 below CoAtNet fold 0 (EXP096_K_SWEEP/B_K36.csv) -> STOP,
decided on the point estimate. The VLM K36 file (if any) is a diagnostic and never enters the decision. Only reads files; writes result.json and
result.md to a new directory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp123_qwen3vl_pilot as g  # noqa: E402

UID = 'StudyInstanceUID'
T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
SCHEMA = 'exp123-qwen3vl-eval-v1'
STOP_MARGIN = -0.01
DEFAULT_PRED = f'artifacts/experiments/{g.TAG}/oof_k24.csv'
DEFAULT_OUT = 'artifacts/experiments/EXP123_QWEN3VL_PILOT_EVAL'
COMPONENTS = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
LABELS = {
    'dread': ('storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv', 'rint'),
    'flight_v4': ('storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv', 'ge0.5'),
    'relabel_v1': ('storage/labels/derived_relabel/relabel_v1.csv', 'ge0.5'),
}
CONSENSUS_SETS = {  # D-B-035 / consensus_oof.py
    'dread': 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv',
    'steven_v4': 'storage/labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv',
    'pilkwang_v2': 'storage/labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv',
    'relabel_v1': 'storage/labels/derived_relabel/relabel_v1.csv',
}
REFERENCES = ('consensus', 'dread', 'flight_v4', 'relabel_v1')


# ------------------------------------------------------------------ loading
def load_soft(path, ids):
    """(n, 12) float soft labels for ids (NaN where the study or value is missing); accepts plain or *_soft columns."""
    d = pd.read_csv(path, dtype={UID: str}).drop_duplicates(UID).set_index(UID)
    cols = [t if t in d.columns else t + '_soft' for t in T]
    miss = [c for c in cols if c not in d.columns]
    if miss:
        raise ValueError(f'{path}: missing columns {miss}')
    out = d.reindex(list(ids))[cols].to_numpy(dtype=np.float64)
    return out


def load_pred(path, ids, exact=True):
    d = pd.read_csv(path, dtype={UID: str})
    if list(d.columns) != [UID, *T]:
        raise ValueError(f'{path}: columns must be {UID} + the 12 targets in order')
    if d[UID].duplicated().any():
        raise ValueError(f'{path}: duplicate UIDs')
    d = d.set_index(UID)
    if exact and set(d.index) != set(ids):
        raise ValueError(f'{path}: rows differ from the fold-0 validation ids ({len(set(d.index) & set(ids))}/{len(ids)} shared, {len(d)} rows)')
    missing = [u for u in ids if u not in d.index]
    if missing:
        raise ValueError(f'{path}: {len(missing)} fold-0 ids missing')
    P = d.loc[list(ids), T].to_numpy(dtype=np.float64)
    if not np.isfinite(P).all():
        raise ValueError(f'{path}: non-finite predictions')
    return P


def binarise(soft, rule):
    y = np.full(soft.shape, np.nan)
    ok = np.isfinite(soft)
    if rule == 'rint':
        y[ok] = np.rint(soft[ok].astype(np.float32))
    elif rule == 'ge0.5':
        y[ok] = (soft[ok] >= 0.5).astype(np.float64)
    else:
        raise ValueError(rule)
    return y


def references(ids, root=ROOT):
    """{name: (Y float 0/1, mask bool, min_class)}."""
    refs = {}
    for name, (path, rule) in LABELS.items():
        y = binarise(load_soft(Path(root) / path, ids), rule)
        refs[name] = (np.nan_to_num(y), np.isfinite(y), 1)
    stack = np.stack([binarise(load_soft(Path(root) / p, ids), 'ge0.5') for p in CONSENSUS_SETS.values()])
    valid = np.isfinite(stack).all(0)
    s = np.nan_to_num(stack).astype(bool)
    agree = valid & (s.all(0) | (~s).all(0))
    refs['consensus'] = (s[0].astype(np.float64), agree, 2)
    return {k: refs[k] for k in REFERENCES}


def component_groups(ids, root=ROOT):
    comp = {}
    for c in json.loads((Path(root) / COMPONENTS).read_text(encoding='utf-8')):
        for m in c['members']:
            comp[m] = c['component_id']
    groups = {}
    for i, u in enumerate(ids):
        groups.setdefault(comp.get(u, 'singleton:' + u), []).append(i)
    return [np.asarray(groups[k], dtype=np.int64) for k in sorted(groups)]


# ------------------------------------------------------------------ metrics
def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def auc_targets(Y, S, M, min_class):
    out = np.full(Y.shape[1], np.nan)
    for t in range(Y.shape[1]):
        sel = M[:, t]
        y, s = Y[sel, t], S[sel, t]
        P = y.sum(); N = len(y) - P
        if P < min_class or N < min_class:
            continue
        r = _rank(s)
        out[t] = (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)
    return out


def macro(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float('nan')


def pct_rank(S):
    from scipy.stats import rankdata
    return np.column_stack([rankdata(S[:, t]) / len(S) for t in range(S.shape[1])])


def score(P, ref):
    Y, M, mc = ref
    a = auc_targets(Y, P, M, mc)
    return dict(macro=macro(a), targets={T[t]: float(a[t]) for t in range(len(T))})


def compare(A, B, ref, draws_idx):
    """Candidate A minus baseline B on one reference: point, component-bootstrap CI and per-target deltas."""
    Y, M, mc = ref
    a, b = auc_targets(Y, A, M, mc), auc_targets(Y, B, M, mc)
    out = dict(candidate=macro(a), baseline=macro(b), delta=macro(a) - macro(b), targets={T[t]: float(a[t] - b[t]) for t in range(len(T))})
    if np.array_equal(A, B):
        out['ci95'] = [0.0, 0.0]
    else:
        ds = [macro(auc_targets(Y[ii], A[ii], M[ii], mc)) - macro(auc_targets(Y[ii], B[ii], M[ii], mc)) for ii in draws_idx]
        out['ci95'] = [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
    return out


def bootstrap_index(groups, draws, seed):
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for _ in range(draws)]


def spearman_by_target(A, B):
    ra, rb = pct_rank(A), pct_rank(B)
    return {T[t]: float(np.corrcoef(ra[:, t], rb[:, t])[0, 1]) for t in range(len(T))}


def evaluate(pred_path, root=ROOT, draws=300, seed=20261003, name='VLM', extra_preds=None):
    root = Path(root)
    _, val = g.fold_ids(root)
    if len(val) != 888:
        raise ValueError('fold-0 validation set is not V888')
    models = {name: load_pred(pred_path, val)}
    models['CoAtNet_f0_K36'] = load_pred(root / g.COATNET_F0_K36, val)
    models['RadR50_f0_K36_T4'] = load_pred(root / g.RADR50_OOF_K36, val, exact=False)  # fold-0 rows of the EXP107 5-fold OOF
    if (root / g.RADR50_LOCAL_F0_K36).is_file():
        models['RadR50_f0_K36_local'] = load_pred(root / g.RADR50_LOCAL_F0_K36, val)
    for k, p in (extra_preds or {}).items():
        models[k] = load_pred(p, val)
    refs = references(val, root)
    idx = bootstrap_index(component_groups(val, root), draws, seed)
    coverage = {k: dict(pairs=int(v[1].sum()), fraction=float(v[1].mean()), positives={T[t]: int(v[0][v[1][:, t], t].sum()) for t in range(len(T))})
                for k, v in refs.items()}
    scores = {m: {r: score(P, refs[r]) for r in REFERENCES} for m, P in models.items()}
    V, C = models[name], models['CoAtNet_f0_K36']
    radr = 'RadR50_f0_K36_local' if 'RadR50_f0_K36_local' in models else 'RadR50_f0_K36_T4'
    R = models[radr]
    rc, rv, rr = pct_rank(C), pct_rank(V), pct_rank(R)
    contrasts = {
        f'{name} - CoAtNet_f0_K36': (V, C),
        f'{name} - {radr}': (V, R),
        f'rank(CoAtNet+{name}) - CoAtNet_f0_K36': ((rc + rv) / 2, C),
        f'rank(CoAtNet+{radr}+{name}) - rank(CoAtNet+{radr})': ((rc + rr + rv) / 3, (rc + rr) / 2),
    }
    deltas = {c: {r: compare(A, B, refs[r], idx) for r in REFERENCES} for c, (A, B) in contrasts.items()}
    primary = deltas[f'{name} - CoAtNet_f0_K36']['consensus']
    stop = primary['delta'] < STOP_MARGIN
    decision = dict(rule=f'consensus macro({name}) - consensus macro(CoAtNet fold 0 K36) < {STOP_MARGIN} -> STOP (point estimate)',
                    delta=primary['delta'], ci95=primary['ci95'], status='STOP' if stop else 'CONTINUE',
                    note='other H-B-43 stops (projected T4 serving > 3 h; fp16 non-finite -> serve fp32) are decided by tools/exp123_fp16_probe.py')
    return dict(schema=SCHEMA, plan_item='H-B-43', observed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), prediction=str(pred_path),
                prediction_sha256=g.sha256(pred_path), rows=len(val), eval_ids_sha256=g.uid_hash(val), models=list(models),
                comparator_files={'CoAtNet_f0_K36': g.COATNET_F0_K36, 'RadR50_f0_K36_T4': g.RADR50_OOF_K36 + ' (fold-0 rows)',
                                  'RadR50_f0_K36_local': g.RADR50_LOCAL_F0_K36 if 'RadR50_f0_K36_local' in models else None},
                references={r: dict(rule=(LABELS[r][1] if r in LABELS else 'all four >= 0.5 agree'),
                                    files=([LABELS[r][0]] if r in LABELS else list(CONSENSUS_SETS.values())), **coverage[r]) for r in REFERENCES},
                bootstrap=dict(unit='component (VALIDATION_GROUP_CONTRACT_V1)', draws=draws, seed=seed),
                scores=scores, deltas=deltas, spearman_vs_coatnet=spearman_by_target(V, C), decision=decision,
                caveats=['one fold, one seed; V888 is exposed G1 validation', 'all references are report-derived (shared report-text ceiling); '
                         'consensus pairs cannot see disputed pairs or shared report silences (Synovitis)', 'RadR50 T4 fold 0 was trained on Kaggle T4 '
                         '(independent trajectory, D-B-008); the local EXP113 F0 replaces it when present'])


def render_md(r):
    name = r['models'][0]
    lines = [f"# EXP123 VLM fold-0 evaluation ({name})", '', f"prediction `{r['prediction']}` ({r['rows']} rows); bootstrap {r['bootstrap']['draws']} "
             f"component draws, seed {r['bootstrap']['seed']}", '', f"**Decision: {r['decision']['status']}** — {r['decision']['rule']}; "
             f"delta {r['decision']['delta']:+.4f} [{r['decision']['ci95'][0]:+.4f}, {r['decision']['ci95'][1]:+.4f}]", '',
             '| model | ' + ' | '.join(REFERENCES) + ' |', '|---|' + '---:|' * len(REFERENCES)]
    for m, s in r['scores'].items():
        lines.append(f'| {m} | ' + ' | '.join(f"{s[x]['macro']:.4f}" for x in REFERENCES) + ' |')
    lines += ['', '| contrast | ' + ' | '.join(REFERENCES) + ' |', '|---|' + '---|' * len(REFERENCES)]
    for c, d in r['deltas'].items():
        lines.append(f'| {c} | ' + ' | '.join(f"{d[x]['delta']:+.4f} [{d[x]['ci95'][0]:+.4f}, {d[x]['ci95'][1]:+.4f}]" for x in REFERENCES) + ' |')
    lines += ['', 'Consensus coverage: ' + ', '.join(f"{k} {v['pairs']} pairs ({v['fraction']:.2f})" for k, v in r['references'].items()), '',
              '| target | consensus ' + name + ' | consensus CoAtNet | Spearman vs CoAtNet |', '|---|---:|---:|---:|']
    for t in T:
        lines.append(f"| {t} | {r['scores'][name]['consensus']['targets'][t]:.4f} | {r['scores']['CoAtNet_f0_K36']['consensus']['targets'][t]:.4f} | "
                     f"{r['spearman_vs_coatnet'][t]:.3f} |")
    lines += ['', 'Caveats: ' + '; '.join(r['caveats']), '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', default=str(ROOT / DEFAULT_PRED))
    ap.add_argument('--name', default='VLM_K24')
    ap.add_argument('--extra', action='append', default=[], help='NAME=PATH diagnostic predictions (e.g. VLM_K36=.../oof_k36.csv)')
    ap.add_argument('--out', default=str(ROOT / DEFAULT_OUT))
    ap.add_argument('--draws', type=int, default=300)
    ap.add_argument('--seed', type=int, default=20261003)
    ap.add_argument('--no-write', action='store_true')
    a = ap.parse_args(argv)
    extra = dict(x.split('=', 1) for x in a.extra)
    if not extra and Path(a.pred).name == 'oof_k24.csv' and (Path(a.pred).parent / 'oof_k36.csv').is_file():
        extra['VLM_K36_diagnostic'] = str(Path(a.pred).parent / 'oof_k36.csv')
    r = evaluate(Path(a.pred), draws=a.draws, seed=a.seed, name=a.name, extra_preds=extra)
    if not a.no_write:
        out = Path(a.out)
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'output exists: {out}')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'result.json').write_text(json.dumps(r, indent=1, allow_nan=True), encoding='utf-8')
        (out / 'result.md').write_text(render_md(r), encoding='utf-8')
    print(json.dumps(dict(decision=r['decision'], macro={m: {x: round(s[x]['macro'], 4) for x in REFERENCES} for m, s in r['scores'].items()}), indent=1))
    return r


if __name__ == '__main__':
    main()
