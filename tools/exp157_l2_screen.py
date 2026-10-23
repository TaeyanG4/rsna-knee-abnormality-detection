"""CPU-only, preregistered five-fold/two-seed EXP157 L2 screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = 'artifacts/experiments/EXP157_L2_ADVANCE/screen'


def control_tag(seed, fold):
    return f'EXP127_AUG1_S{seed}_F0' if fold == 0 else f'EXP134_AUG1_S{seed}_F{fold}'


def seed_mean(values):
    """Normalize each seed within its fold before averaging; preserve ties."""
    import numpy as np
    from tools.exp115_vlm_eval import pct_rank
    if len(values) != 2:
        raise ValueError('exactly seeds 42 and 43 required')
    a, b = [np.asarray(x, dtype=float) for x in values]
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 12 or not len(a):
        raise ValueError('invalid matched seed shapes')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite seed predictions')
    return (pct_rank(a) + pct_rank(b)) / 2


def decision(metrics):
    j = metrics['jev_hybrid']
    gates = {'jev_positive': j['delta'] > 0,
             'jev_ci_lower': j['ci95'][0] > -.001,
             'aquila_nonnegative': metrics['aquila_v1']['delta'] >= 0,
             'dread_floor': metrics['dread']['delta'] >= -.003}
    return {'status': 'ADOPT' if all(gates.values()) else 'HOLD', 'gates': gates}


def verify_controls(root):
    from tools import exp127_radr50_single as e127, exp134_aug1_kfold as e134
    from tools.exp149_label_sources import sha256
    for seed in (42, 43):
        for fold in range(5):
            run = root / 'artifacts/experiments' / control_tag(seed, fold)
            if fold == 0:
                p = e127.load_protocol('AUG1', seed, root)
                e127.verify_run(run, p, sha256(e127.manifest_path('AUG1', seed, root)), root)
            else:
                p = e134.load_protocol(seed, root)
                e134.verify_run(run, p, sha256(e134.manifest_path(seed, root)), root)


def markdown(result):
    lines = ['# EXP157 L2 five-fold two-seed screen', '',
             '**Decision: ' + result['decision']['status'] + '**', '',
             'Each seed is rank-normalised within fold, then the two seeds are averaged. '
             'Paired component bootstrap: 300 draws, seed 20261003.', '',
             '| Reference | L2 | AUG1 control | Delta | CI95 |', '|---|---:|---:|---:|---|']
    for name, m in result['metrics'].items():
        lines.append(f"| {name} | {m['candidate']:.6f} | {m['control']:.6f} | {m['delta']:+.6f} | {m['ci95']} |")
    lines += ['', '## Per-fold macro AUC deltas', '',
              '| Fold | JEV | Aquila | Dread | Flight | Consensus | Relabel |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for fold, scores in result['per_fold'].items():
        lines.append('| ' + fold + ' | ' + ' | '.join(
            f"{scores[name]['delta']:+.6f}" for name in
            ('jev_hybrid', 'aquila_v1', 'dread', 'flight_v4', 'consensus', 'relabel_v1')) + ' |')
    for name, m in result['metrics'].items():
        lines += ['', '## ' + name + ' per-target deltas', '', '| Target | Delta |', '|---|---:|']
        for t, d in m['per_target_delta'].items():
            lines.append(f"| {t} | {d:+.6f} |" if d is not None else f'| {t} | unscorable |')
    lines += ['', 'Report-derived references share report-label biases; this screen does not establish image-ground-truth performance.', '']
    return '\n'.join(lines)


def screen(root=ROOT):
    import numpy as np
    from tools import exp157_l2_advance as advance
    from tools import exp149_label_sources as e149, exp125_radr50_k16_local as e125
    root = Path(root)
    out = root / OUT
    if out.exists():
        raise FileExistsError(str(out))
    advance.verify(root)
    verify_controls(root)
    p = e149._load_manifest('student', root)
    canonical, fold_of = e149.fold_rosters(p, root)
    ids, candidate, control, hashes, per_fold = [], [], [], {}, {}
    for fold in range(5):
        fid = [u for u in canonical if fold_of[u] == fold]
        arms = []
        for kind in ('candidate', 'control'):
            seeds = []
            for seed in (42, 43):
                tag = (advance.source_tag(fold) if seed == 42 and fold < 2 else advance.tag(seed, fold)) if kind == 'candidate' else control_tag(seed, fold)
                file = root / 'artifacts/experiments' / tag / 'oof_k36.csv'
                seeds.append(e125.load_table(file, fid, prediction=True))
                hashes[tag] = e149.sha256(file)
            arms.append(seed_mean(seeds))
        ids.extend(fid)
        candidate.append(arms[0])
        control.append(arms[1])
        fold_refs = e149.registered_references(fid, root)
        from tools import exp115_vlm_eval as v115
        per_fold[str(fold)] = {}
        for name, ref in fold_refs.items():
            y, mask, min_class = ref
            av = v115.auc_targets(y, arms[0], mask, min_class)
            bv = v115.auc_targets(y, arms[1], mask, min_class)
            if not np.isfinite(av).any() or not np.isfinite(bv).any():
                raise ValueError('Unscorable fold/reference: ' + str(fold) + '/' + name)
            per_fold[str(fold)][name] = {
                'delta': float(np.nanmean(av) - np.nanmean(bv)),
                'per_target_delta': {t: float(d) if np.isfinite(d) else None
                                     for t, d in zip(e149.T, av - bv)}}
    if len(ids) != 4343 or len(set(ids)) != 4343:
        raise ValueError('EXP099 pooled study roster differs')
    a, b = np.concatenate(candidate), np.concatenate(control)
    refs = e149.registered_references(ids, root)
    draws = e125.component_draws(ids, e149.read(root / e149.COMPONENTS), draws=300, seed=20261003)
    metrics = {name: e149.paired_metric(a, b, ref, draws) for name, ref in refs.items()}
    result = {'schema': 'exp157-l2-screen-v1', 'pooled_studies': len(ids), 'folds': list(range(5)),
              'seeds': [42, 43], 'k_eval': 36, 'decision': decision(metrics),
              'metrics': metrics, 'per_fold': per_fold,
              'bootstrap': {'unit': 'component', 'draws': 300, 'seed': 20261003},
              'aggregation': 'per-fold average-tie ranks / fold size per seed, then seed mean',
              'oof_k36_sha256': hashes,
              'protocol_sha256': e149.sha256(root / 'artifacts/experiments/EXP157_L2_ADVANCE/PROTOCOL.md'),
              'reference_sha256': {rel: e149.sha256(root / rel) for rel in
                  (e149.JEV, e149.AQUILA, e149.COMPONENTS,
                   *(e149.e127.LABEL_PATHS[n] for n in
                     ('dread', 'flight_v4', 'relabel_v1', 'steven_v4', 'pilkwang_v2')))}}
    out.mkdir(parents=True)
    e149.exclusive_json(out / 'result.json', result)
    with (out / 'result.md').open('x', encoding='utf-8') as f:
        f.write(markdown(result))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    args = ap.parse_args(argv)
    result = screen(args.root)
    print(json.dumps(result['decision'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
