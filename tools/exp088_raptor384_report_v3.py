"""One fixed raw EXP088-versus-B evaluation using existing tested statistics.

prepare binds existing B predictions and the frozen training protocol without
loading labels/arrays/models. run requires the completed full parent and resource
receipt. Outputs are new JSON/Markdown; no retraining, ratios or Gold access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def prepare():
    from tools import exp088_raptor384_v3 as guard
    protocol = ROOT / guard.PROTOCOL / 'manifest.json'
    baseline = ROOT / 'artifacts/experiments/EXP084_B_G1'
    files = [protocol, Path(__file__), ROOT / 'tools/exp088_raptor384_v3.py',
             ROOT / 'training/src/raptor_common.py', ROOT / 'tools/exp087_coverage_report.py',
             ROOT / 'tools/exp084_group_loss_report.py',
             *[baseline / n for n in ('oof_best.csv', 'summary.json', 'provenance.json')]]
    p = guard.read(protocol)
    files += [ROOT / p['paths']['eval_labels'], ROOT / p['ordered_ids']['path'], ROOT / p['groups']['path']]
    registration = dict(schema='exp088-raptor384-report-v1', created_at=datetime.now(timezone.utc).isoformat(),
        protocol=str(protocol.relative_to(ROOT)).replace('\\', '/'),
        source_bindings={str(f.relative_to(ROOT)).replace('\\', '/'): guard.sha256(f) for f in files},
        baseline='EXP084_B_G1', candidate='EXP088_RAPTOR384_G1_V3',
        target_order=p['targets'], bootstrap_reps=500, bootstrap_seed=20261003,
        contrast='raw candidate minus B', ratio_search=False, gold=False)
    path = ROOT / guard.PROTOCOL / 'report-registration-resource-v3.json'
    guard.dump(path, registration)
    print(json.dumps(dict(registration=str(path), sha256=guard.sha256(path), no_evaluation=True)))


def run(registration, expected_sha, out):
    import numpy as np
    import pandas as pd
    from tools import exp088_raptor384_v3 as guard
    from tools import exp087_coverage_report as stats
    if guard.sha256(registration) != expected_sha:
        raise ValueError('Exact evaluation registration required')
    r = guard.read(registration)
    if r['schema'] != 'exp088-raptor384-report-v1':
        raise ValueError('Wrong evaluator contract')
    for path, sha in r['source_bindings'].items():
        if guard.sha256(ROOT / path) != sha:
            raise ValueError('Frozen evaluation input changed: ' + path)
    p = guard.read(ROOT / r['protocol'])
    directory = ROOT / 'artifacts/experiments' / r['candidate']
    logdir = ROOT / 'artifacts/logs' / r['candidate']
    if not (logdir / 'SUCCESS').is_file() or (logdir / 'FAILED').exists():
        raise ValueError('Full GPU parent did not complete successfully')
    supervisor = guard.read(directory / 'resource-supervisor.json')
    if (supervisor['status'] != 'PASS' or supervisor['returncode'] != 0
            or supervisor['protocol_sha256'] != guard.sha256(ROOT / r['protocol'])
            or supervisor['cleanup_errors'] or supervisor['cleanup_killed_pids']
            or supervisor['cleanup_survivors']
            or supervisor['error'] is not None or supervisor['killed'] is not None):
        raise ValueError('Resource/cleanup contract failed')
    guard.validate_completion(directory, p, p['arms']['full'])
    provenance = guard.read(directory / 'provenance.json')
    guard.validate_pretraining(provenance, p['environment'], p, p['arms']['full'])
    if provenance['protocol_sha256'] != guard.sha256(ROOT / r['protocol']):
        raise ValueError('Full provenance protocol differs')
    ordered = guard.read(ROOT / p['ordered_ids']['path'])['val']
    uid = 'StudyInstanceUID'
    frames = [pd.read_csv(directory / 'oof_best.csv', dtype={uid: str}),
              pd.read_csv(ROOT / 'artifacts/experiments' / r['baseline'] / 'oof_best.csv', dtype={uid: str})]
    for frame in frames:
        if list(frame.columns) != [uid, *r['target_order']] or frame[uid].tolist() != ordered:
            raise ValueError('Exact V888 target/UID order mismatch')
    sys.path.insert(0, str(ROOT / 'training/src'))
    from raptor_common import load_label_csv
    labels, semantic_hash = load_label_csv(ROOT / p['paths']['eval_labels'])
    if semantic_hash != p['eval_label_hash']:
        raise ValueError('Fixed weak label semantics changed')
    y = stats.base.binary_labels(np.stack([labels[u] for u in ordered]))
    scores = stats.scores_for(*[f[r['target_order']].to_numpy() for f in frames])
    meta = pd.read_csv(ROOT / p['groups']['path'], dtype={uid: str}).set_index(uid).loc[ordered]
    rows = stats.target_rows(y, scores)
    overall = stats.aggregate(rows)
    bootstrap = stats.paired_bootstrap(y, scores, meta['component_id'].astype(str).to_numpy(),
                                      reps=r['bootstrap_reps'], seed=r['bootstrap_seed'])
    groups = {}
    for col in p['groups']['subgroup_columns']:
        groups[col] = []
        for name in sorted(meta[col].fillna('unknown').astype(str).unique()):
            mask = meta[col].fillna('unknown').astype(str).to_numpy() == name
            detail = stats.target_rows(y[mask], {k: v[mask] for k, v in scores.items()})
            groups[col].append(dict(name=name, count=int(mask.sum()), targets=detail, overall=stats.aggregate(detail)))
    result = dict(schema=r['schema'], observed_at=datetime.now(timezone.utc).isoformat(),
        scope='Fixed exposed weak G1 development comparison; practical recipe effect',
        registration_sha256=expected_sha, protocol_sha256=guard.sha256(ROOT / r['protocol']),
        candidate=r['candidate'], baseline=r['baseline'], rows=len(ordered),
        targets=rows, overall=overall, bootstrap=bootstrap, groups=groups,
        outputs_sha256={n: guard.sha256(directory / n) for n in
                        ('oof_best.csv', 'summary.json', 'provenance.json', 'resource-supervisor.json')},
        limitations=['Conditional intervals exclude label/training-seed/researcher-selection/private uncertainty',
                    'Unknown patient links and labeler ancestry; no independent clinical validation',
                    'Resolution/architecture variant/epochs/precision/microbatch change together',
                    'No .943/Public/private claim; no blend/ratio/target selection; group findings exploratory'])
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError('Report output must stay in workspace')
    out.mkdir(parents=True, exist_ok=False)
    guard.dump(out / 'result.json', result)
    primary = overall['contrasts']['primary_candidate_minus_B']['macro12']
    lines = ['# EXP088 · 384px G1 레시피 비교', '',
             f'V {len(ordered)}건, raw candidate−B macro AUC: {primary}.', '',
             '| 소견 | P | N | B | candidate | 차이 |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['target']} | {row['P']} | {row['N']} | {row['auc']['B_original']} | {row['auc']['candidate']} | {row['delta']['primary_candidate_minus_B']} |")
    lines += ['', '조건부 component bootstrap과 그룹별 수치는 result.json에 보존한다. 한 seed와 반복 노출 weak 개발군의 레시피 비교이며 private/임상 개선으로 확대하지 않는다.', '']
    (out / 'result.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(json.dumps(dict(primary_delta=primary, output=str(out), rows=len(ordered))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    r = sub.add_parser('run')
    r.add_argument('--registration', type=Path, required=True)
    r.add_argument('--sha256', required=True)
    r.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    prepare() if a.command == 'prepare' else run(a.registration, a.sha256, a.out)


if __name__ == '__main__':
    main()
