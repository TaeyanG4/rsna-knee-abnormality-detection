"""Label-free order parity + optional labelled AUC delta between two prediction sets (CPU only).

Rank-based blends and AUC depend only on per-target ordering, so the number of study pairs whose
order differs between environments bounds how far any labelling could move the score.

    python tools/ml_parity_order_check.py REF_predictions.csv CAND_predictions.csv \
        [--ref-members REF.npz --cand-members CAND.npz] [--labels LABELS.csv] [--out report.json]

Predictions CSV: StudyInstanceUID + the 12 targets. Members NPZ: study_uids, member_names, targets,
member_probabilities (M, N, 12) as written by the B_RN3F producer. Labels are rounded as in
raptor_common.macro_auc; the label source is recorded and is not an accuracy claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/src'))
from raptor_common import LAB, UID, macro_auc  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order_changes(a, b):
    """Per-column count of unordered pairs whose sign(a_i - a_j) differs from sign(b_i - b_j)."""
    iu = np.triu_indices(a.shape[0], 1)
    out = []
    for j in range(a.shape[1]):
        da = np.sign(a[:, j][:, None] - a[:, j][None, :])[iu]
        db = np.sign(b[:, j][:, None] - b[:, j][None, :])[iu]
        out.append(int((da != db).sum()))
    return out, int(len(iu[0]))


def load_predictions(path, uids=None):
    df = pd.read_csv(path)
    if df[UID].duplicated().any():
        raise ValueError(f'{path}: duplicate {UID}')
    df = df.set_index(UID)
    uids = list(df.index) if uids is None else list(uids)
    return uids, df.loc[uids, LAB].to_numpy(np.float64)


def compare(ref, cand, label_rows=None):
    changes, pairs = order_changes(ref, cand)
    rep = {'max_abs': float(np.abs(ref - cand).max()), 'mean_abs': float(np.abs(ref - cand).mean()),
           'pairs_per_target': pairs, 'order_changes_per_target': dict(zip(LAB, changes)),
           'max_order_change_fraction': max(changes) / pairs if pairs else 0.0}
    if label_rows is not None:
        a, pa = macro_auc(label_rows, ref)
        b, pb = macro_auc(label_rows, cand)
        rep.update(macro_auc_ref=a, macro_auc_cand=b, macro_auc_delta=b - a, targets_scored=len(pa),
                   max_abs_target_auc_delta=max((abs(pa[k] - pb[k]) for k in pa), default=float('nan')))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ref'); ap.add_argument('cand')
    ap.add_argument('--ref-members'); ap.add_argument('--cand-members')
    ap.add_argument('--labels', help='CSV with StudyInstanceUID + 12 targets (rows missing labels are skipped per target)')
    ap.add_argument('--out')
    args = ap.parse_args()

    uids, ref = load_predictions(args.ref)
    _, cand = load_predictions(args.cand, uids)
    y = None
    report = {'schema': 'ML_PARITY_ORDER_CHECK_V1', 'n': len(uids), 'quality_claim': False,
              'inputs': {p: sha(p) for p in (args.ref, args.cand)}}
    if args.labels:
        lab = pd.read_csv(args.labels).set_index(UID)
        y = lab.reindex(uids)[LAB].to_numpy(np.float64)
        report['inputs'][args.labels] = sha(args.labels)
        report['labelled_rows'] = int(np.isfinite(y).any(1).sum())
    report['final'] = compare(ref, cand, y)

    if args.ref_members and args.cand_members:
        A, B = np.load(args.ref_members), np.load(args.cand_members)
        if not (list(A['study_uids']) == list(B['study_uids']) and list(A['member_names']) == list(B['member_names'])
                and list(A['targets']) == LAB == list(B['targets'])):
            raise ValueError('member files are not aligned (uids/members/targets)')
        order = [list(A['study_uids']).index(u) for u in uids]
        y_m = None if y is None else y
        report['members'] = {str(n): compare(A['member_probabilities'][i][order].astype(np.float64),
                                             B['member_probabilities'][i][order].astype(np.float64), y_m)
                             for i, n in enumerate(A['member_names'])}
        for p in (args.ref_members, args.cand_members):
            report['inputs'][p] = sha(p)

    f = report['final']
    line = (f"n={report['n']} final max_abs={f['max_abs']:.3g} max_order_change_fraction={f['max_order_change_fraction']:.5f}"
            f" changes={list(f['order_changes_per_target'].values())}")
    if 'macro_auc_delta' in f:
        line += f" macro_auc {f['macro_auc_ref']:.6f}->{f['macro_auc_cand']:.6f} ({f['macro_auc_delta']:+.2e})"
    print(line)
    for n, r in report.get('members', {}).items():
        print(f"  {n}: max_abs={r['max_abs']:.3g} order_changes total={sum(r['order_changes_per_target'].values())}"
              f" max={max(r['order_changes_per_target'].values())}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
