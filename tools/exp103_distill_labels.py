"""H-B-20 distilled training labels: per target 0.5 * Dread + 0.5 * teacher, teacher = equal mean of
out-of-fold teacher probabilities (pre-registered 2026-10-07 before any student or teacher OOF was seen).

  python tools/exp103_distill_labels.py build --teacher coatnet=PATH --teacher rn34=PATH [--out DIR]

Inputs are assembled 5-fold OOF CSVs (one row per G1 study, every study predicted by the fold model that
did not train on it). Rows of the Dread label file that no teacher covers (studies outside the 4,343 G1
studies) keep their Dread value. Gold rows are never present (the Dread file excludes them) and are
refused. Writes <out>/distill_v1.csv (same columns and row order as the Dread file) and receipt.json.
Leakage note: a student fold trains on studies whose teacher labels came from teacher folds that saw the
student fold's validation studies, so the student's own OOF is optimistic and must not be used for
decisions or stacking; the decision uses the public LB (test studies were never in any teacher's training).
"""
# Quarantine (EXP116 v1.1.1, D-B-024): build() refuses, before anything is read or written, every teacher file whose directory or any
# parent directory below the workspace root carries DIAGNOSTIC_ONLY.json or a result.json / manifest.json with decision_eligible false
# (tools/diagnostic_quarantine.py): a leaky student OOF is never a teacher.
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.diagnostic_quarantine import require_decision_eligible  # noqa: E402
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
           'Synovitis', "Baker's", 'Contusion', 'Fracture']
DREAD = ROOT / 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv'
TRAIN = ROOT / 'storage/competition/train.csv'
G1 = ROOT / 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json'
ALPHA = 0.5   # weight of the Dread label; fixed before any result


def rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(teachers: dict[str, str], out: Path):
    for path in teachers.values():  # before any read and before --out exists
        require_decision_eligible(path, 'tools/exp103_distill_labels.py', ROOT)
    out.mkdir(parents=True, exist_ok=False)
    dread = pd.read_csv(DREAD)
    assert list(dread.columns[:1]) == [UID] and all(t in dread.columns for t in TARGETS)
    gold = set(pd.read_csv(TRAIN).dropna(subset=['ACL'])[UID])
    assert not (set(dread[UID]) & gold), 'Dread file must not contain gold rows'
    g1 = json.loads(G1.read_text(encoding='utf-8')); g1_ids = list(g1['train']) + list(g1['val'])
    frames = {}
    for name, path in teachers.items():
        t = pd.read_csv(path)
        assert t[UID].is_unique and set(t[UID]) == set(g1_ids), f'{name}: teacher must cover exactly the {len(g1_ids)} G1 studies'
        assert not (set(t[UID]) & gold)
        v = t.set_index(UID)[TARGETS].astype(float)
        assert np.isfinite(v.values).all() and (v.values >= 0).all() and (v.values <= 1).all(), f'{name}: probabilities required'
        frames[name] = v
    teacher = sum(frames.values()) / len(frames)
    mixed = dread.copy().set_index(UID)
    covered = mixed.index.intersection(teacher.index)
    mixed.loc[covered, TARGETS] = ALPHA * mixed.loc[covered, TARGETS].values + (1 - ALPHA) * teacher.loc[covered, TARGETS].values
    mixed = mixed.reset_index()[dread.columns]
    path = out / 'distill_v1.csv'
    mixed.to_csv(path, index=False, float_format='%.6f')
    corr = {n: float(np.mean([np.corrcoef(frames[n].loc[covered, t], dread.set_index(UID).loc[covered, t])[0, 1] for t in TARGETS])) for n in frames}
    tt = None
    if len(frames) == 2:
        a, b = list(frames.values())
        tt = float(np.mean([np.corrcoef(a.loc[covered, t], b.loc[covered, t])[0, 1] for t in TARGETS]))
    receipt = dict(schema='exp103-distill-labels-v1', created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), alpha_dread=ALPHA,
                   teacher_rule='equal mean of teacher OOF probabilities', teachers={n: dict(path=str(p), sha256=sha(p)) for n, p in teachers.items()},
                   dread=dict(path=str(DREAD.relative_to(ROOT)), sha256=sha(DREAD)), rows=len(mixed), covered=len(covered),
                   kept_dread_only=int(len(mixed) - len(covered)), output=dict(path=rel(path), sha256=sha(path)),
                   mean_target_corr_teacher_vs_dread=corr, mean_target_corr_between_teachers=tt,
                   mean_label_before=dread[TARGETS].mean().round(4).to_dict(), mean_label_after=mixed[TARGETS].mean().round(4).to_dict(),
                   leakage_note=__doc__.split('Leakage note: ')[1].strip())
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('rows', 'covered', 'kept_dread_only', 'mean_target_corr_teacher_vs_dread', 'mean_target_corr_between_teachers', 'output')}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--teacher', action='append', required=True, help='NAME=PATH to an assembled OOF CSV')
    b.add_argument('--out', default=str(ROOT / 'storage/labels/derived_distill/v1'))
    a = ap.parse_args()
    teachers = dict(x.split('=', 1) for x in a.teacher)
    build(teachers, Path(a.out))


if __name__ == '__main__':
    main()
