"""EXP122 s43 Tier-B check v1.0.0 (H-B-47 TIER-B RULE, registered 2026-10-08 11:45 KST).

Seed-matched delta multi-head s43 (EXP122) - single-head s43 (EXP118), both per-fold rank-normalised,
pooled K36 OOF, references from tools/exp115_vlm_eval, paired component bootstrap 300 draws seed 20261003.
Pass = dDread >= 0 and no reference below -0.001. Report only; writes a new result file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp115_vlm_eval as mr  # noqa: E402
from tools import exp117_multihead as e117  # noqa: E402

UID = 'StudyInstanceUID'
OUT = ROOT / 'artifacts/experiments/EXP122_S43_TIERB_CHECK/result.json'


def main():
    cand = pd.read_csv(ROOT / 'artifacts/experiments/EXP122_MULTIHEAD_S43_OOF/oof_k36.csv', dtype={UID: str})
    ctrl = pd.read_csv(ROOT / 'artifacts/experiments/EXP118_RADR50_LOCAL_S43_OOF/oof_k36.csv', dtype={UID: str})
    T = [c for c in cand.columns if c != UID]
    ctrl = ctrl.set_index(UID).loc[cand[UID]].reset_index()
    assert list(ctrl.columns) == [UID] + T and len(cand) == len(ctrl)
    ids = cand[UID].tolist()
    folds = {int(k): json.loads((ROOT / f['ordered_ids']['path']).read_text(encoding='utf-8'))['val']
             for k, f in json.loads((ROOT / e117.PROTOCOL / 'manifest.json').read_text(encoding='utf-8'))['folds'].items()}
    row = {u: i for i, u in enumerate(ids)}
    assert set(row) == {u for m in folds.values() for u in m}
    def per_fold_rank(a):
        out = np.empty_like(a)
        for members in folds.values():
            ix = [row[u] for u in members]
            out[ix] = mr.pct_rank(a[ix])
        return out
    A = per_fold_rank(cand[T].to_numpy(np.float64))
    B = per_fold_rank(ctrl[T].to_numpy(np.float64))
    refs = mr.references(ids, ROOT)
    idx = mr.bootstrap_index(mr.component_groups(ids, ROOT), 300, 20261003)
    res = {r: mr.compare(A, B, refs[r], idx) for r in mr.REFERENCES}
    deltas = {r: v['delta'] for r, v in res.items()}
    verdict = 'PASS' if deltas['dread'] >= 0 and all(d >= -0.001 for d in deltas.values()) else 'FAIL'
    OUT.parent.mkdir(parents=True, exist_ok=False)
    OUT.write_text(json.dumps(dict(schema='exp122-s43-tierb-v1', rule='H-B-47 TIER-B 2026-10-08 11:45 KST',
                                   verdict=verdict, deltas=deltas, metrics=res), indent=2, default=float) + '\n', encoding='utf-8')
    print(json.dumps(dict(verdict=verdict, deltas=deltas,
                          dread_ci=res['dread'].get('ci95'), dread_share=res['dread'].get('positive_draw_share')), indent=2, default=float))


if __name__ == '__main__':
    main()
