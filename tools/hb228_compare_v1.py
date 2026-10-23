"""H-B-228 v1.0.0 CPU aggregate comparison; no label rows in outputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from tools.c85r7_rco_overlay_v1 import LABELS as TARGETS, UID


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictions(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        need(r.fieldnames == [UID,*TARGETS], 'prediction schema')
        rows = list(r)
    ids = [r[UID] for r in rows]
    need(len(ids) > 1 and all(u and u == u.strip() for u in ids) and len(ids) == len(set(ids)),
         'empty/duplicate prediction IDs')
    try:
        x = np.array([[float(r[t]) for t in TARGETS] for r in rows])
    except (ValueError, TypeError):
        raise ValueError('malformed prediction number') from None
    need(x.shape == (len(ids),12) and np.isfinite(x).all() and ((0 <= x)&(x <= 1)).all(),
         'prediction values must be finite probabilities')
    return ids,x


def compare(left, right, *, expected_ids=None, rulers_root=None):
    ids,a = predictions(left)
    other,b = predictions(right)
    need(set(ids) == set(other), 'prediction roster differs')
    if expected_ids is not None:
        need(len(expected_ids) == len(set(expected_ids)) and set(ids) == set(expected_ids),
             'selection roster differs')
    index = {u:i for i,u in enumerate(other)}
    b = b[[index[u] for u in ids]]
    rho = [None if np.ptp(a[:,j]) == 0 or np.ptp(b[:,j]) == 0
           else float(spearmanr(a[:,j],b[:,j]).statistic) for j in range(12)]
    report = dict(schema='hb228-comparison/v1', studies=len(ids),
        stride1_sha256=sha(left), stride2_sha256=sha(right),
        max_abs_delta=float(np.abs(a-b).max()),
        per_target={t:dict(max_abs_delta=float(np.abs(a[:,j]-b[:,j]).max()),spearman=rho[j],
                          spearman_status='UNDEFINED_CONSTANT' if rho[j] is None else 'DEFINED')
                    for j,t in enumerate(TARGETS)},
        auc={}, clean_oof=False, exposure='EXPOSED_EXPLORATORY',
        decision='MEASUREMENT_ONLY_NO_AUTOMATIC_PASS', qualified_for_submission=False)
    if rulers_root is not None:
        from tools import evidence_ledger_rulers as rulers
        from sklearn.metrics import roc_auc_score
        need(tuple(TARGETS) == rulers.T, 'ruler target contract')
        specs = dict(rulers.source_specs())
        specs['gold'] = dict(path=rulers.GOLD, rule='binary', ancestors=['gold'], provenance='exposed Gold')
        for name,spec in specs.items():
            path = Path(rulers_root)/spec['path']
            if not path.is_file():
                report['auc'][name] = dict(status='MISSING')
                continue
            _,labels = rulers.load_table(path, ids, binary=spec['rule'] == 'binary')
            hard = rulers.hard_labels(labels,spec['rule'])
            result = {}
            for j,t in enumerate(TARGETS):
                mask = np.isfinite(hard[:,j])
                y = hard[mask,j]
                value = dict(n=int(mask.sum()), status='NO_LABELS' if len(y) == 0 else 'SINGLE_CLASS')
                if len(np.unique(y)) == 2:
                    x1,x2 = [float(roc_auc_score(y,x[mask,j])) for x in (a,b)]
                    value.update(status='DEFINED',stride1_auc=x1,stride2_auc=x2,delta=x2-x1)
                result[t] = value
            report['auc'][name] = dict(status='LOADED', sha256=sha(path), rule=spec['rule'],
                ancestors=spec['ancestors'], per_target=result)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stride1', type=Path, required=True)
    p.add_argument('--stride2', type=Path, required=True)
    p.add_argument('--selection', type=Path)
    p.add_argument('--rulers-root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    ids = json.loads(a.selection.read_bytes())['ids'] if a.selection else None
    result = compare(a.stride1,a.stride2,expected_ids=ids,rulers_root=a.rulers_root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPARED', studies=result['studies'], max_abs_delta=result['max_abs_delta'],
                         output=str(a.out), qualified_for_submission=False)))


if __name__ == '__main__':
    main()
