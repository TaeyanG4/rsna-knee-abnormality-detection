"""H-B-164 R224 addition v1.0.0; independent ranks and authenticated raw output.

The existing C85 outer survivor policy is retained. Only R224 has the newly
registered >3% abort gate; existing R384/R7 failure thresholds are unchanged.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

try:
    from tools import exp169_release_c85r7 as base
except ImportError:
    import exp169_release_c85r7 as base

c85, p, v5q = base.c85, base.p, base.v5q
FORMULA = '0.85*r(P)+0.15*r(O); P=0.5*(0.65*r(R384)+0.35*r(R224))+0.5*r(R7); O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46'
SOURCE = 'raptor224_run.py'
RAW = 'raptor224_raw.csv'
SUB = 'raptor224_sub.csv'
CHECKPOINT = 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'
CHECKPOINT_SHA = '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'


def failure_gate(failed):
    p.need(len(failed) > 0 and int(np.sum(failed)) <= .03 * len(failed),
           'Raptor-224 failed on more than 3% of studies')


def read(directory, ids):
    work = Path(directory) / 'pr7-run'
    receipt = p.read_json(work / 'r224-receipt.json')
    p.need(receipt['checkpoint_sha256'] == CHECKPOINT_SHA and receipt['status'] == 'COMPLETE_R224', 'R224 identity/status')
    phase = p.read_json(Path(directory) / 'R224-exit.json')
    p.need(phase['exit_code'] == 0, 'R224 subprocess failed')
    arrays = []
    for name in (RAW, SUB):
        path = Path(receipt['files'][name]['path'])
        p.check_pin(path, receipt['files'][name]['sha256'])
        uids, values = v5q._read_csv(path)
        v5q.validate_ids(uids)
        p.need(set(uids) == set(ids), 'R224 cohort mismatch')
        values = v5q._probabilities(name, values, len(ids))
        arrays.append(values[[uids.index(uid) for uid in ids]])
    raw, sub = arrays
    failed = np.all(raw == .5, axis=1)
    failure_gate(failed)
    if len(ids) >= 3:
        import pandas as pd
        rho = [round(float(pd.Series(raw[:, j]).corr(pd.Series(sub[:, j]), method='spearman')), 4) for j in range(12)]
        # Same comparison as heliosli cell 8: undefined constant-column rho is not a veto.
        p.need(not min(rho) < .99, 'R224 raw/sub rank disagreement')
    return raw, {uid: 'FailedStudySentinel: public cell 8' for uid, fail in zip(ids, failed) if fail}, receipt


def r_half(r, r224, rf, f224):
    failure_gate(f224)
    a, b = c85.scored_rank(r, rf), c85.scored_rank(r224, f224)
    rr = .65 * a + .35 * b
    rr[rf], rr[f224], rr[rf & f224] = b[rf], a[f224], .5
    return rr, rf & f224


def p_branch(r, r224, ours, rf, f224, sf):
    rr, failed = r_half(r, r224, rf, f224)
    ss = c85.scored_rank(ours, sf)
    public = .5 * rr + .5 * ss
    public[failed], public[sf], public[failed & sf] = ss[failed], rr[sf], .5
    return public


def blend(ids, r, r224, ours, exp143, exp155, q, failures=None):
    ids = v5q.validate_ids(ids)
    names = ['R', 'R224', 'S_R7', 'EXP143v2', 'EXP155', 'Q_A5q']
    values = [v5q._probabilities(name, arr, len(ids)) for name, arr in zip(names, (r, r224, ours, exp143, exp155, q))]
    r, r224, ours, exp143, exp155, q = values
    failures = failures or {}
    masks = [np.asarray([uid in failures.get(name, {}) for uid in ids], dtype=bool) for name in names]
    masks[0] |= np.all(r == .5, axis=1)
    masks[1] |= np.all(r224 == .5, axis=1)
    rf, f224, sf, f143, f155, fq = masks
    public = p_branch(r, r224, ours, rf, f224, sf)
    e143, e155, qq = [c85.scored_rank(v, f) for v, f in zip((exp143, exp155, q), masks[3:])]
    denom = 17 * (~f143).astype(int) + 17 * (~f155).astype(int) + 12 * (~fq).astype(int)
    own = (17 * e143 + 17 * e155 + 12 * qq) / np.maximum(denom, 1)[:, None]
    pf, of = rf & f224 & sf, denom == 0
    pr, oo = c85.scored_rank(public, pf), c85.scored_rank(own, of)
    result = .85 * pr + .15 * oo
    result[of], result[pf], result[pf & of] = pr[of], oo[pf], .5
    return result


def reference(r, r224, ours, exp143, exp155, q):
    import pandas as pd
    def rank(x):
        return pd.DataFrame(x).rank(method='average', pct=True).to_numpy()
    return .85 * rank(.5 * (.65 * rank(r) + .35 * rank(r224)) + .5 * rank(ours)) + .15 * rank(
        (17 * rank(exp143) + 17 * rank(exp155) + 12 * rank(q)) / 46)


def record(work, working):
    """Called only after successful child exit; immutable identity record."""
    payload = dict(status='COMPLETE_R224', checkpoint_sha256=CHECKPOINT_SHA,
        files={name: dict(path=str(Path(working) / name), sha256=p.sha(Path(working) / name)) for name in (RAW, SUB)})
    p.write_once(Path(work) / 'r224-receipt.json', p.json_bytes(payload))
