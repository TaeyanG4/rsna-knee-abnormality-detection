"""v1.0.0: put the unchanged fivefold A5 member inside P, before INF3 and K1.

Changelog: authenticated P/O snapshot, exact A5 phase, and explicit INF3 binding.
No inference scheduling, model, render or A5 receipt implementation changes.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from tools import c85r7_rco_overlay_v1 as util
from tools import p25_a5_runtime_v1 as a5

VERSION = '1.0.0'
FORMULA = 'INF3_A5=.85*rank(Pprime)+.15*rank(O); inherited P/O survivor substitution; float32 CSV'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def split(r224, ids, r, rr, ours, exp143, exp155, q, failures):
    """The parent's exact branch arithmetic, with no change to survivor masks."""
    names = ['R', 'R224', 'S_R7', 'EXP143v2', 'EXP155', 'Q_A5q']
    values = [r224.v5q._probabilities(name, arr, len(ids))
              for name, arr in zip(names, (r, rr, ours, exp143, exp155, q))]
    r, rr, ours, exp143, exp155, q = values
    masks = [np.asarray([uid in failures.get(name, {}) for uid in ids], dtype=bool) for name in names]
    masks[0] |= np.all(r == .5, axis=1)
    masks[1] |= np.all(rr == .5, axis=1)
    rf, f224, sf, f143, f155, fq = masks
    public = r224.p_branch(r, rr, ours, rf, f224, sf)
    e143, e155, qq = [r224.c85.scored_rank(v, f) for v, f in zip((exp143, exp155, q), masks[3:])]
    denom = 17*(~f143).astype(int)+17*(~f155).astype(int)+12*(~fq).astype(int)
    own = (17*e143+17*e155+12*qq)/np.maximum(denom, 1)[:, None]
    return public, own, rf & f224 & sf, denom == 0


def outer(public, own, pf, of, scored_rank):
    pr, oo = scored_rank(public, pf), scored_rank(own, of)
    result = .85*pr+.15*oo
    result[of], result[pf], result[pf & of] = pr[of], oo[pf], .5
    return result


def snapshot(r224, output, ids, arrays, failures, parent_payload):
    public, own, pf, of = split(r224, ids, *arrays, failures)
    # The original INF3 is still published by the unchanged parent collect.
    pbytes = r224.p.csv_bytes(ids, public)
    data = dict(schema='x3-a5-inputs/v1', ids=ids, own=own.tolist(),
                p_failed=pf.tolist(), o_failed=of.tolist(),
                original_inf3_sha256=digest(parent_payload), original_p_sha256=digest(pbytes))
    for name,payload in [('x3-a5-original-inf3.csv',parent_payload),
                         ('x3-a5-inputs.json',util.encoded(data)),
                         ('x3-a5-original-p.csv',pbytes)]:
        path=Path(output)/name
        if path.exists():
            util.need(path.read_bytes()==payload,'immutable X3 snapshot replay differs: '+name)
        else:
            util.write_once(path,payload)
    return dict(x3_a5_inputs_sha256=digest(util.encoded(data)), a5_original_p_sha256=digest(pbytes))


def compose(output, pprime, diagnostic=False, verify_only=False):
    """Bind original INF3, exact P/A5 receipt, unchanged O and final INF3 bytes."""
    import exp169_r224 as r224
    output = Path(output)
    data = json.loads((output/'x3-a5-inputs.json').read_bytes())
    parent = (output/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).read_bytes()
    original_p = (output/'x3-a5-original-p.csv').read_bytes()
    a = json.loads((output/'a5-receipt.json').read_bytes())
    c = json.loads((output/'c85r7-receipt.json').read_bytes())
    util.need(data.get('schema') == 'x3-a5-inputs/v1', 'x3 A5 inputs schema')
    util.need(c['x3_a5_inputs_sha256'] == util.sha(output/'x3-a5-inputs.json'), 'x3 A5 inputs binding')
    util.need(data['original_inf3_sha256'] == digest(parent), 'original INF3 binding')
    util.need(a['status'] == 'COMPLETE_A5' and
              a['original_p_sha256'] == data['original_p_sha256'] == digest(original_p) == c['a5_original_p_sha256'] and
              a['pprime_sha256'] == digest(pprime), 'P -> A5 Pprime binding')
    ids, public = util.table(pprime)
    util.need(ids == data['ids'] == util.table(original_p)[0] == util.table(parent)[0], 'INF3/A5 roster')
    own = np.asarray(data['own'], dtype=np.float64)
    pf, of = np.asarray(data['p_failed'], bool), np.asarray(data['o_failed'], bool)
    util.need(own.shape == public.shape and np.isfinite(own).all() and
              pf.shape == of.shape == (len(ids),), 'INF3 O/mask shape')
    result = r224.p.csv_bytes(ids, outer(public, own, pf, of, r224.c85.scored_rank))
    if not verify_only: util.write_once(output/'x3-a5-inf3.csv', result)
    receipt = dict(schema='x3-a5-inf3/v1', status='COMPLETE_INF3_A5', formula=FORMULA,
        diagnostic=diagnostic, never_submit=diagnostic, studies=len(ids),
        original_inf3_sha256=digest(parent), original_p_sha256=digest(original_p),
        pprime_sha256=digest(pprime), inf3_sha256=digest(result),
        inputs_sha256=util.sha(output/'x3-a5-inputs.json'),
        a5_receipt_sha256=util.sha(output/'a5-receipt.json'))
    if verify_only:
        util.need((output/'x3-a5-inf3.csv').read_bytes() == result and
                  (output/'x3-a5-inf3-receipt.json').read_bytes() == util.encoded(receipt), 'INF3 composition changed')
    else:
        util.write_once(output/'x3-a5-inf3-receipt.json', util.encoded(receipt))
    return result


def run(output, competition, codec, diagnostic=False):
    output = Path(output)
    phase = output/'x3-a5-phase'
    phase.mkdir(exist_ok=False)
    name = 'diagnostic-c85r7.csv' if diagnostic else 'submission.csv'
    util.write_once(phase/name, (output/'x3-a5-original-p.csv').read_bytes())
    pprime = a5.run(phase, competition, codec, diagnostic=diagnostic)
    for name in ('a5-receipt.json', 'a5-pprime.csv', 'a5-original-p.csv'):
        util.write_once(output/name, (phase/name).read_bytes())
    return compose(output, pprime, diagnostic)
