"""U union composition v1.0.0; immutable P/O inputs and full-roster rank outputs.

Changelog: fixed INF3/K1 union, no A5 dependency or inference.
"""
from pathlib import Path
import hashlib
import json
import csv
import io
import numpy as np
from tools import c85r7_rco_overlay_v1 as util
VERSION = '1.0.0'
FORMULA = 'INF3=rank(.85*r(P)+.15*r(O)); K=rank(mean(rank(K1 trio stride2))); X=rank(.8*r(INF3)+.2*r(K)); Y=rank(.75*r(P)+.25*r(K)); U=rank(.5*r(X)+.5*r(Y)); inherited survivor substitution; average ties/full roster/float32 CSV'
CONTRACT = dict(inf3=[.85,.15], x=[.8,.2], y=[.75,.25], union=[.5,.5], k1_stride=2, r6_stride=1, r224_weight=.35, p_weight_mode='public', a5=False, drop_o=False)
def digest(data): return hashlib.sha256(data).hexdigest()
def csv_bytes(ids, values):
    buf=io.StringIO(newline=''); writer=csv.writer(buf,lineterminator='\n')
    writer.writerow([util.UID]+util.LABELS)
    writer.writerows([uid,*[format(float(v),'.9g') for v in row]] for uid,row in zip(ids,np.asarray(values,dtype=np.float32)))
    return buf.getvalue().encode()
def save(output,name,payload):
    path=Path(output)/name
    if path.exists(): util.need(path.read_bytes()==payload,'immutable U replay differs: '+name)
    else: util.write_once(path,payload)
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
    public,own,pf,of=split(r224,ids,*arrays,failures)
    expected=r224.p.csv_bytes(ids,outer(public,own,pf,of,r224.c85.scored_rank))
    util.need(expected==parent_payload,'U snapshot must equal unchanged INF3 parent')
    data=dict(schema='u-union-inputs/v1',ids=ids,public=public.tolist(),own=own.tolist(),
        p_failed=pf.tolist(),o_failed=of.tolist(),original_inf3_sha256=digest(parent_payload),contract=CONTRACT)
    raw=util.encoded(data)
    for name,payload in [('u-inputs.json',raw),('u-original-inf3.csv',parent_payload),
                         ('u-p.csv',csv_bytes(ids,public)),('u-o.csv',csv_bytes(ids,own))]: save(output,name,payload)
    return dict(u_inputs_sha256=digest(raw),u_contract=CONTRACT)

def compose(output,parent_payload,records,runtime,diagnostic=False):
    output=Path(output); data=json.loads((output/'u-inputs.json').read_bytes())
    codec=json.loads((output/'c85r7-receipt.json').read_bytes())
    ids,parent=util.table(parent_payload)
    util.need(data['schema']=='u-union-inputs/v1' and data['contract']==CONTRACT,'U contract')
    util.need(codec['u_inputs_sha256']==util.sha(output/'u-inputs.json'),'U snapshot binding')
    util.need(ids==data['ids'] and digest(parent_payload)==data['original_inf3_sha256'],'U original INF3 binding')
    # Reuse the reviewed inherited full-roster K1 validation and X fallback.
    xbytes=runtime.blend(parent_payload,records); _,x=util.table(xbytes)
    lookup={r['uid']:r for r in records}; failed=np.asarray([not lookup[uid]['ok'] for uid in ids])
    member=[]
    for mid in runtime.MODEL_IDS:
        pred=np.asarray([parent[i] if failed[i] else lookup[uid]['predictions'][mid] for i,uid in enumerate(ids)],dtype=np.float32).astype(np.float64)
        member.append(util.ranks(pred))
    kmean=np.mean(np.stack(member),axis=0); k=util.ranks(kmean)
    # P crosses the same float32 CSV boundary as the P25 parent.
    _,public=util.table((output/'u-p.csv').read_bytes()); pr=util.ranks(public)
    ky=k.copy(); ky[failed]=pr[failed]
    weighted=.75*pr+.25*ky; weighted[failed]=pr[failed]
    y=util.ranks(weighted).astype(np.float32)
    u=util.ranks(.5*util.ranks(x)+.5*util.ranks(y)).astype(np.float32)
    saved={'u-inf3.csv':csv_bytes(ids,util.ranks(parent)),'u-k.csv':csv_bytes(ids,k),
           'u-x.csv':xbytes,'u-y.csv':csv_bytes(ids,y),'u-final.csv':csv_bytes(ids,u)}
    for name,payload in saved.items(): save(output,name,payload)
    receipt=dict(schema='u-union/v1',version=VERSION,status='COMPLETE_U',formula=FORMULA,contract=CONTRACT,
        diagnostic=diagnostic,never_submit=diagnostic,studies=len(ids),a5_executed=False,
        failed_k1_substitution='parent rank per X/Y branch after K ranking; original INF3 placeholders before member ranks',
        inputs_sha256=util.sha(output/'u-inputs.json'),original_inf3_sha256=digest(parent_payload),
        csv_sha256={name:digest(payload) for name,payload in saved.items()},
        k1_records_sha256=digest(util.encoded(records)))
    save(output,'u-receipt.json',util.encoded(receipt))
    return saved['u-final.csv']
