"""H197 INF3 bridge v1.0.0: fixed CPU-only OOF and exposed serving replay.

Changelog 2026-10-18: versioned extension of audited aligned/QPOOL readers;
no fitting, forwards, subprocess, release, download or submission interface.
"""
from __future__ import annotations
import os
import sys
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import hb105_aligned_oof_v1 as aligned  # installs CUDA/checkpoint tripwires
from tools import multi_lens_readout as ml
from tools import ensemble_ablation_v1 as ea
from tools import hb145_output_io as output_io
from tools import c85r7_r6w958q_v1139 as parent
from tools import exp169_release_v2 as qparent
from tools import hb128_gold58_pbranch_v1 as gold
import numpy as np
import pandas as pd

VERSION = '1.0.0'
OUT = ROOT/'.local/scratch/slot-b/batch5-1018/inf3-bridge'
QPOOL = ROOT/'.local/scratch/slot-b/cpu-small/QPOOL'
STAGE = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q'
GATE_REG = ROOT/'.local/scratch/astra-b5/hb189-gate/gate-copy-v145/registration.json'
ROUTING = ROOT/'.local/scratch/slot-b/routing-1016/astra/input_hashes.json'
COMMON = ROOT/'.local/scratch/hb167-common/hb167/gpu'
COMMON_REG = ROOT/'artifacts/experiments/HB167_PARENT_COMMON/v1.0.3/registration.json'
R6 = ROOT/'.local/scratch/r6-gold58-build/r6-gold58/gpu/flip/output'
QSHA = '39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc'
NOTEBOOK_SHA = 'fcb2df01b47d5cc9b83c8a11e35ceca10d19ba5884e2b8fd40aee13867f3e8c9'
FAMILIES = ('l2', 'dino', 'convnext')
EXPECTED = [(f, s, k) for f in FAMILIES for s in ((42, 43) if f == 'l2' else (42,)) for k in range(5)]
NAMES = ('R', 'R224', 'S_R7', 'EXP143v2', 'EXP155', 'Q_A5q')
need = aligned.need


def sha(data): return hashlib.sha256(data).hexdigest()
def encoded(obj): return (json.dumps(ml.clean(obj), sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def save(name, value):
    path = (OUT/name).resolve()
    need(path.is_relative_to(OUT.resolve()) and path != OUT.resolve(), 'output scope')
    data = encoded(value)
    if path.exists():
        need(path.read_bytes() == data, 'immutable output differs: '+name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    def write():
        with temporary.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    output_io.retry(write)
    output_io.replace(temporary, path)


def probabilities(a, n=None):
    a = np.asarray(a)
    need(a.ndim == 2 and a.shape[1] == 12 and len(a) > 0 and
         (n is None or len(a) == n) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), 'probability shape/range/NaN')
    return a


def align(ids, a, wanted):
    ids, wanted = list(map(str, ids)), list(map(str, wanted))
    need(len(ids) == len(set(ids)) and len(wanted) == len(set(wanted)) and
         set(ids) == set(wanted) and all(ids), 'ID set/duplicates')
    a = probabilities(a, len(ids)); lookup = {u:i for i,u in enumerate(ids)}
    return a[[lookup[u] for u in wanted]]


def inverse_cost(q, w):
    q = probabilities(q).astype(np.float64)
    w = np.asarray(w, np.float64)
    need(w.shape == (12,) and np.isfinite(w).all() and (w >= 1).all(), 'invalid actual_pos_weight')
    return q/(w-(w-1)*q)


def validate_entries(entries):
    need([(e['family'],e['seed'],e['fold']) for e in entries] == EXPECTED, 'ordered family/seed/fold coverage')
    need(len({e['source_sha256'] for e in entries}) == 20, 'duplicate source identity')
    for e in entries:
        for k in ('sha256', 'source_sha256'):
            need(len(e[k]) == 64 and all(c in '0123456789abcdef' for c in e[k]), 'invalid member digest')


def member_tuple(e): return tuple(e[k] for k in ('family','seed','fold','source_sha256','sha256'))


def serving_q3(leaves, costs):
    need(set(leaves) == set(costs) == set(FAMILIES), 'missing/extra families')
    means = []
    ns = set()
    for family in FAMILIES:
        a, w = np.asarray(leaves[family]), np.asarray(costs[family])
        count = 10 if family == 'l2' else 5
        need(a.ndim == 3 and a.shape[0] == count and w.shape == (count,12), 'family count/weight shape')
        ns.add(a.shape[1]); transformed = np.stack([inverse_cost(x,y) for x,y in zip(a,w)])
        if family == 'l2': transformed = (transformed[:5]+transformed[5:])/2
        means.append(transformed.mean(0))
    need(len(ns) == 1, 'family cohort mismatch')
    return ml.rank(sum(means)/3)


def oof_values(raw, costs):
    need(set(raw) == set(costs) == set(range(5)), 'five heldout folds required')
    parts, fam = [], {k:[] for k in ('R','D','C')}
    for f in range(5):
        expected = {'R42','R43','DINO','CNX'}
        need(set(raw[f]) == set(costs[f]) == expected, 'OOF member coverage')
        for a in raw[f].values(): probabilities(a)
        need(len({a.shape for a in raw[f].values()}) == 1, 'OOF row shape mismatch')
        r = np.stack([raw[f][k] for k in ('R42','R43')]).mean(0,dtype=np.float32)
        for k,a in zip(('R','D','C'),(r,raw[f]['DINO'],raw[f]['CNX'])): fam[k].append(a)
        r = np.mean([inverse_cost(raw[f][k],costs[f][k]) for k in ('R42','R43')],axis=0)
        parts.append((r+inverse_cost(raw[f]['DINO'],costs[f]['DINO'])+inverse_cost(raw[f]['CNX'],costs[f]['CNX']))/3)
    q0 = ml.rank(sum(ml.rank(np.concatenate(fam[k])) for k in ('R','D','C'))/3)
    return q0, ml.rank(np.concatenate(parts))


def gate(delta):
    d = np.asarray(delta, float)
    need(d.shape == (4,12) and np.isfinite(d).all(), 'gate finite complete 4x12 required')
    return bool((d.mean(1) >= .0005-1e-12).sum() >= 3 and np.all(d >= -.003-1e-12))


def unpack_parent():
    data = (STAGE/'kernel/release.ipynb').read_bytes()
    need(sha(data) == NOTEBOOK_SHA, 'parent notebook pin')
    nb = json.loads(data); nodes = ast.walk(ast.parse(''.join(nb['cells'][2]['source'])))
    node = next(x for x in nodes if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id == 'BLOB' for y in x.targets))
    blob = base64.b64decode(ast.literal_eval(node.value))
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        need(len(z.namelist()) == len(set(z.namelist())) == 31, 'parent member count')
        files = {n:z.read(n) for n in z.namelist()}
    prior = read(STAGE/'new-members.json')
    need({x['name']:x['after_sha256'] for k in ('changed','unchanged') for x in prior[k]} ==
         {n:sha(b) for n,b in files.items()}, 'full embedded inventory')
    need(sha(files['q-a5q-manifest.json']) == QSHA, 'Q manifest pin')
    # Extraction helper delegates only pure math to these current namespaces.
    for name in ('exp169_release_c85.py','exp169_release_v5q.py','exp169_release_p956.py',
                 'exp169_release_v2.py','exp169_partner_bundle_release.py'):
        need((ROOT/'tools'/name).read_bytes() == files[name], 'math helper differs from embedded parent: '+name)
    return files


def authenticate():
    bindings = {}
    def bound(path, expected=None):
        path = Path(path); path = path if path.is_absolute() else ROOT/path
        key = relative(path); data = path.read_bytes(); h = sha(data)
        need(expected is None or h == expected, 'input pin drift: '+key)
        bindings[key] = h
        return data
    def obj(path, expected=None): return json.loads(bound(path,expected))
    historical = obj(QPOOL/'results.json')
    bound(QPOOL/'REGISTRATION.md', historical['registration_sha256'])
    bound(QPOOL/'analyze.py', historical['script_sha256'])
    for rel,h in {**historical['bindings'],**historical['context_bindings']}.items(): bound(rel,h)
    for rel,pin in historical['source_bindings'].items(): bound(rel,pin['sha256'])
    raw_all, rosters, frozen, frozen_pin = aligned.load_frozen()
    bound(ROOT/aligned.EXP/'input-source-manifest.json',frozen_pin['sha256'])
    files = unpack_parent(); entries = json.loads(files['q-a5q-manifest.json'])['entries']; validate_entries(entries)
    records = historical['parent_members']; validate_entries([x['parent'] for x in records])
    need([member_tuple(e) for e in entries] == [member_tuple(x['parent']) for x in records], 'OOF/source/deploy tuple mismatch')
    gate_reg = obj(GATE_REG)
    for name,h in gate_reg['kernel_files'].items(): bound(STAGE/'kernel'/name,h)
    bound(STAGE/'new-members.json')
    raw = {f:{} for f in range(5)}; costs = {f:{} for f in range(5)}
    serving_costs = {f:[] for f in FAMILIES}; members = []
    for record,e in zip(records,entries):
        directory = (ROOT/e['source'].replace('\\','/')).parent
        inv = obj(directory/('receipt.json' if e['family'] == 'dino' else 'output-inventory.json'))
        if e['family'] == 'dino':
            need(inv['status'] == 'PASS' and inv['phase'] == 'full', 'DINO receipt incomplete'); inv = inv['outputs']
        need(inv['model_best.pt']['sha256'] == e['source_sha256'], 'source checkpoint ancestry')
        prov = obj(directory/'provenance.json',inv['provenance.json']['sha256'])
        summary = obj(directory/'summary.json',inv['summary.json']['sha256'])
        f = e['fold']; roster = rosters[f]
        info = dict(train_sha256=aligned.serving.uid_hash(roster['train']), sorted_train_sha256=aligned.serving.uid_hash(sorted(roster['train'])), val_sha256=aligned.serving.uid_hash(roster['val']))
        ea.validate_summary({**summary,'provenance':prov},info,len(roster['val']),f)
        key = 'R'+str(e['seed']) if e['family']=='l2' else 'DINO' if e['family']=='dino' else 'CNX'
        need(record['key'] == key and record['fold'] == f, 'record fold/key')
        w = np.asarray(prov['actual_pos_weight'],float)
        inverse_cost(np.zeros((1,12)),w)
        need(np.array_equal(w,record['actual_pos_weight']), 'recorded cost differs from provenance')
        need(prov.get('loss_policy',prov.get('loss')) == record['loss_policy'] and record['loss_policy'] in ('BCEWithLogitsLoss','clipped_positive_term'), 'loss semantics')
        path = ROOT/record['prediction']; bound(path,historical['bindings'][relative(path)])
        raw[f][key] = aligned.load_csv(path,roster['val']); probabilities(raw[f][key])
        if key.startswith('R'): need(np.array_equal(raw[f][key],raw_all[f][key]), 'aligned R50 differs')
        costs[f][key] = w; serving_costs[e['family']].append(w)
        members.append(dict(identity=e, prediction=relative(path), provenance=relative(directory/'provenance.json'), actual_pos_weight=w, loss_policy=record['loss_policy']))
    ids = sum((r['val'] for r in rosters),[])
    need(len(ids) == len(set(ids)) == 4343, 'heldout unique4343')
    routing = obj(ROUTING)
    worker = obj(COMMON/'worker.json',routing[relative(COMMON/'worker.json')])
    need(worker['state'] == 'INFERENCE_COMPLETE' and worker['studies'] == 58, 'common incomplete')
    creg = obj(COMMON_REG,worker['registration_sha256'])
    need([member_tuple(e) for e in creg['q_entries']] == [member_tuple(e) for e in entries], 'serving leaf identity/order')
    bound(ROOT/'tools/hb167_common_gold58_v1.py',creg['pins']['tools/hb167_common_gold58_v1.py'])
    archives = []
    for name,key in (('predictions.npz','predictions_sha256'),('q-leaves.npz','q_leaves_sha256')):
        data = bound(COMMON/name,routing[relative(COMMON/name)])
        need(sha(data) == worker[key], 'worker output hash')
        with np.load(io.BytesIO(data),allow_pickle=False) as z: archives.append({k:z[k].copy() for k in z.files})
    common, leaves = archives; gids = common['ids'].tolist()
    need(gids == creg['ids'] == leaves['ids'].tolist() and len(set(gids)) == 58 and
         common['targets'].tolist() == leaves['targets'].tolist() == list(ml.TARGETS), 'common roster/targets')
    need(set(leaves) == {'ids','targets',*FAMILIES}, 'leaf columns')
    leaves = {f:leaves[f] for f in FAMILIES}
    for f in FAMILIES:
        need(leaves[f].dtype == np.float32 and leaves[f].shape == (10 if f=='l2' else 5,58,12), 'leaf shape/dtype')
        for a in leaves[f]: probabilities(a,58)
    for name in ('R','R224','EXP143v2','EXP155','Q_A5q'):
        probabilities(common[name],58); mask = common['failed_'+name]
        need(mask.shape == (58,) and mask.dtype == bool and not mask.any(), 'observed fallback/invalid mask')
    r6receipt = obj(R6/'controller/submission_receipt.json',routing[relative(R6/'controller/submission_receipt.json')])
    need(r6receipt['status']=='PASS' and r6receipt['rows']==r6receipt['scored']==58 and r6receipt['fallback_studies']==0 and
         r6receipt['fallback_reasons']=={} and r6receipt['worker_exit_codes'] and all(v==0 for v in r6receipt['worker_exit_codes'].values()), 'R6 failed receipt')
    data = bound(R6/'predictions.csv',routing[relative(R6/'predictions.csv')])
    frame = pd.read_csv(io.BytesIO(data),dtype={ml.UID:str},float_precision='round_trip')
    need(frame.columns.tolist() == [ml.UID,*ml.TARGETS], 'R6 target order')
    common['S_R7'] = align(frame[ml.UID].tolist(),frame[list(ml.TARGETS)].to_numpy(float),gids)
    bound(ROOT/'storage/competition/train.csv',routing['storage/competition/train.csv'])
    for rel in ('tools/c85r7_r6w958q_v1139.py','tools/exp169_release_v2.py','tools/exp169_partner_bundle_release.py',
                'tools/hb197_inf3_bridge_v1.py','tools/test_hb197_inf3_bridge_v1.py',
                '.local/scratch/slot-b/batch5-1018/SLATE-REVIEW.md'):
        bound(rel)
    bound(OUT/'SPEC.md')
    for rel,pin in aligned.source_pins().items(): bound(rel,pin['sha256'])
    proof = dict(version=VERSION, candidate='R6W958-INF3-V1',exposure='EXPOSED',clean_oof=False,
                 bindings=bindings, members=members, parent_embedded={n:sha(b) for n,b in files.items()},
                 oof_roster_sha256=sha(encoded(ids)), gold_roster_sha256=sha(encoded(gids)), targets=list(ml.TARGETS),
                 oof_rows=4343,gold_rows=58,checkpoint_authentication='source/deploy manifests and provenance receipts, no tensor loads or new checkpoint hashing',
                 bootstrap=dict(draws=ml.DRAWS,seed=ml.SEED,oof_unit='connected component',gold_unit='study singleton',conditional=True),
                 gate=dict(macro=.0005,min_rulers=3,target_floor=-.003,tolerance=1e-12),
                 environment=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__))
    return ml.clean(proof), (raw,costs,rosters,ids,common,leaves,{f:np.asarray(v) for f,v in serving_costs.items()},gids,files,historical)


def register():
    proof,_ = authenticate(); save('registration.json',proof)
    return dict(status='REGISTERED_CPU_ONLY',registration_sha256=ml.sha(OUT/'registration.json'),members=20)


def validate():
    saved = read(OUT/'registration.json')
    current,data = authenticate()
    need(current == saved, 'registered identity drift')
    return current,data


def oof_report(ctx, a, b):
    d = ctx.delta(a,b); ca,cb = ctx.score(a),ctx.score(b)
    need(np.isfinite(d[0]).all(), 'point metric support')
    return dict(gate_pass=gate(d[0]), qualifying_rulers=int((d[0].mean(1)>=.0005-1e-12).sum()),min_target_delta=float(d[0].min()),
        primary_weak_four=ml.stat(d[:,:2][:,:,ml.WEAK].mean((1,2))),
        rulers={r:dict(control_macro=float(cb[0,i].mean()),arm_macro=float(ca[0,i].mean()),
            macro_delta=ml.stat(d[:,i].mean(1)),weak_four=ml.stat(d[:,i,ml.WEAK].mean(1)),
            targets={t:dict(control_auc=float(cb[0,i,j]),arm_auc=float(ca[0,i,j]),delta=ml.stat(d[:,i,j])) for j,t in enumerate(ml.TARGETS)}) for i,r in enumerate(ml.RULERS)})


def replay(data):
    raw,costs,rosters,ids,common,leaves,serving_costs,gids,files,historical = data
    q0,q3 = oof_values(raw,costs)
    ctx = ml.make_context(ids,ROOT); primary = oof_report(ctx,q3,q0)
    for r in ml.RULERS:
        need(abs(primary['rulers'][r]['macro_delta']['value']-historical['primary']['INF3']['rulers'][r]['macro_delta']['value'])<1e-12, 'original proxy point reproduction')
    folds = {}; offset = 0
    for f,roster in enumerate(rosters):
        ix = slice(offset,offset+len(roster['val'])); offset += len(roster['val'])
        folds[str(f)] = oof_report(ml.make_context(roster['val'],ROOT),q3[ix],q0[ix])
    cachedq,_ = qparent.aggregate(leaves)
    need(np.array_equal(cachedq,common['Q_A5q']), 'actual Q0 cache reconstruction differs')
    newq = serving_q3(leaves,serving_costs)
    ns = parent.math_functions(files)
    args = [common[name] for name in NAMES]
    baseline = ns['blend'](gids,*args)
    need(np.array_equal(baseline,ns['reference'](*args)), 'parent healthy oracle')
    replacement = args.copy(); replacement[-1] = cachedq
    noop = ns['blend'](gids,*replacement)
    need(baseline.tobytes() == noop.tobytes() and ns['p'].csv_bytes(gids,baseline) == ns['p'].csv_bytes(gids,noop), 'parent no-op bytes')
    replacement[-1] = newq; candidate = ns['blend'](gids,*replacement)
    need(np.array_equal(candidate,ns['reference'](*replacement)), 'candidate unchanged outer oracle')
    values = {}; csv_hashes = {}
    for name,a in (('F0',baseline),('F3',candidate)):
        payload = ns['p'].csv_bytes(gids,a); df = pd.read_csv(io.BytesIO(payload),dtype={ml.UID:str},float_precision='round_trip')
        values[name] = align(df[ml.UID].tolist(),df[list(ml.TARGETS)].to_numpy(float),gids)
        need(np.array_equal(values[name].astype(np.float32),a.astype(np.float32)), 'parent CSV roundtrip')
        csv_hashes[name] = sha(payload)
    label_ids,y = gold.gold(); y = align(label_ids,y,gids)
    weights = gold.bootstrap_weights(gids)
    scores = gold.auc_table({**values,'Q0':cachedq,'Q3':newq},y,weights)
    gold_result = dict(scope='EXPOSED_DESCRIPTIVE_ONLY',gate='NONE; independent review required',
        parent=gold.block(scores['F0']),candidate=gold.block(scores['F3']),delta=gold.block(scores['F3']-scores['F0']),
        q_delta=gold.block(scores['Q3']-scores['Q0']),csv_sha256=csv_hashes,
        parent_noop_float64_and_csv_exact=True,all20_post_tta_leaves_authenticated=True,observed_prediction_fallbacks=0,
        changed_final_float32_cells=int(np.count_nonzero(values['F0'].astype(np.float32)!=values['F3'].astype(np.float32))),
        bootstrap='300 paired study-singleton draws seed20261003, full58 ranks frozen; conditional, fixed target set')
    # Independently verify every point AUC used in primary and Gold results.
    from sklearn.metrics import roc_auc_score
    maxerr = 0.
    for a in (q0,q3):
        score = ctx.score(a)[0]
        for i,r in enumerate(ml.RULERS):
            yy,mask,_ = ctx.refs[r]
            for j in range(12): maxerr=max(maxerr,abs(score[i,j]-roc_auc_score(yy[mask[:,j],j],a[mask[:,j],j])))
    for name,a in {**values,'Q0':cachedq,'Q3':newq}.items():
        for j in range(12): maxerr=max(maxerr,abs(scores[name][0,j]-roc_auc_score(y[:,j],a[:,j])))
    need(maxerr < 1e-12, 'independent sklearn AUC mismatch')
    return ml.clean(dict(status='COMPLETE_EXPOSED_CPU_BRIDGE',version=VERSION,proxy_gate='PASS' if primary['gate_pass'] else 'NO_GO',
        release='HOLD_SEPARATE_RESULT_REVIEW_AND_NEW_PATH_QUALIFICATION',primary=primary,fold_diagnostics=folds,gold58=gold_result,
        identity=dict(oof_rows=4343,serving_rows=58,targets=list(ml.TARGETS),member_count=20),
        independent_auc_checks=144,max_auc_error=maxerr,context_bindings=ctx.bindings,
        seven_lenses=dict(L1='OOF weak4 and exposed Gold weak4 reported; no new gate',L2='original four-ruler macro gate',
            L3='four shared-patient ruler results; not independent panels',L4='all48 original target floors plus descriptive Gold12',
            L5='unavailable: no independent same-recipe paired seed comparison',L6='exact exposed Gold58 arithmetic only; no heldout full-F4343',
            L7='Gold58 image labels exposed/descriptive, no eligible independent OAI confirmation'),
        limitations=['EXPOSED reused development/rulers/model ancestry; clean_oof=false.',
            'OOF generator-heldout Q surrogate differs from all-fivefold unseen-study serving.',
            'Inverse weighted-BCE correction is not proven calibration.',
            'Conditional CIs omit training/selection uncertainty; Gold singleton component assumption is diagnostic.',
            'No release integration, CUDA/model forward, visible/N32, runtime or private-score claim.']))


def run(check=False):
    proof,data = validate()
    if not check:
        need(not (OUT/'run-start.json').exists(), 'fixed comparison already started; use check for read-only replay')
        save('run-start.json',dict(registration_sha256=ml.sha(OUT/'registration.json'),version=VERSION))
    result = replay(data)
    for p,h in proof['bindings'].items(): need(ml.sha(ROOT/p)==h, 'input changed during computation: '+p)
    result['registration_sha256'] = ml.sha(OUT/'registration.json')
    if check: need(result == read(OUT/'results.json'), 'fixed replay differs')
    else: save('results.json',result)
    return dict(status='CHECK_PASS' if check else result['status'],proxy_gate=result['proxy_gate'],release=result['release'],
                results_sha256=ml.sha(OUT/'results.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','validate','run','check'))
    args=parser.parse_args()
    try:
        result = register() if args.command=='register' else dict(status='VALIDATE_PASS',members=len(validate()[0]['members'])) if args.command=='validate' else run(args.command=='check')
        print(json.dumps(result,sort_keys=True))
    except (ValueError,KeyError,OSError) as exc:
        print(json.dumps(dict(status='HOLD',reason=str(exc)),sort_keys=True)); return 2
    return 0


if __name__ == '__main__': raise SystemExit(main())
