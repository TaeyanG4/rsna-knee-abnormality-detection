"""Syn-only frozen-feature experiment v1.0.2.

Changelog v1.0.2: separate reviewed operational pins preserve preregistration bytes.
Changelog v1.0.1: pre-result implementation amendment: authenticate carry-over,
pin evaluator dependency, bind extraction review and checkpoint. Rule unchanged.
Changelog: preregistered train-only soft-stratum BCE, anchored linear readout,
exact977 paired trio gate. No encoder training, sweep, or release-stage mutation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.local/scratch/astra-b4/syn-head-v1'
PARENT = '.local/scratch/l1-rco/k1v3-grouping'
TRAINER = 'tools/exp189_s6_src/compact_train.py'
CHECKPOINT = PARENT + '/fit/models/E_prior_s0_f0.pt'
CHECKPOINT_SHA = '6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5'
TRAINER_SHA = '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80'
BASE = '.local/scratch/slot-b/k1-stride-screen'
BASE_CHECKPOINTS = {
    '.local/scratch/l1-rco/k1v2/fit/models/E_prior_s0_f0.pt': '2c3c033e25605b5fb8bcc08e0030bed081c726e10fc56d8da62524ac08d9638c',
    '.local/scratch/astra-b2/h210-recovery-v1/attempt/fit/models/E_prior_s0_f0.pt': '29fc2b2bbaa4e4e8ad341bba17e47c5110388db697458a8cbb9ff8d83fa9399b'}
SYN, EFF = 8, 7
RULE = dict(version='1.0.0', member='k1_v3g_seed0', train_n=3190, holdout_n=977,
    feature='out_norm[:,8,:], frozen 384-dimensional trained-head interface',
    bins=[0., 1/3, 2/3, 1.], bin_boundary='left closed; 1 included in last bin',
    loss='mean over 3 bins of 0.5*sum(y*softplus(-z))/sum(y) + 0.5*sum((1-y)*softplus(z))/sum(1-y)',
    regularizer='0.01/2 * squared L2 distance of weights AND bias from trained Syn head',
    optimizer=dict(method='L-BFGS-B', maxiter=1000, gtol=1e-7, ftol=1e-12),
    train_bags=dict(count=1, seed=0, epoch=0, windows=64, sampling='cover', cover_stride=3, slice_jitter=.25),
    eval_stride=2, slices=3, phys_mm=3.5, crop=[16,368], chunk=32,
    inference='eval mode, FP16 CUDA autocast, TF32 false; CPU smoke FP32 only',
    operational_seed0_parity_max_abs=.002,
    no_effusion_input=True, other_columns='bit identical to paired trained-head output',
    go=dict(rulers=['jev_hybrid','aquila_v1','dread','consensus'], nonnegative_rulers=3,
            mean_syn_delta=.006, mean_macro_delta=.0005, effusion_unchanged=True),
    baseline='paired fresh seed0 trained head; fixed hash-bound stride2 v2 and seed1 carry-over',
    trio='per-target average-tie percentile ranks, mean of three members, rerank',
    exposure='EXPOSED_EXPLORATORY; reused report priors/rulers; not clean meta-CV or P25 evidence',
    selection='one seed0 head, no sweep, no holdout selection, no seed1 extension')


def need(ok, message):
    if not ok: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def replace(src, dst):
    for attempt in range(40):
        try: os.replace(src, dst); return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.1)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    replace(temp, path)


def npz(path, **arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream: np.savez(stream, **arrays)
    replace(temp, path)


def source_names():
    return ['tools/k1_syn_head_v1.py', 'tools/k1_syn_runtime_v1.py', TRAINER,
            'tools/job_runtime.py', 'tools/gpu_job.py', 'tools/gpu_queue.py',
            'tools/s6plus.py', 'training/src/raptor_common.py', 'config/workspace.toml',
            'tests/test_k1_syn_head_v1.py']


def baseline_provenance(holdout):
    from tools import job_runtime as jobs
    from tools import l1_rco_k1_eval_v1 as ev
    job = read(ROOT/'artifacts/logs/K1_STRIDE_SCREEN_V1/job.json')
    receipt = read(ROOT/'artifacts/logs/K1_STRIDE_SCREEN_V1/receipt.json')
    cmd = job['command']
    need(receipt['state']=='SUCCESS' and receipt['exit_code']==0 and
         receipt['run_id']==job['run_id']=='8c52a77fa1c540df8e3bdca6edbc27f9' and
         receipt['command_sha256']==job['command_sha256']==jobs.command_digest(cmd,ROOT), 'completed baseline job identity')
    need(len(cmd)==4 and cmd[1]=='-B' and Path(cmd[2]).resolve()==(ROOT/BASE/'stride_screen_v1.py').resolve() and cmd[3]=='run', 'baseline command identity')
    for name,pin in BASE_CHECKPOINTS.items(): need(sha(ROOT/name)==pin, 'baseline checkpoint identity')
    outputs = {}
    for member in ('v2','s0','s1'):
        name = BASE+f'/preds/{member}_s2.csv'
        ids, _ = ev.predictions(ROOT/name,holdout)
        need(ids==holdout, 'baseline must cover exact977')
        outputs[name] = sha(ROOT/name)
    return dict(run_id=job['run_id'], receipt_sha256=sha(ROOT/'artifacts/logs/K1_STRIDE_SCREEN_V1/receipt.json'),
        checkpoints=BASE_CHECKPOINTS, predictions=outputs,
        attestation='Retrospective pre-result carry-over binding of current files to completed named job and current checkpoint identities; historical receipt did not hash predictions.')


def register():
    if (OUT/'OPERATIONAL-PINS-v102.json').exists():
        validate()
        return dict(status='REGISTERED', train_n=3190, holdout_n=977,
                    registration_sha256=sha(OUT/'PREREGISTRATION.json'))
    parent = read(ROOT / PARENT / 'registration.json')
    need(parent['contract_id'] == 'a58f1476ea4ca3c8d64fe06c80bc6e24d78fdf2f347c4de68c1eba5d9c2ea83d', 'seed0 parent identity')
    need(sha(ROOT / CHECKPOINT) == CHECKPOINT_SHA and sha(ROOT / TRAINER) == TRAINER_SHA, 'frozen input identity')
    train, hold = parent['train'], parent['holdout']
    need(len(train) == len(set(train)) == 3190 and len(hold) == len(set(hold)) == 977 and not set(train)&set(hold), 'exact disjoint roster')
    names = source_names() + [CHECKPOINT, PARENT+'/registration.json', PARENT+'/prepare.json', PARENT+'/targets.csv',
        BASE+'/stride_screen_v1.py', BASE+'/preds/v2_s2.csv', BASE+'/preds/s0_s2.csv', BASE+'/preds/s1_s2.csv',
        'artifacts/logs/K1_STRIDE_SCREEN_V1/receipt.json', 'artifacts/logs/K1_STRIDE_SCREEN_V1/job.json', *BASE_CHECKPOINTS]
    # Evaluator code and structured ruler tables are bound, but never printed.
    from tools import exp156_k_sweep as r
    names += ['tools/l1_rco_k1_eval_v1.py', 'tools/l1_rco_stage0_v1.py', 'tools/exp156_k_sweep.py', 'tools/exp115_vlm_eval.py']
    names += sorted(set([r.JEV, r.AQUILA, r.DREAD, *r.CONSENSUS]))
    reg = dict(schema='k1-syn-head/v1', rule=RULE, train=train, holdout=hold,
        pins={n:sha(ROOT/n) for n in names}, carry_over=baseline_provenance(hold), clean_oof=False)
    path = OUT/'PREREGISTRATION.json'
    if path.exists(): need(read(path) == reg, 'immutable preregistration drift; version a new experiment')
    else:
        need(not (OUT/'head.json').exists() and not (OUT/'RESULT.json').exists(), 'register before results')
        write(path, reg)
    validate()
    return dict(status='REGISTERED', train_n=3190, holdout_n=977, registration_sha256=sha(path))


def operational_pins(reg):
    path = OUT/'OPERATIONAL-PINS-v102.json'
    if not path.exists(): return reg['pins']
    amendment = read(path)
    need(amendment.get('schema') == 'k1-syn-operational/v1.0.2' and
         amendment.get('registration_sha256') == sha(OUT/'PREREGISTRATION.json') ==
         'bed44f57c9c3ddaa4c4c42f5243563bae1773cada6e675271931e5119a1195b2', 'operational registration binding')
    allowed = {'tools/k1_syn_head_v1.py', 'tools/k1_syn_runtime_v1.py', 'tests/test_k1_syn_head_v1.py'}
    replacements = amendment['replacements']
    need(set(replacements) == allowed, 'operational replacement allowlist')
    pins = reg['pins'].copy()
    for name, change in replacements.items():
        need(change['before'] == pins[name] and len(change['after']) == 64, 'operational original pin')
        pins[name] = change['after']
    return pins


def validate(remote=False):
    reg = read(OUT/'PREREGISTRATION.json')
    need(reg['rule'] == RULE, 'scientific rule drift')
    need(len(reg['train']) == len(set(reg['train'])) == 3190 and len(reg['holdout']) == len(set(reg['holdout'])) == 977 and not set(reg['train'])&set(reg['holdout']), 'roster drift')
    remote_names = set(source_names()+[CHECKPOINT, PARENT+'/registration.json', PARENT+'/prepare.json', PARENT+'/targets.csv'])
    for name, pin in operational_pins(reg).items():
        if not remote or name in remote_names: need(sha(ROOT/name) == pin, 'registered pin mismatch: '+name)
    parent = read(ROOT/PARENT/'registration.json')
    need(reg['train'] == parent['train'] and reg['holdout'] == parent['holdout'], 'parent roster identity')
    need(sha(ROOT/PARENT/'targets.csv') == read(ROOT/PARENT/'prepare.json')['targets_sha256'], 'prepared training targets identity')
    training_targets(reg)
    if not remote: need(reg['carry_over']==baseline_provenance(reg['holdout']), 'baseline carry-over drift')
    return reg


def training_targets(reg):
    import numpy as np
    with (ROOT/PARENT/'targets.csv').open(newline='', encoding='utf-8') as stream: rows = list(csv.DictReader(stream))
    ids = [r['StudyInstanceUID'] for r in rows]
    need(len(ids) == len(set(ids)) == len(reg['train']) and set(ids) == set(reg['train']) and not set(ids)&set(reg['holdout']), 'targets must contain train only')
    by = dict(zip(ids, rows)); y = np.array([float(by[u]['Synovitis|prior']) for u in reg['train']])
    eff = np.array([float(by[u]['Effusion|prior']) for u in reg['train']])
    need(np.isfinite([y,eff]).all() and ((y>=0)&(y<=1)&(eff>=0)&(eff<=1)).all(), 'invalid soft priors')
    return y, eff


def soft_weights(y, eff):
    import numpy as np
    y, eff = np.asarray(y), np.asarray(eff)
    need(y.shape == eff.shape and y.ndim == 1 and np.isfinite([y,eff]).all() and ((y>=0)&(y<=1)&(eff>=0)&(eff<=1)).all(), 'invalid training labels')
    bins = np.searchsorted(RULE['bins'][1:-1], eff, side='right')
    pos, neg, summary = np.zeros_like(y,dtype=float), np.zeros_like(y,dtype=float), []
    for b in range(3):
        take = bins == b; p, n = y[take].sum(), (1-y[take]).sum()
        need(p > 0 and n > 0, 'empty stratum or absent positive/negative soft mass')
        pos[take] = y[take]/(6*p); neg[take] = (1-y[take])/(6*n)
        summary.append(dict(bin=b, n=int(take.sum()), positive_mass=float(p), negative_mass=float(n)))
    return pos, neg, summary


def fit_head(features, y, eff, initial):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    x = np.asarray(features, dtype=np.float64); theta0 = np.asarray(initial, dtype=np.float64)
    need(x.ndim == 2 and x.shape == (len(y),384) and theta0.shape == (385,) and np.isfinite(x).all() and np.isfinite(theta0).all(), 'feature/head shape')
    x = np.column_stack([x, np.ones(len(x))]); pos, neg, bins = soft_weights(y, eff)
    def objective(theta):
        z = x@theta; d = theta-theta0
        loss = np.sum(pos*np.logaddexp(0,-z) + neg*np.logaddexp(0,z)) + .005*(d@d)
        grad = x.T@((pos+neg)*expit(z)-pos) + .01*d
        return loss, grad
    o = RULE['optimizer']
    fit = minimize(objective, theta0, jac=True, method=o['method'], options={k:o[k] for k in ('maxiter','gtol','ftol')})
    need(fit.success and np.isfinite(fit.x).all(), 'head optimizer did not converge; no alternate fit')
    return dict(theta=fit.x.tolist(), objective=float(fit.fun), iterations=int(fit.nit), bins=bins,
                max_abs_gradient=float(np.abs(fit.jac).max()), converged=True)


def replace_syn(pred, features, theta):
    import numpy as np
    from scipy.special import expit
    out = np.asarray(pred).copy(); x, t = np.asarray(features), np.asarray(theta)
    need(out.shape == (len(x),12) and x.shape[1:] == (384,) and t.shape == (385,) and np.isfinite(x).all() and np.isfinite(t).all(), 'replacement shapes')
    out[:, SYN] = expit(x@t[:384]+t[384])
    need(np.isfinite(out).all() and ((out>=0)&(out<=1)).all(), 'invalid probabilities')
    need(np.array_equal(out[:,np.arange(12)!=SYN],pred[:,np.arange(12)!=SYN]), 'other targets changed')
    return out


def ranks(values):
    from scipy.stats import rankdata
    return (rankdata(values, axis=0, method='average')-1)/max(1,len(values)-1)


def trio(v2, seed0, seed1):
    return ranks((ranks(v2)+ranks(seed0)+ranks(seed1))/3)


def go_rule(metrics, unchanged):
    import math
    from tools.l1_rco_k1_eval_v1 import TARGETS
    rows, names = metrics.get('per_target', []), set(RULE['go']['rulers'])
    if not unchanged or len(rows)!=48 or {(r['ruler'],r['target']) for r in rows}!={(n,t) for n in names for t in TARGETS}: return False
    if any(not all(isinstance(r.get(k),(int,float)) and math.isfinite(r[k]) for k in ('rco_auc','r7_auc','delta')) for r in rows): return False
    syn = [r['delta'] for r in rows if r['target']=='Synovitis']
    return (sum(d>=0 for d in syn)>=3 and sum(syn)/4>=.006 and
            all(r['delta']==0 for r in rows if r['target']!='Synovitis'))


def load_features(split, reg):
    import numpy as np
    receipt = read(OUT/'features/receipt.json')
    need(receipt['status']=='COMPLETE' and receipt['registration_sha256']==sha(OUT/'PREREGISTRATION.json'), 'feature receipt identity')
    need(receipt['checkpoint_sha256']==CHECKPOINT_SHA and receipt['review_sha256']==sha(OUT/'REVIEW.json'), 'feature checkpoint/review identity')
    path = OUT/'features'/f'{split}.npz'
    need(sha(path)==receipt['outputs'][split], 'feature payload hash')
    with np.load(path,allow_pickle=False) as data: a = {k:data[k] for k in data.files}
    need(a['ids'].tolist()==reg[split] and a['features'].shape==(len(reg[split]),384) and a['predictions'].shape==(len(reg[split]),12), 'feature roster/layout')
    need(a['initial'].shape==(385,) and all(np.isfinite(a[k]).all() for k in ('features','predictions','initial')), 'nonfinite features')
    return a


def fit():
    from tools.k1_syn_runtime_v1 import require_review
    reg = validate(); require_review()
    need(not (OUT/'head.json').exists(), 'head already fitted; no silent refit')
    train = load_features('train', reg); y, eff = training_targets(reg)
    result = fit_head(train['features'],y,eff,train['initial'])
    result.update(schema='k1-syn-linear/v1', registration_sha256=sha(OUT/'PREREGISTRATION.json'),
                  feature_receipt_sha256=sha(OUT/'features/receipt.json'), member_checkpoint_sha256=CHECKPOINT_SHA,
                  target_index=SYN, clean_oof=False, train_n=3190)
    write(OUT/'head.json',result)
    return dict(status='FITTED',n=3190,iterations=result['iterations'],objective=result['objective'])


def evaluate():
    import numpy as np
    from tools import l1_rco_k1_eval_v1 as ev
    from tools import exp156_k_sweep as rulers
    from tools.k1_syn_runtime_v1 import require_review
    reg = validate(); require_review(); hold = load_features('holdout',reg); head = read(OUT/'head.json')
    need(head['registration_sha256']==sha(OUT/'PREREGISTRATION.json') and head['feature_receipt_sha256']==sha(OUT/'features/receipt.json') and head['member_checkpoint_sha256']==CHECKPOINT_SHA and head['target_index']==SYN, 'head provenance')
    fixed = {}
    for name in ('v2','s0','s1'):
        present, values = ev.predictions(ROOT/BASE/f'preds/{name}_s2.csv',reg['holdout'])
        need(present==reg['holdout'], 'carry-over exact977 identity'); fixed[name]=values
    # This is an operational parity gate, not an AUC/model-selection gate.
    parity = float(np.max(np.abs(hold['predictions']-fixed['s0'])))
    need(parity<=RULE['operational_seed0_parity_max_abs'], 'fresh FP16 seed0 vs serving carry-over parity failed; investigate, do not retune')
    changed = replace_syn(hold['predictions'],hold['features'],head['theta'])
    baseline = trio(fixed['v2'],hold['predictions'],fixed['s1']); candidate = trio(fixed['v2'],changed,fixed['s1'])
    unchanged = bool(np.array_equal(candidate[:,np.arange(12)!=SYN],baseline[:,np.arange(12)!=SYN]))
    metrics = ev.metrics(candidate,baseline,rulers.references(reg['holdout'],root=ROOT))
    syn = [r for r in metrics['per_target'] if r['target']=='Synovitis']
    result = dict(status='GO' if go_rule(metrics,unchanged) else 'NO_GO', n=977,
        syn_per_ruler=syn, mean_syn_delta=float(np.mean([r['delta'] for r in syn])),
        mean_macro_delta=float(np.mean([r['delta'] for r in syn]))/12, other_11_bit_identical=unchanged,
        baseline_parity_max_abs=parity, metrics=metrics, head_sha256=sha(OUT/'head.json'),
        registration_sha256=sha(OUT/'PREREGISTRATION.json'), clean_oof=False, qualified_for_submission=False,
        next_step='GO only: independent result review, P25-V10 Syn-only stage and T4 parity/qualification')
    write(OUT/'RESULT.json',ev.clean(result))
    return {k:result[k] for k in ('status','n','mean_syn_delta','mean_macro_delta','other_11_bit_identical')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['register','validate','prepare','admit-dry-run','run','worker','fit','evaluate','package'])
    p.add_argument('--host',choices=['local','vast'],default='local')
    a=p.parse_args()
    if a.action in ('register','fit','evaluate'): result=globals()[a.action]()
    elif a.action=='validate': validate(a.host=='vast'); result=dict(status='PASS',train_n=3190,holdout_n=977)
    else:
        from tools import k1_syn_runtime_v1 as rt
        result=getattr(rt,a.action.replace('-','_'))(a.host)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
