"""H-B-166 all-fold v1.2.2: preregistered replacement estimand and admission rule.

This module authenticates inputs and decision rules; the future scientific
five-fold evaluator is a separate reviewed implementation of this specification.
No OAI worker or panel inference is introduced by this training package.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.2.2'
MANIFEST = 'artifacts/experiments/HB105_ALIGNED_OOF_V1/input-source-manifest.json'
MANIFEST_SHA = 'f439d836e46eeac920146cfdf3d1b180a10075a61b3c7fb2b1d3e5de5eb35806'
ESTIMAND = 'Held-out-fold prediction replacement: H-B-166 two-seed mean versus authenticated deployed HB105 own-S two-seed mean; observational, not matched-recipe causal treatment.'
BIAS = 'EXPOSED / clean_oof=false. R7-derived labels, shared recipe development and fold2 selection prevent a clean holdout claim. Different seeds/recipes for the deployed reference; no paired treatment claim.'
DEPENDENCY = 'Purpose=measurement only. Before ANY CUDA smoke/fit: authenticate the unchanged completed fold3 NO_PASS/HOLD and NO_COMBINED_SUPPORT result; independent result-validity review PASS bound to its hash; explicit B ADMIT_FRESH_FOLD_MEASUREMENT with purpose=measurement, this registration and hypotheses hashes; independent CPU-bound pre-review PASS. Fit additionally requires full-shape CUDA smokes, measured memory and independent smoke-bound review. No historical outcome is a success prerequisite or rewritten; missing/stale/FAIL review means HOLD. No release admission.'
HISTORICAL_READOUT_SHA256 = 'f50f60e07f862c1940689cf5bde2d5585e2ced4135e89f0d15e248e4c9bb3166'
HISTORICAL_CANONICAL_SHA256 = 'b33e0f65fc2a9d2a84bceaa7ab896b935a0240c6947e0f4814f60c5dfb81895f'
MEASUREMENT_FOLDS = [0, 1, 4]
HARM_TARGETS = ['Fracture', 'Lateral Meniscus', 'Lateral OA', 'PF OA', 'Synovitis']
RULE = dict(name='FRESH_FOLD_MEASUREMENT', purpose='measurement', folds=MEASUREMENT_FOLDS,
    population='Exactly 2615 heldout UIDs from folds0/1/4 (888/864/863); folds2/3 excluded from every hypothesis estimate, rank population, bootstrap and decision. Five-fold results are descriptive only.',
    reference='Authenticated HB105 deployed own-S heldout two-seed mean; observational replacement, not a paired-recipe causal claim. Same fixed R50/C4/T partner families and per-fold members as historical L6 bundle; exact hashes in reference_members.',
    aggregation='Arithmetic probability mean of the two registered seeds within each fold. Concatenate heldout rows from folds0/1/4, then compute pooled AUC (not mean fold AUC or mean seed AUC). Primary labels are the fixed JEV+consensus arithmetic mean of AUC deltas, as historical L6. Same masks in both arms.',
    ranks='Rank each target separately with average ties over the fixed2615 UID population before bootstrap. Equal0.25 mean of R50,C4,T,S percentile ranks. No subset reranking. This pooled estimator and deployed-S comparator differ from the historical within-fold paired-control L6 estimator; not an unchanged-estimator replication.',
    ci='300 connected-component bootstrap draws, seed20261003, tools.multi_lens_readout.weights/auc_draws; shared weights for both routes, arms, rulers and targets. Restrict original components to fixed2615 UIDs and exclude empty components before drawing. Percentile95% interval from300 draws, point estimate unresampled; never rank bootstrap subsets. Undefined/nonfinite point or any required draw/statistic => HOLD.',
    hypotheses={
        'H1': dict(route='Replace all12 S columns with multiview; R50/C4/T unchanged.',
                   required=['H1_L6'], rule='Pooled JEV+consensus macro12 blend-proxy L6 delta >0 AND lower95% CI >0.'),
        'H2': dict(route='Copy deployed own-S predictions; replace only Contusion (index10) with multiview seed mean. All other S columns and all R50/C4/T predictions byte-identical. No routing/weight search.',
                   required=['H2_Contusion','H2_L6'], rule='Pooled JEV+consensus Contusion AUC delta of routed S versus deployed S >0 AND lower95% CI >0; routed equal-family blend macro12 L6 delta >0 AND lower95% CI >0; harm check must pass.')},
    harm='H2 only: no CI-excluding-zero harm (upper95% CI <0) for Fracture or any weak-four target (indices3/5/6/8: Lateral Meniscus, Lateral OA, PF OA, Synovitis), nor weak-four mean. Check routed S and routed L6 for JEV, consensus, dread, aquila individually and JEV+consensus primary; identical unchanged columns must produce exact zero deltas/draws. Nonfinite/undefined required harm intervals => HOLD; no exemption for nominally unchanged columns.',
    multiplicity='Exactly two selected confirmatory hypotheses, nominal95% intervals, no multiplicity adjustment; report both regardless of outcome and disclose familywise limitation. Per-fold/seed/other-target/five-fold analyses are descriptive and cannot rescue either hypothesis.',
    release='OUT_OF_SCOPE. Necessary future conditions: H1 or H2 PASS on fresh folds AND at least one independently reviewed, finite, route-compatible Gold58/OAI/public measurement is not HARM (CI upper >=0). Missing/unmapped/undefined evidence cannot qualify; UNRESOLVED is not positive validation. H2 requires Contusion-compatible measurement; OAI mapped-nine excludes Contusion and cannot qualify H2. This condition never grants release/serving/Kaggle permission; B must separately commission and review a release.')


def hypotheses_sha256():
    return hashlib.sha256(json.dumps(RULE,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()



def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def need(ok, why):
    if not ok:
        raise ValueError(why)


def require_measurement_history(result):
    """Authenticate frozen negative evidence; never relabel it or require success."""
    encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    need(hashlib.sha256(encoded).hexdigest() == HISTORICAL_CANONICAL_SHA256,
         'historical fold3 result missing/changed; preserve NO_PASS/HOLD')


def measurement_gate(statistics, harm):
    """Frozen summary gate only; scientific evaluator remains a reviewed future job."""
    def finite_interval(x):
        return (isinstance(x, (list, tuple)) and len(x) == 3
                and all(type(v) in (int, float) and math.isfinite(v) for v in x)
                and x[1] <= x[2])
    required = ('H1_L6', 'H2_Contusion', 'H2_L6')
    need(set(statistics) == set(required), 'exactly H1/H2 statistics required')
    harm_keys = {f'{route}/{ruler}/{target}' for route in ('S','L6')
                 for ruler in ('jev','consensus','dread','aquila','primary')
                 for target in (*HARM_TARGETS,'weak_four')}
    def gain(key):
        x = statistics[key]
        return ('HOLD' if not finite_interval(x) else
                'PASS' if x[0] > 0 and x[1] > 0 else 'NO_PASS')
    h1 = gain('H1_L6')
    h2_parts = [gain(k) for k in ('H2_Contusion','H2_L6')]
    harm_status = ('HOLD' if set(harm) != harm_keys or not all(finite_interval(x) for x in harm.values())
                   else 'HARM' if any(x[2] < 0 for x in harm.values()) else 'NOT_HARM')
    h2 = ('HOLD' if 'HOLD' in h2_parts or harm_status == 'HOLD' else
          'PASS' if h2_parts == ['PASS','PASS'] and harm_status == 'NOT_HARM' else 'NO_PASS')
    return dict(H1=h1, H2=h2, H2_harm=harm_status, release='OUT_OF_SCOPE')


def replacement_gate(macros, ci_lowers, targets, streams):
    """Pure preregistered gate, tested now; no scientific evaluation at build time."""
    if len(macros)!=4 or len(ci_lowers)!=4 or len(targets)!=12 or len(streams)!=2:
        return 'HOLD'
    values = [*macros,*ci_lowers,*targets,*streams]
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
        return 'HOLD'
    ok = (sum(macros)/4 >= .002-1e-12 and sum(v>0 for v in ci_lowers)>=2
          and min(targets)>=-.005-1e-12 and min(streams)>0)
    return 'SUPPORT' if ok else 'NO_SUPPORT'


def control_inventory():
    from tools import hb166_fold3_build_v1 as p
    out = {}
    for fold,stage,seeds in ((2,p.b155.WORK,(42,43)), (3,p.CONTROL_WORK,(44,45))):
        members=[]
        for seed in seeds:
            d=stage/'fits'/f'control-s{seed}'
            need(read(d/'heartbeat.json').get('status')=='complete', 'control is not complete')
            cp=stage/'configs'/f'control-s{seed}.json'; c=read(cp)
            need(c['fold']==fold and c['epochs']==20 and c['workers']==2 and c['arm']=='control', 'control recipe identity')
            members.append(dict(seed=seed, config=str(cp.relative_to(ROOT)).replace('\\','/'),
                config_sha256=sha(cp), val_uid_sha256=p.uid_hash(c['val']), epochs=20, k=36,flip=True,
                files={str((d/n).relative_to(ROOT)).replace('\\','/'):sha(d/n)
                    for n in ('summary.json','val.csv','val.receipt.json','runtime.json')}))
        out[str(fold)]=dict(status='EXISTING_MATCHED_RECIPE', source='H-B-155' if fold==2 else 'H-B-165',members=members)
    for fold,seeds in ((0,[46,47]),(1,[48,49]),(4,[50,51])):
        out[str(fold)]=dict(status='NO_MATCHED_CONTROL_IDENTIFIED', would_require_seeds=seeds,
            historical=f'EXP099 fold{fold}: CoAtNet/B8, 8 epochs, 336 corpus, accumulation1, workers12; not recipe-matched.',
            proposed='Do not train controls. Use authenticated deployed own-S OOF replacement comparator.')
    return dict(schema='hb166-allfold-controls/v1',folds=out,new_control_fits=0,
        optional_causal_design=dict(extra_control_fits=6,extra_hours=6*4700/3600,
            note='Needs separate preregistration/build; never silently substitute EXP099 historical runs.'),
        cheapest='Six new multiview fits only; existing HB105 deployed own-S heldout OOF comparator across all five folds.')


def make_spec():
    need(sha(ROOT/MANIFEST)==MANIFEST_SHA, 'HB105 authority changed')
    m=read(ROOT/MANIFEST)
    need(sha(ROOT/m['authority']['path'])==m['authority']['sha256'], 'HB105 serving authority changed')
    members=[]
    for x in m['members']:
        member=x['member']; pin=x['prediction_pin']; path=x['prediction']
        need(sha(ROOT/path)==pin['sha256'], 'reference prediction changed: '+path)
        members.append(dict(id=member['id'], family=member['family'],fold=member['fold'],seed=member['seed'],
            prediction=path,prediction_sha256=pin['sha256'],checkpoint_sha256=member['sha256'],
            heldout_uid_sha256=member['heldout_uid_sha256'],heldout_count=member['heldout_count']))
    need(len(members)==35 and sum(x['family']=='S' for x in members)==10, 'reference member coverage')
    return dict(schema='hb166-allfold-readout-spec/v1',version=VERSION,estimand=ESTIMAND,bias=BIAS,
        authority=dict(path=MANIFEST,sha256=MANIFEST_SHA,serving=m['authority']),reference_members=members,
        reference_evidence=m['bindings'],ruler_bindings=m['ruler_bindings'],
        seeds={'0':[46,47],'1':[48,49],'2':[42,43],'3':[44,45],'4':[50,51]},
        new_stage='.local/scratch/slot-b/hb166/allfold-v122',
        existing_stages={'2':'.local/scratch/slot-b/hb166/pilot-v100','3':'.local/scratch/slot-b/hb166/pilot-fold3-v110'},
        prediction_contract='final epoch20; val.csv K36 per view plus horizontal flip mean probabilities. Read only COMPLETE runs. Authenticate config/registration/prepared/runtime/checkpoint/prediction receipts; require 12 finite target columns in registered order, unique exact heldout IDs, no missing/extra rows or fallback. Align by UID, never row position.',
        population='Five EXP099 heldout folds after identical H-B-155 exclusions. Expected counts888/864/864/864/863; union4343, each UID exactly once, components do not cross train/val. Verify each new and reused fit roster against its fold before combining. Do not mix training predictions into OOF.',
        aggregation=RULE, admission_purpose='measurement', hypotheses_sha256=hypotheses_sha256(),
        L1_L4='Use multi_lens_readout_l4_v2 with multi_lens_readout.make_context; four rulers jev/consensus/dread/aquila. Same finite-label masks and target order in both arms, connected components, corrected L4. Record exact counts, ruler and exclusion pins. Fresh folds0/1/4 pooled primary only; five-fold and per-fold metrics descriptive only.',
        L5='INELIGIBLE for matched-seed replication: new seeds/recipe differ from deployed references. Report both seed-position streams as descriptive only; do not call them same-seed pairs. Preserve original paired fold2/3 L5 separately.',
        L6=RULE['ranks'] + ' ' + RULE['hypotheses']['H1']['rule'] + ' ' + RULE['hypotheses']['H2']['rule'],
        L7=dict(gold58=dict(rows=58,targets=12,k=36,flip=True,
            aggregation='Equal0.1 mean over all ten final checkpoints. Confirm existing fold2/3 gold.csv availability/completed receipt; if absent, B runs separately reviewed panel inference. Compare to deployed own-S ten-member panel using same UID order.',
            status='EXPOSED_DESCRIPTIVE; never an eligible independent gate; reused development labels.'),
            oai557=dict(rows=557,targets=12,mapped_target_indices=[0,2,3,4,5,6,7,8,9],
                cache_job='.local/scratch/slot-b/hb166-oai/prompt.md',native_cache='.local/scratch/hb147-lenses/H-B-147/oai448',
                status='PENDING independent cache/geometry/laterality review and OAI-free checkpoint ancestry. Do not read running cache outputs.',
                eligibility='Requires qualified native448 and crop-v220 side-v2 cache pair, exact557 unique IDs with one knee per participant, approved cache/roster hashes and OAI-free ancestry for BOTH ensembles. Otherwise written ineligible reason; unexplained missing lens => HOLD.',
                aggregation='Equal0.1 mean over ten final multiview checkpoints vs pinned deployed own-S ten-member OAI panel. K36 per view, flip. MOAKS-mapped-nine macro descriptive; existing L7 all-target macro uses masks; participant bootstrap via multi_lens_readout. No imputation or invented labels.',
                inference='Separate CPU-qualified, reviewed OAI panel job after cache completion; not included in these training queues. Pin cache, mapping, ancestry, reference panel and source before inference; no tuning or training from OAI.',
                reference='tools/lens_completion_1015_v1.py image_item/mapped_nine and hb122 target convention')),
        holistic='Report L1-L7 eligibility/value/CI/reason and frozen native decision labels separately from fresh-fold H1/H2; no five-fold success gate. Missing unexplained lens HOLD; scientific result needs independent review before decisions. No Kaggle or serving authorization.',
        GPU_dependency=DEPENDENCY,projection=dict(fits=6,seconds_per_fit=4700,fit_hours=6*4700/3600,
            excludes='CUDA smoke, transfer, OAI/Gold panel inference and independent readout review; estimate only, no training timeout.'))
