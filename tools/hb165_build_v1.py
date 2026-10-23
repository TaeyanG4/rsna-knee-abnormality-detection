"""H-B-165 v1.1.0: OA-only crop expert confirmation on fold 3, seeds 44/45 (CPU preparation).

Changelog 1.1.0 (2026-10-15, Claude builder): replaces the unfinished, never-tested astra-x1
draft 1.0.0 (its dispatch FAILED on a revoked token); the draft bytes are preserved in
.local/backups/hb165_build_v1/. New stage .local/scratch/slot-b/hb165/pilot-fold3-v100.
Configs are DERIVED from the four sealed H-B-155 v1.2.0 configs (recipe authority). Only the
fold-3 population roots, the seed, H-B-165 metadata/gate/bindings and an evaluation-only Gold58
panel change. H-B-155 tools/artifacts are imported or read, never modified. The H-B-155 runtime
review is reused ONLY when the new target snapshot has an empty policy diff against it.
"""
import argparse
import ast
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from tools import hb155_build_v1 as parent
from tools import hb155_exclusions_v1 as exclusions
from tools.hb150_io_v1 import atomic_json

ROOT = parent.ROOT
WORK = ROOT/'.local/scratch/slot-b/hb165/pilot-fold3-v100'
VERSION = '1.1.0'
EXPERIMENT = 'H-B-165'
SEEDS = (44, 45)
PARENT_SEED = {44: 42, 45: 43}
ARMS = ('control', 'crop')
GPU = parent.GPU
FOLD_SOURCE = 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold3'
FOLD = '.local/scratch/slot-b/hb165/pilot-fold3-v100/fold3'
CANONICAL = 'artifacts/experiments/EXP084_GROUP_R34_LOSS_PROTOCOL/ordered-ids.json'
VAL_UID_SHA256 = '7eeb179fa3372e16a91ed3341fe57a1f06f3ab081a1ae26db168f6289f9b4c8f'
GOLD_SOURCE = 'storage/competition/train.csv'
READOUT = 'tools/hb165_readout_v1.py'
NEW_SOURCES = ['tools/hb165_build_v1.py', 'tools/hb165_fit_v1.py', 'tools/hb165_watchdog_v1.py']
QUEUES = {'smoke': 'hb165-smoke-queue.json', 'fit': 'hb165-fit-queue.json'}
BUILDER_IDS = ('claude-hb165-builder', 'hb165-builder')
# Sealed H-B-155 v1.2.0 scientific configs (pilot-v120/prepared.json bindings).
PARENT_CONFIG_SHA256 = {
    ('control', 42): '948ed73b5f389414c134d5a1a00706233e5def6170b8e943311699e9c41924d5',
    ('control', 43): 'b81370c2e91bdd3ae354440eb0c658f4a2a55f8f84b04fc40c384c22334e92a4',
    ('crop', 42): 'e4c060ec021124a6047a392a983f74cb61f4eb9f85d80b7ab424572128f429b8',
    ('crop', 43): 'e68767a636f78baaa09375c0e9cdaa306b0950c9d586c14447bcc93154360a00'}
# Disk: measured H-B-155 control-s42 final.pt bytes; an epoch journal holds the model plus
# AdamW exp_avg/exp_avg_sq (fp32) -> estimate 3 x final + 64 MiB (RNG/scheduler/history).
FINAL_BYTES = 198_806_070
EPOCH_BYTES = 3*FINAL_BYTES + (64 << 20)
OUTPUT_BYTES = 256 << 20
DISK_MARGIN = 10 << 30

DECISION_ID = 'B-20261015-H-B-165-FOLD3'
COORDINATOR_DECISION = dict(
    schema='hb165-coordinator-decision/v1', decision_id=DECISION_ID, coordinator='B',
    recorded_at='2026-10-15 12:0x KST (relayed to the Claude builder 12:5x KST)',
    source='explicit B coordinator decision (continue-prompt.md; Claude builder prompt)',
    fold=3, seeds=[44, 45], exposure='EXPOSED', clean_oof=False,
    prior_fold3_exposure=('D-B-157 (2026-10-12): all 864 fold-3 validation studies were evaluated by the '
        'HB105 aligned OOF readout (C4-s42, R50-s42/43, S-s42/43, T-s42/43) to select the ConvNeXt-S MIL '
        'model family whose recipe BOTH arms share'),
    rationale=('The bias to remove is the post hoc choice of the OA targets from the fold-2 crop readout. '
        'Fold 3 was never used for any crop/OA analysis. D-B-157 chose the recipe both arms share (the arms '
        'differ only in the input cache), so it cannot bias the paired crop-minus-control OA delta toward crop.'),
    interpretation='selection-bias-reduced EXPOSED confirmation, not clean OOF',
    estimand='paired fold-3 OA-only routed minus control delta (AUC of seed-mean K36flip probabilities)',
    recipe='exact H-B-155 v1.2.0 recipe, caches and exclusions; only fold and seeds change',
    old_thresholds='H-B-155 thresholds unchanged and not applied retroactively')
SELECTION_EXPOSURE = 'D-B-157 model-family selection on fold 3, identical for both paired arms'
EXPOSURE_DETAILS = ('L2 report-derived training labels; R7 slab-prior crop geometry; D-B-157 model-family '
    'selection evaluated all 864 fold-3 validation studies; four correlated report-derived rulers')
GOLD_POLICY = ('evaluation-only Gold58 panel: the 58 competition-labelled studies (storage/competition/train.csv, '
    'ACL non-missing), disjoint from every EXP099 fold train/validation roster and present in both caches; '
    'K36 flip predictions after the fixed final epoch feed the descriptive L7 lens only; never used for '
    'training, selection or the registered gate. Training is unaffected: gold UIDs enter only shard '
    'verification, zero placeholder labels and the post-training prediction loop.')
EXPOSURE_STATEMENT = (
    'EXPOSED / clean_oof=false. Fold 3 = EXP099 fold3, 864 validation studies (ordered UID sha256 '
    + VAL_UID_SHA256 + '). D-B-157 (2026-10-12) evaluated all 864 of them with the HB105 aligned OOF '
    'members C4-s42, R50-s42/43, S-s42/43 and T-s42/43 to select the ConvNeXt-S MIL model family; that '
    'selection fixed the recipe BOTH arms share, so it cannot bias the paired crop-minus-control OA delta '
    'toward crop. Fold 3 was never used for any crop/OA analysis (H-B-155 used fold 2; H-B-147 folds 0/1). '
    'Remaining exposure: L2 report-derived labels, R7 slab-prior crop geometry, correlated report rulers. '
    'This is a selection-bias-reduced (OA targets chosen post hoc on fold 2), not clean, confirmation.')
GATE = dict(
    name='OA-only routing', routed_targets=['Medial OA', 'Lateral OA'],
    other_targets='control predictions for the other 10 targets; verified byte-identical',
    predictions='val.csv: K36 windows with flip TTA after the fixed final epoch 20',
    aggregate='AUC of arithmetic seed-mean (44, 45) probabilities', rulers=['jev', 'consensus', 'dread', 'aquila'],
    routed_macro_mean_min=0.0005,
    routed_macro_definition=('mean over the 4 rulers of the 12-target macro AUC delta (routed minus control) = '
        '(dAUC Medial OA + dAUC Lateral OA)/12; the 10 non-OA deltas are structurally 0 because their predictions are '
        'byte-identical (a resample where a rare non-OA target AUC is undefined in BOTH models contributes exactly 0)'),
    ci='paired connected-component bootstrap, percentile 95% CI of the 4-ruler mean routed macro delta',
    draws=2000, bootstrap_seed=20261003, ci_lower_strictly_above=0.0,
    oa_target_min=0.003, oa_target_min_rulers=3, each_seed_mean_oa_strictly_above=0.0,
    each_seed_definition='per seed s: mean over 4 rulers x 2 OA targets of AUC(routed_s) - AUC(control_s)',
    non_oa_identical=True, tolerance=1e-12,
    invalid=('HOLD if any OA-target point or any of the 2000 OA-target draws is nonfinite, or the non-OA predictions or '
             'their per-draw AUCs (NaN pattern included) differ from control; no redraw or deletion'),
    descriptive='plain crop-vs-control on all 12 targets and seven lenses for routed and plain; never rescue PASS')


def need(ok, why):
    if not ok:
        raise ValueError(why)


read, rel, uid_hash = parent.read, parent.rel, parent.uid_hash
_SHA = {}


def sha(path):
    """SHA256 with a per-process memo for unchanged large files (size+mtime_ns key)."""
    p = Path(path).resolve(); st = p.stat()
    if st.st_size < (16 << 20):
        return parent.sha(p)
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        _SHA[key] = parent.sha(p)
    return _SHA[key]


def immutable(path, value):
    path = Path(path)
    if path.exists():
        need(read(path) == value, 'immutable binding changed: '+rel(path))
    else:
        atomic_json(path, value)


def member(arm, seed):
    need(arm in ARMS and seed in SEEDS, 'unregistered member')
    return f'{arm}-s{seed}'


def config_path(arm, seed, engineering=False):
    return WORK/('cpu-configs' if engineering else 'configs')/(member(arm, seed)+'.json')


def decision_binding(record=None):
    """Record only the exact coordinator decision; a review PASS never implies it."""
    p = WORK/'coordinator-decision.json'
    if record is not None:
        need(record == DECISION_ID, 'unknown coordinator decision')
        immutable(p, COORDINATOR_DECISION)
    need(p.is_file(), 'explicit coordinator decision missing; register with --coordinator-decision '+DECISION_ID)
    need(read(p) == COORDINATOR_DECISION, 'coordinator decision differs from recorded authorization')
    return {rel(p): sha(p)}


def closure(start):
    """Explicit local import closure (same algorithm as H-B-155, parameterised start set)."""
    found = set(start); todo = [p for p in found if p.endswith('.py')]
    while todo:
        path = todo.pop(); tree = ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    package = path.split('/')[:-node.level]
                    base = '.'.join(package+([base] if base else []))
                names = [base]+[base+'.'+a.name for a in node.names if a.name != '*']
            for name in names:
                stem = name.replace('.', '/')
                for prefix in ('', 'training/src/', 'tools/'):
                    for suffix in ('.py', '/__init__.py'):
                        candidate = prefix+stem+suffix
                        if candidate not in found and (ROOT/candidate).is_file():
                            found.add(candidate); todo.append(candidate)
    return sorted(found)


def source_closure():
    return sorted(set(parent.source_closure()) | set(closure(NEW_SOURCES)))


def gold_from_source():
    """Gold58 UID roster (UIDs only, no label values leave this function)."""
    import pandas as pd
    frame = pd.read_csv(ROOT/GOLD_SOURCE, usecols=['StudyInstanceUID', 'ACL'], dtype={'StudyInstanceUID': str})
    ids = sorted(frame.loc[frame['ACL'].notna(), 'StudyInstanceUID'].tolist())
    need(len(ids) == 58 and len(set(ids)) == 58, 'official Gold58 roster invalid')
    return ids


def gold_record():
    return dict(schema='hb165-gold58/v1', source=GOLD_SOURCE, source_sha256=sha(ROOT/GOLD_SOURCE),
                rule='StudyInstanceUID where ACL is non-missing, sorted', ids=gold_from_source())


def gold_ids():
    g = read(WORK/'gold58-ids.json')
    ids = g['ids']
    need(g.get('schema') == 'hb165-gold58/v1' and len(ids) == 58 and ids == sorted(set(ids)), 'Gold58 roster record invalid')
    if (ROOT/GOLD_SOURCE).is_file():
        need(g == gold_record(), 'Gold58 roster differs from its source')
    return ids


def register_fold():
    exclusions.load()
    original = read(ROOT/FOLD_SOURCE/'ordered-ids.json')
    need(set(original) == {'train', 'val'}, 'fold3 source keys differ')
    new = {**original, **{k: exclusions.filtered(original[k]) for k in ('train', 'val')}}
    immutable(ROOT/FOLD/'ordered-ids.json', new)
    with (ROOT/FOLD_SOURCE/'group-metadata.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames
        rows = [r for r in reader if r['StudyInstanceUID'] not in exclusions.UIDS]
    buffer = io.StringIO(newline=''); writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    p = ROOT/FOLD/'group-metadata.csv'; data = buffer.getvalue().encode()
    if p.exists():
        need(p.read_bytes() == data, 'filtered fold3 groups changed')
    else:
        p.write_bytes(data)
    binding = dict(schema='hb165-fold-binding/v1', fold=3, exclusions_sha256=sha(exclusions.RECORD),
        source={rel(ROOT/FOLD_SOURCE/n): sha(ROOT/FOLD_SOURCE/n) for n in ('ordered-ids.json', 'group-metadata.csv')},
        files={rel(ROOT/FOLD/n): sha(ROOT/FOLD/n) for n in ('ordered-ids.json', 'group-metadata.csv')},
        train=len(new['train']), val=len(new['val']),
        removed_train=len(original['train'])-len(new['train']), removed_val=len(original['val'])-len(new['val']))
    immutable(WORK/'fold-binding.json', binding)
    return binding


def roster():
    b = read(WORK/'fold-binding.json')
    need(b.get('schema') == 'hb165-fold-binding/v1' and b['exclusions_sha256'] == sha(exclusions.RECORD), 'exclusion binding changed')
    for p, d in {**b['source'], **b['files']}.items():
        need(sha(ROOT/p) == d, 'fold3 file changed: '+p)
    raw = read(ROOT/FOLD_SOURCE/'ordered-ids.json'); current = read(ROOT/FOLD/'ordered-ids.json')
    a, v = current['train'], current['val']
    need(all(current[k] == exclusions.filtered(raw[k]) for k in ('train', 'val')), 'asymmetric exclusions')
    need(len(a) == 3479 and len(v) == 864 and len(set(a+v)) == 4343, 'fold3 roster differs')
    need(uid_hash(v) == VAL_UID_SHA256, 'historical fold3 validation digest differs')
    need(not set(exclusions.UIDS) & set(a+v), 'excluded study present')
    with (ROOT/FOLD/'group-metadata.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    need(len(rows) == 4343, 'group count differs')
    for split, ids in (('train', a), ('val', v)):
        need({r['StudyInstanceUID'] for r in rows if r['split'] == split} == set(ids), 'group membership differs')
    need(not {r['component_id'] for r in rows if r['split'] == 'train'} &
         {r['component_id'] for r in rows if r['split'] == 'val'}, 'component crosses fold')
    canonical = read(ROOT/CANONICAL); positions = {u: i for i, u in enumerate(canonical['train']+canonical['val'])}
    need(all(all(positions[y] > positions[x] for x, y in zip(ids, ids[1:])) for ids in (a, v)), 'noncanonical order')
    return a, v


def register_exclusion_check():
    exclusions.load(); a, v = roster(); gold = gold_ids()
    b = read(WORK/'fold-binding.json')
    value = dict(schema='hb165-exclusions-check/v1', fold=3,
        exclusions_record={rel(exclusions.RECORD): sha(exclusions.RECORD)},
        registered_exclusions_sha256=exclusions.REGISTERED_SHA256,
        hb145_record={rel(exclusions.HB145_RECORD): sha(exclusions.HB145_RECORD)},
        excluded_unique_studies=len(exclusions.UIDS),
        excluded_in_fold3_train=len(set(exclusions.UIDS) & set(a)), excluded_in_fold3_val=len(set(exclusions.UIDS) & set(v)),
        removed_train=b['removed_train'], removed_val=b['removed_val'], train=len(a), val=len(v),
        train_uid_sha256=uid_hash(a), val_uid_sha256=uid_hash(v),
        gold58_in_train=len(set(gold) & set(a)), gold58_in_val=len(set(gold) & set(v)),
        component_overlap=0, canonical_order=True,
        policy='Same registered H-B-155 exclusion record and filter for both arms and both splits; '
               'both excluded studies are absent from the EXP099 fold-3 roster, so train/validation are unchanged.')
    need(value['removed_train'] == value['removed_val'] == 0 and value['gold58_in_train'] == value['gold58_in_val'] == 0,
         'fold3 exclusion/Gold58 expectation differs')
    immutable(WORK/'exclusions-check.json', value)
    return value


def registration_value():
    return dict(schema='hb165-registration/v1', experiment=EXPERIMENT, build_version=VERSION,
        decision_id=DECISION_ID, fold=3, seeds=[44, 45],
        members=[member(arm, seed) for seed in SEEDS for arm in ARMS],
        population=dict(train=3479, val=864, val_uid_sha256=VAL_UID_SHA256, gold58=58,
                        gold58_ids={rel(WORK/'gold58-ids.json'): sha(WORK/'gold58-ids.json')}),
        exposure='EXPOSED', clean_oof=False, exposure_statement=EXPOSURE_STATEMENT, gate=GATE,
        readout=dict(module=READOUT, source_sha256=sha(ROOT/READOUT), command='python -B -m tools.hb165_readout_v1 --review <stage>/review.json'),
        lenses=('All seven lenses (tools/multi_lens_readout_l4_v2 over multi_lens_readout v1.0.0) for the routed model '
                'and for plain crop-vs-control; L6 = fixed HB105 fold-3 heldout partner bundle (R50, C4, T; S replaced); '
                'L7 = Gold58 descriptive (OAI MOAKS unavailable for both input contracts, reason recorded). Evidence, not gates.'),
        evaluation_addition=GOLD_POLICY,
        parent=dict(experiment='H-B-155', version='1.2.0', gate_unchanged=parent.GATE,
                    configs={rel(parent.config_path(a, s)): d for (a, s), d in sorted(PARENT_CONFIG_SHA256.items())}),
        prior_result=('D-B-192 (fold 2, seeds 42/43): plain crop NO_GO; post hoc OA-only routing +0.0014 mean macro '
                      '(consensus CI > 0 only); OA target choice is what this registration confirms on fold 3.'),
        multiplicity='single registered primary decision; all other numbers descriptive')


def registration_binding():
    p = WORK/'registration.json'
    need(p.is_file() and read(p) == registration_value(), 'registration missing or changed (readout source or rule)')
    return {rel(p): sha(p)}


def write_registration_md():
    p = WORK/'REGISTRATION.md'
    if p.exists():
        return
    r = read(WORK/'registration.json'); g = r['gate']
    text = '\n'.join([
        '# H-B-165 registration (written before any fit)', '',
        f"Registered {time.strftime('%Y-%m-%d %H:%M:%S')} KST (host clock) by the Claude builder from `registration.json` "
        f"(sha256 {sha(WORK/'registration.json')}); the four scientific configs bind that file by hash.", '',
        '## Exposure', '', r['exposure_statement'], '',
        '## Registered readout (OA-only routing)', '',
        f"- Routed model: crop predictions for {', '.join(g['routed_targets'])}; control predictions for the other 10 targets (verified identical).",
        f"- Predictions: {g['predictions']}; aggregate: {g['aggregate']}; rulers: {', '.join(g['rulers'])}.",
        f"- PASS only if ALL hold: (1) 4-ruler mean routed macro delta >= +{g['routed_macro_mean_min']}; "
        f"(2) paired component-bootstrap 95% lower bound > 0 ({g['draws']} draws, seed {g['bootstrap_seed']}); "
        f"(3) each OA target delta >= +{g['oa_target_min']} on >= {g['oa_target_min_rulers']}/4 rulers; "
        f"(4) each seed's mean OA delta > 0 ({g['each_seed_definition']}); (5) non-OA columns identical.",
        f"- Invalid evidence: {g['invalid']}. Tolerance {g['tolerance']} on >= thresholds.",
        f"- Descriptive: {g['descriptive']}.", '',
        '## Lenses', '', r['lenses'], '',
        '## Evaluation-only addition', '', r['evaluation_addition'], '',
        '## Parent', '', 'H-B-155 v1.2.0 thresholds are unchanged and are not applied retroactively. ' + r['prior_result'], '',
        f"Readout implementation: `{r['readout']['module']}` sha256 {r['readout']['source_sha256']}.", ''])
    p.write_text(text, encoding='utf-8')


def derive(base, *, seed, train, val, gold, fold_root, engineering):
    """Pure transformation of a sealed H-B-155 config. Recipe fields touched: population roots
    and the update counts implied by len(train) (identical for the 3479-study fold-3 train)."""
    c = copy.deepcopy(base)
    roots = c['recipe']['roots']
    roots.update(ordered=fold_root+'/ordered-ids.json', groups=fold_root+'/group-metadata.csv',
                 n_train=len(train), n_val=len(val), train_sha256=uid_hash(train), val_sha256=uid_hash(val))
    steps = len(train)//8*20
    c['recipe']['updates'].update(loader_batches=len(train)//8, steps=steps, max_skips=steps//100)
    c.update(seed=seed, train=list(train), val=list(val), gold=list(gold), optimizer_steps=steps,
             engineering_only=engineering)
    return c


def parent_config(arm, seed):
    ps = PARENT_SEED[seed]; pp = parent.config_path(arm, ps)
    need(sha(pp) == PARENT_CONFIG_SHA256[(arm, ps)], 'sealed H-B-155 parent config changed: '+rel(pp))
    base = read(pp)
    need(base.get('schema') == 'hb155-config/v1' and base.get('version') == '1.2.0' and base.get('arm') == arm
         and base.get('seed') == ps and base.get('fold') == 2 and base.get('engineering_only') is False,
         'parent member identity differs')
    for p, d in base['inputs'].items():
        need(sha(ROOT/p) == d, 'parent dependency drift: '+p)
    return pp, base


def expected(arm, seed, engineering=False):
    name = member(arm, seed)
    pp, base = parent_config(arm, seed)
    train, val = roster(); gold = gold_ids()
    if engineering:
        train, val, gold = train[:2], val[:1], gold[:1]
    c = derive(base, seed=seed, train=train, val=val, gold=gold, fold_root=FOLD, engineering=engineering)
    inputs = dict(base['inputs'])
    inputs.update({p: sha(ROOT/p) for p in source_closure()})
    extra = [pp, ROOT/FOLD_SOURCE/'ordered-ids.json', ROOT/FOLD_SOURCE/'group-metadata.csv',
             ROOT/FOLD/'ordered-ids.json', ROOT/FOLD/'group-metadata.csv', WORK/'fold-binding.json',
             WORK/'exclusions-check.json', WORK/'gold58-ids.json']
    inputs.update({rel(p): sha(p) for p in extra})
    inputs.update(decision_binding()); inputs.update(registration_binding())
    c.update(schema='hb165-config/v1', version=VERSION, experiment=EXPERIMENT, member=name, fold=3,
             inputs=dict(sorted(inputs.items())), gate=GATE, holdout_status='EXPOSED_PAIRED_FOLD3_AUTHORIZED',
             coordinator_decision=COORDINATOR_DECISION, estimand=COORDINATOR_DECISION['estimand'],
             selection_exposure=SELECTION_EXPOSURE, exposure_details=EXPOSURE_DETAILS,
             exposure='EXPOSED', clean_oof=False, parent_config={rel(pp): sha(pp)}, gold_policy=GOLD_POLICY)
    return c


def queue_value(mode):
    command = ['python', '-B', '-m', 'tools.hb165_build_v1', mode+'-sequence', '--attempt', 'v100']
    if mode == 'fit':
        command += ['--review', rel(WORK/'review.json')]
    return [dict(command=command, tag='hb165-'+mode+'-v100')]


def register(record=None):
    decision_binding(record)
    if not (WORK/'gold58-ids.json').exists():
        immutable(WORK/'gold58-ids.json', gold_record())
    gold_ids(); register_fold(); register_exclusion_check()
    immutable(WORK/'registration.json', registration_value()); write_registration_md()
    for engineering in (False, True):
        for seed in SEEDS:
            for arm in ARMS:
                immutable(config_path(arm, seed, engineering), expected(arm, seed, engineering))
    for mode in ('smoke', 'fit'):
        immutable(WORK/QUEUES[mode], queue_value(mode))
    return dict(status='REGISTERED_EXPOSED', fold=3, seeds=[44, 45], members=4, engineering_members=4,
                train=3479, val=864, gold=58, exposure='EXPOSED', clean_oof=False)


def validate(path):
    c = read(path)
    need(c == expected(c['arm'], c['seed'], c['engineering_only']), 'config/source binding differs')
    return c


def gold_in_cache(c):
    import numpy as np
    ids = set(np.load(ROOT/c['corpus']/'ids.npy', allow_pickle=False).astype(str).tolist())
    need(set(c['gold']) <= ids, 'Gold58 study missing from cache')


def prepare(engineering=False):
    bindings, missing = {}, []
    for seed in SEEDS:
        for arm in ARMS:
            p = config_path(arm, seed, engineering); c = validate(p); bindings[rel(p)] = sha(p)
            try:
                bindings.update(parent.cache_binding(c)); gold_in_cache(c)
            except (OSError, ValueError) as exc:
                missing.append(type(exc).__name__+': '+str(exc))
    if missing:
        return dict(status='HOLD', reasons=sorted(set(missing)))
    immutable(WORK/('cpu-prepared.json' if engineering else 'prepared.json'),
              dict(schema='hb165-prepared/v1', bindings=dict(sorted(bindings.items()))))
    return dict(status='PREPARED_ENGINEERING' if engineering else 'PREPARED', members=4)


PINS = dict(timm='1.0.19', pydicom='3.0.2', pylibjpeg='2.1.0', numpy='2.0.2', pandas='2.2.2', sklearn='1.9.0')
POLICY = dict(matmul_tf32=False, cudnn_tf32=False, cudnn_benchmark=False, cudnn_deterministic=True,
              deterministic_algorithms=True, flash_sdp=True, mem_efficient_sdp=True, math_sdp=True)


def check_target(target):
    need(target.get('gpu') == GPU and target.get('capability') == [8, 9], 'target is not 4090 sm89')
    need(target.get('torch') == '2.5.1+cu121' and target.get('cuda') == '12.1', 'torch/CUDA differs')
    for name, value in PINS.items():
        need(target.get('libs', {}).get(name) == value, 'pin differs: '+name)
    for key, value in POLICY.items():
        need(target.get(key) == value, 'runtime policy differs: '+key)


def runtime_evidence():
    """Parent review validity + new snapshot pins + policy diff against the reviewed snapshot."""
    from tools.parity_runtime import diff_runtime
    base = parent.runtime_bindings()
    pr = parent.WORK/'runtime-review.json'; r = read(pr)
    need(r.get('schema') == 'hb155-runtime-review/v1' and r.get('verdict') == 'PASS' and r.get('independent') is True
         and bool(r.get('reviewer')) and r.get('bindings') == base, 'H-B-155 runtime review missing/stale')
    target_path = WORK/'runtime/4090.json'; target = read(target_path); check_target(target)
    diff = diff_runtime(read(parent.WORK/'runtime/4090.json'), target)
    bindings = {**base, rel(pr): sha(pr), rel(target_path): sha(target_path)}
    return dict(sorted(bindings.items())), diff


def runtime_bindings():
    bindings, diff = runtime_evidence()
    if not diff:
        return bindings
    p = WORK/'runtime-review.json'
    need(p.is_file(), 'runtime differs from the reviewed H-B-155 snapshot: '+json.dumps(diff, sort_keys=True)
         + '; an independent hb165-runtime-review/v1 (bindings + diff from runtime-evidence) is required')
    r = read(p)
    need(r.get('schema') == 'hb165-runtime-review/v1' and r.get('verdict') == 'PASS' and r.get('independent') is True
         and bool(r.get('reviewer')) and r.get('reviewer') not in BUILDER_IDS and r.get('bindings') == bindings
         and r.get('diff_vs_h155') == diff, 'independent hb165 runtime review absent/stale')
    return {**bindings, rel(p): sha(p)}


def require_runtime(config, snapshot, review=None):
    from tools.parity_runtime import diff_runtime
    validate(config); runtime_bindings()
    need(snapshot.get('gpu') == GPU and snapshot.get('capability') == [8, 9] and
         not diff_runtime(read(WORK/'runtime/4090.json'), snapshot), 'runtime changed after the stage snapshot')


def cpu_evidence(device_dir='cpu'):
    paths = []
    for seed in SEEDS:
        for arm in ARMS:
            cp = config_path(arm, seed, True); p = WORK/device_dir/member(arm, seed)/'summary.json'; s = read(p)
            need(s.get('passed') is True and s.get('smoke') is True and s.get('engineering_only') is True and
                 s.get('config_sha256') == sha(cp) and s.get('device') == 'cpu' and s.get('optimizer_steps') == 1 and
                 s.get('cpu_cuda_initialized') is False, 'CPU smoke missing/stale: '+member(arm, seed))
            paths += [cp, p]
    return paths


def smoke_gate():
    """Fail closed before any CUDA smoke: target CPU qualification, caches, runtime and crop review."""
    need(prepare()['status'] == 'PREPARED' and prepare(True)['status'] == 'PREPARED_ENGINEERING', 'caches/configs not prepared')
    bindings = {**decision_binding(), **registration_binding(), **runtime_bindings(), **parent.crop_review_bindings()}
    bindings.update({rel(p): sha(p) for p in cpu_evidence()})
    return bindings


def review_bindings():
    bindings = smoke_gate()
    paths = [WORK/'prepared.json', WORK/'cpu-prepared.json', WORK/'fold-binding.json', WORK/'exclusions-check.json',
             WORK/'gold58-ids.json', WORK/QUEUES['smoke'], WORK/QUEUES['fit']]
    for seed in SEEDS:
        for arm in ARMS:
            cp = config_path(arm, seed); p = WORK/'cuda'/member(arm, seed)/'summary.json'; s = read(p)
            need(s.get('passed') is True and s.get('smoke') is True and s.get('engineering_only') is True and
                 s.get('config_sha256') == sha(cp) and s.get('device') == 'cuda' and s.get('optimizer_steps') == 2,
                 'CUDA smoke identity differs: '+member(arm, seed))
            need(s.get('gpu_name') == GPU and s.get('full_shape') is True and 0 < s.get('peak_reserved_bytes', 0) <= 10*(1 << 30),
                 'full-shape VRAM gate')
            need(s['batch'] == 8 and s['val_batch'] == 4 and s['validation_shapes'] == ['k24', 'k36', 'k36flip'], 'smoke shapes differ')
            need(all(s['healthy_rates'].get(k, 0) > 0 for k in ('train_step', 'validation')), 'healthy rates missing')
            paths += [cp, p]
    bindings.update({rel(p): sha(p) for p in paths})
    return dict(sorted(bindings.items()))


def require_fit(path, review=None):
    c = validate(path)
    need(not c['engineering_only'], 'engineering config cannot fit')
    bindings = {**decision_binding(), **registration_binding()}
    need(c.get('holdout_status') == 'EXPOSED_PAIRED_FOLD3_AUTHORIZED' and c.get('fold') == 3 and
         c.get('seed') in SEEDS and c.get('exposure') == 'EXPOSED' and c.get('clean_oof') is False and
         c.get('coordinator_decision') == COORDINATOR_DECISION and c.get('estimand') == COORDINATOR_DECISION['estimand'] and
         c.get('selection_exposure') == SELECTION_EXPOSURE and c.get('gate') == GATE and
         all(c.get('inputs', {}).get(p) == d for p, d in bindings.items()),
         'explicit registered EXPOSED fold-3 decision/registration absent or changed; D-B-157 exposure cannot be waived')
    need(review is not None, 'independent review required'); r = read(review)
    need(r.get('schema') == 'hb165-review/v1' and r.get('verdict') == 'PASS' and r.get('independent') is True and
         bool(r.get('reviewer')) and r.get('reviewer') not in BUILDER_IDS and r.get('bindings') == review_bindings(),
         'independent smoke-bound review required')
    return c


def admit(review=None):
    try:
        for seed in SEEDS:
            for arm in ARMS:
                require_fit(config_path(arm, seed), review)
        return dict(status='ADMITTED', exposure='EXPOSED', clean_oof=False, estimand=COORDINATOR_DECISION['estimand'])
    except (ValueError, OSError, KeyError) as exc:
        return dict(status='HOLD', reason=str(exc))


def disk_requirement(mode):
    epochs = 1 if mode == 'smoke' else 20
    return 4*(epochs*EPOCH_BYTES + FINAL_BYTES + OUTPUT_BYTES) + DISK_MARGIN


def disk_gate(mode, path=None):
    need(mode in ('smoke', 'fit'), 'invalid disk mode')
    target = Path(path or WORK); target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free; required = disk_requirement(mode)
    need(free >= required, f'insufficient disk for {mode}: need {required} free bytes, have {free}')
    return dict(status='DISK_OK', mode=mode, free_bytes=free, required_bytes=required)


def sequence(mode, review=None, attempt='v100'):
    need(mode in ('smoke', 'fit') and attempt.replace('-', '').isalnum(), 'invalid sequence')
    disk_gate(mode)
    if mode == 'fit':
        need(admit(review)['status'] == 'ADMITTED', 'admission HOLD')
    else:
        smoke_gate()
    for seed in SEEDS:
        for arm in ARMS:
            name = member(arm, seed); out = WORK/('cuda' if mode == 'smoke' else 'fits')/name
            cmd = [sys.executable, '-B', '-m', 'tools.hb165_watchdog_v1', '--attempt',
                   str(WORK/'attempts'/f'{mode}-{name}-{attempt}'), '--budget', str(WORK/'gpu-budget.json')]
            if mode == 'smoke':
                cmd += ['--measurement']
            else:
                rates = WORK/'cuda'/name/'healthy-rates.json'
                immutable(rates, read(WORK/'cuda'/name/'summary.json')['healthy_rates'])
                cmd += ['--healthy-rates', str(rates)]
            cmd += ['--', sys.executable, '-B', '-m', 'tools.hb165_fit_v1', mode, '--config', str(config_path(arm, seed)),
                    '--out', str(out), '--device', 'cuda']
            if review:
                cmd += ['--review', str(review)]
            subprocess.run(cmd, cwd=ROOT, check=True)
    return dict(status='FOUR_'+mode.upper()+'_COMPLETE')


def snapshot_runtime(out, device='cpu'):
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from tools import hb165_fit_v1 as f
    f.h.torch.set_num_threads(2)
    f.h.configure_runtime('none' if device == 'cpu' else 'fp16', device, policy='strict')
    atomic_json(out, f.runtime_info())
    return dict(status='SNAPSHOT', device=device, path=str(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('register', 'validate', 'prepare', 'admit', 'runtime-evidence', 'runtime-bindings',
                                        'review-bindings', 'smoke-sequence', 'fit-sequence', 'snapshot', 'disk', 'queue-check'))
    ap.add_argument('--coordinator-decision', choices=(DECISION_ID,))
    ap.add_argument('--engineering', action='store_true'); ap.add_argument('--review', type=Path)
    ap.add_argument('--attempt', default='v100'); ap.add_argument('--out', type=Path)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu'); ap.add_argument('--mode', choices=('smoke', 'fit'), default='fit')
    a = ap.parse_args(argv)
    try:
        if a.command == 'register':
            result = register(a.coordinator_decision)
        elif a.command == 'validate':
            for seed in SEEDS:
                for arm in ARMS:
                    validate(config_path(arm, seed, a.engineering))
            result = dict(status='VALID', members=4, engineering=a.engineering)
        elif a.command == 'prepare':
            result = prepare(a.engineering)
        elif a.command == 'admit':
            result = admit(a.review)
        elif a.command == 'runtime-evidence':
            bindings, diff = runtime_evidence(); result = dict(status='EVIDENCE', bindings=bindings, diff_vs_h155=diff,
                reuse_h155_review=not diff)
        elif a.command == 'runtime-bindings':
            result = runtime_bindings()
        elif a.command == 'review-bindings':
            result = review_bindings()
        elif a.command == 'snapshot':
            need(a.out is not None, 'snapshot needs --out'); result = snapshot_runtime(a.out, a.device)
        elif a.command == 'disk':
            result = disk_gate(a.mode)
        elif a.command == 'queue-check':
            from tools.gpu_queue import validate_plan
            for mode in ('smoke', 'fit'):
                need(read(WORK/QUEUES[mode]) == queue_value(mode), 'queue file differs')
                validate_plan(read(WORK/QUEUES[mode]))
            result = dict(status='QUEUES_VALID', launches=0)
        else:
            result = sequence(a.command.split('-')[0], a.review, a.attempt)
    except (ValueError, OSError, KeyError) as exc:
        if a.command in ('smoke-sequence', 'fit-sequence'):
            raise
        result = dict(status='HOLD', reason=str(exc))
    print(json.dumps(result))
    return 2 if result.get('status') == 'HOLD' else 0


if __name__ == '__main__':
    raise SystemExit(main())
