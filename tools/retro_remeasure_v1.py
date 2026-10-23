"""RM-v1 registered retrospective audit, CPU only (v1.0.4).

The frozen selection is read, never recalculated. An unsupported screen is
explicitly NOT_REMEASURABLE; saved result deltas are never treated as a fresh
measurement. No existing artifact is overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from statistics import mean

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '2'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REG = 'artifacts/evidence_ledger/retro-remeasure-registration-v1.json'
MAP = 'artifacts/evidence_ledger/ruler-map-v1/map.json'
T = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
     'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
UID = 'StudyInstanceUID'
VERSION = '1.0.4'  # Add hash-bound EXP169 partner bundle contrasts.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_registration(reg, root=ROOT):
    """Verify frozen data and selection bytes; the actively edited ruler code is exempt."""
    skip = {'tools/evidence_ledger_rulers.py'}
    for rel, expected in reg['inputs_sha256'].items():
        if rel in skip:
            continue
        path = root / rel
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Registered input hash mismatch: ' + rel)
    return {'checked': len(reg['inputs_sha256'])-len(skip), 'code_exemption': sorted(skip)}


def pointer(data, path):
    for part in filter(None, path.split('/')):
        data = data[part]
    return data


def old_delta_check(screen, contrast, gate):
    """Check saved vector arithmetic, which is weaker than prediction reproduction."""
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location('rm_selection', ROOT/'artifacts/evidence_ledger/retro-remeasure-selection-v1.py')
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    block = pointer(screen, contrast['prefix'])
    vectors = []
    for ref in gate:
        cell = mod.ref_block(block, ref)
        vector = mod.target_vec(cell) if cell else None
        if vector is None or len(vector) != 12 or not all(math.isfinite(x) for x in vector):
            raise ValueError('Old target vector absent or nonfinite')
        vectors.append(vector)
    old = np.mean(np.asarray(vectors, float), axis=0)
    if abs(float(old.mean())-contrast['old_primary_delta']) > 1e-6:
        raise ValueError('Old macro arithmetic mismatch')
    return dict(old_saved_macro=float(old.mean()), old_saved_per_target=dict(zip(T, old.tolist())))


def declared_inputs(screen, root=ROOT):
    """Attest every declared file before attempting a screen-specific adapter."""
    pairs = []
    if isinstance(screen.get('input_bindings'), dict):
        pairs = list(screen['input_bindings'].items())
    elif isinstance(screen.get('input_files'), dict):
        hashes = screen.get('input_sha256', {})
        pairs = [(p, hashes.get(key)) for key, p in screen['input_files'].items()]
    else:
        return dict(status='HASH_ONLY', files=0, reason='No registered input file paths')
    for rel, expected in pairs:
        if not isinstance(expected, str) or len(expected) != 64:
            return dict(status='HASH_UNBOUND', files=len(pairs), reason='A listed input lacks a SHA-256')
        path = Path(rel)
        if not path.is_absolute():
            path = root/path
        if not path.is_file():
            return dict(status='MISSING', files=len(pairs), reason='A listed input is absent')
        if sha(path) != expected:
            return dict(status='HASH_MISMATCH', files=len(pairs), reason='A listed input changed')
    return dict(status='VERIFIED', files=len(pairs))


def bound_score_file(screen, rel):
    if rel.replace('\\', '/') not in {p.replace('\\', '/')
                                     for p in screen['input_bindings']}:
        raise ValueError('Prediction path is not in the registered screen bindings')
    return score_file(rel)


def f0_paths(screen, item):
    names = list(screen['input_bindings'])
    experiment = item['path'].split('/')[2]
    arm = {}
    for seed in (42, 43):
        for role, token in (('candidate', experiment.replace('_AUG1', '_AUG1_') if
                              experiment == 'EXP132_AUG1' else experiment+'_S'),
                            ('control', f'EXP127_AUG1_S{seed}_F0')):
            matches = []
            for name in names:
                norm = name.replace('\\', '/')
                if not norm.endswith('/oof_k36.csv'):
                    continue
                if role == 'candidate':
                    variant = item['path'].split('/')[4] if experiment == 'EXP132_AUG1' else None
                    tag = (f'EXP132_AUG1_{variant}_S{seed}_F0' if variant else
                           f'EXP142_MRN_RESCUE_S{seed}_F0')
                else:
                    tag = token
                if '/'+tag+'/oof_k36.csv' in norm:
                    matches.append(name)
            if len(matches) != 1:
                raise ValueError('F0 prediction binding absent or ambiguous')
            arm[(role, seed)] = matches[0]
    return arm


def component_indices(ids):
    component_file = ROOT/'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
    mapping = {uid: component['component_id']
               for component in json.loads(component_file.read_text(encoding='utf-8'))
               for uid in component['members']}
    components = sorted({mapping[uid] for uid in ids})
    return [np.asarray([i for i, uid in enumerate(ids) if mapping[uid] == c], int)
            for c in components]


def f0_remeasure(screen, item, contrast, ruler_map, labels, draws=2000,
                 prepared=None, exposed=('dread',)):
    paths = None if prepared is not None else f0_paths(screen, item)
    scores = prepared if prepared is not None else {
        (role, seed): score_file(path) for (role, seed), path in paths.items()}
    ids = scores[('candidate', 42)].index.tolist()
    if any(set(x.index) != set(ids) for x in scores.values()):
        raise ValueError('F0 seed/arm roster differs')
    scores = {key: frame.reindex(ids).to_numpy(float) for key, frame in scores.items()}
    sources = {k: value.reindex(ids).to_numpy(float) for k, value in labels.items()}
    old = contrast['old_saved_per_target']
    dread = sources['dread']
    gate = np.mean([np.asarray([exp_auc(scores[('candidate', seed)][:, j], dread[:, j])-
                   exp_auc(scores[('control', seed)][:, j], dread[:, j])
                   for j in range(12)]) for seed in (42, 43)], axis=0)
    if np.max(np.abs(gate-np.asarray([old[t] for t in T]))) > 1e-6:
        raise ValueError('Old per-target Dread deltas not reproduced')
    highmed = [t for t in T if ruler_map['recommendations'][t]['validity'] != 'LOW']
    exposure = set(exposed)
    refs = {t: [r for r in ruler_map['recommendations'][t]['references']
                if r not in exposure and not exposure.intersection(
                    ruler_map['sources'][r]['ancestors'])] for t in highmed}
    if sum(bool(refs[t]) for t in highmed) < 8:
        raise ValueError('Fewer than eight ancestry-safe map targets')
    highmed = [t for t in highmed if refs[t]]

    def evaluate(indices):
        cell = {}
        for target in highmed:
            j = T.index(target)
            for ref in refs[target]:
                y = sources[ref][indices, j]
                mask = np.isfinite(y)
                ds = []
                for seed in (42, 43):
                    a = scores[('candidate', seed)][indices, j][mask]
                    b = scores[('control', seed)][indices, j][mask]
                    yy = y[mask]
                    ds.append(exp_auc(a, yy)-exp_auc(b, yy))
                cell[(target, ref)] = float(np.mean(ds))
        per = {t: mean(cell[(t, ref)] for ref in refs[t]) for t in highmed}
        return cell, per, mean(per.values())

    point_cells, per_target, point = evaluate(np.arange(len(ids)))
    low_report = {}
    for target in ('Synovitis', 'Contusion'):
        j = T.index(target)
        low_report[target] = {}
        for ref in ('jev_hybrid', 'consensus'):
            y = sources[ref][:, j]
            mask = np.isfinite(y)
            low_report[target][ref] = float(np.mean([
                exp_auc(scores[('candidate', seed)][mask, j], y[mask])-
                exp_auc(scores[('control', seed)][mask, j], y[mask])
                for seed in (42, 43)]))
    units = component_indices(ids)
    rng = np.random.default_rng(20261003)
    boot = []
    per_draw = {t: [] for t in highmed}
    cell_draw = {key: [] for key in point_cells}
    for _ in range(draws):
        choice = rng.integers(0, len(units), len(units))
        index = np.concatenate([units[k] for k in choice])
        cells, per, macro = evaluate(index)
        if math.isfinite(macro):
            boot.append(macro)
            for t in highmed:
                per_draw[t].append(per[t])
            for key in cells:
                cell_draw[key].append(cells[key])
    if len(boot) != draws:
        raise ValueError('Nonfinite bootstrap draw')
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    harm = []
    for target in highmed:
        bad = [ref for ref in refs[target]
               if np.percentile(cell_draw[(target, ref)], 97.5) < -.003]
        for a, b in itertools.combinations(bad, 2):
            x, y = ruler_map['sources'][a], ruler_map['sources'][b]
            if x['evidence_group'] != y['evidence_group'] and not set(x['ancestors'])&set(y['ancestors']):
                harm.append(target)
    map9 = mean(per_target[t] for t in highmed if t != 'PF OA')
    # The registered primary Dread threshold is replaced by Delta_map10;
    # consensus is a separate Tier-A floor in these screens.
    consensus_old = np.mean([np.mean([exp_auc(scores[('candidate', seed)][:, j][
        np.isfinite(sources['consensus'][:, j])], sources['consensus'][:, j][
        np.isfinite(sources['consensus'][:, j])])-
        exp_auc(scores[('control', seed)][:, j][np.isfinite(sources['consensus'][:, j])],
                sources['consensus'][:, j][np.isfinite(sources['consensus'][:, j])])
        for j in range(12)]) for seed in (42, 43)])
    bar_met = point >= .004 and ci[0] > 0 and consensus_old >= 0
    status = ('REOPEN_CANDIDATE' if bar_met and map9 >= 0 and not harm else
              'CONFIRMED_NEG')
    return dict(status=status, old_reproduction_max_abs=float(np.max(np.abs(
                gate-np.asarray([old[t] for t in T])))),
                delta_map10=point, delta_map9=map9, ci95=ci,
                per_target={t: dict(delta=per_target[t],
                    ci95=np.percentile(per_draw[t], [2.5, 97.5]).tolist(),
                    rulers=refs[t]) for t in highmed},
                low_report_only=low_report,
                corroborated_harm_targets=sorted(set(harm)),
                bar=dict(point_min=.004, ci_lower_strictly_above=0,
                         old_dread_delta=float(gate.mean()),
                         consensus_floor_met=bool(consensus_old >= 0),
                         consensus_delta=float(consensus_old)),
                bootstrap=dict(draws=draws, seed=20261003, unit='component'))


def exp138_scores(screen, label):
    """Recreate each seed's five fold, within-fold ranked OOF prediction."""
    if label not in ('plain_vs_aug1', 'tta_vs_aug1'):
        raise ValueError('Unsupported EXP138 contrast')
    pred = {}
    for seed in (42, 43):
        tta = (bound_score_file(screen, f'artifacts/experiments/EXP138_BUNDLE_AUG1_MH/tta/s{seed}/exp117/oof_tta_k36.csv')
               if label == 'tta_vs_aug1' else None)
        candidate, control = [], []
        for fold in range(5):
            aug = bound_score_file(screen, f'artifacts/experiments/EXP{127 if fold == 0 else 134}_AUG1_S{seed}_F{fold}/oof_k36.csv')
            if tta is None:
                arm = bound_score_file(screen, f'artifacts/experiments/EXP138_BUNDLE_AUG1_MH_S{seed}_F{fold}/oof_k36.csv')
            else:
                arm = tta.reindex(aug.index)
            if arm.isna().any().any() or set(arm.index) != set(aug.index):
                raise ValueError('EXP138 fold roster differs')
            candidate.append(pct(arm.reindex(aug.index)))
            control.append(pct(aug))
        pred[('candidate', seed)] = pd.concat(candidate)
        pred[('control', seed)] = pd.concat(control)
    return pred


def exp139_scores(screen, item, label):
    variant = item['path'].split('/')[3]
    label = label.removesuffix(' k36')
    roles = {('LATE', 'swa_vs_final'): ('oof_swa_k36.csv', 'oof_k36.csv'),
             ('POS', 'variant_vs_control'): ('oof_k36.csv', 'oof_k36.csv')}
    if (variant, label) not in roles:
        raise ValueError('Unsupported EXP139 contrast')
    candidate_name, control_name = roles[(variant, label)]
    pred = {}
    for seed in (42, 43):
        pred[('candidate', seed)] = bound_score_file(screen,
            f'artifacts/experiments/EXP139_AUG1_{variant}_S{seed}_F0/{candidate_name}')
        base = (f'EXP127_AUG1_S{seed}_F0' if variant == 'POS' else
                f'EXP139_AUG1_LATE_S{seed}_F0')
        pred[('control', seed)] = bound_score_file(screen,
            f'artifacts/experiments/{base}/{control_name}')
    return pred


def exp169_scores(screen):
    """Rebuild the registered within-fold and global rank bundle from bound files."""
    parts = {name: [] for name in ('l2', 'dino', 'convnext', 'h1')}
    ids = []
    for fold in range(5):
        scores = {name: score_file(screen['input_files'][f'{name}_f{fold}'])
                  for name in (f'l2_s42', f'l2_s43', 'dino', 'convnext', 'h1')}
        roster = scores['l2_s42'].index
        if any(set(frame.index) != set(roster) or not np.isfinite(frame.to_numpy()).all()
               for frame in scores.values()):
            raise ValueError('EXP169 fold roster or prediction values differ')
        if set(ids).intersection(roster):
            raise ValueError('EXP169 folds overlap')
        ids.extend(roster.tolist())
        parts['l2'].append((pct(scores['l2_s42']) +
                            pct(scores['l2_s43'].reindex(roster)))/2)
        for name in ('dino', 'convnext', 'h1'):
            parts[name].append(pct(scores[name].reindex(roster)))
    pooled = {name: pd.concat(value).loc[ids].to_numpy(float)
              for name, value in parts.items()}
    def global_rank(values):
        # scipy.stats.rankdata(method='average') expressed with pandas so this
        # CPU audit does not depend on the trainer's SciPy environment.
        return ((pd.DataFrame(values).rank(method='average').to_numpy(float)-1.) /
                max(1, len(values)-1))

    ranked = {name: global_rank(pooled[name]) for name in ('l2','dino','convnext')}
    candidate = global_rank(sum(ranked.values())/3.)
    arms = {'bundle': candidate, **ranked, 'h1': global_rank(pooled['h1'])}
    return {name: pd.DataFrame(value, index=ids, columns=T) for name, value in arms.items()}


def exp169_remeasure(screen, saved, label, ruler_map, labels, draws=2000,
                     prepared=None):
    """Apply RM-v1 only when an EXP169 comparison reproduces its old gate."""
    name = {'bundle minus L2 alone (headline gate, D-B-097)': 'l2',
            'bundle minus DINO-base alone': 'dino',
            'bundle minus ConvNeXt-T alone': 'convnext',
            'bundle minus H1 alone': 'h1'}.get(label)
    if name is None:
        raise ValueError('EXP169 comparison lacks a verified adapter')
    arms = prepared if prepared is not None else exp169_scores(screen)
    ids = arms['bundle'].index.tolist()
    if set(ids) != set(arms[name].index):
        raise ValueError('EXP169 arm roster differs')
    arrays = {arm: arms[arm].reindex(ids).to_numpy(float) for arm in ('bundle', name)}
    sources = {ref: frame.reindex(ids).to_numpy(float) for ref, frame in labels.items()}

    def delta(indices, ref, j):
        y = sources[ref][indices, j]
        mask = np.isfinite(y)
        return (exp_auc(arrays['bundle'][indices, j][mask], y[mask])-
                exp_auc(arrays[name][indices, j][mask], y[mask]))

    full = np.arange(len(ids))
    old = np.asarray([saved['old_saved_per_target'][t] for t in T])
    gate_by_ref = {ref: np.asarray([delta(full, ref, j) for j in range(12)])
                   for ref in ('jev_hybrid', 'aquila_v1')}
    gate = np.mean(list(gate_by_ref.values()), axis=0)
    old_refs = screen['comparisons'][name]['references']
    error_by_ref = [np.max(np.abs(gate_by_ref[ref]-np.asarray([
        old_refs[key]['per_target_delta'][t] for t in T])))
        for ref, key in (('jev_hybrid', 'jev'), ('aquila_v1', 'aquila'))]
    error = float(max(np.max(np.abs(gate-old)), *error_by_ref))
    if not math.isfinite(error) or error > 1e-6:
        raise ValueError('Old per-target EXP169 gate not reproduced')
    # Bundle and its L2 member train on Aquila; DINO and ConvNeXt use Dread.
    exposed = {'dread', 'aquila_v1', *ruler_map['sources']['aquila_v1']['ancestors']}
    highmed = [t for t in T if ruler_map['recommendations'][t]['validity'] != 'LOW']
    refs = {t: [r for r in ruler_map['recommendations'][t]['references']
                if r not in exposed and not exposed.intersection(
                    ruler_map['sources'][r]['ancestors'])] for t in highmed}
    highmed = [t for t in highmed if refs[t]]
    if len(highmed) < 8:
        raise ValueError('Fewer than eight ancestry-safe map targets')

    def evaluate(indices):
        cells = {(t, ref): delta(indices, ref, T.index(t))
                 for t in highmed for ref in refs[t]}
        per = {t: mean(cells[(t, ref)] for ref in refs[t]) for t in highmed}
        return cells, per, mean(per.values())

    cells, per, point = evaluate(full)
    units = component_indices(ids)
    rng = np.random.default_rng(20261003)
    boot, by_cell, by_target = [], {key: [] for key in cells}, {t: [] for t in highmed}
    for _ in range(draws):
        index = np.concatenate([units[k] for k in rng.integers(0, len(units), len(units))])
        drawn, target, value = evaluate(index)
        if not math.isfinite(value) or not all(math.isfinite(v) for v in drawn.values()):
            raise ValueError('Nonfinite EXP169 bootstrap draw')
        boot.append(value)
        for key, value in drawn.items():
            by_cell[key].append(value)
        for t, value in target.items():
            by_target[t].append(value)
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    harm = []
    for t in highmed:
        bad = [r for r in refs[t] if np.percentile(by_cell[(t, r)], 97.5) < -.003]
        for a, b in itertools.combinations(bad, 2):
            x, y = ruler_map['sources'][a], ruler_map['sources'][b]
            if x['evidence_group'] != y['evidence_group'] and not set(x['ancestors'])&set(y['ancestors']):
                harm.append(t)
    map9 = mean(per[t] for t in highmed if t != 'PF OA')
    old_dread_floor = bool(screen['comparisons'][name]['references']['dread']['delta'] >= -.003)
    bar_met = point > 0 and ci[0] > 0 and (name != 'l2' or old_dread_floor)
    old_positive = bool(screen['comparisons'][name]['pass'])
    if old_positive:
        status = ('OVERTURNED' if ci[1] < 0 or harm else
                  'CONFIRMED_POS' if bar_met and map9 >= 0 else 'WEAKENED')
    else:
        status = ('REOPEN_CANDIDATE' if bar_met and map9 >= 0 and not harm else
                  'CONFIRMED_NEG')
    low = {t: {ref: delta(full, ref, T.index(t)) for ref in ('jev_hybrid','consensus')}
           for t in ('Synovitis','Contusion')}
    return dict(status=status, old_reproduction_max_abs=error,
                delta_map10=point, delta_map9=map9, ci95=ci,
                per_target={t: dict(delta=per[t],
                    ci95=np.percentile(by_target[t], [2.5, 97.5]).tolist(), rulers=refs[t])
                    for t in highmed}, low_report_only=low,
                corroborated_harm_targets=sorted(set(harm)),
                bar=dict(point_strictly_above=0, ci_lower_strictly_above=0,
                         old_dread_floor_retained=old_dread_floor if name == 'l2' else None),
                bootstrap=dict(draws=draws, seed=20261003, unit='component'))


def exp171_remeasure(screen, saved, ruler_map, labels, draws=2000):
    """Re-score the registered Small/Tiny flip pair after reproducing its old gate."""
    scores = {arm: score_file(screen['input_files'][arm+'_flip'])
              for arm in ('small', 'tiny')}
    ids = scores['small'].index.tolist()
    if set(ids) != set(scores['tiny'].index):
        raise ValueError('EXP171 arm roster differs')
    arrays = {arm: frame.reindex(ids).to_numpy(float) for arm, frame in scores.items()}
    sources = {k: value.reindex(ids).to_numpy(float) for k, value in labels.items()}

    def delta(indices, ref, j):
        y = sources[ref][indices, j]
        mask = np.isfinite(y)
        a = arrays['small'][indices, j][mask]
        b = arrays['tiny'][indices, j][mask]
        return exp_auc(a, y[mask])-exp_auc(b, y[mask])

    gate = {ref: np.asarray([delta(np.arange(len(ids)), ref, j)
                             for j in range(12)])
            for ref in ('jev_hybrid', 'consensus')}
    observed = (gate['jev_hybrid']+gate['consensus'])/2
    old = np.asarray([saved['old_saved_per_target'][t] for t in T])
    error = float(np.max(np.abs(observed-old)))
    if error > 1e-6:
        raise ValueError('Old per-target EXP171 gate not reproduced')

    highmed = [t for t in T if ruler_map['recommendations'][t]['validity'] != 'LOW']
    refs = {t: ruler_map['recommendations'][t]['references'] for t in highmed}
    # The EXP171 Small and Tiny L2 checkpoints train on L2/Aquila labels.
    # That source is self-referenced and absent from the ruler map.
    if any(not refs[t] for t in highmed):
        raise ValueError('Recommended ruler absent')
    if any('aquila' in ruler_map['sources'][r]['ancestors'] for rr in refs.values() for r in rr):
        raise ValueError('Recommended ruler has L2/Aquila ancestry')

    def evaluate(indices):
        cells = {(t, ref): delta(indices, ref, T.index(t))
                 for t in highmed for ref in refs[t]}
        per = {t: mean(cells[(t, ref)] for ref in refs[t]) for t in highmed}
        return cells, per, mean(per.values())

    cells, per, point = evaluate(np.arange(len(ids)))
    units = component_indices(ids)
    rng = np.random.default_rng(20261003)
    boot, by_cell, by_target = [], {key: [] for key in cells}, {t: [] for t in highmed}
    for _ in range(draws):
        index = np.concatenate([units[k] for k in rng.integers(0, len(units), len(units))])
        drawn, target, value = evaluate(index)
        if not math.isfinite(value) or not all(math.isfinite(v) for v in drawn.values()):
            raise ValueError('Nonfinite EXP171 bootstrap draw')
        boot.append(value)
        for key, value in drawn.items():
            by_cell[key].append(value)
        for t, value in target.items():
            by_target[t].append(value)
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    harm = []
    for t in highmed:
        bad = [r for r in refs[t] if np.percentile(by_cell[(t, r)], 97.5) < -.003]
        for a, b in itertools.combinations(bad, 2):
            x, y = ruler_map['sources'][a], ruler_map['sources'][b]
            if x['evidence_group'] != y['evidence_group'] and not set(x['ancestors'])&set(y['ancestors']):
                harm.append(t)
    map9 = mean(per[t] for t in highmed if t != 'PF OA')
    dread_floor = mean(delta(np.arange(len(ids)), 'dread', j) for j in range(12)) >= -.003
    bar_met = point >= .002 and ci[0] > 0 and dread_floor
    status = ('REOPEN_CANDIDATE' if bar_met and map9 >= 0 and not harm
              else 'CONFIRMED_NEG')
    low = {t: {ref: delta(np.arange(len(ids)), ref, T.index(t))
               for ref in ('jev_hybrid', 'consensus')}
           for t in ('Synovitis', 'Contusion')}
    return dict(status=status, old_reproduction_max_abs=error,
                delta_map10=point, delta_map9=map9, ci95=ci,
                per_target={t: dict(delta=per[t],
                    ci95=np.percentile(by_target[t], [2.5, 97.5]).tolist(), rulers=refs[t])
                    for t in highmed}, low_report_only=low,
                corroborated_harm_targets=sorted(set(harm)),
                bar=dict(point_min=.002, ci_lower_strictly_above=0,
                         old_dread_floor_retained=dread_floor),
                bootstrap=dict(draws=draws, seed=20261003, unit='component'))


def step5(reg, ruler_map=None, root=ROOT, draws=2000):
    ruler_map = ruler_map or json.loads((root/MAP).read_text(encoding='utf-8'))
    labels = None
    output = []
    for item in reg['step5_retrospective_remeasure']['inventory']:
        if not item['selected']:
            continue
        screen = json.loads((root/item['path']).read_text(encoding='utf-8'))
        attestation = declared_inputs(screen, root)
        exp169_prepared = None
        contrasts = []
        for contrast in item['contrasts']:
            if not contrast['selected']:
                continue
            try:
                saved = old_delta_check(screen, contrast, item['gate_reference'])
                check = 'SAVED_VECTOR_ARITHMETIC_ONLY'
            except (ValueError, KeyError, TypeError) as exc:
                saved = None
                check = 'FAILED: ' + str(exc)
            result = dict(status='NOT_REMEASURABLE',
                reason=('Prediction paths unresolved in registered screen (hashes only)' if attestation['status'] == 'HASH_ONLY'
                        else 'No verified adapter for the original prediction aggregation; old per-target AUC not reproduced'),
                delta_map10=None, ci95=None)
            experiment = item['path'].split('/')[2]
            if (attestation['status'] == 'VERIFIED' and saved is not None and
                experiment in ('EXP132_AUG1', 'EXP138_BUNDLE_AUG1_MH',
                               'EXP139_AUG1_VARIANTS',
                               'EXP142_MRN_RESCUE', 'EXP169_PARTNER_BUNDLE',
                               'EXP171_CONVNEXT_S_L2')):
                try:
                    if labels is None:
                        labels = labels_for_map(ruler_map)
                    if experiment == 'EXP171_CONVNEXT_S_L2':
                        result = exp171_remeasure(screen, saved, ruler_map, labels, draws)
                    elif experiment == 'EXP169_PARTNER_BUNDLE':
                        if exp169_prepared is None:
                            exp169_prepared = exp169_scores(screen)
                        result = exp169_remeasure(screen, saved, contrast['label'],
                            ruler_map, labels, draws, prepared=exp169_prepared)
                    elif experiment == 'EXP138_BUNDLE_AUG1_MH':
                        result = f0_remeasure(screen, item, saved, ruler_map, labels, draws,
                            prepared=exp138_scores(screen, contrast['label']),
                            exposed=('dread','relabel_v1','steven_v4','pilkwang_v2'))
                    elif experiment == 'EXP139_AUG1_VARIANTS':
                        result = f0_remeasure(screen, item, saved, ruler_map, labels, draws,
                            prepared=exp139_scores(screen, item, contrast['label']))
                    else:
                        result = f0_remeasure(screen, item, saved, ruler_map, labels, draws)
                except (ValueError, KeyError, TypeError) as exc:
                    result['reason'] = str(exc)
            contrasts.append(dict(label=contrast['label'], priority=contrast['priority'],
                old_verdict=contrast['old_verdict'], verdict_class=contrast['verdict_class'],
                old_check=check, saved_vector=saved, **result,
                changed_verdict=result['status'] in ('REOPEN_CANDIDATE','OVERTURNED')))
        output.append(dict(path=item['path'], inputs=attestation, contrasts=contrasts))
    return dict(schema='retro-remeasure-v1-step5', version=VERSION,
                registration_sha256=sha(root/REG), screens=output,
                counts=dict(screens=len(output), contrasts=sum(len(x['contrasts']) for x in output),
                            reopen_candidate=sum(c['status'] == 'REOPEN_CANDIDATE' for x in output for c in x['contrasts']),
                            overturned=sum(c['status'] == 'OVERTURNED' for x in output for c in x['contrasts']),
                            not_remeasurable=sum(c['status'] == 'NOT_REMEASURABLE' for x in output for c in x['contrasts'])))


def score_file(rel):
    frame = pd.read_csv(ROOT/rel, usecols=[UID, *T], dtype={UID: str})
    if frame[UID].duplicated().any():
        raise ValueError('Duplicate prediction IDs: ' + rel)
    return frame.set_index(UID)[list(T)].astype(float)


def pct(frame):
    return frame.rank(pct=True)


def point_scores():
    exp = 'artifacts/experiments/'
    c = score_file(exp+'EXP099_B8_DREAD_OOF/oof_k36.csv')
    rn = score_file(exp+'EXP101_RN34_DREAD_OOF/oof_k36.csv').loc[c.index]
    rad = score_file(exp+'EXP107_RADR50_DREAD_OOF/oof_k36.csv').loc[c.index]
    b224 = score_file(exp+'EXP084_B_G1/oof_best.csv')
    r384 = score_file(exp+'EXP088_RAPTOR384_G1_V4/oof_best.csv')
    bparts = []
    for b, r in (('EXP078_INPUT_B_F0','EXP079_R34_B_F0'),
                 ('EXP078_INPUT_B_F1','EXP079_R34_B_F1'),
                 ('EXP080_INPUT_B_F2','EXP080_R34_B_F2')):
        bb = score_file(exp+b+'/oof_best.csv')
        rr = score_file(exp+r+'/oof_best.csv').loc[bb.index]
        bparts.append(.9*pct(bb)+.1*pct(rr))
    # EXP140's registered G4343 proxy: the EXP134/EXP127 AUG1 OOF recipe,
    # two seeds with within-fold target ranks. EXP138 binds all these files.
    screen138 = json.loads((ROOT/(exp+'EXP138_BUNDLE_AUG1_MH/screen/result.json')).read_text(encoding='utf-8'))
    bound = screen138['input_bindings']
    proxy_seeds = []
    for seed in (42, 43):
        folds = []
        for fold in range(5):
            rel = exp+(f'EXP127_AUG1_S{seed}_F0' if fold == 0 else
                       f'EXP134_AUG1_S{seed}_F{fold}')+'/oof_k36.csv'
            if bound.get(rel) != sha(ROOT/rel):
                raise ValueError('EXP140 proxy binding mismatch: '+rel)
            provenance = ROOT/(exp+(f'EXP127_AUG1_S{seed}_F0' if fold == 0 else
                                    f'EXP134_AUG1_S{seed}_F{fold}')+'/provenance.json')
            if json.loads(provenance.read_text(encoding='utf-8')).get('label_set') != 'dread':
                raise ValueError('EXP140 proxy ancestry undeclared')
            folds.append(pct(score_file(rel)))
        proxy_seeds.append(pd.concat(folds))
    proxy = (proxy_seeds[0]+proxy_seeds[1].loc[proxy_seeds[0].index])/2
    return {
        'relabel5': score_file(exp+'EXP099_B8_RELABEL_V1_OOF/oof_k36.csv'),
        'coat5rn5': .7*pct(c)+.3*pct(rn), 'coat5': c,
        'student': score_file(exp+'EXP095_STUDENT_B8/oof_best.csv'),
        'E1': .5*pct(b224)+.5*pct(r384.loc[b224.index]),
        'R384': r384, 'B224': b224, 'B_RN3F': pd.concat(bparts),
        'R1': pd.concat([score_file(exp+f'EXP061_R1_F{k}/oof_best.csv') for k in range(5)]),
        'CNX5': score_file(exp+'EXP044_CONVNEXT_S_5FOLD/oof_pred.csv'),
        'rad5': rad, 'coat5rad5': .5*pct(c)+.5*pct(rad),
        'H-B-34-exp140-aug1': proxy,
    }, b224.index.tolist(), c.index.tolist(), proxy_seeds


def exp_auc(s, p):
    if len(s) < 2:
        return math.nan
    order = np.argsort(s, kind='mergesort')
    s, p = s[order], p[order]
    q = 1-p
    _, starts, counts = np.unique(s, return_index=True, return_counts=True)
    gq = np.add.reduceat(q, starts)
    less = np.r_[0., np.cumsum(gq)[:-1]]
    group = np.repeat(np.arange(len(starts)), counts)
    den = p.sum()*q.sum()-np.sum(p*q)
    if p.sum() < 2 or q.sum() < 2 or den <= 0:
        return math.nan
    return float(np.sum(p*(less[group]+.5*(gq[group]-q)))/den)


def auc_vector(scores, labels):
    ss = scores.to_numpy(float)
    yy = labels.reindex(scores.index).to_numpy(float)
    out = []
    for j in range(12):
        mask = np.isfinite(ss[:, j]) & np.isfinite(yy[:, j])
        out.append(exp_auc(ss[mask, j], yy[mask, j]))
    return out


def labels_for_map(ruler_map):
    labels = {}
    for name in ('jev_hybrid', 'aquila_v1', 'flight_v4', 'steven_v4',
                 'pilkwang_v2', 'dread', 'relabel_v1'):
        rel = ruler_map['sources'][name]['path']
        bound = ruler_map['inputs'][name]['sha256']
        if sha(ROOT/rel) != bound:
            raise ValueError('Map label hash mismatch: '+name)
        table = pd.read_csv(ROOT/rel, usecols=[UID, *T], dtype={UID: str}).drop_duplicates(UID).set_index(UID)[list(T)].astype(float)
        a = table.to_numpy()
        if name == 'dread':
            a = np.rint(a)
        elif name != 'jev_hybrid':
            a = np.where(np.isnan(a), np.nan, (a >= .5).astype(float))
        labels[name] = pd.DataFrame(a, index=table.index, columns=T)
    members = [labels[k] for k in ('dread','steven_v4','pilkwang_v2','relabel_v1')]
    aligned = members[0].index
    # Consensus masks are evaluated after alignment to a requested study set;
    # use union here so reindexing never creates an accidental unanimous cell.
    aligned = aligned.union(members[1].index).union(members[2].index).union(members[3].index)
    stack = np.stack([x.reindex(aligned).to_numpy(float) for x in members])
    finite = np.isfinite(stack).all(axis=0)
    unanimous = finite & ((stack == 0).all(axis=0) | (stack == 1).all(axis=0))
    labels['consensus'] = pd.DataFrame(np.where(unanimous, stack[0], np.nan),
                                       index=aligned, columns=T)
    return labels


def agreement(rows, rule, gap=.002):
    pairs = []
    for a, b in itertools.combinations(rows, 2):
        dy = a['public']-b['public']
        if abs(dy) + 1e-12 < gap:
            continue
        va, vb = a.get(rule), b.get(rule)
        if va is None or vb is None:
            continue
        pairs.append((a['point'], b['point'], int((va-vb)*dy > 0)))
    vals = [p[2] for p in pairs]
    return dict(agree=sum(vals), total=len(vals), pairs=pairs)


def exact_mcnemar(b, c):
    n = b+c
    if not n:
        return 1.
    return min(1., 2*sum(math.comb(n, k) for k in range(min(b, c)+1))/2**n)


def spearman(x, y):
    a = pd.Series(x).rank(method='average').to_numpy(float)
    b = pd.Series(y).rank(method='average').to_numpy(float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compare(rows, seed=20261010, draws=10000):
    amap = {x[:2]: x[2] for x in agreement(rows, 'R_map10')['pairs']}
    aold = {x[:2]: x[2] for x in agreement(rows, 'R_old')['pairs']}
    keys = sorted(set(amap)&set(aold))
    b = sum(amap[k] == 1 and aold[k] == 0 for k in keys)
    c = sum(amap[k] == 0 and aold[k] == 1 for k in keys)
    names = [x['point'] for x in rows]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        sampled = rng.choice(names, len(names), replace=True)
        counts = {name: int(np.sum(sampled == name)) for name in names}
        den = num = 0
        for (a, d) in keys:
            weight = counts[a]*counts[d]
            den += weight
            num += weight*(amap[(a,d)]-aold[(a,d)])
        if den:
            diffs.append(num/den)
    return dict(map_right_old_wrong=b, old_right_map_wrong=c, net_pairs=b-c,
                exact_mcnemar_p=exact_mcnemar(b, c),
                candidate_cluster_bootstrap_draws=len(diffs),
                difference_fraction_ci95=np.percentile(diffs, [2.5, 97.5]).tolist() if diffs else None)


def step6(reg, ruler_map):
    protocol = reg['step6_public_agreement_validation']
    specs = protocol['panel_a_primary']['candidates']
    scores, v888, g4343, proxy_seeds = point_scores()
    cohorts = {'V888': v888, 'C529': None, 'G4343': g4343}
    cohorts['C529'] = [u for u in v888 if u in scores['B_RN3F'].index]
    gold_spec = ruler_map['inputs']['gold']
    if sha(ROOT/gold_spec['path']) != gold_spec['sha256']:
        raise ValueError('Gold58 label hash mismatch')
    gold = set(pd.read_csv(ROOT/gold_spec['path'], usecols=[UID], dtype={UID: str})[UID])
    overlap = {name: len(gold.intersection(ids)) for name, ids in cohorts.items()}
    source_mapping = {}
    for ledger_name, map_name in (('flight','flight_v4'), ('steven_v4','steven_v4'),
                                  ('pilkwang_v2','pilkwang_v2'), ('jev_hybrid','jev_hybrid')):
        source = ruler_map['inputs'][map_name]
        if sha(ROOT/source['path']) != source['sha256']:
            raise ValueError('Ledger/map source binding mismatch: '+ledger_name)
        source_mapping[ledger_name] = dict(map_source=map_name, sha256=source['sha256'])
    labels = labels_for_map(ruler_map)
    points = pd.read_csv(ROOT/'.local/ruler-bench-20261007/points.csv')
    measures = pd.read_csv(ROOT/'artifacts/evidence_ledger/measurements.csv')
    output = {}
    for es, ids in cohorts.items():
        rows = []
        for spec in specs:
            name = spec['point']
            if es not in spec['eval_sets']:
                continue
            if name not in scores:
                rows.append(dict(point=name, public=spec['public'], status='NO_BOUND_PREDICTIONS'))
                continue
            s = scores[name].reindex(ids)
            if s.isna().any().any():
                raise ValueError('Incomplete bench point: '+es+'/'+name)
            if name == 'H-B-34-exp140-aug1':
                vectors = {ref: np.mean([auc_vector(seed.reindex(ids), labels[ref])
                                          for seed in proxy_seeds], axis=0).tolist()
                           for ref in labels}
            else:
                vectors = {ref: auc_vector(s, labels[ref]) for ref in labels}
            original = points[(points.eval_set == es)&(points.point == name)]
            proxy = name == 'H-B-34-exp140-aug1'
            if len(original) != (0 if proxy else 1):
                raise ValueError('Bench point not unique: '+es+'/'+name)
            if not proxy:
                for src, ref in (('dread','dread'), ('flight','flight_v4'),
                                 ('steven_v4','steven_v4'), ('pilkwang_v2','pilkwang_v2'),
                                 ('consensus','consensus'), ('relabel_v1','relabel_v1')):
                    old = float(original.iloc[0][src])
                    new = float(np.nanmean(vectors[ref]))
                    if abs(old-new) > 1e-9:
                        raise ValueError(f'Bench macro reproduction failed: {es}/{name}/{src}: {new-old}')
            exposed = set()
            for ref in spec['ruler_map_exposed_references']:
                source = ruler_map['sources'].get(ref)
                if source:
                    exposed.update(source['ancestors'])
                    exposed.add(ref)
            mapvals = {}
            for target in T:
                refs = ruler_map['recommendations'][target]['references']
                eligible = [r for r in refs if r not in exposed and
                            not exposed.intersection(ruler_map['sources'][r]['ancestors'])]
                j = T.index(target)
                vals = [vectors[r][j] for r in eligible if math.isfinite(vectors[r][j])]
                mapvals[target] = mean(vals) if vals else None
            highmed = [t for t in T if ruler_map['recommendations'][t]['validity'] != 'LOW']
            if not all(mapvals[t] is not None for t in highmed):
                raise ValueError('Map target unexpectedly unmeasurable: '+es+'/'+name)
            mrow = measures[(measures.key == spec['ledger_keys'][spec['eval_sets'].index(es)]) &
                            (measures.eval_set == es)]
            if proxy:
                jstored = float(mrow[mrow.source == 'jev_hybrid'].iloc[0].value)
                jnew = float(np.nanmean(vectors['jev_hybrid']))
                if abs(jstored-jnew) > 5e-5:
                    rows.append(dict(point=name, public=spec['public'],
                        status='NOT_REMEASURABLE',
                        reason='Registered proxy uses incompatible seed-mean and score-mean aggregations; JEV macro fails old-value reproduction',
                        jev_seed_mean=jnew, jev_stored=jstored))
                    continue
            for src in ('dread','flight','steven_v4','pilkwang_v2','jev_hybrid',
                        'consensus','relabel_v1'):
                stored = mrow[mrow.source == src]
                ref = {'flight':'flight_v4'}.get(src, src)
                if len(stored) and ref in vectors:
                    tolerance = 5e-5 if proxy else 1e-9
                    if abs(float(stored.iloc[0].value)-float(np.nanmean(vectors[ref]))) > tolerance:
                        raise ValueError('Ledger macro reproduction failed: '+es+'/'+name+'/'+src)
            jrow = mrow[mrow.source == 'jev_hybrid']
            jval = float(jrow.iloc[0].value) if len(jrow) else None
            crow = mrow[mrow.source == 'consensus']
            consensus = float(crow.iloc[0].value) if len(crow) else None
            rows.append(dict(point=name, public=spec['public'], status='MEASURED',
                R_map10=mean(mapvals[t] for t in highmed),
                R_map9=mean(mapvals[t] for t in highmed if t != 'PF OA'),
                R_old=mean((jval, consensus)) if jval is not None and consensus is not None else None,
                **({src: float(original.iloc[0][src]) for src in
                   ('dread','consensus','flight','steven_v4','pilkwang_v2','relabel_v1','yunus')}
                   if not proxy else {src: float(mrow[mrow.source == src].iloc[0].value)
                                      for src in ('dread','consensus','flight','relabel_v1')})))
        complete = [r for r in rows if r['status'] == 'MEASURED']
        ruler_names = ['R_map10','R_map9','R_old','dread','consensus','flight',
                       'steven_v4','pilkwang_v2','relabel_v1','yunus']
        rho = {}
        for k in ruler_names:
            complete_k = [r for r in complete if r.get(k) is not None]
            rho[k] = spearman([r[k] for r in complete_k],
                              [r['public'] for r in complete_k])
        result_agreement = {k: {z:v for z,v in agreement(complete,k).items() if z!='pairs'}
                            for k in ruler_names}
        expected = protocol['panel_a_primary']['per_eval_set'][es]
        if len(rows) != expected['candidates']:
            raise ValueError('Registered candidate count mismatch: '+es)
        if len(complete) == len(rows) and result_agreement['R_map10']['total'] != expected['countable_pairs']:
            raise ValueError('Registered countable pair count mismatch: '+es)
        output[es] = dict(points=rows, agreement=result_agreement,
                          spearman=rho, comparison=compare(complete), complete_points=len(complete),
                          registered_points=len(rows))
    c, v = output['C529'], output['V888']
    if c['complete_points'] != c['registered_points'] or v['complete_points'] != v['registered_points']:
        decision = 'D6_UNDETERMINED_INCOMPLETE_PANEL'
    else:
        cd = c['comparison']['net_pairs']; vd = v['comparison']['net_pairs']
        if cd <= -8 and vd < 0:
            decision = 'D6_a_DISCREDITED'
        elif cd >= 8 and c['comparison']['exact_mcnemar_p'] < .05 and vd >= 0:
            decision = 'D6_c_SUPPORTED'
        else:
            decision = 'D6_b_INCONCLUSIVE'
    headline = {es: output[es]['agreement']['R_map10']['agree'] >=
                output[es]['agreement']['R_old']['agree']
                for es in output if output[es]['complete_points'] == output[es]['registered_points']}
    return dict(schema='retro-remeasure-v1-step6', version=VERSION,
                registration_sha256=sha(ROOT/REG), decision=decision, strata=output,
                headline=dict(map_at_least_old_complete_strata=sum(headline.values()),
                              complete_strata=len(headline), by_stratum=headline),
                preconditions=dict(ledger_map_source_binding=source_mapping,
                                   gold58_overlap_counts=overlap,
                                   gold58_overlap_status='OVERLAP_PRESENT' if any(overlap.values()) else 'DISJOINT',
                                   bench_macro_reproduced=True),
                panel_b=dict(status='DESCRIPTIVE_ONLY', readings=protocol['panel_b_descriptive']['readings']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output_dir', type=Path)
    args = p.parse_args()
    reg = json.loads((ROOT/REG).read_text(encoding='utf-8'))
    checked = verify_registration(reg)
    ruler_map = json.loads((ROOT/MAP).read_text(encoding='utf-8'))
    if ruler_map['tool_version'] != reg['ruler_map']['version']:
        raise ValueError('Frozen map version changed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_new(args.output_dir/'step5.json', dict(step5(reg, ruler_map), input_verification=checked))
    write_new(args.output_dir/'step6.json', step6(reg, ruler_map))


if __name__ == '__main__':
    main()
