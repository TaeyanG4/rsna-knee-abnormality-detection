"""Versioned CPU-only saved-OOF error audit; see oof_error_audit_v2.md.

Import is passive. No training, inference, report text, submission or auto-tuning.
The frozen v1 stays intact. Registry data selects explicit model variants.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

if __name__ == '__main__':
    for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[_key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import oof_error_audit as legacy
from tools import exp169_bundle_screen as bundle_helper
from tools import exp115_vlm_eval as evaluation
from tools import exp177_bundle_aggregation_audit as aggregation

VERSION = '2.0.1'
TARGETS = tuple(legacy.LAB)
UID = legacy.UID
SEED, DRAWS, MAX_SECONDS = 20261003, 300, 1800
CORPUS = 'storage/cache/derived/RAPTOR64_336_C140_S02'
REGISTRY = 'tools/oof_error_audit_v2.registry.json'
REFERENCE_KEYS = {'jev': 'jev_hybrid', 'aquila': 'aquila_v1',
                  'dread': 'dread', 'consensus': 'consensus'}
FRACTIONS = (.20, .35, .50, .65, .80)
MONTAGE_GROUPS = {
    'meniscus': ('Sagittal_fluid1', 'Sagittal_fluid0', 'Coronal_fluid1', 'Coronal_fluid0'),
    'mcl': ('Coronal_fluid1', 'Coronal_fluid0', 'Axial'),
}
CONFUSIONS = {'MCL': ('Contusion', 'Effusion'), 'Fracture': ('Contusion',),
              'Lateral OA': ('Medial OA', 'PF OA'), 'PF OA': ('Medial OA', 'Lateral OA')}
SOURCE_FILES = (
    'tools/oof_error_audit_v2.py', 'tools/oof_error_audit_v2.md',
    'tests/test_oof_error_audit_v2.py', 'tools/oof_error_audit.py',
    'tools/exp149_label_sources.py', 'tools/exp115_vlm_eval.py',
    'tools/exp169_bundle_screen.py', 'tools/exp156_k_sweep.py',
    'tools/exp177_bundle_aggregation_audit.py', 'tools/exp175_image_domain_audit.py',
    'tools/ext_zeroshot.py', 'training/src/raptor_common.py',
)
need, sha, read, write = aggregation.need, aggregation.sha, aggregation.read, aggregation.write


def safe_path(root, relative):
    """Only explicit workspace paths; reject protected sources before opening."""
    p = Path(relative)
    need(not p.is_absolute() and not any(x in str(relative) for x in ('*', '?', '[')),
         'Paths must be explicit and workspace-relative')
    resolved = (Path(root) / p).resolve()
    need(resolved.is_relative_to(Path(root).resolve()), 'Path escapes workspace')
    parts = resolved.relative_to(Path(root).resolve()).as_posix().lower().split('/')
    need(not ('.local' in parts and 'relabel' in parts), 'Protected relabel source')
    need(not any(re.search(r'(^|[_.-])(credentials?|tokens?|secrets?|keys?)([_.-]|$)', s)
                 for s in parts), 'Protected credential source')
    need('/'.join(parts) != 'storage/competition/train.csv', 'Raw training CSV is not an audit input')
    return resolved


def unique_ids(ids, context):
    ids = list(ids)
    need(all(isinstance(u, str) and u.strip() for u in ids) and len(set(ids)) == len(ids),
         'Duplicate/empty study ID: ' + context)
    return ids


def verify_entry(root, entry):
    path = safe_path(root, entry['path'])
    digest = entry.get('sha256')
    need(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
         'Missing pinned sha256: ' + entry['path'])
    need(path.is_file() and sha(path) == digest, 'Input hash mismatch: ' + entry['path'])
    return path


def validate_registry(registry):
    need(registry.get('schema') == 'oof-error-audit-registry-v2' and
         registry.get('version') == VERSION, 'Registry schema/version differs')
    folds = registry['folds']
    need(folds and folds == sorted(set(folds)) and set(folds) <= set(range(5)),
         'Declare a nonempty, unique ordered subset of the five canonical folds')
    variants = registry['variants']
    unique_ids([v['id'] for v in variants], 'variant names')
    active = [v for v in variants if v['enabled']]
    need(active, 'No enabled variants')
    all_paths = []
    for v in active:
        need(all(v.get(k) for k in ('family', 'label_recipe', 'prediction_policy', 'path_regex')),
             'Incomplete variant identity: ' + v['id'])
        seeds = v['expected_seeds']
        need(seeds and len(seeds) == len(set(seeds)), 'Duplicate/empty seed roster')
        grid = [(m['seed'], m['fold']) for m in v['members']]
        need(len(grid) == len(set(grid)) and set(grid) == {(s, f) for s in seeds for f in registry['folds']},
             'Variant seed/fold grid incomplete: ' + v['id'])
        for m in v['members']:
            need(re.fullmatch(v['path_regex'], m['path']) is not None,
                 'Variant path identity mismatch: ' + v['id'])
            need(re.fullmatch('[0-9a-f]{64}', str(m.get('sha256'))) is not None,
                 'Enabled variant has unbound input')
            all_paths.append(m['path'])
    unique_ids(all_paths, 'prediction paths across variants')
    names = {v['id'] for v in active}
    bundle_names = []
    for b in registry['bundles']:
        need(set(b['members']) == set(bundle_helper.FAMILIES) and
             len(set(b['members'].values())) == 3 and set(b['members'].values()) <= names,
             'Bundle must name three enabled, distinct variants by registered roles')
        need(b['id'] not in names, 'Bundle/variant name collision')
        bundle_names.append(b['id'])
    unique_ids(bundle_names, 'bundle names')
    need('bundle' in bundle_names, 'Primary diagnostic bundle must be named bundle')
    return active


def aggregate_variants(registry, ids_by_fold, raw):
    """raw[(variant, seed, fold)] -> probabilities; exact registered rank path."""
    active = validate_registry(registry)
    unique_ids([u for f in registry['folds'] for u in ids_by_fold[f]], 'pooled roster')
    pooled = {}
    for v in active:
        sections = []
        for f in registry['folds']:
            ranked_seeds = []
            for s in v['expected_seeds']:
                p = np.asarray(raw[v['id'], s, f], dtype=float)
                need(p.shape == (len(ids_by_fold[f]), 12) and len(p) > 0 and
                     np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid fold predictions')
                ranked_seeds.append(evaluation.pct_rank(p))
            sections.append(np.mean(ranked_seeds, axis=0))
        pooled[v['id']] = np.concatenate(sections)
    families = {name: bundle_helper.global_rank(p) for name, p in pooled.items()}
    for b in registry['bundles']:
        families[b['id']], _ = bundle_helper.bundle_of({role: pooled[name] for role, name in b['members'].items()})
    return families


def checked_reference(ref, n):
    y, mask, minimum = ref
    y, mask = np.asarray(y, float), np.asarray(mask)
    need(y.shape == mask.shape == (n, 12) and mask.dtype == bool and minimum in (1, 2),
         'Invalid reference shape/mask/minimum')
    need(np.isfinite(y[mask]).all() and np.isin(y[mask], (0, 1)).all(), 'Invalid known reference label')
    # UNKNOWN cells stay excluded, including NaN; never cast NaN to True.
    return np.where(mask, y, 0), mask.copy(), minimum


def reference_paths():
    from tools import exp156_k_sweep as refs
    return sorted(set((refs.JEV, refs.AQUILA, refs.DREAD, *refs.CONSENSUS)))


def assert_reference_ids(path):
    # The registered load_soft drops duplicates. Reject them BEFORE that helper.
    columns = pd.read_csv(path, nrows=0).columns
    # Pilkwang's registered file also carries discrete confidence/verdict columns.
    allowed = {UID, *TARGETS, *(t + suffix for t in TARGETS for suffix in ('_soft', '__conf', '__verdict'))}
    need(set(columns) <= allowed, 'Unexpected non-label column in reference file')
    ids = pd.read_csv(path, usecols=[UID], dtype={UID: str})[UID]
    need(not ids.isna().any(), 'Missing reference study IDs')
    unique_ids(ids.tolist(), 'reference ' + str(path))


def load_inputs(root, registry, budget):
    from tools import exp149_label_sources as rosters
    from tools import exp156_k_sweep as refs
    active = validate_registry(registry)
    entries = [registry['roster_manifest'], *registry['bindings']]
    for v in active:
        entries.extend(v['members'])
        entries.extend(m['summary'] for m in v['members'] if 'summary' in m)
    bound = {}
    for entry in entries:
        budget.check()
        verify_entry(root, entry)
        need(entry['path'] not in bound or bound[entry['path']] == entry['sha256'], 'Conflicting input bindings')
        bound[entry['path']] = entry['sha256']
    manifest = read(safe_path(root, registry['roster_manifest']['path']))
    required = {manifest['paths']['base_ordered_ids'], evaluation.COMPONENTS, *reference_paths()}
    # fold_rosters validates the full split even for an explicitly registered
    # common-fold analysis of a new arm; bind every file it will read.
    required.update(manifest['folds'][str(f)]['ordered_ids']['path'] for f in range(5))
    need(required <= set(bound), 'Roster/reference/component input missing from registry bindings')
    for f in range(5):
        e = manifest['folds'][str(f)]['ordered_ids']
        need(bound[e['path']] == e['sha256'], 'Frozen roster manifest hash differs')
    canonical, fold_of = rosters.fold_rosters(manifest, root)
    unique_ids(canonical, 'canonical roster')
    ids_by_fold = {f: [u for u in canonical if fold_of[u] == f] for f in registry['folds']}
    ids = [u for f in registry['folds'] for u in ids_by_fold[f]]
    need(len(ids) == registry['expected_studies'], 'Declared common-fold cohort size differs')
    raw = {}
    for v in active:
        for m in v['members']:
            budget.check()
            if 'summary' in m:
                summary = read(safe_path(root, m['summary']['path']))
                for dotted, expected in m.get('identity', {}).items():
                    actual = summary
                    for part in dotted.split('.'):
                        need(isinstance(actual, dict) and part in actual, 'Missing run identity: ' + dotted)
                        actual = actual[part]
                    need(actual == expected, 'Run identity differs: ' + dotted)
            raw[v['id'], m['seed'], m['fold']] = bundle_helper.load_oof(
                safe_path(root, m['path']), ids_by_fold[m['fold']])
    families = aggregate_variants(registry, ids_by_fold, raw)
    for path in reference_paths():
        assert_reference_ids(safe_path(root, path))
    references = refs.references(ids, root)
    references = {k: checked_reference(references[v], len(ids)) for k, v in REFERENCE_KEYS.items()}
    weights, bootstrap = aggregation.component_draws(ids, fold_of, root)
    need(weights.shape == (DRAWS, len(ids)) and bootstrap['seed'] == SEED, 'Bootstrap contract differs')
    return ids, np.array([fold_of[u] for u in ids]), families, references, weights, bootstrap, bound


def metric_tables(families, refs, folds, weights, budget):
    """Reuse v1 metrics/deficits; mask-aware vectorized v1 weighted AUC for draws."""
    rows, errors, draws, points = [], {}, {}, {}
    for name, p in families.items():
        for refname, raw_ref in refs.items():
            y, mask, minimum = checked_reference(raw_ref, len(p))
            budget.check()
            # v1's conditional_intervals hardcodes R1, all-known cells and no
            # minimum class size. EXP177 strengthens that same weighted statistic.
            draws[name, refname] = aggregation.weighted_auc(p, (y, mask, minimum), weights, budget)
            points[name, refname] = np.full(12, np.nan)
            for fold in ['all', *sorted(set(folds.tolist()))]:
                in_fold = np.ones(len(p), bool) if fold == 'all' else folds == fold
                dense = legacy.target_metrics(y[in_fold], p[in_fold]) if mask[in_fold].all() else None
                for j, target in enumerate(TARGETS):
                    keep = in_fold & mask[:, j]
                    yy, pp = y[keep, j], p[keep, j]
                    pos, neg = int(yy.sum()), int(len(yy) - yy.sum())
                    valid = pos >= minimum and neg >= minimum
                    value = None
                    if valid:
                        value = legacy.weighted_auc(yy, pp, np.ones(len(yy)))
                        if dense is not None:
                            need(np.isclose(value, dense['targets'][j]['auc']), 'v1 metric disagreement')
                    row = dict(family=name, reference=refname, target=target, fold=fold,
                               n=int(keep.sum()), excluded=int(in_fold.sum() - keep.sum()),
                               pos=pos, neg=neg, min_class=minimum, auc=value,
                               status='OK' if valid else 'INSUFFICIENT_CLASSES',
                               macro_deficit=None if value is None else (1-value)/12)
                    rows.append(row)
                    if fold == 'all':
                        points[name, refname][j] = np.nan if value is None else value
                        contribution = np.full(len(p), np.nan)
                        if valid:
                            contribution[keep] = legacy.row_error_contributions(yy, pp)
                            need(np.isclose(np.nansum(contribution), (1-value)/12), 'Deficit allocation mismatch')
                        errors[name, refname, target] = contribution
    pairs = []
    for left in families:
        for right in families:
            if left == right:
                continue
            for refname in refs:
                delta = points[right, refname] - points[left, refname]
                dd = draws[right, refname] - draws[left, refname]
                for j, target in enumerate(TARGETS):
                    pairs.append(dict(family_A=left, family_B=right, reference=refname, target=target,
                                      delta_auc=None if not np.isfinite(delta[j]) else float(delta[j]),
                                      **aggregation.interval(dd[:, j])))
                # Strict macro12: a draw missing ANY target is invalid (never nanmean).
                macro = np.mean(delta)
                pairs.append(dict(family_A=left, family_B=right, reference=refname, target='macro12',
                                  delta_auc=None if not np.isfinite(macro) else float(macro),
                                  **aggregation.interval(dd.mean(axis=1))))
    return rows, pairs, errors


def error_strata(predictions, refs):
    """v1.1 thresholds; agreement requires ALL three known labels, per cell."""
    n = len(predictions)
    ref = {k: checked_reference(refs[k], n) for k in ('jev', 'aquila', 'dread')}
    y, known, _ = ref['jev']
    common = np.logical_and.reduce([v[1] for v in ref.values()])
    agree_pos = common & np.logical_and.reduce([v[0] == 1 for v in ref.values()])
    agree_neg = common & np.logical_and.reduce([v[0] == 0 for v in ref.values()])
    positive, negative = known & (y == 1), known & (y == 0)
    prevalence = np.divide(positive.sum(0), known.sum(0), out=np.full(12, np.nan), where=known.sum(0) > 0)
    threshold = 1 - prevalence/2
    fn = positive & (predictions < .5)
    fp = negative & (predictions > threshold)
    strata = dict(FN=fn, FP=fp, agreed_pos_FN=fn & agree_pos,
                  agreed_neg_FP=fp & agree_neg, disputed_FN=fn & common & ~agree_pos,
                  disputed_FP=fp & common & ~agree_neg,
                  unknown_agreement_FN=fn & ~common, unknown_agreement_FP=fp & ~common,
                  agreed_pos_TP_controls=agree_pos & (predictions > .9),
                  agreed_neg_low=agree_neg & (predictions < .5),
                  agreed_neg_middle=agree_neg & (predictions >= .5) & (predictions <= threshold),
                  agreed_neg_TN_controls=agree_neg & (predictions < .3))
    rows = []
    for j, target in enumerate(TARGETS):
        row = dict(target=target, jev_known=int(known[:, j].sum()), jev_pos=int(positive[:, j].sum()),
                   jev_neg=int(negative[:, j].sum()), fp_threshold_rank=threshold[j])
        row.update({k: int(v[:, j].sum()) for k, v in strata.items()})
        for kind, base, agreed in [('FN', positive, agree_pos), ('FP', negative, agree_neg)]:
            evaluable = base[:, j] & common[:, j]
            error_evaluable = strata[kind][:, j] & common[:, j]
            row[kind + '_agreement_evaluable'] = int(error_evaluable.sum())
            row[kind + '_disputed_share'] = (float((error_evaluable & ~agreed[:, j]).sum() / error_evaluable.sum())
                                             if error_evaluable.any() else None)
            row['baseline_disputed_share_among_jev_' + ('pos' if kind == 'FN' else 'neg')] = (
                float((evaluable & ~agreed[:, j]).sum() / evaluable.sum()) if evaluable.any() else None)
            row[kind + '_baseline_agreement_evaluable'] = int(evaluable.sum())
        rows.append(row)
    return strata, rows, threshold


def family_rescue(families, refs, pairs):
    strata = {n: error_strata(p, refs)[0] for n, p in families.items()}
    rows = []
    for pair in pairs:
        if pair['target'] == 'macro12':
            continue
        j = TARGETS.index(pair['target'])
        a, b = strata[pair['family_A']], strata[pair['family_B']]
        row = dict(pair)
        for kind, group in [('FN', 'agreed_pos_FN'), ('FP', 'agreed_neg_FP')]:
            source = a[group][:, j]
            rescued = source & ~b[kind][:, j]
            row['A_agreed_' + kind] = int(source.sum())
            row['B_rescues_' + kind] = int(rescued.sum())
            row['B_rescue_' + kind + '_share'] = float(rescued.sum()/source.sum()) if source.any() else None
        # Rescue counts always JEV/aquila/Dread-agreed; AUC uses this row's reference
        # on the full paired cohort, NOT the outcome-selected error subset.
        rows.append(row)
    return rows, strata


def target_confusion(bundle, refs, strata):
    y, known, _ = checked_reference(refs['jev'], len(bundle))
    rows = []
    for target, others in CONFUSIONS.items():
        j = TARGETS.index(target)
        for stratum in ('agreed_neg_FP', 'agreed_neg_low', 'agreed_neg_middle'):
            selected = strata[stratum][:, j]
            for other in others:
                k = TARGETS.index(other)
                ok = selected & known[:, k]
                rows.append(dict(target=target, stratum=stratum, other_target=other,
                                 stratum_n=int(selected.sum()), other_known=int(ok.sum()),
                                 other_unknown=int((selected & ~known[:, k]).sum()),
                                 other_jev_positive=int(y[ok, k].sum()),
                                 other_jev_positive_rate=float(y[ok, k].mean()) if ok.any() else None,
                                 other_bundle_rank_mean=float(bundle[selected, k].mean()) if selected.any() else None))
    return rows


def match_mcl_controls(ids, folds, refs, strata):
    """Exact JEV contusion/effusion match, same fold, no replacement; no forced matches."""
    y, mask, _ = checked_reference(refs['jev'], len(ids))
    mcl = TARGETS.index('MCL')
    confounders = [TARGETS.index('Contusion'), TARGETS.index('Effusion')]
    valid = mask[:, confounders].all(1)
    cases = np.flatnonzero(strata['agreed_neg_FP'][:, mcl])
    controls = set(np.flatnonzero(strata['agreed_neg_low'][:, mcl] & valid).tolist())
    rng = np.random.default_rng(SEED)
    order = sorted(cases.tolist(), key=lambda i: ids[i])
    order = [order[i] for i in rng.permutation(len(order))]
    matches = []
    for i in order:
        eligible = sorted((k for k in controls if folds[k] == folds[i] and
                           np.array_equal(y[k, confounders], y[i, confounders])), key=lambda k: ids[k]) if valid[i] else []
        k = eligible[int(rng.integers(len(eligible)))] if eligible else None
        if k is not None:
            controls.remove(k)
        matches.append(dict(case_id=ids[i], control_id=None if k is None else ids[k],
                            status='MATCHED' if k is not None else ('UNKNOWN_CONFOUNDER' if not valid[i] else 'NO_EXACT_MATCH')))
    return matches


def blind_cases(ids, family_strata, matches, per_stratum=8):
    """Select first, combine roles per study, then independently shuffle blind codes."""
    need(1 <= per_stratum <= 8, 'Montage sample per stratum must be 1..8')
    strata = family_strata['bundle']
    rng = np.random.default_rng(SEED)
    selected = {}

    def add(uid, anatomy, role):
        selected.setdefault((anatomy, uid), set()).add(role)

    def sample(target, name, anatomy):
        pool = sorted([ids[i] for i in np.flatnonzero(strata[name][:, TARGETS.index(target)])])
        chosen = rng.choice(pool, size=min(per_stratum, len(pool)), replace=False).tolist()
        for uid in chosen:
            add(uid, anatomy, target + ':' + name)
        return chosen

    for target in ('Medial Meniscus', 'Lateral Meniscus'):
        sample(target, 'agreed_pos_FN', 'meniscus')
        sample(target, 'agreed_pos_TP_controls', 'meniscus')
    chosen_fp = sample('MCL', 'agreed_neg_FP', 'mcl')
    sample('MCL', 'agreed_neg_TN_controls', 'mcl')
    for pair in matches:
        if pair['case_id'] in chosen_fp and pair['control_id'] is not None:
            add(pair['control_id'], 'mcl', 'MCL:matched_negative_control')
    # Codes do not encode the stratum, UID or target. PNGs have no key metadata.
    keys = sorted(selected)
    order = np.random.default_rng(SEED + 1).permutation(len(keys))
    cases = []
    for i, k in enumerate(order):
        anatomy, uid = keys[k]
        cases.append(dict(code=f'C{i+1:04d}', filename=f'panel_{i+1:04d}.png',
                          anatomy=anatomy, study_id=uid, strata=sorted(selected[anatomy, uid])))
    return cases


def slot_indices(corpus, uid, group):
    from tools.ext_zeroshot import groups
    bounds = {name: (a, b) for name, a, b in groups(corpus.spec)}
    a, b = bounds[group]
    filled = np.flatnonzero(np.asarray(corpus.mask(uid))[a:b] > 0) + a
    return [None if not len(filled) else int(filled[min(len(filled)-1, int(q*(len(filled)-1)+.5))])
            for q in FRACTIONS]


def render_blind_panel(corpus, case, path, tile=168):
    """Renderer accepts ONLY public case code/anatomy + UID for image access.

    Role/stratum, scores, reference labels and study IDs are never rendered.
    Returns the literal rendered text to make the blinding contract testable.
    """
    from PIL import Image, ImageDraw
    need(re.fullmatch(r'C[0-9]{4}', case['code']) is not None, 'Non-neutral case code')
    groups = MONTAGE_GROUPS[case['anatomy']]
    image = Image.new('L', (tile*5, 28 + len(groups)*(tile+24)), 0)
    draw = ImageDraw.Draw(image)
    texts = [case['code']]
    draw.text((4, 5), case['code'], fill=255)
    volume = corpus.volume(case['study_id'])
    indices = {}
    for r, group in enumerate(groups):
        indices[group] = slot_indices(corpus, case['study_id'], group)
        for c, (q, slot) in enumerate(zip(FRACTIONS, indices[group])):
            x, yy = c*tile, 28+r*(tile+24)
            caption = f'{group} {q:.0%} s={slot if slot is not None else "NA"}'
            texts.append(caption)
            draw.text((x+2, yy+3), caption, fill=255)
            if slot is not None:
                pixels = np.asarray(volume[slot])
                need(pixels.dtype == np.uint8 and pixels.ndim == 2, 'Unexpected production image dtype/shape')
                image.paste(Image.fromarray(pixels).resize((tile, tile)), (x, yy+24))
    image.save(path, format='PNG')
    return dict(code=case['code'], filename=Path(path).name, rendered_text=texts, slot_indices=indices)


def source_hashes(root):
    return {p: sha(safe_path(root, p)) for p in SOURCE_FILES}


def verify_note(root, path, registry_path):
    path = Path(path).resolve()
    need(path.is_relative_to(Path(root).resolve()), 'Verification note must be in workspace')
    path = safe_path(root, path.relative_to(Path(root).resolve()).as_posix())
    note = read(path)
    need(note.get('status') == 'PASS' and note.get('tool_version') == VERSION and
         note.get('reviewer') == 'gpt-6-astra' and note.get('separate_pass') is True,
         'A separate astra PASS verification receipt is required')
    need(note.get('source_sha256') == source_hashes(root) and
         note.get('registry_sha256') == sha(registry_path), 'Verification is stale for code/registry')
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def write_csv(path, rows, metadata):
    pd.DataFrame([{**metadata, **row} for row in rows]).to_csv(path, index=False)


def run(root, registry_path, output, verification_note, threads=1, seconds=MAX_SECONDS, per_stratum=8):
    root, registry_path, output = Path(root).resolve(), Path(registry_path).resolve(), Path(output).resolve()
    need(output.is_relative_to(root) and not output.exists(), 'Output must be a NEW workspace directory')
    need(registry_path.is_relative_to(root), 'Registry must be in workspace')
    registry_path = safe_path(root, registry_path.relative_to(root).as_posix())
    need(0 < seconds <= MAX_SECONDS, 'CPU and wall budget must be <= 1800 seconds')
    need(1 <= per_stratum <= 8, 'Montage sample must be 1..8')
    with aggregation.cpu_limits(threads):
        budget = aggregation.Budget(seconds)
        verification = verify_note(root, verification_note, registry_path)
        registry_sha = sha(registry_path)
        registry = read(registry_path)
        output.mkdir(parents=True, exist_ok=False)
        metadata = dict(tool_version=VERSION, verification_note=verification['path'],
                        verification_sha256=verification['sha256'], registry_sha256=registry_sha)
        write(output/'RUNNING.json', {**metadata, 'status': 'RUNNING'})
        try:
            ids, folds, families, refs, weights, bootstrap, bound = load_inputs(root, registry, budget)
            rows, pairs, contributions = metric_tables(families, refs, folds, weights, budget)
            rescues, family_strata = family_rescue(families, refs, pairs)
            all_strata, lists = [], {}
            for name, p in families.items():
                s, r, _ = error_strata(p, refs)
                all_strata.extend(dict(family=name, **v) for v in r)
                lists[name] = {target: {k: [ids[i] for i in np.flatnonzero(v[:, j])] for k, v in s.items()}
                               for j, target in enumerate(TARGETS)}
            variants = [v['id'] for v in registry['variants'] if v['enabled']]
            lists['all_enabled_variants'] = {target: {
                kind: [ids[i] for i in np.flatnonzero(np.logical_and.reduce(
                    [family_strata[n][kind][:, j] for n in variants]))]
                for kind in ('agreed_pos_FN', 'agreed_neg_FP')} for j, target in enumerate(TARGETS)}
            rescue_ids = {a: {b: {target: {
                kind: [ids[i] for i in np.flatnonzero(family_strata[a][group][:, j] &
                                                     ~family_strata[b][kind][:, j])]
                for kind, group in [('FN', 'agreed_pos_FN'), ('FP', 'agreed_neg_FP')]}
                for j, target in enumerate(TARGETS)} for b in families if b != a} for a in families}
            confusion = target_confusion(families['bundle'], refs, family_strata['bundle'])
            matches = match_mcl_controls(ids, folds, refs, family_strata['bundle'])
            cases = blind_cases(ids, family_strata, matches, per_stratum)
            from tools.exp175_image_domain_audit import open_corpus
            corpus = open_corpus(safe_path(root, CORPUS))
            need(all(u in corpus for u in ids), 'OOF studies missing from production cache')
            cache_metadata = {CORPUS+'/'+name: sha(safe_path(root, CORPUS+'/'+name))
                              for name in ('manifest.json', 'ids.npy', 'masks.npy')}
            montage_dir = output/'blind'
            montage_dir.mkdir()
            panels = []
            for case in cases:
                budget.check()
                public_case = {k: case[k] for k in ('code', 'anatomy', 'study_id')}
                panel = render_blind_panel(corpus, public_case, montage_dir/case['filename'])
                panel['sha256'] = sha(montage_dir/case['filename'])
                panels.append(panel)
            # Output no label/prediction rows: metrics are aggregate; lists/keys use IDs.
            write_csv(output/'auc_family_reference_target_fold.csv', rows, metadata)
            write_csv(output/'paired_auc_deltas.csv', pairs, metadata)
            write_csv(output/'family_rescue.csv', rescues, metadata)
            write_csv(output/'error_strata_by_target.csv', all_strata, metadata)
            write_csv(output/'target_confusion.csv', confusion, metadata)
            write(output/'strata_study_ids.json', {**metadata, 'strata': lists})
            write(output/'rescue_study_ids.json', {**metadata, 'rescues': rescue_ids})
            write(output/'montage_key.json', {**metadata, 'cases': cases, 'mcl_matches': matches})
            write(output/'blind'/'index.json', {**metadata, 'panels': panels})
            # V1 allocation retained as aggregate checks, not a second ranking/selection rule.
            deficits = [{'family': n, 'reference': r, 'target': t,
                         'known_n': int(np.isfinite(v).sum()),
                         'allocated_macro_deficit': float(np.nansum(v)) if np.isfinite(v).any() else None}
                        for (n, r, t), v in contributions.items()]
            write_csv(output/'allocated_pair_deficits.csv', deficits, metadata)
            for path, expected in {**bound, **cache_metadata}.items():
                budget.check()
                need(sha(safe_path(root, path)) == expected, 'Input changed during audit: ' + path)
            need(sha(registry_path) == registry_sha, 'Registry changed during audit')
            verify_note(root, verification_note, registry_path)
            budget.check()
            summary = {**metadata, 'status': 'COMPLETE', 'studies': len(ids), 'targets': list(TARGETS),
                       'analysis_folds': registry['folds'],
                       'variants': registry['variants'], 'bundles': registry['bundles'], 'bootstrap': bootstrap,
                       'inputs': bound, 'cache_metadata_sha256': cache_metadata,
                       'cache_pixels': 'Production Corpus mmap; selected PNG bytes hashed; full shards NOT rehashed.',
                       'source_sha256': source_hashes(root), 'resources': {'threads': threads, **budget.usage()},
                       'montage_cases': len(cases), 'mcl_matching': 'Exact JEV contusion/effusion and fold; without replacement; unmatched retained.',
                       'limitations': ['Exploratory weak-reference error strata; FN/FP are rank heuristics, not clinical errors.',
                                       'JEV/aquila/Dread share report ancestry; aquila is an L2 training input; consensus is not independent truth.',
                                       'Intervals condition on saved predictions, fixed ranks and exposed folds, not retraining/seed/private uncertainty.',
                                       'Paired deltas are full-cohort B-A, not oracle rescue gains; no weight or target-routing selection.',
                                       'All twelve targets required for macro12; invalid bootstrap draws reported; no multiplicity correction.',
                                       'MCL matching is observational and can leave cases unmatched.',
                                       'Filled cache slots do not establish clinical coverage or native slice adjacency.',
                                       'Montage blinding hides strata/IDs/scores, not anatomy or image appearance.']}
            # COMPLETE is written last; process exit alone never certifies this package.
            write(output/'result.json', summary)
            return dict(status='COMPLETE', output=str(output), studies=len(ids), panels=len(cases))
        except Exception as exc:
            write(output/'FAILED.json', {**metadata, 'status': 'INCOMPLETE', 'reason': type(exc).__name__ + ': ' + str(exc),
                                        'resources': budget.usage()})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--registry', type=Path, default=ROOT/REGISTRY)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verification-note', type=Path, required=True)
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--max-seconds', type=int, default=MAX_SECONDS)
    parser.add_argument('--montage-per-stratum', type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.registry, args.output, args.verification_note,
                         args.threads, args.max_seconds, args.montage_per_stratum)))


if __name__ == '__main__':
    main()
