"""H-B-42 per-target rulers v1.0.2; CPU-only companion to evidence_ledger.

v1.0.2: LOW-validity targets are report-only in composite decisions.

No candidate predictions/public scores enter map construction. Existing ledger
CSVs are never written. CLI outputs aggregates only; all artifacts are exclusive.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# Apply before numerical imports, including when this module is used as a CLI.
for _env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_env] = '2'

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:
    from contextlib import contextmanager
    @contextmanager
    def threadpool_limits(limits):
        # Thread environment variables above still cap numerical libraries.
        yield
from tools import evidence_ledger as ledger
from tools import exp150_consensus_weight as e150
from tools import exp156_k_sweep as e156
from tools import exp115_vlm_eval as metrics

VERSION = '1.0.2'
SCHEMA = 'evidence-rulers-v1'
T = tuple(e156.TARGETS)
UID = e156.UID
GOLD = 'artifacts/experiments/EXP165_GOLD_TRAIN/gold-labels.csv'
PROTOCOL = 'artifacts/evidence_ledger/rulers-v1/PROTOCOL.md'
BINDINGS = ('tools/evidence_ledger_rulers.py', 'tests/test_evidence_ledger_rulers.py',
            'tools/evidence_ledger.py', 'tools/exp150_consensus_weight.py',
            'tools/exp156_k_sweep.py', 'tools/exp115_vlm_eval.py',
            'tools/exp115_vlm_pilot.py', PROTOCOL)
MARGIN = .003
NOISE_ASSUMPTIONS = (
    'class_conditional_score_independent_noise',
    'confusion_rates_transport_to_exact_screen_mask',
    'true_screen_prevalence_supplied',
    'paired_same_population_binary_reference_auc',
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return e150.sha(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    e150.write_new(path, value)


def source_specs():
    """Registered paths and explicit shared ancestry, not independent votes."""
    need(tuple(e150.T) == T == tuple(metrics.T), 'Target contracts differ')
    paths = {'jev_hybrid': e156.JEV, 'aquila_v1': e156.AQUILA,
             **e150.SETS, 'flight_v4': e150.FLIGHT}
    need(paths['dread'] == e156.DREAD and
         tuple(e150.SETS.values()) == tuple(e156.CONSENSUS), 'Reference contracts differ')
    specs = {}
    for name, path in paths.items():
        ancestors = [name]
        if name == 'flight_v4':
            # D-B-035: flight is built from Steven/Pilkwang, not aquila.
            ancestors = ['flight_v4', 'steven_v4', 'pilkwang_v2']
        elif name == 'aquila_v1':
            # D-B-083 correction: images + flight labels + report text -> aquila.
            # Include transitive label ancestry for training AND teacher exposure.
            ancestors = ['aquila_v1', 'flight_v4', 'steven_v4', 'pilkwang_v2']
        specs[name] = dict(path=path, rule='binary' if name == 'jev_hybrid' else
                          'rint' if name == 'dread' else 'ge0.5',
                          ancestors=ancestors, evidence_group='shared_report_origin',
                          provenance='weak reference; shared reports/errors; Gold58 exposed')
    return specs


def load_table(path, ids=None, binary=False):
    """Strengthens exp150.table: selective columns, no silent coercion/dedup.

    Only UID and the twelve plain or *_soft columns are materialised. Errors
    deliberately do not include cell values or label rows.
    """
    allowed = {UID, *T, *(t + '_soft' for t in T)}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    selected_header = [c for c in header if c in allowed]
    need(len(selected_header) == len(set(selected_header)), 'Duplicate target/UID column header')
    frame = pd.read_csv(path, usecols=lambda c: c in allowed, dtype=str,
                        keep_default_na=False, encoding='utf-8-sig')
    need(UID in frame, 'Missing UID column')
    need(not frame[UID].eq('').any() and frame[UID].eq(frame[UID].str.strip()).all()
         and not frame[UID].duplicated().any(),
         'Blank or duplicate UIDs')
    cols = []
    for target in T:
        found = [c for c in (target, target + '_soft') if c in frame]
        need(len(found) == 1, 'Missing or ambiguous target column: ' + target)
        cols.extend(found)
    raw = frame[cols].replace({'': np.nan, 'NaN': np.nan, 'nan': np.nan,
                               'NA': np.nan, 'N/A': np.nan})
    try:
        values = raw.apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    except (ValueError, TypeError):
        raise ValueError('Malformed numeric label cell') from None
    validate_values(values, binary)
    roster = frame[UID].tolist()
    if ids is not None:
        need(len(set(ids)) == len(ids), 'Duplicate requested UIDs')
        values = pd.DataFrame(values, index=roster).reindex(ids).to_numpy(dtype=np.float64)
        roster = list(ids)
    return roster, values


def validate_values(values, binary=False):
    need(values.ndim == 2 and values.shape[1] == len(T), 'Wrong label matrix shape')
    need(not np.isinf(values).any(), 'Infinite label value')
    finite = values[np.isfinite(values)]
    need(((finite >= 0) & (finite <= 1)).all(), 'Label outside [0,1]')
    if binary:
        need(np.isin(finite, (0, 1)).all(), 'Nonbinary label')


def hard_labels(values, rule):
    validate_values(values, binary=rule == 'binary')
    if rule == 'binary':
        return values.copy()
    need(rule in ('rint', 'ge0.5'), 'Unknown threshold rule')
    return metrics.binarise(values, rule)


def consensus(arrays):
    """Keep all-member finite mask BEFORE thresholding (never NaN -> negative)."""
    stack = np.stack(arrays)
    valid = np.isfinite(stack).all(axis=0)
    votes = stack >= .5
    agree = valid & (votes.all(axis=0) | (~votes).all(axis=0))
    # Avoid all-NaN mean warnings and keep the score undefined outside unanimity.
    soft = np.where(agree, np.where(np.isfinite(stack), stack, 0).mean(axis=0), np.nan)
    hard = np.where(agree, votes[0].astype(float), np.nan)
    return soft, hard, valid, agree


def ratio(n, d):
    return float(n / d) if d else None


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    centre = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., centre-half), min(1., centre+half)]


def cell_metrics(gold, soft, hard, mask):
    """One target; denominators distinguish selection from correctness."""
    selected = mask & np.isfinite(soft) & np.isfinite(hard)
    g, h, s = gold[selected], hard[selected], soft[selected]
    n = len(g)
    pos, neg = int((g == 1).sum()), int((g == 0).sum())
    tp, tn = int(((g == 1) & (h == 1)).sum()), int(((g == 0) & (h == 0)).sum())
    fn, fp = pos-tp, neg-tn
    se, sp = ratio(tp, pos), ratio(tn, neg)
    auc = metrics.auc_targets(g[:, None], s[:, None], np.ones((n, 1), bool), 1)[0]
    return dict(n_gold=len(gold), n=n, coverage=ratio(n, len(gold)),
                gold_positive_total=int(gold.sum()), gold_negative_total=int((gold == 0).sum()),
                positive_coverage=ratio(pos, int(gold.sum())),
                negative_coverage=ratio(neg, int((gold == 0).sum())),
                positives=pos, negatives=neg, tp=tp, fn=fn, tn=tn, fp=fp,
                agreement_count=tp+tn, disagreement_count=fp+fn,
                accuracy=ratio(tp+tn, n), disagreement=ratio(fp+fn, n),
                sensitivity=se, specificity=sp, fnr=ratio(fn, pos), fpr=ratio(fp, neg),
                balanced_accuracy=(se+sp)/2 if se is not None and sp is not None else None,
                soft_auc=float(auc) if np.isfinite(auc) else None,
                accuracy_ci95=wilson(tp+tn, n), sensitivity_ci95=wilson(tp, pos),
                specificity_ci95=wilson(tn, neg))


def validity(m, copy_suspect=False):
    if copy_suspect:
        return 'LOW', 'COPY_SUSPECT: provenance review required'
    if m['n'] == 0:
        return 'LOW', 'NO_GOLD_COVERAGE'
    support = min(m['positives'], m['negatives'])
    if (m['coverage'] >= .8 and m['n'] >= 40 and support >= 10 and
        m['accuracy'] >= .9 and m['balanced_accuracy'] >= .85 and m['soft_auc'] >= .85 and
        m['sensitivity_ci95'][0] >= .65 and m['specificity_ci95'][0] >= .65):
        return 'HIGH', 'Meets all registered HIGH criteria (exposed Gold diagnostic)'
    if (m['coverage'] >= .5 and m['n'] >= 25 and support >= 5 and
        m['accuracy'] >= .8 and m['balanced_accuracy'] >= .7 and m['soft_auc'] >= .7):
        return 'MEDIUM', 'Meets MEDIUM but not all HIGH criteria'
    return 'LOW', 'Insufficient support, coverage or agreement under registered criteria'


def analyse(gold, sources, specs):
    """Pure aggregate map; synthetic arrays supported without file IO."""
    validate_values(gold, True)
    need(len(gold) > 0 and np.isfinite(gold).all(), 'Incomplete Gold')
    need(set(sources) == set(specs), 'Source/spec mismatch')
    data = {}
    for name, a in sources.items():
        need(a.shape == gold.shape, 'Reference alignment differs')
        validate_values(a, specs[name]['rule'] == 'binary')
        data[name] = (a, hard_labels(a, specs[name]['rule']), np.isfinite(a), np.isfinite(a))
    groups = {'consensus': list(e150.SETS),
              'consensus_gold3': ['steven_v4', 'pilkwang_v2', 'relabel_v1']}
    specs = {k: dict(v) for k, v in specs.items()}
    for name, members in groups.items():
        data[name] = consensus([sources[m] for m in members])
        specs[name] = dict(path=None, rule='all_members_ge0.5', ancestors=members,
                           evidence_group='shared_report_origin', members=members,
                           provenance='conditional unanimity; shared errors and selection bias')
    flags = {name: bool(np.isfinite(h).all() and np.array_equal(h, gold))
             for name, (_, h, _, _) in data.items()}
    rows = []
    for name, (soft, hard, available, mask) in data.items():
        for j, target in enumerate(T):
            m = cell_metrics(gold[:, j], soft[:, j], hard[:, j], mask[:, j])
            tier, reason = validity(m, flags[name])
            intersections = {}
            for group in groups:
                gm = data[group][3][:, j]
                q = cell_metrics(gold[:, j], soft[:, j], hard[:, j], mask[:, j] & gm)
                intersections[group] = dict(n=q['n'], coverage_gold=q['coverage'],
                    coverage_reference=ratio(q['n'], m['n']), accuracy=q['accuracy'])
            rows.append(dict(reference=name, target=target, **m,
                available_n=int(available[:, j].sum()),
                agreement_mask_coverage=ratio(int(mask[:, j].sum()), int(available[:, j].sum())),
                mask_sha256=digest(mask[:, j].tolist()), validity=tier, validity_reason=reason,
                copy_suspect=flags[name], consensus_intersections=intersections))
    recommendations = {}
    for target in T:
        eligible = [r for r in rows if r['target'] == target and r['validity'] != 'LOW']
        tier = 'HIGH' if any(r['validity'] == 'HIGH' for r in eligible) else 'MEDIUM' if eligible else 'LOW'
        recommendations[target] = dict(validity=tier,
            references=sorted(r['reference'] for r in eligible if r['validity'] == tier),
            status='DIAGNOSTIC_ONLY' if eligible else 'NEEDS_IMAGE_READ_PANEL')
    return dict(rows=rows, sources=specs, recommendations=recommendations,
                external_panels={
                    'oai_moaks': dict(status='PENDING', dependencies=['H-B-88', 'H-B-65'],
                        admitted_targets=[], exclude_ancestry=['OAI', 'OAIZIB'],
                        reason='certify MRI mappings/linkage/grouping/unknown masks after routing fix; X-ray separate'),
                    'kneemri_acl': dict(status='PENDING', dependencies=['H-B-65'],
                        admitted_targets=[], proposed_targets=['ACL'], exclude_ancestry=['KneeMRI'],
                        reason='certify grades, grouping, preprocessing and model ancestry')})


def verification(note, root=ROOT):
    """A separate reviewer supplies PASS + exact source/test/protocol hashes."""
    root, note = Path(root), Path(note)
    receipt = json.loads(note.read_text(encoding='utf-8'))
    need(receipt.get('verdict') == 'PASS' and receipt.get('reviewer') and
         receipt.get('checklist') and receipt.get('not_checked') is not None,
         'Independent verification PASS receipt required')
    actual = {p: sha(root / p) for p in BINDINGS}
    need(receipt.get('bindings') == actual, 'Verification binding changed; re-verify')
    return dict(path=str(note), sha256=sha(note), reviewer=receipt['reviewer'], bindings=actual)


def build_map(destination, verification_note, root=ROOT):
    root, destination = Path(root), Path(destination)
    receipt = verification(verification_note, root)
    need(not destination.exists(), 'Use a fresh map directory')
    started = now()
    paths = {'gold': GOLD, **{n: s['path'] for n, s in source_specs().items()}}
    hashes = {n: sha(root / p) if (root / p).is_file() else None for n, p in paths.items()}
    with threadpool_limits(limits=2):
        ids, gold = load_table(root / GOLD, binary=True)
        need(len(ids) == 58 and np.isfinite(gold).all(), 'Expected 58 complete official Gold rows')
        sources = {n: load_table(root / p, ids, binary=n == 'jev_hybrid')[1]
                   if (root / p).is_file() else np.full(gold.shape, np.nan)
                   for n, p in paths.items() if n != 'gold'}
        result = analyse(gold, sources, source_specs())
    need(hashes == {n: sha(root / p) if (root / p).is_file() else None for n, p in paths.items()},
         'Input bytes changed during analysis')
    result.update(schema=SCHEMA, tool_version=VERSION, plan='H-B-42',
        created_at=now(), started_at=started, verification=receipt,
        protocol=dict(path=PROTOCOL, sha256=sha(root / PROTOCOL)),
        cohort_sha256=digest(ids), inputs={n: dict(path=p, sha256=hashes[n],
            status='PRESENT' if hashes[n] else 'MISSING') for n, p in paths.items()},
        scope='Exposed Gold58 diagnostic; no private-score validity claim; no candidate results read',
        runtime=dict(cpu_only=True, max_threads=2, numpy=np.__version__, pandas=pd.__version__))
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / 'map.json', result)
    fields = [k for k in result['rows'][0] if k != 'consensus_intersections']
    with (destination / 'map.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in result['rows'])
    lines = ['# Per-target ruler map v' + VERSION, '', result['scope'], '',
             'Protocol: ' + PROTOCOL, 'Verification: ' + str(verification_note), '',
             '| Target | Tier | Recommended diagnostic rulers |', '|---|---|---|']
    for target, r in result['recommendations'].items():
        lines.append(f"| {target} | {r['validity']} | {', '.join(r['references']) or 'none; image-read evidence needed'} |")
    lines += ['', 'Full aggregate metrics, class support and intervals: map.csv / map.json.',
              'The exact four-source consensus and Gold3 are separate masks.',
              'No candidate verdict, GPU work or submission was performed.']
    (destination / 'map.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(destination / 'manifest.json', dict(schema=SCHEMA, tool_version=VERSION,
        verification=receipt, files={p: sha(destination / p) for p in ('map.json', 'map.csv', 'map.md')}))
    return result


def _number(value, label, low=-1., high=1.):
    need(isinstance(value, (int, float)) and not isinstance(value, bool) and
         math.isfinite(value) and low <= value <= high, 'Invalid ' + label)
    return float(value)


def composite_decision(ruler_map, deltas, original_verdict, *, ancestry,
                       retrospective=False, gold_evaluation=False):
    """Advisory decision; preserves registered verdict, never fills missing deltas.

    ancestry: {declared: true, training_references: [...], teacher_references: [...],
               gold_exposed: bool}. Lists are UNION of candidate and control.
    Each delta has reference, target, delta, optional ci95, and optionally paired
    baseline_auc/candidate_auc. Intervals must use the registered paired design.
    """
    need(ruler_map.get('schema') == SCHEMA, 'Wrong ruler map schema')
    need(isinstance(original_verdict, str) and original_verdict, 'Original verdict required')
    need(isinstance(ancestry, dict), 'Ancestry declaration required')
    known = set(ruler_map['sources'])
    declared = ancestry.get('declared') is True
    exposures = set()
    for field in ('training_references', 'teacher_references'):
        values = ancestry.get(field)
        need(isinstance(values, list) and all(isinstance(x, str) for x in values),
             'Ancestry lists required for BOTH candidate and control')
        need(set(values) <= known, 'Unknown ancestry reference')
        for name in values:
            exposures.update(ruler_map['sources'][name]['ancestors'])
            exposures.add(name)
    need(isinstance(ancestry.get('gold_exposed'), bool), 'Gold exposure declaration required')
    cells = {(r['target'], r['reference']): r for r in ruler_map['rows']}
    values, low_losses, flags = {}, [], []
    for row in deltas:
        key = (row['target'], row['reference'])
        need(key in cells and key not in values, 'Unknown or duplicate target/reference delta')
        value = _number(row['delta'], 'AUC delta')
        ci = row.get('ci95')
        if ci is not None:
            need(isinstance(ci, (list, tuple)) and len(ci) == 2, 'Invalid interval')
            ci = [_number(x, 'interval') for x in ci]
            need(ci[0] <= value <= ci[1], 'Interval does not contain delta')
        if 'baseline_auc' in row or 'candidate_auc' in row:
            b = _number(row.get('baseline_auc'), 'baseline AUC', 0, 1)
            c = _number(row.get('candidate_auc'), 'candidate AUC', 0, 1)
            need(abs(c-b-value) <= 1e-10, 'AUC delta arithmetic mismatch')
        values[key] = dict(delta=value, ci95=ci)
        if value < 0 and cells[key]['validity'] == 'LOW':
            low_losses.append(dict(target=key[0], reference=key[1], delta=value))
    per_target, fail_targets = {}, []
    decision_targets, report_only_targets = [], []
    for target in T:
        recommendation = ruler_map['recommendations'][target]
        report_only = recommendation['validity'] == 'LOW'
        (report_only_targets if report_only else decision_targets).append(target)
        refs = recommendation['references']
        used, omitted, losses = [], [], []
        for ref in refs:
            source = ruler_map['sources'][ref]
            key = target, ref
            if not declared or exposures.intersection(source['ancestors']) or ref in exposures:
                omitted.append(ref + ': ancestry exposed/undeclared')
                continue
            if key not in values:
                omitted.append(ref + ': missing delta')
                continue
            v = values[key]
            used.append(dict(reference=ref, **v))
            if v['ci95'] is None or v['ci95'][0] < -MARGIN:
                flags.append(target + '/' + ref + ': uncertainty includes material harm')
            if v['delta'] < -MARGIN:
                flags.append(target + '/' + ref + ': material point loss')
            if v['ci95'] is not None and v['ci95'][1] < -MARGIN:
                losses.append(ref)
        # Never count correlated report extractions or consensus/member as two votes.
        for i, a in enumerate(losses):
            for b in losses[i+1:]:
                sa, sb = ruler_map['sources'][a], ruler_map['sources'][b]
                if (sa['evidence_group'] != sb['evidence_group'] and
                    not set(sa['ancestors']).intersection(sb['ancestors'])):
                    fail_targets.append(target)
        complete = bool(refs) and not omitted and len(used) == len(refs)
        if not complete and not report_only:
            flags.append(target + ': incomplete recommended evidence')
        per_target[target] = dict(recommended=refs, used=used, omitted=omitted,
            complete=complete, report_only=report_only,
            delta=sum(x['delta'] for x in used)/len(used) if complete else None)
    macro = (sum(r['delta'] for r in per_target.values())/len(T)
             if all(r['complete'] for r in per_target.values()) else None)
    decision_macro = (sum(per_target[t]['delta'] for t in decision_targets)/len(decision_targets)
                      if decision_targets and all(per_target[t]['complete'] for t in decision_targets)
                      else None)
    if retrospective:
        flags.append('Retrospective: report-only')
    if gold_evaluation or ancestry['gold_exposed']:
        flags.append('Gold evaluation/exposure: report-only')
    scope_block = (retrospective or gold_evaluation or ancestry['gold_exposed'] or
                   not declared or decision_macro is None)
    if fail_targets and not scope_block:
        verdict = 'FAIL'
    elif (not flags and decision_macro is not None and decision_macro >= 0 and
          original_verdict.upper() in ('PASS', 'ADOPT', 'ADMIT', 'ADVANCE')):
        verdict = 'PASS'
    else:
        verdict = 'REVIEW'
    if original_verdict.upper() not in ('PASS', 'ADOPT', 'ADMIT', 'ADVANCE'):
        flags.append('Original registered verdict retained; no automatic promotion')
    return dict(verdict=verdict, original_verdict=original_verdict, macro12_delta=macro,
                 decision_macro_delta=decision_macro, decision_targets=decision_targets,
                 report_only_targets=report_only_targets, per_target=per_target,
                 low_validity_losses=low_losses,
                corroborated_harm_targets=sorted(set(fail_targets)), reasons=sorted(set(flags)),
                scope='advisory; not a replacement for the registered gate or private validation')


def noise_corrected_delta(delta, calibration, *, true_prevalence, assumptions,
                          reference_kind='single_binary'):
    """Class-conditional AUC attenuation correction; diagnostic, never a gate."""
    _number(delta, 'delta')
    pi = _number(true_prevalence, 'true screen prevalence', 0, 1)
    need(0 < pi < 1, 'True prevalence must be strictly between zero and one')
    need(set(assumptions) == set(NOISE_ASSUMPTIONS), 'Explicit noise assumptions required')
    need(reference_kind == 'single_binary', 'Consensus/soft-label correction is not supported')
    out = dict(status='WITHHELD', corrected_delta=None, assumptions=list(NOISE_ASSUMPTIONS),
               true_screen_prevalence=pi, scope='point sensitivity analysis; no uncertainty propagation')
    if min(calibration['positives'], calibration['negatives']) < 10:
        return dict(out, reason='Insufficient class support')
    se = _number(calibration['sensitivity'], 'sensitivity', 0, 1)
    sp = _number(calibration['specificity'], 'specificity', 0, 1)
    q = se*pi + (1-sp)*(1-pi)
    if se+sp <= 1 or not 0 < q < 1:
        return dict(out, reason='Uninformative/inverted or degenerate noise channel')
    k = pi*(1-pi)*(se+sp-1)/(q*(1-q))
    if k < .2:
        return dict(out, reason='Unstable attenuation below 0.2', attenuation=k)
    corrected = delta/k
    if abs(corrected) > 1:
        return dict(out, reason='Infeasible corrected AUC delta', attenuation=k)
    return dict(out, status='DIAGNOSTIC_ONLY', corrected_delta=corrected, attenuation=k)


def record_screen(screen_path, map_path, destination, verification_note, ledger_root=ledger.DIR):
    """Immutable companion receipt; reuse old ledger's reading lookup only."""
    receipt = verification(verification_note)
    destination = Path(destination)
    need(not destination.exists(), 'Screen receipt already exists (append-only)')
    map_path = Path(map_path)
    manifest = json.loads((map_path.parent / 'manifest.json').read_text(encoding='utf-8'))
    map_bytes = map_path.read_bytes()
    map_hash = hashlib.sha256(map_bytes).hexdigest()
    need(map_path.name == 'map.json' and manifest['files'].get('map.json') == map_hash,
         'Map manifest hash mismatch')
    ruler_map = json.loads(map_bytes)
    need(ruler_map['verification']['bindings'] == receipt['bindings'], 'Map/tool version differs')
    screen_bytes = Path(screen_path).read_bytes()
    screen_hash = hashlib.sha256(screen_bytes).hexdigest()
    screen = json.loads(screen_bytes)
    for field in ('key', 'eval_set', 'cohort_sha256', 'mask_sha256', 'aggregation',
                  'candidate', 'control', 'forecast', 'no_change_forecast'):
        need(field in screen and screen[field] is not None and screen[field] != '',
             'Missing screen identity/forecast: ' + field)
    for field in ('cohort_sha256', 'mask_sha256'):
        value = screen[field]
        need(isinstance(value, str) and len(value) == 64 and
             all(c in '0123456789abcdef' for c in value), 'Invalid ' + field)
    need(isinstance(screen.get('retrospective'), bool) and
         isinstance(screen.get('gold_evaluation'), bool), 'Explicit timing/Gold scope required')
    _number(screen['forecast'], 'forecast AUC delta')
    need(_number(screen['no_change_forecast'], 'no-change AUC delta') == 0,
         'No-change forecast must be zero delta')
    for row in screen['deltas']:
        need('candidate_auc' in row and 'baseline_auc' in row, 'Record paired ruler values, not deltas only')
    reads = ledger._read(Path(ledger_root) / 'readings.csv')
    after = screen['retrospective'] or any(r['key'] == screen['key'] for r in reads)
    result = composite_decision(ruler_map, screen['deltas'], screen['original_verdict'],
        ancestry=screen['ancestry'], retrospective=after, gold_evaluation=screen['gold_evaluation'])
    need(sha(screen_path) == screen_hash and sha(map_path) == map_hash,
         'Screen/map input changed during recording')
    output = dict(schema='evidence-ruler-screen-v1', tool_version=VERSION, recorded_at=now(),
        verification=receipt, map_path=str(map_path), map_sha256=map_hash,
        screen_sha256=screen_hash, retrospective=after, screen=screen, decision=result)
    write_json(destination, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build-map')
    build.add_argument('--out', type=Path, required=True)
    build.add_argument('--verification-note', type=Path, required=True)
    record = sub.add_parser('record-screen')
    record.add_argument('--screen', type=Path, required=True)
    record.add_argument('--map', type=Path, required=True)
    record.add_argument('--out', type=Path, required=True)
    record.add_argument('--verification-note', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'build-map':
        result = build_map(args.out, args.verification_note)
        print(json.dumps(dict(status='COMPLETE', tool_version=VERSION, rows=len(result['rows']),
                              targets=len(result['recommendations']), output=str(args.out))))
    else:
        result = record_screen(args.screen, args.map, args.out, args.verification_note)
        print(json.dumps(dict(status='RECORDED', decision=result['decision']['verdict'], output=str(args.out))))


if __name__ == '__main__':
    main()
