"""H216 v1.0.0: authenticated CPU metadata/native-task preparation, no pixels/models.

Private identities and labels are written only below OUT; stdout is aggregate.
Original sources, previews, L2 and model files are never written.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

from tools import extlabels
from training.arms import oai_exclusions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.local/scratch/slot-b/h216-oai-inputs-v1'
AUDIT = ROOT / 'artifacts/experiments/HB88_OAI_HEADER_AUDIT_V2/run-v1'
AUDIT_SHA = '423c8c4f6fcc87a1cd441535a9e31c5d05719f9192da18a80000be25a4ab6320'
OLD = ROOT / '.local/scratch/astra-oai-extension/r6-oai/registration.json'
PREVIEW = ROOT / '.local/scratch/slot-b/oai-k1-preview8-v1'
CLINICAL = ROOT / 'storage/labels/external/OAI_receipt.json'
DOCROOT = ROOT / 'storage/external_data/oai/clinical/extracted/mr_assess/Semi-Quant Scoring_ASCII'
BASELINE = DOCROOT / 'kMRI_SQ_MOAKS_BICL00.txt'
COMMON = ROOT / '.local/exp189-cpu-v120/input/cloud_code/training/common.py'
REGIONS = tuple(r for rs in extlabels.CART_REGIONS.values() for r in rs)
TASKS = tuple('MCM' + r + '_' + kind for r in REGIONS for kind in ('area', 'full')) + ('MEFFWK', 'MSYIC')
SOURCE_FIELDS = tuple('MCM' + r for r in REGIONS) + ('MEFFWK', 'MSYIC')
EXPECTED = {
    'SAG_IW_TSE': ('Sagittal', 1, 'IW_TSE_FS', 'FAT_SAT'),
    'COR_IW_TSE': ('Coronal', 0, 'IW_TSE_NON_FS', 'NONE'),
    'COR_MPR': ('Coronal', 1, 'DESS_WE', 'WATER_EXCITATION'),
    'AX_MPR': ('Axial', -1, 'DESS_WE', 'WATER_EXCITATION'),
}


def need(ok, code):
    if not ok:
        raise ValueError(code)  # Fixed codes: never include private row values.


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def check_bindings(bindings):
    for path, digest in bindings.items():
        need(sha(path) == digest, 'SOURCE_DRIFT')


def slots():
    tree = ast.parse(COMMON.read_text(encoding='utf-8'))
    matches = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'SLOTS' for t in n.targets)]
    need(len(matches) == 1, 'SLOT_SOURCE')
    return matches[0]


def fs_flag(method):
    return {'FAT_SAT': 1, 'WATER_EXCITATION': 1, 'NONE': 0}.get(method, -1)


def bridge(group, slotmap):
    wanted = EXPECTED.get(group['source_sequence'])
    actual = (group['plane'], int(group['proposed_fluid']), group['contrast_family'],
              group['fat_suppression_method'])
    need(wanted == actual, 'CONTRAST_ROUTE_CONFLICT')
    need(group['contrast_verified'] is True and not group['contrast_issues']
         and group['contrast_certainty'] == 'HEADER_SUPPORTED_SOURCE_PROTOCOL', 'UNVERIFIED_CONTRAST')
    fs = fs_flag(actual[3])
    need(fs != -1, 'UNREGISTERED_FS_METHOD')
    return dict(StudyInstanceUID=group['study_key'], SeriesInstanceUID=group['series_key'],
                Anatomical_Plane=actual[0], Fluid_Sensitive=actual[1], Fat_Suppression=fs,
                fat_suppression_method=actual[3], contrast_family=actual[2],
                source_sequence=group['source_sequence'], group_key=group['group_key'],
                slot=slotmap[(actual[0], fs)], frame_count=int(group['frame_count']))


def native_values(scores):
    values = []
    for region in REGIONS:
        pair = extlabels.cartilage_parts(scores.get('MCM' + region))
        values.extend(pair if pair is not None else (None, None))
    for name in ('MEFFWK', 'MSYIC'):
        v = extlabels.grade(scores.get(name), range(4))
        values.append(None if v is None else int(v))
    need(len(values) == 30, 'TASK_COUNT')
    return values


def exclusion_union(panel, confirmation, preview):
    return set().union(*(set(oai_exclusions.participant(uid) for uid in source)
                         for source in (panel, confirmation, preview)))


def choose(records):
    """Reuse source precedence, but exclude unresolved same-rank task disagreement."""
    by_study = {}
    for record in records:
        need(record.project in extlabels.MOAKS_ORDER, 'UNREGISTERED_READING_PROJECT')
        by_study.setdefault(record.study_key, []).append(record)
    chosen, conflicts = {}, set()
    for uid, group in by_study.items():
        ranks = [extlabels.reading_rank(r, extlabels.MOAKS_ORDER)[:-1] for r in group]
        best = min(ranks)
        tied = [r for r, k in zip(group, ranks) if k == best]
        if len({tuple(native_values(r.scores)) for r in tied}) > 1:
            conflicts.add(uid)
        else:
            chosen[uid] = tied[0]
    return chosen, conflicts


def clinical_records():
    # Parse only the allowlisted fields; narrative/report columns never enter the dataframe.
    import pandas as pd
    wanted = {'ID', 'SIDE', 'READPRJ', 'VERSION'} | {'V00' + f for f in SOURCE_FIELDS}
    with BASELINE.open(encoding='latin-1') as f:
        header = next(csv.reader(f, delimiter='|'))
    clean = {x.strip().upper(): x for x in header}
    need(len(clean) == len(header) and wanted <= set(clean), 'CLINICAL_SCHEMA')
    df = pd.read_csv(BASELINE, sep='|', encoding='latin-1', dtype=str,
                     usecols=[clean[k] for k in sorted(wanted)], keep_default_na=False)
    df.columns = [c.strip().upper() for c in df.columns]
    records = []
    for row in df.to_dict('records'):
        scores = {field: extlabels.code(row['V00' + field]) for field in SOURCE_FIELDS}
        records.append(extlabels.Reading(extlabels.oai_key(row['ID'], row['SIDE'], 'V00'),
                       scores, scores, row['READPRJ'].strip(), row['VERSION'].strip(), BASELINE.name))
    return records


def register():
    need(not OUT.exists(), 'OUTPUT_EXISTS')
    preview_reg = read(PREVIEW / 'REGISTRATION.json')
    need(sha(OLD) == preview_reg['sources'][str(OLD)], 'CONFIRMATION_REGISTRATION_DRIFT')
    need(sha(PREVIEW / 'selection.private.json') == preview_reg['selection_sha256'], 'PREVIEW_SELECTION_DRIFT')
    clinical = read(CLINICAL)
    need(clinical['contract'] == 'EXP136-v1' and clinical['tool_sha256'] == sha(ROOT/'tools/extlabels.py'),
         'CLINICAL_RECEIPT_DRIFT')
    bound = [(p, h) for p, h in clinical['inputs_sha256'].items() if Path(p).name == BASELINE.name]
    need(len(bound) == 1 and sha(BASELINE) == bound[0][1], 'BASELINE_SOURCE_DRIFT')
    need(sha(AUDIT/'summary.json') == AUDIT_SHA, 'AUDIT_SUMMARY_DRIFT')
    policy = oai_exclusions.bind()
    paths = [Path(__file__), ROOT/'tools/extlabels.py', ROOT/'training/arms/oai_exclusions.py',
             ROOT/'tools/extcache/oai_cache_v2.py', ROOT/'tools/extcache/oai_audit_v2.py',
             ROOT/'docs/research/h216-oai-input-contract-v1.md', ROOT/'tests/test_h216_oai_inputs_v1.py',
             COMMON, CLINICAL, BASELINE, OLD, PREVIEW/'REGISTRATION.json',
             PREVIEW/'selection.private.json', ROOT/policy['path']]
    paths += [AUDIT/n for n in ('summary.json', 'routing.jsonl', 'archives.jsonl', 'slot-selections.jsonl')]
    paths += [DOCROOT/n for n in ('kMRI_SQ_MOAKS_BICL_Descrip.pdf', 'kMRI_SQ_MOAKS_BICL00_Contents.pdf',
                                 'kMRI_SQ_MOAKS_BICL00_Stats.pdf')]
    OUT.mkdir(parents=True)
    write(OUT/'REGISTRATION.json', dict(version='1.0.0', sources={str(p): sha(p) for p in paths},
        tasks=TASKS, class_values=[0,1,2,3], scope='CPU_METADATA_NATIVE_TASKS_ONLY',
        full_v4_ready=False, training_ready=False, pixels=False, models=False))


def run():
    started = time.perf_counter()
    reg = read(OUT/'REGISTRATION.json')
    check_bindings(reg['sources'])
    need(not (OUT/'ATTEMPT.json').exists(), 'ATTEMPT_EXISTS')
    write(OUT/'ATTEMPT.json', dict(version='1.0.0', started_unix=time.time()))
    from tools.extcache import oai_cache_v2
    audit = oai_cache_v2.load_audit(AUDIT)
    need(audit.summary_sha256 == AUDIT_SHA, 'AUDIT_DRIFT')
    policy = oai_exclusions.bind()
    panel = oai_exclusions.verify(policy)
    confirmation = read(OLD)['binding']['ids']
    preview = read(PREVIEW/'selection.private.json')['oai']
    excluded = exclusion_union(panel, confirmation, preview)
    smap = slots()
    routed, eligible_groups = {}, 0
    with (AUDIT/'routing.jsonl').open(encoding='utf-8') as f:
        for line in f:
            group = json.loads(line)
            g = audit.groups[group['group_key']]
            if not g['eligible']:
                continue
            row = bridge(group, smap)
            need(row['Anatomical_Plane'] == g['plane'] and row['Fluid_Sensitive'] == g['fluid'], 'AUDIT_BRIDGE_DRIFT')
            routed.setdefault(row['StudyInstanceUID'], []).append(row)
            eligible_groups += 1
    chosen, conflicts = choose(clinical_records())
    roster, rejected = [], Counter()
    for uid in sorted(routed):
        if oai_exclusions.participant(uid) in excluded:
            rejected['excluded_participant'] += 1
            continue
        rows = routed[uid]
        if len({r['SeriesInstanceUID'] for r in rows}) != len(rows):
            rejected['ambiguous_split_series'] += 1
            continue
        if uid in conflicts:
            rejected['conflicting_equal_rank_reads'] += 1
            continue
        if uid not in chosen:
            rejected['no_primary_baseline_reading'] += 1
            continue
        values = native_values(chosen[uid].scores)
        if all(v is None for v in values):
            rejected['all_tasks_masked'] += 1
            continue
        roster.append(dict(study_key=uid, values=values, mask=[int(v is not None) for v in values],
                           reading_project=chosen[uid].project, series=rows))
    need(len(roster) >= 8, 'INSUFFICIENT_ELIGIBLE_STUDIES')
    need(not any(oai_exclusions.participant(r['study_key']) in excluded for r in roster), 'PARTICIPANT_LEAKAGE')
    ids = [r['study_key'] for r in roster]
    pilot = sorted(ids, key=lambda uid: hashlib.sha256(('H216-NATIVE-PILOT-v1:'+uid).encode()).hexdigest())[:8]
    write(OUT/'cohort.private.json', dict(tasks=TASKS, studies=roster, excluded_participants=sorted(excluded)))
    write(OUT/'pilot8.private.json', dict(study_ids=pilot, scope='future_full_v4_pilot_not_executed'))
    rows = [s for r in roster for s in r['series']]
    with (OUT/'train_series.private.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    counts = {task: dict(observed=sum(r['mask'][j] for r in roster),
              missing=sum(1-r['mask'][j] for r in roster),
              classes={str(v):sum(r['values'][j] == v for r in roster) for v in range(4)})
              for j, task in enumerate(TASKS)}
    result = dict(status='CPU_INPUT_TASK_PREPARED', version='1.0.0', studies=len(roster),
        participants=len({oai_exclusions.participant(uid) for uid in ids}), series=len(rows),
        audit_eligible_groups=eligible_groups, audit_routed_studies=len(routed),
        excluded_participants=len(excluded), original_panel_participants=len(panel),
        confirmation_participants=len({oai_exclusions.participant(u) for u in confirmation}),
        rejected_studies=dict(rejected), task_counts=counts,
        zero_training_participant_overlap=True, slots=dict(Counter(str(r['slot']) for r in rows)),
        methods=dict(Counter(r['fat_suppression_method'] for r in rows)),
        fluid_unknown_series=sum(r['Fluid_Sensitive'] == -1 for r in rows),
        native_tasks=30, full_v4_ready=False, training_ready=False, pilot_selected=8,
        pixels_read=0, model_calls=0, seconds=time.perf_counter()-started,
        outputs={n:sha(OUT/n) for n in ('cohort.private.json','pilot8.private.json','train_series.private.csv')})
    check_bindings(reg['sources'])
    write(OUT/'RESULT.json', result)
    return result


def verify():
    reg, result = read(OUT/'REGISTRATION.json'), read(OUT/'RESULT.json')
    check_bindings(reg['sources'])
    check_bindings({str(OUT/n):h for n,h in result['outputs'].items()})
    cohort = read(OUT/'cohort.private.json')
    rows, excluded = cohort['studies'], set(cohort['excluded_participants'])
    need(cohort['tasks'] == list(TASKS), 'TASK_ORDER')
    need(len({r['study_key'] for r in rows}) == len(rows) == result['studies'], 'UNIQUE_STUDIES')
    for row in rows:
        need(oai_exclusions.participant(row['study_key']) not in excluded, 'PARTICIPANT_LEAKAGE')
        need(len(row['values']) == len(row['mask']) == 30, 'TASK_SHAPE')
        need(row['mask'] == [int(v is not None) for v in row['values']], 'MISSING_MASK')
        need(all(v is None or type(v) is int and v in range(4) for v in row['values']), 'TASK_RANGE')
    return dict(status='B_DIRECT_CPU_INPUT_CHECK', studies=len(rows), native_tasks=30,
                independent_review=False, full_v4_ready=False, training_ready=False)


if __name__ == '__main__':
    try:
        if sys.argv[1:] == ['register']:
            register(); print('REGISTERED_CPU_ONLY')
        elif sys.argv[1:] == ['run']:
            r=run(); print(json.dumps({k:v for k,v in r.items() if k not in ('task_counts','outputs')}))
        elif sys.argv[1:] == ['verify']:
            r=verify(); write(OUT/'B-RESULT-CHECK.json',r); print(json.dumps(r))
        else:
            raise ValueError('ACTION')
    except Exception as exc:
        # Do not print dataframe rows/labels/private paths from third-party exceptions.
        print(json.dumps(dict(status='FAILED', error_type=type(exc).__name__,
              code=str(exc) if type(exc) is ValueError and str(exc).replace('_','').isalnum() else 'PRIVATE_DETAILS_SUPPRESSED')))
        sys.exit(1)
