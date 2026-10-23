"""v1.0.0: local provisional expert MRI pilot preparation, never annotation.

2026-10-19: Reuse authenticated H105 two-family outputs and numeric K1 L2
targets. No torch, models, image/report reads, network or training. All actual
identities stay in private scratch. Clinical release remains HOLD.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from tools.hb105_serving_v101 import uid_hash, validate_rosters

VERSION = '1.0.0'
SEED = 20261019
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
OUT = ROOT / '.local/scratch/slot-b/expert-mri-pilot96-v1'
SPEC = 'docs/research/expert-mri-pilot96-preparation-20261019.md'
FROZEN = 'artifacts/experiments/HB105_ALIGNED_OOF_V1/input-source-manifest.json'
AUTH = 'artifacts/experiments/HB105_SERVING_V1/input-manifest.json'
GROUPS = 'artifacts/experiments/VALIDATION_GROUP_CONTRACT_V1/components.json'
ROSTER = '.local/scratch/astra-b4/H-B-146/launch-v115/registration.json'
K1 = '.local/scratch/l1-rco/k1v3-grouping/registration.json'
LABELS = 'storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv'
PINS = {
    FROZEN: 'f439d836e46eeac920146cfdf3d1b180a10075a61b3c7fb2b1d3e5de5eb35806',
    AUTH: '47598e384892f11809a25ca05d5e119fd080ca0041ccf8e39f606a3e9561c84b',
    GROUPS: 'ede3ffcac6102ff575d9b3c0d4ffe9ea354a54496238d63bee6836b901a8cc52',
    ROSTER: '3e98b8d279e18e393f606d15b7e272ebef5a8e8a0620694afc23115e04e7c14d',
    K1: '604bc360c0f7e413557803e701d7c69a11455f103acdd816d9cf3df68d005bd6',
    LABELS: '79ce307e246cc54a9a90f32fd1ab21d58ebd23b4b7e7b1a80207adcea1bcc2c9',
}
LIMITS = [
    'No qualified MSK readers secured; zero actual readings or adjudicated labels.',
    'Gold58 directly excluded; cross-Gold patient linkage unknown (Gold absent from group contract).',
    'Complete prior human-image exposure history and patient linkage not certified.',
    'OOF row/group exclusion verified only for stored disease fits; nested teacher/encoder ancestry unresolved.',
    'Audit/test reserve images previously exposed; not a clean final-system evaluation.',
    'Viewer/access and all12 target-manual signoffs pending; no reading-task release.',
    'Component-uniform random sampling; enriched cases do not estimate population prevalence.',
    'Rare weak-label coverage only; protocol/scanner coverage not measured.',
]


def need(ok, code):
    if not ok:
        raise ValueError(code)  # Codes only: never include a row or identifier.


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def key(namespace, value):
    return hashlib.sha256(f'{SEED}|{namespace}|{value}'.encode()).hexdigest()


def checked(rel, expected, bindings):
    p = ROOT / rel
    need(p.resolve().is_relative_to(ROOT), 'INPUT_OUTSIDE_PROJECT')
    need('.local/relabel' not in p.as_posix(), 'PROHIBITED_INPUT')
    need(p.suffix in ('.json', '.csv'), 'INPUT_TYPE')
    h = sha(p)
    expected = expected['sha256'] if isinstance(expected, dict) else expected
    need(h == expected, 'INPUT_HASH_MISMATCH')
    bindings[rel] = h
    return p


def read_bound(rel, expected, bindings):
    return json.loads(checked(rel, expected, bindings).read_text(encoding='utf-8'))


def numeric(path, ids, ordered=True):
    # usecols is intentional: report and unrelated fields never enter memory.
    df = pd.read_csv(path, usecols=[UID, *TARGETS], dtype={UID: str}, float_precision='round_trip')
    need(not df[UID].duplicated().any(), 'DUPLICATE_NUMERIC_ID')
    if ordered:
        need(df[UID].tolist() == ids, 'NUMERIC_ORDER')
    need(set(ids) <= set(df[UID]), 'NUMERIC_COVERAGE')
    a = df.set_index(UID).loc[ids, TARGETS].to_numpy(dtype=np.float32)
    need(a.shape == (len(ids), 12) and np.isfinite(a).all(), 'NUMERIC_SHAPE_FINITE')
    need(((a >= 0) & (a <= 1)).all(), 'NUMERIC_RANGE')
    return a


def authenticate():
    bindings = {}
    f = read_bound(FROZEN, PINS[FROZEN], bindings)
    auth = read_bound(AUTH, PINS[AUTH], bindings)
    comps = read_bound(GROUPS, PINS[GROUPS], bindings)
    g = {}
    for comp in comps:
        for u in comp['members']:
            need(u not in g, 'DUPLICATE_GROUP_MEMBER')
            g[u] = comp['component_id']
    r = read_bound(ROSTER, PINS[ROSTER], bindings)
    k1 = read_bound(K1, PINS[K1], bindings)
    need(k1['train'] == sorted(r['train']) and k1['holdout'] == sorted(r['val']), 'K1_ROSTER_DRIFT')
    need(k1['pins'][LABELS] == PINS[LABELS], 'K1_LABEL_IDENTITY')
    need(all(u in g for u in r['train'] + r['val']), 'GROUP_COVERAGE')
    need(not set(r['gold']) & set(r['train'] + r['val']), 'DIRECT_GOLD_OVERLAP')
    need(not {g[u] for u in r['train']} & {g[u] for u in r['val']}, 'K1_GROUP_OVERLAP')
    rosters = []
    for fold in range(5):
        rel = f'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold{fold}/ordered-ids.json'
        rr = read_bound(rel, f['bindings'][rel], bindings)
        need(not {g[u] for u in rr['train']} & {g[u] for u in rr['val']}, 'OOF_GROUP_OVERLAP')
        rosters.append(rr)
    validate_rosters(rosters)
    ids = sum((rr['val'] for rr in rosters), [])
    families = {}
    for family in ('R50', 'S'):
        entries = [e for e in f['members'] if e['member']['family'] == family]
        need({(e['member']['seed'], e['member']['fold']) for e in entries} ==
             {(seed, fold) for seed in (42, 43) for fold in range(5)} and len(entries) == 10, 'MEMBER_SET')
        per_fold = []
        for fold, rr in enumerate(rosters):
            values = []
            for seed in (42, 43):
                e = next(e for e in entries if (e['member']['seed'], e['member']['fold']) == (seed, fold))
                m = e['member']
                need(m == next(x for x in auth['members'] if x['id'] == m['id']), 'AUTH_MEMBER_DRIFT')
                base = Path(m['path']).parent.as_posix()
                inv = read_bound(base+'/output-inventory.json', f['bindings'][base+'/output-inventory.json'], bindings)
                prov = read_bound(base+'/provenance.json', inv['provenance.json'], bindings)
                need(inv['model_best.pt']['sha256'] == m['sha256'], 'CHECKPOINT_INVENTORY_BINDING')
                need(m['heldout_uid_sha256'] == prov['eval_ids_ordered_sha256'] == uid_hash(rr['val']), 'HELDOUT_BINDING')
                need(m['train_uid_sha256'] == prov['train_ids_ordered_sha256'] and
                     m['train_uid_sha256'] in (uid_hash(rr['train']), uid_hash(sorted(rr['train']))), 'TRAIN_BINDING')
                need((m['train_count'], m['heldout_count']) == (len(rr['train']), len(rr['val'])), 'MEMBER_COUNTS')
                values.append(numeric(checked(e['prediction'], e['prediction_pin'], bindings), rr['val']))
            per_fold.append(np.stack(values).mean(axis=0, dtype=np.float32))
        families[family] = np.concatenate(per_fold)
    labels = numeric(checked(LABELS, PINS[LABELS], bindings), ids, ordered=False)
    return ids, families, labels, r, g, bindings


def random_groups(ids, groups, n, namespace, used):
    by = {}
    for u in ids:
        if groups[u] not in used:
            by.setdefault(groups[u], []).append(u)
    need(len(by) >= n, 'INSUFFICIENT_COMPONENTS')
    chosen = []
    for group in sorted(by, key=lambda x: key(namespace, x))[:n]:
        chosen.append(min(by[group], key=lambda x: key(namespace+'/representative', x)))
        used.add(group)
    return chosen


def percentiles(values):
    need(len(values) > 0 and np.isfinite(values).all(), 'RANK_INPUT')
    return pd.DataFrame(values).rank(method='average', axis=0).to_numpy() / len(values)


def round_robin(queues, groups, used, n):
    # Queue entries: (UID, private reason). Pop until one unused group per queue.
    positions = [0] * len(queues)
    out = []
    while len(out) < n:
        progress = False
        for j, queue in enumerate(queues):
            while positions[j] < len(queue):
                u, reason = queue[positions[j]]
                positions[j] += 1
                if groups[u] not in used:
                    used.add(groups[u]); out.append((u, reason)); progress = True
                    break
            if len(out) == n:
                break
        if not progress:
            break
    return out


def suspicion_queues(ids, a, b, labels, kind):
    queues = []
    targets = list(range(12))
    if kind == 'rare':
        targets.sort(key=lambda j: (int((labels[:, j] >= .8).sum()), j))
    for j in targets:
        if kind == 'contradiction':
            configs = [('high', (a[:, j] >= .8) & (b[:, j] >= .8) & (labels[:, j] <= .2), np.minimum(a[:, j], b[:, j])),
                       ('low', (a[:, j] <= .2) & (b[:, j] <= .2) & (labels[:, j] >= .8), 1-np.maximum(a[:, j], b[:, j]))]
        elif kind == 'disagreement':
            configs = [('spread', np.ones(len(ids), bool), np.abs(a[:, j]-b[:, j]))]
        else:
            need(kind == 'rare', 'QUEUE_KIND')
            configs = [('weak_positive', labels[:, j] >= .8, np.zeros(len(ids)))]
        for sign, mask, severity in configs:
            ix = sorted(np.flatnonzero(mask), key=lambda i: (-float(severity[i]), key(kind, ids[i])))
            queues.append([(ids[i], f'{kind}:{TARGETS[j]}:{sign}') for i in ix])
    return queues


def select(ids, families, labels, r, g):
    used = set()
    test = random_groups(r['val'], g, 200, 'future_test', used)
    audit = random_groups(r['val'], g, 32, 'audit', used)
    calibration = random_groups(r['train'], g, 16, 'calibration', used)
    pool = sorted(u for u in r['train'] if g[u] not in used)
    index = {u: i for i, u in enumerate(ids)}
    ix = [index[u] for u in pool]
    a, b = (percentiles(families[name][ix]) for name in ('R50', 'S'))
    y = labels[ix]
    train = [(u, 'random') for u in random_groups(pool, g, 18, 'train_random', used)]
    fallback = {}
    for kind, n in [('contradiction', 15), ('disagreement', 9), ('rare', 6)]:
        selected = round_robin(suspicion_queues(pool, a, b, y, kind), g, used, n)
        fallback[kind] = n-len(selected)
        if len(selected) < n:
            selected += [(u, f'random_fallback:{kind}') for u in random_groups(pool, g, n-len(selected), 'fallback/'+kind, used)]
        train.extend(selected)
    entries = [(u, 'calibration', 'random') for u in calibration] + [(u, 'audit_dev', 'random') for u in audit]
    entries += [(u, 'training', reason) for u, reason in train]
    all_ids = [u for u, _, _ in entries] + test
    need(len(entries) == 96 and len(all_ids) == len(set(all_ids)) == len({g[u] for u in all_ids}) == 296, 'SELECTION_OVERLAP')
    need(not set(all_ids) & set(r['gold']), 'SELECTION_GOLD')
    return entries, test, fallback, pool


FORM_FIELDS = ['study_token', 'target', 'reader_id', 'manual_version', 'finding_state',
               'severity_code', 'official_mapping', 'manual_clause', 'evidence_location',
               'confidence', 'limitation_code', 'full_study_reviewed', 'start_time', 'end_time', 'lock_hash']


def blank_forms(tokens):
    return [{**dict.fromkeys(FORM_FIELDS, ''), 'study_token': token, 'target': target}
            for token in tokens for target in TARGETS]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def prepare():
    need(not OUT.exists(), 'OUTPUT_EXISTS_NO_OVERWRITE')
    ids, families, labels, r, g, bindings = authenticate()
    entries, test, fallback, pool = select(ids, families, labels, r, g)
    token_for = {u: 'MRI-'+key('reader_token', u)[:16] for u, _, _ in entries}
    need(len(set(token_for.values())) == 96, 'TOKEN_COLLISION')
    private = [dict(study_token=token_for[u], study_uid=u, component=g[u], role=role, reason=reason)
               for u, role, reason in entries]
    write_json(OUT/'coordinator/roster.private.json', private)
    write_json(OUT/'coordinator/future-test-reserve.private.json',
               dict(status='EXPOSED_CONDITIONAL_RESERVE_NOT_CLEAN_TEST', ids=test))
    counts = {}
    for reader in ('A', 'B', 'adjudication'):
        for phase in ('calibration', 'independent'):
            tokens = [token_for[u] for u, role, _ in entries if (role == 'calibration') == (phase == 'calibration')]
            tokens.sort(key=lambda t: key('order/'+reader+'/'+phase, t))
            rows = blank_forms(tokens)
            rel = f'readers/{reader}-{phase}.blank.csv'
            write_csv(OUT/rel, rows, FORM_FIELDS)
            counts[rel] = len(rows)
    # Re-read emitted templates, not a second selection or model analysis.
    for rel, count in counts.items():
        with (OUT/rel).open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream); rows = list(reader)
            need(reader.fieldnames == FORM_FIELDS and len(rows) == count, 'FORM_SCHEMA_COUNT')
        need(all(row[k] == '' for row in rows for k in FORM_FIELDS[2:]), 'FORM_NONBLANK_FINDING')
        need(all(row['study_token'] in token_for.values() for row in rows), 'FORM_TOKEN')
    summary = dict(version=VERSION, status='PROVISIONAL96_PREPARED_READER_HOLD',
        actual_readings=0, adjudicated_cells=0, trained_models=0, eligible_for_reading_release=False,
        counts=dict(pilot=96, calibration=16, training=48, audit_dev=32, conditional_future_test=200,
                    distinct_components=296, training_percentile_pool=len(pool),
                    oof_universe=len(ids), oof_families=2, oof_members=20),
        training_allocation=dict(random=18, contradiction=15, disagreement=9, rare=6),
        fallback=fallback, reader_templates=counts, study_target_cells_per_reader=1152,
        direct_gold_overlap=0, known_component_overlap=0, cross_gold_patient_linkage='UNKNOWN',
        sampling='one representative per component; component-uniform random parts',
        limits=LIMITS, bindings=bindings,
        sources={p: sha(ROOT/p) for p in [SPEC, 'tools/expert_mri_pilot_v1.py', 'tools/hb105_serving_v101.py']},
        private_roster_sha256=sha(OUT/'coordinator/roster.private.json'),
        checks='B_DIRECT_ONLY_NOT_INDEPENDENT', label_recipe='L2 75%Dread+25%Aquila',
        medical_truth='UNAVAILABLE; selection contradictions are suspicion, not verified errors')
    write_json(OUT/'PREPARATION-RESULT.json', summary)
    return {k: summary[k] for k in ('status', 'counts', 'fallback', 'actual_readings', 'adjudicated_cells')}


if __name__ == '__main__':
    try:
        print(json.dumps(prepare()))
    except Exception as exc:
        # Never echo exception contents: pandas/KeyError may contain medical IDs.
        print(json.dumps(dict(status='PREPARATION_FAILED', error_type=type(exc).__name__,
                              detail='Suppressed to protect identifiers; no reading or training performed.')))
        sys.exit(1)
