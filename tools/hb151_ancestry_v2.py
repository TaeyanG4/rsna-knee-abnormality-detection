"""H-B-151 v2.0.0: CPU-only ancestry gate; no fit or launch capability.

Changelog: supersedes v1's missing-row gate with the authorized four-row
L2 fallback; registers paired local decisions and rejects exposed ancestry.
Existing recipes, scored stages and v1 audit remain untouched.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.local/scratch/astra-b4/H-B-151/ancestry-v2.json'
AUTH = 'artifacts/experiments/HB105_SERVING_V1/input-manifest.json'
AUTH_SHA = '47598e384892f11809a25ca05d5e119fd080ca0041ccf8e39f606a3e9561c84b'
RECIPE = 'artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json'
RECIPE_SHA = 'b2aaca64289249595e532b74d361175ed78b1170109eee302ec29068e0e7af11'
ALIGNED = 'artifacts/experiments/HB105_ALIGNED_OOF_V1/input-source-manifest.json'
ALIGNED_SHA = 'f439d836e46eeac920146cfdf3d1b180a10075a61b3c7fb2b1d3e5de5eb35806'
R7 = '.local/scratch/astra-b5/H-B-145/full-v113-ready/'
L2 = 'storage/labels/derived_constellation/l2_dread75_aquila25_v1'
WEAK = (3, 5, 6, 8)
REGISTRATION = dict(seeds=[42, 43], rulers=['JEV', 'aquila', 'consensus', 'Dread'],
    mean_macro_gain_min=.002, both_seed_macro_strictly_positive=True,
    weak_four_min=0., per_target_floor=-.003,
    aggregation='mean paired AUC deltas over seeds and four rulers; '
                'weak-four and target floors additionally enforced for every seed/ruler',
    old_public_score_thresholds='superseded; never used',
    no_fit_before='ancestry PASS + independent review + measured CUDA smoke gate')


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def uid_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def verify_file(root, rel, expected=None):
    path = root / rel
    need(path.resolve().is_relative_to(root.resolve()), 'outside project')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        need(digest == (expected['sha256'] if isinstance(expected, dict) else expected),
             'hash drift: '+rel)
    return path, digest


def ancestry_eligible(*, overlap, complete, calibrated, ranks=False):
    return overlap == 0 and complete is True and calibrated is True and ranks is False


def decision(deltas, *, ancestry_pass=False):
    """Fixed [seed42,seed43] x [four named rulers] x [12 targets] AUC deltas."""
    need(len(deltas) == 2 and all(len(seed) == 4 for seed in deltas), 'seed/ruler shape')
    need(all(len(row) == 12 for seed in deltas for row in seed), 'target shape')
    need(all(math.isfinite(x) for seed in deltas for row in seed for x in row), 'nonfinite')
    seed_macro = [sum(sum(row) for row in seed) / 48 for seed in deltas]
    checks = dict(ancestry=ancestry_pass is True,
        macro=sum(seed_macro)/2 >= .002, seeds=all(x > 0 for x in seed_macro),
        weak=all(sum(row[i] for i in WEAK)/4 >= 0 for seed in deltas for row in seed),
        target=all(x >= -.003 for seed in deltas for row in seed for x in row))
    return dict(go=all(checks.values()), checks=checks)


def write_json(path, data):
    """Bounded Windows-safe replacement, persistent report failure is explicit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    for attempt in range(40):
        try:
            tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(.05*(attempt+1), 1.))


def audit(root=ROOT):
    pins = {}
    def bound(rel, expected=None):
        path, digest = verify_file(root, rel, expected)
        pins[rel] = digest
        return path
    def read(rel, expected=None):
        return json.loads(bound(rel, expected).read_text(encoding='utf-8-sig'))
    auth = read(AUTH, AUTH_SHA)
    recipe = read(RECIPE, RECIPE_SHA)
    aligned = read(ALIGNED, ALIGNED_SHA)
    rosters = []
    for f in range(5):
        rel = f'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/fold{f}/ordered-ids.json'
        rosters.append(read(rel, auth['bindings'][rel]))
    train = set(recipe['train_ids']); gold = set(recipe['excluded_gold_ids'])
    outer = set(rosters[0]['val'])
    universe = set().union(*(set(r['val']) for r in rosters))
    need(len(outer) == 888 and outer <= train and not train & gold, 'outer cohort changed')
    need(len(universe) == 4343 and universe <= train and len(train-universe) == 4,
         'four-row L2 fallback changed')
    members = []
    for entry in aligned['members']:
        e = entry['member']
        if e['family'] not in ('R50', 'C4', 'S'):
            continue
        roster = rosters[e['fold']]
        need(e['train_uid_sha256'] in (uid_hash(roster['train']), uid_hash(sorted(roster['train'])))
             and e['heldout_uid_sha256'] == uid_hash(roster['val']), 'member roster binding')
        prov_path = str(Path(e['path']).parent / 'provenance.json').replace('\\', '/')
        prov = read(prov_path, auth['bindings'][prov_path])
        need(prov['train_ids_ordered_sha256'] == e['train_uid_sha256']
             and prov['eval_ids_ordered_sha256'] == e['heldout_uid_sha256'], 'provenance binding')
        path = bound(entry['prediction'], entry['prediction_pin'])
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = csv.DictReader(stream)
            fields = rows.fieldnames
            ids = []
            for row in rows:
                ids.append(row['StudyInstanceUID'])
                vals = [float(row[k]) for k in fields if k != 'StudyInstanceUID']
                need(len(vals) == 12 and all(math.isfinite(x) and 0 <= x <= 1 for x in vals),
                     'invalid member probability domain')
        need(ids == roster['val'], 'member prediction roster')
        members.append(dict(id=e['id'], fold=e['fold'], prediction=entry['prediction'],
            prediction_rows=len(ids), outer_validation_seen_in_training=len(outer & set(roster['train'])),
            checkpoint_sha256=e['sha256'], checkpoint_rehashed=False))
    need(len(members) == 25, 'Q7S coverage')
    leaking = [m for m in members if m['fold'] != 0]
    need(len(leaking) == 20 and all(m['outer_validation_seen_in_training'] == 888 for m in leaking),
         'unexpected outer exposure; re-audit')
    r7 = read(R7+'identity.json')
    read(R7+'run_receipt.json')
    l2 = read(L2+'.provenance.json', recipe['label_provenance_sha256'])
    bound(L2+'.csv', recipe['label_sha256'])
    selection = read('artifacts/experiments/EXP149_LABEL_SOURCES/screen/result.json')
    need(selection['folds'] == [0, 1], 'L2 selection fold evidence changed')
    for path in ['.local/scratch/slot-b/type-review/T2-out.md',
                 '.local/scratch/slot-b/type-review/T5-out.md',
                 '.local/scratch/astra-b4/H-B-151/input-audit-v1.json',
                 'tools/hb105_aligned_oof_v1.py', 'tools/hb151_ancestry_v2.py']:
        bound(path)
    return dict(version='2.0.0', status='STOP_ANCESTRY', exposure='EXPOSED', clean_oof=False,
        audit_scope='Known direct exposure is sufficient to STOP; unresolved ancestors are not certified clean. '
            'Checkpoint digests are authenticated manifest assertions, not fresh checkpoint-byte verification.',
        teacher_mix=None, pilot_admitted=False, registration=REGISTRATION,
        split=dict(source='EXP099 fold0 restricted to exact EXP143v2 non-Gold population',
            rationale='fixed fold0 before outcomes; EXP143v2 itself is all-data and has no validation fold',
            validation_count=len(outer), training_count=len(train-outer), gold_excluded=len(gold),
            validation_uid_sha256=uid_hash(sorted(outer)),
            training_uid_sha256=uid_hash(sorted(train-outer)),
            missing_q7s_training_rows=4, missing_row_policy='L2-only, explicitly authorized v2',
            component_isolation='not promoted to admitted split; direct UID leakage already fails'),
        teachers=dict(Q7S=dict(eligible=False, members=members,
            training_target_generators_leaking=len(leaking),
            reason='Ordinary row-OOF is not outer-clean: 20 generators see all 888 validation studies.',
            outer0_only_alternative='Five leaf fold0 models exclude validation at direct fit, but supplied saved '
                'predictions cover validation only; complete initialization/selection/calibration ancestry '
                'and outer0-only training predictions are not supplied by the authenticated OOF manifest.',
            probability_semantics='Member values finite in [0,1] does not prove calibration. Existing combined '
                'Q7S readout is pooled percentile ranks; forbidden as BCE targets. No calibrated outer-clean mix proven.'),
          R7=dict(eligible=False, pooled_target_priors=r7['pooled_target_priors'],
            pooled_state_priors=r7['pooled_state_priors'], anchor_ancestry=r7['anchor_ancestry'],
            side_ancestry=r7['side_ancestry'],
            reason='Heldout role does not exclude pooled-prior/selection/anchor ancestry; unknown means STOP.',
            stackers='heldout_raw skips post-stackers; removing stackers does not remove pooled priors'),
          L2=dict(eligible_as_outer_clean_teacher=False, formula=l2['formula'], selection_folds=selection['folds'],
            reason='Recipe label mix selected by EXP149 on folds0+1, including chosen validation fold0; '
                'rowwise formula has no fitted stacker but complete cross-study label-generation ancestry '
                'and probability calibration are unproved. Valid inherited control supervision, not proof of outer cleanliness.')),
        future_if_go='Only after ancestry PASS and paired registered GO: all-data refit, then strict '
            'C85R7_COATPL384 from v1.1.0 replacing only EXP143v2 checkpoint; preserve '
            '0.85r(P)+0.15r((17CoAtPL+17EXP155+12Q_A5q)/46), serving startup and metadata pins; '
            'full embedded-file/notebook diff and independent review. No stage built.',
        smoke_command=None, pilot_fit_commands=[], expected_gpu_hours=0,
        expected_gpu_hours_reason='STOP: no admitted pilot; historical timing is not a smoke measurement',
        measured_peak_vram=None, measured_epoch_seconds=None, inputs_sha256=pins)


if __name__ == '__main__':
    result = audit()
    write_json(OUT, result)
    print(json.dumps({k: result[k] for k in ('status', 'teacher_mix', 'pilot_admitted')}))
