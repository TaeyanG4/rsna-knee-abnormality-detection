"""vNext-R7-PL 1.1.0 heldout producer, roster and training-only target adapters.

Changelog 2026-10-13: executable A1-A3; imports H-B-146 v1.1.1 mathematics.
Never prints target rows. Engineering artifacts cannot be used for production.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hb146_pl_labels_v111 as pl
from tools import hb145_r7_heldout_v1 as producer
from tools import hb145_r7_pipeline_v111 as pipeline
from tools import hb145_r7_monitor_v111 as monitor
from tools.hb146_teacher_v111 import require_roles
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

VERSION = '1.1.0'
need, read, sha = pl.need, pl.read, pl.sha


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def seal_teacher(source, out, expected_ids=None):
    """Authenticate the actual v1.1.2 dual-role journals; export only heldout raw.

    Producer receipts do not contain CSV hashes. The seal binds every producer
    record and code pin and independently compares each ordered CSV value to
    its journal. No inference, model import or probability text is emitted.
    """
    source, out = Path(source), Path(out)
    names = ['identity.json', 'run_receipt.json', 'receipt_dual.json']
    need(all((source / name).is_file() for name in names), 'HOLD: producer receipts incomplete')
    identity, run, worker = [read(source / n) for n in names]
    for entry in (identity, run, worker):
        pl.require_teacher_exposure(entry)
        require_roles(entry)
        need(entry.get('version') == '1.1.2', 'producer v1.1.2 required')
    uids = identity.get('uids', [])
    need(uids and len(uids) == len(set(uids)) and all(isinstance(u, str) and u for u in uids), 'producer roster invalid')
    if expected_ids is not None:
        need(uids == list(expected_ids), 'producer ordered roster differs')
    fingerprint = producer.fingerprint(identity)
    device = identity.get('auxiliary_device')
    need(device in ('cpu', 'cuda') and identity.get('device') == device and
         all(r.get('identity') == fingerprint and r.get('auxiliary_device') == device for r in (run, worker)), 'producer identity/device differs')
    need(identity.get('models') == run.get('models') == 'heldout,f0' and
         worker.get('models') == producer.r7.MODELS, 'producer model roster differs')
    need(run.get('completed') == run.get('selected') == len(uids) and
         run.get('failures') == 0 and run.get('exit_code') == 0, 'producer incomplete/failed')
    need(worker.get('status') == 'PASS' and worker.get('device') == device and
         type(worker.get('studies')) is int and 0 < worker['studies'] <= len(uids) and
         type(worker.get('scored')) is int and worker['studies'] <= worker['scored'] <= 2 * worker['studies'] and
         worker.get('fallback') == {}, 'producer worker incomplete/failed')
    paths = [source / n for n in names]
    code = {'runner_sha256': Path(producer.__file__), 'pipeline_sha256': Path(pipeline.__file__),
            'monitor_sha256': Path(monitor.__file__), 'runtime_sha256': source / 'stage/r7_runtime/compact_infer.py'}
    for key, path in code.items():
        need(path.is_file() and identity.get(key) == sha(path), 'producer code drift: ' + key)
        paths.append(path)
    bundle, code_identity, deps = [identity.get(k, {}) for k in ('bundle', 'code', 'deps')]
    need(bundle.get('manifest_sha256') == producer.MANIFEST_SHA and bundle.get('manifest_pinned') is True and
         set(producer.r7.MODELS) <= set(bundle.get('file_sha256', {})) and
         all(len(bundle['file_sha256'][m]) == 64 for m in producer.r7.MODELS) and
         code_identity.get('tree_sha256') and code_identity.get('r7_closure_sha256') and
         deps.get('tree_sha256') == producer.DEPS_SHA, 'producer dependency identity missing/differs')
    csv_path = source / 'heldout_raw.csv'
    columns = [pl.UID, 'fold_k', 'status', 'model_role', 'model_fold', 'output_kind',
               'pooled_prior_bias', 'stackers_cross_fitted', 'prediction_reference',
               'roles_coincide', 'auxiliary_device', *pl.TEACHER_EXPOSURE]
    rows = pl.pd.read_csv(csv_path, usecols=columns, dtype=str, keep_default_na=False)
    need(rows[pl.UID].tolist() == uids, 'heldout CSV ordered coverage differs')
    values = pl.table(csv_path)
    for row in rows.to_dict('records'):
        uid = row[pl.UID]
        path = producer.record_path(source, uid)
        rec = read(path)
        pl.require_teacher_exposure(rec)
        pl.require_teacher_exposure(row, csv=True)
        pipeline.validate_record(rec)
        k = pl.hash_fold(uid)
        need(rec.get(pl.UID) == uid and rec.get('identity') == fingerprint and
             rec.get('fold_k') == k and rec.get('models') == 'heldout,f0' and
             rec.get('auxiliary_device') == device and row['auxiliary_device'] == device and
             rec.get('status') in ('ok', 'dropped series') and row['status'] == rec['status'] and
             rec.get('pooled_prior_bias') is True and row['pooled_prior_bias'] == 'True' and
             row['model_role'] == 'heldout' and row['model_fold'] == row['fold_k'] == str(k) and
             row['output_kind'] == 'pre_stacker' and row['stackers_cross_fitted'] == 'not_applied' and
             row['roles_coincide'] == str(k == 0) and
             row['prediction_reference'] == f'{path.name}#predictions/{k}', 'heldout journal metadata differs')
        need(pl.np.array_equal(pl.np.asarray(rec['predictions'][str(k)]['raw'], dtype=pl.np.float32), values[uid]), 'heldout journal prediction differs')
        paths.append(path)
    paths += [csv_path, Path(__file__), Path(pl.__file__)]
    result = dict(version=VERSION, producer='H-B-145', producer_version='1.1.2',
        status='PASS', role='heldout', output_kind='pre_stacker',
        model_fold='int(sha1(uid)[:8],16)%4', complete=True, failures=0,
        clean_oof=False, exposure='EXPOSED_EXPLORATORY', producer_exposure='EXPOSED',
        producer_identity=fingerprint, auxiliary_device=device,
        stackers_cross_fitted='not_applied', csv_path=str(csv_path.resolve()), csv_sha256=sha(csv_path),
        rows=len(uids), ordered_ids=uids, ordered_ids_sha256=ids_hash(uids),
        teacher_limitations=pl.LIMITATIONS, role_provenance=identity['role_provenance'], pins=pl.pins(paths))
    pl.write(out, result)
    return result


def validate_teacher_seal(path):
    s = read(path)
    need(s.get('version') == VERSION and s.get('producer_version') == '1.1.2' and
         s.get('status') == 'PASS' and s.get('role') == 'heldout' and
         s.get('complete') is True and s.get('failures') == 0 and
         s.get('output_kind') == 'pre_stacker' and s.get('stackers_cross_fitted') == 'not_applied' and
         s.get('clean_oof') is False and s.get('exposure') == 'EXPOSED_EXPLORATORY' and
         s.get('producer_exposure') == 'EXPOSED', 'heldout teacher seal required')
    pl.check_pins(s['pins'])
    need(sha(s['csv_path']) == s['csv_sha256'] and ids_hash(s['ordered_ids']) == s['ordered_ids_sha256'], 'teacher seal hashes differ')
    return s


def build_cohort(folds, gold_ids, teacher_ids, author_ids=None):
    """Registered fivefold components, plus screen cross-hash exclusions.

    Author cardinality is never silently equated with the local roster. When an
    author roster is unavailable its ancestry uncertainty remains explicit;
    student membership is completely defined by the pinned local roster.
    """
    rows = pl.pd.read_csv(folds, usecols=[pl.UID, 'component_id', 'fold'], dtype=str) if isinstance(folds, (str, Path)) else folds.copy()
    need(set(rows) == {pl.UID, 'component_id', 'fold'} and rows.notna().all().all() and rows[pl.UID].is_unique, 'invalid fold roster')
    need(rows['fold'].isin([str(i) for i in range(5)]).all(), 'invalid registered fold')
    need(rows.groupby('component_id')['fold'].nunique().max() == 1, 'component crosses validation folds')
    gold = set(gold_ids)
    own = rows[pl.UID].tolist()
    need(len(list(teacher_ids)) == len(set(teacher_ids)), 'duplicate teacher IDs')
    need(set(own) - gold <= set(teacher_ids), 'teacher missing local cohort')
    screen = pl.split(rows, gold)
    excluded = set(screen['excluded_cross_hash'])
    eligible = [u for u in own if u not in gold and u not in excluded]
    need(eligible, 'empty eligible cohort')
    reconciliation = dict(local_count=len(own), documented_author_count=4344,
        policy='student cohort is pinned local IDs; author-only IDs never inserted',
        author_membership='unresolved; EXPOSED ancestry retained')
    if author_ids is not None:
        author_ids = list(author_ids)
        need(len(author_ids) == len(set(author_ids)), 'duplicate author IDs')
        reconciliation.update(author_membership='UID set difference recorded', author_count=len(author_ids),
            author_only=sorted(set(author_ids) - set(own)), local_only=sorted(set(own) - set(author_ids)))
    result = dict(version=VERSION, cohort=own, eligible_ids=eligible, gold_ids=sorted(gold),
        excluded_cross_hash=sorted(excluded), components=screen['components'],
        folds=dict(zip(rows[pl.UID], rows['fold'].astype(int))), reconciliation=reconciliation,
        pins=pl.pins([folds]) if isinstance(folds, (str, Path)) else {})
    result['cohort_sha256'] = digest(result)
    return result


def selected_rosters(cohort, mode, fold):
    need(cohort.get('cohort_sha256') == digest({k:v for k,v in cohort.items() if k != 'cohort_sha256'}), 'cohort seal differs')
    pl.check_pins(cohort['pins'])
    ids = cohort['eligible_ids']
    if mode == 'all-data':
        need(fold == 'all', 'all-data fold must be all')
        train, val = list(ids), []
    else:
        need(mode == 'fivefold' and type(fold) is int and fold in range(5), 'fivefold fold invalid')
        train = [u for u in ids if cohort['folds'][u] != fold]
        val = [u for u in ids if cohort['folds'][u] == fold]
        need(val, 'empty validation fold')
    need(train and not set(train) & set(val) and not (set(train) | set(val)) & set(cohort['gold_ids']), 'Gold/split leakage')
    need(not {cohort['components'][u] for u in train} & {cohort['components'][u] for u in val}, 'component leakage')
    return train, val


def materialize_targets(out, cohort, l2, teacher, teacher_seal, arm, mode, fold, seed,
                        registration_sha256, frozen_mask=None, *, engineering=False):
    """Import H-B-146's exact scale, mask, arm and L2-weight implementations."""
    need(arm in ('global', 'fill') and seed in (42, 43), 'arm/seed differs')
    need(isinstance(registration_sha256, str) and len(registration_sha256) == 64, 'registration hash required')
    train, val = selected_rosters(cohort, mode, fold)
    inputs = [Path(__file__), Path(pl.__file__)]
    if not engineering:
        need(isinstance(l2, (str, Path)) and isinstance(teacher, (str, Path)), 'production target assets must be pinned files')
        base = read(pl.BASE)
        l2_key = base['config']['labels']['path']
        need(sha(l2) == base['inputs']['data'][l2_key], 'L2 asset differs from frozen H-B-146 base')
        seal = validate_teacher_seal(teacher_seal)
        need(sha(teacher) == seal['csv_sha256'], 'teacher CSV differs')
        inputs += [Path(pl.BASE), Path(l2), Path(teacher), Path(teacher_seal)]
    else:
        seal = None
    l2_values = pl.table(l2) if isinstance(l2, (str, Path)) else l2
    teacher_values = pl.table(teacher) if isinstance(teacher, (str, Path)) else teacher
    # Never align validation or Gold labels: every numeric operation uses train.
    scaled, scaling = pl.scaled_teacher(l2_values, teacher_values, train, train)
    weights = pl.frozen_weights(l2_values, train)
    if frozen_mask is None:
        mask, mask_pins = pl.silence_mask(train)
    else:
        need(engineering, 'production mask must use frozen H-B-146 source')
        mask, mask_pins = pl.np.asarray(frozen_mask), {}
    need(mask.dtype == bool and mask.shape == (len(train), 12), 'frozen mask differs')
    targets = pl.arm_targets(pl.aligned(l2_values, train), scaled, mask)[arm]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target_path = out / 'targets.npz'
    pl.np.savez(target_path, ids=pl.np.asarray(train), targets=targets, mask=mask, weights=weights)
    pl.write(out / 'scale.json', scaling)
    pl.write(out / 'cohort.json', cohort)
    spec = dict(version=VERSION, schema='pl-materialized-targets-v1', arm=arm, mode=mode, fold=fold,
        seed=seed, engineering_only=engineering, clean_oof=False, exposure='EXPOSED_EXPLORATORY',
        train_ids=train, val_ids=val, targets=pl.TARGETS, weights=weights.tolist(), pos_weight=weights.tolist(),
        target_path=str(target_path.resolve()), target_sha256=sha(target_path),
        train_ids_sha256=ids_hash(train), val_ids_sha256=ids_hash(val),
        l2_pos_weight_sha256=array_hash(weights), silence_mask_sha256=array_hash(mask),
        weight_source='L2_new_train_roster_only', scale_fit_ids_sha256=scaling['train_uid_sha256'],
        teacher_sha256=sha(teacher_seal) if teacher_seal else None,
        teacher_seal_path=str(Path(teacher_seal).resolve()) if teacher_seal else None,
        registration_sha256=registration_sha256, cohort_sha256=cohort['cohort_sha256'],
        pins={**pl.pins(inputs + [out / 'scale.json', out / 'cohort.json']), **cohort['pins'], **mask_pins})
    spec['spec_sha256'] = digest(spec)
    pl.write(out / 'target-spec.json', spec)
    load_materialized(out)
    return spec


def load_materialized(bundle):
    path = Path(bundle)
    path = path / 'target-spec.json' if path.is_dir() else path
    spec = read(path)
    need(spec.get('version') == VERSION and spec.get('spec_sha256') == digest({k:v for k,v in spec.items() if k != 'spec_sha256'}), 'target spec changed')
    pl.check_pins(spec['pins'])
    if not spec.get('engineering_only'):
        need(spec.get('teacher_seal_path') and sha(spec['teacher_seal_path']) == spec['teacher_sha256'], 'teacher seal changed')
        validate_teacher_seal(spec['teacher_seal_path'])
    need(sha(spec['target_path']) == spec['target_sha256'], 'targets changed')
    with pl.np.load(spec['target_path'], allow_pickle=False) as data:
        ids, values, mask, weights = data['ids'].tolist(), data['targets'], data['mask'], data['weights']
    need(ids == spec['train_ids'] and ids_hash(ids) == spec['train_ids_sha256'] == spec['scale_fit_ids_sha256'] and
         len(ids) == len(set(ids)) and not set(ids) & set(spec['val_ids']), 'target roster differs/leaks')
    need(values.dtype == pl.np.float32 and values.shape == (len(ids), 12) and pl.np.isfinite(values).all() and
         ((values >= 0) & (values <= 1)).all(), 'invalid materialized targets')
    need(mask.dtype == bool and mask.shape == values.shape and array_hash(mask) == spec['silence_mask_sha256'] and
         weights.dtype == pl.np.float32 and weights.shape == (12,) and weights.tolist() == spec['weights'] == spec['pos_weight'] and
         array_hash(weights) == spec['l2_pos_weight_sha256'], 'mask/weight drift')
    return spec, dict(zip(ids, values))


validate_targets = load_materialized


def prepare_targets(stage, member_id, teacher_seal, out, *, folds=pl.FOLDS,
                    gold_registration=pl.WORK/'registration.json', l2=None, author_ids=None):
    """Single CPU production entry point; no bespoke roster/target script needed."""
    from tools import vnext_r7_pl_v1 as production
    reg = production.registration(stage)
    member = next((m for m in reg['members'] if m['id'] == member_id), None)
    need(member is not None and member['kind'] in ('MIL', 'CoAt'), 'unknown PL member')
    teacher = validate_teacher_seal(teacher_seal)
    # This source contains only registered UID lists and numeric metadata; no
    # report or clinical label table is needed to establish Gold membership.
    gold = read(gold_registration)
    need(gold.get('version') == '1.1.1' and len(gold.get('gold', [])) == 58, 'Gold58 registration differs')
    cohort = build_cohort(folds, gold['gold'], teacher['ordered_ids'], author_ids)
    need(len(cohort['cohort']) == 4343 and len(cohort['excluded_cross_hash']) == 176,
         'registered 4343 cohort/cross-hash exclusion differs')
    cohort['pins'].update(pl.pins([gold_registration]))
    cohort['cohort_sha256'] = digest({k:v for k,v in cohort.items() if k != 'cohort_sha256'})
    l2 = l2 or ROOT / pl.read(pl.BASE)['config']['labels']['path']
    return materialize_targets(out, cohort, l2, teacher['csv_path'], teacher_seal, reg['arm'],
        'all-data' if member['kind'] == 'CoAt' else reg['mode'], member['fold'], member['seed'], production.digest(reg))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    seal = sub.add_parser('seal-teacher')
    seal.add_argument('--source', type=Path, required=True)
    seal.add_argument('--out', type=Path, required=True)
    validate = sub.add_parser('validate-targets')
    validate.add_argument('--bundle', type=Path, required=True)
    prepare = sub.add_parser('prepare-targets')
    prepare.add_argument('--stage', type=Path, required=True)
    prepare.add_argument('--member', required=True)
    prepare.add_argument('--teacher-seal', type=Path, required=True)
    prepare.add_argument('--out', type=Path, required=True)
    prepare.add_argument('--folds', type=Path, default=pl.FOLDS)
    prepare.add_argument('--gold-registration', type=Path, default=pl.WORK/'registration.json')
    prepare.add_argument('--l2', type=Path)
    prepare.add_argument('--author-ids', type=Path, help='optional UID-only JSON author membership list')
    a = p.parse_args(argv)
    if a.command == 'seal-teacher':
        s = seal_teacher(a.source, a.out)
        result = dict(status=s['status'], rows=s['rows'], role=s['role'])
    elif a.command == 'prepare-targets':
        s = prepare_targets(a.stage, a.member, a.teacher_seal, a.out, folds=a.folds,
            gold_registration=a.gold_registration, l2=a.l2, author_ids=read(a.author_ids) if a.author_ids else None)
        result = dict(status='PASS', rows=len(s['train_ids']), engineering_only=False)
    else:
        s, values = load_materialized(a.bundle)
        result = dict(status='PASS', rows=len(values), engineering_only=s['engineering_only'])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
