"""Production targets v1.0.0, 2026-10-14.

Changelog: A2/A3 extension for all registered labelled fivefold studies and
authenticated v113 heldout targets; exact registered missing teachers use L2.
All scale, mask, alpha and pos_weight mathematics delegate to A3 unchanged.
"""
from pathlib import Path
from tools import vnext_r7_pl_targets_v110 as a3
from tools import vnext_r7_pl_v1 as p
from tools import pl_production_teacher_v1 as teacher_adapter
from tools import hb145_data_exclusions_v1 as exclusions

VERSION = '1.0.0'
pl = a3.pl


def build_cohort(folds, gold_ids, teacher_ids, excluded):
    rows = pl.pd.read_csv(folds, usecols=[pl.UID, 'component_id', 'fold'], dtype=str) if isinstance(folds, (str, Path)) else folds.copy()
    pl.need(set(rows) == {pl.UID, 'component_id', 'fold'} and rows.notna().all().all()
            and rows[pl.UID].is_unique, 'invalid fold roster')
    pl.need(rows['fold'].isin([str(i) for i in range(5)]).all()
            and rows.groupby('component_id')['fold'].nunique().max() == 1, 'component crosses folds')
    own = rows[pl.UID].tolist()
    gold = set(gold_ids)
    # Final production uses every labelled study; Gold remains descriptive only.
    # A3's screen-time exclusion must not silently shrink the production cohort.
    eligible = list(own)
    pl.need(len(teacher_ids) == len(set(teacher_ids)), 'duplicate teacher roster')
    pl.need(set(excluded) <= set(exclusions.REGISTERED_UIDS), 'unregistered teacher exclusions')
    pl.need(set(eligible) - set(teacher_ids) == set(excluded) & set(eligible), 'unregistered missing teacher')
    pl.need(not set(teacher_ids) & set(excluded), 'excluded study has teacher')
    value = dict(version=VERSION, cohort=own, eligible_ids=eligible, gold_ids=[],
                 descriptive_gold_ids=sorted(gold),
                 excluded_cross_hash=[], data_exclusions=sorted(excluded),
                 components=dict(zip(rows[pl.UID], rows['component_id'])),
                 folds=dict(zip(rows[pl.UID], rows['fold'].astype(int))),
                 pins=pl.pins([folds]) if isinstance(folds, (str, Path)) else {})
    value['cohort_sha256'] = a3.digest(value)
    return value


def materialize(out, cohort, l2, teacher, seal_path, arm, fold, registration_sha256,
                frozen_mask=None, engineering=False):
    pl.need(arm in ('global', 'fill'), 'winning arm required')
    train, val = a3.selected_rosters(cohort, 'fivefold', fold)
    inputs = [Path(__file__), Path(a3.__file__), Path(pl.__file__)]
    if not engineering:
        seal = teacher_adapter.validate(seal_path)
        base = pl.read(pl.BASE)
        pl.need(pl.sha(l2) == base['inputs']['data'][base['config']['labels']['path']], 'L2 pin differs')
        pl.need(pl.sha(teacher) == seal['csv_sha256'], 'teacher pin differs')
        inputs += [pl.BASE, Path(l2), Path(teacher), Path(seal_path)]
    l2 = pl.table(l2) if isinstance(l2, (str, Path)) else l2
    teacher = pl.table(teacher) if isinstance(teacher, (str, Path)) else teacher
    missing = [u for u in train if u not in teacher]
    pl.need(set(missing) == set(train) & set(cohort['data_exclusions']), 'unregistered missing training teacher')
    available = [u for u in train if u in teacher]
    scaled, scale = pl.scaled_teacher(l2, teacher, available, available)
    base = pl.aligned(l2, train)
    weights = pl.frozen_weights(l2, train)
    if frozen_mask is None:
        mask, mask_pins = pl.silence_mask(train)
    else:
        pl.need(engineering, 'synthetic mask forbidden')
        mask, mask_pins = pl.np.asarray(frozen_mask), {}
    pl.need(mask.dtype == bool and mask.shape == base.shape, 'silence mask differs')
    # Apply A3 only to available teacher rows. Missing rows remain byte-exact L2.
    index = [i for i, u in enumerate(train) if u in teacher]
    values = base.copy()
    values[index] = pl.arm_targets(base[index], scaled, mask[index])[arm]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = out/'targets.npz'
    import io
    buffer = io.BytesIO()
    pl.np.savez(buffer, ids=pl.np.asarray(train), targets=values, mask=mask, weights=weights)
    p.metadata.write_once(payload, buffer.getvalue())
    p.dump(out/'scale.json', scale, immutable=True)
    p.dump(out/'cohort.json', cohort, immutable=True)
    spec = dict(version=VERSION, schema='pl-production-targets/v1', arm=arm, mode='fivefold', fold=fold,
                seed=42, engineering_only=engineering, clean_oof=False, exposure='EXPOSED_EXPLORATORY',
                train_ids=train, val_ids=val, targets=pl.TARGETS, weights=weights.tolist(), pos_weight=weights.tolist(),
                target_path=str(payload.resolve()), target_sha256=pl.sha(payload),
                train_ids_sha256=a3.ids_hash(train), val_ids_sha256=a3.ids_hash(val),
                scale_fit_ids=available, scale_fit_ids_sha256=a3.ids_hash(available),
                l2_only_ids=missing, l2_pos_weight_sha256=a3.array_hash(weights), silence_mask_sha256=a3.array_hash(mask),
                teacher_sha256=pl.sha(seal_path) if seal_path else None,
                teacher_seal_path=str(Path(seal_path).resolve()) if seal_path else None,
                registration_sha256=registration_sha256, cohort_sha256=cohort['cohort_sha256'],
                pins={**pl.pins(inputs+[out/'scale.json', out/'cohort.json']), **cohort['pins'], **mask_pins})
    spec['spec_sha256'] = a3.digest(spec)
    p.dump(out/'target-spec.json', spec, immutable=True)
    load_materialized(out)
    return spec


def load_materialized(bundle):
    path = Path(bundle)
    spec = p.read(path/'target-spec.json' if path.is_dir() else path)
    pl.need(spec.get('version') == VERSION and spec.get('spec_sha256') == a3.digest({k:v for k,v in spec.items() if k != 'spec_sha256'}), 'target seal changed')
    pl.check_pins(spec['pins'])
    if not spec['engineering_only']:
        pl.need(pl.sha(spec['teacher_seal_path']) == spec['teacher_sha256'], 'teacher seal changed')
        teacher_adapter.validate(spec['teacher_seal_path'])
    pl.need(pl.sha(spec['target_path']) == spec['target_sha256'], 'target payload changed')
    with pl.np.load(spec['target_path'], allow_pickle=False) as data:
        ids, values, mask, weights = data['ids'].tolist(), data['targets'], data['mask'], data['weights']
    pl.need(ids == spec['train_ids'] and len(ids) == len(set(ids)) and not set(ids)&set(spec['val_ids'])
            and a3.ids_hash(ids) == spec['train_ids_sha256'], 'target membership differs')
    pl.need(spec['scale_fit_ids'] == [u for u in ids if u not in spec['l2_only_ids']]
            and a3.ids_hash(spec['scale_fit_ids']) == spec['scale_fit_ids_sha256']
            and set(spec['l2_only_ids']) <= set(exclusions.REGISTERED_UIDS), 'scale/exclusion roster differs')
    pl.need(values.dtype == pl.np.float32 and values.shape == (len(ids),12) and pl.np.isfinite(values).all()
            and ((values >= 0)&(values <= 1)).all(), 'invalid targets')
    pl.need(mask.dtype == bool and mask.shape == values.shape and a3.array_hash(mask) == spec['silence_mask_sha256']
            and weights.dtype == pl.np.float32 and weights.shape == (12,)
            and weights.tolist() == spec['weights'] == spec['pos_weight']
            and a3.array_hash(weights) == spec['l2_pos_weight_sha256'], 'mask/weight drift')
    return spec, dict(zip(ids, values))


def prepare(stage, member, teacher_seal, out):
    reg = p.registration(stage)
    entry = next((m for m in reg['members'] if m['id'] == member), None)
    pl.need(entry and entry['kind'] == 'MIL' and entry['seed'] == 42, 'seed42 MIL member required')
    seal = teacher_adapter.validate(teacher_seal)
    gold_path = p.ROOT/'.local/scratch/astra-b4/H-B-146/launch-v115/registration.json'
    gold = p.read(gold_path)
    pl.need(len(gold['gold']) == 58, 'Gold58 registration differs')
    excluded = (seal.get('data_exclusions') or {}).get('uids', [])
    cohort = build_cohort(pl.FOLDS, gold['gold'], seal['ordered_ids'], excluded)
    cohort['pins'].update(pl.pins([gold_path]))
    cohort['cohort_sha256'] = a3.digest({k:v for k,v in cohort.items() if k != 'cohort_sha256'})
    l2 = p.ROOT/p.read(pl.BASE)['config']['labels']['path']
    return materialize(out, cohort, l2, seal['csv_path'], teacher_seal, reg['arm'], entry['fold'], p.digest(reg))
