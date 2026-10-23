"""Portable MIL registration v1.1.0 (2026-10-11).

Changelog: production recipe/evidence integrity and no-control registration.

Real recorded controls/recipes, scientific mismatch and create-only fixtures.
No labels, CUDA execution, downloads or new training qualification.
"""
import copy
from pathlib import Path

import pytest

from training.arms import lifecycle as life, mil, mil_registration as reg
from tools import mil_register

ROOT = Path(__file__).resolve().parents[1]


def spec_for(seed=42, fold=0):
    return life.ArmSpec((ROOT/f'training/configs/mil-s{seed}-f{fold}-v3.json').read_bytes())


def migration_args(spec):
    controls = spec.document['inputs']['controls']
    protocol = next(p for p in controls if '/protocols/' in p)
    control = next(p.removesuffix('/summary.json') for p in controls if p.endswith('/summary.json'))
    return dict(protocol=protocol, protocol_sha256=controls[protocol], control=control,
                control_hashes={n: controls[control+'/'+n] for n in ('summary.json', 'provenance.json')},
                arm=spec.document['arm'], attempt=spec.document['attempt'])


@pytest.mark.parametrize('member', [r'.local\exp185-overlay\protocols\S42.json',
                                  '.local/exp185-overlay/protocols/S42.json'])
def test_portable_legacy_member(member):
    assert reg.portable_member(member) == '.local/exp185-overlay/protocols/S42.json'


@pytest.mark.parametrize('member', [r'..\file', r'a\..\file', r'C:\data\file', r'\\server\file',
    '/file', 'a//b', r'a\.\b', 'a:b', '', None, 'a/../b'])
def test_legacy_unsafe_paths_rejected(member):
    with pytest.raises(life.ContractError):
        reg.portable_member(member)


@pytest.mark.parametrize('seed,fold', [(42, f) for f in range(5)] + [(43, 0)])
def test_real_registered_controls_migrate_without_mutation(seed, fold):
    spec = spec_for(seed, fold)
    args = migration_args(spec)
    before = {p: life.file_sha256(ROOT/p) for p in spec.document['inputs']['controls']}
    args['protocol'] = args['protocol'].replace('/', '\\')
    args['control'] = args['control'].replace('/', '\\')
    actual = reg.migrate(ROOT, **args)
    assert actual.sha256 == spec.sha256
    assert all('\\' not in p for p in actual.inventory)
    assert before == {p: life.file_sha256(ROOT/p) for p in before}


@pytest.mark.parametrize('field', ['protocol_sha256', 'summary.json', 'provenance.json'])
def test_wrong_trusted_digest_rejected(field):
    args = migration_args(spec_for())
    if field == 'protocol_sha256':
        args[field] = '0'*64
    else:
        args['control_hashes'][field] = '0'*64
    with pytest.raises(life.ContractError):
        reg.migrate(ROOT, **args)


@pytest.mark.parametrize('seed,fold', [(43, 0), (42, 1), (43, 1)])
def test_mismatched_member_control_rejected(seed, fold):
    doc = spec_for().document
    doc.update(seed=seed, fold=fold)
    if fold:
        doc['config']['roots']['order_policy'] = 'sorted-uid'
    with pytest.raises(life.ContractError, match='control scientific'):
        mil.validate_data(life.ArmSpec.from_document(doc), ROOT, ROOT/doc['config']['roots']['corpus'])


@pytest.mark.parametrize('record,key,value', [
    ('args', 'seed', 43), ('args', 'fold', 1), ('args', 'k', 24), ('args', 'bs', 4),
    ('args', 'bb_lr', .001), ('args', 'amp', 'bf16'), ('args', 'variant', 'CONTROL'),
    ('provenance', 'train_ids_ordered_sha256', '0'*64),
    ('provenance', 'eval_ids_ordered_sha256', '0'*64),
    ('provenance', 'label_semantic_sha256', '0'*64),
    ('provenance', 'label_file_sha256', '0'*64),
    ('provenance', 'initial_state_sha256', '0'*64),
    ('provenance', 'initial_state_matches_registered', False),
    ('provenance', 'scheduler_total_steps', 123),
    ('provenance', 'k_eval_extra', 24), ('provenance', 'flip_tta', 'none'),
    ('summary', 'selected_epoch_1based', 19), ('summary', 'optimizer_updates', 1),
    ('summary', 'skipped_updates', 100), ('summary', 'train', 1),
    ('crop', 'detail_slice', [40, 280]), ('augmentation', 'sagittal_flip', True),
    ('runtime', 'effective_amp', 'bf16'), ('runtime', 'precision_fallback_allowed', True),
])
def test_scientific_control_changes_rejected(record, key, value):
    spec = spec_for()
    summary, provenance = reg.control_records(spec, ROOT)
    records = dict(args=summary['args'], summary=summary, provenance=provenance,
        crop=provenance['exp179'], augmentation=provenance['exp127_variant_config'], runtime=provenance['runtime'])
    records[record][key] = value
    with pytest.raises(life.ContractError):
        reg.validate_control(spec, summary, provenance)


def test_positive_weights_and_precision_backend_changes_rejected():
    spec = spec_for()
    for key in ('actual_pos_weight', 'computed_pos_weight', 'backends'):
        summary, provenance = reg.control_records(spec, ROOT)
        if key == 'backends':
            provenance['runtime'][key]['cudnn_benchmark'] = True
        else:
            provenance[key][0] += .01
        with pytest.raises(life.ContractError):
            reg.validate_control(spec, summary, provenance)


def test_incidental_control_metadata_and_large_elapsed_are_not_gates():
    spec = spec_for()
    summary, provenance = reg.control_records(spec, ROOT)
    summary.update(total_seconds=10**12, tag='relocated')
    summary['args'].update(workers=1, output_root='elsewhere', tag='different', wall_seconds=0)
    provenance.update(requested_workers=1, output_root='elsewhere', total_seconds=10**12)
    provenance['exp127_variant_config'].update(source='new prose', caveat='different prose')
    for record in (summary, provenance):
        record['exp179'].update(verification_note='relocated', arm='different-name')
    reg.validate_control(spec, summary, provenance)


@pytest.mark.parametrize('change', ['windows', 'metadata', 'augmentation', 'crop', 'extra_eval', 'lr', 'seed'])
def test_legacy_boundary_scientific_keys_and_windows_paths(change, monkeypatch):
    """Keep all real hash checks; substitute only parsed legacy metadata."""
    spec = spec_for()
    args = migration_args(spec)
    original_read = mil.read
    old = original_read(ROOT/args['protocol'])
    if change == 'windows':
        old['paths']['corpus'] = old['paths']['corpus'].replace('/', '\\')
        old['paths']['base_ordered_ids'] = old['paths']['base_ordered_ids'].replace('/', '\\')
        old['label_set']['path'] = old['label_set']['path'].replace('/', '\\')
        old['init_weights']['local'] = old['init_weights']['local'].replace('/', '\\')
        for key in ('ordered_ids', 'groups'):
            old['folds']['0'][key]['path'] = old['folds']['0'][key]['path'].replace('/', '\\')
    elif change == 'metadata':
        old['common_args'].update(workers=1, tag='new tag', wall_budget=0)
        old['resource_args'] = dict(wall_seconds=0, max_seconds=0)
        old['variant_config']['source'] = 'different prose'
    elif change == 'augmentation':
        old['variant_config']['gamma'] = [.9, 1.1]
    elif change == 'crop':
        old['exp179']['detail_slice'] = [40, 280]
    elif change == 'extra_eval':
        old['extra_eval']['k_eval'] = 24
    elif change == 'lr':
        old['common_args']['bb_lr'] *= 2
    else:
        old['seed'] = 43
    def read(path):
        return copy.deepcopy(old) if Path(path) == ROOT/args['protocol'] else original_read(path)
    monkeypatch.setattr(mil, 'read', read)
    if change in ('windows', 'metadata'):
        assert reg.migrate(ROOT, **args).sha256 == spec.sha256
    else:
        with pytest.raises(life.ContractError):
            reg.migrate(ROOT, **args)


def test_cli_create_only_and_error_receipts(tmp_path, capsys):
    spec = spec_for()
    args = migration_args(spec)
    out = tmp_path/'spec.json'
    argv = ['--root', str(ROOT), '--out', str(out)]
    for key, value in args.items():
        if key != 'control_hashes':
            argv.extend(['--'+key.replace('_', '-'), value])
    for name, digest in args['control_hashes'].items():
        argv.extend(['--'+name.removesuffix('.json')+'-sha256', digest])
    assert mil_register.main(argv) == 0
    assert life.parse_json(capsys.readouterr().out)['status'] == 'REGISTERED_NO_EXECUTION'
    before = out.read_bytes(), out.stat().st_mtime_ns
    assert mil_register.main(argv) == 2
    assert life.parse_json(capsys.readouterr().out)['status'] == 'REJECTED'
    assert (out.read_bytes(), out.stat().st_mtime_ns) == before
    argv[argv.index('--protocol-sha256')+1] = '0'*64
    argv[argv.index('--out')+1] = str(tmp_path/'invalid.json')
    assert mil_register.main(argv) == 2
    assert life.parse_json(capsys.readouterr().out)['status'] == 'REJECTED'
    assert not (tmp_path/'invalid.json').exists()


def member_spec(fold=1):
    return life.ArmSpec((ROOT/f'training/configs/mil-member-s43-f{fold}-v1.json').read_bytes())


def member_args(spec):
    evidence = spec.document['inputs']['evidence']
    target = next(p for p, h in evidence.items()
                  if h == spec.document['target_environment']['evidence']['declaration.json'])
    recipe = next(p for p in evidence if p != target)
    return dict(recipe=recipe, recipe_sha256=evidence[recipe], target=target,
                target_sha256=evidence[target], arm=spec.document['arm'], attempt=spec.document['attempt'])


@pytest.mark.parametrize('fold', range(1, 5))
def test_real_production_registration_preserves_recipe_without_reading_controls(fold, monkeypatch):
    spec = member_spec(fold)
    args = member_args(spec)
    recipe = life.ArmSpec((ROOT/args['recipe']).read_bytes())
    original_path = life._path
    controls = set(recipe.document['inputs']['controls'])
    def guarded_path(root, name):
        assert name not in controls, 'production member tried to open a paired control'
        return original_path(root, name)
    monkeypatch.setattr(life, '_path', guarded_path)
    args['recipe'] = args['recipe'].replace('/', '\\')
    args['target'] = args['target'].replace('/', '\\')
    actual = reg.register_member(ROOT, **args)
    assert actual.sha256 == spec.sha256 and actual.run_kind == 'member'
    assert actual.scope.startswith('member/') and actual.document['inputs']['controls'] == {}
    assert all(actual.document[k] == recipe.document[k]
               for k in ('family', 'seed', 'fold', 'config', 'outputs'))
    assert all(actual.document['inputs'][k] == recipe.document['inputs'][k] for k in ('data', 'rosters'))
    assert set(actual.inventory).isdisjoint(controls)


@pytest.mark.parametrize('field', ['recipe_sha256', 'target_sha256'])
def test_member_wrong_trusted_digest_rejected(field):
    args = member_args(member_spec())
    args[field] = '0'*64
    with pytest.raises(life.ContractError):
        reg.register_member(ROOT, **args)


@pytest.mark.parametrize('change', ['no_recipe', 'controls', 'run_kind', 'old_schema', 'target_schema', 'no_evidence'])
def test_member_strict_schema_rejects_mixed_or_missing_identity(change):
    spec = member_spec()
    doc = spec.document
    if change == 'no_recipe':
        del doc['inputs']['evidence'][member_args(spec)['recipe']]
    elif change == 'controls':
        doc['inputs']['controls'] = spec_for().document['inputs']['controls']
    elif change == 'run_kind':
        doc['run_kind'] = 'experiment'
    elif change == 'old_schema':
        doc['schema'] = life.SPEC_SCHEMA
    elif change == 'target_schema':
        doc['target_environment'] = spec_for().document['target_environment']
    else:
        del doc['inputs']['evidence']
    with pytest.raises(life.ContractError):
        life.ArmSpec.from_document(doc)


@pytest.mark.parametrize('change', ['seed', 'fold', 'backend', 'labels', 'updates', 'hostname', 'runtime', 'evidence'])
def test_member_recipe_and_target_tampering_rejected(change):
    doc = member_spec().document
    if change in ('seed', 'fold'):
        doc[change] = 42 if change == 'seed' else 2
    elif change == 'backend':
        doc['config']['backend_policy']['cudnn_benchmark'] = True
    elif change == 'labels':
        doc['config']['labels']['semantic_sha256'] = '0'*64
    elif change == 'updates':
        doc['config']['updates']['steps'] += 1
    elif change == 'hostname':
        doc['target_environment']['hostname'] = 'not-declared'
    elif change == 'runtime':
        doc['target_environment']['runtime']['torch'] = 'not-declared'
    else:
        doc['inputs']['evidence'][member_args(member_spec())['recipe']] = '0'*64
    with pytest.raises(life.ContractError):
        life.validate_inputs(life.ArmSpec.from_document(doc), ROOT)


def test_member_target_declaration_never_substitutes_for_fresh_observation():
    spec = member_spec()
    target = life.parse_json((ROOT/member_args(spec)['target']).read_bytes())
    life.check_target_observation(spec, target)
    with pytest.raises(life.ContractError):
        life.check_target_observation(spec, mil.observation('cpu'))
    target['backend_policy']['cudnn_benchmark'] = True
    with pytest.raises(life.ContractError):
        life.check_target_observation(spec, target)


def test_member_cli_create_only_and_mixed_kind_rejected(tmp_path, capsys):
    spec = member_spec()
    argv = ['--kind', 'member', '--root', str(ROOT), '--out', str(tmp_path/'member.json')]
    for key, value in member_args(spec).items():
        argv.extend(['--'+key.replace('_', '-'), value])
    assert mil_register.main(argv) == 0
    assert life.parse_json(capsys.readouterr().out)['status'] == 'REGISTERED_NO_EXECUTION'
    path = tmp_path/'member.json'
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert mil_register.main(argv) == 2
    capsys.readouterr()
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    argv[argv.index('--out')+1] = str(tmp_path/'mixed.json')
    assert mil_register.main(argv + ['--control', 'unused-control']) == 2
    capsys.readouterr()
    argv[argv.index('--kind')+1] = 'experiment'
    assert mil_register.main(argv) == 2
    capsys.readouterr()
    assert not (tmp_path/'mixed.json').exists()
