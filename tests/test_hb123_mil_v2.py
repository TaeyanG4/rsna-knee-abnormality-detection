"""Single-difference integration, immutable bindings and actual CPU card checks."""
import ast
import inspect
from pathlib import Path

import pytest

from training.arms import mil_laterality_v2 as arm
from tools import hb123_build_v2 as build


def test_run_only_rendering_and_provenance_differ():
    source = inspect.getsource(arm.shared.run)
    source = source.replace("mode, device, workers=0", "mode, device, workers=2")
    source = source.replace('need(type(workers) is int and 0 <= workers <= 2, "screen permits at most two workers")',
                            'need(type(workers) is int and workers == 2, "shared control requires exactly two persistent workers")')
    source = source.replace(
        'dataset = MILWindows(corpus_dir, train_ids, selected_labels, 12, True, doc["seed"])',
        'dataset = canonical_dataset(MILWindows(corpus_dir, train_ids, selected_labels, 12, True, doc["seed"]), spec, stage)')
    source = source.replace(
        'ds = MILWindows(corpus_dir, val_ids, selected_labels, k, False, doc["seed"])',
        'ds = canonical_dataset(MILWindows(corpus_dir, val_ids, selected_labels, k, False, doc["seed"]), spec, stage)')
    source = source.replace('screen="HB122_SHARED_V1",',
                            'screen="HB123_LATERALITY_V1", rendering=c["windows"]["laterality_v1"],')
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(inspect.getsource(arm.run)))
    assert ast.dump(ast.parse(inspect.getsource(arm.validate_data))) == ast.dump(ast.parse(inspect.getsource(arm.shared.validate_data)))
    for name in ('MILWindows', 'build_model', 'train_epoch', 'optimizer_stack', 'evaluate',
                 'record_smoke', 'restore_epoch', 'smoke_pair', 'seed_all', 'worker_init'):
        assert getattr(arm, name) is getattr(arm.shared, name)


@pytest.mark.parametrize('seed', [42, 43])
def test_registered_single_difference_and_cpu_evidence(seed):
    spec = build.spec_for(seed)
    c = arm.validate_recipe(spec)
    arm.life.validate_inputs(spec, arm.ROOT)
    control = arm.life.ArmSpec((arm.ROOT / arm.control_path(seed)).read_bytes())
    changed = spec.document['config']
    changed['windows'].pop('laterality_v1')
    assert changed == control.document['config']
    assert spec.document['target_environment'] == control.document['target_environment']
    run = build.WORK / f's{seed}'
    card = arm.read(run / 'cpu-card.json')
    assert arm.read(run / 'cpu/outputs/provenance.json')['identity']['workers'] == 2
    assert card['status'] == 'CPU_CARD_PASS' and card['serving_byte_parity']
    assert len(card['readout_parity']) == 3 and all(r['byte_equal'] for r in card['readout_parity'].values())
    assert card['summary']['results']['optimizer_steps'] == 1
    assert card['summary']['cpu_cuda_initialized'] is False
    assert build.admit(seed)['status'] == 'HOLD'
    assert (run / 'cpu-infer.csv').read_bytes() == (run / 'cpu/outputs/oof_k36_flip_tta.csv').read_bytes()
    assert arm.read(run / 'cpu/outputs/provenance.json')['rendering'] == c['windows']['laterality_v1']


@pytest.mark.parametrize('change', ['crop', 'steps', 'loss', 'target', 'source', 'map', 'control', 'seed', 'plane', 'unknown'])
def test_recipe_drift_rejected(change):
    doc = build.spec_for(42).document
    c = doc['config']
    if change == 'crop': c['windows']['train_k'] = 24
    if change == 'steps': c['updates']['steps'] += 1
    if change == 'loss': c['updates']['weak_rank_v1'] = arm.shared.ranking.CONTRACT
    if change == 'target': doc['target_environment']['hostname'] += '-other'
    if change == 'source': doc['inputs']['sources']['training/arms/mil_screen_v1.py'] = '0'*64
    if change == 'map': c['windows']['laterality_v1']['side_map_sha256'] = '0'*64
    if change == 'control': c['windows']['laterality_v1']['control_spec_sha256'] = '0'*64
    if change == 'seed': doc['seed'] = 43
    if change == 'plane': c['windows']['laterality_v1']['policy']['sagittal'] = 'reverse'
    if change == 'unknown': c['windows']['laterality_v1']['policy']['unknown'] = 'drop'
    with pytest.raises(arm.life.ContractError):
        arm.validate_recipe(arm.life.ArmSpec.from_document(doc))


def test_infer_rejects_legacy_checkpoint(tmp_path, monkeypatch):
    spec = build.spec_for(42)
    run = build.WORK / 's42'
    ids = arm.read(run / 'infer-ids.json')
    monkeypatch.setattr(arm, 'verify_shards', lambda *a: None)
    monkeypatch.setattr(arm, 'configure_runtime', lambda *a, **k: None)
    monkeypatch.setattr(arm.torch, 'load', lambda *a, **k: {'provenance': {}})
    with pytest.raises(arm.life.ContractError, match='laterality provenance'):
        arm.infer(spec, run / 'stage', build.CORPUS, 'synthetic-legacy.pt', ids, tmp_path / 'pred.csv')
    assert not (tmp_path / 'pred.csv').exists()


def test_missing_map_ids_fail_before_window_read():
    spec = build.spec_for(42)
    with pytest.raises(ValueError, match='lacks required'):
        arm.side_map_for(spec, arm.ROOT).require(['synthetic-unknown-id'])


def test_full_cpu_and_workers_rejected_before_execution():
    for kwargs in (dict(mode='full', device='cpu'), dict(mode='smoke', device='cpu', workers=3)):
        with pytest.raises(arm.life.ContractError):
            arm.run(None, None, None, None, **kwargs)
