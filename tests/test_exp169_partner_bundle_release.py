"""CPU only checks for EXP169 aggregation, staging guards, and receipt behavior."""
import csv
import hashlib
import json
import shutil
import zipfile

import numpy as np
import pytest

from tools import exp169_partner_bundle_release as release


def _scores(n=7):
    base = np.tile(np.linspace(.1, .9, n)[:, None], (1, 12))
    return {'l2': np.stack([base + .001 * i for i in range(10)]),
            'dino': np.stack([base[::-1] + .001 * i for i in range(5)]),
            'convnext': np.stack([base + .001 * i for i in range(5)])}


def test_partner_aggregation_requires_all_twenty_arms():
    bundle, family = release.aggregate(_scores())
    assert bundle.shape == (7, 12)
    assert set(family) == {'l2', 'dino', 'convnext'}
    assert np.all((bundle >= 0) & (bundle <= 1))
    values = _scores()
    values['dino'] = values['dino'][:4]
    with pytest.raises(ValueError, match='Invalid family'):
        release.aggregate(values)


def test_final_is_fixed_global_rank_blend():
    trio = np.tile(np.array([.1, .4, .9])[:, None], (1, 12))
    bundle = trio[::-1]
    final = release.final_values(trio, bundle)
    assert np.allclose(final[:, 0], [.15, .5, .85])
    with pytest.raises(ValueError, match='Invalid trio'):
        release.final_values(trio, np.full_like(bundle, np.nan))


def test_checkpoint_matrix_and_guard_attributes():
    entries = release.source_paths()
    assert len(entries) == 20
    assert [(e['seed'], e['fold']) for e in entries if e['family'] == 'l2'] == [
        (seed, fold) for seed in (42, 43) for fold in range(5)]
    assert len([e for e in entries if e['family'] == 'dino']) == 5
    assert len([e for e in entries if e['family'] == 'convnext']) == 5
    assert [e['path'].parent.name for e in entries if e['family'] == 'convnext'] == [
        f'EXP170_CONVNEXT_T_L2_S42_F{fold}' for fold in range(5)]
    assert all('EXP166' not in str(e['path']) and 'EXP167' not in str(e['path'])
               for e in entries if e['family'] == 'convnext')
    assert release.guard_attributes()['raptor_common.py'] == [
        'build_model', 'select_centers', 'windows_from']


def test_label_parity_admission_is_bound_to_screen_and_oof_hashes():
    screen_path, result = release.admitted_screen()
    assert 'EXP170_CONVNEXT_T_L2' in str(screen_path)
    assert result['status']['status'] == 'ADMIT'
    assert result['bundle_minus_l2']['primary_mean_ci95'][0] > 0
    assert result['oof_policy']['convnext'] == 'flip-TTA oof_k36_flip_tta.csv'


def test_l2_seed_average_is_per_fold_before_global_family_rank():
    scores = _scores(5)
    scores['l2'] = np.zeros((10, 5, 12), dtype=float)
    # Two seeds each disagree by fold, but their paired fold mean is constant.
    for fold in range(5):
        scores['l2'][fold] = np.tile(np.arange(5)[:, None] / 8, (1, 12))
        scores['l2'][fold + 5] = 1 - scores['l2'][fold]
    _, family = release.aggregate(scores)
    assert np.allclose(family['l2'], .5)


def test_all_transitive_serving_imports_are_packaged():
    imports = release.guard_transitive_imports()
    assert 'raptor_data' in imports['raptor_radimagenet.py']
    assert 'raptor_common' in imports['raptor_radimagenet.py']


def test_parity_plan_compares_all_three_visible_trio_arms(tmp_path):
    stage = tmp_path / 'stage'
    dataset = stage / 'dataset'
    dataset.mkdir(parents=True)
    archive = dataset / 'exp169-assets.zip.bin'
    archive.write_bytes(b'fixture')
    (stage / 'build-receipt.json').write_text(json.dumps({
        'status': 'STAGED_NOT_UPLOADED', 'archive_sha256': release.sha(archive),
        'notebook_sha256': 'a' * 64}), encoding='utf-8')
    plan_dir = tmp_path / 'plan'
    plan = release.parity_plan(stage, plan_dir)
    assert plan['status'] == 'FROZEN_NOT_RUN'
    remote = tmp_path / 'kernel-output' / 'exp169-trio'
    remote.mkdir(parents=True)
    ref = release.ROOT / 'artifacts/releases/EXP155_S42v1_1_stage_v12/blend3-t4-output'
    for name in ('raw.npz', 'inputs.json', 'binding.json', 'submission.csv', 'receipt.json'):
        shutil.copyfile(ref / name, remote / name)
    result = release.compare_parity(plan_dir, remote.parent, tmp_path / 'parity-result.json')
    assert result['status'] == 'PASS_VISIBLE_TRIO'
    assert set(result['trio_raw']) == {'own', 'public', 'exp143'}
    assert all(value['max_abs'] == 0 for value in result['trio_raw'].values())


def test_archive_unique_members(tmp_path):
    source = tmp_path / 'a'
    source.write_text('a', encoding='utf-8')
    archive = tmp_path / 'a.zip.bin'
    release._zip_unique(archive, {'code/a': source, 'config/workspace.toml': source})
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ['code/a', 'config/workspace.toml']
    with pytest.raises(ValueError, match='Unsafe'):
        release._zip_unique(tmp_path / 'bad.zip.bin', {'../bad': source})


def test_submission_contract_rejects_reorder_and_nonfinite(tmp_path):
    sample, result = tmp_path / 'sample.csv', tmp_path / 'submission.csv'
    with sample.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([release.UID, *release.TARGETS])
        writer.writerows([['a', *([0] * 12)], ['b', *([0] * 12)]])
    release.write_csv(result, ['a', 'b'], np.full((2, 12), .5), sample)
    assert release.validate_csv(result, sample)[0] == ['a', 'b']
    with result.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([release.UID, *release.TARGETS])
        writer.writerows([['b', *([0] * 12)], ['a', *([0] * 12)]])
    with pytest.raises(ValueError, match='order'):
        release.validate_csv(result, sample)


def test_plane_flags_never_flip_sagittal_and_reject_boundary():
    spec = {'slots': [('Sagittal', 1, 5), ('Coronal', 0, 5), ('Axial', -1, 5)], 'depth': 15}
    assert release.plane_flags([2, 7, 12, -1], spec) == [False, True, True, False]
    with pytest.raises(ValueError, match='boundary'):
        release.plane_flags([5], spec)


def test_notebook_routes_trio_into_separate_work_and_marks_fallback(tmp_path):
    folder = tmp_path / 'blend3'
    folder.mkdir()
    (folder / 'release.ipynb').write_text(json.dumps({'cells': [
        {'cell_type': 'markdown', 'source': 'intro'},
        {'cell_type': 'code', 'source': "from pathlib import Path\n_work = Path('/kaggle/working')\nmain()\n"}]}),
        encoding='utf-8')
    notebook = release.notebook(tmp_path, release.DATASET, 'a' * 64, 'b' * 64)
    source = notebook['cells'][-1]['source']
    assert '_work = Path(\'/kaggle/working/exp169-trio\')' in source
    assert "'status': 'FAILED_BUNDLE'" in source
    assert 'CUBLAS_WORKSPACE_CONFIG' in source
    compile(source, '<test-notebook>', 'exec')


def test_projection_is_planning_only():
    projected = release.projection(1300)
    assert projected['projected_seconds'] < 7 * 3600
    assert 'NOT a full-cohort measurement' in projected['basis']
