"""Focused production adapter checks; no GPU, training import, or remote writes."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import d01_candidate_v3 as candidate
import d01_package as package
import d01_release_v2 as suffix

TRIAL = ROOT / '.local/d01-integrated-trial/20261003-v3'
RESULT = TRIAL / 'remote-exact/integrated-trial-result.json'


def mount(p):
    p.mkdir(parents=True)
    for name in ('test.csv', 'test_series.csv', 'sample_submission.csv', 'train.csv'):
        (p / name).write_text('StudyInstanceUID\na\nb\nc\n', encoding='utf-8')
    (p / 'test_series').mkdir()


@pytest.mark.parametrize('layout', ['flat', 'nested'])
def test_actual_single_root_and_ids(tmp_path, layout):
    flat, nested = tmp_path / 'flat', tmp_path / 'nested'; chosen = flat if layout == 'flat' else nested
    mount(chosen)
    assert candidate.resolve_competition([flat, nested, chosen]) == chosen
    assert candidate.test_ids(chosen) == ['a', 'b', 'c']


@pytest.mark.parametrize('mode', ['missing', 'ambiguous', 'incomplete'])
def test_reject_bad_mount(tmp_path, mode):
    a, b = tmp_path / 'a', tmp_path / 'b'
    if mode != 'missing': mount(a)
    if mode == 'ambiguous': mount(b)
    if mode == 'incomplete': (a / 'train.csv').unlink()
    with pytest.raises(ValueError): candidate.resolve_competition([a, b])


@pytest.mark.parametrize('rows', ['StudyInstanceUID\na\na\n', 'StudyInstanceUID\n', 'StudyInstanceUID\n../x\n', 'other\na\n'])
def test_reject_bad_actual_ids(tmp_path, rows):
    (tmp_path / 'test.csv').write_text(rows)
    with pytest.raises(ValueError): candidate.test_ids(tmp_path)


@pytest.mark.parametrize('n', [3, 133, 1300, 1301])
def test_actual_n_and_shared_budget_not_trial_caps(n):
    r = candidate.read(RESULT); b = candidate.budget(n, r)
    assert b['actual_n'] == n and b['effective_n'] == max(n, 133)
    assert b['internal_deadline_seconds'] == 32100 and b['minimum_reserve_pass']
    assert b['projected_seconds'] == max(r['production_core_seconds'] * max(n, 133) / 133, 30938.468180919 * max(n, 133) / 1300)
    assert b['phase_estimates_are_not_caps']


def test_large_actual_n_rejects_estimate_without_truncation():
    b = candidate.budget(1400, candidate.read(RESULT))
    assert b['actual_n'] == 1400 and not b['minimum_reserve_pass']


@pytest.mark.parametrize('n,seconds', [(1, 1800), (2, 1800), (3, 1800), (4, 32100), (133, 32100), (1300, 32100), (1301, 32100)])
def test_actual_cardinality_deadline_boundary(n, seconds):
    assert candidate.execution_seconds(n) == seconds
    # Admission estimate remains the preregistered Neff rule, independently of
    # the tighter small-cohort execution limit.
    assert candidate.budget(n, candidate.read(RESULT))['internal_deadline_seconds'] == 32100


@pytest.mark.parametrize('n', [0, -1, 3.0, True])
def test_deadline_rejects_invalid_cardinality(n):
    with pytest.raises(ValueError): candidate.execution_seconds(n)


def test_exact_trial_bindings_and_original_suffix():
    result, bindings = candidate.verify_trial(TRIAL, RESULT)
    assert result['status'] == 'PASS' and len(bindings) == 8
    assert not bindings['public3-process/phase-receipt.json']['result_byte_sha_binding']
    original = candidate.read(TRIAL / 'payload/anchor-original.ipynb')
    worker = suffix.anchor_worker_notebook(original)
    assert len(worker['cells']) == 31 and worker['cells'][:30] == original['cells']
    assert 'D01_ANCHOR_OUTPUT' in ''.join(worker['cells'][-1]['source'])


def test_stale_result_rejected(tmp_path):
    p = tmp_path / 'result.json'; p.write_bytes(RESULT.read_bytes() + b' ')
    with pytest.raises(ValueError, match='exact integrated'): candidate.verify_trial(TRIAL, p)


def test_changed_dependency_download_version_rejected(tmp_path):
    remote = tmp_path / 'remote'; shutil.copytree(RESULT.parent, remote)
    p = remote / 'r2/receipt.json.download.json'; d = candidate.read(p); d['version'] = 1; candidate.dump(p, d)
    with pytest.raises(ValueError, match='exact-version'): candidate.verify_trial(TRIAL, remote / RESULT.name)


def frame(ids, values):
    x = pd.DataFrame(np.repeat(np.asarray(values)[:, None], 12, axis=1), columns=candidate.LABELS)
    x.insert(0, candidate.UID, ids); return x


def test_whole_cohort_mean5_before_rank_ties_and_order():
    ids = ['a', 'b', 'c', 'd']
    a = frame(ids, [.4, .4, .2, .9])
    raw = np.asarray([[.9, .2, .3, .7], [.1, .8, .5, .2], [.4, .7, .2, .1], [.3, .9, .6, .4], [.2, .8, .7, .6]], dtype=np.float32)
    r = frame(ids, raw.mean(0, dtype=np.float32)); output = package.blend_frames(a.iloc[::-1], r.iloc[::-1], ids)
    candidate.validate_frame(output, ids)
    expected = .9 * a[candidate.LABELS].rank(method='average', pct=True) + .1 * r[candidate.LABELS].rank(method='average', pct=True)
    assert np.array_equal(output[candidate.LABELS].to_numpy(), expected.to_numpy())
    wrong = pd.concat([package.blend_frames(a.iloc[::2], r.iloc[::2], ids[::2]), package.blend_frames(a.iloc[1::2], r.iloc[1::2], ids[1::2])]).set_index(candidate.UID).loc[ids]
    assert not np.array_equal(output[candidate.LABELS].to_numpy(), wrong.to_numpy())


@pytest.mark.parametrize('error', ['missing', 'order', 'nan', 'schema'])
def test_final_schema_fails_closed(error):
    ids = ['a', 'b', 'c']; x = frame(ids, [.1, .2, .3])
    if error == 'missing': x = x.iloc[:2]
    if error == 'order': x = x.iloc[::-1]
    if error == 'nan': x.loc[0, 'ACL'] = np.nan
    if error == 'schema': x = x.drop(columns=['ACL'])
    with pytest.raises(ValueError): candidate.validate_frame(x, ids)


def test_publication_receipt_precedes_atomic_final(tmp_path, monkeypatch):
    ids = ['a', 'b']; x = frame(ids, [.2, .7]); final = tmp_path / 'submission.csv'; seen = []
    original = candidate.os.replace
    def replace(src, dst):
        if Path(dst) == final:
            assert candidate.read(tmp_path / 'candidate-receipt.json')['status'] == 'COMPLETE'
            assert not final.exists(); seen.append('publish')
        return original(src, dst)
    monkeypatch.setattr(candidate.os, 'replace', replace)
    candidate.publish(x, ids, tmp_path, final, {'status': 'RUNNING'}, time.monotonic() + 30)
    assert seen == ['publish'] and final.is_file()
    assert candidate.read(tmp_path / 'candidate-receipt.json')['submission_sha256'] == candidate.sha(final)


def test_expired_deadline_never_publishes(tmp_path):
    with pytest.raises(ValueError, match='deadline'):
        candidate.publish(frame(['a'], [.3]), ['a'], tmp_path, tmp_path / 'submission.csv', {}, time.monotonic() - 1)
    assert not (tmp_path / 'submission.csv').exists()


@pytest.mark.parametrize('error', [None, 'four_members', 'nan', 'wrong_order', 'mean_after_rank', 'wrong_checkpoint'])
def test_five_member_output_contract(tmp_path, error):
    ids = ['a', 'b', 'c']; work = tmp_path / 'r2'; work.mkdir()
    helper = candidate.module(TRIAL / 'payload/helper/d01_release_v3.py', 'd01_candidate_test_helper')
    reference = candidate.read(TRIAL / 'payload/helper/reference-receipt.json')
    receipt = {'status': 'COMPLETE', 'split': 'test', 'ids': ids, 'studies': 3,
               'shape': [5, 3, 12], 'member_count': 5, 'all_finite': True, 'mean_before_rank': True,
               'fp32_fallbacks': 0, 'neutral_fills': 0, 'failures': [],
               'release_helper_sha256': candidate.V3_SHA, 'policy_sha256': helper.POLICY_SHA,
               'member_identities': [{'name': x['name'], 'sha256': x['checkpoint_sha256']}
                                     for x in reference['inference']['member_results']]}
    raw = np.random.default_rng(1929).uniform(.1, .9, (5, 3, 12)).astype(np.float32)
    mean = pd.DataFrame(raw.mean(0, dtype=np.float32), columns=candidate.LABELS); mean.insert(0, candidate.UID, ids)
    if error == 'four_members': raw = raw[:4]
    if error == 'nan': raw[0, 0, 0] = np.nan
    if error == 'wrong_order': ids = ids[::-1]
    if error == 'mean_after_rank': mean.loc[0, 'ACL'] = .999
    if error == 'wrong_checkpoint': receipt['member_identities'][0]['sha256'] = '0' * 64
    candidate.dump(work / 'receipt.json', receipt)
    np.savez_compressed(work / 'members.npz', ids=np.asarray(ids), member_probabilities=raw)
    mean.to_csv(work / 'mean.csv', index=False)
    if error:
        with pytest.raises(ValueError): candidate.verify_r2_output(TRIAL / 'payload', work, ['a', 'b', 'c'], helper)
    else:
        actual = candidate.verify_r2_output(TRIAL / 'payload', work, ids, helper)
        assert np.array_equal(actual[candidate.LABELS].to_numpy(dtype=np.float32), raw.mean(0, dtype=np.float32))


@pytest.mark.parametrize('fail_phase', ['anchor', 'r2', None])
def test_stubbed_production_order_cleanup_and_failures(tmp_path, monkeypatch, fail_phase):
    base = tmp_path / 'working/base'; shutil.copytree(TRIAL / 'payload', base)
    (base / 'evidence').mkdir(); shutil.copy2(RESULT, base / 'evidence/integrated-trial-result.json')
    candidate.dump(base / 'candidate-manifest.json', {'files': {}, 'provenance': {'unknown_versions': True}})
    actual = tmp_path / 'input/rsna-knee-abnormality-detection'; mount(actual)
    assets = tmp_path / 'input/knee-d01-r2-assets'; assets.mkdir(); (assets / 'd01-assets-manifest.json').write_text('{}')
    codec = tmp_path / 'input/knee-d01-codec-assets'; codec.mkdir()
    final = tmp_path / 'working/submission.csv'; order = []; deadlines = []
    real_path = Path
    def path(p):
        s = str(p)
        if s.startswith('/kaggle/'):
            return tmp_path / s[len('/kaggle/'):]
        return real_path(p)
    monkeypatch.setattr(candidate, 'Path', path)
    monkeypatch.setattr(candidate, 'os', SimpleNamespace(name='posix', environ=os.environ, replace=os.replace))
    helper = SimpleNamespace(ASSETS_SHA=candidate.sha(assets / 'd01-assets-manifest.json'), validate_bundle=lambda p: None,
                             blend=lambda assets, a, r, ids: (order.append('blend') or package.blend_frames(a, r, ids)))
    def run_phase(name, command, env, work, cap, deadline, supervisor):
        order.append(name)
        deadlines.append(deadline)
        assert len(set(deadlines)) == 1
        assert cap == 1800 and 1700 < deadline - time.monotonic() <= 1800
        if name == 'anchor':
            assert env['RSNA_COMP_ROOT'] == str(actual)
            assert 'RSNA_PARALLEL_COAT_READERS' not in env
            final.write_text('anchor must not survive failure')
            if fail_phase == 'anchor': raise TimeoutError('owned anchor timeout')
            os.replace(final, env['D01_ANCHOR_OUTPUT'])
            frame(['a', 'b', 'c'], [.1, .4, .3]).to_csv(env['D01_ANCHOR_OUTPUT'], index=False)
            candidate.dump(Path(env['D01_ANCHOR_OUTPUT']).with_suffix('.json'), {'status': 'COMPLETE', 'cohort': 3, 'dino_members': 20, 'a5_folds': 5, 'raptor_views': 4, 'coat_family_members': 4})
        else:
            assert order == ['anchor', 'r2'] and not final.exists()
            assert '--competition' in command and command[command.index('--competition') + 1] == str(actual)
            if fail_phase == 'r2': raise RuntimeError('R2 missing member')
            p = base / 'results/r2'; p.mkdir(); candidate.dump(p / 'receipt.json', {'stub': True})
        order.append(name + '_cleanup') if name == 'r2' else None
        return {'all_owned_processes_reaped': True}
    # Record cleanup separately while keeping the phase-order assertion explicit.
    monkeypatch.setattr(candidate, 'module', lambda p, name: helper if name == 'd01_candidate_helper' else SimpleNamespace(run_phase=run_phase) if name == 'd01_candidate_process_control' else suffix if name == 'd01_candidate_anchor_suffix' else SimpleNamespace())
    monkeypatch.setattr(candidate, 'verify_r2_output', lambda base, work, ids, helper: frame(ids, [.2, .3, .1]))
    if fail_phase:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)): candidate.execute(base, time.monotonic())
        assert not final.exists()
        assert candidate.read(base / 'results/candidate-receipt.json')['status'] == 'FAILED'
    else:
        result = candidate.execute(base, time.monotonic())
        assert result['status'] == 'COMPLETE' and final.is_file()
        assert result['deadline_seconds'] == 1800
        assert order == ['anchor', 'r2', 'r2_cleanup', 'blend']


def test_stage_private_offline_source_preservation(tmp_path):
    out = tmp_path / 'stage'; review = candidate.stage(TRIAL, RESULT, out)
    assert review['status'] == 'CPU_STAGED_NO_REMOTE_OR_GPU' and review['notebook_bytes'] <= 1_000_000
    meta = candidate.read(out / 'kernel-metadata.json'); old = candidate.read(TRIAL / 'kernel-metadata.json')
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is False
    for key in ('dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources', 'docker_image', 'machine_shape'):
        assert meta[key] == old[key]
    manifest = candidate.read(out / 'payload/candidate-manifest.json')
    assert manifest['execution_seconds'] == {'actual_n_le_3': 1800, 'actual_n_gt_3': 32100}
    assert review['planned_kaggle_session_timeout_seconds'] == 32400
    for name, digest in manifest['files'].items(): assert candidate.sha(out / 'payload' / name) == digest
    for name in candidate.read(TRIAL / 'payload/trial-manifest.json')['files']:
        assert (out / 'payload' / name).read_bytes() == (TRIAL / 'payload' / name).read_bytes()
    code = ''.join(candidate.read(out / meta['code_file'])['cells'][1]['source']); ast.parse(code)
    assert 'candidate.execute(base,started)' in code and 'trial.execute' not in code
    tree = ast.parse(Path(candidate.__file__).read_text(encoding='utf-8'))
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not {'install_proxy_probe_mask', 'anchor_notebook', 'runtime_budget', 'build'} & calls
