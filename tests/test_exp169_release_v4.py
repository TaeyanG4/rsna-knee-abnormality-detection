"""CPU checks for the EXP169 v4.1.3 A4/B4 release and its fresh runnable stage.

No checkpoint is deserialized and no GPU, notebook push or account API is
used. Synthetic fixtures exercise serving, timing and output cleanup; the
recorded parent visible-run outputs exercise the neutralisation arithmetic.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from scipy.stats import rankdata

from tools import exp155_own0949_oai_release_v12 as e155
from tools import exp169_release_v2 as v2
from tools import exp169_release_v3 as v3
from tools import exp169_release_v4 as v4


ROOT = v4.ROOT
STAGE = ROOT / 'artifacts/releases/EXP169_V4_A4_B4_s43_stage_v413'
OLD_STAGE = ROOT / 'artifacts/releases/EXP169_V4_A4_B4_s43_stage'  # v4.0.0, uploaded assets
RELEASES = ROOT / 'artifacts/releases'
TARGETS = v2.TARGETS


# ----------------------------------------------------------------- helpers
def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def _csv(path, ids, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([v2.UID, *TARGETS])
        writer.writerows([[uid, *[format(float(x), '.10g') for x in row]] for uid, row in zip(ids, values)])


def _read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == [v2.UID, *TARGETS]
    return [r[0] for r in rows[1:]], [r[1:] for r in rows[1:]]


def _r(a):
    """Independent average-tie percentile rank."""
    a = np.asarray(a, np.float64)
    return (rankdata(a, method='average', axis=0) - 1) / max(1, len(a) - 1)


def _tied(rng, n, levels=None):
    a = rng.random((n, 12))
    if levels:
        a = np.round(a * levels) / levels  # many exact ties
    return a


def _cell(stage, candidate):
    return ''.join(v4.read_json(stage / candidate / 'kernel/release.ipynb')['cells'][-1]['source'])


def _functions(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    return {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}


# ------------------------------------------------------- fresh stage checks
def test_fresh_stage_verifies_and_is_runnable():
    receipt = v4.verify_stage(STAGE)
    assert receipt['status'] == 'PLAN_ONLY_NOT_PUSHED' and receipt['version'] == v4.VERSION == '4.1.3'
    assert receipt['source_sha256'] == v4.sha(Path(v4.__file__))
    assert receipt['uploads'] == receipt['submissions'] == 0
    assert (STAGE / 's43/dataset' / v4.S43_ARCHIVE).is_file()
    assert receipt['s43_pending'] is None
    for candidate in ('A4', 'B4'):
        source = _cell(STAGE, candidate)
        assert not source.startswith('raise RuntimeError')
        assert source.startswith("from pathlib import Path\nprint('EXP169 %s release v4.1.3'" % candidate)
        nb = v4.read_json(STAGE / candidate / 'kernel/release.ipynb')
        assert nb['cells'][0]['source'].startswith('# EXP169 %s v4.1.3' % candidate)
        assert 'S43 (seed-43 fixed SWA/BN' in nb['cells'][0]['source']
        assert 'must be replaced after' not in nb['cells'][0]['source']


def test_uploaded_asset_archives_are_byte_identical():
    receipt = v4.read_json(STAGE / 'build-receipt.json')
    assert set(receipt['uploads_needed']) == {v4.CODE_DATASET}
    for dataset, pin in v4.UPLOADED.items():
        archive = STAGE / ('c4' if dataset == v4.C4_DATASET else 's43') / 'dataset' / pin['archive']
        assert v4.sha(archive) == pin['sha256'] and archive.stat().st_size == pin['bytes']
        assert v4.sha(archive) == v4.sha(OLD_STAGE / archive.relative_to(STAGE))
        assert receipt['uploaded_assets'][dataset]['unchanged'] is True


def test_both_candidates_bind_exact_own_seed_s43_and_datasets():
    for candidate in ('A4', 'B4'):
        source = _cell(STAGE, candidate)
        if candidate == 'A4':
            assert "_work = Path('/kaggle/working/exp169-own42')" in source
            assert source.index('_work155 = _work') < source.index("exp169-own42") < source.index('_work42 = _work')
            assert "work_trio=_work155" in source
        else:
            assert source.index('_work174 = _work') < source.index('_work = _work143') < source.index('_work42 = _work')
            assert 'knee-exp169-b174-v3-code' not in source and 'work174=_work174' in source
        assert v4.sha(STAGE / 's43/dataset' / v4.S43_ARCHIVE) in source
        assert v4.sha(STAGE / 'c4/dataset' / v4.C4_ARCHIVE) in source
        assert v4.sha(STAGE / 'code/dataset' / v4.CODE_ARCHIVE) in source
        meta = v4.read_json(STAGE / candidate / 'kernel/kernel-metadata.json')
        for dataset in ('taeyangg4/knee-exp169-b172-v2-assets', 'taeyangg4/knee-exp169-b172-own-v2-assets',
                        v4.C4_DATASET, v4.S43_DATASET, v4.CODE_DATASET):
            assert dataset in meta['dataset_sources']
        assert 'taeyangg4/knee-exp169-b174-v3-code' not in meta['dataset_sources']
        assert meta['enable_internet'] is False and meta['is_private'] is True
    assert v4.sha(ROOT / 'artifacts/releases/EXP143_S42v2_average_v101/exp143-v2-e14-e11-e12-bn.pt') == v4.S42_SHA


def test_c4_archive_has_only_validated_five_fold_weights():
    archive = STAGE / 'c4/dataset' / v4.C4_ARCHIVE
    manifest = v4.read_json(STAGE / 'c4/release-manifest.json')
    assert len(manifest['entries']) == 5
    assert [e['fold'] for e in manifest['entries']] == list(range(5))
    assert all(e['crop_policy'] == 'mixed' for e in manifest['entries'])
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {'release-manifest.json'} | {e['path'] for e in manifest['entries']}
        assert all(z.getinfo(e['path']).file_size == e['bytes'] for e in manifest['entries'])


def test_formula_manifest_and_timing_plans():
    manifest = v4.read_json(STAGE / 'code/assets/release-manifest.json')
    assert manifest['version'] == '4.1.3'
    assert 'no internal time limit; Kaggle 9 h' in manifest['timing_policy']
    assert manifest['formula'] == {
        'A4': '.85*r((r(public)+r(OWN)+r(EXP155))/3)+.15*r(Q4)',
        'B4': '.35*r(OWN)+.35*r(EXP174)+.30*r(Q4)',
        'OWN': '(P_S42+P_S43)/2',
        'Q4': 'r((r(R)+r(C4)+r(D))/3)',
    }
    assert 'float32' in manifest['arithmetic']['A4_trio']
    for candidate in ('A4', 'B4'):
        timing = v4.read_json(STAGE / candidate / 'timing-plan.json')
        parity = v4.read_json(STAGE / candidate / 'parity-plan.json')
        assert timing['included_passes'][:2] == ['S42 OWN', 'S43 OWN']
        assert timing['internal_time_limit_seconds'] is None
        assert timing['kaggle_seconds'] == v4.KAGGLE_SECONDS == 9 * 3600
        assert 'diagnostic only' in timing['projection_policy']
        estimate = timing['hidden_runtime_estimate']['candidates'][candidate]
        assert estimate['linear_seconds'] < estimate['conservative_seconds'] < 30000
        assert timing['hidden_runtime_estimate']['internal_time_limit_seconds'] is None
        assert parity['partner']['cross_gpu_tolerance_file'] == v4.TOL_REL
        assert parity['partner']['cross_gpu_tolerance_sha256'] == v4.sha(ROOT / v4.TOL_REL)


def test_c4_training_provenance_is_recorded_in_stage_receipt():
    evidence = v4.read_json(STAGE / 'build-receipt.json')['c4_provenance']
    assert evidence['status'] == 'TRAINING_POLICY_BOUND'
    fresh = v4.c4_training_evidence(ROOT)
    assert {k: v for k, v in evidence.items() if k != 'serving_window_parity'} == fresh
    manifest = v4.read_json(STAGE / 'c4/release-manifest.json')
    for fold, (record, entry) in enumerate(zip(evidence['folds'], manifest['entries'])):
        assert record['fold'] == fold and record['checkpoint_sha256'] == entry['sha256']
        assert record['crop_policy'] == 'mixed' and record['allocation'] == 'even=context,odd=detail'
        assert record['detail_slice'] == [48, 288] and record['nominal_mm'] == [140, 100]
        assert record['k_oof'] == 36 and record['window_policy'] == 'within-slot-valid'
        assert record['flip_tta'] == 'coronal_axial_horizontal_mean_probability'
        assert record['loader_sha256'] == v4.sha(ROOT / v4.C4_LOADER)
    parity = evidence['serving_window_parity']
    assert parity['plain_equal'] and parity['mirror_equal'] and parity['flags_equal']


def test_c4_serving_windows_equal_exp179_loader_on_cpu():
    result = v4.c4_window_parity(ROOT, seed=7)
    assert result['plain_equal'] and result['mirror_equal'] and result['dtype'] == 'float32'


def test_failure_envelope_binds_v4_artifact_not_parent():
    manifest_sha = v4.sha(STAGE / 'code/assets/release-manifest.json')
    code_sha = v4.sha(STAGE / 'code/dataset' / v4.CODE_ARCHIVE)
    parents = {'A4': ('39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc', 'VERIFICATION-v2.0.2'),
               'B4': ('7974cda41fec0185165fe1f1968cc6d81fce5896c8b9824a51c8f06925e69068', 'VERIFICATION-v3.0.0')}
    for candidate, (parent_sha, parent_note) in parents.items():
        source = _cell(STAGE, candidate)
        failure = source[source.index(v4.FAILURE_MARKER):]
        assert f'manifest_sha256={manifest_sha!r}' in failure
        assert repr(v4.failure_note(manifest_sha, code_sha)) in failure
        assert parent_sha not in source and parent_note not in source
        assert "candidate='A172'" not in source and "candidate='B174'" not in source
        assert "_fallback.unlink()" not in source
        assert "Path('/kaggle/working/submission.csv').unlink" not in source


def test_member_computations_are_unchanged_from_4_0_0():
    old, new = _functions(OLD_STAGE / 'code/assets/exp169_release_v4.py'), _functions(v4.__file__)
    for name in ('own_values', 'q4_values', 'c4_sources', 'c4_model_meta', 'partner_scores_c4',
                 's43_ready', 'bind_s43', 'authenticated_raw'):
        assert old[name] == new[name], name
    # pilot_c4 now clones the same parent pilot with its deadline set to infinity;
    # the generated-cell harness verifies that change for both candidates.
    # overlay_c4: the only change (4.1.1) is the atomic manifest replacement.
    source = Path(v4.__file__).read_text(encoding='utf-8')
    patched = source.replace("_replace_json(partner/'release-manifest.json', original)",
                             "write_json(partner/'release-manifest.json', original)")
    assert patched != source
    tree = ast.parse(patched)
    overlay = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'overlay_c4')
    assert ast.dump(overlay) == old['overlay_c4']


# ------------------------------------------------------------- arithmetic
def test_rank_is_average_tie_percentile():
    rng = np.random.default_rng(1)
    for n in (1, 2, 3, 11, 400):
        a = _tied(rng, n, levels=7)
        assert np.array_equal(v2.rank(a), _r(a))


@pytest.mark.parametrize('n,levels', [(1, None), (2, None), (3, 5), (7, 3), (50, None), (50, 9), (1300, None), (1300, 40)])
def test_formulas_match_independent_exact_implementation(n, levels):
    rng = np.random.default_rng(n * 101 + (levels or 0))
    s42, s43, pub, ex, e174, q = (_tied(rng, n, levels) for _ in range(6))
    own = (s42 + s43) / 2
    assert np.array_equal(v4.own_values(s42, s43), own)
    # Exact tie structure from integer rank numerators: 2*(average rank - 1).
    numerator = sum(rankdata(a, method='average', axis=0) * 2 - 2 for a in (ex, pub, own))
    expected_a4 = .85 * _r(numerator) + .15 * _r(q)
    expected_b4 = (.35 * _r(own) + .35 * _r(e174)) + .30 * _r(q)
    assert np.array_equal(v4.final_values('A4', s42, s43, q, public=pub, exp155=ex), expected_a4)
    assert np.array_equal(v4.final_values('B4', s42, s43, q, exp174=e174), expected_b4)


def test_q4_is_probability_mean_within_family_then_equal_family_ranks():
    rng = np.random.default_rng(4)
    scores = {f: rng.random((m, 9, 12)) for f, m in (('l2', 10), ('dino', 5), ('convnext', 5))}
    l2 = ((scores['l2'][:5] + scores['l2'][5:]) / 2).mean(0)
    expected = _r((_r(l2) + _r(scores['dino'].mean(0)) + _r(scores['convnext'].mean(0))) / 3)
    assert np.array_equal(v4.q4_values(scores), expected)


def _a172(public, member, exp155, bundle):
    trio = e155.output_values([exp155, public, member], 'blend3', None)
    trio_csv = np.asarray([[float(str(np.float32(x))) for x in row] for row in trio])  # parent CSV
    return v2.final_values(trio_csv, bundle, 'A172')


@pytest.mark.parametrize('n', [3, 50, 1300])
def test_parent_neutralisation_reproduces_a172_b174_b172_exactly(n):
    rng = np.random.default_rng(n)
    for levels in (None, 25):
        s42, pub, ex, e174, q = (_tied(rng, n, levels) for _ in range(5))
        a4 = v4.final_values('A4', s42, s42, q, public=pub, exp155=ex)
        assert np.array_equal(a4, _a172(pub, s42, ex, q))
        b4 = v4.final_values('B4', s42, s42, q, exp174=e174)
        assert np.array_equal(b4, v3.final_values(s42, e174, q))
        b172 = v4.final_values('B4', s42, s42, q, exp174=s42)
        assert np.array_equal(b172, v2.final_values(s42, q, 'B172'))


def test_4_0_0_trio_float64_sum_did_not_reproduce_a172():
    rng = np.random.default_rng(0)
    pub, s42, ex, q = (rng.random((1300, 12)) for _ in range(4))
    old = .85 * v2.rank((v2.rank(pub) + v2.rank(s42) + v2.rank(ex)) / 3) + .15 * v2.rank(q)
    new = v4.final_values('A4', s42, s42, q, public=pub, exp155=ex)
    a172 = _a172(pub, s42, ex, q)
    assert not np.array_equal(old, a172) and np.array_equal(new, a172)


def _visible(stage):
    return RELEASES / stage / 'kernel-output'


def _q(out):
    with np.load(out / 'partner-raw.npz', allow_pickle=False) as p:
        # float32 members exactly as the partner pass returns them to aggregate
        return p['ids'].tolist(), v4.q4_values({f: p[f] for f in ('l2', 'dino', 'convnext')})


def _raw(work, *keys):
    with np.load(work / 'raw.npz', allow_pickle=False) as raw:
        assert raw['targets'].tolist() == TARGETS
        return raw['ids'].tolist(), [raw[k].astype(np.float64) for k in keys]


@pytest.mark.parametrize('parent', ['A172', 'B172', 'B174'])
def test_neutralised_v4_reproduces_recorded_parent_visible_csv(parent):
    stage = {'A172': 'EXP169_A172_v2_0_2_stage', 'B172': 'EXP169_B172_v2_0_2_stage',
             'B174': 'EXP169_B174_v3_0_0_verified_stage'}[parent]
    out = _visible(stage)
    qids, q = _q(out)
    if parent == 'A172':
        ids, (own, public, exp143) = _raw(out / 'exp169-trio', 'own', 'public', 'exp143')
        final = v4.final_values('A4', exp143, exp143, q, public=public, exp155=own)
    elif parent == 'B172':
        ids, (own,) = _raw(out / 'exp169-trio', 'own')
        final = v4.final_values('B4', own, own, q, exp174=own)
    else:
        ids, (own,) = _raw(out / 'exp169-trio', 'own')
        ids174, (e174,) = _raw(out / 'exp169-anchor174', 'own')
        assert ids174 == ids
        final = v4.final_values('B4', own, own, q, exp174=e174)
    csv_ids, values = _read_csv(out / 'submission.csv')
    assert csv_ids == ids == qids and len(set(ids)) == len(ids)
    assert values == [[format(float(x), '.10g') for x in row] for row in final]


# -------------------------------------------------- synthetic serving fixture
class Clock:
    def __init__(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t


def _anchor(folder, ids, sample, keys, binding, seconds, rng):
    folder.mkdir(parents=True)
    arrays = {k: rng.random((len(ids), 12)) for k in keys}
    np.savez(folder / 'raw.npz', ids=np.asarray(ids), targets=np.asarray(TARGETS), **arrays)
    _json(folder / 'binding.json', binding)
    _json(folder / 'inputs.json', {'synthetic': True})
    _json(folder / 'runtime.json', {'device': 'synthetic-cpu'})
    _csv(folder / 'submission.csv', ids, arrays[keys[0]])
    _json(folder / 'receipt.json', {'status': 'COMPLETE_INFERENCE', 'seconds': float(seconds), 'files': {
        n: v4.sha(folder / n) for n in ('raw.npz', 'inputs.json', 'runtime.json', 'binding.json', 'submission.csv')}})
    return arrays


def _release(tmp_path, candidate, *, n=6, anchors=(4000., 3000., 3000.), pilot_elapsed=500.,
             projected=24975., partner_seconds=12500.):
    rng = np.random.default_rng(41)
    ids = [f'synthetic-{i}' for i in range(n)]
    competition, output = tmp_path / 'competition', tmp_path / 'working'
    _csv(competition / 'sample_submission.csv', ids, np.zeros((n, 12)))
    sample = competition / 'sample_submission.csv'
    partner = tmp_path / 'partner'
    files, entries = {}, []

    def add(name, content):
        path = partner / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files[name] = {'sha256': v4.sha(path), 'bytes': path.stat().st_size}
    add('config/workspace.toml', b'# synthetic\n')
    add('code/exp169_partner_bundle_release.py', Path(v2.base.__file__).read_bytes())
    add('code/exp169_release_v2.py', Path(v2.__file__).read_bytes())
    for family, seeds in (('l2', (42, 43)), ('dino', (42,)), ('convnext', (42,))):
        for seed in seeds:
            for fold in range(5):
                name = f'models/{family}_s{seed}_f{fold}.pt'
                add(name, f'synthetic-{family}-{seed}-{fold}'.encode())
                tag = f'EXP172_DINO_BASE_L2_S42_F{fold}' if family == 'dino' else f'SYN_{family}_{seed}_{fold}'
                entry = dict(family=family, seed=seed, fold=fold, tag=tag, path=name,
                             source_sha256=files[name]['sha256'], **files[name])
                if family == 'dino':
                    entry['format'] = 'authenticated-original'
                entries.append(entry)
    _json(partner / 'release-manifest.json', dict(
        schema=v2.base.SCHEMA, generator_version=v2.VERSION, candidate='B172', k_eval=36,
        weights={'trio': .70, 'bundle': .30}, family_weights=dict(l2=1/3, dino=1/3, convnext=1/3),
        entries=entries, files=files, c4_manifest_sha256='c' * 64))
    scores = {f: rng.random((m, n, 12)) for f, m in (('l2', 10), ('dino', 5), ('convnext', 5))}
    np.savez(partner / 'pilot-scores.npz', ids=np.asarray(ids), **scores)
    _json(partner / 'partner-pilot.json', dict(
        status='PASS', gpus=2, studies=n, full_cohort=n, elapsed_seconds=pilot_elapsed,
        projected_full_seconds=projected, peak_vram_bytes=1024, peak_ram_bytes=2048,
        scores_path=str(partner / 'pilot-scores.npz'), scores_sha256=v4.sha(partner / 'pilot-scores.npz'),
        manifest_sha256=v4.sha(partner / 'release-manifest.json')))
    b42 = dict(variant='own', own_sha256=v4.S42_SHA, public_sha256=None)
    b174 = dict(variant='own', own_sha256='f' * 64, public_sha256=None)
    b174_path = tmp_path / 'code/b174-release-manifest.json'
    _json(b174_path, {'candidate': 'B174', 'anchors': {'EXP143V2': {'binding': b42}, 'EXP174': {'binding': b174}}})
    release_manifest = tmp_path / 'code/release-manifest.json'
    _json(release_manifest, {'schema': v4.SCHEMA, 'version': v4.VERSION})
    s43_assets = tmp_path / 's43'
    s43_assets.mkdir()
    (s43_assets / v4.S43_NAME).write_bytes(b'synthetic-s43')
    s43m = dict(schema='exp169-s43-release-v4', seed=43, checkpoint_sha256=v4.sha(s43_assets / v4.S43_NAME),
                serving_sha256='a' * 64, source_sha256='b' * 64)
    _json(s43_assets / 'release-manifest.json', s43m)
    b43 = dict(variant='own', own_sha256=s43m['checkpoint_sha256'], serving_sha256=s43m['serving_sha256'],
               source_sha256=s43m['source_sha256'], public_sha256=None)
    raw = {'S42': _anchor(tmp_path / 'w42', ids, sample, ['own'], b42, anchors[0], rng)['own'],
           'S43': _anchor(tmp_path / 'w43', ids, sample, ['own'], b43, anchors[1], rng)['own']}
    kwargs = dict(b174=b174_path, release_manifest=release_manifest)
    if candidate == 'A4':
        trio = _anchor(tmp_path / 'wtrio', ids, sample, ['own', 'public', 'exp143'], {'variant': 'blend3'},
                       anchors[2], rng)
        raw.update(public=trio['public'], exp155=trio['own'])
        kwargs['work_trio'] = tmp_path / 'wtrio'
    else:
        raw['exp174'] = _anchor(tmp_path / 'w174', ids, sample, ['own'], b174, anchors[2], rng)['own']
        kwargs['work174'] = tmp_path / 'w174'
    clock = Clock(sum(anchors) + pilot_elapsed)
    calls = []

    def fake_partner(work, competition_dir, cohort, manifest, deadline):
        calls.append(deadline)
        clock.t += partner_seconds
        return copy.deepcopy(scores), dict(gpu_peak_bytes=[1024, 1024], ram={'maxrss_kib': 1})

    def run(**overrides):
        args = dict(kwargs, clock=clock, started=0.)
        args.update(overrides)
        return v4.serve(partner, s43_assets, tmp_path / 'w42', tmp_path / 'w43', output, competition,
                        candidate, **args)
    return dict(run=run, clock=clock, calls=calls, fake_partner=fake_partner, output=output, ids=ids,
                raw=raw, scores=scores, release_manifest=release_manifest, partner=partner)


@pytest.fixture(params=['A4', 'B4'])
def served(request, tmp_path, monkeypatch):
    s = _release(tmp_path, request.param)
    monkeypatch.setattr(v4, 'partner_scores_c4', s['fake_partner'])
    s['candidate'] = request.param
    return s


def test_feasible_hidden_run_is_not_rejected_by_projection(served):
    # Anchors 10,000 s + pilot 500 s; projection 24,975 s > 19,500 s remaining;
    # the actual partner run (12,500 s) ends at 23,000 s.
    result = served['run']()
    out = served['output']
    assert result['status'] == 'PASS_BUNDLE' and result['candidate'] == served['candidate']
    assert served['clock'].t == 23000. and served['calls'] == [float('inf')]
    receipt = v4.read_json(out / v4.RECEIPT)
    assert receipt['timing_policy']['projection'] == 'diagnostic-only'
    assert receipt['timing_policy']['projected_full_seconds'] == 24975.
    assert receipt['timing_policy']['wall_clock_seconds'] == 23000.
    assert receipt['submission_sha256'] == v4.sha(out / 'submission.csv')
    assert not (out / v4.STAGED_CSV).exists()


def test_removed_guard_cannot_reject_in_serving_source():
    source = inspect.getsource(v4.serve)
    assert 'GUARD_SECONDS' not in source
    assert 'wall_elapsed' not in source
    assert 'wall-clock guard' not in source
    assert "deadline = float('inf')" in source


@pytest.mark.parametrize('candidate', ['A4', 'B4'])
def test_29000_second_run_publishes(tmp_path, monkeypatch, candidate):
    s = _release(tmp_path, candidate, partner_seconds=18500.)
    monkeypatch.setattr(v4, 'partner_scores_c4', s['fake_partner'])
    result = s['run']()
    out = s['output']
    assert s['clock'].t == 29000. and s['calls'] == [float('inf')]
    assert result['status'] == 'PASS_BUNDLE'
    assert (out / 'submission.csv').is_file() and not (out / v4.STAGED_CSV).exists()
    assert v4.read_json(out / v4.RECEIPT)['submission_sha256'] == v4.sha(out / 'submission.csv')


@pytest.mark.parametrize('candidate', ['A4', 'B4'])
def test_32000_second_run_publishes(tmp_path, monkeypatch, candidate):
    s = _release(tmp_path, candidate, partner_seconds=21500.)
    monkeypatch.setattr(v4, 'partner_scores_c4', s['fake_partner'])
    result = s['run']()
    out = s['output']
    assert s['clock'].t == 32000. and s['calls'] == [float('inf')]
    assert result['status'] == 'PASS_BUNDLE'
    assert (out / 'submission.csv').exists() and not (out / v4.STAGED_CSV).exists()
    receipt = v4.read_json(out / v4.RECEIPT)
    assert receipt['timing_policy']['wall_clock_seconds'] == 32000.
    assert receipt['timing_policy']['internal_time_limit_seconds'] is None
    assert receipt['manifest_sha256'] == v4.sha(s['release_manifest'])
    assert receipt['submission_sha256'] == v4.sha(out / 'submission.csv')


def test_real_wall_clock_is_measured_but_never_gates(tmp_path, monkeypatch):
    # Accounted anchors + pilot = 10,500 s, but the notebook started 30,100 s ago.
    s = _release(tmp_path, 'A4')
    monkeypatch.setattr(v4, 'partner_scores_c4', s['fake_partner'])
    result = s['run'](started=s['clock'].t - 30100.)
    assert result['status'] == 'PASS_BUNDLE' and s['calls'] == [float('inf')]
    assert v4.read_json(s['output'] / v4.RECEIPT)['timing_policy']['wall_clock_seconds'] == 42600.
    assert (s['output'] / 'submission.csv').exists()


def test_successful_preexisting_output_is_never_deleted_or_rewritten(served):
    out = served['output']
    out.mkdir(parents=True)
    (out / 'submission.csv').write_text('previous successful submission\n', encoding='utf-8')
    previous = {'status': 'PASS_BUNDLE', 'submission_sha256': v4.sha(out / 'submission.csv')}
    _json(out / v4.RECEIPT, previous)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError, match='Stale v4 output'):
        served['run']()
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert served['calls'] == []


def test_preexisting_receipt_alone_is_kept_and_no_csv_is_created(served):
    out = served['output']
    out.mkdir(parents=True)
    _json(out / v4.RECEIPT, {'status': 'FAILED_BUNDLE', 'note': 'earlier run'})
    before = (out / v4.RECEIPT).read_bytes()
    with pytest.raises(ValueError, match='Stale v4 output'):
        served['run']()
    assert (out / v4.RECEIPT).read_bytes() == before
    assert not (out / 'submission.csv').exists()


def test_stale_partner_raw_is_rejected_before_the_partner_pass(served):
    # partner-raw.npz is written create-only after the partner pass (4.1.1 preflight).
    out = served['output']
    out.mkdir(parents=True)
    (out / 'partner-raw.npz').write_bytes(b'stale')
    with pytest.raises(ValueError, match='Stale v4 output: partner-raw.npz'):
        served['run']()
    assert served['calls'] == [] and (out / 'partner-raw.npz').read_bytes() == b'stale'
    assert not (out / 'submission.csv').exists()
    assert v4.read_json(out / v4.RECEIPT)['status'] == 'FAILED_BUNDLE'


def test_failure_after_staged_write_removes_only_own_staged_csv(served, monkeypatch):
    original = v2.base.write_csv

    def write_then_fail(path, ids, values, sample):
        assert Path(path).name == v4.STAGED_CSV  # the frozen body never writes the root CSV
        original(path, ids, values, sample)
        raise RuntimeError('synthetic staged-write failure')
    monkeypatch.setattr(v2.base, 'write_csv', write_then_fail)
    with pytest.raises(RuntimeError, match='synthetic staged-write failure'):
        served['run']()
    out = served['output']
    assert not (out / 'submission.csv').exists() and not (out / v4.STAGED_CSV).exists()
    assert v4.read_json(out / v4.RECEIPT)['status'] == 'FAILED_BUNDLE'


def test_pass_receipt_and_published_values_by_id(served):
    result = served['run']()
    out, raw, candidate = served['output'], served['raw'], served['candidate']
    receipt = v4.read_json(out / v4.RECEIPT)
    assert receipt == result
    assert receipt['generator_version'] == v4.VERSION and receipt['candidate'] == candidate
    assert receipt['manifest_sha256'] == v4.sha(served['release_manifest'])
    assert receipt['c4_manifest_sha256'] == 'c' * 64
    expected_arms = (['public', 'OWN_S42_S43', 'EXP155'] if candidate == 'A4' else ['OWN_S42_S43', 'EXP174'])
    assert receipt['executed_arms'] == expected_arms + [
        'L2_RADR50_10', 'EXP172_L2_DINO_BASE_5', 'EXP179_183_L2_CONVNEXT_T_5']
    assert set(receipt['anchors']) == {'S42', 'S43', 'public_EXP155' if candidate == 'A4' else 'EXP174'}
    assert receipt['backend_reset'] == {'benchmark': False, 'deterministic': True}
    q = v4.q4_values(served['scores'])
    others = (dict(public=raw['public'], exp155=raw['exp155']) if candidate == 'A4' else
              dict(exp174=raw['exp174']))
    expected = v4.final_values(candidate, raw['S42'], raw['S43'], q, **others)
    ids, values = _read_csv(out / 'submission.csv')
    assert ids == served['ids']
    assert values == [[format(float(x), '.10g') for x in row] for row in expected]


def test_certified_output_requires_matching_pass_receipt(tmp_path):
    (tmp_path / 'submission.csv').write_text('x\n', encoding='utf-8')
    assert not v4.certified_output(tmp_path)
    _json(tmp_path / v4.RECEIPT, {'status': 'PASS_BUNDLE', 'submission_sha256': 'wrong'})
    assert not v4.certified_output(tmp_path)
    _json(tmp_path / v4.RECEIPT, {'status': 'PASS_BUNDLE', 'submission_sha256': v4.sha(tmp_path / 'submission.csv')})
    assert v4.certified_output(tmp_path)


# ------------------------------------------- authenticated raw anchors, IDs
@pytest.fixture
def anchor(tmp_path):
    ids = ['a', 'b', 'c', 'd']
    sample = tmp_path / 'competition/sample_submission.csv'
    _csv(sample, ids, np.zeros((4, 12)))
    binding = {'variant': 'own', 'own_sha256': 'x' * 64}
    arrays = _anchor(tmp_path / 'anchor', ids, sample, ['own'], binding, 5., np.random.default_rng(3))
    return tmp_path / 'anchor', sample, ids, binding, arrays['own']


def _rewrite(work, **arrays):
    with np.load(work / 'raw.npz', allow_pickle=False) as raw:
        values = {k: raw[k] for k in raw.files}
    values.update(arrays)
    (work / 'raw.npz').unlink()
    np.savez(work / 'raw.npz', **values)
    receipt = v4.read_json(work / 'receipt.json')
    receipt['files']['raw.npz'] = v4.sha(work / 'raw.npz')
    _json(work / 'receipt.json', receipt)


def test_authenticated_raw_accepts_exact_ids(anchor):
    work, sample, ids, binding, own = anchor
    got_ids, values, seconds = v4.authenticated_raw(work, sample, {'own'}, binding)
    assert got_ids == ids and seconds == 5. and np.array_equal(values['own'], own)


@pytest.mark.parametrize('bad', ['reordered_raw', 'duplicate_raw', 'reversed_targets', 'hash_drift',
                                 'binding', 'probability', 'csv_order', 'incomplete'])
def test_authenticated_raw_fails_loudly(anchor, bad):
    work, sample, ids, binding, own = anchor
    if bad == 'reordered_raw':
        _rewrite(work, ids=np.asarray(ids[::-1]))
    elif bad == 'duplicate_raw':
        _rewrite(work, ids=np.asarray(['a', 'a', 'c', 'd']))
    elif bad == 'reversed_targets':
        _rewrite(work, targets=np.asarray(TARGETS[::-1]))
    elif bad == 'hash_drift':
        (work / 'inputs.json').write_text('{"changed": true}', encoding='utf-8')
    elif bad == 'binding':
        binding = dict(binding, own_sha256='y' * 64)
    elif bad == 'probability':
        _rewrite(work, own=np.full((4, 12), 1.5))
    elif bad == 'csv_order':
        _csv(sample, ids[::-1], np.zeros((4, 12)))
    else:
        receipt = v4.read_json(work / 'receipt.json')
        receipt['status'] = 'RUNNING'
        _json(work / 'receipt.json', receipt)
    with pytest.raises(ValueError):
        v4.authenticated_raw(work, sample, {'own'}, binding)


# --------------------------------------------- generated notebook envelope
def _run_cell(tmp_path, monkeypatch, candidate, pre=None):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # the cell sets it; restored after the test
    working, mounts = tmp_path / 'kaggle/working', tmp_path / 'kaggle/input'
    working.mkdir(parents=True)
    mounts.mkdir(parents=True)
    for name, content in (pre or {}).items():
        (working / name).write_bytes(content)
    source = _cell(STAGE, candidate)
    assert '_exp169_signal' not in source and 'SIGALRM' not in source and '.alarm(' not in source
    start = '    _exp169_started = _exp169_time.monotonic()\n'
    assert source.count(start) == 1
    source = source.replace(start, start + "    raise FileNotFoundError('synthetic: datasets are not mounted')\n")
    source = source.replace('/kaggle/working', working.as_posix()).replace('/kaggle/input', mounts.as_posix())
    ns = {'__name__': '__exp169_cell__'}
    with pytest.raises(FileNotFoundError, match='synthetic'):
        exec(compile(source, candidate + '-cell', 'exec'), ns)
    return working, ns


@pytest.mark.parametrize('candidate', ['A4', 'B4'])
def test_notebook_failure_keeps_preexisting_success_and_reports_v4(tmp_path, monkeypatch, candidate):
    csv_bytes = b'StudyInstanceUID\nprevious\n'
    receipt = json.dumps({'status': 'PASS_BUNDLE',
                          'submission_sha256': hashlib.sha256(csv_bytes).hexdigest()}).encode()
    working, _ = _run_cell(tmp_path, monkeypatch, candidate,
                           {'submission.csv': csv_bytes, v4.RECEIPT: receipt})
    assert (working / 'submission.csv').read_bytes() == csv_bytes
    assert (working / v4.RECEIPT).read_bytes() == receipt
    sidecars = list(working.glob('exp169-v4-failure-*.json'))
    assert len(sidecars) == 1
    info = v4.read_json(sidecars[0])
    manifest_sha = v4.sha(STAGE / 'code/assets/release-manifest.json')
    assert info['status'] == 'FAILED_BUNDLE' and info['candidate'] == candidate
    assert info['generator_version'] == v4.VERSION and info['manifest_sha256'] == manifest_sha
    assert info['verification_note'] == v4.failure_note(manifest_sha, v4.sha(STAGE / 'code/dataset' / v4.CODE_ARCHIVE))


@pytest.mark.parametrize('candidate', ['A4', 'B4'])
def test_notebook_failure_writes_v4_envelope_and_cleans_only_own_csv(tmp_path, monkeypatch, candidate):
    working, ns = _run_cell(tmp_path, monkeypatch, candidate)
    info = v4.read_json(working / v4.RECEIPT)
    assert info['status'] == 'FAILED_BUNDLE' and info['candidate'] == candidate
    assert info['manifest_sha256'] == v4.sha(STAGE / 'code/assets/release-manifest.json')
    assert not (working / 'submission.csv').exists() and not list(working.glob('*.tmp'))
    # The cell's own cleanup: an uncertified CSV created after the start is removed ...
    (working / 'submission.csv').write_text('uncertified\n', encoding='utf-8')
    ns['_exp169_cleanup']()
    assert not (working / 'submission.csv').exists()
    # ... while a certified one (PASS receipt bound to its hash) is kept.
    (working / 'submission.csv').write_text('certified\n', encoding='utf-8')
    _json(working / v4.RECEIPT, {'status': 'PASS_BUNDLE', 'submission_sha256': v4.sha(working / 'submission.csv')})
    ns['_exp169_cleanup']()
    assert (working / 'submission.csv').read_text(encoding='utf-8') == 'certified\n'


# ------------------------------------------------- frozen source overlays
def test_frozen_serving_source_overlays_still_compile():
    tree = ast.parse(textwrap.dedent(inspect.getsource(v4.partner_scores_c4)))
    replacements = sorted(
        (node.lineno, ast.literal_eval(node.args[1]), ast.literal_eval(node.args[2]))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'replace_once'
        and len(node.args) == 3
    )
    loader = inspect.getsource(v4.v2.base._load_models)
    loader = v4.v2.replace_once(loader, *replacements[0][1:])
    compile(loader, '<v4-c4-loader>', 'exec')
    partner = inspect.getsource(v4.v2.base.partner_scores)
    for _, old, new in replacements[1:]:
        partner = v4.v2.replace_once(partner, old, new)
    compile(partner, '<v4-c4-partner>', 'exec')

    tree = ast.parse(textwrap.dedent(inspect.getsource(v4.serve)))
    swaps = next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'swaps' for target in node.targets)
    )
    serving = inspect.getsource(v4.v2.base.serve)
    for old, new in ast.literal_eval(swaps):
        serving = v4.v2.replace_once(serving, old, new)
    assert serving.count("output / 'submission.csv'") == 5
    assert 'remaining = LIMIT_SECONDS' not in serving
    assert '<= remaining' not in serving and 'remaining > 0 and' not in serving
    assert 'clock() < deadline' not in serving
    assert 'trio_seconds + p[\'elapsed_seconds\'] + clock() - start <=' not in serving
    assert "deadline = float('inf')" in serving
    compile(serving.replace("output / 'submission.csv'", 'staged_csv'), '<v4-serve>', 'exec')


# ------------------------- real archives: overlay, C4 CPU load, notebook run
B172_ARCHIVE = RELEASES / 'EXP169_B172_v2_0_2_stage/dataset/exp169-assets.zip.bin'
B172_MANIFEST_SHA = '6c47cfc48f74d8422a1909ced52356826ccad7ad2e519fb3c57f4a44a10c61cd'
HARNESS = Path(__file__).resolve().parent / 'exp169_v4_notebook_harness.py'
C4_TAGS = ['EXP179_M42_F0'] + [f'EXP183_M42_F{fold}' for fold in range(1, 5)]
PARTNER_CODE_MODULES = ('raptor_common', 'raptor_data', 'raptor_runtime', 'raptor_radimagenet',
                        'raptor_dicom', 'exp160_base_cpu_pos', 'exp160_base_patch_mil')


@pytest.fixture(scope='module')
def overlaid(tmp_path_factory):
    """overlay_c4 executed on the real extracted B172 partner and C4 stage archives."""
    root = tmp_path_factory.mktemp('overlay-real')
    partner, c4 = root / 'exp169-partner', root / 'exp169-c4'
    try:
        with zipfile.ZipFile(B172_ARCHIVE) as z:
            names = sorted(z.namelist())
            z.extractall(partner)
        with zipfile.ZipFile(STAGE / 'c4/dataset' / v4.C4_ARCHIVE) as z:
            z.extractall(c4)
        preexisting_sha = v4.sha(partner / 'release-manifest.json')  # the file 4.1.0 tried to create
        before = v4.read_json(partner / 'release-manifest.json')
        code_manifest = v4.read_json(STAGE / 'code/assets/release-manifest.json')
        result = v4.overlay_c4(partner, c4, code_manifest['c4_manifest_sha256'])
        yield SimpleNamespace(partner=partner, names=names, before=before, result=result,
                              preexisting_sha=preexisting_sha, code_manifest=code_manifest)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_overlay_c4_runs_on_real_b172_and_c4_archives(overlaid):
    assert overlaid.preexisting_sha == B172_MANIFEST_SHA
    written = v4.read_json(overlaid.partner / 'release-manifest.json')
    c4m = v4.read_json(STAGE / 'c4/release-manifest.json')
    assert written == overlaid.result
    assert written['c4_manifest_sha256'] == overlaid.code_manifest['c4_manifest_sha256'] == \
        v4.sha(STAGE / 'c4/release-manifest.json')
    assert written['entries'][:15] == overlaid.before['entries'][:15]
    assert written['entries'][15:] == c4m['entries']
    c4_paths = {e['path'] for e in c4m['entries']}
    for e in c4m['entries']:
        assert written['files'][e['path']] == {'bytes': e['bytes'], 'sha256': e['sha256']}
        assert v4.sha(overlaid.partner / e['path']) == e['sha256']
    assert {k: v for k, v in written['files'].items() if k not in c4_paths} == \
        {k: v for k, v in overlaid.before['files'].items() if k not in c4_paths}
    assert {k: v for k, v in written.items() if k not in ('entries', 'files', 'c4_manifest_sha256')} == \
        {k: v for k, v in overlaid.before.items() if k not in ('entries', 'files')}
    assert v2.verify_assets(overlaid.partner, written)
    on_disk = sorted(p.relative_to(overlaid.partner).as_posix()
                     for p in overlaid.partner.rglob('*') if p.is_file())
    assert on_disk == overlaid.names  # no temporary or extra file left behind


@contextlib.contextmanager
def _partner_imports(code):
    """Import the packaged partner modules from ``code``; restore sys.path/modules after."""
    saved_path, saved_names = list(sys.path), set(sys.modules)
    shadowed = {n: sys.modules.pop(n) for n in PARTNER_CODE_MODULES if n in sys.modules}
    sys.path.insert(0, str(code))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_names:
            sys.modules.pop(name, None)
        sys.modules.update(shadowed)


def test_c4_checkpoints_load_strictly_through_the_patched_loader_on_cpu(overlaid):
    import torch
    tree = ast.parse(textwrap.dedent(inspect.getsource(v4.partner_scores_c4)))
    loader_swaps = [(ast.literal_eval(n.args[1]), ast.literal_eval(n.args[2])) for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                    n.func.attr == 'replace_once' and len(n.args) == 3 and
                    isinstance(n.args[0], ast.Name) and n.args[0].id == 'loader']
    assert len(loader_swaps) == 1
    loader = v2.replace_once(inspect.getsource(v2.base._load_models), *loader_swaps[0])
    scope = {**v2.base.__dict__, 'model_meta': v4.c4_model_meta}  # exactly as partner_scores_c4
    entries = v4.read_json(overlaid.partner / 'release-manifest.json')['entries'][15:]
    assert [e['tag'] for e in entries] == C4_TAGS
    with _partner_imports(overlaid.partner / 'code'):
        exec(compile(loader, '<v4-c4-loader>', 'exec'), scope)
        models = scope['_load_models'](overlaid.partner, entries, 'cpu')
        with torch.inference_mode():
            logits = models[0](torch.zeros((1, 4, 3, 224, 224)))
    assert len(models) == 5 and tuple(logits.shape) == (1, 12) and bool(torch.isfinite(logits).all())
    # The C4 metadata gate rejects a checkpoint that is not the registered one.
    ck = torch.load(overlaid.partner / entries[0]['path'], map_location='cpu', weights_only=False)
    with pytest.raises(ValueError, match='C4 mixed-crop checkpoint metadata differs'):
        v4.c4_model_meta(dict(ck, tag='EXP170_CONVNEXT_T_L2_S42_F0'), entries[0])
    with pytest.raises(ValueError, match='C4 mixed-crop checkpoint metadata differs'):
        v4.c4_model_meta(dict(ck, crop_policy='context'), entries[0])


@pytest.mark.parametrize('candidate', ['A4', 'B4'])
def test_generated_cell_runs_end_to_end_with_gpu_mocked(tmp_path, candidate):
    """The staged A4/B4 cell: real archives, unpack, overlay, S43 bind + CPU load, pilot, serve."""
    root = tmp_path / 'notebook'
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONUTF8='1')
    try:
        proc = subprocess.run([sys.executable, '-I', str(HARNESS), str(STAGE), str(RELEASES), str(root),
                               candidate, '4'], capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=1800, env=env, cwd=str(tmp_path))
        lines = [line for line in proc.stdout.splitlines() if line.startswith('HARNESS_RESULT ')]
        assert proc.returncode == 0 and lines, proc.stdout[-4000:] + proc.stderr[-4000:]
        r = json.loads(lines[-1][len('HARNESS_RESULT '):])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    code_manifest = v4.read_json(STAGE / 'code/assets/release-manifest.json')
    assert r['error'] is None, r['error']
    assert r['alarm'] == [] and r['counts']['main()'] == 3
    assert r['counts']['_exp169_signal'] == r['counts']['SIGALRM'] == r['counts']['.alarm('] == 0
    assert r['v4_version'] == v4.VERSION
    assert r['v4_module_file'].replace('\\', '/').startswith(r['working_root'] + '/exp169-v4-code/')
    assert r['v2_module_file'].replace('\\', '/').startswith(r['working_root'] + '/exp169-partner/code/')
    receipt = r['receipt']
    arms = (['public', 'OWN_S42_S43', 'EXP155'] if candidate == 'A4' else ['OWN_S42_S43', 'EXP174'])
    assert receipt['status'] == 'PASS_BUNDLE' and receipt['candidate'] == candidate
    assert receipt['generator_version'] == v4.VERSION
    assert receipt['manifest_sha256'] == v4.sha(STAGE / 'code/assets/release-manifest.json')
    assert receipt['c4_manifest_sha256'] == code_manifest['c4_manifest_sha256']
    assert receipt['executed_arms'] == arms + ['L2_RADR50_10', 'EXP172_L2_DINO_BASE_5', 'EXP179_183_L2_CONVNEXT_T_5']
    assert receipt['submission_sha256'] == r['csv_sha256']
    assert receipt['timing_policy']['projection'] == 'diagnostic-only'
    # overlay_c4 replaced the extracted B172 manifest in place, without leftovers
    assert receipt['partner_manifest_sha256'] == r['partner_manifest_sha256'] != B172_MANIFEST_SHA
    assert r['partner_dir_tmp'] == []
    pilot, full = r['partner_calls']
    for call in (pilot, full):
        assert call['n'] == 4 and call['c4_tags'] == C4_TAGS
        assert call['c4_manifest_sha256'] == code_manifest['c4_manifest_sha256']
        assert call['benchmark'] is False and call['deterministic'] is True
    assert pilot['deadline_after_start'] == full['deadline_after_start'] == float('inf')
    assert receipt['timing_policy']['wall_clock_seconds'] > 32000
    works = [(a['module'], a['variant'], a['work']) for a in r['anchors']]
    if candidate == 'A4':
        assert [w[1:] for w in works] == [('blend3', 'exp169-trio'), ('own', 'exp169-own42'),
                                          ('own', 'exp169-s43-own')]
    else:
        assert [w[1:] for w in works] == [('own', 'exp169-trio'), ('own', 'exp169-anchor174'),
                                          ('own', 'exp169-s43-own')]
    assert works[-1][0] == 'exp169_s43_own_runtime'
    assert r['s43_checked_model']['arm'] == v4.S43_NAME and r['s43_checked_model']['res'] == 320
    assert r['csv_header_ok'] and r['csv_ids_ok'] and r['csv_equals_independent'] and r['s43_differs_from_s42']
    assert 'submission.csv' in r['files'] and v4.STAGED_CSV not in r['files']
    assert not [f for f in r['files'] if f.endswith('.tmp') or f.startswith('exp169-v4-failure-')]
