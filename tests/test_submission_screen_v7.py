"""Bounded CPU tests for actual C probe receipt layout plus manifest/CSV binding faults."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('submission_screen_v7_test', ROOT / 'tools/submission_screen_v7.py')
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)
SOURCE = ROOT / '.local/crosscheck-20261008/release-screen/production-v7-synthetic-receipt-copy'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def bundle(tmp_path):
    sd = json.loads((SOURCE / 'release/shared/work/result.json').read_text(encoding='utf-8'))
    fd = json.loads((SOURCE / 'release/combine/result.json').read_text(encoding='utf-8'))
    manifest = {'schema': sd['schema'], 'release': sd['release'], 'arms': [],
                'combination': fd.get('formula')}
    csv = (SOURCE / 'submission.csv').read_bytes()
    sample = tmp_path / 'sample.csv'
    sample.write_bytes(csv)
    for name, rec in sd['arms'].items():
        raw = tmp_path / 'out/release' / name / 'arm.csv'
        raw.parent.mkdir(parents=True)
        raw.write_bytes(csv)
        rec['submission_sha256'] = fd['arms'][name]['submission_sha256'] = v7.base.sha256_file(raw)
        manifest['arms'].append({'name': name, 'weight': rec['rank_weight'], 'k_eval': rec['k_eval'],
                                 'checkpoints': copy.deepcopy(rec['checkpoints'])})
    mp = tmp_path / 'kernel/release-manifest.json'
    write(mp, manifest)
    sd['release_manifest_sha256'] = fd['release_manifest_sha256'] = v7.base.sha256_file(mp)
    out = tmp_path / 'out'
    (out / 'submission.csv').write_bytes(csv)
    write(out / 'release/shared/work/result.json', sd)
    write(out / 'release/combine/result.json', fd)
    members = [dict(name=a['name'], arm=a['name'], weight=a['weight'], oof_declared_null=True,
                    declared_checkpoints=[c['sha256'] for c in a['checkpoints']], k_eval=a['k_eval']) for a in manifest['arms']]
    ctx = v7.Ctx(tmp_path, sample_submission=str(sample))
    return ctx, members, dict(out_dir=str(out), kernel_dir=str(mp.parent)), sd, fd


def test_c_actual_nested_shape_passes(bundle):
    ctx, members, release, _, _ = bundle
    eng = v7.engineering_section(ctx, members, release)
    assert eng['engineering_pass'] and not eng['problems']
    assert eng['checkpoint_fingerprint']
    assert all(a['binding'] == 'declared_match' for a in eng['arms'].values())
    assert v7.base.DEFAULTS['fingerprint_registry'] == 'artifacts/experiments/SUBMISSION_SCREEN/fingerprints.json'


@pytest.mark.parametrize('fault', ['manifest_missing', 'checkpoint_missing', 'final_k_missing', 'final_weights_missing', 'decode_missing', 'csv_missing'])
def test_missing_binding_holds(bundle, fault):
    ctx, members, release, sd, fd = bundle
    if fault == 'manifest_missing':
        Path(release['kernel_dir'], 'release-manifest.json').unlink()
    elif fault == 'csv_missing':
        Path(release['out_dir'], 'release/coat5/arm.csv').unlink()
    elif fault == 'final_k_missing':
        fd['arms']['coat5'].pop('k_eval')
        write(Path(release['out_dir']) / 'release/combine/result.json', fd)
    elif fault == 'final_weights_missing':
        fd.pop('arm_weights')
        write(Path(release['out_dir']) / 'release/combine/result.json', fd)
    elif fault == 'decode_missing':
        fd.pop('shared_decode')
        write(Path(release['out_dir']) / 'release/combine/result.json', fd)
    else:
        sd['arms']['coat5'].pop('checkpoints')
        write(Path(release['out_dir']) / 'release/shared/work/result.json', sd)
    eng = v7.engineering_section(ctx, members, release)
    assert eng['problems'] and eng['v7_adapter']['missing_bindings']
    assert eng['checkpoint_fingerprint'] is None
    assert not eng['hard_failures']  # original verdict policy: incomplete evidence -> HOLD


@pytest.mark.parametrize('fault', ['checkpoint', 'checkpoint_path', 'checkpoint_k', 'final_k', 'decode', 'formula', 'csv', 'weight', 'k', 'manifest_sha', 'arm_order'])
def test_tamper_blocks(bundle, fault):
    ctx, members, release, sd, fd = bundle
    if fault == 'checkpoint':
        sd['arms']['coat5']['checkpoints'][0]['sha256'] = '0'*64
    elif fault == 'checkpoint_path':
        sd['arms']['coat5']['checkpoints'][0]['path'] = 'other.pt'
    elif fault == 'checkpoint_k':
        sd['arms']['coat5']['checkpoints'][0]['k_eval_release'] = 24
    elif fault == 'final_k':
        fd['arms']['coat5']['k_eval'] = 24
    elif fault == 'decode':
        fd['shared_decode']['ordered_volumes_sha256'] = '0'*64
    elif fault == 'formula':
        fd['formula'] = 'wrong blend'
    elif fault == 'csv':
        Path(release['out_dir'], 'release/coat5/arm.csv').write_text('tampered', encoding='utf-8')
    elif fault == 'weight':
        fd['rank_weights']['coat5'] = .9
    elif fault == 'k':
        sd['arms']['coat5']['k_eval'] = 24
    elif fault == 'manifest_sha':
        sd['release_manifest_sha256'] = '0'*64
    else:
        fd['arm_order'].reverse()
    write(Path(release['out_dir']) / 'release/shared/work/result.json', sd)
    write(Path(release['out_dir']) / 'release/combine/result.json', fd)
    eng = v7.engineering_section(ctx, members, release)
    assert not eng['engineering_pass'] and eng['hard_failures']
    assert eng['checkpoint_fingerprint'] is None


def test_single_nested_arm_recognized(bundle):
    ctx, members, release, sd, _ = bundle
    name = 'coat5'
    sd['arms'] = {name: sd['arms'][name]}
    sd['arm_order'] = [name]
    sd['rank_weights'] = {name: 1.0}
    sd['arms'][name]['rank_weight_normalised'] = 1.0
    sd['submission_sha256'] = sd['arms'][name]['submission_sha256']
    mp = Path(release['kernel_dir'], 'release-manifest.json')
    manifest = json.loads(mp.read_text(encoding='utf-8'))
    manifest['arms'] = manifest['arms'][:1]
    write(mp, manifest)
    sd['release_manifest_sha256'] = v7.base.sha256_file(mp)
    Path(release['out_dir'], 'release/combine/result.json').unlink()
    write(Path(release['out_dir']) / 'release/shared/work/result.json', sd)
    eng = v7.engineering_section(ctx, members[:1], release)
    assert eng['engineering_pass'] and not eng['problems']


def test_v7_missing_shared_receipt_holds_even_with_final(bundle):
    ctx, members, release, _, _ = bundle
    Path(release['out_dir'], 'release/shared/work/result.json').unlink()
    eng = v7.engineering_section(ctx, members, release)
    assert eng['v7_adapter']['missing_bindings']
    assert eng['checkpoint_fingerprint'] is None
    assert eng['problems'] and not eng['hard_failures']

@pytest.fixture
def stage2(bundle):
    ctx, members, release, sd, fd = bundle
    out = Path(release['out_dir'])
    manifest_path = Path(release['kernel_dir']) / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['rank_weights'] = sd['rank_weights']
    own_manifest = manifest_path.with_name('exp126-own-manifest.json')
    write(own_manifest, manifest)
    write(manifest_path.with_name('execution-plan.json'),
          {'runtime': {'studies': 1300, 'rad_k': 36, 'total_seconds': 28000,
                       'margin_seconds': 4400, 'fits_one_hour_margin': True}})
    digest = v7.base.sha256_file(own_manifest)
    write(manifest_path.with_name('exp126-config.json'),
          {'manifest_sha256': digest, 'final_weights': [.7, .3], 'rad_k': 36,
           'own': 'coat5rad5', 'plane_flip_tta': False})
    ids = list(v7.base.pd.read_csv(ctx.path(ctx.cfg['sample_submission']), dtype=str)['StudyInstanceUID'])
    worker = dict(schema='exp126-anchor-blend-v8-worker', status='PASS', plane_flip_tta=False,
                  release_manifest_sha256=digest, ids_sha256=v7.base.uid_hash(ids),
                  studies=len(ids), scored_ids=ids, unscored={}, store={'trusted': True}, arms={})
    for arm in manifest['arms']:
        name = arm['name']
        raw = out / 'exp126/shared' / (name + '.csv')
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes((out / 'submission.csv').read_bytes())
        checks = copy.deepcopy(arm['checkpoints'])
        for c in checks: c['k_eval_release'] = arm['k_eval']
        worker['arms'][name] = dict(arm=arm, checkpoints=checks, csv_sha256=v7.base.sha256_file(raw), csv=raw.name)
    backup = out / 'exp126/anchor_submission.csv'
    backup.write_bytes((out / 'submission.csv').read_bytes())
    digest_csv = v7.base.sha256_file(backup)
    receipt = dict(schema='exp126-anchor-blend-v8', path='blended', final_weights=[.7, .3],
                   anchor={'ok': True, 'members': ['resgated_top3', 'global96_top3', 'd4_swa3', 'repairv1_top3'],
                           'audit_members': 4, 'audit_status': 'COMPLETE',
                           'family_reduction': 'rank_of_member_probability_mean', 'failure_events': []},
                   hook={'thread_finished': True, 'hook_errors': [], 'anchor_raised': [], 'dropped_ids': []},
                   anchor_sha256=digest_csv, release_manifest_sha256=digest, worker=worker,
                   own_weights=manifest['rank_weights'], final_sha256=digest_csv, blended_sha256=digest_csv,
                   runtime={'rad_k': 36, 'total_seconds': 28000, 'margin_seconds': 4400,
                            'fits_one_hour_margin': True})
    write(out / 'exp126/shared/receipt.json', worker)
    write(out / 'exp126/receipt.json', receipt)
    return ctx, members, release, receipt


def test_stage2_pass(stage2):
    ctx, members, release, receipt = stage2
    eng = v7.engineering_section(ctx, members, release)
    assert eng['engineering_pass'] and not eng['problems']
    assert eng['checkpoint_fingerprint'] is None
    assert v7.base.fingerprints_of(members, eng)['members'] is None


def test_stage2_own_oof_cannot_admit_anchor_blend(stage2, monkeypatch):
    ctx, members, release, _ = stage2
    monkeypatch.setattr(v7, '_original_oof', lambda *args: {
        'baseline': {}, 'available': True, 'eligibility': 'ELIGIBLE', 'filter': 'PASS'})
    monkeypatch.setattr(v7.base, 'lineage_report', lambda *args: {'status': 'PASS', 'gate_failures': []})
    result = v7.guarded_oof(ctx, members, {}, {}, {'release': release})
    assert result['available'] is False
    assert result['filter'] == 'NOT_AVAILABLE'
    assert 'anchor plus own blend' in result['eligibility']


@pytest.mark.parametrize('fault', ['anchor_gate', 'final_weights', 'final_hash', 'runtime', 'hidden_runtime',
                                    'checkpoint', 'tta_policy', 'runtime_k', 'hidden_runtime_k', 'config_k'])
def test_stage2_tamper_blocks(stage2, fault):
    ctx, members, release, receipt = stage2
    if fault == 'anchor_gate': receipt['anchor']['audit_members'] = 3
    elif fault == 'final_weights': receipt['final_weights'] = [.8, .2]
    elif fault == 'final_hash': receipt['final_sha256'] = '0'*64
    elif fault == 'runtime': receipt['runtime']['total_seconds'] = 30000
    elif fault == 'hidden_runtime':
        write(Path(release['kernel_dir']) / 'execution-plan.json',
          {'runtime': {'studies': 1300, 'rad_k': 36, 'total_seconds': 29116.5,
                           'margin_seconds': 3283.5, 'fits_one_hour_margin': False}})
    elif fault == 'runtime_k': receipt['runtime']['rad_k'] = 24
    elif fault == 'hidden_runtime_k':
        plan = Path(release['kernel_dir']) / 'execution-plan.json'
        data = json.loads(plan.read_text(encoding='utf-8'))
        data['runtime']['rad_k'] = 24
        write(plan, data)
    elif fault == 'config_k':
        path = Path(release['kernel_dir']) / 'exp126-config.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        data['rad_k'] = 24
        write(path, data)
    elif fault == 'tta_policy':
        receipt['worker']['plane_flip_tta'] = True
        write(Path(release['out_dir']) / 'exp126/shared/receipt.json', receipt['worker'])
    else:
        receipt['worker']['arms']['coat5']['checkpoints'][0]['sha256'] = '0'*64
        write(Path(release['out_dir']) / 'exp126/shared/receipt.json', receipt['worker'])
    write(Path(release['out_dir']) / 'exp126/receipt.json', receipt)
    eng = v7.engineering_section(ctx, members, release)
    assert not eng['engineering_pass'] and eng['hard_failures']


def test_stage2_missing_manifest_holds(stage2):
    ctx, members, release, _ = stage2
    Path(release['kernel_dir'], 'exp126-own-manifest.json').unlink()
    eng = v7.engineering_section(ctx, members, release)
    assert eng['problems'] and not eng['hard_failures']
    assert eng['checkpoint_fingerprint'] is None


def test_baseline_overlap_reaches_verdict(tmp_path):
    spec = importlib.util.spec_from_file_location('screen_fixture_v7', ROOT / 'tests/test_submission_screen.py')
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    w = fixture.World(tmp_path, n=30)
    bad_protocol = w.make_protocol('BAD_BASELINE_PROTOCOL', leak_fold=0)
    baseline = w.member('EXP099_B8_DREAD', protocol=bad_protocol)
    candidate = w.member('CANDIDATE')
    release = w.release('baseline_fault', {'CANDIDATE': (candidate, 1.0)})
    screen_spec = fixture.spec('baseline_fault', [candidate], release, basis='oof')
    ctx = v7.Ctx(tmp_path, draws=5)
    report = v7.base.screen(ctx, screen_spec)
    assert report['oof']['baseline']['lineage']['status'] == 'FAIL_DIRECT'
    assert report['verdict']['verdict'] == 'BLOCK'
    assert any('baseline lineage' in x for x in report['verdict']['blockers'])


