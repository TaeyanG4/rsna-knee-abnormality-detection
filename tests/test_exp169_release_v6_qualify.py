"""Qualification gates, including receipt-consistent corruption of saved outputs."""
import json
import shutil
import sys

import numpy as np
import pytest

from tools import exp169_release_v6_qualify as q


@pytest.fixture(autouse=True)
def restore_frozen_imports():
    import tools
    names = ('exp169_partner_bundle_release', 'exp169_release_v2', 'exp152_own0949_release')
    modules = {n:sys.modules.get('tools.' + n) for n in names}
    attrs = {n:getattr(tools, n, None) for n in names}
    yield
    for name in names:
        if modules[name] is None:
            sys.modules.pop('tools.' + name, None)
        else:
            sys.modules['tools.' + name] = modules[name]
        if attrs[name] is None:
            if hasattr(tools, name):
                delattr(tools, name)
        else:
            setattr(tools, name, attrs[name])


@pytest.mark.parametrize('a,b,tol,status', [
    ([0.], [0.], dict(max_abs=0., mean_abs=0.), 'PASS'),
    ([0., 0.], [.00015, 0.], dict(max_abs=.00015, mean_abs=.00015), 'PASS'),
    ([0., 0.], [.000151, 0.], dict(max_abs=.00015, mean_abs=.00015), 'FAIL'),
    ([0., 0.], [.001, .001], dict(max_abs=.002, mean_abs=.0005), 'FAIL'),
])
def test_comparison_both_thresholds(a, b, tol, status):
    assert q.comparison_delta(a, b, tol)['status'] == status


@pytest.mark.parametrize('a,b', [([np.nan], [0.]), ([np.inf], [0.]), ([0.], [0.,0.]), ([],[])])
def test_comparison_rejects_invalid_arrays(a, b):
    with pytest.raises(ValueError):
        q.comparison_delta(a, b, dict(max_abs=1., mean_abs=1.))


def test_health_detects_individual_member_constant_and_nonfinite():
    a = np.random.default_rng(3).random((10, 3, 12))
    assert q.column_health(a)['status'] == 'PASS'
    a[7, :, 8] = .5
    assert q.column_health(a)['constant_columns'] == [[7, 8]]
    a[7, 0, 8] = np.nan
    assert not q.column_health(a)['finite']
    a[7, 0, 8] = 2.
    assert not q.column_health(a)['bounded']


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    source = q.ROOT / '.local/kout-v6-v121-A6'
    policy = q.ROOT / 'artifacts/experiments/EXP169_RELEASE_V6/qualifier-v1.2.2/compare-A6-v2/tolerances.json'
    if not (source / 'v6-run/receipt.json').is_file():
        pytest.skip('real Kaggle fixture unavailable')
    # This qualifier audits v1.2.1 and deliberately authenticates its source.
    # Use that immutable staged source even after the working release advances.
    import zipfile
    with zipfile.ZipFile(q.ROOT / q.v.EXP / 'v1.2.1/build-01/stage/runtime.zip') as archive:
        frozen_path = tmp_path / 'frozen_v121.py'
        frozen_path.write_bytes(archive.read('tools/exp169_release_v6.py'))
    frozen = q.v.load_module('qualifier_frozen_v121', frozen_path)
    frozen.ROOT = q.ROOT
    monkeypatch.setattr(q, 'v', frozen)
    dest = tmp_path / 'download'
    # Only the named prediction/metadata files used by the comparison, no weights.
    for pattern in ('submission.csv', 'v6-code/registration.json', 'v6-run/*.csv',
                    'v6-run/*.npz', 'v6-run/*.npz.json', 'v6-run/receipt.json',
                    'v6-run/partner/release-manifest.json',
                    'v6-run/s6/controller/submission_receipt.json'):
        for path in source.glob(pattern):
            target = dest / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return dest, tmp_path / 'audit', policy


def audit(saved_run):
    download, out, policy = saved_run
    return q.compare_kaggle(download, out, tolerances=policy)


def rewrite_receipt(download, key, path):
    receipt = download / 'v6-run/receipt.json'
    rc = q.v.read(receipt)
    if key == 'A6q':
        rc['candidates'][key]['sha256'] = q.v.sha(path)
    else:
        rc[key] = q.v.sha(path)
    receipt.write_text(json.dumps(rc), encoding='utf-8')


def test_real_comparison_completes_all_gates(saved_run):
    result = audit(saved_run)
    # Numerical identity may legitimately FAIL: never bake a PASS into the fixture.
    assert not any(c['name'] == 'structural-validation' for c in result['checks'])
    assert len([c for c in result['checks'] if c['name'].startswith('candidate/')]) == 6
    assert len([c for c in result['checks'] if c['name'].startswith('member/mil/')]) == 10
    assert result['timing']['projection_1300_hours'] > 0
    assert result['checks'][-1]['status'] == 'PASS'


def test_rejects_candidate_tamper_even_with_updated_receipt(saved_run):
    path = saved_run[0] / 'v6-run/A6q.csv'
    lines = path.read_text().splitlines()
    cells = lines[1].split(',')
    value = float(cells[1])
    cells[1] = str(value + .001 if value < .5 else value - .001)
    lines[1] = ','.join(cells)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    rewrite_receipt(saved_run[0], 'A6q', path)
    result = audit(saved_run)
    assert result['status'] == 'FAIL'
    assert next(c for c in result['checks'] if c['name'] == 'candidate/A6q')['status'] == 'FAIL'


@pytest.mark.parametrize('damage', ['order', 'constant', 'nan', 'missing_member'])
def test_rejects_bad_members(saved_run, damage):
    path = saved_run[0] / 'v6-run/members.npz'
    with np.load(path, allow_pickle=False) as raw:
        arrays = {k:raw[k].copy() for k in raw.files}
    if damage == 'order':
        arrays['ids'] = arrays['ids'][::-1]
    elif damage == 'constant':
        arrays['mil'][4, :, 2] = .5
    elif damage == 'nan':
        arrays['mil'][4, 0, 2] = np.nan
    else:
        del arrays['mil']
    np.savez(path, **arrays)
    rewrite_receipt(saved_run[0], 'members_sha256', path)
    result = audit(saved_run)
    assert result['status'] == 'FAIL'
    if damage == 'constant':
        assert next(c for c in result['checks'] if c['name'] == 'mil/health')['status'] == 'FAIL'
    else:
        assert any(c['name'] == 'structural-validation' for c in result['checks'])


def test_cli_failure_exit(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['qualify', 'compare-kaggle', '--download', 'x',
                                    '--out', 'y', '--tolerances', 'z'])
    monkeypatch.setattr(q, 'compare_kaggle', lambda *a, **kw: dict(status='FAIL'))
    with pytest.raises(SystemExit) as error:
        q.main()
    assert error.value.code == 1
    assert 'FAIL' in capsys.readouterr().out


def test_existing_audit_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        q.compare_kaggle('absent', tmp_path, tolerances='absent')
