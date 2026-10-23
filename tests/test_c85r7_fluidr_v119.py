"""One frozen candidate; tests never build alternate kernel stages or run inference."""
import ast
import copy
import itertools
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tools import c85r7_fluidr_v119 as b


@pytest.fixture(scope='module')
def frozen():
    _, old = b.parent()
    new = b.unpack(b.read(b.OUT / 'kernel/release.ipynb'))[0]
    return old, new, b.math_functions(old), b.math_functions(new)


def rank(x):
    return pd.DataFrame(x).rank(method='average', pct=True).to_numpy()


def test_complete_scope_and_pins(frozen):
    old, new, _, _ = frozen
    assert len(old) == len(new) == 29
    assert len(b.scope_check(old, new)) == 3
    assert sum(old[k] == new[k] for k in old) == 26
    assert b.validate()['status'] == 'PASS'
    for filename, data in new.items():
        if filename.endswith('.py'):
            compile(data, filename, 'exec')
    for wrapper, dependency, name in [('exp169_release_c85r7.py', 'exp169_release_c85.py', 'C85_SHA'),
                                       ('exp169_release_c85r7codec.py', 'exp169_release_c85r7.py', 'C85R7_SHA')]:
        tree = ast.parse(new[wrapper])
        value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)
        assert value == b.digest(new[dependency])


@pytest.mark.parametrize('kind', ['rename', 'argv', 'env', 'import', 'cli', 'arithmetic'])
def test_forbidden_changes_fail_closed(frozen, kind):
    old, new, _, _ = frozen
    bad = dict(new)
    if kind == 'rename':
        bad['renamed_runtime.py'] = bad.pop('exp169_release_c85r7codec.py')
    else:
        suffix = {'argv': b'\nARGV = ["different-worker"]\n', 'env': b'\nos.environ["PYTHONPATH"] = "/different"\n',
                  'import': b'\nimport changed_module\n', 'cli': b'\nCOMMAND = "public-worker-new"\n',
                  'arithmetic': b'\nWEIGHT = .86\n'}[kind]
        bad['exp169_release_c85r7codec.py'] += suffix
    with pytest.raises(ValueError, match='runtime'):
        b.scope_check(old, bad)


@pytest.mark.parametrize('target', ['notebook', 'metadata', 'registration', 'receipt'])
def test_metadata_bootstrap_and_exposure_tampering_rejected(monkeypatch, target):
    real = b.read
    paths = {'notebook': b.OUT / 'kernel/release.ipynb', 'metadata': b.OUT / 'kernel/kernel-metadata.json',
             'registration': b.OUT / 'registration.json', 'receipt': b.OUT / 'build-receipt.json'}
    def altered(path):
        obj = copy.deepcopy(real(path))
        if path == paths[target]:
            if target == 'notebook':
                obj['cells'][-1]['source'].append('\nprint("changed launch")\n')
            elif target == 'metadata':
                obj['enable_internet'] = not obj['enable_internet']
            elif target == 'registration':
                obj['clean_oof'] = True
            else:
                obj['qualified_for_submission'] = True
        return obj
    monkeypatch.setattr(b, 'read', altered)
    with pytest.raises(ValueError):
        b.validate()


@pytest.mark.parametrize('kind', ['random', 'ties', 'constant'])
@pytest.mark.parametrize('n', [1, 3, 301])
def test_formula_reference_and_other_ten_unchanged(frozen, n, kind):
    _, _, old, new = frozen
    values = np.random.default_rng(119).uniform(.01, .99, (5, n, 12))
    if kind == 'ties': values = np.floor(values * 5) / 5
    if kind == 'constant': values[:] = .25
    r, s, x, y, q = values
    ids = [f'study-{i}' for i in range(n)]
    actual = new['blend'](ids, *values)
    public = .5 * rank(r) + .5 * rank(s)
    public[:, [7, 9]] = rank(r)[:, [7, 9]]
    expected = .85 * rank(public) + .15 * rank((17*rank(x)+17*rank(y)+12*rank(q))/46)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, new['reference'](*values))
    keep = [j for j in range(12) if j not in (7, 9)]
    np.testing.assert_array_equal(actual[:, keep], old['blend'](ids, *values)[:, keep])
    values[1] = 1 - s
    np.testing.assert_array_equal(actual[:, [7, 9]], new['blend'](ids, *values)[:, [7, 9]])


@pytest.mark.parametrize('bits', list(itertools.product([False, True], repeat=5)))
def test_all_member_failure_combinations(frozen, bits):
    _, _, old, new = frozen
    ids = ['one', 'two', 'three', 'four']
    values = np.random.default_rng(54).uniform(.01, .99, (5, 4, 12))
    names = ['R', 'S6', 'EXP143v2', 'EXP155', 'Q_A5q']
    failures = {name: {'one': 'synthetic'} for name, bit in zip(names, bits) if bit}
    actual = new['blend'](ids, *values, failures)
    masks = np.zeros((5, 4), dtype=bool)
    masks[:, 0] = bits
    ranked = []
    for v, failed in zip(values, masks):
        a = np.zeros_like(v)
        a[~failed] = rank(v[~failed])
        ranked.append(a)
    denom = sum(w * (~f) for w, f in zip([17, 17, 12], masks[2:]))
    own = sum(w * v for w, v in zip([17, 17, 12], ranked[2:])) / np.maximum(denom, 1)[:, None]
    of = denom == 0
    oo = np.zeros_like(own)
    oo[~of] = rank(own[~of])
    expected = .85*ranked[0]+.15*oo
    for i in range(4):
        if masks[0, i] and of[i]: expected[i] = .5
        elif masks[0, i]: expected[i] = oo[i]
        elif of[i]: expected[i] = ranked[0][i]
    np.testing.assert_array_equal(actual[:, [7, 9]], expected[:, [7, 9]])
    keep = [j for j in range(12) if j not in (7, 9)]
    np.testing.assert_array_equal(actual[:, keep], old['blend'](ids, *values, failures)[:, keep])


def test_r_sentinel_and_old_behavior_regression(frozen):
    _, _, old, new = frozen
    values = np.random.default_rng(19).uniform(.01, .99, (5, 9, 12))
    ids = [str(i) for i in range(9)]
    assert not np.array_equal(old['blend'](ids, *values)[:, [7, 9]], new['blend'](ids, *values)[:, [7, 9]])
    values[0, 0] = .5
    implicit = new['blend'](ids, *values)
    explicit = new['blend'](ids, *values, {'R': {'0': 'sentinel'}})
    np.testing.assert_array_equal(implicit, explicit)


def test_real_visible_replay():
    result = b.replay()
    assert result['studies'] == 3 and result['independent_pandas_exact']
    assert result['baseline_reproduced_byte_for_byte'] and result['unchanged_other_10_targets']
    assert result['changed_fluid_cells'] > 0


def test_bad_or_partial_baseline_is_rejected(monkeypatch, tmp_path):
    (tmp_path / 'c85r7-receipt.json').write_text('{"version":"old","status":"PARTIAL"}')
    monkeypatch.setattr(b, 'VISIBLE', tmp_path)
    with pytest.raises(ValueError, match='hash mismatch'):
        b.replay()


@pytest.mark.parametrize('fault', ['partial', 'off_lattice'])
def test_rank_inversion_rejects_invalid_inputs(fault):
    public = np.tile(np.array([1/3, 2/3, 1])[:, None], (1, 12))
    final = public.copy()
    if fault == 'partial': final = final[:2]
    else: final[0, 0] += .001
    with pytest.raises(ValueError):
        b.recover_own_rank(final, public)


def test_admission_cannot_bypass_review(monkeypatch):
    assert b.admit()['status'] == 'HOLD_INDEPENDENT_REVIEW'
    assert b.admit()['qualified_for_submission'] is False
    monkeypatch.setattr(sys, 'argv', ['builder', 'admit'])
    with pytest.raises(ValueError, match='dry-run only'):
        b.main()


def test_unpacked_runtime_real3_and_receipt_in_isolated_cpu_process():
    code = '''
import sys, json, hashlib
from pathlib import Path
runtime, visible, replay = map(Path, sys.argv[1:])
sys.path.insert(0, str(runtime))
import exp169_release_c85r7codec as c
import numpy as np
import pandas as pd
for module in (c, c.base, c.c85):
    assert Path(module.__file__).parent == runtime
ids, final = c.v5q._read_csv(visible / 'submission.csv')
def aligned(name):
    other, values = c.v5q._read_csv(visible / name)
    return values[[other.index(uid) for uid in ids]]
def rank(x):
    return pd.DataFrame(x).rank(method='average', pct=True).to_numpy()
r = aligned('c85r7-run/pr7/pr7-run/r/submission.csv')
s = aligned('c85r7-run/pr7/pr7-run/s_r7/controller/submission.csv')
own = np.rint((final - .85*rank(.5*rank(r)+.5*rank(s))) / .15 * 6) / 6
result = c.blend(ids, r, s, own, own, own)
np.testing.assert_array_equal(result, c.reference(r, s, own, own, own))
payload = c.p.csv_bytes(ids, result)
assert payload == replay.read_bytes()
parent = json.loads((visible / 'c85r7-receipt.json').read_bytes())
receipt = c.candidate_receipt(ids, parent['a172'], parent['pr7'], payload, {'status':'CPU_ARITHMETIC_ONLY'})
assert receipt['version'] == '1.1.9-fluidR'
assert receipt['exposure'] == 'EXPOSED' and receipt['clean_oof'] is False
assert receipt['qualified_for_submission'] is False
assert 'P_t=r(R)_t' in receipt['formula']
print(json.dumps({'status':'PASS', 'isolated_candidate_import':True, 'real3_csv_sha256':hashlib.sha256(payload).hexdigest(), 'receipt_metadata':'PASS'}))
'''
    run = subprocess.run([sys.executable, '-X', 'faulthandler', '-B', '-I', '-c', code,
                          str(b.SCRATCH / 'candidate-runtime'), str(b.VISIBLE), str(b.SCRATCH / 'replay.csv')],
                         cwd=b.SCRATCH, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report['status'] == 'PASS'
    b.write(b.SCRATCH / 'runtime-import-smoke.json', b.encoded(report))
