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

from tools import c85r7_flipall_v1115 as b


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
    for wrapper, dependency, name in [('exp169_release_c85r7.py', 'exp169_r7_port.py', 'R7_PORT_SHA'),
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


@pytest.mark.parametrize('target', ['notebook', 'metadata', 'registration', 'receipt', 'title', 'datasets', 'machine', 'cell', 'notebook_metadata'])
def test_metadata_bootstrap_and_exposure_tampering_rejected(monkeypatch, target):
    real = b.read
    paths = {'notebook': b.OUT / 'kernel/release.ipynb', 'metadata': b.OUT / 'kernel/kernel-metadata.json',
             'registration': b.OUT / 'registration.json', 'receipt': b.OUT / 'build-receipt.json'}
    paths.update({k: paths['metadata'] for k in ('title', 'datasets', 'machine')})
    paths.update({k: paths['notebook'] for k in ('cell', 'notebook_metadata')})
    def altered(path):
        obj = copy.deepcopy(real(path))
        if path == paths[target]:
            if target == 'notebook':
                obj['cells'][-1]['source'].append('\nprint("changed launch")\n')
            elif target == 'metadata':
                obj['enable_internet'] = not obj['enable_internet']
            elif target == 'registration':
                obj['clean_oof'] = True
            elif target == 'title':
                obj['title'] = b.SLUG
            elif target == 'datasets':
                obj['dataset_sources'] = []
            elif target == 'machine':
                obj['machine_shape'] = 'different'
            elif target == 'cell':
                obj['cells'][0]['source'].append('changed cell')
            elif target == 'notebook_metadata':
                obj['metadata']['changed'] = True
            else:
                obj['qualified_for_submission'] = True
        return obj
    monkeypatch.setattr(b, 'read', altered)
    with pytest.raises(ValueError):
        b.validate()


@pytest.mark.parametrize('kind', ['random', 'ties', 'constant'])
@pytest.mark.parametrize('n', [1, 3, 301])
def test_formula_reference_all_twelve(frozen, n, kind):
    _, _, old, new = frozen
    values = np.random.default_rng(119).uniform(.01, .99, (5, n, 12))
    if kind == 'ties': values = np.floor(values * 5) / 5
    if kind == 'constant': values[:] = .25
    r, s, x, y, q = values
    ids = [f'study-{i}' for i in range(n)]
    actual = new['blend'](ids, *values)
    public = .5 * rank(r) + .5 * rank(s)
    expected = .85 * rank(public) + .15 * rank((17*rank(x)+17*rank(y)+12*rank(q))/46)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, new['reference'](*values))
    parent = old['blend'](ids, *values)
    np.testing.assert_allclose(actual - parent, .85 * (rank(public) - rank(.5*rank(r)+.5*rank(s))), atol=2e-16)


@pytest.mark.parametrize('real_visible', [False, True])
@pytest.mark.parametrize('bits', list(itertools.product([False, True], repeat=5)))
def test_all_member_failure_combinations(frozen, bits, real_visible):
    _, _, old, new = frozen
    ids = ['one', 'two', 'three', 'four']
    values = np.random.default_rng(54).uniform(.01, .99, (5, 4, 12))
    if real_visible:
        from tools import exp169_release_c85r7codec as c
        ids, final = c.v5q._read_csv(b.VISIBLE / 'submission.csv')
        inputs = []
        for name in b.VISIBLE_INPUTS:
            other, v = c.v5q._read_csv(b.VISIBLE / name)
            inputs.append(v[[other.index(uid) for uid in ids]])
        r, s = inputs
        own = b.recover_own_rank(final, rank(.5*rank(r)+.5*rank(s)))
        values = np.array([r, s, own, own, own])
    names = ['R', 'S6', 'EXP143v2', 'EXP155', 'Q_A5q']
    failures = {name: {ids[0]: 'injected failure'} for name, bit in zip(names, bits) if bit}
    actual = new['blend'](ids, *values, failures)
    masks = np.zeros((5, len(ids)), dtype=bool)
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
    rf, sf = masks[:2]
    public = .5*ranked[0] + .5*ranked[1]
    # Independent row-wise pandas reference for all survivor/failure cases.
    for i in range(len(ids)):
        if rf[i] and sf[i]: public[i] = .5
        elif rf[i]: public[i] = ranked[1][i]
        elif sf[i]: public[i] = ranked[0][i]
    pf = rf & sf
    pr = np.zeros_like(public)
    pr[~pf] = rank(public[~pf])
    expected = .85*pr + .15*oo
    for i in range(len(ids)):
        if pf[i] and of[i]: expected[i] = .5
        elif pf[i]: expected[i] = oo[i]
        elif of[i]: expected[i] = pr[i]
    np.testing.assert_array_equal(actual, expected)
    parent = old['blend'](ids, *values, failures)
    np.testing.assert_array_equal(actual[pf], parent[pf])


def test_launch_functions_and_original_public_diagnostic_are_identical(frozen):
    old, new, _, _ = frozen
    # Independent audit of every function definition: no launch/worker/CLI edits.
    for name in old:
        if not name.endswith('.py'):
            continue
        before = {n.name: ast.get_source_segment(old[name].decode(), n)
                  for n in ast.parse(old[name]).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        after = {n.name: ast.get_source_segment(new[name].decode(), n)
                 for n in ast.parse(new[name]).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        assert before.keys() == after.keys()
        allowed = {'worker_args'} if name == 'exp169_r7_port.py' else (
            {'candidate_receipt'} if name == 'exp169_release_c85r7codec.py' else set())
        for function in before.keys() - allowed:
            assert before[function] == after[function], (name, function)
    meta = b.read(b.OUT / 'kernel/kernel-metadata.json')
    assert meta['id'] == 'taeyangg4/knee-lb-flipall-v1'
    assert meta['title'] == 'knee-lb-flipall-v1'



def test_r_sentinel_and_old_behavior_regression(frozen):
    _, _, old, new = frozen
    values = np.random.default_rng(19).uniform(.01, .99, (5, 9, 12))
    ids = [str(i) for i in range(9)]
    assert np.array_equal(old['blend'](ids, *values)[:, [7, 9]], new['blend'](ids, *values)[:, [7, 9]])
    values[0, 0] = .5
    implicit = new['blend'](ids, *values)
    explicit = new['blend'](ids, *values, {'R': {'0': 'sentinel'}})
    np.testing.assert_array_equal(implicit, explicit)


def test_real_visible_replay():
    result = b.replay()
    assert result['studies'] == 3 and result['independent_pandas_exact']
    assert result['baseline_reproduced_byte_for_byte'] and result['outer_and_own_unchanged']
    assert result['changed_cells'] == 0


@pytest.mark.parametrize('manifest_flag', [False, True])
def test_only_worker_arg_delta_is_one_flip_flag(frozen, manifest_flag):
    from pathlib import Path
    old, new, _, _ = frozen
    manifest = b.read(b.ROOT / 'storage/external/heliosli-r7cnxfsss-v1/manifest.json')
    assert len(manifest['models']) == 4
    manifest['flip_tta'] = manifest_flag
    results = []
    for files in (old, new):
        func = next(n for n in ast.parse(files['exp169_r7_port.py']).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'worker_args')
        ns = dict(Path=Path, require=b.need)
        exec(compile(ast.Module(body=[func], type_ignores=[]), '<frozen-worker-args>', 'exec'), ns)
        results.append(ns['worker_args'](manifest, Path('bundle'), Path('anatomy.pt')))
    assert results[1].count('--flip-tta') == 1
    assert [v for v in results[0] if v != '--flip-tta'] == [v for v in results[1] if v != '--flip-tta']
    assert ('--flip-tta' in results[0]) == manifest_flag
    # All four checkpoints and the worker implementing MCL exclusion are unchanged.
    worker = next(k for k in new if k.endswith('/compact_infer.py'))
    assert old[worker] == new[worker]
    assert b'for m in models:' in new[worker]


def test_real_pixel_all_fold_cpu_smoke_receipt():
    report = b.read(b.SCRATCH / 'flip-cpu-smoke.json')
    assert report['status'] == 'PASS' and report['device'] == 'cpu'
    assert report['studies'] == 3 and report['decoded_dicoms'] == 9
    assert len(report['folds']) == 4
    assert all(f['plain_mcl_exact'] and f['state_mcl_exact'] and f['non_mcl_changed'] > 0
               for f in report['folds'])
    for path, sha in report['input_sha256'].items():
        b.pin(b.ROOT / path, sha)


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
    assert b.admit()['status'] == 'HOLD_REVIEW_TIMING_AND_KAGGLE_VISIBLE'
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
assert receipt['version'] == '1.1.15-flipall'
assert receipt['exposure'] == 'EXPOSED' and receipt['clean_oof'] is False
assert receipt['qualified_for_submission'] is False
assert receipt['formula'] == parent['formula']
print(json.dumps({'status':'PASS', 'isolated_candidate_import':True, 'real3_csv_sha256':hashlib.sha256(payload).hexdigest(), 'receipt_metadata':'PASS'}))
'''
    run = subprocess.run([sys.executable, '-X', 'faulthandler', '-B', '-I', '-c', code,
                          str(b.SCRATCH / 'candidate-runtime'), str(b.VISIBLE), str(b.SCRATCH / 'replay.csv')],
                         cwd=b.SCRATCH, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report['status'] == 'PASS'
    b.write(b.SCRATCH / 'runtime-import-smoke.json', b.encoded(report))
