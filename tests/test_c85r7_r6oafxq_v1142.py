"""H-B-193: column truth table, admitted-hunk identity, reference proof, archive scope."""
import ast
import itertools
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tools import c85r7_r6oafxq_v1142 as b


@pytest.fixture(scope='module')
def candidate():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    return b.math_functions(b.payload()[2])


@pytest.fixture(scope='module')
def parent_math():
    return b.math_functions(b.parent()[1])


@pytest.fixture(scope='module')
def fx_math():
    return b.math_functions(b.authenticated(b.FX)[1])


def inputs(n=1200):
    rng = np.random.default_rng(193)
    return [f's{i}' for i in range(n)], [rng.integers(1, 49, (n, 12)) / 50 for _ in range(6)]


def failures(ids, masks):
    return {name: dict.fromkeys([u for u, f in zip(ids, m) if f], 'fixture') for name, m in zip(b.MEMBERS, masks)}


def all64(n):
    masks = [np.zeros(n, bool) for _ in range(6)]
    for row, bits in enumerate(itertools.product((False, True), repeat=6)):
        for mask, bit in zip(masks, bits):
            mask[row] = bit
    return masks


def contiguous(a, cols):
    return np.ascontiguousarray(a[:, list(cols)]).tobytes()


@pytest.mark.parametrize('case', ['healthy', 'all64', 'constant', 'own_failed', 'r384_failed', 'compact_failed', 'both_r_failed'])
def test_independent_oracle_protected_eleven_and_outer_failures(candidate, parent_math, case):
    ids, arrays = inputs()
    masks = [np.zeros(len(ids), bool) for _ in arrays]
    if case == 'all64':
        masks = all64(len(ids))
    elif case == 'constant':
        for a in arrays:
            a[:] = .2
    elif case == 'own_failed':
        for m in masks[3:]:
            m[:] = True
    elif case == 'r384_failed':
        masks[0][:] = True
    elif case == 'compact_failed':
        masks[2][:] = True
    elif case == 'both_r_failed':
        masks[0][:30] = True; masks[1][:30] = True
    actual = candidate['blend'](ids, *arrays, failures(ids, masks))
    assert actual.tobytes() == b.oracle(arrays, masks).tobytes()
    prior = parent_math['blend'](ids, *arrays, failures(ids, masks))
    assert contiguous(actual, b.PROTECTED) == contiguous(prior, b.PROTECTED)
    assert np.isfinite(actual).all() and np.all((actual >= 0) & (actual <= 1))
    if case in ('healthy', 'constant'):
        np.testing.assert_array_equal(actual, candidate['reference'](*arrays))
    if case == 'all64':
        assert np.all(actual[63] == .5)
        assert np.sum(masks[1]) == 32 and 32 / len(ids) < .03
    if case == 'healthy':
        assert np.any(actual[:, 11] != prior[:, 11])


@pytest.mark.parametrize('compact', ['same', 'distinct', 'distinct_failed'])
def test_fracture_equals_admitted_h187_rule(candidate, fx_math, compact):
    ids, arrays = inputs()
    masks = all64(len(ids))
    actual = candidate['blend'](ids, *arrays, failures(ids, masks))
    other = [a.copy() for a in arrays]
    other_masks = [m.copy() for m in masks]
    if compact != 'same':
        other[2] = 1 - arrays[2]
    if compact == 'distinct_failed':
        other_masks[2][:] = True
    reference = fx_math['blend'](ids, *other, failures(ids, other_masks))
    assert contiguous(actual, b.ROUTED) == contiguous(reference, b.ROUTED)


def test_eight_public_truth_rows_and_unchanged_other_columns(candidate, parent_math):
    ids, arrays = inputs(400)
    masks = [np.zeros(len(ids), bool) for _ in range(3)]
    bits = list(itertools.product((False, True), repeat=3))
    for row, combination in enumerate(bits):
        for mask, bit in zip(masks, combination):
            mask[row] = bit

    def rank(a, m):
        z = np.zeros(a.shape)
        z[~m] = pd.DataFrame(a[~m]).rank(method='average', pct=True).to_numpy()
        return z
    ar, br, _ = [rank(a, m) for a, m in zip(arrays, masks)]
    public = candidate['p_branch'](*arrays[:3], *masks)
    for row, (rf, f224, _) in enumerate(bits):
        if rf and f224: expected = np.full(12, .5)
        elif rf: expected = br[row]
        elif f224: expected = ar[row]
        else: expected = .65 * ar[row] + .35 * br[row]
        np.testing.assert_array_equal(public[row, 11], expected[11])
    prior = parent_math['p_branch'](*arrays[:3], *masks)
    assert contiguous(public, b.PROTECTED) == contiguous(prior, b.PROTECTED)
    changed = [a.copy() for a in arrays]
    changed[2] = 1 - changed[2]
    np.testing.assert_array_equal(public[:, 11], candidate['p_branch'](*changed[:3], *masks)[:, 11])


def test_fracture_ignores_compact_but_retains_outer_own(candidate):
    ids, arrays = inputs(100)
    actual = candidate['blend'](ids, *arrays)
    changed = [a.copy() for a in arrays]
    changed[2] = changed[2][::-1].copy()
    alternate = candidate['blend'](ids, *changed, {'S_R7': dict.fromkeys(ids, 'fixture')})
    np.testing.assert_array_equal(actual[:, 11], alternate[:, 11])
    rank = lambda a: pd.DataFrame(a).rank(method='average', pct=True).to_numpy()
    rr = rank(.65 * rank(arrays[0]) + .35 * rank(arrays[1]))
    own = rank((17 * rank(arrays[3]) + 17 * rank(arrays[4]) + 12 * rank(arrays[5])) / 46)
    np.testing.assert_array_equal(actual[:, 11], (.85 * rr + .15 * own)[:, 11])
    assert np.max(np.abs(actual[:, 11] - rr[:, 11])) > .01


def test_no_fluid_route_and_oa_route_kept(candidate, parent_math, fx_math):
    ids, arrays = inputs(300)
    actual = candidate['blend'](ids, *arrays)
    np.testing.assert_array_equal(actual[:, [7, 9]], parent_math['blend'](ids, *arrays)[:, [7, 9]])
    assert np.any(actual[:, [7, 9]] != fx_math['blend'](ids, *arrays)[:, [7, 9]])
    rank = lambda a: pd.DataFrame(a).rank(method='average', pct=True).to_numpy()
    own = rank((17 * rank(arrays[3]) + 17 * rank(arrays[4]) + 12 * rank(arrays[5])) / 46)
    np.testing.assert_array_equal(actual[:, [5, 6]], (.85 * rank(arrays[2]) + .15 * own)[:, [5, 6]])


def test_r224_three_percent_abort_unchanged(candidate):
    ids, arrays = inputs(100)
    arrays[1][:3] = .5
    candidate['blend'](ids, *arrays)
    arrays[1][3] = .5
    with pytest.raises(ValueError, match='more than 3%'):
        candidate['blend'](ids, *arrays)
    with pytest.raises(ValueError):
        candidate['failure_gate'](np.array([], bool))


def test_sentinel_masks_match_explicit_failures(candidate):
    ids, arrays = inputs(100)
    arrays[0][0] = .5; arrays[1][0] = .5
    arrays[0][1] = .5; arrays[1][2] = .5
    masks = [np.zeros(100, bool) for _ in arrays]
    masks[0][[0, 1]] = True; masks[1][[0, 2]] = True
    actual = candidate['blend'](ids, *arrays)
    assert actual.tobytes() == b.oracle(arrays, masks).tobytes()
    np.testing.assert_array_equal(actual, candidate['blend'](ids, *arrays, failures(ids, masks)))
    ids2, arrays2 = inputs(2); arrays2[1][0] = .5
    with pytest.raises(ValueError, match='more than 3%'):
        candidate['blend'](ids2, *arrays2)


@pytest.mark.parametrize('bad', ['nan', 'inf', 'negative', 'over_one', 'shape', 'duplicate'])
def test_invalid_rejected(candidate, bad):
    ids, arrays = inputs(100)
    if bad == 'nan': arrays[0][0, 11] = np.nan
    if bad == 'inf': arrays[1][0, 11] = np.inf
    if bad == 'negative': arrays[0][0, 11] = -.1
    if bad == 'over_one': arrays[1][0, 11] = 1.1
    if bad == 'shape': arrays[0] = arrays[0][:, :11]
    if bad == 'duplicate': ids[0] = ids[1]
    with pytest.raises(ValueError):
        candidate['blend'](ids, *arrays)


def test_all_embedded_files_and_function_scope():
    nb, meta, files, blob, manifest, full = b.artifacts()
    pnb, original, _, _ = b.parent()
    b.scope(nb, meta)
    assert len(manifest['unchanged']) == 29 and len(manifest['changed']) == 2
    assert not manifest['added'] and not manifest['removed']
    assert set(files) == set(original)
    for item in manifest['unchanged']:
        assert files[item['name']] == original[item['name']]
    assert nb['cells'][:-1] == pnb['cells'][:-1]
    for name in (b.HELPER, b.WRAPPER):
        before, after = ast.parse(original[name]), ast.parse(files[name])

        def functions(tree):
            return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        a, z = functions(before), functions(after)
        assert a.keys() == z.keys()
        assert {k for k in a if a[k] != z[k]} == ({'p_branch', 'blend', 'reference'} if name == b.HELPER else {'collect', 'runtime_pins'})
        assert [ast.dump(n) for n in before.body if isinstance(n, (ast.Import, ast.ImportFrom))] == \
               [ast.dump(n) for n in after.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for name in ('failure_gate', 'r_half', 'read', 'record'):
        assert b.function_bytes(original[b.HELPER], name) == b.function_bytes(files[b.HELPER], name)
    pm = b.old.read(b.BASE / 'kernel/kernel-metadata.json')
    assert {k: v for k, v in pm.items() if k not in ('id', 'title')} == {k: v for k, v in meta.items() if k not in ('id', 'title')}
    assert meta['id'] == b.SLUG and meta['title'] == 'knee-lb-r6oafxq-v1' and '/' not in meta['title']
    assert all(k.encode() in full for k in files)
    assert b.payload()[3] == blob


def test_helper_is_restricted_admitted_hunks():
    child = b.payload()[2][b.HELPER].decode()
    sib = b.authenticated(b.SIBLING)[1][b.HELPER].decode()
    fx = b.authenticated(b.FX)[1][b.HELPER].decode()
    strip = lambda text: ''.join(x for x in text.splitlines(True) if not x.startswith('FORMULA = '))
    assert strip(sib).replace('[7,9]', '[11]') == strip(child)
    assert '[7,9' not in child and child.count('[11]') == 6
    assert 'FORMULA = ' + repr(b.FORMULA) + '\n' in child
    for name in b.ROUTE_FUNCTIONS:
        mine = b.function_bytes(child.encode(), name).decode()
        for line in b.OA_LINES:
            mine = mine.replace(line, '')
        assert b.function_bytes(fx.encode(), name).decode().replace('[7,9,11]', '[11]') == mine
    parent = b.parent()[1][b.HELPER].decode()
    removed = strip(child)
    for hunk, anchor in zip(b.HUNKS, b.ANCHORS):
        assert hunk in removed
    assert strip(parent).count('[11]') == 0


def test_reference_proof_blocks_and_gate_validator():
    from tools import strict_variant_gate_r224_v145 as generic
    proof = b.old.read(b.OUT / b.PROOF)
    assert proof == b.reference_proof()
    assert set(proof) == {'blocks'} and len(proof['blocks']) == 6
    child = b.payload()[2][b.HELPER]
    stages = set()
    for block in proof['blocks']:
        assert set(block) == {'reference_stage', 'reference_member', 'child_member', 'text'}
        assert block['reference_member'] == block['child_member'] == b.HELPER
        assert '[11]' not in block['text'] and '[7,9' not in block['text']
        ref = b.gate.kernel(b.ROOT / block['reference_stage'], b.gate.Inputs(b.ROOT))[1]['embedded/' + b.HELPER]
        assert ref.count(block['text'].encode()) == 1 and child.count(block['text'].encode()) == 1
        stages.add(block['reference_stage'])
    assert stages == {b.rel(b.FX['gate']), b.rel(b.SIBLING['gate'])}
    for spec in (b.FX, b.SIBLING):
        assert Path(b.old.read(spec['receipt'])['stage']).resolve() == spec['gate'].resolve()
    pins = b.gate.Inputs(b.ROOT)
    assert len(generic.reference_blocks(b.OUT / b.PROOF, b.gate.kernel(b.OUT, pins), pins)) == 6


def test_reference_proof_tamper_rejected_by_gate_validator(tmp_path):
    from tools import strict_variant_gate_r224_v145 as generic
    proof = b.reference_proof()
    proof['blocks'][1]['text'] = proof['blocks'][1]['text'].replace('.85 * rr', '.80 * rr')
    path = tmp_path / 'proof.json'; path.write_bytes(b.old.encoded(proof))
    pins = b.gate.Inputs(b.ROOT)
    with pytest.raises(ValueError, match='byte-for-byte'):
        generic.reference_blocks(path, b.gate.kernel(b.OUT, pins), pins)


def test_hunk_mutation_rejected(monkeypatch):
    monkeypatch.setattr(b, 'HUNKS', (b.HUNKS[0], b.HUNKS[1].replace('.15 * oo', '.20 * oo'), b.HUNKS[2]))
    with pytest.raises(ValueError, match='admitted H-B-185 hunk'):
        b.admitted_hunks()


def test_patch_refuses_parent_that_already_routes():
    with pytest.raises(ValueError, match='already routes'):
        b.patched(b.authenticated(b.SIBLING)[1])


@pytest.mark.parametrize('kind', ['bootstrap', 'metadata'])
def test_scope_mutation_fails_closed(kind):
    nb, meta, *_ = b.payload()
    if kind == 'bootstrap': nb['cells'][-1]['source'].append('\nprint("extra")\n')
    else: meta['enable_internet'] = not meta.get('enable_internet', False)
    with pytest.raises(ValueError, match='forbidden'):
        b.scope(nb, meta)


@pytest.mark.parametrize('value,result', [(None, 'HOLD'), (.955, 'DROP'), (.956, 'DROP'), (.957, 'RETAIN'), (.958, 'RETAIN'), (1, 'RETAIN'), (0, 'DROP')])
def test_registered_decision(value, result):
    assert b.decision(value) == result


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, -.1, 1.1, .9565, '0.957'])
def test_decision_rejects_invalid(bad):
    with pytest.raises(ValueError):
        b.decision(bad)


def test_registered_stage_identity(candidate):
    reg = b.old.read(b.OUT / 'registration.json')
    assert reg == b.registration() and reg['routed_columns'] == [11] and reg['added_route'] == [11]
    assert reg['protected_columns'] == list(range(11)) and reg['unchanged_oa_columns'] == [5, 6]
    assert reg['exposure'] == 'EXPOSED' and not reg['clean_oof'] and not reg['qualified_for_submission']
    assert reg['public_reading']['reference_value'] == .957 and reg['expected_public'] == .957 and reg['expected_public_upside'] == .958
    assert 'tie is an unresolved alternative' in reg['public_reading']['rule'] and 'DROP at <= 0.956' in reg['public_reading']['rule']
    assert reg['timing']['parent_hidden_hours_upper'] == 6.3 and reg['timing']['includes_queue'] and reg['timing']['inference_delta'] == 'none'
    assert reg['parent_submission'] == 57263371 and 'dataset_sources_delta' not in reg
    assert b.validate()['identical'] == 29
    assert candidate['p'].LABELS[11] == 'Fracture' and list(candidate['p'].LABELS[5:7]) == ['Lateral OA', 'PF OA']


def test_isolated_embedded_collect(tmp_path):
    """Smoke the unchanged collect wiring; R224 here is an explicit surrogate.

    Separate cpu-replay uses authenticated real R224/R6-flip/own data. This test
    checks collection/reference plumbing only; production must reject the S6 stand-in.
    """
    runtime = b.SCRATCH / 'candidate-runtime'
    script = r'''
import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
runtime=Path(sys.argv[1]); tmp=Path(sys.argv[2]); sys.path.insert(0,str(runtime))
import exp169_release_c85r7codec as c
import exp169_r224 as h
assert Path(c.__file__).resolve().parent==runtime.resolve() and c.r224 is h
assert all(not n.startswith('tools.') for n in sys.modules)
assert c.VERSION=='1.1.42-r6oafxq' and c.KERNEL_ID=='taeyangg4/knee-lb-r6oafxq-v1'
c.p.REPLAY=Path(sys.argv[3])
staged=c.stage_replay_input(tmp/'pr7')
production_manifest=c.r7.MANIFEST_SHA256
assert production_manifest=='6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648'
try:
    c.authenticated_pr7(staged)
except ValueError as error:
    assert str(error)=='S_R7 identity: manifest pin'
else:
    raise AssertionError('production must reject the S6 stand-in identity')
with patch.object(c.r7,'MANIFEST_SHA256','stand-in'):
    ids,r,ours,_=c.authenticated_pr7(staged)
work=staged/'pr7-run'; csvdir=staged/'csv'; csvdir.mkdir(exist_ok=True)
for name in (h.RAW,h.SUB):
    frame=pd.DataFrame(r,columns=c.p.LABELS); frame.insert(0,'StudyInstanceUID',ids)
    frame.to_csv(csvdir/name,index=False)
h.record(work,csvdir)
(staged/'R224-exit.json').write_text('{"exit_code":0}')
with patch.object(c.r7,'MANIFEST_SHA256','stand-in'):
    receipt,payload=c.collect(Path(sys.argv[4]),staged,tmp/'out')
assert c.r7.MANIFEST_SHA256==production_manifest
assert receipt['status']=='COMPLETE_C85R7' and receipt['telemetry']['healthy_reference_exact']
assert receipt['fallbacks']['r224_prediction']==0 and receipt['version']=='1.1.42-r6oafxq'
assert (tmp/'out/submission.csv').read_bytes()==payload
import torch
assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
print(json.dumps({'status':'PASS','studies':len(ids),'embedded_collect':True,'R224':'explicit R384 surrogate','compact':'recorded S6 stand-in; test-only manifest monkeypatch','production_identity_rejects_stand_in':True}))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    from tools import exp169_release_c85r7codec as original
    run = subprocess.run([sys.executable, '-B', '-I', '-c', script, str(runtime), str(tmp_path), str(original.p.REPLAY), str(original.c85.A5Q)],
                         cwd=tmp_path, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-7000:]
    assert '"embedded_collect": true' in run.stdout


def test_admit_rejects_stale_smoke(monkeypatch):
    original = b.old.read

    def read(path):
        if path == b.SCRATCH / 'replay.json':
            return {'status': 'PASS', 'kernel_sha256': 'stale'}
        return original(path)
    monkeypatch.setattr(b.old, 'read', read)
    with pytest.raises(ValueError, match='missing/stale'):
        b.admit()


@pytest.mark.parametrize('which', ['PARENT', 'FX', 'SIBLING'])
def test_reject_unpinned_receipts(monkeypatch, tmp_path, which):
    fake = tmp_path / 'receipt.json'; fake.write_text('{}')
    monkeypatch.setitem(getattr(b, which), 'receipt', fake)
    with pytest.raises(ValueError):
        b.authenticated(getattr(b, which))


def test_reject_wrong_gate_copy(monkeypatch):
    monkeypatch.setitem(b.FX, 'gate', b.SIBLING['gate'])
    with pytest.raises(ValueError, match='receipt stage'):
        b.authenticated(b.FX)


def test_pre_execution_rejects_spec_tamper(monkeypatch, tmp_path):
    path = tmp_path / 'SPEC.md'; path.write_bytes(b.SPEC.read_bytes() + b'changed')
    monkeypatch.setattr(b, 'SPEC', path)
    with pytest.raises(ValueError):
        b.pre_execution()


def test_r6_dataset_and_worker_identity():
    from tools import c85r7_r224r6oaq_v1134 as h181
    nb, meta, files, _ = b.payload()
    assert meta['dataset_sources'] == h181.PRIVATE_DATASETS
    original = b.parent()[1]
    for name in files:
        if name not in (b.HELPER, b.WRAPPER):
            assert files[name] == original[name]
    sib = b.authenticated(b.SIBLING)[1]
    for name in files:
        if name not in (b.HELPER, b.WRAPPER):
            assert files[name] == sib[name]
