"""H-B-191: w_Effusion=w_Fracture=1.0 on admitted H-B-189; admitted fluid survivor rule for 7/11.

Test file v1.0.0 (claude-opus-b1). Synthetic oracle/failure/scope tests; the
real 2/58-study embedded replay lives in the builder's cpu-replay command.
"""
import ast
import itertools
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tools import c85r7_r6w958fxq_v1141 as b
from tools import c85r7_r6w958q_v1139 as h189


@pytest.fixture(scope='module')
def funcs():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    return dict(candidate=b.math_functions(b.payload()[2]), parent=b.math_functions(b.parent()[1]),
                h187=b.math_functions(b.reference('h187')[1]), h168=b.math_functions(b.reference('h168')[1]))


def inputs(n=1200, seed=191):
    rng = np.random.default_rng(seed)
    return [f's{i}' for i in range(n)], [rng.integers(1, 49, (n, 12))/50 for _ in range(6)]


def failures(ids, masks):
    return {name: dict.fromkeys([u for u, f in zip(ids, m) if f], 'fixture') for name, m in zip(b.MEMBERS, masks)}


def cols(x, c):
    return np.ascontiguousarray(x[:, list(c)]).tobytes()


@pytest.mark.parametrize('case', ['healthy', 'all64', 'constant', 'own_failed', 'r384_failed', 'compact_failed',
                                  'both_r_failed', 'both_r_and_own_failed', 'mixed'])
def test_oracle_protected_ten_and_admitted_route(funcs, case):
    ids, arrays = inputs()
    masks = [np.zeros(len(ids), bool) for _ in arrays]
    if case == 'all64':
        for row, bits in enumerate(itertools.product((False, True), repeat=6)):
            for mask, bit in zip(masks, bits): mask[row] = bit
    elif case == 'constant':
        for a in arrays: a[:] = .2
    elif case == 'own_failed':
        for m in masks[3:]: m[:] = True
    elif case == 'r384_failed': masks[0][:] = True
    elif case == 'compact_failed': masks[2][:] = True
    elif case == 'both_r_failed':
        masks[0][:30] = True; masks[1][:30] = True  # 2.5% < 3% R224 gate
    elif case == 'both_r_and_own_failed':
        masks[0][:30] = True; masks[1][:30] = True
        for m in masks[3:]: m[20:60] = True
    elif case == 'mixed':
        rng = np.random.default_rng(7)
        for i, m in enumerate(masks): m[rng.choice(len(ids), 30 if i == 1 else 120, replace=False)] = True
    fl = failures(ids, masks)
    actual = funcs['candidate']['blend'](ids, *arrays, fl)
    assert actual.tobytes() == b.oracle(arrays, masks).tobytes()
    prior = funcs['parent']['blend'](ids, *arrays, fl)
    assert cols(actual, b.PROTECTED) == cols(prior, b.PROTECTED)
    # Columns 7/11 byte-identical to the admitted H-B-187 R-only route; col 7 to H-B-168.
    assert cols(actual, b.ROUTED) == cols(funcs['h187']['blend'](ids, *arrays, fl), b.ROUTED)
    assert cols(actual, (7,)) == cols(funcs['h168']['blend'](ids, *arrays, fl), (7,))
    assert np.isfinite(actual).all() and np.all((actual >= 0) & (actual <= 1))
    if case in ('healthy', 'constant'):
        np.testing.assert_array_equal(actual, funcs['candidate']['reference'](*arrays))
        assert cols(actual, b.ROUTED) == cols(funcs['h187']['reference'](*arrays), b.ROUTED)
    if case == 'all64':
        assert np.all(actual[63] == .5)
        assert np.sum(masks[1]) == 32 and 32/len(ids) < .03
    if case in ('healthy', 'constant'):
        assert cols(actual, b.ROUTED) != cols(prior, b.ROUTED) or case == 'constant'


def test_both_r_failed_uses_own_only_not_compact(funcs):
    """The registered reading: both R arms failed => O only for 7/11 (not the compact)."""
    ids, arrays = inputs(400)
    masks = [np.zeros(len(ids), bool) for _ in arrays]
    masks[0][:5] = True; masks[1][:5] = True
    fl = failures(ids, masks)
    actual = funcs['candidate']['blend'](ids, *arrays, fl)
    rank = lambda a: pd.DataFrame(a).rank(method='average', pct=True).to_numpy()
    oo = rank((17*rank(arrays[3]) + 17*rank(arrays[4]) + 12*rank(arrays[5]))/46)  # own healthy here
    assert actual[:5][:, list(b.ROUTED)].tobytes() == np.ascontiguousarray(oo[:5][:, list(b.ROUTED)]).tobytes()
    changed = [a.copy() for a in arrays]; changed[2] = changed[2][::-1].copy()
    other = funcs['candidate']['blend'](ids, *changed, fl)
    assert cols(other, b.ROUTED) == cols(actual, b.ROUTED)  # compact never enters 7/11
    assert cols(other, b.PROTECTED) != cols(actual, b.PROTECTED)
    # The parent H-B-189 (w=.5, compact fallback inside P) differs on these rows for 7/11;
    # the true vector-only variant is covered by test_mutations_fail_the_registered_contract.
    prior = funcs['parent']['blend'](ids, *arrays, fl)
    assert not np.array_equal(prior[:5][:, list(b.ROUTED)], actual[:5][:, list(b.ROUTED)])


def test_eight_public_truth_rows_and_success_only_ranks(funcs):
    ids, arrays = inputs(400)
    masks = [np.zeros(len(ids), bool) for _ in range(3)]
    bits = list(itertools.product((False, True), repeat=3))
    for row, combination in enumerate(bits):
        for mask, bit in zip(masks, combination): mask[row] = bit
    def rank(a, m):
        z = np.zeros(a.shape)
        z[~m] = pd.DataFrame(a[~m]).rank(method='average', pct=True).to_numpy()
        return z
    ar, br, sr = [rank(a, m) for a, m in zip(arrays, masks)]
    public = funcs['candidate']['p_branch'](*arrays[:3], *masks)
    w = np.asarray(b.WEIGHTS)
    for row, (rf, f224, sf) in enumerate(bits):
        rr = br[row] if rf else ar[row] if f224 else .65*ar[row] + .35*br[row]
        if rf and f224: expected = np.full(12, .5) if sf else sr[row]
        elif sf: expected = rr
        else: expected = w*rr + (1-w)*sr[row]
        assert public[row].tobytes() == expected.tobytes()
        if not (rf and f224): assert public[row, list(b.ROUTED)].tobytes() == np.asarray(rr)[list(b.ROUTED)].tobytes()
    prior = funcs['parent']['p_branch'](*arrays[:3], *masks)
    assert cols(public, b.PROTECTED) == cols(prior, b.PROTECTED)


def _mutant(transform):
    files = dict(b.payload()[2]); raw = files[b.HELPER]
    files[b.HELPER] = transform(raw)
    assert files[b.HELPER] != raw
    return b.math_functions(files)


def _passes(f, parent):
    """Registered contract on synthetic all64 + many-row failures + healthy reference."""
    ids, arrays = inputs()
    masks = [np.zeros(len(ids), bool) for _ in arrays]
    for row, bits in enumerate(itertools.product((False, True), repeat=6)):
        for mask, bit in zip(masks, bits): mask[row] = bit
    masks[0][100:120] = True; masks[1][100:104] = True  # 4 extra both-R-failed rows; R224 36/1200 = 3%
    masks[3][150:170] = True; masks[4][150:170] = True; masks[5][150:170] = True
    fl = failures(ids, masks)
    out = f['blend'](ids, *arrays, fl)
    ok = out.tobytes() == b.oracle(arrays, masks).tobytes()
    ok &= cols(out, b.PROTECTED) == cols(parent['blend'](ids, *arrays, fl), b.PROTECTED)
    ok &= np.array_equal(f['blend'](ids, *arrays), f['reference'](*arrays))
    return bool(ok)


@pytest.mark.parametrize('mutation', ['vector_only', 'selector_7_9_11', 'reference_at_half', 'selector_before_outer'])
def test_mutations_fail_the_registered_contract(funcs, mutation):
    assert _passes(funcs['candidate'], funcs['parent'])
    block = (b.SURVIVOR + b.SELECT).encode()
    outer = b'    result[of], result[pf], result[pf & of] = pr[of], oo[pf], .5\n'
    vec = lambda w: ('w = np.asarray(' + repr(w) + ')\n    return .85').encode()
    transform = {
        'vector_only': lambda raw: raw.replace(block, b''),
        'selector_7_9_11': lambda raw: raw.replace(b.SELECT.encode(), b'    result[:, [7,9,11]] = routed[:, [7,9,11]]\n'),
        'reference_at_half': lambda raw: raw.replace(vec(b.WEIGHTS), vec(h189.WEIGHTS)),
        'selector_before_outer': lambda raw: raw.replace(outer + block, block + outer),
    }[mutation]
    assert not _passes(_mutant(transform), funcs['parent'])


def test_registered_weights_and_labels(funcs):
    assert b.WEIGHTS == tuple(1.0 if i in (7, 11) else w for i, w in enumerate(h189.WEIGHTS))
    assert [funcs['candidate']['p'].LABELS[i] for i in b.ROUTED] == ['Effusion', 'Fracture']
    assert b.WEIGHTS[9] == .35 and b.WEIGHTS[5] == .2 and b.WEIGHTS[6] == .6 and b.WEIGHTS[8] == .8
    assert b.PROTECTED == (0, 1, 2, 3, 4, 5, 6, 8, 9, 10)


def test_r224_three_percent_abort_unchanged(funcs):
    ids, arrays = inputs(100)
    arrays[1][:3] = .5
    funcs['candidate']['blend'](ids, *arrays)
    arrays[1][3] = .5
    with pytest.raises(ValueError, match='more than 3%'): funcs['candidate']['blend'](ids, *arrays)
    with pytest.raises(ValueError): funcs['candidate']['failure_gate'](np.array([], bool))


@pytest.mark.parametrize('bad', ['nan', 'inf', 'negative', 'over_one', 'shape', 'duplicate', 'r_nan'])
def test_invalid_rejected(funcs, bad):
    ids, arrays = inputs(100)
    if bad == 'nan': arrays[2][0, 5] = np.nan
    if bad == 'inf': arrays[2][0, 6] = np.inf
    if bad == 'negative': arrays[2][0, 5] = -.1
    if bad == 'over_one': arrays[2][0, 6] = 1.1
    if bad == 'shape': arrays[2] = arrays[2][:, :11]
    if bad == 'duplicate': ids[0] = ids[1]
    if bad == 'r_nan': arrays[0][0, 7] = np.nan
    with pytest.raises(ValueError): funcs['candidate']['blend'](ids, *arrays)


def test_all_embedded_files_and_function_scope():
    nb, meta, files, blob, manifest, full = b.artifacts()
    pnb, original, _, _ = b.parent()
    b.scope(nb, meta)
    assert len(manifest['unchanged']) == 29 and len(manifest['changed']) == 2
    assert not manifest['added'] and not manifest['removed']
    assert set(files) == set(original)
    for item in manifest['unchanged']: assert files[item['name']] == original[item['name']]
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
    # Exact line-level delta of the helper: FORMULA, two vectors, five inserted blend lines.
    import difflib
    delta = [l for l in difflib.ndiff(original[b.HELPER].decode().splitlines(True), files[b.HELPER].decode().splitlines(True))
             if l[:1] in '+-']
    removed = [l[2:] for l in delta if l[0] == '-']; added = [l[2:] for l in delta if l[0] == '+']
    assert len(removed) == 3 and len(added) == 8
    assert sum(l.startswith('FORMULA = ') for l in removed) == 1 and sum('w = np.asarray(' in l for l in removed) == 2
    assert ''.join(added[2:7]) == b.SURVIVOR + b.SELECT
    assert files[b.HELPER].count((b.SURVIVOR + b.SELECT).encode()) == 1
    assert b.old.replace(original[b.WRAPPER], b.old.digest(original[b.HELPER]), b.old.digest(files[b.HELPER])).decode().split('def runtime_pins', 1)[1].split('\ndef ', 1)[0] == \
           files[b.WRAPPER].decode().split('def runtime_pins', 1)[1].split('\ndef ', 1)[0]
    pm = b.old.read(b.BASE/'kernel/kernel-metadata.json')
    assert {k: v for k, v in pm.items() if k not in ('id', 'title')} == {k: v for k, v in meta.items() if k not in ('id', 'title')}
    assert meta['id'] == b.SLUG and meta['title'] == 'knee-lb-r6w958fxq-v1'
    assert all(k.encode() in full for k in files)
    assert b.payload()[3] == blob
    assert meta['dataset_sources'] == b.PRIVATE_DATASETS
    for name in ('exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp189_s6_port.py'):
        assert files[name] == original[name]


def test_survivor_block_byte_identical_to_admitted_references():
    child = b.payload()[2][b.HELPER]
    assert child.count(b.SURVIVOR.encode()) == 1
    for label in ('h187', 'h168'):
        helper = b.reference(label)[1][b.HELPER]
        assert helper.count(b.SURVIVOR.encode()) == 1
        routed = list(b.REFERENCES[label]['routed'])
        selector = '    result[:, %s] = routed[:, %s]\n' % (str(routed).replace(' ', ''), str(routed).replace(' ', ''))
        assert (b.SURVIVOR + selector).encode() in helper
    proof = b.reference_proof()
    assert [blk['reference_stage'] for blk in proof['blocks']] == [
        '.local/scratch/astra-b5/hb187-gate/gate-copy-v145', '.local/scratch/astra-b5/hb168-gate/gate-copy-v144']


def test_gate_reference_blocks_accept_stage_proof_and_reject_mutation(tmp_path):
    from tools import strict_variant_gate_r224_v145 as generic
    inputs = b.gate.Inputs(b.ROOT)
    child = b.gate.kernel(b.OUT, inputs)
    proofs = generic.reference_blocks(b.OUT/b.PROOF, child, inputs)
    assert len(proofs) == 2 and all(p['block_sha256'] == b.gate.sha(b.SURVIVOR.encode()) for p in proofs)
    bad = b.reference_proof(); bad['blocks'][0]['text'] = b.SURVIVOR.replace('.85', '.86')
    path = tmp_path/'proof-mutation.json'  # fixed lane basetemp inside the project
    path.write_bytes(b.old.encoded(bad))
    with pytest.raises(ValueError, match='byte-for-byte'):
        generic.reference_blocks(path, child, b.gate.Inputs(b.ROOT))


@pytest.mark.parametrize('kind', ['bootstrap', 'metadata'])
def test_scope_mutation_fails_closed(kind):
    nb, meta, *_ = b.payload()
    if kind == 'bootstrap': nb['cells'][-1]['source'].append('\nprint("extra")\n')
    else: meta['enable_internet'] = not meta.get('enable_internet', False)
    with pytest.raises(ValueError, match='forbidden'): b.scope(nb, meta)


@pytest.mark.parametrize('value,parent,result', [(.956, None, 'DROP'), (.956, .956, 'DROP'), (.957, None, 'HOLD'),
    (.957, .957, 'RETAIN'), (.958, .957, 'RETAIN'), (.957, .958, 'DROP'), (.958, .958, 'RETAIN'), (None, .958, 'HOLD')])
def test_registered_decision(value, parent, result):
    assert b.decision(value, parent) == result


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, -.1, 1.1, '0.958'])
def test_decision_rejects_invalid(bad):
    with pytest.raises(ValueError): b.decision(bad, .957)
    with pytest.raises(ValueError): b.decision(.958, bad)


def test_pre_review_cannot_authorize_spec_tamper(tmp_path, monkeypatch):
    altered = tmp_path/'SPEC.md'; altered.write_bytes(b.SPEC.read_bytes() + b'changed')
    monkeypatch.setattr(b, 'SPEC', altered)
    with pytest.raises(ValueError): b.pre_review()


@pytest.mark.parametrize('which', ['parent_receipt', 'parent_kernel', 'reference_kernel'])
def test_wrong_parent_or_reference_identity_fails_closed(monkeypatch, which):
    if which == 'parent_receipt':
        monkeypatch.setattr(b, 'PARENT_RECEIPT', '0'*64)
        with pytest.raises(ValueError): b.parent()
    elif which == 'parent_kernel':
        monkeypatch.setattr(b, 'PARENT_KERNEL', '0'*64)
        with pytest.raises(ValueError): b.parent()
    else:
        refs = {k: dict(v) for k, v in b.REFERENCES.items()}; refs['h187']['kernel'] = '0'*64
        monkeypatch.setattr(b, 'REFERENCES', refs)
        with pytest.raises(ValueError): b.reference_proof()


def test_r6_cache_stale_registered_input_fails_closed(monkeypatch):
    from tools import r6_gold58_precheck_v1 as r6
    def reject(): raise ValueError('registered input drift: deliberate fixture')
    monkeypatch.setattr(r6, 'review_ready', reject)
    with pytest.raises(ValueError, match='registered input drift'): b.reuse.authenticated_r6()


def test_registered_stage_identity(funcs):
    reg = b.old.read(b.OUT/'registration.json')
    assert reg == b.registration() and reg['routed_columns'] == [7, 11]
    assert reg['weights'] == list(b.WEIGHTS)
    assert reg['exposure'] == 'EXPOSED' and not reg['clean_oof'] and not reg['qualified_for_submission']
    assert reg['public_reading']['expected'] == .958 and reg['public_reading']['expected_if_transfer_fails'] == .957
    assert reg['timing']['inference_delta'] == 'none' and reg['timing']['hidden_hours'] is None
    assert reg['runtime_parent_submission'] == 57251520
    assert reg['gold58']['use'] == 'descriptive only, not a gate'
    assert b.validate()['identical'] == 29


def test_isolated_embedded_collect(tmp_path):
    """Smoke the unchanged collect wiring of the actual candidate archive (H189 fixture pattern)."""
    runtime = b.SCRATCH/'candidate-runtime'
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
assert c.VERSION=='1.1.41-r6w958fxq' and c.KERNEL_ID=='taeyangg4/knee-lb-r6w958fxq-v1'
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
assert receipt['fallbacks']['r224_prediction']==0 and receipt['version']=='1.1.41-r6w958fxq'
assert (tmp/'out/submission.csv').read_bytes()==payload
import torch
assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
print(json.dumps({'status':'PASS','studies':len(ids),'embedded_collect':True}))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    from tools import exp169_release_c85r7codec as original
    run = subprocess.run([sys.executable, '-B', '-I', '-c', script, str(runtime), str(tmp_path), str(original.p.REPLAY),
                          str(original.c85.A5Q)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-7000:]
    assert '"embedded_collect": true' in run.stdout


def test_admit_rejects_stale_smoke(monkeypatch):
    original = b.old.read
    def read(path):
        if path == b.SCRATCH/'replay.json': return {'status': 'PASS', 'kernel_sha256': 'stale'}
        return original(path)
    monkeypatch.setattr(b.old, 'read', read)
    with pytest.raises(ValueError, match='missing/stale'): b.admit()
