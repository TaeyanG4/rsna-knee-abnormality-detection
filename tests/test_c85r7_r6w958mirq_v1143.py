"""H-B-195: admitted R-224 anatomical mirror (H-B-179/H-B-170 bytes) on admitted H-B-189.

Test file v1.0.0 (claude-opus-b6). Synthetic oracle/failure/scope/proof tests;
the real neural parity and the 2/58-study embedded replay live in
tools/c85r7_r6w958mirq_cpu_v1143.py (builder command cpu-replay).
"""
import ast
import difflib
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tools import c85r7_r6w958mirq_v1143 as b
from tools import c85r7_r6w958mirq_cpu_v1143 as cpu
from tools import c85r7_r6w958q_v1139 as h189


@pytest.fixture(scope='module')
def funcs():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    return dict(candidate=b.math_functions(b.payload()[2]), parent=b.math_functions(b.parent()[1]),
                h179=b.math_functions(b.reference('h179')[1]))


def inputs(n=1200, seed=195):
    rng = np.random.default_rng(seed)
    return [f's{i}' for i in range(n)], [rng.integers(1, 49, (n, 12))/50 for _ in range(6)]


def failures(ids, masks):
    return {name: dict.fromkeys([u for u, f in zip(ids, m) if f], 'fixture') for name, m in zip(b.MEMBERS, masks)}


def cols(x, c):
    return np.ascontiguousarray(x[:, list(c)]).tobytes()


@pytest.mark.parametrize('case', ['healthy', 'all64', 'constant', 'own_failed', 'r384_failed', 'compact_failed',
                                  'r224_sentinel', 'r224_sentinel_and_dict', 'both_r_failed', 'mixed'])
def test_blend_unchanged_vs_h189_and_half_columns_vs_h179(funcs, case):
    ids, arrays = inputs()
    masks = [np.zeros(len(ids), bool) for _ in arrays]
    sentinel = []
    if case == 'all64':
        for row, bits in enumerate(itertools.product((False, True), repeat=6)):
            for mask, bit in zip(masks, bits): mask[row] = bit
    elif case == 'constant':
        for a in arrays: a[:] = .2
    elif case == 'own_failed':
        for m in masks[3:]: m[:] = True
    elif case == 'r384_failed': masks[0][:] = True
    elif case == 'compact_failed': masks[2][:] = True
    elif case == 'r224_sentinel': sentinel = list(range(0, 1200, 40))       # 30 = 2.5% (mirror FALLBACK rows)
    elif case == 'r224_sentinel_and_dict':
        sentinel = list(range(0, 600, 40)); masks[1][700:715] = True          # 15 + 15 = 2.5%
    elif case == 'both_r_failed':
        masks[0][:30] = True; masks[1][:30] = True
    elif case == 'mixed':
        rng = np.random.default_rng(7)
        for i, m in enumerate(masks): m[rng.choice(len(ids), 30 if i == 1 else 120, replace=False)] = True
    for row in sentinel: arrays[1][row] = .5
    oracle_masks = [m.copy() for m in masks]; oracle_masks[1][sentinel] = True
    fl = failures(ids, masks)
    actual = funcs['candidate']['blend'](ids, *arrays, fl)
    assert actual.tobytes() == b.oracle(arrays, oracle_masks).tobytes()
    assert actual.tobytes() == funcs['parent']['blend'](ids, *arrays, fl).tobytes()
    assert cols(actual, b.HALF_COLUMNS) == cols(funcs['h179']['blend'](ids, *arrays, fl), b.HALF_COLUMNS)
    assert np.isfinite(actual).all() and np.all((actual >= 0) & (actual <= 1))
    if case in ('healthy', 'constant'):
        np.testing.assert_array_equal(actual, funcs['candidate']['reference'](*arrays))
    if case == 'healthy':
        assert cols(actual, b.W_COLUMNS) != cols(funcs['h179']['blend'](ids, *arrays, fl), b.W_COLUMNS)


def test_r224_sentinel_rows_use_r384_only_in_rr(funcs):
    """Mirror FALLBACK row (all .5) is R224-failed: rr for that row = success-only rank of R384."""
    ids, arrays = inputs(400)
    arrays[1][:5] = .5
    rf = np.zeros(400, bool); f224 = np.all(arrays[1] == .5, axis=1); sf = np.zeros(400, bool)
    public = funcs['candidate']['p_branch'](*arrays[:3], rf, f224, sf)
    import pandas as pd
    def rank(a, m):
        z = np.zeros(a.shape); z[~m] = pd.DataFrame(a[~m]).rank(method='average', pct=True).to_numpy(); return z
    ar, br, sr = rank(arrays[0], rf), rank(arrays[1], f224), rank(arrays[2], sf)
    w = np.asarray(b.WEIGHTS)
    expected = w*(.65*ar + .35*br) + (1-w)*sr
    expected[:5] = w*ar[:5] + (1-w)*sr[:5]
    assert public.tobytes() == expected.tobytes()
    assert funcs['candidate']['blend'](ids, *arrays).tobytes() == funcs['parent']['blend'](ids, *arrays).tobytes()


@pytest.mark.parametrize('rows,abort', [(3, False), (4, True)])
def test_r224_three_percent_abort_unchanged_including_sentinel(funcs, rows, abort):
    ids, arrays = inputs(100)
    arrays[1][:rows] = .5
    if abort:
        for f in ('candidate', 'parent', 'h179'):
            with pytest.raises(ValueError, match='more than 3%'): funcs[f]['blend'](ids, *arrays)
    else:
        assert funcs['candidate']['blend'](ids, *arrays).tobytes() == funcs['parent']['blend'](ids, *arrays).tobytes()
    ids, arrays = inputs(100)
    arrays[1][:2] = .5
    fl = {'R224': {ids[50]: 'fixture', ids[51]: 'fixture'}}           # union 4 > 3
    with pytest.raises(ValueError, match='more than 3%'): funcs['candidate']['blend'](ids, *arrays, fl)
    with pytest.raises(ValueError): funcs['candidate']['failure_gate'](np.array([], bool))


@pytest.mark.parametrize('bad', ['nan', 'inf', 'negative', 'over_one', 'shape', 'duplicate', 'r224_nan'])
def test_invalid_rejected(funcs, bad):
    ids, arrays = inputs(100)
    if bad == 'nan': arrays[2][0, 5] = np.nan
    if bad == 'inf': arrays[2][0, 6] = np.inf
    if bad == 'negative': arrays[1][0, 5] = -.1
    if bad == 'over_one': arrays[1][0, 6] = 1.1
    if bad == 'shape': arrays[1] = arrays[1][:, :11]
    if bad == 'duplicate': ids[0] = ids[1]
    if bad == 'r224_nan': arrays[1][0, 7] = np.nan
    with pytest.raises(ValueError): funcs['candidate']['blend'](ids, *arrays)


def test_mirror_member_byte_identical_to_admitted_references():
    child = b.payload()[2][b.SOURCE]
    h179, h170 = b.reference('h179')[1][b.SOURCE], b.reference('h170')[1][b.SOURCE]
    assert child == h179 == h170 and b.old.digest(child) == b.MIRROR_SHA
    parent = b.parent()[1][b.SOURCE]
    assert b.old.digest(parent) == b.PLAIN_SHA and parent == h189.parent()[1][b.SOURCE]
    hunks = b.mirror_hunks(parent, child)
    assert len(hunks) == 4
    joined = b''.join(hunks)
    for token in (b'def _mirror_tri(', b'def infer_pair_probs(', b'mirror=False', b'tri = _mirror_tri(tri, cs, D)',
                  b'mirror=True', b'infer_pair_probs(model, xw, xm, d)', b'return 0.5 * (p + q)'):
        assert token in joined
    removed = [l for l in difflib.ndiff(parent.decode().splitlines(True), child.decode().splitlines(True)) if l[:1] == '-']
    assert [l[2:] for l in removed] == ['def gpu_windows(vol, mask, k, res, dev, norm=NORM):\n',
                                        '                        arm_probs[a][i] = infer_probs(model, xw, d)\n']
    # Consumer FALLBACK (row stays 0.5 on any plain/mirror exception) is the admitted code, unchanged.
    assert child.count(b'FALLBACK ({type(e).__name__}: {e})') == parent.count(b'FALLBACK ({type(e).__name__}: {e})') == 1
    assert child.count(b'np.full((N, len(LAB)), 0.5, np.float32)') == 1


def test_no_new_dataset_or_checkpoint():
    nb, meta, files, blob = b.payload()
    parent = b.parent()[1][b.SOURCE].decode().splitlines()
    child = files[b.SOURCE].decode().splitlines()
    for marker in ('_weight_name =', 'swa-weights-20261007', 'c08r224', '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'):
        assert [l for l in parent if marker in l] == [l for l in child if marker in l] and any(marker in l for l in child)
    assert meta['dataset_sources'] == b.PRIVATE_DATASETS
    h179_meta = b.old.read(b.REFERENCES['h179']['stage']/'kernel/kernel-metadata.json')
    pm = b.old.read(b.PARENT_GATE/'kernel/kernel-metadata.json')
    strip = lambda m: {k: v for k, v in m.items() if k not in ('id', 'title')}
    assert strip(h179_meta) == strip(pm) == strip(meta)


def test_all_embedded_files_and_function_scope():
    nb, meta, files, blob, manifest, full = b.artifacts()
    pnb, original, _, _ = b.parent()
    b.scope(nb, meta)
    assert len(manifest['unchanged']) == 29
    assert sorted(c['name'] for c in manifest['changed']) == sorted([b.WRAPPER, b.SOURCE])
    assert not manifest['added'] and not manifest['removed']
    for item in manifest['unchanged']: assert files[item['name']] == original[item['name']]
    assert files[b.HELPER] == original[b.HELPER]
    assert nb['cells'][:-1] == pnb['cells'][:-1]
    before, after = ast.parse(original[b.WRAPPER]), ast.parse(files[b.WRAPPER])
    functions = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    a, z = functions(before), functions(after)
    assert a.keys() == z.keys() and {k for k in a if a[k] != z[k]} == {'runtime_pins'}
    assert [ast.dump(n) for n in before.body if isinstance(n, (ast.Import, ast.ImportFrom))] == \
           [ast.dump(n) for n in after.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    delta = [l for l in difflib.ndiff(original[b.WRAPPER].decode().splitlines(True), files[b.WRAPPER].decode().splitlines(True))
             if l[:1] in '+-']
    added = [l[2:] for l in delta if l[0] == '+']
    assert len(added) == 5 and len(delta) == 10
    assert added == ['VERSION = "1.1.43-r6w958mirq"\n',
                     'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.43-r6w958mirq"\n',
                     'KERNEL_ID = "taeyangg4/knee-lb-r6w958mirq-v1"\n',
                     'RELEASE = "C85R7+codecs R224 R6 FLIP_ALL per-target W958 R224 mirror"\n',
                     "    p.check_pin(runtime / r224.SOURCE, '" + b.MIRROR_SHA + "')\n"]
    raw = ''.join(nb['cells'][-1]['source']).encode()
    assignments = b.old.unpack(b.fast.restored(nb))[2]
    passign = b.old.unpack(b.fast.restored(pnb))[2]
    for key in ('BLOB', 'PINS'):
        raw = b.old.replace(raw, key + ' = ' + repr(assignments[key]), key + ' = ' + repr(passign[key]))
    raw = b.old.replace(raw, repr(b.old.digest(blob)), repr(b.old.digest(b.parent()[2])))
    assert raw == ''.join(pnb['cells'][-1]['source']).encode()
    assert meta['id'] == b.SLUG and meta['title'] == 'knee-lb-r6w958mirq-v1'
    assert all(k.encode() in full for k in files)
    for name in ('exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp189_s6_port.py', 'public-serving.py'):
        assert files[name] == original[name]


def test_merge_orders_equal_and_stage_proof():
    proof = b.merge_proof()
    assert proof['both_merge_orders_equal'] and proof['identical_to_h189'] == proof['identical_to_h179'] == 29
    assert b.old.read(b.OUT/b.MERGE) == json.loads(b.old.encoded(proof))
    assert len(proof['h189_wrapper_hunks_applied_to_h179']) == 3


@pytest.mark.parametrize('kind', ['h179_source', 'h170_source', 'h179_wrapper', 'h179_helper'])
def test_reference_drift_fails_closed(monkeypatch, kind):
    real = b.reference
    def fake(label):
        nb, files, blob, assignments = real(label)
        files = dict(files)
        target = {'h179_source': ('h179', b.SOURCE), 'h170_source': ('h170', b.SOURCE),
                  'h179_wrapper': ('h179', b.WRAPPER), 'h179_helper': ('h179', b.HELPER)}[kind]
        if label == target[0]: files[target[1]] += b'\n# drift\n'
        return nb, files, blob, assignments
    monkeypatch.setattr(b, 'reference', fake)
    with pytest.raises(ValueError):
        b.merge_proof()


def test_reference_proof_blocks_and_gate_acceptance(tmp_path):
    from tools import strict_variant_gate_r224_v145 as generic
    proof = b.reference_proof()
    stages = [blk['reference_stage'] for blk in proof['blocks']]
    assert stages == ['.local/scratch/astra-b5/hb179-gate/gate-copy-v145']*5 + ['.local/scratch/astra-b5/hb170-gate/gate-copy-v144']*5
    child = b.payload()[2][b.SOURCE]
    assert proof['blocks'][0]['text'].encode() == child == proof['blocks'][5]['text'].encode()
    assert b.old.read(b.OUT/b.PROOF) == json.loads(b.old.encoded(proof))
    inputs = b.gate.Inputs(b.ROOT)
    kernel = b.gate.kernel(b.OUT, inputs)
    accepted = generic.reference_blocks(b.OUT/b.PROOF, kernel, inputs)
    assert len(accepted) == 10 and accepted[0]['block_sha256'] == b.MIRROR_SHA
    bad = b.reference_proof(); bad['blocks'][3]['text'] = bad['blocks'][3]['text'].replace('mirror', 'mirr0r', 1)
    path = tmp_path/'proof-mutation.json'
    path.write_bytes(b.old.encoded(bad))
    with pytest.raises(ValueError, match='byte-for-byte'):
        generic.reference_blocks(path, kernel, b.gate.Inputs(b.ROOT))
    # The parent (plain) member does not contain the mirror blocks: the proof has power.
    parent_kernel = b.gate.kernel(b.PARENT_GATE, b.gate.Inputs(b.ROOT))
    with pytest.raises(ValueError, match='byte-for-byte'):
        generic.reference_blocks(b.OUT/b.PROOF, parent_kernel, b.gate.Inputs(b.ROOT))


@pytest.mark.parametrize('kind', ['bootstrap', 'metadata', 'helper'])
def test_scope_mutation_fails_closed(kind, monkeypatch):
    nb, meta, files, blob = b.payload()
    if kind == 'bootstrap':
        nb['cells'][-1]['source'].append('\nprint("extra")\n')
        with pytest.raises(ValueError, match='forbidden'): b.scope(nb, meta)
    elif kind == 'metadata':
        meta['enable_internet'] = not meta.get('enable_internet', False)
        with pytest.raises(ValueError, match='forbidden'): b.scope(nb, meta)
    else:
        real = b.patched
        def mutated(original, mirror=None):
            out = real(original, mirror); out[b.HELPER] = out[b.HELPER] + b'\n'; return out
        monkeypatch.setattr(b, 'patched', mutated)
        with pytest.raises(ValueError): b.scope(*b.payload()[:2])


@pytest.mark.parametrize('value,result', [(.958, 'RETAIN'), (.959, 'RETAIN'), (1, 'RETAIN'), (.957, 'DROP'),
                                          (.9579, 'DROP'), (.956, 'DROP'), (0, 'DROP')])
def test_registered_decision(value, result):
    assert b.decision(value) == result


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, -.1, 1.1, '0.958', None])
def test_decision_rejects_invalid(bad):
    with pytest.raises(ValueError): b.decision(bad)


@pytest.mark.parametrize('which', ['parent_receipt', 'parent_kernel', 'reference_kernel', 'reference_receipt', 'mirror_sha'])
def test_wrong_parent_or_reference_identity_fails_closed(monkeypatch, which):
    if which == 'parent_receipt':
        monkeypatch.setattr(b, 'PARENT_RECEIPT', '0'*64)
        with pytest.raises(ValueError): b.parent()
    elif which == 'parent_kernel':
        monkeypatch.setattr(b, 'PARENT_KERNEL', '0'*64)
        with pytest.raises(ValueError): b.parent()
    elif which == 'mirror_sha':
        monkeypatch.setattr(b, 'MIRROR_SHA', '0'*64)
        with pytest.raises(ValueError): b.mirror_member()
    else:
        refs = {k: dict(v) for k, v in b.REFERENCES.items()}
        refs['h170']['kernel' if which == 'reference_kernel' else 'receipt_sha'] = '0'*64
        monkeypatch.setattr(b, 'REFERENCES', refs)
        with pytest.raises(ValueError): b.mirror_member()


def test_runtime_basis_and_registration():
    reg = b.old.read(b.OUT/'registration.json')
    assert reg == b.registration()
    t = reg['timing']
    assert t['parent_submission'] == 57287199 and t['parent_hidden_upper_seconds'] == 26039
    m = t['mirror_increment']
    assert m['sha256'] == b.MIRROR_TIMING_SHA and abs(m['projected_hidden_conservative_seconds'] - 1737.75) < .01
    assert abs(m['projected_hidden_inference_only_seconds'] - 303.44) < .01
    assert t['projected_hidden_seconds_conservative'] == 26039 + m['projected_hidden_conservative_seconds'] <= 32400
    assert t['runtime_pass'] is True and t['limit_seconds'] == 32400
    assert reg['public_reading']['rule'].startswith('RETAIN iff displayed >= 0.958')
    assert reg['public_reading']['expected'] == .958 and reg['public_reading']['upside'] == .959
    assert reg['exposure'] == 'EXPOSED' and not reg['clean_oof'] and not reg['qualified_for_submission']
    assert reg['dataset_sources'] == b.PRIVATE_DATASETS and 'dataset_sources_delta' not in reg
    assert reg['weights'] == list(h189.WEIGHTS) and reg['mirror_member_sha256'] == b.MIRROR_SHA
    assert reg['pre_execution']['spec_sha256'] == b.old.digest(b.SPEC.read_bytes())
    assert b.validate()['identical'] == 29


def test_runtime_measurement_pin_fails_closed(monkeypatch):
    monkeypatch.setattr(b, 'MIRROR_TIMING_SHA', '0'*64)
    with pytest.raises(ValueError): b.timing()


def test_replay_plan_scenarios():
    plan = cpu.scenarios()
    keys = [p['key'] for p in plan]
    assert len(keys) == len(set(keys)) == 2*(64 + 32) + 3
    by = {p['key']: p for p in plan}
    assert all(by[f'n2-dict{c}']['expect_abort'] == bool(bits[1]) for c, bits in enumerate(itertools.product((0, 1), repeat=6)))
    assert all(by[f'n2-sentinel{c}']['expect_abort'] for c in range(32))
    assert not any(by[f'n58-sentinel{c}']['expect_abort'] or by[f'n58-dict{c}']['expect_abort'] for c in range(32))
    assert all(by[k]['expect_abort'] for k in ('n58-sentinel-two-rows', 'n58-dict-two-rows', 'n58-sentinel-plus-dict'))
    assert all(by[f'n58-sentinel{c}']['sentinel'] == [0] and by[f'n58-sentinel{c}']['dict_rows'][1] == [] for c in range(32))


def test_measured_mirror_bound_to_child_member_bytes():
    from tools import c85r7_gold58_candidates as cache
    reg = cache.io.read_json(cache.WORK/'registration.json')
    assert reg['binding']['sources']['embedded']['mirror'][b.SOURCE] == b.MIRROR_SHA
    assert b.old.digest(b.payload()[2][b.SOURCE]) == b.MIRROR_SHA


def test_isolated_embedded_collect(tmp_path):
    """Smoke the unchanged collect wiring of the actual candidate archive (H189/H191 fixture pattern)."""
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
assert c.VERSION=='1.1.43-r6w958mirq' and c.KERNEL_ID=='taeyangg4/knee-lb-r6w958mirq-v1'
import hashlib
assert hashlib.sha256((runtime/h.SOURCE).read_bytes()).hexdigest()=='699ee7d15e21d62fe83df76810f0949bdcda79f28b19c14ff67f931c2b78f428'
c.runtime_pins(runtime)
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
assert receipt['fallbacks']['r224_prediction']==0 and receipt['version']=='1.1.43-r6w958mirq'
assert receipt['release']=='C85R7+codecs R224 R6 FLIP_ALL per-target W958 R224 mirror'
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


@pytest.mark.parametrize('stale', ['replay', 'neural', 'tool'])
def test_admit_rejects_stale_replay(monkeypatch, stale):
    original = b.old.read
    current = b.validate()['kernel_sha256']
    fake = {'replay.json': dict(status='PASS', kernel_sha256=current, replay_tool_sha256=cpu.tool_sha(),
                                all12_vs_h189_float64_csv_exact=True, half_columns_vs_h179_float64_csv_exact=True,
                                studies=[2, 58]),
            'neural-parity.json': dict(status='PASS_REAL_TWO_STUDY_MIRROR_NEURAL_PARITY', kernel_sha256=current,
                                       replay_tool_sha256=cpu.tool_sha())}
    if stale == 'replay': fake['replay.json']['kernel_sha256'] = 'stale'
    if stale == 'neural': fake['neural-parity.json']['kernel_sha256'] = 'stale'
    if stale == 'tool': fake['neural-parity.json']['replay_tool_sha256'] = '0'*64
    def read(path):
        if path.parent == b.SCRATCH and path.name in fake: return fake[path.name]
        return original(path)
    monkeypatch.setattr(b.old, 'read', read)
    with pytest.raises(ValueError, match={'replay': 'missing/stale', 'neural': 'missing/stale', 'tool': 'tool drift'}[stale]):
        b.admit()
    fake['replay.json']['kernel_sha256'] = fake['neural-parity.json']['kernel_sha256'] = current
    fake['neural-parity.json']['replay_tool_sha256'] = cpu.tool_sha()
    assert b.admit()['status'] == 'HOLD_B_SLUG_GENERIC_GATE_AND_INDEPENDENT_REVIEW'
