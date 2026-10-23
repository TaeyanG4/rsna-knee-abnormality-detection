"""Exact reviewed bytes; static CPU fixtures cannot publish admission."""
import importlib.util
from pathlib import Path
import pytest

import test_strict_variant_gate_r224 as legacy
import test_strict_variant_gate_r224_v141 as previous
from tools import strict_variant_gate as g
from tools import strict_variant_gate_r224_v142 as r
from tools import kaggle_submit_once as ks
from test_strict_variant_gate import put


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    inputs = g.Inputs(g.ROOT)
    g.dispatch(g.ROOT / '.local/scratch/slot-b/hb164/build2-out.md.receipt.json', inputs)
    real = g.kernel(g.ROOT / r.CANONICAL_STAGE, inputs)
    c = previous.candidate.__wrapped__(tmp_path, monkeypatch)
    for name in (r.RAPTOR, r.PROJECT, r.WRAPPER):
        c['files'][name] = real[1]['embedded/' + name]
    monkeypatch.setattr(r, 'STAGE', c['source'].relative_to(g.ROOT).as_posix())
    monkeypatch.setattr(previous, 'r', r)
    previous.reseal(c)
    return c


def test_exact_pin_bind_evaluate(candidate):
    previous.test_exact_two_members_pin_bind_evaluate(candidate)
    saved = g.parse(candidate['pin'].read_bytes())
    exact = saved['exact_wrapper_diff']
    assert exact['before_sha256'] == r.WRAPPER_BEFORE
    assert exact['after_sha256'] == r.WRAPPER_AFTER
    assert exact['unified_diff'].count('@@ -') == 7
    assert '    def infer_r224():' in exact['unified_diff']


@pytest.mark.parametrize('mutation', ['extra-project', 'missing-project', 'renamed-project',
    'raptor-byte', 'wrong-slug', 'placeholder', 'launch', 'origin-kind', 'sha',
    'project-byte-stale-sha', 'empty-project'])
def test_refusals(candidate, mutation):
    previous.test_refusal_boundaries(candidate, mutation)


@pytest.mark.parametrize('mutation', ['wrapper-tail', 'wrapper-hunk', 'project-resealed',
                                    'unchanged-member', 'removed-parent', 'fast', 'guard'])
def test_resealed_drift_refused(candidate, mutation):
    c = candidate
    kwargs = {}
    if mutation == 'wrapper-tail': c['files'][r.WRAPPER] += b'\n'
    if mutation == 'wrapper-hunk':
        c['files'][r.WRAPPER] = c['files'][r.WRAPPER].replace(b'completed.append("R224")', b'completed.append("R224!")')
    if mutation == 'project-resealed': c['files'][r.PROJECT] += b'\n'
    if mutation in ('unchanged-member', 'removed-parent'):
        name = next(n for n in c['parent'] if n != r.WRAPPER)
        if mutation == 'unchanged-member': c['files'][name] += b'\n'
        else: del c['files'][name]
    if mutation == 'fast':
        kwargs['notebook_change'] = lambda nb: nb['cells'][0]['source'].append('# extra\n')
    if mutation == 'guard':
        def change(nb):
            nb['cells'][2]['source'][0] = 'if True:\n'
        kwargs['notebook_change'] = change
    if mutation == 'removed-parent':
        # The old test sidecar helper assumes existing members; construct the
        # malformed archive before its missing-member sidecar access fails.
        with pytest.raises(KeyError): previous.reseal(c, **kwargs)
        hashes = {n:g.sha((c['source']/'kernel'/n).read_bytes()) for n in ('release.ipynb','kernel-metadata.json')}
        from test_strict_variant_gate import dispatched
        c['builder'] = dispatched(c['tmp'], 'r224-builder', c['tmp']/'build.md',
            ('Kernel-SHA256: '+g.sha(g.encoded(hashes))+'\n').encode())
    else: previous.reseal(c, **kwargs)
    with pytest.raises(ValueError): r.pin(c['source'], c['builder'], c['pin'])


@pytest.mark.parametrize('mode', ['registered', 'same', 'conflict', 'empty', 'type', 'missing'])
def test_reason_alias(candidate, mode):
    c = candidate
    side = g.parse((c['source']/'new-members.json').read_bytes())
    row = side['changed'][0]
    reason = row.pop('reason')
    if mode != 'missing': row['registered_reason'] = reason
    if mode == 'same': row['reason'] = reason
    if mode == 'conflict': row['reason'] = 'conflict'
    if mode == 'empty': row['registered_reason'] = ''
    if mode == 'type': row['registered_reason'] = []
    put(c['source']/'new-members.json', side)
    if mode in ('registered', 'same'):
        r.pin(c['source'], c['builder'], c['pin'])
    else:
        with pytest.raises(ValueError, match='reason'): r.pin(c['source'], c['builder'], c['pin'])


@pytest.mark.parametrize('content', ['missing', 'wrong-hash', 'duplicate'])
def test_review_binding(candidate, content):
    previous.test_independent_project_review_binding(candidate, content)


def test_pin_hunks_cannot_be_changed(candidate):
    c = candidate
    r.pin(c['source'], c['builder'], c['pin'])
    pin = g.parse(c['pin'].read_bytes())
    pin['exact_wrapper_diff']['unified_diff'] += '\n'
    put(c['pin'], pin)
    with pytest.raises(ValueError, match='PIN differs'):
        r.bind(c['pin'], c['output'], c['review'])


def test_launch_check_independent_of_hash(candidate):
    c = candidate
    before, after = c['parent'][r.WRAPPER], c['files'][r.WRAPPER]
    r.existing_launches(before, after)
    for bad in (after.replace(b'label="C85R7 R"', b'label="changed R"'),
                after.replace(r.R224_INSERTION.encode(), b''),
                after.replace(b'    def infer_s():', b'    # extra\n    def infer_s():')):
        with pytest.raises(ValueError, match='launch bytes'): r.existing_launches(before, bad)


def test_consumer_and_preflight(candidate, monkeypatch):
    c = candidate
    previous.bind(c)
    result = r.evaluate(c['output'], c['review'])
    assert result['status'] == 'PASS', result['errors']
    path = ks.ROOT/'.local/state/gates'/ (r.SLUG.split('/')[1]+'.json')
    original = ks._json_evidence
    def evidence(p, hashes):
        if p == path:
            hashes[str(path)] = 'test'
            return result
        return original(p, hashes)
    monkeypatch.setattr(ks, '_json_evidence', evidence)
    def consume(): return ks._strict_gate(r.SLUG, {'gate_sha256':'test'}, {str(path):'test'})
    with pytest.raises(ValueError, match='fixture-only'): consume()
    # Test-only in-memory admission; never write a production receipt.
    result['fixture_only'] = False
    pinpath = c['output']/'R224-PIN.json'
    pin = g.parse(pinpath.read_bytes()); pin['fixture_only'] = False; put(pinpath,pin)
    allowpath = c['output']/'allowlist.json'
    allow = g.parse(allowpath.read_bytes()); allow['pin_sha256'] = g.sha(pinpath.read_bytes()); put(allowpath,allow)
    for p in (pinpath,allowpath): result['input_sha256'][p.relative_to(g.ROOT).as_posix()] = g.sha(p.read_bytes())
    assert consume() == result
    for name in ('tools/strict_variant_gate_r224_v142.py','tools/strict_variant_gate_standalone.py'):
        value = result['input_sha256'].pop(name)
        with pytest.raises(ValueError, match='required stage/tool'): consume()
        result['input_sha256'][name] = value
    for version in ('1.4.1', '1.4.3'):
        result['gate_version'] = version
        with pytest.raises(ValueError, match='supported PASS'): consume()
    result['gate_version'] = r.VERSION
    result['kernel_id'] = 'taeyangg4/wrong-slug'
    with pytest.raises(ValueError): consume()
    result['kernel_id'] = r.SLUG
    spec = importlib.util.spec_from_file_location('preflight142', g.ROOT/'.local/run-scripts/b-preflight-v1.py')
    pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
    monkeypatch.setattr(pre, 'read', lambda p: result if p == path else g.parse(p.read_bytes()))
    # Real consumer policy on the in-memory gate, with an existing newer file
    # standing in for the gate timestamp (no state/gates write).
    stamp = c['tmp']/'stamp'; stamp.write_text('stamp')
    original_local = pre.local
    monkeypatch.setattr(pre, 'local', lambda p: stamp if str(p) == '.local/state/gates/'+path.name else original_local(p))
    monkeypatch.setattr(pre, 'read', lambda p: result if p == stamp else g.parse(p.read_bytes()))
    monkeypatch.setattr(pre, 'sha', lambda p: 'test' if p == stamp else g.sha(p.read_bytes()))
    real_spec = importlib.util.spec_from_file_location
    class Loader:
        def create_module(self, spec): return ks
        def exec_module(self, module): pass
    from importlib.machinery import ModuleSpec
    monkeypatch.setattr(importlib.util, 'spec_from_file_location',
        lambda name,path: ModuleSpec(name, Loader()) if name == 'preflight_submit_policy' else real_spec(name,path))
    pre.gate_check(c['output'], {'id':r.SLUG,'title':r.SLUG.split('/')[1]})
    result['stage'] = str(c['source'])
    with pytest.raises(pre.CheckFailed):
        pre.gate_check(c['output'], {'id':r.SLUG,'title':r.SLUG.split('/')[1]})


def test_older_contract_bytes_preserved():
    for name, digest in {
        'strict_variant_gate_r224_v141.py':'83f078c2d0077ac56ca9ed989f8a671cb991993a7d14dc12650e157cc6b154f5',
        'strict_variant_gate_r224.py':'57e632183b4c01b97ae16b6c2a6b7daa4d090774a9c2ff718d2b958e93cad62b',
    }.items(): assert g.sha((g.ROOT/'tools'/name).read_bytes()) == digest
