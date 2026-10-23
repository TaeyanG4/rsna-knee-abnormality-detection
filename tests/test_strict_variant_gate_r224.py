"""Static CPU contract tests using real immutable parent/source bytes.

Test candidate is NOT astra-b1's release and no fixture may publish admission.
Its arithmetic is deliberately unqualified; dispatched test reviews exercise
binding mechanics, not scientific qualification of that candidate.
"""
import ast
import base64
import copy
import io
import zipfile

import pytest

from tools import strict_variant_gate as g
from tools import strict_variant_gate_r224 as r
from tools import kaggle_submit_once as ks
from test_strict_variant_gate import dispatched, put


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    source = tmp_path / 'candidate'
    monkeypatch.setattr(r, 'STAGE', source.relative_to(g.ROOT).as_posix())
    original_dispatch = g.dispatch
    def isolated(path, inputs):
        scope = g.Inputs(tmp_path)
        result = original_dispatch(g.local_path(path, inputs.root), scope)
        inputs.data.update(scope.data)
        return result
    monkeypatch.setattr(g, 'dispatch', isolated)
    old = g.kernel(g.ROOT / r.PARENT, g.Inputs(g.ROOT))
    files = {n[9:]: b for n,b in old[1].items() if n.startswith('embedded/')}
    parent = dict(files)
    check, code = r.original_parts(g.Inputs(g.ROOT))
    files['r224-stage.py'] = check + b'\n' + code
    wrapper = files[r.WRAPPER].decode()
    wrapper = wrapper.replace('VERSION = "1.1.0"', 'VERSION = "1.1.23-r224q"')
    wrapper = wrapper.replace('KERNEL_ID = "taeyangg4/knee-exp169-c85r7codec-v1"', 'KERNEL_ID = '+repr(r.SLUG))
    wrapper = wrapper.replace('    def infer_s():\n', r.invocation('r224-stage.py') + '    def infer_s():\n')
    files[r.WRAPPER] = wrapper.encode()
    state = dict(source=source, old=old, files=files, parent=parent, tmp=tmp_path,
                 review=tmp_path/'review.md', output=tmp_path/'bound', pin=tmp_path/'PIN.json')
    reseal(state)
    return state


def reseal(state, *, slug=r.SLUG, placeholder=False, notebook_change=None):
    old, files, source = (state[k] for k in ('old','files','source'))
    nb = g.parse(old[3]['release.ipynb'])
    src = ''.join(nb['cells'][1]['source'])
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in sorted(files.items()): z.writestr(name, content)
    blob = stream.getvalue()
    lines = src.splitlines(True)
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key == 'BLOB': lines[node.lineno-1] = 'BLOB = '+repr(base64.b64encode(blob).decode())+'\n'
            if key == 'PINS': lines[node.lineno-1] = 'PINS = '+repr({n:g.sha(b) for n,b in files.items()})+'\n'
    src = ''.join(lines).replace(old[2]['archive_sha256'], g.sha(blob))
    nb['cells'][1]['source'] = ('if not FAST:\n'+''.join('    '+line for line in src.splitlines(True))).splitlines(True)
    donor = g.parse((g.ROOT/g.FAST_REFERENCE['stage']/'kernel/release.ipynb').read_bytes())
    nb['cells'].insert(0, copy.deepcopy(donor['cells'][0]))
    if notebook_change: notebook_change(nb)
    meta = dict(old[0], id=slug, title=slug.split('/')[1])
    put(source/'kernel/release.ipynb', nb)
    put(source/'kernel/kernel-metadata.json', meta)
    before = state['parent']
    side = dict(placeholder=placeholder,
        added=[dict(name=n, sha256=g.sha(files[n]), origin=dict(source_sha256=r.SOURCE_SHA,
            edits=['raptor224_sub.csv','raptor224_raw.csv'])) for n in files.keys()-before.keys()],
        changed=[dict(name=n, before_sha256=g.sha(before[n]), after_sha256=g.sha(files[n]),
                      reason='R224 numerical formula/invocation/version') for n in before if before[n]!=files.get(n)])
    put(source/'new-members.json', side)
    hashes = {n:g.sha((source/'kernel'/n).read_bytes()) for n in ('release.ipynb','kernel-metadata.json')}
    state['builder'] = dispatched(state['tmp'], 'r224-builder', state['tmp']/'build.md',
        ('Kernel-SHA256: '+g.sha(g.encoded(hashes))+'\n').encode())


def endorse(state, *, lane='r224-reviewer', scope=True):
    output = state['output']
    allow = g.parse((output/'allowlist.json').read_bytes())
    text = (f'Reviewer-Lane: {lane}\nSHA256SUMS: '+allow['stage_sha256sums']+
        '\nAllowlist-SHA256: '+g.sha((output/'allowlist.json').read_bytes())+
        '\nR224-PIN-SHA256: '+allow['pin_sha256']+'\n')
    if scope: text += 'R224-Scope: '+r.SCOPE+'\n'
    dispatched(state['tmp'], lane, state['review'], (text+'VERDICT: PASS\n').encode())


def bind(state):
    r.pin(state['source'],state['builder'],state['pin'])
    r.bind(state['pin'],state['output'],state['review'])
    endorse(state)


def test_exact_added_source_pin_bind_review_evaluate(candidate):
    c = candidate
    assert r.main(['pin','--stage',str(c['source']),'--builder-receipt',str(c['builder']),
                   '--output',str(c['pin'])]) == 0
    pin = g.parse(c['pin'].read_bytes())
    assert pin['status'] == 'REVIEW_REQUIRED' and pin['fixture_only'] is True
    assert pin['added'] == {'r224-stage.py':g.sha(c['files']['r224-stage.py'])}
    assert r.main(['bind','--pin',str(c['pin']),'--output',str(c['output']),
                   '--review',str(c['review'])]) == 0
    assert r.evaluate(c['output'],c['review'])['status'] == 'FAIL'
    endorse(c)
    result = r.evaluate(c['output'],c['review'])
    assert result['status'] == 'PASS', result['errors']
    assert result['fixture_only'] is True
    assert result['full_diff'].count(': IDENTICAL ') >= 28
    assert 'raw-parent/release.ipynb' in result['full_diff']
    args=['evaluate','--stage',str(c['output']),'--review',str(c['review'])]
    assert r.main(args+['--check-only']) == 0
    assert r.main(args) == 2  # Test source cannot publish a production receipt.


@pytest.mark.parametrize('separators', [0, 1, 2])
def test_numerical_helper_and_changed_collect(candidate, separators):
    c = candidate
    wrapper = c['files'][r.WRAPPER].decode()
    helper = 'def r224_identity(values):\n    return values\n' + '\n' * separators
    assert 'r, ours = r[order], ours[order]' in wrapper
    wrapper = wrapper.replace('def collect(', helper + 'def collect(', 1)
    wrapper = wrapper.replace('r, ours = r[order], ours[order]',
                              'r, ours = r224_identity(r[order]), ours[order]', 1)
    c['files'][r.WRAPPER] = wrapper.encode()
    reseal(c)
    bind(c)
    result = r.evaluate(c['output'], c['review'])
    assert result['status'] == 'PASS', result['errors']


def test_unrelated_whitespace_rejected(candidate):
    c = candidate
    c['files'][r.WRAPPER] += b'\n'
    reseal(c)
    with pytest.raises(ValueError, match='outside registered spans'):
        r.pin(c['source'], c['builder'], c['pin'])


@pytest.mark.parametrize('mutation', ['added-byte','extra-added','launch','wrong-slug','placeholder',
    'deleted','untouched','import','fast','guard','metadata','source','sidecar','origin','split-source'])
def test_tampering_rejected_even_after_resealing(candidate, mutation):
    c = candidate
    kw = {}
    if mutation == 'added-byte': c['files']['r224-stage.py'] += b'\n'
    if mutation == 'extra-added': c['files']['extra.py'] = b'# extra\n'
    if mutation == 'launch': c['files'][r.WRAPPER] = c['files'][r.WRAPPER].replace(b'label="C85R7 R"', b'label="changed R"')
    if mutation == 'wrong-slug': kw['slug'] = 'taeyangg4/another-child'
    if mutation == 'placeholder': kw['placeholder'] = True
    if mutation == 'deleted': del c['files']['public-serving.py']; del c['parent']['public-serving.py']
    if mutation == 'untouched': c['files']['public-serving.py'] += b'\n'
    if mutation == 'import': c['files'][r.WRAPPER] += b'\nimport multiprocessing\n'
    if mutation == 'fast': kw['notebook_change'] = lambda nb: nb['cells'][0].update(source=['FAST = True\n'])
    if mutation == 'guard': kw['notebook_change'] = lambda nb: nb['cells'][2]['source'].__setitem__(0, 'if True:\n')
    if mutation == 'metadata': kw['slug'] = r.SLUG
    if mutation == 'split-source':
        check, code = r.original_parts(g.Inputs(g.ROOT))
        c['files']['r224-stage.py'] = code; c['files']['check.py'] = check
    reseal(c, **kw)
    if mutation == 'metadata':
        path=c['source']/'kernel/kernel-metadata.json'; meta=g.parse(path.read_bytes())
        meta['enable_internet']=True; put(path,meta)
    if mutation == 'source':
        # Override only this Inputs read: never modify the real original notebook.
        original = g.Inputs.read
        def changed(self,path):
            data=original(self,path)
            return data+b'\n' if g.local_path(path,self.root)==g.ROOT/r.ORIGINAL else data
        from unittest.mock import patch
        with patch.object(g.Inputs,'read',changed), pytest.raises(ValueError, match='original R224'):
            r.pin(c['source'],c['builder'],c['pin'])
        return
    if mutation in ('sidecar','origin'):
        path=c['source']/'new-members.json'; data=g.parse(path.read_bytes())
        data['added'][0]['sha256' if mutation=='sidecar' else 'origin']='placeholder'
        put(path,data)
    with pytest.raises(ValueError): r.pin(c['source'],c['builder'],c['pin'])
    assert not c['pin'].exists()


def test_unfinished_refuses_before_candidate_read(candidate, monkeypatch):
    c=candidate
    put(c['builder'], dict(schema='dispatch-terminal/v1', state='RUNNING', exit_code=None))
    original = g.kernel
    def guarded(stage, inputs):
        assert stage != c['source'], 'read unfinished candidate'
        return original(stage,inputs)
    monkeypatch.setattr(g,'kernel',guarded)
    with pytest.raises(ValueError, match='unsuccessful dispatch'):
        r.pin(c['source'],c['builder'],c['pin'])


@pytest.mark.parametrize('mutation', ['pin','allowlist','diff','parent','same-lane','scope','source-drift','copy','review'])
def test_bound_rechecks_every_evidence(candidate, mutation):
    c=candidate; bind(c)
    if mutation in ('pin','allowlist'):
        path=c['output']/('R224-PIN.json' if mutation=='pin' else 'allowlist.json')
        data=g.parse(path.read_bytes()); data['scope']='changed'; put(path,data)
    if mutation=='diff': put(c['output']/'full-diff.txt', b'no diff\n')
    if mutation=='parent':
        path=c['output']/'parent/kernel/release.ipynb'; put(path,path.read_bytes()+b'\n')
    if mutation=='same-lane': endorse(c,lane='r224-builder')
    if mutation=='scope': endorse(c,scope=False)
    if mutation=='source-drift':
        c['files']['r224-stage.py']+=b'\n'; reseal(c)
    if mutation=='copy':
        path=c['output']/'kernel/release.ipynb'; put(path,path.read_bytes()+b'\n')
    if mutation=='review': put(c['review'],b'VERDICT: PASS\n')
    result=r.evaluate(c['output'],c['review'])
    assert result['status']=='FAIL', mutation


def test_consumer_exact_family_and_required_hashes(candidate, monkeypatch):
    c=candidate; bind(c)
    result=r.evaluate(c['output'],c['review'])
    path=ks.ROOT/'.local/state/gates'/ (r.SLUG.split('/')[1]+'.json')
    original=ks._json_evidence
    monkeypatch.setattr(ks,'_json_evidence',lambda p,h: result if p==path else original(p,h))
    def consume(): return ks._strict_gate(r.SLUG,{'gate_sha256':'test'}, {str(path):'test'})
    with pytest.raises(ValueError,match='fixture-only'): consume()
    # Explicit test-only promotion exercises the consumer; no production writes.
    result['fixture_only']=False
    pinpath=c['output']/'R224-PIN.json'; pin=g.parse(pinpath.read_bytes()); pin['fixture_only']=False
    put(pinpath,pin)
    allowpath=c['output']/'allowlist.json'; allow=g.parse(allowpath.read_bytes())
    allow['pin_sha256']=g.sha(pinpath.read_bytes()); put(allowpath,allow)
    for p in (pinpath,allowpath): result['input_sha256'][p.relative_to(g.ROOT).as_posix()]=g.sha(p.read_bytes())
    assert consume()==result
    for name in ('tools/strict_variant_gate_r224.py','tools/strict_variant_gate_standalone.py',
                 pinpath.relative_to(g.ROOT).as_posix()):
        value=result['input_sha256'].pop(name)
        with pytest.raises(ValueError,match='required stage/tool'): consume()
        result['input_sha256'][name]=value
    result['kernel_id']='taeyangg4/wrong-slug'
    with pytest.raises(ValueError): consume()


def test_old_gate_sources_byte_identical():
    for name, digest in {
        'strict_variant_gate.py':'9a8341d6b971188b331d2a0bbca6abbc77784fd5543f057e2ca764f3761adbca',
        'strict_variant_gate_standalone.py':'e90f614d02764fe2d32158791432c516bcc98a6b429f4ca1fe54e7ac0617d085',
    }.items():
        assert g.sha((g.ROOT/'tools'/name).read_bytes())==digest
