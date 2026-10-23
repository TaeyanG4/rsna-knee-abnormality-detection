"""H-B-167 exact artifact tests. Fixture dispatch archives never enter production."""
import ast
import base64
import copy
import importlib.util
import io
from pathlib import Path
import zipfile
import pytest
from tools import strict_variant_gate as g
from tools import strict_variant_gate_r224_v143 as r
from tools import kaggle_submit_once_v137 as ks
from test_strict_variant_gate import dispatched, put


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    real = g.ROOT / r.CANONICAL_STAGE
    g.dispatch(g.ROOT/'.local/scratch/slot-b/hb167/build-out.md.receipt.json', g.Inputs(g.ROOT))
    source = tmp_path/'candidate'
    for name in ('kernel/release.ipynb','kernel/kernel-metadata.json','new-members.json',r.SOURCE_DIFF):
        put(source/name, (real/name).read_bytes())
    monkeypatch.setattr(r,'STAGE',source.relative_to(g.ROOT).as_posix())
    original = g.dispatch
    def isolated(path, inputs):
        scope = g.Inputs(tmp_path)
        result = original(g.local_path(path,inputs.root),scope)
        inputs.data.update(scope.data)
        return result
    monkeypatch.setattr(g,'dispatch',isolated)
    c=dict(tmp=tmp_path,source=source,pin=tmp_path/'PIN.json',output=tmp_path/'bound',review=tmp_path/'review.md')
    reseal(c)
    return c


def reseal(c):
    hashes={n:g.sha((c['source']/'kernel'/n).read_bytes()) for n in ('release.ipynb','kernel-metadata.json')}
    c['builder']=dispatched(c['tmp'],'flip-builder',c['tmp']/'build.md',
        ('Kernel-SHA256: '+g.sha(g.encoded(hashes))+'\n').encode())


def endorse(c, mutation=None):
    text=(c['output']/'review-request.md').read_text(encoding='utf-8')
    text=text.replace('<independent lane>','flip-reviewer').replace('VERDICT: PENDING','VERDICT: PASS')
    lane='flip-reviewer'
    if mutation=='same-lane':
        text=text.replace('Reviewer-Lane: flip-reviewer','Reviewer-Lane: flip-builder'); lane='flip-builder'
    for field in ('SHA256SUMS','Allowlist-SHA256','R224-PIN-SHA256','R224-Scope','R224-Flip-Content'):
        if mutation==field: text='\n'.join(x for x in text.splitlines() if not x.startswith(field+':'))+'\n'
    dispatched(c['tmp'],lane,c['review'],text.encode())


def bind(c, review=True):
    r.pin(c['source'],c['builder'],c['pin'])
    r.bind(c['pin'],c['output'],c['review'])
    if review: endorse(c)


def test_pin_bind_evaluate_fixture_only(candidate):
    c=candidate
    bind(c,False)
    saved=g.parse(c['pin'].read_bytes())
    assert saved['fixture_only'] is True
    assert saved['exact_member_diffs']==r.EXACT_CHANGES
    assert len(saved['identical_members'])==28
    assert saved['flip_reference']['exact_diff']==r.EXACT_CHANGES[r.FLIP_MEMBER]
    assert r.evaluate(c['output'],c['review'])['status']=='FAIL'
    endorse(c)
    result=r.evaluate(c['output'],c['review'])
    assert result['status']=='PASS',result['errors']
    assert result['full_diff'].count(': IDENTICAL ')>=28
    args=['evaluate','--stage',str(c['output']),'--review',str(c['review'])]
    assert r.main(args+['--check-only'])==0
    assert r.main(args)==2
    assert r.pin(c['source'],c['builder'],c['pin'])['status']=='REVIEW_REQUIRED'
    with pytest.raises(ValueError,match='output exists'): r.bind(c['pin'],c['output'],c['review'])


@pytest.mark.parametrize('mutation',['same-lane','SHA256SUMS','Allowlist-SHA256','R224-PIN-SHA256','R224-Scope','R224-Flip-Content'])
def test_review_refusal(candidate,mutation):
    c=candidate; bind(c); endorse(c,mutation)
    assert r.evaluate(c['output'],c['review'])['status']=='FAIL'


@pytest.mark.parametrize('name',['kernel/release.ipynb','kernel/kernel-metadata.json','new-members.json',r.SOURCE_DIFF])
def test_artifact_drift_resealed_builder(candidate,name):
    c=candidate; path=c['source']/name; path.write_bytes(path.read_bytes()+b'\n'); reseal(c)
    with pytest.raises(ValueError): r.pin(c['source'],c['builder'],c['pin'])


@pytest.mark.parametrize('name',['R224-PIN.json','full-diff.txt','allowlist.json','registration.json','parent/kernel/release.ipynb','kernel/release.ipynb'])
def test_bound_drift(candidate,name):
    c=candidate; bind(c); path=c['output']/name
    if name=='R224-PIN.json':
        obj=g.parse(path.read_bytes()); obj['exact_member_diffs'][r.FLIP_MEMBER]['unified_diff']+='\n'; put(path,obj)
    else: path.write_bytes(path.read_bytes()+b'\n')
    assert r.evaluate(c['output'],c['review'])['status']=='FAIL'


@pytest.mark.parametrize('mutation',['fast','guard','launch','extra-member','removed-member','renamed-member','changed-hunk','identical-member'])
def test_deep_boundaries_even_without_child_kernel_pin(candidate,monkeypatch,mutation):
    c=candidate
    path=c['source']/'kernel/release.ipynb'; nb=g.parse(path.read_bytes())
    if mutation=='fast': nb['cells'][0]['source'].append('# drift\n')
    elif mutation=='guard': nb['cells'][2]['source'][0]='if True:\n'
    else:
        kernel=g.kernel(c['source'],g.Inputs(g.ROOT))
        members={n[9:]:b for n,b in kernel[1].items() if n.startswith('embedded/')}
        if mutation=='extra-member': members['extra.py']=b'X=1\n'
        if mutation=='removed-member': del members['exp169_r224.py']
        if mutation=='renamed-member': members['renamed.py']=members.pop('exp169_r224.py')
        if mutation=='changed-hunk': members[r.FLIP_MEMBER]=members[r.FLIP_MEMBER].replace(b'args += ["--flip-tta"]',b'args += ["--flip-bad"]')
        if mutation=='identical-member': members['exp169_r224.py']+=b'\n'
        if mutation=='launch': members['exp169_release_c85r7codec.py']=members['exp169_release_c85r7codec.py'].replace(b'label="C85R7 R"',b'label="drift R"')
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for n,b in sorted(members.items()): z.writestr(n,b)
        blob=stream.getvalue()
        # Parse only the unindented guarded body; never execute it.
        src=''.join(line[4:] for line in nb['cells'][2]['source'][1:]); lines=src.splitlines(True)
        for node in ast.parse(src).body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
                key=node.targets[0].id
                if key=='BLOB': lines[node.lineno-1]='BLOB = '+repr(base64.b64encode(blob).decode())+'\n'
                if key=='PINS': lines[node.lineno-1]='PINS = '+repr({n:g.sha(b) for n,b in members.items()})+'\n'
        src=''.join(lines).replace(kernel[2]['archive_sha256'],g.sha(blob))
        nb['cells'][2]['source']=('if not FAST:\n'+''.join('    '+line for line in src.splitlines(True))).splitlines(True)
    put(path,nb); reseal(c)
    monkeypatch.setattr(r,'CHILD_PINS',{n:g.sha((c['source']/'kernel'/n).read_bytes()) for n in r.CHILD_PINS})
    with pytest.raises(ValueError): r.pin(c['source'],c['builder'],c['pin'])


def test_parent_and_reference_drift(monkeypatch):
    original=g.Inputs.read
    for relative in (r.PARENT_RECEIPT, r.PARENT+'/kernel/release.ipynb',r.FLIP_REFERENCE+'/kernel/release.ipynb'):
        def changed(self,path):
            data=original(self,path)
            return data+b'\n' if g.local_path(path,self.root)==self.root/relative else data
        with monkeypatch.context() as m:
            m.setattr(g.Inputs,'read',changed)
            with pytest.raises(ValueError): r.pin_payload(g.ROOT/r.STAGE,g.ROOT/'.local/scratch/slot-b/hb167/build-out.md.receipt.json',g.Inputs(g.ROOT))


def test_flip_equivalence_independent_of_child_pin(monkeypatch):
    inputs=g.Inputs(g.ROOT)
    old=g.kernel(g.ROOT/r.PARENT,inputs); new=g.kernel(g.ROOT/r.STAGE,inputs)
    r.flip_equivalence(old,new,inputs)
    bad=list(new); bad[1]=dict(new[1]); bad[1]['embedded/'+r.FLIP_MEMBER]+=b'\n'
    with pytest.raises(ValueError,match='byte-for-byte'): r.flip_equivalence(old,bad,inputs)


def test_consumer_and_preflight(candidate,monkeypatch):
    c=candidate; bind(c)
    result=r.evaluate(c['output'],c['review']); assert result['status']=='PASS',result['errors']
    path=ks.ROOT/'.local/state/gates'/(r.SLUG.split('/')[1]+'.json')
    original=ks._json_evidence
    def evidence(p,hashes):
        if p==path: hashes[str(path)]='test'; return result
        return original(p,hashes)
    monkeypatch.setattr(ks,'_json_evidence',evidence)
    def consume(): return ks._strict_gate(r.SLUG,{'gate_sha256':'test'},{str(path):'test'})
    with pytest.raises(ValueError,match='fixture-only'): consume()
    # In-memory admission fixture only, never publish a state/gates receipt.
    result['fixture_only']=False
    monkeypatch.setattr(r,'CANONICAL_STAGE',r.STAGE)
    pinpath=c['output']/'R224-PIN.json'; pin=g.parse(pinpath.read_bytes()); pin['fixture_only']=False; put(pinpath,pin)
    allowpath=c['output']/'allowlist.json'; allow=g.parse(allowpath.read_bytes()); allow['pin_sha256']=g.sha(pinpath.read_bytes()); put(allowpath,allow)
    for p in (pinpath,allowpath): result['input_sha256'][p.relative_to(g.ROOT).as_posix()]=g.sha(p.read_bytes())
    assert consume()==result
    for name in ('tools/strict_variant_gate_r224_v143.py','tools/strict_variant_gate_r224_v142.py','tools/strict_variant_gate_standalone.py',r.PARENT_RECEIPT):
        value=result['input_sha256'].pop(name)
        with pytest.raises(ValueError,match='required stage/tool'): consume()
        result['input_sha256'][name]=value
    for mutation in ('wrong-slug','old-parent','hunk','source'):
        saved=copy.deepcopy(pin)
        if mutation=='wrong-slug': pin['kernel_id']='taeyangg4/wrong-slug'
        if mutation=='old-parent': pin['parent']=r.previous.PARENT
        if mutation=='hunk': pin['exact_member_diffs'][r.FLIP_MEMBER]['unified_diff']+='\n'
        if mutation=='source': pin['source_stage']='artifacts/other'
        put(pinpath,pin); allow['pin_sha256']=g.sha(pinpath.read_bytes()); put(allowpath,allow)
        for p in (pinpath,allowpath): result['input_sha256'][p.relative_to(g.ROOT).as_posix()]=g.sha(p.read_bytes())
        with pytest.raises(ValueError,match='PIN binding differs'): consume()
        pin=saved
    put(pinpath,pin); allow['pin_sha256']=g.sha(pinpath.read_bytes()); put(allowpath,allow)
    for p in (pinpath,allowpath): result['input_sha256'][p.relative_to(g.ROOT).as_posix()]=g.sha(p.read_bytes())
    spec=importlib.util.spec_from_file_location('preflight143',g.ROOT/'.local/run-scripts/b-preflight-v105.py')
    pre=importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)
    stamp=c['tmp']/'stamp'; stamp.write_text('stamp')
    original_local=pre.local
    monkeypatch.setattr(pre,'local',lambda p:stamp if str(p)=='.local/state/gates/'+path.name else original_local(p))
    monkeypatch.setattr(pre,'read',lambda p:result if p==stamp else g.parse(p.read_bytes()))
    monkeypatch.setattr(pre,'sha',lambda p:'test' if p==stamp else g.sha(p.read_bytes()))
    real_spec=importlib.util.spec_from_file_location
    class Loader:
        def create_module(self,spec): return ks
        def exec_module(self,module): pass
    from importlib.machinery import ModuleSpec
    monkeypatch.setattr(importlib.util,'spec_from_file_location',lambda name,path:ModuleSpec(name,Loader()) if name=='preflight_submit_policy' else real_spec(name,path))
    pre.gate_check(c['output'],{'id':r.SLUG,'title':r.SLUG.split('/')[1]})
    result['stage']=str(c['source'])
    with pytest.raises(pre.CheckFailed): pre.gate_check(c['output'],{'id':r.SLUG,'title':r.SLUG.split('/')[1]})


def test_frozen_inputs_preserved():
    for name,digest in {
        'tools/strict_variant_gate_r224_v142.py':'8509ec83cbe2997ae7d253ac9f13b186ff071b6a86aaff66417eb56266fe90b3',
        '.local/state/gates/knee-lb-r224q-v1.json':'25e9c03e00293e4cbb0ac5088afca7875dfaa1d56d9d16b6ff7248d98e69afe3',
        'tools/kaggle_submit_once.py':'26d8de41c3bfa981b3bf3f24b23e88d7f2a709f0a09314bd07987a8375f252aa',
        '.local/run-scripts/b-preflight-v1.py':'56a42888532fc5353843a4ae1bc295989c114ef9fe9bacecf670b14975eb1043',
    }.items():
        assert g.sha((g.ROOT/name).read_bytes())==digest
        assert (g.ROOT/name).read_bytes()==(g.ROOT/'.local/backups/strict-variant-gate-v1.4.3'/Path(name).name).read_bytes()
