"""Generic pin/admission regressions on real archived artifacts, CPU only.

Dispatcher fixtures are confined to pytest scratch; never publish admissions.
"""
import copy
import importlib.util
from pathlib import Path

import pytest
from tools import strict_variant_gate as g
from tools import strict_variant_gate_r224_v144 as r
from tools import kaggle_submit_once_v138 as ks
from test_strict_variant_gate import dispatched, put

STAGE = 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.25-stackq'
PARENT = '.local/scratch/astra-b5/hb167-gate/gate-copy-v143'
DIFF = 'full-diff-vs-v1.1.24-r224flipq.txt'
SLUG = 'taeyangg4/knee-lb-stackq-v1'
PROOF = '.local/scratch/astra-b5/hb168-gate/fluid-reference.json'


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    source=tmp_path/'candidate'
    for name in ('kernel/release.ipynb','kernel/kernel-metadata.json','new-members.json',DIFF):
        put(source/name,(g.ROOT/STAGE/name).read_bytes())
    original=g.dispatch
    def isolated(path,inputs):
        path=g.local_path(path,inputs.root)
        if not path.is_relative_to(tmp_path): return original(path,inputs)
        scope=g.Inputs(tmp_path); value=original(path,scope); inputs.data.update(scope.data)
        return value
    monkeypatch.setattr(g,'dispatch',isolated)
    c=dict(tmp=tmp_path,source=source,pin=tmp_path/'PIN.json',output=tmp_path/'bound',review=tmp_path/'review.md',slug=SLUG,parent=PARENT,proof=PROOF)
    reseal(c)
    return c


def reseal(c):
    hashes={n:g.sha((c['source']/'kernel'/n).read_bytes()) for n in ('release.ipynb','kernel-metadata.json')}
    c['builder']=dispatched(c['tmp'],c.get('builder_lane','generic-builder'),c['tmp']/c.get('build_name','build.md'),('Kernel-SHA256: '+g.sha(g.encoded(hashes))+'\n').encode())


def pin(c):
    return r.pin(c['source'],c['slug'],c['parent'],c['builder'],c['source']/DIFF,c['pin'],c['proof'])


def endorse(c,mutation=None):
    text=(c['output']/'review-request.md').read_text(encoding='utf-8')
    lane='generic-builder' if mutation=='same-lane' else 'generic-reviewer'
    text=text.replace('<independent lane>',lane).replace('VERDICT: PENDING','VERDICT: PASS')
    if mutation and mutation!='same-lane':
        text='\n'.join(line for line in text.splitlines() if not line.startswith(mutation+':'))+'\n'
    dispatched(c['tmp'],lane,c['review'],text.encode())


def bind(c):
    pin(c); r.bind(c['pin'],c['output'],c['review']); endorse(c)


def virtual_receipt(c,result,monkeypatch):
    """Route a canonical receipt lookup to pytest data; no state/gates writes."""
    path=g.ROOT/'.local/state/gates'/(c['slug'].split('/')[1]+'.json')
    data=c['tmp']/'admission.json'; put(data,result)
    original=g.Inputs.read
    def read(self,p):
        return original(self,data if g.local_path(p,self.root)==path else p)
    monkeypatch.setattr(g.Inputs,'read',read)
    original_evidence=ks._read_evidence
    def evidence(p,hashes):
        p=ks._local(p)
        if p==path:
            value=original_evidence(data,hashes); hashes[str(path)]=g.sha(value); return value
        return original_evidence(p,hashes)
    monkeypatch.setattr(ks,'_read_evidence',evidence)
    return path,data


def test_pin_bind_evaluate_and_fixture_publish_refusal(candidate):
    c=candidate; bind(c)
    saved=g.parse(c['pin'].read_bytes())
    assert saved['fixture_only'] is True
    assert len(saved['exact_member_diffs'])==2 and len(saved['identical_members'])==29
    assert saved['reference_blocks'][0]['child_member']=='exp169_r224.py'
    result=r.evaluate(c['output'],c['review']); assert result['status']=='PASS',result['errors']
    args=['evaluate','--stage',str(c['output']),'--review',str(c['review'])]
    assert r.main(args+['--check-only'])==0
    assert r.main(args)==2
    assert pin(c)['status']=='REVIEW_REQUIRED'
    with pytest.raises(ValueError,match='output exists'): r.bind(c['pin'],c['output'],c['review'])


@pytest.mark.parametrize('ident',[SLUG,'taeyangg4/knee-lb-r224mirq-v1','taeyangg4/future-exact-child-v99'])
def test_new_slug_requires_no_new_gate_code(candidate,ident):
    c=candidate; c['slug']=ident
    meta=g.parse((c['source']/'kernel/kernel-metadata.json').read_bytes()); meta.update(id=ident,title=ident.split('/')[1])
    put(c['source']/'kernel/kernel-metadata.json',meta); reseal(c); bind(c)
    result=r.evaluate(c['output'],c['review']); assert result['status']=='PASS',result['errors']
    assert result['kernel_id']==ident and result['fixture_only']


@pytest.mark.parametrize('mutation',['same-lane','Reviewer-Lane','SHA256SUMS','Allowlist-SHA256','R224-PIN-SHA256','R224-Scope','Exact-Diff-Content'])
def test_review_refusal(candidate,mutation):
    c=candidate; bind(c); endorse(c,mutation)
    assert r.evaluate(c['output'],c['review'])['status']=='FAIL'


def test_review_missing_or_unauthenticated(candidate):
    c=candidate; pin(c); r.bind(c['pin'],c['output'],c['review'])
    assert r.evaluate(c['output'],c['review'])['status']=='FAIL'
    put(c['review'],(c['output']/'review-request.md').read_bytes().replace(b'PENDING',b'PASS'))
    assert r.evaluate(c['output'],c['review'])['status']=='FAIL'


@pytest.mark.parametrize('name',['R224-PIN.json','full-diff.txt','allowlist.json','registration.json','SHA256SUMS','review-request.md','parent/kernel/release.ipynb','kernel/release.ipynb'])
def test_bound_mutations(candidate,name):
    c=candidate; bind(c); path=c['output']/name; path.write_bytes(path.read_bytes()+b'\n')
    assert r.evaluate(c['output'],c['review'])['status']=='FAIL'


@pytest.mark.parametrize('name',['kernel/release.ipynb','kernel/kernel-metadata.json','new-members.json',DIFF])
def test_completed_source_drift(candidate,name):
    c=candidate; bind(c); p=c['source']/name; p.write_bytes(p.read_bytes()+b'\n'); reseal(c)
    assert r.evaluate(c['output'],c['review'])['status']=='FAIL'
    with pytest.raises(ValueError): pin(c)


@pytest.mark.parametrize('mutation',['slug','metadata','fast','guard','changed','unchanged','diff','reference'])
def test_invalid_source_material(candidate,mutation):
    c=candidate
    if mutation=='slug': c['slug']='taeyangg4/wrong-child'
    if mutation=='metadata':
        p=c['source']/'kernel/kernel-metadata.json'; obj=g.parse(p.read_bytes()); obj['enable_internet']=True; put(p,obj)
    if mutation in ('fast','guard'):
        p=c['source']/'kernel/release.ipynb'; obj=g.parse(p.read_bytes())
        if mutation=='fast': obj['cells'][0]['source'].append('# drift\n')
        else: obj['cells'][2]['source'][0]='if True:\n'
        put(p,obj)
    if mutation in ('changed','unchanged'):
        p=c['source']/'new-members.json'; obj=g.parse(p.read_bytes()); obj[mutation][0]['after_sha256']='0'*64; put(p,obj)
    if mutation=='diff': put(c['source']/DIFF,b'no hunks\n')
    if mutation=='reference':
        p=c['tmp']/'proof.json'; obj=g.parse((g.ROOT/PROOF).read_bytes()); obj['blocks'][0]['text']+='wrong\n'; put(p,obj); c['proof']=p
    reseal(c)
    with pytest.raises(ValueError): pin(c)


def test_parent_receipt_requires_full_trusted_replay(candidate,monkeypatch):
    path=g.ROOT/'.local/state/gates/knee-lb-r224flipq-v1.json'
    original=g.Inputs.read
    def forged(self,p):
        data=original(self,p)
        if g.local_path(p,self.root)==path:
            obj=g.parse(data); obj['input_sha256']={}; return g.encoded(obj)
        return data
    monkeypatch.setattr(g.Inputs,'read',forged)
    with pytest.raises(ValueError,match='trusted evaluator replay'): pin(candidate)


def test_parent_must_be_gate_copy(candidate):
    candidate['parent']='artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq'
    with pytest.raises(ValueError,match='exact admitted gate copy'): pin(candidate)


@pytest.mark.parametrize('target',[PARENT,'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq'])
def test_recursive_delayed_directory_addition(monkeypatch,target):
    original_eval=r.v143.evaluate; original_glob=Path.rglob
    folder=g.ROOT/target/'kernel'; added=[False]
    def listing(p,pattern):
        yield from original_glob(p,pattern)
        if added[0] and p==folder:
            # Existing real file makes is_file() true, but is absent from the
            # authenticated kernel inventory. No production files are written.
            yield g.ROOT/'tools/strict_variant_gate_r224_v144.py'
    def evaluated(*a,**kw):
        result=original_eval(*a,**kw); added[0]=True; return result
    monkeypatch.setattr(Path,'rglob',listing)
    monkeypatch.setattr(r.v143,'evaluate',evaluated)
    with pytest.raises(ValueError,match='kernel file set changed'):
        r.verify_receipt(g.ROOT/'.local/state/gates/knee-lb-r224flipq-v1.json',g.Inputs(g.ROOT))


def test_consumer_replays_generic_receipt(candidate,monkeypatch):
    c=candidate; monkeypatch.setattr(r,'canonical_stage',lambda *a:True)
    bind(c); result=r.evaluate(c['output'],c['review']); assert result['status']=='PASS'
    path,data=virtual_receipt(c,result,monkeypatch)
    def consume(): return ks._strict_gate(c['slug'],{'gate_sha256':g.sha(data.read_bytes())},{})
    assert consume()==result
    for key,value in [('fixture_only',True),('reviewer_lane','forged'),('input_sha256',{}),('kernel_id','taeyangg4/wrong-child')]:
        bad=copy.deepcopy(result); bad[key]=value; put(data,bad)
        with pytest.raises(ValueError): consume()
    put(data,result)
    p=c['output']/'allowlist.json'; p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError): consume()


def test_generic_parent_chain_and_cycle_refusal(candidate,monkeypatch):
    c=candidate; monkeypatch.setattr(r,'canonical_stage',lambda *a:True)
    bind(c); result=r.evaluate(c['output'],c['review']); assert result['status']=='PASS'
    path,_=virtual_receipt(c,result,monkeypatch)
    assert r.verify_receipt(path,g.Inputs(g.ROOT))==result
    with pytest.raises(ValueError,match='cyclic'): r.verify_receipt(path,g.Inputs(g.ROOT),(path,))
    # A second proposal consumes the generic admitted copy without new gate code.
    old=g.kernel(c['output'],g.Inputs(g.ROOT))
    source=c['tmp']/'second'; source.mkdir()
    ancestor=g.kernel(g.ROOT/PARENT,g.Inputs(g.ROOT))
    for n,b in ancestor[3].items(): put(source/'kernel'/n,b)
    ident='taeyangg4/future-chain-fixture-v1'
    meta=copy.deepcopy(ancestor[0]); meta.update(id=ident,title=ident.split('/')[1]); put(source/'kernel/kernel-metadata.json',meta)
    before={n[9:]:b for n,b in old[1].items() if n.startswith('embedded/')}
    after={n[9:]:b for n,b in ancestor[1].items() if n.startswith('embedded/')}
    changes={n:r.v143.member_diff(n,before[n],after[n]) for n in sorted(before) if before[n]!=after[n]}
    put(source/'new-members.json',dict(added=[],removed=[],changed=[dict(name=n,before_sha256=d['before_sha256'],after_sha256=d['after_sha256']) for n,d in changes.items()],unchanged=[dict(name=n,sha256=g.sha(b)) for n,b in sorted(before.items()) if b==after[n]]))
    put(source/DIFF,''.join(d['unified_diff'] for d in changes.values()).encode())
    second=dict(tmp=c['tmp'],build_name='second-build.md',builder_lane='second-builder',source=source,pin=c['tmp']/'second-pin.json',output=c['tmp']/'second-bound',review=c['tmp']/'second-review.md',slug=ident,parent=c['output'],proof=None)
    reseal(second); bind(second)
    checked=r.evaluate(second['output'],second['review']); assert checked['status']=='PASS',checked['errors']


def test_preflight_generic_admission(candidate,monkeypatch):
    c=candidate; monkeypatch.setattr(r,'canonical_stage',lambda *a:True)
    bind(c); result=r.evaluate(c['output'],c['review']); assert result['status']=='PASS'
    path,data=virtual_receipt(c,result,monkeypatch)
    spec=importlib.util.spec_from_file_location('generic_preflight',g.ROOT/'.local/run-scripts/b-preflight-v106.py')
    b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    original_bytes=Path.read_bytes; original_stat=Path.stat
    monkeypatch.setattr(Path,'read_bytes',lambda p: original_bytes(data if p.resolve()==path else p))
    monkeypatch.setattr(Path,'stat',lambda p,*a,**kw: original_stat(data if p.absolute()==path else p,*a,**kw))
    original_read=b.read
    monkeypatch.setattr(b,'read',lambda p: g.parse(data.read_bytes()) if b.local(p)==path else original_read(p))
    original_sha=b.sha
    monkeypatch.setattr(b,'sha',lambda p: g.sha(data.read_bytes()) if b.local(p)==path else original_sha(p))
    meta=g.parse((c['output']/'kernel/kernel-metadata.json').read_bytes())
    b.gate_check(c['output'],meta)
    bad=copy.deepcopy(result); bad['reviewer_lane']='forged'; put(data,bad)
    with pytest.raises(b.CheckFailed): b.gate_check(c['output'],meta)


@pytest.mark.parametrize('mutation',['missing','wrong','duplicate','archive'])
def test_builder_authentication(candidate,mutation):
    c=candidate
    report=c['tmp']/'build.md'; raw=report.read_bytes()
    text=b'no kernel binding\n' if mutation=='missing' else (b'Kernel-SHA256: '+b'0'*64+b'\n' if mutation=='wrong' else raw+raw)
    if mutation=='archive': report.write_bytes(raw+b'changed\n')
    else: c['builder']=dispatched(c['tmp'],'generic-builder',report,text)
    with pytest.raises(ValueError): pin(c)


def test_transitive_tool_hash_is_pinned(candidate,monkeypatch):
    c=candidate; bind(c)
    name=r.relative(g.hb151.__file__,g.ROOT)
    saved=g.parse(c['pin'].read_bytes())
    assert name in saved['provenance_sha256']
    result=r.evaluate(c['output'],c['review'])
    assert name in result['input_sha256']
    original=Path.read_bytes
    def changed(p):
        data=original(p)
        return data+b'\n' if p.resolve()==Path(g.hb151.__file__).resolve() else data
    monkeypatch.setattr(Path,'read_bytes',changed)
    assert r.evaluate(c['output'],c['review'])['status']=='FAIL'


def test_frozen_files_unchanged():
    backup=g.ROOT/'.local/backups/strict-variant-gate-v1.4.4'
    for name,digest in g.parse((backup/'SHA256.json').read_bytes()).items():
        assert g.sha((g.ROOT/name).read_bytes())==digest
        assert (g.ROOT/name).read_bytes()==(backup/Path(name).name).read_bytes()
