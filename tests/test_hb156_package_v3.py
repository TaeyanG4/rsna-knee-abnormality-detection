"""Seed43 exact-byte parent proof and fail-closed registration tests."""
import ast
import copy
from pathlib import Path
import pytest
from tools import hb156_package_v3 as p
from tools import hb156_vast_v3 as g


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(g,'ROOT',p.PACKAGE)
    monkeypatch.setattr(g,'STAGE',p.STAGE)
    monkeypatch.setattr(g,'PROTOCOL',p.STAGE/'manifest.json')
    return g.load_registration(g.PROTOCOL)


def test_complete_parent_proof_and_seed_only(registered):
    report=p.proof()
    assert report['status']=='PASS' and len(report['changed'])==4
    before=p.read(p.PARENT/p.OLD_STAGE/'manifest.json')
    assert bool(registered==p.expected_manifest(before))
    assert bool(registered['recipe']==dict(before['recipe'],seed_local=43))
    assert p.sha(p.STAGE/registered['label_file_name'])==p.LABEL_SHA
    for item in report['unchanged']:
        assert p.sha(p.PACKAGE/item['candidate_path'])==item['parent_sha256']
    assert (p.STAGE/'seed42-manifest.json').read_bytes()==(p.PARENT/p.OLD_STAGE/'seed42-manifest.json').read_bytes()
    assert not (p.STAGE/'S42').exists()


@pytest.mark.parametrize('field',['seed','labels','roster','weight','budget','init','binding'])
def test_rehashed_registration_drift_refused(registered,monkeypatch,field):
    mutated=copy.deepcopy(registered)
    if field=='seed': mutated['recipe']['seed_local']=42
    if field=='labels': mutated['train_labels'][mutated['train_ids'][0]][0]=-.01
    if field=='roster': mutated['train_ids'].reverse()
    if field=='weight': mutated['pos_weight'][0]+=1
    if field=='budget': mutated['budgets']['local']+=1
    if field=='init': mutated['init_sha256']='0'*64
    if field=='binding': mutated['source_bindings'].pop('tools/hb156_package_v3.py')
    mutated.pop('content_sha256');mutated['content_sha256']=p.json_hash(mutated)
    original=g.read_json
    monkeypatch.setattr(g,'read_json',lambda path:copy.deepcopy(mutated) if Path(path)==g.PROTOCOL else original(path))
    with pytest.raises(ValueError): g.load_registration(g.PROTOCOL)


def test_seed42_execution_rejected(registered):
    with pytest.raises(ValueError,match='seed43'): g.execution_settings(registered,1,'bf16',42,False)
    assert g.execution_settings(registered,1,'bf16',43,False)['total_steps']==4352


def test_all_seed_sites_and_unchanged_runtime_functions():
    before=(p.PARENT/'tools/hb156_vast_v1.py').read_bytes()
    after=(p.PACKAGE/'tools/hb156_vast_v1.py').read_bytes()
    assert after==p.runtime(before)
    trees=[ast.parse(data) for data in (before,after)]
    funcs=[{n.name:ast.dump(n) for n in t.body if isinstance(n,ast.FunctionDef)} for t in trees]
    changed={k for k in funcs[0] if funcs[0][k]!=funcs[1][k]}
    assert changed=={'load_registration','execution_settings','registered_run','cpu_validate','cuda_smoke','ready','main'}
    seed_functions={'execution_settings','cpu_validate','cuda_smoke','ready','main'}
    for node in trees[1].body:
        if isinstance(node,ast.FunctionDef) and node.name in seed_functions:
            assert not any(isinstance(n,ast.Constant) and n.value==42 for n in ast.walk(node)),node.name
    assert b"read_json(STAGE/'seed42-manifest.json')" in after
    assert funcs[0]['public_decision']==funcs[1]['public_decision']


@pytest.mark.parametrize('rel,old,new',[
    ('tools/hb156_vast_v1.py',b'total=165',b'total=166'),
    ('run.sh',b'nohup python',b'nohup python3'),
    ('training/src/hb156_vast_v1.py',b'args.seed',b'42'),
])
def test_proof_rejects_unregistered_byte_changes(monkeypatch,rel,old,new):
    original=Path.read_bytes
    target=p.PACKAGE/rel
    monkeypatch.setattr(Path,'read_bytes',lambda path:original(path).replace(old,new) if path==target else original(path))
    with pytest.raises(ValueError,match='non-allowlisted'): p.proof()


def test_parent_identity_refused(monkeypatch):
    monkeypatch.setattr(p,'PARENT_PIN','0'*64)
    with pytest.raises(ValueError,match='parent inventory identity'): p.parent_inventory()


def test_missing_patch_anchor_refused():
    with pytest.raises(ValueError,match='patch anchor drift'): p.runtime(b'wrong parent')


@pytest.mark.parametrize('state',['S43','budget.json','independent-review.json','smoke-attempt','export'])
def test_rebuild_refuses_execution_state(monkeypatch,tmp_path,state):
    monkeypatch.setattr(p,'STAGE',tmp_path)
    (tmp_path/state).touch()
    with pytest.raises(ValueError,match='execution state'): p.safe_build_target()


def test_transfer_paths_and_parent_pin():
    old=(p.ROOT/'tools/hb156_transfer_v2.ps1').read_bytes()
    new=(p.ROOT/'tools/hb156_transfer_v3.ps1').read_bytes()
    assert p.sha(p.ROOT/'tools/hb156_transfer_v2.ps1')==p.TRANSFER_PIN
    assert p.transfer(old)==new
    assert b'hb156-convnext-l2-s42-v1' not in new
    assert b'hb156-l2-s43\\H-B-156-L2-S43' in new
    assert new.count(b'/workspace/hb156-l2-s43-upload-v1')==3


def test_cpu_receipt_bound_to_seed43(registered):
    receipt=p.read(p.STAGE/'cpu-validation.json')
    assert receipt['status']=='PASS_CPU' and receipt['cuda_executed'] is False
    assert receipt['protocol_sha256']==p.sha(p.STAGE/'manifest.json')
    assert receipt['real_forward'] and receipt['real_strict_checkpoint_roundtrip']
    assert receipt['protocol_sha256']!=p.sha(p.PARENT/p.OLD_STAGE/'manifest.json')


@pytest.mark.parametrize('fault',['none','pin','roster','sums','member','duplicate'])
def test_archive_audit_rejects_tampering(monkeypatch,tmp_path,fault):
    import io,tarfile
    package=tmp_path/'package';package.mkdir()
    (package/'file.txt').write_bytes(b'fixture')
    files=[dict(path='file.txt',bytes=7,sha256=p.sha(package/'file.txt'))]
    p.write_json(package/'INVENTORY.json',dict(files=files))
    p.write(package/'SHA256SUMS',files[0]['sha256']+'  file.txt\n')
    target=tmp_path/(p.DEPLOY+'.tar')
    with tarfile.open(target,'w') as tar:
        for name in ('file.txt','INVENTORY.json','SHA256SUMS'):
            data=(package/name).read_bytes()
            if fault=='member' and name=='file.txt': data=b'altered'
            info=tarfile.TarInfo(p.DEPLOY+'/'+name);info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
        if fault=='duplicate': tar.add(package/'file.txt',arcname=p.DEPLOY+'/file.txt')
    p.write(target.with_suffix('.tar.sha256'),('0'*64 if fault=='pin' else p.sha(target))+'  '+target.name+'\n')
    if fault=='roster': (package/'extra.txt').touch()
    if fault=='sums': p.write(package/'SHA256SUMS','incorrect\n')
    monkeypatch.setattr(p,'PACKAGE',package);monkeypatch.setattr(p,'WORK',tmp_path)
    monkeypatch.setattr(p,'proof',lambda:dict(parent_files_verified=73,unchanged=[],changed=[]))
    if fault=='none': p.validate()
    else:
        with pytest.raises(ValueError): p.validate()
