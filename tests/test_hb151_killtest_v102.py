"""Strict HB151 kernel scope and CPU-only qualification; no account/GPU operations."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import hb151_killtest_v102 as b
from tools.hb151_killtest_v102_fixtures import case_specs, run_case, loader_smoke


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    import socket
    monkeypatch.setattr(socket, 'create_connection', lambda *a,**k: pytest.fail('network forbidden'))


@pytest.mark.parametrize('name,kwargs', case_specs(), ids=[n for n,_ in case_specs()])
def test_fast_cases(tmp_path, name, kwargs):
    assert run_case(tmp_path, **kwargs)['status'] == 'PASS'


def test_authentication_and_full_scope():
    assert b.validate()['runtime_files_identical'] == 1
    assert b.authenticate_parent_archive()['status'] == 'PASS'
    nb = b.read(b.OUT/'kernel/release.ipynb')
    assert nb['cells'][0] == b.read(b.DONOR/'kernel/release.ipynb')['cells'][0]
    before,_ = b.parent()
    assert nb['cells'][1:5] == before['cells'][:4]


@pytest.mark.parametrize('mutation', ['fast','guard','markdown','extra-cell','metadata','runtime','manifest'])
def test_forbidden_diff(mutation):
    r=b.read(b.OUT/'build-receipt.json')
    nb,meta=b.payload(r['checkpoint_sha256'],r['archive_sha256'])
    hashes,small=b.archive_files(b.OUT/'dataset'/b.ARCHIVE)
    if mutation=='fast': nb['cells'][0]['source'].append('\nFAST=True\n')
    if mutation=='guard': nb['cells'][5]['source']=nb['cells'][5]['source'].replace('if not FAST:', 'if True:')
    if mutation=='markdown': nb['cells'][1]['source']+=' changed'
    if mutation=='extra-cell': nb['cells'].append(copy.deepcopy(nb['cells'][0]))
    if mutation=='metadata': meta['enable_internet']=True
    if mutation=='runtime': small['public-serving.py']+=b'\n# changed\n'
    if mutation=='manifest':
        m=json.loads(small['release-manifest.json']); m['policy']['chunk_windows']=32
        small['release-manifest.json']=b.encoded(m)
    with pytest.raises(ValueError): b.scope(nb,meta,hashes,small,r['archive_sha256'])


def test_reject_partial_parent(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'PARENT',tmp_path)
    with pytest.raises((ValueError,FileNotFoundError)): b.parent()


@pytest.mark.parametrize('score,expected',[(.950,'alive'),(.947,'dead'),(.949,'seed43'),(.946,'dead'),(.951,'alive')])
def test_registered_decisions(score,expected):
    assert b.outcome(score)==expected


def test_real_loader_prefix():
    assert loader_smoke(b.SCRATCH/'loader-test')['status']=='PASS'


def test_placeholder_not_admissible():
    assert b.admit()['status']=='HOLD_PLACEHOLDER'
    assert b.admit()['qualified_for_submission'] is False


def test_bind_mismatch_before_live_inputs(tmp_path):
    ck=tmp_path/b.NAME; ck.write_bytes(b'placeholder')
    pin=ck.with_suffix('.sha256'); pin.write_text('0'*64+'  '+b.NAME+'\n')
    with pytest.raises(ValueError,match='mismatch'): b.bind(ck,pin,source='seed42')


def test_conflicting_pins(tmp_path):
    ck=tmp_path/b.NAME; ck.write_bytes(b'placeholder')
    ck.with_suffix('.sha256').write_text(b.sha(ck)+'  '+b.NAME+'\n')
    Path(str(ck)+'.sha256').write_text('0'*64+'  '+b.NAME+'\n')
    with pytest.raises(ValueError,match='conflicting'): b.bind(ck,source='seed42')


def test_older_cpu_checkpoint_rejected():
    fixture=b.ROOT/'.local/scratch/astra-b4/H-B-151/v3/cpu-checkpoint.pt'
    with pytest.raises(ValueError,match='protocol SHA mismatch'):
        b.compatibility(fixture,b.binding())


def test_retry_windows_publication(tmp_path,monkeypatch):
    real=b.os.replace; calls=[]
    def flaky(a,c):
        calls.append(1)
        if len(calls)<3: raise PermissionError('sharing violation')
        return real(a,c)
    monkeypatch.setattr(b.os,'replace',flaky)
    monkeypatch.setattr(b.time,'sleep',lambda _:None)
    b.write(tmp_path/'receipt.json',b'{}')
    assert len(calls)==3 and (tmp_path/'receipt.json').read_bytes()==b'{}'


def test_persistent_windows_failure_bounded(monkeypatch):
    calls=[]
    def fail():
        calls.append(1); raise PermissionError('locked')
    monkeypatch.setattr(b.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError): b.retry(fail)
    assert len(calls)==40


@pytest.mark.parametrize('state',['missing','stale','failed'])
def test_admit_review_bypass_rejected(state, monkeypatch):
    # Keep real artifacts and hashes, simulate only the missing/invalid review record.
    real=b.read
    def reader(path):
        if str(path)=='review.json':
            if state=='missing': raise FileNotFoundError(path)
            return dict(verdict='FAIL' if state=='failed' else 'PASS',independent=True,reviewer='other',
                        build_receipt_sha256='stale',smoke_sha256='stale')
        value=real(path)
        if Path(path)==b.OUT/'build-receipt.json': value=dict(value,placeholder=False,fixture=False)
        return value
    monkeypatch.setattr(b,'validate',lambda:None)
    monkeypatch.setattr(b,'read',reader)
    with pytest.raises((ValueError,FileNotFoundError)): b.admit('review.json')


def test_missing_smoke_blocks(monkeypatch):
    monkeypatch.setattr(b,'validate',lambda:None)
    with patch.object(b,'read',side_effect=FileNotFoundError('missing smoke')):
        with pytest.raises(FileNotFoundError): b.admit()


@pytest.mark.parametrize('field,value', [('arch','wrong'),('res',320),('seed',43),
    ('checkpoint_kind','wrong'),('swa_epochs_1based',[1]),('exposure','CLEAN'),('clean_oof',True)])
def test_metadata_checks_preserved(field,value,monkeypatch):
    m=b.runtime()
    ck=m.safe_load(b.SCRATCH/'placeholder-v102'/b.NAME)
    ck[field]=value
    monkeypatch.setattr(b,'runtime',lambda:m)
    monkeypatch.setattr(m,'safe_load',lambda _:ck)
    with pytest.raises(ValueError): b.compatibility(Path('unused'),b.binding())


def test_exact_loader_wrong_sha_before_model_construction():
    m=b.runtime()
    path=b.SCRATCH/'placeholder-v102'/b.NAME
    manifest=b.manifest(b.read(b.OUT/'build-receipt.json')['checkpoint_sha256'])
    manifest['checkpoints'][b.NAME]['sha256']='0'*64
    ns={'_E152_MANIFEST':manifest,'_E152_FILES':{b.NAME:path},
        'build_backbone':lambda *a,**kw:pytest.fail('constructed before hash gate')}
    with pytest.raises(ValueError,match='Bound checkpoint SHA mismatch'):
        m.checked_model(ns,path,m.ARCH,384,'cpu')


def test_end_to_end_fixture_bind():
    from tools.hb151_killtest_v102_fixtures import fixture_bind_smoke
    assert fixture_bind_smoke()['status']=='PASS'


def test_admit_cli_holds_with_nonzero_exit(monkeypatch):
    monkeypatch.setattr(b.sys,'argv',['hb151_killtest_v102.py','admit','--dry-run'])
    with pytest.raises(SystemExit) as exc:
        b.main()
    assert exc.value.code==2


def test_wrong_state_schema_rejected(monkeypatch):
    m=b.runtime()
    ck=m.safe_load(b.SCRATCH/'placeholder-v102'/b.NAME)
    ck['model'].pop('clsW')
    monkeypatch.setattr(b,'runtime',lambda:m)
    monkeypatch.setattr(m,'safe_load',lambda _:ck)
    with pytest.raises(ValueError,match='state schema mismatch'):
        b.compatibility(Path('unused'),b.binding())


def test_runtime_only_authorized_provenance_lines():
    original=(b.PARENT/'assets'/b.RUNTIME_NAME).read_bytes()
    expected=original.replace(b'own_meta(ck, manifest["protocol_sha256"])',
        b'own_meta(ck, manifest["protocol_sha256"], run_dir=manifest["hb151_export"]["run_dir"], seed=manifest["hb151_export"]["seed"])')
    expected=expected.replace(b'def own_meta(ck, protocol_sha, run_dir=RUN):',
                              b'def own_meta(ck, protocol_sha, run_dir=RUN, seed=42):')
    expected=expected.replace(b'swa_epochs_1based=EPOCHS, seed=42)', b'swa_epochs_1based=EPOCHS, seed=seed)')
    assert b.runtime_bytes()==expected
    from tools import hb151_killtest_v101 as old
    hashes,small=b.archive_files(b.OUT/'dataset'/b.ARCHIVE)
    _,prior=old.archive_files(old.OUT/'dataset'/old.ARCHIVE)
    assert small['public-serving.py']==prior['public-serving.py']==(b.PARENT/'assets/public-serving.py').read_bytes()
    assert b.read(b.OUT/'kernel/release.ipynb')['cells'][0]==old.read(old.OUT/'kernel/release.ipynb')['cells'][0]


@pytest.mark.parametrize('selected', ['seed42','seed43'])
@pytest.mark.parametrize('fault', ['none','sha','mixed-sidecar','mixed-manifest','checkpoint-protocol',
    'checkpoint-seed','checkpoint-run','clean','exposure','missing-pin','summary','canonical'])
def test_selected_export_binding(tmp_path,monkeypatch,selected,fault):
    from types import SimpleNamespace
    spec=b.source_spec(selected)
    other=b.source_spec('seed43' if selected=='seed42' else 'seed42')
    ck=tmp_path/b.NAME
    ck.write_bytes(b'fixture '+selected.encode())
    digest=b.sha(ck)
    pin=ck.with_suffix('.sha256')
    pin.write_text(('0'*64 if fault=='sha' else digest)+'  '+b.NAME+'\n')
    meta=dict(sha256=digest,protocol_sha256=spec['protocol_sha256'],exposure='EXPOSED',
              clean_oof=False,serving_name=b.NAME)
    if fault=='mixed-sidecar': meta['protocol_sha256']=other['protocol_sha256']
    if fault=='clean': meta['clean_oof']=True
    if fault=='exposure': meta['exposure']='CLEAN'
    ck.with_suffix('.json').write_bytes(b.encoded(meta))
    identity=dict(seed=spec['seed'],protocol_sha256=spec['protocol_sha256'],run_dir=spec['run_dir'])
    payload=dict(provenance=dict(identity),seed=spec['seed'],exposure='EXPOSED',clean_oof=False)
    if fault=='checkpoint-protocol': payload['provenance']['protocol_sha256']=other['protocol_sha256']
    if fault=='checkpoint-seed': payload['seed']=other['seed']
    if fault=='checkpoint-run': payload['provenance']['run_dir']=other['run_dir']
    if fault=='mixed-manifest': monkeypatch.setitem(b.SOURCES,selected,dict(spec,protocol=other['protocol']))
    if fault=='missing-pin': pin.unlink()
    (tmp_path/'summary.json').write_bytes(b.encoded(dict(status='INCOMPLETE' if fault=='summary' else
        'COMPLETE_TRAINING_NO_EVALUATION',model_sha256=digest,identity=identity,exposure='EXPOSED',clean_oof=False)))
    (tmp_path/'model_best.pt').write_bytes(b'wrong' if fault=='canonical' else ck.read_bytes())
    monkeypatch.setattr(b,'runtime',lambda:SimpleNamespace(safe_load=lambda _:payload))
    calls=[]
    monkeypatch.setattr(b,'build',lambda path,**kwargs:calls.append(kwargs) or {'status':'ACCEPT'})
    if fault=='none' or (selected=='seed43' and fault in ['summary','canonical']):
        assert b.bind(ck,source=selected,fixture=True)['status']=='ACCEPT'
        assert calls[0]['identity']['source']==selected
        assert calls[0]['identity']['seed']==spec['seed']
    else:
        with pytest.raises((ValueError,FileNotFoundError)): b.bind(ck,source=selected,fixture=True)
        assert not calls


def test_selection_required():
    with pytest.raises(ValueError,match='explicit --source'): b.bind()


@pytest.mark.parametrize('selected',['seed42','seed43'])
def test_final_placeholder_rejected(selected,monkeypatch):
    from tools.hb151_killtest_v102_fixtures import placeholder_export
    ck,_=placeholder_export(selected)
    spec=b.source_spec(selected)
    monkeypatch.setitem(b.SOURCES,selected,dict(spec,export=ck))
    # Map only the completed seed42 summary/canonical inputs to the fixture.
    original_read,original_sha=b.read,b.sha
    run=b.ROOT/spec['run_dir']
    monkeypatch.setattr(b,'read',lambda p:original_read(ck.parent/'summary.json' if Path(p)==run/'summary.json' else p))
    monkeypatch.setattr(b,'sha',lambda p:original_sha(ck.parent/Path(p).name
        if Path(p) in (run/'model_best.pt',run/'summary.json') else p))
    with pytest.raises(ValueError,match='placeholder cannot'): b.bind(source=selected)


@pytest.mark.parametrize('selected',['seed42','seed43'])
def test_arbitrary_export_path_rejected(tmp_path,selected):
    ck=tmp_path/b.NAME; ck.write_bytes(b'not the pinned location')
    ck.with_suffix('.sha256').write_text(b.sha(ck)+'  '+b.NAME+'\n')
    with pytest.raises(ValueError,match='selected pinned export path'): b.bind(ck,source=selected)


def test_source_cli_forwarding(monkeypatch):
    seen=[]
    monkeypatch.setattr(b,'bind',lambda ck,pin,**kw:seen.append((ck,pin,kw)) or {'status':'PASS'})
    monkeypatch.setattr(b.sys,'argv',['builder','bind','--source','seed43'])
    b.main()
    assert seen==[(None,None,{'source':'seed43'})]
