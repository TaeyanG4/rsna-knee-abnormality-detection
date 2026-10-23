"""Strict HB151 kernel scope and CPU-only qualification; no account/GPU operations."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import hb151_killtest_v1 as b
from tools.hb151_killtest_cpu_fixtures import case_specs, run_case, loader_smoke


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    import socket
    monkeypatch.setattr(socket, 'create_connection', lambda *a,**k: pytest.fail('network forbidden'))


@pytest.mark.parametrize('name,kwargs', case_specs(), ids=[n for n,_ in case_specs()])
def test_fast_cases(tmp_path, name, kwargs):
    assert run_case(tmp_path, **kwargs)['status'] == 'PASS'


def test_authentication_and_full_scope():
    assert b.validate()['runtime_files_identical'] == 2
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


def test_bind_placeholder_and_mismatch(tmp_path):
    ck=tmp_path/b.NAME; ck.write_bytes(b.PLACEHOLDER)
    pin=ck.with_suffix('.sha256'); pin.write_text(b.digest(b.PLACEHOLDER)+'  '+b.NAME+'\n')
    with pytest.raises(ValueError,match='placeholder'): b.bind(ck,pin)
    pin.write_text('0'*64+'  '+b.NAME+'\n')
    with pytest.raises(ValueError,match='mismatch'): b.bind(ck,pin)


def test_export_must_have_hb151_provenance(tmp_path):
    ck=tmp_path/b.NAME; ck.write_bytes(b'not HB151')
    pin=ck.with_suffix('.sha256'); pin.write_text(b.sha(ck)+'  '+b.NAME+'\n')
    ck.with_suffix('.json').write_text('{}')
    with pytest.raises(ValueError,match='provenance'): b.bind(ck,pin)


def test_actual_cpu_checkpoint_rejected_by_frozen_provenance():
    # Existing completed CPU fixture, never the running fit output.
    fixture=b.ROOT/'.local/scratch/astra-b4/H-B-151/v3/cpu-checkpoint.pt'
    with pytest.raises(ValueError,match='protocol SHA mismatch'): b.compatibility(fixture)
    m=b.runtime(); ck=m.safe_load(fixture)
    # Even a new manifest protocol pin alone cannot satisfy the frozen run-dir check.
    with pytest.raises(ValueError,match='run mismatch'):
        m.own_meta(ck,ck['provenance']['protocol_sha256'])
    assert m.state_schema(ck['model'])==b.read(b.PARENT/'assets/release-manifest.json')['own_state_schema']


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


@pytest.mark.parametrize('fault', ['none','missing-summary','partial','canonical','summary-hash',
                                  'protocol','run-dir','seed','exposure','checkpoint'])
def test_completed_fit_binding(tmp_path, monkeypatch, fault):
    # All final-fit paths are synthetic. Never read the running fit outputs.
    from types import SimpleNamespace
    from tools import hb151_build_v3 as fit
    protocol = tmp_path/'protocol.json'
    protocol.write_bytes(b.encoded({'run_dirs': {'local': 'synthetic/S42'}}))
    monkeypatch.setattr(fit, 'PROTOCOL', protocol)
    monkeypatch.setattr(fit, 'RUN', tmp_path)
    ck = tmp_path/b.NAME; ck.write_bytes(b'completed synthetic checkpoint')
    digest = b.sha(ck)
    pin = ck.with_suffix('.sha256'); pin.write_text(digest+'  '+b.NAME+'\n')
    ck.with_suffix('.json').write_bytes(b.encoded(dict(sha256=digest, exposure='EXPOSED',
        clean_oof=False, protocol_sha256=b.sha(protocol), serving_name=b.NAME)))
    (tmp_path/'model_best.pt').write_bytes(ck.read_bytes() if fault != 'canonical' else b'wrong')
    identity = dict(protocol_sha256=b.sha(protocol), run_dir='synthetic/S42', seed=42)
    summary = dict(status='COMPLETE_TRAINING_NO_EVALUATION', model_sha256=digest,
                   identity=identity, exposure='EXPOSED', clean_oof=False)
    if fault == 'partial': summary['status'] = 'INCOMPLETE'
    if fault == 'summary-hash': summary['model_sha256'] = '0'*64
    if fault == 'protocol': identity['protocol_sha256'] = '0'*64
    if fault == 'run-dir': identity['run_dir'] = 'older/S42'
    if fault == 'seed': identity['seed'] = 43
    if fault == 'exposure': summary['clean_oof'] = True
    if fault != 'missing-summary': (tmp_path/'summary.json').write_bytes(b.encoded(summary))
    payload = dict(exposure='EXPOSED', clean_oof=False, provenance=dict(
        protocol_sha256=b.sha(protocol), run_dir='synthetic/S42'))
    if fault == 'checkpoint': payload['provenance']['run_dir'] = 'older/S42'
    monkeypatch.setattr(b, 'runtime', lambda: SimpleNamespace(safe_load=lambda _:payload))
    calls = []
    monkeypatch.setattr(b, 'build', lambda path: calls.append(path) or {'status':'REACHED_FROZEN_LOADER_GATE'})
    if fault == 'none':
        assert b.bind(ck,pin)['status'] == 'REACHED_FROZEN_LOADER_GATE'
        assert calls == [ck]
    else:
        with pytest.raises((ValueError,FileNotFoundError)): b.bind(ck,pin)
        assert calls == []
