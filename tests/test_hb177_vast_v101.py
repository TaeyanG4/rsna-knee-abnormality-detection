"""Vast transport, environment and admission regressions (CPU only)."""
import copy
import io
import json
import tarfile
from pathlib import Path
import pytest
from tools import hb177_vast_v101 as v
from tools import hb177_build_v101 as b
from tools import hb177_build_v1 as old


def test_recipe_and_preservation():
    assert b.VERSION=='1.0.1' and b.RECIPE==old.RECIPE
    pins=v.read(b.STAGE/'v100-preservation.json')
    assert len(pins)>=13
    assert all(v.sha(b.ROOT/p)==h for p,h in pins.items())


def smoke():
    return dict(device_name='NVIDIA GeForce RTX 4090',total_mib=49140,desktop_used_mib=100,
        train_updates=3,train_shape=[1,12,3,384,384],accumulation=16,serving_views=2,serving_k=94,
        precision='bf16',grad_checkpointing=True,train_peak_reserved_bytes=12*1024**3,
        serve_peak_reserved_bytes=11*1024**3,seconds_per_update=5.)


def test_memory_identity_and_headroom():
    b.check_memory(smoke())
    for change in ({'total_mib':24564},{'device_name':'RTX 4070 Ti SUPER'},
                   {'train_peak_reserved_bytes':12*1024**3+1},{'desktop_used_mib':40000},
                   {'accumulation':8},{'train_updates':2},{'serving_k':12},
                   {'seconds_per_update':float('nan')}):
        with pytest.raises(ValueError): b.check_memory(dict(smoke(),**change))


def env():
    return dict(python='3.12.6',platform='Windows',cuda='12.1',cudnn=90100,
                packages={'torch':'2.5.1+cu121','torchvision':'0.20.1','timm':'1.0.19'},
                imports={},settings={'cudnn_benchmark':False})


def test_environment_allowed_and_blocked_differences():
    a=env(); z=copy.deepcopy(a)
    z.update(python='3.11.10',platform='Linux'); z['packages']['torchvision']='0.20.1+cu121'
    differences,mismatch=v.environment_diff(a,z)
    assert not mismatch and set(differences)=={'python','platform','packages'}
    z['packages']['timm']=None
    assert 'timm' in v.environment_diff(a,z)[1]
    z['settings']['cudnn_benchmark']=True
    assert 'settings' in v.environment_diff(a,z)[1]
    z['python']='3.10.0'
    with pytest.raises(ValueError): v.environment_diff(a,z)


@pytest.mark.parametrize('name',['../x','/x','C:/x','a\\x','a\nx'])
def test_unsafe_paths(name):
    with pytest.raises(ValueError): v.safe_name(name)


def test_manifest_tampering_and_duplicates(tmp_path):
    p=tmp_path/'a'; p.write_bytes(b'payload')
    item=dict(path='a',bytes=7,sha256=v.sha(p)); m={'files':[item]}
    assert v.verify_manifest(tmp_path,m)['files']==1
    with pytest.raises(ValueError,match='duplicate'): v.verify_manifest(tmp_path,{'files':[item,item]})
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatch'): v.verify_manifest(tmp_path,m)


def test_results_roundtrip_and_reject_unlisted(tmp_path):
    payload=b'fixture'; h=__import__('hashlib').sha256(payload).hexdigest()
    archive=tmp_path/'out.tar'
    def make(extra=False):
        manifest=json.dumps({'files':[dict(path='fit/result.pt',bytes=len(payload),sha256=h)]}).encode()
        with tarfile.open(archive,'w') as tar:
            for name,data in [('fit/result.pt',payload),('results.json',manifest)]+([('extra',b'x')] if extra else []):
                info=tarfile.TarInfo(name); info.size=len(data); tar.addfile(info,io.BytesIO(data))
    make()
    assert v.unpack_results(archive,v.sha(archive),tmp_path/'returned')['bytes']==len(payload)
    with pytest.raises(ValueError,match='SHA256'): v.unpack_results(archive,'bad',tmp_path/'returned')
    make(True)
    with pytest.raises(ValueError,match='unmanifested'): v.unpack_results(archive,v.sha(archive),tmp_path/'returned')


def test_gpu_admission_requires_exact_remote_and_review(monkeypatch,tmp_path):
    monkeypatch.setattr(b,'registration',lambda:None)
    monkeypatch.setattr(b,'require_cpu',lambda:None)
    monkeypatch.setattr(b,'ROOT',tmp_path)
    calls=[]
    monkeypatch.setattr(b,'require_gpu_job',lambda *a,**k:calls.append(True))
    with pytest.raises(ValueError,match='Vast-only'): b.supervise('fit')
    assert not calls
    binding={'protocol_sha256':'x','environment_sha256':'y'}
    review=dict(verdict='PASS',independent=True,reviewer='other',scope='pre-execution',**binding)
    b.check_review(review,binding,'pre-execution')
    with pytest.raises(ValueError,match='stale'): b.check_review(review,dict(binding,environment_sha256='z'),'pre-execution')


def test_remote_commands_and_no_launch_during_cpu():
    for action in ('remote-install','remote-cpu','launch-smoke','launch-fit','status'):
        cmd=v.remote_command(action)
        assert cmd[:len(v.SSH)]==v.SSH and 'StrictHostKeyChecking=yes' in cmd
        assert v.REMOTE in cmd[-1]
    cpu=v.remote_command('remote-cpu')[-1]
    assert 'gpu_queue' not in cpu and 'admit-dry-run' in cpu
    assert 'fit-queue.json' in v.remote_command('launch-fit')[-1]
    assert '--foreground' not in v.remote_command('launch-fit')[-1]
    assert 'pip install -r' in v.remote_command('remote-install')[-1]


def test_running_results_are_not_read(monkeypatch,tmp_path):
    from tools import job_runtime
    monkeypatch.setattr(v,'STAGE',tmp_path)
    monkeypatch.setattr(job_runtime,'job_status',lambda *a,**k:{'state':'running'})
    with pytest.raises(ValueError,match='completed successful'): v.result_pack('fit')


def test_secret_scan_suppresses_match(tmp_path):
    v.scan(Path(v.__file__))
    p=tmp_path/'source.py'; p.write_text('token='+('gh'+'p_')+'a'*25)
    with pytest.raises(ValueError,match='value suppressed') as exc: v.scan(p)
    assert 'a'*25 not in str(exc.value)


def test_send_sealed_archive_and_exact_host(monkeypatch,tmp_path):
    monkeypatch.setattr(v,'ROOT',tmp_path); monkeypatch.setattr(v,'STAGE',tmp_path)
    payload=tmp_path/'code.py'; payload.write_bytes(b'pass\n')
    v.write(tmp_path/'transfer-inventory.json',{'files':[dict(path='code.py',bytes=5,sha256=v.sha(payload))]})
    archive=tmp_path/'transfer.tar'; archive.write_bytes(b'fixture')
    (tmp_path/'transfer.tar.sha256').write_text(v.sha(archive))
    calls=[]; monkeypatch.setattr(v,'run',lambda args:calls.append(args))
    v.send()
    assert len(calls)==3 and 'test ! -e' in calls[0][-1]
    assert calls[1][0]=='scp' and v.HOST in calls[1][-1]
    assert 'sha256sum --quiet -c' in calls[2][-1]
    archive.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='archive drift'): v.send()
    assert len(calls)==3


def test_environment_receipt_live_drift(monkeypatch,tmp_path):
    monkeypatch.setattr(v,'STAGE',tmp_path)
    v.write(tmp_path/'local-environment.json',{'reference':True})
    actual=env(); actual['packages']={name:'1.0' for name in v.PACKAGES}
    v.write(tmp_path/'remote-environment.json',dict(status='PASS_ENV_CPU',actual=actual,
                                                 local_sha256=v.sha(tmp_path/'local-environment.json')))
    monkeypatch.setattr(v.platform,'python_version',lambda:actual['python'])
    monkeypatch.setattr(v.platform,'system',lambda:'Linux')
    monkeypatch.setattr(v.metadata,'version',lambda name:'1.0')
    v.require_environment()
    monkeypatch.setattr(v.metadata,'version',lambda name:'changed' if name=='timm' else '1.0')
    with pytest.raises(ValueError,match='package drift'): v.require_environment()


def test_approval_transfer_missing_fails_before_network(monkeypatch,tmp_path):
    monkeypatch.setattr(v,'STAGE',tmp_path)
    calls=[]; monkeypatch.setattr(v,'run',lambda args:calls.append(args))
    with pytest.raises(ValueError,match='missing independent'): v.send_approval()
    assert not calls


def test_remote_review_and_fit_b_authorization(monkeypatch,tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(b,'os',SimpleNamespace(name='posix'))
    monkeypatch.setattr(b,'ROOT',tmp_path); monkeypatch.setattr(b,'REMOTE_ROOT',tmp_path)
    monkeypatch.setattr(b,'STAGE',tmp_path); monkeypatch.setattr(b,'PROTOCOL',tmp_path/'registration.json')
    monkeypatch.setattr(b,'registration',lambda:None); monkeypatch.setattr(b,'require_cpu',lambda:None)
    monkeypatch.setattr(v,'require_environment',lambda:None)
    for name in ('registration.json','cpu-validation.json','remote-environment.json','cuda-runtime.json'):
        v.write(tmp_path/name,{'fixture':True})
    with pytest.raises(FileNotFoundError): b.require_admission()
    pre=dict(verdict='PASS',independent=True,reviewer='other',scope='pre-execution',**b.review_bindings())
    v.write(tmp_path/'pre-review.json',pre)
    auth=dict(authorized_by='B',protocol_sha256=v.sha(b.PROTOCOL),pre_review_sha256=v.sha(tmp_path/'pre-review.json'),allow_smoke=True)
    v.write(tmp_path/'b-authorization.json',auth); b.require_admission()
    v.write(tmp_path/'cuda-smoke.json',dict(smoke(),protocol_sha256=v.sha(b.PROTOCOL)))
    v.write(tmp_path/'smoke-attempt/receipt.json',dict(exit_code=0,reason='completed',survivors=[]))
    review=dict(verdict='PASS',independent=True,reviewer='other',scope='smoke-result-and-fit',**b.review_bindings(True))
    v.write(tmp_path/'fit-review.json',review)
    with pytest.raises(ValueError,match='B fit authorization'): b.require_admission(True)
    auth.update(allow_fit=True,fit_review_sha256=v.sha(tmp_path/'fit-review.json'))
    v.write(tmp_path/'b-authorization.json',auth); b.require_admission(True)
    v.write(tmp_path/'cuda-runtime.json',{'drift':True})
    with pytest.raises(ValueError,match='stale review'): b.require_admission(True)
