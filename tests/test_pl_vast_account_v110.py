import json
import pytest
from tools import pl_vast_account_v110 as a


def row(**overrides):
    value = dict(id=a.INSTANCE, machine_id=a.MACHINE, gpu_name='RTX 3090', num_gpus=1,
                 dph_total=.228, actual_status='exited', intended_status='stopped')
    value.update(overrides)
    return value


def test_budget_and_good():
    assert a.budget()['allocation_usd'] == 1.512
    assert a.stop_reason(row(), 9.26, 0) is None
    assert a.stop_reason(row(), 8, 1000, gpu_admitted=True) is None


@pytest.mark.parametrize('r,credit,elapsed,admitted', [
    (row(id=1),9.26,0,False), (row(machine_id=1),9.26,0,False),
    (row(num_gpus=2),9.26,0,False), (row(gpu_name='RTX 4090'),9.26,0,False),
    (row(dph_total=.229),9.26,0,False), (row(dph_total=float('nan')),9.26,0,False),
    (row(),float('inf'),0,False), (row(),5.6,0,False), (row(),6,0,False),
    (row(),9.26,14400,True), (row(),9.26,300,False), (row(),9.26,-1,False)])
def test_fail_closed(r,credit,elapsed,admitted):
    assert a.stop_reason(r,credit,elapsed,gpu_admitted=admitted)


def gpu():
    return dict(gpus=[dict(name='NVIDIA GeForce RTX 3090', uuid='GPU-test',memory_mib=24576)], processes=[])


def test_admission():
    assert a.admit_gpu(gpu(),1000,1050)['status']=='PHYSICAL_GPU_ADMITTED'
    for snapshot,now in [(gpu(),1300),(dict(gpus=[],processes=[]),1050),
                         (dict(gpus=gpu()['gpus'],processes=['python']),1050)]:
        with pytest.raises(ValueError):
            a.admit_gpu(snapshot,1000,now)


class Fake:
    def __init__(self, fails=False): self.fails=fails; self.stops=0
    def stop(self, instance):
        self.stops+=1
        if self.fails: raise RuntimeError('ambiguous')
    def call(self,*args): return row()
    def user(self): return {'credit':9.26}


def test_stop_reconciles_one_request():
    p=Fake(True)
    assert a.stop_and_confirm(p)['status']=='VERIFIED_EXITED_STOPPED'
    assert p.stops==1


def test_nonfinite_sample_serializable():
    class InvalidRate(Fake):
        def call(self,*args): return row(dph_total=float('nan'))
    value=a.sample(InvalidRate(),'credit',None,False)
    assert value['stop_reason']
    assert json.dumps(value,allow_nan=False)


def test_check_cannot_refresh_watch_heartbeat(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'WORK',tmp_path)
    monkeypatch.setattr(a,'load_provider',lambda:(Fake(),{'balance_field':'credit'}))
    assert a.main(['check','--execute-B-only'])==0
    assert (tmp_path/'guard-check.json').exists()
    assert not (tmp_path/'guard-heartbeat.json').exists()


def test_live_disabled(monkeypatch):
    monkeypatch.setattr(a,'load_provider',lambda: pytest.fail('must not load provider'))
    for action in ('start','check','stop','watch','admit-gpu'):
        with pytest.raises(SystemExit): a.main([action])
    assert a.main(['budget'])==0


def test_exclusive_receipt(tmp_path):
    target=tmp_path/'attempt.json'
    a.write(target,{'a':1},exclusive=True)
    with pytest.raises(FileExistsError): a.write(target,{'a':2},exclusive=True)
    assert json.loads(target.read_text())=={'a':1}


def test_start_once_then_no_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'WORK',tmp_path)
    p=Fake()
    monkeypatch.setattr(a,'load_provider',lambda:(p,{'balance_field':'credit'}))
    assert a.main(['start','--execute-B-only'])==0
    with pytest.raises(FileExistsError): a.main(['start','--execute-B-only'])
