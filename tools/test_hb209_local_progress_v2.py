"""Regress real H209v1 failure, sampled multi-study progression and true stalls."""
import pytest
from tools.hb209_local_progress_v2 import Progress
from tools.hb209_local_progress_v1 import Progress as Old

def hb(t, phase='renderer-source-read', completed=0, windows=0, count=1):
    return dict(pid=123, time=t, completed=completed, windows=windows,
                phase=phase, setup_completed=count)

def policy(cls=Progress):
    return cls(0, dict(pid=123, create_time=0), 58)

def test_reproduce_old_failure_actual_observed_counters():
    p=policy(Old)
    with pytest.raises(ValueError):
        p.update(81,hb(81,completed=1,windows=224,count=11126),None,81)

def test_first_sample_can_skip_completed_first_study():
    p=policy()
    assert p.update(81,hb(81,completed=1,windows=224,count=11126),None,81) is None
    assert (p.completed,p.windows,p.rank)==(1,224,0)

def test_multi_study_render_source_read_and_classifier():
    p=policy()
    samples=[hb(1,'configured'),hb(2,'codec-header',count=10),
             hb(3,'renderer-ready',count=11),hb(4,'render',count=11),
             hb(5,count=12),hb(6,'classifier',count=12),
             hb(7,'classifier-chunk',windows=224,count=12),
             hb(8,completed=1,windows=224,count=13),
             hb(9,'classifier',completed=1,windows=224,count=13),
             hb(10,'classifier-chunk',completed=1,windows=448,count=13),
             hb(11,'complete',completed=58,windows=13000,count=13)]
    for h in samples: assert p.update(h['time'],h,None,h['time']) is None
    assert p.completed==58 and p.windows==13000

def test_same_study_real_reads_extend_liveness_beyond_900():
    p=policy()
    for t in range(100,2001,100):
        assert p.update(t,hb(t,count=t//100),None,t) is None
    assert p.windows==0 and p.active==0

def test_timestamps_without_real_work_still_timeout():
    p=policy();p.update(1,hb(1),None,1)
    assert p.update(902,hb(902),None,902) is not None

@pytest.mark.parametrize('change',[dict(pid=9),dict(time=-2),dict(time=99),
    dict(setup_completed=0),dict(windows=-1),dict(completed=-1),
    dict(completed=59),dict(setup_completed=True)])
def test_invalid_identity_counters_rejected(change):
    p=policy();p.update(1,hb(1),None,1);h=hb(2,count=2);h.update(change)
    with pytest.raises(ValueError):p.update(2,h,None,2)

def test_read_cannot_regress_classifier_within_same_study():
    p=policy();p.update(1,hb(1,'classifier-chunk',windows=32),None,1)
    with pytest.raises(ValueError):p.update(2,hb(2,windows=32,count=2),None,2)

def test_setup_cannot_restart_after_inference():
    p=policy();p.update(1,hb(1,'render'),None,1)
    with pytest.raises(ValueError):p.update(2,hb(2,'model-loaded',count=2),None,2)

def test_no_invented_windows_on_study_transition():
    p=policy();p.update(1,hb(1,windows=0,count=1),None,1)
    p.update(2,hb(2,completed=2,windows=448,count=2),None,2)
    assert p.windows==448 and p.active==0

def test_classifier_time_pauses_during_render():
    p=policy();p.update(1,hb(1,'classifier'),None,1)
    p.update(4,hb(4,completed=1,windows=224,count=2),None,4)
    active=p.active
    p.update(10,hb(10,completed=1,windows=224,count=3),None,10)
    assert p.active==active==3

def test_actual_producer_multiple_studies(tmp_path):
    from tools.l1_rco_k2_worker_v103g import Watchdog,read
    w=Watchdog(tmp_path,tmp_path/'failure.json',False);p=policy()
    for t,index in enumerate(range(3),1):
        w.completed=index;w.progress('render');w.progress('renderer-source-read')
        h=read(tmp_path/'status.json');h.update(pid=123,time=t)
        assert p.update(t,h,None,t) is None
        w.progress('classifier-chunk',224)
    w.close()
    assert p.completed==2 and p.windows==448
