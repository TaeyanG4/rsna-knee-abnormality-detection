"""Meaningful failure-directed tests for H209 setup progress consumer."""
import pytest
from tools.hb209_local_progress_v1 import Progress, parent
def hb(count,t,**extra):
    return dict(pid=123,time=t,completed=0,windows=0,phase='codec-header',setup_completed=count,**extra)
def test_original_rejects_repaired_worker_setup():
    with pytest.raises(ValueError):parent.Progress(0,dict(pid=123,create_time=0),58).update(10,hb(1,10),None,10)
def test_real_setup_advances_beyond_old_timeout():
    p=Progress(0,dict(pid=123,create_time=0),58)
    for t in range(100,2001,100):assert p.update(t,hb(t//100,t),None,t) is None
    assert p.setup_count==20 and p.events==20
def test_repeated_timer_without_work_does_not_keep_alive():
    p=Progress(0,dict(pid=123,create_time=0),58);p.update(1,hb(1,1),None,1)
    assert p.update(902,hb(1,902),None,902) is not None
def test_regressing_count_or_wrong_pid_rejected():
    p=Progress(0,dict(pid=123,create_time=0),58);p.update(1,hb(2,1),None,1)
    with pytest.raises(ValueError):p.update(2,hb(1,2),None,2)
    value=hb(3,3);value['pid']=999
    with pytest.raises(ValueError):p.update(3,value,None,3)
def test_inference_transition_and_regression():
    p=Progress(0,dict(pid=123,create_time=0),58);p.update(1,hb(1,1),None,1)
    v=dict(pid=123,time=2,completed=0,windows=0,phase='render')
    assert p.update(2,v,None,2) is None
    v.update(time=3,phase='classifier');assert p.update(3,v,None,3) is None
    v.update(time=4,phase='classifier-chunk',windows=32);assert p.update(4,v,None,4) is None
    with pytest.raises(ValueError):p.update(5,hb(2,5),None,5)
def test_actual_producer_publishes_completed_counter(tmp_path):
    from tools import l1_rco_k2_worker_v103g as w
    wd=w.Watchdog(tmp_path,tmp_path/'fail.json',False)
    wd.progress('codec-header');d=w.read(tmp_path/'status.json')
    assert d['setup_completed']==1 and d['windows']==0
    wd.progress('classifier-chunk',32);d=w.read(tmp_path/'status.json')
    assert d['setup_completed']==1 and d['windows']==32
    wd.close()

