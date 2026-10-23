"""v1.0.1: each real watchdog fault path runs in its own clean interpreter."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

ROOT=Path(__file__).resolve().parents[1]


def alive(identity):
    try:
        p=psutil.Process(identity['pid'])
        return p.is_running() and p.create_time()==identity['created']
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(os.name!='nt',reason='registered Windows host only')
@pytest.mark.parametrize('mode',[
    'normal','child_crash','supervisor_crash','guardian_crash','supervisor_exception',
    'idle_stall','wddm_stall','termination_unconfirmed'])
def test_production_containment_in_fresh_process(mode,tmp_path):
    # Unrelated sibling remains outside the kernel Job Object throughout.
    sibling=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(50)'])
    try:
        result=subprocess.run([sys.executable,'-B','-m','tools.latoa_expert_fault_probe_v1',
            mode,'guardian',str(tmp_path)],cwd=ROOT,timeout=35,capture_output=True,text=True)
        record=json.loads((tmp_path/'attempt/receipt.json').read_text())
        tree=json.loads((tmp_path/'tree.json').read_text())
        probe=json.loads((tmp_path/'probe.json').read_text())
        assert probe['pid']!=os.getpid() and probe['cpu_only']
        deadline=time.monotonic()+5
        while any(alive(p) for p in tree['processes']) and time.monotonic()<deadline:
            time.sleep(.02)
        assert not any(alive(p) for p in tree['processes']), (mode,record)
        assert sibling.poll() is None
        assert record['status']==('PASS' if mode=='normal' else 'FAILED'), record
        assert (result.returncode==0)==(mode=='normal'), (result.stderr,record)
        if mode=='guardian_crash':
            assert record['reason']=='guardian_incomplete'
            assert result.returncode & 0xffffffff == 0xC0000005
        else:
            assert record['cleanup_confirmed'] is True
        reasons={'idle_stall':'no_heartbeat_15m','wddm_stall':'wddm_no_heartbeat_10m',
            'termination_unconfirmed':'termination_unconfirmed','child_crash':'child_failed',
            'supervisor_exception':'supervisor_error'}
        if mode in reasons: assert record['watchdog_reason']==reasons[mode],record
        if mode=='supervisor_crash':
            assert record['watchdog_reason']=='missing_receipt'
            assert record['supervisor_exit_code'] & 0xffffffff == 0xC0000005
    finally:
        sibling.kill()
        sibling.wait(timeout=5)


@pytest.mark.parametrize('mode',['real_fit','real_skip'])
def test_real_crop_cpu_fit_and_skip_failure_under_production_watchdog(mode,tmp_path):
    result=subprocess.run([sys.executable,'-B','-m','tools.latoa_expert_fault_probe_v1',
        mode,'guardian',str(tmp_path)],cwd=ROOT,timeout=90,capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    record=json.loads((tmp_path/'attempt/receipt.json').read_text())
    assert record['cleanup_confirmed'] is True
    if mode=='real_fit':
        assert result.returncode==0 and record['status']=='PASS',record
        summary=json.loads((tmp_path/'fit/summary.json').read_text())
        assert summary['optimizer_steps']==summary['attempted_steps']==1
        assert summary['counters']=={'updates':1,'skips':0}
        assert summary['passed'] and summary['cuda_initialized'] is False
    else:
        assert result.returncode!=0 and record['status']=='FAILED'
        failure=json.loads((tmp_path/'fit/failure.json').read_text())
        assert failure['status']=='FAILED' and failure['optimizer_steps']==0
        assert failure['counters']=={'updates':1,'skips':1}
        assert not (tmp_path/'fit/final.pt').exists()
        assert not (tmp_path/'fit/summary.json').exists()
