"""v1.0.1 CPU-only fresh-process fault injection for the candidate watchdog.

Changelog: exercise the production guardian/worker/inherited supervision path.
Never imported by production. No CUDA or GPU telemetry; faults are test-only.
"""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tools.hb150_io_v1 import atomic_json


def native_crash():
    ctypes.windll.kernel32.SetErrorMode(0x8003)
    # Uncatchable native process death with STATUS_ACCESS_VIOLATION. This
    # reproduces bypass of Python finally, not the unknown original DLL fault.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.TerminateProcess.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    kernel.TerminateProcess(kernel.GetCurrentProcess(),0xC0000005)
    raise RuntimeError('native termination unexpectedly returned')


def await_tree(path):
    deadline=time.monotonic()+15
    while not path.exists():
        if time.monotonic()>deadline: raise TimeoutError('CPU descendant marker')
        time.sleep(.02)


def cpu_fit(directory, skip):
    """Real crop IO/forward/backward; optional CPU emulation of one skipped step.

    AMP itself is not tested on CPU. The scaler deliberately does not call
    optimizer.step; the inherited CPU loop's counter is annotated accordingly.
    """
    from tools import latoa_expert_fit_v1 as fit
    from tools import latoa_expert_v1 as build
    if skip:
        original_stack=fit.h.optimizer_stack
        original_train=fit.h.train_epoch_accum
        def stack(*args,**kwargs):
            opt,sched,scaler=original_stack(*args,**kwargs)
            scaler.step=lambda opt:None
            return opt,sched,scaler
        def train(*args,**kwargs):
            record=original_train(*args,**kwargs)
            counters=args[-1]
            counters['skips']+=record['batches']
            return record
        fit.h.optimizer_stack=stack
        fit.h.train_epoch_accum=train
    result=fit.fit(build.config_path(46),directory/'fit',device='cpu',smoke=True)
    return 0 if result['passed'] else 1


def main():
    mode, action, directory=sys.argv[1:4]
    directory=Path(directory)
    marker=directory/'tree.json'
    if action=='cpu-fit':
        return cpu_fit(directory,mode=='real_skip')
    if action=='leaf':
        time.sleep(50)
        return 0
    if action=='child':
        import psutil
        leaf=subprocess.Popen([sys.executable,'-B','-m',__spec__.name,mode,'leaf',str(directory)])
        atomic_json(marker,{'processes':[{'pid':p.pid,'created':p.create_time()}
            for p in (psutil.Process(),psutil.Process(leaf.pid))]})
        if mode=='normal': return 0
        if mode=='child_crash': native_crash()
        time.sleep(50)
        return 0
    from tools import latoa_expert_watchdog_v1 as guardian
    if action=='worker':
        from tools import hb155_watchdog_v1 as inherited
        request=Path(sys.argv[4])
        if mode in ('supervisor_crash','supervisor_exception'):
            original=inherited.Policy.check
            def check(self,*args):
                if marker.exists():
                    if mode=='supervisor_crash': native_crash()
                    raise RuntimeError('forced supervisor exception')
                return original(self,*args)
            inherited.Policy.check=check
        if mode=='termination_unconfirmed':
            def refuse(root,owned):
                return [{'pid':root.pid,'create_time':root.create_time()}]
            inherited.terminate_owned=refuse
        if mode=='wddm_stall':
            value=json.loads(request.read_text()); value.update(cpu=False,idle=20,stall=1)
            atomic_json(request,value)
            inherited.telemetry=lambda:dict(util=100,power=40,used_mib=16000,total_mib=16384)
        return guardian.worker(request)
    if action!='guardian': raise ValueError(action)
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'probe.json',{'pid':os.getpid(),'mode':mode,'cpu_only':True})
    guardian._worker_command=lambda request:[sys.executable,'-B','-m',__spec__.name,
        mode,'worker',str(directory),str(request)]
    if mode in ('real_fit','real_skip'):
        # No worker fault injection: use the exact production worker command.
        guardian._worker_command=lambda request:[sys.executable,'-B','-m',
            'tools.latoa_expert_watchdog_v1',str(request)]
        return guardian.supervise([sys.executable,'-B','-m',__spec__.name,mode,'cpu-fit',str(directory)],
            directory/'attempt',cpu=True,poll=.1)
    if mode=='guardian_crash':
        def crash_after_spawn():
            await_tree(marker)
            native_crash()
        threading.Thread(target=crash_after_spawn,daemon=True).start()
    return guardian.supervise([sys.executable,'-B','-m',__spec__.name,mode,'child',str(directory)],
        directory/'attempt',cpu=True,idle=2,stall=1,poll=.03)


if __name__=='__main__':
    raise SystemExit(main())
