"""v1.0.1: isolate the inherited watchdog; kernel-owned descendant containment.

Windows-only candidate repair for LATOA-R1. The supervisor starts behind a pipe
barrier and enters a kill-on-close Job Object before it can spawn training.
No breakaway or inheritable job handle: even native guardian death closes the
last handle and kills its entire job. A provisional FAILED receipt precedes
launch; only a clean supervisor receipt AND verified empty job can earn PASS.
Historical hb155 watchdog policy/implementation is unchanged.
"""
import ctypes
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.hb150_io_v1 import atomic_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.1'


class WindowsJob:
    def __init__(self):
        if os.name != 'nt':
            raise OSError('local Windows containment required')
        k = self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, W.LPCWSTR]
        k.CreateJobObjectW.restype = W.HANDLE
        k.SetInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]
        k.SetInformationJobObject.restype = W.BOOL
        k.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
        k.AssignProcessToJobObject.restype = W.BOOL
        k.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
        k.TerminateJobObject.restype = W.BOOL
        k.QueryInformationJobObject.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p]
        k.QueryInformationJobObject.restype = W.BOOL
        k.CloseHandle.argtypes = [W.HANDLE]
        k.CloseHandle.restype = W.BOOL

        class Limits(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                ('flags', W.DWORD), ('min_working', ctypes.c_size_t), ('max_working', ctypes.c_size_t),
                ('active_limit', W.DWORD), ('affinity', ctypes.c_size_t),
                ('priority', W.DWORD), ('scheduling', W.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [('basic', Limits), ('io', ctypes.c_ulonglong * 6),
                ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]

        self.handle = k.CreateJobObjectW(None, None)  # non-inheritable, unnamed
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway
        try:
            self.check(k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, child):
        self.check(self.k.AssignProcessToJobObject(self.handle, W.HANDLE(int(child._handle))))

    def active(self):
        class Accounting(ctypes.Structure):
            _fields_ = [('user', ctypes.c_longlong), ('kernel', ctypes.c_longlong),
                ('period_user', ctypes.c_longlong), ('period_kernel', ctypes.c_longlong),
                ('faults', W.DWORD), ('total', W.DWORD), ('active', W.DWORD), ('terminated', W.DWORD)]
        value = Accounting()
        self.check(self.k.QueryInformationJobObject(self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None))
        return value.active

    def terminate(self):
        self.check(self.k.TerminateJobObject(self.handle, 125))
        deadline = time.monotonic() + 10
        while self.active():
            if time.monotonic() >= deadline:
                return False
            time.sleep(.02)
        return True

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None


def _worker_command(request):
    return [sys.executable, '-B', '-m', 'tools.latoa_expert_watchdog_v1', str(request)]


def supervise(command, attempt, *, healthy_rates=None, cpu=False, idle=900., stall=600., poll=5.):
    if not cpu and (idle != 900. or stall != 600. or poll != 5.):
        raise ValueError('production watchdog policy is fixed')
    attempt = Path(attempt).resolve()
    attempt.mkdir(parents=True, exist_ok=False)
    receipt = dict(version=VERSION, status='FAILED', reason='guardian_incomplete', exit_code=125,
        cleanup_confirmed=False, containment='Windows Job Object kill-on-close')
    atomic_json(attempt/'receipt.json', receipt)
    request = attempt/'request.json'
    atomic_json(request, dict(command=command, attempt=str(attempt/'watchdog'),
        healthy_rates=healthy_rates, cpu=cpu, idle=idle, stall=stall, poll=poll))
    job = child = None
    assigned = False
    try:
        job = WindowsJob()
        with (attempt/'supervisor.log').open('xb', buffering=0) as log:
            child = subprocess.Popen(_worker_command(request), cwd=ROOT, stdin=subprocess.PIPE,
                stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(child)
            assigned = True
            child.stdin.write(b'GO\n')
            child.stdin.close()
            code = child.wait()
        receipt['supervisor_exit_code'] = code
        inner = attempt/'watchdog/receipt.json'
        record = json.loads(inner.read_text()) if inner.exists() else {}
        good = (code == 0 and record.get('status') == 'PASS' and record.get('exit_code') == 0 and
                record.get('reason') == 'completed' and not record.get('survivors'))
        receipt.update(reason='completed' if good else 'watchdog_failed',
            watchdog_reason=record.get('reason', 'missing_receipt'), exit_code=0 if good else 125)
    except BaseException as exc:
        receipt.update(reason='guardian_error', error=type(exc).__name__, exit_code=125)
    finally:
        try:
            if job is not None:
                receipt['cleanup_confirmed'] = job.terminate()
            if child is not None:
                if not assigned and child.poll() is None:
                    child.kill()  # exact Popen handle; barrier has not released
                child.wait(timeout=10)
        except BaseException as exc:
            receipt.update(cleanup_confirmed=False, cleanup_error=type(exc).__name__)
        finally:
            if job is not None:
                job.close()  # final kernel kill, including on failed cleanup query
        if not receipt['cleanup_confirmed']:
            receipt.update(reason='termination_unconfirmed', exit_code=125)
        receipt['status'] = 'PASS' if receipt['exit_code'] == 0 else 'FAILED'
        atomic_json(attempt/'receipt.json', receipt)
    return receipt['exit_code']


def worker(request):
    # No training process is created until the guardian has assigned our job.
    if sys.stdin.buffer.readline() != b'GO\n':
        return 125
    from tools import hb155_watchdog_v1 as inherited
    value = json.loads(Path(request).read_text())
    path = Path(value['attempt'])/'receipt.json'
    try:
        return inherited.supervise(**value)
    finally:
        if path.exists():
            record = json.loads(path.read_text())
            record['status'] = ('PASS' if record.get('exit_code') == 0 and
                record.get('reason') == 'completed' and not record.get('survivors') else 'FAILED')
            atomic_json(path, record)


if __name__ == '__main__':
    raise SystemExit(worker(sys.argv[1]))
