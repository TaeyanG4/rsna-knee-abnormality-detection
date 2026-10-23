"""Target-only Linux/NVIDIA cleanup evidence. Never invoked by CPU qualification.

Observe recursive descendants, session/group members and newly appearing same-user
processes (isolated diagnostic kernel required), retaining PID/create-time pairs.
GPU totals and compute PID allocations must return to their pre-case baselines.
"""
import os
import subprocess
import threading
import time


def processes():
    import psutil
    result = {}
    for process in psutil.process_iter(['pid', 'ppid', 'create_time', 'username', 'status']):
        try:
            info = process.info
            info['pgid'], info['sid'] = os.getpgid(info['pid']), os.getsid(info['pid'])
            result[(info['pid'], info['create_time'])] = info
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    return result


def gpu():
    def query(flag):
        return subprocess.check_output(['nvidia-smi', flag, '--format=csv,noheader,nounits'],
                                       text=True, timeout=10).strip().splitlines()
    memory = {parts[0].strip(): int(parts[1]) for line in query('--query-gpu=uuid,memory.used')
              if line.strip() for parts in [line.split(',')]}
    allocations = {int(parts[0]): (parts[1].strip(), int(parts[2]))
                   for line in query('--query-compute-apps=pid,gpu_uuid,used_gpu_memory')
                   if line.strip() for parts in [line.split(',')]}
    assert memory, 'GPU inventory unavailable; cannot certify cleanup'
    return memory, allocations


class CleanupAudit:
    def __init__(self, *, process_snapshot=processes, gpu_snapshot=gpu, poll=.05):
        self.process_snapshot, self.gpu_snapshot = process_snapshot, gpu_snapshot
        self.poll = poll
        self.before = process_snapshot()
        self.gpu_before = gpu_snapshot()
        self.pid = os.getpid()
        self.owner = next((p['username'] for p in self.before.values() if p['pid']==self.pid), None)
        assert self.owner is not None, 'root process not observed'
        self.seen = {}
        self.error = None
        self.stop = threading.Event()

    def sample(self):
        current = self.process_snapshot()
        # New same-user processes conservatively include briefly detached children.
        self.seen.update({k:v for k,v in current.items() if k not in self.before and
                          v['username']==self.owner and v['pid']!=self.pid})
        return current

    def __enter__(self):
        def watch():
            while not self.stop.wait(self.poll):
                try: self.sample()
                except Exception as exc:
                    self.error = repr(exc)
                    return
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def verify(self, anchor, *, reserve_seconds=1200, gpu_tolerance_mib=0):
        # Verification itself consumes the publication reserve. No success based
        # solely on killpg return or waiting for the direct child.
        deadline = anchor + reserve_seconds
        while True:
            current = self.sample()
            remaining = [k for k in self.seen if k in current]
            memory, allocations = self.gpu_snapshot()
            baseline_mem, baseline_alloc = self.gpu_before
            gpu_ok = (set(memory)==set(baseline_mem) and all(
                memory[k] <= baseline_mem[k] + gpu_tolerance_mib for k in memory)
                and all(pid in baseline_alloc and value==baseline_alloc[pid]
                        for pid,value in allocations.items()))
            elapsed = time.monotonic()-anchor
            assert self.error is None, 'descendant observer failed: '+str(self.error)
            if not remaining and gpu_ok:
                assert elapsed <= reserve_seconds, 'cleanup/publication exceeded reserve'
                return dict(descendants_observed=len(self.seen), descendants_remaining=0,
                            gpu_memory_released=True, gpu_tolerance_mib=gpu_tolerance_mib,
                            cleanup_and_publication_seconds=elapsed, reserve_seconds=reserve_seconds)
            assert time.monotonic()<deadline, 'descendants/GPU allocations remain after reserve'
            time.sleep(min(.1, max(0.,deadline-time.monotonic())))

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)


def publication_anchor(clock, marker=None):
    now = time.monotonic()
    cutoff = clock.started + clock.seconds
    if marker is not None and marker.exists():
        triggered = now - max(0., time.time()-marker.stat().st_mtime)
        return min(cutoff, triggered)
    return min(now, cutoff)
