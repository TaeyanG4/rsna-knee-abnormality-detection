"""v1.5.0 process-local I/O retries and progress watchdog; no import-time work."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

VERSION = '1.5.0'
_replace, _rename, _unlink = os.replace, os.rename, os.unlink
_commits = {}


def retry(action, *args, attempts=40, pause=.05, **kwargs):
    for i in range(attempts):
        try:
            return action(*args, **kwargs)
        except OSError as exc:
            if not (isinstance(exc, PermissionError) or getattr(exc, 'winerror', None) in (5, 32)) or i+1 == attempts:
                raise
            time.sleep(min(pause*(i+1), 1.))


def atomic_write(path, data, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise ValueError('immutable artifact differs: '+str(path))
        return len(data)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        retry(_replace, temporary, path)
    finally:
        try: retry(_unlink, temporary)
        except FileNotFoundError: pass
        except OSError: pass  # orphan temp is retained; never masks journal commit error
    return len(data)


def json_write(path, value):
    return atomic_write(path, (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode())


def progress(phase, *, committed=False):
    path = os.environ.get('C85R7_V150_PROGRESS')
    if path:
        try:
            json_write(path, dict(phase=phase, monotonic=time.monotonic(), pid=os.getpid()))
            if committed:
                key=(path,os.getpid());_commits[key]=_commits.get(key,0)+1
                json_write(path+'.commits-'+str(os.getpid()),dict(count=_commits[key]))
        except OSError: pass


def install_io():
    """Also loaded by new-stage sitecustomize in native grandchildren."""
    if getattr(os, '_c85r7_v150_io', False): return
    os._c85r7_v150_io = True
    os.replace = lambda *a, **k: retry(_replace, *a, **k)
    os.rename = lambda *a, **k: retry(_rename, *a, **k)
    os.unlink = lambda *a, **k: retry(_unlink, *a, **k)


@contextmanager
def metadata_io(module):
    install_io()
    original = module.atomic_write
    def commit(path, data, **kwargs):
        try:
            value = atomic_write(path, data, **kwargs)
        except OSError as exc:
            # Frozen callers retain telemetry-vs-required-journal policy.
            try: json_write(Path(path).parent/'persistent-io-failure.json',
                            dict(status='HOLD', phase='commit', file=Path(path).name, error=type(exc).__name__))
            except OSError: pass
            raise
        if not any(s in Path(path).name.lower() for s in ('heartbeat','progress','telemetry')):
            progress('commit:'+Path(path).name, committed='journal-rows' in Path(path).parts or Path(path).name=='predictions.jsonl')
        return value
    module.atomic_write = commit
    try:
        progress('setup')
        yield
    finally:
        module.atomic_write = original


def stall_reason(age, samples=()):
    if age >= 900: return 'no_progress_15min'
    if age >= 600 and any(util >= 95 and power < 90 and used/total >= .90
                          for util, power, used, total in samples if total > 0):
        return 'wddm_stall_10min'
    return None


def gpu_samples():
    try:
        p = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,power.draw,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        return [tuple(float(x.strip()) for x in row.split(',')) for row in p.stdout.splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError): return []


class ThroughputGuard:
    """Thirty consecutive one-minute windows below 25% of measured journal rate."""
    def __init__(self, healthy_rate, now):
        if not math.isfinite(healthy_rate) or healthy_rate<=0:raise ValueError('healthy commit rate required')
        self.healthy_rate=healthy_rate;self.at=now;self.count=0;self.low_since=None
    def update(self, now, count):
        elapsed=now-self.at
        if elapsed<60:return None
        rate=max(0,count-self.count)/elapsed
        if rate<.25*self.healthy_rate:
            if self.low_since is None:self.low_since=self.at
        else:self.low_since=None
        self.at,self.count=now,count
        return 'throughput_below_25pct_30min' if self.low_since is not None and now-self.low_since>=1800 else None


def committed_records(marker):
    total=0
    for path in marker.parent.glob(marker.name+'.commits-*'):
        try:total+=max(0,int(json.loads(path.read_text())['count']))
        except (OSError,ValueError,KeyError):pass
    return total


@contextmanager
def watchdog(deadline_module, directory, bootstrap, healthy_rates=None):
    """Per-phase real commit progress; inherited deadline and owned-tree cleanup."""
    original_start = deadline_module.Supervisor.start
    original_monitor = deadline_module.Supervisor._monitor
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    def start(self, args, **kwargs):
        env = dict(os.environ, **kwargs.pop('env', {}))
        marker = root/('progress-'+str(os.getpid())+'-'+str(time.time_ns())+'.json')
        env['C85R7_V150_PROGRESS'] = str(marker)
        env['PYTHONPATH'] = str(bootstrap)+os.pathsep+env.get('PYTHONPATH','')
        env['KNEE_RELEASE_POOL_STALL_SECONDS_V1'] = '900'
        child = original_start(self, args, env=env, **kwargs)
        child.progress_marker, child.last_progress, child.progress_stamp = marker, time.monotonic(), None
        if healthy_rates is not None and child.label not in ('setup','codecs'):
            rate=healthy_rates.get(child.label)
            if rate is None:
                self._finish(child,'unmeasured_healthy_commit_rate')
                raise ValueError('healthy commit rate missing for '+child.label)
            child.throughput_guard=ThroughputGuard(rate,time.monotonic())
        return child
    def monitor(self):
        last_query, samples = 0., []
        while not self._closed.wait(self.poll):
            now = time.monotonic()
            if now-last_query >= 10:
                samples, last_query = gpu_samples(), now
            with self._lock:
                for child in self.children:
                    if child.returncode is not None: continue
                    if self.deadline.expired: self._finish(child, 'deadline'); continue
                    if child.process and child.process.poll() is not None:
                        self._finish(child, 'completed' if child.process.returncode == 0 else 'child_exit'); continue
                    marker = getattr(child, 'progress_marker', None)
                    if marker is None: continue
                    try:
                        stamp = marker.stat().st_mtime_ns
                        if stamp != child.progress_stamp:
                            child.last_progress, child.progress_stamp = now, stamp
                    except OSError: pass
                    reason = stall_reason(now-child.last_progress, samples)
                    guard=getattr(child,'throughput_guard',None)
                    if guard is not None:reason=reason or guard.update(now,committed_records(marker))
                    if reason: self._finish(child, 'watchdog:'+reason)
    deadline_module.Supervisor.start, deadline_module.Supervisor._monitor = start, monitor
    try: yield
    finally:
        deadline_module.Supervisor.start, deadline_module.Supervisor._monitor = original_start, original_monitor
