"""INF3 diagnostic observer v1.0.1. Explicit runtime use only; no import actions.

GPU occupancy is sampled telemetry, never allocator peak or a hidden upper bound.
Inference code/process setup is not patched. Watchdog owns only observed descendants.
"""
from __future__ import annotations
import csv
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time

VERSION = '1.0.1'
TERM_GRACE_SECONDS = 10.
KILL_GRACE_SECONDS = 10.
CLEANUP_POLL_SECONDS = .1


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + '\n').encode()
    path = Path(path)
    for attempt in range(40):
        try:
            with path.open('xb') as stream: stream.write(data)
            return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(min(.05 * (attempt + 1), 1))


def seal_cohort(root, ids):
    root = Path(root).resolve()
    if len(ids) != 32 or len(set(ids)) != 32: raise ValueError('exact32 unique IDs required')
    names = ('test.csv', 'test_series.csv', 'sample_submission.csv', 'selection.json')
    record = dict(schema='h197-n32-cohort/v1', studies=32, ids=ids, root=str(root),
                  sha256={n: digest(root / n) for n in names})
    dump(root / 'h197-cohort.json', record)
    return record


def process_table():
    if platform.system() != 'Linux': raise ValueError('diagnostic observer requires actual Linux /proc')
    records = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal(): continue
        try:
            stat = (p / 'stat').read_text(); tail = stat[stat.rindex(')') + 2:].split()
            records[int(p.name)] = dict(pid=int(p.name), ppid=int(tail[1]), start_ticks=int(tail[19]),
                state=tail[0], cpu_ticks=int(tail[11]) + int(tail[12]),
                command=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip())
        except (OSError, ValueError, IndexError): continue  # a process may exit during observation
    return records


def descendants(table, owner):
    found = set(); parents = {owner}
    while True:
        extra = {pid for pid, r in table.items() if r['ppid'] in parents} - found
        if not extra: return found
        found.update(extra); parents = extra


def gpu_sample():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,power.draw',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 6: raise ValueError('GPU telemetry columns')
        uuid, name, total, used, util, watts = [v.strip() for v in row]
        rows.append(dict(uuid=uuid, name=name, total_mib=float(total), used_mib=float(used),
                         utilization_percent=float(util), power_watts=float(watts)))
    if not rows: raise ValueError('no actual GPU telemetry')
    return rows


def progress_bytes(root):
    total = 0
    # Never walk the symlinked DICOM tree, datasets or installed packages.
    for p in (Path(root) / 'c85r7-run').rglob('*'):
        if p.name in ('predictions.jsonl', 'predictions.csv', 'receipt.json', 'q-timings.json') or p.name.endswith('-exit.json'):
            try: total += p.stat().st_size
            except OSError: pass
    for name in ('raptor224_raw.csv', 'raptor224_sub.csv'):
        p = Path(root) / name
        if p.exists(): total += p.stat().st_size
    return total


class Observer:
    def __init__(self, root, started):
        self.root = Path(root); self.started = started; self.owner = os.getpid()
        self.stop = threading.Event(); self.thread = None; self.seen = {}; self.samples = []
        self.errors = []; self.abort = None; self.last_progress = started; self.last_size = 0
        self.history = []; self.healthy = 0.; self.low_since = None; self.stall_since = None
        self.phase_signature = None
        self.abort_started = None
        self.owner_start = None; self.baseline = set(); self.observer_started = False
        self.cleanup_signals = []

    def start(self):
        table = process_table(); self.owner_start = table[self.owner]['start_ticks']
        self.baseline = {(p, table[p]['start_ticks']) for p in descendants(table, self.owner)}
        self.thread = threading.Thread(target=self._loop, name='h197-evidence-observer', daemon=True)
        self.thread.start()
        self.observer_started = True

    @contextmanager
    def lifecycle(self):
        """Cover activation, bootstrap and inference; preserve the primary failure."""
        failure = None
        try:
            self.start()
            yield self
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                self.finish(failure)
            except BaseException as cleanup_error:
                if failure is None: raise
                failure.add_note('INF3 cleanup failed: ' + type(cleanup_error).__name__)
                try:
                    print('H197_CLEANUP_FAILURE: ' + type(cleanup_error).__name__, file=sys.stderr, flush=True)
                except BaseException:
                    pass  # reporting must never replace the original exception

    def _discover_owned(self, table):
        """Track only current descendants of verified owner/previously owned roots."""
        now = time.monotonic() - self.started
        owner = table.get(self.owner)
        roots = {self.owner} if owner and owner['start_ticks'] == self.owner_start else set()
        for key, old in list(self.seen.items()):
            current = table.get(old['pid'])
            if current and current['start_ticks'] == old['start_ticks']:
                roots.add(old['pid'])  # includes a verified child reparented after failure
        visited = set(roots)
        while roots:
            added = set()
            for pid, rec in table.items():
                key = (pid, rec['start_ticks'])
                if rec['ppid'] in roots and pid not in visited and pid != self.owner:
                    visited.add(pid)
                    if key in self.baseline:
                        continue  # pre-existing processes and their subtrees are not ours
                    self.seen[key] = dict(rec, first_seen=self.seen.get(key, {}).get('first_seen', now), last_seen=now)
                    added.add(pid)
            roots = added
        return [dict(table[old['pid']]) for _, old in list(self.seen.items())
                if old['pid'] in table and table[old['pid']]['start_ticks'] == old['start_ticks']
                and table[old['pid']]['state'] not in ('Z', 'X')]

    def _signal_owned(self, key, force=False):
        # Fresh process identity immediately before EACH signal, including escalation.
        old = self.seen.get(key)
        if old is None: return False
        current = process_table().get(old['pid'])
        if (current is None or current['start_ticks'] != old['start_ticks'] or
                current['state'] in ('Z', 'X') or old['pid'] == self.owner):
            return False
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.kill(old['pid'], sig)
            self.cleanup_signals.append(dict(pid=old['pid'], start_ticks=old['start_ticks'],
                                             signal=int(sig), elapsed_seconds=time.monotonic() - self.started))
            return True
        except ProcessLookupError:
            return False

    def _kill_owned(self, force=False):
        # Recheck PID+start identity immediately before each signal; never kill by name.
        self._discover_owned(process_table())
        for key in list(self.seen): self._signal_owned(key, force)

    def _cleanup_owned(self):
        """Bound cleanup only, never inference time. Rediscover late descendants."""
        began = time.monotonic(); term_at = {}; killed = set(); errors = []; quiet = 0
        alive = []; verified = False
        while True:
            now = time.monotonic()
            try:
                alive = self._discover_owned(process_table()); verified = True
                quiet = quiet + 1 if not alive else 0
                if quiet >= 2: break
                for rec in alive:
                    key = (rec['pid'], rec['start_ticks'])
                    if key not in term_at:
                        term_at[key] = now
                        self._signal_owned(key)
                    if key not in killed and (now - term_at[key] >= TERM_GRACE_SECONDS or
                                              now - began >= TERM_GRACE_SECONDS):
                        self._signal_owned(key, force=True); killed.add(key)
            except Exception as exc:
                verified = False; quiet = 0
                errors.append(dict(kind='cleanup:' + type(exc).__name__))
            if now - began >= TERM_GRACE_SECONDS + KILL_GRACE_SECONDS: break
            time.sleep(CLEANUP_POLL_SECONDS)
        # A signal attempt is not death. Re-enumerate and verify every remembered identity.
        try:
            alive = self._discover_owned(process_table()); verified = True
        except Exception as exc:
            verified = False; errors.append(dict(kind='cleanup-final:' + type(exc).__name__))
        return dict(complete=verified and not alive and not errors, alive=alive, errors=errors)

    def _loop(self):
        while not self.stop.is_set():
            now = time.monotonic(); row = dict(elapsed_seconds=now - self.started)
            try:
                self._discover_owned(process_table())
                row['gpu'] = gpu_sample()
            except Exception as exc:
                self.errors.append(dict(elapsed_seconds=now - self.started, kind=type(exc).__name__))
            try: size = progress_bytes(self.root)
            except Exception as exc:
                self.errors.append(dict(kind='progress:' + type(exc).__name__)); size = self.last_size
            # Throughput units differ between sequential branches. Compare only
            # within a phase; a completed phase resets the rolling reference.
            signature = tuple(sorted(str(p) for p in (self.root / 'c85r7-run').rglob('*-exit.json')))
            if signature != self.phase_signature:
                self.phase_signature = signature; self.history = []; self.healthy = 0.; self.low_since = None
            if size > self.last_size: self.last_size = size; self.last_progress = now
            row['progress_bytes'] = size
            self.history.append((now, size)); self.history = [(t, n) for t, n in self.history if now - t <= 300]
            if len(self.history) > 1 and now - self.history[0][0] >= 240:
                rate = (size - self.history[0][1]) / max(1, now - self.history[0][0])
                self.healthy = max(self.healthy, rate)
                if self.healthy > 0 and rate < .25 * self.healthy:
                    self.low_since = self.low_since or now
                else: self.low_since = None
            low_power_stall = any(g['utilization_percent'] >= 99 and g['power_watts'] < 90 and
                                 g['used_mib'] >= .95 * g['total_mib'] for g in row.get('gpu', []))
            self.stall_since = (self.stall_since or now) if low_power_stall else None
            if now - self.last_progress >= 900: self.abort = '15min without output progress'
            if self.stall_since and now - self.stall_since >= 600 and now - self.last_progress >= 600:
                self.abort = '10min high-util/low-power/memory stall'
            if self.low_since and now - self.low_since >= 1800: self.abort = '30min throughput below25pct observed healthy rate'
            self.samples.append(row)
            try:
                with (self.root / 'h197-samples.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
            except OSError as exc:
                self.errors.append(dict(elapsed_seconds=now - self.started, kind='heartbeat:' + type(exc).__name__))
            if self.abort:
                self.abort_started = self.abort_started or now
                # Continue observing until finally: original survivor control may
                # start a later child after a failed earlier phase. Kill only
                # verified descendants, escalating after10s, never by process name.
                self._kill_owned(force=now - self.abort_started >= 10)
            self.stop.wait(5)

    def finish(self, failure=None):
        initial_errors = []
        try:
            early_stop = self.stop.is_set() or (self.observer_started and self.thread and not self.thread.is_alive())
        except Exception as exc:
            early_stop = True; initial_errors.append(dict(kind='observer-state:' + type(exc).__name__))
        self.stop.set()
        # Cleanup must run before any join/telemetry/write failure can escape.
        first = self._cleanup_owned()
        join_errors = []
        try:
            if self.thread: self.thread.join(timeout=20)
        except Exception as exc:
            join_errors.append(dict(kind='observer-join:' + type(exc).__name__))
        finally:
            last = self._cleanup_owned()  # catch descendants born while the observer stopped
        cleanup_errors = initial_errors + first['errors'] + join_errors + last['errors']
        try: stopped = self.thread is not None and not self.thread.is_alive()
        except Exception as exc:
            stopped = False; cleanup_errors.append(dict(kind='observer-final-state:' + type(exc).__name__))
        if early_stop: cleanup_errors.append(dict(kind='observer-stopped-before-finally'))
        complete = last['complete'] and not cleanup_errors and stopped and self.observer_started
        try: self.samples.append(dict(elapsed_seconds=time.monotonic() - self.started, gpu=gpu_sample(), progress_bytes=progress_bytes(self.root)))
        except Exception as exc: self.errors.append(dict(kind='final_sample:' + type(exc).__name__))
        record = dict(schema='h197-observer/v1.0.1', status='STOPPED' if complete else 'CLEANUP_FAILED', owner_pid=self.owner, owner_start_ticks=self.owner_start,
            version=VERSION, platform=platform.platform(), source_sha256=digest(__file__),
            monotonic_wall_seconds=time.monotonic() - self.started, samples=self.samples,
            process_inventory=list(self.seen.values()), owned_children_alive=last['alive'], watchdog_abort=self.abort,
            cleanup_complete=complete, cleanup_errors=cleanup_errors, cleanup_signals=self.cleanup_signals,
            observer_started=self.observer_started, observer_thread_stopped=stopped,
            original_failure=None if failure is None else type(failure).__name__,
            telemetry_errors=self.errors, measurement='sampled device occupancy, NOT allocator reserved peak',
            qualified_for_submission=False)
        dump(self.root / 'h197-observer.json', record)
        if not complete or self.abort: raise RuntimeError('diagnostic child cleanup/watchdog failed')
        return record


def benchmark_functions(functions, source_sha256, hidden=1300, repeats=3):
    import numpy as np
    if type(hidden) is not int or hidden < 32 or type(repeats) is not int or repeats < 3:
        raise ValueError('full hidden cohort and at least3 measured repeats')
    rng = np.random.default_rng(197)
    leaves = {f: rng.uniform(.01, .99, (10 if f == 'l2' else 5, hidden, 12)).astype(np.float32)
              for f in ('l2', 'dino', 'convnext')}
    branches = [rng.uniform(.01, .99, (hidden, 12)).astype(np.float64) for _ in range(5)]
    ids = [f'SYNTHETIC-COST-{i:06d}' for i in range(hidden)]
    f = functions
    def operation():
        q0, families = f['q0'](leaves)
        q3 = f['q3'](leaves, np)
        ranked = [f['rank'](x) for x in branches]
        result = f['blend'](ids, *branches, q3)
        reference = f['reference'](*branches, q3)
        failures = np.zeros(hidden, dtype=bool)
        pbranch = f['p_branch'](*branches[:3], failures, failures, failures)
        outputs = [q0, q3, *ranked, result, reference, pbranch]
        for a in outputs:
            if np.asarray(a).shape != (hidden, 12) or not np.isfinite(a).all(): raise ValueError('benchmark shape/nonfinite')
            f['csv'](ids, a)
        stream = io.BytesIO(); np.savez(stream, ids=np.asarray(ids), **leaves)
        return sum(np.asarray(x).nbytes for x in outputs)
    operation()  # explicit warmup, never counted as inference
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter(); output_bytes = operation(); seconds.append(time.perf_counter() - start)
    if not all(math.isfinite(x) and x > 0 for x in seconds): raise ValueError('benchmark timer')
    return dict(schema='h197-global-cost/v1', status='COMPLETE_SYNTHETIC_COST_ONLY', hidden_studies=hidden,
        source_sha256=source_sha256, measured_seconds=seconds, seconds=max(seconds), repeats=repeats, warmup=1,
        seed=197, leaf_shapes={k: list(v.shape) for k, v in leaves.items()}, leaf_dtype='float32', pool_dtype='float64',
        operations=['actual Q0 aggregate', 'actual INF3 Q3 aggregate', 'five branch ranks', 'actual final blend',
                    'actual healthy reference', 'actual P branch', 'ten actual CSV serializations', 'full leaf NPZ serialization'],
        output_bytes=output_bytes, platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
        model_forward=False, real_predictions=False, qualified_for_submission=False)


def benchmark(release, runtime, output, hidden=1300):
    runtime, output = Path(runtime), Path(output)
    functions = dict(q0=release.c85.a172.aggregate, q3=release.c85.inf3_aggregate, rank=release.c85.rank,
        blend=release.r224.blend, reference=release.r224.reference, p_branch=release.r224.p_branch, csv=release.p.csv_bytes)
    sources = {p.relative_to(runtime).as_posix(): digest(p) for p in runtime.rglob('*') if p.is_file() and p.suffix in ('.py', '.txt', '.json')}
    record = benchmark_functions(functions, sources, hidden)
    record['observer_source_sha256'] = digest(__file__)
    dump(output / 'h197-global-cost.json', record)
    return record
