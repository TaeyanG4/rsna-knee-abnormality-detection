"""Windows private-commit/host-RAM supervisor and atomic recovery snapshots.

Summed RSS is telemetry only: read-only corpus mmap pages can appear in several
working sets. Private commit is not unique resident RAM; host available RAM is
a separate pressure gate. Both gates are sampled, not OS-enforced limits.
Only owned PID+creation-time identities are cleaned; no retries or global kill.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from tools.process_resource_receipt import OwnedTree


def atomic_write(path, value, writer):
    """Keep the previous complete snapshot if serialization fails."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open('xb') as stream:
            writer(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path, value):
    atomic_write(path, value, lambda v, f: f.write(
        (json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')))


def memory_sample(processes, psutil):
    rss = private = 0
    rows = []
    for identity, process in processes:
        try:
            if process.create_time() != identity[1]:
                raise RuntimeError('PID identity changed during memory sample')
            info = process.memory_info()
            # Windows PROCESS_MEMORY_COUNTERS_EX.PrivateUsage, not RSS/PSS/USS.
            value = getattr(info, 'private', None)
            if value is None or value < 0:
                raise RuntimeError('Windows private commit counter unavailable')
            if process.create_time() != identity[1]:
                raise RuntimeError('PID identity changed during memory sample')
            rss += info.rss
            private += value
            rows.append(dict(pid=identity[0], create_time=identity[1],
                             rss_bytes=info.rss, private_commit_bytes=value))
        except psutil.NoSuchProcess:
            continue
    vm = psutil.virtual_memory()
    return dict(tree_rss_bytes=rss, tree_private_commit_bytes=private,
                host_available_bytes=vm.available, host_total_bytes=vm.total,
                processes=rows)


def limit_reason(sample, elapsed, seconds, private_cap, available_floor):
    if elapsed >= seconds:
        return 'time_limit'
    if sample['tree_private_commit_bytes'] > private_cap:
        return 'private_commit_limit'
    if sample['host_available_bytes'] < available_floor:
        return 'host_available_floor'
    return None


def supervise(command, output, seconds, private_cap, available_floor, env, protocol_sha):
    import psutil
    if os.name != 'nt' or min(seconds, private_cap, available_floor) <= 0:
        raise ValueError('Windows with positive explicit resource limits required')
    output = Path(output)
    started = time.monotonic()
    proc = None
    tree = OwnedTree(psutil)
    owner = psutil.Process()
    owner_identity = (owner.pid, owner.create_time())
    peak_rss = peak_private = 0
    minimum_available = None
    samples = 0
    failure = killed = last = None
    cleanup = dict(actions=[], errors=[], survivors=[])
    try:
        with (output / 'supervisor.stdout.log').open('x', encoding='utf-8') as stdout, \
             (output / 'supervisor.stderr.log').open('x', encoding='utf-8') as stderr:
            proc = subprocess.Popen(command, env=env, cwd=Path(__file__).resolve().parents[1],
                                    stdout=stdout, stderr=stderr,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            child = psutil.Process(proc.pid)
            tree.record(child)
            write_json(output / 'supervisor-job.json', dict(pid=proc.pid, argv=command,
                protocol_sha256=protocol_sha, phase=env['EXP087_PHASE'],
                phase_authorization_sha256=env['EXP087_PHASE_AUTHORIZATION_SHA256'],
                started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                seconds_cap=seconds, tree_private_commit_cap=private_cap,
                host_available_floor=available_floor, interval_seconds=.1, retry=0))
            next_record = 0.0
            while proc.poll() is None:
                # Discover from the root once per sample. Retain observed
                # descendants for cleanup even after they exit/reparent.
                try:
                    for q in child.children(recursive=True):
                        tree.record(q)
                except psutil.NoSuchProcess:
                    pass
                last = memory_sample([(owner_identity, owner), *tree.live()], psutil)
                samples += 1
                peak_rss = max(peak_rss, last['tree_rss_bytes'])
                peak_private = max(peak_private, last['tree_private_commit_bytes'])
                minimum_available = last['host_available_bytes'] if minimum_available is None else min(
                    minimum_available, last['host_available_bytes'])
                elapsed = time.monotonic() - started
                killed = limit_reason(last, elapsed, seconds, private_cap, available_floor)
                if elapsed >= next_record or killed:
                    write_json(output / 'resource-latest.json', dict(elapsed_seconds=elapsed,
                        sampled_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **last))
                    next_record = elapsed + 30
                if killed:
                    break
                time.sleep(.1)
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        if proc is not None:
            cleanup = tree.cleanup()
            try:
                proc.wait(timeout=10)
            except BaseException as exc:
                cleanup['errors'].append(str(exc))
    elapsed = time.monotonic() - started
    status = 'PASS' if (proc is not None and proc.returncode == 0 and not failure and not killed
                       and not any(cleanup.values()) and elapsed < seconds and samples > 0) else 'FAIL'
    receipt = dict(status=status, returncode=None if proc is None else proc.returncode,
        error=failure, killed=killed, cleanup_errors=cleanup['errors'],
        cleanup_killed_pids=[x['pid'] for x in cleanup['actions']],
        cleanup_survivors=cleanup['survivors'], wall_seconds=elapsed,
        peak_tree_RSS=peak_rss, peak_tree_private_commit_bytes=peak_private,
        min_host_available_bytes=minimum_available, sample_count=samples,
        last_sample=last, protocol_sha256=protocol_sha, phase=env['EXP087_PHASE'],
        phase_authorization_sha256=env['EXP087_PHASE_AUTHORIZATION_SHA256'],
        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        limits=dict(seconds=seconds, tree_private_commit_bytes=private_cap,
                    host_available_bytes_floor=available_floor, interval_seconds=.1),
        limitations=['RSS can double-count mmap; diagnostic only',
            'Private commit is not unique resident RAM; host available covers global pressure',
            'Sampling can miss transient peaks and unobserved descendants; no OS hard cap',
            'Includes helper and observed trainer tree, excludes small gpu_job wrapper'])
    write_json(output / 'resource-supervisor.json', receipt)
    if status != 'PASS':
        raise RuntimeError('supervised run failed; no retry')
    return receipt


def save_recovery(out, torch, payload):
    """One epoch state for future reviewed recovery; never select or auto-resume."""
    atomic_write(Path(out) / 'recovery_latest.pt', payload, torch.save)

