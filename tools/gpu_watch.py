"""gpu_watch v1.0.1 — wake slot B when the local GPU needs attention (B's prevention for the 2026-10-08 05:16-10:38 idle GPU).

Run in the background from the Claude Code session (its exit re-invokes the session). Exits with a one-line reason when:
  - GPU utilisation stays below --idle-util for --idle-minutes consecutive minutes (default 10 % for 10 min),
  - a new FAILED marker appears under artifacts/logs/ after start,
  - a queue log under artifacts/logs/queue/ gains a "plan finished" line after start,
  - any --watch-file appears (e.g. a sub-agent's output file),
  - --max-minutes elapse (periodic check-in; default 60).
Usage: python tools/gpu_watch.py [--idle-util 10] [--idle-minutes 10] [--max-minutes 60] [--watch-file PATH ...]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'artifacts' / 'logs'


def gpu_util():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
        return max(int(x) for x in out.split())
    except Exception:
        return -1


def failed_markers():
    return {p for p in LOGS.glob('*/FAILED')}


def finished_queues():
    res = set()
    for p in (LOGS / 'queue').glob('*.log'):
        try:
            n = p.read_text(encoding='utf-8', errors='replace').count('plan finished')
        except OSError:
            continue
        if n:  # v1.0.1: a new queue log with no 'plan finished' line is not a finished queue
            res.add((p.name, n))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idle-util', type=int, default=10)
    ap.add_argument('--idle-minutes', type=float, default=10)
    ap.add_argument('--max-minutes', type=float, default=60)
    ap.add_argument('--watch-file', action='append', default=[])
    ap.add_argument('--poll', type=float, default=30)
    a = ap.parse_args()
    t0 = time.time()
    fail0, fin0 = failed_markers(), finished_queues()
    idle_since = None
    print(f'gpu_watch v1.0.1 start {time.strftime("%H:%M:%S")}', flush=True)
    while True:
        now = time.time()
        u = gpu_util()
        if 0 <= u < a.idle_util:
            idle_since = idle_since or now
            if now - idle_since >= a.idle_minutes * 60:
                print(f'WAKE gpu idle: util {u}% for {a.idle_minutes:g} min', flush=True); return 0
        else:
            idle_since = None
        new_fail = failed_markers() - fail0
        if new_fail:
            print('WAKE job failed: ' + ', '.join(sorted(p.parent.name for p in new_fail)), flush=True); return 0
        new_fin = finished_queues() - fin0
        if new_fin:
            print('WAKE queue finished: ' + ', '.join(sorted(n for n, _ in new_fin)), flush=True); return 0
        for f in a.watch_file:
            if Path(f).exists():
                print(f'WAKE file appeared: {f}', flush=True); return 0
        if now - t0 >= a.max_minutes * 60:
            print(f'WAKE periodic check-in after {a.max_minutes:g} min (util {u}%)', flush=True); return 0
        time.sleep(a.poll)


if __name__ == '__main__':
    sys.exit(main())
