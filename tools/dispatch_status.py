"""v1.0.0 (2026-10-12): lane-aware status without private session-store reads.

Changelog: preserve per-log status lines, prefer dispatcher terminal provenance,
and show BUSY/ERROR lanes even when no report or log exists yet.
"""
from pathlib import Path
import json
import os
import re
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import dispatch_runtime as dr, job_runtime as rt


def log_status(log):
    """Legacy logs remain diagnostic; no terminal success is inferred from them."""
    tokens, error, ended = 0, '', False
    with log.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            tokens += sum(map(int, re.findall(r'"input_tokens"\s*:\s*(\d+)', line)))
            if re.search(r'"type"\s*:\s*"(?:error|turn\.failed)"|ERROR: unexpected status|Reconnecting\.\.\. 5/5|hit your usage limit|out of credits', line):
                match = re.search(r'"message"\s*:\s*"([^"\n]{0,90})', line)
                error = match[1] if match else 'execution error'
            ended = ended or bool(re.search(r'tokens used|"type"\s*:\s*"turn\.completed"', line))
    state = ('ERROR '+error if error else 'ENDED-NO-OUTPUT' if ended else
             'STALE' if time.time()-log.stat().st_mtime > 1200 else 'RUNNING')
    return state, f' in_tokens={tokens/1e6:.2f}M' if tokens else ''


def report_state(out):
    pointer = Path(str(out)+'.dispatch.json')
    if not pointer.exists():
        return None
    try:
        p = rt.read_json(pointer)
        run = Path(p['run'])
        job = rt.read_json(run/'job.json')
        if job['run_id'] != p['run_id'] or job['out'] != str(out.resolve()):
            return 'ERROR dispatch identity differs'
        if job['phase'] != 'terminal':
            return 'BUSY lane '+p['lane']
        if job.get('release'):
            return 'ERROR released after process tree death'
        receipt = rt.read_json(run/'receipt.json')
        if (receipt['run_id'] != p['run_id'] or receipt['out'] != job['out'] or
                rt.read_json(str(out)+'.receipt.json') != receipt):
            return 'ERROR terminal identity differs'
        if receipt['state'] != 'SUCCESS' or receipt['exit_code'] != 0:
            return 'ERROR terminal FAILED'
        if (receipt['report_sha256'] != dr.sha(out) or
                receipt['report_sha256'] != dr.sha(run/'report.md')):
            return 'ERROR report hash differs'
        return 'DONE'
    except (OSError, ValueError, KeyError, TypeError):
        return 'ERROR terminal evidence missing or invalid'


def main(args=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    args = sys.argv[1:] if args is None else args
    if len(args) != 1:
        print('usage: astra_status.sh <directory>')
        return 2
    directory = Path(args[0])
    runtime = dr.Lanes(os.environ.get('KNEE_WORKSPACE_ROOT', rt.CODE_ROOT))
    try:
        for log in sorted(directory.glob('*.log')):
            out = log.with_suffix('.md')
            alternative = log.with_name(log.stem+'-out.md')
            if alternative.is_file():
                out = alternative
            legacy, tokens = log_status(log)
            state = report_state(out)
            if state is None:
                state = 'DONE' if out.is_file() and out.stat().st_size else legacy
            print(f'{log.stem}: {state}{tokens}')
        # Read each active lane directly: registry text may be old or absent.
        for lane in dr.LANES:
            path = runtime.file(lane, 'state.json')
            if not path.exists():
                if runtime.busy(lane):
                    print(f'{lane}: BUSY pid {runtime.read(lane,"lock","reserved")}')
                continue
            try:
                job = rt.read_json(path)
                if job['phase'] != 'terminal':
                    state = 'BUSY pid '+runtime.read(lane,'lock','reserved')
                else:
                    state = report_state(Path(job['out'])) or 'ERROR terminal evidence missing'
                print(f'{lane}: {state}')
            except (OSError, ValueError, KeyError, TypeError):
                print(f'{lane}: ERROR lane metadata invalid')
        return runtime.dispatch(['lanes'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('ERROR '+str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
