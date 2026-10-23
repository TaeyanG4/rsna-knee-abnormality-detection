"""Archive actual native Codex app tasks; coordinator-attested, not CLI execution.

Version 1.0.0, 2026-10-18: additive adapter for the existing dispatch-terminal/v1
trust boundary. Never changes gates, old receipts, or claims platform signatures.
Only B records a completion observed through the native collaboration tool.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

VERSION = '1.0.0'
ROOT = Path(__file__).resolve().parents[1]
HOST = 'Codex app native subagent'
MODEL = 'gpt-6-astra'
SCHEMA = 'native-codex-completion/v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def wire(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def parse(data):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data.decode('utf-8-sig'), object_pairs_hook=pairs,
                      parse_constant=lambda _: need(False, 'nonfinite JSON'))


def local(path, root):
    root = Path(root).resolve()
    path = Path(path)
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    need(resolved.is_relative_to(root), 'path outside project')
    for part in (path, *path.parents):
        if part == root:
            break
        need(not part.is_symlink() and not getattr(part, 'is_junction', lambda: False)(),
             'symlink/junction path refused')
    return resolved


def nonempty(path):
    need(path.is_file(), 'missing input file')
    data = path.read_bytes()
    need(bool(data.strip()), 'empty input file')
    return data


def completion(path, root):
    path = local(path, root)
    raw = nonempty(path)
    evidence = parse(raw)
    required = {'schema', 'host', 'model', 'agent_id', 'lane', 'prompt', 'report',
                'terminal_status', 'observed_at', 'prompt_sha256', 'report_sha256',
                'completion_message'}
    need(isinstance(evidence, dict) and set(evidence) == required, 'completion evidence schema fields')
    need(all(isinstance(v, str) and v.strip() for v in evidence.values()), 'empty/nontext completion field')
    need(evidence['schema'] == SCHEMA and evidence['host'] == HOST and evidence['model'] == MODEL,
         'native host/model/schema mismatch')
    need(evidence['terminal_status'] == 'completed', 'native task did not complete')
    need(re.fullmatch(r'native-astra-[br][1-9][0-9]*', evidence['lane']), 'invalid native lane')
    need(re.fullmatch(r'/root/[a-z0-9_]+(?:/[a-z0-9_]+)*', evidence['agent_id']), 'canonical native agent id required')
    timestamp = dt.datetime.fromisoformat(evidence['observed_at'])
    need(timestamp.tzinfo is not None and timestamp.utcoffset() is not None, 'timezone required')
    prompt, report = (local(evidence[k], root) for k in ('prompt', 'report'))
    need(len({path, prompt, report}) == 3, 'completion/prompt/report paths overlap')
    pdata, rdata = nonempty(prompt), nonempty(report)
    need(evidence['prompt_sha256'] == digest(pdata) and evidence['report_sha256'] == digest(rdata),
         'completion prompt/report hash mismatch')
    return path, raw, evidence, prompt, pdata, report, rdata


def exclusive(path, data):
    # An existing or partially written file is never replaced or silently repaired.
    for attempt in range(40):
        try:
            stream = path.open('xb')
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(.05 * (attempt + 1), 1))
            continue
        with stream:
            stream.write(data)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        return


def record(completion_path, note, root=ROOT):
    root = Path(root).resolve()
    source, raw, evidence, prompt, pdata, report, rdata = completion(completion_path, root)
    need(isinstance(note, str) and note.strip(), 'coordinator provenance note required')
    sidecar = local(str(report) + '.receipt.json', root)
    need(not sidecar.exists(), 'receipt already exists; never overwrite')
    run_id = uuid.uuid4().hex
    archive = local(root / '.local/astra-improve/lanes/runs' / evidence['lane'] / run_id, root)
    archive.mkdir(parents=True, exist_ok=False)
    log = wire({'recorder': 'native_codex_dispatch_record_v1', 'version': VERSION,
                'host': HOST, 'model': MODEL, 'agent_id': evidence['agent_id'],
                'lane': evidence['lane'], 'note': note,
                'trust_boundary': 'B attests an observed native collaboration completion; no CLI run or platform signature'})
    result = {
        'schema': 'dispatch-terminal/v1', 'run_id': run_id, 'lane': evidence['lane'],
        'out': str(report), 'state': 'SUCCESS', 'exit_code': 0,
        'prompt_sha256': digest(pdata), 'report_sha256': digest(rdata), 'log_sha256': digest(log),
        'host': HOST, 'model': MODEL, 'agent_id': evidence['agent_id'],
        'completion_source': str(source), 'completion_evidence_sha256': digest(raw),
        'native_terminal_status': evidence['terminal_status'],
        'completed': evidence['observed_at'], 'recorded_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'note': note, 'recorder': {'tool': 'tools/native_codex_dispatch_record_v1.py', 'version': VERSION},
        'trust_boundary': 'coordinator-attested native task completion; not a CLI process exit or cryptographic platform proof',
    }
    for name, data in [('prompt.txt', pdata), ('report.md', rdata), ('completion.json', raw),
                       ('run.log', log), ('receipt.json', wire(result))]:
        exclusive(archive / name, data)
    # Refuse a race or input mutation before publishing the gate-facing sidecar.
    need(source.read_bytes() == raw and prompt.read_bytes() == pdata and report.read_bytes() == rdata,
         'source changed while recording; partial archive retained')
    exclusive(sidecar, wire(result))
    verify(sidecar, root)
    return sidecar


def verify(receipt_path, root=ROOT):
    root = Path(root).resolve()
    sidecar = local(receipt_path, root)
    raw = nonempty(sidecar)
    value = parse(raw)
    need(value['host'] == HOST and value['model'] == MODEL and value['state'] == 'SUCCESS'
         and value['schema'] == 'dispatch-terminal/v1' and type(value['exit_code']) is int
         and value['exit_code'] == 0, 'receipt identity/status mismatch')
    need(re.fullmatch(r'native-astra-[br][1-9][0-9]*', value['lane'])
         and re.fullmatch(r'[a-f0-9]{32}', value['run_id']), 'receipt archive identity mismatch')
    archive = local(root / '.local/astra-improve/lanes/runs' / value['lane'] / value['run_id'], root)
    need(nonempty(archive / 'receipt.json') == raw, 'archive receipt mismatch')
    _, craw, evidence, prompt, pdata, report, rdata = completion(value['completion_source'], root)
    need(craw == nonempty(archive / 'completion.json') and digest(craw) == value['completion_evidence_sha256'],
         'completion evidence mismatch')
    need(value['lane'] == evidence['lane'] and value['agent_id'] == evidence['agent_id']
         and value['completed'] == evidence['observed_at'] and value['native_terminal_status'] == 'completed',
         'native evidence identity mismatch')
    need(local(value['out'], root) == report and sidecar == Path(str(report) + '.receipt.json'),
         'report-sidecar binding mismatch')
    for name, data, field in [('prompt.txt', pdata, 'prompt_sha256'), ('report.md', rdata, 'report_sha256')]:
        need(nonempty(archive / name) == data and digest(data) == value[field], 'archive ' + name + ' mismatch')
    need(digest(nonempty(archive / 'run.log')) == value['log_sha256'], 'archive log mismatch')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('record')
    p.add_argument('--completion', type=Path, required=True)
    p.add_argument('--note', required=True)
    p = sub.add_parser('verify')
    p.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'record':
            print(record(args.completion, args.note))
        else:
            value = verify(args.receipt)
            print(json.dumps({'status': 'VERIFIED', 'lane': value['lane'], 'agent_id': value['agent_id']}))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
