"""Vast split ops v1.1.0. Offline by default; live actions require B's explicit flag.

Changelog 1.1.0: fixed two-fold/4h allocation; one start, five-minute GPU regain,
rate/credit guard, immutable attempts, fresh receipts separate from historical prep.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.local/scratch/pl-vast-split/release-v110/account'
INSTANCE = 54941539
MACHINE = 140617
MAX_HOURS = 4
MAX_RATE = .228
PROTECTED = 5.0
RESERVE = .60


def budget():
    return dict(version='1.1.0', instance=INSTANCE, max_hours=MAX_HOURS,
                max_rate=MAX_RATE, maximum_compute_usd=MAX_HOURS*MAX_RATE,
                reserve_usd=RESERVE, allocation_usd=round(MAX_HOURS*MAX_RATE+RESERVE, 3),
                assumed_usable_usd=4.26, protected_usd=PROTECTED,
                remaining_usable_usd=round(4.26-MAX_HOURS*MAX_RATE-RESERVE, 3),
                status='CONDITIONAL_UNMEASURED_3090')


def stop_reason(row, credit, elapsed, *, gpu_admitted=False):
    if not isinstance(row, dict) or row.get('id') != INSTANCE or row.get('machine_id') != MACHINE:
        return 'instance/machine identity unavailable or changed'
    if row.get('gpu_name') not in ('RTX 3090', 'RTX_3090') or row.get('num_gpus') != 1:
        return 'GPU contract changed'
    try:
        values = (float(credit), float(row['dph_total']), float(elapsed))
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite')
        credit, rate, elapsed = values
    except (ValueError, TypeError, KeyError):
        return 'credit/rate/time unavailable'
    if not 0 < rate <= MAX_RATE or elapsed < 0:
        return 'rate/time outside cap'
    if credit <= PROTECTED + RESERVE:
        return 'protected credit/reserve floor'
    if elapsed >= MAX_HOURS * 3600:
        return 'original-start four-hour deadline'
    if not gpu_admitted and elapsed >= 300:
        return 'GPU not physically regained within five minutes'
    if credit - PROTECTED < max(0, MAX_HOURS-elapsed/3600)*rate + RESERVE:
        return 'remaining four-hour allocation exceeds live credit'
    return None


def load_provider():
    # Import ONLY after explicit live flag. Never inspect provider/key files here.
    path = ROOT / '.local/hc06-illinois-stable-20261009-v10/create_once.py'
    spec = importlib.util.spec_from_file_location('pl_vast_existing_provider', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.provider()


def write(path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers open briefly; no rename/delete. Retry bounded Windows sharing errors.
    for attempt in range(40):
        try:
            with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def stop_and_confirm(provider):
    error = None
    try:
        provider.stop(INSTANCE)
    except Exception as exc:
        error = type(exc).__name__
    try:
        row = provider.call('show', 'instance', str(INSTANCE))
        ok = row.get('id') == INSTANCE and row.get('actual_status') == 'exited' and row.get('intended_status') == 'stopped'
    except Exception:
        ok = False
    return dict(version='1.1.0', id=INSTANCE, at=time.time(), cli_error_type=error,
                status='VERIFIED_EXITED_STOPPED' if ok else 'STOP_UNCONFIRMED_B_MUST_RECONCILE',
                disk_preserved=True if ok else None)


def sample(provider, balance_field, started, gpu_admitted):
    try:
        row = provider.call('show', 'instance', str(INSTANCE))
        credit = provider.user()[balance_field]
        elapsed = time.time()-started if started is not None else 0
        reason = stop_reason(row, credit, elapsed, gpu_admitted=gpu_admitted)
        result = dict(version='1.1.0', at=time.time(), id=INSTANCE,
                      credit=float(credit) if math.isfinite(float(credit)) else None,
                      stop_reason=reason, actual=row.get('actual_status'), intended=row.get('intended_status'),
                      ssh_host=row.get('ssh_host'), ssh_port=row.get('ssh_port'),
                      dph_total=float(row['dph_total']) if math.isfinite(float(row['dph_total'])) else None)
    except Exception:
        result = dict(version='1.1.0', at=time.time(), id=INSTANCE, stop_reason='account/instance read failed')
    return result


def admit_gpu(snapshot, started, now=None):
    now = time.time() if now is None else now
    if not 0 <= now-started < 300:
        raise ValueError('GPU regain deadline expired')
    gpus = snapshot.get('gpus', [])
    if len(gpus) != 1 or 'RTX 3090' not in gpus[0].get('name', '') or float(gpus[0].get('memory_mib', 0)) < 23000:
        raise ValueError('requires one physical RTX3090 / 24GB')
    if snapshot.get('processes') != [] or not gpus[0].get('uuid'):
        raise ValueError('foreign GPU process or missing physical UUID')
    return dict(version='1.1.0', status='PHYSICAL_GPU_ADMITTED', at=now,
                started_unix=started, gpu=gpus[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['budget', 'check', 'start', 'watch', 'stop', 'admit-gpu'])
    ap.add_argument('--execute-B-only', action='store_true')
    ap.add_argument('--snapshot', type=Path)
    a = ap.parse_args(argv)
    if a.action == 'budget':
        print(json.dumps(budget()))
        return 0
    if not a.execute_B_only:
        ap.error('live operations disabled; B must pass --execute-B-only')
    start_path = WORK/'start-attempt.json'
    started = json.loads(start_path.read_text())['started_unix'] if start_path.exists() else None
    if a.action in ('watch', 'admit-gpu') and (started is None or not math.isfinite(started) or started > time.time()):
        ap.error('valid original start receipt required')
    if a.action == 'admit-gpu':
        if a.snapshot is None:
            ap.error('--snapshot required')
        result = admit_gpu(json.loads(a.snapshot.read_text(encoding='utf-8-sig')), started)
        write(WORK/'gpu-admission.json', result, exclusive=True)
        print(json.dumps(result))
        return 0
    provider, config = load_provider()
    if a.action == 'stop':
        result = stop_and_confirm(provider)
        write(WORK/'stop-result.json', result)
        print(json.dumps(result))
        return 0 if result['status'] == 'VERIFIED_EXITED_STOPPED' else 2
    if a.action == 'start':
        preflight = sample(provider, config['balance_field'], None, False)
        if preflight['stop_reason'] or preflight.get('actual') != 'exited' or preflight.get('intended') != 'stopped':
            raise ValueError('pre-start identity/budget/stopped-state HOLD')
        result = dict(version='1.1.0', id=INSTANCE, started_unix=time.time(), deadline_hours=MAX_HOURS)
        write(start_path, result, exclusive=True)
        try:
            provider.call('start', 'instance', str(INSTANCE))
        except Exception as exc:
            result['cli_error_type'] = type(exc).__name__
        print(json.dumps(dict(result, action='START_ATTEMPTED_RECONCILE_STATUS_NEVER_RETRY')))
        return 0
    while True:
        admission = WORK/'gpu-admission.json'
        admitted = False
        if admission.exists():
            try:
                proof = json.loads(admission.read_text())
                admitted = proof['started_unix'] == started and proof['status'] == 'PHYSICAL_GPU_ADMITTED'
            except (ValueError, KeyError):
                pass
        result = sample(provider, config['balance_field'], started, admitted)
        try:
            write(WORK/('guard-check.json' if a.action == 'check' else 'guard-heartbeat.json'), result)
        except PermissionError:
            pass  # Heartbeat write failure is nonfatal; operator freshness check stops legs.
        if a.action == 'check':
            print(json.dumps(result))
            return 2 if result['stop_reason'] else 0
        if result['stop_reason']:
            stopped = stop_and_confirm(provider)
            write(WORK/'stop-result.json', stopped)
            print(json.dumps(dict(reason=result['stop_reason'], **stopped)))
            return 2
        if result.get('actual') == 'exited' and result.get('intended') == 'stopped' and admitted:
            return 0
        time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
