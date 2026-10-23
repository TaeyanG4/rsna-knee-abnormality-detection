"""H-B-165 transfer v1.0.0: ship ONLY new/changed small files to the Vast 4090 that already holds
the H-B-155 tree and both caches; verify the remote state; return fit results. Plan-only by default.

Inventory classes: `ship` = files absent on the remote (new H-B-165 code/tests/stage files and the
EXP099 fold-3 source files), `expect` = files the H-B-165 gates read that the remote already holds
(H-B-155 shipped inventory or returned H-B-155 evidence, identical SHA256), never caches/weights.
The receiver checks every ship AND expect file's size and SHA256. Only B runs --execute, after
the independent package review; strict SSH host-key checking stays on; no credentials are read.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from tools import hb165_build_v1 as b
from tools import hb155_build_v1 as parent
from tools import hb155_exclusions_v1 as exclusions
from tools.hb150_io_v1 import atomic_json

VERSION = '1.0.0'
DEST = '/workspace/rsna-knee-abnormality-detection'
PARENT_INVENTORY = parent.WORK/'transfer-inventory.json'
PARENT_INVENTORY_SHA256 = '5741184245a535009f06139830b7d34ffcce278882db7eb2ab2024edc8c3d2ae'
PARENT_RESULTS = parent.WORK/'result-inventory.json'
PARENT_RESULTS_SHA256 = '639b14a270be4cd8962695eddec8c658998fc62ee463f27a624fd92e10de63e5'
MAX_SHIP_BYTES = 16 << 20
NEW_CODE = ['tools/hb165_build_v1.py', 'tools/hb165_fit_v1.py', 'tools/hb165_watchdog_v1.py', 'tools/hb165_readout_v1.py',
            'tools/hb165_transfer_v1.py', 'tests/test_hb165_build_v1.py', 'tests/test_hb165_readout_v1.py',
            'tests/test_hb165_copies_v1.py', 'tests/test_hb165_transfer_v1.py']
STAGE = ['coordinator-decision.json', 'registration.json', 'REGISTRATION.md', 'gold58-ids.json', 'fold3/ordered-ids.json',
         'fold3/group-metadata.csv', 'fold-binding.json', 'exclusions-check.json', 'prepared.json', 'cpu-prepared.json',
         b.QUEUES['smoke'], b.QUEUES['fit'], 'LAUNCH.md']
FORBIDDEN = ('storage/', '.local/scratch/astra-b3/H-B-147/native448/', '.local/scratch/slot-b/hb155/crop-cache-v220/', '.local/relabel')
VERIFY = r'''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(); manifest=root/sys.argv[2]
with manifest.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==sys.argv[3], 'inventory hash mismatch'
m=json.loads(manifest.read_text()); assert m['schema']=='hb165-transfer/v1' and m['status']=='READY'
bad=[]; n=0
for kind in ('ship','expect'):
  for name,info in m[kind].items():
    p=root/name
    try:
      assert p.resolve().is_relative_to(root) and not any(x.is_symlink() for x in (p,*p.parents) if x!=root)
      assert p.stat().st_size==info['bytes']
      with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==info['sha256']
      n+=1
    except (AssertionError,OSError) as e: bad.append(kind+':'+name)
print(json.dumps(dict(status='VERIFIED' if not bad else 'MISMATCH',checked=n,bad=bad[:50],bad_count=len(bad))))
sys.exit(1 if bad else 0)
'''
PRECHECK = r'''import json,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); need=[root/x for x in sys.argv[2:]]
missing=[str(p.relative_to(root)) for p in need if not p.is_file()]
u=shutil.disk_usage(root) if root.is_dir() else None
print(json.dumps(dict(root_exists=root.is_dir(),missing=missing,free_bytes=u.free if u else None,total_bytes=u.total if u else None)))
'''


def need(ok, why):
    b.need(ok, why)


def rel(p):
    return b.rel(p)


def entry(path):
    p = b.ROOT/path
    return dict(bytes=p.stat().st_size, sha256=b.sha(p))


def parent_files():
    need(b.sha(PARENT_INVENTORY) == PARENT_INVENTORY_SHA256, 'H-B-155 transfer inventory changed')
    need(b.sha(PARENT_RESULTS) == PARENT_RESULTS_SHA256, 'H-B-155 result inventory changed')
    shipped = {k: v['sha256'] for k, v in b.read(PARENT_INVENTORY)['files'].items()}
    returned = {k: v['sha256'] for k, v in b.read(PARENT_RESULTS)['files'].items()}
    return shipped, returned


def parent_evidence():
    """Files the H-B-165 gates read from the H-B-155 tree (runtime/crop reviews, exclusions, cache metadata)."""
    P = parent.WORK; files = [P/'runtime'/n for n in ('4090.json', '4070.json', 'kaggle.json', 'vs4070.json', 'vskaggle.json')]
    files += [P/'runtime-review.json', P/'crop-review.json', exclusions.QA, exclusions.RECORD, exclusions.OUT/'fold-binding.json',
              exclusions.HB145_RECORD, P/'fold2/ordered-ids.json', P/'fold2/group-metadata.csv']
    crop, control = b.ROOT/parent.CROP, b.ROOT/parent.CONTROL
    files += [crop/n for n in ('manifest.json', 'ids.npy', 'masks.npy', 'side_unknown.npy', 'qa-summary.json')]
    files += [control/n for n in ('manifest.json', 'ids.npy', 'masks.npy')]
    files += [crop/name for name in b.read(crop/'qa-summary.json')['overlay_sha256']]
    m = b.read(crop/'manifest.json')
    for index, s in enumerate(m['shards']):
        if any(s['start'] <= pos < s['stop'] for pos in (2917, 2918)):
            files += [crop/s['qa_file'], crop/'side-audit'/f'chunk-{index:03d}.json']
    return sorted({rel(p) for p in files})


def classify():
    shipped, returned = parent_files()
    needed = set(parent_evidence())
    for seed in b.SEEDS:
        for arm in b.ARMS:
            for engineering in (False, True):
                cp = b.config_path(arm, seed, engineering); c = b.validate(cp)
                needed.add(rel(cp)); needed.update(c['inputs'])
    needed.update(NEW_CODE); needed.update(rel(b.WORK/n) for n in STAGE)
    ship, expect, drift = {}, {}, []
    for path in sorted(needed):
        p = b.ROOT/path
        need(p.is_file(), 'missing local file: '+path)
        e = entry(path); known = shipped.get(path) or returned.get(path)
        if known == e['sha256']:
            expect[path] = e
            continue
        if known is not None:
            drift.append(path)
        need(not path.startswith(FORBIDDEN) and e['bytes'] <= MAX_SHIP_BYTES, 'refusing to ship cache/weights/labels/large file: '+path)
        ship[path] = e
    return ship, expect, drift


def inventory():
    ship, expect, drift = classify()
    need(not drift, 'H-B-155 files changed locally since its transfer; explicit review required: '+', '.join(drift))
    result = dict(schema='hb165-transfer/v1', version=VERSION, status='READY', destination=DEST,
                  ship=ship, expect=expect, bytes=sum(v['bytes'] for v in ship.values()),
                  expect_bytes=sum(v['bytes'] for v in expect.values()), drift=drift,
                  note='expect = already on the remote (H-B-155 transfer/returned evidence), verified by SHA256 after extraction; '
                       'no cache, weights or label bytes are shipped')
    atomic_json(b.WORK/'transfer-inventory.json', result)
    (b.WORK/'transfer-files.txt').write_text(''.join(p+'\n' for p in sorted(ship)), encoding='utf-8', newline='\n')
    (b.WORK/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+p+'\n' for p, v in sorted({**ship, **expect}.items())),
                                     encoding='utf-8', newline='\n')
    return dict(status='READY', ship_files=len(ship), ship_bytes=result['bytes'], expect_files=len(expect),
                expect_bytes=result['expect_bytes'], inventory_sha256=b.sha(b.WORK/'transfer-inventory.json'))


def ssh_base(host, port):
    need(re.fullmatch(r'[A-Za-z0-9_.@-]+', host) is not None and not host.startswith('-'), 'invalid SSH host')
    need(1 <= int(port) <= 65535, 'invalid port')
    return ['ssh', '-p', str(int(port)), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', host]


def plan(host, port):
    ssh = ssh_base(host, port); manifest = b.WORK/'transfer-inventory.json'; m = b.read(manifest)
    need(m.get('schema') == 'hb165-transfer/v1' and m.get('status') == 'READY', 'inventory not ready')
    probe = [rel(parent.WORK/'review.json'), parent.CONTROL+'/manifest.json', parent.CROP+'/manifest.json']
    precheck = ssh+['python3 -c '+shlex.quote(PRECHECK)+' '+shlex.quote(DEST)+' '+' '.join(shlex.quote(p) for p in probe)]
    # Bootstrap metadata (inventory + its derived list/sums) travels with the payload, outside its byte total.
    tar = ['tar', '-cf', '-', '-T', str(b.WORK/'transfer-files.txt'), rel(manifest), rel(b.WORK/'transfer-files.txt'),
           rel(b.WORK/'SHA256SUMS')]
    receive = ssh+['tar -xf - -C '+shlex.quote(DEST)]
    verify = ssh+['python3 -c '+shlex.quote(VERIFY)+' '+shlex.quote(DEST)+' '+shlex.quote(rel(manifest))+' '+b.sha(manifest)]
    return dict(status='PLAN_ONLY', ship_files=len(m['ship']), ship_bytes=m['bytes'], expect_files=len(m['expect']),
                precheck=precheck, tar=tar, receive=receive, verify=verify, fit_disk_required_bytes=b.disk_requirement('fit'))


def pipeline(producer_cmd, consumer_cmd, cwd):
    producer = subprocess.Popen(producer_cmd, cwd=cwd, stdout=subprocess.PIPE)
    consumer = None
    try:
        consumer = subprocess.Popen(consumer_cmd, cwd=cwd, stdin=producer.stdout)
        producer.stdout.close()
        remote = consumer.wait(); local = producer.wait()
        need(local == 0 and remote == 0, f'pipeline failed: producer {local} / consumer {remote}; diagnose before rerun')
    finally:
        for child in (consumer, producer):
            if child is not None and child.poll() is None:
                child.kill(); child.wait()


def send(host, port):
    v = plan(host, port)
    pre = json.loads(subprocess.run(v['precheck'], check=True, capture_output=True).stdout.decode())
    need(pre['root_exists'] and not pre['missing'], 'remote is not the H-B-155 tree: '+json.dumps(pre))
    pipeline(v['tar'], v['receive'], b.ROOT)
    out = subprocess.run(v['verify'], capture_output=True)
    report = json.loads(out.stdout.decode() or '{}')
    need(out.returncode == 0 and report.get('status') == 'VERIFIED', 'remote verification failed: '+json.dumps(report))
    return dict(status='TRANSFER_VERIFIED', checked=report['checked'], free_bytes=pre['free_bytes'],
                fit_disk_required_bytes=v['fit_disk_required_bytes'],
                disk_ok_for_fits=pre['free_bytes'] >= v['fit_disk_required_bytes'])


RESULT_NAMES = ('heartbeat.json', 'summary.json', 'runtime.json', 'final.pt', 'val.csv', 'val.receipt.json', 'val_k24.csv',
                'val_k24.receipt.json', 'val_k36.csv', 'val_k36.receipt.json', 'gold.csv', 'gold.receipt.json')


def results():
    """Run on the remote after the fit queue: exact return list; touches only status first."""
    for seed in b.SEEDS:
        for arm in b.ARMS:
            need(b.read(b.WORK/'fits'/b.member(arm, seed)/'heartbeat.json').get('status') == 'complete', 'fit incomplete')
    W = b.WORK; files = [W/'runtime/4090.json', W/'review.json', W/'gpu-budget.json']
    if (W/'runtime-review.json').exists():
        files.append(W/'runtime-review.json')
    for seed in b.SEEDS:
        for arm in b.ARMS:
            name = b.member(arm, seed)
            files += [W/'cpu'/name/'summary.json', W/'cuda'/name/'summary.json', W/'cuda'/name/'healthy-rates.json']
            files += [W/'fits'/name/n for n in RESULT_NAMES]
            for mode in ('smoke', 'fit'):
                files += [W/'attempts'/f'{mode}-{name}-v100'/n for n in ('command.json', 'receipt.json')]
    value = dict(schema='hb165-results/v1', status='RESULTS_COMPLETE', files={rel(p): dict(bytes=p.stat().st_size, sha256=b.sha(p)) for p in files},
                 shipped_unchanged={p: entry(p) for p in sorted(rel(b.WORK/n) for n in STAGE if n != 'LAUNCH.md')})
    atomic_json(W/'result-inventory.json', value)
    (W/'result-files.txt').write_text(''.join(p+'\n' for p in sorted(value['files'])), encoding='utf-8', newline='\n')
    return dict(status='RESULTS_COMPLETE', files=len(value['files']), bytes=sum(f['bytes'] for f in value['files'].values()))


def verify_results(root=None):
    root = Path(root or b.ROOT); m = b.read(b.WORK/'result-inventory.json')
    need(m.get('schema') == 'hb165-results/v1' and m.get('status') == 'RESULTS_COMPLETE', 'result inventory invalid')
    for kind in ('files', 'shipped_unchanged'):
        for name, f in m[kind].items():
            p = (root/name).resolve()
            need(p.is_relative_to(b.WORK.resolve()) and p.stat().st_size == f['bytes'] and b.sha(p) == f['sha256'], 'hash mismatch: '+name)
    return dict(status='RESULTS_VERIFIED', files=len(m['files']), shipped_unchanged=len(m['shipped_unchanged']))


def fetch(host, port):
    ssh = ssh_base(host, port); inv = rel(b.WORK/'result-inventory.json'); lst = rel(b.WORK/'result-files.txt')
    raw = subprocess.run(ssh+['cat '+shlex.quote(DEST+'/'+inv)], check=True, capture_output=True).stdout
    m = json.loads(raw.decode())
    need(m.get('schema') == 'hb165-results/v1', 'remote result inventory invalid')
    for name, f in m['files'].items():
        p = (b.ROOT/name).resolve()
        need(p.is_relative_to(b.WORK.resolve()), 'result path outside the stage: '+name)
        need(not p.exists() or b.sha(p) == f['sha256'], 'local file differs; refusing to overwrite: '+name)
    pipeline(ssh+['cd '+shlex.quote(DEST)+' && tar -cf - -T '+shlex.quote(lst)+' '+shlex.quote(inv)+' '+shlex.quote(lst)],
             ['tar', '-xf', '-', '-C', str(b.ROOT)], b.ROOT)
    return verify_results()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('inventory', 'plan', 'send', 'results', 'fetch', 'verify-results'))
    p.add_argument('--host'); p.add_argument('--port', type=int); p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    try:
        if a.command == 'inventory':
            result = inventory()
        elif a.command in ('plan', 'send', 'fetch'):
            need(a.host and a.port, '--host and --port required')
            if a.command == 'plan' or not a.execute:
                result = plan(a.host, a.port) if a.command != 'fetch' else dict(status='PLAN_ONLY', note='fetch requires --execute')
            else:
                result = send(a.host, a.port) if a.command == 'send' else fetch(a.host, a.port)
        elif a.command == 'results':
            result = results()
        else:
            result = verify_results()
    except (ValueError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        result = dict(status='HOLD', reason=type(exc).__name__+': '+str(exc))
    print(json.dumps(result)); return 2 if result['status'] == 'HOLD' else 0


if __name__ == '__main__':
    sys.exit(main())
