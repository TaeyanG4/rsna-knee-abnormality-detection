"""H-B-155 transfer v1.0.0: binary-safe four tar/SSH streams; default plan only.

Only B may run --execute after host trust and transfer review. No credentials
are read. Destination is fixed; receiver validates every exact byte and hash.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import shlex
import subprocess
import sys
from tools import hb155_transfer_v1 as t
from tools import hb155_build_v1 as b

DEST='/workspace/rsna-knee-abnormality-detection'
VERIFY='''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
manifest=root/sys.argv[2]
if len(sys.argv)>3:
 with manifest.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==sys.argv[3], 'inventory hash mismatch'
m=json.loads(manifest.read_text())
assert m['status']=='READY'
total=0
for name,info in m['files'].items():
 p=root/name
 assert p.resolve().is_relative_to(root) and not any(x.is_symlink() for x in (p,*p.parents))
 assert p.stat().st_size==info['bytes'], name
 with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==info['sha256'], name
 total+=p.stat().st_size
assert total==m['bytes']
print(json.dumps(dict(status='VERIFIED',files=len(m['files']),bytes=total)))
'''
CAPACITY="import pathlib,shutil; p=pathlib.Path('/workspace'); assert shutil.disk_usage(p).total>=250000000000; assert shutil.disk_usage(p).free>=REQUIRED; (p/'rsna-knee-abnormality-detection').mkdir(exist_ok=True)"

def pipeline(tar_command,ssh_command):
    # Keep bytes in OS pipes, never PowerShell text conversion or full buffering.
    producer=subprocess.Popen(tar_command,cwd=b.ROOT,stdout=subprocess.PIPE)
    consumer=None
    try:
        consumer=subprocess.Popen(ssh_command,stdin=producer.stdout)
        producer.stdout.close()
        remote=consumer.wait(); local=producer.wait()
        b.need(local==0 and remote==0,f'tar/SSH failed: {local}/{remote}; rerun only after diagnosis')
    finally:
        for child in (consumer,producer):
            if child is not None and child.poll() is None:child.kill();child.wait()

def plan(host,port):
    b.need(re.fullmatch(r'[A-Za-z0-9_.@-]+',host) is not None and not host.startswith('-'),'invalid SSH host')
    b.need(1<=port<=65535,'invalid port')
    manifest=b.WORK/'transfer-inventory.json';m=b.read(manifest)
    t.streams(manifest)
    ssh=['ssh','-p',str(port),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',host]
    commands=[]
    for index in range(4):
        names=b.WORK/f'transfer-stream-{index}.txt'
        # Inventory is bootstrap metadata; all payload files are disjoint.
        tar=['tar','-cf','-','-T',str(names)]
        if index==0:tar += [b.rel(manifest)]
        commands.append((tar,ssh+['tar -xf - -C '+shlex.quote(DEST)]))
    verify=ssh+['python3 -c '+shlex.quote(VERIFY)+' '+shlex.quote(DEST)+' '+shlex.quote(b.rel(manifest))+' '+b.sha(manifest)]
    capacity=ssh+['python3 -c '+shlex.quote(CAPACITY.replace('REQUIRED',str(m['bytes']+100*(1<<30))))]
    return dict(status='PLAN_ONLY',bytes=m['bytes'],streams=commands,capacity=capacity,verify=verify)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args(argv);v=plan(a.host,a.port)
    if a.execute:
        subprocess.run(v['capacity'],check=True)
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda pair:pipeline(*pair),v['streams']))
        subprocess.run(v['verify'],check=True);v['status']='TRANSFER_VERIFIED'
    print(json.dumps(v));return 0

if __name__=='__main__':sys.exit(main())
