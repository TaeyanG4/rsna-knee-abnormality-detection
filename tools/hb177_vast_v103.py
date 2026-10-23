"""H-B-177 Vast transport/environment v1.0.3; explicit B-only network actions.

Changelog: add local environment admission; Vast transport/gates unchanged apart from v103 paths.
Inherited: minimal sealed bundle, CPU environment diff, isolated remote root,
hash-verified evidence/results roundtrip. Does not install, admit or launch GPU.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'.local/scratch/hb177/H-B-177/v1.0.3'
REMOTE='/workspace/rsna-knee-abnormality-detection/.local/hb177-v103'
REL='.local/scratch/hb177/H-B-177/v1.0.3'
HOST='root@137.175.22.196'
PORT='50985'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=30','-p',PORT,HOST]
SCP=['scp','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-P',PORT]
PACKAGES=('torch','torchvision','timm','numpy','opencv-python-headless','psutil','pydicom','safetensors','pillow','huggingface-hub','pyyaml')
# Same broad token/private-key patterns as team-repo-sync v1.0.2. Never print matches.
SECRET=re.compile(rb'AIza[0-9A-Za-z_-]{35}|KGAT_[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(ant-|proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY|"private_' rb'key_id"|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|sk_live_[A-Za-z0-9]{20,}|GOCSPX-[A-Za-z0-9_-]{20,}|ya29\.[A-Za-z0-9_-]{20,}')


def need(value,message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    from tools.hb150_io_v1 import atomic_json
    atomic_json(path,value)


def snapshot(cuda=False):
    """CPU snapshot never probes CUDA availability. CUDA details only inside admitted smoke."""
    import torch
    libs={}
    for name in PACKAGES:
        try: libs[name]=metadata.version(name)
        except metadata.PackageNotFoundError: libs[name]=None
    # Import actual runtime modules, not just distribution metadata.
    import cv2, timm, numpy, torchvision, pydicom
    value=dict(python=platform.python_version(),platform=platform.system(),packages=libs,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        imports=dict(cv2=cv2.__version__,timm=timm.__version__,numpy=numpy.__version__,
                     torchvision=torchvision.__version__,pydicom=pydicom.__version__),
        settings=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
            mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math_sdp=torch.backends.cuda.math_sdp_enabled()),cuda_initialized=torch.cuda.is_initialized())
    if cuda:
        p=torch.cuda.get_device_properties(0)
        value.update(gpu=p.name,capability=list(torch.cuda.get_device_capability(0)),gpu_bytes=p.total_memory)
    else:
        need(not torch.cuda.is_initialized(),'CPU environment check touched CUDA')
    return value


def environment_diff(reference,actual):
    differences={}
    for key in ('python','platform','cuda','cudnn','packages','imports','settings'):
        if reference[key]!=actual[key]: differences[key]=dict(local=reference[key],remote=actual[key])
    # Only OS/Python patch/minor and torchvision CUDA build suffix may differ silently.
    mismatch=[]
    for name,version in reference['packages'].items():
        other=actual['packages'].get(name)
        normalize=lambda x: x.split('+')[0] if name=='torchvision' and x else x
        if normalize(version)!=normalize(other): mismatch.append(name)
    for key in ('cuda','cudnn','settings'):
        if reference[key]!=actual[key]: mismatch.append(key)
    need(actual['python'].split('.')[:2] in (['3','11'],['3','12']),'unqualified Python major/minor')
    return differences,mismatch


def env_check():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    reference=read(STAGE/'local-environment.json')
    actual=snapshot()
    differences,mismatch=environment_diff(reference,actual)
    result=dict(status='PASS_ENV_CPU' if not mismatch else 'BLOCKED_ENV',
        actual=actual,differences=differences,mismatches=mismatch,
        local_sha256=sha(STAGE/'local-environment.json'),
        python_difference='3.12 Windows -> 3.11 Linux recorded; remote CPU replay required',
        kaggle_runtime='NOT CHECKED: no authenticated current Kaggle snapshot supplied; T4 inference parity required before release')
    write(STAGE/'remote-environment.json',result)
    need(not mismatch,'environment mismatch: '+','.join(mismatch)+'; use environment-pins.txt; no automatic install')
    return dict(status=result['status'],differences=list(differences),cuda_initialized=False)


def require_environment():
    e=read(STAGE/'remote-environment.json')
    need(e['status']=='PASS_ENV_CPU' and e['local_sha256']==sha(STAGE/'local-environment.json'),'missing/stale environment receipt')
    # Imports/settings are checked by CPU replay; live distribution drift fails every GPU gate.
    current={}
    for name in PACKAGES:
        current[name]=metadata.version(name)
    need(current==e['actual']['packages'],'live package drift')
    need(platform.python_version()==e['actual']['python'] and platform.system()=='Linux','live Python/platform drift')


def require_local_environment():
    reference=read(STAGE/'local-environment.json')
    current={name:metadata.version(name) for name in PACKAGES}
    need(current==reference['packages'],'live local package drift')
    need(platform.python_version()==reference['python'] and platform.system()==reference['platform']=='Windows',
         'live local Python/platform drift')


def safe_name(name):
    p=PurePosixPath(name)
    need(bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name,'unsafe archive path')
    need(not any(c in name for c in ('\n','\r','\x00')),'unsafe manifest path')
    return name


def verify_manifest(root,manifest):
    root=Path(root).resolve()
    need(len({x['path'] for x in manifest['files']})==len(manifest['files']),'duplicate manifest entries')
    for item in manifest['files']:
        p=root/safe_name(item['path'])
        need(p.resolve().is_relative_to(root) and not p.is_symlink(),'path escapes root')
        need(p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'transfer mismatch: '+item['path'])
    return dict(status='VERIFIED',files=len(manifest['files']),bytes=sum(x['bytes'] for x in manifest['files']))


def scan(path):
    # Scan whole allowlisted files with overlap, including the one binary init.
    with Path(path).open('rb') as f:
        tail=b''
        while block:=f.read(1024*1024):
            need(not SECRET.search(tail+block),'broad secret scan hit (value suppressed): '+Path(path).name)
            tail=block[-256:]


def pack():
    from tools import hb177_build_v103 as b
    b.select_host('vast'); b.registration(); b.require_cpu()
    p=b.parent()
    paths=[ROOT/x for x in b.SOURCES]
    paths += [b.PARENT_MANIFEST,b.PARENT_MANIFEST.parent/p['label_file_name'],b.INIT]
    paths += [STAGE/x for x in ('registration.json','local-environment.json','cpu-validation.json')]
    # Exactly pinned init is an offline serialized artifact, not a substitutable hub download.
    # New version has no need for any trained checkpoint, serving archive or Gold table.
    paths=sorted(set(paths))
    reference=read(STAGE/'local-environment.json')
    pins='\n'.join(f'{n}=={v}' for n,v in reference['packages'].items() if n not in ('torch','torchvision'))+'\n'
    (STAGE/'environment-pins.txt').write_text(pins,encoding='utf-8',newline='\n')
    paths.append(STAGE/'environment-pins.txt')
    entries=[]
    for pth in paths:
        rel=pth.relative_to(ROOT).as_posix(); safe_name(rel); scan(pth)
        need(pth.suffix not in ('.pt','.pth','.bin') or pth==b.INIT,'unapproved checkpoint')
        entries.append(dict(path=rel,bytes=pth.stat().st_size,sha256=sha(pth)))
    manifest=dict(schema='hb177-transfer/v1.0.3',contract='56132574',remote_root=REMOTE,files=entries,
        secret_scan='team-repo-sync v1.0.2 broad patterns: PASS',cache_included=False)
    write(STAGE/'transfer-inventory.json',manifest)
    sums=''.join(x['sha256']+'  '+x['path']+'\n' for x in entries)
    (STAGE/'transfer.sha256').write_text(sums,encoding='utf-8',newline='\n')
    with tarfile.open(STAGE/'transfer.tar','w') as tar:
        for pth in paths+[STAGE/'transfer-inventory.json',STAGE/'transfer.sha256']:
            tar.add(pth,arcname=pth.relative_to(ROOT).as_posix(),recursive=False)
    verify_manifest(ROOT,manifest)
    (STAGE/'transfer.tar.sha256').write_text(sha(STAGE/'transfer.tar')+'\n',encoding='ascii')
    return dict(status='PACKED',files=len(entries),payload_bytes=sum(x['bytes'] for x in entries),
        tar_bytes=(STAGE/'transfer.tar').stat().st_size,sha256=sha(STAGE/'transfer.tar'))


def run(args,**kwargs):
    subprocess.run(args,check=True,**kwargs)


def send():
    # Explicit action for B. Never called by register/validate/prepare/dry-run/pack.
    verify_manifest(ROOT,read(STAGE/'transfer-inventory.json'))
    need(sha(STAGE/'transfer.tar')==(STAGE/'transfer.tar.sha256').read_text().strip(),'sealed transfer archive drift')
    for item in read(STAGE/'transfer-inventory.json')['files']: scan(ROOT/item['path'])
    # Refuse an existing isolated root: no overwrite of active or reviewed code.
    run(SSH+[f'test ! -e {REMOTE} && mkdir -p {REMOTE}'])
    run(SCP+[str(STAGE/'transfer.tar'),HOST+':'+REMOTE+'/transfer.tar'])
    h=sha(STAGE/'transfer.tar')
    run(SSH+[f"cd {REMOTE} && echo '{h}  transfer.tar' | sha256sum -c - && tar -xf transfer.tar && sha256sum --quiet -c {REL}/transfer.sha256"])


def send_approval(fit=False):
    names=['pre-review.json','b-authorization.json']+(['fit-review.json'] if fit else [])
    for name in names:
        path=STAGE/name
        need(path.is_file(),'missing independent review/B authorization: '+name)
        scan(path)
    # B prepares these after reviewing remote CPU or completed CUDA evidence.
    for name in names:
        path=STAGE/name
        run(SCP+[str(path),HOST+':'+REMOTE+'/'+REL+'/'+name])
        run(SSH+[f"cd {REMOTE} && echo '{sha(path)}  {REL}/{name}' | sha256sum -c -"])


def result_pack(mode):
    from tools import hb177_build_v103 as b
    from tools.job_runtime import job_status
    if mode=='cpu':
        names=['registration.json','local-environment.json','remote-environment.json','cpu-validation.json','admit-dry-run.json']
    else:
        tag='HB177_FIT'+('_SMOKE' if mode=='smoke' else '')
        state=job_status(ROOT/'artifacts/logs',tag,tail=False)
        need(state['state']=='SUCCESS','results require completed successful gpu_job; no running-output reads')
        names=['registration.json','remote-environment.json','cpu-validation.json','pre-review.json','b-authorization.json',
               'cuda-smoke.json','cuda-runtime.json','smoke-attempt/receipt.json','smoke-attempt/command.json']
        if mode=='fit':
            complete=read(STAGE/'fit/complete.json')
            need(complete['checkpoint_sha256']==sha(STAGE/'fit/hb177-final-ema.pt'),'final checkpoint hash mismatch')
            names+=['fit-review.json','fit/complete.json','fit/hb177-final-ema.pt','fit/resume.pt']
            for attempt in sorted(STAGE.glob('fit-attempt-*')):
                for name in ('receipt.json','command.json'):
                    if (attempt/name).is_file(): names.append((attempt/name).relative_to(STAGE).as_posix())
        names+=['@logs/'+tag+'/'+name for name in ('job.json','receipt.json','run.log')]
    entries=[]
    paths=[]
    for name in names:
        path=ROOT/'artifacts'/name[1:] if name.startswith('@logs/') else STAGE/name
        dest=name.replace('@logs/','logs/')
        safe_name(dest); paths.append((path,dest))
        entries.append(dict(path=dest,bytes=path.stat().st_size,sha256=sha(path)))
    manifest=dict(schema='hb177-results/v1.0.3',mode=mode,files=entries)
    write(STAGE/('results-'+mode+'.json'),manifest)
    out=STAGE/('results-'+mode+'.tar')
    with tarfile.open(out,'w') as tar:
        for path,dest in paths: tar.add(path,arcname=dest,recursive=False)
        tar.add(STAGE/('results-'+mode+'.json'),arcname='results.json')
    (STAGE/('results-'+mode+'.sha256')).write_text(sha(out)+'\n',encoding='ascii')
    return dict(status='RESULTS_PACKED',mode=mode,bytes=out.stat().st_size,sha256=sha(out))


def unpack_results(archive,expected,destination):
    need(sha(archive)==expected,'results tar SHA256 mismatch')
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        need(len({m.name for m in members})==len(members),'duplicate archive entries')
        for m in members:
            safe_name(m.name); need(m.isfile(),'non-regular result member')
            target=destination/m.name
            need(target.resolve().is_relative_to(destination.resolve()) and not target.is_symlink(),'unsafe result destination')
        manifest=json.load(tar.extractfile('results.json'))
        need({m.name for m in members}=={'results.json'}|{x['path'] for x in manifest['files']},'unmanifested result members')
        tar.extractall(destination,filter='data')
    return verify_manifest(destination,read(destination/'results.json'))


def pull(mode):
    command=f'cd {REMOTE} && python3 -B -m tools.hb177_vast_v103 results-{mode}'
    run(SSH+[command])
    for ext in ('tar','sha256'):
        run(SCP+[HOST+':'+REMOTE+'/'+REL+'/results-'+mode+'.'+ext,str(STAGE/('results-'+mode+'.'+ext))])
    return unpack_results(STAGE/('results-'+mode+'.tar'),(STAGE/('results-'+mode+'.sha256')).read_text().strip(),STAGE/('returned-'+mode))


def remote_command(action):
    prefix=f'cd {REMOTE} && export KNEE_WORKSPACE_ROOT={REMOTE} && '
    if action=='remote-install':
        # Keep the existing torch/torchvision CUDA wheels; do not reinstall them.
        command=f'python3 -m pip install -r {REL}/environment-pins.txt && python3 -m pip check'
    elif action=='remote-cpu':
        commands=['python3 -B -m tools.hb177_vast_v103 verify-transfer',
                  'python3 -B -m tools.hb177_vast_v103 env-check']
        commands += ['python3 -B -m tools.hb177_build_v103 '+x for x in ('validate','prepare','admit-dry-run')]
        command=' && '.join(commands)
    elif action in ('launch-smoke','launch-fit'):
        mode=action.split('-')[1]
        # Admission also fails closed in both outer wrapper and watchdog child.
        command=f'python3 -B -m tools.gpu_queue {REL}/{mode}-queue.json'
    elif action=='status':
        command='python3 -B -m tools.gpu_job --list'
    else:
        raise ValueError('unknown remote action')
    return SSH+[prefix+command]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['pack','env-check','send','send-smoke-approval','send-fit-approval',
        'pull-cpu','pull-smoke','pull-fit','results-cpu','results-smoke','results-fit','verify-transfer',
        'remote-install','remote-cpu','launch-smoke','launch-fit','status'])
    parser.add_argument('--dry-run',action='store_true',help='Print remote command only; no SSH or mutation')
    args=parser.parse_args(); action=args.action
    if action in ('remote-install','remote-cpu','launch-smoke','launch-fit','status'):
        command=remote_command(action)
        value={'dry_run':True,'command':command} if args.dry_run else run(command)
    elif args.dry_run: raise ValueError('--dry-run is only for remote command actions')
    elif action=='pack': value=pack()
    elif action=='env-check': value=env_check()
    elif action=='send': value=send()
    elif action.startswith('send-'): value=send_approval(action=='send-fit-approval')
    elif action.startswith('pull-'): value=pull(action[5:])
    elif action.startswith('results-'): value=result_pack(action[8:])
    else: value=verify_manifest(ROOT,read(STAGE/'transfer-inventory.json'))
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__': main()
