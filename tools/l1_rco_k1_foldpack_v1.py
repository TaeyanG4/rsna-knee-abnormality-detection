"""H224 portable bundle v1.0.0: explicit input allowlist, byte pins, no downloader.
Changelog: offline source/weights bundle; raw ZIP and cache intentionally excluded.
"""
from __future__ import annotations
import argparse, importlib.metadata, json, os, shutil, sys, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from tools import l1_rco_k1_folds_v1 as k
BASE=k.BASE; PACKAGE=BASE/'package'; ARCHIVE=BASE/'h224-folds-v1.tar'

def files():
    paths=set(k.code_pins()) | set(k.s.PINS) | set(k.read(k.FULL_REG)['pins'])
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/k.s.CODE).rglob('*') if p.is_file())
    paths.update(p.relative_to(ROOT).as_posix() for p in (k.PARENT,k.FULL_REG,k.parent.K0_RESULT,BASE/'local32.json'))
    paths.update(['config/workspace.toml','training/src/raptor_common.py',
                  'tools/gpu_job.py','tools/job_runtime.py','tools/parity_runtime.py',
                  'tools/l1_rco_stage0_v120.py','tools/l1_rco_runtime_v120.py',
                  'tools/l1_rco_k1_state_v1.py','tools/l1_rco_k1_state_v2.py',
                  'tools/l1_rco_k1_v1.py','tools/l1_rco_k1_v2.py','tools/l1_rco_k1_v3.py',
                  'tools/l1_rco_k1_optim_v3.py','tools/l1_rco_k1_folds_v1.md'])
    for name in paths:
        p=ROOT/name
        k.need(p.resolve().is_relative_to(ROOT) and p.is_file(),'missing package input: '+name)
        k.need('__pycache__' not in p.parts and p.suffix not in ('.pyc','.zip'),'non-source package input')
        k.need(not any(x in name.lower() for x in ('api-key','credential','.relabel','.ssh')),'prohibited package input')
    return sorted(paths)

def build():
    k.s.verify_pins(); paths=files(); PACKAGE.mkdir(parents=True,exist_ok=True)
    existing={p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*') if p.is_file()}
    k.need(existing <= set(paths)|{'INVENTORY.json','SHA256SUMS','environment-local.json'},'package has extra files; do not overwrite a used bundle')
    entries=[]
    for name in paths:
        src=ROOT/name;dst=PACKAGE/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst);entries.append(dict(path=name,bytes=dst.stat().st_size,sha256=k.sha(dst)))
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    k.write(PACKAGE/'environment-local.json',dict(python=sys.version,packages=packages,
        note='Reference only. Linux CUDA wheels need B provisioning; no install or download in this package.'))
    entries.append(dict(path='environment-local.json',bytes=(PACKAGE/'environment-local.json').stat().st_size,sha256=k.sha(PACKAGE/'environment-local.json')))
    k.write(PACKAGE/'INVENTORY.json',dict(version='1.0.0',files=entries))
    (PACKAGE/'SHA256SUMS').write_text(''.join(e['sha256']+'  '+e['path']+'\n' for e in entries),encoding='utf-8',newline='\n')
    with tarfile.open(ARCHIVE,'w') as tar:
        for name in [e['path'] for e in entries]+['INVENTORY.json','SHA256SUMS']:
            tar.add(PACKAGE/name,arcname='h224-folds-v1/'+name,recursive=False)
    ARCHIVE.with_suffix('.tar.sha256').write_text(k.sha(ARCHIVE)+'  '+ARCHIVE.name+'\n',encoding='ascii',newline='\n')
    return dict(status='PASS',files=len(entries),bytes=ARCHIVE.stat().st_size,sha256=k.sha(ARCHIVE))

def verify(root=ROOT):
    root=Path(root);inv=k.read(root/'INVENTORY.json')
    for e in inv['files']:
        p=(root/e['path']).resolve()
        k.need(p.is_relative_to(root.resolve()),'unsafe inventory path')
        k.need(p.stat().st_size==e['bytes'] and k.sha(p)==e['sha256'],'package drift: '+e['path'])
    return dict(status='PASS',files=len(inv['files']))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['build','verify'])
    ap.add_argument('--root',type=Path,default=ROOT);a=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    print(json.dumps(build() if a.action=='build' else verify(a.root)))
if __name__=='__main__':main()
