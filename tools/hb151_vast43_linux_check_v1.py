"""Read-only SSH Linux CPU qualification v1.0.0; modules sent in memory, no upload."""
from pathlib import Path
import base64
import hashlib
import json
import subprocess
import zlib

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.local/scratch/astra-b5/H-B-151-seed43'
PACKAGE=WORK/'package'

REMOTE=r'''
import os,sys,json,base64,zlib,importlib.abc,importlib.util,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
sources=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
from pathlib import Path
original_is_file=Path.is_file
def virtual_is_file(path):
    if str(path).startswith('/virtual/hb151/'):
        return str(path)[len('/virtual/hb151/'):] in sources
    return original_is_file(path)
Path.is_file=virtual_is_file
class MemoryModules(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def find_spec(self,fullname,path=None,target=None):
        rel=fullname.replace('.','/')
        candidates=[rel+'.py',rel+'/__init__.py']
        if '.' not in fullname: candidates += ['training/src/'+rel+'.py']
        for key in candidates:
            if key in sources:
                spec=importlib.util.spec_from_loader(fullname,self,is_package=key.endswith('/__init__.py'))
                spec.loader_state=key
                return spec
        if rel in ('tools','tools/ml_parity','training'):
            spec=importlib.util.spec_from_loader(fullname,self,is_package=True)
            spec.loader_state=None
            return spec
    def create_module(self,spec): return None
    def exec_module(self,module):
        key=module.__spec__.loader_state
        module.__file__='/virtual/hb151/'+(key or module.__name__.replace('.','/')+'/__init__.py')
        if key: exec(compile(sources[key],module.__file__,'exec'),module.__dict__)
sys.meta_path.insert(0,MemoryModules())
for key,source in sources.items():
    if key.endswith('.py'): compile(source,'/virtual/hb151/'+key,'exec')
import torch, numpy as np
torch.set_num_threads(1)
from tools import hb151_vast43_v1 as guard
import hb151_vast43_v1 as trainer
from tools.exp113_radr50_local import install_gold_blocker
from tools.exp112_full_swa import recompute_bn
from tools import hb150_watchdog_v1 as watchdog
result=trainer.tiny_validation()
real_model=trainer.build_model()
result['real_model_parameters']=sum(p.numel() for p in real_model.parameters())
del real_model
from torch.utils.data import DataLoader,Dataset,DistributedSampler
class Fixture(Dataset):
    def __len__(self): return 32
    def __getitem__(self,index):
        # Import again inside real fork workers: exercises inherited import locks.
        import hb151_vast43_v1
        return torch.zeros((3,3,8,8)),torch.zeros(12),'synthetic'
ds=Fixture()
sampler=DistributedSampler(ds,1,0,shuffle=True,seed=43,drop_last=True)
loader=DataLoader(ds,batch_size=1,sampler=sampler,drop_last=True,num_workers=12,
    pin_memory=True,collate_fn=trainer.collate,persistent_workers=True,worker_init_fn=trainer.worker_init)
assert torch.multiprocessing.get_start_method()=='fork'
for epoch in range(2):
    sampler.set_epoch(epoch)
    count=0
    for batch in loader:
        x,y,uids=trainer.unpack_batch(batch)
        assert x.shape==(1,3,3,8,8) and y.shape==(1,12)
        count+=1
    assert count==32
loader._iterator._shutdown_workers()
guard.admission_snapshot=lambda:''
guard.read_json=lambda path:dict(mode='fit',pid=-1,created=0)
try: guard.require_gpu_job('fit',child=True)
except ValueError as exc: assert 'ancestry' in str(exc)
else: raise AssertionError('direct child bypass')
assert watchdog.Policy(0,idle=900).check(900)=='no_heartbeat_15m'
assert not torch.cuda.is_initialized()
result.update(status='PASS_LINUX_CPU',python=sys.version.split()[0],torch=str(torch.__version__),
    runtime_effective=guard.runtime_snapshot(),
    source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
    compiled_modules=sum(k.endswith('.py') for k in sources),fork_workers=12,epochs=2,batches=64,
    direct_child_bypass_rejected=True,remote_files_written=False,cuda_initialized=False)
print(json.dumps(result))
'''

def main():
    sources={p.relative_to(PACKAGE).as_posix():p.read_text(encoding='utf-8') for p in PACKAGE.rglob('*.py')}
    sources['config/workspace.toml']=(PACKAGE/'config/workspace.toml').read_text(encoding='utf-8')
    payload=base64.b64encode(zlib.compress(json.dumps(sources).encode())).decode()
    script='PAYLOAD='+repr(payload)+'\n'+REMOTE
    result=subprocess.run(['ssh','-p','19960','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'root@<remote-host>','python -B -'],input=script,text=True,capture_output=True,timeout=90)
    (WORK/'linux-cpu.stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:
        print(result.stderr);raise SystemExit(result.returncode)
    receipt=json.loads(result.stdout)
    (WORK/'linux-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (WORK/'vast-effective.json').write_text(json.dumps(receipt['runtime_effective'],indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'}))

if __name__=='__main__': main()
