"""H-B-155 v1.2.0: explicit cache transfer manifest, capacity and result hashes.

Changelog v1.2.0: exact side-v2/exclusion inventory and tar stream plans.
Changelog: four balanced stream lists and versioned decision/QA provenance.
No SSH, network or broad storage scan. Draft inventory is never transfer-ready.
"""
import argparse
import json
from pathlib import Path
import shutil
from tools import hb155_build_v1 as b
from tools.hb150_io_v1 import atomic_json

def disk_gate(path,additional_bytes=250_000_000_000):
    free=shutil.disk_usage(path).free
    b.need(free>=additional_bytes,'insufficient disk: need '+str(additional_bytes)+' free bytes; have '+str(free))
    return dict(status='DISK_OK',free_bytes=free,required_bytes=additional_bytes)

def safe_path(name,root=None):
    root=Path(root or b.ROOT).resolve(); unresolved=root/name; p=unresolved.resolve()
    b.need(p.is_relative_to(root) and p!=root,'unsafe manifest path')
    b.need(not any(part.is_symlink() for part in (unresolved,*unresolved.parents) if part!=root),'symlink forbidden')
    return p

def inventory(draft=False):
    files={}; missing=[]; estimated_missing=0
    def add(path,digest=None,size=None):
        p=safe_path(path)
        if not p.is_file(): missing.append(path); return
        actual=p.stat().st_size
        if size is not None: b.need(actual==size,'size mismatch: '+path)
        files[path]=dict(bytes=actual,sha256=digest or b.sha(p),hash_source='cache manifest' if digest else 'local bytes')
    for arm in ('control','crop'):
        for seed in (42,43):
            for engineering in (False,True):
                cp=b.config_path(arm,seed,engineering); c=b.validate(cp); add(b.rel(cp))
                for path,digest in c['inputs'].items(): add(path,digest)
    for directory in (b.ROOT/b.CONTROL,b.ROOT/b.CROP):
        if not (directory/'manifest.json').exists():
            missing.append(b.rel(directory/'manifest.json'))
            estimated_missing+=4410*64*448*448+45*128
            continue
        m=b.read(directory/'manifest.json')
        if not m['finished']:
            missing.append(b.rel(directory/'manifest.json')+' (unfinished)')
            estimated_missing+=4410*64*448*448+45*128
            continue
        for name in ('manifest.json','ids.npy','masks.npy'): add(b.rel(directory/name))
        for s in m['shards']: add(b.rel(directory/s['file']),s['sha256'],s['bytes'])
        # Explicit cache metadata allowlist, never recursive storage discovery.
        for pattern in ('*.json','*.npy','side-audit/*.json','overlays/*.png','overlays/*.json'):
            for path in directory.glob(pattern):
                if b.rel(path) not in files: add(b.rel(path))
        if directory==b.ROOT/b.CROP:
            for name,digest in b.read(directory/'qa-summary.json')['overlay_sha256'].items():
                add(b.rel(directory/name),digest)
    from tools.hb155_exclusions_v1 import QA, RECORD, OUT, HB145_RECORD
    for path in (QA,RECORD,OUT/'fold-binding.json',HB145_RECORD):add(b.rel(path))
    for name in ('coordinator-decision.json','cpu-prepared.json','prepared.json','crop-review.json','runtime-review.json','review.json','LAUNCH.md'):
        p=b.WORK/name
        if p.exists(): add(b.rel(p))
    for name in ('4090.json','4070.json','kaggle.json','vs4070.json','vskaggle.json'):
        p=b.WORK/'runtime'/name
        if p.exists(): add(b.rel(p))
    for arm in ('control','crop'):
        for seed in (42,43):
            p=b.WORK/'cpu'/b.member(arm,seed)/'summary.json'
            if p.exists(): add(b.rel(p))
    for name in ('exclusion-pre-review.md','exclusion-result-review.md'):
        p=b.WORK.parent/name
        if p.exists():add(b.rel(p))
    total=sum(f['bytes'] for f in files.values())
    result=dict(schema='hb155-transfer/v1',status='DRAFT_NOT_READY' if missing else 'READY',
        files=dict(sorted(files.items())),missing=sorted(set(missing)),bytes=total,estimated_missing_bytes=estimated_missing,
        projected_bytes=total+estimated_missing,minimum_empty_disk_bytes=250_000_000_000,
        note='Shard digests inherited from bound build manifests; verify every byte after tar-over-SSH. Readout runs locally.')
    atomic_json(b.WORK/'transfer-inventory.json',result)
    # Safe: even draft includes only existing explicitly enumerated files.
    (b.WORK/'transfer-files.txt').write_text(''.join(p+'\n' for p in sorted(files)),encoding='utf-8')
    (b.WORK/'SHA256SUMS').write_text(''.join(f['sha256']+'  '+p+'\n' for p,f in sorted(files.items())),encoding='utf-8')
    if not draft: b.need(not missing,'transfer not ready: missing full crop cache; use --draft only for sizing')
    return {k:v for k,v in result.items() if k!='files'}

def verify(manifest,root=None):
    m=b.read(manifest); b.need(m.get('status') in ('READY','RESULTS_COMPLETE'),'draft/partial inventory cannot verify')
    for name,f in m['files'].items():
        p=safe_path(name,root)
        b.need(p.stat().st_size==f['bytes'] and b.sha(p)==f['sha256'],'hash mismatch: '+name)
    return dict(status='VERIFIED',files=len(m['files']),bytes=sum(f['bytes'] for f in m['files'].values()))


def streams(manifest,out=None,count=4):
    """Generate disjoint byte-balanced tar lists; never initiate a transfer."""
    m=b.read(manifest); b.need(m.get('status')=='READY','draft inventory cannot transfer')
    b.need(1<=count<=4,'streams must be 1..4')
    out=Path(out or b.WORK); out.mkdir(parents=True,exist_ok=True)
    groups=[[] for _ in range(count)]; sizes=[0]*count
    for name,info in sorted(m['files'].items(),key=lambda item:(-item[1]['bytes'],item[0])):
        safe_path(name); b.need('\n' not in name and '\r' not in name,'unsafe list name')
        index=min(range(count),key=lambda i:sizes[i]); groups[index].append(name); sizes[index]+=info['bytes']
    for i,group in enumerate(groups):
        (out/f'transfer-stream-{i}.txt').write_text(''.join(name+'\n' for name in sorted(group)),encoding='utf-8',newline='\n')
    return dict(status='STREAMS_READY',streams=count,bytes=sizes,files=sum(map(len,groups)))

def results():
    # Test statuses before touching any fit output besides status JSON.
    for arm in ('control','crop'):
        for seed in (42,43):
            b.need(b.read(b.WORK/'fits'/b.member(arm,seed)/'heartbeat.json').get('status')=='complete','fit incomplete')
    files={}
    # Bring back admission evidence with predictions, so local readout can
    # authenticate the exact target-host qualification and reviews.
    evidence=[b.WORK/name for name in ('coordinator-decision.json','prepared.json','runtime-review.json','crop-review.json','review.json')]
    evidence += [b.WORK/'runtime'/name for name in ('4090.json','4070.json','kaggle.json','vs4070.json','vskaggle.json')]
    evidence += [b.WORK/'cuda'/b.member(arm,seed)/'summary.json' for arm in ('control','crop') for seed in (42,43)]
    for p in evidence:
        files[b.rel(p)]=dict(bytes=p.stat().st_size,sha256=b.sha(p))
    for arm in ('control','crop'):
        for seed in (42,43):
            d=b.WORK/'fits'/b.member(arm,seed)
            for name in ('heartbeat.json','summary.json','runtime.json','final.pt','final.receipt.json',
                         'val.csv','val.receipt.json','val_k24.csv','val_k24.receipt.json','val_k36.csv','val_k36.receipt.json'):
                p=d/name
                if name=='final.receipt.json' and not p.exists(): continue
                files[b.rel(p)]=dict(bytes=p.stat().st_size,sha256=b.sha(p))
    result=dict(schema='hb155-transfer/v1',status='RESULTS_COMPLETE',files=files)
    atomic_json(b.WORK/'result-inventory.json',result)
    (b.WORK/'result-files.txt').write_text(''.join(p+'\n' for p in sorted(files)),encoding='utf-8')
    return dict(status='RESULTS_COMPLETE',files=len(files),bytes=sum(f['bytes'] for f in files.values()))

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('inventory','disk','verify','results','streams'))
    p.add_argument('--draft',action='store_true'); p.add_argument('--manifest',type=Path); p.add_argument('--root',type=Path)
    a=p.parse_args()
    try:
        if a.command=='inventory': result=inventory(a.draft)
        elif a.command=='disk': result=disk_gate(a.root or b.ROOT)
        elif a.command=='results': result=results()
        elif a.command=='streams': result=streams(a.manifest or b.WORK/'transfer-inventory.json')
        else:
            b.need(a.manifest is not None,'manifest required'); result=verify(a.manifest,a.root)
    except (ValueError,OSError) as exc: result=dict(status='HOLD',reason=str(exc))
    print(json.dumps(result)); return 2 if result['status']=='HOLD' else 0
if __name__=='__main__': raise SystemExit(main())
