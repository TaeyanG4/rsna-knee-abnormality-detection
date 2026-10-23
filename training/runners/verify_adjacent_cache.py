try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(32<<20),b''): h.update(b)
    return h.hexdigest()

ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); a=ap.parse_args(); r=workspace_path(a.cache_dir)
m=json.loads((r/'adjacent_cache_manifest.json').read_text())
assert int(m['studies'])==4407, m['studies']
for s in m['shards']:
    p=r/s['file']; assert p.is_file(), p; assert p.stat().st_size==int(s['bytes']), (p,p.stat().st_size,s['bytes'])
    got=sha256(p); assert got==s['sha256'], (p,got,s['sha256']); print('OK',p.name,got,flush=True)
mp=r/m['mask_file']; assert sha256(mp)==m['mask_sha256']; mask=np.load(mp,mmap_mode='r'); assert mask.shape[0]==4407
uids=pd.read_csv(r/m['uid_file']); assert len(uids)==4407 and uids.StudyInstanceUID.is_unique
print(json.dumps({'contract':m['contract'],'studies':m['studies'],'shape_total':m['shape_total'],'failure_count':m['failure_count'],'geometry_slots':m['geometry_slots'],'fallback_slots':m['fallback_slots'],'missing_slots':m['missing_slots'],'short_slots':m['short_slots'],'decode_failures':m['decode_failures'],'all_hashes_ok':True},indent=2),flush=True)
