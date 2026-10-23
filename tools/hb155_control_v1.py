"""H-B-155 control-cache reuse v1.0.0: read-only verification, no cache mutation.

The existing native cache is shared with other experiments. Never rewrite its
manifest just to reuse it. Missing/incomplete caches require B's native builder.
"""
import argparse
import json
import time
import numpy as np
from tools import hb155_build_v1 as b
from tools.hb150_io_v1 import atomic_json


def verify(limit_shards=0):
    b.need(limit_shards>=0,'nonnegative shard limit')
    directory=b.ROOT/b.CONTROL; m=b.read(directory/'manifest.json')
    from tools.hb155_crop_v1 import SOURCE
    b.need(m['source_manifest_sha256']==b.sha(SOURCE/'manifest.json'),'control source differs')
    ids=np.load(directory/'ids.npy',allow_pickle=False).astype(str).tolist()
    b.need(len(ids)==4410,'control roster differs')
    from tools.hb147_native_cache_v1 import raw
    b.need(m['raw_builder_sha256']==b.sha(raw.__file__),'raw builder differs')
    b.cache_binding(dict(corpus=b.CONTROL,corpus_contract='hb147_raptor64_448_c140_s02_v1',train=ids,val=[]))
    b.need(b.sha(directory/'ids.npy')==b.sha(SOURCE/'ids.npy') and
           b.sha(directory/'masks.npy')==b.sha(SOURCE/'masks.npy'),'control frozen roster/masks differ')
    shards=m['shards'][:limit_shards] if limit_shards else m['shards']; then=time.perf_counter()
    for s in shards: b.need(b.sha(directory/s['file'])==s['sha256'],'control shard hash mismatch')
    result=dict(status='VERIFIED' if len(shards)==len(m['shards']) else 'SAMPLE_ONLY',
        verified_shards=len(shards),total_shards=len(m['shards']),
        verified_bytes=sum(s['bytes'] for s in shards),total_bytes=sum(s['bytes'] for s in m['shards']),
        manifest_sha256=b.sha(directory/'manifest.json'),seconds=time.perf_counter()-then,
        reuse=True,mutated_control=False)
    atomic_json(b.WORK/('control-verification.json' if not limit_shards else 'control-verification-smoke.json'),result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--limit-shards',type=int,default=0)
    print(json.dumps(verify(p.parse_args().limit_shards)))
