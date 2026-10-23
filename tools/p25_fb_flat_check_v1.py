"""F4 CPU saved-prediction check v1.0.0.

Changelog 1.0.0: authenticated real 58-study flat P arithmetic and base/N32
parent boundary replay. No labels, model inference, GPU or qualification.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from tools import p25_fb_flat_stage_v1 as b
from tools import k1n_flat_p_check_v1 as mathcheck
from tools import k1n_cpu_replay_v1 as cpu


def inputs():
    from tools import c85r7_r6w958inf3rco_v1150 as source
    assert b.util.sha(source.INVENTORY)==source.INVENTORY_SHA
    inventory=json.loads(source.INVENTORY.read_bytes());pins={}
    def pinned(rel):
        path=b.ROOT/rel;digest=b.util.sha(path)
        assert digest==inventory['files'][rel]['sha256']
        pins[rel]=digest;return path
    with np.load(pinned('.local/scratch/hb167-common/hb167/gpu/predictions.npz'),allow_pickle=False) as z:
        ids=z['ids'].tolist();r=z['R'].astype(float);r224=z['R224'].astype(float)
    frame=pd.read_csv(pinned('.local/scratch/r6-gold58-build/r6-gold58/gpu/flip/output/predictions.csv'),
                      dtype={b.util.UID:str})
    assert not frame[b.util.UID].duplicated().any() and set(frame[b.util.UID])==set(ids)
    s=frame.set_index(b.util.UID).loc[ids,b.util.LABELS].to_numpy(float)
    return ids,[r,r224,s],pins


def flat_parent():
    ids,arrays,pins=inputs();parts=b.package()
    _,after,result=mathcheck.compare(arrays,[np.zeros(len(ids),bool) for _ in range(3)],parts[9],parts[10])
    return ids,after,pins,result


def run():
    ids,arrays,pins=inputs();parts=b.package()
    _,after,result=mathcheck.compare(arrays,[np.zeros(len(ids),bool) for _ in range(3)],parts[9],parts[10])
    paths=[]
    for diagnostic in (False,True):
        count=32 if diagnostic else len(ids)
        dest=b.SCRATCH/'saved-p-replay'/('n32' if diagnostic else 'base')
        # Reuse deterministic scratch outputs only after checking their pinned
        # inputs and exact payload. Never remove an earlier run's files.
        summary=dest/'CHECK.json'
        files=b.package(diagnostic)[10]
        if summary.exists():
            item=json.loads(summary.read_bytes())
            assert item['inputs']==pins and item['sources']=={n:b.digest(v) for n,v in files.items()}
            payload=(dest/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).read_bytes()
            assert item['output_sha256']==b.digest(payload)
        else:
            replay,payload=cpu.replay(files,dest,ids[:count],[x[:count] for x in arrays],diagnostic=diagnostic)
            assert replay['p_branch_exact'] and not replay['o_executed']
            item=dict(diagnostic=diagnostic,studies=count,inputs=pins,
                sources={n:b.digest(v) for n,v in files.items()},output_sha256=b.digest(payload),
                actual_parent_preflight_collect_math=True,external_process_asset_codec_stubs=True)
            summary.write_bytes(b.util.encoded(item))
        # N32 rank recalculates on its own roster.
        expected=mathcheck.oracle([x[:count] for x in arrays],
            [np.zeros(count,bool) for _ in range(3)],b.flat.FLAT).astype(np.float32)
        np.testing.assert_array_equal(b.util.table(payload)[1],expected)
        paths.append(item)
    result.update(status='PASS_CPU_SAVED_PREDICTIONS',studies=len(ids),inputs=pins,paths=paths,
        exposure='EXPOSED_VISIBLE_58; cached predictions; no labels or quality claim',
        qualified_for_submission=False)
    (b.SCRATCH/'SAVED-P-RESULT.json').write_bytes(b.util.encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k!='paths'},indent=2))
    return result


if __name__=='__main__':run()
