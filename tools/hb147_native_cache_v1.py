"""H-B-147 native448 corpus builder v1.0.1, 2026-10-13.

Changelog: reuse frozen roster, raw decoding and picked slices from RAPTOR64;
change only rasterization width. No labels, extraction or network. Output is
confined to H-B-147 scratch. Finished shards are checked and never overwritten.
v1.0.1: preserve frozen masks; on raster-dependent differences require exact
336 pixel/mask/pick replay. Record mixed builder provenance on safe resume.
"""
from __future__ import annotations
import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from tools.hb147_native_audit_v1 import ROOT, WORK, SOURCE, sha, write
from training.src import raptor_corpus as raw

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

OUT = WORK / 'native448'


def checked_study(payload):
    task, frozen_mask = payload
    result = raw.build_study(task)
    uid, split, vol, mask, picks = result
    changed = np.flatnonzero(mask != frozen_mask)
    if len(changed):
        from training.src.raptor_common import Corpus
        raw._WORKER['spec']['img'] = 336
        try:
            _, _, replay, replay_mask, replay_picks = raw.build_study(task)
        finally:
            raw._WORKER['spec']['img'] = 448
        assert np.array_equal(replay, Corpus(SOURCE).volume(uid)), '336 replay pixels differ'
        assert np.array_equal(replay_mask, frozen_mask), '336 replay masks differ'
        assert replay_picks == picks, '336 replay picks differ'
    return uid, split, vol, mask, picks, changed.tolist()


def build(limit=0, workers=4):
    if workers not in range(1, 5) or limit < 0:
        raise ValueError('1..4 workers, nonnegative limit required')
    source = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    source_hash = sha(SOURCE / 'manifest.json')
    assert source_hash == 'f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e'
    assert sha(SOURCE / 'ids.npy') == source['ids_sha256']
    assert sha(SOURCE / 'masks.npy') == source['masks_sha256']
    ids = np.load(SOURCE / 'ids.npy', allow_pickle=False)
    old_masks = np.load(SOURCE / 'masks.npy')
    if limit:
        ids, old_masks = ids[:limit], old_masks[:limit]
    out = WORK / 'cache-smoke' if limit else OUT
    out.mkdir(parents=True, exist_ok=True)
    expected = {}
    with (SOURCE / 'picks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            expected[rec['uid']] = rec
    recs = raw.series_records()
    spec = copy.deepcopy(source['spec']); spec['img'] = 448
    path = out / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    if manifest:
        assert manifest['source_manifest_sha256'] == source_hash and manifest['studies'] == len(ids)
        assert manifest['raw_builder_sha256'] == sha(raw.__file__) and manifest['spec'] == spec
        if manifest['builder_sha256'] != sha(__file__):
            previous = ROOT / '.local/backups/H-B-147/hb147_native_cache_v1.0.0.py'
            assert manifest['builder_sha256'] == sha(previous), 'unknown resume builder'
            assert manifest['masks_policy'] == 'identical to source; reject mismatch'
            manifest['previous_builder_sha256'] = manifest['builder_sha256']
            manifest['previous_builder_committed_shards'] = len(manifest['shards'])
            manifest['builder_sha256'] = sha(__file__)
    else:
        manifest = dict(contract='hb147_raptor64_448_c140_s02_v1', spec=spec,
            studies=len(ids), dtype='uint8', shape_total=[len(ids),64,448,448],
            shard_size=100, shards=[], source_manifest_sha256=source_hash,
            builder_sha256=sha(__file__), raw_builder_sha256=sha(raw.__file__),
            archive=source['archive'], finished=False, roster_policy='frozen source ids; no label access',
            masks_policy='frozen source masks; exact 336 replay required on raster-mask differences')
    manifest['masks_policy'] = 'frozen source masks; exact 336 replay required on raster-mask differences'
    manifest.setdefault('raster_mask_differences', [])
    np.save(out / 'ids.npy', ids)
    np.save(out / 'masks.npy', old_masks)
    manifest.update(ids_sha256=sha(out / 'ids.npy'), masks_sha256=sha(out / 'masks.npy'))
    remaining = len(ids)*64*448*448-sum(s['bytes'] for s in manifest['shards'])
    assert shutil.disk_usage(ROOT).free >= remaining + 10*(1<<30), 'insufficient disk headroom'
    zip_path,index,_ = raw.archive_paths()
    start = time.perf_counter()
    with mp.Pool(workers, initializer=raw._init_worker, initargs=(str(zip_path),str(index),spec)) as pool:
        for k, begin in enumerate(range(0,len(ids),100)):
            stop = min(begin+100,len(ids)); file = f'vols_shard{k:02d}.npy'
            if k < len(manifest['shards']):
                s = manifest['shards'][k]
                assert s['file'] == file and sha(out/file) == s['sha256'], 'finished shard drift'
                continue
            assert not (out/file).exists(), 'uncommitted shard requires manual review'
            tasks = [(expected[u]['split'],str(u),recs[(expected[u]['split'],u)]) for u in ids[begin:stop]]
            arr = np.lib.format.open_memmap(out/(file+'.partial'),mode='w+',dtype=np.uint8,shape=(stop-begin,64,448,448))
            then=time.perf_counter()
            payloads = zip(tasks, old_masks[begin:stop])
            for row,(uid,split,vol,mask,picks,changed) in enumerate(pool.imap(checked_study,payloads,chunksize=1)):
                assert uid == ids[begin+row] and picks == expected[uid]['slots'], 'source picks differ'
                if changed:
                    manifest['raster_mask_differences'].append(dict(position=begin+row, planes=changed,
                        source_336_replay_exact=True, frozen_valid=old_masks[begin+row][changed].tolist(),
                        raster448_valid=mask[changed].tolist()))
                    print(json.dumps(dict(mask_difference_position=begin+row,planes=changed,
                                          replay336_exact=True)),flush=True)
                arr[row]=vol
            arr.flush(); del arr
            (out/(file+'.partial')).replace(out/file)
            manifest['shards'].append(dict(shard=k,start=begin,stop=stop,file=file,done=True,
                bytes=(out/file).stat().st_size,sha256=sha(out/file),seconds=time.perf_counter()-then))
            write(path,manifest)
            print(json.dumps(dict(shard=k,studies=stop,seconds=round(time.perf_counter()-then,2))),flush=True)
    manifest.update(finished=True,elapsed_this_run_seconds=time.perf_counter()-start,workers=workers)
    write(path,manifest)
    return dict(status='CACHE_COMPLETE',studies=len(ids),shards=len(manifest['shards']),
                bytes=sum(s['bytes'] for s in manifest['shards']),seconds=manifest['elapsed_this_run_seconds'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args(); print(json.dumps(build(a.limit,a.workers)))


if __name__=='__main__':
    mp.freeze_support();main()
