"""H212 v1.0.0: aggregate-only, label-free metadata census of the K1 train cap.

One preregistered reference draw; no efficacy, GPU, pixels, checkpoint or trainer
import. Does not reconstruct historical shuffled pool order. No training launch.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / '.local/scratch/l1-rco/k1v3-grouping'
CACHE = Path('G:/temp_data/rsna-knee/l1rco-cache-full')


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def run():
    source = (PARENT / 'compact_train_l1.py').read_bytes()
    require(hashlib.sha256(source).hexdigest() == '0d5fbae473eb1dc451871be0d610689d6baff8e6ac258acb091e66343ad0b791', 'source drift')
    names = {'covering_centers', 'phys_step', 'window_rows'}
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in functions} == names, 'function identity')
    ns = {'math': math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), '<parent-window-rows>', 'exec'), ns)
    reg = json.loads((PARENT / 'registration.json').read_text(encoding='utf-8'))
    ids = sorted(reg['train'])
    require(len(ids) == len(set(ids)) == 3190 and not set(ids) & set(reg['holdout']), 'training cohort')
    raw_index = (CACHE / 'index.json').read_bytes()
    index = json.loads(raw_index)
    meta_path = CACHE / 'meta.f32.npy'
    before = meta_path.stat()
    meta = np.load(meta_path, mmap_mode='r', allow_pickle=False)
    h = hashlib.sha256()
    spacing_cache = {}
    def meta_of(file, row):
        require(file == 0, 'unexpected add-on cache')
        if row not in spacing_cache:
            value = np.array(meta[row], copy=True)
            require(np.isfinite(value[5]), 'nonfinite spacing')
            spacing_cache[row] = value
            h.update(str(row).encode() + value.tobytes())
        return spacing_cache[row]
    capped = lost_group = lost_plane = oversized = groups_total = groups_lost = 0
    candidate_counts = []
    for i, uid in enumerate(ids):
        blocks = [dict(b, views=[(0, v, st) for v, st in b['views']]) for b in index['blocks'][uid]]
        rng = np.random.default_rng(9000+i)
        candidates = ns['window_rows'](blocks, meta_of, 'train', 3, 3, 3, 3.5, rng, .25, 'cover')
        require(candidates, 'empty existing candidate bag')
        intervals = [(st, st+b['n_slices'], (bi, v), b['plane']) for bi,b in enumerate(blocks) for _,v,st in b['views']]
        def identity(row):
            matches = [(g,p) for lo,hi,g,p in intervals if lo <= row[1] < hi]
            require(len(matches) == 1, 'ambiguous group interval')
            return matches[0]
        groups = [identity(row) for row in candidates]
        all_groups = {g for g,p in groups}
        all_planes = {p for g,p in groups}
        candidate_counts.append(len(candidates))
        oversized += len(all_groups) > 64
        if len(candidates) <= 64:
            continue
        capped += 1
        keep = np.sort(rng.choice(len(candidates), 64, replace=False))
        kept_groups = {groups[j][0] for j in keep}
        kept_planes = {groups[j][1] for j in keep}
        missing = all_groups-kept_groups
        lost_group += bool(missing)
        lost_plane += bool(all_planes-kept_planes)
        groups_total += len(all_groups)
        groups_lost += len(missing)
    after = meta_path.stat()
    require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), 'metadata changed during audit')
    fraction = lost_group/capped if capped else 0.0
    return dict(version='1.0.0', status='COMPLETE_CPU_METADATA_ONLY',
                train_studies=len(ids), reference='sorted training IDs, seed9000+i, epoch0; not historical draw order',
                capped_studies=capped, studies_losing_series_view_group=lost_group,
                losing_group_fraction_of_capped=fraction, studies_losing_plane=lost_plane,
                capped_groups_total=groups_total, capped_groups_lost=groups_lost,
                studies_with_more_than64groups=oversized,
                candidate_count_quantiles=dict(zip(['min','p50','p95','max'], np.quantile(candidate_counts,[0,.5,.95,1]).tolist())),
                metadata_rows_read=len(spacing_cache), accessed_metadata_sha256=h.hexdigest(),
                index_sha256=hashlib.sha256(raw_index).hexdigest(),
                train_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                decision='PREPARE_STRATIFIED_CAP_DESIGN' if capped and fraction >= .1 and not oversized else 'PARK',
                efficacy_measured=False, launch_allowed=False, review='B_DIRECT_NOT_INDEPENDENT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'preserve original evidence')
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result))
