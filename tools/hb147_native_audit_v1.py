"""H-B-147 native-detail CPU audit v1.0.0 (2026-10-13).

Changelog: local DICOM replay, no labels/reports/network; bounded real-image
samples for engineering smoke. Does not build or admit a scientific pilot.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.local/scratch/astra-b3/H-B-147'
SOURCE = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
VERSION = '1.0.0'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def required_cache_side(resolution=320):
    # Exact centered 5/7 inner crop with even pixel boundaries.
    if resolution not in (224, 320):
        raise ValueError('Only registered resolutions allowed')
    return 448


def geometry(shape, spacing, side=448):
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Native spacing must be known and positive')
    native = min(round(140. / spacing), min(shape))
    return dict(source_side=int(min(shape)), native_outer_pixels=int(native),
                native_inner_pixels=float(native * 5 / 7),
                nominal_fov_mm=float(native * spacing),
                full_140mm=abs(native * spacing - 140) <= spacing / 2 + 1e-6,
                outer_320=bool(native >= 320), inner_320=bool(native * 5 >= 320 * 7))


def audit(samples=8):
    if not 2 <= samples <= 16:
        raise ValueError('Audit requires 2..16 deterministic studies')
    from training.src import raptor_corpus as raw
    from training.src import raptor_common as common
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    assert sha(SOURCE / 'manifest.json') == 'f3afc9ab81fe068fe3ef3ac31f3f7a64d5bffb821594d8cb30e60329624b838e'
    ids = np.load(SOURCE / 'ids.npy', allow_pickle=False).tolist()
    recs = raw.series_records()  # series metadata only, never train.csv
    lookup = {uid: (split, rows) for (split, uid), rows in recs.items()}
    assert len(lookup) == len(recs) and all(uid in lookup for uid in ids)
    positions = np.linspace(0, len(ids) - 1, samples).round().astype(int).tolist()
    zip_path, index_path, _ = raw.archive_paths()
    corpus = common.Corpus(SOURCE)
    spec = copy.deepcopy(manifest['spec'])
    sample_dir = WORK / 'native-samples'
    sample_dir.mkdir(parents=True, exist_ok=True)
    geometries, timings, parity, metadata = [], [], [], []
    crop = raw._crop_resize
    def observed_crop(a, ps, crop_mm, img):
        if img == 448:
            geometries.append(geometry(a.shape, ps))
        return crop(a, ps, crop_mm, img)
    raw._crop_resize = observed_crop
    started = time.perf_counter()
    try:
        for i, position in enumerate(positions):
            uid = ids[position]
            split, rows = lookup[uid]
            spec['img'] = 336
            raw._init_worker(str(zip_path), str(index_path), spec)
            t = time.perf_counter()
            _, _, old, mask_old, picks_old = raw.build_study((split, uid, rows))
            t336 = time.perf_counter() - t
            equal = bool(np.array_equal(old, corpus.volume(uid)))
            parity.append(equal)
            raw._WORKER['zip'].close(); raw._WORKER['con'].close()
            spec['img'] = 448
            raw._init_worker(str(zip_path), str(index_path), spec)
            t = time.perf_counter()
            _, _, new, mask_new, picks_new = raw.build_study((split, uid, rows))
            t448 = time.perf_counter() - t
            raw._WORKER['zip'].close(); raw._WORKER['con'].close()
            assert equal, 'Raw replay does not reproduce frozen source pixels'
            assert picks_old == picks_new, 'Resolution changed series/slice/intensity selection'
            assert np.array_equal(mask_old, mask_new), 'Resolution changed valid-plane masks'
            np.save(sample_dir / f'volume-{i}.npy', new)
            np.save(sample_dir / f'mask-{i}.npy', mask_new)
            metadata.append(dict(index=i, source_position=position, uid=uid,
                                 volume_sha256=sha(sample_dir / f'volume-{i}.npy')))
            timings.append(dict(seconds_336=t336, seconds_448=t448))
            print(json.dumps(dict(sample=i, parity_336=equal, seconds_448=round(t448, 3))), flush=True)
    finally:
        raw._crop_resize = crop
    spacing = []
    with (SOURCE / 'picks.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            for pick in record['slots']:
                if pick and pick.get('median_spacing', 0) > 0:
                    spacing.append(pick['median_spacing'])
    space = shutil.disk_usage(ROOT)
    times = np.array([t['seconds_448'] for t in timings])
    total_bytes = len(ids) * 64 * 448 * 448
    result = dict(version=VERSION, scientific_status='ENGINEERING_ONLY_NO_MODEL_SCREEN',
        source_manifest_sha256=sha(SOURCE / 'manifest.json'), raw_archive_bytes=zip_path.stat().st_size,
        source_cache_bytes=sum(s['bytes'] for s in manifest['shards']), studies=len(ids),
        sample_count=samples, source_positions=positions, source_spec=manifest['spec'],
        source_336_replay_equal=all(parity), raw448_same_picks_and_masks=True,
        source_retained_pixels=dict(outer140=336, inner100=240),
        proposed_retained_pixels=dict(outer140=448, inner100=320),
        proposed_cache_payload_bytes=total_bytes, disk_free_bytes=space.free,
        remaining_after_payload_bytes=space.free-total_bytes,
        selected_series=len(spacing), series_median_spacing_quantiles=np.quantile(spacing,[0,.1,.25,.5,.75,.9,1]).tolist(),
        series_nominal_native320_inner=sum(p <= .3125 for p in spacing),
        series_nominal_native320_outer=sum(p <= .4375 for p in spacing),
        sampled_rendered_planes=len(geometries), sampled_native320_inner=sum(g['inner_320'] for g in geometries),
        sampled_native320_outer=sum(g['outer_320'] for g in geometries),
        sampled_full140=sum(g['full_140mm'] for g in geometries),
        timings=timings, mean_seconds_per_study_448=float(times.mean()),
        serial_full_build_hours=float(times.mean()*len(ids)/3600),
        ideal_six_worker_hours=float(times.mean()*len(ids)/3600/6),
        estimate_note='Small systematic sample, warm filesystem; no CI or parallel scaling guarantee. Excludes hashing/write overhead.',
        elapsed_seconds=time.perf_counter()-started)
    write(sample_dir / 'manifest.json', dict(spec={**manifest['spec'], 'img':448}, samples=metadata,
        source_manifest_sha256=result['source_manifest_sha256'], purpose='engineering only; no label data'))
    write(WORK / 'audit.json', result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=int, default=8)
    args = ap.parse_args()
    print(json.dumps(audit(args.samples), indent=2))


if __name__ == '__main__':
    main()
