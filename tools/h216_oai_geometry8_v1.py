"""H216 v1.0.0: geometry only from completed native8; no decode/model/GPU."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import h216_oai_native8_v101 as native
OUT = ROOT / '.local/scratch/slot-b/h216-oai-geometry8-v1'
CODE = native.CODE
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import numpy as np
from training import dicom_geometry as geom
from training import geometry_bank as bank

need, sha, read, write, check = native.need, native.sha, native.read, native.write, native.check


def frame_binding(directory, members, expected):
    """Verify already-extracted bytes against the original extraction digest."""
    files = {name: directory / (hashlib.sha256(name.encode()).hexdigest() + '.dcm') for name in members}
    need(len(files) == len(members) == expected['frames'], 'DUPLICATE_OR_MISSING_FRAMES')
    need(set(directory.glob('*.dcm')) == set(files.values()), 'RAW_FILE_ROSTER')
    hashes = {name: sha(path) for name, path in files.items()}
    need(hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() ==
         expected['frame_bytes_sha256'], 'EXTRACTION_BYTES_DRIFT')
    return {str(files[name]): digest for name, digest in hashes.items()}


def raw_points(ds, plane, side):
    """Independent pixel-index replay of the native orientation, without pixels."""
    h, w = int(ds.Rows), int(ds.Columns)
    row, col = np.asarray(ds.ImageOrientationPatient, float).reshape(2, 3)
    op = geom.canonical_inplane_axes(row, col, plane)[2]
    grid = geom._apply_inplane_transform(np.arange(h*w).reshape(h, w), op)
    if geom.flip_for_plane(plane, side):
        grid = grid[:, ::-1]
    nh, nw = grid.shape
    coords = [(0, 0), (0, nw-1), (nh-1, 0), (nh-1, nw-1), (nh//2, nw//2)]
    sy, sx = map(float, ds.PixelSpacing)
    origin = np.asarray(ds.ImagePositionPatient, float)
    expected = [origin + (int(grid[r, c]) % w)*sx*row + (int(grid[r, c])//w)*sy*col
                for r, c in coords]
    return grid.shape, coords, np.asarray(expected)


def validate_record(record, series, headers):
    a = np.asarray(record['affines_col_row_origin_lps_mm'], float)
    n = len(series['files'])
    need(a.shape == (n, 3, 3) and len(headers) == n and n > 0, 'AFFINE_FRAME_COUNT')
    need(np.isfinite(a).all(), 'NONFINITE_AFFINE')
    need(np.all(np.linalg.norm(np.cross(a[:, :, 0], a[:, :, 1]), axis=1) > 1e-8), 'DEGENERATE_AFFINE')
    need(record['quality']['aligned'] and bank.alignment_quality(a)['aligned'], 'UNALIGNED_GEOMETRY')
    tags = set()
    maximum = 0.0
    for aff, ds in zip(a, headers):
        tags.update(bank.explicit_header_sides(ds))
        shape, coords, expected = raw_points(ds, series['plane'], series['side'])
        need(tuple(series['native_shape'][1:]) == shape, 'NATIVE_SHAPE_GEOMETRY')
        actual = np.asarray([aff @ [c, r, 1.] for r, c in coords])
        maximum = max(maximum, float(np.max(np.abs(actual-expected))))
        need(np.allclose(actual, expected, rtol=0, atol=1e-7), 'NATIVE_AFFINE_MISMATCH')
    need(set(record['raw_explicit_sides']) == tags, 'RAW_LATERALITY_REWRITTEN')
    need(not tags or tags == {series['side']}, 'DICOM_AUDIT_SIDE_CONFLICT')
    return maximum


def register():
    need(not OUT.exists(), 'OUTPUT_EXISTS')
    nr = read(native.OUT / 'REGISTRATION.json')
    result = read(native.OUT / 'RESULT.json')
    need(result['status'] == 'NATIVE8_PREPARED' and result['studies'] == 8, 'NATIVE_NOT_COMPLETE')
    check(nr['sources'])
    check({str(native.OUT / n): h for n, h in result['receipt_hashes'].items()})
    paths = [Path(__file__), ROOT/'docs/research/h216-oai-geometry8-v1.md',
             ROOT/'tests/test_h216_oai_geometry8_v1.py', ROOT/'tools/h216_oai_native8_v101.py',
             native.OUT/'REGISTRATION.json', native.OUT/'RESULT.json']
    paths += [CODE/'training'/n for n in ('geometry_bank.py', 'geometry_runtime.py', 'dicom_geometry.py',
                                        'common.py', 'archive_cache.py')]
    bindings = {str(p): sha(p) for p in paths}
    from tools.extcache import oai_cache_v2
    audit = oai_cache_v2.load_audit()  # Read the existing authenticated audit; no new census.
    cases = []
    for i, uid in enumerate(nr['ids'], 1):
        rp = native.OUT / f'INPUT{i:02d}-RECEIPT.private.json'
        rec = read(rp); bindings[str(rp)] = sha(rp)
        if rec.get('reused'):
            sp = Path(rec['source']['study']); rp = Path(rec['source']['receipt']); rec = read(rp)
            bindings[str(rp)] = sha(rp)
        else:
            sp = native.OUT/'native-private/studies'/(hashlib.sha256(uid.encode()).hexdigest()[:24]+'.json')
        need(sha(sp) == rec['study_json_sha256'], 'NATIVE_STUDY_DRIFT')
        bindings[str(sp)] = sha(sp)
        st = read(sp); native.validate_native(st, nr['rows'][uid])
        data = (native.PREVIOUS if i <= 5 else native.OUT) / 'data'
        expected_by_archive = {x['archive']: x for x in rec['extractions']}
        for row in nr['rows'][uid]:
            sid = row['SeriesInstanceUID']; group = audit.groups[row['group_key']]
            ar = audit.archives[sid]
            expected = expected_by_archive[str(ar['path'])]
            directory = data/'test_series'/uid/sid
            bindings.update(frame_binding(directory, group['members'], expected))
        for s in st['series']:
            need(sha(s['path']) == rec['volumes'][s['SeriesInstanceUID']]['sha256'], 'NATIVE_VOLUME_DRIFT')
            need(len(set(s['files'])) == len(s['files']), 'DUPLICATE_NATIVE_FRAMES')
            raw = data/'test_series'/uid/s['SeriesInstanceUID']
            need({str(raw/n) for n in s['files']} == {str(p) for p in raw.glob('*.dcm')}, 'NATIVE_RAW_ROSTER')
            bindings[s['path']] = rec['volumes'][s['SeriesInstanceUID']]['sha256']
        etag = 'h216-verified-directory:' + hashlib.sha256(json.dumps(rec['extractions'], sort_keys=True).encode()).hexdigest()
        cases.append(dict(token=f'INPUT{i:02d}', uid=uid, study=str(sp), root=str(data), etag=etag))
    need(len(cases) == 8, 'PILOT_CARDINALITY')
    OUT.mkdir(parents=True)
    write(OUT/'REGISTRATION.json', dict(version='1.0.0', sources=bindings, cases=cases,
          cuda=False, model_calls=0, pixel_decode_calls=0, full_v4_ready=False))


def run():
    reg = read(OUT/'REGISTRATION.json'); check(reg['sources'])
    need(not (OUT/'ATTEMPT.json').exists(), 'ATTEMPT_EXISTS')
    write(OUT/'ATTEMPT.json', dict(started_unix=time.time(), version='1.0.0'))
    import torch, pydicom
    torch.set_num_threads(1)
    need(not torch.cuda.is_initialized(), 'CUDA_INITIALIZED')
    def forbidden(*args, **kwargs):
        raise RuntimeError('MODEL_LOAD_FORBIDDEN')
    torch.load = forbidden
    started = time.perf_counter(); counts = []; outputs = {}
    for case in reg['cases']:
        st = read(case['study']); uid = case['uid']; dest = OUT/case['token']; dest.mkdir()
        mp = dest/'native-manifest.private.json'
        write(mp, dict(config=dict(archive_etag=case['etag']), studies=[st]))
        with contextlib.redirect_stdout(io.StringIO()):
            stats = bank.build(case['root'], mp, dest/'geometry', 'test', True)
        records = []; frames = 0; max_error = 0.0
        for s in st['series']:
            key = hashlib.sha256((uid+'/'+s['SeriesInstanceUID']).encode()).hexdigest()[:24]
            gp = dest/'geometry'/(key+'.json'); r = read(gp); records.append(r)
            need(r['StudyInstanceUID'] == uid and r['SeriesInstanceUID'] == s['SeriesInstanceUID'], 'GEOMETRY_IDENTITY')
            hs = [pydicom.dcmread(Path(case['root'])/'test_series'/uid/s['SeriesInstanceUID']/n,
                  stop_before_pixels=True, specific_tags=['ImageOrientationPatient', 'ImagePositionPatient',
                  'PixelSpacing', 'Rows', 'Columns', 'Laterality', 'ImageLaterality']) for n in s['files']]
            max_error = max(max_error, validate_record(r, s, hs)); frames += len(hs)
            outputs[str(gp.relative_to(OUT))] = sha(gp)
        side, evidence = bank.resolve_anatomy_laterality([dict(raw_explicit_sides=r['raw_explicit_sides']) for r in records])
        for r in records:
            need(r['explicit_side'] == side and r['laterality_status'] == evidence['status'] and
                 r['study_explicit_sides'] == evidence['explicit_sides'], 'EXPLICIT_LATERALITY_PROMOTION')
        counts.append(dict(token=case['token'], series=len(records), frames=frames,
                      explicit_side_status=evidence['status'], max_mapping_error_mm=max_error,
                      geometry_seconds=stats['seconds']))
        for p in (mp, dest/'geometry/manifest.json'):
            outputs[str(p.relative_to(OUT))] = sha(p)
        print(json.dumps(dict(completed=len(counts), total=8, series=len(records), frames=frames)), flush=True)
    check(reg['sources'])
    need(not torch.cuda.is_initialized(), 'CUDA_INITIALIZED')
    need(sum(c['series'] for c in counts) == 31 and sum(c['frames'] for c in counts) == 1705, 'FINAL_CARDINALITY')
    result = dict(status='GEOMETRY8_PREPARED', version='1.0.0', studies=8, series=31, frames=1705,
                  aligned_series=31, counts=counts, seconds=time.perf_counter()-started,
                  max_mapping_error_mm=max(c['max_mapping_error_mm'] for c in counts),
                  unknown_explicit_side_studies=sum(c['explicit_side_status']=='unknown_no_explicit_tags' for c in counts),
                  native_arrays_reused=True, cuda_initialized=False, model_calls=0, pixel_decode_calls=0,
                  full_v4_ready=False, training_ready=False, outputs=outputs)
    write(OUT/'RESULT.json', result)
    return {k:v for k,v in result.items() if k not in ('counts', 'outputs')}


if __name__ == '__main__':
    try:
        if sys.argv[1:] == ['register']:
            register(); print('REGISTERED_GEOMETRY8_CPU_ONLY')
        elif sys.argv[1:] == ['run']:
            print(json.dumps(run()))
        else:
            raise ValueError('ACTION')
    except Exception as exc:
        code = str(exc) if type(exc) is ValueError and str(exc).replace('_', '').isalnum() else 'PRIVATE_DETAILS_SUPPRESSED'
        fail = dict(status='FAILED', error_type=type(exc).__name__, code=code)
        if OUT.exists() and (OUT/'ATTEMPT.json').exists() and not (OUT/'FAILURE.json').exists():
            write(OUT/'FAILURE.json', fail)
        print(json.dumps(fail)); sys.exit(1)
