"""Future CPU HC15 ZIP-to-series driver; import is stdlib-only.

No command line, allocation permission or automatic full-cache run is provided.
Use only from a separately reviewed CPU-process wrapper with a hard timeout.
The index must be explicitly hashed in the allocation contract; source ZIP CRCs
are verified by ZipFile. Enhanced/multiframe inputs fail rather than silently
dropping frames. The original sample decoder and frozen adapter are unchanged.
"""
from __future__ import annotations

import copy
from contextlib import closing
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import zipfile

from training.adapters.hc15_cache_store import canonical, digest_file, require


class IndexedZip(zipfile.ZipFile):
    def _RealGetContents(self):
        self.start_dir = os.fstat(self.fp.fileno()).st_size


def zip_info(record):
    value = zipfile.ZipInfo(record['path'])
    allowed = {'compress_type', 'comment', 'extra', 'create_system', 'create_version',
               'extract_version', 'reserved', 'flag_bits', 'volume', 'internal_attr',
               'external_attr', 'header_offset', 'CRC', 'compress_size', 'file_size',
               'date_time', '_raw_time', '_end_offset', 'orig_filename', 'filename'}
    for key, item in record['zip_info'].items():
        require(key in allowed, 'Unknown frozen ZIP metadata field')
        setattr(value, key, tuple(item) if key == 'date_time' else item)
    require(value.filename == record['path'], 'ZIP member identity differs')
    return value


def decode_records(row, members, read_member, decode_frame, preprocess, deadline):
    """One series, checked between frames; caller must enforce hard timeout."""
    import hashlib
    require(bool(members), 'Source series contains no DICOM members')
    require(len({m['path'] for m in members}) == len(members), 'Duplicate source members')
    prefix = f"train_series/{row['StudyInstanceUID']}/{row['SeriesInstanceUID']}/"
    require(all(m['path'].startswith(prefix) and m['path'].endswith('.dcm') for m in members),
            'Source member belongs to a different series')
    decoded = []
    for member in members:
        require(time.monotonic() < deadline, 'Decode deadline exhausted')
        # Exceptions are failures. Never convert them to missing/ineligible data.
        decoded.append(decode_frame(read_member(member), member['path']))
    require(time.monotonic() < deadline, 'Decode deadline exhausted')
    n = len(decoded)
    payload = None if n < 3 else preprocess(decoded, row['Anatomical_Plane'])
    require(time.monotonic() < deadline, 'Decode deadline exhausted')
    return payload, n, hashlib.sha256(canonical(members)).hexdigest()


def check_single_frame(ds):
    require(int(ds.get('NumberOfFrames', 1)) == 1
            and 'SharedFunctionalGroupsSequence' not in ds
            and 'PerFrameFunctionalGroupsSequence' not in ds,
            'Enhanced/multiframe DICOM needs separately reviewed per-frame geometry')


def cpu_backend():
    """Load the existing reviewed geometry adapter only inside a CPU worker."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'CPU worker must hide CUDA before imports')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'CPU worker must use one numerical thread')
    import torch
    require(torch.version.cuda is None and not torch.cuda.is_initialized(), 'Pinned CPU-only torch required')
    import cv2
    import numpy as np
    import pydicom
    from training.adapters.goodpjw_reader import DecodedSlice, preprocess_decoded_series
    cv2.setNumThreads(1); torch.set_num_threads(1)

    def decode(data, path):
        ds = pydicom.dcmread(io.BytesIO(data))
        check_single_frame(ds)
        pixels = ds.pixel_array
        require(pixels.ndim == 2, 'Unsupported single-frame pixel layout')
        return DecodedSlice(
            pixels=pixels,
            pixel_spacing=list(map(float, ds.PixelSpacing)) if 'PixelSpacing' in ds else None,
            image_orientation=list(map(float, ds.ImageOrientationPatient)) if 'ImageOrientationPatient' in ds else None,
            image_position=list(map(float, ds.ImagePositionPatient)) if 'ImagePositionPatient' in ds else None,
            instance_number=float(ds.get('InstanceNumber', 0) or 0), source_id=path+'#0',
            photometric=str(ds.get('PhotometricInterpretation', 'MONOCHROME2')),
            rescale_slope=float(ds.RescaleSlope) if 'RescaleSlope' in ds else None,
            rescale_intercept=float(ds.RescaleIntercept) if 'RescaleIntercept' in ds else None)

    def preprocess(slices, plane):
        volume, _ = preprocess_decoded_series(slices, plane_name=plane)
        require(volume.dtype == np.uint8 and len(volume) == min(len(slices),64)
                and volume.shape[1:] == (384,384), 'Decoded cache geometry differs')
        target = io.BytesIO(); np.save(target, volume, allow_pickle=False)
        return target.getvalue()
    return decode, preprocess


class ArchiveDecoder:
    """Read-only index and direct ZIP-member access, without full ZIP extraction."""
    def __init__(self, archive_contract, index_sha256, decode_frame, preprocess):
        self.archive_contract = copy.deepcopy(archive_contract)
        self.index_sha256 = index_sha256
        self.index = Path(self.archive_contract['index']).resolve()
        self.archive = Path(self.archive_contract['path']).resolve()
        require(digest_file(self.index) == index_sha256, 'Archive index drift')
        require(self.archive.stat().st_size == archive_contract['bytes'], 'Archive size drift')
        self.stat_identity = self._stats()
        self.decode_frame, self.preprocess = decode_frame, preprocess

    def _stats(self):
        return [(p.stat().st_size, p.stat().st_mtime_ns) for p in (self.index,self.archive)]

    def __call__(self, row, deadline):
        require(time.monotonic() < deadline, 'Decode deadline exhausted')
        require(self._stats() == self.stat_identity, 'Archive/index changed during production')
        with closing(sqlite3.connect(self.index.as_uri()+'?mode=ro', uri=True)) as db:
            db.execute('PRAGMA query_only=ON')
            records = db.execute("SELECT path,info FROM files WHERE split='train' AND uid=? AND series=? ORDER BY path",
                                 (row['StudyInstanceUID'],row['SeriesInstanceUID'])).fetchall()
        members = [{'path':p,'zip_info':json.loads(info)} for p,info in records if p.endswith('.dcm')]
        with IndexedZip(self.archive) as source:
            value = decode_records(row,members,lambda m:source.read(zip_info(m)),
                                   self.decode_frame,self.preprocess,deadline)
        require(self._stats() == self.stat_identity, 'Archive/index changed during decode')
        return value


def produce_chunk(store, decoder, *, decoder_sha256, max_series, deadline):
    """Commit at most max_series new records; no retries or suppressed failures.

    The future reviewed outer wrapper creates the matching CacheStore/source
    contract and calls this in a CPU-only process with a hard aggregate deadline.
    Returning an incomplete chunk is not a complete-cache or training receipt.
    """
    require(type(max_series) is int and max_series > 0, 'Invalid chunk size')
    require(type(deadline) in (int,float) and math.isfinite(deadline)
            and time.monotonic() < deadline, 'Chunk deadline exhausted/invalid')
    require(decoder_sha256 == store.contract['decoder_sha256'], 'Decoder identity differs')
    require(getattr(decoder,'index_sha256',None) == store.contract['archive_index_sha256'], 'Source index binding differs')
    require(getattr(decoder,'archive_contract',None) == store.contract['source_archive'], 'Source archive binding differs')
    completed = 0
    for row in store.contract['series']:
        sid = row['SeriesInstanceUID']
        if store.contains(sid):
            continue
        if completed == max_series or time.monotonic() >= deadline:
            break
        store._gate()
        payload, count, members_hash = decoder(copy.deepcopy(row), deadline)
        store.commit(sid,payload,decoded_count=count,source_members_sha256=members_hash,
                     decoder_sha256=decoder_sha256)
        completed += 1
    complete = set(store.records) == set(store.rows)
    cache = store.finalize(deadline=deadline) if complete else None
    return {'completed_this_chunk':completed, 'complete':complete,
            'cache_manifest':str(cache) if cache else None, 'gpu_dispatch_allowed':False}
