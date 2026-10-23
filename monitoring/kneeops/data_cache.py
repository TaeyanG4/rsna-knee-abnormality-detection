"""Read only cache inspection; headers and small metadata, never pickle or volumes."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import struct
from pathlib import Path

from .store import dumps, now


def cache_origin(manifest_sha256):
    """Curated provenance is tied to an exact manifest, never inferred from a folder name."""
    registry = json.loads(Path(__file__).with_name('cache_origins.json').read_text(encoding='utf-8'))
    return registry['by_manifest_sha256'].get(manifest_sha256)


def within(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Cache manifest must use relative filenames')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('Cache filename escapes its directory')
    return path


def npy_header(path):
    with path.open('rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('Not an NPY file')
        major, minor = f.read(2)
        if major not in (1, 2, 3):
            raise ValueError('Unsupported NPY version')
        width = 2 if major == 1 else 4
        size = struct.unpack('<H' if width == 2 else '<I', f.read(width))[0]
        if size > 65536:
            raise ValueError('NPY header too large')
        header = ast.literal_eval(f.read(size).decode('utf-8' if major == 3 else 'latin1'))
        shape = header['shape']
        if header['descr'] not in ('|u1', '<u1', '|b1') or header['fortran_order']:
            raise ValueError('Only C-order uint8/bool caches are supported')
        if not isinstance(shape, tuple) or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('Invalid cache shape')
        if path.stat().st_size != f.tell() + math.prod(shape):
            raise ValueError('Truncated or unexpected NPY size')
        return list(shape)


def inspect_cache(path):
    result = {'manifest': str(path), 'name': path.parent.name, 'valid': False, 'errors': [], 'hash_verified': False}
    try:
        if path.stat().st_size > 2_000_000:
            raise ValueError('Manifest too large')
        raw = path.read_bytes()
        m = json.loads(raw)
        result.update(contract=m['contract'], studies=m['studies'], shape=m['shape_total'],
                      manifest_sha256=hashlib.sha256(raw).hexdigest(), sampling=m.get('sampling', ''),
                      normalization=m.get('normalization', ''), missing_slots=m.get('missing_slots', 0),
                      short_slots=m.get('short_slots', 0), bytes=0)
        shape = m['shape_total']
        # Slot caches are (study, slot, slice, H, W); Raptor training corpora are flat (study, slice, H, W).
        flat = str(m['contract']).startswith('raptor')
        if m['dtype'] != 'uint8' or len(shape) != (4 if flat else 5) or shape[0] != m['studies']:
            raise ValueError('Unsupported cache contract')
        cursor = 0
        for shard in m['shards']:
            file = within(path.parent, shard['file'])
            if shard['start'] != cursor or shard['stop'] <= cursor:
                raise ValueError('Non-contiguous cache shards')
            if npy_header(file) != [shard['stop'] - cursor, *shape[1:]] or file.stat().st_size != shard['bytes']:
                raise ValueError('Shard shape/size differs from manifest')
            result['bytes'] += file.stat().st_size
            cursor = shard['stop']
        if cursor != shape[0]:
            raise ValueError('Incomplete study coverage')
        mask = within(path.parent, m['mask_file'])
        mask_shape = shape[:2] if (flat or m['contract'].startswith('orthodiffusion_')) else shape[:3]
        if npy_header(mask) != mask_shape:
            raise ValueError('Mask shape differs from cache')
        # Masks are small; verify their hash and values without NumPy.
        if mask.stat().st_size > 20_000_000:
            raise ValueError('Mask exceeds metadata limit')
        if m.get('mask_sha256') and hashlib.sha256(mask.read_bytes()).hexdigest() != m['mask_sha256']:
            raise ValueError('Mask checksum mismatch')
        uid_file = within(path.parent, m['uid_file'])
        if uid_file.stat().st_size > 20_000_000:
            raise ValueError('UID metadata exceeds limit')
        with uid_file.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        ids = [r['StudyInstanceUID'] for r in rows]
        if len(ids) != shape[0] or len(set(ids)) != len(ids) or any(not u for u in ids):
            raise ValueError('Missing or duplicate study IDs')
        if 'index' in rows[0] and any(int(r['index']) != i for i, r in enumerate(rows)):
            raise ValueError('UID order differs from cache index')
        result.update(valid=True, uid_sha256=hashlib.sha256(uid_file.read_bytes()).hexdigest(),
                      bytes=result['bytes'] + mask.stat().st_size + uid_file.stat().st_size,
                      fingerprint=hashlib.sha256(dumps([s.get('sha256') for s in m['shards']]).encode()).hexdigest(),
                      provenance=cache_origin(result['manifest_sha256']))
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, struct.error) as exc:
        result['errors'].append(str(exc)[:250])
    return result


def cache_catalog(store):
    roots = [store.ws.resolve('cache')]
    custom = store.get_meta('data_cache_manifest')
    paths = set()
    for root in roots:
        # Supported existing contracts only; no recursive volume scans.
        for pattern in ('*/*manifest.json', '*manifest.json'):
            paths.update(root.glob(pattern))
    if custom:
        paths.add(Path(custom))
    result, seen = [], set()
    for p in sorted(paths):
        item = inspect_cache(p)
        if 'contract' not in item:
            if custom == str(p):
                result.append(item)
            continue
        key = (str(p.parent), item.get('fingerprint', item['manifest']))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    # Collapse only copies whose actual payload files now share the same inode.
    # Equal manifest hashes alone are insufficient evidence for deduplication.
    grouped, identities = [], {}
    for item in sorted(result, key=lambda c: c['manifest'] != custom):
        identity = None
        if item['valid']:
            path = Path(item['manifest'])
            m = json.loads(path.read_text(encoding='utf-8'))
            files = [m['uid_file'], m['mask_file'], *[s['file'] for s in m['shards']]]
            identity = (item['manifest_sha256'], tuple((os.stat(within(path.parent, f)).st_dev,
                         os.stat(within(path.parent, f)).st_ino) for f in files))
        if identity is not None and identity in identities:
            identities[identity].setdefault('aliases', []).append(item['manifest'])
        else:
            grouped.append(item)
            if identity is not None:
                identities[identity] = item
    result = grouped
    for item in result:
        item['same_manifest_content_elsewhere'] = any(other is not item and item.get('fingerprint') and item.get('fingerprint') == other.get('fingerprint') for other in result)
    return sorted(result, key=lambda item: (not item.get('contract', '').startswith('adjacent_'), item['manifest']))


def cache_descriptor(store, value):
    path = store.ws.resolve(value)
    check = inspect_cache(path)
    if not check['valid']:
        raise ValueError('; '.join(check['errors']))
    try:
        uri = store.ws.uri(path)
    except ValueError:
        uri = str(path)
    return {'schema': 1, 'created_at': now(), 'competition': store.ws.config['competition'],
            'cache_manifest': uri, 'manifest_sha256': check['manifest_sha256'],
            'uid_sha256': check['uid_sha256'], 'contract': check['contract'], 'shape': check['shape'],
            'provenance': check.get('provenance'),
            'loader': 'numpy.load(shard, mmap_mode="r", allow_pickle=False)',
            'validation': 'headers, file sizes, UID coverage and mask checksum; volume hashes NOT verified',
            'training_note': 'Preserve grouped folds and label/mask contracts. Gold58 is exposed; no inference Report.'}
