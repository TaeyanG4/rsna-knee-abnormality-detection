"""Explicit local maintenance, never run by scanning or by starting the app."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from .data_archive import bounded, signature, uri
from .store import dumps, now


def file_sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def cache_plan(store, canonical, duplicate, progress=print):
    base = store.ws.resolve('cache').resolve()
    first, second = Path(canonical).resolve(), Path(duplicate).resolve()
    if first == second or not all(p.is_relative_to(base) and p != base for p in (first, second)):
        raise ValueError('서로 다른 캐시 하위 폴더만 정리할 수 있습니다.')
    source = {}
    for file in first.iterdir():
        if file.is_file() and not file.is_symlink():
            sig = signature(file)
            sha = file_sha(file)
            if signature(file) != sig:
                raise ValueError('검사 도중 캐시 변경')
            source.setdefault((sig[0], sha), file)
            progress(f'원본 SHA256 확인: {file.name}')
    pairs = []
    for file in second.iterdir():
        if file.is_file() and not file.is_symlink():
            sig = signature(file); sha = file_sha(file)
            if signature(file) != sig:
                raise ValueError('검사 도중 캐시 변경')
            match = source.get((sig[0], sha))
            if match and not os.path.samefile(file, match):
                pairs.append({'keep': uri(store, match), 'duplicate': uri(store, file),
                              'sha256': sha, 'bytes': sig[0], 'signature': sig,
                              'keep_signature': signature(match)})
            progress(f'복사본 SHA256 확인: {file.name}')
    plan = {'created_at': now(), 'kind': 'replace-identical-cache-copies-with-hardlinks', 'pairs': pairs,
            'reclaimable_bytes': sum(p['bytes'] for p in pairs)}
    directory = store.state / 'cleanup'; directory.mkdir(exist_ok=True)
    path = directory / ('cache-' + uuid.uuid4().hex + '.json')
    path.write_text(dumps(plan), encoding='utf-8')
    return path, plan


def apply_cache_plan(store, plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    base = store.ws.resolve('cache').resolve()
    for pair in plan['pairs']:
        keep, duplicate = (store.ws.resolve(pair[key]) for key in ('keep', 'duplicate'))
        for file in (keep, duplicate):
            if not file.resolve().is_relative_to(base):
                raise ValueError('캐시 범위 밖의 파일')
            bounded(base, file.relative_to(base))
        if os.path.samefile(keep, duplicate):
            continue
        if signature(keep) != pair['keep_signature'] or signature(duplicate) != pair['signature']:
            raise ValueError('검사 이후 파일 변경: 정리를 중단했습니다.')
        link = duplicate.with_name(duplicate.name + '.dedup-' + uuid.uuid4().hex)
        try:
            os.link(keep, link)
            # Atomic replacement removes only the redundant bytes, preserving old paths.
            os.replace(link, duplicate)
        finally:
            if link.exists():
                link.unlink()
        pair['applied_at'] = now()
        plan_path.write_text(dumps(plan), encoding='utf-8')
    return plan


def remove_verified_loose(store, archive, study, loose_root):
    """Remove only bytes verified identical to a CRC-checked extracted study."""
    from .data_archive import materialize
    prepared, ready = materialize(store, study, archive)
    root = Path(loose_root).resolve()
    expected_root = store.ws.resolve(archive['path']).parent.resolve()
    if root != expected_root:
        raise ValueError('해당 ZIP 다운로드 폴더만 정리할 수 있습니다.')
    source = bounded(root, f"{study['split']}_series/{study['uid']}")
    removed = []
    for file in sorted(source.glob('*/*.dcm')):
        bounded(root, file.relative_to(root))
        target = bounded(prepared, file.relative_to(source))
        if file.is_file() and target.is_file() and file.stat().st_size == target.stat().st_size:
            sha = file_sha(file)
            if sha == file_sha(target):
                removed.append({'path': uri(store, file), 'canonical': uri(store, target),
                                'sha256': sha, 'bytes': file.stat().st_size})
    directory = store.state / 'cleanup'; directory.mkdir(exist_ok=True)
    ledger = directory / ('loose-' + uuid.uuid4().hex + '.json')
    result = {'created_at': now(), 'files': removed}
    ledger.write_text(dumps(result), encoding='utf-8')
    for record in removed:
        file = store.ws.resolve(record['path'])
        bounded(root, file.relative_to(root))
        if file_sha(file) != record['sha256']:
            raise ValueError('검사 도중 파일 변경')
        file.unlink()
        record['deleted_at'] = now()
        ledger.write_text(dumps(result), encoding='utf-8')
    # Empty directories only; never recursively delete an original-data tree.
    for directory in sorted(source.glob('*')):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    if source.is_dir() and not any(source.iterdir()):
        source.rmdir()
    return result
