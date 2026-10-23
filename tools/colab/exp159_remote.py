"""EXP159-only Kaggle dataset access; leaves hash-bound run_remote.py unchanged."""
from __future__ import annotations

import csv
import io
import re
import subprocess
import time
from pathlib import Path


def _column(value: str) -> str:
    return re.sub(r'[^a-z0-9]', '', value.lstrip('\ufeff').lower())


def parse_dataset_files(output: str):
    """Ignore CLI notices before CSV and accept the known file/byte column spellings."""
    lines = output.splitlines()
    names = {'name', 'filename', 'file'}
    sizes = {'size', 'sizebytes', 'totalbytes', 'bytes'}
    for offset, line in enumerate(lines):
        header = next(csv.reader([line]), [])
        columns = [_column(field.strip()) for field in header]
        if not columns or columns[0] not in names or len(columns) < 2 or columns[1] not in sizes:
            continue
        reader = csv.DictReader(io.StringIO('\n'.join(lines[offset:])))
        name_key, size_key = reader.fieldnames[:2]
        return [(row[name_key], int(row[size_key])) for row in reader if row.get(name_key)]
    raise ValueError('Kaggle dataset files output has no name,size CSV header')


def dataset_files(ref: str):
    result = subprocess.run(['kaggle', 'datasets', 'files', ref, '--csv', '--page-size', '200'],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'cannot list {ref}: {(result.stderr or result.stdout)[-300:]}')
    return parse_dataset_files(result.stdout)


def ensure_dataset(ref: str, dest: Path):
    """Download one file at a time, reusing complete files after a Colab reconnect."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = dataset_files(ref)
    if not files:
        raise RuntimeError(f'{ref}: dataset listing is empty')
    total = sum(size for _, size in files)
    start, downloaded = time.time(), 0
    for name, size in files:
        target = dest / name
        if target.exists() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['kaggle', 'datasets', 'download', '-d', ref, '-f', name,
                        '-p', str(target.parent), '--unzip', '-q', '-o'], check=True)
        if not target.exists() or target.stat().st_size != size:
            raise RuntimeError(f'{name}: expected {size} bytes, got '
                               f'{target.stat().st_size if target.exists() else "nothing"}')
        downloaded += size
    print(f'{ref}: {len(files)} files, {total / 1e9:.1f} GB in {dest} '
          f'(downloaded {downloaded / 1e9:.1f} GB in {time.time() - start:.0f}s)', flush=True)
    return dest
