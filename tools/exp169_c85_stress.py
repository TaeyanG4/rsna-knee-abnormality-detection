"""Label-free C85 N32 selector adapted from .local/v6-hidden-debug/stress-N32.ipynb.
Only train_series.csv and DICOM filenames are inspected; no labels or reports.
"""
import collections
import csv
import json
import os
from pathlib import Path
import time
try:
    from tools import exp169_release_c85 as c
except ImportError:
    import exp169_release_c85 as c
UID = c.p.UID
LABELS = c.p.LABELS
COLS = [UID, 'SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive']

def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)


def table(path, header, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def prepare(competition, out, n=32):
    start = time.monotonic()
    competition, out = Path(competition).resolve(), Path(out).resolve()
    assert n > 0 and not out.exists(), 'positive N and NEW output required'
    by_uid = collections.defaultdict(list)
    with (competition / 'train_series.csv').open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assert set(COLS) <= set(reader.fieldnames)
        for row in reader:
            # Do not read train.csv, labels, or reports.
            r = {k: row[k] for k in COLS}
            for key in COLS[:2]:
                assert r[key] and Path(r[key]).name == r[key] and r[key] not in ('.', '..')
            by_uid[r[UID]].append(r)
    ids = sorted(u for u in by_uid if (competition / 'train_series' / u).is_dir())
    assert len(ids) >= n, f'only {len(ids)} mounted studies; requested {n}'
    signatures = {u: tuple(sorted({(r[COLS[2]], r[COLS[3]]) for r in by_uid[u]})) for u in ids}
    counts = {u: sum(1 for _ in (competition / 'train_series' / u).rglob('*.dcm')) for u in ids}
    queues = [sorted(ids, key=lambda u: (-len(by_uid[u]), u)),
              sorted(ids, key=lambda u: (len(by_uid[u]), u)),
              sorted(ids, key=lambda u: (-counts[u], u)),
              sorted(ids, key=lambda u: (counts[u], u))]
    queues = [collections.deque(q) for q in queues]
    selected, seen = [], set()
    while len(selected) < n:
        for q in queues:
            while q and q[0] in seen:
                q.popleft()
            if q and len(selected) < n:
                u = q.popleft()
                selected.append(u)
                seen.add(u)
    out.mkdir(parents=True)
    # Link the WHOLE root. Per-UID symlinks fail DirectoryDicomSource containment.
    try:
        os.symlink(competition / 'train_series', out / 'test_series', target_is_directory=True)
    except OSError as e:
        if os.name != 'nt' or getattr(e, 'winerror', None) != 1314:
            raise
        # Windows-only CPU self-test: new hardlinks, no source mutation/copy.
        for u in selected:
            for source in (competition / 'train_series' / u).rglob('*.dcm'):
                dest = out / 'test_series' / source.relative_to(competition / 'train_series')
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, dest)
    table(out / 'test_series.csv', COLS, ([r[k] for k in COLS] for u in selected for r in by_uid[u]))
    table(out / 'test.csv', [UID], ([u] for u in selected))
    table(out / 'sample_submission.csv', [UID, *LABELS], ([u, *([0.5] * 12)] for u in selected))
    dump(out / 'selection.json', {'n': n, 'method': 'round-robin max/min series/max/min slices', 'seconds': time.monotonic()-start,
         'selection': [{'uid': u, 'series': len(by_uid[u]), 'dicoms': counts.get(u),
                        'plane_contrast_signature': signatures[u]} for u in selected]})
    return selected


