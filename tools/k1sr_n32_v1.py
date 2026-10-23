"""Label-free full-parent N32 preparation v1.0.0. Diagnostic only, never submit.

Changelog: largest eight by DICOM count, then diverse series/slice counts.
Only series metadata and filenames are read. No training targets or reports.
"""
import collections
import csv
import json
import os
from pathlib import Path

from tools import c85r7_rco_overlay_v1 as util

COLS = [util.UID, 'SeriesInstanceUID', 'Anatomical_Plane', 'Fat_Suppression', 'Fluid_Sensitive']


def prepare(competition, out):
    competition, out = Path(competition).resolve(), Path(out).resolve()
    grouped = collections.defaultdict(list)
    with (competition / 'train_series.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        util.need(set(COLS) <= set(reader.fieldnames), 'series metadata columns')
        for row in reader:
            row = {key: row[key] for key in COLS}
            util.need(all(row[k] and Path(row[k]).name == row[k] and row[k] not in ('.', '..')
                          for k in COLS[:2]), 'unsafe series path')
            grouped[row[util.UID]].append(row)
    ids = sorted(u for u in grouped if (competition / 'train_series' / u).is_dir())
    counts = {u: sum(1 for _ in (competition / 'train_series' / u).rglob('*.dcm')) for u in ids}
    ids = [u for u in ids if counts[u] > 0]
    util.need(len(ids) >= 32, 'N32 needs 32 mounted nonempty studies')
    largest = sorted(ids, key=lambda u: (-counts[u], u))[:8]
    selected = list(largest)
    queues = [collections.deque(sorted(ids, key=key)) for key in
              (lambda u: (-len(grouped[u]), u), lambda u: (len(grouped[u]), u),
               lambda u: (counts[u], u), lambda u: (-counts[u], u))]
    while len(selected) < 32:
        for queue in queues:
            while queue and queue[0] in selected:
                queue.popleft()
            if queue and len(selected) < 32:
                selected.append(queue.popleft())
    def csv_bytes(header, rows):
        import io
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(header); writer.writerows(rows)
        return buffer.getvalue().encode()
    outputs = {
        'test.csv': csv_bytes([util.UID], ([u] for u in selected)),
        'test_series.csv': csv_bytes(COLS, ([r[k] for k in COLS] for u in selected for r in grouped[u])),
        'sample_submission.csv': csv_bytes([util.UID]+util.LABELS, ([u]+[.5]*12 for u in selected)),
        'selection.json': util.encoded(dict(n=32, largest8=largest, ids=selected,
            counts={u: counts[u] for u in selected}, never_submit=True,
            method='largest8 then round-robin max/min series and min/max slices')),
    }
    out.mkdir(parents=True, exist_ok=True)
    link = out / 'test_series'
    if not link.exists():
        try:
            os.symlink(competition / 'train_series', link, target_is_directory=True)
        except OSError as exc:
            if os.name != 'nt' or getattr(exc, 'winerror', None) != 1314:
                raise
            for uid in selected:
                for source in (competition / 'train_series' / uid).rglob('*.dcm'):
                    dest = link / source.relative_to(competition / 'train_series')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, dest)
    for name, value in outputs.items():
        dest = out / name
        if dest.exists():
            util.need(dest.read_bytes() == value, 'N32 preparation drift')
        else:
            util.write_once(dest, value)
    util.need(all((link / u).is_dir() for u in selected), 'N32 mount incomplete')
    return dict(status='PREPARED_N32_NEVER_SUBMIT', studies=32, largest=8)
