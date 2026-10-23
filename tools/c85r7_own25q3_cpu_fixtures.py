"""CPU-only fixtures for own25q3; never import or execute model code."""
from __future__ import annotations
import ast
import contextlib
import csv
import io
import os
from pathlib import Path
from unittest.mock import patch

from tools import c85r7_own25q3_v1118 as b

MARKERS = [None, '', '0', 'False', '1', 'false', 'true', ' ', '\n']
COHORTS = [[], b.VISIBLE_IDS[:2], b.VISIBLE_IDS, ['other-a', 'other-b', 'other-c'], b.VISIBLE_IDS + ['fourth']]
FAULTS = ['open', 'enter', 'header', 'rows', 'close', 'unlink-transient', 'unlink-persistent', 'print']
BAD_MOUNTS = ['absent', 'ambiguous', 'missing-series', 'missing-manifest', 'bad-override', 'list-error', 'read-error']
BAD_CSV = ['wrong_header\nfoo\n', 'StudyInstanceUID,StudyInstanceUID\nfoo,foo\n',
           'StudyInstanceUID\n"unclosed', '', 'StudyInstanceUID\n\n', '\ufffd\n']


def run_case(area, *, ids=None, manifest_ids=None, marker=None, layout='flat',
             raw_manifest=None, malformation=None, fault=None, preexisting=False,
             real_manifest=False):
    """Execute actual cell zero and actual guard; stop after original import statement."""
    ids = b.VISIBLE_IDS if ids is None else ids
    manifest_ids = b.VISIBLE_IDS if manifest_ids is None else manifest_ids
    root = area / 'input'
    root.mkdir(parents=True, exist_ok=True)
    middle = {'flat':'', 'nested':'competitions', 'dataset':'datasets/owner', 'override':'override'}[layout]
    comp = root / middle / 'rsna-knee-abnormality-detection'
    series = comp / 'test_series'
    if malformation not in ['absent', 'bad-override']:
        comp.mkdir(parents=True, exist_ok=True)
        if malformation != 'missing-series':
            series.mkdir(exist_ok=True)
            for uid in ids:
                (series / uid).mkdir(exist_ok=True)
            (series / 'ignored-file').write_text('not a study', encoding='utf-8')
        if malformation != 'missing-manifest':
            if real_manifest:
                (comp / 'test.csv').write_bytes((b.ROOT / 'storage/competition/test.csv').read_bytes())
            elif raw_manifest is not None:
                (comp / 'test.csv').write_text(raw_manifest, encoding='utf-8')
            else:
                with (comp / 'test.csv').open('w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['StudyInstanceUID'])
                    w.writerows([[uid] for uid in manifest_ids])
    if malformation == 'ambiguous':
        other = root / 'competitions/rsna-knee-abnormality-detection'
        other.mkdir(parents=True, exist_ok=True)
        (other / 'test.csv').write_bytes((comp / 'test.csv').read_bytes())
    work = area / 'working'
    work.mkdir(exist_ok=True)
    output = work / 'submission.csv'
    if preexisting:
        output.write_bytes(b'existing submission must survive')
    ns = {}
    log = io.StringIO()
    unlink_calls = 0
    actual_open, actual_unlink, actual_iterdir, actual_writer = Path.open, Path.unlink, Path.iterdir, csv.writer

    class Stream:
        def __init__(self, stream):
            self.stream = stream
        def write(self, value):
            return self.stream.write(value)
        def __enter__(self):
            if fault == 'enter':
                self.stream.close()
                raise RuntimeError('injected enter failure')
            return self.stream
        def __exit__(self, *args):
            assert ns['FAST'] is False, 'FAST committed before close'
            self.stream.close()
            if fault == 'close':
                raise OSError('injected close failure')

    class Writer:
        def __init__(self, stream):
            self.stream = stream
            self.writer = actual_writer(stream)
        def writerow(self, row):
            assert ns['FAST'] is False, 'FAST committed before writing'
            if fault == 'header':
                self.stream.write('partial')
                raise RuntimeError('injected header failure')
            return self.writer.writerow(row)
        def writerows(self, rows):
            assert ns['FAST'] is False, 'FAST committed before writing'
            if fault in ['rows', 'unlink-transient', 'unlink-persistent']:
                self.writer.writerow(next(iter(rows)))
                raise OSError('injected rows failure')
            return self.writer.writerows(rows)

    def mapped(value, *parts):
        if str(value) == '/kaggle/input':
            return root
        if str(value) == '/kaggle/working/submission.csv':
            return output
        return Path(value, *parts)

    def opened(path, *args, **kwargs):
        if path == comp / 'test.csv' and malformation == 'read-error':
            raise PermissionError('injected manifest read failure')
        if path == output:
            assert ns['FAST'] is False, 'FAST committed before open'
            if fault == 'open':
                raise PermissionError('injected open failure')
            return Stream(actual_open(path, *args, **kwargs))
        return actual_open(path, *args, **kwargs)

    def unlinked(path, *args, **kwargs):
        nonlocal unlink_calls
        if path == output:
            unlink_calls += 1
            if fault == 'unlink-persistent' or (fault == 'unlink-transient' and unlink_calls <= 2):
                raise PermissionError('injected sharing violation')
        return actual_unlink(path, *args, **kwargs)

    def listed(path):
        if path == series and malformation == 'list-error':
            raise RuntimeError('injected directory enumeration failure')
        return actual_iterdir(path)

    def printed(*args, **kwargs):
        if fault == 'print':
            raise BrokenPipeError('injected print failure')
        print(*args, **kwargs)

    with patch.dict(os.environ, {}, clear=False), patch('pathlib.Path', mapped), \
         patch.object(Path, 'open', opened), patch.object(Path, 'unlink', unlinked), \
         patch.object(Path, 'iterdir', listed), patch('csv.writer', Writer), \
         patch('time.sleep'), contextlib.redirect_stdout(log):
        os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
        os.environ.pop('C85_COMPETITION', None)
        if marker is not None:
            os.environ['KAGGLE_IS_COMPETITION_RERUN'] = marker
        if layout == 'override' or malformation in ['missing-manifest', 'bad-override']:
            os.environ['C85_COMPETITION'] = str(comp)
        ns['print'] = printed
        # Deliberately no exception catcher: any escaped exception fails this case.
        exec(''.join(b.old.read(b.OUT / 'kernel/release.ipynb')['cells'][0]['source']), ns)
        first_reached = False
        for cell in b.old.read(b.OUT / 'kernel/release.ipynb')['cells'][1:]:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)
            tree.body[0].body = tree.body[0].body[:1]
            exec(compile(tree, '<original-first-statement-only>', 'exec'), ns)
            first_reached = 'base64' in ns
        assert first_reached == (not ns['FAST'])
        assert 'BLOB' not in ns and 'exp169_release_c85r7codec' not in ns
    expected = (not marker and sorted(ids) == b.VISIBLE_IDS and
                len(manifest_ids) == 3 and set(manifest_ids) == set(b.VISIBLE_IDS) and
                raw_manifest is None and malformation is None and fault is None and not preexisting)
    assert ns['FAST'] is bool(expected), (str(area), ns['FAST'], expected, log.getvalue())
    if ns['FAST']:
        with output.open(newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        assert rows == [b.reference.parent_builder.HEADER] + [[uid] + ['0.5'] * 12 for uid in b.VISIBLE_IDS]
        assert 'FAST_PATH_VISIBLE' in log.getvalue()
    elif preexisting:
        assert output.read_bytes() == b'existing submission must survive'
    elif fault == 'unlink-persistent':
        assert output.exists() and unlink_calls == 40
        assert 'cleanup=PermissionError' in log.getvalue()
    else:
        assert not output.exists(), 'partial output was not removed'
    if not ns['FAST'] and fault != 'print':
        assert 'FAST_PATH_DISABLED' in log.getvalue()
    return dict(status='PASS', FAST=ns['FAST'], error=None, original_first_statement=first_reached,
                output_exists=output.exists(), unlink_calls=unlink_calls,
                diagnostic=log.getvalue().strip())


def case_specs():
    cases = []
    for m, marker in enumerate(MARKERS):
        for c, ids in enumerate(COHORTS):
            cases.append((f'm{m}-c{c}', dict(ids=ids, marker=marker, real_manifest=True)))
    cases += [(name, dict(layout=name, real_manifest=True)) for name in ['nested', 'dataset', 'override']]
    cases += [(f'mount-{name}', dict(malformation=name, real_manifest=True)) for name in BAD_MOUNTS]
    cases += [(f'csv-{i}', dict(raw_manifest=raw)) for i, raw in enumerate(BAD_CSV)]
    cases += [(f'manifest-{i}', dict(manifest_ids=ids)) for i, ids in enumerate([
        [], b.VISIBLE_IDS[:2], ['a','b','c'], [b.VISIBLE_IDS[0]] * 3, b.VISIBLE_IDS + ['fourth']])]
    cases += [(f'fault-{name}', dict(fault=name, real_manifest=True)) for name in FAULTS]
    cases += [('existing-visible', dict(preexisting=True, real_manifest=True)),
              ('existing-malformed', dict(preexisting=True, raw_manifest='StudyInstanceUID\n"unclosed')),
              ('reversed', dict(ids=list(reversed(b.VISIBLE_IDS)), manifest_ids=list(reversed(b.VISIBLE_IDS))))]
    return cases


def run_smoke(area):
    import shutil
    # Fixed fixture directories only; never remove input/raw data or stage artifacts.
    assert area.resolve().is_relative_to((b.ROOT / '.local/scratch/own25q3-v1118/build').resolve())
    for name, _ in case_specs():
        fixture = (area / name).resolve()
        assert fixture.is_relative_to(area.resolve())
        if fixture.exists():
            shutil.rmtree(fixture)
    results = [dict(case=name, **run_case(area / name, **kwargs)) for name, kwargs in case_specs()]
    return dict(status='PASS', cases=results,
                limitation='Local real test.csv copied unchanged; authenticated real visible UIDs in synthetic directory fixtures (local test_series absent). No DICOM/GPU inference. Persistent OS denial of deletion is reported, bounded, nonfatal; removal cannot be guaranteed while the filesystem forbids it.')
