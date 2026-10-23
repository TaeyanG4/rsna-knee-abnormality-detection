"""H-B-190 N32 diagnostic kernel: structure, local harness plumbing, run_reader on N32-format data, projection.

CPU only. Embedded code runs in an isolated interpreter with the candidate runtime on sys.path (no tools.*).
"""
import csv
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import c85r7_r6w958rdq_n32_v1 as n32
from tools import c85r7_r6w958rdq_v1140 as b

RUNTIME = b.SCRATCH / 'candidate-runtime'
PRELUDE = r'''
import os, sys, json, types
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
runtime = Path(sys.argv[1]); ARGS = sys.argv[2:]
sys.path.insert(0, str(runtime))
import exp169_release_c85r7codec as release
assert Path(release.__file__).resolve().parent == runtime.resolve()
assert all(not n.startswith('tools.') for n in sys.modules)
'''


def isolated(body, *args, env_extra=None, timeout=1800):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    env.pop('PYTHONPATH', None)
    env.pop('C85_COMPETITION', None)
    env.update(env_extra or {})
    run = subprocess.run([sys.executable, '-B', '-I', '-c', PRELUDE + body, str(RUNTIME), *map(str, args)],
                         capture_output=True, text=True, env=env, timeout=timeout)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-6000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


def stress_cells():
    return [''.join(c['source']) for c in json.loads((n32.OUT / 'release.ipynb').read_text(encoding='utf-8'))['cells']]


def test_structure_one_bootstrap_line_identical_archive_and_metadata():
    result = n32.validate()
    assert result['embedded_members'] == 31 and result['kernel_id'] == n32.SLUG != b.SLUG
    nb, meta = n32.candidate()
    cand = ''.join(nb['cells'][2]['source'])
    cells = stress_cells()
    changed = [x for x in difflib.unified_diff(cand.splitlines(), cells[2].splitlines(), lineterm='', n=0)
               if x[:1] in '+-' and not x.startswith(('+++', '---'))]
    assert changed == ['-' + n32.ANCHOR.rstrip('\n'), '+' + n32.REPLACEMENT.rstrip('\n')]
    assert cells[0] == n32.CELL0.read_text(encoding='utf-8') and 'FAST = False' in cells[0]
    smeta = json.loads((n32.OUT / 'kernel-metadata.json').read_text())
    assert smeta['title'] == 'knee-lb-r6w958rdq-n32-v1' and '/' not in smeta['title']
    assert smeta['dataset_sources'] == meta['dataset_sources'] and b.READER_SLUG in smeta['dataset_sources']
    assert {k for k in smeta if smeta[k] != meta[k]} == {'id', 'title'}
    for forbidden in ('train.csv', 'kaggle competitions submit', 'subprocess', 'requests', 'urllib'):
        assert forbidden not in cells[0], forbidden


HARNESS = r'''
cell0 = Path(ARGS[0]).read_text(encoding='utf-8')
g = {'__name__': '__hb190_n32__'}
exec(compile(cell0, '<cell0>', 'exec'), g)
class Proxy:
    def __getattr__(self, name): return getattr(release, name)
    def run_release(self, runtime, output): raise RuntimeError('local CPU test: no Kaggle mounts for part A')
report = g['_hb190_n32_run'](Proxy(), runtime)
work = Path(os.environ['HB190_N32_WORKING'])
data = work / 'hb190-n32-data'
import csv
with (data / 'test_series.csv').open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f); header = reader.fieldnames; series = list(reader)
with (data / 'test.csv').open(newline='', encoding='utf-8') as f:
    tests = [r['StudyInstanceUID'] for r in csv.DictReader(f)]
out = work / 'hb190-n32-reader'
print(json.dumps(dict(report=report, header=header, n_series=len(series), tests=tests,
    series_studies=sorted({r['StudyInstanceUID'] for r in series}),
    base_equals_final=(out / 'submission.csv').read_bytes() == (out / 'c85r7-run/reader/hb189-submission.csv').read_bytes(),
    marker=(work / 'HB190_N32_DONE').read_text().strip(), report_file=(work / 'hb190-n32-report.json').is_file()), default=str))
'''


@pytest.fixture(scope='module')
def harness(tmp_path_factory):
    assert RUNTIME.is_dir()
    work = tmp_path_factory.mktemp('hb190-n32-working')
    cell0 = work.parent / 'cell0.txt'
    cell0.write_text(stress_cells()[0], encoding='utf-8')
    out = isolated(HARNESS, cell0, env_extra=dict(HB190_N32_WORKING=str(work), HB190_N32_COMPETITION=str(b.GOLD_DICOM),
                                                    HB190_N32_N='2'))
    return work, out


def test_local_harness_plumbing_selects_rewrites_and_fails_closed_without_cuda(harness):
    work, out = harness
    r = out['report']
    assert out['report_file'] and out['marker'] == 'FAIL' == r['verdict']
    assert 'local CPU test' in r['part_a']['error'] and r['checks']['a_c85_complete'] is False
    pb = r['part_b']
    assert pb['studies'] == 2 and len(out['tests']) == 2 and out['series_studies'] == sorted(out['tests'])
    assert out['header'] == ['StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']
    assert r['checks']['b_full_series_columns'] is True and out['n_series'] >= 6
    # The reader child refuses to run without CUDA: phase exit 1, whole fallback, H-B-189-style base kept.
    assert pb['phase']['exit_code'] == 1 and 'exit code' in pb['phase']['error']
    assert pb['reader_receipt']['status'] == 'FALLBACK_ALL' and out['base_equals_final'] is True
    assert pb['rows'] == 2 and pb['finite'] is True and 'projection' not in pb


RUN_READER = r'''
import torch
data, output, store = map(Path, ARGS)
output.mkdir(parents=True, exist_ok=True)
import csv
with (data / 'test.csv').open(newline='', encoding='utf-8') as f:
    ids = [r['StudyInstanceUID'] for r in csv.DictReader(f)]
(output / 'ids.json').write_text(json.dumps(ids))
torch.cuda.is_available = lambda: True          # test-only: lets run_reader pass its CUDA gate on CPU
original_predict, original_mount = release.reader_predict, release.v5q._mount
release.reader_predict = lambda *a, **k: original_predict(*a, device='cpu', workers=0, **k)
release.v5q._mount = lambda slug, req: store if slug == release.READER_SLUG else original_mount(slug, req)
torch.set_num_threads(6)
summary = release.run_reader(runtime, output)
rows = [json.loads(x) for x in (output / 'journal.jsonl').read_text().splitlines() if '"uid"' in x]
print(json.dumps(dict(summary=summary, ok=[r['ok'] for r in rows], uids=[r['uid'] for r in rows], ids=ids,
                      worker=(output / 'worker.json').is_file())))
'''


def test_run_reader_end_to_end_on_n32_format_data(harness, tmp_path):
    work, _ = harness
    out = isolated(RUN_READER, work / 'hb190-n32-data', tmp_path / 'reader', b.READER_STORE / 'extracted',
                   env_extra=dict(C85_COMPETITION=str(work / 'hb190-n32-data')))
    assert out['worker'] and out['uids'] == out['ids'] and out['ok'] == [True, True]
    s = out['summary']
    assert s['ok'] == 2 and s['failed'] == 0 and s['k'] == 16 and s['counterpart'] is True and s['checkpoints'] == 3


def cell0_namespace():
    g = {'__name__': '__hb190_n32_unit__'}
    saved = os.environ.get('C85_COMPETITION')
    exec(compile(n32.CELL0.read_text(encoding='utf-8'), '<cell0>', 'exec'), g)
    if saved is not None:
        os.environ['C85_COMPETITION'] = saved
    return g


def test_projection_arithmetic_gate_charges_model_load_per_study():
    g = cell0_namespace()
    p = g['_n32_project'](200.0, dict(seconds_total=150.0, seconds_models=10.0), 32)
    assert p['reader_seconds_per_study'] == pytest.approx(150 / 32) and p['reader_startup_seconds'] == pytest.approx(50.0)
    assert p['reader_hidden_hours']['gate'] == pytest.approx((50 + 150 / 32 * 1300) / 3600)
    assert p['reader_hidden_hours']['central'] == pytest.approx((50 + 10 + 140 / 32 * 1300) / 3600)
    assert p['reader_hidden_hours']['conservative'] == pytest.approx(200 / 32 * 1300 / 3600)
    assert p['total_hours']['gate'] == pytest.approx(7.12 + (50 + 150 / 32 * 1300) / 3600)
    assert p['reader_hidden_hours']['central'] <= p['reader_hidden_hours']['gate'] <= p['reader_hidden_hours']['conservative']


def synthetic_pass():
    return dict(part_a=dict(c85_status='COMPLETE_C85R7', rows=3, finite=True, phases=dict(READER=dict(exit_code=0)),
                            reader_receipt=dict(status='APPLIED', fallback_studies=0, reader_ok=3)),
                part_b=dict(studies=32, rows=32, finite=True, journal_ok=32, journal_failed=0, phase=dict(exit_code=0),
                            series_columns=['StudyInstanceUID', 'Fat_Suppression'],
                            reader_receipt=dict(status='APPLIED', fallback_studies=0, reader_ok=32),
                            projection=dict(total_hours=dict(central=8.1, gate=8.2, conservative=8.9))))


@pytest.mark.parametrize('breaker', [None, 'a_status', 'a_fallback', 'b_fallback', 'b_journal', 'b_columns', 'slow',
                                     'b_exit', 'a_rows'])
def test_checks_truth_table(breaker):
    g = cell0_namespace()
    r = synthetic_pass()
    if breaker == 'a_status': r['part_a']['c85_status'] = 'FAILED_EXECUTION'
    if breaker == 'a_fallback': r['part_a']['reader_receipt']['fallback_studies'] = 1
    if breaker == 'b_fallback': r['part_b']['reader_receipt'].update(fallback_studies=1, reader_ok=31)
    if breaker == 'b_journal': r['part_b']['journal_failed'] = 1
    if breaker == 'b_columns': r['part_b']['series_columns'] = ['StudyInstanceUID']
    if breaker == 'slow': r['part_b']['projection']['total_hours']['gate'] = 9.01
    if breaker == 'b_exit': r['part_b']['phase']['exit_code'] = 1
    if breaker == 'a_rows': r['part_a']['finite'] = False
    checks, verdict = g['_n32_checks'](r)
    assert verdict == ('PASS' if breaker is None else 'FAIL')
    assert sum(not v for v in checks.values()) == (0 if breaker is None else 1)
