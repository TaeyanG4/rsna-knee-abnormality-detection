"""CPU tests v1.0.0: the H-B-165 fit/watchdog forks differ from H-B-155 ONLY by the registered namespace edits."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIT_EDITS = [
    ('# Changelog v1.2.0: exclusion-bound configs', '# H-B-165 fork v1.0.0 (2026-10-15): namespace-only copy of tools/hb155_fit_v1.py v1.2.0 (build module,\n'
     '# experiment/status strings, VERSION); training, IO, progress and evaluation code byte-identical.\n'
     '# Changelog v1.2.0: exclusion-bound configs'),
    ('"""H-B-155 compartment-crop paired pilot fits v1.2.0 (2026-10-15).',
     '"""H-B-165 fold-3 OA crop-expert confirmation fits v1.0.0 (2026-10-15; fork of H-B-155 fits v1.2.0).'),
    ('VERSION = "1.2.0"', 'VERSION = "1.0.0"'),
    ('from tools.hb155_build_v1 import validate', 'from tools.hb165_build_v1 import validate'),
    ('from tools.hb155_build_v1 import require_fit', 'from tools.hb165_build_v1 import require_fit'),
    ('from tools.hb155_build_v1 import require_runtime', 'from tools.hb165_build_v1 import require_runtime'),
    ('experiment="H-B-155 compartment crop paired pilot"', 'experiment="H-B-165 OA crop expert fold3 confirmation"'),
    ('status="EXPOSED_PAIRED_FOLD2"', 'status="EXPOSED_PAIRED_FOLD3"')]
WATCHDOG_EDITS = [
    ('"""H-B-155 compartment-crop paired pilot independent supervisor v1.1.0 (2026-10-15).',
     '"""H-B-165 fold-3 OA crop-expert supervisor v1.0.0 (2026-10-15; fork of H-B-155 supervisor v1.1.0).\n\n'
     'Changelog H-B-165 v1.0.0: namespace only (stage WORK, fit module name, messages); policy unchanged.'),
    ('VERSION = "1.1.0"', 'VERSION = "1.0.0"'),
    ('WORK = ROOT / ".local/scratch/slot-b/hb155/pilot-v120"', 'WORK = ROOT / ".local/scratch/slot-b/hb165/pilot-fold3-v100"'),
    ('help="Only the bounded H-B-155 compartment-crop paired pilot CUDA full-shape smoke"',
     'help="Only the bounded H-B-165 fold-3 CUDA full-shape smoke"'),
    ('ap.error("attempt and budget must stay inside the H-B-155 stage")', 'ap.error("attempt and budget must stay inside the H-B-165 stage")'),
    ('command[1:4] != ["-B", "-m", "tools.hb155_fit_v1"]', 'command[1:4] != ["-B", "-m", "tools.hb165_fit_v1"]'),
    ('ap.error("measurement is restricted to python -B -m tools.hb155_fit_v1 smoke --device cuda")',
     'ap.error("measurement is restricted to python -B -m tools.hb165_fit_v1 smoke --device cuda")')]


@pytest.mark.parametrize('src,dst,edits', [('tools/hb155_fit_v1.py', 'tools/hb165_fit_v1.py', FIT_EDITS),
                                           ('tools/hb155_watchdog_v1.py', 'tools/hb165_watchdog_v1.py', WATCHDOG_EDITS)])
def test_fork_is_parent_plus_exact_registered_edits(src, dst, edits):
    text = (ROOT/src).read_bytes().decode('utf-8')
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    assert (ROOT/dst).read_bytes().decode('utf-8') == text


def test_fit_fork_uses_only_the_hb165_build_gates():
    text = (ROOT/'tools/hb165_fit_v1.py').read_text(encoding='utf-8')
    assert text.count('from tools.hb165_build_v1 import') == 3 and 'hb155_build_v1' not in text


def test_watchdog_measurement_restricted_to_hb165_fit(tmp_path):
    from tools import hb165_watchdog_v1 as w
    attempt, budget = w.WORK/'attempts'/'never-created', w.WORK/'never-budget.json'
    base = ['--attempt', str(attempt), '--budget', str(budget), '--measurement', '--']
    for command in (['python', '-B', '-m', 'tools.hb155_fit_v1', 'smoke', '--device', 'cuda'],
                    ['python', '-B', '-m', 'tools.hb165_fit_v1', 'fit', '--device', 'cuda'],
                    ['python', '-B', '-m', 'tools.hb165_fit_v1', 'smoke', '--device', 'cpu']):
        with pytest.raises(SystemExit):
            w.main(base + command)
    with pytest.raises(SystemExit):
        w.main(['--attempt', str(tmp_path/'outside'), '--budget', str(budget), '--measurement', '--', 'python'])
    assert not attempt.exists() and not budget.exists()


def test_watchdog_gpu_fit_requires_measured_rates():
    from tools import hb165_watchdog_v1 as w
    with pytest.raises(SystemExit):
        w.main(['--attempt', str(w.WORK/'attempts'/'never-created'), '--budget', str(w.WORK/'never-budget.json'),
                '--', 'python', '-B', '-m', 'tools.hb165_fit_v1', 'fit'])
