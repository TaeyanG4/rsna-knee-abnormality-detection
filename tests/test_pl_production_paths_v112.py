"""v1.1.2 regression: registered roots and serial local-split completion."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import pl_production_launch_v1 as launch
from tools import pl_production_teacher_v1 as teacher
from tools import pl_production_watchdog_v1 as watchdog


class ReachedInputs(Exception):
    pass


def reached(*args, **kwargs):
    raise ReachedInputs


ROOTS = [launch.WORK.parent / 'PL-PRODUCTION-v1', launch.WORK]


@pytest.mark.parametrize('root', ROOTS)
def test_teacher_accepts_registered_launcher_roots_without_reading_source(root, tmp_path):
    stub = SimpleNamespace(need=teacher.pl.need, read=reached)
    with pytest.raises(ReachedInputs):
        teacher.seal(tmp_path/'absent-source', root/'teacher/seal.json',
                     exposed_exploratory=True, pl_module=stub)


@pytest.mark.parametrize('root', ROOTS)
def test_watchdog_accepts_registered_launcher_arguments(root, monkeypatch):
    monkeypatch.setattr(watchdog, 'validate_memory', reached)
    with pytest.raises(ReachedInputs):
        watchdog.main(['--attempt', str(root/'watchdog/f0-split-v111'),
                       '--budget', str(root/'watchdog/budget.json'),
                       '--memory-evidence', 'absent.json', '--go', 'absent.json',
                       '--arm', 'global', '--cpu', '--', sys.executable, '-c', 'pass'])


@pytest.mark.parametrize('outside', [launch.WORK.parent/'PL-PRODUCTION-v11-evil',
                                     launch.WORK/'..'/'outside'])
def test_unregistered_siblings_and_traversal_refused(outside, tmp_path, monkeypatch):
    stub = SimpleNamespace(need=teacher.pl.need, read=reached)
    with pytest.raises(ValueError, match='production scratch'):
        teacher.seal(tmp_path/'absent', outside/'teacher.json',
                     exposed_exploratory=True, pl_module=stub)
    monkeypatch.setattr(watchdog, 'validate_memory', reached)
    with pytest.raises(ValueError, match='production scratch'):
        watchdog.main(['--attempt', str(outside/'attempt'), '--budget', str(outside/'budget.json'),
                       '--memory-evidence', 'absent', '--go', 'absent', '--arm', 'global',
                       '--cpu', '--', sys.executable, '-c', 'pass'])


@pytest.mark.parametrize('child_code', [0, 9])
def test_split_waits_and_propagates_child_failure_before_next_fold(tmp_path, child_code):
    # Execute the actual launcher body with only external Python/GPU commands mocked.
    # No nvidia-smi, GPU process, protected inputs or production output is accessed.
    script = Path('.local/run-scripts/pl-local-split-v110.ps1').read_text()
    script = script.replace("$Py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'",
                            "$Py = 'Invoke-FakePython'")
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(arm='global', members=[
        dict(member=dict(fold=n), spec=str(tmp_path/f'spec{n}'), output=str(tmp_path/f'fit{n}'))
        for n in (0, 1)])))
    probe = tmp_path/'serial-probe.ps1'
    prelude = r'''
$ErrorActionPreference='Stop'
$global:Events=New-Object System.Collections.Generic.List[string]
function nvidia-smi { $global:LASTEXITCODE=0 }
function Invoke-FakePython {
    $global:LASTEXITCODE=0
    if ($args -contains 'completed') {
        $global:Events.Add('check'); '{"complete":false}'; return
    }
    if ($args -notcontains 'tools/gpu_job.py') { throw 'Unexpected command' }
    $global:Events.Add('dispatch')
    $WaitIndex=[Array]::IndexOf($args,'--wait')
    $SepIndex=[Array]::IndexOf($args,'--')
    if ($WaitIndex -ge 0 -and $WaitIndex -lt $SepIndex) {
        $global:Events.Add('complete')
        $global:LASTEXITCODE=CHILD_CODE
    }
}
$Caught=$false
try {
& {
'''.replace('CHILD_CODE', str(child_code))
    suffix = "\n} -MemoryEvidence unused -Plan '" + str(plan).replace("'", "''") + "' -Folds 0,1\n"
    suffix += "} catch { if ($_.Exception.Message -notmatch 'Split local command failed') { throw }; $Caught=$true }\n"
    expected = 'check,dispatch,complete,check,dispatch,complete' if child_code == 0 else 'check,dispatch,complete'
    suffix += f"if (($global:Events -join ',') -ne '{expected}') {{ throw 'serial completion sequence mismatch' }}\n"
    suffix += f"if ($Caught -ne ${'false' if child_code == 0 else 'true'}) {{ throw 'child failure not propagated' }}\n"
    probe.write_text(prelude + script + suffix)
    result = subprocess.run(['powershell', '-NoProfile', '-File', str(probe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
