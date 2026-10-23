"""Fail-closed H-B-123 PowerShell preflight; no GPU dispatch or label access."""
from pathlib import Path
import json
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode,code", [("Validate", 0), ("Train", 2)])
def test_real_map_launch_gate(mode, code):
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(ROOT / ".local/run-scripts/hb123-arm-v2.ps1"), "-Mode", mode],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == code, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert records[0]["status"] == "VALIDATED"
    if mode != "Validate":
        assert records[-1]["status"] == "HOLD"
        assert records[-1]["launch_authorized"] is False
        assert records[-1]["required_launcher"] == "tools/gpu_job.py"


@pytest.mark.parametrize('mode', ['TargetSmokes', 'Train'])
def test_gpu_card_dry_run_never_dispatches(mode):
    result = subprocess.run(
        ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
         str(ROOT / '.local/run-scripts/hb123-arm-v2.ps1'), '-Mode', mode, '-DryRun'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=180,
    )
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    commands = [r for r in records if r['status'] == 'GPU_COMMAND_ONLY']
    assert len(commands) == 2
    for row in commands:
        assert row['launched'] is False
        assert 'tools/gpu_job.py' in row['command']
        assert 'training.runners.mil_laterality_v2' in row['command']
        assert 'training.runners.mil_screen_v1' not in row['command']

        assert row["command"][row["command"].index("--workers") + 1] == "2"
        assert "-f0-v2.json" in row["command"][row["command"].index("--spec") + 1]
