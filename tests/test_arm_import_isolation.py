"""v1.1.0: verify explicit EXP155 API after S6 collection, without a time gate.
Changelog: v1.0.0 isolated bare trainer import precedence."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("s6_first", [False, True])
def test_s6_collection_preserves_trainer_resolution(s6_first):
    # A clean interpreter makes the test independent of pytest collection order
    # and sys.modules caching. Do not fix precedence in the regression itself.
    root = Path(__file__).resolve().parents[1]
    code = f"""
from pathlib import Path
import runpy
import sys
root = Path.cwd()
def collect_s6():
    before = sys.path[:]
    runpy.run_path(str(root / 'tests/test_exp189_s6_port.py'), run_name='_s6_collection')
    assert sys.path == before, 'S6 collection mutated import search order'
if {s6_first!r}:
    collect_s6()
from tools import exp190_coatnet_oai_f0 as arm
if not {s6_first!r}:
    collect_s6()
import exp155_own0949_oai_v11 as trainer
assert Path(trainer.__file__).resolve() == root / 'training/src/exp155_own0949_oai_v11.py'
assert callable(trainer.train) and callable(trainer.tiny_validation)
assert arm.trainer_contract(trainer) == 'exp155-mixed-v2'
print('isolated trainer resolution OK')
"""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "isolated trainer resolution OK" in result.stdout
