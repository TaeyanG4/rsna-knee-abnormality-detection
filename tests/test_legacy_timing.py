"""v1.0.0: execute production timing predicates with long valid measurements.

GPU orchestration is inspected, never launched. Predicate extraction executes
the exact production AST; real data execution is covered by CPU launch cards.
"""
from __future__ import annotations
import ast
import copy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['exp170_convnext_t_l2', 'exp171_convnext_s_l2',
        'exp179_multiscale', 'exp183_multiscale_5fold']
SWEEP = ARMS + ['exp172_dino_base_l2', 'exp173_gold_crossfit',
                'exp174_own0949_jev', 'exp182_synovitis_fill', 'exp184_aquila_share']

def check(ok, message):
    if not ok:
        raise ValueError(message)

def exact_calls(module, text):
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8-sig'))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'need'
            and text in ast.unparse(node)]

def run_call(node, **values):
    return eval(compile(ast.Expression(copy.deepcopy(node)), '<production-predicate>', 'eval'),
                {'math': math, 'need': check}, values)

@pytest.mark.parametrize('name', ARMS)
def test_pilot_accepts_long_valid_but_rejects_bad_measurements(name):
    module = importlib.import_module('tools.' + name)
    nodes = exact_calls(module, 'invalid pilot timing or memory')
    assert len(nodes) == 1
    for seconds in [1., 1e9]:
        run_call(nodes[0], pilot={'epoch_seconds': seconds, 'peak_gpu_gib': 12.})
    for key, value in [('epoch_seconds', 0), ('epoch_seconds', float('nan')),
                       ('epoch_seconds', float('inf')), ('peak_gpu_gib', -1),
                       ('peak_gpu_gib', 20), ('peak_gpu_gib', float('nan'))]:
        pilot = {'epoch_seconds': 1e9, 'peak_gpu_gib': 12.}
        pilot[key] = value
        with pytest.raises(ValueError):
            run_call(nodes[0], pilot=pilot)

@pytest.mark.parametrize('name', SWEEP)
def test_editable_orchestration_has_no_subprocess_deadline(name):
    module = importlib.import_module('tools.' + name)
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8-sig'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not [n for n in calls if any(k.arg == 'timeout' for k in n.keywords)]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Raise)
                and ('Timeout' in ast.unparse(n) or 'deadline' in ast.unparse(n))]

@pytest.mark.parametrize('name', ['exp155_own0949_oai', 'exp163_own0949_oai_ext'])
def test_old_training_timing_observes_without_aborting(tmp_path, name):
    # Execute the real nested training timing function on CPU. Training state,
    # optimizer and model code are not replaced or run on the active GPU.
    source = ROOT / 'training/src' / (name + '.py')
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name == 'record_timing')
    fn = copy.deepcopy(node)
    fn.body = [n for n in fn.body if not isinstance(n, ast.Nonlocal)]
    logs = []
    scope = dict(projected_once=False, wall=lambda: 1e9, world=1,
                 this_updates=100, rank=0, total_steps=5000, updates=100,
                 skips=0, reserve=600, limit=3600, rate_start=0,
                 time=SimpleNamespace(monotonic=lambda: 1e9, time_ns=lambda: 1),
                 guard=SimpleNamespace(write_new_json=lambda p, value: logs.append(value)),
                 out=tmp_path, start_epoch=0)
    # nonlocal becomes global in this standalone execution fixture only.
    fn.body.insert(0, ast.Global(names=['projected_once']))
    unit=ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(unit, str(source), 'exec'), scope)
    scope['record_timing']()
    scope['record_timing']()
    assert len(logs) == 1 and logs[0]['projected_seconds'] > 1e9
    assert logs[0]['limit_seconds'] == 3600
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Raise)
                and 'TimeoutError' in ast.unparse(n)]

def test_dino_long_projection_keeps_scientific_checks():
    from tools import exp172_dino_base_l2 as module
    node, = exact_calls(module, 'invalid pilot measurement or memory gate failed')
    receipt = dict(updates=32, validation_rows=0, projected_full_seconds=1e9,
                   peak_gpu_allocated_bytes=1024)
    run_call(node, receipt=receipt, BUDGET=module.BUDGET)
    for key, value in [('updates', 31), ('validation_rows', 1),
                       ('projected_full_seconds', float('nan')),
                       ('peak_gpu_allocated_bytes', module.BUDGET['peak_gpu_allocated_cap_bytes']+1)]:
        wrong = dict(receipt, **{key:value})
        with pytest.raises(ValueError):
            run_call(node, receipt=wrong, BUDGET=module.BUDGET)
