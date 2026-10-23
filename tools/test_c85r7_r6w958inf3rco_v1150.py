"""H208 changed delivery boundaries only; the unchanged K3 suite is not rerun."""
import ast
import copy
import json

import pytest

from tools import c85r7_r6w958inf3rco_v1150 as b


def test_delivered_parent_is_exact_inf3_and_uses_existing_full_execution():
    nb, meta, _, _, files = b.package()
    _, _, outer = b.archive(''.join(nb['cells'][1]['source']))
    assert outer == files
    original = json.loads((b.INF3 / 'kernel/release.ipynb').read_bytes())
    source = ''.join(original['cells'][2]['source'])
    assert outer['parent-run.py'] == ('FAST = False\n' + source).encode()
    assert b.archive(outer['parent-run.py'].decode())[2] == b.archive(source)[2]
    assert len(b.archive(source)[2]) == 31
    assert meta['is_private'] and not meta['enable_internet']
    assert meta['id'] == 'taeyangg4/knee-lb-r6w958inf3rco-v1'


def test_no_execution_or_numeric_change_in_delivered_overlay():
    _, _, _, old, new = b.package()
    before = ast.parse(old['c85r7_rco_overlay_v1.py'])
    after = ast.parse(new['c85r7_rco_overlay_v1.py'])
    # Undo receipt display labels only; all function and import ASTs must agree.
    class UndoLabel(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant('H189'), node) if node.value == 'INF3' else node
    after = UndoLabel().visit(after)
    bodies = lambda t: [ast.dump(n) for n in t.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    assert bodies(before) == bodies(after)
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in after.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert literals['PARENT_VERSION'] == '1.1.46-r6w958inf3'
    assert literals['FORMULA'] == b.FORMULA
    assert literals['MODEL_SHA'] == b.overlay.MODEL_SHA


def test_corrupt_notebook_input_is_rejected(monkeypatch):
    monkeypatch.setattr(b, 'INF3_SHA', '0' * 64)
    with pytest.raises(ValueError, match='INF3 notebook pin'):
        b.package()


def test_unregistered_coordinator_edit_is_rejected():
    source = b.Path(b.overlay.__file__).read_bytes()
    with pytest.raises(ValueError, match='runtime source drift'):
        b.runtime_source(source.replace(b'.90 * ranks(parent)', b'.80 * ranks(parent)'))


def test_immutable_build_and_delivered_tamper(tmp_path):
    stage, work = tmp_path / 'stage', tmp_path / 'work'
    b.materialize(stage, work)
    assert b.validate(stage)['status'] == 'PASS_STATIC_B_DIRECT'
    with pytest.raises(ValueError, match='immutable stage'):
        b.materialize(stage, work)
    path = stage / 'kernel/kernel-metadata.json'
    meta = json.loads(path.read_bytes())
    meta['id'] = 'wrong/candidate'
    path.write_bytes(b.encoded(meta))
    with pytest.raises(ValueError, match='delivered drift'):
        b.validate(stage)
