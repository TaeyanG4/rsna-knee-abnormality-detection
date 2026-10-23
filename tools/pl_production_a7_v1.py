"""Scoped A7 producer compatibility 1.0.0; preserve every original binding check.

The authenticated heldout producer is 1.1.3. Compile the existing A7 validator
with only that version literal updated, and additionally validate its production
seal. No receipt metadata is changed and the shared A7 module stays untouched.
"""
import ast
import importlib.util
import inspect
from tools import vnext_r7_pl_v1 as original
from tools import pl_production_teacher_v1 as teacher

spec = importlib.util.spec_from_file_location('_production_a7', original.__file__)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
tree = ast.parse(inspect.getsource(original.validate_bindings))
changes = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Compare) and isinstance(node.left, ast.Call):
        call = node.left
        if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                and call.func.value.id == 'teacher' and call.func.attr == 'get'
                and len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == 'producer_version'):
            original.need(len(node.comparators) == 1 and
                          isinstance(node.comparators[0], ast.Constant) and
                          node.comparators[0].value == '1.1.2', 'A7 producer check changed')
            node.comparators[0].value = teacher.PRODUCER_VERSION
            changes += 1
original.need(changes == 1, 'exactly one A7 version compatibility adjustment required')
exec(compile(tree, original.__file__, 'exec'), adapter.__dict__)
inherited_validate = adapter.validate_bindings


def validate_bindings(reg, bindings):
    teacher.validate(adapter.pin(bindings['teacher']))
    return inherited_validate(reg, bindings)


adapter.validate_bindings = validate_bindings
