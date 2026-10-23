"""Exercise the real runner startup predicate without importing a trainer."""
import ast
from pathlib import Path


def _predicate(directory):
    source = Path('training/runners/exp088_raptor384_v3.py').read_text()
    tree = ast.parse(source)
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'allowed_startup_files' for t in node.targets))
    condition = next(node.test for node in ast.walk(tree) if isinstance(node, ast.If)
                     and 'allowed_startup_files' in ast.unparse(node.test))
    allowed = ast.literal_eval(assignment.value)
    return eval(compile(ast.Expression(condition), '<real startup predicate>', 'eval'),
                {'out': directory, 'allowed_startup_files': allowed})


def test_owned_telemetry_and_replace_temps_are_allowed(tmp_path):
    for name in ('phase-authorization.json', 'supervisor.stdout.log', 'supervisor.stderr.log',
                 'supervisor-job.json', 'supervisor-job.json.tmp',
                 'resource-latest.json', 'resource-latest.json.tmp'):
        (tmp_path / name).write_text('synthetic metadata')
        assert not _predicate(tmp_path)


def test_previous_training_output_still_rejected(tmp_path):
    (tmp_path / 'resource-latest.json').write_text('{}')
    (tmp_path / 'recovery_latest.pt').write_bytes(b'previous run')
    assert _predicate(tmp_path)

