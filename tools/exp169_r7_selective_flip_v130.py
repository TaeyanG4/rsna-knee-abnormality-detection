"""v1.3.0: sealed native R7 flip policy; original sources remain immutable.

Changelog 2026-10-13: fixed checkpoint-bound f0+f2 / all-fold policies.
"""
from contextlib import contextmanager
import ast
import hashlib
import json
from pathlib import Path

VERSION = '1.3.0'
VARIANTS = {'FLIP_F0F2': (0, 2), 'FLIP_ALL': (0, 1, 2, 3)}
ROSTER = tuple(f'E_lrprior_s0_f{k}.pt' for k in range(4))


def policy(variant):
    return dict(version=VERSION, variant=variant, flip_folds=list(VARIANTS[variant]),
                flip_models=[ROSTER[k] for k in VARIANTS[variant]],
                registration='H-B-149 v1.2.1', mcl='plain-only')


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError('selective flip anchor drift: ' + old[:80])
    return source.replace(old, new)


def patch_worker(source, variant):
    selected = policy(variant)
    # v1.2.2 atomic adapters use ast.unparse; bind to syntax, not whitespace.
    tree = ast.parse(source)
    def named(name):
        found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(found) != 1:
            raise ValueError('selective flip anchor drift: '+name)
        return found[0]
    load, score, main = (named(n) for n in ('load_group', 'score_v4c', 'main'))
    def insert_after(body, predicate, statements):
        positions = [i for i, n in enumerate(body) if predicate(n)]
        if len(positions) != 1:
            raise ValueError('selective flip anchor drift')
        i = positions[0]
        body[i+1:i+1] = ast.parse(statements).body
    insert_after(load.body, lambda n: isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'fs',
        f'if tuple(p.name for p in fs) != {ROSTER!r}:\n raise ValueError("sealed R7 fold roster mismatch")')
    loop = next(n for n in load.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'p')
    loop.body.insert(0, ast.parse(f'_selected_flip = p.name in {tuple(selected["flip_models"])!r}').body[0])
    insert_after(loop.body, lambda n: isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'm',
                 'm._c85r7_flip = _selected_flip')
    matches = [n for n in ast.walk(score) if isinstance(n, ast.If) and ast.unparse(n.test) == 'FLIP is None']
    if len(matches) != 1 or ast.unparse(matches[0].body[0]) != 'out.append(z[0])':
        raise ValueError('selective flip anchor drift: score_v4c')
    matches[0].test = ast.parse('FLIP is None or not m._c85r7_flip', mode='eval').body
    insert_after(main.body, lambda n: isinstance(n, ast.Assign) and ast.unparse(n) == 'a = ap.parse_args()',
        'if not a.flip_tta or a.models2:\n raise ValueError("sealed R7 policy requires native flip and one model group")')
    receipts = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'receipt' for t in n.targets) and isinstance(n.value, ast.Dict)]
    if len(receipts) != 1:
        raise ValueError('selective flip anchor drift: receipt')
    receipts[0].value.keys.append(ast.Constant('flip_policy'))
    receipts[0].value.values.append(ast.parse(repr(selected), mode='eval').body)
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    compile(source, '<sealed-r7-flip>', 'exec')
    return source


@contextmanager
def installed(r7, variant, metadata_io):
    """Install after the v1.2.2 atomic/deadline adapters, within R7 only."""
    selected = policy(variant)
    original_source, original_args = r7.diagnostic_source, r7.worker_args
    original_provenance = r7.write_provenance
    def diagnostic_source(*args, **kwargs):
        runtime = original_source(*args, **kwargs)
        worker = runtime / 'compact_infer.py'
        metadata_io.atomic_text(worker, patch_worker(worker.read_text(encoding='utf-8'), variant))
        metadata_io.atomic_text(runtime / 'runtime_sha256.json', json.dumps(
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(runtime.glob('*.py'))}, indent=2) + '\n')
        return runtime
    def worker_args(manifest, *args, **kwargs):
        return original_args(dict(manifest, flip_tta=True), *args, **kwargs)
    def write_provenance(path, **kwargs):
        result = original_provenance(path, **kwargs)
        result['flip_policy'] = selected
        metadata_io.atomic_text(Path(path), json.dumps(result, indent=2) + '\n')
        return result
    r7.diagnostic_source, r7.worker_args, r7.write_provenance = diagnostic_source, worker_args, write_provenance
    try:
        yield
    finally:
        r7.diagnostic_source, r7.worker_args, r7.write_provenance = original_source, original_args, original_provenance
