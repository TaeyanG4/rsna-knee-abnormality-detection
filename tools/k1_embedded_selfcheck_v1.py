"""v1.0.0: isolated CPU validation of the notebook's own pinned payload.

Changelog: execute the initial runtime imports with -I and no project imports;
validate the embedded REGISTERED config, notebook binding and module origins.
Never execute inference, install packages, access checkpoints or initialize CUDA.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

VERSION = '1.0.0'
ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / '.local/scratch/astra-b1/a5-flat-v1/embedded-selfcheck'


def payload(notebook):
    tree = ast.parse(''.join(notebook['cells'][2]['source']))
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('BLOB', 'PINS'):
                values[node.targets[0].id] = ast.literal_eval(node.value)
    blob = base64.b64decode(values['BLOB'], validate=True)
    blob_hash = hashlib.sha256(blob).hexdigest()
    assert any(isinstance(n, ast.Assert) and blob_hash in ast.unparse(n) for n in ast.walk(tree)), 'blob binding'
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == len(values['PINS']) and set(names) == set(values['PINS'])
        files = {}
        for name in names:
            assert not name.startswith(('/', '\\')) and '\\' not in name and ':' not in name and '..' not in Path(name).parts
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == values['PINS'][name], name
            files[name] = data
    return files, tree


def check(notebook, label):
    assert label and label not in ('.', '..') and all(c.isalnum() or c in '-_.' for c in label)
    files, _ = payload(notebook)
    work = SCRATCH / label
    runtime = work / 'payload'
    runtime.mkdir(parents=True, exist_ok=True)
    # Reuse a fixed folder, but fail on stale files rather than deleting anything.
    assert {p.relative_to(runtime).as_posix() for p in runtime.rglob('*') if p.is_file()} <= set(files), 'stale payload'
    for name, data in files.items():
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    nbfile = work / 'notebook.json'
    nbfile.write_text(json.dumps(notebook), encoding='utf-8')
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--worker', str(nbfile)]
    run = subprocess.run(command, cwd=work, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1'),
                         text=True, capture_output=True, timeout=90)
    (work / 'process.log').write_text(run.stdout + run.stderr, encoding='utf-8')
    if run.returncode:
        raise ValueError('embedded self-check failed: ' + label + '\n' + run.stderr[-4000:])
    result = json.loads(run.stdout.splitlines()[-1])
    assert result['status'] == 'PASS_ISOLATED_EMBEDDED'
    (work / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def worker(nbfile):
    assert sys.flags.isolated and sys.flags.dont_write_bytecode
    assert str(ROOT).lower() not in [str(Path(p).resolve()).lower() for p in sys.path]
    assert not any(n == 'tools' or n.startswith('tools.') for n in sys.modules)
    nbfile = Path(nbfile).resolve()
    notebook = json.loads(nbfile.read_bytes())
    files, tree = payload(notebook)
    runtime = nbfile.parent / 'payload'
    assert all((runtime / n).read_bytes() == data for n, data in files.items())
    ns = {'__name__': '__embedded_cpu_selfcheck__'}
    exec(compile(''.join(notebook['cells'][0]['source']), '<notebook initial cell>', 'exec'), ns)
    assert ns['FAST'] is False
    body = tree.body[0].body
    for node in body:
        if isinstance(node, ast.Import) and all(a.name in ('base64', 'hashlib', 'io', 'sys', 'zipfile') for a in node.names) or isinstance(node, ast.ImportFrom) and node.module == 'pathlib':
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook standard imports>', 'exec'), ns)
    ns['runtime'] = runtime
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Import) and any(a.name == 'exp169_release_c85r7codec' for a in n.names))
    stop = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'release.run_release')
    assert start < stop
    sys.path.insert(0, str(runtime))
    imports = [n for n in body[start:stop] if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) >= 2
    exec(compile(ast.Module(body=imports, type_ignores=[]), '<notebook runtime import prefix>', 'exec'), ns)
    # The later A5 import is checked too, without running its inference call.
    a5imports = [n for n in body if isinstance(n, ast.ImportFrom) and n.module == 'tools' and any(a.name == 'p25_a5_runtime_v1' for a in n.names)]
    for node in a5imports:
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook A5 import>', 'exec'), ns)
    from tools import k1n_config_v1 as schema
    k1 = ns['_k1sr']
    source = ast.parse(files['tools/k1sr_runtime_v10.py'])
    literal = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REGISTERED' for t in n.targets) and isinstance(n.value, ast.Dict))
    assert schema.validate(literal) == k1.REGISTERED
    finish = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == '_k1sr.finish')
    config = ast.literal_eval(finish.args[2])
    assert config['config_sha256'] == config['registered_config_sha256'] == k1.CONFIG_SHA
    assert config['kernel_id'] == literal['kernel_id'] + ('-n32' if config['diagnostic'] else '')
    origins = {}
    for name, module in list(sys.modules.items()):
        if name == 'tools' or name.startswith('tools.') or name.startswith('exp169_'):
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(runtime), 'project module leak: ' + name
            origins[name] = path.relative_to(runtime).as_posix()
    assert 'tools.k1n_config_v1' in origins and 'tools.k1sr_runtime_v10' in origins
    print(json.dumps(dict(status='PASS_ISOLATED_EMBEDDED', selfcheck_version=VERSION,
        isolated=True, project_tools_on_sys_path=False, module_origins=origins,
        config_sha256=k1.CONFIG_SHA, schema_sha256=hashlib.sha256(files['tools/k1n_config_v1.py']).hexdigest(),
        kernel_id=config['kernel_id'], diagnostic=config['diagnostic'], a5_imported=bool(a5imports),
        inference_executed=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--stage')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        stage = Path(args.stage)
        print(json.dumps(check(json.loads((stage/'kernel/release.ipynb').read_bytes()), stage.name), indent=2))


if __name__ == '__main__':
    main()
