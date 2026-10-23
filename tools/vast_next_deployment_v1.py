"""v1.1.0: CPU-only recursive source audit for isolated Vast next-job sources.

Changelog: F2 accepts an explicit isolated source root; no runtime source edit.
F1: code pins alone are not an import closure.
Never imports training modules, transfers files, or reads data/credential files.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = '.local/scratch/astra-b2/vast-next-v1'
TERMINAL = '.local/scratch/astra-b2/fullbag-post-v1/fullbag_terminal_v1.py'
MODES = ('fullbag_all', 'all_seed2')
SOURCE = 'tools/exp189_s6_src'
SEARCH = ('', 'tools', SOURCE, 'training/src', '.local/exp189-cpu-v120/input/cloud_code')
ROOTS = ('tools/l1_rco_k1_fullbag_all_v1.py', 'tools/l1_rco_k1_all_seed2_v1.py',
         'tools/l1_rco_k1_nextlaunch_v1.py', 'tools/l1_rco_k1_nextlaunch_v2.py',
         'tools/gpu_job.py', 'tools/l1_rco_k1_foldcache_v2.py', TERMINAL)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_path(root, module, current, *, optional=False):
    """Resolve project imports without importing them; site-packages stay external."""
    relative = module.replace('.', '/')
    bases = [str(Path(current).parent)] + list(SEARCH)
    for base in bases:
        for suffix in ('.py', '/__init__.py'):
            p = Path(base) / (relative + suffix)
            if (root / p).is_file():
                return p.as_posix()
    if module.startswith('tools.') and not optional:
        raise ValueError('missing project import: ' + module + ' from ' + current)
    return None


def source_edges(root, name):
    tree = ast.parse((root/name).read_text(encoding='utf-8-sig'), filename=name)
    edges, external, dynamic = set(), set(), []
    def add(module):
        p = module_path(root, module, name)
        if p: edges.add(p)
        else: external.add(module.split('.')[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names: add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base=Path(name).parent
                for _ in range(node.level-1): base=base.parent
                if node.module: base=base.joinpath(*node.module.split('.'))
                candidates=[base.with_suffix('.py'),base/'__init__.py']
                candidates += [base/(alias.name+'.py') for alias in node.names]
                candidates += [base/alias.name/'__init__.py' for alias in node.names]
                found=[p.as_posix() for p in candidates if (root/p).is_file()]
                if not found: raise ValueError('unresolved relative import: '+name+':'+str(node.lineno))
                edges.update(found)
            elif node.module == 'tools':
                for alias in node.names: add('tools.'+alias.name)
            elif node.module:
                add(node.module)
                for alias in node.names:
                    p=module_path(root,node.module+'.'+alias.name,name,optional=True)
                    if p: edges.add(p)
        elif isinstance(node, ast.Call):
            fn = ast.unparse(node.func)
            if fn in ('__import__', 'importlib.import_module') and node.args and isinstance(node.args[0], ast.Constant):
                add(node.args[0].value)
            elif fn.endswith('spec_from_file_location') or fn in ('__import__', 'importlib.import_module', 'exec'):
                dynamic.append(dict(line=node.lineno, call=ast.unparse(node)))
    # Include package initialization as well as its imported module.
    for edge in list(edges):
        for parent in Path(edge).parents:
            init=(parent/'__init__.py').as_posix()
            if (root/init).is_file(): edges.add(init)
    return edges, external, dynamic


def audit(root=ROOT):
    required=set(ROOTS)
    pins={}
    for mode in MODES:
        reg=json.loads((root/TASK/mode/'registration.json').read_text())
        for name,h in {**reg['pins'], **reg['code_pins']}.items():
            if name.endswith('.py'):
                if name in pins and pins[name]!=h: raise ValueError('conflicting pin: '+name)
                pins[name]=h; required.add(name)
    # Explicit dynamically selected backend/shims; conservative factory closure
    # also contains historical seed1 and local-only branches, never executed here.
    for mode in ('fullbag_all','all_seed1','all_seed2','local_all_seed1'):
        required.update(('tools/l1_rco_k1_'+mode+'_v1.py', 'tools/l1_k1'+mode+'_trainer_v1.py'))
    required.add('tools/l1_rco_k1_fullbag_v1.py')
    # Derived trainer files are generated ON HOST by prepare, never deployed.
    generated={TASK+'/'+m+'/compact_train_l1.py' for m in MODES}
    pending=required|generated
    seen=set(); graph={}; external=set(); dynamic={}
    while pending:
        name=min(pending);pending.remove(name)
        if name in seen: continue
        if not (root/name).is_file(): raise ValueError('missing audit source: '+name)
        seen.add(name)
        edges,ext,dyn=source_edges(root,name)
        graph[name]=sorted(edges);external.update(ext)
        if dyn: dynamic[name]=dyn
        pending.update(edges-seen)
    required=seen-generated
    for name,h in pins.items():
        if sha(root/name)!=h: raise ValueError('registered source changed: '+name)
    return dict(files=sorted(required), graph=graph, dynamic_calls=dynamic,
                external_packages=sorted(external), generated_scan_only=sorted(generated),
                registered_pins=pins)


def validate_inventory(inventory, root=ROOT):
    result=audit(root)
    entries={row['path']:row['sha256'] for row in inventory['files']}
    missing=set(result['files'])-entries.keys()
    if missing: raise ValueError('deployment list missing imports/pins: '+', '.join(sorted(missing)))
    for name,h in entries.items():
        if sha(root/name)!=h: raise ValueError('deployment hash mismatch: '+name)
    if inventory['dynamic_calls']!=result['dynamic_calls']:
        raise ValueError('dynamic import sites changed; explicit re-audit required')
    if inventory['external_packages']!=result['external_packages']:
        raise ValueError('unresolved/external imports changed; explicit re-audit required')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['audit','validate'])
    p.add_argument('--inventory', type=Path, default=ROOT/TASK/'DEPLOYMENT-v1.json')
    p.add_argument('--source-root', type=Path, default=ROOT/TASK/'deploy-tree')
    a=p.parse_args()
    if a.action=='audit':
        print(json.dumps(audit(a.source_root),indent=2))
    else:
        r=validate_inventory(json.loads(a.inventory.read_text()),a.source_root)
        print(json.dumps(dict(status='PASS',required_source_files=len(r['files']),dynamic_files=len(r['dynamic_calls']))))


if __name__=='__main__': main()
