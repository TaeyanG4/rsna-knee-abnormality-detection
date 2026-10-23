"""Local static checks only. Does not import public source or deserialize weights."""
from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import ast
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile

ROOT=workspace_root()

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()

def main():
    package=ROOT/'external'/'mattiaangeli__rsna-knee-coat-resgated-ep10-top3'
    manifest=json.loads((package/'coat_resgated_ep10_top3_manifest.json').read_text())
    weights=[];sources=[]
    paths=list((ROOT/'weights').rglob('*.pt'))+list(package.glob('*.pt'))
    for p in sorted(paths):
        sha=digest(p);expected=manifest['files'].get(p.name)
        if expected is not None and expected!=sha:raise ValueError('Checkpoint hash mismatch: '+p.name)
        symbols=[]
        with zipfile.ZipFile(p) as z:
            entries=[i for i in z.infolist() if i.filename.endswith('/data.pkl')]
            if len(entries)!=1 or entries[0].file_size>20_000_000:raise ValueError('Unsupported checkpoint container')
            for op,arg,pos in pickletools.genops(z.read(entries[0])):
                if op.name=='GLOBAL':symbols.append(arg)
                elif op.name=='STACK_GLOBAL':symbols.append('STACK_GLOBAL_REQUIRES_MANUAL_REVIEW')
        row={'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha,'publisher_hash_matched':expected==sha if expected else None,'pickle_global_symbols':sorted(set(symbols)),'deserialized':False}
        weights.append(row);print('WEIGHT',json.dumps(row))
    for p in sorted(package.glob('*.py')):
        text=p.read_text(encoding='utf-8-sig');tree=ast.parse(text);sha=digest(p)
        if manifest['files'].get(p.name)!=sha:raise ValueError('Source hash mismatch: '+p.name)
        imports=sorted({n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module}|{a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names})
        flagged=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                call=ast.unparse(node.func)
                if any(x in call for x in ['subprocess','torch.load','pickle','exec','eval','requests','urlopen','remove','unlink','rmtree']):
                    flagged.append({'line':node.lineno,'call':call,'expression':ast.unparse(node)[:400]})
        row={'file':p.name,'sha256':sha,'hash_matched':True,'lines':len(text.splitlines()),'imports':imports,'review_calls':flagged}
        sources.append(row);print('SOURCE',p.name,'LINES',row['lines'],'IMPORTS',imports)
    result={'weights':weights,'sources':sources,'public_code_executed_locally':False,'notes':'Hash equality proves identity, not ownership or safety. Calls and serialization globals require contextual review.'}
    (ROOT/'metadata'/'component_static_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
