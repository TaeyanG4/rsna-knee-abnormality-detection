"""Seed44 package v1.0.0: copy the reviewed seed43 artifact, never regenerate it.

Changelog: seed/name-only science delta; LF archive pin; fail-fast busy-GPU
admission. Frozen originals remain unchanged.
"""
from pathlib import Path
import copy
import difflib
import hashlib
import json
import re
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / '.local/scratch/astra-b5/H-B-151-seed43/package'
WORK = ROOT / '.local/scratch/astra-b5/H-B-151-seed44'
PACKAGE = WORK / 'package'
STAGE = PACKAGE / 'stage-seed44-v1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')


def names(text):
    return text.replace('vast43', 'vast44').replace('seed43', 'seed44').replace('Seed43', 'Seed44').replace('S43', 'S44')


def runtime(source):
    text = re.sub(r'\b43\b', '44', names(source))
    text = text.replace("VERSION='1.0.0'", "# Seed44 v1.0.0: seed/name delta + fail-fast busy-GPU fix.\nVERSION='1.0.0'")
    text = text.replace("    if not child: return", "    if not child:\n        from tools.hb151_vast44_admission_v1 import require_idle\n        require_idle(desktop_memory_snapshot)\n        return")
    text = text.replace("choices=['verify'", "choices=['launch-check','verify'")
    text = text.replace("    if args.action in ('smoke','fit'):", "    if args.action in ('launch-check','smoke','fit'):")
    text = text.replace("            ready();require_validation(PROTOCOL)",
        "            from tools.hb151_vast44_admission_v1 import require_idle\n            require_idle(desktop_memory_snapshot)\n            if args.action=='launch-check': return 0\n            ready();require_validation(PROTOCOL)")
    return text


def build():
    WORK.mkdir(parents=True, exist_ok=True)
    PACKAGE.mkdir(exist_ok=True)
    assert not any((STAGE/n).exists() for n in ('S44','smoke-attempt','independent-review.json')), 'package in use'
    inv=json.loads((PARENT/'INVENTORY.json').read_text(encoding='utf-8'))
    diffs=[]; copied=[]
    ignored={'BUILD.md','LAUNCH.md','stage-seed43-v1/cpu-validation.json'}
    for entry in inv['files']:
        rel=entry['path']; src=PARENT/rel
        assert sha(src)==entry['sha256'], 'frozen parent drift: '+rel
        if rel in ignored or rel.startswith('evidence/'):
            continue
        # Keep archived parent generator/template/references literally unchanged.
        renamed = names(rel) if rel in ('tools/hb151_vast43_v1.py','training/src/hb151_vast43_v1.py') or rel.startswith('stage-seed43-v1/') else rel
        dest=PACKAGE/renamed;dest.parent.mkdir(parents=True,exist_ok=True)
        payload=src.read_bytes()
        if rel=='tools/hb151_vast43_v1.py':
            payload=runtime(payload.decode()).encode()
        elif rel=='training/src/hb151_vast43_v1.py':
            payload=names(payload.decode()).encode()
        elif rel=='run.sh':
            payload=names(payload.decode()).replace('python -B -m tools.hb151_vast44_v1 verify',
                'python -B -m tools.hb151_vast44_v1 launch-check\npython -B -m tools.hb151_vast44_v1 verify').encode()
        dest.write_bytes(payload)
        copied.append(renamed)
        if payload!=src.read_bytes() and rel!='stage-seed43-v1/manifest.json':
            diffs.extend(difflib.unified_diff(src.read_text().splitlines(True),payload.decode().splitlines(True),fromfile=rel,tofile=renamed))
    helper='tools/hb151_vast44_admission_v1.py'
    shutil.copyfile(ROOT/helper,PACKAGE/helper)
    diffs.extend(difflib.unified_diff([], (PACKAGE/helper).read_text().splitlines(True),
                                    fromfile='/dev/null',tofile=helper))
    p=json.loads((PARENT/'stage-seed43-v1/manifest.json').read_text())
    p['schema']=names(p['schema']);p['run_dirs']={k:names(v) for k,v in p['run_dirs'].items()}
    p['recipe']['seed_local']=44;p['decision']['seed']=44
    p['source_bindings']={names(k) if k in ('tools/hb151_vast43_v1.py','training/src/hb151_vast43_v1.py') else k:
        sha(PACKAGE/(names(k) if k in ('tools/hb151_vast43_v1.py','training/src/hb151_vast43_v1.py') else k)) for k in p['source_bindings']}
    p['source_bindings'][helper]=sha(PACKAGE/helper)
    p.pop('content_sha256')
    p['content_sha256']=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    write_json(STAGE/'manifest.json',p);write_json(STAGE/'decision.json',p['decision'])
    # No label rows in diff/report. Verify entire scientific registration in memory.
    old=json.loads((PARENT/'stage-seed43-v1/manifest.json').read_text())
    expected=copy.deepcopy(old)
    for key in ('schema','run_dirs','source_bindings','content_sha256'): expected[key]=p[key]
    expected['recipe']['seed_local']=44;expected['decision']['seed']=44
    assert p==expected, 'non-allowlisted registration delta'
    # The only row-bearing field is unchanged; never emit labels in evidence.
    public_keys=('schema','version','run_dirs','recipe','decision','source_bindings','content_sha256')
    for name, before, after in [
        ('manifest.json (non-row fields)',{k:old[k] for k in public_keys},{k:p[k] for k in public_keys}),
        ('decision.json',old['decision'],p['decision'])]:
        diffs.extend(difflib.unified_diff(json.dumps(before,indent=2).splitlines(True),
            json.dumps(after,indent=2).splitlines(True),fromfile='seed43/'+name,tofile='seed44/'+name))
    (WORK/'full-code.diff').write_text(''.join(diffs),encoding='utf-8',newline='\n')
    for rel in ('tools/hb151_vast44_v1.py','training/src/hb151_vast44_v1.py'):
        shutil.copyfile(PACKAGE/rel,ROOT/rel)
    shutil.copyfile(PACKAGE/'run.sh',ROOT/'tools/hb151_vast44_run_v1.sh')
    transfer=names((ROOT/'tools/hb151_vast43_transfer_v1.ps1').read_text())
    (ROOT/'tools/hb151_vast44_transfer_v1.ps1').write_text(transfer,encoding='utf-8',newline='\n')
    write_json(WORK/'mechanical-diff.json',dict(status='PASS',parent_files_verified=len(inv['files']),
        scientific_delta={'recipe.seed_local':[43,44],'decision.seed':[43,44]},
        allowed_code_fixes=['LF archive pin','fail-fast HB151 process/free-memory gate'],
        recipe_otherwise_identical=True,labels_and_init_byte_identical=True))
    print(json.dumps(dict(status='PACKAGED',seed=44,train_rows=len(p['train_ids']),cache_files=len(p['cache_files']))))


def archive():
    paths=sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.name not in ('cpu-checkpoint.pt','INVENTORY.json','SHA256SUMS'))
    assert not any((STAGE/n).exists() for n in ('S44','smoke-attempt','independent-review.json'))
    inv=[dict(path=p.relative_to(PACKAGE).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    write_json(PACKAGE/'INVENTORY.json',dict(version='1.0.0',files=inv))
    (PACKAGE/'SHA256SUMS').write_text(''.join(f"{p['sha256']}  {p['path']}\n" for p in inv),encoding='utf-8',newline='\n')
    target=WORK/'hb151-seed44-v1.tar'
    with tarfile.open(target,'w') as tar:
        for p in paths+[PACKAGE/'INVENTORY.json',PACKAGE/'SHA256SUMS']:
            tar.add(p,arcname='hb151-seed44-v1/'+p.relative_to(PACKAGE).as_posix(),recursive=False)
    pin=sha(target)+'  hb151-seed44-v1.tar\n'
    target.with_suffix('.tar.sha256').write_bytes(pin.encode('ascii'))
    assert b'\r' not in target.with_suffix('.tar.sha256').read_bytes()
    print(json.dumps(dict(status='ARCHIVED',files=len(inv),bytes=target.stat().st_size,sha256=sha(target))))


if __name__=='__main__':
    import sys
    archive() if sys.argv[1:]==['archive'] else build()
