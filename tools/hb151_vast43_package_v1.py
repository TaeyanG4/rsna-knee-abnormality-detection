"""HB151 seed43 Vast package v1.0.0. CPU only; originals remain untouched.

Changelog: portable Linux guard and mechanically identical seed42 trainer,
seed43 registration, exact transfer inventory, host parity and launch gates.
"""
from pathlib import Path
import ast
import copy
import difflib
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.local/scratch/astra-b5/H-B-151-seed43'
PACKAGE = WORK / 'package'
STAGE = PACKAGE / 'stage-seed43-v1'


def generate():
    source = (ROOT/'tools/hb151_build_v3.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    names = ('probability_rows', 'write_bytes', 'write_new_json', 'save_torch_atomic',
             'export_checkpoint', 'recover_export', 'require_validation', 'cpu_validate',
             'python_blockers', 'admission_snapshot', 'desktop_memory_snapshot',
             'startup_publication', 'check_smoke', 'charged_seconds', 'cuda_smoke', 'supervise')
    functions = {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    transformed = []
    for name in names:
        text = functions[name].replace('tools.hb151_build_v3', 'tools.hb151_vast43_v1')
        text = text.replace('import hb151_v3 as', 'import hb151_vast43_v1 as')
        # These are the only literal scientific changes: seed 42 -> 43.
        text = text.replace("'bf16', 42, False", "'bf16', 43, False")
        text = text.replace(', 42)', ', 43)').replace(', 42,', ', 43,')
        text = text.replace(',42)', ',43)').replace(',42,', ',43,')
        text = text.replace('_cpu_seed(42)', '_cpu_seed(43)').replace('manual_seed_all(42)', 'manual_seed_all(43)')
        text = text.replace("tag='S42'", "tag='S43'").replace('seed=42', 'seed=43')
        text = text.replace("'4070 Ti SUPER'", "'RTX 4090'").replace('16000 <=', '24000 <=').replace('<= 16400', '<= 25000')
        text = text.replace('real 4070 measurement required', 'real 4090 measurement required')
        if name == 'cpu_validate':
            text = text.replace('corpus = parent.check_corpus(CORPUS)', 'verify_cache(p)\n    corpus = parent.check_corpus(CORPUS)')
        if name == 'cuda_smoke':
            text = text.replace("atomic_json(STAGE/'cuda-smoke.json',s); check_smoke(s)",
                "s['runtime_sha256'] = sha256(STAGE/'runtime-effective.json')\n    atomic_json(STAGE/'cuda-smoke.json',s); check_smoke(s)")
        transformed.append(text)
    body = (ROOT/'tools/hb151_vast43_guard_v1.txt').read_text(encoding='utf-8')
    body = body.replace('# INSERT_PARENT_FUNCTIONS', '\n\n\n'.join(transformed)+'\n\nsave_torch_new = save_torch_atomic')
    (ROOT/'tools/hb151_vast43_v1.py').write_text(body, encoding='utf-8', newline='\n')
    trainer = (ROOT/'training/src/hb151_v3.py').read_text(encoding='utf-8')
    assert trainer.count('hb151_build_v3') == 2
    new = trainer.replace('hb151_build_v3', 'hb151_vast43_v1')
    (ROOT/'training/src/hb151_vast43_v1.py').write_text(new, encoding='utf-8', newline='\n')
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/'trainer.diff').write_text(''.join(difflib.unified_diff(trainer.splitlines(True), new.splitlines(True),
        fromfile='seed42/trainer', tofile='seed43/trainer')), encoding='utf-8')


def build():
    import sys
    sys.path.insert(0, str(ROOT))
    from tools import hb151_build_v3 as old
    from tools.hb151_trainer_codegen_v3 import verify
    verify()
    generate()
    STAGE.mkdir(parents=True, exist_ok=True)
    assert not any((STAGE/n).exists() for n in ['S43','smoke-attempt','independent-review.json']), 'package already in use'
    source_stage = ROOT/'.local/scratch/astra-b4/H-B-151/v3'
    source = old.read_json(source_stage/'manifest.json')
    current = old.read_json(old.PROTOCOL)
    for key in ['recipe','pos_weight','train_ids','excluded_gold_ids','train_labels','label_sha256','init_sha256']:
        assert source[key] == current[key], 'v3 vs v3.0.2 data/recipe drift: '+key
    assert old.sha256(source_stage/'soft-labels.csv') == source['label_sha256']
    assert old.sha256(old.INIT) == source['init_sha256']
    files = sorted(set(old.SOURCES + ['tools/hb151_vast43_v1.py','training/src/hb151_vast43_v1.py',
        'tools/hb151_vast43_package_v1.py','tools/hb151_vast43_guard_v1.txt','tools/ml_parity/parity_runtime.py',
        'config/workspace.toml']))
    for rel in files:
        dest = PACKAGE/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/rel, dest)
    for rel in ['tools/__init__.py','training/__init__.py','tools/ml_parity/__init__.py']:
        if (ROOT/rel).exists():
            shutil.copyfile(ROOT/rel, PACKAGE/rel); files.append(rel)
    for name in ['soft-labels.csv','label-provenance.json']:
        shutil.copyfile(source_stage/name, STAGE/name)
    shutil.copyfile(ROOT/'tools/hb151_vast43_run_v1.sh',PACKAGE/'run.sh')
    files.append('run.sh')
    shutil.copyfile(source_stage/'manifest.json', STAGE/'seed42-manifest.json')
    (PACKAGE/'weights').mkdir(exist_ok=True)
    shutil.copyfile(old.INIT, PACKAGE/'weights/coatnet384-imagenet.pt')
    (PACKAGE/'references').mkdir(exist_ok=True)
    for name in ['4070-reference.json','kaggle-reference.json','vast-observed.json','trainer.diff']:
        shutil.copyfile(WORK/name, PACKAGE/'references'/name)
    shutil.copyfile(ROOT/'.local/vast-4090-20261015/runtime-snapshot.md', PACKAGE/'references/runtime-snapshot.md')
    files += [p.relative_to(PACKAGE).as_posix() for p in (PACKAGE/'references').iterdir() if p.is_file()]
    cache_files = {p.name:p.stat().st_size for p in old.CORPUS.iterdir() if p.is_file()}
    p = copy.deepcopy(source)
    p.update(version='1.0.0', schema='hb151-vast-seed43/v1', run_dirs={'local':'stage-seed43-v1/S43'},
             source_bindings={rel:old.sha256(PACKAGE/rel) for rel in files},
             seed42_manifest_sha256=old.sha256(STAGE/'seed42-manifest.json'),
             cache_files=cache_files, parent_guard_sha256=old.sha256(ROOT/'tools/hb151_build_v3.py'))
    p['recipe']['seed_local'] = 43
    p['decision']['seed'] = 43
    p.pop('content_sha256', None); p['content_sha256'] = old.parent.json_hash(p)
    old.atomic_json(STAGE/'manifest.json', p)
    old.atomic_json(STAGE/'decision.json', p['decision'])
    # The original files are archived as evidence, never modified.
    backup = ROOT/'.local/backups/hb151-vast43-v1'; backup.mkdir(parents=True, exist_ok=True)
    for rel in ['tools/hb151_build_v3.py','training/src/hb151_v3.py']:
        dest=backup/Path(rel).name
        if dest.exists(): assert old.sha256(dest)==old.sha256(ROOT/rel), 'backup collision'
        else: shutil.copyfile(ROOT/rel,dest)
    return p


def archive():
    from tools.hb151_build_v3 import sha256, atomic_json
    # Never package run artifacts or CPU checkpoint fixtures.
    paths = sorted(p for p in PACKAGE.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts and p.name not in ['cpu-checkpoint.pt','INVENTORY.json','SHA256SUMS'])
    assert not (STAGE/'S43').exists()
    assert not (STAGE/'smoke-attempt').exists()
    inventory = [dict(path=p.relative_to(PACKAGE).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in paths]
    atomic_json(PACKAGE/'INVENTORY.json',dict(version='1.0.0',files=inventory))
    (PACKAGE/'SHA256SUMS').write_text(''.join(f"{p['sha256']}  {p['path']}\n" for p in inventory),encoding='utf-8',newline='\n')
    archive_path=WORK/'hb151-seed43-v1.tar'
    with tarfile.open(archive_path,'w') as tar:
        for p in paths+[PACKAGE/'INVENTORY.json',PACKAGE/'SHA256SUMS']:
            tar.add(p,arcname='hb151-seed43-v1/'+p.relative_to(PACKAGE).as_posix(),recursive=False)
    (WORK/'hb151-seed43-v1.tar.sha256').write_text(sha256(archive_path)+'  hb151-seed43-v1.tar\n')
    print(json.dumps(dict(files=len(inventory),bytes=sum(p['bytes'] for p in inventory),archive_sha256=sha256(archive_path))))


if __name__=='__main__':
    import sys
    if sys.argv[1:] == ['archive']: archive()
    else:
        p=build();print(json.dumps(dict(status='PACKAGED',train_rows=len(p['train_ids']),cache_files=len(p['cache_files']),cache_bytes=sum(p['cache_files'].values()))))
