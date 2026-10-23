"""PL15 release transport 1.1.1. Offline build; all account actions belong to B.

Changelog: portable source/checkpoint seal, notebook-pinned manifest, fivefold
production binding, private upload layout, review and target admission gates.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from tools import c85r7_v150_safety as io_safe

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.1.1'
PARENT = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.4.0'
STAGE = PARENT/'production-release-v1.1.1'
WORK = ROOT/'.local/scratch/pl15-release/release-v110'
SLUG = 'taeyangg4/knee-c85r7-pl15-s42-v111'
IMAGE = '2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'
OWN_SOURCES = ['tools/pl15_transport_v110.py', 'tools/pl15_runtime_v110.py',
               'tools/c85r7_v150_safety.py']
SMOKE = ROOT/'.local/scratch/astra-b4/H-B-146/launch-v115/smoke-cuda/control-s42/final.pt'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def safe_path(root, name):
    need(isinstance(name, str) and '\\' not in name and ':' not in name, 'nonportable path')
    part = PurePosixPath(name)
    need(not part.is_absolute() and name and '..' not in part.parts, 'unsafe relative path')
    path = (Path(root)/name).resolve()
    need(path.is_relative_to(Path(root).resolve()), 'path escapes package')
    return path


def write(path, data, immutable=False):
    return io_safe.atomic_write(path, data, immutable=immutable)


def dump(path, value, immutable=False):
    return write(path, encoded(value), immutable)


def pin(root, name):
    path = safe_path(root, name)
    return dict(path=name, bytes=path.stat().st_size, sha256=sha(path))


def check(root, asset):
    path = safe_path(root, asset['path'])
    need(pin(root, asset['path']) == asset, 'asset hash/size differs: '+asset['path'])
    return path


def copy(source, destination):
    """Bounded replacement; a partial copy is never advertised by a manifest."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+'.part')
    with Path(source).open('rb') as inp, temporary.open('wb') as out:
        shutil.copyfileobj(inp, out, 8*1024*1024)
        out.flush()
        import os
        os.fsync(out.fileno())
    io_safe.retry(io_safe._replace, temporary, destination)


def source_archive():
    from tools import vnext_r7_pl_v1 as pl
    from tools import exp169_release_c85r7_pl15_v140 as blend
    files = pl.source_payloads()
    # These build-only metadata files contain local asset paths; not runtime inputs.
    for name in ('sealed-registration.json', 'q-authority.json'):
        files.pop(name, None)
    for name in [*OWN_SOURCES, 'tools/exp169_release_c85r7_pl15_v140.py']:
        files[name] = (ROOT/name).read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            safe_path(ROOT, name)
            info = zipfile.ZipInfo(name, (2026, 10, 14, 0, 0, 0))
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue(), {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def register(stage=STAGE):
    from tools import pl_production_release_v1 as old
    from tools import hb116_vnext_release as h
    old.validate(old.OUT)
    stage = Path(stage)
    archive, sources = source_archive()
    immutable = stage.resolve() != (WORK/'stage').resolve()
    write(stage/'runtime.zip', archive, immutable)
    for name in ('base-v122-runtime.zip', 'base-v122-pins.json'):
        write(stage/name, (PARENT/name).read_bytes(), True)
    manifest = h.OWN_DIR/'release-manifest.json'
    need(sha(manifest) == h.OWN_MANIFEST_SHA, 'own reader manifest changed')
    write(stage/'own-manifest.json', manifest.read_bytes(), True)
    reg = dict(version=VERSION, checklist='1.0.3', parent_sha256=sha(PARENT/'registration.json'),
        child_sha256=sha(old.OUT/'registration.json'), sources=sources,
        files=[pin(stage, name) for name in ('runtime.zip', 'base-v122-runtime.zip',
                                             'base-v122-pins.json', 'own-manifest.json')],
        members=old.MIL, k=36, resolution=224, tta='coronal_axial_horizontal_mean_probability',
        variants=old.release.VARIANTS, compute_seconds=27000, publication_reserve_seconds=1200,
        base_projection_seconds=16920, hidden_studies=1300, exposed='EXPOSED_EXPLORATORY', clean_oof=False)
    dump(stage/'registration.json', reg, immutable)
    write(stage/'registration.sha256', (sha(stage/'registration.json')+'\n').encode(), immutable)
    return dict(status='REGISTERED', version=VERSION, registration_sha256=sha(stage/'registration.json'))


def validate(stage=STAGE):
    stage = Path(stage)
    reg = read(stage/'registration.json')
    need(sha(stage/'registration.json') == (stage/'registration.sha256').read_text().strip(), 'stage seal')
    for asset in reg['files']:
        check(stage, asset)
    for name in OWN_SOURCES:
        need(sha(ROOT/name) == reg['sources'][name], 'build source drift: '+name)
    need(reg['version']==VERSION and len(reg['members'])==5 and reg['k']==36, 'stage recipe')
    return reg


def review(stage, suite, smoke, reviewer, output, independent=None):
    reg = validate(stage)
    need(reviewer and reviewer.lower() not in ('builder', 'self'), 'independent reviewer identity required')
    receipts = []
    for name, path in [('suite', suite), ('real_cpu_smoke', smoke)]:
        r = read(path)
        need(r['status']=='PASS' and r['kind']==name and r['registration_sha256']==sha(Path(stage)/'registration.json'), 'review receipt mismatch')
        need(r['sources']==reg['sources'], 'smoke source mismatch')
        receipts.append(dict(kind=name, sha256=sha(path)))
    need(independent is not None, 'independent review artifact required')
    report = read(independent)
    need(report.get('status')=='PASS' and report.get('reviewer')==reviewer
         and report.get('scope')=='independent_source_review_only', 'independent review HOLD')
    need(all(report.get('sources',{}).get(n)==reg['sources'][n] for n in OWN_SOURCES[:2]),
         'independent review source drift')
    need(report.get('receipts')=={x['kind']:x['sha256'] for x in receipts}, 'independent review receipt drift')
    result = dict(status='PASS', independent=True, reviewer=reviewer,
                  registration_sha256=sha(Path(stage)/'registration.json'), receipts=receipts,
                  independent_artifact_sha256=sha(independent), independent_artifact=report)
    dump(output, result)
    return result


def suite(stage, output):
    """The receipt exists only after the actual fixed-lane test command succeeds."""
    import subprocess
    import sys
    reg=validate(stage)
    names=['pl15_release_v110','pl_production_release_v1','pl_production_launch_v1',
        'pl_production_watchdog_v1','pl_production_train_v1','vnext_r7_pl_v1',
        'vnext_r7_pl_train_v110','vnext_r7_pl_targets_v110','vnext_r7_pl_go_v111',
        'exp169_release_c85r7_pl15_v140','exp169_release_c85r7_pl15_v150',
        'exp169_release_c85r7codec_deadline_v122','release_deadline_v1','hb116_vnext_release']
    paths=['tests/test_'+n+'.py' for n in names]
    command=[sys.executable,'-B','-m','pytest','-q',*paths,'--basetemp=.local/pytest-tmp/pl15-release']
    dump(output,dict(status='INCOMPLETE',kind='suite'))
    result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log=Path(output).with_suffix('.log');write(log,result.stdout.encode())
    need(result.returncode==0,'test suite failed; '+str(log))
    need(validate(stage)==reg,'source changed during suite')
    receipt=dict(status='PASS',kind='suite',registration_sha256=sha(Path(stage)/'registration.json'),
        sources=reg['sources'],tests={p:sha(ROOT/p) for p in paths},command=command,
        log_sha256=sha(log),summary=result.stdout.strip().splitlines()[-1])
    dump(output,receipt);return dict(status='PASS',summary=receipt['summary'],receipt=str(output))


def prepare(stage=STAGE, binding=None, review_path=None, output=WORK/'dataset', smoke=False):
    stage, output = Path(stage), Path(output)
    reg = validate(stage)
    if not smoke:
        if output.exists():
            need(not any(p.is_dir() for p in output.iterdir()), 'upload directory must be flat')
        need(binding is not None and review_path is not None, 'binding and independent review required')
        reviewed = read(review_path)
        need(reviewed.get('status')=='PASS' and reviewed.get('independent') is True and reviewed.get('reviewer')
             and reviewed['registration_sha256']==sha(stage/'registration.json')
             and {x['kind'] for x in reviewed['receipts']}=={'suite','real_cpu_smoke'}, 'release review HOLD')
        report=reviewed.get('independent_artifact',{})
        need(report.get('status')=='PASS' and report.get('reviewer')==reviewed['reviewer']
             and report.get('receipts')=={x['kind']:x['sha256'] for x in reviewed['receipts']}
             and all(report.get('sources',{}).get(n)==reg['sources'][n] for n in OWN_SOURCES[:2]),
             'independent artifact HOLD')
        from tools import pl_production_release_v1 as old
        b = read(binding)
        student = old.pl.registration(old.OUT/'student'/b['arm'])
        members = old.production_a7.adapter.validate_bindings(student, b)
        science = dict(arm=b['arm'], registration_sha256=old.pl.digest(student), binding_sha256=sha(binding),
                       go_sha256=b['go']['sha256'], teacher_sha256=b['teacher']['sha256'])
    else:
        members = [dict(m, path=str(SMOKE)) for m in reg['members']]
        science = dict(arm='CONTROL_SMOKE_ONLY', registration_sha256='diagnostic-only')
        reviewed = None
    files = []
    for asset in reg['files']:
        # Flat opaque blobs avoid Kaggle directory skipping/archive expansion.
        name=asset['path'].replace('.zip','.blob')
        copy(check(stage, asset), output/name)
        files.append(pin(output, name))
    entries = []
    for member in members:
        name = 'checkpoint-'+('control-smoke.pt' if smoke else member['id']+'.pt')
        if not any(x['path']==name for x in files):
            copy(member['path'], output/name)
            files.append(pin(output, name))
        entry = dict(next(m for m in reg['members'] if m['id']==member['id']), **pin(output, name))
        if not smoke:
            name = 'receipt-'+member['id']+'.json'
            copy(member['training_receipt']['path'], output/name)
            files.append(pin(output, name))
            entry['training_receipt'] = pin(output, name)
            # The original receipt is evidence only. Its local GO paths are never followed at runtime.
        entries.append(entry)
    result = dict(version=VERSION, diagnostic_only=smoke, registration_sha256=sha(stage/'registration.json'),
        science=science, members=entries, files=files, sources=reg['sources'], review=reviewed,
        recipe={k:reg[k] for k in ('k','resolution','tta','variants','compute_seconds','publication_reserve_seconds',
                                 'base_projection_seconds','hidden_studies')}, exposure='EXPOSED_EXPLORATORY', clean_oof=False)
    dump(output/'manifest.json', result)
    write(output/'manifest.sha256', (sha(output/'manifest.json')+'\n').encode())
    dump(output/'dataset-metadata.json', dict(title='C85 R7 PL15 seed42 release v1.1.1', id=SLUG,
         licenses=[{'name':'other'}], description='Private model release. EXPOSED_EXPLORATORY; clean_oof=false.'))
    if not smoke:
        need({p.name for p in output.iterdir()}=={x['path'] for x in files}
             | {'manifest.json','manifest.sha256','dataset-metadata.json'}, 'unexpected upload file')
    verify(output, sha(output/'manifest.json'), allow_smoke=smoke)
    return dict(status='SMOKE_ONLY' if smoke else 'PREPARED', manifest_sha256=sha(output/'manifest.json'), members=5)


def verify(dataset, expected, allow_smoke=False, checkpoints=True):
    dataset = Path(dataset)
    need(expected and sha(dataset/'manifest.json')==expected, 'notebook-pinned manifest differs')
    m = read(dataset/'manifest.json')
    need(m['version']==VERSION and (allow_smoke or not m['diagnostic_only']), 'diagnostic package cannot release')
    for asset in m['files']:
        if checkpoints or not asset['path'].startswith('checkpoint-'):
            check(dataset, asset)
    need([e['id'] for e in m['members']]==['PL-S-s42-f'+str(i) for i in range(5)], 'fivefold order')
    need(m['recipe']['k']==36 and m['recipe']['compute_seconds']==27000 and m['clean_oof'] is False, 'runtime recipe')
    inventory = {x['path']:x for x in m['files']}
    for entry in m['members']:
        need(inventory[entry['path']]=={k:entry[k] for k in ('path','bytes','sha256')}, 'member/inventory mismatch')
        if not m['diagnostic_only']:
            need(inventory[entry['training_receipt']['path']]==entry['training_receipt'], 'receipt inventory')
            r = read(check(dataset, entry['training_receipt']))
            need(r['status']=='COMPLETE' and r['checkpoint_sha256']==entry['sha256'] and r['fold']==entry['fold']
                 and r['seed']==42 and r['registration_sha256']==m['science']['registration_sha256']
                 and r['arm']==m['science']['arm'] and r['teacher_sha256']==m['science']['teacher_sha256']
                 and r['clean_oof'] is False and r['exposure']=='EXPOSED_EXPLORATORY', 'portable science binding')
    return m


def notebook(dataset, output, mode, variant='a', qualification=None):
    need(mode in ('visible','n32','submit') and variant in ('a','b'), 'notebook mode')
    need(variant=='a' or mode=='visible', 'b is visible-only')
    dataset, output = Path(dataset), Path(output)
    expected = sha(dataset/'manifest.json')
    m = verify(dataset, expected)
    q = read(qualification) if qualification else None
    if mode=='submit':
        from tools.pl15_runtime_v110 import admit
        admit(m, expected, q)
    runtime = next(x for x in m['files'] if x['path']=='runtime.blob')
    # Bootstrap verifies code bytes BEFORE importing any dataset code.
    code = '\n'.join([
        'import os, time',
        "assert 'KNEE_RELEASE_START_MONOTONIC_V1' not in os.environ, 'fresh kernel required'",
        "os.environ['KNEE_RELEASE_START_MONOTONIC_V1']=repr(time.monotonic())",
        "os.environ['KNEE_RELEASE_DEADLINE_SECONDS_V1']='27000'",
        'import hashlib, json, sys, zipfile', 'from pathlib import Path',
        f"dataset=Path('/kaggle/input/{SLUG.split('/')[1]}')", f'expected={expected!r}',
        "assert hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()==expected",
        f"assert hashlib.sha256((dataset/'runtime.blob').read_bytes()).hexdigest()=={runtime['sha256']!r}",
        "root=Path('/kaggle/working/pl15-code'); root.mkdir(exist_ok=False)",
        "with zipfile.ZipFile(dataset/'runtime.blob') as z:",
        "    for n in z.namelist():", "        p=(root/n).resolve(); assert p.is_relative_to(root) and ':' not in n and '\\\\' not in n",
        "    z.extractall(root)", "sys.path[:0]=[str(root),str(root/'tools'),str(root/'legacy')]",
        'from tools.pl15_runtime_v110 import run',
        f"run(dataset,expected,Path('/kaggle/working'),mode={mode!r},variant={variant!r},qualification={q!r})", ''])
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
              cells=[dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=code.splitlines(True))])
    dump(output/'release.ipynb', nb)
    meta = read(ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.2.2/kernel/kernel-metadata.json')
    meta.update(id=f'taeyangg4/knee-c85r7-pl15-v111-{variant}-{mode}', title=f'PL15 v111 {variant} {mode}', code_file='release.ipynb')
    meta['dataset_sources']=[*meta['dataset_sources'], SLUG]
    dump(output/'kernel-metadata.json', meta)
    return dict(status='PREPARED_KERNEL', mode=mode, variant=variant, manifest_sha256=expected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['register','validate','prepare','verify','review','kernel','admit','suite'])
    p.add_argument('--stage',type=Path,default=STAGE); p.add_argument('--dataset',type=Path,default=WORK/'dataset')
    p.add_argument('--bindings',type=Path); p.add_argument('--review',type=Path); p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--suite',type=Path); p.add_argument('--smoke-receipt',type=Path); p.add_argument('--reviewer')
    p.add_argument('--independent-review',type=Path)
    p.add_argument('--mode',choices=['visible','n32','submit'],default='visible'); p.add_argument('--variant',choices=['a','b'],default='a')
    p.add_argument('--qualification',type=Path)
    a=p.parse_args()
    if a.command=='register': result=register(a.stage)
    elif a.command=='suite': result=suite(a.stage,a.output)
    elif a.command=='validate': result=dict(status='PASS',members=len(validate(a.stage)['members']))
    elif a.command=='review': result=review(a.stage,a.suite,a.smoke_receipt,a.reviewer,a.output,a.independent_review)
    elif a.command=='prepare':
        if a.dry_run:
            validate(a.stage)
            result=dict(status='HOLD', reasons=['completed fivefold binding and independent review required'], account_actions=0)
        else: result=prepare(a.stage,a.bindings,a.review,a.output or a.dataset,a.smoke)
    elif a.command=='verify':
        m=verify(a.dataset,sha(a.dataset/'manifest.json'),allow_smoke=a.smoke)
        result=dict(status='PASS',diagnostic_only=m['diagnostic_only'],members=len(m['members']))
    elif a.command=='kernel': result=notebook(a.dataset,a.output,a.mode,a.variant,a.qualification)
    else:
        from tools.pl15_runtime_v110 import admit
        if a.dry_run: result=dict(status='HOLD',reasons=['B CUDA, visible, N32 and measured runtime evidence required'])
        else: result=admit(read(a.dataset/'manifest.json'),sha(a.dataset/'manifest.json'),read(a.qualification))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
