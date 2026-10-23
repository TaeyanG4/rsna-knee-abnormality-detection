"""H-B-174 builder v1.0.0: one R224 FULL-flip R6 bundle substitution.

Changelog: new versioned stage; parent backed up under .local/backups/hb174.
No original stage, source module, launch cell or gate receipt is modified.
"""
from __future__ import annotations
import argparse
import ast
import base64
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_stackq_v1125 as parent_builder
from tools import hb145_output_io as output
old, fast, gate = parent_builder.old, parent_builder.fast, parent_builder.gate
VERSION = '1.1.30-r224r6fq'
SLUG = 'taeyangg4/knee-lb-r224r6fq-v1'
PRIVATE = 'taeyangg4/knee-r6-fe055406-v1'
PUBLIC = 'rostigor/rsna-knee-r6-fe055406'
BASE = parent_builder.BASE
OUT = parent_builder.OUT.parent / ('v'+VERSION)
WORK = ROOT/'.local/scratch/astra-b1/hb174'
BUNDLE = ROOT/'storage/external/heliosli-r6-fe055406-v1'
MANIFEST_SHA = '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648'
PUBLIC_NOTEBOOK = ROOT/'.local/scratch/slot-b/ext-1016/heliosli9576/this-is-new-version-0-9576-in-one-click.ipynb'
PORT, CHAIN, WRAPPER = 'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py'
need, sha = old.need, old.digest

# Bundle lookup only. It does not catch an invalid preferred bundle and fall back.
# Multiple physical mounts of an alias fail closed; verification follows lookup.
RESOLVER = '''def resolve_bundle(root: str | Path = "/kaggle/input") -> Path:
    root = Path(root)
    for slug in (DATASET_SLUG, PUBLIC_DATASET_SLUG):
        name = slug.split("/")[-1]
        candidates = [root / name, root / "datasets" / slug]
        candidates.extend((root / "datasets" / slug).glob("*"))
        candidates.extend((root / name).glob("*"))
        matches = list(dict.fromkeys(p for p in candidates if (p / "manifest.json").is_file()))
        require(len(matches) <= 1, "ambiguous R6 dataset mount: " + slug)
        if matches:
            return matches[0]
    raise ValueError("R6 dataset mount absent: " + DATASET_SLUG + " or " + PUBLIC_DATASET_SLUG)


'''


def public_sources():
    nb = old.read(PUBLIC_NOTEBOOK)
    sources = []
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            continue
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in n.targets):
                sources.append(ast.literal_eval(n.value))
    need(len(sources) == 1, 'unique public SOURCES required')
    return {k:v.encode() for k,v in sources[0].items()}


def registration():
    return dict(item='H-B-174', version=VERSION, parent=str(BASE.relative_to(ROOT)),
        parent_kernel_sha256=parent_builder.PARENT_KERNEL, parent_receipt_sha256=parent_builder.PARENT_RECEIPT,
        parent_hidden_submission=57224580, parent_public=.957, keep_score_min=.956,
        rule='keep iff user-provided candidate score >= 0.957 - 0.001 and reviewed runtime <= 9 hours',
        change='R7 slot bundle replaced by pinned R6; existing FULL flip including MCL plain-only exception retained',
        bundle_id='fe055406fad6a12f', tag='cnx_r6fm_c352', arm='E_lrprior', manifest_sha256=MANIFEST_SHA,
        public_source=PUBLIC, preferred_private_copy=PRIVATE,
        public_notebook_sha256=sha(PUBLIC_NOTEBOOK.read_bytes()),
        timing=dict(tag='HB174_N32', order=['r7-1','r6-1','r6-2','r7-2'], studies=32,
                    baseline_hours=6.7, parent_upper_hours=6.73, limit_hours=9,
                    method='Parent 6.7 h plus mean paired supervisor-wall R6-minus-R7 delta times authenticated R7 hidden/local-plain scale. Gate: 6.73 h plus maximum nonnegative paired wall/GPU delta times that scale; no speedup credit.',
                    assumption='4070-to-Kaggle incremental slot-cost transfer; calibration review PENDING; parent runtime is user evidence, not a rigorous bound'),
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        admission='No gate copy. Independent build review, GPU N32, real-output post-run replay and independent result review required.')


def patched(original):
    files = dict(original)
    s = files[PORT]
    for before,after in [
        ('VERSION = "1.0.0"','VERSION = "1.0.1-r6"'),
        ('BUNDLE_ID = "8c7b666e27dc0e8f"','BUNDLE_ID = "fe055406fad6a12f"'),
        ('DATASET_SLUG = "heliosli/rsna-knee-compact-r7cnxfsss"',
         'DATASET_SLUG = '+json.dumps(PRIVATE)+'\nPUBLIC_DATASET_SLUG = '+json.dumps(PUBLIC)),
        ('MANIFEST_SHA256: str | None = None', 'MANIFEST_SHA256: str | None = '+repr(MANIFEST_SHA)),
        ('def verify_source(',RESOLVER+'def verify_source(')]:
        s = old.replace(s,before,after)
    files[PORT] = s
    for name in (CHAIN, WRAPPER):
        s = files[name]
        before = b'v5q._mount(r7.DATASET_SLUG, "manifest.json")'
        need(s.count(before) == (2 if name == CHAIN else 1), 'bundle call site count')
        s = s.replace(before,b'r7.resolve_bundle()')
        if name == CHAIN:
            s = old.replace(s,sha(original[PORT]),sha(files[PORT]))
        else:
            s = old.replace(s,sha(original[CHAIN]),sha(files[CHAIN]))
            for before,after in [('VERSION = "1.1.24-r224flipq"','VERSION = "'+VERSION+'"'),
                ('RELEASE = "C85R7+codecs R224 FLIP_ALL"','RELEASE = "C85R7+codecs R224 R6 FLIP_ALL"'),
                ('KERNEL_ID = "taeyangg4/knee-lb-r224flipq-v1"','KERNEL_ID = "'+SLUG+'"'),
                ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.24-r224flipq"',
                 'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v'+VERSION+'"')]:
                s = old.replace(s,before,after)
        files[name] = s
    return files


def payload():
    nb,original,old_blob,assignments = parent_builder.parent()
    public = public_sources()
    need(len(public) == 7 and all(original['exp169_r7_src/'+k] == v for k,v in public.items()),
         'public9576 worker differs from parent')
    files = patched(original)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source, zipfile.ZipFile(buf,'w') as target:
        for info in source.infolist():
            target.writestr(info,files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before,after in [('BLOB = '+repr(assignments['BLOB']),'BLOB = '+repr(base64.b64encode(blob).decode())),
        ('PINS = '+repr(assignments['PINS']),'PINS = '+repr({k:sha(v) for k,v in files.items()})),
        (repr(sha(old_blob)),repr(sha(blob)))]:
        code = old.replace(code,before,after)
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    meta = old.read(BASE/'kernel/kernel-metadata.json')
    need(meta['dataset_sources'].count('heliosli/rsna-knee-compact-r7cnxfsss') == 1,'parent attachment')
    meta['dataset_sources'] = [PRIVATE if x=='heliosli/rsna-knee-compact-r7cnxfsss' else x for x in meta['dataset_sources']]
    meta.update(id=SLUG,title=SLUG.split('/')[1])
    return nb,meta,files,blob


def port_module():
    from tools.c85r7_r224flip_diagnostics import import_path
    return import_path('hb174_serving_port',WORK/'runtime'/PORT)


def register():
    old.write(OUT/'registration.json',old.encoded(registration()))
    return dict(status='REGISTERED_BEFORE_EVALUATION',keep_min=.956,timing_tag='HB174_N32')


def prepare():
    need((OUT/'registration.json').read_bytes()==old.encoded(registration()),'register first')
    nb,meta,files,blob = payload()
    original_nb,original,_,_ = parent_builder.parent()
    manifest = dict(changed=[],unchanged=[],added=[],removed=[])
    diff = 'H-B-174: ALL EMBEDDED FILES, NOTEBOOK CELLS AND METADATA\n'
    for k in sorted(files):
        changed = original[k] != files[k]
        manifest['changed' if changed else 'unchanged'].append(dict(name=k,before=sha(original[k]),after=sha(files[k])))
        diff += f'{k}: {sha(original[k])} -> {sha(files[k])}\n'
        diff += parent_builder.rbuild.diff(original[k],files[k],'parent/'+k,'candidate/'+k)
        old.write(WORK/'runtime'/k,files[k])
    for k,a,b in [('release.ipynb',original_nb,nb),('kernel-metadata.json',old.read(BASE/'kernel/kernel-metadata.json'),meta)]:
        diff += parent_builder.rbuild.diff(old.encoded(a),old.encoded(b),'parent/'+k,'candidate/'+k)
        backup = ROOT/'.local/backups/hb174'/k
        if backup.exists():
            need(backup.read_bytes()==(BASE/'kernel'/k).read_bytes(),'backup drift')
        else:
            old.write(backup,(BASE/'kernel'/k).read_bytes())
    for name,data in [('kernel/release.ipynb',old.encoded(nb)),('kernel/kernel-metadata.json',old.encoded(meta)),
        ('full-diff-vs-parent.txt',diff.encode()),('embedded-diff.json',old.encoded(manifest)),
        ('CHANGELOG.md',b'# v1.1.30-r224r6fq\nR6 fe055406 replaces R7 in the FULL-flip R224 parent. Private/public mount aliases and mandatory manifest pin. No worker, FAST, launch, pooling, thread or blend changes. Parent backup: .local/backups/hb174.\n')]:
        old.write(OUT/name,data)
    port=port_module()
    m,identity=port.verify_bundle(BUNDLE,source_dir=WORK/'runtime/exp169_r7_src')
    upload = OUT/'dataset-upload-spec'
    metadata=dict(id=PRIVATE,title=PRIVATE.split('/')[1],licenses=[dict(name='other')],
        description='Private versioned copy of heliosli/rsna-knee-r6-frozen-fe055406 version 1 via rostigor/rsna-knee-r6-fe055406. Original author heliosli. Bundle fe055406fad6a12f; unmodified files.')
    old.write(upload/'dataset-metadata.json',old.encoded(metadata))
    inventory=[dict(name=n,source=str((BUNDLE/n).relative_to(ROOT)),size=(BUNDLE/n).stat().st_size,sha256=identity['file_sha256'][n]) for n in port.ROSTER]
    old.write(upload/'files.json',old.encoded(dict(files=inventory,private=True,upload_by='B only; no upload performed',
        source_urls=['https://www.kaggle.com/datasets/'+PUBLIC,'https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406/versions/1'],
        attachment_fallback='If B chooses public attachment, change ONLY dataset_sources entry to '+PUBLIC+' before independent metadata review and re-pin kernel; do not modify scored/admitted bytes.')))
    rec=dict(status='PREPARED_CPU_ONLY',builder_sha256=sha(Path(__file__).read_bytes()),
        kernel_files=gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files'],archive_sha256=sha(blob),
        embedded={k:sha(v) for k,v in files.items()},changed=[PORT,CHAIN,WRAPPER],unchanged=28,
        qualified_for_submission=False,exposure='EXPOSED',clean_oof=False)
    receipt = OUT/'build-receipt.json'
    if receipt.exists() and receipt.read_bytes() != old.encoded(rec):
        previous = old.read(receipt)
        need(previous['kernel_files'] == rec['kernel_files'] and previous['embedded'] == rec['embedded'],
             'refuse to revise receipt for different candidate bytes')
        old.write(ROOT/'.local/backups/hb174'/('build-receipt-'+sha(receipt.read_bytes())+'.json'), receipt.read_bytes())
        output.atomic_json(receipt, rec)
    else:
        old.write(receipt,old.encoded(rec))
    return validate()


def validate():
    nb,meta,files,blob=payload()
    need(old.read(OUT/'registration.json')==registration(),'registration drift')
    need(old.read(OUT/'kernel/release.ipynb')==nb and old.read(OUT/'kernel/kernel-metadata.json')==meta,'stage diff outside allowlist')
    actual=old.unpack(fast.restored(nb))[0]
    original=parent_builder.parent()[1]
    need(actual==files and set(files)==set(original),'archive inventory drift')
    need({k for k in files if original[k]!=files[k]}=={PORT,CHAIN,WRAPPER},'embedded change scope')
    for name,data in files.items():
        if name.endswith('.py'):
            compile(data,name,'exec')
    rec=old.read(OUT/'build-receipt.json')
    need(rec['builder_sha256']==sha(Path(__file__).read_bytes()),'builder drift')
    need(rec['embedded']=={k:sha(v) for k,v in files.items()} and rec['archive_sha256']==sha(blob),'receipt drift')
    kfiles=gate.kernel(OUT,gate.Inputs(ROOT))[2]['kernel_files']
    need(rec['kernel_files']==kfiles,'kernel receipt drift')
    return dict(status='PASS_CPU_VALIDATE',kernel_sha256=gate.sha(gate.encoded(kfiles)),identical=28,changed=3)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['register','prepare','validate'])
    args=parser.parse_args()
    print(json.dumps(globals()[args.action](),indent=2))


if __name__=='__main__':
    main()
