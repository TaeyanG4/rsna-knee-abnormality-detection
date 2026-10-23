"""H198 C4 builder v1.0.0: one immutable H189 derivative; CPU only.

Changelog: authentic original five C4 members, original mixed renderer, explicit
full-visible policy. No launch, submission, source-stage mutation or forwards.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import hb145_output_io as output_io
VERSION = '1.1.47-r6w958c4'
KEY = 'R6W958-C4-V1'
SLUG = 'taeyangg4/knee-lb-r6w958c4-v1'
BASE = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q'
OUT = BASE.parent/('v'+VERSION)
SCRATCH = ROOT/'.local/scratch/slot-b/batch5-1018/c4-bridge'
REVIEW = SCRATCH.parent/'C4-DESIGN-REVIEW.md'
REVIEW_SHA = '6047624c41205bbbfeae31550b3a87c056d367fdbfa1f025ab1ab5582073b3be'
PARENT_SHA = 'fcb2df01b47d5cc9b83c8a11e35ceca10d19ba5884e2b8fd40aee13867f3e8c9'
META_SHA = 'f794876c887318a85887458dee54c0dffa1ca89edf5823db7c1242f17c15dbf9'
Q_SHA = '39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc'
C4_DIR = ROOT/'artifacts/releases/EXP169_V4_A4_B4_s43_stage_v413/c4'
ARCHIVE = C4_DIR/'dataset/exp169-c4-v4.zip.bin'
ARCHIVE_SHA = 'df11aa92535c9487ce6852b3ac159a162a6fb8cea989ac5a625d35faff4dd634'
DATASET = 'taeyangg4/knee-exp169-c4-v4-assets'
GATE = ROOT/'.local/state/gates/knee-lb-r6w958q-v1.json'
BUILDER = '/root/candidate_scout'


def need(ok, msg):
    if not ok: raise ValueError(msg)


def digest(b): return hashlib.sha256(b).hexdigest()
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def encoded(v): return (json.dumps(v, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()


def write(p, b):
    p = Path(p); b = b.encode() if isinstance(b, str) else b
    if p.exists():
        need(p.read_bytes() == b, 'immutable output drift: '+rel(p)); return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    output_io.retry(lambda: tmp.write_bytes(b))
    output_io.replace(tmp, p)


def replace(data, old, new):
    if isinstance(data, bytes): return replace(data.decode(), old, new).encode()
    need(data.count(old) == 1, 'exact patch anchor: '+old[:90])
    return data.replace(old, new)


def unpack(nb):
    tree = ast.parse(''.join(nb['cells'][2]['source']))
    assignments = {x.targets[0].id: ast.literal_eval(x.value) for x in ast.walk(tree)
                   if isinstance(x,ast.Assign) and len(x.targets)==1 and isinstance(x.targets[0],ast.Name)
                   and x.targets[0].id in ('BLOB','PINS')}
    blob = base64.b64decode(assignments['BLOB'])
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) == 31 and all('..' not in Path(n).parts and not n.startswith('/') for n in names), 'member set')
        files = {n:z.read(n) for n in names}
    need(assignments['PINS'] == {n:digest(v) for n,v in files.items()}, 'embedded pin drift')
    return files, blob, assignments


def parent(replay=False):
    need(sha(BASE/'kernel/release.ipynb') == PARENT_SHA and sha(BASE/'kernel/kernel-metadata.json') == META_SHA, 'H189 bytes differ')
    need(sha(REVIEW) == REVIEW_SHA, 'design review changed')
    receipt = read(str(REVIEW)+'.receipt.json')
    need(receipt['state']=='SUCCESS' and receipt['agent_id'] != BUILDER and receipt['report_sha256']==REVIEW_SHA, 'design review identity')
    if replay:
        from tools import strict_variant_gate_r224_v145 as gate
        rec = gate.verify_receipt(GATE, gate.g.Inputs(ROOT))
        need(rec['kernel_id']=='taeyangg4/knee-lb-r6w958q-v1', 'parent gate slug')
    nb = read(BASE/'kernel/release.ipynb'); files,blob,assignments = unpack(nb)
    need(digest(files['q-a5q-manifest.json'])==Q_SHA, 'parent Q pin')
    return nb,files,blob,assignments


def archive_check():
    need(ARCHIVE.stat().st_size==560952802 and sha(ARCHIVE)==ARCHIVE_SHA, 'C4 archive pin')
    cm = read(C4_DIR/'release-manifest.json')
    need(cm['schema']=='exp169-c4-v4' and cm['crop_policy']=='mixed' and cm['k_eval']==36 and cm['tta']=='plane-aware-coronal-axial-flip', 'C4 schema')
    expected = ['release-manifest.json']+[e['path'] for e in cm['entries']]
    with zipfile.ZipFile(ARCHIVE) as z:
        need(len(z.namelist())==len(expected) and set(z.namelist())==set(expected), 'C4 safe exact archive members')
        need(z.read('release-manifest.json')==(C4_DIR/'release-manifest.json').read_bytes(), 'archive manifest bytes')
        for f,e in enumerate(cm['entries']):
            need((e['family'],e['seed'],e['fold'],e['path'])==('convnext',42,f,f'models/convnext_s42_f{f}.pt'), 'C4 identity')
            source=ROOT/e['source'].replace('\\','/')
            need(source.stat().st_size==e['bytes']==112189626 and sha(source)==e['source_sha256']==e['sha256'], 'original C4 bytes')
            with z.open(e['path']) as stream:
                need(hashlib.file_digest(stream,'sha256').hexdigest()==e['sha256'], 'deployed C4 bytes')
    return cm


def overlay_source(cm_sha):
    # Appended to the existing c85 module: no new module/import/worker architecture.
    # The old archive is authenticated before any fresh working-copy overwrite.
    return '''\n\ndef overlay_c4(partner):
    """Authenticated five-member and two-source overlay in fresh writable work."""
    import time
    partner = Path(partner)
    original = read_json(partner/'release-manifest.json')
    old_sha = OLD_Q_LITERAL
    p.check_pin(partner/'release-manifest.json', old_sha)
    a172.base.verify_assets(partner, original)
    archive = v5q._mount(C4_DATASET_LITERAL, 'exp169-c4-v4.zip.bin')/'exp169-c4-v4.zip.bin'
    p.check_pin(archive, C4_ARCHIVE_LITERAL)
    runtime = Path(__file__).resolve().parent
    candidate = manifest(runtime/'q-a5q-manifest.json')
    need(candidate['entries'][:15] == original['entries'][:15], 'retained15 drift')
    def commit(path, data):
        tmp = path.with_suffix(path.suffix+'.c4-tmp')
        for attempt in range(40):
            try:
                tmp.write_bytes(data)
                os.replace(tmp, path)
                return
            except OSError as exc:
                if not isinstance(exc, PermissionError) and getattr(exc, 'winerror', None) not in (5,32): raise
                if attempt == 39: raise
                time.sleep(min(.05*(attempt+1),1.0))
    with zipfile.ZipFile(archive) as z:
        expected = ['release-manifest.json']+[e['path'] for e in candidate['entries'][15:]]
        need(len(z.namelist()) == 6 and set(z.namelist()) == set(expected), 'C4 archive members')
        raw = z.read('release-manifest.json')
        need(hashlib.sha256(raw).hexdigest() == C4_MANIFEST_LITERAL, 'C4 manifest hash')
        cm = json.loads(raw)
        need(cm['entries'] == candidate['entries'][15:] and cm['crop_policy']=='mixed' and cm['k_eval']==36 and cm['tta']=='plane-aware-coronal-axial-flip', 'C4 membership/policy')
        for entry in cm['entries']:
            data = z.read(entry['path'])
            need(len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']==entry['source_sha256'], 'C4 model bytes')
            commit(partner/entry['path'], data)
    for name in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
        data = (runtime/name).read_bytes()
        pin = candidate['files']['code/'+name]
        need(len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256'], 'C4 source bytes')
        commit(partner/'code'/name, data)
    commit(partner/'release-manifest.json', (runtime/'q-a5q-manifest.json').read_bytes())
    a172.verify_assets(partner, candidate)
'''.replace('OLD_Q_LITERAL',repr(Q_SHA[:32])+'+'+repr(Q_SHA[32:])).replace('C4_DATASET_LITERAL',repr(DATASET)).replace('C4_ARCHIVE_LITERAL',repr(ARCHIVE_SHA)).replace('C4_MANIFEST_LITERAL',repr(cm_sha))


def patched(original, cm):
    files = dict(original)
    name='exp169_partner_bundle_release.py'
    files[name]=replace(files[name], '                center3 = center3_windows_from(volume, centers, 224)\n',
        '                center3 = center3_windows_from(volume, centers, 224)\n'
        '                c4 = center3.copy()\n'
        '                odd = list(range(1, len(centers), 2))\n'
        '                c4[odd] = center3_windows_from(volume[:, 48:288, 48:288], [centers[j] for j in odd], 224)\n')
    files[name]=replace(files[name], "normalize_input(center3, 'imagenet'),", "normalize_input(c4, 'imagenet'),")
    files[name]=replace(files[name], "    if family == 'convnext':\n", "    if family == 'convnext' and entry.get('crop_policy') == 'mixed':\n        need(ck.get('crop_policy', 'mixed') == 'mixed', 'C4 crop metadata differs')\n    if family == 'convnext':\n")
    files['a172-cell.py.txt']=replace(files['a172-cell.py.txt'], "        sys.path.insert(0, str(_partner/'code'))", "        _c85.overlay_c4(_partner)\n        sys.path.insert(0, str(_partner/'code'))")
    files['exp169_release_c85.py'] += overlay_source(sha(C4_DIR/'release-manifest.json')).encode()
    wrapper='exp169_release_c85r7codec.py'
    for old,new in [('VERSION = "1.1.39-r6w958q"','VERSION = "'+VERSION+'"'),('taeyangg4/knee-lb-r6w958q-v1',SLUG),('v1.1.39-r6w958q','v'+VERSION)]:
        files[wrapper]=replace(files[wrapper],old,new)
    initial=dict(files)
    # Topological hash closure. Original SHA literals are replaced only where they
    # are runtime dependencies; the original extraction-manifest check is retained.
    for _ in range(20):
        mapping={digest(v):digest(files[n]) for n,v in original.items() if v!=files[n]}
        updated={}
        for n,b in initial.items():
            if n=='q-a5q-manifest.json': continue
            for before,after in mapping.items():
                if n=='a172-cell.py.txt' and before==Q_SHA: continue
                b=b.replace(before.encode(),after.encode())
            updated[n]=b
        m=json.loads(original['q-a5q-manifest.json'])
        m['entries'][15:]=copy.deepcopy(cm['entries'])
        for e in cm['entries']: m['files'][e['path']]={k:e[k] for k in ('bytes','sha256')}
        for n in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
            m['files']['code/'+n]={'bytes':len(updated[n]),'sha256':digest(updated[n])}
            m['source_code'][n]=digest(updated[n])
        m['c4_manifest_sha256']=sha(C4_DIR/'release-manifest.json')
        updated['q-a5q-manifest.json']=encoded(m)
        if updated==files: break
        files=updated
    else: raise ValueError('hash dependency cycle')
    need(json.loads(files['q-a5q-manifest.json'])['entries'][:15]==json.loads(original['q-a5q-manifest.json'])['entries'][:15], 'retained15')
    for n,b in files.items():
        if n.endswith('.py'): compile(b,n,'exec')
    return files


def payload():
    nb,original,old_blob,a=parent(); cm=read(C4_DIR/'release-manifest.json')
    files=patched(original,cm); buf=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as src,zipfile.ZipFile(buf,'w') as dst:
        for info in src.infolist(): dst.writestr(info,files[info.filename])
    blob=buf.getvalue(); source=''.join(nb['cells'][2]['source'])
    source=replace(source,repr(a['BLOB']),repr(base64.b64encode(blob).decode()))
    source=replace(source,'PINS = '+repr(a['PINS']),'PINS = '+repr({n:digest(v) for n,v in files.items()}))
    source=replace(source,repr(digest(old_blob)),repr(digest(blob)))
    nb['cells'][2]['source']=source.splitlines(True)
    need('FAST = True' in ''.join(nb['cells'][0]['source']), 'parent visible shortcut drift')
    nb['cells'][0]['source']=['FAST = False\n']
    meta=read(BASE/'kernel/kernel-metadata.json'); meta.update(id=SLUG,title=SLUG.split('/')[1])
    need(DATASET not in meta['dataset_sources'] and len(meta['dataset_sources'])==6,'parent datasets')
    meta['dataset_sources'].append(DATASET)
    return nb,meta,files,blob


def prepare():
    parent(replay=True); cm=archive_check(); nb,meta,files,blob=payload()
    original=parent()[1]
    inventory={n:dict(parent=digest(original[n]),candidate=digest(b),changed=b!=original[n]) for n,b in files.items()}
    diff=''.join(''.join(difflib.unified_diff(original[n].decode().splitlines(True),files[n].decode().splitlines(True),fromfile='H189/'+n,tofile='H198/'+n)) for n in files)
    for name in ('kernel/release.ipynb','kernel/kernel-metadata.json'):
        write(SCRATCH/'parent-backup'/name,(BASE/name).read_bytes())
    write(SCRATCH/'parent-backup/admission.json',GATE.read_bytes())
    for name,data in [('kernel/release.ipynb',encoded(nb)),('kernel/kernel-metadata.json',encoded(meta)),('embedded-diff.json',encoded(inventory)),('full-embedded-diff.txt',diff),('SPEC.md',(SCRATCH/'SPEC.md').read_bytes()),('CHANGELOG.md','# '+VERSION+'\nOne H189 C4 mixed-crop child. Only five C4 originals, mixed image field, explicit FAST=False, asset overlay and transitive pins. New path; all release admissions HOLD.\n')]: write(OUT/name,data)
    for n,b in files.items(): write(SCRATCH/'candidate-runtime'/n,b)
    record=dict(schema='h198-build/v1',candidate=KEY,version=VERSION,kernel_id=SLUG,builder=BUILDER,design_review_sha256=REVIEW_SHA,spec_sha256=sha(SCRATCH/'SPEC.md'),builder_sha256=sha(__file__),parent_receipt_sha256=sha(GATE),kernel_sha256=sha(OUT/'kernel/release.ipynb'),metadata_sha256=sha(OUT/'kernel/kernel-metadata.json'),archive_sha256=digest(blob),c4_archive_sha256=ARCHIVE_SHA,changed=[n for n,v in inventory.items() if v['changed']],unchanged=sum(not v['changed'] for v in inventory.values()),all_embedded=inventory,qualified_for_submission=False,status='PREPARED_CPU_ONLY')
    write(OUT/'build-receipt.json',encoded(record)); return record


def validate():
    nb,meta,files,blob=payload()
    need(read(OUT/'kernel/release.ipynb')==nb and read(OUT/'kernel/kernel-metadata.json')==meta,'exact stage diff')
    need(unpack(nb)[0]==files,'embedded candidate differs')
    rec=read(OUT/'build-receipt.json')
    need(rec['kernel_sha256']==sha(OUT/'kernel/release.ipynb') and rec['builder_sha256']==sha(__file__),'build receipt drift')
    return dict(status='BUILD_BYTES_VALID',kernel_sha256=rec['kernel_sha256'],changed=rec['changed'],unchanged=rec['unchanged'])


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','validate')); a=p.parse_args()
    try:
        r=prepare() if a.command=='prepare' else validate()
        print(json.dumps({k:r[k] for k in ('status','kernel_sha256','changed','unchanged')},sort_keys=True)); return 0
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps(dict(status='HOLD',reason=str(exc)))); return 2


if __name__=='__main__': raise SystemExit(main())
