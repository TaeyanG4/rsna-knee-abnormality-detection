"""INF2 release builder v1.0.0: one H189 derivative; static CPU packaging only.

Changelog: exact HB105 Small10, original spatial MIL/mixed renderer, fixed Q2
eight-column substitution, ordinary visible execution. No admission framework.
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
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import hb145_output_io as output_io
VERSION = '1.1.48-r6w958inf2'
KEY = 'R6W958-INF2-V1'
SLUG = 'taeyangg4/knee-lb-r6w958inf2-v1'
DATASET = 'taeyangg4/knee-inf2-small10-v1-assets'
BASE = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q'
OUT = BASE.parent/('v'+VERSION)
SCRATCH = ROOT/'.local/scratch/slot-b/batch5-1018/inf2-release-lean'
BRIDGE = SCRATCH.parent/'inf2-bridge'
HELPER = ROOT/'tools/c85r7_inf2_small_v1.py'
PARENT_SHA = 'fcb2df01b47d5cc9b83c8a11e35ceca10d19ba5884e2b8fd40aee13867f3e8c9'
META_SHA = 'f794876c887318a85887458dee54c0dffa1ca89edf5823db7c1242f17c15dbf9'
Q_SHA = '39ff0cb9f4e0b4fb0967a00c8c0897a7de90a697242d3042fe3e51b5f2cf36bc'
BRIDGE_SHA = '624bab62c7ae6b8298adfcb0587dd95b2070b9f489a67c6296ec4f4962b26f79'
REG_SHA = 'e22cb92a96e3b3eba661f4a29d57b27fb5466fc94eba08e5015c1bb67ff33342'
REVIEW_SHA = '1355c0a1b5b9f39cca5f13117be6fda094c4388a02da6ec8f90fd8cd5553e8ef'
ARCHIVE_NAME = 'inf2-small10.zip.bin'
ASSET_MANIFEST = 'inf2-small-manifest.json'
BUILDER = '/root/candidate_scout'


def need(ok, message):
    if not ok: raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest()
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def encoded(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def write(path, data):
    path = Path(path); data = data.encode() if isinstance(data, str) else data
    if path.exists():
        need(path.read_bytes() == data, 'immutable output differs: '+rel(path)); return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    output_io.retry(lambda: tmp.write_bytes(data))
    output_io.replace(tmp, path)


def replace(data, before, after):
    if isinstance(data, bytes): return replace(data.decode(), before, after).encode()
    need(data.count(before) == 1, 'exact patch anchor: '+before[:100])
    return data.replace(before, after)


def unpack(nb):
    tree = ast.parse(''.join(nb['cells'][2]['source']))
    assignments = {x.targets[0].id: ast.literal_eval(x.value) for x in ast.walk(tree)
                   if isinstance(x, ast.Assign) and len(x.targets)==1 and isinstance(x.targets[0], ast.Name)
                   and x.targets[0].id in ('BLOB', 'PINS')}
    blob = base64.b64decode(assignments['BLOB'])
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        need(len(names)==len(set(names))==31 and all('..' not in Path(n).parts and not n.startswith('/') for n in names), 'embedded members')
        files = {name: archive.read(name) for name in names}
    need(assignments['PINS']=={name:digest(data) for name,data in files.items()}, 'embedded pins')
    return files, blob, assignments


def parent():
    need(sha(BASE/'kernel/release.ipynb')==PARENT_SHA and sha(BASE/'kernel/kernel-metadata.json')==META_SHA, 'H189 bytes differ')
    for name,pin in [('missing-serving-inputs.json',BRIDGE_SHA),('registration.json',REG_SHA),('RESULT-REVIEW.md',REVIEW_SHA)]:
        need(sha(BRIDGE/name)==pin, 'scientific identity changed: '+name)
    nb = read(BASE/'kernel/release.ipynb'); files,blob,assignments = unpack(nb)
    need(digest(files['q-a5q-manifest.json'])==Q_SHA, 'parent Q manifest differs')
    return nb,files,blob,assignments


def asset_manifest():
    inputs = read(BRIDGE/'missing-serving-inputs.json')
    identities = [copy.deepcopy(member['identity']) for member in inputs['small_members']]
    need([(e['family'],e['seed'],e['fold']) for e in identities]==[('S',s,f) for s in (42,43) for f in range(5)], 'Small roster/order')
    need(len({e['sha256'] for e in identities})==10, 'Small duplicate weight')
    entries = []
    for identity in identities:
        entry = dict(identity, family='small', source=identity['path'], source_sha256=identity['sha256'],
                     path='models/inf2_'+identity['id']+'.pt', source_identity=identity)
        need(entry['bytes']==198775478 and entry['arch']=='convnext_small.fb_in22k_ft_in1k', 'Small identity')
        entries.append(entry)
    return dict(schema='inf2-small10-v1', candidate=KEY, entries=entries,
        registration_sha256=REG_SHA, original_inputs_sha256=BRIDGE_SHA,
        checkpoint_bytes=sum(e['bytes'] for e in entries),
        original_source_pins=inputs['sources'],
        upstream_reference_urls=['https://www.kaggle.com/competitions/rsna-knee-abnormality-detection',
            'https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k'],
        checkpoint_origin='Own HB105 training; exact local source checkpoints; no public checkpoint URL asserted.',
        serving=dict(centers=36,resolution=224,input='center3, mixed full/240 crop, bilinear align_corners=False',
                     normalization='ImageNet',pooling='finding_spatial_tokens_v1',
                     tta='normal and coronal/axial horizontal flip; FP16 CUDA logits -> FP32 sigmoid -> FP32 probability mean',
                     aggregation='FP32 seed pair by fold; FP32 five-fold probability mean; global family rank',
                     small_columns=list(range(8)),dino_columns=[8,9,10,11]))


def package_assets():
    manifest = asset_manifest(); folder = OUT/'assets'; archive_path = folder/ARCHIVE_NAME
    raw = encoded(manifest)
    for entry in manifest['entries']:
        source = ROOT/entry['source']
        need(source.stat().st_size==entry['bytes'] and sha(source)==entry['sha256'], 'Small source bytes: '+entry['id'])
    write(OUT/ASSET_MANIFEST, raw)
    write(folder/'dataset-metadata.json', encoded(dict(id=DATASET,title=DATASET.split('/')[1],isPrivate=True,
        licenses=[dict(name='other')],description='Exact ten own HB105 SmallMIL checkpoints, seeds 42/43 folds 0..4. No images, labels or reports. INF2 fixed eight-target substitution.')))
    if not archive_path.exists():
        tmp = folder/(ARCHIVE_NAME+'.tmp')
        with zipfile.ZipFile(tmp, 'x', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(ASSET_MANIFEST, raw)
            for entry in manifest['entries']: archive.write(ROOT/entry['source'], entry['path'])
        output_io.replace(tmp, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        need(archive.namelist()==[ASSET_MANIFEST]+[e['path'] for e in manifest['entries']], 'asset ZIP inventory')
        need(archive.read(ASSET_MANIFEST)==raw, 'asset manifest bytes')
        for entry in manifest['entries']:
            need(archive.getinfo(entry['path']).file_size==entry['bytes'], 'deployed Small size')
            with archive.open(entry['path']) as stream:
                need(hashlib.file_digest(stream,'sha256').hexdigest()==entry['sha256'], 'deployed Small hash')
    return manifest


def overlay_source(manifest_sha, archive_sha):
    return '''\n\ndef overlay_inf2(partner):
    """Authenticate old assets first, then add Small10 in the fresh working copy."""
    import time
    partner = Path(partner)
    original = read_json(partner/'release-manifest.json')
    p.check_pin(partner/'release-manifest.json', OLD_Q_LITERAL)
    a172.base.verify_assets(partner, original)
    archive = v5q._mount(DATASET_LITERAL, ARCHIVE_LITERAL)/ARCHIVE_LITERAL
    p.check_pin(archive, ARCHIVE_SHA_LITERAL)
    runtime = Path(__file__).resolve().parent
    candidate = manifest(runtime/'q-a5q-manifest.json')
    need(candidate['entries'] == original['entries'], 'retained20 drift')
    def commit(path, data):
        tmp = path.with_suffix(path.suffix+'.inf2-tmp')
        for attempt in range(40):
            try:
                tmp.write_bytes(data)
                os.replace(tmp, path)
                return
            except OSError as exc:
                if not isinstance(exc, PermissionError) and getattr(exc,'winerror',None) not in (5,32): raise
                if attempt == 39: raise
                time.sleep(min(.05*(attempt+1),1.0))
    with zipfile.ZipFile(archive) as z:
        expected = [ASSET_MANIFEST_LITERAL]+[e['path'] for e in candidate['inf2_entries']]
        need(len(z.namelist())==11 and set(z.namelist())==set(expected), 'Small10 archive members')
        raw = z.read(ASSET_MANIFEST_LITERAL)
        need(hashlib.sha256(raw).hexdigest()==ASSET_MANIFEST_SHA_LITERAL, 'Small10 manifest hash')
        small = json.loads(raw)
        need(small['entries']==candidate['inf2_entries'], 'Small10 manifest identity')
        for entry in small['entries']:
            data = z.read(entry['path'])
            need(len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']==entry['source_sha256'], 'Small checkpoint bytes')
            commit(partner/entry['path'], data)
    for name in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
        data = (runtime/name).read_bytes()
        pin = candidate['files']['code/'+name]
        need(len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256'], 'INF2 source bytes')
        commit(partner/'code'/name, data)
    commit(partner/'release-manifest.json', (runtime/'q-a5q-manifest.json').read_bytes())
    a172.verify_assets(partner, candidate)
'''.replace('OLD_Q_LITERAL',repr(Q_SHA[:32])+'+'+repr(Q_SHA[32:])).replace('DATASET_LITERAL',repr(DATASET)).replace('ARCHIVE_SHA_LITERAL',repr(archive_sha)).replace('ARCHIVE_LITERAL',repr(ARCHIVE_NAME)).replace('ASSET_MANIFEST_SHA_LITERAL',repr(manifest_sha)).replace('ASSET_MANIFEST_LITERAL',repr(ASSET_MANIFEST))


def patched(original, assets, archive_sha):
    files = dict(original); name = 'exp169_partner_bundle_release.py'
    # The first 20 entries, their loaders and tensors remain unchanged. Small10
    # extends the same scientific loop; the existing queue/thread code is intact.
    files[name] = replace(files[name], "    entries = manifest['entries']\n", "    entries = manifest['entries'] + manifest['inf2_entries']\n")
    files[name] = files[name].replace(b"10 if name == 'l2' else 5", b"10 if name in ('l2', 'small') else 5")
    files[name] = replace(files[name], "\n              for name in ('l2', 'dino', 'convnext')}\n", "\n              for name in ('l2', 'dino', 'convnext', 'small')}\n")
    files[name] = replace(files[name], "set(scores) == {'l2', 'dino', 'convnext'}", "set(scores) == {'l2', 'dino', 'convnext', 'small'}")
    files[name] = replace(files[name], "    bundle = rank(sum(rank(x) for x in family.values()) / 3)",
        "    small = np.asarray(scores['small'])\n"
        "    need(small.dtype == np.float32, 'Small probabilities must remain FP32')\n"
        "    small_rank = rank(((small[:5] + small[5:]) / 2).mean(0))\n"
        "    family['dino'][:, :8] = small_rank[:, :8]\n"
        "    bundle = rank(sum(rank(x) for x in family.values()) / 3)")
    files[name] = replace(files[name], "        ck = torch.load(path, map_location='cpu', weights_only=False)\n",
        "        if entry['family'] == 'small':\n"
        "            models.append(inf2_load_small(path, entry, device))\n"
        "            continue\n"
        "        ck = torch.load(path, map_location='cpu', weights_only=False)\n")
    files[name] = replace(files[name], "                payload = (i, normalise(center3, 'radimagenet'),",
        "                small = inf2_small_windows(volume, centers)\n"
        "                payload = (i, normalise(center3, 'radimagenet'),")
    files[name] = replace(files[name], "normalize_input(triplet, 'imagenet'), flags)", "normalize_input(triplet, 'imagenet'), flags, small)")
    files[name] = replace(files[name], '                i, rad, img, triplet, flags = job', '                i, rad, img, triplet, flags, small = job')
    files[name] = replace(files[name], "x = triplet if family == 'dino' else rad if family == 'l2' else img", "x = small if family == 'small' else triplet if family == 'dino' else rad if family == 'l2' else img")
    files[name] = replace(files[name], "slot = entry['seed'] == 43 and family == 'l2'", "slot = entry['seed'] == 43 and family in ('l2', 'small')")
    files[name] += b'\n\n' + HELPER.read_bytes()
    c = 'exp169_release_c85.py'
    files[c] = replace(files[c], '    return value\n\n\ndef optional_json',
        "    need([(e['family'],e['seed'],e['fold']) for e in value['inf2_entries']] ==\n"
        "         [('small',s,f) for s in (42,43) for f in range(5)], 'INF2 Small roster/order')\n"
        "    return value\n\n\ndef optional_json")
    files[c] = replace(files[c], "('l2', 'dino', 'convnext')}", "('l2', 'dino', 'convnext', 'small')}")
    files[c] = replace(files[c], 'qnames = [str(i) for i in range(20)]', 'qnames = [str(i) for i in range(30)]')
    files[c] = replace(files[c], "('convnext', range(15,20))]}", "('convnext', range(15,20)), ('small', range(20,30))]}")
    files[c] = replace(files[c], "else ('convnext', 15)\n", "else ('convnext', 15) if i < 20 else ('small', 20)\n")
    files[c] = replace(files[c], "10 if name == 'l2' else 5", "10 if name in ('l2', 'small') else 5")
    files[c] += overlay_source(digest(encoded(assets)), archive_sha).encode()
    files['a172-cell.py.txt'] = replace(files['a172-cell.py.txt'], "        sys.path.insert(0, str(_partner/'code'))", "        _c85.overlay_inf2(_partner)\n        sys.path.insert(0, str(_partner/'code'))")
    wrapper = 'exp169_release_c85r7codec.py'
    for before,after in [('VERSION = "1.1.39-r6w958q"','VERSION = "'+VERSION+'"'),('taeyangg4/knee-lb-r6w958q-v1',SLUG),('v1.1.39-r6w958q','v'+VERSION)]:
        files[wrapper] = replace(files[wrapper],before,after)
    files[wrapper] = replace(files[wrapper], '    return receipt\n',
        "    receipt['q_variant'] = dict(id='INF2', small_members=10, small_columns=list(range(8)),\n"
        "        dino_columns=[8,9,10,11], probability_order='FP32 seed-pair then fold mean; whole-cohort ranks',\n"
        "        manifest_sha256=c85.MANIFEST_SHA, failure_policy='existing all-members Q branch failure; no Small omission or substitution')\n"
        '    return receipt\n') if files[wrapper].count(b'    return receipt\n')==1 else replace(files[wrapper], '    receipt["execution_passes"].append(dict(serving=r224.SOURCE, arms=["R224"]))\n',
        '    receipt["execution_passes"].append(dict(serving=r224.SOURCE, arms=["R224"]))\n'
        "    receipt['q_variant'] = dict(id='INF2', small_members=10, small_columns=list(range(8)),\n"
        "        dino_columns=[8,9,10,11], probability_order='FP32 seed-pair then fold mean; whole-cohort ranks',\n"
        "        manifest_sha256=c85.MANIFEST_SHA, failure_policy='existing all-members Q branch failure; no Small omission or substitution')\n")
    initial = dict(files)
    for _ in range(20):
        mapping = {digest(data):digest(files[name]) for name,data in original.items() if data!=files[name]}
        updated = {}
        for name,data in initial.items():
            if name=='q-a5q-manifest.json': continue
            for before,after in mapping.items():
                if name=='a172-cell.py.txt' and before==Q_SHA: continue
                data = data.replace(before.encode(),after.encode())
            updated[name] = data
        manifest = json.loads(original['q-a5q-manifest.json'])
        manifest['inf2_entries'] = copy.deepcopy(assets['entries'])
        manifest['inf2_serving'] = assets['serving']
        manifest['inf2_asset_manifest_sha256'] = digest(encoded(assets))
        for entry in assets['entries']: manifest['files'][entry['path']] = {k:entry[k] for k in ('bytes','sha256')}
        for name in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
            manifest['files']['code/'+name] = dict(bytes=len(updated[name]),sha256=digest(updated[name]))
            manifest['source_code'][name] = digest(updated[name])
        updated['q-a5q-manifest.json'] = encoded(manifest)
        if updated==files: break
        files = updated
    else: raise ValueError('hash dependency cycle')
    need(json.loads(files['q-a5q-manifest.json'])['entries']==json.loads(original['q-a5q-manifest.json'])['entries'], 'retained20')
    for name,data in files.items():
        if name.endswith('.py'): compile(data,name,'exec')
    return files


def payload():
    nb,original,old_blob,assignments = parent()
    assets = asset_manifest(); archive = OUT/'assets'/ARCHIVE_NAME
    files = patched(original,assets,sha(archive)); buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source, zipfile.ZipFile(buffer,'w') as dest:
        for info in source.infolist(): dest.writestr(info,files[info.filename])
    blob = buffer.getvalue(); source = ''.join(nb['cells'][2]['source'])
    source = replace(source,repr(assignments['BLOB']),repr(base64.b64encode(blob).decode()))
    source = replace(source,'PINS = '+repr(assignments['PINS']),'PINS = '+repr({name:digest(data) for name,data in files.items()}))
    source = replace(source,repr(digest(old_blob)),repr(digest(blob)))
    nb['cells'][2]['source'] = source.splitlines(True)
    need('FAST = True' in ''.join(nb['cells'][0]['source']), 'parent visible shortcut drift')
    nb['cells'][0]['source'] = ['FAST = False\n']
    meta = read(BASE/'kernel/kernel-metadata.json'); meta.update(id=SLUG,title=SLUG.split('/')[1])
    need(DATASET not in meta['dataset_sources'] and len(meta['dataset_sources'])==6,'parent datasets')
    meta['dataset_sources'].append(DATASET)
    return nb,meta,files,blob


def prepare():
    parent(); assets = package_assets(); nb,meta,files,blob = payload(); original = parent()[1]
    inventory = {name:dict(parent=digest(original[name]),candidate=digest(data),changed=data!=original[name]) for name,data in files.items()}
    diff = ''.join(''.join(difflib.unified_diff(original[name].decode().splitlines(True),data.decode().splitlines(True),fromfile='H189/'+name,tofile='INF2/'+name)) for name,data in files.items())
    for name in ('kernel/release.ipynb','kernel/kernel-metadata.json'): write(SCRATCH/'parent-backup'/name,(BASE/name).read_bytes())
    for name,data in [('kernel/release.ipynb',encoded(nb)),('kernel/kernel-metadata.json',encoded(meta)),
        ('embedded-diff.json',encoded(inventory)),('full-embedded-diff.txt',diff),
        ('CHANGELOG.md','# '+VERSION+'\nOne H189 derivative: exact Small10 spatial MIL and original mixed renderer; fixed eight-target Q2 substitution; original last4 DINO/R50/plainTiny/outer F; FAST=False; private asset addition and hash pins. Existing worker/process/queue/thread/timing code retained.\n')]: write(OUT/name,data)
    for name,data in files.items(): write(SCRATCH/'candidate-runtime'/name,data)
    record = dict(schema='inf2-release-build-v1',candidate=KEY,version=VERSION,kernel_id=SLUG,builder=BUILDER,
        registration_sha256=REG_SHA,scientific_result_review_sha256=REVIEW_SHA,
        source_pins={rel(p):sha(p) for p in (Path(__file__),HELPER)},
        parent_kernel_sha256=PARENT_SHA, kernel_sha256=sha(OUT/'kernel/release.ipynb'),
        metadata_sha256=sha(OUT/'kernel/kernel-metadata.json'),archive_sha256=digest(blob),
        assets=dict(dataset=DATASET,archive=rel(OUT/'assets'/ARCHIVE_NAME),bytes=(OUT/'assets'/ARCHIVE_NAME).stat().st_size,
            sha256=sha(OUT/'assets'/ARCHIVE_NAME),checkpoint_bytes=assets['checkpoint_bytes']),
        changed=[name for name,v in inventory.items() if v['changed']],unchanged=sum(not v['changed'] for v in inventory.values()),
        all_embedded=inventory,status='BUILT_STATIC_CPU_ONLY',qualified_for_submission=False)
    write(OUT/'build-receipt.json',encoded(record)); return record


def validate():
    nb,meta,files,blob = payload()
    need(read(OUT/'kernel/release.ipynb')==nb and read(OUT/'kernel/kernel-metadata.json')==meta,'exact stage bytes')
    need(unpack(nb)[0]==files,'embedded candidate differs')
    receipt = read(OUT/'build-receipt.json')
    need(receipt['kernel_sha256']==sha(OUT/'kernel/release.ipynb'),'kernel receipt differs')
    for name,pin in receipt['source_pins'].items(): need(sha(ROOT/name)==pin,'builder/helper changed')
    return dict(status='BUILD_BYTES_VALID',kernel_sha256=receipt['kernel_sha256'],changed=receipt['changed'],unchanged=receipt['unchanged'])


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','validate')); args=parser.parse_args()
    result = prepare() if args.command=='prepare' else validate()
    print(json.dumps({key:result[key] for key in ('status','kernel_sha256','changed','unchanged')},sort_keys=True))


if __name__=='__main__': main()
