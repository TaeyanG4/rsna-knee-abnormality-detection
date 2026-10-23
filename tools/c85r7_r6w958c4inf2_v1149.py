"""H207 C4+INF2 release builder v1.0.0; one H189 derivative, CPU only.

Reuse the two exact component scientific transforms; one coherent asset overlay.
No model forward, metric evaluation, admission, launch, network or submission.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
from datetime import datetime, timezone, timedelta
import difflib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import c85r7_r6w958inf2_v1148 as small
from tools import c85r7_r6w958c4_v1147 as c4

need, read, sha, digest, encoded, replace, unpack = (
    small.need, small.read, small.sha, small.digest, small.encoded, small.replace, small.unpack)
VERSION = '1.1.49-r6w958c4inf2'
KEY = 'R6W958-C4INF2-V1'
SLUG = 'taeyangg4/knee-lb-r6w958c4inf2-v1'
BASE = small.BASE
OUT = BASE.parent/('v'+VERSION)
PREP = ROOT/'.local/scratch/slot-b/research-resume-1018'
WORK = PREP/'h207-build'
SPEC = PREP/'BUILD-SPEC.md'
SELF = Path(__file__)
TEST = ROOT/'tools/test_c85r7_r6w958c4inf2_v1149.py'
BUILDER = '/root/k1_result_review'
SMALL_SHA = '205b57215b8bc0736410a236f8b305dc2c7aeca2e94cca86dda0af3631689d2d'
C4_SHA = 'd86ca0e93e11773ad6119bb9c84ea632130723d899a5f4ef5d5faa4029fc28d9'
SMALL_ARCHIVE_SHA = 'd001f9fb9d9ac2b93357447f7a7ed52aa93dc72655ddefa97dc2449e958e40c2'
ALLOWED = {'a172-cell.py.txt', 'exp169_partner_bundle_release.py', 'q-a5q-manifest.json',
    'exp169_release_c85.py', 'exp169_release_v2.py', 'exp169_release_v5q.py',
    'exp169_release_p956.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py'}
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def stamp(): return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')
def pin(path): return dict(bytes=Path(path).stat().st_size, sha256=sha(path))


def write_new(path, data):
    """Create-only bounded Windows publication; frozen stages are never rewritten."""
    data = data.encode() if isinstance(data, str) else data
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists(), 'create-only destination exists: '+str(path))
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('xb') as stream:
        stream.write(data); stream.flush()
        import os
        os.fsync(stream.fileno())
    # OUT itself is acquired via mkdir(exist_ok=False); no competing stage writer.
    small.output_io.replace(tmp, path)


def components():
    nb, original, blob, assignments = small.parent()
    need(sha(c4.OUT/'kernel/release.ipynb') == C4_SHA, 'C4 component identity')
    need(sha(small.OUT/'kernel/release.ipynb') == SMALL_SHA, 'INF2 component identity')
    assets = small.asset_manifest()
    need(read(small.OUT/small.ASSET_MANIFEST) == assets, 'original Small manifest')
    cm = read(c4.C4_DIR/'release-manifest.json')
    need((cm['schema'], cm['crop_policy'], cm['k_eval'], cm['tta']) ==
         ('exp169-c4-v4', 'mixed', 36, 'plane-aware-coronal-axial-flip'), 'C4 original policy')
    need([(e['family'], e['seed'], e['fold']) for e in cm['entries']] ==
         [('convnext',42,f) for f in range(5)], 'five exact C4 members')
    sc = small.patched(original, assets, SMALL_ARCHIVE_SHA)
    cc = c4.patched(original, cm)
    need(sc == unpack(read(small.OUT/'kernel/release.ipynb'))[0], 'INF2 source transform drift')
    need(cc == unpack(read(c4.OUT/'kernel/release.ipynb'))[0], 'C4 source transform drift')
    for folder, files in ((small.OUT,sc),(c4.OUT,cc)):
        receipt = read(folder/'build-receipt.json')
        actual = {n:dict(parent=digest(original[n]),candidate=digest(v),changed=v!=original[n])
                  for n,v in files.items()}
        need(actual == receipt['all_embedded'], 'component all-file receipt drift')
    return nb, original, blob, assignments, assets, cm, sc, cc


def overlay_source(assets, cm):
    """Merge the already-authenticated component overlays, committing one manifest."""
    text = small.overlay_source(digest(encoded(assets)), SMALL_ARCHIVE_SHA)
    text = replace(text, 'def overlay_inf2(partner):', 'def overlay_c4inf2(partner):')
    text = replace(text, 'Authenticate old assets first, then add Small10 in the fresh working copy.',
                   'Authenticate original assets once, then replace C4 and add Small10.')
    text = replace(text, "need(candidate['entries'] == original['entries'], 'retained20 drift')",
                   "need(candidate['entries'][:15] == original['entries'][:15], 'retained15 drift')")
    text = text.replace("'.inf2-tmp'", "'.c4inf2-tmp'")
    original_c4 = c4.overlay_source(sha(c4.C4_DIR/'release-manifest.json'))
    block = original_c4[original_c4.index('    with zipfile.ZipFile(archive) as z:'):
                        original_c4.index("    for name in ('exp169_partner_bundle_release.py'")]
    block = block.replace('zipfile.ZipFile(archive)', 'zipfile.ZipFile(c4_archive)')
    mount = ("    c4_archive = v5q._mount("+repr(c4.DATASET)+", 'exp169-c4-v4.zip.bin')/'exp169-c4-v4.zip.bin'\n"
             "    p.check_pin(c4_archive, "+repr(c4.ARCHIVE_SHA)+")\n")
    text = replace(text, '    with zipfile.ZipFile(archive) as z:', mount+block+'    with zipfile.ZipFile(archive) as z:')
    text = text.replace("'INF2 source bytes'", "'C4INF2 source bytes'")
    return text


def patched(original, assets, cm, sc):
    # Start from H189 and its reproducible pure INF2 transformation, not a child
    # stage as parent. Reverse only transitive embedded pins before joint closure.
    reverse = {digest(v):digest(original[n]) for n,v in sc.items() if v != original[n]}
    pattern = re.compile('|'.join(map(re.escape, reverse)))
    files = {n:pattern.sub(lambda m:reverse[m[0]], v.decode()).encode() for n,v in sc.items()}
    q = 'exp169_partner_bundle_release.py'
    files[q] = replace(files[q], '                center3 = center3_windows_from(volume, centers, 224)\n',
        '                center3 = center3_windows_from(volume, centers, 224)\n'
        '                c4 = center3.copy()\n'
        '                odd = list(range(1, len(centers), 2))\n'
        '                c4[odd] = center3_windows_from(volume[:, 48:288, 48:288], [centers[j] for j in odd], 224)\n')
    files[q] = replace(files[q], "normalize_input(center3, 'imagenet'),", "normalize_input(c4, 'imagenet'),")
    files[q] = replace(files[q], "    if family == 'convnext':\n",
        "    if family == 'convnext' and entry.get('crop_policy') == 'mixed':\n"
        "        need(ck.get('crop_policy', 'mixed') == 'mixed', 'C4 crop metadata differs')\n"
        "    if family == 'convnext':\n")
    c = 'exp169_release_c85.py'
    # Drop the component overlay as a complete function, retaining all original
    # INF2 scientific roster/receipt support, and append exactly one joint overlay.
    marker = b'\n\ndef overlay_inf2(partner):'
    need(files[c].count(marker) == 1, 'exact component overlay')
    files[c] = files[c].split(marker)[0] + overlay_source(assets, cm).encode()
    files['a172-cell.py.txt'] = replace(files['a172-cell.py.txt'], '_c85.overlay_inf2(_partner)', '_c85.overlay_c4inf2(_partner)')
    wrapper = 'exp169_release_c85r7codec.py'
    for before,after in [(small.VERSION, VERSION), (small.SLUG, SLUG)]:
        need(before.encode() in files[wrapper], 'joint version metadata anchor')
        files[wrapper] = files[wrapper].replace(before.encode(), after.encode())
    files[wrapper] = replace(files[wrapper], "dict(id='INF2', small_members=10,", "dict(id='C4INF2', c4_members=5, total_q_members=30, small_members=10,")
    files[wrapper] = replace(files[wrapper], 'no Small omission or substitution', 'no Small/C4 omission or substitution')
    initial = dict(files)
    for _ in range(20):
        mapping = {digest(v):digest(files[n]) for n,v in original.items() if v != files[n]}
        updated = {}
        for n,data in initial.items():
            if n == 'q-a5q-manifest.json': continue
            for before,after in mapping.items():
                if n == 'a172-cell.py.txt' and before == small.Q_SHA: continue
                data = data.replace(before.encode(), after.encode())
            updated[n] = data
        manifest = json.loads(original['q-a5q-manifest.json'])
        manifest['entries'][15:] = copy.deepcopy(cm['entries'])
        manifest['c4_manifest_sha256'] = sha(c4.C4_DIR/'release-manifest.json')
        manifest['inf2_entries'] = copy.deepcopy(assets['entries'])
        manifest['inf2_serving'] = assets['serving']
        manifest['inf2_asset_manifest_sha256'] = digest(encoded(assets))
        for e in cm['entries']+assets['entries']:
            manifest['files'][e['path']] = {k:e[k] for k in ('bytes','sha256')}
        for n in ('exp169_partner_bundle_release.py','exp169_release_v2.py'):
            manifest['files']['code/'+n] = dict(bytes=len(updated[n]),sha256=digest(updated[n]))
            manifest['source_code'][n] = digest(updated[n])
        updated['q-a5q-manifest.json'] = encoded(manifest)
        if updated == files: break
        files = updated
    else: raise ValueError('joint pin dependency cycle')
    need(set(files) == set(original) and len(files) == 31, 'exact embedded names')
    need({n for n in original if original[n] != files[n]} == ALLOWED, 'strict nine-file change boundary')
    for n,data in files.items():
        if n.endswith('.py'): compile(data,n,'exec')
    return files


def payload():
    nb, original, old_blob, assignments, assets, cm, sc, _ = components()
    files = patched(original,assets,cm,sc)
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as src, zipfile.ZipFile(buffer,'w') as dst:
        for item in src.infolist(): dst.writestr(item,files[item.filename])
    blob = buffer.getvalue(); source = ''.join(nb['cells'][2]['source'])
    source = replace(source,repr(assignments['BLOB']),repr(base64.b64encode(blob).decode()))
    source = replace(source,'PINS = '+repr(assignments['PINS']),'PINS = '+repr({n:digest(v) for n,v in files.items()}))
    source = replace(source,repr(digest(old_blob)),repr(digest(blob)))
    nb['cells'][2]['source'] = source.splitlines(True)
    need('FAST = True' in ''.join(nb['cells'][0]['source']), 'parent visible policy')
    nb['cells'][0]['source'] = ['FAST = False\n']
    meta = read(BASE/'kernel/kernel-metadata.json')
    need(len(meta['dataset_sources']) == 6, 'six original parent datasets')
    meta.update(id=SLUG,title=SLUG.split('/')[1])
    meta['dataset_sources'] += [c4.DATASET,small.DATASET]
    need(len(set(meta['dataset_sources'])) == 8, 'unique datasets')
    return nb,meta,files,blob


def authenticate_assets():
    """Hash existing bytes only; no torch.load, copying, upload or repackaging."""
    _,_,_,_,assets,cm,_,_ = components()
    result = {}
    for p,expected,size in [(c4.ARCHIVE,c4.ARCHIVE_SHA,560952802),
            (small.OUT/'assets'/small.ARCHIVE_NAME,SMALL_ARCHIVE_SHA,1987773862)]:
        actual = pin(p); need(actual == dict(bytes=size,sha256=expected), 'component archive bytes')
        result[rel(p)] = actual
    for e in cm['entries']+assets['entries']:
        p = ROOT/e['source'].replace('\\','/')
        actual = pin(p)
        need(actual == {k:e[k] for k in ('bytes','sha256')} and e['sha256'] == e['source_sha256'], 'original member bytes')
        result[rel(p)] = actual
    for p in [c4.C4_DIR/'release-manifest.json',small.OUT/small.ASSET_MANIFEST]: result[rel(p)] = pin(p)
    for p,value in assets['original_source_pins'].items():
        actual = pin(ROOT/p); need(actual == {k:value[k] for k in ('bytes','sha256')}, 'Small original source pin')
        result[p] = actual
    return result


def decision_readout(points, contrasts, valid=True):
    """Prospective registered point-dominance rule, never an automatic GO."""
    import numpy as np
    required = {'Q24-Q0','Q24-Q4','Q24-Q2'}
    if not valid or set(contrasts) != required: return 'HOLD_MISSING_OR_INVALID_CONTRAST'
    if set(points) != {'Q0','Q4','Q2','Q24'}: return 'HOLD_INVALID_ARMS'
    arrays = {k:np.asarray(v,float) for k,v in points.items()}
    if any(a.shape != (4,2) or not np.isfinite(a).all() for a in arrays.values()): return 'HOLD_INVALID_POINTS'
    if any(np.asarray(contrasts[k]).shape != (4,12) or not np.isfinite(contrasts[k]).all() for k in required):
        return 'HOLD_MISSING_OR_INVALID_CONTRAST'
    joint = arrays['Q24']
    if any(np.all(joint <= arrays[k]) and np.any(joint < arrays[k]) for k in ('Q4','Q2')):
        return 'NO_DEMONSTRATED_INCREMENTAL_ROLE'
    return 'REPORT_FOR_B_HOLISTIC_JUDGMENT'


def registration(files, kernel_sha):
    sr = read(small.BRIDGE/'registration.json'); cr = read(c4.SCRATCH/'input-pins.json')
    need(sr['targets'] == TARGETS and sr['roster']['rows'] == cr['rows'] == 4343, 'registered cohort/targets')
    need([r['rows'] for r in sr['roster']['folds']] == cr['fold_sizes'], 'registered fold counts')
    need(sr['ruler_bindings'] == cr['ruler_bindings'], 'registered ruler inputs differ')
    return dict(schema='h207-joint-preregister/v1',version='1.0.0',created_kst=stamp(),candidate=KEY,
        kernel_sha256=kernel_sha, parent_kernel_sha256=small.PARENT_SHA,
        arms=['Q0','Q4','Q2','Q24'],contrasts=['Q24-Q0','Q24-Q4','Q24-Q2'],
        targets=TARGETS,small_columns=list(range(8)),dino_columns=[8,9,10,11],
        roster=sr['roster'],ruler_bindings=sr['ruler_bindings'],bootstrap=sr['bootstrap'],
        rank_conventions=sr['rank_conventions'],input_registration_refs={rel(p):pin(p) for p in
            [small.BRIDGE/'registration.json',c4.SCRATCH/'input-pins.json']},
        input_bindings_status='Inherited frozen component closures; no new label/numeric loading. Reauthenticate through existing aligned loaders before reviewed evaluation.',
        sources={rel(p):pin(p) for p in [SELF,TEST,SPEC,ROOT/'tools/hb198_c4_bridge_v1.py',ROOT/'tools/hb199_inf2_bridge_v1.py',ROOT/'tools/hb105_aligned_oof_v1.py']},
        operators={n:digest(files[n]) for n in ['exp169_partner_bundle_release.py','exp169_r224.py','exp169_release_c85.py','exp169_release_p956.py']},
        oof='Generator heldout fold only; FP32 seed pair; concatenate4343 before fixed full-roster ranks. All4 rulers/masks,12 targets, macro/weak4,5 fold slices;300 paired component draws seed20261003. Existing gates unchanged.',
        action_rule='Invalid identity/nonfinite/mask/roster/parity or missing joint/either marginal contrast => HOLD. Weakly worse joint macro+weak4 on every ruler than either single, strictly worse on one => no demonstrated incremental role. Otherwise report all evidence to B; never automatic GO.',
        overlap=dict(discoveries=['D-B-131','D-B-140','D-B-157'],novel_mechanism=False,
            scope='Old Q7 T-to-S already contains R50+C4+Small. H207 asks exact masked current-H189 four-arm joint/marginal effects; no additive-gain assertion.'),
        final_F=dict(state='NOT_MEASURED',roster_count=58,exposure='EXPOSED_GOLD58',clean_oof=False,
            small_owner='H202',c4_owner='H204',reuse='B-delivered independently reviewed terminal caches only; no duplicate inference',
            contrasts=['F24-F0','F24-F4','F24-F2']),
        lenses=dict(L1='exposed Q weak4, pending joint',L2='exposed Q macro, pending joint',L3='correlated rulers/folds; no independence claim',
            L4='all targets/folds pending joint',L5='missing independent paired-seed replication',L6='exact-F awaits reviewed H202/H204 caches',L7='no independently eligible image-labelled joint contrast'),
        exposure='EXPOSED',clean_oof=False,metrics_executed=False,qualified_for_submission=False)


def prepare():
    need(not OUT.exists() and not WORK.exists(), 'fresh target/work directory required')
    nb,meta,files,blob = payload()
    asset_pins = authenticate_assets()
    original = components()[1]
    inventory = {n:dict(parent=digest(original[n]),candidate=digest(v),changed=v!=original[n]) for n,v in files.items()}
    OUT.mkdir(exist_ok=False); WORK.mkdir(parents=True,exist_ok=False)
    for n in ('kernel/release.ipynb','kernel/kernel-metadata.json'): write_new(WORK/'parent-backup'/n,(BASE/n).read_bytes())
    for n,data in [('kernel/release.ipynb',encoded(nb)),('kernel/kernel-metadata.json',encoded(meta)),
            ('embedded-diff.json',encoded(inventory)),('SPEC.md',SPEC.read_bytes()),
            ('CHANGELOG.md','# '+VERSION+'\nOne H189 C4+INF2 union: C4five plus Small10 fixed eight-target substitution; original launch/pool/thread/timing, outer F, failures and codecs. Eight existing datasets; ordinary visible FAST=False. No asset repackaging.\n')]:
        write_new(OUT/n,data)
    diff = ''.join(''.join(difflib.unified_diff(original[n].decode().splitlines(True),v.decode().splitlines(True),fromfile='H189/'+n,tofile='C4INF2/'+n)) for n,v in files.items())
    write_new(OUT/'full-embedded-diff.txt',diff)
    outer = ''.join(''.join(difflib.unified_diff((BASE/n).read_text().splitlines(True),(OUT/n).read_text().splitlines(True),fromfile='H189/'+n,tofile='C4INF2/'+n)) for n in ('kernel/release.ipynb','kernel/kernel-metadata.json'))
    write_new(OUT/'full-notebook-metadata-diff.txt',outer)
    for n,v in files.items(): write_new(WORK/'candidate-runtime'/n,v)
    kernel_sha = sha(OUT/'kernel/release.ipynb')
    write_new(WORK/'JOINT-REGISTRATION.json',encoded(registration(files,kernel_sha)))
    write_new(WORK/'ASSET-AUTH.json',encoded(dict(status='HASHED_EXISTING_BYTES_NO_MODEL_LOAD',created_kst=stamp(),pins=asset_pins)))
    sources = [SELF,TEST,SPEC,Path(small.__file__),Path(c4.__file__),small.HELPER,
        small.BRIDGE/'RESULT-REVIEW.md',small.BRIDGE/'SPEC-ERRATA.md',c4.SCRATCH/'BUILD-REVIEW.md',c4.SCRATCH/'ASSET-AUTH.json']
    receipt = dict(schema='h207-build/v1',candidate=KEY,version=VERSION,builder=BUILDER,created_kst=stamp(),
        status='BUILT_CPU_ONLY_PENDING_INDEPENDENT_REVIEW',kernel_id=SLUG,kernel_sha256=kernel_sha,
        metadata_sha256=sha(OUT/'kernel/kernel-metadata.json'),parent_kernel_sha256=small.PARENT_SHA,
        component_kernel_sha256=dict(C4=C4_SHA,INF2=SMALL_SHA),source_pins={rel(p):pin(p) for p in sources},
        all_embedded=inventory,changed=sorted(ALLOWED),unchanged=22,
        joint_registration_sha256=sha(WORK/'JOINT-REGISTRATION.json'),asset_auth_sha256=sha(WORK/'ASSET-AUTH.json'),
        model_forwards=0,gpu_execution=False,metrics_executed=False,qualified_for_submission=False)
    write_new(OUT/'build-receipt.json',encoded(receipt))
    return receipt


def validate():
    nb,meta,files,_ = payload(); r = read(OUT/'build-receipt.json')
    need(read(OUT/'kernel/release.ipynb') == nb and read(OUT/'kernel/kernel-metadata.json') == meta, 'exact stage bytes')
    need(unpack(nb)[0] == files, 'embedded payload drift')
    need(r['kernel_sha256'] == sha(OUT/'kernel/release.ipynb') and r['metadata_sha256'] == sha(OUT/'kernel/kernel-metadata.json'), 'stage receipt drift')
    for p,value in r['source_pins'].items(): need(pin(ROOT/p) == value, 'source pin drift: '+p)
    for name,key in [('JOINT-REGISTRATION.json','joint_registration_sha256'),('ASSET-AUTH.json','asset_auth_sha256')]:
        need(sha(WORK/name) == r[key], 'build evidence drift')
    need(read(OUT/'embedded-diff.json') == r['all_embedded'], 'all-file diff record drift')
    return dict(status='BUILD_BYTES_VALID',kernel_sha256=r['kernel_sha256'],unchanged=22,changed=sorted(ALLOWED))


def renderer():
    """Bounded CPU replay: two synthetic and the existing C4 card's first3 studies."""
    validate()
    import numpy as np
    import torch
    import textwrap
    from tools import hb198_c4_bridge_v1 as bridge
    torch.set_num_threads(1)
    need(not torch.cuda.is_initialized(), 'CPU renderer only')
    def forbidden(*a,**k): raise RuntimeError('model/CUDA execution forbidden in renderer replay')
    torch.load = forbidden
    torch.nn.Module.__call__ = forbidden
    torch.cuda._lazy_init = forbidden
    common,rad,runtime,_ = bridge.modules()
    corpus = common.Corpus(bridge.CORPUS)
    card = read(c4.SCRATCH/'gold58-launch.json')
    for n,h in card['corpus']['metadata'].items(): need(sha(bridge.CORPUS/n) == h, 'renderer corpus metadata pin')
    _, original,_,_,_,_,sc,cc = components()
    actual = unpack(read(OUT/'kernel/release.ipynb'))[0]
    scope = dict(center3_windows_from=rad.center3_windows_from,windows_from=common.windows_from,
        normalise=rad.normalise,normalize_input=runtime.normalize_input)
    small_fn = bridge.functions(actual['exp169_partner_bundle_release.py'],['inf2_small_windows','plane_flags'],dict(need=need))
    def produce(files, volume, centers):
        src = files['exp169_partner_bundle_release.py'].decode()
        start = src.index('                center3 = center3_windows_from(volume, centers, 224)')
        end = src.index('\n                prepared = time.monotonic()',start)
        ns = dict(scope,**small_fn,volume=volume,centers=centers,i=0,flags=small_fn['plane_flags'](centers,corpus.spec))
        exec(compile(textwrap.dedent(src[start:end]),'<actual-joint-producer>','exec'),ns)
        return ns['payload']
    rng=np.random.default_rng(20261018)
    synthetic=rng.integers(0,256,(64,336,336),dtype=np.uint8)
    cases=[('synthetic',synthetic,np.ones(64,np.uint8)),('empty',synthetic,np.zeros(64,np.uint8))]
    for i,uid in enumerate(card['corpus']['ids'][:3]):
        volume=corpus.volume(uid)
        need(digest(volume.tobytes()) == card['corpus']['volumes'][i]['sha256'], 'registered real row hash')
        cases.append(('real-index-'+str(i),volume,corpus.mask(uid)))
    records=[]
    for label,volume,mask in cases:
        centers=common.select_centers(mask,64,36,corpus.spec,'within-slot-valid')
        if label == 'synthetic': centers[3]=centers[8]=-1
        p,c,s,j=[produce(f,volume,centers) for f in (original,cc,sc,actual)]
        need(len(j)==6 and np.array_equal(j[1],p[1]) and np.array_equal(j[3],p[3]), 'unchanged R50/DINO tensor')
        need(np.array_equal(j[2],c[2]) and np.array_equal(j[5],s[5]), 'exact existing C4/Small tensors')
        flags=np.asarray(j[4],bool)
        need(np.array_equal(flags,c[4]) and np.array_equal(flags,s[4]), 'exact plane flags')
        for index,reference in [(1,p[1]),(2,c[2]),(3,p[3]),(5,s[5])]:
            mirror=j[index].copy();mirror[flags]=mirror[flags,:,:,::-1]
            expected=reference.copy();expected[flags]=expected[flags,:,:,::-1]
            need(np.array_equal(mirror,expected) and j[index].dtype==np.float32, 'exact FP32 mirrored input')
        records.append(dict(case=label,centers=36,empty=centers.count(-1),exact=True,
            input_sha256={name:digest(j[index].tobytes()) for name,index in [('R50',1),('C4',2),('DINO',3),('Small',5)]}))
    need(not torch.cuda.is_initialized(), 'no CUDA initialization')
    result=dict(status='PASS_CPU_INPUT_RENDERER_ONLY',created_kst=stamp(),kernel_sha256=sha(OUT/'kernel/release.ipynb'),
        cases=records,real_studies=3,real_source='Existing C4 Gold58 card indices0..2; not the Kaggle public-test DICOM cohort',
        model_forwards=0,cuda_initialized=False,volume_bytes_read=3*64*336*336,
        card_sha256=sha(c4.SCRATCH/'gold58-launch.json'),public_test_dicom_renderer_checked=False)
    write_new(WORK/'RENDERER-PARITY.json',encoded(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','validate','renderer'))
    args=parser.parse_args(); result={'prepare':prepare,'validate':validate,'renderer':renderer}[args.command]()
    print(json.dumps({k:result[k] for k in ('status','kernel_sha256','unchanged','changed','real_studies','model_forwards') if k in result}))


if __name__ == '__main__': main()
