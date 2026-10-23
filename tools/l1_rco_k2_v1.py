"""K2 offline builder/CPU replay/collector v1.0.0. No GPU or network launch.

Changelog: new reviewed single-model FP32 workers0 release path. Historical
sources are immutable. register/build is create-only; evidence is separate.
"""
from __future__ import annotations
import argparse
import ast
import csv
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp189_s6_port as s6
from tools import l1_rco_k2_worker_v1 as w

VERSION = '1.0.0'
CANDIDATE = 'K2-RCOF0-V1'
STAGE = ROOT / 'artifacts/experiments/L1_RCO_K2/v1.0.0-rcof0'
WORK = ROOT / '.local/scratch/slot-b/k2-release'
ASSETS = WORK / 'assets-v1'
DATASET = 'taeyangg4/knee-l1-rcof0-v1-assets'
CODEC_DATASET = 'taeyangg4/knee-hb126-codecs-cp313-v1'
K1 = ROOT / '.local/scratch/l1-rco/k1v2'
CODE = ROOT / '.local/exp189-cpu-v120/input/cloud_code'
ANATOMY = ROOT / 'storage/weights/external/heliosli_rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt'
MODEL = K1 / 'fit/models/E_prior_s0_f0.pt'
RAW = ROOT / 'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip'
VISIBLE = ROOT / '.local/exp189-cpu-v120/input/visible_data'
VISIBLE_MANIFEST = ROOT / '.local/exp189-cpu-v120/input/visible_dicom_manifest.json'
WHEELS = ROOT / '.local/exp189-cpu-v120/wheels'
INIT = ROOT / 'storage/weights/exp171/convnext_small_fb_in22k_ft_in1k.safetensors'
NEW = ('tools/l1_rco_k2_v1.py', 'tools/l1_rco_k2_worker_v1.py',
       'tools/strict_variant_gate_k2_v1.py', 'tools/l1_rco_k2_admission_v1.py',
       'tools/kaggle_submit_once_k2_v1.py', '.local/run-scripts/b-preflight-k2-v1.py',
       'tools/test_l1_rco_k2_v1.py')
FROZEN = ('tools/strict_variant_gate.py', 'tools/strict_variant_gate_standalone.py',
          'tools/strict_variant_gate_hb177.py', 'tools/hb177_release_admission_v1.py',
          'tools/strict_variant_gate_r224_v145.py', 'tools/kaggle_submit_once.py',
          'tools/kaggle_submit_once_v139.py', '.local/run-scripts/b-preflight-v107.py',
          '.local/run-scripts/kaggle-visible-wait-v3.sh', 'tools/l1_rco_k1_v2.py',
          'tools/l1_rco_stage0_v1.py', 'tools/l1_rco_stage0_v120.py',
          'tools/l1_rco_runtime_v120.py', 'tools/exp189_s6_port.py')
PINS = {str(MODEL.relative_to(ROOT)): w.MODEL_SHA, str(ANATOMY.relative_to(ROOT)): w.ANATOMY_SHA,
    '.local/scratch/l1-rco/k1v2/registration.json': '79ebcbdf85222647715de0b9d2529bbc164696954f581c44991e031bfbb80cb0',
    '.local/scratch/l1-rco/k1v2/compact_train_l1.py': '34dae104527a2e8c27ed328c5e2bca4039d28e7e596ab530b32de63a95848d8e',
    '.local/scratch/l1-rco/k1v2/fit/holdout.csv': '65bc291954861d9ab890bc016229a8a92d43e4de854b8a74ae6507b0dd04776b',
    str(INIT.relative_to(ROOT)): '2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9'}


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def ast_identity():
    a = ast.parse((s6.SOURCE / 'compact_train.py').read_bytes())
    b = ast.parse((K1 / 'compact_train_l1.py').read_bytes())
    names = ('Net', 'Studies', 'collate', 'window_rows', 'covering_centers', 'phys_step')
    def selected(tree):
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if getattr(n, 'name', '') in names}
    x, y = selected(a), selected(b)
    w.need(set(x) == set(names) and x == y, 'K1 serving AST differs')
    return {k: hashlib.sha256(v.encode()).hexdigest() for k, v in x.items()}


def inputs():
    for name, pin in PINS.items():
        w.need(w.sha(ROOT / name) == pin, 'frozen K1 pin: ' + Path(name).name)
    s6.verify_source()
    full = s6.verify_code(CODE, ANATOMY)
    s6.verify_wheels([WHEELS / n for n in s6.WHEEL_HASHES])
    fit, reg = w.read(K1 / 'fit/fit-complete.json'), w.read(K1 / 'registration.json')
    w.need(fit['status'] == 'COMPLETE' and fit['epochs'] == 10 and fit['model_fold'] == 0 and
           fit['contract_id'] == reg['contract_id'] and fit['model_sha256'] == w.MODEL_SHA, 'completed K1 fit')
    closure = full['closure_sha256'].copy()
    # anatomy.load_segmenter changes sys.path and MONAI recursively discovers modules.
    # Include that exact frozen package as a dynamic-import closure over-approximation.
    for p in sorted((CODE / 'training/vendor_runtime/monai060').rglob('*.py')):
        closure[p.relative_to(CODE).as_posix()] = w.sha(p)
    return full, dict(sorted(closure.items())), ast_identity()


def package():
    w.need(not ASSETS.exists(), 'asset destination exists; preserve it and use reviewed repair')
    full, closure, asts = inputs()
    ASSETS.mkdir(parents=True)
    files, origins = {}, {}
    pairs = [(MODEL, 'weights/E_prior_s0_f0.pt'), (ANATOMY, 'weights/anatomy.pt'),
             (ROOT / NEW[1], 'source/l1_rco_k2_worker_v1.py')]
    pairs += [(s6.SOURCE / n, 'source/' + n) for n in s6.SOURCE_HASHES]
    pairs += [(CODE / n, 'cloud_code/' + n) for n in closure]
    pairs += [(WHEELS / n, 'wheels/' + n) for n in s6.WHEEL_HASHES]
    for source, name in pairs:
        dest = ASSETS / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        files[name] = w.sha(dest)
        origins[name] = dict(source=relative(source), sha256=w.sha(source), relation='byte-identical')
        w.need(files[name] == origins[name]['sha256'], 'copy identity')
    source_info = dict(sources=['https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-v1',
        'https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
        'https://pypi.org/project/timm/1.0.25/', 'https://pypi.org/project/pydicom/3.0.1/'],
        public_code_full_tree_sha256=full['tree_sha256'], public_code_full_tree_files=full['files'],
        packaged_closure_files=len(closure), closure_method='static56 plus frozen dynamically discovered MONAI package',
        code_subset_sha256=s6.tree_digest(closure), clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    w.write(ASSETS / 'sources.json', source_info)
    files['sources.json'] = w.sha(ASSETS / 'sources.json')
    manifest = dict(schema='k2-assets/v1', version=w.VERSION, dataset=DATASET, private=True,
        contract=w.CONTRACT, files=dict(sorted(files.items())), wheels=s6.WHEEL_HASHES,
        deployment_versions=s6.WHEEL_VERSIONS, codec_dataset=CODEC_DATASET,
        codec_ABI='NOT-CHECKED: B must authenticate installed Kaggle ABI and run decoder samples',
        code_closure=closure, source_info=source_info)
    w.write(ASSETS / 'asset-manifest.json', manifest)
    w.write(ASSETS / 'dataset-metadata.json', dict(id=DATASET, title=DATASET.split('/')[1],
        licenses=[{'name': 'other'}], description='Private K2 single deploy weight and frozen offline code. EXPOSED_EXPLORATORY; clean_oof=false. No data, labels or prediction tables.'))
    w.write(WORK / 'asset-origins.json', origins)
    return dict(manifest_sha256=w.sha(ASSETS / 'asset-manifest.json'), asset_count=len(files), asts=asts)


def notebook(manifest_sha):
    # Minimal new controller; no public notebook and no FAST branch.
    source = '''import hashlib, json, os, pathlib, subprocess, sys, time
started = time.monotonic()
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
assets = pathlib.Path('/kaggle/input/knee-l1-rcof0-v1-assets')
expected = MANIFEST_SHA
assert hashlib.sha256((assets/'asset-manifest.json').read_bytes()).hexdigest() == expected
manifest = json.loads((assets/'asset-manifest.json').read_bytes())
for name, pin in manifest['files'].items():
    with (assets/name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == pin, name
# Only authenticated pure-Python wheels. The existing private codec dataset is
# mounted for B's ABI qualification; no unpinned codec install is attempted.
wheels = [str(assets/'wheels'/name) for name in sorted(manifest['wheels'])]
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', *wheels], check=True)
data = pathlib.Path('/kaggle/input/rsna-knee-abnormality-detection')
out = pathlib.Path('/kaggle/working')
subprocess.run([sys.executable, '-B', str(assets/'source/l1_rco_k2_worker_v1.py'),
    '--assets', str(assets), '--manifest-sha', expected, '--data', str(data),
    '--template', str(data/'sample_submission.csv'), '--output', str(out/'submission.csv'),
    '--scratch', str(out/'k2-owned-scratch'), '--device', 'cuda:0'], check=True)
(out/'bootstrap-timing.json').write_text(json.dumps({'schema':'k2-bootstrap/v1',
    'total_seconds':time.monotonic()-started, 'manifest_sha256':expected}))
print('K2 real inference complete; independent qualification still required')
'''.replace('MANIFEST_SHA', repr(manifest_sha))
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3')),
        cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source.splitlines(True))])


def register():
    w.need(not STAGE.exists(), 'stage exists; never overwrite sealed registration')
    full, closure, asts = inputs()
    if not ASSETS.exists():
        package()
    manifest_sha = w.sha(ASSETS / 'asset-manifest.json')
    w.assets_check(ASSETS, manifest_sha)
    reg1 = w.read(K1 / 'registration.json')
    # Selection is fixed without reading label/score values or K1 prediction rows.
    holdout = sorted(reg1['holdout'], key=lambda uid: hashlib.sha256(('K2-engineering-N32-v1\0'+uid).encode()).hexdigest())[:32]
    visible_manifest = w.read(VISIBLE_MANIFEST)
    visible = w.read_template(VISIBLE/'sample_submission.csv')
    w.need(set(visible)=={v['path'].split('/')[1] for v in visible_manifest},'visible template/raw UID set')
    w.need(len(visible) == 3 and len(holdout) == 32, 'registered cohort sizes')
    for item in visible_manifest:
        p = VISIBLE / item['path']
        w.need(p.is_file() and w.sha(p) == item['sha256'], 'visible raw-study byte identity')
    STAGE.mkdir(parents=True)
    (STAGE / 'kernel').mkdir()
    sources = {n: w.sha(ROOT/n) for n in NEW + FROZEN}
    sources.update({relative(s6.SOURCE/n): pin for n, pin in s6.SOURCE_HASHES.items()})
    sources.update({relative(CODE/n): pin for n, pin in closure.items()})
    sources.update({Path(n).as_posix(): pin for n, pin in PINS.items()})
    for p in (K1/'prepare.json', K1/'fit/result.json', K1/'fit/fit-complete.json', K1/'evaluate/result.json',
              WORK/'DESIGN-REVIEW.md', ROOT/'.local/scratch/slot-b/k1v2-result-review/native/REPORT.md',
              WORK/'asset-origins.json', VISIBLE_MANIFEST,VISIBLE/'sample_submission.csv'):
        sources[relative(p)] = w.sha(p)
    w.write(STAGE / 'source-manifest.json', sources)
    reg = dict(schema='k2-registration/v1', candidate_id=CANDIDATE, kernel_id=w.SLUG, version=w.VERSION,
        contract=w.CONTRACT, assets=relative(ASSETS), asset_manifest_sha256=manifest_sha,
        source_manifest_sha256=w.sha(STAGE/'source-manifest.json'), ast_identity=asts,
        k1_registration_sha256=PINS['.local/scratch/l1-rco/k1v2/registration.json'],
        k1_model_sha256=w.MODEL_SHA, visible_ids=visible, smoke_holdout_ids=holdout[:2], n32_ids=holdout,
        visible_roster_sha256=w.digest(visible), n32_roster_sha256=w.digest(holdout),
        cache=w.read(K1/'prepare.json')['cache'], full_code_tree_sha256=full['tree_sha256'],
        parity=dict(same_runtime=dict(atol=1e-6, rtol=1e-5), cross_runtime=dict(atol=1e-5, rtol=1e-4),
                    k1_bf16=dict(max_abs=.01, mean_abs=.001, per_target_mean_abs=.002)),
        runtime=dict(population=1300, population_source='tools/c85r7_r6w958rdq_v1140.py:76',
                     population_affirmation_required=True, formula='setup_seconds + 1300 * max(visible_mean_e2e, n32_mean_e2e)',
                     gate_seconds=32400, workload_rule='complete studies, all windows, decode through final write'),
        reading_rule=dict(alive_gte=.950, dead_lte=.946), qualified_for_submission=False,
        remote_identity_check='PENDING B live exact-slug duplicate check before publication',
        code_path='NEW standalone orchestration, not a checkpoint-only hidden-proven child')
    w.write(STAGE/'registration.json', reg)
    w.write(STAGE/'kernel/release.ipynb', notebook(manifest_sha))
    w.write(STAGE/'kernel/kernel-metadata.json', dict(id=w.SLUG, title=w.SLUG.split('/')[1],
        code_file='release.ipynb', language='python', kernel_type='notebook', is_private=True,
        enable_gpu=True, enable_internet=False, dataset_sources=[DATASET, CODEC_DATASET],
        competition_sources=['rsna-knee-abnormality-detection'], kernel_sources=[]))
    # Entire newly authored notebook is explicit; inherited files are byte-identical.
    diff = ['NEW CODE PATH: no hidden-proven orchestration claim.\n']
    for n in NEW:
        diff.extend(difflib.unified_diff([], (ROOT/n).read_text(encoding='utf-8').splitlines(True), fromfile='/dev/null', tofile=n))
    diff.extend(difflib.unified_diff([], (STAGE/'kernel/release.ipynb').read_text().splitlines(True), fromfile='/dev/null', tofile='kernel/release.ipynb'))
    origins = w.read(WORK/'asset-origins.json')
    for name, item in sorted(origins.items()):
        diff.append('UNCHANGED '+name+' '+item['sha256']+' FROM '+item['source']+'\n')
    (STAGE/'full-inventory-diff.txt').write_text(''.join(diff), encoding='utf-8')
    inventory = dict(new_sources={n:sources[n] for n in NEW}, inherited_assets=origins,
        removed=[], changed_inherited=[], notebook_sha256=w.sha(STAGE/'kernel/release.ipynb'),
        metadata_sha256=w.sha(STAGE/'kernel/kernel-metadata.json'), asset_manifest_sha256=manifest_sha,
        complete_diff_sha256=w.sha(STAGE/'full-inventory-diff.txt'))
    w.write(STAGE/'embedded-file-manifest.json', inventory)
    sealed = {p.relative_to(STAGE).as_posix():w.sha(p) for p in STAGE.rglob('*') if p.is_file()}
    (STAGE/'SHA256SUMS.txt').write_text(''.join(pin+'  '+n+'\n' for n,pin in sorted(sealed.items())), encoding='utf-8')
    kernel_hash = w.digest({n:w.sha(STAGE/'kernel'/n) for n in ('kernel-metadata.json','release.ipynb')})
    return dict(status='BUILT_NOT_ADMITTED', kernel_sha256=kernel_hash, stage=relative(STAGE), qualified_for_submission=False)


def template(path, ids):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([w.UID]+w.LABELS)
        for uid in ids:
            writer.writerow([uid]+['']*12)


def prediction_table(path, ids, k1=False):
    import numpy as np
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        w.need(reader.fieldnames == [w.UID]+w.LABELS+(['status','model_fold'] if k1 else []), 'prediction header')
        rows = {}
        for row in reader:
            uid = row[w.UID]
            w.need(uid not in rows, 'duplicate prediction UID')
            if k1:
                w.need(row['status']=='ok' and row['model_fold']=='0', 'saved K1 row status/fold')
            rows[uid] = [float(row[x]) for x in w.LABELS]
    w.need(set(ids) <= set(rows) if k1 else set(ids)==set(rows), 'prediction roster')
    x = np.asarray([rows[u] for u in ids], np.float32)
    w.need(x.shape==(len(ids),12) and np.isfinite(x).all() and (x>=0).all() and (x<=1).all(), 'prediction values')
    return x


def parity(a, b, kind):
    import numpy as np
    a,b = np.asarray(a),np.asarray(b)
    w.need(a.shape==b.shape and a.ndim==2 and a.shape[1]==12 and len(a)>0 and np.isfinite(a).all() and np.isfinite(b).all(), 'parity arrays')
    d=np.abs(a-b)
    if kind=='k1_bf16':
        passed = bool(d.max()<=.01 and d.mean()<=.001 and (d.mean(axis=0)<=.002).all())
    else:
        w.need(kind in ('same_runtime','cross_runtime'), 'parity kind')
        atol,rtol=(1e-6,1e-5) if kind=='same_runtime' else (1e-5,1e-4)
        passed=bool(np.allclose(a,b,atol=atol,rtol=rtol))
    disagreements=[int(((np.sign(a[:,j,None]-a[None,:,j]) != np.sign(b[:,j,None]-b[None,:,j])) & np.triu(np.ones((len(a),len(a)),bool),1)).sum()) for j in range(12)]
    return dict(status='PASS' if passed else 'FAIL', kind=kind, studies=len(a), max_abs=float(d.max()),
        mean_abs=float(d.mean()), per_target_mean_abs=d.mean(axis=0).tolist(), rank_disagreements=disagreements)


def cpu_smoke(part, destination, deployment_libraries=False):
    destination=Path(destination)
    w.need(destination.resolve().is_relative_to(WORK.resolve()) and not destination.exists(), 'fresh named CPU evidence only')
    destination.mkdir(parents=True)
    if deployment_libraries:
        libs=WORK/'deployment-libs'
        w.need(libs.is_dir(),'install pinned offline wheels into the named deployment-libs directory first')
        s6.verify_wheels([WHEELS/n for n in s6.WHEEL_HASHES])
        for wheel in s6.WHEEL_HASHES:
            with zipfile.ZipFile(WHEELS/wheel) as archive:
                for member in archive.namelist():
                    if member.endswith('/') or member.endswith('.dist-info/RECORD'):
                        continue  # pip rewrites its installation accounting only.
                    path=(libs/member).resolve()
                    w.need(path.is_relative_to(libs.resolve()) and path.read_bytes()==archive.read(member),
                           'installed deployment wheel bytes changed')
        sys.path.insert(0,str(libs.resolve()))
    torch=w.configure('cpu')
    manifest_sha=w.sha(ASSETS/'asset-manifest.json')
    w.assets_check(ASSETS,manifest_sha)
    reg=w.read(STAGE/'registration.json')
    ct,ci=w.import_frozen(ASSETS)
    if deployment_libraries:
        import importlib.metadata
        w.need({k:importlib.metadata.version(k) for k in s6.WHEEL_VERSIONS}==s6.WHEEL_VERSIONS,
               'deployment library versions differ from pinned wheels')
    started=time.perf_counter()
    common=dict(schema='k2-cpu-smoke/v1',part=part,registration_sha256=w.sha(STAGE/'registration.json'),
        manifest_sha256=manifest_sha,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        qualified_for_submission=False)
    try:
        with w.offline():
            if part=='model':
                model=w.load_model(ct,ASSETS/'weights/E_prior_s0_f0.pt','cpu')
                w.need(w.sha(INIT)==PINS[str(INIT.relative_to(ROOT))], 'initialization pin')
                initial=ct.Net('convnext_small',str(INIT),crop=352,in_chans=3,spatial=True,extent=False,state_spec=None)
                initial.load_deploy_state_dict(torch.load(MODEL,map_location='cpu',weights_only=True))
                w.need(all(torch.equal(v,initial.state_dict()[k]) for k,v in model.state_dict().items()), 'init-none state equivalence')
                rec=dict(common,status='PASS',keys=len(model.state_dict()),initialization_plus_deploy_exact=True,ast_identity=ast_identity())
            elif part=='spawn':
                ds=ct.Studies(reg['cache'],reg['smoke_holdout_ids'],64,'eval',0,eval_stride=3,slices=3,phys_mm=3.5)
                zero=list(torch.utils.data.DataLoader(ds,batch_size=1,num_workers=0,collate_fn=ct.collate))
                # Never pickle a Dataset after worker0 has opened its huge memmaps.
                # Fresh canonical instances let each spawn child lazily map the cache.
                ds_spawn=ct.Studies(reg['cache'],reg['smoke_holdout_ids'],64,'eval',0,eval_stride=3,slices=3,phys_mm=3.5)
                three=list(torch.utils.data.DataLoader(ds_spawn,batch_size=1,num_workers=3,multiprocessing_context='spawn',collate_fn=ct.collate))
                w.need(len(zero)==len(three)==2 and all(torch.equal(x,y) for a,b in zip(zero,three) for x,y in zip(a,b)), 'spawn3 parity')
                rec=dict(common,status='PASS',studies=2,workers=[0,3],windows=[int(a[2].sum()) for a in zero],exact=True)
            elif part in ('raw-visible','raw-holdout'):
                ids=reg['visible_ids'] if part=='raw-visible' else reg['smoke_holdout_ids']
                data=VISIBLE if part=='raw-visible' else extract_holdout(ids,destination/'raw')
                split='test' if part=='raw-visible' else 'train'
                template(destination/'template.csv',ids)
                rec=w.run(ASSETS,manifest_sha,data,destination/'template.csv',destination/'submission.csv',destination/'scratch','cpu',split)
                rec=dict(common,status='PASS',studies=len(ids),worker_receipt_sha256=w.sha(destination/'worker-receipt.json'),
                    prediction_sha256=w.sha(destination/'submission.csv'))
            elif part in ('cached-reference','cached-reference-n32'):
                ids=reg['smoke_holdout_ids'] if part=='cached-reference' else reg['n32_ids']
                ds=ct.Studies(reg['cache'],ids,64,'eval',0,eval_stride=3,slices=3,phys_mm=3.5)
                model=w.load_model(ct,MODEL,'cpu')
                values=[]; windows=[]; hashes=[]
                for i in range(len(ds)):
                    x,m,index=ds[i]
                    values.append(w.predict(model,x,m,'cpu')); windows.append(len(x))
                    hashes.append(dict(images_sha256=w.tensor_sha(x),metadata_sha256=w.tensor_sha(m),windows=len(x)))
                import numpy as np
                arr=np.asarray(values)
                with (destination/'reference.csv').open('w',newline='',encoding='utf-8') as f:
                    writer=csv.writer(f);writer.writerow([w.UID]+w.LABELS)
                    for uid,row in zip(ids,arr):writer.writerow([uid]+[format(float(v),'.9g') for v in row])
                p=parity(arr,prediction_table(K1/'fit/holdout.csv',ids,k1=True),'k1_bf16')
                rec=dict(common,status=p['status'],studies=len(ids),windows=windows,tensors=hashes,k1_bf16=p,
                    reference_sha256=w.sha(destination/'reference.csv'),cache_index_sha256=w.sha(Path(reg['cache'])/'index.json'))
                if part=='cached-reference-n32':
                    w.write(destination/'fp32-reference-receipt.json',dict(schema='k2-fp32-reference/v1',status='COMPLETE',
                        model_sha256=w.MODEL_SHA,roster_sha256=reg['n32_roster_sha256'],
                        output_sha256=rec['reference_sha256'],contract=w.CONTRACT,tensors=hashes,
                        cache_index_sha256=rec['cache_index_sha256'],environment=w.environment('cpu'),
                        registration_sha256=common['registration_sha256'],source_sha256=w.sha(Path(__file__))))
            else:
                raise ValueError('unknown CPU smoke part')
        rec.update(seconds=time.perf_counter()-started,environment=w.environment('cpu'))
        w.write(destination/'result.json',rec)
        return rec
    except BaseException as exc:
        w.write(destination/'result.json',dict(common,status='FAIL',exception_type=type(exc).__name__,
            error=str(exc)[:300],seconds=time.perf_counter()-started))
        raise


def extract_holdout(ids,destination):
    """Extract only exact registered image members, never train.csv or reports."""
    destination=Path(destination)
    w.need(not destination.exists(), 'raw destination exists')
    destination.mkdir(parents=True)
    records=[]
    with zipfile.ZipFile(RAW) as z:
        # Series metadata contains no study target columns. Never read train.csv.
        with z.open('train_series.csv') as stream:
            reader=csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8'))
            columns=[w.UID,'SeriesInstanceUID','Anatomical_Plane','Fat_Suppression','Fluid_Sensitive']
            w.need(set(columns)<=set(reader.fieldnames),'series metadata columns')
            metadata=[{key:row[key] for key in columns} for row in reader if row[w.UID] in ids]
        with (destination/'train_series.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(metadata)
        for item in z.infolist():
            parts=Path(item.filename).as_posix().split('/')
            if len(parts)!=4 or parts[0]!='train_series' or parts[1] not in ids or not parts[-1].endswith('.dcm'):
                continue
            target=(destination/item.filename).resolve()
            w.need(target.is_relative_to(destination.resolve()), 'unsafe archive member')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(item) as src,target.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            records.append(dict(path=item.filename,sha256=w.sha(target),bytes=item.file_size))
    w.need({v['path'].split('/')[1] for v in records}==set(ids), 'raw holdout studies missing')
    w.write(destination.parent/'raw-input-manifest.json',dict(archive=relative(RAW),archive_bytes=RAW.stat().st_size,
        roster_sha256=w.digest(ids),members=records))
    return destination


def collect_cpu(paths,output):
    records={}
    for path in paths:
        p=Path(path); rec=w.read(p)
        w.need(rec['part'] not in records and rec['status']=='PASS', 'CPU part duplicate/failure')
        w.need(rec['registration_sha256']==w.sha(STAGE/'registration.json') and
               rec['manifest_sha256']==w.sha(ASSETS/'asset-manifest.json'), 'CPU stale identity')
        records[rec['part']]=dict(path=relative(p),sha256=w.sha(p))
    w.need(set(records)=={'model','spawn','raw-visible','raw-holdout','cached-reference'}, 'incomplete CPU smoke set')
    raw_path=ROOT/records['raw-holdout']['path']
    raw=w.read(raw_path.parent/'worker-receipt.json')
    cached=w.read(ROOT/records['cached-reference']['path'])
    w.need([{k:s[k] for k in ('images_sha256','metadata_sha256','windows')} for s in raw['studies']]==cached['tensors'],
           'raw versus authenticated cached windows differ: diagnose geometry/anchor route')
    reg=w.read(STAGE/'registration.json'); ids=reg['smoke_holdout_ids']
    p=parity(prediction_table(raw_path.parent/'submission.csv',ids),
        prediction_table((ROOT/records['cached-reference']['path']).parent/'reference.csv',ids),'same_runtime')
    w.need(p['status']=='PASS','same-runtime CPU probability parity')
    result=dict(schema='k2-cpu-collection/v1',status='PASS',registration_sha256=w.sha(STAGE/'registration.json'),
        parts=records,same_runtime=p,raw_cached_images_exact=True,qualified_for_submission=False)
    w.write(output,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('package');sub.add_parser('build')
    q=sub.add_parser('cpu-smoke');q.add_argument('--part',choices=['model','spawn','raw-visible','raw-holdout','cached-reference','cached-reference-n32'],required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--deployment-libraries',action='store_true')
    q=sub.add_parser('collect-cpu');q.add_argument('--receipts',nargs='+',required=True);q.add_argument('--output',required=True)
    a=p.parse_args(argv)
    if a.command=='package':result=package()
    elif a.command=='build':result=register()
    elif a.command=='cpu-smoke':result=cpu_smoke(a.part,a.output,a.deployment_libraries)
    else:result=collect_cpu(a.receipts,a.output)
    # IDs and probabilities remain in private local files; console is aggregate only.
    print(json.dumps({k:v for k,v in result.items() if k in ('status','kernel_sha256','stage','manifest_sha256','asset_count','seconds','studies','keys','qualified_for_submission')},indent=2))


if __name__=='__main__':main()
