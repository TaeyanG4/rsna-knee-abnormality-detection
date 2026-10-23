"""K2 prospective BUILD/final-transport builder v1.0.2. No GPU or network launch.

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
from tools import l1_rco_k2_worker_v101 as w

VERSION = '1.0.2'
STAGE_VERSION = '1.0.2-rcof0'
PROVENANCE_CONTRACT = 'k2-build-final-transport/v1'
CANDIDATE = 'K2-RCOF0-V1'
STAGE = ROOT / 'artifacts/experiments/L1_RCO_K2/v1.0.2-rcof0'
WORK = ROOT / '.local/scratch/slot-b/k2-release/reseal-v102'
ASSET_WORK = ROOT / '.local/scratch/slot-b/k2-release/fix1'
ASSETS = ASSET_WORK / 'assets-v1'
OLD_STAGE = ROOT / 'artifacts/experiments/L1_RCO_K2/v1.0.1-rcof0'
OLD_REG_SHA = '064ce70162a40b4782b45bcf1f884b3220a10ed091dab1f69d1ec67bde79302c'
OLD_MANIFEST_SHA = '23b7970965fb0d3197d1e8e3c02bc1fe03a7e5b7c78c82a0b6ed499b6da28253'
ASSET_SHA = 'a529559ad7efac3e88111b0007338ca931bde3e3a35b8e88b1d09ca3f44e68f3'
WORKER_SHA = 'cfcba1a26dddda54ebfeadd874d58538c1ac358a81e8b45bbabb2563c8b45e1f'
NOTEBOOK_SHA = '47ab99bde2a64d6afc1d0f4a7df2d501cc77c19ecb582c36ed418745db24a399'
METADATA_SHA = '013c8e2365c42cb0e69cb7fa876088a788e217951a0326b501988213bf6f93e0'
NATIVE = 'tools/native_codex_dispatch_record_v1.py'
NATIVE_SHA = 'a00b801f8336a4fa48f70724e0350d5d2337b433b8a0796d987063dde8cbccaa'
CPU_DLL_SHA = '712cde267b942a6dbbea6e860a8a53372ad71ee6aadd94cca6bd1a424ab7e14b'
CPU_DLL_PATH = r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
FINAL_ONLY_SOURCES = ('tools/kaggle_submit_once.py',)
CPU_PARTS = ('model','spawn','raw-visible','raw-holdout','cached-reference')
TIMING_POLICY = dict(schema='k2-runtime-policy/v1.0.1',population=1300,gate_seconds=32400,
    formula='max(fixed) + 1300*max(linear_seconds/studies) + 1300*1301*max(journal_seconds/(studies*(studies+1)))',
    fixed='measured bootstrap startup/install/asset hashing + worker configure/import/asset hashing/model load only',
    linear='codec/header/roster + cohort renderer setup + study bodies + export + all non-fixed/non-journal residual',
    journal='cumulative full-prefix journal writes; quadratic in cumulative row count',
    wall='max(measured bootstrap wall, B-attested remote completed-started wall); reconcile worker phases',
    uncertainty='Official approximately1300 scenario, not a population upper bound; node/workload uncertainty retained')
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
NEW = ('tools/l1_rco_k2_v102.py', 'tools/l1_rco_k2_worker_v101.py',
       'tools/strict_variant_gate_k2_v102.py', 'tools/l1_rco_k2_admission_v102.py',
       'tools/kaggle_submit_once_k2_v102.py', '.local/run-scripts/b-preflight-k2-v102.py',
       'tools/test_l1_rco_k2_v102.py')
FROZEN = ('tools/strict_variant_gate.py', 'tools/strict_variant_gate_standalone.py',
          'tools/strict_variant_gate_hb177.py', 'tools/hb177_release_admission_v1.py',
          'tools/strict_variant_gate_r224_v145.py', 'tools/strict_variant_gate_hb151.py',
          NATIVE, 'tools/s6plus.py',
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
worker_started = time.monotonic()
subprocess.run([sys.executable, '-B', str(assets/'source/l1_rco_k2_worker_v101.py'),
    '--assets', str(assets), '--manifest-sha', expected, '--data', str(data),
    '--template', str(data/'sample_submission.csv'), '--output', str(out/'submission.csv'),
    '--scratch', str(out/'k2-owned-scratch'), '--device', 'cuda:0'], check=True)
(out/'bootstrap-timing.json').write_text(json.dumps({'schema':'k2-bootstrap/v1',
    'total_seconds':time.monotonic()-started, 'fixed_startup_seconds':worker_started-started,
    'manifest_sha256':expected, 'timing_policy':'k2-runtime-policy/v1.0.1'}))
print('K2 real inference complete; independent qualification still required')
'''.replace('MANIFEST_SHA', repr(manifest_sha))
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3')),
        cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source.splitlines(True))])


def n32_notebook(reg):
    """Only the exact registered train/template/output harness; no source-policy drift."""
    nb=notebook(reg['asset_manifest_sha256'])
    source=''.join(nb['cells'][0]['source'])
    source=source.replace("out = pathlib.Path('/kaggle/working')", "out = pathlib.Path('/kaggle/working/k2-n32')\nout.mkdir()\nimport csv\nwith (out/'template.csv').open('w',newline='') as f:\n    cw=csv.writer(f);cw.writerow("+repr([w.UID]+w.LABELS)+")\n    cw.writerows([[uid]+['']*12 for uid in "+repr(reg['n32_ids'])+"])" )
    source=source.replace("str(data/'sample_submission.csv')", "str(out/'template.csv')")
    source=source.replace("'--device', 'cuda:0']", "'--device', 'cuda:0', '--split', 'train']")
    nb['cells'][0]['source']=source.splitlines(True)
    return nb


def harness_diff(reg):
    return ''.join(difflib.unified_diff(w.encoded(notebook(reg['asset_manifest_sha256'])).decode().splitlines(True),
        w.encoded(n32_notebook(reg)).decode().splitlines(True),fromfile='release.ipynb',tofile='n32.ipynb'))








def historical(inputs=None):
    """Read immutable parent metadata only; never replay its live source paths."""
    def read(p):
        data=p.read_bytes() if inputs is None else inputs.read(p.relative_to(ROOT))
        return data
    for p,h in ((OLD_STAGE/'registration.json',OLD_REG_SHA),(OLD_STAGE/'source-manifest.json',OLD_MANIFEST_SHA),
                (OLD_STAGE/'kernel/release.ipynb',NOTEBOOK_SHA),(OLD_STAGE/'kernel/kernel-metadata.json',METADATA_SHA)):
        w.need(hashlib.sha256(read(p)).hexdigest()==h,'immutable v101 metadata identity')
    return json.loads(read(OLD_STAGE/'registration.json')),json.loads(read(OLD_STAGE/'source-manifest.json'))


def required_sources(inputs=None):
    _,old=historical(inputs)
    renamed={n.replace('_v102.py','_v101.py').replace('-v102.py','-v101.py'):n for n in NEW}
    names={renamed.get(n,n) for n in old if n not in FINAL_ONLY_SOURCES}|set(NEW+FROZEN)
    names|={relative(OLD_STAGE/n) for n in ('registration.json','source-manifest.json','kernel/release.ipynb','kernel/kernel-metadata.json')}
    w.need(not set(FINAL_ONLY_SOURCES)&names,'canonical must remain final-only')
    return names


def source_inventory():
    _,old=historical()
    renamed={n.replace('_v102.py','_v101.py').replace('-v102.py','-v101.py'):n for n in NEW}
    sources={n:w.sha(ROOT/n) for n in sorted(required_sources())}
    for n,h in old.items():
        if n not in FINAL_ONLY_SOURCES and n not in renamed:
            w.need(sources[n]==h,'unchanged BUILD source drift: '+n)
    w.need(sources[NATIVE]==NATIVE_SHA and sources['tools/l1_rco_k2_worker_v101.py']==WORKER_SHA,'exact native/worker source')
    return sources


def exclusive_json(path,value):
    """New evidence only, bounded Windows sharing retries; never overwrite."""
    data=w.encoded(value);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(40):
        try:
            with path.open('xb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
            return
        except PermissionError:
            if attempt==39:raise
            time.sleep(min(.05*(attempt+1),1))


def register():
    w.need(not STAGE.exists(),'stage exists; immutable create-only registration')
    old,_=historical();sources=source_inventory()
    w.need(w.sha(ASSETS/'asset-manifest.json')==ASSET_SHA and relative(ASSETS)==old['assets'],'same immutable asset path/hash')
    am=w.assets_check(ASSETS,ASSET_SHA)
    w.need(w.VERSION==old['version'],'runtime identity unchanged')
    STAGE.mkdir(parents=True);(STAGE/'kernel').mkdir()
    exclusive_json(STAGE/'source-manifest.json',sources)
    reg=dict(old,schema='k2-registration/v1.0.2',provenance_contract=PROVENANCE_CONTRACT,
        stage_version=STAGE_VERSION,builder_version=VERSION,version=w.VERSION,worker_version=w.VERSION,asset_version=w.VERSION,
        stage=relative(STAGE),workspace=str(ROOT.resolve()),final_only_sources=list(FINAL_ONLY_SOURCES),
        worker_source='tools/l1_rco_k2_worker_v101.py',worker_sha256=WORKER_SHA,
        source_manifest_sha256=w.sha(STAGE/'source-manifest.json'),
        source_stage_provenance=dict(path=relative(OLD_STAGE/'registration.json'),sha256=OLD_REG_SHA),
        qualified_for_submission=False,fixture_only=False)
    exclusive_json(STAGE/'registration.json',reg)
    for name in ('release.ipynb','kernel-metadata.json'):
        with (STAGE/'kernel'/name).open('xb') as stream:stream.write((OLD_STAGE/'kernel'/name).read_bytes())
    origins=w.read(ASSET_WORK/'asset-origins.json')
    delta=[]
    for name in NEW:
        before=name.replace('_v102.py','_v101.py').replace('-v102.py','-v101.py')
        delta.extend(difflib.unified_diff((ROOT/before).read_text().splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile=before,tofile=name))
    for name,pin in sorted(w.read(ASSETS/'asset-manifest.json')['files'].items()):
        w.need(w.sha(ASSETS/name)==pin,'retained asset equality')
        delta.append('UNCHANGED ASSET '+name+' '+pin+'\n')
    for name in ('release.ipynb','kernel-metadata.json'):
        w.need((STAGE/'kernel'/name).read_bytes()==(OLD_STAGE/'kernel'/name).read_bytes(),'retained notebook/metadata equality')
        delta.append('UNCHANGED KERNEL '+name+' '+w.sha(STAGE/'kernel'/name)+'\n')
    (STAGE/'full-inventory-diff.txt').write_text(''.join(delta),encoding='utf-8')
    (STAGE/'CHANGELOG.md').write_text('# K2 v1.0.2-rcof0\n\nProspective BUILD/final transport separation; stage/build version differs from unchanged runtime1.0.1-rcof0. Fresh five-part CPU evidence and complete actual native provenance. Live canonical is the only final-only source. Scientific/runtime/assets/notebook unchanged. No inherited measurements relabelled.\n',encoding='utf-8')
    exclusive_json(STAGE/'embedded-file-manifest.json',dict(new_sources={n:sources[n] for n in NEW},inherited_assets=origins,
        removed=[],changed_inherited=[],notebook_sha256=w.sha(STAGE/'kernel/release.ipynb'),metadata_sha256=w.sha(STAGE/'kernel/kernel-metadata.json'),
        asset_manifest_sha256=ASSET_SHA,complete_diff_sha256=w.sha(STAGE/'full-inventory-diff.txt')))
    sealed={p.relative_to(STAGE).as_posix():w.sha(p) for p in STAGE.rglob('*') if p.is_file()}
    (STAGE/'SHA256SUMS.txt').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(sealed.items())),encoding='utf-8')
    return dict(status='BUILT_NOT_ADMITTED',stage=relative(STAGE),kernel_sha256=w.digest({n:w.sha(STAGE/'kernel'/n) for n in ('kernel-metadata.json','release.ipynb')}),qualified_for_submission=False)


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


def cpu_smoke(part, destination, deployment_libraries=False, admission=None):
    from tools import strict_variant_gate_k2_v102 as gate
    gate.replay_cpu_admission(admission, part, destination)
    # The documented B invocation imports this already-pinned environment helper
    # before this builder, exactly as in the completed historical CPU runs.
    driver_module=sys.modules.get('tools.s6plus')
    w.need(driver_module is not None and driver_module.PINNED_DRIVER_SHA256==CPU_DLL_SHA and
        os.environ.get('S6PLUS_PIN_DLL')==CPU_DLL_PATH,'reviewed process-local CPU DLL environment required')
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
    from datetime import datetime, timezone
    import uuid
    common=dict(schema='k2-cpu-smoke/v1.0.2',part=part,registration_sha256=w.sha(STAGE/'registration.json'),
        builder_source_sha256=w.sha(ROOT/'tools/l1_rco_k2_v102.py'), worker_sha256=w.sha(ROOT/'tools/l1_rco_k2_worker_v101.py'),
        runtime_version=w.VERSION, provenance_contract=PROVENANCE_CONTRACT,
        driver_sha256=driver_module.PINNED_DRIVER_SHA256,driver_path=os.environ['S6PLUS_PIN_DLL'],
        run_id=uuid.uuid4().hex, started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        output_directory=relative(destination), cpu_admission=dict(path=relative(admission),sha256=w.sha(admission)),
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
                        runtime_version=w.VERSION,worker_sha256=WORKER_SHA,manifest_sha256=manifest_sha,
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        model_sha256=w.MODEL_SHA,roster_sha256=reg['n32_roster_sha256'],
                        output_sha256=rec['reference_sha256'],contract=w.CONTRACT,tensors=hashes,
                        cache_index_sha256=rec['cache_index_sha256'],environment=w.environment('cpu'),
                        registration_sha256=common['registration_sha256'],source_sha256=w.sha(Path(__file__))))
            else:
                raise ValueError('unknown CPU smoke part')
        rec.update(seconds=time.perf_counter()-started,environment=w.environment('cpu'),completed_at=datetime.now(timezone.utc).isoformat())
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
    from tools import strict_variant_gate_k2_v102 as gate
    from tools import strict_variant_gate as base
    inputs=base.Inputs(ROOT);stage=STAGE
    gate.snapshot(stage,inputs)
    records={}
    for path in paths:
        p=Path(path);rec=inputs.json(p)
        w.need(rec['part'] not in records,'duplicate fresh CPU part')
        records[rec['part']]=dict(path=relative(p),sha256=inputs.digest(p))
    w.need(set(records)==set(CPU_PARTS),'incomplete CPU smoke set')
    raw_path=ROOT/records['raw-holdout']['path'];cached_path=ROOT/records['cached-reference']['path']
    raw=inputs.json(raw_path.parent/'worker-receipt.json');cached=inputs.json(cached_path)
    w.need([{k:s[k] for k in ('images_sha256','metadata_sha256','windows')} for s in raw['studies']]==cached['tensors'],'raw/cache tensors')
    reg=inputs.json(stage/'registration.json');ids=reg['smoke_holdout_ids']
    comparison=parity(prediction_table(raw_path.parent/'submission.csv',ids),prediction_table(cached_path.parent/'reference.csv',ids),'same_runtime')
    result=dict(schema='k2-cpu-collection/v1.0.2',provenance_contract=PROVENANCE_CONTRACT,status='PASS',
        registration_sha256=inputs.digest(stage/'registration.json'),builder_source_sha256=inputs.digest('tools/l1_rco_k2_v102.py'),
        manifest_sha256=reg['asset_manifest_sha256'],parts=records,same_runtime=comparison,raw_cached_images_exact=True,qualified_for_submission=False)
    gate.cpu_collection(result,inputs,stage);inputs.recheck();exclusive_json(output,result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('register');sub.add_parser('validate')
    q=sub.add_parser('cpu-smoke');q.add_argument('--part',choices=[*CPU_PARTS,'cached-reference-n32'],required=True)
    q.add_argument('--output',type=Path,required=True);q.add_argument('--admission',type=Path,required=True);q.add_argument('--deployment-libraries',action='store_true')
    q=sub.add_parser('collect-cpu');q.add_argument('--receipts',nargs='+',required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=='register':result=register()
    elif a.command=='validate':
        from tools import strict_variant_gate_k2_v102 as gate
        from tools import strict_variant_gate as base
        inputs=base.Inputs(ROOT);gate.snapshot(STAGE,inputs);inputs.recheck()
        result=dict(status='STATIC_VALIDATED_NOT_ADMITTED',qualified_for_submission=False)
    elif a.command=='cpu-smoke':result=cpu_smoke(a.part,a.output,a.deployment_libraries,a.admission)
    else:result=collect_cpu(a.receipts,a.output)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','kernel_sha256','stage','seconds','studies','keys','qualified_for_submission')},indent=2))


if __name__=='__main__':main()
