"""EXP169 v2 local GPU references for the visible Kaggle T4 parity check.

Claude opus for B, 2026-10-10. Version 1.1.0.

Changelog v1.1.0: add a staged B174 EXP174 own-anchor CPU/fp32 reference path;
the v1.0.0 partner and B172 own-anchor outputs are unchanged.

Subcommands
  partner        20-member partner reference on the local GPU from the staged runtime:
                 local-refs/partner-raw.npz (ids, l2, dino, convnext; the kernel layout),
                 partner-members.npz (ids + one (n, 12) array per member, for the parity
                 skill's compare_arrays.py), partner-runtime.json, partner-receipt.json.
  own-anchor     B172 own-only EXP143-v2 anchor, run the way the kernel runs it:
                 local-refs/b172-own-anchor/{raw.npz, inputs.json, runtime.json,
                 binding.json, submission.csv, receipt.json} written by the frozen EXP152
                 runtime, plus local-refs/b172-own-anchor.local-receipt.json.
  b174-anchor174 B174 EXP174 own-only anchor on CPU/fp32, using the staged
                 EXP152 serving code and authenticated B174 binding.
  split-members  any partner-raw.npz (local or kernel) -> per-member npz.
  members        CPU proof that two stages carry the same 20 partner members.

The staged runtime is imported from the stage folder, never copied. Local-only
changes are explicit text substitutions on the frozen sources, recorded with
hashes: one local consumer instead of two T4 consumers (partner), and a one-thread
decode pool instead of a fork pool (own-anchor; Windows has no fork). Backend: the
release policy plus the T4-effective settings pre-registered in
pre-reading/partner-crossgpu-tolerance.json (TF32 off, flash SDPA off; T4 has
neither). No network, no Kaggle, no labels, no credentials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import zipfile

VERSION = '1.1.0'
ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'artifacts/experiments/EXP169_RELEASE_V2'
OUT_ROOT = EXP / 'local-refs'
REGISTRATION = EXP / 'pre-reading/partner-crossgpu-tolerance.json'
TOL_JSON = EXP / 'pre-reading/partner-crossgpu-tol.json'
STAGES = {'A172': ROOT / 'artifacts/releases/EXP169_A172_v2_0_2_stage',
          'B172': ROOT / 'artifacts/releases/EXP169_B172_v2_0_2_stage'}
COMPETITION = ROOT / 'storage/competition'
DICOM_ROOT = ROOT / ('storage/downloads/rsna-knee-abnormality-detection/prepared/'
                     '9175e113e4a00fe679c4088b/test_series')
FROZEN_T4_INPUTS = [ROOT / 'artifacts/releases/EXP155_S42v1_1_stage_v12' / name / 'inputs.json'
                    for name in ('blend3-t4-output', 'own-t4-output')]
UID = 'StudyInstanceUID'
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
FAMILIES = (('l2', 10), ('dino', 5), ('convnext', 5))
PARTNER_FILES = ('partner-raw.npz', 'partner-members.npz', 'partner-runtime.json', 'partner-receipt.json')
ANCHOR_FILES = ('raw.npz', 'inputs.json', 'runtime.json', 'binding.json', 'submission.csv')
ENTRY_FIELDS = ('family', 'seed', 'fold', 'tag', 'path', 'sha256', 'source_sha256', 'bytes', 'format')
PARTNER_PREFIXES = ('models/', 'code/', 'config/')
RUNTIME_MODULES = ('raptor_common', 'raptor_dicom', 'raptor_runtime', 'raptor_radimagenet',
                   'raptor_data', 'exp160_base_cpu_pos', 'exp160_base_patch_mil')
OWN_NAME = 'exp143-v2-e14-e11-e12-bn.pt'
EXP152_FILE = 'exp152_own0949_release.py'
OWN_ARCHIVE_MEMBERS = {OWN_NAME, EXP152_FILE, 'public-serving.py', 'release-manifest.json'}
B174_STAGE = ROOT / 'artifacts/releases/EXP169_B174_v3_0_0_verified_stage'
B174_OWN_NAME = 'exp174-jev-e14-e11-e12-bn.pt'
B174_HELPER = 'exp174_own0949_jev_release.py'
B174_ARCHIVE_MEMBERS = {B174_OWN_NAME, B174_HELPER, EXP152_FILE,
                        'public-serving.py', 'release-manifest.json'}

# Frozen exp169_partner_bundle_release.partner_scores: Kaggle-only two-T4 fan-out ->
# one local consumer. Preprocessing, per-member inference, flip TTA and the
# score layout are executed from the staged source text unchanged.
PARTNER_PATCHES = (
    ("    need(torch.cuda.device_count() == 2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2)),\n"
     "         'Exactly two visible T4 GPUs required')",
     "    need(torch.cuda.device_count() >= 1, 'Local CUDA GPU required')  # EXP169 local reference: one consumer"),
    ('    loaded_barrier = threading.Barrier(2)', '    loaded_barrier = threading.Barrier(1)'),
    ('gpu_peak_bytes=[0, 0], completed=[0, 0])', 'gpu_peak_bytes=[0], completed=[0])'),
    ('            for _ in range(2):\n                work_queue.put(None)',
     '            for _ in range(1):\n                work_queue.put(None)'),
    ('    workers = [threading.Thread(target=consumer, args=(i,), daemon=True) for i in range(2)]',
     '    workers = [threading.Thread(target=consumer, args=(i,), daemon=True) for i in range(1)]'),
)
# Frozen EXP152 public-serving.py: Linux fork pool -> one decode thread (CPU decode
# scheduling only; the decoded uint8 volume/mask are hashed into inputs.json).
SERVING_PATCHES = (
    ('mp.get_context("fork").Pool(nproc, initializer=_pre_init)',
     '_EXP169_LOCAL_POOL(nproc, initializer=_pre_init)'),
)
B174_CPU_PATCHES = (
    ('    dev = "cuda" if torch.cuda.is_available() else "cpu"', '    dev = "cpu"  # local reference only'),
    ('    ngpu = torch.cuda.device_count()', '    ngpu = 0  # never use the busy local GPU'),
    ('    assert str(device).startswith("cuda"), "Qualified inference requires CUDA/fp16"',
     '    assert str(device) == "cpu", "Local B174 reference requires CPU"'),
    ('    with torch.autocast("cuda", dtype=torch.float16):',
     '    with torch.autocast("cpu", enabled=False):  # local fp32 reference'),
    ('FR_HDR, FR_PX = 32, 16', 'FR_HDR, FR_PX = 4, 2  # preserve training loader CPU headroom'),
)
B174_BIND_PATCHES = (
    ('    need(torch.cuda.is_available(), "CUDA/fp16 is required")',
     '    need(_EXP169_LOCAL_DEVICE == "cpu", "Local B174 CPU binding required")'),
)
# Frozen fastread module (base64 inside public-serving.py): POSIX-only raw I/O ->
# portable equivalents. O_BINARY is 0 on Linux, and the pread fallback is used only
# where os.pread is missing, so on the Kaggle host both lines behave as before.
FASTREAD_PATCHES = (
    ('import numpy as np\n\nCROP_MM = 140.0',
     'import numpy as np\n\n_EXP169_PREAD_LOCK = threading.Lock()\n\n\n'
     'def _exp169_pread(fd, n, off):\n'
     '    """EXP169 local shim: os.pread is POSIX-only; serialized seek + read."""\n'
     '    with _EXP169_PREAD_LOCK:\n'
     '        os.lseek(fd, off, os.SEEK_SET)\n'
     '        return os.read(fd, n)\n\n\nCROP_MM = 140.0'),
    ('_O_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)\n',
     '_O_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)\n'),
    ('    b = os.pread(fd, n, off)\n',
     '    b = os.pread(fd, n, off) if hasattr(os, "pread") else _exp169_pread(fd, n, off)\n'),
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def replace_once(source, old, new):
    need(source.count(old) == 1, 'Frozen source anchor changed: ' + old[:90])
    return source.replace(old, new)


def patch_source(source, patches):
    patched = source
    for old, new in patches:
        patched = replace_once(patched, old, new)
    return patched, dict(original_sha256=text_sha(source), patched_sha256=text_sha(patched),
                         replacements=[dict(old=old, new=new) for old, new in patches])


# ----------------------------------------------------------------- partner layout

def member_layout():
    """(key, family, row) in the frozen consumer layout: row = fold + 5 for L2 seed 43."""
    out = []
    for family, rows in FAMILIES:
        for row in range(rows):
            seed, fold = (43, row - 5) if family == 'l2' and row >= 5 else (42, row)
            out.append((f'{family}_s{seed}_f{fold}', family, row))
    return out


def layout_from_manifest(entries):
    """Bind the static member keys to the manifest with the frozen row rule."""
    found = []
    for e in entries:
        row = e['fold'] + (5 if e['family'] == 'l2' and e['seed'] == 43 else 0)
        found.append((f"{e['family']}_s{e['seed']}_f{e['fold']}", e['family'], row))
    need(sorted(found) == sorted(member_layout()) and len(found) == 20,
         'Manifest partner members differ from the 20-member layout')
    return found


def check_partner_arrays(arrays, n=None):
    """Kernel partner-raw.npz contract: ids + three (members, n, 12) families in [0, 1]."""
    import numpy as np
    need(set(arrays) == {'ids', 'l2', 'dino', 'convnext'}, 'Partner keys differ: ' + repr(sorted(arrays)))
    ids = [str(x) for x in np.asarray(arrays['ids']).tolist()]
    need(np.asarray(arrays['ids']).ndim == 1 and len(ids) > 0 and len(set(ids)) == len(ids),
         'Partner ids invalid')
    need(n is None or len(ids) == n, 'Partner cohort size differs')
    for family, rows in FAMILIES:
        a = np.asarray(arrays[family])
        need(a.shape == (rows, len(ids), 12) and a.dtype.kind == 'f' and np.isfinite(a).all() and
             ((a >= 0) & (a <= 1)).all(), 'Partner family invalid: ' + family)
    return ids


def split_members(raw, out):
    """Per-member npz: ids + <family>_s<seed>_f<fold> (n, 12), same dtype, no rounding."""
    import numpy as np
    raw, out = Path(raw), Path(out)
    with np.load(raw, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    ids = check_partner_arrays(arrays)
    members = {key: np.ascontiguousarray(arrays[family][row]) for key, family, row in member_layout()}
    with out.open('xb') as stream:
        np.savez(stream, ids=np.asarray(arrays['ids']), **members)
    return dict(source=str(raw), source_sha256=sha(raw), out=str(out), sha256=sha(out),
                studies=len(ids), members=[key for key, _, _ in member_layout()])


def summarize(scores):
    import numpy as np
    return {family: dict(shape=list(np.asarray(scores[family]).shape), dtype=str(np.asarray(scores[family]).dtype),
                         finite=bool(np.isfinite(scores[family]).all()),
                         min=float(np.min(scores[family])), max=float(np.max(scores[family])))
            for family, _ in FAMILIES}


# ----------------------------------------------------------------- stage identity

def stage_identity(stage):
    stage = Path(stage)
    receipt = read_json(stage / 'build-receipt.json')
    manifest_path = stage / 'assets/release-manifest.json'
    need(receipt.get('status') == 'PLAN_ONLY_NOT_PUSHED' and receipt.get('candidate') in ('A172', 'B172') and
         receipt.get('manifest_sha256') == sha(manifest_path) and
         read_json(manifest_path).get('candidate') == receipt['candidate'],
         'Stage receipt/manifest binding differs: ' + str(stage))
    return dict(path=str(stage.resolve()), candidate=receipt['candidate'], version=receipt.get('version'),
                build_receipt_sha256=sha(stage / 'build-receipt.json'),
                manifest_sha256=receipt['manifest_sha256'])


def partner_projection(manifest):
    entries = [{k: e.get(k) for k in ENTRY_FIELDS} for e in manifest['entries']]
    files = {k: v for k, v in sorted(manifest['files'].items()) if k.startswith(PARTNER_PREFIXES)}
    need(len(entries) == 20 and any(k.startswith('models/') for k in files), 'Partner manifest incomplete')
    return dict(entries=entries, files=files, k_eval=manifest.get('k_eval'))


def partner_identity(manifests):
    """All manifests must carry the same 20 members, runtime code and config."""
    digests = {}
    for name, manifest in manifests.items():
        blob = json.dumps(partner_projection(manifest), sort_keys=True, separators=(',', ':'))
        digests[name] = hashlib.sha256(blob.encode()).hexdigest()
    need(len(set(digests.values())) == 1, 'Partner members differ between stages: ' + repr(digests))
    return dict(digest=next(iter(digests.values())), stages=sorted(digests))


def verify_checkpoints(work, manifest):
    """Fail loudly unless every partner checkpoint matches the manifest bytes and SHA256."""
    out = []
    for e in manifest['entries']:
        path = Path(work) / e['path']
        need(path.is_file(), 'Missing partner checkpoint: ' + e['path'])
        need(path.stat().st_size == e['bytes'] and sha(path) == e['sha256'] and
             manifest['files'][e['path']]['sha256'] == e['sha256'],
             'Partner checkpoint SHA/bytes mismatch vs manifest: ' + e['path'])
        out.append(dict(path=e['path'], sha256=e['sha256'], bytes=e['bytes'], tag=e['tag']))
    return out


def registration_binding():
    reg = read_json(REGISTRATION)
    need(reg.get('status') == 'PRE_REGISTERED_BEFORE_ANY_LOCAL_REFERENCE_OUTPUT' and
         reg['tol_json']['sha256'] == sha(TOL_JSON), 'Partner cross-GPU registration missing or changed')
    return dict(path=str(REGISTRATION), sha256=sha(REGISTRATION), created_at=reg['created_at'],
                tol_json=str(TOL_JSON), tol_json_sha256=sha(TOL_JSON))


# ----------------------------------------------------------------- visible inputs

def _column(path, column):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), [row[column] for row in reader]


def tree_digest(root):
    root = Path(root)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    h = hashlib.sha256()
    total = 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        total += size
        h.update(f'{rel}\0{size}\0{sha(p)}\n'.encode())
    return dict(files=len(files), bytes=total, sha256=h.hexdigest())


def visible_inputs(competition, dicom_root):
    """Visible roster in sample-submission order (the kernel's serve order); no labels read."""
    competition, dicom_root = Path(competition), Path(dicom_root)
    header, ids = _column(competition / 'sample_submission.csv', UID)
    _, test_ids = _column(competition / 'test.csv', UID)
    _, series_ids = _column(competition / 'test_series.csv', UID)
    need(header == [UID, *TARGETS], 'Sample submission columns differ')
    need(ids and len(set(ids)) == len(ids) and ids == test_ids and set(series_ids) == set(ids),
         'Visible roster differs between sample, test and test_series')
    need(sorted(p.name for p in dicom_root.iterdir() if p.is_dir()) == sorted(ids),
         'DICOM study folders differ from the visible roster')
    return ids, dict(competition=str(competition.resolve()), dicom_root=str(dicom_root.resolve()),
                     csv_sha256={name: sha(competition / name) for name in
                                 ('test.csv', 'test_series.csv', 'sample_submission.csv')},
                     dicom_tree=tree_digest(dicom_root), studies=len(ids))


def competition_view(folder, competition, dicom_root):
    """Kaggle-like competition root (CSV copies + DICOM copy); sources stay untouched."""
    folder = Path(folder)
    folder.mkdir(parents=True)
    for name in ('test.csv', 'test_series.csv', 'sample_submission.csv'):
        shutil.copyfile(Path(competition) / name, folder / name)
    shutil.copytree(dicom_root, folder / 'test_series')
    return folder


# ----------------------------------------------------------------- backend/runtime

def t4_effective_backend(torch, benchmark=None, deterministic=None):
    """T4 (sm75) has no TF32 and no flash SDPA: disable both on the local GPU."""
    if benchmark is not None:
        torch.backends.cudnn.benchmark = benchmark
    if deterministic is not None:
        torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    return backend_snapshot(torch)


def backend_snapshot(torch):
    return dict(cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                flash_sdp=bool(torch.backends.cuda.flash_sdp_enabled()),
                mem_efficient_sdp=bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                math_sdp=bool(torch.backends.cuda.math_sdp_enabled()),
                threads=torch.get_num_threads(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def runtime_record(torch):
    import importlib.metadata as md
    packages = {}
    for name in ('torch', 'timm', 'numpy', 'scipy', 'pandas', 'pydicom', 'opencv-python',
                 'opencv-python-headless', 'pylibjpeg', 'pillow'):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                devices=[dict(name=torch.cuda.get_device_name(i),
                              capability=list(torch.cuda.get_device_capability(i)))
                         for i in range(torch.cuda.device_count())],
                backend=backend_snapshot(torch),
                env={k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG', 'KNEE_WORKSPACE_ROOT',
                                                    'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED')})


def require_free_vram(torch, gib):
    need(torch.cuda.is_available(), 'CUDA GPU required')
    free, total = torch.cuda.mem_get_info(0)
    need(free >= gib * 2**30, f'Insufficient free VRAM: {free / 2**30:.2f} GiB < {gib} GiB '
                              '(queue behind the running GPU job)')
    return dict(free_bytes=int(free), total_bytes=int(total))


# ----------------------------------------------------------------- staged imports

def _load_module(name, path):
    path = Path(path).resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        need(Path(existing.__file__).resolve() == path, f'{name} already imported from {existing.__file__}')
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, 'Cannot import ' + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_staged_partner(stage):
    """Import the staged v2 wrapper and frozen base from <stage>/assets/code only."""
    code = (Path(stage) / 'assets/code').resolve()
    for name in RUNTIME_MODULES:
        module = sys.modules.get(name)
        need(module is None or Path(module.__file__).resolve().parent == code,
             f'{name} already imported from outside the stage; use a fresh process')
    if not sys.path or Path(sys.path[0]).resolve() != code:
        sys.path.insert(0, str(code))
    base = _load_module('exp169_partner_bundle_release', code / 'exp169_partner_bundle_release.py')
    alias = 'tools.exp169_partner_bundle_release'
    prior = sys.modules.get(alias)
    need(prior is None or prior is base, 'Project base module already imported; use a fresh process')
    sys.modules[alias] = base  # the wrapper's first import path must resolve to the staged file
    try:
        v2 = _load_module('exp169_release_v2', code / 'exp169_release_v2.py')
    finally:
        if prior is None:
            sys.modules.pop(alias, None)
    need(v2.base is base and Path(v2.__file__).resolve().parent == code, 'Staged wrapper/base binding differs')
    return base, v2, code


def staged_module_files(code):
    out = {}
    for name in ('exp169_partner_bundle_release', 'exp169_release_v2', *RUNTIME_MODULES):
        module = sys.modules.get(name)
        if module is None:
            continue
        path = Path(module.__file__).resolve()
        need(path.parent == Path(code).resolve(), f'{name} executed from outside the stage: {path}')
        out[name] = dict(path=str(path), sha256=sha(path))
    return out


def local_partner_function(base, v2, overrides=None):
    """Frozen partner_scores text with the PARTNER_PATCHES fan-out substitutions only.

    ``overrides`` is a CPU-test seam (fake model calls); the real run passes none.
    """
    source = inspect.getsource(base.partner_scores)
    patched, info = patch_source(source, PARTNER_PATCHES)
    scope = {**base.__dict__, '_load_models': v2._load_models, **(overrides or {})}
    exec(compile(patched, '<exp169-local-partner>', 'exec'), scope)
    return scope['partner_scores'], info


def install_fastread(ns):
    """Pre-write the patched frozen fastread where _load_fastread looks (it writes only if absent)."""
    import base64
    raw = base64.b64decode(ns['_FASTREAD_B64'])
    patched, info = patch_source(raw.decode('utf-8'), FASTREAD_PATCHES)
    info['original_bytes_sha256'] = hashlib.sha256(raw).hexdigest()
    need(not os.path.isdir('/kaggle/working'), 'Unexpected /kaggle/working on the local host')
    path = Path(os.path.dirname(os.path.abspath(ns['__file__']))) / '_fastread.py'
    need(not path.exists(), 'Stale _fastread.py beside the extracted serving source')
    path.write_bytes(patched.encode('utf-8'))
    module = ns['_load_fastread']()
    need(Path(module.__file__).resolve() == path.resolve() and hasattr(module, '_exp169_pread'),
         'Serving runtime did not load the patched fastread module')
    info['path'] = str(path)
    return info


def decode_records(ns, view, ids):
    """CPU decode exactly as _pre_init/_pre_one (kernel series table, study_lut builder)."""
    import pandas as pd
    tser = pd.read_csv(str(view) + '/test_series.csv')
    tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
    tser['SeriesInstanceUID'] = tser['SeriesInstanceUID'].astype(str)
    series = {k: v.to_dict('records') for k, v in tser.groupby('StudyInstanceUID')}
    reader = ns['_load_fastread']().make_builder(ns['FASTREAD_VARIANT'], ns['IMG'], ns['SLOTS'],
                                                 hdr_threads=ns['FR_HDR'], px_threads=ns['FR_PX'])
    out = {}
    for uid in ids:
        vol, mask = reader(uid, series, str(view) + '/test_series')
        need(mask.any(), 'Empty decoded study: ' + uid)
        out[uid] = ns['_E152'].input_record(ns, vol, mask, None, None, detailed=False)
    return out


def _single_thread_pool(processes, initializer=None):
    from multiprocessing.pool import ThreadPool
    return ThreadPool(1, initializer=initializer)


# ----------------------------------------------------------------- partner

def finalize_partner(out_root, ids, scores, runtime, provenance):
    """Write the four partner files atomically; refuse any overwrite."""
    import numpy as np
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    targets = [out_root / name for name in PARTNER_FILES]
    need(not any(p.exists() for p in targets), 'Fresh partner outputs required')
    check_partner_arrays({'ids': np.asarray(ids), **scores}, n=len(ids))
    tmp = out_root / '.partner-incomplete'
    tmp.mkdir()
    with (tmp / 'partner-raw.npz').open('xb') as stream:  # same call as the frozen serve
        np.savez(stream, ids=np.asarray(ids), **scores)
    members = split_members(tmp / 'partner-raw.npz', tmp / 'partner-members.npz')
    write_json(tmp / 'partner-runtime.json', runtime)
    receipt = dict(schema='exp169-v2-local-partner-ref-v1', status='COMPLETE_LOCAL_PARTNER_REFERENCE',
                   tool='tools/exp169_v2_local_refs.py', tool_version=VERSION, tool_sha256=sha(__file__),
                   created_at=now(), ids=list(ids), studies=len(ids), summary=summarize(scores),
                   member_keys=members['members'],
                   files={name: sha(tmp / name) for name in PARTNER_FILES[:3]}, **provenance)
    write_json(tmp / 'partner-receipt.json', receipt)
    for name in PARTNER_FILES:
        need(not (out_root / name).exists(), 'Partner output appeared during the run: ' + name)
        os.replace(tmp / name, out_root / name)
    tmp.rmdir()
    return receipt


def run_partner(stage, other_stage, competition, dicom_root, out_root, min_free_gib, t4_effective=True,
                dry_run=False):
    t0, cpu0 = time.monotonic(), time.process_time()
    stage, out_root = Path(stage).resolve(), Path(out_root)
    need(not any((out_root / name).exists() for name in PARTNER_FILES), 'Fresh partner outputs required')
    registration = registration_binding()
    identities = [stage_identity(stage)] + ([stage_identity(other_stage)] if other_stage else [])
    manifests = {i['candidate']: read_json(Path(i['path']) / 'assets/release-manifest.json') for i in identities}
    identity = partner_identity(manifests)
    ids, inputs = visible_inputs(competition, dicom_root)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['KNEE_WORKSPACE_ROOT'] = str(stage / 'assets')
    base, v2, code = load_staged_partner(stage)
    work = stage / 'assets'
    manifest = v2.read_json(work / 'release-manifest.json')
    need(sha(work / 'release-manifest.json') == identities[0]['manifest_sha256'], 'Stage manifest changed')
    v2.verify_assets(work, manifest)
    checkpoints = verify_checkpoints(work, manifest)
    layout = layout_from_manifest(manifest['entries'])
    partner, patch = local_partner_function(base, v2)
    if dry_run:  # every CPU check above passed; no torch import, no output
        return dict(status='DRY_RUN_OK', candidate_stages=identities, partner_identity=identity,
                    checkpoints_verified=len(checkpoints), local_patches=patch, inputs=inputs,
                    staged_modules=staged_module_files(code), registration=registration)
    import torch
    vram = require_free_vram(torch, min_free_gib)
    with tempfile.TemporaryDirectory(prefix='exp169-localrefs-partner-', ignore_cleanup_errors=True) as tmp:
        view = competition_view(Path(tmp) / 'competition', competition, dicom_root)
        need(tree_digest(view / 'test_series') == inputs['dicom_tree'], 'DICOM copy differs from source')
        v2.reset_backend(torch)  # the wrapper's own reset, as in exp169_release_v2.partner_scores
        backend = t4_effective_backend(torch) if t4_effective else backend_snapshot(torch)
        began = time.monotonic()
        scores, metrics = partner(work, view, ids, manifest, began + 3600)
        partner_seconds = time.monotonic() - began
    modules = staged_module_files(code)
    runtime = runtime_record(torch)
    runtime.update(staged_runtime_snapshot=sys.modules['raptor_runtime'].runtime_snapshot(torch),
                   backend_at_inference=backend, t4_effective_backend=t4_effective)
    provenance = dict(
        candidate_stages=identities, partner_identity=identity, registration=registration,
        staged_modules=modules, checkpoints_verified=checkpoints, layout=layout,
        local_patches=dict(partner_scores=patch, note='two-T4 check and two-consumer fan-out -> one local consumer'),
        policy=dict(autocast='fp16', cudnn_benchmark=False, cudnn_deterministic=True,
                    deterministic_algorithms=True, k_eval=manifest['k_eval'],
                    tta='flip RadR50/ConvNeXt-T, plain DINO', t4_effective_backend=t4_effective),
        inputs=inputs, vram_before=vram, metrics=metrics, partner_seconds=partner_seconds,
        wall_seconds=time.monotonic() - t0, cpu_seconds=time.process_time() - cpu0)
    return finalize_partner(out_root, ids, scores, runtime, provenance)


# ----------------------------------------------------------------- own anchor

def notebook_source(path):
    nb = read_json(path)
    source = nb['cells'][-1]['source']
    return ''.join(source) if isinstance(source, list) else source


def check_anchor_folder(folder, binding, ids):
    """Own-only anchor contract the release tool reads (CPU; numpy + csv)."""
    import numpy as np
    folder = Path(folder)
    receipt = read_json(folder / 'receipt.json')
    need(receipt.get('status') == 'COMPLETE_INFERENCE' and set(receipt.get('files', {})) == set(ANCHOR_FILES),
         'Anchor receipt incomplete')
    for name in ANCHOR_FILES:
        need(sha(folder / name) == receipt['files'][name], 'Anchor file hash differs: ' + name)
    need(read_json(folder / 'binding.json') == binding, 'Anchor binding differs from the stage manifest')
    with np.load(folder / 'raw.npz', allow_pickle=False) as raw:
        need(set(raw.files) == {'ids', 'targets', 'own'} and raw['ids'].tolist() == list(ids) and
             raw['targets'].tolist() == TARGETS, 'Anchor raw roster/targets/arms differ')
        own = raw['own']
        need(own.shape == (len(ids), 12) and np.isfinite(own).all() and ((own >= 0) & (own <= 1)).all(),
             'Anchor own probabilities invalid')
        summary = dict(shape=list(own.shape), dtype=str(own.dtype), finite=True,
                       min=float(own.min()), max=float(own.max()))
    header, csv_ids = _column(folder / 'submission.csv', UID)
    need(header == [UID, *TARGETS] and csv_ids == list(ids), 'Anchor submission header/roster differs')
    inputs = read_json(folder / 'inputs.json')
    need(set(inputs) == set(ids), 'Anchor input diagnostics roster differs')
    return summary


def compare_inputs(local, references, keys=None):
    out = {}
    for ref in references:
        ref = Path(ref)
        if not ref.is_file():
            out[str(ref)] = 'missing'
            continue
        other = read_json(ref)
        if keys is not None:
            other = {uid: {k: v for k, v in rec.items() if k in keys} for uid, rec in other.items()}
        keys_ = {uid: sorted(k for k in local.get(uid, {}) if other.get(uid, {}).get(k) != local[uid][k])
                for uid in sorted(set(local) | set(other))}
        out[str(ref)] = dict(equal=local == other, sha256=sha(ref),
                             differing_keys={u: k for u, k in keys_.items() if k})
    return out


def run_own_anchor(stage, competition, dicom_root, out_root, min_free_gib, t4_effective=True,
                   reference_inputs=FROZEN_T4_INPUTS, dry_run=False, decode_only=False):
    t0, cpu0 = time.monotonic(), time.process_time()
    stage, out_root = Path(stage).resolve(), Path(out_root)
    out = out_root / 'b172-own-anchor'
    local_receipt = out_root / 'b172-own-anchor.local-receipt.json'
    work = out_root / 'b172-own-anchor.incomplete'
    need(not out.exists() and not local_receipt.exists() and not work.exists(), 'Fresh own-anchor output required')
    identity = stage_identity(stage)
    need(identity['candidate'] == 'B172', 'Own-only anchor reference is B172 only')
    manifest = read_json(stage / 'assets/release-manifest.json')
    binding = manifest['anchor_binding']
    own_build = read_json(stage / 'own-anchor/build-receipt.json')
    archive = stage / 'own-anchor/dataset/exp152-assets.zip.bin'
    need(sha(archive) == own_build['archive_sha256'], 'Own-anchor archive differs from its build receipt')
    kernel = notebook_source(stage / 'kernel/release.ipynb')
    ids, inputs = visible_inputs(competition, dicom_root)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if not dry_run and not decode_only:
        import torch
        vram = require_free_vram(torch, min_free_gib)
    with tempfile.TemporaryDirectory(prefix='exp169-localrefs-own-', ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        assets = tmp / 'exp152-assets'  # mirrors /kaggle/working/exp152-assets
        assets.mkdir()
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            need(len(names) == len(OWN_ARCHIVE_MEMBERS) and set(names) == OWN_ARCHIVE_MEMBERS,
                 'Own-anchor archive member mismatch')
            z.extractall(assets)
        own_manifest = read_json(assets / 'release-manifest.json')
        module_sha = own_manifest['asset_sha256'][EXP152_FILE]
        need(sha(assets / 'release-manifest.json') == own_build['manifest_sha256'] and
             sha(assets / EXP152_FILE) == module_sha and
             all(s in kernel for s in (own_build['archive_sha256'], own_build['manifest_sha256'], module_sha)),
             'Extracted own-anchor assets differ from the staged kernel bindings')
        e152 = _load_module('exp152_own0949_release', assets / EXP152_FILE)
        e152.verify_assets(assets, own_manifest)
        need(e152.jhash(own_manifest) == binding['manifest_sha256'], 'Own manifest differs from the anchor binding')
        view = competition_view(tmp / 'competition', competition, dicom_root)
        need(tree_digest(view / 'test_series') == inputs['dicom_tree'], 'DICOM copy differs from source')
        serving = (assets / 'public-serving.py').read_text(encoding='utf-8')
        need(sha(assets / 'public-serving.py') == own_manifest['serving_sha256'] == binding['serving_sha256'],
             'Serving source differs from the anchor binding')
        patched, patch = patch_source(serving, SERVING_PATCHES)
        compile(patched, 'public-serving.py', 'exec')
        if dry_run:  # extraction, bindings and patch checked; no torch import, no output
            return dict(status='DRY_RUN_OK', stage=identity, anchor_binding=binding,
                        own_manifest_jhash=e152.jhash(own_manifest), local_patches=patch, inputs=inputs)
        ns = {'__name__': 'exp169_local_own_anchor', '__file__': str(assets / 'public-serving.py'),
              '_EXP169_LOCAL_POOL': _single_thread_pool}
        if decode_only:  # CPU: decoded uint8 volume/mask/centers vs the frozen T4 inputs.json
            exec(compile(patched, 'public-serving.py', 'exec'), ns)
            fastread = install_fastread(ns)
            e152.bind(ns, own_manifest, {OWN_NAME: assets / OWN_NAME}, 'own', tmp)
            records = decode_records(ns, view, ids)
            return dict(status='DECODE_CHECKED', stage=identity, studies=len(ids),
                        local_patches=dict(public_serving=patch, fastread=fastread), records=records,
                        frozen_t4_input_comparison=compare_inputs(records, reference_inputs,
                                                                  keys=('volume', 'mask', 'centers')),
                        seconds=time.monotonic() - t0)
        work.mkdir(parents=True)
        try:
            exec(compile(patched, 'public-serving.py', 'exec'), ns)
            fastread = install_fastread(ns)
            ns['find_test_root'] = lambda: str(view)  # /kaggle/input discovery -> local view
            policy_flags = backend_snapshot(torch)  # what public-serving.py set at exec
            backend = t4_effective_backend(torch) if t4_effective else policy_flags
            e152.bind(ns, own_manifest, {OWN_NAME: assets / OWN_NAME}, 'own', work)
            began = time.monotonic()
            ns['main']()
            main_seconds = time.monotonic() - began
            summary = check_anchor_folder(work, binding, ids)
        except BaseException:
            if work.exists():
                work.rename(out_root / f'b172-own-anchor.failed-{time.strftime("%Y%m%d-%H%M%S")}')
            raise
        exp152_module = dict(path_in_archive=EXP152_FILE, sha256=module_sha)
    # Kernel serve check with the staged wrapper (anchor_values: receipt, binding, raw roster).
    base, v2, code = load_staged_partner(stage)
    v2_ids, _ = v2.anchor_values(work, Path(competition) / 'sample_submission.csv', manifest)
    need(v2_ids == ids, 'Staged anchor_values roster differs')
    os.replace(work, out)
    receipt = dict(
        schema='exp169-v2-local-own-anchor-ref-v1', status='COMPLETE_LOCAL_OWN_ANCHOR_REFERENCE',
        tool='tools/exp169_v2_local_refs.py', tool_version=VERSION, tool_sha256=sha(__file__),
        created_at=now(), stage=identity, anchor_binding_equal=True, staged_anchor_values='PASS',
        own_anchor_build_receipt_sha256=sha(stage / 'own-anchor/build-receipt.json'),
        archive_sha256=own_build['archive_sha256'], own_manifest_sha256=own_build['manifest_sha256'],
        exp152_module=exp152_module, local_patches=dict(public_serving=patch, fastread=fastread,
            note='fork pool -> one decode thread; POSIX pread/O_BINARY shim in fastread; '
                 'find_test_root -> local Kaggle-like view'),
        backend_policy_flags=policy_flags, backend_at_inference=backend, t4_effective_backend=t4_effective,
        runtime=runtime_record(torch), staged_wrapper=staged_module_files(code), inputs=inputs,
        vram_before=vram, ids=ids, raw_summary=summary,
        files={name: sha(out / name) for name in (*ANCHOR_FILES, 'receipt.json')},
        frozen_t4_input_comparison=compare_inputs(read_json(out / 'inputs.json'), reference_inputs),
        main_seconds=main_seconds, wall_seconds=time.monotonic() - t0, cpu_seconds=time.process_time() - cpu0)
    write_json(local_receipt, receipt)
    return receipt


def b174_cpu_binding(helper):
    """Change only the isolated staged EXP152 binding used by the EXP174 adapter."""
    shared = helper._shared
    need(shared.bind is helper.bind and helper._original_bind.__module__ == shared.__name__,
         'EXP174 no longer uses the pinned EXP152 binding')
    source = inspect.getsource(helper._original_bind)
    patched, info = patch_source(source, B174_BIND_PATCHES)
    scope = {**shared.__dict__, '_EXP169_LOCAL_DEVICE': 'cpu'}
    exec(compile(patched, '<exp169-local-b174-bind>', 'exec'), scope)
    helper._original_bind = scope['bind']
    return info


def b174_cpu_runtime(torch):
    """Avoid CUDA discovery in this CPU diagnostic while recording its actual device."""
    return dict(python=sys.version, torch=str(torch.__version__), device='cpu', precision='fp32',
                threads=torch.get_num_threads(), policy='local CPU reference; CUDA/fp16 release unchanged')


def publish_b174_anchor(work, out, local_receipt, binding, ids, provenance):
    """Validate the frozen EXP152 layout and receipt before publishing a fresh reference."""
    need(not out.exists() and not local_receipt.exists(), 'Fresh B174 anchor output required')
    summary = check_anchor_folder(work, binding, ids)
    os.replace(work, out)
    receipt = dict(schema='exp169-b174-local-anchor174-ref-v1',
                   status='COMPLETE_LOCAL_OWN_ANCHOR_REFERENCE',
                   tool='tools/exp169_v2_local_refs.py', tool_version=VERSION,
                   tool_sha256=sha(__file__), created_at=now(), device='cpu', precision='fp32',
                   ids=ids, raw_summary=summary,
                   files={name: sha(out / name) for name in (*ANCHOR_FILES, 'receipt.json')},
                   **provenance)
    write_json(local_receipt, receipt)
    return receipt


def run_b174_anchor174(stage=B174_STAGE, competition=COMPETITION, dicom_root=DICOM_ROOT,
                       out_root=OUT_ROOT, device='cpu', dry_run=False):
    """Run the authenticated B174 EXP174 archive through staged EXP152 serving on CPU."""
    from tools import exp169_release_v3 as v3
    t0, cpu0 = time.monotonic(), time.process_time()
    need(device == 'cpu', 'B174 local reference currently supports --device cpu only')
    stage, out_root = Path(stage).resolve(), Path(out_root)
    out = out_root / 'b174-anchor174'
    local_receipt = out_root / 'b174-anchor174.local-receipt.json'
    work = out_root / 'b174-anchor174.incomplete'
    need(not out.exists() and not local_receipt.exists() and not work.exists(),
         'Fresh B174 anchor output required')
    build = v3.verify_stage(stage)
    manifest = read_json(stage / 'code/assets/release-manifest.json')
    anchor = manifest['anchors']['EXP174']
    binding = anchor['binding']
    archive = stage / 'anchor174/dataset/exp174-assets.zip.bin'
    need(sha(archive) == anchor['archive_sha256'] == build['anchor174_archive_sha256'],
         'B174 EXP174 archive differs from staged binding')
    ids, inputs = visible_inputs(competition, dicom_root)
    with tempfile.TemporaryDirectory(prefix='exp169-localrefs-b174-', ignore_cleanup_errors=True) as tmpdir:
        tmp = Path(tmpdir)
        assets = tmp / 'exp174-assets'
        assets.mkdir()
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            need(len(names) == len(B174_ARCHIVE_MEMBERS) and set(names) == B174_ARCHIVE_MEMBERS,
                 'B174 EXP174 archive member mismatch')
            z.extractall(assets)
        own_manifest = read_json(assets / 'release-manifest.json')
        need(sha(assets / 'release-manifest.json') == anchor['manifest_sha256'] and
             all(sha(assets / name) == sha(stage / 'anchor174/assets' / name)
                 for name in B174_ARCHIVE_MEMBERS), 'B174 extracted assets differ from staged assets')
        helper = _load_module('exp174_own0949_jev_release', assets / B174_HELPER)
        helper.verify_assets(assets, own_manifest)
        shared = helper._shared
        need(shared.jhash(own_manifest) == binding['manifest_sha256'] and
             sha(assets / B174_OWN_NAME) == anchor['checkpoint_sha256'] == binding['own_sha256'] and
             sha(assets / EXP152_FILE) == own_manifest['asset_sha256'][EXP152_FILE] and
             sha(assets / B174_HELPER) == own_manifest['asset_sha256'][B174_HELPER] and
             sha(assets / 'public-serving.py') == binding['serving_sha256'],
             'B174 EXP174 source/checkpoint/serving binding differs')
        serving = (assets / 'public-serving.py').read_text(encoding='utf-8')
        patched, serving_patch = patch_source(serving, SERVING_PATCHES + B174_CPU_PATCHES)
        compile(patched, '<exp169-local-b174-serving>', 'exec')
        bind_patch = b174_cpu_binding(helper)
        if dry_run:
            return dict(status='DRY_RUN_OK', device=device, stage=build['candidate'],
                        anchor_binding=binding, inputs=inputs,
                        local_patches=dict(public_serving=serving_patch, bind=bind_patch))
        view = competition_view(tmp / 'competition', competition, dicom_root)
        need(tree_digest(view / 'test_series') == inputs['dicom_tree'], 'DICOM copy differs from source')
        import torch
        shared.runtime_info = lambda: b174_cpu_runtime(torch)
        ns = {'__name__': 'exp169_local_b174_anchor174',
              '__file__': str(assets / 'public-serving.py'), '_EXP169_LOCAL_POOL': _single_thread_pool}
        work.mkdir(parents=True)
        try:
            exec(compile(patched, '<exp169-local-b174-serving>', 'exec'), ns)
            fastread = install_fastread(ns)
            ns['find_test_root'] = lambda: str(view)
            helper.bind(ns, own_manifest, {B174_OWN_NAME: assets / B174_OWN_NAME}, 'own', work)
            began = time.monotonic()
            ns['main']()
            main_seconds = time.monotonic() - began
            summary = check_anchor_folder(work, binding, ids)
            need(read_json(work / 'runtime.json')['device'] == 'cpu', 'CPU runtime record missing')
        except BaseException:
            if work.exists():
                work.rename(out_root / f'b174-anchor174.failed-{time.strftime("%Y%m%d-%H%M%S")}')
            raise
    return publish_b174_anchor(work, out, local_receipt, binding, ids, dict(
        stage=str(stage), stage_build_receipt_sha256=sha(stage / 'build-receipt.json'),
        archive_sha256=sha(archive), manifest_sha256=anchor['manifest_sha256'],
        binding_equal=True, inputs=inputs,
        local_patches=dict(public_serving=serving_patch, bind=bind_patch, fastread=fastread,
                           find_test_root='local visible competition view'),
        main_seconds=main_seconds, wall_seconds=time.monotonic()-t0,
        cpu_seconds=time.process_time()-cpu0))


# ----------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('partner')
    a.add_argument('--stage', type=Path, default=STAGES['B172'])
    a.add_argument('--other-stage', type=Path, default=STAGES['A172'])
    a.add_argument('--min-free-gib', type=float, default=6.)
    b = sub.add_parser('own-anchor')
    b.add_argument('--stage', type=Path, default=STAGES['B172'])
    b.add_argument('--min-free-gib', type=float, default=4.)
    for x in (a, b):
        x.add_argument('--competition', type=Path, default=COMPETITION)
        x.add_argument('--dicom-root', type=Path, default=DICOM_ROOT)
        x.add_argument('--out-root', type=Path, default=OUT_ROOT)
        x.add_argument('--native-backend', action='store_true',
                       help='diagnosis only: keep TF32/flash SDPA as the code leaves them')
        x.add_argument('--dry-run', action='store_true', help='all CPU checks; no torch, no output')
    b.add_argument('--decode-only', action='store_true',
                   help='CPU decode of the visible studies vs the frozen T4 inputs.json; no GPU, no output')
    e = sub.add_parser('b174-anchor174')
    e.add_argument('--stage', type=Path, default=B174_STAGE)
    e.add_argument('--competition', type=Path, default=COMPETITION)
    e.add_argument('--dicom-root', type=Path, default=DICOM_ROOT)
    e.add_argument('--out-root', type=Path, default=OUT_ROOT)
    e.add_argument('--device', choices=('cpu',), default='cpu')
    e.add_argument('--dry-run', action='store_true')
    c = sub.add_parser('split-members')
    c.add_argument('--raw', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    d = sub.add_parser('members')
    d.add_argument('--stage', type=Path, action='append', required=True)
    args = p.parse_args(argv)
    print(f'exp169_v2_local_refs v{VERSION} {args.command}', flush=True)
    if args.command == 'partner':
        r = run_partner(args.stage, args.other_stage, args.competition, args.dicom_root, args.out_root,
                        args.min_free_gib, not args.native_backend, args.dry_run)
        result = r if args.dry_run else dict(status=r['status'], studies=r['studies'], summary=r['summary'], files=r['files'],
                      partner_seconds=r['partner_seconds'])
    elif args.command == 'own-anchor':
        r = run_own_anchor(args.stage, args.competition, args.dicom_root, args.out_root,
                           args.min_free_gib, not args.native_backend, dry_run=args.dry_run,
                           decode_only=args.decode_only)
        result = r if args.dry_run or args.decode_only else dict(status=r['status'], raw_summary=r['raw_summary'], files=r['files'],
                      frozen_t4_input_comparison=r['frozen_t4_input_comparison'], main_seconds=r['main_seconds'])
    elif args.command == 'b174-anchor174':
        r = run_b174_anchor174(args.stage, args.competition, args.dicom_root, args.out_root,
                               args.device, args.dry_run)
        result = r if args.dry_run else dict(status=r['status'], device=r['device'],
                                            raw_summary=r['raw_summary'], files=r['files'],
                                            main_seconds=r['main_seconds'])
    elif args.command == 'split-members':
        result = split_members(args.raw, args.out)
    else:
        manifests = {}
        for stage in args.stage:
            identity = stage_identity(stage)
            manifests[identity['path']] = read_json(Path(stage) / 'assets/release-manifest.json')
        result = partner_identity(manifests)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
