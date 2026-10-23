"""H-B-144 descriptive public-member OAI adapter, v1.0.1.

Changelog 2026-10-13 v1.0.1: accept serving-defined zero-background slice
masks after successful source decode; two bounded independent CPU study workers.
Changelog 2026-10-13 v1.0.0: audited original DICOM -> frozen R serving;
strict checkpoint identities, two-study smoke, fixed-panel descriptive bootstrap.
No training, downloads, competition operations, or original-panel admission edits.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import types

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

VERSION = '1.0.1'
SCRATCH = ROOT / '.local/scratch/astra-x1/H-B-144'
PANEL = ROOT / 'artifacts/experiments/OAI_MOAKS_PANEL_V1'
FULL = PANEL / ('full-v103/public-hb144-v' + VERSION)
SERVING = ROOT / 'artifacts/experiments/EXP169_RELEASE_V5Q/v1.2.0/visible/a5_172/exp155-assets/public-serving.py'
SERVING_SHA = '25ef6ab9d2e942ac291433a498246d0507481c2aefa5630cdd601d0a36852cf2'
HELPER = ROOT / 'tools/exp152_own0949_release.py'
HELPER_SHA = 'ebb8e06734b72f40f090389b23d8767137c8eb6c92dee547367c1f18188f94e2'
RELEASE_IDENTITY = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0/input-identities.json'
PANEL_SHA = '81e2ed2a0befd47258bd89730d627bf0610fe470316f986076fbbfa68f828dae'
AUDIT_SHA = '423c8c4f6fcc87a1cd441535a9e31c5d05719f9192da18a80000be25a4ab6320'
MEMBERS = {
    'R': ('.local/rplus-v1/public-R-download.bin', '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'),
    'Rplus_s43': ('artifacts/experiments/H_B_109_RPLUS_PUBLIC/v1.0.0/member/outputs/model_best.pt', '1ddbc89c0f233d7f37c4811250d05f7446e2aeb4af8b522f45d9d1106b07b359'),
    'Rplus_s44': ('artifacts/experiments/H_B_109_RPLUS_PUBLIC/v1.1.0/member/outputs/model_best.pt', '2b69a0552666d74b0b3871f3cb199303777621a3c090e322372f8fed0617a4a4'),
}
TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
MAPPED = (0, 2, 3, 4, 5, 6, 7, 8, 9)
WEAK = (3, 6, 5, 8)
SEED, DRAWS = 20261003, 1000
MAPPING = {
    'SAG_IW_TSE': ('Sagittal', 1, 'IW_TSE_FS'),
    'COR_IW_TSE': ('Coronal', 0, 'IW_TSE_NON_FS'),
    'COR_MPR': ('Coronal', 1, 'DESS_WE'),
    'AX_MPR': ('Axial', -1, 'DESS_WE'),
}
BLOCKED = {
    'R7': 'Local bundle 8c7b666e27dc0e8f (manifest c4bb0f7399ada9a0...), common code and anatomy assets not supplied in permitted references; native OAI input mapping unverified.',
    'S6': 'Local bundle 209a32c8b50a6b1c, common code and anatomy assets not supplied in permitted references; native OAI input mapping unverified.',
    'R2': 'Seed combination formula/provenance not supplied; raw s43/s44 are retained, no guessed formula.',
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, allow_nan=False) + '\n'
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(data, encoding='utf-8')
    os.replace(tmp, path)


def output_path(path):
    path = Path(path).resolve()
    need(path.is_relative_to(SCRATCH) or path == FULL or path.is_relative_to(FULL), 'Output outside fixed task scratch/new versioned panel folder')
    return path


def panel_module():
    from tools import oai_moaks_panel
    return oai_moaks_panel


def ids_only():
    need(sha(PANEL / 'panel.csv') == PANEL_SHA, 'Frozen panel ID hash changed')
    with (PANEL / 'panel.csv').open(newline='') as f:
        r = csv.DictReader(f)
        need(r.fieldnames == ['study_key'], 'ID-only panel schema differs')
        ids = [x['study_key'] for x in r]
    need(len(ids) == 557 and ids == sorted(set(ids)), 'Panel size/order differs')
    return ids


def runtime():
    import torch
    import cv2
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    need(sha(SERVING) == SERVING_SHA and sha(HELPER) == HELPER_SHA, 'Frozen R serving/helper changed')
    mod = types.ModuleType('hb144_frozen_r')
    mod.__file__ = str(SERVING)
    exec(compile(SERVING.read_text(encoding='utf-8'), str(SERVING), 'exec'), mod.__dict__)
    spec = importlib.util.spec_from_file_location('hb144_e152', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    mod._E152 = helper
    mod._E152_FILES = {k: ROOT / v[0] for k, v in MEMBERS.items()}
    mod._E152_MANIFEST = {'checkpoints': {k: {'sha256': v[1]} for k, v in MEMBERS.items()}}
    mod.HDR_THREADS = mod.PX_THREADS = 1
    need(tuple(mod.LAB) == TARGETS and mod.K_EVAL == 94 and mod.IMG == 384 and mod.INPUT_RES == 320, 'R serving contract changed')
    return mod


def audit():
    from tools.extcache import oai_cache_v2
    a = oai_cache_v2.load_audit()
    need(a.summary_sha256 == AUDIT_SHA, 'OAI routing audit changed')
    return a


def records(a, uid):
    groups = sorted((g for g in a.groups.values() if g['study_key'] == uid and g['eligible']), key=lambda g: g['group_key'])
    need(groups and len({g['series_key'] for g in groups}) == len(groups), 'Ambiguous split series in audit; review required')
    rows = []
    for g in groups:
        need(MAPPING.get(g['source_sequence']) == (g['plane'], g['fluid'], g['contrast_family']), 'Unregistered OAI contrast/plane mapping')
        need(g['contrast_verified'], 'Unverified contrast')
        rows.append(dict(StudyInstanceUID=uid, SeriesInstanceUID=g['series_key'], Anatomical_Plane=g['plane'], Fluid_Sensitive=g['fluid'], group_key=g['group_key']))
    return rows


def routes(mod, rows):
    used, result = set(), []
    for plane, fluid, n in mod.SLOTS:
        r = mod._pick_series_for_slot(rows, plane, fluid, used)
        if r:
            used.add(r['SeriesInstanceUID'])
        result.append(dict(plane=plane, fluid=fluid, slices=n, series=None if r is None else r['SeriesInstanceUID'], group=None if r is None else r['group_key']))
    return result


def source_pins():
    from tools.extcache import oai_cache_v2
    identity = read(RELEASE_IDENTITY)['r']
    need(identity['sha256'] == MEMBERS['R'][1] and identity['serving_pins']['public-serving.py'] == SERVING_SHA and
         identity['serving_pins']['exp152_own0949_release.py'] == HELPER_SHA, 'C85-R7 R serving identity differs')
    paths = [Path(__file__), SERVING, HELPER, RELEASE_IDENTITY, ROOT / 'tools/oai_moaks_panel.py',
             PANEL / 'PROTOCOL.md', PANEL / 'registration.json', PANEL / 'runtime-v1.0.3.json', PANEL / 'vendor/manifest.json']
    paths += [ROOT / p for p in oai_cache_v2.CODE_FILES]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def register():
    ids = panel_module().validate()
    need(ids == ids_only(), 'Scorer cohort differs')
    a, mod = audit(), runtime()
    rs = {u: routes(mod, records(a, u)) for u in ids}
    for name, (p, digest) in MEMBERS.items():
        need(sha(ROOT / p) == digest, 'Checkpoint changed: ' + name)
    value = dict(version=VERSION, scope='DESCRIPTIVE_PUBLIC_OAI_ANCESTRY_NOT_OAI_FREE', ids=ids,
                 panel_sha256=PANEL_SHA, targets=TARGETS, mapped_targets=[TARGETS[j] for j in MAPPED],
                 seed=SEED, draws=DRAWS, sources=source_pins(), audit=a.binding(), routes=rs,
                 members=MEMBERS, blocked=BLOCKED, mapping=MAPPING,
                 r2_formula=None, rank='average 1-based rank / N over full 557 study cohort',
                 series_order='lexicographic audited group_key; frozen serving chooses first unused preferred contrast, then first unused plane',
                 missing_slots=sum(r['series'] is None for rr in rs.values() for r in rr))
    write(SCRATCH / 'registration.json', value)
    return dict(status='REGISTERED_DESCRIPTIVE_ONLY', studies=len(ids), missing_slots=value['missing_slots'], blocked=BLOCKED)


def validate():
    reg = read(SCRATCH / 'registration.json')
    need(reg['version'] == VERSION and reg['ids'] == ids_only(), 'Adapter registration/cohort changed')
    need(reg['sources'] == source_pins(), 'Adapter source dependency drift; register before review')
    need(reg['blocked'] == BLOCKED and reg['members'] == {k: list(v) for k, v in MEMBERS.items()}, 'Member contract changed')
    for name, (path, digest) in MEMBERS.items():
        need(sha(ROOT / path) == digest, 'Checkpoint changed: ' + name)
    need(panel_module().validate() == reg['ids'], 'Frozen panel validation failed')
    return reg


def extract_group(a, g, destination):
    """Copy only audited DICOM bytes, without rewriting headers or unsafe tar paths."""
    import pydicom
    ar = a.archives[g['series_key']]
    p = Path(ar['path'])
    st = p.stat()
    need(st.st_size == ar['size'] and st.st_mtime_ns == ar['mtime_ns'], 'Raw archive identity changed')
    wanted = dict(zip(g['members'], g['sops']))
    need(len(wanted) == g['frame_count'], 'Duplicate audit frame')
    destination.mkdir(parents=True, exist_ok=True)
    found = {}
    with tarfile.open(p, 'r|gz') as tf:
        for member in tf:
            name = PurePosixPath(member.name.replace('\\', '/')).as_posix()
            if member.isfile() and name in wanted:
                need(name not in found, 'Duplicate archive member')
                content = tf.extractfile(member).read()
                d = pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True)
                need(str(d.SOPInstanceUID) == wanted[name], 'Audit SOP mismatch')
                need(hasattr(d, 'ImageOrientationPatient') and hasattr(d, 'ImagePositionPatient') and hasattr(d, 'PixelSpacing'), 'Missing source geometry')
                target = destination / (hashlib.sha256(name.encode()).hexdigest() + '.dcm')
                digest = hashlib.sha256(content).hexdigest()
                if target.exists():
                    need(sha(target) == digest, 'Staged original DICOM changed')
                else:
                    target.write_bytes(content)
                found[name] = digest
    need(set(found) == set(wanted), 'Audited frames missing')
    need(len(list(destination.glob('*.dcm'))) == len(wanted), 'Unexpected staged DICOM')
    return dict(archive=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns, frames=len(found),
                frame_bytes_sha256=hashlib.sha256(json.dumps(found, sort_keys=True).encode()).hexdigest())


def validate_volume(vol, mask, rr):
    need(vol.shape == (96, 384, 384) and vol.dtype == np.uint8 and mask.shape == (96,), 'Invalid R volume')
    expected = (vol.reshape(96, -1).sum(1) > 0).astype(np.uint8)
    need(np.array_equal(mask, expected) and mask.any(), 'Mask differs from frozen serving content rule')
    at, background = 0, 0
    for r in rr:
        slot = mask[at:at+r['slices']]
        need(bool(slot.any()) if r['series'] else not bool(slot.any()), 'Unexpected decoded/absent slot mask')
        if r['series']:
            background += int((slot == 0).sum())
        at += r['slices']
    need(at == 96, 'Slot depth differs')
    return background


def prepare(*, full=False, dry_run=False):
    reg, a, mod = validate(), audit(), runtime()
    ids = reg['ids'] if full else reg['ids'][:2]
    t0 = time.monotonic()
    receipts = []
    if dry_run:
        archives = {}
        for uid in ids:
            rr = routes(mod, records(a, uid))
            need(rr == reg['routes'][uid], 'Study route changed')
            for r in rr:
                if r['series']:
                    ar = a.archives[r['series']]
                    st = Path(ar['path']).stat()
                    need(st.st_size == ar['size'] and st.st_mtime_ns == ar['mtime_ns'], 'Raw archive identity changed')
                    archives[ar['path']] = st.st_size
        result = dict(status='PASS_PREPARE_DRY_RUN', studies=len(ids), full_panel=full, archives=len(archives),
                      archive_bytes=sum(archives.values()), pixels_decoded=False, seconds=time.monotonic()-t0)
        write(SCRATCH / 'prepare-dry-run.json', result)
        return result
    def prepare_one(uid):
        rows = records(a, uid)
        rr = routes(mod, rows)
        need(rr == reg['routes'][uid], 'Study route changed')
        source = {}
        for r in rr:
            if r['group']:
                source[r['group']] = extract_group(a, a.groups[r['group']], SCRATCH / 'dicom/test_images' / uid / r['series'])
        reader = mod._make_reader()
        # Original reader swallows errors; preflight every selected image before calling it.
        for r in rr:
            if r['series']:
                files, _ = reader[0](str(SCRATCH / 'dicom/test_images' / uid / r['series']))
                need(files, 'No input files')
                for f, ps in files:
                    arr = reader[1](f)
                    need(arr.ndim == 2 and np.isfinite(arr).all() and ps > 0, 'Invalid DICOM pixels/spacing')
        vol, mask = mod.build_study(uid, {uid: rows}, str(SCRATCH / 'dicom/test_images'), reader)
        background = validate_volume(vol, mask, rr)
        dest = SCRATCH / 'prepared' / (uid + '.npz')
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, vol=vol, mask=mask)
        receipt = dict(status='PASS', uid=uid, registration_sha256=sha(SCRATCH / 'registration.json'),
                       sha256=sha(dest), routes=rr, sources=source, available_slices=int(mask.sum()), background_slices=background,
                       preprocessing='unchanged pinned R build_study/_make_reader; no cached Sag0 copy')
        write(str(dest) + '.json', receipt)
        return dict(uid=uid, available_slices=int(mask.sum()), background_slices=background, sha256=receipt['sha256'])
    # Two study workers plus at most one frozen reader worker each; BLAS/OpenCV/
    # torch remain one-threaded. No shared study destinations or CUDA operations.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for receipt in pool.map(prepare_one, ids):
            receipts.append(receipt)
            print(json.dumps(dict(prepared=len(receipts), total=len(ids))), flush=True)
    result = dict(status='PASS_PREPARE', full_panel=full, studies=len(ids), seconds=time.monotonic()-t0,
                  background_slices=sum(r['background_slices'] for r in receipts), outputs=receipts)
    write(SCRATCH / ('prepare-full.json' if full else 'prepare-two.json'), result)
    return result


def volume(uid):
    p = SCRATCH / 'prepared' / (uid + '.npz')
    r = read(str(p) + '.json')
    need(r['status'] == 'PASS' and r['uid'] == uid and r['registration_sha256'] == sha(SCRATCH / 'registration.json') and r['sha256'] == sha(p), 'Prepared input no longer authentic')
    with np.load(p, allow_pickle=False) as z:
        return z['vol'], z['mask']


def probs(p, n):
    p = np.asarray(p, dtype=np.float64)
    need(p.shape == (n, 12) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(), 'Invalid predictions')
    return p


def cpu_smoke():
    import torch
    validate()
    mod = runtime()
    uid = ids_only()[0]
    vol, mask = volume(uid)
    plain = mod.eval_windows(vol, mask, 94, 320)
    mirror = mod.eval_windows(vol, mask, 94, 320, mirror=True)
    need(plain.shape == mirror.shape == (94, 3, 320, 320), 'Window shape differs')
    t0, loaded = time.monotonic(), []
    for name in MEMBERS:
        model, _ = mod.load_model(ROOT / MEMBERS[name][0], mod._E152.ARCH, 384, 'cpu')
        ck = mod._E152.safe_load(ROOT / MEMBERS[name][0])
        need(tuple(ck['lab']) == TARGETS, 'Checkpoint target order differs')
        del ck
        with torch.inference_mode():
            # Engineering smoke only: original forward, five real windows per TTA.
            # Qualified serving remains the unchanged CUDA/fp16 infer_pair_probs.
            ix = torch.linspace(0, 93, 5).round().long()
            p = model(plain[ix].unsqueeze(0)).float().sigmoid()
            q = model(mirror[ix].unsqueeze(0)).float().sigmoid()
            pred = probs((.5 * (p + q)).numpy(), 1)
        loaded.append(dict(member=name, finite=True, nonconstant=bool(np.ptp(pred) > 0)))
        del model
        gc.collect()
    need(not torch.cuda.is_initialized(), 'CPU command initialized CUDA')
    r = dict(status='PASS_CPU_SMOKE', studies=1, windows_per_tta=5, precision='fp32',
             qualified_serving_parity=False, members=loaded, seconds=time.monotonic()-t0, cuda_initialized=False)
    write(SCRATCH / 'cpu-smoke.json', r)
    return r


def smoke_binding():
    p = SCRATCH / 'cuda-two/receipt.json'
    if not p.exists():
        return None
    r = read(p)
    if (r.get('status') != 'PASS_TWO_STUDY_CUDA_SMOKE' or r.get('studies') != 2 or r.get('full_panel') is not False or
        r.get('cpu_cuda_windows_exact') is not True or r.get('registration_sha256') != sha(SCRATCH / 'registration.json') or
        r.get('tool_sha256') != sha(__file__) or [m['member'] for m in r.get('members', [])] != list(MEMBERS)):
        return None
    return sha(p)


def admit(review=None):
    reg = validate()
    result = dict(status='HOLD_FULL_RUN_REQUIRES_INDEPENDENT_REVIEW', available=list(MEMBERS), blocked=BLOCKED,
                  registration_sha256=sha(SCRATCH / 'registration.json'), tool_sha256=sha(__file__),
                  cuda_smoke_sha256=smoke_binding())
    if review:
        need(result['cuda_smoke_sha256'] is not None, 'Successful current two-study CUDA smoke required before full admission')
        v = read(review)
        need(v.get('verdict') == 'PASS' and v.get('independent') is True and v.get('reviewer') and
             v.get('scope') == 'DESCRIPTIVE_PUBLIC_OAI_ANCESTRY_NOT_OAI_FREE' and
             v.get('registration_sha256') == result['registration_sha256'] and
             v.get('tool_sha256') == result['tool_sha256'] and v.get('members') == list(MEMBERS) and
             v.get('cuda_smoke_sha256') == result['cuda_smoke_sha256'], 'Independent review missing/stale/wrong scope')
        result.update(status='PASS_AVAILABLE_MEMBERS_ONLY', review_sha256=sha(review))
    return result


def write_predictions(directory, name, ids, values, extra):
    directory = output_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / (name + '.csv')
    values = probs(values, len(ids))
    if directory.is_relative_to(FULL):
        need(not p.exists() and not Path(str(p) + '.json').exists(), 'Versioned full output exists')
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['study_key', *TARGETS])
        w.writerows([u, *map(float, v)] for u, v in zip(ids, values))
    receipt = dict(version=VERSION, scope='DESCRIPTIVE_PUBLIC_OAI_ANCESTRY_NOT_OAI_FREE', key=name,
                   predictions_sha256=sha(p), rows=len(ids), full_panel=ids == ids_only(),
                   registration_sha256=sha(PANEL / 'registration.json'), protocol_sha256=sha(PANEL / 'PROTOCOL.md'),
                   adapter_registration_sha256=sha(SCRATCH / 'registration.json'), tool_sha256=sha(__file__), **extra)
    write(str(p) + '.json', receipt)
    return receipt


def infer(*, full=False, review=None):
    import torch
    reg = validate()
    if full:
        need(review is not None and admit(review)['status'] == 'PASS_AVAILABLE_MEMBERS_ONLY', 'B-only full inference needs independent PASS')
    ids = reg['ids'] if full else reg['ids'][:2]
    need(full or len(ids) == 2, 'Only two-study CUDA smoke is permitted')
    for uid in ids:
        need((SCRATCH / 'prepared' / (uid + '.npz')).is_file() and
             (SCRATCH / 'prepared' / (uid + '.npz.json')).is_file(), 'Prepare every requested study before CUDA launch')
    mod, t0 = runtime(), time.monotonic()
    destination = FULL if full else SCRATCH / 'cuda-two'
    telemetry = []
    for name, (path, digest) in MEMBERS.items():
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        model, _ = mod.load_model(ROOT / path, mod._E152.ARCH, 384, 'cuda')
        result = []
        for uid in ids:
            vol, mask = volume(uid)
            with torch.inference_mode():
                x = mod.gpu_windows(vol, mask, 94, 320, 'cuda')
                xm = mod.gpu_windows(vol, mask, 94, 320, 'cuda', mirror=True)
                if not full:
                    # CUDA LUT and CPU reference must agree bit-for-bit, both TTA branches.
                    need(torch.equal(x.cpu(), mod.eval_windows(vol, mask, 94, 320)) and
                         torch.equal(xm.cpu(), mod.eval_windows(vol, mask, 94, 320, mirror=True)), 'CPU/CUDA window parity failed')
                result.append(mod.infer_pair_probs(model, x, xm, 'cuda'))
                del x, xm
        torch.cuda.synchronize()
        info = dict(member=name, seconds=time.monotonic()-begin,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_predictions(destination, name, ids, result, dict(device='cuda', checkpoint_sha256=digest,
                          input_receipts={u: sha(str(SCRATCH / 'prepared' / (u + '.npz')) + '.json') for u in ids}, telemetry=info))
        telemetry.append(info)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    receipt = dict(status='PASS_FULL_AVAILABLE_MEMBERS' if full else 'PASS_TWO_STUDY_CUDA_SMOKE', studies=len(ids),
                   full_panel=full, seconds=time.monotonic()-t0, device=torch.cuda.get_device_name(),
                   cpu_cuda_windows_exact=not full, members=telemetry, blocked=BLOCKED,
                   registration_sha256=sha(SCRATCH / 'registration.json'), tool_sha256=sha(__file__))
    write(destination / 'receipt.json', receipt)
    return receipt


def rank(p):
    from scipy.stats import rankdata
    return rankdata(p, method='average', axis=0) / len(p)


def blends(members):
    """Only construct named fixed blends when every required real member exists."""
    result = dict(members)
    for name, a, b, w in [('P956', 'R', 'S6', .5), ('P_R7', 'R', 'R7', .5),
                          ('R_R7_40_60', 'R', 'R7', .4), ('R_R7_30_70', 'R', 'R7', .3),
                          ('R2_in_P_R7', 'R2', 'R7', .5)]:
        if a in members and b in members:
            result[name] = w * rank(members[a]) + (1-w) * rank(members[b])
    return result


def statistics(y, members, *, draws=DRAWS, seed=SEED):
    """Existing target/mask/AUC; shared study bootstrap, fixed-nine/weak-four means."""
    from scipy.stats import spearmanr
    scorer = panel_module()
    need(y.ndim == 2 and y.shape[1] == 12 and draws > 0, 'Outcome/draw schema differs')
    members = {k: probs(v, len(y)) for k, v in members.items()}
    all_p = blends(members)
    names = list(all_p)
    def aucs(ix):
        return np.asarray([[scorer.auc(y[ix, j], all_p[k][ix, j]) for j in range(12)] for k in names])
    point = aucs(np.arange(len(y)))
    samples = np.empty((draws, len(names), 12))
    rng = np.random.default_rng(seed)
    for b in range(draws):
        samples[b] = aucs(rng.integers(0, len(y), len(y)))
    eligible = np.asarray([min((y[:, j] == 0).sum(), (y[:, j] == 1).sum()) >= 20 for j in range(12)])
    def estimate(value, boot, ok):
        valid = boot[np.isfinite(boot)]
        good = bool(ok and len(valid) >= .95 * draws and np.isfinite(value))
        return dict(auc=float(value) if np.isfinite(value) else None,
                    ci95=np.percentile(valid, [2.5, 97.5]).tolist() if good else None,
                    valid_draws=len(valid), undefined_draws=draws-len(valid), eligible=good)
    summaries = {}
    for k, name in enumerate(names):
        summaries[name] = dict(per_target={t: estimate(point[k, j], samples[:, k, j], eligible[j]) for j, t in enumerate(TARGETS)},
            macro9=estimate(point[k, list(MAPPED)].mean(), samples[:, k, list(MAPPED)].mean(1), eligible[list(MAPPED)].all()),
            weak4=estimate(point[k, list(WEAK)].mean(), samples[:, k, list(WEAK)].mean(1), eligible[list(WEAK)].all()))
    correlations = {}
    for a, b in itertools.combinations(members, 2):
        per = {}
        for j, t in enumerate(TARGETS):
            x, z = members[a][:, j], members[b][:, j]
            per[t] = float(spearmanr(x, z).statistic) if np.ptp(x) > 0 and np.ptp(z) > 0 else None
        correlations[a + ':' + b] = per
    # Protocol's paired-delta analysis, same draws/seed and no varying-target mean.
    comparisons = {}
    for a, b in [('P_R7', 'P956'), ('R_R7_40_60', 'P_R7'), ('R_R7_30_70', 'P_R7'), ('R2_in_P_R7', 'P_R7')]:
        if a in names and b in names:
            comparisons[a + '_minus_' + b] = scorer.paired(y, all_p[a][None], all_p[b][None], draws=draws, seed=seed)
    return dict(scope='DESCRIPTIVE_ONLY_NO_PROMOTION_OR_MACRO12_SAFETY', studies=len(y), draws=draws, seed=seed,
                macro_targets=[TARGETS[j] for j in MAPPED], members=summaries, spearman=correlations,
                paired_comparisons=comparisons, blocked=BLOCKED)


def read_prediction(p, ids):
    p = Path(p)
    receipt = read(str(p) + '.json')
    need(receipt['full_panel'] and receipt['rows'] == 557 and receipt['predictions_sha256'] == sha(p), 'Partial/changed predictions cannot be scored')
    need(receipt['adapter_registration_sha256'] == sha(SCRATCH / 'registration.json') and
         receipt['protocol_sha256'] == sha(PANEL / 'PROTOCOL.md') and receipt['tool_sha256'] == sha(__file__), 'Prediction contract drift')
    need(receipt['key'] == p.stem and p.stem in MEMBERS and receipt['device'] == 'cuda' and
         receipt['checkpoint_sha256'] == MEMBERS[p.stem][1] and
         receipt['registration_sha256'] == sha(PANEL / 'registration.json'), 'Prediction member identity differs')
    need(receipt['input_receipts'] == {u: sha(str(SCRATCH / 'prepared' / (u + '.npz')) + '.json') for u in ids},
         'Prediction input receipt drift')
    with p.open(newline='') as f:
        r = csv.DictReader(f)
        need(r.fieldnames == ['study_key', *TARGETS], 'Prediction schema/order changed')
        rows = list(r)
    need([r['study_key'] for r in rows] == ids, 'Prediction cohort/order differs')
    return probs([[float(r[t]) for t in TARGETS] for r in rows], len(ids))


def score():
    reg = validate()
    members = {name: read_prediction(FULL / (name + '.csv'), reg['ids']) for name in MEMBERS}
    result = statistics(panel_module().outcomes(reg['ids']), members)
    need(not (FULL / 'analysis.json').exists(), 'Analysis exists; make a new version')
    write(FULL / 'analysis.json', result)
    return dict(status='SCORED_AVAILABLE_MEMBERS_DESCRIPTIVE_ONLY', studies=557, members=list(members), blocked=BLOCKED)


def analysis_smoke():
    """Real frozen own-model outputs exercise statistics, not public-member claims."""
    validate()
    scorer, ids = panel_module(), ids_only()
    sources = [PANEL / 'full-v103' / (k + '.csv') for k in ('EXP179_M42', 'EXP179_M43')]
    predictions = {p.stem: scorer.prediction(p, ids)[0] for p in sources}
    t0 = time.monotonic()
    result = statistics(scorer.outcomes(ids), predictions)
    result.update(status='PASS_ANALYSIS_SMOKE_OWN_MODELS_ONLY', seconds=time.monotonic()-t0,
                  source_hashes={p.name: sha(p) for p in sources})
    write(SCRATCH / 'analysis-smoke.json', result)
    return dict(status=result['status'], studies=len(ids), draws=DRAWS, seconds=result['seconds'],
                macro9={k: v['macro9'] for k, v in result['members'].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['register', 'validate', 'prepare', 'cpu-smoke', 'analysis-smoke', 'admit', 'infer', 'score'])
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--review', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.command == 'register': result = register()
    elif a.command == 'validate':
        result = dict(status='PASS_VALIDATION', studies=len(validate()['ids']), blocked=BLOCKED)
    elif a.command == 'prepare': result = prepare(full=a.full, dry_run=a.dry_run)
    elif a.command == 'cpu-smoke': result = cpu_smoke()
    elif a.command == 'analysis-smoke': result = analysis_smoke()
    elif a.command == 'admit':
        need(a.dry_run, 'admit requires --dry-run; no job launch')
        result = admit(a.review)
    elif a.command == 'infer': result = infer(full=a.full, review=a.review)
    else: result = score()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
