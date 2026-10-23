"""OAI L7 readout of the public goodpjw2008 2.5D ConvNeXt MIL reader, v1.0.1.

Changelog
- 1.0.1 (2026-10-17, claude-opus-b4): pre-GPU review FAIL fix (claude-opus-r2): run(), worker() and seal()
  refuse to replay a finished attempt once measurement.json is sealed; GAIN/HARM labels are marked as
  based on the unadjusted 95 % CI; registered limitations list. Backup of 1.0.0:
  .local/scratch/slot-b/oai-reader-sim/oai_reader_readout_v1.0.0.py.bak.
- 1.0.0 (2026-10-17, builder claude-opus-b4): first version. Clean-room
  re-implementation of the reader inference path (the dataset's infer.py /
  knee.py / preprocess.py were read as text only; nothing from the dataset is
  imported or executed). 3 folds, k = 12 windows, res from checkpoint (256),
  fp16 autocast on CUDA as in the source infer.py, fp32 on CPU. Inputs: the
  byte-pinned OAI DICOMs staged by r6_oai_readout_v1 (pilot 96) and
  r6_oai_readout_v110 (confirmation 461), same knee ids and series routes.
  Comparators: authenticated R6flip / R7flip OAI predictions of the same two
  predecessors. R / R224 are OAI-trained and are never compared here.

Subcommands: register | parity | dry-run | validate | estimate | run | _worker |
evaluate [--synthetic | --replay]. Every subcommand except run/_worker forces CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

VERSION = '1.0.1'
TAG = 'OAI_READER_L7_V1'
BUILDER = 'claude-opus-b4'
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
WORK = ROOT / '.local/scratch/slot-b/oai-reader'
TEST_FILE = ROOT / 'tests/test_oai_reader_readout_v1.py'
SYS_PY = r'C:\Users\Taeyang\AppData\Local\Programs\Python\Python312\python.exe'
READER = ROOT / 'storage/external/goodpjw2008-rsna-knee-2-5d-convnext-reader-20261017'
CKPT_SHA = ('0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80',
            '61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb',
            'f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6')
PILOT_W = ROOT / '.local/scratch/astra-b1/r6-oai'              # r6_oai_readout_v1 (pilot 96)
EXT_W = ROOT / '.local/scratch/astra-oai-extension/r6-oai'     # r6_oai_readout_v110 (confirmation 461)
GOLD_DICOM = ROOT / '.local/scratch/claude-hb128/dicom'
HB190 = ROOT / '.local/backups/partial-codex-1017/hb190/scratch/hb190'
K = 12
PARITY_TOL = 1e-4          # fixed before the first parity run (CPU fp32 vs CPU fp32 reference)
IDLE_S = 900               # no-progress watchdog; no wall-clock cap
PEAK_CAP = 12 * 2**30
HEADROOM = 2 * 2**30
FS_RULE = {'1': 1, '0': 0, '-1': 1}
FS_RULE_TEXT = ('Fat_Suppression := Fluid_Sensitive for 0/1 (identical in every competition train_series row); '
                '-1 occurs only for AX_MPR (DESS water-excitation, hb144 MAPPING) -> fat-suppressed (1). '
                'OAI slots: SAG_IW_TSE_FS->0, COR_MPR_DESS_WE->1, AX_MPR_DESS_WE->2, COR_IW_TSE_nonFS->4.')
ARMS = ('Reader', 'R6flip', 'R7flip')
COMPARISONS = (('Reader', 'R6flip'), ('Reader', 'R7flip'), ('R6flip', 'R7flip'))

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
PLANES = ['Sagittal', 'Coronal', 'Axial']
N_SLOTS, SIZE, MM_PER_PX, MAX_SLICES = 6, 384, 0.4, 64
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CANON = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}


class Stop(RuntimeError):
    pass


def need(ok, msg):
    if not ok:
        raise Stop(msg)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return sha_bytes(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode())


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    for i in range(50):
        try:
            os.replace(tmp, path); return obj
        except PermissionError:
            if i == 49: raise
            time.sleep(.1)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now_kst():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def stamp(**kw):
    return dict(identity=TAG, version=VERSION, builder=BUILDER, **kw)


def cpu_only():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
    if 'torch' in sys.modules:
        need(not sys.modules['torch'].cuda.is_initialized(), 'CUDA initialised in a CPU subcommand')


# ----------------------------------------------------------------------------------------------
# Clean-room reader (semantics of preprocess.load_series / knee.KneeNet / infer.run, eval only)
# ----------------------------------------------------------------------------------------------

def read_slices(blob):
    """One DICOM file (bytes) -> list of (frame, iop, ipp, ps, inst); raises on unreadable file."""
    import numpy as np
    import pydicom
    ds = pydicom.dcmread(io.BytesIO(blob))
    arr = ds.pixel_array.astype(np.float32)
    if ds.get('PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
        arr = arr.max() - arr
    slope, inter = ds.get('RescaleSlope'), ds.get('RescaleIntercept')
    if slope is not None and inter is not None:
        arr = arr * float(slope) + float(inter)
    iop = [float(v) for v in ds.ImageOrientationPatient] if 'ImageOrientationPatient' in ds else None
    ipp = [float(v) for v in ds.ImagePositionPatient] if 'ImagePositionPatient' in ds else None
    ps = [float(v) for v in ds.PixelSpacing] if 'PixelSpacing' in ds else None
    inst = float(ds.get('InstanceNumber') or 0)
    frames = list(arr) if arr.ndim == 3 else [arr]
    return [(f, iop, ipp, ps, inst + i * 1e-3) for i, f in enumerate(frames)]


def canon_slice(img, iop, ps, plane):
    import numpy as np
    ps = ps or [MM_PER_PX * img.shape[0] / SIZE] * 2
    if iop is None:
        return img, ps
    r, c = np.array(iop[:3]), np.array(iop[3:])
    _, (cax, csgn), (rax, rsgn) = CANON[plane]
    if abs(r[cax]) < abs(r[rax]):
        img, r, c, ps = img.T, c, r, ps[::-1]
    if r[cax] * csgn < 0:
        img = img[:, ::-1]
    if c[rax] * rsgn < 0:
        img = img[::-1]
    return img, ps


def resample(img, ps):
    import cv2
    import numpy as np
    out = np.zeros((SIZE, SIZE), np.float32)
    h, w = img.shape
    ch = min(h, int(round(SIZE * MM_PER_PX / ps[0])))
    cw = min(w, int(round(SIZE * MM_PER_PX / ps[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    nh = max(1, min(SIZE, int(round(ch * ps[0] / MM_PER_PX))))
    nw = max(1, min(SIZE, int(round(cw * ps[1] / MM_PER_PX))))
    interp = cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR
    res = cv2.resize(crop, (nw, nh), interpolation=interp)
    oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = res
    return out


def series_volume(blobs):
    """blobs: list of file bytes in directory-listing order -> (uint8 [n,384,384], info)."""
    import numpy as np
    slices, bad = [], 0
    for b in blobs:
        try:
            slices.extend(read_slices(b))
        except Exception:
            bad += 1
    info = {'n_raw': len(slices), 'n_bad': bad}
    if not slices:
        return np.zeros((1, SIZE, SIZE), np.uint8), info
    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is not None:
        normal = np.cross(iop[:3], iop[3:]); plane = int(np.abs(normal).argmax())
    else:
        normal, plane = None, 0
    info['plane'] = PLANES[plane]
    if normal is not None and all(s[2] is not None for s in slices):
        sgn = CANON[plane][0] * np.sign(normal[plane])
        slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sgn)
    else:
        slices.sort(key=lambda s: s[4])
    if len(slices) > MAX_SLICES:
        keep = np.linspace(0, len(slices) - 1, MAX_SLICES).round().astype(int)
        slices = [slices[i] for i in keep]
    vol = np.stack([resample(*canon_slice(s[0], s[1], s[3], plane)) for s in slices])
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    info['n'] = len(vol)
    return (vol * 255).round().astype(np.uint8), info


def slot_of(plane, fat_sup):
    need(plane in PLANES and int(fat_sup) in (0, 1), 'invalid plane/fat-suppression flag')
    return PLANES.index(plane) + 3 * (1 - int(fat_sup))


def build_table(rows, n_files):
    """rows: dicts with StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane, Fat_Suppression (CSV order)."""
    table = {}
    for r in rows:
        if n_files.get(r['SeriesInstanceUID'], 0) < 3:
            continue
        table.setdefault(r['StudyInstanceUID'], [[] for _ in range(N_SLOTS)])[
            slot_of(r['Anatomical_Plane'], r['Fat_Suppression'])].append(r['SeriesInstanceUID'])
    for slots in table.values():
        for cands in slots:
            cands.sort(key=lambda s: -n_files[s])   # stable: CSV order on ties
    return table


def window_centres(n, k, lo=0.04, hi=0.96):
    import numpy as np
    a, b = max(1.0, lo * (n - 1)), min(n - 2.0, hi * (n - 1))
    if b < a:
        a = b = (n - 1) / 2
    c = np.linspace(a, b, k)
    return np.clip(np.round(c).astype(int), 1, max(1, n - 2))


def study_input(slots, load_blobs, k=K):
    """slots: 6 candidate lists; load_blobs(series) -> list of bytes. Mirrors infer.DicomStudyDataset."""
    import numpy as np
    x = np.zeros((N_SLOTS, k, 3, SIZE, SIZE), np.uint8)
    mask, pos = np.zeros(N_SLOTS, bool), np.zeros((N_SLOTS, k), np.float32)
    used = [None] * N_SLOTS; infos = [None] * N_SLOTS
    for s, cands in enumerate(slots):
        if not cands:
            continue
        try:
            vol, info = series_volume(load_blobs(cands[0]))
        except Stop:
            raise
        except Exception as exc:  # source: a failing slot is left empty
            infos[s] = {'error': type(exc).__name__}
            continue
        infos[s] = info
        n = len(vol)
        if n < 3:
            continue
        c = window_centres(n, k)
        x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in c])
        mask[s], pos[s] = True, c / max(n - 1, 1)
        used[s] = cands[0]
    return x, mask, pos, used, infos


def make_model():
    import timm
    import torch
    from torch import nn

    class KneeNet(nn.Module):
        def __init__(self, dim=384, depth=2, drop=0.2, drop_path=0.1):
            super().__init__()
            self.enc = timm.create_model('convnext_tiny', pretrained=False, num_classes=0, drop_path_rate=drop_path)
            self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(self.enc.num_features, dim), nn.LayerNorm(dim))
            self.slot_emb = nn.Embedding(N_SLOTS, dim)
            self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
            layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
            self.cls_w = nn.Parameter(torch.zeros(len(LABELS), dim))
            self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))

        def head(self, feat, mask, pos):
            b, s, k, _ = feat.shape
            slot = torch.arange(s, device=feat.device).view(1, s, 1).expand(b, s, k)
            h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1).to(feat.dtype))
            h = h.view(b, s * k, -1)
            pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
            pad = pad & ~pad.all(1, keepdim=True)
            h = self.tf(h, src_key_padding_mask=pad)
            a = self.att(h).float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
            pooled = torch.einsum('btc,btd->bcd', a.to(h.dtype), h)
            return (pooled * self.cls_w).sum(-1) + self.cls_b

        def forward(self, x, mask, pos, res=256):
            b, s, k = x.shape[:3]
            keep = mask.view(-1)
            xv = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
            f = self.enc(make_views(xv, res, k).contiguous(memory_format=torch.channels_last))
            feat = f.new_zeros(b * s, k, f.shape[-1])
            feat[keep] = f.view(-1, k, f.shape[-1])
            return self.head(feat.view(b, s, k, -1), mask, pos)

    return KneeNet()


def make_views(x, res, group, fov=0.92):
    import torch
    import torch.nn.functional as F
    n = x.shape[0]; ns = n // group; dev = x.device
    x = x.float().div_(255.0)
    ang = torch.zeros(ns, device=dev)
    sc = torch.full((ns,), fov, device=dev)
    tx = ty = torch.zeros(ns, device=dev)
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    theta = theta.repeat_interleave(group, 0)
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    if x.shape[-1] > 1.5 * res:
        x = F.avg_pool2d(x, 2) if x.shape[-1] >= 2 * res else F.interpolate(x, scale_factor=0.75, mode='bilinear', antialias=True)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


def checkpoint_paths(root=None):
    root = Path(root) if root else READER / 'extracted'
    return [root / f'cnxt_v0_fold{i}.pt' for i in range(3)]


def manifest_hashes():
    m = read_json(READER / 'MANIFEST.json')
    h = {Path(f['path']).name: f['sha256'] for f in m['files']}
    got = tuple(h[f'cnxt_v0_fold{i}.pt'] for i in range(3))
    need(got == CKPT_SHA, 'MANIFEST checkpoint hashes differ from the registered constants')
    return got


def load_models(device='cpu', paths=None, expected=None):
    """Hash-check, then torch.load(weights_only=True) only. Any failure is a STOP."""
    import torch
    paths = paths or checkpoint_paths(); expected = expected or manifest_hashes()
    models = []
    for i, (p, h) in enumerate(zip(paths, expected)):
        need(sha(p) == h, f'checkpoint hash mismatch: {Path(p).name}')
        try:
            ck = torch.load(p, map_location='cpu', weights_only=True)
        except Exception as exc:
            raise Stop(f'STOP weights_only load failed for {Path(p).name}: {type(exc).__name__}') from exc
        a = ck['args']
        need(set(ck) == {'model', 'args'} and a['fold'] == i and a['backbone'].split('.')[0] == 'convnext_tiny'
             and a['res'] == 256 and a['k_eval'] == K, 'checkpoint recipe differs')
        m = make_model()
        m.load_state_dict(ck['model'], strict=True)
        models.append((m.to(device).to(memory_format=torch.channels_last).eval(), int(a['res'])))
    return models


def predict(models, x, mask, pos, device):
    import torch
    xt = torch.from_numpy(x)[None].to(device)
    mt = torch.from_numpy(mask)[None].to(device)
    pt = torch.from_numpy(pos)[None].to(device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=(device == 'cuda')):
        p = torch.stack([m(xt, mt, pt, res=res).float().sigmoid() for m, res in models]).mean(0)
    return p[0].cpu().numpy().astype('float64')


# ----------------------------------------------------------------------------------------------
# OAI inputs (same ids, routes and byte pins as r6_oai_readout_v1 / v110)
# ----------------------------------------------------------------------------------------------

def predecessor_files():
    names = ('registration', 'prepare', 'attempt', 'measurement', 'evaluation', 'labels', 'gpu-review', 'validate')
    out = {}
    for w in (PILOT_W, EXT_W):
        for n in names:
            p = w / (n + '.json'); out[p.relative_to(ROOT).as_posix()] = sha(p)
    return out


def oai_binding():
    ext = read_json(EXT_W / 'registration.json'); pil = read_json(PILOT_W / 'registration.json')
    b = ext['binding']
    # Predecessor bindings are authenticated by pinning their file hashes in our registration.
    conf, pilot, pooled = list(b['ids']), list(b['pilot_ids']), list(b['pooled_ids'])
    need(len(conf) == 461 and len(pilot) == 96 and len(pooled) == 557 and not set(conf) & set(pilot)
         and sorted(conf + pilot) == pooled, 'cohort roster differs from the registered 96 + 461 split')
    need(pil['binding']['ids'] == pilot and b['pilot_binding_sha256'] == pil['binding_sha256'], 'pilot roster binding differs')
    routes = {}
    for u in pooled:
        rows = b['routes'][u]
        need(len(rows) >= 4 and len({r['SeriesInstanceUID'] for r in rows}) == len(rows), 'route drift')
        out = []
        for r in rows:
            fl = str(r['Fluid_Sensitive'])
            need(fl in FS_RULE and (fl != '-1' or r['Anatomical_Plane'] == 'Axial'), 'unregistered Fluid_Sensitive value')
            out.append(dict(StudyInstanceUID=u, SeriesInstanceUID=r['SeriesInstanceUID'],
                            Anatomical_Plane=r['Anatomical_Plane'], Fluid_Sensitive=int(r['Fluid_Sensitive']),
                            Fat_Suppression=FS_RULE[fl]))
        routes[u] = out
    return dict(pilot_ids=pilot, confirmation_ids=conf, pooled_ids=pooled, routes=routes,
                ext_binding_sha256=ext['binding_sha256'], pilot_binding_sha256=pil['binding_sha256'])


def input_pins():
    """uid -> (data root, {relpath: sha}) from the predecessors' sealed prepare.json."""
    out = {}
    pil = read_json(PILOT_W / 'prepare.json'); ext = read_json(EXT_W / 'prepare.json')
    need(pil['state'] == ext['state'] == 'PASS', 'predecessor prepare not PASS')
    b = oai_binding()
    for ids, w, pre in ((b['pilot_ids'], PILOT_W, pil), (b['confirmation_ids'], EXT_W, ext)):
        for u in ids:
            pref = 'test_series/' + u + '/'
            sel = {p: h for p, h in pre['data'].items() if p.startswith(pref)}
            need(sel, 'no pinned input for a registered knee')
            out[u] = (w / 'data', sel)
    return out


class KneeLoader:
    """Reads each pinned DICOM once, verifies its SHA256, keeps os.listdir order (as the source)."""

    def __init__(self, uid, root, pins):
        self.uid, self.root, self.pins = uid, Path(root), pins
        self.bytes_read = 0

    def series_dir(self, series):
        return self.root / 'test_series' / self.uid / series

    def n_files(self, series):
        d = self.series_dir(series)
        return min(len(os.listdir(d)), MAX_SLICES) if d.is_dir() else 0

    def __call__(self, series):
        d = self.series_dir(series); names = os.listdir(d)
        pref = f'test_series/{self.uid}/{series}/'
        expected = {p[len(pref):] for p in self.pins if p.startswith(pref)}
        need(set(names) == expected and names, 'staged series file set differs from the sealed pins')
        blobs = []
        for n in names:
            b = (d / n).read_bytes(); self.bytes_read += len(b)
            need(sha_bytes(b) == self.pins[pref + n], 'pinned DICOM byte drift')
            blobs.append(b)
        return blobs


def knee_predict(models, uid, routes, root, pins, device):
    t0 = time.perf_counter()
    loader = KneeLoader(uid, root, pins)
    rows = routes[uid]
    counts = {r['SeriesInstanceUID']: loader.n_files(r['SeriesInstanceUID']) for r in rows}
    table = build_table(rows, counts)
    slots = table.get(uid, [[] for _ in range(N_SLOTS)])
    x, mask, pos, used, infos = study_input(slots, loader)
    need(mask.any(), 'no usable series for a registered knee')
    t1 = time.perf_counter()
    p = predict(models, x, mask, pos, device)
    need(all(math.isfinite(v) and 0 <= v <= 1 for v in p), 'nonfinite reader output')
    return dict(study=uid, prediction=p.tolist(), slots_used=used, slot_info=infos,
                mask=mask.tolist(), decode_seconds=t1 - t0, forward_seconds=time.perf_counter() - t1,
                bytes_read=loader.bytes_read)


# ----------------------------------------------------------------------------------------------
# Registration / CPU checks
# ----------------------------------------------------------------------------------------------

RULES = dict(
    cohorts=dict(primary='confirmation (461)', secondary=['pilot (96)', 'pooled (557, descriptive precision update)']),
    comparisons=['Reader-minus-R6flip', 'Reader-minus-R7flip', 'R6flip-minus-R7flip (reference; must reproduce r6_oai_readout_v110)'],
    endpoints=dict(macro7=[2, 3, 5, 6, 7, 8, 9], weak_four=[3, 5, 6, 8], per_target='MAPPED [0,2,3,4,5,6,7,8,9] with >= 5 per class',
                   standalone='Reader AUC per target and macro7 with bootstrap CI'),
    bootstrap='r6_oai_readout_v110.bootstrap: seed 174175, 2000 joint study draws on the fixed macro7 set, secondary targets individually, linear 95% quantiles',
    family='Bonferroni 97.5% (quantiles 1.25/98.75) over the two Reader comparisons',
    p_normal='P(delta>0) = Phi(delta/se), se = bootstrap SD (ddof 1)',
    reading_rule=('DESCRIPTIVE L7-OAI lens only; no submission decision from this readout alone. '
                  'Reader is competition-data-only (Apache-2.0) and clean on OAI. R6flip/R7flip are EXPOSED: '
                  'R7 fm_aux present; R6 complete roster absent; RSNA-only/OAI-free unproved -> any OAI exposure '
                  'inflates R6/R7, so Reader-minus-R6/R7 > 0 is conservative and < 0 is inconclusive. '
                  'R / R224 are OAI-trained and are not compared. OAI MOAKS labels are proxies (B mapping), '
                  'not competition labels; 1/10/11 unmapped.'),
    reader=dict(folds=3, k=K, res='checkpoint args (256)', cuda_precision='fp16 autocast (source infer.py)',
                cpu_precision='fp32', tta='none (source has none)', laterality='no flip; IOP canonicalisation as source',
                slot_rule=FS_RULE_TEXT, failure='a failing series leaves its slot empty (source); a knee with no usable '
                'series, a byte-pin mismatch, nonfinite output or VRAM gate failure is a STOP'),
    limitations=['parity reference is the unreviewed Codex hb190 CPU fp32 runner (its 15/58 shards matched EXP131 within 0.0031)',
                 'CUDA fp16 path not parity-tested (CPU fp32 only); B=1 per knee vs source bs=4',
                 'OAI decoding/routing proven on only 2 knees (CPU dry run) before the GPU run',
                 'AX_MPR (Fluid_Sensitive -1) -> fat-suppressed slot is an assumption; no sensitivity arm',
                 'GAIN/HARM labels use the unadjusted 95% CI; the two Reader comparisons also carry Bonferroni 97.5% family CIs'],
    gpu=dict(device='RTX 4070 Ti SUPER only', vram_gate='desktop + peak_reserved + 2 GiB <= total and peak <= 12 GiB',
             watchdog=f'idle/no-progress {IDLE_S} s, no wall-clock cap', resume='per-knee shards bound to registration'),
)


def registration_body():
    ob = oai_binding()
    return dict(tag=TAG, builder=BUILDER, tool_sha256=sha(SCRIPT),
                test_sha256=sha(TEST_FILE) if TEST_FILE.exists() else None,
                checkpoints=dict(zip([p.name for p in checkpoint_paths()], manifest_hashes())),
                manifest_sha256=sha(READER / 'MANIFEST.json'),
                pilot_ids=ob['pilot_ids'], confirmation_ids=ob['confirmation_ids'], pooled_ids=ob['pooled_ids'],
                routes=ob['routes'], ext_binding_sha256=ob['ext_binding_sha256'],
                pilot_binding_sha256=ob['pilot_binding_sha256'], predecessor_files=predecessor_files(),
                rules=RULES, command=launch_command())


def launch_command():
    return [SYS_PY, '-B', str(SCRIPT), 'run']


def queue_plan():
    return [dict(tag=TAG, command=launch_command())]


def register():
    cpu_only()
    need(not (WORK / 'gpu').exists(), 'GPU output exists; registration is immutable now')
    body = registration_body()
    reg = stamp(state='REGISTERED_BEFORE_READER_OAI_OUTPUT', registered_kst=now_kst(), binding=body,
                binding_sha256=digest(body))
    p = WORK / 'registration.json'
    if p.exists():
        old = read_json(p)
        if old['binding'] == body:
            return old
        n = len(list(WORK.glob('registration-prev-*.json')))
        os.replace(p, WORK / f'registration-prev-{n}.json')
        for stale in ('dry-run.json', 'parity.json', 'replay.json', 'validate.json', 'gpu-review.json', 'synthetic-eval.json'):
            if (WORK / stale).exists(): os.replace(WORK / stale, WORK / f'{stale[:-5]}-prev-{n}.json')
    write_json(p, reg)
    write_json(WORK / 'registration.sha256.json', dict(registration_sha256=sha(p), binding_sha256=reg['binding_sha256']))
    write_json(WORK / 'gpu-queue.json', queue_plan())
    return reg


def current():
    p = WORK / 'registration.json'; need(p.exists(), 'register first')
    reg = read_json(p); b = reg['binding']
    need(digest(b) == reg['binding_sha256'], 'registration binding digest drift')
    need(read_json(WORK / 'registration.sha256.json')['registration_sha256'] == sha(p), 'registration file hash drift')
    need(b['tool_sha256'] == sha(SCRIPT), 'tool changed after registration')
    need(b['checkpoints'] == dict(zip([q.name for q in checkpoint_paths()], manifest_hashes())), 'checkpoint pins drift')
    for rel, h in b['predecessor_files'].items(): need(sha(ROOT / rel) == h, 'predecessor file drift: ' + rel)
    return reg


def parity():
    """CPU fp32 on 2 real Gold58 studies vs the hb190 k=12 fp32 runner shards (reference only)."""
    cpu_only()
    import csv
    import numpy as np
    import torch
    torch.set_num_threads(4)
    reg = current()
    ids = read_json(HB190 / 'ids58.json')[:2]
    refs = []
    for i, u in enumerate(ids):
        r = read_json(HB190 / 'real58' / f'study-{i:04d}.json'); need(r['study'] == u and r['success'], 'reference shard differs')
        refs.append(r['prediction'])
    with open(GOLD_DICOM / 'train_series.csv', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['StudyInstanceUID'] in ids]
    models = load_models('cpu')
    out = []
    for u, ref in zip(ids, refs):
        sdir = lambda s: GOLD_DICOM / 'train_series' / u / s
        counts = {r['SeriesInstanceUID']: (min(len(os.listdir(sdir(r['SeriesInstanceUID']))), MAX_SLICES)
                                           if sdir(r['SeriesInstanceUID']).is_dir() else 0) for r in rows if r['StudyInstanceUID'] == u}
        table = build_table([r for r in rows if r['StudyInstanceUID'] == u], counts)
        load = lambda s: [(sdir(s) / n).read_bytes() for n in os.listdir(sdir(s))]
        t0 = time.perf_counter()
        x, mask, pos, used, infos = study_input(table[u], load)
        p = predict(models, x, mask, pos, 'cpu')
        out.append(dict(study_sha256=sha_bytes(u.encode()), seconds=time.perf_counter() - t0, slots=int(mask.sum()),
                        max_abs_delta=float(np.max(np.abs(p - np.asarray(ref)))), ours=p.tolist(), reference=ref))
    worst = max(o['max_abs_delta'] for o in out)
    state = 'PASS' if worst <= PARITY_TOL else 'FAIL'
    res = stamp(state=state, binding_sha256=reg['binding_sha256'], tolerance=PARITY_TOL, max_abs_delta=worst,
                studies=out, torch=torch.__version__, reference='hb190 real58 study-0000/0001 (CPU fp32 k=12, unreviewed Codex runner; '
                'its 15/58 shards matched EXP131 T4 within 0.0031)')
    write_json(WORK / 'parity.json', res)
    need(state == 'PASS', f'parity FAIL max_abs_delta={worst}')
    return res


def dry_run():
    """Real CPU dry run on 2 OAI knees (first pilot + first confirmation id): pins, decode, routing, timing."""
    cpu_only()
    import torch
    torch.set_num_threads(4)
    reg = current(); b = reg['binding']; pins = input_pins()
    ids = [b['pilot_ids'][0], b['confirmation_ids'][0]]
    models = load_models('cpu'); rows = []
    for u in ids:
        root, sel = pins[u]
        r = knee_predict(models, u, b['routes'], root, sel, 'cpu')
        need(sum(r['mask']) == 4 and [s is not None for s in r['slots_used']] == [True, True, True, False, True, False],
             'OAI slot routing differs from the registered rule')
        r['study'] = sha_bytes(u.encode())  # no patient id in the receipt
        rows.append(r)
    dec = sum(r['decode_seconds'] for r in rows) / len(rows)
    est = estimate_minutes(dec)
    res = stamp(state='PASS', binding_sha256=reg['binding_sha256'], knees=rows, mean_decode_seconds=dec,
                estimated_gpu_minutes=est, note='CPU fp32 receipts; no labels read')
    write_json(WORK / 'dry-run.json', res)
    return res


def estimate_minutes(decode_s=None):
    if decode_s is None:
        p = WORK / 'dry-run.json'
        decode_s = read_json(p)['mean_decode_seconds'] if p.exists() else 6.0
    lo = 557 * (decode_s * 0.8 + 0.3) / 60 + 1
    hi = 557 * (decode_s * 1.5 + 1.0) / 60 + 2
    return [round(lo), round(hi)]


def validate():
    cpu_only()
    reg = current(); ev = {}
    for name, ok in (('parity', 'PASS'), ('dry-run', 'PASS'), ('replay', 'PASS_REAL_LABEL_IDENTITY_CONTROL')):
        v = read_json(WORK / f'{name}.json')
        need(v['state'] == ok and v['binding_sha256'] == reg['binding_sha256'], 'stale or failed ' + name)
        ev[name] = sha(WORK / f'{name}.json')
    sys.path.insert(0, str(ROOT))
    from tools import gpu_queue
    plan = read_json(WORK / 'gpu-queue.json'); gpu_queue.validate_plan(plan)
    need(plan == queue_plan(), 'queue plan differs')
    return write_json(WORK / 'validate.json', stamp(state='CPU_VALIDATED', binding_sha256=reg['binding_sha256'],
                                                     evidence=ev, estimated_gpu_minutes=estimate_minutes()))


def check_review(reg):
    p = WORK / 'gpu-review.json'; need(p.exists(), 'independent gpu-review.json (verdict PASS) required before run')
    r = read_json(p)
    need(r.get('verdict') == 'PASS' and r.get('reviewer') not in (None, '', BUILDER) and r.get('tag') == TAG
         and r.get('binding_sha256') == reg['binding_sha256'] and r.get('validation_sha256') == sha(WORK / 'validate.json'),
         'independent bound PASS review required')


# ----------------------------------------------------------------------------------------------
# GPU run (supervisor + worker)
# ----------------------------------------------------------------------------------------------

def nvidia(query):
    out = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60)
    need(out.returncode == 0, 'nvidia-smi failed')
    return [[x.strip() for x in line.split(',')] for line in out.stdout.strip().splitlines() if line.strip()]


def gpu_desktop():
    rows = nvidia('--query-gpu=name,uuid,memory.used,memory.total')
    need(len(rows) == 1, 'exactly one visible GPU expected')
    name, uuid, used, total = rows[0]
    return dict(name=name, uuid=uuid, used_bytes=int(float(used)) * 2**20, total_bytes=int(float(total)) * 2**20)


def other_gpu_python(own=()):
    try:
        rows = nvidia('--query-compute-apps=pid,process_name,used_memory')
    except Stop:
        raise
    return [r for r in rows if len(r) >= 2 and 'python' in r[1].lower() and r[0] not in {str(x) for x in own}]


def vram_gate(desktop, peak):
    return peak <= PEAK_CAP and desktop['used_bytes'] + peak + HEADROOM <= desktop['total_bytes']


def shard_binding(reg, device):
    import torch
    return digest(dict(binding=reg['binding_sha256'], tool=sha(SCRIPT), device=device['name'], uuid=device['uuid'],
                       torch=torch.__version__, cuda=torch.version.cuda, k=K))


def worker():
    need(os.environ.get('OAI_READER_SUPERVISOR') == str(os.getppid()), 'supervised worker only')
    need(not (WORK / 'measurement.json').exists(), 'finished attempt sealed; replay refused')
    reg = current(); check_review(reg); b = reg['binding']
    desktop = read_json(WORK / 'gpu' / 'desktop.json')
    import torch
    need(torch.cuda.is_available(), 'CUDA unavailable')
    name = torch.cuda.get_device_name(0)
    need('4070 Ti SUPER' in name and (name in desktop['name'] or desktop['name'] in name), 'RTX 4070 Ti SUPER required')
    torch.backends.cudnn.benchmark = False
    pins = input_pins(); models = load_models('cuda')
    bind = shard_binding(reg, desktop)
    shards = WORK / 'gpu' / 'knees'; shards.mkdir(parents=True, exist_ok=True)
    status = dict(state='RUNNING', total=len(b['pooled_ids']), done=0, fresh=0, progress=0, started_kst=now_kst())

    def beat(**kw):
        status.update(kw, progress=status['progress'] + 1, heartbeat=time.time(), heartbeat_kst=now_kst())
        write_json(WORK / 'gpu' / 'status.json', status)

    beat(phase='models_loaded')
    peak_all = 0; t_start = time.perf_counter()
    for i, u in enumerate(b['pooled_ids']):
        sp = shards / f'{sha_bytes(u.encode())[:24]}.json'
        if sp.exists():
            r = read_json(sp); need(r['binding'] == bind and r['study'] == u, 'resume identity mismatch')
            status['done'] = i + 1; continue
        beat(phase='knee', index=i)
        torch.cuda.reset_peak_memory_stats()
        root, sel = pins[u]
        r = knee_predict(models, u, b['routes'], root, sel, 'cuda')
        peak = int(torch.cuda.max_memory_reserved()); peak_all = max(peak_all, peak)
        need(vram_gate(desktop, peak), f'VRAM gate failed: peak {peak} desktop {desktop["used_bytes"]}')
        r.update(binding=bind, peak_reserved_bytes=peak, peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
        write_json(sp, r)
        status['fresh'] += 1
        el = time.perf_counter() - t_start
        beat(phase='committed', done=i + 1, peak_reserved_bytes=peak_all,
             eta_minutes=round(el / status['fresh'] * (len(b['pooled_ids']) - i - 1) / 60, 1))
    beat(state='WORKER_COMPLETE', phase='done', peak_reserved_bytes=peak_all)
    return 0


def run():
    need(not (WORK / 'measurement.json').exists(), 'finished attempt sealed; replay refused')
    reg = current(); check_review(reg); b = reg['binding']
    for name in ('parity', 'dry-run'):
        v = read_json(WORK / f'{name}.json'); need(v['state'] == 'PASS' and v['binding_sha256'] == reg['binding_sha256'], 'stale ' + name)
    others = other_gpu_python(own=(os.getpid(),))
    need(not others, f'another Python GPU process is running: {others}')
    gdir = WORK / 'gpu'; gdir.mkdir(parents=True, exist_ok=True)
    lock = gdir / 'run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        raise Stop(f'lock exists ({lock}); another run may be active - remove only after confirming no worker')
    try:
        desktop = gpu_desktop()
        need('4070 Ti SUPER' in desktop['name'], 'RTX 4070 Ti SUPER required')
        need(vram_gate(desktop, 3 * 2**30), 'desktop VRAM too high for the planning admission (3 GiB + 2 GiB headroom)')
        prev = gdir / 'desktop.json'
        if prev.exists():
            old = read_json(prev); need(old['uuid'] == desktop['uuid'], 'resume on a different GPU')
            desktop = dict(desktop, first_desktop_used_bytes=old.get('first_desktop_used_bytes', old['used_bytes']))
        else:
            desktop['first_desktop_used_bytes'] = desktop['used_bytes']
        write_json(prev, desktop)
        print(json.dumps(dict(tag=TAG, knees=len(b['pooled_ids']), estimated_gpu_minutes=estimate_minutes(),
                              desktop_used_mib=desktop['used_bytes'] // 2**20)), flush=True)
        env = dict(os.environ, OAI_READER_SUPERVISOR=str(os.getpid()), HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        log = open(gdir / 'worker.log', 'ab')
        child = subprocess.Popen([sys.executable, '-B', str(SCRIPT), '_worker'], env=env, stdout=log, stderr=subprocess.STDOUT)
        last_progress, last_change, err = None, time.monotonic(), None
        try:
            while child.poll() is None:
                time.sleep(5)
                try:
                    st = read_json(gdir / 'status.json'); pr = st.get('progress')
                except (OSError, ValueError, KeyError):
                    pr = None
                if pr is not None and pr != last_progress:
                    last_progress, last_change = pr, time.monotonic()
                if time.monotonic() - last_change > IDLE_S:
                    err = f'idle/no-progress watchdog: no progress for {IDLE_S} s'
                    break
        finally:
            if child.poll() is None:
                child.kill(); child.wait()
            log.close()
        need(err is None, err or '')
        need(child.returncode == 0, f'worker failed with exit code {child.returncode}; see {gdir / "worker.log"}')
        return seal(reg)
    finally:
        try: os.remove(lock)
        except OSError: pass


def seal(reg):
    need(not (WORK / 'measurement.json').exists(), 'finished attempt sealed; replay refused')
    b = reg['binding']; desktop = read_json(WORK / 'gpu' / 'desktop.json')
    st = read_json(WORK / 'gpu' / 'status.json'); need(st['state'] == 'WORKER_COMPLETE', 'worker incomplete')
    bind = None; preds = []; files = {}; peaks = []
    for u in b['pooled_ids']:
        sp = WORK / 'gpu' / 'knees' / f'{sha_bytes(u.encode())[:24]}.json'
        r = read_json(sp); need(r['study'] == u, 'shard identity')
        bind = bind or r['binding']; need(r['binding'] == bind, 'mixed shard bindings')
        preds.append(r['prediction']); peaks.append(r['peak_reserved_bytes'])
        files[sp.relative_to(WORK).as_posix()] = sha(sp)
    need(max(peaks) <= PEAK_CAP, 'peak cap')
    write_json(WORK / 'gpu' / 'predictions.json', dict(ids=b['pooled_ids'], predictions=preds, labels=LABELS))
    files['gpu/predictions.json'] = sha(WORK / 'gpu' / 'predictions.json')
    m = stamp(state='COMPLETE_PENDING_RESULT_REVIEW', binding_sha256=reg['binding_sha256'], shard_binding=bind,
              knees=len(preds), peak_reserved_bytes=max(peaks), desktop=desktop, files=files, sealed_kst=now_kst())
    write_json(WORK / 'measurement.json', m)
    return m


# ----------------------------------------------------------------------------------------------
# Evaluation (CPU)
# ----------------------------------------------------------------------------------------------

def p_normal(delta, se):
    if not se or not math.isfinite(se) or se <= 0:
        return 1.0 if delta > 0 else 0.0 if delta < 0 else 0.5
    return 0.5 * (1 + math.erf(delta / (se * math.sqrt(2))))


def metrics(y, pred, contract, draws=None, bootstrap=None, metric=None, macro7=None, weak_four=None):
    """pred: {arm: [n,12]} for ARMS. Reuses r6_oai_readout_v110 bootstrap/metric conventions when supplied."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    need(set(pred) == set(ARMS), 'three arms required')
    arrays = [np.asarray(pred[a], float) for a in ARMS]
    for v in arrays: need(v.shape == (len(y), 12) and np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), 'invalid predictions')
    eligible = contract['eligible']; macro7 = macro7 or [2, 3, 5, 6, 7, 8, 9]; weak_four = weak_four or [3, 5, 6, 8]
    auc = np.array([[roc_auc_score(y[np.isfinite(y[:, j]), j], v[np.isfinite(y[:, j]), j]) for j in eligible] for v in arrays])
    primary = [j for j in macro7 if j in eligible]
    pb, weights, attempts = bootstrap(y, arrays, primary, draws)
    boot = np.empty((len(arrays), len(weights), len(eligible)))
    for q, j in enumerate(eligible):
        boot[:, :, q] = pb[:, :, primary.index(j)] if j in primary else bootstrap(y, arrays, [j], draws)[0][:, :, 0]

    def summ(point, rep, family):
        r = metric(point, rep); r['p_normal_gt0'] = p_normal(r['delta'], r['se'])
        r['statistical_basis'] = 'unadjusted 95% CI (GAIN/HARM label); family CI reported separately'
        if family: r['ci97_5_family_two'] = np.percentile(rep, [1.25, 98.75], method='linear').tolist()
        return r

    def group(vec, bvec, idx, family):
        if not set(idx) <= set(eligible): return None
        ix = [eligible.index(j) for j in idx]
        return summ(float(vec[ix].mean()), bvec[:, ix].mean(1), family)

    comps = {}
    for a, c in COMPARISONS:
        ia, ic = ARMS.index(a), ARMS.index(c); fam = a == 'Reader'
        delta = auc[ia] - auc[ic]; bd = boot[ia] - boot[ic]
        comps[f'{a}-minus-{c}'] = dict(macro7=group(delta, bd, macro7, fam), weak_four=group(delta, bd, weak_four, fam),
                                       per_target={str(j): summ(float(delta[q]), bd[:, q], fam) for q, j in enumerate(eligible)})
    ir = ARMS.index('Reader')
    standalone = dict(macro7=group(auc[ir], boot[ir], macro7, False), weak_four=group(auc[ir], boot[ir], weak_four, False))
    return dict(comparisons=comps, reader_standalone=standalone,
                per_target_auc={a: dict(zip(map(str, eligible), auc[k].tolist())) for k, a in enumerate(ARMS)},
                eligible=eligible, valid_draws=len(weights), attempts=attempts)


def comparator_predictions(reg):
    """R6flip/R7flip from the two sealed predecessor attempts, after hash checks only."""
    import numpy as np
    b = reg['binding']; out = {a: {} for a in ('R6flip', 'R7flip')}
    for w, key in ((PILOT_W, 'pilot_ids'), (EXT_W, 'confirmation_ids')):
        m = read_json(w / 'measurement.json'); a = read_json(w / 'attempt.json')
        need(m['state'] == a['state'] == 'COMPLETE_PENDING_RESULT_REVIEW' and m['attempt_sha256'] == sha(w / 'attempt.json')
             and m['binding_sha256'] == a['binding_sha256'], 'predecessor measurement seal drift')
        for arm in out:
            v = a['arms'][arm]; need(v['ids'] == b[key] and v['state'] == 'COMPLETE', 'predecessor cohort order drift')
            out[arm].update(zip(v['ids'], v['predictions']))
    return {arm: np.asarray([d[u] for u in b['pooled_ids']], float) for arm, d in out.items()}


def evaluate(synthetic=False, replay=False):
    """synthetic: fabricated labels/predictions identity control. replay: real labels and real R6flip/R7flip,
    Reader := R7flip copy (identity control; no reader output is read). Default: the registered readout."""
    cpu_only()
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from tools import r6_oai_readout_v110 as pre
    kw = dict(bootstrap=pre.bootstrap, metric=pre.g.metric, macro7=pre.MACRO7, weak_four=pre.RULES['weak_four'])
    if synthetic:
        rng = np.random.default_rng(0); n = 120
        y = np.full((n, 12), np.nan)
        for j in pre.MAPPED: y[:, j] = (rng.random(n) < .3).astype(float)
        base = np.clip(.5 * np.nan_to_num(y, nan=0.) + .5 * rng.random((n, 12)), 0, 1)
        res = metrics(y, dict(Reader=base, R6flip=base, R7flip=np.clip(base * .5 + .25, 0, 1)), pre.label_contract(y), **kw)
        c = res['comparisons']['Reader-minus-R6flip']['macro7']
        need(c['delta'] == 0 and c['ci95'] == [0., 0.], 'identity control failed')
        return write_json(WORK / 'synthetic-eval.json', stamp(state='PASS_SYNTHETIC_IDENTITY', result=res))
    from tools import oai_moaks_panel as panel
    reg = current(); b = reg['binding']
    comp = comparator_predictions(reg)
    if replay:
        reader = comp['R7flip'].copy()
    else:
        m = read_json(WORK / 'measurement.json')
        need(m['state'] == 'COMPLETE_PENDING_RESULT_REVIEW' and m['binding_sha256'] == reg['binding_sha256'], 'run incomplete')
        for rel, h in m['files'].items(): need(sha(WORK / rel) == h, 'GPU output drift: ' + rel)
        rp = read_json(WORK / 'gpu' / 'predictions.json'); need(rp['ids'] == b['pooled_ids'], 'reader order drift')
        reader = np.asarray(rp['predictions'], float)
    roster = panel.validate(); need(set(b['pooled_ids']) <= set(roster), 'frozen OAI panel roster differs')
    frozen_labels = read_json(EXT_W / 'labels.json'); frozen_eval = read_json(EXT_W / 'evaluation.json')['result']
    index = {u: i for i, u in enumerate(b['pooled_ids'])}; result = {}
    for name, key in (('pilot', 'pilot_ids'), ('confirmation', 'confirmation_ids'), ('pooled', 'pooled_ids')):
        ids = b[key]; ix = [index[u] for u in ids]; y = panel.outcomes(ids)
        contract = pre.label_contract(y); need(contract == frozen_labels[name], 'label contract differs from r6_oai_readout_v110')
        r = metrics(y, dict(Reader=reader[ix], R6flip=comp['R6flip'][ix], R7flip=comp['R7flip'][ix]), contract, **kw)
        fz = frozen_eval[name]
        for arm in ('R6flip', 'R7flip'):
            need(r['per_target_auc'][arm] == fz['per_target_auc'][arm], f'{arm} AUC does not reproduce the predecessor ({name})')
        ref = r['comparisons']['R6flip-minus-R7flip']['macro7']; old = fz['comparisons']['R6flip-minus-R7flip']['macro_available']
        need((old is None) == (ref is None) and (old is None or all(ref[k] == old[k] for k in old)),
             f'R6flip-R7flip macro7 does not reproduce the predecessor ({name})')
        if replay:
            c = r['comparisons']['Reader-minus-R7flip']['macro7']
            need(c is not None and c['delta'] == 0 and c['ci95'] == [0., 0.], 'real-label identity control failed')
            r = dict(valid_draws=r['valid_draws'], eligible=r['eligible'], studies=len(ids), r6_r7_reproduced=True)
        result[name] = r
    if replay:
        return write_json(WORK / 'replay.json', stamp(state='PASS_REAL_LABEL_IDENTITY_CONTROL', binding_sha256=reg['binding_sha256'],
                                                       cohorts=result, reader_output_read=False))
    return write_json(WORK / 'evaluation.json', stamp(
        state='COMPLETE_PENDING_RESULT_REVIEW', binding_sha256=reg['binding_sha256'], measurement_sha256=sha(WORK / 'measurement.json'),
        primary_cohort='confirmation', result=result, reading_rule=RULES['reading_rule'],
        decision='DESCRIPTIVE_ONLY_HOLD_PENDING_INDEPENDENT_RESULT_REVIEW', qualified_for_submission=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['register', 'parity', 'dry-run', 'validate', 'estimate', 'run', '_worker', 'evaluate'])
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--replay', action='store_true')
    a = ap.parse_args(argv)
    need(not (a.synthetic or a.replay) or a.action == 'evaluate', '--synthetic/--replay only with evaluate')
    need(not (a.synthetic and a.replay), 'choose one of --synthetic/--replay')
    print(f'oai_reader_readout v{VERSION} {a.action}', flush=True)
    if a.action not in ('run', '_worker'):
        cpu_only()
    os.environ['HF_HUB_OFFLINE'] = '1'
    if a.action == '_worker':
        return worker()
    fn = dict(register=register, parity=parity, validate=validate, run=run,
              evaluate=lambda: evaluate(a.synthetic, a.replay), estimate=lambda: dict(estimated_gpu_minutes=estimate_minutes()))
    fn['dry-run'] = dry_run
    res = fn[a.action]()
    keys = ('state', 'binding_sha256', 'max_abs_delta', 'estimated_gpu_minutes', 'knees', 'peak_reserved_bytes', 'command')
    print(json.dumps({k: (v if k != 'knees' or not isinstance(v, list) else len(v)) for k, v in res.items() if k in keys}, indent=1))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Stop as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        sys.exit(2)
