import os, glob, time, gc, hashlib
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
IMG = 336
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
         ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(slot[2] for slot in SLOTS)
K_EVAL = 62
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
            ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
_SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10),
            ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]
ARMS = [
    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.60},
    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.10},
    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": True, "w": 0.10},
    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20},
]

def build_backbone(arch, pretrained=False):
    hybrid = arch.startswith(('maxvit', 'maxxvit', 'coatnet', 'coat_', 'convnext'))
    is_vit = not hybrid and any((k in arch for k in ('vit', 'deit', 'dinov2', 'eva', 'beit')))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool='token', dynamic_img_size=True)
    else:
        kw.update(global_pool='avg')
    return timm.create_model(arch, **kw)

class RaptorClassifier(nn.Module):

    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))

def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    arch = ck.get('arch', arch_default)
    ck_res = int(ck.get('res', res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck['model'], strict=True)
    model.eval().to(device)
    del ck
    gc.collect()
    return (model, ck_res)

def load_refit_head(pt_path, feature_dim, device):
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    state = ck.get('model', ck)
    head = RaptorClassifier(nn.Identity(), F_dim=int(feature_dim))
    head_state = {name: tensor for name, tensor in state.items()
                  if not name.startswith('backbone.')}
    head.load_state_dict(head_state, strict=True)
    head.eval().to(device)
    del ck, state, head_state
    gc.collect()
    return head

def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = (int(valid.min()), int(valid.max()))
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]

def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == 'imagenet':
        x = (x - _MEAN) / _STD
    return x

@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != 'cpu' and str(device).startswith('cuda')
    if use_cuda:
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()

@torch.no_grad()
def infer_probs_two_heads(model, refit_head, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != 'cpu' and str(device).startswith('cuda')
    def forward_heads():
        features = model.encode(x)
        original = torch.sigmoid(model.head(features).float())
        refitted = torch.sigmoid(refit_head.head(features).float())
        return (original[0].cpu().numpy(), refitted[0].cpu().numpy())
    if use_cuda:
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                return forward_heads()
        except RuntimeError:
            torch.cuda.empty_cache()
            return forward_heads()
    return forward_heads()

def rankpct(x):
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, x.shape[0] - 1)

def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + '/*.dcm')
        recs = []
        ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and (len(iop) == 6):
                    r = np.array(iop[:3], float)
                    c = np.array(iop[3:], float)
                    n = np.cross(r, c)
                    pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None)
                ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps)
                recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return ([(f, ps) for _, f, ps in recs], med_ps)

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape
        cpx = int(round(CROP_MM / max(ps, 0.001)))
        cpx = min(cpx, min(h, w))
        y0 = (h - cpx) // 2
        x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return (order_and_meta, read_px, mm_crop_resize)

def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None

def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx = 0
    used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = (int(n * SPAN_LO), int(n * SPAN_HI) - 1)
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []
        pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp))
                pss.append(ps)
            except Exception:
                arrs.append(None)
                pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = (0.0, 1.0)
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-06), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8)
            idx += 1
        if idx >= MAXS:
            break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return (vol, mask)

def find_test_root():
    candidates = [
        '/kaggle/input/competitions/rsna-knee-abnormality-detection',
        '/kaggle/input/rsna-knee-abnormality-detection',
    ]
    hits = [base for base in candidates
            if os.path.isfile(base + '/test.csv')
            and os.path.isfile(base + '/test_series.csv')
            and os.path.isdir(base + '/test_series')]
    if len(hits) != 1:
        raise RuntimeError(f'expected one direct competition root, found {hits}')
    return hits[0]

def find_weight_file(fname):
    by_name = {
        'raptor_ft_coatnet_v5_full_swa.pt': [
            '/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt',
            '/kaggle/input/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt',
        ],
        'raptor_ft_coatnet_v10_full.pt': [
            '/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384dense/raptor_ft_coatnet_v10_full.pt',
            '/kaggle/input/raptor-knee-native384dense/raptor_ft_coatnet_v10_full.pt',
        ],
        'raptor_ft_coatnet_v8_full_swa.pt': [
            '/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384/raptor_ft_coatnet_v8_full_swa.pt',
            '/kaggle/input/raptor-knee-native384/raptor_ft_coatnet_v8_full_swa.pt',
        ],
    }
    if fname not in by_name:
        raise RuntimeError(f'unpinned Raptor checkpoint: {fname}')
    hits = [path for path in by_name[fname] if os.path.isfile(path)]
    if len(hits) != 1:
        raise RuntimeError(f'expected one direct Raptor checkpoint {fname}, found {hits}')
    expected_size = {
        'raptor_ft_coatnet_v5_full_swa.pt': 292831426,
        'raptor_ft_coatnet_v10_full.pt': 292829892,
        'raptor_ft_coatnet_v8_full_swa.pt': 292831426,
    }[fname]
    if os.path.getsize(hits[0]) != expected_size:
        raise RuntimeError(f'Raptor checkpoint size mismatch: {hits[0]}')
    return hits[0]

def main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev} | gpus {torch.cuda.device_count()} | torch {torch.__version__}", flush=True)
    root = find_test_root()
    tsdir = root + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = root + "/test_images"
    print("test root:", root, "| series dir:", tsdir, flush=True)
    test = pd.read_csv(root + "/test.csv")
    test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(root + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    series = {key: frame.to_dict("records") for key, frame in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)
    sub_cols = ["StudyInstanceUID"] + LAB
    sample = os.path.join(root, "sample_submission.csv")
    if os.path.exists(sample):
        sub_cols = list(pd.read_csv(sample, nrows=1).columns)
    reader = _make_reader()
    n_study, n_arm = len(test_ids), len(ARMS)
    arm_probs = [np.full((n_study, len(LAB)), 0.5, np.float32) for _ in range(n_arm)]
    for arm_index, arm in enumerate(ARMS):
        globals()["IMG"] = int(arm["img"])
        globals()["SLOTS"] = list(arm["slots"])
        globals()["MAXS"] = sum(slot[2] for slot in SLOTS)
        globals()["SPAN_LO"], globals()["SPAN_HI"] = map(float, arm["span"])
        globals()["K_EVAL"] = int(arm["k_eval"])
        weight_path = find_weight_file(arm["file"])
        model, resolution = load_model(weight_path, arm["arch"], arm["res"], dev)
        print(f"[arm {arm_index}] {arm['name']} | img {IMG} | slices {MAXS} | "
              f"span {SPAN_LO:.2f}-{SPAN_HI:.2f} | windows {K_EVAL} | "
              f"res {resolution} | {time.time() - t0:.0f}s", flush=True)
        for study_index, study_uid in enumerate(test_ids):
            try:
                volume, mask = build_study(study_uid, series, tsdir, reader)
                windows = eval_windows(volume, mask, k=K_EVAL, res=resolution, norm=NORM)
                if bool(arm.get("reverse", False)):
                    windows = windows.flip(1).contiguous()
                arm_probs[arm_index][study_index] = infer_probs(model, windows, dev)
                del volume, mask, windows
            except Exception as error:
                print(f"  [arm {arm_index}] study {study_index} {study_uid[:16]} FALLBACK "
                      f"({type(error).__name__}: {error})", flush=True)
            if (study_index + 1) % 100 == 0 or study_index + 1 == n_study:
                print(f"  [arm {arm_index}] {study_index + 1}/{n_study} | "
                      f"{time.time() - t0:.0f}s", flush=True)
        del model
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[arm {arm_index}] done + freed | {time.time() - t0:.0f}s", flush=True)
    weights = np.array([float(arm.get("w", 1.0)) for arm in ARMS], dtype=np.float64)
    weights /= weights.sum()
    print(f"[blend] global probability mean w="
          f"{dict(zip([arm['name'] for arm in ARMS], weights.round(4)))}", flush=True)
    probability_blend = np.tensordot(
        weights, np.stack([np.clip(values, 0, 1) for values in arm_probs]), axes=(0, 0))
    ranks = rankpct(probability_blend)
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5
    submission = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    submission.insert(0, "StudyInstanceUID", test_ids)
    submission = submission[sub_cols]
    assert submission["StudyInstanceUID"].tolist() == test_ids
    assert np.isfinite(submission[LAB].values).all()
    out = "/kaggle/working/_raptor.csv"
    submission.to_csv(out, index=False)
    print("wrote", out, "|", len(submission), "rows x", len(submission.columns), "cols", flush=True)
    print(submission.head().to_string(index=False), flush=True)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


# --- EXP014: exact public 4-arm Raptor + released residual-CoAt Gold58 diagnostic ---
import json as _json
import pandas as _pd
from sklearn.metrics import roc_auc_score as _roc_auc_score
from scipy.stats import spearmanr as _spearmanr


def _macro_auc(y, p):
    vals = [_roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]
    return float(np.mean(vals)), [float(v) for v in vals]


def _paired_bootstrap(y, p0, p1, repeats=3000, seed=20260912):
    rng = np.random.default_rng(seed)
    n = len(y); deltas = []
    for _ in range(repeats):
        ix = rng.integers(0, n, n)
        yy = y[ix]
        if any(len(np.unique(yy[:, j])) < 2 for j in range(yy.shape[1])):
            continue
        a0, _ = _macro_auc(yy, p0[ix]); a1, _ = _macro_auc(yy, p1[ix])
        deltas.append(a1 - a0)
    a = np.asarray(deltas, np.float64)
    return {
        'valid_repeats': int(len(a)), 'mean_delta': float(a.mean()),
        'std_delta': float(a.std()), 'ci95_low': float(np.quantile(a, .025)),
        'ci95_high': float(np.quantile(a, .975)),
        'fraction_positive': float((a > 0).mean()), 'seed': int(seed)
    }


def _coat_gold_npz():
    candidates = [
        '/kaggle/input/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3/gold58_top3_predictions.npz',
        '/kaggle/input/rsna-knee-coat-resgated-ep10-top3/gold58_top3_predictions.npz',
    ]
    hits = [p for p in candidates if os.path.isfile(p)]
    if len(hits) != 1:
        raise RuntimeError(f'expected one CoAt gold58 npz, found {hits}')
    return hits[0]


def exp014_main():
    t0 = time.time()
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if not torch.cuda.is_available():
        raise RuntimeError('EXP014 requires GPU')
    gpu_name = torch.cuda.get_device_name(0)
    print('EXP014 device', dev, '| gpu', gpu_name, '| gpu_count', torch.cuda.device_count(), flush=True)
    if 'T4' not in gpu_name:
        raise RuntimeError(f'EXP014 pinned to T4-compatible runtime, got {gpu_name}')

    root = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
    train = _pd.read_csv(root + '/train.csv')
    train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
    gold = train.loc[train[LAB].notna().all(axis=1), ['StudyInstanceUID'] + LAB].copy()
    assert len(gold) == 58, len(gold)
    ids = gold['StudyInstanceUID'].tolist()
    y = gold[LAB].to_numpy(np.float64)

    tser = _pd.read_csv(root + '/train_series.csv')
    tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
    tser['SeriesInstanceUID'] = tser['SeriesInstanceUID'].astype(str)
    tser = tser[tser['StudyInstanceUID'].isin(set(ids))].copy()
    series = {k: g.to_dict('records') for k, g in tser.groupby('StudyInstanceUID')}
    tsdir = root + '/train_series'
    reader = _make_reader()

    arm_preds = {}
    arm_receipts = {}
    for arm_index, arm in enumerate(ARMS):
        globals()['IMG'] = int(arm['img'])
        globals()['SLOTS'] = list(arm['slots'])
        globals()['MAXS'] = sum(slot[2] for slot in SLOTS)
        globals()['SPAN_LO'], globals()['SPAN_HI'] = map(float, arm['span'])
        globals()['K_EVAL'] = int(arm['k_eval'])
        model, resolution = load_model(find_weight_file(arm['file']), arm['arch'], arm['res'], dev)
        pred = np.full((len(ids), len(LAB)), np.nan, np.float32)
        a0 = time.time()
        failures = []
        for i, uid in enumerate(ids):
            try:
                vol, mask = build_study(uid, series, tsdir, reader)
                windows = eval_windows(vol, mask, k=K_EVAL, res=resolution, norm=NORM)
                if bool(arm.get('reverse', False)):
                    windows = windows.flip(1).contiguous()
                pred[i] = infer_probs(model, windows, dev)
                del vol, mask, windows
            except Exception as exc:
                failures.append({'uid': uid, 'error': f'{type(exc).__name__}: {exc}'})
            if (i + 1) % 10 == 0 or i + 1 == len(ids):
                print(f"EXP014 {arm['name']} {i+1}/{len(ids)} elapsed={time.time()-a0:.1f}s", flush=True)
        if failures or not np.isfinite(pred).all():
            raise RuntimeError(f"arm {arm['name']} failures={failures[:3]} nonfinite={int((~np.isfinite(pred)).sum())}")
        macro, per = _macro_auc(y, pred)
        arm_preds[arm['name']] = pred
        arm_receipts[arm['name']] = {
            'macro_auc': macro, 'per_target_auc': dict(zip(LAB, per)),
            'elapsed_seconds': float(time.time()-a0), 'file': arm['file'],
            'img': int(arm['img']), 'span': list(map(float, arm['span'])),
            'k_eval': int(arm['k_eval']), 'reverse': bool(arm.get('reverse', False))
        }
        out = _pd.DataFrame(pred, columns=LAB); out.insert(0, 'StudyInstanceUID', ids)
        out.to_csv(f"/kaggle/working/exp014_{arm['name']}.csv", index=False)
        del model
        gc.collect(); torch.cuda.empty_cache()
        print('EXP014 ARM DONE', arm['name'], macro, flush=True)

    names = [a['name'] for a in ARMS]
    weights = np.asarray([float(a['w']) for a in ARMS], np.float64); weights /= weights.sum()
    stack = np.stack([arm_preds[n] for n in names])
    raptor_prob = np.tensordot(weights, stack, axes=(0,0))
    raptor_rank = _pd.DataFrame(raptor_prob, columns=LAB).rank(method='average', pct=True).to_numpy(np.float64)
    raptor_macro, raptor_per = _macro_auc(y, raptor_prob)

    cnp = np.load(_coat_gold_npz(), allow_pickle=False)
    cuids = [str(x) for x in cnp['study_uids'].tolist()]
    cmap = {u:i for i,u in enumerate(cuids)}
    if set(cuids) != set(ids):
        raise RuntimeError('CoAt Gold58 UID set mismatch')
    ci = np.asarray([cmap[u] for u in ids], np.int64)
    coat_raw = cnp['probability_mean'][ci].astype(np.float64)
    coat_rank_ens = cnp['rank_ensemble'][ci].astype(np.float64)
    coat_rank = _pd.DataFrame(coat_rank_ens, columns=LAB).rank(method='average', pct=True).to_numpy(np.float64)
    coat_macro, coat_per = _macro_auc(y, coat_raw)

    hybrid = 0.60 * raptor_rank + 0.40 * coat_rank
    hybrid_macro, hybrid_per = _macro_auc(y, hybrid)
    hybrid_rerank = _pd.DataFrame(hybrid, columns=LAB).rank(method='average', pct=True).to_numpy(np.float64)
    hybrid_rerank_macro, _ = _macro_auc(y, hybrid_rerank)

    per_rows = []
    for j,t in enumerate(LAB):
        row = {
            'target': t,
            'raptor4_auc': raptor_per[j],
            'coat_auc': coat_per[j],
            'hybrid_auc': hybrid_per[j],
            'delta_hybrid_vs_raptor4': hybrid_per[j]-raptor_per[j],
            'spearman_raptor4_coat': float(_spearmanr(raptor_rank[:,j], coat_rank[:,j]).statistic),
        }
        for n in names:
            row['auc_'+n] = arm_receipts[n]['per_target_auc'][t]
        per_rows.append(row)
    _pd.DataFrame(per_rows).to_csv('/kaggle/working/exp014_per_target.csv', index=False)

    for name, arr in [('raptor4', raptor_prob), ('coat', coat_raw), ('hybrid', hybrid), ('hybrid_rerank', hybrid_rerank)]:
        d = _pd.DataFrame(arr, columns=LAB); d.insert(0, 'StudyInstanceUID', ids)
        d.to_csv(f'/kaggle/working/exp014_{name}.csv', index=False)

    result = {
        'experiment': 'EXP014_RAPTOR4_COAT_GOLD58',
        'gpu': gpu_name,
        'studies': len(ids),
        'elapsed_seconds': float(time.time()-t0),
        'arms': arm_receipts,
        'raptor4_weights': dict(zip(names, map(float, weights))),
        'raptor4_macro_auc': raptor_macro,
        'raptor4_per_target_auc': dict(zip(LAB, map(float, raptor_per))),
        'coat_probability_macro_auc': coat_macro,
        'hybrid_60_40_macro_auc': hybrid_macro,
        'hybrid_rerank_macro_auc': hybrid_rerank_macro,
        'paired_bootstrap_hybrid_vs_raptor4': _paired_bootstrap(y, raptor_prob, hybrid),
        'selection_warning': 'Gold58 is diagnostic and some released checkpoints were selected using Gold58; do not treat blend choice as untouched validation.'
    }
    with open('/kaggle/working/exp014_results.json','w') as f:
        _json.dump(result, f, indent=2)
    print(_json.dumps(result, indent=2), flush=True)

exp014_main()