
# --- EXP006A: Gold58 dominant Raptor arm, original vs same-slot 2.5D windows ---
import json as _json
import pandas as _pd
from sklearn.metrics import roc_auc_score as _roc_auc_score

def _build_study_with_slot_ids(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    slot_ids = np.full(MAXS, -1, np.int16)
    idx = 0
    used = set()
    for slot_index, (plane, fluid, k) in enumerate(SLOTS):
        r = _pick_series_for_slot(rows, plane, fluid, used)
        start = idx
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
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = (0.0, 1.0)
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is not None:
                aw = np.clip((a - loq) / (hiq - loq + 1e-06), 0, 1)
                aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
                vol[idx] = (aw * 255).astype(np.uint8)
                slot_ids[idx] = slot_index
            idx += 1
        expected_end = start + k
        if idx < expected_end:
            idx = expected_end
        if idx >= MAXS:
            break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask, slot_ids

def _safe_centers(mask, slot_ids):
    out = []
    for c in range(1, len(mask) - 1):
        ids = slot_ids[c-1:c+2]
        if mask[c-1:c+2].all() and ids[0] >= 0 and ids[0] == ids[1] == ids[2]:
            out.append(c)
    return out

def _windows_at(vol, centers, res, norm=NORM):
    wins = np.empty((len(centers), 3, res, res), np.float32)
    for j, c in enumerate(centers):
        tri = np.stack([vol[c-1], vol[c], vol[c+1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == 'imagenet':
        x = (x - _MEAN) / _STD
    return x

def _resample_centers(centers, k):
    if not centers:
        return []
    ii = np.linspace(0, len(centers)-1, k).round().astype(int)
    return [centers[i] for i in ii]

def _macro_auc(y, p):
    vals = [_roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])]
    return float(np.mean(vals)), vals

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

def exp006a_main():
    t0 = time.time()
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('EXP006A device', dev, '| gpu_count', torch.cuda.device_count(), flush=True)
    root = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
    train = _pd.read_csv(root + '/train.csv')
    train['StudyInstanceUID'] = train['StudyInstanceUID'].astype(str)
    gold = train.loc[train[LAB].notna().all(axis=1), ['StudyInstanceUID'] + LAB].copy()
    assert len(gold) == 58, len(gold)
    tser = _pd.read_csv(root + '/train_series.csv')
    tser['StudyInstanceUID'] = tser['StudyInstanceUID'].astype(str)
    tser['SeriesInstanceUID'] = tser['SeriesInstanceUID'].astype(str)
    wanted = set(gold['StudyInstanceUID'])
    tser = tser[tser['StudyInstanceUID'].isin(wanted)].copy()
    series = {k: g.to_dict('records') for k, g in tser.groupby('StudyInstanceUID')}
    tsdir = root + '/train_series'
    reader = _make_reader()

    arm = dict(ARMS[0])
    globals()['IMG'] = int(arm['img']); globals()['SLOTS'] = list(arm['slots'])
    globals()['MAXS'] = sum(slot[2] for slot in SLOTS)
    globals()['SPAN_LO'], globals()['SPAN_HI'] = map(float, arm['span'])
    globals()['K_EVAL'] = int(arm['k_eval'])
    model, resolution = load_model(find_weight_file(arm['file']), arm['arch'], arm['res'], dev)
    ids = gold['StudyInstanceUID'].tolist(); n = len(ids)
    pred_orig = np.full((n, len(LAB)), np.nan, np.float32)
    pred_safe_k = np.full_like(pred_orig, np.nan)
    pred_safe_unique = np.full_like(pred_orig, np.nan)
    diag = []
    for i, uid in enumerate(ids):
        vol, mask, slot_ids = _build_study_with_slot_ids(uid, series, tsdir, reader)
        original_centers = _eval_centers(mask, len(mask), K_EVAL)
        safe_unique = _safe_centers(mask, slot_ids)
        if not safe_unique:
            safe_unique = list(dict.fromkeys(original_centers))
            fallback = True
        else:
            fallback = False
        safe_k = _resample_centers(safe_unique, K_EVAL)
        original_unique = list(dict.fromkeys(original_centers))
        invalid_original = 0
        for c in original_unique:
            ids3 = slot_ids[c-1:c+2]
            if not (mask[c-1:c+2].all() and ids3[0] >= 0 and ids3[0] == ids3[1] == ids3[2]):
                invalid_original += 1
        pred_orig[i] = infer_probs(model, _windows_at(vol, original_centers, resolution), dev)
        pred_safe_k[i] = infer_probs(model, _windows_at(vol, safe_k, resolution), dev)
        pred_safe_unique[i] = infer_probs(model, _windows_at(vol, safe_unique, resolution), dev)
        diag.append({'StudyInstanceUID': uid, 'orig_windows': len(original_centers),
                     'orig_unique_centers': len(original_unique), 'orig_invalid_unique_centers': invalid_original,
                     'safe_unique_centers': len(safe_unique), 'safe_resampled_windows': len(safe_k),
                     'safe_fallback': fallback})
        if (i + 1) % 5 == 0 or i + 1 == n:
            print(f'EXP006A {i+1}/{n} elapsed={time.time()-t0:.1f}s', flush=True)

    y = gold[LAB].to_numpy(np.float64)
    results = {'experiment': 'EXP006A_RAPTOR_BOUNDARY_STAGE_A', 'arm': arm, 'studies': n,
               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
               'elapsed_seconds': time.time()-t0, 'variants': {}}
    for name, pred in [('original', pred_orig), ('same_slot_resample_k', pred_safe_k), ('same_slot_unique', pred_safe_unique)]:
        macro, per = _macro_auc(y, pred)
        results['variants'][name] = {'macro_auc': macro, 'per_target_auc': dict(zip(LAB, map(float, per)))}
    results['paired_bootstrap_vs_original'] = {
        'same_slot_resample_k': _paired_bootstrap(y, pred_orig, pred_safe_k),
        'same_slot_unique': _paired_bootstrap(y, pred_orig, pred_safe_unique),
    }
    ddf = _pd.DataFrame(diag)
    results['window_diagnostics'] = {
        'mean_orig_invalid_unique_centers': float(ddf.orig_invalid_unique_centers.mean()),
        'median_orig_invalid_unique_centers': float(ddf.orig_invalid_unique_centers.median()),
        'studies_with_invalid_original_centers': int((ddf.orig_invalid_unique_centers > 0).sum()),
        'mean_safe_unique_centers': float(ddf.safe_unique_centers.mean()),
        'safe_fallback_studies': int(ddf.safe_fallback.sum()),
    }
    _pd.DataFrame(pred_orig, columns=LAB).assign(StudyInstanceUID=ids).to_csv('/kaggle/working/exp006a_original.csv', index=False)
    _pd.DataFrame(pred_safe_k, columns=LAB).assign(StudyInstanceUID=ids).to_csv('/kaggle/working/exp006a_safe_k.csv', index=False)
    _pd.DataFrame(pred_safe_unique, columns=LAB).assign(StudyInstanceUID=ids).to_csv('/kaggle/working/exp006a_safe_unique.csv', index=False)
    ddf.to_csv('/kaggle/working/exp006a_window_diagnostics.csv', index=False)
    with open('/kaggle/working/exp006a_results.json', 'w') as f:
        _json.dump(results, f, indent=2)
    print(_json.dumps(results, indent=2), flush=True)

exp006a_main()
