"""v1.0.0: fixed fivefold anchor A5 member of P, weight .15.

Changelog: separate post-parent A5 phase; no change to parent serving workers.
Missing acquisitions alone permit parent-only rows. Operational errors are fatal.
"""
from __future__ import annotations
import copy
import csv
import gc
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tools import c85r7_rco_overlay_v1 as util

VERSION = '1.0.0'
DATASET = 'mattiaangeli/knee-mri-fold-weights'
FORMULA = "Pprime=rank(.85*rank(P)+.15*rank(A5)); A5=mean(five full-available-roster fold endpoint ranks)"
CFG = dict(backbone='vit_small_patch16_dinov3.lvd1689m', cond='token', img=336,
           meta='none', n_meta=0, n_slice=16, norm='none', pool='xcodex', stem='native')
HASHES = (
    'b447a665a92ed65e0f4cacf6800940ba7a6d585d0c36154ce18dee60ec48d5cd',
    '30bbe5f9afa24af57fd69bad293a4b9fe3ca2e1d737381bc06e469c5ed380e76',
    '5bdc4f72471e7c5db01f2f919ba39212087cbb5f2f76b619e4ba38446800b782',
    'f757ddc9531f903c6cd0729cadc7051fcdca03b281c89c39cc7957ab44fc2243',
    '066490a5c17b3887d1bfda1427f45b14cab416e7b1dcc16afc9a003cdcc08aad')


def checkpoint_root():
    paths = [Path('/kaggle/input/datasets')/DATASET,
             Path('/kaggle/input')/DATASET.split('/')[1]]
    hits = [p for p in paths if (p/'m_f0.pt').is_file()]
    util.need(len(hits) == 1, 'A5 requires exactly one direct dataset mount')
    return hits[0]


def check_checkpoint(z, fold):
    util.need(z.get('fold') == fold, 'A5 fold identity')
    cfg = z['cfg']
    fields = {k:cfg[k] if k in ('backbone', 'cond', 'pool', 'img') else cfg.get(k, default)
              for k,default in CFG.items()}
    util.need(fields == CFG, 'A5 input/model contract')
    util.need('labels' not in z or list(z['labels']) == util.LABELS, 'A5 label order')
    util.need(len(z['state_dict']) == 180, 'A5 requires 180 state tensors')


def model():
    import timm
    from tools.p25_a5_model_v1 import Net, ViTSlotToken
    util.need(timm.__version__ == '1.0.25', 'A5 requires inherited timm 1.0.25')
    enc = timm.create_model(CFG['backbone'], pretrained=False, num_classes=0,
                            in_chans=16, img_size=336)
    return Net(ViTSlotToken(enc, 6), 'token', 0, 'xcodex', stem='native', n_slice=16)


def load_fold(path, fold):
    import torch
    path = Path(path)
    util.need(path.name == f'm_f{fold}.pt' and util.sha(path) == HASHES[fold],
              'A5 checkpoint hash/file identity')
    # Only authenticated anchor files may reach pickle deserialization.
    z = torch.load(path, map_location='cpu', weights_only=False)
    check_checkpoint(z, fold)
    net = model()
    net.load_state_dict(z['state_dict'], strict=True)
    return net.eval()


def micro(images, masks, *, dev, model_group, amp=True):
    """Exact active-slot packing and fivefold sigmoid; CPU uses FP32."""
    import torch
    active = np.flatnonzero((masks > 0).any(axis=1))
    out = np.full((len(model_group), len(masks), 12), np.nan, np.float32)
    if not len(active):
        return out
    ims, slots, sidx, vms = [], [], [], []
    for compact, original in enumerate(active):
        present = np.flatnonzero(masks[original] > 0)
        blk = images[original][present]
        ims.append(torch.from_numpy(blk))
        vms.append(torch.from_numpy(blk.reshape(blk.shape[0], blk.shape[1], -1).max(2) > 0))
        slots.append(torch.from_numpy(present + 1).long())
        sidx.append(torch.full((len(present),), compact, dtype=torch.long))
    im = torch.cat(ims).to(dev, non_blocking=True).float().div_(255.0)
    sl, si, vm = (torch.cat(values).to(dev) for values in (slots, sidx, vms))
    sm = torch.zeros(len(sl), 0, device=dev)
    use_amp = amp and str(dev).startswith('cuda')
    def forward(enabled):
        per = torch.zeros(len(model_group), len(active), 12, device=dev, dtype=torch.float32)
        with torch.no_grad(), torch.autocast('cuda' if str(dev).startswith('cuda') else 'cpu',
                dtype=torch.bfloat16 if use_amp else torch.float32, enabled=enabled):
            for i, net in enumerate(model_group):
                per[i] = torch.sigmoid(net(im, sl, sm, si, len(active), vm=vm).float())
        return per.cpu().numpy()
    got = forward(use_amp)
    if use_amp and not np.isfinite(got).all():
        got = forward(False)
    util.need(np.isfinite(got).all() and ((got >= 0) & (got <= 1)).all(),
              'A5 acquired-row model failure; neutral fallback forbidden')
    out[:, active] = got
    return out


def ranges(n):
    return [[(s, min(s+8, n)) for i,s in enumerate(range(0, n, 8)) if i % 2 == d]
            for d in range(2)]


def predict(images, masks, groups, executor):
    import torch
    result = np.full((5, len(masks), 12), np.nan, np.float32)
    def owner(d, plan):
        with torch.cuda.device(d):
            return [(s,e,micro(images[s:e], masks[s:e], dev=f'cuda:{d}', model_group=groups[d]))
                    for s,e in plan]
    futures = [executor.submit(owner, d, plan) for d,plan in enumerate(ranges(len(masks))) if plan]
    for future in futures:
        for s,e,value in future.result():
            result[:,s:e] = value
    return result


def blend(parent_payload, raw, eligible):
    """All fold ranks use the full available cohort, never a decode chunk."""
    ids, parent = util.table(parent_payload)
    mask = np.asarray(eligible, dtype=bool)
    raw = np.asarray(raw)
    util.need(raw.shape == (5, len(ids), 12) and mask.shape == (len(ids),), 'A5 roster shape')
    util.need(np.isfinite(raw[:,mask]).all() and ((raw[:,mask] >= 0) & (raw[:,mask] <= 1)).all(),
              'A5 acquired probabilities')
    util.need(np.isnan(raw[:,~mask]).all(), 'A5 absent rows require NaN sentinels')
    base = util.ranks(parent)
    mixed = base.copy()
    if mask.any():
        n = int(mask.sum())
        folds = [(util.ranks(f[mask])*n-1)/(n-1) if n > 1 else np.full((1,12), .5) for f in raw]
        # Anchor A5_PREDS quantizes fold mean to FP32 before outer percentile rank.
        a5 = np.mean(folds, axis=0).astype(np.float32).astype(np.float64)
        mixed[mask] = .85*base[mask] + .15*util.ranks(a5)
    result = util.ranks(mixed).astype(np.float32)
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow([util.UID]+util.LABELS)
    writer.writerows([uid, *[format(float(x), '.9g') for x in row]] for uid,row in zip(ids,result))
    return buffer.getvalue().encode()


def activate_codec(receipt):
    util.need(receipt.get('status') == 'ACTIVE', 'A5 requires active inherited codec')
    path = Path(receipt['hook_dir'])/'knee_hb126_hook.py'
    util.need(util.sha(path) == receipt['hook_sha256'], 'A5 codec hook hash')
    sys.path.append(str(path.parent))
    os.environ['KNEE_CODEC_BRANCH'] = 'P25_A5'
    hook = importlib.import_module('knee_hb126_hook')
    util.need(Path(hook.__file__).resolve() == path.resolve(), 'A5 codec module identity')
    import pydicom
    state = hook.KNEE_HB126_STATE
    util.need(state.get('patched') and not state.get('errors'), 'A5 codec activation')
    return pydicom.__version__


def run(output, competition, codec, diagnostic=False):
    """Runs after all parent children finish; receipt never claims qualification."""
    from tools import k1sr_runtime_v10 as k1
    from tools import l1_rco_k2_worker_v102 as w
    output, competition = Path(output), Path(competition)
    rec = dict(schema='p25-a5/v1', version=VERSION, status='STARTED', formula=FORMULA,
               diagnostic=diagnostic, never_submit=diagnostic, qualified_for_submission=False,
               clean_oof=False, exposure='EXPOSED_EXPLORATORY', dataset=DATASET)
    receipt_path = output/'a5-receipt.json'
    w.write(receipt_path, rec)
    started = time.perf_counter()
    try:
        rec['pydicom'] = activate_codec(codec)
        os.environ['KNEE_CODEC_TELEMETRY'] = str(output/'a5-codec-telemetry')
        import cv2
        import pandas as pd
        import torch
        import timm
        from tools import p25_a5_render_v1 as render
        util.need(torch.cuda.device_count() == 2, 'A5 anchor requires two GPU owners')
        util.need(timm.__version__ == '1.0.25', 'inherited timm dependency drift')
        rec.update(torch=torch.__version__, timm=timm.__version__, anchor_timm='1.0.22')
        cv2.setNumThreads(1)
        render.SERIES_ROOT = competition/'test_series'
        util.need(render.SERIES_ROOT.is_dir(), 'A5 test series missing; no train fallback')
        parent = (output/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).read_bytes()
        ids, _ = util.table(parent)
        util.write_once(output/'a5-original-p.csv', parent)
        rows = pd.read_csv(competition/'test_series.csv', dtype={util.UID:str,'SeriesInstanceUID':str})
        util.need(set(ids) <= set(rows[util.UID]), 'A5 metadata roster incomplete')
        by = {str(u):frame.to_dict('records') for u,frame in rows.groupby(util.UID, sort=False)}
        root = checkpoint_root()
        models = []
        with w.offline():
            for f in range(5):
                with k1.phase_deadline(900, 'A5 fold load'):
                    models.append(load_fold(root/f'm_f{f}.pt', f))
                w.write(receipt_path, dict(rec, loaded_folds=len(models)))
            with k1.phase_deadline(900, 'A5 model transfer'):
                second = [copy.deepcopy(m).to('cuda:1').eval() for m in models]
                groups = [[m.to('cuda:0').eval() for m in models], second]
            raw = np.full((5,len(ids),12), np.nan, np.float32)
            eligible = np.zeros(len(ids), dtype=bool)
            with ThreadPoolExecutor(max_workers=max(1,min(4,os.cpu_count() or 4))) as decode, \
                    ThreadPoolExecutor(max_workers=2) as gpu:
                def block(start):
                    return [decode.submit(render.build_study, (i,u,by[u]))
                            for i,u in enumerate(ids[start:start+48])]
                pending = block(0)
                for start in range(0,len(ids),48):
                    with k1.phase_deadline(900, 'A5 decode/inference block'):
                        current = pending
                        pending = block(start+48) if start+48 < len(ids) else None
                        count = len(current)
                        images = np.zeros((count,6,16,336,336), np.uint8)
                        masks = np.zeros((count,6), np.uint8)
                        for future in current:
                            i, images_i, masks_i = future.result()
                            images[i], masks[i] = images_i, masks_i
                        eligible[start:start+count] = (masks > 0).any(axis=1)
                        raw[:,start:start+count] = predict(images,masks,groups,gpu)
                        del images, masks, current, images_i, masks_i
                    w.write(receipt_path, dict(rec, completed_studies=start+count,
                                               elapsed_seconds=time.perf_counter()-started))
                    gc.collect()
            # Retain raw outputs and availability for independent replay, no labels.
            np.savez_compressed(output/'a5-raw.npz', study_uids=np.asarray(ids),
                                raw_probabilities=raw, applicable=eligible)
            result = blend(parent, raw, eligible)
        util.write_once(output/'a5-pprime.csv', result)
        rec.update(status='COMPLETE_A5', studies=len(ids), applicable=int(eligible.sum()),
                   checkpoint_hashes=list(HASHES), strict_load=True, folds=5,
                   original_p_sha256=hashlib.sha256(parent).hexdigest(),
                   pprime_sha256=hashlib.sha256(result).hexdigest(),
                   raw_sha256=util.sha(output/'a5-raw.npz'), seconds=time.perf_counter()-started)
        del groups, models, second
        gc.collect()
        for d in range(2):
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        w.write(receipt_path, rec)
        return result
    except Exception as exc:
        rec.update(status='FAILED_A5', error=type(exc).__name__+': '+str(exc)[:500],
                   seconds=time.perf_counter()-started)
        w.write(receipt_path, rec)
        raise
