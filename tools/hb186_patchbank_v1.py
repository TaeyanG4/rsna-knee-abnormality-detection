"""H-B-186 v1.0.1: registered, EXPOSED R7-f0 normal-patch rejection screen.

Changelog 2026-10-16: first CPU-buildable extraction/screen tool. No downloads,
training of the encoder, Kaggle calls, or changes to frozen serving sources.
Scientific extraction is B-only through gpu_queue/gpu_job, after review admission.
Changelog v1.0.1: matching-only DICOM contrast provenance; unknown cannot match;
invalid distance coverage forces HOLD regardless of descriptive numerical rejects.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import hb145_output_io as safeio

VERSION = '1.0.1'
OUT = ROOT / '.local/scratch/slot-b/hb186'
COHORT = ROOT / '.local/scratch/astra-b4/H-B-146/launch-v115/registration.json'
BUNDLE = ROOT / 'storage/external/heliosli-r7cnxfsss-v1'
CODE = ROOT / '.local/scratch/claude-hb128/runtime/code/mainline_code'
DEPS = ROOT / '.local/scratch/claude-hb128/runtime/deps'
ANCHOR = ROOT / 'storage/weights/external/heliosli_rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt'
ARCHIVE = ROOT / 'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip'
SOURCE = ROOT / 'tools/exp169_r7_src'
TARGET = 'Lateral Meniscus'
SEED = 20261016
EXPOSURE = dict(exposure='EXPOSED', clean_oof=False,
                reason='image-fold-excluded; pooled priors, fastMRI+ auxiliary and unresolved anchor/side ancestry')
PINNED = {'manifest.json': 'c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6',
          'E_lrprior_s0_f0.pt': 'ca8a76e024828793e0663e5a2a13461de17e820bed6789eafe3f155072d3612f',
          'state_spec.json': '2c976e70773b745cb357b992ba04946c157aa478e65d82f2883039c038b6ab3c',
          'side_cls.pt': 'ab623e2b619db4917639220fed15202e8c8aedd07c8e298a66a2afc0f153650a'}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return safeio.read_json(path)


def atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    def write():
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
    safeio.retry(write)
    safeio.replace(tmp, path)


def own_labels(path):
    # Whitelist just UID and numeric target: never open any report CSV.
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            uid, value = row['StudyInstanceUID'], float(row[TARGET])
            need(uid not in result and 0 <= value <= 1, 'invalid/duplicate own-label record')
            result[uid] = value
    return result


def assert_exclusion(train, bank, dev, confirm, groups):
    sets = [set(x) for x in (train, bank, dev, confirm)]
    need(all(len(x) == len(set(x)) for x in (train, bank, dev, confirm)), 'duplicate roster')
    need(sets[1] <= sets[0], 'bank must be encoder-training-only')
    grouped = [{groups[u] for u in s} for s in sets]
    need(not grouped[0] & (grouped[2] | grouped[3]), 'encoder/held-out patient overlap')
    need(not grouped[1] & (grouped[2] | grouped[3]), 'bank/held-out patient overlap')
    need(not grouped[2] & grouped[3], 'fusion inner exclusion failed')
    return True


def registration():
    from tools import exp156_k_sweep as refs
    from tools import exp169_r7_port as port
    from tools.hb145_r7_heldout_v1 import default_paths
    series_path = default_paths()[1]
    r = read(COHORT)
    labels_path = ROOT / r['recipe']['labels']['path']
    labels = own_labels(labels_path)
    train, held = r['train'], r['val']
    groups = {u: r['components'][u] for u in train + held}
    need(len(held) == 977 and len(train) == 3190, 'unexpected H-B-146 cohort')
    need(all(int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 == 0 for u in held), 'not R7 fold zero')
    need(all(int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 != 0 for u in train), 'bank encoder fold mismatch')
    dev = [u for u in held if int(digest(['inner', SEED, groups[u]])[:8], 16) % 2 == 0]
    confirm = [u for u in held if u not in set(dev)]
    normal = [u for u in train if labels[u] < .5]
    # At most one study per duplicate/patient group in reference selection.
    bank, seen = [], set()
    for u in sorted(normal, key=lambda u: digest(['bank', SEED, u])):
        if groups[u] not in seen:
            bank.append(u)
            seen.add(groups[u])
        if len(bank) == 192:
            break
    assert_exclusion(train, bank, dev, confirm, groups)
    port.verify_source()
    for name, expected in PINNED.items():
        need(sha(BUNDLE / name) == expected, 'R7-f0 identity mismatch: ' + name)
    need(sha(ANCHOR) == '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e', 'anchor identity')
    files = [COHORT, labels_path, series_path, ANCHOR, Path(__file__),
             ROOT / '.local/scratch/slot-b/top1-1016/astra-fresh-REPORT.md',
             ROOT / 'tools/exp156_k_sweep.py', ROOT / 'tools/exp115_vlm_eval.py',
             ROOT / 'tools/hb145_r7_heldout_v1.py', ROOT / 'tools/hb145_output_io.py']
    files += [BUNDLE / n for n in PINNED] + list(SOURCE.glob('*.py'))
    files += [ROOT / p for p in set((refs.JEV, refs.AQUILA, *refs.CONSENSUS))]
    # Include all runtime Python sources; extracted dependency tree is checked separately.
    files += list(CODE.rglob('*.py'))
    return dict(schema='hb186-registration-v1', version=VERSION, seed=SEED, target=TARGET,
                target_index=3, **EXPOSURE, train=train, bank=bank, development=dev,
                confirmation=confirm, heldout=held, groups=groups, labels_path=str(labels_path), series_path=str(series_path),
                inputs={str(p.resolve()): sha(p) for p in files},
                source_archive=dict(path=str(ARCHIVE), bytes=ARCHIVE.stat().st_size),
                arms=['classifier', 'matched', 'shuffled'],
                classifier='raw R7-f0 logit, no post-stackers, eager fp32 CPU / fp16 CUDA, no TTA',
                extraction=dict(size=384, crop=352, chunk=32, views='v4', eval_stride=1,
                                slices=3, phys_mm=3.5, feature_dim=32, projection='fixed seeded Gaussian; per-vector L2',
                                region='coronal lateral view3 + sagittal joint view1; 3x3 map cells; 3 relative-position bins'),
                contrast=dict(source='separate per-slice DICOM whitelist; frozen model metadata unchanged',
                              classes='0 T1-like TR<1000 TE<30; 1 PD-like TR>=1500 TE<50; 2 T2-like TR>=1500 TE>=50; finite positive milliseconds',
                              admission='2D, ScanningSequence exactly SE, no inversion; explicit ScanOptions empty or subset FS/PFP/PER/FC; FS flag controls suppression matching',
                              unknown='missing/unsupported/inconsistent series excluded from bank and cannot match; retained in coverage denominator'),
                bank_rule=dict(normal_threshold=.5, strict_threshold=.1, studies=192,
                               per_stratum_cap=128, global_cap=30000, min_vectors=10000,
                               coreset='seeded uniform subset per plane/contrast/fatsat/region then global cap',
                               shuffle='permute anatomy labels within plane/contrast/fatsat, preserving counts and vectors'),
                distance='bank-only coordinate median/MAD; clip standardized coordinates at 5; median 5 nearest Euclidean distances',
                aggregation='per identical series/view/cell, median of 3 adjacent sampled centres at registered phys_step (3.5mm); study 95th percentile',
                fusion=dict(head='StandardScaler + LogisticRegression(C=1,l2,lbfgs)',
                            features=['raw_logit', 'distance', 'missing_distance'],
                            imputation='development median only', tuning='none',
                            fit_labels='own >= 0.5; confirmation never fits or selects anything'),
                bootstrap=dict(unit='study', count=1000, seed=SEED, ci=[2.5, 97.5], paired=True),
                protocol='dominant eligible-window plane/contrast/fatsat/native-full-FOV-extent-bin (<100,100-160,>160 mm)',
                lopo='refit identical head excluding each protocol from development; evaluate that protocol in confirmation; fixed bank retained',
                reject_rules=['matched confirmation AUC <= shuffled confirmation AUC',
                              'more than 25/50 top confirmation deviations adjudicated mainly protocol/FOV',
                              'any evaluable LOPO matched-minus-classifier AUC < 0'],
                holds=['missing top50 blinded image QA', 'missing bank/retrieval/position image QA',
                       'any missing study distance or patch coverage below 0.5', 'insufficient class support or protocol coverage', 'no positive matched-minus-classifier benefit',
                       'unresolved public-member ancestry prevents clean OOF promotion'],
                invalid_coverage_rule='any missing distance or coverage<0.5 forces HOLD; numerical rejects remain descriptive only',
                next_gate='Any non-rejection is exploratory only; independent result review and second patient-disjoint fold before deployment')


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    need(not (OUT / 'gpu').exists(), 'registration cannot be changed after GPU output exists')
    reg = registration()
    path = OUT / 'registration.json'
    if path.exists():
        need(read(path) == reg, 'existing registration differs; no silent overwrite')
    else:
        safeio.atomic_json(path, reg)
    safeio.atomic_json(OUT / 'registration.sha256.json', dict(sha256=sha(path)))
    queue = [dict(tag='HB186_PATCHBANK_F0_V1', command=[str(Path(sys.executable).resolve()),
                  str(Path(__file__).resolve()), 'extract'], after='Q12B_R_COHORT976_V1', continue_on_failure=True)]
    from tools.gpu_queue import validate_plan
    validate_plan(queue)
    safeio.atomic_json(OUT / 'gpu-queue.json', queue)
    print(json.dumps(dict(registration_sha256=sha(path), bank=len(reg['bank']),
                         development=len(reg['development']), confirmation=len(reg['confirmation']))))


def validate(check_deps=False):
    r = read(OUT / 'registration.json')
    need(sha(OUT / 'registration.json') == read(OUT / 'registration.sha256.json')['sha256'], 'registration seal differs')
    for path, expected in r['inputs'].items():
        need(sha(path) == expected, 'input/source hash changed: ' + Path(path).name)
    labels = own_labels(r['labels_path'])
    need(all(labels[u] < .5 for u in r['bank']), 'non-normal bank admission')
    assert_exclusion(r['train'], r['bank'], r['development'], r['confirmation'], r['groups'])
    from tools.gpu_queue import validate_plan
    q = validate_plan(read(OUT / 'gpu-queue.json'))
    need(len(q) == 1 and q[0]['tag'] == 'HB186_PATCHBANK_F0_V1' and
         q[0]['after'] == 'Q12B_R_COHORT976_V1' and q[0]['continue_on_failure'] is True,
         'queue contract mismatch')
    need(q[0]['command'] == [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), 'extract'], 'queue command differs')
    need(ARCHIVE.stat().st_size == r['source_archive']['bytes'], 'archive size changed')
    if check_deps:
        from tools.hb145_r7_heldout_v1 import verify_deps
        verify_deps(DEPS, CODE)
    return r


def review_gate():
    a = read(OUT / 'admission.json')
    need(a['verdict'] == 'PASS' and a['builder'] != a['reviewer'], 'independent review required')
    for key, name in [('registration_sha256', 'registration.json'), ('smoke_sha256', 'cpu-smoke.json'),
                      ('tests_sha256', 'tests.txt'), ('review_sha256', 'REVIEW.md')]:
        need(a[key] == sha(OUT / name), 'admission evidence mismatch: ' + name)
    need(read(OUT / 'cpu-smoke.json')['pass'] is True, 'CPU smoke has not passed')
    return a


def setup_runtime(cpu):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # Windows CUDA runtime can treat empty as unset; PyTorch's NVML visibility
        # parser correctly treats it as zero devices, including frozen aux loaders.
        os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    for p in (CODE, SOURCE, DEPS):
        sys.path.insert(0, str(p))
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cpu:
        need(not torch.cuda.is_available(), 'CPU command must not expose CUDA')
    return torch


class DenseHook:
    """Pure observation: returns None, never mutates input/output/model tensors."""
    def __init__(self, model, progress=lambda phase: None):
        self.maps, self.projection, self.progress = [], None, progress
        self.handle = model.enc.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        import numpy as np
        import torch
        need(output.ndim == 4, 'expected spatial ConvNeXt map')
        n, c, h, w = output.shape
        if self.projection is None:
            arr = np.random.default_rng(SEED).normal(size=(c, 32)).astype('float32') / np.sqrt(32)
            self.projection = torch.as_tensor(arr, device=output.device, dtype=torch.float32)
        ys = [round((h - 1) * f) for f in (.4, .5, .6)]
        xs = [round((w - 1) * f) for f in (.3, .5, .7)]
        cells = output.detach()[:, :, ys, :][:, :, :, xs].permute(0, 2, 3, 1).reshape(n, 9, c).float()
        reduced = cells @ self.projection
        reduced = reduced / reduced.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.maps.append(reduced.cpu().numpy())
        self.progress('encoder_chunk')

    def close(self):
        self.handle.remove()


CONTRAST_TAGS = ['EchoTime', 'RepetitionTime', 'InversionTime', 'ScanningSequence',
                 'ScanOptions', 'MRAcquisitionType']


def dicom_contrast(ds):
    """Matching-only conservative SE contrast; never feeds frozen model metadata.

    Codes: 0=T1-like, 1=PD-like, 2=T2-like; -1=unsupported/unknown.
    Explicit empty ScanOptions means no declared suppression; absence is unknown.
    """
    def tokens(value):
        return {str(v).strip().upper() for v in (value if isinstance(value, (list, tuple)) or
                type(value).__name__ == 'MultiValue' else str(value).split('\\'))} - {''}
    unknown = (-1, -1)
    try:
        te, tr = float(ds.EchoTime), float(ds.RepetitionTime)
        if not (math.isfinite(te) and math.isfinite(tr) and te > 0 and tr > 0):
            return unknown
        if tokens(ds.ScanningSequence) != {'SE'} or str(ds.MRAcquisitionType) != '2D':
            return unknown
        if float(getattr(ds, 'InversionTime', 0) or 0) != 0 or 'ScanOptions' not in ds:
            return unknown
        options = tokens(ds.ScanOptions)
        if not options <= {'FS', 'PFP', 'PER', 'FC'}:
            return unknown
        contrast = 0 if tr < 1000 and te < 30 else 1 if tr >= 1500 and te < 50 else 2 if tr >= 1500 and te >= 50 else -1
        return (contrast, int('FS' in options)) if contrast >= 0 else unknown
    except (AttributeError, TypeError, ValueError):
        return unknown


def series_contrasts(study):
    import pydicom
    result, audit = {}, {}
    for folder in sorted(Path(study).iterdir()):
        if not folder.is_dir():
            continue
        values = [dicom_contrast(pydicom.dcmread(p, stop_before_pixels=True,
                  specific_tags=CONTRAST_TAGS)) for p in sorted(folder.glob('*.dcm'))]
        code = values[0] if values and len(set(values)) == 1 else (-1, -1)
        result[folder.name] = code
        audit[folder.name] = dict(code=list(code), slices=len(values), consistent=len(set(values)) == 1)
    return result, audit


def series_lookup(blocks, raw):
    result = {i: b['series'] for i, b in enumerate(blocks)}
    need(len(set(result.values())) == len(result), 'duplicate series UID in frozen blocks')
    need(set(int(m[0]) for m in raw) == set(result), 'raw/block series mapping mismatch')
    return result


def patch_rows(maps, raw, contrast=None):
    import numpy as np
    # metadata: plane, contrast, fatsat, view, rel-bin, cell, series, native slice, full-FOV bin, phys-step
    full_fov = {int(m[0]): float(m[6])*130 for m in raw if int(m[3]) == 0}
    vectors, meta = [], []
    for i, m in enumerate(raw):
        plane, view = int(m[2]), int(m[3])
        if (plane, view) not in ((0, 1), (1, 3)):
            continue
        region = int(np.clip(np.floor((float(m[4]) + 1) * 1.5), 0, 2))
        need(int(m[0]) in full_fov, 'missing full-field geometry for eligible series')
        extent = full_fov[int(m[0])]
        step = max(1, int(round(3.5 / max(math.expm1(float(m[5])), .1))))
        for cell in range(9):
            vectors.append(maps[i, cell])
            code = contrast.get(int(m[0]), (-1, -1)) if contrast is not None else (int(m[7]), int(m[8]))
            meta.append([plane, *code, view, region, cell,
                         int(m[0]), int(m[9]), 0 if extent < 100 else 1 if extent <= 160 else 2, step])
    need(bool(vectors), 'no eligible meniscal patches')
    return np.asarray(vectors, 'float32'), np.asarray(meta, 'int32')


class Engine:
    def __init__(self, device, work, progress=lambda phase: None):
        self.torch = setup_runtime(device == 'cpu')
        import compact_train as CT
        import compact_infer as CI
        from tools.hb145_r7_heldout_v1 import TrainSource
        self.device, self.work, self.progress = device, Path(work), progress
        self.work.mkdir(parents=True, exist_ok=True)
        sd = self.torch.load(BUNDLE / 'E_lrprior_s0_f0.pt', map_location='cpu', weights_only=True)
        self.model = CT.Net('convnext_small', None, crop=352, state_spec=read(BUNDLE / 'state_spec.json'),
                            in_chans=3, spatial=True, extent=True, syn_decomp=True)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().requires_grad_(False).to(device).to(memory_format=self.torch.channels_last)
        del sd
        self.source = TrainSource(ARCHIVE)
        from tools.hb145_r7_heldout_v1 import default_paths
        data = self.work / 'data'
        data.mkdir(exist_ok=True)
        cols = ['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane']
        with default_paths()[1].open(encoding='utf-8-sig', newline='') as f, (data/'test_series.csv').open('w', encoding='utf-8', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=cols)
            writer.writeheader()
            for row in csv.DictReader(f):
                writer.writerow({k: row[k] for k in cols})
        with (data/'test.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f); writer.writerow(['StudyInstanceUID'])
            writer.writerows([u] for u in sorted(self.source.members))
        self.ds = CI.CStudies(self.work / 'data', 'test', [], self.work / 'native', CODE,
                             384, 1, 'v4', 3, ANCHOR, 3.5, BUNDLE / 'side_cls.pt', 2., 0., 20., True)
        self.ds._setup()
        build = self.ds.builder.build
        def observe_build(uid):
            item = build(uid)
            self.series = series_lookup(item['blocks'], item['meta'])
            return item
        self.ds.builder.build = observe_build
        original = self.ds.window_rows
        def capture(blocks, getter, *args, **kwargs):
            import numpy as np
            rows = original(blocks, getter, *args, **kwargs)
            self.raw = np.asarray([getter(r[0], r[1]) for r in rows], dtype='float32')
            return rows
        self.ds.window_rows = capture

    def study(self, uid, identity=False):
        import numpy as np
        torch = self.torch
        from tools.hb145_r7_heldout_v1 import owned_remove
        self.progress('stage')
        self.source.stage(uid, self.work / 'data')
        try:
            self.ds.uids = [uid]
            self.progress('decode_anchor_render')
            with contextlib.redirect_stdout(io.StringIO()):
                item = self.ds[0]
            need(item['error'] is None, 'real study preprocessing failed: ' + str(item['error']))
            contrasts, contrast_audit = series_contrasts(self.work / 'data' / 'test_series' / uid)
            matching = {i: contrasts.get(series, (-1, -1)) for i, series in self.series.items()}
            img = item['images'][None, :, :, 16:368, 16:368].to(self.device)
            meta = item['meta'][None].to(self.device)
            mask = torch.ones(meta.shape[:2], dtype=torch.bool, device=self.device)
            def forward():
                ctx = torch.autocast('cuda', dtype=torch.float16) if self.device == 'cuda' else contextlib.nullcontext()
                with torch.inference_mode(), ctx:
                    return self.model(img, meta, mask, chunk=32).float().cpu().numpy()
            before = forward() if identity else None
            hook = DenseHook(self.model, self.progress)
            try:
                logits = forward()
            finally:
                hook.close()
            need(np.isfinite(logits).all(), 'nonfinite logits')
            if identity:
                need(before.tobytes() == logits.tobytes(), 'hook changed logits bytes')
            maps = np.concatenate(hook.maps)[:len(self.raw)]
            x, pm = patch_rows(maps, self.raw, matching)
            # Save a label-free location montage for the first two real CPU studies.
            if identity:
                self.montage(item, uid)
            extra = {}
            if identity:
                from PIL import Image
                eligible = [i for i, m in enumerate(self.raw) if (int(m[2]),int(m[3])) in ((0,1),(1,3))]
                extra['preview'] = np.stack([np.asarray(Image.fromarray(item['images'][i,1,16:368,16:368].numpy()).resize((192,192))) for i in eligible])
            return dict(**extra, x=x, meta=pm, contrast_audit=np.asarray(json.dumps(contrast_audit)),
                        logit=np.asarray(logits[0, 3], 'float32'),
                        identity=np.asarray(identity), uid=np.asarray(uid),
                        logits_sha=np.asarray(hashlib.sha256(logits.tobytes()).hexdigest()))
        finally:
            owned_remove(self.work / 'data' / 'test_series' / uid, self.work)

    def montage(self, item, uid):
        from PIL import Image, ImageDraw
        import numpy as np
        eligible = [i for i, m in enumerate(self.raw) if (int(m[2]), int(m[3])) in ((0, 1), (1, 3))]
        picks = [eligible[k] for k in np.linspace(0, len(eligible)-1, 6).astype(int)]
        canvas = Image.new('RGB', (3 * 192, 2 * 212), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, i in enumerate(picks):
            tile = Image.fromarray(item['images'][i, 1, 16:368, 16:368].numpy()).resize((192, 192)).convert('RGB')
            td = ImageDraw.Draw(tile)
            for y in (.4, .5, .6):
                for x in (.3, .5, .7):
                    cx, cy = x * 192, y * 192
                    td.rectangle((cx-5, cy-5, cx+5, cy+5), outline='red')
            ox, oy = j % 3 * 192, j // 3 * 212
            canvas.paste(tile, (ox, oy))
            draw.text((ox, oy+193), f'plane {int(self.raw[i,2])} view {int(self.raw[i,3])} slice {int(self.raw[i,9])}', fill='black')
        canvas.save(self.work / (hashlib.sha256(uid.encode()).hexdigest()[:12] + '-positions.png'))

    def close(self):
        self.source.close()


def protocol(meta):
    return ':'.join(map(str, collections.Counter(tuple(row[[0, 1, 2, 8]]) for row in meta).most_common(1)[0][0]))


def strata(meta):
    return [tuple(row[:6]) for row in meta]


def build_bank(records, uids, labels, strict=False):
    import numpy as np
    entries = collections.defaultdict(list)
    for u in uids:
        if strict and labels[u] >= .1:
            continue
        for j, key in enumerate(strata(records[u]['meta'])):
            if key[1] not in (0, 1, 2) or key[2] not in (0, 1):
                continue
            entries[key].append((u, j))
    selected = []
    rng = np.random.default_rng(SEED)
    for key in sorted(entries):
        rows = entries[key]
        ix = rng.permutation(len(rows))[:128]
        selected.extend(rows[i] for i in ix)
    if len(selected) > 30000:
        selected = [selected[i] for i in rng.permutation(len(selected))[:30000]]
    need(len(selected) >= 5, 'insufficient reference bank')
    x = np.asarray([records[u]['x'][j] for u, j in selected], 'float32')
    m = np.asarray([records[u]['meta'][j] for u, j in selected], 'int32')
    center = np.median(x, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(x - center), axis=0), .01)
    x = np.clip((x - center) / scale, -5, 5)
    return dict(x=x, meta=m, center=center, scale=scale,
                owners=np.asarray([u for u, j in selected]))


def shuffle_bank(bank):
    import numpy as np
    result = {**bank, 'source_meta': bank['meta'].copy(), 'meta': bank['meta'].copy()}
    rng = np.random.default_rng(SEED + 1)
    groups = collections.defaultdict(list)
    for i, row in enumerate(bank['meta']):
        groups[tuple(row[:3])].append(i)
    for indices in groups.values():
        p = rng.permutation(indices)
        result['meta'][np.asarray(indices), 3:6] = bank['meta'][p, 3:6]
    need(not np.array_equal(result['meta'][:, 3:6], bank['meta'][:, 3:6]), 'shuffled anatomy unchanged')
    return result


class Distance:
    def __init__(self, bank):
        import numpy as np
        from scipy.spatial import cKDTree
        self.bank = bank
        rows = collections.defaultdict(list)
        for i, k in enumerate(strata(bank['meta'])):
            need(k[1] in (0, 1, 2) and k[2] in (0, 1), 'unknown bank contrast')
            rows[k].append(i)
        self.indices = {k: np.asarray(ix) for k, ix in rows.items() if len(ix) >= 5}
        self.trees = {k: cKDTree(bank['x'][ix]) for k, ix in self.indices.items()}

    def score(self, record, evidence=False):
        import numpy as np
        x = np.clip((record['x'] - self.bank['center']) / self.bank['scale'], -5, 5)
        rows = collections.defaultdict(list)
        for i, key in enumerate(strata(record['meta'])):
            rows[key].append(i)
        distances = np.full(len(x), np.nan)
        nearest = np.full(len(x), -1, dtype=int)
        for k, ix in rows.items():
            if k in self.trees:
                d, local = self.trees[k].query(x[ix], k=5, workers=1)
                nearest[ix] = self.indices[k][local[:, 0]]
                distances[ix] = np.median(d, axis=1)
        persistent, triplets = [], []
        tracks = collections.defaultdict(list)
        for i, row in enumerate(record['meta']):
            tracks[tuple(row[[6, 3, 5]])].append((int(row[7]), i))
        for track in tracks.values():
            track.sort()
            for j in range(len(track) - 2):
                triplet = track[j:j+3]
                step = int(record['meta'][triplet[0][1], 9])
                if [a for a, _ in triplet] != [triplet[0][0] + step*k for k in range(3)]:
                    continue
                values = distances[[i for _, i in triplet]]
                if np.isfinite(values).all():
                    persistent.append(float(np.median(values)))
                    triplets.append([i for _, i in triplet])
        answer = (float(np.quantile(persistent, .95)) if persistent else float('nan'),
                  float(np.isfinite(distances).mean()))
        if not evidence:
            return answer
        detail = None
        if persistent:
            best = int(np.argmax(persistent))
            detail = dict(max_persistent_distance=persistent[best], patches=[])
            for i in triplets[best]:
                b = int(nearest[i])
                detail['patches'].append(dict(query_meta=record['meta'][i].tolist(),
                    nearest_bank_uid=str(self.bank['owners'][b]),
                    nearest_bank_meta=self.bank.get('source_meta', self.bank['meta'])[b].tolist(),
                    matched_stratum=self.bank['meta'][b, :6].tolist(), distance=float(distances[i])))
        return (*answer, detail)


def auc(y, score):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    need(np.isfinite(score).all(), 'nonfinite evaluation score')
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def fit_head(train_ids, confirm_ids, groups, z, d, y):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    need(not {groups[u] for u in train_ids} & {groups[u] for u in confirm_ids}, 'head patient leakage')
    tr, te = np.asarray(train_ids), np.asarray(confirm_ids)
    def features(ids, median):
        values = np.array([d[u] for u in ids])
        return np.c_[[z[u] for u in ids], np.where(np.isfinite(values), values, median), ~np.isfinite(values)]
    values = np.array([d[u] for u in tr])
    median = float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else 0.
    yy = np.array([y[u] >= .5 for u in tr], int)
    need(len(np.unique(yy)) == 2, 'fusion development single-class')
    scaler = StandardScaler().fit(features(tr, median))
    head = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, random_state=SEED)
    head.fit(scaler.transform(features(tr, median)), yy)
    return head.decision_function(scaler.transform(features(te, median)))


def evaluate(y, scores, boot=1000):
    import numpy as np
    point = {k: auc(y, s) for k, s in scores.items()}
    samples = {k: [] for k in scores}
    delta = {k: [] for k in ('matched-classifier', 'matched-shuffled')}
    rng = np.random.default_rng(SEED)
    for _ in range(boot if len(y) else 0):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        a = {k: auc(y[ix], s[ix]) for k, s in scores.items()}
        for k in a:
            samples[k].append(a[k])
        for k in delta:
            left, right = k.split('-')
            delta[k].append(a[left] - a[right])
    ci = lambda values: np.percentile(values, [2.5, 97.5]).tolist() if values else None
    return dict(n=len(y), positives=int(y.sum()), auc=point,
                ci95={k: ci(v) for k, v in samples.items()},
                paired_delta_ci95={k: ci(v) for k, v in delta.items()},
                valid_bootstraps=len(samples['matched']))


def rejection(point, lopo, qa, top_ids):
    rejects, holds = [], []
    if any(point.get(k) is None for k in ('classifier', 'matched', 'shuffled')):
        holds.append('confirmation lacks both classes')
    else:
        if point['matched'] <= point['shuffled']:
            rejects.append('matched <= shuffled')
        if point['matched'] <= point['classifier']:
            holds.append('no positive benefit over classifier')
    if any(r.get('delta') is not None and r['delta'] < 0 for r in lopo.values()):
        rejects.append('LOPO reverses benefit')
    if not lopo or any(r.get('delta') is None for r in lopo.values()):
        holds.append('LOPO coverage/class support incomplete')
    marks = qa.get('top50', {}) if qa else {}
    if len(top_ids) != 50 or set(marks) != set(top_ids) or any(type(v) is not bool for v in marks.values()) or not qa.get('reviewer'):
        holds.append('top50 protocol/FOV image QA missing')
    elif sum(marks.values()) > 25:
        rejects.append('top deviations mainly protocol/FOV')
    if not qa or qa.get('bank_retrieval_position_pass') is not True:
        holds.append('bank retrieval/position image QA missing')
    return dict(verdict='REJECT' if rejects else 'HOLD' if holds else 'NOT_REJECTED_EXPLORATORY', rejects=rejects, holds=holds,
                deployment_allowed=False, next_gate='independent result review + second patient-disjoint fold; EXPOSED ancestry remains')


def analyze(r, records, labels, refs=None, qa=None, tiny=False):
    import numpy as np
    assert_exclusion(r['train'], r['bank'], r['development'], r['confirmation'], r['groups'])
    bank = build_bank(records, r['bank'], labels)
    need(tiny or len(bank['x']) >= 10000, 'bank below registered tens-of-thousands minimum')
    shuffled = shuffle_bank(bank)
    models = {'matched': Distance(bank), 'shuffled': Distance(shuffled)}
    ds, coverage = {}, {}
    for name, model in models.items():
        pairs = {u: model.score(records[u]) for u in r['heldout']}
        ds[name] = {u: pairs[u][0] for u in pairs}
        coverage[name] = {u: pairs[u][1] for u in pairs}
    z = {u: float(records[u]['logit']) for u in r['heldout']}
    confirm, dev = r['confirmation'], r['development']
    scores = {'classifier': np.array([z[u] for u in confirm])}
    for name in models:
        scores[name] = fit_head(dev, confirm, r['groups'], z, ds[name], labels)
    y = np.array([labels[u] >= .5 for u in confirm], int)
    primary = evaluate(y, scores, 20 if tiny else 1000)
    protocols = {u: protocol(records[u]['meta']) for u in r['heldout']}
    lopo = {}
    for p in sorted(set(protocols.values())):
        tr = [u for u in dev if protocols[u] != p]
        te = [u for u in confirm if protocols[u] == p]
        yy = np.array([labels[u] >= .5 for u in te], int)
        if len(set(labels[u] >= .5 for u in tr)) < 2 or len(np.unique(yy)) < 2 or len(te) < (2 if tiny else 20):
            lopo[p] = dict(n=len(te), delta=None, reason='insufficient fixed protocol support')
            continue
        pred = fit_head(tr, te, r['groups'], z, ds['matched'], labels)
        base = np.array([z[u] for u in te])
        lopo[p] = dict(n=len(te), delta=auc(yy, pred) - auc(yy, base))
    top = sorted([u for u in confirm if np.isfinite(ds['matched'][u])], key=lambda u: (-ds['matched'][u], u))[:50]
    report_rulers = {}
    if refs is not None:
        for name, (arr, mask, weight) in refs.items():
            use = mask[:, 3].astype(bool)
            report_rulers[name] = evaluate(arr[use, 3].astype(int), {k: v[use] for k, v in scores.items()}, 1000)
    # Stricter target-negative nomination, a prespecified bank-contamination sensitivity.
    try:
        strict = build_bank(records, r['bank'], labels, strict=True)
        strict_model = Distance(strict)
        strict_d = {u: strict_model.score(records[u])[0] for u in r['heldout']}
        strict_pred = fit_head(dev, confirm, r['groups'], z, strict_d, labels)
        sensitivity = dict(bank_vectors=len(strict['x']), auc=auc(y, strict_pred),
                           delta_vs_primary=auc(y, strict_pred)-primary['auc']['matched'] if primary['auc']['matched'] is not None else None,
                           missing_studies=sum(not np.isfinite(v) for v in strict_d.values()),
                           caveat='label strictness sensitivity, not proof of clinically normal bank')
    except ValueError as exc:
        if str(exc) != 'insufficient reference bank':
            raise
        sensitivity = dict(bank_vectors=0, auc=None, reason='insufficient strict-negative references')
    result = dict(version=VERSION, **EXPOSURE, primary=primary, report_rulers=report_rulers,
                  bank_vectors=len(bank['x']), strict_bank_sensitivity=sensitivity, lopo=lopo,
                  top50=top, top50_retrieval={u: models['matched'].score(records[u], evidence=True)[2] for u in top},
                  metadata_columns=['plane','contrast','fatsat','view','relative_bin','grid_cell','series_index','native_slice','full_fov_bin','phys_step'],
                  top50_protocol_mix=dict(collections.Counter(protocols[u] for u in top)),
                  confirmation_protocol_mix=dict(collections.Counter(protocols[u] for u in confirm)),
                  coverage={k: dict(mean=float(np.mean(list(v.values()))),
                                    missing_studies=sum(not np.isfinite(ds[k][u]) for u in r['heldout'])) for k,v in coverage.items()},
                  decision=rejection(primary['auc'], lopo, qa, top))
    if any(not np.isfinite(ds[k][u]) or coverage[k][u] < .5 for k in models for u in r['heldout']):
        result['decision']['holds'].append('missing study distance or patch coverage below 0.5')
        result['decision']['verdict'] = 'HOLD'
    return result


def load_shard(path, uid, identity):
    import numpy as np
    seal = read(path.with_suffix('.json'))
    need(seal['registration_sha256'] == identity and seal['sha256'] == sha(path), 'shard integrity/registration mismatch')
    with np.load(path, allow_pickle=False) as data:
        row = {k: data[k].copy() for k in data.files}
    need(str(row['uid']) == uid and row['x'].ndim == 2 and row['x'].shape[1] == 32 and
         row['meta'].shape == (len(row['x']), 10) and np.isfinite(row['x']).all() and
         np.isfinite(row['logit']).all(), 'invalid shard')
    return row


def shard_path(folder, uid):
    return folder / (hashlib.sha256(uid.encode()).hexdigest() + '.npz')


class Watchdog:
    def __init__(self, folder):
        self.folder, self.last, self.phase = folder, time.monotonic(), 'start'
        self.commits, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def progress(self, phase):
        self.last, self.phase = time.monotonic(), phase
        safeio.receipt(self.folder / 'status.json', dict(phase=phase, progress_time=time.time(), committed=len(self.commits)))

    def failed(self, now, telemetry=None):
        if telemetry and now-self.last >= 600:
            util, watts, used, total = telemetry
            if util >= 95 and watts < 90 and used >= .9*total:
                return 'WDDM stall signature and no progress for 10 minutes'
        if now-self.last >= 900:
            return 'no progress for 15 minutes'
        if len(self.commits) >= 6:
            healthy = 5 / (self.commits[5]-self.commits[0])
            if now-self.commits[5] >= 1800:
                recent = sum(t >= now-1800 for t in self.commits) / 1800
                if recent < .25*healthy:
                    return 'commit rate below 25% healthy for 30 minutes'
        return None

    def run(self):
        while not self.stop.wait(30):
            telemetry = None
            try:
                proc = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,power.draw,memory.used,memory.total',
                    '--format=csv,noheader,nounits', '--id=0'], capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                if proc.returncode == 0:
                    telemetry = tuple(map(float, proc.stdout.strip().split(',')))
                    if len(telemetry) != 4:
                        telemetry = None
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass  # Best-effort telemetry never blocks inference.
            reason = self.failed(time.monotonic(), telemetry)
            if reason:
                safeio.receipt(self.folder / 'failure.json', dict(reason=reason, phase=self.phase, resumable=True))
                os._exit(70)  # Single process, no loaders/children; GPU context dies with owner.

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(2)


def memory_gate(before, peak, total):
    need(peak <= 12 * 1024**3 and before + peak + 2 * 1024**3 <= total, 'measured VRAM admission failed')


def extract():
    r = validate(True)
    review_gate()
    from tools.hb145_r7_heldout_v1 import exclusive_run
    folder = OUT / 'gpu'
    folder.mkdir(exist_ok=True)
    with exclusive_run(folder), Watchdog(folder) as watch:
        torch = setup_runtime(False)
        need(torch.cuda.is_available(), 'CUDA required; B launches extraction')
        free, total = torch.cuda.mem_get_info()
        before = total-free
        torch.cuda.reset_peak_memory_stats()
        engine = Engine('cuda', folder / 'work', watch.progress)
        identity = sha(OUT / 'registration.json')
        try:
            for uid in r['bank'] + r['heldout']:
                path = shard_path(folder, uid)
                if path.exists() and path.with_suffix('.json').exists():
                    load_shard(path, uid, identity)
                    watch.progress('resume_verified')
                    continue
                row = engine.study(uid)
                peak = torch.cuda.max_memory_reserved()
                memory_gate(before, peak, total)
                atomic_npz(path, **row)
                safeio.atomic_json(path.with_suffix('.json'), dict(sha256=sha(path), registration_sha256=identity))
                watch.commits.append(time.monotonic())
                watch.progress('shard_committed')
                safeio.receipt(folder / 'memory.json', dict(baseline=before, peak_reserved=peak, total=total, pass_=True))
            safeio.atomic_json(folder / 'complete.json', dict(registration_sha256=identity, count=len(r['bank'])+len(r['heldout'])))
        finally:
            engine.close()


def cpu_smoke():
    r = validate(True)
    work = OUT / 'cpu-smoke'
    work.mkdir(exist_ok=True)
    # TWO real complete study forwards, including full preprocessing, all windows,
    # both unhooked and hooked. One normal training reference and one held-out study.
    selected = [r['bank'][0], r['heldout'][0]]
    engine = Engine('cpu', work)
    rows, timing = {}, []
    try:
        for i, uid in enumerate(selected):
            started = time.monotonic()
            print(f'CPU real study {i+1}/2 started', flush=True)
            row = engine.study(uid, identity=True)
            timing.append(time.monotonic()-started)
            rows[uid] = row
            atomic_npz(shard_path(work, uid), **row)
            print(f'CPU real study {i+1}/2 identity PASS; patches={len(row["x"])} seconds={timing[-1]:.1f}', flush=True)
    finally:
        engine.close()
    # Real feature bank/control retrieval, no scientific fit on these two patients.
    labels = own_labels(r['labels_path'])
    bank = build_bank(rows, [selected[0]], labels)
    matched = Distance(bank).score(rows[selected[1]], evidence=True)
    shuffled = Distance(shuffle_bank(bank)).score(rows[selected[1]])
    import numpy as np
    need(np.isfinite(matched[0]) and np.isfinite(shuffled[0]), 'real tiny retrieval has no persistent matches')
    safeio.atomic_json(work/'retrieval.json', dict(query_uid=selected[1], evidence=matched[2],
        metadata_columns=['plane','contrast','fatsat','view','relative_bin','grid_cell','series_index','native_slice','full_fov_bin','phys_step']))
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (384, 3*212), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, patch in enumerate(matched[2]['patches']):
        for j, (u, m) in enumerate([(selected[1],patch['query_meta']), (patch['nearest_bank_uid'],patch['nearest_bank_meta'])]):
            idx = np.flatnonzero((rows[u]['meta'] == np.asarray(m)).all(axis=1))[0]
            tile = Image.fromarray(rows[u]['preview'][idx//9]).convert('RGB')
            td = ImageDraw.Draw(tile); cell = m[5]
            cx, cy = (.3,.5,.7)[cell%3]*192, (.4,.5,.6)[cell//3]*192
            td.rectangle((cx-6,cy-6,cx+6,cy+6), outline='red', width=2)
            canvas.paste(tile, (j*192,i*212))
            draw.text((j*192,i*212+193), f'{"query" if j == 0 else "reference"} s{m[6]} slice{m[7]} c{cell}', fill='black')
    canvas.save(work/'retrieval.png')
    result = dict(pass_=True, version=VERSION, registration_sha256=sha(OUT/'registration.json'),
                  real_studies=2, complete_forward=True, hook_byte_identity=True,
                  logits_sha256=[str(rows[u]['logits_sha']) for u in selected], seconds=timing,
                  real_bank_vectors=len(bank['x']), real_query_coverage=[matched[1], shuffled[1]],
                  contrast_source='DICOM per-slice matching-only; frozen inputs unchanged',
                  contrast_counts=[dict(collections.Counter(':'.join(map(str,m[:3])) for m in rows[u]['meta'])) for u in selected],
                  real_tiny_scope='raw DICOM -> exact R7 -> dense hook -> bank/shuffle -> persistent distance; synthetic fit tested separately',
                  **EXPOSURE)
    result['pass'] = result.pop('pass_')
    safeio.atomic_json(OUT / 'cpu-smoke.json', result)
    print('CPU smoke PASS', flush=True)


def screen():
    setup_runtime(True)
    r = validate()
    review_gate()
    identity = sha(OUT / 'registration.json')
    done = read(OUT / 'gpu/complete.json')
    need(done['registration_sha256'] == identity, 'extraction registration differs')
    rows = {u: load_shard(shard_path(OUT / 'gpu', u), u, identity) for u in r['bank'] + r['heldout']}
    from tools.exp156_k_sweep import references
    qa = read(OUT / 'image-qa.json') if (OUT / 'image-qa.json').exists() else None
    if qa:
        need(qa['registration_sha256'] == identity, 'image QA registration differs')
    result = analyze(r, rows, own_labels(r['labels_path']), references(r['confirmation']), qa)
    result['registration_sha256'] = identity
    result['result_review_required'] = True
    safeio.atomic_json(OUT / 'screen.json', result)
    if qa is None:
        safeio.atomic_json(OUT / 'image-qa-template.json', dict(registration_sha256=identity,
            reviewer='', bank_retrieval_position_pass=None, top50={u: None for u in result['top50']},
            instructions='Inspect native images blinded to labels. True means deviation mainly protocol/FOV; false means not. Never enter report text. Save completed copy as image-qa.json.'))
    print(json.dumps(dict(verdict=result['decision']['verdict'], output=str(OUT / 'screen.json'), **EXPOSURE)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['register', 'validate', 'prepare', 'admit', 'cpu-smoke', 'extract', 'screen'])
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.command == 'register':
        register()
    elif a.command in ('validate', 'prepare'):
        validate(a.command == 'prepare')
        print(a.command.upper() + ' PASS (CPU only)')
    elif a.command == 'admit':
        need(a.dry_run, 'admit supports --dry-run only; actual memory measured by B extraction')
        validate()
        review_gate()
        print('ADMIT CPU PASS; real GPU peak gate remains mandatory at extraction')
    elif a.command == 'cpu-smoke':
        cpu_smoke()
    elif a.command == 'extract':
        extract()
    else:
        screen()


if __name__ == '__main__':
    main()
