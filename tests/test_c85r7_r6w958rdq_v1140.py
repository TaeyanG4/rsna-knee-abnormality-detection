"""H-B-190: reader overlay on admitted H-B-189 — weights_only load, source semantics, truth tables, fallbacks, scope.

Embedded code always runs in an isolated interpreter (-I) with only the candidate runtime on sys.path, so
the frozen tools/* modules can never shadow the embedded members. CPU only (CUDA_VISIBLE_DEVICES='').
"""
import ast
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tools import c85r7_r6w958rdq_v1140 as b

ROOT = b.ROOT
PROMPT_A = {'ACL': .025, 'MCL': .020, 'Medial Meniscus': .020, 'Lateral Meniscus': .008, 'Medial OA': .025,
            'Lateral OA': .008, 'PF OA': .008, 'Effusion': .004, 'Synovitis': .003, "Baker's": .035,
            'Contusion': .030, 'Fracture': .008}
PRELUDE = r'''
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
runtime = Path(sys.argv[1]); ARGS = sys.argv[2:]
sys.path.insert(0, str(runtime))
import numpy as np
import exp169_release_c85r7codec as c
assert Path(c.__file__).resolve().parent == runtime.resolve()
assert all(not n.startswith('tools.') for n in sys.modules)
'''


def embedded(runtime, body, *args, timeout=1800):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    env.pop('PYTHONPATH', None)
    env.pop('C85_COMPETITION', None)
    run = subprocess.run([sys.executable, '-B', '-I', '-c', PRELUDE + body, str(runtime), *map(str, args)],
                         capture_output=True, text=True, env=env, timeout=timeout)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-6000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    files = b.payload()[2]
    folder = tmp_path_factory.mktemp('hb190-runtime')
    for name, data in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    receipt = b.OUT / 'build-receipt.json'
    if receipt.exists():  # the tested bytes are the staged bytes
        assert json.loads(receipt.read_text())['embedded'] == {k: b.digest(v) for k, v in files.items()}
    return folder


# ----------------------------------------------------------------------------------------------- weights / source
def test_weights_only_load_all_three_checkpoints_and_strict_state_dict(runtime):
    import torch
    for name, digest in b.CHECKPOINTS:
        path = b.READER_STORE / 'extracted' / name
        assert b.sha_file(path) == digest
        ck = torch.load(path, map_location='cpu', weights_only=True)
        assert set(ck) == {'args', 'model'} and ck['args']['res'] == 256
        assert ck['args']['backbone'].split('.')[0] == 'convnext_tiny'
    out = embedded(runtime, r'''
import torch
models = c.reader_models(ARGS, torch.device('cpu'))
print(json.dumps(dict(n=len(models), res=[r for _, r in models],
    params=[sum(p.numel() for p in m.parameters()) for m, _ in models], training=[m.training for m, _ in models])))
''', *b.checkpoints())
    assert out['n'] == 3 and out['res'] == [256] * 3 and out['training'] == [False] * 3
    assert len(set(out['params'])) == 1 and out['params'][0] > 30_000_000


def test_no_pickle_fallback_weights_only_refuses_code_objects(runtime, tmp_path):
    import argparse
    import torch
    bad = tmp_path / 'bad.pt'
    torch.save({'args': argparse.Namespace(backbone='convnext_tiny', res=256), 'model': {}}, bad)
    out = embedded(runtime, r'''
import torch
try:
    c.reader_models(ARGS, torch.device('cpu')); print(json.dumps(dict(raised=False)))
except Exception as exc:
    print(json.dumps(dict(raised=True, kind=type(exc).__name__)))
''', bad)
    assert out['raised'] is True
    source = b.BLOCK.read_text(encoding='utf-8')
    assert source.count('torch.load(') == 1 and 'weights_only=True' in source and 'weights_only=False' not in source


def test_a_t_literal_from_public_source_matches_prompt_and_embedded(runtime):
    assert b.a_from_source() == b.A == tuple(PROMPT_A[label] for label in b.LABELS)
    out = embedded(runtime, r'''
print(json.dumps(dict(a=list(c.READER_A), labels=list(c.READER_LABELS), p=list(c.p.LABELS), k=c.READER_K,
                      bs=c.READER_BATCH, workers=c.READER_WORKERS, idle=c.READER_IDLE_SECONDS, slug=c.READER_SLUG,
                      ck=[list(x) for x in c.READER_CHECKPOINTS])))
''')
    assert tuple(out['a']) == b.A and tuple(out['labels']) == b.LABELS == tuple(out['p'])
    assert (out['k'], out['bs'], out['workers'], out['idle']) == (16, 4, 4, 900) and out['slug'] == b.READER_SLUG
    assert [tuple(x) for x in out['ck']] == list(b.CHECKPOINTS)


def test_notebook_own_src_differs_from_dataset_code_by_exactly_three_semantic_hunks():
    text = b.SOURCE.read_bytes().decode('utf-8').replace('\r\n', '\n')
    store = b.READER_STORE / 'extracted'
    hunks = {}
    for name in ('preprocess.py', 'knee.py', 'infer.py'):
        m = re.search(r'%%writefile /kaggle/working/own_src/' + re.escape(name) + r'\n(.*?)(?=\n# ---- (code|markdown) cell ----)',
                      text, re.S)
        ds = (store / name).read_bytes().decode('utf-8').replace('\r\n', '\n')
        nb = m.group(1).replace('\r\n', '\n')
        hunks[name] = [l for l in difflib.unified_diff(ds.splitlines(), nb.splitlines(), lineterm='', n=0)
                       if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    assert hunks['preprocess.py'] == []
    assert '+def run(data_dir, split, ckpts, k=16, studies=None, workers=4, bs=4):' in hunks['infer.py']
    assert len(hunks['infer.py']) == 4
    added = '\n'.join(l for l in hunks['knee.py'] if l.startswith('+'))
    assert 'slots[fs_s] = list(slots[nonfs_s])' in added and 'slots[nonfs_s] = list(slots[fs_s])' in added
    assert 'kw["img_size"] = img_size' in added  # inert for convnext_tiny (vit/eva/beit only)


# ----------------------------------------------------------------------------------------------- table / windows
def test_table_counterpart_truth_table_and_robust_rows(runtime):
    out = embedded(runtime, r'''
rows = [('A','a1',1,'Sagittal'), ('A','a2',0,'Coronal'), ('A','a3',1,'Axial'), ('A','a4',0,'Axial'),
        ('B','b1',1,'Sagittal'), ('B','b2',1,'Sagittal'), ('B','b3',0,'Oblique'), ('B','b4',2,'Coronal'),
        ('B','b5',float('nan'),'Coronal'), ('C','c1',1,'Coronal'), ('C','c2',0,'Coronal')]
n = {'a1':20,'a2':30,'a3':2,'a4':40,'b1':10,'b2':25,'b3':30,'b4':30,'b5':30,'c1':5,'c2':6}
t, skipped = c.reader_table(rows, n, True)
t0, _ = c.reader_table(rows, n, False)
print(json.dumps(dict(t=t, t0=t0, skipped=skipped)))
''')
    t, t0 = out['t'], out['t0']
    # Slots: 0 sag-FS, 1 cor-FS, 2 ax-FS, 3 sag, 4 cor, 5 ax. a3 has 2 files (< 3) and is excluded.
    assert t0['A'] == [['a1'], [], [], [], ['a2'], ['a4']]
    assert t['A'] == [['a1'], ['a2'], ['a4'], ['a1'], ['a2'], ['a4']]
    assert t0['B'] == [['b2', 'b1'], [], [], [], [], []] and t['B'] == [['b2', 'b1'], [], [], ['b2', 'b1'], [], []]
    assert t['C'] == t0['C'] == [[], ['c1'], [], [], ['c2'], []]
    assert out['skipped'] == 3  # unknown plane, fat=2, NaN fat


def test_window_centres_match_literal_formula(runtime):
    out = embedded(runtime, r'''
res = {f'{n}-{k}': c.reader_window_centres(n, k).tolist() for n in range(1, 90) for k in (12, 16)}
print(json.dumps(res))
''')
    for n in range(1, 90):
        for k in (12, 16):
            a, bb = max(1.0, .04 * (n - 1)), min(n - 2.0, .96 * (n - 1))
            if bb < a:
                a = bb = (n - 1) / 2
            expected = np.clip(np.round(np.linspace(a, bb, k)).astype(int), 1, max(1, n - 2)).tolist()
            assert out[f'{n}-{k}'] == expected


# ----------------------------------------------------------------------------------------------- parity B (2 real)
ORACLE = r'''
# Independent literal re-statement of the 0.958 notebook own_src (preprocess.load_series, knee.build_study_table
# with counterpart fill, infer.DicomStudyDataset k=16), written from the SOURCE text; no embedded helper used.
import cv2, pydicom, pandas as pd
MM, SIZE, MAXS = 0.4, 384, 64
CANON = {0: (+1, (1, +1), (2, -1)), 1: (+1, (0, +1), (2, -1)), 2: (-1, (0, +1), (1, +1))}
PLANES = ["Sagittal", "Coronal", "Axial"]
def rd(path):
    ds = pydicom.dcmread(path); arr = ds.pixel_array.astype(np.float32)
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1": arr = arr.max() - arr
    sl, ic = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
    if sl is not None and ic is not None: arr = arr * float(sl) + float(ic)
    iop = [float(v) for v in ds.ImageOrientationPatient] if "ImageOrientationPatient" in ds else None
    ipp = [float(v) for v in ds.ImagePositionPatient] if "ImagePositionPatient" in ds else None
    ps = [float(v) for v in ds.PixelSpacing] if "PixelSpacing" in ds else None
    inst = float(ds.get("InstanceNumber") or 0)
    frames = list(arr) if arr.ndim == 3 else [arr]
    return [(f, iop, ipp, ps, inst + i * 1e-3) for i, f in enumerate(frames)]
def canon(img, iop, ps, plane):
    ps = ps or [MM * img.shape[0] / SIZE] * 2
    if iop is None: return img, ps
    r, cc = np.array(iop[:3]), np.array(iop[3:]); _, (cax, csgn), (rax, rsgn) = CANON[plane]
    if abs(r[cax]) < abs(r[rax]): img, r, cc, ps = img.T, cc, r, ps[::-1]
    if r[cax] * csgn < 0: img = img[:, ::-1]
    if cc[rax] * rsgn < 0: img = img[::-1]
    return img, ps
def resample(img, ps):
    out = np.zeros((SIZE, SIZE), np.float32); h, w = img.shape
    ch = min(h, int(round(SIZE * MM / ps[0]))); cw = min(w, int(round(SIZE * MM / ps[1])))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
    nh = max(1, min(SIZE, int(round(ch * ps[0] / MM)))); nw = max(1, min(SIZE, int(round(cw * ps[1] / MM))))
    res = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if nh < ch else cv2.INTER_LINEAR)
    oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2; out[oy:oy + nh, ox:ox + nw] = res; return out
def load_series(sdir):
    slices = []
    for fn in os.listdir(sdir):
        try: slices.extend(rd(os.path.join(sdir, fn)))
        except Exception: pass
    if not slices: return np.zeros((1, SIZE, SIZE), np.uint8)
    iop = next((s[1] for s in slices if s[1] is not None), None)
    if iop is not None: normal = np.cross(iop[:3], iop[3:]); plane = int(np.abs(normal).argmax())
    else: normal, plane = None, 0
    if normal is not None and all(s[2] is not None for s in slices):
        sgn = CANON[plane][0] * np.sign(normal[plane]); slices.sort(key=lambda s: float(np.dot(s[2], normal)) * sgn)
    else: slices.sort(key=lambda s: s[4])
    if len(slices) > MAXS:
        keep = np.linspace(0, len(slices) - 1, MAXS).round().astype(int); slices = [slices[i] for i in keep]
    vol = np.stack([resample(*canon(s[0], s[1], s[3], plane)) for s in slices])
    lo, hi = np.percentile(vol[:, ::4, ::4], [1, 99.5]); vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return (vol * 255).round().astype(np.uint8)
def centres(n, k):
    a, bb = max(1.0, 0.04 * (n - 1)), min(n - 2.0, 0.96 * (n - 1))
    if bb < a: a = bb = (n - 1) / 2
    return np.clip(np.round(np.linspace(a, bb, k)).astype(int), 1, max(1, n - 2))
studies, series_csv, root = json.loads(ARGS[0]), ARGS[1], ARGS[2]
se = pd.read_csv(series_csv); se = se[se.StudyInstanceUID.isin(set(studies))]
nf = {}
for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
    d = f"{root}/{st}/{sr}"; nf[sr] = min(len(os.listdir(d)), MAXS) if os.path.isdir(d) else 0
table = {}
for st, sr, fs, pl in zip(se.StudyInstanceUID, se.SeriesInstanceUID, se.Fat_Suppression, se.Anatomical_Plane):
    if nf.get(sr, 0) < 3: continue
    table.setdefault(st, [[] for _ in range(6)])[PLANES.index(pl) + 3 * (1 - int(fs))].append(sr)
for slots in table.values():
    for cands in slots: cands.sort(key=lambda s: -nf[s])
    for p_ in range(3):
        if not slots[p_] and slots[p_ + 3]: slots[p_] = list(slots[p_ + 3])
        elif not slots[p_ + 3] and slots[p_]: slots[p_ + 3] = list(slots[p_])
K = 16
result = {}
import torch
emb_table, _ = c.reader_table(list(zip(se.StudyInstanceUID.astype(str), se.SeriesInstanceUID.astype(str),
                                       se.Fat_Suppression, se.Anatomical_Plane)), nf, True)
data = c.ReaderStudies(studies, emb_table, root, K)
xs, ms, ps_ = [], [], []
for i, st in enumerate(studies):
    x = np.zeros((6, K, 3, SIZE, SIZE), np.uint8); mask = np.zeros(6, bool); pos = np.zeros((6, K), np.float32)
    for s, cands in enumerate(table.get(st, [[] for _ in range(6)])):
        if not cands: continue
        vol = load_series(f"{root}/{st}/{cands[0]}"); n = len(vol)
        if n < 3: continue
        cc = centres(n, K); x[s] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, n - 1)] for j in cc])
        mask[s], pos[s] = True, cc / max(n - 1, 1)
    item = data[i]
    result[st] = dict(x=bool(np.array_equal(item['x'], x)), mask=bool(np.array_equal(item['mask'], mask)),
                      pos=bool(np.array_equal(item['pos'], pos)), error=item['error'], slots=int(mask.sum()),
                      table_equal=emb_table.get(st) == table.get(st))
    xs.append(x); ms.append(mask); ps_.append(pos)
models = c.reader_models(json.loads(ARGS[3]), torch.device('cpu'))
torch.set_num_threads(6)
probs = c.reader_forward(models, torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms)),
                         torch.from_numpy(np.stack(ps_)), torch.device('cpu'))
print(json.dumps(dict(result=result, probs=probs.tolist())))
'''


def test_reader_parity_two_real_studies_vs_literal_notebook_semantics(runtime):
    ids = b.gold_ids()[:2]
    out = embedded(runtime, ORACLE, json.dumps(ids), b.GOLD_DICOM / 'train_series.csv', b.GOLD_DICOM / 'train_series',
                   json.dumps(b.checkpoints()))
    for uid in ids:
        r = out['result'][uid]
        assert r['x'] and r['mask'] and r['pos'] and r['table_equal'] and r['error'] == '' and r['slots'] >= 3
    journal = b.SCRATCH / 'reader-runs/serving2/journal.jsonl'
    if journal.exists():  # the smoke run's serving predictions are the same model on the same tensors
        rows = b.journal_rows(journal)
        assert [r['uid'] for r in rows] == ids
        np.testing.assert_allclose(np.asarray([r['p'] for r in rows]), np.asarray(out['probs']), atol=2e-5, rtol=0)


# ----------------------------------------------------------------------------------------------- overlay arithmetic
def oracle(base, q, ok, a=b.A):
    result = np.empty_like(base)
    for j, w in enumerate(a):
        rf = pd.Series(base[:, j]).rank(method='average', pct=True).to_numpy()
        rq = rf.copy()
        rq[ok] = pd.Series(q[ok, j]).rank(method='average', pct=True).to_numpy()
        result[:, j] = pd.Series((1 - w) * rf + w * rq).rank(method='average', pct=True).to_numpy()
    return result


BLEND = r'''
data = np.load(ARGS[0], allow_pickle=False)
ids = [str(x) for x in data['ids']]
reader = {u: data['q'][i].tolist() for i, u in enumerate(ids) if data['ok'][i]}
y, stats = c.reader_blend(ids, data['base'], reader)
if y is not None: np.save(ARGS[1], y, allow_pickle=False)
print(json.dumps(dict(stats=stats, none=y is None)))
'''


@pytest.mark.parametrize('case', ['healthy', 'ties', 'partial', 'one_ok', 'none_ok', 'constant'])
def test_blend_independent_pandas_oracle_and_whole_fallbacks(runtime, tmp_path, case):
    rng = np.random.default_rng(190)
    n = 1500
    base = rng.random((n, 12))
    q = rng.random((n, 12))
    ok = np.ones(n, bool)
    if case == 'ties':
        base, q = np.round(base * 20) / 20, np.round(q * 10) / 10
    elif case == 'partial':
        ok[rng.choice(n, 300, replace=False)] = False
    elif case == 'one_ok':
        ok[:] = False; ok[7] = True
    elif case == 'none_ok':
        ok[:] = False
    elif case == 'constant':
        q[:, 9] = .25
    src = tmp_path / 'in.npz'
    np.savez(src, ids=np.asarray([f's{i:05d}' for i in range(n)]), base=base, q=q, ok=ok)
    out = embedded(runtime, BLEND, src, tmp_path / 'y.npy')
    if case in ('one_ok', 'none_ok', 'constant'):
        assert out['none'] is True and out['stats']['reason']
        return
    y = np.load(tmp_path / 'y.npy')
    assert y.tobytes() == oracle(base, q, ok).tobytes()
    assert out['stats']['fallback_studies'] == int((~ok).sum()) and out['stats']['reader_ok'] == int(ok.sum())


def test_column_truth_table_each_target_uses_its_own_a_t(runtime, tmp_path):
    """Per column: the embedded result equals the oracle with a_t and differs when a_t is moved by +-0.001;
    perturbing Q in one column changes only that column."""
    rng = np.random.default_rng(1902)
    n = 4000
    base, q = rng.random((n, 12)), rng.random((n, 12))
    ok = np.ones(n, bool)
    src = tmp_path / 'in.npz'
    np.savez(src, ids=np.asarray([f's{i:05d}' for i in range(n)]), base=base, q=q, ok=ok)
    embedded(runtime, BLEND, src, tmp_path / 'y.npy')
    y = np.load(tmp_path / 'y.npy')
    for j, w in enumerate(b.A):
        assert y[:, j].tobytes() == oracle(base, q, ok)[:, j].tobytes()
        for moved in (w - .001, w + .001):
            a = list(b.A); a[j] = moved
            assert not np.array_equal(y[:, j], oracle(base, q, ok, tuple(a))[:, j]), (j, moved)
    q2 = q.copy(); q2[:, 4] = rng.random(n)
    np.savez(src, ids=np.asarray([f's{i:05d}' for i in range(n)]), base=base, q=q2, ok=ok)
    embedded(runtime, BLEND, src, tmp_path / 'y2.npy')
    y2 = np.load(tmp_path / 'y2.npy')
    changed = [j for j in range(12) if not np.array_equal(y[:, j], y2[:, j])]
    assert changed == [4]


def test_journal_parsing_failures_duplicates_and_partial_tail(runtime, tmp_path):
    good = [0.5] * 12
    lines = [dict(event='start'), dict(i=0, uid='u0', ok=True, p=good), dict(i=1, uid='u1', ok=False, p=None, error='x'),
             dict(i=2, uid='u2', ok=True, p=[0.1] * 11), dict(i=3, uid='u3', ok=True, p=[float('nan')] + [0.1] * 11),
             dict(i=4, uid='u4', ok=True, p=[1.5] + [0.1] * 11), dict(i=5, uid='u5', ok=True, p=[True] + [0.1] * 11),
             dict(i=6, uid='u6', ok=True, p=good), dict(i=6, uid='u6', ok=True, p=good),
             dict(i=7, uid='zz', ok=True, p=good), dict(i=8, uid='u8', ok='yes', p=good), dict(i=9, uid='u9', ok=True, p=good)]
    raw = b''.join((json.dumps(x) + '\n').encode() for x in lines) + b'{"i": 10, "uid": "u10", "ok": true, "p": [0.5'
    path = tmp_path / 'journal.jsonl'
    path.write_bytes(raw + b'\n' + b'{"uid": "u11", "ok": true')  # one unreadable complete line, one partial tail
    out = embedded(runtime, r'''
reader, stats = c.reader_journal(Path(ARGS[0]), [f'u{i}' for i in range(12)])
print(json.dumps(dict(ok=sorted(reader), stats=stats)))
''', path)
    assert out['ok'] == ['u0', 'u9']
    s = out['stats']
    assert s['unreadable_lines'] == 1 and s['unknown'] == 1 and s['missing'] == 3  # u7, u10, u11
    assert s['journal_failed'] == 7  # u1..u6 (u6 duplicate) and u8


# ----------------------------------------------------------------------------------------------- phase fallbacks
def base_csv(n=6, seed=3):
    rng = np.random.default_rng(seed)
    ids = [f'1.2.3.{i}' for i in range(n)]
    frame = pd.DataFrame(rng.random((n, 12)).astype(np.float32), columns=list(b.LABELS))
    frame.insert(0, 'StudyInstanceUID', ids)
    return ids, frame.to_csv(index=False, lineterminator='\n').encode()


PHASE = r'''
import time
output = Path(ARGS[0]); mode = ARGS[1]
(output / c.RUN).mkdir(parents=True, exist_ok=True)
if mode != 'real_child':
    def fake_wait(args, *, env, journal, **kw):
        assert args[3] == 'reader-worker' and env.get('KNEE_CODEC_BRANCH') is None
        if mode in ('applied', 'apply_error'):
            Path(journal).write_bytes(Path(ARGS[2]).read_bytes())
    c.reader_wait = fake_wait
if mode == 'apply_error':
    def broken(sub, work):
        sub.write_bytes(b'garbage'); raise RuntimeError('injected overlay failure')
    c.reader_apply = broken
env = c.child_base_env(runtime)
record = c.reader_phase(runtime, output, {'status': 'INACTIVE'}, env, output / c.RUN / 'codecs' / 'telemetry')
exit_record = json.loads((output / c.RUN / c.READER_DIR / 'READER-exit.json').read_text())
receipt = json.loads((output / c.RUN / c.READER_DIR / 'reader-receipt.json').read_text())
print(json.dumps(dict(record={k: record.get(k) for k in ('status', 'reason', 'error', 'reader_ok', 'fallback_studies',
      'base_sha256', 'submission_sha256')}, exit=exit_record, receipt_status=receipt['status'])))
'''


@pytest.mark.parametrize('mode', ['real_child', 'no_rows', 'applied', 'apply_error'])
def test_reader_phase_never_breaks_the_h189_submission(runtime, tmp_path, mode):
    ids, data = base_csv()
    out_dir = tmp_path / 'working'
    out_dir.mkdir()
    (out_dir / 'submission.csv').write_bytes(data)
    journal = tmp_path / 'journal.jsonl'
    rng = np.random.default_rng(9)
    q = rng.random((len(ids), 12))
    journal.write_bytes(b''.join((json.dumps(dict(i=i, uid=u, ok=True, p=q[i].tolist())) + '\n').encode()
                                 for i, u in enumerate(ids)))
    out = embedded(runtime, PHASE, out_dir, mode, journal, timeout=600)
    final = (out_dir / 'submission.csv').read_bytes()
    rec = out['record']
    assert rec['base_sha256'] == hashlib.sha256(data).hexdigest() and out['receipt_status'] == rec['status']
    if mode == 'applied':
        assert rec['status'] == 'APPLIED' and rec['fallback_studies'] == 0
        frame = pd.read_csv(io.BytesIO(data), dtype={'StudyInstanceUID': str})
        expected = oracle(frame[list(b.LABELS)].to_numpy(float), q, np.ones(len(ids), bool))
        exp = pd.DataFrame(expected.astype(np.float32), columns=list(b.LABELS))
        exp.insert(0, 'StudyInstanceUID', ids)
        assert final == exp.to_csv(index=False, lineterminator='\n').encode()
    else:
        assert final == data, mode  # byte-identical H-B-189 output
        assert rec['submission_sha256'] == rec['base_sha256']
        assert rec['status'] == ('FALLBACK_ERROR' if mode == 'apply_error' else 'FALLBACK_ALL')
    if mode == 'real_child':  # the real worker exits nonzero without Kaggle mounts
        assert out['exit']['exit_code'] == 1 and 'exit code' in out['exit']['error']


def test_child_failure_after_partial_journal_keeps_completed_studies(runtime, tmp_path):
    """Nonzero exit / watchdog kill after some rows: completed studies blend, the rest keep r(F)."""
    ids, data = base_csv(8, seed=4)
    out_dir = tmp_path / 'working'
    out_dir.mkdir()
    (out_dir / 'submission.csv').write_bytes(data)
    q = np.random.default_rng(11).random((len(ids), 12))
    journal = tmp_path / 'journal.jsonl'
    rows = [(json.dumps(dict(i=i, uid=u, ok=True, p=q[i].tolist())) + '\n').encode() for i, u in enumerate(ids[:5])]
    journal.write_bytes(b'{"event": "start"}\n' + b''.join(rows) + b'{"i": 5, "uid": "' + ids[5].encode() + b'", "ok": tr')
    out = embedded(runtime, r'''
output = Path(ARGS[0]); (output / c.RUN).mkdir(parents=True, exist_ok=True)
def dying_wait(args, *, env, journal, **kw):
    Path(journal).write_bytes(Path(ARGS[1]).read_bytes())
    raise RuntimeError('reader child failed with exit code 1')
c.reader_wait = dying_wait
record = c.reader_phase(runtime, output, {'status': 'INACTIVE'}, c.child_base_env(runtime), output / c.RUN / 'codecs' / 't')
exit_record = json.loads((output / c.RUN / c.READER_DIR / 'READER-exit.json').read_text())
print(json.dumps(dict(status=record['status'], ok=record['reader_ok'], fb=record['fallback_studies'], exit=exit_record)))
''', out_dir, journal)
    assert out['status'] == 'APPLIED' and out['ok'] == 5 and out['fb'] == 3 and out['exit']['exit_code'] == 1
    frame = pd.read_csv(io.BytesIO(data), dtype={'StudyInstanceUID': str})
    mask = np.zeros(len(ids), bool); mask[:5] = True
    exp = pd.DataFrame(oracle(frame[list(b.LABELS)].to_numpy(float), q, mask).astype(np.float32), columns=list(b.LABELS))
    exp.insert(0, 'StudyInstanceUID', ids)
    assert (out_dir / 'submission.csv').read_bytes() == exp.to_csv(index=False, lineterminator='\n').encode()


def test_forward_retry_isolates_failing_studies(runtime, tmp_path):
    out = embedded(runtime, r'''
class Fake:
    def __init__(self, studies, table, root, k): self.studies, self.k = list(studies), k
    def __len__(self): return len(self.studies)
    def __getitem__(self, i):
        x = np.zeros((6, self.k, 3, 4, 4), np.uint8); mask = np.zeros(6, bool); mask[:2] = i != 3
        pos = np.full((6, self.k), float(i), np.float32)
        return {'x': x, 'mask': mask, 'pos': pos, 'index': i, 'error': 'decode boom' if i == 7 else ''}
def fake_forward(models, x, mask, pos, device):
    idx = [int(v) for v in pos[:, 0, 0]]
    if len(idx) > 1 and any(i in (1, 5) for i in idx): raise RuntimeError('batch fails')
    if idx == [5]: raise RuntimeError('study fails')
    return np.asarray([[0.2 + 0.01 * i] * 12 for i in idx], np.float32)
c.ReaderStudies, c.reader_models, c.reader_forward = Fake, (lambda paths, device: ['m']), fake_forward
ids = [f'u{i}' for i in range(10)]
summary = c.reader_predict(ids, [], ARGS[0], [], ARGS[1], device='cpu', k=2, batch=4, workers=0)
reader, stats = c.reader_journal(Path(ARGS[1]), ids)
print(json.dumps(dict(summary=summary, ok=sorted(reader), stats=stats)))
''', tmp_path / 'none', tmp_path / 'journal.jsonl')
    assert out['ok'] == ['u0', 'u1', 'u2', 'u4', 'u6', 'u8', 'u9']  # u3 no series, u5 forward, u7 decode error
    assert out['summary']['ok'] == 7 and out['summary']['failed'] == 3 and out['summary']['forward_retries'] == 2
    assert out['stats']['journal_failed'] == 3 and out['stats']['missing'] == 0


def test_idle_watchdog_kills_a_silent_child_and_spares_a_progressing_one(runtime, tmp_path):
    out = embedded(runtime, r'''
import time
journal = Path(ARGS[0])
t0 = time.monotonic()
try:
    c.reader_wait([sys.executable, '-c', 'import time; time.sleep(120)'], env=dict(os.environ), journal=journal,
                  idle_seconds=3, poll_seconds=0.5)
    killed = False
except RuntimeError as exc:
    killed = 'idle watchdog' in str(exc)
silent = time.monotonic() - t0
code = 'import time,sys\nfor i in range(8):\n    open(sys.argv[1],"a").write("x\\n"); time.sleep(1)\n'
t0 = time.monotonic()
c.reader_wait([sys.executable, '-c', code, str(journal)], env=dict(os.environ), journal=journal,
              idle_seconds=3, poll_seconds=0.5)
print(json.dumps(dict(killed=killed, silent=silent, progressing=time.monotonic() - t0)))
''', tmp_path / 'journal.jsonl', timeout=300)
    assert out['killed'] is True and out['silent'] < 30
    assert 7 <= out['progressing'] < 60


# ----------------------------------------------------------------------------------------------- scope / gate inputs
def test_strict_scope_one_member_and_metadata_delta():
    nb, meta, files, blob = b.payload()
    original = b.parent()[1]
    assert set(files) == set(original) and len(files) == 31
    assert [k for k in sorted(files) if files[k] != original[k]] == [b.WRAPPER]
    b.scope(nb, meta)
    pnb = b.parent()[0]
    assert pnb['cells'][:-1] == nb['cells'][:-1]  # FAST cell and markdown untouched
    a, c_ = ''.join(pnb['cells'][-1]['source']), ''.join(nb['cells'][-1]['source'])
    changed = [x for x in difflib.unified_diff(a.splitlines(), c_.splitlines(), lineterm='', n=0)
               if x[:1] in '+-' and not x.startswith(('+++', '---'))]
    assert len(changed) == 6 and all(re.match(r'[+-]\s*(BLOB = |PINS = |assert hashlib\.sha256\(blob\))', x) for x in changed)
    old_meta = b.old.read(b.BASE / 'kernel/kernel-metadata.json')
    assert {k for k in meta if meta[k] != old_meta[k]} == {'id', 'title', 'dataset_sources'}
    assert meta['dataset_sources'] == old_meta['dataset_sources'] + [b.READER_SLUG] and meta['title'] == 'knee-lb-r6w958rdq-v1'
    from tools import strict_variant_gate_r224_v145 as generic
    from tools import strict_variant_gate as g
    if (b.OUT / 'kernel').exists():
        pins = g.Inputs(ROOT)
        assert generic.metadata_delta(g.kernel(b.BASE, pins), g.kernel(b.OUT, pins), b.SLUG,
                                      json.loads((b.OUT / 'registration.json').read_text())) == \
            dict(added=[b.READER_SLUG], removed=[])


def test_wrapper_change_is_insertions_plus_four_version_lines_and_unchanged_launches():
    original, files = b.parent()[1], b.payload()[2]
    a = original[b.WRAPPER].decode().splitlines(True)
    c_ = files[b.WRAPPER].decode().splitlines(True)
    ops = difflib.SequenceMatcher(a=a, b=c_, autojunk=False).get_opcodes()
    removed = [x.rstrip('\n') for op, i, j, _, _ in ops if op in ('replace', 'delete') for x in a[i:j]]
    assert sorted(removed) == sorted(e[0] for e in b.VERSION_EDITS)
    text = files[b.WRAPPER].decode()
    tree = ast.parse(text)
    run_release = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_release')
    body = ast.get_source_segment(text, run_release)
    assert body.count('reader_phase(') == 1 and body.index('collect(adir, pdir, output)') < body.index('reader_phase(')
    # Every pre-existing function except run_release and main is byte-identical.
    old_tree = ast.parse(original[b.WRAPPER].decode())
    old_src = {n.name: ast.get_source_segment(original[b.WRAPPER].decode(), n) for n in old_tree.body
               if isinstance(n, ast.FunctionDef)}
    new_src = {n.name: ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {k for k in old_src if old_src[k] != new_src.get(k)} == {'run_release', 'main'}
    block = b.BLOCK.read_text(encoding='utf-8')
    btree = ast.parse(block)
    owners = {}
    for fn in [n for n in btree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]:
        for node in ast.walk(fn):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                for name in names:
                    owners.setdefault(name.split('.')[0], set()).add(fn.name)
    assert not [n for n in btree.body if isinstance(n, (ast.Import, ast.ImportFrom))]  # no module-level imports
    assert owners.get('subprocess') == {'reader_wait'}  # only the parent-side launcher spawns (the reader child)
    assert set(owners) <= {'numpy', 'pydicom', 'cv2', 'timm', 'torch', 'math', 'time', 'csv', 'pandas', 'subprocess'}
    calls = {node.func.id for node in ast.walk(btree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & {'exec', 'eval', 'compile', '__import__', 'open'}
    for forbidden in ('urllib', 'requests', 'socket', 'http.client', 'pip ', 'os.system', 'rmtree', 'weights_only=False'):
        assert forbidden not in block, forbidden


def test_stage_full_diff_holds_exact_gate_hunk_and_member_inventory():
    if not (b.OUT / 'new-members.json').exists():
        pytest.fail('prepare --dry-run must run before the stage tests')
    nb, meta, files, blob = b.payload()
    original = b.parent()[1]
    side = json.loads((b.OUT / 'new-members.json').read_text())
    assert side['added'] == side['removed'] == []
    assert side['changed'] == [dict(name=b.WRAPPER, before_sha256=b.digest(original[b.WRAPPER]),
                                    after_sha256=b.digest(files[b.WRAPPER]))]
    assert side['unchanged'] == [dict(name=k, before_sha256=b.digest(original[k]), after_sha256=b.digest(original[k]))
                                 for k in sorted(original) if k != b.WRAPPER]
    from tools import strict_variant_gate_r224_v143 as v143
    delta = v143.member_diff(b.WRAPPER, original[b.WRAPPER], files[b.WRAPPER])
    hunk = ''.join(delta['unified_diff'].splitlines(keepends=True)[2:]).encode()
    assert hunk and hunk in (b.OUT / b.DIFF).read_bytes()
    reg = json.loads((b.OUT / 'registration.json').read_text())
    assert reg['kernel_id'] == b.SLUG and reg['dataset_sources_delta_version'] == '1.0.0'
    assert reg['exposure'] == 'EXPOSED' and reg['clean_oof'] is False and reg['qualified_for_submission'] is False
    assert reg['keep_rule']['expected_public'] == .958 and reg['a'] == list(b.A)
    assert (b.OUT / 'registration.json').stat().st_mtime <= (b.OUT / 'kernel/release.ipynb').stat().st_mtime


def test_runtime_projection_within_kaggle_limit():
    path = b.SCRATCH / 'runtime-projection.json'
    if not path.exists():
        pytest.fail('cpu-replay/finalize must run before the runtime test')
    proj = json.loads(path.read_text())
    assert proj['total_hours']['conservative'] < 9 and proj['studies_measured'] == 58
    assert proj['reader_hours']['conservative'] >= proj['reader_hours']['central'] > 0
