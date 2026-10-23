"""H-B-190 builder 1.0.0: admitted H-B-189 + the 0.958 notebook's 2.5D ConvNeXt MIL reader overlay.

Changelog 1.0.0 (claude-opus-b2 stand-in, 2026-10-17): ONE candidate v1.1.40-r6w958rdq. Only the
wrapper member exp169_release_c85r7codec.py changes (inserted reader section from the pinned
tools/c85r7_r6w958rdq_v1140_reader_block.txt, a reader-worker command, one post-collect call in
run_release, version/id/release strings); 30 of 31 embedded files byte-identical; metadata id/title and a
registered dataset_sources append. The dead Codex job's partial files were moved to
.local/backups/partial-codex-1017/hb190/ and are not reused. No frozen/scored/original file is edited.
Public notebook/dataset code is read as text only; checkpoints load with weights_only=True only.
"""
from __future__ import annotations
import argparse
import base64
import difflib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r6w958q_v1139 as h189
old, fast, gate, reuse = h189.old, h189.fast, h189.gate, h189.reuse

VERSION = '1.1.40-r6w958rdq'
SLUG = 'taeyangg4/knee-lb-r6w958rdq-v1'
PARENT_VERSION = '1.1.39-r6w958q'
PARENT_SLUG = 'taeyangg4/knee-lb-r6w958q-v1'
BASE = ROOT / '.local/scratch/astra-b5/hb189-gate/gate-copy-v145'
RECEIPT = ROOT / '.local/state/gates/knee-lb-r6w958q-v1.json'
PARENT_RECEIPT = '8781c905cc86a7ad35bdaf49720796631e352ce8b58e5e0532bbe6ccbadbd7a6'
PARENT_KERNEL = 'c76b1ddb4d83110ee6556f5f5282ad843af9f6d0cbc932dc6630e3868d4e084f'
PARENT_NOTEBOOK = 'fcb2df01b47d5cc9b83c8a11e35ceca10d19ba5884e2b8fd40aee13867f3e8c9'
PARENT_METADATA = 'f794876c887318a85887458dee54c0dffa1ca89edf5823db7c1242f17c15dbf9'
PARENT_ARCHIVE = '92cec8b3f10a768a73ad221d5aa6ce2fefda41a591d0b39dcba6cbf10a5816bd'
WRAPPER = 'exp169_release_c85r7codec.py'
OUT = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7' / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/claude-opus-b2/hb190'
BACKUP = ROOT / '.local/backups/hb190-build/hb190-parent'
SPEC = SCRATCH / 'SPEC.md'
BLOCK = ROOT / 'tools/c85r7_r6w958rdq_v1140_reader_block.txt'
DIFF = 'full-diff-vs-v1.1.39-r6w958q.txt'
READER_SLUG = 'goodpjw2008/rsna-knee-2-5d-convnext-reader'
READER_STORE = ROOT / 'storage/external/goodpjw2008-rsna-knee-2-5d-convnext-reader-20261017'
READER_MANIFEST_SHA = 'c629686609b07ae12d38844e1d7d0e2fd5416f4bf09a834424e2e78b311f9d1e'
CHECKPOINTS = (('cnxt_v0_fold0.pt', '0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80'),
               ('cnxt_v0_fold1.pt', '61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb'),
               ('cnxt_v0_fold2.pt', 'f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6'))
SOURCE = h189.SOURCE
SOURCE_SHA = '2bb7b6aeb3ae6b38b1681e2d811775bf28f00ea71eeff85921cb00b2779f5b2d'
LABELS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
A = (0.025, 0.02, 0.02, 0.008, 0.025, 0.008, 0.008, 0.004, 0.003, 0.035, 0.03, 0.008)
GOLD_DICOM = ROOT / '.local/scratch/claude-hb128/dicom'
GOLD_SERIES_SHA = '5f4d2cae2589023b44f9c18a2c199816dd6afdb0e32a4ff0d9daa88677631387'
EXP131 = ROOT / '.local/exp131-gold58-cnx/cnx-output'
EXP131_CSV_SHA = 'b8f8e1dc787fb3aa3bd46ac8e821c9eddeb8001207180570135a771c167c1038'
EXP131_TIMING_SHA = '4c06e15552ae7a650c8972b19ef72c523271c2ac11a13dd926ba5ba749459520'
H189_SCRATCH = ROOT / '.local/scratch/hb189-build/hb189'
H189_REPLAY_SHA = '5f176aa8ff2fe2f6909eef11dbc84517f41be4fd1d7ed1bc901dc5f03a4fc7fb'
H189_BASE = {2: ('2ced2b241680287dad5c02cfecff2043d8016f181dc0cc3c2008660db8c61eab',
                 'c23e016b705c0a9b8ef507a02f921b7aa7e72953c97776eb11d0725120db3c33'),
             58: ('56957b03951921634d131873b157d024abc0e649eb78fddb6ef66567935bb777',
                  '97b35fd7f307ae53fd99e8c71bf2cc69fb7dbb46278a54bbda339d1d5019c805')}
FABLE = h189.FABLE
THREADS = 6  # CPU measurement threads: leaves headroom for the running HB186 GPU job's loaders
H174_HIDDEN_UPPER_HOURS = 7.12  # 57251520 submitted 11:17:50, first seen COMPLETE 18:25 KST (D-B-207)
HIDDEN_STUDIES = 1300  # exp169_release_v5q.COHORT_STUDIES
LIMIT_HOURS = 9.0
FORMULA = ('Y_t=r((1-a_t)*r(F)_t+a_t*r(Q)_t); F=H-B-189 C85 submission.csv '
           '(0.85*r(P)+0.15*r(O); rr=0.65*r(R384)+0.35*r(R224); P_t=w_t*rr_t+(1-w_t)*r(R6flip)_t; '
           "w=.5 except LatOA .2, PFOA .6, Syn .8, Baker's .35; O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46); "
           'Q=mean over 3 folds of sigmoid(2.5D ConvNeXt MIL reader, k=16, counterpart fill); '
           'r=average-tie percentile rank; reader-failed study: r(Q)_i:=r(F)_i; whole failure keeps F bytes')
VERSION_EDITS = [
    ('VERSION = "1.1.39-r6w958q"', 'VERSION = "' + VERSION + '"'),
    ('OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.39-r6w958q"',
     'OUT = ROOT / "artifacts/experiments/EXP169_RELEASE_C85R7/v' + VERSION + '"'),
    ('KERNEL_ID = "taeyangg4/knee-lb-r6w958q-v1"', 'KERNEL_ID = "' + SLUG + '"'),
    ('RELEASE = "C85R7+codecs R224 R6 FLIP_ALL per-target W958"',
     'RELEASE = "C85R7+codecs R224 R6 FLIP_ALL per-target W958 + 2.5D reader"')]
PACKAGING = b'# ---------------------------------------------------------------------------------------- packaging\n'
CALL_BEFORE = b'          flush=True)\n    return receipt\n'
CALL_AFTER = b'          flush=True)\n    reader_phase(runtime, output, codec, env, tele)\n    return receipt\n'
CLI_BEFORE = b'    cmd = sub.add_parser("run")\n'
CLI_AFTER = (b'    cmd = sub.add_parser("reader-worker")\n'
             b'    cmd.add_argument("--runtime", type=Path, required=True)\n'
             b'    cmd.add_argument("--output", type=Path, required=True)\n'
             b'    cmd = sub.add_parser("run")\n')
DISPATCH_BEFORE = b'    if args.command == "public-worker": result = run_public(args.runtime, args.output)\n'
DISPATCH_AFTER = DISPATCH_BEFORE + b'    elif args.command == "reader-worker": result = run_reader(args.runtime, args.output)\n'
need, digest = old.need, old.digest


def sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parent():
    """Admitted H-B-189 gate copy, re-derived from the H-B-189 builder's own payload (lineage)."""
    nb, files, blob, assignments = reuse.admitted(BASE, RECEIPT, PARENT_RECEIPT, PARENT_KERNEL,
                                                  PARENT_NOTEBOOK, PARENT_METADATA, PARENT_ARCHIVE)
    need(files == h189.payload()[2], 'H-B-189 admitted members differ from the H-B-189 builder payload')
    need(old.read(RECEIPT)['kernel_id'] == PARENT_SLUG, 'parent admission slug')
    return nb, files, blob, assignments


def block():
    data = BLOCK.read_bytes()
    compile(data, str(BLOCK), 'exec')
    return data


def patched(original):
    files = dict(original)
    s = files[WRAPPER]
    s = old.replace(s, PACKAGING, block() + PACKAGING)
    s = old.replace(s, CALL_BEFORE, CALL_AFTER)
    s = old.replace(s, CLI_BEFORE, CLI_AFTER)
    s = old.replace(s, DISPATCH_BEFORE, DISPATCH_AFTER)
    for before, after in VERSION_EDITS:
        s = old.replace(s, before, after)
    files[WRAPPER] = s
    return files


def payload():
    nb, original, old_blob, assignments = parent()
    files = patched(original)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as source, zipfile.ZipFile(buf, 'w') as target:
        for info in source.infolist():
            target.writestr(info, files[info.filename])
    blob = buf.getvalue()
    code = ''.join(nb['cells'][-1]['source']).encode()
    for before, after in [
            ('BLOB = ' + repr(assignments['BLOB']), 'BLOB = ' + repr(base64.b64encode(blob).decode())),
            ('PINS = ' + repr(assignments['PINS']), 'PINS = ' + repr({k: digest(v) for k, v in files.items()})),
            (repr(digest(old_blob)), repr(digest(blob)))]:
        code = old.replace(code, before, after)
    nb['cells'][-1]['source'] = code.decode().splitlines(True)
    meta = old.read(BASE / 'kernel/kernel-metadata.json')
    need(READER_SLUG not in meta['dataset_sources'] and meta['dataset_sources'] == reuse.PRIVATE_DATASETS,
         'parent dataset list drift')
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    meta['dataset_sources'] = [*meta['dataset_sources'], READER_SLUG]
    return nb, meta, files, blob


def member_diff(name, before, after):
    return ''.join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True),
                                        fromfile='parent/' + name, tofile='stage/' + name))


def artifacts():
    nb, meta, files, blob = payload()
    pnb, original, _, _ = parent()
    manifest = dict(version=VERSION, added=[], removed=[], changed=[], unchanged=[])
    full = ('H-B-190 v1.1.40-r6w958rdq vs admitted H-B-189 v1.1.39-r6w958q (gate copy %s)\n'
            'ALL 31 EMBEDDED FILES: 30 IDENTICAL, 1 REGISTERED CHANGE; NO ADDED/REMOVED MEMBERS\n'
            % BASE.relative_to(ROOT).as_posix())
    for k in sorted(files):
        a, b = digest(original[k]), digest(files[k])
        manifest['changed' if a != b else 'unchanged'].append(dict(name=k, before_sha256=a, after_sha256=b))
        full += f'{k}: {a} -> {b} ' + ('REGISTERED_CHANGE' if a != b else 'IDENTICAL') + '\n'
        full += member_diff(k, original[k], files[k])
    for name, before, after in [('release.ipynb', pnb, nb),
                                ('kernel-metadata.json', old.read(BASE / 'kernel/kernel-metadata.json'), meta)]:
        full += ''.join(difflib.unified_diff(old.encoded(before).decode().splitlines(True),
                                             old.encoded(after).decode().splitlines(True),
                                             fromfile='parent/' + name, tofile='stage/' + name))
    return nb, meta, files, blob, manifest, full.encode()


def scope(nb, meta):
    import ast
    expected, em, files, blob = payload()
    need(nb == expected and meta == em, 'forbidden notebook/metadata diff')
    actual, ab, _ = old.unpack(fast.restored(nb))
    need(actual == files and ab == blob, 'forbidden embedded diff')
    original = parent()[1]
    need(set(files) == set(original) and {k for k in files if files[k] != original[k]} == {WRAPPER}, 'member scope')
    # Only the four version lines are removed; everything else in the wrapper is an insertion.
    a, b = original[WRAPPER].decode().splitlines(True), files[WRAPPER].decode().splitlines(True)
    removed = [x for op, i, j, _, _ in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
               if op in ('replace', 'delete') for x in a[i:j]]
    need(sorted(x.rstrip('\n') for x in removed) == sorted(e[0] for e in VERSION_EDITS), 'wrapper removed lines')
    for name, data in files.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    return files, blob


def a_from_source():
    """Text parsing of the public notebook (never executed): OWN_W_PER_TARGET literal in label order."""
    import ast
    raw = SOURCE.read_bytes()
    need(digest(raw) == SOURCE_SHA, 'public source drift')
    text = raw.decode('utf-8')
    lines = [x for x in text.splitlines() if x.strip().startswith('OWN_W_PER_TARGET = {')]
    need(len(lines) == 1, 'unique OWN_W_PER_TARGET line')
    literal = lines[0].split('=', 1)[1].split('#', 1)[0].strip()
    mapping = ast.literal_eval(literal)
    need(set(mapping) == set(LABELS), 'source weight labels')
    return tuple(mapping[label] for label in LABELS)


def registration():
    parent()
    need(a_from_source() == A, 'a_t differs from the public source literal')
    return dict(item='H-B-190', version=VERSION, kernel_id=SLUG,
        purpose='Full 0.958 transfer with O kept: admitted H-B-189 (R6 line + source per-target w) plus the source-fixed '
                '2.5D ConvNeXt MIL reader overlay with the 0.958 notebook final per-target weights a_t',
        parent=BASE.relative_to(ROOT).as_posix(), parent_kernel_sha256=PARENT_KERNEL,
        parent_receipt_sha256=PARENT_RECEIPT, parent_slug=PARENT_SLUG,
        formula=FORMULA, a=list(A), labels=list(LABELS), rank_convention='percentile-rank blend then final rerank (source OWN_MODE end)',
        spec_sha256=digest(SPEC.read_bytes()), spec_path=SPEC.relative_to(ROOT).as_posix(),
        source_path=SOURCE.relative_to(ROOT).as_posix(), source_sha256=SOURCE_SHA,
        source_lines=['1870-2305 own_src preprocess/knee/infer', '2307-2415 final writer'],
        reader_dataset=dict(slug=READER_SLUG, licence='apache-2.0', manifest_sha256=READER_MANIFEST_SHA,
                            checkpoints=[dict(name=n, sha256=h) for n, h in CHECKPOINTS],
                            version_rule='attach the version whose checkpoint bytes equal these SHA256 pins; '
                                         'runtime pins make any other version a whole fallback'),
        reader_semantics='notebook own_src: k=16 windows, fat-sat counterpart fill, 3 folds mean sigmoid, fp16 autocast, '
                         'bs 4, workers 4, test.csv order, no mirror TTA; weights_only checkpoint load',
        source_differences=['per-study fallback rQ_i:=r(F)_i instead of all-or-nothing; zero-loaded-series study = failed',
                            'no pylibjpeg pip install; existing hb126 gdcm codec env (branch READER)',
                            'no 8.2 h start-skip; 900 s journal idle watchdog with per-study fallback',
                            'unknown plane / non-0/1 fat-sat rows skipped instead of aborting all studies',
                            'series used by two slots decoded once (identical volume)',
                            'DataLoader iterator created before CUDA model load',
                            'checkpoint SHA256 pins (mismatch = whole fallback)'],
        failure_rule='whole fallback keeps H-B-189 submission.csv bytes: child error with no valid rows, <2 successes, '
                     'a constant reader column, or any overlay error; otherwise per-study fallback',
        keep_rule=dict(submit='only after H-B-189 reads COMPLETE', retain="iff displayed >= H-B-189's displayed score",
                       expected_public=.958, source='user supplied; no polling'),
        timing=dict(limit_hours=LIMIT_HOURS, parent_basis_hours=H174_HIDDEN_UPPER_HOURS,
                    parent_basis='H-B-174 57251520 hidden COMPLETE first seen 7.12 h after submission (upper bound incl. queue); '
                                 'H-B-189 is weights-only on H-B-174',
                    reader_method='EXP131 measured Kaggle T4 2.0456 s/study (author infer.py k=12, incl. decode/setup) x '
                                  'measured local CPU serving/dataset ratio on the same 58 studies x 1300 studies; '
                                  'conservative: x window-slot ratio',
                    exp131_timing_sha256=EXP131_TIMING_SHA, hidden_studies=HIDDEN_STUDIES,
                    measured_record='runtime-projection.json (stage), never edited into this registration'),
        gold58=dict(exposure='EXPOSED', use='descriptive only, not a gate', weak_four=[3, 5, 6, 8],
                    replay=FABLE.relative_to(ROOT).as_posix(), replay_sha256=digest(FABLE.read_bytes())),
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        runtime_qualification='CPU only: real-DICOM reader parity and cached overlay replay; no CUDA/Linux/Kaggle qualification. '
                              'The visible run takes the FAST path and never executes the reader.',
        review='one independent Claude sub-review (Codex unavailable)',
        gate_binding='B pins slug-generic strict_variant_gate_r224_v145 with authentic dispatcher builder/reviewer receipts',
        dataset_sources_delta_version='1.0.0', dataset_sources_delta=dict(added=[READER_SLUG], removed=[]))


CHANGELOG = b'''# v1.1.40-r6w958rdq
H-B-190: admitted H-B-189 + the 0.958 notebook's final 2.5D ConvNeXt MIL reader overlay,
Y_t = r((1-a_t) r(F)_t + a_t r(Q)_t) with the source's fixed 12 a_t; O and all C85 members unchanged.
Only exp169_release_c85r7codec.py changes (reader section, reader-worker command, one post-collect call,
version/id/release); 30/31 embedded files identical. Metadata: id/title + dataset_sources append
goodpjw2008/rsna-knee-2-5d-convnext-reader (registered delta). Per-study reader fallback keeps H-B-189 ranks;
whole failure keeps H-B-189 bytes. EXPOSED. Builder claude-opus-b2 (tools/c85r7_r6w958rdq_v1140.py 1.0.0).
Parent backup: .local/backups/hb190-build/hb190-parent. No original/frozen/scored files edited.
'''


def register():
    data = old.encoded(registration())
    path = OUT / 'registration.json'
    if path.exists():
        need(path.read_bytes() == data, 'immutable registration drift')
    else:
        need(not (OUT / 'kernel').exists(), 'registration must precede kernel')
        old.write(path, data)
    old.write(OUT / 'registration.sha256', (digest(data) + '\n').encode())
    return dict(status='REGISTERED_BEFORE_EVALUATION', sha256=digest(data), exposure='EXPOSED')


def prepare():
    need((OUT / 'registration.json').read_bytes() == old.encoded(registration()), 'register first')
    nb, meta, files, blob, manifest, full = artifacts()
    scope(nb, meta)
    for name in ('kernel/release.ipynb', 'kernel/kernel-metadata.json'):
        old.write(BACKUP / name, (BASE / name).read_bytes())
    old.write(BACKUP / 'parent-admission-receipt.json', RECEIPT.read_bytes())
    for name, data in [('kernel/release.ipynb', old.encoded(nb)), ('kernel/kernel-metadata.json', old.encoded(meta)),
                       ('CHANGELOG.md', CHANGELOG), ('new-members.json', old.encoded(manifest)), (DIFF, full),
                       ('SPEC.md', SPEC.read_bytes())]:
        old.write(OUT / name, data)
    for label, contents in [('candidate', files), ('parent', parent()[1])]:
        for name, data in contents.items():
            old.write(SCRATCH / (label + '-runtime') / name, data)
    rec = dict(status='PREPARED_CPU_ONLY', version=VERSION, builder_sha256=digest(Path(__file__).read_bytes()),
               block_sha256=digest(block()),
               kernel_files=gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files'], archive_sha256=digest(blob),
               embedded={k: digest(v) for k, v in files.items()}, unchanged=30, changed=[WRAPPER], added=[], removed=[],
               parent_kernel_sha256=PARENT_KERNEL, parent_receipt_sha256=PARENT_RECEIPT, qualified_for_submission=False)
    old.write(OUT / 'build-receipt.json', old.encoded(rec))
    return validate()


def validate():
    nb, meta, files, blob, manifest, full = artifacts()
    for name, data in [('registration.json', old.encoded(registration())), ('kernel/release.ipynb', old.encoded(nb)),
                       ('kernel/kernel-metadata.json', old.encoded(meta)), ('CHANGELOG.md', CHANGELOG),
                       ('new-members.json', old.encoded(manifest)), (DIFF, full), ('SPEC.md', SPEC.read_bytes())]:
        need((OUT / name).read_bytes() == data, 'stage drift: ' + name)
    scope(old.read(OUT / 'kernel/release.ipynb'), old.read(OUT / 'kernel/kernel-metadata.json'))
    rec = old.read(OUT / 'build-receipt.json')
    need(rec['builder_sha256'] == digest(Path(__file__).read_bytes()) and rec['block_sha256'] == digest(block()),
         'builder drift')
    need(rec['embedded'] == {k: digest(v) for k, v in files.items()} and rec['archive_sha256'] == digest(blob), 'archive drift')
    kfiles = gate.kernel(OUT, gate.Inputs(ROOT))[2]['kernel_files']
    need(rec['kernel_files'] == kfiles, 'kernel drift')
    need(rec['qualified_for_submission'] is False and rec['unchanged'] == 30, 'receipt scope')
    for name, data in files.items():
        need((SCRATCH / 'candidate-runtime' / name).read_bytes() == data, 'candidate runtime drift: ' + name)
    from tools import strict_variant_gate_r224_v145 as generic
    pins = gate.Inputs(ROOT)
    delta = generic.metadata_delta(gate.kernel(BASE, pins), gate.kernel(OUT, pins), SLUG, registration())
    need(delta == dict(added=[READER_SLUG], removed=[]), 'dataset delta')
    return dict(status='PASS_CPU_VALIDATE', kernel_sha256=gate.sha(gate.encoded(kfiles)), identical=30, changed=1,
                dataset_sources_delta=delta)


# ----------------------------------------------------------------------------------------- real-data runs
READER_SCRIPT = r'''
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
runtime, spec_path = map(Path, sys.argv[1:3])
spec = json.loads(spec_path.read_text(encoding='utf-8'))
sys.path.insert(0, str(runtime))
import torch
torch.set_num_threads(int(spec['threads']))
import pandas as pd
import exp169_release_c85r7codec as c
assert Path(c.__file__).resolve().parent == runtime.resolve()
assert all(not name.startswith('tools.') for name in sys.modules)
se = pd.read_csv(spec['series_csv'])
rows = list(zip(se.StudyInstanceUID.astype(str), se.SeriesInstanceUID.astype(str), se.Fat_Suppression, se.Anatomical_Plane))
result = {}
if spec.get('decode_only'):
    wanted = set(spec['studies'])
    sub = [r for r in rows if r[0] in wanted]
    n_files = {sr: (min(len(os.listdir(f"{spec['root']}/{st}/{sr}")), c.READER_MAX_SLICES)
                    if os.path.isdir(f"{spec['root']}/{st}/{sr}") else 0) for st, sr, _, _ in sub}
    table, _ = c.reader_table(sub, n_files, spec['counterpart'])
    data = c.ReaderStudies(spec['studies'], table, spec['root'], spec['k'])
    t0 = time.perf_counter()
    slots = [int(data[i]['mask'].sum()) for i in range(len(data))]
    result = dict(decode_wall=time.perf_counter() - t0, slots=slots)
else:
    t0 = time.perf_counter()
    summary = c.reader_predict(spec['studies'], rows, spec['root'], spec['checkpoints'], spec['journal'], device='cpu',
                               k=spec['k'], batch=spec['batch'], workers=spec['workers'], counterpart=spec['counterpart'])
    result = dict(summary=summary, wall=time.perf_counter() - t0)
assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
print(json.dumps(dict(result, threads=torch.get_num_threads(), version=c.VERSION, embedded_imports=True)))
'''

OVERLAY_SCRIPT = r'''
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
runtime, work = map(Path, sys.argv[1:3])
sys.path.insert(0, str(runtime))
import exp169_release_c85r7codec as c
assert Path(c.__file__).resolve().parent == runtime.resolve()
assert all(not name.startswith('tools.') for name in sys.modules)
result = c.reader_apply(work / 'submission.csv', work / 'reader')
import torch
assert not torch.cuda.is_initialized()
print(json.dumps(result))
'''


def gold_ids():
    data = old.read(GOLD_DICOM / 'manifest.json')
    need(sha_file(GOLD_DICOM / 'train_series.csv') == GOLD_SERIES_SHA, 'Gold58 series CSV drift')
    ids = list(data['ids'])
    need(len(ids) == 58 == len(set(ids)) and ids == sorted(ids), 'Gold58 roster')
    return ids


def checkpoints():
    paths = []
    for name, h in CHECKPOINTS:
        path = READER_STORE / 'extracted' / name
        need(sha_file(path) == h, 'checkpoint pin: ' + name)
        paths.append(str(path))
    manifest = READER_STORE / 'MANIFEST.json'
    need(sha_file(manifest) == READER_MANIFEST_SHA, 'reader MANIFEST drift')
    entries = {e['path']: e['sha256'] for e in old.read(manifest)['files']}
    need(all(entries['extracted/' + n] == h for n, h in CHECKPOINTS), 'MANIFEST checkpoint hashes')
    return paths


def run_isolated(script, args, label):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
    env.pop('PYTHONPATH', None)
    run = subprocess.run([sys.executable, '-B', '-I', '-c', script, *map(str, args)], cwd=SCRATCH, env=env,
                         capture_output=True, text=True)
    (SCRATCH / (label + '.log')).write_text(run.stdout[-20000:] + '\n--- stderr ---\n' + run.stderr[-20000:], encoding='utf-8')
    need(run.returncode == 0, label + ' failed: ' + run.stderr[-4000:])
    return json.loads(run.stdout.strip().splitlines()[-1])


def reader_run(label, studies, *, k, counterpart, decode_only=False):
    """Embedded candidate reader on real Gold DICOMs (CPU, workers 0, bs 4). Overwrites its fixed folder."""
    folder = SCRATCH / 'reader-runs' / label
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    spec = dict(studies=studies, series_csv=str(GOLD_DICOM / 'train_series.csv'), root=str(GOLD_DICOM / 'train_series'),
                checkpoints=checkpoints(), journal=str(folder / 'journal.jsonl'), k=k, batch=4, workers=0,
                counterpart=counterpart, threads=THREADS, decode_only=decode_only)
    (folder / 'spec.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    result = run_isolated(READER_SCRIPT, [SCRATCH / 'candidate-runtime', folder / 'spec.json'], 'reader-runs/' + label + '/run')
    (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def journal_rows(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    return [r for r in rows if 'uid' in r]


def reader_smoke():
    """Real 2- and 58-study CPU reader runs (serving + dataset mode) and the decode-only split."""
    validate()
    ids = gold_ids()
    out = {}
    out['serving2'] = reader_run('serving2', ids[:2], k=16, counterpart=True)
    out['serving58'] = reader_run('serving58', ids, k=16, counterpart=True)
    out['dataset58'] = reader_run('dataset58', ids, k=12, counterpart=False)
    out['decode58'] = reader_run('decode58', ids, k=16, counterpart=True, decode_only=True)
    for key in ('serving2', 'serving58', 'dataset58'):
        rows = journal_rows(SCRATCH / 'reader-runs' / key / 'journal.jsonl')
        need([r['uid'] for r in rows] == (ids[:2] if key == 'serving2' else ids), key + ' journal order')
        out[key]['ok'] = sum(r['ok'] for r in rows)
        out[key]['slots'] = [r['slots'] for r in rows]
    # Fixed scratch evidence: reruns overwrite (lane rule), stage evidence is copied by finalize.
    (SCRATCH / 'reader-smoke.json').write_bytes(
        old.encoded(dict(status='PASS', kernel_sha256=validate()['kernel_sha256'], **out)))
    return dict(status='PASS', **{k: dict(wall=v.get('wall', v.get('decode_wall')), ok=v.get('ok')) for k, v in out.items()})


def oracle_overlay(base, q, ok):
    """Independent pandas oracle of the registered overlay (column loop, NaN-free imputation)."""
    import numpy as np
    import pandas as pd
    result = np.empty_like(base)
    for j, a in enumerate(A):
        rf = pd.Series(base[:, j]).rank(method='average', pct=True).to_numpy()
        rq = rf.copy()
        rq[ok] = pd.Series(q[ok, j]).rank(method='average', pct=True).to_numpy()
        result[:, j] = pd.Series((1 - a) * rf + a * rq).rank(method='average', pct=True).to_numpy()
    return result


def overlay_case(n, label, journal_lines):
    """Run the embedded reader_apply on a copy of H-B-189's real n-study CSV; returns (result, CSV bytes)."""
    csv_sha, npy_sha = H189_BASE[n]
    base_csv = H189_SCRATCH / f'real{n}-healthy-candidate.csv'
    need(sha_file(base_csv) == csv_sha, 'H-B-189 base CSV pin')
    work = SCRATCH / 'overlay' / f'{n}-{label}'
    if work.exists():
        shutil.rmtree(work)
    (work / 'reader').mkdir(parents=True)
    (work / 'submission.csv').write_bytes(base_csv.read_bytes())
    (work / 'reader/hb189-submission.csv').write_bytes(base_csv.read_bytes())
    (work / 'reader/journal.jsonl').write_bytes(b''.join(journal_lines))
    result = run_isolated(OVERLAY_SCRIPT, [SCRATCH / 'candidate-runtime', work], f'overlay/{n}-{label}/run')
    return result, (work / 'submission.csv').read_bytes(), base_csv.read_bytes()


def parse_csv(data):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(io.BytesIO(data), dtype={'StudyInstanceUID': str})
    need(list(frame.columns) == ['StudyInstanceUID', *LABELS], 'CSV schema')
    return frame.StudyInstanceUID.tolist(), frame[list(LABELS)].to_numpy(np.float64)


def cpu_replay():
    import numpy as np
    import pandas as pd
    import importlib.util
    import torch
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only required')
    valid = validate()
    smoke = old.read(SCRATCH / 'reader-smoke.json')
    need(smoke['status'] == 'PASS' and smoke['kernel_sha256'] == valid['kernel_sha256'], 'reader smoke missing/stale')
    need(sha_file(H189_SCRATCH / 'replay.json') == H189_REPLAY_SHA, 'H-B-189 replay pin')
    ids = gold_ids()
    report = dict(status='PASS', kernel_sha256=valid['kernel_sha256'], exposure='EXPOSED', clean_oof=False, gold58_gate=False)
    # Parity A: dataset mode vs EXP131 (author infer.py k=12 executed on Kaggle T4 fp16, 2026-10-08).
    need(sha_file(EXP131 / 'gold58_cnx.csv') == EXP131_CSV_SHA, 'EXP131 CSV pin')
    eids, exp131 = parse_csv((EXP131 / 'gold58_cnx.csv').read_bytes())
    need(eids == ids, 'EXP131 roster/order')
    rows = journal_rows(SCRATCH / 'reader-runs/dataset58/journal.jsonl')
    need(all(r['ok'] for r in rows) and [r['uid'] for r in rows] == ids, 'dataset-mode reader must succeed on all 58')
    dataset = np.asarray([r['p'] for r in rows], float)
    rho = [float(pd.Series(dataset[:, j]).corr(pd.Series(exp131[:, j]), method='spearman')) for j in range(12)]
    diff = np.abs(dataset - exp131)
    report['parity_exp131'] = dict(max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
                                   min_spearman=min(rho), spearman=rho, rule='min Spearman >= .999 and max |diff| <= .02')
    need(min(rho) >= .999 and float(diff.max()) <= .02, 'reader parity vs EXP131 author output failed')
    # Overlay replay through the isolated embedded wrapper on real H-B-189 bases.
    cases = []
    for n, key in ((2, 'serving2'), (58, 'serving58')):
        lines = (SCRATCH / 'reader-runs' / key / 'journal.jsonl').read_bytes().splitlines(True)
        study_lines = [x for x in lines if b'"uid"' in x]
        need(len(study_lines) == n, 'journal study lines')
        bids, base = parse_csv((H189_SCRATCH / f'real{n}-healthy-candidate.csv').read_bytes())
        need(bids == ids[:n], 'H-B-189 base roster')
        q = np.asarray([json.loads(x)['p'] for x in study_lines], float)
        scenarios = {'healthy': (lines, np.ones(n, bool), 'APPLIED'),
                     'all_failed': ([x for x in lines if b'"uid"' not in x], np.zeros(n, bool), 'FALLBACK_ALL'),
                     'truncated_tail': (lines[:-1] + [lines[-1][:-5]] if n == 2 else lines, np.ones(n, bool), 'APPLIED')}
        if n == 58:
            bad = [json.loads(x) for x in study_lines]
            bad[0].update(ok=False, p=None, error='injected'); bad[5]['p'][3] = float('nan')
            failed_lines = [x for x in lines if b'"uid"' not in x] + [(json.dumps(r) + '\n').encode() for r in bad]
            mask = np.ones(n, bool); mask[[0, 5]] = False
            scenarios['partial_failed'] = (failed_lines, mask, 'APPLIED')
            const = [json.loads(x) for x in study_lines]
            for r in const: r['p'][7] = 0.5
            scenarios['constant_column'] = ([(json.dumps(r) + '\n').encode() for r in const], np.ones(n, bool), 'FALLBACK_ALL')
            dup = list(study_lines) + [study_lines[1]]
            mask_dup = np.ones(n, bool); mask_dup[1] = False
            scenarios['duplicate_row'] = (dup, mask_dup, 'APPLIED')
        for label, (journal, mask, status) in scenarios.items():
            result, final, base_bytes = overlay_case(n, label, journal)
            need(result['status'] == status, f'{n}-{label} status {result["status"]}')
            if status == 'FALLBACK_ALL':
                need(final == base_bytes, f'{n}-{label} whole fallback must keep H-B-189 bytes')
                expected_bytes = base_bytes
            else:
                expected = oracle_overlay(base, q, mask)
                expected_bytes = pd.DataFrame(expected.astype(np.float32), columns=list(LABELS)).assign(
                    StudyInstanceUID=bids)[['StudyInstanceUID', *LABELS]].to_csv(index=False, lineterminator='\n').encode()
                need(final == expected_bytes, f'{n}-{label} overlay CSV differs from independent pandas oracle')
                need(result['fallback_studies'] == int((~mask).sum()) and result['reader_ok'] == int(mask.sum()),
                     f'{n}-{label} fallback counts')
            cases.append(dict(studies=n, case=label, status=result['status'], fallback_studies=result.get('fallback_studies'),
                              reason=result.get('reason'), oracle_csv_bytes_exact=True,
                              final_sha256=digest(final), base_sha256=digest(base_bytes)))
    report['overlay_cases'] = cases
    # Gold58 descriptive (EXPOSED, not a gate): H-B-189 vs H-B-190 on the same 58, fable helpers.
    spec = importlib.util.spec_from_file_location('hb190_fable', FABLE)
    fable = importlib.util.module_from_spec(spec); spec.loader.exec_module(fable)
    label_ids, labels = fable.d.gold.gold(); labels = np.asarray(labels, float)
    need(list(label_ids) == ids and labels.shape == (58, 12), 'Gold58 labels/roster alignment')
    parent_values = np.load(H189_SCRATCH / 'real58-healthy-candidate.npy', allow_pickle=False)
    need(digest(parent_values.tobytes()) == H189_BASE[58][1], 'H-B-189 Gold58 output pin')
    _, child_csv = parse_csv((SCRATCH / 'overlay/58-healthy/submission.csv').read_bytes())
    serving = np.asarray([r['p'] for r in journal_rows(SCRATCH / 'reader-runs/serving58/journal.jsonl')], float)
    def metrics(values):
        aucs = np.asarray([fable.auc(labels[:, j], values[:, j]) for j in range(12)])
        return dict(macro=float(aucs.mean()), weak_four=float(aucs[fable.WEAK].mean()), per_target=[float(x) for x in aucs])
    m = {k: metrics(v) for k, v in dict(parent=parent_values, child=child_csv, reader_serving=serving,
                                        reader_dataset=dataset, exp131=exp131).items()}
    m['delta'] = {k: m['child'][k] - m['parent'][k] for k in ('macro', 'weak_four')}
    m['delta']['per_target'] = [c - p for c, p in zip(m['child']['per_target'], m['parent']['per_target'])]
    report['gold58'] = m
    report['limitation'] = ('CPU fp32 reader on real Gold DICOMs + cached H-B-189 bases through the isolated embedded '
                            'wrapper; no CUDA/fp16/Linux/DataLoader-worker qualification. Labels read internally only.')
    need(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
    (SCRATCH / 'replay.json').write_bytes(old.encoded(report))
    return dict(status='PASS', parity=report['parity_exp131'], gold58_delta=m['delta'], cases=len(cases))


def runtime_projection():
    """Measured local CPU numbers -> projected Kaggle T4 hidden hours (written to the stage)."""
    smoke = old.read(SCRATCH / 'reader-smoke.json')
    t = old.read(EXP131 / 'gold58_cnx_timing.json')
    need(sha_file(EXP131 / 'gold58_cnx_timing.json') == EXP131_TIMING_SHA, 'EXP131 timing pin')
    t4 = float(t['seconds_per_study_total'])
    s58, d58, dec = smoke['serving58'], smoke['dataset58'], smoke['decode58']
    cpu_serv, cpu_ds = s58['wall'] / 58, d58['wall'] / 58
    cpu_decode = dec['decode_wall'] / 58
    w_serv = sum(s58['slots']) * 16
    w_ds = sum(d58['slots']) * 12
    ratio, w_ratio = cpu_serv / cpu_ds, w_serv / w_ds
    reader_h = t4 * ratio * HIDDEN_STUDIES / 3600
    conservative_h = t4 * max(ratio, w_ratio) * HIDDEN_STUDIES / 3600
    result = dict(version=VERSION, threads=THREADS, studies_measured=58,
        cpu_seconds_per_study=dict(serving=cpu_serv, dataset=cpu_ds, decode_serving=cpu_decode,
                                   forward_serving=cpu_serv - cpu_decode, serving2=smoke['serving2']['wall'] / 2),
        cpu_ratio_serving_over_dataset=ratio, window_slot_ratio=w_ratio,
        slots=dict(serving=sum(s58['slots']), dataset=sum(d58['slots'])),
        exp131_t4_seconds_per_study=t4, hidden_studies=HIDDEN_STUDIES,
        projected_t4_seconds_per_study=dict(central=t4 * ratio, conservative=t4 * max(ratio, w_ratio)),
        reader_hours=dict(central=reader_h, conservative=conservative_h),
        parent_basis_hours=H174_HIDDEN_UPPER_HOURS,
        total_hours=dict(central=H174_HIDDEN_UPPER_HOURS + reader_h, conservative=H174_HIDDEN_UPPER_HOURS + conservative_h),
        limit_hours=LIMIT_HOURS,
        basis='EXP131 measured T4 rate (author code k=12, one T4, DataLoader workers 4, incl. decode) scaled by measured '
              'local CPU serving/dataset ratio (central) or max(ratio, window-slot ratio) (conservative); parent = H-B-174 '
              'hidden first-seen-complete upper bound incl. queue; reader runs after all members on a free GPU.')
    need(result['total_hours']['conservative'] < LIMIT_HOURS, 'projected hidden runtime exceeds 9 h')
    (SCRATCH / 'runtime-projection.json').write_bytes(old.encoded(result))
    return result


def finalize():
    """Copy measured evidence into the stage (new files only; kernel bytes are untouched)."""
    valid = validate()
    replay = old.read(SCRATCH / 'replay.json')
    need(replay['status'] == 'PASS' and replay['kernel_sha256'] == valid['kernel_sha256'], 'replay stale')
    proj = runtime_projection()
    for name in ('runtime-projection.json', 'replay.json', 'reader-smoke.json'):
        target = OUT / 'evidence' / name  # evidence sidecars only; never kernel/registration/gate inputs
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((SCRATCH / name).read_bytes())
    return dict(status='EVIDENCE_COPIED', total_hours=proj['total_hours'])


def admit():
    valid = validate()
    replay = old.read(SCRATCH / 'replay.json')
    need(replay['status'] == 'PASS' and replay['kernel_sha256'] == valid['kernel_sha256'], 'CPU replay missing/stale')
    proj = old.read(SCRATCH / 'runtime-projection.json')
    need(proj['total_hours']['conservative'] < LIMIT_HOURS, 'runtime projection')
    return dict(status='HOLD_B_SLUG_GENERIC_GATE_AND_INDEPENDENT_REVIEW', qualified_for_submission=False,
                kernel_sha256=valid['kernel_sha256'], gate='tools/strict_variant_gate_r224_v145.py',
                remaining=['B: v1.4.5 pin/bind with authentic builder receipt, independent gate review, evaluate',
                           'B: Linux/CUDA qualification of the new reader path (visible run is FAST and skips it)',
                           'B: confirm the attached reader dataset version has the pinned checkpoint bytes'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['register', 'prepare', 'validate', 'reader-smoke', 'cpu-replay', 'finalize', 'admit'])
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.command in ('prepare', 'admit'):
        need(args.dry_run, 'CPU dry-run required')
    result = {'register': register, 'prepare': prepare, 'validate': validate, 'reader-smoke': reader_smoke,
              'cpu-replay': cpu_replay, 'finalize': finalize, 'admit': admit}[args.command]()
    path = SCRATCH / (args.command + '-result.json')
    path.write_bytes(old.encoded(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
