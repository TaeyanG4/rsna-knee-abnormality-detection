"""Build local, unpushed EXP189 Kaggle notebook specs from pinned sources.

Changelog
- v1.2.0 (astra, 2026-10-11): versioned local kernels; authenticated public
  reference with unbounded execution; pinned Gold58 roster; CPU reference policy.
  CPU reference explicitly pins anatomy/side helpers and disables CUDA with -1.
- v1.0.0 (sol): parity / gold58 / timing specs.
- v1.1.0 (Claude opus for B, 2026-10-11): owner taeyangg4 with title == slug, the
  release candidates' T4 machine shape and pinned docker image (read from the A172
  v2.0.2 stage metadata); every port path first records the observed code tree
  (exp189_code_observed.json, input for `exp189_s6_port.py pin-code`) and runs the
  port with that explicitly observed digest; the timing helper samples
  min(24, visible) studies (works with the 3 visible test studies) and reports the
  per-study service time including the anatomy-anchor term and anchor fallbacks.
  New CPU-only `observe` spec (no GPU quota): records exp189_code_observed.json
  for `exp189_s6_port.py pin-code` before any GPU kernel runs.
"""
from __future__ import annotations

import base64
import ast
import csv
import hashlib
import inspect
import io
import json
import zipfile
from pathlib import Path

VERSION = "1.2.0"
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / ".local/idea-scout/20261011-1030/notebooks/heliosli__rsna-knee-blend-gold-dataset-restored/rsna-knee-blend-gold-dataset-restored.ipynb"
OUT = ROOT / "artifacts/experiments/EXP189_S6_PORT/v1.2.0/kernels"
PORT = ROOT / "tools/exp189_s6_port.py"
VENDOR = ROOT / "tools/exp189_s6_src"
RELEASE_META = ROOT / "artifacts/releases/EXP169_A172_v2_0_2_stage/kernel/kernel-metadata.json"
GOLD_PROTOCOLS = (
    ROOT / "artifacts/experiments/EXP174_OWN0949_JEV/protocol-v1/manifest.json",
    ROOT / "artifacts/experiments/EXP143_OWN0949/protocol-v2/manifest.json",
)
TRAIN_SERIES = ROOT / "storage/competition/train_series.csv"
OWNER = "taeyangg4"
PUBLIC_CELL_SHA256 = {
    4: "0afde5dda7ffe021aec321f8367f0da9f847fb5b0d93a98ba91267bcbefbff23",
    5: "0309e4d13898c7981c4b74a47373a7e8b011e5fa5ed15a584e35a7e21c44f594",
}


def reference_runtime_sources(sources: dict[str, str], *, cpu: bool = False) -> dict[str, str]:
    """Independent public-reference derivative: execution policy only, no port code."""
    result = dict(sources)

    def once(source, old, new):
        if source.count(old) != 1:
            raise ValueError(f"public execution-policy patch expected once: {old!r}")
        return source.replace(old, new, 1)

    worker = result['compact_infer.py']
    for old, new in (
        ('    ap.add_argument("--item-timeout", type=float, default=900.)\n', ''),
        ('    ap.add_argument("--deadline", type=float, default=0.)\n', ''),
        ('    timed_out = False\n    while pos < len(uids) and not timed_out:', '    while pos < len(uids):'),
        ('multiprocessing_context="spawn" if a.workers else None,\n                                             timeout=a.item_timeout if a.workers else 0)',
         'multiprocessing_context="spawn" if a.workers else None)'),
        ('                if a.deadline and time.time() > a.deadline:\n                    timed_out = True\n                    break\n', ''),
        ('    for u in uids[pos:]:\n        fallback[u] = "not reached before the deadline"\n', ''),
        ('"timed_out": timed_out,', '"device": "cpu",' if cpu else '"device": "cuda",'),
        ('("studies", "scored", "timed_out", "seconds", "amp")', '("studies", "scored", "device", "seconds", "amp")'),
    ):
        worker = once(worker, old, new)
    if cpu:
        worker = once(worker, 'dev = torch.device("cuda")', 'dev = torch.device("cpu")\n    torch.set_num_threads(2)')
        worker = once(worker, '            dev = "cuda" if torch.cuda.is_available() else "cpu"', '            dev = "cpu"')
        worker = once(worker, '            sdev = "cuda" if torch.cuda.is_available() else "cpu"', '            sdev = "cpu"')
        worker = once(worker, '    receipt = {"status": "PASS",',
                      '    if torch.cuda.is_initialized():\n        raise RuntimeError("CPU reference initialized CUDA")\n    receipt = {"status": "PASS",')
        worker = once(worker, '"device": "cpu",',
                      '"device": "cpu", "auxiliary_device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),')
        worker = once(worker, 'torch.cuda.get_device_capability()[0]', '0')
        if worker.count('torch.cuda.get_device_name()') != 2:
            raise ValueError('public expected two device-name diagnostics')
        worker = worker.replace('torch.cuda.get_device_name()', '"CPU"')
    controller = result['probe_dual.py']
    controller = once(controller, '    ap.add_argument("--deadline", type=float, default=0.)\n', '')
    controller = once(controller, ', "--deadline", str(a.deadline)', '')
    if cpu:
        begin = controller.index('    try:\n        names = subprocess.check_output')
        end = controller.index('\n\n    def cost', begin)
        controller = controller[:begin] + '    names = ["CPU"]\n    n_gpu = 1' + controller[end:]
        controller = once(controller, 'CUDA_VISIBLE_DEVICES=str(gi)', 'CUDA_VISIBLE_DEVICES="-1"')
    result.update({'compact_infer.py': worker, 'probe_dual.py': controller})
    for name, source in result.items():
        compile(source, name, 'exec')
    return result


def public_reference_cells(public: dict) -> tuple[dict, dict]:
    originals = {i: ''.join(public['cells'][i]['source']) for i in (4, 5)}
    for i, source in originals.items():
        if hashlib.sha256(source.encode()).hexdigest() != PUBLIC_CELL_SHA256[i]:
            raise ValueError(f'public cell {i} differs from registration')
    source4 = originals[4].replace('DEADLINE_HOURS = 8.3\n', '')
    source4 += '\n# Only execution-policy changes; original SOURCES remain authenticated above.\n'
    source4 += inspect.getsource(reference_runtime_sources)
    source4 += '''
for name, source in reference_runtime_sources(SOURCES).items():
    (SRC / name).write_text(source, encoding='utf-8', newline='\\n')
(WORK / 'exp189_reference_runtime_sha256.json').write_text(json.dumps(
    {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SRC.glob('*.py'))}, indent=2))
'''
    source5 = originals[5].replace('deadline = T0 + DEADLINE_HOURS * 3600\n', '')
    source5 = source5.replace(", '--deadline', str(deadline)", '')
    source5 = source5.replace("display(pd.read_csv(WORK / 'ours_sub.csv').head())", '')
    return cell(source4), cell(source5)


def cell(code: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": code.splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def registered_gold58() -> tuple[list[str], str]:
    """Pin the matching Gold exclusion rosters and check local series membership."""
    rosters = [json.loads(path.read_text(encoding="utf-8"))["excluded_gold_ids"]
               for path in GOLD_PROTOCOLS]
    ids = rosters[0]
    if len(ids) != 58 or len(set(ids)) != 58 or ids != sorted(ids) or rosters[1] != ids:
        raise ValueError("EXP174 and EXP143 v2 Gold58 registrations disagree")
    with TRAIN_SERIES.open(encoding="utf-8-sig", newline="") as stream:
        series_ids = {row["StudyInstanceUID"] for row in csv.DictReader(stream)}
    missing = sorted(set(ids) - series_ids)
    if missing:
        raise ValueError(f"Gold58 registration absent from local train_series.csv: {missing}")
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode("ascii")).hexdigest()
    return ids, digest


def write_spec(name: str, cells: list[dict], slug: str, *, gpu: bool = True,
               datasets: tuple[str, ...] = ("heliosli/rsna-knee-compact-v1",
                                            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"),
               competition: bool = True) -> None:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    file = f"exp189_{name}.ipynb"
    for item in cells:
        compile("".join(item["source"]), f"<exp189-{name}>", "exec")
    (folder / file).write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    release = json.loads(RELEASE_META.read_text(encoding="utf-8"))
    metadata = {
        "id": f"{OWNER}/{slug}", "title": slug,
        "code_file": file, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": False,
        **({"machine_shape": release["machine_shape"]} if gpu else {}),
        "docker_image": release["docker_image"],
        "dataset_sources": list(datasets),
        "competition_sources": ["rsna-knee-abnormality-detection"] if competition else [],
        "kernel_sources": [], "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def bootstrap() -> str:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in [PORT, *sorted(VENDOR.glob("*.py"))]:
            archive.write(file, "tools/" + ("exp189_s6_src/" if file.parent == VENDOR else "") + file.name)
    blob = memory.getvalue()
    b64 = base64.b64encode(blob).decode("ascii")
    digest = hashlib.sha256(blob).hexdigest()
    return f'''# EXP189 isolated serving source, generated locally; archive SHA256 pinned.
import base64, hashlib, io, os, sys, zipfile
from pathlib import Path
PORT_ARCHIVE_B64 = {b64!r}
PORT_ARCHIVE_SHA256 = {digest!r}
blob = base64.b64decode(PORT_ARCHIVE_B64)
assert hashlib.sha256(blob).hexdigest() == PORT_ARCHIVE_SHA256
PORT_ROOT = Path('/kaggle/working/exp189_s6_port')
PORT_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    archive.extractall(PORT_ROOT)
sys.path.insert(0, str(PORT_ROOT / 'tools'))
import exp189_s6_port as s6
s6.verify_source()
del blob, PORT_ARCHIVE_B64
'''


SETUP = '''import json, shutil, subprocess, time
from pathlib import Path
INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
CODE_SLUG = 'rsna-orthofoundation-fold0-epoch7-20261004'
COMPETITION = 'rsna-knee-abnormality-detection'
C_SLUG = 'rsna-knee-compact-v1'

def find_mount(slug, required):
    cands = [INPUT / slug, INPUT / 'competitions' / slug]
    cands += list(INPUT.glob('datasets/*/' + slug)) + list(INPUT.glob('datasets/*/' + slug + '/*'))
    cands += list(INPUT.glob('*/' + slug)) + list(INPUT.glob(slug + '/*'))
    found = []
    for p in cands:
        if (p / required).is_file() and p not in found:
            found.append(p)
    if len(found) != 1:
        raise RuntimeError('Cannot resolve %s: %s' % (slug, found))
    return found[0]

CODE_SRC = find_mount(CODE_SLUG, 'receipt.json')
BUNDLE = find_mount(C_SLUG, 'manifest.json')
DATA = find_mount(COMPETITION, 'train.csv')
# Measurement kernel: record the observed code tree first (pin-code input), then run the
# port against exactly that observed digest. Release candidates use the source pin only.
OBSERVED = s6.observe_code(CODE_SRC, WORK / 'exp189_code_observed.json', work_dir=WORK)
print('EXP189_CODE_OBSERVED', json.dumps({k: OBSERVED[k] for k in ('tree_sha256', 'files', 'wheel_pins_match',
      'public_module_pins_match', 'direct_modules_present')}), flush=True)
CODE, ANATOMY = s6.prepare_code(CODE_SRC, WORK / 'exp189_code', install_wheels=True,
                                code_tree_sha256=OBSERVED['tree_sha256'])
s6.verify_bundle(BUNDLE)
'''


def main() -> None:
    public = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    gold_ids, gold_sha256 = registered_gold58()
    public4, public5 = public_reference_cells(public)
    setup = bootstrap()
    tol = {"probability_atol": 0.000002, "probability_rtol": 0.0,
           "max_row_mismatches": 0, "max_fallback_fraction": 0.03,
           "expected_columns": ["StudyInstanceUID", "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                                "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
                                "Contusion", "Fracture"]}
    parity_dir = OUT / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    (parity_dir / "tol.json").write_text(json.dumps(tol, indent=2) + "\n", encoding="utf-8")
    parity = [cell(setup), cell("""# Both S6 implementations run sequentially on this same 2xT4 allocation.
import subprocess, time
GPU_NAMES = subprocess.check_output(['nvidia-smi', '--query-gpu=name',
                                     '--format=csv,noheader'], text=True).splitlines()
assert len(GPU_NAMES) == 2 and all('T4' in name for name in GPU_NAMES), GPU_NAMES
NB_T0 = time.time()
"""), public4, public5,
              cell('''# Run the isolated port on the same visible test studies and same T4 runtime.
# The observed code tree (exp189_code_observed.json) is the input for `pin-code`; the port
# prepares its own copy (wheels were installed by the public cell) against that digest.
import pandas as pd, numpy as np, json, shutil
OBSERVED = s6.observe_code(CODE_SRC, WORK / 'exp189_code_observed.json', work_dir=WORK)
print('EXP189_CODE_OBSERVED', json.dumps({k: OBSERVED[k] for k in ('tree_sha256', 'files', 'wheel_pins_match',
      'public_module_pins_match', 'direct_modules_present')}), flush=True)
PORT_CODE, PORT_ANATOMY = s6.prepare_code(CODE_SRC, WORK / 'exp189_port_code', install_wheels=False,
                                          code_tree_sha256=OBSERVED['tree_sha256'])
ids = pd.read_csv(DATA / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
rows = s6.infer_studies([DATA / 'test_series' / uid for uid in ids],
                        bundle_dir=M_SRC, code_dir=PORT_CODE, anatomy_weights=PORT_ANATOMY,
                        output_dir=WORK / 'exp189_port', workers=2, amp='auto',
                        max_fallback_fraction=0.03, code_tree_sha256=OBSERVED['tree_sha256'])
shutil.copyfile(WORK / 'exp189_port/predictions.csv', WORK / 'exp189_port_predictions.csv')
controller = json.loads((WORK / 'exp189_port/controller/submission_receipt.json').read_text())
TOL = ''' + repr(tol) + '''
(WORK / 'exp189_parity_tolerance.json').write_text(json.dumps(TOL, indent=2))
public = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
ported = pd.read_csv(WORK / 'exp189_port_predictions.csv', dtype={'StudyInstanceUID': str})
assert list(public.columns) == list(ported.columns) == TOL['expected_columns']
assert public['StudyInstanceUID'].tolist() == ported['StudyInstanceUID'].tolist() == ids
delta = np.abs(public.iloc[:, 1:].to_numpy(dtype=np.float64) - ported.iloc[:, 1:].to_numpy(dtype=np.float64))
limit = TOL['probability_atol'] + TOL['probability_rtol'] * np.abs(public.iloc[:, 1:].to_numpy(dtype=np.float64))
bad = int((delta > limit).sum())
receipt = {'rows': len(ids), 'columns': list(public.columns), 'gpu_names': GPU_NAMES,
           'max_abs_delta': float(delta.max()), 'bad_cells': bad, 'tolerance': TOL,
           'code_tree_sha256': OBSERVED['tree_sha256'], 'code_tree_files': OBSERVED['files'],
           'wheel_pins_match': OBSERVED['wheel_pins_match'],
           'service_components': s6.service_components(controller),
           'anatomy_anchors': s6.anatomy_fallbacks(controller), 'port_version': s6.VERSION}
(WORK / 'exp189_parity_receipt.json').write_text(json.dumps(receipt, indent=2))
assert bad <= TOL['max_row_mismatches'], receipt
print('EXP189_PARITY_PASS', receipt)
''')]
    write_spec("parity", parity, "exp189-s6-parity")

    gold = [cell(setup), cell(SETUP), cell('''# Gold58 is the EXP174 / EXP143 v2 registered exclusion roster, not a train.csv flag.
# SHA256 covers the ASCII IDs in this order, joined with LF and one terminal LF.
import csv, hashlib, json, shutil
GOLD58_IDS = ''' + repr(gold_ids) + '''
GOLD58_SHA256 = ''' + repr(gold_sha256) + '''
assert len(GOLD58_IDS) == len(set(GOLD58_IDS)) == 58
assert hashlib.sha256(('\\n'.join(GOLD58_IDS) + '\\n').encode('ascii')).hexdigest() == GOLD58_SHA256
with (DATA / 'train_series.csv').open(encoding='utf-8-sig', newline='') as stream:
    series_ids = {row['StudyInstanceUID'] for row in csv.DictReader(stream)}
missing_from_series = sorted(set(GOLD58_IDS) - series_ids)
assert not missing_from_series, 'registered Gold58 IDs absent from train_series.csv: ' + repr(missing_from_series)
dirs = [DATA / 'train_series' / uid for uid in GOLD58_IDS]
missing_dirs = [str(path) for path in dirs if not path.is_dir() or not any(path.glob('*/*.dcm'))]
assert not missing_dirs, 'Gold58 DICOM study/series files absent: ' + repr(missing_dirs)
output = WORK / 'exp189_gold58'
try:
    rows = s6.infer_studies(dirs, bundle_dir=BUNDLE, code_dir=CODE, anatomy_weights=ANATOMY,
                            output_dir=output, max_fallback_fraction=0.0,
                            code_tree_sha256=OBSERVED['tree_sha256'])
finally:
    receipt_path = output / 'controller/submission_receipt.json'
    if receipt_path.is_file():
        controller = json.loads(receipt_path.read_text(encoding='utf-8'))
        timings, fallbacks = {}, {}
        for worker in controller.get('workers', []):
            for study in worker.get('per_study', []):
                uid, windows, decode_s, render_s, gpu_s, anchor_s = study
                timings[uid] = {'decode_s': decode_s, 'render_s': render_s, 'gpu_s': gpu_s,
                                'anchor_s': anchor_s, 'windows': windows}
            fallbacks.update(worker.get('fallback') or {})
        fallbacks.update(controller.get('fallback_reasons') or {})
        report = {'gold58_ids_sha256': GOLD58_SHA256, 'rows': controller['rows'],
                  'scored': controller['scored'], 'fallback_studies': controller['fallback_studies'],
                  'per_study': [{'StudyInstanceUID': uid, **timings.get(uid, {
                      'decode_s': None, 'render_s': None, 'gpu_s': None,
                      'anchor_s': None, 'windows': None}),
                      'fallback_reason': fallbacks.get(uid)} for uid in GOLD58_IDS],
                  'anatomy_anchors': s6.anatomy_fallbacks(controller) if controller['scored'] else None}
        (WORK / 'exp189_gold58_timing.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('EXP189_GOLD58_TIMING', json.dumps(report), flush=True)
assert len(rows) == 58 and list(rows[0]) == [s6.UID, *s6.LABELS]
shutil.copyfile(output / 'predictions.csv', WORK / 'exp189_gold58_predictions.csv')
print('EXP189_GOLD58_PREDICTIONS', len(rows), GOLD58_SHA256)
''')]
    write_spec("gold58", gold, "exp189-s6-gold58")

    timing = [cell(setup), cell(SETUP.replace("'train.csv'", "'test.csv'")), cell('''# Spread the visible-test sample across the DICOM-count distribution (all studies when < 24).
import pandas as pd, json, subprocess, time
names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
assert len(names) == 2 and all('T4' in name for name in names), names
ids = pd.read_csv(DATA / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
assert len(ids) >= 1
dirs = [DATA / 'test_series' / uid for uid in ids]
costed = sorted(dirs, key=lambda p: (sum(1 for series in p.iterdir() if series.is_dir()
                                           for _ in series.iterdir()), p.name))
n_sample = min(24, len(costed))
sample = (costed if n_sample == len(costed) else
          [costed[round(i * (len(costed) - 1) / (n_sample - 1))] for i in range(n_sample)])
t0 = time.perf_counter()
rows = s6.infer_studies(sample, bundle_dir=BUNDLE, code_dir=CODE, anatomy_weights=ANATOMY,
                        output_dir=WORK / 'exp189_timing', max_fallback_fraction=0.0,
                        code_tree_sha256=OBSERVED['tree_sha256'])
wall = time.perf_counter() - t0
worker = json.loads((WORK / 'exp189_timing/controller/submission_receipt.json').read_text())
service = s6.service_components(worker)
rate = service['service_seconds_per_study'] + 0.003
report = {'gpu_names': names, 'sample_studies': len(rows), 'study_ids': [p.name for p in sample],
          'wall_seconds_including_hash_and_stage': wall, 'controller_seconds': worker['seconds'],
          'seconds_per_study': worker['seconds'] / len(rows),
          'service_per_study': service['per_study'],
          'service_seconds_per_study_incl_anchor_plus_0.003': rate,
          'startup_seconds': max(0.0, wall - len(rows) * rate),
          'extrapolated_1300_hours': worker['seconds'] / len(rows) * 1300 / 3600,
          'service_extrapolated_1300_hours': rate * 1300 / 3600,
          'anatomy_anchors': s6.anatomy_fallbacks(worker),
          'code_tree_sha256': OBSERVED['tree_sha256'], 'port_version': s6.VERSION,
          'scope': 'visible-sample extrapolation (startup-dominated with 3 studies), not full hidden-run timing'}
(WORK / 'exp189_timing.json').write_text(json.dumps(report, indent=2))
print('EXP189_TIMING', report)
''')]
    write_spec("timing", timing, "exp189-s6-timing")

    observe = [cell(setup), cell('''# CPU only: record the attached code dataset exactly as prepare_code sees it. No import of
# dataset code, no install, no GPU. Download exp189_code_observed.json, then run locally:
#   python tools/exp189_s6_port.py pin-code --observed exp189_code_observed.json --write
import json
from pathlib import Path
INPUT, WORK = Path('/kaggle/input'), Path('/kaggle/working')
slug = 'rsna-orthofoundation-fold0-epoch7-20261004'
cands = [INPUT / slug, INPUT / 'competitions' / slug]
cands += list(INPUT.glob('datasets/*/' + slug)) + list(INPUT.glob('datasets/*/' + slug + '/*'))
cands += list(INPUT.glob('*/' + slug)) + list(INPUT.glob(slug + '/*'))
found = list(dict.fromkeys(p for p in cands if (p / 'receipt.json').is_file()))
assert len(found) == 1, found
record = s6.observe_code(found[0], WORK / 'exp189_code_observed.json', work_dir=WORK)
print('EXP189_CODE_OBSERVED', json.dumps({k: record[k] for k in ('tree_sha256', 'files', 'wheel_sha256',
      'wheel_pins_match', 'public_module_pins_match', 'direct_modules_present')}), flush=True)
try:
    print('PIN_CHECK_PASS', json.dumps(s6.check_observation(record)), flush=True)
except ValueError as exc:  # the observation file is still saved for review
    print('PIN_CHECK_FAIL', str(exc), flush=True)
''')]
    write_spec("observe", observe, "exp189-s6-code-observe", gpu=False,
               datasets=("heliosli/rsna-orthofoundation-fold0-epoch7-20261004",), competition=False)
    print("Generated", OUT)


if __name__ == "__main__":
    main()
