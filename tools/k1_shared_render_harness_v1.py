"""v1.0.1: label-free preparation and isolated-arm parity/timing harness.

Changelog v1.0.1: require requested FP16 results/gates; invalidate aggregate before comparison.

CPU by default. CUDA execution belongs to B after review/resource admission.
Raw metadata/UIDs/predictions remain private scratch; stdout is aggregate only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile
import zlib

from tools import l1_rco_k2_worker_v102 as w
from tools import k1_shared_render_v1 as shared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.local/scratch/astra-b1/k1-shared-render'
ASSETS = ROOT / '.local/scratch/slot-b/k2-release/fix1/assets-v1'
ASSETS_SHA = 'a529559ad7efac3e88111b0007338ca931bde3e3a35b8e88b1d09ca3f44e68f3'
RAW = ROOT / 'storage/downloads/rsna-knee-abnormality-detection/rsna-knee-abnormality-detection.zip'
BUNDLE = ROOT / 'storage/external/heliosli-r6-fe055406-v1'
NOTEBOOK = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.46-r6w958inf3/kernel/release.ipynb'
REPAIRS = ROOT / 'storage/downloads/redownload-20261012-corrupt-zip-region/tree'


def open_raw(archive, entry):
    try:
        return archive.open(entry)
    except zipfile.BadZipFile:
        repaired = (REPAIRS / entry.filename).resolve()
        w.need(repaired.is_relative_to(REPAIRS.resolve()) and repaired.is_file(), 'unavailable local ZIP repair')
        payload = repaired.read_bytes()
        w.need(len(payload) == entry.file_size and zlib.crc32(payload) == entry.CRC, 'ZIP repair identity mismatch')
        return io.BytesIO(payload)


def extract_parent():
    from tools.c85r7_r6w958inf3rco_v1150 import archive, INF3_SHA
    w.need(w.sha(NOTEBOOK) == INF3_SHA, 'INF3 notebook hash')
    nb = json.loads(NOTEBOOK.read_bytes())
    _, pins, files = archive(''.join(nb['cells'][2]['source']))
    for name, payload in files.items():
        dest = (OUT / 'parent' / name).resolve()
        w.need(dest.is_relative_to((OUT / 'parent').resolve()), 'unsafe embedded path')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    return pins


def select_roster(counts, compressed=()):
    w.need(len(counts) >= 32 and min(counts.values()) > 0, 'need >=32 nonempty train studies')
    largest = sorted(counts, key=lambda u: (-counts[u], u))[:8]
    special = sorted(set(compressed) - set(largest))[:3]
    rest = sorted(set(counts) - set(largest) - set(special),
                  key=lambda u: hashlib.sha256(('K1-shared-v1\0' + u).encode()).hexdigest())
    return largest + special + rest[:32 - len(largest) - len(special)], largest


def prepare():
    """Existing local ZIP only. Scan one header/series, not every cohort file.

    Survey every series representative for compressed syntax. For selected
    studies the renderer later decodes every needed frame. Mixed syntax inside
    an otherwise-uncompressed series is not exhaustively surveyed here.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import pydicom
    pins = extract_parent()
    OUT.mkdir(parents=True, exist_ok=True)
    data = OUT / 'raw'
    counts, by_uid, representatives = {}, defaultdict(list), {}
    compressed, syntaxes, unreadable = set(), defaultdict(int), []
    with zipfile.ZipFile(RAW) as z:
        for entry in z.infolist():
            parts = entry.filename.split('/')
            if len(parts) == 4 and parts[0] == 'train_series' and parts[3].endswith('.dcm'):
                w.need(all(p not in ('', '.', '..') for p in parts), 'unsafe DICOM path')
                uid = parts[1]
                by_uid[uid].append(entry)
                representatives.setdefault((uid, parts[2]), entry)
        counts = {u: len(entries) for u, entries in by_uid.items()}
        # Only TransferSyntaxUID is inspected; no report or patient tags.
        for (uid, series), entry in sorted(representatives.items()):
            try:
                with open_raw(z, entry) as stream:
                    header = pydicom.dcmread(stream, stop_before_pixels=True, specific_tags=[])
            except (zipfile.BadZipFile, ValueError):
                unreadable.append(entry.filename)
                continue  # Survey limitation only: a selected missing file still fails extraction.
            syntax = header.file_meta.TransferSyntaxUID
            syntaxes[str(syntax)] += 1
            if syntax.is_compressed:
                compressed.add(uid)
        ids, largest = select_roster(counts, compressed)
        # Smallest member first: CPU smoke remains small; full runs cover all32.
        ids.sort(key=lambda u: (counts[u], u))
        data.mkdir(parents=True, exist_ok=True)
        columns = [w.UID, 'SeriesInstanceUID', 'Anatomical_Plane', 'Fat_Suppression', 'Fluid_Sensitive']
        with z.open('train_series.csv') as f:
            rows = csv.DictReader(io.TextIOWrapper(f, encoding='utf-8'))
            w.need(set(columns) <= set(rows.fieldnames), 'series columns')
            metadata = [{k: row[k] for k in columns} for row in rows if row[w.UID] in ids]
        with (data / 'train_series.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader(); writer.writerows(metadata)
        identities = []
        for uid in ids:
            for entry in by_uid[uid]:
                dest = (data / entry.filename).resolve()
                w.need(dest.is_relative_to(data.resolve()), 'unsafe raw target')
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Overwrite only this lane's same named prepared artifact.
                tmp = dest.with_name(dest.name + '.tmp')
                with open_raw(z, entry) as src, tmp.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                w.replace(tmp, dest)
                identities.append((entry.filename, entry.CRC, entry.file_size, w.sha(dest)))
    rec = dict(schema='k1-shared-selection/v1', ids=ids, largest8=largest,
               counts={u: counts[u] for u in ids}, available=len(counts),
               compressed_selected=sorted(set(ids) & compressed),
               compressed_studies_observed=len(compressed),
               syntax_survey='first frame of every train series; mixed-syntax series not exhaustive',
               representative_syntax_counts=dict(syntaxes),
               unreadable_representatives=unreadable,
               raw_archive_bytes=RAW.stat().st_size,
               members=identities, embedded_pins=pins,
               series_sha256=w.sha(data / 'train_series.csv'),
               harness_sha256=w.sha(__file__), adapter_sha256=w.sha(shared.__file__),
               qualified_for_submission=False)
    w.write(OUT / 'selection.private.json', rec)
    print(json.dumps(dict(status='PREPARED', studies=32, largest=8,
                          compressed_selected=len(rec['compressed_selected']),
                          representative_syntax_counts=dict(syntaxes))), flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_source():
    """Use the parent's exact diagnostic/device patch, without a serving launch."""
    from tools import exp189_s6_port as s6
    selection = w.read(OUT / 'selection.private.json')
    for name, sha in selection['embedded_pins'].items():
        w.need(w.sha(OUT / 'parent' / name) == sha, 'parent source changed')
    w.need(w.sha(s6.__file__) == w.sha(OUT / 'parent/exp189_s6_port.py'), 'parent patch helper changed')
    runtime = OUT / 'runtime/r7_runtime'
    if not runtime.exists():
        runtime.parent.mkdir(parents=True, exist_ok=True)
        port = load_module('k1_parent_r7_port', OUT / 'parent/exp169_r7_port.py')
        port.diagnostic_source(OUT / 'parent/exp169_r7_src', runtime.parent)
    w.need(w.sha(runtime / 'compact_infer.py') == shared.R6_RUNTIME_SHA, 'runtime patch changed')
    for name, sha in shared.R6_SOURCES.items():
        w.need(w.sha(runtime / (name + '.py')) == sha, 'runtime dependency changed')
    return runtime


def timed_prediction(model, x, m, device, fp16, torch):
    if device != 'cpu':
        torch.cuda.synchronize()
    started = time.perf_counter()
    p = shared.predict(model, x, m, device, fp16=fp16)
    if device != 'cpu':
        torch.cuda.synchronize()
    return p.tolist(), time.perf_counter() - started


def run(arm, device='cpu', limit=None, classify=False, fp16=False, exercise_side=False):
    w.need(not fp16 or (device != 'cpu' and arm == 'shared' and limit is None),
           'FP16 requires full32 shared CUDA run')
    w.need(not exercise_side or (device == 'cpu' and arm == 'shared' and limit == 1),
           'side branch probe is CPU smoke only')
    tag = arm + ('-cpu-smoke' if limit else '-gpu' if device != 'cpu' else '-cpu')
    dest = OUT / tag
    dest.mkdir(parents=True, exist_ok=True)
    receipt = dest / 'receipt.private.json'
    # Invalidate previous success even if a native library kills Python later.
    w.write(receipt, dict(status='STARTING', records=[], qualified_for_submission=False))
    driver_sha = None
    if os.name == 'nt':
        # Existing reviewed process-local workaround; does not initialize CUDA.
        driver = r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
        driver_sha = '712cde267b942a6dbbea6e860a8a53372ad71ee6aadd94cca6bd1a424ab7e14b'
        w.need(w.sha(driver) == driver_sha, 'installed Windows DLL changed')
        os.environ['S6PLUS_PIN_DLL'] = driver
        from tools import s6plus
        w.need(s6plus.PINNED_DRIVER_SHA256 == driver_sha, 'process-local Windows DLL pin')
    torch = w.configure(device)
    selection = w.read(OUT / 'selection.private.json')
    ids = selection['ids'][:limit] if limit else selection['ids']
    w.need(limit == 1 or (len(ids) >= 32 and set(selection['largest8']) <= set(ids)), 'invalid roster')
    # Bind actual extracted bytes, not just a file count or external receipt.
    for name, crc, size, sha in selection['members']:
        if name.split('/')[1] in ids:
            w.need(w.sha(OUT / 'raw' / name) == sha, 'raw identity changed')
    w.need(w.sha(OUT / 'raw/train_series.csv') == selection['series_sha256'], 'series identity changed')
    wd = w.Watchdog(dest, receipt, device != 'cpu')
    records = []
    started = time.perf_counter()
    try:
        with w.offline():
            w.assets_check(ASSETS, ASSETS_SHA, wd.progress)
            if arm == 'original':
                ct, ci = w.import_frozen(ASSETS)
                w.need(w.sha(ci.__file__) == shared.K1_INFER_SHA, 'K1 source hash')
            else:
                src = runtime_source()
                sys.path[:0] = [str(src), str(ASSETS / 'cloud_code')]
                import compact_infer as ci
                # K1 classifier is ALWAYS built by K1's frozen trainer.
                ct = load_module('k1_original_trainer', ASSETS / 'source/compact_train.py')
            w.install_label_free_source(wd.progress)
            ds = ci.CStudies(OUT / 'raw', 'train', ids, dest / 'render', ASSETS / 'cloud_code', 384,
                             eval_stride=3 if arm == 'original' else 1, views='v4', slices=3,
                             anchor_weights=ASSETS / 'weights/anatomy.pt', phys_mm=3.5,
                             side_model=None if arm == 'original' else BUNDLE / 'side_cls.pt',
                             extent_meta=arm == 'shared', **({'fluid': True, 'device': device} if arm == 'shared' else {}))
            adapter = shared.SharedR6(ds, ci) if arm == 'shared' else None
            model = w.load_model(ct, ASSETS / 'weights/E_prior_s0_f0.pt', device) if classify or device != 'cpu' else None
            for index, uid in enumerate(ids):
                wd.progress('render-start')
                t = time.perf_counter()
                if adapter:
                    parent, item, detail = adapter[index]
                    # Parent is deliberately not classified by this engineering harness.
                    del parent
                else:
                    item, detail = ds[index], dict(route='original')
                render_seconds = time.perf_counter() - t
                x, m = w.validate_item(item, uid)
                row = dict(uid=uid, windows=len(x), images_sha256=w.tensor_sha(x),
                           meta_sha256=w.tensor_sha(m), render_seconds=render_seconds,
                           **detail)
                if exercise_side:
                    ci.FORCED_SIDE[uid] = 'L'
                    probe = shared.original_side(ds, ci, index)
                    row['forced_side_reset_bit_exact'] = (torch.equal(x, probe['images']) and torch.equal(m, probe['meta']))
                    w.need(row['forced_side_reset_bit_exact'], 'side rerender smoke differs')
                    ci.FORCED_SIDE.pop(uid, None)
                    del probe
                if model is not None:
                    wd.progress('classifier-fp32', len(x))
                    row['fp32'], row['fp32_seconds'] = timed_prediction(model, x, m, device, False, torch)
                    if fp16:
                        wd.progress('classifier-fp16', len(x))
                        row['fp16'], row['fp16_seconds'] = timed_prediction(model, x, m, device, True, torch)
                records.append(row)
                wd.completed += 1
                wd.progress('study-complete', len(x))
                w.write(receipt, dict(status='RUNNING', records=records, qualified_for_submission=False))
                print(json.dumps(dict(arm=arm, completed=len(records), windows=len(x), route=row['route'])), flush=True)
                del x, m, item
            rec = dict(status='COMPLETE', arm=arm, device=device, records=records,
                       selection_sha256=w.sha(OUT / 'selection.private.json'),
                       adapter_sha256=w.sha(shared.__file__), harness_sha256=w.sha(__file__),
                       renderer_sha256=w.sha(ci.__file__), worker_sha256=w.sha(w.__file__),
                       assets_manifest_sha256=ASSETS_SHA,
                       windows_driver_sha256=driver_sha,
                       environment=w.environment(device), fp16_requested=fp16,
                       total_seconds=time.perf_counter() - started,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved() if device != 'cpu' else None,
                       qualification_only=True, qualified_for_submission=False,
                       clean_oof=False, exposure='EXPOSED_EXPLORATORY')
            w.write(receipt, rec)
    except BaseException as exc:
        w.write(receipt, dict(status='FAILED', records=records, error=type(exc).__name__ + ': ' + str(exc)[:300],
                             qualified_for_submission=False))
        raise
    finally:
        wd.close()


def compare(cpu_smoke=False):
    aggregate = OUT / ('CPU-PARITY.json' if cpu_smoke else 'GPU-PARITY.json')
    aggregate.unlink(missing_ok=True)  # Never retain an old PASS after a failed/crashed comparison.
    suffix = '-cpu-smoke' if cpu_smoke else '-gpu'
    a = w.read(OUT / ('original' + suffix) / 'receipt.private.json')
    b = w.read(OUT / ('shared' + suffix) / 'receipt.private.json')
    for rec in (a, b):
        w.need(rec['status'] == 'COMPLETE' and rec['adapter_sha256'] == w.sha(shared.__file__)
               and rec['harness_sha256'] == w.sha(__file__), 'stale/incomplete result')
    w.need(a['selection_sha256'] == b['selection_sha256'] == w.sha(OUT / 'selection.private.json'), 'roster differs')
    ea, eb = ({k: v for k, v in rec['environment'].items() if k != 'pid'} for rec in (a, b))
    w.need(a['device'] == b['device'] and ea == eb and
           a['windows_driver_sha256'] == b['windows_driver_sha256'], 'paired environment differs')
    w.need(a['renderer_sha256'] == shared.K1_INFER_SHA and b['renderer_sha256'] == shared.R6_RUNTIME_SHA,
           'wrong renderer pair')
    w.need(a['worker_sha256'] == b['worker_sha256'] == w.sha(w.__file__), 'worker changed')
    ra, rb = a['records'], b['records']
    w.need([r['uid'] for r in ra] == [r['uid'] for r in rb], 'UID order differs')
    w.need(cpu_smoke or len(ra) >= 32, 'full parity needs >=32')
    fp16_requested = b.get('fp16_requested', False)
    if fp16_requested:
        w.need(len(rb) >= 32 and all('fp16' in r for r in rb),
               'requested FP16 needs results for every study (>=32)')
    exact = all(x['images_sha256'] == y['images_sha256'] and x['meta_sha256'] == y['meta_sha256']
                and x['windows'] == y['windows'] for x, y in zip(ra, rb))
    result = dict(tensors_bit_exact=exact, studies=len(ra), cpu_smoke=cpu_smoke,
                  prediction_status='NOT_RUN', qualified_for_submission=False)
    import statistics
    result['timing_seconds'] = {
        name: {key: dict(mean=statistics.mean(r[key] for r in records),
                        median=statistics.median(r[key] for r in records),
                        maximum=max(r[key] for r in records))
               for key in ('render_seconds', 'extra_render_seconds', 'fp32_seconds', 'fp16_seconds')
               if all(key in r for r in records)}
        for name, records in (('original', ra), ('shared', rb))}
    result['side_rerender_studies'] = sum(r['route'] == 'original-side-rerender' for r in rb)
    if all('fp32' in r for r in ra + rb):
        half = [r['fp16'] for r in rb] if all('fp16' in r for r in rb) else None
        result.update(shared.compare_predictions([r['fp32'] for r in ra], [r['fp32'] for r in rb], half))
        result['prediction_status'] = 'PASS' if result['fp32_pass'] else 'FAIL'
    result['status'] = 'PASS' if exact and result['prediction_status'] != 'FAIL' and result.get('fp16_status') != 'FAIL' else 'FAIL'
    if fp16_requested and result.get('fp16_status') != 'PASS':
        result['status'] = 'FAIL'
    if not cpu_smoke:
        w.need(result['prediction_status'] != 'NOT_RUN', 'missing CUDA predictions')
    w.write(aggregate, result)
    print(json.dumps(result), flush=True)
    w.need(result['status'] == 'PASS', 'parity gate failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    r = sub.add_parser('run')
    r.add_argument('--arm', choices=['original', 'shared'], required=True)
    r.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    r.add_argument('--limit', type=int, choices=[1])
    r.add_argument('--classify', action='store_true')
    r.add_argument('--fp16', action='store_true')
    r.add_argument('--exercise-side', action='store_true')
    c = sub.add_parser('compare'); c.add_argument('--cpu-smoke', action='store_true')
    a = p.parse_args()
    if a.command == 'prepare': prepare()
    elif a.command == 'compare': compare(a.cpu_smoke)
    else: run(a.arm, a.device, a.limit, a.classify, a.fp16, a.exercise_side)


if __name__ == '__main__':
    main()
