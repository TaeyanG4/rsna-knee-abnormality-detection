"""Offline EXP140 AUG1 release builder and Public3 verifier.

The two kernels are private candidates. This tool never uploads, pushes, runs
Kaggle, initializes CUDA, or submits. `stage` copies only after SHA checks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import exp133_oai_b3 as e133

OUT = ROOT / '.local/exp140-aug1-release'
OLD = ROOT / '.local/exp133-oai-b3/blend_b3/release.ipynb'
DATASET = 'taeyangg4/knee-exp140-radr50-aug1-2seed'
ASSET_NAME = 'exp133-rad-assets.json'  # existing release adapter's marker
OLD_DATASET = e133.RAD_DATASET
BASE_SHA = '3ec638eae23a78abe143baef12978e9d7491a05bf041aa0cd8ad946e96a14189'
RATE_60 = 301.32550326700016 / 60  # EXP133 T4 timing60, includes load/decode


def need(value, reason):
    if not value:
        raise ValueError(reason)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def tag(seed, fold):
    return ('EXP127_AUG1_S%d_F0' % seed if fold == 0 else
            'EXP134_AUG1_S%d_F%d' % (seed, fold))


def checkpoint_entry(root, seed, fold, hash_weight=True):
    name = tag(seed, fold)
    run = Path(root) / 'artifacts/experiments' / name
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    provenance = json.loads((run / 'provenance.json').read_text(encoding='utf-8'))
    inventory = json.loads((run / 'output-inventory.json').read_text(encoding='utf-8'))
    model = run / 'model_best.pt'
    expected = inventory['model_best.pt']
    need(all(digest(run / name) == inventory[name]['sha256'] and
             (run / name).stat().st_size == inventory[name]['bytes']
             for name in ('summary.json', 'provenance.json')),
         'AUG1 receipt inventory mismatch: ' + name)
    # These runs have no literal result.json/status=SUCCESS. Their completed
    # summary, clean workers, provenance and hashed output inventory are the
    # success evidence; do not fabricate a SUCCESS field.
    need(summary['fold'] == fold and summary['mode'] == 'full' and
         summary['selected_checkpoint'] == 'fixed-final' and
         summary['worker_cleanup']['clean'] is True and summary['total_seconds'] > 0,
         'Incomplete training summary: ' + name)
    need(summary['provenance']['seed'] == seed and
         summary['provenance']['fold'] == fold and
         provenance['seed'] == seed and provenance['fold'] == fold and
         provenance['exp127_variant'] == 'AUG1' and
         summary['provenance']['protocol_sha256'] == provenance['protocol_sha256'],
         'AUG1 provenance mismatch: ' + name)
    need(provenance['arch'] == 'resnet50' and provenance['norm'] == 'radimagenet' and
         provenance['input_policy'] == 'center3' and model.stat().st_size == expected['bytes'],
         'AUG1 serving contract/bytes mismatch: ' + name)
    if hash_weight:
        need(digest(model) == expected['sha256'], 'AUG1 model hash mismatch: ' + name)
    return {'seed': seed, 'fold': fold, 'tag': name,
            'path': 'checkpoints/rad%d_f%d.pt' % (seed, fold),
            'sha256': expected['sha256'], 'bytes': expected['bytes'],
            'protocol_sha256': provenance['protocol_sha256']}


def manifest(root=ROOT, hash_weight=True):
    checks = [checkpoint_entry(root, seed, fold, hash_weight)
              for seed in (42, 43) for fold in range(5)]
    need(len(checks) == 10, 'Ten checkpoints required')
    return {'schema': 'exp133-rad-assets-v1', 'dataset': DATASET,
            'training': 'AUG1, D-B-066; serving unchanged from EXP133',
            'success_evidence': 'completed summary + clean workers + matching provenance + output-inventory SHA-256',
            'checkpoints': checks}


def runtime_projection(n=1300):
    need(type(n) is int and n > 0, 'Positive study count required')
    # 30% headroom plus 600 seconds load/install allowance; 120 s reserve.
    seconds = 1.3 * RATE_60 * n + 600
    return {'studies': n, 'measured_seconds_per_study': RATE_60,
            'basis': '.local/exp133-oai-b3/timing60-output/timing-rad.json: 301.325503267 s / 60 studies',
            'projected_seconds': seconds, 'deadline_seconds': 30600,
            'publication_reserve_seconds': 120, 'margin_seconds': 30600 - 120 - seconds}


def source133():
    original = (HERE / 'exp133_oai_b3.py').read_text(encoding='utf-8')
    before = "RAD_DATASET = '" + OLD_DATASET + "'"
    after = "RAD_DATASET = '" + DATASET + "'"
    need(original.count(before) == 1, 'EXP133 dataset binding changed')
    adapted = original.replace(before, after)
    compile(adapted, 'exp133_oai_b3.py', 'exec')
    return original, adapted


def old_payload_literal(tail):
    prefix = '_EXP133_PAYLOAD = json.loads(_exp133.e128.unpack('
    start = tail.index(prefix) + len(prefix)
    # B fix: the payload closes with ')))' (tuple, unpack, json.loads); keep only the tuple's ')' in the literal
    end = tail.index(')))\n', start)
    literal = tail[start:end + 1]  # includes outer parenthesis
    return literal, json.loads(e133.e128.unpack(ast.literal_eval(literal)))


def old_source_literal(tail):
    prefix = '_exp133_write("exp133_oai_b3.py", '
    start = tail.index(prefix) + len(prefix)
    end = tail.index('))\n', start)
    literal = tail[start:end + 1]
    return literal, e133.e128.unpack(ast.literal_eval(literal))


def b_notebook(old_raw, manifest_data):
    need(hashlib.sha256(old_raw.encode('utf-8')).hexdigest() == BASE_SHA,
         'Frozen EXP133 notebook changed')
    old_nb = json.loads(old_raw)
    tail = old_nb['cells'][-1]['source']
    old_source, adapted = source133()
    old_src_literal, embedded_source = old_source_literal(tail)
    need(embedded_source == old_source, 'Embedded EXP133 source changed')
    old_literal, payload = old_payload_literal(tail)
    need(payload['rad_manifest']['dataset'] == OLD_DATASET, 'Old payload binding changed')
    payload['rad_manifest'] = manifest_data
    new_tail = tail.replace(old_src_literal, e133.literal_bytes(adapted), 1)
    new_tail = new_tail.replace(old_literal, e133.literal_bytes(json.dumps(payload)), 1)
    # Replace exactly one JSON string: all bytes outside the Rad binding cell
    # stay byte-for-byte equal to the released EXP133 notebook.
    old_encoded = json.dumps(tail, ensure_ascii=False)[1:-1]
    new_encoded = json.dumps(new_tail, ensure_ascii=False)[1:-1]
    need(old_raw.count(old_encoded) == 1, 'Notebook tail serialization differs')
    new_raw = old_raw.replace(old_encoded, new_encoded, 1)
    new_nb = json.loads(new_raw)
    need(new_nb['cells'][:-1] == old_nb['cells'][:-1], 'Frozen notebook cells changed')
    for i, cell in enumerate(new_nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'EXP140 B cell %d' % i, 'exec')
    return new_raw


def a_notebook(manifest_data):
    _, adapted = source133()
    source128 = (HERE / 'exp128_oai_light.py').read_text(encoding='utf-8')
    payload = {'rad_manifest': manifest_data,
               'files': e133.payload()['files']}  # frozen serving files only
    setup = ('import base64, zlib, pathlib, json, time, hashlib, sys\n'
             'def _write(name, value):\n'
             '    pathlib.Path("/kaggle/working", name).write_text(zlib.decompress(base64.b64decode(value)).decode(), encoding="utf-8", newline="\\n")\n'
             '_write("exp128_oai_light.py", ' + e133.literal_bytes(source128) + ')\n'
             '_write("exp133_oai_b3.py", ' + e133.literal_bytes(adapted) + ')\n'
             'sys.path.insert(0, "/kaggle/working")\n'
             'import exp133_oai_b3 as e\n'
             'data = json.loads(e.e128.unpack(' + e133.literal_bytes(json.dumps(payload)) + '))\n'
             'working = pathlib.Path("/kaggle/working")\n'
             'start = time.monotonic()\n'
             'sample = e.input_dir(e.e128.COMPETITION, "sample_submission.csv") / "sample_submission.csv"\n'
             'ids = e.sample_ids(sample)\n'
             'projection = 1.3 * ' + repr(RATE_60) + ' * len(ids) + 600\n'
             'e.need(e.can_start(0, projection), "EXP140 A cannot finish by 30600 s deadline")\n'
             'record = {"schema": "exp140-aug1-rad-standalone-v1", "status": "RUNNING", "studies": len(ids), "projection_seconds": projection}\n'
             'e.write_json(working / "exp140_stage_timing.json", record)\n'
             'try:\n'
             '    stage = {}\n'
             '    values = e.rad_stage(data, start, ids, stage, working)\n'
             '    e.need(time.monotonic() - start <= e.CUTOFF_S - e.RESERVE_S, "EXP140 A deadline exceeded")\n'
             '    record["worker"] = stage["worker"]\n'
             '    record["seed_reduction"] = "0.5 rank(seed42 five-fold TTA mean) + 0.5 rank(seed43 five-fold TTA mean), average ties per target over all test studies"\n'
             '    record["submission_sha256"] = e.publish(working, ids, values)\n'
             '    record["status"] = "PASS"\n'
             'finally:\n'
             '    record["elapsed_seconds"] = time.monotonic() - start\n'
             '    e.write_json(working / "exp140_stage_timing.json", record)\n'
             'print("EXP140 A", record["status"], record.get("submission_sha256"))\n')
    cells = [e133.e128.cell('# EXP140 AUG1 standalone RadR50. Competition and private own-weight dataset only.\n'
                             'K36, center3, RadImageNet, plane-aware flip TTA. Five fold probabilities\n'
                             'are averaged within each seed; each seed is ranked per target across the\n'
                             'entire test cohort with average ties; the two percentile ranks are averaged\n'
                             '50:50 and written as bounded AUC scores.\n', 'markdown'),
             e133.e128.cell(setup)]
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'EXP140 A cell %d' % i, 'exec')
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}}, 'cells': cells}


def screen(kind, directory, entries):
    prior = json.loads((ROOT / '.local/exp133-oai-b3/blend_b3/submission_screen_spec.json').read_text(encoding='utf-8'))
    prior['candidate'] = 'EXP140_AUG1_RAD' if kind == 'A' else 'EXP140_B3_AUG1'
    prior['variant'] = 'rad_standalone' if kind == 'A' else 'blend_b3_aug1'
    prior['members'] = ([{'name': 'rad', 'arm': 'rad', 'weight': 1.0, 'oof': None,
                          'oof_reason': 'AUG1 weak-label OOF only; no independent selection OOF',
                          'checkpoint_sha256': [c['sha256'] for c in entries]}]
                        if kind == 'A' else prior['members'])
    if kind == 'B':
        prior['members'][2]['checkpoint_sha256'] = [c['sha256'] for c in entries]
    prior['release'] = {'kernel_dir': str(directory), 'out_dir': str(OUT / ('public3-a-output' if kind == 'A' else 'public3-b-output'))}
    prior['decision'] = {'plan_item': 'H-B-58' if kind == 'B' else 'selection A / within-family ruler',
                         'basis': 'public', 'text': 'Public3 integration first; B controls selection and user approval.'}
    prior['blend'] = ({'weights': {'rad': 1.0}, 'seed_reduction':
                       '0.5 rank(seed42 five-fold TTA mean) + 0.5 rank(seed43 five-fold TTA mean)'}
                      if kind == 'A' else prior['blend'])
    prior['screen_limitation'] = 'HOLD until Public3 validates receipt, hashes, cohort and arithmetic; no score inferred.'
    prior['approval_required'] = kind == 'B'
    return prior


def build(out=OUT, root=ROOT):
    out, root = Path(out), Path(root)
    need(not out.exists(), 'Choose a fresh output directory')
    data = manifest(root)
    old_raw = OLD.read_text(encoding='utf-8')
    a_dir, b_dir = out / 'rad_standalone', out / 'blend_b3_aug1'
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    asset_dir = out / 'upload-stage'
    asset_dir.mkdir()
    save(asset_dir / ASSET_NAME, data)
    save(asset_dir / 'dataset-metadata.json', {
        'title': DATASET.split('/')[-1], 'id': DATASET, 'isPrivate': True,
        'licenses': [{'name': 'other'}],
        'description': 'Private AUG1 RadR50 ten-fold release checkpoints; competition data only.'})
    files = []
    for ck in data['checkpoints']:
        files.append({'source': str(root / 'artifacts/experiments' / ck['tag'] / 'model_best.pt'),
                      'destination': ck['path'], 'sha256': ck['sha256'], 'bytes': ck['bytes']})
    asset = asset_dir / ASSET_NAME
    files.append({'source': str(asset), 'destination': ASSET_NAME,
                  'sha256': digest(asset), 'bytes': asset.stat().st_size})
    metadata_file = asset_dir / 'dataset-metadata.json'
    files.append({'source': str(metadata_file), 'destination': 'dataset-metadata.json',
                  'sha256': digest(metadata_file), 'bytes': metadata_file.stat().st_size})
    save(out / 'upload-manifest.json', {'schema': 'exp140-stage-v1', 'dataset': DATASET,
                                        'status': 'NOT_UPLOADED', 'files': files})
    save(a_dir / 'release.ipynb', a_notebook(data))
    (b_dir / 'release.ipynb').write_text(b_notebook(old_raw, data), encoding='utf-8')
    for kind, directory, slug in [('A', a_dir, 'knee-exp140-radr50-aug1'),
                                  ('B', b_dir, 'knee-exp140-b3-aug1')]:
        meta = (e133.e128.metadata('blend_b1') if kind == 'B' else e133.e128.metadata('fidelity'))
        meta.update(id='taeyangg4/' + slug, title=slug,
                    dataset_sources=([e133.e128.DATASET, e133.e128.READER_DATASET, DATASET]
                                     if kind == 'B' else [DATASET]))
        save(directory / 'kernel-metadata.json', meta)
        save(directory / 'submission_screen_spec.json', screen(kind, directory, data['checkpoints']))
        save(directory / 'runtime-projection-1300.json', runtime_projection())
    save(b_dir / 'build-receipt.json', {
        'schema': 'exp140-b3-build-v1', 'status': 'UNPUSHED',
        'exp133_notebook_sha256': BASE_SHA,
        'notebook_sha256': digest(b_dir / 'release.ipynb'),
        'rad_manifest_sha256': hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        'registered_weights': e133.WEIGHTS})
    save(out / 'build-receipt.json', {'schema': 'exp140-build-v1', 'status': 'UNPUSHED',
                                     'exp133_notebook_sha256': BASE_SHA,
                                     'notebook_sha256': {k: digest(d / 'release.ipynb') for k, d in [('A', a_dir), ('B', b_dir)]},
                                     'asset_sha256': digest(asset), 'checkpoints': data['checkpoints'],
                                     'network_calls': 0, 'gpu_runs': 0, 'submissions': 0})
    return {'status': 'BUILT', 'out': str(out)}


def stage(out=OUT):
    out = Path(out)
    manifest_data = json.loads((out / 'upload-manifest.json').read_text(encoding='utf-8'))
    stage_dir = out / 'upload-stage'
    for entry in manifest_data['files']:
        source, destination = Path(entry['source']), stage_dir / entry['destination']
        need(source.stat().st_size == entry['bytes'] and digest(source) == entry['sha256'],
             'Source hash/bytes differ: ' + str(source))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            need(not destination.exists(), 'Refuse to overwrite: ' + str(destination))
            shutil.copyfile(source, destination)
        need(destination.stat().st_size == entry['bytes'] and digest(destination) == entry['sha256'],
             'Staged hash/bytes differ: ' + str(destination))
    return {'status': 'STAGED_HASH_VERIFIED', 'files': len(manifest_data['files'])}


def validate_public3(kind, output, sample, build_dir=None):
    output, sample = Path(output), Path(sample)
    build_dir = Path(build_dir or OUT / ('rad_standalone' if kind == 'A' else 'blend_b3_aug1'))
    if kind == 'B':
        return e133.validate_public3(output, sample, build_dir)
    ids, final = e133.validate_csv(output / 'submission.csv', sample)
    need(len(ids) == 3, 'Public3 requires exactly three studies')
    receipt = json.loads((output / 'exp140_stage_timing.json').read_text(encoding='utf-8'))
    need(receipt['schema'] == 'exp140-aug1-rad-standalone-v1' and receipt['status'] == 'PASS',
         'Standalone receipt differs')
    need(receipt['submission_sha256'] == digest(output / 'submission.csv'), 'Submission hash differs')
    _, partner = e133.validate_csv(output / 'exp133_rad.csv', sample)
    need(final == partner, 'Standalone rank arithmetic differs')
    worker = receipt['worker']
    need(worker['status'] == 'PASS' and worker['studies'] == 3, 'Rad worker failed')
    for seed in (42, 43):
        path = output / 'exp133_rad' / ('seed%d.csv' % seed)
        need(digest(path) == worker['csv_sha256'][str(seed)], 'Seed receipt hash differs')
    return {'status': 'PASS_A_PUBLIC3', 'studies': 3,
            'submission_sha256': digest(output / 'submission.csv'), 'submission_authorized': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('build').add_argument('--out', type=Path, default=OUT)
    sub.add_parser('stage').add_argument('--out', type=Path, default=OUT)
    val = sub.add_parser('validate-public3')
    val.add_argument('--kernel', choices=('A', 'B'), required=True)
    val.add_argument('--output', type=Path, required=True)
    val.add_argument('--sample', type=Path, default=ROOT / 'storage/competition/sample_submission.csv')
    val.add_argument('--build-dir', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'build':
        result = build(args.out)
    elif args.command == 'stage':
        result = stage(args.out)
    else:
        result = validate_public3(args.kernel, args.output, args.sample, args.build_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
