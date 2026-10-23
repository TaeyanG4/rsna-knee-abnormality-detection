"""Build two local-only, sealed v1.3.0 variants from the immutable v1.2.2 archive.

Changelog 2026-10-13: native flip on fixed f0+f2 / all R7 folds; no other math changes.
"""
from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import exp169_r7_selective_flip_v130 as flip

VERSION = '1.3.0'
BASE = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.2.2'
OUT = BASE.parent / ('v' + VERSION)
SCRATCH = ROOT / '.local/scratch/astra-b5/c85r7-flip'
MODULE = 'exp169_c85r7flip_runtime_v130'
CHECKS = 'exp169_c85r7flip_checks_v130'
ADAPTER = 'exp169_r7_selective_flip_v130'
BASE_RECEIPT_SHA = '423f09f9d16d2e60effec095ed4d5b50d48dd50df90a91c5694b27b71beb06ee'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('sealed artifact already differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def frozen_files():
    receipt_bytes = (BASE / 'build-receipt.json').read_bytes()
    if digest(receipt_bytes) != BASE_RECEIPT_SHA:
        raise ValueError('v1.2.2 receipt drift')
    receipt = json.loads(receipt_bytes)
    nb_bytes = (BASE / 'kernel/release.ipynb').read_bytes()
    assert digest(nb_bytes) == receipt['artifacts']['kernel/release.ipynb']
    code = ''.join(json.loads(nb_bytes)['cells'][-1]['source'])
    tree = ast.parse(code)
    blob = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'BLOB' for t in n.targets))
    raw = base64.b64decode(blob)
    assert digest(raw) == receipt['archive_sha256']
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    assert {k: digest(v) for k, v in files.items()} == receipt['files']
    return files


def runtime_source(files, variant):
    source = files['exp169_release_c85r7codec_deadline_v122.py'].decode()
    source = source.replace("VERSION, RELEASE = '1.2.2', 'C85R7+codecs+insurance'",
                            f"VERSION, RELEASE = '1.3.0', 'C85R7+codecs+insurance+{variant}'")
    source = source.replace("MODULE = 'exp169_release_c85r7codec_deadline_v122'", f"MODULE = '{MODULE}'")
    source = flip.replace(source, "OUT = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.2.2'",
        f"OUT = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.3.0/{variant}'")
    source = flip.replace(source, "KERNEL_ID = 'taeyangg4/knee-exp169-c85r7ins-v122'",
        f"KERNEL_ID = 'taeyangg4/knee-c85r7-{variant.lower().replace('_', '-')}-v130'")
    source = flip.replace(source, '    args = parser.parse_args()',
        '    args = parser.parse_args()\n'
        '    if args.command not in ("run", "child"):\n'
        '        parser.error("sealed runtime: use the v1.3.0 builder for preparation")')
    source = flip.replace(source, 'def run_s(runtime, output):',
        f'FLIP_VARIANT = {variant!r}\n\n\ndef run_s(runtime, output):')
    start = source.index('    r7.infer_studies(', source.index('def run_s('))
    end = source.index('\n\n\ndef _child_entry', start)
    source = source[:start] + f'    import {ADAPTER} as selective_flip\n' + \
        '    with selective_flip.installed(r7, FLIP_VARIANT, metadata_io):\n' + \
        ''.join('    ' + line for line in source[start:end].splitlines(True)) + source[end:]
    source = flip.replace(source, '    return receipt\n\n\ndef failed_execution',
        f'    receipt["flip_policy"] = __import__({ADAPTER!r}).policy(FLIP_VARIANT)\n'
        '    return receipt\n\n\ndef failed_execution')
    compile(source, '<flip-release>', 'exec')
    return source.encode()


def runtime_projection(variant, studies=1300):
    count = len(flip.VARIANTS[variant])
    slope = (680.704481005 - 344.507129447) / (32 - 3)
    extra = studies * slope * count / 4 / 3600
    total = 4.7 + extra
    return dict(studies=studies, base_hours=4.7, incremental_hours=extra, projected_hours=total,
        margin_to_9h_minutes=(9-total)*60, within_7_5h=total <= 7.5,
        conditional_calibrated_hours=[4.7 + extra*c for c in (.75, .84)],
        formula='4.7 + N * ((680.704481005-344.507129447)/(32-3)) * (flip_fold_count/4) * c / 3600',
        go_uses_c=1, assumptions='N~1300 unknown hidden size; equal fold cost; .75-.84 incremental-only calibration unproven')


def payloads(variant):
    files = frozen_files()
    files[MODULE + '.py'] = runtime_source(files, variant)
    files[ADAPTER + '.py'] = Path(flip.__file__).read_bytes()
    checks = files['exp169_c85r7ins_checks_v122.py'].decode().replace(
        'exp169_release_c85r7codec_deadline_v122', MODULE)
    files[CHECKS + '.py'] = checks.encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            archive.writestr(info, data)
    blob = buffer.getvalue()
    pins = {k: digest(v) for k, v in sorted(files.items())}
    artifacts = {}
    for kind in ('kernel', 'stress-kernel-clean', 'stress-kernel'):
        original = (BASE / kind / 'release.ipynb').read_bytes()
        assert digest(original) == read(BASE / 'build-receipt.json')['artifacts'][kind + '/release.ipynb']
        nb = json.loads(original)
        code = ''.join(nb['cells'][-1]['source'])
        lines = code.splitlines(True)
        for i, line in enumerate(lines):
            if line.startswith('BLOB = '):
                lines[i] = f'BLOB = {base64.b64encode(blob).decode()!r}\n'
            if line.startswith('PINS = '):
                lines[i] = f'PINS = {pins!r}\n'
        code = ''.join(lines).replace('import exp169_release_c85r7codec_deadline_v122 as release',
                                     f'import {MODULE} as release').replace(
                                     'import exp169_c85r7ins_checks_v122 as checks', f'import {CHECKS} as checks')
        code = flip.replace(code,
            f"assert hashlib.sha256(blob).hexdigest() == {read(BASE / 'build-receipt.json')['archive_sha256']!r}",
            f"assert hashlib.sha256(blob).hexdigest() == {digest(blob)!r}")
        code = flip.replace(code, "print('C85R7+codecs+insurance v1.2.2', flush=True)",
                            f"print('C85R7+codecs+insurance v1.3.0 {variant}', flush=True)")
        compile(code, '<sealed-notebook>', 'exec')
        nb['cells'][-1]['source'] = code.splitlines(True)
        nb['cells'][0]['source'] += [f'\nC85-R7 flip v1.3.0 / {variant}; sealed fixed folds {flip.VARIANTS[variant]}. '
                                    'LOCAL ONLY; select at most one variant by H-B-149 gate and <=7.5h.\n']
        meta = read(BASE / kind / 'kernel-metadata.json')
        suffix = {'kernel': '', 'stress-kernel-clean': '-clean-n32', 'stress-kernel': '-stress-n32'}[kind]
        meta.update(id=f'taeyangg4/knee-c85r7-{variant.lower().replace("_", "-")}-v130{suffix}')
        meta['title'] = meta['id'].split('/')[1]
        artifacts[kind + '/release.ipynb'] = encoded(nb)
        artifacts[kind + '/kernel-metadata.json'] = encoded(meta)
    record = dict(status='PREPARED_LOCAL_ONLY', version=VERSION, policy=flip.policy(variant),
        base='C85-R7 v1.2.2 insurance/deadline', base_receipt_sha256=BASE_RECEIPT_SHA,
        files=pins, archive_sha256=digest(blob), runtime=runtime_projection(variant),
        artifacts={k: digest(v) for k, v in artifacts.items()}, qualified_for_submission=False)
    artifacts['build-receipt.json'] = encoded(record)
    return artifacts, files


def prepare(out=OUT):
    for variant in flip.VARIANTS:
        artifacts, files = payloads(variant)
        for name, data in artifacts.items():
            write(out / variant / name, data)
        # Small added source files for review; the original closure remains in the archive.
        for name in (MODULE+'.py', CHECKS+'.py', ADAPTER+'.py'):
            write(out / variant / 'sources' / name, files[name])
    write(out / 'sources' / Path(__file__).name, Path(__file__).read_bytes())
    return dict(status='PREPARED_LOCAL_ONLY', variants=list(flip.VARIANTS))


def validate(out=OUT):
    for variant in flip.VARIANTS:
        artifacts, files = payloads(variant)
        for name, data in artifacts.items():
            if (out / variant / name).read_bytes() != data:
                raise ValueError('sealed artifact drift: '+variant+'/'+name)
        for name in (MODULE+'.py', CHECKS+'.py', ADAPTER+'.py'):
            assert (out / variant / 'sources' / name).read_bytes() == files[name]
    assert (out / 'sources' / Path(__file__).name).read_bytes() == Path(__file__).read_bytes()
    return dict(status='PASS_CPU_VALIDATE', variants=2, base_files_unchanged=len(frozen_files()),
                qualified_for_submission=False)


def registration():
    return dict(version=VERSION, variants={v: dict(policy=flip.policy(v), runtime=runtime_projection(v)) for v in flip.VARIANTS},
        paired_readout='.local/scratch/astra-b5/H-B-149/full-v121',
        rule='B chooses ONE named GO variant under H-B-149 v1.2.1: macro >=.001, CI lower >0, >=95% defined draws, every target >=-.005, unscaled projection <=7.5h; otherwise neither',
        real_paired_auc='PENDING; no labels accessed',
        scope='EXPOSED heldout model-policy screen excludes f0; not four-model serving ensemble or C85 blend AUC',
        unchosen_variant='NEVER PUSH', qualified_for_submission=False)


def register(out=OUT):
    validate(out)
    write(out / 'registration.json', encoded(registration()))
    return dict(status='REGISTERED_LOCAL_ONLY', variants=list(flip.VARIANTS))


def cpu_replay(output):
    from tools import exp169_release_c85r7codec_deadline_v122 as base
    result = base.cpu_replay(output)
    files = frozen_files()
    # Source equality covers every non-R7 branch; recorded real numeric replay
    # covers final/P arithmetic. Actual target inference parity remains required.
    for variant in flip.VARIANTS:
        _, augmented = payloads(variant)
        assert all(augmented[k] == v for k, v in files.items())
        runtime = output / variant / 'runtime'
        for name, data in augmented.items():
            write(runtime / name, data)
        sys.path.insert(0, str(runtime))
        name = MODULE + '_' + variant
        spec = importlib.util.spec_from_file_location(name, runtime / (MODULE+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        receipt, payload = module.collect(base.c85.A5Q, output / 'replay-input/pr7',
                                          output / variant, diagnostic=True)
        assert payload == (output / base.DIAGNOSTIC).read_bytes()
        assert (output / variant / base.P_BRANCH).read_bytes() == (output / base.P_BRANCH).read_bytes()
        assert receipt['flip_policy'] == flip.policy(variant)
        sys.path.remove(str(runtime))
    result.update(v122_non_r7_source_bytes_identical=True, v122_csv_and_p_bytes_identical=True,
                  variants=list(flip.VARIANTS))
    write(output / 'flip-replay.json', encoded(result))
    return result


def admit(out=OUT, replay_dir=None):
    validate(out)
    assert read(out / 'registration.json') == registration()
    replay = read(replay_dir / 'flip-replay.json')
    assert replay['v122_non_r7_source_bytes_identical'] and replay['v110_csv_identical']
    return dict(status='HOLD_NOT_QUALIFIED', cpu_preparation='PASS', qualified_for_submission=False,
        remaining=['real paired readout and B single-variant choice', 'same-image visible and clean/fault N32 T4 qualification',
                   'actual healthy non-R7 byte parity; full-cohort deadline publication'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','validate','register','admit','cpu-replay'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--output', type=Path, default=SCRATCH/'replay')
    parser.add_argument('--replay-dir', type=Path, default=SCRATCH/'replay')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.command == 'admit':
        if not args.dry_run:
            parser.error('admit requires --dry-run')
        result = admit(args.out, args.replay_dir)
    elif args.command == 'cpu-replay':
        result = cpu_replay(args.output)
    else:
        result = globals()[args.command](args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
