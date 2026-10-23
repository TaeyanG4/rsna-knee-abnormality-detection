"""H207 v1.0.2 qualification-only N32 packaging; no remote execution.

Changelog: copied R224 root selector honors the existing C85_COMPETITION input
override; one copied runtime pin is updated. All inference/launch code is frozen.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import io
import os
from pathlib import Path
import sys
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import c85r7_r6w958inf2_v1148 as existing

need, sha, digest, read, encoded, replace = (existing.need, existing.sha,
    existing.digest, existing.read, existing.encoded, existing.replace)
VERSION = '1.0.2'
WORK = ROOT / '.local/scratch/slot-b/research-resume-1018/h207-build'
STAGE = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.49-r6w958c4inf2'
OUT = WORK / 'stress-v102'
SLUG = 'taeyangg4/knee-lb-r6w958c4inf2-v1-stress-n32'
R224 = 'raptor224_run.py'
CODEC = 'exp169_release_c85r7codec.py'
NORMAL_PINS = {
    'kernel/release.ipynb': '4089a26aaf74a7536fa8563da970587ee2bf5fc5b87e737c7baf78b004bca61c',
    'kernel/kernel-metadata.json': '428a565bce87fa67acbb458e1e3968cf13ef98ebea3c75a8e0a813909ddd0531',
    'build-receipt.json': '7b0b4657b5751f70c534999d915c3722c7003b3af6546dc0cefd1de9c6f01763',
}
V1_PINS = {
    'BUILD-REPORT.md': 'ba9d3db88ae27def0b14d42881b26ab15637253c5aab53095fd2a16bcb5a95d6',
    'CPU-COUNTEREXAMPLE.json': '27d26aa0250c54bf5ed53695eb565bd599f5f13802145507513805245bf4a593',
    'cpu-root-counterexample.py': 'f679f64acf0d0c3ad4a28dd96098ba8f14347e6e472a418e2a8de1ea75de958c',
}
ADAPTER = '''    override = os.environ.get("C85_COMPETITION")
    if override is not None:
        if not override or not os.path.isfile(os.path.join(override, "test.csv")) or not (
                os.path.isdir(os.path.join(override, "test_series")) or
                os.path.isdir(os.path.join(override, "test_images"))):
            raise RuntimeError("Invalid C85_COMPETITION test root")
        return override
'''


def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def extract(source, names, env):
    fs = functions(source)
    scope = dict(env)
    module = ast.Module(body=[fs[n] for n in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<pinned-packaging-functions-only>', 'exec'), scope)
    return scope


def inputs():
    need({p: sha(STAGE / p) for p in NORMAL_PINS} == NORMAL_PINS, 'normal stage identity')
    need({p: sha(WORK / 'stress-v1' / p) for p in V1_PINS} == V1_PINS, 'preserved blocker evidence')
    nb = read(STAGE / 'kernel/release.ipynb')
    files, blob, assignments = existing.unpack(nb)
    need(len(files) == 31, '31 embedded files')
    need(all((WORK / 'candidate-runtime' / n).read_bytes() == v for n, v in files.items()),
         'normal candidate runtime drift')
    meta = read(STAGE / 'kernel/kernel-metadata.json')
    need(meta['id'] == SLUG.removesuffix('-stress-n32')
         and len(meta['dataset_sources']) == len(set(meta['dataset_sources'])) == 8
         and meta['competition_sources'] == ['rsna-knee-abnormality-detection']
         and meta['machine_shape'] == 'NvidiaTeslaT4' and meta['enable_gpu'] is True
         and meta['enable_tpu'] is False and meta['enable_internet'] is False
         and meta['is_private'] is True, 'normal metadata')
    need(''.join(nb['cells'][0]['source']).strip() == 'FAST = False', 'ordinary no-FAST policy')
    return nb, files, blob, assignments, meta


def patch_files(original):
    """Only one function prefix and its exact runtime pin; no broad repinning."""
    files = dict(original)
    before = original[R224].decode()
    need('C85_COMPETITION' not in before, 'already adapted R224 is not the normal parent')
    files[R224] = replace(before, 'def find_test_root():\n', 'def find_test_root():\n' + ADAPTER).encode()
    old_pin, new_pin = digest(original[R224]), digest(files[R224])
    pin_line = f"    p.check_pin(runtime / r224.SOURCE, '{old_pin}')"
    files[CODEC] = replace(original[CODEC], pin_line,
                          f"    p.check_pin(runtime / r224.SOURCE, '{new_pin}')")
    prove_delta(original, files)
    return files


def prove_delta(original, files):
    need(set(files) == set(original) and len(files) == 31, 'exact payload roster')
    changed = {n for n in files if files[n] != original[n]}
    need(changed == {R224, CODEC}, 'only adapter and consequential pin may change')
    restored = replace(files[R224], ADAPTER, '')
    need(restored == original[R224], 'non-adapter R224 changes')
    need(replace(files[CODEC], digest(files[R224]), digest(original[R224])) == original[CODEC],
         'non-pin codec changes')
    # Mechanical function/statement identity outside exactly find_test_root.
    before, after = ast.parse(original[R224]), ast.parse(files[R224])
    need(len(before.body) == len(after.body), 'R224 top-level statements')
    for a, b in zip(before.body, after.body):
        if isinstance(a, ast.FunctionDef) and a.name == 'find_test_root':
            restored_function = copy.deepcopy(b)
            restored_function.body = restored_function.body[2:]
            need(ast.dump(a) == ast.dump(restored_function), 'original discovery behavior changed')
        else:
            need(ast.dump(a) == ast.dump(b), 'R224 execution/model/import change')


def archive(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 18, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            z.writestr(info, data)
    return stream.getvalue(), {n: digest(v) for n, v in sorted(files.items())}


def notebook(normal, original, old_blob, old_assignments, files):
    blob, pins = archive(files)
    source = ''.join(normal['cells'][2]['source'])
    need(source.startswith('if not FAST:\n'), 'ordinary wrapper shape')
    # FAST is sealed False; unwrap that guard so stress cannot take a shortcut.
    source = textwrap.dedent(source.removeprefix('if not FAST:\n'))
    source = replace(source, repr(old_assignments['BLOB']), repr(base64.b64encode(blob).decode()))
    source = replace(source, 'PINS = ' + repr(old_assignments['PINS']), 'PINS = ' + repr(pins))
    source = replace(source, repr(digest(old_blob)), repr(digest(blob)))
    base = copy.deepcopy(normal)
    base['cells'] = [copy.deepcopy(normal['cells'][1]), copy.deepcopy(normal['cells'][2])]
    base['cells'][1]['source'] = source.splitlines(True)
    # Reuse the actual frozen stress entrypoint only, never inherited prepare/package.
    env = extract(original[CODEC].decode(), ['stress_notebook'],
                  dict(notebook_payload=lambda *_: copy.deepcopy(base),
                       VERSION='1.1.49-r6w958c4inf2', need=need))
    result = env['stress_notebook'](blob, pins)
    result['cells'][0]['source'].append('Qualification input adapter v1.0.2: R224 honors the same '
        'C85_COMPETITION as other branches. Same H207 inference; never submit this stress kernel.\n')
    code = ''.join(result['cells'][1]['source'])
    compile(code, '<h207-stress-v102>', 'exec')
    need('FAST' not in code and code.count('stress.prepare(competition, data, n=32)') == 1
         and code.count("release.run_release(runtime, Path('/kaggle/working'))") == 1,
         'full N32 call without FAST')
    return result, blob, pins


def code_for_diff(nb):
    # Full code comparison except blob bytes, whose complete31-file map is separate.
    lines = []
    for cell in nb['cells']:
        for line in cell.get('source', []):
            stripped = line.lstrip()
            if stripped.startswith('BLOB = '):
                value = ast.literal_eval(stripped.split('=', 1)[1].strip())
                line = line[:len(line)-len(stripped)] + 'BLOB_SHA256 = ' + repr(digest(base64.b64decode(value))) + '\n'
            lines.append(line)
    return lines


def render():
    normal, old, old_blob, assignments, metadata = inputs()
    files = patch_files(old)
    nb, blob, pins = notebook(normal, old, old_blob, assignments, files)
    meta = copy.deepcopy(metadata)
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    need({k for k in meta if meta[k] != metadata[k]} == {'id', 'title'}, 'metadata scope')
    full_diff = ''.join(''.join(difflib.unified_diff(old[n].decode().splitlines(True),
        files[n].decode().splitlines(True), fromfile='normal/' + n, tofile='stress-v102/' + n))
        for n in sorted(files) if files[n] != old[n])
    identity = {n: dict(normal_sha256=digest(old[n]), stress_sha256=digest(files[n]),
                        identical=old[n] == files[n],
                        change='input-root adapter' if n == R224 else 'one required runtime pin literal'
                               if n == CODEC else 'none') for n in sorted(files)}
    outputs = {'kernel/release.ipynb': encoded(nb), 'kernel/kernel-metadata.json': encoded(meta),
               'embedded-identity.json': encoded(identity), 'full-embedded-diff.txt': full_diff.encode(),
               'notebook-diff-with-archive-digests.txt': ''.join(difflib.unified_diff(
                   code_for_diff(normal), code_for_diff(nb), fromfile='normal/notebook',
                   tofile='stress-v102/notebook')).encode(),
               'metadata-diff.txt': ''.join(difflib.unified_diff(encoded(metadata).decode().splitlines(True),
                   encoded(meta).decode().splitlines(True), fromfile='normal/metadata',
                   tofile='stress-v102/metadata')).encode()}
    outputs.update({'runtime/' + n: v for n, v in files.items()})
    receipt = dict(version=VERSION, status='BUILT_CPU_ONLY_REQUIRES_INDEPENDENT_REVIEW',
        candidate='R6W958-C4INF2-V1', qualification_only=True, never_submit=True,
        stress_slug=SLUG, normal_stage_pins=NORMAL_PINS, preserved_v1_pins=V1_PINS,
        normal_archive_sha256=digest(old_blob), stress_archive_sha256=digest(blob),
        source_sha256=sha(__file__), tests_sha256=sha(ROOT / 'tools/test_hb207_stress_package_v102.py'),
        shared_helpers={p:sha(ROOT / p) for p in ('tools/c85r7_r6w958inf2_v1148.py','tools/hb145_output_io.py')},
        prompt_sha256=sha(WORK / 'STRESS-INPUT-FIX-PROMPT.md'),
        changed_payloads=[R224, CODEC], identical_payloads=29, all_payloads=31,
        dataset_sources=meta['dataset_sources'], files={p:digest(v) for p,v in outputs.items()},
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        model_forwards=0, real_runtime_measured=False, independent_review='REQUIRED')
    outputs['build-receipt.json'] = encoded(receipt)
    inputs()  # Recheck immutable input evidence before returning a publication payload.
    return outputs


def write_new(path, payload):
    path = Path(path)
    need(not path.exists(), 'create-only output exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    existing.output_io.replace(temp, path)


def build(out=OUT):
    outputs = render()
    out = Path(out)
    need(not any((out / name).exists() for name in outputs), 'immutable package already exists')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'kernel').mkdir(exist_ok=False)  # Exclusive claim before any publication.
    for name, payload in outputs.items():
        write_new(out / name, payload)
    return validate(out)


def validate(out=OUT):
    out = Path(out)
    outputs = render()
    for name, expected in outputs.items():
        need((out / name).read_bytes() == expected, 'stress package drift: ' + name)
    need('torch' not in sys.modules, 'packaging unexpectedly imported torch')
    return dict(status='CPU_PACKAGE_BYTES_VALID_REVIEW_REQUIRED',
                notebook_sha256=digest(outputs['kernel/release.ipynb']),
                metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
                payload_files=31, identical_payload_files=29, dataset_mounts=8,
                normal_stage_unchanged=True, qualification_pass=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'validate'))
    args = parser.parse_args()
    import json
    print(json.dumps(build() if args.command == 'build' else validate()))


if __name__ == '__main__':
    main()
