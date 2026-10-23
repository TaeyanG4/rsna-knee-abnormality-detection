"""INF3 diagnostic-only full N32 builder v1.0.0; never launches or submits."""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r6w958inf3_v1146 as science
from tools import strict_variant_gate_inf3_v103 as prior

VERSION = '1.0.0'
BUILDER = '/root/k1_result_review'
WORK = ROOT / '.local/scratch/slot-b/batch5-1018/inf3-qualification'
OUT = WORK / 'diagnostic-kernel'
SLUG = 'taeyangg4/knee-lb-r6w958inf3-n32-v1'
BUILD = WORK.parent / 'inf3-release/fix1/build-admission.json'
KERNEL = 'baec47351a8b7823a49bd6013e95b9b5a86721f31e51efa5064bfc3c3bb0d150'
METADATA = 'e423666f06da0a5176b07a8dccfd454a292e9b5725d3289e9e514d0dcde7b257'
ENV = 'H197_INF3_N32_ROOT'
SOURCE_FILES = ('tools/hb197_inf3_qualification_v1.py', 'tools/hb197_inf3_observer_v1.py',
                'tools/hb197_inf3_collect_v1.py', 'tools/strict_variant_gate_inf3_qualification_v1.py',
                'tests/test_hb197_inf3_qualification_v1.py',
                '.local/scratch/slot-b/batch5-1018/inf3-qualification/SPEC.md',
                '.local/scratch/slot-b/batch5-1018/inf3-qualification/B-RUNBOOK.md',
                '.local/scratch/slot-b/batch5-1018/inf3-qualification/REVIEW-TEMPLATE.md')
need, sha, encoded = science.need, science.sha, science.encoded


def write(path, data):
    path = Path(path).resolve()
    need(path.is_relative_to(WORK.resolve()), 'qualification output scope')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'immutable qualification artifact differs: ' + path.name)
        return
    for attempt in range(40):
        try:
            with path.open('xb') as stream: stream.write(data)
            return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(min(.05 * (attempt + 1), 1))

ROOT_PREFIX = '''    # Diagnostic only: absent override preserves the original body verbatim.
    if "H197_INF3_N32_ROOT" in os.environ:
        import csv as _csv
        import hashlib as _hashlib
        from pathlib import Path as _Path
        _raw = os.environ["H197_INF3_N32_ROOT"]
        if not _raw or not _Path(_raw).is_absolute():
            raise RuntimeError("invalid INF3 diagnostic root")
        _root = _Path(_raw).resolve()
        if os.environ.get("C85_COMPETITION") != str(_root):
            raise RuntimeError("INF3 branch root mismatch")
        _marker = json.loads((_root / "h197-cohort.json").read_text())
        if _marker.get("schema") != "h197-n32-cohort/v1" or _marker.get("studies") != 32:
            raise RuntimeError("invalid INF3 cohort marker")
        _tables = {}
        for _name in ("test.csv", "test_series.csv", "sample_submission.csv", "selection.json"):
            _bytes = (_root / _name).read_bytes()
            if _hashlib.sha256(_bytes).hexdigest() != _marker["sha256"][_name]:
                raise RuntimeError("stale INF3 cohort metadata")
            if _name.endswith(".csv"):
                _tables[_name] = list(_csv.DictReader(_bytes.decode("utf-8-sig").splitlines()))
        _ids = [r["StudyInstanceUID"] for r in _tables["test.csv"]]
        _selected = json.loads((_root / "selection.json").read_text())
        if (len(_ids) != 32 or len(set(_ids)) != 32 or _ids != _marker["ids"] or
            _selected.get("n") != 32 or [r["uid"] for r in _selected["selection"]] != _ids or
            [r["StudyInstanceUID"] for r in _tables["sample_submission.csv"]] != _ids or
            {r["StudyInstanceUID"] for r in _tables["test_series.csv"]} != set(_ids)):
            raise RuntimeError("INF3 same32 cohort mismatch")
        if not all(u and _Path(u).name == u and u not in (".", "..") and
                   (_root / "test_series" / u).is_dir() for u in _ids):
            raise RuntimeError("INF3 missing study directory")
        return str(_root)
'''


def sources():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def prerequisite():
    result = prior.verify_receipt(prior.ref(BUILD), phase='BUILD_PASS')
    need(result['kernel_sha256'] == KERNEL and result['qualified_for_submission'] is False,
         'exact independently reviewed scientific BUILD')
    need(sha(science.OUT / 'kernel/release.ipynb') == KERNEL and
         sha(science.OUT / 'kernel/kernel-metadata.json') == METADATA, 'sealed release drift')
    return result


def root_override(data):
    text = data.decode('utf-8')
    tree = ast.parse(text)
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'find_test_root']
    need(len(node) == 1 and text.count('def find_test_root():\n') == 1, 'unique original root function')
    changed = text.replace('def find_test_root():\n', 'def find_test_root():\n' + ROOT_PREFIX, 1)
    other = ast.parse(changed)
    for a, c in zip(tree.body, other.body):
        if a is not node[0]:
            need(ast.dump(a, include_attributes=False) == ast.dump(c, include_attributes=False), 'outside root function')
    need(changed.replace(ROOT_PREFIX, '', 1) == text, 'ordinary root body changed')
    return changed.encode('utf-8')


def payload():
    nb = science.read(science.OUT / 'kernel/release.ipynb')
    original, old_blob, assignments = science.unpack(nb)
    files = dict(original)
    files['raptor224_run.py'] = root_override(original['raptor224_run.py'])
    old_digest = science.digest(original['raptor224_run.py'])
    new_digest = science.digest(files['raptor224_run.py'])
    needle = "p.check_pin(runtime / r224.SOURCE, '" + old_digest + "')"
    code = files['exp169_release_c85r7codec.py'].decode()
    need(code.count(needle) == 1, 'exact direct R224 source pin')
    files['exp169_release_c85r7codec.py'] = code.replace(needle, needle.replace(old_digest, new_digest)).encode()
    changed = {n for n in files if files[n] != original[n]}
    need(changed == {'raptor224_run.py', 'exp169_release_c85r7codec.py'} and len(files) == 31, 'minimal diagnostic payload')
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as src, zipfile.ZipFile(buffer, 'w') as dst:
        for info in src.infolist():
            dst.writestr(info, files[info.filename])
    blob = buffer.getvalue()
    result = copy.deepcopy(nb)
    observer = (ROOT / 'tools/hb197_inf3_observer_v1.py').read_bytes()
    setup = "\n".join([
        'FAST = False', 'import time as _h197_time', '_h197_start = _h197_time.monotonic()',
        'import base64 as _h197_b64, hashlib as _h197_hash, os as _h197_os',
        'from pathlib import Path as _h197_Path',
        f'_h197_observer = _h197_b64.b64decode({base64.b64encode(observer).decode()!r})',
        f'assert _h197_hash.sha256(_h197_observer).hexdigest() == {science.digest(observer)!r}',
        "_h197_file = _h197_Path('/kaggle/working/hb197_inf3_observer_v1.py')",
        "with _h197_file.open('xb') as _h197_stream: _h197_stream.write(_h197_observer)",
        'import hb197_inf3_observer_v1 as _h197',
        "_h197_watch = _h197.Observer('/kaggle/working', _h197_start)", '_h197_watch.start()', ''])
    result['cells'][0]['source'] = setup.splitlines(True)
    result['cells'][1]['source'] = ['# INF3 full-path same32 diagnostic. NEVER SUBMIT.\n',
        'One diagnostic root override plus its pin; complete release inference unchanged. Separate observer and synthetic global-cost benchmark.\n']
    code = ''.join(nb['cells'][2]['source'])
    code = science.replace(code, repr(assignments['BLOB']), repr(base64.b64encode(blob).decode()))
    code = science.replace(code, 'PINS = ' + repr(assignments['PINS']), 'PINS = ' + repr({n: science.digest(v) for n, v in files.items()}))
    code = science.replace(code, repr(science.digest(old_blob)), repr(science.digest(blob)))
    anchor = "    release.run_release(runtime, Path('/kaggle/working'))\n"
    replacement = "\n".join([
        '    import exp169_c85_stress as _h197_stress',
        "    _h197_comp = release.v5q._mount(release.p.COMPETITION, 'train_series.csv')",
        "    _h197_data = Path('/kaggle/working/inf3-n32-data')",
        '    _h197_ids = _h197_stress.prepare(_h197_comp, _h197_data, n=32)',
        '    _h197.seal_cohort(_h197_data, _h197_ids)',
        "    _h197_os.environ['C85_COMPETITION'] = str(_h197_data)",
        f'    _h197_os.environ[{ENV!r}] = str(_h197_data)',
        '    try:', "        release.run_release(runtime, Path('/kaggle/working'))",
        '    finally:', '        _h197_watch.finish()',
        "    _h197.benchmark(release, runtime, Path('/kaggle/working'), hidden=1300)", ''])
    code = science.replace(code, anchor, replacement)
    result['cells'][2]['source'] = code.splitlines(True)
    for cell in result['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), '<INF3-N32-diagnostic>', 'exec')
    metadata = science.read(science.OUT / 'kernel/kernel-metadata.json')
    metadata.update(id=SLUG, title=SLUG.split('/')[1])
    proof = dict(schema='h197-diagnostic-diff/v1', candidate=science.KEY, diagnostic_slug=SLUG,
        release_kernel_sha256=KERNEL, release_metadata_sha256=METADATA, members=31, unchanged=29,
        changed=sorted(changed), original={n: science.digest(v) for n, v in original.items()},
        diagnostic={n: science.digest(v) for n, v in files.items()}, observer_sha256=science.digest(observer),
        root_change='one function prefix only; original body verbatim when override absent', never_submit=True)
    diff = ''.join(''.join(difflib.unified_diff(original[n].decode().splitlines(True), files[n].decode().splitlines(True),
                    fromfile='release/' + n, tofile='diagnostic/' + n)) for n in original if files[n] != original[n])
    notebook_diff = ''.join(difflib.unified_diff(science.encoded(nb).decode().splitlines(True),
                        science.encoded(result).decode().splitlines(True), fromfile='release/release.ipynb', tofile='diagnostic/release.ipynb'))
    return result, metadata, files, proof, diff, notebook_diff


def prepare():
    parent = prerequisite()
    nb, metadata, files, proof, diff, ndiff = payload()
    for name, data in [('diagnostic-kernel/release.ipynb', encoded(nb)), ('diagnostic-kernel/kernel-metadata.json', encoded(metadata)),
                       ('EMBEDDED-DIFF.json', encoded(proof)), ('FULL-EMBEDDED.diff', diff.encode()),
                       ('FULL-NOTEBOOK.diff', ndiff.encode())]:
        write(WORK / name, data)
    for name, data in files.items(): write(WORK / 'diagnostic-runtime' / name, data)
    return validate()


def validate():
    prerequisite()
    return construction()


def construction():
    """Deterministic source/artifact check; caller separately replays prerequisite."""
    nb, metadata, files, proof, diff, ndiff = payload()
    expected = {'diagnostic-kernel/release.ipynb': encoded(nb), 'diagnostic-kernel/kernel-metadata.json': encoded(metadata),
                'EMBEDDED-DIFF.json': encoded(proof), 'FULL-EMBEDDED.diff': diff.encode(), 'FULL-NOTEBOOK.diff': ndiff.encode()}
    for name, data in expected.items(): need((WORK / name).read_bytes() == data, 'diagnostic artifact drift: ' + name)
    for name, data in files.items(): need((WORK / 'diagnostic-runtime' / name).read_bytes() == data, 'diagnostic payload drift: ' + name)
    need(set(p.relative_to(WORK / 'diagnostic-runtime').as_posix() for p in (WORK / 'diagnostic-runtime').rglob('*') if p.is_file()) == set(files), 'extra diagnostic runtime file')
    return dict(schema='h197-qualification-build/v1', status='BUILT_NOT_LAUNCH_ADMITTED', version=VERSION,
        candidate=science.KEY, builder=BUILDER, release_kernel_sha256=KERNEL, diagnostic_slug=SLUG,
        diagnostic_kernel_sha256=sha(OUT / 'release.ipynb'), diagnostic_metadata_sha256=sha(OUT / 'kernel-metadata.json'),
        embedded_members=31, unchanged=29, changed=proof['changed'], source_sha256=sources(), qualified_for_submission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=['prepare', 'validate'])
    a = p.parse_args(); result = prepare() if a.command == 'prepare' else validate()
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__': raise SystemExit(main())
