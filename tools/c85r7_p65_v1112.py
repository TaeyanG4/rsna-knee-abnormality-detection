"""Single-candidate byte patch of the frozen v1.1.0 archive; no serving refactor.

Builder 1.0.0, release 1.1.12-p65. CPU only; never invokes Kaggle or inference.
1.0.0: exact inner-public-weight patch; full runtime/notebook/metadata diff; sealed real3 replay.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0'
OUT = BASE.parent / 'v1.1.12-p65'
SCRATCH = ROOT / '.local/scratch/p65-v1112/build'
BACKUP = ROOT / '.local/backups/p65-v1112/v110'
VISIBLE = ROOT / '.local/scratch/slot-b/knee-exp169-c85r7codec-v1'
VERSION = '1.1.12-p65'
SLUG = 'taeyangg4/knee-lb-p65-v1'
BASE_SHA = '71580a566ee936569dda69411cf5e952d30cce60069bfec432a6f54fb09d86e9'
VISIBLE_SHA = 'e0eb466f3b0d545a3f17142c75c9a6742e3e90c2aba5916267ec7909f3caa3f3'
VISIBLE_INPUTS = {
    'c85r7-run/pr7/pr7-run/r/submission.csv': 'a27282ae3241fc4c1882383259d9ab2df6b2a48cc9412f0d2b6190b2568c1ab9',
    'c85r7-run/pr7/pr7-run/s_r7/controller/submission.csv': 'fa4d89746bef047af6b14866c3272d9818aa2af433dfe33c4fe4ed0593ac5ab3',
}
FORMULA = '0.85*r(P)+0.15*r(O); P=0.35*r(R)+0.65*r(R7); O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46'
BLEND_OLD = '    public = .5 * rr + .5 * ss'
BLEND_NEW = '    public = .35 * rr + .65 * ss'
REF_OLD = '    return .85 * rank(.5 * rank(r) + .5 * rank(ours)) + .15 * rank('
REF_NEW = '    return .85 * rank(.35 * rank(r) + .65 * rank(ours)) + .15 * rank('

def need(ok, message):
    if not ok:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def pin(path, expected):
    need(digest(path.read_bytes()) == expected, 'hash mismatch: ' + str(path))

def write(path, data):
    """Immutable build artifacts: identical reruns allowed; never replace scored bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'refusing to replace existing artifact: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(data)

def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()

def replace(data, old, new):
    if isinstance(old, str):
        old, new = old.encode(), new.encode()
    need(data.count(old) == 1, 'patch anchor must occur exactly once: ' + repr(old[:100]))
    return data.replace(old, new)

def unpack(nb):
    source = ''.join(nb['cells'][-1]['source'])
    assignments = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in ('BLOB', 'PINS')}
    blob = base64.b64decode(assignments['BLOB'])
    need(repr(digest(blob)) in source, 'archive hash not pinned in bootstrap')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        need(len(z.namelist()) == len(set(z.namelist())), 'duplicate archive member')
        files = {name: z.read(name) for name in z.namelist()}
    need({k: digest(v) for k, v in files.items()} == assignments['PINS'], 'embedded pins')
    return files, blob, assignments

def parent():
    pin(BASE / 'build-receipt.json', BASE_SHA)
    rec = read(BASE / 'build-receipt.json')
    for name, key in [('release.ipynb', 'notebook_sha256'), ('kernel-metadata.json', 'metadata_sha256')]:
        pin(BASE / 'kernel' / name, rec[key])
    nb = read(BASE / 'kernel/release.ipynb')
    files, blob, _ = unpack(nb)
    need(digest(blob) == rec['archive_sha256'] and {k: digest(v) for k, v in files.items()} == rec['files'], 'parent runtime identity')
    return nb, files

def allowed_files(original):
    """Exact byte allowlist, not a permissive changed-line regular expression."""
    files = dict(original)
    name = 'exp169_release_c85.py'
    files[name] = replace(files[name], BLEND_OLD, BLEND_NEW)
    files[name] = replace(files[name], 'P=.5*r(R)+.5*r(S6)', 'P=.35*r(R)+.65*r(S6)')
    files[name] = replace(files[name], REF_OLD, REF_NEW)
    name = 'exp169_release_c85r7.py'
    files[name] = replace(files[name], digest(original['exp169_release_c85.py']), digest(files['exp169_release_c85.py']))
    source = files[name].decode()
    formula_line = next(line for line in source.splitlines() if line.startswith('FORMULA = '))
    files[name] = replace(files[name], formula_line, 'FORMULA = ' + json.dumps(FORMULA))
    name = 'exp169_release_c85r7codec.py'
    files[name] = replace(files[name], digest(original['exp169_release_c85r7.py']), digest(files['exp169_release_c85r7.py']))
    files[name] = replace(files[name], 'VERSION = "1.1.0"', 'VERSION = "1.1.12-p65"')
    files[name] = replace(files[name], 'RELEASE = "C85R7+codecs"', 'RELEASE = "C85R7+codecs p65"')
    files[name] = replace(files[name], 'receipt.update(version=VERSION, release=RELEASE, codecs=codec)',
                          'receipt.update(version=VERSION, release=RELEASE, codecs=codec, exposure="EXPOSED", clean_oof=False)')
    return files

def scope_check(original, actual):
    need(set(actual) == set(original), 'runtime filenames changed')
    expected = allowed_files(original)
    for name in sorted(original):
        need(actual[name] == expected[name], 'forbidden runtime diff: ' + name)
    return [name for name in sorted(original) if original[name] != actual[name]]

def payload():
    nb, original = parent()
    files = allowed_files(original)
    _, old_blob, assignments = unpack(nb)
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(old_blob)) as old, zipfile.ZipFile(buf, 'w') as new:
        for info in old.infolist():
            new.writestr(info, files[info.filename])
    blob = buf.getvalue()
    pins = {k: digest(v) for k, v in files.items()}
    source = ''.join(nb['cells'][-1]['source']).encode()
    source = replace(source, 'BLOB = ' + repr(assignments['BLOB']), 'BLOB = ' + repr(base64.b64encode(blob).decode()))
    source = replace(source, 'PINS = ' + repr(assignments['PINS']), 'PINS = ' + repr(pins))
    source = replace(source, repr(digest(old_blob)), repr(digest(blob)))
    nb['cells'][-1]['source'] = source.decode().splitlines(True)
    # Keep every executable bootstrap line, including the original banner, unchanged.
    meta = read(BASE / 'kernel/kernel-metadata.json')
    meta.update(id=SLUG, title=SLUG.split('/')[1])
    return nb, meta, files

def full_diff(original, files):
    changed = scope_check(original, files)
    result = ['ALL 29 EMBEDDED FILES; exact byte allowlist PASS\n']
    for name in sorted(original):
        result.append(f'FILE {name}: {digest(original[name])} -> {digest(files[name])}' + (' IDENTICAL\n' if name not in changed else ' ALLOWED_CHANGE\n'))
        if name in changed:
            result.extend(difflib.unified_diff(original[name].decode().splitlines(True), files[name].decode().splitlines(True), fromfile='v1.1.0/' + name, tofile=VERSION + '/' + name))
    parent_nb = read(BASE / 'kernel/release.ipynb')
    candidate_nb, candidate_meta, _ = payload()
    need(len(parent_nb['cells']) == len(candidate_nb['cells']), 'cell count changed')
    pairs = [('notebook-top-level', {k:v for k,v in parent_nb.items() if k != 'cells'},
              {k:v for k,v in candidate_nb.items() if k != 'cells'})]
    pairs += [(f'notebook-cell-{i}', a, b) for i, (a,b) in enumerate(zip(parent_nb['cells'], candidate_nb['cells']))]
    pairs += [('kernel-metadata.json', read(BASE / 'kernel/kernel-metadata.json'), candidate_meta)]
    for name, before, after in pairs:
        left, right = encoded(before), encoded(after)
        result.append(f'FILE {name}: {digest(left)} -> {digest(right)}' + (' IDENTICAL\n' if left == right else ' ALLOWED_CHANGE\n'))
        result.extend(difflib.unified_diff(left.decode().splitlines(True), right.decode().splitlines(True),
                      fromfile='v1.1.0/' + name, tofile=VERSION + '/' + name))
    return ''.join(result).encode()

def prepare():
    nb, meta, files = payload()
    _, original = parent()
    for kind, contents in [('parent-runtime', original), ('candidate-runtime', files)]:
        for name, data in contents.items():
            write(SCRATCH / kind / name, data)
    for name in ('build-receipt.json', 'registration.json', 'kernel/release.ipynb', 'kernel/kernel-metadata.json'):
        write(BACKUP / name, (BASE / name).read_bytes())
    write(OUT / 'kernel/release.ipynb', encoded(nb))
    write(OUT / 'kernel/kernel-metadata.json', encoded(meta))
    write(OUT / 'full-diff-vs-v110.txt', full_diff(original, files))
    write(OUT / 'registration.json', encoded(dict(version=VERSION, kernel_id=SLUG, formula=FORMULA,
        parent=str(BASE.relative_to(ROOT)), parent_receipt_sha256=BASE_SHA, registered_change='User instruction: inner P weights 0.35 R / 0.65 R7; outer 0.85/0.15 and O unchanged',
        exposure='EXPOSED', clean_oof=False, qualified_for_submission=False,
        failure_rule='Parent survivor rules unchanged: P failed => O; O failed => P; both failed => 0.5; inner member failures renormalize as v1.1.0.',
        files={k: digest(v) for k, v in files.items()})))
    write(OUT / 'CHANGELOG.md', ('# 1.1.12-p65\nOne candidate: v1.1.0 runtime with only registered inner-public-weight blend arithmetic, receipt/version metadata and dependent pins changed.\nFrozen parents untouched; backup: ' + str(BACKUP.relative_to(ROOT)).replace('\\', '/') + '.\n').encode())
    write(OUT / 'build-receipt.json', encoded(dict(status='PREPARED_CPU_ONLY', version=VERSION,
        notebook_sha256=digest(encoded(nb)), metadata_sha256=digest(encoded(meta)),
        archive_sha256=digest(unpack(nb)[1]), changed_files=scope_check(original, files), unchanged_files=26,
        builder_sha256=digest(Path(__file__).read_bytes()), qualified_for_submission=False)))
    return validate()

def validate():
    nb, meta, expected = payload()
    actual = read(OUT / 'kernel/release.ipynb')
    files, _, _ = unpack(actual)
    _, original = parent()
    changed = scope_check(original, files)
    need(actual == nb, 'notebook/bootstrap drift')
    need(read(OUT / 'kernel/kernel-metadata.json') == meta, 'metadata drift beyond id/title')
    need(meta['id'] == SLUG and meta['title'] == SLUG.split('/')[1] and '/' not in meta['title'], 'kernel identity/title')
    need((OUT / 'full-diff-vs-v110.txt').read_bytes() == full_diff(original, files), 'full diff drift')
    need(read(OUT / 'registration.json')['files'] == {k: digest(v) for k, v in expected.items()}, 'registration pins')
    rec = read(OUT / 'build-receipt.json')
    need(rec['status'] == 'PREPARED_CPU_ONLY' and rec['version'] == VERSION and
         rec['archive_sha256'] == digest(unpack(actual)[1]) and
         rec['changed_files'] == changed and rec['unchanged_files'] == 26 and
         rec['qualified_for_submission'] is False, 'receipt identity')
    reg = read(OUT / 'registration.json')
    need(reg['version'] == VERSION and reg['kernel_id'] == SLUG and reg['formula'] == FORMULA and
         reg['parent_receipt_sha256'] == BASE_SHA and
         reg['exposure'] == 'EXPOSED' and reg['clean_oof'] is False and
         reg['qualified_for_submission'] is False, 'registration identity/exposure')
    pin(OUT / 'kernel/release.ipynb', rec['notebook_sha256'])
    pin(OUT / 'kernel/kernel-metadata.json', rec['metadata_sha256'])
    pin(Path(__file__), rec['builder_sha256'])
    return dict(status='PASS', files=29, identical_files=26, changed_files=changed, bootstrap_identical_except_archive_and_pins=True)

def math_functions(files):
    # Execute only the arithmetic functions, with frozen runtime dependencies supplied below.
    tree = ast.parse(files['exp169_release_c85.py'])
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('rank', 'scored_rank', 'blend', 'reference')]
    from tools import exp169_release_c85 as c85
    ns = dict(need=c85.need, s6=c85.s6, v5q=c85.v5q, MEMBERS=c85.MEMBERS)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<frozen-arithmetic>', 'exec'), ns)
    return ns

def recover_own_rank(final, public_rank):
    """For n=3 average ranks are multiples of 1/6; float32 error is <1e-6.

    The 0.15 own coefficient separates adjacent possibilities by 0.025,
    making inversion unique. These are sufficient statistics, not raw scores.
    """
    import numpy as np
    import pandas as pd
    need(final.shape == public_rank.shape == (3, 12), 'complete real3 matrices required')
    own = np.rint(((final - .85 * public_rank) / .15) * 6) / 6
    need(np.isfinite(own).all() and ((own >= 1/3) & (own <= 1)).all(), 'invalid recovered own rank')
    need(np.max(np.abs(.85 * public_rank + .15 * own - final)) < 1e-6, 'own rank inversion failed')
    need(np.array_equal(pd.DataFrame(own).rank(method='average', pct=True).to_numpy(), own), 'invalid rank lattice')
    return own


def replay():
    import numpy as np
    import pandas as pd
    from tools import exp169_release_c85r7codec as parent_runtime
    validate()
    pin(VISIBLE / 'c85r7-receipt.json', VISIBLE_SHA)
    rec = read(VISIBLE / 'c85r7-receipt.json')
    need(rec['version'] == '1.1.0' and rec['status'] == 'COMPLETE_C85R7' and rec['studies'] == 3, 'visible parent completion')
    pin(VISIBLE / 'submission.csv', rec['submission_sha256'])
    need(all(v == 0 for v in rec['fallbacks'].values()), 'visible fallbacks')
    ids, final = parent_runtime.v5q._read_csv(VISIBLE / 'submission.csv')
    def aligned(name):
        other, values = parent_runtime.v5q._read_csv(VISIBLE / name)
        need(len(ids) == len(other) == 3 and set(ids) == set(other), 'real visible cohort')
        return values[[other.index(uid) for uid in ids]]
    for name, expected in VISIBLE_INPUTS.items():
        pin(VISIBLE / name, expected)
    r, s = [aligned(name) for name in VISIBLE_INPUTS]
    pin(VISIBLE / 'p-branch-r7.csv', rec['telemetry']['p_branch']['sha256'])
    def rank(v):
        return pd.DataFrame(v).rank(method='average', pct=True).to_numpy()
    public = (rank(r) + rank(s)) / 2
    need(parent_runtime.p.csv_bytes(ids, public) == (VISIBLE / 'p-branch-r7.csv').read_bytes(), 'authenticated public branch byte parity')
    # Missing own raw journals are not fabricated: recover only rank(O), which
    # is all the final formula uses. Equal member surrogates preserve that rank.
    own_rank = recover_own_rank(final, rank(public))
    x = y = q = own_rank
    _, original = parent()
    candidate = unpack(read(OUT / 'kernel/release.ipynb'))[0]
    old, new = math_functions(original), math_functions(candidate)
    baseline = old['blend'](ids, r, s, x, y, q)
    need(parent_runtime.p.csv_bytes(ids, baseline) == (VISIBLE / 'submission.csv').read_bytes(), 'real v110 replay byte parity')
    result = new['blend'](ids, r, s, x, y, q)
    reference = .85 * rank(.35 * rank(r) + .65 * rank(s)) + .15 * own_rank
    np.testing.assert_array_equal(result, reference)
    np.testing.assert_array_equal(result, new['reference'](r, s, x, y, q))
    output = parent_runtime.p.csv_bytes(ids, result)
    write(SCRATCH / 'replay.csv', output)
    report = dict(status='PASS', studies=3, targets=12, baseline_receipt_sha256=VISIBLE_SHA,
        baseline_csv_sha256=rec['submission_sha256'], candidate_csv_sha256=digest(output),
        baseline_reproduced_byte_for_byte=True, independent_pandas_exact=True, outer_and_own_unchanged=True,
        changed_cells=int(np.count_nonzero(result != baseline)),
        diagnostic='Original public cell-7 P diagnostic remains 0.5/0.5 by design; final blend uses p65.',
        input_sha256=VISIBLE_INPUTS, public_branch_reproduced_byte_for_byte=True,
        own_recovery='Unique rank(O) inversion on n=3 average-rank lattice; rank-equivalent member surrogates, not original member probabilities; full baseline CSV byte parity.',
        limitation='Downloaded run lacks own raw journals; proves final formula on real visible sufficient statistics, not neural inference or original member replay.',
        exposure='EXPOSED', clean_oof=False, scope='authenticated completed v1.1.0 visible predictions; CPU arithmetic only; no neural inference')
    write(SCRATCH / 'replay.json', encoded(report))
    return report

def admit():
    validate()
    replay_result = read(SCRATCH / 'replay.json')
    need(replay_result['status'] == 'PASS', 'CPU replay required')
    # This builder does not author or waive the mandatory different-lane review.
    return dict(status='HOLD_INDEPENDENT_REVIEW', qualified_for_submission=False,
                notebook_sha256=digest((OUT / 'kernel/release.ipynb').read_bytes()),
                smoke_sha256=digest((SCRATCH / 'replay.json').read_bytes()))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'register', 'cpu-replay', 'admit'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(ROOT))
    if args.command == 'admit':
        need(args.dry_run, 'admission is dry-run only; independent review required')
    action = {'prepare': prepare, 'validate': validate, 'register': validate, 'cpu-replay': replay, 'admit': admit}[args.command]
    print(json.dumps(action(), indent=2))

if __name__ == '__main__':
    main()
