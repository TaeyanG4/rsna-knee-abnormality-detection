"""Strict variant contract v1.4.0: C85-R7 R224 only; static, CPU-only.

Changelog 1.4.0: separate CLI preserves v1.2/v1.3 tool/receipt bytes.
PIN is a review proposal, never an approval. Numerical changes require a
dispatcher-authenticated independent scope review bound to PIN and full diff.
"""
from __future__ import annotations

import argparse
import ast
import copy
import os
from pathlib import Path
import re
import sys
import time
import zipfile

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import strict_variant_gate as g
from tools import strict_variant_gate_standalone as s

VERSION = '1.4.0'
SLUG = 'taeyangg4/knee-lb-r224q-v1'
PARENT = g.RELEASES + 'v1.1.0'
STAGE = g.RELEASES + 'v1.1.23-r224q'
CANONICAL_STAGE = STAGE  # No CLI override can turn a test path into admission.
PARENT_PINS = {
    'kernel-metadata.json': 'ee4920c4679f7c3ca5244cb62acb0139210a2563d69de8c19917dbacf49a1663',
    'release.ipynb': '71c3a8f02e42e8fc7b3b734a10a33c12b700fac3e48d540b4802a5aaceb26eba',
}
ORIGINAL = '.local/scratch/hb164-r224/hb164/nartaa-original/rsna-knee-0945-efficient-224crop.ipynb'
ORIGINAL_SHA = 'b80d3b72a1c34f2b39b3ea0d2952f039b8d27c3feb5c4db0a7d0ab62ae75354e'
SOURCE_SHA = 'e85d994fa93caf1bbc2df04f396649207a69d2463924c41fa3b05ba8801cd38c'
CHECKPOINT_SHA = '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'
WRAPPER = 'exp169_release_c85r7codec.py'
RAW_EDIT = ('    _raw = pd.DataFrame(arm_probs[0].astype(np.float32), columns=LAB)   # [blend] raw probabilities; a failed study stays 0.5\n'
            '    _raw.insert(0, "StudyInstanceUID", test_ids)\n'
            '    _raw.to_csv("/kaggle/working/raptor224_raw.csv", index=False)\n')
OUT_BEFORE = '    out = "/kaggle/working/submission.csv"'
OUT_AFTER = '    out = "/kaggle/working/raptor224_sub.csv"   # [blend] Raptor-224 stage output; the blend cell writes submission.csv'
RAW_ANCHOR = '    for d in devs:\n        for m, _ in models[d]: del m'
SCOPE = 'R=.65*rank(R384)+.35*rank(R224); independent failures; survivor renormalization; own branch unchanged; no existing launch/import/pool/thread/timing change'


def original_parts(inputs):
    raw = inputs.read(inputs.root / ORIGINAL)
    g.need(g.sha(raw) == ORIGINAL_SHA, 'original R224 notebook pin differs')
    nb = g.parse(raw)
    check, source = (''.join(nb['cells'][i]['source']) for i in (3, 4))
    g.need(g.sha(source.encode()) == SOURCE_SHA and CHECKPOINT_SHA in check, 'original source/checkpoint pin differs')
    edited = s.replace(source, OUT_BEFORE, OUT_AFTER, 1, g)
    edited = s.replace(edited, RAW_ANCHOR, RAW_EDIT + RAW_ANCHOR, 1, g)
    return check.encode(), edited.encode()


def added_source(files, inputs):
    """One directly executable checkpoint+scored-source member; no wrappers."""
    check, code = original_parts(inputs)
    pairs = {(check + sep + code) for sep in (b'\n', b'\n\n', b'\n# %% [code]\n', b'\n\n# %% [code]\n')}
    g.need(len(files) == 1 and next(iter(files.values())) in pairs,
           'added members must reconstruct original checkpoint cell + scored source + exactly two edits')


def nodes(data):
    text = data.decode()
    lines = text.splitlines(True)
    result = {}
    other_index = 0
    for n in ast.parse(text).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            key = 'def:' + n.name
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            key = 'assign:' + n.targets[0].id
        else:
            key = 'other:' + str(other_index)
            other_index += 1
        g.need(key not in result, 'duplicate top-level definition')
        result[key] = (n, ''.join(lines[n.lineno-1:n.end_lineno]))
    return result


def invocation(name):
    # Direct script subprocess, identical runner/environment style to R384.
    g.safe_name(name)
    return ('    env224 = codecs.child_env(dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1"), codec,\n'
            '                              branch="R224", telemetry_dir=tele)\n'
            '    c85.phase(output, "R224", lambda: v5q.run_process_group([sys.executable, "-u", str(runtime / ' + repr(name) + ')],\n'
            '        label="C85R7 R224", env=env224))\n')


def wrapper_scope(before, after, added):
    """Lock every byte except numerical collection, explicit metadata and splice.

    Formula bodies are exact PIN-bound review material, not auto-approved math.
    No import or orchestration code can be hidden inside those bodies.
    """
    a, b = nodes(before), nodes(after)
    allowed = {'def:collect', 'def:candidate_receipt', 'assign:VERSION', 'assign:OUT',
               'assign:KERNEL_ID', 'assign:RELEASE', 'assign:FORMULA', 'assign:MEMBERS'}
    g.need(set(a) <= set(b), 'existing wrapper definition removed/renamed')
    new_functions = set(b) - set(a)
    g.need(all(k.startswith('def:r224_') for k in new_functions), 'only r224_ numerical helpers may be added')
    allowed |= new_functions
    # Exactly one direct invocation is inserted before the unchanged S closure.
    old_run = a['def:run_public'][1]
    candidates = [old_run.replace('    def infer_s():\n', invocation(n) + '    def infer_s():\n') for n in added]
    g.need(b['def:run_public'][1] in candidates and old_run.count('    def infer_s():\n') == 1,
           'existing launch changed or new R224 invocation differs from exact R384-style template')
    replacements = [(a['def:run_public'][1], b['def:run_public'][1])]
    for key, (node, content) in b.items():
        if key in allowed:
            if key.startswith('assign:'):
                # Metadata expressions may reference existing constants, but cannot call code.
                g.need(not any(isinstance(n, (ast.Call, ast.Lambda, ast.NamedExpr)) for n in ast.walk(node)), 'executable metadata change')
            else:
                banned = {'__import__', 'eval', 'exec', 'compile', 'open', 'getattr', 'setattr', 'globals', 'locals',
                          'run_process_group', 'phase', 'system', 'popen', 'sleep', 'Thread', 'Pool', 'Process'}
                for n in ast.walk(node):
                    if isinstance(n, (ast.Import, ast.ImportFrom)):
                        g.need(isinstance(n, ast.Import) and len(n.names) == 1 and n.names[0].name == 'numpy', 'new non-numerical import')
                    if isinstance(n, ast.Call):
                        target = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ''
                        g.need(target and target not in banned, 'orchestration/dynamic call in numerical delta')
                g.need(not any(isinstance(n, ast.Name) and n.id in {'os','sys','subprocess','threading','multiprocessing','time','importlib'} for n in ast.walk(node)), 'orchestration dependency in numerical delta')
            if key in a:
                replacements.append((a[key][1], content))
        elif key != 'def:run_public':
            g.need(key in a and a[key][1] == content, 'unregistered wrapper definition change: ' + key)
    # Outside AST spans (including comments and spacing), preserve every byte.
    stripped = after.decode()
    for old, new in reversed(replacements):
        g.need(stripped.count(new) == 1, 'ambiguous wrapper replacement')
        stripped = stripped.replace(new, old, 1)
    helpers = []
    for key in new_functions:
        content = b[key][1]
        g.need(stripped.count(content) == 1, 'ambiguous numerical helper')
        helpers.append((stripped.index(content), content))
    parts, cursor = [], 0
    for start, content in sorted(helpers):
        parts.append(stripped[cursor:start])
        cursor = start + len(content)
    parts.append(stripped[cursor:])
    parent = before.decode()
    offsets = {len(parts[0])} if parent.startswith(parts[0]) else set()
    for part in parts[1:]:
        # Only newly inserted helpers may carry up to two LF separators.
        # All original whitespace, comments and unrelated bytes stay exact.
        separators = min(2, len(part) - len(part.lstrip('\n')))
        offsets = {offset + len(part[skip:]) for offset in offsets
                   for skip in range(separators + 1)
                   if parent.startswith(part[skip:], offset)}
    g.need(len(parent) in offsets, 'wrapper bytes outside registered spans changed')
    g.need(ast.literal_eval(b['assign:VERSION'][0].value) == '1.1.23-r224q' and
           ast.literal_eval(b['assign:KERNEL_ID'][0].value) == SLUG, 'candidate version/slug metadata differs')


def sidecar_check(record, old, new):
    g.need(isinstance(record, dict) and record.get('placeholder', False) is False, 'placeholder new-members')
    def rows(key):
        values = record.get(key)
        g.need(isinstance(values, list), 'new-members requires added and changed lists')
        result = {}
        for row in values:
            name = g.safe_name(row['name'])
            g.need(name not in result, 'duplicate new-members name')
            result[name] = row
        return result
    added, changed = rows('added'), rows('changed')
    g.need(set(added) == new.keys()-old.keys(), 'new-members added set mismatch')
    g.need(set(changed) == {n for n in old if old[n] != new.get(n)}, 'new-members changed set mismatch')
    for name, row in added.items():
        origin = g.encoded(row.get('origin')).decode()
        g.need(row.get('sha256') == g.sha(new[name]) and
               (SOURCE_SHA in origin or ORIGINAL_SHA in origin) and
               all(s in origin for s in ('raptor224_sub.csv', 'raptor224_raw.csv')),
               'new member full hash/original source/two edits provenance missing')
    for name, row in changed.items():
        g.need(row.get('before_sha256') == g.sha(old[name]) and row.get('after_sha256') == g.sha(new[name])
               and isinstance(row.get('reason'), str) and row['reason'].strip(), 'changed member hash/reason mismatch')


def material(stage, receipt, inputs):
    # Authenticate completion BEFORE reading/listing any candidate output.
    lane, report, _ = g.dispatch(receipt, inputs)
    g.need(stage == inputs.root / STAGE, 'R224 source stage must be exact registered path')
    old = g.kernel(inputs.root / PARENT, inputs)
    g.need(old[2]['kernel_files'] == PARENT_PINS and len(old[2]['embedded']) == 29, 'C85-R7 parent pin differs')
    new = g.kernel(stage, inputs)
    g.need(new[0]['id'] == SLUG and set(new[3]) == set(PARENT_PINS), 'R224 slug/kernel file set differs')
    digest = g.sha(g.encoded(new[2]['kernel_files']))
    g.need(re.findall(r'^Kernel-SHA256: (' + g.HEX + r')\s*$', report.decode('utf-8-sig'), re.M) == [digest], 'completed builder report does not bind this kernel')
    meta = copy.deepcopy(old[0]); meta.update(id=SLUG, title=SLUG.split('/')[1])
    g.need(g.encoded(new[0]) == g.encoded(meta), 'metadata exceeds slug/title change')
    normalized = g.fast_inventory(old, new, g.FAST_REFERENCE, inputs)
    before = {n[9:]:v for n,v in old[1].items() if n.startswith('embedded/')}
    after = {n[9:]:v for n,v in normalized.items() if n.startswith('embedded/')}
    g.need(before.keys() <= after.keys(), 'parent member removed/renamed')
    added = {n:after[n] for n in after.keys()-before.keys()}
    added_source(added, inputs)
    changed = {n for n in before if before[n] != after[n]}
    g.need(changed == {WRAPPER}, 'existing members changed beyond R224 wrapper')
    wrapper_scope(before[WRAPPER], after[WRAPPER], added)
    sidecar_check(inputs.json(stage / 'new-members.json'), before, after)
    # Full inventory and raw notebook diff, including explicit added-member sides.
    rules, diff, a, b = s.comparison(old[1], (new[0], normalized, *new[2:]), old[3], g)
    return old, new, lane, rules, diff, a, b


def pin_payload(stage, receipt, inputs):
    for tool in (g.__file__, s.__file__, __file__): inputs.read(Path(tool))
    old, new, lane, rules, diff, _, _ = material(stage, receipt, inputs)
    return dict(version=VERSION, status='REVIEW_REQUIRED', fixture_only=STAGE != CANONICAL_STAGE, kernel_id=SLUG,
        parent=PARENT, parent_kernel_files=PARENT_PINS, source_stage=STAGE,
        builder_receipt=receipt.relative_to(inputs.root).as_posix(), builder_lane=lane,
        builder_receipt_sha256=g.sha(inputs.read(receipt)), kernel_files=new[2]['kernel_files'],
        new_members_sha256=g.sha(inputs.read(stage / 'new-members.json')),
        added={n:h for n,h in new[2]['embedded'].items() if n not in old[2]['embedded']},
        files=rules, diff_sha256=g.sha(diff.encode()), original_sha256=ORIGINAL_SHA,
        source_sha256=SOURCE_SHA, checkpoint_sha256=CHECKPOINT_SHA, scope=SCOPE,
        fast_path=dict(g.FAST_REFERENCE),
        tools={Path(tool).name:g.sha(inputs.read(Path(tool))) for tool in (g.__file__,s.__file__,__file__)}), (old, new, lane, rules, diff)


def write_pin(path, payload):
    """Create-once proposal, including explicitly marked test proposals."""
    g.need(payload.get('status') == 'REVIEW_REQUIRED', 'PIN cannot publish approval')
    data = g.encoded(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        for attempt in range(40):
            try:
                if path.exists():
                    g.need(path.read_bytes() == data, 'immutable PIN differs; use a new proposal path')
                else:
                    os.link(temp, path)
                break
            except FileExistsError:
                g.need(path.read_bytes() == data, 'PIN race with different inputs')
                break
            except PermissionError:
                if attempt == 39: raise
                time.sleep(.05)
    finally:
        for attempt in range(40):
            try:
                temp.unlink()
                break
            except PermissionError:
                if attempt == 39: raise
                time.sleep(.05)


def pin(stage, builder_receipt, output, root=g.ROOT):
    inputs = g.Inputs(root)
    stage, receipt, output = [g.local_path(p, root) for p in (stage, builder_receipt, output)]
    payload, _ = pin_payload(stage, receipt, inputs)
    inputs.read(Path(__file__)); inputs.recheck()
    g.need(not output.is_relative_to(root / '.local/state/gates'), 'PIN is not an admission receipt')
    write_pin(output, payload)
    return dict(status='REVIEW_REQUIRED', pin=str(output), pin_sha256=g.sha(g.encoded(payload)))


def authenticated_pin(path, inputs):
    saved = inputs.json(path)
    g.need(saved.get('version') == VERSION and saved.get('fixture_only') is (STAGE != CANONICAL_STAGE) and
           saved.get('status') == 'REVIEW_REQUIRED', 'placeholder/invalid R224 PIN')
    expected, materialized = pin_payload(g.local_path(saved['source_stage'], inputs.root),
        g.local_path(saved['builder_receipt'], inputs.root), inputs)
    g.need(g.encoded(saved) == g.encoded(expected), 'R224 PIN differs from reconstructed completed stage')
    return saved, materialized


def parent_files(old):
    files = {'kernel/' + n:b for n,b in old[3].items()}
    files['registration.json'] = g.encoded(dict(version=VERSION, kernel_id=old[0]['id'], canonical_parent=PARENT))
    files['SHA256SUMS'] = s.sums(files, g)
    return files


def bind(pin_path, output, review, root=g.ROOT):
    inputs = g.Inputs(root)
    pin_path, output, review = [g.local_path(p, root) for p in (pin_path, output, review)]
    g.need(not output.exists(), 'bind output exists; use a fresh versioned gate copy')
    saved, (old, new, lane, rules, diff) = authenticated_pin(pin_path, inputs)
    g.need(not output.is_relative_to(inputs.root / STAGE) and not output.is_relative_to(inputs.root / PARENT), 'bind output inside source/parent')
    pfiles = parent_files(old)
    binding = dict(source_stage=STAGE, review=review.relative_to(inputs.root).as_posix(),
        builder_receipt=saved['builder_receipt'], builder_receipt_sha256=saved['builder_receipt_sha256'])
    reg = dict(version='1.1.23-r224q', binding_version=VERSION, kernel_id=SLUG, parent=str(output/'parent'),
        parent_sha256sums=g.sha(pfiles['SHA256SUMS']), source_stage=STAGE, kernel_files=new[2]['kernel_files'],
        builder_lane=lane, qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    files = {'kernel/' + n:b for n,b in new[3].items()}
    files.update({'registration.json':g.encoded(reg), 'full-diff.txt':diff.encode(), 'R224-PIN.json':inputs.read(pin_path)})
    files['SHA256SUMS'] = s.sums(files, g)
    allow = dict(version=VERSION, review_binding=binding, parent_sha256sums=reg['parent_sha256sums'],
        stage_sha256sums=g.sha(files['SHA256SUMS']), files=rules, pin_sha256=g.sha(files['R224-PIN.json']), scope=SCOPE)
    files['allowlist.json'] = g.encoded(allow)
    files['review-request.md'] = (f'Independent scope review required; no automatic mathematical approval.\n'
        f'Reviewer-Lane: <independent lane>\nSHA256SUMS: {allow["stage_sha256sums"]}\n'
        f'Allowlist-SHA256: {g.sha(files["allowlist.json"])}\nR224-PIN-SHA256: {allow["pin_sha256"]}\n'
        f'R224-Scope: {SCOPE}\nCheck all 29 parent members and added provenance; formula, failure masks, '
        'survivor denominators, target/UID order and untouched existing launches.\nVERDICT: PENDING\n').encode()
    files.update({'parent/' + n:b for n,b in pfiles.items()})
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target=output/name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(data)
    return dict(status='REVIEW_REQUIRED', stage=str(output), review=str(review))


def evaluate(stage, review, root=g.ROOT):
    inputs = g.Inputs(root)
    stage, review = [g.local_path(p, root) for p in (stage, review)]
    parent = stage/'parent'
    errors=[]; old=new=None; diff=''; reviewer=None; a=b={}
    try:
        saved, (old,new,lane,rules,diff) = authenticated_pin(stage/'R224-PIN.json', inputs)
        copy_info = g.kernel(stage, inputs)
        g.need(copy_info[3] == new[3], 'gate copy differs from completed source')
        pfiles = parent_files(old)
        for name, data in pfiles.items():
            g.need(inputs.read(parent/name) == data, 'parent copy differs: '+name)
        g.need(g.kernel(parent, inputs)[3] == old[3], 'parent copy kernel set differs')
        pm=g.manifest(parent,inputs,[parent/n for n in pfiles if n!='SHA256SUMS'])[1]
        sm=g.manifest(stage,inputs,copy_info[4]+[stage/n for n in ('registration.json','full-diff.txt','R224-PIN.json')])[1]
        allow=inputs.json(stage/'allowlist.json')
        expected=dict(version=VERSION, review_binding=dict(source_stage=STAGE,
            review=review.relative_to(inputs.root).as_posix(), builder_receipt=saved['builder_receipt'],
            builder_receipt_sha256=saved['builder_receipt_sha256']), parent_sha256sums=pm,
            stage_sha256sums=sm, files=rules, pin_sha256=g.sha(inputs.read(stage/'R224-PIN.json')), scope=SCOPE)
        g.need(g.encoded(allow)==g.encoded(expected), 'R224 allowlist differs')
        reg=inputs.json(stage/'registration.json')
        expected_reg=dict(version='1.1.23-r224q', binding_version=VERSION, kernel_id=SLUG, parent=str(parent),
            parent_sha256sums=pm, source_stage=STAGE, kernel_files=new[2]['kernel_files'], builder_lane=lane,
            qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
        g.need(g.encoded(reg)==g.encoded(expected_reg), 'R224 registration differs')
        g.need(inputs.read(stage/'full-diff.txt')==diff.encode(), 'R224 full diff differs')
        reviewer=g.review_check(review,allow,sm,inputs,new,stage/'allowlist.json')
        text=inputs.read(review).decode('utf-8-sig')
        g.need(re.findall(r'^R224-PIN-SHA256: ('+g.HEX+r')\s*$',text,re.M)==[allow['pin_sha256']], 'review does not bind R224 PIN')
        g.need(re.findall(r'^R224-Scope: (.+)$',text,re.M)==[SCOPE], 'independent R224 scope review missing')
        normalized=g.fast_inventory(old,new,g.FAST_REFERENCE,inputs)
        _,_,a,b=s.comparison(old[1],(new[0],normalized,*new[2:]),old[3],g)
    except (ValueError,OSError,KeyError,TypeError,SyntaxError,IndexError,zipfile.BadZipFile) as exc:
        errors.append(str(exc))
    for tool in (g.__file__,s.__file__,__file__): inputs.read(Path(tool))
    try: inputs.recheck()
    except (ValueError,OSError) as exc: errors.append(str(exc))
    return dict(gate_version=VERSION,fixture_only=STAGE != CANONICAL_STAGE,status='FAIL' if errors else 'PASS',errors=sorted(set(errors)),
        stage=str(stage),parent=str(parent),kernel_id=new[0]['id'] if new else None,reviewer_lane=reviewer,
        input_sha256=inputs.hashes(),parent_inventory=old[2] if old else None,stage_inventory=new[2] if new else None,
        comparison_sha256=dict(parent={n:g.sha(v) for n,v in a.items()},stage={n:g.sha(v) for n,v in b.items()}),
        full_diff=diff,diff_sha256=g.sha(diff.encode()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    cmd=commands.add_parser('pin'); cmd.add_argument('--stage',default=STAGE)
    cmd.add_argument('--builder-receipt',required=True); cmd.add_argument('--output',required=True)
    cmd=commands.add_parser('bind'); cmd.add_argument('--pin',required=True)
    cmd.add_argument('--output',required=True); cmd.add_argument('--review',required=True)
    cmd=commands.add_parser('evaluate'); cmd.add_argument('--stage',required=True)
    cmd.add_argument('--review',required=True); cmd.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='pin': result=pin(args.stage,args.builder_receipt,args.output)
        elif args.command=='bind': result=bind(args.pin,args.output,args.review)
        else:
            result=evaluate(args.stage,args.review)
            if result['status']=='PASS' and not args.check_only:
                g.need(result['fixture_only'] is False, 'fixture-only gate cannot publish admission')
                g.write_receipt(g.ROOT/'.local/state/gates'/ (SLUG.split('/')[1]+'.json'),result)
        print(g.encoded({k:v for k,v in result.items() if k!='full_diff'}).decode())
        return 2 if result.get('status')=='FAIL' else 0
    except (ValueError,OSError,KeyError,TypeError,SyntaxError,IndexError,zipfile.BadZipFile) as exc:
        print('REFUSED: '+str(exc)); return 2


if __name__=='__main__':
    raise SystemExit(main())
