"""Strict exact-diff gate 1.4.5: immutable proposals for admitted-parent children.

Changelog 1.4.5: exact registered dataset deltas; other metadata values byte-identical.
Backup: .local/backups/strict-variant-gate-v1.4.5/.
Changelog 1.4.4: slug/parent/builder inputs replace per-child Python constants.
Every admission replays the trusted parent chain and the exact pinned proposal;
independent dispatcher review remains mandatory. No inference or network calls.
Backup: .local/backups/strict-variant-gate-v1.4.4/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import strict_variant_gate as g
from tools import strict_variant_gate_standalone as s
from tools import strict_variant_gate_r224_v142 as v142
from tools import strict_variant_gate_r224_v143 as v143
from tools import strict_variant_gate_r224_v144 as v144

VERSION = '1.4.5'
RELEASE_ROOT = 'artifacts/experiments/EXP169_RELEASE_C85R7'
SCOPE = 'exact-admitted-parent-child; all member hashes and unified hunks pinned; FAST unchanged'
TOOLS = (g.__file__, g.hb151.__file__, s.__file__, v142.__file__, v143.__file__, v144.__file__, __file__)
ERRORS = (ValueError, OSError, KeyError, TypeError, SyntaxError, IndexError, zipfile.BadZipFile)


def relative(path, root):
    return g.local_path(path, root).relative_to(root).as_posix()


def slug(value):
    g.need(isinstance(value, str) and re.fullmatch(r'taeyangg4/[a-z0-9][a-z0-9-]*', value), 'invalid child slug')
    return value


def canonical_stage(stage, root):
    return stage.parent == root / RELEASE_ROOT and re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+-[A-Za-z0-9_-]+', stage.name) is not None


def verify_receipt(path, inputs, seen=()):
    """Recompute, do not trust a self-declared PASS or caller-selected evaluator."""
    path = g.local_path(path, inputs.root)
    g.need(path not in seen and len(seen) < 64, 'cyclic/overlong admission chain')
    receipt = inputs.json(path)
    ident = slug(receipt['kernel_id'])
    g.need(path == inputs.root / '.local/state/gates' / (ident.split('/')[1] + '.json'), 'noncanonical admission receipt')
    g.need(receipt.get('status') == 'PASS' and receipt.get('errors') == [] and receipt.get('fixture_only') is False,
           'parent/consumer admission is not a nonfixture PASS')
    stage = g.local_path(receipt['stage'], inputs.root)
    allow = inputs.json(stage / 'allowlist.json')
    review = allow['review_binding']['review']
    version = receipt.get('gate_version')
    g.need(version in ('1.4.2', '1.4.3', '1.4.4', VERSION), 'unsupported admitted parent gate version')
    if version == VERSION:
        current = evaluate(stage, review, inputs.root, (*seen, path))
    else:
        current = {'1.4.2': v142, '1.4.3': v143, '1.4.4': v144}[version].evaluate(stage, review, inputs.root)
    g.need(current == receipt, 'admission differs from trusted evaluator replay')
    for name, digest in receipt['input_sha256'].items():
        g.need(g.sha(inputs.read(name)) == digest, 'stale admission input: ' + name)
    # Frozen evaluators expose their complete file hashes but not Inputs.folders.
    # Every C85 kernel inventory includes kernel-metadata.json. Restore its exact
    # file set from the replay-authenticated receipt, never from a later listing.
    paths = sorted(g.local_path(name, inputs.root) for name in receipt['input_sha256'])
    for folder in {p.parent for p in paths if p.name == 'kernel-metadata.json'}:
        expected = [p for p in paths if p.is_relative_to(folder)]
        g.need(folder not in inputs.folders or inputs.folders[folder] == expected,
               'admission kernel inventory changed during recursive replay')
        inputs.folders[folder] = expected
    inputs.recheck()
    return receipt


def reference_blocks(path, child, inputs):
    if path is None:
        return []
    spec = inputs.json(path)
    g.need(set(spec) == {'blocks'} and isinstance(spec['blocks'], list) and spec['blocks'], 'invalid reference proof')
    result = []
    for block in spec['blocks']:
        g.need(set(block) == {'reference_stage', 'reference_member', 'child_member', 'text'}, 'invalid reference block')
        source = g.local_path(block['reference_stage'], inputs.root)
        ref = g.kernel(source, inputs)
        data = block['text'].encode()
        g.need(data and ref[1]['embedded/' + block['reference_member']].count(data) == 1
               and child[1]['embedded/' + block['child_member']].count(data) == 1,
               'reference block differs byte-for-byte or is ambiguous')
        result.append(dict(**block, kernel_files=ref[2]['kernel_files'], block_sha256=g.sha(data)))
    return result


def metadata_values(raw):
    """Return exact JSON value bytes, including nested formatting, per key."""
    obj = g.parse(raw)  # Reject duplicate keys, including nested keys.
    g.need(isinstance(obj, dict), 'metadata must be an object')
    text = raw.decode('utf-8-sig')
    decoder = json.JSONDecoder(parse_constant=lambda value: g.need(False, 'nonfinite JSON constant'))
    values = {}
    pos = len(text) - len(text.lstrip()) + 1
    while True:
        while pos < len(text) and text[pos].isspace(): pos += 1
        if text[pos] == '}': break
        key, pos = decoder.raw_decode(text, pos)
        while text[pos].isspace(): pos += 1
        g.need(text[pos] == ':', 'invalid metadata separator')
        pos += 1
        while text[pos].isspace(): pos += 1
        start = pos
        _, pos = decoder.raw_decode(text, pos)
        values[key] = text[start:pos].encode('utf-8')
        while text[pos].isspace(): pos += 1
        if text[pos] == '}': break
        g.need(text[pos] == ',', 'invalid metadata separator')
        pos += 1
    g.need(values.keys() == obj.keys(), 'metadata key parsing mismatch')
    return values


def metadata_delta(old, new, ident, registration):
    """Only exact registered source additions/removals and id/title may differ."""
    g.need(isinstance(registration, dict), 'child registration must be an object')
    delta = registration.get('dataset_sources_delta', dict(added=[], removed=[]))
    g.need(isinstance(delta, dict) and set(delta) == {'added', 'removed'}, 'invalid dataset_sources delta')
    lists = [old[0].get('dataset_sources'), new[0].get('dataset_sources'), delta['added'], delta['removed']]
    for values in lists:
        g.need(isinstance(values, list) and all(isinstance(v, str) and
               re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*', v) for v in values),
               'dataset_sources requires exact slugs; no wildcard')
        g.need(len(values) == len(set(values)), 'duplicate dataset_sources slug')
    before, after, added, removed = lists
    g.need(not set(added) & set(removed), 'dataset_sources delta overlaps')
    g.need(set(after)-set(before) == set(added) and set(before)-set(after) == set(removed),
           'unlisted or inexact dataset_sources change')
    g.need([v for v in before if v not in removed] == [v for v in after if v not in added],
           'dataset_sources survivor order changed')
    if added or removed:
        g.need(registration.get('kernel_id') == ident and registration.get('dataset_sources_delta_version') == '1.0.0',
               'dataset_sources registration identity/schema mismatch')
    a = metadata_values(old[3]['kernel-metadata.json'])
    b = metadata_values(new[3]['kernel-metadata.json'])
    g.need(a.keys() == b.keys(), 'metadata keys changed')
    g.need(new[0]['id'] == ident and new[0]['title'] == ident.split('/')[1], 'metadata id/title mismatch')
    exempt = {'id', 'title'} | ({'dataset_sources'} if added or removed else set())
    g.need(all(a[key] == b[key] for key in a.keys()-exempt), 'metadata exceeds registered change; value bytes differ')
    return dict(added=sorted(added), removed=sorted(removed))


def pin_payload(spec, observed, seen=()):
    # A private snapshot keeps proposal provenance independent of bind/evaluate inputs.
    inputs = g.Inputs(observed.root)
    root = inputs.root
    g.need(set(spec) == {'slug', 'stage', 'parent', 'builder_receipt', 'members', 'diff', 'reference_proof'}, 'invalid pin inputs')
    ident = slug(spec['slug'])
    stage, parent, builder, members, source_diff = [g.local_path(spec[k], root) for k in ('stage', 'parent', 'builder_receipt', 'members', 'diff')]
    g.need(stage != parent and not stage.is_relative_to(parent) and not parent.is_relative_to(stage), 'source/parent overlap')
    g.need(members == stage/'new-members.json' and source_diff.parent == stage and source_diff.name.startswith('full-diff')
           and source_diff.suffix == '.txt', 'builder sidecars must belong to completed stage')
    old = g.kernel(parent, inputs)
    parent_receipt = root/'.local/state/gates'/(slug(old[0]['id']).split('/')[1]+'.json')
    admission = verify_receipt(parent_receipt, inputs, seen)
    g.need(g.local_path(admission['stage'], root) == parent and admission['stage_inventory']['kernel_files'] == old[2]['kernel_files'],
           'parent is not the exact admitted gate copy')
    g.need(ident != old[0]['id'], 'child must have a new slug')
    lane, report, _ = g.dispatch(builder, inputs)
    new = g.kernel(stage, inputs)
    g.need(new[0]['id'] == ident, 'child slug differs from source')
    digest = g.sha(g.encoded(new[2]['kernel_files']))
    g.need(re.findall(r'^Kernel-SHA256: (' + g.HEX + r')\s*$', report.decode('utf-8-sig'), re.M) == [digest],
           'completed builder report does not bind this kernel')
    registration = inputs.json(stage/'registration.json')
    dataset_delta = metadata_delta(old, new, ident, registration)
    normalized = v143.fast_inventory(old, new, inputs)
    before = {n[9:]: b for n,b in old[1].items() if n.startswith('embedded/')}
    after = {n[9:]: b for n,b in new[1].items() if n.startswith('embedded/')}
    g.need(before.keys() == after.keys(), 'added/removed members forbidden')
    changed = {n:v143.member_diff(n,before[n],after[n]) for n in sorted(before) if before[n] != after[n]}
    same = {n:g.sha(before[n]) for n in sorted(before) if before[n] == after[n]}
    g.need(changed, 'empty member change')
    side = inputs.json(members)
    g.need(side['added'] == side['removed'] == [], 'sidecar added/removed members')
    expected = [dict(name=n, before_sha256=d['before_sha256'], after_sha256=d['after_sha256']) for n,d in changed.items()]
    g.need(side['changed'] == expected, 'builder changed-member inventory differs')
    unchanged = side['unchanged']
    old_style = [dict(name=n,sha256=h) for n,h in same.items()]
    new_style = [dict(name=n,before_sha256=h,after_sha256=h) for n,h in same.items()]
    g.need(unchanged in (old_style,new_style), 'builder identical-member inventory differs')
    finished_diff = inputs.read(source_diff)
    for name, delta in changed.items():
        hunk = ''.join(delta['unified_diff'].splitlines(keepends=True)[2:]).encode()
        g.need(hunk and hunk in finished_diff, 'builder full-diff lacks exact hunk: ' + name)
    proofs = reference_blocks(spec['reference_proof'], new, inputs)
    rules, diff, a, b = s.comparison(old[1], (new[0], normalized, *new[2:]), old[3], g)
    for tool in TOOLS: inputs.read(Path(tool))
    payload = dict(version=VERSION, status='REVIEW_REQUIRED', fixture_only=not canonical_stage(stage, root),
        inputs=spec, kernel_id=ident, source_stage=spec['stage'], parent=spec['parent'],
        parent_receipt=relative(parent_receipt,root), parent_receipt_sha256=g.sha(inputs.read(parent_receipt)),
        parent_kernel_files=old[2]['kernel_files'], builder_receipt=spec['builder_receipt'], builder_lane=lane,
        builder_receipt_sha256=g.sha(inputs.read(builder)), kernel_files=new[2]['kernel_files'],
        exact_member_diffs=changed, identical_members=same, reference_blocks=proofs,
        dataset_sources_delta=dataset_delta, registration_sha256=g.sha(inputs.read(stage/'registration.json')),
        files=rules, diff_sha256=g.sha(diff.encode()), scope=SCOPE, fast_path=dict(g.FAST_REFERENCE),
        provenance_sha256=inputs.hashes())
    inputs.recheck()
    observed.data.update(inputs.data); observed.folders.update(inputs.folders)
    return payload, (old,new,lane,rules,diff,a,b)


def pin(stage, child_slug, parent, builder_receipt, diff, output, reference_proof=None, root=g.ROOT):
    inputs = g.Inputs(root); root=inputs.root
    spec = dict(slug=slug(child_slug), **{k:relative(v,root) for k,v in dict(stage=stage,parent=parent,
        builder_receipt=builder_receipt,members=g.local_path(stage,root)/'new-members.json',diff=diff).items()},
        reference_proof=relative(reference_proof,root) if reference_proof else None)
    output = g.local_path(output,root)
    g.need(not output.is_relative_to(root/'.local/state/gates'), 'PIN is not an admission receipt')
    payload,_ = pin_payload(spec,inputs)
    inputs.recheck(); v143.write_pin(output,payload)
    return dict(status='REVIEW_REQUIRED',pin=str(output),pin_sha256=g.sha(g.encoded(payload)))


def authenticated_pin(path, inputs, seen=()):
    saved = inputs.json(path)
    g.need(saved.get('version') == VERSION and saved.get('status') == 'REVIEW_REQUIRED', 'invalid exact-diff PIN')
    expected, material = pin_payload(saved['inputs'],inputs,seen)
    g.need(saved == expected, 'PIN differs from reconstructed completed stage')
    return saved,material


def content_line(saved):
    return (f'Exact-Diff-Content: {saved["kernel_id"]}; {len(saved["exact_member_diffs"])} changed, '
            f'{len(saved["identical_members"])} identical; FAST unchanged; references-sha256='
            +g.sha(g.encoded(saved['reference_blocks'])))


def bound_files(saved, material, pin_bytes, output, review, root):
    old,new,lane,rules,diff,_,_ = material
    parent = {'kernel/'+n:b for n,b in old[3].items()}
    parent['registration.json'] = g.encoded(dict(version=VERSION,kernel_id=old[0]['id'],canonical_parent=saved['parent']))
    parent['SHA256SUMS'] = s.sums(parent,g)
    binding = dict(source_stage=saved['source_stage'],review=relative(review,root),builder_receipt=saved['builder_receipt'],
                   builder_receipt_sha256=saved['builder_receipt_sha256'])
    reg = dict(version=Path(saved['source_stage']).name.removeprefix('v'),binding_version=VERSION,kernel_id=saved['kernel_id'],
        parent=str(output/'parent'),parent_sha256sums=g.sha(parent['SHA256SUMS']),source_stage=saved['source_stage'],
        kernel_files=new[2]['kernel_files'],builder_lane=lane,qualified_for_submission=False,exposure='EXPOSED',clean_oof=False,
        dataset_sources_delta=saved['dataset_sources_delta'])
    files = {'kernel/'+n:b for n,b in new[3].items()}
    files.update({'registration.json':g.encoded(reg),'full-diff.txt':diff.encode(),'R224-PIN.json':pin_bytes})
    files['SHA256SUMS'] = s.sums(files,g)
    allow = dict(version=VERSION,review_binding=binding,parent_sha256sums=reg['parent_sha256sums'],
        stage_sha256sums=g.sha(files['SHA256SUMS']),files=rules,pin_sha256=g.sha(pin_bytes),scope=SCOPE)
    files['allowlist.json'] = g.encoded(allow)
    files['review-request.md'] = (f'Independent exact-diff review required; PIN grants no approval.\n'
        f'Dataset-Sources-Delta: {json.dumps(saved["dataset_sources_delta"], sort_keys=True)}\n'
        f'Source-Registration-SHA256: {saved["registration_sha256"]}\n'
        f'{content_line(saved)}\nReviewer-Lane: <independent lane>\nSHA256SUMS: {allow["stage_sha256sums"]}\n'
        f'Allowlist-SHA256: {g.sha(files["allowlist.json"])}\nR224-PIN-SHA256: {allow["pin_sha256"]}\n'
        f'R224-Scope: {SCOPE}\nVerify exact parent admission, all member SHA256/hunks, reference blocks and FAST.\n'
        'Check registered formula, failure masks, survivor denominators, target/UID order, and untouched launches.\n'
        'Reference blocks prove only the named bytes; adapters and other changes require independent scope review.\n'
        'Runtime-Basis: see source registration timing; independent runtime review required.\n'
        'Runtime-Increment-Evidence: pending independent timing evidence.\n').encode()
    files.update({'parent/'+n:b for n,b in parent.items()})
    return files,allow


def bind(pin_path, output, review, root=g.ROOT):
    inputs=g.Inputs(root); root=inputs.root
    pin_path,output,review = [g.local_path(p,root) for p in (pin_path,output,review)]
    g.need(not output.exists(), 'bind output exists; use a fresh versioned gate copy')
    saved,material=authenticated_pin(pin_path,inputs)
    g.need(all(not output.is_relative_to(g.local_path(saved[k],root)) for k in ('source_stage','parent')),
           'bind output inside source/parent')
    files,_=bound_files(saved,material,inputs.read(pin_path),output,review,root)
    inputs.recheck(); output.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        target=output/name; target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream: stream.write(data)
    return dict(status='REVIEW_REQUIRED',stage=str(output),review=str(review))


def evaluate(stage, review, root=g.ROOT, seen=()):
    inputs=g.Inputs(root); root=inputs.root
    stage,review=[g.local_path(p,root) for p in (stage,review)]
    errors=[]; old=new=None; saved={}; diff=''; a=b={}; reviewer=None
    try:
        saved,material=authenticated_pin(stage/'R224-PIN.json',inputs,seen)
        old,new,lane,rules,diff,a,b=material
        expected,allow=bound_files(saved,material,inputs.read(stage/'R224-PIN.json'),stage,review,root)
        g.need(g.kernel(stage,inputs)[3] == new[3], 'gate copy differs from completed source')
        g.need(g.kernel(stage/'parent',inputs)[3] == old[3], 'parent copy kernel set differs')
        for name,data in expected.items():
            g.need(inputs.read(stage/name) == data, 'bound file differs: '+name)
        g.manifest(stage,inputs,[stage/n for n in expected if n.startswith('kernel/') or n in ('registration.json','full-diff.txt','R224-PIN.json')])
        g.manifest(stage/'parent',inputs,[stage/n for n in expected if n.startswith('parent/') and n != 'parent/SHA256SUMS'])
        reviewer=g.review_check(review,allow,allow['stage_sha256sums'],inputs,new,stage/'allowlist.json')
        text=inputs.read(review).decode('utf-8-sig')
        for prefix,value in (('Dataset-Sources-Delta',json.dumps(saved['dataset_sources_delta'], sort_keys=True)),('R224-PIN-SHA256',allow['pin_sha256']),('R224-Scope',SCOPE),('Exact-Diff-Content',content_line(saved).split(': ',1)[1])):
            g.need([line for line in text.splitlines() if line.startswith(prefix+':')] == [prefix+': '+value],
                   'independent review missing/stale '+prefix)
    except ERRORS as exc: errors.append(str(exc))
    for tool in TOOLS: inputs.read(Path(tool))
    try: inputs.recheck()
    except ERRORS as exc: errors.append(str(exc))
    return dict(gate_version=VERSION,fixture_only=saved.get('fixture_only',True),status='FAIL' if errors else 'PASS',
        errors=sorted(set(errors)),stage=str(stage),parent=str(stage/'parent'),kernel_id=new[0]['id'] if new else None,
        reviewer_lane=reviewer,input_sha256=inputs.hashes(),parent_inventory=old[2] if old else None,
        stage_inventory=new[2] if new else None,comparison_sha256=dict(parent={n:g.sha(v) for n,v in a.items()},stage={n:g.sha(v) for n,v in b.items()}),
        full_diff=diff,diff_sha256=g.sha(diff.encode()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    cmd=commands.add_parser('pin')
    for arg in ('stage','slug','parent','builder-receipt','diff','output'): cmd.add_argument('--'+arg,required=True)
    cmd.add_argument('--reference-proof')
    cmd=commands.add_parser('bind')
    for arg in ('pin','output','review'): cmd.add_argument('--'+arg,required=True)
    cmd=commands.add_parser('evaluate'); cmd.add_argument('--stage',required=True); cmd.add_argument('--review',required=True)
    cmd.add_argument('--check-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command == 'pin': result=pin(args.stage,args.slug,args.parent,args.builder_receipt,args.diff,args.output,args.reference_proof)
        elif args.command == 'bind': result=bind(args.pin,args.output,args.review)
        else:
            result=evaluate(args.stage,args.review)
            if result['status']=='PASS' and not args.check_only:
                g.need(result['fixture_only'] is False,'fixture-only gate cannot publish admission')
                g.write_receipt(g.ROOT/'.local/state/gates'/(slug(result['kernel_id']).split('/')[1]+'.json'),result)
        print(g.encoded({k:v for k,v in result.items() if k!='full_diff'}).decode())
        return 2 if result.get('status')=='FAIL' else 0
    except ERRORS as exc:
        print('REFUSED: '+str(exc)); return 2


if __name__=='__main__':
    raise SystemExit(main())
