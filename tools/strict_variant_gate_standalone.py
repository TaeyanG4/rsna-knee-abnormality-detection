"""v1.3.0: exact static R-KD and ConvNeXt-S L2 contracts; no model execution."""
from __future__ import annotations
import ast
import base64
import copy
import difflib
from pathlib import Path
import pickletools
import re
import zipfile

from tools import strict_variant_gate_hb151 as h

VERSION = '1.3.0'
A = 'taeyangg4/knee-hb157-rkd-standalone-v1'
B = 'taeyangg4/knee-hb156-cnxs-l2-v1'
IDS = {A, B}
AP = 'storage/external/nartaa__rsna-knee-0949-anatomical-mirror/rsna-knee-0949-anatomical-mirror.ipynb'
BP = 'artifacts/experiments/HB151_KILLTEST/v1.0.2'
AP_ID = 'nartaa/rsna-knee-0949-anatomical-mirror'
AP_SHA = '8f1f96530b08543bd6caeed48489eea7c01657d6994f22e21a8400047fd4e100'
AP_CK = '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
A_NAME = 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
A_PROTOCOL = '1a6cbf74548dcad4ae440eab659b2230492384a4eb52915782362c20149c5185'
B_PROTOCOL = '788d98ed4b22a970bc4e09e9fcd17a864918c62e4c635fe88380e6d556a2f39c'
B_SCHEMA = '2c18c2b48bc57546931f144ccd59c17140e5b0762f5f56b61426122c19a8f2da'
B_PLACEHOLDER = 'ad3baf5c3c37e7bc15e55cd2ec5438230932e64a43f7c9ed4ebbd6bd01b84c59'
BPINS = {
    'build-receipt.json': '289aeee90284e82827c7eb8e3009eae25dcb09d4d7b6b7564b4a70ddb64eb436',
    'kernel/release.ipynb': 'd1a557f282fd429b61d0f1b70998d5604741410511773da5cf52f3cd51e3d8ed',
    'kernel/kernel-metadata.json': 'a9f0296f00e9acdd35bb3b7016e5c9e8cc4eeb1dc732fa5c95cc5b9e012f1680',
    'dataset/dataset-metadata.json': '1315737d9366e4c08d696e2c0d4d942a9505af78108cdb5c6a8b439e01bd9ae7',
    'dataset/' + h.ARCHIVE: 'dd4198a770c65ad1eccb8e0b90f278e31e6c699d2e4432aa30d9a4c48783e312',
}
BSOURCES = {
    'training/src/hb156_model_v1.py': '9f93a04336038da17bf3627ad8e002d70c016224600ae5db845485983538ce88',
    'training/src/exp143_own0949.py': '77aab67cd9abb8a2d90cbe74334f66b23137806c64db40ab548b523bbeecca58',
    'training/src/raptor_common.py': 'bc6db5b3e8cd13b52fbff8c039d3674cdf573f86c1959e22fe9b2fd94255d7dc',
}
DATASETS = {A: 'taeyangg4/knee-hb157-rkd-weights-v1', B: 'taeyangg4/knee-hb156-cnxs-l2-weights-v1'}


def a_meta(parent=False):
    ident = AP_ID if parent else A
    return dict(id=ident, title=ident.split('/')[1], code_file='release.ipynb', language='python',
        kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=False,
        dataset_sources=(['nartaa/rsna-knee-publication-swa-weights-20261007',
                          'nartaa/rsna-knee-oai-original384-s42-pair'] if parent else [DATASETS[A]]),
        competition_sources=['rsna-knee-abnormality-detection'], kernel_sources=[])


def recognizes(stage, g):
    path = stage / 'kernel/kernel-metadata.json'
    return path.is_file() and g.parse(path.read_bytes()).get('id') in IDS


def raw_kernel(stage, inputs, g):
    folder = stage / 'kernel'
    paths = sorted(p for p in folder.rglob('*') if p.is_file())
    inputs.folders[folder] = paths
    g.need(all(not p.is_symlink() and p.resolve().is_relative_to(folder.resolve()) for p in paths), 'linked kernel file')
    raw = {p.relative_to(folder).as_posix(): inputs.read(p) for p in paths}
    g.need(set(raw) == {'release.ipynb', 'kernel-metadata.json'}, 'standalone kernel file set differs')
    return raw, paths


def embedded(nb, g):
    result = {}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.walk(ast.parse(''.join(cell['source']))):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith('_B64'):
                        g.need(target.id not in result and isinstance(node.value, ast.Constant), 'duplicate/nonliteral embedded file')
                        result[target.id] = base64.b64decode(node.value.value, validate=True)
    return result


def checkpoint_metadata(stream, g):
    """Inspect pickle opcodes, never unpickle or execute checkpoint code."""
    with zipfile.ZipFile(stream) as z:
        names = [n for n in z.namelist() if n.endswith('/data.pkl') or n == 'data.pkl']
        g.need(len(names) == 1 and z.getinfo(names[0]).file_size <= g.MAX_BYTES, 'checkpoint metadata missing/oversize')
        markers = {'placeholder', 'fixture', 'hb156_killtest_placeholder', 'hb151_killtest_placeholder'}
        g.need(not any(isinstance(arg, str) and arg in markers for _, arg, _ in pickletools.genops(z.read(names[0]))),
               'placeholder checkpoint metadata')


def parent_material(ident, inputs, g):
    if ident == A:
        raw = {'release.ipynb': inputs.read(inputs.root / AP), 'kernel-metadata.json': g.encoded(a_meta(True))}
        g.need(g.sha(raw['release.ipynb']) == AP_SHA, 'public R parent pin differs')
        small = {'embedded/' + n: b for n, b in embedded(g.parse(raw['release.ipynb']), g).items()}
        g.need(set(small) == {'embedded/_FASTREAD_B64'}, 'public R embedded inventory')
        small['external/checkpoint.sha256'] = (AP_CK + '\n').encode()
        return raw, small, {}, None
    source = inputs.root / BP
    for name, pin in BPINS.items():
        digest = inputs.digest(source / name) if name.endswith('.bin') else g.sha(inputs.read(source / name))
        g.need(digest == pin, 'HB151 parent pin differs: ' + name)
    raw = {n: inputs.read(source / 'kernel' / n) for n in ('release.ipynb', 'kernel-metadata.json')}
    hashes, small = h.archive(source / 'dataset' / h.ARCHIVE, BPINS['dataset/' + h.ARCHIVE], inputs, g)
    g.need(hashes == inputs.json(source / 'build-receipt.json')['files'], 'HB151 parent member hashes differ')
    return raw, small, hashes, inputs.json(source / 'dataset/dataset-metadata.json')


def replace(data, before, after, count, g):
    g.need(data.count(before) == count, 'registered patch anchor differs')
    return data.replace(before, after)


def expected_a(parent, checkpoint, donor, g):
    nb = copy.deepcopy(parent)
    changes = {AP_CK: checkpoint}
    for slug in ('rsna-knee-publication-swa-weights-20261007', 'rsna-knee-oai-original384-s42-pair'):
        changes['/kaggle/input/' + slug] = '/kaggle/input/' + DATASETS[A].split('/')[1]
        changes['/kaggle/input/datasets/nartaa/' + slug] = '/kaggle/input/datasets/' + DATASETS[A]
    joined = ''.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    g.need(all(a in joined for a in changes), 'public R patch anchor missing')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            for a, b in changes.items(): src = src.replace(a, b)
            cell['source'] = 'if not FAST:\n' + ''.join('    ' + line for line in src.splitlines(True))
    nb['cells'].insert(0, copy.deepcopy(donor['cells'][0]))
    return nb


def expected_b(parent_raw, parent_small, hashes, binding, schema, archive_sha, g):
    code = {
        h.RUNTIME: replace(parent_small[h.RUNTIME], b'ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"',
                           b'ARCH = "convnext_small.fb_in22k_ft_in1k_384"', 1, g),
        'public-serving.py': replace(parent_small['public-serving.py'],
            b'kw = dict(pretrained=pretrained, num_classes=0, in_chans=3, img_size=INPUT_RES)',
            b'kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)', 1, g),
    }
    m = g.parse(parent_small['release-manifest.json'])
    g.need(g.sha(g.encoded(schema)) == B_SCHEMA, 'ConvNeXt state schema differs')
    m.update(protocol_sha256=B_PROTOCOL, own_state_schema=schema, hb151_export=binding,
             asset_sha256={h.NAME: binding['sha256'], **{n:g.sha(b) for n,b in code.items()}},
             serving_sha256=g.sha(code['public-serving.py']), dataset=DATASETS[B],
             hb156_provenance=dict(version='1.0.0', training_sources=BSOURCES,
                 inherited_fields_are_parent_ancestry=['average_receipt_sha256','source_final_sha256','source_bindings']))
    m['checkpoints'][h.NAME]['sha256'] = binding['sha256']
    code['release-manifest.json'] = h.wire(m)
    nb, meta = g.parse(parent_raw['release.ipynb']), g.parse(parent_raw['kernel-metadata.json'])
    src = ''.join(nb['cells'][5]['source'])
    for before, after, count in (
        ('knee-hb151-pl-coat-ckpt-v1', DATASETS[B].split('/')[1], 2),
        (BPINS['dataset/' + h.ARCHIVE], archive_sha, 1),
        (hashes['release-manifest.json'], g.sha(code['release-manifest.json']), 1),
        (hashes[h.RUNTIME], g.sha(code[h.RUNTIME]), 1)):
        src = replace(src, before, after, count, g)
    nb['cells'][5]['source'] = src
    meta.update(id=B, title=B.split('/')[1], dataset_sources=[DATASETS[B]])
    return nb, meta, code


def inventory(raw, small, hashes, g):
    inv = h.inventory(raw['release.ipynb'], g)
    inv['kernel/kernel-metadata.json'] = raw['kernel-metadata.json']
    if hashes:
        inv.update({'external/' + n: b for n, b in small.items()})
        inv['external/' + h.NAME + '.sha256'] = (hashes[h.NAME] + '\n').encode()
    else:
        inv.update(small)
    # B's runtime contains nested base64 files: inventory every one too.
    if h.RUNTIME in small:
        pseudo = {'cells': [dict(cell_type='code', source=small[h.RUNTIME].decode())]}
        inv.update({'embedded/' + n: b for n, b in embedded(pseudo, g).items()})
    return inv


def kernel(stage, inputs, g):
    raw, paths = raw_kernel(stage, inputs, g)
    ident = g.parse(raw['kernel-metadata.json'])['id']
    g.need(ident in IDS, 'unsupported standalone family')
    reg = inputs.json(stage / 'registration.json') if (stage / 'registration.json').is_file() else {}
    source = g.local_path(reg.get('external_source', str(stage)), inputs.root)
    original, _ = raw_kernel(source, inputs, g)
    g.need(original == raw, 'bound source kernel differs')
    dataset = source / 'dataset'
    paths_dataset = sorted(p for p in dataset.rglob('*') if p.is_file())
    inputs.folders[dataset] = paths_dataset
    expected_names = ({'dataset-metadata.json', A_NAME} if ident == A else
        {'dataset-metadata.json', h.ARCHIVE, h.NAME + '.sha256', h.NAME + '.json'})
    g.need({p.relative_to(dataset).as_posix() for p in paths_dataset} == expected_names
           and all(not p.is_symlink() and p.resolve().is_relative_to(dataset.resolve()) for p in paths_dataset),
           'standalone dataset file set differs')
    parent_raw, parent_small, parent_hashes, parent_dmeta = parent_material(ident, inputs, g)
    rec = inputs.json(source / 'build-receipt.json')
    g.need(rec['placeholder'] is False, 'placeholder checkpoint cannot be admitted')
    g.need(rec['version'] == '1.0.0' and rec['exposure'] == 'EXPOSED' and rec['clean_oof'] is False, 'standalone receipt identity')
    g.fast_reference(g.FAST_REFERENCE, inputs)
    donor = inputs.json(inputs.root / g.FAST_REFERENCE['stage'] / 'kernel/release.ipynb')
    if ident == A:
        ck = rec['checkpoint_sha256']
        g.need(re.fullmatch(g.HEX, ck) and ck != AP_CK, 'placeholder/public parent checkpoint')
        g.need(rec['parent_sha256'] == AP_SHA and rec['donor_sha256'] == g.HISTORICAL['v1.1.18-own25q3']['kernel_files']['release.ipynb']
               and rec['protocol_sha256'] == A_PROTOCOL, 'R-KD parent/donor/protocol differs')
        checkpoint = source / 'dataset' / A_NAME
        g.need(inputs.digest(checkpoint) == ck, 'R-KD checkpoint hash differs')
        with checkpoint.open('rb') as stream: checkpoint_metadata(stream, g)
        nb, meta = expected_a(g.parse(parent_raw['release.ipynb']), ck, donor, g), a_meta()
        small = {k:v for k,v in parent_small.items()}
        small['external/checkpoint.sha256'] = (ck + '\n').encode()
        hashes = {}
        dmeta = dict(id=DATASETS[A], title=DATASETS[A].split('/')[1], licenses=[dict(name='other')],
            description='Private H-B-157 R-KD EXPOSED clean_oof=false checkpoint. Prepared locally; not uploaded.')
        smoke = inputs.json(source / 'cpu-smoke.json')
        g.need(smoke['status'] == 'PASS' and smoke['build_receipt_sha256'] == g.sha(inputs.read(source / 'build-receipt.json'))
               and smoke['notebook_sha256'] == g.sha(raw['release.ipynb']) and smoke['protocol_sha256'] == A_PROTOCOL, 'R-KD CPU smoke stale')
        archive_sha = None
    else:
        binding = inputs.json(source / 'export-binding.json')
        ck = binding['sha256']
        g.need(re.fullmatch(g.HEX, ck) and ck not in (B_PLACEHOLDER, parent_hashes[h.NAME]), 'placeholder/parent checkpoint')
        fixed = dict(protocol_sha256=B_PROTOCOL, run_dir='stage-convnext-l2-s42-v1/S42', source='H-B-156-L2-seed42',
            seed=42, exposure='EXPOSED', clean_oof=False, serving_name=h.NAME, placeholder=False)
        g.need(set(binding) == set(fixed) | {'sha256','native_sidecars','native_json_sha256'}
               and all(binding[k] == v and type(binding[k]) is type(v) for k,v in fixed.items()), 'HB156 export identity differs')
        pin = (ck + '  ' + h.NAME + '\n').encode()
        g.need(binding['native_sidecars'] == {Path(h.NAME).with_suffix('.sha256').name: g.sha(pin)}
               and re.fullmatch(g.HEX, binding['native_json_sha256']), 'HB156 native sidecar binding differs')
        g.need(inputs.read(source / 'dataset' / (h.NAME + '.sha256')) == pin and
               g.encoded(inputs.json(source / 'dataset' / (h.NAME + '.json'))) == g.encoded(binding), 'HB156 export sidecars differ')
        g.need(rec['parent_pins'] == BPINS and rec['source_pins'] == BSOURCES and rec['status'] == 'BOUND_NOT_REVIEWED', 'HB156 receipt lineage differs')
        archive_path = source / 'dataset' / h.ARCHIVE
        archive_sha = rec['artifacts']['dataset/' + h.ARCHIVE]
        hashes, small = h.archive(archive_path, archive_sha, inputs, g)
        g.need(hashes == rec['files'] and hashes[h.NAME] == ck, 'HB156 archive checkpoint differs')
        with zipfile.ZipFile(archive_path) as z, z.open(h.NAME) as stream: checkpoint_metadata(stream, g)
        nb, meta, expected = expected_b(parent_raw, parent_small, parent_hashes, binding,
                                      g.parse(small['release-manifest.json'])['own_state_schema'], archive_sha, g)
        g.need(small == expected, 'HB156 embedded code/manifest exceeds registered delta')
        dmeta = copy.deepcopy(parent_dmeta)
        dmeta.update(id=DATASETS[B], title=DATASETS[B].split('/')[1],
            description='Private H-B-156 L2 ConvNeXt-S seed42 SWA [14,11,12]; EXPOSED / clean_oof=false. No training data.')
        for name in ('kernel/release.ipynb','kernel/kernel-metadata.json','dataset/dataset-metadata.json', 'export-binding.json'):
            g.need(g.sha(inputs.read(source / name)) == rec['artifacts'][name], 'HB156 artifact hash differs: ' + name)
        smoke = inputs.json(source / 'cpu-smoke.json')
        g.need(smoke['status'] == 'PASS' and smoke['placeholder'] is False and
               smoke['build_receipt_sha256'] == g.sha(inputs.read(source / 'build-receipt.json')), 'HB156 CPU smoke stale')
    g.need(g.encoded(g.parse(raw['release.ipynb'])) == g.encoded(nb), 'standalone notebook exceeds registered delta')
    g.need(g.encoded(g.parse(raw['kernel-metadata.json'])) == g.encoded(meta), 'standalone metadata exceeds registered delta (title/id)')
    g.need(g.encoded(inputs.json(source / 'dataset/dataset-metadata.json')) == g.encoded(dmeta), 'standalone dataset metadata differs')
    actual_embedded = embedded(g.parse(raw['release.ipynb']), g)
    g.need(actual_embedded == embedded(g.parse(parent_raw['release.ipynb']), g), 'notebook embedded files differ')
    inv = inventory(raw, small, hashes, g)
    inv['dataset/dataset-metadata.json'] = inputs.read(source / 'dataset/dataset-metadata.json')
    details = dict(kernel_files={n:g.sha(b) for n,b in raw.items()}, embedded={n:g.sha(b) for n,b in small.items()},
        archive_sha256=archive_sha, fixture_only=False, standalone_policy=dict(version=VERSION, family=ident,
            parent_kernel_sha256=g.sha(parent_raw['release.ipynb']), checkpoint_sha256=ck,
            source_stage=source.relative_to(inputs.root).as_posix(), dataset=DATASETS[ident],
            build_receipt_sha256=g.sha(inputs.read(source / 'build-receipt.json'))))
    inputs.read(Path(__file__))
    inputs.read(Path(h.__file__))
    return meta, inv, details, raw, paths


def parent_files(ident, inputs, g):
    raw, small, hashes, dmeta = parent_material(ident, inputs, g)
    files = {'kernel/' + n: b for n, b in raw.items()}
    files['registration.json'] = g.encoded(dict(version=VERSION, kernel_id=g.parse(raw['kernel-metadata.json'])['id'],
        canonical_parent=AP if ident == A else BP, synthetic_metadata=ident == A))
    files['SHA256SUMS'] = sums(files, g)
    inv = inventory(raw, small, hashes, g)
    if dmeta is not None:
        inv['dataset/dataset-metadata.json'] = inputs.read(inputs.root / BP / 'dataset/dataset-metadata.json')
    details = dict(kernel_files={n:g.sha(b) for n,b in raw.items()}, embedded={n:g.sha(b) for n,b in small.items()},
                   synthetic_metadata=ident == A)
    return files, inv, details, raw


def sums(files, g):
    return ''.join(g.sha(b) + '  ' + n + '\n' for n,b in sorted(files.items())).encode()


def comparison(before, new, parent_raw, g):
    # Missing inventory entries are explicit empty sides, never silently omitted.
    # Semantic reconstruction in kernel() has already proved every added cell.
    names = before.keys() | new[1].keys()
    a, b = ({n:x.get(n, b'') for n in sorted(names)} for x in (before, new[1]))
    rules = g.exact_rules(a, b)
    # kernel() proves the exact transformation; evaluate() requires equality to
    # these recomputed rules. Avoid the generic quadratic regex cross-product
    # for thousands of changed schema lines (no caller-supplied rule is trusted).
    diff = ''.join(''.join(difflib.unified_diff(a[n].decode().splitlines(True),
        b[n].decode().splitlines(True), fromfile='parent/' + n, tofile='stage/' + n))
        for n in sorted(names) if a[n] != b[n])
    identity = ''.join(f'{n}: {"IDENTICAL" if a[n] == b[n] else "REGISTERED_CHANGE"} '
                       f'{g.sha(a[n])} -> {g.sha(b[n])}\n' for n in sorted(names))
    for name in sorted(parent_raw):
        diff += ''.join(difflib.unified_diff(parent_raw[name].decode().splitlines(True),
            new[3][name].decode().splitlines(True), fromfile='raw-parent/' + name, tofile='raw-stage/' + name))
    return rules, identity + diff, a, b


def bind(stage, parent, output, builder_receipt, review, fast, root, g):
    inputs = g.Inputs(root)
    source, parent, output, builder_receipt, review = [g.local_path(p, root) for p in
        (stage, parent, output, builder_receipt, review)]
    g.need(not output.exists(), 'bind output already exists; use a fresh versioned stage')
    g.need(not output.is_relative_to(source) and not output.is_relative_to(parent), 'bind output must be outside input stages')
    new = kernel(source, inputs, g)
    ident = new[0]['id']
    g.need(parent == inputs.root / (AP if ident == A else BP), 'standalone canonical parent differs')
    g.need(fast, 'standalone requires pinned FAST policy')
    pfiles, before, _, praw = parent_files(ident, inputs, g)
    rules, diff, _, _ = comparison(before, new, praw, g)
    binding = dict(source_stage=source.relative_to(inputs.root).as_posix(),
        review=review.relative_to(inputs.root).as_posix(), builder_receipt=builder_receipt.relative_to(inputs.root).as_posix(),
        builder_receipt_sha256=g.sha(inputs.read(builder_receipt)))
    allow = dict(version=VERSION, review_binding=binding, parent_sha256sums=g.sha(pfiles['SHA256SUMS']),
        stage_sha256sums='', files=rules, standalone_policy=new[2]['standalone_policy'], fast_path=dict(g.FAST_REFERENCE))
    lane, anchor = g.source_binding(allow, new, inputs)
    g.need(anchor is None, 'standalone cannot inherit historical review')
    reg = dict(version='1.0.0', binding_version=VERSION, kernel_id=ident, parent=str(output / 'parent'),
        parent_sha256sums=allow['parent_sha256sums'], source_stage=binding['source_stage'], external_source=str(source),
        kernel_files=new[2]['kernel_files'], builder_lane=lane, builder_receipt_sha256=binding['builder_receipt_sha256'],
        qualified_for_submission=False, exposure='EXPOSED', clean_oof=False)
    files = {'kernel/' + n:b for n,b in new[3].items()}
    files.update({'registration.json':g.encoded(reg), 'full-diff.txt':diff.encode()})
    files['SHA256SUMS'] = sums(files, g)
    allow['stage_sha256sums'] = g.sha(files['SHA256SUMS'])
    files['allowlist.json'] = g.encoded(allow)
    files['review-request.md'] = (
        'Review request only; not an approval. Inspect full-diff.txt and pinned parent/FAST/checkpoint provenance.\n'
        f'Builder-Lane (dispatcher): {lane}\nReviewer-Lane: <independent dispatched lane>\n'
        f'SHA256SUMS: {allow["stage_sha256sums"]}\nAllowlist-SHA256: {g.sha(files["allowlist.json"])}\n'
        f'Kernel-SHA256: {g.sha(g.encoded(new[2]["kernel_files"]))}\nVERDICT: PENDING\n').encode()
    files.update({'parent/' + n:b for n,b in pfiles.items()})
    inputs.recheck()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(data)
    return dict(status='REVIEW_REQUIRED', stage=str(output), parent=str(output / 'parent'),
        allow=str(output / 'allowlist.json'), review=str(review), sha256sums=allow['stage_sha256sums'], builder_lane=lane)


def evaluate(stage, parent, allow_path, review, root, g):
    inputs = g.Inputs(root)
    stage, parent, allow_path = [g.local_path(p, root) for p in (stage, parent, allow_path)]
    errors, diff, new, old_details, a, b, reviewer = [], '', None, None, {}, {}, None
    def check(action):
        try: return action()
        except (ValueError, OSError, KeyError, TypeError, SyntaxError, zipfile.BadZipFile) as exc:
            errors.append(str(exc))
    def scope():
        nonlocal diff, new, old_details, a, b, reviewer
        new = kernel(stage, inputs, g)
        pfiles, before, old_details, praw = parent_files(new[0]['id'], inputs, g)
        g.need(parent == stage / 'parent', 'standalone gate parent must be nested pinned copy')
        raw, ppaths = raw_kernel(parent, inputs, g)
        g.need(raw == praw, 'standalone parent kernel copy differs')
        for name, data in pfiles.items():
            g.need(inputs.read(parent / name) == data, 'standalone parent copy differs: ' + name)
        pm = g.manifest(parent, inputs, ppaths + [parent / 'registration.json'])[1]
        sm = g.manifest(stage, inputs, new[4] + [stage / 'registration.json', stage / 'full-diff.txt'])[1]
        allow = inputs.json(allow_path)
        g.need(set(allow) == {'version','review_binding','parent_sha256sums','stage_sha256sums','files','standalone_policy','fast_path'}
            and allow['version'] == VERSION, 'invalid standalone allowlist schema/version')
        g.need(allow['parent_sha256sums'] == pm and allow['stage_sha256sums'] == sm, 'standalone manifest binding differs')
        g.need(allow['standalone_policy'] == new[2]['standalone_policy'] and allow['fast_path'] == g.FAST_REFERENCE,
               'standalone/FAST policy differs')
        reg = inputs.json(stage / 'registration.json')
        g.need(reg['kernel_id'] == new[0]['id'] and g.local_path(reg['parent'], root) == parent
            and reg['parent_sha256sums'] == pm and reg['kernel_files'] == new[2]['kernel_files']
            and reg['binding_version'] == VERSION and reg['version'] == '1.0.0'
            and reg['exposure'] == 'EXPOSED' and reg['clean_oof'] is False and reg['qualified_for_submission'] is False,
            'standalone registration differs')
        g.need(g.local_path(allow['review_binding']['source_stage'], root) == g.local_path(reg['external_source'], root)
            and reg['source_stage'] == allow['review_binding']['source_stage']
            and reg['builder_receipt_sha256'] == allow['review_binding']['builder_receipt_sha256'], 'standalone source binding differs')
        rules, diff, a, b = comparison(before, new, praw, g)
        g.need(allow['files'] == rules, 'standalone exact allowlist differs')
        g.need(inputs.read(stage / 'full-diff.txt') == diff.encode(), 'standalone full diff differs')
        lane, _ = g.source_binding(allow, new, inputs)
        g.need(reg['builder_lane'] == lane, 'standalone builder lane differs')
        reviewer = g.review_check(review, allow, sm, inputs, new, allow_path)
    check(scope)
    inputs.read(Path(g.__file__))
    inputs.read(Path(__file__))
    inputs.read(Path(h.__file__))
    check(inputs.recheck)
    return dict(fixture_only=bool(new and new[2]['fixture_only']), gate_version=VERSION,
        status='FAIL' if errors else 'PASS', errors=sorted(set(errors)), stage=str(stage), parent=str(parent),
        kernel_id=new[0]['id'] if new else None, reviewer_lane=reviewer, input_sha256=inputs.hashes(),
        parent_inventory=old_details, stage_inventory=new[2] if new else None,
        comparison_sha256=dict(parent={n:g.sha(v) for n,v in a.items()}, stage={n:g.sha(v) for n,v in b.items()}),
        diff_sha256=g.sha(diff.encode()), full_diff=diff)
