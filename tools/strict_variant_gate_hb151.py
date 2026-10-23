"""v1.2.0: static, exact EXP143v2 -> HB151 v1.0.2 external-archive contract.

No notebook/builder execution, checkpoint deserialization, network or GPU use.
Archive members are streamed, never extracted. Serving source is immutable;
only the three registered provenance expressions may change in the loader.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import zipfile

PARENT = '.local/exp152-own0949-v1/stage'
PARENT_ID = 'taeyangg4/knee-exp152-own0949-v2'
CHILD_ID = 'taeyangg4/knee-hb151-pl-coat-v1'
DATASET = 'taeyangg4/knee-hb151-pl-coat-ckpt-v1'
NAME = 'exp143-v2-e14-e11-e12-bn.pt'
ARCHIVE = 'exp152-assets.zip.bin'
RUNTIME = 'exp152_own0949_release.py'
PINS = {
    'own/release.ipynb': '1045c22cd6de8e28dfeb60fc47e62b0356d069487d520c63e67421586aadb08a',
    'own/kernel-metadata.json': 'cff7e3ea949caca229869d379c26b237696e7fe2c84175af38b47114b91daa58',
    'build-receipt.json': '672ebedf84ebd8fb2f3e1dcf016995f5f44a20fb5184017dfe94b5d30ea0c198',
    'assets/release-manifest.json': 'd85946bd08c6087ef170e705bf695260c5f398e57479bd30b48ba65e0838b295',
    'assets/' + RUNTIME: 'ebb8e06734b72f40f090389b23d8767137c8eb6c92dee547367c1f18188f94e2',
    'assets/public-serving.py': '25ef6ab9d2e942ac291433a498246d0507481c2aefa5630cdd601d0a36852cf2',
}
SOURCES = {
    'seed42': (42, '.local/scratch/astra-b4/H-B-151/v3.0.2/S42', 'ef0bb758076fdfbc78d1514252c23ac2dc6b862e0b6a36f6bf3ad8e2ba13d9d8'),
    'seed43': (43, 'stage-seed43-v1/S43', '4ca1bfe8036871284658bb85b6c2742e0b480690a75b037b5e1da5a6513f163c'),
}


def wire(obj):
    # The serving producer's serialization is part of its manifest pin.
    return (json.dumps(obj, indent=2, ensure_ascii=False) + '\n').encode()


def inventory(raw, g):
    nb = g.parse(raw)
    result = {'notebook/top.json': g.encoded({k: v for k, v in nb.items() if k != 'cells'})}
    for i, cell in enumerate(nb['cells']):
        source = cell['source']
        g.need(isinstance(source, str) or isinstance(source, list) and all(isinstance(s, str) for s in source), 'invalid cell source')
        result[f'notebook/cells/{i}/source.txt'] = ''.join(source).encode()
        result[f'notebook/cells/{i}/fields.json'] = g.encoded({k: v for k, v in cell.items() if k != 'source'})
    return result


def archive(path, expected_sha, inputs, g):
    g.need(inputs.digest(path) == expected_sha, 'external archive SHA mismatch')
    hashes, small = {}, {}
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        names = {NAME, RUNTIME, 'public-serving.py', 'release-manifest.json'}
        g.need(len(members) == 4 and {m.filename for m in members} == names, 'external archive member set differs')
        for info in members:
            g.safe_name(info.filename)
            g.need(not info.is_dir() and (info.external_attr >> 16) & 0o170000 != 0o120000, 'linked external member')
            g.need(info.file_size <= (2 * 1024**3 if info.filename == NAME else g.MAX_BYTES), 'oversize external member')
            with z.open(info) as f:
                hashes[info.filename] = hashlib.file_digest(f, 'sha256').hexdigest()
            if info.filename != NAME:
                small[info.filename] = z.read(info)
    return hashes, small


def parent_material(inputs, g):
    source = inputs.root / PARENT
    for name, pin in PINS.items():
        g.need(g.sha(inputs.read(source / name)) == pin, 'EXP143 parent pin differs: ' + name)
    receipt = inputs.json(source / 'build-receipt.json')
    hashes, small = archive(source / 'dataset' / ARCHIVE, receipt['archive_sha256'], inputs, g)
    manifest = inputs.json(source / 'assets/release-manifest.json')
    g.need(hashes['release-manifest.json'] == PINS['assets/release-manifest.json'], 'parent archive manifest differs')
    g.need(all(hashes[n] == h for n, h in manifest['asset_sha256'].items()), 'parent archive assets differ')
    g.need(all(data == inputs.read(source / 'assets' / n) for n, data in small.items()), 'parent extracted code differs')
    return source, receipt, manifest, hashes, small


def kernel(stage, inputs, raw, paths, g):
    """Recognize only the two pinned families; return a complete static inventory."""
    canonical, parent_receipt, parent_manifest, parent_hashes, parent_small = parent_material(inputs, g)
    meta = g.parse(raw['kernel-metadata.json'])
    is_parent = meta['id'] == PARENT_ID
    reg = inputs.json(stage / 'registration.json') if (stage / 'registration.json').is_file() else {}
    source = g.local_path(reg.get('external_source', str(stage)), inputs.root)
    if is_parent:
        g.need(source == canonical, 'EXP143 parent source path differs')
        g.need({n: g.sha(b) for n, b in raw.items()} == {n.removeprefix('own/'): h for n, h in PINS.items() if n.startswith('own/')}, 'EXP143 parent kernel differs')
        hashes, small, asha = parent_hashes, parent_small, parent_receipt['archive_sha256']
        policy, fixture = None, False
    else:
        g.need(meta['id'] == CHILD_ID and set(raw) == {'release.ipynb', 'kernel-metadata.json'}, 'unsupported external kernel')
        g.need(all(inputs.read(source / 'kernel' / n) == data for n, data in raw.items()), 'external source kernel differs')
        rec = inputs.json(source / 'build-receipt.json')
        binding = inputs.json(source / 'export-binding.json')
        g.need(rec['version'] == '1.0.2' and rec['parent_ref'] == '57005140' and rec['parent_pins'] == PINS, 'HB151 bind receipt identity differs')
        g.need(rec['export_binding_sha256'] == g.sha(inputs.read(source / 'export-binding.json')), 'HB151 bind receipt hash differs')
        g.need(rec['notebook_sha256'] == g.sha(raw['release.ipynb']) and rec['metadata_sha256'] == g.sha(raw['kernel-metadata.json']), 'HB151 bound kernel hash differs')
        g.need(binding['source'] in SOURCES, 'unsupported HB151 source')
        seed, run, protocol = SOURCES[binding['source']]
        g.need((binding['seed'], binding['run_dir'], binding['protocol_sha256']) == (seed, run, protocol), 'HB151 export provenance differs')
        g.need(binding['protocol_manifest_sha256'] == protocol and binding['serving_name'] == NAME, 'HB151 protocol/filename differs')
        g.need(binding['exposure'] == rec['exposure'] == 'EXPOSED' and binding['clean_oof'] is rec['clean_oof'] is False, 'HB151 exposure differs')
        g.need(type(binding['placeholder']) is bool and type(rec['fixture']) is bool and rec['placeholder'] == binding['placeholder'], 'HB151 fixture identity differs')
        fixture = rec['fixture'] or binding['placeholder']
        asha = rec['archive_sha256']
        hashes, small = archive(source / 'dataset' / ARCHIVE, asha, inputs, g)
        g.need(hashes == rec['files'] and hashes[NAME] == binding['sha256'] == rec['checkpoint_sha256'], 'HB151 checkpoint hash differs')
        g.need(inputs.read(source / 'dataset' / (NAME + '.sha256')).decode().split() == [hashes[NAME], NAME], 'HB151 checkpoint pin sidecar differs')
        g.need(inputs.json(source / 'dataset' / (NAME + '.json')) == binding, 'HB151 checkpoint provenance sidecar differs')
        dmeta = inputs.json(source / 'dataset/dataset-metadata.json')
        g.need(dmeta['id'] == DATASET and dmeta['title'] == DATASET.split('/')[1] and meta['dataset_sources'] == [DATASET], 'unpinned HB151 dataset')
        g.need(g.sha(inputs.read(source / 'dataset/dataset-metadata.json')) == rec['dataset_metadata_sha256'], 'HB151 dataset metadata hash differs')
        runtime = parent_small[RUNTIME]
        for a, b in [
            (b'own_meta(ck, manifest["protocol_sha256"])', b'own_meta(ck, manifest["protocol_sha256"], run_dir=manifest["hb151_export"]["run_dir"], seed=manifest["hb151_export"]["seed"])'),
            (b'def own_meta(ck, protocol_sha, run_dir=RUN):', b'def own_meta(ck, protocol_sha, run_dir=RUN, seed=42):'),
            (b'swa_epochs_1based=EPOCHS, seed=42)', b'swa_epochs_1based=EPOCHS, seed=seed)'),
        ]:
            g.need(runtime.count(a) == 1, 'parent provenance expression differs')
            runtime = runtime.replace(a, b)
        g.need(small[RUNTIME] == runtime and small['public-serving.py'] == parent_small['public-serving.py'], 'modified serving file or runtime outside provenance exception')
        expected_manifest = copy.deepcopy(parent_manifest)
        expected_manifest['checkpoints'][NAME]['sha256'] = hashes[NAME]
        expected_manifest['asset_sha256'][NAME] = hashes[NAME]
        expected_manifest['asset_sha256'][RUNTIME] = g.sha(runtime)
        expected_manifest['protocol_sha256'] = protocol
        expected_manifest['hb151_export'] = binding
        g.need(small['release-manifest.json'] == wire(expected_manifest), 'HB151 manifest exceeds registered delta')
        # Exact metadata delta; no extra key or protected policy can change.
        expected_meta = inputs.json(canonical / 'own/kernel-metadata.json')
        expected_meta.update(id=CHILD_ID, title=CHILD_ID.split('/')[1], dataset_sources=[DATASET])
        g.need(meta == expected_meta, 'HB151 metadata exceeds registered delta')
        policy = {
            'parent_kernel': str(canonical / 'own/release.ipynb'), 'parent_kernel_sha256': PINS['own/release.ipynb'],
            'parent_ref': '57005140', 'dataset_slug': DATASET, 'checkpoint_sha256': hashes[NAME],
            'bind_receipt': str(source / 'build-receipt.json'), 'bind_receipt_sha256': g.sha(inputs.read(source / 'build-receipt.json')),
            'export_binding': str(source / 'export-binding.json'), 'export_binding_sha256': rec['export_binding_sha256'],
            'archive': str(source / 'dataset' / ARCHIVE), 'archive_sha256': asha,
            'manifest_sha256': hashes['release-manifest.json'], 'fixture_only': fixture,
        }
    inv = inventory(raw['release.ipynb'], g)
    inv.update({'external/' + n: b for n, b in small.items()})
    inv['external/' + NAME + '.sha256'] = (hashes[NAME] + '\n').encode()
    inv.update({'kernel/' + n: b for n, b in raw.items() if n != 'release.ipynb'})
    details = dict(archive_sha256=asha, embedded=hashes, kernel_files={n: g.sha(b) for n, b in raw.items()},
                   external_policy=policy, fixture_only=fixture)
    inputs.read(Path(__file__))
    return meta, inv, details, raw, paths


def fast_inventory(old, new, policy, inputs, g):
    g.need(old[0]['id'] == PARENT_ID and new[0]['id'] == CHILD_ID, 'external parent/child identity differs')
    # Reuse the immutable reference authentication, without weakening C85 checks.
    g.fast_reference(policy, inputs)
    before, after = [g.parse(x[3]['release.ipynb']) for x in (old, new)]
    g.need(len(before['cells']) == 5 and len(after['cells']) == 6, 'HB151 FAST requires exactly 5-to-6 cells')
    cell = after['cells'][0]
    g.need(g.sha(g.encoded(cell)) == policy['cell0_object_sha256'] and g.sha(''.join(cell['source']).encode()) == policy['cell0_sha256'], 'FAST cell 0 hash mismatch')
    g.need(all(c['cell_type'] == 'markdown' for c in before['cells'][:4]) and before['cells'][4]['cell_type'] == 'code', 'HB151 parent cell layout differs')
    expected = copy.deepcopy(before)
    source = ''.join(before['cells'][4]['source'])
    replacements = {
        'knee-exp152-own0949-v2-assets': DATASET.split('/')[1],
        old[2]['archive_sha256']: new[2]['archive_sha256'],
        old[2]['embedded']['release-manifest.json']: new[2]['embedded']['release-manifest.json'],
        old[2]['embedded'][RUNTIME]: new[2]['embedded'][RUNTIME],
    }
    for a, b in replacements.items():
        g.need(a in source, 'parent notebook pin missing')
        source = source.replace(a, b)
    expected['cells'][4]['source'] = 'if not FAST:\n' + ''.join('    ' + line for line in source.splitlines(True))
    expected['cells'].insert(0, cell)
    g.need(after == expected, 'HB151 notebook exceeds pins/exact FAST guard delta')
    after['cells'] = after['cells'][1:]
    after['cells'][4]['source'] = source
    result = inventory(g.encoded(after), g)
    result.update({n: b for n, b in new[1].items() if not n.startswith('notebook/')})
    return result


def parent_files(inputs, g):
    source, _, _, _, _ = parent_material(inputs, g)
    files = {'kernel/' + n: inputs.read(source / 'own' / n) for n in ('release.ipynb', 'kernel-metadata.json')}
    files['registration.json'] = g.encoded(dict(version='1', binding_version=g.VERSION, kernel_id=PARENT_ID,
                                               external_source=str(source), parent_ref='57005140'))
    files['SHA256SUMS'] = ''.join(g.sha(b) + '  ' + n + '\n' for n, b in sorted(files.items())).encode()
    return files
