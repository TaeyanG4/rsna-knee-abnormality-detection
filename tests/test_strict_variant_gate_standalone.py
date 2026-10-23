"""v1.3.0 real bound stages, static hashes only; isolated test dispatches."""
import copy
import io
import pickle
import zipfile

import pytest

from tools import strict_variant_gate as g
from tools import strict_variant_gate_standalone as s
from tools import strict_variant_gate_hb151 as h
from tools import kaggle_submit_once as ks
from test_strict_variant_gate import dispatched, put

SOURCES = [g.ROOT / '.local/scratch/hb157-rkd/H-B-157/v1.0.2/killtest-v1',
           g.ROOT / 'artifacts/experiments/HB156_KILLTEST/v1.0.0']


@pytest.fixture(params=SOURCES, ids=['R-KD', 'ConvNeXt-L2'])
def source(request):
    return request.param


def endorse(tmp_path, output, review, lane='standalone-reviewer'):
    allow = g.parse((output / 'allowlist.json').read_bytes())
    dispatched(tmp_path, lane, review, (
        f'Reviewer-Lane: {lane}\nSHA256SUMS: ' + allow['stage_sha256sums'] +
        '\nAllowlist-SHA256: ' + g.sha((output / 'allowlist.json').read_bytes()) +
        '\nVERDICT: PASS\n').encode())


@pytest.fixture
def bound(source, tmp_path, monkeypatch):
    original = g.dispatch
    def isolated(path, inputs):
        scope = g.Inputs(tmp_path)
        result = original(g.local_path(path, inputs.root), scope)
        inputs.data.update(scope.data)
        return result
    monkeypatch.setattr(g, 'dispatch', isolated)
    info = g.kernel(source, g.Inputs(g.ROOT))
    builder = dispatched(tmp_path, 'standalone-builder', tmp_path / 'build.md',
        ('Kernel-SHA256: ' + g.sha(g.encoded(info[2]['kernel_files'])) + '\n').encode())
    output, review = tmp_path / 'bound', tmp_path / 'review.md'
    parent = g.ROOT / (s.AP if info[0]['id'] == s.A else s.BP)
    assert g.main(['bind', '--stage', str(source), '--parent', str(parent), '--output', str(output),
        '--builder-receipt', str(builder), '--review', str(review), '--fast']) == 0
    endorse(tmp_path, output, review)
    return dict(stage=output, parent=output / 'parent', allow_path=output / 'allowlist.json', review=review)


def test_real_bound_accept_bind_review_check_only_and_offline_consumer(bound, monkeypatch):
    result = g.evaluate(**bound)
    assert result['status'] == 'PASS', result['errors']
    assert result['gate_version'] == '1.3.0' and result['fixture_only'] is False
    assert 'IDENTICAL' in result['full_diff'] and 'REGISTERED_CHANGE' in result['full_diff']
    assert 'raw-parent/release.ipynb' in result['full_diff']
    assert 'raw-parent/kernel-metadata.json' in result['full_diff']
    argv = [v for k,p in bound.items() for v in ['--' + ('allow' if k == 'allow_path' else k), str(p)]]
    assert g.main(argv + ['--check-only']) == 0
    # Publish ONLY to the isolated pytest folder, never the production gate path.
    receipt = bound['stage'].parent / 'test-only-admission.json'
    g.write_receipt(receipt, result)
    path = ks.ROOT / '.local/state/gates' / (result['kernel_id'].split('/')[1] + '.json')
    read = ks._json_evidence
    monkeypatch.setattr(ks, '_json_evidence', lambda p, hashes: result if p == path else read(p, hashes))
    assert ks._strict_gate(result['kernel_id'], {'gate_sha256':'fixture'}, {str(path):'fixture'}) == result
    result['fixture_only'] = True
    with pytest.raises(ValueError, match='fixture-only'):
        ks._strict_gate(result['kernel_id'], {'gate_sha256':'fixture'}, {str(path):'fixture'})
    result['fixture_only'] = False
    for helper in ('strict_variant_gate_standalone.py', 'strict_variant_gate_hb151.py'):
        saved = result['input_sha256'].pop('tools/' + helper)
        with pytest.raises(ValueError, match='required stage/tool hashes'):
            ks._strict_gate(result['kernel_id'], {'gate_sha256':'fixture'}, {str(path):'fixture'})
        result['input_sha256']['tools/' + helper] = saved
    result['gate_version'] = '1.2.0'
    with pytest.raises(ValueError, match='unsupported parent identity'):
        ks._strict_gate(result['kernel_id'], {'gate_sha256':'fixture'}, {str(path):'fixture'})


@pytest.mark.parametrize('mutation', ['cell-byte','launch','title','placeholder','placeholder-type',
    'fast','top','outputs','checkpoint','parent','dataset','donor','metadata-type','notebook-type','sidecar-type'])
def test_semantic_changes_rejected_even_with_refreshed_receipt_hashes(source, mutation):
    inputs = g.Inputs(g.ROOT)
    rec = inputs.json(source / 'build-receipt.json')
    nbpath, mpath = source / 'kernel/release.ipynb', source / 'kernel/kernel-metadata.json'
    nb, meta = inputs.json(nbpath), inputs.json(mpath)
    if mutation == 'cell-byte':
        nb['cells'][1]['source'] = ''.join(nb['cells'][1]['source']) + 'x'
    if mutation == 'launch':
        nb['cells'][-1]['source'] = ''.join(nb['cells'][-1]['source']) + '\nimport subprocess\n'
    if mutation == 'fast': nb['cells'][0]['source'] = ''.join(nb['cells'][0]['source']) + '\nFAST=True\n'
    if mutation == 'top': nb['metadata']['unregistered'] = True
    if mutation == 'outputs': nb['cells'][-1]['outputs'] = [{'output_type':'stream','name':'stdout','text':'x'}]
    if mutation == 'title': meta['title'] = 'wrong-title'
    if mutation == 'metadata-type': meta.update(enable_gpu=1, enable_internet=0)
    if mutation == 'notebook-type': nb['nbformat'] = float(nb['nbformat'])
    if mutation == 'sidecar-type':
        if meta['id'] == s.A: meta['is_private'] = 1
        else:
            p = source / 'dataset' / (h.NAME + '.json'); sidecar = inputs.json(p)
            sidecar['seed'] = 42.0; inputs.data[p] = g.encoded(sidecar)
    if mutation.startswith('placeholder'): rec['placeholder'] = True if mutation == 'placeholder' else 0
    if mutation == 'checkpoint':
        if meta['id'] == s.A: rec['checkpoint_sha256'] = '0' * 64
        else:
            binding = inputs.json(source / 'export-binding.json'); binding['sha256'] = s.B_PLACEHOLDER
            inputs.data[source / 'export-binding.json'] = g.encoded(binding)
    if mutation == 'parent':
        p = g.ROOT / (s.AP if meta['id'] == s.A else s.BP + '/kernel/release.ipynb')
        inputs.data[p] = inputs.read(p) + b'\n'
    if mutation == 'donor':
        p = g.ROOT / g.FAST_REFERENCE['stage'] / 'kernel/release.ipynb'
        inputs.data[p] = inputs.read(p) + b'\n'
    if mutation == 'dataset':
        p = source / 'dataset/dataset-metadata.json'; dm = inputs.json(p); dm['id'] = 'wrong/dataset'
        inputs.data[p] = g.encoded(dm)
    inputs.data[nbpath], inputs.data[mpath] = g.encoded(nb), g.encoded(meta)
    rec['notebook_sha256'] = g.sha(inputs.data[nbpath])
    if meta['id'] == s.B:
        for name in ('kernel/release.ipynb','kernel/kernel-metadata.json','dataset/dataset-metadata.json','export-binding.json'):
            rec['artifacts'][name] = g.sha(inputs.read(source / name))
    inputs.data[source / 'build-receipt.json'] = g.encoded(rec)
    smoke = inputs.json(source / 'cpu-smoke.json')
    smoke['build_receipt_sha256'] = g.sha(g.encoded(rec)); smoke['notebook_sha256'] = g.sha(inputs.data[nbpath])
    inputs.data[source / 'cpu-smoke.json'] = g.encoded(smoke)
    with pytest.raises(g.GateError): g.kernel(source, inputs)


@pytest.mark.parametrize('mutation', ['runtime','serving','schema','manifest','nested'])
def test_b_archive_delta_cannot_be_authorized_by_resealing(monkeypatch, mutation):
    source = SOURCES[1]; inputs = g.Inputs(g.ROOT); original = h.archive
    rec = inputs.json(source / 'build-receipt.json')
    hashes, small = original(source / 'dataset' / h.ARCHIVE, rec['artifacts']['dataset/' + h.ARCHIVE], inputs, g)
    if mutation in ('runtime','serving','nested'):
        name = 'public-serving.py' if mutation == 'serving' else h.RUNTIME
        small[name] += b'\nimport multiprocessing # unregistered\n'
        if mutation == 'nested': small[name] += b'EXTRA_B64 = "eA=="\n'
    else:
        name = 'release-manifest.json'; m = g.parse(small[name])
        if mutation == 'schema': m['own_state_schema']['unregistered'] = [1]
        else: m['dataset'] = 'wrong/data'
        small[name] = h.wire(m)
    hashes[name] = g.sha(small[name]); rec['files'] = hashes
    inputs.data[source / 'build-receipt.json'] = g.encoded(rec)
    def changed(path, pin, scope, gate):
        if path == source / 'dataset' / h.ARCHIVE:
            return hashes, small
        return original(path, pin, scope, gate)
    monkeypatch.setattr(h, 'archive', changed)
    with pytest.raises(g.GateError, match='schema differs|exceeds registered delta'): g.kernel(source, inputs)


@pytest.mark.parametrize('mutation', ['allowlist','full-diff','same-lane','missing-review','parent','builder-digest','manifest-review'])
def test_bound_protocol_fail_closed(bound, tmp_path, mutation):
    if mutation == 'allowlist':
        allow = g.parse(bound['allow_path'].read_bytes()); allow['files'] = {}
        put(bound['allow_path'], allow); endorse(tmp_path, bound['stage'], bound['review'])
    if mutation == 'full-diff': put(bound['stage'] / 'full-diff.txt', b'No changes\n')
    if mutation == 'same-lane': endorse(tmp_path, bound['stage'], bound['review'], 'standalone-builder')
    if mutation == 'missing-review': bound['review'] = tmp_path / 'absent.md'
    if mutation == 'parent': put(bound['parent'] / 'kernel/release.ipynb', b'{}')
    if mutation == 'builder-digest':
        dispatched(tmp_path, 'standalone-builder', tmp_path / 'build.md', b'Kernel-SHA256: '+ b'0'*64+b'\n')
        allow = g.parse(bound['allow_path'].read_bytes())
        allow['review_binding']['builder_receipt_sha256'] = g.sha((tmp_path / 'build.md.receipt.json').read_bytes())
        put(bound['allow_path'], allow)
        reg = g.parse((bound['stage'] / 'registration.json').read_bytes())
        reg['builder_receipt_sha256'] = allow['review_binding']['builder_receipt_sha256']
        put(bound['stage'] / 'registration.json', reg)
    if mutation == 'manifest-review': put(bound['review'], b'VERDICT: PASS\n')
    if mutation in ('full-diff', 'builder-digest'):
        names = ['kernel/release.ipynb','kernel/kernel-metadata.json','registration.json','full-diff.txt']
        manifest = s.sums({n:(bound['stage'] / n).read_bytes() for n in names}, g)
        put(bound['stage'] / 'SHA256SUMS', manifest)
        allow = g.parse(bound['allow_path'].read_bytes())
        allow['stage_sha256sums'] = g.sha(manifest)
        put(bound['allow_path'], allow)
        endorse(tmp_path, bound['stage'], bound['review'])
    result = g.evaluate(**bound)
    assert result['status'] == 'FAIL', mutation
    if mutation == 'full-diff': assert any('full diff differs' in e for e in result['errors'])
    if mutation == 'builder-digest': assert any('Kernel-SHA256' in e for e in result['errors']), result['errors']


@pytest.mark.parametrize('marker', ['placeholder','fixture','hb156_killtest_placeholder','hb151_killtest_placeholder'])
def test_checkpoint_pickle_markers_rejected_without_deserialization(marker):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z: z.writestr('archive/data.pkl', pickle.dumps({marker:True}))
    buf.seek(0)
    with pytest.raises(g.GateError, match='placeholder checkpoint metadata'): s.checkpoint_metadata(buf, g)


@pytest.mark.parametrize('mode', ['extra','missing','duplicate'])
def test_archive_inventory_refused(tmp_path, mode):
    names = [h.NAME,h.RUNTIME,'public-serving.py','release-manifest.json']
    if mode == 'extra': names.append('extra.py')
    if mode == 'missing': names.pop()
    if mode == 'duplicate': names[-1] = names[0]
    path = tmp_path / 'assets.bin'
    with zipfile.ZipFile(path, 'w') as z:
        for name in names: z.writestr(name, b'x')
    with pytest.raises(g.GateError, match='member set differs'):
        h.archive(path, g.sha(path.read_bytes()), g.Inputs(g.ROOT), g)


def test_input_file_set_and_digest_races(tmp_path):
    p = tmp_path / 'dataset/file.bin'; put(p, b'old')
    inputs = g.Inputs(g.ROOT); inputs.digest(p); put(p, b'new')
    with pytest.raises(g.GateError, match='input changed'): inputs.recheck()
    inputs = g.Inputs(g.ROOT); inputs.folders[p.parent] = [p]
    put(p.parent / 'unexpected.bin', b'x')
    with pytest.raises(g.GateError, match='file set changed'): inputs.recheck()


def test_schema_pin_matches_bound_export_and_synthetic_parent_pin():
    inputs = g.Inputs(g.ROOT); rec = inputs.json(SOURCES[1] / 'build-receipt.json')
    _, small = h.archive(SOURCES[1] / 'dataset' / h.ARCHIVE, rec['artifacts']['dataset/' + h.ARCHIVE], inputs, g)
    assert g.sha(g.encoded(g.parse(small['release-manifest.json'])['own_state_schema'])) == s.B_SCHEMA
    assert g.sha(g.encoded(s.a_meta(True))) == ks.STANDALONE_PARENT_PINS[s.A][1]['kernel-metadata.json']


@pytest.mark.parametrize('mutation', ['metadata-pin', 'notebook-pin', 'cross-child'])
def test_consumer_rejects_resealed_parent_and_cross_child(bound, mutation):
    result = g.evaluate(**bound)
    assert result['status'] == 'PASS', result['errors']
    parent = bound['parent']
    if mutation == 'cross-child':
        result['kernel_id'] = s.B if result['kernel_id'] == s.A else s.A
    else:
        name = 'kernel-metadata.json' if mutation == 'metadata-pin' else 'release.ipynb'
        path = parent / 'kernel' / name
        path.write_bytes(path.read_bytes() + b'\n')
    # Refresh every recorded digest: refusal must come from the pinned identity,
    # not merely a stale receipt hash.
    recorded = {ks._local(p) for p in result['input_sha256']}
    hashes = {str(p): g.sha(p.read_bytes()) for p in recorded}
    reg = g.parse((bound['stage'] / 'registration.json').read_bytes())
    with pytest.raises(ValueError, match='hidden-proven pins|unsupported parent identity'):
        ks._registered_parent(result, reg, recorded, hashes)
