"""Real seed43 CPU fixture and exact external-archive gate; no account access."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import strict_variant_gate as g
from tools import strict_variant_gate_hb151 as h
from tools import hb151_killtest_v102 as b
from tools import hb151_killtest_v102_fixtures as f
from tools import kaggle_submit_once as ks
from test_strict_variant_gate import dispatched, put

AREA = g.ROOT / '.local/scratch/astra-b5/strict-variant-gate-v120'


@pytest.fixture(scope='module')
def source():
    with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '-1'}), patch.object(b, 'SCRATCH', AREA), patch.object(b, 'OUT', AREA / 'source-seed43'):
        ck, _ = f.placeholder_export('seed43')
        result = b.bind(ck, source='seed43', fixture=True)
        assert result['status'] == 'PASS_CPU_SCOPE'
        assert b.validate()['status'] == 'PASS_CPU_SCOPE'
        assert f.loader_smoke(AREA / 'loader-seed43')['status'] == 'PASS'
        csv = (AREA / 'real-test-csv/working/submission.csv').resolve()
        assert csv.is_relative_to(AREA.resolve())
        if csv.exists(): b.retry(csv.unlink)
        assert f.run_case(AREA / 'real-test-csv', real_manifest=True)['status'] == 'PASS'
        put(AREA / 'real-cpu-smoke.json', dict(status='PASS', source='seed43', fixture_only=True,
            validate='PASS', real_test_csv='PASS', real_cpu_model_load='PASS', gpu=False))
        yield b.OUT


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
    builder = dispatched(tmp_path, 'fixture-builder', tmp_path / 'build.md',
        ('Kernel-SHA256: ' + g.sha(g.encoded(info[2]['kernel_files'])) + '\n').encode())
    review = tmp_path / 'review.md'
    output = tmp_path / 'bound'
    assert g.main(['bind', '--stage', str(source), '--parent', str(g.ROOT / h.PARENT),
        '--output', str(output), '--builder-receipt', str(builder), '--review', str(review), '--fast']) == 0
    assert (output / 'full-diff.txt').is_file()
    allow = g.parse((output / 'allowlist.json').read_bytes())
    dispatched(tmp_path, 'fixture-reviewer', review,
        ('Reviewer-Lane: fixture-reviewer\nSHA256SUMS: ' + allow['stage_sha256sums'] +
         '\nAllowlist-SHA256: ' + g.sha((output / 'allowlist.json').read_bytes()) + '\nVERDICT: PASS\n').encode())
    return dict(stage=output, parent=output / 'parent', allow_path=output / 'allowlist.json', review=review)


def test_real_seed43_bound_gate_and_cli_pass_fixture_publication_refused(bound, capsys):
    result = g.evaluate(**bound)
    assert result['status'] == 'PASS', result['errors']
    assert result['fixture_only'] is True
    policy = g.parse(bound['allow_path'].read_bytes())['external_archive']
    assert policy['parent_kernel_sha256'] == h.PINS['own/release.ipynb']
    assert policy['dataset_slug'] == h.DATASET
    assert policy['checkpoint_sha256'] == result['stage_inventory']['embedded'][h.NAME]
    argv = [v for k, p in bound.items() for v in ['--' + ('allow' if k == 'allow_path' else k), str(p)]]
    assert g.main(argv + ['--check-only']) == 0
    assert g.main(argv) == 1
    assert 'fixture-only gate cannot publish' in capsys.readouterr().out
    with pytest.raises(g.GateError, match='fixture-only'):
        g.write_receipt(bound['stage'] / 'not-admission.json', result)
    put(AREA / 'gate-fixture-result.json', result)


@pytest.mark.parametrize('mutation', ['serving', 'runtime', 'dataset', 'fast', 'markdown', 'guard', 'manifest', 'provenance', 'parent'])
def test_semantic_mutations_cannot_be_blessed(source, monkeypatch, mutation):
    inputs = g.Inputs(g.ROOT)
    original = h.archive
    rec = inputs.json(source / 'build-receipt.json')
    raw = {p.name: inputs.read(p) for p in (source / 'kernel').iterdir()}
    if mutation in ('serving', 'runtime', 'manifest'):
        def mutated(path, pin, scope, gate):
            hashes, small = original(path, pin, scope, gate)
            if path == source / 'dataset' / h.ARCHIVE:
                name = {'serving': 'public-serving.py', 'runtime': h.RUNTIME, 'manifest': 'release-manifest.json'}[mutation]
                small[name] += b'\n# unauthorized change\n'
                hashes[name] = g.sha(small[name])
                rec['files'] = hashes
                inputs.data[source / 'build-receipt.json'] = g.encoded(rec)
            return hashes, small
        monkeypatch.setattr(h, 'archive', mutated)
    if mutation == 'dataset':
        meta = g.parse(raw['kernel-metadata.json']); meta['dataset_sources'] = ['taeyangg4/unpinned']
        raw['kernel-metadata.json'] = g.encoded(meta)
        rec['metadata_sha256'] = g.sha(raw['kernel-metadata.json'])
    if mutation in ('fast', 'markdown', 'guard'):
        nb = g.parse(raw['release.ipynb'])
        if mutation == 'fast': nb['cells'][0]['source'].append('\nFAST=True\n')
        if mutation == 'markdown': nb['cells'][1]['source'] += 'unauthorized'
        if mutation == 'guard': nb['cells'][5]['source'] = nb['cells'][5]['source'].replace('if not FAST:', 'if True:')
        raw['release.ipynb'] = g.encoded(nb)
        rec['notebook_sha256'] = g.sha(raw['release.ipynb'])
    if mutation == 'provenance':
        binding = inputs.json(source / 'export-binding.json'); binding['protocol_sha256'] = '0' * 64
        inputs.data[source / 'export-binding.json'] = g.encoded(binding)
        rec['export_binding_sha256'] = g.sha(g.encoded(binding))
    if mutation == 'parent':
        p = g.ROOT / h.PARENT / 'own/release.ipynb'
        inputs.data[p] = inputs.read(p) + b'\n'
    for name, data in raw.items(): inputs.data[source / 'kernel' / name] = data
    inputs.data[source / 'build-receipt.json'] = g.encoded(rec)
    with pytest.raises(g.GateError):
        new = g.kernel(source, inputs)
        old = g.kernel(g.ROOT / h.PARENT, inputs)
        g.fast_inventory(old, new, g.FAST_REFERENCE, inputs)


@pytest.mark.parametrize('mutation', ['missing-policy', 'old-schema', 'dataset', 'checkpoint', 'parent-path', 'fast-policy', 'parent-registration'])
def test_external_policy_is_mandatory_and_hash_bound(bound, mutation):
    allow = g.parse(bound['allow_path'].read_bytes())
    if mutation == 'missing-policy': allow.pop('external_archive')
    if mutation == 'old-schema': allow['version'] = '1.1.0'
    if mutation == 'dataset': allow['external_archive']['dataset_slug'] = 'u/unpinned'
    if mutation == 'checkpoint': allow['external_archive']['checkpoint_sha256'] = '0' * 64
    if mutation == 'parent-path': allow['external_archive']['parent_kernel'] = 'wrong'
    if mutation == 'fast-policy': allow.pop('fast_path')
    if mutation == 'parent-registration':
        reg = g.parse((bound['stage'] / 'registration.json').read_bytes()); reg['parent'] = str(bound['stage'])
        put(bound['stage'] / 'registration.json', reg)
    put(bound['allow_path'], allow)
    result = g.evaluate(**bound)
    assert result['status'] == 'FAIL'
    assert any('external' in e for e in result['errors']), result['errors']


def test_submission_refuses_generated_fixture_gate(bound, monkeypatch):
    result = g.evaluate(**bound)
    assert result['status'] == 'PASS'
    gate_path = ks.ROOT / '.local/state/gates/knee-hb151-pl-coat-v1.json'
    monkeypatch.setattr(ks, '_json_evidence', lambda path, hashes: result)
    with pytest.raises(ValueError, match='fixture-only'):
        ks._strict_gate(h.CHILD_ID, {'gate_sha256': 'test'}, {str(gate_path): 'test'})


def test_large_input_recheck_detects_mutation(tmp_path):
    p = tmp_path / 'archive.bin'; p.write_bytes(b'first')
    inputs = g.Inputs(g.ROOT); inputs.digest(p); p.write_bytes(b'second')
    with pytest.raises(g.GateError, match='input changed'):
        inputs.recheck()


def test_external_review_diff_cannot_be_replaced(bound):
    (bound['stage'] / 'full-diff.txt').write_text('Nothing changed.\n', encoding='utf-8')
    result = g.evaluate(**bound)
    assert result['status'] == 'FAIL'
    assert any('full diff differs' in e for e in result['errors'])
