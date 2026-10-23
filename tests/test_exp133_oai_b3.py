"""CPU-only tiny fakes. No external code execution, labels, network or CUDA."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import exp133_oai_b3 as m


def rows(values):
    return [[float(v)] * 12 for v in values]


def setup_csv(tmp_path):
    ids = ['a', 'b', 'c']
    pure = rows([.11, .22, .33])
    m.write_csv(tmp_path / 'submission.csv', ids, pure)
    m.write_csv(tmp_path / 'sample.csv', ids, rows([0, 0, 0]))
    return ids, pure


def test_fixed_rank_math_per_target_ties_and_no_final_rank():
    oai = rows([3/4, 1/4, 1/4, 1])
    rad = rows([1/4, 3/4, 2/4, 1])
    cnx = rows([1, 3/4, 2/4, 1/4])
    # A different target order must rank independently, not flatten the matrix.
    for i in range(4):
        cnx[i][1] = oai[i][0]
    path, got = m.blend(oai, rad, cnx)
    assert path == 'B3'
    assert got[0][0] == pytest.approx(.8*.75 + .1*1 + .1*.25)
    assert got[1][0] == pytest.approx(.8*.375 + .1*.75 + .1*.75)
    assert got[0][1] == pytest.approx(.8*.75 + .1*.75 + .1*.25)
    assert got[0][0] != m.e128.rank_average([r[0] for r in got])[0]
    assert m.WEIGHTS['B3'] == {'oai': .8, 'cnx': .1, 'rad': .1}


@pytest.mark.parametrize('rad_ok,cnx_ok,path', [(True, True, 'B3'), (True, False, 'B2'),
                                               (False, True, 'B1'), (False, False, 'pure_oai')])
def test_stage_order_and_every_fallback(tmp_path, rad_ok, cnx_ok, path):
    ids, pure = setup_csv(tmp_path)
    before = (tmp_path / 'submission.csv').read_bytes()
    calls = []
    partner = rows([.8, .5, .2])
    def stage(name, ok):
        def run(payload, start, got_ids, record, working):
            assert got_ids == ids
            calls.append(name)
            if name == 'cnx' and rad_ok:
                # B2 already safely published BEFORE reader starts.
                assert m.e128.read_submission(working / 'submission.csv')[1] == m.blend(pure, rad=partner)[1]
            if not ok:
                raise RuntimeError(name + ' synthetic failure')
            return partner
        return run
    result = m.run_after_oai({'rad_manifest': {}}, 0, tmp_path, tmp_path / 'sample.csv',
                            stages={'rad': stage('rad', rad_ok), 'cnx': stage('cnx', cnx_ok)},
                            clock=lambda: 0)
    assert calls == ['rad', 'cnx']
    assert result['path'] == path and result['final_weights'] == m.WEIGHTS[path]
    actual = m.validate_csv(tmp_path / 'submission.csv', tmp_path / 'sample.csv')[1]
    assert actual == m.blend(pure, partner if rad_ok else None, partner if cnx_ok else None)[1]
    if path == 'pure_oai':
        assert (tmp_path / 'submission.csv').read_bytes() == before
    receipt = json.loads((tmp_path / 'exp133_stage_timing.json').read_text())
    assert receipt['final_sha256'] == m.sha(tmp_path / 'submission.csv')


def test_exact_guard_arithmetic_and_projection():
    p = m.projection(1300)
    assert p['stages_seconds']['oai'] == 1950
    assert p['stages_seconds']['cnx'] == pytest.approx(3265)  # B: 2.05*1.15*1300+200 is 3264.9999… in float
    assert p['stages_seconds']['rad'] == pytest.approx(1.3 * (6.52503319258 + 10*2*.25)*1300 + 600)
    assert p['total_seconds'] == pytest.approx(25292.3060954602)
    assert m.can_start(100, 30380)
    assert not m.can_start(100, 30380.001)
    assert not m.can_start(float('nan'), 1)
    assert not m.can_start(1, float('inf'))
    assert not m.can_start(-1, 1)


def test_guard_skips_large_rad_but_allows_reader(tmp_path):
    setup_csv(tmp_path)
    # At N=3, reader costs 606.15 and Rad costs ~644.95; only reader fits.
    calls = []
    def stage(payload, start, ids, record, working):
        calls.append('cnx')
        return rows([.3, .2, .1])
    def no_rad(*args):
        pytest.fail('Rad must not start')
    r = m.run_after_oai({'rad_manifest': {}}, 0, tmp_path, tmp_path / 'sample.csv',
                       stages={'rad': no_rad, 'cnx': stage}, clock=lambda: 29860)
    assert r['path'] == 'B1' and calls == ['cnx']
    assert r['stages']['rad']['status'] == 'SKIPPED'


def test_late_rad_failure_prevents_reader_and_keeps_pure(tmp_path):
    setup_csv(tmp_path)
    before = (tmp_path / 'submission.csv').read_bytes()
    now = [0]
    def late(*args):
        now[0] = m.CUTOFF_S
        return rows([.2, .3, .4])
    r = m.run_after_oai({'rad_manifest': {}}, 0, tmp_path, tmp_path / 'sample.csv',
                       stages={'rad': late, 'cnx': lambda *a: pytest.fail('reader started late')},
                       clock=lambda: now[0])
    assert r['path'] == 'pure_oai'
    assert r['stages']['rad']['status'] == 'FAILED'
    assert r['stages']['cnx']['status'] == 'SKIPPED'
    assert (tmp_path / 'submission.csv').read_bytes() == before


@pytest.mark.parametrize('invalid', [rows([float('nan'), .2, .3]), rows([2, .2, .3]), [[.2]], rows([.2, .3])])
def test_invalid_partner_falls_back_and_reader_still_runs(tmp_path, invalid):
    setup_csv(tmp_path)
    r = m.run_after_oai({'rad_manifest': {}}, 0, tmp_path, tmp_path / 'sample.csv',
                       stages={'rad': lambda *a: invalid, 'cnx': lambda *a: rows([.4, .2, .1])},
                       clock=lambda: 0)
    assert r['path'] == 'B1'
    assert r['stages']['rad']['status'] == 'FAILED'


def test_partner_baseexception_and_diagnostic_write_failure_never_erase_default(tmp_path, monkeypatch):
    setup_csv(tmp_path)
    before = (tmp_path / 'submission.csv').read_bytes()
    def fail(*a):
        raise SystemExit('synthetic child exit')
    monkeypatch.setattr(m, 'write_json', lambda *a: (_ for _ in ()).throw(OSError('receipt unwritable')))
    result = m.run_after_oai({'rad_manifest': {}}, 0, tmp_path, tmp_path / 'sample.csv',
                            stages={'rad': fail, 'cnx': fail}, clock=lambda: 0)
    assert result['path'] == 'pure_oai'
    assert (tmp_path / 'submission.csv').read_bytes() == before


def test_publication_failure_preserves_previously_published_b2(tmp_path, monkeypatch):
    setup_csv(tmp_path)
    original = m.publish
    first = []
    def publish(working, ids, values):
        if first:
            raise OSError('synthetic second publish failure')
        result = original(working, ids, values)
        first.append((tmp_path / 'submission.csv').read_bytes())
        return result
    monkeypatch.setattr(m, 'publish', publish)
    r = m.run_after_oai({'rad_manifest': {}}, 0, tmp_path, tmp_path / 'sample.csv',
                       stages={'rad': lambda *a: rows([.3, .2, .1]), 'cnx': lambda *a: rows([.2, .1, .3])},
                       clock=lambda: 0)
    assert r['path'] == 'B2'
    assert (tmp_path / 'submission.csv').read_bytes() == first[0]
    assert 'publication_error' in r['stages']['cnx']
    assert r['stages']['cnx']['status'] == 'FAILED'


def test_every_notebook_code_cell_compiles_and_public_cells_exact():
    public, code = m.e128.load_source()
    data = m.payload()
    nb = m.notebook(data)
    assert nb['cells'][5:7] == code
    assert nb['cells'][2:5] == public['cells'][:3]
    for index, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<exp133-cell-%d>' % index, 'exec')
    for name, item in data['files'].items():
        source = m.e128.unpack(item['z'])
        assert hashlib.sha256(source.encode()).hexdigest() == item['sha256']
        compile(source, name, 'exec')
    credit = nb['cells'][1]['source']
    assert all(s in credit for s in ('nartaa', 'dreaddevelopment', 'goodpjw2008', 'Apache License',
                                      'OAI', 'licences=[other]', 'explicit user approval', 'EXP113', 'EXP118'))
    assert max(map(len, nb['cells'][-1]['source'].splitlines())) < 5000


def test_delivered_notebook_cells_compile_and_embed_current_modules():
    path = ROOT / '.local/exp133-oai-b3/blend_b3/release.ipynb'
    if not path.exists():
        pytest.skip('Release not built yet')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    assert notebook['cells'][5:7] == m.e128.load_source()[1]
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<delivered-cell-%d>' % i, 'exec')
    tree = ast.parse(notebook['cells'][-1]['source'])
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == '_exp133_write']
    assert len(calls) == 2
    for call in calls:
        name, encoded = [ast.literal_eval(a) for a in call.args]
        assert m.e128.unpack(encoded) == (ROOT / 'tools' / name).read_text(encoding='utf-8')


def test_source_embedding_and_tail_are_executable_with_tiny_fakes(tmp_path, monkeypatch):
    # Execute OUR generated tail only; never execute the two public code cells.
    data = {'rad_manifest': {}, 'files': {}}
    nb = m.notebook(data)
    import types
    fake = types.SimpleNamespace(e128=m.e128, run_after_oai=lambda payload, start: {
        'path': 'B3', 'final_sha256': 'fake', 'payload': payload})
    monkeypatch.setitem(sys.modules, 'exp133_oai_b3', fake)
    original_path = Path
    import pathlib
    monkeypatch.setattr(pathlib, 'Path', lambda *p: tmp_path / p[1] if len(p) == 2 and p[0] == '/kaggle/working'
                        else original_path(*p))
    env = {'_EXP133_START': 42}
    exec(compile(nb['cells'][-1]['source'], '<generated-tail>', 'exec'), env)
    assert env['_EXP133_RECEIPT']['path'] == 'B3'
    for name in ('exp128_oai_light.py', 'exp133_oai_b3.py'):
        assert (tmp_path / name).read_text(encoding='utf-8') == (ROOT / 'tools' / name).read_text(encoding='utf-8')


def test_metadata_and_exact_checkpoint_identity():
    meta = m.metadata()
    assert meta['id'] == 'taeyangg4/knee-exp133-oai-b3'
    assert meta['is_private'] and not meta['enable_internet'] and meta['enable_gpu']
    assert meta['machine_shape'] == 'NvidiaTeslaT4' and meta['docker_image'] == m.e128.DOCKER
    assert meta['competition_sources'] == [m.e128.COMPETITION]
    assert meta['dataset_sources'] == [m.e128.DATASET, m.e128.READER_DATASET, m.RAD_DATASET]
    data = m.payload()
    checks = data['rad_manifest']['checkpoints']
    assert len(checks) == 10
    assert [(c['seed'], c['fold']) for c in checks] == [(s, f) for s in (42, 43) for f in range(5)]
    assert all(c['tag'].startswith('EXP113_' if c['seed'] == 42 else 'EXP118_') for c in checks)
    assert all('EXP107' not in c['tag'] for c in checks)
    for ck in checks:
        meta = dict(arch='resnet50', res=224, lab=m.TARGETS, contract='raptor64_336_c140_s02',
                    norm='radimagenet', window_policy='within-slot-valid', input_policy='center3',
                    tag=ck['tag'], fold=ck['fold'], provenance={'protocol_sha256': ck['protocol_sha256']})
        m.check_checkpoint(meta, ck)
        meta['fold'] = 99
        with pytest.raises(ValueError):
            m.check_checkpoint(meta, ck)


def test_reader_calls_run_test_three_checkpoints_and_install_is_optional():
    tree = ast.parse((ROOT / 'tools/exp133_oai_b3.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reader_stage')
    children = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and 'module.run(' in n.value]
    assert len(children) == 1
    compile(children[0], '<reader-child>', 'exec')
    assert "module.run(data, 'test', ckpts, studies=json.loads(ids))" in children[0]
    optional = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == 'run_group' and any(k.arg == 'check' and isinstance(k.value, ast.Constant)
                                                     and k.value.value is False for k in c.keywords)]
    assert len(optional) == 1


def test_build_is_fresh_and_screen_spec_ingests(tmp_path):
    dest = tmp_path / 'build'
    result = m.build(dest)
    assert result['status'] == 'UNPUSHED_REQUIRES_RAD_DATASET'
    spec = json.loads((dest / 'blend_b3/submission_screen_spec.json').read_text())
    # Import only the contract validator AST: importing/running the full screen
    # can inspect training labels through its unrelated default baseline.
    source = ast.parse((ROOT / 'tools/submission_screen.py').read_text(encoding='utf-8'))
    validate = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_spec')
    import re
    scope = {'re': re, 'BASES': ('public', 'oof')}
    exec(compile(ast.Module(body=[validate], type_ignores=[]), '<screen-ingestion>', 'exec'), scope)
    assert scope['validate_spec'](spec) == spec
    assert spec['decision']['basis'] == 'public' and spec['approval_required']
    assert 'registered_at' not in spec['decision']
    uploads = json.loads((dest / 'required-upload-files.json').read_text())
    assert len(uploads['files']) == 11
    assert uploads['private_required'] and uploads['status'] == 'NOT_UPLOADED'
    with pytest.raises(ValueError, match='fresh'):
        m.build(dest)


def test_public3_validator_reconstructs_blend_and_rejects_tampering(tmp_path):
    ids, pure = setup_csv(tmp_path)
    payload = {'rad_manifest': {}}
    def stage(name):
        def run(p, start, got_ids, record, working):
            values = rows([.3, .1, .2])
            path = working / ('exp133_%s.csv' % name)
            m.write_csv(path, got_ids, values)
            record['csv_sha256'] = m.sha(path)
            return values
        return run
    result = m.run_after_oai(payload, 0, tmp_path, tmp_path / 'sample.csv',
                            stages={'rad': stage('rad'), 'cnx': stage('cnx')}, clock=lambda: 0)
    build = tmp_path / 'build'
    build.mkdir()
    m.write_json(build / 'build-receipt.json', {'rad_manifest_sha256': result['rad_manifest_sha256']})
    assert m.validate_public3(tmp_path, tmp_path / 'sample.csv', build)['status'] == 'PASS_B3_PUBLIC3'
    m.write_csv(tmp_path / 'submission.csv', ids, pure)
    with pytest.raises(ValueError, match='hash differs'):
        m.validate_public3(tmp_path, tmp_path / 'sample.csv', build)


def test_public3_fallback_is_valid_but_not_b3_integration(tmp_path):
    setup_csv(tmp_path)
    def fail(*args):
        raise TimeoutError('fake timeout')
    result = m.run_after_oai({'rad_manifest': {}}, 0, tmp_path, tmp_path / 'sample.csv',
                            stages={'rad': fail, 'cnx': fail}, clock=lambda: 0)
    build = tmp_path / 'build'
    build.mkdir()
    m.write_json(build / 'build-receipt.json', {'rad_manifest_sha256': result['rad_manifest_sha256']})
    with pytest.raises(ValueError, match='B3 integration NOT demonstrated'):
        m.validate_public3(tmp_path, tmp_path / 'sample.csv', build)
    assert m.validate_public3(tmp_path, tmp_path / 'sample.csv', build, True)['status'] == 'PASS_FALLBACK_PUBLIC3'


def test_old_exp128_artifacts_still_match_their_receipts():
    root = ROOT / '.local/exp128-oai-light'
    for variant in ('fidelity', 'blend_b1', 'blend_b2'):
        receipt = json.loads((root / variant / 'build-receipt.json').read_text())
        assert m.sha(root / variant / 'release.ipynb') == receipt['notebook_sha256']
