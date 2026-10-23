"""Source-only H207 N32 adapter tests: no scored-module import or real data."""
from __future__ import annotations
import ast
import base64
import builtins
import copy
import hashlib
import io
import json
from pathlib import Path
import posixpath
import sys
from types import SimpleNamespace
import zipfile

import pytest
from tools import hb207_stress_package_v102 as m

CANONICAL = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
LEGACY = '/kaggle/input/rsna-knee-abnormality-detection'
STRESS = '/kaggle/working/c85-stress-data'
DATASETS = [
    'taeyangg4/knee-exp169-a172-v2-assets',
    'taeyangg4/knee-exp155-own0949-oai-v1-2-assets',
    'nartaa/rsna-knee-publication-swa-weights-20261007',
    'taeyangg4/knee-r6-fe055406-v1',
    'heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
    'taeyangg4/knee-hb126-codecs-cp313-v1',
    'taeyangg4/knee-exp169-c4-v4-assets',
    'taeyangg4/knee-inf2-small10-v1-assets',
]


@pytest.fixture(scope='module')
def source():
    normal, original, blob, assignments, meta = m.inputs()
    return normal, original, blob, assignments, meta, m.patch_files(original)


def selector(source, *, files=(), dirs=(), walk=(), env=None):
    files, dirs = set(files), set(dirs)
    virtual_os = SimpleNamespace(environ={} if env is None else env,
        path=SimpleNamespace(join=posixpath.join, exists=lambda p:p in files or p in dirs,
                             isfile=lambda p:p in files, isdir=lambda p:p in dirs),
        walk=lambda p:iter(walk))
    return m.extract(source.decode(), ['find_test_root'], {'os':virtual_os})['find_test_root']


@pytest.mark.parametrize('mount', [CANONICAL, LEGACY])
@pytest.mark.parametrize('data_dir', ['test_series', 'test_images'])
def test_actual_original_counterexample_and_adapter(source, mount, data_dir):
    original, modified = source[1][m.R224], source[5][m.R224]
    fs = dict(files=[mount+'/test.csv', STRESS+'/test.csv'],
              dirs=[STRESS+'/'+data_dir], env={'C85_COMPETITION':STRESS})
    assert selector(original, **fs)() == mount  # This fails for a silently fixed parent.
    assert selector(modified, **fs)() == STRESS  # This fails for the original blocker.


@pytest.mark.parametrize('files,dirs,walk,expected', [
    ([CANONICAL+'/test.csv'], [], [], CANONICAL),
    ([LEGACY+'/test.csv'], [], [], LEGACY),
    ([LEGACY+'/test.csv', CANONICAL+'/test.csv'], [], [], CANONICAL),
    ([], ['/kaggle/input/custom/test_series'],
     [('/kaggle/input/csv-only', [], ['test.csv']),
      ('/kaggle/input/custom', [], ['test.csv'])], '/kaggle/input/custom'),
    ([], ['/kaggle/input/custom/test_images'],
     [('/kaggle/input/custom', [], ['test.csv'])], '/kaggle/input/custom'),
    ([], [], [('/kaggle/input/csv-only', [], ['test.csv'])], '/kaggle/input/csv-only'),
    ([], [], [], None),
])
def test_unset_override_preserves_all_original_discovery_branches(source, files, dirs, walk, expected):
    for payload in (source[1][m.R224], source[5][m.R224]):
        fn = selector(payload, files=files, dirs=dirs, walk=walk)
        if expected is None:
            with pytest.raises(RuntimeError, match='no test root under /kaggle/input'):
                fn()
        else:
            assert fn() == expected


@pytest.mark.parametrize('override,files,dirs', [
    ('', [], []),
    (STRESS, [], [STRESS+'/test_series']),
    (STRESS, [], [STRESS+'/test.csv', STRESS+'/test_series']),
    (STRESS, [STRESS+'/test.csv'], []),
    (STRESS, [STRESS+'/test.csv', STRESS+'/test_series'], []),
    (STRESS+'/missing', [STRESS+'/test.csv'], [STRESS+'/test_series']),
])
def test_invalid_override_fails_closed_despite_valid_public_mount(source, override, files, dirs):
    fn = selector(source[5][m.R224], files=[CANONICAL+'/test.csv', *files], dirs=dirs,
                  env={'C85_COMPETITION':override})
    with pytest.raises(RuntimeError, match='Invalid C85_COMPETITION test root'):
        fn()


def test_override_works_without_any_kaggle_input_mount(source):
    fs = dict(files=[STRESS+'/test.csv'], dirs=[STRESS+'/test_series'],
              env={'C85_COMPETITION':STRESS})
    with pytest.raises(RuntimeError, match='no test root under /kaggle/input'):
        selector(source[1][m.R224], **fs)()
    assert selector(source[5][m.R224], **fs)() == STRESS


def test_all_payloads_and_outside_function_bytes_are_preserved(source):
    old, new = source[1], source[5]
    assert len(old) == len(new) == 31 and set(old) == set(new)
    assert [n for n in old if old[n] != new[n]] == [m.CODEC, m.R224]
    assert sum(old[n] == new[n] for n in old) == 29
    assert new[m.R224].replace(m.ADAPTER.encode(), b'', 1) == old[m.R224]
    assert new[m.CODEC].replace(m.digest(new[m.R224]).encode(),
                               m.digest(old[m.R224]).encode(), 1) == old[m.CODEC]
    # Frozen label-free selector is carried over without execution.
    assert m.digest(new['exp169_c85_stress.py']) == '4d10508371139e9a9125f96c183db08e079b6502c41ad4ebb4b568b5f5cb42d1'
    m.prove_delta(old, new)


@pytest.mark.parametrize('target', [m.R224, m.CODEC, 'exp169_c85_stress.py'])
def test_delta_rejects_any_extra_payload_change(source, target):
    corrupt = dict(source[5])
    corrupt[target] += b'\n# unexpected change\n'
    with pytest.raises(ValueError):
        m.prove_delta(source[1], corrupt)


def literals(source, names):
    return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Assign) and len(n.targets)==1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names}


def test_actual_runtime_pin_enforcement_binds_adapter_and_rejects_corruption(source, tmp_path):
    files = source[5]
    codec = files[m.CODEC].decode()
    env = literals(codec, {'C85R7_SHA', 'CODECS_SHA'})
    checked, parent_calls = [], []
    def check_pin(path, expected):
        checked.append(path.name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('runtime pin mismatch')
    for name in (m.R224, 'exp169_r224.py', 'exp169_release_c85r7.py', 'hb126_codecs.py'):
        (tmp_path/name).write_bytes(files[name])
    env.update(base=SimpleNamespace(runtime_pins=lambda path:parent_calls.append(path)),
               p=SimpleNamespace(check_pin=check_pin), r224=SimpleNamespace(SOURCE=m.R224))
    pins = m.extract(codec, ['runtime_pins'], env)['runtime_pins']
    pins(tmp_path)
    assert parent_calls == [tmp_path]
    assert checked == [m.R224, 'exp169_r224.py', 'exp169_release_c85r7.py', 'hb126_codecs.py']
    (tmp_path/m.R224).write_bytes(source[1][m.R224])  # Original unadapted bytes must fail.
    with pytest.raises(ValueError, match='runtime pin mismatch'):
        pins(tmp_path)


def unpack_stress(nb):
    assert len(nb['cells']) == 2
    code = ''.join(nb['cells'][1]['source'])
    values = literals(code, {'BLOB', 'PINS'})
    blob = base64.b64decode(values['BLOB'], validate=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == 31
        assert all(not n.startswith(('/', '\\')) and '..' not in Path(n).parts
                   and ':' not in n and '\\' not in n for n in names)
        files = {n:archive.read(n) for n in names}
    assert values['PINS'] == {n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    assert repr(hashlib.sha256(blob).hexdigest()) in code
    return code, files, blob


def test_real_notebook_bootstrap_pins_n32_and_exact_full_release(source):
    nb, blob, pins = m.notebook(source[0], source[1], source[2], source[3], source[5])
    code, files, unpacked = unpack_stress(nb)
    assert unpacked == blob and files == source[5]
    assert pins == {n:m.digest(v) for n,v in files.items()}
    assert 'FAST' not in code
    assert 'never submit' in ''.join(nb['cells'][0]['source'])
    tree = ast.parse(code)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    prepare = [n for n in calls if ast.unparse(n.func) == 'stress.prepare']
    run = [n for n in calls if ast.unparse(n.func) == 'release.run_release']
    assert len(prepare) == len(run) == 1
    assert ast.unparse(prepare[0]) == 'stress.prepare(competition, data, n=32)'
    assert ast.unparse(run[0]) == "release.run_release(runtime, Path('/kaggle/working'))"
    assert "os.environ['C85_COMPETITION'] = str(data)" in code
    assert 'C85R7CODEC_STRESS_TOTAL_SECONDS' in code
    assert code.index('stress.prepare(') < code.index('started = time.monotonic()')
    assert code.index('started = time.monotonic()') < code.index('release.run_release(')
    assert nb['metadata'] == source[0]['metadata']
    # Apart from payload identity, actual inherited stress wrapper and guard removal,
    # every bootstrap line must be the ordinary release's exact line.
    expected = ''.join(source[0]['cells'][2]['source'])
    expected = __import__('textwrap').dedent(expected.removeprefix('if not FAST:\n'))
    expected = expected.replace(repr(source[3]['BLOB']), repr(base64.b64encode(blob).decode()))
    expected = expected.replace('PINS = '+repr(source[3]['PINS']), 'PINS = '+repr(pins))
    expected = expected.replace(repr(m.digest(source[2])), repr(m.digest(blob)))
    inherited = m.extract(source[1][m.CODEC].decode(), ['stress_notebook'],
        {'notebook_payload':lambda *_:{'cells':[{'source':[]},{'source':expected.splitlines(True)}]},
         'VERSION':'1.1.49-r6w958c4inf2', 'need':m.need})['stress_notebook']
    assert code == ''.join(inherited(blob, pins)['cells'][1]['source'])


def test_real_render_metadata_proof_determinism_and_safe_reads(source, monkeypatch):
    allowed = {str((m.STAGE/p).resolve()).lower() for p in m.NORMAL_PINS}
    allowed |= {str((m.WORK/'stress-v1'/p).resolve()).lower() for p in m.V1_PINS}
    allowed |= {str((m.WORK/'candidate-runtime'/p).resolve()).lower() for p in source[1]}
    allowed |= {str((m.ROOT/p).resolve()).lower() for p in (
        'tools/hb207_stress_package_v102.py', 'tools/test_hb207_stress_package_v102.py',
        'tools/c85r7_r6w958inf2_v1148.py', 'tools/hb145_output_io.py')}
    allowed.add(str((m.WORK/'STRESS-INPUT-FIX-PROMPT.md').resolve()).lower())
    seen = set()
    original_io_open, original_open = io.open, builtins.open
    def guarded(original):
        def checked(path, *args, **kwargs):
            resolved = str(Path(path).resolve()).lower()
            assert resolved in allowed, 'unexpected input file'
            mode = args[0] if args else kwargs.get('mode', 'r')
            assert not any(c in mode for c in 'wax+')
            seen.add(resolved)
            return original(path, *args, **kwargs)
        return checked
    with monkeypatch.context() as ctx:
        ctx.setattr(io, 'open', guarded(original_io_open))
        ctx.setattr(builtins, 'open', guarded(original_open))
        first, second = m.render(), m.render()
    assert first == second and seen == allowed
    meta = json.loads(first['kernel/kernel-metadata.json'])
    assert {k for k in meta if meta[k] != source[4][k]} == {'id', 'title'}
    assert meta['dataset_sources'] == DATASETS == source[4]['dataset_sources']
    assert meta['id'] == m.SLUG and meta['title'] == m.SLUG.split('/')[1]
    assert '/' not in meta['title']
    identity = json.loads(first['embedded-identity.json'])
    assert len(identity)==31 and sum(v['identical'] for v in identity.values())==29
    assert {n for n,v in identity.items() if not v['identical']} == {m.R224,m.CODEC}
    receipt = json.loads(first['build-receipt.json'])
    assert receipt['qualified_for_submission'] is False
    assert receipt['model_forwards']==0 and receipt['never_submit'] is True
    assert receipt['normal_stage_pins']==m.NORMAL_PINS and receipt['preserved_v1_pins']==m.V1_PINS
    _, files, _ = unpack_stress(json.loads(first['kernel/release.ipynb']))
    assert files == source[5]
    assert not any(n in sys.modules for n in ('torch','numpy','pandas','pydicom'))


def test_create_only_package_and_validation_rejects_mutation(tmp_path):
    out = tmp_path/'package'
    result = m.build(out)
    assert result['normal_stage_unchanged'] and not result['qualification_pass']
    original = (out/'kernel/release.ipynb').read_bytes()
    with pytest.raises(ValueError, match='immutable package already exists'):
        m.build(out)
    assert (out/'kernel/release.ipynb').read_bytes()==original
    with pytest.raises(ValueError, match='create-only output exists'):
        m.write_new(out/'kernel/release.ipynb', b'bad')
    with (out/'runtime'/m.R224).open('ab') as stream:
        stream.write(b'\n# mutation\n')
    with pytest.raises(ValueError, match='stress package drift'):
        m.validate(out)
    assert {p:m.sha(m.STAGE/p) for p in m.NORMAL_PINS} == m.NORMAL_PINS
    assert {p:m.sha(m.WORK/'stress-v1'/p) for p in m.V1_PINS} == m.V1_PINS


def test_existing_kernel_directory_prevents_partial_publication(tmp_path):
    out = tmp_path/'claimed'
    (out/'kernel').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        m.build(out)
    assert list(out.iterdir()) == [out/'kernel']
    assert list((out/'kernel').iterdir()) == []
