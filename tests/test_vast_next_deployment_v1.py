"""F2 regression: isolated sibling tree, read-only predecessor, exact import pins."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tools import vast_next_deployment_v1 as audit

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/audit.TASK
TREE = TASK/'deploy-tree'
spec = importlib.util.spec_from_file_location('deploy_guard', TASK/'deploy_guard_v1.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.fixture(scope='module')
def inventory():
    return json.loads((TASK/'DEPLOYMENT-v1.json').read_text())


def test_real_recursive_inventory(inventory):
    result = audit.validate_inventory(inventory, TREE)
    assert audit.TERMINAL in result['files']
    assert 'training/src/raptor_common.py' in result['files']
    assert all(n not in {r['path'] for r in inventory['files']} for n in result['generated_scan_only'])
    assert guard.verify(TREE, inventory) == []


@pytest.mark.parametrize('missing', [audit.TERMINAL, 'tools/job_runtime.py',
                                   'training/src/raptor_common.py',
                                   '.local/exp189-cpu-v120/input/cloud_code/training/geometry_bank.py'])
def test_missing_vast_import_rejected(inventory, missing):
    broken = copy.deepcopy(inventory)
    broken['files'] = [r for r in broken['files'] if r['path'] != missing]
    with pytest.raises(ValueError, match='deployment list missing imports/pins'):
        audit.validate_inventory(broken, TREE)


def test_new_transitive_import_detected(inventory, monkeypatch):
    original = audit.source_edges
    def changed(root, name):
        if name == 'tools/k1n_config_v1.py':
            return set(), set(), []
        edges, ext, dyn = original(root, name)
        if name == 'tools/job_runtime.py':
            edges.add('tools/k1n_config_v1.py')
        return edges, ext, dyn
    monkeypatch.setattr(audit, 'source_edges', changed)
    with pytest.raises(ValueError, match='deployment list missing imports/pins'):
        audit.validate_inventory(inventory, TREE)


def test_dynamic_and_external_drift_rejected(inventory):
    for field in ('dynamic_calls', 'external_packages'):
        broken = copy.deepcopy(inventory)
        broken[field] = {} if field == 'dynamic_calls' else []
        with pytest.raises(ValueError, match='imports? .*changed|import sites changed'):
            audit.validate_inventory(broken, TREE)


def test_relative_package_resolution(tmp_path):
    pkg = tmp_path/'tools/pkg'
    pkg.mkdir(parents=True)
    (pkg/'__init__.py').write_text('from . import leaf\n')
    (pkg/'leaf.py').write_text('VALUE=1\n')
    edges, _, _ = audit.source_edges(tmp_path, 'tools/pkg/__init__.py')
    assert 'tools/pkg/leaf.py' in edges


def fixture_manifest(root, payload=b'# source\n'):
    path = root/'tools/fixture.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return dict(schema='vast-next-deployment/v1', files=[dict(path='tools/fixture.py', sha256=guard.sha(path))])


def test_verify_readonly_missing_and_mismatch(tmp_path):
    manifest = fixture_manifest(tmp_path/'stage')
    root = tmp_path/'project'
    with pytest.raises(ValueError, match='missing deployment'):
        guard.verify(root, manifest)
    assert not root.exists()
    fixture_manifest(root, b'# different\n')
    before = (root/'tools/fixture.py').read_bytes()
    with pytest.raises(ValueError, match='hash mismatch'):
        guard.verify(root, manifest)
    assert (root/'tools/fixture.py').read_bytes() == before


@pytest.mark.parametrize('name', ['../oops.py', '/tmp/oops.py',
    '.local/scratch/astra-b2/k1-fullbag-v1/fit/oops.py',
    '.local/scratch/astra-b2/h224-folds-v1/gpu0-owner/oops.py',
    'tools/cache.npy', 'tools/../oops.py'])
def test_prohibited_destinations(tmp_path, name):
    with pytest.raises(ValueError):
        guard.safe_path(tmp_path, name)


def test_install_gated_exclusive_and_idempotent(tmp_path, monkeypatch):
    stage, project = tmp_path/'stage', tmp_path/'project'
    manifest = fixture_manifest(stage)
    def blocked(*_):
        raise ValueError('predecessor not terminal SUCCESS: WAIT')
    monkeypatch.setattr(guard, 'require_success', blocked)
    with pytest.raises(ValueError, match='not terminal SUCCESS'):
        guard.install_missing(project, stage, manifest, tmp_path/'predecessor')
    assert not project.exists()
    monkeypatch.setattr(guard, 'require_success', lambda *_: {'state': 'SUCCESS-fixture'})
    assert guard.install_missing(project, stage, manifest, tmp_path/'predecessor')['installed_missing'] == 1
    stat = (project/'tools/fixture.py').stat()
    assert guard.install_missing(project, stage, manifest, tmp_path/'predecessor')['installed_missing'] == 0
    assert (project/'tools/fixture.py').stat().st_mtime_ns == stat.st_mtime_ns


def test_install_preflight_rejects_mismatch_before_any_write(tmp_path, monkeypatch):
    stage, project = tmp_path/'stage', tmp_path/'project'
    manifest = fixture_manifest(stage)
    fixture_manifest(project, b'# changed\n')
    monkeypatch.setattr(guard, 'require_success', lambda *_: pytest.fail('must fail before SUCCESS probe'))
    with pytest.raises(ValueError, match='hash mismatch'):
        guard.install_missing(project, stage, manifest, tmp_path/'predecessor')
    assert (project/'tools/fixture.py').read_bytes() == b'# changed\n'


def test_terminal_contract_reused(monkeypatch, tmp_path):
    class Terminal:
        @staticmethod
        def binding(root): return {'run_id': 'fixture'}
        @staticmethod
        def probe(root, bound):
            assert bound == {'run_id': 'fixture'}
            return 'WAIT'
    class Loader:
        @staticmethod
        def exec_module(module): pass
    class Spec: loader = Loader()
    monkeypatch.setattr(guard.importlib.util, 'spec_from_file_location', lambda *_: Spec())
    monkeypatch.setattr(guard.importlib.util, 'module_from_spec', lambda *_: Terminal())
    with pytest.raises(ValueError, match='not terminal SUCCESS: WAIT'):
        guard.require_success(tmp_path/'project', tmp_path/'stage')
    monkeypatch.setattr(Terminal, 'probe', staticmethod(lambda *_: 'SUCCESS'))
    assert guard.require_success(tmp_path/'project', tmp_path/'stage') == {'run_id': 'fixture'}


def test_real_source_deployment_smoke(inventory, tmp_path):
    from test_fullbag_post_v1 import waiter_fixture
    project = (tmp_path/'project').resolve()
    predecessor = tmp_path/'predecessor'
    waiter_fixture(predecessor, 'SUCCESS')
    conflict = predecessor/'tools/l1_rco_k1_foldgate_v1.py'
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_bytes(b'# live bytes differ: MUST NOT OVERWRITE\n')
    protected = {p: p.read_bytes() for p in predecessor.rglob('*') if p.is_file()}
    result = guard.install_missing(project, TREE, inventory, predecessor)
    assert result['installed_missing'] == len(inventory['files'])
    assert all(p.read_bytes() == payload for p, payload in protected.items())
    manifest = TASK/'DEPLOYMENT-v1.json'
    cli = subprocess.run([sys.executable, '-B', str(TASK/'deploy_guard_v1.py'),
        'verify', '--manifest', str(manifest), '--root', str(project)],
        capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)['checked'] == len(inventory['files'])
    assert not list(project.rglob('*.pyc'))


@pytest.mark.parametrize('state', ['STARTED', 'STOPPED'])
def test_real_non_success_predecessor_never_installs(inventory, tmp_path, state):
    from test_fullbag_post_v1 import waiter_fixture
    predecessor, project = tmp_path/'old', tmp_path/'new'
    waiter_fixture(predecessor, state)
    with pytest.raises(ValueError, match='not terminal SUCCESS'):
        guard.install_missing(project, TREE, inventory, predecessor)
    assert not project.exists()


@pytest.mark.parametrize('dest', ['old', 'old/child', 'unrelated/new'])
def test_sibling_only(tmp_path, dest):
    with pytest.raises(ValueError, match='separate sibling'):
        guard.separate(tmp_path/dest, tmp_path/'old')


def test_input_copy_preflight_readonly_and_idempotent(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    names=['.local/scratch/astra-b2/h224-folds-v1/local32.json',
           '.local/scratch/astra-b2/h224-meta-eps-v1/local32-meta.npz']
    rows=[]
    for name in names:
        p=old/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'immutable fixture')
        rows.append(dict(path=name,sha256=guard.sha(p)))
    manifest=dict(schema='vast-next-inputs/v1',files=rows)
    broken=copy.deepcopy(manifest); broken['files'][-1]['sha256']='0'*64
    with pytest.raises(ValueError,match='source input mismatch'):
        guard.provision_inputs(new,old,broken)
    assert not new.exists()
    assert guard.provision_inputs(new,old,manifest)['copied']==2
    assert guard.provision_inputs(new,old,manifest)['copied']==0
    assert all((old/n).read_bytes()==(new/n).read_bytes()==b'immutable fixture' for n in names)
    (new/names[-1]).write_bytes(b'keep conflicting destination')
    with pytest.raises(ValueError,match='destination input mismatch'):
        guard.provision_inputs(new,old,manifest)
    assert (new/names[-1]).read_bytes()==b'keep conflicting destination'


@pytest.mark.parametrize('name',['../x','.local/exp189-cpu-v120/input/cloud_code/../../x',
    '.local/scratch/astra-b2/h224-folds-v1/cache/index.json','/absolute.py',
    '.local/exp189-cpu-v120/input/cloud_code/evil\\x.py'])
def test_input_paths_rejected(tmp_path,name):
    manifest=dict(schema='vast-next-inputs/v1',files=[dict(path=name,sha256='0'*64)])
    with pytest.raises(ValueError,match='not allowlisted'):
        guard.provision_inputs(tmp_path/'new',tmp_path/'old',manifest)
    assert not (tmp_path/'new').exists()


def test_real_immutable_input_hashes_and_complete_cloud_pin():
    manifest=json.loads((TASK/'INPUTS-v1.json').read_text())
    assert len(manifest['files'])==831
    for row in manifest['files']:
        assert guard.sha(ROOT/row['path'])==guard.sha(TREE/row['path'])==row['sha256']
    prefix='.local/exp189-cpu-v120/input/cloud_code/'
    assert sum(r['path'].startswith(prefix) for r in manifest['files'])==820


def test_isolated_runtime_and_package(tmp_path):
    result=subprocess.run([sys.executable,'-B',str(TASK/'check_f2_runtime.py'),str(tmp_path)],
        cwd=TREE,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.splitlines()[-1])['status']=='PASS'


def f3_identity_fixture():
    spec=importlib.util.spec_from_file_location('_f3_identity', TASK/'check_f3_identity.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('interpreter', ['/opt/conda/bin/python3', '/opt/conda/bin/python'])
def test_recorded_vast_historical_argv(interpreter):
    assert f3_identity_fixture().check(interpreter)['status']=='PASS'


@pytest.mark.parametrize('change', ['run', 'argument', 'owner', 'digest', 'receipt_run', 'receipt_digest'])
def test_recorded_vast_identity_stays_strict(change):
    x=f3_identity_fixture().fixture()
    spec=x.qualification
    row=x.records[spec['predecessor']['tag']]
    if change=='run': spec['predecessor']['run_id']='different-run'
    elif change=='argument': spec['predecessor']['command'][-1]='different-f3'
    elif change=='owner': spec['owner']='/root/work/unrelated-owner'
    elif change=='digest': row['job.json']['command_sha256']='0'*64
    elif change=='receipt_run': row['receipt.json']['run_id']='different-run'
    elif change=='receipt_digest': row['receipt.json']['command_sha256']='0'*64
    with pytest.raises(ValueError, match='binding'):
        x.verify(spec)
