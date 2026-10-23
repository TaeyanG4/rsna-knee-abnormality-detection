"""v1.0.0: real CPU process observation, failure and receipt tamper checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest
from tools import process_resource_observer as observer

@pytest.fixture(scope='module')
def execution(tmp_path_factory):
    out = tmp_path_factory.mktemp('observer') / 'run'
    # Actual CPU work gives the sampler an observable owned process.
    command = [sys.executable, '-B', '-c',
               'import hashlib; x=b"cpu";\nfor _ in range(500000): x=hashlib.sha256(x).digest()\nprint(len(x))']
    result = observer.supervise(command, out)
    assert json.loads((out/'receipt.json').read_text()) == result
    assert observer.resource_gate(result, command) is result
    return result, command, out

def test_actual_process_and_late_valid_receipt(execution):
    result, command, out = execution
    assert (out/'stdout.log').read_text().strip() == '32'
    late = dict(result, elapsed_seconds=1e9)
    assert observer.resource_gate(late, command) is late
    with pytest.raises(FileExistsError):
        observer.supervise(command, out)

@pytest.mark.parametrize('key,value', [
    ('argv', ['wrong']), ('supervisor_sha256', '0'*64), ('helper_sha256', '0'*64),
    ('child_exit_code', 1), ('wrapper_exit_code', 1), ('status', 'FAILED'),
    ('failure', {}), ('retries', 1), ('sample_count', 0), ('sample_attempts', -1),
    ('sampled_tree_rss_max_bytes', 0), ('access_failure_count', 1),
    ('failed_sample_attempts', 1), ('elapsed_seconds', float('nan')),
    ('elapsed_seconds', -1), ('observed_identities', [])])
def test_bad_execution_still_rejected(execution, key, value):
    result, command, _ = execution
    with pytest.raises(ValueError):
        observer.resource_gate(dict(result, **{key:value}), command)

def test_creation_identity_and_cleanup_rejected(execution):
    result, command, _ = execution
    bad = deepcopy(result)
    bad['child_identity']['create_time'] += 1
    with pytest.raises(ValueError):
        observer.resource_gate(bad, command)
    bad = deepcopy(result)
    bad['cleanup']['survivors'] = [{'pid':123}]
    with pytest.raises(ValueError):
        observer.resource_gate(bad, command)

def test_failed_real_process_is_never_complete(tmp_path):
    command = [sys.executable, '-B', '-c', 'raise SystemExit(7)']
    result = observer.supervise(command, tmp_path/'failed')
    assert result['status'] == 'FAILED' and result['wrapper_exit_code'] == 1
    with pytest.raises(ValueError):
        observer.resource_gate(result, command)
