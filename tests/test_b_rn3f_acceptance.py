import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('acceptance', ROOT / 'tools/b_rn3f_acceptance.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def setup(tmp_path, action='evaluate'):
    args = SimpleNamespace(action=action, protocol_sha256='frozen', out=tmp_path/'numeric.json',
                           acceptance_receipt=tmp_path/'acceptance.json')
    for key in ['local_work', 't4_work']:
        work = tmp_path/key; (work/'result').mkdir(parents=True)
        (work/'result/receipt.json').write_text('{}', encoding='utf-8')
        (work/'run-receipt.json').write_text(json.dumps({'schema': 'B_RN3F_SERVING_V1',
            'status': 'COMPLETE_ENGINEERING_OUTPUT_ONLY', 'elapsed_seconds': 1,
            'worker_receipt_sha256': a.sha(work/'result/receipt.json')}), encoding='utf-8')
        setattr(args, key, work)
    return args


class Helper:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def evaluate(self, args):
        self.calls += 1
        Path(args.out).write_text(json.dumps(self.result), encoding='utf-8')
        return self.result

    run_input = evaluate


@pytest.mark.parametrize('bad', ['failed', 'running', 'missing', 'hash', 'worker', 'nan'])
def test_parent_failure_blocks_even_valid_numeric_output(tmp_path, bad):
    args = setup(tmp_path); parent = args.t4_work/'run-receipt.json'
    value = a.read(parent)
    if bad in ['failed', 'running']: value['status'] = bad.upper()
    if bad == 'hash': value['worker_receipt_sha256'] = 'bad'
    if bad == 'nan': value['elapsed_seconds'] = float('nan')
    parent.write_text(json.dumps(value), encoding='utf-8')
    if bad == 'missing': parent.unlink()
    if bad == 'worker': (args.t4_work/'worker-failure.json').write_text('{}')
    helper = Helper({'status': 'PASS_NUMERICAL_ONLY'})
    assert a.execute(args, helper) == 1
    assert helper.calls == 0
    assert a.read(args.acceptance_receipt)['status'] == 'FAILED'


@pytest.mark.parametrize('action,result,code', [
    ('evaluate', {'status': 'FAILED'}, 1),
    ('evaluate', {'status': 'PASS_NUMERICAL_ONLY'}, 0),
    ('input', {'pass': False}, 1),
    ('input', {'pass': True}, 0),
    ('input', {'pass': 'true'}, 1),
])
def test_explicit_gate_decides_exit_and_preserves_report(tmp_path, action, result, code):
    args = setup(tmp_path, action); helper = Helper(result)
    assert a.execute(args, helper) == code
    assert a.read(args.out) == result
    receipt = a.read(args.acceptance_receipt)
    assert receipt['numerical_output_sha256'] == a.sha(args.out)
    assert receipt['submission_ready'] is False
    with pytest.raises(ValueError, match='new acceptance'):
        a.execute(args, helper)


def test_helper_mutation_rejected_before_import(tmp_path):
    path = tmp_path/'untrusted.py'; path.write_text('raise RuntimeError("must not import")')
    with pytest.raises(ValueError, match='helper SHA'):
        a.load_helper(path)


def test_real_cli_failed_parent_returns_nonzero_before_numerical_run(tmp_path):
    import subprocess
    import sys
    args = setup(tmp_path)
    parent = args.local_work/'run-receipt.json'; value = a.read(parent)
    value['status'] = 'FAILED'; parent.write_text(json.dumps(value), encoding='utf-8')
    command = [sys.executable, str(ROOT/'tools/b_rn3f_acceptance.py'), 'evaluate',
               '--parity-helper', str(ROOT/'tools/b_rn3f_parity.py'),
               '--protocol', 'unused-before-parent-check', '--protocol-sha256', 'frozen',
               '--assets', 'unused-before-parent-check', '--out', str(args.out),
               '--acceptance-receipt', str(args.acceptance_receipt),
               '--local-work', str(args.local_work), '--t4-work', str(args.t4_work),
               '--local-input', 'unused', '--t4-input', 'unused']
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    assert a.read(args.acceptance_receipt)['status'] == 'FAILED'
    assert not args.out.exists()
