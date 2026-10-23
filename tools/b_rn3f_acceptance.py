"""Completion checks around frozen B numerical V1; no tolerance/source mutation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

PARITY_SHA = '298f11855c7e8e86007be612bf7e0831469da54d601aeedabab9f85a6259754b'
SCHEMA = 'B_RN3F_ACCEPTANCE_V1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parent_completion(work):
    work = Path(work)
    need(not (work / 'worker-failure.json').exists(), 'worker failure marker exists')
    parent = work / 'run-receipt.json'
    worker = work / 'result/receipt.json'
    value = read(parent)
    need(value.get('schema') == 'B_RN3F_SERVING_V1', 'parent schema mismatch')
    need(value.get('status') == 'COMPLETE_ENGINEERING_OUTPUT_ONLY', 'parent run incomplete/failed')
    elapsed = value.get('elapsed_seconds')
    need(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0,
         'parent elapsed time invalid')
    need(value.get('worker_receipt_sha256') == sha(worker), 'parent does not bind worker receipt')
    return {'work': str(work), 'parent_sha256': sha(parent), 'worker_sha256': sha(worker)}


def load_helper(path):
    need(sha(path) == PARITY_SHA, 'frozen numerical helper SHA mismatch')
    spec = importlib.util.spec_from_file_location('frozen_b_rn3f_parity_v1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(args, helper):
    receipt_path = Path(args.acceptance_receipt)
    need(not receipt_path.exists(), 'new acceptance receipt required')
    need(receipt_path.resolve() != Path(args.out).resolve(), 'acceptance and numerical outputs must differ')
    record = {'schema': SCHEMA, 'action': args.action, 'status': 'FAILED',
              'observed_at': datetime.now(timezone.utc).isoformat(),
              'adapter_sha256': sha(__file__), 'parity_helper_sha256': PARITY_SHA,
              'protocol_sha256': args.protocol_sha256, 'submission_ready': False,
              'runtime_release_claim': False}
    try:
        if args.action == 'evaluate':
            record['parent_receipts'] = [parent_completion(args.local_work), parent_completion(args.t4_work)]
            result = helper.evaluate(args)
            passed = result.get('status') == 'PASS_NUMERICAL_ONLY'
        else:
            result = helper.run_input(args)
            passed = result.get('pass') is True
        record['numerical_output'] = str(args.out)
        record['numerical_output_sha256'] = sha(args.out)
        record['numerical_status'] = result.get('status')
        record['input_pass'] = result.get('pass')
        need(passed, 'explicit numerical/input gate did not pass')
        record['status'] = 'PASS_COMPLETION_AND_NUMERICAL_ONLY' if args.action == 'evaluate' else 'PASS_INPUT_ONLY'
    except Exception as exc:
        record.update(error=str(exc), exception_type=type(exc).__name__)
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if record['status'].startswith('PASS_') else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['input', 'evaluate'])
    for name in ['parity-helper', 'protocol', 'protocol-sha256', 'assets', 'out', 'acceptance-receipt']:
        parser.add_argument('--' + name, required=True)
    for name in ['target', 'cache-root', 'competition', 'local-work', 't4-work', 'local-input', 't4-input']:
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    required = ['target'] if args.action == 'input' else ['local_work', 't4_work', 'local_input', 't4_input']
    need(all(getattr(args, key) for key in required), 'missing action arguments')
    if args.action == 'input':
        need(args.target in ['local-reference', 't4-engineering'], 'invalid input target')
    return execute(args, load_helper(args.parity_helper))


if __name__ == '__main__':
    raise SystemExit(main())
