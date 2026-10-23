"""v1.0.2: offline v1.3.2 admission smoke using immutable real receipt inputs.

All successful cases are SYNTHETIC test fixtures, never release qualifications.
No Kaggle/network/subprocess calls; fixed project scratch only.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import kaggle_submit_once as ks

SCRATCH = ROOT / '.local/scratch/astra-b5/kaggle-submit-once-v132'
REAL = ROOT / '.local/scratch/slot-b/taeyangg4-knee-lb-fluidrd-v1-visible/c85r7-receipt.json'
REG = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.9-fluidR/registration.json'
KERNEL = 'taeyangg4/taeyangg4-knee-lb-fluidrd-v1'
BASELINE = ROOT / '.local/scratch/slot-b/knee-exp169-c85r7codec-v1/c85r7-receipt.json'
BASE_REG = ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0/registration.json'
HB127 = ROOT / '.local/scratch/sol-1/hb127/r7-smoke/A/export/hb127-receipt.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def forbidden(*args, **kwargs):
    raise AssertionError('real subprocess/network/CLI forbidden in smoke')


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    original_hashes = {str(p): sha(p) for p in (REAL, REG, BASELINE, BASE_REG, HB127)}
    results = []

    def fixture(name, data, kernel=KERNEL):
        folder = SCRATCH / name / (kernel.split('/')[1] + '-visible')
        receipt, review = folder / 'visible.json', folder / 'review.md'
        put(receipt, data)
        review.write_text('SYNTHETIC TEST FIXTURE ONLY; not a release approval.\n' + kernel + '\nVERDICT: PASS\n', encoding='utf-8')
        return receipt, review

    def check(name, receipt, review, expected, kernel=KERNEL, reason=None):
        argv = ['kaggle_submit_once.py', '--check-only', '--id', 'synthetic-smoke',
                '--kernel', kernel, '--version', '1', '--receipt', str(receipt), '--review', str(review)]
        output = io.StringIO()
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output):
            code = ks.main()
        text = output.getvalue()
        (SCRATCH / (name + '.log')).write_text(text, encoding='utf-8')
        assert code == expected, (name, code, text)
        if reason:
            assert reason in text, (name, text)
        results.append(dict(case=name, exit=code, expected=expected, matched=True, argv=argv))

    raw = json.loads(REAL.read_text(encoding='utf-8'))
    clean = copy.deepcopy(raw)  # Preserve real anchor and prediction counters.
    transport = dict(kernel=KERNEL, kernel_version=1, registration=str(REG), registration_sha256=sha(REG))
    native = {**clean, **transport}
    receipt, review = fixture('native', native)
    check('native-admit', receipt, review, 0)
    legacy, legacy_review = fixture('sidecar', clean)
    legacy.write_bytes(REAL.read_bytes())
    put(legacy.parent / 'submission-binding.json', dict(schema_version='1.0.0', receipt_sha256=sha(legacy), **transport))
    check('real-fluidRd-admit', legacy, legacy_review, 0)
    baseline_kernel = 'taeyangg4/knee-exp169-c85r7codec-v1'
    br, bv = fixture('baseline', json.loads(BASELINE.read_text()), baseline_kernel)
    br.write_bytes(BASELINE.read_bytes())
    put(br.parent / 'submission-binding.json', dict(schema_version='1.0.0', receipt_sha256=sha(br),
        kernel=baseline_kernel, kernel_version=1, registration=str(BASE_REG), registration_sha256=sha(BASE_REG)))
    check('real-v110-admit', br, bv, 0, baseline_kernel)

    for name, changes, reason in [
        ('incomplete', {'status': 'RUNNING'}, 'COMPLETE'),
        ('inactive-codecs', {'codecs': {'status': 'INACTIVE'}}, 'codecs'),
        ('wrong-version', {'kernel_version': 2}, 'version mismatch'),
        ('wrong-kernel', {'kernel': 'taeyangg4/other'}, 'kernel mismatch'),
        ('prediction-fallback', {'s_r7_prediction_fallbacks': 1}, 'fallback counter'),
        ('bad-telemetry', {'telemetry': {'prediction_fallbacks': True}}, 'fallback counter'),
        ('fast-missing-gate', {'fast_path': True}, 'admission:'),
    ]:
        r, v = fixture(name, {**native, **changes})
        check(name, r, v, 2, reason=reason)
    r, v = fixture('bad-review', native)
    v.write_text(KERNEL + '\nVERDICT: FAIL\n', encoding='utf-8')
    check('bad-review', r, v, 2, reason='VERDICT: PASS')

    # Real kernel bytes with explicitly MOCKED PASS receipts, isolated from production.
    # This tests the consumer, not approval by the producer (see parent-probe.json).
    tool = SCRATCH / 'tools/strict_variant_gate.py'
    tool.parent.mkdir(exist_ok=True)
    tool.write_bytes((ROOT / 'tools/strict_variant_gate.py').read_bytes())
    for kind, parent_source, candidate_source in [
        ('c85', ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0/kernel',
         ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.18-own25q3/kernel'),
        ('exp143', ROOT / '.local/exp152-own0949-v1/stage/own',
         ROOT / 'artifacts/experiments/HB151_KILLTEST/v1.0.2/kernel'),
    ]:
        parent, stage = SCRATCH / (kind + '-parent'), SCRATCH / (kind + '-stage')
        for target, source in [(parent, parent_source), (stage, candidate_source)]:
            (target / 'kernel').mkdir(parents=True, exist_ok=True)
            for name in ('release.ipynb', 'kernel-metadata.json'):
                original = source / name
                original_hashes[str(original)] = sha(original)
                (target / 'kernel' / name).write_bytes(original.read_bytes())
            identity = json.loads((target / 'kernel/kernel-metadata.json').read_bytes())['id']
            put(target / 'registration.json', dict(kernel_id=identity, version='1.0.1'))
            if target == stage:
                put(target / 'registration.json', dict(kernel_id=identity, version='1.0.1',
                    parent=str(parent), parent_sha256sums=sha(parent / 'SHA256SUMS')))
            members = [target / 'registration.json', *sorted((target / 'kernel').iterdir())]
            (target / 'SHA256SUMS').write_bytes(''.join(sha(p) + '  ' +
                p.relative_to(target).as_posix() + '\n' for p in members).encode())
        kernel = json.loads((stage / 'registration.json').read_bytes())['kernel_id']
        inputs = [tool, *[p for folder in (parent, stage) for p in
            [folder / 'registration.json', folder / 'SHA256SUMS', *sorted((folder / 'kernel').iterdir())]]]
        gate_path = SCRATCH / '.local/state/gates' / (kernel.split('/')[1] + '.json')
        gate = dict(gate_version='1.1.0', status='PASS', errors=[], kernel_id=kernel,
                    stage=str(stage), parent=str(parent), input_sha256={str(p): sha(p) for p in inputs})
        put(gate_path, gate)
        fast = dict(status='COMPLETE_FAST_PATH_VISIBLE', fast_path=True, version='1.0.1',
                    kernel=kernel, kernel_version=1, fallbacks={'placeholder_errors': 0},
                    registration=str(stage / 'registration.json'), registration_sha256=sha(stage / 'registration.json'),
                    gate_sha256=sha(gate_path))
        with patch.object(ks, 'ROOT', SCRATCH):
            fr, fv = fixture(kind + '-fast', fast, kernel)
            check(kind + '-fast-gate-admit', fr, fv, 0, kernel)
            fr, fv = fixture(kind + '-stale', {**fast, 'gate_sha256': '0' * 64}, kernel)
            check(kind + '-stale-gate', fr, fv, 2, kernel, 'stale')
            # Fully re-bound wrong parent must still fail.
            put(gate_path, {**gate, 'parent': str(stage)})
            fr, fv = fixture(kind + '-wrong-parent', {**fast, 'gate_sha256': sha(gate_path)}, kernel)
            check(kind + '-wrong-parent', fr, fv, 2, kernel, 'registered parent')
            put(gate_path, gate)

    from tools import strict_variant_gate as sg
    parent_stage = ROOT / '.local/exp152-own0949-v1/stage'
    probe = dict(parent=str(parent_stage), candidate_v102_exists=(ROOT /
                 'artifacts/experiments/HB151_KILLTEST/v1.0.2').exists())
    probe['missing_parent_files'] = [x for x in ['kernel', 'registration.json', 'SHA256SUMS']
                                    if not (parent_stage / x).exists()]
    for label, notebook in [('parent', parent_stage / 'own/release.ipynb'),
            ('candidate_v102', ROOT / 'artifacts/experiments/HB151_KILLTEST/v1.0.2/kernel/release.ipynb')]:
        try:
            sg.unpack_notebook(notebook.read_bytes())
        except ValueError as exc:
            probe[label + '_gate_error'] = str(exc)
        else:
            raise AssertionError('recheck gate compatibility: notebook layout changed')
    old_meta = json.loads((parent_stage / 'own/kernel-metadata.json').read_bytes())
    new_meta = json.loads((ROOT / 'artifacts/experiments/HB151_KILLTEST/v1.0.2/kernel/kernel-metadata.json').read_bytes())
    probe['protected_metadata_differences'] = [k for k in sg.PROTECTED if old_meta[k] != new_meta[k]]
    put(SCRATCH / 'parent-probe.json', probe)

    hb = json.loads(HB127.read_text(encoding='utf-8'))
    try:
        ks._prediction_evidence(hb)
    except ValueError as exc:
        assert 'r_prediction_fallbacks' in str(exc)
    else:
        raise AssertionError('real CPU diagnostic HB127 has 3 R prediction fallbacks')
    results.append(dict(case='real-hb127-prediction-refusal', exit=2, matched=True))
    hb['r7_telemetry']['r_prediction_fallbacks'] = 0  # Test copy only.
    hb['r7_telemetry']['telemetry']['r_fallback_count'] = True
    ks._prediction_evidence(hb)
    results.append(dict(case='hb127-schema-healthy-test-copy', exit=0, matched=True))

    # Full preflight/ledger/one-attempt path with the real receipt schema and a fake CLI.
    ledger, lock = SCRATCH / 'mock-ledger.jsonl', SCRATCH / 'mock-submit.lock'
    assert not lock.exists(), 'previous smoke lock needs inspection'
    ledger.write_text('', encoding='utf-8')
    rows, submits = [], []

    def fake_cli(*args):
        if args[1] == 'submissions':
            return 0, '\n'.join([','.join(ks.HEADER), *rows])
        assert args[1] == 'submit'
        submits.append(args)
        rows.append('1,submission.csv,2026-10-15 00:00:00,[synthetic-smoke] smoke,COMPLETE,,')
        return 0, '4 submissions remaining today.'

    kw = dict(cli=fake_cli, sleep=lambda _: None, poll_tries=1,
              ledger=ledger, lock=lock, receipt=receipt, review=review)
    first, _ = ks.submit_once('synthetic-smoke', KERNEL, 1, 'smoke', **kw)
    second, _ = ks.submit_once('synthetic-smoke', KERNEL, 1, 'smoke', **kw)
    assert first == 0 and second == 2 and len(submits) == 1
    results.append(dict(case='mock-one-attempt', exit=[first, second], submits=1, matched=True))
    assert {p: sha(Path(p)) for p in original_hashes} == original_hashes
    put(SCRATCH / 'results.json', dict(tool_version=ks.VERSION, fixture_only=True,
        original_sha256=original_hashes, results=results, real_endpoint_calls=0))
    for result in results:
        print(f"{result['case']}: exit={result['exit']} matched=True")
    print(f'{len(results)} offline smoke cases matched; real endpoints=0; immutable inputs unchanged')
    return 0


if __name__ == '__main__':
    with patch.object(subprocess, 'run', forbidden), patch.object(socket, 'socket', forbidden), \
            patch.object(ks, 'kaggle_cli', forbidden):
        sys.exit(main())
