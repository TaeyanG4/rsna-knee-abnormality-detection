"""v1.3.7 regression mirror of the frozen submit-once suite; all endpoints mocked."""
import datetime as dt
import hashlib
import json
import subprocess
import threading

import pytest

from tools import kaggle_submit_once_v137 as ks

HEADER = 'ref,fileName,date,description,status,publicScore,privateScore'
NOW = dt.datetime(2026, 10, 9, 0, 5, tzinfo=dt.timezone.utc)


def evidence(tmp_path, kernel='u/k', version=2):
    folder = tmp_path / kernel.split('/')[1]
    folder.mkdir(exist_ok=True)
    reg = folder / 'registration.json'
    reg.write_text(json.dumps(dict(kernel_id=kernel, version='1.2.0')), encoding='utf-8')
    receipt = folder / 'visible.json'
    receipt.write_text(json.dumps(dict(status='COMPLETE_C85R7', version='1.2.0', kernel=kernel,
        kernel_version=version, registration=str(reg), registration_sha256=hashlib.sha256(reg.read_bytes()).hexdigest(),
        fallbacks={'a172_prediction': 0, 'r_prediction': 0, 's_r7_prediction': 0}, codecs={'status': 'ACTIVE'})), encoding='utf-8')
    review = folder / 'review.md'
    review.write_text(kernel + '\nVERDICT: PASS\n', encoding='utf-8')
    return receipt, review


@pytest.fixture(autouse=True)
def offline_evidence(tmp_path, monkeypatch):
    # Even an accidental use of the real CLI cannot contact an endpoint.
    def forbidden(*args, **kwargs):
        pytest.fail('real subprocess/network forbidden in submit tests')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    for kernel, version in [('u/k', 2), ('u/other', 2), ('u/k2', 1)]:
        evidence(tmp_path, kernel, version)


def row(ref, desc, status='SubmissionStatus.COMPLETE', date='2026-10-09 00:01:00'):
    return f'{ref},submission.csv,{date},"{desc}",{status},0.9,'


class FakeKaggle:
    """Mimics the current CLI: a successful submit prints only 'N submissions remaining today.'"""

    def __init__(self, rows, submit_out='4 submissions remaining today.', appear=True, raise_on_submit=None):
        self.rows, self.submit_out, self.appear, self.submits = list(rows), submit_out, appear, 0
        self.raise_on_submit = raise_on_submit
        self.lock = threading.Lock()

    def __call__(self, *args):
        if args[1] == 'submissions':
            with self.lock:
                return 0, '\n'.join([HEADER, *self.rows])
        with self.lock:
            self.submits += 1
            n = self.submits
        if self.appear:
            desc = args[args.index('-m') + 1]
            with self.lock:
                self.rows.insert(0, row(str(900 + n), desc, 'SubmissionStatus.PENDING'))
        if self.raise_on_submit:
            raise self.raise_on_submit
        return 0, self.submit_out


def run(fake, tmp_path, sub_id='H-B-31-rad5-v6', kernel='u/k', version=2, polls=10, approved=None):
    return ks.submit_once(sub_id, kernel, version, 'msg, with comma', cli=fake, sleep=lambda s: None, now=NOW,
                          poll_tries=polls, ledger=tmp_path / 'ledger.jsonl', lock=tmp_path / 'submit.lock',
                          user_approved_retry=approved, receipt=tmp_path / kernel.split('/')[1] / 'visible.json',
                          review=tmp_path / kernel.split('/')[1] / 'review.md')


def test_success_without_successfully_word_submits_once(tmp_path):
    fake = FakeKaggle([])
    code, rec = run(fake, tmp_path)
    assert code == 0 and rec['outcome'] == 'SUBMITTED' and fake.submits == 1
    assert not (tmp_path / 'submit.lock').exists()


def test_second_invocation_refuses_duplicate(tmp_path):
    fake = FakeKaggle([])
    run(fake, tmp_path)
    code, rec = run(fake, tmp_path)
    assert code == 2 and fake.submits == 1


def test_same_kernel_version_under_new_id_is_refused(tmp_path):
    fake = FakeKaggle([])
    run(fake, tmp_path)
    code, rec = run(fake, tmp_path, sub_id='H-B-31-rad5-v6-again')
    assert code == 2 and 'already submitted' in rec['reason'] and fake.submits == 1


def test_errored_previous_attempt_needs_user_approval(tmp_path):
    fake = FakeKaggle([row('1', '[H-B-31-rad5-v6] msg', 'SubmissionStatus.ERROR')])
    code, rec = run(fake, tmp_path)
    assert code == 2 and 'already attempted' in rec['reason'] and fake.submits == 0
    code, _ = run(fake, tmp_path, approved='user 2026-10-09 09:30: retry it')
    assert code == 0 and fake.submits == 1


def test_unconfirmed_then_replay_is_blocked_until_reconcile(tmp_path):
    lagging = FakeKaggle([], appear=False)
    code, rec = run(lagging, tmp_path)
    assert code == 3 and rec['outcome'] == 'UNCONFIRMED' and lagging.submits == 1
    code, rec = run(lagging, tmp_path)
    assert code == 2 and 'unresolved' in rec['reason'] and lagging.submits == 1
    # reconcile: the row is still absent -> stays open without --declare-absent
    kw = dict(cli=lagging, ledger=tmp_path / 'ledger.jsonl', lock=tmp_path / 'submit.lock')
    assert ks.reconcile('H-B-31-rad5-v6', 'u/k', 2, **kw)[0] == 3
    # the row shows up later -> reconcile marks it SUBMITTED, and a further attempt is refused as a duplicate
    lagging.rows.insert(0, row('950', '[H-B-31-rad5-v6] msg', 'SubmissionStatus.PENDING'))
    code, rec = ks.reconcile('H-B-31-rad5-v6', 'u/k', 2, **kw)
    assert code == 0 and rec['outcome'] == 'SUBMITTED' and rec['ref'] == '950'
    assert run(lagging, tmp_path)[0] == 2 and lagging.submits == 1


def test_declared_absent_still_needs_user_approval(tmp_path):
    lagging = FakeKaggle([], appear=False)
    run(lagging, tmp_path)
    kw = dict(cli=lagging, ledger=tmp_path / 'ledger.jsonl', lock=tmp_path / 'submit.lock')
    assert ks.reconcile('H-B-31-rad5-v6', 'u/k', 2, declare_absent=True, **kw)[0] == 0
    lagging.appear = True
    assert run(lagging, tmp_path)[0] == 2 and lagging.submits == 1
    assert run(lagging, tmp_path, approved='user: retry')[0] == 0 and lagging.submits == 2


def test_timeout_is_recorded_and_blocks_replay(tmp_path):
    fake = FakeKaggle([], appear=False, raise_on_submit=subprocess.TimeoutExpired('kaggle', 600))
    code, rec = run(fake, tmp_path)
    assert code == 3 and 'TimeoutExpired' in rec['cli_exception'] and fake.submits == 1
    outcomes = [l for l in (tmp_path / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    assert any('IN_FLIGHT' in l for l in outcomes) and any('UNCONFIRMED' in l for l in outcomes)
    assert run(fake, tmp_path)[0] == 2 and fake.submits == 1


def test_lock_blocks_concurrent_caller(tmp_path):
    (tmp_path / 'submit.lock').write_text('pid 1')
    fake = FakeKaggle([])
    code, rec = run(fake, tmp_path)
    assert code == 2 and 'lock exists' in rec['reason'] and fake.submits == 0


def test_concurrent_same_id_submits_once(tmp_path):
    fake = FakeKaggle([])
    results = []
    threads = [threading.Thread(target=lambda: results.append(run(fake, tmp_path))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.submits == 1 and sorted(r[0] for r in results).count(0) == 1


def test_malformed_lists_refuse_and_valid_empty_list_proceeds(tmp_path):
    for bad in ('ref,', 'ref,fileName\n1,x', HEADER + '\nabc,submission.csv,d,"x",s,,', HEADER + '\n1,submission.csv'):
        def cli(*args, out=bad):
            return (0, out) if args[1] == 'submissions' else (0, '4 submissions remaining today.')
        assert run(cli, tmp_path, sub_id='H-B-1-bad')[0] == 2
    assert run(FakeKaggle([]), tmp_path, sub_id='H-B-1-good', kernel='u/other')[0] == 0


def test_daily_limit_and_unreadable_list_refuse(tmp_path):
    full = FakeKaggle([row(str(i), f'[x{i}x] m') for i in range(5)])
    assert run(full, tmp_path)[0] == 2 and full.submits == 0

    def broken(*args):
        return 1, 'HTTPSConnectionPool error'
    assert ks.submit_once('H-B-1-x', 'u/k2', 1, 'm', cli=broken, sleep=lambda s: None, now=NOW,
                          ledger=tmp_path / 'l.jsonl', lock=tmp_path / 'l.lock',
                          receipt=tmp_path / 'k2/visible.json', review=tmp_path / 'k2/review.md')[0] == 2


def rewrite(path, **updates):
    data = json.loads(path.read_text(encoding='utf-8'))
    data.update(updates)
    path.write_text(json.dumps(data), encoding='utf-8')


def no_calls(*args):
    pytest.fail('admission failure must not call list or submit')


@pytest.mark.parametrize('change', [
    {'status': 'RUNNING'}, {'status': 'FAILED'}, {'status': None},
    {'fallbacks': {}}, {'fallbacks': {'a': 1}}, {'fallbacks': {'a': False}},
    {'fallbacks': {'a': '0'}}, {'fallbacks': {'a': -1}}, {'fallbacks': {'a': None}},
    {'fallbacks': {'a': float('nan')}}, {'fallbacks': {'a': []}},
    {'a172': {'prediction_fallbacks': 1}}, {'s_r7_prediction_fallbacks': 1},
    {'nested': [{'r_fallback_count': 2}]},
    {'nested': [[{'r_fallback_count': 2}]]},
    {'telemetry': {'r_fallback_count': True}},
    {'telemetry': {'prediction_fallbacks': True}},
    {'pr7': {'telemetry': {'prediction_fallbacks': False}}},
    {'pr7': {'telemetry': {'r_fallback_count': False}}},
    {'pr7': {'telemetry': {'r_fallback_count': 1}}},
    {'codecs': {'status': 'INACTIVE'}}, {'codecs': None},
    {'codecs': {'status': 'ACTIVE', 's_r7_gate': 'ABSENT'}},
    {'kernel': 'u/other'}, {'kernel_id': 'u/other'}, {'kernel_version': 3},
    {'kernel_version': True}, {'kernel_version': '02'}, {'version': '1.1.0'},
    {'version': 3}, {'version': None}, {'registration_sha256': 'bad'},
    {'version': True}, {'version': 1.2}, {'version': {}}, {'version': 'garbage'},
    {'registration': '../outside/registration.json'}, {'fast_path': 'true'},
    {'status': 'COMPLETE_FAST_PATH_VISIBLE'}, {'fast_path': True}, {'mode': 'FAST'},
])
def test_bad_admission_no_endpoint_calls(tmp_path, change):
    rewrite(tmp_path / 'k/visible.json', **change)
    code, rec = run(no_calls, tmp_path)
    assert code == 2 and rec['reason'].startswith('admission:')
    assert all(r['outcome'] == 'REFUSED' for r in ks.read_ledger(tmp_path / 'ledger.jsonl'))


@pytest.mark.parametrize('review', ['u/k\nVERDICT: FAIL', 'u/k\nVERDICT: PASS\ntrailing',
    'u/k-extra\nVERDICT: PASS', 'u/other\nVERDICT: PASS', 'u/k\nVERDICT: FAIL\nVERDICT: PASS'])
def test_bad_reviews_no_endpoint_calls(tmp_path, review):
    (tmp_path / 'k/review.md').write_text(review, encoding='utf-8')
    assert run(no_calls, tmp_path)[0] == 2


def test_crlf_review_and_boolean_telemetry(tmp_path):
    (tmp_path / 'k/review.md').write_bytes(b'u/k\r\nVERDICT: PASS\r\n')
    rewrite(tmp_path / 'k/visible.json', pr7={'telemetry': {'r_fallback_count': True}})
    assert run(FakeKaggle([]), tmp_path)[0] == 0


def test_missing_arguments_direct_api_refuses(tmp_path):
    assert ks.submit_once('abc', 'u/k', 2, 'm', cli=no_calls,
        ledger=tmp_path / 'ledger.jsonl', lock=tmp_path / 'submit.lock')[0] == 2


@pytest.mark.parametrize('missing', ['--receipt', '--review'])
def test_missing_cli_argument_exits_two(tmp_path, monkeypatch, missing):
    args = ['tool', '--id', 'abc', '--kernel', 'u/k', '--version', '2', '--message', 'm']
    for flag, value in [('--receipt', str(tmp_path / 'k/visible.json')), ('--review', str(tmp_path / 'k/review.md'))]:
        if flag != missing:
            args += [flag, value]
    monkeypatch.setattr('sys.argv', args)
    with pytest.raises(SystemExit) as exc:
        ks.main()
    assert exc.value.code == 2


def test_review_mutates_during_list_refuses_before_submit(tmp_path):
    def cli(*args):
        assert args[1] == 'submissions'
        (tmp_path / 'k/review.md').write_text('u/k\nVERDICT: FAIL', encoding='utf-8')
        return 0, HEADER
    code, rec = run(cli, tmp_path)
    assert code == 2 and 'recheck' in rec['reason']


def test_duplicate_json_keys_refused(tmp_path):
    p = tmp_path / 'k/visible.json'
    p.write_text(p.read_text().replace('"kernel": "u/k"', '"kernel": "u/other", "kernel": "u/k"'))
    assert run(no_calls, tmp_path)[0] == 2


def alias_fixture(tmp_path, bound=True):
    actual = 'taeyangg4/taeyangg4-knee-lb-fluidrd-v1'
    directory = tmp_path / (actual.split('/')[1] + '-visible' if bound else 'unbound')
    directory.mkdir()
    receipt, review = evidence(directory, actual, 1)
    # evidence helper creates a subdirectory; use the actual download directory.
    for path in (receipt, review, receipt.parent / 'registration.json'):
        (directory / path.name).write_bytes(path.read_bytes())
    receipt, review = directory / 'visible.json', directory / 'review.md'
    reg = directory / 'registration.json'
    rewrite(reg, kernel_id='taeyangg4/knee-lb-fluidRd-v1')
    rewrite(receipt, registration=str(reg), registration_sha256=hashlib.sha256(reg.read_bytes()).hexdigest())
    return actual, receipt, review


@pytest.mark.parametrize('bound', [True, False])
def test_prefixed_slug_needs_explicit_directory_binding(tmp_path, bound):
    kernel, receipt, review = alias_fixture(tmp_path, bound)
    if bound:
        admitted = ks.qualify(kernel, 1, receipt, review)
        assert admitted['registered_kernel'] == 'taeyangg4/knee-lb-fluidRd-v1'
    else:
        with pytest.raises(ValueError, match='binding'):
            ks.qualify(kernel, 1, receipt, review)


def test_legacy_sidecar_binds_exact_receipt_and_version(tmp_path):
    kernel, receipt, review = alias_fixture(tmp_path)
    original = json.loads(receipt.read_text())
    transport_keys = ('kernel', 'kernel_version', 'registration', 'registration_sha256')
    raw = {k: v for k, v in original.items() if k not in transport_keys}
    receipt.write_text(json.dumps(raw))
    sidecar = receipt.parent / 'submission-binding.json'
    sidecar.write_text(json.dumps(dict(schema_version='1.0.0',
        receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(), **{k: original[k] for k in transport_keys})))
    assert ks.qualify(kernel, 1, receipt, review)['kernel'] == kernel
    with pytest.raises(ValueError, match='version mismatch'):
        ks.qualify(kernel, 2, receipt, review)
    receipt.write_text(receipt.read_text() + '\n')
    with pytest.raises(ValueError, match='binding hash mismatch'):
        ks.qualify(kernel, 1, receipt, review)


def gate_fixture(tmp_path, monkeypatch, parent_kind='c85', gate_version='1.1.0', kernel='u/k'):
    # Real immutable parent kernel bytes, synthetic PASS only in isolated project.
    source = ks.ROOT / ('artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0/kernel'
                        if parent_kind == 'c85' else '.local/exp152-own0949-v1/stage/own')
    parent = tmp_path / ('C85-R7-v1.1.0' if parent_kind == 'c85' else 'EXP143v2-ref57005140')
    (parent / 'kernel').mkdir(parents=True)
    for name in ('release.ipynb', 'kernel-metadata.json'):
        (parent / 'kernel' / name).write_bytes((source / name).read_bytes())
    identity = json.loads((parent / 'kernel/kernel-metadata.json').read_bytes())['id']
    (parent / 'registration.json').write_text(json.dumps(dict(kernel_id=identity)))
    (parent / 'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() +
        '  ' + p.relative_to(parent).as_posix() + '\n' for p in
        [parent / 'registration.json', *sorted((parent / 'kernel').iterdir())]))
    monkeypatch.setattr(ks, 'ROOT', tmp_path)
    stage = tmp_path / kernel.split('/')[1]
    if not stage.exists():
        evidence(tmp_path, kernel)
    if kernel == 'taeyangg4/knee-hb151-pl-coat-v1':
        rewrite(stage / 'registration.json', version='1.0.2')
        rewrite(stage / 'visible.json', version='1.0.2')
    rewrite(stage / 'registration.json', parent=str(parent),
            parent_sha256sums=hashlib.sha256((parent / 'SHA256SUMS').read_bytes()).hexdigest())
    rewrite(stage / 'visible.json', registration_sha256=hashlib.sha256((stage / 'registration.json').read_bytes()).hexdigest())
    (stage / 'kernel').mkdir()
    (stage / 'kernel/kernel-metadata.json').write_text(json.dumps(dict(id=kernel)))
    (stage / 'kernel/code.py').write_text('print(1)')
    (stage / 'SHA256SUMS').write_text('fixture manifest')
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'tools/strict_variant_gate.py').write_text('# fixture gate')
    inputs = [stage / 'registration.json', stage / 'SHA256SUMS',
              *list((stage / 'kernel').iterdir()), tmp_path / 'tools/strict_variant_gate.py',
              parent / 'registration.json', parent / 'SHA256SUMS', *list((parent / 'kernel').iterdir())]
    gate = dict(gate_version=gate_version, status='PASS', errors=[], kernel_id=kernel, stage=str(stage), parent=str(parent),
        input_sha256={str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    gpath = tmp_path / '.local/state/gates' / (kernel.split('/')[1] + '.json')
    gpath.parent.mkdir(parents=True)
    gpath.write_text(json.dumps(gate))
    rewrite(stage / 'visible.json', fast_path=True, gate_sha256=hashlib.sha256(gpath.read_bytes()).hexdigest())
    return gpath, stage


@pytest.mark.parametrize('parent_kind,kernel', [('c85', 'u/k'), ('exp143', 'taeyangg4/knee-hb151-pl-coat-v1')])
@pytest.mark.parametrize('gate_version', ['1.0.0', '1.1.0', '1.2.0'])
def test_fast_gate_pass_records_all_hashes(tmp_path, monkeypatch, parent_kind, kernel, gate_version):
    gate_fixture(tmp_path, monkeypatch, parent_kind, gate_version, kernel)
    code, rec = run(FakeKaggle([]), tmp_path, kernel=kernel)
    assert code == 0 and rec['admission']['fast_path'] is True
    assert len(rec['admission']['input_sha256']) >= 8
    assert ks.read_ledger(tmp_path / 'ledger.jsonl')[0]['admission'] == rec['admission']


@pytest.mark.parametrize('parent_kind', ['c85', 'exp143'])
@pytest.mark.parametrize('mutation', ['no_gate', 'gate_parent', 'no_registered_parent', 'manifest',
    'parent_coverage', 'parent_pin', 'parent_added', 'unsupported', 'stage'])
def test_registered_parent_fail_closed(tmp_path, monkeypatch, parent_kind, mutation):
    gate, stage = gate_fixture(tmp_path, monkeypatch, parent_kind)
    data = json.loads(gate.read_text())
    reg = stage / 'registration.json'
    parent = ks._local(data['parent'])
    if mutation == 'no_gate':
        gate.unlink()
        assert run(no_calls, tmp_path)[0] == 2
        return
    if mutation == 'gate_parent':
        data['parent'] = str(tmp_path / 'other-parent')
    elif mutation == 'no_registered_parent':
        rewrite(reg, parent=None)
    elif mutation == 'manifest':
        rewrite(reg, parent_sha256sums='0' * 64)
    elif mutation == 'parent_coverage':
        del data['input_sha256'][str((parent / 'registration.json').relative_to(tmp_path))]
    elif mutation == 'parent_pin':
        (parent / 'kernel/release.ipynb').write_text('{}')
    elif mutation == 'parent_added':
        (parent / 'kernel/extra.py').write_text('extra')
    elif mutation == 'unsupported':
        rewrite(parent / 'kernel/kernel-metadata.json', id='test/unsupported')
        rewrite(parent / 'registration.json', kernel_id='test/unsupported')
    elif mutation == 'stage':
        data['stage'] = str(parent)
    # Refresh hashes: these are semantic binding failures, not merely stale bytes.
    data['input_sha256'] = {p: hashlib.sha256(ks._local(p).read_bytes()).hexdigest()
                            for p in data['input_sha256']}
    gate.write_text(json.dumps(data))
    rewrite(stage / 'visible.json', registration_sha256=hashlib.sha256(reg.read_bytes()).hexdigest(),
            gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest())
    assert run(no_calls, tmp_path)[0] == 2


@pytest.mark.parametrize('mutation', ['absent', 'status', 'kernel', 'version', 'hash', 'input', 'new_file', 'rename', 'empty', 'reg'])
def test_fast_gate_fail_closed(tmp_path, monkeypatch, mutation):
    gate, stage = gate_fixture(tmp_path, monkeypatch)
    if mutation == 'absent':
        gate.unlink()
    elif mutation == 'hash':
        rewrite(stage / 'visible.json', gate_sha256='0' * 64)
    elif mutation == 'input':
        (stage / 'kernel/code.py').write_text('changed')
    elif mutation == 'new_file':
        (stage / 'kernel/extra.py').write_text('extra')
    elif mutation == 'rename':
        (stage / 'kernel/code.py').rename(stage / 'kernel/renamed.py')
    elif mutation == 'reg':
        rewrite(stage / 'registration.json', kernel_id='u/other')
    else:
        changes = {'status': {'status': 'FAIL'}, 'kernel': {'kernel_id': 'u/other'},
            'version': {'gate_version': '0.9.0'}, 'empty': {'input_sha256': {}}}
        rewrite(gate, **changes[mutation])
        rewrite(stage / 'visible.json', gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest())
    assert run(no_calls, tmp_path)[0] == 2


def test_fast_marker_requires_gate_even_without_fast_flag(tmp_path):
    (tmp_path / 'k/FAST_PATH_VISIBLE').write_text('FAST')
    assert run(no_calls, tmp_path)[0] == 2


@pytest.mark.parametrize('bad', [False, True])
def test_check_only_never_calls_endpoints_or_writes_ledger(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(ks, 'submit_once', no_calls)
    monkeypatch.setattr(ks, 'reconcile', no_calls)
    if bad:
        rewrite(tmp_path / 'k/visible.json', status='FAILED')
    monkeypatch.setattr('sys.argv', ['tool', '--check-only', '--id', 'abc', '--kernel', 'u/k',
        '--version', '2', '--receipt', str(tmp_path / 'k/visible.json'), '--review', str(tmp_path / 'k/review.md')])
    assert ks.main() == (2 if bad else 0)
    assert not (tmp_path / 'ledger.jsonl').exists()


@pytest.mark.parametrize('persistent', [False, True])
def test_windows_lock_cleanup_bounded_retry(tmp_path, monkeypatch, persistent):
    path = tmp_path / 'retry.lock'
    path.write_text('owned')
    calls, sleeps = [], []
    original = type(path).unlink

    def unlink(self, **kwargs):
        calls.append(self)
        if persistent or len(calls) < 3:
            raise PermissionError('sharing violation')
        original(self, **kwargs)

    monkeypatch.setattr(type(path), 'unlink', unlink)
    ks._release_lock(path, sleep=sleeps.append)
    assert len(calls) == (40 if persistent else 3)
    assert len(sleeps) == len(calls) - 1 and all(s == 0.05 for s in sleeps)
    assert path.exists() == persistent


@pytest.mark.parametrize('source,registered,kernel', [
    ('taeyangg4-knee-lb-fluidrd-v1-visible', 'v1.1.9-fluidR',
     'taeyangg4/taeyangg4-knee-lb-fluidrd-v1'),
    ('knee-exp169-c85r7codec-v1', 'v1.1.0', 'taeyangg4/knee-exp169-c85r7codec-v1'),
])
def test_immutable_real_receipt_accepted(tmp_path, source, registered, kernel):
    original = ks.ROOT / '.local/scratch/slot-b' / source / 'c85r7-receipt.json'
    raw = original.read_bytes()
    data = json.loads(raw)
    assert data['s_r7_anatomy_anchors']['fallback'] == 1
    assert data['pr7']['s_r7_anatomy_anchors']['fallback'] == 1
    folder = tmp_path / (kernel.split('/')[1] + '-visible')
    folder.mkdir()
    receipt, review = folder / 'visible.json', folder / 'review.md'
    receipt.write_bytes(raw)
    review.write_text('TEST ONLY: ' + kernel + '\nVERDICT: PASS\n', encoding='utf-8')
    reg = ks.ROOT / 'artifacts/experiments/EXP169_RELEASE_C85R7' / registered / 'registration.json'
    binding = dict(schema_version='1.0.0', receipt_sha256=hashlib.sha256(raw).hexdigest(),
        kernel=kernel, kernel_version=1, registration=str(reg),
        registration_sha256=hashlib.sha256(reg.read_bytes()).hexdigest())
    (folder / 'submission-binding.json').write_text(json.dumps(binding), encoding='utf-8')
    assert ks.qualify(kernel, 1, receipt, review)['kernel'] == kernel
    assert receipt.read_bytes() == original.read_bytes() == raw


@pytest.mark.parametrize('field', ['a172_prediction', 'r_prediction', 's_r7_prediction'])
@pytest.mark.parametrize('bad', [1, -1, True, '0', None, float('nan')])
def test_prediction_counter_must_be_numeric_zero(tmp_path, field, bad):
    data = json.loads((tmp_path / 'k/visible.json').read_text())
    data['fallbacks'][field] = bad
    rewrite(tmp_path / 'k/visible.json', **data)
    assert run(no_calls, tmp_path)[0] == 2


@pytest.mark.parametrize('nested', [False, True])
def test_anchor_methods_do_not_block_prediction_admission(tmp_path, nested):
    change = {'s_r7_anatomy_anchors': {'fallback': 1, 'methods': {'slab_prior': 1}}}
    if nested:
        change = {'fallbacks': {'a172_prediction': 0, 'r_prediction': 0,
                               's_r7_prediction': 0, **change}}
    rewrite(tmp_path / 'k/visible.json', **change)
    assert run(FakeKaggle([]), tmp_path)[0] == 0


@pytest.mark.parametrize('mutation', ['healthy', 'missing', 'prediction', 'no_counters', 'no_r7'])
def test_hb127_prediction_equivalents(tmp_path, mutation):
    path = tmp_path / 'k/visible.json'
    data = json.loads(path.read_text())
    data.pop('fallbacks')
    data.update(status='COMPLETE_HB127', missing={'R2-s43': 0, 'R2-s44': 0, 'S_R7': 0, 'EXP155': 0},
        r7_telemetry={'r_prediction_fallbacks': 0, 's_r7_prediction_fallbacks': 0,
                      's_r7_anatomy_anchors': {'fallback': 1},
                      'telemetry': {'r_fallback_count': True}})
    if mutation == 'missing':
        data['missing']['EXP155'] = 1
    elif mutation == 'prediction':
        data['r7_telemetry']['s_r7_prediction_fallbacks'] = 1
    elif mutation == 'no_counters':
        data.pop('missing')
    elif mutation == 'no_r7':
        data.pop('r7_telemetry')
    path.write_text(json.dumps(data), encoding='utf-8')
    assert run(FakeKaggle([]) if mutation == 'healthy' else no_calls, tmp_path)[0] == (0 if mutation == 'healthy' else 2)
