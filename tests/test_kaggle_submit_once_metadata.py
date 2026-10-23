"""v1.3.11 hidden-error classification; synthetic SDK/transport, no endpoints."""
import contextlib
import copy
import datetime as dt
import enum
import hashlib
import json
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
from tools import kaggle_submit_once as ks

NOW = dt.datetime(2026, 10, 18, 9, 50, tzinfo=dt.timezone.utc)
ID = 'K2-RCOF0-V1'
ERROR = 'Your notebook hit an unhandled error while rerunning your code.'


@pytest.fixture(autouse=True)
def no_endpoints(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('real subprocess/network forbidden')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)


@pytest.fixture
def admitted(tmp_path):
    registration = tmp_path / 'registration.json'
    registration.write_text(json.dumps(dict(kernel_id='u/k', version='1.0.4')))
    receipt = tmp_path / 'visible.json'
    receipt.write_text(json.dumps(dict(status='COMPLETE', version='1.0.4', kernel='u/k', kernel_version=3,
        registration=str(registration), registration_sha256=hashlib.sha256(registration.read_bytes()).hexdigest(),
        fallbacks={'prediction': 0}, codecs={'status': 'ACTIVE'})))
    review = tmp_path / 'review.md'; review.write_text('u/k\nVERDICT: PASS\n')
    return dict(receipt=receipt, review=review, ledger=tmp_path / 'ledger.jsonl', lock=tmp_path / 'submit.lock',
                now=NOW, sleep=lambda _: None, poll_tries=1)


def row(ref='57347653', status='COMPLETE', error=ERROR, description=None):
    return dict(ref=ref, date=NOW.isoformat(), fileName='submission.csv',
                description=description or f'[{ID}] original', status='SubmissionStatus.' + status,
                error_description=error)


class Transport:
    """Exercise the real production routing, replacing only the OS subprocess."""
    def __init__(self, rows, ledger):
        self.rows = list(rows); self.ledger = ledger; self.sends = 0; self.reads = 0
        self.bad_payload = None; self.exit_code = 0; self.raise_list = False
        self.new_status = 'PENDING'; self.new_error = ''

    def __call__(self, command, **kwargs):
        if command[-1] == '--_list-submission-metadata':
            self.reads += 1
            assert command[0] == str(ks.ROOT / '.venv/Scripts/python.exe')
            assert command[1:4] == ['-X', 'utf8', '-B']
            if self.raise_list: raise TimeoutError('synthetic list failure')
            payload = self.bad_payload if self.bad_payload is not None else dict(schema=ks.METADATA_SCHEMA, rows=self.rows)
            return SimpleNamespace(returncode=self.exit_code, stdout=json.dumps(payload), stderr='never returned SDK chatter')
        assert command[3:7] == ['-m', 'kaggle', 'competitions', 'submit']
        assert ks.read_ledger(self.ledger)[-1]['outcome'] == 'IN_FLIGHT'
        self.sends += 1
        self.rows.insert(0, row('900', self.new_status, self.new_error, command[command.index('-m', 4) + 1]))
        return SimpleNamespace(returncode=0, stdout='4 submissions remaining today.', stderr='')


def call(kwargs, approved=None):
    return ks.submit_once(ID, 'u/k', 3, 'repaired', user_approved_retry=approved, **kwargs)


def previous(kwargs, outcome='SUBMITTED', ref='57347653'):
    ks.append_ledger(dict(id=ID, kernel='u/k', version='2', attempt='old', outcome=outcome, ref=ref), kwargs['ledger'])


def test_complete_hidden_error_requires_explicit_retry_and_allows_one_send(monkeypatch, admitted):
    previous(admitted)
    transport = Transport([row()], admitted['ledger']); monkeypatch.setattr(subprocess, 'run', transport)
    assert call(admitted)[0] == 2 and transport.sends == 0
    code, rec = call(admitted, 'Taeyang 2026-10-18 18:43 KST: fix and resubmit')
    assert code == 0 and transport.sends == 1 and rec['outcome'] == 'SUBMITTED'
    assert rec['status'] == 'SubmissionStatus.PENDING' and rec['hidden_error'] is False
    assert 'acceptance only' in rec['confirmation']
    assert call(admitted, 'same authorization')[0] == 2 and transport.sends == 1
    records = ks.read_ledger(admitted['ledger'])
    assert records[0]['ref'] == '57347653' and records[0]['version'] == '2'
    assert sum(r['outcome'] == 'IN_FLIGHT' for r in records) == 1
    assert records[-2]['outcome'] == 'SUBMITTED' and not admitted['lock'].exists()


@pytest.mark.parametrize('status,error', [('COMPLETE', ''), ('PENDING', ''), ('PENDING', ERROR)])
def test_live_rows_always_block_even_with_retry_words(monkeypatch, admitted, status, error):
    transport = Transport([row(status=status, error=error)], admitted['ledger'])
    monkeypatch.setattr(subprocess, 'run', transport)
    assert call(admitted, 'explicit retry')[0] == 2 and transport.sends == 0


@pytest.mark.parametrize('outcome,ref', [('SUBMITTED', 'absent-ref'), ('IN_FLIGHT', '57347653'), ('UNCONFIRMED', '57347653')])
def test_missing_historical_or_open_attempt_still_blocks(monkeypatch, admitted, outcome, ref):
    previous(admitted, outcome, ref)
    transport = Transport([row()], admitted['ledger']); monkeypatch.setattr(subprocess, 'run', transport)
    assert call(admitted, 'explicit retry')[0] == 2 and transport.sends == 0


@pytest.mark.parametrize('mutation', ['missing_error', 'null_error', 'missing_date', 'bad_date', 'unknown_status',
    'duplicate_ref', 'extra_score', 'bad_envelope', 'bad_schema', 'failed_child', 'exception'])
def test_bad_metadata_never_sends_or_falls_back_to_csv(monkeypatch, admitted, mutation):
    transport = Transport([row()], admitted['ledger'])
    r = transport.rows[0]
    if mutation == 'missing_error': del r['error_description']
    elif mutation == 'null_error': r['error_description'] = None
    elif mutation == 'missing_date': del r['date']
    elif mutation == 'bad_date': r['date'] = 'not-a-date'
    elif mutation == 'unknown_status': r['status'] = 'COMPLETE?'
    elif mutation == 'duplicate_ref': transport.rows.append(copy.deepcopy(r))
    elif mutation == 'extra_score': r['publicScore'] = 'SECRET_TEST_SCORE'
    elif mutation == 'bad_envelope': transport.bad_payload = {'rows': transport.rows}
    elif mutation == 'bad_schema': transport.bad_payload = dict(schema='old', rows=transport.rows)
    elif mutation == 'failed_child': transport.exit_code = 2
    elif mutation == 'exception': transport.raise_list = True
    monkeypatch.setattr(subprocess, 'run', transport)
    code, rec = call(admitted, 'explicit retry')
    assert code == 2 and 'list unreadable' in rec['reason'] and transport.sends == 0
    assert 'SECRET_TEST_SCORE' not in admitted['ledger'].read_text()


def test_lock_and_bad_receipt_prevent_metadata_query(monkeypatch, admitted):
    transport = Transport([], admitted['ledger']); monkeypatch.setattr(subprocess, 'run', transport)
    admitted['lock'].write_text('existing lock')
    assert call(admitted)[0] == 2 and transport.reads == transport.sends == 0
    admitted['lock'].unlink()
    admitted['receipt'].write_text('{}')
    assert call(admitted)[0] == 2 and transport.reads == transport.sends == 0


def test_daily_count_and_new_error_row_only_confirm_acceptance(monkeypatch, admitted):
    rows = [row(str(i+1), 'COMPLETE', ERROR, f'[other{i}] old') for i in range(5)]
    assert ks.used_today(rows, '2026-10-18') == 0
    assert ks.used_today(rows + [row('8', 'PENDING', '')], '2026-10-18') == 1
    transport = Transport(rows, admitted['ledger']); transport.new_status = 'COMPLETE'; transport.new_error = ERROR
    monkeypatch.setattr(subprocess, 'run', transport)
    code, rec = call(admitted)
    assert code == 0 and rec['hidden_error'] is True and rec['error_description'] == ERROR
    assert 'acceptance only' in rec['confirmation'] and transport.sends == 1
    assert call(admitted)[0] == 2 and transport.sends == 1


def test_reconcile_carries_error_metadata_without_resubmission(monkeypatch, admitted):
    previous(admitted, 'UNCONFIRMED')
    transport = Transport([row()], admitted['ledger']); monkeypatch.setattr(subprocess, 'run', transport)
    code, rec = ks.reconcile(ID, 'u/k', 3, ledger=admitted['ledger'], lock=admitted['lock'])
    assert code == 0 and rec['hidden_error'] is True and rec['error_description'] == ERROR and transport.sends == 0


class SubmissionStatus(enum.Enum):
    PENDING = 0
    COMPLETE = 1
    ERROR = 2


class SDKRow:
    ref = 57347653
    date = NOW
    status = SubmissionStatus.COMPLETE
    description = f'[{ID}] original'
    file_name = 'submission.csv'
    error_description = ERROR

    @property
    def public_score(self): raise AssertionError('must not access scores')
    @property
    def private_score(self): raise AssertionError('must not access scores')
    def to_dict(self): raise AssertionError('must not serialize whole SDK object')
    def __repr__(self): raise AssertionError('must not print whole SDK object')


def test_sdk_allowlist_pagination_never_touches_scores():
    second = SDKRow(); second.ref = 57347654
    pages = {'': SimpleNamespace(submissions=[SDKRow()], next_page_token='next'),
             'next': SimpleNamespace(submissions=[second], next_page_token='')}
    seen = []
    def fetch(token): seen.append(token); return pages[token]
    rows = ks._collect_submission_metadata(fetch)
    assert seen == ['', 'next'] and len(rows) == 2 and set(rows[0]) == set(ks.METADATA_FIELDS)
    assert ks.is_error(rows[0]) and 'score' not in json.dumps(rows).lower()


@pytest.mark.parametrize('bad', ['missing_error', 'null_error', 'bad_ref', 'bad_date', 'bad_status', 'missing_token',
    'null_page', 'null_member', 'repeat_token', 'duplicate_ref', 'empty_next', 'api_exception'])
def test_sdk_malformed_or_incomplete_pages_fail_closed(bad):
    item = SDKRow(); response = SimpleNamespace(submissions=[item], next_page_token='')
    if bad == 'missing_error':
        item = SimpleNamespace(**{k:getattr(item,k) for k in ('ref','date','status','description','file_name')}); response.submissions=[item]
    elif bad == 'null_error': item.error_description = None
    elif bad == 'bad_ref': item.ref = True
    elif bad == 'bad_date': item.date = '2026-10-18'
    elif bad == 'bad_status': item.status = 'UNKNOWN'
    elif bad == 'missing_token': del response.next_page_token
    elif bad == 'null_page': response.submissions = None
    elif bad == 'null_member': response.submissions = [None]
    elif bad in ('repeat_token','duplicate_ref'): response.next_page_token = 'next'
    elif bad == 'empty_next': response.submissions = []; response.next_page_token = 'next'
    calls = []
    def fetch(token):
        calls.append(token)
        if bad == 'api_exception': raise RuntimeError('API failure')
        if bad == 'repeat_token':
            new = SDKRow(); new.ref += len(calls); response.submissions = [new]
        return response
    with pytest.raises((ValueError, AttributeError, TypeError, RuntimeError)):
        ks._collect_submission_metadata(fetch)
    assert len(calls) <= 2


def test_sdk_installed_interface_auth_and_token_request(monkeypatch):
    requests = []; auth = []
    def list_submissions(req):
        requests.append(req)
        return SimpleNamespace(submissions=[SDKRow()] if len(requests)==1 else [],
                               next_page_token='page-two' if len(requests)==1 else '')
    client = SimpleNamespace(competitions=SimpleNamespace(competition_api_client=SimpleNamespace(list_submissions=list_submissions)))
    api = SimpleNamespace(authenticate=lambda:auth.append(True), build_kaggle_client=lambda:contextlib.nullcontext(client))
    monkeypatch.setitem(sys.modules, 'kaggle', SimpleNamespace(api=api))
    monkeypatch.setitem(sys.modules, 'kagglesdk.competitions.types.competition_api_service',
                        SimpleNamespace(ApiListSubmissionsRequest=SimpleNamespace))
    assert len(ks._sdk_submission_metadata()) == 1 and auth == [True]
    assert vars(requests[0]) == dict(competition_name=ks.COMP,page_size=100,page=1)
    assert vars(requests[1]) == dict(competition_name=ks.COMP,page_size=100,page_token='page-two')


def test_child_suppresses_chatter_and_exception_detail(monkeypatch, capsys):
    def fail():
        print('synthetic sensitive SDK chatter')
        raise ValueError('synthetic sensitive exception')
    monkeypatch.setattr(ks, '_sdk_submission_metadata', fail)
    assert ks._metadata_child() == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == dict(schema=ks.METADATA_SCHEMA, error='metadata unavailable or malformed')
    assert not captured.err and 'sensitive' not in captured.out


def test_naive_and_offset_sdk_dates_are_normalized_to_utc():
    item=SDKRow(); item.date=dt.datetime(2026,10,18,0,5)
    assert ks._sdk_row(item)['date']=='2026-10-18T00:05:00+00:00'
    item.date=dt.datetime(2026,10,18,0,5,tzinfo=dt.timezone(dt.timedelta(hours=9)))
    assert ks._sdk_row(item)['date']=='2026-10-17T15:05:00+00:00'


def test_legacy_injected_fake_cli_is_supported_but_scores_are_discarded():
    csv=','.join(ks.HEADER)+'\n1,submission.csv,2026-10-18 00:00:00,[synthetic] x,SubmissionStatus.COMPLETE,SCORE_SENTINEL,'
    rows=ks.read_submissions(lambda *args:(0,csv))
    assert set(rows[0])==set(ks.METADATA_FIELDS) and rows[0]['error_description']==''
    assert 'SCORE_SENTINEL' not in json.dumps(rows)
