import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from monitoring.kneeops import collectors
from monitoring.kneeops.api import create_app
from monitoring.kneeops.history import claimed_place, collect_history, is_solution, solution_order
from monitoring.kneeops.service import Service, schedule_status
from monitoring.kneeops.store import Store, dumps
from monitoring.kneeops.textutil import html_text


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv('KNEE_WORKSPACE_ROOT', raising=False)
    root = tmp_path / 'project'; (root / 'config').mkdir(parents=True)
    (root / 'config/workspace.toml').write_text('schema_version=1\ncompetition="test-competition"\n', encoding='utf-8')
    return root


def sdk_topic(tid=10, comments=2, content='<p>Opening <strong>post</strong></p>'):
    tree = [{'id': 100, 'authorName': 'b', 'postDate': '2026-09-01T00:00:00Z', 'votes': 3, 'content': '<p>first comment</p>',
             'replies': [{'id': 101, 'authorName': 'c', 'postDate': '2026-09-02T00:00:00Z', 'content': '<p>nested reply</p>'}]}]
    return {'id': tid, 'complete': True, 'pages': 1, 'comments': tree[:comments],
            'topic': {'id': tid, 'title': 'A topic', 'authorName': 'a', 'postDate': '2026-08-01T00:00:00Z', 'votes': 5,
                      'commentCount': comments, 'content': content}}


def test_html_text_keeps_structure_and_links():
    text = html_text('<h2>Plan</h2><ul><li><p>Crop</p></li><li><code>x=1</code></li></ul>'
                     '<p>See <a href="https://example.com/a">paper</a> and <a href="https://example.com/b">https://example.com/b</a></p>'
                     '<pre><code>a  =  1\nb=2</code></pre>')
    assert '## Plan' in text and '- Crop' in text and '- `x=1`' in text
    assert 'paper (https://example.com/a)' in text and 'https://example.com/b (' not in text
    assert 'a  =  1\nb=2' in text and html_text(None) == '' and html_text('plain') == 'plain'


def test_saved_topic_has_opening_post_and_reply_tree(root):
    s = Store(root)
    s.upsert_source(id='topic:10', kind='discussion', title='A topic', content='listing stub')
    s.upsert_source(id='topic:10', kind='discussion', title='A topic', content='old CLI comments only')
    collectors.save_topic(s, sdk_topic(), kind='solution')
    src = s.source('topic:10'); body = src['revision']['content']; payload = src['revision']['payload']
    assert src['kind'] == 'solution' and len(src['revisions']) == 3
    assert body.index('[원글 10] a') < body.index('[댓글 100] b') < body.index('[답글 101 → 100] c')
    assert 'Opening post' in body and payload['original_post'] and payload['fetched_comments'] == 2
    assert payload['body_source'] == 'kaggle-sdk' and payload['topic']['commentCount'] == 2
    assert (s.state / payload['raw_file']).is_file()
    missing = collectors.render_topic('Gone', {'commentCount': 0}, [])
    assert '원글 본문이 비어 있거나 삭제되었습니다' in missing


def test_topic_queue_prioritises_unfetched_then_new_comments_and_defers_recruiting(root):
    s = Store(root)
    collectors.save_topic(s, {**sdk_topic(1, 1), 'id': 1})
    listing = [
        {'id': 1, 'title': 'Already fetched', 'votes': 1, 'commentCount': 4, 'postDate': '2026-08-01'},
        {'id': 2, 'title': 'Unfetched small', 'votes': 1, 'commentCount': 0, 'postDate': '2026-09-01'},
        {'id': 3, 'title': 'Unfetched popular', 'votes': 50, 'commentCount': 20, 'postDate': '2026-08-01'},
        {'id': 4, 'title': 'Looking for teammates', 'votes': 99, 'commentCount': 99, 'postDate': '2026-09-01'},
    ]
    collectors.register_topics(s, 'test-competition', listing)
    assert [t['id'] for t in collectors.topic_queue(s, 'test-competition', listing)] == [3, 2, 1, 4]  # recruiting thread last, not dropped
    listing[0]['commentCount'] = 1
    assert 1 not in [t['id'] for t in collectors.topic_queue(s, 'test-competition', listing)]
    # A listing stub never replaces a fetched body.
    collectors.register_topics(s, 'test-competition', listing)
    assert s.source('topic:1')['revision']['payload']['body_source'] == 'kaggle-sdk'


def test_listing_pages_end_on_plain_text(root):
    pages = {1: json.dumps([{'id': i, 'title': f't{i}'} for i in range(20)]), 2: 'No topics found'}
    with patch.object(collectors, 'kaggle', side_effect=lambda args, timeout=90: pages[int(args[args.index('--page') + 1])]):
        listing, complete = collectors.topic_listing('c')
    assert len(listing) == 20 and complete
    with patch.object(collectors, 'kaggle', return_value='Not found'):
        assert collectors.kaggle_page(['kernels', 'list'], 'kernels', 1) == []
    with patch.object(collectors, 'kaggle', return_value='unexpected text'):
        with pytest.raises(ValueError):
            collectors.kaggle_page(['kernels', 'list'], 'kernels', 1)


def test_notebook_catalog_registration_and_queues(root):
    s = Store(root)
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
    rows = [
        {'ref': 'a/popular', 'title': 'Popular', 'author': 'a', 'lastRunTime': '2026-09-01T00:00:00', 'totalVotes': 40},
        {'ref': 'b/claim', 'title': 'Knee public0946 blend', 'author': 'b', 'lastRunTime': '2026-09-02T00:00:00', 'totalVotes': 0},
        {'ref': 'c/fork', 'title': '0.941 fork', 'author': 'c', 'lastRunTime': '2026-09-03T00:00:00', 'totalVotes': 0},
        {'ref': 'me/mine', 'title': 'Mine', 'author': 'me', 'lastRunTime': recent, 'totalVotes': 1},
    ]
    def page(args, key, page):
        sort = args[args.index('--sort-by') + 1]
        return (rows if sort == 'dateRun' else rows[:1]) if page == 1 else []
    with patch.object(collectors, 'kaggle_page', side_effect=page), patch.object(collectors.time, 'sleep'):
        catalog = collectors.notebook_catalog('test-competition')
    items = {i['ref']: i for i in catalog['items']}
    assert catalog['total'] == 4 and catalog['complete'] and items['b/claim']['title_score_claim'] == 0.946
    assert items['a/popular']['score_rank'] == 1 and 'score_rank' not in items['b/claim']
    assert collectors.register_notebooks(s, 'test-competition', catalog, 'me') == 3
    assert s.source('notebook:me/mine')['kind'] == 'own_notebook'
    assert not s.rows("SELECT id FROM sources WHERE id='notebook:c/fork'")
    new, changed = collectors.notebook_queues(s, 'test-competition', catalog)
    assert [i['ref'] for i in new] == ['b/claim', 'a/popular', 'me/mine'] and changed == []
    # A versioned download older than the latest run is queued for refresh.
    s.upsert_source(id='notebook:a/popular', kind='notebook', title='Popular', content='code', version='3', competition='test-competition',
                    payload={'source_downloaded': True, 'listing': {'lastRunTime': '2026-08-01T00:00:00'}})
    with s.connect() as con: con.execute("UPDATE sources SET observed_at='2026-08-01T00:00:00+00:00' WHERE id='notebook:a/popular'")
    new, changed = collectors.notebook_queues(s, 'test-competition', catalog)
    assert 'a/popular' not in [i['ref'] for i in new] and [i['ref'] for i in changed] == ['a/popular']


def test_leaderboard_summary_finds_own_rank_and_medal_zones():
    lines = ['﻿Rank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,TeamMemberUserNames']
    lines += [f'{i},{i},team{i},2026-10-01,{0.99 - i * 0.00001:.5f},1,user{i}' for i in range(1, 1201)]
    lines[600] = '600,600,Mine,2026-10-01,0.98400,7,someone, me'
    def fake(args, timeout=90):
        folder = Path(args[args.index('-p') + 1])
        with zipfile.ZipFile(folder / 'lb.zip', 'w') as z: z.writestr('lb.csv', '\n'.join(lines).replace(',someone, me', ',"someone, me"'))
        return ''
    with patch.object(collectors, 'kaggle', side_effect=fake):
        summary = collectors.leaderboard_summary('c', 'ME')
    assert summary['teams'] == 1200 and summary['own']['rank'] == 600 and summary['own']['score'] == 0.984
    assert [m['rank'] for m in summary['medals']] == [12, 60, 120]
    assert summary['rank_scores']['1'] == pytest.approx(0.98999)


def test_solution_titles_and_place_order():
    assert is_solution('1st Place Solution') and is_solution('[7th solution, my part] Single Stage')
    assert is_solution('RSNA 2024: Bronze Medal ( 176th) Solution')
    assert not is_solution('What\'s the solution for the following problem? Any suggestions?')
    assert not is_solution('Do we need to resample the pixels to isomorphic resolution?')
    assert not is_solution('Welcome to the challenge')
    assert claimed_place('55st Place Solution') == 55 and claimed_place('51 Place Solution') == 51
    assert claimed_place('Summary of Top Team Solutions') is None
    topics = [{'title': 'Summary of Top Team Solutions', 'votes': 99}, {'title': '10th place', 'votes': 1}, {'title': '2nd place', 'votes': 1}]
    assert [t['title'] for t in sorted(topics, key=solution_order)] == ['2nd place', '10th place', 'Summary of Top Team Solutions']


def test_history_fetches_solutions_by_place_within_budget(root):
    s = Store(root)
    listing = [{'id': i, 'title': t, 'votes': v, 'commentCount': 0, 'postDate': '2025-10-15'} for i, t, v in
               [(1, '9th place solution', 50), (2, '1st Place Solution', 10), (3, 'Overview of the clinical problem', 90),
                (4, 'How to submit?', 5), (5, '3rd place solution', 1)]]
    fetched = []
    def fake_collect(store, items, progress=lambda x: None):
        fetched.extend(items)
        for tid, comp, kind in items: collectors.save_topic(store, {**sdk_topic(tid, 0), 'id': tid}, comp, kind)
        return {'fetched': len(items), 'errors': []}
    with patch('monitoring.kneeops.history.kaggle', return_value='[]'), \
         patch('monitoring.kneeops.history.kaggle_page', return_value=[]), \
         patch('monitoring.kneeops.history.topic_listing', return_value=(listing, True)), \
         patch('monitoring.kneeops.history.collect_topics', side_effect=fake_collect), \
         patch('monitoring.kneeops.history.time.sleep'):
        result = collect_history(s, 'past-comp', solution_budget=2, discussion_budget=1)
    assert [x[0] for x in fetched] == [2, 5, 3] and [x[2] for x in fetched] == ['solution', 'solution', 'discussion']
    assert result['solutions'] == 3 and result['solution_bodies'] == 2 and result['discussion_bodies'] == 1
    assert s.source('topic:1')['kind'] == 'solution' and s.source('topic:4')['kind'] == 'discussion'
    assert json.loads(s.get_meta('topic_catalog:past-comp'))['complete']


def test_schedule_survives_restart_and_does_not_retry_every_tick(root):
    s = Store(root); service = Service(s)
    clock = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    try:
        with patch.object(service, 'enqueue') as enqueue:
            assert not service.schedule_once(clock) and not schedule_status(s, clock)['due']
            s.set_meta('preferences', dumps({'sync_enabled': True, 'sync_minutes': 60}))
            s.set_meta('last_sync_attempt', (clock - timedelta(minutes=30)).isoformat())
            status = schedule_status(s, clock)
            assert status['next_due'] == (clock + timedelta(minutes=30)).isoformat() and not status['due']
            assert not service.schedule_once(clock)
            later = clock + timedelta(minutes=31)
            assert service.schedule_once(later); enqueue.assert_called_once_with('scheduled')
            # The run crashed before recording an attempt: wait a full interval, not 10 seconds.
            assert not service.schedule_once(later + timedelta(seconds=10))
            assert service.schedule_once(later + timedelta(minutes=60))
            enqueue.side_effect = ValueError('busy')
            assert not service.schedule_once(later + timedelta(minutes=200))
    finally:
        service.close()
    assert Service(s).close() is None and schedule_status(s, later + timedelta(minutes=121))['due']


def test_schedule_api_merges_only_schedule_fields(root):
    app = create_app(root)
    with TestClient(app) as client:
        token = client.get('/api/session').json()['token']; headers = {'X-Kneeops-Token': token}
        client.put('/api/preferences', headers=headers, json={'theme': 'dark', 'sync_history': ['past-comp'], 'research_view': 'cards'})
        assert client.put('/api/schedule', json={'enabled': True, 'minutes': 120}).status_code == 403
        assert client.put('/api/schedule', headers=headers, json={'enabled': True, 'minutes': 5}).status_code == 422
        body = client.put('/api/schedule', headers=headers, json={'enabled': True, 'minutes': 120}).json()
        assert body['enabled'] and body['minutes'] == 120 and body['coverage']['notebooks']['live'] == 0
        prefs = client.get('/api/preferences').json()
        assert prefs['theme'] == 'dark' and prefs['sync_history'] == ['past-comp'] and prefs['sync_minutes'] == 120
        assert client.get('/api/schedule').json()['history'] == ['past-comp']


def test_sync_respects_budgets_and_records_connectors(root):
    s = Store(root)
    catalog = {'observed_at': '2026-10-02', 'total': 3, 'complete': True, 'items': [
        {'ref': f'u/n{i}', 'title': f'n{i}', 'author': 'u', 'lastRunTime': '2026-09-01T00:00:00', 'totalVotes': 10 + i} for i in range(3)]}
    listing = [{'id': i, 'title': f't{i}', 'votes': i, 'commentCount': 0, 'postDate': '2026-09-01'} for i in range(1, 6)]
    got = {}
    def fake_notebooks(store, comp, items, progress=None, label='new'):
        got[label] = [i['ref'] for i in items]; return {'fetched': len(items), 'errors': []}
    def fake_topics(store, items, progress=None):
        got['topics'] = [t[0] for t in items]; return {'fetched': len(items), 'errors': []}
    s.set_meta('competition', dumps({'id': 7}))
    with patch.object(collectors, 'kaggle', return_value='[]'), \
         patch('monitoring.kneeops.kaggle_web.competition_notebooks', return_value={'u/n0': {'public_score': .95, 'medal': None, 'runtime_s': 1}}), \
         patch.object(collectors, 'kaggle_username', return_value='me'), \
         patch.object(collectors, 'leaderboard_summary', return_value={'teams': 9, 'own': None}), \
         patch.object(collectors, 'notebook_catalog', return_value=catalog), \
         patch.object(collectors, 'topic_listing', return_value=(listing, True)), \
         patch.object(collectors, 'collect_notebooks', side_effect=fake_notebooks), \
         patch.object(collectors, 'collect_topics', side_effect=fake_topics), \
         patch.object(collectors.time, 'sleep'):
        result = collectors.sync_kaggle(s, topic_budget=2, notebook_budget=1)
    # u/n0 has a verified public score, which outranks u/n2's higher vote count.
    assert not result['errors'] and got['topics'] == [5, 4] and got['new'] == ['u/n0'] and got['노트북 버전'] == []
    connectors = {r['name']: r for r in s.rows('SELECT * FROM connectors')}
    assert {'leaderboard_full', 'notebook_scores', 'notebook_sources', 'notebook_versions', 'discussions'} <= set(connectors)
    assert s.source_meta('notebook:u/n0')['public_score'] == .95
    assert '대기 3개' in connectors['discussions']['detail'] and s.get_meta('last_sync_attempt')
