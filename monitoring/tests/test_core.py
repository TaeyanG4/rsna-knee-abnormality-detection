import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from monitoring.kneeops.api import create_app
from monitoring.kneeops.importer import compare_runs, import_local
from monitoring.kneeops.service import Service, backup
from monitoring.kneeops.store import Store, dumps
from monitoring.kneeops.research import strategy_excerpts


@pytest.fixture
def root(tmp_path,monkeypatch):
    monkeypatch.delenv('KNEE_WORKSPACE_ROOT',raising=False)
    root=tmp_path/'relocated project';(root/'config').mkdir(parents=True)
    (root/'config/workspace.toml').write_text('schema_version=1\ncompetition="test-competition"\n[paths]\nexperiments="artifacts/experiments"\nreports="research/reports"\nkaggle_notebooks="research/notebooks"\n',encoding='utf-8')
    return root


def test_revisions_first_added_and_old_version(root):
    s=Store(root)
    a=s.upsert_source(id='notebook:u/test',kind='notebook',title='test',content='first',version='1',observed_at='2020-01-01')
    added=s.source('notebook:u/test')['added_at']
    b=s.upsert_source(id='notebook:u/test',kind='notebook',title='test',content='second',version='2',observed_at='2021-01-01')
    s.upsert_source(id='notebook:u/test',kind='notebook',title='test',content='first',version='1',observed_at='2022-01-01',promote=False)
    item=s.source('notebook:u/test')
    assert item['current_revision']==b and item['added_at']==added
    assert len(item['revisions'])==2 and s.source('notebook:u/test',a)['revision']['content']=='first'
    s.upsert_source(id='notebook:u/test',kind='notebook',title='test',content='second',version='2',observed_at='2023-01-01')
    assert len(s.source('notebook:u/test')['revisions'])==2


def test_search_literal_and_competition_scope(root):
    s=Store(root)
    s.upsert_source(id='a',kind='discussion',title='한글 제목',content='alpha OR beta',competition='past-comp')
    s.upsert_source(id='b',kind='notebook',title='다른 자료',content='alpha',competition='now-comp')
    assert s.sources('한글')['total']==1
    assert s.sources('alpha OR beta')['total']==1
    assert s.sources(competition='past-comp')['items'][0]['id']=='a'
    # FTS ignores punctuation, but OR must remain a literal token, not a query operator.
    assert s.sources('" OR "')['total']==1


def test_import_idempotent_and_metric_contract(root):
    folder=root/'artifacts/experiments/EXP_SPECIALIST';folder.mkdir(parents=True)
    (folder/'summary.json').write_text(json.dumps({'selected_targets':['ACL','MCL','Effusion','Synovitis'],'blend50_macro_auc':.9,'rows':100}),encoding='utf-8')
    s=Store(root)
    import_local(s);import_local(s)
    runs=s.runs();assert len(runs)==1
    assert runs[0]['kind']=='weak_oof' and len(runs[0]['targets'])==4
    other=dict(runs[0]);other['id']='second';other['score']=.92
    with patch.object(s,'runs',return_value=[runs[0],other]):
        result=compare_runs(s,[runs[0]['id'],'second'])
        assert not result['comparable'] and result['delta'] is None
        assert any('label_hash' in r for r in result['reasons'])


def test_comparison_requires_labels_and_all_versions(root):
    s=Store(root)
    base={'id':'a','kind':'weak_oof','targets':['ACL'],'rows_count':20,'folds':[0,1],'score':.8,
          'contract':{k:'hash' for k in ['split_hash','mask_hash','study_set_hash','label_hash','transform_hash']}}
    base['contract'].update(label_source='workspace://labels/x',metric='auc')
    other={**base,'id':'b','score':.81}
    with patch.object(s,'runs',return_value=[base,other]):
        assert compare_runs(s,['a','b'])['comparable']
        other['targets']=['MCL']
        assert compare_runs(s,['a','b'])['delta'] is None


def test_api_security_preferences_and_diff(root):
    app=create_app(root)
    with TestClient(app) as client:
        assert client.get('/api/health',headers={'Host':'evil.example'}).status_code==403
        assert client.get('/api/session',headers={'Origin':'https://evil.example'}).status_code==403
        assert client.put('/api/preferences',json={}).status_code==403
        token=client.get('/api/session').json()['token'];headers={'X-Kneeops-Token':token}
        assert client.put('/api/preferences',headers=headers,json={'sync_minutes':1}).status_code==422
        assert client.put('/api/preferences',headers=headers,json={'watch_targets':['wrong']}).status_code==400
        assert client.put('/api/preferences',headers=headers,json={'sync_minutes':180,'sync_enabled':True,'research_view':'list'}).status_code==200
        assert client.get('/api/preferences').json()['sync_minutes']==180
        s=app.state.store
        a=s.upsert_source(id='x',kind='notebook',title='x',content='line1',version='1')
        b=s.upsert_source(id='x',kind='notebook',title='x',content='line2',version='2')
        result=client.get('/api/source/diff',params={'id':'x','before':a,'after':b}).json()
        assert '-line1' in result['diff'] and '+line2' in result['diff']
        assert client.get('/api/sources').json()['items'][0]['revision_count']==2
        assert client.post('/api/jobs',headers=headers,json={'kind':'topic','ref':'bad'}).status_code==400


def test_independent_state_and_backup(root,tmp_path):
    s=Store(root,tmp_path/'state1');other=Store(root,tmp_path/'state2')
    assert s.instance!=other.instance
    s.upsert_source(id='local',kind='report',title='one',content='data')
    assert other.sources()['total']==0
    path=backup(s)
    with sqlite3.connect(path) as con:
        assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert con.execute('SELECT count(*) FROM sources').fetchone()[0]==1


def test_job_exclusion_failure_and_recovery(root):
    s=Store(root);started=threading.Event();release=threading.Event()
    def fake(store,progress):
        started.set();release.wait(4);raise RuntimeError('simulated offline')
    with patch('monitoring.kneeops.service.sync_kaggle',side_effect=fake):
        service=Service(s)
        job=service.enqueue('sync');assert started.wait(2)
        with pytest.raises(ValueError):service.enqueue('import')
        release.set();service.close()
    result=s.rows('SELECT * FROM jobs WHERE id=?',(job['id'],))[0]
    assert result['status']=='failed' and 'offline' in result['error']
    with s.connect() as con:con.execute("UPDATE jobs SET status='running' WHERE id=?",(job['id'],))
    Service(s).close()
    assert s.rows('SELECT status FROM jobs')[0]['status']=='interrupted'


def test_schedule_opt_in_interval_and_no_overlap(root):
    s=Store(root);service=Service(s)
    try:
        from datetime import datetime,timedelta,timezone
        start=datetime(2026,10,2,tzinfo=timezone.utc);s.set_meta('last_sync_attempt',start.isoformat())
        with patch.object(service,'enqueue') as enqueue:
            service.schedule_once(start+timedelta(minutes=500));enqueue.assert_not_called()
            s.set_meta('preferences',dumps({'sync_enabled':True,'sync_minutes':180}))
            service.schedule_once(start+timedelta(minutes=170));enqueue.assert_not_called()
            service.schedule_once(start+timedelta(minutes=185));enqueue.assert_called_once_with('scheduled')
            service.schedule_once(start+timedelta(minutes=186));enqueue.assert_called_once()
    finally:service.close()


def test_strategy_quotes_are_literal():
    content='A claim without evidence.\nWe use patient-level 5 folds.\nInference uses a 3D backbone.'
    hits=strategy_excerpts(content)
    assert hits and all(h['quote'] in content and content.splitlines()[h['line']-1]==h['quote'] for h in hits)


def test_notebook_score_direction_uses_official_metric():
    from monitoring.kneeops.history import notebook_sort
    assert notebook_sort('average of sample weighted log losses')=='scoreAscending'
    assert notebook_sort('average area under the ROC curve')=='scoreDescending'
    assert notebook_sort('a novel custom metric')=='voteCount'
