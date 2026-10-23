import csv
import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from monitoring.kneeops import collectors, predictions
from monitoring.kneeops.api import create_app
from monitoring.kneeops.store import TARGETS, Store


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv('KNEE_WORKSPACE_ROOT', raising=False)
    root = tmp_path / 'project'; (root / 'config').mkdir(parents=True)
    (root / 'config/workspace.toml').write_text('schema_version=1\ncompetition="test-competition"\n', encoding='utf-8')
    predictions._cache.clear()
    return root


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(columns); w.writerows(rows)


def test_rank_auc_and_spearman_match_definitions():
    assert predictions.rankdata(np.array([3., 1., 3., 2.])).tolist() == [3.5, 1, 3.5, 2]
    y = np.array([0, 0, 1, 1, 1]); s = np.array([.1, .4, .35, .8, .9])
    pairs = [(p, n) for p in s[y == 1] for n in s[y == 0]]
    assert predictions.auc(y, s) == pytest.approx(sum(1.0 if p > n else .5 if p == n else 0 for p, n in pairs) / len(pairs))
    assert predictions.auc(np.array([1, 1]), np.array([.2, .3])) is None
    assert predictions.spearman(np.array([1., 2, 3, 4]), np.array([10., 20, 30, 40])) == pytest.approx(1)
    assert predictions.binarize(np.array([0., 1, np.nan]))[1] == 'binary'
    assert predictions.binarize(np.array([.2, .7]))[0].tolist() == [0, 1]
    assert predictions.binarize(np.array([-1., 3]))[1] == 'unusable'


def test_prediction_compare_reports_auc_diversity_blend_and_guards(root):
    rng = np.random.default_rng(0)
    uids = [f'1.2.{i}' for i in range(60)]
    truth = rng.integers(0, 2, (60, 12))
    noise_a, noise_b = rng.normal(0, .8, (60, 12)), rng.normal(0, .8, (60, 12))
    write_csv(root / 'artifacts/predictions/a.csv', [[u, *map(float, truth[i] + noise_a[i])] for i, u in enumerate(uids)], ['StudyInstanceUID', *TARGETS])
    # Different column order, logits outside [0,1], one study missing.
    write_csv(root / 'artifacts/experiments/x/b.csv', [[*map(float, truth[i] + noise_b[i]), u] for i, u in enumerate(uids[:-1])], [*TARGETS, 'StudyInstanceUID'])
    labels = [[u, 'report', *[int(v) for v in truth[i]]] for i, u in enumerate(uids)]
    labels[0][2 + TARGETS.index('Fracture')] = ''
    write_csv(root / 'storage/competition/train.csv', labels, ['StudyInstanceUID', 'Report', *TARGETS])
    write_csv(root / 'artifacts/predictions/not_a_prediction.csv', [['x', 1]], ['id', 'value'])
    s = Store(root)
    found = predictions.discover(s, refresh=True)
    assert [f['id'] for f in found['predictions']] == ['artifacts/experiments/x/b.csv', 'artifacts/predictions/a.csv']
    assert [f['id'] for f in found['labels']] == ['storage/competition/train.csv']
    result = predictions.compare(s, ['artifacts/predictions/a.csv', 'artifacts/experiments/x/b.csv'], 'storage/competition/train.csv', bootstrap=50)
    assert result['common_studies'] == 59 and len(result['correlations']) == 1
    assert any('공통 study' in w for w in result['warnings']) and any('Gold58' in w for w in result['warnings'])
    assert result['files'][0]['outside_probability']
    label = result['label']
    assert label['computable_targets'] == TARGETS
    fracture = next(r for r in label['per_target'] if r['target'] == 'Fracture')
    assert fracture['n_pos'] + fracture['n_neg'] == 58
    assert set(label['macro']) == {'artifacts/predictions/a.csv', 'artifacts/experiments/x/b.csv', 'blend'}
    assert label['macro']['blend'] > 0.5 and label['bootstrap']['blend_vs_best']['reps'] == 50
    with pytest.raises(ValueError):
        predictions.compare(s, ['../outside.csv'])
    with pytest.raises(ValueError):
        predictions.compare(s, ['artifacts/predictions/a.csv'] * 2)


def test_source_meta_sorting_filters_inputs_and_compare_api(root):
    app = create_app(root)
    with TestClient(app) as client:
        s = app.state.store
        s.upsert_source(id='notebook:u/a', kind='notebook', title='A 0.946 blend', content='x = 1\ny = 2', version='3',
                        payload={'source_downloaded': True})
        s.upsert_source(id='notebook:u/b', kind='notebook', title='B fork', content='x = 1\ny = 3', version='1',
                        payload={'source_downloaded': True})
        s.upsert_source(id='notebook:u/c', kind='notebook', title='C stub', content='목록', payload={'listing_only': True})
        s.set_source_meta('notebook:u/a', votes=5, score_claim=.946, inputs={'datasets': ['w/one', 'w/two'], 'models': ['m/dino']})
        s.set_source_meta('notebook:u/b', votes=50, score_claim=0.0, inputs={'datasets': ['w/one']})
        s.set_source_meta('notebook:u/b', votes=None)  # a missing observation keeps the known value
        assert s.source_meta('notebook:u/b')['votes'] == 50
        ids = lambda **q: [i['id'] for i in client.get('/api/sources', params={'kind': 'notebook', **q}).json()['items']]
        assert ids(sort='claim')[0] == 'notebook:u/a' and ids(sort='votes')[0] == 'notebook:u/b'
        assert ids(body='listing') == ['notebook:u/c'] and set(ids(body='full')) == {'notebook:u/a', 'notebook:u/b'}
        assert ids(input='m/dino') == ['notebook:u/a']
        assert client.get('/api/sources', params={'sort': 'drop table'}).status_code == 422
        top = client.get('/api/notebook-inputs').json()
        assert top['items'][0] == {'kind': 'datasets', 'ref': 'w/one', 'n': 2, 'best_score': 0, 'best_claim': .946, 'top_votes': 50}
        cmp = client.get('/api/source/compare', params={'a': 'notebook:u/a', 'b': 'notebook:u/b'}).json()
        assert cmp['inputs']['datasets'] == {'only_a': ['w/two'], 'only_b': [], 'both': ['w/one']}
        assert cmp['inputs']['models']['only_a'] == ['m/dino'] and cmp['added'] == 1 and cmp['removed'] == 1
        assert client.get('/api/source', params={'id': 'notebook:u/a'}).json()['meta']['votes'] == 5


def test_title_claim_ignores_author_and_inputs_drop_blanks():
    assert collectors.title_claim({'title': 'notebookff482d2270', 'ref': 'vidish0987/notebookff482d2270'}) == 0.0
    assert collectors.title_claim({'title': 'RSNA Knee parent0957 routing', 'ref': 'x/rsna-knee-parent0957-routing'}) == .957
    assert collectors.notebook_inputs({'inputs': ['', 'a/b', 'a/b'], 'model_inputs': ['m/x/PyTorch/s/1']}) == {
        'datasets': ['a/b'], 'models': ['m/x/PyTorch/s/1'], 'kernels': [], 'competitions': []}


def test_submission_limits_parse():
    with patch.object(collectors, 'kaggle', return_value='{\n "numTotal": 7,\n "numAllowedNow": 5\n}'):
        limits = collectors.submission_limits('c')
    assert limits['remaining_today'] == 5 and limits['used_total'] == 7


def test_public_scores_from_web_api_are_optional_and_sortable(root):
    from monitoring.kneeops import kaggle_web
    pages = {1: {'kernels': [{'author': {'userName': 'u'}, 'currentUrlSlug': 'a', 'bestPublicScore': 0.946, 'medal': 'SILVER',
                              'lastRunExecutionTimeSeconds': 400, 'totalVotes': 3},
                             {'author': {'userName': 'u'}, 'currentUrlSlug': 'b', 'bestPublicScore': None}]}}
    scores = kaggle_web.competition_notebooks(1, fetch=lambda service, body: pages.get(body['kernelFilterCriteria']['listRequest']['page'], {}))
    assert scores['u/a']['public_score'] == 0.946 and scores['u/b']['public_score'] is None
    s = Store(root)
    s.upsert_source(id='notebook:u/a', kind='notebook', title='A 0.957 claim', content='x')
    s.upsert_source(id='notebook:u/b', kind='notebook', title='B', content='x')
    s.set_source_meta('notebook:u/a', score_claim=.957)
    s.set_source_meta('notebook:u/b', public_score=.95, medal='GOLD')
    s.set_source_meta('notebook:u/a', public_score=.946, medal='SILVER', runtime_s=400, score_observed_at='2026-10-02')
    s.set_source_meta('notebook:u/a', votes=9)  # a later listing keeps the verified score
    rows = s.sources(kind='notebook', sort='score')['items']
    assert [r['id'] for r in rows] == ['notebook:u/b', 'notebook:u/a']
    assert rows[1]['public_score'] == .946 and rows[1]['medal'] == 'SILVER' and rows[1]['score_claim'] == .957
    s.set_meta('competition', '{"id": 1}')
    with patch('monitoring.kneeops.kaggle_web.call', side_effect=RuntimeError('HTTP 500')), \
         patch.object(collectors, 'kaggle', return_value='[]'), patch.object(collectors, 'kaggle_username', return_value='me'), \
         patch.object(collectors, 'leaderboard_summary', return_value={'teams': 1, 'own': None}), \
         patch.object(collectors, 'notebook_catalog', return_value={'items': [], 'total': 0, 'complete': True, 'observed_at': 'x'}), \
         patch.object(collectors, 'topic_listing', return_value=([], True)), patch.object(collectors.time, 'sleep'):
        result = collectors.sync_kaggle(s, topic_budget=0, notebook_budget=0)
    assert [e['connector'] for e in result['errors']] == ['notebook_scores'] and 'notebook_sources' in result['ok']
    assert s.source_meta('notebook:u/a')['public_score'] == .946
