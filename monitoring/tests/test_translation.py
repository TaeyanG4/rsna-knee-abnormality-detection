import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from monitoring.kneeops.api import create_app
from monitoring.kneeops.store import Store
from monitoring.kneeops.translation import ROOT, catalog, translation_for


def test_translation_catalog_has_verified_originals_and_complete_rules():
    items = catalog()
    assert len(items) == 10
    for item in items:
        original = (ROOT / 'en' / (item['slug'] + '.md')).read_text(encoding='utf-8')
        assert hashlib.sha256(original.encode()).hexdigest() == item['source_sha']
        assert item['scope'] == 'full_text' and item['content'].strip()
    rules = next(r['content'] for r in items if r['page'] == 'rules')
    # Guard against accidentally replacing the legal translation with a summary.
    for section in ['### 1.', '### 2.', '### 3.', '#### 18.', 'CC-BY-NC 4.0', 'MIRA', '최대 5개', '최대 2개', 'Private Leaderboard']:
        assert section in rules


def test_old_translation_never_attaches_to_changed_rules(tmp_path, monkeypatch):
    monkeypatch.delenv('KNEE_WORKSPACE_ROOT', raising=False)
    root = tmp_path / 'workspace'; (root / 'config').mkdir(parents=True)
    (root / 'config/workspace.toml').write_text('schema_version=1\ncompetition="rsna-knee-abnormality-detection"\n[paths]\n', encoding='utf-8')
    store = Store(root)
    original = (ROOT / 'en/rules.md').read_text(encoding='utf-8')
    old = store.upsert_source(id='competition:rules', kind='official', title='rules', content=original, observed_at='2026-10-01T00:00:00Z')
    added = store.source('competition:rules')['added_at']
    current = store.upsert_source(id='competition:rules', kind='official', title='rules', content=original + '\nUpdated rule', observed_at='2026-10-02T00:00:00Z')
    result = translation_for(store.source('competition:rules'))
    assert result['status'] == 'outdated' and 'content' not in result
    assert translation_for(store.source('competition:rules', old))['status'] == 'available'
    assert store.source('competition:rules')['current_revision'] == current
    assert store.source('competition:rules')['added_at'] == added
    app = create_app(root, store.state)
    with TestClient(app) as client:
        latest = client.get('/api/source', params={'id': 'competition:rules'}).json()
        older = client.get('/api/source', params={'id': 'competition:rules', 'revision': old}).json()
        assert latest['translation']['status'] == 'outdated'
        assert older['translation']['source_sha'] == older['revision']['sha']
        assert older['revision']['content'] == original


def test_other_competition_and_nonofficial_sources_do_not_inherit_translation():
    rule = next(r for r in catalog() if r['page'] == 'rules')
    source = {'kind': 'official', 'competition': 'different-competition', 'title': 'rules', 'revision': {'sha': rule['source_sha']}}
    assert translation_for(source)['status'] == 'unavailable'
    source['competition'] = rule['competition']; source['kind'] = 'notebook'
    assert translation_for(source) is None
