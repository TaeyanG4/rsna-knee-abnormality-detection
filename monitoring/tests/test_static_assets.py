from fastapi.testclient import TestClient

from monitoring.kneeops.api import create_app
from monitoring.tests.test_core import root


def test_entry_revalidates_and_versioned_assets_remain_cacheable(root, tmp_path, monkeypatch):
    static = tmp_path / 'static'
    assets = static / 'assets'; assets.mkdir(parents=True)
    entry = static / 'index.html'; entry.write_text('<h1>current</h1>')
    (assets / 'ViewerPage-fixture.js').write_text('export default {};')
    monkeypatch.setenv('KNEEOPS_STATIC_DIR', str(static))
    with TestClient(create_app(root)) as client:
        page = client.get('/')
        assert page.headers['cache-control'] == 'no-store'
        assert client.get('/?raw=example').headers['cache-control'] == 'no-store'
        entry.write_text('<h1>new version</h1>')
        assert 'new version' in client.get('/').text
        old_asset = client.get('/assets/ViewerPage-fixture.js')
        assert old_asset.status_code == 200
        assert 'immutable' in old_asset.headers['cache-control']
        missing = client.get('/assets/ViewerPage-missing.js')
        assert missing.status_code == 404 and 'immutable' not in missing.headers.get('cache-control', '')
        assert client.get('/api/health').headers['cache-control'] == 'no-store'
