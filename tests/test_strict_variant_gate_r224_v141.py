"""v1.4.1 CPU binding tests; fixture reviews do not qualify release arithmetic."""
import pytest

import test_strict_variant_gate_r224 as legacy
from tools import strict_variant_gate as g
from tools import strict_variant_gate_r224 as old
from tools import strict_variant_gate_r224_v141 as r
from test_strict_variant_gate import dispatched, put


def reseal(c, **kwargs):
    legacy.reseal(c, **kwargs)
    path = c['source'] / 'new-members.json'
    side = g.parse(path.read_bytes())
    for row in side['added']:
        if row['name'] == r.PROJECT:
            row['origin'] = {'kind': r.PROJECT_KIND}
    put(path, side)


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, 'r', r)
    c = legacy.candidate.__wrapped__(tmp_path, monkeypatch)
    c['files'][r.RAPTOR] = c['files'].pop('r224-stage.py')
    c['files'][r.WRAPPER] = c['files'][r.WRAPPER].replace(b'r224-stage.py', r.RAPTOR.encode())
    # Deliberately unqualified fixture content: independent content review, not
    # ast.parse or a self-supplied hash, is the semantic trust boundary.
    c['files'][r.PROJECT] = b'"""Unqualified binding test fixture."""\nVALUE = 1\n'
    reseal(c)
    return c


def endorse(c, content='exact'):
    allow = g.parse((c['output'] / 'allowlist.json').read_bytes())
    pin = g.parse(c['pin'].read_bytes())
    text = ('Reviewer-Lane: r224-reviewer\nSHA256SUMS: ' + allow['stage_sha256sums'] +
            '\nAllowlist-SHA256: ' + g.sha((c['output'] / 'allowlist.json').read_bytes()) +
            '\nR224-PIN-SHA256: ' + allow['pin_sha256'] + '\nR224-Scope: ' + r.SCOPE + '\n')
    line = r.project_review_line(pin)
    if content == 'exact': text += line + '\n'
    if content == 'wrong-hash': text += line.replace(pin['added'][r.PROJECT], '0' * 64) + '\n'
    if content == 'duplicate': text += (line + '\n') * 2
    dispatched(c['tmp'], 'r224-reviewer', c['review'], (text + 'VERDICT: PASS\n').encode())


def bind(c):
    r.pin(c['source'], c['builder'], c['pin'])
    r.bind(c['pin'], c['output'], c['review'])
    endorse(c)


def test_exact_two_members_pin_bind_evaluate(candidate):
    c = candidate
    assert r.main(['pin', '--stage', str(c['source']), '--builder-receipt', str(c['builder']),
                   '--output', str(c['pin'])]) == 0
    saved = g.parse(c['pin'].read_bytes())
    assert saved['added'] == {n: g.sha(c['files'][n]) for n in (r.RAPTOR, r.PROJECT)}
    assert saved['project_content_review_required'] == {r.PROJECT: g.sha(c['files'][r.PROJECT])}
    assert r.main(['bind', '--pin', str(c['pin']), '--output', str(c['output']),
                   '--review', str(c['review'])]) == 0
    request = (c['output'] / 'review-request.md').read_text()
    assert r.project_review_line(saved) in request
    assert r.evaluate(c['output'], c['review'])['status'] == 'FAIL'
    endorse(c)
    result = r.evaluate(c['output'], c['review'])
    assert result['status'] == 'PASS', result['errors']
    assert result['full_diff'].count(': IDENTICAL ') >= 28
    args = ['evaluate', '--stage', str(c['output']), '--review', str(c['review'])]
    assert r.main(args + ['--check-only']) == 0
    assert r.main(args) == 2  # Fixture evidence cannot publish admission.


@pytest.mark.parametrize('mutation', ['extra-project', 'missing-project', 'renamed-project',
    'raptor-byte', 'wrong-slug', 'placeholder', 'launch', 'origin-kind', 'sha',
    'project-byte-stale-sha', 'empty-project', 'reason'])
def test_refusal_boundaries(candidate, mutation):
    c = candidate
    kwargs = {}
    stale_hash = g.sha(c['files'][r.PROJECT])
    if mutation == 'extra-project': c['files']['extra_project.py'] = b'VALUE = 1\n'
    if mutation == 'missing-project': del c['files'][r.PROJECT]
    if mutation == 'renamed-project': c['files']['renamed.py'] = c['files'].pop(r.PROJECT)
    if mutation == 'raptor-byte': c['files'][r.RAPTOR] += b'\n'
    if mutation == 'wrong-slug': kwargs['slug'] = 'taeyangg4/wrong-slug'
    if mutation == 'placeholder': kwargs['placeholder'] = True
    if mutation == 'launch':
        assert b'label="C85R7 R"' in c['files'][r.WRAPPER]
        c['files'][r.WRAPPER] = c['files'][r.WRAPPER].replace(b'label="C85R7 R"', b'label="changed R"')
    if mutation == 'project-byte-stale-sha': c['files'][r.PROJECT] += b'\n'
    if mutation == 'empty-project': c['files'][r.PROJECT] = b'\n'
    reseal(c, **kwargs)
    side_path = c['source'] / 'new-members.json'
    side = g.parse(side_path.read_bytes())
    project = next((x for x in side['added'] if x['name'] == r.PROJECT), None)
    if mutation == 'origin-kind': project['origin']['kind'] = 'nartaa original'
    if mutation == 'sha': project['sha256'] = '0' * 64
    if mutation == 'project-byte-stale-sha': project['sha256'] = stale_hash
    if mutation == 'reason':
        for row in side['changed']: row['registered_reason'] = row.pop('reason')
    put(side_path, side)
    with pytest.raises(ValueError): r.pin(c['source'], c['builder'], c['pin'])
    assert not c['pin'].exists()


@pytest.mark.parametrize('content', ['missing', 'wrong-hash', 'duplicate'])
def test_independent_project_review_binding(candidate, content):
    c = candidate
    bind(c)
    endorse(c, content=content)
    result = r.evaluate(c['output'], c['review'])
    assert result['status'] == 'FAIL'
    assert 'independent project content review missing or stale' in result['errors']


def test_project_drift_after_pin_even_with_new_sidecar_hash(candidate):
    c = candidate
    bind(c)
    c['files'][r.PROJECT] += b'\n'
    reseal(c)
    result = r.evaluate(c['output'], c['review'])
    assert result['status'] == 'FAIL'
    assert 'R224 PIN differs from reconstructed completed stage' in result['errors']


def test_v140_still_refuses_two_added_members(candidate):
    c = candidate
    with pytest.raises(ValueError, match='reconstruct original'):
        old.added_source({n: c['files'][n] for n in (r.RAPTOR, r.PROJECT)}, g.Inputs(g.ROOT))


def test_v140_and_pre_v142_consumer_snapshots_unchanged():
    assert g.sha((g.ROOT / 'tools/strict_variant_gate_r224.py').read_bytes()) == \
        '57e632183b4c01b97ae16b6c2a6b7daa4d090774a9c2ff718d2b958e93cad62b'
    for live, backup in [('tools/kaggle_submit_once.py', 'kaggle_submit_once.py'),
                         ('.local/run-scripts/b-preflight-v1.py', 'b-preflight-v1.py')]:
        # v142 explicitly authorizes consumer changes; preserve the prior bytes in backup.
        assert (g.ROOT / '.local/backups/strict-variant-gate-v1.4.2' / backup).read_bytes() == \
            (g.ROOT / '.local/backups/strict-variant-gate-v1.4.1' / backup).read_bytes()
