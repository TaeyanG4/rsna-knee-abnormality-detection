"""v1.1.0: CPU member rank contracts; fixtures never qualify a release.

Changelog v1.1.0: run the same arithmetic checks on v9 and v10 templates;
compare unchanged scientific projections through the registered v9 recipe.
"""
import copy
import hashlib
import json
from types import ModuleType

import numpy as np
import pytest

from tools import k1n_stage_v1 as b
from tools import k1n_members_v1 as members
from test_k1n_v7 import rank, records, replay


# Fullbag descendants have their own lineage tests in test_fullbag_post_v1.py.
TEMPLATES = sorted(p for p in (b.ROOT/'configs/k1n').glob('*-v9*.template.json') if '-fb-' not in p.name)
RUNNABLE = TEMPLATES + sorted((b.ROOT/'configs/k1n').glob('*-fb-*.template.json'))


def filled(path=None):
    c = json.loads((path or TEMPLATES[0]).read_bytes())
    c['kernel_id'] = c['kernel_id'].replace('/replace-', '/cpu-fixture-')
    for m in c['models']:
        for key in ('checkpoint_sha256', 'manifest_sha256'):
            if not b.schema.valid_hash(m[key]):
                m[key] = hashlib.sha256((m['id']+key).encode()).hexdigest()
    return b.schema.validate(c)


def runtime(c):
    r = ModuleType('member_test_runtime')
    files = b.package(c)[3]
    # Runtime v10 templates share the same arithmetic contract (v1.8.2).
    exec(compile(files[f"tools/k1sr_runtime_v{c.get('k1_runtime_version', 9)}.py"],
                 '<baked members>', 'exec'), r.__dict__)
    return r


@pytest.mark.parametrize('weights', [[], [1], [1,2], [0,0,0], [-1,1,1],
    [True,1,1], ['1',1,1], [None,1,1], [float('nan'),1,1],
    [float('inf'),1,1], {'a':1}])
def test_invalid_weights(weights):
    c = b.runtime9.registered(True)
    with pytest.raises(ValueError, match='member_weights'):
        b.schema.validate(dict(c, member_weights=weights))


@pytest.mark.parametrize('weights', [[1,2,3], [1e308,1e308,0], [0,0,1],
                                   [1e-300,2e-300,3e-300]])
def test_normalized_idempotent(weights):
    c = b.schema.validate(dict(b.runtime9.registered(True), member_weights=weights))
    assert sum(c['member_weights']) == pytest.approx(1)
    assert b.schema.validate(c) == c
    assert b.util.encoded(b.schema.validate(c)) == b.util.encoded(c)


@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('weight', [1, 1/3, 7.5, 1e308])
def test_existing_equal_artifacts_byte_identical(diagnostic, weight):
    c = b.runtime9.registered(True)
    assert b.artifacts(c, diagnostic) == b.artifacts(dict(c, member_weights=[weight]*3), diagnostic)


@pytest.mark.parametrize('path', TEMPLATES, ids=lambda p:p.name)
def test_templates_refused_and_registered_recipe(path):
    raw = json.loads(path.read_bytes())
    assert {p.name for p in TEMPLATES} == {
        f'{name}.template.json' for name in (
            'all-data-r6s2-av2-v9', 'all-data-r6s2-av2-v9-balanced',
            'folds5-r6s2-av2-v9', 'folds5-r6s2-av2-v9-balanced',
            'folds6-r6s2-av2-v9', 'folds6-r6s2-av2-v9-balanced',
            'b5-flat-r6s2-v9', 'a7-bal-flat-r6s2-v9', 't3-flat-r6s2-v9',
            'c5-dropo-p25-v9', 'c7-bal-dropo-p25-v9', 'c7-bal-dropo-p25s3-v9')}
    with pytest.raises(ValueError):
        b.run('prepare', raw)
    raw['kernel_id'] = raw['kernel_id'].replace('/replace-', '/cpu-fixture-')
    with pytest.raises(ValueError, match='real SHA256'):
        b.run('prepare', raw)
    c = filled(path)
    parent = members.recipe_parent(c)
    if path.name.startswith(('b5-', 'c5-')):
        assert [m['id'] for m in c['models']] == [
            'k1_v2', 'k1_v3g_f1', 'k1_v3g_f2', 'k1_v3g_f3', 'k1_v3g_all']
    else:
        assert c['models'][:3] == parent['models']
    assert {k for k in c.keys() | parent.keys() if c.get(k) != parent.get(k)} <= {
        'models', 'member_weights', 'release_version', 'kernel_id', 'k1_guard_hours', 'projection_basis'}
    if '-flat-' in path.name:
        assert c['alpha'] == .2 and 'alpha_vector' not in c and c['k1_guard_hours'] == 8.8
    if 'member_weights' in c:
        w = c['member_weights']
        np.testing.assert_allclose(sum(w[:3]), w[3])
        assert all(x == w[3] for x in w[3:])
    for diagnostic in (False, True):
        out, reg = b.artifacts(c, diagnostic)
        assert reg['never_submit'] is diagnostic
        assert reg['runtime_projection']['hours'] == (
            (7.459 if c['k1_eval_stride'] == 3 else (8.9072 if len(c['models']) == 7 else 7.5536)) if c['drop_o'] else {4:6.9,5:7.35,6:7.8,7:8.25}[len(c['models'])])
        assert reg['models'] == c['models']
        assert reg.get('member_weights') == c.get('member_weights')
        assert reg['config_sha256'] == b.digest(b.util.encoded(c))
        assert reg.get('alpha_vector') == c.get('alpha_vector')
        assert reg['k1_guard_hours'] == c['k1_guard_hours']
        assert reg['guard_seconds'] == c['k1_guard_hours']*3600
        assert b.runtime9.notebook_config(json.loads(out['kernel/release.ipynb'])) == b.config(c, diagnostic)
        scope = json.loads(out['payload-scope.json'])
        if c['drop_o']:
            assert set(scope['changed']) == {
                'exp169_r7_port.py', 'exp169_release_c85r7.py',
                'exp169_release_c85r7codec.py', 'tools/k1n_config_v1.py'}
            assert 'tools/k1_shared_render_v2.py' in scope['untouched']
        else:
            assert len(scope['untouched']) >= 36
        assert scope['launch_pool_thread_budgets_unchanged']
        meta = json.loads(out['kernel/kernel-metadata.json'])
        assert meta['dataset_sources'][-len(c['models']):] == [m['dataset_slug'] for m in c['models']]
        assert meta['is_private'] and not meta['enable_internet']
        if 'member_weights' not in c:
            assert b.artifacts(dict(c, member_weights=[2]*len(c['models'])), diagnostic)[0] == out


@pytest.mark.parametrize('n,expected,warn', [(4,6.9,False),(5,7.35,False),(6,7.8,False),
                                           (7,8.25,False),(8,8.7,True)])
def test_projection(n, expected, warn):
    p = members.projection_hours(n)
    assert p['hours'] == expected and bool(p['warning']) is warn


def test_projection_refusal():
    with pytest.raises(ValueError, match='exceeds 8.8'):
        members.projection_hours(9)


@pytest.mark.parametrize('field,value', [('drop_o',True),('r6_eval_stride',1),
    ('k1_eval_stride',2),('k1_guard_hours',8.5),('alpha_vector',[.2]*12)])
def test_uncalibrated_recipe_refused(field, value):
    with pytest.raises(ValueError, match='recipe drift'):
        b.package(dict(filled(), **{field:value}))


@pytest.mark.parametrize('path', RUNNABLE, ids=lambda p:p.name)
@pytest.mark.parametrize('diagnostic', [False, True])
@pytest.mark.parametrize('fault', ['healthy', 'fallback', 'late', 'zero'])
def test_actual_baked_finish_math(tmp_path, monkeypatch, path, diagnostic, fault):
    c = filled(path)
    if fault == 'zero':
        c['member_weights'] = [0.]*(len(c['models'])-1)+[1.]
    r = runtime(c)
    # All K1 rows complete on time, even when notebook finishes after guard.
    monkeypatch.setenv('K1SR_STARTED', str(r.time.time()-r.GUARD_SECONDS-1))
    rng = np.random.default_rng(918)
    values = [rng.integers(0,9,(40,12)).astype(np.float32)/8 for _ in range(len(c['models'])+1)]
    parent, arrays = values[0], values[1:]
    payload = replay.csv_bytes(list(map(str,range(40))), parent)
    (tmp_path/('diagnostic-c85r7.csv' if diagnostic else 'submission.csv')).write_bytes(payload)
    rows = records(r, arrays)
    failures = 1 if fault == 'fallback' else 0
    if failures:
        rows[0].update(ok=False)
        rows[0].pop('predictions')
    if fault == 'late':
        rows[0]['finished_elapsed_seconds'] = r.GUARD_SECONDS+1
    worker = tmp_path/'run/pr7/pr7-run/s_r7/controller/worker0'
    worker.mkdir(parents=True)
    (worker/'k1-studies.jsonl').write_text('\n'.join(map(json.dumps,reversed(rows))))
    if fault == 'late':
        with pytest.raises(RuntimeError, match='fail-loud'):
            r.finish(tmp_path,tmp_path/'run',b.config(c,diagnostic),10.)
    else:
        r.finish(tmp_path,tmp_path/'run',b.config(c,diagnostic),10.)
    rec = json.loads((tmp_path/'k1sr-receipt.json').read_bytes())
    output = (tmp_path/('diagnostic-k1n.csv' if diagnostic else 'submission.csv')).read_bytes()
    assert rec.get('member_weights') == c.get('member_weights')
    assert rec['config_sha256'] == r.CONFIG_SHA
    # Flat P changes blend weights, not the public recipe's runtime calibration.
    projection_config = {k:v for k,v in c.items() if k != 'p_weight_mode'}
    # v10 inherits the exact v9 scientific recipe and calibration.
    if projection_config.get('k1_runtime_version') == 10:
        projection_config['k1_runtime_version'] = 9
    assert rec['runtime_projection'] == members.projection(projection_config)
    assert rec['fail_loud_triggered'] is (fault == 'late')
    if fault == 'late':
        assert output == payload
    else:
        for a in arrays:
            a[:failures] = parent[:failures]
        ranked = [rank(a) for a in arrays]
        # Independent scipy ranks and per-member Python summation.
        mean = (sum(w*a for w,a in zip(c['member_weights'],ranked))
                if 'member_weights' in c else np.mean(ranked,axis=0))
        kr, pr = rank(mean), rank(parent)
        kr[:failures] = pr[:failures]
        alpha = np.asarray(c.get('alpha_vector', c['alpha']))
        blend = (1-alpha)*pr+alpha*kr
        blend[:failures] = pr[:failures]
        np.testing.assert_array_equal(r.util.table(output)[1],rank(blend).astype(np.float32))
    if diagnostic:
        assert rec['never_submit'] and rec['submission_sha256'] is None
        assert not (tmp_path/'submission.csv').exists()


def test_zero_weight_member_is_still_required():
    c = filled()
    c['member_weights'] = [0.]*(len(c['models'])-1)+[1.]
    r = runtime(c)
    arrays = [np.full((3,12),.3) for _ in c['models']]
    rows = records(r,arrays)
    del rows[0]['predictions'][r.MODEL_IDS[0]]
    with pytest.raises(ValueError, match='all registered'):
        r.blend(replay.csv_bytes(['0','1','2'],np.full((3,12),.5)),rows)


@pytest.mark.parametrize('name', ['b5-flat-r6s2-v9', 'a7-bal-flat-r6s2-v9', 't3-flat-r6s2-v9'])
def test_flat_prepare_checks_recipe_before_sha(name, capsys, monkeypatch):
    raw = json.loads((b.ROOT/f'configs/k1n/{name}.template.json').read_bytes())
    def no_stage(*args, **kwargs):
        pytest.fail('placeholder must fail before stage generation')
    monkeypatch.setattr(b, 'artifacts', no_stage)
    with pytest.raises(ValueError, match='real SHA256 required'):
        b.run('prepare', raw)
    projection = json.loads(capsys.readouterr().out)['runtime_projection']
    assert projection['hours'] == (8.25 if name.startswith('a7') else 7.35)
    assert projection['measured_for_this_candidate'] is False
    raw['k1_eval_stride'] = 2
    with pytest.raises(ValueError, match='recipe drift'):
        b.run('prepare', raw)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('n', [7, 8])
def test_expanded_model_schema_baked_and_guard(vector, weighted, n):
    c = filled(next(p for p in TEMPLATES if p.name.startswith('a7-')))
    if n == 8:
        m = dict(c['models'][-1], id='k1_cpu_eighth',
                 dataset_slug='taeyangg4/knee-cpu-eighth-assets',
                 checkpoint_sha256=hashlib.sha256(b'cpu-eighth').hexdigest())
        c['models'].append(m)
    c.pop('member_weights', None)
    if weighted:
        c['member_weights'] = [1]*(n-1)+[2]
    if vector:
        c['alpha_vector'] = b.runtime9.registered(True)['alpha_vector']
    c = b.schema.validate(c)
    files = b.package(c)[3]
    validator = ModuleType('expanded_schema')
    exec(compile(files['tools/k1n_config_v1.py'], '<baked schema>', 'exec'), validator.__dict__)
    assert validator.validate(c) == c
    assert runtime(c).GUARD_SECONDS == 8.8*3600
    assert b.config(c).get('alpha_vector') == c.get('alpha_vector')
    assert members.projection(c)['hours'] == (8.7 if n == 8 else 8.25)
    c['models'].extend([dict(c['models'][-1], id='k1_cpu_ninth')]*(9-n))
    with pytest.raises(ValueError, match='register 1..8 models'):
        b.schema.validate(c)


@pytest.mark.parametrize('path', [p for p in TEMPLATES if '-av2-' in p.name], ids=lambda p:p.name)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_v170_member_artifacts_byte_identical(path, diagnostic):
    from tools import k1n_replay_v170
    c = filled(path)
    assert b.artifacts(c, diagnostic) == k1n_replay_v170.load().artifacts(c, diagnostic)


@pytest.mark.parametrize('path', [p for p in TEMPLATES if not p.name.startswith(('c5-', 'c7-'))], ids=lambda p:p.name)
@pytest.mark.parametrize('diagnostic', [False, True])
def test_v171_member_artifacts_byte_identical(path, diagnostic):
    from tools import k1n_replay_v171
    c = filled(path)
    assert b.artifacts(c, diagnostic) == k1n_replay_v171.load().artifacts(c, diagnostic)


@pytest.mark.parametrize('name,hours', [('c5-dropo-p25-v9', 7.5536),
                                       ('c7-bal-dropo-p25-v9', 8.9072)])
def test_p25_placeholder_projection_before_sha(name, hours, capsys, monkeypatch):
    raw = json.loads((b.ROOT/f'configs/k1n/{name}.template.json').read_bytes())
    monkeypatch.setattr(b, 'artifacts', lambda *a: pytest.fail('must not generate placeholders'))
    with pytest.raises(ValueError, match='real SHA256 required'):
        b.run('prepare', raw)
    p = json.loads(capsys.readouterr().out)['runtime_projection']
    assert p['hours'] == hours
    assert p['anchor_hours'] == 6.2 and p['anchor_models'] == 3
    assert p['incremental_model_hours'] == .47 and p['stride2_multiplier'] == 1.44
    assert p['anchor_observed_range_hours'] == [6.12, 6.20] and p['anchor_includes_queue']
    assert bool(p['warning']) is (hours > 8.3)
    assert p['measured_for_this_candidate'] is False


@pytest.mark.parametrize('field,value', [
    ('alpha', .2), ('alpha', .3), ('alpha_vector', [.25]*12),
    ('drop_o', False), ('k1_eval_stride', 4), ('r6_eval_stride', 1),
    ('r6_eval_stride', 2), ('k1_guard_hours', 8.6), ('k1_max_failed_fraction', .1),
    ('k1_failure_policy', 'fail_open'), ('k1_runtime_version', 8),
    ('encoder_chunk', 64), ('unknown', True)])
def test_p25_accepts_only_exact_recipe(field, value, capsys):
    raw = json.loads(next(p for p in TEMPLATES if p.name.startswith('c5-')).read_bytes())
    raw[field] = value
    with pytest.raises(ValueError) as exc:
        b.run('prepare', raw)
    assert 'real SHA256' not in str(exc.value)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('n,hours', [(3,6.2), (5,7.5536), (6,8.2304), (7,8.9072), (8,9.584)])
def test_p25_projection_and_cap(n, hours):
    c = members.p25_parent()
    c['models'] = [c['models'][0]]*n
    assert members.projection(c, enforce=False)['hours'] == hours
    if hours > 8.8:
        with pytest.raises(ValueError, match='exceeds 8.8'):
            members.projection(c)
    else:
        assert members.projection(c)['hours'] == hours


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('diagnostic', [False, True])
def test_p25_exact_parent_payload_scope(weighted, diagnostic):
    c = filled(next(p for p in TEMPLATES if p.name.startswith('c5-')))
    if weighted:
        c['member_weights'] = [.1, .1, .2, .2, .4]
    nb, meta, old, new, old_nb, old_meta = b.package(c, diagnostic)
    base = members.p25_legacy()
    engine, _ = b.baseline(base)
    parent_outputs, _ = b.frozen_parent(engine, base, diagnostic)
    assert b.util.encoded(old_nb) == parent_outputs['kernel/release.ipynb']
    assert 'r6_eval_stride' not in c and 'alpha_vector' not in c
    assert new['tools/k1_shared_render_v2.py'] == old['tools/k1_shared_render_v2.py']
    port = 'exp169_r7_port.py'
    assert new[port] == old[port].replace(b'k1sr_runtime_v7', b'k1sr_runtime_v9')
    assert new.keys()-old.keys() == {'tools/k1sr_runtime_v9.py'}
    assert old.keys()-new.keys() == {'tools/k1sr_runtime_v7.py'}
    cfg = b.config(c, diagnostic)
    assert cfg['formula'].startswith('rank(0.75*rank(INF3_P_R224_float32_csv)+0.25*rank(')
    assert cfg['drop_o'] and cfg['k1_eval_stride'] == 2 and 'r6_eval_stride' not in cfg
    assert b.runtime9.notebook_config(nb) == cfg
    r = runtime(c)
    assert r.REGISTERED == members.canonical(c) and r.K1_WEIGHT == .25
    assert r.VERSION == ('9.1.0' if weighted else '9.0.0')
    prior_slugs = {m['dataset_slug'] for m in base['models']}
    assert meta['dataset_sources'][:-len(c['models'])] == [
        s for s in old_meta['dataset_sources'] if s not in prior_slugs]
    scope = json.loads(b.artifacts(c, diagnostic)[0]['payload-scope.json'])
    assert scope['drop_o'] and scope['k1_stride'] == 2
    assert scope['parent_stage'].endswith('v1.1.67-r6w958k1n3s2p25'+('-n32' if diagnostic else ''))
