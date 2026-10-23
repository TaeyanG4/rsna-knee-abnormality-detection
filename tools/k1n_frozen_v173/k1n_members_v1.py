"""v1.0.3: exact S1/S2/P25/P25S3 recipes and calibrated CPU projections.

Changelog v1.0.3: opt-in N32 ratio, P25S3; pinned v1.7.2 replay for existing configs.
Changelog v1.0.2: P25 stride2/drop-O recipe; pinned v1.7.1 member replay.
Changelog v1.0.1: flat S1 recipes, guard8.8 and eight models; preserve v1.7.0 bytes.
Changelog: new identities only; frozen v9.0.0 remains untouched. Bake v9.1.0
only for unequal member weights; equal uses the original np.mean arithmetic.
"""
import base64
import copy
import difflib
import json

from tools import k1n_config_v1 as schema
from tools import k1n_replay_v160 as replay160
from tools import k1n_replay_v171 as replay171
from tools import k1n_replay_v172 as replay172
from tools import c85r7_r6w958k1x3sr_v1164 as before

ROOT, util, exact = replay160.ROOT, replay160.util, replay160.exact
digest = before.digest
VERSION = '1.0.3'
P25_SHA256 = '8e12268f985ad3f1f6bc67640006737f9fc91083119f93edd926a60284e25c6d'


def selected(c):
    return c.get('k1_runtime_version') == 9 and (
        c.get('release_version'), c.get('kernel_id')) not in (
            ('1.1.73-r6s2k1x3fl', 'taeyangg4/knee-lb-r6s2k1x3fl-v1'),
            ('1.1.73-r6s2k1x3avfl', 'taeyangg4/knee-lb-r6s2k1x3avfl-v1'))


def projection_hours(n_models):
    util.need(type(n_models) is int and n_models > 0, 'projection model count')
    hours = round(6.6 - 1.5 + .45 * n_models, 10)
    util.need(hours <= 8.8, 'runtime projection exceeds 8.8 h; prepare refused')
    return dict(schema='k1n-runtime-projection/v1', parent_with_o_hours=6.6,
        r6_stride2_savings_hours=1.5, k1_stride3_hours_per_model=.45,
        n_models=n_models, hours=hours, warn_above_hours=8.3, refuse_above_hours=8.8,
        warning='PROJECTED_RUNTIME_ABOVE_8.3H' if hours > 8.3 else None,
        source='Taeyang 2026-10-20 hidden calibration; R6 stride2 N32 -35%',
        measured_for_this_candidate=False, zero_weight_members_still_execute=True)


def projection(c, enforce=True):
    if c['drop_o']:
        recipe_parent(c)
        n = len(c['models'])
        hours = round(6.2 + .47 * 1.44 * (n - 3), 10)
        result = dict(schema='k1n-runtime-projection/v2', anchor_hours=6.2,
            anchor_observed_range_hours=[6.12, 6.20], anchor_models=3,
            incremental_model_hours=.47, stride2_multiplier=1.44,
            formula='6.2 + 0.47 * 1.44 * (n_models - 3)',
            anchor_includes_queue=True, n_models=n, hours=hours,
            warn_above_hours=8.3, refuse_above_hours=8.8,
            warning='PROJECTED_RUNTIME_ABOVE_8.3H' if hours > 8.3 else None,
            source='Taeyang 2026-10-21 hidden P25 v1.1.67 calibration',
            measured_for_this_candidate=False, zero_weight_members_still_execute=True,
            drop_o=True, r6_eval_stride=None, k1_eval_stride=2)
        if c['k1_eval_stride'] == 3:
            hours = round(6.2 - 3*.677 + n*.47, 10)
            result.update(schema='k1n-runtime-projection/v3', hours=hours,
                anchor_stride2_model_hours=.677, k1_eval_stride=3,
                formula='6.2 - 3 * 0.677 + n_models * 0.47',
                warning='PROJECTED_RUNTIME_ABOVE_8.3H' if hours > 8.3 else None)
        if c.get('projection_basis') == 'n32_ratio':
            result.update(projection_basis='n32_ratio', qualification_required=True,
                qualification_limit_hours=8.6, anchor_n32_parent_seconds=1366,
                anchor_n32_k1_seconds=419, anchor_n32_total_seconds=1785,
                qualification_formula='6.2 * (parent_seconds + k1_seconds) / 1785')
        if enforce and c.get('projection_basis') != 'n32_ratio':
            util.need(hours <= 8.8, 'runtime projection exceeds 8.8 h; prepare refused')
        return result
    util.need(not c['drop_o'] and c.get('r6_eval_stride') == 2 and
              c['k1_eval_stride'] == 3, 'projection requires O, R6 stride2, K1 stride3')
    return dict(projection_hours(len(c['models'])), drop_o=False, r6_eval_stride=2,
                k1_eval_stride=3)


def recipe_parent(value):
    """Check the complete recipe before inspecting placeholder model identities."""
    base = (p25_parent() if value.get('drop_o') else
            replay160.load().runtime9.registered('alpha_vector' in value))
    if value.get('drop_o') and value.get('k1_eval_stride') == 3:
        base['k1_eval_stride'] = 3
    allowed = {'release_version', 'kernel_id', 'models', 'member_weights'}
    if 'projection_basis' in value:
        base['projection_basis'] = value['projection_basis']
        schema.validate(base)
    probe = {k:v for k,v in value.items() if k not in allowed}
    probe.update({k:base[k] for k in ('release_version', 'kernel_id', 'models')})
    probe = schema.validate(probe)
    expected = dict(base)
    if probe.get('k1_guard_hours') == 8.8:
        expected['k1_guard_hours'] = 8.8
    util.need({k:v for k,v in probe.items() if k not in allowed} ==
              {k:v for k,v in expected.items() if k not in allowed}, 'member S1/S2/P25/P25S3 recipe drift')
    return base


def p25_legacy():
    """Exact hidden-proven P25 parent; source edits must fail closed."""
    path = ROOT/'configs/k1n/x3-stride2-drop-o.json'
    util.need(util.sha(path) == P25_SHA256, 'P25 registered parent drift')
    return schema.validate(json.loads(path.read_bytes()))


def p25_parent():
    return schema.validate(dict(p25_legacy(), k1_runtime_version=9,
        k1_failure_policy='study_fallback_fail_loud', k1_guard_hours=8.8))


def historical(c):
    return ('alpha_vector' in c and c['k1_guard_hours'] == 8.6 and len(c['models']) <= 6)


def canonical(value, report_projection=False):
    base = recipe_parent(value)
    models = value.get('models')
    util.need(isinstance(models, list) and 1 <= len(models) <= 8, 'register 1..8 models')
    if report_projection and not historical(value):
        print(json.dumps(dict(runtime_projection=projection(dict(base, models=models), enforce=False))))
    c = schema.validate(value)
    util.need(selected(c), 'member ensembles require a new release and slug')
    parents = [replay160.load().runtime9.registered(v) for v in (False, True)] + [p25_parent()]
    util.need(all(c['release_version'] != p['release_version'] and
                 c['kernel_id'] != p['kernel_id'] for p in parents),
              'member ensembles require new identifiers')
    allowed = {'release_version', 'kernel_id', 'models', 'member_weights', 'k1_guard_hours'}
    util.need({k:v for k,v in c.items() if k not in allowed} ==
              {k:v for k,v in base.items() if k not in allowed}, 'member S1/S2/P25 recipe drift')
    # Stable insertion order makes field-order changes irrelevant.
    if new_path(c):
        c['models'] = [{k:m[k] for k in sorted(m)} for m in c['models']]
    base.update({k:c[k] for k in ('release_version', 'kernel_id', 'models')})
    base['k1_guard_hours'] = c['k1_guard_hours']
    if 'member_weights' in c:
        base['member_weights'] = c['member_weights']
    projection(base)
    return base


def config(value, diagnostic=False):
    c = canonical(value)
    if not new_path(c):
        return replay172.load().config(c, diagnostic)
    from tools import k1n_stage_v1 as builder
    parent = p25_legacy()
    engine, _ = builder.baseline(parent)
    cfg = engine.config(parent, diagnostic)
    cfg['k1_eval_stride'] = c['k1_eval_stride']
    cfg['contract']['eval_stride'] = c['k1_eval_stride']
    if 'projection_basis' in c:
        cfg['projection_basis'] = c['projection_basis']
    cfg.update({k:c[k] for k in ('k1_runtime_version', 'k1_failure_policy',
                                 'k1_max_failed_fraction', 'k1_guard_hours')})
    cfg['formula'] += ('; failed-study K1 rank input = parent rank before outer rank; '
                       'systemic failure or failed fraction above registered limit raises after publication')
    cfg.update(release_version=c['release_version']+('-n32' if diagnostic else ''),
        kernel_id=c['kernel_id']+('-n32' if diagnostic else ''),
        models=c['models'], k1_models=[m['id'] for m in c['models']],
        model_sha256={m['id']:m['checkpoint_sha256'] for m in c['models']},
        asset_manifest_sha256={m['id']:m['manifest_sha256'] for m in c['models']},
        config_sha256=digest(util.encoded(c)), registered_config_sha256=digest(util.encoded(c)),
        runtime_projection=projection(c), k1_guard_hours=c['k1_guard_hours'])
    if 'member_weights' in c:
        cfg['member_weights'] = c['member_weights']
        cfg['formula'] = cfg['formula'].replace('mean(per-target rank(K1_model_FP16))',
            'sum_i(member_weights[i]*per-target rank(K1_i_FP16))')
    return cfg


def new_path(c):
    return c.get('projection_basis') == 'n32_ratio' or (
        c['drop_o'] and c['k1_eval_stride'] == 3)


def runtime_source(source, old_config, c):
    text = exact(source.decode(), 'REGISTERED = '+repr(old_config), 'REGISTERED = '+repr(c))
    if 'member_weights' in c:
        text = exact(text, 'runtime v9.0.0.', 'runtime v9.1.0.\n\n'
            'Changelog v9.1.0: registered normalized member weights in the inner rank blend.')
        text = exact(text, "VERSION = '9.0.0'", "VERSION = '9.1.0'")
        text = exact(text, '    k1_rank_mean = np.mean(np.stack(ranks), axis=0)',
            "    weights = REGISTERED.get('member_weights')\n"
            "    k1_rank_mean = (np.mean(np.stack(ranks), axis=0) if weights is None else\n"
            "                    np.sum(np.stack(ranks) * np.asarray(weights, dtype=np.float64)[:, None, None], axis=0))")
    return text.encode()


def package(value, diagnostic=False):
    c = canonical(value)
    if not new_path(c):
        return replay172.load().package(c, diagnostic)
    from tools import k1n_stage_v1 as builder
    base = p25_legacy()
    engine, _ = builder.baseline(base)
    builder.frozen_parent(engine, base, diagnostic)
    old_nb, old_meta, _, old, _, _ = engine.package(base, diagnostic)
    nb, meta, new = copy.deepcopy(old_nb), copy.deepcopy(old_meta), dict(old)
    oc, cfg = engine.config(base, diagnostic), config(c, diagnostic)
    runtime, codec = 'tools/k1sr_runtime_v9.py', 'exp169_release_c85r7codec.py'
    source = (ROOT/runtime).read_text(encoding='utf-8')
    source = exact(source,
        "REGISTERED = dict(config_schema.current(), k1_failure_policy='study_fallback_fail_loud')",
        'REGISTERED = '+repr(base))
    new[runtime] = runtime_source(source.encode(), base, c)
    del new['tools/k1sr_runtime_v7.py']
    new['tools/k1n_config_v1.py'] = (ROOT/'tools/k1n_config_v1.py').read_bytes()
    port, release = 'exp169_r7_port.py', 'exp169_release_c85r7.py'
    new[port] = exact(old[port].decode(), 'k1sr_runtime_v7', 'k1sr_runtime_v9').encode()
    new[release] = exact(old[release].decode(), digest(old[port]), digest(new[port])).encode()
    text = exact(old[codec].decode(), digest(old[release]), digest(new[release]))
    util.need(text.count(oc['release_version']) == 2, 'codec identity anchors')
    text = text.replace(oc['release_version'], cfg['release_version'])
    new[codec] = exact(text, oc['kernel_id'], cfg['kernel_id']).encode()
    code = ''.join(old_nb['cells'][2]['source'])
    blob, pins, _ = before.parent.archive(code)
    packed = before.packed(new)
    code = exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(packed).decode()))
    code = exact(code, 'PINS = '+repr(pins), 'PINS = '+repr({n:digest(v) for n,v in new.items()}))
    code = exact(code, repr(digest(blob)), repr(digest(packed)))
    code = exact(code, 'from tools import k1sr_runtime_v7 as _k1sr',
                 'from tools import k1sr_runtime_v9 as _k1sr')
    code = exact(code, repr(oc), repr(cfg))
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = ['# K1 member ensemble '+cfg['release_version']+'\n',
        cfg['formula']+'\n', 'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else
        'CPU prepared; independent review and actual inference qualification pending.\n']
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1])
    prior_slugs = {m['dataset_slug'] for m in base['models']}
    meta['dataset_sources'] = [s for s in old_meta['dataset_sources'] if s not in prior_slugs]
    meta['dataset_sources'] += [m['dataset_slug'] for m in c['models']]
    util.need(len(meta['dataset_sources']) == len(set(meta['dataset_sources'])), 'duplicate mounts')
    allowed = {port, release, codec, 'tools/k1n_config_v1.py'}
    util.need({n for n in old.keys() & new.keys() if old[n] != new[n]} == allowed,
              'member payload scope drift')
    util.need(new.keys()-old.keys() == {runtime} and
              old.keys()-new.keys() == {'tools/k1sr_runtime_v7.py'}, 'member runtime scope drift')
    util.need(before.parent.archive(code)[2] == new, 'member archive roundtrip')
    util.need(nb['cells'][:1] == old_nb['cells'][:1] and nb['cells'][3:] == old_nb['cells'][3:],
              'unrelated notebook cells drift')
    for name, data in new.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<member notebook>', 'exec')
    return nb, meta, old, new, old_nb, old_meta


def artifacts(value, diagnostic=False):
    from tools import k1n_stage_v1 as builder
    c = canonical(value)
    if not new_path(c):
        return replay172.load().artifacts(c, diagnostic)
    cfg = config(c, diagnostic)
    nb, meta, old, new, old_nb, old_meta = package(c, diagnostic)
    base = p25_legacy()
    scope = dict(status='PASS', parent_stage=builder.stage(base, diagnostic).relative_to(ROOT).as_posix(),
        changed=sorted(n for n in old.keys() & new.keys() if old[n] != new[n]),
        added=sorted(new.keys()-old.keys()), removed=sorted(old.keys()-new.keys()),
        untouched={n:digest(v) for n,v in old.items() if new.get(n) == v},
        config_changed_fields=sorted(k for k in base.keys() | c.keys() if base.get(k) != c.get(k)),
        launch_pool_thread_budgets_unchanged=True, drop_o=True, k1_stride=c['k1_eval_stride'],
        registered_change='P25/P25S3 members, optional N32 ratio, v9 fail-loud, guard8.8 and identifiers',
        metadata_change='id/title and registered member datasets only',
        notebook_change='runtime import, archive/pins, config and markdown only')
    diff = []
    pairs = [(n, old.get(n,b''), new.get(n,b'')) for n in sorted(old.keys() | new.keys())
             if old.get(n) != new.get(n)]
    pairs += [(n, util.encoded(a), util.encoded(z)) for n,a,z in (
        ('notebook',old_nb,nb), ('metadata',old_meta,meta), ('config',base,c))]
    for name, left, right in pairs:
        diff += difflib.unified_diff(left.decode().splitlines(True), right.decode().splitlines(True),
                                    fromfile='P25/'+name,
                                    tofile='members/'+name)
    outputs = {'kernel/release.ipynb':util.encoded(nb), 'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c), 'runtime-projection.json':util.encoded(projection(c)),
        'payload-scope.json':util.encoded(scope), 'full-payload-notebook.diff':''.join(diff).encode(),
        'CHANGELOG.md':b'# Generator v1.7.3\n\nRegistered P25/P25S3 member ensemble; optional N32-ratio qualification.\n'
            b'Projection is an estimate, not qualification. N32 is never-submit.\n'}
    paths = ['tools/k1n_stage_v1.py', 'tools/k1n_config_v1.py', 'tools/k1n_members_v1.py',
             'tools/k1n_compat_v1.py', 'tools/k1sr_runtime_v9.py']
    paths += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tools/k1n_compat').rglob('*.py'))
    # Bind the complete pinned parent's provenance as well as this transformation.
    engine, _ = builder.baseline(base)
    _, parent_reg = builder.frozen_parent(engine, base, diagnostic)
    reg = dict(schema='k1n-registration/v1', **cfg, builder_version=builder.VERSION,
        builder_sources={n:util.sha(ROOT/n) for n in paths},
        parent_registration_sha256=digest(util.encoded(parent_reg)),
        stage=builder.stage(c, diagnostic).relative_to(ROOT).as_posix(),
        parent_stage=scope['parent_stage'], sources={n:digest(v) for n,v in new.items()},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),
        metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        guard_seconds=c['k1_guard_hours']*3600, execution_verified=False,
        independent_review='REQUIRED', qualified_for_submission=False,
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg
