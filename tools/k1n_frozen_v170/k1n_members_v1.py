"""v1.0.0: S2-derived member ensembles and hidden-calibrated CPU projection.

Changelog: new identities only; frozen v9.0.0 remains untouched. Bake v9.1.0
only for unequal member weights; equal uses the original np.mean arithmetic.
"""
import base64
import copy
import difflib
import json

from tools import k1n_config_v1 as schema
from tools import k1n_replay_v160 as replay160
from tools import c85r7_r6w958k1x3sr_v1164 as before

ROOT, util, exact = replay160.ROOT, replay160.util, replay160.exact
digest = before.digest
VERSION = '1.0.0'


def selected(c):
    return c.get('k1_runtime_version') == 9 and (
        c['release_version'], c['kernel_id']) not in (
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


def projection(c):
    util.need(not c['drop_o'] and c.get('r6_eval_stride') == 2 and
              c['k1_eval_stride'] == 3, 'projection requires O, R6 stride2, K1 stride3')
    return dict(projection_hours(len(c['models'])), drop_o=False, r6_eval_stride=2,
                k1_eval_stride=3)


def canonical(value):
    c = schema.validate(value)
    util.need(selected(c), 'member ensembles require a new release and slug')
    base = replay160.load().runtime9.registered(True)
    util.need(c['release_version'] != base['release_version'] and
              c['kernel_id'] != base['kernel_id'], 'member ensembles require new identifiers')
    allowed = {'release_version', 'kernel_id', 'models', 'member_weights'}
    util.need({k:v for k,v in c.items() if k not in allowed} ==
              {k:v for k,v in base.items() if k not in allowed}, 'member S2 recipe drift')
    # Stable insertion order makes field-order changes irrelevant.
    base.update({k:c[k] for k in ('release_version', 'kernel_id', 'models')})
    if 'member_weights' in c:
        base['member_weights'] = c['member_weights']
    projection(base)
    return base


def config(value, diagnostic=False):
    c = canonical(value)
    old = replay160.load().runtime9
    cfg = old.config(old.registered(True), diagnostic)
    cfg.update(release_version=c['release_version']+('-n32' if diagnostic else ''),
        kernel_id=c['kernel_id']+('-n32' if diagnostic else ''),
        models=c['models'], k1_models=[m['id'] for m in c['models']],
        model_sha256={m['id']:m['checkpoint_sha256'] for m in c['models']},
        asset_manifest_sha256={m['id']:m['manifest_sha256'] for m in c['models']},
        config_sha256=digest(util.encoded(c)), registered_config_sha256=digest(util.encoded(c)),
        runtime_projection=projection(c))
    if 'member_weights' in c:
        cfg['member_weights'] = c['member_weights']
        cfg['formula'] = cfg['formula'].replace('mean(per-target rank(K1_model_FP16))',
            'sum_i(member_weights[i]*per-target rank(K1_i_FP16))')
    return cfg


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
    pinned = replay160.load()
    base = pinned.runtime9.registered(True)
    old_nb, old_meta, _, old, _, _ = pinned.package(base, diagnostic)
    nb, meta, new = copy.deepcopy(old_nb), copy.deepcopy(old_meta), dict(old)
    oc, cfg = pinned.config(base, diagnostic), config(c, diagnostic)
    runtime, codec = 'tools/k1sr_runtime_v9.py', 'exp169_release_c85r7codec.py'
    new[runtime] = runtime_source(old[runtime], base, c)
    # Equal member stages need no new runtime validator; nonuniform weights do.
    if 'member_weights' in c:
        new['tools/k1n_config_v1.py'] = (ROOT/'tools/k1n_config_v1.py').read_bytes()
    text = old[codec].decode()
    util.need(text.count(oc['release_version']) == 2, 'codec identity anchors')
    text = text.replace(oc['release_version'], cfg['release_version'])
    new[codec] = exact(text, oc['kernel_id'], cfg['kernel_id']).encode()
    code = ''.join(old_nb['cells'][2]['source'])
    before = pinned.before
    blob, pins, _ = before.parent.archive(code)
    packed = before.packed(new)
    code = exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(packed).decode()))
    code = exact(code, 'PINS = '+repr(pins), 'PINS = '+repr({n:digest(v) for n,v in new.items()}))
    code = exact(code, repr(digest(blob)), repr(digest(packed)))
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
    allowed = {runtime, codec} | ({'tools/k1n_config_v1.py'} if 'member_weights' in c else set())
    util.need(new.keys() == old.keys() and
              {n for n in old if old[n] != new[n]} == allowed, 'member payload scope drift')
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
    c, cfg = canonical(value), config(value, diagnostic)
    nb, meta, old, new, old_nb, old_meta = package(c, diagnostic)
    base = replay160.load().runtime9.registered(True)
    scope = dict(status='PASS', parent_stage=builder.stage(base, diagnostic).relative_to(ROOT).as_posix(),
        changed=sorted(n for n in old if old[n] != new[n]), added=[], removed=[],
        untouched={n:digest(v) for n,v in old.items() if new[n] == v},
        config_changed_fields=sorted(k for k in base.keys() | c.keys() if base.get(k) != c.get(k)),
        launch_pool_thread_budgets_unchanged=True, drop_o=False, k1_stride=3,
        metadata_change='id/title and registered member datasets only',
        notebook_change='archive/pins, config and markdown only')
    diff = []
    pairs = [(n, old[n], new[n]) for n in sorted(old) if old[n] != new[n]]
    pairs += [(n, util.encoded(a), util.encoded(z)) for n,a,z in (
        ('notebook',old_nb,nb), ('metadata',old_meta,meta), ('config',base,c))]
    for name, left, right in pairs:
        diff += difflib.unified_diff(left.decode().splitlines(True), right.decode().splitlines(True),
                                    fromfile='S2/'+name, tofile='members/'+name)
    outputs = {'kernel/release.ipynb':util.encoded(nb), 'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c), 'runtime-projection.json':util.encoded(projection(c)),
        'payload-scope.json':util.encoded(scope), 'full-payload-notebook.diff':''.join(diff).encode(),
        'CHANGELOG.md':b'# Generator v1.7.0\n\nRegistered S2 member ensemble; optional weighted rank mean.\n'
            b'Projection is an estimate, not qualification. N32 is never-submit.\n'}
    paths = ['tools/k1n_stage_v1.py', 'tools/k1n_config_v1.py', 'tools/k1n_members_v1.py',
             'tools/k1n_replay_v160.py', 'tools/k1sr_runtime_v9.py']
    paths += ['tools/k1n_frozen_v160/'+n for n in replay160.PINS]
    # Bind the complete pinned parent's provenance as well as this transformation.
    _, parent_reg = replay160.load().artifacts(base, diagnostic)
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
