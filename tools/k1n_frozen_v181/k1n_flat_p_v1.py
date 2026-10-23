"""v1.0.1: registered flat P transformation of all accepted K1N recipes.

Changelog v1.0.1: compose with v1.8.1 N32-ratio/P25S3 contracts and 9 h limit.

Only four Raptor/R6 shares change. Runtime, ranks, failure substitution, R224,
O and K1 arithmetic remain inherited. No data/device/network access.
"""
from __future__ import annotations

import base64
import copy
import difflib

from tools import k1n_config_v1 as schema
from tools import k1n_replay_v173 as replay
from tools import k1n_members_v1 as members

ROOT, util, exact = replay.ROOT, replay.util, replay.exact
VERSION = '1.0.1'
PUBLIC = (.5, .5, .5, .5, .5, .2, .6, .5, .8, .35, .5, .5)
FLAT = (.5,) * 12
CHANGED = (5, 6, 8, 9)
HELPER = 'exp169_r224.py'
CODEC = 'exp169_release_c85r7codec.py'
RUNTIME = 'tools/k1sr_runtime_v9.py'
SCHEMA = 'tools/k1n_config_v1.py'
FORMULA = ('P_t=0.5*(0.65*r(R384)_t+0.35*r(R224)_t)+0.5*r(R6flip)_t; '
           'own-success average pct ranks; inherited survivor overrides')


def canonical(value):
    c = schema.validate(value)
    util.need(c.get('p_weight_mode') == 'flat', 'registered flat P required')
    public = dict(c)
    del public['p_weight_mode']
    # This also refuses historical S1/S2 identities and recipe drift.
    public = members.canonical(public)
    # Serialized configs sort mapping keys; embedded REGISTERED repr must not
    # depend on the original template's model field order after reloading.
    public['models'] = [{k:m[k] for k in sorted(m)} for m in public['models']]
    return dict(public, p_weight_mode='flat')


def parent_engine(c):
    # Historical recipes retain exact pinned public bytes. New ratio/P25S3
    # recipes use the current metadata contract before applying flat P.
    if members.new_path(c):
        from tools import k1n_stage_v1 as builder
        return builder
    return replay.load()


def parent_config(c):
    engine = replay.load()
    public = dict(c)
    del public['p_weight_mode']
    # Use the actual reviewed S1 stage when the entire recipe matches.
    s1 = engine.runtime9.registered()
    recipe = lambda v: {k:x for k,x in v.items() if k not in ('release_version', 'kernel_id')}
    if recipe(public) == recipe(s1):
        return s1
    return public


def config(value, diagnostic=False):
    c = canonical(value)
    engine = parent_engine(c)
    cfg = engine.config(parent_config(c), diagnostic)
    cfg.update(release_version=c['release_version']+('-n32' if diagnostic else ''),
        kernel_id=c['kernel_id']+('-n32' if diagnostic else ''),
        p_weight_mode='flat', p_raptor_weights=list(FLAT),
        config_sha256=engine.digest(util.encoded(c)),
        registered_config_sha256=engine.digest(util.encoded(c)))
    cfg['formula'] += '; p_weight_mode=flat; '+FORMULA
    return cfg


def patch_math(files, drop_o):
    """Exact anchors fail closed if the inherited scientific recipe changes."""
    engine = replay.load()
    result = dict(files)
    text = files[HELPER].decode()
    anchor = '    w = np.asarray('+repr(PUBLIC)+')'
    util.need(text.count(anchor) == 2, 'P arithmetic/reference anchors')
    text = text.replace(anchor, '    w = np.asarray('+repr(FLAT)+')')
    text = exact(text, "w=.5 except Lateral OA=.2, PF OA=.6, Synovitis=.8, Baker's=.35",
                 'p_weight_mode=flat; w=.5 for every target')
    result[HELPER] = text.encode()
    text = exact(files[CODEC].decode(), engine.digest(files[HELPER]), engine.digest(result[HELPER]))
    if drop_o:
        text = exact(text, 'weight = np.asarray((.5, .5, .5, .5, .5, .2, .6, .5, .8, .35, .5, .5))',
                     'weight = np.asarray('+repr(FLAT)+')')
        text = exact(text, 'FORMULA = "INF3 P branch only; R224 retained; registered drop_o"',
                     'FORMULA = '+repr('INF3 P branch only; registered drop_o; p_weight_mode=flat; '+FORMULA))
    # P diagnostics carry their own formula in both O and drop-O variants.
    text = exact(text, 'w=.5 except LatOA=.2/PFOA=.6/Syn=.8/Baker=.35',
                 'p_weight_mode=flat; w=.5 for every target')
    result[CODEC] = text.encode()
    return result


def package(value, diagnostic=False):
    c = canonical(value)
    engine = parent_engine(c)
    pc = parent_config(c)
    old_nb, old_meta, _, old, _, _ = engine.package(pc, diagnostic)
    nb, meta = copy.deepcopy(old_nb), copy.deepcopy(old_meta)
    cfg, oc = config(c, diagnostic), engine.config(pc, diagnostic)
    new = patch_math(old, c['drop_o'])
    # Preserve all runtime code; register the new mode in the existing identity hash.
    new[RUNTIME] = exact(old[RUNTIME].decode(), 'REGISTERED = '+repr(pc),
                         'REGISTERED = '+repr(c)).encode()
    new[SCHEMA] = (ROOT/SCHEMA).read_bytes()
    text = new[CODEC].decode()
    if oc['release_version'] != cfg['release_version']:
        util.need(text.count(oc['release_version']) == 2, 'codec identity anchors')
        text = text.replace(oc['release_version'], cfg['release_version'])
        text = exact(text, oc['kernel_id'], cfg['kernel_id'])
    new[CODEC] = text.encode()
    code = ''.join(old_nb['cells'][2]['source'])
    blob, pins, _ = engine.before.parent.archive(code)
    packed = engine.before.packed(new)
    code = exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(packed).decode()))
    code = exact(code, 'PINS = '+repr(pins), 'PINS = '+repr({n:engine.digest(v) for n,v in new.items()}))
    code = exact(code, repr(engine.digest(blob)), repr(engine.digest(packed)))
    code = exact(code, repr(oc), repr(cfg))
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = ['# Registered flat P twin '+cfg['release_version']+'\n',
        cfg['formula']+'\n', 'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else
        'CPU prepared; independent review and inference qualification pending.\n']
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1])
    expected = {HELPER, CODEC, RUNTIME}
    if old[SCHEMA] != new[SCHEMA]:
        expected.add(SCHEMA)
    util.need(set(old) == set(new) and {n for n in old if old[n] != new[n]} ==
              expected, 'flat P payload scope drift')
    util.need(meta == dict(old_meta, id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1]),
              'flat P metadata scope drift')
    util.need(nb['cells'][:1] == old_nb['cells'][:1] and nb['cells'][3:] == old_nb['cells'][3:],
              'flat P unrelated notebook cells drift')
    util.need(engine.before.parent.archive(code)[2] == new, 'flat P archive roundtrip')
    for name, data in new.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<flat P notebook>', 'exec')
    return nb, meta, old, new, old_nb, old_meta


def artifacts(value, diagnostic=False):
    from tools import k1n_stage_v1 as builder
    c = canonical(value)
    engine = parent_engine(c)
    pc = parent_config(c)
    nb, meta, old, new, old_nb, old_meta = package(c, diagnostic)
    cfg = config(c, diagnostic)
    projection = members.projection({k:v for k,v in c.items() if k != 'p_weight_mode'})
    scope = dict(status='PASS', changed=sorted(n for n in old if old[n] != new[n]),
        added=[], removed=[], untouched={n:engine.digest(v) for n,v in old.items() if new[n] == v},
        changed_p_columns=list(CHANGED), unchanged_p_columns=[j for j in range(12) if j not in CHANGED],
        parent_config=pc, parent_is_existing_stage=(
            engine.stage(pc, diagnostic) != builder.stage(c, diagnostic)
            and engine.stage(pc, diagnostic).exists()),
        registered_change='four P shares to .5; identity, formula, schema and receipt binding',
        launch_pool_thread_budgets_unchanged=True, r224_flip_o_k1_alpha_ranks_unchanged=True)
    pairs = [(n,old[n],new[n]) for n in old if old[n] != new[n]]
    pairs += [(n,util.encoded(a),util.encoded(z)) for n,a,z in (
        ('notebook',old_nb,nb), ('metadata',old_meta,meta), ('config',pc,c))]
    diff = []
    for name,left,right in pairs:
        diff += difflib.unified_diff(left.decode().splitlines(True), right.decode().splitlines(True),
                                    fromfile='public/'+name, tofile='flat/'+name)
    outputs = {'kernel/release.ipynb':util.encoded(nb), 'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c), 'payload-scope.json':util.encoded(scope),
        'runtime-projection.json':util.encoded(projection),
        'full-payload-notebook.diff':''.join(diff).encode(),
        'CHANGELOG.md':b'# Generator v1.8.1 / flat P v1.0.1\n\nFour P shares to .5; N32 qualification <= 9 h.\n'
                       b'Historical public replay preserved. N32 NEVER SUBMIT. Needs verifier.\n'}
    _, parent_reg = engine.artifacts(pc, diagnostic)
    paths = ['tools/k1n_flat_p_v1.py','tools/k1n_config_v1.py','tools/k1n_stage_v1.py',
             'tools/k1n_members_v1.py', 'tools/k1n_replay_v173.py']+['tools/k1n_frozen_v173/'+n for n in replay.PINS]
    reg = dict(schema='k1n-registration/v1', **cfg, builder_version=builder.VERSION,
        builder_sources={n:util.sha(ROOT/n) for n in paths},
        parent_registration_sha256=engine.digest(util.encoded(parent_reg)),
        stage=builder.stage(c,diagnostic).relative_to(ROOT).as_posix(),
        parent_stage=engine.stage(pc,diagnostic).relative_to(ROOT).as_posix(),
        parent_is_existing_stage=scope['parent_is_existing_stage'],
        sources={n:engine.digest(v) for n,v in new.items()},
        kernel_sha256=engine.digest(outputs['kernel/release.ipynb']),
        metadata_sha256=engine.digest(outputs['kernel/kernel-metadata.json']),
        guard_seconds=c['k1_guard_hours']*3600, execution_verified=False,
        independent_review='REQUIRED', qualified_for_submission=False,
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg


def run(command, value):
    from tools import k1n_stage_v1 as builder
    c = canonical(value)
    util.need(command in ('prepare','validate'), 'unknown command')
    pairs = [(d,builder.stage(c,d),*artifacts(c,d)) for d in (False,True)]
    if command == 'prepare':
        util.need(all(not dest.exists() for _,dest,_,_ in pairs), 'immutable stage exists; use validate')
        for _,dest,outputs,_ in pairs:
            dest.mkdir()
            for name,data in outputs.items():
                util.write_once(dest/name,data)
    for _,dest,outputs,_ in pairs:
        util.need({p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()} == set(outputs),
                  'immutable artifact inventory drift')
        for name,data in outputs.items():
            util.need((dest/name).read_bytes() == data, 'immutable artifact drift: '+name)
    return dict(status='PREPARED_LOCAL_ONLY' if command == 'prepare' else 'PASS_CPU_VALIDATE',
        stages=[dict(stage=reg['stage'], kernel_id=reg['kernel_id'],
                     kernel_sha256=reg['kernel_sha256'], never_submit=d) for d,_,_,reg in pairs],
        qualified_for_submission=False)
