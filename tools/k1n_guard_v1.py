"""K1 guard-only generator v1.0.0; CPU-only, immutable v1.4.0 source replay.

Changelog: register hours without changing the v8 serving implementation.
The baked guard and config hash are constants. The old embedded model schema
still validates its original fields; the full guard-bearing config is validated
here, hashed into CONFIG_SHA, and passed intact to finish/receipt.
"""
import base64
import copy
import difflib
import json
from types import ModuleType

from tools import k1n_config_v1 as schema
from tools import c85r7_r6w958k1x3sr_v1164 as before

VERSION = '1.0.0'
ROOT, util = before.ROOT, before.util
exact, digest = before.exact, before.digest
PINS = {
    'k1n_config_v1.py': 'ae4658c911ec85deb0499adbe917846f29184502d81ab8672490a010af4ef03f',
    'k1n_stage_v1.py': 'b4241279677be9c0dc76f0c7b4617e6fc47d030b3e81977d9b1ac76d4e1432a8',
    'k1n_stage_v8.py': '1103d5be1a5c3f1f71d5506929e0fb524c3d841eec958b57ba0061d0d590f393',
    'k1sr_runtime_v8.py': '5a0c680512ca160db8bfdfff4c6de550d0b7649d8cde8c6d9e19261dbc81f104',
}
PARENTS = {
    ('1.1.72-r6w958k1n3s2dav2fl', 'taeyangg4/knee-lb-r6w958k1n3s2dav2fl-v2'):
        'configs/k1n/x3-stride2-drop-o-alpha-vector-v2-v8.json',
    ('1.1.72-r6s2k1n3s2av2fl', 'taeyangg4/knee-lb-r6s2k1n3s2av2fl-v2'):
        'configs/k1n/x3-stride2-alpha-vector-v2-r6s2-v8.json',
}


def legacy_engine():
    """Execute pinned source, restoring its original source paths/inventory."""
    directory = ROOT/'tools/k1n_compat/v140'
    for name, pin in PINS.items():
        util.need(util.sha(directory/name) == pin, 'v140 source drift: '+name)
    frozen_schema = ModuleType('k1n_v140_schema')
    exec(compile((directory/'k1n_config_v1.py').read_bytes(), '<v140 schema>', 'exec'),
         frozen_schema.__dict__)
    builder, runtime_builder = ModuleType('k1n_v140_builder'), ModuleType('k1n_v140_v8')
    runtime_builder.schema, runtime_builder.builder = frozen_schema, builder
    runtime_builder.source_path = lambda name: directory/name.split('/')[-1] if name in (
        'tools/'+n for n in PINS) else ROOT/name
    text = (directory/'k1n_stage_v8.py').read_text(encoding='utf-8')
    text = exact(text, '\nfrom tools import k1n_config_v1 as schema\n', '\n# pinned schema supplied\n')
    text = exact(text, 'from tools import k1n_stage_v1 as builder', '# pinned builder supplied')
    for name in ('k1sr_runtime_v8.py', 'k1n_config_v1.py'):
        text = exact(text, f"(ROOT/'tools/{name}')", f"source_path('tools/{name}')")
    text = exact(text, 'builder_sources={n:util.sha(ROOT/n) for n in paths}',
                 'builder_sources={n:util.sha(source_path(n)) for n in paths}')
    text = exact(text, "(ROOT/'tools/k1n_compat').rglob('*.py'))",
                 "(ROOT/'tools/k1n_compat').rglob('*.py') if p.parent.name in ('v110', 'v120', 'v131'))")
    exec(compile(text, '<v140 v8>', 'exec'), runtime_builder.__dict__)
    builder.schema, builder.fail_loud = frozen_schema, runtime_builder
    text = (directory/'k1n_stage_v1.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema supplied')
    text = exact(text, 'from tools import k1n_stage_v8 as fail_loud', '# pinned v8 supplied')
    exec(compile(text, '<v140 builder>', 'exec'), builder.__dict__)
    return builder


def without_guard(value):
    c = schema.validate(value)
    c.pop('k1_guard_hours', None)
    return c


def parent(value):
    c = schema.validate(value)
    filename = PARENTS.get((c['release_version'], c['kernel_id']))
    util.need(filename is not None, 'guard requires an explicit registered parent identity')
    p = schema.validate(json.loads((ROOT/filename).read_bytes()))
    recipe = lambda v: {k:x for k,x in without_guard(v).items()
                        if k not in ('release_version', 'kernel_id')}
    util.need(recipe(c) == recipe(p), 'guard parent recipe drift')
    return p


def config(value, diagnostic=False):
    c = schema.validate(value)
    cfg = legacy_engine().config(without_guard(c), diagnostic)
    if 'k1_guard_hours' in c:
        cfg.update(k1_guard_hours=c['k1_guard_hours'],
                   config_sha256=digest(util.encoded(c)),
                   registered_config_sha256=digest(util.encoded(c)))
    return cfg


def package(value, diagnostic=False):
    c = schema.validate(value)
    engine = legacy_engine()
    if 'k1_guard_hours' not in c:
        return engine.package(c, diagnostic)
    p = parent(c)
    outputs, _ = engine.artifacts(p, diagnostic)
    dest = engine.stage(p, diagnostic)
    util.need({f.relative_to(dest).as_posix() for f in dest.rglob('*') if f.is_file()} == set(outputs),
              'guard parent inventory drift')
    for name, data in outputs.items():
        util.need((dest/name).read_bytes() == data, 'guard parent artifact drift: '+name)
    old_nb = json.loads(outputs['kernel/release.ipynb'])
    old_meta = json.loads(outputs['kernel/kernel-metadata.json'])
    old_code = ''.join(old_nb['cells'][2]['source'])
    blob, pins, old = before.parent.archive(old_code)
    nb, meta, new = copy.deepcopy(old_nb), copy.deepcopy(old_meta), dict(old)
    oc, cfg = engine.config(p, diagnostic), config(c, diagnostic)
    runtime, codec = 'tools/k1sr_runtime_v8.py', 'exp169_release_c85r7codec.py'
    text = exact(old[runtime].decode(), 'REGISTERED = '+repr(p),
                 'REGISTERED = '+repr(without_guard(c)))
    text = exact(text, 'CONFIG_SHA = hashlib.sha256(util.encoded(REGISTERED)).hexdigest()',
                 'CONFIG_SHA = '+repr(digest(util.encoded(c))))
    new[runtime] = exact(text, 'GUARD_SECONDS = 8 * 3600',
                         'GUARD_SECONDS = '+repr(c['k1_guard_hours'] * 3600)).encode()
    text = old[codec].decode()
    util.need(text.count(oc['release_version']) == 2, 'guard codec identity anchors')
    text = text.replace(oc['release_version'], cfg['release_version'])
    new[codec] = exact(text, oc['kernel_id'], cfg['kernel_id']).encode()
    new_blob = before.packed(new)
    code = exact(old_code, repr(base64.b64encode(blob).decode()),
                 repr(base64.b64encode(new_blob).decode()))
    code = exact(code, 'PINS = '+repr(pins), 'PINS = '+repr({n:digest(v) for n,v in new.items()}))
    code = exact(code, repr(digest(blob)), repr(digest(new_blob)))
    code = exact(code, repr(oc), repr(cfg))
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = [
        line.replace(oc['release_version'], cfg['release_version'])
        for line in nb['cells'][1]['source']]
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1])
    util.need({n for n in old if old[n] != new[n]} == {runtime, codec},
              'guard payload scope drift')
    util.need(before.parent.archive(code)[2] == new, 'guard archive roundtrip')
    for name, data in new.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<guard notebook>', 'exec')
    return nb, meta, old, new, old_nb, old_meta


def artifacts(value, diagnostic=False):
    from tools import k1n_stage_v1 as builder
    c = schema.validate(value)
    if 'k1_guard_hours' not in c:
        return legacy_engine().artifacts(c, diagnostic)
    cfg = config(c, diagnostic)
    nb, meta, old, new, old_nb, old_meta = package(c, diagnostic)
    diff = []
    for name in sorted(old):
        if old[name] != new[name]:
            diff += difflib.unified_diff(old[name].decode().splitlines(True),
                new[name].decode().splitlines(True), fromfile='parent/'+name, tofile='guard/'+name)
    for name, left, right in [('notebook', old_nb, nb), ('metadata', old_meta, meta)]:
        diff += difflib.unified_diff(util.encoded(left).decode().splitlines(True),
            util.encoded(right).decode().splitlines(True), fromfile='parent/'+name, tofile='guard/'+name)
    scope = dict(status='PASS', parent_stage=builder.stage(parent(c), diagnostic).relative_to(ROOT).as_posix(),
        changed=sorted(n for n in old if old[n] != new[n]), added=[], removed=[],
        untouched={n:digest(v) for n,v in old.items() if new[n] == v},
        registered_change='K1 guard constant and config identity/hash, release/slug only',
        launch_pool_thread_unchanged=True, call_load_timeouts_unchanged=True,
        scientific_recipe_unchanged=True,
        notebook_change='archive/pins, registered config and release identity only',
        metadata_change='id/title only')
    outputs = {
        'kernel/release.ipynb': util.encoded(nb), 'kernel/kernel-metadata.json': util.encoded(meta),
        'config.json': util.encoded(c), 'payload-scope.json': util.encoded(scope),
        'full-payload-notebook.diff': ''.join(diff).encode(),
        'CHANGELOG.md': ('# Generator v1.5.0 / runtime v8 guard registration\n\n'
            f"Registered k1_guard_hours={c['k1_guard_hours']}; default 8.0 replays v1.4.0 bytes.\n"
            'Only guard constant and identifiers change in serving payload.\n'
            'Full registration hash includes hours; receipt includes hours and guard_seconds.\n'
            'Model schema payload is unchanged; CONFIG_SHA pins the complete stage config.\n'
            'N32 is never-submit. Independent review and GPU qualification pending.\n').encode(),
    }
    paths = ['tools/k1n_stage_v1.py', 'tools/k1n_config_v1.py', 'tools/k1n_guard_v1.py',
             'tools/k1n_stage_v8.py', 'tools/k1sr_runtime_v8.py',
             'tools/k1n_compat_v1.py', 'tools/k1n_parent_v1.py']
    paths += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tools/k1n_compat').rglob('*.py'))
    reg = dict(schema='k1n-registration/v1', **cfg, builder_version=builder.VERSION,
        builder_sources={n:util.sha(ROOT/n) for n in paths},
        stage=builder.stage(c, diagnostic).relative_to(ROOT).as_posix(),
        parent_stage=scope['parent_stage'], sources={n:digest(v) for n,v in new.items()},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),
        metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        guard_seconds=c['k1_guard_hours']*3600, load_timeout_seconds=1500, call_timeout_seconds=120,
        call_timeout_scope='transfer + all classifiers per study; excludes loading',
        execution_verified=False, independent_review='REQUIRED', qualified_for_submission=False,
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg
