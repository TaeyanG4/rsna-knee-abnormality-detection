"""v1.0.0: opt-in v8 payload construction and pinned v1.3.1 regeneration.

Changelog: preserve all policy-absent artifacts from source; v8 compares to the
actual registered parent, including the R6 stride2 parent and N32 siblings.
"""
import base64
import copy
import difflib
import json
from types import ModuleType

from tools import c85r7_r6w958k1x3sr_v1164 as before
from tools import k1n_config_v1 as schema

ROOT, util = before.ROOT, before.util
exact, digest = before.exact, before.digest
PINS = {
    'k1n_stage_v1.py': 'af59f9c97854d95acc13d74d0e5e84fb56dd2d0df3e9a9bd22f7f696da976af2',
    'k1n_config_v1.py': '928d8a8010600726e754435735f8e673f0f469e61a6b6873585cb5a4eb5b8035',
}
PARENTS = {
    ('1.1.71-r6w958k1n3s2dav2fl', 'taeyangg4/knee-lb-r6w958k1n3s2dav2fl-v1'):
        'configs/k1n/x3-stride2-drop-o-alpha-vector-v2.json',
    ('1.1.71-r6s2k1n3s2av2fl', 'taeyangg4/knee-lb-r6s2k1n3s2av2fl-v1'):
        'configs/k1n/x3-stride2-alpha-vector-v2-r6s2.json',
}


def legacy_engine():
    """Replay the original source, including its original source inventory.

    Only source locations are redirected. No stored output is returned as
    regeneration evidence. New compatibility sources were not in that inventory.
    """
    directory = ROOT/'tools/k1n_compat/v131'
    for name, pin in PINS.items():
        util.need(util.sha(directory/name) == pin, 'v131 compat source drift: '+name)
    frozen_schema = ModuleType('k1n_v131_schema')
    exec(compile((directory/'k1n_config_v1.py').read_bytes(), '<v131 schema>', 'exec'),
         frozen_schema.__dict__)
    engine = ModuleType('k1n_v131_builder')
    engine.schema = frozen_schema
    engine.source_sha = lambda name: util.sha(
        directory/name.split('/')[-1] if name in ('tools/'+n for n in PINS) else ROOT/name)
    text = (directory/'k1n_stage_v1.py').read_text(encoding='utf-8')
    text = exact(text, 'from tools import k1n_config_v1 as schema', '# pinned schema supplied')
    text = exact(text, 'util.sha(ROOT/n)', 'source_sha(n)')
    text = exact(text, "(ROOT/'tools/k1n_compat').rglob('*.py')]",
                 "(ROOT/'tools/k1n_compat').rglob('*.py') if p.parent.name in ('v110', 'v120')]")
    exec(compile(text, '<v131 builder>', 'exec'), engine.__dict__)
    return engine


def without_policy(value):
    c = schema.validate(value)
    c.pop('k1_failure_policy', None)
    c.pop('k1_max_failed_fraction', None)
    return c


def parent(value):
    c = schema.validate(value)
    filename = PARENTS.get((c['release_version'], c['kernel_id']))
    util.need(filename is not None, 'v8 requires an explicit registered parent identity')
    p = schema.validate(json.loads((ROOT/filename).read_bytes()))
    recipe = lambda v: {k:x for k,x in v.items() if k not in ('release_version', 'kernel_id')}
    util.need(recipe(without_policy(c)) == recipe(p), 'v8 parent recipe drift')
    return p


def config(value, diagnostic=False):
    c = schema.validate(value)
    cfg = legacy_engine().config(without_policy(c), diagnostic)
    cfg.update(config_sha256=digest(util.encoded(c)), registered_config_sha256=digest(util.encoded(c)),
               k1_failure_policy=c['k1_failure_policy'], k1_max_failed_fraction=c['k1_max_failed_fraction'])
    cfg['formula'] += ('; failed-study K1 rank input = parent rank before outer rank; '
                       'systemic failure or failed fraction above registered limit raises after publication')
    return cfg


def package(value, diagnostic=False):
    c = schema.validate(value)
    p = parent(c)
    engine = legacy_engine()
    outputs, _ = engine.artifacts(p, diagnostic)
    dest = engine.stage(p, diagnostic)
    util.need(dest.is_dir(), 'v8 immutable parent missing')
    util.need({f.relative_to(dest).as_posix() for f in dest.rglob('*') if f.is_file()} == set(outputs),
              'v8 parent inventory drift')
    for name, data in outputs.items():
        util.need((dest/name).read_bytes() == data, 'v8 parent artifact drift: '+name)
    old_nb, old_meta, _, old, _, _ = engine.package(p, diagnostic)
    nb, meta, new = copy.deepcopy(old_nb), copy.deepcopy(old_meta), dict(old)
    oc, cfg = engine.config(p, diagnostic), config(c, diagnostic)
    del new['tools/k1sr_runtime_v7.py']
    source = (ROOT/'tools/k1sr_runtime_v8.py').read_text(encoding='utf-8')
    new['tools/k1sr_runtime_v8.py'] = exact(source,
        "REGISTERED = dict(config_schema.current(), k1_failure_policy='study_fallback_fail_loud')",
        'REGISTERED = '+repr(c)).encode()
    new['tools/k1n_config_v1.py'] = (ROOT/'tools/k1n_config_v1.py').read_bytes()
    port, release, codec = 'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py'
    new[port] = exact(old[port].decode(), 'k1sr_runtime_v7', 'k1sr_runtime_v8').encode()
    new[release] = exact(old[release].decode(), digest(old[port]), digest(new[port])).encode()
    text = exact(old[codec].decode(), digest(old[release]), digest(new[release]))
    util.need(text.count(oc['release_version']) == 2, 'v8 codec identity anchors')
    text = text.replace(oc['release_version'], cfg['release_version'])
    new[codec] = exact(text, oc['kernel_id'], cfg['kernel_id']).encode()
    old_code = ''.join(old_nb['cells'][2]['source'])
    blob, pins, _ = before.parent.archive(old_code)
    new_blob = before.packed(new)
    code = exact(old_code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(new_blob).decode()))
    code = exact(code, 'PINS = '+repr(pins), 'PINS = '+repr({n:digest(v) for n,v in new.items()}))
    code = exact(code, repr(digest(blob)), repr(digest(new_blob)))
    code = exact(code, 'from tools import k1sr_runtime_v7 as _k1sr',
                 'from tools import k1sr_runtime_v8 as _k1sr')
    code = exact(code, repr(oc), repr(cfg))
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = ['# K1N runtime v8 '+cfg['release_version']+'\n', cfg['formula']+'\n',
        'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else 'CPU prepared; independent/GPU qualification pending.\n']
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1])
    util.need(meta == dict(old_meta, id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1]),
              'v8 metadata scope drift')
    util.need({n for n in old.keys() & new.keys() if old[n] != new[n]} ==
              {port, release, codec, 'tools/k1n_config_v1.py'}, 'v8 payload scope drift')
    util.need(before.parent.archive(code)[2] == new, 'v8 archive roundtrip')
    util.need(nb['cells'][:1] == old_nb['cells'][:1] and nb['cells'][3:] == old_nb['cells'][3:],
              'v8 unrelated notebook cells drift')
    for name, data in new.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<v8 notebook>', 'exec')
    return nb, meta, old, new, old_nb, old_meta


def artifacts(value, diagnostic=False):
    from tools import k1n_stage_v1 as builder
    c = schema.validate(value)
    cfg = config(c, diagnostic)
    nb, meta, old, new, old_nb, old_meta = package(c, diagnostic)
    diff = []
    for name in sorted(old.keys() | new.keys()):
        if old.get(name) != new.get(name):
            diff += difflib.unified_diff(old.get(name,b'').decode().splitlines(True),
                new.get(name,b'').decode().splitlines(True), fromfile='parent/'+name, tofile='v8/'+name)
    for name, left, right in [('notebook', old_nb, nb), ('metadata', old_meta, meta)]:
        diff += difflib.unified_diff(util.encoded(left).decode().splitlines(True),
            util.encoded(right).decode().splitlines(True), fromfile='parent/'+name, tofile='v8/'+name)
    scope = dict(status='PASS', parent_stage=str(builder.stage(parent(c), diagnostic).relative_to(ROOT)),
        changed=sorted(n for n in old.keys() & new.keys() if old[n] != new[n]),
        added=sorted(new.keys()-old.keys()), removed=sorted(old.keys()-new.keys()),
        untouched={n:digest(v) for n,v in old.items() if new.get(n) == v},
        registered_change='v8 study fallback/fail-loud policy, schema and identity/chained pins only',
        alpha_unchanged=True, drop_o_unchanged=True, r6_stride_unchanged=True,
        launch_pool_thread_budgets_unchanged=True,
        notebook_change='runtime import, archive/pins, registered config and markdown only',
        metadata_change='id/title only')
    outputs = {
        'kernel/release.ipynb': util.encoded(nb), 'kernel/kernel-metadata.json': util.encoded(meta),
        'config.json': util.encoded(c), 'payload-scope.json': util.encoded(scope),
        'full-payload-notebook.diff': ''.join(diff).encode(),
        'CHANGELOG.md': ('# Runtime v8.0.0 / generator v1.4.0\n\n'+cfg['formula']+'\n'
            'Per-study fallback continues K1; systemic errors raise after preserving parent CSV/receipt.\n'
            'All model, alpha, O and R6 choices match the parent. N32 is never-submit.\n').encode(),
    }
    paths = ['tools/k1n_stage_v1.py', 'tools/k1n_stage_v8.py', 'tools/k1n_config_v1.py',
             'tools/k1sr_runtime_v8.py', 'tools/k1n_compat_v1.py', 'tools/k1n_parent_v1.py']
    paths += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tools/k1n_compat').rglob('*.py'))
    reg = dict(schema='k1n-registration/v1', **cfg, builder_version=builder.VERSION,
        builder_sources={n:util.sha(ROOT/n) for n in paths},
        stage=str(builder.stage(c, diagnostic).relative_to(ROOT)), parent_stage=scope['parent_stage'],
        sources={n:digest(v) for n,v in new.items()}, kernel_sha256=digest(outputs['kernel/release.ipynb']),
        metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        guard_seconds=28800, load_timeout_seconds=1500, call_timeout_seconds=120,
        call_timeout_scope='transfer + all classifiers per study; excludes loading',
        execution_verified=False, independent_review='REQUIRED', qualified_for_submission=False,
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg
