"""v1.0.0: registered S1/S2 runtime-v9 construction, CPU only.

Changelog: immutable current-x3 -> S1 (R6 stride2/v9/guard8.6), then
S1 -> S2 (alpha vector v2). All embedded files and notebook cells are diffed.
"""
import ast
import base64
import copy
import difflib
import json

from tools import c85r7_r6w958k1x3sr_v1164 as before
from tools import k1n_config_v1 as schema
from tools import k1n_compat_v1 as compat

ROOT, util = before.ROOT, before.util
exact, digest = before.exact, before.digest
S1 = ('1.1.73-r6s2k1x3fl', 'taeyangg4/knee-lb-r6s2k1x3fl-v1')
S2 = ('1.1.73-r6s2k1x3avfl', 'taeyangg4/knee-lb-r6s2k1x3avfl-v1')
BASE = 'v1.1.65-r6w958k1n3sr20'
PINS = {
    False: ('ae555d220fe45a085c4605177808b1213cbaa0503672cbef0e7933bd33ca4112',
            '6965d6b2d78d0875f84aa20d2dad58e10d20b8dadb7aaa940f234c193fcd881e',
            '594ee479ce25a12c6ab1a78ff6b2edee54e1979083bf4f48cd56e7c73f048ac1'),
    True: ('fd5f4adbe64fe1fcbcf62d210858a85bb907636107755c8e861b21b7340a7e1e',
           '26925ba8e9fed18c17edbd382221516f4ebeab3cbfd3761441b13bdfa3bfa999',
           'e506cb167be84ac16b2b5db087d236fc32673629943f68bed2c69456c3430cd4'),
}


def registered(vector=False):
    c = json.loads((ROOT/'configs/k1n/current-x3.json').read_bytes())
    version, slug = S2 if vector else S1
    c.update(release_version=version, kernel_id=slug, r6_eval_stride=2,
             k1_runtime_version=9, k1_failure_policy='study_fallback_fail_loud',
             k1_guard_hours=8.6)
    if vector:
        c['alpha_vector'] = json.loads(
            (ROOT/'configs/k1n/x3-stride2-alpha-vector-v2.json').read_bytes())['alpha_vector']
    return schema.validate(c)


def canonical(value):
    c = schema.validate(value)
    identity = c['release_version'], c['kernel_id']
    util.need(identity in (S1, S2), 'v9 requires registered S1/S2 identity')
    expected = registered(identity == S2)
    util.need(c == expected, 'v9 registered recipe drift')
    # New path is insertion-order independent. Historical paths remain frozen.
    return expected


def config(value, diagnostic=False):
    c = canonical(value)
    legacy = {k:v for k,v in c.items() if k not in (
        'r6_eval_stride', 'k1_runtime_version', 'k1_failure_policy',
        'k1_max_failed_fraction', 'k1_guard_hours')}
    cfg = compat.load('v120').config(legacy, diagnostic)
    cfg.update(r6_eval_stride=2, r6_manifest_eval_stride=1,
               shared_adapter_version='2.0.1-hb228-r6s2',
               config_sha256=digest(util.encoded(c)),
               registered_config_sha256=digest(util.encoded(c)),
               **{k:c[k] for k in ('k1_runtime_version', 'k1_failure_policy',
                                  'k1_max_failed_fraction', 'k1_guard_hours')})
    cfg['formula'] += ('; failed-study K1 rank input = parent rank before outer rank; '
                       'systemic failure or failed fraction above registered limit raises after publication')
    return cfg


def original(diagnostic):
    dest = before.parent.INF3.parent/(BASE+('-n32' if diagnostic else ''))
    values = []
    for name, pin in zip(('kernel/release.ipynb', 'kernel/kernel-metadata.json',
                          'registration.json'), PINS[diagnostic]):
        data = (dest/name).read_bytes()
        util.need(digest(data) == pin, 'current-x3 parent drift: '+name)
        values.append(json.loads(data))
    return values[:2]


def notebook_config(nb):
    tree = ast.parse(''.join(nb['cells'][2]['source']))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == '_k1sr.finish']
    util.need(len(calls) == 1, 'finish call anchor')
    return ast.literal_eval(calls[0].args[2])


def package(value, diagnostic=False):
    from tools import k1n_stage_v1 as builder
    c = canonical(value)
    vector = (c['release_version'], c['kernel_id']) == S2
    if vector:
        old_nb, old_meta, _, old, _, _ = package(registered(), diagnostic)
    else:
        old_nb, old_meta = original(diagnostic)
        old = before.parent.archive(''.join(old_nb['cells'][2]['source']))[2]
    nb, meta, new = copy.deepcopy(old_nb), copy.deepcopy(old_meta), dict(old)
    oc, cfg = notebook_config(old_nb), config(c, diagnostic)
    runtime = 'tools/k1sr_runtime_v9.py'
    if vector:
        new[runtime] = exact(old[runtime].decode(), 'REGISTERED = '+repr(registered()),
                             'REGISTERED = '+repr(c)).encode()
    else:
        del new['tools/k1sr_runtime_v7.py']
        source = (ROOT/runtime).read_text(encoding='utf-8')
        new[runtime] = exact(source,
            "REGISTERED = dict(config_schema.current(), k1_failure_policy='study_fallback_fail_loud')",
            'REGISTERED = '+repr(c)).encode()
        new['tools/k1n_config_v1.py'] = (ROOT/'tools/k1n_config_v1.py').read_bytes()
        new['tools/k1_shared_render_v2.py'] = builder.adapter(
            (ROOT/'tools/k1_shared_render_v2.py').read_bytes())
    port, release, codec = 'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py'
    if not vector:
        text = exact(old[port].decode(), 'k1sr_runtime_v7', 'k1sr_runtime_v9')
        text = exact(text, builder.STRIDE_ANCHOR, '"--eval-stride", "2"')
        new[port] = exact(text, 'VERSION = "1.0.1-r6"', 'VERSION = "1.0.2-hb228-r6s2"').encode()
        new[release] = exact(old[release].decode(), digest(old[port]), digest(new[port])).encode()
        text = exact(old[codec].decode(), digest(old[release]), digest(new[release]))
    else:
        text = old[codec].decode()
    util.need(text.count(oc['release_version']) == 2, 'codec identity anchors')
    text = text.replace(oc['release_version'], cfg['release_version'])
    new[codec] = exact(text, oc['kernel_id'], cfg['kernel_id']).encode()
    code = ''.join(old_nb['cells'][2]['source'])
    blob, pins, _ = before.parent.archive(code)
    new_blob = before.packed(new)
    code = exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(new_blob).decode()))
    code = exact(code, 'PINS = '+repr(pins), 'PINS = '+repr({n:digest(v) for n,v in new.items()}))
    code = exact(code, repr(digest(blob)), repr(digest(new_blob)))
    if not vector:
        code = exact(code, 'from tools import k1sr_runtime_v7 as _k1sr',
                     'from tools import k1sr_runtime_v9 as _k1sr')
    code = exact(code, repr(oc), repr(cfg))
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = ['# K1N runtime v9 '+cfg['release_version']+'\n', cfg['formula']+'\n',
        'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else 'CPU prepared; independent/GPU qualification pending.\n']
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1])
    changed = {n for n in old.keys() & new.keys() if old[n] != new[n]}
    util.need(changed == ({runtime, codec} if vector else
              {port, release, codec, 'tools/k1n_config_v1.py'}), 'v9 payload scope drift')
    util.need(new.keys()-old.keys() == (set() if vector else
              {runtime, 'tools/k1_shared_render_v2.py'}), 'v9 added scope drift')
    util.need(old.keys()-new.keys() == (set() if vector else
              {'tools/k1sr_runtime_v7.py'}), 'v9 removed scope drift')
    util.need(before.parent.archive(code)[2] == new, 'v9 archive roundtrip')
    util.need(nb['cells'][:1] == old_nb['cells'][:1] and nb['cells'][3:] == old_nb['cells'][3:],
              'v9 unrelated notebook cells drift')
    for name, data in new.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<v9 notebook>', 'exec')
    return nb, meta, old, new, old_nb, old_meta


def artifacts(value, diagnostic=False):
    from tools import k1n_stage_v1 as builder
    c = canonical(value)
    cfg = config(c, diagnostic)
    nb, meta, old, new, old_nb, old_meta = package(c, diagnostic)
    vector = (c['release_version'], c['kernel_id']) == S2
    parent_config = registered() if vector else schema.validate(
        json.loads((ROOT/'configs/k1n/current-x3.json').read_bytes()))
    parent_stage = builder.stage(parent_config, diagnostic)
    if vector and parent_stage.exists():
        prior, _ = artifacts(parent_config, diagnostic)
        util.need({p.relative_to(parent_stage).as_posix() for p in parent_stage.rglob('*')
                   if p.is_file()} == set(prior), 'S1 parent inventory drift')
        for name, data in prior.items():
            util.need((parent_stage/name).read_bytes() == data, 'S1 parent drift: '+name)
    diff = []
    for name in sorted(old.keys() | new.keys()):
        if old.get(name) != new.get(name):
            diff += difflib.unified_diff(old.get(name,b'').decode().splitlines(True),
                new.get(name,b'').decode().splitlines(True), fromfile='parent/'+name, tofile='v9/'+name)
    for name, left, right in [('notebook', old_nb, nb), ('metadata', old_meta, meta),
                              ('config', parent_config, c)]:
        diff += difflib.unified_diff(util.encoded(left).decode().splitlines(True),
            util.encoded(right).decode().splitlines(True), fromfile='parent/'+name, tofile='v9/'+name)
    fields = sorted(k for k in parent_config.keys() | c.keys() if parent_config.get(k) != c.get(k))
    scope = dict(status='PASS', parent_stage=parent_stage.relative_to(ROOT).as_posix(),
        changed=sorted(n for n in old.keys() & new.keys() if old[n] != new[n]),
        added=sorted(new.keys()-old.keys()), removed=sorted(old.keys()-new.keys()),
        untouched={n:digest(v) for n,v in old.items() if new.get(n) == v},
        config_changed_fields=fields,
        registered_change='alpha_vector v2 and identifiers' if vector else
            'runtime v9, fail-loud policy, guard8.6, R6 stride2 and identifiers',
        models_unchanged=True, k1_stride=3, drop_o=False,
        launch_pool_thread_budgets_unchanged=True,
        notebook_change='runtime import, archive/pins, registered config and markdown only',
        metadata_change='id/title only')
    outputs = {
        'kernel/release.ipynb': util.encoded(nb), 'kernel/kernel-metadata.json': util.encoded(meta),
        'config.json': util.encoded(c), 'payload-scope.json': util.encoded(scope),
        'full-payload-notebook.diff': ''.join(diff).encode(),
        'CHANGELOG.md': ('# Runtime v9.0.0 / generator v1.6.0\n\n'
            'Remove finish-only wall-clock failure; retain per-row completion and systemic guards.\n'
            +scope['registered_change']+'.\n'+cfg['formula']+'\n'
            'N32 is never-submit. Independent review and actual inference qualification pending.\n').encode(),
    }
    paths = ['tools/k1n_stage_v1.py', 'tools/k1n_stage_v9.py', 'tools/k1n_config_v1.py',
             'tools/k1sr_runtime_v9.py', 'tools/k1_shared_render_v2.py',
             'tools/k1n_replay_v150.py', 'tools/k1n_compat_v1.py']
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
