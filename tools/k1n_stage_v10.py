"""v1.0.0: runtime-v10 timeout-only byte-patch family over pinned v9.

Changelog: D-B-261 capture 900 s; transfer/classifiers 300 s. Preserve all
scientific math, study-local fallback, 5% fail-loud, guard and load timeout.
"""
import base64
import copy
import difflib
import json
import re

from tools import k1n_config_v1 as schema
from tools import k1n_replay_v181 as replay

ROOT, util, exact = replay.ROOT, replay.util, replay.exact
VERSION = '1.0.0'
RUNTIME9 = 'tools/k1sr_runtime_v9.py'
RUNTIME10 = 'tools/k1sr_runtime_v10.py'
SCHEMA = 'tools/k1n_config_v1.py'
TIMEOUT_FORMULA = ('; k1_runtime_version=10; capture_timeout_seconds=900; '
                   'call_timeout_seconds=300 (transfer + all classifiers per study); '
                   'load_timeout_seconds=1500; all deadlines bounded by remaining guard')
# Existing immutable parents. New names never select a different recipe.
FAMILY = {
    'b5-flat-r6s2-v10': 'b5-flat-r6s2-v9',
    'c5-dropo-p25-flatp-v10': 'c5-dropo-p25-flatp',
    'a7-bal-flat-r6s2-v10': 'a7-bal-flat-r6s2-v9',
    'c5-dropo-p25-v10': 'c5-dropo-p25-v9',
}


def parent_config(value):
    c = schema.validate(value)
    util.need(c.get('k1_runtime_version') == 10, 'runtime v10 required')
    probe = dict(c, k1_runtime_version=9)
    for name in FAMILY.values():
        pc = json.loads((ROOT/'configs/k1n'/f'{name}.json').read_bytes())
        suffix = pc['release_version'].split('-', 1)[1]
        release = '1.1.78-'+suffix+'10'
        slug = pc['kernel_id'].removesuffix('-v1')+'10-v1'
        if c['release_version'] == release or c['kernel_id'] == slug:
            util.need(c['release_version'] == release and c['kernel_id'] == slug,
                      'v10 family identifiers must match')
            engine = replay.load()
            probe.update(release_version=pc['release_version'], kernel_id=pc['kernel_id'])
            util.need(engine.canonical(probe) == engine.canonical(pc), 'v10 parent recipe drift')
            return engine.canonical(pc)
    # Future filled templates use the same registered v9 recipe at a new identity.
    # Their hypothetical v9 construction is explicitly not an existing parent stage.
    return replay.load().canonical(probe)


def canonical(value):
    pc = parent_config(value)
    return dict(pc, release_version=value['release_version'], kernel_id=value['kernel_id'],
                k1_runtime_version=10)


def config(value, diagnostic=False):
    c = canonical(value)
    engine = replay.load()
    cfg = engine.config(parent_config(c), diagnostic)
    cfg.update(release_version=c['release_version']+('-n32' if diagnostic else ''),
        kernel_id=c['kernel_id']+('-n32' if diagnostic else ''), k1_runtime_version=10,
        config_sha256=engine.digest(util.encoded(c)),
        registered_config_sha256=engine.digest(util.encoded(c)))
    cfg['formula'] += TIMEOUT_FORMULA
    return cfg


def runtime_delta(source):
    """Exact timeout/version changes; keep the optional weighted-mean hunk intact."""
    text = source.decode()
    version = '9.1.0' if "VERSION = '9.1.0'" in text else '9.0.0'
    new_version = '10.1.0' if version == '9.1.0' else '10.0.0'
    for a,z in [
        ('runtime v'+version+'.', 'runtime v'+new_version+'.\n\n'
         'Changelog v10.0.0: separate capture 900 s and transfer/classifier 300 s deadlines.'),
        ("VERSION = '"+version+"'", "VERSION = '"+new_version+"'"),
        ('CALL_TIMEOUT_SECONDS = 120', 'CAPTURE_TIMEOUT_SECONDS = 900\nCALL_TIMEOUT_SECONDS = 300'),
        ('with deadline(min(remaining(), CALL_TIMEOUT_SECONDS)):',
         'with deadline(min(remaining(), CAPTURE_TIMEOUT_SECONDS)):'),
        ('tools.k1sr_runtime_v9 import', 'tools.k1sr_runtime_v10 import'),
        ("schema='k1-shared-r6-release/v9'", "schema='k1-shared-r6-release/v10'"),
        ('fp16=True, guard_seconds=GUARD_SECONDS, call_timeout_seconds=CALL_TIMEOUT_SECONDS,',
         'fp16=True, guard_seconds=GUARD_SECONDS, call_timeout_seconds=CALL_TIMEOUT_SECONDS,\n'
         '        capture_timeout_seconds=CAPTURE_TIMEOUT_SECONDS,'),
    ]:
        text = exact(text, a, z)
    return text.encode()


def embedded_schema(data):
    """Only extend the parent's runtime selector; preserve its remaining schema."""
    text = data.decode()
    text = text.replace("c.get('k1_runtime_version') == 9", "c.get('k1_runtime_version') in (9, 10)")
    text = exact(text, "c['k1_runtime_version'] == 9", "c['k1_runtime_version'] in (9, 10)")
    text = text.replace('runtime v9', 'runtime v9 or v10').replace(
        'k1_runtime_version must be 9', 'k1_runtime_version must be 9 or 10')
    text = re.sub(r"VERSION = '[0-9.]+?'", "VERSION = '1.8.2'", text, count=1)
    text = text.replace('\n\n', '\n\nChangelog v1.8.2: accept runtime v10 timeout-only successor.\n\n', 1)
    return text.encode()


def package(value, diagnostic=False):
    c = canonical(value)
    pc, engine = parent_config(c), replay.load()
    old_nb, old_meta, _, old, _, _ = engine.package(pc, diagnostic)
    nb, meta, new = copy.deepcopy(old_nb), copy.deepcopy(old_meta), dict(old)
    oc, cfg = engine.config(pc, diagnostic), config(c, diagnostic)
    runtime = exact(old[RUNTIME9].decode(), 'REGISTERED = '+repr(pc), 'REGISTERED = '+repr(c))
    new[RUNTIME10] = runtime_delta(runtime.encode())
    del new[RUNTIME9]
    new[SCHEMA] = embedded_schema(old[SCHEMA])
    port, release, codec = 'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py'
    new[port] = exact(old[port].decode(), 'k1sr_runtime_v9', 'k1sr_runtime_v10').encode()
    new[release] = exact(old[release].decode(), engine.digest(old[port]), engine.digest(new[port])).encode()
    text = exact(old[codec].decode(), engine.digest(old[release]), engine.digest(new[release]))
    if oc['release_version'] != cfg['release_version']:
        util.need(text.count(oc['release_version']) == 2, 'v10 codec version anchors')
        text = text.replace(oc['release_version'], cfg['release_version'])
        text = exact(text, oc['kernel_id'], cfg['kernel_id'])
    new[codec] = text.encode()
    code = ''.join(old_nb['cells'][2]['source'])
    blob, pins, _ = engine.before.parent.archive(code)
    packed = engine.before.packed(new)
    code = exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(packed).decode()))
    code = exact(code, 'PINS = '+repr(pins), 'PINS = '+repr({n:engine.digest(v) for n,v in new.items()}))
    code = exact(code, repr(engine.digest(blob)), repr(engine.digest(packed)))
    code = exact(code, 'from tools import k1sr_runtime_v9 as _k1sr',
                 'from tools import k1sr_runtime_v10 as _k1sr')
    code = exact(code, repr(oc), repr(cfg))
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = ['# K1N runtime v10 '+cfg['release_version']+'\n', cfg['formula']+'\n',
        'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else
        'CPU prepared; independent review and actual inference qualification pending.\n']
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1])
    util.need({n for n in old.keys() & new.keys() if old[n] != new[n]} ==
              {port, release, codec, SCHEMA}, 'v10 payload scope drift')
    util.need(new.keys()-old.keys() == {RUNTIME10} and old.keys()-new.keys() == {RUNTIME9},
              'v10 runtime replacement scope drift')
    util.need(engine.before.parent.archive(code)[2] == new, 'v10 archive roundtrip')
    util.need(nb['cells'][:1] == old_nb['cells'][:1] and nb['cells'][3:] == old_nb['cells'][3:],
              'v10 unrelated notebook cells drift')
    for name, data in new.items():
        if name.endswith('.py'):
            compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<v10 notebook>', 'exec')
    return nb, meta, old, new, old_nb, old_meta


def artifacts(value, diagnostic=False):
    from tools import k1n_stage_v1 as builder
    c, engine = canonical(value), replay.load()
    pc = parent_config(c)
    parent_outputs, parent_reg = engine.artifacts(pc, diagnostic)
    parent_stage = engine.stage(pc, diagnostic)
    # A materialized declared parent must match the frozen generator byte-for-byte.
    if parent_stage != builder.stage(c, diagnostic) and parent_stage.exists():
        util.need({p.relative_to(parent_stage).as_posix() for p in parent_stage.rglob('*')
                   if p.is_file()} == set(parent_outputs), 'v10 parent inventory drift')
        for name, data in parent_outputs.items():
            util.need((parent_stage/name).read_bytes() == data, 'v10 parent drift: '+name)
    nb, meta, old, new, old_nb, old_meta = package(c, diagnostic)
    cfg = config(c, diagnostic)
    fields = sorted(k for k in pc.keys() | c.keys() if pc.get(k) != c.get(k))
    util.need(set(fields) <= {'k1_runtime_version','release_version','kernel_id'}, 'v10 config drift')
    scope = dict(status='PASS', parent_stage=parent_stage.relative_to(ROOT).as_posix(),
        parent_is_existing_stage=parent_stage != builder.stage(c, diagnostic) and parent_stage.exists(),
        changed=sorted(n for n in old.keys() & new.keys() if old[n] != new[n]),
        added=[RUNTIME10], removed=[RUNTIME9],
        untouched={n:engine.digest(v) for n,v in old.items() if new.get(n) == v},
        config_changed_fields=fields, metadata_changed_fields=['id','title'],
        registered_change='capture 120 -> 900 s; transfer/classifiers 120 -> 300 s; runtime10 and identity',
        models_weights_ranks_stride_failure_policy_guard_load_unchanged=True,
        launch_pool_thread_budgets_unchanged=True,
        notebook_change='runtime import, archive/pins, config binding and markdown only')
    pairs = [(n,old.get(n,b''),new.get(n,b'')) for n in sorted(old.keys() | new.keys())
             if old.get(n) != new.get(n)]
    # Aligned runtime diff avoids hiding a timeout change in a delete/add diff.
    pairs += [('runtime-v9-to-v10',old[RUNTIME9],new[RUNTIME10])]
    pairs += [(n,util.encoded(a),util.encoded(z)) for n,a,z in (
        ('notebook',old_nb,nb), ('metadata',old_meta,meta), ('config',pc,c))]
    diff = []
    for name, left, right in pairs:
        diff += difflib.unified_diff(left.decode().splitlines(True), right.decode().splitlines(True),
                                    fromfile='v9/'+name, tofile='v10/'+name)
    outputs = {'kernel/release.ipynb':util.encoded(nb), 'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c), 'payload-scope.json':util.encoded(scope),
        'full-payload-notebook.diff':''.join(diff).encode(),
        'CHANGELOG.md':('# Generator v1.8.2 / runtime v10\n\n'+scope['registered_change']+'.\n'
            +cfg['formula']+'\nN32 NEVER SUBMIT. Independent review and inference qualification pending.\n').encode()}
    if 'runtime-projection.json' in parent_outputs:
        outputs['runtime-projection.json'] = parent_outputs['runtime-projection.json']
    paths = ['tools/k1n_stage_v1.py', SCHEMA, 'tools/k1n_stage_v10.py', RUNTIME10,
             'tools/k1n_replay_v181.py']+['tools/k1n_frozen_v181/'+n for n in replay.PINS]
    reg = dict(schema='k1n-registration/v1', **cfg, builder_version=builder.VERSION,
        builder_sources={n:util.sha(ROOT/n) for n in paths},
        parent_registration_sha256=engine.digest(util.encoded(parent_reg)),
        stage=builder.stage(c,diagnostic).relative_to(ROOT).as_posix(),
        parent_stage=scope['parent_stage'], parent_is_existing_stage=scope['parent_is_existing_stage'],
        sources={n:engine.digest(v) for n,v in new.items()},
        kernel_sha256=engine.digest(outputs['kernel/release.ipynb']),
        metadata_sha256=engine.digest(outputs['kernel/kernel-metadata.json']),
        guard_seconds=c['k1_guard_hours']*3600, load_timeout_seconds=1500,
        capture_timeout_seconds=900, call_timeout_seconds=300,
        call_timeout_scope='transfer + all classifiers per study; excludes loading',
        execution_verified=False, independent_review='REQUIRED', qualified_for_submission=False,
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg
