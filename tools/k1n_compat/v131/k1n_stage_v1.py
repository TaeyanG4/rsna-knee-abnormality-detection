"""K1 N-model stage generator v1.3.1, local CPU only.

Changelog v1.3.1: bind historical replay to exact registered identities;
unrelated configs with the same recipe keep their v1.2.0 runtime.
Changelog v1.3.0: registered R6 stride 1/2, frozen-source compatibility replay,
exact parent-relative stride-only diffs; no K1/runtime/scientific recipe drift.
Absent/explicit-1 R6 stride preserves all legacy generated artifact bytes.
"""
from __future__ import annotations

import argparse
import base64
import copy
import difflib
import json
from pathlib import Path

from tools import c85r7_r6w958k1x3sr_v1164 as before
from tools import k1n_config_v1 as schema
from tools import k1n_parent_v1 as parent_edit
from tools import k1n_compat_v1 as compat

VERSION = '1.3.1'
ROOT, util = before.ROOT, before.util
exact, digest = before.exact, before.digest
ADDED = {'tools/k1sr_runtime_v7.py', 'tools/k1n_config_v1.py', 'tools/k1_shared_render_v2.py'}
# Exact historical source profiles, not an inference-time recipe selector.
PARENTS = (
    ('configs/k1n/x3-stride2-drop-o.json', 'v110'),
    ('configs/k1n/x3-stride2-alpha-vector-v2.json', 'v120'),
    ('configs/k1n/x3-stride2-drop-o-alpha-vector-v2.json', 'v120'),
)
# Explicit successors requested by H-B-228; never infer lineage from recipe alone.
SUCCESSORS = {
    ('1.1.70-r6s2k1n3s2p25', 'taeyangg4/knee-lb-r6s2k1n3s2p25-v1'):
        'configs/k1n/x3-stride2-drop-o.json',
    ('1.1.70-r6s2k1n3s2av2', 'taeyangg4/knee-lb-r6s2k1n3s2av2-v1'):
        'configs/k1n/x3-stride2-alpha-vector-v2.json',
}
STRIDE_ANCHOR = '"--eval-stride", str(manifest.get("eval_stride", 3))'
ADAPTER_ANCHOR = "dataset.eval_stride) == (384, 'v4', 3, 3.5, 1)"


def legacy(value):
    c = schema.validate(value)
    c.pop('r6_eval_stride', None)
    return c


def recipe(value):
    return {k:v for k,v in legacy(value).items() if k not in ('release_version', 'kernel_id')}


def baseline(value):
    """Use the exact frozen parent for known recipes; v1.2.0 for other configs.

    Without stride2, only the historical release identity selects v1.1.0.
    For the two explicitly registered stride2 successors, require the entire
    parent recipe to match. Other identities retain the v1.2.0 generator.
    """
    c = legacy(value)
    for filename, profile in PARENTS:
        registered = legacy(json.loads((ROOT/filename).read_bytes()))
        successor = (value.get('r6_eval_stride', 1) == 2 and
                     SUCCESSORS.get((c['release_version'], c['kernel_id'])) == filename)
        if successor:
            util.need(recipe(c) == recipe(registered), 'registered successor recipe drift')
        if c == registered or successor:
            return compat.load(profile), registered
    return compat.load('v120'), c


def config(value, diagnostic=False):
    c = schema.validate(value)
    engine, _ = baseline(c)
    cfg = engine.config(legacy(c), diagnostic)
    if c.get('r6_eval_stride', 1) == 2:
        # K1 identity remains bound to its unchanged runtime config. The complete
        # stage config independently binds the new R6 choice.
        cfg.update(r6_eval_stride=2, r6_manifest_eval_stride=1,
                   shared_adapter_version='2.0.1-hb228-r6s2',
                   registered_config_sha256=digest(util.encoded(c)))
    return cfg


def stage(value, diagnostic=False):
    c = schema.validate(value)
    return before.parent.INF3.parent / ('v'+c['release_version']+('-n32' if diagnostic else ''))


def adapter(data):
    """Same two-line, fail-closed adapter change as H-B-228."""
    text = exact(data.decode(), ADAPTER_ANCHOR,
                 "dataset.eval_stride) == (384, 'v4', 3, 3.5, 2)")
    return exact(text, "VERSION = '2.0.0'", "VERSION = '2.0.1-hb228-r6s2'").encode()


def frozen_parent(engine, registered, diagnostic):
    """Regenerate, then compare every artifact against the immutable parent."""
    outputs, reg = engine.artifacts(registered, diagnostic)
    dest = engine.stage(registered, diagnostic)
    if dest.exists():
        util.need({p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()} == set(outputs),
                  'parent artifact inventory drift')
        for name, data in outputs.items():
            util.need((dest/name).read_bytes() == data, 'parent artifact drift: '+name)
    return outputs, reg


def package(value, diagnostic=False):
    c = schema.validate(value)
    engine, registered = baseline(c)
    if c.get('r6_eval_stride', 1) == 1:
        return engine.package(legacy(c), diagnostic)
    frozen_parent(engine, registered, diagnostic)
    # Full comparison uses the actual historical identity, not a recipe with
    # adjusted alpha/O/runtime. New identities affect metadata only.
    old_nb, old_meta, _, old, _, _ = engine.package(registered, diagnostic)
    nb, meta, _, files, _, _ = engine.package(legacy(c), diagnostic)
    files = dict(files)
    port, release, codec = 'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py'
    before_port = files[port]
    text = exact(before_port.decode(), STRIDE_ANCHOR, '"--eval-stride", "2"')
    files[port] = exact(text, 'VERSION = "1.0.1-r6"', 'VERSION = "1.0.2-hb228-r6s2"').encode()
    before_release = files[release]
    files[release] = exact(before_release.decode(), digest(before_port), digest(files[port])).encode()
    files[codec] = exact(files[codec].decode(), digest(before_release), digest(files[release])).encode()
    files['tools/k1_shared_render_v2.py'] = adapter(files['tools/k1_shared_render_v2.py'])
    code = ''.join(nb['cells'][2]['source'])
    blob, pins, _ = before.parent.archive(code)
    new_blob = before.packed(files)
    code = exact(code, repr(base64.b64encode(blob).decode()), repr(base64.b64encode(new_blob).decode()))
    code = exact(code, 'PINS = '+repr(pins), 'PINS = '+repr({n:digest(v) for n,v in files.items()}))
    code = exact(code, repr(digest(blob)), repr(digest(new_blob)))
    code = exact(code, repr(engine.config(legacy(c), diagnostic)), repr(config(c, diagnostic)))
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] += ['Registered R6 eval_stride=2; K1 raw-row capture and recipe unchanged.\n']
    # Fail closed on the full embedded diff, including the runtime literal.
    expected = dict(old)
    expected[port] = exact(exact(old[port].decode(), STRIDE_ANCHOR, '"--eval-stride", "2"'),
                          'VERSION = "1.0.1-r6"', 'VERSION = "1.0.2-hb228-r6s2"').encode()
    expected[release] = exact(old[release].decode(), digest(old[port]), digest(expected[port])).encode()
    text = exact(old[codec].decode(), digest(old[release]), digest(expected[release]))
    oc, nc = engine.config(registered, diagnostic), engine.config(legacy(c), diagnostic)
    util.need(text.count(oc['release_version']) == 2, 'codec identity anchors')
    text = text.replace(oc['release_version'], nc['release_version'])
    if oc['kernel_id'] != nc['kernel_id']:
        text = exact(text, oc['kernel_id'], nc['kernel_id'])
    expected[codec] = text.encode()
    expected['tools/k1_shared_render_v2.py'] = adapter(old['tools/k1_shared_render_v2.py'])
    if registered != legacy(c):
        expected['tools/k1sr_runtime_v7.py'] = exact(old['tools/k1sr_runtime_v7.py'].decode(),
            'REGISTERED = '+repr(registered), 'REGISTERED = '+repr(legacy(c))).encode()
    util.need(files == expected, 'R6 stride-only payload proof failed')
    util.need(meta == dict(old_meta, id=nc['kernel_id'], title=nc['kernel_id'].split('/')[1]),
              'R6 stride-only metadata proof failed')
    util.need(before.parent.archive(code)[2] == files, 'archive roundtrip')
    for name, data in files.items():
        if name.endswith('.py'): compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), '<k1n notebook>', 'exec')
    return nb, meta, old, files, old_nb, old_meta


def artifacts(value, diagnostic=False):
    c = schema.validate(value)
    engine, registered = baseline(c)
    if c.get('r6_eval_stride', 1) == 1:
        return engine.artifacts(legacy(c), diagnostic)
    cfg = config(c, diagnostic)
    nb, meta, old, new, old_nb, old_meta = package(c, diagnostic)
    diff = []
    for name in sorted(old.keys() | new.keys()):
        if old.get(name) != new.get(name):
            diff += difflib.unified_diff(old.get(name,b'').decode().splitlines(True),
                new.get(name,b'').decode().splitlines(True), fromfile='parent/'+name, tofile='r6s2/'+name)
    for name, left, right in [('notebook',old_nb,nb), ('metadata',old_meta,meta)]:
        diff += difflib.unified_diff(util.encoded(left).decode().splitlines(True),
            util.encoded(right).decode().splitlines(True), fromfile='parent/'+name, tofile='r6s2/'+name)
    scope = dict(status='PASS', parent_stage=str(engine.stage(registered,diagnostic).relative_to(ROOT)),
        parent_exists=engine.stage(registered,diagnostic).exists(),
        parent_builder_version=engine.VERSION,
        changed=sorted(n for n in old if old[n] != new[n]), added=[], removed=[],
        untouched={n:digest(v) for n,v in old.items() if new[n] == v},
        registered_change='R6 eval_stride 1 -> 2 only; identities, chained pins and receipt metadata',
        k1_raw_capture_unchanged=True, alpha_unchanged=True, drop_o_unchanged=True,
        runtime_only_registered_identity=True, schema_byte_identical=True,
        launch_import_pool_thread_timing_unchanged=True)
    outputs = {'kernel/release.ipynb':util.encoded(nb), 'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c), 'full-payload-notebook.diff':''.join(diff).encode(),
        'payload-scope.json':util.encoded(scope),
        'CHANGELOG.md': ('# K1N generator v1.3.1\n\nRegistered R6 eval_stride=2; K1 raw rows precede R6 windows.\n'+
            cfg['formula']+'\nParent alpha/O/runtime/schema preserved; no qualification claim.\nN32 is never-submit.\n').encode()}
    paths = ['tools/k1n_stage_v1.py','tools/k1n_config_v1.py','tools/k1n_parent_v1.py','tools/k1n_compat_v1.py']
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tools/k1n_compat').rglob('*.py')]
    reg = dict(schema='k1n-registration/v1', **cfg, builder_version=VERSION,
        builder_sources={n:util.sha(ROOT/n) for n in paths},
        stage=str(stage(c,diagnostic).relative_to(ROOT)), parent_stage=scope['parent_stage'],
        parent_builder_version=engine.VERSION, sources={n:digest(v) for n,v in new.items()},
        kernel_sha256=digest(outputs['kernel/release.ipynb']), metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        guard_seconds=28800, load_timeout_seconds=1500, call_timeout_seconds=120,
        call_timeout_scope='transfer + all classifiers per study; excludes loading',
        execution_verified=False, independent_review='REQUIRED', qualified_for_submission=False,
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg


def run(command, value):
    """Validate all bytes and both stride-only proofs before creating either stage."""
    c = schema.validate(value)
    pairs = [(d, stage(c,d), *artifacts(c,d)) for d in (False,True)]
    if command == 'prepare':
        util.need(all(not dest.exists() for _,dest,_,_ in pairs), 'immutable stage exists; use validate')
        for _, dest, outputs, _ in pairs:
            dest.mkdir()
            for name, data in outputs.items(): util.write_once(dest/name, data)
    else:
        util.need(command == 'validate', 'unknown command')
    result = []
    for diagnostic, dest, outputs, reg in pairs:
        util.need({p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()} == set(outputs),
                  'immutable artifact inventory drift')
        for name, data in outputs.items():
            util.need((dest/name).read_bytes() == data, 'immutable artifact drift: '+name)
        result.append(dict(stage=reg['stage'], kernel_id=reg['kernel_id'],
            kernel_sha256=reg['kernel_sha256'], never_submit=diagnostic))
    return dict(status='PREPARED_LOCAL_ONLY' if command == 'prepare' else 'PASS_CPU_VALIDATE',
        stages=result, qualified_for_submission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','validate'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.command, json.loads(args.config.read_bytes())), indent=2))


if __name__ == '__main__':
    main()
