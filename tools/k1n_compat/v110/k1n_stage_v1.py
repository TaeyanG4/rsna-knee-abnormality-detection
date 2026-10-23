"""K1 N-model stage generator v1.1.0, local CPU only.

Changelog v1.1.0: register stride 2/3 and embed v2 shared-row adapter.
Changelog: config-bound runtime v7, optional exact INF3 P parent, paired N32.
prepare creates new immutable stages; validate compares every artifact byte.
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

VERSION = '1.1.0'
ROOT, util = before.ROOT, before.util
exact, digest = before.exact, before.digest
ADDED = {'tools/k1sr_runtime_v7.py', 'tools/k1n_config_v1.py', 'tools/k1_shared_render_v2.py'}


def config(value, diagnostic=False):
    c = schema.validate(value)
    parent = 'INF3_P_R224_float32_csv' if c['drop_o'] else 'INF3_float32_csv'
    a = c['alpha']
    return dict(variant='k1n', release_version=c['release_version']+('-n32' if diagnostic else ''),
        kernel_id=c['kernel_id']+('-n32' if diagnostic else ''),
        diagnostic=diagnostic, never_submit=diagnostic, drop_o=c['drop_o'],
        parent_weight=1-a, k1_weight=a, r224_weight=.35, encoder_chunk=c['encoder_chunk'],
        k1_eval_stride=c['k1_eval_stride'],
        input_transfer='once per study', k1_models=[m['id'] for m in c['models']],
        models=c['models'], config_sha256=digest(util.encoded(c)),
        model_sha256={m['id']: m['checkpoint_sha256'] for m in c['models']},
        asset_manifest_sha256={m['id']: m['manifest_sha256'] for m in c['models']},
        formula=f'rank({1-a:g}*rank({parent})+{a:g}*rank(mean(per-target rank(K1_model_FP16)))); '
                'average ties/full roster; model probabilities float32; final float32',
        contract=dict(before.config('k1x3sr20')['contract'], eval_stride=c['k1_eval_stride']))


def stage(value, diagnostic=False):
    return before.parent.INF3.parent / ('v'+config(value, diagnostic)['release_version'])


def package(value, diagnostic=False):
    c = schema.validate(value)
    cfg = config(c, diagnostic)
    nb, meta, _, old = before.package('k1x3sr20', diagnostic)
    old_nb, old_meta = copy.deepcopy(nb), copy.deepcopy(meta)
    # Bind the actual immutable v6 predecessor, not only a regenerated approximation.
    previous = before.stage('k1x3sr20', diagnostic)
    util.need((previous/'kernel/release.ipynb').read_bytes() == util.encoded(nb), 'v6 notebook drift')
    util.need((previous/'kernel/kernel-metadata.json').read_bytes() == util.encoded(meta), 'v6 metadata drift')
    new = dict(old)
    del new['tools/k1sr_runtime_v6.py']
    source = (ROOT/'tools/k1sr_runtime_v7.py').read_text(encoding='utf-8')
    source = exact(source, 'REGISTERED = config_schema.current()', 'REGISTERED = '+repr(c))
    new['tools/k1sr_runtime_v7.py'] = source.encode()
    new['tools/k1n_config_v1.py'] = (ROOT/'tools/k1n_config_v1.py').read_bytes()
    new['tools/k1_shared_render_v2.py'] = (ROOT/'tools/k1_shared_render_v2.py').read_bytes()
    port, release, codec = 'exp169_r7_port.py', 'exp169_release_c85r7.py', 'exp169_release_c85r7codec.py'
    new[port] = exact(old[port].decode(), 'k1sr_runtime_v6', 'k1sr_runtime_v7').encode()
    release_source = exact(old[release].decode(), digest(old[port]), digest(new[port]))
    if c['drop_o']:
        release_source = parent_edit.drop_preflight(release_source)
    new[release] = release_source.encode()
    codec_source = exact(old[codec].decode(), digest(old[release]), digest(new[release]))
    oc = before.config('k1x3sr20', diagnostic)
    # Version occurs in VERSION and OUT; both are registered metadata only.
    util.need(codec_source.count(oc['release_version']) == 2, 'codec version anchors')
    codec_source = codec_source.replace(oc['release_version'], cfg['release_version'])
    codec_source = exact(codec_source, oc['kernel_id'], cfg['kernel_id'])
    if c['drop_o']:
        codec_source = parent_edit.drop_codec(codec_source)
    new[codec] = codec_source.encode()
    blob = before.packed(new)
    code = ''.join(nb['cells'][2]['source'])
    old_blob, old_pins, _ = before.parent.archive(code)
    code = exact(code, repr(base64.b64encode(old_blob).decode()), repr(base64.b64encode(blob).decode()))
    code = exact(code, 'PINS = '+repr(old_pins), 'PINS = '+repr({n:digest(v) for n,v in new.items()}))
    code = exact(code, repr(digest(old_blob)), repr(digest(blob)))
    code = exact(code, 'from tools import k1sr_runtime_v6 as _k1sr', 'from tools import k1sr_runtime_v7 as _k1sr')
    code = exact(code, repr(oc), repr(cfg))
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = ['# K1 N-model '+cfg['release_version']+'\n', cfg['formula']+'\n',
        'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else 'CPU prepared; independent/GPU qualification pending.\n']
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1])
    old_k1 = {m['dataset_slug'] for m in schema.current()['models']}
    meta['dataset_sources'] = [s for s in meta['dataset_sources'] if s not in old_k1]
    if c['drop_o']:
        # O asset mounts are absent from preflight/run; retain public code/anatomy datasets.
        inf3_meta = json.loads((before.parent.INF3/'kernel/kernel-metadata.json').read_bytes())
        meta['dataset_sources'] = [s for s in meta['dataset_sources'] if s not in inf3_meta['dataset_sources'][:2]]
    meta['dataset_sources'] += [m['dataset_slug'] for m in c['models']]
    util.need(len(meta['dataset_sources']) == len(set(meta['dataset_sources'])), 'duplicate dataset mount')
    util.need(meta['is_private'] and not meta['enable_internet'], 'private/offline policy')
    allowed = {port, release, codec}
    changed = {n for n in old.keys() & new.keys() if old[n] != new[n]}
    util.need(changed == allowed and new.keys()-old.keys() == ADDED and
              old.keys()-new.keys() == {'tools/k1sr_runtime_v6.py'}, 'payload scope drift')
    for member, data in new.items():
        if member.endswith('.py'): compile(data, member, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), '<k1n notebook>', 'exec')
    util.need(before.parent.archive(code)[2] == new, 'archive verification')
    return nb, meta, old, new, old_nb, old_meta


def artifacts(value, diagnostic=False):
    c = schema.validate(value)
    cfg = config(c, diagnostic)
    nb, meta, old, new, old_nb, old_meta = package(c, diagnostic)
    diff = []
    for name in sorted(old.keys() | new.keys()):
        if old.get(name) != new.get(name):
            diff += difflib.unified_diff(old.get(name,b'').decode().splitlines(True),
                new.get(name,b'').decode().splitlines(True), fromfile='v6/'+name, tofile='v7/'+name)
    for name, left, right in [('notebook',old_nb,nb), ('metadata',old_meta,meta)]:
        diff += difflib.unified_diff(util.encoded(left).decode().splitlines(True),
            util.encoded(right).decode().splitlines(True), fromfile='v6/'+name, tofile='v7/'+name)
    scope = dict(status='PASS', parent_stage=str(before.stage('k1x3sr20',diagnostic).relative_to(ROOT)),
        changed=sorted(n for n in old.keys() & new.keys() if old[n] != new[n]),
        added=sorted(new.keys()-old.keys()), removed=sorted(old.keys()-new.keys()),
        untouched={n:digest(old[n]) for n in old.keys() & new.keys() if old[n] == new[n]},
        drop_o=c['drop_o'], r224_unchanged=True, public_inference_unchanged=True,
        notebook_change='runtime import, archive pins, config literal and markdown only',
        metadata_change='id/title; configured K1 mounts; O mounts removed only for drop_o')
    outputs = {'kernel/release.ipynb':util.encoded(nb), 'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c), 'full-payload-notebook.diff':''.join(diff).encode(),
        'payload-scope.json':util.encoded(scope),
        'CHANGELOG.md': ('# K1 N-model runtime v7.1 / generator v1.1.0\n\n'+cfg['formula']+
            '\nRegistered K1 eval_stride='+str(c['k1_eval_stride'])+'; raw-row windows via adapter v2.\n'+
            '\nOne input transfer; chunk32; 1500s/model load; 120s/study classification; 8h K1 guard.\n'
            'Any K1 failure preserves exact selected-parent bytes. N32 is never-submit.\n').encode()}
    reg = dict(schema='k1n-registration/v1', **cfg, builder_version=VERSION,
        builder_sources={n:util.sha(ROOT/n) for n in ('tools/k1n_stage_v1.py','tools/k1n_parent_v1.py','tools/k1n_config_v1.py')},
        stage=str(stage(c,diagnostic).relative_to(ROOT)), sources={n:digest(v) for n,v in new.items()},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),
        metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        guard_seconds=28800, load_timeout_seconds=1500, call_timeout_seconds=120,
        call_timeout_scope='transfer + all classifiers per study; excludes loading',
        execution_verified=False, independent_review='REQUIRED', qualified_for_submission=False,
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg


def run(command, value):
    """Validate everything before creating either member of the immutable pair."""
    c = schema.validate(value)
    pairs = [(d, stage(c,d), *artifacts(c,d)) for d in (False,True)]
    if command == 'prepare':
        util.need(all(not dest.exists() for _,dest,_,_ in pairs), 'immutable stage exists; use validate')
        for _, dest, outputs, _ in pairs:
            dest.mkdir()  # exclusive; failure leaves an inspectable partial stage
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
