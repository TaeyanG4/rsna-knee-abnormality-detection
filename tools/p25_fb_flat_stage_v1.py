"""F4 builder v1.0.0: the immutable P4 four-member stage with flat P weights.

Changelog 1.0.0: apply the proven P25F arithmetic/schema to P4; retain all
FB serving, launch, timing and fail-loud behavior. Isolated embedded checks
are mandatory before preparing either immutable stage.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import json

from tools import p25_fb_stage_v1 as parent
from tools import k1n_flat_p_v1 as flat
from tools import k1_embedded_selfcheck_v1 as selfcheck

VERSION = '1.0.0'
ROOT, util, exact, digest = parent.ROOT, parent.util, parent.exact, parent.digest
PARENT = parent.stage()
FLAT_PARENT = PARENT.parent/'v1.1.79-r6w958k1n3s2p25flatpv10'
RELEASE = '1.1.87-r6w958k1n4fbp25flatpv10'
SLUG = 'taeyangg4/knee-lb-r6w958k1n4fbp25flatpv10-v1'
SCRATCH = ROOT/'.local/scratch/astra-b1/f4-v1'
RUNTIME, CODEC = parent.RUNTIME, parent.CODEC
FB_ID, FB_SHA = parent.FB_ID, parent.FB_SHA
DATASET, DATASET_ID, seed0 = parent.DATASET, parent.DATASET_ID, parent.seed0
assets, verify_files = parent.assets, parent.verify_files
legacy = parent.parent.legacy


def configuration():
    return legacy.canonical(dict(parent.configuration(), release_version=RELEASE,
                                 kernel_id=SLUG, p_weight_mode='flat'))


def config(c, diagnostic=False):
    return legacy.config(c, diagnostic)


def stage(diagnostic=False):
    return PARENT.parent/('v'+RELEASE+('-n32' if diagnostic else ''))


def functions(data):
    return {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(data).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def package(diagnostic=False):
    outputs, reg = parent.artifacts(diagnostic)
    parent.verify_files(parent.stage(diagnostic), outputs)
    pc, c = parent.configuration(), configuration()
    oc, cfg = config(pc, diagnostic), config(c, diagnostic)
    old_nb = json.loads(outputs['kernel/release.ipynb'])
    old_meta = json.loads(outputs['kernel/kernel-metadata.json'])
    nb, meta = copy.deepcopy(old_nb), copy.deepcopy(old_meta)
    code = ''.join(old_nb['cells'][2]['source'])
    engine = parent.parent.v10.replay.load()
    blob, pins, old = engine.before.parent.archive(code)
    new = flat.patch_math(old, c['drop_o'])
    new[RUNTIME] = exact(old[RUNTIME].decode(), 'REGISTERED = '+repr(pc),
                         'REGISTERED = '+repr(c)).encode()
    fp = FLAT_PARENT.with_name(FLAT_PARENT.name+('-n32' if diagnostic else ''))
    fnb = parent.read(fp/'kernel/release.ipynb')
    freg = parent.read(fp/'registration.json')
    util.need(util.sha(fp/'kernel/release.ipynb') == freg['kernel_sha256'], 'P25F notebook binding')
    ff = selfcheck.payload(fnb)[0]
    util.need({n:digest(v) for n,v in ff.items()} == freg['sources'], 'P25F payload binding')
    fc = legacy.canonical(parent.read(fp/'config.json'))
    # Independently obtain the identical runtime by adding P4's exact FB patch
    # to the hidden-proven flat parent. No inference function changes vs P4.
    util.need(parent.runtime_patch(ff[RUNTIME], fc, c) == new[RUNTIME],
              'P25F plus exact FB runtime patch parity')
    new[flat.SCHEMA] = ff[flat.SCHEMA]
    util.need(new[flat.HELPER] == ff[flat.HELPER], 'P25F flat arithmetic byte parity')
    codec = new[CODEC].decode()
    util.need(codec.count(oc['release_version']) == 2, 'codec identity anchors')
    new[CODEC] = exact(codec.replace(oc['release_version'],cfg['release_version']),
                        oc['kernel_id'],cfg['kernel_id']).encode()
    fcodec = ff[CODEC].decode()
    util.need(fcodec.count(freg['release_version']) == 2, 'P25F codec identity anchors')
    util.need(exact(fcodec.replace(freg['release_version'],cfg['release_version']),
                    freg['kernel_id'],cfg['kernel_id']).encode() == new[CODEC],
              'P25F codec identical apart from identity')
    util.need(set(ff) == set(new) and all(ff[n] == new[n] for n in ff
              if n not in (RUNTIME,CODEC)), 'P25F payload outside FB/identity unchanged')
    packed = engine.before.packed(new)
    replacements = [
        (repr(base64.b64encode(blob).decode()),repr(base64.b64encode(packed).decode())),
        ('PINS = '+repr(pins),'PINS = '+repr({n:digest(v) for n,v in new.items()})),
        (repr(digest(blob)),repr(digest(packed))), (repr(oc),repr(cfg))]
    for before, after in replacements: code = exact(code,before,after)
    restored = code
    for before, after in reversed(replacements): restored = exact(restored,after,before)
    util.need(restored == ''.join(old_nb['cells'][2]['source']), 'P4 notebook launch drift')
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = [s.replace(oc['release_version'],cfg['release_version'])
                               .replace(oc['formula'],cfg['formula'])
                               for s in nb['cells'][1]['source']]
    meta.update(id=cfg['kernel_id'],title=cfg['kernel_id'].split('/')[1])
    util.need(engine.before.parent.archive(code)[2] == new,'archive roundtrip')
    for name,data in new.items():
        if name.endswith('.py'): compile(data,name,'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']),'<F4 notebook>','exec')
    return outputs,reg,pc,c,cfg,old_nb,nb,old_meta,meta,old,new,ff,fnb,fp


def artifacts(diagnostic=False):
    po,pr,pc,c,cfg,onb,nb,om,meta,old,new,ff,fnb,fp = package(diagnostic)
    fields = [dict(field=k,parent=pc.get(k),candidate=c.get(k))
              for k in sorted(pc.keys()|c.keys()) if pc.get(k) != c.get(k)]
    util.need({f['field'] for f in fields} == {'release_version','kernel_id','p_weight_mode'},
              'F4 config scope')
    changed = sorted(n for n in old if old[n] != new[n])
    util.need(set(old) == set(new) and changed == sorted([flat.HELPER,CODEC,RUNTIME,flat.SCHEMA]),
              'F4 payload scope')
    util.need(functions(old[RUNTIME]) == functions(new[RUNTIME]),'P4 runtime function drift')
    util.need(meta == dict(om,id=cfg['kernel_id'],title=cfg['kernel_id'].split('/')[1]),'metadata scope')
    util.need(nb['cells'][:1] == onb['cells'][:1] and nb['cells'][3:] == onb['cells'][3:],
              'unrelated notebook cells drift')
    scope = dict(status='PASS_CPU_SCOPE',parent_stage=parent.stage(diagnostic).relative_to(ROOT).as_posix(),
        changed=changed,added=[],removed=[],untouched={n:digest(v) for n,v in old.items() if v==new[n]},
        config_changed_fields=[f['field'] for f in fields],metadata_changed_fields=['id','title'],
        scientific_change=dict(p_weight_mode='flat',p_raptor_weights=list(flat.FLAT),
                               changed_target_indices=list(flat.CHANGED)),
        k1_functions_identical=True,parent_launch_pool_threads_timing_identical=True,
        parent_artifacts_byte_identical=True,flat_schema_source=fp.relative_to(ROOT).as_posix(),
        flat_schema_byte_identical=True,flat_runtime_plus_exact_fb_patch=True,
        vs_flat_changed=[n for n in ff if ff[n]!=new[n]],
        vs_flat_untouched={n:digest(v) for n,v in ff.items() if v==new[n]})
    outputs = {'kernel/release.ipynb':util.encoded(nb),'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c),'payload-scope.json':util.encoded(scope),
        'field-level-diff.json':util.encoded(fields),
        'CHANGELOG.md':('# F4 v1.1.87 / builder v1.0.0\n\n'
            'P4 with only flat P weights: all twelve .5 (indices 5,6,8,9 change).\n'
            'Embed the authenticated successful P25F schema accepting p_weight_mode.\n'
            'P25F plus the exact pinned fourth FB member; no O/A5.\n'
            'Final = rank(.75 rank(P_flat) + .25 rank(mean of four per-target K1 ranks)).\n'
            'Stride2, runtime v10, guard8.8h, failure fraction .05 and launch unchanged.\n'
            'Independent review and exact-candidate visible/N32 qualification pending.\n'
            'N32 diagnostic NEVER SUBMIT.\n').encode()}
    for label,source,snb,smeta,sc in [('P4',old,onb,om,pc),
            ('P25F',ff,fnb,parent.read(fp/'kernel/kernel-metadata.json'),parent.read(fp/'config.json'))]:
        pairs = [(n,source[n],new[n]) for n in source if source[n]!=new[n]]
        pairs += [(n,util.encoded(a),util.encoded(z)) for n,a,z in
                  [('notebook',snb,nb),('metadata',smeta,meta),('config',sc,c)]]
        outputs['full-diff-vs-'+label+'.patch'] = ''.join(line for name,a,z in pairs
            for line in difflib.unified_diff(a.decode().splitlines(True),z.decode().splitlines(True),
                                            fromfile=label+'/'+name,tofile='F4/'+name)).encode()
    reg = dict(pr)
    reg.update(cfg,builder_version=VERSION,builder='tools/p25_fb_flat_stage_v1.py',
        stage=stage(diagnostic).relative_to(ROOT).as_posix(),parent_stage=parent.stage(diagnostic).relative_to(ROOT).as_posix(),
        parent_registration_sha256=digest(po['registration.json']),
        flat_parent_registration_sha256=util.sha(fp/'registration.json'),
        sources={n:digest(v) for n,v in new.items()},
        builder_sources={**pr['builder_sources'],**{n:util.sha(ROOT/n) for n in
            ['tools/p25_fb_flat_stage_v1.py','tools/k1n_flat_p_v1.py','tools/k1_embedded_selfcheck_v1.py']}},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        independent_review='REQUIRED',execution_verified=False,qualified_for_submission=False)
    outputs['registration.json'] = util.encoded(reg)
    return outputs,reg


def checks(pairs, controls=False):
    selfcheck.SCRATCH = SCRATCH/'embedded-selfcheck'
    results = [selfcheck.check(json.loads(out['kernel/release.ipynb']),path.name) for path,out in pairs]
    if controls:
        for base in (PARENT,FLAT_PARENT):
            for d in (False,True):
                path = base.with_name(base.name+('-n32' if d else ''))
                results.append(selfcheck.check(parent.read(path/'kernel/release.ipynb'),path.name))
    return results


def run(command,dry_run=False):
    pairs = [(stage(d),*artifacts(d)) for d in (False,True)]
    if command == 'selfcheck':
        return dict(status='PASS_CPU_SELFCHECK',checks=checks([(p,o) for p,o,r in pairs],True))
    if command == 'prepare' and not dry_run:
        util.need(all(not p.exists() for p,o,r in pairs),'immutable stage exists; validate instead')
        checks([(p,o) for p,o,r in pairs])
        for p,o,r in pairs:
            for name,data in o.items(): util.write_once(p/name,data)
    if command == 'validate' or (command == 'prepare' and not dry_run):
        for p,o,r in pairs: verify_files(p,o)
    result = dict(status='PASS_CPU_'+command.upper(),dry_run=dry_run,qualified_for_submission=False,
        stages=[dict(stage=p.relative_to(ROOT).as_posix(),kernel_id=r['kernel_id'],
                     kernel_sha256=r['kernel_sha256'],never_submit=r['never_submit']) for p,o,r in pairs])
    if command == 'admit':
        util.need(dry_run,'use p25_fb_admit_v1 for actual receipt admission')
        result.update(status='HOLD_NOT_ADMITTED',reasons=[
            'Independent review and exact F4 visible/N32 execution required; parent receipts cannot qualify F4'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','prepare','validate','selfcheck','admit'])
    p.add_argument('--dry-run',action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.command,a.dry_run),indent=2))


if __name__ == '__main__': main()
