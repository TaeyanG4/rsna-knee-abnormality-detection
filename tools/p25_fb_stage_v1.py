"""P4 builder v1.0.0: immutable P25-v10 plus one full-bag K1 member.

Changelog: materialized v1.1.79 parent; four equal K1 members; FB asset pins.
No launch, pooling, timing, precision, renderer or P arithmetic changes.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import json
from pathlib import Path

from tools import p25_a5_stage_v1 as parent
from tools import k1_embedded_selfcheck_v1 as selfcheck
from tools import k1x3sr_assets_v1 as seed0

VERSION = '1.0.0'
ROOT, util, exact, digest = parent.ROOT, parent.util, parent.exact, parent.digest
PARENT = parent.PARENT
RELEASE = '1.1.85-r6w958k1n4fbp25v10'
SLUG = 'taeyangg4/knee-lb-r6w958k1n4fbp25v10-v1'
SCRATCH = ROOT / '.local/scratch/astra-b1/p4-v1'
DATASET = SCRATCH / 'dataset'
DATASET_ID = 'taeyangg4/knee-k1-fb-v1-assets'
FB_ID = 'k1_v3g_fullbag'
EXPORT = ROOT / '.local/scratch/astra-b2/h224-device-plan-v1/exports/fb'
CHECKPOINT = EXPORT / 'fit/models/E_prior_s0_f0.pt'
FB_SHA = 'ef3f42d76f6d2d57e6bda2c225b051e56652bcc8c07281470792907d3ec6ea98'
RUNTIME, CODEC = parent.RUNTIME, parent.CODEC
MANIFEST = dict(schema='k1-fb-assets/v1', version=VERSION,
    files={'weights/E_prior_s0_f0.pt': FB_SHA, 'source/compact_train.py': seed0.SOURCE_SHA},
    model_seed=0, model_fold=0, provenance='D-B-264; H224 full-bag fold-0 Vast export',
    training_source_sha256='3b0b1e4c7d093c5feacf92e6981da084c08f0f514241a68bf828b18d53cac15b',
    serving_architecture=seed0.MANIFEST['serving_architecture'],
    architecture_evidence='strict serving state-dict keys/shapes/dtypes and real-study CPU forward; see build receipt',
    clean_oof=False, exposure='EXPOSED_EXPLORATORY')
MANIFEST_SHA = digest(util.encoded(MANIFEST))
SUMS = ''.join(f'{pin}  {name}\n' for name, pin in sorted(MANIFEST['files'].items())).encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def asset_outputs():
    util.need(util.sha(CHECKPOINT) == FB_SHA, 'FB checkpoint pin')
    util.need(read(EXPORT/'EXPORT.json')['sha256']['fit/models/E_prior_s0_f0.pt'] == FB_SHA,
              'FB export binding')
    util.need(util.sha(seed0.SERVING) == seed0.SOURCE_SHA, 'frozen serving source pin')
    return {'weights/E_prior_s0_f0.pt': CHECKPOINT.read_bytes(),
        'source/compact_train.py': seed0.SERVING.read_bytes(),
        'asset-manifest.json': util.encoded(MANIFEST), 'SHA256SUMS': SUMS,
        'dataset-metadata.json': util.encoded(dict(id=DATASET_ID,
            title=DATASET_ID.split('/')[1], isPrivate=True, licenses=[dict(name='other')],
            description='Private full-bag fold-0 K1 deploy checkpoint and frozen serving source. D-B-264. No data, labels or predictions. EXPOSED_EXPLORATORY; clean_oof=false.'))}


def assets(command, dry_run=False):
    outputs = asset_outputs()
    if command == 'prepare' and not dry_run:
        util.need(not DATASET.exists(), 'asset destination exists; validate instead')
        for name, data in outputs.items(): util.write_once(DATASET/name, data)
    if command == 'validate' or (command == 'prepare' and not dry_run):
        verify_files(DATASET, outputs)
    return dict(status='PASS_CPU_ASSETS', command=command, dry_run=dry_run,
                files=len(outputs), bytes=sum(map(len, outputs.values())), checkpoint_sha256=FB_SHA,
                manifest_sha256=MANIFEST_SHA, private=True)


def configuration():
    c = read(PARENT/'config.json')
    c.update(release_version=RELEASE, kernel_id=SLUG)
    c['models'].append(dict(id=FB_ID, dataset_slug=DATASET_ID,
        checkpoint='weights/E_prior_s0_f0.pt', checkpoint_sha256=FB_SHA,
        trainer_source='source/compact_train.py', trainer_sha256=seed0.SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA))
    return parent.legacy.canonical(c)


def stage(diagnostic=False):
    return PARENT.parent / ('v'+RELEASE+('-n32' if diagnostic else ''))


def verify_files(path, outputs):
    util.need({p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()} == set(outputs),
              'artifact inventory drift: '+str(path))
    for name, data in outputs.items():
        util.need((path/name).read_bytes() == data, 'artifact byte drift: '+name)


def runtime_patch(data, pc, c):
    text = exact(data.decode(), 'REGISTERED = '+repr(pc), 'REGISTERED = '+repr(c))
    anchor = "        w.need(files - {'asset-manifest.json', 'dataset-metadata.json'} == set(expected), 'K1 inventory')"
    # SHA256SUMS is requested in the FB asset. Accept it ONLY at the pinned FB identity,
    # verify its exact bytes, then apply the original strict inventory check.
    addition = ("        if model_id == "+repr(FB_ID)+":\n"
        "            checksum = assets / 'SHA256SUMS'\n"
        "            w.need(checksum.is_file() and not checksum.is_symlink() and\n"
        "                   checksum.read_bytes() == "+repr(SUMS)+", 'FB SHA256SUMS pin')\n"
        "            files.remove('SHA256SUMS')\n")
    return exact(text, anchor, addition+anchor).encode()


def artifacts(diagnostic=False):
    p = PARENT.with_name(PARENT.name+('-n32' if diagnostic else ''))
    pc = parent.legacy.canonical(read(PARENT/'config.json'))
    po, pr = parent.legacy.artifacts(pc, diagnostic)
    verify_files(p, po)
    c = configuration()
    oc, cfg = parent.legacy.config(pc, diagnostic), parent.legacy.config(c, diagnostic)
    old_nb, old_meta = read(p/'kernel/release.ipynb'), read(p/'kernel/kernel-metadata.json')
    nb, meta = copy.deepcopy(old_nb), copy.deepcopy(old_meta)
    code = ''.join(old_nb['cells'][2]['source'])
    engine = parent.v10.replay.load()
    blob, pins, old = engine.before.parent.archive(code)
    new = dict(old)
    new[RUNTIME] = runtime_patch(old[RUNTIME], pc, c)
    codec = old[CODEC].decode()
    util.need(codec.count(oc['release_version']) == 2, 'codec identity anchors')
    new[CODEC] = exact(codec.replace(oc['release_version'], cfg['release_version']),
                        oc['kernel_id'], cfg['kernel_id']).encode()
    packed = engine.before.packed(new)
    changes = [(repr(base64.b64encode(blob).decode()), repr(base64.b64encode(packed).decode())),
        ('PINS = '+repr(pins), 'PINS = '+repr({n:digest(v) for n,v in new.items()})),
        (repr(digest(blob)), repr(digest(packed))), (repr(oc), repr(cfg))]
    for before, after in changes: code = exact(code, before, after)
    restored = code
    for before, after in reversed(changes): restored = exact(restored, after, before)
    util.need(restored == ''.join(old_nb['cells'][2]['source']), 'launch code drift')
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = [s.replace(oc['release_version'],cfg['release_version']) for s in nb['cells'][1]['source']]
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1],
                dataset_sources=old_meta['dataset_sources']+[DATASET_ID])
    util.need(engine.before.parent.archive(code)[2] == new, 'archive roundtrip')
    for name, data in new.items():
        if name.endswith('.py'): compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), '<P4 notebook>', 'exec')
    fields = [dict(field=k, parent=pc.get(k), candidate=c.get(k))
              for k in sorted(pc.keys()|c.keys()) if pc.get(k) != c.get(k)]
    util.need({f['field'] for f in fields} == {'release_version','kernel_id','models'}, 'P4 config scope')
    changed = sorted(n for n in old if old[n] != new[n])
    util.need(set(old) == set(new) and changed == sorted([RUNTIME,CODEC]), 'P4 payload scope')
    def functions(data):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(data).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,z = functions(old[RUNTIME]),functions(new[RUNTIME])
    util.need(a.keys() == z.keys() and [n for n in a if a[n] != z[n]] == ['load_classifier'],
              'runtime functions changed outside FB checksum verification')
    scope = dict(status='PASS_CPU_SCOPE', parent_stage=p.relative_to(ROOT).as_posix(),
        changed=changed, added=[], removed=[], untouched={n:digest(v) for n,v in old.items() if new[n]==v},
        config_changed_fields=[f['field'] for f in fields], metadata_changed_fields=['id','title','dataset_sources'],
        registered_change='Append pinned FB fourth equal K1 member and verify FB SHA256SUMS; identity/receipts only',
        parent_artifacts_byte_identical=True, parent_launch_pool_thread_timing_identical=True,
        all_runtime_functions_identical_except_fb_asset_inventory=True,
        public_math_rendering_window_precision_tta_identical=True)
    outputs = {'kernel/release.ipynb':util.encoded(nb),'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c),'payload-scope.json':util.encoded(scope),'field-level-diff.json':util.encoded(fields),
        'CHANGELOG.md':('# P4 v1.1.85 / builder v1.0.0\n\n'
            'Only FB fourth K1 member plus its assets/identities/receipts. Public target weights unchanged; no O or A5.\n'
            'K1 equal four-member rank mean, stride2; R6 stride1; R224 .35; outer P/K .75/.25.\n'
            'v10 guard8.8h/capture900/call300/load1500/failure.05 unchanged.\n'
            'FB SHA256SUMS validated before original strict asset inventory.\n'
            'Independent review and exact-candidate visible/N32 execution pending. N32 NEVER SUBMIT.\n').encode()}
    pairs = [(n,old[n],new[n]) for n in changed]+[(n,util.encoded(a),util.encoded(z)) for n,a,z in
        [('notebook',old_nb,nb),('metadata',old_meta,meta),('config',pc,c)]]
    outputs['full-payload-notebook.diff'] = ''.join(line for n,a,z in pairs for line in
        difflib.unified_diff(a.decode().splitlines(True),z.decode().splitlines(True),
                            fromfile='P25/'+n,tofile='P4/'+n)).encode()
    reg = dict(pr)
    reg.update(cfg, builder_version=VERSION, builder='tools/p25_fb_stage_v1.py',
        stage=stage(diagnostic).relative_to(ROOT).as_posix(), parent_stage=p.relative_to(ROOT).as_posix(),
        parent_is_existing_stage=True, parent_registration_sha256=digest(po['registration.json']),
        sources={n:digest(v) for n,v in new.items()},
        builder_sources={'tools/p25_fb_stage_v1.py':util.sha(Path(__file__)),
                         'tools/k1_embedded_selfcheck_v1.py':util.sha(Path(selfcheck.__file__))},
        kernel_sha256=digest(outputs['kernel/release.ipynb']), metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        independent_review='REQUIRED', execution_verified=False, qualified_for_submission=False)
    outputs['registration.json'] = util.encoded(reg)
    return outputs,reg


def checks(pairs, controls=False):
    selfcheck.SCRATCH = SCRATCH/'embedded-selfcheck'
    results = [selfcheck.check(json.loads(out['kernel/release.ipynb']), path.name) for path,out in pairs]
    if controls:
        for base in (PARENT, parent.stage()):
            for diagnostic in (False,True):
                path = base.with_name(base.name+('-n32' if diagnostic else ''))
                results.append(selfcheck.check(read(path/'kernel/release.ipynb'),path.name))
    return results


def run(command, dry_run=False):
    if command.startswith('assets-'): return assets(command.split('-',1)[1],dry_run)
    pairs = [(stage(d), *artifacts(d)) for d in (False,True)]
    if command == 'selfcheck':
        return dict(status='PASS_CPU_SELFCHECK', checks=checks([(p,o) for p,o,r in pairs],True))
    if command == 'prepare' and not dry_run:
        util.need(all(not p.exists() for p,o,r in pairs), 'immutable stage exists; validate instead')
        checks([(p,o) for p,o,r in pairs])
        for path, outputs, reg in pairs:
            for name,data in outputs.items(): util.write_once(path/name,data)
    if command == 'validate' or (command=='prepare' and not dry_run):
        for path, outputs, reg in pairs: verify_files(path,outputs)
    result = dict(status='PASS_CPU_'+command.upper(),dry_run=dry_run,qualified_for_submission=False,
        stages=[dict(stage=str(p.relative_to(ROOT)),kernel_id=r['kernel_id'],kernel_sha256=r['kernel_sha256'],
                     never_submit=r['never_submit']) for p,o,r in pairs])
    if command == 'admit':
        util.need(dry_run,'use p25_fb_admit_v1 for actual receipt admission')
        result.update(status='HOLD_NOT_ADMITTED',reasons=['Independent review and exact P4 visible/N32 receipts required',
            'Stride3 FB gain is not stride2 serving qualification; projected 6.65-7.52h is unmeasured'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','prepare','validate','selfcheck','admit','assets-prepare','assets-validate'])
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    print(json.dumps(run(a.command,a.dry_run),indent=2))


if __name__ == '__main__': main()
