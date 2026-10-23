"""v1.0.1: repaired flat-P twin of the materialized P25+A5 v10 stage.

Changelog v1.0.1: pin the successful flat schema/runtime and require isolated
embedded import validation for both notebooks. Failed v1.1.82 stays immutable.

Changelog v1.0.0: apply only the registered all-target .5 P weights, identity and
chained hash/formula metadata. Reuse p25_a5_stage_v1 v1.0.0 unchanged.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import json

from tools import p25_a5_stage_v1 as parent
from tools import k1n_flat_p_v1 as flat
from tools import k1_embedded_selfcheck_v1 as selfcheck

VERSION = '1.0.1'
ROOT, util, exact = parent.ROOT, parent.util, parent.exact
PARENT = parent.stage()
RELEASE = '1.1.82-b-r6w958k1n3s2p25a5flatpv10'
SLUG = 'taeyangg4/knee-lb-r6w958k1n3s2p25a5flatpv10b-v1'
SCRATCH = ROOT/'.local/scratch/astra-b1/a5-flat-v1'
RUNTIME, CODEC = parent.RUNTIME, parent.CODEC
FLAT_PARENT = PARENT.parent/'v1.1.79-r6w958k1n3s2p25flatpv10'
digest = parent.digest


def stage(diagnostic=False):
    return PARENT.parent/('v'+RELEASE+('-n32' if diagnostic else ''))


def config(c, diagnostic=False):
    cfg = parent.legacy.config(c, diagnostic)
    cfg['formula'] = parent.a5.FORMULA+'; inherited outer K1: '+cfg['formula']
    cfg['a5_contract_sha256'] = digest(util.encoded(parent.contract()))
    return cfg


def package(diagnostic=False):
    outputs, reg = parent.artifacts(diagnostic)
    path = parent.stage(diagnostic)
    util.need({p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}
              == set(outputs), 'A5 parent inventory drift')
    for name, data in outputs.items():
        util.need((path/name).read_bytes() == data, 'A5 parent byte drift: '+name)
    old_config = json.loads(outputs['config.json'])
    pc = parent.legacy.canonical({k:v for k,v in old_config.items() if k != 'a5'})
    c = parent.legacy.canonical(dict(pc, release_version=RELEASE, kernel_id=SLUG,
                                    p_weight_mode='flat'))
    cfg = config(c, diagnostic)
    old_nb = json.loads(outputs['kernel/release.ipynb'])
    old_meta = json.loads(outputs['kernel/kernel-metadata.json'])
    nb, meta = copy.deepcopy(old_nb), copy.deepcopy(old_meta)
    code = ''.join(old_nb['cells'][2]['source'])
    engine = parent.v10.replay.load()
    blob, pins, old = engine.before.parent.archive(code)
    new = flat.patch_math(old, c['drop_o'])
    new[RUNTIME] = exact(old[RUNTIME].decode(), 'REGISTERED = '+repr(pc),
                         'REGISTERED = '+repr(c)).encode()
    # Source the schema and runtime from the successfully executed flat stage.
    fp = FLAT_PARENT.with_name(FLAT_PARENT.name+('-n32' if diagnostic else ''))
    fnb = json.loads((fp/'kernel/release.ipynb').read_bytes())
    freg = json.loads((fp/'registration.json').read_bytes())
    util.need(util.sha(fp/'kernel/release.ipynb') == freg['kernel_sha256'], 'flat parent notebook binding')
    ff = selfcheck.payload(fnb)[0]
    util.need({n:digest(v) for n,v in ff.items()} == freg['sources'], 'flat parent payload binding')
    fc = parent.legacy.canonical(json.loads((fp/'config.json').read_bytes()))
    repaired_runtime = parent.runtime_patch(ff[RUNTIME], fc, c)
    util.need(repaired_runtime == new[RUNTIME], 'flat runtime plus exact A5 patch parity')
    new[RUNTIME] = repaired_runtime
    new[flat.SCHEMA] = ff[flat.SCHEMA]
    codec = new[CODEC].decode()
    util.need(codec.count(reg['release_version']) == 2, 'codec identity anchors')
    codec = codec.replace(reg['release_version'], cfg['release_version'])
    new[CODEC] = exact(codec, reg['kernel_id'], cfg['kernel_id']).encode()
    oc = config(pc, diagnostic)
    packed = engine.before.packed(new)
    replacements = [
        (repr(base64.b64encode(blob).decode()), repr(base64.b64encode(packed).decode())),
        ('PINS = '+repr(pins), 'PINS = '+repr({n:digest(v) for n,v in new.items()})),
        (repr(digest(blob)), repr(digest(packed))), (repr(oc), repr(cfg)),
    ]
    for before, after in replacements:
        code = exact(code, before, after)
    nb['cells'][2]['source'] = code.splitlines(True)
    # Only the formula/identity metadata changes outside the embedded payload.
    nb['cells'][1]['source'] = [s.replace(oc['formula'], cfg['formula'])
                               for s in nb['cells'][1]['source']]
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1])
    util.need(engine.before.parent.archive(code)[2] == new, 'flat A5 archive roundtrip')
    for name, data in new.items():
        if name.endswith('.py'): compile(data, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']), '<flat A5 notebook>', 'exec')
    # Reverse every allowed notebook substitution to prove the launch is identical.
    restored = code
    for before, after in reversed(replacements): restored = exact(restored, after, before)
    util.need(restored == ''.join(old_nb['cells'][2]['source']), 'notebook execution drift')
    return outputs, reg, old_config, c, cfg, old_nb, nb, old_meta, meta, old, new


def artifacts(diagnostic=False):
    po, pr, old_config, c, cfg, old_nb, nb, old_meta, meta, old, new = package(diagnostic)
    full_config = dict(c, a5=parent.contract())
    fields = [{'field':k, 'parent':old_config.get(k), 'candidate':full_config.get(k)}
              for k in sorted(old_config.keys()|full_config.keys())
              if old_config.get(k) != full_config.get(k)]
    util.need({f['field'] for f in fields} == {'kernel_id','release_version','p_weight_mode'},
              'only flat P and identity config fields may change')
    changed = sorted(n for n in old if new[n] != old[n])
    util.need(set(old) == set(new) and changed == sorted([flat.HELPER,CODEC,RUNTIME,flat.SCHEMA]),
              'flat A5 payload scope')
    def functions(data):
        return {n.name:ast.dump(n, include_attributes=False) for n in ast.parse(data).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    util.need(functions(old[RUNTIME]) == functions(new[RUNTIME]), 'K1 function drift')
    util.need(all(old[n] == new[n] for n in parent.MODULES), 'A5 module drift')
    scope = dict(status='PASS_CPU_SCOPE', parent_stage=str(parent.stage(diagnostic).relative_to(ROOT)),
        changed=changed, added=[], removed=[],
        untouched={n:digest(v) for n,v in old.items() if new[n] == v},
        config_changed_fields=[f['field'] for f in fields], metadata_changed_fields=['id','title'],
        scientific_change=dict(p_weight_mode='flat', p_raptor_weights=list(flat.FLAT),
                               changed_target_indices=list(flat.CHANGED)),
        k1_functions_identical=True, a5_modules_byte_identical=True,
        flat_schema_source=str(FLAT_PARENT.relative_to(ROOT)),
        flat_schema_byte_identical=True, flat_runtime_plus_exact_a5_patch=True,
        parent_launch_pool_threads_timing_identical=True, parent_artifacts_byte_identical=True)
    outputs = {'kernel/release.ipynb':util.encoded(nb), 'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(full_config), 'payload-scope.json':util.encoded(scope),
        'field-level-diff.json':util.encoded(fields),
        'CHANGELOG.md':('# P25+A5 flat-P extension v1.0.1\n\n'
            'Repair b: embed the successful flat schema; isolate embedded CPU imports.\n'
            'Parent generator: p25_a5_stage_v1 v1.0.0 (unchanged).\n'
            'Only p_weight_mode=flat: all twelve P weights .5 before the unchanged A5 mix.\n'
            "Pprime_t = rank(.85*r(P_flat)_t + .15*r(A5)_t).\n"
            'K1 trio/stride 2/.25, drop O, A5 fivefold/.15, runtime v10 unchanged.\n'
            'Speculative CPU build; independent review and new visible/N32 qualification pending.\n'
            'N32 is NEVER SUBMIT.\n').encode()}
    pairs = [(n,old[n],new[n]) for n in changed]
    pairs += [(n,util.encoded(a),util.encoded(z)) for n,a,z in
              [('notebook',old_nb,nb),('metadata',old_meta,meta),('config',old_config,full_config)]]
    outputs['full-payload-notebook.diff'] = ''.join(line for name,a,z in pairs
        for line in difflib.unified_diff(a.decode().splitlines(True),z.decode().splitlines(True),
                                        fromfile='A5/'+name,tofile='A5-flat-P/'+name)).encode()
    reg = dict(pr, **{})
    reg.update(cfg, stage=str(stage(diagnostic).relative_to(ROOT)).replace('\\','/'),
        builder_version=VERSION, builder='tools/p25_a5_flat_stage_v1.py',
        parent_generator_version=parent.VERSION, parent_registration_sha256=digest(po['registration.json']),
        sources={n:digest(v) for n,v in new.items()},
        builder_sources={**pr['builder_sources'], **{n:util.sha(ROOT/n) for n in
            ['tools/p25_a5_flat_stage_v1.py','tools/k1n_flat_p_v1.py',
             'tools/k1_embedded_selfcheck_v1.py']}},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),
        metadata_sha256=digest(outputs['kernel/kernel-metadata.json']))
    outputs['registration.json'] = util.encoded(reg)
    return outputs, reg


def run(command, dry_run=False):
    pairs = [(d,stage(d),*artifacts(d)) for d in (False,True)]
    checks = [selfcheck.check(json.loads(outputs['kernel/release.ipynb']), path.name)
              for _,path,outputs,_ in pairs]
    if command == 'prepare' and not dry_run:
        util.need(all(not path.exists() for _,path,_,_ in pairs), 'immutable stage exists; validate instead')
        for _,path,outputs,_ in pairs:
            for name,data in outputs.items(): util.write_once(path/name,data)
    if command == 'validate' or (command == 'prepare' and not dry_run):
        for _,path,outputs,_ in pairs:
            util.need({p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()} == set(outputs),
                      'flat A5 stage inventory drift')
            for name,data in outputs.items():
                util.need((path/name).read_bytes() == data, 'flat A5 stage byte drift: '+name)
    result = dict(embedded_selfchecks=checks, status='PASS_CPU_'+command.upper(), dry_run=dry_run, qualified_for_submission=False,
        stages=[dict(stage=str(path.relative_to(ROOT)), kernel_id=reg['kernel_id'],
                     kernel_sha256=reg['kernel_sha256'], never_submit=d) for d,path,_,reg in pairs])
    if command == 'admit':
        util.need(dry_run, 'admission execution is not supported by this CPU builder')
        result.update(status='HOLD_NOT_ADMITTED', reasons=[
            'Speculative candidate: independent review and exact-candidate visible/N32 qualification pending',
            'Parent A5 receipts cannot qualify the new flat-P candidate'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register','prepare','validate','admit'])
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.command,args.dry_run),indent=2))


if __name__ == '__main__': main()

