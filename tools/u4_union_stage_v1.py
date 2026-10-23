"""U4 immutable builder v1.0.0.

Changelog: U plus exactly the P4 pinned FB fourth member at stride2;
reuse P4 checksum loader patch, with unchanged U composition and v10 guards.
"""
import argparse
import ast
import base64
import copy
import difflib
import json
from tools import u_union_stage_v1 as parent
from tools import p25_fb_stage_v1 as fb
from tools import k1_embedded_selfcheck_v1 as selfcheck

VERSION = '1.0.0'
ROOT, util, exact, digest = parent.ROOT, parent.util, parent.exact, parent.digest
PARENT = parent.stage()
P4 = fb.stage()
RELEASE = '1.1.86-r6w958k1u4fbv10'
SLUG = 'taeyangg4/knee-lb-r6w958k1u4fbv10-v1'
SCRATCH = ROOT / '.local/scratch/astra-b5/u4-v1'
RUNTIME, CODEC, INTEGRATION = parent.RUNTIME, fb.CODEC, parent.INTEGRATION
FORMULA = parent.union.FORMULA.replace('K1 trio stride2', 'K1 four (v2, v3-G seed0, v3-G seed1, FB fold0) stride2')

def read(path): return json.loads(path.read_bytes())
def stage(diagnostic=False): return PARENT.parent / ('v'+RELEASE+('-n32' if diagnostic else ''))

def configuration():
    c = read(PARENT/'config.json')
    p4 = read(P4/'config.json')
    util.need(p4['models'][:3] == c['models'] and len(p4['models']) == 4, 'exact P4 trio/FB')
    util.need(p4['models'][-1] == fb.configuration()['models'][-1], 'pinned P4 FB registration')
    c.update(release_version=RELEASE, kernel_id=SLUG, models=p4['models'])
    return c

def functions(data):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(data).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def artifacts(diagnostic=False):
    p = parent.stage(diagnostic)
    po, pr = parent.artifacts(diagnostic)
    fb.verify_files(p, po)
    pc, c = read(p/'config.json'), configuration()
    oldnb, oldmeta = read(p/'kernel/release.ipynb'), read(p/'kernel/kernel-metadata.json')
    nb, meta = copy.deepcopy(oldnb), copy.deepcopy(oldmeta)
    original = ''.join(nb['cells'][2]['source'])
    blob, pins, old = parent.archive.parent.archive(original)
    call = next(n for n in ast.walk(ast.parse(original)) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='finish')
    oc = ast.literal_eval(call.args[2]); cfg = copy.deepcopy(oc)
    cfg.update(release_version=RELEASE+('-n32' if diagnostic else ''),
        kernel_id=SLUG+('-n32' if diagnostic else ''), formula=FORMULA,
        config_sha256=digest(util.encoded(c)), registered_config_sha256=digest(util.encoded(c)),
        models=c['models'], k1_models=[m['id'] for m in c['models']],
        model_sha256={m['id']:m['checkpoint_sha256'] for m in c['models']},
        asset_manifest_sha256={m['id']:m['manifest_sha256'] for m in c['models']})
    new = dict(old)
    # Literal insertion order in the embedded REGISTERED differs from sorted JSON.
    registered = next(ast.literal_eval(n.value) for n in ast.parse(old[RUNTIME]).body
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REGISTERED' for t in n.targets))
    util.need(registered == pc, 'U embedded registration')
    new[RUNTIME] = fb.runtime_patch(old[RUNTIME], registered, c)
    new[RUNTIME] = exact(new[RUNTIME].decode(), "VERSION = '10.0.3-u'", "VERSION = '10.0.4-u4'").encode()
    new[RUNTIME] = exact(new[RUNTIME].decode(), 'runtime v10.0.3-u.',
        'runtime v10.0.4-u4.\n\nChangelog: pinned P4 FB fourth member and checksum inventory; U arithmetic unchanged.').encode()
    new[INTEGRATION] = exact(old[INTEGRATION].decode(), repr(parent.union.FORMULA), repr(FORMULA)).encode()
    new[INTEGRATION] = exact(new[INTEGRATION].decode(), "VERSION = '1.0.0'", "VERSION = '1.0.1-u4'").encode()
    new[CODEC] = old[CODEC].decode().replace(oc['release_version'],cfg['release_version']).replace(oc['kernel_id'],cfg['kernel_id']).encode()
    p4files, _ = selfcheck.payload(read(fb.stage(diagnostic)/'kernel/release.ipynb'))
    util.need(functions(p4files[RUNTIME])['load_classifier'] == functions(new[RUNTIME])['load_classifier'], 'EXACT P4 FB loader')
    util.need(functions(old[INTEGRATION]) == functions(new[INTEGRATION]), 'unchanged U composition')
    before, after = functions(old[RUNTIME]), functions(new[RUNTIME])
    util.need(before.keys()==after.keys() and [n for n in before if before[n]!=after[n]]==['load_classifier'], 'runtime scope')
    packed = parent.archive.packed(new)
    replacements = [(repr(base64.b64encode(blob).decode()),repr(base64.b64encode(packed).decode())),
        ('PINS = '+repr(pins),'PINS = '+repr({n:digest(v) for n,v in new.items()})),
        (repr(digest(blob)),repr(digest(packed))),(repr(oc),repr(cfg))]
    code = original
    for a,z in replacements: code=exact(code,a,z)
    restored = code
    for a,z in reversed(replacements): restored=exact(restored,z,a)
    util.need(restored==original, 'unchanged notebook execution')
    nb['cells'][2]['source']=code.splitlines(True)
    nb['cells'][1]['source']=['# U4 union v1.1.86\n',FORMULA+'\n',
        'N32 diagnostic; NEVER SUBMIT.\n' if diagnostic else 'Independent review and real qualification required.\n']
    meta.update(id=cfg['kernel_id'],title=cfg['kernel_id'].split('/')[1],
        dataset_sources=oldmeta['dataset_sources']+[fb.DATASET_ID])
    util.need(parent.archive.parent.archive(code)[2]==new, 'payload roundtrip')
    changed = sorted(n for n in old if old[n]!=new[n])
    util.need(set(old)==set(new) and set(changed)=={RUNTIME,CODEC,INTEGRATION}, 'payload scope')
    fields = [dict(field=k,parent=pc.get(k),candidate=c.get(k)) for k in sorted(pc.keys()|c.keys()) if pc.get(k)!=c.get(k)]
    util.need({f['field'] for f in fields}=={'models','release_version','kernel_id'}, 'config scope')
    for n,data in new.items():
        if n.endswith('.py'): compile(data,n,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']),'<U4 notebook>','exec')
    scope=dict(status='PASS_CPU_SCOPE',parent_stage=p.relative_to(ROOT).as_posix(),changed=changed,added=[],removed=[],
        untouched={n:digest(v) for n,v in old.items() if new[n]==v},runtime_functions_changed=['load_classifier'],
        exact_p4_fb_loader=True,union_functions_identical=True,launch_pool_thread_timing_identical=True,
        config_changed_fields=[f['field'] for f in fields],metadata_changed_fields=['id','title','dataset_sources'])
    outputs={'kernel/release.ipynb':util.encoded(nb),'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c),'payload-scope.json':util.encoded(scope),'field-level-diff.json':util.encoded(fields),
        'CHANGELOG.md':b'# U4 v1.1.86 / builder v1.0.0\n\nU v1.1.84 plus exact P4 FB member/loader at stride2.\nU weights, retained O, no A5, v10 guard8.8h/capture900/call300/load1500/failure.05 unchanged.\nEmbedded runtime10.0.4-u4, composition1.0.1-u4 (formula/version text only).\nIndependent review and real visible/N32 execution pending. N32 NEVER SUBMIT.\n'}
    pairs=[(n,old[n],new[n]) for n in changed]+[(n,util.encoded(a),util.encoded(z)) for n,a,z in
        [('notebook',oldnb,nb),('metadata',oldmeta,meta),('config',pc,c)]]
    outputs['full-payload-notebook.diff']=''.join(line for n,a,z in pairs for line in difflib.unified_diff(
        a.decode().splitlines(True),z.decode().splitlines(True),fromfile='U/'+n,tofile='U4/'+n)).encode()
    reg=dict(pr,**{})
    reg.update(cfg,builder_version=VERSION,builder='tools/u4_union_stage_v1.py',
        stage=stage(diagnostic).relative_to(ROOT).as_posix(),parent_stage=p.relative_to(ROOT).as_posix(),
        parent_registration_sha256=digest(po['registration.json']),
        fb_parent_registration_sha256=util.sha(fb.stage(diagnostic)/'registration.json'),
        sources={n:digest(v) for n,v in new.items()},
        builder_sources={n:util.sha(ROOT/n) for n in ['tools/u4_union_stage_v1.py','tools/p25_fb_stage_v1.py','tools/k1_embedded_selfcheck_v1.py']},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        independent_review='REQUIRED',execution_verified=False,qualified_for_submission=False)
    outputs['registration.json']=util.encoded(reg)
    return outputs,reg

def run(command,dry_run=False):
    pairs=[(stage(d),*artifacts(d)) for d in (False,True)]
    if command=='selfcheck':
        selfcheck.SCRATCH=SCRATCH/'embedded-selfcheck'
        candidates=[(p,json.loads(o['kernel/release.ipynb'])) for p,o,r in pairs]
        controls=[(p,read(p/'kernel/release.ipynb')) for base in (PARENT,P4)
            for p in (base,base.with_name(base.name+'-n32'))]
        return dict(status='PASS_CPU_SELFCHECK',checks=[selfcheck.check(nb,p.name) for p,nb in candidates+controls])
    if command=='prepare' and not dry_run:
        util.need(all(not p.exists() for p,o,r in pairs),'immutable stage exists')
        for p,o,r in pairs:
            for n,data in o.items(): util.write_once(p/n,data)
    if command=='validate' or command=='prepare' and not dry_run:
        for p,o,r in pairs: fb.verify_files(p,o)
    return dict(status='HOLD_NOT_ADMITTED' if command=='admit' else 'PASS_CPU_'+command.upper(),dry_run=dry_run,
        stages=[str(p.relative_to(ROOT)) for p,o,r in pairs],qualified_for_submission=False,
        remaining=['independent review','real U4 visible/N32 execution','B runtime gate'] if command=='admit' else [])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['register','prepare','validate','selfcheck','admit'])
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();print(json.dumps(run(a.command,a.dry_run),indent=2))

if __name__=='__main__':main()
