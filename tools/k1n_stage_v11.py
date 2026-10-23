"""v1.0.0: one immutable P25-V10 successor, seed1 stride2 -> stride1 only.

Changelog: isolated runtime11/schema11/adapter3, exact parent archive proof.
Existing v9/v10 generators, stages, runtime and schema bytes are untouched.
"""
import argparse
import base64
import copy
import difflib
import json
from pathlib import Path

from tools import k1n_stage_v10 as prior
from tools import k1n_config_v11 as schema

ROOT, util, exact = prior.ROOT, prior.util, prior.exact
VERSION = '1.0.0'
PARENT = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.79-r6w958k1n3s2p25v10'
RELEASE = '1.1.81-r6w958k1n3s2p25s1v11'
SLUG = 'taeyangg4/knee-lb-r6w958k1n3s2p25s1v11-v1'
NEW = ('tools/k1n_config_v11.py', 'tools/k1_shared_render_v3.py', 'tools/k1sr_runtime_v11.py')
N32 = ROOT/'.local/scratch/slot-b/knee-lb-r6w958k1n3s2p25-v1-n32-v1-visible/k1sr-receipt.json'


def registered():
    c = json.loads((PARENT/'config.json').read_bytes())
    return schema.validate(dict(c, release_version=RELEASE, kernel_id=SLUG,
        k1_runtime_version=11, k1_member_eval_strides={
            'k1_v2':2, 'k1_v3g_seed0':2, 'k1_v3g_seed1':1}))


def projection():
    rec = json.loads(N32.read_bytes())
    seconds = rec['per_model_seconds']['k1_v3g_seed1']['classify']
    extra = seconds * 1300 / 32 / 3600
    return dict(schema='k1n-stride1-projection/v1', source=str(N32.relative_to(ROOT)),
        source_sha256=util.sha(N32), seed1_classify_n32_seconds=seconds,
        assumption='seed1 classification doubles; rendering/transfer overhead not measured',
        extra_classify_hours=extra, conditional_parent_hours=6.2,
        conditional_total_hours=6.2+extra, additional_render_transfer_hours=None,
        measured_for_this_candidate=False, upper_bound_proven=False,
        caveat='6.2 h is wait-inclusive parent observation, not a verified bound; actual new N32 required',
        parent_pipeline_n32_seconds=rec['parent_seconds'],
        parent_pipeline_includes_k1=True, qualified_for_submission=False)


def stage(diagnostic=False):
    return PARENT.parent/('v'+RELEASE+('-n32' if diagnostic else ''))


def package(diagnostic=False):
    engine = prior.replay.load()
    parent = PARENT.with_name(PARENT.name+('-n32' if diagnostic else ''))
    pc = json.loads((PARENT/'config.json').read_bytes())
    c = registered()
    old_nb = json.loads((parent/'kernel/release.ipynb').read_bytes())
    old_meta = json.loads((parent/'kernel/kernel-metadata.json').read_bytes())
    old_reg = json.loads((parent/'registration.json').read_bytes())
    util.need(util.sha(parent/'kernel/release.ipynb') == old_reg['kernel_sha256'] and
              util.sha(parent/'kernel/kernel-metadata.json') == old_reg['metadata_sha256'], 'parent binding')
    code = ''.join(old_nb['cells'][2]['source'])
    blob,pins,old = engine.before.parent.archive(code)
    util.need(pins == {n:engine.digest(v) for n,v in old.items()} == old_reg['sources'], 'parent payload pins')
    oc = prior.config(pc, diagnostic)
    cfg = copy.deepcopy(oc)
    cfg.update(release_version=RELEASE+('-n32' if diagnostic else ''),
        kernel_id=SLUG+('-n32' if diagnostic else ''), k1_runtime_version=11,
        k1_member_eval_strides=c['k1_member_eval_strides'],
        config_sha256=engine.digest(util.encoded(c)), registered_config_sha256=engine.digest(util.encoded(c)),
        input_transfer='once per contiguous member stride group', runtime_projection=projection())
    cfg['contract']['member_eval_strides'] = c['k1_member_eval_strides']
    cfg['formula'] = cfg['formula'].replace('k1_runtime_version=10','k1_runtime_version=11') + '; member eval strides: k1_v2=2, k1_v3g_seed0=2, k1_v3g_seed1=1'
    new = dict(old)
    for name in NEW:
        new[name] = (ROOT/name).read_bytes()
    new[NEW[-1]] = exact(new[NEW[-1]].decode(), 'REGISTERED = config_schema.current()', 'REGISTERED = '+repr(c)).encode()
    port,release,codec = 'exp169_r7_port.py','exp169_release_c85r7.py','exp169_release_c85r7codec.py'
    new[port] = exact(old[port].decode(),'k1sr_runtime_v10','k1sr_runtime_v11').encode()
    new[release] = exact(old[release].decode(),engine.digest(old[port]),engine.digest(new[port])).encode()
    text = exact(old[codec].decode(),engine.digest(old[release]),engine.digest(new[release]))
    util.need(text.count(oc['release_version']) == 2,'codec version anchors')
    text = text.replace(oc['release_version'],cfg['release_version'])
    new[codec] = exact(text,oc['kernel_id'],cfg['kernel_id']).encode()
    packed = engine.before.packed(new)
    code = exact(code,repr(base64.b64encode(blob).decode()),repr(base64.b64encode(packed).decode()))
    code = exact(code,'PINS = '+repr(pins),'PINS = '+repr({n:engine.digest(v) for n,v in new.items()}))
    code = exact(code,repr(engine.digest(blob)),repr(engine.digest(packed)))
    code = exact(code,'from tools import k1sr_runtime_v10 as _k1sr','from tools import k1sr_runtime_v11 as _k1sr')
    code = exact(code,repr(oc),repr(cfg))
    nb,meta = copy.deepcopy(old_nb),copy.deepcopy(old_meta)
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = ['# P25 seed1 stride1 runtime11\n',cfg['formula']+'\n',
        'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else 'UNQUALIFIED: paired pilot, independent review and inference qualification required.\n']
    meta.update(id=cfg['kernel_id'],title=cfg['kernel_id'].split('/')[1])
    util.need(set(new)-set(old) == set(NEW) and not set(old)-set(new), 'added payload scope')
    util.need({n for n in old if old[n]!=new[n]} == {port,release,codec}, 'changed payload scope')
    util.need(engine.before.parent.archive(code)[2] == new, 'archive roundtrip')
    util.need(nb['cells'][:1] == old_nb['cells'][:1] and nb['cells'][3:] == old_nb['cells'][3:], 'notebook scope')
    for name,data in new.items():
        if name.endswith('.py'): compile(data,name,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']),'<notebook>','exec')
    return nb,meta,old,new,old_nb,old_meta,cfg,old_reg


def artifacts(diagnostic=False):
    engine = prior.replay.load()
    nb,meta,old,new,on,om,cfg,oreg = package(diagnostic)
    c = registered()
    pc = json.loads((PARENT/'config.json').read_bytes())
    fields = {k:dict(before=pc.get(k),after=c.get(k)) for k in pc.keys()|c.keys() if pc.get(k)!=c.get(k)}
    util.need(set(fields)=={'release_version','kernel_id','k1_runtime_version','k1_member_eval_strides'}, 'config scope')
    scope = dict(status='PASS',parent_stage=str(PARENT.relative_to(ROOT)),config_diff=fields,
        effective_stride_diff={m:dict(before=2,after=c['k1_member_eval_strides'][m]) for m in c['k1_member_eval_strides']},
        changed=[n for n in old if old[n]!=new[n]],added=list(NEW),removed=[],
        untouched={n:engine.digest(v) for n,v in old.items() if new[n]==v},
        registered_change='Only k1_v3g_seed1 evaluation stride 2 -> 1; other members remain 2',
        scientific_fields_unchanged=['models','alpha','drop_o','encoder_chunk','k1_eval_stride','k1_failure_policy','k1_max_failed_fraction','k1_guard_hours'],
        timeout_launch_pool_thread_contracts_unchanged=True)
    pairs = [(n,old.get(n,b''),new[n]) for n in new if old.get(n)!=new[n]]
    pairs += [('aligned-runtime10-to11',old['tools/k1sr_runtime_v10.py'],new[NEW[-1]]),
              ('aligned-adapter2-to3',old['tools/k1_shared_render_v2.py'],new[NEW[1]])]
    pairs += [(n,util.encoded(a),util.encoded(z)) for n,a,z in [('notebook',on,nb),('metadata',om,meta),('config',pc,c)]]
    diff=''.join(line for n,a,z in pairs for line in difflib.unified_diff(a.decode().splitlines(True),z.decode().splitlines(True),fromfile='parent/'+n,tofile='v11/'+n))
    outputs={'kernel/release.ipynb':util.encoded(nb),'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(c),'payload-scope.json':util.encoded(scope),
        'full-payload-notebook.diff':diff.encode(),'runtime-projection.json':util.encoded(projection()),
        'CHANGELOG.md':('# Runtime v11 / builder v1.0.0\n\n'+scope['registered_change']+'.\nIsolated schema11/adapter3; member strides in receipts. Legacy bytes unchanged.\nUNQUALIFIED. N32 sibling NEVER SUBMIT.\n').encode()}
    reg=dict(oreg,**{k:v for k,v in cfg.items() if k not in oreg})
    reg.update(cfg)
    reg.update(builder_version=VERSION,builder_sources={n:util.sha(ROOT/n) for n in NEW+('tools/k1n_stage_v11.py',)},
        parent_stage=str(PARENT.relative_to(ROOT))+('-n32' if diagnostic else ''),
        parent_registration_sha256=util.sha(PARENT.with_name(PARENT.name+('-n32' if diagnostic else ''))/'registration.json'),
        stage=str(stage(diagnostic).relative_to(ROOT)),sources={n:engine.digest(v) for n,v in new.items()},
        kernel_sha256=engine.digest(outputs['kernel/release.ipynb']),metadata_sha256=engine.digest(outputs['kernel/kernel-metadata.json']),
        execution_verified=False,independent_review='REQUIRED',qualified_for_submission=False)
    outputs['registration.json']=util.encoded(reg)
    return outputs,reg


def run(command):
    pairs=[(d,*artifacts(d)) for d in (False,True)]
    if command=='prepare':
        util.need(all(not stage(d).exists() for d,_,_ in pairs),'immutable stage exists; validate instead')
        for d,outputs,_ in pairs:
            for n,data in outputs.items(): util.write_once(stage(d)/n,data)
    else: util.need(command=='validate','unknown command')
    for d,outputs,_ in pairs:
        util.need({p.relative_to(stage(d)).as_posix() for p in stage(d).rglob('*') if p.is_file()}==set(outputs),'stage inventory')
        for n,data in outputs.items(): util.need((stage(d)/n).read_bytes()==data,'stage bytes: '+n)
    return dict(status='PASS',command=command,stages=[dict(stage=r['stage'],kernel_sha256=r['kernel_sha256'],never_submit=d) for d,_,r in pairs],qualified_for_submission=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','validate'))
    print(json.dumps(run(parser.parse_args().command),indent=2))
