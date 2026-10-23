"""U union immutable stage generator v1.0.0.

Changelog: derive XA, remove A5, stride2, fixed U composer; retain embedded schema.
Existing stages and generators remain byte-identical.
"""
import argparse
import ast
import base64
import copy
import difflib
import hashlib
import json
from pathlib import Path
from tools import k1n_stage_v10 as base
from tools import c85r7_r6w958k1x3sr_v1162 as archive
from tools import u_union_integration_v1 as union
VERSION='1.0.0'
ROOT,util,exact=base.ROOT,base.util,base.exact
PARENT=ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.83-r6w958k1x3a5v10'
RELEASE='1.1.84-r6w958k1uxpv10'
SLUG='taeyangg4/knee-lb-r6w958k1uxpv10-v1'
RUNTIME='tools/k1sr_runtime_v10.py'
INTEGRATION='tools/u_union_integration_v1.py'
def digest(data): return hashlib.sha256(data).hexdigest()
def stage(diagnostic=False): return PARENT.parent/('v'+RELEASE+('-n32' if diagnostic else ''))
def artifacts(diagnostic=False):
    path=Path(str(PARENT)+('-n32' if diagnostic else ''))
    nb=json.loads((path/'kernel/release.ipynb').read_bytes()); meta=json.loads((path/'kernel/kernel-metadata.json').read_bytes())
    reg=json.loads((path/'registration.json').read_bytes())
    util.need(util.sha(path/'kernel/release.ipynb')==reg['kernel_sha256'],'XA notebook pin')
    util.need(util.sha(path/'kernel/kernel-metadata.json')==reg['metadata_sha256'],'XA metadata pin')
    code=''.join(nb['cells'][2]['source']); blob,pins,old=archive.parent.archive(code)
    call=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='finish')
    oc=ast.literal_eval(call.args[2]); cfg=copy.deepcopy(oc)
    pc=json.loads((path/'config.json').read_bytes()); pc.pop('a5')
    from tools import k1n_config_v1 as schema
    c=dict(pc,release_version=RELEASE,kernel_id=SLUG,k1_eval_stride=2); c=schema.validate(c)
    cfg.update(release_version=RELEASE+('-n32' if diagnostic else ''),kernel_id=SLUG+('-n32' if diagnostic else ''),
        config_sha256=digest(util.encoded(c)),registered_config_sha256=digest(util.encoded(c)),k1_eval_stride=2,formula=union.FORMULA)
    cfg.pop('a5_contract_sha256',None)
    cfg['contract']['eval_stride']=2
    cfg['variant']='u-union'
    new=dict(old)
    for name in ['tools/p25_a5_model_v1.py','tools/p25_a5_render_v1.py','tools/p25_a5_runtime_v1.py','tools/x3_a5_integration_v1.py']: del new[name]
    baked=next(n for n in ast.parse(old[RUNTIME]).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REGISTERED' for t in n.targets))
    runtime=exact(old[RUNTIME].decode(),ast.get_source_segment(old[RUNTIME].decode(),baked),'REGISTERED = '+repr(c))
    runtime=exact(runtime,"VERSION = '10.0.2-x3a5'","VERSION = '10.0.3-u'")
    runtime=exact(runtime,'K1 N-model shared-R6 release runtime v10.0.1-a5.','K1 N-model shared-R6 release runtime v10.0.3-u.')
    runtime=exact(runtime,'Changelog: authenticated Pprime input; K1 arithmetic unchanged.','Changelog: retained INF3/P inputs and fixed U union; no A5; stride2 registration.')
    runtime=exact(runtime,'def finish(output, parent_run, config, parent_seconds, parent_override=None):','def finish(output, parent_run, config, parent_seconds):')
    start=runtime.index("    w.need(parent_override is not None"); end=runtime.index("    util.write_once(output / 'k1sr-parent.csv', parent)",start)
    runtime=runtime[:start]+runtime[end:]
    runtime=exact(runtime,"parent pipeline including in-pass K1 plus A5 phase","parent pipeline including in-pass K1; no A5")
    start=runtime.index("    rec['parent_mode'] =");end=runtime.index("    rec['k1_models'] =",start)
    runtime=runtime[:start]+"    rec['parent_mode'] = 'U_UNION_INF3_PUBLIC'\n"+runtime[end:]
    runtime=exact(runtime,'        payload = blend(parent, rows)',
        "        from tools import u_union_integration_v1 as union\n        payload = union.compose(output, parent, rows, sys.modules[__name__], diagnostic)\n        rec['u_receipt_sha256'] = w.sha(output / 'u-receipt.json')")
    new[RUNTIME]=runtime.encode()
    new[INTEGRATION]=(ROOT/INTEGRATION).read_bytes()
    # The exact parent schema already accepts this runtime10/stride2 registration.
    codec=old['exp169_release_c85r7codec.py'].decode()
    codec=exact(codec,'from tools import x3_a5_integration_v1 as x3a5','from tools import u_union_integration_v1 as union')
    codec=exact(codec,'receipt.update(x3a5.snapshot(','receipt.update(union.snapshot(')
    codec=codec.replace(reg['release_version'],cfg['release_version']).replace(reg['kernel_id'],cfg['kernel_id'])
    new['exp169_release_c85r7codec.py']=codec.encode()
    start=code.index('    from tools import x3_a5_integration_v1 as _x3a5'); end=code.index('    _result = _k1sr.finish',start)
    code=code[:start]+code[end:]
    code=exact(code,', parent_override=_inf3)',')')
    packed=archive.packed(new)
    for before,after in [(repr(base64.b64encode(blob).decode()),repr(base64.b64encode(packed).decode())),
        ('PINS = '+repr(pins),'PINS = '+repr({n:digest(v) for n,v in new.items()})),(repr(digest(blob)),repr(digest(packed))),(repr(oc),repr(cfg))]: code=exact(code,before,after)
    outnb=copy.deepcopy(nb);outnb['cells'][2]['source']=code.splitlines(True)
    outnb['cells'][1]['source']=['# U union v1.1.84\n',union.FORMULA+'\n','N32 diagnostic; NEVER SUBMIT.\n' if diagnostic else 'Independent admission and review required.\n']
    outmeta=copy.deepcopy(meta);outmeta.update(id=cfg['kernel_id'],title=cfg['kernel_id'].split('/')[1])
    outmeta['dataset_sources']=[v for v in meta['dataset_sources'] if v!='mattiaangeli/knee-mri-fold-weights']
    assert archive.parent.archive(code)[2]==new
    for n,data in new.items():
        if n.endswith('.py'): compile(data,n,'exec')
    for cell in outnb['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<U notebook>','exec')
    changed=sorted(n for n in old.keys()&new.keys() if old[n]!=new[n]);added=sorted(new.keys()-old.keys());removed=sorted(old.keys()-new.keys())
    assert set(changed)=={RUNTIME,'exp169_release_c85r7codec.py'}
    def funcs(data): return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(data).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    of,nf=funcs(old[RUNTIME]),funcs(new[RUNTIME]);assert of.keys()==nf.keys() and [n for n in of if of[n]!=nf[n]]==['finish']
    scope=dict(status='PASS_CPU_SCOPE',parent=str(path.relative_to(ROOT)),changed=changed,added=added,removed=removed,
        untouched={n:digest(v) for n,v in old.items() if new.get(n)==v},runtime_functions_changed=['finish'],
        model_render_launch_pool_thread_timing_unchanged=True,contract=union.CONTRACT)
    outputs={'kernel/release.ipynb':util.encoded(outnb),'kernel/kernel-metadata.json':util.encoded(outmeta),
        'config.json':util.encoded(c),'payload-scope.json':util.encoded(scope),
        'CHANGELOG.md':b'# Release 1.1.84 / builder 1.0.0\n\nXA parent: remove A5 modules/dataset/execution; K1 stride 3 to 2; compose fixed U union.\nRetain the exact parent runtime10-compatible schema. Save P/O/INF3/K/X/Y/U with receipt hashes.\nNo original files/stages edited. Inference scheduling, timeouts and model code unchanged.\n'}
    diff=[]
    for n in sorted(old.keys()|new.keys()):
        if old.get(n)!=new.get(n): diff.extend(difflib.unified_diff(old.get(n,b'').decode().splitlines(True),new.get(n,b'').decode().splitlines(True),fromfile='XA/'+n,tofile='U/'+n))
    diff.extend(difflib.unified_diff(util.encoded(nb).decode().splitlines(True),util.encoded(outnb).decode().splitlines(True),fromfile='XA/notebook',tofile='U/notebook'))
    diff.extend(difflib.unified_diff(util.encoded(meta).decode().splitlines(True),util.encoded(outmeta).decode().splitlines(True),fromfile='XA/metadata',tofile='U/metadata'))
    outputs['full-payload-notebook.diff']=''.join(diff).encode()
    registration=dict(cfg,schema='u-union-registration/v1',u_contract=union.CONTRACT,stage=str(stage(diagnostic).relative_to(ROOT)).replace('\\','/'),builder_version=VERSION,
        parent_registration_sha256=util.sha(path/'registration.json'),sources={n:digest(v) for n,v in new.items()},
        builder_sources={n:util.sha(ROOT/n) for n in [INTEGRATION,'tools/u_union_stage_v1.py']},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        independent_review='REQUIRED',execution_verified=False,qualified_for_submission=False)
    outputs['registration.json']=util.encoded(registration)
    return outputs,registration

def run(command,dry_run=False):
    pairs=[(d,stage(d),*artifacts(d)) for d in (False,True)]
    if command=='prepare' and not dry_run:
        util.need(all(not p.exists() for _,p,_,_ in pairs),'immutable stage exists')
        for _,p,outputs,_ in pairs:
            for name,data in outputs.items():util.write_once(p/name,data)
    if command=='validate' or command=='prepare' and not dry_run:
        for _,p,outputs,_ in pairs:
            for name,data in outputs.items():util.need((p/name).read_bytes()==data,'stage drift: '+name)
    return dict(status='HOLD_NOT_ADMITTED' if command=='admit' else 'PASS_CPU_'+command.upper(),
        stages=[str(p.relative_to(ROOT)) for _,p,_,_ in pairs],dry_run=dry_run,qualified_for_submission=False,
        remaining=['independent review','candidate visible and N32 execution','U admission + canonical submit-once'] if command=='admit' else [])
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','prepare','validate','admit']);p.add_argument('--dry-run',action='store_true');a=p.parse_args();print(json.dumps(run(a.command,a.dry_run),indent=2))
if __name__=='__main__':main()
