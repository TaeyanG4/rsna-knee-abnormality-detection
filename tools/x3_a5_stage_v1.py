"""v1.0.0: one fixed X3-INF3-V10+A5 candidate and its never-submit N32 sibling.

Changelog: new generator over the exact materialized X3-v2 runtime-v5 parent. Legacy
generators are neither modified nor dispatched through this extension.
"""
from __future__ import annotations
import argparse
import ast
import base64
import copy
import difflib
import hashlib
import json
from pathlib import Path

from tools import k1n_stage_v1 as legacy
from tools import k1n_stage_v10 as v10
from tools import p25_a5_runtime_v1 as a5

VERSION = '1.0.0'
ROOT, util, exact = v10.ROOT, v10.util, v10.exact
PARENT = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.62-r6w958k1x3sr20'
RELEASE = '1.1.83-r6w958k1x3a5v10'
SLUG = 'taeyangg4/knee-lb-r6w958k1x3a5v10-v1'
MODULES = ['tools/p25_a5_model_v1.py', 'tools/p25_a5_render_v1.py', 'tools/p25_a5_runtime_v1.py']
RUNTIME = 'tools/k1sr_runtime_v10.py'
CODEC = 'exp169_release_c85r7codec.py'
SCRATCH = ROOT/'.local/scratch/astra-b5/x3a5-v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stage(diagnostic=False):
    return PARENT.parent/('v'+RELEASE+('-n32' if diagnostic else ''))


def contract():
    return dict(schema='p25-a5-contract/v1', dataset=a5.DATASET, weight=.15,
                formula=a5.FORMULA, checkpoint_sha256=list(a5.HASHES), model=a5.CFG,
                folds=list(range(5)), anchor_timm='1.0.22', inherited_timm='1.0.25',
                missing_acquisition='parent rank before outer Pprime rank',
                operational_failure='fatal', additional_budget_hours=1.5,
                anchor100_seconds=87.74157786369324, hidden_studies_estimate=1300,
                runtime_estimate_hours=87.74157786369324/100*1300/3600,
                trained_numerical_parity_verified=False)


A5_STAGE = PARENT.parent/'v1.1.80-r6w958k1n3s2p25a5v10'
INTEGRATION = 'tools/x3_a5_integration_v1.py'


def runtime_patch(data, pc, c):
    text = exact(data.decode(), 'REGISTERED = '+repr(pc), 'REGISTERED = '+repr(c))
    text = exact(text, "VERSION = '10.0.1-a5'", "VERSION = '10.0.2-x3a5'")
    begin = text.index("    w.need(parent_override is not None")
    end = text.index("    util.write_once(output / 'k1sr-parent.csv', parent)", begin)
    text = text[:begin] + """    w.need(parent_override is not None, 'INF3 A5 input required')
    from tools import x3_a5_integration_v1 as integration
    checked = integration.compose(output, (output/'a5-pprime.csv').read_bytes(), diagnostic, verify_only=True)
    w.need(checked == parent_override, 'INF3 A5 finish binding')
    parent = parent_override
""" + text[end:]
    text = exact(text, "rec['parent_mode'] = 'P25_A5_PPRIME'", "rec['parent_mode'] = 'INF3_A5_PPRIME'")
    text = exact(text, "    rec['a5_receipt_sha256'] = w.sha(output / 'a5-receipt.json')",
        "    rec['a5_receipt_sha256'] = w.sha(output / 'a5-receipt.json')\n"
        "    rec['inf3_a5_receipt_sha256'] = w.sha(output / 'x3-a5-inf3-receipt.json')")
    return text.encode()


def artifacts(diagnostic=False):
    from tools import c85r7_r6w958k1x3sr_v1162 as parent
    from tools import k1n_config_v1 as schema
    from tools import x3_a5_integration_v1 as integration
    parent_stage = Path(str(PARENT)+('-n32' if diagnostic else ''))
    old_nb = json.loads((parent_stage/'kernel/release.ipynb').read_bytes())
    old_meta = json.loads((parent_stage/'kernel/kernel-metadata.json').read_bytes())
    preg = json.loads((parent_stage/'registration.json').read_bytes())
    util.need(util.sha(parent_stage/'kernel/release.ipynb') == preg['kernel_sha256'], 'exact parent notebook')
    util.need(util.sha(parent_stage/'kernel/kernel-metadata.json') == preg['metadata_sha256'], 'exact parent metadata')
    code = ''.join(old_nb['cells'][2]['source'])
    blob,pins,old = parent.parent.archive(code)
    an = json.loads((A5_STAGE/'kernel/release.ipynb').read_bytes())
    _,_,aold = parent.parent.archive(''.join(an['cells'][2]['source']))
    pc = json.loads((A5_STAGE/'config.json').read_bytes()); pc.pop('a5')
    pc = legacy.canonical(pc)
    c = schema.current()
    c.update(release_version=RELEASE, kernel_id=SLUG, k1_failure_policy='study_fallback_fail_loud',
        k1_max_failed_fraction=.05,k1_guard_hours=8.8,k1_runtime_version=10,k1_eval_stride=3)
    c = schema.validate(c)
    from tools import k1n_compat_v1 as compat
    core = {k:v for k,v in c.items() if k not in ('k1_failure_policy','k1_max_failed_fraction','k1_guard_hours','k1_runtime_version')}
    cfg = compat.load('v120').config(core,diagnostic)
    cfg.update(config_sha256=digest(util.encoded(c)), registered_config_sha256=digest(util.encoded(c)),
        **{k:c[k] for k in ('k1_failure_policy','k1_max_failed_fraction','k1_guard_hours','k1_runtime_version')})
    cfg['formula'] += '; failed-study K1 rank input = parent rank before outer rank; systemic failure or failed fraction above registered limit raises after publication'+v10.TIMEOUT_FORMULA
    cfg['formula'] = a5.FORMULA+'; inherited outer K1: '+integration.FORMULA+'; '+cfg['formula']
    cfg['a5_contract_sha256'] = digest(util.encoded(contract()))
    new = dict(old)
    del new['tools/k1sr_runtime_v5.py']
    new[RUNTIME] = runtime_patch(aold[RUNTIME],pc,c)
    for n in MODULES+['tools/k1n_config_v1.py','tools/k1_shared_render_v2.py']:
        new[n] = aold[n]
    new[INTEGRATION] = (ROOT/INTEGRATION).read_bytes()
    port,release = 'exp169_r7_port.py','exp169_release_c85r7.py'
    new[port] = exact(old[port].decode(),'k1sr_runtime_v5','k1sr_runtime_v10').encode()
    new[release] = exact(old[release].decode(),digest(old[port]),digest(new[port])).encode()
    codec = exact(old[CODEC].decode(),digest(old[release]),digest(new[release]))
    util.need(codec.count(preg['release_version']) == 2, 'parent codec version anchors')
    codec = codec.replace(preg['release_version'],cfg['release_version'])
    codec = exact(codec,preg['kernel_id'],cfg['kernel_id'])
    anchor = '        receipt = candidate_receipt(ids, a, b, payload, codec_summary(output))'
    codec = exact(codec,anchor,anchor+"\n        from tools import x3_a5_integration_v1 as x3a5\n"
        '        receipt.update(x3a5.snapshot(r224, output, ids, (r, r224_values, ours, exp143, exp155, q),\n'
        '            {**a["failures"], **b["failures"]}, payload))')
    new[CODEC] = codec.encode()
    packed = parent.packed(new)
    oc = parent.config('k1x3sr20',diagnostic)
    for before,after in [
        (repr(base64.b64encode(blob).decode()),repr(base64.b64encode(packed).decode())),
        ('PINS = '+repr(pins),'PINS = '+repr({n:digest(v) for n,v in new.items()})),
        (repr(digest(blob)),repr(digest(packed))), (repr(oc),repr(cfg)),
        ('from tools import k1sr_runtime_v5 as _k1sr','from tools import k1sr_runtime_v10 as _k1sr'),
        ("    _result = _k1sr.finish(Path('/kaggle/working'), Path('/kaggle/working') / release.RUN,",
         "    from tools import x3_a5_integration_v1 as _x3a5\n"
         "    _inf3 = _x3a5.run(Path('/kaggle/working'), release.competition_root(),\n"
         "        release.codec_receipt(Path('/kaggle/working') / release.RUN), diagnostic="+repr(diagnostic)+")\n"
         "    _result = _k1sr.finish(Path('/kaggle/working'), Path('/kaggle/working') / release.RUN,"),
        ('time.perf_counter() - _parent_start)', 'time.perf_counter() - _parent_start, parent_override=_inf3)'),
    ]: code = exact(code,before,after)
    nb,meta = copy.deepcopy(old_nb),copy.deepcopy(old_meta)
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = ['# X3 INF3 plus A5 inside P, runtime v10\n',cfg['formula']+'\n',
        'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else 'Speculative; independent review and trained GPU qualification pending.\n']
    meta.update(id=cfg['kernel_id'],title=cfg['kernel_id'].split('/')[1],dataset_sources=old_meta['dataset_sources']+[a5.DATASET])
    util.need(parent.parent.archive(code)[2] == new,'archive roundtrip')
    for n,data in new.items():
        if n.endswith('.py'): compile(data,n,'exec')
    for cell in nb['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']),'<x3a5 notebook>','exec')
    changed=sorted(n for n in old.keys()&new.keys() if old[n]!=new[n])
    added=sorted(new.keys()-old.keys()); removed=sorted(old.keys()-new.keys())
    util.need(changed==sorted([port,release,CODEC]),'unexpected parent payload change')
    util.need(new['exp169_r224.py']==old['exp169_r224.py'],'P and INF3 math unchanged')
    def funcs(data):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(data).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    af,nf=funcs(aold[RUNTIME]),funcs(new[RUNTIME])
    util.need(af.keys()==nf.keys() and [n for n in af if af[n]!=nf[n]]==['finish'],'reviewed v10 functions drift')
    scope=dict(status='PASS_CPU_SCOPE',parent_stage=parent_stage.relative_to(ROOT).as_posix(),
        parent_registration_sha256=util.sha(parent_stage/'registration.json'),changed=changed,added=added,removed=removed,
        untouched={n:digest(v) for n,v in old.items() if n in new and new[n]==v},
        a5_reference_stage=A5_STAGE.relative_to(ROOT).as_posix(),a5_identical={n:digest(new[n]) for n in MODULES},
        runtime_functions_identical_to_reviewed_a5_except_finish=True,
        r6_stride=1,k1_stride=3,k1_alpha=.20,drop_o=False,r224_weight=.35,
        runtime_projection_hours=[8.20,8.58],projection_includes_parent_queue=True,
        speedup_measured=False,qualification='PENDING')
    full_config=dict(c,a5=contract())
    outputs={'kernel/release.ipynb':util.encoded(nb),'kernel/kernel-metadata.json':util.encoded(meta),
        'config.json':util.encoded(full_config),'payload-scope.json':util.encoded(scope),
        'CHANGELOG.md':b'# X3+A5 v1.0.0 / release 1.1.83\n\nExact x3-v2 parent; runtime v5 to reviewed v10 and A5 inside P only.\nNew immutable stage; no original files edited. Trained qualification pending.\n'}
    diff=[]
    pairs=[(n,old.get(n,b''),new.get(n,b'')) for n in sorted(old.keys()|new.keys()) if old.get(n)!=new.get(n)]
    pairs += [('notebook',util.encoded(old_nb),util.encoded(nb)),('metadata',util.encoded(old_meta),util.encoded(meta))]
    for n,a,z in pairs:
        diff.extend(difflib.unified_diff(a.decode().splitlines(True),z.decode().splitlines(True),fromfile='x3-v2/'+n,tofile='x3-a5-v10/'+n))
    outputs['full-payload-notebook.diff']=''.join(diff).encode()
    outputs['vs-reviewed-a5-runtime.diff']=''.join(difflib.unified_diff(aold[RUNTIME].decode().splitlines(True),new[RUNTIME].decode().splitlines(True),fromfile='reviewed-A5/runtime-v10',tofile='x3-A5/runtime-v10')).encode()
    reg=dict(schema='p25-a5-registration/v1',**cfg,stage=stage(diagnostic).relative_to(ROOT).as_posix(),
        builder_version=VERSION,parent_registration_sha256=util.sha(parent_stage/'registration.json'),
        a5_contract=contract(),inf3_a5=True,sources={n:digest(v) for n,v in new.items()},
        builder_sources={n:util.sha(ROOT/n) for n in [INTEGRATION,'tools/x3_a5_stage_v1.py']},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        independent_review='REQUIRED',execution_verified=False,qualified_for_submission=False,
        clean_oof=False,exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json']=util.encoded(reg)
    return outputs,reg


def run(command, dry_run=False):
    pairs = [(d,stage(d),*artifacts(d)) for d in (False,True)]
    if command == 'prepare' and not dry_run:
        util.need(all(not path.exists() for _,path,_,_ in pairs), 'immutable stage exists; validate instead')
        for _,path,outputs,_ in pairs:
            for name,data in outputs.items(): util.write_once(path/name,data)
    if command == 'validate' or (command == 'prepare' and not dry_run):
        for _,path,outputs,_ in pairs:
            util.need({p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()} == set(outputs),
                      'X3 A5 stage inventory drift')
            for name,data in outputs.items():
                util.need((path/name).read_bytes() == data, 'X3 A5 stage byte drift: '+name)
    result = dict(status='PASS_CPU_'+command.upper(), dry_run=dry_run,
                  stages=[dict(stage=str(path.relative_to(ROOT)), kernel_id=reg['kernel_id'],
                               kernel_sha256=reg['kernel_sha256'],never_submit=d)
                          for d,path,_,reg in pairs], qualified_for_submission=False)
    if command == 'admit':
        util.need(dry_run, 'admission execution is not supported by this CPU builder')
        result.update(status='HOLD_NOT_ADMITTED', reasons=[
            'Independent review pending', 'Original trained checkpoint CPU forward unavailable locally',
            'Anchor timm 1.0.22 versus inherited 1.0.25 trained parity unverified',
            'New candidate T4x2/N32 output and memory qualification pending',
            'Estimated 8.20-8.58h incl parent queue: close to 8.8h guard and 9h limit; no measured v10 speedup',
            'Original admission v1.0.5 cannot bind nested INF3; separate v1.0.7-x3 extension requires review'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register','prepare','validate','admit'])
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.command,args.dry_run),indent=2))


if __name__ == '__main__':
    main()
