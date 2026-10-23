"""v1.0.0: one fixed P25-V10+A5 candidate and its never-submit N32 sibling.

Changelog: new generator over the exact materialized P25-V10 parent. Legacy
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
PARENT = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.79-r6w958k1n3s2p25v10'
RELEASE = '1.1.80-r6w958k1n3s2p25a5v10'
SLUG = 'taeyangg4/knee-lb-r6w958k1n3s2p25a5v10-v1'
MODULES = ['tools/p25_a5_model_v1.py', 'tools/p25_a5_render_v1.py', 'tools/p25_a5_runtime_v1.py']
RUNTIME = 'tools/k1sr_runtime_v10.py'
CODEC = 'exp169_release_c85r7codec.py'
SCRATCH = ROOT/'.local/scratch/astra-b1/a5-member-v1'


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


def runtime_patch(data, pc, c):
    text = data.decode()
    text = exact(text, 'REGISTERED = '+repr(pc), 'REGISTERED = '+repr(c))
    text = exact(text, 'runtime v10.0.0.', 'runtime v10.0.1-a5.\n\nChangelog: authenticated Pprime input; K1 arithmetic unchanged.')
    text = exact(text, "VERSION = '10.0.0'", "VERSION = '10.0.1-a5'")
    text = exact(text, 'def finish(output, parent_run, config, parent_seconds):',
                 'def finish(output, parent_run, config, parent_seconds, parent_override=None):')
    anchor = "    util.write_once(output / 'k1sr-parent.csv', parent)"
    replacement = '''    w.need(parent_override is not None, 'A5 Pprime required')
    a5_receipt = json.loads((output / 'a5-receipt.json').read_bytes())
    w.need(a5_receipt.get('status') == 'COMPLETE_A5' and
           a5_receipt.get('original_p_sha256') == hashlib.sha256(parent).hexdigest() and
           a5_receipt.get('pprime_sha256') == hashlib.sha256(parent_override).hexdigest(),
           'A5 Pprime receipt binding')
    w.need(util.table(parent)[0] == util.table(parent_override)[0], 'A5 Pprime roster')
    parent = parent_override
    util.write_once(output / 'k1sr-parent.csv', parent)'''
    text = exact(text, anchor, replacement)
    text = exact(text, "parent_seconds_scope='parent pipeline wall including in-pass K1'",
                 "parent_seconds_scope='parent pipeline including in-pass K1 plus A5 phase'")
    text = exact(text, "    rec['parent_mode'] = 'INF3_P_R224' if REGISTERED['drop_o'] else 'INF3'",
                 "    rec['parent_mode'] = 'P25_A5_PPRIME'\n    rec['a5_receipt_sha256'] = w.sha(output / 'a5-receipt.json')")
    return text.encode()


def artifacts(diagnostic=False):
    pc = json.loads((PARENT/'config.json').read_bytes())
    # Full legacy regeneration is deliberately part of every CPU command.
    parent_outputs, parent_reg = legacy.artifacts(pc, diagnostic)
    parent_stage = legacy.stage(pc, diagnostic)
    util.need({p.relative_to(parent_stage).as_posix() for p in parent_stage.rglob('*') if p.is_file()}
              == set(parent_outputs), 'P25 parent inventory drift')
    for name,data in parent_outputs.items():
        util.need((parent_stage/name).read_bytes() == data, 'P25 parent byte drift: '+name)
    old_nb = json.loads(parent_outputs['kernel/release.ipynb'])
    old_meta = json.loads(parent_outputs['kernel/kernel-metadata.json'])
    code = ''.join(old_nb['cells'][2]['source'])
    engine = v10.replay.load()
    blob, pins, old = engine.before.parent.archive(code)
    pc = legacy.canonical(pc)
    c = dict(pc, release_version=RELEASE, kernel_id=SLUG)
    oc, cfg = legacy.config(pc, diagnostic), legacy.config(c, diagnostic)
    cfg['formula'] = a5.FORMULA+'; inherited outer K1: '+cfg['formula']
    cfg['a5_contract_sha256'] = digest(util.encoded(contract()))
    new = dict(old)
    new[RUNTIME] = runtime_patch(old[RUNTIME], pc, c)
    codec = old[CODEC].decode()
    util.need(codec.count(oc['release_version']) == 2, 'codec identity anchors')
    codec = codec.replace(oc['release_version'], cfg['release_version'])
    new[CODEC] = exact(codec, oc['kernel_id'], cfg['kernel_id']).encode()
    for name in MODULES:
        new[name] = (ROOT/name).read_bytes()
    packed = engine.before.packed(new)
    for before,after in [
        (repr(base64.b64encode(blob).decode()),repr(base64.b64encode(packed).decode())),
        ('PINS = '+repr(pins),'PINS = '+repr({n:digest(v) for n,v in new.items()})),
        (repr(digest(blob)),repr(digest(packed))), (repr(oc),repr(cfg)),
        ("    _result = _k1sr.finish(Path('/kaggle/working'), Path('/kaggle/working') / release.RUN,",
         "    from tools import p25_a5_runtime_v1 as _a5\n"
         "    _pprime = _a5.run(Path('/kaggle/working'), release.competition_root(),\n"
         "        release.codec_receipt(Path('/kaggle/working') / release.RUN), diagnostic="+repr(diagnostic)+")\n"
         "    _result = _k1sr.finish(Path('/kaggle/working'), Path('/kaggle/working') / release.RUN,"),
        ('time.perf_counter() - _parent_start)', 'time.perf_counter() - _parent_start, parent_override=_pprime)'),
    ]:
        code = exact(code,before,after)
    nb,meta = copy.deepcopy(old_nb),copy.deepcopy(old_meta)
    nb['cells'][2]['source'] = code.splitlines(True)
    nb['cells'][1]['source'] = ['# P25-V10 plus fixed fivefold A5 P member\n', cfg['formula']+'\n',
        'N32 DIAGNOSTIC: NEVER SUBMIT.\n' if diagnostic else
        'CPU build only. Independent review and actual trained T4 qualification required.\n']
    meta.update(id=cfg['kernel_id'], title=cfg['kernel_id'].split('/')[1],
                dataset_sources=old_meta['dataset_sources']+[a5.DATASET])
    changed = sorted(n for n in old if old[n] != new[n])
    util.need(changed == sorted([RUNTIME,CODEC]) and set(new)-set(old) == set(MODULES), 'A5 payload scope')
    util.need(engine.before.parent.archive(code)[2] == new, 'A5 archive roundtrip')
    for name,data in new.items():
        if name.endswith('.py'): compile(data,name,'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code': compile(''.join(cell['source']),'<A5 notebook>','exec')
    # Mechanically prove every existing K1 function except finish is identical.
    def functions(data):
        tree = ast.parse(data)
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old_functions, new_functions = functions(old[RUNTIME]), functions(new[RUNTIME])
    util.need(old_functions.keys() == new_functions.keys() and
              [n for n in old_functions if old_functions[n] != new_functions[n]] == ['finish'],
              'K1 function drift outside finish input/receipt')
    scope = dict(status='PASS_CPU_SCOPE', parent_stage=parent_stage.relative_to(ROOT).as_posix(),
        changed=changed, added=MODULES, removed=[],
        untouched={n:digest(v) for n,v in old.items() if new[n] == v},
        config_changed_fields=['kernel_id','release_version','a5'],
        metadata_changed_fields=['id','title','dataset_sources'],
        k1_functions_identical_except_finish=True, parent_launch_pool_threads_identical=True,
        a5_execution='new inline phase after parent; original anchor two GPU owners/four decode threads',
        projection_scope='parent wall including K1 plus A5 wall',
        parent_artifacts_byte_identical=True)
    full_config = dict(c, a5=contract())
    outputs = {'kernel/release.ipynb':util.encoded(nb), 'kernel/kernel-metadata.json':util.encoded(meta),
               'config.json':util.encoded(full_config), 'payload-scope.json':util.encoded(scope),
               'CHANGELOG.md':('# P25+A5 generator v1.0.0\n\n'+a5.FORMULA+
                   '\nOnly the registered A5 member and identity/receipt bindings change.\n'
                   'Inherited timm 1.0.25 vs anchor 1.0.22: trained parity unverified.\n'
                   'No submission admission. N32 is NEVER SUBMIT.\n').encode()}
    pairs = [(n,old.get(n,b''),new[n]) for n in sorted(new) if old.get(n) != new[n]]
    # Full actual notebook diff includes the complete embedded archive/pins.
    pairs += [(n,util.encoded(a),util.encoded(z)) for n,a,z in
              [('notebook',old_nb,nb),('metadata',old_meta,meta),('config',pc,full_config)]]
    diff = []
    for name,a,z in pairs:
        diff.extend(difflib.unified_diff(a.decode().splitlines(True),z.decode().splitlines(True),
                    fromfile='P25-V10/'+name,tofile='P25-A5/'+name))
    outputs['full-payload-notebook.diff'] = ''.join(diff).encode()
    fields = [{'field':k,'parent':pc.get(k),'candidate':full_config.get(k)}
              for k in sorted(pc.keys()|full_config.keys()) if pc.get(k) != full_config.get(k)]
    outputs['field-level-diff.json'] = util.encoded(fields)
    reg = dict(schema='p25-a5-registration/v1', **cfg, stage=stage(diagnostic).relative_to(ROOT).as_posix(),
        builder_version=VERSION, parent_registration_sha256=digest(parent_outputs['registration.json']),
        a5_contract=contract(), sources={n:digest(v) for n,v in new.items()},
        builder_sources={n:util.sha(ROOT/n) for n in [*MODULES,'tools/p25_a5_stage_v1.py']},
        kernel_sha256=digest(outputs['kernel/release.ipynb']),
        metadata_sha256=digest(outputs['kernel/kernel-metadata.json']),
        independent_review='REQUIRED', execution_verified=False, qualified_for_submission=False,
        clean_oof=False, exposure='EXPOSED_EXPLORATORY')
    outputs['registration.json'] = util.encoded(reg)
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
                      'A5 stage inventory drift')
            for name,data in outputs.items():
                util.need((path/name).read_bytes() == data, 'A5 stage byte drift: '+name)
    result = dict(status='PASS_CPU_'+command.upper(), dry_run=dry_run,
                  stages=[dict(stage=str(path.relative_to(ROOT)), kernel_id=reg['kernel_id'],
                               kernel_sha256=reg['kernel_sha256'],never_submit=d)
                          for d,path,_,reg in pairs], qualified_for_submission=False)
    if command == 'admit':
        util.need(dry_run, 'admission execution is not supported by this CPU builder')
        result.update(status='HOLD_NOT_ADMITTED', reasons=[
            'Independent review pending', 'Original trained checkpoint CPU forward unavailable locally',
            'Anchor timm 1.0.22 versus inherited 1.0.25 trained parity unverified',
            'New candidate T4x2/N32 output, memory and <=1.5h additional runtime qualification pending'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['register','prepare','validate','admit'])
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.command,args.dry_run),indent=2))


if __name__ == '__main__':
    main()
