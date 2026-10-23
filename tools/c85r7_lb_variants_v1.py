"""Offline sealed LB hypotheses v1.0.0; never uploads or launches a GPU.

Changelog 2026-10-14: composition15, own25, fluidR and R2; frozen parents intact.
"""
from __future__ import annotations
import argparse
import ast
import base64
import difflib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from tools import exp169_release_c85r7codec_deadline_v122 as b
from tools import hb127_vnextr7_release as n
h = n.h
SCRATCH = ROOT/'.local/scratch/astra-b1/c85r7-lb-variants'
BACKUP = ROOT/'.local/backups/astra-b1/c85r7-lb-variants'
VERSION = '1.0.0'
R2_DATASET = 'taeyangg4/knee-hb113-rplus-weights-v2'
ORIGINAL_O = 'O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46'
SPECS = {
 'compo15': dict(version='1.3.0-compo15', root='HB116_VNEXT', parent='artifacts/experiments/HB116_VNEXT/v1.2.0/release/kernels/a',
    hypothesis='Upgraded own composition at fixed own weight 0.15',
    formula='0.85*r(P)+0.15*r(O_A); P=0.5*r(R)+0.5*r(R7); OWN2=mean_prob(EXP143v2,S43); O_A=(17*r(OWN2)+17*r(EXP155)+12*r(Q7S))/46; Q7S=mean_rank(R50,C4,S)', hours=7.4),
 'own25': dict(version='1.6.0-own25', root='EXP169_RELEASE_C85R7', parent='artifacts/experiments/EXP169_RELEASE_C85R7/v1.2.2/kernel',
    hypothesis='Increase original own weight to 0.25', formula='0.75*r(P)+0.25*r(O); P=0.5*r(R)+0.5*r(R7); '+ORIGINAL_O, hours=4.7),
 'fluidR': dict(version='1.7.0-fluidR', root='EXP169_RELEASE_C85R7', parent='artifacts/experiments/EXP169_RELEASE_C85R7/v1.2.2/kernel',
    hypothesis="Use R alone in P for Effusion and Baker's Cyst", formula="0.85*r(P)+0.15*r(O); P_t=r(R)_t for Effusion and Baker's Cyst; otherwise P=0.5*r(R)+0.5*r(R7); "+ORIGINAL_O, hours=4.7),
 'r2': dict(version='1.8.0-r2', root='EXP169_RELEASE_C85R7', parent='artifacts/experiments/EXP169_RELEASE_C85R7/v1.2.2/kernel',
    hypothesis='Replace public R with two public fine-tuned R+ seeds', formula='0.85*r(P)+0.15*r(O); P=0.5*r(R2)+0.5*r(R7); R2=mean_prob(R+s43,R+s44) in float32; '+ORIGINAL_O, hours=7.4),
}

def digest(data): return hashlib.sha256(data).hexdigest()
def stage(tag):
    s = SPECS[tag]
    return ROOT/'artifacts/experiments'/s['root']/('v'+s['version'])
def dump(path, value):
    b.metadata_io.atomic_write(Path(path), (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())
def replace(source, old, new):
    b.need(source.count(old)==1, 'source anchor drift: '+old[:80])
    return source.replace(old,new)

def math_source(weight=.15, fluid=False):
    """Frozen arithmetic with exactly the registered coefficient/routing delta."""
    source = inspect.getsource(b.c85.blend)
    source = replace(source, 'zip(MEMBERS, (r, ours, exp143, exp155, q))', "zip(['R','S_R7','EXP143v2','EXP155','Q_A5q'], (r, ours, exp143, exp155, q))")
    source = replace(source, 'for name in MEMBERS]', "for name in ['R','S_R7','EXP143v2','EXP155','Q_A5q']]")
    source = replace(source, 'result = .85 * pr + .15 * oo', f'result = {1-weight!r} * pr + {weight!r} * oo')
    if fluid:
        # Recompute only the two affected columns; preserve per-target survivor policy.
        source = replace(source, '    return result', '''    routed = .85 * rr + .15 * oo
    routed[of], routed[rf], routed[rf & of] = rr[of], oo[rf], .5
    result[:, [7, 9]] = routed[:, [7, 9]]
    return result''')
    ref = inspect.getsource(b.c85.reference)
    if fluid:
        ref = replace(ref, '    return .85 * rank(.5 * rank(r) + .5 * rank(ours)) + .15 * rank(', '''    public = .5 * rank(r) + .5 * rank(ours)
    public[:, [7,9]] = rank(r)[:, [7,9]]
    return .85 * rank(public) + .15 * rank(''')
    else:
        ref = replace(ref, 'return .85 * rank', f'return {1-weight!r} * rank')
        ref = replace(ref, '+ .15 * rank(', f'+ {weight!r} * rank(')
    return 'scored_rank = c85.scored_rank\n'+source+'\n'+ref

def c85_source(tag):
    s = SPECS[tag]
    source = Path(b.__file__).read_text(encoding='utf-8')
    module = 'c85r7_lb_'+tag.lower()+'_runtime'
    source = replace(source, "MODULE = 'exp169_release_c85r7codec_deadline_v122'", f'MODULE = {module!r}')
    source = replace(source, "VERSION, RELEASE = '1.2.2', 'C85R7+codecs+insurance'", f"VERSION, RELEASE = {s['version']!r}, {'C85R7-LB-'+tag!r}")
    if tag in ('fluidR', 'r2'):
        branch_formula = ('r(R) for Effusion/Bakers; .5 r(R)+.5 r(S_R7) otherwise'
                          if tag == 'fluidR' else '.5 r(R2)+.5 r(S_R7); R2=float32 mean_prob(s43,s44)')
        source = replace(source, 'public cell 7: .5 r(R)+.5 r(S_R7), each over own non-failed rows; float32',
                         branch_formula + '; each over own non-failed rows; float32')
    inject = '\nFORMULA = '+repr(s['formula'])+'\n'+math_source(.25 if tag=='own25' else .15,tag=='fluidR')
    if tag=='fluidR':
        inject += '''\n_original_p_branch = p_branch
def p_branch(r, ours, failed):
    import numpy as np
    value = _original_p_branch(r, ours, failed)
    rr = c85.scored_rank(r, np.all(r == .5, axis=1))
    value[:, [7,9]] = rr[:, [7,9]]
    return value
'''
    if tag=='r2':
        inject += '\nfrom tools import c85r7_lb_r2_adapter_v1 as r2_adapter\n'
        source = replace(source, "        weights = v5q._mount(p.DATASETS[0], p.R_NAME) / p.R_NAME\n        c85.run_r(runtime, weights, competition_root(), output / RUN / 'pr7/pr7-run/r')",
            "        r2_adapter.run(runtime, competition_root(), output / RUN / 'pr7/pr7-run/r2')")
        source = replace(source, "            need(base.preflight(runtime, output) == ids, 'setup cohort changed')",
            "            need(base.preflight(runtime, output) == ids, 'setup cohort changed')\n            r2_adapter.preflight(runtime)")
        source = replace(source, '    a, b = metadata_io.collectors(sys.modules[__name__], cohort)',
            '''    a, original_pr7 = metadata_io.collectors(sys.modules[__name__], cohort)
    def b(directory):
        ids, _, ours, info = original_pr7(directory)
        r, failures, identity = r2_adapter.recover(directory / 'pr7-run/r2', ids)
        info['failures']['R'] = failures
        info['r_prediction_fallbacks'] = len(failures)
        info['r2_identity'] = identity
        info['executed_arms'] = ['R2-s43','R2-s44','S_R7']
        return ids, r, ours, info''')
    source = replace(source, '\ndef main():', inject+'\ndef main():')
    source = replace(source, '    return receipt\n\n\ndef failed_execution',
        "    receipt.update(formula=FORMULA, exposure='EXPOSED', clean_oof=False, hypothesis="+repr(s['hypothesis'])+")\n"+
        ("    receipt['intended_members'] = receipt['executed_members'] = ['R2-s43','R2-s44','S_R7','EXP143v2','EXP155','Q_A5q']\n    receipt['execution_passes'][1] = dict(serving='HB116 frozen native R2 + R7', arms=['R2-s43','R2-s44','S_R7'])\n" if tag=='r2' else '')+
        '    return receipt\n\n\ndef failed_execution')
    return module,source

def composition_source():
    source = Path(n.__file__).read_text(encoding='utf-8')
    source = source.replace("'R2'", "'R'")
    source = source.replace('NATIVE_O_R2', 'NATIVE_O_R').replace('R2 remains in fused', 'R remains in fused')
    source = replace(source, "candidate in h.CONSTANTS and (not stress or candidate == 'A')", "candidate == 'A' and not stress")
    source = replace(source, "VERSION = '1.2.0'", "VERSION = '1.3.0-compo15'")
    source = replace(source, "    return h.blend(reg, mapped, candidate, 'R', True)", '''    import numpy as np
    _, scores = h.blend(reg, mapped, candidate, 'R', True)
    final = h.mean_survivors([h.ranks(scores['P']), h.ranks(scores['O'])], [.85,.15])
    final[~np.isfinite(final).all(axis=1)] = .5
    return final.astype(np.float32), scores''')
    source = replace(source, 'return dict(version=VERSION, candidate=candidate, constants=h.CONSTANTS[candidate],',
        'return dict(version=VERSION, outer=[.85,.15], exposure="EXPOSED", clean_oof=False, candidate=candidate, constants=h.CONSTANTS[candidate],')
    source = replace(source, '    values, scores = blend(reg, raw, candidate)',
        "    need(candidate == 'A', 'sealed composition A only')\n    values, scores = blend(reg, raw, candidate)")
    source = replace(source, '    dump(output/\'hb127-receipt.json\', rec)',
        "    rec.update(formula="+repr(SPECS['compo15']['formula'])+", exposure='EXPOSED', clean_oof=False)\n    dump(output/'hb127-receipt.json', rec)")
    return 'c85r7_lb_compo15_runtime',source

def payload(tag):
    s = SPECS[tag]
    meta = h.read(ROOT/s['parent']/'kernel-metadata.json')
    meta.update(id='taeyangg4/knee-lb-'+tag+'-v1',title='knee-lb-'+tag+'-v1')
    if tag=='compo15':
        files = n.source_payloads()
        module,source = composition_source()
        files['tools/'+module+'.py']=source.encode()
        cfg = dict(n.config(),version=s['version'],rx='R',outer=[.85,.15],exposure='EXPOSED',clean_oof=False)
        files['build-config.json']=(json.dumps(cfg,sort_keys=True)+'\n').encode()
        blob,pins = h.package(files)
        nb=h.notebook(blob,pins,cfg)
        code=''.join(nb['cells'][-1]['source']).replace('from tools import hb116_vnext_release as release',f'from tools import {module} as release')
        code=code.replace(",rx='R',s43=True",'')
        code=code.replace('hb116-runtime','lb-compo15-runtime')
        meta['dataset_sources']=list(meta['dataset_sources'])+[b.p.DATASETS[0]]
        parent_source=Path(n.__file__)
    else:
        files=b.source_payloads()
        module,source=c85_source(tag)
        files[module+'.py']=source.encode()
        if tag=='r2':
            files.update(h.source_payloads())
            files['tools/c85r7_lb_r2_adapter_v1.py']=(ROOT/'tools/c85r7_lb_r2_adapter_v1.py').read_bytes()
            files['r2-own-manifest.json']=(h.OWN_DIR/'release-manifest.json').read_bytes()
            h.pin(h.OWN_DIR/'release-manifest.json',{'sha256':h.OWN_MANIFEST_SHA})
            meta['dataset_sources']=list(meta['dataset_sources'])+[R2_DATASET]
        blob,pins=h.package(files)
        nb=b.notebook_payload(blob,pins)
        code=''.join(nb['cells'][-1]['source']).replace('import '+b.MODULE+' as release','import '+module+' as release')
        code=code.replace(b.RUNTIME_DIR,'/kaggle/working/lb-'+tag.lower()+'-runtime')
        parent_source=Path(b.__file__)
    compile(source,module,'exec'); compile(code,'<notebook>','exec')
    nb['cells'][-1]['source']=code.splitlines(True)
    nb['cells'][0]['source']=[f"# {tag} {s['version']}\n{s['hypothesis']}\n{s['formula']}\nPrivate/offline. EXPOSED; clean_oof=false. Credits: nartaa public R; heliosli R7; frozen parent sources retain provenance.\n"]
    return files,nb,meta,parent_source,source

def launch(tag):
    s=SPECS[tag]; path=stage(tag).relative_to(ROOT).as_posix(); slug='taeyangg4/knee-lb-'+tag+'-v1'
    return f'''# {tag} v{s['version']}
{s['hypothesis']}. Formula: `{s['formula']}`.
Sealed before results; r means average rank / successful cohort size. EXPOSED; clean_oof=false.
Parent: `{s['parent']}`. No N32 kernel. B performs the mandatory private visible save/run and inspects its CSV/member receipt before submitting once.

## Offline CPU commands (run by builder)
```powershell
& C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe -B .local/scratch/astra-b1/c85r7-lb-variants/cpu.py tools/c85r7_lb_variants_v1.py validate --tag {tag}
& C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe -B .local/scratch/astra-b1/c85r7-lb-variants/cpu.py tools/c85r7_lb_variants_v1.py replay --tag {tag}
```
## B only: push, then submit after successful visible run
```powershell
kaggle kernels push -p {path}/kernel
.venv/Scripts/python.exe tools/kaggle_submit_once.py --id lb-{tag}-v1 --kernel {slug} --version 1 --message "{s['hypothesis']}; {s['formula']}"
```
The submit command is for the first pushed version; if B creates another version, use its verified version number. Never retry a submit blindly.

## Runtime and scope
Planning projection: {s['hours']:.1f} h compute + 20 min publication reserve = {s['hours']+1/3:.3f} h versus Kaggle's 9 h; headroom {9-s['hours']-1/3:.3f} h. This is a projection, not a measurement of this kernel. V3/V4 inherit the 4.7 h C85-R7 budget; V4 adds no inference. V2 uses the hidden-proven native <=7.4 h envelope with fewer Rx members. V5 conservatively budgets the <=7.4 h native R2 envelope versus the 4.7 h baseline (2.7 h allowance); original Q differs, so this is not a measured upper bound. v1.2.2's 8 h 40 min compute deadline and 20 min publication reserve remain for V3-V5.
Metadata preserves parent's private/offline/image/machine/competition policy and existing sources. V2 additionally mounts the requested original R dataset; V5 additionally mounts the requested R+ dataset. No dataset uploads.
CPU evidence replays authenticated completed predictions on the actual three visible studies, not labels or a new neural inference benchmark. New math, ordered CSV and no-op parent byte parity are checked. Target CUDA/codec/runtime confirmation belongs to B's visible run. Frozen native/insurance operational behavior is inherited; this minimal hypothesis build does not claim new desktop watchdog or VRAM qualification.
'''

def prepare():
    BACKUP.mkdir(parents=True,exist_ok=True)
    for tag,s in SPECS.items():
        out=stage(tag)
        b.need(not out.exists(),'new stages only: '+str(out))
        files,nb,meta,parent,source=payload(tag)
        # Snapshot parents without mutating any original or scored receipt.
        (BACKUP/parent.name).write_bytes(parent.read_bytes())
        parent_meta=ROOT/s['parent']/'kernel-metadata.json'
        (BACKUP/(tag+'-parent-metadata.json')).write_bytes(parent_meta.read_bytes())
        out.mkdir(parents=True)
        dump(out/'kernel/release.ipynb',nb); dump(out/'kernel/kernel-metadata.json',meta)
        pins={k:digest(v) for k,v in files.items()}
        dump(out/'source-pins.json',pins)
        dump(out/'registration.json',dict(version=s['version'],tag=tag,hypothesis=s['hypothesis'],formula=s['formula'],
            exposure='EXPOSED',clean_oof=False,parent_source=str(parent.relative_to(ROOT)),parent_sha256=h.sha(parent),
            projected_compute_hours=s['hours'],reserve_hours=1/3,limit_hours=9,checkpoint_sources=meta['dataset_sources'],
            source_pins=pins,selection='Each independent hypothesis sealed; B evaluates once on public LB',qualified_for_submission=False))
        (out/'CHANGELOG.md').write_text(f"# v{s['version']} — 2026-10-14\n{s['hypothesis']}. {s['formula']}.\nNew stage; frozen parent unchanged. Backup: {BACKUP.relative_to(ROOT).as_posix()}.\n",encoding='utf-8')
        (out/'LAUNCH.md').write_text(launch(tag),encoding='utf-8')
        (out/'parent-runtime.diff').write_text(''.join(difflib.unified_diff(parent.read_text(encoding='utf-8').splitlines(True),source.splitlines(True),fromfile=parent.name,tofile='generated-runtime.py')),encoding='utf-8')
        dump(out/'seal.json',{p.relative_to(out).as_posix():h.sha(p) for p in out.rglob('*') if p.is_file()})
    return dict(status='PREPARED',kernels=4,partial_previous_attempt='none found; nothing removed')

def validate(tag):
    out=stage(tag); reg=h.read(out/'registration.json')
    h.pin(ROOT/reg['parent_source'],{'sha256':reg['parent_sha256']})
    for name,sha in h.read(out/'seal.json').items(): h.pin(out/name,{'sha256':sha})
    files,nb,meta,_,_=payload(tag)
    b.need(h.read(out/'kernel/release.ipynb')==nb,'notebook drift')
    b.need(h.read(out/'kernel/kernel-metadata.json')==meta,'metadata drift')
    b.need(meta['is_private'] is True and meta['enable_internet'] is False,'private offline')
    b.need(reg['source_pins']=={k:digest(v) for k,v in files.items()},'source drift')
    b.need(reg['projected_compute_hours']+reg['reserve_hours']<reg['limit_hours'],'runtime planning gate')
    return dict(status='PASS_VALIDATE',tag=tag,files=len(files),parent_unchanged=True,private=True,offline=True)

def replay(tag):
    from tools import c85r7_lb_variants_verify_v1 as verify
    return verify.replay(tag)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','validate','replay'])
    parser.add_argument('--tag',choices=SPECS)
    args=parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if args.command=='prepare': result=prepare()
    else:
        b.need(args.tag,'--tag required')
        result=globals()[args.command](args.tag)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
