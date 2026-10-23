"""v2.0.0: restore the exact v1.1.0 own child for four offline rebases, 2026-10-14. Frozen stages unchanged."""
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
from tools import exp169_release_c85r7codec as b
from tools import hb116_vnext_release as h
from tools import c85r7_codec_rebase_io_v1 as metadata_io

VERSION = '2.0.0'
BASE = ROOT/'artifacts/experiments/EXP169_RELEASE_C85R7/v1.1.0'
BASE_SHA = '71580a566ee936569dda69411cf5e952d30cce60069bfec432a6f54fb09d86e9'
SCRATCH = ROOT/'.local/scratch/c85r7-fork-fix/build'
BACKUP = ROOT/'.local/backups/c85r7-fork-fix/v2'
O = 'O=(17*r(EXP143v2)+17*r(EXP155)+12*r(Q_A5q))/46'
SPECS = {
 'flipall': dict(version='1.1.5-flipall',slug='flipall115',hypothesis='Native R7 flip on all four folds; MCL plain only',formula='0.85*r(P)+0.15*r(O); P=0.5*r(R)+0.5*r(R7_FLIP_ALL); '+O,hours=8.8863654887),
 'own25': dict(version='1.1.6-own25',slug='own25c',hypothesis='Increase original own weight to 0.25',formula='0.75*r(P)+0.25*r(O); P=0.5*r(R)+0.5*r(R7); '+O,hours=4.7),
 'fluidR': dict(version='1.1.7-fluidR',slug='fluidRc',hypothesis='Use R alone in P for Effusion and Bakers Cyst',formula='0.85*r(P)+0.15*r(O); P_t=r(R)_t for Effusion and Bakers Cyst; otherwise P=0.5*r(R)+0.5*r(R7); '+O,hours=4.7),
 'r2': dict(version='1.1.8-r2',slug='r2c',hypothesis='Replace R by two public fine-tuned R+ seeds',formula='0.85*r(P)+0.15*r(O); P=0.5*r(R2)+0.5*r(R7); R2=float32 mean_prob(R+s43,R+s44); '+O,hours=7.4),
}

def digest(data): return hashlib.sha256(data).hexdigest()
def stage(tag): return BASE.parent/('v'+SPECS[tag]['version'])
def dump(path, value): metadata_io.atomic_text(path,json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def replace(source,old,new):
    b.need(source.count(old)==1,'source anchor drift: '+old[:100])
    return source.replace(old,new)

def own_path(source):
    """Exact source bytes controlling own code generation and process launch."""
    tree = ast.parse(source)
    names = ('child_base_env', 'a172_script')
    result = {n.name: ast.get_source_segment(source, n) for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names}
    start = source.index('    codec = install_codecs(runtime, output, input_root=input_root)',
                         source.index('def run_release('))
    end = source.index('    c85.phase(pdir, "child"', start)
    result['own_child'] = source[start:end]
    return result

def own_proof(tag, source=None):
    files = frozen_files()
    source = runtime_source(tag, files)[1] if source is None else source
    parent = files['exp169_release_c85r7codec.py'].decode()
    before, after = own_path(parent), own_path(source)
    b.need(before == after, 'own child differs from frozen v1.1.0')
    b.need('c85_own_serving' not in source, 'import-time own wrapper forbidden')
    return dict(status='PASS_EXACT_OWN_PATH', parent='1.1.0',
                source_blocks_sha256={k:digest(v.encode()) for k,v in before.items()},
                frozen_cell_sha256=digest(files['a172-cell.py.txt']),
                frozen_public_serving_sha256=digest(files['public-serving.py']),
                script_and_child_invocation='byte-identical at the same runtime/output paths')

def frozen_files():
    b.p.check_pin(BASE/'build-receipt.json',BASE_SHA)
    rec=h.read(BASE/'build-receipt.json')
    b.p.check_pin(BASE/'kernel/release.ipynb',rec['notebook_sha256'])
    b.p.check_pin(BASE/'kernel/kernel-metadata.json',rec['metadata_sha256'])
    code=''.join(h.read(BASE/'kernel/release.ipynb')['cells'][-1]['source'])
    blob=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BLOB' for t in n.targets))
    raw=base64.b64decode(blob)
    b.need(digest(raw)==rec['archive_sha256'],'parent archive')
    with zipfile.ZipFile(io.BytesIO(raw)) as z: files={n:z.read(n) for n in z.namelist()}
    b.need({k:digest(v) for k,v in files.items()}==rec['files'],'parent members')
    return files

def math_source(weight=.15,fluid=False):
    source=inspect.getsource(b.c85.blend)
    source=replace(source,'zip(MEMBERS, (r, ours, exp143, exp155, q))',"zip(['R','S_R7','EXP143v2','EXP155','Q_A5q'], (r, ours, exp143, exp155, q))")
    source=replace(source,'for name in MEMBERS]',"for name in ['R','S_R7','EXP143v2','EXP155','Q_A5q']]")
    source=replace(source,'result = .85 * pr + .15 * oo',f'result = {1-weight!r} * pr + {weight!r} * oo')
    if fluid:
        source=replace(source,'    return result','''    routed = .85 * rr + .15 * oo
    routed[of], routed[rf], routed[rf & of] = rr[of], oo[rf], .5
    result[:, [7,9]] = routed[:, [7,9]]
    return result''')
    ref=inspect.getsource(b.c85.reference)
    if fluid:
        ref=replace(ref,'    return .85 * rank(.5 * rank(r) + .5 * rank(ours)) + .15 * rank(','''    public = .5 * rank(r) + .5 * rank(ours)
    public[:, [7,9]] = rank(r)[:, [7,9]]
    return .85 * rank(public) + .15 * rank(''')
    else:
        ref=replace(ref,'return .85 * rank',f'return {1-weight!r} * rank')
        ref=replace(ref,'+ .15 * rank(',f'+ {weight!r} * rank(')
    return 'scored_rank = c85.scored_rank\n'+source+'\n'+ref

def runtime_source(tag,files=None):
    files=files or frozen_files();s=SPECS[tag]
    module='c85r7_codec_'+tag.lower()+'_runtime'
    source=files['exp169_release_c85r7codec.py'].decode()
    source=replace(source,'VERSION = "1.1.0"',f'VERSION = {s["version"]!r}')
    source=replace(source,'RELEASE = "C85R7+codecs"',f'RELEASE = {"C85R7+codecs+"+tag!r}')
    source=replace(source,'FORMULA = base.FORMULA',f'FORMULA = {s["formula"]!r}')
    source=replace(source,'str(runtime / "exp169_release_c85r7codec.py")',f'str(runtime / {module+".py"!r})')
    source=replace(source,'    receipt.update(version=VERSION, release=RELEASE, codecs=codec)',
        f'    receipt.update(version=VERSION, release=RELEASE, codecs=codec, formula=FORMULA, hypothesis={s["hypothesis"]!r}, base_version="1.1.0", exposure="EXPOSED", clean_oof=False)'+
        ('\n    receipt["flip_policy"] = selective_flip.policy("FLIP_ALL")' if tag=='flipall' else '')+
        ('\n    receipt["intended_members"] = receipt["executed_members"] = ["R2-s43","R2-s44","S_R7","EXP143v2","EXP155","Q_A5q"]\n    receipt["execution_passes"][1] = dict(serving="HB116 native R2 + R7", arms=["R2-s43","R2-s44","S_R7"])' if tag=='r2' else ''))
    inject=('\n'+math_source(.25 if tag=='own25' else .15,tag=='fluidR')) if tag in ('own25','fluidR') else ''
    if tag=='fluidR':
        inject+='''
_original_p_branch = p_branch
def p_branch(r, ours, failed):
    import numpy as np
    value = _original_p_branch(r, ours, failed)
    rr = c85.scored_rank(r, np.all(r == .5, axis=1))
    value[:, [7,9]] = rr[:, [7,9]]
    return value
'''
    if tag in ('fluidR','r2'):
        source=replace(source,'public cell 7: .5 r(R)+.5 r(S_R7), each over own non-failed rows; float32',s['formula'].split('; O=')[0]+'; successful-row ranks; float32')
    if tag=='flipall':
        inject+='\nimport exp169_r7_selective_flip_v130 as selective_flip\nimport c85r7_codec_rebase_io_v1 as flip_io\n'
        start=source.index('        r7.infer_studies(',source.index('def run_public('))
        end=source.index('\n    c85.phase(output, "S_R7", infer_s)',start)
        source=source[:start]+'        with selective_flip.installed(r7, "FLIP_ALL", flip_io):\n'+''.join('    '+line for line in source[start:end].splitlines(True))+source[end:]
    if tag=='r2':
        inject+='''
from tools import c85r7_lb_r2_adapter_v1 as r2_adapter
_original_authenticated_pr7 = authenticated_pr7
def authenticated_pr7(directory):
    ids, _, ours, info = _original_authenticated_pr7(directory)
    r, failures, identity = r2_adapter.recover(directory / 'pr7-run/r2', ids)
    info['failures']['R'] = failures
    info['r_prediction_fallbacks'] = len(failures)
    info['r2_identity'] = identity
    info['executed_arms'] = ['R2-s43','R2-s44','S_R7']
    return ids, r, ours, info
'''
        source=replace(source,'    ids = preflight(runtime, output)','    ids = preflight(runtime, output)\n    r2_adapter.preflight(runtime)')
        start=source.index('    c85.phase(output, "R", lambda:')
        end=source.index('\n    def infer_s():',start)
        source=source[:start]+'''    # Native R2 subprocesses inherit the same offline codec environment.
    previous_env = dict(os.environ)
    try:
        os.environ.update(env)
        c85.phase(output, "R2", lambda: r2_adapter.run(runtime, comp, work / "r2"))
    finally:
        os.environ.clear(); os.environ.update(previous_env)
'''.rstrip()+source[end:]
    source=replace(source,'\ndef main():',inject+'\ndef main():')
    source=replace(source,'    args = parser.parse_args()','    args = parser.parse_args()\n    if args.command not in ("run", "public-worker"):\n        parser.error("sealed runtime; use codec rebase builder")')
    compile(source,module,'exec')
    own_proof(tag, source)
    return module,source

def payload(tag):
    files=frozen_files(); parent=dict(files)
    module,source=runtime_source(tag,files)
    files[module+'.py']=source.encode()
    if tag=='flipall':
        for name in ('exp169_r7_selective_flip_v130','c85r7_codec_rebase_io_v1'):
            files[name+'.py']=(ROOT/'tools'/(name+'.py')).read_bytes()
    meta=h.read(BASE/'kernel/kernel-metadata.json')
    if tag=='r2':
        for name,data in h.source_payloads().items():
            b.need(name not in files or files[name]==data,'native collision: '+name)
            files[name]=data
        files['tools/c85r7_lb_r2_adapter_v1.py']=(ROOT/'tools/c85r7_lb_r2_adapter_v1.py').read_bytes()
        h.pin(h.OWN_DIR/'release-manifest.json',{'sha256':h.OWN_MANIFEST_SHA})
        files['r2-own-manifest.json']=(h.OWN_DIR/'release-manifest.json').read_bytes()
        meta['dataset_sources'] += ['taeyangg4/knee-hb113-rplus-weights-v2']
    b.need(all(files[k]==v for k,v in parent.items()),'frozen parent bytes')
    b.need(not any('deadline' in k or 'release_metadata_v122' in k or 'insurance' in k for k in files),'insurance contamination')
    blob,pins=h.package(files);nb=b.notebook_payload(blob,pins)
    code=''.join(nb['cells'][-1]['source']).replace('import exp169_release_c85r7codec as release',f'import {module} as release').replace(b.RUNTIME_DIR,'/kaggle/working/lb-'+SPECS[tag]['slug']+'-runtime')
    code=code.replace("print('C85R7+codecs v1.1.0',",f"print('C85R7+codecs {tag} v{SPECS[tag]['version']}',")
    nb['cells'][-1]['source']=code.splitlines(True)
    nb['cells'][0]['source']=[f"# {tag} {SPECS[tag]['version']}\n{SPECS[tag]['hypothesis']}. {SPECS[tag]['formula']}; base v1.1.0.\nPrivate/offline; EXPOSED; clean_oof=false. Original source credits and licences preserved in archive.\n"]
    meta.update(id='taeyangg4/knee-lb-'+SPECS[tag]['slug']+'-v1',title='knee-lb-'+SPECS[tag]['slug']+'-v1')
    return files,nb,meta,source

def launch(tag):
    s=SPECS[tag];path=stage(tag).relative_to(ROOT).as_posix();slug='taeyangg4/knee-lb-'+s['slug']+'-v1'
    warning='FLIP_ALL exceeds 9 h including reserve; timing is NOT qualified.' if tag=='flipall' else 'Projection only; target runtime is not newly measured.'
    return f'''# {tag} v{s['version']}
{s['hypothesis']}. Formula: `{s['formula']}`; **base v1.1.0**.
EXPOSED; clean_oof=false. Sealed hypothesis; original members except the declared R2 substitution.
No insurance/deadline module and no N32 kernel. Parent sources remain byte-identical. Own a172.py bytes, direct __main__ execution and child command/environment are identical to v1.1.0. No named-import wrapper.

## Builder offline CPU commands (from project root)
```powershell
& C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe -B .local/scratch/c85r7-fork-fix/build/cpu.py tools/c85r7_codec_rebase_v2.py validate --tag {tag}
& C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe -B .local/scratch/c85r7-fork-fix/build/cpu.py tools/c85r7_codec_rebase_v2.py replay --tag {tag}
```
## B only: push, inspect successful visible CSV/member receipts, submit once
```powershell
kaggle kernels push -p {path}/kernel
.venv/Scripts/python.exe tools/kaggle_submit_once.py --id lb-{s['slug']}-v1 --kernel {slug} --version 1 --message "{s['hypothesis']}; {s['formula']}; base v1.1.0"
```
The submit version is the first pushed version; B must use the actual successfully run version if different. B retains duplicate/gain review. Builder does not push or submit.

## Runtime / evidence scope
{s['hours']:.6f} h projected compute + 20 min reserve = {s['hours']+1/3:.6f} h versus 9 h. {warning}
Baseline is 4.7 h. FLIP_ALL adds N1300*((680.704481005-344.507129447)/(32-3))/3600=4.186365 h. Own25/fluidR add zero inference. R2 conservatively uses the native <=7.4 h envelope (2.7 h allowance); original own Q differs, so this is not a measured bound.
CPU replay uses authenticated predictions on the three actual visible studies, verifies independent formula/ordered CSV/publication and no-op bytes. FLIP_ALL additionally checks the reviewed scoring transform; replayed R7 predictions are plain, not a fresh flipped neural inference. No CUDA, Linux execution, new VRAM or watchdog qualification claimed. Frozen v1.1.0 publication/failure behavior is inherited; insurance is deliberately excluded per task.
'''

def prepare():
    b.need(all(not stage(t).exists() for t in SPECS),'new stages only')
    BACKUP.mkdir(parents=True,exist_ok=True)
    for name in ('build-receipt.json','kernel/release.ipynb','kernel/kernel-metadata.json'):
        p=BACKUP/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((BASE/name).read_bytes())
    for tag,s in SPECS.items():
        files,nb,meta,source=payload(tag);out=stage(tag);out.mkdir()
        dump(out/'kernel/release.ipynb',nb);dump(out/'kernel/kernel-metadata.json',meta)
        dump(out/'registration.json',dict(version=s['version'],base='1.1.0',base_receipt_sha256=BASE_SHA,**{k:v for k,v in s.items() if k!='version'},exposure='EXPOSED',clean_oof=False,qualified_for_submission=False,files={k:digest(v) for k,v in files.items()},runtime_with_reserve=s['hours']+1/3))
        metadata_io.atomic_text(out/'LAUNCH.md',launch(tag))
        metadata_io.atomic_text(out/'CHANGELOG.md',f"# v{s['version']} - 2026-10-14\nRestore exact v1.1.0 own a172.py direct execution; remove import-time pool wrapper. Retain only {s['hypothesis']} plus version/provenance/sealed runtime dispatch. No insurance.\nBackup: {BACKUP.relative_to(ROOT).as_posix()}. Frozen/pushed stages untouched.\n")
        metadata_io.atomic_text(out/'parent-runtime.diff',''.join(difflib.unified_diff(frozen_files()['exp169_release_c85r7codec.py'].decode().splitlines(True),source.splitlines(True),fromfile='v1.1.0',tofile=s['version'])))
        dump(out/'own-path-proof.json', own_proof(tag, source))
        metadata_io.atomic_text(out/'own-path.diff', ''.join(difflib.unified_diff(
            ''.join(own_path(frozen_files()['exp169_release_c85r7codec.py'].decode()).values()).splitlines(True),
            ''.join(own_path(source).values()).splitlines(True),fromfile='v1.1.0 own',tofile=s['version']+' own')))
        dump(out/'seal.json',{p.relative_to(out).as_posix():h.sha(p) for p in out.rglob('*') if p.is_file()})
    return dict(status='PREPARED',kernels=4,partial_stages='all older stages preserved; v1.1.5 through v1.1.8 are new')

def validate(tag):
    out=stage(tag)
    for name,sha in h.read(out/'seal.json').items(): h.pin(out/name,{'sha256':sha})
    files,nb,meta,_=payload(tag)
    b.need(h.read(out/'kernel/release.ipynb')==nb,'notebook drift')
    b.need(h.read(out/'kernel/kernel-metadata.json')==meta,'metadata drift')
    b.need(meta['is_private'] is True and meta['enable_internet'] is False,'offline private')
    b.need(h.read(out/'registration.json')['files']=={k:digest(v) for k,v in files.items()},'pins')
    own_proof(tag)
    return dict(status='PASS_VALIDATE',tag=tag,parent_bytes_preserved=True,own_path_byte_identical=True,insurance_absent=True,files=len(files),runtime_with_reserve=SPECS[tag]['hours']+1/3)

def replay(tag):
    from tools import c85r7_codec_rebase_verify_v2 as v
    return v.replay(tag)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['prepare','validate','replay']);ap.add_argument('--tag',choices=SPECS)
    args=ap.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if args.command=='prepare': result=prepare()
    else:
        b.need(args.tag,'--tag required');result=globals()[args.command](args.tag)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
