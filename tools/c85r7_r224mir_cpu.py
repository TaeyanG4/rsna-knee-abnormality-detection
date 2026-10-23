"""H-B-170 real-data CPU mirror replay 1.0.0; AST-only, no notebook execution."""
from __future__ import annotations
import ast
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import c85r7_r224mirq_v1126 as b
from tools import c85r7_r224_cpu_smoke as prior
from tools import c85r7_r224_timing as t

EXTRA = ('_slot_index', '_mirror_tri', 'infer_pair_probs')


def function_bytes(raw, name):
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return b'\n'.join(raw.splitlines()[node.lineno-1:node.end_lineno])


def namespace():
    b.validate()
    raw = b.old.unpack(b.flipq.restored(b.old.read(b.OUT/'kernel/release.ipynb')))[0][b.SOURCE]
    ns = t.public_namespace()
    tree = ast.parse(raw)
    allowed = set(t.FUNCTIONS) | set(EXTRA)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in allowed
                 or isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in ('SLOT_SIZES','SAGITTAL_SLOTS') for x in n.targets)]
    exec(compile(tree, '<hb170-candidate-allowlist>', 'exec'), ns)
    return ns


def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    ns = namespace()
    np, torch = ns['np'], ns['torch']
    torch.set_num_threads(4)
    old = t.io.read_json(prior.WORK/'receipt.json')
    t.need(old['status'] == 'PASS_CPU_TWO_REAL_STUDIES', 'real source receipt missing')
    for name, value in old['input_files'].items():
        t.need(t.sha(prior.WORK/'data'/name) == value, 'real DICOM hash changed')
    inv = t.inventory()
    t.need(t.digest(inv) == old['inventory_sha256'], 'inventory drift')
    grouped, reader = t.group_series(inv['series']), ns['_make_reader']()
    files = b.old.unpack(b.flipq.restored(b.old.read(b.OUT/'kernel/release.ipynb')))[0]
    t.need(function_bytes(files[b.SOURCE], '_mirror_tri') == function_bytes(files['public-serving.py'], '_mirror_tri'), 'mirror source bytes differ')
    t.need(t.sha(t.CHECKPOINT) == t.CHECKPOINT_SHA, 'checkpoint drift')
    arm = ns['ARMS'][0]
    model, res = ns['load_model'](str(t.CHECKPOINT), arm['arch'], arm['res'], 'cpu')
    results = []
    coverage = set()
    for item in old['studies']:
        vol, mask = ns['build_study'](item['study'], grouped, str(prior.WORK/'data/train_series'), reader)
        t.validate_volume(vol, mask)
        t.need(hashlib.sha256(vol.tobytes()).hexdigest() == item['volume_sha256'], 'real volume drift')
        cs = np.array([max(1,min(c,94)) for c in ns['_eval_centers'](mask,96,94)])
        tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs-1,cs,cs+1],1)]))
        mirrored = ns['_mirror_tri'](tri, cs, 96)
        slots = np.searchsorted(np.cumsum([0,26,22,18,12,18]), cs, side='right')-1
        coverage.update(int(x) for x in slots)
        oracle = torch.stack([x.flip(0) if slot in (0,1) else x.flip(-1) for x,slot in zip(tri,slots)])
        t.need(torch.equal(mirrored,oracle), 'uint8 anatomical axes differ')
        plain = ns['gpu_windows'](vol,mask,94,res,'cpu')
        t.need(hashlib.sha256(plain.numpy().tobytes()).hexdigest() == item['windows_sha256'], 'plain window bytes changed')
        win = ns['gpu_windows'](vol,mask,94,res,'cpu',mirror=True)
        float_tri = oracle.float()/255.0
        expected = torch.nn.functional.interpolate(float_tri[:,:,38:345,38:345], size=(224,224), mode='bilinear', align_corners=False)
        normalized = (expected-ns['_MEAN'])/ns['_STD']
        t.need(torch.equal(win,normalized), 'normalized mirrored uint8 oracle differs')
        # The parent uses different LUT shapes for standalone norm=none; isolate caches.
        ns['_WLUT'].clear()
        unnormalized = ns['gpu_windows'](vol,mask,94,res,'cpu',norm='none',mirror=True)
        t.need(torch.equal(unnormalized,expected), 'unnormalized mirror differs')
        ns['_WLUT'].clear()
        p = ns['infer_probs'](model,plain,'cpu')
        q = ns['infer_probs'](model,win,'cpu')
        pair = ns['infer_pair_probs'](model,plain,win,'cpu')
        t.need(pair.shape==(12,) and np.isfinite(pair).all() and np.array_equal(pair,.5*(p+q)), 'CPU pair mismatch')
        results.append(dict(study=item['study'], windows_sha256=item['windows_sha256'],
            centre_slots=sorted(set(int(x) for x in slots)),
            mirror_windows_sha256=hashlib.sha256(win.numpy().tobytes()).hexdigest(),
            uint8_exact=True, normalized_exact=True, none_norm_exact=True, pair_forward_exact=True))
    t.need(coverage=={0,1,2,3,4}, 'all five anatomical centre slots required')
    result = dict(status='PASS_CPU_REAL_MIRROR', version='1.0.0', studies=results, centre_slots=sorted(coverage),
        kernel_sha256=b.validate()['kernel_sha256'], checkpoint_sha256=t.CHECKPOINT_SHA,
        tool_sha256=t.sha(__file__), source_receipt_sha256=t.sha(prior.WORK/'receipt.json'),
        cpu_only=True, exposure='EXPOSED',clean_oof=False,
        limitation='CPU cannot prove CUDA memory, Linux serving parity or runtime')
    t.io.journal(b.SCRATCH/'cpu-mirror.json',result,b.SCRATCH)
    return {k:v for k,v in result.items() if k!='studies'}


if __name__=='__main__':
    import json
    print(json.dumps(run()))
