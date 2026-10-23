"""v1.0.0: actual generated adapters/runtimes on pinned real DICOM, CPU only.

Compare K1 stride 2/3 with frozen original CStudies. No classifier, labels,
GPU, network or public notebook execution. All writes use this lane's scratch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if os.name == 'nt':
    os.environ['S6PLUS_PIN_DLL'] = r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
    from tools import s6plus

from tools import k1_shared_render_harness_v1 as h
from tools import k1_shared_render_v2 as shared_reference
from tools import k1n_stage_v1 as b

w = shared_reference.w
OUT = h.ROOT/'.local/scratch/astra-b4/k1n-r6-stride/cpu-smoke'
ARMS = ('original2', 'original3', 'scalar', 'vector', 'generic3')


def configuration(name):
    filename = ('x3-stride2-alpha-vector-v2-r6s2.json' if name == 'vector'
                else 'x3-stride2-drop-o-r6s2.json')
    c = json.loads((b.ROOT/'configs/k1n'/filename).read_bytes())
    if name == 'generic3':
        c.update(k1_eval_stride=3, release_version='1.9.99-cpu-smoke-r6s2',
                 kernel_id='test-owner/knee-cpu-smoke-r6s2-v1')
    return c


def arm(name):
    out = OUT/name
    out.mkdir(parents=True, exist_ok=True)
    w.write(out/'receipt.json', dict(status='STARTING'))
    started = time.perf_counter()
    torch = w.configure('cpu')
    original = name.startswith('original')
    stride = int(name[-1]) if original else configuration(name)['k1_eval_stride']
    files = b.package(configuration(name))[3] if not original else {}
    shared = ModuleType('actual_r6_stride_adapter')
    if not original:
        exec(compile(files['tools/k1_shared_render_v2.py'], '<actual adapter>', 'exec'), shared.__dict__)
    selection = w.read(h.OUT/'selection.private.json')
    uid = selection['ids'][0]
    for member, crc, size, sha in selection['members']:
        if member.split('/')[1] == uid:
            w.need(w.sha(h.OUT/'raw'/member) == sha, 'raw identity')
    w.need(w.sha(h.OUT/'raw/train_series.csv') == selection['series_sha256'], 'series identity')
    with w.offline():
        w.assets_check(h.ASSETS, h.ASSETS_SHA)
        if original:
            _, ci = w.import_frozen(h.ASSETS)
            w.need(w.sha(ci.__file__) == shared_reference.K1_INFER_SHA, 'original renderer pin')
        else:
            # This harness accessor is read-only when the existing pinned cache exists.
            w.need((h.OUT/'runtime/r7_runtime/compact_infer.py').is_file(), 'existing runtime cache required')
            sys.path[:0] = [str(h.runtime_source()), str(h.ASSETS/'cloud_code')]
            import compact_infer as ci
        w.install_label_free_source()
        ds = ci.CStudies(h.OUT/'raw', 'train', [uid], out/'render', h.ASSETS/'cloud_code', 384,
            eval_stride=stride if original else 2, views='v4', slices=3,
            anchor_weights=h.ASSETS/'weights/anatomy.pt', phys_mm=3.5,
            side_model=None if original else h.BUNDLE/'side_cls.pt',
            extent_meta=not original, **(dict(fluid=True, device='cpu') if not original else {}))
        detail = dict(route='original-cstudies')
        side_exact = None
        parent_windows = None
        if original:
            item = ds[0]
        else:
            runtime = ModuleType('actual_r6_stride_runtime')
            exec(compile(files['tools/k1sr_runtime_v7.py'], '<actual runtime>', 'exec'), runtime.__dict__)
            runtime.shared = shared
            os.environ['K1SR_STARTED'] = str(time.time())
            parent = runtime.LoaderDataset(ds, out/'worker0')[0]
            w.need(parent['error'] is None and 'item' in parent['_k1sr'], 'actual loader capture')
            item, detail = parent['_k1sr']['item'], parent['_k1sr']['detail']
            parent_windows = len(parent['images'])
            w.need(detail['k1_eval_stride'] == stride, 'loader stride')
            ci.FORCED_SIDE[uid] = 'L'
            side = shared.original_side(ds, ci, 0, stride)
            side_exact = torch.equal(item['images'], side['images']) and torch.equal(item['meta'], side['meta'])
            w.need(side_exact and ci.FORCED_SIDE[uid] == 'L', 'original-side parity/restoration')
            ci.FORCED_SIDE.pop(uid)
            w.need(ds.eval_stride == 2 and ds.extent_meta and ds.fluid, 'R6 state restoration')
        x, m = w.validate_item(item, uid)
        w.need(not torch.cuda.is_initialized(), 'CUDA initialized')
        receipt = dict(status='PASS', arm=name, k1_eval_stride=stride, windows=len(x),
            r6_eval_stride=None if original else 2, r6_windows=parent_windows,
            images_sha256=w.tensor_sha(x), meta_sha256=w.tensor_sha(m),
            uid_sha256=hashlib.sha256(uid.encode()).hexdigest(),
            selection_sha256=w.sha(h.OUT/'selection.private.json'), renderer_sha256=w.sha(ci.__file__),
            embedded_sources={n:b.digest(v) for n,v in files.items()},
            original_side_reset_bit_exact=side_exact, cuda_initialized=False,
            seconds=time.perf_counter()-started, route=detail['route'])
        w.write(out/'receipt.json', receipt)
        print(json.dumps(dict(arm=name, status='PASS', windows=len(x), r6_windows=parent_windows)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.arm:
        arm(args.arm)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    w.write(OUT/'comparison.json', dict(status='STARTING', qualified_for_submission=False))
    for name in ARMS:
        subprocess.run([sys.executable, '-m', 'tools.k1n_r6_cpu_smoke_v1', '--arm', name],
                       cwd=h.ROOT, check=True)
    receipts = {name:w.read(OUT/name/'receipt.json') for name in ARMS}
    for name in ('scalar', 'vector', 'generic3'):
        actual = receipts[name]
        expected = receipts['original'+str(actual['k1_eval_stride'])]
        for key in ('windows', 'images_sha256', 'meta_sha256', 'uid_sha256', 'selection_sha256'):
            w.need(actual[key] == expected[key], 'original/shared mismatch: '+name+'/'+key)
        w.need(actual['original_side_reset_bit_exact'] and not actual['cuda_initialized'], 'smoke failed')
        w.need(actual['embedded_sources'] == {n:b.digest(v) for n,v in b.package(configuration(name))[3].items()},
               'generated payload changed during smoke')
    result = dict(status='PASS', studies=1, arms=list(ARMS), tensors_bit_exact=True,
        original_side_reset_bit_exact=True, cuda_initialized=False, qualified_for_submission=False,
        windows={n:r['windows'] for n,r in receipts.items()},
        seconds={n:r['seconds'] for n,r in receipts.items()},
        scope='Real DICOM renderer/raw-row capture only; no classifier, FP16, full inference or accuracy claim.')
    w.write(OUT/'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
