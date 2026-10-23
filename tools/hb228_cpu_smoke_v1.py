"""HB228 v1.0.0: real original K1 stride2 vs R6-stride2 shared capture on CPU.

Separate processes avoid frozen-module import collisions. No labels, GPU,
network or classifier inference. Outputs overwrite one fixed lane directory.
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
from tools import k1_shared_render_v2 as original_shared
from tools import hb228_stage_v1 as builder
shared = ModuleType("hb228_shared")
exec(compile(builder.adapter(Path(original_shared.__file__).read_bytes()), "<hb228 adapter>", "exec"), shared.__dict__)
shared.__file__ = original_shared.__file__
from tools import k1n_config_v1 as schema
from tools import k1sr_runtime_v7 as runtime

w = shared.w
OUT = h.ROOT/'.local/scratch/astra-b4/hb228/cpu-smoke'
CONFIG = builder.CONFIG


def sources():
    return {Path(p).name:w.sha(p) for p in
            (__file__, builder.__file__, shared.__file__, schema.__file__, runtime.__file__, CONFIG)}


def arm(name):
    out = OUT/name
    out.mkdir(parents=True, exist_ok=True)
    receipt = out/'receipt.json'
    w.write(receipt, dict(status='STARTING'))
    pins = sources()
    started = time.perf_counter()
    torch = w.configure('cpu')
    config = schema.validate(w.read(CONFIG))
    stride = config['k1_eval_stride']
    selection = w.read(h.OUT/'selection.private.json')
    uid = selection['ids'][0]
    for member, crc, size, sha in selection['members']:
        if member.split('/')[1] == uid:
            w.need(w.sha(h.OUT/'raw'/member) == sha, 'raw identity')
    w.need(w.sha(h.OUT/'raw/train_series.csv') == selection['series_sha256'], 'series identity')
    with w.offline():
        w.assets_check(h.ASSETS,h.ASSETS_SHA)
        if name == 'original':
            _, ci = w.import_frozen(h.ASSETS)
            w.need(w.sha(ci.__file__) == shared.K1_INFER_SHA, 'original renderer pin')
        else:
            sys.path[:0] = [str(h.runtime_source()),str(h.ASSETS/'cloud_code')]
            import compact_infer as ci
        w.install_label_free_source()
        ds = ci.CStudies(h.OUT/'raw','train',[uid],out/'render',h.ASSETS/'cloud_code',384,
            eval_stride=stride, views='v4', slices=3,
            anchor_weights=h.ASSETS/'weights/anatomy.pt',phys_mm=3.5,
            side_model=None if name == 'original' else h.BUNDLE/'side_cls.pt',
            extent_meta=name == 'shared', **(dict(fluid=True,device='cpu') if name == 'shared' else {}))
        detail = dict(route='original-cstudies')
        side_exact = None
        if name == 'original':
            item = ds[0]
        else:
            module = ModuleType('k1n_stride_baked')
            source = Path(runtime.__file__).read_text(encoding='utf-8').replace(
                'REGISTERED = config_schema.current()', 'REGISTERED = '+repr(config))
            exec(compile(source,'<registered runtime>','exec'),module.__dict__)
            module.shared = shared
            os.environ['K1SR_STARTED'] = str(time.time())
            parent = module.LoaderDataset(ds,out/'worker0')[0]
            w.need(parent['error'] is None and 'item' in parent['_k1sr'], 'registered loader capture')
            item,detail = parent['_k1sr']['item'],parent['_k1sr']['detail']
            w.need(detail['k1_eval_stride'] == stride, 'loader stride')
            ci.FORCED_SIDE[uid] = 'L'
            side = shared.original_side(ds,ci,0,stride)
            side_exact = torch.equal(item['images'],side['images']) and torch.equal(item['meta'],side['meta'])
            w.need(side_exact and ci.FORCED_SIDE[uid] == 'L', 'original-side parity/restoration')
            ci.FORCED_SIDE.pop(uid)
            w.need(ds.eval_stride == 2 and ds.extent_meta and ds.fluid, 'parent state restoration')
        x,m = w.validate_item(item,uid)
        w.need(not torch.cuda.is_initialized(), 'CUDA initialized')
        w.need(pins == sources(), 'source changed during CPU smoke')
        result = dict(status='PASS', arm=name, k1_eval_stride=stride, windows=len(x),
            images_sha256=w.tensor_sha(x), meta_sha256=w.tensor_sha(m),
            uid_sha256=hashlib.sha256(uid.encode()).hexdigest(), sources=pins,
            selection_sha256=w.sha(h.OUT/'selection.private.json'), renderer_sha256=w.sha(ci.__file__),
            original_side_reset_bit_exact=side_exact, cuda_initialized=False,
            seconds=time.perf_counter()-started, detail=detail, route=detail['route'])
        w.write(receipt,result)
        print(json.dumps(dict(arm=name,status='PASS',windows=len(x))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('original','shared'))
    args = parser.parse_args()
    if args.arm:
        arm(args.arm)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    w.write(OUT/'comparison.json',dict(status='STARTING',qualified_for_submission=False))
    pins = sources()
    for name in ('original','shared'):
        subprocess.run([sys.executable,'-m','tools.hb228_cpu_smoke_v1','--arm',name],
                       cwd=h.ROOT,check=True)
    a,b = [w.read(OUT/name/'receipt.json') for name in ('original','shared')]
    for r in (a,b):
        w.need(r['status'] == 'PASS' and r['sources'] == sources() == pins, 'incomplete/stale CPU smoke')
    for key in ('k1_eval_stride','windows','images_sha256','meta_sha256','uid_sha256','selection_sha256'):
        w.need(a[key] == b[key], 'original/shared mismatch: '+key)
    w.need(a['renderer_sha256'] == shared.K1_INFER_SHA and
           b['renderer_sha256'] == shared.R6_RUNTIME_SHA, 'renderer pair')
    result = dict(status='PASS', studies=1,k1_eval_stride=2,windows=a['windows'],
        tensors_bit_exact=True,original_side_reset_bit_exact=b['original_side_reset_bit_exact'],
        route=b['route'],cuda_initialized=False,qualified_for_submission=False,
        scope='Real DICOM render/windows only; no classifier, FP16 or parent inference.',
        sources=pins,seconds={r['arm']:r['seconds'] for r in (a,b)})
    w.write(OUT/'comparison.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
