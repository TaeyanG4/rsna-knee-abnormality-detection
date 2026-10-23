"""v1.0.0: real DICOM parity for stride1/2 and bounded CPU classifier replay.

Read existing prepared raw input; all output stays in the fixed astra-b2 lane.
No labels, network, CUDA, data copies, or frozen-source writes.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']='-1'
if os.name=='nt':
    os.environ['S6PLUS_PIN_DLL']=r'C:\Windows\System32\DriverStore\FileRepository\nv_dispui.inf_amd64_49a28c45e7d7dc62\nvdxgdmal64.dll'
    from tools import s6plus
from tools import k1_shared_render_harness_v1 as h
from tools import k1_shared_render_v3 as shared
from tools import k1n_stage_v11 as builder
from tools import k1sr_runtime_v11 as runtime
w=shared.w
OUT=h.ROOT/'.local/scratch/astra-b2/k1s1stride1-v1/cpu-smoke'


def sources():
    return {str(Path(p).relative_to(h.ROOT)):w.sha(p) for p in
            (__file__,shared.__file__,runtime.__file__,builder.schema.__file__)}


def arm(name):
    import numpy as np
    out=OUT/name; out.mkdir(parents=True,exist_ok=True)
    w.write(out/'receipt.json',dict(status='STARTING'))
    started=time.perf_counter(); pins=sources(); torch=w.configure('cpu')
    selection=w.read(h.OUT/'selection.private.json'); uid=selection['ids'][0]
    for member,crc,size,sha in selection['members']:
        if member.split('/')[1]==uid: w.need(w.sha(h.OUT/'raw'/member)==sha,'raw identity')
    w.need(w.sha(h.OUT/'raw/train_series.csv')==selection['series_sha256'],'series identity')
    with w.offline():
        w.assets_check(h.ASSETS,h.ASSETS_SHA)
        original=name.startswith('original')
        stride=int(name[-1]) if original else 1
        if original:
            ct,ci=w.import_frozen(h.ASSETS)
            w.need(w.sha(ci.__file__)==shared.K1_INFER_SHA,'original pin')
        else:
            src=h.OUT/'runtime/r7_runtime'
            w.need(src.is_dir(),'existing frozen renderer required; no writes outside lane')
            sys.path[:0]=[str(src),str(h.ASSETS/'cloud_code')]
            import compact_infer as ci
        w.install_label_free_source()
        ds=ci.CStudies(h.OUT/'raw','train',[uid],out/'render',h.ASSETS/'cloud_code',384,
            eval_stride=stride if original else 1,views='v4',slices=3,
            anchor_weights=h.ASSETS/'weights/anatomy.pt',phys_mm=3.5,
            side_model=None if original else h.BUNDLE/'side_cls.pt',
            extent_meta=not original,**({} if original else dict(fluid=True,device='cpu')))
        detail=dict(route='original-cstudies'); side_exact={}; forward={}
        if original: parts={stride:ds[0]}
        else:
            os.environ['K1SR_STARTED']=str(time.time())
            parent=runtime.LoaderDataset(ds,out/'worker0')[0]
            w.need(parent.get('error') is None and 'item' in parent['_k1sr'],'runtime11 loader')
            parts=parent['_k1sr']['item']; detail=parent['_k1sr']['detail']
            w.need(set(parts)=={1,2},'both captured strides')
            for s,item in parts.items():
                ci.FORCED_SIDE[uid]='L'
                side=shared.original_side(ds,ci,0,s)
                side_exact[s]=torch.equal(item['images'],side['images']) and torch.equal(item['meta'],side['meta'])
                w.need(side_exact[s] and ci.FORCED_SIDE[uid]=='L','original-side parity/restoration')
                ci.FORCED_SIDE.pop(uid)
                w.need(ds.eval_stride==1 and ds.extent_meta and ds.fluid,'parent state restored')
            # CPU only; bounded real tensors exercise new grouping's unchanged forward.
            import importlib.util
            spec=importlib.util.spec_from_file_location('stride1_smoke_trainer',h.ASSETS/'source/compact_train.py')
            ct=importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
            torch.manual_seed(181)
            model=ct.Net('convnext_small',None,crop=352,in_chans=3,spatial=True,extent=False).eval()
            for s,item in parts.items():
                x,m=w.validate_item(item,uid)
                prepared=runtime.transfer_inputs(x[:2],m[:2],'cpu')
                pred=runtime.predict_prepared(model,prepared,fp16=False)
                w.need(pred.shape==(12,) and np.isfinite(pred).all(),'CPU forward')
                forward[s]=dict(windows=2,finite=True,checkpoint='random engineering smoke only')
        results={s:dict(windows=len(item['images']),images_sha256=w.tensor_sha(item['images']),
                       meta_sha256=w.tensor_sha(item['meta'])) for s,item in parts.items()}
        w.need(not torch.cuda.is_initialized() and pins==sources(),'CPU isolation/source drift')
        result=dict(status='PASS',arm=name,parts=results,uid_sha256=hashlib.sha256(uid.encode()).hexdigest(),
            selection_sha256=w.sha(h.OUT/'selection.private.json'),sources=pins,detail=detail,
            original_side_reset_bit_exact=side_exact,cpu_forward=forward,cuda_initialized=False,
            seconds=time.perf_counter()-started)
        w.write(out/'receipt.json',result)
        print(name,'PASS', {s:r['windows'] for s,r in results.items()},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('original2','original1','shared'))
    args=parser.parse_args()
    if args.arm: return arm(args.arm)
    OUT.mkdir(parents=True,exist_ok=True)
    w.write(OUT/'comparison.json',dict(status='STARTING'))
    for name in ('original2','original1','shared'):
        subprocess.run([sys.executable,'-m','tools.k1n_stride1_cpu_smoke_v1','--arm',name],cwd=h.ROOT,check=True)
    paired=w.read(OUT/'shared/receipt.json')
    for s in (2,1):
        base=w.read(OUT/f'original{s}/receipt.json')
        for r in (base,paired): w.need(r['status']=='PASS' and r['sources']==sources(),'fresh smoke binding')
        w.need(base['parts'][str(s)]==paired['parts'][str(s)],'tensor parity')
        w.need(base['uid_sha256']==paired['uid_sha256'],'paired study')
    result=dict(status='PASS',strides=[2,1],studies=1,tensors_bit_exact=True,
        original_side_reset_bit_exact=paired['original_side_reset_bit_exact'],
        windows={s:p['windows'] for s,p in paired['parts'].items()},cpu_forward=paired['cpu_forward'],
        cuda_initialized=False,sources=sources(),qualified_for_submission=False,
        scope='one real DICOM study, both exact full inputs; original-side restoration; two-window random FP32 forward per stride')
    w.write(OUT/'comparison.json',result)
    print('CPU_SMOKE_PASS',result['windows'],flush=True)


if __name__=='__main__': main()
