from pathlib import Path
import json, os, sys, time
receipt={'python':sys.version,'cwd':str(Path.cwd()),'env':{k:v for k,v in os.environ.items() if 'TPU' in k or 'PJRT' in k or 'XLA' in k}}
try:
    import torch
    receipt['torch']=torch.__version__
except Exception as e:
    receipt['torch_error']=repr(e)
try:
    import torch_xla
    receipt['torch_xla']=getattr(torch_xla,'__version__','unknown')
    import torch_xla.core.xla_model as xm
    receipt['xla_device']=str(xm.xla_device())
    try: receipt['xrt_world_size']=int(xm.xrt_world_size())
    except Exception as e: receipt['xrt_world_size_error']=repr(e)
    try:
        import torch_xla.runtime as xr
        receipt['runtime_device_count']=int(xr.global_runtime_device_count())
        receipt['runtime_type']=str(xr.device_type())
    except Exception as e: receipt['runtime_error']=repr(e)
except Exception as e:
    receipt['torch_xla_error']=repr(e)
base=Path('/kaggle/input')
hits=[]
if base.exists():
    for p in base.rglob('adjacent_cache_manifest.json'):
        try:
            m=json.loads(p.read_text())
            hits.append({'path':str(p),'studies':m.get('studies'),'shape_total':m.get('shape_total'),'failure_count':m.get('failure_count'),'shards':len(m.get('shards',[]))})
        except Exception as e: hits.append({'path':str(p),'error':repr(e)})
receipt['adjacent_cache_hits']=hits
Path('/kaggle/working/tpu_preflight_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2),flush=True)