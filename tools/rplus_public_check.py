"""R+ v1.0.0 CPU compatibility audit of the frozen public-serving class and R weights."""
import argparse
import ast
import json
from pathlib import Path
import torch
from torch import nn
import timm
from training.arms import coatnet as c, lifecycle as life
from training.arms.rplus import write

SERVING='artifacts/releases/EXP169_B172_v2_0_2_stage/own-anchor/assets/public-serving.py'
PUBLIC_NAME='raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt'
PUBLIC_SHA='7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'


def check(root,checkpoint=None):
    root=Path(root)
    torch.set_num_threads(2)
    tree=ast.parse((root/SERVING).read_text(encoding='utf-8'))
    # Extract only model definitions: never execute serving/file/account actions.
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
           and n.name in ('build_backbone','RaptorClassifier')]
    if len(nodes)!=2: raise ValueError('public model definitions differ')
    ns=dict(torch=torch,nn=nn,timm=timm,INPUT_RES=384)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),SERVING,'exec'),ns)
    model=c.build_model(c.Config())
    public=ns['RaptorClassifier'](ns['build_backbone'](c.base.ARCH),F_dim=model.backbone.num_features)
    def schema(m): return {k:(list(v.shape),str(v.dtype)) for k,v in m.state_dict().items()}
    own_schema,public_schema=schema(model),schema(public)
    result=dict(schema='rplus-public-compatibility/v1',serving_sha256=life.file_sha256(root/SERVING),
        public_sha256=PUBLIC_SHA,model_class_schema_equal=own_schema==public_schema,
        tensor_keys=len(own_schema),checkpoint_strict_load=None,checkpoint_status='UNAVAILABLE',
        cuda_initialized=torch.cuda.is_initialized())
    if own_schema!=public_schema: raise ValueError('public serving / trainer schemas differ')
    del public
    if checkpoint:
        checkpoint=Path(checkpoint)
        if life.file_sha256(checkpoint)!=PUBLIC_SHA: raise ValueError('public R hash differs')
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if ck.get('arch')!=c.base.ARCH or ck.get('res')!=384: raise ValueError('public architecture metadata differs')
        incompatible=model.load_state_dict(ck['model'],strict=True)
        result.update(checkpoint_status='CHECKED',checkpoint_strict_load=True,
                      missing_keys=incompatible.missing_keys,unexpected_keys=incompatible.unexpected_keys)
        del model
        evaluation=c.build_model(c.Config(),res=320)
        evaluation.load_state_dict(ck['model'],strict=True)
        result['inference_320_strict_load']=True
    if torch.cuda.is_initialized(): raise ValueError('CPU check initialized CUDA')
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',default='.')
    p.add_argument('--checkpoint');p.add_argument('--out',required=True);a=p.parse_args()
    result=check(a.root,a.checkpoint);write(a.out,result);print(json.dumps(result))


if __name__=='__main__': main()
