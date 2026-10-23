"""CPU verification only; never execute a public notebook's serving body."""
from __future__ import annotations
import ast
import builtins
import copy
from pathlib import Path
import shutil
import sys
import types
from unittest.mock import patch
from tools import hb156_killtest_v1 as b


def extract():
    """Run the exact guarded archive/hash prelude, stopping before serving exec."""
    area = b.SCRATCH/'loader'
    assert area.resolve().is_relative_to(b.SCRATCH.resolve())
    if area.exists(): b.retry(lambda:shutil.rmtree(area))
    area.mkdir(parents=True)
    tree = ast.parse(''.join(b.read(b.OUT/'kernel/release.ipynb')['cells'][5]['source']))
    body = tree.body[0].body
    stop = next(i for i,n in enumerate(body) if isinstance(n,ast.Expr) and
        isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='exec')
    tree.body[0].body = body[:stop]
    original_import = builtins.__import__
    def imported(name,*args,**kwargs):
        if name == b.RUNTIME.removesuffix('.py'):
            return b.runtime((area/'exp152-assets'/b.RUNTIME).read_bytes())
        return original_import(name,*args,**kwargs)
    def mapped(value,*parts):
        if str(value) in ['/kaggle/input/'+b.DATASET.split('/')[1],'/kaggle/input/datasets/'+b.DATASET]:
            return b.OUT/'dataset'
        if str(value)=='/kaggle/working/exp152-assets': return area/'exp152-assets'
        if str(value)=='/kaggle/working': return area
        return Path(value,*parts)
    ns = {'FAST':False}
    paths = sys.path[:]
    try:
        with patch('pathlib.Path',mapped),patch('builtins.__import__',imported):
            exec(compile(tree,'<exact-cell5-prelude>','exec'),ns)
    finally:
        sys.path[:] = paths
    assert 'main' not in ns and not (area/'submission.csv').exists()
    return ns


def model_namespace(prelude):
    import torch
    import timm
    assert timm.__version__ == '1.0.19'
    source = (prelude['_assets']/'public-serving.py').read_text(encoding='utf-8')
    names = {'build_backbone','RaptorClassifier','load_model'}
    nodes = [n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==3
    ns = dict(torch=torch,nn=torch.nn,timm=timm,INPUT_RES=384,
        _E152=prelude['_e152'],_E152_FILES=prelude['_files'],_E152_MANIFEST=copy.deepcopy(prelude['_manifest']))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<exact-staged-model-definitions>','exec'),ns)
    return ns


def loader(prelude):
    import torch
    ns = model_namespace(prelude)
    checkpoint = prelude['_files'][b.NAME]
    rejects = {}
    for kind in ('sha','protocol','run','schema'):
        bad = copy.deepcopy(prelude['_manifest'])
        if kind=='sha': bad['checkpoints'][b.NAME]['sha256']='0'*64
        elif kind=='protocol': bad['protocol_sha256']='0'*64
        elif kind=='run': bad['hb151_export']['run_dir']='wrong/S42'
        else: bad['own_state_schema']={}
        ns['_E152_MANIFEST'] = bad
        try: ns['load_model'](checkpoint,b.ARCH,384,torch.device('cpu'))
        except ValueError as exc:
            expected = {'sha':'Bound checkpoint SHA mismatch','protocol':'protocol SHA mismatch',
                        'run':'run mismatch','schema':'keys/shapes/dtypes mismatch'}[kind]
            assert expected in str(exc)
            rejects[kind]='REJECT'
        else: raise AssertionError('invalid '+kind+' accepted')
    ns['_E152_MANIFEST'] = copy.deepcopy(prelude['_manifest'])
    model,res = ns['load_model'](checkpoint,b.ARCH,384,torch.device('cpu'))
    assert res==320 and not model.training and next(model.parameters()).device.type=='cpu'
    assert prelude['_e152'].state_schema(model.state_dict())==b.expected_schema()
    return model,dict(status='PASS',wrong=rejects,strict_load=True,device='cpu',
                      parameters=sum(p.numel() for p in model.parameters()))


def real_data(model,checkpoint,rt):
    import numpy as np
    import torch
    from exp143_own0949 import Corpus,windows
    corpus_dir = b.ROOT/'storage/cache/derived/EXP143_RAPTOR96_384_C140_S02'
    corpus = Corpus(corpus_dir)
    vol,mask = corpus.get(corpus.ids[0])
    reference = b.training_model()
    reference.load_state_dict(rt.safe_load(checkpoint)['model'],strict=True)
    reports=[]
    # Full K94 CPU serving forward; compare identical input/weights/head with training model.
    # Training reference encodes in chunks here to bound CPU RAM; attention still sees all K94.
    with torch.inference_mode():
        for mirror in (False,True):
            x = torch.from_numpy(windows(vol,mask,mirror=mirror)).unsqueeze(0)
            assert tuple(x.shape)==(1,94,3,320,320) and torch.isfinite(x).all()
            actual = model(x)
            feats = torch.cat([reference.backbone(c) for c in x.flatten(0,1).split(16)]).view(1,94,-1)
            expected = reference.head(feats)
            assert actual.shape==(1,12) and torch.isfinite(actual).all()
            torch.testing.assert_close(actual,expected,rtol=0,atol=1e-6)
            reports.append(dict(mirror=mirror,input_sha256=b.digest(x.numpy().tobytes()),
                shape=list(x.shape),max_abs=float((actual-expected).abs().max()),finite=True))
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',study_count=1,volume_sha256=b.sha(corpus_dir/'study_00000.npy'),
        mask_sha256=b.digest(np.asarray(mask).tobytes()),windows=reports,
        limitation='Cached real native volume, CPU float32; no DICOM decode, CUDA/fp16/Linux or score qualification.')


def run():
    b.cpu_only()
    from tools import hb151_killtest_v102_fixtures as fixtures
    # Reuse reviewed fast-path cases against this candidate; never change parent module globals.
    shim = types.SimpleNamespace(ROOT=b.ROOT,OUT=b.OUT,SCRATCH=b.SCRATCH,read=b.read)
    with patch.object(fixtures,'b',shim): fast = fixtures.run_smoke(b.SCRATCH/'fast-cases')
    prelude = extract()
    model,loaded = loader(prelude)
    data = real_data(model,prelude['_files'][b.NAME],prelude['_e152'])
    return dict(status='PASS',fast_path_cases=len(fast['cases']),fast_path=fast,
        archive_extract_verify_assets=True,loader=loaded,real_data=data,
        placeholder=b.read(b.OUT/'build-receipt.json')['placeholder'],qualified_for_submission=False)
