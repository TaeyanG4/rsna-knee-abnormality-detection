"""K1 v3-G 3.0.0: grouping-only AdamW policy; no fit, data or GPU entrypoint.

Pretrained includes the relocated sp_head_norm. No decay: normalization affine,
bias, ConvNeXt LayerScale, and scalar/1D parameters. Other tensors retain wd .02.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

VERSION = '3.0.0'
GROUPS = ('pretrained_decay', 'pretrained_no_decay', 'head_decay', 'head_no_decay')
POLICY = dict(schema='k1-grouping/v3', lr=1e-4, head_lr=3e-4, weight_decay=.02, llrd=1.,
              pretrained_prefixes=['enc.', 'sp_head_norm.'], group_order=list(GROUPS),
              no_decay=['normalization affine', 'bias', 'LayerScale gamma', 'scalar/1D'],
              other='all remaining trainable matrix/convolution/embedding/query tensors decay')


def need(ok, message):
    if not ok: raise ValueError(message)


def _norm_kind(module):
    import torch.nn as nn
    classes = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm,
               nn.modules.instancenorm._InstanceNorm, nn.LocalResponseNorm)
    if hasattr(nn, 'RMSNorm'): classes += (nn.RMSNorm,)
    if isinstance(module, classes): return type(module).__name__
    # timm LayerNorm2d derives from LayerNorm in the installed runtime; explicit
    # known module provenance also handles its RMSNorm/fast implementations.
    cls = type(module)
    known = {'LayerNorm', 'LayerNorm2d', 'LayerNormExp2d', 'RmsNorm', 'RmsNorm2d',
             'RmsNormExp2d', 'GroupNorm', 'GroupNorm1', 'BatchNormAct2d', 'GroupNormAct'}
    if cls.__module__.startswith('timm.') and cls.__name__ in known: return cls.__name__
    need('norm' not in cls.__name__.lower() or not list(module.parameters(recurse=False)),
         'unclassified normalization module: ' + cls.__module__ + '.' + cls.__name__)
    return None


def parameter_groups(model, *, lr=1e-4, head_lr=3e-4, weight_decay=.02, llrd=1.):
    """Return exactly four disjoint groups plus full names/shapes, without tensor reads."""
    need((lr, head_lr, weight_decay, llrd) == (1e-4, 3e-4, .02, 1.), 'fixed v2 LR/wd/llrd required')
    modules = dict(model.named_modules())
    named = list(model.named_parameters(remove_duplicate=False))
    identities = {}; rows = []; frozen = []; groups = [dict(params=[], lr=lr if i<2 else head_lr,
        weight_decay=weight_decay if i%2==0 else 0., group_name=name) for i,name in enumerate(GROUPS)]
    for name, param in named:
        if not param.requires_grad:
            frozen.append(name); continue
        need(id(param) not in identities, 'duplicate trainable parameter alias: ' + name)
        identities[id(param)] = name
        owner_name, _, leaf = name.rpartition('.')
        need(owner_name in modules, 'parameter has no owning module: ' + name)
        module = modules[owner_name]; norm = _norm_kind(module)
        pretrained = name.startswith(('enc.', 'sp_head_norm.'))
        reasons = []
        if norm: reasons.append('normalization')
        if leaf == 'bias': reasons.append('bias')
        layerscale = 'layerscale' in type(module).__name__.lower()
        if layerscale:
            need(type(module).__name__ in {'LayerScale', 'LayerScale2d'} and
                 type(module).__module__.startswith('timm.'), 'unclassified LayerScale module: ' + name)
        if layerscale or (leaf == 'gamma' and name.startswith('enc.')):
            need(leaf == 'gamma' and (layerscale or type(module).__name__ == 'ConvNeXtBlock'),
                 'unclassified LayerScale parameter: ' + name)
            reasons.append('LayerScale')
        if param.ndim <= 1: reasons.append('scalar/1D')
        if name.startswith('sp_head_norm.'):
            need(norm is not None, 'relocated pretrained norm is not a recognized normalization module')
        no_decay = bool(reasons); index = (0 if pretrained else 2) + int(no_decay)
        group = groups[index]; group['params'].append(param)
        rows.append(dict(name=name, shape=list(param.shape), numel=param.numel(),
            dtype=str(param.dtype), module=type(module).__module__+'.'+type(module).__name__,
            group=group['group_name'], lr=group['lr'], weight_decay=group['weight_decay'], reasons=reasons))
    need(identities and len(rows)==len(identities), 'empty/incomplete parameter coverage')
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    flat = [id(p) for group in groups for p in group['params']]
    need(len(flat)==len(set(flat)) and set(flat)==expected, 'optimizer parameter identity coverage')
    manifest = dict(schema='k1-parameter-groups/v3', version=VERSION, policy=POLICY,
        parameters=rows, frozen_parameters=frozen, trainable_tensors=len(rows),
        trainable_numel=sum(r['numel'] for r in rows),
        groups=[dict(name=g['group_name'], lr=g['lr'], weight_decay=g['weight_decay'],
                     tensors=len(g['params']), numel=sum(p.numel() for p in g['params'])) for g in groups])
    return groups, manifest


def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def training_groups(model, args):
    """Called at the parent's exact optimizer-construction site, before AdamW/state."""
    from tools.l1_rco_stage0_v1 import read, write
    groups, manifest = parameter_groups(model, lr=args.lr, head_lr=args.head_lr, weight_decay=args.wd, llrd=args.llrd)
    path = Path(args.out) / 'parameter-groups.json'
    if path.exists(): need(read(path)==manifest, 'resume optimizer grouping manifest differs')
    else: write(path, manifest)
    print('K1 v3-G parameter groups: tensors=%d numel=%d manifest=%s' %
          (manifest['trainable_tensors'], manifest['trainable_numel'], manifest_digest(manifest)), flush=True)
    return groups
