"""HB177 v1.0.4: explicit legacy HB156 loader envelope for truthful EMA payloads.

Changelog: authoritative seed/host provenance; immutable legacy seed42 ABI.
Legacy seed/checkpoint_kind/SWA epochs/provenance are wire-compatibility tags ONLY.
training_provenance is authoritative for actual training; no SWA was performed.
No frozen kernel is edited or executed as a notebook/module body.
"""
from pathlib import Path
import ast
import hashlib
import json
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'.local/scratch/hb177/H-B-177/v1.0.4'
ARCHIVE = ROOT/'artifacts/experiments/HB156_KILLTEST/v1.0.0/dataset/exp152-assets.zip.bin'
FILES = ('kernel-loader.py', 'kernel-model.py', 'kernel-contract.json')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def seal_loader():
    """Local registration only: preserve exact source segments, never rewrite functions."""
    from tools import hb177_build_v1 as original
    if original.serving_inventory() != original.read(original.PROTOCOL)['serving_files']:
        raise ValueError('frozen HB156 serving inventory drift')
    from tools.hb150_io_v1 import atomic_json
    evidence = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        selections = [
            ('exp152_own0949_release.py', FILES[0],
             {'need','sha','state_schema','safe_load','own_meta','checked_model'},
             {'ARCH','CONTRACT','TARGETS','EPOCHS','OWN_NAME','RUN'}),
            ('public-serving.py', FILES[1], {'build_backbone','RaptorClassifier','load_model'}, set()),
        ]
        for member, dest, functions, constants in selections:
            source = z.read(member).decode('utf-8')
            selected = []
            for node in ast.parse(source).body:
                if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions
                    or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)):
                    selected.append(node)
            assert len(selected) == len(functions)+len(constants), 'missing kernel segment'
            text = '\n\n'.join(ast.get_source_segment(source, node) for node in selected)+'\n'
            (STAGE/dest).write_text(text, encoding='utf-8', newline='\n')
            evidence[dest] = dict(member=member, member_sha256=hashlib.sha256(z.read(member)).hexdigest(),
                                  segments=[[n.lineno,n.end_lineno] for n in selected], sha256=sha(STAGE/dest))
        manifest = json.loads(z.read('release-manifest.json'))
    atomic_json(STAGE/FILES[2], dict(archive_sha256=sha(ARCHIVE), source=evidence,
        protocol_sha256=manifest['protocol_sha256'], hb151_export=manifest['hb151_export'],
        own_state_schema=manifest['own_state_schema']))


def kernel_namespace():
    """Load ONLY sealed definitions; no imports, launch code or notebook cells from kernel."""
    import torch
    import timm
    contract = json.loads((STAGE/FILES[2]).read_text(encoding='utf-8'))
    for name in FILES[:2]:
        if sha(STAGE/name) != contract['source'][name]['sha256']:
            raise ValueError('kernel source drift')
    runtime = types.ModuleType('_hb177_frozen_loader')
    runtime.__dict__.update(Path=Path, hashlib=hashlib)
    exec(compile((STAGE/FILES[0]).read_text(encoding='utf-8'), str(STAGE/FILES[0]), 'exec'), runtime.__dict__)
    ns = dict(torch=torch, nn=torch.nn, timm=timm, _E152=runtime)
    exec(compile((STAGE/FILES[1]).read_text(encoding='utf-8'), str(STAGE/FILES[1]), 'exec'), ns)
    return runtime, ns, contract


def export(ema, epoch, protocol, run, recipe, version, *, seed, host):
    """No numerical transform: export the EMA module state, not EMA wrapper keys."""
    rt, _, contract = kernel_namespace()
    return dict(model={k:v.detach().cpu() for k,v in ema.module.state_dict().items()},
        arch=recipe['arch'], seed=42, res=384, inference_res=320, norm='imagenet',
        window_policy='legacy', oai=False, label_set=recipe['labels'], exposure='EXPOSED', clean_oof=False,
        lab=rt.TARGETS, k_eval=94, contract=rt.CONTRACT,
        checkpoint_kind='fixed-swa-bn', swa_epochs_1based=rt.EPOCHS,
        provenance=dict(protocol_sha256=contract['protocol_sha256'], seed=42,
                        run_dir=contract['hb151_export']['run_dir']),
        legacy_loader_compatibility=dict(schema='HB156-frozen-loader-envelope/v1',
            fields=['seed','checkpoint_kind','swa_epochs_1based','provenance'],
            meaning='Legacy ABI tags only; NOT training history. No SWA or BN replay performed.',
            kernel_archive_sha256=contract['archive_sha256']),
        training_provenance=dict(checkpoint_kind='final-epoch-EMA', swa_epochs_1based=[],
            protocol_sha256=sha(protocol), run_dir=run, epoch=epoch, seed=seed, host=host,
            ema_updates=ema.updates, recipe_version=version),
        epoch=epoch, recipe_version=version, fixture=epoch != recipe['epochs'],
        artifact_role='trained-final' if epoch == recipe['epochs'] else 'CPU-SMOKE-FIXTURE-NOT-FOR-SUBMISSION')


def load_with_kernel(path):
    """Call the actual public load_model -> checked_model -> strict=True implementation.

Only bind the replacement checkpoint SHA in memory. Original model schema and
legacy metadata contract remain sealed; no scored stage/manifest is modified.
"""
    rt, ns, contract = kernel_namespace()
    ns['_E152_MANIFEST'] = dict(protocol_sha256=contract['protocol_sha256'],
        hb151_export=contract['hb151_export'], own_state_schema=contract['own_state_schema'],
        checkpoints={rt.OWN_NAME: dict(sha256=sha(path))})
    ns['_E152_FILES'] = {rt.OWN_NAME: str(path)}
    return ns['load_model'](path, rt.ARCH, 384, 'cpu')[0]
