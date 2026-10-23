"""v1.0.0: local-only seed0 asset packaging; no labels, network or GPU.

Changelog: pinned checkpoint plus byte-exact frozen serving trainer. The seed0
training trainer's Net/OFLEncoder AST must match the frozen serving classes.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.local/scratch/astra-b1/k1-shared-render/dataset-k1-v3gs0'
CHECKPOINT = ROOT / '.local/scratch/l1-rco/k1v3-grouping/fit/models/E_prior_s0_f0.pt'
TRAINER = ROOT / '.local/scratch/l1-rco/k1v3-grouping/compact_train_l1.py'
SERVING = ROOT / '.local/scratch/slot-b/k2-release/fix1/assets-v1/source/compact_train.py'
MODEL_SHA = '6c03830df077c34d712f7764d43f6f9a828f50666e7c42ed1e19665fc5063bd5'
SOURCE_SHA = '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80'
TRAINER_SHA = '0d5fbae473eb1dc451871be0d610689d6baff8e6ac258acb091e66343ad0b791'


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


MANIFEST = dict(schema='k1-v3gs0-assets/v1', version='1.0.0',
    files={'weights/E_prior_s0_f0.pt': MODEL_SHA, 'source/compact_train.py': SOURCE_SHA},
    model_seed=0, model_fold=0, provenance='local k1v3-grouping final deploy checkpoint',
    training_source_sha256=TRAINER_SHA, serving_architecture='convnext_small; spatial=True; extent=False; crop=352; in_chans=3; state_spec=None',
    architecture_evidence='Net and OFLEncoder AST identical to seed0 training source',
    clean_oof=False, exposure='EXPOSED_EXPLORATORY')
MANIFEST_SHA = hashlib.sha256(encoded(MANIFEST)).hexdigest()


def run(command):
    from tools import c85r7_rco_overlay_v1 as util
    assert util.sha(CHECKPOINT) == MODEL_SHA
    assert util.sha(SERVING) == SOURCE_SHA and util.sha(TRAINER) == TRAINER_SHA
    a, b = [ast.parse(p.read_bytes()) for p in (SERVING, TRAINER)]
    for name in ('Net', 'OFLEncoder'):
        node = lambda tree: next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        assert ast.dump(node(a)) == ast.dump(node(b)), name
    outputs = {'weights/E_prior_s0_f0.pt': CHECKPOINT.read_bytes(),
        'source/compact_train.py': SERVING.read_bytes(), 'asset-manifest.json': encoded(MANIFEST),
        'dataset-metadata.json': encoded(dict(id='taeyangg4/knee-k1-v3gs0-v1-assets',
            title='knee-k1-v3gs0-v1-assets', isPrivate=True, licenses=[dict(name='other')],
            description='Private K1 v3-G seed0 deploy checkpoint and exact frozen serving source. No data, labels or predictions. EXPOSED_EXPLORATORY; clean_oof=false.'))}
    if command == 'prepare':
        for name, payload in outputs.items():
            util.write_once(DEST/name, payload)
    assert {p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()} == set(outputs)
    for name, payload in outputs.items():
        assert (DEST/name).read_bytes() == payload, name
    return dict(status='PASS_LOCAL_ONLY', directory=str(DEST), files=len(outputs),
                model_sha256=MODEL_SHA, manifest_sha256=MANIFEST_SHA,
                architecture_ast_identical=True, bytes=sum(map(len, outputs.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate'])
    print(json.dumps(run(parser.parse_args().command), indent=2))
