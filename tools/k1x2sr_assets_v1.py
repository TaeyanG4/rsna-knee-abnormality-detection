"""v1.0.0: local-only seed1 asset packaging; no labels, network or GPU.

Changelog: pinned checkpoint plus byte-exact frozen serving trainer. The seed1
training trainer's Net/OFLEncoder AST must match the frozen serving classes.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.local/scratch/astra-b1/k1-shared-render/dataset-k1-v3gs1'
CHECKPOINT = ROOT / '.local/scratch/astra-b2/h210-recovery-v1/attempt/fit/models/E_prior_s0_f0.pt'
TRAINER = ROOT / '.local/scratch/l1-rco/k1v3-seed1/compact_train_l1.py'
SERVING = ROOT / '.local/scratch/slot-b/k2-release/fix1/assets-v1/source/compact_train.py'
MODEL_SHA = '29fc2b2bbaa4e4e8ad341bba17e47c5110388db697458a8cbb9ff8d83fa9399b'
SOURCE_SHA = '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80'
TRAINER_SHA = '8ed7ea4a9fcf3094bbe9b14e9ff69f858c91500abff596609c790eb982a49e34'


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


MANIFEST = dict(schema='k1-v3gs1-assets/v1', version='1.0.0',
    files={'weights/E_prior_s0_f0.pt': MODEL_SHA, 'source/compact_train.py': SOURCE_SHA},
    model_seed=1, model_fold=0, training_contract='15cc54198844a7b503b0d9da87e7d7c678f74072217e5d0c051c08302e3c3fdc',
    training_source_sha256=TRAINER_SHA, serving_architecture='convnext_small; spatial=True; extent=False; crop=352; in_chans=3; state_spec=None',
    architecture_evidence='Net and OFLEncoder AST identical to seed1 training source',
    holdout_sha256='750c9370d453203397c4c03cf3adae5da971e41818f407b1f4d423d800a5006d',
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
        'dataset-metadata.json': encoded(dict(id='taeyangg4/knee-k1-v3gs1-v1-assets',
            title='knee-k1-v3gs1-v1-assets', isPrivate=True, licenses=[dict(name='other')],
            description='Private K1 v3-G seed1 deploy checkpoint and exact frozen serving source. No data, labels or predictions. EXPOSED_EXPLORATORY; clean_oof=false.'))}
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
