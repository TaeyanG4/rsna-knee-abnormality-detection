"""EXP169 registered, CPU-only five-fold partner-bundle screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = Path('artifacts/experiments/EXP169_PARTNER_BUNDLE')
PROTOCOL = EXP / 'PROTOCOL.md'
FAMILIES = ('l2', 'dino', 'convnext')
REFERENCES = ('jev', 'aquila', 'dread', 'consensus')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paths(root, fold):
    base = root / 'artifacts/experiments'
    conv = (base / 'EXP166_CONVNEXT_T/outputs/vast/EXP166_CONVNEXT_T_S42_F0'
            if fold == 0 else base / 'EXP167_CONVNEXT_T_FOLDS/outputs/results' /
            f'EXP167_CONVNEXT_T_S42_F{fold}')
    dino = (root / '.local/exp160-collected-v1/BASE' if fold == 0 else
            base / 'EXP164_DINO_BASE_FOLDS/outputs' / f'EXP164_DINO_BASE_S42_F{fold}')
    h1 = base / (f'EXP149_H1_S42_F{fold}' if fold < 2 else f'EXP153_H1_S42_F{fold}')
    l2_s42 = base / (f'EXP149_L2_S42_F{fold}' if fold < 2 else f'EXP157_L2_S42_F{fold}')
    return dict(l2_s42=l2_s42 / 'oof_k36.csv',
                l2_s43=base / f'EXP157_L2_S43_F{fold}' / 'oof_k36.csv',
                dino=dino / 'oof_best.csv',
                convnext=conv / 'oof_k36_flip_tta.csv',
                h1=h1 / 'oof_k36.csv')


def load_oof(path, ids):
    import numpy as np
    from tools.exp162_dino_folds_kaggle import TARGETS, UID
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames == [UID, *TARGETS], 'OOF columns differ: ' + str(path))
        rows = list(reader)
    by_id = {row[UID]: row for row in rows}
    need(len(by_id) == len(rows) == len(ids) and set(by_id) == set(ids),
         'OOF held-out IDs differ: ' + str(path))
    values = np.asarray([[float(by_id[uid][t]) for t in TARGETS] for uid in ids], np.float64)
    need(values.shape == (len(ids), 12) and np.isfinite(values).all() and
         ((values >= 0) & (values <= 1)).all(), 'Invalid OOF values: ' + str(path))
    return values


def admission(pair):
    d = pair['references']['dread']['delta']
    return {'status': 'ADMIT' if pair['primary_mean_delta'] > 0 and
            pair['primary_mean_ci95'][0] > 0 and d is not None and d >= -.003 else 'HOLD',
            'mean_positive': pair['primary_mean_delta'] > 0,
            'mean_ci_lower_positive': pair['primary_mean_ci95'][0] > 0,
            'dread_floor': d is not None and d >= -.003}


def global_rank(values):
    import numpy as np
    from scipy.stats import rankdata
    a = np.asarray(values, dtype=np.float64)
    need(a.ndim == 2 and a.shape[1] == 12 and np.isfinite(a).all(), 'Invalid rank input')
    return (rankdata(a, axis=0, method='average') - 1.0) / max(1, len(a)-1)


def bundle_of(families):
    import numpy as np
    need(set(families) == set(FAMILIES), 'Exactly three bundle families required')
    shapes = {np.asarray(x).shape for x in families.values()}
    need(len(shapes) == 1, 'Bundle family shapes differ')
    ranked = {name: global_rank(value) for name, value in families.items()}
    return global_rank(sum(ranked.values()) / 3.0), ranked


def _verify_inventory(path):
    inv_path = path.parent / 'output-inventory.json'
    if inv_path.is_file():
        inventory = json.loads(inv_path.read_text(encoding='utf-8'))
        entry = inventory[path.name]
        need(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
             'Prediction inventory differs: ' + str(path))
    else:
        # Vast collected runs use collect/receipt records rather than local trainer inventory.
        receipt = path.parent / 'receipt.json'
        if receipt.is_file():
            r = json.loads(receipt.read_text(encoding='utf-8'))
            need(r.get('status') == 'PASS' and
                 r['outputs'][path.name]['sha256'] == sha(path), 'Prediction receipt differs: ' + str(path))
        elif (path.parent / 'collect-receipt.json').is_file():
            r = json.loads((path.parent / 'collect-receipt.json').read_text(encoding='utf-8'))
            need(r.get('status') == 'COLLECTED_VALIDATED' and
                 r.get('prediction_sha256') == sha(path), 'Fold-0 collect receipt differs')
        else:
            need((path.parent / 'summary.json').is_file(), 'Missing prediction run receipt: ' + str(path))


def readiness(root):
    root = Path(root)
    need((root / PROTOCOL).is_file(), 'Register EXP169 protocol first')
    collection = root / 'artifacts/experiments/EXP167_CONVNEXT_T_FOLDS/outputs/collection.json'
    need(collection.is_file(), 'EXP167 collection is not present; screen held')
    c = json.loads(collection.read_text(encoding='utf-8'))
    need(c.get('status') == 'VERIFIED' and c.get('completed_folds') == [1, 2, 3, 4],
         'EXP167 collection is incomplete; screen held')
    gate = root / 'artifacts/experiments/EXP167_CONVNEXT_T_FOLDS/screen/result.json'
    need(gate.is_file() and json.loads(gate.read_text(encoding='utf-8')).get('verdict') == 'PASS',
         'EXP167 registered pooled gate has not passed; screen held')
    for fold in range(5):
        for path in paths(root, fold).values():
            need(path.is_file(), 'Missing required OOF file: ' + str(path))


def screen(root=ROOT, output=None):
    import numpy as np
    from tools import exp149_label_sources as e149
    from tools import exp115_vlm_eval as ev
    from tools import exp162_dino_folds_kaggle as folds
    from tools import exp156_k_sweep as e156
    root = Path(root)
    output = Path(output) if output else root / EXP / 'screen'
    need(not output.exists(), 'Use a fresh screen output directory')
    readiness(root)
    manifest = e149._load_manifest('student', root)
    canonical, fold_of = e149.fold_rosters(manifest, root)
    ids_by_fold = {f: [uid for uid in canonical if fold_of[uid] == f] for f in range(5)}
    ids = [uid for f in range(5) for uid in ids_by_fold[f]]
    need(len(ids) == len(set(ids)) == 4343, 'Pooled EXP099 coverage differs')
    sections = {n: [] for n in (*FAMILIES, 'h1')}
    files, hashes = {}, {}
    for f in range(5):
        p = paths(root, f)
        for name, path in p.items():
            _verify_inventory(path)
            files[f'{name}_f{f}'] = str(path.relative_to(root))
            hashes[f'{name}_f{f}'] = sha(path)
        fid = ids_by_fold[f]
        s42 = load_oof(p['l2_s42'], fid)
        s43 = load_oof(p['l2_s43'], fid)
        sections['l2'].append((ev.pct_rank(s42) + ev.pct_rank(s43)) / 2.0)
        for name in ('dino', 'convnext', 'h1'):
            sections[name].append(ev.pct_rank(load_oof(p[name], fid)))
    pooled = {name: np.concatenate(part) for name, part in sections.items()}
    bundle, ranked = bundle_of({name: pooled[name] for name in FAMILIES})
    h1 = global_rank(pooled['h1'])
    refs_raw = e156.references(ids, root)
    refs = {name: refs_raw[key] for name, key in
            (('jev', 'jev_hybrid'), ('aquila', 'aquila_v1'), ('dread', 'dread'),
             ('consensus', 'consensus'))}
    draws = ev.bootstrap_index(ev.component_groups(ids, root), 300, 20261003)
    standalone = {name: {ref: folds._score(values, r, ev) for ref, r in refs.items()}
                  for name, values in {**ranked, 'h1': h1, 'bundle': bundle}.items()}
    comparisons = {name: folds._paired(bundle, baseline, refs, draws, ev)
                   for name, baseline in {**ranked, 'h1': h1}.items()}
    result = dict(schema='exp169-bundle-screen-v1', protocol_sha256=sha(root / PROTOCOL),
                  status=admission(comparisons['l2']), studies=len(ids), folds=list(range(5)),
                  bootstrap=dict(unit='validation component', draws=300, seed=20261003),
                  policy=dict(l2='plain oof_k36.csv, two within-fold ranked seeds',
                              dino='plain oof_best.csv, K36',
                              convnext='oof_k36_flip_tta.csv, coronal/axial horizontal mean probability',
                              h1='plain oof_k36.csv'),
                  aggregation='within-fold target ranks; L2 seed mean; pooled global family ranks; equal thirds; global bundle rerank',
                  input_files=files, input_sha256=hashes,
                  standalone=standalone, comparisons=comparisons,
                  reference_sha256={k: sha(root / v) for k, v in
                                    (('jev', e156.JEV), ('aquila', e156.AQUILA),
                                     ('dread', e156.DREAD),
                                     *[(f'consensus_{i}', p) for i, p in enumerate(e156.CONSENSUS)])})
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return {'status': result['status']['status'], 'result': str(output / 'result.json'),
            'bundle_minus_l2': comparisons['l2']['primary_mean_delta'],
            'ci95': comparisons['l2']['primary_mean_ci95'],
            'dread_delta': comparisons['l2']['references']['dread']['delta']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(argv)
    print(json.dumps(screen(args.root, args.output), indent=2))


if __name__ == '__main__':
    main()
