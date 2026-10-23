"""Build a compact learning dataset (personal Colab study) from the project's 336 px cache.

For each of the 4,343 G1 studies (Gold58 excluded): 12 within-slot-valid window centres (the same rule as the K12
training windows), the CENTRE slice of each window resized 336 -> 224 (INTER_AREA), uint8.
Output (.local/learning-compact-v1/):
  images.npy   uint8 [N, 12, 224, 224]  (~2.6 GB; open with np.load(..., mmap_mode='r'))
  meta.csv     StudyInstanceUID, row, fold (EXP099 validation fold 0-4), the 12 Dread soft labels,
               plus flight_* and relabel_* columns (binary references for comparison)
  README.md    how it was made
  python tools/learning/build_compact_dataset.py [--limit 20] [--out DIR]
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'src'))
from raptor_common import Corpus, select_centers  # noqa: E402

T = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CORPUS = ROOT / 'storage/cache/derived/RAPTOR64_336_C140_S02'
PROTOCOL = ROOT / 'artifacts/experiments/EXP099_KFOLD_PROTOCOL_V1/manifest.json'
LABELS = {'dread': 'storage/labels/dreaddevelopment__rsna-knee-labels/labels_llm_soft.csv',
          'flight': 'storage/labels/flight0234__rsna-knee-hybrid-report-labels/report_labels_v4hybrid.csv',
          'relabel': 'storage/labels/derived_relabel/relabel_v1.csv'}
K, RES = 12, 224


def load(path):
    d = pd.read_csv(ROOT / path, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    cols = [t if t in d.columns else t + '_soft' for t in T]
    out = d[cols]; out.columns = T
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--out', default=str(ROOT / '.local/learning-compact-v1'))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    fold_of = {}
    for k, f in m['folds'].items():
        for u in json.loads((ROOT / f['ordered_ids']['path']).read_text(encoding='utf-8'))['val']:
            fold_of[u] = int(k)
    uids = sorted(fold_of)
    if a.limit:
        uids = uids[:a.limit]
    labs = {k: load(v) for k, v in LABELS.items()}
    corpus = Corpus(CORPUS)
    img = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(len(uids), K, RES, RES))
    t0 = time.time()
    for i, u in enumerate(uids):
        vol, mask = corpus.volume(u), corpus.mask(u)
        cs = select_centers(mask, corpus.depth, K, corpus.spec, 'within-slot-valid', None)
        for j, c in enumerate(cs):
            if c < 0:
                img[i, j] = 0
            else:
                img[i, j] = cv2.resize(np.ascontiguousarray(vol[c]), (RES, RES), interpolation=cv2.INTER_AREA)
        if i % 500 == 0:
            print(i, len(uids), f'{time.time() - t0:.0f}s', flush=True)
    img.flush()
    meta = pd.DataFrame({'StudyInstanceUID': uids, 'row': range(len(uids)), 'fold': [fold_of[u] for u in uids]})
    for t in T:
        meta[t] = labs['dread'].reindex(uids)[t].values
    for name in ('flight', 'relabel'):
        for t in T:
            meta[f'{name}_{t}'] = (labs[name].reindex(uids)[t].values >= 0.5).astype(int)
    meta.to_csv(out / 'meta.csv', index=False)
    (out / 'README.md').write_text(
        '# Compact knee MRI learning set (personal study; do not share outside the team)\n\n'
        f'- {len(uids)} G1 studies (Gold58 excluded), folds from EXP099 (validation fold 0-4).\n'
        f'- images.npy uint8 [N, {K}, {RES}, {RES}]: for each study {K} within-slot-valid window centres (64-slice cache '
        'RAPTOR64_336_C140_S02: 140 mm crop, IOP/IPP-sorted, p2/p98 window, 336 px), centre slice resized to 224 px.\n'
        '- meta.csv: Dread soft labels (training targets) and binarised flight_* / relabel_* references.\n'
        '- Slot layout of the 64-slice cache: sagittal fluid 18, sagittal other 14, coronal fluid 12, coronal other 8, axial 12.\n',
        encoding='utf-8')
    print('done', len(uids), f'{time.time() - t0:.0f}s', out)


if __name__ == '__main__':
    main()
