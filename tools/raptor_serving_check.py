"""Compare explicit DICOM adapter inputs to an immutable cached corpus (CPU only)."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/src'))
from raptor_common import Corpus, UID, sha256_file
from raptor_dicom import DicomCorpus, IndexedZipDicomSource, DicomPreparationError


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--series', required=True)
    ap.add_argument('--folds', required=True)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(Path(a.corpus))
    series = pd.read_csv(a.series, dtype={UID: str, 'SeriesInstanceUID': str})
    folds = pd.read_csv(a.folds, dtype={UID: str})
    valid_ids = [u for u in folds.loc[folds.fold >= 0, UID] if u in corpus]
    healthy = [u for u in valid_ids if corpus.mask(u).sum() >= 3]
    bad = [u for u in valid_ids if not corpus.mask(u).any()]
    counts = series.groupby(UID).size().to_dict()
    rng = np.random.default_rng(20261003)
    ordered = list(rng.permutation(healthy))
    # Protocol-rich and sparse-mask cases plus deterministic random coverage.
    selected = list(dict.fromkeys(sorted(healthy, key=lambda u: (-counts.get(u, 0), u))[:10] +
                                 sorted(healthy, key=lambda u: (int(corpus.mask(u).sum()), u))[:10] + ordered))[:a.count]
    result = {'observed_at': datetime.now(timezone.utc).isoformat(), 'selection_seed': 20261003,
              'selection': '10 most series, 10 smallest nonzero masks, deterministic random fill; development train only',
              'uids': selected, 'scope': 'CPU DICOM-to-cache bytes only, no model quality or full Kaggle runtime',
              'input_hashes': {name: sha256_file(Path(path)) for name, path in
                               [('series', a.series), ('folds', a.folds), ('manifest', Path(a.corpus)/'manifest.json')]},
              'code_hashes': {p.name:sha256_file(p) for p in
                              [Path(__file__), Path(__file__).parents[1]/'training/src/raptor_dicom.py']},
              'studies': [], 'known_empty_studies': bad, 'known_empty_checks': []}
    t0 = time.perf_counter()
    with IndexedZipDicomSource(a.archive, a.index) as source:
        live = DicomCorpus.from_records(selected + bad, series.to_dict('records'), source,
                                       contract=corpus.contract, split='train')
        for uid in selected:
            start = time.perf_counter()
            try:
                volume, mask = live.volume(uid), live.mask(uid)
                ref = corpus.volume(uid)
                same_volume = bool(np.array_equal(volume, ref))
                row = {'uid':uid, 'volume_equal':same_volume,
                       'mask_equal':bool(np.array_equal(mask,corpus.mask(uid))),
                       'max_abs_uint8':int(np.abs(volume.astype(np.int16)-ref.astype(np.int16)).max()),
                       'receipt':live.receipts[uid]}
            except DicomPreparationError as e:
                row = {'uid':uid,'volume_equal':False,'mask_equal':False,'error':str(e),'receipt':e.receipt}
            row['seconds'] = time.perf_counter()-start
            result['studies'].append(row)
            if len(result['studies']) % 10 == 0:
                print(f"{len(result['studies'])}/{len(selected)} studies, {time.perf_counter()-t0:.1f}s", flush=True)
        for uid in bad:
            try:
                live.volume(uid)
                result['known_empty_checks'].append({'uid':uid,'raised':False,'receipt':live.receipts[uid]})
            except DicomPreparationError as e:
                result['known_empty_checks'].append({'uid':uid,'raised':True,'receipt':e.receipt})
    times = [r['seconds'] for r in result['studies']]
    result.update(seconds=time.perf_counter()-t0, matched=sum(r['volume_equal'] and r['mask_equal'] for r in result['studies']),
                  median_study_seconds=float(np.median(times)), p95_study_seconds=float(np.quantile(times,.95)),
                  max_study_seconds=float(max(times)))
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['matched','seconds','median_study_seconds','p95_study_seconds','max_study_seconds']},ensure_ascii=False))
    if result['matched'] != len(selected):
        raise SystemExit('DICOM/cache parity mismatch; inspect receipt before proceeding')


if __name__ == '__main__':
    main()
