"""H-B-25 / EXP108: checks of the native-adjacent triplet cache (CPU only; nothing here trains or touches the GPU).

  python tools/exp108_native3_verify.py dicom [--cache DIR] [--random 8] [--seed 20261007] [--uid UID ...] [--json OUT]
      independent render: raptor_native3.prepare_study_native3 (raptor_dicom header/decode path, IndexedZipDicomSource) vs the cache
      bytes (all 3 planes), and its centre plane vs raptor_dicom.prepare_study (= the 64-slot serving path) and the 64-slot cache;
      the selection adds studies with interior edge replication (short series), duplicate positions, repeated picks and a missing slot.
  python tools/exp108_native3_verify.py consistency [--cache DIR] [--json OUT]
      whole cache: ids / masks equal the 64-slot cache; for every pick, a neighbour that is the pick itself (edge) or another pick of
      the same slot must equal that pick's centre plane byte for byte; neighbour offsets (mm) summary; native vs cache-neighbour change.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training' / 'src'))
import raptor_native3 as n3  # noqa: E402
from raptor_common import CACHE, COMP, Corpus  # noqa: E402

UID, SERIES = 'StudyInstanceUID', 'SeriesInstanceUID'
CACHE_DIR = CACHE / n3.CORPUS_DIRNAME
BASE_DIR = CACHE / n3.BASE_CONTRACT.upper()


def read_picks(cache_dir):
    out = {}
    with open(Path(cache_dir) / 'picks.jsonl', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            out[rec['uid']] = rec
    return out


def slot_offsets(spec):
    return np.cumsum([0] + [int(s[2]) for s in spec['slots']]).tolist()


def interior(spec):
    """Positions that can be window centres under within-slot-valid (never the first/last of a slot)."""
    off = slot_offsets(spec)
    return {c for j in range(len(off) - 1) for c in range(off[j] + 1, off[j + 1] - 1)}


def classify(rec, spec):
    """Features used to choose verification studies."""
    off, inner = slot_offsets(spec), interior(spec)
    kinds = set()
    for j, slot in enumerate(rec['slots']):
        if slot is None or not slot.get('files'):
            kinds.add('missing_slot'); continue
        sl = slot['slices']
        if len(set(sl)) < len(sl):
            kinds.add('repeated_picks')
        for t, (q, pv, nx) in enumerate(zip(sl, slot['prev'], slot['next'])):
            if off[j] + t in inner and (pv == q or nx == q):
                kinds.add('interior_edge')
        if any(abs(d) < 1e-6 for d, pv, q in zip(slot['dpos_prev'], slot['prev'], sl) if pv != q) or \
                any(abs(d) < 1e-6 for d, nx, q in zip(slot['dpos_next'], slot['next'], sl) if nx != q):
            kinds.add('duplicate_position')
    return kinds


def series_rows(uids):
    want, rows = set(uids), {}
    for name in ('train_series.csv', 'test_series.csv'):
        with open(COMP / name, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r[UID] in want:
                    rows.setdefault(r[UID], []).append({UID: r[UID], SERIES: r[SERIES], 'Anatomical_Plane': r['Anatomical_Plane'],
                                                        'Fluid_Sensitive': r.get('Fluid_Sensitive')})
    return rows


def dicom_check(cache_dir, base_dir, uids, picks):
    import raptor_corpus as rc
    from raptor_dicom import DicomCorpus, IndexedZipDicomSource
    zip_path, index_path, _ = rc.archive_paths()
    cache, base = Corpus(cache_dir), Corpus(base_dir)
    rows = series_rows(uids)
    DC3 = n3.dicom_corpus_class()
    results = []
    with IndexedZipDicomSource(zip_path, index_path) as source:
        for uid in uids:
            t0 = time.time()
            split = picks[uid]['split']
            res = dict(uid=uid, split=split, kinds=sorted(classify(picks[uid], cache.spec)))
            errors = {}
            try:
                d3 = DC3.from_records([uid], rows.get(uid, []), source, n3.CONTRACT, split)
                v3, m3 = d3.volume(uid), d3.mask(uid)
            except Exception as exc:
                errors['native3'] = f'{type(exc).__name__}: {exc}'
            try:
                d64 = DicomCorpus.from_records([uid], rows.get(uid, []), source, n3.BASE_CONTRACT, split)
                v64 = d64.volume(uid)
            except Exception as exc:
                errors['base64'] = f'{type(exc).__name__}: {exc}'
            if errors:
                # raptor_dicom raises on an unreadable header / bad archive CRC, the cache builders (raptor_corpus and the native3 builder)
                # place such a file at position 0 and leave its plane empty: a pre-existing serving-vs-cache difference when both fail
                res.update(status='render_failed', error=errors.get('native3'), base64_error=errors.get('base64'),
                           base64_fails_too='native3' in errors and 'base64' in errors and errors['native3'] == errors['base64'],
                           cache_all_zero=not np.asarray(cache.volume(uid)).any())
                results.append(res); print(json.dumps(res), flush=True); continue
            c3 = np.asarray(cache.volume(uid))
            planes_equal = [bool(np.array_equal(v3[:, p], c3[:, p])) for p in range(3)]
            inner = sorted(interior(cache.spec))
            cache_tri_prev = np.asarray(base.volume(uid))[[c - 1 for c in inner]]
            res.update(status='ok', cache_equal_all_planes=all(planes_equal), planes_equal=planes_equal,
                       differing_planes=int(sum(int(not np.array_equal(v3[i, p], c3[i, p])) for i in range(v3.shape[0]) for p in range(3))),
                       centre_equal_dicom64=bool(np.array_equal(v3[:, 1], v64)), centre_equal_base_cache=bool(np.array_equal(c3[:, 1], base.volume(uid))),
                       mask_equal=bool(np.array_equal(m3, cache.mask(uid))) and bool(np.array_equal(m3, base.mask(uid))),
                       interior_prev_differs_from_cache_prev=round(float(np.mean([not np.array_equal(c3[c, 0], cache_tri_prev[t]) for t, c in enumerate(inner)])), 3),
                       edge_replications=d3.receipts[uid]['edge_replications'], neighbour_decode_failures=d3.receipts[uid]['neighbour_decode_failures'],
                       seconds=round(time.time() - t0, 2))
            results.append(res)
            print(json.dumps(res), flush=True)
    ok = [r for r in results if r['status'] == 'ok']
    return dict(studies=len(results), rendered=len(ok), cache_equal=sum(r['cache_equal_all_planes'] for r in ok),
                centre_equal_dicom64=sum(r['centre_equal_dicom64'] for r in ok), centre_equal_base_cache=sum(r['centre_equal_base_cache'] for r in ok),
                mask_equal=sum(r['mask_equal'] for r in ok), render_failed=[dict(uid=r['uid'], error=r['error'], base64_fails_too=r['base64_fails_too'], cache_all_zero=r['cache_all_zero'])
                                                                          for r in results if r['status'] != 'ok'],
                kinds={k: sum(k in r['kinds'] for r in results) for k in ('interior_edge', 'duplicate_position', 'repeated_picks', 'missing_slot')},
                results=results)


def choose(picks, spec, n_random, seed, explicit):
    uids = list(explicit)
    pool = sorted(picks)
    rng = random.Random(seed)
    uids += [u for u in rng.sample(pool, min(n_random, len(pool))) if u not in uids]
    by_kind = {}
    for u in pool:
        for k in classify(picks[u], spec):
            by_kind.setdefault(k, []).append(u)
    for kind in ('interior_edge', 'duplicate_position', 'repeated_picks', 'missing_slot'):
        cands = [u for u in by_kind.get(kind, []) if u not in uids]
        uids += rng.sample(cands, min(2 if kind != 'missing_slot' else 1, len(cands)))
    return uids, {k: len(v) for k, v in by_kind.items()}


def consistency(cache_dir, base_dir):
    cache, base = Corpus(cache_dir), Corpus(base_dir)
    picks = read_picks(cache_dir)
    off, inner = slot_offsets(cache.spec), interior(cache.spec)
    out = dict(studies=len(cache), ids_equal_base=bool(np.array_equal(cache.ids, base.ids)), masks_equal_base=bool(np.array_equal(cache.masks, base.masks)),
               picks_records=len(picks), checks=0, failures=0, failure_examples=[], edge_checks=0, pick_checks=0)
    dpos, kinds, changed, interior_edge_studies = [], {}, [], 0
    t0 = time.time()
    for i, uid in enumerate(cache.ids):
        rec = picks[uid]
        vol = None
        for k in classify(rec, cache.spec):
            kinds[k] = kinds.get(k, 0) + 1
        interior_edge_studies += 'interior_edge' in classify(rec, cache.spec)
        for j, slot in enumerate(rec['slots']):
            if slot is None or not slot.get('files'):
                continue
            sl = slot['slices']
            first = {}
            for t, q in enumerate(sl):
                first.setdefault(q, t)
            for t, q in enumerate(sl):
                pos = off[j] + t
                if not cache.masks[i, pos]:
                    continue
                if vol is None:
                    vol = np.asarray(cache.volume(uid))
                for channel, nb, d in ((0, slot['prev'][t], slot['dpos_prev'][t]), (2, slot['next'][t], slot['dpos_next'][t])):
                    if nb == q:
                        ref, kind = vol[pos, 1], 'edge'
                    elif nb in first and cache.masks[i, off[j] + first[nb]]:
                        ref, kind = vol[off[j] + first[nb], 1], 'pick'
                    else:
                        if pos in inner:
                            dpos.append(abs(d))
                        continue
                    out['checks'] += 1; out[kind + '_checks'] += 1
                    if pos in inner and nb != q:
                        dpos.append(abs(d))
                    if not np.array_equal(vol[pos, channel], ref):
                        out['failures'] += 1
                        if len(out['failure_examples']) < 5:
                            out['failure_examples'].append(dict(uid=uid, pos=pos, channel=channel, nb=nb, kind=kind))
        if i % 200 == 0 and vol is not None:  # how often the native neighbour differs from the cache neighbour at interior centres
            b = np.asarray(base.volume(uid))
            changed += [not np.array_equal(vol[c, 0], b[c - 1]) for c in inner if cache.masks[i, c - 1:c + 2].all()]
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(cache)} studies, {time.time() - t0:.0f}s, failures {out["failures"]}', flush=True)
    d = np.asarray(dpos)
    out.update(kinds=kinds, interior_edge_studies=interior_edge_studies, seconds=round(time.time() - t0),
               interior_neighbour_offset_mm=dict(n=int(d.size), median=float(np.median(d)), p05=float(np.percentile(d, 5)), p95=float(np.percentile(d, 95)),
                                                 zero=int((d < 1e-6).sum())),
               sampled_interior_prev_differs_from_cache_prev=round(float(np.mean(changed)), 4) if changed else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    s = sub.add_parser('dicom'); s.add_argument('--cache', default=str(CACHE_DIR)); s.add_argument('--base', default=str(BASE_DIR))
    s.add_argument('--random', type=int, default=8); s.add_argument('--seed', type=int, default=20261007); s.add_argument('--uid', action='append', default=[])
    s.add_argument('--json', default=None)
    s = sub.add_parser('consistency'); s.add_argument('--cache', default=str(CACHE_DIR)); s.add_argument('--base', default=str(BASE_DIR))
    s.add_argument('--json', default=None)
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if a.command == 'dicom':
        picks = read_picks(a.cache)
        spec = Corpus(Path(a.cache)).spec
        uids, kind_counts = choose(picks, spec, a.random, a.seed, a.uid)
        report = dicom_check(Path(a.cache), Path(a.base), uids, picks)
        report['cache_kind_counts'] = kind_counts
    else:
        report = consistency(Path(a.cache), Path(a.base))
    text = json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=1)
    print(text)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
