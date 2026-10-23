"""Stage a private engineering run of the unchanged anchor on 50 or 100 studies.

Build only reads explicit local artifacts and writes a new notebook directory.
The Kaggle wrapper creates a temporary, label-free test proxy using file links;
all original anchor cells remain byte-for-byte equivalent JSON values.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_cell(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


def build(root, selection_path, out, count=50):
    root, selection_path, out = map(Path, (root, selection_path, out))
    if out.exists():
        raise FileExistsError(f"refuse to overwrite staging: {out}")
    if type(count) is not int or count not in (50, 100):
        raise ValueError("benchmark count must be 50 or 100 to retain the 49+ branch")
    source = root / "kaggle/packages/KNEE_ANCHOR_0943"
    metadata = json.loads((source / "kernel-metadata.json").read_text(encoding="utf-8"))
    anchor_path = source / metadata["code_file"]
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    ids = selection["ids"][:count]
    if len(ids) != count or len(set(ids)) != count or set(ids) & set(selection["excluded_corrupt_uids"]):
        raise ValueError(f"expected {count} unique, non-quarantined benchmark studies")
    contract = {"ids": ids, "source_selection_sha256": sha(selection_path),
                "train_series_sha256": selection["train_series_sha256"],
                "anchor_notebook_sha256": sha(anchor_path), "count": count,
                "purpose": "runtime/branch engineering only; these studies may have trained the public models"}
    prefix = f'''from pathlib import Path as _BenchPath
import os as _bench_os, json as _bench_json, hashlib as _bench_hash, time as _bench_time
import pandas as _bench_pd
_BENCH_T0 = _bench_time.perf_counter()
_BENCH_CONTRACT = {_literal(contract)}
_bench_comp = _BenchPath('/kaggle/input/competitions/rsna-knee-abnormality-detection')
assert (_bench_comp/'train_series.csv').is_file(), 'benchmark requires the verified modern competition mount'
assert not _BenchPath('/kaggle/input/rsna-knee-abnormality-detection/test.csv').exists(), 'legacy mount would make original anchor root selection ambiguous'
assert _bench_hash.sha256((_bench_comp/'train_series.csv').read_bytes()).hexdigest() == _BENCH_CONTRACT['train_series_sha256']
_bench_meta = _bench_pd.read_csv(_bench_comp/'train_series.csv', dtype={{'StudyInstanceUID':str, 'SeriesInstanceUID':str}})
_bench_ids = _BENCH_CONTRACT['ids']
_bench_rows = _bench_meta[_bench_meta.StudyInstanceUID.isin(_bench_ids)].copy()
assert set(_bench_rows.StudyInstanceUID) == set(_bench_ids)
_bench_proxy = _BenchPath('/kaggle/working/anchor-benchmark-proxy')
_bench_proxy.mkdir(exist_ok=False)
(_bench_proxy/'test_series').mkdir()
# Only UIDs are needed by original anchor's train-row cache planning.
_bench_pd.DataFrame({{'StudyInstanceUID':_bench_meta.StudyInstanceUID.drop_duplicates().tolist()}}).to_csv(_bench_proxy/'train.csv', index=False)
_bench_pd.DataFrame({{'StudyInstanceUID':_bench_ids}}).to_csv(_bench_proxy/'test.csv', index=False)
_bench_rows.to_csv(_bench_proxy/'test_series.csv', index=False)
_bench_columns = list(_bench_pd.read_csv(_bench_comp/'sample_submission.csv', nrows=0).columns)
_bench_sample = _bench_pd.DataFrame(0.5, index=range(len(_bench_ids)), columns=_bench_columns)
_bench_sample['StudyInstanceUID'] = _bench_ids
_bench_sample.to_csv(_bench_proxy/'sample_submission.csv', index=False)
_bench_links = 0
for _uid, _sid in _bench_rows[['StudyInstanceUID','SeriesInstanceUID']].drop_duplicates().itertuples(index=False, name=None):
    assert '/' not in _uid and '/' not in _sid and '..' not in (_uid,_sid)
    _src = _bench_comp/'train_series'/_uid/_sid
    assert _src.is_dir() and _src.resolve().is_relative_to((_bench_comp/'train_series').resolve())
    _dst = _bench_proxy/'test_series'/_uid/_sid
    _dst.mkdir(parents=True, exist_ok=True)
    for _file in sorted(_src.glob('*.dcm')):
        assert _file.resolve().is_relative_to((_bench_comp/'train_series').resolve())
        (_dst/_file.name).symlink_to(_file)
        _bench_links += 1
assert _bench_links > 0
_bench_os.environ['RSNA_COMP_ROOT'] = str(_bench_proxy)
_bench_os.environ.pop('RSNA_PARALLEL_COAT_READERS', None)
_BENCH_CONTRACT.update(linked_dicom_files=_bench_links, proxy_setup_seconds=_bench_time.perf_counter()-_BENCH_T0, source_reports_read=False)
_BenchPath('/kaggle/working/anchor-benchmark-contract.json').write_text(_bench_json.dumps(_BENCH_CONTRACT, indent=2))
print('Engineering benchmark, {count} train studies, no labels or accuracy metrics; original anchor cells follow.', flush=True)
'''
    suffix = '''# Verify the original anchor reached the intended large-cohort branch.
_bench_result = dict(_RSNA_AUDIT)
assert _bench_result.get('status') == 'COMPLETE'
assert _bench_result.get('coat_readers_parallel_pairs') is False
assert _bench_result.get('coat_family_members') == 4
_bench_frame = _bench_pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID':str})
assert len(_bench_frame) == _BENCH_CONTRACT['count'] and set(_bench_frame.StudyInstanceUID) == set(_bench_ids)
_bench_result.update(benchmark_studies=_BENCH_CONTRACT['count'], total_cell_seconds=_bench_time.perf_counter()-_BENCH_T0,
    anchor_notebook_sha256=_BENCH_CONTRACT['anchor_notebook_sha256'], scope='engineering sample; no quality evidence or hidden-full-runtime guarantee')
_BenchPath('/kaggle/working/anchor-benchmark-result.json').write_text(_bench_json.dumps(_bench_result, indent=2, default=str))
print(_bench_json.dumps(_bench_result, default=str), flush=True)
'''
    notebook = copy.deepcopy(anchor)
    notebook["cells"] = [code_cell(prefix), *anchor["cells"], code_cell(suffix)]
    if notebook["cells"][1:-1] != anchor["cells"]:
        raise AssertionError("original anchor cells changed")
    notebook.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"}
    metadata.update(id=f"taeyangg4/knee-anchor-runtime{count}", title=f"knee-anchor-runtime{count}",
                    code_file=f"anchor-runtime{count}.ipynb", is_private=True, enable_internet=False)
    out.mkdir(parents=True)
    for name, value in [("kernel-metadata.json", metadata), (metadata["code_file"], notebook),
                        ("selection-contract.json", contract)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"state": "STAGED_ONLY", "original_anchor_cells_unchanged": True,
              "original_anchor_cells": len(anchor["cells"]), "studies": count,
              "asset_sources": "original anchor refs retained; exact mounted versions require runtime receipts",
              "notebook_sha256": sha(out / metadata["code_file"]),
              "budget_seconds": 3600, "max_gpu_hours_if_two_t4": 2,
              "scope": "original49+ sequential CoAt path only; not new model quality evaluation"}
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _literal(value):
    return repr(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, choices=(50, 100), default=50)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.selection, args.out, args.count), indent=2))
