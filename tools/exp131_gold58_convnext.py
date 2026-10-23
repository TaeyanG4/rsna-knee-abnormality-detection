"""Build and compare the Apex ConvNeXt reader on the EXP130 Gold58 cohort.

The attached reader is untrusted third-party code. This module only packages
and inspects it locally; its infer.py runs in a child process on Kaggle.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import exp130_gold58_compare as e130

BASE = ROOT / ".local/exp131-gold58-cnx"
EXP130 = ROOT / ".local/exp130-gold58"
APEX = ROOT / "storage/external/sujanmajhisuzan__rsna-knee-apex-grandmaster-stack/rsna-knee-apex-grandmaster-stack.ipynb"
KERNEL_ID = "taeyangg4/knee-exp131-cnx-gold58"
DATASET = "goodpjw2008/rsna-knee-2-5d-convnext-reader"

# Transcribed from the Apex notebook TARGET_WEIGHTS; check against its AST at build/compare.
TARGET_WEIGHTS = {
    "Baker's": .35, "Contusion": .30, "Medial OA": .25, "ACL": .25,
    "Medial Meniscus": .20, "MCL": .20, "Lateral Meniscus": .08,
    "Fracture": .08, "Lateral OA": .08, "PF OA": .08,
    "Effusion": .04, "Synovitis": .03,
}


def write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def source_cell(path: Path, marker: str, *, at_start: bool = False) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    matches = [source for source in cells if (source.lstrip().startswith(marker) if at_start else marker in source)]
    e130.need(len(matches) == 1, f"expected one notebook cell containing {marker!r}")
    return matches[0]


def gold_uids(path: Path) -> list[str]:
    source = source_cell(path, "GOLD58_UIDS =")
    assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "GOLD58_UIDS" for target in node.targets)]
    e130.need(len(assignments) == 1, "EXP130 embedded UID assignment differs")
    uids = list(ast.literal_eval(assignments[0].value))
    e130.need(len(uids) == 58 and len(set(uids)) == 58 and all(isinstance(uid, str) and uid for uid in uids),
              "EXP130 Gold58 UID constant invalid")
    return uids


def check_apex(path: Path = APEX) -> str:
    source = source_cell(path, "# ====", at_start=True)
    assignments = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "TARGET_WEIGHTS" for target in node.targets)]
    e130.need(len(assignments) == 1 and ast.literal_eval(assignments[0].value) == TARGET_WEIGHTS,
              "Apex TARGET_WEIGHTS drift")
    e130.need(set(TARGET_WEIGHTS) == set(e130.TARGETS), "Apex target set differs")
    return e130.sha256(path)


def kernel_code(uids: list[str]) -> str:
    """Only a wrapper: import the attached dataset's infer.py in a child process."""
    return '''import glob, hashlib, json, os, pathlib, subprocess, sys, time
import numpy as np
import pandas as pd
import torch

GOLD58_UIDS = ''' + repr(tuple(uids)) + '''
TARGETS = ''' + repr(e130.TARGETS) + '''
START = time.time()
assert len(GOLD58_UIDS) == len(set(GOLD58_UIDS)) == 58
assert torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0)

def asset_dir():
    paths = [pathlib.Path('/kaggle/input/rsna-knee-2-5d-convnext-reader'),
             pathlib.Path('/kaggle/input/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader')]
    matches = [p for p in paths if (p / 'infer.py').is_file()]
    assert len(matches) == 1, 'ConvNeXt dataset infer.py not found uniquely'
    p = matches[0]
    assert (p / 'knee.py').is_file() and (p / 'preprocess.py').is_file()
    return p

assets = asset_dir()
data_candidates = [pathlib.Path('/kaggle/input/rsna-knee-abnormality-detection'),
                   pathlib.Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
data = next((p for p in data_candidates if (p / 'train_series.csv').is_file()
             and (p / 'train_series').is_dir()), None)
assert data is not None, 'competition training DICOM root missing'
ckpts = [assets / f'cnxt_v0_fold{i}.pt' for i in range(3)]
assert all(p.is_file() for p in ckpts), 'three ConvNeXt checkpoints required'

# Match the Apex notebook's offline decoder installation.
wheel_target = '/kaggle/working/_own_env'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                '--target', wheel_target, '--find-links', str(assets), 'pylibjpeg',
                'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'], check=True)
timm_wheels = sorted(glob.glob('/kaggle/input/*/timm-*.whl') +
                     glob.glob('/kaggle/input/*/*/timm-*.whl'))
if timm_wheels:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
                    '--target', wheel_target, timm_wheels[0]], check=True)
READY = time.time()
out = pathlib.Path('/kaggle/working/gold58_cnx.csv')
env = dict(os.environ)
env['PYTHONPATH'] = wheel_target + os.pathsep + env.get('PYTHONPATH', '')
env['HF_HUB_OFFLINE'] = '1'
env['TRANSFORMERS_OFFLINE'] = '1'
# The dataset infer.run accepts an explicit study list; invoke it in a child
# interpreter so its imports and GPU state remain isolated from this wrapper.
child = """import importlib.util, json, pathlib, sys
assets, data, out, ids, *ckpts = sys.argv[1:]
sys.path.insert(0, assets)
spec = importlib.util.spec_from_file_location('infer', str(pathlib.Path(assets) / 'infer.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.run(data, 'train', ckpts, studies=json.loads(ids))
result.to_csv(out)
"""
subprocess.run([sys.executable, '-c', child, str(assets), str(data), str(out),
                json.dumps(GOLD58_UIDS), *map(str, ckpts)], env=env, check=True)
INFER_DONE = time.time()
pred = pd.read_csv(out, dtype={'StudyInstanceUID': str})
assert list(pred.columns) == ['StudyInstanceUID', *TARGETS]
assert pred.StudyInstanceUID.tolist() == list(GOLD58_UIDS)
values = pred[TARGETS].to_numpy(dtype=float)
assert values.shape == (58, 12) and np.isfinite(values).all()
assert ((values >= 0) & (values <= 1)).all()
timing = {'schema': 'exp131-gold58-cnx-timing-v1', 'studies': 58,
          'checkpoint_count': 3, 'gpu': torch.cuda.get_device_name(0),
          'seconds_setup': READY - START, 'seconds_inference': INFER_DONE - READY,
          'seconds_total': time.time() - START,
          'seconds_per_study_total': (time.time() - START) / 58,
          'csv_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
          'reader': 'attached goodpjw2008 infer.py run(studies=Gold58) in subprocess'}
pathlib.Path('/kaggle/working/gold58_cnx_timing.json').write_text(
    json.dumps(timing, indent=2) + '\\n', encoding='utf-8')
print('Gold58 ConvNeXt predictions:', len(pred), 'seconds:', timing['seconds_total'])
'''


def build(root: Path = ROOT, dest: Path = BASE / "cnx_gold58") -> dict:
    root, dest = Path(root), Path(dest)
    exp130_nb = root / ".local/exp130-gold58/oai_gold58/release.ipynb"
    apex = root / "storage/external/sujanmajhisuzan__rsna-knee-apex-grandmaster-stack/rsna-knee-apex-grandmaster-stack.ipynb"
    uids = gold_uids(exp130_nb)
    apex_hash = check_apex(apex)
    code = kernel_code(uids)
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": [
                        "# EXP131 Gold58 ConvNeXt diagnostic\n",
                        "Reader and weights: goodpjw2008/rsna-knee-2-5d-convnext-reader, Apache 2.0. ",
                        "Competition training images only. Source runs out of process from the attached dataset. ",
                        "Gold58 is exposed; this diagnostic is report only.\n"]},
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": code.splitlines(keepends=True)}]}
    metadata = {"id": KERNEL_ID, "title": "knee-exp131-cnx-gold58", "code_file": "release.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": [DATASET], "competition_sources": ["rsna-knee-abnormality-detection"],
                "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
                "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"}
    e130.need(not dest.exists(), "EXP131 kernel directory exists; refusing overwrite")
    write_new(dest / "release.ipynb", json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    write_new(dest / "kernel-metadata.json", json.dumps(metadata, indent=2) + "\n")
    receipt = {"schema": "exp131-build-v1", "status": "UNPUSHED", "uids": 58,
               "uid_source_sha256": e130.sha256(exp130_nb), "apex_source_sha256": apex_hash,
               "notebook_sha256": e130.sha256(dest / "release.ipynb"),
               "metadata_sha256": e130.sha256(dest / "kernel-metadata.json")}
    write_new(dest / "build-receipt.json", json.dumps(receipt, indent=2) + "\n")
    return receipt


def weighted_rank(oai: list[list[float]], cnx: list[list[float]]) -> list[list[float]]:
    a, b = e130.rank_matrix(oai), e130.rank_matrix(cnx)
    blended = [[(1 - TARGET_WEIGHTS[target]) * x + TARGET_WEIGHTS[target] * y
                for target, x, y in zip(e130.TARGETS, ra, rb)] for ra, rb in zip(a, b)]
    return e130.rank_matrix(blended)


def blend(*arms: tuple[float, list[list[float]]]) -> list[list[float]]:
    e130.need(abs(sum(w for w, _ in arms) - 1) < 1e-12, "blend weights must sum to one")
    n = len(arms[0][1])
    e130.need(all(len(rows) == n for _, rows in arms), "blend cohort differs")
    return e130.rank_matrix([[sum(weight * rows[i][t] for weight, rows in arms)
                              for t in range(len(e130.TARGETS))] for i in range(n)])


def spearman(a: list[float], b: list[float]) -> float | None:
    x, y = e130.rank_average(a), e130.rank_average(b)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    xx = sum((v - mx)**2 for v in x)
    yy = sum((v - my)**2 for v in y)
    return sum((v - mx) * (w - my) for v, w in zip(x, y)) / math.sqrt(xx * yy) if xx and yy else None


def compare(oai: Path = EXP130 / "oai-output/gold58_oai.csv",
            cnx: Path = BASE / "cnx-output/gold58_cnx.csv",
            local: Path = EXP130 / "local", root: Path = ROOT,
            out: Path = BASE / "comparison", draws: int = 1000) -> dict:
    root, oai, cnx, local, out = map(Path, (root, oai, cnx, local, out))
    e130.need(draws > 0 and not out.exists(), "draws must be positive and output must be new")
    apex_hash = check_apex(root / APEX.relative_to(ROOT))
    ids, labels = e130.official_gold(root)
    e130.need(ids == gold_uids(root / ".local/exp130-gold58/oai_gold58/release.ipynb"),
              "official Gold58 differs from EXP130 embedded UIDs")
    raw_oai = e130.prediction_csv(oai, ids)
    raw_cnx = e130.prediction_csv(cnx, ids)
    s42 = e130.prediction_csv(local / "gold58_s42_plain.csv", ids)
    s42_tta = e130.prediction_csv(local / "gold58_s42_tta.csv", ids)
    s43_tta = e130.prediction_csv(local / "gold58_s43_tta.csv", ids)
    oai_rank, cnx_rank = e130.rank_matrix(raw_oai), e130.rank_matrix(raw_cnx)
    radr = e130.mix(e130.rank_matrix(s42_tta), e130.rank_matrix(s43_tta), .5)
    radr_rank = e130.rank_matrix(radr)
    candidates = {
        "oai_0949": raw_oai, "convnext": raw_cnx,
        "radr50_s42": s42, "radr50_s42_tta": s42_tta,
        "radr50_2seed_rank_tta": radr,
        "apex_0950": weighted_rank(raw_oai, raw_cnx),
        "B1_cnx_085_015": blend((.85, oai_rank), (.15, cnx_rank)),
        "B2_radr_090_010": blend((.9, oai_rank), (.1, radr_rank)),
        "B3_both_080_010_010": blend((.8, oai_rank), (.1, cnx_rank), (.1, radr_rank)),
    }
    counts = {target: {"positive": sum(row[t] for row in labels),
                       "negative": len(labels) - sum(row[t] for row in labels)}
              for t, target in enumerate(e130.TARGETS)}
    eligible = [t for t, target in enumerate(e130.TARGETS) if min(counts[target].values()) >= 3]
    e130.need(bool(eligible), "no eligible Gold58 targets")
    result = e130.scores(labels, candidates, eligible)
    pairs = [(name, "oai_0949") for name in candidates if name != "oai_0949"]
    pairs += [("B1_cnx_085_015", "B2_radr_090_010")]
    ci, rejected = e130.bootstrap(labels, candidates, eligible, pairs, draws=draws)
    differences = {f"{a} - {b}": {"delta": result[a]["macro"] - result[b]["macro"],
                                  **ci[f"{a} - {b}"]} for a, b in pairs}
    correlations = {}
    for name, rows in (("convnext", raw_cnx), ("radr50_2seed_rank_tta", radr)):
        per = {target: spearman([row[t] for row in raw_oai], [row[t] for row in rows])
               for t, target in enumerate(e130.TARGETS)}
        defined = [per[e130.TARGETS[t]] for t in eligible if per[e130.TARGETS[t]] is not None]
        correlations[name] = {"per_target": per,
                              "mean_eligible": sum(defined) / len(defined) if defined else None,
                              "defined_eligible_targets": len(defined)}
    inputs = {"oai": oai, "cnx": cnx, "s42_plain": local / "gold58_s42_plain.csv",
              "s42_tta": local / "gold58_s42_tta.csv", "s43_tta": local / "gold58_s43_tta.csv"}
    body = {"schema": "exp131-gold58-comparison-v1", "studies": 58,
            "counts": counts, "macro_targets": [e130.TARGETS[t] for t in eligible],
            "excluded_from_macro": [t for t in e130.TARGETS if t not in [e130.TARGETS[i] for i in eligible]],
            "models": result, "target_weights": TARGET_WEIGHTS,
            "partner_spearman_vs_oai": correlations,
            "paired_study_bootstrap": {"draws": draws, "seed": 20261008,
                "rejected_draws_under_3_per_class": rejected, "differences": differences},
            "inputs_sha256": {**{key: e130.sha256(path) for key, path in inputs.items()},
                              "exp130_uid_notebook": e130.sha256(root / ".local/exp130-gold58/oai_gold58/release.ipynb")},
            "apex_source_sha256": apex_hash,
            "interpretation": "Report only. Gold58 is exposed and enriched. Apex per-target weights were likely chosen on Gold58/public and are optimistic on Gold58. Bootstrap resamples studies, retaining the original 58-study rank blends and excluding targets with under three positives or negatives. This does not estimate private leaderboard lift."}
    out.mkdir(parents=True)
    write_new(out / "comparison.json", json.dumps(body, indent=2, allow_nan=False) + "\n")
    lines = ["# EXP131 Gold58 ConvNeXt comparison", "", body["interpretation"], "",
             "| Model | Macro AUC |", "|---|---:|"]
    lines.extend(f"| {name} | {score['macro']:.4f} |" for name, score in result.items())
    lines += ["", "| Difference | Delta | 95% paired CI | P(delta > 0) |", "|---|---:|---:|---:|"]
    for name, value in differences.items():
        lines.append(f"| {name} | {value['delta']:+.4f} | [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}] | {value['share_positive']:.3f} |")
    lines += ["", "| Partner vs 0.949 | Mean eligible target Spearman |", "|---|---:|"]
    lines.extend(f"| {name} | {value['mean_eligible']:.4f} |" if value["mean_eligible"] is not None
                 else f"| {name} | undefined |" for name, value in correlations.items())
    lines += ["", "Target counts, exclusions, per-target AUC and Spearman are in comparison.json."]
    write_new(out / "comparison.md", "\n".join(lines) + "\n")
    return {"studies": 58, "macro_targets": len(eligible),
            "macros": {name: value["macro"] for name, value in result.items()}, "output": str(out)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--out", type=Path, default=BASE / "cnx_gold58")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--oai", type=Path, default=EXP130 / "oai-output/gold58_oai.csv")
    compare_parser.add_argument("--cnx", type=Path, default=BASE / "cnx-output/gold58_cnx.csv")
    compare_parser.add_argument("--local", type=Path, default=EXP130 / "local")
    compare_parser.add_argument("--out", type=Path, default=BASE / "comparison")
    compare_parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args(argv)
    result = build(dest=args.out) if args.action == "build" else compare(
        oai=args.oai, cnx=args.cnx, local=args.local, out=args.out, draws=args.draws)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
