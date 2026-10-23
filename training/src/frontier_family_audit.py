"""Static provenance audit of current high-scoring public RSNA knee notebooks.

This script never executes downloaded notebook code. It extracts code text,
input dependencies, hashes, and a few predeclared inference-recipe markers so
that public-score clones are not mistaken for independent model diversity.
"""
from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = workspace_root()
NBROOT = ROOT / "kaggle_notebooks"

NOTEBOOKS = [
    "evgendvorkin__rsna-baseline",
    "mattiaangeli__bend-the-knee-to-the-dinosaurs",
    "maverickss26__rsna-knee-0941-restructured",
    "renta0426__rsna-knee-hybrid-raptor-meniscus-specialist",
    "renta0426__rsna-knee-hybrid-raptor-lateral-meniscus-r100",
    "vaihero__rsna-knee-rentafork",
    "manderson240__cohezion-rsna-knee-sota-ensemble",
    "ask812__bee-knees-fork-v1",
    "prvsiyan__head-and-shoulders-knees-and-toes",
    "romantamrazov__rsna-knee-dinosaur-v4",
    "renta0426__rsna-knee-0-937-weak-label-dinov2-meniscus-resid",
]

MARKERS = {
    "raptor_v5": r"raptor_ft_coatnet_v5",
    "raptor_v8": r"raptor_ft_coatnet_v8",
    "raptor_v10": r"raptor_ft_coatnet_v10",
    "coat_top3": r"coat_resgated_ep10_top3",
    "radimagenet": r"radimagenet|ResNet50\.pt|v52_radimagenet_heads",
    "dino_public": r"dinov2|dino v2|DINOv2",
    "a5": r"A5_W|knee-mri-fold-weights",
    "meniscus": r"meniscus",
    "specialist": r"specialist|residual|r100|ortho",
    "rank_blend": r"rank\(method=['\"]average['\"]|rankpct|argsort\(0\)\.argsort",
    "physical_crop": r"CROP_MM|crop_mm|140\.0|130\.0",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_from_notebook(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8-sig"))
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        chunks.append(source if isinstance(source, str) else "".join(source))
    return "\n\n".join(chunks)


def normalize(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = re.sub(r"\s+", " ", line.strip())
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return "\n".join(lines)


def shingles(text: str, k: int = 5) -> set[str]:
    lines = text.splitlines()
    if len(lines) < k:
        return set(lines)
    return {"\n".join(lines[i:i+k]) for i in range(len(lines)-k+1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    rows = []
    blobs: dict[str, str] = {}
    shingle_map: dict[str, set[str]] = {}
    for dirname in NOTEBOOKS:
        d = NBROOT / dirname
        if not d.exists():
            continue
        meta = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8-sig"))
        files = list(d.glob("*.ipynb")) + list(d.glob("*.py"))
        if not files:
            continue
        path = files[0]
        raw = path.read_bytes()
        code = code_from_notebook(path) if path.suffix == ".ipynb" else path.read_text(encoding="utf-8-sig")
        norm = normalize(code)
        blobs[dirname] = norm
        shingle_map[dirname] = shingles(norm)
        deps = sorted(meta.get("dataset_sources", []))
        models = sorted(meta.get("model_sources", []))
        kernels = sorted(meta.get("kernel_sources", []))
        row = {
            "dir": dirname,
            "ref": meta.get("id"),
            "title": meta.get("title"),
            "file": path.name,
            "file_bytes": len(raw),
            "file_sha256": sha(raw),
            "normalized_code_sha256": sha(norm.encode()),
            "code_lines": len(norm.splitlines()),
            "dataset_count": len(deps),
            "model_count": len(models),
            "kernel_input_count": len(kernels),
            "dependency_signature": sha(("\n".join(deps+models+kernels)).encode()),
            "datasets": ";".join(deps),
            "models": ";".join(models),
            "kernel_inputs": ";".join(kernels),
        }
        for name, pattern in MARKERS.items():
            row[name] = bool(re.search(pattern, code, flags=re.I))
        rows.append(row)

    frame = pd.DataFrame(rows)
    control = "mattiaangeli__bend-the-knee-to-the-dinosaurs"
    frame["shingle_jaccard_to_control"] = frame["dir"].map(
        lambda x: jaccard(shingle_map[x], shingle_map.get(control, set())))
    # Groups of byte-identical or normalized-code-identical notebooks.
    exact_groups = defaultdict(list)
    norm_groups = defaultdict(list)
    dep_groups = defaultdict(list)
    for r in rows:
        exact_groups[r["file_sha256"]].append(r["dir"])
        norm_groups[r["normalized_code_sha256"]].append(r["dir"])
        dep_groups[r["dependency_signature"]].append(r["dir"])
    pairs = []
    names = list(blobs)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            pairs.append({"a": a, "b": b, "shingle_jaccard": jaccard(shingle_map[a], shingle_map[b])})

    frame.to_csv(ROOT / "experiments" / "frontier_notebook_families.csv", index=False)
    pd.DataFrame(pairs).sort_values("shingle_jaccard", ascending=False).to_csv(
        ROOT / "experiments" / "frontier_notebook_similarity.csv", index=False)
    receipt = {
        "notebooks": len(frame),
        "exact_duplicate_groups": [v for v in exact_groups.values() if len(v) > 1],
        "normalized_duplicate_groups": [v for v in norm_groups.values() if len(v) > 1],
        "shared_dependency_groups": [v for v in dep_groups.values() if len(v) > 1],
        "warning": "Static provenance only; public score must be bound to exact Kaggle script version separately.",
    }
    (ROOT / "reports" / "frontier_family_audit.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    cols = ["ref", "file_sha256", "normalized_code_sha256", "shingle_jaccard_to_control",
            "dataset_count", "raptor_v5", "raptor_v8", "raptor_v10", "coat_top3",
            "radimagenet", "dino_public", "a5", "meniscus", "specialist", "rank_blend", "physical_crop"]
    print(frame[cols].to_string(index=False))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
