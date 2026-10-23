"""Package our own private, train-only CPU audit. No model or test inference."""
from __future__ import annotations
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import json
from pathlib import Path

ROOT=workspace_root()

def main():
    source=(ROOT/"src"/"kaggle_train_geometry_audit.py").read_text(encoding="utf-8")
    folder=ROOT/"kaggle_notebooks"/"own_train_geometry_audit"
    folder.mkdir(exist_ok=True)
    nb={"cells":[{"cell_type":"markdown","metadata":{},"source":["# Training MRI geometry audit\n","Private diagnostic only. No test images, test reports, model weights, or submission predictions are used.\n"]},
                 {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":source.splitlines(keepends=True)}],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
    (folder/"train_geometry_audit.ipynb").write_text(json.dumps(nb,indent=2),encoding="utf-8")
    metadata={"id":"taeyangg4/rsna-knee-training-geometry-audit","title":"RSNA Knee training geometry audit","code_file":"train_geometry_audit.ipynb", "language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_tpu":False,"enable_internet":False,"dataset_sources":[],"competition_sources":["rsna-knee-abnormality-detection"],"kernel_sources":[],"model_sources":[]}
    (folder/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(folder)

if __name__=="__main__":main()
