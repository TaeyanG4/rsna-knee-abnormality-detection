"""Local fallback after Kaggle rejected EXP027 due to both GPU slots occupied."""
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
from pathlib import Path
import importlib.util,json
ROOT=workspace_root()
p=ROOT/'kaggle_submission'/'EXP027_RADIODINO_FROZEN'/'exp027_radiodino_frozen.py'
spec=importlib.util.spec_from_file_location('radio_probe',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.OUT=ROOT/'experiments'/'EXP027_RADIODINO_FROZEN_LOCAL'
paths={
 'train_cache_shards_manifest.json':ROOT/'kaggle_notebooks'/'own_exp022_shard_cache'/'train_cache_shards_manifest.json',
 'train.csv':ROOT/'data'/'train.csv',
 'report_labels_v4hybrid.csv':ROOT/'labels'/'flight0234__rsna-knee-hybrid-report-labels'/'report_labels_v4hybrid.csv'
}
def resolve(name):
    p=paths[name]
    if not p.is_file():raise FileNotFoundError(p)
    return p
m.unique_file=resolve
m.OUT.mkdir(exist_ok=True)
(m.OUT/'execution_location.json').write_text(json.dumps({'resource':'local RTX 4070 Ti SUPER','reason':'Kaggle push returned Maximum batch GPU session count of 2 reached','not_submission':True},indent=2),encoding='utf-8')
m.main()
