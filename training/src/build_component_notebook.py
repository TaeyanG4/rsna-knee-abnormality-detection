"""Create a private, bounded Kaggle train-only component reproduction notebook."""
try:
    from ._workspace import workspace_root, workspace_path
except ImportError:
    from _workspace import workspace_root, workspace_path
import json
from pathlib import Path
ROOT=workspace_root()
folder=ROOT/'kaggle_notebooks'/'own_coat_gold_component';folder.mkdir(exist_ok=True)
source=(ROOT/'src'/'kaggle_coat_gold_repro.py').read_text(encoding='utf-8')
launch="from pathlib import Path\nimport subprocess, sys\nsource = "+repr(source)+"\nscript=Path('/kaggle/working/coat_gold_repro.py')\nscript.write_text(source)\nsubprocess.run([sys.executable, str(script)], check=True, timeout=1800)\n"
nb={'cells':[{'cell_type':'markdown','metadata':{},'source':['# Public CoAtNet branch: train Gold58 reproduction\n','No test data or reports. Not the full public ensemble, not a competition submission.\n','Source: mattiaangeli/rsna-knee-coat-resgated-ep10-top3 (declared CC0). Original source/checkpoint hashes are pinned before import.\n']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':launch.splitlines(keepends=True)}],
'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
(folder/'coat_gold_component.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
meta={'id':'taeyangg4/rsna-knee-coat-gold-component-repro','title':'RSNA Knee CoAt gold component reproduction','code_file':'coat_gold_component.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
'dataset_sources':['mattiaangeli/rsna-knee-coat-resgated-ep10-top3'],'competition_sources':['rsna-knee-abnormality-detection'],'model_sources':[],'kernel_sources':[],
'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461','machine_shape':'NvidiaTeslaT4'}
(folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(folder)
