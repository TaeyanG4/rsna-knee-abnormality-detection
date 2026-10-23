"""HB177 N32 diagnostic v1.0.0; never submit this diagnostic slug.

Changelog: label-free inherited selector and exact release cells, with only a
diagnostic data-root override and FAST disabled. Candidate bytes stay unchanged.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import hb177_postfit_v1 as b
VERSION='1.0.0'
OUT=b.OUT/'stress-kernel'


def selector():
    from tools import exp169_c85_stress as s
    source=Path(s.__file__).read_text(encoding='utf-8'); tree=ast.parse(source)
    functions='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef))
    return 'import collections, csv, json, os, time\nfrom pathlib import Path\n'+\
        f'UID={s.UID!r}\nLABELS={s.LABELS!r}\nCOLS={s.COLS!r}\n'+functions+'\n'


def payload():
    nb=b.read(b.OUT/'kernel/release.ipynb'); meta=b.read(b.OUT/'kernel/kernel-metadata.json')
    setup=selector()+"""
_competition=next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
    Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')] if (p/'train_series.csv').is_file())
_stress_root=Path('/kaggle/working/hb177-n32-data')
_selected=prepare(_competition,_stress_root,n=32)
assert len(set(_selected))==32
FAST=False
print('HB177_N32_DIAGNOSTIC_NEVER_SUBMIT',len(_selected),flush=True)
"""
    # The inherited FAST selector only recognizes public three-study data. This
    # diagnostic explicitly disables it, then runs the exact non-FAST body.
    nb['cells'][0]['source']=setup
    last=''.join(nb['cells'][5]['source'])
    anchor="    _e152.bind(globals(), _manifest, _files, 'own', _work)"
    b.need(last.count(anchor)==1,'diagnostic bind anchor')
    nb['cells'][5]['source']=last.replace(anchor,
        '    find_test_root = lambda: str(_stress_root)\n'+anchor)
    meta.update(id=b.SLUG+'-stress-n32',title=b.SLUG.split('/')[1]+'-stress-n32')
    for cell in nb['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']),'<diagnostic>','exec')
    return nb,meta


def prepare():
    b.validate(); nb,meta=payload()
    for name,value in [('release.ipynb',nb),('kernel-metadata.json',meta)]:
        path=OUT/name
        if path.exists(): b.need(path.read_bytes()==b.encoded(value),'immutable diagnostic drift')
        else: b.write(path,b.encoded(value))
    return dict(status='PREPARED_DIAGNOSTIC_NOT_QUALIFIED',kernel_id=meta['id'],
        candidate_kernel_sha256=b.validate()['kernel_sha256'],
        kernel_files={n:b.sha(OUT/n) for n in ('release.ipynb','kernel-metadata.json')})


def cpu_prepare():
    """Exercise the exact embedded selector with real metadata/DICOM filenames."""
    area=b.SCRATCH/'n32-cpu-data'
    ns={}; exec(compile(selector(),'<embedded-selector>','exec'),ns)
    if area.exists():
        record=b.read(area/'selection.json'); b.need(record['n']==2,'stale selector output')
        # Verify the persisted selection on reruns without deleting data links.
        import csv
        with (area/'test.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        b.need([r['StudyInstanceUID'] for r in rows]==[r['uid'] for r in record['selection']],'selector roster drift')
    else:
        # The local prepared corpus contains two raw studies. Redirect only the
        # train_series mount; metadata and selector code are otherwise exact.
        competition=(ROOT/'storage/competition').resolve()
        mounted=ROOT/'storage/downloads/rsna-knee-abnormality-detection/prepared/9175e113e4a00fe679c4088b/train_series'
        class MountedPath(type(Path())):
            def __truediv__(self,key):
                if self==competition and key=='train_series': return mounted
                return super().__truediv__(key)
        ns['Path']=MountedPath
        ns['prepare'](competition,area,n=2)
    result=dict(status='PASS_REAL_SELECTOR_CPU',studies=2,diagnostic_requested_studies=32,cuda_visible_devices='',
        selection_sha256=b.sha(area/'selection.json'),inference=False)
    b.write(b.OUT/'n32-cpu-prepare.json',b.encoded(result)); return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['prepare','cpu-prepare'])
    a=p.parse_args(); b.cpu_only(); print(json.dumps(prepare() if a.command=='prepare' else cpu_prepare(),indent=2))


if __name__=='__main__': main()
