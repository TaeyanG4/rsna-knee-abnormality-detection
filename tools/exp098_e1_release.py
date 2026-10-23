"""EXP098 (H-B-12): E1 test-only release = mean of per-target average-tie percentile ranks of B224 and R384.

  python tools/exp098_e1_release.py prepare --stage .local/exp089-recipe-transfer/release-v3 --out .local/exp098-e1-release/v1

Reuses the reviewed release v3 tool unchanged: the notebook embeds its exact source and the two v3 arm manifests,
runs `run` for B224 then R384 (each writes its own validated CSV under /kaggle/working/exp098/), then combines:
for every target, rank each arm's probabilities over the full supplied cohort (average ties, divided by N), average
the two ranks with equal weight, validate (same UID order as sample_submission, 12 columns, finite, in [0,1]) and
publish /kaggle/working/submission.csv atomically. No weights, no reranking afterwards, no tuning. Pre-registered in D-B-001.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'tools/exp089_recipe_release_v31.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


CODE = r'''import time
started=time.monotonic()
import base64,csv,hashlib,json,math,os,subprocess,sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
final=Path('/kaggle/working/submission.csv')
base=Path('/kaggle/working/exp098')
assert not base.exists() and not final.exists()
base.mkdir()
source=base64.b85decode(SOURCE)
assert hashlib.sha256(source).hexdigest()==SOURCE_SHA
tool=base/'exp089_recipe_release.py'; tool.write_bytes(source)
manifests=MANIFESTS
candidates=[Path('/kaggle/input/knee-exp089-recipe-transfer-v1'),Path('/kaggle/input/datasets/taeyangg4/knee-exp089-recipe-transfer-v1')]
assets=[p for p in candidates if (p/'exp089-t4-protocol.json').is_file()]; assert len(assets)==1
opencv_roots=[Path('/kaggle/input/opencv-python-headless-4120088-x86'),Path('/kaggle/input/datasets/mattiaangeli/opencv-python-headless-4120088-x86')]
opencv=[p for d in opencv_roots for p in d.glob('opencv_python_headless-4.12.0.88-*.whl')]; assert len(opencv)==1
comp_roots=[Path('/kaggle/input/rsna-knee-abnormality-detection'),Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
competition=[p for p in comp_roots if all((p/n).is_file() for n in ['test.csv','test_series.csv','sample_submission.csv']) and (p/'test_series').is_dir()]
assert len(competition)==1
n_test=sum(1 for _ in open(competition[0]/'sample_submission.csv',encoding='utf-8-sig'))-1
assert n_test>0
if n_test>3:
    assert 2*1.3*6.52503319258*n_test+600<=30600, 'E1 two-arm time screen failed before any GPU work'
results={}
for arm in ['B224','R384']:
    d=base/arm; d.mkdir()
    (d/'release-manifest.json').write_text(manifests[arm]['text'],encoding='utf-8')
    assert hashlib.sha256((d/'release-manifest.json').read_bytes()).hexdigest()==manifests[arm]['sha256']
    out=d/'arm.csv'
    subprocess.run([sys.executable,str(tool),'run','--manifest',str(d/'release-manifest.json'),'--sha256',manifests[arm]['sha256'],'--assets',str(assets[0]),'--work',str(d/'work'),'--competition',str(competition[0]),'--opencv',str(opencv[0]),'--out',str(out),'--started',str(started),'--bootstrap-max-seconds',('600' if n_test<=3 else '30000')],check=True)
    r=json.loads((d/'work/result.json').read_text()); assert r['status']=='PASS_TEST_COHORT_ONLY'
    assert hashlib.sha256(out.read_bytes()).hexdigest()==r['submission_sha256']
    results[arm]=r
TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
def load(p):
    rows=list(csv.reader(open(p,encoding='utf-8',newline='')))
    assert rows[0]==['StudyInstanceUID',*TARGETS]
    return [r[0] for r in rows[1:]],[[float(x) for x in r[1:]] for r in rows[1:]]
sample=list(csv.reader(open(competition[0]/'sample_submission.csv',encoding='utf-8-sig',newline='')))
ids=[r[0] for r in sample[1:]]
uB,pB=load(base/'B224/arm.csv'); uR,pR=load(base/'R384/arm.csv')
assert uB==ids and uR==ids and len(ids)==len(set(ids))
def rank(col):
    n=len(col); order=sorted(range(n),key=lambda i:col[i]); out=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and col[order[j+1]]==col[order[i]]: j+=1
        v=((i+j)/2+1)/n
        for k in range(i,j+1): out[order[k]]=v
        i=j+1
    return out
cols=[]
for t in range(12):
    rb=rank([r[t] for r in pB]); rr=rank([r[t] for r in pR]); cols.append([(a+b)/2 for a,b in zip(rb,rr)])
assert all(math.isfinite(v) and 0<=v<=1 for c in cols for v in c)
deadline=started+(600 if len(ids)<=3 else 32100)
assert time.monotonic()<deadline
tmp=final.with_name('submission.csv.exp098-tmp')
with open(tmp,'w',encoding='utf-8',newline='') as f:
    w=csv.writer(f); w.writerow(['StudyInstanceUID',*TARGETS])
    for i,u in enumerate(ids): w.writerow([u,*[repr(cols[t][i]) for t in range(12)]])
check=list(csv.reader(open(tmp,encoding='utf-8',newline='')))
assert check[0]==['StudyInstanceUID',*TARGETS] and [r[0] for r in check[1:]]==ids and len(check)==len(ids)+1
os.replace(tmp,final)
if time.monotonic()>=deadline:
    final.unlink(); raise SystemExit('deadline passed during publication')
summary={'status':'PASS_E1_TEST_COHORT_ONLY','formula':'mean of per-target average-tie percentile ranks (B224,R384), equal weights','studies':len(ids),'submission_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'arms':{a:{k:results[a][k] for k in ('submission_sha256','studies','elapsed_seconds','sampled_tree_rss_bytes')} for a in results},'elapsed_seconds':time.monotonic()-started}
(base/'result.json').write_text(json.dumps(summary,indent=1))
print(json.dumps(summary))
'''


def prepare(a):
    stage, out = Path(a.stage), Path(a.out)
    assert not out.exists(), 'fresh output required'
    plan = json.loads((stage / 'execution-plan.json').read_text(encoding='utf-8'))
    assert plan['source_sha256'] == sha(V3), 'v3 tool changed since the stage was prepared'
    manifests = {}
    for arm in ('B224', 'R384'):
        text = (stage / arm / 'release-manifest.json').read_text(encoding='utf-8')
        rec = next(r for r in plan['arms'] if r['arm'] == arm)
        assert sha(stage / arm / 'release-manifest.json') == rec['manifest_sha256']
        manifests[arm] = {'text': text, 'sha256': rec['manifest_sha256']}
    source = V3.read_bytes()
    code = CODE.replace('SOURCE_SHA', repr(sha(V3))).replace('SOURCE', repr(base64.b85encode(source).decode())).replace('MANIFESTS', repr(manifests))
    compile(code, 'e1-notebook', 'exec')
    out.mkdir(parents=True)
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
          'language_info': {'name': 'python', 'version': '3.12'}}, 'cells': [{'cell_type': 'code', 'id': 'e1-rank-mean-release', 'metadata': {},
          'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)}]}
    nbdir = out / 'notebook'; nbdir.mkdir()
    write(nbdir / 'release.ipynb', nb)
    meta = json.loads((stage / 'B224' / 'notebook' / 'kernel-metadata.json').read_text(encoding='utf-8'))
    meta.update(id='taeyangg4/knee-exp098-e1-release-v3', title='knee-exp098-e1-release-v3', code_file='release.ipynb', is_private=True, enable_internet=False)
    write(nbdir / 'kernel-metadata.json', meta)
    rec = dict(status='CPU_PREPARED', created_at=datetime.now(timezone.utc).isoformat(), plan_item='H-B-12', formula='mean of per-target average-tie percentile ranks of B224 and R384',
               v3_source_sha256=sha(V3), arm_manifest_sha256={k: v['sha256'] for k, v in manifests.items()}, notebook_sha256=sha(nbdir / 'release.ipynb'),
               metadata_sha256=sha(nbdir / 'kernel-metadata.json'), kernel=meta['id'],
               push_argv=[str(ROOT / '.venv/Scripts/kaggle.exe'), 'kernels', 'push', '-p', str(nbdir.resolve()), '--timeout', '32400'],
               time_note='both arms share the first-cell clock; each v3.1 run keeps its deadline at started+internal_seconds (600 s for N<=3, 32100 s otherwise); the second arm may launch late (bootstrap max 30000 s for N>3, 600 s for Public3) only if its conservative projection still fits; an up-front two-arm screen runs before any GPU work')
    write(out / 'execution-plan.json', rec)
    print(json.dumps({k: rec[k] for k in ('status', 'kernel', 'notebook_sha256')}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--stage', required=True); p.add_argument('--out', required=True)
    a = ap.parse_args(); prepare(a)


if __name__ == '__main__':
    main()
