from __future__ import annotations

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .store import Store, TARGETS, dumps, now


def read_json(path):
    raw=path.read_bytes()
    return json.loads(raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig'))


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def tags(text):
    names=['Raptor','ConvNeXt','CoAtNet','DINO','OOF','weak','Gold58','Synovitis','Lateral OA','Meniscus','runtime','label']
    return [t for t in names if t.casefold() in text.casefold()]


def file_time(p):
    return datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()


def import_evidence(store: Store):
    path=store.ws.resolve('reports/dashboard_design_20261001/evidence.json')
    if not path.is_file():return 0
    data=read_json(path)
    observed=data['observed_at_utc'].replace(' UTC','+00:00').replace(' ','T')
    store.set_meta('competition',dumps(data['official']['metadata']))
    for page in data['official']['pages']:
        name=page['name']
        suffix='rules' if name=='rules' else 'data' if name=='data-description' else 'overview/'+name.lower().replace(' ','-')
        store.upsert_source(id='competition:'+name,kind='official',title=name,content=page['content'],
            authority='official',url='https://www.kaggle.com/competitions/'+data['competition']+'/'+suffix,
            observed_at=observed,tags=['공식'])
    for row in data['own_submissions']['records']:store.upsert_submission(row,observed)
    store.leaderboard(data['leaderboard']['records'],observed)
    for topic in data['discussions']:
        content='\n\n'.join(f"[{m['id']}] {m.get('author','')} · {m.get('author_type','participant')} · {m.get('date','')}\n{m['text']}" for m in topic['messages'])
        title={733965:'Use of Commercially Hosted LLMs',735304:'Best single-model score',743416:'External knee MRI datasets — clarification'}.get(topic['id'],str(topic['id']))
        store.upsert_source(id='topic:'+str(topic['id']),kind='discussion',title=title,content=content,url=topic['url'],
            tags=tags(content),observed_at=observed,payload={'coverage':topic['coverage'],'messages':topic['messages']})
    for row in data['public_notebooks']['records']:
        if store.rows('SELECT id FROM sources WHERE id=?', ('notebook:'+row['ref'],)): continue
        store.upsert_source(id='notebook:'+row['ref'],kind='notebook',title=row['title'],url='https://www.kaggle.com/code/'+row['ref'],
            content='공개 목록에서 발견한 노트북입니다. 숫자 점수와 소스 재현은 확인하지 않았습니다.',
            observed_at=observed,tags=tags(row['title']),payload=row)
    store.connector('saved_evidence','ok','저장된 2026-10-01 조사 snapshot을 불러왔습니다. 현재 live 조회와 구분합니다.','snapshot; discussions partial, LB top 10')
    return len(data['official']['pages'])


def run_record(store, path):
    data=clean(read_json(path))
    if not isinstance(data,dict):raise ValueError('summary must be an object')
    name=path.parent.name
    args=data.get('args') or {}
    if not isinstance(args,dict):args={}
    targets=data.get('selected_targets') or data.get('targets') or TARGETS
    if not isinstance(targets,list) or not all(isinstance(x,str) for x in targets):targets=TARGETS
    if isinstance(data.get('targets'),int):targets=TARGETS
    family=str(data.get('backbone') or args.get('backbone') or ('Raptor' if 'RAPTOR' in name.upper() else 'ConvNeXt' if 'CONVNEXT' in name.upper() else 'other'))
    warning=[]; score=None; metric_name=''; kind='diagnostic'
    keys=[('weak_oof_macro_auc','weak_oof'),('selected_checkpoint_weak_highconf_auc','weak_oof'),
          ('best_weak_highconf_auc','weak_oof'),('blend50_macro_auc','weak_oof'),
          ('fixed_blend_macro','pseudo_oof_leaky'),('macro_auc','diagnostic')]
    for key,k in keys:
        value=data.get(key)
        if isinstance(value,(int,float)) and math.isfinite(value):score=float(value);kind=k;metric_name=key;break
    if 'pseudo' in str(data.get('caveat','')).lower() or 'PSEUDOOOF' in name.upper():kind='pseudo_oof_leaky'
    if 'weak' in kind:warning.append('보고서 기반 weak-label 대리지표입니다.')
    if kind=='pseudo_oof_leaky':warning.append('공개 backbone의 학습 노출로 독립 OOF가 아닙니다.')
    if len(targets)!=12:warning.append(f'{len(targets)}개 선택 target 결과이며 대회 12-target 점수가 아닙니다.')
    if data.get('caveat'):warning.append(str(data['caveat']))
    folds=data.get('folds',data.get('fold_best_epochs',args.get('fold',None)))
    count=data.get('rows',data.get('studies',data.get('weak_oof_rows',data.get('val'))))
    contract={'name':data.get('contract','unknown'),'label_source':args.get('weak_labels','unknown'),
              'split_hash':data.get('split_hash'),'mask_hash':data.get('mask_hash'),
              'study_set_hash':data.get('study_set_hash'),'label_hash':data.get('label_hash'),
              'transform_hash':data.get('transform_hash'),'metric':metric_name}
    uri=store.ws.uri(path)
    rid=str(uuid.uuid5(uuid.NAMESPACE_URL,store.instance+':'+uri))
    return {'id':rid,'name':name,'family':family,'kind':kind,'score':score,'targets':targets,
            'rows_count':count if isinstance(count,int) else None,'folds':folds,'contract':contract,
            'warnings':warning,'uri':uri,'payload':data,'observed_at':file_time(path),'origin':store.instance}


def import_local(store: Store):
    result={'runs':0,'sources':0,'errors':[]}
    import_evidence(store)
    paths=sorted(store.ws.resolve('experiments').glob('*/summary.json'))
    paths+=sorted(store.ws.resolve('experiments').glob('*/result.json'))
    seen=set()
    for path in paths:
        if path.parent in seen:continue
        seen.add(path.parent)
        try:
            row=run_record(store,path)
            keys=list(row); values=[dumps(row[k]) if k in ['targets','folds','contract','warnings','payload'] else row[k] for k in keys]
            with store.connect() as con:
                con.execute(f"INSERT INTO runs({','.join(keys)}) VALUES ({','.join('?' for _ in keys)}) ON CONFLICT(id) DO UPDATE SET "+','.join(f'{k}=excluded.{k}' for k in keys if k!='id'),values)
            result['runs']+=1
        except Exception as e:result['errors'].append({'path':store.ws.uri(path),'error':str(e)[:300]})
    for path in [*sorted(store.ws.resolve('reports').glob('*.md')),*sorted((store.ws.root/'docs').rglob('*.md'))]:
        if path.stat().st_size>500_000:continue
        content=path.read_text(encoding='utf-8-sig',errors='replace')
        title=next((line.lstrip('# ').strip() for line in content.splitlines() if line.startswith('# ')),path.stem)
        uri=store.ws.uri(path)
        store.upsert_source(id='local:'+uri,kind='report',title=title,content=content,authority='local',
            tags=tags(content),observed_at=file_time(path),local_uri=uri)
        result['sources']+=1
    for folder in sorted(store.ws.resolve('kaggle_notebooks').glob('*')):
        meta=folder/'kernel-metadata.json'
        if not meta.is_file():continue
        try:
            metadata=read_json(meta);ref=metadata.get('id',folder.name)
            source=folder/metadata.get('code_file','')
            if not source.is_file() or source.stat().st_size>5_000_000:continue
            if source.suffix=='.ipynb':
                nb=read_json(source)
                content='\n\n'.join(f"[cell {i} · {cell.get('cell_type')}]\n"+(''.join(cell.get('source',[])) if isinstance(cell.get('source'),list) else str(cell.get('source',''))) for i,cell in enumerate(nb.get('cells',[])))
            else:content=source.read_text(encoding='utf-8-sig',errors='replace')
            store.upsert_source(id='notebook:'+ref,kind='own_notebook' if folder.name.startswith('own_') or metadata.get('is_private') else 'notebook',
                title=metadata.get('title',ref),content=content,url='https://www.kaggle.com/code/'+ref,
                tags=tags(content[:12000]),observed_at=file_time(source),local_uri=store.ws.uri(source),
                version=str(metadata.get('version','unknown')),payload={'inputs':metadata.get('dataset_sources',[]),'ref':ref,'score_status':'not_verified'})
            result['sources']+=1
        except Exception as e:result['errors'].append({'path':folder.name,'error':str(e)[:300]})
    store.connector('local_import','ok' if not result['errors'] else 'partial',f"실험 {result['runs']}개 · 문서/노트북 {result['sources']}개",dumps(result['errors'][:10]))
    return result


def compare_runs(store, ids):
    by_id={row['id']:row for row in store.runs()}
    records=[by_id[i] for i in ids if i in by_id]
    if len(records)!=len(ids):raise KeyError('Unknown run')
    reasons=[]
    if len(records)!=2:reasons.append('두 개의 실험을 선택하세요.')
    else:
        a,b=records
        for key in ['kind','targets','rows_count','folds']:
            if a[key]!=b[key]:reasons.append(f'{key} 조건이 다릅니다.')
        if a['score'] is None or b['score'] is None:reasons.append('비교할 점수가 없습니다.')
        for key in ['split_hash','mask_hash','study_set_hash','label_hash','transform_hash']:
            if not a['contract'].get(key) or not b['contract'].get(key):reasons.append(f'{key}가 기록되지 않아 비교 조건을 확인할 수 없습니다.')
            elif a['contract'][key]!=b['contract'][key]:reasons.append(f'{key}가 다릅니다.')
        if a['contract'].get('label_source')!=b['contract'].get('label_source'):reasons.append('라벨 출처가 다릅니다.')
        if a['contract'].get('label_source')=='unknown':reasons.append('라벨 버전이 확인되지 않았습니다.')
        if a['contract'].get('metric')!=b['contract'].get('metric'):reasons.append('측정 지표가 다릅니다.')
    delta=records[1]['score']-records[0]['score'] if not reasons and all(r['score'] is not None for r in records) else None
    return {'records':records,'comparable':not reasons,'delta':delta,'reasons':reasons}
