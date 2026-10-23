from __future__ import annotations

import re
import time

from .collectors import (collect_notebook, collect_topics, kaggle, kaggle_page, parse_json_output, records,
                         register_topics, topic_body_state, topic_listing, topic_needs_body)
from .importer import tags
from .store import dumps, now

SOLUTION = re.compile(r'solution|write.?up|\d+\s*(?:st|nd|rd|th)\s*place|\bplace\b|winning|(?:gold|silver|bronze) medal', re.I)
QUESTION = re.compile(r'\?\s*$|^(?:how|what|why|which|any|is|are|does|do|can|should|anyone)\b', re.I)
DEFAULT_SOLUTION_BUDGET = 40
DEFAULT_DISCUSSION_BUDGET = 10
DEFAULT_NOTEBOOK_BODIES = 5


def notebook_sort(evaluation):
    text=evaluation.lower()
    if re.search(r'log\s*loss|logarithmic\s*loss|root mean squared|mean absolute error',text):return 'scoreAscending'
    if re.search(r'area under|roc.?auc|\bauc\b',text):return 'scoreDescending'
    return 'voteCount'  # Unknown metric direction: do not imply a performance ranking.


def is_solution(title):
    """Writeup-like titles. Questions that merely mention 'solution' are excluded."""
    return bool(SOLUTION.search(title)) and not QUESTION.search(title.strip())


def claimed_place(title):
    match=re.search(r'(\d{1,4})\s*(?:st|nd|rd|th)\b',title,re.I) or re.search(r'(?<![\d.])(\d{1,4})\s+place\b',title,re.I)
    return int(match[1]) if match else None


def solution_order(topic):
    place=claimed_place(topic['title'])
    return (place is None,place or 0,-topic['votes'])


def collect_history(store, comp, progress=lambda x: None, solution_budget=DEFAULT_SOLUTION_BUDGET,
                    discussion_budget=DEFAULT_DISCUSSION_BUDGET, notebook_bodies=DEFAULT_NOTEBOOK_BODIES):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,120}',comp):raise ValueError('Invalid competition slug')
    result={'competition':comp,'pages':0,'notebooks':0,'notebook_bodies':0,'discussions':0,'solutions':0,
            'solution_bodies':0,'discussion_bodies':0,'errors':[]}
    def stage(name,call):
        progress(comp+' · '+name)
        try:call()
        except Exception as exc:result['errors'].append({'stage':name,'error':str(exc)[:700]})
        time.sleep(.2)
    def pages():
        data=records(parse_json_output(kaggle(['competitions','pages',comp,'list',comp,'--content','--format','json'])),'pages')
        for row in data:
            store.upsert_source(id=f"competition:{comp}:{row['name']}",kind='official',title=f"{comp} · {row['name']}",content=row.get('content',''),authority='official',
                url=f'https://www.kaggle.com/competitions/{comp}/overview',competition=comp,tags=['공식'])
        result['pages']=len(data)
    def leaders():
        data=records(parse_json_output(kaggle(['competitions','leaderboard',comp,'--show','--page-size','100','--format','json'])),'submissions')
        body=('Kaggle CLI 리더보드 조회 결과. 반환된 순서이며 Public/Private 구분과 최종 입상 순위는 원문에서 확인하세요. '
              '점수 1.0 같은 비정상 행은 누출·테스트 제출일 수 있습니다.\n\n')
        body+='\n'.join(f"{i}. {r.get('teamName',r.get('team_name',''))} | {r.get('score','')}" for i,r in enumerate(data,1))
        store.upsert_source(id='leaderboard:'+comp,kind='leaderboard',title=comp+' · 리더보드 관측',content=body,url=f'https://www.kaggle.com/competitions/{comp}/leaderboard',authority='official',competition=comp,payload={'records':data,'coverage':'first 100; final private rank not asserted'})
    def notebooks():
        evaluation=store.rows("SELECT r.content FROM sources s JOIN revisions r ON r.id=s.current_revision WHERE s.competition=? AND lower(s.title) LIKE '%evaluation%' AND s.kind='official'",(comp,))
        sort=notebook_sort('\n'.join(r['content'] for r in evaluation))
        result['notebook_sort']=sort
        data=kaggle_page(['kernels','list','--competition',comp,'--sort-by',sort,'--page-size','50'],'kernels',1)
        for row in data:
            sid='notebook:'+row['ref']
            if not store.rows('SELECT id FROM sources WHERE id=?',(sid,)):
                store.upsert_source(id=sid,kind='notebook',title=row.get('title',row['ref']),content=f'{sort} 목록에서 발견한 공개 노트북입니다. 실제 숫자 점수·최종 입상·재현 성능은 별도 확인이 필요합니다.',
                    url='https://www.kaggle.com/code/'+row['ref'],competition=comp,tags=tags(row.get('title','')),payload={**row,'discovery_sort':sort,'rank_status':'sorted discovery only'})
            store.set_source_meta(sid,votes=row.get('totalVotes'),last_run=row.get('lastRunTime') or None)
        result['notebooks']=len(data)
        downloaded={r['id'] for r in store.rows("SELECT s.id FROM sources s JOIN revisions r ON r.id=s.current_revision WHERE s.competition=? AND json_extract(r.payload,'$.source_downloaded')=1",(comp,))}
        for row in [r for r in data if 'notebook:'+r['ref'] not in downloaded][:notebook_bodies]:
            try:collect_notebook(store,row['ref'],comp);result['notebook_bodies']+=1
            except Exception as exc:result['errors'].append({'stage':'notebook body','ref':row['ref'],'error':str(exc)[:300]})
    def discussions():
        listing,complete=topic_listing(comp,'top')
        result['discussions']=len(listing);result['discussions_complete']=complete
        register_topics(store,comp,listing,lambda t:'solution' if is_solution(t['title']) else 'discussion',
                        '토론 목록에서 발견했습니다. 본문은 원글과 댓글 전체를 수집합니다. 제목의 순위는 작성자 주장입니다.')
        store.set_meta('topic_catalog:'+comp,dumps({'observed_at':now(),'complete':complete,'items':listing}))
        state=topic_body_state(store,comp)
        solutions=sorted([t for t in listing if is_solution(t['title'])],key=solution_order)
        result['solutions']=len(solutions)
        others=sorted([t for t in listing if not is_solution(t['title'])],key=lambda t:-(t['votes']))
        queue=[(t['id'],comp,'solution') for t in solutions[:solution_budget] if topic_needs_body(state,t)]
        queue+=[(t['id'],comp,'discussion') for t in others[:discussion_budget] if topic_needs_body(state,t)]
        done=collect_topics(store,queue,progress)
        result['errors'].extend({'stage':'topic body',**e} for e in done['errors'])
        state=topic_body_state(store,comp)
        result['solution_bodies']=sum(1 for t in solutions if (state.get(t['id']) or {}).get('src')=='kaggle-sdk')
        result['discussion_bodies']=sum(1 for t in others if (state.get(t['id']) or {}).get('src')=='kaggle-sdk')
        result['fetched_now']=done['fetched']
    stage('공식 안내',pages);stage('리더보드',leaders);stage('공개 노트북',notebooks);stage('솔루션 토론',discussions)
    coverage=(f'all topic pages; solutions by claimed place up to {solution_budget} and top {discussion_budget} other threads, '
              f'opening post + full comments; {result["notebooks"]} notebooks listed with up to {notebook_bodies} new sources; LB first 100')
    store.connector('history:'+comp,'partial' if result['errors'] else 'ok',dumps(result),coverage)
    store.set_meta('history:'+comp,dumps({**result,'observed_at':now()}))
    return result
