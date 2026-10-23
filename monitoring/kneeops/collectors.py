from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .importer import tags
from .research import strategy_excerpts
from .store import Store, digest, dumps, now
from .textutil import html_text

ROOT = Path(__file__).resolve().parents[2]
SECRET = re.compile(r'(KGAT_|kagat_|kagrt_)\S+')
# Recruiting/social threads are fetched last, never skipped: a technical thread may also ask for teammates.
TOPIC_LATE = re.compile(r'teammate|team ?up|team merger|merging|looking for (?:help|a team|partner)|meme thread', re.I)
# A title such as "0.946", "0-943", "public0946" or "speedy947" is the author's claim, never a verified score.
SCORE_CLAIM = re.compile(r'(?<![\d.])0?[.,_-]?(9[4-9]\d)(?!\d)')
DEFAULT_TOPIC_BUDGET = 25


def title_claim(item):
    """Highest score written in a notebook title or slug; author names are excluded. 0 = no claim."""
    claims=[float('0.'+m) for m in SCORE_CLAIM.findall(item.get('title','')+' '+item.get('ref','').split('/',1)[-1])]
    return max(claims) if claims else 0.0
DEFAULT_NOTEBOOK_BUDGET = 12
DEFAULT_VERSION_BUDGET = 8


def parse_json_output(output):
    decoder=json.JSONDecoder()
    for match in re.finditer(r'[\[{]',output):
        try:return decoder.raw_decode(output[match.start():])[0]
        except json.JSONDecodeError:continue
    raise ValueError('Kaggle returned no JSON; existing records were retained')


def _env():
    return {**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}


def kaggle(args, timeout=90):
    local_executable=Path(sys.executable).parent/('kaggle.exe' if os.name=='nt' else 'kaggle')
    executable=str(local_executable) if local_executable.is_file() else shutil.which('kaggle')
    if not executable:raise RuntimeError('Kaggle CLI가 없습니다. 설치 안내를 확인하세요.')
    for attempt in range(3):
        try:
            p=subprocess.run([executable,*args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,env=_env(),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except subprocess.TimeoutExpired:
            if attempt==2:raise RuntimeError('Kaggle 응답 시간이 초과됐습니다. 기존 자료는 보존했습니다.')
            time.sleep(2**attempt);continue
        transient=any(x in (p.stderr+p.stdout) for x in ['INVALID_SESSION_ID','429','502','503','504','ConnectionResetError'])
        if not p.returncode or not transient or attempt==2:break
        time.sleep(2**attempt)
    if p.returncode:
        text=SECRET.sub('[REDACTED]',(p.stderr or p.stdout)[-1000:])
        raise RuntimeError(text or f'Kaggle exit {p.returncode}')
    return p.stdout


def records(value, key):
    if isinstance(value,list):return value
    return value.get(key,[]) if isinstance(value,dict) else []


def kaggle_page(args, key, page):
    """One listing page. Kaggle prints plain text instead of JSON past the last page."""
    output=kaggle([*args,'--page',str(page),'--format','json'])
    try:return records(parse_json_output(output),key)
    except ValueError:
        if page>1 or re.search(r'No \w+ found|Not found',output,re.I):return []
        raise


def kaggle_username(store=None):
    cached=store.get_meta('kaggle_username') if store else None
    if cached:return cached
    match=re.search(r'^- username:\s*(\S+)',kaggle(['config','view']),re.M)
    name=match[1] if match and match[1]!='None' else ''
    if store and name:store.set_meta('kaggle_username',name)
    return name


def _stamp(value):
    """Comparable UTC text. Kaggle list times are naive UTC; ours carry +00:00."""
    return str(value or '').replace('Z','')[:19]


# ---------------------------------------------------------------- discussions

def fetch_topics(ids, chunk=20):
    ids=[int(i) for i in ids]
    for start in range(0,len(ids),chunk):
        part=ids[start:start+chunk]
        process=subprocess.run([sys.executable,'-m','monitoring.kneeops.topic_fetch',*map(str,part)],capture_output=True,text=True,
            encoding='utf-8',errors='replace',timeout=90+45*len(part),env=_env(),cwd=ROOT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        results={}
        for line in process.stdout.splitlines():
            line=line.strip()
            if line.startswith('{'):
                try:item=json.loads(line)
                except json.JSONDecodeError:continue
                if 'id' in item:results[int(item['id'])]=item
        if process.returncode and not results:
            raise RuntimeError(SECRET.sub('[REDACTED]',process.stderr[-1000:]) or 'Topic download failed')
        for tid in part:yield results.get(tid,{'id':tid,'error':'응답이 없습니다. 다음 수집에서 다시 시도합니다.'})


def topic_messages(data):
    topic=data.get('topic') or {}
    messages=[]
    if topic.get('content'):
        messages.append({'id':topic.get('id'),'parent_id':None,'role':'original','depth':0,'author':topic.get('authorName',''),
                         'posted_at':topic.get('postDate',''),'votes':topic.get('votes'),'text':html_text(topic['content'])})
    def walk(items,parent,depth):
        for c in items or []:
            messages.append({'id':c.get('id'),'parent_id':parent,'role':'comment' if depth==1 else 'reply','depth':depth,
                             'author':c.get('authorName',''),'posted_at':c.get('postDate',''),'votes':c.get('votes'),'text':html_text(c.get('content'))})
            walk(c.get('replies'),c.get('id'),depth+1)
    walk(data.get('comments'),topic.get('id'),1)
    return messages


def render_topic(title, topic, messages):
    head=[title,f"작성자 {topic.get('authorName','')} · 게시 {_stamp(topic.get('postDate'))[:10]} · 추천 {topic.get('votes','—')} · 댓글 {topic.get('commentCount','—')}"]
    if not any(m['role']=='original' for m in messages):head.append('(원글 본문이 비어 있거나 삭제되었습니다)')
    parts=['\n'.join(head)]
    for m in messages:
        label={'original':'원글','comment':'댓글','reply':'답글'}[m['role']]
        ref=f"{label} {m['id']}"+(f" → {m['parent_id']}" if m['role']=='reply' else '')
        votes=f" · 추천 {m['votes']}" if m.get('votes') not in (None,'') else ''
        parts.append(f"[{ref}] {m['author']} · {_stamp(m['posted_at'])[:16].replace('T',' ')}{votes}\n{m['text']}")
    return '\n\n'.join(parts)


def save_topic(store, data, competition=None, kind=None):
    tid=int(data['id']);sid=f'topic:{tid}'
    previous=store.rows('SELECT competition,kind,title FROM sources WHERE id=?',(sid,))
    comp=competition or (previous[0]['competition'] if previous else store.ws.config['competition'])
    source_kind=kind or (previous[0]['kind'] if previous else 'discussion')
    topic=data.get('topic') or {}
    title=topic.get('title') or (previous[0]['title'] if previous else f'Discussion {tid}')
    messages=topic_messages(data)
    content=render_topic(title,topic,messages)
    raw=dumps(data);raw_hash=digest(raw)
    folder=store.state/'topics'/str(tid);folder.mkdir(parents=True,exist_ok=True)
    raw_file=folder/(raw_hash[:16]+'.json')
    if not raw_file.exists():raw_file.write_text(raw,encoding='utf-8')
    comments=sum(1 for m in messages if m['role']!='original')
    meta={k:topic.get(k) for k in ('id','title','authorName','postDate','votes','commentCount','forumName','url')}
    store.set_source_meta(sid,votes=topic.get('votes'),comments=topic.get('commentCount'),posted_at=topic.get('postDate') or None)
    return store.upsert_source(id=sid,kind=source_kind,title=title,content=content,competition=comp,
        url=f'https://www.kaggle.com/competitions/{comp}/discussion/{tid}',tags=tags(content[:20000]),
        payload={'messages':messages,'topic':meta,'body_source':'kaggle-sdk','original_post':any(m['role']=='original' for m in messages),
                 'fetched_comments':comments,'complete':bool(data.get('complete')),'comment_pages':data.get('pages'),
                 'raw_file':raw_file.relative_to(store.state).as_posix(),'raw_sha256':raw_hash,
                 'strategy_evidence':strategy_excerpts(content),
                 'rank_status':'participant claim; final placement not independently verified' if source_kind=='solution' else 'not applicable',
                 'coverage':'opening post and full comment tree via Kaggle SDK GetTopic+ListComments; HTML rendered as text'})


def collect_topics(store, items, progress=lambda x:None):
    """items: [(topic_id, competition, kind)]. Failures are reported per topic."""
    result={'fetched':0,'errors':[]}
    meta={int(t):(c,k) for t,c,k in items}
    for i,data in enumerate(fetch_topics(list(meta))):
        tid=int(data['id'])
        if i%5==0:progress(f'토론 본문 {i+1}/{len(meta)}')
        if data.get('error'):
            result['errors'].append({'topic':tid,'error':data['error']});continue
        try:
            save_topic(store,data,*meta[tid]);result['fetched']+=1
        except Exception as exc:result['errors'].append({'topic':tid,'error':str(exc)[:300]})
    return result


def collect_topic(store, id, competition=None, kind=None):
    data=next(fetch_topics([int(id)]))
    if data.get('error'):raise RuntimeError(data['error'])
    return save_topic(store,data,competition,kind)


def topic_listing(comp, sort='recent', max_pages=30):
    seen={};complete=False
    for page in range(1,max_pages+1):
        data=kaggle_page(['competitions','topics','list',comp,'--sort-by',sort],'topics',page)
        added=0
        for row in data:
            tid=row.get('id',row.get('topicId'))
            if tid is None or int(tid) in seen:continue
            seen[int(tid)]={'id':int(tid),'title':row.get('title',str(tid)),'votes':int(row.get('votes') or 0),
                            'commentCount':int(row.get('commentCount') or 0),'postDate':row.get('postDate','')};added+=1
        if not added or len(data)<20:complete=True;break
        time.sleep(.2)
    return list(seen.values()),complete


def topic_body_state(store, comp):
    rows=store.rows("""SELECT s.id,s.kind,json_extract(r.payload,'$.body_source') AS src,
        json_extract(r.payload,'$.topic.commentCount') AS cc FROM sources s JOIN revisions r ON r.id=s.current_revision
        WHERE s.competition=? AND s.id LIKE 'topic:%'""",(comp,))
    return {int(r['id'].split(':')[1]):r for r in rows}


def topic_needs_body(state, item):
    row=state.get(item['id'])
    if not row or row['src']!='kaggle-sdk':return True
    return item['commentCount']>int(row['cc'] or 0)


def topic_queue(store, comp, listing):
    state=topic_body_state(store,comp)
    pending=[t for t in listing if topic_needs_body(state,t)]
    # Never-fetched threads first (most discussed first), then threads with new comments; recruiting titles last.
    def key(t):
        row=state.get(t['id']);fresh=not row or row['src']!='kaggle-sdk'
        return (1 if TOPIC_LATE.search(t['title']) else 0,0 if fresh else 1,-(t['votes']+t['commentCount']) if fresh else -(t['commentCount']-int(row['cc'] or 0)),t['postDate'])
    return sorted(pending,key=key)


def register_topics(store, comp, listing, kind_for=lambda t:'discussion', note='토론 목록에서 발견했습니다. 본문은 원글과 댓글 전체를 수집합니다.'):
    existing={int(r['id'].split(':')[1]):r['kind'] for r in store.rows("SELECT id,kind FROM sources WHERE id LIKE 'topic:%'")}
    for t in listing:
        store.set_source_meta(f"topic:{t['id']}",votes=t.get('votes'),comments=t.get('commentCount'),posted_at=t.get('postDate') or None)
        kind=kind_for(t)
        if t['id'] not in existing:
            store.upsert_source(id=f"topic:{t['id']}",kind=kind,title=t['title'],content=note,competition=comp,
                url=f"https://www.kaggle.com/competitions/{comp}/discussion/{t['id']}",payload={**t,'listing_only':True},tags=tags(t['title']))
        elif kind=='solution' and existing[t['id']]=='discussion':
            with store.connect() as con:con.execute("UPDATE sources SET kind='solution' WHERE id=?",(f"topic:{t['id']}",))


# ------------------------------------------------------------------ notebooks

def notebook_inputs(payload):
    """Attached Kaggle inputs by kind; empty refs (deleted or private sources) are dropped."""
    clean=lambda key:sorted({str(x).strip() for x in (payload.get(key) or []) if str(x).strip()})
    return {'datasets':clean('inputs'),'models':clean('model_inputs'),'kernels':clean('kernel_inputs'),'competitions':clean('competition_inputs')}


def listing_meta(store, item):
    store.set_source_meta('notebook:'+item['ref'],votes=item.get('totalVotes'),last_run=item.get('lastRunTime') or None,
                          score_claim=title_claim(item))

def collect_notebook(store, ref, competition=None, listing=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[1-9][0-9]*)?',ref):raise ValueError('Invalid notebook ref')
    process=subprocess.run([sys.executable,'-m','monitoring.kneeops.notebook_fetch',ref],capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=90,env=_env(),cwd=ROOT,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if process.returncode:
        raise RuntimeError(SECRET.sub('[REDACTED]',process.stderr[-1000:]) or 'Notebook download failed')
    data=parse_json_output(process.stdout)
    source=data.pop('source');base_ref=data['ref']
    if len(source.encode('utf-8'))>10_000_000:raise ValueError('Source exceeds 10 MB')
    if data['kernel_type'].lower()=='notebook':
        nb=json.loads(source)
        body='\n\n'.join(f"[cell {i} · {c.get('cell_type')}]\n"+(''.join(c.get('source',[])) if isinstance(c.get('source'),list) else str(c.get('source','')))
            for i,c in enumerate(nb.get('cells',[])))
    else:body=source
    previous=store.rows('SELECT competition,kind,current_revision FROM sources WHERE id=?',('notebook:'+base_ref,))
    comp=competition or (previous[0]['competition'] if previous else store.ws.config['competition'])
    kind=previous[0]['kind'] if previous else 'notebook'
    # Fetching v1 later must not make it the notebook's newest known source.
    current=store.source('notebook:'+base_ref) if previous else None
    promote=not data['requested_version'] or not current or not current['revision']['payload'].get('source_downloaded')
    destination=store.state/'notebooks'/base_ref
    destination.mkdir(parents=True,exist_ok=True)
    raw_hash=digest(source)
    raw_file=destination/(str(data['version'])+'-'+raw_hash[:16]+'.'+('ipynb' if data['kernel_type'].lower()=='notebook' else 'txt'))
    if not raw_file.exists():raw_file.write_text(source,encoding='utf-8')
    extra={'listing':{k:listing.get(k) for k in ('lastRunTime','totalVotes','title_score_claim')}} if listing else {}
    rev=store.upsert_source(id='notebook:'+base_ref,kind=kind,title=data['title'],content=body,competition=comp,
        url='https://www.kaggle.com/code/'+base_ref,tags=tags(body[:15000]),version=data['version'],promote=promote,
        payload={**data,**extra,'score_status':'not_verified','source_downloaded':True,'raw_sha256':raw_hash,
                 'raw_file':str(raw_file.relative_to(store.state).as_posix()),'strategy_evidence':strategy_excerpts(body)})
    if promote:store.set_source_meta('notebook:'+base_ref,inputs=notebook_inputs(data))
    if listing:listing_meta(store,{**listing,'ref':base_ref})
    return rev


def notebook_catalog(comp, max_pages=20):
    """Every public notebook Kaggle lists for the competition, plus listing ranks.

    Ranks are listing positions only; Kaggle's list API exposes no numeric score.
    """
    items={}
    def take(sort,pages,rank_key=None):
        for page in range(1,pages+1):
            data=kaggle_page(['kernels','list','--competition',comp,'--sort-by',sort,'--page-size','100'],'kernels',page)
            for i,row in enumerate(data):
                ref=row.get('ref')
                if not ref:continue
                item=items.setdefault(ref,{'ref':ref,'title':row.get('title',ref),'author':row.get('author',''),
                    'lastRunTime':row.get('lastRunTime',''),'totalVotes':int(row.get('totalVotes') or 0)})
                if rank_key and rank_key not in item:item[rank_key]=(page-1)*100+i+1
            if len(data)<100:return True
            time.sleep(.2)
        return False
    complete=take('dateRun',max_pages)
    for sort,key in (('voteCount','vote_rank'),('scoreDescending','score_rank'),('hotness','hot_rank')):
        take(sort,1,key)
    for item in items.values():
        item['title_score_claim']=title_claim(item)
    return {'observed_at':now(),'competition':comp,'total':len(items),'complete':complete,
            'items':sorted(items.values(),key=lambda x:x['lastRunTime'],reverse=True)}


def notebook_relevant(item, reference=None):
    reference=reference or datetime.now(timezone.utc)
    recent=_stamp(item.get('lastRunTime'))>=_stamp((reference-timedelta(days=7)).isoformat())
    return (item.get('totalVotes',0)>=5 or item.get('score_rank',999)<=50 or item.get('hot_rank',999)<=50
            or item.get('title_score_claim',0)>=0.942 or (item.get('public_score') or 0)>=0.94 or (recent and item.get('totalVotes',0)>=1))


def notebook_state(store, comp):
    # downloaded: versioned source from the Kaggle API. local: an earlier unversioned workspace snapshot.
    rows=store.rows("""SELECT s.id,s.kind,s.observed_at,json_extract(r.payload,'$.source_downloaded') AS downloaded,
        s.local_uri<>'' AS local,json_extract(r.payload,'$.listing.lastRunTime') AS listed_run FROM sources s JOIN revisions r ON r.id=s.current_revision
        WHERE s.competition=? AND s.kind IN ('notebook','own_notebook')""",(comp,))
    return {r['id'].removeprefix('notebook:'):r for r in rows}


def notebook_queues(store, comp, catalog):
    state=notebook_state(store,comp)
    # Catalog items are newest-run first; the stable sort keeps recency as the final tie-break.
    scores={r['source_id'].removeprefix('notebook:'):r['public_score'] for r in store.rows('SELECT source_id,public_score FROM source_meta WHERE public_score IS NOT NULL')}
    def priority(item):
        return (-(scores.get(item['ref']) or 0),-item.get('title_score_claim',0),item.get('score_rank',999),-item.get('totalVotes',0))
    new=[i for i in catalog.get('items',[]) if i['ref'] in state and not state[i['ref']]['downloaded'] and notebook_relevant(i)]
    changed=[i for i in catalog.get('items',[]) if i['ref'] in state and state[i['ref']]['downloaded']
             and _stamp(i.get('lastRunTime'))>max(_stamp(state[i['ref']]['listed_run']),_stamp(state[i['ref']]['observed_at']))]
    return sorted(new,key=priority),sorted(changed,key=lambda i:(-i.get('totalVotes',0),i['ref']))


def register_notebooks(store, comp, catalog, username=''):
    known={r['id'] for r in store.rows("SELECT id FROM sources WHERE id LIKE 'notebook:%'")}
    added=0
    for item in catalog['items']:
        sid='notebook:'+item['ref']
        if sid in known:listing_meta(store,item)
        if sid in known or not notebook_relevant(item):continue
        own=bool(username) and item['ref'].lower().startswith(username.lower()+'/')
        claim=f" 제목의 점수 표기 {item['title_score_claim']:.3f}는 작성자 주장입니다." if item.get('title_score_claim') else ''
        store.upsert_source(id=sid,kind='own_notebook' if own else 'notebook',title=item['title'],competition=comp,
            content=f"공개 노트북 목록에서 발견했습니다. 추천 {item['totalVotes']} · 마지막 실행 {_stamp(item['lastRunTime'])[:16]} UTC.{claim} 소스는 우선순위에 따라 자동 수집합니다.",
            url='https://www.kaggle.com/code/'+item['ref'],tags=tags(item['title']),payload={**item,'listing_only':True,'score_status':'not_verified'})
        listing_meta(store,item);added+=1
    return added


def collect_notebooks(store, comp, items, progress=lambda x:None, label='노트북 소스'):
    result={'fetched':0,'errors':[]}
    for i,item in enumerate(items):
        progress(f'{label} {i+1}/{len(items)}')
        try:collect_notebook(store,item['ref'],comp,item);result['fetched']+=1
        except Exception as exc:result['errors'].append({'ref':item['ref'],'error':str(exc)[:300]})
        time.sleep(.2)
    return result


# ---------------------------------------------------------------- leaderboard

MEDAL_NOTE='Kaggle 메달 기준(참가 1000팀 이상): 금 상위 10+0.2%, 은 상위 5%, 동 상위 10%. 공개 LB 기준 참고값이며 최종 순위는 Private LB로 결정됩니다.'


def leaderboard_summary(comp, username=''):
    with tempfile.TemporaryDirectory() as tmp:
        kaggle(['competitions','leaderboard',comp,'--download','-p',tmp],timeout=180)
        text=''
        for path in Path(tmp).iterdir():
            if path.suffix=='.zip':
                with zipfile.ZipFile(path) as z:
                    name=next(n for n in z.namelist() if n.endswith('.csv'))
                    text=z.read(name).decode('utf-8-sig')
            elif path.suffix=='.csv':text=path.read_text(encoding='utf-8-sig')
    rows=[r for r in csv.DictReader(io.StringIO(text)) if r.get('Score') not in (None,'')]
    if not rows:raise ValueError('전체 리더보드 파일에 점수 행이 없습니다.')
    scores=sorted((float(r['Score']) for r in rows),reverse=True);teams=len(rows)
    at=lambda rank:scores[min(rank,teams)-1]
    own=None
    if username:
        for r in rows:
            members=[m.strip().lower() for m in (r.get('TeamMemberUserNames') or '').split(',')]
            if username.lower() in members:
                own={'rank':int(r.get('Rank') or 0),'score':float(r['Score']),'team_name':r.get('TeamName',''),
                     'submission_count':int(r.get('SubmissionCount') or 0),'percentile':round(int(r.get('Rank') or 0)/teams*100,2)}
                break
    medals=[{'medal':'gold','rank':10+int(teams*.002)},{'medal':'silver','rank':max(1,int(teams*.05))},{'medal':'bronze','rank':max(1,int(teams*.10))}] if teams>=1000 else []
    for m in medals:m['score']=at(m['rank'])
    thresholds=[{'score':t,'teams':sum(1 for s in scores if s>=t-1e-9)} for t in sorted({round(x,3) for x in (scores[0],at(10),at(50),at(100),0.95,0.946,0.943,0.941,0.94,0.937)},reverse=True)]
    return {'observed_at':now(),'competition':comp,'teams':teams,'own':own,'medals':medals,'medal_note':MEDAL_NOTE if medals else '',
            'thresholds':thresholds,'rank_scores':{str(k):at(k) for k in (1,10,50,100,200,500,1000) if k<=teams},
            'coverage':'public leaderboard download; ties share displayed precision'}


# ---------------------------------------------------------------- sync

def rebuild_source_meta(store):
    """Fill source_meta from stored catalogs and revision payloads (no network)."""
    n=0
    for r in store.rows("SELECT s.id,s.kind,s.title,r.payload FROM sources s JOIN revisions r ON r.id=s.current_revision WHERE s.kind IN ('notebook','own_notebook','discussion','solution')"):
        p=json.loads(r['payload'] or '{}')
        if r['kind'] in ('notebook','own_notebook'):
            lst=p.get('listing') or {}
            store.set_source_meta(r['id'],votes=lst.get('totalVotes',p.get('totalVotes')),last_run=lst.get('lastRunTime') or p.get('lastRunTime') or None,
                                  score_claim=title_claim({'title':r['title'],'ref':r['id'].removeprefix('notebook:')}),
                                  inputs=notebook_inputs(p) if p.get('source_downloaded') or p.get('inputs') else None)
        else:
            t=p.get('topic') or p
            store.set_source_meta(r['id'],votes=t.get('votes'),comments=t.get('commentCount'),posted_at=t.get('postDate') or None)
        n+=1
    for row in store.rows("SELECT key,value FROM meta WHERE key LIKE 'notebook_catalog:%' OR key LIKE 'topic_catalog:%'"):
        data=json.loads(row['value'] or '{}')
        for item in data.get('items',[]):
            if row['key'].startswith('notebook_catalog:'):
                if store.rows('SELECT 1 FROM sources WHERE id=?',('notebook:'+item['ref'],)):listing_meta(store,item);n+=1
            else:
                store.set_source_meta(f"topic:{item['id']}",votes=item.get('votes'),comments=item.get('commentCount'),posted_at=item.get('postDate') or None);n+=1
    return n


def submission_limits(comp):
    data=parse_json_output(kaggle(['competitions','submission-limits',comp,'--json']))
    return {'used_total':data.get('numTotal'),'remaining_today':data.get('numAllowedNow'),'observed_at':now(),
            'note':'Kaggle CLI submission-limits. 일일 한도는 UTC 0시에 초기화됩니다.'}

def collection_coverage(store, comp=None):
    comp=comp or store.ws.config['competition']
    catalog=json.loads(store.get_meta('notebook_catalog:'+comp,'null') or 'null') or {'items':[]}
    topics=json.loads(store.get_meta('topic_catalog:'+comp,'null') or 'null') or {'items':[]}
    nb=notebook_state(store,comp);new,changed=notebook_queues(store,comp,catalog)
    bodies=topic_body_state(store,comp)
    return {'competition':comp,
        'notebooks':{'live':catalog.get('total',0),'listed_at':catalog.get('observed_at'),'complete':catalog.get('complete'),
                     'registered':len(nb),'sources':sum(1 for r in nb.values() if r['downloaded'] or r['local']),
                     'versioned':sum(1 for r in nb.values() if r['downloaded']),'pending_new':len(new),'pending_changed':len(changed),
                     'relevant':sum(1 for i in catalog['items'] if notebook_relevant(i))},
        'discussions':{'live':len(topics['items']),'listed_at':topics.get('observed_at'),'registered':len(bodies),
                       'bodies':sum(1 for r in bodies.values() if r['src']=='kaggle-sdk'),'pending':len(topic_queue(store,comp,topics['items']))},
        'leaderboard':json.loads(store.get_meta('leaderboard_summary:'+comp,'null') or 'null')}


def sync_kaggle(store: Store, progress=lambda message:None, topic_budget=DEFAULT_TOPIC_BUDGET,
                notebook_budget=DEFAULT_NOTEBOOK_BUDGET, version_budget=DEFAULT_VERSION_BUDGET):
    comp=store.ws.config['competition']; result={'ok':[],'errors':[]}
    state={'catalog':None}
    def stage(name, callback, coverage):
        progress(name)
        try:
            detail=callback(); store.connector(name,'ok',detail if isinstance(detail,str) else f'{detail}개 항목 조회',coverage);result['ok'].append(name)
        except Exception as e:
            message=str(e)[:1000];store.connector(name,'error',message,coverage);result['errors'].append({'connector':name,'message':message})
        time.sleep(.2)
    def submissions():
        output=kaggle(['competitions','submissions',comp,'--page-size','200','--format','json'])
        data=records(parse_json_output(output),'submissions')
        for row in data:store.upsert_submission(row)
        try:
            limits=submission_limits(comp);store.set_meta('submission_limits:'+comp,dumps(limits))
        except Exception:
            return len(data)
        return f"{len(data)}개 · 오늘 남은 제출 {limits['remaining_today']}회"
    def leaderboard():
        data=records(parse_json_output(kaggle(['competitions','leaderboard',comp,'--show','--page-size','100','--format','json'])),'submissions')
        store.leaderboard(data);return len(data)
    def leaderboard_full():
        try:username=kaggle_username(store)
        except Exception:username=''
        summary=leaderboard_summary(comp,username)
        store.set_meta('leaderboard_summary:'+comp,dumps(summary))
        own=summary['own']
        return f"{summary['teams']}팀"+(f" · 내 순위 {own['rank']}위 ({own['score']:.3f})" if own else ' · 내 팀을 찾지 못함')
    def pages():
        # Kaggle 2.2.4 repeats the optional positional at both parser levels.
        data=records(parse_json_output(kaggle(['competitions','pages',comp,'list',comp,'--content','--format','json'])),'pages')
        for row in data:
            name=row.get('name','page');body=row.get('content','')
            store.upsert_source(id='competition:'+name,kind='official',title=name,content=body,authority='official',
                url=f'https://www.kaggle.com/competitions/{comp}',tags=['공식'])
        timeline='\n'.join(r.get('content','') for r in data if str(r.get('name','')).lower()=='timeline')
        match=re.search(r'([A-Z][a-z]+ \d{1,2}, \d{4})\*?\*?\s*-\s*Final Submission Deadline',timeline)
        if match:
            info=json.loads(store.get_meta('competition','{}'))
            info['deadline']=datetime.strptime(match[1],'%B %d, %Y').replace(hour=23,minute=59,tzinfo=timezone.utc).isoformat()
            info['metadata_observed_at']=now()
            store.set_meta('competition',dumps(info))
        return len(data)
    def notebooks():
        catalog=notebook_catalog(comp);state['catalog']=catalog
        store.set_meta('notebook_catalog:'+comp,dumps(catalog))
        try:username=kaggle_username(store)
        except Exception:username=''
        added=register_notebooks(store,comp,catalog,username)
        return f"공개 노트북 {catalog['total']}개 확인 · 관련 {sum(1 for i in catalog['items'] if notebook_relevant(i))}개 · 새 등록 {added}개"
    def notebook_scores():
        # Unofficial website API: the only source of a notebook's real public score. Optional stage.
        from .kaggle_web import competition_notebooks
        from .kaggle_web import call as web_call
        info=json.loads(store.get_meta('competition','{}'))
        if not info.get('id'):
            found=web_call('competitions.CompetitionService/GetCompetition',{'competitionName':comp}).get('id')
            if not found:raise RuntimeError('competition id unknown')
            info['id']=found;store.set_meta('competition',dumps(info))
        scores=competition_notebooks(info['id'])
        stamp=now();known={r['id'] for r in store.rows("SELECT id FROM sources WHERE id LIKE 'notebook:%'")};updated=0
        for ref,row in scores.items():
            if 'notebook:'+ref in known:
                store.set_source_meta('notebook:'+ref,public_score=row['public_score'],medal=row['medal'],runtime_s=row['runtime_s'],
                                      score_observed_at=stamp if row['public_score'] is not None else None);updated+=1
        store.set_meta('notebook_scores:'+comp,dumps({'observed_at':stamp,'total':len(scores),'scored':sum(1 for r in scores.values() if r['public_score'] is not None),
                                                      'source':'kaggle website API (unofficial)'}))
        return f"웹 점수 {sum(1 for r in scores.values() if r['public_score'] is not None)}/{len(scores)}개 · 등록 노트북 {updated}개 갱신"
    def notebook_sources():
        catalog=state['catalog'] or json.loads(store.get_meta('notebook_catalog:'+comp,'null') or '{"items":[]}')
        new,_=notebook_queues(store,comp,catalog)
        done=collect_notebooks(store,comp,new[:notebook_budget],progress)
        if done['errors'] and not done['fetched']:raise RuntimeError(dumps(done['errors'][:5]))
        return f"새 소스 {done['fetched']}개 수집 · 대기 {max(0,len(new)-done['fetched'])}개"+(f" · 실패 {len(done['errors'])}" if done['errors'] else '')
    def notebook_versions():
        catalog=state['catalog'] or json.loads(store.get_meta('notebook_catalog:'+comp,'null') or '{"items":[]}')
        _,changed=notebook_queues(store,comp,catalog)
        done=collect_notebooks(store,comp,changed[:version_budget],progress,'노트북 버전')
        if done['errors'] and not done['fetched']:raise RuntimeError(dumps(done['errors'][:5]))
        return f"실행 기록이 바뀐 노트북 {done['fetched']}개 갱신 · 대기 {max(0,len(changed)-done['fetched'])}개"
    def topics():
        listing,complete=topic_listing(comp)
        store.set_meta('topic_catalog:'+comp,dumps({'observed_at':now(),'complete':complete,'items':listing}))
        register_topics(store,comp,listing)
        queue=topic_queue(store,comp,listing)
        done=collect_topics(store,[(t['id'],comp,None) for t in queue[:topic_budget]],progress)
        if done['errors'] and not done['fetched']:raise RuntimeError(dumps(done['errors'][:5]))
        return f"토론 {len(listing)}개 · 본문 {done['fetched']}개 수집 · 대기 {max(0,len(queue)-done['fetched'])}개"+(f" · 실패 {len(done['errors'])}" if done['errors'] else '')
    stage('submissions',submissions,'latest page, max 200; further pages not claimed')
    stage('leaderboard',leaderboard,'public top 100 rows')
    stage('leaderboard_full',leaderboard_full,'full public leaderboard download; own rank, thresholds, medal zone reference')
    stage('official_pages',pages,'all pages returned by CLI')
    stage('notebook_discovery',notebooks,'all public notebooks by dateRun (up to 2000) + vote/score/hotness ranks; listing has no numeric scores')
    stage('notebook_scores',notebook_scores,'website API (unofficial): best public score, medal, last run time; failure leaves earlier values')
    stage('notebook_sources',notebook_sources,f'new relevant notebooks, up to {notebook_budget} sources per run; priority: public score, title claim, score rank, votes')
    stage('notebook_versions',notebook_versions,f'downloaded notebooks whose lastRunTime changed, up to {version_budget} per run')
    stage('discussions',topics,f'all topic list pages; up to {topic_budget} new or newly commented topics per run with opening post and full comment tree')
    store.set_meta('last_sync_attempt',now())
    return result
