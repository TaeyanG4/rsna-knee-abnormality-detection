from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from knee_workspace import Workspace


TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def now():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


SCHEMA = '''
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL DEFAULT '',
 authority TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', current_revision TEXT,
 observed_at TEXT NOT NULL, origin TEXT NOT NULL, local_uri TEXT NOT NULL DEFAULT '', competition TEXT NOT NULL DEFAULT '', added_at TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS revisions(
 id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), content TEXT NOT NULL,
 sha TEXT NOT NULL, version TEXT NOT NULL DEFAULT '', observed_at TEXT NOT NULL,
 payload TEXT NOT NULL DEFAULT '{}');
CREATE INDEX IF NOT EXISTS revisions_source ON revisions(source_id, observed_at);
CREATE VIRTUAL TABLE IF NOT EXISTS source_search USING fts5(source_id UNINDEXED, title, content, tags);
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, name TEXT NOT NULL, family TEXT NOT NULL,
 kind TEXT NOT NULL, score REAL, targets TEXT NOT NULL, rows_count INTEGER, folds TEXT NOT NULL,
 contract TEXT NOT NULL, warnings TEXT NOT NULL, uri TEXT NOT NULL, payload TEXT NOT NULL,
 observed_at TEXT NOT NULL, origin TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS submissions(id TEXT PRIMARY KEY, description TEXT NOT NULL,
 score REAL, status TEXT NOT NULL, submitted_at TEXT NOT NULL, observed_at TEXT NOT NULL,
 payload TEXT NOT NULL, origin TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS leaderboard(id INTEGER PRIMARY KEY AUTOINCREMENT, observed_at TEXT NOT NULL,
 rank INTEGER NOT NULL, team_id TEXT NOT NULL, team_name TEXT NOT NULL, score REAL NOT NULL,
 UNIQUE(observed_at, team_id));
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
 started_at TEXT NOT NULL, finished_at TEXT, progress TEXT NOT NULL, result TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS hypotheses(id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL,
 status TEXT NOT NULL, source_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 version INTEGER NOT NULL DEFAULT 1, origin TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS connectors(name TEXT PRIMARY KEY, status TEXT NOT NULL,
 last_attempt TEXT NOT NULL, last_success TEXT, detail TEXT NOT NULL, coverage TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS imports(id TEXT PRIMARY KEY, imported_at TEXT NOT NULL, result TEXT NOT NULL);
-- Latest observed listing facts (votes, activity, inputs). Mutable; revisions stay immutable.
CREATE TABLE IF NOT EXISTS source_meta(source_id TEXT PRIMARY KEY, votes INTEGER, comments INTEGER, posted_at TEXT,
 last_run TEXT, score_claim REAL, inputs TEXT, observed_at TEXT NOT NULL);
'''


class Store:
    def __init__(self, root: str | Path | None = None, state: str | Path | None = None):
        self.ws = Workspace.load(root)
        self.state = Path(state or os.environ.get('KNEEOPS_STATE_DIR') or
                          self.ws.root / self.ws.config.get('dashboard', {}).get('state_dir', '.local/kneeops')).resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        self.db = self.state / 'app.sqlite'
        self.lock = threading.RLock()
        with self.connect() as con:
            con.executescript(SCHEMA)
            if 'competition' not in {r['name'] for r in con.execute('PRAGMA table_info(sources)')}:
                con.execute("ALTER TABLE sources ADD COLUMN competition TEXT NOT NULL DEFAULT ''")
            if 'added_at' not in {r['name'] for r in con.execute('PRAGMA table_info(sources)')}:
                con.execute("ALTER TABLE sources ADD COLUMN added_at TEXT NOT NULL DEFAULT ''")
            con.execute('UPDATE sources SET added_at=? WHERE added_at=?', (now(), ''))
            con.execute('UPDATE sources SET competition=? WHERE competition=?', (self.ws.config['competition'], ''))
            for column, kind in (('public_score', 'REAL'), ('medal', 'TEXT'), ('runtime_s', 'INTEGER'), ('score_observed_at', 'TEXT')):
                if column not in {r['name'] for r in con.execute('PRAGMA table_info(source_meta)')}:
                    con.execute(f'ALTER TABLE source_meta ADD COLUMN {column} {kind}')
            con.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('schema_version', '3'))
            con.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('instance_id', str(uuid.uuid4())))
            con.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('preferences', dumps({
                'theme': 'light', 'compact': False, 'watch_targets': ['Synovitis', 'Lateral OA'],
                'hidden_cards': [], 'sync_enabled': False, 'sync_minutes': 60})))
        self.instance = self.get_meta('instance_id')
        if not self.get_meta('competition'):
            info=self.ws.config.get('competition_info',{})
            self.set_meta('competition',dumps({'title':self.ws.config['competition'],**info}))

    @contextmanager
    def connect(self):
        with self.lock:
            con = sqlite3.connect(self.db, timeout=20)
            con.row_factory = sqlite3.Row
            con.execute('PRAGMA journal_mode=WAL')
            con.execute('PRAGMA foreign_keys=ON')
            try:
                yield con
                con.commit()
            except Exception:
                con.rollback()
                raise
            finally:
                con.close()

    def rows(self, query, args=()):
        with self.connect() as con:
            return [dict(r) for r in con.execute(query, args).fetchall()]

    def get_meta(self, key, default=None):
        rows = self.rows('SELECT value FROM meta WHERE key=?', (key,))
        return rows[0]['value'] if rows else default

    def set_meta(self, key, value):
        with self.connect() as con:
            con.execute('INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, value))

    def upsert_source(self, *, id, kind, title, content='', url='', authority='participant', tags=None,
                      version='', payload=None, observed_at=None, local_uri='', origin=None, competition=None, promote=True):
        stamp = observed_at or now()
        sha = digest(content)
        rev = digest(id + '\n' + str(version) + '\n' + sha)
        with self.connect() as con:
            old = con.execute('SELECT current_revision, observed_at FROM sources WHERE id=?', (id,)).fetchone()
            con.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                        (id, kind, title, url, authority, dumps(tags or []), rev, stamp, origin or self.instance, local_uri, competition or self.ws.config['competition'], now()))
            con.execute('INSERT OR IGNORE INTO revisions VALUES (?,?,?,?,?,?,?)',
                        (rev, id, content, sha, str(version), stamp, dumps(payload or {})))
            # Historical imports must not move a live source backwards in time.
            if not old or (promote and stamp >= old['observed_at']):
                con.execute('UPDATE sources SET kind=?,title=?,url=?,authority=?,tags=?,current_revision=?,observed_at=?,local_uri=?,competition=? WHERE id=?',
                            (kind,title,url,authority,dumps(tags or []),rev,stamp,local_uri,competition or self.ws.config['competition'],id))
                con.execute('DELETE FROM source_search WHERE source_id=?', (id,))
                con.execute('INSERT INTO source_search VALUES (?,?,?,?)', (id,title,content,dumps(tags or [])))
        return rev

    def set_source_meta(self, source_id, *, votes=None, comments=None, posted_at=None, last_run=None, score_claim=None, inputs=None, observed_at=None,
                        public_score=None, medal=None, runtime_s=None, score_observed_at=None):
        # Missing values keep the earlier observation; a newer listing never erases known inputs.
        with self.connect() as con:
            con.execute('''INSERT INTO source_meta(source_id,votes,comments,posted_at,last_run,score_claim,inputs,observed_at,public_score,medal,runtime_s,score_observed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET
                votes=COALESCE(excluded.votes,votes),comments=COALESCE(excluded.comments,comments),
                posted_at=COALESCE(excluded.posted_at,posted_at),last_run=COALESCE(excluded.last_run,last_run),
                score_claim=COALESCE(excluded.score_claim,score_claim),inputs=COALESCE(excluded.inputs,inputs),
                observed_at=excluded.observed_at,public_score=COALESCE(excluded.public_score,public_score),medal=COALESCE(excluded.medal,medal),
                runtime_s=COALESCE(excluded.runtime_s,runtime_s),score_observed_at=COALESCE(excluded.score_observed_at,score_observed_at)''',
                (source_id, votes, comments, posted_at, last_run, score_claim, dumps(inputs) if inputs is not None else None, observed_at or now(),
                 public_score, medal, runtime_s, score_observed_at))

    BODY_STATE = '''CASE WHEN json_extract(r.payload,'$.source_downloaded') OR json_extract(r.payload,'$.body_source')='kaggle-sdk'
        OR (s.local_uri<>'' AND length(r.content)>500) OR s.kind IN ('official','report','leaderboard') THEN 'full'
        WHEN json_extract(r.payload,'$.listing_only') OR length(r.content)<300 THEN 'listing' ELSE 'partial' END'''
    SORTS = {'added': 's.added_at DESC,s.observed_at DESC,s.title',
             'activity': "COALESCE(m.last_run,m.posted_at,s.observed_at) DESC,s.title",
             'votes': 'COALESCE(m.votes,-1) DESC,s.added_at DESC',
             'comments': 'COALESCE(m.comments,-1) DESC,s.added_at DESC',
             'claim': 'COALESCE(m.score_claim,0) DESC,COALESCE(m.votes,-1) DESC',
             'score': 'COALESCE(m.public_score,-1) DESC,COALESCE(m.votes,-1) DESC',
             'title': 's.title COLLATE NOCASE'}

    def sources(self, q='', kind='', limit=100, offset=0, competition='', sort='added', body='', input_ref=''):
        where, args = [], []
        if kind:
            where.append('s.kind=?'); args.append(kind)
        if competition:
            where.append('s.competition=?'); args.append(competition)
        if q.strip():
            # Literal phrase, not user-authored FTS syntax. LIKE also supports Korean substrings.
            phrase = '"' + q.replace('"', '""') + '"'
            where.append('(s.id IN (SELECT source_id FROM source_search WHERE source_search MATCH ?) OR s.title LIKE ? OR r.content LIKE ?)')
            args.extend([phrase, '%' + q + '%', '%' + q + '%'])
        if body in {'full', 'listing'}:
            where.append(f'({self.BODY_STATE})=?'); args.append(body)
        if input_ref:
            where.append("EXISTS (SELECT 1 FROM json_each(m.inputs) g, json_each(g.value) i WHERE i.value=?)"); args.append(input_ref)
        sql = f'''SELECT s.*,r.version,r.sha,length(r.content) AS content_length,
                 (SELECT count(*) FROM revisions v WHERE v.source_id=s.id) AS revision_count,
                 substr(r.content,1,220) AS excerpt,m.votes,m.comments,m.posted_at,m.last_run,m.score_claim,m.public_score,m.medal,m.runtime_s,
                 ({self.BODY_STATE}) AS body_state FROM sources s
                 LEFT JOIN revisions r ON r.id=s.current_revision LEFT JOIN source_meta m ON m.source_id=s.id'''
        if where: sql += ' WHERE ' + ' AND '.join(where)
        count = self.rows('SELECT count(*) AS n FROM ('+sql+')',args)[0]['n']
        rows = self.rows(sql+f' ORDER BY {self.SORTS.get(sort, self.SORTS["added"])} LIMIT ? OFFSET ?', [*args,limit,offset])
        for row in rows: row['tags'] = json.loads(row['tags'])
        return {'items': rows, 'total': count, 'offset': offset, 'limit': limit}

    def source_meta(self, source_id):
        rows = self.rows('SELECT * FROM source_meta WHERE source_id=?', (source_id,))
        if not rows: return None
        row = rows[0]; row['inputs'] = json.loads(row['inputs']) if row['inputs'] else None
        return row

    def source(self, id, revision=None):
        rows = self.rows('SELECT * FROM sources WHERE id=?', (id,))
        if not rows: raise KeyError(id)
        source = rows[0]
        rev = self.rows('SELECT * FROM revisions WHERE id=? AND source_id=?', (revision or source['current_revision'], id))
        if not rev: raise KeyError(revision)
        source.update({'tags': json.loads(source['tags']), 'revision': dict(rev[0]),
                       'revisions': self.rows('SELECT id,version,sha,observed_at,length(content) AS size FROM revisions WHERE source_id=? ORDER BY observed_at DESC', (id,))})
        source['revision']['payload'] = json.loads(source['revision']['payload'])
        source['meta'] = self.source_meta(id)
        return source

    def runs(self):
        rows = self.rows('SELECT * FROM runs ORDER BY name DESC')
        for row in rows:
            for key in ['targets','folds','contract','warnings','payload']: row[key] = json.loads(row[key])
        return rows

    def connector(self, name, status, detail, coverage=''):
        stamp = now()
        with self.connect() as con:
            con.execute('''INSERT INTO connectors VALUES (?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET
                status=excluded.status,last_attempt=excluded.last_attempt,
                last_success=CASE WHEN excluded.status='ok' THEN excluded.last_success ELSE connectors.last_success END,
                detail=excluded.detail,coverage=excluded.coverage''',
                (name,status,stamp,stamp if status=='ok' else None,detail,coverage))

    def upsert_submission(self, row, observed_at=None):
        stamp=observed_at or now()
        sid=str(row.get('ref',row.get('id','')))
        if not sid: return
        raw=row.get('publicScore',row.get('public_score'))
        score=float(raw) if raw not in (None,'') else None
        with self.connect() as con:
            con.execute('''INSERT INTO submissions VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                description=excluded.description,score=excluded.score,status=excluded.status,
                observed_at=excluded.observed_at,payload=excluded.payload
                WHERE excluded.observed_at>=submissions.observed_at''',
                (sid,row.get('description',''),score,str(row.get('status','unknown')),row.get('date',''),stamp,dumps(row),self.instance))

    def leaderboard(self, records, observed_at=None):
        stamp=observed_at or now()
        with self.connect() as con:
            for i,row in enumerate(records,1):
                if row.get('score') in (None,''):continue
                con.execute('INSERT OR IGNORE INTO leaderboard(observed_at,rank,team_id,team_name,score) VALUES (?,?,?,?,?)',
                    (stamp,i,str(row.get('teamId',row.get('team_id',i))),row.get('teamName',row.get('team_name','')),float(row['score'])))

    def briefing(self):
        counts={name:self.rows(f'SELECT count(*) AS n FROM {name}')[0]['n'] for name in ['sources','runs','submissions']}
        submissions=self.rows('SELECT id,description,score,status,submitted_at,observed_at FROM submissions ORDER BY submitted_at DESC')
        latest=self.rows('SELECT max(observed_at) AS t FROM leaderboard')[0]['t']
        leaders=self.rows('SELECT rank,team_name,score,observed_at FROM leaderboard WHERE observed_at=? ORDER BY rank', (latest,)) if latest else []
        best=max((x['score'] for x in submissions if x['score'] is not None and 'COMPLETE' in x['status'].upper()),default=None)
        competition=json.loads(self.get_meta('competition','{}'))
        return {'schema_version':1,'generated_at':now(),'instance_id':self.instance,'counts':counts,
            'source_counts':{r['kind']:r['n'] for r in self.rows('SELECT kind,count(*) AS n FROM sources GROUP BY kind')},
            'competition_source_counts':{r['kind']:r['n'] for r in self.rows('SELECT kind,count(*) AS n FROM sources WHERE competition=? GROUP BY kind',(self.ws.config['competition'],))},
            'recent_sources':self.rows("SELECT id,kind,title,authority,observed_at,added_at FROM sources WHERE kind IN ('notebook','discussion','official','solution') ORDER BY added_at DESC LIMIT 6"),
            'competition':competition,'best_score':best,'leader_score':leaders[0]['score'] if leaders else None,
            'gap':round(leaders[0]['score']-best,6) if leaders and best is not None else None,
            'leaderboard':leaders,'submissions':submissions,
            'leaderboard_summary':json.loads(self.get_meta('leaderboard_summary:'+self.ws.config['competition'],'null') or 'null'),
            'submission_limits':json.loads(self.get_meta('submission_limits:'+self.ws.config['competition'],'null') or 'null'),
            'connectors':self.rows('SELECT * FROM connectors ORDER BY name'),
            'jobs':self.rows('SELECT * FROM jobs ORDER BY started_at DESC LIMIT 8'),
            'data_as_of':max([x['observed_at'] for x in submissions]+([latest] if latest else []),default=None),
            'warnings':['Gold58은 선택에 노출된 진단 자료입니다.','서로 다른 target·label·split의 점수를 직접 비교하지 마세요.'],
            'source_refs':['competition:Evaluation','competition:Timeline']}
