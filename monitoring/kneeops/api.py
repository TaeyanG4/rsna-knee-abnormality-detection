from __future__ import annotations

import difflib
import json
import os
import secrets
import sqlite3
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .importer import compare_runs
from .service import Service, backup, schedule_status
from .store import Store, TARGETS, dumps, now
from .research import DEFAULT_COMPETITIONS, strategy_excerpts
from .data_manager import DataManager, destination, overview as data_overview, settings as data_settings, studies as data_studies
from .data_cache import cache_descriptor
from .translation import translation_for
from .viewer_api import viewer_router


class Preferences(BaseModel):
    theme: Literal['light', 'dark'] = 'light'
    compact: bool = False
    watch_targets: list[str] = Field(default_factory=list, max_length=12)
    hidden_cards: list[Literal['leaderboard', 'focus']] = Field(default_factory=list, max_length=2)
    sync_enabled: bool = False
    sync_minutes: int = Field(default=60, ge=15, le=1440)
    sync_history: list[str] = Field(default_factory=list, max_length=5)
    research_view: Literal['list','cards'] = 'list'


class PredictionCompare(BaseModel):
    files: list[str] = Field(min_length=1, max_length=4)
    label: str = Field(default='', max_length=500)


class Schedule(BaseModel):
    enabled: bool
    minutes: int = Field(default=60, ge=15, le=1440)


class Job(BaseModel):
    kind: Literal['import', 'sync', 'notebook', 'topic', 'history', 'data_index']
    ref: str = Field(default='', max_length=200)


class Competition(BaseModel):
    slug: str = Field(pattern=r'^[a-z0-9][a-z0-9-]{2,120}$')
    title: str = Field(min_length=1,max_length=150)
    relevance: str = Field(default='',max_length=1500)


class DataSettings(BaseModel):
    destination: str = Field(min_length=1, max_length=1000)


class DataPlan(BaseModel):
    mode: Literal['metadata', 'sample', 'selected', 'all']
    studies: list[str] = Field(default_factory=list, max_length=5)


class CachePath(BaseModel):
    manifest: str = Field(min_length=1, max_length=1000)


def create_app(root=None, state=None):
    store = Store(root, state)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = Service(store)
        app.state.downloads = DataManager(store)
        yield
        app.state.downloads.close()
        app.state.service.close()

    app = FastAPI(title='KneeOps Local', version='0.1.0', lifespan=lifespan)
    app.state.store = store
    app.include_router(viewer_router(store))

    @app.middleware('http')
    async def local_access(request: Request, call_next):
        host = request.url.hostname
        if host not in {'localhost', '127.0.0.1', '::1', 'testserver'}:
            return JSONResponse({'detail': 'Local access only'}, status_code=403)
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Cross-origin access denied'}, status_code=403)
        if request.method not in {'GET', 'HEAD', 'OPTIONS'} and not secrets.compare_digest(request.headers.get('x-kneeops-token', ''), token):
            return JSONResponse({'detail': 'Refresh the local page before writing'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        if request.url.path.startswith('/api/') or request.url.path == '/':
            response.headers['Cache-Control'] = 'no-store'
        elif request.url.path.startswith('/assets/') and response.status_code == 200:
            response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
        return response

    @app.exception_handler(ValueError)
    async def bad_value(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse({'detail': 'Record not found'}, status_code=404)

    @app.get('/api/health')
    def health(): return {'app':'kneeops','status': 'ok', 'version': '0.1.0', 'schema': 3}

    @app.get('/api/session')
    def session(): return {'token': token, 'instance_id': store.instance, 'targets': TARGETS}

    @app.get('/api/briefing')
    def briefing(): return store.briefing()

    @app.get('/api/sources')
    def sources(q: str = Query('', max_length=200), kind: str = '', limit: int = Query(80, ge=1, le=200), offset: int = Query(0, ge=0), competition: str = '',
                sort: Literal['added', 'activity', 'votes', 'comments', 'claim', 'score', 'title'] = 'added', body: Literal['', 'full', 'listing'] = '',
                input: str = Query('', max_length=300)):
        return store.sources(q, kind, limit, offset, competition, sort, body, input)

    @app.get('/api/notebook-inputs')
    def notebook_inputs(competition: str = Query('', max_length=130), limit: int = Query(40, ge=1, le=200)):
        # Which attached datasets/models/notebooks recur across collected notebooks.
        rows = store.rows('''SELECT g.key AS kind,i.value AS ref,count(*) AS n,max(COALESCE(m.public_score,0)) AS best_score,max(COALESCE(m.score_claim,0)) AS best_claim,
            max(COALESCE(m.votes,0)) AS top_votes FROM source_meta m JOIN sources s ON s.id=m.source_id,
            json_each(m.inputs) g, json_each(g.value) i WHERE s.kind IN ('notebook','own_notebook') AND g.key<>'competitions'
            AND (?='' OR s.competition=?) GROUP BY g.key,i.value ORDER BY n DESC,best_score DESC,best_claim DESC LIMIT ?''', (competition, competition, limit))
        total = store.rows('''SELECT count(*) AS n FROM source_meta m JOIN sources s ON s.id=m.source_id
            WHERE s.kind IN ('notebook','own_notebook') AND m.inputs IS NOT NULL AND (?='' OR s.competition=?)''', (competition, competition))[0]['n']
        return {'items': rows, 'notebooks_with_inputs': total}

    @app.get('/api/source/compare')
    def compare_sources(a: str, b: str):
        left, right = store.source(a), store.source(b)
        def inputs(src):
            meta = src.get('meta') or {}
            return {k: set(v) for k, v in (meta.get('inputs') or {}).items()}
        ia, ib = inputs(left), inputs(right)
        kinds = sorted(set(ia) | set(ib))
        lines = list(difflib.unified_diff(left['revision']['content'].splitlines(), right['revision']['content'].splitlines(),
                                          fromfile=left['title'][:60], tofile=right['title'][:60], lineterm='', n=2))
        added = sum(1 for x in lines if x.startswith('+') and not x.startswith('+++'))
        removed = sum(1 for x in lines if x.startswith('-') and not x.startswith('---'))
        return {'a': {k: left[k] for k in ('id', 'title', 'kind', 'url')} | {'version': left['revision']['version'], 'meta': left.get('meta')},
                'b': {k: right[k] for k in ('id', 'title', 'kind', 'url')} | {'version': right['revision']['version'], 'meta': right.get('meta')},
                'inputs': {k: {'only_a': sorted(ia.get(k, set()) - ib.get(k, set())), 'only_b': sorted(ib.get(k, set()) - ia.get(k, set())),
                               'both': sorted(ia.get(k, set()) & ib.get(k, set()))} for k in kinds},
                'diff': '\n'.join(lines[:6000]), 'truncated': len(lines) > 6000, 'added': added, 'removed': removed,
                'similarity': round(difflib.SequenceMatcher(None, left['revision']['content'][:200000], right['revision']['content'][:200000], autojunk=True).quick_ratio(), 4)}

    @app.get('/api/predictions/files')
    def prediction_files(refresh: bool = False):
        from .predictions import discover
        return discover(store, refresh)

    @app.post('/api/predictions/compare')
    def prediction_compare(body: PredictionCompare):
        from .predictions import compare
        return compare(store, body.files, body.label)

    @app.get('/api/source')
    def source(id: str, revision: str | None = None):
        result=store.source(id, revision)
        result['strategy_evidence']=strategy_excerpts(result['revision']['content'])
        result['translation']=translation_for(result)
        return result

    @app.get('/api/competitions')
    def competitions():
        items=json.loads(store.get_meta('related_competitions',dumps(DEFAULT_COMPETITIONS)))
        return [{**item,'snapshot':json.loads(store.get_meta('history:'+item['slug'],'null')),
                 'count':store.rows('SELECT count(*) n FROM sources WHERE competition=?',(item['slug'],))[0]['n']} for item in items]

    @app.put('/api/competitions')
    def put_competitions(body: list[Competition]):
        if len(body)>20 or len({r.slug for r in body})!=len(body):raise ValueError('Use up to 20 unique competitions')
        store.set_meta('related_competitions',dumps([r.model_dump() for r in body]))
        return competitions()

    @app.get('/api/source/diff')
    def diff(id: str, before: str, after: str):
        a, b = store.source(id, before)['revision'], store.source(id, after)['revision']
        lines = list(difflib.unified_diff(a['content'].splitlines(), b['content'].splitlines(), fromfile=before[:12], tofile=after[:12], lineterm=''))
        return {'diff': '\n'.join(lines[:5000]), 'truncated': len(lines) > 5000}

    @app.get('/api/runs')
    def runs(): return store.runs()

    @app.get('/api/runs/compare')
    def compare(a: str, b: str): return compare_runs(store, [a, b])

    @app.get('/api/submissions')
    def submissions(): return store.briefing()['submissions']

    @app.get('/api/jobs')
    def jobs(): return store.rows('SELECT * FROM jobs ORDER BY started_at DESC LIMIT 20')

    @app.post('/api/jobs', status_code=202)
    def job(body: Job):
        if body.kind == 'topic' and not body.ref.isdecimal(): raise ValueError('Topic ID must be numeric')
        return app.state.service.enqueue(body.kind, body.ref)

    @app.get('/api/preferences')
    def preferences(): return json.loads(store.get_meta('preferences'))

    @app.put('/api/preferences')
    def put_preferences(body: Preferences):
        if not set(body.watch_targets) <= set(TARGETS): raise ValueError('Unknown target')
        import re
        if not all(re.fullmatch(r'[a-z0-9][a-z0-9-]{2,120}',x) for x in body.sync_history):raise ValueError('Invalid competition slug')
        store.set_meta('preferences', dumps(body.model_dump()))
        return body

    def schedule_payload():
        from .collectors import collection_coverage
        return {**schedule_status(store), 'coverage': collection_coverage(store)}

    @app.get('/api/schedule')
    def get_schedule(): return schedule_payload()

    @app.put('/api/schedule')
    def put_schedule(body: Schedule):
        # Merge only the schedule fields so unsaved edits on other screens are not written.
        prefs = json.loads(store.get_meta('preferences', '{}'))
        prefs.update(sync_enabled=body.enabled, sync_minutes=body.minutes)
        store.set_meta('preferences', dumps(prefs))
        return schedule_payload()

    @app.post('/api/backup')
    def make_backup():
        path = backup(store)
        return FileResponse(path, filename=path.name, media_type='application/vnd.sqlite3')

    @app.get('/api/data')
    def data(): return data_overview(store)

    @app.get('/api/data/library')
    def data_library():
        from .data_library import library
        return library(store)

    @app.get('/api/data/structure')
    def data_structure(raw: str = Query(max_length=24), series: str = Query('', max_length=100),
                       offset: int = Query(0, ge=0)):
        from .data_archive import structure
        return structure(store, raw, series, offset)

    @app.get('/api/data/library/descriptor')
    def data_input_descriptor():
        from .data_library import input_descriptor
        return JSONResponse(input_descriptor(store), headers={'Content-Disposition': 'attachment; filename="kneeops-local-data-inputs.json"'})

    @app.put('/api/data/settings')
    def put_data_settings(body: DataSettings):
        path = destination(store, body.destination)
        store.set_meta('data_destination', str(path))
        return data_settings(store)

    @app.get('/api/data/studies')
    def studies(q: str = Query('', max_length=100), split: Literal['train', 'test'] = 'train'):
        return data_studies(store, split, q)

    @app.get('/api/data/downloads')
    def downloads(): return app.state.downloads.list()

    @app.post('/api/data/preview')
    def data_preview(body: DataPlan): return app.state.downloads.preview(body.mode, body.studies)

    @app.post('/api/data/downloads/{id}/start', status_code=202)
    def start_download(id: str): return app.state.downloads.start(id)

    @app.post('/api/data/downloads/{id}/cancel')
    def cancel_download(id: str): return app.state.downloads.cancel(id)

    @app.get('/api/data/downloads/{id}/manifest')
    def download_manifest(id: str):
        return JSONResponse(app.state.downloads.get(id), headers={'Content-Disposition': f'attachment; filename="download-{uuid.UUID(id)}.json"'})

    @app.put('/api/data/cache')
    def put_cache(body: CachePath):
        descriptor = cache_descriptor(store, body.manifest)
        store.set_meta('data_cache_manifest', str(store.ws.resolve(body.manifest)))
        return descriptor

    @app.get('/api/data/cache/descriptor')
    def cache_export():
        path = store.get_meta('data_cache_manifest')
        if not path: raise ValueError('먼저 캐시를 연결하세요.')
        return JSONResponse(cache_descriptor(store, path), headers={'Content-Disposition': 'attachment; filename="cache-input.json"'})

    static = Path(os.environ.get('KNEEOPS_STATIC_DIR', Path(__file__).parent / 'static'))
    if (static / 'assets').is_dir(): app.mount('/assets', StaticFiles(directory=static / 'assets'), name='assets')

    @app.get('/')
    def index():
        if not (static / 'index.html').is_file(): raise HTTPException(503, 'Build monitoring/web first: npm ci && npm run build')
        return FileResponse(static / 'index.html')

    return app
