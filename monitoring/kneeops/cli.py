from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .store import Store, dumps


def main():
    parser = argparse.ArgumentParser(description='KneeOps local, structured JSON interface for people and agents')
    parser.add_argument('--root', help='Workspace containing config/workspace.toml')
    parser.add_argument('--state', help='Independent local state directory')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['import', 'briefing', 'backup', 'schedule']: sub.add_parser(command).add_argument('--json', action='store_true')
    sync = sub.add_parser('sync', help='Kaggle collection; budgets cap per-run body/source downloads')
    sync.add_argument('--json', action='store_true')
    sync.add_argument('--topic-budget', type=int, default=None); sync.add_argument('--notebook-budget', type=int, default=None)
    sync.add_argument('--version-budget', type=int, default=None)
    serve = sub.add_parser('serve'); serve.add_argument('--port', type=int, default=8765)
    serve.add_argument('--host', choices=['127.0.0.1', '0.0.0.0'], default='127.0.0.1', help='0.0.0.0 only inside a localhost-published Docker container')
    open_app = sub.add_parser('open'); open_app.add_argument('--port',type=int,default=8765)
    sources = sub.add_parser('sources'); sources.add_argument('action', choices=['search', 'show'])
    sources.add_argument('value', nargs='?', default=''); sources.add_argument('--kind', default=''); sources.add_argument('--json', action='store_true')
    sources.add_argument('--competition', default='');sources.add_argument('--offset',type=int,default=0);sources.add_argument('--limit',type=int,default=50)
    runs = sub.add_parser('experiments'); runs.add_argument('action', choices=['list', 'compare']); runs.add_argument('ids', nargs='*'); runs.add_argument('--json', action='store_true')
    fetch = sub.add_parser('fetch'); fetch.add_argument('kind', choices=['notebook', 'topic']); fetch.add_argument('ref')
    history = sub.add_parser('history'); history.add_argument('competition'); history.add_argument('--json', action='store_true')
    history.add_argument('--solution-budget', type=int, default=None); history.add_argument('--discussion-budget', type=int, default=None)
    history.add_argument('--notebook-bodies', type=int, default=None)
    data = sub.add_parser('data'); data.add_argument('action', choices=['status', 'cache-descriptor', 'index', 'library', 'prepare', 'structure', 'descriptor']); data.add_argument('--manifest', default=''); data.add_argument('--json', action='store_true')
    data.add_argument('--raw', default=''); data.add_argument('--series', default=''); data.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command in {'serve','open'}:
            import uvicorn
            from .api import create_app
            if args.command=='open':
                import threading,time,urllib.request,webbrowser
                def open_browser():
                    for _ in range(40):
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/health',timeout=1) as response:
                                if response.status==200:break
                        except OSError:time.sleep(.25)
                    else:return
                    webbrowser.open(f'http://127.0.0.1:{args.port}')
                # Reuse an existing copy of this app rather than starting a second worker.
                try:
                    import json
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/health',timeout=1) as response:
                        info=json.load(response)
                    if info.get('app')=='kneeops':
                        webbrowser.open(f'http://127.0.0.1:{args.port}');return
                except OSError:pass
                threading.Thread(target=open_browser,daemon=True).start()
            from .serving import config
            uvicorn.Server(config(create_app(args.root, args.state), host=getattr(args,'host','127.0.0.1'), port=args.port, access_log=False)).run()
            return
        store = Store(args.root, args.state)
        from .importer import import_local, compare_runs
        from .collectors import sync_kaggle, collect_notebook, collect_topic
        if args.command == 'briefing': result = store.briefing()
        elif args.command == 'data':
            from .data_manager import overview
            from .data_cache import cache_descriptor
            from .data_library import build_library, library
            if args.action == 'index': result = build_library(store)
            elif args.action == 'library': result = library(store)
            elif args.action == 'prepare':
                from .data_library import study_path
                path, item = study_path(store, args.raw)
                result = {'path': str(path), 'study': item}
            elif args.action == 'structure':
                from .data_archive import structure
                result = structure(store, args.raw, args.series, max(0, args.offset))
            elif args.action == 'descriptor':
                from .data_library import input_descriptor
                result = input_descriptor(store)
            else: result = overview(store) if args.action == 'status' else cache_descriptor(store, args.manifest or store.get_meta('data_cache_manifest', ''))
        elif args.command == 'import': result = import_local(store)
        elif args.command == 'sync':
            budgets = {k: max(0, v) for k, v in {'topic_budget': args.topic_budget, 'notebook_budget': args.notebook_budget,
                       'version_budget': args.version_budget}.items() if v is not None}
            result = sync_kaggle(store, lambda message: print(message, file=sys.stderr, flush=True), **budgets)
        elif args.command == 'schedule':
            from .service import schedule_status
            from .collectors import collection_coverage
            result = {**schedule_status(store), 'coverage': collection_coverage(store)}
        elif args.command == 'sources': result = store.sources(args.value, args.kind, max(1,min(args.limit,200)),max(0,args.offset),args.competition) if args.action == 'search' else store.source(args.value)
        elif args.command == 'experiments': result = store.runs() if args.action == 'list' else compare_runs(store, args.ids)
        elif args.command == 'fetch': result = {'revision': collect_notebook(store, args.ref) if args.kind == 'notebook' else collect_topic(store, int(args.ref))}
        elif args.command == 'history':
            from .history import collect_history
            budgets = {k: max(0, v) for k, v in {'solution_budget': args.solution_budget, 'discussion_budget': args.discussion_budget,
                       'notebook_bodies': args.notebook_bodies}.items() if v is not None}
            result = collect_history(store, args.competition, lambda message: print(message, file=sys.stderr, flush=True), **budgets)
        elif args.command == 'backup':
            from .service import backup
            result = {'path': str(backup(store))}
        print(dumps(result))
        if isinstance(result, dict) and result.get('errors'): sys.exit(2)
    except (Exception,) as exc:
        print(dumps({'error': str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__': main()
