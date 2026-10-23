"""Isolated Kaggle data access. Never emit signed URLs or credential errors."""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


def safe_remote(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
        raise ValueError('Invalid remote path')
    parts = value.split('/')
    if any(p in {'', '.', '..'} or p.endswith('.') for p in parts):
        raise ValueError('Invalid remote path')
    return value


def tree(client, competition, path=''):
    from kagglesdk.competitions.types.competition_api_service import ApiListDataTreeFilesRequest
    directories, files, token, seen = [], [], '', set()
    for _ in range(60):
        r = ApiListDataTreeFilesRequest()
        r.competition_name, r.path, r.page_size, r.page_token = competition, path, 200, token
        v = client.competitions.competition_api_client.list_data_tree_files(r)
        def child(item):
            value = item.relative_url or ((path + '/') if path else '') + item.name
            safe_remote(value)
            if str(PurePosixPath(value).parent) != (path or '.'):
                raise ValueError('Unexpected directory response')
            return value
        directories.extend(child(d) for d in v.directories)
        files.extend({'path': child(f), 'bytes': f.total_bytes} for f in v.files)
        token = v.next_page_token
        if not token:
            return directories, files
        if token in seen:
            raise ValueError('Repeated page token')
        seen.add(token)
    raise ValueError('Directory listing exceeds limit')


def catalog(request):
    from kaggle import api
    competition = request['competition']
    with api.build_kaggle_client() as client:
        if request['mode'] == 'metadata':
            _, files = tree(client, competition)
            files = [f for f in files if f['path'].endswith('.csv')]
        else:
            files = []
            for study in request['studies']:
                base = safe_remote(request['split'] + '/' + study)
                dirs, direct = tree(client, competition, base)
                files.extend(direct)
                for folder in dirs:
                    deeper, items = tree(client, competition, folder)
                    if deeper:
                        raise ValueError('Unexpected study directory depth')
                    files.extend(items)
                if len(files) > 15000:
                    raise ValueError('Select fewer studies (15,000 file limit)')
        if not files or any(f['bytes'] <= 0 for f in files):
            raise ValueError('Empty or incomplete file metadata')
        return {'files': sorted(files, key=lambda f: f['path'])}


def download(request):
    from kaggle import api
    from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFileRequest, ApiDownloadDataFilesRequest
    folder = Path(request['staging'])
    folder.mkdir(parents=True, exist_ok=True)
    payload = folder / 'payload'
    with api.build_kaggle_client() as client:
        if request['mode'] == 'all':
            r = ApiDownloadDataFilesRequest()
            r.competition_name = request['competition']
            response = client.competitions.competition_api_client.download_data_files(r)
        else:
            r = ApiDownloadDataFileRequest()
            r.competition_name, r.file_name = request['competition'], safe_remote(request['file']['path'])
            response = client.competitions.competition_api_client.download_data_file(r)
        try:
            # SDK checks the remote identity before resuming a retained partial.
            api.download_file(response, str(payload), client.http_client(), quiet=True, resume=True)
        finally:
            response.close()
    if request['mode'] == 'all':
        if not zipfile.is_zipfile(payload):
            raise ValueError('Expected a competition ZIP archive')
        result = payload
    elif zipfile.is_zipfile(payload):
        result = folder / 'unpacked'
        with zipfile.ZipFile(payload) as archive:
            members = [i for i in archive.infolist() if not i.is_dir()]
            if len(members) != 1 or PurePosixPath(members[0].filename).name != PurePosixPath(request['file']['path']).name:
                raise ValueError('Unexpected individual-file archive')
            if members[0].file_size != request['file']['bytes']:
                raise ValueError('Remote size changed; create a new preview')
            with archive.open(members[0]) as source, result.open('wb') as dest:
                shutil.copyfileobj(source, dest, 1024 * 1024)
    else:
        result = payload
    if request['mode'] != 'all' and result.stat().st_size != request['file']['bytes']:
        raise ValueError('Downloaded size differs from preview; create a new preview')
    (folder / 'complete.json').write_text(json.dumps({'file': result.name, 'bytes': result.stat().st_size}), encoding='utf-8')


def main():
    request = json.loads(sys.stdin.read())
    if sys.argv[1] == 'catalog':
        print(json.dumps(catalog(request)))
    elif sys.argv[1] == 'download':
        download(request)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Exception strings from the SDK can contain signed storage URLs.
        code = getattr(getattr(exc, 'response', None), 'status_code', None)
        message = str(exc) if isinstance(exc, ValueError) else 'Kaggle 요청 실패. 인증·네트워크·대회 접근 권한을 확인하세요.'
        print(json.dumps({'error': message[:200], 'http_status': code}), file=sys.stderr)
        sys.exit(1)
