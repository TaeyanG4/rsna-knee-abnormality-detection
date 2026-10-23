"""Read-only calls to Kaggle's website API (the one the notebook list page uses).

This is NOT the supported public API: the request shape and fields can change without
notice. It is used only for facts the public API does not expose (a notebook's best
public score, medal, last run time) and every caller treats failure as "no update".
The bearer token is the person's own Kaggle token; it is never logged.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'https://www.kaggle.com/api/i/'
PAGE_SIZE = 100
MAX_PAGES = 40


def token():
    value = os.environ.get('KAGGLE_API_TOKEN', '').strip()
    if value:
        return value
    path = Path(os.environ.get('KAGGLE_CONFIG_DIR') or Path.home() / '.kaggle') / 'access_token'
    if path.is_file():
        return path.read_text(encoding='utf-8').strip()
    raise RuntimeError('Kaggle token not found (~/.kaggle/access_token or KAGGLE_API_TOKEN)')


def call(service, body, timeout=60):
    request = urllib.request.Request(BASE + service, data=json.dumps(body).encode('utf-8'), method='POST',
                                     headers={'Authorization': 'Bearer ' + token(), 'Content-Type': 'application/json', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', 'replace')[:300]
        raise RuntimeError(f'Kaggle web API {service} -> HTTP {exc.code}: {detail}') from None


def competition_notebooks(competition_id, sort_by='SCORE_DESCENDING', sleep=0.3, fetch=None):
    """All notebooks attached to a competition with the website's score column.

    Returns {ref: {'public_score', 'medal', 'runtime_s', 'votes', 'last_run', 'title'}}.
    """
    out = {}
    fetch = fetch or call  # resolved at call time so tests can patch `call`
    for page in range(1, MAX_PAGES + 1):
        data = fetch('kernels.KernelsService/ListKernels',
                     {'kernelFilterCriteria': {'listRequest': {'competitionId': int(competition_id), 'sortBy': sort_by, 'pageSize': PAGE_SIZE, 'page': page}}})
        kernels = data.get('kernels') or []
        for k in kernels:
            author = (k.get('author') or {}).get('userName')
            slug = k.get('currentUrlSlug')
            if not author or not slug:
                continue
            out[f'{author}/{slug}'] = {'public_score': k.get('bestPublicScore'), 'medal': k.get('medal'),
                                       'runtime_s': k.get('lastRunExecutionTimeSeconds'), 'votes': k.get('totalVotes'),
                                       'last_run': k.get('lastRunTime'), 'title': k.get('title')}
        if len(kernels) < PAGE_SIZE:
            break
        time.sleep(sleep)
    return out
